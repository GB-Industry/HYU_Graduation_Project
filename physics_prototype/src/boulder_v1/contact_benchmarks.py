"""Native-contact certificates, not a production controller or pose initializer.

The fixed allocation below was solved offline once. Only its recomputed hinge
torques enter the mixed rollout; planned contact forces are never injected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Callable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_geometry import canonical_geometry
from .grasp import GraspManager
from .mjcf_builder import build_mjcf, get_grasp_equality_name
from .schema import Affordance, BoulderScene, ClimberProfile, ContactRegion, Limb, SourceType, WallSurface
from .support import FootStatus, FootSupportSensor

SHOE_SIZE = np.array([.040, .095, .018])
SHOE_CENTER = np.array([0., .035, -.018])
MASS, GRAVITY = 1.2, 9.81
SIDES = ("left", "right")
HANDS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)
FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
# /tmp/opencode/stage2_forefoot_staticff_dt0.002_kp80_kd1/fixed_allocation.json
CORNER_FORCES = np.array([
    [0., 0., 102.05489780995418], [0., -.9, 1.],
    [0., -47.10201060759078, 270.4869572789135], [0., -.9, 1.],
    [0., 0., 175.03015432015414], [0., 0., 1.],
    [0., 2.073011524066368, 136.34167421072473], [0., 0., 1.],
])
HAND_FORCES = np.array([[0., 34.60058372853675, 33.89171720974012],
                        [0., 12.228415354987657, 46.317599170513304]])


@dataclass(frozen=True)
class FootCase:
    name: str
    mu: float = .8
    angle_deg: float = 0.
    surface: str = "box"
    press: float = 0.
    tangent: float = 0.
    pull: float = 0.
    expected: str = "stable"
    reverse_order: bool = False


FOOT_CASES = (
    FootCase("support_low", mu=.3), FootCase("support_high"),
    FootCase("load_low", mu=.3, press=6., tangent=8., expected="slip"),
    FootCase("load_high", press=6., tangent=8.),
    FootCase("shoe_friction_cap", mu=3., press=6., tangent=8.),
    FootCase("incline15_low", mu=.3, angle_deg=15.),
    FootCase("incline35_low", mu=.3, angle_deg=35., expected="slip"),
    FootCase("incline35_high", mu=1.2, angle_deg=35.),
    FootCase("pull_away", pull=2 * MASS * GRAVITY, expected="separate"),
    FootCase("plane_support", surface="plane"),
    FootCase("swapped_order", press=6., tangent=8., reverse_order=True),
)


@dataclass(frozen=True)
class ContactFixture:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene: BoulderScene
    profile: ClimberProfile
    manager: GraspManager | None = None
    reference: np.ndarray | None = None
    xml: str = ""


def _vec(values) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def _scene(regions=(), walls=()) -> BoulderScene:
    return BoulderScene("MuJoCo x horizontal, y forward, z up", 1., tuple(walls), tuple(regions))


def _fresh(model, data):
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)
    return scratch


def _finite(data) -> bool:
    return all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc,
                                              data.qfrc_constraint, data.efc_force))


def _feet(model, data, scene, *, fresh=True):
    # Deliberately use the same sampler as the production contact snapshot.
    return FootSupportSensor().measure(model, data, scene, fresh=fresh)


def _capture_before(fixture, limb, region):
    model, data = fixture.model, _fresh(fixture.model, fixture.data)
    source, target = data.site(f"{limb.value.lower()}_site"), data.site(f"site_{region.id}")
    jp, jr, tp, tr = (np.zeros((3, model.nv)) for _ in range(4))
    mujoco.mj_jacSite(model, data, jp, jr, source.id)
    mujoco.mj_jacSite(model, data, tp, tr, target.id)
    velocity = (jp - tp) @ data.qvel
    momentum = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, momentum, data.qvel)
    signed_distance = float(mujoco.mj_geomDistance(model, data, model.geom(f"{limb.value.lower()}_geom").id,
                                                model.geom(f"geom_{region.id}").id, 1., None))
    return {"phase": "pre_activation_fresh_forward", "time_s": float(data.time),
            "limb": limb.value, "region_id": region.id,
            "gap_m": float(np.linalg.norm(source.xpos - target.xpos)),
            "relative_velocity_world_m_s": velocity.tolist(), "relative_speed_m_s": float(np.linalg.norm(velocity)),
            "orientation": float(source.xmat.reshape(3, 3)[:, 2] @ target.xmat.reshape(3, 3)[:, 2]),
            "signed_geom_distance_m": signed_distance, "penetration_m": max(0., -signed_distance),
            "kinetic_energy_J": float(data.qvel @ momentum / 2)}


def _model_proof(fixture):
    model = fixture.model
    equalities = [model.equality(i).name for i in range(model.neq)]
    return {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq,
            "equality_names": equalities, "foot_equalities": sum("foot" in name for name in equalities),
            "contact_mode_numeric": float(model.numeric("contact_mode").data[0]),
            "dynamic_mass_kg": float(model.body_mass[model.body_weldid != 0].sum()),
            "pair_friction": model.pair_friction.tolist(), "gravity_m_s2": model.opt.gravity.tolist(),
            "profile": fixture.profile.to_dict(),
            "force_roles": {"grip": "active hand connect equality", "foot": "native unilateral shoe collision",
                            "detached_palm": "environment collision, not a grip"}}


def make_foot_fixture(case: FootCase, dt: float = .002) -> ContactFixture:
    """A fully free production-sized shoe, with no equality or motor."""
    if dt not in (.002, .001):
        raise ValueError("benchmark timestep must be 0.002 or 0.001 seconds")
    angle = math.radians(case.angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    normal = rotation[:, 2]
    quat = [math.cos(angle / 2), math.sin(angle / 2), 0., 0.]
    origin = normal * (SHOE_SIZE[2] + .001) - rotation @ SHOE_CENTER
    center = -.05 * normal
    region = ContactRegion("SUPPORT", SourceType.HOLD, tuple(center),
                           tuple(rotation @ [0., -1., 0.]), case.mu,
                           frozenset({Affordance.STEP}), half_size=(5., 5., .05))
    geometry = canonical_geometry(region)
    np.testing.assert_allclose(geometry.body_frame.rotation, rotation, atol=1e-14, rtol=0.)
    shoe = f'''<body name="left_foot" pos="{_vec(origin)}" quat="{_vec(quat)}">
      <freejoint name="shoe_free"/>
      <geom name="left_foot_geom" type="box" size="{_vec(SHOE_SIZE)}"
            pos="{_vec(SHOE_CENTER)}" mass="1.2" friction="1.8 0.05 0.005"/>
      <site name="left_foot_site" pos="0 0.09 -0.025"/>
      </body>'''
    surface = f'''<body name="contact_SUPPORT" pos="{_vec(geometry.body_frame.position if case.surface == 'box' else np.zeros(3))}"
          quat="{_vec(geometry.body_frame.quaternion)}"><geom name="geom_SUPPORT" type="{case.surface}"
          size="5 5 0.05" friction="{case.mu} 0.05 0.005"/></body>'''
    first, second = ("left_foot_geom", "geom_SUPPORT") if case.reverse_order else (
        "geom_SUPPORT", "left_foot_geom")
    mu = min(1.8, case.mu)
    xml = f'''<mujoco model="stage2_free_shoe">
      <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"
              solver="Newton" iterations="100" tolerance="1e-8" cone="pyramidal"/>
      <default><geom solref="0.01 1" condim="3"/></default>
      <custom><numeric name="contact_mode" data="0"/></custom>
      <worldbody>{shoe + surface if case.reverse_order else surface + shoe}</worldbody>
      <contact><pair geom1="{first}" geom2="{second}" condim="3"
        friction="{mu} {mu} 0 0 0" solref="0.01 1"/></contact></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ContactFixture(model, data, _scene((region,)), ClimberProfile("free_shoe"), xml=xml)


def shoe_measurement(fixture: ContactFixture, *, fresh=True) -> dict:
    """Cross-check the production sampler against native contact-frame rows."""
    model, live = fixture.model, fixture.data
    data = _fresh(model, live) if fresh else live
    geom = model.geom("left_foot_geom").id
    body = model.body("left_foot").id
    surface = model.geom("geom_SUPPORT").id
    normal = data.geom_xmat[surface].reshape(3, 3)[:, 2]
    state = _feet(model, data, fixture.scene, fresh=False)[Limb.LEFT_FOOT]
    force, generalized = np.zeros(3), np.zeros(model.nv)
    records = []
    for index, contact in enumerate(data.contact):
        if geom not in (contact.geom1, contact.geom2):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        sign = 1 if contact.geom2 == geom else -1
        world = sign * contact.frame.reshape(3, 3).T @ wrench[:3]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, contact.pos, body)
        generalized += jp.T @ world
        force += world
        fn, ft = float(wrench[0]), float(np.linalg.norm(wrench[1:3]))
        records.append({"geoms": [model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                        "native_force_N": wrench[:3].tolist(), "force_world_on_shoe_N": world.tolist(),
                        "normal_world_on_shoe": (sign * contact.frame[:3]).tolist(),
                        "mu": float(contact.friction[0]), "Fn_N": fn, "Ft_N": ft,
                        "pyramid_excess_N": float(np.abs(wrench[1:3]).sum() - contact.friction[0] * fn)})
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jp, jr, body)
    velocity = jp @ data.qvel
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * SHOE_SIZE
    world_corners = data.geom_xpos[geom] + corners @ data.geom_xmat[geom].reshape(3, 3).T
    top = data.geom_xpos[surface] + (model.geom_size[surface, 2] * normal
                                     if model.geom_type[surface] == mujoco.mjtGeom.mjGEOM_BOX else 0.)
    first_dof = model.joint("shoe_free").dofadr[0]
    shoe_dofs = slice(first_dof, first_dof + 6)
    return {"time_s": float(data.time), "force_state_time_s": float(data.time),
            "force_epoch": "fresh_forward_after_native_step",
            "frame": "world; contact frame rows; force acts on geom2",
            "com_m": data.xipos[body].tolist(), "speed_m_s": float(np.linalg.norm(velocity)),
            "Fn_N": float(force @ normal), "Ft_N": float(np.linalg.norm(force - (force @ normal) * normal)),
            "production_Fn_N": float(state.normal_force),
            "production_Ft_N": float(state.tangential_force),
            "slip_m_s": float(state.tangential_speed), "status": state.status.value,
            "supporting": state.supporting, "contacting": state.contacting,
            "utilization": float(state.friction_utilization),
            "minimum_gap_m": float(np.min((world_corners - top) @ normal)),
            "mapping_error": float(np.max(np.abs(generalized[shoe_dofs] - data.qfrc_constraint[shoe_dofs]))),
            "contacts": records, "finite": _finite(data)}


def run_foot_case(case: FootCase, dt: float = .002) -> dict:
    fixture = make_foot_fixture(case, dt)
    model, data = fixture.model, fixture.data
    rotation = data.geom("geom_SUPPORT").xmat.reshape(3, 3).copy()
    samples = []
    for step in range(round(1.4 / dt)):
        data.xfrc_applied[:] = 0.
        if step >= round(.4 / dt):
            data.xfrc_applied[model.body("left_foot").id, :3] = (
                rotation[:, 1] * case.tangent + rotation[:, 2] * (case.pull - case.press))
        mujoco.mj_step(model, data)
        if step >= round(.4 / dt):
            samples.append(shoe_measurement(fixture))
    tail = [row for row in samples if row["time_s"] >= .9 - dt / 4]
    expected_fn = MASS * GRAVITY * math.cos(math.radians(case.angle_deg)) + case.press
    mean_fn = float(np.mean([r["Fn_N"] for r in tail]))
    displacement = float(np.linalg.norm((rotation.T @ (np.array(samples[-1]["com_m"])
                                                      - samples[0]["com_m"]))[:2]))
    tail_displacement = float(np.linalg.norm((rotation.T @ (np.array(tail[-1]["com_m"])
                                                           - tail[0]["com_m"]))[:2]))
    contacts = [c for row in samples for c in row["contacts"]]
    checks = {"finite": all(r["finite"] for r in samples),
              "no_warnings": not any(w.number for w in data.warning),
              "free_unpinned_shoe": (model.nq, model.nv, model.neq, model.nu) == (7, 6, 0, 0),
              "unilateral": all(c["Fn_N"] >= -1e-7 for c in contacts),
              "actual_mu": all(abs(c["mu"] - min(1.8, case.mu)) < 1e-12 for c in contacts),
              "pyramid": all(c["pyramid_excess_N"] <= 1e-6 + 1e-5 * expected_fn for c in contacts),
              "native_frame_and_sign": max(r["mapping_error"] for r in samples) < 1e-7,
              "production_normal": max(abs(r["Fn_N"] - r["production_Fn_N"]) for r in samples) < 1e-7}
    if case.expected == "stable":
        checks.update(normal_load=abs(mean_fn - expected_fn) < .02 * expected_fn,
                      sustained_contact=all(r["contacting"] for r in tail),
                      stable_speed=max(r["speed_m_s"] for r in tail) < .01,
                      stable_contact_points=max(r["slip_m_s"] for r in tail) <= .01,
                      stable_displacement=displacement < .01,
                      production_support=all(r["supporting"] for r in tail))
    elif case.expected == "slip":
        checks.update(sliding=displacement > .02, sustained_slide=tail_displacement > .02,
                      production_slip=any(r["status"] == FootStatus.SLIPPING.value for r in tail))
    else:
        checks.update(separated=samples[-1]["minimum_gap_m"] > .02,
                      no_tensile_force=min(r["Fn_N"] for r in samples) >= -1e-7,
                      no_tail_force=max(abs(r["Fn_N"]) for r in tail) < 1e-7,
                      no_phantom_support=all(not r["contacting"] and not r["supporting"] for r in tail))
    return {"kind": "foot", "case": case.name, "dt_s": dt, "expected": case.expected,
            "parameters": asdict(case),
            "model_proof": _model_proof(fixture),
            "checks": checks, "contact_acceptance": all(checks.values()),
            "tail_normal_mean_N": mean_fn, "expected_normal_N": 0. if case.pull else expected_fn,
            "tail_speed_max_m_s": max(r["speed_m_s"] for r in tail),
            "tail_slip_max_m_s": max(r["slip_m_s"] for r in tail),
            "test_displacement_m": displacement, "tail_displacement_m": tail_displacement,
            "max_mapping_error": max(r["mapping_error"] for r in samples),
            "representative_contacts": next((r["contacts"] for r in tail if r["contacts"]), []),
            "initial": samples[0], "final": samples[-1], "samples": samples}


def make_hand_fixture(dt: float = .002, capacity: float = 150., *, gap: float = .0005,
                      speed: float = .025, angle_deg: float = 0.) -> ContactFixture:
    """Independent free palm and free shoe; capacity is the only profile change."""
    foot = make_foot_fixture(FootCase("hand_support"), dt)
    profile = ClimberProfile("stage2_hand", grip_capacity=capacity)
    source_xml = build_mjcf(_scene(), profile)
    source = mujoco.MjModel.from_xml_string(source_xml)
    tree = ET.fromstring(foot.xml)
    world = tree.find("worldbody")
    angle = math.pi / 2 + math.radians(angle_deg)
    quat = np.array([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    target, normal = np.array([0., 0., 1.]), np.array([0., -1., 0.])
    origin = target + gap * normal - rotation.reshape(3, 3) @ source.site("left_hand_site").pos
    hand = ET.SubElement(world, "body", name="left_hand", pos=_vec(origin), quat=_vec(quat), gravcomp="1")
    ET.SubElement(hand, "freejoint", name="hand_free")
    body = source.body("left_hand").id
    ET.SubElement(hand, "inertial", mass=str(source.body_mass[body]), pos=_vec(source.body_ipos[body]),
                  quat=_vec(source.body_iquat[body]), diaginertia=_vec(source.body_inertia[body]))
    original_hand = ET.fromstring(source_xml).find(".//body[@name='left_hand']")
    for node in original_hand:
        if node.tag in ("geom", "site"):
            hand.append(ET.fromstring(ET.tostring(node, encoding="unicode")))
    region = ContactRegion("left_hand", SourceType.HOLD, tuple(target - .026 * normal),
                           tuple(normal), .9, frozenset({Affordance.GRASP}), half_size=(.05, .02, .05))
    geometry = canonical_geometry(region)
    hold = ET.SubElement(world, "body", name="contact_left_hand", pos=_vec(geometry.body_frame.position),
                        quat=_vec(geometry.body_frame.quaternion))
    ET.SubElement(hold, "geom", name="geom_left_hand", type="box", size="0.05 0.02 0.05", friction="0.9 0.05 0.005")
    body_rotation = np.array(geometry.body_frame.rotation)
    target_position = body_rotation.T @ (np.array(geometry.hand_frame.position) - geometry.body_frame.position)
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, (body_rotation.T @ geometry.hand_frame.rotation).reshape(-1))
    ET.SubElement(hold, "site", name="site_left_hand", pos=_vec(target_position), quat=_vec(target_quat))
    equalities = ET.SubElement(tree, "equality")
    ET.SubElement(equalities, "connect", name=get_grasp_equality_name(Limb.LEFT_HAND, region.id),
                  site1="left_hand_site", site2="site_left_hand", active="false",
                  solref="0.004 1", solimp="0.99 0.99 0.001")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[model.joint("hand_free").dofadr[0]:model.joint("hand_free").dofadr[0] + 3] = -speed * normal
    mujoco.mj_forward(model, data)
    scene = _scene((*foot.scene.contact_regions, region))
    return ContactFixture(model, data, scene, profile, GraspManager(model, data, scene, profile=profile), xml=xml)


def run_hand_case(dt: float = .002, capacity: float = 150.) -> dict:
    fixture = make_hand_fixture(dt, capacity)
    model, data, manager = fixture.model, fixture.data, fixture.manager
    seed, seed_velocity = data.qpos.copy(), data.qvel.copy()
    pre_capture = _capture_before(fixture, Limb.LEFT_HAND, fixture.scene.region("left_hand"))
    accepted = manager.attach(Limb.LEFT_HAND, fixture.scene.region("left_hand"))
    manager.synchronize_from_live()
    seed_unchanged = bool(np.array_equal(seed, data.qpos) and np.array_equal(seed_velocity, data.qvel))
    samples, transitions = [], []
    safe, force_mapping, max_reaction_window, max_penetration = True, True, 0., 0.
    for step in range(round(.6 / dt)):
        data.xfrc_applied[model.body("left_hand").id, :3] = [0., -100. if step >= round(.4 / dt) else 0., 0.]
        for phase in ("pre", "post"):
            if phase == "post":
                held_during_step = manager.is_attached(Limb.LEFT_HAND)
                mujoco.mj_step(model, data)
                applied_world = _hand_reactions(model, data)[0]
                applied = float(np.linalg.norm(applied_world))
                safe &= applied <= capacity + 1e-9
                if data.time <= .005 + 1e-12:
                    max_reaction_window = max(max_reaction_window, applied)
            decisions = manager.evaluate_and_update(applied_data=data if phase == "post" else None)
            for decision in decisions.values():
                safe &= decision.required_load <= decision.effective_capacity + 1e-9 or not decision.maintain
                if not decision.maintain:
                    transitions.append({"time_s": float(data.time), "phase": phase,
                                        "load_N": decision.required_load, "capacity_N": decision.effective_capacity})
        fresh = _fresh(model, data)
        fresh_world = manager.get_grasp_force(Limb.LEFT_HAND)
        native_world = _hand_reactions(model, fresh)[0]
        force_mapping &= bool(np.max(np.abs(np.array(fresh_world) - native_world)) < 1e-8)
        penetration = max(0., -float(mujoco.mj_geomDistance(model, fresh,
                          model.geom("left_hand_geom").id, model.geom("geom_left_hand").id, 1., None)))
        active = manager.is_attached(Limb.LEFT_HAND)
        if held_during_step:
            max_penetration = max(max_penetration, penetration)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, fresh, mass)
        shoe = shoe_measurement(fixture)
        contacts, _ = _native_contacts(model, fresh, {
            frozenset(("left_foot_geom", "geom_SUPPORT")), frozenset(("left_hand_geom", "geom_left_hand"))})
        palm_collision = sum(float(np.linalg.norm(c["native_wrench"][:3])) for c in contacts
                             if "left_hand_geom" in c["geoms"])
        samples.append({"time_s": float(data.time), "force_state_time_s": float(data.time - dt),
                        "phase": "fixed_field" if step >= round(.4 / dt) else "settling",
                        "force_epoch": "native_mj_step_preintegration", "applied_load_N": applied,
                        "applied_force_world_N": applied_world, "held_during_step": held_during_step,
                        "fresh_load_N": manager.get_grasp_load(Limb.LEFT_HAND), "active": active,
                        "fresh_force_world_N": list(fresh_world),
                        "environment_palm_collision_load_N": palm_collision, "contacts": contacts,
                        "shoe_Fn_N": shoe["Fn_N"], "shoe_status": shoe["status"],
                        "kinetic_energy_J": float(.5 * data.qvel @ mass @ data.qvel),
                        "penetration_m": penetration, "finite": _finite(fresh)})
    loaded = [s for s in samples if s["time_s"] >= .45]
    checks = {"near_capture_accepted": accepted, "capture_preserves_root_state": seed_unchanged,
              "capacity_bounded_every_step": bool(safe), "held_penetration": max_penetration <= .001,
              "production_world_force_mapping": force_mapping,
              "finite": all(s["finite"] and math.isfinite(s["kinetic_energy_J"]) for s in samples),
              "independent_shoe_mg": all(abs(s["shoe_Fn_N"] - MASS * GRAVITY) < .02 * MASS * GRAVITY for s in loaded),
              "weak_releases_strong_holds": (not manager.is_attached(Limb.LEFT_HAND) and bool(transitions)
                                              if capacity == 70. else all(s["active"] for s in samples)),
              "no_foot_equalities_or_motors": model.neq == 1 and model.nu == 0}
    return {"kind": "hand", "capacity_N": capacity, "dt_s": dt, "checks": checks,
            "model_proof": _model_proof(fixture), "pre_capture": pre_capture,
            "capture_events": manager.capture_events, "releases": manager.releases,
            "weak_rejected_load_N": max((r["required_load_N"] for r in manager.releases), default=0.)
                                    if capacity == 70. else None,
            "energy_scope": "Instantaneous generalized kinetic energy; no claim of energy conservation across capture/steps",
            "contact_acceptance": all(checks.values()), "max_reaction_first_0p005s_N": max_reaction_window,
            "max_applied_load_N": max(s["applied_load_N"] for s in samples),
            "fixed_field_max_applied_load_N": max(s["applied_load_N"] for s in samples if s["phase"] == "fixed_field"),
            "max_kinetic_energy_J": max(s["kinetic_energy_J"] for s in samples),
            "max_held_penetration_m": max_penetration, "transitions": transitions,
            "shoe_normal_mean_N": float(np.mean([s["shoe_Fn_N"] for s in loaded])), "samples": samples}


def run_capture_gates(dt: float = .002) -> dict:
    records = []
    for gap, speed, angle, expected in ((.0005, .025, 0., True), (.001, .025, 0., True),
                                       (.0005, .05, 0., True), (.002, 0., 0., False),
                                       (.0005, .05001, 0., False), (.0005, .06, 0., False), (0., 0., 45., False)):
        fixture = make_hand_fixture(dt, gap=gap, speed=speed, angle_deg=angle)
        model, data, manager = fixture.model, fixture.data, fixture.manager
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        region = fixture.scene.region("left_hand")
        pre_capture = _capture_before(fixture, Limb.LEFT_HAND, region)
        allowed = manager.can_attach(Limb.LEFT_HAND, region)
        acquired = manager.attach(Limb.LEFT_HAND, region)
        records.append({"gap_m": gap, "speed_m_s": speed, "angle_deg": angle,
                        "eligible": allowed, "acquired": acquired, "expected": expected,
                        "pre_capture": pre_capture, "capture_events": manager.capture_events,
                        "state_unchanged": bool(np.array_equal(qpos, data.qpos) and np.array_equal(qvel, data.qvel))})
    far = make_hand_fixture(dt, gap=.002, speed=0.)
    try:
        forced = far.manager.attach(Limb.LEFT_HAND, far.scene.region("left_hand"), force=True)
    except ValueError:
        forced = False
    checks = {"physical_capture_gates": all(r["eligible"] == r["expected"] and r["acquired"] == r["expected"] for r in records),
              "read_only_root_state": all(r["state_unchanged"] for r in records),
              "no_physical_force_bypass": not forced and not far.manager.is_attached(Limb.LEFT_HAND)}
    return {"kind": "hand_capture", "dt_s": dt, "cases": records, "checks": checks,
            "contact_acceptance": all(checks.values())}


def make_mixed_fixture(dt: float = .002) -> ContactFixture:
    if dt not in (.002, .001):
        raise ValueError("benchmark timestep must be 0.002 or 0.001 seconds")
    profile = ClimberProfile("stage2_mixed")
    empty = mujoco.MjModel.from_xml_string(build_mjcf(_scene(), profile))
    reference = empty.qpos0.copy()
    np.testing.assert_array_equal(reference[:7], [0., -.46, 1.25, 1., 0., 0., 0.])
    for side in SIDES:
        for name, value in (("shoulder_pitch", .3), ("elbow", math.pi / 2), ("wrist", -.3)):
            reference[empty.joint(f"{side}_{name}").qposadr[0]] = value
    probe = mujoco.MjData(empty)
    probe.qpos[:] = reference
    mujoco.mj_forward(empty, probe)
    regions = []
    for side, x in zip(SIDES, (-.1275, .1275)):
        regions.append(ContactRegion(f"{side}_foot", SourceType.HOLD, (x, -.36, .315),
                       (0., -1., 0.), .9, frozenset({Affordance.STEP}), half_size=(.06, .05, .03)))
        center = probe.site(f"{side}_hand_site").xpos - .026 * np.array([0., -1., 0.])
        regions.append(ContactRegion(f"{side}_hand", SourceType.HOLD, tuple(center),
                       (0., -1., 0.), .9, frozenset({Affordance.GRASP}), radius=.02))
    scene = _scene(regions, (WallSurface("bench_wall", (0., .05, 1.5), (1.15, .05, 1.5)),))
    tree = ET.fromstring(build_mjcf(scene, profile))
    # Only episode timestep/solve accuracy differ from the production builder.
    tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    manager = GraspManager(model, data, scene, profile=profile)
    pre_captures = []
    for limb, side in zip(HANDS, SIDES):
        pre_captures.append(_capture_before(ContactFixture(model, data, scene, profile), limb,
                                            scene.region(f"{side}_hand")))
        if not manager.attach(limb, scene.region(f"{side}_hand")):
            raise RuntimeError(f"canonical mixed fixture could not acquire {limb.value}")
    manager.synchronize_from_live()
    if not np.array_equal(data.qpos, reference) or np.any(data.qvel):
        raise AssertionError("initialization changed the caller's root/pose/velocity")
    fixture = ContactFixture(model, data, scene, profile, manager, reference, ET.tostring(tree, encoding="unicode"))
    for event, before in zip(manager.capture_events, pre_captures):
        event["pre_activation"] = before
    return fixture


def verify_fixed_allocation(fixture: ContactFixture) -> dict:
    """Recompute all 31 balance rows using NumPy, without solving an allocation."""
    model = fixture.model
    data = mujoco.MjData(model)
    data.qpos[:] = fixture.reference
    data.eq_active[:] = False
    mujoco.mj_forward(model, data)
    generalized = np.zeros(model.nv)
    corners, cops = [], {}
    for side_index, side in enumerate(SIDES):
        shoe, ledge = model.geom(f"{side}_foot_geom").id, model.geom(f"geom_{side}_foot").id
        np.testing.assert_allclose(data.geom_xmat[shoe].reshape(3, 3), np.eye(3), atol=1e-14)
        np.testing.assert_allclose(data.geom_xmat[ledge].reshape(3, 3), np.eye(3), atol=1e-14)
        lo = np.maximum(data.geom_xpos[shoe, :2] - model.geom_size[shoe, :2],
                        data.geom_xpos[ledge, :2] - model.geom_size[ledge, :2])
        hi = np.minimum(data.geom_xpos[shoe, :2] + model.geom_size[shoe, :2],
                        data.geom_xpos[ledge, :2] + model.geom_size[ledge, :2])
        z = data.geom_xpos[ledge, 2] + model.geom_size[ledge, 2]
        if np.any(hi <= lo) or abs(z - data.geom_xpos[shoe, 2] + model.geom_size[shoe, 2]) > 1e-12:
            raise AssertionError("fixed certificate does not match the real overlap patch")
        points = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        forces = CORNER_FORCES[4 * side_index:4 * side_index + 4]
        for point, force in zip(points, forces):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, point, model.body(f"{side}_foot").id)
            generalized += jp.T @ force
            corners.append({"side": side, "point_world_m": point.tolist(), "force_world_N": force.tolist()})
        cop = np.sum(points * forces[:, 2, None], axis=0) / forces[:, 2].sum()
        cops[side] = {"world_m": cop.tolist(), "inside_overlap": bool(np.all(cop[:2] >= lo)
                                                                        and np.all(cop[:2] <= hi))}
    for side, force in zip(SIDES, HAND_FORCES):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{side}_hand_site").id)
        generalized += jp.T @ force
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    tau = data.qfrc_bias[dofs] - generalized[dofs]
    balance = generalized - data.qfrc_bias
    balance[dofs] += tau
    checks = {"all_31_balance_rows": model.nv == 31 and np.max(np.abs(balance)) < 1e-9,
              "all_six_root_rows": np.max(np.abs(balance[:6])) < 1e-9,
              "positive_corner_pressure": bool(np.all(CORNER_FORCES[:, 2] >= 1.)),
              "friction_pyramid": bool(np.all(np.abs(CORNER_FORCES[:, :2]).sum(axis=1)
                                               <= .9 * CORNER_FORCES[:, 2] + 1e-9)),
              "cop_in_real_overlap": all(c["inside_overlap"] for c in cops.values()),
              "motor_limits": bool(np.all(np.abs(tau) <= model.actuator_gear[:, 0])),
              "hand_capacity": bool(np.all(np.linalg.norm(HAND_FORCES, axis=1) <= fixture.profile.grip_capacity))}
    checks = {key: bool(value) for key, value in checks.items()}
    return {"checks": checks, "valid": all(checks.values()), "tau_ff_Nm": tau.tolist(),
            "max_balance_residual": float(np.max(np.abs(balance))), "root_residual": balance[:6].tolist(),
            "max_motor_utilization": float(np.max(np.abs(tau) / model.actuator_gear[:, 0])),
            "corner_forces": corners, "hands_world_N": HAND_FORCES.tolist(), "cop": cops,
            "scope": "offline fixed-force certificate; not initial native contact reactions"}


def _native_contacts(model, data, allowed):
    records, unexpected = [], []
    for index, contact in enumerate(data.contact):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        names = [model.geom(contact.geom1).name, model.geom(contact.geom2).name]
        row = {"geoms": names, "native_wrench": wrench.tolist(), "distance_m": float(contact.dist),
               "force_world_on_geom2_N": (contact.frame.reshape(3, 3).T @ wrench[:3]).tolist()}
        records.append(row)
        if frozenset(names) not in allowed and np.linalg.norm(wrench[:3]) > 1e-6:
            unexpected.append(row)
    return records, unexpected


def _hand_reactions(model, data):
    vectors = []
    for limb, side in zip(HANDS, SIDES):
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                             get_grasp_equality_name(limb, f"{side}_hand"))
        rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq))
        if len(rows):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{side}_hand_site").id)
            # A site1-to-static-site2 connect has positive world-Jacobian rows.
            np.testing.assert_allclose(data.efc_J.reshape(data.nefc, model.nv)[rows], jp, atol=1e-10, rtol=0.)
        vectors.append(data.efc_force[rows].tolist() if len(rows) else [0., 0., 0.])
    return vectors


def _penetrations(fixture, data):
    return [max(0., -float(mujoco.mj_geomDistance(fixture.model, data,
                 fixture.model.geom(f"{side}_hand_geom").id,
                 fixture.model.geom(f"geom_{side}_hand").id, 1., None))) for side in SIDES]


def _mixed_measurement(fixture, data, epoch):
    model, manager = fixture.model, fixture.manager
    feet = _feet(model, data, fixture.scene, fresh=False)
    vectors = _hand_reactions(model, data)
    allowed = {frozenset((f"{side}_{part}_geom", f"geom_{side}_{part}"))
               for side in SIDES for part in ("foot", "hand")}
    contacts, unexpected = _native_contacts(model, data, allowed)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    row = {"state_time_s": float(data.time), "force_epoch": epoch, "force_frame": "world",
           "root_z_m": float(data.qpos[2]), "root_linear_m_s": float(np.linalg.norm(data.qvel[:3])),
           "root_angular_rad_s": float(np.linalg.norm(data.qvel[3:6])),
           "joint_max_rad_s": float(np.max(np.abs(data.qvel[dofs]))),
           "joint_rms_rad_s": float(np.sqrt(np.mean(data.qvel[dofs] ** 2))),
           "torque_utilization": float(np.max(np.abs(data.ctrl))),
           "hand_forces_world_N": vectors, "hand_loads_N": np.linalg.norm(vectors, axis=1).tolist(),
           "hand_active": [manager.is_attached(limb) for limb in HANDS],
           "hand_penetration_m": _penetrations(fixture, data), "feet": {},
           "contacts": contacts, "unexpected_contacts": unexpected, "finite": _finite(data)}
    for side, limb in zip(SIDES, FEET):
        state = feet[limb]
        rotation = data.geom(f"{side}_foot_geom").xmat.reshape(3, 3)
        alignment = float(rotation[:, 2] @ data.geom(f"geom_{side}_foot").xmat.reshape(3, 3)[:, 2])
        row["feet"][side] = {"Fn_N": float(state.normal_force), "Ft_N": float(state.tangential_force),
                             "slip_m_s": float(state.tangential_speed), "alignment": alignment,
                             "utilization": float(state.friction_utilization), "status": state.status.value,
                             "regions": list(state.support_regions),
                             "valid_support": bool(state.supporting and state.normal_force > 5.
                                                   and state.tangential_speed <= .01 and alignment >= .9
                                                   and state.friction_utilization <= 1. + 1e-5)}
    return row


def run_mixed_case(dt: float = .002, *, duration: float = 6., settle: float = 1.,
                   observer: Callable | None = None, keep_samples: bool = True) -> dict:
    if not 0. <= settle < duration or abs(duration / dt - round(duration / dt)) > 1e-9:
        raise ValueError("need a nonempty scored interval and integral native steps")
    fixture = make_mixed_fixture(dt)
    model, data, manager = fixture.model, fixture.data, fixture.manager
    allocation = verify_fixed_allocation(fixture)
    if not allocation["valid"]:
        raise AssertionError("fixed force certificate invalid for production geometry")
    joints = model.actuator_trnid[:, 0]
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    tau_ff, gear = np.array(allocation["tau_ff_Nm"]), model.actuator_gear[:, 0]
    data.ctrl[:] = tau_ff / gear
    initial_data = _fresh(model, data)
    initial = _mixed_measurement(fixture, initial_data, "initial_fresh_forward")
    initial["root_qacc"] = initial_data.qacc[:6].tolist()
    initial["root_linear_qacc_norm"] = float(np.linalg.norm(initial_data.qacc[:3]))
    initial["root_angular_qacc_norm"] = float(np.linalg.norm(initial_data.qacc[3:6]))
    initial["qpos_matches_caller"] = bool(np.array_equal(data.qpos, fixture.reference))
    initial["qvel_zero"] = bool(not np.any(data.qvel))
    initial["no_external_forces"] = bool(not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied))
    samples, applied, transitions = [], [], []
    safe, time_jumps, max_held_penetration = True, 0, 0.
    for step in range(round(duration / dt)):
        # Benchmark-only Nm PD, normalized by unchanged production motor gears.
        data.ctrl[:] = np.clip((tau_ff + 80. * (fixture.reference[qids] - data.qpos[qids])
                                - data.qvel[dofs]) / gear, -1., 1.)
        for limb, decision in manager.evaluate_and_update().items():
            if not decision.maintain:
                transitions.append({"time_s": float(data.time), "phase": "pre_step", "limb": limb.value,
                                    "load_N": decision.required_load, "capacity_N": decision.effective_capacity})
        # Endpoint preview operates on owned data, never on the live root.
        preview = mujoco.MjData(model)
        mujoco.mj_copyData(preview, model, data)
        mujoco.mj_step(model, preview)
        mujoco.mj_forward(model, preview)
        for limb, penetration in zip(HANDS, _penetrations(fixture, preview)):
            if manager.is_attached(limb) and penetration > .001:
                manager.detach(limb)
                transitions.append({"time_s": float(data.time), "phase": "preview", "limb": limb.value,
                                    "reason": "penetration", "penetration_m": penetration})
        before = float(data.time)
        active = [manager.is_attached(limb) for limb in HANDS]
        pre_row = _mixed_measurement(fixture, _fresh(model, data), "pre_step_fresh_forward")
        mujoco.mj_step(model, data)
        native = _hand_reactions(model, data)
        applied_loads = np.linalg.norm(native, axis=1)
        native_feet = _feet(model, data, fixture.scene, fresh=False)
        _, native_unexpected = _native_contacts(model, data, {
            frozenset((f"{side}_{part}_geom", f"geom_{side}_{part}"))
            for side in SIDES for part in ("foot", "hand")})
        interval = {"interval_start_s": before, "interval_end_s": float(data.time),
                    "force_state_time_s": before, "force_epoch": "native_mj_step_preintegration",
                    "hand_forces_world_N": native, "hand_loads_N": applied_loads.tolist(),
                    "hand_active_during_step": active,
                    "feet": {side: {"Fn_N": float(native_feet[limb].normal_force),
                                    "Ft_N": float(native_feet[limb].tangential_force)}
                             for side, limb in zip(SIDES, FEET)},
                    "unexpected_contacts": native_unexpected}
        applied.append(interval)
        for limb, decision in manager.evaluate_and_update(applied_data=data).items():
            safe &= decision.required_load <= decision.effective_capacity + 1e-9
            if not decision.maintain:
                transitions.append({"time_s": float(data.time), "phase": "post_step", "limb": limb.value,
                                    "load_N": decision.required_load, "capacity_N": decision.effective_capacity})
        endpoint = _fresh(model, data)
        row = _mixed_measurement(fixture, endpoint, "post_step_fresh_forward")
        for was_active, penetration in zip(active, row["hand_penetration_m"]):
            if was_active:
                max_held_penetration = max(max_held_penetration, penetration)
        safe &= bool(_finite(data) and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
                     and np.all(applied_loads <= fixture.profile.grip_capacity + 1e-9))
        time_jumps += int(abs(data.time - before - dt) > 1e-12)
        samples.append({"pre": pre_row, "post": row})
        if observer is not None and step % max(1, round(.05 / dt)) == 0:
            observed_qpos, observed_qvel = data.qpos.copy(), data.qvel.copy()
            observer(fixture, manager.contact_snapshot())
            if not np.array_equal(data.qpos, observed_qpos) or not np.array_equal(data.qvel, observed_qvel):
                raise AssertionError("render observer changed live generalized state")
    scored = [s["pre"] for s in samples if s["pre"]["state_time_s"] >= settle - 1e-9]
    scored += [s["post"] for s in samples if s["pre"]["state_time_s"] >= settle - 1e-9]
    checks = {"finite_no_external_forces_capacity_bounded": bool(safe),
              "capacity_every_step_including_settle": len(applied) == round(duration / dt),
              "contiguous_time": time_jumps == 0,
              "held_penetration": max_held_penetration <= .001,
              "both_feet_loaded_non_slip": all(all(f["valid_support"] for f in s["feet"].values()) for s in scored),
              "both_hands_loaded": all(all(s["hand_active"]) and min(s["hand_loads_N"]) > 5. for s in scored),
              "no_secondary_support": all(not s[phase]["unexpected_contacts"] for s in samples for phase in ("pre", "post"))
                                      and all(not s["unexpected_contacts"] for s in applied),
              "no_warnings": not any(w.number for w in data.warning)}
    controller = {"root_linear_lt_0p05": max(s["root_linear_m_s"] for s in scored) < .05,
                  "root_angular_lt_0p1": max(s["root_angular_rad_s"] for s in scored) < .1,
                  "max_hinge_lt_0p2": max(s["joint_max_rad_s"] for s in scored) < .2}
    result = {"kind": "mixed", "dt_s": dt, "duration_s": float(data.time), "settle_s": settle,
              "model_proof": _model_proof(fixture), "capture_events": manager.capture_events,
              "releases": manager.releases,
              "scored_duration_s": duration - settle, "checks": checks, "contact_acceptance": all(checks.values()),
              "controller_checks": controller, "controller_convergence": all(controller.values()),
              "controller_status": "VALIDATED" if all(controller.values()) else "NOT VALIDATED",
              "allocation": allocation, "initial": initial, "time_jump_count": time_jumps,
              "max_held_penetration_m": max_held_penetration, "transitions": transitions,
              "root_linear_max_m_s": max(s["root_linear_m_s"] for s in scored),
              "root_linear_rms_m_s": float(np.sqrt(np.mean([s["root_linear_m_s"] ** 2 for s in scored]))),
              "root_angular_max_rad_s": max(s["root_angular_rad_s"] for s in scored),
              "root_angular_rms_rad_s": float(np.sqrt(np.mean([s["root_angular_rad_s"] ** 2 for s in scored]))),
              "root_z_min_m": min(s[p]["root_z_m"] for s in samples for p in ("pre", "post")),
              "joint_max_rad_s": max(s["joint_max_rad_s"] for s in scored),
              "hinge_limit_violation_fraction": float(np.mean([s["joint_max_rad_s"] >= .2 for s in scored])),
              "joint_rms_rad_s": float(np.sqrt(np.mean([s["joint_rms_rad_s"] ** 2 for s in scored]))),
              "max_torque_utilization": max(s[p]["torque_utilization"] for s in samples for p in ("pre", "post")),
              "hand_applied_max_N": np.max([r["hand_loads_N"] for r in applied], axis=0).tolist(),
              "feet": {side: {"support_fraction": float(np.mean([s["feet"][side]["valid_support"] for s in scored])),
                              "Fn_min_N": min(s["feet"][side]["Fn_N"] for s in scored),
                              "Fn_mean_N": float(np.mean([s["feet"][side]["Fn_N"] for s in scored])),
                              "Fn_max_N": max(s["feet"][side]["Fn_N"] for s in scored),
                              "slip_max_m_s": max(s["feet"][side]["slip_m_s"] for s in scored)} for side in SIDES}}
    if keep_samples:
        result.update(samples=samples, applied=applied)
    return result
