"""Fixed-scene morphology experiments over an explicit hand-transfer contract."""
from dataclasses import replace
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_geometry import canonical_geometry
from .contact_ik import solve_hand_reference
from .locomotion import get_state_summary
from .mjcf_builder import build_mjcf
from .schema import Affordance, ClimberProfile, Limb
from .static_control import execute_static_hold
from .static_state import _integration_state, initialize_static_reference
from .transfer_feasibility import assess_hand_transfer
from .transfers import TransferRequest, execute_transfer
from .whole_body_demo import make_whole_body_fixture


def study_profiles():
    base = ClimberProfile("baseline")
    profiles = {"baseline": base}
    for name, scale in (("shorter", .95), ("longer", 1.05)):
        profiles[name] = replace(base, name=name, upper_arm_length=base.upper_arm_length * scale,
                                 forearm_length=base.forearm_length * scale,
                                 thigh_length=base.thigh_length * scale, shin_length=base.shin_length * scale)
    profiles["reduced_grip"] = replace(base, name="reduced_grip", grip_capacity=.5 * base.grip_capacity)
    return profiles


def target_perturbations():
    _, _, scene, base, _ = make_whole_body_fixture()
    radius = scene.region("reach_target").radius
    distance = min(.02 * base.arm_reach, .5 * radius)
    return {"nominal": (0., 0., 0.), "up": (0., distance, 0.), "down": (0., -distance, 0.),
            "lateral_minus": (-distance, 0., 0.), "lateral_plus": (distance, 0., 0.)}


def make_envelope_fixture(profile, target_case="nominal", timestep=.002):
    """Same baseline world for all profiles; independently solve each source pose.

    Source crouch preference is a fixed fraction of compiled knee ROM, with flat
    sole hip/ankle compensation. Only explicit episode initialization may adopt
    this scratch-generated pose. No recorded trajectory or profile pose is used.
    """
    _, _, scene, _, _ = make_whole_body_fixture(timestep)
    target_id = "reach_target"
    source = scene.region(scene.start_configuration[Limb.RIGHT_HAND])
    target = scene.region(target_id)
    frame = canonical_geometry(target).hand_frame
    offsets = target_perturbations()
    if target_case in offsets:
        offset = np.array(frame.rotation) @ offsets[target_case]
    elif target_case == "beyond_reach":
        offset = 1.5 * profile.arm_reach * np.array(frame.rotation)[:, 0]
    elif target_case == "blocked_path":
        offset = np.zeros(3)
    else:
        raise ValueError("Unknown envelope target case")
    scene = replace(scene, contact_regions=tuple(
        replace(r, position=tuple(np.array(r.position) + offset)) if r.id == target_id else r
        for r in scene.contact_regions))
    if target_case == "blocked_path":
        start = canonical_geometry(source).hand_frame
        goal = canonical_geometry(scene.region(target_id)).hand_frame
        midpoint = .5 * (np.array(start.position) + goal.position) + .02 * np.array(goal.normal)
        obstacle = replace(source, id="path_obstacle", position=tuple(midpoint),
                           affordances=frozenset((Affordance.STEP,)), half_size=(.015, .018, .015))
        scene = replace(scene, contact_regions=(*scene.contact_regions, obstacle))
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    seed = model.qpos0.copy()
    for side in ("left", "right"):
        knee = model.joint(f"{side}_knee")
        bend = float(knee.range[0] + .4 * (knee.range[1] - knee.range[0]))
        for name, angle in (("hip_pitch", .5 * bend), ("knee", bend), ("ankle_pitch", .5 * bend)):
            seed[int(model.joint(f"{side}_{name}").qposadr[0])] = angle
        elbow = model.joint(f"{side}_elbow")
        seed[int(elbow.qposadr[0])] = float(np.mean(elbow.range))
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    foot = canonical_geometry(scene.region(scene.start_configuration[Limb.LEFT_FOOT])).foot_frame
    # Center the shoe on the actual bounded face, retaining the validated toe offset.
    planted = np.array(foot.position) + np.array(foot.rotation) @ [0., .01, 0.]
    qa = int(model.joint("root").qposadr[0])
    seed[qa:qa + 3] += planted - scratch.site("left_foot_site").xpos
    for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
        arm = solve_hand_reference(model, seed, limb,
                                   canonical_geometry(scene.region(scene.start_configuration[limb])).hand_frame)
        if not arm.converged:
            raise ValueError("Profile source arm could not be recomputed: " + arm.reason)
        seed = np.array(arm.qpos)
    metadata = {"profile": profile.to_dict(), "target_case": target_case,
                "target_offset_world_m": tuple(offset), "scene": scene.to_dict(),
                "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "seed_qpos": tuple(seed),
                "compiled_rom_rad": {model.joint(j).name: tuple(model.jnt_range[j])
                                     for j in range(model.njnt) if model.jnt_limited[j]},
                "motor_ceiling_Nm": tuple(model.actuator_gear[:, 0]),
                "mass_kg": float(model.body_subtreemass[model.body("climber_root").id])}
    return model, mujoco.MjData(model), scene, profile, seed, metadata


def run_envelope_case(profile, target_case="nominal", timestep=.002, *, keep_samples=True):
    model, data, scene, profile, seed, metadata = make_envelope_fixture(profile, target_case, timestep)
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager,
                                   duration=2., settle=1., score_window=.5, keep_samples=False)
    source_state = get_state_summary(model, data, manager)
    common = {"profile": profile.to_dict(), "fixture_inputs": metadata, "dt_s": timestep,
              "target_case": target_case, "initial_static": initial, "kind": "right_hand",
              "initial_state": source_state, "initial_reference": reference,
              "moving_limb": Limb.RIGHT_HAND, "source": scene.start_configuration[Limb.RIGHT_HAND],
              "target": "reach_target"}
    if not initial["success"]:
        return {**common, "success": False, "classification": "SOURCE_STATIC_FAILURE",
                "status": initial["status"], "reason": initial["reason"], "steps": 0,
                "duration_s": 0., "final_state": source_state, "assessment": None, "samples": []}
    contacts = dict(reference.contact_intent)
    request = TransferRequest(Limb.RIGHT_HAND, contacts[Limb.RIGHT_HAND], "reach_target", contacts,
                              support_contacts={l: h for l, h in contacts.items() if l != Limb.RIGHT_HAND},
                              hand_reach_s=5.)
    before = _integration_state(model, data)
    assessment = assess_hand_transfer(model, data, scene, profile, reference, manager, request)
    if not np.array_equal(before, _integration_state(model, data)):
        raise RuntimeError("Assessment changed the authoritative native integration state")
    if not assessment.feasible:
        return {**common, "assessment": assessment, "success": False,
                "classification": assessment.classification, "status": assessment.classification,
                "reason": assessment.reason, "steps": 0, "duration_s": 0., "final_state": source_state,
                "initial_integration_state": tuple(before), "final_integration_state": tuple(before),
                "final_reference": None, "released": False, "events": [], "phases": [], "samples": []}
    result = execute_transfer(model, data, scene, profile, assessment.source_reference, manager,
                              replace(request, whole_body=assessment.motion), keep_samples=keep_samples)
    classification = "PHYSICAL_SUCCESS" if result["success"] else "DYNAMIC_CONTACT_INFEASIBLE"
    return {**common, **result, "assessment": assessment, "classification": classification}
