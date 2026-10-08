"""Bounded hand-transfer candidates, not dynamic success or global reach proofs.

Every pose here is virtual. Only an admitted source reference and a motion guide
are intended for the maintained executor; failed poses remain diagnostic evidence.
No live commands, resets, attachments, force edits, or native steps occur here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import math
from numbers import Real
from types import MappingProxyType

import mujoco
import numpy as np

from .contact import effective_grip_capacity
from .contact_geometry import ContactMode, Frame, canonical_geometry
from .contact_ik import solve_contact_pose
from .grasp import AttachmentStateError
from .locomotion import _supported_hold_geometry
from .motion_support import estimate_support_torques
from .runtime import validate_reference_pose
from .schema import Limb
from .single_hand import minimum_jerk, reach_frame
from .static_state import (StaticReference, _canonical_surface, _finite_state, _foot_residual,
                           _integration_state, _unexpected_loaded_contacts,
                           initialize_static_reference)
from .support import FORCE_TOLERANCE, MAX_SLIP_SPEED, fresh_data
from .transfers import TransferRequest, _request_error
from .whole_body_motion import WholeBodyMotion, reference_frames
from .whole_body_reference import WholeBodyReferenceResult, solve_whole_body_reference


def _freeze(value):
    if is_dataclass(value):
        return MappingProxyType({f.name: _freeze(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ReferencePolicy:
    """Universal geometry coefficients; never controller gains or guard overrides."""

    rise_fraction: float = .28
    rise_leg_bound: float = .06
    support_lateral_fraction: float = .1
    target_lateral_fraction: float = .1
    lateral_leg_bound: float = .06
    yaw_fraction: float = .3
    yaw_max_rad: float = math.radians(10.)
    waist_pitch_fraction: float = .3
    prepare_s: float = 4.
    sample_interval_s: float = .05

    def __post_init__(self):
        for name in ("rise_fraction", "rise_leg_bound", "support_lateral_fraction",
                     "target_lateral_fraction", "lateral_leg_bound", "yaw_fraction",
                     "waist_pitch_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite coefficient in [0, 1]")
        if (isinstance(self.yaw_max_rad, bool) or not isinstance(self.yaw_max_rad, Real)
                or not math.isfinite(self.yaw_max_rad) or not 0 <= self.yaw_max_rad <= math.pi / 6):
            raise ValueError("yaw_max_rad must be finite and bounded by 30 degrees")
        for name, fixed in (("prepare_s", 4.), ("sample_interval_s", .05)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or value != fixed:
                raise ValueError(f"{name} is fixed at {fixed} seconds")


@dataclass(frozen=True)
class TransferAssessment:
    feasible: bool
    classification: str
    reason: str
    motion: WholeBodyMotion | None
    geometric_feasible: bool | None
    support_feasible: bool | None
    source_reference: StaticReference | None
    preparation_qpos: tuple[tuple[float, ...], ...]
    endpoint: WholeBodyReferenceResult | None
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "preparation_qpos", tuple(tuple(float(v) for v in q) for q in self.preparation_qpos))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))


def _candidate_motion(model, qpos, scene, profile, request, policy):
    root, actual_frames = reference_frames(model, qpos)
    target = canonical_geometry(scene.region(request.target)).hand_frame
    basis = np.asarray(target.rotation)
    lateral, up = basis[:, 0], basis[:, 1]
    moving = actual_frames[request.limb]
    delta = np.asarray(target.position) - moving.position
    dz = max(0., float(delta @ up))
    centroid = np.mean([actual_frames[l].position for l in Limb if l != request.limb], axis=0)
    shift = (policy.support_lateral_fraction * float((centroid - root.position) @ lateral)
             + policy.target_lateral_fraction * float(delta @ lateral))
    shift = float(np.clip(shift, -policy.lateral_leg_bound * profile.leg_reach,
                          policy.lateral_leg_bound * profile.leg_reach))
    rise = min(policy.rise_fraction * dz, policy.rise_leg_bound * profile.leg_reach)
    other = next(l for l in Limb if l.is_hand and l != request.limb)
    handedness = float(np.sign((np.asarray(moving.position) - actual_frames[other].position) @ lateral))
    angle = min(policy.yaw_max_rad, policy.yaw_fraction * math.atan2(dz, profile.arm_reach)) * handedness
    # Yaw is about world up, relative to the actual root orientation, not identity.
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.))) @ np.asarray(root.rotation)
    goal = Frame(tuple(np.asarray(root.position) + rise * up + shift * lateral), tuple(map(tuple, rotation)))
    waist = {n: float(qpos[int(model.joint(n).qposadr[0])])
             for n in ("waist_yaw", "waist_pitch", "waist_roll")}
    waist["waist_pitch"] += policy.waist_pitch_fraction * math.atan2(dz, profile.arm_reach)
    return WholeBodyMotion(goal, waist, policy.prepare_s), target, root, actual_frames


def _support_evidence(model, scene, profile, qpos, contacts, *, foot_points=None):
    """The maintained estimator decides acceptance; unconstrained math explains it.

    Repeat its gravity-share projection only for numeric rejected-candidate
    diagnostics. These proposed forces are never capability-bypassed acceptance.
    """
    result = {"admitted": False, "contacts": dict(contacts), "nominal_foot_share": .9,
              "nominal_hand_share": .1, "scope": "estimated reactions, not measured physics"}
    try:
        result["estimate"] = estimate_support_torques(model, scene, profile, qpos, contacts,
                                                     foot_points=foot_points)
        result["admitted"] = True
    except ValueError as error:
        result["reason"] = str(error)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    feet = tuple(l for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT) if l in contacts)
    hands = tuple(l for l in (Limb.LEFT_HAND, Limb.RIGHT_HAND) if l in contacts)
    weight = -model.body_subtreemass[model.body("climber_root").id] * model.opt.gravity
    blocks, points, nominal = [], [], []
    try:
        for limb in (*feet, *hands):
            if limb.is_foot:
                region = scene.region(contacts[limb])
                point = (np.asarray(foot_points[limb]) if foot_points is not None and limb in foot_points else
                         np.asarray(_foot_residual(model, scratch, limb, region,
                                                  canonical_geometry(region)).support_point_world))
                body = int(model.geom(f"{limb.value.lower()}_geom").bodyid[0])
                share = .9 / len(feet)
            else:
                site = scratch.site(f"{limb.value.lower()}_site")
                point, body, share = site.xpos.copy(), int(model.site_bodyid[site.id]), .1 / len(hands)
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, scratch, jac, None, point, body)
            blocks.append(jac.T)
            points.append(tuple(point))
            nominal.extend(share * weight)
        jacobian = np.concatenate(blocks, axis=1)
        root, nominal = jacobian[:6], np.asarray(nominal)
        force = nominal + np.linalg.lstsq(root, scratch.qfrc_bias[:6] - root @ nominal, rcond=None)[0]
        margins = {}
        for index, limb in enumerate((*feet, *hands)):
            region, vector = scene.region(contacts[limb]), force[3 * index:3 * index + 3]
            if limb.is_foot:
                local = np.asarray(canonical_geometry(region).foot_surface_frame.rotation).T @ vector
                margins[limb] = {"normal_N": float(local[2]), "minimum_load_margin_N": float(local[2] - 5.),
                                 "friction_margin_N": float(min(1.8, region.friction) * local[2] - np.abs(local[:2]).sum())}
            else:
                load = float(np.linalg.norm(vector))
                capacity = effective_grip_capacity(profile, region, region.normal)
                margins[limb] = {"load_N": load, "capacity_N": capacity, "margin_N": capacity - load}
        joints = model.actuator_trnid[:, 0]
        torque = scratch.qfrc_bias[model.jnt_dofadr[joints]] - (jacobian @ force)[model.jnt_dofadr[joints]]
        ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        result.update(proposed_forces_world_N=force.reshape(-1, 3), points_world_m=points, margins=margins,
                      root_balance_residual=float(np.max(np.abs(root @ force - scratch.qfrc_bias[:6]))),
                      proposed_torques_Nm={model.joint(int(j)).name: float(t) for j, t in zip(joints, torque)},
                      motor_margin_min_Nm=float(np.min(ceiling - np.abs(torque))),
                      motor_utilization_max=float(np.max(np.abs(torque) / ceiling)))
    except ValueError as error:
        result["diagnostic_error"] = str(error)
    return result


def _declared_hand_path(scene, request, target, policy):
    source = canonical_geometry(scene.region(request.source)).hand_frame
    clear = Frame(tuple(np.asarray(source.position) + .01 * np.asarray(source.normal)), source.rotation)
    for phase, start, goal, duration, clearance in (("clearance", source, clear, .5, 0.),
                                                   ("reach", clear, target, request.hand_reach_s, .02)):
        for index in range(math.ceil(duration / policy.sample_interval_s) + 1):
            time = min(duration, index * policy.sample_interval_s)
            yield phase, index, time, reach_frame(start, goal, time, duration, clearance)


def _path_collision_certificate(model, source_qpos, request, samples):
    """Prove intersection for a prescribed END point, independent of arm twist.

    A point interior to both rigid solids contains a shared open ball. Thus no
    exact position solution for this sampled path can avoid their intersection.
    Absence of a witness proves nothing about the rest of the path or the IK.
    """
    disabled = mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if model.opt.disableflags & disabled:
        return None
    hand = model.geom(f"{request.limb.value.lower()}_geom").id
    site = model.site(f"{request.limb.value.lower()}_site").id
    hand_body = int(model.geom_bodyid[hand])
    if (model.geom_type[hand] != mujoco.mjtGeom.mjGEOM_BOX or model.site_bodyid[site] != hand_body
            or model.body_weldid[hand_body] == 0):
        return None
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[hand])
    local_site = rotation.reshape(3, 3).T @ (model.site_pos[site] - model.geom_pos[hand])
    hand_margin = float(np.min(model.geom_size[hand] - np.abs(local_site)))
    if hand_margin <= .001:
        return None
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = source_qpos
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    intentional = {model.geom(f"geom_{hold}").id for hold in (request.source, request.target)}
    explicit = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    obstacles = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        kind = int(model.geom_type[geom])
        if (geom in intentional or model.body_weldid[body] != 0
                or kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE)):
            continue
        paired = frozenset((hand, geom)) in explicit
        signature = (min(hand_body, body) << 16) | max(hand_body, body)
        # Explicit pairs bypass masks and exclusions. Custom filters can depend
        # on a future pose, so automatic pairs cannot provide this certificate.
        if not paired and (signature in model.exclude_signature or mujoco.get_mjcb_contactfilter() is not None
                           or not ((int(model.geom_contype[hand]) & int(model.geom_conaffinity[geom]))
                                   or (int(model.geom_contype[geom]) & int(model.geom_conaffinity[hand])))):
            continue
        obstacles.append((geom, body, kind, paired))
    for phase, index, time, frame in samples:
        point = np.asarray(frame.position)
        for geom, body, kind, paired in obstacles:
            local = scratch.geom_xmat[geom].reshape(3, 3).T @ (point - scratch.geom_xpos[geom])
            margin = float(np.min(model.geom_size[geom] - np.abs(local)) if kind == mujoco.mjtGeom.mjGEOM_BOX
                           else model.geom_size[geom, 0] - np.linalg.norm(local))
            if margin <= .001:
                continue
            return {"phase": phase, "index": index, "time_s": time, "task_frame": frame,
                    "point_world_m": tuple(point), "point_obstacle_local_m": tuple(local),
                    "site_hand_local_m": tuple(local_site), "hand_interior_margin_m": hand_margin,
                    "obstacle_interior_margin_m": margin,
                    "intersection_ball_radius_m": min(hand_margin, margin),
                    "moving_geom": model.geom(hand).name, "moving_site": model.site(site).name,
                    "static_geom": model.geom(geom).name, "static_body": model.body(body).name,
                    "static_shape": "box" if kind == mujoco.mjtGeom.mjGEOM_BOX else "sphere",
                    "static_geom_frame": Frame(tuple(scratch.geom_xpos[geom]),
                                               tuple(map(tuple, scratch.geom_xmat[geom].reshape(3, 3)))),
                    "static_body_frame": Frame(tuple(scratch.xpos[body]),
                                               tuple(map(tuple, scratch.xmat[body].reshape(3, 3)))),
                    "explicit_pair": paired, "static_body_weldid": int(model.body_weldid[body]),
                    "collision_masks": {"hand": (int(model.geom_contype[hand]), int(model.geom_conaffinity[hand])),
                                        "obstacle": (int(model.geom_contype[geom]), int(model.geom_conaffinity[geom]))},
                    "proof": "Required END point is strictly interior to both collidable rigid solids",
                    "scope": "specified sampled effector path only; not a global transfer or native force claim"}
    return None


def assess_hand_transfer(model, data, scene, profile, reference, manager, request: TransferRequest,
                         *, policy: ReferencePolicy = ReferencePolicy()) -> TransferAssessment:
    """Assess one explicit bounded candidate without mutating the physical session.

    True means sampled geometry, strict scratch admission, and maintained static
    support estimates passed. It does NOT mean dynamics/capture/readiness passed.
    A failure rejects this candidate only, never all motions of this climber.
    """
    state = _integration_state(model, data)
    diagnostics = {"scope": "this bounded policy candidate only; no dynamic or readiness certificate",
                   "actual_state_digest": hashlib.sha256(state.tobytes()).hexdigest(),
                   "actual_integration_state": tuple(state), "actual_qpos": tuple(data.qpos),
                   "preparation_results": [], "reach_results": [], "candidate_support": [],
                   "endpoint_admitted": False, "path_collision_certificate": None}
    motion = source = endpoint = None
    path = []
    geometric = support = None

    def finish(classification, reason, feasible=False):
        return TransferAssessment(feasible, classification, reason, motion, geometric, support,
                                  source, tuple(path), endpoint, diagnostics)

    try:
        manager.require_session(model, data, scene)
    except AttachmentStateError as error:
        return finish("SOURCE_INFEASIBLE", str(error))
    error = _request_error(scene, request, float(model.opt.timestep))
    if error:
        return finish(*error)
    if not request.limb.is_hand:
        return finish("UNSUPPORTED_PRIMITIVE", "This policy assesses hand transfers only")
    if request.whole_body is not None or not isinstance(policy, ReferencePolicy):
        return finish("INVALID_REQUEST", "Supply ReferencePolicy, not a pre-existing whole-body goal")
    dt = float(model.opt.timestep)
    if any(abs(t / dt - round(t / dt)) > 1e-8 for t in (policy.prepare_s, policy.sample_interval_s)):
        return finish("INVALID_REQUEST", "Policy sampling and preparation require integral native intervals")
    if (manager.mode != ContactMode.PHYSICAL or manager.profile is not profile or not _finite_state(model, data)
            or data.qfrc_applied.any() or data.xfrc_applied.any()):
        return finish("SOURCE_INFEASIBLE", "Require bound physical profile, finite native state and zero external force channels")
    try:
        validate_reference_pose(model, data.qpos)
    except ValueError as error:
        diagnostics["invalid_actual_qpos"] = tuple(data.qpos)
        return finish("ROM_INFEASIBLE" if "compiled ROM" in str(error) else "SOURCE_INFEASIBLE", str(error))
    intent = dict(request.source_contacts)
    supports = {l: h for l, h in intent.items() if l != request.limb}
    diagnostics["compiled_rom_rad"] = {model.joint(j).name: tuple(model.jnt_range[j])
                                       for j in range(model.njnt) if model.jnt_limited[j]}
    diagnostics["actual_rom_margin_min_rad"] = min(
        min(data.qpos[int(model.jnt_qposadr[j])] - model.jnt_range[j, 0],
            model.jnt_range[j, 1] - data.qpos[int(model.jnt_qposadr[j])])
        for j in range(model.njnt) if model.jnt_limited[j])
    diagnostics["motor_ceiling_Nm"] = tuple(model.actuator_gear[:, 0] * np.minimum(
        -model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]))
    try:
        snapshot = manager.contact_snapshot()
        diagnostics["native_contacts"] = snapshot
        if snapshot.configuration != intent or not isinstance(reference, StaticReference) or dict(reference.contact_intent) != intent:
            return finish("SOURCE_INFEASIBLE", "Explicit source contacts must match the actual native source and reference intent")
        for limb, hand in snapshot.hands.items():
            if (not hand.active or not hand.valid or not hand.measurement_valid or hand.capacity is None
                    or not np.isfinite([hand.load, hand.capacity, *hand.force_world]).all()
                    or hand.load > hand.capacity or hand.region_id != intent[limb]
                    or np.linalg.norm(manager._reaction(limb, data)) > hand.capacity):
                return finish("SOURCE_INFEASIBLE", "Both actual native hands must be active, finite, valid and bounded")
        for limb, foot in snapshot.feet.items():
            if (not foot.measurement_valid or not foot.supporting or foot.slipping or foot.normal_force <= 5.
                    or not np.isfinite([foot.normal_force, foot.tangential_speed, foot.friction_utilization]).all()
                    or foot.tangential_speed > MAX_SLIP_SPEED or not _supported_hold_geometry(foot, intent[limb])
                    or any(c.normal_force > FORCE_TOLERANCE and not c.admissible for c in foot.contacts)):
                return finish("SOURCE_INFEASIBLE", "Both actual native feet must carry more than 5 N without slip on declared holds")
        allowed = {frozenset((f"{l.value.lower()}_geom", f"geom_{h}")) for l, h in intent.items()}
        collisions = (_unexpected_loaded_contacts(model, data, allowed)
                      + _unexpected_loaded_contacts(model, fresh_data(model, data), allowed))
        diagnostics["actual_unexpected_loaded_contacts"] = collisions
        if collisions:
            return finish("SOURCE_INFEASIBLE", "Actual source has unexpected loaded contacts")
        manager.registered_attachment(request.limb, scene.region(request.target))
        _canonical_surface(model, fresh_data(model, data), scene.region(request.target))
        motion, target, actual_root, actual_frames = _candidate_motion(model, data.qpos.copy(), scene, profile, request, policy)
        diagnostics.update(actual_root=actual_root, actual_frames=actual_frames, target_frame=target, policy=policy)
        for name, angle in motion.waist_target.items():
            joint = model.joint(name)
            if joint.limited[0] and not joint.range[0] <= angle <= joint.range[1]:
                geometric = False
                return finish("ROM_INFEASIBLE", f"Policy waist goal {name} is outside compiled ROM; no clipping")
        source_result = solve_contact_pose(model, scene, profile, data.qpos.copy(), intent)
        diagnostics["source_result"] = source_result
        if not source_result.admitted:
            return finish("SOURCE_INFEASIBLE", "Measured source could not be scratch-admitted: " + source_result.reason)
        source = source_result.reference
    except (ValueError, KeyError, AttachmentStateError) as error:
        return finish("SOURCE_INFEASIBLE", str(error))

    # A triangle-inequality proof for this fixed virtual root/waist, before IK.
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = source.qpos
    root_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    qa = int(model.jnt_qposadr[root_joint])
    scratch.qpos[qa:qa + 7] = (*motion.root_target.position, *motion.root_target.quaternion)
    for name, angle in motion.waist_target.items():
        scratch.qpos[int(model.joint(name).qposadr[0])] = angle
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    side = request.limb.value.lower().removesuffix("_hand")
    shoulder = model.body(f"{side}_upper_arm").id
    site = model.site(f"{side}_hand_site").id
    body = int(model.site_bodyid[site])
    maximum = float(np.linalg.norm(model.site_pos[site]))
    while body != shoulder:
        maximum += float(np.linalg.norm(model.body_pos[body]))
        body = int(model.body_parentid[body])
        if body == 0:
            return finish("INVALID_REQUEST", "Hand site is not in the compiled shoulder chain")
    distance = float(np.linalg.norm(np.asarray(target.position) - scratch.xpos[shoulder]))
    diagnostics["reach_bound"] = {"shoulder_world_m": tuple(scratch.xpos[shoulder]), "distance_m": distance,
                                  "maximum_m": maximum, "gap_m": distance - maximum,
                                  "scope": "fixed policy endpoint root/waist; position-only upper bound"}
    if distance > maximum + 1e-6:
        geometric = False
        return finish("GEOMETRY_INFEASIBLE", "Target lies outside compiled hand-chain sphere at this policy root/waist")

    root, frames = reference_frames(model, source.qpos)
    waist = {n: source.target_pose[n] for n in motion.waist_target}
    seed = np.asarray(source.qpos)
    native_points = {l: np.mean([c.point for c in snapshot.feet[l].contacts
                                if c.surface_geom == "geom_" + intent[l] and c.admissible], axis=0)
                     for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)}
    native_support = _support_evidence(model, scene, profile, data.qpos.copy(), supports, foot_points=native_points)
    diagnostics["actual_candidate_support"] = native_support
    support_failures = []
    if not native_support["admitted"]:
        support = False
        support_failures.append(native_support.get("reason", "Native-pose support estimate failed"))
    count = round(policy.prepare_s / policy.sample_interval_s)
    initial_four = None

    def record_support(qpos, contacts, phase, index):
        nonlocal support
        evidence = _support_evidence(model, scene, profile, qpos, contacts)
        evidence["rom_margin_min_rad"] = min(
            min(qpos[int(model.jnt_qposadr[j])] - model.jnt_range[j, 0],
                model.jnt_range[j, 1] - qpos[int(model.jnt_qposadr[j])])
            for j in range(model.njnt) if model.jnt_limited[j])
        diagnostics["candidate_support"].append({"phase": phase, "index": index, **evidence})
        if not evidence["admitted"]:
            support = False
            support_failures.append(evidence.get("reason", "Support estimate failed"))
        return evidence

    def search_failure(solution):
        diagnostics["failed_result"] = solution
        diagnostics["failed_qpos"] = solution.qpos
        active = []
        for jid in model.actuator_trnid[:, 0]:
            angle = solution.qpos[int(model.jnt_qposadr[jid])]
            if model.jnt_limited[jid] and min(abs(angle - model.jnt_range[jid])) <= 1e-7:
                active.append(model.joint(int(jid)).name)
        diagnostics["active_rom_limits"] = active
        return finish("UNKNOWN_COLLISION_BLOCKED_SEARCH" if solution.collisions else
                      "ROM_LIMITED_SEARCH" if active else "LOCAL_SEARCH_UNRESOLVED", solution.reason)

    try:
        for index in range(count + 1):
            time = index * policy.sample_interval_s
            blend, _ = minimum_jerk(time, policy.prepare_s)
            solution = solve_whole_body_reference(
                model, scene, profile, seed, intent,
                root_target=reach_frame(root, motion.root_target, time, policy.prepare_s, 0.),
                waist_target={n: (1. - blend) * waist[n] + blend * motion.waist_target[n] for n in waist},
                support_frames=frames)
            diagnostics["preparation_results"].append(solution)
            if not solution.converged:
                return search_failure(solution)
            seed = np.asarray(solution.qpos)
            four = record_support(seed, intent, "prepare_four", index)
            three = record_support(seed, supports, "prepare_three", index)
            initialize_static_reference(model, mujoco.MjData(model), replace(scene, start_configuration=intent),
                                        profile, solution.qpos, intent)
            path.append(solution.qpos)
            if initial_four is None:
                initial_four = four
            if "proposed_torques_Nm" in initial_four and "proposed_torques_Nm" in three:
                torques = {n: (1. - blend) * initial_four["proposed_torques_Nm"][n]
                           + blend * three["proposed_torques_Nm"][n] for n in three["proposed_torques_Nm"]}
                motor_margin = min(ceiling - abs(torques[model.joint(int(j)).name])
                                   for j, ceiling in zip(model.actuator_trnid[:, 0], diagnostics["motor_ceiling_Nm"]))
                diagnostics.setdefault("load_plan", []).append({
                    "index": index, "blend": blend, "moving_hand_nominal_weight_share": .05 * (1. - blend),
                    "torques_Nm": torques, "motor_margin_min_Nm": float(motor_margin),
                    "scope": "executor's fixed source four-contact FF blended with candidate three-contact FF"})
        samples = tuple(_declared_hand_path(scene, request, target, policy))
        certificate = _path_collision_certificate(model, source.qpos, request, samples)
        diagnostics["path_collision_certificate"] = certificate
        if certificate is not None:
            geometric = False
            return finish("COLLISION_INFEASIBLE", "Specified sampled effector path requires a hand END point inside both "
                          "the moving hand and static obstacle " + certificate["static_geom"] +
                          "; this is a path-specific geometric certificate, not global transfer infeasibility")
        for phase, index, time, frame in samples:
            solution = solve_whole_body_reference(
                model, scene, profile, seed, {**intent, request.limb: request.target},
                root_target=motion.root_target, waist_target=motion.waist_target, support_frames=frames,
                hand_targets={request.limb: frame}, allowed_hand_holds={request.limb: (request.source, request.target)})
            diagnostics["reach_results"].append({"phase": phase, "index": index, "time_s": time, "result": solution})
            endpoint = solution
            if not solution.converged:
                return search_failure(solution)
            seed = np.asarray(solution.qpos)
            record_support(seed, supports, phase, index)
        geometric = True
        final_intent = {**intent, request.limb: request.target}
        record_support(endpoint.qpos, final_intent, "endpoint_four", 0)
        admitted, _ = initialize_static_reference(model, mujoco.MjData(model),
                                                  replace(scene, start_configuration=final_intent),
                                                  profile, endpoint.qpos, final_intent)
        diagnostics.update(endpoint_admitted=True, endpoint_reference=admitted)
    except ValueError as error:
        diagnostics.update(admission_error=str(error), failed_qpos=tuple(seed))
        witness = mujoco.MjData(model)
        witness.qpos[:] = diagnostics["preparation_results"][-1].qpos if not geometric and not endpoint else seed
        witness.eq_active[:] = False
        mujoco.mj_forward(model, witness)
        penetrations = tuple((model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name, float(c.dist))
                             for c in witness.contact if c.dist <= -.001)
        diagnostics["admission_penetrations"] = penetrations
        if penetrations:
            return finish("COLLISION_INFEASIBLE", "This candidate has a measured penetration/admission witness: " + str(error))
        if any(word in str(error).lower() for word in ("capacity", "reaction", "capability", "friction")):
            support = False
            return finish("SUPPORT_INFEASIBLE", "This candidate failed bounded scratch admission: " + str(error))
        return finish("LOCAL_SEARCH_UNRESOLVED", "Exact scratch admission failed for this candidate: " + str(error))
    support = not support_failures
    diagnostics["support_failures"] = support_failures
    if not support:
        return finish("SUPPORT_INFEASIBLE", "This candidate violates maintained support estimates: " + support_failures[0])
    return finish("GEOMETRICALLY_FEASIBLE", "Bounded sampled candidate admitted with static support estimates; dynamics, capture and readiness untested", True)
