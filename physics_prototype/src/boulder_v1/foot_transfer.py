"""Maintained native foot release/landing; reference forces are never applied."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_geometry import ContactMode, Frame, canonical_geometry
from .contact_ik import solve_contact_pose, solve_foot_reference
from .hand_family import make_hand_family_fixture
from .locomotion import get_state_summary, _supported_hold_geometry
from .mjcf_builder import build_mjcf
from .motion_support import estimate_support_torques
from .runtime import compute_pose_control, validate_reference_pose
from .schema import Affordance, Limb, SourceType
from .single_hand import _shape_reference, minimum_jerk, reach_frame
from .static_state import (ReadinessEvidence, ReadinessTracker, _finite_state,
                           _canonical_surface, _foot_residual, _integration_state,
                           _unexpected_loaded_contacts, initialize_static_reference)
from .support import FORCE_TOLERANCE, MAX_SLIP_SPEED, MIN_FOOT_LOAD, fresh_data


class FootTransferStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INITIALIZATION_FAILURE = "INITIALIZATION_FAILURE"
    INELIGIBLE_TARGET = "INELIGIBLE_TARGET"
    REACH_INFEASIBLE = "REACH_INFEASIBLE"
    SUPPORT_INFEASIBLE = "SUPPORT_INFEASIBLE"
    ROM_FAILURE = "ROM_FAILURE"
    CONTACT_LOSS = "CONTACT_LOSS"
    GRIP_FAILURE = "GRIP_FAILURE"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    LANDING_FAILURE = "LANDING_FAILURE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class FootRequest:
    limb: Limb
    source: str
    target: str
    unload_s: float = 2.
    lift_s: float = 1.5
    support_s: float = .5
    reach_s: float = 3.
    contact_timeout_s: float = 3.
    load_s: float = 2.
    settle_timeout_s: float = 5.
    lift_m: float = .03
    clearance_m: float = .02
    airborne_pitch_rad: float = 0.


def make_foot_transfer_fixture(timestep=.002, *, profile=None, target_friction=.9):
    """Unchanged Stage4 body/source stance plus a separate canonical STEP box.

    The initial shoe's x minimum is -0.1675m; the target's right edge is
    -0.1725m, so it cannot secretly support the starting shoe.
    """
    _, _, scene, profile, seed = make_hand_family_fixture(timestep, profile=profile)
    source = scene.region("left_foot")
    target = replace(source, id="foot_target", friction=target_friction,
                     position=(source.position[0] - .105, source.position[1], source.position[2] + .02))
    scene = replace(scene, contact_regions=(*scene.contact_regions, target))
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    return model, mujoco.MjData(model), scene, profile, seed


def execute_foot_transfer(model, data, scene, profile, reference, manager, request, *,
                          observer=None, keep_samples=True, fault=None, whole_body=None):
    """Continue the bound episode with six-leg-hinge IK and default impedance.

    Source admission and candidate support allocation are scratch-only. Native
    contact, not an allocation/cache/equality, owns release and acquisition.
    Optional whole-body preparation changes scratch hinge references only.
    A declared support_loss fault is the sole optional external-force channel.
    Observers receive independent full copies; their exceptions propagate.
    """
    manager.require_session(model, data, scene)
    initial = get_state_summary(model, data, manager)
    start, dt = float(data.time), float(model.opt.timestep)
    phase, status, reason = "PREFLIGHT", FootTransferStatus.INITIALIZATION_FAILURE, "Execution not complete"
    steps, samples, phases, events = 0, [], [], []
    intent, supports, new_contacts = {}, {}, {}
    request_summary, preflight, support_admission, final_reference = None, None, None, None
    command, qref, desired_frame, tracker = None, None, None, None
    qdref = np.zeros(model.nv)
    source_admitted, released = False, False
    release, touchdown, acquisition, first_support = None, None, None, None
    load_start, load_complete, support_since = None, None, None
    guard_failure, observer_error, owned_force = None, None, None
    source_changes, last_source_state = [], None
    landing_point = None
    cache_uses = 0
    hand_decisions = {}
    body_path = None
    joints = model.actuator_trnid[:, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    names = [model.joint(int(j)).name for j in joints]
    warnings = tuple(int(w.number) for w in data.warning)
    release_count = len(manager.releases)

    def observe(row):
        nonlocal observer_error
        if observer is not None:
            display_model = copy.copy(model)
            display_data = mujoco.MjData(display_model)
            mujoco.mj_copyData(display_data, display_model, data)
            before = _integration_state(model, data)
            try:
                observer(copy.deepcopy(row), display_model, display_data)
            except Exception as error:
                observer_error = error
                raise
            if not np.array_equal(before, _integration_state(model, data), equal_nan=True):
                if row["terminal"]:
                    observer_error = RuntimeError("Observer changed the authoritative live integration state")
                    raise observer_error
                audit_forces()
                raise RuntimeError("Live state changed outside the authoritative integration owner")

    def source_state(foot):
        contacts = [c for c in foot.contacts if c.surface_geom == "geom_" + request.source]
        return {"contact_count": len(contacts), "normal_force_N": sum(c.normal_force for c in contacts),
                "supporting": _supported_hold_geometry(foot, request.source)}

    def row(terminal=False):
        snapshot = manager.contact_snapshot()
        actual, distance, measurement_error, source_contact = None, None, None, None
        if isinstance(request, FootRequest) and isinstance(request.limb, Limb) and request.limb.is_foot:
            try:
                scratch = fresh_data(model, data)
                actual = scratch.site(f"{request.limb.value.lower()}_site").xpos.tolist()
                distance = float(mujoco.mj_geomDistance(model, scratch,
                                 model.geom(f"{request.limb.value.lower()}_geom").id,
                                 model.geom("geom_" + request.target).id, 1., None))
                source_contact = source_state(snapshot.feet[request.limb])
            except (ValueError, KeyError, TypeError) as error:
                measurement_error = str(error)
        control = asdict(command) if command is not None else None
        if terminal and control is not None:
            torque = data.ctrl * model.actuator_gear[:, 0]
            control.update(commanded_Nm=tuple(float(v) for v in torque),
                           utilization=tuple(float(v) for v in np.abs(torque) / command.limits_Nm),
                           matches_last_request=bool(np.array_equal(data.ctrl,
                               np.array(command.commanded_Nm) / model.actuator_gear[:, 0])))
        evidence = asdict(tracker.evidence if tracker is not None else ReadinessEvidence(reason=reason))
        linear, angular = float(np.linalg.norm(data.qvel[:3])), float(np.linalg.norm(data.qvel[3:6]))
        maximum = float(np.max(np.abs(data.qvel[vi]), initial=0.))
        if terminal:
            evidence.update(time=float(data.time), root_linear_speed=linear, root_angular_speed=angular,
                            max_hinge_speed=maximum, rms_hinge_speed=float(np.sqrt(np.mean(data.qvel[vi] ** 2))))
            if status != FootTransferStatus.SUCCESS:
                evidence.update(ready=False, duration=0., reason=reason)
        return {"phase": phase, "time_s": float(data.time), "elapsed_s": float(data.time - start),
                "steps": steps, "integrated_elapsed_s": steps * dt,
                "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                "q_ref": qref.tolist() if qref is not None else None,
                "qd_ref": qdref.tolist() if qref is not None else None,
                "source_reference_admitted": source_admitted, "root_pose": data.qpos[:7].tolist(),
                "root_linear_m_s": linear, "root_angular_rad_s": angular, "joint_max_rad_s": maximum,
                "ctrl": data.ctrl.tolist(), "command": control, "eq_active": data.eq_active.tolist(),
                "hands": {l.value: asdict(s) for l, s in snapshot.hands.items()},
                "feet": {l.value: asdict(s) for l, s in snapshot.feet.items()},
                "contacts": {l.value: h for l, h in snapshot.configuration.items()},
                "declared_support_contacts": {l.value: h for l, h in
                    (intent if phase in ("PREFLIGHT", "SOURCE_STABILIZE") else
                     new_contacts if acquisition is not None else supports).items()},
                "hand_decisions": copy.deepcopy(hand_decisions),
                "actual_foot_position_world_m": actual, "foot_distance_m": distance,
                "foot_position_measurement_error": measurement_error, "source_contact": source_contact,
                "tracking_error_m": float(np.linalg.norm(np.array(desired_frame.position) - actual))
                                    if desired_frame is not None and actual is not None else None,
                "desired_foot_frame": asdict(desired_frame) if desired_frame else None,
                "released": released, "touchdown": copy.deepcopy(touchdown),
                "first_support": copy.deepcopy(first_support), "acquisition": copy.deepcopy(acquisition),
                "readiness": evidence, "pose_available": _finite_state(model, data) and measurement_error is None,
                "external_force_world_N": data.xfrc_applied.tolist(), "qfrc_applied": data.qfrc_applied.tolist(),
                "terminal": terminal, "status": status.value if terminal else "RUNNING",
                "reason": reason if terminal else "Actual native foot execution"}

    def finish():
        terminal = row(terminal=True)
        final = get_state_summary(model, data, manager)
        observe(terminal)
        three = [s for s in samples if s["phase"] in ("THREE_POINT", "REACH", "LANDING")]
        result = {"success": status == FootTransferStatus.SUCCESS, "status": status.value, "reason": reason,
                  "request": request_summary, "initial_state": initial, "final_state": final,
                  "terminal_observation": terminal, "initial_contacts": intent,
                  "new_contacts": manager.contact_configuration(), "steps": steps, "duration_s": steps * dt,
                  "dt_s": dt, "phases": phases, "events": events, "released": released, "release": release,
                  "release_time_s": release["time_s"] if release else None, "touchdown": touchdown,
                  "first_support": first_support, "acquisition": acquisition,
                  "acquisition_time_s": acquisition["time_s"] if acquisition else None,
                  "load_start_time_s": load_start, "load_complete_time_s": load_complete,
                  "load_duration_s": load_complete - load_start if load_complete is not None else None,
                  "source_contact_changes": source_changes, "preflight": preflight,
                  "landing_point_cache_uses": cache_uses,
                  "support_admission": support_admission, "final_reference": final_reference,
                  "guard_failure": guard_failure, "readiness": terminal["readiness"], "fault": fault,
                  "release_events": copy.deepcopy(manager.releases),
                  "grip_releases": copy.deepcopy(manager.releases[release_count:]),
                  "three_point": {"duration_s": len(three) * dt,
                      "root_linear_max_m_s": max((s["root_linear_m_s"] for s in three), default=0.),
                      "root_angular_max_rad_s": max((s["root_angular_rad_s"] for s in three), default=0.),
                      "joint_max_rad_s": max((s["joint_max_rad_s"] for s in three), default=0.)},
                  "hand_load_max_N": {l.value: max((max(s["hands"][l.value]["load"],
                                          s["hand_decisions"].get(l.value, {}).get("required_load", 0.))
                                          for s in samples), default=0.)
                                      for l in (Limb.LEFT_HAND, Limb.RIGHT_HAND)},
                  "foot_slip_max_m_s": {l.value: max((s["feet"][l.value]["tangential_speed"] for s in samples), default=0.)
                                        for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)},
                  "actuator_utilization_max": max((max(s["command"]["utilization"]) for s in samples), default=0.),
                  "max_tracking_error_m": max((s["tracking_error_m"] for s in three
                                               if s["tracking_error_m"] is not None), default=None)}
        if keep_samples:
            result["samples"] = samples
        if whole_body is not None and body_path is not None:
            result["whole_body"] = {"root_target": asdict(whole_body.root_target),
                                    "waist_target": dict(whole_body.waist_target),
                                    "prepare_s": whole_body.prepare_s,
                                    "preparation_samples": len(body_path)}
        return result

    def enter(value):
        nonlocal phase
        phase = value
        phases.append(value)
        events.append({"phase": value, "time_s": float(data.time), "step": steps})

    def event(value, **details):
        item = {"event": value, "time_s": float(data.time), "step": steps, **details}
        events.append(item)
        return item.copy()

    def audit_forces():
        external = data.xfrc_applied.copy()
        if owned_force is not None:
            body, original, declared = owned_force
            if not np.array_equal(external[body, :3], declared):
                raise ValueError("Declared fault force channel was changed")
            external[body, :3] = original
        if np.any(data.qfrc_applied) or np.any(external):
            raise ValueError("Foot execution forbids undeclared generalized forces or body wrenches")

    try:
        audit_forces()
        if (not math.isfinite(dt) or dt <= 0 or manager.mode != ContactMode.PHYSICAL
                or manager.profile is not profile or not _finite_state(model, data)):
            raise ValueError("Invalid bound physical profile or full live state")
        seed = np.asarray(reference.qpos, dtype=float).copy()
        if seed.shape != (model.nq,) or not np.isfinite(seed).all():
            raise ValueError("Reference requires a complete finite model.nq pose")
        try:
            validate_reference_pose(model, seed)
        except ValueError:
            status = FootTransferStatus.ROM_FAILURE
            raise
        proposed = reference.contact_intent
        if (not isinstance(proposed, Mapping) or set(proposed) != set(Limb)
                or any(not isinstance(l, Limb) for l in proposed)
                or any(not isinstance(h, str) or not h.strip() for h in proposed.values())):
            raise ValueError("Reference requires all four actual Limb keys and nonempty HOLD strings")
        if (not isinstance(reference.target_pose, Mapping) or set(reference.target_pose) != set(names)
                or any(isinstance(reference.target_pose[n], bool) or not isinstance(reference.target_pose[n], Real)
                       or not math.isfinite(reference.target_pose[n]) or float(reference.target_pose[n]) != float(seed[q])
                       for n, q in zip(names, qi))):
            raise ValueError("Reference target_pose must exactly match compiled hinge coordinates")
        if manager.contact_configuration() != dict(proposed):
            raise ValueError("Source reference intent differs from actual initialized contacts")
        initialize_static_reference(model, mujoco.MjData(model), replace(scene, start_configuration=dict(proposed)),
                                    profile, seed, proposed)
        tracker = ReadinessTracker(model, data, manager)
        intent, qref, source_admitted = dict(proposed), seed, True
    except Exception as error:
        reason = str(error)
        return finish()

    try:
        if (not isinstance(request, FootRequest) or not isinstance(request.limb, Limb) or not request.limb.is_foot
                or any(not isinstance(v, str) or not v.strip() for v in (request.source, request.target))
                or request.source == request.target or intent[request.limb] != request.source):
            raise ValueError("Invalid moving foot or source/target intent")
        if fault is not None and (not isinstance(fault, str) or fault != "support_loss"):
            raise ValueError("Unknown declared foot fault")
        durations = (request.unload_s, request.lift_s, request.support_s, request.reach_s,
                     request.contact_timeout_s, request.load_s, request.settle_timeout_s)
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
               or not math.isfinite(v / dt) or abs(v / dt - round(v / dt)) > 1e-8 for v in durations):
            raise ValueError("Durations must be positive integral native intervals")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
                for v in (request.lift_m, request.clearance_m)):
            raise ValueError("Invalid lift/clearance distance")
        if (isinstance(request.airborne_pitch_rad, bool) or not isinstance(request.airborne_pitch_rad, Real)
                or not math.isfinite(request.airborne_pitch_rad) or abs(request.airborne_pitch_rad) > math.pi / 6):
            raise ValueError("Airborne shoe pitch must be finite and bounded by 30 degrees")
        request_summary = asdict(request)
        target = scene.region(request.target)
        if target.source_type != SourceType.HOLD or Affordance.STEP not in target.affordances:
            status = FootTransferStatus.INELIGIBLE_TARGET
            raise ValueError("Foot target must be an eligible STEP HOLD")
        geometry = _canonical_surface(model, fresh_data(model, data), target)
        frame = geometry.foot_frame
        if whole_body is None:
            endpoint = solve_foot_reference(model, data.qpos, request.limb, frame)
        else:
            from .whole_body_motion import prepare_whole_body
            _, body_path, _ = prepare_whole_body(model, scene, profile, reference, whole_body)
            endpoint = solve_foot_reference(model, body_path[-1], request.limb, frame)
        preflight = asdict(endpoint)
        if not endpoint.converged:
            status, reason = FootTransferStatus.REACH_INFEASIBLE, endpoint.reason
            return finish()
        side = request.limb.value.lower().removesuffix("_foot")
        leg_q = model.jnt_qposadr[[model.joint(f"{side}_{n}").id for n in
                                 ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]]
        opposite = Limb.RIGHT_FOOT if request.limb == Limb.LEFT_FOOT else Limb.LEFT_FOOT
        supports = {l: h for l, h in intent.items() if l != request.limb}
        new_contacts = {**intent, request.limb: request.target}
    except Exception as error:
        if error is observer_error:
            raise
        if status not in (FootTransferStatus.INELIGIBLE_TARGET, FootTransferStatus.ROM_FAILURE):
            status = FootTransferStatus.REACH_INFEASIBLE
        reason = str(error)
        return finish()

    def allocation(contacts):
        nonlocal landing_point, cache_uses
        snapshot = manager.contact_snapshot()
        points = {}
        for limb, hold in contacts.items():
            if not limb.is_foot:
                continue
            foot = snapshot.feet[limb]
            actual = [c for c in foot.contacts if c.surface_geom == "geom_" + hold and c.admissible]
            if actual:
                points[limb] = np.mean([c.point for c in actual], axis=0)
                if limb == request.limb and hold == request.target and landing_point is None:
                    landing_point = points[limb].copy()
            elif (limb == request.limb and hold == request.target and touchdown is not None
                  and first_support is None and landing_point is not None):
                # A lightly loaded single corner may vanish for one solve. This
                # is an estimated torque point, NEVER evidence of actual support.
                points[limb] = landing_point
                cache_uses += 1
            else:
                raise ValueError(f"{limb.value}: no own admissible native force point for {hold}")
        return estimate_support_torques(model, scene, profile, data.qpos, contacts, foot_points=points)

    def audit_support(epoch, measured, snapshot):
        nonlocal guard_failure
        allowed_contacts = dict(supports)
        if phase == "SOURCE_STABILIZE" or phase in ("UNLOAD", "LIFT") and not released:
            allowed_contacts[request.limb] = request.source
        elif phase in ("LANDING", "LOAD", "SETTLE"):
            allowed_contacts[request.limb] = request.target
        allowed = {frozenset((f"{l.value.lower()}_geom", "geom_" + h)) for l, h in allowed_contacts.items()}
        unintended = _unexpected_loaded_contacts(model, measured, allowed)
        feet = manager.foot_sensor.measure(model, measured, scene, fresh=False)
        bad = unintended or any(not f.measurement_valid for f in feet.values())
        bad = bad or not _supported_hold_geometry(feet[opposite], intent[opposite]) or feet[opposite].slipping
        moving = feet[request.limb]
        source_contact = source_state(moving)
        bad = bad or released and source_contact["contact_count"] != 0
        loaded = [c for c in moving.contacts if c.normal_force > FORCE_TOLERANCE]
        bad = bad or any(not c.admissible or c.tangential_speed > MAX_SLIP_SPEED for c in loaded)
        if phase == "SOURCE_STABILIZE":
            bad = bad or not _supported_hold_geometry(moving, request.source) or moving.slipping
        elif phase in ("THREE_POINT", "REACH"):
            bad = bad or moving.contacting or moving.supporting
        elif phase in ("LIFT", "UNLOAD") and released:
            bad = bad or moving.contacting or moving.supporting
        if acquisition is not None:
            bad = bad or not _supported_hold_geometry(moving, request.target) or moving.primary_surface != "geom_" + request.target
        bad = bad or any(not snapshot.hands[l].active or not snapshot.hands[l].valid
                        or snapshot.hands[l].region_id != intent[l] for l in (Limb.LEFT_HAND, Limb.RIGHT_HAND))
        if bad:
            guard_failure = {"force_epoch": epoch, "unintended_contacts": unintended,
                             "source_contact": source_contact,
                             "feet": {l.value: asdict(f) for l, f in feet.items()},
                             "hands": {l.value: asdict(h) for l, h in snapshot.hands.items()}}
            raise ValueError("Native support/slip/identity or unintended loaded body contact failed")

    support_admission = {"admitted": False, "contacts": dict(supports),
                         "scope": "candidate reference estimate and native source; not global physical infeasibility"}
    try:
        snapshot = manager.contact_snapshot()
        points = {l: np.mean([c.point for c in snapshot.feet[l].contacts
                             if c.surface_geom == "geom_" + intent[l] and c.admissible], axis=0)
                  for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)}
        support_admission["native_feet"] = {l.value: asdict(f) for l, f in snapshot.feet.items()}
        support_admission["native_hands"] = {l.value: asdict(h) for l, h in snapshot.hands.items()}
        phase = "SOURCE_STABILIZE"
        audit_support("preflight_applied", data, snapshot)
        audit_support("preflight_endpoint", fresh_data(model, data), snapshot)
        phase = "PREFLIGHT"
        remaining_point = {opposite: points[opposite]}
        estimate = estimate_support_torques(model, scene, profile, data.qpos, supports, foot_points=remaining_point)
        candidate = mujoco.MjData(model)
        candidate.qpos[:] = endpoint.qpos
        candidate.eq_active[:] = False
        mujoco.mj_forward(model, candidate)
        target_point = _foot_residual(model, candidate, request.limb, target, geometry).support_point_world
        candidate_points = {opposite: points[opposite], request.limb: target_point}
        endpoint_estimate = estimate_support_torques(model, scene, profile, endpoint.qpos, supports, foot_points=remaining_point)
        landing_estimate = estimate_support_torques(model, scene, profile, endpoint.qpos, new_contacts,
                                                   foot_points=candidate_points)
        if body_path is not None:
            # Four-contact admission applies only to preparation, never the lifted leg.
            preparation_estimates = []
            for pose in body_path:
                preparation_estimates.append(estimate_support_torques(
                    model, scene, profile, pose, intent, foot_points=points))
                preparation_estimates.append(estimate_support_torques(
                    model, scene, profile, pose, supports, foot_points=remaining_point))
        support_admission.update(admitted=True, estimate=estimate, endpoint_estimate=endpoint_estimate,
                                 landing_estimate=landing_estimate,
                                 root_balance_residual=estimate["root_balance_residual"])
        ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        support_admission["motor_utilization_max"] = max(float(np.max(np.abs(
            [e["torques_Nm"][n] for n in names]) / ceiling)) for e in (estimate, endpoint_estimate, landing_estimate))
        if body_path is not None:
            support_admission["preparation_motor_utilization_max"] = max(float(np.max(np.abs(
                [e["torques_Nm"][n] for n in names]) / ceiling)) for e in preparation_estimates)
    except Exception as error:
        phase = "PREFLIGHT"
        status, reason = FootTransferStatus.SUPPORT_INFEASIBLE, str(error)
        support_admission["reason"] = reason
        return finish()

    expected_state = _integration_state(model, data)
    session_token = tracker._session_token()

    def step(goal, ff):
        nonlocal steps, command, qref, qdref, status, reason, released, release
        nonlocal last_source_state, first_support, acquisition, support_since, expected_state, hand_decisions, guard_failure
        before = float(data.time)
        manager.require_session(model, data, scene)
        audit_forces()
        if (manager.profile is not profile or tracker._session_token() != session_token
                or float(model.opt.timestep) != dt or tuple(int(w.number) for w in data.warning) != warnings
                or not np.array_equal(_integration_state(model, data), expected_state)):
            status = FootTransferStatus.CONTROL_FAILURE
            raise ValueError("Live state/session changed outside the authoritative native step")
        audit_support("pre_command_endpoint", fresh_data(model, data), manager.contact_snapshot())
        shaped, velocity = _shape_reference(model, goal, qref, qdref, dt)
        command = compute_pose_control(model, data, dict(zip(names, shaped[qi])),
                                       target_velocity=dict(zip(names, velocity[vi])), feedforward=ff)
        qref, qdref = shaped.copy(), velocity.copy()
        decisions = manager.evaluate_and_update()
        hand_decisions = {l.value: asdict(d) for l, d in decisions.items()}
        if any(not d.maintain for d in decisions.values()):
            status = FootTransferStatus.GRIP_FAILURE
            guard_failure = {"force_epoch": "pre_command_solve", "decisions": copy.deepcopy(hand_decisions)}
            raise ValueError("Bounded grasp failed before native integration")
        audit_forces()
        mujoco.mj_step(model, data)
        steps += 1
        if abs(float(data.time) - before - dt) > 1e-12 or tuple(int(w.number) for w in data.warning) != warnings:
            manager.synchronize_from_live()
            raise ValueError("Numerical recovery or discontinuous clock")
        audit_forces()
        decisions = manager.evaluate_and_update(applied_data=data)
        hand_decisions = {l.value: asdict(d) for l, d in decisions.items()}
        if any(not d.maintain for d in decisions.values()):
            status = FootTransferStatus.GRIP_FAILURE
            guard_failure = {"force_epoch": "applied_and_endpoint_solve", "decisions": copy.deepcopy(hand_decisions)}
            raise ValueError("Bounded grasp failed in applied/endpoint solve")
        snapshot = manager.contact_snapshot()
        audit_support("applied_preintegration_solve", data, snapshot)
        audit_support("fresh_endpoint_solve", fresh_data(model, data), snapshot)
        if not _finite_state(model, data) or (np.linalg.norm(data.qvel[:3]) > .10
                or np.linalg.norm(data.qvel[3:6]) > .50 or np.max(np.abs(data.qvel[vi])) > 1.):
            raise ValueError("Explicit transition motion safety bounds failed")
        moving = snapshot.feet[request.limb]
        current = source_state(moving)
        signature = (current["contact_count"], current["supporting"])
        if signature != last_source_state:
            source_changes.append({"time_s": float(data.time), "phase": phase, **current})
            last_source_state = signature
        if phase == "LIFT" and not released and not moving.contacts and moving.normal_force == 0.:
            released = True
            scratch = fresh_data(model, data)
            distance = float(mujoco.mj_geomDistance(model, scratch, model.geom(f"{request.limb.value.lower()}_geom").id,
                                                  model.geom("geom_" + request.source).id, 1., None))
            release = event("SEPARATION", limb=request.limb.value, source=request.source,
                            signed_source_geom_distance_m=distance, **current)
        valid_target = (_supported_hold_geometry(moving, request.target) and not moving.slipping
                        and moving.primary_surface == "geom_" + request.target and current["contact_count"] == 0)
        if phase == "LOAD" and valid_target:
            if first_support is None:
                first_support = event("NATIVE_SUPPORT", normal_force_N=moving.normal_force,
                                      foot=asdict(moving))
            if support_since is None:
                support_since = float(data.time)
            if acquisition is None and data.time - support_since >= .1 - 1e-12:
                acquisition = event("ACQUIRED", first_support_time_s=first_support["time_s"],
                                    sustained_s=float(data.time - support_since), normal_force_N=moving.normal_force,
                                    foot=asdict(moving))
        elif phase == "LOAD":
            support_since = None
        tracker.sample_after_step(before)
        sample = row()
        samples.append(sample)
        expected_state = _integration_state(model, data)
        if steps % max(1, round(.05 / dt)) == 0:
            observe(sample)
        return moving

    def leg_goal(frame):
        nonlocal status
        solution = solve_foot_reference(model, data.qpos, request.limb, frame)
        if not solution.converged:
            status = FootTransferStatus.REACH_INFEASIBLE
            raise ValueError("Leg-only local reference failed: " + solution.reason)
        goal = qref.copy()
        goal[leg_q] = np.array(solution.qpos)[leg_q]
        return goal

    status = FootTransferStatus.CONTROL_FAILURE
    try:
        enter("SOURCE_STABILIZE")
        for _ in range(round(2. / dt)):
            step(seed, allocation(intent)["torques_Nm"])
            if tracker.evidence.ready:
                event("SOURCE_READY")
                break
        if not tracker.evidence.ready:
            status = FootTransferStatus.INITIALIZATION_FAILURE
            raise ValueError("Source did not reach sustained Stage3 readiness")
        enter("UNLOAD")
        ff4 = allocation(intent)["torques_Nm"]
        preparation_s = request.unload_s if whole_body is None else whole_body.prepare_s
        for i in range(round(preparation_s / dt)):
            b, _ = minimum_jerk((i + 1) * dt, preparation_s)
            ff3 = allocation(supports)["torques_Nm"]
            goal = seed
            if whole_body is not None:
                from .whole_body_motion import preparation_reference
                goal = preparation_reference(model, body_path, (i + 1) * dt, preparation_s)
            step(goal, {n: (1 - b) * ff4[n] + b * ff3[n] for n in names})
        scratch = fresh_data(model, data)
        site = scratch.site(f"{request.limb.value.lower()}_site")
        begin = Frame(tuple(site.xpos), tuple(map(tuple, scratch.geom(f"{request.limb.value.lower()}_geom").xmat.reshape(3, 3))))
        clear = Frame(tuple(np.array(begin.position) + request.lift_m * np.array(frame.normal)), begin.rotation)
        if request.airborne_pitch_rad:
            # A raised flight can need toe-down ankle room. The target landing
            # frame and every native loaded-foot requirement remain unchanged.
            c, s = math.cos(request.airborne_pitch_rad), math.sin(request.airborne_pitch_rad)
            rotation = np.array(begin.rotation) @ np.array(((1., 0., 0.), (0., c, -s), (0., s, c)))
            clear = Frame(clear.position, tuple(map(tuple, rotation)))
        enter("LIFT")
        for i in range(round(request.lift_s / dt)):
            desired_frame = reach_frame(begin, clear, (i + 1) * dt, request.lift_s, 0.)
            step(leg_goal(desired_frame), allocation(supports)["torques_Nm"])
        if not released or manager.contact_snapshot().feet[request.limb].contacting:
            status = FootTransferStatus.CONTACT_LOSS
            raise ValueError("Moving foot did not physically separate by lift completion")
        enter("THREE_POINT")
        support_ref = qref.copy()
        for _ in range(round(request.support_s / dt)):
            step(support_ref, allocation(supports)["torques_Nm"])
        scratch = fresh_data(model, data)
        site = scratch.site(f"{request.limb.value.lower()}_site")
        begin = Frame(tuple(site.xpos), tuple(map(tuple, scratch.geom(f"{request.limb.value.lower()}_geom").xmat.reshape(3, 3))))
        enter("REACH")
        for i in range(round((request.reach_s + request.contact_timeout_s) / dt)):
            if phase == "REACH" and i * dt >= request.reach_s - .25 - 1e-12:
                enter("LANDING")
            desired_frame = reach_frame(begin, frame, i * dt, request.reach_s, request.clearance_m)
            if fault == "support_loss" and i == round(.3 / dt):
                audit_forces()
                if not np.array_equal(_integration_state(model, data), expected_state):
                    status = FootTransferStatus.CONTROL_FAILURE
                    raise ValueError("Live state changed before declared fault")
                body = model.geom(f"{opposite.value.lower()}_geom").bodyid[0]
                owned_force = (body, data.xfrc_applied[body, :3].copy(), np.array([1500., 0., 0.]))
                data.xfrc_applied[body, :3] = owned_force[2]
                expected_state = _integration_state(model, data)
            moving = step(leg_goal(desired_frame), allocation(supports)["torques_Nm"])
            touch = [c for c in moving.contacts if c.surface_geom == "geom_" + request.target
                     and c.admissible and c.normal_force > 0. and c.tangential_speed <= MAX_SLIP_SPEED]
            if phase == "LANDING" and touch:
                touchdown = event("TOUCHDOWN", normal_force_N=sum(c.normal_force for c in touch),
                                  supporting=moving.supporting, foot=asdict(moving))
                landing_point = np.mean([c.point for c in touch], axis=0)
                break
        if touchdown is None:
            status = FootTransferStatus.LANDING_FAILURE
            raise ValueError("No compressive own-target native touchdown before contact timeout")
        enter("LOAD")
        load_start = float(data.time)
        desired_frame = frame
        for i in range(round(request.load_s / dt)):
            b, _ = minimum_jerk((i + 1) * dt, request.load_s)
            ff3, ff4 = allocation(supports)["torques_Nm"], allocation(new_contacts)["torques_Nm"]
            step(leg_goal(frame), {n: (1 - b) * ff3[n] + b * ff4[n] for n in names})
        load_complete = float(data.time)
        event("LOAD_COMPLETE", duration_s=load_complete - load_start)
        if acquisition is None:
            status = FootTransferStatus.LANDING_FAILURE
            raise ValueError("Target did not sustain native support for 0.1s during full loading")
        final_pose = solve_contact_pose(model, scene, profile, data.qpos, new_contacts)
        if not final_pose.admitted:
            raise ValueError("Loaded state could not form an admitted final reference: " + final_pose.reason)
        enter("SETTLE")
        settle_start, settle_goal = qref.copy(), np.array(final_pose.reference.qpos)
        tangent = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, tangent, 1., settle_start, settle_goal)
        tracker = ReadinessTracker(model, data, manager)
        reference_stopped = False
        for i in range(round(request.settle_timeout_s / dt)):
            b, _ = minimum_jerk((i + 1) * dt, .5)
            goal = settle_start.copy()
            mujoco.mj_integratePos(model, goal, tangent, b)
            step(goal, allocation(new_contacts)["torques_Nm"])
            stopped = ((i + 1) * dt >= .5 and np.max(np.abs(qref[qi] - settle_goal[qi])) <= 1e-10
                       and np.max(np.abs(qdref[vi])) <= 1e-8)
            if stopped and not reference_stopped:
                reference_stopped = True
                tracker = ReadinessTracker(model, data, manager)
                event("REFERENCE_STOPPED")
            if stopped and tracker.evidence.ready:
                status, reason = FootTransferStatus.SUCCESS, "Separated, natively landed/loaded and sustained new Stage3 readiness"
                final_reference = final_pose.reference
                event("FINAL_READY")
                break
        else:
            status, reason = FootTransferStatus.TIMEOUT, "Post-load admitted-reference readiness timeout"
    except Exception as error:
        if error is observer_error:
            raise
        if guard_failure is not None and status != FootTransferStatus.GRIP_FAILURE:
            status = FootTransferStatus.CONTACT_LOSS
        reason = str(error)
    finally:
        if owned_force is not None:
            body, original, declared = owned_force
            force = data.xfrc_applied[body, :3]
            # Caller replacements are no longer owned by this executor.
            unchanged = force == declared
            force[unchanged] = original[unchanged]
            owned_force = None
    return finish()
