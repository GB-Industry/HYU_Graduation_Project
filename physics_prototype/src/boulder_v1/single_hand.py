"""A single measured release/reach/capture episode, not a route planner."""
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

from .contact import CAPTURE_DISTANCE, CAPTURE_ORIENTATION, effective_grip_capacity
from .contact_geometry import ContactMode, Frame, canonical_geometry
from .contact_ik import solve_contact_pose, solve_hand_reference
from .locomotion import get_state_summary, _supported_hold_geometry
from .mjcf_builder import build_mjcf
from .motion_support import estimate_support_torques
from .runtime import compute_pose_control, validate_reference_pose
from .schema import Limb
from .static_control import execute_static_hold
from .static_state import ReadinessEvidence, ReadinessTracker, _finite_state, _integration_state, _unexpected_loaded_contacts, initialize_static_reference
from .support import fresh_data


_FAULTS = ("support_loss", "grip_after_capture")
_SCENARIOS = ("success", "unreachable", "orientation_invalid", *_FAULTS)
_REFERENCE_SPEED = .5
_REFERENCE_ACCELERATION = 2.


class SingleHandStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INITIALIZATION_FAILURE = "INITIALIZATION_FAILURE"
    THREE_POINT_SUPPORT_FAILURE = "THREE_POINT_SUPPORT_FAILURE"
    REACH_INFEASIBLE = "REACH_INFEASIBLE"
    ROM_FAILURE = "ROM_FAILURE"
    CONTACT_LOSS = "CONTACT_LOSS"
    CAPTURE_FAILURE = "CAPTURE_FAILURE"
    GRIP_FAILURE = "GRIP_FAILURE"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class SingleHandRequest:
    limb: Limb
    source: str
    target: str
    transfer_s: float = 1.
    support_s: float = .5
    reach_s: float = 4.
    capture_timeout_s: float = 2.
    settle_timeout_s: float = 5.
    clearance_m: float = .02
    approach_normal: tuple[float, float, float] | None = None
    acquisition_policy: str = "first_eligible"


def minimum_jerk(time, duration):
    """Position fraction and its per-second derivative, with zero end rates."""
    if not math.isfinite(time) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Finite time and positive trajectory duration required")
    s = min(1., max(0., time / duration))
    return s ** 3 * (10. + s * (-15. + 6. * s)), 30. * s ** 2 * (1. - s) ** 2 / duration


def reach_frame(start, goal, time, duration, clearance):
    """Minimum-jerk endpoint motion with an outward, zero-rate clearance arch."""
    b, _ = minimum_jerk(time, duration)
    s = min(1., max(0., time / duration))
    position = (1. - b) * np.array(start.position) + b * np.array(goal.position)
    position += clearance * math.sin(math.pi * s) ** 2 * np.array(goal.normal)
    q0, q1 = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(q0, np.array(start.rotation).ravel())
    mujoco.mju_mat2Quat(q1, np.array(goal.rotation).ravel())
    if q0 @ q1 < 0:
        q1 *= -1
    relative = np.empty(4)
    conjugate = q0.copy(); conjugate[1:] *= -1
    mujoco.mju_mulQuat(relative, conjugate, q1)
    velocity = np.empty(3)
    mujoco.mju_quat2Vel(velocity, relative, 1.)
    quaternion = q0.copy()
    mujoco.mju_quatIntegrate(quaternion, velocity, b)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return Frame(tuple(float(v) for v in position), tuple(tuple(float(v) for v in row) for row in rotation.reshape(3, 3)))


def _shape_reference(model, goal, previous, previous_velocity, dt):
    """Shape only motor references, with discrete braking and no live state writes."""
    validate_reference_pose(model, goal)
    joints = model.actuator_trnid[:, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    acceleration_step = _REFERENCE_ACCELERATION * dt

    def braking_speed(distance):
        # Invert the distance of successive v, v-a*dt, ... positive-rate ticks.
        ticks = np.maximum(1., np.ceil((np.sqrt(1. + 8. * distance / (acceleration_step * dt)) - 1.) / 2.))
        return (distance / dt + acceleration_step * ticks * (ticks - 1.) / 2.) / ticks

    error = goal[qi] - previous[qi]
    desired = np.sign(error) * np.minimum(_REFERENCE_SPEED, braking_speed(np.abs(error)))
    lower = np.maximum(-_REFERENCE_SPEED, previous_velocity[vi] - acceleration_step)
    upper = np.minimum(_REFERENCE_SPEED, previous_velocity[vi] + acceleration_step)
    limited = model.jnt_limited[joints].astype(bool)
    lower[limited] = np.maximum(lower[limited], -braking_speed(
        np.maximum(0., previous[qi[limited]] - model.jnt_range[joints[limited], 0])))
    upper[limited] = np.minimum(upper[limited], braking_speed(
        np.maximum(0., model.jnt_range[joints[limited], 1] - previous[qi[limited]])))
    if np.any(lower > upper + 1e-10):
        raise ValueError("Reference rate cannot respect compiled ROM and acceleration bounds")
    velocity = np.clip(desired, np.minimum(lower, upper), upper)
    candidate = goal.copy()
    candidate[qi] = previous[qi] + dt * velocity
    candidate[qi[limited]] = np.clip(candidate[qi[limited]], model.jnt_range[joints[limited], 0],
                                    model.jnt_range[joints[limited], 1])
    validate_reference_pose(model, candidate)
    qd = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qd, dt, previous, candidate)
    if (np.max(np.abs(qd[vi]), initial=0.) > _REFERENCE_SPEED + 1e-10
            or np.max(np.abs(qd[vi] - previous_velocity[vi]), initial=0.) > acceleration_step + 1e-10):
        raise ValueError("Reference velocity/acceleration bounds failed")
    return candidate, qd


def execute_single_hand(model, data, scene, profile, reference, manager, request, *, observer=None,
                        fault=None, keep_samples=True, whole_body=None):
    """Continue a ready physical episode using unchanged Stage3 torque impedance.

    Safety bounds during motion are 0.10m/s root translation, 0.50rad/s root
    rotation and 1rad/s maximum hinge speed. Native foot support/slip and all hand
    capacity/geometry bounds remain unchanged. Final readiness is Stage3's 0.5s
    sustained criterion. A declared fault is a negative-test force, not support.
    Motor references retain <=0.5rad/s speed and <=2rad/s^2 acceleration across
    phases. Observer exceptions propagate after only owned force cleanup.
    The opt-in endpoint_settle acquisition preference waits for the complete
    reference plus 0.1s and a nonincreasing measured gap/speed. It does not change
    the physical gate or permit an unattached palm to carry support.
    Opt-in whole-body preparation coordinates all motor hinge references using
    scratch root/waist goals; root is never commanded. Reach IK compensates the
    actual measured body pose. Dynamic support uses real native foot points.
    """
    manager.require_session(model, data, scene)
    initial = get_state_summary(model, data, manager)
    start_time = float(data.time)
    dt = float(model.opt.timestep)
    phase, status, reason, steps = "PREFLIGHT", SingleHandStatus.CONTROL_FAILURE, "Execution not complete", 0
    samples, phases, events = [], [], []
    captures, released, eligible = None, False, None
    tracker = None
    command = None
    last_qref = None
    last_qdref = np.zeros(model.nv)
    desired_frame = None
    preflight = None
    support_admission = None
    final_reference = None
    intent = {}
    request_summary = None
    source_admitted = False
    other = None
    joints = model.actuator_trnid[:, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    names = [model.joint(int(j)).name for j in joints]
    flags = tuple(int(w.number) for w in data.warning)
    owned_force = None
    status_failure = None
    observer_error = None
    body_frames, body_path = None, None

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

    def state_row(snapshot=None, terminal=False):
        snapshot = manager.contact_snapshot() if snapshot is None else snapshot
        try:
            metric = (manager._capture_measurement(request.limb, scene.region(request.target))
                      if isinstance(request, SingleHandRequest) and isinstance(request.limb, Limb)
                      and request.limb.is_hand and isinstance(request.target, str) and _finite_state(model, data) else None)
        except (ValueError, KeyError, AttributeError, TypeError):
            metric = None
        actual_hand = None
        hand_measurement_error = None
        if (_finite_state(model, data) and isinstance(request, SingleHandRequest)
                and isinstance(request.limb, Limb) and request.limb.is_hand):
            try:
                actual_hand = fresh_data(model, data).site(f"{request.limb.value.lower()}_site").xpos.tolist()
            except (ValueError, KeyError) as error:
                hand_measurement_error = f"{type(error).__name__}: {error}"
        control = asdict(command) if command is not None else None
        if terminal and control is not None:
            torque = data.ctrl * model.actuator_gear[:, 0]
            control.update(commanded_Nm=tuple(float(v) for v in torque),
                           utilization=tuple(float(v) for v in np.abs(torque) / command.limits_Nm),
                           matches_last_request=bool(np.array_equal(
                               data.ctrl, np.array(command.commanded_Nm) / model.actuator_gear[:, 0])))
        evidence = tracker.evidence if tracker is not None else ReadinessEvidence(reason=reason)
        row = {"phase": phase, "time_s": float(data.time), "elapsed_s": float(data.time - start_time),
               "steps": steps, "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
               "q_ref": last_qref.tolist() if last_qref is not None else None,
               "qd_ref": last_qdref.tolist() if last_qref is not None else None,
               "source_reference_admitted": source_admitted,
               "root_pose": data.qpos[:7].tolist(), "root_linear_m_s": float(np.linalg.norm(data.qvel[:3])),
               "root_angular_rad_s": float(np.linalg.norm(data.qvel[3:6])),
               "joint_max_rad_s": float(np.max(np.abs(data.qvel[vi]))),
               "ctrl": data.ctrl.tolist(), "command": control,
               "hands": {l.value: asdict(h) for l, h in snapshot.hands.items()},
               "feet": {l.value: asdict(f) for l, f in snapshot.feet.items()},
               "contacts": {l.value: v for l, v in snapshot.configuration.items()},
               "eq_active": data.eq_active.tolist(), "capture_measurement": metric,
               "actual_hand_position_world_m": actual_hand,
               "hand_position_measurement_error": hand_measurement_error,
               "tracking_error_m": float(np.linalg.norm(np.array(desired_frame.position) - actual_hand))
                                   if desired_frame is not None and actual_hand is not None else None,
               "desired_hand_frame": asdict(desired_frame) if desired_frame else None,
               "readiness": asdict(evidence), "pose_available": _finite_state(model, data) and hand_measurement_error is None,
               "terminal": terminal, "status": status.value if terminal else "RUNNING",
               "reason": reason if terminal else "Actual native single-hand execution",
               "external_force_world_N": data.xfrc_applied.tolist(), "qfrc_applied": data.qfrc_applied.tolist()}
        if terminal:
            row["readiness"].update(time=float(data.time), root_linear_speed=row["root_linear_m_s"],
                                    root_angular_speed=row["root_angular_rad_s"], max_hinge_speed=row["joint_max_rad_s"],
                                    rms_hinge_speed=float(np.sqrt(np.mean(data.qvel[vi] ** 2))))
        if terminal and status != SingleHandStatus.SUCCESS:
            row["readiness"].update(ready=False, duration=0., reason=reason)
        return row

    def finish():
        final = get_state_summary(model, data, manager)
        terminal = state_row(terminal=True)
        observe(terminal)
        reach = [s for s in samples if s["phase"] in ("REACH", "CAPTURE")]
        support_rows = [s for s in samples if s["phase"] in ("THREE_POINT", "REACH", "CAPTURE")]
        result = {"success": status == SingleHandStatus.SUCCESS, "status": status.value, "reason": reason,
                   "request": request_summary, "initial_state": initial, "final_state": final,
                  "terminal_observation": terminal, "initial_contacts": intent,
                  "new_contacts": manager.contact_configuration(), "steps": steps, "duration_s": steps * dt,
                  "dt_s": dt, "phases": phases, "events": events, "released": released,
                  "capture": captures, "first_eligible": eligible,
                   "preflight": preflight, "support_admission": support_admission,
                   "final_reference": final_reference,
                   "capture_events": [{**copy.deepcopy(event), "capture_margin_m": CAPTURE_DISTANCE - event["gap_m"]}
                                      for event in manager.capture_events],
                   "release_events": copy.deepcopy(manager.releases),
                  "three_point": {"duration_s": len(support_rows) * dt,
                                  "root_linear_max_m_s": max((s["root_linear_m_s"] for s in support_rows), default=0.),
                                  "root_angular_max_rad_s": max((s["root_angular_rad_s"] for s in support_rows), default=0.),
                                  "joint_max_rad_s": max((s["joint_max_rad_s"] for s in support_rows), default=0.)},
                  "max_reach_error_m": max((s["capture_measurement"]["gap_m"] for s in reach if s["capture_measurement"]), default=None),
                  "actuator_utilization_max": max((max(s["command"]["utilization"]) for s in samples if s["command"]), default=0.),
                  "foot_support_fraction": {l.value: float(np.mean([s["feet"][l.value]["supporting"] for s in samples]))
                                            if samples else None for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)},
                  "foot_slip_max_m_s": {l.value: max((s["feet"][l.value]["tangential_speed"] for s in samples), default=0.)
                                         for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)},
                  "remaining_hand_load_max_N": max((s["hands"][other.value]["load"] for s in samples), default=0.)
                                                if other is not None else None,
                  "readiness": terminal["readiness"], "fault": fault}
        result["guard_failure"] = status_failure
        result["max_tracking_error_m"] = max((s["tracking_error_m"] for s in reach if s["tracking_error_m"] is not None), default=None)
        result["foot_loads"] = {l.value: {"normal_min_N": min((s["feet"][l.value]["normal_force"] for s in samples), default=None),
                                         "normal_mean_N": float(np.mean([s["feet"][l.value]["normal_force"] for s in samples])) if samples else None,
                                         "tangential_max_N": max((s["feet"][l.value]["tangential_force"] for s in samples), default=None)}
                                for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)}
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

    def audit_forces():
        external = data.xfrc_applied.copy()
        if owned_force is not None:
            body, original, declared = owned_force
            if not np.array_equal(external[body, :3], declared):
                raise ValueError("Declared fault force channel was changed")
            external[body, :3] = original
        if np.any(data.qfrc_applied) or np.any(external):
            raise ValueError("Single-hand execution forbids undeclared generalized forces or body wrenches")

    status = SingleHandStatus.INITIALIZATION_FAILURE
    try:
        audit_forces()
        if (not math.isfinite(dt) or dt <= 0 or manager.mode != ContactMode.PHYSICAL
                or manager.profile != profile or not _finite_state(model, data)):
            raise ValueError("Invalid bound physical profile or full live state")
        seed = np.asarray(reference.qpos, dtype=float).copy()
        if seed.shape != (model.nq,) or not np.isfinite(seed).all():
            raise ValueError("Reference requires a complete finite model.nq pose")
        try:
            validate_reference_pose(model, seed)
        except ValueError:
            status = SingleHandStatus.ROM_FAILURE
            raise
        if (not isinstance(reference.contact_intent, Mapping)
                or any(not isinstance(limb, Limb) for limb in reference.contact_intent)
                or set(reference.contact_intent) != set(Limb)
                or any(not isinstance(hold, str) or not hold.strip() for hold in reference.contact_intent.values())):
            raise ValueError("Reference requires all four actual Limb keys and nonempty HOLD strings")
        proposed_intent = dict(reference.contact_intent)
        if (not isinstance(reference.target_pose, Mapping) or set(reference.target_pose) != set(names)
                or any(isinstance(reference.target_pose[n], bool) or not isinstance(reference.target_pose[n], Real)
                       or not math.isfinite(reference.target_pose[n]) or float(reference.target_pose[n]) != float(seed[q])
                       for n, q in zip(names, qi))):
            raise ValueError("Reference target_pose must exactly match its compiled hinge coordinates")
        if manager.contact_configuration() != proposed_intent:
            raise ValueError("Source reference intent differs from actual initialized contacts")
        initialize_static_reference(model, mujoco.MjData(model), replace(scene, start_configuration=proposed_intent),
                                    profile, seed, proposed_intent)
        intent, last_qref, source_admitted = proposed_intent, seed, True
        tracker = ReadinessTracker(model, data, manager)
    except Exception as error:
        reason = str(error)
        return finish()

    try:
        if (not isinstance(request, SingleHandRequest)
                or not isinstance(request.limb, Limb) or not request.limb.is_hand
                or any(not isinstance(v, str) or not v.strip() for v in (request.source, request.target))
                or request.source == request.target
                or intent.get(request.limb) != request.source or scene.region(request.target).id != request.target
                or not initial.finite or manager.contact_configuration() != intent):
            raise ValueError("Invalid physical source, target, profile or state")
        if fault is not None and (not isinstance(fault, str) or fault not in _FAULTS):
            raise ValueError("Unknown declared single-hand fault")
        durations = (request.transfer_s, request.support_s, request.reach_s, request.capture_timeout_s, request.settle_timeout_s)
        if (not math.isfinite(dt) or dt <= 0 or any(isinstance(v, bool) or not isinstance(v, Real)
                or not math.isfinite(v) or v <= 0 or abs(v / dt - round(v / dt)) > 1e-8 for v in durations)):
            raise ValueError("Durations must be positive integral native intervals")
        if (isinstance(request.clearance_m, bool) or not isinstance(request.clearance_m, Real)
                or not math.isfinite(request.clearance_m) or request.clearance_m < 0):
            raise ValueError("Invalid clearance distance")
        if request.acquisition_policy not in ("first_eligible", "endpoint_settle"):
            raise ValueError("Unknown hand acquisition policy")
        target = scene.region(request.target)
        manager.registered_attachment(request.limb, target)
        other = Limb.LEFT_HAND if request.limb == Limb.RIGHT_HAND else Limb.RIGHT_HAND
        frame = canonical_geometry(target).hand_frame
        if request.approach_normal is not None:
            n = np.array(request.approach_normal, dtype=float)
            if n.shape != (3,) or not np.isfinite(n).all() or math.hypot(*n) <= 1e-12:
                raise ValueError("Approach normal requires three finite nonzero components")
            n /= math.hypot(*n)
            axis = np.array(frame.rotation)[:, 0]
            axis -= n * (axis @ n)
            if np.linalg.norm(axis) <= 1e-12:
                raise ValueError("Approach normal is parallel to the target tangent axis")
            axis /= np.linalg.norm(axis)
            frame = Frame(frame.position, tuple(zip(axis, np.cross(n, axis), n)))
        request_summary = asdict(request)
        if whole_body is None:
            endpoint = solve_hand_reference(model, data.qpos, request.limb, frame)
        else:
            from .whole_body_motion import prepare_whole_body
            body_frames, body_path, endpoint = prepare_whole_body(
                model, scene, profile, reference, whole_body, hand_target=frame,
                limb=request.limb, target=request.target)
        if not endpoint.converged:
            status, reason = SingleHandStatus.REACH_INFEASIBLE, endpoint.reason
            return finish()
        scratch = mujoco.MjData(model); scratch.qpos[:] = endpoint.qpos; mujoco.mj_forward(model, scratch)
        proposed = manager._capture_measurement(request.limb, target, scratch)
        preflight = proposed
        if proposed["orientation"] < CAPTURE_ORIENTATION:
            status, reason = SingleHandStatus.CAPTURE_FAILURE, "Near target but proposed facing violates the unchanged capture gate"
            return finish()
    except Exception as error:
        if error is observer_error:
            raise
        status, reason = SingleHandStatus.REACH_INFEASIBLE, str(error)
        return finish()

    supports = {l: h for l, h in intent.items() if l != request.limb}
    support_admission = {"admitted": False, "contacts": dict(supports),
                         "scope": "reference estimate and current native support; not dynamic feasibility proof"}
    try:
        snapshot = manager.contact_snapshot()
        allowed = {frozenset((f"{l.value.lower()}_geom", f"geom_{h}")) for l, h in intent.items()}
        applied_other = _unexpected_loaded_contacts(model, data, allowed)
        endpoint_other = _unexpected_loaded_contacts(model, fresh_data(model, data), allowed)
        support_admission.update(native_hands={l.value: asdict(s) for l, s in snapshot.hands.items()},
                                 native_feet={l.value: asdict(s) for l, s in snapshot.feet.items()},
                                 finite=_finite_state(model, data), applied_unintended=applied_other,
                                 endpoint_unintended=endpoint_other)
        points = None
        if whole_body is not None:
            points = {}
            for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
                actual = [c.point for c in snapshot.feet[limb].contacts
                          if c.surface_geom == "geom_" + intent[limb] and c.admissible]
                if not actual:
                    raise ValueError("Whole-body admission requires actual native foot contacts")
                points[limb] = np.mean(actual, axis=0)
        estimate = (estimate_support_torques(model, scene, profile, data.qpos, supports)
                    if whole_body is None else estimate_support_torques(
                        model, scene, profile, data.qpos, supports, foot_points=points))
        ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        torque = np.array([estimate["torques_Nm"][n] for n in names])
        estimated_hands, estimated_feet = {}, {}
        for l, vector in zip((Limb.LEFT_FOOT, Limb.RIGHT_FOOT, other), estimate["forces_world_N"]):
            region = scene.region(supports[l])
            if l.is_hand:
                load = float(np.linalg.norm(vector))
                capacity = effective_grip_capacity(profile, region, region.normal)
                estimated_hands[l.value] = {"load_N": load, "capacity_N": capacity, "margin_N": capacity - load}
            else:
                local = np.array(canonical_geometry(region).foot_surface_frame.rotation).T @ vector
                friction = min(1.8, region.friction)
                estimated_feet[l.value] = {"normal_N": float(local[2]), "minimum_load_margin_N": float(local[2] - 5.),
                                           "friction": friction,
                                           "friction_margin_N": float(friction * local[2] - np.abs(local[:2]).sum())}
        support_admission.update(estimate=estimate, estimated_hands=estimated_hands, estimated_feet=estimated_feet,
                                 root_balance_residual=estimate["root_balance_residual"],
                                 motor_utilization_max=float(np.max(np.abs(torque) / ceiling)),
                                 motor_margin_min_Nm=float(np.min(ceiling - np.abs(torque))))
        if (not support_admission["finite"] or applied_other or endpoint_other
                or any(not _supported_hold_geometry(snapshot.feet[l], intent[l]) or snapshot.feet[l].slipping
                       for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT))
                or any(not snapshot.hands[l].active or not snapshot.hands[l].valid
                       or snapshot.hands[l].region_id != intent[l] for l in (Limb.LEFT_HAND, Limb.RIGHT_HAND))):
            raise ValueError("Current native source support is invalid/nonfinite or has unintended loaded contacts")
        support_admission["admitted"] = True
    except Exception as error:
        status, reason = SingleHandStatus.THREE_POINT_SUPPORT_FAILURE, str(error)
        support_admission["reason"] = reason
        return finish()

    def allocation(contacts):
        if whole_body is not None:
            snapshot = manager.contact_snapshot()
            points = {}
            for limb, hold in contacts.items():
                if limb.is_foot:
                    actual = [c.point for c in snapshot.feet[limb].contacts
                              if c.surface_geom == "geom_" + hold and c.admissible]
                    if not actual:
                        raise ValueError("Whole-body allocation requires actual native foot contacts")
                    points[limb] = np.mean(actual, axis=0)
            return estimate_support_torques(model, scene, profile, data.qpos, contacts,
                                            foot_points=points)["torques_Nm"]
        return estimate_support_torques(model, scene, profile, data.qpos, contacts)["torques_Nm"]

    def step(qref, ff):
        nonlocal steps, status, reason, command, last_qref, last_qdref, status_failure
        before = float(data.time)
        audit_forces()
        qref, qd = _shape_reference(model, qref, last_qref, last_qdref, dt)
        command = compute_pose_control(model, data, dict(zip(names, qref[qi])),
                                       target_velocity=dict(zip(names, qd[vi])), feedforward=ff)
        last_qref, last_qdref = qref.copy(), qd.copy()
        decisions = manager.evaluate_and_update()
        if any(not d.maintain for d in decisions.values()):
            status, reason = SingleHandStatus.GRIP_FAILURE, "Bounded grasp failed before native integration"
            return False
        audit_forces()
        mujoco.mj_step(model, data); steps += 1
        if abs(data.time - before - dt) > 1e-12 or tuple(int(w.number) for w in data.warning) != flags:
            manager.synchronize_from_live()
            status, reason = SingleHandStatus.CONTROL_FAILURE, "Numerical recovery or discontinuous clock"
            return False
        audit_forces()
        decisions = manager.evaluate_and_update(applied_data=data)
        if any(not d.maintain for d in decisions.values()):
            status, reason = SingleHandStatus.GRIP_FAILURE, "Bounded grasp failed in applied/endpoint solve"
            return False
        snapshot = manager.contact_snapshot()
        required = supports if released and captures is None else ({**intent, request.limb: request.target} if captures else intent)
        allowed = {frozenset((f"{l.value.lower()}_geom", f"geom_{h}")) for l, h in required.items()}
        applied_other = _unexpected_loaded_contacts(model, data, allowed)
        endpoint_other = _unexpected_loaded_contacts(model, fresh_data(model, data), allowed)
        if (applied_other or endpoint_other
                or any(not _supported_hold_geometry(snapshot.feet[l], intent[l]) or snapshot.feet[l].slipping
                       for l in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT))
                or any(not snapshot.hands[l].active or not snapshot.hands[l].valid
                       or snapshot.hands[l].region_id != h for l, h in required.items() if l.is_hand)):
            status, reason = SingleHandStatus.CONTACT_LOSS, "Native support/slip/identity or unintended body support failed"
            status_failure = {"applied_unintended": applied_other, "endpoint_unintended": endpoint_other,
                              "feet": {l.value: asdict(s) for l, s in snapshot.feet.items()},
                              "hands": {l.value: asdict(s) for l, s in snapshot.hands.items()}}
            return False
        if not _finite_state(model, data) or (np.linalg.norm(data.qvel[:3]) > .10
                or np.linalg.norm(data.qvel[3:6]) > .50 or np.max(np.abs(data.qvel[vi])) > 1.):
            status, reason = SingleHandStatus.CONTROL_FAILURE, "Explicit transition motion safety bounds failed"
            return False
        tracker.sample_after_step(before)
        row = state_row(snapshot)
        samples.append(row)
        if steps % max(1, round(.05 / dt)) == 0:
            observe(row)
        return True

    def run():
        nonlocal status, reason, released, captures, eligible, tracker, desired_frame, owned_force, status_failure, final_reference
        if not tracker.check()[0]:
            # A fresh owner must sample sustained native history; queries cannot inherit it.
            enter("SOURCE_STABILIZE")
        for _ in range(round(2. / dt)):
            if not step(np.array(reference.qpos), allocation(intent)):
                return
            if tracker.evidence.ready:
                events.append({"event": "SOURCE_READY", "time_s": float(data.time), "step": steps})
                break
        if not tracker.evidence.ready:
            status, reason = SingleHandStatus.INITIALIZATION_FAILURE, "Source did not reach sustained Stage3 readiness"
            return
        enter("LOAD_TRANSFER")
        ff4 = allocation(intent)
        preparation_s = request.transfer_s if whole_body is None else whole_body.prepare_s
        for i in range(round(preparation_s / dt)):
            b, _ = minimum_jerk((i + 1) * dt, preparation_s)
            ff3 = allocation(supports)
            qref = np.array(reference.qpos)
            if whole_body is not None:
                from .whole_body_motion import preparation_reference
                qref = preparation_reference(model, body_path, (i + 1) * dt, preparation_s)
            if not step(qref, {n: (1 - b) * ff4[n] + b * ff3[n] for n in names}):
                return
        # Unload physical palm collision before disabling its bounded point grasp.
        enter("RELEASE_CLEARANCE")
        side = "left" if request.limb == Limb.LEFT_HAND else "right"
        arm_joints = [model.joint(f"{side}_{n}").id for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
        arm_q = model.jnt_qposadr[arm_joints]
        source_frame = canonical_geometry(scene.region(request.source)).hand_frame
        clearance_frame = Frame(tuple(np.array(source_frame.position) + .01 * np.array(source_frame.normal)), source_frame.rotation)
        base_q = last_qref.copy()
        for i in range(round(.5 / dt)):
            goal_clear = reach_frame(source_frame, clearance_frame, (i + 1) * dt, .5, 0.)
            desired_frame = goal_clear
            solution = solve_hand_reference(model, data.qpos, request.limb, goal_clear)
            if not solution.converged:
                status, reason = SingleHandStatus.REACH_INFEASIBLE, solution.reason
                return
            qref = base_q.copy(); qref[arm_q] = np.array(solution.qpos)[arm_q]
            if not step(qref, allocation(supports)):
                return
        enter("RELEASE")
        if not manager.detach(request.limb):
            status, reason = SingleHandStatus.CONTACT_LOSS, "Requested source grasp was not released"
            return
        released = True
        events.append({"event": "RELEASED", "limb": request.limb.value, "source": request.source,
                       "time_s": float(data.time), "step": steps})
        enter("THREE_POINT")
        support_ref = last_qref.copy()
        for _ in range(round(request.support_s / dt)):
            if not step(support_ref, allocation(supports)):
                return
        scratch = fresh_data(model, data)
        arm_joints = [model.joint(f"{'left' if request.limb == Limb.LEFT_HAND else 'right'}_{n}").id
                      for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
        arm_q = model.jnt_qposadr[arm_joints]
        scratch.qpos[arm_q] = last_qref[arm_q]; mujoco.mj_forward(model, scratch)
        site = scratch.site(f"{request.limb.value.lower()}_site")
        begin_frame = Frame(tuple(site.xpos), tuple(tuple(v for v in row) for row in site.xmat.reshape(3, 3)))
        enter("REACH")
        reach_start = float(data.time)
        preceding = manager._capture_measurement(request.limb, target)
        for i in range(round((request.reach_s + request.capture_timeout_s) / dt)):
            desired_frame = reach_frame(begin_frame, frame, i * dt, request.reach_s, request.clearance_m)
            solution = solve_hand_reference(model, data.qpos, request.limb, desired_frame)
            if not solution.converged:
                status, reason = SingleHandStatus.REACH_INFEASIBLE, solution.reason
                return
            qref = last_qref.copy(); qref[arm_q] = np.array(solution.qpos)[arm_q]
            if fault == "support_loss" and i == round(.3 / dt):
                body = model.body("left_foot").id
                owned_force = (body, data.xfrc_applied[body, :3].copy(), np.array([1500., 0., 0.]))
                data.xfrc_applied[body, :3] = owned_force[2]
            if not step(qref, allocation(supports)):
                return
            measurement = manager._capture_measurement(request.limb, target)
            measurement["capture_margin_m"] = CAPTURE_DISTANCE - measurement["gap_m"]
            if manager.can_attach(request.limb, target):
                if eligible is None:
                    eligible = {"time_s": float(data.time), **measurement, "reach_elapsed_s": float(data.time - reach_start),
                                "reference_time_s": i * dt, "preceding_measurement": dict(preceding)}
                preferred = (request.acquisition_policy == "first_eligible"
                             or i * dt >= request.reach_s + .1 - 1e-12
                             and measurement["relative_speed_m_s"] <= preceding["relative_speed_m_s"]
                             and measurement["gap_m"] <= preceding["gap_m"])
                if preferred and manager.attach(request.limb, target):
                    captures = {"time_s": float(data.time), "step": steps, **measurement,
                                 "preceding_measurement": dict(preceding), "first_eligible_time_s": eligible["time_s"],
                                 "reach_elapsed_s": float(data.time - reach_start), "reference_time_s": i * dt,
                                 "reference_endpoint_error_m": float(np.linalg.norm(np.array(desired_frame.position) - frame.position)),
                                 "acquisition_policy": request.acquisition_policy,
                                 "initial_reaction_N": manager.capture_events[-1]["initial_reaction_N"]}
                    events.append({"event": "CAPTURED", "time_s": float(data.time), "step": steps})
                    decisions = manager.evaluate_and_update()
                    captures["post_activation_decisions"] = {l.value: asdict(d) for l, d in decisions.items()}
                    if any(not d.maintain for d in decisions.values()):
                        status, reason = SingleHandStatus.GRIP_FAILURE, "Bounded grasp failed in fresh post-activation solve"
                        status_failure = {"force_epoch": "fresh_post_activation_solve",
                                          "decisions": captures["post_activation_decisions"]}
                        return
                    break
            preceding = measurement
        if captures is None:
            status, reason = SingleHandStatus.CAPTURE_FAILURE, "No physical acquisition before declared capture timeout"
            return
        enter("SETTLE")
        desired_frame = canonical_geometry(target).hand_frame
        tracker = ReadinessTracker(model, data, manager)
        new_contacts = {**intent, request.limb: request.target}
        new_pose = solve_contact_pose(model, scene, profile, data.qpos, new_contacts)
        if not new_pose.admitted:
            status, reason = SingleHandStatus.CONTROL_FAILURE, "Captured state could not form an admitted final static reference: " + new_pose.reason
            return
        settle_start_ref = last_qref.copy()
        settle_goal_ref = np.array(new_pose.qpos)
        for i in range(round(request.settle_timeout_s / dt)):
            if fault == "grip_after_capture" and i == 2:
                body = model.body(request.limb.value.lower()).id
                owned_force = (body, data.xfrc_applied[body, :3].copy(), np.array([0., -2000., 0.]))
                data.xfrc_applied[body, :3] = owned_force[2]
            b, _ = minimum_jerk((i + 1) * dt, .5)
            tangent = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, tangent, 1., settle_start_ref, settle_goal_ref)
            qref = settle_start_ref.copy(); mujoco.mj_integratePos(model, qref, tangent, b)
            if not step(qref, allocation(new_contacts)):
                return
            if ((i + 1) * dt >= .5 and np.max(np.abs(last_qref[qi] - settle_goal_ref[qi])) <= 1e-10
                    and np.max(np.abs(last_qdref[vi])) <= 1e-8 and tracker.evidence.ready):
                status, reason = SingleHandStatus.SUCCESS, "Released, physically captured new hold and sustained new Stage3 readiness"
                final_reference = new_pose.reference
                events.append({"event": "FINAL_READY", "time_s": float(data.time), "step": steps})
                return
        status, reason = SingleHandStatus.TIMEOUT, "Post-capture sustained readiness timeout"
    try:
        run()
    except Exception as error:
        if error is observer_error:
            raise
        status, reason = SingleHandStatus.CONTROL_FAILURE, str(error)
    finally:
        if owned_force is not None:
            body, original, declared = owned_force
            force = data.xfrc_applied[body, :3]
            # Caller replacements are no longer owned by this executor.
            unchanged = force == declared
            force[unchanged] = original[unchanged]
            owned_force = None
    return finish()


def make_single_hand_fixture(timestep=.002, *, rise=.06, target_quality=1.):
    """Stage3 body/controller with flexed-knee room and one extra static hold."""
    from .contact_benchmarks import make_mixed_fixture
    fixture = make_mixed_fixture(timestep)
    seed = fixture.reference.copy()
    for side in ("left", "right"):
        for joint, angle in (("hip_pitch", .15), ("knee", .3), ("ankle_pitch", .15)):
            seed[fixture.model.joint(f"{side}_{joint}").qposadr[0]] = angle
    probe = mujoco.MjData(fixture.model); probe.qpos[:] = seed; mujoco.mj_forward(fixture.model, probe)
    regions = tuple(replace(r, position=(r.position[0], r.position[1], probe.site(f"{r.id}_site").xpos[2] - .041))
                    if r.id.endswith("foot") else r for r in fixture.scene.contact_regions)
    source = fixture.scene.region("right_hand")
    target = replace(source, id="reach_target", position=(source.position[0], source.position[1], source.position[2] + rise),
                     grip_quality=target_quality)
    intent = {l: l.value.lower() for l in Limb}
    scene = replace(fixture.scene, contact_regions=(*regions, target), start_configuration=intent)
    tree = ET.fromstring(build_mjcf(scene, fixture.profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data, scene, fixture.profile, seed


def run_single_hand_benchmark(timestep=.002, *, scenario="success", observer=None, keep_samples=True):
    if not isinstance(scenario, str) or scenario not in _SCENARIOS:
        raise ValueError("Unknown single-hand benchmark scenario")
    rise = 2. if scenario == "unreachable" else .06
    model, data, scene, profile, seed = make_single_hand_fixture(timestep, rise=rise)
    retarget = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
    if not retarget.admitted:
        return {"success": False, "status": SingleHandStatus.INITIALIZATION_FAILURE.value, "reason": retarget.reason,
                "retarget": retarget, "steps": 0, "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(), "time_s": float(data.time)}
    reference, manager = initialize_static_reference(model, data, scene, profile, retarget.qpos, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager, duration=2., settle=1., score_window=.5,
                                  keep_samples=False)
    if not initial["success"]:
        return {"success": False, "status": SingleHandStatus.INITIALIZATION_FAILURE.value, "reason": initial["reason"],
                "initial_static": initial, "retarget": retarget}
    request = SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target",
                                approach_normal=(math.sin(math.radians(31)), -math.cos(math.radians(31)), 0.)
                                if scenario == "orientation_invalid" else None)
    result = execute_single_hand(model, data, scene, profile, reference, manager, request,
                                 fault=scenario if scenario in ("support_loss", "grip_after_capture") else None,
                                 observer=observer, keep_samples=keep_samples)
    result.update(initial_static=initial, retarget=retarget, scenario=scenario,
                  model_proof={"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq,
                               "nexclude": model.nexclude, "mass_kg": float(model.body_subtreemass[model.body("climber_root").id]),
                               "equalities": [model.equality(i).name for i in range(model.neq)]})
    return result
