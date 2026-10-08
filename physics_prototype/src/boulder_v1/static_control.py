"""Static torque control on admitted contacts; no contact forces or root wrench injection."""
from __future__ import annotations

import copy
from dataclasses import asdict
from enum import Enum
import math

import mujoco
import numpy as np

from .contact import effective_grip_capacity
from .contact_geometry import ContactMode, canonical_geometry
from .grasp import AttachmentStateError
from .locomotion import get_state_summary
from .runtime import compute_pose_control, validate_reference_pose
from .static_state import FEET, HANDS, ReadinessTracker, _finite_state, _unexpected_loaded_contacts
from .support import fresh_data


class StaticStatus(str, Enum):
    SUCCESS = "SUCCESS"
    REFERENCE_INVALID = "REFERENCE_INVALID"
    TORQUE_LIMITED = "TORQUE_LIMITED"
    CONTACT_LOSS = "CONTACT_LOSS"
    GRASP_OVERLOAD = "GRASP_OVERLOAD"
    STABILIZATION_TIMEOUT = "STABILIZATION_TIMEOUT"
    NONFINITE_STATE = "NONFINITE_STATE"


def estimate_static_feedforward(model, scene, profile, reference):
    """Minimum-change six-root-row balance at real overlap centers, checked not optimized.

    Nominal weight shares are 45% per foot and 5% per hand. An ordinary NumPy
    least-squares projection balances root force/moment. Only derived hinge torque
    is commanded. Friction, hand capacity and original motor ceilings must admit
    this candidate; failure means this estimate failed, not physical infeasibility.
    """
    validate_reference_pose(model, reference.qpos)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = reference.qpos
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    weight = -model.body_subtreemass[model.body("climber_root").id] * model.opt.gravity
    blocks, nominal, points = [], [], []
    for limb in (*FEET, *HANDS):
        region = scene.region(reference.contact_intent[limb])
        if limb.is_foot:
            evidence = next(r for r in reference.residuals if r.limb == limb)
            if evidence.support_point_world is None:
                raise ValueError("Reference requires a measured sole/surface overlap point")
            point = np.array(evidence.support_point_world)
            body = model.geom(f"{limb.value.lower()}_geom").bodyid[0]
            share = .45
        else:
            site = scratch.site(f"{limb.value.lower()}_site")
            point, body, share = site.xpos.copy(), model.site_bodyid[site.id], .05
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jp, jr, point, int(body))
        blocks.append(jp.T)
        nominal.extend(share * weight)
        points.append(tuple(float(v) for v in point))
    jacobian = np.concatenate(blocks, axis=1)
    root = jacobian[:6]
    base = np.array(nominal)
    forces = base + np.linalg.lstsq(root, scratch.qfrc_bias[:6] - root @ base, rcond=None)[0]
    balance = root @ forces - scratch.qfrc_bias[:6]
    if not np.isfinite(forces).all() or np.max(np.abs(balance)) > 1e-8:
        raise ValueError("Static feedforward root balance did not converge")
    for index, limb in enumerate((*FEET, *HANDS)):
        region = scene.region(reference.contact_intent[limb])
        force = forces[3 * index:3 * index + 3]
        if limb.is_foot:
            local = np.array(canonical_geometry(region).foot_surface_frame.rotation).T @ force
            if local[2] <= 5. or np.abs(local[:2]).sum() > min(1.8, region.friction) * local[2] + 1e-8:
                raise ValueError("Static force estimate violates unilateral foot/friction bounds")
        elif np.linalg.norm(force) > effective_grip_capacity(profile, region, region.normal):
            raise ValueError("Static force estimate exceeds hand capacity")
    joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joints]
    torque = scratch.qfrc_bias[dofs] - (jacobian @ forces)[dofs]
    ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    if not np.isfinite(torque).all() or np.any(np.abs(torque) > ceiling):
        raise ValueError("Static force estimate exceeds available motor torque")
    return {"torques_Nm": {model.joint(int(j)).name: float(v) for j, v in zip(joints, torque)},
            "planned_forces_world_N": forces.reshape(4, 3).tolist(), "support_points_world_m": points,
            "max_root_balance_residual": float(np.max(np.abs(balance))),
            "scope": "reference force estimate; only derived hinge torques applied, no root/contact injection"}


def execute_static_hold(model, data, scene, profile, reference, manager, *, duration=11., settle=1.,
                        score_window=2., disturbance=None, observer=None, keep_samples=True):
    """Continue one explicit episode: native stepping, bounded grasps, isolated observers.

    Default protocol is 1s settling + 10s hold, final 2s scored. Feedforward ramps
    over 0.2s while collisions load naturally. A disturbance is a declared finite
    COM force pulse (body, start_s, duration_s, force_world_N), not a state edit.
    Pulse timing must identify whole native intervals relative to episode start.
    Observers receive nonterminal samples and a separate actual terminal snapshot;
    their exceptions propagate after declared external forces are restored.
    """
    manager.require_session(model, data, scene)
    if manager.mode != ContactMode.PHYSICAL or manager.profile != profile:
        raise AttachmentStateError("Static physical execution requires its bound physical profile")
    dt = float(model.opt.timestep)
    if (not all(math.isfinite(v) for v in (duration, settle, score_window, dt)) or dt <= 0 or settle < .2
            or score_window <= 0 or duration < settle + score_window
            or abs(duration / dt - round(duration / dt)) > 1e-8):
        raise ValueError("Invalid static integration/settling/scoring budgets")
    total_steps = round(duration / dt)
    initial = get_state_summary(model, data, manager)
    start = float(data.time)
    tracker = ReadinessTracker(model, data, manager)
    allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{hold}"))
               for limb, hold in reference.contact_intent.items()}
    samples = []
    status, reason, steps = StaticStatus.STABILIZATION_TIMEOUT, "Sustained stabilization not attained", 0
    max_applied = np.zeros(2)
    saturation_duration = 0.
    numerical_warnings = tuple(int(w.number) for w in data.warning)
    applied_external = data.xfrc_applied.copy()
    if np.any(data.qfrc_applied) or np.any(applied_external):
        raise ValueError("Static hold starts without undeclared external force")
    pulse_body = None
    realization = None
    if disturbance is not None:
        force = np.asarray(disturbance["force_world_N"], dtype=float)
        onset, pulse_duration = disturbance["start_s"], disturbance["duration_s"]
        pulse_body = model.body(disturbance["body"]).id
        if (force.shape != (3,) or not np.isfinite(force).all() or not math.isfinite(onset)
                or not math.isfinite(pulse_duration) or onset < settle or pulse_duration <= 0
                or onset + pulse_duration >= duration - score_window
                or model.body_subtreemass[pulse_body] <= 0 or model.body_weldid[pulse_body] == 0):
            raise ValueError("Invalid declared disturbance")
        pulse_start_step, pulse_steps = round(onset / dt), round(pulse_duration / dt)
        if (pulse_steps <= 0 or abs(onset / dt - pulse_start_step) > 1e-8
                or abs(pulse_duration / dt - pulse_steps) > 1e-8):
            raise ValueError("Disturbance timing must use whole native intervals")
        pulse_end_step = pulse_start_step + pulse_steps
        realization = {"applied_steps": 0, "integrated_impulse_world_Ns": [0., 0., 0.],
                       "realized_start_elapsed_s": None, "realized_end_elapsed_s": None,
                       "time_basis": "native integration intervals relative to episode start"}
    try:
        if dict(reference.contact_intent) != dict(scene.start_configuration):
            raise ValueError("Reference contact intent differs from initialized episode")
        allocation = estimate_static_feedforward(model, scene, profile, reference)
        command = compute_pose_control(model, data, reference.target_pose, feedforward={})
    except ValueError as exc:
        return {"success": False, "status": StaticStatus.REFERENCE_INVALID.value, "reason": str(exc),
                "steps": 0, "duration_s": 0., "initial_state": initial, "final_state": initial,
                "samples": [], "contact_acceptance": False, "controller_convergence": False,
                "disturbance_realization": realization}

    joints = model.actuator_trnid[:, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    before, pulse_active = start, False

    def measure_row(hands, feet, evidence, unintended, *, terminal=False):
        linear, angular = float(np.linalg.norm(data.qvel[:3])), float(np.linalg.norm(data.qvel[3:6]))
        maximum = float(np.max(np.abs(data.qvel[vi])))
        rms = float(np.sqrt(np.mean(data.qvel[vi] ** 2)))
        readiness, control = asdict(evidence), asdict(command)
        if terminal:
            # Recovery may replace ctrl. Keep the last requested impedance metadata,
            # but never present its old commanded torque as the current live control.
            torque = data.ctrl * model.actuator_gear[:, 0]
            control.update(commanded_Nm=tuple(float(v) for v in torque),
                           utilization=tuple(float(v) for v in np.abs(torque) / command.limits_Nm),
                           matches_last_request=bool(np.array_equal(
                               data.ctrl, np.array(command.commanded_Nm) / model.actuator_gear[:, 0])))
            readiness.update(time=float(data.time), root_linear_speed=linear, root_angular_speed=angular,
                             max_hinge_speed=maximum, rms_hinge_speed=rms)
            if status != StaticStatus.SUCCESS:
                readiness.update(ready=False, duration=0., reason=reason)
        return {"time_s": float(data.time), "elapsed_s": float(data.time - start),
                "integrated_elapsed_s": steps * dt, "interval_index": steps - 1 if steps else None,
                "command_interval_start_s": before,
                "root_linear_m_s": linear, "root_angular_rad_s": angular,
                "joint_max_rad_s": maximum, "joint_rms_rad_s": rms,
                "joint_error_max_rad": float(np.max(np.abs(np.array(reference.qpos)[qi] - data.qpos[qi]))),
                "joint_velocity_rad_s": data.qvel[vi].tolist(), "ctrl": data.ctrl.tolist(), "command": control,
                "hands": {limb.value: asdict(hand) for limb, hand in hands.items()},
                "feet": {limb.value: asdict(foot) for limb, foot in feet.items()},
                "readiness": readiness, "unintended_contacts": unintended,
                "disturbance_active": bool(pulse_active and not terminal),
                "pose_available": bool(_finite_state(model, data)),
                "status": status.value if terminal else "RUNNING", "terminal": terminal,
                "reason": reason if terminal else "Static hold is running"}

    def observe(row):
        if observer is not None:
            observed_model = copy.copy(model)
            observed_data = mujoco.MjData(observed_model)
            mujoco.mj_copyData(observed_data, observed_model, data)
            observer(copy.deepcopy(row), observed_model, observed_data)

    try:
        for step in range(total_steps):
            before = float(data.time)
            elapsed = before - start
            scale = min(1., elapsed / .2)
            pulse_active = bool(pulse_body is not None and pulse_start_step <= step < pulse_end_step)
            if pulse_body is not None:
                data.xfrc_applied[pulse_body, :3] = force if pulse_active else 0.
            try:
                command = compute_pose_control(model, data, reference.target_pose,
                                               feedforward={name: scale * value for name, value in allocation["torques_Nm"].items()})
                decisions = manager.evaluate_and_update()
                if any(not d.maintain for d in decisions.values()) or any(not manager.is_attached(limb) for limb in HANDS):
                    status, reason = StaticStatus.GRASP_OVERLOAD, "Hand capacity/geometry failed before integration"
                    break
                mujoco.mj_step(model, data)
                steps += 1
                continuous_clock = abs(data.time - before - dt) <= 1e-12
                if pulse_active and continuous_clock and np.array_equal(data.xfrc_applied[pulse_body, :3], force):
                    realization["applied_steps"] += 1
                    realization["integrated_impulse_world_Ns"] = (force * (realization["applied_steps"] * dt)).tolist()
                    if realization["realized_start_elapsed_s"] is None:
                        realization["realized_start_elapsed_s"] = step * dt
                    realization["realized_end_elapsed_s"] = (step + 1) * dt
                if (not continuous_clock or tuple(int(w.number) for w in data.warning) != numerical_warnings):
                    manager.synchronize_from_live()
                    status, reason = StaticStatus.NONFINITE_STATE, "Numerical recovery or discontinuous clock"
                    break
                applied_loads = [math.hypot(*manager._reaction(limb, data)) for limb in HANDS]
                max_applied = np.maximum(max_applied, applied_loads)
                if _unexpected_loaded_contacts(model, data, allowed):
                    status, reason = StaticStatus.CONTACT_LOSS, "Unintended body support in applied native solve"
                    break
                decisions = manager.evaluate_and_update(applied_data=data)
                if any(not d.maintain for d in decisions.values()):
                    status, reason = StaticStatus.GRASP_OVERLOAD, "Hand capacity/geometry failed over integration interval"
                    break
                snapshot = manager.contact_snapshot()
                endpoint = fresh_data(model, data)
                unintended = _unexpected_loaded_contacts(model, endpoint, allowed)
                if (any(not h.active or not h.valid for h in snapshot.hands.values()) or unintended
                        or any(not f.supporting or f.slipping for f in snapshot.feet.values())):
                    status, reason = StaticStatus.CONTACT_LOSS, "Native support/slip/hand geometry or unintended body contact failed"
                    break
                evidence = tracker.sample_after_step(before)
                saturation_duration = saturation_duration + dt if any(command.saturated) else 0.
                row = measure_row(snapshot.hands, snapshot.feet, evidence, unintended)
                samples.append(row)
            except (ValueError, FloatingPointError) as exc:
                status, reason = StaticStatus.NONFINITE_STATE, str(exc)
                break
            if step % max(1, round(.05 / model.opt.timestep)) == 0:
                observe(row)
            if saturation_duration >= .5:
                status, reason = StaticStatus.TORQUE_LIMITED, "Motor capability saturation persisted for 0.5s"
                break
    finally:
        if pulse_body is not None:
            data.xfrc_applied[pulse_body, :3] = applied_external[pulse_body, :3]
    final = get_state_summary(model, data, manager)
    scored = [s for s in samples if s["elapsed_s"] >= duration - score_window - 1e-9]
    complete = steps == total_steps
    convergence = bool(status == StaticStatus.STABILIZATION_TIMEOUT and complete and len(samples) == steps
                       and scored and all(s["readiness"]["ready"] for s in scored)
                       and max(s["root_linear_m_s"] for s in scored) <= .02
                       and max(s["root_angular_rad_s"] for s in scored) <= .05
                       and max(s["joint_max_rad_s"] for s in scored) <= .1)
    if convergence:
        status, reason = StaticStatus.SUCCESS, "Complete hold with sustained native-contact static convergence"
    try:
        terminal_contacts = _unexpected_loaded_contacts(model, fresh_data(model, data), allowed)
    except ValueError:
        terminal_contacts = None
    terminal_row = measure_row(final.hand_states, final.foot_states, tracker.evidence, terminal_contacts, terminal=True)
    observe(terminal_row)
    result = {"success": convergence, "status": status.value, "reason": reason, "steps": steps,
              "duration_s": steps * dt, "settle_s": settle, "controlled_hold_s": duration - settle,
              "scored_window_s": score_window, "dt_s": dt,
              "contact_acceptance": bool(complete and len(samples) == steps
                                         and status not in (StaticStatus.CONTACT_LOSS, StaticStatus.GRASP_OVERLOAD,
                                                            StaticStatus.NONFINITE_STATE)),
              "controller_convergence": convergence,
              "initial_state": initial, "final_state": final, "allocation": allocation,
              "readiness": terminal_row["readiness"], "terminal_observation": terminal_row,
              "capture_events": copy.deepcopy(manager.capture_events),
              "release_events": copy.deepcopy(manager.releases), "max_applied_hand_load_N": max_applied.tolist(),
              "disturbance": disturbance, "disturbance_realization": realization,
              "scored": {key: max(s[key] for s in scored) if scored else None for key in
                         ("root_linear_m_s", "root_angular_rad_s", "joint_max_rad_s", "joint_error_max_rad")},
              "joint_rms_rad_s": float(np.sqrt(np.mean([s["joint_rms_rad_s"] ** 2 for s in scored]))) if scored else None,
              "actuator_utilization_max": max((max(s["command"]["utilization"]) for s in samples), default=0.),
              "actuator_utilization_mean": float(np.mean([s["command"]["utilization"] for s in samples])) if samples else 0.,
              "saturation_fraction": float(np.mean([any(s["command"]["saturated"]) for s in samples])) if samples else 0.}
    result["feet"] = {limb.value: {"support_fraction": float(np.mean([s["feet"][limb.value]["supporting"] for s in samples])),
                                  "slip_fraction": float(np.mean([s["feet"][limb.value]["slipping"] for s in samples])),
                                  "normal_min_N": min(s["feet"][limb.value]["normal_force"] for s in samples),
                                  "normal_mean_N": float(np.mean([s["feet"][limb.value]["normal_force"] for s in scored])) if scored else None,
                                  "tangential_max_N": max(s["feet"][limb.value]["tangential_force"] for s in samples),
                                  "slip_max_m_s": max(s["feet"][limb.value]["tangential_speed"] for s in samples)}
                      for limb in FEET} if samples else {}
    if disturbance is not None:
        recovery = next((s["integrated_elapsed_s"] for s in samples if s["interval_index"] >= pulse_end_step
                         and not s["disturbance_active"] and s["readiness"]["ready"]), None)
        response = [s for s in (*samples, terminal_row) if s["interval_index"] is not None
                    and pulse_start_step <= s["interval_index"] < pulse_end_step + round(1. / dt)] if realization["applied_steps"] else []
        result["recovery"] = {"ready_after_pulse_s": recovery,
                              "peak_root_speed_m_s": max((s["root_linear_m_s"] for s in response
                                                         if math.isfinite(s["root_linear_m_s"])), default=None),
                              "peak_joint_speed_rad_s": max((s["joint_max_rad_s"] for s in response
                                                             if math.isfinite(s["joint_max_rad_s"])), default=None)}
    if scored:
        from .controller_diagnostics import _signal_metrics, VELOCITY_NOISE_FLOOR, TORQUE_NOISE_FLOOR
        velocity = np.array([s["joint_velocity_rad_s"] for s in scored])
        torque = np.array([s["command"]["commanded_Nm"] for s in scored])
        worst = int(np.argmax(np.mean(velocity ** 2, axis=0)))
        result["chatter"] = {"joint": samples[-1]["command"]["joint_names"][worst],
                             "velocity": _signal_metrics(velocity[:, worst], VELOCITY_NOISE_FLOOR, model.opt.timestep),
                             "torque": _signal_metrics(torque[:, worst], TORQUE_NOISE_FLOOR, model.opt.timestep)}
    if keep_samples:
        result["samples"] = samples
    return result


def run_static_benchmark(timestep=.002, *, disturbance=None, observer=None, keep_samples=True):
    """Unmodified Stage2 mixed scene/body/contact model with the Stage3 controller."""
    from .contact_benchmarks import make_mixed_fixture, _model_proof
    from .static_state import initialize_static_reference
    from .schema import Limb

    fixture = make_mixed_fixture(timestep)
    intent = {limb: limb.value.lower() for limb in Limb}
    reference, manager = initialize_static_reference(fixture.model, fixture.data, fixture.scene,
                                                     fixture.profile, fixture.reference, intent)
    result = execute_static_hold(fixture.model, fixture.data, manager.scene, fixture.profile,
                                 reference, manager, disturbance=disturbance, observer=observer,
                                 keep_samples=keep_samples)
    result["reference_admission"] = [asdict(r) for r in reference.residuals]
    result["model_proof"] = _model_proof(fixture)
    return result
