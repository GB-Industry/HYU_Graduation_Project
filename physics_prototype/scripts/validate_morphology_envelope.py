#!/usr/bin/env python3
"""Stage 5.2 fixed-scene native morphology study, not a global feasibility proof.

Default: 40 primary episodes plus four baseline counterexamples. Each execution
owns its source hold and retained native samples; reconstruction runs no physics.
Filtered invocations certify only their requested evidence, never the full study.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from boulder_v1 import morphology_envelope as envelope
from boulder_v1.contact import CAPTURE_DISTANCE, CAPTURE_SPEED, CAPTURE_ORIENTATION, effective_grip_capacity
from boulder_v1.contact_geometry import Frame, canonical_geometry
from boulder_v1.schema import Affordance, ClimberProfile, ContactRegion, SourceType
from boulder_v1.single_hand import SingleHandRequest, reach_frame
from boulder_v1.support import FORCE_TOLERANCE, MAX_SLIP_SPEED
from scripts.transfer_motion_audit import audit_motion
from scripts.validate_transfers import _physical_success
from scripts.validate_whole_body import (_evidence_value, _finite_evidence, _hash,
                                        _json_value, _model_hash, _write_json)
from scripts import validate_whole_body as whole_body

NEGATIVES = ("beyond_reach", "blocked_path")
CLASSIFICATIONS = frozenset((
    "GEOMETRY_INFEASIBLE", "ROM_INFEASIBLE", "SUPPORT_INFEASIBLE", "COLLISION_INFEASIBLE",
    "UNKNOWN_COLLISION_BLOCKED_SEARCH", "ROM_LIMITED_SEARCH", "LOCAL_SEARCH_UNRESOLVED",
    "SOURCE_INFEASIBLE", "SOURCE_STATIC_FAILURE", "PHYSICAL_SUCCESS", "DYNAMIC_CONTACT_INFEASIBLE"))
LENGTHS = ("upper_arm_length", "forearm_length", "thigh_length", "shin_length")


def _jobs(args):
    profiles = envelope.study_profiles()
    cases = envelope.target_perturbations()
    dts = (args.dt,) if args.dt is not None else (.002, .001)
    jobs = []
    if args.suite in ("all", "matrix"):
        jobs.extend(("matrix", p, c, dt) for p in profiles for c in cases for dt in dts)
    if args.suite in ("all", "negative"):
        jobs.extend(("negative", "baseline", c, dt) for c in NEGATIVES for dt in dts)
    return [job for job in jobs if (not args.profile or job[1] in args.profile)
            and (not args.case or job[2] in args.case)]


def _provenance(argv):
    provenance = whole_body._provenance(argv)
    provenance.update(command=[sys.executable, str(Path(__file__).resolve()), *argv],
                      validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    provenance["audit_dependencies"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (ROOT / "scripts" / name for name in ("validate_whole_body.py", "validate_transition.py",
                                             "validate_transfers.py", "transfer_motion_audit.py"))}
    return provenance


def _fixture_metadata(profile, case, dt):
    base = ClimberProfile("baseline").to_dict()
    parameters = profile.to_dict()
    scales = [parameters[key] / base[key] for key in LENGTHS]
    if (any(parameters[key] != value for key, value in base.items() if key not in (*LENGTHS, "name", "grip_capacity"))
            or not any(np.allclose(scales, scale, rtol=0., atol=1e-12) for scale in (.95, 1., 1.05))
            or parameters["grip_capacity"] not in (425., 850.)
            or parameters["grip_capacity"] == 425. and not np.allclose(scales, 1., rtol=0., atol=1e-12)):
        raise ValueError("Study requires one-factor limb geometry or grip changes with all other parameters fixed")
    model, _, scene, compiled_profile, seed, metadata = envelope.make_envelope_fixture(profile, case, dt)
    if not _finite_evidence(metadata):
        raise ValueError("Nonfinite reconstructed fixture inputs")
    inputs = _evidence_value(metadata)
    if (inputs["profile"] != profile.to_dict() or compiled_profile != profile
            or inputs["scene"] != _evidence_value(scene.to_dict())):
        raise ValueError("Fixture builder changed declared profile/scene")
    if inputs["seed_qpos"] != _json_value(seed) or model.opt.timestep != dt:
        raise ValueError("Fixture builder changed declared seed/timestep")
    # Hash all compiled parameters, not only the XML or morphology label.
    geometry = {name: getattr(model, name).tolist() for name in
                ("body_pos", "body_mass", "body_inertia", "geom_type", "geom_size", "geom_pos",
                 "geom_quat", "site_pos", "jnt_pos", "jnt_axis")}
    return model, {**inputs, "factory": "boulder_v1.morphology_envelope.make_envelope_fixture",
                   "dt_s": dt, "compiled_model_sha256": _model_hash(model),
                   "compiled_geometry_sha256": _hash(geometry), "scene_sha256": _hash(inputs["scene"]),
                   "input_sha256": _hash(inputs),
                   "expected_hand_capacity_N": {r.id: effective_grip_capacity(profile, r, (0., 0., 1.))
                                                for r in scene.contact_regions},
                   "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq")}}, inputs


def _execute_case(profile, case, dt):
    return envelope.run_envelope_case(profile, case, dt, keep_samples=True)


def _integration_checks(model, result):
    """Decode saved integration states on scratch, without forward/reset/steps."""
    checks = {}
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for endpoint in ("initial", "final"):
        vector = result.get(endpoint + "_integration_state")
        state = result.get(endpoint + "_state", {})
        if vector is None:
            checks[endpoint] = False
            continue
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (mujoco.mj_stateSize(model, spec),) or not np.isfinite(vector).all():
            checks[endpoint] = False
            continue
        scratch = mujoco.MjData(model)
        mujoco.mj_setState(model, scratch, vector, spec)
        matches = scratch.time == state.get("time") and all(
            np.array_equal(getattr(scratch, key), state.get(key)) for key in
            ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active"))
        checks[endpoint] = bool(matches and not scratch.qfrc_applied.any() and not scratch.xfrc_applied.any())
    return checks


def _limit_checks(model, inputs, saved):
    limits = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    rows = [saved["initial_state"], *saved.get("samples", []), saved["final_state"]]
    if saved.get("terminal_observation"):
        rows.append(saved["terminal_observation"])
    for row in rows:
        if "time_s" in row:
            for key, shape in (("qfrc_applied", (model.nv,)), ("external_force_world_N", (model.nbody, 6))):
                force = np.asarray(row.get(key), dtype=float)
                if force.shape != shape or not np.isfinite(force).all() or np.any(force):
                    return False
        if np.any(np.abs(np.asarray(row["ctrl"]) * model.actuator_gear[:, 0]) > limits + 1e-10):
            return False
        for hand in (row.get("hands", row.get("hand_states")) or {}).values():
            if hand.get("active") and (hand.get("capacity") is None or not math.isclose(
                    hand["capacity"], inputs["expected_hand_capacity_N"].get(hand.get("region_id"), -1.),
                    rel_tol=0., abs_tol=1e-10)):
                return False
        command = row.get("command")
        if command and not np.array_equal(command.get("limits_Nm"), limits):
            return False
        velocities = np.asarray(row["qvel"])
        actual_speeds = {"root_linear_m_s": np.linalg.norm(velocities[:3]),
                         "root_angular_rad_s": np.linalg.norm(velocities[3:6]),
                         "joint_max_rad_s": np.max(np.abs(velocities[model.jnt_dofadr[model.actuator_trnid[:, 0]]]))}
        if any(key in row and not math.isclose(row[key], value, rel_tol=0., abs_tol=1e-9)
               for key, value in actual_speeds.items()):
            return False
    for event in saved.get("release_events", []):
        if "capacity_N" in event and not np.isclose(event["capacity_N"],
                inputs["expected_hand_capacity_N"].get(event.get("region_id"), -1.), rtol=0., atol=1e-10):
            return False
    guard = saved.get("guard_failure") or {}
    for pair in list(guard.get("applied_unintended", [])) + list(guard.get("endpoint_unintended", [])):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(name, str) for name in pair) or len(set(pair)) != 2):
            return False
        try:
            for name in pair:
                model.geom(name)
        except (KeyError, ValueError, TypeError):
            return False
    reference = saved.get("final_reference")
    if reference is not None:
        qpos = np.asarray(reference.get("qpos"), dtype=float)
        if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
            return False
        for joint in model.actuator_trnid[:, 0]:
            angle = qpos[model.jnt_qposadr[joint]]
            if (not model.jnt_range[joint, 0] <= angle <= model.jnt_range[joint, 1]
                    or (reference.get("target_pose") or {}).get(model.joint(int(joint)).name) != angle):
                return False
    return True


def _held_contacts(row, contacts):
    if row.get("contacts", row.get("contact_configuration")) != contacts:
        return False
    hands = row.get("hands", row.get("hand_states")) or {}
    feet = row.get("feet", row.get("foot_states")) or {}
    if set(hands) != {"LEFT_HAND", "RIGHT_HAND"} or set(feet) != {"LEFT_FOOT", "RIGHT_FOOT"}:
        return False
    if any(hand.get("active") and limb not in contacts for limb, hand in hands.items()):
        return False
    for limb, hold in contacts.items():
        if limb.endswith("HAND"):
            hand = hands.get(limb, {})
            capacity, load = hand.get("capacity"), hand.get("load")
            if (not all(hand.get(key) is True for key in ("active", "valid", "measurement_valid"))
                    or hand.get("region_id") != hold or capacity is None or load is None
                    or not 0 <= load <= capacity or len(hand.get("force_world", [])) != 3
                    or not math.isclose(math.hypot(*hand["force_world"]), load, rel_tol=0., abs_tol=1e-8)
                    or not math.isclose(hand.get("margin", -1.), capacity - load, rel_tol=0., abs_tol=1e-8)):
                return False
        else:
            foot = feet.get(limb, {})
            matching = [c for c in foot.get("contacts", []) if c.get("surface_geom") == "geom_" + hold
                        and c.get("shoe_geom") == limb.lower() + "_geom" and c.get("admissible")]
            if (not foot.get("measurement_valid") or not foot.get("supporting") or foot.get("slipping")
                    or foot.get("normal_force", 0.) <= 5. or foot.get("tangential_speed", 1.) > MAX_SLIP_SPEED
                    or hold not in foot.get("support_regions", ()) or foot.get("idealized_attachment") is not None
                    or "geom_" + hold not in foot.get("support_surfaces", ())
                    or sum(c.get("normal_force", 0.) for c in matching) <= 5.
                    or any(c.get("tangential_speed", 1.) > MAX_SLIP_SPEED for c in matching)
                    or any(c.get("normal_force", 0.) > FORCE_TOLERANCE and not c.get("admissible")
                           for c in foot.get("contacts", []))):
                return False
    return True


def _speeds(row, motion):
    velocity = np.asarray(row["qvel"])
    hinges = velocity[[joint["dof_address"] for joint in motion["whole_case"]["joints"].values()]]
    return (float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:6])),
            float(np.max(np.abs(hinges))), float(np.sqrt(np.mean(hinges ** 2))))


def _terminal_matches(result, motion):
    terminal, final = result.get("terminal_observation") or {}, result["final_state"]
    if (terminal.get("terminal") is not True or terminal.get("pose_available") is not True
            or any(terminal.get(key) != result.get(key) for key in ("steps", "status", "reason", "readiness"))
            or terminal.get("time_s") != final["time"] or terminal.get("root_pose") != final["qpos"][:7]
            or any(terminal.get(key) != final.get(key) for key in ("qpos", "qvel", "ctrl", "eq_active"))
            or terminal.get("hands") != final.get("hand_states") or terminal.get("feet") != final.get("foot_states")
            or terminal.get("contacts") != final.get("contact_configuration")
            or any(key not in terminal or np.any(terminal[key]) for key in
                   ("qfrc_applied", "external_force_world_N"))):
        return False
    speeds = _speeds(final, motion)
    readiness = result.get("readiness") or {}
    return bool(readiness.get("time") == final["time"] and all(
        np.isclose(readiness.get(key, -1.), value, rtol=0., atol=1e-9) for key, value in zip(
            ("root_linear_speed", "root_angular_speed", "max_hinge_speed", "rms_hinge_speed"), speeds))
        and all(np.isclose(terminal.get(key, -1.), value, rtol=0., atol=1e-9) for key, value in zip(
            ("root_linear_m_s", "root_angular_rad_s", "joint_max_rad_s"), speeds)))


def _event_clock(result):
    start, end, dt = result["initial_state"]["time"], result["final_state"]["time"], result["dt_s"]
    previous = start
    for event in result.get("events", []):
        time, step = event.get("time_s"), event.get("step")
        if (time is None or not isinstance(step, int) or not 0 <= step <= result["steps"]
                or time < previous - 1e-10 or not start - 1e-10 <= time <= end + 1e-10
                or abs(time - start - step * dt) > 1e-8):
            return False
        previous = time
    return True


def _capture_valid(result, goal, *, bounded=True):
    capture = result.get("capture") or {}
    time, gap = capture.get("time_s"), capture.get("gap_m")
    if (time is None or gap is None or not 0 <= gap <= CAPTURE_DISTANCE + 1e-12
            or not 0 <= capture.get("relative_speed_m_s", -1.) <= CAPTURE_SPEED + 1e-12
            or not CAPTURE_ORIENTATION <= capture.get("orientation", -1.) <= 1.
            or not 0 <= capture.get("penetration_m", -1.) < .001
            or len(capture.get("relative_velocity_world_m_s", [])) != 3
            or not np.isclose(np.linalg.norm(capture["relative_velocity_world_m_s"]), capture["relative_speed_m_s"], atol=1e-12, rtol=0.)
            or not np.isclose(capture.get("capture_margin_m", -1.), CAPTURE_DISTANCE - gap, atol=1e-12, rtol=0.)
            or result.get("capture_time_s") != time or result.get("capture_error_m") != gap
            or result.get("capture_margin_m") != capture["capture_margin_m"]):
        return False
    events = [event for event in result.get("events", []) if event.get("event") == "CAPTURED"]
    if len(events) != 1 or events[0]["time_s"] != time or events[0]["step"] != capture.get("step"):
        return False
    native = [event for event in result.get("capture_events", []) if event.get("limb") == result["moving_limb"]
              and event.get("region_id") == result["target"] and event.get("time_s") == time]
    if len(native) != 1 or native[0].get("mode") != "physical":
        return False
    event = native[0]
    if (any(event.get(key) != capture.get(key) for key in ("gap_m", "relative_speed_m_s", "orientation", "penetration_m", "initial_reaction_N"))
            or event.get("initial_reaction_epoch") != "fresh_post_activation_solve"
            or len(event.get("initial_reaction_world_N", [])) != 3
            or not np.isclose(np.linalg.norm(event["initial_reaction_world_N"]), capture.get("initial_reaction_N", -1.), rtol=0., atol=1e-8)):
        return False
    request = result.get("request") or {}
    reach = result.get("reach_start_time_s")
    reach_events = [event for event in result.get("events", []) if event.get("phase") == "REACH"]
    if (reach is None or not reach < time <= result["final_state"]["time"]
            or len(reach_events) != 1 or reach_events[0]["time_s"] != reach
            or not result.get("release_time_s", time) < reach
            or request.get("hand_acquisition_policy") not in ("first_eligible", "endpoint_settle")
            or capture.get("acquisition_policy") != request.get("hand_acquisition_policy")
            or not np.isclose(capture.get("reach_elapsed_s", -1.), time - reach, rtol=0., atol=1e-8)
            or request.get("hand_acquisition_policy") == "endpoint_settle"
            and capture.get("reference_time_s", -1.) < request.get("hand_reach_s", 5.) + .1 - 1e-12):
        return False
    if bounded:
        decisions = capture.get("post_activation_decisions") or {}
        hands = result["final_state"].get("hand_states") or {}
        if set(decisions) != {limb for limb in goal if limb.endswith("HAND")}:
            return False
        for limb, decision in decisions.items():
            capacity = hands.get(limb, {}).get("capacity")
            load = decision.get("required_load")
            if (capacity is None or load is None or decision.get("maintain") is not True
                    or decision.get("effective_capacity") != capacity or not 0 <= load <= capacity
                    or not np.isclose(decision.get("utilization", -1.), load / capacity, rtol=0., atol=1e-10)):
                return False
        if not 0 <= capture.get("initial_reaction_N", -1.) <= hands[result["moving_limb"]]["capacity"]:
            return False
    return True


def _dynamic_failure(result, motion, goal, inputs):
    """Admit only witnessed physical failures of this chosen execution, not errors."""
    if (result.get("success") is not False or result.get("final_reference") is not None
            or (result.get("readiness") or {}).get("ready") is not False
            or result["readiness"].get("reason") != result["reason"] or result.get("steps", 0) <= 0):
        return False
    status, terminal = result["status"], result["terminal_observation"]
    final, guard = result["final_state"], result.get("guard_failure") or {}
    if status == "CONTACT_LOSS":
        if guard.get("feet") != terminal.get("feet") or guard.get("hands") != terminal.get("hands"):
            return False
        feet = guard["feet"]
        lost = any(not foot.get("supporting") or foot.get("slipping") or foot.get("normal_force", 0.) <= 5.
                   or foot.get("tangential_speed", 0.) > MAX_SLIP_SPEED
                   or any(c.get("normal_force", 0.) > FORCE_TOLERANCE and not c.get("admissible")
                          for c in foot.get("contacts", [])) for foot in feet.values())
        required = (goal if result.get("capture") else result["request"]["support_contacts"]
                    if result.get("released") else result["request"]["source_contacts"])
        lost = lost or any(hold not in feet.get(limb, {}).get("support_regions", ())
                           for limb, hold in required.items() if limb.endswith("FOOT"))
        lost = lost or any(not guard["hands"].get(limb, {}).get("active")
                           or not guard["hands"].get(limb, {}).get("valid")
                           or guard["hands"].get(limb, {}).get("region_id") != hold
                           for limb, hold in required.items() if limb.endswith("HAND"))
        unintended = guard.get("applied_unintended", []) + guard.get("endpoint_unintended", [])
        allowed = {frozenset((limb.lower() + "_geom", "geom_" + hold))
                   for limb, hold in final["contact_configuration"].items()}
        return bool(lost or any(isinstance(pair, (list, tuple)) and len(pair) == 2
                               and all(isinstance(name, str) and name for name in pair)
                               and frozenset(pair) not in allowed for pair in unintended))
    if status == "GRIP_FAILURE":
        for release in result.get("release_events", []):
            capacity = (inputs or {}).get("expected_hand_capacity_N", {}).get(release.get("region_id"))
            if (capacity is not None and release.get("capacity_N") == capacity
                    and release.get("force_epoch") in ("fresh_endpoint_solve", "applied_preintegration_solve")
                    and abs(release.get("time_s", -1.) - final["time"]) <= 1e-8
                    and abs(release.get("force_state_time_s", -1.) - (final["time"] -
                            (result["dt_s"] if release["force_epoch"] == "applied_preintegration_solve" else 0.))) <= 1e-8
                    and len(release.get("force_world_N", [])) == 3
                    and np.isclose(np.linalg.norm(release["force_world_N"]), release.get("required_load_N", -1.), atol=1e-8, rtol=0.)
                    and release.get("required_load_N", 0.) > capacity
                    and not terminal["hands"].get(release.get("limb"), {}).get("active")):
                return True
        return False
    if status == "ROM_FAILURE":
        return any(not joint["compiled_range_rad"][0] <= final["qpos"][joint["qpos_address"]] <= joint["compiled_range_rad"][1]
                   for joint in motion["whole_case"]["joints"].values())
    if status not in ("CAPTURE_FAILURE", "TIMEOUT"):
        return False
    request = result["request"]
    budget = SingleHandRequest(request["limb"], request["source"], request["target"], reach_s=request["hand_reach_s"])
    phase = "REACH" if status == "CAPTURE_FAILURE" else "SETTLE"
    events = [event for event in result["events"] if event.get("phase") == phase]
    duration = budget.reach_s + budget.capture_timeout_s if phase == "REACH" else budget.settle_timeout_s
    phase_rows = [row for row in result["samples"] if row.get("phase") == phase]
    if (len(events) != 1 or terminal.get("phase") != phase or not result.get("released")
            or not result["release_time_s"] < events[0]["time_s"] <= final["time"]
            or abs(final["time"] - events[0]["time_s"] - duration) > 1e-8
            or len(phase_rows) != round(duration / result["dt_s"]) or len(result["samples"]) != result["steps"]):
        return False
    if status == "CAPTURE_FAILURE":
        supports = request["support_contacts"]
        return bool(result.get("capture") is None and result.get("capture_time_s") is None
                    and not any(event.get("event") == "CAPTURED" for event in result["events"])
                    and result.get("reach_start_time_s") == events[0]["time_s"]
                    and not any(event.get("region_id") == result["target"] and event.get("limb") == result["moving_limb"]
                                for event in result.get("capture_events", []))
                    and _held_contacts(final, supports) and all(_held_contacts(row, supports) for row in phase_rows))
    return bool(_capture_valid(result, goal) and _held_contacts(final, goal)
                and phase_rows[-1].get("readiness", {}).get("ready") is False
                and all(_held_contacts(row, goal) for row in phase_rows))


def _valid_source(result):
    state = result.get("initial_state") or {}
    initial = result.get("initial_static") or {}
    if (not initial.get("success") or initial.get("final_state") != state
            or state.get("finite") is not True or state.get("contact_mode") != "physical"):
        return False
    contacts = state.get("contact_configuration") or {}
    hands, feet = state.get("hand_states") or {}, state.get("foot_states") or {}
    if set(contacts) != {"LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"}:
        return False
    for limb in ("LEFT_HAND", "RIGHT_HAND"):
        hand = hands.get(limb, {})
        if (not all(hand.get(key) is True for key in ("active", "valid", "measurement_valid"))
                or hand.get("region_id") != contacts[limb] or hand.get("capacity") is None
                or hand.get("load") is None or not 0 <= hand["load"] <= hand["capacity"]):
            return False
    for limb in ("LEFT_FOOT", "RIGHT_FOOT"):
        foot = feet.get(limb, {})
        if (not foot.get("measurement_valid") or not foot.get("supporting") or foot.get("slipping")
                or foot.get("normal_force", 0.) <= 5. or foot.get("tangential_speed", 1.) > MAX_SLIP_SPEED
                or contacts[limb] not in foot.get("support_regions", ()) or foot.get("idealized_attachment") is not None):
            return False
    terminal = initial.get("terminal_observation") or {}
    dt = result.get("dt_s", 0.)
    start = initial.get("initial_state") or {}
    # Static-hold rows have no force arrays. The matching native initial
    # integration state is decoded and checked for zero forces by the caller.
    source_readiness = initial.get("readiness") or {}
    return bool(dt > 0 and initial.get("steps") == round(2. / dt) and initial.get("duration_s") == 2.
                and abs(state.get("time", 0.) - start.get("time", 0.) - 2.) <= 1e-8
                and _held_contacts(state, contacts)
                and initial.get("status") == "SUCCESS" and initial.get("contact_acceptance") is True
                and initial.get("controller_convergence") is True and initial.get("disturbance") is None
                and initial.get("disturbance_realization") is None
                and terminal.get("terminal") is True and terminal.get("pose_available") is True
                and terminal.get("status") == initial["status"] and terminal.get("reason") == initial.get("reason")
                and terminal.get("time_s") == state["time"] and terminal.get("ctrl") == state["ctrl"]
                and terminal.get("hands") == state["hand_states"] and terminal.get("feet") == state["foot_states"]
                and terminal.get("readiness") == source_readiness and not terminal.get("unintended_contacts")
                and terminal.get("disturbance_active") is False and source_readiness.get("ready") is True
                and source_readiness.get("time") == state["time"] and source_readiness.get("duration", 0.) >= .5 - 1e-10
                and all(0 <= source_readiness.get(key, -1.) <= limit for key, limit in
                        (("root_linear_speed", .02), ("root_angular_speed", .05), ("max_hinge_speed", .10))))


def _case_verdict(result, motion, integration, inputs=None):
    if (not _finite_evidence(result) or motion is None or not all(integration.get(k) for k in ("initial", "final"))
            or not _valid_source(result) or result.get("classification") not in CLASSIFICATIONS
            or not isinstance(result.get("reason"), str) or not result["reason"]):
        return False
    source_speeds = _speeds(result["initial_state"], motion)
    source_readiness = result["initial_static"]["readiness"]
    if (any(value > limit for value, limit in zip(source_speeds, (.02, .05, .10)))
            or any(not math.isclose(source_readiness.get(key, -1.), value, rel_tol=0., abs_tol=1e-9)
                   for key, value in zip(("root_linear_speed", "root_angular_speed", "max_hinge_speed", "rms_hinge_speed"), source_speeds))):
        return False
    clock = motion["whole_case"]["timing"]
    clean = not any(clock[key] for key in ("time_reset_count", "sample_gap_count",
                                         "changed_pose_at_duplicate_time_count", "step_counter_reset_count"))
    clean = clean and all(item[key] for item in motion["native_accounting"] for key in
                         ("step_clock_matches", "duration_matches_steps", "zero_recorded_applied_forces",
                          "samples_have_both_force_channels"))
    clean = clean and all(motion[key] is True for key in
                         ("source_matches_initial_static_final_state", "case_initial_matches_first_move",
                          "case_final_matches_last_move"))
    assessment = result.get("assessment")
    if not clean or not isinstance(assessment, dict) or assessment.get("geometric_feasible") not in (True, False, None):
        return False
    classification = result["classification"]
    if classification == "SOURCE_STATIC_FAILURE":
        return False
    diagnostics = assessment.get("diagnostics") or {}
    if (diagnostics.get("actual_integration_state") != result.get("initial_integration_state")
            or diagnostics.get("actual_qpos") != result["initial_state"]["qpos"]):
        return False
    if not assessment.get("feasible"):
        # Unknown search failure is not a proof of geometric impossibility.
        return bool(classification not in ("PHYSICAL_SUCCESS", "DYNAMIC_CONTACT_INFEASIBLE")
                    and classification == assessment.get("classification") and result.get("status") == classification
                    and result["reason"] == assessment.get("reason") and not result.get("success")
                    and result.get("steps") == 0 and result.get("duration_s") == 0.
                    and result.get("released") is False and result.get("final_reference") is None
                    and not result.get("samples") and not result.get("events")
                    and result["initial_state"] == result["final_state"]
                    and result["initial_integration_state"] == result["final_integration_state"]
                    and not (result.get("readiness") or {}).get("ready"))
    if assessment.get("geometric_feasible") is not True or assessment.get("support_feasible") is not True:
        return False
    if (not result.get("samples") or not _terminal_matches(result, motion) or not _event_clock(result)
            or len(result["samples"]) not in (result["steps"], result["steps"] - 1)):
        return False
    request = result.get("request") or {}
    source = result["initial_state"]["contact_configuration"]
    supports = {limb: hold for limb, hold in source.items() if limb != result["moving_limb"]}
    goal = {**source, result["moving_limb"]: result["target"]}
    if (any(request.get(key) != result.get(key) for key in ("source", "target"))
            or request.get("limb") != result["moving_limb"] or request.get("source_contacts") != source
            or request.get("support_contacts") != supports or request.get("hand_reach_s") != 5.):
        return False
    whole, dt = motion["whole_case"], result["dt_s"]
    if (whole["bodies"]["root"]["largest_successive_step_m"] > .10 * dt + 1e-8
            or whole["root_orientation"]["largest_successive_angle_deg"] > np.degrees(.50 * dt) + 1e-5
            or whole["largest_joint_increment"]["rad"] > dt + 1e-8):
        return False
    for row in [*result["samples"], result["terminal_observation"]]:
        if any(value > limit for value, limit in zip(_speeds(row, motion), (.10, .50, 1.))):
            return False
    releases = [event for event in result["events"] if event.get("event") == "RELEASED"]
    if result.get("released") and (len(releases) != 1 or releases[0].get("limb") != result["moving_limb"]
                                  or releases[0].get("source") != result["source"]
                                  or result.get("release_time_s") != releases[0]["time_s"]):
        return False
    for index, row in enumerate(result["samples"], 1):
        if row.get("steps") != index or abs(row["time_s"] - result["initial_state"]["time"] - index * dt) > 1e-8:
            return False
        release = result.get("release_time_s")
        capture = result.get("capture_time_s")
        contacts = (source if release is None or row["time_s"] <= release + 1e-10 else
                    supports if capture is None or row["time_s"] <= capture + 1e-10 else goal)
        if not _held_contacts(row, contacts):
            return False
    if classification == "DYNAMIC_CONTACT_INFEASIBLE":
        return _dynamic_failure(result, motion, goal, inputs)
    if classification != "PHYSICAL_SUCCESS" or not _physical_success(result) or not result.get("released"):
        return False
    final, readiness = result["final_state"], result["readiness"]
    ready_events = [event for event in result["events"] if event.get("event") == "FINAL_READY"]
    if (not _capture_valid(result, goal) or len(ready_events) != 1 or ready_events[0]["time_s"] != final["time"]
            or not result["initial_state"]["time"] <= releases[0]["time_s"] < result["capture_time_s"] < final["time"]
            or result.get("release_time_s") != releases[0]["time_s"] or result.get("readiness_time_s") != final["time"]
            or result.get("final_contacts") != goal or result.get("new_contacts") != goal
            or result["final_reference"].get("contact_intent") != goal or not _held_contacts(final, goal)
            or readiness.get("duration", 0.) < .5 - 1e-10 or readiness.get("reason") != "Sustained physical readiness"
            or final["time"] - readiness["duration"] < result["capture_time_s"] + dt - 1e-10
            or len(result["samples"]) != result["steps"]):
        return False
    window = [row for row in result["samples"] if row["time_s"] >= final["time"] - .5 - 1e-10]
    if not window or final["time"] - window[0]["time_s"] < .5 - 1e-10:
        return False
    return bool(all(row.get("phase") == "SETTLE" and _held_contacts(row, goal)
                    and all(value <= limit for value, limit in zip(_speeds(row, motion), (.02, .05, .10)))
                    for row in window))


def _negative_certificate(model, result, case, inputs):
    """Recompute bounded geometry witnesses using compiled parameters and FK only."""
    assessment = result.get("assessment") or {}
    diagnostics = assessment.get("diagnostics") or {}
    if case not in NEGATIVES or result.get("success") or result.get("steps") != 0:
        return False
    regions = {}
    for item in inputs["scene"]["contact_regions"]:
        region = ContactRegion(**{**item, "source_type": SourceType(item["source_type"]),
                                  "affordances": frozenset(Affordance(v) for v in item["affordances"])})
        regions[region.id] = region
    target = canonical_geometry(regions[result["target"]]).hand_frame
    source_reference = assessment.get("source_reference") or {}
    qpos = np.asarray(source_reference.get("qpos"), dtype=float)
    if qpos.shape != (model.nq,):
        return False
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    if case == "beyond_reach":
        bound, motion = diagnostics.get("reach_bound") or {}, assessment.get("motion") or {}
        root = motion.get("root_target") or {}
        if not bound or not root or result.get("classification") != "GEOMETRY_INFEASIBLE":
            return False
        frame = Frame(tuple(root["position"]), tuple(map(tuple, root["rotation"])))
        address = int(model.joint("root").qposadr[0])
        scratch.qpos[address:address + 7] = (*frame.position, *frame.quaternion)
        for name, angle in motion["waist_target"].items():
            scratch.qpos[int(model.joint(name).qposadr[0])] = angle
        mujoco.mj_kinematics(model, scratch)
        shoulder = model.body(result["moving_limb"].lower().removesuffix("_hand") + "_upper_arm").id
        site = model.site(result["moving_limb"].lower() + "_site").id
        body, maximum = int(model.site_bodyid[site]), float(np.linalg.norm(model.site_pos[site]))
        while body != shoulder:
            if body == 0:
                return False
            maximum += float(np.linalg.norm(model.body_pos[body]))
            body = int(model.body_parentid[body])
        distance = float(np.linalg.norm(np.asarray(target.position) - scratch.xpos[shoulder]))
        return bool(result.get("classification") == "GEOMETRY_INFEASIBLE"
                    and assessment.get("geometric_feasible") is False and distance - maximum > 1e-6
                    and bound.get("scope") == "fixed policy endpoint root/waist; position-only upper bound"
                    and np.allclose(bound.get("shoulder_world_m"), scratch.xpos[shoulder], atol=1e-10, rtol=0.)
                    and all(np.isclose(bound.get(key, -1.), value, atol=1e-10, rtol=0.) for key, value in
                            (("distance_m", distance), ("maximum_m", maximum), ("gap_m", distance - maximum))))
    certificate = diagnostics.get("path_collision_certificate") or {}
    if (result.get("classification") != "COLLISION_INFEASIBLE"
            or certificate.get("moving_geom") != result["moving_limb"].lower() + "_geom"
            or certificate.get("moving_site") != result["moving_limb"].lower() + "_site"
            or certificate.get("proof") != "Required END point is strictly interior to both collidable rigid solids"
            or certificate.get("scope") != "specified sampled effector path only; not a global transfer or native force claim"
            or model.opt.disableflags & (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        return False
    hand, site, obstacle = (model.geom(certificate["moving_geom"]).id, model.site(certificate["moving_site"]).id,
                            model.geom(certificate["static_geom"]).id)
    hb, ob = int(model.geom_bodyid[hand]), int(model.geom_bodyid[obstacle])
    if (model.geom_type[hand] != mujoco.mjtGeom.mjGEOM_BOX or model.site_bodyid[site] != hb
            or model.body_weldid[hb] == 0 or model.body_weldid[ob] != 0
            or certificate.get("static_body") != model.body(ob).name or certificate.get("static_body_weldid") != 0
            or certificate["static_geom"] in ("geom_" + result["source"], "geom_" + result["target"])):
        return False
    paired = any({int(a), int(b)} == {hand, obstacle} for a, b in zip(model.pair_geom1, model.pair_geom2))
    masks = {"hand": [int(model.geom_contype[hand]), int(model.geom_conaffinity[hand])],
             "obstacle": [int(model.geom_contype[obstacle]), int(model.geom_conaffinity[obstacle])]}
    signature = (min(hb, ob) << 16) | max(hb, ob)
    if (certificate.get("explicit_pair") != paired or certificate.get("collision_masks") != masks
            or not paired and (signature in model.exclude_signature or mujoco.get_mjcb_contactfilter() is not None
                or not (masks["hand"][0] & masks["obstacle"][1] or masks["obstacle"][0] & masks["hand"][1]))):
        return False
    source = canonical_geometry(regions[result["source"]]).hand_frame
    clear = Frame(tuple(np.asarray(source.position) + .01 * np.asarray(source.normal)), source.rotation)
    phase, index, time = certificate.get("phase"), certificate.get("index"), certificate.get("time_s")
    duration = .5 if phase == "clearance" else 5.
    if (phase not in ("clearance", "reach") or not isinstance(index, int) or not 0 <= index <= round(duration / .05)
            or time != min(duration, index * .05)):
        return False
    frame = reach_frame(source if phase == "clearance" else clear, clear if phase == "clearance" else target,
                        time, duration, 0. if phase == "clearance" else .02)
    point = np.asarray(frame.position)
    mujoco.mj_kinematics(model, scratch)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[hand])
    local_site = rotation.reshape(3, 3).T @ (model.site_pos[site] - model.geom_pos[hand])
    hand_margin = float(np.min(model.geom_size[hand] - np.abs(local_site)))
    local = scratch.geom_xmat[obstacle].reshape(3, 3).T @ (point - scratch.geom_xpos[obstacle])
    if model.geom_type[obstacle] == mujoco.mjtGeom.mjGEOM_BOX:
        margin, shape = float(np.min(model.geom_size[obstacle] - np.abs(local))), "box"
    elif model.geom_type[obstacle] == mujoco.mjtGeom.mjGEOM_SPHERE:
        margin, shape = float(model.geom_size[obstacle, 0] - np.linalg.norm(local)), "sphere"
    else:
        return False
    return bool(hand_margin > .001 and margin > .001 and certificate.get("static_shape") == shape
                and all(np.allclose(certificate.get(key), value, rtol=0., atol=1e-10) for key, value in
                        (("point_world_m", point), ("point_obstacle_local_m", local), ("site_hand_local_m", local_site)))
                and np.allclose(certificate["task_frame"]["position"], point, rtol=0., atol=1e-10)
                and np.allclose(certificate["task_frame"]["rotation"], frame.rotation, rtol=0., atol=1e-10)
                and np.allclose(certificate["static_geom_frame"]["position"], scratch.geom_xpos[obstacle], rtol=0., atol=1e-10)
                and np.allclose(certificate["static_geom_frame"]["rotation"], scratch.geom_xmat[obstacle].reshape(3, 3), rtol=0., atol=1e-10)
                and np.allclose(certificate["static_body_frame"]["position"], scratch.xpos[ob], rtol=0., atol=1e-10)
                and np.allclose(certificate["static_body_frame"]["rotation"], scratch.xmat[ob].reshape(3, 3), rtol=0., atol=1e-10)
                and all(np.isclose(certificate.get(key, -1.), value, atol=1e-10, rtol=0.) for key, value in
                        (("hand_interior_margin_m", hand_margin), ("obstacle_interior_margin_m", margin),
                         ("intersection_ball_radius_m", min(hand_margin, margin)))))


def _metrics(model, saved, motion):
    if motion is None:
        return None
    whole = motion["whole_case"]
    assessment = saved.get("assessment") or {}
    preparation = assessment.get("preparation_qpos") or []
    rows = saved.get("samples") or []
    load = [row for row in rows if row.get("phase") == "LOAD_TRANSFER"]
    qi = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    final_torque = np.asarray(saved["final_state"]["ctrl"]) * model.actuator_gear[:, 0]
    utilization = [v for row in rows for v in (row.get("command") or {}).get("utilization", [])]
    return {"pelvis": whole["bodies"]["pelvis"], "com": whole["bodies"]["climber_com"],
            "root_orientation": whole["root_orientation"],
            "actual_elbows": {name: item for name, item in whole["joints"].items() if name.endswith("_elbow")},
            "waist_hip_knee": {name: item for name, item in whole["joints"].items()
                               if name.startswith("waist_") or "hip_" in name or name.endswith("_knee")},
            "support_loads": whole["support_loads"], "actuator_utilization_max": max(utilization, default=None),
            "final_torque_Nm": final_torque.tolist(),
            "load_transfer": {"sample_count": len(load), "first": load[0] if load else None,
                              "last": load[-1] if load else None},
            "capture": {key: saved.get(key) for key in ("capture_time_s", "capture_error_m", "capture_margin_m")},
            "capture_elapsed_s": (saved["capture_time_s"] - saved["initial_state"]["time"]
                                  if saved.get("capture_time_s") is not None else None),
            "initial_qpos_sha256": _hash(saved["initial_state"]["qpos"]),
            "final_hinge_rad": np.asarray(saved["final_state"]["qpos"])[qi].tolist(),
            "preparation_first_sha256": _hash(preparation[0]) if preparation else None,
            "preparation_last_sha256": _hash(preparation[-1]) if preparation else None,
            "preparation_path_sha256": _hash(preparation) if preparation else None,
            "reference_path_sha256": _hash([row.get("q_ref") for row in rows]),
            "geometric_path_sha256": _hash((assessment.get("diagnostics") or {}).get("reach_results", []))}


def _load_existing(path, provenance, profile, case, dt, fixture):
    if path.is_symlink() or not path.is_file() or path.resolve().parent != path.parent.resolve():
        raise ValueError("Resume requires an owned regular native evidence file, not a symlink")
    raw = path.read_bytes()
    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Resume rejects duplicate JSON fields: " + key)
            value[key] = item
        return value

    packet = json.loads(raw, object_pairs_hook=unique_fields)
    if not _finite_evidence(packet) or (packet.get("validation") or {}).get("finite_episode") is not True:
        raise ValueError("Resume requires finite original native evidence; sanitized nonfinite records cannot certify")
    current = _json_value(provenance)
    generation = packet.get("provenance") or {}
    # Validator revisions are allowed; native sources, dependencies and environment are not.
    for key in ("modules", "audit_dependencies", "project_root", "python_executable", "python_version",
                "canonical_python_executable", "using_canonical_executable", "working_directory", "package_versions",
                "package_file", "model_builder_module", "serializer", "MUJOCO_GL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
        if key not in generation or key not in current or generation[key] != current[key]:
            raise ValueError("Resume native generation provenance differs: " + key)
    if (generation.get("command", [])[:2] != current.get("command", [])[:2]
            or generation.get("using_canonical_executable") is not True
            or current.get("using_canonical_executable") is not True):
        raise ValueError("Resume requires the same canonical executable and native generation script path")
    _, inputs, native_inputs = fixture
    metadata = packet.get("fixture_inputs") or {}
    if (packet.get("profile") != profile.to_dict() or packet.get("target_case") != case or packet.get("dt_s") != dt
            or packet.get("kind") != "right_hand" or packet.get("moving_limb") != "RIGHT_HAND"
            or metadata.get("native_metadata") != native_inputs
            or any(metadata.get(key) != _json_value(value) for key, value in inputs.items())):
        raise ValueError("Resume native profile/case/timestep or compiled fixture inputs differ")
    digest = hashlib.sha256(raw).hexdigest()
    report_path = path.parent / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_bytes(), object_pairs_hook=unique_fields)
        for run in report.get("runs", []):
            if run.get("name") == path.stem and (run.get("evidence_sha256") != digest
                    or run.get("evidence_json") != str(path)):
                raise ValueError("Resume artifact differs from the saved report's evidence hash/path")
    saved = {key: value for key, value in packet.items() if key not in
             ("validation", "motion_audit", "integration_audit", "audit_source", "audit_provenance", "provenance")}
    saved["fixture_inputs"] = native_inputs
    return saved, generation, digest


def _run_case(suite, name, case, dt, output, provenance, profile, *, resume=False, fixture=None):
    path = output / f"{name}_{case}_{dt * 1000:g}ms.json"
    resumed = resume and (path.exists() or path.is_symlink())
    print(f"[{'RE-AUDIT' if resumed else 'EXECUTE'}] {path.stem}", flush=True)
    if not resume and (path.exists() or path.is_symlink()):
        raise FileExistsError("Native evidence already exists; use --resume instead of overwriting it")
    # Independently reconstructed geometry is used only for saved-state FK/COM.
    model, inputs, native_inputs = fixture or _fixture_metadata(profile, case, dt)
    generation, original_digest = provenance, None
    if resumed:
        result, generation, original_digest = _load_existing(path, provenance, profile, case, dt,
                                                             (model, inputs, native_inputs))
    else:
        result = _execute_case(profile, case, dt)
    finite = _finite_evidence(result)
    saved = _evidence_value(result)
    motion, integration, error = None, {}, None
    try:
        if not finite:
            raise ValueError("Nonfinite original native evidence before serialization")
        if (saved.get("profile") != profile.to_dict() or saved.get("dt_s") != dt
                or saved.get("target_case") != case or saved.get("fixture_inputs") != native_inputs):
            raise ValueError("Native result differs from reconstructed profile/case/timestep/fixture inputs")
        motion = audit_motion(model, saved)
        integration = _integration_checks(model, saved)
        if not _limit_checks(model, inputs, saved):
            raise ValueError("Original compiled motor/grip capacity limits differ from recorded evidence")
    except (ValueError, KeyError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    metrics = _metrics(model, saved, motion)
    certified = False
    try:
        certified = bool(finite and error is None and _case_verdict(saved, motion, integration, inputs))
        negative_proof = _negative_certificate(model, saved, case, native_inputs) if suite == "negative" and certified else None
    except (ValueError, KeyError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        negative_proof = False
    accepted = bool(certified and error is None and (suite != "negative" or negative_proof))
    if resumed and not accepted:
        raise ValueError("Saved native evidence failed current certification: " + (error or str(saved.get("status"))))
    summary = {"name": path.stem, "suite": suite, "profile_name": name, "case": case, "dt_s": dt,
               "profile": profile.to_dict(), "classification": saved.get("classification"),
               "status": saved.get("status"), "reason": saved.get("reason"),
               "geometric_feasible": (saved.get("assessment") or {}).get("geometric_feasible"),
               "native_success": saved.get("success"), "physical_success": certified and _physical_success(saved),
               "finite_episode": finite, "valid_source": bool(_valid_source(saved) and integration.get("initial")),
               "accepted": accepted, "diagnostic_evidence_accepted": accepted and not saved.get("success"),
               "negative_certificate_verified": negative_proof, "resumed": resumed,
               "resumed_from": {"path": path, "evidence_sha256": original_digest} if resumed else None,
               "steps": saved.get("steps"),
               "duration_s": saved.get("duration_s"), "audit_error": error,
               "fixture_fingerprints": {key: inputs[key] for key in ("model_xml_sha256", "compiled_model_sha256",
                   "compiled_geometry_sha256", "scene_sha256", "input_sha256", "compiled_rom_rad",
                   "motor_ceiling_Nm", "mass_kg", "target_offset_world_m")},
               "seed_qpos_sha256": _hash(inputs["seed_qpos"]), "metrics": metrics, "evidence_json": path}
    audit_path = path.with_suffix(".audit.json") if resumed else path
    if audit_path.is_symlink():
        raise ValueError("Audit output must not be a symlink")
    if resumed and hashlib.sha256(path.read_bytes()).hexdigest() != original_digest:
        raise ValueError("Native artifact changed during re-audit")
    _write_json(audit_path, {**({} if resumed else saved), "validation": summary, "provenance": generation,
                      "audit_provenance": provenance,
                      "fixture_inputs": {**inputs, "native_metadata": native_inputs}, "motion_audit": motion,
                      "integration_audit": integration,
                      "audit_source": {"method": "saved actual native poses; owned FK/COM only; no dynamics",
                                        "input_sha256": inputs["input_sha256"], "quality_verdict": None}}, compact=True)
    summary["evidence_sha256"] = original_digest if resumed else hashlib.sha256(path.read_bytes()).hexdigest()
    if resumed:
        summary.update(audit_json=audit_path, audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
    return summary


def _comparisons(runs):
    pairs, design, responses, capacities = [], [], [], []
    groups = {}
    for run in runs:
        groups.setdefault((run["profile_name"], run["case"]), []).append(run)
    for (profile, case), group in groups.items():
        if len({run["dt_s"] for run in group}) != 2:
            continue
        a, b = sorted(group, key=lambda run: -run["dt_s"])
        fields = ("classification", "status", "geometric_feasible", "physical_success")
        pairs.append({"profile": profile, "case": case,
                      "consistent": all(a[k] == b[k] for k in fields)
                                    and (a["physical_success"] or a["reason"] == b["reason"]),
                      "2ms": {k: a[k] for k in (*fields, "reason")},
                      "1ms": {k: b[k] for k in (*fields, "reason")},
                      "metric_differences": ({
                          "pelvis_net_m": b["metrics"]["pelvis"]["net_displacement_m"] - a["metrics"]["pelvis"]["net_displacement_m"],
                          "com_net_m": b["metrics"]["com"]["net_displacement_m"] - a["metrics"]["com"]["net_displacement_m"],
                          "root_net_deg": b["metrics"]["root_orientation"]["net_relative_angle_deg"]
                                          - a["metrics"]["root_orientation"]["net_relative_angle_deg"]}
                          if a.get("metrics") and b.get("metrics") else None)})
    primary = [run for run in runs if run["suite"] == "matrix"]
    for index, a in enumerate(primary):
        for b in primary[index + 1:]:
            if a["case"] != b["case"] or a["dt_s"] != b["dt_s"] or a["profile_name"] == b["profile_name"]:
                continue
            fa, fb = a["fixture_fingerprints"], b["fixture_fingerprints"]
            geometry_differs = any(a["profile"][key] != b["profile"][key] for key in LENGTHS)
            clean_a = {k: v for k, v in a["profile"].items() if k not in (*LENGTHS, "name", "grip_capacity")}
            clean_b = {k: v for k, v in b["profile"].items() if k not in (*LENGTHS, "name", "grip_capacity")}
            consistent = (clean_a == clean_b and all(fa[k] == fb[k] for k in
                          ("scene_sha256", "compiled_rom_rad", "motor_ceiling_Nm", "target_offset_world_m"))
                          and abs(fa["mass_kg"] - 78.3) < 1e-8 and abs(fb["mass_kg"] - 78.3) < 1e-8)
            if geometry_differs:
                consistent = consistent and fa["compiled_geometry_sha256"] != fb["compiled_geometry_sha256"]
                consistent = consistent and a["seed_qpos_sha256"] != b["seed_qpos_sha256"]
            else:
                consistent = consistent and fa["compiled_geometry_sha256"] == fb["compiled_geometry_sha256"]
            design.append({"a": a["name"], "b": b["name"], "consistent": bool(consistent)})
            ma, mb = a.get("metrics"), b.get("metrics")
            if not (ma and mb and a["physical_success"] and b["physical_success"]):
                continue
            if geometry_differs:
                elbow = max(abs(ma["actual_elbows"][key]["final_deg"] - mb["actual_elbows"][key]["final_deg"])
                            for key in ma["actual_elbows"])
                com = float(np.linalg.norm(np.asarray(ma["com"]["final_world_m"]) - mb["com"]["final_world_m"]))
                torque = float(np.max(np.abs(np.asarray(ma["final_torque_Nm"]) - mb["final_torque_Nm"])))
                path_differs = bool(ma["preparation_path_sha256"] and mb["preparation_path_sha256"]
                                    and ma["preparation_path_sha256"] != mb["preparation_path_sha256"])
                responses.append({"a": a["name"], "b": b["name"], "elbow_difference_deg": elbow,
                                  "com_difference_m": com, "torque_difference_Nm": torque,
                                  "preparation_path_differs": path_differs,
                                  "observable_response": path_differs and (elbow >= .5 or com >= .0005 or torque > 1.)})
            elif a["profile"]["grip_capacity"] != b["profile"]["grip_capacity"]:
                ratio = b["profile"]["grip_capacity"] / a["profile"]["grip_capacity"]
                checks = []
                for limb, hand in ma["support_loads"]["hands"].items():
                    other = mb["support_loads"]["hands"][limb]
                    checks.extend(hand["capacity"] is not None and other["capacity"] is not None
                                  and np.isclose(other["capacity"][key], ratio * hand["capacity"][key], rtol=1e-8)
                                  for key in ("min_N", "max_N"))
                capacities.append({"a": a["name"], "b": b["name"], "capacity_ratio": ratio,
                                   "capacity_scales": bool(checks and all(checks)),
                                   "identical_paths_allowed_if_limit_inactive": True})
    return {"timestep_pairs": pairs, "fixed_scene_design": design,
            "morphology_responses": responses, "grip_capacity_comparisons": capacities}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "matrix", "negative"), default="all")
    parser.add_argument("--profile", action="append", choices=tuple(envelope.study_profiles()))
    parser.add_argument("--case", action="append", choices=(*envelope.target_perturbations(), *NEGATIVES))
    parser.add_argument("--dt", "--timestep", type=float, choices=(.002, .001))
    parser.add_argument("--summary", action="store_true", help="Print a compact per-episode table")
    parser.add_argument("--resume", action="store_true", help="Re-audit matching native records read-only; execute only missing cases")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "morphology-envelope-stage5.2")
    args = parser.parse_args(argv)
    jobs = _jobs(args)
    if not jobs:
        parser.error("no profile/case combinations selected within this suite")
    if (set(args.profile or ()) - {job[1] for job in jobs}
            or set(args.case or ()) - {job[2] for job in jobs}):
        parser.error("a requested profile/case has no episodes within this suite and selection")
    outputs, output = (ROOT / "outputs").resolve(), args.output.resolve()
    if (not output.parent.is_dir() or output.exists() and not output.is_dir()
            or not output.is_relative_to(outputs) or output == outputs):
        parser.error("output must be a dedicated directory under this worktree's existing outputs")
    if any("-baseline-" in part.lower() or "-preserved-" in part.lower()
           for part in output.relative_to(outputs).parts):
        parser.error("output must not overwrite preservation baseline directories")
    output.mkdir(exist_ok=True)
    provenance = _provenance(list(sys.argv[1:] if argv is None else argv))
    profiles, runs, errors = envelope.study_profiles(), [], []
    fixtures = {}
    if args.resume:
        # Validate all existing artifacts before allowing any missing native episode.
        for suite, profile, case, dt in jobs:
            path = output / f"{profile}_{case}_{dt * 1000:g}ms.json"
            if not (path.exists() or path.is_symlink()):
                continue
            try:
                print(f"[CHECK EXISTING] {path.stem}", flush=True)
                fixture = _fixture_metadata(profiles[profile], case, dt)
                _load_existing(path, provenance, profiles[profile], case, dt, fixture)
                fixtures[(profile, case, dt)] = fixture
            except Exception as exc:
                errors.append({"profile": profile, "case": case, "dt_s": dt, "error": f"{type(exc).__name__}: {exc}"})
        if errors:
            for error in errors:
                print(f"[RESUME ERROR] {error['profile']}/{error['case']}: {error['error']}", file=sys.stderr, flush=True)
    ordered_jobs = sorted(jobs, key=lambda job: (job[1], job[2], job[3]) not in fixtures) if args.resume else jobs
    for suite, profile, case, dt in ([] if errors else ordered_jobs):
        try:
            options = {"resume": True, "fixture": fixtures.get((profile, case, dt))} if args.resume else {}
            runs.append(_run_case(suite, profile, case, dt, output, provenance, profiles[profile], **options))
        except Exception as exc:
            error = {"profile": profile, "case": case, "dt_s": dt, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            print(f"[EVIDENCE ERROR] {profile}/{case}/{dt:g}: {error['error']}", file=sys.stderr)
            if args.resume:
                break
    current = _provenance([])
    stable = all(_json_value(current[key]) == _json_value(provenance[key]) for key in
                 ("modules", "validator_sha256", "audit_dependencies"))
    primary = [run for run in runs if run["suite"] == "matrix"]
    expected = {(p, c, dt) for p in profiles for c in envelope.target_perturbations() for dt in (.002, .001)}
    complete = {(r["profile_name"], r["case"], r["dt_s"]) for r in primary} == expected
    comparisons = _comparisons(runs)
    success_geometries = {tuple(r["profile"][key] for key in LENGTHS) for r in primary if r["physical_success"]}
    flags = []
    if not stable:
        flags.append("SOURCE_HASHES_CHANGED")
    if errors or len(runs) != len(jobs):
        flags.append("MISSING_NATIVE_EVIDENCE")
    if any(not run["accepted"] for run in runs):
        flags.append("EPISODE_AUDIT_OR_NATIVE_EXECUTION_FAILED")
    if any(not pair["consistent"] for pair in comparisons["timestep_pairs"]):
        flags.append("TIMESTEP_QUALITATIVE_DISAGREEMENT")
    if any(not pair["consistent"] for pair in comparisons["fixed_scene_design"]):
        flags.append("FIXED_SCENE_OR_PROFILE_RECOMPUTATION_FAILED")
    if any(not pair["capacity_scales"] for pair in comparisons["grip_capacity_comparisons"]):
        flags.append("GRIP_CAPACITY_SCALING_FAILED")
    if complete and len(success_geometries) < 2:
        flags.append("INSUFFICIENT_DISTINCT_GEOMETRY_PHYSICAL_SUCCESSES")
    if complete and not any(pair["observable_response"] for pair in comparisons["morphology_responses"]):
        flags.append("NO_MEASURED_MORPHOLOGY_RESPONSE")
    acceptance = bool(runs and not flags)
    report = {"stage": "5.2", "suite": args.suite, "complete_primary_matrix": complete,
              "partial_scope": not complete, "requested_case_count": len(jobs), "primary_case_count": len(primary),
              "negative_case_count": sum(r["suite"] == "negative" for r in runs),
              "physical_success_count": sum(r["physical_success"] for r in primary),
              "resumed_case_count": sum(r.get("resumed", False) for r in runs),
              "executed_case_count": sum(not r.get("resumed", False) for r in runs),
              "diagnostic_evidence_accepted_count": sum(r.get("diagnostic_evidence_accepted", False) for r in runs),
              "distinct_success_geometry_count": len(success_geometries),
              "classification_counts": dict(Counter(r["classification"] for r in runs)),
              "source_hashes_stable_during_run": stable, "acceptance": acceptance, "passed": acceptance,
              "flags": flags, "runs": runs, "errors": errors, "comparisons": comparisons, "provenance": provenance,
              "protocol": {"benchmark": "boulder_v1.morphology_envelope.run_envelope_case", "keep_samples": True,
                            "source_static_s": 2., "source_replay": False, "metadata_reconstruction_physics": False,
                            "resume": "original native JSON untouched; current audit sidecar; matching native hashes/environment/compiled inputs",
                            "authenticity": "consistency and saved hashes, not cryptographic authentication of unmanifested native records",
                           "target_offsets_local_m": envelope.target_perturbations(),
                           "scope": "bounded policy search and chosen native execution only; no global impossibility proof",
                           "acceptance": "finite original evidence, native source/clock/forces/integration, paired outcomes; "
                                         "complete matrix additionally needs distinct geometry successes and measured response",
                           "response_thresholds": {"elbow_deg": .5, "com_m": .0005, "torque_Nm": 1.},
                           "coefficients": "geometry policy, not controller gains or relaxed limits"},
              "visual_quality_verdict": None, "report_json": output / "report.json"}
    _write_json(output / "report.json", report)
    if args.summary:
        print("profile case dt classification accepted steps pelvis_net_m com_net_m")
        for run in runs:
            metrics = run.get("metrics") or {}
            print(f"{run['profile_name']} {run['case']} {run['dt_s'] * 1000:g}ms {run['classification']} "
                  f"{run['accepted']} {run['steps']} {metrics.get('pelvis', {}).get('net_displacement_m')} "
                  f"{metrics.get('com', {}).get('net_displacement_m')}")
        print(json.dumps(_json_value({k: report[k] for k in ("complete_primary_matrix", "partial_scope",
              "acceptance", "flags", "classification_counts", "physical_success_count", "report_json")}), allow_nan=False))
    else:
        print(json.dumps(_json_value(report), allow_nan=False, indent=2))
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
