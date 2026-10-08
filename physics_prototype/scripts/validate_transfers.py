#!/usr/bin/env python3
"""Stage 5 evidence for four explicit primitives, not a planner or personalization.

Default ALL runs RH, LH, native LF and RH -> LH at 2/1 ms, the maintained
Stage 4 negatives plus Stage 5 negatives at 2 ms, two fixed-scene profiles at
2 ms, and tiny scratch-seed perturbations on both hands at 2/1 ms. --dt selects
one timestep throughout. Profile failures are diagnostics, not physical passes.
--render exports the selected family episodes at 640x480/20 fps with EGL from
their own detached callbacks. The returned source hold is evidence, not replay.
Full compact JSON always retains states, references, samples and exact reasons.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_transition import CANONICAL_PYTHON, _json_value, _write_json

# Fixture metadata only. Execution and rendering below use the selected limb.
GOALS = {
    "right_hand": ("RIGHT_HAND", "right_hand", "reach_target"),
    "left_hand": ("LEFT_HAND", "left_hand", "left_reach_target"),
    "foot": ("LEFT_FOOT", "left_foot", "foot_target"),
}
FAMILY = (*GOALS, "sequence")
STAGE4_NEGATIVES = {
    "unreachable": "REACH_INFEASIBLE",
    "orientation_invalid": "CAPTURE_FAILURE",
    "support_loss": "CONTACT_LOSS",
    "grip_after_capture": "GRIP_FAILURE",
}
NEGATIVES = {
    **STAGE4_NEGATIVES,
    "hand_support": "THREE_POINT_SUPPORT_FAILURE",
    "foot_missing_step": "INELIGIBLE_TARGET",
    "foot_low_friction": "SUPPORT_INFEASIBLE",
    "foot_support_loss": "CONTACT_LOSS",
    "sequence_second_unreachable": "REACH_INFEASIBLE",
}
PROFILE_CASES = ("compact_strong", "long_reach_lower_grip")
PERTURBATIONS = {"perturbed_right_hand": ("RIGHT_HAND", 1e-4),
                 "perturbed_left_hand": ("LEFT_HAND", -1e-4)}
SENSITIVITY = (*PROFILE_CASES, *PERTURBATIONS)


def _jobs(args):
    dts = (args.dt,) if args.dt is not None else (.002, .001)
    diagnostic_dt = args.dt if args.dt is not None else .002
    jobs = []
    if args.suite in ("all", "family"):
        jobs.extend(("family", case, dt) for case in FAMILY for dt in dts)
    if args.suite in ("all", "negative"):
        jobs.extend(("negative", case, diagnostic_dt) for case in NEGATIVES)
    if args.suite in ("all", "sensitivity"):
        jobs.extend(("sensitivity", case, diagnostic_dt) for case in PROFILE_CASES)
        jobs.extend(("sensitivity", case, dt) for case in PERTURBATIONS for dt in dts)
    return [job for job in jobs if not args.case or job[1] in args.case]


def _execute_case(case, dt, observer):
    from boulder_v1 import foot_transfer, hand_family, single_hand, transfers
    from boulder_v1.contact_ik import solve_contact_pose
    from boulder_v1.mjcf_builder import build_mjcf
    from boulder_v1.schema import Affordance, Limb
    from boulder_v1.static_control import execute_static_hold
    from boulder_v1.static_state import initialize_static_reference

    if case in FAMILY or case == "sequence_second_unreachable":
        return transfers.run_transfer_benchmark(
            timestep=dt, kind="sequence" if case == "sequence_second_unreachable" else case,
            negative="second_unreachable" if case == "sequence_second_unreachable" else None,
            observer=observer, keep_samples=True)
    if case in STAGE4_NEGATIVES:
        # Do not opt Stage 4 into endpoint_settle or replace its first-eligible gate.
        return single_hand.run_single_hand_benchmark(timestep=dt, scenario=case,
                                                     observer=observer, keep_samples=True)
    if case in SENSITIVITY:
        from scripts.view_scene import PROFILES

        limb, perturbation = PERTURBATIONS.get(case, ("RIGHT_HAND", 0.))
        result = hand_family.run_hand_family_benchmark(
            timestep=dt, limb=Limb(limb), profile=PROFILES[case] if case in PROFILE_CASES else None,
            pose_perturbation_rad=perturbation, observer=observer, keep_samples=True)
        result["diagnostic"] = {
            "fixed_holds": True, "targets_retuned": False, "controller_retuned": False,
            "personalization_claim": False, "pose_perturbation_rad": perturbation,
            "seed_joint": "waist_yaw", "scope": "fixed-scene geometric admission and native motion",
        }
        return result
    if case not in NEGATIVES:
        raise ValueError(f"Unknown transfer case: {case}")

    import mujoco

    fixture_options = {"target_friction": .005} if case == "foot_low_friction" else {}
    model, data, scene, profile, seed = foot_transfer.make_foot_transfer_fixture(dt, **fixture_options)
    original_profile = profile
    changes = dict(fixture_options)
    if case == "hand_support":
        profile = replace(profile, name="stage5_weak_hand_support", grip_capacity=60.)
        changes["grip_capacity_N"] = 60.
    elif case == "foot_missing_step":
        target = replace(scene.region("foot_target"), affordances=frozenset({Affordance.GRASP}))
        scene = replace(scene, contact_regions=tuple(target if r.id == target.id else r
                                                    for r in scene.contact_regions))
        changes["target_affordances"] = target.affordances
    if case in ("hand_support", "foot_missing_step"):
        # Compile the declared diagnostic scene/profile; source geometry is untouched.
        tree = ET.fromstring(build_mjcf(scene, profile))
        tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        data = mujoco.MjData(model)
    retarget = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
    setup = {"retarget": retarget, "profile": profile.to_dict(), "dt_s": dt,
             "negative_fixture": changes, "initial_static": None}
    if not retarget.admitted:
        return {**setup, "success": False, "status": "INITIALIZATION_FAILURE", "reason": retarget.reason,
                "steps": 0, "final_reference": None}
    reference, manager = initialize_static_reference(model, data, scene, profile, retarget.qpos,
                                                      scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager,
                                  duration=2., settle=1., score_window=.5, keep_samples=False)
    setup.update(initial_static=initial, initial_reference=reference)
    if not initial["success"]:
        return {**setup, "success": False, "status": "INITIALIZATION_FAILURE", "reason": initial["reason"],
                "steps": 0, "final_reference": None}
    if case == "hand_support":
        moving, source, target = GOALS["right_hand"]
        request = single_hand.SingleHandRequest(Limb(moving), source, target,
                                               acquisition_policy="endpoint_settle")
        result = single_hand.execute_single_hand(model, data, scene, profile, reference, manager,
                                                 request, observer=observer, keep_samples=True)
        if result["status"] == NEGATIVES[case] and result["steps"] == 0:
            from boulder_v1.contact import effective_grip_capacity
            from boulder_v1.motion_support import estimate_support_torques

            supports = {limb: hold for limb, hold in reference.contact_intent.items() if limb != request.limb}
            diagnostic = {"scope": "scratch source-pose force allocation with original capacity; never applied",
                          "source_physically_valid": initial["success"], "capacity_check_bypassed_in_execution": False}
            try:
                estimate = estimate_support_torques(model, scene, original_profile, data.qpos, supports)
                order = [limb for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT, Limb.LEFT_HAND, Limb.RIGHT_HAND)
                         if limb in supports]
                remaining = next(limb for limb in supports if limb.is_hand)
                required = float(np.linalg.norm(estimate["forces_world_N"][order.index(remaining)]))
                capacity = effective_grip_capacity(profile, scene.region(supports[remaining]),
                                                   scene.region(supports[remaining]).normal)
                diagnostic.update(available=True, remaining_limb=remaining,
                                  required_remaining_hand_load_N=required, effective_capacity_N=capacity,
                                  margin_N=capacity - required, estimated_feedforward=estimate)
            except ValueError as exc:
                diagnostic.update(available=False, reason=str(exc))
            result["support_capacity_diagnostic"] = diagnostic
    else:
        moving, source, target = GOALS["foot"]
        request = foot_transfer.FootRequest(Limb(moving), source, target)
        result = foot_transfer.execute_foot_transfer(
            model, data, scene, profile, reference, manager, request, observer=observer, keep_samples=True,
            fault="support_loss" if case == "foot_support_loss" else None)
    result.update(setup)
    if case == "foot_low_friction":
        result["diagnostic_scope"] = (
            "Candidate foot friction demand is not admissible at declared target mu=0.005. "
            "Native support-estimate preflight rejection, not an executed zero-normal-force landing or global infeasibility.")
    return result


def _move_context(case, row):
    from boulder_v1.schema import Limb

    if case == "sequence":
        index = row["move_index"]
        if row["move_count"] != 2 or index not in (0, 1):
            raise ValueError("Unexpected declared sequence observer metadata")
        kind = ("right_hand", "left_hand")[index]
    else:
        kind = case
    limb, source, target = GOALS[kind]
    return Limb(row.get("moving_limb", limb)), row.get("source", source), row.get("target", target)


def _site_measurement(mujoco, model, data, limb, target):
    source = data.site(f"{limb.value.lower()}_site")
    goal = data.site(("site_" if limb.is_hand else "site_step_") + target)
    jp, tp = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, None, source.id)
    mujoco.mj_jacSite(model, data, tp, None, goal.id)
    return {"gap_m": float(np.linalg.norm(source.xpos - goal.xpos)),
            "orientation": float(np.clip(source.xmat.reshape(3, 3)[:, 2]
                                          @ goal.xmat.reshape(3, 3)[:, 2], -1., 1.)),
            "relative_speed_m_s": float(np.linalg.norm((jp - tp) @ data.qvel))}


def _draw_hud(image, row, dt, limb, source, target, reference=None, actual=None, pose_metrics=None):
    from PIL import ImageDraw, ImageFont
    from boulder_v1.contact import CAPTURE_DISTANCE
    from boulder_v1.schema import Limb

    limb = Limb(limb)
    labels = {Limb.LEFT_HAND.value: "LH", Limb.RIGHT_HAND.value: "RH",
              Limb.LEFT_FOOT.value: "LF", Limb.RIGHT_FOOT.value: "RF"}
    def number(value, precision=3, scale=1.):
        return f"{scale * value:.{precision}f}" if value is not None and math.isfinite(value) else "n/a"

    draw, font = ImageDraw.Draw(image, "RGBA"), ImageFont.load_default(size=10)
    width, spacing = image.width - 24, 13
    heading = [f"Stage5 | PHYSICAL | dt={dt * 1000:g}ms | {limb.value} {source} -> {target}",
               f"{row['status']} | {row['phase']} | t={number(row.get('time_s'))}s "
               f"elapsed={number(row.get('elapsed_s'))}s | step={row.get('steps', 0)}"]
    if "move_index" in row:
        heading[0] += f" | move {row['move_index'] + 1}/{row['move_count']}"
    if not row.get("pose_available", True):
        heading.append("POSE UNAVAILABLE: actual state/measurement invalid; see JSON")
    command, ready = row.get("command") or {}, row.get("readiness") or {}
    metrics = pose_metrics or {}
    actual = actual if actual is not None else row.get("capture_measurement") or {}
    margin = actual.get("capture_margin_m")
    if limb.is_hand and margin is None and actual.get("gap_m") is not None:
        margin = CAPTURE_DISTANCE - actual["gap_m"]
    lines = []
    for pair in ((Limb.LEFT_HAND, Limb.RIGHT_HAND), (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)):
        lines.append("Contacts: " + "  ".join(f"{labels[l.value]}={row.get('contacts', {}).get(l.value, 'none')}"
                                             for l in pair))
    lines.extend([
        f"Root v={number(row.get('root_linear_m_s'), 4)}m/s w={number(row.get('root_angular_rad_s'), 4)}rad/s "
        f"hinge max/rms={number(row.get('joint_max_rad_s'), 4)}/{number(ready.get('rms_hinge_speed'), 4)}rad/s",
        f"q/ref hinge rms/max={number(metrics.get('hinge_error_rms_rad'), 4)}/"
        f"{number(metrics.get('hinge_error_max_rad'), 4)}rad ref v={number(metrics.get('reference_speed_max_rad_s'), 4)}rad/s",
        f"{labels[limb.value]} goal={target} | ref geometry is NOT physical support",
        f"Ref error={number((reference or {}).get('gap_m'), 3, 1000)}mm "
        f"facing={number((reference or {}).get('orientation'), 4)} "
        f"speed={number((reference or {}).get('relative_speed_m_s'), 4)}m/s",
        f"Actual error={number(actual.get('gap_m'), 3, 1000)}mm margin={number(margin, 3, 1000)}mm "
        f"facing={number(actual.get('orientation'), 4)} speed={number(actual.get('relative_speed_m_s'), 4)}m/s",
        f"Trajectory tracking={number(row.get('tracking_error_m'), 3, 1000)}mm "
        f"signed target distance={number(row.get('foot_distance_m'), 3, 1000)}mm",
    ])
    if limb.is_hand:
        moving = row.get("hands", {}).get(limb.value, {})
        captured = bool(moving.get("active") and moving.get("region_id") == target)
        released = bool(not moving.get("active") or moving.get("region_id") != source)
        lines.append(f"{labels[limb.value]} released={released} captured={captured} "
                     f"active={moving.get('active', False)} hold={moving.get('region_id') or 'none'}")
    else:
        contact, touch, acquired = row.get("source_contact") or {}, row.get("touchdown") or {}, row.get("acquisition") or {}
        lines.extend([
            f"{labels[limb.value]} released={row.get('released', False)} source n={contact.get('contact_count', 'n/a')} "
            f"Fn={number(contact.get('normal_force_N'), 1)}N touchdown Fn={number(touch.get('normal_force_N'), 2)}N",
            f"Native acquired={bool(acquired)} t={number(acquired.get('time_s'))}s "
            f"sustained={number(acquired.get('sustained_s'))}s Fn={number(acquired.get('normal_force_N'), 1)}N",
        ])
    for hand_limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
        hand = row.get("hands", {}).get(hand_limb.value, {})
        lines.append(f"{labels[hand_limb.value]}: hold={hand.get('region_id') or 'none'} "
                     f"active={hand.get('active', False)} valid={hand.get('valid', False)} "
                     f"load/cap={number(hand.get('load'), 1)}/{number(hand.get('capacity'), 1)}N")
    for foot_limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
        foot = row.get("feet", {}).get(foot_limb.value, {})
        lines.append(f"{labels[foot_limb.value]}: Fn={number(foot.get('normal_force'), 1)}N "
                     f"Ft={number(foot.get('tangential_force'), 1)}N slip v={number(foot.get('tangential_speed'), 4)}m/s "
                     f"contact={foot.get('contacting', False)} support={foot.get('supporting', False)} "
                     f"slipping={foot.get('slipping', False)}")
    torques, utilization = command.get("commanded_Nm") or [], command.get("utilization") or []
    lines.extend([
        f"Motor |tau|max={number(max(map(abs, torques), default=None), 2)}Nm "
        f"util={number(max(utilization, default=None), 1, 100)}% "
        f"saturated={sum(command.get('saturated') or [])}/{len(utilization)}",
        f"Readiness={ready.get('ready', False)} sustained={number(ready.get('duration'))}s",
    ])
    # Only prose may be shortened; mandatory metrics stay at the fixed 10px font.
    for label, reason in (("State", ready.get("reason", "n/a")), ("Outcome", row.get("reason", "n/a"))):
        text = f"{label}: {reason}"
        if draw.textbbox((0, 0), text, font=font)[2] > width:
            suffix = "... (full reason: JSON)"
            while reason and draw.textbbox((0, 0), f"{label}: {reason}{suffix}", font=font)[2] > width:
                reason = reason[:-1]
            text = f"{label}: {reason}{suffix}"
        lines.append(text)
    blocks = []
    for block in (heading, lines):
        wrapped = []
        for line in block:
            while draw.textbbox((0, 0), line, font=font)[2] > width:
                cut = len(line)
                while draw.textbbox((0, 0), line[:cut], font=font)[2] > width:
                    cut -= 1
                boundary = line.rfind(" ", 0, cut + 1)
                cut = boundary if boundary > 0 else cut
                if cut <= 0:
                    raise ValueError("HUD canvas too narrow for the required font")
                wrapped.append(line[:cut])
                line = line[cut:].lstrip()
            wrapped.append(line)
        blocks.append(wrapped)
    heading, lines = blocks
    bottom = image.height - len(lines) * spacing - 12
    if bottom < 8 + len(heading) * spacing + 12:
        raise ValueError("Mandatory HUD fields do not fit the canvas at font size 10")
    for block, y in ((heading, 8), (lines, bottom)):
        draw.rectangle([(6, y - 3), (image.width - 6, y + len(block) * spacing + 3)],
                       fill=(12, 18, 28, 220), outline=(50, 90, 140, 240), width=1)
        for index, line in enumerate(block):
            draw.text((12, y + index * spacing), line, font=font, fill=(210, 230, 245))
    return image


def _episode_finite(result):
    """Check measured state/reference arrays before JSON turns nonfinite into null."""
    observed = False
    episodes = result.get("moves") or [result]
    for episode in [result.get("initial_static") or {}, *episodes]:
        for state in (episode.get("initial_state"), episode.get("final_state"), episode.get("retarget")):
            if state is None:
                continue
            state = vars(state) if not isinstance(state, dict) else state
            if state.get("finite") is False:
                return False
            for key in ("time", "qpos", "qvel", "ctrl", "qacc_warmstart", "integration_state"):
                value = state.get(key)
                if value is not None:
                    observed = True
                    if not np.isfinite(np.asarray(value, dtype=float)).all():
                        return False
        for row in episode.get("samples", []):
            for key in ("time_s", "qpos", "qvel", "q_ref", "qd_ref", "ctrl"):
                if row.get(key) is not None:
                    observed = True
                    if not np.isfinite(np.asarray(row[key], dtype=float)).all():
                        return False
    return observed


def _physical_success(result):
    moves = result.get("moves")
    initial = result.get("initial_static")
    return bool(result.get("success") and result.get("status") == "SUCCESS"
                and result.get("final_reference") is not None
                and (initial is None or initial.get("success", False))
                and (bool(moves) and all(_physical_success(move) for move in moves) if moves is not None else
                     (result.get("readiness") or {}).get("ready", False)))


def _case_verdict(suite, case, result):
    physical, finite = _physical_success(result), _episode_finite(result)
    if suite == "family":
        return physical and finite
    if suite == "sensitivity":
        from boulder_v1.single_hand import SingleHandStatus

        known = {status.value for status in SingleHandStatus}
        valid = result.get("status") in known and bool(result.get("reason"))
        valid = valid and (physical if result.get("success") else
                          result.get("status") != "SUCCESS" and result.get("final_reference") is None
                          and not (result.get("readiness") or {}).get("ready", False))
        return bool(finite and valid)
    accepted = result.get("status") == NEGATIVES[case] and not result.get("success") and finite
    accepted = accepted and result.get("final_reference") is None
    accepted = accepted and not (result.get("readiness") or {}).get("ready", False)
    if case in ("hand_support", "foot_missing_step", "foot_low_friction"):
        accepted = accepted and bool((result.get("initial_static") or {}).get("success"))
        accepted = accepted and result.get("steps") == 0 and not result.get("released")
        accepted = accepted and result.get("final_reference") is None
        if case != "foot_missing_step":
            accepted = accepted and not (result.get("support_admission") or {}).get("admitted")
    elif case == "foot_support_loss":
        accepted = accepted and result.get("steps", 0) > 0 and result.get("final_reference") is None
    elif case == "sequence_second_unreachable":
        moves = result.get("moves") or []
        accepted = accepted and result.get("completed_moves") == 1 and result.get("failed_index") == 1
        accepted = accepted and len(moves) == 2
        if accepted:
            first, second = moves
            accepted = (_physical_success(first) and second.get("steps") == 0 and not second.get("released")
                        and _json_value(first["final_state"]) == _json_value(second["initial_state"])
                        and _json_value(second["final_state"]) == _json_value(result["final_state"]))
    return bool(accepted)


def _move_summary(move):
    keys = ("moving_limb", "source", "target", "status", "reason", "success", "steps", "duration_s",
            "phases", "events", "release_time_s", "released", "capture", "first_eligible", "capture_error_m",
            "capture_margin_m", "capture_time_s", "reach_start_time_s", "readiness_time_s", "readiness",
            "release", "touchdown", "first_support", "acquisition", "acquisition_time_s",
            "foot_acquisition_time_s", "load_start_time_s", "load_complete_time_s", "load_duration_s",
            "source_contact_changes", "three_point", "actuator_utilization_max", "foot_support_fraction",
            "foot_slip_max_m_s", "foot_loads", "remaining_hand_load_max_N", "support_admission",
            "final_contacts", "new_contacts", "guard_failure", "preflight", "max_tracking_error_m",
            "support_capacity_diagnostic")
    summary = {key: move[key] for key in keys if key in move}
    capture = move.get("capture") or {}
    summary.setdefault("capture_error_m", capture.get("gap_m"))
    summary.setdefault("capture_margin_m", capture.get("capture_margin_m"))
    summary["final_reference_admitted"] = move.get("final_reference") is not None
    rows = move.get("samples") or []
    summary["hand_load_max_N"] = dict(move.get("hand_load_max_N") or {})
    summary["hand_margin_min_N"] = {}
    for limb in ("LEFT_HAND", "RIGHT_HAND"):
        hands = [row["hands"][limb] for row in rows if limb in row.get("hands", {})]
        summary["hand_load_max_N"].setdefault(limb, max((h["load"] for h in hands
                                                       if h.get("load") is not None), default=None))
        summary["hand_margin_min_N"][limb] = min((h["capacity"] - h["load"] for h in hands
            if h.get("active") and h.get("capacity") is not None and h.get("load") is not None), default=None)
    summary["endpoint_contacts"] = (move.get("terminal_observation") or {}).get("contacts")
    summary["endpoint_feet"] = (move.get("terminal_observation") or {}).get("feet")
    return summary


def _run_case(suite, case, dt, output, render, provenance, baselines):
    name = f"{case}_{dt * 1000:g}ms"
    evidence_path, observer_path = output / f"{name}.json", output / f"{name}_observer.json"
    video_path, endpoint_path = output / f"{name}.mp4", output / f"{name}_endpoint.png"
    renderer = writer = render_model = scratch = reference_data = camera = None
    observations, render_errors = [], []
    frame_count = 0

    def render_error(label, exc):
        error = f"{name} {label}: {type(exc).__name__}: {exc}"
        render_errors.append(error)
        print(f"[RENDER ERROR] {error}", file=sys.stderr)

    def observe(row, detached_model, detached_data):
        nonlocal renderer, writer, render_model, scratch, reference_data, camera, frame_count
        observed = {"row": _json_value(row), "reference_goal_measurement": None,
                    "actual_goal_measurement": None, "state_reference_metrics": None,
                    "rendered": False, "frame_index": None}
        observations.append(observed)
        try:
            import imageio.v2 as iio
            import mujoco
            from PIL import Image
            from boulder_v1.contact import CAPTURE_DISTANCE
            from scripts.render_demo import _configure_camera, _render_frame

            limb, source, target = _move_context(case, row)
            observed.update(moving_limb=limb.value, source=source, target=target)
            if renderer is None:
                render_model = copy.copy(detached_model)
                scratch, reference_data = mujoco.MjData(render_model), mujoco.MjData(render_model)
                camera = mujoco.MjvCamera()
                _configure_camera(camera, render_model)
                renderer = mujoco.Renderer(render_model, height=480, width=640)
                writer = iio.get_writer(str(video_path), fps=20, codec="libx264", pixelformat="yuv420p")
            reference, actual, pose = None, None, None
            finite_pose = row.get("pose_available", False) and np.isfinite(detached_data.qpos).all()
            finite_pose = finite_pose and np.isfinite(detached_data.qvel).all()
            if finite_pose:
                image = _render_frame(mujoco, render_model, detached_data, scratch, renderer, camera)
                actual = (row.get("capture_measurement") if limb.is_hand else
                          _site_measurement(mujoco, render_model, scratch, limb, target))
                actual = _json_value(actual)
                if actual is not None:
                    if limb.is_hand and actual.get("gap_m") is not None:
                        actual["capture_margin_m"] = CAPTURE_DISTANCE - actual["gap_m"]
                    elif limb.is_foot:
                        actual["signed_geom_distance_m"] = row.get("foot_distance_m")
                qref, qdref = row.get("q_ref"), row.get("qd_ref")
                if qref is not None and qdref is not None and np.isfinite(qref).all() and np.isfinite(qdref).all():
                    reference_data.qpos[:] = qref
                    reference_data.qvel[:] = qdref
                    reference_data.eq_active[:] = False
                    mujoco.mj_forward(render_model, reference_data)
                    reference = _site_measurement(mujoco, render_model, reference_data, limb, target)
                    joints = render_model.actuator_trnid[:, 0]
                    qi, vi = render_model.jnt_qposadr[joints], render_model.jnt_dofadr[joints]
                    delta = detached_data.qpos[qi] - np.asarray(qref)[qi]
                    pose = {"hinge_error_rms_rad": float(np.sqrt(np.mean(delta ** 2))),
                            "hinge_error_max_rad": float(np.max(np.abs(delta))),
                            "reference_speed_max_rad_s": float(np.max(np.abs(np.asarray(qdref)[vi])))}
            else:
                image = Image.new("RGB", (640, 480), color=(10, 14, 22))
            _draw_hud(image, row, dt, limb, source, target, reference, actual, pose)
            writer.append_data(np.asarray(image))
            observed.update(reference_goal_measurement=_json_value(reference), actual_goal_measurement=_json_value(actual),
                            state_reference_metrics=_json_value(pose), rendered=True, frame_index=frame_count)
            frame_count += 1
            if row["terminal"]:
                image.save(endpoint_path)
        except Exception as exc:
            render_error("frame", exc)
            raise

    try:
        result = _execute_case(case, dt, observe if render else None)
    except Exception as exc:
        result = {"success": False, "status": "EXECUTION_ERROR", "reason": f"{type(exc).__name__}: {exc}",
                  "steps": None, "duration_s": None, "samples": [], "final_reference": None,
                  "last_observation": observations[-1]["row"] if observations else None}
        print(f"[EXECUTION ERROR] {name}: {result['reason']}", file=sys.stderr)
    finally:
        for label, resource in (("video close", writer), ("renderer close", renderer)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    render_error(label, exc)
    if render and not render_errors and (not frame_count or not video_path.is_file()
            or not video_path.stat().st_size or not endpoint_path.is_file()
            or not observations or not observations[-1]["row"]["terminal"]):
        render_error("export", RuntimeError("Missing video, endpoint, or actual terminal observation"))
    rendering = {"requested": render, "success": bool(frame_count and not render_errors) if render else None,
                 "observer_only": True, "replayed": False, "resolution": [640, 480], "fps": 20,
                 "observer_interval_s": .05, "frame_count": frame_count, "observation_count": len(observations),
                 "errors": render_errors, "video": video_path if render and not render_errors else None,
                 "endpoint": endpoint_path if render and endpoint_path.is_file() else None,
                 "observer_json": observer_path if render else None}
    expected = NEGATIVES[case] if suite == "negative" else "SUCCESS"
    if case == "long_reach_lower_grip":
        expected = "CONTROL_FAILURE"
    physical, accepted = _physical_success(result), _case_verdict(suite, case, result)
    summary = {"name": name, "suite": suite, "case": case, "dt_s": dt, "expected_status": expected,
               "expected_status_is_gate": suite != "sensitivity", "observed_expected_status": result["status"] == expected,
               "status": result["status"], "reason": result["reason"], "success": result["success"],
               "physical_success": physical, "finite_episode": _episode_finite(result), "accepted": accepted,
               "expected_failure_observed": accepted if suite == "negative" else None,
               "completed_moves": result.get("completed_moves", int(physical)), "failed_index": result.get("failed_index"),
               "steps": result.get("steps", 0), "duration_s": result.get("duration_s"),
               "initial_static": {key: (result.get("initial_static") or {}).get(key)
                                  for key in ("success", "status", "reason", "readiness")},
               "profile": result.get("profile"), "diagnostic": result.get("diagnostic"),
               "moves": [_move_summary(move) for move in result.get("moves", [result])],
               "evidence_json": evidence_path, "rendering": rendering}
    if render:
        _write_json(observer_path, {"name": name, "rendering": rendering, "observations": observations}, compact=True)
    _write_json(evidence_path, {**result, "validation": summary, "provenance": provenance,
                               "preservation_baselines": baselines}, compact=True)
    return summary


def _provenance(argv):
    import boulder_v1
    from boulder_v1 import contact_ik, foot_transfer, hand_family, mjcf_builder, single_hand, transfers

    versions = {}
    for package in ("boulder-prototype-v1", "mujoco", "numpy", "Pillow", "imageio", "imageio-ffmpeg"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    modules = {}
    for module in (contact_ik, foot_transfer, hand_family, mjcf_builder, single_hand, transfers):
        path = Path(module.__file__).resolve()
        modules[module.__name__] = {"file": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"project_root": ROOT, "python_executable": sys.executable, "python_version": sys.version,
            "canonical_python_executable": CANONICAL_PYTHON,
            "using_canonical_executable": Path(sys.executable) == Path(CANONICAL_PYTHON),
            "command": [sys.executable, str(Path(__file__).resolve()), *argv], "working_directory": Path.cwd(),
            "package_versions": versions, "package_file": boulder_v1.__file__, "modules": modules,
            "model_builder_module": mjcf_builder.__name__, "serializer": "scripts.validate_transition._json_value",
            "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}


def _baselines():
    baselines = {}
    for label in ("contacts", "controller", "transition"):
        path = ROOT / "outputs" / f"stage5-baseline-{label}" / "report.json"
        baselines[label] = {"report_json": path, "available": path.is_file(), "rerun": False,
                            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
    return baselines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("all", "family", "negative", "sensitivity"), default="all")
    parser.add_argument("--dt", type=float, choices=(.002, .001), help="seconds; default family/perturbations at both")
    parser.add_argument("--case", action="append", choices=(*FAMILY, *NEGATIVES, *SENSITIVITY),
                        help="select cases within the suite; repeatable")
    parser.add_argument("--render", action="store_true", help="EGL exports from selected family runs only")
    parser.add_argument("--summary", action="store_true", help="concise stdout; full per-case JSON is always saved")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transfers-stage5")
    args = parser.parse_args(argv)
    jobs = _jobs(args)
    if not jobs:
        parser.error("no cases selected within this suite")
    if args.render and not any(suite == "family" for suite, _, _ in jobs):
        parser.error("--render requires selected family cases (--suite family or all)")
    output = args.output.resolve()
    if not output.parent.is_dir() or output.exists() and not output.is_dir():
        parser.error("output must be a directory with an existing parent")
    outputs = ROOT / "outputs"
    if not output.is_relative_to(outputs) or output == outputs:
        parser.error("output must be a dedicated generated directory under this worktree's outputs")
    if any(part.startswith(("stage4-baseline-", "stage4-preserved-", "stage5-baseline-", "stage5-preserved-"))
           for part in output.relative_to(outputs).parts):
        parser.error("output must not overwrite preservation baseline directories")
    output.mkdir(exist_ok=True)
    if args.render:
        os.environ["MUJOCO_GL"] = "egl"
    provenance, baselines = _provenance(list(sys.argv[1:] if argv is None else argv)), _baselines()
    runs, errors = [], []
    for suite, case, dt in jobs:
        try:
            run = _run_case(suite, case, dt, output, args.render and suite == "family", provenance, baselines)
            runs.append(run)
            if run["status"] == "EXECUTION_ERROR":
                errors.append({"case": run["name"], "error": run["reason"]})
        except Exception as exc:
            error = {"case": f"{case}_{dt * 1000:g}ms", "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            print(f"[EVIDENCE ERROR] {error['case']}: {error['error']}", file=sys.stderr)
    family = [run for run in runs if run["suite"] == "family"]
    negatives = [run for run in runs if run["suite"] == "negative"]
    sensitivity = [run for run in runs if run["suite"] == "sensitivity"]
    family_count = sum(suite == "family" for suite, _, _ in jobs)
    negative_count = sum(suite == "negative" for suite, _, _ in jobs)
    sensitivity_count = sum(suite == "sensitivity" for suite, _, _ in jobs)
    rendered = [run for run in runs if run["rendering"]["requested"]]
    render_errors = [error for run in rendered for error in run["rendering"]["errors"]]
    acceptance = len(runs) == len(jobs) and not errors and all(run["accepted"] for run in runs)
    rendering_success = (len(rendered) == family_count and all(run["rendering"]["success"] for run in rendered)
                         if args.render else None)
    margins = [move["capture_margin_m"] for run in family for move in run["moves"]
               if move.get("capture_margin_m") is not None]
    report = {
        "stage": 5, "suite": args.suite, "scope": "Four explicit primitives on a fixed fixture; not route planning or personalization",
        "provenance": provenance, "preservation_baselines": baselines, "requested_case_count": len(jobs),
        "physical_acceptance": len(family) == family_count and all(run["accepted"] for run in family) if family_count else None,
        "physical_success_count": sum(run["physical_success"] for run in family),
        "family_case_count": len(family), "completed_moves": sum(run["completed_moves"] for run in family),
        "capture_margin_min_m": min(margins, default=None),
        "negative_acceptance": len(negatives) == negative_count and all(run["accepted"] for run in negatives) if negative_count else None,
        "negative_all_finite_episodes": len(negatives) == negative_count and all(run["finite_episode"] for run in negatives) if negative_count else None,
        "expected_failure_count": sum(run["accepted"] for run in negatives),
        "sensitivity_acceptance": len(sensitivity) == sensitivity_count and all(run["accepted"] for run in sensitivity) if sensitivity_count else None,
        "sensitivity_expected_all_finite_episodes": True,
        "sensitivity_all_finite_episodes": len(sensitivity) == sensitivity_count and all(run["finite_episode"] for run in sensitivity) if sensitivity_count else None,
        "sensitivity_physical_success_count": sum(run["physical_success"] for run in sensitivity),
        "acceptance": bool(acceptance), "rendering_success": rendering_success,
        "requested_render_case_count": family_count if args.render else 0, "rendered_case_count": len(rendered),
        "passed": bool(acceptance and rendering_success is not False), "runs": runs,
        "errors": errors, "render_errors": render_errors, "report_json": output / "report.json",
        "protocol": {"benchmark": "boulder_v1.transfers.run_transfer_benchmark", "keep_samples": True,
                     "source_static_duration_s": 2., "source_static_display_rerun": False,
                     "sequence_reset": False, "observer_interval_s": .05, "observer_terminal": "actual endpoint",
                     "physics": "canonical positives: unchanged production body, native unilateral feet, zero foot equalities",
                     "forces": "positive hinge torques only; declared support_loss negative pulses are cleaned up natively",
                     "hand_policy": "Stage5 endpoint_settle; Stage4 first_eligible preserved separately",
                     "foot_acquisition": "native load >5N sustained 100ms during complete LOAD; admitted final reference/readiness",
                     "diagnostics": "fixed scene/profiles; finite classified outcomes, not required failures or physical passes",
                     "negative_expected_statuses": NEGATIVES},
        "cautions": ["Expected negative failures are accepted tests, never successful transfers.",
                     "Profile outcomes are recorded exactly; anticipated status is not an acceptance gate.",
                     "Low-friction case tests candidate foot friction-demand admission, not an executed zero-normal-force landing.",
                     "Initial-static evidence is returned by the run; no initial or sequence-start replay frames.",
                     "Existing baseline paths/hashes are reported, not rerun or newly certified."],
        "exit_policy": "nonzero on acceptance mismatch, execution/evidence error, or requested render failure",
    }
    _write_json(output / "report.json", report)
    if args.summary:
        display = {key: value for key, value in report.items() if key not in ("provenance", "protocol", "cautions")}
        display["runs"] = []
        for run in runs:
            brief = {key: run[key] for key in ("name", "suite", "status", "reason", "expected_status",
                     "observed_expected_status", "expected_status_is_gate", "physical_success", "accepted", "finite_episode",
                     "completed_moves", "failed_index", "steps", "duration_s", "evidence_json") if key in run}
            brief["rendering"] = {key: run["rendering"].get(key) for key in
                                  ("requested", "success", "frame_count", "errors", "video", "endpoint", "observer_json")}
            brief["moves"] = []
            for move in run["moves"]:
                metrics = {key: move[key] for key in ("moving_limb", "status", "steps", "duration_s",
                           "capture_error_m", "capture_margin_m", "release_time_s", "capture_time_s",
                           "acquisition_time_s", "load_duration_s", "hand_load_max_N", "hand_margin_min_N",
                           "foot_loads", "foot_slip_max_m_s", "actuator_utilization_max", "final_reference_admitted")
                           if key in move}
                metrics["readiness"] = {key: (move.get("readiness") or {}).get(key) for key in ("ready", "duration")}
                if "support_capacity_diagnostic" in move:
                    metrics["support_capacity_diagnostic"] = {key: value for key, value in
                        move["support_capacity_diagnostic"].items() if key != "estimated_feedforward"}
                for milestone in ("release", "touchdown", "first_support", "acquisition"):
                    event = move.get(milestone)
                    if event is not None:
                        metrics[milestone] = {key: event[key] for key in ("step", "time_s", "normal_force_N",
                                              "contact_count", "signed_source_geom_distance_m", "sustained_s") if key in event}
                metrics["endpoint_foot_normal_N"] = {limb: foot.get("normal_force") for limb, foot in
                                                       (move.get("endpoint_feet") or {}).items()}
                brief["moves"].append(metrics)
            display["runs"].append(brief)
    else:
        display = report
    print(json.dumps(_json_value(display), allow_nan=False, indent=None if args.summary else 2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
