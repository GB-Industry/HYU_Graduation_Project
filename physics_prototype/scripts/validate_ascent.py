#!/usr/bin/env python3
"""Save and audit the declared RH/LF/LH native ascent, not route feasibility.

Timing trials are independent native episodes. Only a certified baseline trial
can select demo_timing; authority checks use that same timing at both timesteps
and on the longer profile. Shorter exhaustion is a bounded-search observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from boulder_v1 import ascending_sequence as ascent
from boulder_v1.contact import effective_grip_capacity
from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.morphology_envelope import study_profiles
from scripts import validate_morphology_envelope as envelope
from scripts import validate_whole_body as wb
from scripts.transfer_motion_audit import audit_motion

FIXTURE = "ascending_stage5.3"
FACTORY = "boulder_v1.ascending_sequence.make_ascending_fixture"
NEGATIVES = ("outside_workspace", "support_infeasible", "blocked_path", "candidate_exhaustion")
LIMBS = ("RIGHT_HAND", "LEFT_FOOT", "LEFT_HAND")
TARGETS = ("reach_target", "foot_target", "left_reach_target")
TIMINGS = ("conservative", "moderate", "fast")


def _provenance(argv):
    value = wb._provenance(argv)
    value.update(command=[sys.executable, str(Path(__file__).resolve()), *argv],
                 validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    value["audit_dependencies"] = {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
        for name in ("validate_morphology_envelope.py", "validate_whole_body.py", "validate_transfers.py",
                     "validate_transition.py", "transfer_motion_audit.py")}
    return wb._evidence_value(value)


def _fixture_metadata(profile, dt, negative=None):
    if profile not in tuple(study_profiles()[name] for name in ("baseline", "longer", "shorter")):
        raise ValueError("Ascent requires the explicit baseline/+5%/-5% geometry-only profiles")
    model, _, scene, compiled_profile, seed, native = ascent.make_ascending_fixture(profile, dt, negative=negative)
    native = wb._evidence_value(native)
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    if (compiled_profile != profile or model.opt.timestep != dt or not wb._finite_evidence(native)
            or native["profile"] != profile.to_dict() or native["scene"] != wb._evidence_value(scene.to_dict())
            or native["seed_qpos"] != wb._evidence_value(seed)
            or native["model_xml_sha256"] != hashlib.sha256(xml.encode()).hexdigest()
            or native["contact_geometry"] != wb._evidence_value({r.id: canonical_geometry(r) for r in scene.contact_regions})):
        raise ValueError("Native fixture differs from declared scene/profile/seed/typed geometry/XML")
    inputs = {**native, "factory": FACTORY, "native_metadata": native, "model_xml": xml,
              "compiled_model_sha256": wb._model_hash(model),
              "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq")},
              "expected_hand_capacity_N": {r.id: effective_grip_capacity(profile, r, r.normal)
                                           for r in scene.contact_regions},
              "mass_kg": float(model.body_subtreemass[model.body("climber_root").id])}
    return model, {**inputs, "input_sha256": wb._hash(inputs)}


def _reference_identity(result):
    """Observe Python identity BEFORE walking dataclasses/immutable mappings."""
    moves = result.get("moves", [])
    return [{"before_move_index": i - 1, "after_move_index": i,
             "same_object": moves[i].get("incoming_reference") is not None
                            and moves[i]["incoming_reference"] is moves[i - 1].get("final_reference"),
             "method": "native return objects compared with is before serialization"}
            for i in range(1, len(moves))]


def _candidate_checks(saved):
    expected = [i for i in (0, 2) if i < len(saved["moves"])] if saved["success"] else [0]
    if [item["move_index"] for item in saved["candidates"]] != expected:
        return False
    for item in saved["candidates"]:
        index, selection = item["move_index"], item["selection"]
        state = saved["moves"][index]["initial_state"] if index < len(saved["moves"]) else saved["final_state"]
        vector = saved["moves"][index]["initial_integration_state"] if index < len(saved["moves"]) else saved["final_integration_state"]
        assessments = selection["assessments"]
        if [label for label, _ in assessments] != ["default", "neutral_yaw", "conservative"]:
            return False
        for _, assessment in assessments:
            diagnostics = assessment["diagnostics"]
            if (diagnostics.get("actual_qpos") != state["qpos"] or diagnostics.get("actual_integration_state") != vector
                    or diagnostics.get("actual_state_digest") != hashlib.sha256(np.asarray(vector, dtype=float).tobytes()).hexdigest()):
                return False
        admitted = [i for i, (_, a) in enumerate(assessments) if a["feasible"]]
        if (selection["feasible"] != bool(admitted) or selection["selected_index"] != (admitted[0] if admitted else None)
                or selection["selected_name"] != (assessments[admitted[0]][0] if admitted else None)):
            return False
        if admitted:
            chosen = assessments[admitted[0]][1]
            if not chosen["geometric_feasible"] or not chosen["support_feasible"] or not chosen["diagnostics"].get("endpoint_admitted"):
                return False
    return True


def _move_checks(model, inputs, move, motion, timing):
    dt, source, final = move["dt_s"], move["initial_state"], move["final_state"]
    request, limb = move["request"], move["moving_limb"]
    contacts = source["contact_configuration"]
    supports = {l: h for l, h in contacts.items() if l != limb}
    goal = {**contacts, limb: move["target"]}
    ready = move.get("readiness") or {}
    events = move.get("events", [])
    source_ready = [e for e in events if e.get("event") == "SOURCE_READY"]
    final_ready = [e for e in events if e.get("event") == "FINAL_READY"]
    if (not wb._physical_success(move) or not envelope._terminal_matches(move, motion)
            or not envelope._event_clock(move) or not all(envelope._integration_checks(model, move).values())
            or not envelope._limit_checks(model, inputs, move) or not envelope._held_contacts(source, contacts)
            or not envelope._held_contacts(final, goal) or len(move.get("samples", [])) != move["steps"]
            or not move.get("released") or not (move.get("admission") or {}).get("source_reference_admitted")
            or request["source_contacts"] != contacts or request["support_contacts"] != supports
            or request["limb"] != limb or request["source"] != move["source"] or request["target"] != move["target"]
            or request["hand_reach_s"] != ascent.TIMINGS[timing][1]
            or request["whole_body"]["prepare_s"] != (4. if limb.endswith("FOOT") else ascent.TIMINGS[timing][0])
            or move["final_contacts"] != goal or move["final_reference"]["contact_intent"] != goal
            or ready.get("ready") is not True or ready.get("duration", 0.) < .5 - 1e-10
            or ready.get("reason") != "Sustained physical readiness" or ready.get("time") != final["time"]
            or len(source_ready) != 1 or source_ready[0]["time_s"] - source["time"] < .5 - 1e-10
            or len(final_ready) != 1 or final_ready[0]["time_s"] != final["time"]):
        return False
    phases = [(e["phase"], e["time_s"]) for e in events if "phase" in e]
    durations = {phase: (phases[i + 1][1] if i + 1 < len(phases) else final["time"]) - time
                 for i, (phase, time) in enumerate(phases)}
    fixed = {"UNLOAD" if limb.endswith("FOOT") else "LOAD_TRANSFER":
             4. if limb.endswith("FOOT") else ascent.TIMINGS[timing][0], "THREE_POINT": .5}
    fixed.update({"LIFT": 1.5, "LOAD": 2.} if limb.endswith("FOOT") else {"RELEASE_CLEARANCE": .5})
    if (len(durations) != len(phases) or any(not np.isclose(durations.get(phase, -1.), duration, atol=1e-8, rtol=0.)
            for phase, duration in fixed.items()) or durations.get("SOURCE_STABILIZE", 0.) < .5 - 1e-10
            or durations.get("SETTLE", 0.) < .5 - 1e-10):
        return False
    release = move.get("release_time_s")
    acquisition = move.get("capture_time_s") if limb.endswith("HAND") else move.get("acquisition_time_s")
    if release is None or acquisition is None or not source_ready[0]["time_s"] <= release < acquisition < final["time"]:
        return False
    if limb.endswith("HAND"):
        releases = [e for e in events if e.get("event") == "RELEASED"]
        if (not envelope._capture_valid(move, goal) or len(releases) != 1 or releases[0]["time_s"] != release
                or releases[0].get("limb") != limb or releases[0].get("source") != move["source"]):
            return False
    else:
        foot = request["foot_request"]
        separation, touch, acquired = move.get("release") or {}, move.get("touchdown") or {}, move.get("acquisition") or {}
        if (foot["lift_s"] != 1.5 or foot["support_s"] != .5 or foot["reach_s"] != 3. or foot["load_s"] != 2.
                or not np.isclose(foot["lift_m"], inputs["step_height_m"] + .02, rtol=0., atol=1e-12)
                or foot["airborne_pitch_rad"] != -.25 or separation.get("time_s") != release
                or separation.get("contact_count") != 0 or separation.get("normal_force_N") != 0.
                or separation.get("signed_source_geom_distance_m", -1.) <= 0.
                or touch.get("normal_force_N", 0.) <= 0. or acquired.get("normal_force_N", 0.) <= 5.
                or acquired.get("sustained_s", 0.) < .1 - 1e-10 or acquired.get("time_s") != acquisition
                or not release < touch.get("time_s", -1.) <= move.get("load_start_time_s", -1.) < acquisition
                or not acquisition <= move.get("load_complete_time_s", -1.) < final["time"]
                or not np.isclose(move.get("load_duration_s", -1.), 2., rtol=0., atol=1e-8)
                or not np.isclose(move["load_complete_time_s"] - move["load_start_time_s"], 2., rtol=0., atol=1e-8)):
            return False
    for index, row in enumerate(move["samples"], 1):
        time = row["time_s"]
        if (row.get("steps") != index or abs(time - source["time"] - index * dt) > 1e-8
                or any(value > limit for value, limit in zip(envelope._speeds(row, motion), (.10, .50, 1.)))):
            return False
        if limb.endswith("HAND"):
            required = contacts if time <= release + 1e-10 else supports if time <= acquisition + 1e-10 else goal
            if not envelope._held_contacts(row, required):
                return False
        else:
            # Contact configuration can acquire the target before ACQUIRED. The
            # three stationary supports remain mandatory throughout that interval.
            required = goal if time >= acquisition - 1e-10 else contacts if row["phase"] == "SOURCE_STABILIZE" else supports
            reduced = {**row, "contacts": required}
            if not envelope._held_contacts(reduced, required):
                return False
            moving = row["feet"][limb]
            observed = row.get("contacts") or {}
            if (any(observed.get(l) != h for l, h in supports.items())
                    or any(l not in goal for l in observed) or observed.get(limb) not in (None, move["source"], move["target"])
                    or time >= acquisition - 1e-10 and observed != goal
                    or time >= release - 1e-10 and row["phase"] not in ("LANDING", "LOAD", "SETTLE") and moving.get("contacts")
                    or any(c.get("normal_force", 0.) > envelope.FORCE_TOLERANCE
                           and (not c.get("admissible") or c.get("tangential_speed", 1.) > .01) for c in moving["contacts"])):
                return False
            if time >= release - 1e-10 and any(c["surface_geom"] == "geom_" + move["source"] for c in moving["contacts"]):
                return False
            if moving.get("idealized_attachment") is not None:
                return False
        attachments = {"grasp_" + l.lower() + "_" + h for l, h in row["contacts"].items() if l.endswith("HAND")}
        if len(row["eq_active"]) != model.neq:
            return False
        for eid, active in enumerate(row["eq_active"]):
            if bool(active) != (model.equality(eid).name in attachments):
                return False
    if limb.endswith("FOOT"):
        window = [r for r in move["samples"] if acquisition - .1 - 1e-10 <= r["time_s"] <= acquisition + 1e-10]
        if not window or not all(envelope._held_contacts({**r, "contacts": goal}, goal) for r in window):
            return False
        load = [r for r in move["samples"] if r["phase"] == "LOAD"]
        if len(load) != round(2. / dt):
            return False
    window = [r for r in move["samples"] if r["time_s"] >= final["time"] - .5 - 1e-10]
    return bool(window and final["time"] - window[0]["time_s"] >= .5 - 1e-10
                and all(r["phase"] == "SETTLE" and envelope._held_contacts(r, goal)
                        and all(v <= limit for v, limit in zip(envelope._speeds(r, motion), (.02, .05, .10))) for r in window))


def _negative_checks(model, saved, inputs, negative):
    if (saved["success"] is not False or saved["moves"] or saved["completed_moves"] != 0 or saved["steps"] != 0
            or saved["duration_s"] != 0. or saved["failed_index"] != 0 or saved["final_reference"] is not None
            or saved["initial_state"] != saved["final_state"]
            or saved["initial_integration_state"] != saved["final_integration_state"]
            or saved["status"] != "BOUNDED_CANDIDATE_SET_EXHAUSTED" or not _candidate_checks(saved)):
        return False
    for _, assessment in saved["candidates"][0]["selection"]["assessments"]:
        diagnostics = assessment["diagnostics"]
        if assessment["feasible"]:
            return False
        if negative in ("outside_workspace", "blocked_path"):
            proof = {"assessment": assessment, "classification": assessment["classification"],
                     "success": False, "steps": 0, "moving_limb": "RIGHT_HAND",
                     "source": saved["initial_state"]["contact_configuration"]["RIGHT_HAND"], "target": TARGETS[0]}
            if not envelope._negative_certificate(model, proof, "beyond_reach" if negative == "outside_workspace" else "blocked_path", inputs):
                return False
        elif negative == "support_infeasible":
            support = [diagnostics.get("actual_candidate_support") or {}, *diagnostics.get("candidate_support", [])]
            witnessed = False
            remaining = {l: h for l, h in saved["initial_state"]["contact_configuration"].items() if l != "RIGHT_HAND"}
            landed = {**saved["initial_state"]["contact_configuration"], "RIGHT_HAND": TARGETS[0]}
            for evidence in support:
                if evidence.get("admitted") is not False or evidence.get("contacts") not in (remaining, landed):
                    continue
                for limb, margin in (evidence.get("margins") or {}).items():
                    hold = (evidence.get("contacts") or {}).get(limb)
                    if limb.endswith("HAND") and hold in inputs["expected_hand_capacity_N"]:
                        capacity = inputs["expected_hand_capacity_N"][hold]
                        load = margin.get("load_N", -1.)
                        witnessed |= (margin.get("capacity_N") == capacity and load > capacity
                                      and np.isclose(margin.get("margin_N", 0.), capacity - load, atol=1e-8, rtol=0.))
            if assessment["classification"] != "SUPPORT_INFEASIBLE" or not witnessed:
                return False
        elif negative == "candidate_exhaustion":
            names, qpos = diagnostics.get("active_rom_limits", []), diagnostics.get("failed_qpos", [])
            failed = diagnostics.get("failed_result") or {}
            if (assessment["classification"] != "ROM_LIMITED_SEARCH" or not names or len(qpos) != model.nq
                    or failed.get("converged") is not False or failed.get("qpos") != qpos or failed.get("collisions")):
                return False
            for name in names:
                joint = model.joint(name)
                if not joint.limited[0] or min(abs(qpos[int(joint.qposadr[0])] - joint.range)) > 1e-7:
                    return False
        else:
            return False
    return True


def _case_verdict(model, saved, inputs, motion, identity, negative=None):
    if (not wb._finite_evidence(saved) or not envelope._valid_source(saved)
            or not all(envelope._integration_checks(model, saved).values())
            or not envelope._limit_checks(model, inputs, saved)):
        return False
    if negative is not None:
        return _negative_checks(model, saved, inputs, negative)
    if (motion is None or not wb._physical_success(saved) or saved["completed_moves"] != 3
            or saved["failed_index"] is not None or [m["moving_limb"] for m in saved["moves"]] != list(LIMBS)
            or [m["target"] for m in saved["moves"]] != list(TARGETS) or not _candidate_checks(saved)
            or len(identity) != 2 or not all(b["same_object"] is True for b in identity)
            or saved["final_reference"] != saved["moves"][-1]["final_reference"]
            or any(saved["moves"][i]["incoming_reference"] != saved["moves"][i - 1]["final_reference"] for i in (1, 2))
            or saved["steps"] != sum(m["steps"] for m in saved["moves"])
            or not np.isclose(saved["duration_s"], saved["steps"] * saved["dt_s"], atol=1e-8, rtol=0.)):
        return False
    first, third = [item["selection"] for item in saved["candidates"]]
    if (first["selected_name"] != "default" or not all(a["feasible"] and a["geometric_feasible"] is True
            for _, a in first["assessments"]) or third["selected_name"] != "conservative"
            or any(a["feasible"] for _, a in third["assessments"][:2])):
        return False
    clock = motion["whole_case"]["timing"]
    if (any(clock[k] for k in ("time_reset_count", "sample_gap_count", "changed_pose_at_duplicate_time_count", "step_counter_reset_count"))
            or not all(item[k] for item in motion["native_accounting"] for k in
                       ("step_clock_matches", "duration_matches_steps", "zero_recorded_applied_forces", "samples_have_both_force_channels"))
            or not all(motion[k] is True for k in ("source_matches_initial_static_final_state", "case_initial_matches_first_move", "case_final_matches_last_move"))
            or len(motion["sequence_boundaries"]) != 2
            or not all(b["full_state_equal"] and b["integration_state_equal"] is True for b in motion["sequence_boundaries"])
            or any(motion["whole_case"]["bodies"][key]["displacement_world_m"][2] <= 0.
                   for key in ("root", "pelvis", "climber_com"))):
        return False
    whole, dt = motion["whole_case"], saved["dt_s"]
    if (whole["bodies"]["root"]["largest_successive_step_m"] > .10 * dt + 1e-8
            or whole["root_orientation"]["largest_successive_angle_deg"] > np.degrees(.50 * dt) + 1e-5
            or whole["largest_joint_increment"]["rad"] > dt + 1e-8):
        return False
    if any(joint["min_rad"] < joint["compiled_range_rad"][0] - 1e-10
           or joint["max_rad"] > joint["compiled_range_rad"][1] + 1e-10 for joint in whole["joints"].values()):
        return False
    return all(_move_checks(model, inputs, m, motion, saved["timing"]) for m in saved["moves"])


def _run_case(name, timing, dt, output, provenance, negative=None):
    profile = study_profiles()[name]
    path = output / f"{name}_{timing}_{dt * 1000:g}ms{('_' + negative) if negative else ''}.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refuse to overwrite native evidence: " + str(path))
    model, inputs = _fixture_metadata(profile, dt, negative)
    result = ascent.run_ascending_sequence(profile, dt, timing=timing, negative=negative, keep_samples=True)
    finite, identity = wb._finite_evidence(result), _reference_identity(result)
    saved = wb._evidence_value(result)
    motion, accepted, error, integration = None, False, None, {}
    try:
        if (not finite or saved["fixture_inputs"] != inputs["native_metadata"] or saved["profile"] != profile.to_dict()
                or saved["dt_s"] != dt or saved["timing"] != timing or saved["negative"] != negative
                or saved["kind"] != "ascending_sequence" or saved["fixture"] != FIXTURE):
            raise ValueError("Original native result differs from finite declared fixture/profile/timing/negative")
        integration = envelope._integration_checks(model, saved)
        # A zero-execution negative has no transfer trajectory to audit. Do not
        # fabricate a move, event, capture or readiness history for it.
        motion = audit_motion(model, saved) if saved["moves"] else None
        accepted = _case_verdict(model, saved, inputs, motion, identity, negative)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    summary = {"name": path.stem, "profile_name": name, "timing": timing, "dt_s": dt, "negative": negative,
               "status": saved.get("status"), "reason": saved.get("reason"), "native_success": saved.get("success"),
               "accepted": bool(accepted), "physical_success": bool(accepted and negative is None),
               "valid_source": bool(finite and integration.get("initial") and envelope._valid_source(saved)),
               "finite_episode": finite, "completed_moves": saved.get("completed_moves"), "steps": saved.get("steps"),
               "duration_s": saved.get("duration_s"), "reference_identity": identity, "audit_error": error,
               "actual_vertical_progress_m": ({key: motion["whole_case"]["bodies"][key]["displacement_world_m"][2]
                    for key in ("root", "pelvis", "climber_com")} if motion else None),
               "diagnostic_scope": "Only these three local reference strategies; never global physical impossibility",
               "evidence_json": str(path)}
    wb._write_json(path, {**saved, "fixture_inputs": inputs, "provenance": provenance,
                         "validation": summary, "integration_audit": integration, "motion_audit": motion,
                         "reference_identity": identity}, compact=True)
    with path.open("rb") as file:
        summary["evidence_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    return summary


def _select_timing(runs):
    return next((timing for timing in reversed(TIMINGS) if any(r["profile_name"] == "baseline"
        and r["timing"] == timing and r["dt_s"] == .002 and r["negative"] is None
        and r["physical_success"] for r in runs)), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("timing", "authority", "negative", "all"), default="all")
    parser.add_argument("--timing", choices=TIMINGS, help="Explicit authority/negative timing; trials always test all three")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ascent-stage5.3")
    args = parser.parse_args(argv)
    if (args.suite in ("timing", "all") and args.timing is not None
            or args.suite == "negative" and args.timing not in (None, "conservative")):
        parser.error("Trials select their own timing; negative witnesses use conservative timing")
    output, outputs = args.output.resolve(), ROOT / "outputs"
    if (not output.parent.is_dir() or not output.is_relative_to(outputs) or output == outputs
            or output.exists() and not output.is_dir() or any(part.startswith(("stage4-baseline-", "stage4-preserved-",
                "stage5-baseline-", "stage5-preserved-")) for part in output.relative_to(outputs).parts)):
        parser.error("Use a dedicated directory beneath this worktree's existing outputs, not a preservation namespace")
    if any((output / name).exists() or (output / name).is_symlink() for name in ("report.json", "manifest.json")):
        parser.error("Use a new output directory; do not replace an existing native evidence report/manifest")
    output.mkdir(exist_ok=True)
    provenance, runs, errors = _provenance(list(sys.argv[1:] if argv is None else argv)), [], []

    def run(name, timing, dt, negative=None):
        previous = next((r for r in runs if (r["profile_name"], r["timing"], r["dt_s"], r["negative"]) ==
                        (name, timing, dt, negative)), None)
        if previous is not None:
            return previous
        try:
            print(f"[EXECUTE] {name}/{timing}/{dt * 1000:g}ms/{negative or 'sequence'}", flush=True)
            summary = _run_case(name, timing, dt, output, provenance, negative)
            runs.append(summary)
            return summary
        except Exception as exc:
            errors.append({"profile": name, "timing": timing, "dt_s": dt, "negative": negative,
                           "error": f"{type(exc).__name__}: {exc}"})
            return None

    if args.suite in ("timing", "all") or args.suite == "authority" and args.timing is None:
        for timing in TIMINGS:
            run("baseline", timing, .002)
    selected = args.timing or _select_timing(runs)
    authority, negatives = [], []
    if args.suite in ("authority", "all") and selected:
        authority = [run(name, selected, dt) for name in ("baseline", "longer") for dt in (.002, .001)]
    if args.suite in ("negative", "all"):
        # Fixed conservative hand paths also bind the independent geometry proofs.
        negatives = [run("shorter" if negative == "candidate_exhaustion" else "baseline", "conservative", dt, negative)
                     for negative in NEGATIVES for dt in (.002, .001)]
    stable = _provenance([])
    stable = all(stable[key] == provenance[key] for key in ("modules", "validator_sha256", "audit_dependencies"))
    authority_pass = bool(len(authority) == 4 and all(r and r["physical_success"] for r in authority))
    negative_pass = bool(len(negatives) == 8 and all(r and r["accepted"] for r in negatives))
    passed = bool(stable and not errors and (selected is not None if args.suite == "timing" else
                  authority_pass if args.suite == "authority" else negative_pass if args.suite == "negative" else
                  authority_pass and negative_pass))
    report = {"stage": "5.3", "fixture": FIXTURE, "suite": args.suite, "passed": passed,
              "selected_timing": selected, "demo_timing": selected if authority_pass and stable else None,
              "timing_selection_scope": "fastest certified baseline 2ms trial; same timing must pass all four authority cases",
              "authority_acceptance": authority_pass and stable if args.suite in ("authority", "all") else None,
              "negative_acceptance": negative_pass and stable if args.suite in ("negative", "all") else None,
              "source_hashes_stable_during_run": stable, "provenance": provenance, "runs": runs, "errors": errors,
              "native_episode_count": len(runs), "trial_reuse": "Identical baseline authority job reuses its original native trial record, not replay",
              "visual_quality_verdict": None, "global_feasibility_claim": False}
    wb._write_json(output / "report.json", report)
    wb._write_json(output / "manifest.json", {"fixture": FIXTURE, "passed": passed, "demo_timing": report["demo_timing"],
                  "episodes": [{key: r[key] for key in ("name", "profile_name", "timing", "dt_s", "negative",
                                "accepted", "evidence_json", "evidence_sha256")} for r in runs]})
    print(json.dumps(wb._json_value(report), allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
