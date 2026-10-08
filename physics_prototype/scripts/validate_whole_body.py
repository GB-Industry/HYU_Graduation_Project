#!/usr/bin/env python3
"""Save Stage 5.1 native whole-body episodes, not planning or visual certification.

Default ALL is four positives and four atomic negatives at both 2 and 1 ms.
Rendering/auditing consumes these saved records; no display physics is run here.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_transition import _json_value, _write_json
from scripts.validate_transfers import FAMILY, _move_summary, _physical_success, _provenance as transfer_provenance
from scripts.transfer_motion_audit import audit_motion

FIXTURE = "whole_body_stage5.1"
NEGATIVES = {"unreachable": "REACH_INFEASIBLE", "root_unreachable": "REACH_INFEASIBLE",
             "waist_rom": "REACH_INFEASIBLE", "invalid_support": "INVALID_REQUEST"}


def _jobs(args):
    dts = (args.dt,) if args.dt is not None else (.002, .001)
    jobs = []
    if args.suite in ("all", "family"):
        jobs.extend(("family", kind, dt) for kind in FAMILY for dt in dts)
    if args.suite in ("all", "negative"):
        jobs.extend(("negative", case, dt) for case in NEGATIVES for dt in dts)
    return [job for job in jobs if not args.case or job[1] in args.case]


def _evidence_value(value):
    """Walk immutable dataclasses; omit only duplicate primitive wrappers."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _evidence_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _evidence_value(item) for key, item in value.items()
                if key != "underlying_result"}
    if isinstance(value, (list, tuple)):
        return [_evidence_value(item) for item in value]
    return _json_value(value)


def _finite_evidence(value):
    # Inspect original numbers before the shared serializer represents NaN as null.
    if is_dataclass(value) and not isinstance(value, type):
        return all(_finite_evidence(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, Mapping):
        return value.get("finite") is not False and all(_finite_evidence(item) for key, item in value.items()
                                                       if key != "underlying_result")
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, np.generic):
        return _finite_evidence(value.item())
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_finite_evidence(item) for item in value)
    return math.isfinite(value) if isinstance(value, float) else True


def _hash(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _model_hash(model):
    import mujoco

    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _fixture_metadata(dt):
    from boulder_v1.mjcf_builder import build_mjcf
    from boulder_v1.whole_body_demo import make_whole_body_fixture

    # Geometry/seed construction only, never execute or display a second static hold.
    model, _, scene, profile, seed = make_whole_body_fixture(dt)
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    inputs = {"fixture": FIXTURE, "factory": "boulder_v1.whole_body_demo.make_whole_body_fixture",
              "dt_s": dt, "scene": scene.to_dict(), "profile": profile.to_dict(), "seed_qpos": seed,
              "model_xml": xml, "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "compiled_model_sha256": _model_hash(model),
              "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq")}}
    inputs = _evidence_value(inputs)
    return model, {**inputs, "input_sha256": _hash(inputs)}


def _provenance(argv):
    provenance = transfer_provenance(argv)
    # Hash the entire native package, including all transitive dynamics dependencies.
    provenance["modules"] = {
        "boulder_v1." + path.stem: {"file": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted((ROOT / "src" / "boulder_v1").glob("*.py"))}
    provenance.update(command=[sys.executable, str(Path(__file__).resolve()), *argv],
                      validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return provenance


def _execute_case(case, dt, observer=None):
    from boulder_v1.whole_body_demo import run_whole_body_benchmark

    if case not in (*FAMILY, *NEGATIVES):
        raise ValueError(f"Unknown whole-body case: {case}")
    return run_whole_body_benchmark(timestep=dt, kind=case if case in FAMILY else "right_hand",
                                   negative=case if case in NEGATIVES else None,
                                   keep_samples=True, observer=observer)


def _case_verdict(suite, case, result, motion):
    if not _finite_evidence(result) or motion is None:
        return False
    moves = result.get("moves") or [result]
    clock = motion["whole_case"]["timing"]
    clean = not any(clock[key] for key in ("time_reset_count", "sample_gap_count",
                                          "changed_pose_at_duplicate_time_count", "step_counter_reset_count"))
    clean = clean and all(item["step_clock_matches"] and item["duration_matches_steps"]
                          and item["zero_recorded_applied_forces"] and item["samples_have_both_force_channels"]
                          for item in motion["native_accounting"])
    clean = clean and motion["source_matches_initial_static_final_state"] is True
    clean = clean and motion["case_initial_matches_first_move"] and motion["case_final_matches_last_move"]
    if suite == "family":
        accepted = clean and _physical_success(result) and all(move.get("samples") for move in moves)
        limbs = {"right_hand": ["RIGHT_HAND"], "left_hand": ["LEFT_HAND"], "foot": ["LEFT_FOOT"],
                 "sequence": ["RIGHT_HAND", "LEFT_HAND"]}
        accepted = accepted and [move.get("moving_limb") for move in moves] == limbs[case]
        for move in moves:
            for row in move.get("samples", []):
                accepted = accepted and all(row.get(key) is not None and 0 <= row[key] <= limit for key, limit in
                    (("root_linear_m_s", .10), ("root_angular_rad_s", .50), ("joint_max_rad_s", 1.)))
        if case == "sequence":
            accepted = accepted and len(moves) == 2 and result.get("completed_moves") == 2
            accepted = accepted and len(motion["sequence_boundaries"]) == 1
            accepted = accepted and all(boundary["full_state_equal"] and boundary["integration_state_equal"] is True
                                       for boundary in motion["sequence_boundaries"])
        return bool(accepted)
    return bool(clean and result.get("status") == NEGATIVES[case] and not result.get("success")
                and result.get("steps") == 0 and not result.get("released")
                and result.get("final_reference") is None and not (result.get("readiness") or {}).get("ready")
                and bool((result.get("initial_static") or {}).get("success"))
                and result.get("initial_state") == result.get("final_state")
                and not result.get("samples"))


def _run_case(suite, case, dt, output, provenance, fixture):
    model, inputs = fixture
    name = f"{case}_{dt * 1000:g}ms"
    path = output / f"{name}.json"
    result = _execute_case(case, dt)
    finite = _finite_evidence(result)
    saved = _evidence_value(result)
    motion, audit_error = None, None
    try:
        if not finite:
            raise ValueError("Nonfinite native evidence before serialization")
        kind = case if suite == "family" else "right_hand"
        negative = case if suite == "negative" else None
        if (saved.get("fixture") != FIXTURE or saved.get("dt_s") != dt or saved.get("profile") != inputs["profile"]
                or saved.get("kind") != kind or saved.get("negative") != negative):
            raise ValueError("Native result differs from declared fixture/timestep/profile")
        motion = audit_motion(model, saved)
    except (ValueError, KeyError, TypeError) as exc:
        audit_error = f"{type(exc).__name__}: {exc}"
    summary = {"name": name, "suite": suite, "case": case, "dt_s": dt,
               "expected_status": NEGATIVES[case] if suite == "negative" else "SUCCESS",
               "status": saved["status"], "reason": saved["reason"], "success": saved["success"],
               "physical_success": _physical_success(saved), "finite_episode": finite,
               "accepted": _case_verdict(suite, case, saved, motion) if finite else False,
               "steps": saved.get("steps"), "duration_s": saved.get("duration_s"),
               "completed_moves": saved.get("completed_moves", int(_physical_success(saved))),
               "profile": saved.get("profile"), "moves": [_move_summary(move) for move in saved.get("moves", [saved])],
               "audit_error": audit_error, "evidence_json": path}
    _write_json(path, {**saved, "validation": summary, "provenance": provenance, "fixture_inputs": inputs,
                       "motion_audit": motion,
                       "audit_source": {"fixture": FIXTURE, "factory": inputs["factory"],
                                        "fixture_input_sha256": inputs["input_sha256"],
                                        "method": "saved actual native states; owned FK/COM only; no dynamics",
                                        "quality_verdict": None}}, compact=True)
    summary["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("family", "negative", "all"), default="all")
    parser.add_argument("--dt", "--timestep", type=float, choices=(.002, .001))
    parser.add_argument("--case", action="append", choices=(*FAMILY, *NEGATIVES))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "whole-body-stage5.1")
    args = parser.parse_args(argv)
    jobs = _jobs(args)
    if not jobs:
        parser.error("no cases selected within this suite")
    output, outputs = args.output.resolve(), ROOT / "outputs"
    if (not output.parent.is_dir() or output.exists() and not output.is_dir()
            or not output.is_relative_to(outputs) or output == outputs):
        parser.error("output must be a dedicated directory under this worktree's existing outputs")
    if any(part.startswith(("stage4-baseline-", "stage4-preserved-", "stage5-baseline-", "stage5-preserved-"))
           for part in output.relative_to(outputs).parts):
        parser.error("output must not overwrite preservation baseline directories")
    output.mkdir(exist_ok=True)
    provenance = _provenance(list(sys.argv[1:] if argv is None else argv))
    fixtures, runs, errors = {}, [], []
    for suite, case, dt in jobs:
        try:
            if dt not in fixtures:
                fixtures[dt] = _fixture_metadata(dt)
            runs.append(_run_case(suite, case, dt, output, provenance, fixtures[dt]))
        except Exception as exc:
            error = {"case": f"{case}_{dt * 1000:g}ms", "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            print(f"[EVIDENCE ERROR] {error['case']}: {error['error']}", file=sys.stderr)
    current = _provenance([])
    stable = _json_value(current["modules"]) == _json_value(provenance["modules"])
    stable = stable and current["validator_sha256"] == provenance["validator_sha256"]
    family = [run for run in runs if run["suite"] == "family"]
    negatives = [run for run in runs if run["suite"] == "negative"]
    acceptance = len(runs) == len(jobs) and not errors and stable and all(run["accepted"] for run in runs)
    report = {"stage": "5.1", "fixture": FIXTURE, "suite": args.suite, "provenance": provenance,
              "source_hashes_stable_during_run": stable, "requested_case_count": len(jobs),
              "family_case_count": len(family), "negative_case_count": len(negatives),
              "physical_success_count": sum(run["physical_success"] for run in family),
              "expected_failure_count": sum(run["accepted"] for run in negatives),
              "physical_acceptance": (stable and len(family) == sum(s == "family" for s, _, _ in jobs)
                                      and all(run["accepted"] for run in family))
                                      if any(s == "family" for s, _, _ in jobs) else None,
              "negative_acceptance": (stable and len(negatives) == sum(s == "negative" for s, _, _ in jobs)
                                      and all(run["accepted"] for run in negatives))
                                      if any(s == "negative" for s, _, _ in jobs) else None,
              "acceptance": bool(acceptance), "passed": bool(acceptance), "runs": runs, "errors": errors,
              "protocol": {"benchmark": "boulder_v1.whole_body_demo.run_whole_body_benchmark",
                           "keep_samples": True, "source_static_display_rerun": False, "sequence_reset": False,
                           "negative_kind": "right_hand", "negative_expected_statuses": NEGATIVES,
                           "forces": "hinge motors only; zero recorded generalized/external forces",
                           "fixture_metadata": "helper geometry/seed construction only; no second static hold"},
              "visual_quality_verdict": None, "report_json": output / "report.json"}
    _write_json(output / "report.json", report)
    _write_json(output / "manifest.json", {"fixture": FIXTURE, "source_hashes_stable_during_run": stable,
                "passed": bool(acceptance), "episodes": [{key: run[key] for key in
                ("name", "suite", "case", "dt_s", "accepted", "evidence_json", "evidence_sha256")} for run in runs]})
    display = report
    if args.summary:
        display = {key: value for key, value in report.items() if key not in ("provenance", "protocol", "runs")}
        display["runs"] = [{key: run[key] for key in ("name", "status", "reason", "physical_success", "accepted",
                            "finite_episode", "steps", "duration_s", "audit_error", "evidence_json")} for run in runs]
    print(json.dumps(_json_value(display), allow_nan=False, indent=None if args.summary else 2))
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
