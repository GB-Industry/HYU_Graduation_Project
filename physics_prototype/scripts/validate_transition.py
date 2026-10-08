#!/usr/bin/env python3
"""Stage 4 single-hand physical transition acceptance and observer evidence.

Default: success at 2 ms and 1 ms, plus four classified negatives at 2 ms.
--dt selects one timestep for both success and negative cases. --render records
only the same authoritative success runs, at 640x480/20 fps using EGL. There is
no physics replay, live-state adoption, controller retuning, or forced capture.
The returned initial_static evidence is reused, not rerun for display. Observer
frames start after the two-second source hold and cover only the transition.

Legacy synthetic-wall IK is a separate scratch-only geometric diagnosis. Even
an admitted candidate is NOT dynamic route validation or a trackability claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
CANONICAL_PYTHON = "/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python"
EXPECTED = {
    "success": "SUCCESS",
    "unreachable": "REACH_INFEASIBLE",
    "orientation_invalid": "CAPTURE_FAILURE",
    "support_loss": "CONTACT_LOSS",
    "grip_after_capture": "GRIP_FAILURE",
}


def _json_value(value):
    """Strict JSON without asdict/deepcopy of immutable MappingProxy references."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported JSON evidence type: {type(value).__name__}")


def _write_json(path, value, *, compact=False):
    path.write_text(json.dumps(_json_value(value), allow_nan=False,
                               indent=None if compact else 2) + "\n", encoding="utf-8")


def _legacy_diagnosis(output, provenance, baselines):
    import mujoco
    from boulder_v1.contact_ik import solve_contact_pose
    from boulder_v1.grasp import STATIC_STANCE_QPOS
    from boulder_v1.mjcf_builder import build_mjcf
    from boulder_v1.scene_factory import make_synthetic_scene
    from boulder_v1.schema import ClimberProfile

    path = output / "legacy_seed_diagnosis.json"
    scope = ("Scratch-only geometric retargeting and exact Stage3 admission; no native time integration, "
             "no live candidate adoption, no dynamic route or trajectory acceptance")
    try:
        scene, profile = make_synthetic_scene(), ClimberProfile("base")
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
        scratch = mujoco.MjData(model)
        seed = np.array(STATIC_STANCE_QPOS)
        mujoco.mj_normalizeQuat(model, seed)
        scratch.qpos[:] = seed
        scratch.eq_active[:] = False
        mujoco.mj_forward(model, scratch)
        result = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
        evidence = {"scope": scope, "provenance": provenance, "preservation_baselines": baselines,
                    "scene": scene, "profile": profile, "legacy_qpos": STATIC_STANCE_QPOS,
                    "normalized_seed_qpos": seed, "result": result,
                    "admitted_candidate_qpos": result.qpos if result.admitted else None,
                    "adopted_live": False, "physics_steps": 0}
        summary = {"evidence_json": path, "admitted": result.admitted, "converged": result.converged,
                   "iterations": result.iterations, "reason": result.reason,
                   "initial_reason": result.initial_reason,
                   "position_error_max_m": max(r.position_error for r in result.residuals),
                   "orientation_error_max_rad": max(r.orientation_error for r in result.residuals),
                   "adopted_live": False, "affects_physical_acceptance": False, "error": None}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        evidence = {"scope": scope, "provenance": provenance, "error": error, "adopted_live": False}
        summary = {"evidence_json": path, "error": error, "adopted_live": False}
    _write_json(path, evidence)
    return summary


def _draw_hud(image, row, dt, reference):
    from PIL import ImageDraw, ImageFont

    def number(value, precision=3):
        return f"{value:.{precision}f}" if value is not None and math.isfinite(value) else "n/a"

    actual = row.get("capture_measurement") or {}
    command, readiness = row.get("command") or {}, row.get("readiness") or {}
    moving = row["hands"]["RIGHT_HAND"]
    captured = moving["active"] and moving["region_id"] == "reach_target"
    released = captured or row["phase"] in ("RELEASE", "THREE_POINT", "REACH", "CAPTURE", "SETTLE")
    heading = [f"Stage4 RH +60mm | PHYSICAL | dt={dt * 1000:g}ms",
               f"{row['status']} | Phase: {row['phase']} | elapsed={row['elapsed_s']:.3f}s"]
    if not row.get("pose_available", True):
        heading.append("POSE UNAVAILABLE: actual state is nonfinite")
    lines = [
        f"Root v={number(row['root_linear_m_s'], 4)}m/s w={number(row['root_angular_rad_s'], 4)}rad/s "
        f"joint max={number(row['joint_max_rad_s'], 4)}rad/s",
        "RH metrics relative to reach_target (reference is not physical)",
    ]
    for label, metric in (("Ref", reference or {}), ("Actual", actual)):
        gap = metric.get("gap_m")
        lines.append(f"{label}: error={number(1000 * gap if gap is not None else None, 2)}mm "
                     f"facing={number(metric.get('orientation'), 4)} "
                     f"speed={number(metric.get('relative_speed_m_s'), 4)}m/s")
    lines.append(f"RH released={'Y' if released else 'N'} captured={'Y' if captured else 'N'} "
                 f"active={'Y' if moving['active'] else 'N'} hold={moving['region_id'] or 'none'}")
    for limb, label in (("LEFT_HAND", "LH remaining"), ("RIGHT_HAND", "RH moving")):
        hand = row["hands"][limb]
        lines.append(f"{label}: load/cap={number(hand['load'], 1)}/{number(hand['capacity'], 1)}N "
                     f"active={'Y' if hand['active'] else 'N'} valid={'Y' if hand['valid'] else 'N'}")
    for limb, label in (("LEFT_FOOT", "LF"), ("RIGHT_FOOT", "RF")):
        foot = row["feet"][limb]
        lines.append(f"{label}: Fn={number(foot['normal_force'], 1)}N Ft={number(foot['tangential_force'], 1)}N "
                     f"slip={number(foot['tangential_speed'], 4)}m/s "
                     f"support={'Y' if foot['supporting'] else 'N'} slipping={'Y' if foot['slipping'] else 'N'}")
    utilization = command.get("utilization", [])
    torques = command.get("commanded_Nm", [])
    lines.extend([
        f"Motor |tau|max={number(max(map(abs, torques), default=None), 2)}Nm "
        f"util max={number(100 * max(utilization) if utilization else None, 1)}% "
        f"saturated={sum(command.get('saturated', []))}/{len(utilization)}",
        f"Readiness={readiness.get('ready', False)} sustained={number(readiness.get('duration'))}s",
        f"State: {readiness.get('reason', 'n/a')}",
        f"Outcome: {row['reason']}",
    ])
    draw = ImageDraw.Draw(image, "RGBA")
    # Fit the existing normal-color HUD to the canvas, never enlarge the video.
    for size in (11, 10, 9):
        font = ImageFont.load_default(size=size)
        if all(draw.textbbox((0, 0), line, font=font)[2] <= image.width - 24 for line in heading + lines):
            break
    spacing = draw.textbbox((0, 0), "Ag", font=font)[3] + 3
    for block, y in ((heading, 8), (lines, image.height - len(lines) * spacing - 12)):
        draw.rectangle([(6, y - 3), (image.width - 6, y + len(block) * spacing + 3)],
                       fill=(12, 18, 28, 220), outline=(50, 90, 140, 240), width=1)
        for index, line in enumerate(block):
            if draw.textbbox((0, 0), line, font=font)[2] > image.width - 24:
                while line and draw.textbbox((0, 0), line + "...", font=font)[2] > image.width - 24:
                    line = line[:-1]
                line += "..."
            draw.text((12, y + index * spacing), line, font=font, fill=(210, 230, 245))
    return image


def _run_case(dt, scenario, output, render, provenance, baselines):
    import mujoco
    from boulder_v1.single_hand import run_single_hand_benchmark

    name = f"{scenario}_{dt * 1000:g}ms"
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
        observed = {"row": row, "reference_hand_measurement": None,
                    "rendered": False, "frame_index": None}
        observations.append(observed)
        if render_errors:
            return
        try:
            from PIL import Image
            from scripts.render_demo import _configure_camera, _render_frame

            if renderer is None:
                import imageio.v2 as iio

                # The API supplies detached copies. Retain only the first model;
                # forward/render/reference work never receives authoritative data.
                render_model = detached_model
                scratch, reference_data = mujoco.MjData(render_model), mujoco.MjData(render_model)
                camera = mujoco.MjvCamera()
                _configure_camera(camera, render_model)
                renderer = mujoco.Renderer(render_model, height=480, width=640)
                writer = iio.get_writer(str(video_path), fps=20, codec="libx264", pixelformat="yuv420p")
            finite_pose = (row.get("pose_available", False)
                           and np.isfinite(detached_data.qpos).all() and np.isfinite(detached_data.qvel).all())
            reference = None
            if finite_pose:
                image = _render_frame(mujoco, render_model, detached_data, scratch, renderer, camera)
                qref, qdref = np.asarray(row["q_ref"]), np.asarray(row["qd_ref"])
                if np.isfinite(qref).all() and np.isfinite(qdref).all():
                    reference_data.qpos[:] = qref
                    reference_data.qvel[:] = qdref
                    reference_data.eq_active[:] = False
                    mujoco.mj_forward(render_model, reference_data)
                    source, target = reference_data.site("right_hand_site"), reference_data.site("site_reach_target")
                    jp, tp = np.zeros((3, render_model.nv)), np.zeros((3, render_model.nv))
                    mujoco.mj_jacSite(render_model, reference_data, jp, None, source.id)
                    mujoco.mj_jacSite(render_model, reference_data, tp, None, target.id)
                    reference = {"gap_m": float(np.linalg.norm(source.xpos - target.xpos)),
                                 "orientation": float(np.clip(source.xmat.reshape(3, 3)[:, 2]
                                                               @ target.xmat.reshape(3, 3)[:, 2], 0., 1.)),
                                 "relative_speed_m_s": float(np.linalg.norm((jp - tp) @ qdref))}
            else:
                image = Image.new("RGB", (640, 480), color=(10, 14, 22))
            _draw_hud(image, row, dt, reference)
            writer.append_data(np.asarray(image))
            observed.update(reference_hand_measurement=reference, rendered=True, frame_index=frame_count)
            frame_count += 1
            if row["terminal"]:
                image.save(endpoint_path)
        except Exception as exc:
            render_error("frame", exc)

    try:
        result = run_single_hand_benchmark(timestep=dt, scenario=scenario,
                                           observer=observe if render else None, keep_samples=True)
    except Exception as exc:
        result = {"success": False, "status": "EXECUTION_ERROR",
                  "reason": f"{type(exc).__name__}: {exc}", "samples": []}
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
                 "frame_count": frame_count, "observer_interval_s": .05,
                 "observation_count": len(observations), "errors": render_errors,
                 "video": video_path if render and not render_errors else None,
                 "endpoint": endpoint_path if render and endpoint_path.is_file() else None,
                 "observer_json": observer_path if render else None}
    expected_success = scenario == "success"
    accepted = result["status"] == EXPECTED[scenario] and bool(result["success"]) == expected_success
    physical_success = bool(result["success"] and result["status"] == "SUCCESS")
    summary = {"name": name, "scenario": scenario, "dt_s": dt, "expected_status": EXPECTED[scenario],
               "success": result["success"], "physical_success": physical_success,
               "status": result["status"], "reason": result["reason"], "accepted": accepted,
               "expected_failure_observed": accepted if not expected_success else None,
               "evidence_json": evidence_path, "rendering": rendering}
    for key in ("steps", "duration_s", "phases", "events", "released", "capture", "first_eligible",
                "three_point", "actuator_utilization_max", "foot_support_fraction", "foot_slip_max_m_s",
                "readiness", "new_contacts", "model_proof"):
        if key in result:
            summary[key] = result[key]
    initial = result.get("initial_static", {})
    for key in ("max_reach_error_m", "max_tracking_error_m", "remaining_hand_load_max_N", "foot_loads", "preflight"):
        if key in result:
            summary[key] = result[key]
    summary["initial_static"] = {key: initial[key] for key in ("success", "status", "reason", "readiness") if key in initial}
    retarget = result.get("retarget")
    if retarget is not None:
        summary["retarget"] = {key: getattr(retarget, key) for key in ("admitted", "converged", "iterations", "reason")}
    if render:
        _write_json(observer_path, {"name": name, "rendering": rendering, "observations": observations}, compact=True)
    _write_json(evidence_path, {**result, "name": name, "scenario": scenario, "dt_s": dt,
                              "expected_status": EXPECTED[scenario], "accepted": accepted,
                              "physical_success": physical_success, "rendering": rendering,
                              "provenance": provenance, "preservation_baselines": baselines}, compact=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dt", type=float, choices=(.002, .001), help="seconds; default: success at both, negatives at 2 ms")
    parser.add_argument("--suite", choices=("all", "success", "negative"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transition-stage4")
    parser.add_argument("--summary", action="store_true", help="concise stdout; full per-case traces are always saved")
    parser.add_argument("--render", action="store_true", help="EGL videos of selected success runs, never negative-case replay")
    args = parser.parse_args(argv)
    if args.render and args.suite == "negative":
        parser.error("--render requires --suite all or success; negatives are not rendered")
    output = args.output.resolve()
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    if output.exists() and not output.is_dir():
        parser.error("output must be a directory")
    if output == ROOT / "outputs" or any(output.is_relative_to(ROOT / "outputs" / directory)
            for directory in ("stage4-baseline-controller", "stage4-baseline-contacts")):
        parser.error("output must not overwrite preservation baseline reports")
    output.mkdir(exist_ok=True)
    if args.render:
        os.environ["MUJOCO_GL"] = "egl"

    import boulder_v1
    import mujoco
    from boulder_v1 import contact_ik, single_hand

    versions = {}
    for package in ("boulder-prototype-v1", "mujoco", "numpy", "Pillow", "imageio", "imageio-ffmpeg"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {"project_root": ROOT, "python_executable": sys.executable,
                  "canonical_python_executable": CANONICAL_PYTHON,
                  "using_canonical_executable": sys.executable == CANONICAL_PYTHON,
                  "python_version": sys.version, "working_directory": Path.cwd(),
                  "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                  "package_versions": versions, "package_file": boulder_v1.__file__,
                  "modules": {name: module.__file__ for name, module in sorted(sys.modules.items())
                              if (name.startswith("boulder_v1.") or name in ("mujoco", "numpy"))
                              and getattr(module, "__file__", None)},
                  "json_serializer": Path(__file__).resolve(), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                  "PYTHONPATH": os.environ.get("PYTHONPATH"),
                  "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}
    baselines = {}
    errors, runs = [], []
    for label in ("controller", "contacts"):
        directory = ROOT / "outputs" / f"stage4-baseline-{label}"
        path = directory / "report.json"
        baseline = {"report_json": path, "available": path.is_file(),
                    "artifact_paths": sorted(directory.glob("*.json")), "rerun": False}
        if path.is_file():
            try:
                raw = path.read_bytes()
                old = json.loads(raw)
                baseline.update(report_sha256=hashlib.sha256(raw).hexdigest(),
                                reported_verdicts={key: old[key] for key in ("passed", "physical_acceptance",
                                    "contact_acceptance", "controller_convergence") if key in old})
            except Exception as exc:
                errors.append({"case": f"baseline_{label}", "error": f"{type(exc).__name__}: {exc}"})
        baselines[label] = baseline
    legacy = _legacy_diagnosis(output, provenance, baselines)
    if legacy["error"]:
        errors.append({"case": "legacy_seed_diagnosis", "error": legacy["error"]})
    success_dts = (args.dt,) if args.dt is not None else (.002, .001)
    negative_dt = args.dt if args.dt is not None else .002
    jobs = [(dt, "success") for dt in success_dts] if args.suite in ("all", "success") else []
    if args.suite in ("all", "negative"):
        jobs.extend((negative_dt, scenario) for scenario in EXPECTED if scenario != "success")
    for dt, scenario in jobs:
        try:
            run = _run_case(dt, scenario, output, args.render and scenario == "success", provenance, baselines)
            runs.append(run)
            if run["status"] == "EXECUTION_ERROR":
                errors.append({"case": run["name"], "error": run["reason"]})
        except Exception as exc:
            errors.append({"case": f"{scenario}_{dt * 1000:g}ms", "error": f"{type(exc).__name__}: {exc}"})
    positives = [run for run in runs if run["scenario"] == "success"]
    negatives = [run for run in runs if run["scenario"] != "success"]
    physical_acceptance = all(run["accepted"] for run in positives) if positives else None
    negative_acceptance = all(run["accepted"] for run in negatives) if negatives else None
    acceptance = bool(runs) and len(runs) == len(jobs) and not errors and all(run["accepted"] for run in runs)
    rendered = [run for run in runs if run["rendering"]["requested"]]
    rendering_success = bool(rendered) and all(run["rendering"]["success"] for run in rendered) if args.render else None
    render_errors = [error for run in runs for error in run["rendering"]["errors"]]
    passed = acceptance and rendering_success is not False
    report_path = output / "report.json"
    report = {"stage": 4, "suite": args.suite, "success_timesteps_s": success_dts if args.suite != "negative" else [],
              "negative_timestep_s": negative_dt if args.suite != "success" else None,
              "provenance": provenance, "preservation_baselines": baselines,
              "scope": "Single RH +60mm physical transition on the canonical benchmark; not synthetic-wall route validation",
              "protocol": {"benchmark": f"{single_hand.__name__}.run_single_hand_benchmark",
                           "ik": f"{contact_ik.__name__}.solve_contact_pose", "keep_samples": True,
                           "source_static_duration_s": 2., "observer_interval_s": .05,
                           "observer_terminal": "actual terminal, even off the regular sampling grid",
                           "initial_static_display_rerun": False, "transition_reset": False,
                           "root": "physical free joint; all generalized coordinates retained in evidence",
                           "foot_support": "native unilateral contacts, no foot equalities",
                           "controller": "unchanged Stage3 torque impedance and motor limits",
                           "capture": "unchanged physical distance/facing/speed/penetration/reaction gates, never forced",
                           "forces": "success uses hinge torques only; negatives use only declared benchmark faults",
                           "expected_statuses": EXPECTED, "negative_rendering": False},
              "integration_cautions": [
                  "Legacy seed admission is geometric only; its candidate is never adopted into the transition.",
                  "Initial two-second source-static evidence is returned by the API, not shown as replayed video.",
                  "Instantaneous release/capture are recorded in events; no unobserved phase or capture frame is fabricated.",
                  "Correctly classified negative failures are accepted tests, not successful physical transitions.",
                  "Preservation reports are existing artifacts, not rerun or newly certified by this validator.",
              ],
              "exit_policy": "nonzero on status/success mismatch, execution/evidence error, or requested render failure",
              "physical_acceptance": physical_acceptance, "negative_acceptance": negative_acceptance,
              "expected_failure_count": sum(run["accepted"] for run in negatives),
              "acceptance": acceptance, "rendering_success": rendering_success, "passed": passed,
              "legacy_seed_diagnosis": legacy, "runs": runs, "errors": errors,
              "render_errors": render_errors, "report_json": report_path}
    _write_json(report_path, report)
    if args.summary:
        summary = {key: report[key] for key in ("passed", "acceptance", "physical_acceptance", "negative_acceptance",
                   "expected_failure_count", "rendering_success", "errors", "render_errors", "report_json")}
        summary.update(runs=[{key: run[key] for key in ("name", "success", "physical_success", "status",
                               "expected_status", "accepted", "evidence_json")} for run in runs],
                       legacy_seed_diagnosis=legacy)
        print(json.dumps(_json_value(summary), allow_nan=False))
    else:
        print(json.dumps(_json_value(report), indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
