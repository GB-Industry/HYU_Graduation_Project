#!/usr/bin/env python3
"""Stage 3 production impedance and native-contact static-hold acceptance.

Default: isolated hinges and nominal/disturbed mixed holds at 2 ms and 1 ms.
Rendering observes those same authoritative holds; it never reruns physics.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DISTURBANCE = {"body": "climber_root", "start_s": 5., "duration_s": .1,
               "force_world_N": [2., 0., 0.]}


def _write_json(path, value):
    from scripts.render_demo import _json_value

    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _draw_hud(image, row, name, dt, phase, disturbance):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    def number(value, precision=3):
        return f"{value:.{precision}f}" if value is not None and math.isfinite(value) else "n/a"

    command, readiness = row["command"], row["readiness"]
    utilization = command["utilization"]
    lines = [
        f"Root v={number(row['root_linear_m_s'], 4)}m/s w={number(row['root_angular_rad_s'], 4)}rad/s",
        f"Hinge max={number(row['joint_max_rad_s'], 4)} RMS={number(row['joint_rms_rad_s'], 4)}rad/s",
        f"Motor |tau|max={number(max(map(abs, command['commanded_Nm'])), 2)}Nm "
        f"util max/mean={number(100 * max(utilization), 1)}/{number(100 * sum(utilization) / len(utilization), 1)}% "
        f"sat={sum(command['saturated'])}/{len(utilization)}",
    ]
    for limb, label in (("LEFT_HAND", "LH"), ("RIGHT_HAND", "RH")):
        hand = row["hands"][limb]
        lines.append(f"{label} load/cap={number(hand['load'], 1)}/{number(hand['capacity'], 1)}N "
                     f"active={'Y' if hand['active'] else 'N'} valid={'Y' if hand['valid'] else 'N'}")
    for limb, label in (("LEFT_FOOT", "LF"), ("RIGHT_FOOT", "RF")):
        foot = row["feet"][limb]
        lines.append(f"{label} Fn={number(foot['normal_force'], 1)}N Ft={number(foot['tangential_force'], 1)}N "
                     f"support={'Y' if foot['supporting'] else 'N'} slip={'Y' if foot['slipping'] else 'N'} "
                     f"vT={number(foot['tangential_speed'], 4)}m/s")
    lines.extend([
        f"Readiness={'READY' if readiness['ready'] else 'WAIT'} duration={number(readiness['duration'])}s",
        f"State: {readiness['reason']}",
        (f"COM pulse: {disturbance['body']} {disturbance['force_world_N'][0]:+g}N X, "
         f"{disturbance['start_s']:g}-{disturbance['start_s'] + disturbance['duration_s']:g}s "
         f"active={'Y' if row['disturbance_active'] else 'N'}" if disturbance else
         "Input: hinge torques only; no COM force pulse"),
    ])
    heading = [f"Stage3 {name} | PHYSICAL | dt={dt * 1000:g}ms",
               f"{row.get('status', 'RUNNING')} | Phase: {phase} | t={row['elapsed_s']:.3f}s"]
    if not row.get("pose_available", True):
        heading.append("POSE UNAVAILABLE: actual endpoint is nonfinite")
    # Match the existing normal-color HUD; fit measured text, not a larger framebuffer.
    spacing = max(12, draw.textbbox((0, 0), "Ag", font=font)[3] + 3)
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


def _run_mixed(dt, disturbance, output, render):
    import mujoco
    import numpy as np
    from boulder_v1.static_control import run_static_benchmark

    name = f"mixed_{'disturbance' if disturbance else 'nominal'}_{dt * 1000:g}ms"
    renderer = writer = render_model = scratch = camera = None
    observer_rows, render_errors = [], []
    frame_count = 0
    video_path = output / f"{name}.mp4"

    def observe(row, detached_model, detached_data):
        nonlocal renderer, writer, render_model, scratch, camera, frame_count
        phase = ("Settling" if row["elapsed_s"] < 1. else
                 "Disturbance" if row["disturbance_active"] else
                 "Recovery" if disturbance and row["elapsed_s"] >= disturbance["start_s"] + disturbance["duration_s"] else
                 "Hold")
        observed = {"dt_s": dt, "phase": phase, "row": row,
                    "root_pos_m": detached_data.qpos[:3].tolist(),
                    "rendered": False, "frame_index": None}
        observer_rows.append(observed)
        if render_errors:
            return
        try:
            from scripts.render_demo import _configure_camera, _render_frame

            if renderer is None:
                import imageio.v2 as iio

                # Each callback has a new detached model. Retain the first one only
                # for rendering, and copy subsequent native data into its scratch.
                render_model = detached_model
                scratch = mujoco.MjData(render_model)
                camera = mujoco.MjvCamera()
                _configure_camera(camera, render_model)
                renderer = mujoco.Renderer(render_model, height=480, width=640)
                writer = iio.get_writer(str(video_path), fps=20, codec="libx264", pixelformat="yuv420p")
            if row.get("pose_available", True):
                image = _render_frame(mujoco, render_model, detached_data, scratch, renderer, camera)
            else:
                from PIL import Image
                image = Image.new("RGB", (640, 480), color=(10, 14, 22))
            writer.append_data(np.asarray(_draw_hud(image, row, name, dt, phase, disturbance)))
            observed.update(rendered=True, frame_index=frame_count)
            frame_count += 1
        except Exception as exc:
            error = f"{name}: {type(exc).__name__}: {exc}"
            render_errors.append(error)
            print(f"[RENDER ERROR] {error}", file=sys.stderr)

    try:
        result = run_static_benchmark(timestep=dt, disturbance=disturbance,
                                      observer=observe if render else None, keep_samples=True)
    except Exception as exc:
        result = {"success": False, "status": "EXECUTION_ERROR",
                  "reason": f"{type(exc).__name__}: {exc}", "dt_s": dt, "disturbance": disturbance,
                  "contact_acceptance": False, "controller_convergence": False, "samples": []}
    finally:
        for label, resource in (("video close", writer), ("renderer close", renderer)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    error = f"{name} {label}: {type(exc).__name__}: {exc}"
                    render_errors.append(error)
                    print(f"[RENDER ERROR] {error}", file=sys.stderr)
    result.setdefault("dt_s", dt)
    result.setdefault("disturbance", disturbance)
    if render and not render_errors and (not frame_count or not video_path.is_file() or not video_path.stat().st_size):
        render_errors.append(f"{name}: no video frames/artifact produced")
        print(f"[RENDER ERROR] {render_errors[-1]}", file=sys.stderr)
    rendering = {"requested": render,
                 "success": bool(frame_count and not render_errors) if render else None,
                 "observer_only": True, "resolution": [640, 480], "fps": 20,
                 "frame_count": frame_count, "observation_count": len(observer_rows),
                 "last_observed_elapsed_s": observer_rows[-1]["row"]["elapsed_s"] if observer_rows else None,
                 "video": video_path if render and not render_errors else None,
                 "observer_json": output / f"{name}_observer.json" if render else None,
                 "errors": render_errors}
    if render:
        _write_json(rendering["observer_json"], {"run": name, "disturbance": disturbance,
                    "rendering": rendering, "success": result["success"], "status": result["status"],
                    "contact_acceptance": result["contact_acceptance"],
                    "controller_convergence": result["controller_convergence"], "observations": observer_rows})
    evidence_path = output / f"{name}.json"
    _write_json(evidence_path, {**result, "name": name, "kind": "mixed", "rendering": rendering})
    fields = ("success", "status", "reason", "steps", "duration_s", "dt_s", "disturbance",
              "contact_acceptance", "controller_convergence", "scored", "joint_rms_rad_s",
              "actuator_utilization_max", "actuator_utilization_mean", "saturation_fraction",
               "max_applied_hand_load_N", "feet", "readiness", "recovery", "chatter", "disturbance_realization")
    return {"name": name, "kind": "mixed", "evidence_json": evidence_path,
            **{key: result[key] for key in fields if key in result}, "rendering": rendering}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "isolated", "mixed"), default="all")
    parser.add_argument("--dt", type=float, choices=(.002, .001), help="seconds; default: both timesteps")
    parser.add_argument("--render", action="store_true", help="EGL 640x480 observer videos of the same mixed runs")
    parser.add_argument("--summary", action="store_true", help="concise stdout; full per-run evidence is always saved")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "controller-stage3")
    parser.add_argument("--include-traces", action="store_true", help="also save isolated position/velocity/torque traces")
    args = parser.parse_args(argv)
    if args.render and args.suite == "isolated":
        parser.error("--render requires --suite all or mixed; isolated hinges are not full-climber videos")
    output = args.output.resolve()
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    if output.exists() and not output.is_dir():
        parser.error("output must be a directory")
    output.mkdir(exist_ok=True)
    if args.render:
        os.environ["MUJOCO_GL"] = "egl"

    import boulder_v1
    from boulder_v1 import controller_diagnostics, runtime, static_control
    from scripts.render_demo import _json_value

    timesteps = (args.dt,) if args.dt is not None else (.002, .001)
    runs, errors = [], []
    isolated = None
    if args.suite in ("all", "isolated"):
        try:
            evidence = controller_diagnostics.run_isolated_suite(timesteps=timesteps, include_traces=args.include_traces)
            physical = [case for case in evidence["cases"] if case["configuration"]["controller"] == "physical"]
            legacy = [case for case in evidence["cases"]
                      if case["configuration"]["controller"] == controller_diagnostics.LEGACY_BASELINE]
            groups = {}
            for case in physical:
                config = case["configuration"]
                groups.setdefault((config["joint"], config["timestep_s"]), {})[case["case"]] = case
            strength_checks = []
            for (joint, dt), cases in groups.items():
                base = cases["small"]["configuration"]
                weak = cases.get("demanding_weak_0.1", cases["demanding_weak"])
                strong = cases["demanding_strong"]
                checks = {
                    "physical_gains_unchanged": all(all(c["configuration"][key] == base[key]
                        for key in ("stiffness_Nm_rad", "damping_Nms_rad")) for c in cases.values()),
                    "strength_scales_ceiling_only": all(math.isclose(
                        c["configuration"]["limit_Nm"] / c["configuration"]["strength_scale"],
                        base["limit_Nm"] / base["strength_scale"], rel_tol=1e-12) for c in cases.values()),
                    "small_weak_and_strong_unsaturated": all(cases[label]["metrics"]["saturated_steps"] == 0
                                                             for label in ("small_weak", "small_strong")),
                    "demanding_weak_saturates_strong_does_not": (
                        weak["configuration"]["target_rad"] == strong["configuration"]["target_rad"]
                        and weak["metrics"]["saturated_steps"] > 0 and strong["metrics"]["saturated_steps"] == 0),
                }
                strength_checks.append({"joint": joint, "dt_s": dt, "checks": checks, "passed": all(checks.values())})
            strength_passed = bool(strength_checks) and all(c["passed"] for c in strength_checks)
            evidence["controller_strength_semantics"] = {"passed": strength_passed, "cases": strength_checks,
                "scope": "physical gains independent of strength; motor ceilings alone scale; no gain retuning"}
            isolated_path = output / "isolated.json"
            _write_json(isolated_path, evidence)
            failures = [{"case": c["case"], "joint": c["configuration"]["joint"],
                         "dt_s": c["configuration"]["timestep_s"],
                         "failed_checks": [key for key, passed in c["checks"].items() if not passed]}
                        for c in physical if not c["passed"]]
            isolated = {"passed": bool(physical) and evidence["passed"] and not failures,
                        "case_count": len(evidence["cases"]), "physical_case_count": len(physical),
                        "physical_passed_count": sum(c["passed"] for c in physical),
                        "physical_failures": failures, "legacy_case_count": len(legacy),
                        "legacy_passed_count": sum(c["passed"] for c in legacy),
                        "legacy_failed_count": sum(not c["passed"] for c in legacy),
                        "legacy_affects_production_verdict": False,
                        "controller_strength_semantics": strength_passed, "evidence_json": isolated_path}
        except Exception as exc:
            error = {"case": "isolated", "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            isolated = {"passed": False, "error": error["error"]}
            _write_json(output / "isolated.json", isolated)
    if args.suite in ("all", "mixed"):
        for dt in timesteps:
            for disturbance in (None, DISTURBANCE.copy()):
                run = _run_mixed(dt, disturbance, output, args.render)
                runs.append(run)
                if run["status"] == "EXECUTION_ERROR":
                    errors.append({"case": run["name"], "error": run["reason"]})

    versions = {}
    for package in ("boulder-prototype-v1", "mujoco", "numpy", "Pillow", "imageio", "imageio-ffmpeg"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    isolated_acceptance = isolated["passed"] if isolated is not None else None
    strength_acceptance = isolated.get("controller_strength_semantics", False) if isolated is not None else None
    static_acceptance = all(r["success"] for r in runs) if runs else None
    contact_acceptance = all(r["contact_acceptance"] for r in runs) if runs else None
    controller_convergence = all(r["controller_convergence"] for r in runs) if runs else None
    physical_acceptance = not errors and all(verdict is not False for verdict in (
        isolated_acceptance, strength_acceptance, static_acceptance, contact_acceptance, controller_convergence))
    rendering_success = all(r["rendering"]["success"] for r in runs) if args.render else None
    render_errors = [error for r in runs for error in r["rendering"]["errors"]]
    passed = physical_acceptance and rendering_success is not False
    report = {
        "stage": 3, "suite": args.suite, "timesteps_s": timesteps,
        "provenance": {"project_root": ROOT, "python_executable": sys.executable,
                       "python_version": sys.version, "working_directory": Path.cwd(),
                       "command": [sys.executable, str(Path(__file__).resolve()),
                                   *(sys.argv[1:] if argv is None else argv)],
                       "package_versions": versions, "package_file": boulder_v1.__file__,
                       "modules": {module.__name__: module.__file__ for module in
                                   (controller_diagnostics, runtime, static_control)},
                       "json_serializer": sys.modules[_json_value.__module__].__file__,
                       "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
        "scope": "production physical controller; fixed-mount hinges and canonical mixed static scene, not route validation",
        "protocol": {"duration_s": 11., "settle_s": 1., "controlled_hold_s": 10., "scored_window_s": 2.,
                     "feedforward_ramp_s": .2, "disturbance": DISTURBANCE,
                     "observer_interval_s": .05, "include_isolated_traces": args.include_traces,
                     "force_scope": "estimated reference forces are not applied; only hinge torques plus the declared COM pulse"},
        "exit_policy": "nonzero for any selected physical isolated/strength/static/contact/convergence failure or requested render failure; legacy baselines excluded",
        "isolated_acceptance": isolated_acceptance, "controller_strength_semantics": strength_acceptance,
        "static_acceptance": static_acceptance, "contact_acceptance": contact_acceptance,
        "controller_convergence": controller_convergence, "physical_acceptance": physical_acceptance,
        "rendering_success": rendering_success, "passed": passed,
        "isolated": isolated, "runs": runs, "errors": errors, "render_errors": render_errors,
    }
    _write_json(output / "report.json", report)
    if args.summary:
        summary = {key: report[key] for key in ("passed", "physical_acceptance", "isolated_acceptance",
                   "controller_strength_semantics", "static_acceptance", "contact_acceptance",
                   "controller_convergence", "rendering_success")}
        summary.update(isolated=isolated, mixed_runs=[{key: r[key] for key in
                       ("name", "success", "status", "contact_acceptance", "controller_convergence")} for r in runs],
                       output=output, errors=errors, render_errors=render_errors)
        print(json.dumps(_json_value(summary), allow_nan=False))
    else:
        print(json.dumps(_json_value(report), indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
