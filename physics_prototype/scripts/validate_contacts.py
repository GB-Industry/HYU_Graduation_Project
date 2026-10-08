"""Canonical Stage 2 native-contact validation; controller convergence is separate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, choices=(.002, .001), help="default: both timesteps")
    parser.add_argument("--suite", choices=("all", "foot", "hand", "mixed"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "contact-stage2")
    parser.add_argument("--render", action="store_true", help="optional EGL observer, never part of acceptance")
    parser.add_argument("--summary", action="store_true", help="print concise verdicts; full evidence is always saved")
    args = parser.parse_args()
    if args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from boulder_v1.contact_benchmarks import FOOT_CASES, run_capture_gates, run_foot_case, run_hand_case, run_mixed_case

    if not args.output.parent.is_dir():
        parser.error("output parent must already exist")
    args.output.mkdir(exist_ok=True)
    renderer = None
    rendered_model = None
    frames = []
    observer_states = []
    render_errors = []

    def observe(fixture, snapshot):
        nonlocal renderer, rendered_model
        from boulder_v1.contact_benchmarks import _fresh

        if rendered_model is not fixture.model:
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(fixture.model, height=480, width=640)
            rendered_model = fixture.model
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0., -.25, 1.]
        camera.distance, camera.azimuth, camera.elevation = 3.5, 145., -15.
        state = _fresh(fixture.model, fixture.data)
        renderer.update_scene(state, camera=camera)
        scene = renderer.scene
        markers = [(state.subtree_com[fixture.model.body("climber_root").id], [.95, .75, .15, 1.])]
        for limb, support in snapshot.feet.items():
            side = "left" if limb.value.startswith("LEFT") else "right"
            color = [1., .2, .15, 1.] if support.slipping else ([.15, .9, .45, 1.] if support.supporting else [.5, .5, .5, 1.])
            markers.append((state.site(f"{side}_foot_site").xpos, color))
        for limb, hand in snapshot.hands.items():
            side = "left" if limb.value.startswith("LEFT") else "right"
            markers.append((state.site(f"{side}_hand_site").xpos, [.2, .6, 1., 1.] if hand.loaded else [.5, .5, .5, 1.]))
        for position, color in markers:
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([.018, .018, .018]), position, np.eye(3).reshape(-1), np.array(color))
                scene.ngeom += 1
        frames.append(renderer.render().copy())
        observer_states.append({"dt_s": float(fixture.model.opt.timestep), "time_s": float(state.time),
                                "COM_world_m": state.subtree_com[fixture.model.body("climber_root").id].tolist(),
                                "feet": {limb.value: support.status.value for limb, support in snapshot.feet.items()},
                                "hands": {limb.value: {"active": hand.active, "loaded": hand.loaded}
                                          for limb, hand in snapshot.hands.items()}})

    def render_observer(fixture, snapshot):
        if not render_errors:
            try:
                observe(fixture, snapshot)
            except Exception as exc:
                render_errors.append(f"{type(exc).__name__}: {exc}")

    runs = []
    errors = []
    try:
        for dt in (args.dt,) if args.dt is not None else (.002, .001):
            jobs = []
            if args.suite in ("all", "foot"):
                jobs.extend((f"foot_{case.name}_{dt:g}", lambda case=case: run_foot_case(case, dt)) for case in FOOT_CASES)
            if args.suite in ("all", "hand"):
                jobs.append((f"hand_capture_{dt:g}", lambda: run_capture_gates(dt)))
                jobs.extend((f"hand_{capacity:g}_{dt:g}", lambda capacity=capacity: run_hand_case(dt, capacity))
                            for capacity in (70., 150.))
            if args.suite in ("all", "mixed"):
                jobs.append((f"mixed_{dt:g}", lambda: run_mixed_case(dt, observer=render_observer if args.render else None)))
            for name, job in jobs:
                try:
                    result = job()
                    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    runs.append({key: value for key, value in result.items() if key not in ("samples", "applied")})
                except Exception as exc:
                    errors.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
        if args.render and frames:
            try:
                import imageio.v3 as iio

                iio.imwrite(args.output / "mixed_observer.mp4", np.array(frames), fps=20)
                (args.output / "observer_states.json").write_text(json.dumps(observer_states, indent=2, allow_nan=False) + "\n")
            except Exception as exc:
                render_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as exc:
                render_errors.append(f"{type(exc).__name__}: {exc}")
    report = {"project_root": str(ROOT),
              "benchmark_module": sys.modules["boulder_v1.contact_benchmarks"].__file__,
              "environment": {"python_executable": sys.executable, "python_version": sys.version,
                               "numpy": np.__version__, "mujoco": mujoco.__version__},
              "scope": "canonical production geometry, foot sensor and bounded hand manager; no temporary imports",
              "data_source": "native MuJoCo contacts/equalities plus production FootSupportSensor and GraspManager telemetry",
              "exit_policy": "nonzero for contact failures only; controller convergence is independently reported",
              "contact_acceptance": bool(runs) and not errors and all(r["contact_acceptance"] for r in runs),
              "controller_convergence": all(r["controller_convergence"] for r in runs if r["kind"] == "mixed")
                                        if any(r["kind"] == "mixed" for r in runs) else None,
              "runs": runs, "errors": errors, "render_errors": render_errors}
    serialized = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "report.json").write_text(serialized + "\n")
    if args.summary:
        print(json.dumps({"environment": report["environment"], "contact_acceptance": report["contact_acceptance"],
                          "controller_convergence": report["controller_convergence"], "case_count": len(runs),
                          "output": str(args.output), "errors": errors, "render_errors": render_errors}, allow_nan=False))
    else:
        print(serialized)
    return 0 if report["contact_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
