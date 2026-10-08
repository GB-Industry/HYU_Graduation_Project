#!/usr/bin/env python3
"""Write Stage 1 compiled-model evidence; rendering is opt-in observation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from boulder_v1.model_validation import validate_physical_model
from boulder_v1.schema import ClimberProfile


def _render_neutral(profile, destination):
    # No optional graphics imports or renderer creation on the JSON-only path.
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from boulder_v1.mjcf_builder import build_mjcf
    from boulder_v1.model_validation import _require_finite_sample, climber_body_ids
    from boulder_v1.runtime import compile_model
    from boulder_v1.scene_factory import make_synthetic_scene

    model = compile_model(build_mjcf(make_synthetic_scene(), profile))
    scratch = mujoco.MjData(model)
    scratch.eq_active[:] = 0
    mujoco.mj_forward(model, scratch)
    _require_finite_sample(scratch)
    com = scratch.subtree_com[int(climber_body_ids(model)[0])].copy()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, -0.46, 1.1]
    camera.distance, camera.azimuth, camera.elevation = 3.5, 100, -8
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(scratch, camera=camera)
        image = Image.fromarray(renderer.render())
        # Project the true COM, rather than moving a hidden 3D marker to the surface.
        view = mujoco.mjv_averageCamera(renderer.scene.camera[0], renderer.scene.camera[1])
        offset = com - view.pos
        depth = float(offset @ view.forward)
        focal = image.height * view.frustum_near / (view.frustum_top - view.frustum_bottom)
        x = image.width / 2 + focal * float(offset @ np.cross(view.forward, view.up)) / depth
        y = image.height / 2 - focal * float(offset @ view.up) / depth
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(255, 220, 35), outline=(0, 0, 0), width=2)
        draw.text((x + 12, y - 7), "COM", fill=(255, 220, 35), stroke_width=2, stroke_fill=(0, 0, 0))
        image.save(destination)
    return {"com_world": com.tolist(), "marker_pixel": [x, y]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "physical-model-stage1")
    parser.add_argument("--profile", default="base", help="Profile label only; physical variants are in the evidence")
    parser.add_argument("--render", action="store_true", help="Also render an observer-only neutral frame with COM marker")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "validation.json"
    try:
        profile = ClimberProfile(name=args.profile)
        evidence = validate_physical_model(profile=profile)
    except Exception as error:
        evidence = {"stage": 1, "passed": False, "error": {"type": type(error).__name__, "message": str(error)}}
    if args.render and "error" not in evidence and evidence.get("compiled_numerics", {}).get("passed", False):
        try:
            marker = _render_neutral(profile, args.output / "neutral-com.png")
            evidence["render"] = {"passed": True, "path": "neutral-com.png", "observer_only": True,
                                  **marker}
        except Exception as error:
            evidence["render"] = {"passed": False, "error": str(error), "observer_only": True}
            evidence["passed"] = False
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps({"passed": evidence["passed"], "output": str(destination),
                      "failed_sections": [name for name, passed in evidence.get("checks", {}).items() if not passed],
                      "error": evidence.get("error")}, allow_nan=False))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
