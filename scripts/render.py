"""Render the climber (T-pose + wall stance, two angles) to a PNG, headless.

    python scripts/render.py                          # -> renders/climber_baseline.png
    python scripts/render.py --profile short_limbs --out my.png
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image

from climber import PRESETS, build_model, climb_pose, t_pose


def shoot(model, qpos, lookat, distance, azimuth, elevation, w=700, h=820):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam)
        return Image.fromarray(r.render())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    out = args.out or ROOT / "renders" / f"climber_{args.profile}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(PRESETS[args.profile])
    stand = t_pose(model)
    climb, info = climb_pose(model)
    wall_center = np.array([0.0, -0.2, 1.25])
    shots = [shoot(model, stand, np.array([0, -1.6, 0.95]), 2.7, 112, -12),
             shoot(model, climb, wall_center, 2.7, 65, -8),
             shoot(model, climb, wall_center, 2.5, 22, -5)]
    sheet = Image.new("RGB", (sum(s.width for s in shots), shots[0].height))
    x = 0
    for s in shots:
        sheet.paste(s, (x, 0))
        x += s.width
    sheet.save(out)
    res = {k: round(v, 3) for k, v in info["residual"].items()}
    print(f"saved {out}  (hands on {info['hands']}, IK residual m: {res})")


if __name__ == "__main__":
    main()
