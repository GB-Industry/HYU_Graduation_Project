"""Open the mannequin climber in the interactive MuJoCo viewer.

    python scripts/view.py                      # four-point stance on the wall, frozen
    python scripts/view.py --pose t             # T-pose on the floor
    python scripts/view.py --profile long_limbs
    python scripts/view.py --simulate           # run physics (zero torque, hands gripped)

Frozen mode never steps physics; it only shows the pose.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from climber import PRESETS, build_model, climb_pose, t_pose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--pose", default="climb", choices=["climb", "t"])
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    model = build_model(PRESETS[args.profile])
    data = mujoco.MjData(model)
    if args.pose == "t":
        data.qpos[:] = t_pose(model)
    else:
        q, info = climb_pose(model)
        data.qpos[:] = q
        if args.simulate:
            for side, hold in zip(("left", "right"), info["hands"]):
                data.eq_active[model.equality(f"grasp_{side}_{hold}").id] = 1
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            start = time.time()
            if args.simulate:
                mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
