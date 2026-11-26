"""Open the mannequin climber in the interactive MuJoCo viewer.

    python scripts/view.py                      # four-point stance on the wall, frozen
    python scripts/view.py --pose t             # T-pose on the floor
    python scripts/view.py --profile long_limbs
    python scripts/view.py --simulate           # run physics (zero torque, hands gripped)

Frozen mode never steps physics; it only shows the pose.

[한국어 설명]
캐릭터를 MuJoCo 뷰어 창에 띄워 마우스로 돌려 보는 스크립트.
  --pose climb|t       : 벽에 매달린 자세(기본) / T자 자세
  --profile 이름       : baseline / short_limbs / long_limbs
  --simulate           : 물리 켜기 (손을 홀드에 붙이고, 모터 힘 0으로 실행 → 몸이 처지는 것 확인용)
기본(--simulate 없음)은 물리를 돌리지 않고 자세만 보여 준다.
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
            # 물리를 켤 때만 양손 그립을 켠다 (손을 홀드에 붙이기)
            for side, hold in zip(("left", "right"), info["hands"]):
                data.eq_active[model.equality(f"grasp_{side}_{hold}").id] = 1
    mujoco.mj_forward(model, data)

    # 뷰어 창이 열려 있는 동안 반복: (물리 켜짐이면 한 스텝 진행) → 화면 갱신 → 실제 시간에 맞춰 대기
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            start = time.time()
            if args.simulate:
                mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
