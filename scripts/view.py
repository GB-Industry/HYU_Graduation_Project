"""Open the mannequin climber in the interactive MuJoCo viewer.

    python scripts/view.py                      # wall stance, frozen (no physics)
    python scripts/view.py --mode hang          # physics on, joints hold the pose: it hangs by its own strength
    python scripts/view.py --mode assist        # physics on, pelvis also held up (to pull hands with the mouse)
    python scripts/view.py --mode fall          # physics on, zero torque (no posture control yet)
    python scripts/view.py --pose t             # T-pose on the floor
    python scripts/view.py --profile long_limbs

[한국어 설명]
캐릭터를 MuJoCo 뷰어 창에 띄우는 스크립트. 벽은 175cm 몸에 맞춘 4홀드 테스트 벽(test_wall).
  --mode frozen  (기본) 물리를 돌리지 않고 자세만 보여 준다
  --mode hang    물리 켬 + 관절이 시작 자세를 유지하려고 힘을 씀(PoseHold) → 스스로 매달려 버틴다
  --mode assist  hang + 골반을 공중에 붙잡아 줌(보조). 몸이 안 떨어지니 마우스로 손을 당겨 놓치는 걸 보기 좋다
  --mode fall    물리 켬 + 모터 힘 0. 자세를 버티는 제어기가 아직 없어서 벽에서 떨어진다 (현재 상태 그대로)
  --pose climb|t       벽에 매달린 자세(기본) / T자 자세
  --profile 이름       baseline / short_limbs / long_limbs
손이 잡히거나 놓칠 때마다 터미널에 이유와 함께 출력하고, 5초마다 악력(지치면서 감소)과 하중을 출력한다.

뷰어 조작 (MuJoCo 기본)
  마우스 왼쪽 드래그: 회전 / 오른쪽 드래그: 이동 / 휠: 확대
  몸 일부를 더블클릭해 선택 → Ctrl + 오른쪽 드래그: 그 부위를 잡아당김 (Ctrl + 왼쪽 드래그: 비틀기)
  Space: 일시정지/재생, Backspace: 처음으로, C: 접촉점 표시, F: 접촉력 표시
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from climber import PRESETS, ContactModel, PoseHold, build_xml, climb_pose, t_pose, test_wall

# assist 모드에서 골반을 붙잡아 주는 보조 장치 (뻣뻣한 weld). 연구용 몸 모델에는 들어 있지 않다.
ASSIST = '<weld name="assist_pin" body1="pelvis" active="false" solref="0.004 1" solimp="0.99 0.999 0.001"/>'


def pin_pelvis(model, data):
    """골반을 지금 위치·방향 그대로 공중에 고정한다."""
    e, pb = model.equality("assist_pin").id, model.body("pelvis").id
    p, qw, R = data.xpos[pb].copy(), data.xquat[pb].copy(), data.xmat[pb].reshape(3, 3)
    model.eq_data[e][0:3] = 0
    model.eq_data[e][3:6] = R.T @ (-p)
    model.eq_data[e][6:10] = [qw[0], -qw[1], -qw[2], -qw[3]]
    data.eq_active[e] = 1


def describe(contacts) -> tuple:
    h, f = contacts.hands, contacts.feet
    return (h["left"].hold, h["right"].hold, f["left"].supported, f["right"].supported)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--pose", default="climb", choices=["climb", "t"])
    ap.add_argument("--mode", default="frozen", choices=["frozen", "hang", "assist", "fall"])
    ap.add_argument("--simulate", action="store_true", help="old name for --mode fall")
    args = ap.parse_args()
    mode = "fall" if args.simulate else args.mode
    profile = PRESETS[args.profile]

    xml = build_xml(profile, test_wall()).replace("</equality>", ASSIST + "</equality>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if args.pose == "t":
        q0 = t_pose(model)
    else:
        q0, _ = climb_pose(model, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    if mode == "assist":
        pin_pelvis(model, data)

    contacts = ContactModel(model, profile=profile)  # 악력·피로 값은 이 체형 것으로
    contacts.update(data)  # 손바닥이 홀드를 향하고 있으면 여기서 바로 잡힌다
    print(f"[mode={mode}] start:", contacts.summary())

    # hang/assist: 관절이 시작 자세를 유지하려는 단순 PD 제어 (climber/control.py)
    hold_pose = PoseHold(model, q0)
    last = describe(contacts)
    next_report = 5.0

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            start = time.time()
            if mode != "frozen":
                if mode in ("hang", "assist"):
                    data.ctrl[:] = hold_pose(data)
                mujoco.mj_step(model, data)
                contacts.update(data)
                now = describe(contacts)
                if now != last:
                    h = contacts.hands
                    print(f"t={data.time:6.2f}s  손 L={h['left'].hold or '-'}({h['left'].last_release or ''}) "
                          f"R={h['right'].hold or '-'}({h['right'].last_release or ''})  "
                          f"발 지지 L={now[2]} R={now[3]}")
                    last = now
                if data.time >= next_report:   # 5초마다 악력 상태
                    h = contacts.hands
                    print(f"t={data.time:6.1f}s  악력 L={h['left'].capacity:5.0f}N R={h['right'].capacity:5.0f}N  "
                          f"하중 L={np.linalg.norm(h['left'].load):4.0f}N R={np.linalg.norm(h['right'].load):4.0f}N")
                    next_report += 5.0
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
