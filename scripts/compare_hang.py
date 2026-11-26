"""Hang several climbers side by side and show grip strength / load / fatigue above each one.

    python scripts/compare_hang.py                                # 850 / 500 / 350 N grips -> renders/compare_hang.mp4
    python scripts/compare_hang.py --live                         # show it in a window while it runs (q: quit)
    python scripts/compare_hang.py --grips 900,600,400 --seconds 90 --speed 6

[한국어 설명]
클라이머 여러 명(기본 3명, 악력만 다름)을 각자 같은 벽에 매달아 동시에 시뮬레이션하고,
나란히 보여 준다. 각 클라이머 위에 손별로
  - 악력 막대(파랑): 지금 악력 / 처음 악력. 지칠수록 줄어든다
  - 하중 표시(주황 선): 지금 손에 걸린 힘. 파랑 막대가 주황 선보다 짧아지면 손을 놓친다
  - 숫자: 악력 N, 하중 N, 피로도 % (= 처음보다 줄어든 악력 비율)
를 그린다. 기본은 4배속 영상(renders/compare_hang.mp4), --live면 창으로 바로 본다.
각 클라이머는 PoseHold(시작 자세 유지)로 매달려 있고, 손 규칙·악력 피로는 contact.py 그대로다.
"""
import argparse
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from climber import PRESETS, ContactModel, PoseHold, build_model, climb_pose, test_wall

W, H, HEAD = 520, 640, 200          # 클라이머 한 칸 크기, 위쪽 정보 패널 높이
BG, INK, INK2, TRACK = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 223)
BLUE, ORANGE, RED = (42, 120, 214), (235, 104, 52), (200, 40, 40)
FONT = r"C:\Windows\Fonts\malgun.ttf"
FONT_B = r"C:\Windows\Fonts\malgunbd.ttf"


class Climber:
    def __init__(self, profile):
        self.p = profile
        self.m = build_model(profile, test_wall())
        self.d = mujoco.MjData(self.m)
        q0, _ = climb_pose(self.m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
        self.d.qpos[:] = q0
        mujoco.mj_forward(self.m, self.d)
        self.contacts = ContactModel(self.m, profile=profile)
        self.ctrl = PoseHold(self.m, q0)
        self.contacts.update(self.d)
        self.released = {"left": None, "right": None}   # 손을 놓친 시각
        self.renderer = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, -0.25, 1.05]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.2, 68, -6

    def advance_to(self, t):
        while self.d.time < t - 1e-9:
            self.d.ctrl[:] = self.ctrl(self.d)
            mujoco.mj_step(self.m, self.d)
            self.contacts.update(self.d)
            for s, h in self.contacts.hands.items():
                if h.hold is None and self.released[s] is None:
                    self.released[s] = (self.d.time, h.last_release)

    def image(self):
        self.renderer.update_scene(self.d, camera=self.cam)
        return Image.fromarray(self.renderer.render())


def draw_panel(draw, x0, c: Climber, fonts, scale_max: float):
    """scale_max: 모든 클라이머 막대의 공통 최대값 (가장 센 악력). 그래서 클라이머끼리 막대 길이를 바로 비교할 수 있다."""
    big, mid, small = fonts
    base = c.contacts.grip_capacity
    draw.text((x0 + 16, 10), f"악력 {base:.0f} N 클라이머", fill=INK, font=big)
    if any(c.released.values()):
        first = min(v[0] for v in c.released.values() if v)
        draw.text((x0 + W - 16, 14), f"손 놓침 {first:.1f}초", fill=RED, font=mid, anchor="ra")
    else:
        draw.text((x0 + W - 16, 14), "매달리는 중", fill=INK2, font=mid, anchor="ra")
    for row, (s, label) in enumerate((("left", "왼손"), ("right", "오른손"))):
        h = c.contacts.hands[s]
        y = 62 + row * 66
        load = float(np.linalg.norm(h.load)) if h.hold else 0.0
        fatigue = 100.0 * (1.0 - h.capacity / base)
        bar_x, bar_w, bar_h = x0 + 84, W - 84 - 16, 16
        draw.text((x0 + 16, y - 2), label, fill=INK, font=mid)
        draw.rounded_rectangle((bar_x, y, bar_x + bar_w, y + bar_h), radius=4, fill=TRACK)
        # 처음 악력 위치 (얇은 회색 테두리) → 그 안에서 파랑이 줄어드는 만큼이 피로
        draw.rounded_rectangle((bar_x, y, bar_x + bar_w * base / scale_max, y + bar_h), radius=4,
                               outline=INK2, width=1)
        cap_w = bar_w * h.capacity / scale_max
        draw.rounded_rectangle((bar_x, y, bar_x + max(cap_w, 6), y + bar_h), radius=4, fill=BLUE)
        if h.hold:
            lx = bar_x + bar_w * min(load / scale_max, 1.0)
            draw.rectangle((lx - 2, y - 5, lx + 2, y + bar_h + 5), fill=ORANGE)
        state = "" if h.hold else "  (놓침)"
        draw.text((bar_x, y + bar_h + 6),
                  f"악력 {h.capacity:4.0f} N   하중 {load:4.0f} N   피로도 {fatigue:4.1f}%{state}",
                  fill=INK2 if h.hold else RED, font=small)


def compose(climbers, t, speed, fonts):
    canvas = Image.new("RGB", (W * len(climbers), 44 + HEAD + H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((16, 8), f"t = {t:5.1f}초   ({speed:g}배속)", fill=INK, font=fonts[0])
    scale_max = max(c.contacts.grip_capacity for c in climbers)
    d.text((W * len(climbers) - 16, 12),
           f"파랑 막대 = 악력(지치면 줄어듦, 회색 테두리 = 처음 악력)   주황 선 = 손에 걸린 하중   (막대 끝 = {scale_max:.0f} N)",
           fill=INK2, font=fonts[2], anchor="ra")
    for i, c in enumerate(climbers):
        x0 = i * W
        top = Image.new("RGB", (W, HEAD), BG)
        draw_panel(ImageDraw.Draw(top), 0, c, fonts, scale_max)
        canvas.paste(top, (x0, 44))
        canvas.paste(c.image(), (x0, 44 + HEAD))
        if i:
            d.line((x0, 44, x0, canvas.height), fill=TRACK, width=2)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grips", default="850,500,350", help="grip capacities (N), comma separated")
    ap.add_argument("--profile", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--seconds", type=float, default=70.0, help="simulated seconds")
    ap.add_argument("--speed", type=float, default=4.0, help="playback speed (sim s per real s)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--live", action="store_true", help="show a window instead of only writing a video")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "renders" / "compare_hang.mp4")
    args = ap.parse_args()

    base = PRESETS[args.profile]
    climbers = [Climber(replace(base, name=f"grip{int(g)}", grip_capacity=float(g)))
                for g in args.grips.split(",")]
    fonts = (ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    size = (W * len(climbers), 44 + HEAD + H)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    dt = args.speed / args.fps
    frames = int(args.seconds / dt) + 1
    for k in range(frames):
        t = k * dt
        for c in climbers:
            c.advance_to(t)
        frame = cv2.cvtColor(np.asarray(compose(climbers, t, args.speed, fonts)), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        if args.live:
            cv2.imshow("compare_hang (q: quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if k % (args.fps * 5) == 0:
            print(f"t={t:5.1f}s  " + "  ".join(
                f"{c.contacts.grip_capacity:.0f}N: L {c.contacts.hands['left'].capacity:.0f}/R "
                f"{c.contacts.hands['right'].capacity:.0f}" for c in climbers))
    writer.release()
    cv2.destroyAllWindows()
    last = compose(climbers, climbers[0].d.time, args.speed, fonts)
    last.save(args.out.with_suffix(".png"))
    for c in climbers:
        rel = {s: (round(v[0], 2), v[1]) for s, v in c.released.items() if v}
        print(f"grip {c.contacts.grip_capacity:.0f} N: released {rel or 'never'}")
    print("saved", args.out, "and", args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
