"""Hang until the grip gives out, and log grip strength vs load over time.

    python scripts/hang_test.py                          # baseline (850 N grip)
    python scripts/hang_test.py --grip 400 --max 300     # weaker grip, up to 300 s

[한국어 설명]
벽에 매달려(PoseHold) 버티다가 악력이 떨어져 손을 놓칠 때까지 시뮬레이션하고,
0.5초마다 손별 악력·하중을 CSV로 저장한다 (renders/hang_<이름>.csv). 창 없이 빠르게 돈다.
"""
import argparse
import csv
import pathlib
import sys
import time
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from climber import PRESETS, ContactModel, PoseHold, build_model, climb_pose, test_wall


def hang(profile, max_seconds: float, log_every: float = 0.5):
    m = build_model(profile, test_wall())
    d = mujoco.MjData(m)
    q0, _ = climb_pose(m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    cm, ctrl = ContactModel(m, profile=profile), PoseHold(m, q0)
    cm.update(d)
    rows, next_log, fell = [], 0.0, None
    while d.time < max_seconds:
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(m, d)
        cm.update(d)
        h = cm.hands
        if d.time >= next_log:
            rows.append({"t": round(d.time, 3),
                         "cap_L": round(h["left"].capacity, 1), "cap_R": round(h["right"].capacity, 1),
                         "load_L": round(float(np.linalg.norm(h["left"].load)), 1),
                         "load_R": round(float(np.linalg.norm(h["right"].load)), 1),
                         "hold_L": h["left"].hold or "", "hold_R": h["right"].hold or ""})
            next_log += log_every
        if fell is None and (h["left"].hold is None or h["right"].hold is None):
            side = "left" if h["left"].hold is None else "right"
            fell = (round(d.time, 2), side, h[side].last_release)
        if h["left"].hold is None and h["right"].hold is None:
            break
    # 마지막 순간(손을 놓친 시점 포함)도 기록
    h = cm.hands
    rows.append({"t": round(d.time, 3),
                 "cap_L": round(h["left"].capacity, 1), "cap_R": round(h["right"].capacity, 1),
                 "load_L": round(float(np.linalg.norm(h["left"].load)), 1),
                 "load_R": round(float(np.linalg.norm(h["right"].load)), 1),
                 "hold_L": h["left"].hold or "", "hold_R": h["right"].hold or ""})
    return rows, fell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--grip", type=float, help="override grip_capacity (N)")
    ap.add_argument("--max", type=float, default=180.0, help="max simulated seconds")
    args = ap.parse_args()
    profile = PRESETS[args.profile]
    if args.grip:
        profile = replace(profile, name=f"{profile.name}_grip{int(args.grip)}", grip_capacity=args.grip)
    t0 = time.time()
    rows, fell = hang(profile, args.max)
    out = ROOT / "renders" / f"hang_{profile.name}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{profile.name}: first hand off at {fell} | simulated {rows[-1]['t']} s in {time.time() - t0:.0f} s | {out}")


if __name__ == "__main__":
    main()
