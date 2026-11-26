"""Plot grip strength vs hand load over time from hang_test.py CSVs.

    python scripts/plot_hang.py renders/hang_baseline.csv renders/hang_baseline_grip500.csv ...

[한국어 설명]
hang_test.py가 만든 CSV를 읽어, 체형(악력)별로 한 칸씩 '악력(파랑) vs 손에 걸린 하중(주황)' 그래프를 그린다.
두 선이 만나는 순간 = 악력이 하중보다 작아져 손을 놓치는 순간. 결과: renders/hang_fatigue.png
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAPACITY, LOAD = "#2a78d6", "#eb6834"   # categorical slots 1, 2 (reference palette, light)
plt.rcParams["font.family"] = "Malgun Gothic"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t"]) for r in rows]
    cap = [float(r["cap_L"]) for r in rows]
    # 놓친 뒤에는 하중이 0이 되므로, 잡고 있는 동안만 하중을 그린다
    ld = [float(r["load_L"]) if r["hold_L"] else None for r in rows]
    return t, cap, ld


def main(paths):
    fig, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.2), sharey=True, facecolor=SURFACE)
    axes = axes if len(paths) > 1 else [axes]
    ymax = 0
    for ax, path in zip(axes, paths):
        t, cap, ld = load(path)
        ymax = max(ymax, max(cap))
        ax.set_facecolor(SURFACE)
        ax.plot(t, cap, color=CAPACITY, lw=2, solid_capstyle="round")
        lt = [x for x, y in zip(t, ld) if y is not None]
        ly = [y for y in ld if y is not None]
        ax.plot(lt, ly, color=LOAD, lw=2, solid_capstyle="round")
        name = pathlib.Path(path).stem.replace("hang_", "")
        base = cap[0]
        if lt and len(lt) < len(t):   # 마지막 기록에서 손을 놓친 상태
            ax.axvline(lt[-1], color=INK2, lw=1, ls=(0, (3, 3)))
            ax.annotate(f"손 놓침 {lt[-1]:.1f}초", (lt[-1], ly[-1]), xytext=(8, 18), textcoords="offset points",
                        color=INK, fontsize=10)
        else:
            ax.annotate(f"{t[-1]:.0f}초 동안 버팀", (t[-1], cap[-1]), xytext=(-8, 10), textcoords="offset points",
                        ha="right", color=INK, fontsize=10)
        ax.annotate("악력", (t[0], cap[0]), xytext=(4, 6), textcoords="offset points", color=INK2, fontsize=9)
        if ly:
            ax.annotate("손에 걸린 하중", (lt[0], ly[0]), xytext=(4, 6), textcoords="offset points", color=INK2, fontsize=9)
        ax.set_title(f"악력 {base:.0f} N", color=INK, fontsize=12, loc="left")
        ax.set_xlabel("시간 (초)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("힘 (N, 왼손)", color=INK2)
    axes[0].set_ylim(0, ymax * 1.12)
    fig.legend(handles=[plt.Line2D([], [], color=CAPACITY, lw=2, label="악력 (지치면서 감소)"),
                        plt.Line2D([], [], color=LOAD, lw=2, label="손에 걸린 하중")],
               loc="upper right", frameon=False, ncol=2, labelcolor=INK)
    fig.suptitle("매달려 버티기: 악력이 하중 아래로 떨어지면 손을 놓친다", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = ROOT / "renders" / "hang_fatigue.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
