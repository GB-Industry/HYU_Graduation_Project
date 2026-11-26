# HYU_Graduation_Project
Adaptive bouldering character control in MuJoCo, with body-aware, natural-looking climbing motion.

## Mannequin climber (`climber/`)

A light-orange wooden-mannequin humanoid built only from MuJoCo primitives (no meshes),
so it stays cheap to simulate for RL.

- Free root + 27 actuated hinges (waist 3, each arm 6 incl. forearm twist, each leg 6), torque motors with `ctrlrange -1..1`.
- Baseline: adult male, 1.75 m, 70 kg, arm span 1.75 m. Segment lengths follow Winter's
  height ratios; `total_mass` is split per segment with de Leva (1996) mass fractions
  (`climber.model.MASS_FRACTION`). Each segment shape carries its own mass and collides
  with the world. Joint balls are visual only. The climber does not collide with itself (yet).
- `climber.body_stats(model)` measures height, shoulder/hip height, arm span and mass of any profile.
- Hand grip / foot support: `climber.ContactModel` (call `update(data)` every step).
  A hand grips when its palm touches a hold and faces it within 20 degrees; the link is anchored at
  the actual contact point. It lets go when the load on the hold leaves that hold's usable pull
  direction (`Hold.pull_dir` +/- `pull_cone_deg`, default: a jug pulled downward +/- 40 deg) or exceeds
  the grip capacity. Grip capacity fatigues while holding (above a 30% floor, halving every 60 s at
  full load, slower at lighter load) and recovers when the hand rests (95% in 30 s); values live in
  `BodyProfile`. Feet are never fixed; they are only judged
  "supported" (pressing >= 10 N, sole facing the surface within 20 degrees, not slipping).
- `climber.test_wall()`: 4-hold wall sized for the 1.75 m body, used by tests, viewer and render.
- `climber.PoseHold`: baseline PD controller that holds the start pose; with it the climber hangs on its own.
- `BodyProfile` holds the per-climber traits (segment lengths, hip/shoulder width, mass,
  range of motion, strength, grip). Presets: `baseline`, `short_limbs` (-5% limbs, 1.71 m),
  `long_limbs` (+5% limbs, 1.79 m).

```python
from climber import BASELINE, build_model, climb_pose
model = build_model(BASELINE)
qpos, info = climb_pose(model)   # four-point stance on the default wall
```

## Scripts

```bash
pip install -r requirements.txt
python scripts/view.py                       # interactive viewer, frozen wall stance
python scripts/view.py --pose t --profile long_limbs
python scripts/view.py --mode hang           # physics on, PoseHold: hangs from both holds on its own
python scripts/view.py --mode assist         # hang + pelvis held up: pull a hand with the mouse to see it let go
python scripts/view.py --mode fall           # physics on, zero torque (no posture control yet -> falls)
python scripts/render.py                     # headless PNG -> renders/
python scripts/hang_test.py --grip 500       # hang until the grip gives out, log strength/load -> renders/*.csv
python scripts/plot_hang.py renders/hang_*.csv   # chart of grip strength vs load
python scripts/compare_hang.py --live        # 3 climbers (850/500/350 N grip) side by side, grip/load/fatigue above each -> renders/compare_hang.mp4
python -m unittest discover tests
```
