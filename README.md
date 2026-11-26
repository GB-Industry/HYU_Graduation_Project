# HYU_Graduation_Project
Adaptive bouldering character control in MuJoCo, with body-aware, natural-looking climbing motion.

## Mannequin climber (`climber/`)

A light-orange wooden-mannequin humanoid built only from MuJoCo primitives (no meshes),
so it stays cheap to simulate for RL.

- Free root + 25 actuated hinges (waist 3, each arm 5, each leg 6), torque motors with `ctrlrange -1..1`.
- Baseline: adult male, 1.75 m, 70 kg, arm span 1.75 m. Segment lengths follow Winter's
  height ratios; `total_mass` is split per segment with de Leva (1996) mass fractions
  (`climber.model.MASS_FRACTION`). Each segment shape carries its own mass and collides
  with the world. Joint balls are visual only. The climber does not collide with itself (yet).
- `climber.body_stats(model)` measures height, shoulder/hip height, arm span and mass of any profile.
- Hand grip: inactive `connect` equalities `grasp_<side>_<hold>`; set `data.eq_active` to grip.
  The grip-force limit (`BodyProfile.grip_capacity`) is not enforced yet.
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
python scripts/view.py --simulate            # physics on, hands gripped, zero torque
python scripts/render.py                     # headless PNG -> renders/
python -m unittest discover tests
```
