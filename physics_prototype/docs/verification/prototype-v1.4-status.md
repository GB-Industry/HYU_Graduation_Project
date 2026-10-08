# Prototype v1.4: Human-like Humanoid Fidelity Upgrade — Verification Status

## Executive Summary
Prototype v1.4 upgrades the character model from the reduced 14-DoF prototype topology to an anthropomorphic 25-DoF humanoid climber model (within the required 23–27 DoF design space), complete with explicit physiological body segments, realistic mass/inertia distribution, climbing range of motion (ROM), joint-group strength scaling, retargeted static four-point wall stance, and comprehensive headless test verification.

---

## 1. Anthropomorphic Model Architecture

### Kinematic Topology & DoF (25 Actuated Joints)
- **Pelvis / Root**: Freejoint root (`nq=32`, `nv=31`, `nu=25`).
- **Torso / Spine Chain** (3 DoF):
  - `waist_yaw` (hinge, Z axis, [-35, 35] deg)
  - `waist_pitch` (hinge, X axis, [-25, 45] deg)
  - `waist_roll` (hinge, Y axis, [-25, 25] deg)
  - Segment bodies: `pelvis` (9 kg), `abdomen` (12 kg), `chest` (18 kg), `head` (5 kg with neck capsule + head sphere).
- **Upper Limbs** (5 DoF x 2 = 10 DoF):
  - Shoulder spherical joint via 3 hinges: pitch ([-120, 160] deg), roll ([-95, 120] deg), yaw ([-80, 80] deg)
  - Elbow flexion: `left_elbow`, `right_elbow` (hinge, [0, 145] deg)
  - Wrist flexion: `left_wrist`, `right_wrist` (hinge, [-60, 60] deg)
  - End-effector sites preserved: `left_hand_site`, `right_hand_site` (and grasp sites `left_grasp_site`, `right_grasp_site`).
- **Lower Limbs** (6 DoF x 2 = 12 DoF):
  - Hip spherical joint via 3 hinges: pitch ([-110, 120] deg), roll ([-50, 60] deg), yaw ([-45, 45] deg)
  - Knee flexion: `left_knee`, `right_knee` (hinge, [0, 150] deg)
  - Ankle 2-axis joint: `left_ankle_pitch`, `left_ankle_roll` (pitch [-45, 30] deg, roll [-25, 25] deg)
  - Foot geom: Box geom with climbing shoe high-friction rubber (friction 1.8).
  - End-effector sites preserved: `left_foot_site`, `right_foot_site`.

### Anthropometric Mass Distribution
- Total character mass: **~76.5 kg** at `mass_scale=1.0` (weight = ~750 N).
- Head & Torso: ~44 kg (57.5% of total mass).
- Legs: ~24.4 kg (12.2 kg per leg; 31.9% of total mass).
- Arms: ~8.1 kg (4.05 kg per arm; 10.6% of total mass).
- All segment masses scale linearly with `ClimberProfile.mass_scale`.

### Joint-Group Actuator Strengths (Scaled by `strength_scale`)
- **Torso / Waist**: Yaw (90 N·m), Pitch (110 N·m), Roll (90 N·m)
- **Shoulders**: Pitch (85 N·m), Roll (85 N·m), Yaw (75 N·m)
- **Elbows**: 65 N·m
- **Wrists**: 35 N·m
- **Hips**: Pitch (140 N·m), Roll (110 N·m), Yaw (80 N·m)
- **Knees**: 130 N·m
- **Ankles**: Pitch (55 N·m), Roll (40 N·m)

---

## 2. Contact Exclusions & Physics Stability
1. **Hold Exclusions**: Hands, forearms, and shins excluded from hold collision spheres (`contact_{region.id}`) to prevent solver penalty fights with equality constraints and foot placement.
2. **Wall Exclusions**: Hands excluded from `wall_main` to prevent hand-wall collision normal forces from fighting `<connect>` tension when grasping holds close to the wall.
3. **Internal Segment Exclusions**: Adjacent body segments (`pelvis`-`abdomen`, `abdomen`-`chest`, `chest`-`head`, `pelvis`-`thighs`, `chest`-`upper_arms`, `thighs`-`shins`, `shins`-`feet`, `upper_arms`-`forearms`, `forearms`-`hands`) excluded from self-collision.

---

## 3. Retargeted Deterministic Static Climbing Stance
- **Initial Stance Configuration**:
  - Pelvis root: $x=-0.0134, y=-0.6419, z=1.2730, \text{quat}=[0.9968, -0.0794, -0.0096, -0.0006]$
  - Hand sites accurately reach H3 $(-0.38, -0.02, 1.48)$ and H4 $(0.35, -0.02, 1.55)$ within $< 1\,\text{mm}$.
  - Foot sites rest on top of H1 and H2 at $z=0.822\,\text{m}$ with $< 1\,\text{mm}$ penetration.
- **Simulation Under Full Gravity (Free Root, NO Root Stabilization)**:
  - Base profile settles at $z_{\text{root}} \approx 1.147\,\text{m}$, firmly supported on the wall.
  - Physical holding loads under settled equilibrium:
    - Left Hand: **~390–425 N**
    - Right Hand: **~480–495 N**
    - Footholds H1 and H2: Supporting substantial body weight with firm shoe contact.
  - With `check_grip=True`, base profile maintains hold stably throughout all 500 steps.
  - Real-physics break test: Weak profile (capacity 50 N) physically breaks and detaches (`data.eq_active == 0`), while strong profile holds (`data.eq_active == 1`).

---

## 4. Verification Suite Results

### Full Automated Unit Test Suite (37 tests)
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."; .venv\Scripts\python.exe -m unittest discover -s tests -v
```
**Results**:
- Ran 37 tests in 3.32s: **36 passed, 1 expected skip** (`test_missing_mujoco_has_actionable_error`).
- `test_mjcf_builder.py`: 11 tests pass (25 DoF verified, bilateral symmetry, mass scaling, ROM scaling, exclusions).
- `test_grasp.py`: 6 tests pass (load measurement, attach/detach, breakable grip with real physics, 500-step static stance).
- `test_runtime.py`: 6 tests pass (model compile & step, 4 end-effector sites, deterministic pose control error reduction, profile conditioning).
- `test_contact.py`: 7 tests pass (affordances, orientation, grip capacity evaluation).
- `test_schema.py`: 3 tests pass (profile bounds, wall validation, synthetic scene).
- `test_view_scene.py`: 4 tests pass (headless validation across all profiles and modes: `stance`, `pose`, `free`).

### Demo Script (`scripts/run_demo.py`)
```
Generated canonical scene, two profile-conditioned MJCF models, and schematic image.
long_reach_lower_grip SmokeResult(steps=200, time=0.400s, nq=32, nv=31, nu=25, finite=True)
long_reach_lower_grip pose control: steps=300 initial_err=0.311rad final_err=0.154rad error_reduced=True finite=True
compact_strong SmokeResult(steps=200, time=0.400s, nq=32, nv=31, nu=25, finite=True)
compact_strong pose control: steps=300 initial_err=0.311rad final_err=0.143rad error_reduced=True finite=True

--- Prototype v1.3: Contact & Grasp Foundation Demo ---
Contact eligibility check (H3): Hand=True, Foot=False
long_reach_lower_grip static stance: steps=400 finite=True supported=True root_z=+0.877m LH_load=496.8N RH_load=458.8N LH_attached=True RH_attached=True
compact_strong static stance: steps=400 finite=True supported=True root_z=+1.139m LH_load=557.5N RH_load=421.9N LH_attached=True RH_attached=True
Real physics breakable grip test (actual MuJoCo LH load = 378.3 N):
  weak profile (cap=41.1 N): maintain=False -> physically detached=True
  compact_strong (cap=821.3 N): maintain=True -> physically held=True
```

### Headless Scene & Viewer Validation (`scripts/view_scene.py --headless --mode stance`)
```powershell
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode stance
```
All modes (`stance`, `pose`, `free`) pass headless validation cleanly over SSH.

---

## 5. Scope Compliance
- No reinforcement learning introduced.
- No finger articulation (rigid hand segment with site contracts preserved).
- Native MuJoCo geometric primitives only (capsules, spheres, boxes).
- Free-root physics strictly preserved in `stance` and `free` modes.
- Visual debug root stabilization strictly isolated to `--mode pose`.
- Windows / SSH compatibility fully verified.
