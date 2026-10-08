# Prototype v1.5: Pose Retargeting, 3-Point Support & Reach Transition Foundation — Verification Status

## Executive Summary
Prototype v1.5 establishes the locomotion and stance foundation for the adaptive physics-based bouldering character, replacing brittle manual joint-angle posing with a structured retargeting mechanism, validating deterministic 3-point wall support under free-root physics, and establishing a 6-stage single-limb reach-and-reattach transition sequence.

Key accomplishments:
- **Stance Specification & Retargeting Engine (`boulder_v1.retargeter`)**:
  - Defined [`StanceSpecification`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/retargeter.py) to represent climbing intent via hold assignments, pelvis wall distance, lateral bias, and biomechanical posture priors.
  - Implemented [`solve_retargeted_stance()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/retargeter.py) using Damped Least Squares (DLS) Jacobian IK with nullspace projection toward natural climbing postures and joint limit constraints.
  - Converges to sub-millimeter accuracy ($< 1\,\text{mm}$ error) in 8–15 iterations across all climber profiles (`base`, `long_reach_lower_grip`, `compact_strong`).
- **Deterministic 3-Point Support (`boulder_v1.locomotion`)**:
  - Implemented [`simulate_three_point_support()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/locomotion.py), demonstrating that after detaching one hand via real `GraspManager.detach()`, the character maintains physical wall support under full gravity with a free root (NO artificial root stabilization).
  - Maintains root height $z > 1.0\,\text{m}$ ($> 0.8\,\text{m}$ requirement) throughout the simulation.
  - Preserves breakable grip physics: standard/strong profiles maintain support under measured MuJoCo equality loads, while weak profiles physically break and detach when capacity is exceeded.
- **Single-Limb Reach Foundation (`boulder_v1.locomotion`)**:
  - Implemented [`simulate_single_limb_reach()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/locomotion.py) executing a deterministic 6-stage sequence:
    1. `INITIAL_STANCE`: Equilibrium settling in 4-point wall stance.
    2. `RELEASE_LIMB`: Real detachment of the reaching limb via `GraspManager.detach()`.
    3. `SUPPORT_PHASE`: Free-root 3-point support maintenance under gravity.
    4. `REACH_PHASE`: Coordinated limb excursion away from the hold.
    5. `ATTACH_PHASE`: Approach to target hold, dynamic contact eligibility detection (`can_attach()`), and deterministic constraint reconnection (`attach()`).
    6. `STABILIZED_STANCE`: Settle into four-point stance with both hands secured and balanced load distribution.
  - Verified symmetrically for both right-hand (`RH -> H4`) and left-hand (`LH -> H3`) reach transitions.
- **Comprehensive Test Suite & Demo Updates**:
  - Added [`tests/test_retargeter.py`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/tests/test_retargeter.py) and [`tests/test_locomotion.py`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/tests/test_locomotion.py).
  - All 49 unit tests pass cleanly (48 passed, 1 expected skip, 0 failures).
  - Updated [`scripts/run_demo.py`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/scripts/run_demo.py) and [`scripts/view_scene.py`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/scripts/view_scene.py) to support `--mode transition` alongside `--mode stance`, `--mode pose`, and `--mode free`.

---

## 1. Stance Retargeting Engine Design

### Multi-Objective DLS IK Formulation
Rather than hardcoding 25 joint angles, [`solve_retargeted_stance()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/retargeter.py) formulates a multi-objective task Jacobian:
1. **Primary Task**: Active end-effector sites must align with target hold coordinates (hands at hold centers, feet wedged onto hold apex):
   $$\mathbf{J}_{\text{eff}} \Delta \mathbf{q} = \mathbf{x}_{\text{target}} - \mathbf{x}_{\text{site}}$$
2. **Secondary Task**: Pelvis root coordinates follow intended wall distance, lateral bias, and height:
   $$\mathbf{J}_{\text{pelvis}} \Delta \mathbf{q} = \mathbf{x}_{\text{pelvis\_target}} - \mathbf{x}_{\text{pelvis}}$$
3. **Nullspace Prior**: Biomechanical preference vector $\mathbf{q}_{\text{pref}}$ projected onto the nullspace of the primary task ensures natural elbow/knee flexion, torso orientation, and hip abduction without disturbing hold contact:
   $$\Delta \mathbf{q}_{\text{null}} = (\mathbf{I} - \mathbf{J}^{\dagger} \mathbf{J}) (\mathbf{q}_{\text{pref}} - \mathbf{q})$$
4. **Joint Limit Clamping**: Every iteration integrates positions via `mj_integratePos` and clamps actuated joints within physical ROM limits.

### Foothold Geometry & Wall Clearance
Footholds H1 and H2 are spherical holds centered at $y = -0.02\,\text{m}, z = 0.72\,\text{m}$ with radius $0.075\,\text{m}$ (apex at $z = 0.795\,\text{m}$). The vertical wall front surface is at $y = 0.00\,\text{m}$.
- Placing foot sites at $y \ge -0.015\,\text{m}$ causes the foot collision box to intersect the wall geom, creating explosive repulsive normal contact forces ($> 1500\,\text{N}$) that destabilize the root.
- The retargeter accurately positions the foot sites wedged at $y = -0.0595\,\text{m}, z = 0.8128\,\text{m}$, seating the shoe sole rubber firmly on the hold dome without wall penetration.

### Morphology Conditioning
The retargeter automatically adapts pelvis positioning according to the climber's segmental reach deltas:
$$\Delta z_{\text{pelvis}} = (\text{leg\_reach} - 0.82\,\text{m}) \times 0.85 + \text{pelvis\_height\_bias}$$
$$\Delta y_{\text{pelvis}} = -d_{\text{wall}} - (\text{arm\_reach} - 0.58\,\text{m}) \times 0.70$$
This ensures morphological variations (`long_reach_lower_grip`, `compact_strong`) achieve sub-millimeter convergence without manual per-profile tuning.

---

## 2. Deterministic 3-Point Wall Support

### Physical Protocol
In [`simulate_three_point_support()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/locomotion.py):
1. Initializes into the 4-point wall stance with both hands attached.
2. Settles for equilibrium.
3. Smoothly shifts posture toward the supporting side (e.g. lateral bias $-0.16\,\text{m}$ for right hand release).
4. Detaches the released limb via `gm.detach()`.
5. Simulates under full gravity with a free root (no artificial root stabilization).

### Verification Metrics
- **Numerical Stability**: All state values remain strictly finite ($0$ NaNs, $0$ infinities).
- **Wall Support**: Final root elevation remains at $z \approx 1.035\,\text{m}$ (well above $0.8\,\text{m}$).
- **Load Measurement**: Measured load on supporting hand (e.g. LH on H3) is $\approx 424\,\text{N}$, which is within the $644\,\text{N}$ effective capacity of the base climber profile.
- **Physical Breakable Grip**: When evaluated with `check_grip=True`, a profile with low grip capacity ($200\,\text{N}$) correctly detaches and falls, confirming that support is maintained by real physical grip strength.

---

## 3. Single-Limb Reach & Reattach Foundation

### 6-Stage Transition Architecture
[`simulate_single_limb_reach()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/locomotion.py) implements the full locomotion transition:

```
[INITIAL_STANCE] -> [RELEASE_LIMB] -> [SUPPORT_PHASE] -> [REACH_PHASE] -> [ATTACH_PHASE] -> [STABILIZED_STANCE]
     (0..30)             (30)            (31..50)          (51..80)         (81..120)          (121..200+)
```

1. **Initial Stance**: Humanoid starts in static four-point stance with both hands attached.
2. **Release Limb**: At step 30, `GraspManager.detach(limb)` deactivates the equality constraint.
3. **Support Phase**: Supporting limbs bear body weight under free-root physics.
4. **Reach Phase**: Reaching arm flexes back into an exploratory hover pose ($\sim 10$–$15\,\text{cm}$ from hold).
5. **Attach Phase**: Reaching arm extends toward target hold. Dynamic contact eligibility is verified via `GraspManager.can_attach()`. Upon satisfying distance and orientation thresholds, `GraspManager.attach()` activates the equality constraint without model recompilation.
6. **Stabilized Stance**: Both hands secured; body settles into balanced four-point stance ($z_{\text{root}} > 1.0\,\text{m}$).

### Symmetrical Transition Verification
- **Right Hand (`RH -> H4`)**:
  - `phase_history`: `('initial_stance', 'release_limb', 'support_phase', 'reach_phase', 'attach_phase', 'stabilized_stance')`
  - `eligibility_detected`: `True`
  - `reattached`: `True`
  - `final_root_z`: $1.019\,\text{m}$
  - `attachment_loads`: LH = $478.1\,\text{N}$, RH = $477.2\,\text{N}$
- **Left Hand (`LH -> H3`)**:
  - `phase_history`: `('initial_stance', 'release_limb', 'support_phase', 'reach_phase', 'attach_phase', 'stabilized_stance')`
  - `eligibility_detected`: `True`
  - `reattached`: `True`
  - `final_root_z`: $1.026\,\text{m}$
  - `attachment_loads`: LH = $540.9\,\text{N}$, RH = $476.9\,\text{N}$

---

## 4. Test Suite & Verification Results

### Complete Test Discovery (49 tests)
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
All 49 unit tests passed (48 passed, 1 expected skip):
- `tests/test_retargeter.py`: 5 tests pass (defaults, submillimeter convergence, joint limits, profile variations, free limbs).
- `tests/test_locomotion.py`: 5 tests pass (3-point RH/LH support, breakable grip detachment, 6-phase RH/LH reach transitions).
- `tests/test_contact.py`: 7 tests pass.
- `tests/test_grasp.py`: 6 tests pass.
- `tests/test_mjcf_builder.py`: 13 tests pass.
- `tests/test_runtime.py`: 6 tests pass (1 skip for missing mujoco test).
- `tests/test_schema.py`: 3 tests pass.
- `tests/test_view_scene.py`: 4 tests pass (including `--mode transition`).

### Demo Verification (`scripts/run_demo.py`)
```
--- Prototype v1.5: Retargeting, 3-Point Support & Reach Transition ---
Stance Retargeter (long_reach_lower_grip): converged=True iters=6 max_err=0.99mm
3-Point Support (long_reach_lower_grip, RH released): finite=True supported=True min_z=1.041m LH_load=724.9N
Single-Limb Reach (long_reach_lower_grip, RH): finite=True supported=True eligibility=True reattached=True final_z=1.027m
Phases traversed: initial_stance -> release_limb -> support_phase -> reach_phase -> attach_phase -> stabilized_stance
```

### Headless Scene Verification Across Modes
```powershell
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode transition
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode stance
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode pose
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode free
```
All modes pass with return code 0.
