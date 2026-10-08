# Prototype v1.2: Visible Physics Foundation — Verification Status

## Milestone Overview
**Prototype v1.2: Visible Physics Foundation** extends the M0 physics foundation by providing:
1. An interactive and headless MuJoCo viewer entry point (`scripts/view_scene.py`) displaying the wall, holds/contact regions, and humanoid climber in the same scene.
2. Four explicit, high-contrast end-effector sites (`left_hand_site`, `right_hand_site`, `left_foot_site`, `right_foot_site`) queryable from both the model and simulation data.
3. A deterministic pose-control sanity check demonstrating that actuators can drive the character toward a predefined climbing pose without numerical instability (no RL).
4. Full headless verification compatibility for SSH sessions where Windows desktop window stations are restricted.

---

## Environment
- OS: Windows 11
- Python: 3.11 (`.venv\Scripts\python.exe`)
- MuJoCo: 3.14.0
- Path handling: Standard library `pathlib.Path` throughout (no hardcoded OS paths)

---

## Verified Capabilities

### 1. Scene & Visual Foundation
- **Unified Scene**: Wall box geometry, 7 contact regions (holds with visual spherical geoms + bright sites), and the 14-DoF humanoid climber compiled together in MuJoCo.
- **End-Effector Sites**:
  - `left_hand_site`: sphere `size=0.024`, crimson `rgba="0.92 0.22 0.20 0.90"`
  - `right_hand_site`: sphere `size=0.024`, azure `rgba="0.20 0.50 0.92 0.90"`
  - `left_foot_site`: sphere `size=0.024`, amber `rgba="0.92 0.58 0.12 0.90"`
  - `right_foot_site`: sphere `size=0.024`, emerald `rgba="0.18 0.82 0.38 0.90"`
- **Queryable End-Effectors**: `get_end_effector_positions(model, data)` extracts 3D Cartesian coordinates for all 4 limbs, verified to return finite coordinates.
- **Hold Visualization**: Contact regions contain distinct visual sphere sites (`rgba="1.0 0.82 0.1 0.9"`) on holds (`geom rgba="0.25 0.45 0.8 1"`).

### 2. Deterministic Pose Control Sanity Check
- **Controller**: Proportional-Derivative (PD) closed-loop control on the 14 actuated joints:
  $$\tau_i = \text{clip}(K_p (q_i^* - q_i) - K_d \dot{q}_i, -1.0, 1.0)$$
- **Target Pose**: Predefined climbing stance (`DEFAULT_TARGET_POSE`) with shoulder reach (+45°), elbow flexion (+60°), hip flexion (+30°), and knee flexion (+45°).
- **Convergence & Stability**:
  - Initial mean error: ~27.86° (0.486 rad)
  - Final mean error: ~1.25° (0.022 rad) after 400 steps
  - `result.error_reduced`: True
  - `result.finite`: True (all `qpos`, `qvel`, and site positions remain strictly finite; no NaNs, Infs, or divergence)
- **Profile Conditioning Preserved**: Tested across `base`, `compact_strong`, and `long_reach_lower_grip` profiles. End-effector positions differ deterministically based on segment lengths.

### 3. Viewer Architecture & Headless SSH Compatibility
- **Script**: `scripts/view_scene.py`
- **Headless Mode (`--headless`)**: Full model compilation, end-effector query, and 400-step pose-control simulation executed headlessly without opening any GUI window.
- **SSH Auto-Detection**: When run over SSH without desktop window access, automatically runs headless validation, reports all metrics, and prints the exact local Windows launch command.
- **Interactive Mode**: Local Windows launch opens `mujoco.viewer.launch_passive` with real-time PD pose control or passive physics, supporting interactive camera controls and pause/play.

---

## Test Suite Results

Command:
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."; .venv\Scripts\python.exe -m unittest discover -s tests -v
```

Output:
```
test_grip_breaks_above_capacity (test_contact.ContactTests.test_grip_breaks_above_capacity) ... ok
test_limb_affordance_compatibility (test_contact.ContactTests.test_limb_affordance_compatibility) ... ok
test_orientation_penalizes_wrong_facing_hand (test_contact.ContactTests.test_orientation_penalizes_wrong_facing_hand) ... ok
test_contact_region_sites_have_rgba (test_mjcf_builder.MJCFBuilderTests.test_contact_region_sites_have_rgba) ... ok
test_end_effector_sites_defined (test_mjcf_builder.MJCFBuilderTests.test_end_effector_sites_defined) ... ok
test_free_root_has_positive_inertial (test_mjcf_builder.MJCFBuilderTests.test_free_root_has_positive_inertial) ... ok
test_generated_xml_is_well_formed (test_mjcf_builder.MJCFBuilderTests.test_generated_xml_is_well_formed) ... ok
test_mass_scale_changes_density (test_mjcf_builder.MJCFBuilderTests.test_mass_scale_changes_density) ... ok
test_morphology_changes_segment_length (test_mjcf_builder.MJCFBuilderTests.test_morphology_changes_segment_length) ... ok
test_rom_changes_joint_range (test_mjcf_builder.MJCFBuilderTests.test_rom_changes_joint_range) ... ok
test_strength_changes_actuator_gear (test_mjcf_builder.MJCFBuilderTests.test_strength_changes_actuator_gear) ... ok
test_deterministic_pose_control_stays_finite (test_runtime.RuntimeTests.test_deterministic_pose_control_stays_finite) ... ok
test_end_effector_sites_queryable (test_runtime.RuntimeTests.test_end_effector_sites_queryable) ... ok
test_missing_mujoco_has_actionable_error (test_runtime.RuntimeTests.test_missing_mujoco_has_actionable_error) ... skipped 'MuJoCo is installed; missing-dependency path not applicable'
test_model_compiles_and_steps (test_runtime.RuntimeTests.test_model_compiles_and_steps) ... ok
test_pose_control_preserves_profile_variations (test_runtime.RuntimeTests.test_pose_control_preserves_profile_variations) ... ok
test_profile_rejects_nonpositive_morphology (test_schema.SchemaTests.test_profile_rejects_nonpositive_morphology) ... ok
test_synthetic_scene_references_known_contacts (test_schema.SchemaTests.test_synthetic_scene_references_known_contacts) ... ok
test_wall_rejects_nonpositive_friction (test_schema.SchemaTests.test_wall_rejects_nonpositive_friction) ... ok
test_cli_headless_flag (test_view_scene.ViewSceneTests.test_cli_headless_flag) ... ok
test_headless_validation_all_profiles (test_view_scene.ViewSceneTests.test_headless_validation_all_profiles) ... ok
test_headless_validation_base_profile (test_view_scene.ViewSceneTests.test_headless_validation_base_profile) ... ok

----------------------------------------------------------------------
Ran 22 tests in 0.934s

OK (skipped=1)
```

Result: **22 tests total: 21 passed, 1 expected skip**.

---

## Visual Viewer Launch Command

From a local Windows terminal (PowerShell or CMD) on the host desktop:
```powershell
cd C:\Users\오유찬\Desktop\project\boulder_prototype
.venv\Scripts\python.exe scripts\view_scene.py --profile compact_strong --mode pose
```

Available options:
- `--profile {base, compact_strong, long_reach_lower_grip}`
- `--mode {pose, passive}`
- `--headless` (for non-GUI validation)
- `--duration SECONDS`

---

## Known Limitations
- No active contact equality constraints (hand-to-hold attachment) are engaged during free pose control; the root remains a freejoint in `free` mode.
- Actuators are 14 DoF prototype motors rather than the planned 23–27 full humanoid biomechanical actuators.
- Contact friction dynamics on the wall are computed by MuJoCo solver, but temporary adhesive grasping is reserved for Prototype v1.3.

---

## v1.2.1 Usability & Control Stabilization Note

### 1. Root Cause Analysis
- **Falling and Jittering in `pose` mode**:
  - Initial state: Root pos at `(0.0, -0.55, 1.55)`, feet at $z \approx 0.485$m. There is no ground or wall contact support at $t=0$.
  - The humanoid is in mid-air and falls under gravity ($g = -9.81$ m/s$^2$) to the floor ($z=0$) in ~1.0s.
  - The PD controller only drives the 14 actuated joints; the 6-DoF root is unactuated and unsupported.
  - After landing on the floor, the limbs collide with the floor plane. The PD controller continues applying torques fighting floor contact constraints, resulting in continuous micro-chatter/jitter.
- **Initial Camera Zoom**:
  - Default MuJoCo camera points at $(0, 0, 0)$ with a fixed 2m distance, clipping the 3m wall and placing the humanoid at the screen edge.

### 2. Implemented Fixes
- **Dynamic Camera Framing**:
  - Camera parameters are dynamically computed from `model.stat.center` and `model.stat.extent`:
    - `lookat`: `(center[0], center[1] - 0.30, center[2] * 0.95)` (centers between wall and climber at mid-height: `[0.0, -0.25, 1.42]`).
    - `distance`: `extent * 1.15` ($\approx 4.35$m), comfortably framing the full 2.3m $\times$ 3.0m wall and humanoid.
    - `elevation`: $-12.0^\circ$, `azimuth`: $105.0^\circ$ (slight 3/4 perspective for clear 3D depth).
  - User camera mouse interaction remains fully available.
- **Explicit Viewer Modes**:
  - `--mode pose` (default): Deterministic actuator/morphology/site visualization. Temporary root support holds the root at initial position so the character can be inspected assuming its target pose without falling or floor jitter. (Explicitly documented as an inspection aid, not a balance policy).
  - `--mode free`: Real free-root physics without stabilization. Falling under gravity is expected and unconstrained.
  - Startup banner explicitly states active mode and clarifies CLI selection.
- **Deterministic Headless Tests**:
  - Added unit test for camera calculation (`test_camera_framing_derived_from_model_stat`).
  - Added tests for `run_pose_control(..., stabilize_root=True)`.
  - Added CLI tests for `--headless --mode pose` and `--headless --mode free`.
  - Full suite: **24 tests total: 23 passed, 1 expected skip**.

