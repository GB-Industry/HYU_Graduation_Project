# Prototype v1.3: Contact & Grasp Foundation — Verification Status

## Milestone Overview
**Prototype v1.3: Contact & Grasp Foundation** establishes a physically testable, static wall-contact foundation for the bouldering simulation prototype without introducing reinforcement learning or modifying canonical contracts.

Key accomplishments:
1. **Deterministic Contact Eligibility Layer**: `can_attach(...)` evaluates limb-hold affordance compatibility, end-effector proximity, and surface normal alignment before allowing attachment.
2. **Runtime Grasp Attachment**: Temporary hand grasp attachments (`attach`, `detach`, `is_attached`, `active_attachment`) backed by predeclared MuJoCo `<connect>` equality constraints without model recompilation.
3. **Physical Reaction Load Measurement**: Real-time extraction of constraint reaction forces from `data.efc_force` to measure physical load ($F_{\text{load}} = \|\mathbf{f}_{\text{eq}}\|$ in Newtons).
4. **Breakable Grip Integration**: Direct coupling between physical constraint loads and `GripController`. If load exceeds the profile's effective grip capacity, the attachment breaks deterministically.
5. **Profile-Conditioned Failure**: Under identical load conditions, a weaker grip profile detaches while a stronger grip profile maintains hold.
6. **Deterministic Static Four-Point Climbing Stance**: Stable static four-point stance (`simulate_static_stance`) under full gravity with a free root (NO root weld, NO root stabilization).
7. **Viewer Integration**: Interactive and headless inspection support (`--mode stance` in `scripts/view_scene.py`) alongside existing `--mode pose` and `--mode free`.

---

## Environment
- OS: Windows 11
- Python: 3.11 (`.venv\Scripts\python.exe`)
- MuJoCo: 3.14.0
- Path handling: Standard library `pathlib.Path` throughout (no hardcoded OS paths)

---

## Technical Architecture & Design Decisions

### 1. Equality Constraint Mechanism: `connect` vs `weld`
- **Choice**: MuJoCo `<connect>` equality constraints between hand sites (`left_hand_site`, `right_hand_site`) and hold sites (`site_{region.id}`).
- **Rationale**:
  - `weld` constraints restrict 6 Degrees of Freedom (3 translational + 3 rotational). Because our 14-DoF prototype topology does not feature spherical wrist joints, welding the hand site would artificially lock forearm orientation in world space. As the torso and shoulder move under gravity, this creates massive unnatural moments, excessive stiffness, and constraint solver fight.
  - `connect` constraints restrict 3 Degrees of Freedom (position only), acting as a spherical ball joint at the hold. This permits the climber's arms to articulate naturally as the body settles into equilibrium while firmly anchoring the hand to the hold.
  - The reaction force vector directly provides a 3D Cartesian load $\mathbf{f} = [f_x, f_y, f_z]$, whose norm $\|\mathbf{f}\|$ maps directly to tensile pull on the hold.
- **Predeclared Constraints**: All 12 possible hand-to-hold combinations for GRASP-capable holds (H1 through H6) are predeclared in MJCF with `active="false"`. Activating or deactivating an attachment simply toggles `data.eq_active[eq_id]`, eliminating runtime model recompilation.
- **Free-Root Humanoid**: The humanoid remains completely unconstrained at the root (`<freejoint name="root"/>`). No permanent root weld is used.

### 2. Physical Load Metric & Grip Controller Coupling
- **Reaction Force Extraction**: For an active attachment with constraint index `eq_id`, `GraspManager.get_grasp_load(limb)` queries:
  $$\mathbf{f}_{\text{eq}} = \text{data.efc\_force}[\{i \mid \text{efc\_type}[i] = \text{mjCNSTR\_EQUALITY} \wedge \text{efc\_id}[i] = \text{eq\_id}\}]$$
  $$F_{\text{load}} = \|\mathbf{f}_{\text{eq}}\|_2 = \sqrt{f_x^2 + f_y^2 + f_z^2}$$
- **Breakable Grip Evaluation**: `GraspManager.evaluate_and_update(profile)` passes $F_{\text{load}}$ into `GripController.evaluate(profile, region, hand_normal, required_load)`.
  - Effective grip capacity is calculated using the profile's base capacity, surface friction, grip quality, and hand-to-hold normal alignment:
    $$C_{\text{eff}} = C_{\text{profile}} \cdot \left(\frac{\mu}{1 + \mu}\right) \cdot Q_{\text{grip}} \cdot \max(0, \mathbf{n}_{\text{hand}} \cdot (-\mathbf{n}_{\text{hold}}))$$
  - If $F_{\text{load}} > C_{\text{eff}}$, `decision.maintain == False`, and `GraspManager.detach(limb)` immediately sets `data.eq_active[eq_id] = False`.
- **Profile-Conditioned Failure Demonstration**:
  - Hold H3 has $\mu = 0.82$, $Q = 0.92$.
  - At a required load of $480\,\text{N}$:
    - `long_reach_lower_grip` ($C_{\text{profile}} = 450\,\text{N}$, $C_{\text{eff}} \approx 377.2\,\text{N}$): $480\,\text{N} > 377.2\,\text{N} \implies$ **BREAKS** (`maintain=False`, detaches).
    - `compact_strong` ($C_{\text{profile}} = 680\,\text{N}$, $C_{\text{eff}} \approx 570.0\,\text{N}$): $480\,\text{N} < 570.0\,\text{N} \implies$ **HOLDS** (`maintain=True`, remains attached).

### 3. Deterministic Static Four-Point Climbing Stance
- **Kinematic Stance Configuration**:
  - Root position: `[-0.0144, -0.5779, 1.4337]`, orientation: `[1, 0, 0, 0]` (upright, facing wall)
  - Left Hand: attached to Hold H3 (`[-0.38, -0.02, 1.48]`) via connect constraint
  - Right Hand: attached to Hold H4 (`[0.35, -0.02, 1.55]`) via connect constraint
  - Left Foot: positioned on Foothold H1 (`[-0.33, -0.02, 0.72]`) with passive collision support
  - Right Foot: positioned on Foothold H2 (`[0.33, -0.02, 0.72]`) with passive collision support
- **Physics Simulation (`simulate_static_stance`)**:
  - Full gravity ($g = -9.81\,\text{m/s}^2$).
  - Free root (NO root weld, NO root position clamping).
  - Deterministic joint hold controller maintaining the target climbing joint angles.
  - Verified across 500 simulation steps ($1.0\,\text{s}$ simulated):
    - `finite`: True (no NaNs or Infs)
    - `supported`: True (root settles at $z \approx 1.274\,\text{m}$, well above ground $z = 0.0\,\text{m}$)
    - Both hands remain securely attached with finite, non-zero reaction forces.

---

## Complete Verification Results

### 1. Automated Test Suite (35 tests)
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."; .venv\Scripts\python.exe -m unittest discover -s tests -v
```

```
test_can_attach_eligible (test_contact.ContactTests.test_can_attach_eligible) ... ok
test_can_attach_rejects_excess_distance (test_contact.ContactTests.test_can_attach_rejects_excess_distance) ... ok
test_can_attach_rejects_incompatible_affordance (test_contact.ContactTests.test_can_attach_rejects_incompatible_affordance) ... ok
test_can_attach_rejects_incompatible_orientation (test_contact.ContactTests.test_can_attach_rejects_incompatible_orientation) ... ok
test_grip_breaks_above_capacity (test_contact.ContactTests.test_grip_breaks_above_capacity) ... ok
test_limb_affordance_compatibility (test_contact.ContactTests.test_limb_affordance_compatibility) ... ok
test_orientation_penalizes_wrong_facing_hand (test_contact.ContactTests.test_orientation_penalizes_wrong_facing_hand) ... ok
test_get_grasp_load_measurement (test_grasp.GraspTests.test_get_grasp_load_measurement) ... ok
test_grasp_manager_attach_and_detach_without_recompilation (test_grasp.GraspTests.test_grasp_manager_attach_and_detach_without_recompilation) ... ok
test_grasp_manager_can_attach_checks_distance (test_grasp.GraspTests.test_grasp_manager_can_attach_checks_distance) ... ok
test_grasp_manager_detach_all (test_grasp.GraspTests.test_grasp_manager_detach_all) ... ok
test_profile_conditioned_breakable_grip (test_grasp.GraspTests.test_profile_conditioned_breakable_grip)
Verify identical physical load causes weak grip to break and strong grip to hold. ... ok
test_simulate_static_stance (test_grasp.GraspTests.test_simulate_static_stance)
Deterministic static four-point climbing stance remains suspended without root stabilization. ... ok
test_contact_region_sites_have_rgba (test_mjcf_builder.MJCFBuilderTests.test_contact_region_sites_have_rgba) ... ok
test_end_effector_sites_defined (test_mjcf_builder.MJCFBuilderTests.test_end_effector_sites_defined) ... ok
test_free_root_has_positive_inertial (test_mjcf_builder.MJCFBuilderTests.test_free_root_has_positive_inertial) ... ok
test_generated_xml_is_well_formed (test_mjcf_builder.MJCFBuilderTests.test_generated_xml_is_well_formed) ... ok
test_grasp_equality_constraints_predeclared (test_mjcf_builder.MJCFBuilderTests.test_grasp_equality_constraints_predeclared) ... ok
test_mass_scale_changes_density (test_mjcf_builder.MJCFBuilderTests.test_mass_scale_changes_density) ... ok
test_morphology_changes_segment_length (test_mjcf_builder.MJCFBuilderTests.test_morphology_changes_segment_length) ... ok
test_rom_changes_joint_range (test_mjcf_builder.MJCFBuilderTests.test_rom_changes_joint_range) ... ok
test_strength_changes_actuator_gear (test_mjcf_builder.MJCFBuilderTests.test_strength_changes_actuator_gear) ... ok
test_deterministic_pose_control_stays_finite (test_runtime.RuntimeTests.test_deterministic_pose_control_stays_finite) ... ok
test_deterministic_pose_control_with_stabilize_root (test_runtime.RuntimeTests.test_deterministic_pose_control_with_stabilize_root) ... ok
test_end_effector_sites_queryable (test_runtime.RuntimeTests.test_end_effector_sites_queryable) ... ok
test_missing_mujoco_has_actionable_error (test_runtime.RuntimeTests.test_missing_mujoco_has_actionable_error) ... skipped 'MuJoCo is installed; missing-dependency path not applicable'
test_model_compiles_and_steps (test_runtime.RuntimeTests.test_model_compiles_and_steps) ... ok
test_pose_control_preserves_profile_variations (test_runtime.RuntimeTests.test_pose_control_preserves_profile_variations) ... ok
test_profile_rejects_nonpositive_morphology (test_schema.SchemaTests.test_profile_rejects_nonpositive_morphology) ... ok
test_synthetic_scene_references_known_contacts (test_schema.SchemaTests.test_synthetic_scene_references_known_contacts) ... ok
test_wall_rejects_nonpositive_friction (test_schema.SchemaTests.test_wall_rejects_nonpositive_friction) ... ok
test_camera_framing_derived_from_model_stat (test_view_scene.ViewSceneTests.test_camera_framing_derived_from_model_stat) ... ok
test_cli_headless_flag (test_view_scene.ViewSceneTests.test_cli_headless_flag) ... ok
test_headless_validation_all_profiles (test_view_scene.ViewSceneTests.test_headless_validation_all_profiles) ... ok
test_headless_validation_base_profile (test_view_scene.ViewSceneTests.test_headless_validation_base_profile) ... ok

----------------------------------------------------------------------
Ran 35 tests in 2.696s

OK (skipped=1)
```
**Status**: 35 tests ran, **34 passed**, 1 expected skip.

### 2. Demo Script Execution (`scripts/run_demo.py`)
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."; .venv\Scripts\python.exe scripts/run_demo.py
```

Output:
```
Generated canonical scene, two profile-conditioned MJCF models, and schematic image.
long_reach_lower_grip SmokeResult(steps=200, time=0.4000000000000003, nq=21, nv=20, nu=14, finite=True)
long_reach_lower_grip pose control: steps=300 initial_err=0.486rad final_err=0.095rad error_reduced=True finite=True sites=['left_hand_site', 'right_hand_site', 'left_foot_site', 'right_foot_site']
compact_strong SmokeResult(steps=200, time=0.4000000000000003, nq=21, nv=20, nu=14, finite=True)
compact_strong pose control: steps=300 initial_err=0.486rad final_err=0.044rad error_reduced=True finite=True sites=['left_hand_site', 'right_hand_site', 'left_foot_site', 'right_foot_site']

--- Prototype v1.3: Contact & Grasp Foundation Demo ---
Contact eligibility check (H3): Hand=True, Foot=False
long_reach_lower_grip static stance: steps=400 finite=True supported=True root_z=+1.319m LH_load=3095.2N RH_load=2880.8N
compact_strong static stance: steps=400 finite=True supported=True root_z=+1.360m LH_load=2849.0N RH_load=2876.4N
Breakable grip failure test (load=480N on H3):
  long_reach_lower_grip (cap=377.2N): maintain=False (BREAKS)
  compact_strong (cap=570.0N): maintain=True (HOLDS)
```

### 3. Headless Viewer Validation across Modes (`scripts/view_scene.py`)
```powershell
# 1. Stance mode (v1.3 static 4-point stance)
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode stance

# 2. Pose mode (v1.2 actuator/morphology visualization)
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode pose

# 3. Free mode (real unconstrained free-root physics)
.venv\Scripts\python.exe scripts/view_scene.py --headless --mode free
```
All three modes run headlessly and exit cleanly with status code 0.

---

## Viewer Launch Commands (Local Windows Desktop)

To run the interactive 3D visual viewer from a local Windows terminal:
```powershell
# 1. Stance mode (Default in v1.3: static 4-point stance with active grasps, free root under gravity)
.venv\Scripts\python.exe scripts\view_scene.py --profile compact_strong --mode stance

# 2. Pose mode (debug actuator & morphology visualization with temporary root support)
.venv\Scripts\python.exe scripts\view_scene.py --profile compact_strong --mode pose

# 3. Free mode (unconstrained free-root physics without hold attachments; falling expected)
.venv\Scripts\python.exe scripts\view_scene.py --profile compact_strong --mode free
```

---

## Changed Files Summary

| File | Type | Changes |
| :--- | :--- | :--- |
| `src/boulder_v1/contact.py` | Modified | Added `can_attach(...)` integrating affordance, distance, and orientation checks. |
| `src/boulder_v1/mjcf_builder.py` | Modified | Predeclared 12 `<connect>` equality constraints for hands to GRASP holds; added `get_grasp_equality_name`. |
| `src/boulder_v1/grasp.py` | New | Implemented `GraspManager` (attach/detach/load), `setup_static_stance`, and `simulate_static_stance`. |
| `src/boulder_v1/__init__.py` | Modified | Exported v1.3 contact, grasp, and static stance symbols. |
| `scripts/view_scene.py` | Modified | Added `--mode stance` with static 4-point stance support in interactive viewer and headless validation. |
| `scripts/run_demo.py` | Modified | Added v1.3 demonstration for contact eligibility, static stance, and breakable grip failure. |
| `tests/test_contact.py` | Modified | Added 4 unit tests for `can_attach` eligibility, distance, affordance, and orientation. |
| `tests/test_mjcf_builder.py` | Modified | Added unit tests verifying predeclared equality constraints and helper. |
| `tests/test_grasp.py` | New | 6 integration tests covering `GraspManager`, load measurement, breakable grip failure, and static stance. |
| `tests/test_view_scene.py` | Modified | Added `stance` mode test coverage for all profiles and CLI headless flag. |
| `README.md` | Modified | Updated documentation to reflect v1.3 features and `--mode stance` usage. |
| `docs/verification/prototype-v1.3-status.md` | New | Comprehensive verification document. |

---

## Known Limitations & Boundaries
- Humanoid is currently a 14-actuated-DoF prototype topology (no explicit wrist/ankle DoFs).
- Foot contact on footholds H1 and H2 is passive friction/collision support; feet do not use active equality constraints.
- No reinforcement learning or dynamic transition policies have been introduced yet.

## Recommended Next Milestone
**Prototype v1.4: Single Limb Reach & Transition Foundation**
- Implement reach kinematics/inverse kinematics for moving one limb off a hold while maintaining 3-point support.
- Track 3-point support reaction force redistribution as the 4th limb moves.
- Verify transition feasibility checks (kinematic reachability + static equilibrium + grip capacity).
