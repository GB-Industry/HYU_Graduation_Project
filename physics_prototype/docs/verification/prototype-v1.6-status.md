# Prototype v1.6: Generalized Multi-Limb Transition & Sequential Execution — Verification Status

## Executive Summary
Prototype v1.6 achieves a major architectural milestone for the Adaptive Physics-Based Bouldering Character Control project. The deterministic locomotion foundation has been upgraded from a single isolated right-hand reach into a generalized, reusable multi-limb transition engine and sequential execution pipeline.

This milestone operates strictly within the **Pre-RL deterministic locomotion foundation**:
- **Zero Reinforcement Learning**: All kinematics, retargeting, and PD joint control remain analytical and deterministic.
- **Zero State Resets**: Sequential transitions execute continuously from the exact physical state (`qpos`, `qvel`, active grasp constraints, load distributions) produced by the preceding move. `mj_resetData`, pose overwriting, and coordinate teleportation are strictly prohibited and verified absent.
- **True Multi-Limb Generality**: Both hands (`RIGHT_HAND`, `LEFT_HAND`) and feet (`LEFT_FOOT`, `RIGHT_FOOT`) operate through the exact same transition engine.
- **Explicit Failure Semantics**: Failure modes do not collapse into a single boolean; structured `TransitionStatus` codes provide granular inspectability.
- **Deterministic 3-Move Sequence**: Successfully demonstrated a continuous 3-move ascending sequence on the climbing wall:
  1. Move 1: `RIGHT_HAND: H4 -> H5` ($\Delta p = 0.166\,\text{m}$, inward reach)
  2. Move 2: `LEFT_FOOT: H1 -> H6` ($\Delta p = 0.164\,\text{m}$, stepping up)
  3. Move 3: `LEFT_HAND: H3 -> H7` ($\Delta p = 0.256\,\text{m}$, inward reach)

All 65 automated tests pass (64 passed, 1 expected skip, 0 failures). Full headless offscreen rendering pipeline verified on Linux with EGL hardware acceleration.

---

## 1. Unified Multi-Limb Transition Architecture (`locomotion.py`)

### 1.1 Transition Request & Structured Outcome

The single-limb transition primitive is fully generalized via typed request and result dataclasses:

```python
class TransitionRequest:
    limb: Limb
    source_hold: str
    target_hold: str
    steps: int = 210
    kp: float = 30.0
    kd: float = 3.0
    max_attach_distance: float = 0.15
    check_grip: bool = False
    settle_steps_after: int = 20
```

### 1.2 Explicit Failure Semantics (`TransitionStatus`)

Transitions return explicit status codes distinguishing physics failures, geometric errors, and state divergence:

| Status Code | Description | Verification Condition |
|:---|:---|:---|
| `SUCCESS` | Move executed, attached, and settled within stabilization thresholds. | `check_stabilization_readiness()` passes; limb attached. |
| `INVALID_REQUEST` | Ill-formed request (e.g. source == target, non-positive step budget). | Immediate rejection before simulation begins. |
| `SOURCE_NOT_ATTACHED` | Requested moving limb is not currently attached to the claimed source hold. | Validated against active contact configuration. |
| `INELIGIBLE_TARGET` | Target hold is unknown or lacks required affordance (`GRASP` for hands, `STEP` for feet). | Scene affordance matrix validation. |
| `SUPPORT_FAILURE` | Climber fell or root collapsed below minimum safety height ($z < 0.70\,\text{m}$). | Continuously monitored during free-root simulation. |
| `GRIP_FAILURE` | Physical load on supporting limbs exceeded breakable grip capacity. | Evaluated when `check_grip=True`. |
| `REACH_FAILURE` | End-effector could not reach vicinity of target hold. | Target distance limit exceeded. |
| `ATTACH_FAILURE` | End-effector failed to satisfy distance / alignment threshold for latching. | Distance to hold $> 0.15\,\text{m}$ at end of attach phase. |
| `UNSTABLE_FINAL_STATE` | Character attached but is swinging wildly or unstable for subsequent moves. | Evaluated by `check_stabilization_readiness()`. |

### 1.3 State Snapshot Summary (`StateSummary`)

Each transition captures complete initial and final physical states:
- Simulation timestamp ($t$)
- Root 3D position and linear velocity
- Root angular velocity and joint velocity norm
- Numerical finiteness flag
- Active contact configuration dictionary
- Real physical attachment loads across all supporting limbs

---

## 2. Foot Reposition Foundation & Hold Geometry Extension

### 2.1 Scene Extension (`scene_factory.py`)

Two deterministic holds were added to provide realistic, reachable footholds and handholds:
- **`H6` (Foothold)**: Pos `(-0.20, -0.02, 0.82)` m, radius `0.070` m. Affordances: `[STEP, GRASP]`. Displaced $0.164\,\text{m}$ from starting foothold `H1` ($(-0.33, -0.02, 0.72)\,\text{m}$), providing a natural $0.10\,\text{m}$ step-up.
- **`H7` (Handhold)**: Pos `(-0.22, -0.02, 1.68)` m, radius `0.065` m. Affordances: `[GRASP, STEP]`. Displaced $0.256\,\text{m}$ from starting handhold `H3` ($(-0.38, -0.02, 1.48)\,\text{m}$), providing a natural bilateral reach counterpart to H5.

### 2.2 Predeclared Foot Connect Equality Constraints (`mjcf_builder.py`)

- **Step Sites**: Added `<site name="site_step_{region.id}" pos="0 -0.04 0.098" .../>` on all holds with `Affordance.STEP`.
- **Connect Constraints**: Predeclared `<connect name="connect_{limb}_{region.id}" site1="{foot_site}" site2="site_step_{region.id}" active="false"/>` for both `LEFT_FOOT` and `RIGHT_FOOT`.
- **Zero-Recompilation Contact**: Foot step constraints toggle via `data.eq_active` dynamically without rebuilding or compiling the MuJoCo model.

### 2.3 Biomechanical Foot Orientation & Retargeting (`contact.py`, `grasp.py`)

- **Foot Normal Orientation**: The humanoid climbing shoe toe points along local $+Y$. In `grasp.py`, foot normal is calculated as $+R_{:, 1}$, pointing inward toward the wall surface (opposing the wall normal $[0, -1, 0]$), achieving orientation alignment $\approx 0.95$.
- **Step Site Offset Alignment**: `can_attach` calculates foot proximity against the elevated top step boundary $(x, y - 0.04, z + 0.098)$, yielding sub-millimeter positioning accuracy ($< 0.85\,\text{mm}$).
- **Upright Pitch Bias**: When stepping up onto H6, the torso orientation pitch bias is set to $-0.10$ (upright), preventing hip joint limits from impeding knee flexion.

---

## 3. Sequential Execution Without Simulation Reset (`execute_transition_sequence`)

### 3.1 Non-Reset Invariance Guarantee

The sequence executor guarantees that:
$$\text{State}_{\text{init}}^{(i+1)} \equiv \text{State}_{\text{final}}^{(i)}$$
- $\text{time}^{(i+1)}_{\text{start}} = \text{time}^{(i)}_{\text{final}}$: Strictly monotonic simulation clock.
- $q_{\text{pos}}^{(i+1)} = q_{\text{pos}}^{(i)}$, $q_{\text{vel}}^{(i+1)} = q_{\text{vel}}^{(i)}$: Zero coordinate or velocity resetting.
- Grasp attachments and equality constraint activations are preserved across moves.
- Root position maintains continuous climbing height ($z \approx 1.21\,\text{m} \to 1.23\,\text{m} \to 1.23\,\text{m}$).

### 3.2 Pragmatic Stabilization Readiness Criterion (`check_stabilization_readiness`)

Rather than relying on fixed arbitrary delays, readiness for the next move is determined physically:
1. **Numerical Finiteness**: No NaNs or infinities in `qpos` or `qvel`.
2. **Support Preservation**: Minimum active supports $\ge 3$.
3. **Root Elevation**: Root $z \ge 0.85\,\text{m}$ (no ground collapse or hanging detachment).
4. **Linear Velocity Threshold**: $\|v_{\text{lin}}\| < 0.60\,\text{m/s}$ (distinguishes wild swinging/falling from steady contact).
5. **Angular Velocity Threshold**: $\|\omega_{\text{ang}}\| < 3.50\,\text{rad/s}$ (tolerates closed-chain micro-vibrations while detecting uncontrolled body rotation).
6. **Joint Velocity Norm**: Mean joint velocity $< 4.00\,\text{rad/s}$.

### 3.3 Sequence Failure Propagation

If any move in a sequence fails (e.g. invalid target hold, grip rupture, support failure):
- The sequence immediately halts.
- Remaining moves in the queue are never initiated.
- The failed move index, final contact configuration, and inspectable state summary are preserved and returned in `TransitionSequenceResult`.

---

## 4. Verification Results & Test Suite

### 4.1 Automated Test Suite
- **Total Tests**: 65 tests.
- **Status**: **64 Passed**, **1 Skipped** (conditional MuJoCo missing test), **0 Failures**.
- **Execution Time**: ~12.5 seconds.
- **Coverage**:
  - Request validation (`test_transition_request_validation`)
  - Bilateral hand reach (`test_bilateral_hand_transitions`)
  - Foot reposition and step site latching (`test_foot_reposition_foundation`)
  - Stabilization readiness criterion (`test_stabilization_readiness_criterion`)
  - Sequential execution without reset (`test_multi_move_sequence_without_reset`)
  - Sequence failure propagation (`test_sequence_failure_propagation`)
  - Multi-profile sequence execution (`test_sequence_profile_variations`)

### 4.2 Cross-Profile Performance Summary

| Climber Profile | Arm Reach | Leg Reach | Move 1 (RH H4$\to$H5) | Move 2 (LF H1$\to$H6) | Move 3 (LH H3$\to$H7) | Sequence Status | Final Root $z$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `base` | $0.58\,\text{m}$ | $0.82\,\text{m}$ | SUCCESS | SUCCESS | SUCCESS | **SUCCESS (3/3)** | $+1.226\,\text{m}$ |
| `compact_strong` | $0.52\,\text{m}$ | $0.76\,\text{m}$ | SUCCESS | SUCCESS | SUCCESS | **SUCCESS (3/3)** | $+1.181\,\text{m}$ |
| `long_reach_lower_grip` | $0.64\,\text{m}$ | $0.86\,\text{m}$ | SUCCESS | SUCCESS | SUCCESS | **SUCCESS (3/3)** | $+1.258\,\text{m}$ |

---

## 5. Visual Artifacts (`outputs/visual/`)

Generated via headless EGL rendering pipeline:
- **`sequence_base.mp4`**: 375 frames @ 30fps (12.5s) continuous multi-move climbing sequence.
- **`sequence_base.gif`**: 375 frames animated GIF with HUD overlay.
- **`sequence_montage.png`**: $1280 \times 720$ 2x2 comparison montage displaying:
  1. Initial 4-Point Stance (H1, H2, H3, H4)
  2. Move 1: RH H4 $\to$ H5 (Inward reach & attach)
  3. Move 2: LF H1 $\to$ H6 (Step up & support)
  4. Move 3: LH H3 $\to$ H7 (Inward reach & stabilize)
- **Keyframe Images**: `sequence_00_initial.png`, `sequence_01_move1_rh.png`, `sequence_02_move2_lf.png`, `sequence_03_move3_lh.png`.
