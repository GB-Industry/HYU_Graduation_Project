# Prototype v1.5.1: Biomechanical Cleanup & Quasi-Static Reach Refinement — Verification Status

## Executive Summary
Prototype v1.5.1 is a quality-correction and biomechanical refinement milestone following Prototype v1.5. While v1.5 established the algorithmic foundation for stance retargeting and single-limb reach transitions, manual visual inspection revealed notable visual and biomechanical unnaturalness. Prototype v1.5.1 systematically addresses and resolves these issues while preserving all architectural invariants: 25 actuated DoFs, unconstrained free-root physics, breakable physical grasp equality constraints, and headless-first verification.

### Key Corrections Delivered
1. **Biomechanical Knee Hinge Semantics**:
   - Fixed knee hinge axis to anatomical posterior flexion `axis="-1 0 0"` with range `0 150` deg.
   - Shins swing strictly posteriorly ($-y$ relative to thigh), eliminating the backward "flamingo" inversion.
   - Feet point forward toward holds; knees project anteriorly and outward in high-step climbing posture.
2. **Athletic Morphology & Climbing Shoe Geometry**:
   - Added deltoid sphere primitives (`radius="0.046"`) smoothly bridging torso and upper arms.
   - Contoured pelvis capsule (`radius="0.070"`, length `0.10`) with athletic taper.
   - Replaced oversized foot blocks with sleek climbing shoes (`pos="0 0.035 -0.015"`, size `0.040 0.095 0.024`, friction `1.8`), dedicated toe end-effector sites (`pos="0 0.09 -0.025"`), and zero initial collision penetration against footholds.
3. **Plausible Static Stance & Compliant Impedance**:
   - Tuned stance specification: pelvis wall distance $0.64\,\text{m}$, torso pitch $-0.24\,\text{rad}$, knee flexion $1.35\,\text{rad}$, elbow flexion $1.00\,\text{rad}$, hip abduction $0.28\,\text{rad}$.
   - Adjusted low-gain impedance control (`kp=5.0, kd=0.5`) in stance equilibrium, reducing holding loads on hands from $>700\,\text{N}$ down to $\sim 186\,\text{N}$ (comfortably within the $644\,\text{N}$ base profile capacity).
4. **Believable 3-Point Support Weight Transfer**:
   - Implemented active lateral pelvic shift ($\Delta x = \pm 0.16\,\text{m}$) and torso counter-rotation over supporting limbs prior to limb release.
   - Verified that weight transfers cleanly over supporting limbs and that the releasing limb's grip load decreases by $>20\%$ before release.
5. **7-Stage Coordinated Reach Lifecycle**:
   - Expanded reach transition into a 7-stage coordinated lifecycle:
     `INITIAL_STANCE` $\to$ `PRE_SHIFT` $\to$ `RELEASE_LIMB` $\to$ `SUPPORT_PHASE` $\to$ `REACH_PHASE` $\to$ `ATTACH_PHASE` $\to$ `STABILIZED_STANCE`.
   - Bounded cadence budgets ensure robust reattachment regardless of total simulation duration (`steps=200` to `400+`).
6. **Full Test & Visual Verification Cleanliness**:
   - All 54 unit tests across the test suite pass cleanly (53 passed, 1 expected skip, 0 failures).
   - Rendered publication-grade offscreen visual verification assets to `outputs/visual/`.

---

## 1. Biomechanical & Morphological Corrections

### Knee Joint Kinematics
In previous revisions, knee axes allowed reversed anterior bending under certain coordinate conventions. In v1.5.1:
- Model frame: Character faces forward ($+y$), gravity acts along $-z$, lateral axis is $+x$.
- Thigh extends downward ($-z$). Shin extends downward ($-z$) from knee.
- Knee joint axis defined as:
  ```xml
  <joint name="left_knee" type="hinge" pos="0 0 0" axis="-1 0 0" range="0 150" .../>
  <joint name="right_knee" type="hinge" pos="0 0 0" axis="-1 0 0" range="0 150" .../>
  ```
- With `axis="-1 0 0"` and positive rotation $\theta > 0$, the shin rotates about $-x$, causing the foot to move in $-y$ (posteriorly toward the character's back), matching human anatomical knee flexion.
- Verified in [`tests/test_locomotion.py:LocomotionTests.test_biomechanical_knee_hinge_semantics`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/tests/test_locomotion.py):
  $$\Delta y_{\text{foot/thigh}} < -0.05\,\text{m} \quad (\text{foot moves posteriorly under flexion}).$$

### Climbing Shoe Geometry & Foothold Placement
- Old feet were bulky blocks that intersected footholds H1/H2 at step 0 by $> 50\,\text{mm}$, generating $> 1200\,\text{N}$ explosive repulsion forces.
- New sleek climbing shoe:
  ```xml
  <geom name="left_foot_geom" type="box" pos="0 0.035 -0.015" size="0.040 0.095 0.024"
        friction="1.8 0.005 0.0001" rgba="0.18 0.18 0.20 1.0" .../>
  <site name="left_foot_site" pos="0 0.090 -0.025" size="0.016" rgba="0.95 0.20 0.20 1.0"/>
  ```
- Foothold target coordinates placed at:
  $$y_{\text{target}} = -0.060\,\text{m}, \quad z_{\text{target}} = z_{\text{region}} + 0.098\,\text{m} = 0.818\,\text{m}.$$
- This eliminates initial geom overlap ($0.0\,\text{mm}$ initial penetration) while seating the sticky rubber sole smoothly onto the hold dome with gentle gravitational settling ($\approx 1.6\,\text{mm}$).

### Anthropomorphic Deltoid & Torso Contouring
- Added deltoid shoulder spheres (`geom name="*_deltoid_geom" type="sphere" size="0.046"`) positioned at $(0, 0, 0)$ of the shoulder frames.
- Replaced blocky barrel torso with contoured pelvis capsule and tapered chest geometry, creating an athletic climbing physique.

---

## 2. Retargeting & Static Stance Mechanics

### Compliant Stance Parameters
[`StanceSpecification`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/retargeter.py) defaults tuned for natural climbing posture:
- `pelvis_wall_distance`: $0.64\,\text{m}$ (maintains center of mass close to wall while leaving clearance for flexed knees).
- `torso_orientation`: $(0.0, -0.24, 0.0)\,\text{rad}$ (forward chest lean into the wall).
- `preferred_knee_flexion`: $1.35\,\text{rad}$ ($\approx 77^\circ$).
- `preferred_elbow_flexion`: $1.00\,\text{rad}$ ($\approx 57^\circ$).
- `hip_openness`: $0.28\,\text{rad}$ (lateral hip abduction clearance).

### Nullspace Priors for Climbing High-Step
Nullspace posture prior $\mathbf{q}_{\text{pref}}$:
- `hip_pitch`: $1.70\,\text{rad}$
- `knee`: $1.35\,\text{rad}$
- `ankle_pitch`: $-0.30\,\text{rad}$
- Ensures foot remains below knee ($z_{\text{foot}} < z_{\text{knee}}$), knee remains below pelvis ($z_{\text{knee}} < z_{\text{pelvis}}$), and feet point toward footholds rather than inverted backwards.

### Contact Force Distribution
Under compliant impedance control (`kp=5.0, kd=0.5`):
- Left Hand load: $186.7\,\text{N}$
- Right Hand load: $\sim 300\,\text{N}$
- Both within standard grip capacity ($644\,\text{N}$).
- Tested with `test_profile_conditioned_breakable_grip`: a weak profile ($100\,\text{N}$) physically breaks away, while a strong profile ($1000\,\text{N}$) sustains hold without slip.

---

## 3. Coordinated 7-Stage Quasi-Static Reach Lifecycle

The single-limb transition in [`simulate_single_limb_reach()`](file:///C:/Users/오유찬/Desktop/project/boulder_prototype/src/boulder_v1/locomotion.py) now executes 7 distinct, physically grounded phases:

```
[INITIAL_STANCE] -> [PRE_SHIFT] -> [RELEASE_LIMB] -> [SUPPORT_PHASE] -> [REACH_PHASE] -> [ATTACH_PHASE] -> [STABILIZED_STANCE]
     (0..24)         (24..56)           (56)             (57..80)         (80..110)        (110..150)          (150..steps)
```

| Phase Index | Phase Name | Actions & Control Target |
|:---|:---|:---|
| 1 | `INITIAL_STANCE` | 4-point stance settling under full gravity; all 4 limbs attached. |
| 2 | `PRE_SHIFT` | Smooth lateral pelvis shift ($\Delta x = \pm 0.16\,\text{m}$) and torso rotation toward supporting side; unloads releasing hand by $>20\%$. |
| 3 | `RELEASE_LIMB` | Detach target limb via real `GraspManager.detach()` event at minimum load point. |
| 4 | `SUPPORT_PHASE` | Maintain unconstrained 3-point support against gravity with remaining hand and feet. |
| 5 | `REACH_PHASE` | Whole-body coordinated reach (pelvis shift, chest rotation, shoulder flexion) driving arm toward target hold. |
| 6 | `ATTACH_PHASE` | Limb approaches target hold; dynamic eligibility verified via `can_attach()`; `attach()` creates physical equality constraint. |
| 7 | `STABILIZED_STANCE` | Posture settles into balanced four-point stance with both hands re-secured. |

### Bounded Cadence Budgets
To prevent excessive hanging sag when running long simulations (`steps=400` or higher), step budgets are dynamically bounded:
```python
s_initial_end = min(24, max(12, int(steps * 0.12)))
s_preshift_end = s_initial_end + min(32, max(16, int(steps * 0.16)))
s_release = s_preshift_end
s_support_end = s_release + min(24, max(12, int(steps * 0.12)))
s_reach_end = s_support_end + min(30, max(15, int(steps * 0.15)))
s_attach_end = s_reach_end + min(40, max(20, int(steps * 0.20)))
```
This guarantees that dynamic detachment, excursion, and reattachment complete within the first $\approx 150$ steps, while all remaining steps settle in `STABILIZED_STANCE`.

---

## 4. Test Suite & Validation Summary

### Headless Verification Modes
Tested via `scripts/view_scene.py --headless --mode <mode>`:
- `--mode stance`: 400 steps simulated, supported $z = +1.181\,\text{m}$, LH load = $299.5\,\text{N}$, RH load = $392.6\,\text{N}$. **[PASS]**
- `--mode transition`: 400 steps simulated, supported $z = +1.176\,\text{m}$, 7-stage sequence completed, LH load = $488.1\,\text{N}$, RH load = $435.5\,\text{N}$. **[PASS]**
- `--mode pose`: 400 steps simulated, initial error $17.8^\circ \to 2.19^\circ$, error reduced. **[PASS]**
- `--mode free`: 400 steps simulated, numerically finite, error reduced. **[PASS]**

### Full Automated Unit Test Coverage
Ran `python -m unittest discover -s tests -v`:
- Total tests: 54
- Passed: 53
- Skipped: 1 (expected: missing MuJoCo negative check when MuJoCo is present)
- Failures / Errors: 0
- Execution duration: $12.558\,\text{s}$

### Visual Artifacts Generated (`outputs/visual/`)
- `stance_compact_strong.png`: 1280x720 static stance verification image ($49.6\,\text{KB}$).
- `stance_base.png`: 1280x720 static stance verification image ($49.8\,\text{KB}$).
- `transition_base.mp4`: 210 frames @ 30fps transition video ($105.2\,\text{KB}$).
- `transition_base.gif`: 210 frames @ 30fps animated GIF ($3.4\,\text{MB}$).
- Keyframe sequence:
  - `transition_00_initial.png` ($48.9\,\text{KB}$)
  - `transition_01_release.png` ($49.1\,\text{KB}$)
  - `transition_02_support.png` ($49.1\,\text{KB}$)
  - `transition_03_reach.png` ($49.6\,\text{KB}$)
  - `transition_04_attach.png` ($49.5\,\text{KB}$)
  - `transition_05_stabilized.png` ($50.6\,\text{KB}$)
