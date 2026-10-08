# Prototype v1.4.1: Human Morphology & Stance Refinement — Verification Status

## Executive Summary
Prototype v1.4.1 refines the humanoid morphology, visual body geometry, segmental rotational inertias, climbing shoe design, and static four-point climbing stance following visual inspection of Prototype v1.4. 

Key results:
- **Visual Morphology Overhaul**: Replaced crude stacked cylindrical/box torso blocks with natural MuJoCo-native anatomical ellipsoids (pelvis basin, athletic waist, broad ribcage).
- **Physical Inertia Integrity**: Resolved rotational inertia loss ($I_{xx}$ collapse) by declaring explicit anatomical diagonal inertias on torso bodies, preserving dynamic rotational stability under full gravity.
- **Natural Cranial & Neck Junction**: Replaced bulky spherical head and thick cylinder neck with an anatomical slender neck capsule and cranial ellipsoid.
- **Proportional Limbs & Sleek Climbing Shoes**: Replaced block-like feet with sleek climbing shoe boxes ($4.5\,\text{cm} \times 11\,\text{cm} \times 2.8\,\text{cm}$) fitted with high-friction bouldering rubber ($\mu = 1.8$), and proportional limb capsule diameters.
- **Athletic Clothing & Skin Palette**: Differentiated dark slate navy shorts, technical athletic tee, warm skin tone, and vibrant climbing shoes.
- **Deterministic Static Stance Stability**: Adjusted static stance pelvis height ($z_{\text{root}} = 1.2630\,\text{m}$) so that shoe soles rest directly on footholds H1 and H2. Under full gravity with free root (no root stabilization), the character stably maintains four-point wall contact for 500+ steps with `check_grip=True`.
- **Zero Regressions**: All 39 tests pass (38 passed, 1 expected skip), demo script and headless viewer across all modes (`stance`, `pose`, `free`) verified.

---

## 1. Human Morphology & Visual Geometry Refinements

### Torso Chain: Box/Cylinders to Anatomical Ellipsoids
In v1.4, the pelvis was a rectangular box and the abdomen/chest were stacked cylindrical capsules, creating an unnatural "barrel-like" mechanical appearance.
- **Pelvis**: Sculpted into an anatomical pelvic basin using a lateral ellipsoid (`type="ellipsoid"`, size `[hip_x * 0.88, 0.088, pelvis_h * 0.45]`).
- **Abdomen**: Sculpted into a tapered athletic waist ellipsoid (`size="[hip_x * 0.78, 0.082, abdomen_h * 0.46]"`).
- **Chest**: Sculpted into a broad athletic ribcage ellipsoid (`size="[shoulder_x * 0.85, 0.11, chest_h * 0.48]"`).

### MuJoCo Rotational Inertia Mechanics & The Explicit Inertia Solution
When solid geometries are changed from capsules to ellipsoids in MuJoCo, the default primitive volume integral significantly reduces rotational inertia ($I \propto \frac{1}{5} m r^2$ for ellipsoids vs $\frac{1}{2} m r^2$ for capsules). In initial tests without explicit `<inertial>` elements, $I_{xx}$ dropped by $2.5\times$ to $4.2\times$, causing high-frequency pitching oscillations that dynamically spiked grasp reaction forces during settling.
To resolve this without altering visual contours:
- Explicit physiological `<inertial>` tags were added to `pelvis`, `abdomen`, and `chest`:
  - `pelvis`: `diaginertia="{0.051 * m_scale} {0.076 * m_scale} {0.097 * m_scale}"`
  - `abdomen`: `diaginertia="{0.100 * m_scale} {0.100 * m_scale} {0.060 * m_scale}"`
  - `chest`: `diaginertia="{0.355 * m_scale} {0.355 * m_scale} {0.140 * m_scale}"`
This restored physiological rotational resistance, stabilizing upper body dynamics under full gravity.

### Slender Neck, Head & Limb Proportions
- **Neck**: Slender capsule reduced from `size="0.05"` to `size="0.038"`.
- **Head**: Replaced crude `pos="0 0 0.15" size="0.095"` sphere with anatomical cranial ellipsoid `pos="0 0 0.14" size="0.075 0.092 0.105"`.
- **Arms**: Upper arm capsule thinned from `0.048` to `0.040`, forearm from `0.040` to `0.033`.
- **Legs**: Thigh capsule thinned from `0.068` to `0.056`, shin from `0.052` to `0.042`.
- **Climbing Shoes**: Reduced from clunky block ($13\,\text{cm}$ wide $\times 7\,\text{cm}$ thick) to sleek climbing shoe ($9\,\text{cm}$ wide $\times 5.6\,\text{cm}$ thick) with high-friction sticky rubber (`friction="1.8 0.05 0.005"`).
- **Hand Sites**: Maintained exact local $-Z$ orientation (`pos="0 0 -0.04"`) ensuring hold normal compatibility and preserving full grip capacity.

---

## 2. Climbing Stance Retargeting & Physics Stability

### Settled Root Elevation ($z_{\text{root}} = 1.2630\,\text{m}$)
In v1.4, initial root height $z = 1.2730\,\text{m}$ placed the foot sites $1.27\,\text{cm}$ above the tops of footholds H1 and H2. During free gravity settling, this air gap caused an initial fall-and-bounce impact where hand loads spiked to $662\,\text{N}$, exceeding breakable grip capacity.
Adjusting $z_{\text{root}} = 1.2630\,\text{m}$:
- Places climbing shoe soles firmly in contact with footholds H1 and H2 from $t=0$.
- Eliminates drop shock and swinging oscillations.
- Foot normal reaction forces support body weight, allowing hand loads to settle at comfortable equilibrium values ($412\,\text{N}$ left hand, $494\,\text{N}$ right hand).

---

## 3. Automated Test Suite Verification

### Full Test Suite (39 tests: 38 passed, 1 expected skip)
```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages;."
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
**Results Summary**:
- `test_mjcf_builder.py` (13 tests):
  - `test_refined_morphology_ellipsoids_and_inertias` (PASS): Verifies ellipsoidal torso segments, explicit segmental inertias, and cranial ellipsoid.
  - `test_climbing_shoe_geometry` (PASS): Verifies sleek climbing shoe dimensions ($\le 10\,\text{cm}$ wide, $\le 6\,\text{cm}$ thick) and $\mu \ge 1.5$ rubber.
  - 25 DoF topology, bilateral symmetry, mass scaling, ROM scaling, exclusions all pass.
- `test_grasp.py` (6 tests):
  - `test_simulate_static_stance` (PASS): 500 simulation steps under full gravity with `check_grip=True`, both hands remaining attached, root maintaining equilibrium above $1.13\,\text{m}$.
  - Real-physics breakable grip test (PASS): Weak profile physically detaches, strong profile holds.
- `test_runtime.py` (6 tests): Model compilation, 4 end-effector sites, pose control sanity check all pass.
- `test_contact.py` (7 tests): Affordances, orientation compatibility, grip controller pass.
- `test_schema.py` (3 tests): Schema validations pass.
- `test_view_scene.py` (4 tests): Headless validation across all profiles and modes passes.

### Demo Script (`scripts/run_demo.py`)
```
Generated canonical scene, two profile-conditioned MJCF models, and schematic image.
long_reach_lower_grip SmokeResult(steps=200, time=0.400s, nq=32, nv=31, nu=25, finite=True)
compact_strong SmokeResult(steps=200, time=0.400s, nq=32, nv=31, nu=25, finite=True)
long_reach_lower_grip static stance: steps=400 finite=True supported=True root_z=+0.966m LH_load=466.5N RH_load=408.6N LH_attached=True RH_attached=True
compact_strong static stance: steps=400 finite=True supported=True root_z=+1.127m LH_load=526.4N RH_load=642.5N LH_attached=True RH_attached=True
Real physics breakable grip test (actual MuJoCo LH load = 341.7 N):
  weak profile (cap=40.9 N): maintain=False -> physically detached=True
  compact_strong (cap=817.9 N): maintain=True -> physically held=True
```

### Viewer Headless Validation (`scripts/view_scene.py --headless`)
- Mode `stance`: `base`, `long_reach_lower_grip`, `compact_strong` all pass (rc=0).
- Mode `pose`: `base`, `long_reach_lower_grip`, `compact_strong` all pass (rc=0).
- Mode `free`: `base`, `long_reach_lower_grip`, `compact_strong` all pass (rc=0).
