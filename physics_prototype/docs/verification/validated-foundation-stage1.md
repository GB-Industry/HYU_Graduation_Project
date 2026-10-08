# Validated Static Climbing Foundation: Stage 1

## Scope And Evidence

Stage 1 validates a deliberately simplified rigid-body humanoid and its model
boundary, not a clinical human model or climbing-performance baseline. It starts
from Stage 0 commit `e577b266d21d09d43b0ea84ebd3f8c5e9d03a30e`.

**Not validated: hand grasp, foot friction/support, capture mechanics, climbing
feasibility, the locomotion controller, planning, or RL.** No Stage 2 work was
started. The existing contact laws, 15 cm capture radius, grip defaults, PD law,
gains, readiness thresholds, transition cadence, hold geometry, and choreography
remain unchanged.

The primary machine-readable evidence is generated, not handwritten:

- `python scripts/validate_model.py` produces
  `outputs/physical-model-stage1/validation.json` without graphics.
- Adding `--render` produces `neutral-com.png`, with a marker projected from the
  actual compiled subtree COM. Rendering owns scratch data and does not advance
  or reconstruct a climbing trajectory.
- `src/boulder_v1/model_validation.py` records all bodies, parents, weld groups,
  masses, local/world COMs, principal inertias and their orientations, body-frame
  tensors, joints, ranges, addresses, actuator mappings, sites, collision geoms,
  exclusion signatures, signed joint probes, physical variants, and numerical
  experiments.

Recorded environment: Python 3.12.13, MuJoCo 3.14.0, NumPy 2.5.3, Linux. Graphics
use EGL only on requested rendering paths. This is not an all-version guarantee
across the package's broad MuJoCo dependency range.

## Actual Topology

Compiled dimensions are `nq=32`, `nv=31`, `nu=25`, with one freejoint and 25
scalar hinges. There are 27 bodies including world/fixed environment, 74 geoms,
21 sites, 30 inactive-by-default climbing equalities, and 65 explicit exclusions.

The climber subtree contains 17 body frames:

```text
climber_root: unactuated free root, massless coordinate frame
  pelvis: rigidly welded to root, 10 kg
    abdomen: waist yaw / pitch / roll
      chest: rigidly welded to abdomen
        head: rigid head/neck compound
        left_upper_arm: shoulder pitch / roll / yaw
          left_forearm: elbow
            left_hand: wrist
        right_upper_arm: shoulder pitch / roll / yaw
          right_forearm: elbow
            right_hand: wrist
    left_thigh: hip pitch / roll / yaw
      left_shin: knee
        left_foot: ankle pitch / roll
    right_thigh: hip pitch / roll / yaw
      right_shin: knee
        right_foot: ankle pitch / roll
```

There is no separate scapula, neck joint, pronation joint, or finger mechanism.
The root's six velocity coordinates are not actuated. The 25 motors map one to
one, in the order below, to the 25 scalar joints. Joint positions are radians in
compiled data; the builder emits ranges in degrees.

## Joint Convention Table

Conventions refer to the identity-root neutral configuration: X is horizontal
(left negative, right positive), +Y points toward the wall, and +Z is up. Serial
joint order matters away from neutral. The axes are the actual compiled local
axes, not inferred from old anatomical comments.

| Joint ID / name | Axis | Base range, degrees | Base unit-control torque, Nm | Positive convention |
| --- | --- | --- | ---: | --- |
| 1 `waist_yaw` | +Z | -35, 35 | 90 | Local +X rotates toward +Y; left shoulder recedes toward -Y. |
| 2 `waist_pitch` | +X | -25, 45 | 110 | Upright chest/head move toward -Y: backward/away from wall. |
| 3 `waist_roll` | +Y | -25, 25 | 90 | Upright torso leans toward +X. |
| 4 `left_shoulder_pitch` | +X | -120, 160 | 85 | Downward arm moves forward toward +Y. |
| 5 `left_shoulder_roll` | +Y | -80, 80 | 85 | Downward arm moves toward -X: left-side abduction. |
| 6 `left_shoulder_yaw` | +Z | -80, 80 | 75 | Local +X rotates toward +Y; straight arm centerline does not translate. |
| 7 `left_elbow` | +X | 0, 145 | 65 | Forearm/hand flex forward toward +Y. |
| 8 `left_wrist` | +X | -60, 60 | 35 | Hand tip moves toward +Y. |
| 9 `right_shoulder_pitch` | +X | -120, 160 | 85 | Downward arm moves forward toward +Y. |
| 10 `right_shoulder_roll` | +Y | -80, 80 | 85 | Positive moves toward -X; negative abducts the right arm toward +X. |
| 11 `right_shoulder_yaw` | +Z | -80, 80 | 75 | Local +X rotates toward +Y. |
| 12 `right_elbow` | +X | 0, 145 | 65 | Forearm/hand flex forward toward +Y. |
| 13 `right_wrist` | +X | -60, 60 | 35 | Hand tip moves toward +Y. |
| 14 `left_hip_pitch` | +X | -110, 120 | 140 | Downward leg moves forward toward +Y. |
| 15 `left_hip_roll` | +Y | -50, 60 | 110 | Leg moves toward -X: left-side abduction. |
| 16 `left_hip_yaw` | +Z | -45, 45 | 80 | Forward toe direction rotates toward -X. |
| 17 `left_knee` | -X | 0, 150 | 130 | Shin bends posteriorly toward -Y. |
| 18 `left_ankle_pitch` | +X | -45, 35 | 55 | Toe rises toward +Z: dorsiflexion; negative is plantarflexion. |
| 19 `left_ankle_roll` | +Y | -25, 25 | 40 | Foot site's below-axis component moves toward -X. |
| 20 `right_hip_pitch` | +X | -110, 120 | 140 | Downward leg moves forward toward +Y. |
| 21 `right_hip_roll` | +Y | -60, 50 | 110 | Positive moves toward -X; negative abducts toward +X. |
| 22 `right_hip_yaw` | +Z | -45, 45 | 80 | Forward toe direction rotates toward -X. |
| 23 `right_knee` | -X | 0, 150 | 130 | Shin bends posteriorly toward -Y. |
| 24 `right_ankle_pitch` | +X | -45, 35 | 55 | Positive raises toe; negative plantarflexes. |
| 25 `right_ankle_roll` | +Y | -25, 25 | 40 | Foot site's below-axis component moves toward -X. |

Every joint is perturbed by +0.05 and -0.05 radians using `mj_forward` only. Body
and site rotation matrices, positional changes, and off-axis probes are recorded.
Negative elbow/knee probes diagnose sign outside their ROM; they are never
integrated as admissible physical initial states.

Positive waist pitch gives chest delta Y = -0.0096209901 m and head-top delta Y =
-0.0281257801 m. The below-axis hand can simultaneously move toward +Y; endpoint
motion alone must not be mislabeled as torso forward lean. Old waist-pitch and
ankle-pitch comments were corrected. No axis was reversed to match a comment.

Mirroring across X retains X-axis coordinate signs, reverses Y/Z-axis signs, and
swaps limbs. Kinematics, COM, inertias, ranges, generalized mass matrices,
accelerations, and passive rollouts are checked with this independent transform.

## ROM And Reference Policy

This is an explicit simplified technical ROM model, not a cited clinical envelope.
Base ranges are in the table. Endpoints scale about zero with `rom_scale`, except
shoulder roll is capped to +/-85 degrees.

The former left/right shoulder-roll ranges included Euler-chart singularities at
+/-90 degrees. They now use a symmetric base range of +/-80 degrees. The cap
prevents supported profiles from crossing the singular middle coordinate. At the
85-degree cap, rotational-Jacobian singular values are approximately
`1.41286750, 1.0, 0.06168818`; all three directions remain independent.

The backend supports `rom_scale` from `1e-6` to `1.2`. Compiled intervals must be
finite, ordered, include neutral zero, and preserve bilateral mirror semantics.
This support interval is a prototype policy, not an assertion about people.
MuJoCo limits are soft constraints; this stage does not certify zero dynamic
overshoot in every possible motion.

The synthetic climbing reference still contains approximately 101.30/99.95 degree
hip flexion. At ROM scale 0.8, the hip upper limit is 96 degrees. At scale 0.5,
additional joints violate limits. `initialize_episode` therefore rejects these
unsupported demo references before resetting or mutating the episode. It does
not clip joint values and pretend the old attachments remain geometrically valid.
Neutral `qpos0` remains legal for those models.

Initialization also normalizes the rounded root quaternion. Reference validation
rejects zero/nonunit quaternions. The debug pose helper rejects targets outside
compiled ROM or targeting non-actuated/non-hinge coordinates. Its PD law is not
changed. Tests include rejection on already-live data without state mutation.

## Mass, Inertia, And COM

Chosen policy: retain fixed nominal segment masses, scaled uniformly by
`mass_scale`. Geometry deforms the primary mass proxy and its inertia, not the
segment's mass fraction. This is **not constant-density anthropometry**; the
implied density changes with dimensions. The policy is independent of controller
performance and is not calibrated to make a climbing demo pass.

The old separate root kilogram is folded into the rigid pelvis budget. The root
is a massless coordinate frame welded to a 10 kg pelvis. Every moving weld group
has positive mass and the complete generalized mass matrix is positive definite.
There is no unexplained extra kilogram and no requirement for every coordinate
frame to carry local mass.

| Segment | Base mass, kg | Inertia/COM proxy |
| --- | ---: | --- |
| Root frame | 0 | No independent physical segment. |
| Pelvis | 10 | Primary uniform ellipsoid, COM at frame origin. |
| Abdomen | 12 | Primary uniform ellipsoid, COM at frame origin. |
| Chest | 18 | Primary uniform ellipsoid, COM at frame origin. |
| Head/neck | 5 | 1.5 kg capsule plus 3.5 kg ellipsoid, inferred compound COM/tensor. |
| Upper arm, each | 2.4 | Uniform capsule, COM at half segment length. |
| Forearm, each | 1.5 | Uniform 31 mm-radius primary capsule. |
| Hand, each | 0.55 | Box at local `(0,0,-0.03)`. |
| Thigh, each | 7.5 | Uniform capsule. |
| Shin, each | 3.5 | Full-length primary capsule. |
| Foot, each | 1.2 | Existing box at local `(0,0.035,-0.018)`. |

Torso group mass is 40 kg, head/neck 5 kg, each arm 4.45 kg, and each leg 12.2 kg:
total **78.3 kg**. Auxiliary torso caps/contours carry zero additional mass.

For an ellipsoid with semiaxes `(a,b,c)`:

```text
Ixx = m * (b*b + c*c) / 5
Iyy = m * (a*a + c*c) / 5
Izz = m * (a*a + b*b) / 5
```

Torso dimensions and explicit tensors use the same full-precision quantities.
Limb capsule and hand/foot box inertia comes from MuJoCo primitive inference with
explicit mass. Independent capsule/ellipsoid formulas verify the compiled result.
Compound principal moments and `body_iquat` are distinguished from body-axis
tensors; the head includes a nonzero yz tensor component.

| Body | Base body-axis inertia diagonal, kg*m^2 |
| --- | --- |
| Pelvis | 0.027081680, 0.045481680, 0.054500000 |
| Abdomen | 0.032741376, 0.050465376, 0.054895200 |
| Chest | 0.099450000, 0.174783744, 0.170553744 |

All mass-carrying principal inertias are finite, positive, and satisfy the triangle
inequalities. The neutral generalized mass-matrix minimum eigenvalue is about
0.010475973. Joint armature is still 0.01 kg*m^2 per hinge and zero on the root;
it is reported separately from segment inertia and is not secretly retuned.

Neutral climber COM is approximately `(0, -0.457867177522, 1.260261813538)` metres.
The independently weighted body-COM sum agrees with MuJoCo's subtree COM. COM is
finite, bilaterally centered, and inside the primary proxy bounds. Uniform mass
scaling preserves COM; shape changes update the appropriate tensors and offsets.

Fixed wall/hold body masses total approximately 701.094010857 kg from the existing
primitive inference. They are not articulated climber weight. The overall model
mass sum includes them; diagnostics deliberately separate the climber subtree.

## Profile Propagation

| Field | Physical interpretation in Stage 1 |
| --- | --- |
| `name` | Label only in MJCF; physically identical profiles with different names have identical numerical physics arrays. |
| `torso_length` | Torso geometry/offsets and all relevant ellipsoid inertia change. |
| `shoulder_width` | Shoulder origins/chest width and chest inertia change. |
| `hip_width` | Hip origins/pelvis-abdomen widths and their inertias change. |
| Arm/leg lengths | Segment offsets, primary/collision capsule lengths, COM positions, and inferred inertias change. |
| `mass_scale` | Uniform segment mass and inertia scaling; fixed environment and armature do not scale. |
| `rom_scale` | Actual compiled joint limits, with the documented shoulder chart cap. |
| `strength_scale` | Motor gear and normalized-control torque ceiling. |
| `grip_capacity` | Existing optional Python grip decision only; not validated here. |
| `power_scale` | Reserved/unsupported metadata, retained in serialization; no physical power limit is claimed or introduced. |

The unchanged controller still contains profile-name-dependent reference choices.
Model-parameter name invariance does not certify controller-name invariance; that
is explicit Stage 3 debt, not silently repaired in this stage.

## Actuator Measurements

All motors have `ctrlrange=(-1,1)`, unit gain, no activation dynamics, and no
separate enabled force limit. Engine-side clipping is measured with requested
controls `-2, -1, -0.5, 0.5, 1, 2` on every motor, rather than pre-clamping Python
inputs. Strength variants include 0.7, 1.0, and 1.3.

```text
actuator_force = clipped normalized motor control
hinge torque (Nm) = gear * actuator_force
```

`data.actuator_force` is not itself the joint torque. `qfrc_actuator` on the mapped
hinge is the measured generalized torque. Nominal all-motor conversion error is
zero in this environment.

Shoulder pitch at control 0.5 produces 29.75 / 42.5 / 55.25 Nm at strength scales
0.7 / 1.0 / 1.3. Its maximum magnitude is 59.5 / 85 / 110.5 Nm respectively.

Capability is the torque ceiling. Control is the feedback law. The unchanged PD
computes normalized output, so its unsaturated torque-space gains also scale with
gear. For shoulder gear 85, `kp=5`, `kd=0.5` mean 425 Nm/rad and 42.5 Nm*s/rad,
before passive damping. Stage 3 must separate feedback gains from capability;
Stage 1 does not redesign this controller or claim power/torque-speed/fatigue laws.

## Passive And Timestep Evidence

Each probe owns model/data. There are no active climbing equalities, motor
commands, external forces, root stabilization, or repeated pose reconstruction.
Contact-enabled freefall is placed at altitude outside all environment geometry.
Passive damping uses zero gravity, disabled contacts, legal ROM midpoints, and
enabled joint limits that remain inactive. The reference uses the full coupled
mass matrix, not an isolated scalar-body approximation.

| Measurement | 2 ms | 1 ms |
| --- | ---: | ---: |
| Freefall duration | 0.4 s | 0.4 s |
| Ballistic COM position bias | 3.924 mm | 1.962 mm |
| Ballistic velocity error | less than 2e-14 m/s | less than 2e-14 m/s |
| All-joint passive final/initial energy | 0.6470443143 | 0.6468605020 |
| Coupled exponential velocity error, relative to initial norm | 0.0016106896 | 0.0008058463 |

Freefall position bias matches semi-implicit integration: `g*t*h/2`; it halves
with timestep. There is zero contact/internal motion and no warning in these
runs. Passive kinetic energy decreases monotonically. There is no stiffness
restoring a displaced pose; undamped free-root modes can retain momentum and
energy. The damping test does not claim return to a neutral stance.

Over 0.2 s, selected 2 ms energy ratios are approximately 0.750445 (waist pitch),
0.162235 (left elbow), 0.334127 (right knee), and 0.184651 (left shoulder yaw).
The all-joint timestep energy-ratio difference is 0.0001838123 and velocity-norm
difference about 2.84e-8. Qualitative gravity/damping conclusions survive timestep
reduction. No bitwise trajectory equality is required and the production timestep
remains 2 ms.

Mirrored body/site position error is about 5.55e-17 m and acceleration error about
4.94e-15 in the unforced paired rollout. The two-second uncontrolled neutral/floor
rollout has 865 floor-contact steps, maximum absolute generalized speed about
11.1648, maximum absolute generalized acceleration about 4720.70, and no warnings.
This is a finite-impact smoke test, not standing balance or climbing stability.

## Collision Policy

Visual styling is unchanged. Two massless, invisible forearm collision capsules
use radius 28 mm instead of the visible/inertial proxy's 31 mm. This removes the
old 2.5 mm hip/forearm and 0.5 mm thigh/forearm neutral penetrations without adding
nonadjacent exclusions. Invisible shin capsules end 3 mm earlier at nominal
length, placing the unchanged shoe sole 1 mm below their envelope. Primary shin
mass/inertia still uses the full-length capsule.

No hand or foot geometry, shoe friction, equality attachment, or grasp target was
redesigned. The compiled model has 21 collision-enabled humanoid geoms and no
neutral contacts for the default tested morphology.

| Explicit exclusion family | Count | Classification and action |
| --- | ---: | --- |
| Adjacent or rigidly welded torso/limb self pairs | 15 | Necessary overlap-policy intent; redundant with current built-in parent/weld filtering. Retained as an explicit policy ledger. |
| Hands against all holds | 16 | Temporary contact workaround: hold-center connects conflict with hand geometry. Stage 2 debt. |
| Forearms against all holds | 16 | Temporary broader legacy clearance workaround; not validated released-limb collision behavior. Stage 2 debt. |
| Shins against all holds | 16 | Unjustified as a general physical policy; retained solely as existing foot/route-scaffold debt pending Stage 2. No new blanket exclusion added. |
| Hands against wall | 2 | Temporary attachment/penetration workaround; Stage 2 debt. |

All 65 actual signatures and body identities are in the generated report. Default
MuJoCo parent/weld filters also skip some pairs beyond the literal 15 entries;
that is an intentional simplification, not exhaustive anatomically correct
self-contact. Other nonadjacent self interactions remain enabled.

The environment exclusions are explicitly *not* certified by Stage 1. Removing
them while retaining incompatible hold-center capture would be a Stage 2 contact
redesign. Generic humanoid collision-envelope corrections do not endorse those
attachment semantics.

## Input And Numerical Boundary

Generic schema validation rejects booleans/non-real values, NaN/Inf, nonpositive
physical scalars, invalid three-vectors, zero/overflowed normals, wrong enums,
duplicate IDs, and bad start/goal references. Normals use robust unit normalization.
Container inputs are defensively frozen; scene mappings remain compatible with
deepcopy, replacement, explicit serialization, and Stage 0 observer isolation.

The current backend explicitly rejects non-unit scene scale, rotated walls,
VOLUME/EDGE geometry, WALL GRASP/STEP, and starts that cannot be represented by
its attachment scaffold. WALL PRESS/SMEAR patches remain metadata-only. Coordinates
are metres in the existing X-horizontal/Y-wall-normal/Z-up world; there is no
generic coordinate conversion or scene-format expansion.

Backend numerical support, distinct from generic schema and clinical ranges:

| Quantity | Supported interval |
| --- | --- |
| Humanoid dimensions | 0.05 to 2.0 m |
| Mass scale | 0.1 to 10 |
| Strength scale | 1e-6 to 10 |
| ROM scale | 1e-6 to 1.2, with shoulder cap |
| Wall/region friction | 1e-6 to 10 |
| Wall half-size / region radius | 1e-6 to 100 m |
| Scene position components | absolute value at most 1000 m |

These conservative bounds exclude reproduced numerical pathologies, not infer a
human population. Twenty-one compile-only boundary/corner samples check native
array/option/stat finiteness. They do not certify every accepted morphology's
neutral collision separation or dynamics. Independent profile dimensions can
still require a different admissible pose.

Full-precision float serialization no longer rounds tiny valid geometry or motor
gear to zero. Derived primitive inertia/gear is checked, and `compile_model`
rejects nonfinite engine-derived arrays such as `actuator_acc0` or `dof_invweight0`.
The diagnostics check sampled and terminal state as well, before reductions.
Nonfinite forces/errors cannot disappear through `max(0, NaN)`; failed measurements
are reported as failure/null or an explicit diagnostic error, never a passing
strict-JSON report. Zero/sub-step diagnostic budgets are rejected.

## Tests And Changed Regressions

Full suite: **130 tests run, 129 passed, one missing-dependency-path test skipped**
because MuJoCo is installed. Thirteen grouped compiled-model tests cover topology,
all 25 signed axes, ROM/charts, mass/inertia/COM, geometry propagation, numerical
boundaries, name/power invariance, all direct motors, passive physics, symmetry,
timestep convergence, CLI evidence, output-fault injection, and observer safety.

Other changes are explicit:

- The root-positive-local-inertia XML assumption is replaced with rigid-pelvis
  mass accounting and compiled moving-weld/mass-matrix checks.
- Pelvis mass expectations change from 9 kg to 10 kg; total climber mass does not
  change. XML structural checks are not labeled anatomical validation.
- Narrowed-ROM initialization and debug targets are rejected without stepping or
  mutating live state. Invalid root quaternions are rejected too.
- The hostile observer now asserts scene-map mutation rejection while retaining
  all model/data/result isolation checks; immutability does not weaken Stage 0.
- The old production-long natural `ATTACH_FAILURE` expectation is retired. With
  the corrected physical model, its unchanged default sequence now returns three
  actual successes, 750 integrations, and 1.5 s. Base and compact also complete.
- Capture-failure integrity is retained using an explicit test-only LH/H7
  eligibility refusal: real MuJoCo stepping produces 710 integrations, actual LH
  detachment, `ATTACH_FAILURE`, truthful HUD/JSON, and no later moves. No production
  contact rule is altered by this fault injection.

The historical Stage 0 document's long-profile exit-1 result refers to its commit,
not current Stage 1 behavior. Neither old failure nor new success is a physical
feasibility result. Stage 0 release/target identity, invalid-request non-mutation,
ownership, continuity, accounting, numerical-failure, and renderer contracts remain
tested without weaker acceptance thresholds.

## Verification Commands

From `/home/yuchan/Desktop/project/boulder_prototype-gpt61`:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python -m unittest discover -s tests -v
python scripts/validate_model.py
MUJOCO_GL=egl python scripts/validate_model.py --render
python scripts/view_scene.py --headless --mode sequence
python scripts/view_scene.py --headless --mode sequence --profile long_reach_lower_grip
MUJOCO_GL=egl python scripts/render_demo.py --mode sequence --profile long_reach_lower_grip --output outputs/physical-model-stage1/climbing-regression
```

These nominal commands exit 0. The physical validator's positive and negative CLI
paths are tested; invalid compiled/sampled numerics exit 1 with strict JSON.
Generated outputs are ignored by Git. EGL exercises actual rendering; desktop GUI
interaction remains covered by Stage 0 fake-viewer isolation tests, not a claimed
live-display verification.

## Remaining Limitations

This coherent proxy still lacks calibrated human segment data, coupled clinical
ROM, scapula/neck/pronation/finger dynamics, independent mass-distribution traits,
torque-speed/power/fatigue models, and realistic soft tissue. Its mass/inertia and
collision proxies need not coincide with a true human material volume.

Arbitrary accepted profile combinations are not certified collision-free, and a
different physically admissible initializer may be needed. The old synthetic
climbing reference is explicitly unsupported under sufficiently narrowed ROM.

Hand/foot environment exclusions, equality feet, grip/capture semantics, PD
chatter, gain/capability confounding, permissive readiness, IK feasibility, and
route-specific references remain later-stage debt. No physics-based climbing
feasibility, planner, learned controller, or RL result is claimed.

Stop after Stage 1; do not begin Stage 2.
