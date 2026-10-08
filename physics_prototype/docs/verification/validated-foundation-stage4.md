# Validated Foundation: Stage 4 Single-Hand Transition

## Verdict

**Stage 4: COMPLETE.** One deterministic right-hand release, physical reach,
strict capture and new four-contact readiness passes at **2 ms and 1 ms**.
All four negative cases fail for their declared reasons. This is a single-limb
primitive, not generalized route planning, reinforcement learning, or validation
of the old three-move synthetic-wall route. Stage 5 is not implemented.

Starting verified commit: `33f103f9f4fb2485b0b30e3013ba261c141383ed`.
Authorized worktree: `/home/yuchan/Desktop/project/boulder_prototype-gpt61`.
Canonical Python 3.12.13, MuJoCo 3.14.0, NumPy 2.5.3 and Pillow 12.3.0 were used.
No dependency was added.

## Baseline Preservation

Before code edits, HEAD matched the verified Stage 3 commit and the worktree was
clean. Model validation, all 30 Stage 2 contact cases, all 86 Stage 3 physical
isolated cases, and four nominal/disturbed Stage 3 static holds were rerun and
passed. Baseline reports are stored separately under:

- `outputs/stage4-baseline-contacts/`
- `outputs/stage4-baseline-controller/`
- `outputs/physical-model-stage1/validation.json` for the model CLI.

After implementation the same acceptance suites passed again, with separate
contact/controller evidence under `outputs/stage4-preserved-contacts/` and
`outputs/stage4-preserved-controller/`. Stage 2's unchanged historical benchmark
still reports controller convergence false; this is its archived 80/1 PD baseline,
not a failure of the preserved Stage 3 controller.

The humanoid builder, schema, mass/inertia/ROM/axes, motor gear/limits, damping,
armature, contact geometry/sensor, friction, compliance, grasp/capacity law,
Stage 3 controller gains and physical readiness code are unchanged. Ordinary
feet still have zero equalities. The fixture has 32 qpos, 31 velocities, 25 motors,
78.3 kg mass, 15 self exclusions and zero added environment exclusions. Its six
equalities are hand point connects, two active before and after the transition.
No root wrench or undeclared generalized force is accepted in successful runs.

## Legacy Seed Diagnosis

The frozen `STATIC_STANCE_QPOS` was fitted near the old hold centers. Canonical
hand anchors instead lie at `center + (radius + 0.006) * outward_normal`.
H3/H4 require 71/66 mm offsets, matching the saved **70.817/65.747 mm** seed gaps.
The old palm penetration was about **70.731/65.622 mm**, previously hidden by
legacy environment exclusions. The foot assumptions were also inconsistent:
selected spherical sole normals are `(0,-0.5,0.866025)`, not flat soles with fixed
world-Y/98-mm offsets. Foot sites are 11 mm above the actual sole and 55 mm ahead
of its center. Positional IK alone omitted these orientation/footprint constraints.

`solve_contact_pose` now solves a complete explicit HOLD intent on scratch data:
canonical hand anchors and facing normals, sole normals/height/real overlap,
compiled ROM, free-root tangent coordinates and collision clearance. Bounded
NumPy DLS, backtracking and quaternion-manifold updates are used. Profile names
and hold IDs do not select special offsets or solver behavior. An already admitted
seed is preserved exactly; otherwise full scratch `initialize_static_reference`
is the final admission authority. Local failure is not global infeasibility proof.

The base legacy route now has an admitted geometric candidate in **7 accepted IK
steps**, without modifying capture rules:

| Contact | Initial Site Error | Final Site Error | Final Normal Error |
| --- | ---: | ---: | ---: |
| Left hand, H3 | 70.817 mm | 1.027e-8 m | 1.735e-8 rad |
| Right hand, H4 | 65.747 mm | 1.108e-8 m | 1.246e-8 rad |
| Left foot, H1 | 23.043 mm | 1.121e-8 m | 8.008e-9 rad |
| Right foot, H2 | 23.206 mm | 9.405e-9 m | 1.813e-9 rad |

Final hand signed distances are approximately +7.31e-9 m; foot normal residuals
are about 9.41e-9/7.07e-9 m. Scratch admitted hand reactions are 10.752/11.337 N,
below the unchanged H3/H4 capacities of 782/731 N. No unintended clearance
residual remains. `legacy_seed_diagnosis.json` records the candidate and initial
errors. It is **not adopted into the acceptance episode**, integrated, or claimed
to be a dynamically stable old-route solution.

## Authoritative Fixture

The validated Stage 3 body is reused with hip/knee/ankle reference angles
0.15/0.30/0.15 rad. These leave knee ROM margin for load redistribution while
keeping soles flat. Box ledge heights follow the actual sole geometry. Only
fixture hold placement differs; no body, contact law or controller parameter is
changed. A new 20 mm-radius hand hold is placed 60 mm above the right source.

Initial contact set:

```text
LEFT_HAND  -> left_hand
RIGHT_HAND -> right_hand
LEFT_FOOT  -> left_foot
RIGHT_FOOT -> right_foot
```

The source seed passes exact Stage 3 scratch admission and a two-second static
hold, including sustained physical readiness. Transition execution begins at
absolute simulation time 2 s and never resets. The source readiness history is
then independently sampled by the transition owner; it is not inherited by a
one-frame query.

The released limb is **RIGHT_HAND**, and target is **reach_target** with canonical
anchor `(0.21, -0.0704478838710712, 1.5081361441696237)` metres. The opposite hand
and both feet retain their original named supports throughout the successful
reach. Final contact set changes only RIGHT_HAND to `reach_target`.

## Architecture And Iteration

`single_hand.py` adds an explicitly single-hand executor rather than extending a
route planner or hiding new semantics in the archived multi-move scaffold.
`contact_ik.py` supplies scratch pose admission, intermediate whole-body references,
and a measured-body-frame arm reference generator. `motion_support.py` projects
nominal support shares onto six free-root equilibrium rows and derives motor
feedforward. Only hinge torque is applied; estimated contact forces are not.

The phases are:

```text
SOURCE_STABILIZE -> LOAD_TRANSFER -> RELEASE_CLEARANCE -> RELEASE
  -> THREE_POINT -> REACH -> SETTLE -> sustained new readiness
```

Several honest candidates failed. Fixed-pose three-contact feedforward led to
body drift during longer reaches, and unconstrained scratch root motion produced
references that the unactuated physical root could not directly track. Centered
arm candidates could also collide with the chest and were rejected by admission.
No gains, gravity, contact exclusions or slip limits were altered to rescue them.

The successful solution updates support feedforward from measured pose, preserves
support-joint reference intent, and generates the moving five-joint arm reference
in the measured root/torso frame. This eliminates the need for a Cartesian
integrator that would request virtual hold penetration. The physical root remains
free; no inverse-kinematic pose is copied into live data after execution begins.

## Physical Three-Contact Support

Load transfer blends four-contact to three-contact hinge feedforward over one
second. A further half-second outward reference first unloads real source-palm
collision while its bounded hand equality is still active. This matters: simply
turning off the connect left one applied collision interval carrying source-hand
load, which the hidden-support audit correctly rejected.

Only after physical palm unloading does the real grasp manager disable the source
equality. A declared **0.5 s three-contact interval** precedes reaching. Throughout
that interval and reach, both feet must have their own loaded native HOLD surfaces,
normal force >5 N, alignment/friction validity and slip speed <=0.01 m/s. The
opposite hand remains valid and capacity bounded. Applied and fresh endpoint
contacts are audited; moving-hand environmental support is not counted or allowed.

Intentional-motion safety bounds are declared separately from final readiness:
root linear <=0.10 m/s, root angular <=0.50 rad/s, max hinge <=1 rad/s, and finite
state. Contact/capture/capacity thresholds remain identical to Stage 2. After
capture, ordinary Stage 3 0.5-second readiness uses root linear <=0.02 m/s,
angular <=0.05 rad/s and maximum hinge <=0.10 rad/s.

## Trajectory And Controller

The task-space path uses a minimum-jerk fraction:

```text
s = clamp(t / reach_duration, 0, 1)
b = 10*s^3 - 15*s^4 + 6*s^5
p_ref = (1-b)*p_start + b*p_target + 0.02*sin(pi*s)^2*outward_normal
```

Declared reach duration is **4 s**, with a further **2 s capture timeout**. The
outward clearance arch has zero endpoint rate. Quaternion interpolation uses the
matching local-frame increment, verified for rotated goals, not only this fixture.
The commanded starting frame preserves existing reference continuity rather than
teleporting the hand to an ideal FK pose.

`solve_hand_reference` changes only the selected shoulder pitch/roll/yaw, elbow
and wrist on scratch data, solving position plus facing-normal tasks. Root,
waist and every other coordinate remain untouched by that solver. Its five DOFs
need not satisfy an artificial sixth tangent-twist task: capture requires facing,
not a fixed hand roll. Compiled ROM, finite state and bounded local search apply.
Geometric convergence is never itself capture or dynamic acceptance.

A shared motor-reference shaper bounds velocity to 0.5 rad/s and acceleration to
2 rad/s^2, including clearance/support/capture/settling phase boundaries. Reference
positions and rates are mutually consistent; actual maxima were approximately
0.259313/0.258807 rad/s and 2 rad/s^2 at 2/1 ms. This shapes references only.
The low-level torque equation, gains and capability limits are the unchanged
Stage 3 controller. No second torque controller, physical damping or armature is
introduced.

## Strict Capture And New Readiness

Every actual attempt is through `GraspManager.can_attach/attach`, with distance
<=1 mm, speed <=0.05 m/s, facing >=cos(30 degrees), <1 mm penetration and bounded
candidate reaction. Target equality never activates while outside the gate.
The first geometric-eligible frame also passes capacity and becomes the actual
capture in both success runs. Immediately after activation, **all active hands**
are checked at that same force epoch, before any four-contact control changes.

After capture, a new four-contact static reference is scratch-admitted from the
actual captured configuration, then blended into over 0.5 s. It is not adopted
into live pose. A fresh readiness owner requires the unchanged sustained Stage 3
criteria and completed reference blending before SUCCESS. The initial and final
states are therefore both measured ready states, not merely reached targets.

## Two-Timestep Results

Times below are absolute episode times; transition work starts at 2 s. Foot loads
and utilization cover the native transition, not the prior initialization hold.

| Measured Metric | 2 ms | 1 ms |
| --- | ---: | ---: |
| Source readiness recertified | 2.502 s | 2.501 s |
| Source equality released | 4.002 s | 4.001 s |
| Reach starts | 4.502 s | 4.501 s |
| First valid capture time | **8.264 s** | **8.262 s** |
| Capture distance | **0.988638016 mm** | **0.999716305 mm** |
| Facing cosine | 0.9999999621 | 0.9999999612 |
| Capture speed | 0.007171870 m/s | 0.007194650 m/s |
| Capture penetration | 0 mm | 0 mm |
| Fresh initial target reaction | 169.416129 N | 171.463151 N |
| Simultaneous opposite-hand post-activation reaction | 84.742778 N | 84.743556 N |
| New sustained readiness time | **8.776 s** | **8.774 s** |
| Transition steps / duration | 3388 / 6.776 s | 6774 / 6.774 s |
| Three-contact support plus reach duration | 4.262 s | 4.261 s |
| Three-contact max root linear speed | 0.029842598 m/s | 0.029772110 m/s |
| Three-contact max root angular speed | 0.048448925 rad/s | 0.048439539 rad/s |
| Three-contact max hinge speed | 0.344999121 rad/s | 0.344193091 rad/s |
| Remaining-hand peak reaction | 88.411311 N | 88.401471 N |
| Maximum motor utilization | 45.553537% | 45.554622% |
| Maximum distance to target over reach | 62.947834 mm | 62.933345 mm |
| Maximum Cartesian reference-tracking error | 1.380808 mm | 1.423840 mm |
| Both-foot support fraction | 100% | 100% |
| Max slip left/right | 0.005197491 / 0.002313532 m/s | 0.005196074 / 0.002312170 m/s |
| Min Fn left/right | 347.399959 / 343.635991 N | 347.460083 / 343.657786 N |
| Mean Fn left/right | 359.584149 / 351.067882 N | 359.587334 / 351.060810 N |
| Max Ft left/right | 70.374656 / 34.084700 N | 70.374608 / 34.086073 N |
| Final root linear speed | 0.000655258 m/s | 0.000651632 m/s |
| Final root angular speed | 0.002464001 rad/s | 0.002482259 rad/s |
| Final max / RMS hinge speed | 0.006841111 / 0.002376777 rad/s | 0.006879964 / 0.002379607 rad/s |
| Sustained final readiness | 0.500 s | 0.500 s |

Hand capacities are 850 N each. No successful-run release from overload, foot
slipping, hidden body support, numerical recovery, reset or live qpos/qvel write
outside native integration occurred. The ~63 mm maximum target distance includes
the start/clearance path; it is not tracking error. The millimetre-scale tracking
error during intentional motion is not substituted for strict actual capture.

## Negative Cases And Failure Truthfulness

The default negative matrix runs at 2 ms; `--dt 0.001 --suite negative` can select
the alternate cadence. Correct expected failures count as test acceptance, never
as successful physical transitions.

| Case | Expected/Observed Failure | Actual Evidence |
| --- | --- | --- |
| Target 2 m higher, outside local arm reach/ROM | REACH_INFEASIBLE | Zero transition steps and no release; measured local search failure, not universal infeasibility proof. |
| Near target with a 31-degree incompatible approach | CAPTURE_FAILURE | Scratch site gap 1.9206e-12 m, facing 0.857167301 below cos(30); also 3.713650 mm penetration. No activation or live pose adoption. |
| Support loss during reach | CONTACT_LOSS | Declared +1500 N X foot disturbance at reach +0.3 s; actual native support/slip failure, stopped after 1402 transition steps. |
| Insufficient grasp for post-capture load | GRIP_FAILURE | Declared -2000 N Y pull after two settling steps; native demand **1989.219777 N > 850 N**, target detached before another integration; 3134 completed steps. |

The last case demonstrates insufficient physical capacity for a declared imposed
load, not a universal weak-profile feasibility study. Fault forces are visible
in evidence and are not present in success. All generalized/body forces are
rejected on entry and audited every step, except the exact declared fault channel.
Cleanup restores only that owned force component; caller changes are not erased.

Invalid references are rejected before control writes or integration, including
nonunit quaternions, ROM violations, forged contact admission and string keys
masquerading as Limb enums. Unknown scenarios/faults cannot silently run nominal
physics. Failure results preserve actual endpoint/equality/controls, force epoch,
step count and release logs. Final-step failures cannot become SUCCESS. Terminal
torque/utilization reflect live control even after native recovery. Callback
exceptions propagate after cleanup without being reclassified or retried as a
second terminal observation. Unexpected guard failures produce CONTROL_FAILURE
with actual state rather than resetting the episode.
An unavailable optional hand-position solve is explicitly nullable and records
its diagnostic error; it cannot mask the original failure or create a valid
render pose. Its failure counterexample is included in the final suite.

## Verification And Artifacts

Final canonical full discovery: **294 tests run, 293 passed, 1 skipped**,
**1278.460 s**, exit 0. The skip is the absent-MuJoCo path because MuJoCo is installed.
No previous tests were weakened or removed. New coverage comprises 31 contact IK
tests, 16 actual single-hand tests and 15 boundary tests. Actual compiled-model
audits verify native-only pose evolution, unchanged model/controller arrays,
strict capture, continuous bounded references, palm unloading, simultaneous grip
guards, final readiness and adversarial observer isolation.

Stage 1 model CLI, Stage 2's 30 cases, and Stage 3's 86 physical isolated plus four
static cases all pass both baseline and preservation runs. Stage 4 CLI reports
two successful transitions, four correct failures and two successful EGL renders,
with no execution/render errors. Existing deliberate nonfinite and small-video
padding warnings remain expected regression output.

Outputs are Git-ignored under `outputs/transition-stage4/`:

- `report.json`: provenance, baseline hashes, independent success/negative/render verdicts and metrics.
- `legacy_seed_diagnosis.json`: exact scratch candidate and geometric evidence; no live adoption.
- `success_2ms.json`, `success_1ms.json`: full authoritative q/qdot, q_ref/qd_ref,
  root state, torque/utilization, active contacts, forces/slip, hand error/facing,
  equality state, phase and readiness traces.
- `success_2ms.mp4`, `success_1ms.mp4`: same-run 640x480, 20 fps videos, 136 observations each.
- `success_2ms_endpoint.png`, `success_1ms_endpoint.png`: actual successful terminal frames.
- `success_2ms_observer.json`, `success_1ms_observer.json`: detached same-run observer records.
- Four named negative-case JSON files: actual terminal states, steps and reasons.

Rendering observes the authoritative executor through copied model/data/row
snapshots. No physics is replayed. The two-second initial static evidence is reused
without manufacturing video frames; rendered time begins at transition entry.
Instantaneous release/capture events are recorded even when between display ticks.
HUDs separate virtual reference error from physical capture measurements and show
actual terminal SUCCESS/readiness. Nonfinite state is not forwarded as a valid pose.

Run from the authorized worktree:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python scripts/validate_model.py
python scripts/validate_contacts.py --summary --output outputs/stage4-preserved-contacts
python scripts/validate_controller.py --summary --output outputs/stage4-preserved-controller
MUJOCO_GL=egl python scripts/validate_transition.py --suite all --render --summary
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

Baseline commands used the same contact/controller arguments with output names
`stage4-baseline-contacts` and `stage4-baseline-controller`, before code edits.
Preservation artifact directories are separate and the transition CLI does not
overwrite or newly certify the baseline reports merely by referencing them.

## Changed Files And Limits

New mechanisms: `src/boulder_v1/contact_ik.py`, `motion_support.py`, `single_hand.py`.
New diagnostic: `scripts/validate_transition.py`. Public exports are added in
`src/boulder_v1/__init__.py`. New tests are `test_contact_ik.py`, `test_single_hand.py`
and `test_single_hand_boundaries.py`. README and this report describe scope.
All validated physics/controller/contact/readiness source files remain unchanged.

This certifies one nearby upward hand transfer, not arbitrary layouts, limb orders,
holding indefinitely, large disturbances, optimal speed or route feasibility.
IK is local and references alone do not certify dynamic trajectories. The old
synthetic route's new geometric seed still needs independent static/dynamic
verification before any old multi-move success claim. The hand remains a force-only
point surrogate with no calibrated finger moments, hooks or fatigue.

Recommended Stage 5 scope: generalize this verified execution/admission contract
to a small explicit family of limb transfers and short contact sequences, with
measured support feasibility, collision-aware references, failure propagation and
unchanged contact bounds. Do not treat this single fixture as a route planner or
introduce RL before those movement contracts are established.

**Stop after Stage 4. No Stage 5 implementation is started.**
