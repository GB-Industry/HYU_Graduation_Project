# Validated Foundation: Stage 5 Explicit Transfer Family

## Verdict

**Stage 5: COMPLETE.** Full regression discovery passes, and the canonical
all-case CLI rerun recorded 23 finite episodes: eight
successful family cases at 2 ms and 1 ms, nine correctly classified negatives,
and six sensitivity diagnostics. The family completed ten physical moves,
including a right-hand then left-hand sequence at both timesteps. All eight
requested same-run EGL videos succeeded, with no execution or render errors.

Sensitivity acceptance is not six physical successes: five diagnostics succeeded
physically and one returned an honest `CONTROL_FAILURE`. Expected negative
failures are accepted tests, never successful transfers. The CLI's `passed: true`
is complemented by independent full-suite and Stage 1-4 preservation results.

Starting verified commit: `b14ea3b`. The run owner verified a clean worktree before
implementation. Authorized worktree:
`/home/yuchan/Desktop/project/boulder_prototype-gpt61`.
Canonical runtime: Python 3.12.13, MuJoCo 3.14.0, NumPy 2.5.3, with EGL rendering.
The interpreter is
`/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python`; imports and
artifacts belong to the authorized worktree. No dependency was added.

This establishes a small explicit family: either hand, one left-foot transfer,
and a two-hand chain. It is not arbitrary route planning, an optimizer, RL,
personalization, or validation of every limb order/contact layout.

The numerical verdict does not certify visual motion quality. The original
640x480 HUD videos were subsequently rejected as usable full-body evidence.
The separate `validated-foundation-stage5-visual.md` audit records clean HD
multi-camera exports and **PHYSICS SUCCESS / MOTION DEMONSTRATION INADEQUATE**
for the small whole-body hand/sequence motion, without changing the physics.

## Baseline Preservation

The pre-edit baseline and post-implementation preservation runs retain the same
acceptance matrix:

| Foundation | Before Implementation | After Implementation |
| --- | --- | --- |
| Stage 1 compiled model | PASS | PASS |
| Stage 2 physical contact matrix | 30/30 accepted | 30/30 accepted |
| Stage 3 physical isolated controller matrix | 86/86 accepted | 86/86 accepted |
| Stage 3 nominal/disturbed static holds | Four successes, both timesteps | Same four successes |
| Stage 4 default single-hand transition | Success at 2 ms and 1 ms | Same two successes |
| Stage 4 negative matrix | Four expected failures | Same four expected failures |

Separate reports are retained at:

- `outputs/stage5-baseline-contacts/report.json`
- `outputs/stage5-baseline-controller/report.json`
- `outputs/stage5-baseline-transition/report.json`
- `outputs/stage5-preserved-contacts/report.json`
- `outputs/stage5-preserved-controller/report.json`
- `outputs/stage5-preserved-transition/report.json`
- `outputs/physical-model-stage1/validation.json` for Stage 1 model evidence.

Stage 2's archived controller-convergence result remains false; its independent
contact acceptance remains true. Stage 3's 86 physical isolated cases exclude
four archived controller baselines, whose two deliberate failures do not affect
production acceptance. These are unchanged distinctions, not newly waived gates.
The Stage 5 report references baseline paths/hashes with `rerun: false`; merely
including those references does not rerun or newly certify their contents.

The production builder, schema, contact geometry/law, grasp manager/capacity law,
foot support sensor, Stage 3 static controller and readiness implementation are
untouched. Mass/inertia, body dimensions, ROM/axes, motor gear/limits, damping,
armature, friction and compliance are not retuned. The default body remains
78.3 kg with 32 qpos, 31 velocities and 25 motors. Shoe dimensions and collision
semantics are unchanged; no environment collision exclusions or foot equalities
are introduced. New fixture holds are placements of existing canonical geometry,
not new shoe geometry or altered contact types. Boxes and spheres retain their
validated semantics.

`motion_support.py` extends the Stage 4 reference estimator to one foot plus two
hands and checked world-space native force points. Its default two-foot
feedforward allocation is unchanged. The Stage 4 request still defaults to
`first_eligible` and preserves its original capture at reference time 3.76 s.
Latest verification work made no physics, gains, capture or readiness changes.

## Execution Contract

`TransferRequest` declares an actual typed `Limb`, source HOLD, target HOLD and
complete `source_contacts`. Optional `support_contacts` must be exactly the source
contact set with the one moving limb excluded. Contact identities, eligible
affordances, session/profile binding, finite state and request structure are
validated before commands or integration. Successful entry requires the actual
native four-contact source, not an inferred contact set from the scene's initial
configuration. Undeclared `qfrc_applied` and `xfrc_applied` are rejected atomically.

Intentional-motion safety bounds are fixed at root linear speed <=0.10 m/s,
root angular speed <=0.50 rad/s and maximum hinge speed <=1 rad/s. Request fields
cannot raise those limits. Final acceptance uses the unchanged Stage 3 readiness
gate: sustained 0.5 s with valid physical contacts, root linear <=0.02 m/s,
root angular <=0.05 rad/s and maximum hinge <=0.10 rad/s.

Execution uses the existing Stage 3 torque PD/impedance equation and gains, with
the original profile-scaled motor capabilities. There is no name-based controller
selection. Five-arm-joint and six-leg-joint reference solves use measured FK and
compiled live-model morphology/ROM on scratch data. They do not copy solved
root/limb coordinates into live data. Reference shaping retains the 0.5 rad/s
speed and 2 rad/s^2 acceleration bounds; these constrain references, not physical
capture acceptance. Only `mj_step` evolves successful live poses and velocities.

After real acquisition, the new contact intent is scratch-admitted into an
immutable static reference. Success requires that reference and ordinary fresh
readiness, not just endpoint IK convergence. The bound model, data, scene and
manager can be reused with that exact returned reference. Every failure returns
`final_reference: None` and preserves the actual endpoint rather than fabricating
a usable handoff or resetting the episode.

## Authoritative Fixture

The Stage 4 source body/stance is reused. Source contacts are:

```text
LEFT_HAND  -> left_hand
RIGHT_HAND -> right_hand
LEFT_FOOT  -> left_foot
RIGHT_FOOT -> right_foot
```

Both hand targets are 60 mm above their respective sources, using the same
canonical anchor/facing convention. The left-foot STEP target is 105 mm outward
and 20 mm upward from its source ledge. There is a 5 mm initial shoe-to-target
horizontal gap: the starting shoe's minimum X is -0.1675 m and the target's right
edge is -0.1725 m. The new ledge cannot secretly support the starting foot.
Default target friction is 0.9; the existing shoe friction cap is 1.8.

Scratch source admission precedes a real two-second Stage 3 static hold. Transfer
execution starts at absolute time 2 s and independently recertifies source
readiness. The initial hold is evidence, not replayed video or a second reset.

## Support Allocation

For left-foot unloading, the support set is both hands plus RIGHT_FOOT. Nominal
shares allocate 0.9 of body weight to the remaining foot and 0.1 divided between
the hands. With two feet the original 0.9 share is divided between them. A least
squares correction balances the six unactuated free-root equilibrium rows.
Unilateral compression, the existing friction pyramid, bounded grip capacities
and original motor limits remain enforced. Estimated reactions are never injected
as contact forces or a secret root wrench; only derived hinge feedforward is
passed to the unchanged controller.

Dynamic foot force points are centroids of the foot's own admissible native
contacts on its declared surface. Supplied points must be finite world vectors
inside the real shoe and canonical surface patch, including box/sphere geometry
checks. The default geometric two-foot estimate retains its old behavior.

During early `LOAD`, a temporary cache may retain the first native target contact
point until the first native >5 N support observation. It is only a torque-estimate
point, never evidence of support. Cached points still undergo geometric bounds
checks and cannot replace real acquisition or later support measurements. After
source separation, any old-source foot contact record is invalid, even at zero
load, in `LANDING`, `LOAD` and `SETTLE`, at both applied and fresh endpoint epochs.

This is a bounded candidate allocation with declared nominal weights, not a
universal contact-force optimizer. Candidate rejection or local IK failure is not
proof that the physical task is globally infeasible.

## Hand Capture Audit

Actual attachment still uses `GraspManager.can_attach/attach`: distance <=1 mm,
relative speed <=0.05 m/s, facing >=cos(30 degrees), penetration <1 mm, and bounded
candidate reaction. Canonical target frames remain correct. No equality activates
outside this gate and no capture threshold is relaxed.

The preserved Stage 4 first-eligible event occurs before the four-second reference
endpoint. Its tiny margins are an early gate crossing, not an endpoint tracking
floor. The optional family `endpoint_settle` policy waits until reference time
4.1 s and requires non-increasing measured distance and speed under the same
physical gate. It changes acquisition timing, not physical acceptance thresholds.

| Measured Hand Quantity | 2 ms | 1 ms |
| --- | ---: | ---: |
| Preserved Stage 4 first-eligible reference time | 3.76 s | 3.76 s |
| Preserved first-eligible distance | 0.988638016 mm | 0.999716305 mm |
| Preserved first-eligible distance margin | 0.011361984 mm | 0.000283695 mm |
| Stage 5 isolated RH/LH actual capture distance | 0.081555745 mm | 0.082696027 mm |
| Stage 5 isolated RH/LH actual capture margin | 0.918444255 mm | 0.917303973 mm |
| Stage 5 fresh initial target reaction | About 9.37 N | About 9.51 N |
| Stage 4 fresh initial target reaction | About 169.42 N | About 171.46 N |

Both isolated hands exhibit the stated Stage 5 distances to the shown precision.
The minimum family capture margin is 0.917303973 mm. Immediately after attachment,
all active hands are checked using fresh post-activation reaction evidence before
any four-contact control change. The Stage 4 and Stage 5 reaction values are
different actual capture epochs, not interchangeable telemetry.

The observed margins above 0.9 mm are empirical regression observations, not a
new capture criterion. Four tiny scratch-seed perturbation diagnostics, using
`waist_yaw` +/-1e-4 rad before admitted initialization, also succeed with margins
approximately 0.918443 mm at 2 ms and 0.917303 mm at 1 ms. This is limited local
sensitivity evidence, not a broad robustness envelope.

## Native Foot Acquisition

The left-foot protocol is:

```text
SOURCE_STABILIZE -> UNLOAD -> LIFT -> THREE_POINT -> REACH
  -> LANDING -> LOAD -> SETTLE -> stationary-reference Stage 3 readiness
```

Unloading is two seconds, lift is 1.5 seconds, the declared three-point interval
is 0.5 seconds, and reach is three seconds with a bounded contact timeout. Source
release means actual native separation: zero source contacts and zero normal
force, not a disabled foot equality. Both hands and the opposite foot retain
their named supports. No foot equality exists.

Touchdown requires own-target compressive native contact but does not count as
acquisition below 5 N. `ACQUIRED` requires the existing sensor's normal load >5 N,
non-slip speed <=0.01 m/s, valid friction/alignment and the intended support
surface continuously for 0.1 s. The full two-second `LOAD` still runs after this
milestone. A new static reference is then admitted and settled; fresh readiness
is owned after the reference stops, rather than inherited from moving references.

| Foot Milestone, Absolute Episode Time | 2 ms | 1 ms |
| --- | ---: | ---: |
| Source separation | 4.626 s | 4.625 s |
| Native touchdown | 9.496 s | 9.495 s |
| Touchdown normal load, not supporting | 0.896233 N | 0.887666 N |
| First native >5 N support | 9.732 s | 9.730 s |
| First supporting normal load | 5.096235 N | 5.040368 N |
| `ACQUIRED`, after sustained 0.1 s | 9.832 s | 9.830 s |
| Acquired normal load | 12.847367 N | 12.748181 N |
| Full loading completed | 11.496 s | 11.495 s |
| Final stationary-reference readiness | 12.628 s | 12.628 s |
| Transfer native steps / duration | 5314 / 10.628 s | 10628 / 10.628 s |

After separation, the source foot has zero contacts and Fn=0 through the rest of
the successful transfer. Light touchdown is explicitly recorded as contacting,
not supporting. Geometric references and the loading cache do not count as native
support or manufacture an adhesive landing.

## Physical Results

All family episodes begin at absolute time 2 s. Durations and steps below exclude
the initial static hold; all eight finish with an admitted final reference and
sustained readiness.

| Family Case | 2 ms Steps / Duration | 1 ms Steps / Duration | Result |
| --- | ---: | ---: | --- |
| RIGHT_HAND -> reach_target | 3553 / 7.106 s | 7103 / 7.103 s | SUCCESS at both |
| LEFT_HAND -> left_reach_target | 3553 / 7.106 s | 7103 / 7.103 s | SUCCESS at both |
| LEFT_FOOT -> foot_target | 5314 / 10.628 s | 10628 / 10.628 s | SUCCESS at both |
| RIGHT_HAND then LEFT_HAND | 7106 / 14.212 s | 14206 / 14.206 s | Two moves at both |

Native summary measurements, not reference acceptance proxies:

| Metric | Isolated RH, 2 ms | Isolated RH, 1 ms | LF, 2 ms | LF, 1 ms |
| --- | ---: | ---: | ---: | ---: |
| Three-point max root linear speed, m/s | 0.029842598 | 0.029772110 | 0.003745864 | 0.003782295 |
| Three-point max root angular speed, rad/s | 0.048448925 | 0.048439539 | 0.014196104 | 0.014114349 |
| Three-point max hinge speed, rad/s | 0.344999121 | 0.344193091 | 0.223567231 | 0.223575539 |
| Maximum actuator utilization | 45.607279% | 45.608494% | 79.076137% | 79.078665% |
| Maximum LF/RF slip speed, m/s | 0.005197491 / 0.002378566 | 0.005196074 / 0.002376324 | 0.004081232 / 0.007783971 | 0.004096118 / 0.007838218 |

The report's `three_point` aggregates the declared three-point/reach interval;
for the foot it includes landing before loading. Both feet retain 100% support
during isolated hand moves. At 2 ms the RH move's minimum LF/RF normal loads are
347.399959/343.635991 N, and the remaining-hand peak reaction is 88.411311 N.
The left-hand move mirrors these measurements with LF/RF labels exchanged.
During foot motion only the opposite foot is required to remain supporting until
target acquisition. Slip maxima for the moving foot concern contact observations,
not airborne support.
The detailed report retains foot loads, hand loads/capacity margins and endpoint
support identities. Both timesteps give the same qualitative outcome; no claim
of bit-identical trajectories across timesteps is made.

## Sequence Continuity

The declared RH -> LH chain initializes only once and does not reset between
moves. The first move's final qpos, qvel, ctrl, warmstart, equalities, time and
capture/release history are the second move's initial state. The second receives
the exact first `final_reference` and the same bound model/data/scene/manager.
`scene.start_configuration` remains the original source configuration; it is not
rewritten to conceal a reset or replace native contact identity.

| Contact State | LEFT_HAND | RIGHT_HAND | LEFT_FOOT | RIGHT_FOOT |
| --- | --- | --- | --- | --- |
| S0 | left_hand | right_hand | left_foot | right_foot |
| S1, after RH | left_hand | reach_target | left_foot | right_foot |
| S2, after LH | left_reach_target | reach_target | left_foot | right_foot |

The sequence ends at absolute 16.212 s at 2 ms and 16.206 s at 1 ms. Its second
hand capture distances are 0.078049196/0.079186390 mm, with respective margins
0.921950804/0.920813610 mm. Each move independently reacquires physical readiness
and delivers its immutable admitted reference. No limb/root teleport, scratch
pose adoption or initial-static replay supplies a sequence endpoint.

## Negative Matrix

All nine default negatives run at 2 ms, remain finite and preserve their actual
terminal state. Status matching is acceptance of the declared counterexample,
not a successful physical move.

| Case | Expected And Observed Status | Native Steps | Evidence And Boundary |
| --- | --- | ---: | --- |
| `unreachable` | REACH_INFEASIBLE | 0 | Target 2 m higher; bounded local arm search fails before release. Not global infeasibility proof. |
| `orientation_invalid` | CAPTURE_FAILURE | 0 | Scratch site gap about 1.92e-12 m but facing is cos(31 degrees)=0.857167301; penetration is also 3.713650 mm. No activation or live adoption. |
| `support_loss` | CONTACT_LOSS | 1402 | Preserved Stage 4 +1500 N X disturbance on the supporting left foot at reach +0.3 s produces native support/slip failure. |
| `grip_after_capture` | GRIP_FAILURE | 3134 | Preserved Stage 4 -2000 N Y pull after two settling steps demands 1989.219777 N >850 N; target detaches before another integration. |
| `hand_support` | THREE_POINT_SUPPORT_FAILURE | 0 | Declared 60 N grip permits source readiness but the candidate remaining hand requires 87.733191 N. Capacity rejection precedes commands/release. |
| `foot_missing_step` | INELIGIBLE_TARGET | 0 | Target lacks STEP affordance; no foot attachment, release or integration. |
| `foot_low_friction` | SUPPORT_INFEASIBLE | 0 | Declared target mu=0.005 rejects candidate predicted friction demand. Not an executed failed landing or a zero-load adhesive test. |
| `foot_support_loss` | CONTACT_LOSS | 2402 | Declared +1500 N X force on the supporting right foot at reach +0.3 s causes native support failure during LF execution. |
| `sequence_second_unreachable` | REACH_INFEASIBLE | 3553 total | RH completes one real move; LH then fails at zero additional steps. No reset, command, force or time/state writes rescue the second move. |

The weak-hand diagnostic keeps a physically valid source with roughly 50 N hand
loads before predicting the 87.733191 N remaining-hand demand. It does not bypass
capacity in execution. The low-friction result concerns the fixed candidate
allocation, not all possible landing/controller strategies. The two disturbance
cases and post-capture overload are actual declared physics faults, distinct
from scratch geometry/affordance rejection. Their forces are absent in success.

## Sensitivity Diagnostics

Both fixed-scene profiles are geometrically admitted and pass their actual source
static hold. All six diagnostics remain finite. Profile outcomes are recorded
without requiring a particular success/failure as the diagnostic acceptance gate.
No holds, target placements or controller gains are changed to manufacture profile
differences; these are not personalization benchmarks.

| Diagnostic | Timestep | Physical Outcome | Evidence |
| --- | --- | --- | --- |
| `compact_strong` | 2 ms | SUCCESS | Capture error 0.066933296 mm, margin 0.933066704 mm, maximum motor utilization 40.184601%. |
| `long_reach_lower_grip` | 2 ms | CONTROL_FAILURE | 3167 steps; conservative left-sole reference geometry check fails during reach, before capture. |
| `perturbed_right_hand` | 2 ms | SUCCESS | Admitted +1e-4 rad waist-yaw scratch seed; unchanged physics/gates. |
| `perturbed_right_hand` | 1 ms | SUCCESS | Same declared seed perturbation, independently integrated. |
| `perturbed_left_hand` | 2 ms | SUCCESS | Admitted -1e-4 rad waist-yaw scratch seed; unchanged physics/gates. |
| `perturbed_left_hand` | 1 ms | SUCCESS | Same declared seed perturbation, independently integrated. |

The compact profile uses its existing 1.15 strength scale, 1.08 ROM scale and
1000 N grip capacity. The long-reach profile retains its existing 0.92 strength,
0.95 ROM scale and 850 N grip capacity despite its descriptive name. Morphology,
mass/ROM/strength parameters flow through the normal compiled model and immutable
profile contract, not controller branches keyed on profile names.

The long-reach failure reason is
`LEFT_FOOT: sole is not touching the intended canonical support face with real overlap`.
This is a conservative reference/controller boundary: a static force-point
geometry estimate is inconsistent with measured foot support. It is not a claim
of real native contact loss. At its actual endpoint, both feet are supporting
and non-slipping with Fn=366.036410/348.587921 N. Whole-episode slip maxima are
0.006211254/0.002615051 m/s, below the unchanged 0.01 m/s limit. Local geometry and
IK admission do not prove dynamic controller feasibility; neither this failure
nor the compact success establishes physical infeasibility or personalized skill.

## Failure Boundaries

- All active hands are guarded with fresh post-activation reactions before
  post-capture control changes; overload is not hidden by a stale force epoch.
- Applied and fresh endpoint foot evidence must retain named native support.
  Old-source contact records after separation are invalid even with zero force.
- Undeclared generalized/body forces are forbidden throughout execution. Declared
  fault cleanup restores only components still equal to the owned force, retaining
  caller replacements and unrelated channels rather than erasing live evidence.
- Observers receive detached model/data/row copies, observational with respect to
  live physics. Editing the supplied copies cannot change the authoritative run.
- A closure that writes running live state produces `CONTROL_FAILURE` before
  another command, retaining the actual mutated endpoint without rollback. A
  terminal live write propagates an error and cannot return a false valid handoff.
- The original observer exception object propagates after owned-force cleanup;
  there is no retry or second terminal observation hiding the original failure.
- Numerical recovery or a discontinuous native clock cannot manufacture elapsed
  success. Terminal state/equalities/time remain actual; terminal commanded Nm
  and utilization are derived from live `ctrl` and original motor gear/limits.
- Invalid requests/references are rejected before integration. Failure readiness
  is false and final references are absent; a later sequence move cannot run from
  a fabricated reference after the first failure.

Boundary clock/contact stubs test these delivery/guard contracts, not physical
success. Native acceptance evidence comes from the real episodes and retained
state/contact traces, not from stubbed readiness or support.

## Regression Status

Final canonical discovery: **375 tests run, 374 passed, 1 skipped**, **4125.989 s**,
exit 0. The skip is the missing-MuJoCo path because MuJoCo is installed. An earlier
parallel one-hour attempt timed out; it is not counted as acceptance. Full discovery
was rerun separately with a two-hour budget and completed successfully.

Some archived renderer regression exports reported resource-related FFmpeg
`Cannot allocate memory` warnings. Their actual-state/export-failure semantics
passed; those temporary exports are not claimed as successful video evidence.
The independent canonical Stage 5 export produced all eight requested videos
without errors. Deliberate numerical-fault tests retain their expected warnings.

Existing assertions and physical thresholds were not weakened or removed. The
old support-allocation stub gained the required diagnostic fields; recovery
telemetry fault injection now occurs after the first native step rather than
preflight, preserving the intended failure assertions. The new native foot and
transfer test classes clear their large cached episodes at class teardown to
reduce retained memory, without cutting cases or assertions.

Coverage includes measured six-joint foot references, both-hand capture margins,
native LF unloading/separation/touchdown/acquisition, one-foot support allocation,
strict request admission, immutable reference handoff, exact two-hand native
continuity, second-move failure, both force epochs, zero-load source recontact,
observer isolation/exception identity, force ownership cleanup and terminal
telemetry. Relevant files are `tests/test_foot_reference.py`,
`test_hand_family.py`, `test_foot_transfer.py`, `test_motion_support.py`,
`test_transfers.py`, `test_transfer_boundaries.py` and `test_validate_transfers.py`.
All of these tests are included in the completed full discovery result.

## Artifacts And Commands

Authoritative Stage 5 outputs are Git-ignored under `outputs/transfers-stage5/`:

- `report.json`: canonical runtime/module provenance and hashes, separate family,
  negative/sensitivity/render verdicts, all 23 case summaries and detailed native
  measurements. Its execution and render error arrays are empty.
- Twenty-three named case JSON files: full traces, actual initial/terminal states,
  references, forces/contacts, phases/events, exact failure reasons and validation
  metadata. Stage 5 has no separate `isolated.json`; Stage 3's isolated evidence
  remains in its controller baseline/preservation directories.
- `right_hand_2ms.mp4`, `right_hand_1ms.mp4`
- `left_hand_2ms.mp4`, `left_hand_1ms.mp4`
- `foot_2ms.mp4`, `foot_1ms.mp4`
- `sequence_2ms.mp4`, `sequence_1ms.mp4`
- Same-prefix `_endpoint.png` and `_observer.json` files for each of the eight
  family videos, retaining actual endpoints and detached observation records.

Videos are 640x480 at 20 fps, from callbacks observing the same authoritative
native episode. Neither initial hold nor sequence-start physics is replayed.
Regular observations use a 0.05 s interval and include actual terminal frames.
The isolated hand videos have 143 frames each, foot videos 213 each, and sequence
videos 286 each. HUDs distinguish actual versus reference error, q/reference and
torque/utilization, named contacts, hand reaction/capacity, foot pressure/slip,
hinge maximum/RMS speed and sustained readiness. Render success does not replace
the native contact/capture/readiness gates.

Recorded CLI forms and the final discovery command, from the authorized worktree:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python
"$PYTHON" scripts/validate_model.py
"$PYTHON" scripts/validate_contacts.py --summary --output outputs/stage5-preserved-contacts
"$PYTHON" scripts/validate_controller.py --summary --output outputs/stage5-preserved-controller
"$PYTHON" scripts/validate_transition.py --suite all --summary --output outputs/stage5-preserved-transition
MUJOCO_GL=egl "$PYTHON" scripts/validate_transfers.py --suite all --render --summary
MUJOCO_GL=egl "$PYTHON" -m unittest discover -s tests -v
```

Baseline contact/controller/transition runs used the corresponding
`stage5-baseline-*` output names before implementation. The all-case transfer
command uses both timesteps for family and perturbation cases; negatives and the
two profile diagnostics default to 2 ms. All reported measurements come from the
maintained canonical commands, not temporary experimental imports.

## Scope And Next Boundary

New maintained execution modules are `foot_transfer.py`, `hand_family.py` and
`transfers.py`, with `scripts/validate_transfers.py` for evidence export.
`contact_ik.py` extends scratch measured-reference solving; `single_hand.py`
retains Stage 4's default and supports the optional family acquisition policy
and reusable final-reference handoff. `motion_support.py` adds checked one-foot
allocation without changing default two-foot behavior. Public exports and tests
expose these contracts; validated physics/controller/readiness source remains
unchanged.

Only LEFT_FOOT is natively transfer-validated here. Scratch right-leg reference
coverage is not a right-foot transfer certification. There is no hand-foot mixed
sequence acceptance, arbitrary contact graph/layout guarantee, global feasibility
solver, optimal timing claim, long-term holding/fatigue model, or broad disturbance
robustness claim. Hands remain bounded force-only point-grasp surrogates.

A later Stage 6 should first add the other leg side, additional contact layouts,
explicit hand-foot mixed sequences and a measured geometry/controller feasibility
envelope before RL or route planning. This is a recommendation, not authorization
or implementation. **Stop at Stage 5; do not start Stage 6.**
