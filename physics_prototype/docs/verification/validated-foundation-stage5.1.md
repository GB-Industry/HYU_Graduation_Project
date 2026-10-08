# Stage 5.1 Whole-Body Foundation

## Verdict

**Native fixture acceptance: PASS at 2 ms and 1 ms.** The 16-case matrix has
eight physical successes and eight correctly classified negative outcomes.
**Whole-body visual demonstration: PASS for the declared fixtures only.**
Independent inspection of all twelve native 1 ms decoded image sheets, with six
representative 2 ms comparisons, shows discernible knee/hip extension and torso
reorientation before hand release, a real foot step with a changed support stance,
and cumulative posture change through the two-hand sequence.

**Stage 5.1: COMPLETE for the declared demonstrations.** Full discovery ran
446 tests in 5516.624 s: 445 passed, one skipped, exit 0. All 24 MP4s completed
full FFmpeg decoding with `-xerror`, exit 0. Stage 1-5 post-edit preservation
also passes. This work is recorded as a separate Stage 5.1 commit; no push or
Stage 6 implementation is included.

The historical [Stage 5 visual failure](validated-foundation-stage5-visual.md)
remains valid for its original small-motion recordings. Stage 5.1 resolves that
demonstration deficiency only on new, explicitly declared whole-body fixtures.
It does not retroactively turn the old videos into whole-body evidence, certify
general climbing naturalism, or establish pressure from rendered geometry.

## Evidence And Environment

- Native matrix and per-move physical metrics: [outputs/whole-body-stage5.1/report.json](../../outputs/whole-body-stage5.1/report.json).
- Native episodes: `outputs/whole-body-stage5.1/{right_hand,left_hand,foot,sequence}_{2ms,1ms}.json`.
- Motion audits and rendered artifacts: `outputs/whole-body-clean-{2ms,1ms}/clean_<case>_report.json`, with `<case>` in `right_hand`, `left_hand`, `left_foot`, `sequence`.
- Inspected image sheets: `outputs/whole-body-clean-1ms/audit_<case>_<view>_decoded.png` for all four cases and all three views (`rear`, `side`, `front`).
- Independently inspected 2 ms comparisons: all three views of `right_hand` and `sequence` in `outputs/whole-body-clean-2ms/`.

The native validator records the authorized project root, canonical executable,
module paths and SHA-256 hashes, validator hash and per-episode evidence hashes.
`source_hashes_stable_during_run` is `true`. The environment is the existing
`/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python`: Python
3.12.13, NumPy 2.5.3, MuJoCo 3.14.0 and Pillow 12.3.0. No new dependency,
sibling-worktree change or Stage 6 implementation is part of this task.

## Declared Motion

The source is a legal planted crouch: bilateral hip pitch 0.5 rad, knee 1.0 rad
and ankle pitch 0.5 rad. Its initial root offset is derived by planted forward
kinematics, with the fixed Stage 5 source holds and approximately 0.351 m support
surface retained. This is fixture initialization, not a root write during motion.
The knee crouch provides extension/ROM margin; it is not a ROM exception.

New hand target spacing is `0.25 * profile.arm_reach`, or 145 mm for this rig,
rather than the old 60 mm spacing. This is an explicitly changed,
morphology-derived target fixture with scratch IK, support and capacity guards,
not a claim that the original handholds stayed unchanged or an arbitrary
relocation solely for dramatic rendering. The Stage 5 foot target remains
105 mm outward and 20 mm up.

| Case | Prescribed Scratch Reference Goal | Execution |
| --- | --- | --- |
| RH | Root +40 mm z, -8 mm x, +4 degrees yaw; waist pitch 0.07 rad | Four-second preparation while source contacts remain; five-second hand reach with 20 mm normal arch |
| LH | Root +40 mm z, +8 mm x, -4 degrees yaw; waist pitch 0.07 rad | Mirrored standalone preparation/reach contract |
| LF | Root +15 mm x, +40 mm z, +4 degrees yaw; waist pitch 0.07 rad | Native unloading, separation, leg reposition, touchdown, acquisition, loading and readiness |
| RH then LH | Second cumulative root goal +70 mm z and +8 degrees yaw; waist pitch 0.10 rad | Second move starts from the first move's immutable final reference and exact live terminal state |

These root goals are **virtual frames used to generate hinge references**, not
actuated root targets. The physical root remains free and unactuated and moves
under native contacts and the existing 25 geared hinge torque motors. There is
no root stabilization force, foot weld, live root injection or visual pose fixup.
The observed root angle below is a relative-quaternion angle, not the prescribed
yaw and not a yaw-tracking certificate.

The local NumPy bounded damped-least-squares solver works on owned scratch data,
with root fixed to the prescribed virtual frame and all 25 motor hinges available.
Each hand constrains position and its outward normal, leaving tangent twist free
for the point-grasp model; each box shoe has a six-dimensional pose task, and
the waist adds three angle tasks. Source/preparation and endpoint contact
admission remain prerequisites, not substitutes for measured dynamic acceptance.
Failed local solves do not prove global physical infeasibility.

The existing rate shaping, hinge PD and feedforward realize the references.
During hand reach, arm task IK uses the **measured physical root**, not a copied
virtual-root command. No adaptive whole-body planner or full-body closed-loop
task controller is established beyond hinge PD and this arm task IK.

## Native Outcomes

All four positive cases succeed at both native timesteps. Steps and duration
below exclude the two-second initial static hold; final times use the native
episode clock. Every successful move reports sustained final physical readiness.

| Case | Steps, 2 / 1 ms | Duration (s), 2 / 1 ms | Final Time (s), 2 / 1 ms | Hand Capture Error (mm), 2 / 1 ms |
| --- | ---: | ---: | ---: | ---: |
| RH | 5554 / 11106 | 11.108 / 11.106 | 13.108 / 13.106 | 0.173537 / 0.174602 |
| LH | 5554 / 11106 | 11.108 / 11.106 | 13.108 / 13.106 | 0.173537 / 0.174602 |
| LF | 6259 / 12515 | 12.518 / 12.515 | 14.518 / 14.515 | Not a hand capture |
| Sequence | 11323 / 22643 | 22.646 / 22.643 | 24.646 / 24.643 | RH as above; second LH 0.094690 / 0.096094 |

The strict hand gates remain 1 mm position, 0.05 m/s relative speed,
30-degree orientation and less than 1 mm penetration. The existing
`endpoint_settle` acquisition policy waits for reference time at least 5.1 s
and measured nonincreasing endpoint error; it does not enlarge the capture
radius. Standalone RH capture speed is 0.000248/0.000253 m/s, penetration is
zero, and positional margin is 0.826463/0.825398 mm. Sequence LH margins are
0.905310/0.903906 mm. First eligibility and actual acquisition are distinct
recorded events, not interchangeable success claims.

| Final Readiness | Root Linear (m/s) | Root Angular (rad/s) | Max Hinge (rad/s) | RMS Hinge (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| RH, 2 ms | 0.000472 | 0.003782 | 0.007230 | 0.003056 |
| RH, 1 ms | 0.000459 | 0.003825 | 0.007276 | 0.003075 |
| LH, 2 ms | 0.000472 | 0.003782 | 0.007230 | 0.003056 |
| LH, 1 ms | 0.000459 | 0.003825 | 0.007276 | 0.003075 |
| LF, 2 ms | 0.000514 | 0.002228 | 0.001935 | 0.000695 |
| LF, 1 ms | 0.000494 | 0.002223 | 0.001955 | 0.000700 |
| Sequence final LH, 2 ms | 0.002087 | 0.001592 | 0.003739 | 0.001613 |
| Sequence final LH, 1 ms | 0.002092 | 0.001592 | 0.003745 | 0.001616 |

All windows are approximately 0.5 s sustained readiness, with floating-point
clock roundoff. Sequence move 1 has the same readiness as standalone RH, so
the table covers all ten successful moves in the eight positive episodes.

Negative fixtures are unreachable hand target, unreachable root reference,
waist ROM violation and invalid support intent, each at 2/1 ms. The first three
return `REACH_INFEASIBLE`; invalid support returns `INVALID_REQUEST`.
All eight are accepted finite negative outcomes with zero transfer steps,
no release and no adoption of a failed reference into live state. The separate
whole-body IK tests include source-obstacle collision rejection; that test is
not an additional ninth negative episode in this matrix.

## Measured Motion

Audits use every recorded actual native pose, not just rendered 20 Hz samples
or commanded `q_ref`. Distances are net displacement / maximum distance from
the case's initial pose. COM is the mass-weighted climber subtree COM. Pelvis
and root position coincide for this rig's zero local pelvis offset.

| Case | Effector Net (mm), 2 / 1 ms | Pelvis Net / Max (mm), 2 ms | Pelvis Net / Max (mm), 1 ms | COM Net / Max (mm), 2 ms | COM Net / Max (mm), 1 ms | Root Angle Net / Max (degrees), 2 / 1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RH | 145.000 / 145.000 | 40.190 / 41.690 | 40.190 / 41.682 | 36.643 / 37.380 | 36.643 / 37.380 | 3.102 / 4.574; 3.101 / 4.571 |
| LF | 109.952 / 109.948 | 41.073 / 43.436 | 41.074 / 43.437 | 30.886 / 36.522 | 30.887 / 36.522 | 4.862 / 4.994; 4.862 / 4.994 |
| Sequence | Each hand 145.000 | 68.806 / 71.436 | 68.806 / 71.426 | 63.881 / 64.684 | 63.881 / 64.685 | 9.375 / 9.408; 9.376 / 9.408 |

Standalone LH's 2 ms excursion magnitudes match RH, with mirrored lateral
motion. Measured final vertical root rise is 37.319 mm for RH and 68.427 mm for
the sequence, not exactly the prescribed 40/70 mm virtual-frame goals.

| Historical 2 ms Stage 5 Versus New 2 ms Stage 5.1 | Historical | New |
| --- | ---: | ---: |
| RH/LH effector net (mm) | 60.000 | 145.000 |
| RH/LH pelvis maximum (mm) | 11.118 | 41.690 |
| RH/LH COM maximum (mm) | 9.889 | 37.380 |
| RH/LH root maximum angle (degrees) | 1.070 | 4.574 |
| LF effector net (mm) | 109.993 | 109.952 |
| LF pelvis maximum (mm) | 12.094 | 43.436 |
| LF COM maximum (mm) | 9.706 | 36.522 |
| Sequence pelvis maximum (mm) | 15.002 | 71.436 |
| Sequence COM maximum (mm) | About 13.76 | 64.684 |
| Sequence root maximum angle (degrees) | About 1.66 | 9.408 |

The foot target did not become more distant; the larger body participation is
not explained by a larger foot step. The hand cases intentionally change both
source crouch/reference design and target spacing, so this is a fixture-level
comparison, not a controlled attribution of improvement to one parameter.

### Joint Participation

The command is a coupled 25-hinge reference solve, not a prescribed knee-only
animation. Below are measured actual peak-to-peak excursions at 2 ms, in degrees.
The commanded waist pitch endpoints are 0.07 rad (4.011 degrees) and 0.10 rad
(5.730 degrees for sequence move 2); commanded waist yaw/roll are zero. The
nonzero actual yaw/roll excursions illustrate coupled physical tracking, not
new nonzero waist commands. Hip/knee/ankle targets are derived by scratch IK
from the root/contact goals, not independently hand-selected angular amplitudes.

| Actual Joint Excursion | RH | LF | Whole Sequence |
| --- | ---: | ---: | ---: |
| Waist yaw / pitch / roll | 0.950 / 4.141 / 0.225 | 0.170 / 3.944 / 0.869 | 1.646 / 5.842 / 1.497 |
| Left hip pitch | 6.091 | 6.938 | 12.277 |
| Right hip pitch | 8.646 | 7.497 | 14.870 |
| Left / right knee | 14.459 / 13.637 | 16.553 / 13.988 | 31.133 / 28.790 |
| Left / right ankle pitch | 10.195 / 6.309 | 9.343 / 8.128 | 20.030 / 15.537 |
| Left hip roll / yaw | 2.187 / 3.790 | 9.295 / 8.942 | 2.511 / 8.956 |
| Moving RH shoulder pitch / roll / yaw | 23.304 / 24.060 / 21.916 | Not the moving limb | 23.304 / 24.060 / 28.154 |
| RH elbow / wrist | 12.805 / 25.630 | 6.068 / 8.152 | 12.805 / 25.630 |

For RH, actual final left/right knee changes are -13.734/-12.896 degrees,
and hip-pitch changes are -5.290/-8.620 degrees: genuine extension from the
crouch, not only arm motion. All 25 joint extrema, increments, actual pose
segments and recorded support-load summaries remain in the clean reports.

## Loading And Continuity

RH preparation reduces moving-hand load from about 41.26 N to 0.724 N before
`RELEASE_CLEARANCE`. The main native-load review records the accompanying
remaining-hand increase from about 41.26 N to 85.02 N, with LF/RF normal loads
changing from about 353.33/353.33 N to 364.88/346.21 N. These are native load
observations, not estimates inferred from the rendered posture.

**The moving hand is not exactly unloaded at detach.** Clearance control while
the point grasp remains active brings its last pre-detach load back to
11.461 N, within its unchanged 850 N capacity. The first post-detach observation
is inactive with zero load; detached capacity/margin are unavailable (`null`),
not zero-capacity measurements. Thus preparation illustrates substantial load
redistribution, not a zero-load-at-release claim.

LF preparation reaches 2.631 N on the moving foot while the main native-load
review records approximately 699.461 N on RF. The moving foot is then only
`FOOT_CONTACTING`, not support above the unchanged 5 N threshold. Separation
has zero source contacts/load. Acquisition requires native compression,
friction, slip and 0.1 s persistence; measured acquired target load is
13.116/13.012 N at 2/1 ms. Final LF target normal load is
341.428/341.423 N. Touchdown, acquisition, load completion and final readiness
are distinct events. Neither the `LOAD` sidebar nor rigid shoes visually prove
force or pressure magnitudes.

Both sequence audits record exact `S0 -> S1 -> S2` full-state continuity:
initial static final equals first move initial; each move final equals the next
move initial; whole-case final equals last move final. The separate core fields
`moves[0].final_integration_state` and `moves[1].initial_integration_state`
also match exactly, and the audit reports `integration_state_equal: true`.
Handoff time is 13.108/13.106 s with zero time, qpos, pelvis and COM delta.
There is no intervening root write, qvel zeroing, warmstart reset or ctrl reset.
Both native clocks have zero resets, gaps or changed poses at duplicate times;
recorded external applied-force channels are zero throughout the positive moves.

## Independent Visual Review

All twelve 1 ms sheets were opened as image attachments, not accepted merely
because files exist. The comparison set was six decoded 2 ms sheets: RH and
sequence, each rear/side/front. The main reviewer separately reports positive
inspection of all twelve 2 ms sheets; that is not attributed to this independent
inspection. No human real-time player-playback claim is made.
The clean audit's automatic `quality_verdict` remains `null`; the fixture-scoped
visual PASS above is this documented manual judgment, not an automatic verdict.

| Case | Observed At 1 ms | Scope Or Remaining Visual Issue |
| --- | --- | --- |
| RH | Side view exposes reduced crouch and higher pelvis before release; oblique views show torso reorientation and a clearly higher final hand | Release itself is subtle; rear view occludes the other arm; grasp activation is not finger closure |
| LH | Mirrored preparation, changed leg angles and higher moving-hand endpoint are visible across the three views | Some arm overlap in side view; rely on the complementary oblique/front views, not one silhouette |
| LF | Airborne leg reposition, outward/higher ledge and changed final stance are visible, especially from the front | Lateral step is foreshortened in side view; initial contact loss and progressive pressure cannot be read reliably from geometry |
| Sequence | The first changed posture is retained, then the body rises/extends further; final bilateral higher-hand stance differs from the source | Limb occlusion remains view-dependent; precise handoff and capture timing require native evidence |

No geometric snap was visible in inspected release/capture brackets and no
return to the canonical source pose was visible at handoff. This is a sampled
frame-sequence finding, not a universal no-snap proof. Largest successive native
effector steps at 2/1 ms are RH 0.108278/0.054139 mm, LF
0.138360/0.069179 mm and sequence LH 0.108872/0.054432 mm.

**Encoded-label caveat:** decoded sheets are 20 Hz brackets, not exact event
states. In the 1 ms sequence, `move2_start` labels the nearest 13.100 s encoded
frame, whose sidebar still shows move 1 settling; `move1_ready` appears at
13.150 s, already in move 2 source stabilization. The exact native boundary
is 13.106 s. These labels locate event neighborhoods; their order is not proof
of an incorrect simulation handoff or exact identical-frame equality. Use the
native JSON and separate exact-state PNGs for that claim. Front views also
explicitly hide the wall for display only; they do not certify contact by a
visible wall surface.

Both render runs report **1920x1080, 20 fps, zero physics steps and no dynamics
replay**. They copy saved actual qpos/qvel into owned display data, refreshing
only `mj_kinematics`, `mj_comPos` and `mj_camlight`. Fixed cameras share native
record indices; the essentials panel is off-body. Poses are neither interpolated
nor corrected, and render-time forces are not new physical evidence. Full
24-video FFmpeg decoding passed separately from this image review. No real-time
human playback study or automatic visual-quality certification is claimed.

## Preserved Foundation

The main verification run reports the following post-edit commands completed
sequentially with exit 0. They were not duplicated by this documentation task.

| Validator | Post-Edit Evidence | Result |
| --- | --- | --- |
| `validate_model.py` | Model integrity validation | PASS |
| `validate_contacts.py --summary --output outputs/stage5.1-preserved-contacts` | `outputs/stage5.1-preserved-contacts/report.json` | 30/30 contact cases accepted; legacy controller convergence remains false, not hidden |
| `validate_controller.py --summary --output outputs/stage5.1-preserved-controller` | `outputs/stage5.1-preserved-controller/report.json` | 86/86 physical isolated cases and four 11 s nominal/disturbance mixed runs pass |
| `validate_transition.py --suite all --summary --output outputs/stage5.1-preserved-transition` | `outputs/stage5.1-preserved-transition/report.json` | Two physical successes, four expected negatives; original first-eligible capture behavior preserved |
| `validate_transfers.py --suite all --summary --output outputs/stage5.1-preserved-transfers` | `outputs/stage5.1-preserved-transfers/report.json` | 23 accepted cases: eight family successes/ten moves, nine expected negatives, six sensitivity cases with five physical successes |

The remaining sensitivity case, `long_reach_lower_grip_2ms`, truthfully records
`CONTROL_FAILURE` (the actual report enum), because the LF sole is not touching
the intended canonical support face with real overlap. Its old fixed-geometry
endpoint cold-sole admission failure is not generalized physical infeasibility.
The shared support estimate on actual native points is confined to the optional
whole-body branch; it avoids false cold-sole rejection there without weakening
the physical gates or rewriting legacy outcomes.

Stage 4 capture errors remain 0.988638/0.999716 mm. Original Stage 5 foot
successes retain 5314/10628 steps, the `Fn > 5 N` support threshold and
`footSlip <= 0.01 m/s`. Archived baselines and original capture gates are not
retuned to make Stage 5.1 pass.

The same 78.3 kg rig, friction, grip capacity, strength ceilings, passive physics
and readiness criteria are retained. Torque remains
`Kp*(q_ref-q) + Kd*(qd_ref-qdot) + tau_ff`, clamped by geared motor capability.
No profile-specific gain or physics exception is introduced.

| Joint Class | Kp (Nm/rad) | Kd (Nm*s/rad) |
| --- | ---: | ---: |
| Waist / hip | 160 | 25 |
| Shoulder | 80 | 12 |
| Elbow | 60 | 6 |
| Wrist | 20 | 1 |
| Knee | 120 | 12 |
| Ankle | 40 | 2 |

Nominal support allocation is 90% feet / 10% hands for feedforward estimation.
The virtual root's six balance-force components are estimates only, never
applied forces; only derived hinge motor torques enter the controller. Native
foot point allocation is not a plantar-pressure controller.

## Reproduction

Run from `/home/yuchan/Desktop/project/boulder_prototype-gpt61` using the existing
canonical environment. The fixture flag is explicit so old Stage 5 evidence is
not silently interpreted as whole-body evidence.

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python scripts/validate_whole_body.py --suite all --summary --output outputs/whole-body-stage5.1
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture whole_body --input outputs/whole-body-stage5.1 --dt .002 --output outputs/whole-body-clean-2ms
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture whole_body --input outputs/whole-body-stage5.1 --dt .001 --output outputs/whole-body-clean-1ms
```

The completed full regression command is
`MUJOCO_GL=egl python -m unittest discover -s tests -v`: 446 tests in 5516.624 s,
445 passed and one skipped, exit 0. It includes scratch whole-body geometry,
atomic invalid-goal rejection, both-timestep native foot and two-hand sequence
ownership audits, and saved-evidence/rendering tests, alongside all prior tests.
The complete video-decode check was:

```bash
for video in outputs/whole-body-clean-2ms/*.mp4 outputs/whole-body-clean-1ms/*.mp4; do
  ffmpeg -v error -xerror -threads 1 -i "$video" -f null - || exit
done
```

All 24 videos decoded, exit 0. No additional expensive native run is required
merely to inspect the saved images or reports.

## Limits And Handoff

- Fixed root/reference goals are nonadaptive declared demonstrations, not a global planner, RL policy, route optimization or generalized climbing controller.
- Native forces quantify support redistribution; rigid geometry does not certify pressure, tendon physiology, anatomical finger grip or human naturalism.
- Morphology-derived target construction on this fixture does not validate a family of personalized body profiles or arbitrary target perturbations.
- Exact sequence state carryover and strict endpoint acceptance do not establish general multi-route feasibility. The old legacy route is not dynamically admitted by this result.
- Stage 6 has not started. The recommended next task is a bounded reference-feasibility envelope under unchanged physics across more morphologies and small target perturbations, with honest local-failure classification and no RL or route optimization.
