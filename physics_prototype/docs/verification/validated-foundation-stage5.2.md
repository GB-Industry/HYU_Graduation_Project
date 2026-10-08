# Stage 5.2 Morphology Envelope

## Verdict

**Bounded native study: PASS and complete at 2 ms and 1 ms.** The
[native report](../../outputs/morphology-envelope-stage5.2/report.json) contains
40 primary records and four counterexamples, with `acceptance: true`,
`complete_primary_matrix: true`, `partial_scope: false`, `flags: []`, and stable
source hashes. All 22 timestep pairs agree qualitatively. There are 30 physical
successes, ten locally ROM-limited searches, two fixed-frame reach-bound
rejections, and two prescribed-path collision certificates.

**Stage 5.2: COMPLETE for this bounded study.** Full discovery ran 489 tests in
6561.892 s: **488 passed, one skipped, exit 0**. It includes the original
foundation tests and the new profile-regeneration, native ownership, feasibility
classification, certificate, milestone, and interrupted-run audit tests. The
historical Stage 5.1 count is not substituted for this result. This work is
recorded as a separate Stage 5.2 commit. Nothing is pushed; Stage 6 has not started.

Acceptance of a diagnostic record is **not** physical success. In particular,
the shorter profile's geometric and physical transfer feasibility remain
unknown. This is evidence for one bounded reference policy and one fixed hand
task family, not a global climbing envelope, population study, or route planner.

## Evidence

- Aggregate report: `outputs/morphology-envelope-stage5.2/report.json`.
- Original native records: `outputs/morphology-envelope-stage5.2/<profile>_<case>_<2ms|1ms>.json`.
- Current re-audits of resumed records: matching `*.audit.json` sidecars; per-record paths and SHA-256 values are in `runs`.
- Study fixture and execution wrapper: `src/boulder_v1/morphology_envelope.py`.
- Typed, bounded assessment contract: `src/boulder_v1/transfer_feasibility.py`, consuming an explicit `TransferRequest`.
- Maintained validator: `scripts/validate_morphology_envelope.py`.
- Assessment, study, and evidence checks: `tests/test_transfer_feasibility.py`, `tests/test_morphology_envelope.py`, and `tests/test_validate_morphology_envelope.py`.
- Preserved foundation and prior visual scope: [Stage 5.1 evidence](validated-foundation-stage5.1.md) and [historical Stage 5 visual limits](validated-foundation-stage5-visual.md).

The authorized worktree is
`/home/yuchan/Desktop/project/boulder_prototype-gpt61`. Its starting foundation
HEAD is `de57f774`, reported clean before this work. At documentation review,
all 71 tracked files under `src`, `scripts`, and `tests` were compared bytewise
with HEAD and matched. Stage 5.2 adds modules, a validator, and tests rather than
modifying those foundation files. No anticipated final commit hash is used.

The report records the existing canonical executable
`/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python`, Python 3.12.13,
NumPy 2.5.3, MuJoCo 3.14.0, and Pillow 12.3.0. Using that runtime does not imply
editing the sibling checkout. No new dependency or optimizer framework is
introduced; bounded numerical work uses the existing NumPy and MuJoCo stack.

## Controlled Design

The primary matrix is **four profiles x five targets x two timesteps = 40**.
The task is a right-hand transfer with a nominal 145 mm vertical displacement.
The baseline Stage 5.1 source holds remain fixed in world coordinates. Target
positions are also fixed across profiles for a given case, rather than scaled
to make each climber succeed. The scene SHA agrees across all profiles in each
case; compiled geometry and independently generated source seeds differ when
limb lengths differ.

| Profile | Upper Arm / Forearm (m) | Thigh / Shin (m) | Grip Capacity (N) | Controlled Change |
| --- | --- | --- | ---: | --- |
| `baseline` | 0.3100 / 0.2700 | 0.4200 / 0.4000 | 850 | Reference geometry |
| `shorter` | 0.2945 / 0.2565 | 0.3990 / 0.3800 | 850 | All four limb lengths -5% |
| `longer` | 0.3255 / 0.2835 | 0.4410 / 0.4200 | 850 | All four limb lengths +5% |
| `reduced_grip` | 0.3100 / 0.2700 | 0.4200 / 0.4000 | 425 | Baseline geometry, capacity halved |

Torso length 0.55 m, shoulder width 0.42 m, hip width 0.30 m, total climber mass
78.3 kg, and mass/ROM/strength/power scales of 1 are held fixed. Compiled joint
ranges, axes, and motor gears retain the foundation definitions. Limb-length
changes alter body geometry, placement, and the compiled inertia mechanism,
not controller gains or total mass. These labels are controlled model proxies,
not measured anthropometric or physiological populations. The 425 N profile
is a bounded capability probe, not an extreme anatomical grip model.

| Target Case | Local Lateral Offset (mm) | Local Vertical Offset (mm) |
| --- | ---: | ---: |
| `nominal` | 0 | 0 |
| `up` | 0 | +10 |
| `down` | 0 | -10 |
| `lateral_minus` | -10 | 0 |
| `lateral_plus` | +10 | 0 |

Offsets use the actual canonical target-frame basis. Their common magnitude
is `min(0.02 * baseline.arm_reach, 0.5 * target.radius)`:
`min(0.0116, 0.0100) = 0.0100 m`, with arm reach 0.58 m and radius 0.02 m.
The perturbations do not change target normal or orientation. Two additional
baseline counterexamples, each at both timesteps, bring the total to 44.

### Source Generation

Each profile's source is recomputed from compiled `qpos0`, not copied from an
old trajectory or selected by profile name. The universal preference bends
each knee to 40% of its own compiled ROM, sets hip and ankle pitch to half that
bend for a flat sole, and starts elbows at their compiled ROM midpoint.
Planted-foot forward kinematics derives the initial root shift. Arm IK then
solves to the actual canonical source anchors. The same profile/case source
construction is repeated independently, without recorded trajectory replay.

The study fixture declares physical task coordinates using the baseline
Stage 5.1 scene. The assessment itself consumes model/scene geometry, the actual
native state, reference, manager, support map, moving limb, source, and target
through typed arguments. It has no Stage 5.1 hold-ID or profile-name pose
special cases. Label-renaming checks exercise this distinction.

Explicit pose initialization occurs once at episode start. A two-second native
static hold must produce a valid, sustained physical source before assessment.
All 44 records have valid sources. Failed candidates have zero transfer steps,
no release, no final reference, and unchanged full integration state; the
initial source hold is not counted as a transfer execution.

## Assessment Contract

`assess_hand_transfer` generates a virtual candidate from the **actual** native
root, waist, effector frames, target delta, and remaining-support centroid.
With nonnegative vertical target delta `dz`, its universal geometry policy uses:

- Root rise `min(0.28 * dz, 0.06 * leg_reach)`.
- Lateral shift from 0.1 of the source support-centroid offset plus 0.1 of target lateral delta, bounded by `+/-0.06 * leg_reach`.
- Handed relative root yaw and waist-pitch increment based on `0.3 * atan2(dz, arm_reach)`, with root yaw capped at 10 degrees.
- Four-second preparation, 50 ms assessment sampling, and the existing five-second hand reach.

Root yaw is relative to the measured orientation about world up. Virtual root
coordinates exist only in owned scratch data to construct hinge references;
they are neither motor targets nor root wrenches. The bounded solve makes all
25 actuated hinges available, uses full six-dimensional shoe tasks, hand
position/normal tasks without a fictitious point-grasp moment, and three waist
angle tasks. Compiled ROM bounds remain enforced. Projected or failed scratch
solutions are diagnostic evidence, never adopted into live state.

The measured source is freshly solved and strictly admitted as a source
reference, without adopting that scratch pose into live data. Preparation has
81 samples, each requiring exact static admission. Clearance/reach geometry
is sampled at 50 ms with support estimates and endpoint admission. These checks
are **not continuous-path or dynamic certificates**. The nominal 90% feet / 10%
hands allocation and four-to-three-contact load plan are estimated reactions.
They supply derived hinge feedforward torques, never applied support forces.

`feasible: true`, `GEOMETRICALLY_FEASIBLE`, and `support_feasible: true` mean
this sampled candidate passed the declared geometric/admission/static-estimate
checks. They do not establish physical capture or readiness. Only the stock
`execute_transfer` cycle can establish `PHYSICAL_SUCCESS`. Its live hand IK
uses the measured root; the existing impedance/feedforward law and rate shaper
retain 0.5 rad/s and 2 rad/s^2 bounds, with no gain retuning or root teleport.

The strict hand gates remain 1 mm position, 0.05 m/s relative speed, 30-degree
orientation, and less than 1 mm penetration. `endpoint_settle` waits for the
existing endpoint policy; it does not enlarge the capture gate. Physical foot
support remains native compression above 5 N with slip at most 0.01 m/s.
Final physical readiness must be sustained for at least 0.5 s.

## Native Outcomes

Each cell below gives the classification at **both 2 ms and 1 ms**.

| Profile | Nominal | Up | Down | Lateral Minus | Lateral Plus |
| --- | --- | --- | --- | --- | --- |
| `baseline` | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS |
| `shorter` | ROM_LIMITED_SEARCH | ROM_LIMITED_SEARCH | ROM_LIMITED_SEARCH | ROM_LIMITED_SEARCH | ROM_LIMITED_SEARCH |
| `longer` | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS |
| `reduced_grip` | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS | PHYSICAL_SUCCESS |

Thus 30/40 primary records are physical successes, spanning two distinct
successful limb geometries. The ten shorter-profile records are accepted
diagnostics, not successes or demonstrated physical impossibilities. All 81
preparation samples are admitted in every shorter case. The subsequent local
clearance/reach solve stalls at the right-shoulder-yaw bound after 82-89 recorded
path solves, depending on target. `geometric_feasible` is `null`, not `false`.
No failed `qpos` is adopted and no native transfer is attempted. This proves
only that this bounded policy/search did not resolve a legal path.

| Nominal Profile | Steps, 2 / 1 ms | Transfer Duration (s), 2 / 1 ms | Final Native Time (s), 2 / 1 ms | Capture Error (mm), 2 / 1 ms | Peak Motor Utilization, 2 / 1 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 5556 / 11111 | 11.112 / 11.111 | 13.112 / 13.111 | 0.332370 / 0.332354 | 0.468415 / 0.468517 |
| `longer` | 5553 / 11103 | 11.106 / 11.103 | 13.106 / 13.103 | 0.083695 / 0.085072 | 0.468011 / 0.468077 |
| `reduced_grip` | 5556 / 11111 | 11.112 / 11.111 | 13.112 / 13.111 | 0.332370 / 0.332354 | 0.468415 / 0.468517 |

Transfer steps/durations exclude the two-second source static hold. Nominal
capture occurs at native time 12.604 / 12.602 s, distinct from final readiness.
Across all 30 successes, capture error ranges from 0.082401 to 0.911419 mm;
the maximum is the baseline/reduced-grip `up` case at 2 ms and still lies inside
the unchanged 1 mm gate. Peak motor utilization across successes is 0.471593.

## Measured Response

Metrics below use recorded **actual native poses**, with scratch FK and
mass-weighted climber-subtree COM, not desired poses or render samples. Net
displacement is relative to each episode's own initial state; maximum means
maximum distance from that initial state. Pelvis and root positions coincide
for this rig. Relative root angle is not a yaw-tracking certificate.

| Nominal Profile | Pelvis Net / Max (mm), 2 ms | COM Net / Max (mm), 2 ms | Pelvis Net (mm), 1 ms | COM Net (mm), 1 ms | Root Net Angle (degrees), 2 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 39.9045 / 41.9554 | 36.9776 / 38.2289 | 39.9056 | 36.9789 | 3.2595 |
| `longer` | 40.4389 / 41.9922 | 36.9367 / 37.2599 | 40.4394 | 36.9364 | 3.1206 |
| `reduced_grip` | 39.9045 / 41.9554 | 36.9776 / 38.2289 | 39.9056 | 36.9789 | 3.2595 |

The longer source root is about 35.440 mm higher because planted leg geometry
is different: initial root heights are 1.155953 / 1.191394 m for baseline/longer
at 2 ms. Their final COM separation of 27.4311 mm is an **absolute pose
separation**, not an extra 27 mm of COM movement. Similar net displacement
does not mean the same body posture or torque strategy.

| Actual Nominal Joint Metric (degrees), 2 ms | Baseline | Longer |
| --- | ---: | ---: |
| Left elbow initial / final | 98.971 / 85.485 | 100.292 / 87.802 |
| Right elbow initial / final | 98.971 / 100.261 | 100.292 / 104.120 |
| Left / right elbow peak-to-peak | 13.486 / 12.238 | 12.819 / 13.318 |
| Waist yaw / pitch / roll peak-to-peak | 0.938 / 4.299 / 0.214 | 0.873 / 4.102 / 0.446 |
| Left / right hip pitch peak-to-peak | 5.760 / 8.346 | 5.439 / 7.900 |
| Left / right knee peak-to-peak | 13.857 / 13.022 | 13.068 / 12.287 |

Independently assessed preparation/reference paths differ for changed limb
geometry, including the shorter profile's admitted preparation before its
local failure. Desired poses and actual tracking are distinct; measured-root
arm IK and physical drift couple their evolution under unchanged gains.
For each successful baseline/longer comparison, the report measures maximum
final elbow-angle difference and maximum absolute final commanded-hinge-torque
difference, not an all-time trajectory maximum:

| Same Target, 2 ms | Final Elbow Difference (degrees) | Final COM Separation (mm) | Final Torque Difference (Nm) |
| --- | ---: | ---: | ---: |
| Nominal | 3.8597 | 27.4311 | 3.5152 |
| Up | 4.0280 | 26.2790 | 3.7938 |
| Down | 3.6110 | 27.6695 | 3.8155 |
| Lateral minus | 3.8144 | 27.7171 | 3.8790 |
| Lateral plus | 3.8375 | 26.3987 | 3.7371 |

All ten baseline/longer target/timestep comparisons have differing preparation
paths and observable responses. The report also contains ten longer/reduced-grip
comparisons; those repeat the baseline-geometry contrast, not a third successful
geometry. The measured differences exceed the declared 0.5-degree elbow or
1 Nm torque response thresholds, independently of absolute source-height shifts.

Target perturbations also alter actual body participation. At 2 ms, baseline
`down` / `nominal` / `up` pelvis net displacement is 37.2668 / 39.9045 / 42.5177 mm,
with final waist pitch 3.7195 / 4.0552 / 4.3559 degrees. Actual root relative
angles are 3.0488 / 3.2595 / 3.4824 degrees. The two lateral cases give pelvis net
40.5167 / 39.6254 mm and right-hip-pitch excursions 8.3895 / 8.3151 degrees.
These are measured responses to geometry, not merely a count of positive cases.

### Native Loading

| Nominal Native Load (N), 2 ms | Baseline | Longer |
| --- | ---: | ---: |
| Preparation first LH / RH load | 41.478 / 41.479 | 41.506 / 41.507 |
| Preparation last LH / RH load | 84.919 / 0.521 | 84.943 / 0.467 |
| Preparation first LF / RF normal force | 353.017 / 353.020 | 353.018 / 353.020 |
| Preparation last LF / RF normal force | 364.315 / 346.326 | 364.197 / 346.293 |
| Whole-transfer observation-mean LH / RH load | 72.064 / 11.588 | 72.120 / 11.563 |
| Whole-transfer observation-mean LF / RF normal force | 361.387 / 348.303 | 361.325 / 348.221 |

Preparation endpoints above are recorded near native times 2.504 and 6.502 s,
not exact detach states. They demonstrate real redistribution from moving hand
to remaining supports. They do not prove zero moving-hand load at detach.
Observation means are not phase-restricted or time-integrated force estimates;
moving-hand means include detached zero-load observations. Detached capacity
and margin remain unavailable, not zero. Native load changes between baseline
and longer are small here; no invented 10 N morphology-specific difference is
needed to substantiate the separately measured posture/torque response.

The 425 N case has exactly the baseline geometry, root/joint paths, native
forces, and capture behavior for each paired case. Capacity is halved and active
grip margins decrease by 425 N. Nominal 2 ms minimum remaining-hand margin is
763.141 N at 850 N capacity and 338.141 N at 425 N; the smallest active-hand
margin anywhere in the reduced-capacity successes is 337.802 N. This limit is
inactive, so unchanged physical strategy is the correct inactive-capability
response, not evidence of fake adaptation or a need to retune gains. It does
not certify adaptation at a capacity boundary.

### Timestep Agreement

All 22 `comparisons.timestep_pairs` have `consistent: true`. Maximum absolute
1 ms versus 2 ms differences in the report are:

| Metric | Maximum Absolute Difference |
| --- | ---: |
| Pelvis net displacement | 0.003690 mm |
| COM net displacement | 0.002281 mm |
| Root net relative angle | 0.001166 degrees |
| Successful capture error, computed from paired report metrics | 0.001377 mm |

This is qualitative agreement and small differences in these measured summary
metrics, not bitwise trajectory equality or a continuous-time convergence proof.

## Counterexample Scope

| Counterexample | Records | Numeric Witness At 2 ms | Meaning |
| --- | ---: | --- | --- |
| `beyond_reach` | 2 | Compiled chain upper bound 0.620000 m; shoulder-target distance 0.913319 m; excess 0.293319 m | `GEOMETRY_INFEASIBLE` for this fixed policy endpoint root/waist |
| `blocked_path` | 2 | Reach sample time 2.25 s; hand interior margin 0.006000 m; obstacle interior/shared-ball radius 0.001497 m | `COLLISION_INFEASIBLE` for the prescribed sampled effector path |

All four certificates are verified by the current validator. They start from
valid physical sources and have zero transfer steps, no release, and no failed
reference adoption.

`beyond_reach` shifts the baseline target laterally by
`1.5 * profile.arm_reach = 0.87 m`. The triangle-inequality bound uses the actual
compiled hand chain, including the end-site offset, with a 1 micrometer excess
tolerance. It is a position-only upper bound at the chosen endpoint root/waist,
not a proof that this climber cannot reach after another root motion or policy.

`blocked_path` adds an explicit collidable STEP box at the source/target midpoint
plus 20 mm along the outward normal, with half-sizes `(0.015, 0.018, 0.015) m`.
Source holds/root geometry remain valid. After 81 admitted preparation samples,
the prescribed hand END site at reach sample index 45 lies strictly inside both
its own rigid hand box and the static obstacle, with margins above 1 mm. That
common interior certifies solid intersection independently of arm twist. The
certificate conservatively checks compiled collision masks, exclusions, native
pair rules, rigid ownership, and static geometry; no reach IK failure is used as
the proof. No reach IK is run after this witness. Alternate paths remain unknown.

`ROM_INFEASIBLE` is separately available for an actual or hard policy goal
outside compiled ROM and is tested as atomic rejection. It is not the shorter
profile's `ROM_LIMITED_SEARCH`. Likewise, numeric support-rejection diagnostics
are tested with deliberately weak parameters outside the primary matrix,
including a negative remaining-hand capacity margin and finite motor margin.
Those branch checks are not additional authoritative envelope episodes.

The primary matrix produces **no `SUPPORT_INFEASIBLE` or
`DYNAMIC_CONTACT_INFEASIBLE` outcomes**; every admitted native transfer passes.
The implementation and evidence tests distinguish genuine capture, contact,
capacity, and readiness failures from malformed evidence or unresolved local
search, but this study does not empirically cover every physical failure mode
or map their boundaries. Small perturbations are not a pressure sweep.

## Integrity And Resume

The initial long matrix run produced 35 native records before the user stopped
it. Evidence review found the first certifier insufficiently strict; validator
and test corrections preceded final classification. The native generation and
controller code did not change for those resumed records. Strict `--resume`
re-audited all 35 originals without rewriting them and executed the nine missing
cases once. Original generation provenance remains separate from current audit
provenance; differing checker hashes are not represented as a new physics run.

The report records original evidence SHA-256 values, environment and native-code
hashes, profile/scene inputs, compiled-model/geometry hashes, and current audit
sidecars. Re-audit uses saved native states and owned FK/COM reconstruction,
not dynamics replay, source replay, new capture events, or manufactured native
measurements. Resume preflights all existing records before any missing episode;
stale, wrong-source, or malformed artifacts block the batch. Hash/consistency
checks are not cryptographic authentication of an unmanifested original record.

The finalized physical audit requires a real native capture record and unchanged
strict gate, fresh post-activation capacity decisions for every hand, bounded
native support loads, correct final contact intent/reference, and sustained
terminal readiness. It cross-checks all 25 controls, complete native step/clock
accounting, terminal observation versus final state, and saved full integration
states. Initial static final state equals transfer initial state exactly.
Positive records have zero recorded `qfrc_applied` and external body-force
channels. There is no root stabilization, body glue, foot equality support,
intermediate teleport, replay reset, or hidden force injection. Native bounded
hand point-grasp constraints remain the foundation contact model.

## Foundation Preservation

The main verification workflow completed the following preservation checks;
this documentation task did not duplicate their expensive native runs.

| Stage | Evidence | Recorded Outcome |
| --- | --- | --- |
| 1 | Pre-edit `validate_model.py` run | Model integrity PASS, reported by main verification owner |
| 2 | [Contact baseline](../../outputs/stage5.2-baseline-contacts/report.json) | 30 contact cases accepted; historical controller convergence remains `false` |
| 3 | [Controller baseline](../../outputs/stage5.2-baseline-controller/report.json) | 86 isolated physical cases and four mixed static runs pass |
| 4 | [Transition baseline](../../outputs/stage5.2-baseline-transition/report.json) | Two physical successes and four expected negatives accepted |
| 5 | [Transfer baseline](../../outputs/stage5.2-baseline-transfers/report.json) | 23 accepted cases: eight family successes/ten moves, nine negatives, six sensitivity cases with five physical successes |
| 5.1 | [Whole-body baseline](../../outputs/stage5.2-baseline-whole-body/report.json) | 16 accepted cases: eight physical successes and eight expected negatives at both timesteps |

The remaining old Stage 5 sensitivity case, `long_reach_lower_grip_2ms`, retains
its truthful `CONTROL_FAILURE`; it is not substituted for a Stage 5.2 profile
or weakened into success. Stage 2's contact-only acceptance and false legacy
controller convergence also remain distinct. Foundation foot/capture/friction,
physical impedance units, strength ceilings, and readiness gates are unchanged.

The combined preservation command initially reached its 3,600,000 ms timeout
before completing Stage 5.1. That timed-out run is not claimed to have passed.
The full Stage 5.1 validator was separately rerun with a 7,200,000 ms budget and
produced the accepted 16-case report linked above. The subsequent full test
discovery is a separate completed result: 489 tests in 6561.892 s, 488 passed,
one skipped, exit 0. All tracked foundation source, script, and test files remain
unchanged; the new assessment and study are opt-in additions.

## Reproduction

From the authorized worktree, using the canonical environment:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python scripts/validate_morphology_envelope.py --suite all --summary --output outputs/morphology-envelope-stage5.2
```

That command performs the fresh full 1 ms/2 ms study. After interruption, add
`--resume` to re-audit matching saved records and execute only missing cases.
Filtered `--profile`, `--case`, `--dt`, or suite invocations certify only their
selected scope; they are not replacements for complete-matrix evidence.
The completed full-regression command is
`MUJOCO_GL=egl python -m unittest discover -s tests -v`. Large native artifacts
under `outputs` remain offline/ignored and are not committed.

## Limits And Next Step

- This is one right-hand task family on a fixed vertical-hold fixture, with +/-5% limb-length proxies and +/-10 mm vertical/lateral offsets, not multi-route or population generalization.
- No target-normal/orientation perturbation, anatomical finger grip, tendon model, fatigue model, or pressure inference is certified. Capability is the bounded point-grasp proxy.
- Foot morphology is not certified. The explicit new assessment contract is hand-only; the prior Stage 5.1 native foot primitive is preserved, not expanded into a new morphology-aware foot policy.
- Static sampled admission is not continuous-path safety or dynamic readiness. Failed local search does not prove global ROM, geometric, or physical impossibility.
- Reduced grip remains inactive in this study; success does not locate a capability boundary or demonstrate boundary-triggered strategy adaptation.
- No new Stage 5.2 visual PASS is claimed: `visual_quality_verdict` is `null`. Prior fixture-scoped Stage 5.1 visual evidence and historical Stage 5 limitations remain unchanged.
- Completion covers the bounded study and full regression, not general physical feasibility for rejected or untested strategies. No new visual-quality judgment, RL, route planning, or Stage 6 implementation is included.

**Recommendation for Stage 6, later only:** evaluate a
small finite set of alternate bounded reference candidates to better resolve
the locally ROM-limited cases. Add small normal/orientation perturbations and
more bounded profiles or capacity-boundary probes under the same physical
gates. Preserve unknown/local-failure semantics and native evidence ownership.
This is a recommendation, not a started implementation; it does not authorize
RL, route planning, or anatomical-model expansion.
