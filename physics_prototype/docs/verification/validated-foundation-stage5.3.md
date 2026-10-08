# Stage 5.3 Ascending Sequence

## Verdict

**COMPLETE for one bounded deterministic ascending fixture.** The authoritative
RH-up -> LF-up -> LH-up sequence passes for baseline and +5% limb geometry at
both 2 ms and 1 ms. Actual baseline pelvis/COM vertical gains are
88.541/83.534 mm at 2 ms. The final contacts and stance differ from the initial
state. Both handoffs retain the exact native integration state and the exact
preceding immutable reference object.

The final native bank is `outputs/ascent-stage5.3-final/`: six successful episodes
(three baseline timing trials, with the selected trial also serving as its
baseline 2 ms authority episode, plus three other authority episodes) and eight
accepted negative episodes. Its report records stable source hashes, physical
authority acceptance, and negative acceptance. Earlier exploratory banks are
not the final evidence.

Full regression: **530 tests in 7271.088 s, 529 passed, one skipped, exit 0**.
The preceding run exposed only an exact-equality test assertion on the derived
floating-point step height; that assertion was corrected and full discovery was
rerun. No mechanics or physical threshold changed for that correction.

Manual scene-only visual review passes for the declared baseline/longer fixture
at both timesteps. All twelve new 1920x1080 rear/side/front videos fully decode.
Rendering executes zero physics steps. No RL, route planning, or Stage 6 work
is included. This work is committed separately; nothing is pushed.

## Playback Audit

The pre-motion audit inspected all 24 clean Stage 5.1 videos and twelve older
Stage 5 clean videos. Stage 5.2 created no additional videos. Evidence is
`outputs/stage5.3-playback-audit/report.json`. All 9,288 encoded frames across
36 MP4s were decoded. All streams have matching 20/1 average and nominal frame
rates, strictly increasing PTS in 0.05 s intervals, and frame counts matching
the saved native selection arrays.

Representative rows apply to all three camera views:

| Recording | First / Last Native Time (s) | Native Duration (s) | Frames | FPS | Encoded Duration (s) | Encoded / Native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stage 5.1 RH, 2 ms | 2.000 / 13.108 | 11.108 | 224 | 20 | 11.200 | 1.008282 |
| Stage 5.1 LF, 2 ms | 2.000 / 14.518 | 12.518 | 252 | 20 | 12.600 | 1.006551 |
| Stage 5.1 sequence, 2 ms | 2.000 / 24.646 | 22.646 | 454 | 20 | 22.700 | 1.002385 |
| Stage 5.1 sequence, 1 ms | 2.000 / 24.643 | 22.643 | 454 | 20 | 22.700 | 1.002517 |
| Older Stage 5 RH, 2 ms | 2.000 / 9.106 | 7.106 | 144 | 20 | 7.200 | 1.013228 |

The maximum excess duration is 0.094 s, below two 20 Hz frame intervals. The
renderer samples native time, appends the exact terminal pose, and holds the last
encoded frame for one frame interval. This fencepost padding is not motion-wide
stretching. There are zero repeated rendered indices/timestamps and zero
consecutive exact decoded-RGB duplicates in the audited videos. Native endpoint
or boundary timestamp duplicates have identical stored poses. Native 500/1000 Hz
states are decimated to 20 Hz, not encoded one frame per native step.

The installed ImageIO writer receives input `-r 20.00`, without a separate
output `-r` override; output timing was independently checked with ffprobe.
**Playback is 1.0x within terminal-frame padding. Apparent slow motion comes
from conservative physical reference timing, not video retiming.**

New scene-only sequence timing is also checked by ffprobe during export:

| Profile / Timestep | First / Last Native Time (s) | Native Duration (s) | Frames | FPS | Encoded Duration (s) | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, 2 ms | 2.000 / 28.794 | 26.794 | 537 | 20 | 26.850 | 1.002090 |
| Baseline, 1 ms | 2.000 / 28.790 | 26.790 | 537 | 20 | 26.850 | 1.002240 |
| Longer, 2 ms | 2.000 / 28.772 | 26.772 | 537 | 20 | 26.850 | 1.002913 |
| Longer, 1 ms | 2.000 / 28.765 | 26.765 | 537 | 20 | 26.850 | 1.003176 |

## Fixture And Contacts

The source uses the Stage 5.2 fixed physical scene and independently recomputed
profile-specific planted crouch, followed by a two-second native static hold.
The baseline geometry has arm reach 0.58 m and leg reach 0.82 m. Longer and
shorter variants change all four limb lengths by +/-5%, retaining torso/widths,
78.3 kg total mass, ROM, motor capabilities, grip capacity, and controller gains.

The two hand targets retain the morphology-derived baseline spacing of 145 mm
above their respective source anchors. Hand holds are 20 mm radius spheres,
facing outward from the wall; canonical grasp anchors remain radius +6 mm from
their centers. Source hands are at x = +/-0.21 m, with anchor z approximately
1.448 m; target anchors are approximately 1.593 m high. Exact world frames,
profile, scene, seed, XML/model hashes, and typed contact geometry are in each
record's `fixture_inputs`.

The LF target remains 105 mm outward, retaining the nominal 5 mm initial
horizontal shoe/target separation. Its height is changed to
`0.1 * baseline.leg_reach = 0.082 m` above the source ledge. Both ledges use the
unchanged box half-extents `(0.06, 0.05, 0.03) m`; the ordinary shoe remains
`(0.040, 0.095, 0.018) m`. The sole reference retains its 11 mm site offset.
The new scene is common across the tested morphologies, not rescaled to make
individual profiles succeed.

| State | LH | RH | LF | RF |
| --- | --- | --- | --- | --- |
| S0 | left_hand | right_hand | left_foot | right_foot |
| S1 | left_hand | reach_target | left_foot | right_foot |
| S2 | left_hand | reach_target | foot_target | right_foot |
| S3 | left_reach_target | reach_target | foot_target | right_foot |

This is a declared three-move chain, not a route selected by a planner. Requests
are structurally validated before the first move. Each subsequent reference
selection uses the current native state and support configuration. It never
reinitializes or adopts a scratch pose between moves.

## Reference Candidates

All profiles use the same three candidates, evaluated deterministically on the
same unchanged live state. The first admitted candidate is selected; all three
results, including failures, are retained. There is no additional search or
physical retry after an executing move fails.

| Candidate | Vertical Rise Fraction | Root Yaw Fraction | Waist Pitch Fraction |
| --- | ---: | ---: | ---: |
| default | 0.28 | 0.30 | 0.30 |
| neutral_yaw | 0.28 | 0 | 0.30 |
| conservative | 0.18 | 0 | 0 |

These are geometry-policy coefficients, not impedance gains. They operate on
current native root/waist coordinates, target delta, support centroid, compiled
geometry and ROM. Zero yaw/pitch increment preserves the measured orientation,
not identity or a zeroed waist. Assessment retains Stage 5.2's 4 s preparation
and 50 ms sampling. Shorter execution duration is separately preflighted and
physically tested, not certified merely by a slower geometric assessment.

For both successful profiles, move 1 selects `default`. Before move 3, the two
higher-rise candidates remain locally unresolved while the lower-rise
`conservative` candidate is admitted and executes successfully. This is a real
bounded fallback, not a morphology-name special case. The executor receives
the exact previous `final_reference`, not the assessor's newly admitted scratch
source reference. Native return objects are checked with `is` before serialization,
and both full-state and integration-state equality are checked at each boundary.

The shorter profile reaches the right-shoulder-yaw bound under all three
candidates at both timesteps. Every candidate admits all 81 preparation samples
but fails the local hand path search. It has a valid native source and zero
transfer steps, returning `BOUNDED_CANDIDATE_SET_EXHAUSTED` with three
`ROM_LIMITED_SEARCH` witnesses. **Its global geometric/physical feasibility
remains unknown; no short-profile ascending success is claimed.**

## Foot Flight And Ascent

The stance-centering foot guide is computed from the current root, remaining
foot, profile leg reach, and step height. Its bounded vertical preference is
`min(0.36 * step_height, 0.04 * leg_reach)` and lateral preference is 0.12 of
the remaining-foot displacement, bounded by 0.04 leg reach. Root coordinates
are scratch-only inputs to coordinated hinge references, never actuated root
commands. Waist coordinates and root orientation are preserved from measured
state for this guide.

An 82 mm higher ledge requires a 102 mm lift: step height plus 20 mm clearance.
Holding a flat shoe throughout this flight exhausted the existing 35-degree
ankle-pitch limit. The explicit `FootRequest.airborne_pitch_rad=-0.25` introduces
a 14.3-degree toe-down flight orientation and blends back to the unchanged flat
target landing frame. The optional parameter is bounded by 30 degrees; its
default is zero, preserving the old primitive. No ROM, alignment, contact,
friction, load, or slip gate is relaxed. `TransferRequest.foot_request` exposes
the existing primitive's declared parameters with atomic structural validation.

Every move coordinates preparation/load transfer, release, hinge motion,
physical acquisition, and sustained readiness. The foot move additionally
records native separation, touchdown, >5 N support persisted for 0.1 s, two
seconds of progressive loading, and stationary-reference readiness. Touchdown
alone is not support acquisition.

## Signed Vertical Gains

The following values are world-z displacements of actual native FK/COM, not
Euclidean travel or requested root goals. Baseline at 2 ms:

| Move | Moving Limb Delta Z (mm) | Pelvis Delta Z (mm) | COM Delta Z (mm) |
| --- | ---: | ---: | ---: |
| RH up | 145.000 | 39.146 | 35.992 |
| LF up | 82.004 | 24.934 | 24.783 |
| LH up | 145.000 | 24.461 | 22.759 |
| Full sequence | Separate limb trajectories | **88.541** | **83.534** |

Baseline total pelvis displacement is 92.039 mm, distinct from its vertical
gain. Head/chest centers rise 85.211/86.399 mm. The higher foot placement and
more extended supporting leg remain in the terminal stance; the body does not
return to its source posture between moves.

| Profile / Timestep | Pelvis Net Z (mm) | COM Net Z (mm) | Duration (s) | Steps |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 2 ms | 88.541 | 83.534 | 26.794 | 13397 |
| Baseline, 1 ms | 88.541 | 83.535 | 26.790 | 26790 |
| Longer, 2 ms | 88.811 | 83.530 | 26.772 | 13386 |
| Longer, 1 ms | 88.811 | 83.530 | 26.765 | 26765 |

Longer-profile move-by-move pelvis gains are 39.155, 24.971, and 24.686 mm;
COM gains are 35.229, 25.454, and 22.847 mm at 2 ms. Its head/chest gains are
85.751/86.839 mm. Similar progress is not identical joint strategy: actual
left/right knee peak-to-peak motion is 40.353/43.481 degrees for baseline and
38.628/38.723 degrees for longer. Baseline hip-pitch excursions are
20.433/22.935 degrees, ankle-pitch excursions 18.521/24.457 degrees, and waist
pitch excursion 4.291 degrees. All physical joint ranges remain enforced.

## Timing Study

Three baseline 2 ms timing trials use independently executed native episodes:

| Trial | Hand Preparation (s) | Hand Reach Reference (s) | Foot Preparation (s) | Full Duration (s) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| conservative | 4 | 5 | 4 | 35.208 | PASS |
| moderate | 3 | 4 | 4 | 31.138 | PASS |
| fast | 2 | 3 | 4 | 26.794 | PASS |

`fast` is the fastest stable member of this declared set, then independently
passes the baseline 1 ms and longer 2/1 ms authority checks. The identical
baseline 2 ms authority job uses the already executed trial record; no recorded
trajectory is replayed into physics. Total physical sequence duration decreases
by 8.414 s, about 23.9%, without video speedup or controller gain changes.

Earlier exploratory 2/3 s foot preparations caused forbidden source recontact
after native separation. They were failures, not accepted faster foot timing.
Four-second foot preparation is therefore retained. This is a scoped hand-timing
choice, not universal speed optimization or a claim of fastest human climbing.

Baseline 2 ms selected phase durations:

| Phase | RH (s) | LF (s) | LH (s) |
| --- | ---: | ---: | ---: |
| Source stabilization | 0.502 | 0.502 | 0.502 |
| Load transfer / unload | 2.000 | 4.000 | 2.000 |
| Release clearance / lift | 0.500 | 1.500 | 0.500 |
| Three-point support | 0.500 | 0.500 | 0.500 |
| Reach / reach plus landing | 3.102 | 2.988 | 3.102 |
| Progressive target loading | Included in hand settle | 2.000 | Included in hand settle |
| Post-acquisition settle/readiness | 0.508 | 1.086 | 0.502 |
| Full move | 7.112 | 12.576 | 7.106 |

Hand release-to-capture time includes the 0.5 s three-point interval and is
3.602 s, not only the reach phase. The initial two-second static hold is excluded
from move/sequence duration. Assessment wall-clock time does not advance the
native simulation clock.

## Native Load Transfer

Baseline 2 ms preparation endpoints, not exact detach states:

| Move | First Loads | Last Loads |
| --- | --- | --- |
| RH preparation | LH/RH 40.995/41.008 N; LF/RF 353.297/353.320 N | LH/RH 85.423/0.539 N; LF/RF 364.616/346.032 N |
| LF unloading | LH/RH 40.930/43.358 N; LF/RF 352.892/352.880 N | LH/RH 75.526/96.083 N; LF/RF 4.086/695.606 N |
| LH preparation | LH/RH 44.498/45.141 N; LF/RF 345.215/360.370 N | LH/RH 0.610/85.624 N; LF/RF 337.540/365.257 N |

LF separates at native time 13.670 s with zero source contacts/load; touchdown
at 18.602 s is only 0.617 N. Sustained acquisition occurs at 18.942 s at
13.096 N, and final target normal load is 345.316 N. These are fresh native
observations, not estimated or render-time forces. Preparation strongly reduces
moving-limb load; it does not claim exactly zero hand load at detach.

Peak actuator utilization is 0.762614 across the selected authority episodes,
below the unchanged motor ceilings. Maximum foot slip is 0.009079 m/s, below
0.01 m/s. Baseline hand capture errors are 0.891576/0.898449 mm for RH and
0.142005/0.144268 mm for LH at 2/1 ms. The first faster capture has less margin
than the old slow demonstration, but remains within the unchanged 1 mm gate.
Longer RH errors are 0.178625/0.181057 mm; LH errors are
0.142079/0.144069 mm. Every final readiness window is at least 0.5 s.

The contact sequence, selected references, signed gains, force redistribution,
capture and foot-acquisition order, and final readiness agree at both timesteps.
Baseline pelvis/COM paired net-z differences are 0.000207/0.001123 mm. Longer
differences are 0.000477/0.000054 mm. Durations differ by 4/7 ms for baseline/
longer. This is qualitative consistency, not bitwise trajectory equivalence.

## Negatives

All four kinds are evaluated at 2 ms and 1 ms from valid physical sources.
Each returns bounded candidate exhaustion with zero transfer steps, no release,
no final reference, and exact unchanged native integration state.

| Case | Each Candidate's Evidence | Scope |
| --- | --- | --- |
| outside_workspace | GEOMETRY_INFEASIBLE, compiled hand-chain bound | Chosen endpoint root/waist, not every possible posture |
| support_infeasible | SUPPORT_INFEASIBLE, predicted load exceeds declared capacity | Poor-quality upward target has quality 0.04 and effective capacity 34 N; source holds retain full capacity |
| blocked_path | COLLISION_INFEASIBLE, required site point inside a collidable static obstacle and rigid hand | This prescribed sampled path, not all alternate paths |
| candidate_exhaustion | Three ROM_LIMITED_SEARCH witnesses at right_shoulder_yaw | Shorter profile remains unresolved under this set |

An earlier weak-source support fixture failed startup and was not accepted as
a valid-source transfer rejection. It remains in the non-authoritative
`outputs/ascent-stage5.3/` exploratory bank. Final evidence uses the explicit
poor target, leaving the source mechanics intact. The poor target is a declared
negative input, not relaxed positive-case capacity. No existing negative is
removed or relabeled as success.

## Visual Review

All twelve decoded event sheets were opened and inspected as image attachments,
covering baseline/longer, 2/1 ms, and rear/side/front views. The encoded videos
have no sidebar or telemetry. They show reduced crouch and body rise before
release, upward hand travel, actual airborne foot reposition onto the higher
ledge, a stable changed stance, and further retained ascent before the opposite
hand reaches upward. The final stance is visibly higher and asymmetric compared
with the source. No reset or geometric capture snap is apparent in the inspected
brackets. Complementary views expose limbs occluded in another view.

This is a fixture-scoped visual PASS, not a human real-time playback study or
anatomical-grip judgment. Geometry illustrates preparation/posture change but
cannot independently quantify force or plantar pressure. Exact handoff and
acquisition claims come from native state/evidence, not 20 Hz event labels.
Front rendering hides the wall on the display-owned model only.

Videos follow these paths, with `<profile>` = `baseline` or `longer` and
`<view>` = `rear`, `side`, or `front`:

- `outputs/ascent-clean-2ms/clean_ascending_<profile>_<view>.mp4`
- `outputs/ascent-clean-1ms/clean_ascending_<profile>_<view>.mp4`

Each export records input/code hashes, exact native indices and timestamps,
keyframe PNGs, decoded sheets, camera framing, native motion audits, and ffprobe
playback measurements. Rendering refreshes only owned kinematics/COM/cameras
from saved actual poses, with no controller, reference generation, force solve,
or physics steps. All twelve MP4s fully decoded with FFmpeg `-xerror`, exit 0.

## Preservation And Reproduction

Pre-edit checks passed from clean HEAD `233092b08a9a10541ab7431d03e5c91fc955dd89`.
Post-edit validators were then rerun under the canonical Python 3.12.13,
MuJoCo 3.14.0, NumPy 2.5.3 and Pillow 12.3.0 environment:

| Foundation | Post-Edit Evidence | Result |
| --- | --- | --- |
| Stage 1 | `outputs/physical-model-stage1/validation.json` | PASS |
| Stage 2 | `outputs/stage5.3-preserved-contacts/report.json` | 30 contact cases PASS; legacy controller verdict false as intended |
| Stage 3 | `outputs/stage5.3-preserved-controller/report.json` | 86 physical isolated cases and four static holds PASS |
| Stage 4 | `outputs/stage5.3-preserved-transition/report.json` | Two successes and four expected negatives PASS |
| Stage 5 | `outputs/stage5.3-preserved-transfers/report.json` | Eight family successes, nine negatives and six finite sensitivity cases preserved |
| Stage 5.1 | `outputs/stage5.3-preserved-whole-body/report.json` | Eight successes and eight expected negatives PASS |
| Stage 5.2 | `outputs/morphology-envelope-stage5.3-preservation/report.json` | Fresh 44-case matrix PASS with unchanged classifications and numerical outcomes |

Controller gains, strength semantics, compliance, contact friction/capacities,
capture thresholds, collision rules, ROM, readiness, force epochs and failure
ownership are not retuned. No root wrench, teleport, hidden support, ordinary
foot equality or native trajectory replay is introduced. Default generic foot
dispatch and zero airborne pitch preserve prior physical behavior.

From the authorized worktree, use a fresh evidence directory; the ascent CLI
deliberately refuses to overwrite native records:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python scripts/validate_ascent.py --suite all --output outputs/ascent-stage5.3-new
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture ascending --scene-only --dt .002 --input outputs/ascent-stage5.3-new --output outputs/ascent-clean-new-2ms
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture ascending --scene-only --dt .001 --input outputs/ascent-stage5.3-new --output outputs/ascent-clean-new-1ms
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

## Limits And RL Recommendation

- One predefined chain and small reference/timing sets, not generalized climbing or route search.
- The shorter profile remains unresolved; no extra search, ROM exception or physiology claim is used to force success.
- The foot step is bounded by current ankle/leg geometry; default flat flight and faster foot preparation failed in exploratory trials.
- The selected timing is the fastest tested stable hand-timing combination, not a global optimum or natural human speed certificate.
- Pressure, finger articulation, fatigue, dynos and anatomical grip remain outside scope.
- RL has not started. A later Stage 6 could first specify a contact-aware observation/action/reset contract and safety-constrained primitive selection on bounded fixtures. Rewards should use actual vertical progress, native contact validity and force/ROM/slip margins, with truthful failure termination. Keep the deterministic baselines and do not expose root force/teleport or reward unsupported visual motion. Broad route planning or unconstrained dynamics learning should not be implied by this foundation result.
