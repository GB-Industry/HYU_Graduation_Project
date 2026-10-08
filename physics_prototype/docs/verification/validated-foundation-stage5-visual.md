# Stage 5 Visual And Motion-Quality Audit

## Verdict

**Stage 5 visual evidence: FAIL as a convincing whole-body climbing-motion
demonstration.** **Clean layout and multi-camera visibility: PASS.** Recorded
Stage 5 numerical/physics success remains preserved.

Classification for the hand transfers and RH -> LH sequence:

**PHYSICS SUCCESS / MOTION DEMONSTRATION INADEQUATE.**

The new videos expose real hand displacement and a clear local foot step, but
they do not reveal a substantial whole-body climbing progression. The hand
targets are only 60 mm apart, pelvis/COM motion is millimetre-scale, and most
support-body posture stays nearly fixed. This limitation is not hidden behind
telemetry, reframing, slow motion, hold relocation or new controller behavior.

No Stage 6 work was started. No physics, gains, references, capture/contact rules,
readiness criteria, or transfer semantics were changed. No dynamics were rerun.

## Rendering Method

The source is the already verified canonical 2 ms Stage 5 native recordings at
`outputs/transfers-stage5/{right_hand,left_hand,foot,sequence}_2ms.json`.
`scripts/render_clean_transfers.py` verifies the complete six-module recorded
source-hash set and expected module paths before reconstructing the matching
fixture. Input file SHA-256 values are retained in the new reports.

The renderer copies recorded actual qpos/qvel into its own display data and runs
only `mj_kinematics`, `mj_comPos` and `mj_camlight`. It never calls `mj_step`, runs
a controller, regenerates a reach, or forwards a new constraint solve as physical
force evidence. All three cameras share exactly the same native record indices
and recorded times. Body poses are not interpolated or corrected. This is
rendering recorded physics, not a separate simulation for each view.

Output is **1920x1080 at 20 fps**. The unobstructed scene occupies pixels
`[0,1440) x [0,1080)`; a separate 480-pixel right sidebar contains only view,
move number, moving limb, source -> target, phase, actual contacts and outcome,
plus recorded time. Detailed forces, speeds and excursions remain in JSON.
Tests verify that sidebar composition does not modify any scene pixel.

Cameras are fixed for each complete trajectory. Their lookat and distance are
fit to the union of the climber's actual visible geometry over every native pose,
with a bounding-sphere margin factor of 1.12. The wall and far diagnostic hold
do not determine actor framing. The front view is a disclosed cutaway: wall
alpha is hidden only on a display-owned model, restored for the other views.
Collision geometry, masks, inertia and physical contacts are unaffected.
Sites are hidden for display so colored site spheres no longer cover the real
palms and shoe boxes. Holds and actual body/shoe geometry remain visible.

| Case | Rear Azimuth | Side Azimuth | Front Azimuth |
| --- | ---: | ---: | ---: |
| Right hand | 135 degrees | 180 degrees | 225 degrees |
| Left hand | 45 degrees | 0 degrees | 315 degrees |
| Left foot | 45 degrees | 0 degrees | 315 degrees |
| RH -> LH sequence | 135 degrees | 0 degrees | 315 degrees |

All use elevation **-5 degrees** and vertical FOV **45 degrees**. Distances are
approximately 2.62 m; exact per-case lookat/distance values are in each
`clean_<case>_report.json`. For RH, lookat is approximately
`(0.000856238, -0.339964313, 1.152658891)` m and distance is
`2.615604386` m. The cameras do not follow pelvis/COM motion and conceal it.

## Inspection Performed

Inspection used decoded event sequences from all twelve final MP4 views, plus
full-resolution native keyframes around release, reach, capture, foot landing
and sequence handoff. All twelve videos were also fully decoded by FFmpeg with
exit 0 and no errors. This is frame-sequence inspection of the encoded videos,
not a claim of a human real-time playback study or an automated motion-quality
PASS based only on file generation.

Decoded before/after sheets use directional encoded-frame brackets. For example,
RH capture-before uses recorded 8.600 s and capture-after uses 8.650 s; a pre-event
image is not labeled as after capture. Exact native PNGs remain separate. The
native-to-decoded index/time mapping and timing caveat are stored in the reports.
MP4 duration is nominal frame count / 20 fps. Inclusion of the initial frame and
exact final frame adds less than two frame intervals to the native state-time
span; sidebar timestamps remain actual recorded physics time. For example, the
sequence's state-time span is 14.212 s and its encoded duration is 14.300 s.

## Hand Findings

RH and LH give the same basic result; side and front views expose the moving hand
better than a view naturally occluded by the torso.

| Requested Visual Feature | Finding |
| --- | --- |
| Clear source release | Subtle at the release instant. Subsequent withdrawal/separation is visible in side/front keyframes; the equality event itself is not an anatomical finger-opening animation. |
| Hand displacement | Visible 60 mm local upward transfer. More legible at HD, but small relative to the full climber. |
| Whole-body weight shift | Not clearly evident as a deliberate climbing weight shift. Pelvis moves only 5.43 mm net / 11.12 mm maximum; COM 5.25 mm net / 9.89 mm maximum. |
| Reach toward target | Visible forearm/wrist motion and transfer between nearby holds. |
| Capture without snap | No visible geometric snap in the inspected brackets. Native motion increments are small; no pose is corrected by the renderer. |
| Stable changed final pose | The hand ends at the upper hold and the changed arm pose is stable, but the overall standing posture barely changes. |

The result is a usable view of a local contact-transfer primitive, not adequate
evidence of substantial whole-body climbing movement. The improved camera/HUD
does not resolve that motion-generation limitation.

## Foot Findings

The local left-foot transfer is visually more meaningful than the hand transfers.
The moving leg is visible in the left side, front and left rear three-quarter
views rather than obscured by a large telemetry block.

| Requested Visual Feature | Finding |
| --- | --- |
| Source separation | Visible after lifting. The initial microscopic loss of contact is not itself large visual motion, but the subsequent airborne gap is clear. |
| Hip/knee/ankle motion | Clearly visible leg bend/reposition. Peak-to-peak hip pitch is 12.30 degrees, knee 28.06 degrees, ankle pitch 15.26 degrees. |
| Flight/reposition | Clear local step to the adjacent ledge; actual net foot displacement is 109.99 mm. |
| Touchdown | Visually identifiable at the new ledge without a snap. |
| Progressive loading | Not independently obvious from rigid geometry. The native load trace establishes the progression; labeling LOAD is not visual proof of force magnitude. |
| Changed pelvis/support configuration | Changed foot placement and leg/support stance are clear. Pelvis displacement remains small: 10.58 mm net / 12.09 mm maximum. |

The foot clip passes local leg/foot-motion visibility, but does not demonstrate a
large coordinated pelvis/COM progression. Loading rises from about 0.90 N at
touchdown to >5 N, sustained acquisition and approximately 346.91 N at the final
endpoint. Those are original native JSON measurements, not new render-time
constraint-force solves. The large load redistribution is not accompanied by a
similarly large visible body shift.

## Sequence Findings

Move 2 starts from move 1's exact complete terminal state. The boundary has zero
qpos, pelvis and COM delta, and exact equality of serialized full states. The
separate `move1_ready` and `move2_start` PNGs show the same body pose at 9.106 s;
only the move/phase sidebar changes. No reset or canonical-pose replacement is
observed or inserted.

The final pose retains both hands on their higher holds. This cumulative arm/contact
change is visible, particularly from the front and side, but remains restrained.
Across the whole sequence, pelvis net motion is 5.95 mm and maximum deviation
15.00 mm; COM net motion is 6.21 mm and maximum deviation 13.76 mm. Root rotation
is only 1.66 degrees. The sequence still reads primarily as two small arm transfers
from a nearly fixed standing support pose, not a clear full-body climbing passage.

Classification: **PHYSICS SUCCESS / MOTION DEMONSTRATION INADEQUATE.**

## Measured Motion

Measurements use every recorded native pose, not just 20 Hz display samples or
q_ref. Distances are net displacement / maximum distance from that transfer's
initial state. Root-angle changes use relative unit quaternions, not quaternion
component subtraction. COM is the mass-weighted `climber_root` subtree COM.

| Transfer | Moving Effector (mm) | Pelvis (mm) | COM (mm) | Root Angle Net / Max (degrees) |
| --- | ---: | ---: | ---: | ---: |
| RH | 60.000 / 60.074 | 5.428 / 11.118 | 5.247 / 9.889 | 1.067 / 1.070 |
| LH | 60.000 / 60.074 | 5.428 / 11.118 | 5.247 / 9.889 | 1.067 / 1.070 |
| LF | 109.993 / 109.993 | 10.584 / 12.094 | 9.174 / 9.706 | 2.289 / 2.289 |
| Sequence move 1, RH | 60.000 / 60.074 | 5.428 / 11.118 | 5.247 / 9.889 | 1.067 / 1.070 |
| Sequence move 2, LH | 60.000 / 60.072 | 5.085 / 10.062 | 5.168 / 9.012 | 1.047 / 1.049 |

Relevant actual hinge peak-to-peak excursions:

| Joint Group | Excursions (degrees) |
| --- | --- |
| RH shoulder pitch / roll / yaw | 8.444 / 2.029 / 2.426 |
| RH elbow / wrist | 10.324 / 11.798 |
| LH arm | Mirrored standalone excursion magnitudes; largest wrist excursion 11.798 |
| LF hip pitch / roll / yaw | 12.300 / 9.587 / 2.982 |
| LF knee / ankle pitch / ankle roll | 28.064 / 15.260 / 8.390 |
| Sequence second LH wrist | 11.920 |

All 25 hinge excursion and increment records, XYZ position curves' extrema, net
vectors, path lengths, support-load extrema and event-step brackets are in the
four `clean_<case>_report.json` files.

## Snap And Reset Findings

No visible capture snap or sequence reset was found in the inspected encoded
brackets/keyframes. Recorded native clocks have zero resets/gaps, and duplicate
timestamps have no changed pose. The sequence handoff states match exactly.

Largest successive native effector steps are approximately 0.076706 mm for
standalone hands/sequence RH, 0.139032 mm for LF and 0.079084 mm for sequence LH.
Largest corresponding hinge increments are about 0.039534, 0.044927 and
0.039482 degrees. These are measurements of the saved native trajectory, not
an arbitrary new physical acceptance threshold or a universal no-snap proof.

## Cause Of The Inadequate Demonstration

The original camera/HUD issue is corrected. The remaining deficiency is in the
demonstrated motion itself:

- Hand target geometry is deliberately nearby: 60 mm vertical spacing, while the
  feet stay on the same ledges and overall body elevation barely changes.
- Hand reference generation is a conservative local five-joint arm objective;
  the supporting posture is largely held rather than assigned a new substantial
  whole-body/COM-repositioning objective.
- The controller realizes these small references accurately. This audit does
  not show that its gains are inadequate or that capture anchors are wrong.
- The model's rigid shoes and force-only hand surrogate do not visually encode
  pressure buildup or anatomical finger grasp action. Native JSON, not geometry
  alone, proves loading and equality activation.

No holds were moved to make the video dramatic, and no body motion was added.
Work stops at this disclosed limitation rather than proceeding to Stage 6.

## Outputs And Reproduction

All outputs are under
`/home/yuchan/Desktop/project/boulder_prototype-gpt61/outputs/stage5-clean/`.

| Case | Rear Video | Side Video | Front Video |
| --- | --- | --- | --- |
| RH | `clean_right_hand_rear.mp4` | `clean_right_hand_side.mp4` | `clean_right_hand_front.mp4` |
| LH | `clean_left_hand_rear.mp4` | `clean_left_hand_side.mp4` | `clean_left_hand_front.mp4` |
| LF | `clean_left_foot_rear.mp4` | `clean_left_foot_side.mp4` | `clean_left_foot_front.mp4` |
| Sequence | `clean_sequence_rear.mp4` | `clean_sequence_side.mp4` | `clean_sequence_front.mp4` |

Examples of useful exact-state screenshots:

- `clean_right_hand_side_move1_capture_before.png`
- `clean_right_hand_side_move1_capture_after.png`
- `clean_left_hand_rear_move1_reach_mid.png`
- `clean_left_foot_rear_move1_reach_mid.png`
- `clean_left_foot_side_move1_touchdown.png`
- `clean_sequence_front_move1_ready.png`
- `clean_sequence_front_move2_start.png`
- `clean_sequence_front_final.png`

Each camera also has `audit_<case>_<view>_decoded.png`, made from the actual MP4
frames. Exact-state screenshots must not be confused with those directed 20 Hz
brackets. `render_manifest.json` indexes all artifacts and shared camera-state
indices. `visual_review.json` records the manual verdict separately from the
meter's null/pending automatic motion-quality field.

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python scripts/render_clean_transfers.py
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_*clean*.py' -v
python -m unittest discover -s tests -p test_transfer_motion_audit.py -v
```

Verification: **30 focused renderer/motion-audit tests passed**, including actual
EGL composition, independent mass-weighted COM, no-dynamics guards, scene-pixel
preservation, directed decoded brackets and provenance rejection. The renderer
executed **zero physics steps**. The previous Stage 5 numerical acceptance is
preserved; the expensive full foundation suite was not rerun for this display-only
task. Imageio's reader tests emit nonfatal ffmpeg-pipe ResourceWarnings despite
closing the reader and passing cleanup assertions.

**No Stage 6 work, physics edits, motion-reference edits, commit or push.**
