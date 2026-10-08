# Validated Static Climbing Foundation: Stage 3

## Verdict And Scope

**Stage 3: COMPLETE. Static deterministic control: PASS.**

This milestone validates deterministic static torque control on the admitted,
unchanged Stage 2 mixed-contact scene, at both 2 ms and 1 ms, including small
declared disturbances. **It does NOT validate generalized climbing movement or
complete climbing feasibility.** No Stage 4 profile-feasibility study or Stage 5
movement system is implemented.

Starting commit: `d0c9efcf04a55bcf0ca6108500a273ece8eb4e2b`.
Authorized worktree: `/home/yuchan/Desktop/project/boulder_prototype-gpt61`.
Canonical evidence uses Python 3.12.13, MuJoCo 3.14.0, NumPy 2.5.3, Pillow 12.3.0,
and the prescribed project venv. No production dependency is added.

## Preserved Physics

The Stage 1 humanoid topology, 78.3 kg base mass, inertias, axes, ROM, actuator
gears/limits, passive damping and armature remain unchanged. The Stage 2 builder,
canonical contact geometry, capture/capacity law, foot sensor and maintained
contact benchmarks are unchanged.

Ordinary feet have **zero equalities** and obtain support from native unilateral
frictional collision. Hands remain bounded, compliant 3D point connects, not
welds or moment-capable finger models. Capture remains <=1 mm gap, <=0.05 m/s
relative speed, facing >=cos(30 degrees), and <1 mm penetration. Capacity remains
`profile.grip_capacity * hold.grip_quality`, enforced before and after every step.
No environment exclusion, friction, compliance or capture criterion is relaxed.

The mixed fixture retains the exact Stage 2 model arrays and solve settings:
32 qpos, 31 velocities, 25 motors, four hand equalities (two active), 15 justified
self exclusions, no environment exclusions, and the existing native shoe pairs.
Only timestep differs between the two declared runs. No root pinning, contact
force injection, planned contact moment or post-initialization teleport is used.

## Failure Mechanisms

The former production controller used normalized gains:

```text
ctrl = clip(kp * (q_ref - q) - kd * qdot, -1, 1)
joint_torque = gear * ctrl
```

Its actual stiffness and damping were `gear*kp` and `gear*kd`. Strength scaling
therefore changed intended impedance as well as capability. On light distal
joints, explicit sampled damping drove saturated near-period-two oscillation at
2 ms. The maintained wrist/ankle legacy diagnostics reproduce approximately
250 Hz chatter; halving timestep makes those particular legacy cases converge.

The Stage 2 mixed benchmark was already torque-space PD, however. Its failure
was not explained solely by normalized gains. The old fixed force certificate
was a valid rigid-static balance, but put foot CoP approximately 0.5 mm from a
ledge boundary with asymmetric load distribution. Finite compliance and feedback
produced rocking/oscillation and longer-hold drift. Neither changing damping nor
simply increasing stiffness fixed the full acceptance problem.

Exploratory 11-second runs at 2 ms with the old allocation measured final-window
joint maxima about 0.899 rad/s (80/1 scalar impedance), 0.930 (80/4), and 0.194
(320/4). The selected class gains alone still reached about 0.948 rad/s and lost
non-slipping support. Those were rejected candidates, not acceptance evidence.
The maintained Stage 2 six-second baseline still reports 0.564701 / 0.282234
rad/s at 2/1 ms and remains independently labeled controller NOT VALIDATED.

## Controller Equation And Units

`runtime.compute_pose_control` now computes physical torque:

```text
tau_des = Kp * (q_ref - q) + Kd * (qd_ref - qdot) + tau_ff
tau_max = gear * symmetric_ctrl_limit
tau_cmd = clamp(tau_des, -tau_max, tau_max)
ctrl = tau_cmd / gear
```

| Quantity | Unit |
| --- | --- |
| q, q_ref | rad |
| qdot, qd_ref | rad/s |
| Kp | Nm/rad |
| Kd | Nm*s/rad |
| tau_des, tau_cmd, tau_ff, tau_max | Nm |
| ctrl | Motor-interface command; its mapping is explicit, not the gain unit |

All references, gains, state and direct-motor capability are checked before any
control writes. Targets must respect actual hinge ROM. Only the validated
positive-gear, symmetric-limit, fixed-gain direct hinge motors are supported.
Optional desired velocities and feedforward are explicit joint-name mappings.
`TorqueCommand` records configured impedance, requested/applied torque, ceilings,
utilization and saturation. `kp=None`, `kd=None` select immutable class defaults;
explicit numeric/mapped arguments now also have physical units.

## Gain Strategy

The fixed class catalog is inertia-informed and experimentally checked against
production-derived isolated hinges with unchanged passive damping/armature. It
does not claim exact critical damping for every coupled posture. No route,
hold-ID or profile-name gain selection is used.

| Joint Class | Kp (Nm/rad) | Kd (Nm*s/rad) |
| --- | ---: | ---: |
| Waist | 160 | 25 |
| Shoulder | 80 | 12 |
| Elbow | 60 | 6 |
| Wrist | 20 | 1 |
| Hip | 160 | 25 |
| Knee | 120 | 12 |
| Ankle | 40 | 2 |

Strength changes only the existing compiled motor gear/torque ceiling. Under
unsaturated small steps, strength 0.5 and 1.5 trajectories match within 1e-12
with identical impedance. Demanding references saturate weaker motors while
stronger motors retain capability. The ankle additionally uses a 0.1-strength
demanding case because its ROM-admitted reference does not saturate strength 0.5.
The response is not artificially normalized to erase strength differences.

## Isolated Joint Evidence

`controller_diagnostics.py` extracts a production body/subtree, selected hinge,
axis, ROM, damping, armature and motor. Other subtree joints are explicitly removed
to make a fixed-proximal-mount single-axis diagnostic. Gravity is zero and
collisions are disabled only in these isolated fixtures; there are no equalities.
These are not modified full-climber physics or contact benchmarks.

The matrix covers 0.05 rad and 0.5 rad steps, a 0.5 Nm / 50 ms torque pulse,
demanding saturation, weak/strong capability and both timesteps. All **86 physical
cases pass**. Four legacy normalized cases are separate diagnostic counterexamples:
two converge at 1 ms; two deliberately fail at 2 ms. They do not affect the
production verdict.

Rise is the 10%-90% crossing. Settling is entry into a 2% position band, minimum
0.001 rad, for the remaining run, with a tail-speed criterion. Results below apply
to the unsaturated small/moderate step responses:

| Joint | Rise 2/1 ms (s) | Settle 2/1 ms (s) | Overshoot at 2 ms |
| --- | ---: | ---: | ---: |
| Shoulder | 0.296 / 0.295 | 0.516 / 0.514 | 0% |
| Elbow | 0.246 / 0.246 | 0.448 / 0.447 | 0% |
| Wrist | 0.206 / 0.208 | 0.374 / 0.375 | 0% |
| Hip | 0.266 / 0.266 | 0.786 / 0.787 | 5.88% |
| Knee | 0.204 / 0.204 | 0.360 / 0.358 | 0% |
| Ankle | 0.212 / 0.211 | 0.380 / 0.379 | 0% |

Across small/moderate/pulse cases, worst tail error is approximately 1.18e-6 rad
and worst tail speed 1.46e-5 rad/s. Pulse recovery is within 226 ms after pulse
end; the hip remains inside its settling band throughout that perturbation.
For 0.5 rad steps, maximum 2-versus-1-ms trajectory differences are about
0.00247 rad and 0.036 rad/s, rather than an assertion of identical trajectories.

| 2 ms Tail Chatter | Legacy Normalized | Physical Impedance |
| --- | ---: | ---: |
| Wrist RMS speed | 2.964 rad/s | <1e-11 rad/s |
| Ankle RMS speed | 3.125 rad/s | <1e-11 rad/s |
| Wrist peak torque | 35 Nm | 1 Nm |
| Ankle peak torque | 55 Nm | 2 Nm |
| Saturated steps in distal small-step runs | 1497/1500 | 0 |
| Tail velocity/torque sign changes | 250 / 249 | 0 / 0 |
| Dominant legacy frequency | Approximately 249-250 Hz | No residual period-two chatter |

Sign changes use 1e-5 rad/s and 1e-4 Nm noise floors. Converged floating-point
noise is not counted as pathological chatter.

## Admissible Initialization

`initialize_static_reference(model, data, scene, profile, initial_qpos, contact_intent)`
accepts a complete explicit seed and four-limb HOLD intent. It validates compiled
ROM/quaternion, canonical Stage 2 body/site frames, actual sole/face overlap,
signed distance and normal alignment, strict hand acquisition and simultaneous
bounded reactions on scratch data **before live reset**. Missing start intent is
completed in an immutable scene copy; conflicting declared intent is rejected.
Callers use the returned manager and its bound scene.

The reference stores immutable target angles and measured residual evidence. A
foot site need not coincide with the center of a box face: the Stage 2 mixed seed
has a legitimate 10 mm tangential offset, exact sole-height alignment and real
overlap. Initially touching soft contacts may be unloaded; they are not mislabeled
as already supporting. Both feet support from the first controlled endpoint.

The benchmark reuses the existing Stage 2 legal upright seed and scene without
body/contact changes. No IK is needed for this admitted seed. Existing positional
IK remains only a reference generator, not orientation, capture or dynamic
feasibility proof. No invalid old initializer is restored.

## Contact-Aware Feedforward

Static support estimates use the center of the actual clipped shoe/ledge overlap,
not an edge-biased frozen allocation. Starting from 45% of weight per foot and
5% per hand, a small NumPy least-squares projection satisfies six free-root
force/moment balance rows. Friction pyramids, positive foot compression, hand
capacity and original motor torque ceilings must admit the candidate.

Only `qfrc_bias - J.T * estimated_forces` at actuated hinge coordinates is used as
feedforward. Estimated contact forces, contact moments and root wrenches are
**never applied**. The estimate is fixed at the reference; actual support is
subsequently verified from native collisions/equalities, not from this certificate.
Failure of this simple estimate does not prove physical infeasibility.

For the maintained seed, estimated forces are approximately 355.079 N vertical
and -35.904 N wall-normal per foot, and +35.904 N wall-normal / +28.982 N vertical
per hand. Root-balance residual is around 2e-14. CoP is comfortably inside actual
overlap. A **0.2 s feedforward ramp** lets compressive contacts load naturally and
avoids the immediate-full-feedforward startup slip seen during investigation.
It changes actuation, not contact mechanics or live pose.

## Mixed Static Acceptance

Protocol: **1 s settling + 10 s controlled hold**, final **2 s scored**. Native
steps are 5500 at 2 ms and 11000 at 1 ms. Both hands are guarded before/applied/
endpoint solves. Both feet support without slipping at every controlled endpoint,
including ramp/settling. Unintended body contacts are checked in applied and fresh
endpoint solves. There are no resets, hidden supports, grasp releases or saturation.

Speed/error metrics below are final-window maxima, RMS speed is over that window,
motor utilization and applied hand peaks cover the whole run, and foot Fn mean is
over the final window. Foot minima/Ft/slip maxima cover all controlled endpoints.

| Nominal Metric | 2 ms | 1 ms | Acceptance |
| --- | ---: | ---: | --- |
| Root linear speed max | 0.000187436 m/s | 0.000187438 m/s | <=0.02 m/s |
| Root angular speed max | 0.002319612 rad/s | 0.002319609 rad/s | <=0.05 rad/s |
| Joint speed max | 0.003240191 rad/s | 0.003240185 rad/s | <=0.10 rad/s |
| Joint speed RMS | 0.001098961 rad/s | 0.001098959 rad/s | Reported, not a substitute for max |
| Joint position error max | 0.032231865 rad | 0.032231503 rad | Reported |
| Motor utilization max / mean | 48.432384% / 7.003739% | 48.432379% / 7.003935% | No saturation |
| Saturation fraction | 0% | 0% | 0% observed |
| Hand applied peak left/right | 49.930156 / 49.930156 N | 49.930534 / 49.930534 N | Capacity 850 N each |
| Foot Fn mean left/right | 356.998360 / 356.998360 N | 356.998127 / 356.998127 N | Actual compressive support |
| Foot Fn min, each | 145.841005 N | 145.269889 N | >5 N |
| Foot Ft max, each | 42.746071 N | 42.745645 N | Native friction-admissible |
| Foot slip speed max, each | 0.002745518 m/s | 0.002745513 m/s | <=0.01 m/s |
| Foot support / slipping endpoint fractions | 100% / 0% | 100% / 0% | Both feet throughout |
| Scored sustained readiness | 100% | 100% | All scored endpoints |

The same qualitative convergence verdict holds at production 2 ms and comparison
1 ms. Both final-window velocity and torque sign-change counts are zero for the
largest-RMS joint. Its weak residual trend has dominant frequency about 0.5 Hz,
with only about 0.062% / 0.031% power above 80% of Nyquist. It is not a hidden
step-to-step oscillation. Nonzero tracking error and slow compliant drift remain
reported; this is finite-duration prototype acceptance, not infinite-time proof.

## Disturbance Recovery

A declared +2 N world-X force acts on the dynamic welded `climber_root` body from
5.0 to 5.1 s. That massless coordinate body is rigidly connected to the massive
pelvis; its zero direct mass does not mean its composite dynamics are fixed.
Pulse timing uses integer native intervals: 50 at 2 ms, 100 at 1 ms, realized
impulse **[0.2, 0, 0] Ns**. No velocity/pose perturbation or reset is used.

| Disturbed Metric | 2 ms | 1 ms |
| --- | ---: | ---: |
| Pulse-response root speed peak | 0.003999650 m/s | 0.003984559 m/s |
| Pulse-response joint speed peak | 0.009262898 rad/s | 0.009257740 rad/s |
| Final root linear max | 0.000187759 m/s | 0.000187761 m/s |
| Final root angular max | 0.002319858 rad/s | 0.002319854 rad/s |
| Final joint speed max | 0.003245577 rad/s | 0.003245571 rad/s |
| Final joint RMS | 0.001099018 rad/s | 0.001099016 rad/s |
| Hand applied peak left/right | 49.947453 / 49.912849 N | 49.947828 / 49.913230 N |
| Foot Fn mean left/right | 356.963528 / 357.033185 N | 356.963301 / 357.032945 N |
| Largest all-run foot slip | 0.002747177 m/s | 0.002747548 m/s |
| First unforced ready endpoint after pulse | 5.102 s | 5.101 s |

Both disturbed runs retain 100% foot support, no slipping endpoints, bounded valid
hands, zero releases and zero saturation. Readiness is never lost for this modest
pulse, so the listed ready timestamps are not a claim of reacquiring a lost
window. The actual pulse produces a measurable response; it returns toward the
nominal state, with final qpos within 0.002 and qvel within 0.005 of the matching
nominal run. Larger, stance-breaking disturbances are not promised recoverable.

## Sustained Readiness And Failures

`ReadinessTracker` is session-bound and sampled by the authoritative integration
owner after each native step. It requires **0.5 s of continuous valid endpoints**:
root linear <=0.02 m/s, root angular <=0.05 rad/s, maximum hinge <=0.10 rad/s,
finite integration/control state, both valid bounded hands, both actual supporting
non-slipping feet, and no unintended loaded body contact. RMS is reported but
cannot conceal a fast hinge. Queries and duplicate samples cannot accrue time.
Gaps, reset/adoption, invalid state or contact loss clear the window. Physical
`check_stabilization_readiness` without owned history is false; old thresholds
cannot relax it. Explicit debug readiness is NONPHYSICAL software behavior only.

Static results expose `REFERENCE_INVALID`, `TORQUE_LIMITED`, `CONTACT_LOSS`,
`GRASP_OVERLOAD`, `STABILIZATION_TIMEOUT`, `NONFINITE_STATE`, and `SUCCESS`.
Persistent requested-torque saturation for 0.5 s stops as TORQUE_LIMITED. A final
step failure cannot be overwritten by successful step count or earlier ready rows.
Numerical recovery reports actual endpoint and integrated work, not a restored pose.

Terminal observations are separately measured from current data, carry actual
status/reason and never pair stale successful contact telemetry with a failed
pose. Invalid state is not forwarded into a fabricated render pose. Callback
exceptions propagate without being relabeled as physics failure; declared force
cleanup is guaranteed in `finally`, restoring only the owned root-force channel.
Failures before pulse onset retain steps, states and release logs, with unavailable
recovery peaks rather than a secondary reporting exception. Nonaligned or empty
native pulse intervals are rejected before mutation.

## Old Route And Foundation Regressions

After static success, `view_scene.py --headless --mode sequence` still exits 1
with honest INITIALIZATION_FAILURE: base hand gaps are approximately 70.8/65.7 mm,
far outside 1 mm acquisition, with substantial proposed palm penetration. Time
remains zero; no reset/execution or later moves occur. No route-specific repair,
forced capture or generalized transition redesign is introduced.

Physical static helpers now track the admitted caller pose rather than silently
substituting the old synthetic joint target. Physical controller arguments have
physical units throughout. Old movement references/choreography remain later-work
debt and are not Stage 3 acceptance requirements.

Strong Stage 0 software success/source-release/target-identity/phase/accounting/
continuity/observer-isolation regressions are retained. Their historical positive
IDEALIZED_DEBUG fixtures explicitly patch a **test-only archived normalized PD**
that writes only controls and refuses physical models. This is not production
compatibility behavior or Stage 3 controller evidence. Physical overload tests now
inject known overloads rather than depend on a bad controller creating them.

## Verification And Rendering

Final canonical acceptance includes:

- Final full unittest discovery on 2026-10-05: **232 run, 231 passed, 1 skipped**,
  668.137 s, exit 0, after the final nonfinite-render safeguard. The skipped path
  is missing MuJoCo, because MuJoCo is installed.
- Stage 1 model-validation CLI: PASS, no failed sections.
- Stage 2 contact-validation CLI: 30/30 PASS; its unchanged historical static PD
  baseline still correctly reports controller convergence false.
- Stage 3 CLI: all 86 physical isolated cases and four mixed holds PASS, strength
  semantics PASS, contact/convergence PASS, four EGL videos PASS, no errors.
- Stage 0 boundary tests, 19 static-state tests, 15 impedance tests, 17 actual
  static-control tests and seven short failure-boundary tests are included.
- Deliberate nonfinite injections emit expected native warnings; existing small
  archived-render tests emit the unchanged FFmpeg macroblock padding warning.

`validate_controller.py --render` observes the same authoritative runs, never a
physics replay. Four 640x480, 20 fps videos have 221 observations each, showing
settling, hold, optional pulse and recovery. HUDs report root/max/RMS speeds,
torque/utilization/saturation, hand loads/capacities, foot Fn/Ft/support/slip,
sustained readiness and actual terminal status. Model/data/row copies are detached;
an adversarial observer cannot alter the benchmark result.

Artifacts are ignored by Git under `outputs/controller-stage3/`:

- `report.json`: canonical environment, separate verdicts and summarized metrics.
- `isolated.json`: all isolated/controller-strength/legacy diagnostic evidence.
- `mixed_nominal_2ms.json`, `mixed_nominal_1ms.json`: native state/contact/control traces.
- `mixed_disturbance_2ms.json`, `mixed_disturbance_1ms.json`: pulse realization and recovery traces.
- Corresponding `*.mp4` and `*_observer.json`: same-run observations, including terminal state.
- `disturbance_frame.png`: extracted illustration from the actual 2 ms pulse video.

Run from the authorized worktree:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python -m unittest discover -s tests -v
python scripts/validate_model.py
python scripts/validate_contacts.py --summary
MUJOCO_GL=egl python scripts/validate_controller.py --render --summary
python scripts/view_scene.py --headless --mode sequence
```

The last command intentionally exits 1. Optional isolated traces are available
with `--include-traces`; `--suite isolated` or `--suite mixed` selects diagnostics,
and `--dt 0.002` / `--dt 0.001` selects a timestep without retuning gains.

## Changed Files

| Area | Files |
| --- | --- |
| Physical torque API and integration | `src/boulder_v1/runtime.py`, `src/boulder_v1/grasp.py`, `src/boulder_v1/locomotion.py`, `src/boulder_v1/__init__.py` |
| Maintained Stage 3 mechanisms | `src/boulder_v1/controller_diagnostics.py`, `src/boulder_v1/static_state.py`, `src/boulder_v1/static_control.py` |
| Diagnostics and viewer | `scripts/validate_controller.py`, `scripts/view_scene.py` |
| New controller/state/failure tests | `tests/test_impedance_control.py`, `tests/test_static_state.py`, `tests/test_static_control.py`, `tests/test_static_failures.py` |
| Preserved foundation fixtures | `tests/foundation_fixture_control.py`, `tests/test_locomotion.py`, `tests/test_grasp.py`, `tests/test_contact_integrity.py` |
| Documentation | `README.md`, `docs/verification/validated-foundation-stage3.md` |

## Remaining Limitations

The gain catalog is validated for these prototype diagnostics, not universally
for every mass/morphology/ROM/contact combination. The simple force estimate can
fail where another allocation would work; that is not fundamental infeasibility.
Compliant drift and nonzero joint error remain visible. Human friction, grasp
moments and anatomical finger behavior are not calibrated. Positional IK,
contact-aware moving references, release/reach objectives and generalized
transition convergence remain later work. No RL, planning, dynos, hooks, fatigue,
profile-feasibility study or major graphics redesign is added.

**Stop after Stage 3. Do not begin Stage 4.**
