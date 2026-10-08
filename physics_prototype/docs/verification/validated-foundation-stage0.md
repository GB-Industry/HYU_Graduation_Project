# Validated Static Climbing Foundation: Stage 0

## Scope

Stage 0 establishes truthful execution and observation semantics. It does not
improve the climbing controller or replace contact mechanics.

**Stage 0 DOES NOT certify physical climbing feasibility.**

The recovery starts from `e5292aa`; that commit and all earlier milestones are
not treated as validated physical baselines. Successful acquisition below means
an actual live MuJoCo attachment to the requested hold, not anatomically or
frictionally feasible climbing.

## Previous Execution Defects

| Previous behavior | Stage 0 behavior |
| --- | --- |
| A one-step H4 -> H5 request could succeed while still attached to H4. | Truncated progression returns `INCOMPLETE`; actual H4 remains reported. |
| Caller contact dictionaries were accepted as physical truth. | Dictionaries are assertions against verified live attachments. |
| Execution without a manager rewrote pose, cleared velocity, and attached starts before validation. | An intentional `AttachmentStateError` rejects missing/uninitialized/mismatched ownership. |
| A new manager could ignore old active equalities. | It must explicitly initialize or adopt live state; drift and duplicates raise. |
| Invalid requests could change live state. | Validation performs no control, attachment, pose, velocity, or forward writes. |
| Failure results retained source contacts after release, or claimed requested targets. | Every result is derived from the actual final active equality identities. |
| `REACH_FAILURE` was advertised without an implementation. | Removed; Stage 0 does not certify geometric infeasibility. |
| Step counts excluded settling and overstated early failures. | Count completed `mj_step` calls, including settling; report execution duration separately from absolute endpoint time. |
| Sequence rendering ignored failures and advanced target bookkeeping. | One sequence executor halts; rendering displays its actual terminal result and attachments. |
| Render/viewer paths implemented separate transition choreography. | Both observe the authoritative transition/sequence executor. |
| Continuity tests compared only root translation and time. | Full integration state, solver warm start, controls, and attachment identities are compared. |

Additional adversarial checks discovered and closed inactive-target topology
validation, observer result-dictionary aliasing, observer model/derived-data
mutation, native numerical-recovery reporting, and nonfinite HUD formatting.

## Episode Lifecycle

`initialize_episode(model, data, scene, profile, attach_feet=...)` is the explicit
episode-reset operation. It validates initial attachment registration, resets
MuJoCo data, clears old equality activations and registry entries, writes the
existing demo stance and morphology-dependent root offsets, and establishes the
intended initial attachments. Episode time, controls, and integration state start
from the reset state. This is a synthetic initialization, not a feasibility test.

`setup_static_stance(...)` remains an existing explicit entry point delegating to
this operation. It now resets the whole episode, including the clock. Neither
transition executor calls either initializer.

```python
import mujoco
from boulder_v1 import (
    ClimberProfile, Limb, TransitionRequest, build_mjcf, compile_model,
    execute_transition_sequence, initialize_episode, make_synthetic_scene,
)

scene = make_synthetic_scene()
profile = ClimberProfile(name="base")
model = compile_model(build_mjcf(scene, profile))
data = mujoco.MjData(model)
manager = initialize_episode(model, data, scene, profile, attach_feet=True)
result = execute_transition_sequence(
    model, data, scene, profile,
    [TransitionRequest(Limb.RIGHT_HAND, "H4", "H5")],
    manager=manager,
)
```

`execute_transition`, `execute_transition_sequence`, and the existing
`simulate_single_limb_reach` adapter require an explicitly initialized manager
bound to the exact model/data/scene. The legacy support demo also requires an
explicit manager. Static demo helpers may explicitly initialize their owned
episode, but passing an existing manager continues that episode without reset.

An empty transition sequence is rejected with `ValueError`; it cannot manufacture
a vacuous successful movement.

## Authoritative Attachment State

The manager registers equality IDs with their exact names, limb, region, connect
type, site object type, and both site IDs. `active_attachments()` reads
`data.eq_active`, verifies active identities, rejects multiple attachments on one
limb, and compares the derived mapping with the manager registry.

`contact_configuration()` returns that verified limb -> hold mapping. Detached
limbs are absent. This mapping describes equality attachments, not all collision
contacts or validated load-bearing supports.

Missing or incorrect inactive target constraints are checked before any observer,
IK reference computation, control write, release, or integration. Attachment
replacement resolves and validates the new constraint before detaching the old
one.

A freshly constructed manager does not silently claim ownership of an existing
activation set. There are two explicit operations:

- `initialize_episode(...)`: reset the episode and establish intended starts.
- `manager.synchronize_from_live()`: adopt an existing unambiguous activation set
  without changing pose, velocity, controls, warm start, or time.

Registry drift, duplicate activations, wrong equality sites, and mismatched
session bindings raise `AttachmentStateError`. There is no automatic repair in
ordinary execution. A stale manager after another owner's reset cannot silently
report its old contacts.

`current_contacts`, `source_contact_configuration`, and
`target_contact_configuration` are optional assertions. The requested final
configuration must equal the actual starting mapping with exactly the requested
limb changed. They never command or construct physical state.

## Exact Success Contract

`SUCCESS` requires all of the following:

- The requested source equality was actually active at entry.
- The release event actually deactivated that source equality.
- The executor observed successful activation of the requested target equality.
- The requested target remains active and the source remains inactive at exit.
- Actual final attachments match the requested single-limb change, including
  unchanged supporting attachments.
- All seven existing phases ran in order, followed by the requested settling
  work.
- The sampled live state remained finite, with no detected numerical-recovery
  warning or clock discontinuity.
- The existing endpoint readiness predicate accepted the state.

`released`, `target_captured`, and `eligibility_detected` record actual execution
events. A prior capture is not the same as still being attached at completion.
The reach adapter and renderer's `reattached` flag require both a capture event
and the actual final target identity.

The readiness thresholds, PD law, equality feet, and capture rules remain
unchanged. The legacy phase name `STABILIZED_STANCE` means final-reference
tracking/settling, not a stability certificate. Its observation status remains
`RUNNING` until a terminal result exists.

## Failure And Incomplete Semantics

| Outcome | Exact meaning |
| --- | --- |
| `INVALID_REQUEST` | Invalid budgets, limb/hold syntax, gains, distance, same-source/target request, unknown source, or contradictory configuration assertion; zero integration and no state mutation. |
| `SOURCE_NOT_ATTACHED` | The requested source is not the actual active attachment; zero integration. |
| `INELIGIBLE_TARGET` | Unknown target, missing required affordance, or missing compiled target attachment; zero integration. |
| `INCOMPLETE` | Main budget was exhausted before all seven phases ran; report the actual partial endpoint and do not add settling to disguise it. |
| `ATTACH_FAILURE` | Full phase progression occurred, but the required source-release/target-acquisition/final-attachment contract was not satisfied. This does not prove geometric infeasibility. |
| `SUPPORT_FAILURE` | Missing actual starting attachments required by the current four-attachment scaffold, or root below the existing collapse threshold. |
| `GRIP_FAILURE` | The unchanged optional grip evaluator detached a limb during its existing checked phases. Its physical semantics remain unvalidated. |
| `UNSTABLE_FINAL_STATE` | Actual target acquisition occurred, but the existing readiness predicate rejected the final state. Actual target attachment is still reported if present. |
| `NONFINITE_STATE` | Nonfinite sampled state, numerical-recovery warning, or unexpected integration-clock change. Stop and preserve the actual resulting state. |
| `AttachmentStateError` | Intentional API error for ambiguous/inconsistent ownership or invalid constraint identity; never silently reset or guess a mapping. |

`RUNNING` is observation-only, never a terminal successful result. Observer
exceptions propagate; closing a GUI or its duration expiring cancels playback
and is reported as interruption, not completion. There is no automatic headless
rerun after GUI failure.

A sequence stops on its first unsuccessful result. `completed_moves` counts only
successful moves, `failed_move_index` is zero-based, and final mappings come from
the failed move's actual endpoint rather than its requested or pre-move contacts.

MuJoCo can internally reset data during numerical recovery. The executor detects
warnings/clock behavior before registry-dependent snapshotting, explicitly adopts
the recovered activation set, and emits `NONFINITE_STATE`. It does not restore
pose, reattach supports, or continue the sequence. This failure is explicitly
outside the normal no-reset continuity guarantee.

## Accounting And Continuity

`TransitionResult.steps` counts completed calls to physics integration, including
settling and partial execution. It is not the requested main budget, video-frame
count, or solver iteration count. A 210-step move plus 40 settling steps reports
250; the successful three-move fixture reports 750 and 1.5 simulated seconds.

`time` and snapshot times are the actual absolute `data.time` values. `duration`
accumulates observed integration-clock increments when continuous. If recovery
produces a discontinuous clock increment, the completed integration cadence is
counted using that call's timestep and `clock_discontinuity` is set. Numerical
warnings still cause failure when the clock happens to look continuous. Recovery from time
7.0 to time 0.002 still reports 0.002 seconds of execution, not a negative
duration. Sequence `total_time` sums actual move durations. Recovery on call ten
reports ten calls and 0.020 seconds even if the recovered clock is 0.002.

For normal transition boundaries, the next initial snapshot exactly equals the
previous final snapshot for:

- Complete `qpos`, including root quaternion and all joint positions.
- Complete `qvel`, including root angular velocity.
- Absolute simulation time.
- All `eq_active` entries.
- All actuator controls.
- `qacc_warmstart`.
- Active limb -> equality identity and limb -> hold mapping.

Tests additionally compare MuJoCo's `mjSTATE_INTEGRATION` vector and derived
acceleration/site/constraint arrays against actual callback-time data, prohibit
`mj_resetData` inside normal sequences, and instrument every live `mj_step` to
ensure pose and velocity are not reconstructed between integrations.

## Rendering Architecture

`TransitionObservation` carries the request, actual phase, actual per-move step
count, state snapshot, sequence index, and optional terminal result. The callback
signature is `(observation, observed_data, observed_manager)`.

Observers receive detached model/data/manager copies, a copied scene and request,
and a defensive copy of the terminal result. They cannot alter authoritative
integration state, model options, registry, request mappings, or result mappings
through the supplied arguments. This is object isolation, not a sandbox against
a callback deliberately accessing unrelated captured/global originals.

Single and sequence offscreen rendering use one shared observer helper and
`execute_transition_sequence`; a single move is a one-request sequence. There is
no renderer PD controller or phase schedule. Static rendering delegates to the
existing static simulator and labels the output as observation only.

Rendering refreshes derived geometry on scratch data. Nonfinite state uses an
explicit pose-unavailable image, not a reconstructed/recovered attractive pose.
Every terminal outcome gets an endpoint PNG and strict-JSON manifest, including
failed or rejected moves. Nonfinite numeric values serialize as `null` and are
displayed safely. Encoding failures are recorded separately and do not erase the
physics result or endpoint.

HUDs and manifests display actual status, moving limb, requested source/target,
actual active attachments, and phase. Failed endpoints say `FAILURE/<status>`.
Montage annotations derive from the same observations. Keyframes correspond to
observed phases, including the actual release event, not nominal frame indices.
No later moves are rendered after authoritative failure.

The transition/sequence GUI also initializes once and observes the same
executor. Its native viewer uses a separate model and display data because GUI
sync may modify both. Success/failure endpoints remain frozen; there is no
periodic reset, replay, or additional unreported stepping. Pose/free debug modes
remain explicitly separate from transition validation.

Video playback remains slow motion: single moves normally retain every physics
step, sequences subsample every second step while preserving phase/terminal
events. Frame count must not be confused with integration steps or real-time
movement speed.

## Regression Coverage

| Requirement | Tests |
| --- | --- |
| Invalid requests do not mutate full live state | `test_invalid_requests_leave_complete_live_state_unchanged`; canonical integration vector plus derived arrays and registry, with nonzero time/control/velocity/warm start. |
| Explicit lifecycle, no implicit reset | `test_missing_uninitialized_or_wrong_session_never_resets`; missing, uninitialized, wrong-data managers and empty sequences. |
| Actual target/source/registry/result agreement | Strengthened bilateral-hand, foot-reposition, and production-profile sequence tests; independently enumerate active equality names. |
| Duplicate/drift/topology rejection | `test_duplicate_drift_and_wrong_constraint_identity_are_rejected`; `test_inactive_target_topology_is_rejected_before_mutation`. |
| Explicit ownership adoption/reset and atomic replacement | `test_new_owner_requires_explicit_adoption_or_initialization`; `test_failed_attachment_replacement_preserves_current_constraint`. |
| Partial budgets are not success | `test_short_budgets_cannot_claim_target_or_completion`; 1, 28, and 40 steps plus the existing reach adapter. |
| Settling and nonzero-time duration accounting | `test_actual_duration_includes_settling_at_nonzero_start_time`; strengthened sequence totals. |
| Complete normal continuity | `test_multi_move_sequence_without_reset`; `test_motion_state_changes_only_inside_live_integration`. |
| Mid-physics and validation failure propagation | `test_physics_failure_stops_sequence_and_preserves_actual_endpoint`; existing validation-failure sequence regression. |
| Nonfinite main/settling/native recovery | `test_nonfinite_main_or_settling_state_never_succeeds`; nonzero-clock and late recovery duration cases. |
| Observer isolation | `test_observer_model_data_and_results_are_detached_from_execution`; hostile supplied-copy mutations. |
| Rendering respects incomplete/failure outcomes | Short single/sequence regressions; endpoint manifests, actual phase/attachments, no later moves. |
| Production failing-profile rendering | `test_production_sequence_failure_and_observer_physics_equality`; production CLI regression and endpoint artifacts. |
| Numerical failure artifacts | `test_numerical_failures_preserve_actual_endpoint_without_fake_pose`; native recovery and NaN pose, safe HUD values. |
| GUI observer and interruption semantics | Fake viewer deliberately forwards/resets/perturbs display inputs; exact integration state and result parity, frozen endpoint, cancellation, no automatic rerun. |

The ineffective old no-teleportation test was replaced rather than preserved.
Profile tests now use exact production presets. The historical passive-foot
pre-shift test is explicitly labeled as its own demo regression, not validation
of the generalized executor.

## Verification Commands

Run from `/home/yuchan/Desktop/project/boulder_prototype-gpt61`:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python -m unittest discover -s tests -v
python scripts/view_scene.py --headless --mode stance
python scripts/view_scene.py --headless --mode transition
python scripts/view_scene.py --headless --mode sequence
python scripts/view_scene.py --headless --mode sequence --steps 40
python scripts/view_scene.py --headless --mode sequence --profile long_reach_lower_grip
MUJOCO_GL=egl python scripts/render_demo.py --mode all
MUJOCO_GL=egl python scripts/render_demo.py --mode sequence
MUJOCO_GL=egl python scripts/render_demo.py --mode sequence --profile long_reach_lower_grip --output outputs/stage0-long
python scripts/run_demo.py
```

The 40-step sequence and both production-long sequence commands must exit 1.
Production long completes two moves, then reports LH H3 -> H7 `ATTACH_FAILURE`,
710 integrations and 1.420 seconds, with LH absent from actual attachments.
The remaining base execution/render commands must exit 0.

Verification environment: Python 3.12.13, MuJoCo 3.14.0, NumPy 2.5.3, Linux EGL.
Executed verification results:

| Run | Result |
| --- | --- |
| Full tests | 91 run, 90 passed, 1 skipped (MuJoCo is installed, so the missing-dependency branch does not apply). |
| Base headless stance / transition / sequence | Exit 0; transition reports 250 integrations / 0.500s, sequence 750 / 1.500s. |
| Base 40-step headless sequence | Exit 1; first move `INCOMPLETE`, 40 / 0.080s, no later moves, moving hand absent. |
| Production-long headless sequence | Exit 1; third move `ATTACH_FAILURE`, 710 / 1.420s total, LH absent. |
| EGL `--mode all` and `--mode sequence` | Exit 0; base transitions and actual results preserved in PNG/video/JSON artifacts. |
| Production-long EGL sequence | Exit 1; endpoint visibly labeled `FAILURE/ATTACH_FAILURE`, attachments `LH:- RH:H5 LF:H6 RF:H2`, matching manifest and executor. |
| `run_demo.py` | Exit 0 with explicitly initialized support and reach sessions. |

Base artifacts are under `outputs/visual/`; the failing production-profile
endpoint and manifest are under `outputs/stage0-long/`. Generated outputs are
ignored by Git. Native warnings in deliberately corrupted-state tests are
expected. Real desktop GUI interaction is not verified by EGL; native display
isolation and interruption are covered by fake-viewer tests.

## Remaining Physical Invalidity

The following are deliberately unchanged and not certified by Stage 0:

- Equality-attached feet are bilateral point attachments, not unilateral
  frictional shoe support; they can hide separation and slip.
- Hand/foot capture still permits approximately 15 cm separation and has no
  capture-speed gate. Capture constraints can supply significant assistance.
- Grip checks remain disabled by default, apply hand-capacity semantics to feet
  when enabled, and retain their existing phase coverage/calibration limits.
- The normalized saturated PD controller still chatters. Gains, actuator gear,
  timestep, readiness thresholds, and same-side pre-shift relaxation are not
  redesigned.
- Readiness remains a permissive instantaneous predicate, not sustained
  stabilization or a balance/load-feasibility certificate.
- Broad environment collision exclusions, simplified grip orientation/force
  semantics, synthetic hold geometry, and step-site offsets remain.
- Anthropometric inertia, mass distribution, anatomical ROM, and profile power
  conditioning remain unvalidated or incomplete.
- IK remains positional reference generation, not force/collision feasibility;
  its geometry assumptions, priors, and convergence limitations remain.
- Profile-name/hold-ID tuning and hand-authored choreography remain. No route
  generalization, planning, profile-feasibility study, or RL was implemented.

Stop after Stage 0. A truthful software `SUCCESS` must not be presented as proof
that a human-like profile could physically perform the movement.
