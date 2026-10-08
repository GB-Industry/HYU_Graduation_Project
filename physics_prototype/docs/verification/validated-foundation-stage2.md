# Validated Static Climbing Foundation: Stage 2

## Two Independent Verdicts

**Contact mechanics: PASS.**

**Static controller convergence: NOT YET VALIDATED / DEFERRED TO STAGE 3.**

These verdicts are independent. The maintained mixed fixture retains both loaded,
non-slipping feet and bounded hand grasps throughout five scored seconds. Its
maximum hinge speed still violates the unchanged proposed Stage 3 target of
0.2 rad/s. That violation is reported, not relaxed, replaced by RMS, or hidden.

**Stage 2 DOES NOT validate the locomotion controller or complete climbing
feasibility.** It does not implement planning, RL, dynos, hooks, fatigue, detailed
fingers, or optimal body positioning.

The integration starts from Stage 1 commit `074f907`. All final contact evidence
is rerun in the prescribed project environment: Python 3.12.13, NumPy 2.5.3,
MuJoCo 3.14.0. Temporary exploratory measurements are not certification. No SciPy
or temporary-environment imports are used by runtime mechanics or maintained
diagnostics.

## Canonical Geometry

`contact_geometry.canonical_geometry(region)` is the single contract used by
scene descriptors, the MJCF builder, capture measurements, position-only IK,
foot sensing, and debug/render metadata.

Existing regions are spheres. Optional positive finite `ContactRegion.half_size`
selects a box with half-extents in (right, into-hold, projected-up) order. Radius
remains a serialized field for compatibility, but does not size a box. All Stage 1
container, normal, finite-number, backend-domain, and serialization checks remain.

The region's outward normal `n` and projected world up define a right-handed body
frame. Parallel-to-up normals use a deterministic alternate tangent. For the
default normal `(0,-1,0)`, the body frame is the identity. Spherical shape is
rotation-invariant, but its named contact patches follow that frame.

| Geometry product | Meaning |
| --- | --- |
| `body_frame` | Physical collision center and orientation. |
| `hand_frame` | Outside point-grasp anchor, local +Z outward, local -Z approaching the surface. |
| `foot_surface_frame` | Actual selected sole-touch surface point and normal. |
| `foot_frame` | Existing foot-site reference corresponding to that sole surface. |

Hand target = front surface + **6 mm outward**. The existing hand site is local
`z=-0.04`; the box's distal physical face is `z=-0.046`. This offset places that
face at the surface for exact facing alignment without putting the hand at the
hold center. Body hand geometry, mass, inertia, and END-site locations do not
change.

Sphere foot patches are on the upper-front cap, with normal
`0.5*n + sqrt(3)/2*projected_up`. Box foot patches are on the named top face.
The foot reference is **11 mm above the sole surface along its normal**, matching
the existing foot site's `z=-0.025` versus physical sole `z=-0.036`. Actual support
still comes from collisions, never reference-point proximity.

Compiled `site_HOLD` and `site_step_HOLD` use these frames. IK uses their canonical
world positions, not sphere centers, fixed world Y, or a separate 98 mm offset.
Depth, size, and orientation regression tests compare all consumers against the
compiled geometry. IK remains position-only reference generation, not orientation,
collision, load, or capture-feasibility certification.

## Physical Foot Model

Default `ContactMode.PHYSICAL` creates **no foot equalities**. The existing solid
shoe box is unchanged: half-size `(0.040,0.095,0.018)` m, center
`(0,0.035,-0.018)`, and base mass 1.2 kg. No toes or deformable material are added.

Shoe/hold, shoe/wall, and shoe/floor pairs use an explicit effective sliding
coefficient `min(1.8, surface_friction)`. This avoids MuJoCo's automatic maximum
mixing, which previously let the 1.8 shoe coefficient override a low-friction hold.
The five pair coefficients are `(mu,mu,0,0,0)`, with `condim=3`. Rolling/torsional
adhesion is not introduced.

The solver remains implicitfast/Newton with the existing production 2 ms timestep.
Contact `solref` is `(0.01,1)`. Pyramidal friction is retained; its two-axis diamond
is not treated as the circular elliptic cone. Diagnostic accuracy settings and
1 ms comparisons are explicitly separate from the production controller.

## Live Foot State

`FootSupportSensor` samples native collision contacts/forces, not a target
dictionary or hand registry. A copied endpoint is freshly forwarded without
changing live pose, velocities, controls, time, or warmstart. Measurements retain
native surface geometry names, contact positions/normals, force vectors, effective
mu, point-relative tangential speed, and force utilization.

| State | Meaning |
| --- | --- |
| `FOOT_NO_CONTACT` | No registered shoe/surface contact. |
| `FOOT_CONTACTING` | Touching, but not an admissible loaded static support. |
| `FOOT_SUPPORTING` | Compressive, eligible, aligned, friction-admissible and non-slipping support. |
| `FOOT_SLIPPING` | A positively loaded contact exceeds the declared tangential-speed criterion. |
| `FOOT_INVALID_MEASUREMENT` | Nonfinite geometry/state/solve; never counted as support. |

Ordinary support requires the correct shoe and eligible surface, assembled native
contact, its **own** total normal force greater than **5 N**, sole and selected
surface-normal alignment at least **0.9**, and tangential point speed no greater
than **0.01 m/s**. Native tangential force must satisfy the declared friction bound
within `1e-6 N` numerical allowance. One foot counts once, not once per manifold
point. Multiple supported surfaces remain visible.

Loads are grouped by native `surface_geom`, not labels. A floor and a scene HOLD
both named `floor` cannot share normal-load credit or authorize each other.
Controller admission, source release, target acquisition, and final checks match
the actual `geom_HOLD` and that geometry's own loaded contacts. Floor/wall starts
are explicitly unsupported by the current hold-only scripted reference generator;
they fail before IK rather than leaking a KeyError or pretending floor is a hold.

`mj_contactForce` is transformed with contact-frame axes stored as rows. Its force
acts on geom2 from geom1; sign is reversed when the shoe is geom1. Native normal
magnitude is compressive, not signed world Z. Point velocity includes rotational
lever-arm motion and subtracts the contacted body's own point velocity.

## Foot Benchmark Evidence

Maintained free-shoe fixtures have a full six-DoF freejoint, zero motors, zero
equalities, and no pose/rotation pinning. Surface geometry is fixed. Known loads
act at the COM without imposed torque. Settling is 0.4 s; measured loading lasts
one second with a final 0.5 s stability window.

| Case | Measured result at 2/1 ms |
| --- | --- |
| Horizontal support | Normal force 11.772 N, equal to `1.2*9.81`, within 2% tolerance. |
| 8 N tangent + 6 N downward press, mu 0.3 | Sustained sliding, about 1.11 m over the measured second. |
| Same load, mu 0.8 | About 1.76 mm compliant creep; stable within 0.01 m/s criterion. |
| 15 degree incline, mu 0.3 | Stable; `tan(15)=0.267949 < 0.3`. |
| 35 degree incline, mu 0.3 | Sustained slip; `tan(35)=0.700208 > 0.3`. |
| 35 degree incline, mu 1.2 | Stable with clear friction margin. |
| Pull upward with twice shoe weight | Physical separation, no tail contact/support and no tensile normal force. |

Stable/sliding/separation classifications persist under timestep reduction.
Sliding boxes can rock and intermittently separate; impact peaks are recorded and
not forced to equal static `mg*cos(angle)`. Force-frame sign and generalized-load
mapping are checked independently, including reversed native geom order.

## Bounded Hand Surrogate

Hands use a compliant 3D site connect, **not a weld**. Rotation remains free and
no unlimited moment capacity is claimed. Physical equalities use
`solref=(0.004,1)` and `solimp=(0.99,0.99,0.001,0.5,2)`. These are explicit
engineering compliance settings, not anatomical finger calibration.

Physical acquisition requires all of:

- Canonical site gap at most **1 mm**.
- Relative site speed at most **0.05 m/s**, including articulated/root rotation.
- Facing cosine at least **cos(30 degrees)**.
- Finite signed hand/target geometry distance, surface gap at most **1 mm**, and
  penetration less than **1 mm**.
- Candidate equality reaction within the bound profile's capacity.

Neither force=True nor a larger request radius can bypass physical mode. A valid
replacement is resolved/admitted on scratch data before the source is detached.
Native target identity and registered site topology remain exact.

Retained grasps permit at most **10 mm** compliant surface separation and less
than **1 mm** penetration. This is a maintained-geometry bound, not an expanded
acquisition gate. Invalid geometry/numerics marks the measured hand invalid or
raises a deliberate boundary error; NaN is never converted into zero penetration.

## Capacity And Force Provenance

```text
capacity_N = bound_profile.grip_capacity * hold.grip_quality
demand_N = norm(world Cartesian equality reaction)
```

Grip quality is a dimensionless purchase factor. The force envelope is isotropic
and simplified. No undocumented orientation/friction multiplier is applied after
capture. Orientation is acquisition admissibility; collision friction remains its
own mechanics. Equality load is surrogate demand, not physiological finger force.

Physical execution checks capacity and geometry before and after every native
step, including initial/support/reach phases and settling. Static/support helpers
use the same pre/applied/endpoint guards. Overload detaches/fails; feet never enter
`GripController`. Supplied profile mismatches fail before initialization mutation
or execution. Physical requests cannot disable checks with check_grip=False.

Pre-step endpoint force, the force actually applied over the last integration
interval, and a freshly forwarded endpoint are different epochs. Releases preserve
both applied and endpoint vectors/loads; the selected demand vector and solve-state
time match the selected maximum. No peak magnitude is paired with another epoch's
direction. Complete current-episode logs are cleared on successful explicit reset.
Static/support helpers reject a mismatched physical profile before generating
references or writing controls, not merely before the first integration step.

Enforcement is timestep-sampled, not an analog force-limited solver. Native motion
is not advanced and rolled back to repair a violation. Guards and reaction traces
make applied or prospective overloads visible rather than claiming every possible
within-step excursion is impossible.

## Acquisition Transients And Profile Bracket

Capture records include pre-gap, relative velocity, facing, penetration, target
position, system and hand kinetic-energy metrics, fresh post-activation reaction,
and separate peak applied/endpoint reactions over 20 ms. Hand-only kinetic energy
avoids attributing the independent falling shoe's energy to hand acquisition.
No exact energy conservation is claimed for this inelastic surrogate.
Only the latest acquisition for a limb owns its reaction samples, even when the
same hold is immediately reacquired. Early detachment explicitly truncates the
window and records its time/reason; later motion cannot rewrite that capture.

The canonical maintained near/slow fixture uses 0.5 mm gap and 0.025 m/s approach;
initial hand kinetic energy is approximately `0.000171875 J`. Its measured first
5 ms reaction peak is about **10.3125 N**. Exact 1 mm acquisition is tested; far,
0.05001 m/s, and orientation-invalid attempts are rejected without changing pose,
velocity, or time. Window energy/reaction traces are saved in the hand results.

| Near/slow hand energy over the sampled 20 ms window | 2 ms | 1 ms |
| --- | ---: | ---: |
| Initial hand kinetic energy | 0.000171875 J | 0.000171875 J |
| Peak hand kinetic energy | 0.001074219 J | 0.000703029 J |
| Hand kinetic energy at 20 ms | 0.000006056 J | 0.000004668 J |

The compliant finite-gap capture increases hand kinetic energy transiently; this
is visible and timestep-dependent, not disguised as energy-conserving acquisition.
Both normal benchmark windows reach 20 ms without truncation.

The fixed-load fixture applies a known **100 N** outward field after capture to
identical physical hands, with an independent real-contact shoe unchanged:

| Capacity | Native demand/outcome at 2/1 ms |
| --- | --- |
| Weak, 70 N | Rejects about 99.000015 / 99.000004 N candidate demand; releases before fixed-field grasp integration. |
| Strong, 150 N | Retains the grasp; measured applied peaks about 100.250011 / 100.152229 N. |

The weak fixed-field applied grasp reaction is zero after rejection. Both trials
use the same body masses, geometry, friction, and foot mechanics, whose normal
load remains approximately 11.772 N. This validates contact-capacity response only,
not complete profile-dependent climbing feasibility.

## Mixed Contact Acceptance

The maintained fixture uses the unmodified 78.3 kg humanoid, an explicit upright
legal pose, two forefoot box ledges, two small spherical hand holds, all environment
collisions enabled, and no foot equality. It runs one second settling plus five
seconds scoring. Every scored support/load is measured from native physics.

A fixed force certificate is stored from the earlier diagnostic allocation.
NumPy recomputes all Jacobians, 31 generalized equilibrium rows, positive shoe
corner pressure, friction pyramids, CoP inside the actual overlap, hand capacity,
and original motor limits. Only the derived hinge torque vector is applied; no
planned contact forces, external root wrench, or contact moments are injected.
There is no runtime SciPy solve or pose optimization.

The benchmark-only control is fixed feedforward plus 80 Nm/rad and 1 Nm*s/rad PD,
divided by the existing motor gears and clipped to their limits. It is not promoted
into `compute_pose_control` or the production locomotion controller.

| Measured scored metric | 2 ms | 1 ms |
| --- | ---: | ---: |
| Both feet loaded/non-slipping | 100% | 100% |
| Mean foot Fn, left/right | 373.737 / 314.250 N | 373.772 / 314.208 N |
| Maximum foot slip, left/right | 0.008532 / 0.001526 m/s | 0.005594 / 0.001274 m/s |
| Peak applied hand reaction, left/right | 56.516 / 48.577 N | 56.315 / 48.532 N |
| Maximum held-palm penetration | 0.008264 mm | 0.007718 mm |
| Peak motor utilization | 19.0935% | 18.9268% |
| Saturation / unintended support | None | None |
| Maximum root linear speed | 0.010280 m/s | 0.008408 m/s |
| Maximum root angular speed | 0.033251 rad/s | 0.019275 rad/s |
| Maximum hinge speed | **0.564701 rad/s** | **0.282234 rad/s** |
| RMS hinge speed | 0.041731 rad/s | 0.022494 rad/s |
| Scored samples exceeding 0.2 rad/s maximum-hinge criterion | 17.2% | 5.29% |

**CONTACT ACCEPTANCE: PASS. CONTROLLER CONVERGENCE: NOT YET VALIDATED.**

The 0.2 rad/s maximum-hinge target remains failed and timestep-sensitive. Root
linear/angular and max/RMS hinge velocities are reported separately. No falling,
slipping foot, or secondary shin/forearm/wall/floor support is accepted as a stance
pass. A deliberately altered full-depth ledge test exposes unwanted shin support
and fails contact acceptance rather than counting it as shoe load.

The actual initial touching solve emits zero shoe reactions; ideal rigid statics
does not imply an already-loaded soft-contact initialization. Contacts develop
naturally during the declared settling interval, without live pose resets.

## State, Execution, And Rendering

`ContactSnapshot` and `StateSummary` expose separate hand and foot state families,
mode, capture/release records, native force/support measurements, and complete
integration-state snapshots. The legacy flat limb-to-surface projection is a
convenience, not physical truth.

Hand transitions validate the actual source equality and target identity. Physical
foot transitions validate native loaded source geometry, observe real source-contact
loss under integration, and acquire real target support. They never activate a
foot constraint. Final expected foot supports are validated by native geom and
their own load, not labels. Readiness counts distinct valid hands and non-slipping
supporting feet; its old velocity/root thresholds remain Stage 3 debt.

Runtime queries freshly forward copies only. Original observer copies preserve
raw integration state, and their authoritative physical measurement snapshot is
passed to the renderer. HUDs explicitly separate `HAND_GRASP` target/load/capacity/
margin from `FOOT_SUPPORT` contact/support/slip/Fn/Ft/velocity. A physical STEP is
never labeled attached.
At 320x180 the HUD uses a compact two-line-per-limb layout at the normal font size;
desktop keeps the verbose labels. Required contact fields and the physical/debug
caveat are checked against real text bounds at both resolutions. Overlong reasons
explicitly point to the full manifest rather than clipping the contact evidence.
The reach-result adapter also requires actual valid supporting limbs in physical
mode; root height alone cannot report support after a SUPPORT_FAILURE.

## Physical Defaults And Explicit Debug

Every builder/CLI defaults to **physical**. That model has 16 hand equalities,
zero foot equalities, 15 justified self exclusions, zero environment exclusions,
and 20 explicit shoe/surface pairs in the existing route.

`idealized_debug` is a deliberately NONPHYSICAL legacy software fixture. Only an
explicit compiled mode creates its 14 foot connects and 50 legacy environment
exclusions. Forced/large-distance acquisition and disabled capacity are allowed
only there. Its result/HUD/manifests say NONPHYSICAL / NOT SCIENTIFIC, and its feet
have `FOOT_IDEALIZED_DEBUG_ATTACHMENT`, not physical support. Scientific and debug
results are never interchangeable.

## Old Scripted Regressions

The production synthetic start pose is not an admissible physical contact pose:
base hand gaps are about 70.8/65.7 mm, compact 75.0/74.0 mm, and long 80.0/76.9 mm.
Some proposed palms also deeply penetrate their holds. Initialization rejects
these references **before live reset or integration**, rather than force-grasping
them. This is an initialization/reference assumption failure, not a failing native
foot or hand benchmark. An explicit valid pose can be supplied by the caller.

Old physical transition/render commands therefore return honest initialization
failure, save the actual untouched neutral endpoint and reason, and never run later
moves. There is no fabricated initialized manager or TransitionResult. Positive
bounded-hand/real-foot evidence comes from the maintained contact fixtures.

Stage 0 success/source-release/target-identity, phase budget, validation non-mutation,
full-state continuity, actual accounting, numerical recovery, and observer isolation
regressions remain, with legacy positive sequence fixtures explicitly debug-only.
Base/compact debug sequences complete; long debug H5 acquisition fails and is
recorded as controller/reference debt. No gate or readiness threshold is expanded.
Stage 1 mass/inertia/COM/topology/axis/ROM/actuator/passive/numerical tests remain;
only the intended contact-policy counts and canonical-target references change.

## Verification And Artifacts

Maintained outputs under `outputs/contact-stage2/` include strict-JSON per-case
force/state/energy evidence and `report.json` with canonical executable/package
versions. The default CLI runs 30 contact cases across both timesteps. Contact
failure yields exit 1; controller convergence is reported independently. Optional
EGL rendering is observer-only and not used to make acceptance pass.

Final verification on 2026-10-05, after the final boundary and HUD fixes:

| Check | Result |
| --- | --- |
| Full unittest discovery | **173 run: 172 passed, 1 skipped**, 158.158 s; exit 0. |
| Canonical geometry regressions | 14 passed, included in the full suite. |
| Contact-integrity counterexamples | 6 passed, included in the full suite. |
| Native foot/hand/mixed mechanics | 8 passed, included in the full suite. |
| Renderer regressions | 16 passed, including real small/desktop text bounds. |
| Stage 1 model-validation CLI | PASS; no failed sections; exit 0. |
| Contact-validation CLI | **30/30 contact cases PASS**, no errors; exit 0; controller convergence false. |
| Optional EGL mixed observer | 2/2 contact runs PASS, no render errors; exit 0; metrics identical to unrendered runs. |
| Physical old-route renderer, all modes | Expected exit 1; initialization failures preserve actual neutral endpoints. |
| Physical old-route renderer, 320x180 | Expected exit 1; compact HUD and strict-JSON manifest generated. |
| Headless physical sequence | Expected exit 1; time remains 0; no execution/reset. |
| Whitespace check | `git diff --check` passed. |

The only skipped test is the missing-MuJoCo dependency path, because MuJoCo is
installed. Deliberate nonfinite-injection tests emit expected native warnings.
The existing FFmpeg macroblock warning pads 320x180 video frames to 320x192; PNG
endpoints retain the requested dimensions. It does not alter physics or metadata.

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python -m unittest discover -s tests -v
python scripts/validate_model.py
python scripts/validate_contacts.py --summary
MUJOCO_GL=egl python scripts/validate_contacts.py --suite mixed --render --summary --output outputs/contact-stage2-visual
MUJOCO_GL=egl python scripts/render_demo.py --mode all --output outputs/contact-stage2-old-route
python scripts/view_scene.py --headless --mode sequence
```

Run from `/home/yuchan/Desktop/project/boulder_prototype-gpt61`. The last two
physical old-route commands intentionally exit 1. Generated outputs remain ignored
by Git. No production dependency was added. Full tests include targeted profile,
geometry, event-epoch, reset-history, helper enforcement, surface-alias, and native
load-attribution counterexamples, plus actual force/frame and collision benchmarks.

Final artifact locations:

- `outputs/contact-stage2/report.json` and per-case JSON: all 30 canonical cases.
- `outputs/physical-model-stage1/validation.json`: preserved Stage 1 validation.
- `outputs/contact-stage2-visual/mixed_observer.mp4` and `observer_states.json`:
  both-timestep isolated EGL observation and native state labels.
- `outputs/contact-stage2-visual/mixed_endpoint.png`: extracted 2 ms scored-frame
  illustration; not an acceptance input.
- `outputs/contact-stage2-old-route/`: desktop static/transition/sequence failure
  endpoints and manifests, including `physics_result=null` for rejected initialization.
- `outputs/contact-stage2-old-route-small/`: 320x180 transition failure endpoint and
  manifest, generated with `--mode transition --width 320 --height 180`.

## Changed Files

| Area | Files |
| --- | --- |
| Canonical geometry and schema | `src/boulder_v1/contact_geometry.py`, `src/boulder_v1/schema.py`, `src/boulder_v1/mjcf_builder.py`, `src/boulder_v1/retargeter.py` |
| Runtime contact mechanics and telemetry | `src/boulder_v1/contact.py`, `src/boulder_v1/grasp.py`, `src/boulder_v1/support.py`, `src/boulder_v1/locomotion.py`, `src/boulder_v1/__init__.py` |
| Maintained diagnostics | `src/boulder_v1/contact_benchmarks.py`, `src/boulder_v1/model_validation.py`, `scripts/validate_contacts.py` |
| Observers and demos | `scripts/render_demo.py`, `scripts/view_scene.py`, `scripts/run_demo.py` |
| New contact regressions | `tests/test_contact_geometry.py`, `tests/test_contact_integrity.py`, `tests/test_contact_mechanics.py` |
| Preserved/adapted regressions | `tests/test_contact.py`, `tests/test_grasp.py`, `tests/test_locomotion.py`, `tests/test_mjcf_builder.py`, `tests/test_physical_model.py`, `tests/test_render_demo.py`, `tests/test_retargeter.py`, `tests/test_runtime.py`, `tests/test_schema.py`, `tests/test_view_scene.py` |
| Verification handoff | `docs/verification/validated-foundation-stage2.md` |

## Remaining Stage 3 Debt

The old controller cannot be certified by this milestone. It retains normalized
gain/capability confounding, chatter, position-only IK, route-specific references,
and permissive instantaneous readiness. It also needs admissible initial/reference
generation and actual contact-aware movement objectives. The benchmark's residual
ankle oscillation is reported as numerical/controller convergence debt.

Point grasp is a force-only, rotationally free surrogate; no anatomical fingers,
grasp-moment envelope, fatigue, power limit, or calibrated human friction is claimed.
Sphere/box contact patches and pair coefficients are explicit engineering models.
Accepted scene/profile combinations are not universally feasible. STEP support does
not implement heel/toe hooks or general wall-smear strategies. No route planning,
learned control, or complete climbing-feasibility result is claimed.

Stop after Stage 2; do not begin Stage 3.
