# Prototype v1 Verification Status

## Verified in this environment

- Canonical `BoulderScene` schema and reference validation
- `ClimberProfile` positive-parameter validation
- GRASP/STEP/SMEAR/PRESS limb-affordance compatibility logic
- Breakable-grip load threshold behavior
- Grip orientation compatibility
- MJCF XML well-formedness
- Morphology -> MJCF segment length binding
- ROM -> MJCF joint range binding
- strength -> MJCF actuator gear binding
- mass -> MJCF density binding
- deterministic demo artifact generation

Test result: **13 tests total, 12 passed, 1 skipped**.

## Not verified here

Actual `mujoco.MjModel.from_xml_string(...)` compilation and `mj_step(...)` execution.

Reason: the execution environment does not have the `mujoco` Python package and cannot reach PyPI to install it. The optional runtime smoke test is included and will run automatically once MuJoCo is installed.

This means Prototype v1 is an **M0 code prototype**, not yet evidence that the generated humanoid is dynamically stable or physically climbable.


## v1.1 correction

- User-side MuJoCo 3.14/Windows runtime revealed: `climber_root` had a `freejoint` but zero local mass/inertia.
- Root cause: mass-carrying geoms were only on descendant `climber`; MuJoCo requires a moving/jointed body itself to have valid mass/inertia.
- Patch: add explicit positive inertial to `climber_root`, scaled with `ClimberProfile.mass_scale`.
- Added regression test asserting positive root mass and diagonal inertia.
- Runtime compile/step still requires verification on a host with the MuJoCo package; local packaging environment did not contain MuJoCo.
