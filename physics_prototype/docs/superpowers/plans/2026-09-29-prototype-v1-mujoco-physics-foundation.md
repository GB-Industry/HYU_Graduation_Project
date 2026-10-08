# Prototype v1 MuJoCo Physics Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a small M0 prototype that preserves the approved `BoulderScene` / `ClimberProfile` architecture, binds profile parameters into MuJoCo MJCF, and exercises contact/grip semantics without starting RL.

**Architecture:** Keep scene/profile/contact logic independent from the MuJoCo runtime. Generate deterministic MJCF from the canonical scene and a climber profile; use a pure-Python contact model for GRASP/STEP/SMEAR/PRESS and breakable-grip decisions so core semantics are testable even when MuJoCo is unavailable.

**Tech Stack:** Python 3.11+, standard library, optional MuJoCo 3.x, matplotlib for proposal/demo visualization.

**Spec:** `/mnt/data/2026-09-17-adaptive-bouldering-character-control-design.md`

## Global Constraints

- Do not start RL training in Prototype v1.
- `BoulderScene` remains the canonical scene representation.
- `ClimberProfile` must change actual MJCF morphology, ROM, actuator strength, mass, and grip logic.
- Core affordances are GRASP, STEP, SMEAR, PRESS.
- Grip is breakable; SMEAR/PRESS are not modeled as welded grasps.
- Prototype humanoid may use fewer DoF than the final 23–27 DoF target, but the simplification must be explicit.

## Review Focus

- Invalid/non-positive profile dimensions must be rejected rather than creating malformed MJCF.
- Affordance/limb mismatches (e.g. foot GRASP) must be rejected.
- Grip failure must occur deterministically above effective capacity and survive below it.
- Changing reach/morphology must visibly change generated segment dimensions.
- Missing MuJoCo installation must fail with a clear actionable message, not an import-time crash.

---

### Task 1: Canonical schema and synthetic scene

**Files:**
- Create: `src/boulder_v1/schema.py`
- Create: `src/boulder_v1/scene_factory.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `ClimberProfile`, `ContactRegion`, `WallSurface`, `BoulderScene`, `make_synthetic_scene()`.

- [x] Write schema validation tests for positive morphology and valid friction.
- [x] Run tests and confirm failure before implementation.
- [x] Implement dataclasses/enums and a deterministic synthetic wall.
- [x] Run tests and confirm pass.

### Task 2: Affordance and breakable-grip semantics

**Files:**
- Create: `src/boulder_v1/contact.py`
- Test: `tests/test_contact.py`

**Interfaces:**
- Consumes: schema types.
- Produces: `contact_allowed(...)`, `effective_grip_capacity(...)`, `GripController.evaluate(...)`.

- [x] Add tests for limb/affordance compatibility, orientation penalty, and break threshold.
- [x] Run tests and confirm failure.
- [x] Implement deterministic contact/grip model.
- [x] Run tests and confirm pass.

### Task 3: Profile-conditioned MJCF generation

**Files:**
- Create: `src/boulder_v1/mjcf_builder.py`
- Test: `tests/test_mjcf_builder.py`

**Interfaces:**
- Consumes: `BoulderScene`, `ClimberProfile`.
- Produces: `build_mjcf(scene, profile) -> str`.

- [x] Add tests asserting valid XML and profile-dependent segment length, joint range, mass, and actuator gear.
- [x] Run tests and confirm failure.
- [x] Implement reduced-DoF humanoid MJCF generator with free root, wall, floor, holds, hand/foot sites, and motors.
- [x] Run tests and confirm pass.

### Task 4: Optional MuJoCo runtime adapter and smoke runner

**Files:**
- Create: `src/boulder_v1/runtime.py`
- Create: `scripts/run_demo.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Produces: `mujoco_available()`, `compile_model(xml)`, `smoke_step(xml, steps)`.

- [x] Add missing-dependency behavior test.
- [x] Implement lazy MuJoCo import and compile/step helpers.
- [x] Add optional compile smoke test that skips when MuJoCo is absent.
- [x] Run full deterministic suite.
- [ ] Record real MuJoCo compile/step evidence (blocked in this environment because the `mujoco` package is unavailable and network installation is disabled).

### Task 5: Demo artifacts and documentation

**Files:**
- Create: `src/boulder_v1/visualize.py`
- Create: `README.md`
- Create: `requirements.txt`
- Create outputs under `outputs/`.

**Interfaces:**
- Produces: profile comparison image, canonical scene JSON, profile-specific MJCF files.

- [x] Render two profile variants from the same scene.
- [x] Export long/short profile MJCF and scene JSON.
- [x] Document exactly what is and is not demonstrated.
- [x] Run full tests and package the prototype.
