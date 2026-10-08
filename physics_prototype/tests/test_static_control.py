"""Stage3 static control on the actual, unmodified compiled Stage2 fixture."""

import copy
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import static_control
from boulder_v1.contact import CAPTURE_DISTANCE, CAPTURE_ORIENTATION, CAPTURE_SPEED
from boulder_v1.contact_benchmarks import _model_proof, make_mixed_fixture
from boulder_v1.contact_geometry import ContactMode, canonical_geometry
from boulder_v1.grasp import GraspManager, InitialContactError
from boulder_v1.locomotion import StateSummary
from boulder_v1.runtime import compute_pose_control
from boulder_v1.schema import Limb
from boulder_v1.static_state import ReadinessTracker, initialize_static_reference
from boulder_v1.support import FORCE_TOLERANCE, FootStatus, MAX_SLIP_SPEED, MIN_FOOT_LOAD


INTENT = {limb: limb.value.lower() for limb in Limb}
HANDS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)
FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
TIMESTEPS = (.002, .001)
PULSE = {"body": "climber_root", "start_s": 5., "duration_s": .1,
         "force_world_N": (2., 0., 0.)}
GAINS = {"waist": (160., 25.), "shoulder": (80., 12.), "elbow": (60., 6.),
         "wrist": (20., 1.), "hip": (160., 25.), "knee": (120., 12.), "ankle": (40., 2.)}


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, values, spec)
    return values


def model_arrays(model):
    return {f"{label}.{name}": value.copy()
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), np.ndarray)}


class NativeAudit:
    """Observe the authoritative integration boundary, not the detached callback."""

    def __init__(self, fixture):
        self.fixture = fixture
        self.before_arrays = model_arrays(fixture.model)
        self.violations = set()
        self.applied_loads = []
        self.external_forces = []
        self.joint_errors = []
        self.pre_guards = self.applied_guards = self.readiness_samples = 0
        self.controls = 0
        self.captures = []
        self.previous_qpos = fixture.reference.copy()
        self.previous_qvel = np.zeros(fixture.model.nv)
        self.previous_time = 0.
        self.manager = self.reference = None

    def record(self, condition, reason):
        if not condition:
            self.violations.add(reason)

    def control(self, original, model, data, *args, **kwargs):
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        command = original(model, data, *args, **kwargs)
        self.record(np.array_equal(data.qpos, qpos) and np.array_equal(data.qvel, qvel),
                    "control assigned generalized state")
        self.record(not args[1:] and not ({"kp", "kd", "target_velocity"} & kwargs.keys()),
                    "controller overrode physical class defaults")
        ff = kwargs.get("feedforward", {})
        if ff:
            allocation = static_control.estimate_static_feedforward(
                model, self.manager.scene, self.fixture.profile, self.reference) if self.controls == 1 else self.allocation
            self.allocation = allocation
            scale = min(1., float(data.time) / .2)
            self.record(all(abs(ff[name] - scale * value) <= 1e-12
                            for name, value in allocation["torques_Nm"].items()),
                        "feedforward did not use 0.2s startup ramp")
        elif self.controls > 0:
            self.record(False, "hold command omitted feedforward")
        for index, name in enumerate(command.joint_names):
            kind = name.removeprefix("left_").removeprefix("right_").split("_")[0]
            self.record((command.stiffness_Nm_rad[index], command.damping_Nms_rad[index]) == GAINS[kind],
                        "physical gains changed")
        self.record(np.allclose(data.ctrl * model.actuator_gear[:, 0], command.commanded_Nm,
                                rtol=0, atol=1e-12), "Nm command/normalized control mismatch")
        self.controls += 1
        return command

    def step(self, original, model, data, *args, **kwargs):
        self.record(np.array_equal(data.qpos, self.previous_qpos), "qpos changed between native steps")
        self.record(np.array_equal(data.qvel, self.previous_qvel), "qvel changed between native steps")
        self.record(data.time == self.previous_time, "clock changed between native steps")
        self.record(self.pre_guards == len(self.applied_loads) + 1, "missing native pre-step grasp guard")
        self.record(self.applied_guards == len(self.applied_loads), "missing applied grasp guard")
        self.record(self.readiness_samples == len(self.applied_loads), "missing authoritative readiness sample")
        root = model.body("climber_root").id
        other_forces = data.xfrc_applied.copy()
        other_forces[root, :3] = 0.
        self.record(not np.any(data.qfrc_applied) and not np.any(other_forces),
                    "undeclared generalized/body wrench injection")
        self.external_forces.append(tuple(float(v) for v in data.xfrc_applied[root, :3]))
        original(model, data, *args, **kwargs)
        self.record(abs(data.time - self.previous_time - model.opt.timestep) < 1e-12,
                    "native clock discontinuity")
        joints = model.actuator_trnid[:, 0]
        dofs = model.jnt_dofadr[joints]
        self.record(np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE), "non-hinge actuator")
        self.record(not np.any(data.qfrc_actuator[:6]), "actuator injected root wrench")
        self.record(np.allclose(data.qfrc_actuator[dofs], data.ctrl * model.actuator_gear[:, 0],
                                rtol=0, atol=1e-10), "native motor torque differs from command")
        loads = []
        for limb in HANDS:
            eq = self.manager.active_attachments()[limb].eq_id
            rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                  & (data.efc_id == eq))
            self.record(len(rows) == 3, "hand lacks three native Cartesian equality rows")
            loads.append(float(np.linalg.norm(data.efc_force[rows])))
        self.applied_loads.append(loads)
        qi = model.jnt_qposadr[joints]
        self.joint_errors.append(float(np.max(np.abs(np.asarray(self.reference.qpos)[qi] - data.qpos[qi]))))
        allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{hold}")) for limb, hold in INTENT.items()}
        for index, contact in enumerate(data.contact):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            pair = frozenset(model.geom(int(g)).name for g in (contact.geom1, contact.geom2))
            self.record(pair in allowed or np.linalg.norm(wrench[:3]) <= FORCE_TOLERANCE,
                        "hidden body/self/floor contact in native applied solve")
        self.record(all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc, data.efc_force)),
                    "nonfinite native state/forces")
        self.record(not any(w.number for w in data.warning), "native numerical warning/recovery")
        self.previous_qpos, self.previous_qvel = data.qpos.copy(), data.qvel.copy()
        self.previous_time = float(data.time)


class StaticControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.audits, cls.run_errors = {}, {}, {}
        for dt in TIMESTEPS:
            for disturbed in (False, True):
                key = (dt, disturbed)
                try:
                    cls.results[key], cls.audits[key] = cls.audited_benchmark(
                        dt, disturbance=copy.deepcopy(PULSE) if disturbed else None)
                except ValueError as error:
                    # Do not let one broken production admission hide the other
                    # timestep or independent controller safety tests.
                    cls.run_errors[key] = error
                    print("STATIC_METRICS " + json.dumps({"dt_s": dt, "disturbed": disturbed,
                          "error": f"{type(error).__name__}: {error}"}), flush=True)
                    continue
                result = cls.results[key]
                print("STATIC_METRICS " + json.dumps({
                    "dt_s": dt, "disturbed": disturbed, "status": result["status"], "steps": result["steps"],
                    "scored": result.get("scored"), "joint_rms_rad_s": result.get("joint_rms_rad_s"),
                    "feet": result.get("feet"), "max_applied_hand_load_N": result.get("max_applied_hand_load_N"),
                    "actuator_utilization_max": result.get("actuator_utilization_max"),
                    "saturation_fraction": result.get("saturation_fraction"), "recovery": result.get("recovery")},
                    allow_nan=False), flush=True)

    @staticmethod
    def audited_benchmark(dt, *, disturbance=None, observer=None):
        audits = []
        native_step, native_control = mujoco.mj_step, static_control.compute_pose_control
        native_guard, native_sample = GraspManager.evaluate_and_update, ReadinessTracker.sample_after_step
        native_attach = GraspManager.attach
        with ExitStack() as stack:
            def factory(timestep):
                fixture = make_mixed_fixture(timestep)
                audits.append(NativeAudit(fixture))
                return fixture

            def initialize(model, data, scene, profile, seed, intent):
                audit = audits[-1]

                def attach(manager, limb, region, force=False):
                    qpos, qvel = manager.data.qpos.copy(), manager.data.qvel.copy()
                    measured = manager._capture_measurement(limb, region)
                    accepted = native_attach(manager, limb, region, force=force)
                    audit.captures.append((limb, force, accepted, measured,
                                           np.array_equal(qpos, manager.data.qpos)
                                           and np.array_equal(qvel, manager.data.qvel)))
                    return accepted

                with patch.object(GraspManager, "attach", attach):
                    reference, manager = initialize_static_reference(model, data, scene, profile, seed, intent)
                audit.reference, audit.manager = reference, manager
                audit.seed_after_initializer = data.qpos.copy()
                audit.initial_arrays = model_arrays(model)
                # Initialization may reset; the entire subsequent operation may not.
                audit.reset = stack.enter_context(patch("mujoco.mj_resetData", side_effect=AssertionError(
                    "static hold called mj_resetData after initialization")))
                return reference, manager

            def step(model, data, *args, **kwargs):
                audit = audits[-1]
                if data is audit.fixture.data:
                    return audit.step(native_step, model, data, *args, **kwargs)
                return native_step(model, data, *args, **kwargs)

            def control(model, data, *args, **kwargs):
                audit = audits[-1]
                if data is audit.fixture.data:
                    return audit.control(native_control, model, data, *args, **kwargs)
                return native_control(model, data, *args, **kwargs)

            def guard(manager, *args, **kwargs):
                audit = audits[-1]
                if manager is audit.manager:
                    if kwargs.get("applied_data") is audit.fixture.data:
                        audit.applied_guards += 1
                    else:
                        audit.pre_guards += 1
                return native_guard(manager, *args, **kwargs)

            def sample(tracker, before):
                audit = audits[-1]
                if tracker.data is audit.fixture.data:
                    audit.record(audit.applied_guards == len(audit.applied_loads),
                                 "readiness sampled before applied grasp guard")
                    audit.readiness_samples += 1
                return native_sample(tracker, before)

            stack.enter_context(patch("boulder_v1.contact_benchmarks.make_mixed_fixture", side_effect=factory))
            initializer = stack.enter_context(patch("boulder_v1.static_state.initialize_static_reference", side_effect=initialize))
            stack.enter_context(patch("mujoco.mj_step", side_effect=step))
            stack.enter_context(patch("boulder_v1.static_control.compute_pose_control", side_effect=control))
            stack.enter_context(patch.object(GraspManager, "evaluate_and_update", guard))
            stack.enter_context(patch.object(ReadinessTracker, "sample_after_step", sample))
            result = static_control.run_static_benchmark(dt, disturbance=disturbance, observer=observer)
            audit = audits[-1]
            audit.record(len(audits) == 1 and initializer.call_count == 1, "benchmark did not reuse exactly one Stage2 fixture")
            audit.record(np.array_equal(audit.fixture.data.qpos, audit.previous_qpos)
                         and np.array_equal(audit.fixture.data.qvel, audit.previous_qvel),
                         "generalized state assigned after final native step")
            audit.after_arrays = model_arrays(audit.fixture.model)
        return result, audit

    def test_all_four_benchmarks_admit_the_requested_protocol(self):
        for dt in TIMESTEPS:
            for disturbed in (False, True):
                with self.subTest(dt=dt, disturbed=disturbed):
                    self.assertNotIn((dt, disturbed), self.run_errors,
                                     f"requested benchmark rejected: {self.run_errors.get((dt, disturbed))}")
                    self.assertIn((dt, disturbed), self.results)

    def test_protocol_full_native_steps_and_state_summaries(self):
        for key, result in self.results.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                self.assertIsInstance(result, dict)
                self.assertEqual(result["status"], "SUCCESS", result["reason"])
                self.assertTrue(result["success"])
                self.assertEqual(result["steps"], 5500 if key[0] == .002 else 11000)
                self.assertEqual(len(result["samples"]), result["steps"])
                self.assertAlmostEqual(result["duration_s"], 11., places=9)
                self.assertEqual((result["settle_s"], result["controlled_hold_s"], result["scored_window_s"]), (1., 10., 2.))
                self.assertEqual(result["dt_s"], key[0])
                for label in ("initial_state", "final_state"):
                    state = result[label]
                    self.assertIsInstance(state, StateSummary)
                    self.assertTrue(state.finite)
                    self.assertEqual(state.contact_mode, ContactMode.PHYSICAL)
                initial, final = result["initial_state"], result["final_state"]
                self.assertEqual(initial.time, 0.)
                np.testing.assert_array_equal(initial.qpos, self.audits[key].fixture.reference)
                np.testing.assert_array_equal(initial.qvel, 0.)
                self.assertAlmostEqual(final.time - initial.time, 11., places=9)
                self.assertEqual(final.contact_configuration, INTENT)
                self.assertNotEqual(final.qpos, initial.qpos)
                rows = result["samples"]
                np.testing.assert_allclose([r["elapsed_s"] for r in rows],
                                           np.arange(1, len(rows) + 1) * key[0], rtol=0, atol=2e-9)
                np.testing.assert_allclose([r["command_interval_start_s"] for r in rows],
                                           np.arange(len(rows)) * key[0], rtol=0, atol=2e-9)

    def test_final_two_seconds_sustained_strict_readiness(self):
        thresholds = {"root_linear_m_s": .02, "root_angular_rad_s": .05, "joint_max_rad_s": .10}
        for key, result in self.results.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                rows = result["samples"]
                scored = [r for r in rows if r["elapsed_s"] >= 9. - 1e-9]
                self.assertGreaterEqual(len(scored), round(2. / key[0]))
                self.assertFalse(rows[0]["readiness"]["ready"])
                self.assertEqual(rows[0]["readiness"]["duration"], 0.)
                for field, threshold in thresholds.items():
                    self.assertLessEqual(max(r[field] for r in scored), threshold)
                    self.assertEqual(result["scored"][field], max(r[field] for r in scored))
                self.assertTrue(all(r["readiness"]["ready"] and r["readiness"]["duration"] >= .5 - 1e-12 for r in scored))
                self.assertTrue(result["readiness"]["ready"])
                self.assertTrue(result["controller_convergence"])
                self.assertTrue(result["contact_acceptance"])
                ready = next(r for r in rows if r["readiness"]["ready"])
                self.assertGreaterEqual(ready["elapsed_s"], .5 + key[0] - 1e-12)
                self.assertEqual(result["scored"]["joint_error_max_rad"], max(r["joint_error_max_rad"] for r in scored))
                self.assertAlmostEqual(result["joint_rms_rad_s"],
                                       np.sqrt(np.mean([r["joint_rms_rad_s"] ** 2 for r in scored])), places=14)

    def test_every_endpoint_supports_both_feet_without_slip_from_first_step(self):
        for key, result in self.results.items():
            for limb in FEET:
                with self.subTest(dt=key[0], disturbed=key[1], limb=limb):
                    feet = [row["feet"][limb.value] for row in result["samples"]]
                    self.assertTrue(feet)
                    self.assertTrue(all(f["measurement_valid"] and f["contacting"] and f["supporting"]
                                        and not f["slipping"] and f["status"] == FootStatus.SUPPORTING for f in feet))
                    self.assertTrue(all(f["normal_force"] > 5. and f["tangential_speed"] <= .01 for f in feet))
                    self.assertTrue(all(f["support_surfaces"] == (f"geom_{INTENT[limb]}",)
                                        and f["support_regions"] == (INTENT[limb],) for f in feet))
                    for foot in feet:
                        for contact in foot["contacts"]:
                            if contact["normal_force"] > FORCE_TOLERANCE:
                                self.assertTrue(contact["admissible"])
                                self.assertGreaterEqual(contact["sole_alignment"], .9)
                                self.assertGreaterEqual(contact["surface_alignment"], .9)
                                self.assertLessEqual(contact["tangential_speed"], .01)
                                self.assertLessEqual(contact["tangential_force"],
                                                     contact["friction"] * contact["normal_force"] + FORCE_TOLERANCE)
                    self.assertEqual(result["feet"][limb.value]["support_fraction"], 1.)
                    self.assertEqual(result["feet"][limb.value]["slip_fraction"], 0.)

    def test_bounded_native_hands_and_all_run_applied_peak_not_capture_window(self):
        for key, result in self.results.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                self.assertEqual(result["release_events"], [])
                self.assertEqual(result["final_state"].release_events, ())
                peaks = np.max(self.audits[key].applied_loads, axis=0)
                np.testing.assert_allclose(result["max_applied_hand_load_N"], peaks, rtol=0, atol=1e-10)
                for index, limb in enumerate(HANDS):
                    hands = [r["hands"][limb.value] for r in result["samples"]]
                    self.assertTrue(all(h["active"] and h["valid"] and h["measurement_valid"] for h in hands))
                    self.assertTrue(all(h["load"] <= h["capacity"] and h["penetration"] < .001 for h in hands))
                    self.assertLessEqual(peaks[index], hands[0]["capacity"])
                    np.testing.assert_allclose([h["load"] for h in hands],
                                               np.linalg.norm([h["force_world"] for h in hands], axis=1), rtol=0, atol=1e-10)
                self.assertEqual(self.audits[key].pre_guards, result["steps"])
                self.assertEqual(self.audits[key].applied_guards, result["steps"])
                self.assertEqual(self.audits[key].readiness_samples, result["steps"])

    def test_native_clock_state_continuity_no_reset_or_hidden_body_support(self):
        for key, audit in self.audits.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                self.assertEqual(audit.violations, set())
                audit.reset.assert_not_called()
                self.assertTrue(all(not r["unintended_contacts"] for r in self.results[key]["samples"]))
                self.assertFalse(np.any(audit.fixture.data.qfrc_applied))
                self.assertFalse(np.any(audit.fixture.data.xfrc_applied))
                if not key[1]:
                    np.testing.assert_array_equal(audit.external_forces, 0.)

    def test_seed_canonical_completed_immutable_scene_and_strict_captures(self):
        for key, audit in self.audits.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                fixture, manager, reference = audit.fixture, audit.manager, audit.reference
                self.assertIsNot(manager.scene, fixture.scene)
                self.assertEqual(dict(fixture.scene.start_configuration), {})
                self.assertEqual(dict(manager.scene.start_configuration), INTENT)
                self.assertEqual(manager.scene.contact_regions, fixture.scene.contact_regions)
                self.assertEqual(manager.scene.walls, fixture.scene.walls)
                manager.require_session(fixture.model, fixture.data, manager.scene)
                np.testing.assert_array_equal(reference.qpos, fixture.reference)
                np.testing.assert_array_equal(audit.seed_after_initializer, fixture.reference)
                with self.assertRaises(FrozenInstanceError):
                    manager.scene.scale = 2.
                with self.assertRaises(TypeError):
                    manager.scene.start_configuration[Limb.LEFT_FOOT] = "right_foot"
                with self.assertRaises(TypeError):
                    reference.target_pose["left_elbow"] = 0.
                # initialize_episode also preflights its own nested candidates.
                self.assertGreaterEqual(len(audit.captures), 4)
                for limb, forced, accepted, measured, unchanged in audit.captures:
                    self.assertIn(limb, HANDS)
                    self.assertFalse(forced)
                    self.assertTrue(accepted and unchanged)
                    self.assertLessEqual(measured["gap_m"], CAPTURE_DISTANCE + 1e-12)
                    self.assertLessEqual(measured["relative_speed_m_s"], CAPTURE_SPEED + 1e-12)
                    self.assertGreaterEqual(measured["orientation"], CAPTURE_ORIENTATION)
                    self.assertLess(measured["penetration_m"], .001)
                    self.assertLessEqual(measured["signed_geom_distance_m"], .001 + 1e-12)
                events = self.results[key]["capture_events"]
                self.assertEqual({e["limb"] for e in events}, {limb.value for limb in HANDS})
                self.assertEqual(len(events), 2)
                for event in events:
                    self.assertEqual(event["mode"], ContactMode.PHYSICAL.value)
                    self.assertEqual(event["time_s"], 0.)
                    self.assertFalse(event["window_truncated"])
                    self.assertLessEqual(event["gap_m"], .001 + 1e-12)
                    self.assertGreaterEqual(event["orientation"], CAPTURE_ORIENTATION)
                    self.assertLessEqual(event["relative_speed_m_s"], .05 + 1e-12)

    def test_all_compiled_model_arrays_and_recorded_proof_equal_stage2_fixture(self):
        for key, audit in self.audits.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                for field, values in audit.before_arrays.items():
                    np.testing.assert_array_equal(audit.initial_arrays[field], values, err_msg=f"initializer changed {field}")
                    np.testing.assert_array_equal(audit.after_arrays[field], values, err_msg=f"controller changed {field}")
                result = self.results[key]
                self.assertEqual(result["model_proof"], _model_proof(audit.fixture))
                proof = result["model_proof"]
                self.assertEqual((proof["nq"], proof["nv"], proof["nu"]), (32, 31, 25))
                self.assertEqual(proof["contact_mode_numeric"], 0.)
                self.assertEqual(proof["foot_equalities"], 0)
                self.assertEqual(proof["gravity_m_s2"], [0., 0., -9.81])
                model = audit.fixture.model
                foot_bodies = {model.body(limb.value.lower()).id for limb in FEET}
                for eq in range(model.neq):
                    self.assertNotIn("foot", model.equality(eq).name.lower())
                    self.assertEqual(model.eq_objtype[eq], mujoco.mjtObj.mjOBJ_SITE)
                    self.assertFalse({int(model.site_bodyid[model.eq_obj1id[eq]]),
                                      int(model.site_bodyid[model.eq_obj2id[eq]])} & foot_bodies)
                for region in audit.fixture.scene.contact_regions:
                    geometry = canonical_geometry(region)
                    geom = model.geom(f"geom_{region.id}")
                    body = model.body(int(geom.bodyid[0]))
                    np.testing.assert_allclose(body.pos, geometry.body_frame.position, rtol=0, atol=1e-14)
                    np.testing.assert_allclose(body.quat, geometry.body_frame.quaternion, rtol=0, atol=1e-14)
                    np.testing.assert_array_equal(geom.size[:len(geometry.size)], geometry.size)

    def test_sample_command_units_joint_velocities_errors_and_readiness_agree(self):
        for key, result in self.results.items():
            with self.subTest(dt=key[0], disturbed=key[1]):
                model = self.audits[key].fixture.model
                ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                for row in result["samples"]:
                    command = row["command"]
                    velocity = np.array(row["joint_velocity_rad_s"])
                    self.assertEqual(len(velocity), model.nu)
                    self.assertAlmostEqual(row["joint_max_rad_s"], np.max(np.abs(velocity)), places=14)
                    self.assertAlmostEqual(row["joint_rms_rad_s"], np.sqrt(np.mean(velocity ** 2)), places=14)
                    self.assertAlmostEqual(row["readiness"]["time"], row["time_s"], places=14)
                    self.assertEqual(row["readiness"]["root_linear_speed"], row["root_linear_m_s"])
                    self.assertEqual(row["readiness"]["root_angular_speed"], row["root_angular_rad_s"])
                    self.assertEqual(row["readiness"]["max_hinge_speed"], row["joint_max_rad_s"])
                    self.assertTrue(np.isfinite(row["joint_error_max_rad"]))
                    np.testing.assert_array_equal(command["limits_Nm"], ceiling)
                    np.testing.assert_allclose(command["commanded_Nm"], np.clip(command["desired_Nm"], -ceiling, ceiling), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(command["utilization"], np.abs(command["commanded_Nm"]) / ceiling, rtol=0, atol=1e-12)
                self.assertLessEqual(result["actuator_utilization_max"], 1.)
                self.assertEqual(result["saturation_fraction"], 0.)
                np.testing.assert_array_equal([r["joint_error_max_rad"] for r in result["samples"]],
                                              self.audits[key].joint_errors)

    def test_feedforward_least_squares_balance_at_actual_foot_overlap_centers(self):
        for dt in TIMESTEPS:
            with self.subTest(dt=dt):
                audit = self.audits[dt, False]
                model, scene, profile, reference = audit.fixture.model, audit.manager.scene, audit.fixture.profile, audit.reference
                before = integration_state(model, audit.fixture.data)
                allocation = static_control.estimate_static_feedforward(model, scene, profile, reference)
                np.testing.assert_array_equal(integration_state(model, audit.fixture.data), before)
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = reference.qpos
                scratch.eq_active[:] = False
                mujoco.mj_forward(model, scratch)
                blocks, points = [], []
                for limb in (*FEET, *HANDS):
                    if limb.is_foot:
                        shoe, hold = model.geom(f"{limb.value.lower()}_geom"), model.geom(f"geom_{INTENT[limb]}")
                        lo = np.maximum(scratch.geom_xpos[shoe.id, :2] - shoe.size[:2], scratch.geom_xpos[hold.id, :2] - hold.size[:2])
                        hi = np.minimum(scratch.geom_xpos[shoe.id, :2] + shoe.size[:2], scratch.geom_xpos[hold.id, :2] + hold.size[:2])
                        self.assertTrue(np.all(hi > lo))
                        point = np.r_[(lo + hi) / 2, scratch.geom_xpos[hold.id, 2] + hold.size[2]]
                        body = int(shoe.bodyid[0])
                        self.assertGreater(np.linalg.norm(point - scratch.site(f"{limb.value.lower()}_site").xpos), .001)
                    else:
                        site = scratch.site(f"{limb.value.lower()}_site")
                        point, body = site.xpos.copy(), int(model.site_bodyid[site.id])
                    points.append(point)
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jac(model, scratch, jp, jr, point, body)
                    blocks.append(jp.T)
                jacobian = np.concatenate(blocks, axis=1)
                weight = -model.body_subtreemass[model.body("climber_root").id] * model.opt.gravity
                nominal = np.concatenate([share * weight for share in (.45, .45, .05, .05)])
                forces = nominal + np.linalg.lstsq(jacobian[:6], scratch.qfrc_bias[:6] - jacobian[:6] @ nominal, rcond=None)[0]
                np.testing.assert_allclose(allocation["support_points_world_m"], points, rtol=0, atol=1e-14)
                np.testing.assert_allclose(np.ravel(allocation["planned_forces_world_N"]), forces, rtol=0, atol=1e-10)
                np.testing.assert_allclose(jacobian[:6] @ forces, scratch.qfrc_bias[:6], rtol=0, atol=1e-8)
                joints = model.actuator_trnid[:, 0]
                dofs = model.jnt_dofadr[joints]
                torque = scratch.qfrc_bias[dofs] - (jacobian @ forces)[dofs]
                np.testing.assert_allclose([allocation["torques_Nm"][model.joint(int(j)).name] for j in joints], torque, rtol=0, atol=1e-10)
                self.assertLessEqual(allocation["max_root_balance_residual"], 1e-8)
                self.assertTrue(np.all(np.abs(torque) <= model.actuator_gear[:, 0]))

    def test_declared_disturbance_actual_native_forces_response_and_recovery(self):
        for dt in TIMESTEPS:
            with self.subTest(dt=dt):
                self.assertNotIn((dt, True), self.run_errors,
                                 f"climber_root disturbance rejected: {self.run_errors.get((dt, True))}")
                nominal, disturbed = self.results[dt, False], self.results[dt, True]
                audit = self.audits[dt, True]
                self.assertEqual(disturbed["disturbance"], PULSE)
                self.assertIsNone(nominal["disturbance"])
                self.assertNotIn("recovery", nominal)
                starts = np.array([r["command_interval_start_s"] for r in disturbed["samples"]])
                indices = np.array([r["interval_index"] for r in disturbed["samples"]])
                begin = round(PULSE["start_s"] / dt)
                end = begin + round(PULSE["duration_s"] / dt)
                active = (indices >= begin) & (indices < end)
                expected = np.zeros((len(starts), 3))
                expected[active] = PULSE["force_world_N"]
                np.testing.assert_array_equal(audit.external_forces, expected)
                np.testing.assert_array_equal([r["disturbance_active"] for r in disturbed["samples"]], active)
                self.assertEqual(int(active.sum()), round(PULSE["duration_s"] / dt))
                np.testing.assert_allclose(np.sum(audit.external_forces, axis=0) * dt, [.2, 0., 0.], rtol=0, atol=1e-12)
                first = int(np.flatnonzero(active)[0])
                self.assertEqual(nominal["samples"][:first], disturbed["samples"][:first])
                response = [r["root_linear_m_s"] for r in disturbed["samples"][first:first + round(.5 / dt)]]
                baseline = [r["root_linear_m_s"] for r in nominal["samples"][first:first + round(.5 / dt)]]
                self.assertGreater(np.max(np.abs(np.array(response) - baseline)), 1e-5)
                recovery = disturbed["recovery"]["ready_after_pulse_s"]
                self.assertIsNotNone(recovery)
                self.assertGreaterEqual(recovery, 5.1)
                self.assertLess(recovery, 9.)
                scored = disturbed["scored"]
                for field in ("root_linear_m_s", "root_angular_rad_s", "joint_max_rad_s", "joint_error_max_rad"):
                    self.assertLess(abs(scored[field] - nominal["scored"][field]), .005,
                                    f"pulse did not return to nominal endpoint metric {field}")
                np.testing.assert_allclose(disturbed["final_state"].qpos, nominal["final_state"].qpos, rtol=0, atol=.002)
                np.testing.assert_allclose(disturbed["final_state"].qvel, nominal["final_state"].qvel, rtol=0, atol=.005)

    def test_adversarial_observer_cannot_mutate_authoritative_result(self):
        observations = []

        def observer(row, model, data):
            observations.append(row["time_s"])
            self.assertIsNot(model, self.audits[.002, False].fixture.model)
            self.assertIsNot(data, self.audits[.002, False].fixture.data)
            row["readiness"]["ready"] = False
            row["command"]["commanded_Nm"] = (1e9,) * model.nu
            row["feet"][Limb.LEFT_FOOT.value]["supporting"] = False
            row["hands"][Limb.LEFT_HAND.value]["load"] = 1e9
            model.body_mass[:] = 1e6
            model.body_inertia[:] = 1e6
            model.geom_size[:] = 1e6
            model.pair_friction[:] = 0.
            model.eq_solref[:] = 1.
            model.actuator_gear[:] = 0.
            model.opt.gravity[:] = 0.
            model.opt.timestep = .1
            data.qpos[:] = 1e6
            data.qvel[:] = 1e6
            data.ctrl[:] = 1e6
            data.eq_active[:] = False
            data.xfrc_applied[:] = 1e6
            data.time = -1.

        observed, audit = self.audited_benchmark(.002, observer=observer)
        self.assertGreater(len(observations), 200)
        self.assertEqual(observed, self.results[.002, False])
        self.assertEqual(audit.violations, set())
        for field, values in audit.before_arrays.items():
            np.testing.assert_array_equal(audit.after_arrays[field], values, err_msg=field)

    def test_timestep_verdicts_identical_without_gain_or_contact_relaxation(self):
        self.assertEqual((MIN_FOOT_LOAD, MAX_SLIP_SPEED), (5., .01))
        self.assertEqual({status.value for status in static_control.StaticStatus}, {
            "SUCCESS", "REFERENCE_INVALID", "TORQUE_LIMITED", "CONTACT_LOSS", "GRASP_OVERLOAD",
            "STABILIZATION_TIMEOUT", "NONFINITE_STATE"})
        for disturbed in (False, True):
            with self.subTest(disturbed=disturbed):
                self.assertNotIn((.002, disturbed), self.run_errors,
                                 f"coarse benchmark rejected: {self.run_errors.get((.002, disturbed))}")
                self.assertNotIn((.001, disturbed), self.run_errors,
                                 f"fine benchmark rejected: {self.run_errors.get((.001, disturbed))}")
                coarse, fine = self.results[.002, disturbed], self.results[.001, disturbed]
                for field in ("status", "success", "contact_acceptance", "controller_convergence"):
                    self.assertEqual(coarse[field], fine[field], (disturbed, field))
                self.assertEqual(coarse["status"], "SUCCESS")

    def test_explicit_invalid_reference_rejected_without_live_mutation(self):
        fixture = make_mixed_fixture()
        reference, manager = initialize_static_reference(fixture.model, fixture.data, fixture.scene,
                                                         fixture.profile, fixture.reference, INTENT)
        invalid_qpos = list(reference.qpos)
        joint = fixture.model.joint("left_elbow").id
        invalid_qpos[int(fixture.model.jnt_qposadr[joint])] = fixture.model.jnt_range[joint, 1] + .01
        invalid_target = dict(reference.target_pose)
        invalid_target["right_ankle_roll"] = np.nan
        invalid_intent = {**INTENT, Limb.LEFT_HAND: "right_hand"}
        invalid = (replace(reference, qpos=invalid_qpos), replace(reference, target_pose=invalid_target),
                   replace(reference, contact_intent=invalid_intent))
        fixture.data.ctrl[:] = .123
        for bad in invalid:
            with self.subTest(reference=bad):
                before = integration_state(fixture.model, fixture.data)
                arrays = model_arrays(fixture.model)
                captures, releases = copy.deepcopy(manager.capture_events), copy.deepcopy(manager.releases)
                with patch("mujoco.mj_resetData", side_effect=AssertionError("invalid reference reset live data")) as reset, \
                        patch("mujoco.mj_step", side_effect=AssertionError("invalid reference stepped live data")) as step:
                    result = static_control.execute_static_hold(fixture.model, fixture.data, manager.scene,
                                                               fixture.profile, bad, manager)
                self.assertEqual(result["status"], "REFERENCE_INVALID")
                self.assertFalse(result["success"] or result["contact_acceptance"] or result["controller_convergence"])
                self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
                self.assertEqual(result["initial_state"], result["final_state"])
                reset.assert_not_called()
                step.assert_not_called()
                np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)
                self.assertEqual(manager.capture_events, captures)
                self.assertEqual(manager.releases, releases)
                after_arrays = model_arrays(fixture.model)
                for field, values in arrays.items():
                    np.testing.assert_array_equal(after_arrays[field], values, err_msg=field)
        before = integration_state(fixture.model, fixture.data)
        with self.assertRaises((InitialContactError, ValueError)):
            initialize_static_reference(fixture.model, fixture.data, fixture.scene, fixture.profile,
                                        invalid_qpos, INTENT)
        np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)

    def test_forced_native_hand_overloads_detach_and_stop_at_each_guard(self):
        for phase in ("pre", "applied", "endpoint"):
            with self.subTest(phase=phase):
                fixture = make_mixed_fixture()
                model, data = fixture.model, fixture.data
                reference, manager = initialize_static_reference(model, data, fixture.scene, fixture.profile,
                                                                 fixture.reference, INTENT)
                original = manager._reaction
                capacity = fixture.profile.grip_capacity

                def overloaded(limb, measured_data):
                    forced = phase == "pre" or (measured_data.time > 0.
                             and ((phase == "applied" and measured_data is data)
                                  or (phase == "endpoint" and measured_data is not data)))
                    return (2. * capacity, 0., 0.) if forced and limb == Limb.LEFT_HAND else original(limb, measured_data)

                native = mujoco.mj_step
                with patch.object(manager, "_reaction", side_effect=overloaded), \
                        patch("mujoco.mj_resetData", side_effect=AssertionError("overload reset")), \
                        patch("mujoco.mj_step", wraps=native) as step:
                    result = static_control.execute_static_hold(model, data, manager.scene, fixture.profile,
                                                               reference, manager, duration=2., settle=1., score_window=.5)
                self.assertEqual(result["status"], "GRASP_OVERLOAD")
                self.assertFalse(result["success"] or result["contact_acceptance"])
                expected_steps = 0 if phase == "pre" else 1
                self.assertEqual(result["steps"], expected_steps)
                self.assertEqual(step.call_count, expected_steps)
                self.assertAlmostEqual(data.time, expected_steps * model.opt.timestep, places=14)
                self.assertFalse(manager.is_attached(Limb.LEFT_HAND))
                self.assertTrue(manager.is_attached(Limb.RIGHT_HAND))
                self.assertEqual(len(result["release_events"]), 1)
                release = result["release_events"][0]
                self.assertEqual(release["limb"], Limb.LEFT_HAND.value)
                self.assertGreater(release["required_load_N"], release["capacity_N"])
                if phase == "pre":
                    np.testing.assert_array_equal(data.qpos, reference.qpos)
                    np.testing.assert_array_equal(data.qvel, 0.)
                if phase == "applied":
                    self.assertEqual(result["max_applied_hand_load_N"][0], 2. * capacity)

    def test_real_force_induced_slipping_foot_cannot_pass_contact_acceptance(self):
        fixture = make_mixed_fixture()
        model, data = fixture.model, fixture.data
        reference, manager = initialize_static_reference(model, data, fixture.scene, fixture.profile,
                                                         fixture.reference, INTENT)
        # This stronger fault pulse acts directly on a massive shoe; the required
        # root-body disturbance is covered separately without substituting it.
        pulse = {"body": "left_foot", "start_s": 1.1, "duration_s": .02, "force_world_N": (400., 0., 0.)}
        with patch("mujoco.mj_resetData", side_effect=AssertionError("slip reset")):
            result = static_control.execute_static_hold(model, data, manager.scene, fixture.profile,
                                                       reference, manager, duration=2., settle=1., score_window=.5,
                                                       disturbance=pulse)
        self.assertTrue(any(f.slipping for f in manager.contact_snapshot().feet.values()), result["reason"])
        self.assertEqual(result["status"], "CONTACT_LOSS")
        self.assertFalse(result["contact_acceptance"] or result["controller_convergence"] or result["success"])
        self.assertLess(result["steps"], 1000)
        self.assertFalse(np.any(data.xfrc_applied))

    def test_last_endpoint_slip_failure_is_not_overwritten_by_completion(self):
        for already_settled in (False, True):
            with self.subTest(already_settled=already_settled):
                fixture = make_mixed_fixture()
                model, data = fixture.model, fixture.data
                reference, manager = initialize_static_reference(model, data, fixture.scene, fixture.profile,
                                                                 fixture.reference, INTENT)
                if already_settled:
                    # Continue a detached copy of the real cached nominal state,
                    # without synthesizing qpos/qvel or mocking readiness.
                    mujoco.mj_copyData(data, model, self.audits[.002, False].fixture.data)
                duration = 2. if already_settled else .202
                end = float(data.time) + duration
                original = manager.contact_snapshot

                def final_slip():
                    snapshot = original()
                    if data.time >= end - 1e-12:
                        # Fault injection tightens only the last endpoint gate;
                        # all integration and sustained readiness remain native.
                        foot = snapshot.feet[Limb.LEFT_FOOT]
                        return replace(snapshot, feet={**snapshot.feet, Limb.LEFT_FOOT: replace(
                            foot, status=FootStatus.SLIPPING, supporting=False, slipping=True, tangential_speed=.02)})
                    return snapshot

                with patch.object(manager, "contact_snapshot", side_effect=final_slip), \
                        patch("mujoco.mj_resetData", side_effect=AssertionError("final endpoint reset")):
                    result = static_control.execute_static_hold(model, data, manager.scene, fixture.profile,
                                                               reference, manager, duration=duration,
                                                               settle=1. if already_settled else .2, score_window=.002)
                self.assertEqual(result["steps"], 1000 if already_settled else 101)
                self.assertTrue(result["final_state"].foot_states[Limb.LEFT_FOOT].slipping)
                self.assertEqual(result["status"], "CONTACT_LOSS", "a failed last endpoint must not become SUCCESS")
                self.assertFalse(result["success"] or result["controller_convergence"])
                self.assertFalse(result["contact_acceptance"], "completed step count must not admit a slipping final endpoint")


if __name__ == "__main__":
    unittest.main()
