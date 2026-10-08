"""Native Stage5 foot protocol and atomic boundary counterexamples."""
import copy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1 import foot_transfer as ft
from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.contact_ik import solve_foot_reference
from boulder_v1.grasp import GraspManager
from boulder_v1.hand_family import make_hand_family_fixture
from boulder_v1.locomotion import get_state_summary
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Affordance, Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import _integration_state, initialize_static_reference


COMPILED = ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat",
            "geom_friction", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
            "site_pos", "site_quat", "jnt_range", "jnt_axis", "dof_damping", "dof_armature",
            "actuator_gear", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
            "eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp")


def prepare(dt=.002, **kwargs):
    model, data, scene, profile, seed = ft.make_foot_transfer_fixture(dt, **kwargs)
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager, duration=2., settle=1.,
                                  score_window=.5, keep_samples=False)
    if not initial["success"]:
        raise AssertionError(initial["reason"])
    return model, data, scene, profile, reference, manager


def clone(source):
    original_model, original_data, scene, profile, reference, original_manager = source
    model = copy.copy(original_model)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, original_data)
    manager = GraspManager(model, data, scene, profile=profile)
    manager.synchronize_from_live()
    manager.capture_events = copy.deepcopy(original_manager.capture_events)
    manager.releases = copy.deepcopy(original_manager.releases)
    return model, data, scene, profile, reference, manager


class FootTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.positive = {}, {}
        for dt in (.002, .001):
            source = prepare(dt)
            cls.sources[dt] = source
            args = clone(source)
            model, data, scene, profile, reference, manager = args
            compiled = {n: getattr(model, n).copy() for n in COMPILED}
            expected = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy()}
            native_step, native_control = mujoco.mj_step, ft.compute_pose_control
            native_admit, native_ik = ft.initialize_static_reference, ft.solve_foot_reference
            callbacks = []
            leg = model.jnt_qposadr[[model.joint("left_" + n).id for n in
                                   ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]]
            frozen = np.ones(model.nq, dtype=bool)
            frozen[leg] = False

            def step(m, d):
                np.testing.assert_array_equal(d.qpos, expected["qpos"])
                np.testing.assert_array_equal(d.qvel, expected["qvel"])
                native_step(m, d)
                expected.update(qpos=d.qpos.copy(), qvel=d.qvel.copy())

            def control(m, d, *a, **kw):
                np.testing.assert_array_equal(d.qpos, expected["qpos"])
                np.testing.assert_array_equal(d.qvel, expected["qvel"])
                if "kp" in kw or "kd" in kw:
                    raise AssertionError("Foot execution changed default impedance")
                return native_control(m, d, *a, **kw)

            def admit(m, d, *a, **kw):
                if d is data:
                    raise AssertionError("Foot execution reset the live episode")
                return native_admit(m, d, *a, **kw)

            def ik(m, measured, limb, frame, **kw):
                np.testing.assert_array_equal(measured, data.qpos)
                solution = native_ik(m, measured, limb, frame, **kw)
                np.testing.assert_array_equal(np.array(solution.qpos)[frozen], data.qpos[frozen])
                validate_reference_pose(m, solution.qpos)
                return solution

            def observer(row, display_model, display_data):
                np.testing.assert_array_equal(row["qpos"], data.qpos)
                np.testing.assert_array_equal(row["qvel"], data.qvel)
                np.testing.assert_array_equal(display_data.qpos, data.qpos)
                callbacks.append(copy.deepcopy(row))
                # Deliberately corrupt every display channel, including physics.
                row["qpos"][0] = 999.
                display_model.body_mass[:] = 999.
                display_model.geom_friction[:] = .0001
                display_model.eq_data[:] = 999.
                display_model.opt.timestep = 1.
                display_data.qpos[:] = 999.
                display_data.qvel[:] = 999.
                display_data.ctrl[:] = 999.
                display_data.eq_active[:] = False
                display_data.xfrc_applied[:] = 999.

            with patch.object(ft, "compute_pose_control", side_effect=control), \
                    patch.object(ft, "initialize_static_reference", side_effect=admit), \
                    patch.object(ft, "solve_foot_reference", side_effect=ik), \
                    patch("mujoco.mj_step", side_effect=step):
                result = ft.execute_foot_transfer(*args, ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target"),
                                                  observer=observer)
            for n, values in compiled.items():
                np.testing.assert_array_equal(getattr(model, n), values, err_msg=n)
            if not result["success"]:
                raise AssertionError(f"dt={dt} phase={result['phases']} status={result['status']} "
                                     f"reason={result['reason']} guard={result['guard_failure']}")
            cls.positive[dt] = (result, args, callbacks)
            unload = [s for s in result["samples"] if s["phase"] == "UNLOAD"][-1]
            print(f"foot dt={dt}: steps={result['steps']} duration={result['duration_s']:.6f} "
                  f"release={result['release_time_s']:.6f} touchdown={result['touchdown']['time_s']:.6f} "
                  f"first_support={result['first_support']['time_s']:.6f} "
                  f"acquired={result['acquisition_time_s']:.6f} load_complete={result['load_complete_time_s']:.6f} "
                  f"final={data.time:.6f} hands={result['hand_load_max_N']} "
                  f"slip={result['foot_slip_max_m_s']} tracking={result['max_tracking_error_m']} "
                  f"unload_Fn=({unload['feet']['LEFT_FOOT']['normal_force']},"
                  f"{unload['feet']['RIGHT_FOOT']['normal_force']}) "
                  f"final_Fn={result['terminal_observation']['feet']['LEFT_FOOT']['normal_force']} "
                  f"motor_utilization={result['actuator_utilization_max']} "
                  f"cache_uses={result['landing_point_cache_uses']} three_point={result['three_point']}", flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.positive.clear()
        cls.sources.clear()

    def setUp(self):
        self.args = clone(self.sources[.002])
        self.model, self.data, self.scene, self.profile, self.reference, self.manager = self.args
        self.request = ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target")

    def reject_atomically(self, *, reference=None, request=None, fault=None):
        args = list(self.args)
        if reference is not None:
            args[4] = reference
        before = _integration_state(self.model, self.data)
        releases, captures = copy.deepcopy(self.manager.releases), copy.deepcopy(self.manager.capture_events)
        rows = []
        with patch.object(ft, "compute_pose_control", side_effect=AssertionError("rejected input commanded")), \
                patch("mujoco.mj_step", side_effect=AssertionError("rejected input integrated")):
            result = ft.execute_foot_transfer(*args, self.request if request is None else request, fault=fault,
                                              observer=lambda row, *_: rows.append(row))
        self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
        self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
        self.assertIsNone(result["final_reference"])
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        self.assertEqual(self.manager.releases, releases)
        self.assertEqual(self.manager.capture_events, captures)
        self.assertEqual(rows, [result["terminal_observation"]])
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        return result

    def test_native_protocol_acquisition_and_stationary_final_reference_both_timesteps(self):
        for dt, (result, args, callbacks) in self.positive.items():
            with self.subTest(dt=dt):
                model, data, scene, profile, _, manager = args
                self.assertEqual(result["phases"], ["SOURCE_STABILIZE", "UNLOAD", "LIFT", "THREE_POINT",
                                                    "REACH", "LANDING", "LOAD", "SETTLE"])
                self.assertEqual(result["grip_releases"], [])
                self.assertTrue(result["support_admission"]["admitted"])
                self.assertEqual(result["release"]["contact_count"], 0)
                self.assertEqual(result["release"]["normal_force_N"], 0.)
                self.assertGreater(result["release"]["signed_source_geom_distance_m"], 0.)
                self.assertGreater(result["touchdown"]["normal_force_N"], 0.)
                self.assertGreater(result["first_support"]["normal_force_N"], 5.)
                self.assertGreaterEqual(result["acquisition"]["sustained_s"], .1 - 1e-12)
                self.assertGreaterEqual(result["acquisition_time_s"], result["first_support"]["time_s"] + .1 - 1e-12)
                self.assertAlmostEqual(result["load_duration_s"], self.request.load_s)
                self.assertTrue(result["readiness"]["ready"])
                self.assertGreaterEqual(result["readiness"]["duration"], .5 - 1e-12)
                stopped = next(e["time_s"] for e in result["events"] if e.get("event") == "REFERENCE_STOPPED")
                self.assertGreaterEqual(data.time - stopped, .5 - 1e-12)
                reference = result["final_reference"]
                self.assertEqual(dict(reference.contact_intent), {**self.reference.contact_intent, Limb.LEFT_FOOT: "foot_target"})
                self.assertEqual(result["final_state"], get_state_summary(model, data, manager))
                np.testing.assert_allclose(result["terminal_observation"]["q_ref"], reference.qpos, atol=1e-10, rtol=0)
                np.testing.assert_allclose(result["terminal_observation"]["qd_ref"], 0., atol=1e-8, rtol=0)
                with self.assertRaises(TypeError):
                    reference.contact_intent[Limb.LEFT_FOOT] = "left_foot"
                initialize_static_reference(model, mujoco.MjData(model), replace(scene, start_configuration=reference.contact_intent),
                                            profile, reference.qpos, reference.contact_intent)
                self.assertEqual(callbacks[-1], result["terminal_observation"])
                self.assertTrue(callbacks[-1]["terminal"])
                self.assertFalse(any("foot" in model.equality(i).name for i in range(model.neq)))

    def test_all_native_samples_support_safety_and_no_hidden_forces(self):
        for dt, (result, args, _) in self.positive.items():
            with self.subTest(dt=dt):
                model = args[0]
                joints = model.actuator_trnid[:, 0]
                qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
                previous = np.array(args[4].qpos)
                velocity = np.zeros(model.nv)
                for sample in result["samples"]:
                    self.assertTrue(sample["feet"]["RIGHT_FOOT"]["supporting"])
                    self.assertFalse(sample["feet"]["RIGHT_FOOT"]["slipping"])
                    for hand in sample["hands"].values():
                        self.assertTrue(hand["active"] and hand["valid"])
                        self.assertLessEqual(hand["load"], hand["capacity"])
                        self.assertLessEqual(hand["load"], 850.)
                    for decision in sample["hand_decisions"].values():
                        self.assertTrue(decision["maintain"])
                        self.assertLessEqual(decision["required_load"], decision["effective_capacity"])
                    self.assertLessEqual(sample["root_linear_m_s"], .10)
                    self.assertLessEqual(sample["root_angular_rad_s"], .50)
                    self.assertLessEqual(sample["joint_max_rad_s"], 1.)
                    self.assertFalse(np.any(sample["external_force_world_N"]))
                    self.assertFalse(np.any(sample["qfrc_applied"]))
                    q, qd = np.array(sample["q_ref"]), np.array(sample["qd_ref"])
                    validate_reference_pose(model, q)
                    self.assertLessEqual(np.max(np.abs(qd[vi])), .5 + 1e-10)
                    self.assertLessEqual(np.max(np.abs(qd[vi] - velocity[vi])), 2. * dt + 1e-10)
                    np.testing.assert_allclose((q[qi] - previous[qi]) / dt, qd[vi], atol=1e-10, rtol=0)
                    previous, velocity = q, qd
                    if sample["released"]:
                        self.assertEqual(sample["source_contact"]["contact_count"], 0)
                        self.assertEqual(sample["source_contact"]["normal_force_N"], 0.)
                    if sample["phase"] in ("THREE_POINT", "REACH"):
                        self.assertFalse(sample["feet"]["LEFT_FOOT"]["contacting"])
                        self.assertNotIn("LEFT_FOOT", sample["contacts"])
                    if sample["phase"] == "UNLOAD":
                        self.assertNotIn("LEFT_FOOT", sample["declared_support_contacts"])
                    if sample["acquisition"] is not None:
                        self.assertEqual(sample["feet"]["LEFT_FOOT"]["support_surfaces"], ("geom_foot_target",))
                self.assertGreater(np.linalg.norm(np.array(result["final_state"].qpos) - result["initial_state"].qpos), .01)

    def test_fixture_preserves_stage4_body_and_has_initial_geometric_gap(self):
        model, data, scene, profile, seed = ft.make_foot_transfer_fixture()
        base, _, base_scene, _, _ = make_hand_family_fixture()
        self.assertEqual(scene.contact_regions[:-1], base_scene.contact_regions)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (base.nq, base.nv, base.nu, base.neq))
        for name in ("climber_root", "left_foot", "right_foot", "left_hand", "right_hand"):
            for field in ("mass", "inertia", "pos", "quat"):
                np.testing.assert_array_equal(getattr(model.body(name), field), getattr(base.body(name), field))
        for name in ("left_foot_geom", "right_foot_geom", "left_hand_geom", "right_hand_geom"):
            for field in ("size", "pos", "friction", "solref", "solimp"):
                np.testing.assert_array_equal(getattr(model.geom(name), field), getattr(base.geom(name), field))
        data.qpos[:] = seed
        mujoco.mj_forward(model, data)
        target = scene.region("foot_target")
        self.assertAlmostEqual(target.position[0] - scene.region("left_foot").position[0], -.105)
        self.assertAlmostEqual(target.position[2] - scene.region("left_foot").position[2], .02)
        self.assertAlmostEqual(target.position[0] + target.half_size[0], -.1725)
        self.assertAlmostEqual(data.geom("left_foot_geom").xpos[0] - model.geom("left_foot_geom").size[0], -.1675)
        self.assertGreater(mujoco.mj_geomDistance(model, data, model.geom("left_foot_geom").id,
                                                model.geom("geom_foot_target").id, 1., None), 0.)
        solution = solve_foot_reference(model, seed, Limb.LEFT_FOOT, canonical_geometry(target).foot_frame)
        self.assertTrue(solution.converged, solution.reason)
        validate_reference_pose(model, solution.qpos)

    def test_bad_full_references_are_atomic(self):
        seed = np.array(self.reference.qpos)
        nonunit = seed.copy(); nonunit[3:7] *= 2.
        nonfinite = seed.copy(); nonfinite[0] = math.nan
        outside = seed.copy(); outside[self.model.joint("left_knee").qposadr[0]] = 10.
        wrong_pose = dict(self.reference.target_pose); wrong_pose["left_knee"] += .01
        bad = (replace(self.reference, qpos=nonunit), replace(self.reference, qpos=nonfinite),
               replace(self.reference, qpos=outside), replace(self.reference, qpos=seed[:-1]),
               replace(self.reference, target_pose=wrong_pose),
               replace(self.reference, contact_intent={l.value: h for l, h in self.reference.contact_intent.items()}),
               replace(self.reference, contact_intent={Limb.LEFT_FOOT: "left_foot"}),
               SimpleNamespace(qpos=seed[:, None], contact_intent={}, target_pose={}), object())
        for reference in bad:
            with self.subTest(reference=reference):
                result = self.reject_atomically(reference=reference)
                self.assertIn(result["status"], ("INITIALIZATION_FAILURE", "ROM_FAILURE"))
                self.assertFalse(result["terminal_observation"]["source_reference_admitted"])
                self.assertIsNone(result["terminal_observation"]["q_ref"])

    def test_bad_requests_and_external_forces_are_atomic(self):
        bad = (object(), replace(self.request, limb=Limb.LEFT_HAND), replace(self.request, target=[]),
               replace(self.request, target="missing"), replace(self.request, source="right_foot"),
               replace(self.request, unload_s=True), replace(self.request, lift_s=None),
               replace(self.request, load_s=.0025), replace(self.request, lift_m=".03"))
        for request in bad:
            with self.subTest(request=request):
                self.assertEqual(self.reject_atomically(request=request)["status"], "REACH_INFEASIBLE")
        for field, index in (("qfrc_applied", 5), ("xfrc_applied", (self.model.body("climber_root").id, 5))):
            getattr(self.data, field)[index] = .125
            self.assertIn("undeclared", self.reject_atomically()["reason"])
            self.assertEqual(getattr(self.data, field)[index], .125)
            getattr(self.data, field)[index] = 0.

    def test_target_without_step_is_ineligible_before_motion(self):
        self.assertEqual(self.reject_atomically(request=replace(self.request, target="reach_target"))["status"],
                         "INELIGIBLE_TARGET")

    def test_low_target_friction_is_native_declared_negative_before_release(self):
        self.args = prepare(target_friction=.001)
        self.model, self.data, self.scene, self.profile, self.reference, self.manager = self.args
        result = self.reject_atomically()
        self.assertEqual(result["status"], "SUPPORT_INFEASIBLE")
        self.assertIn("friction", result["reason"])
        self.assertFalse(result["support_admission"]["admitted"])

    def test_native_support_loss_fault_keeps_actual_terminal_state(self):
        result = ft.execute_foot_transfer(*self.args, self.request, fault="support_loss", keep_samples=False)
        self.assertFalse(result["success"])
        self.assertGreater(result["steps"], 0)
        self.assertEqual(result["duration_s"], result["steps"] * .002)
        self.assertIsNone(result["final_reference"])
        self.assertFalse(np.any(self.data.xfrc_applied))
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        np.testing.assert_array_equal(result["terminal_observation"]["qpos"], self.data.qpos)
        self.assertFalse(result["readiness"]["ready"])

    def test_callback_live_state_edit_is_rejected_before_another_command(self):
        commands, rows = [], []
        native_control = ft.compute_pose_control
        def control(*args, **kwargs):
            commands.append(float(self.data.time))
            return native_control(*args, **kwargs)
        def observer(row, *_):
            rows.append(row)
            if not row["terminal"]:
                self.data.qpos[0] += .00001
        with patch.object(ft, "compute_pose_control", side_effect=control):
            result = ft.execute_foot_transfer(*self.args, self.request, observer=observer)
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertIn("outside the authoritative", result["reason"])
        self.assertEqual(len(commands), result["steps"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[-1]["terminal"])
        np.testing.assert_array_equal(rows[-1]["qpos"], self.data.qpos)

    def test_native_foot_guard_failure_is_checked_in_both_force_epochs(self):
        original = ft._unexpected_loaded_contacts
        for epoch in ("applied_preintegration_solve", "fresh_endpoint_solve"):
            with self.subTest(epoch=epoch):
                self.args = clone(self.sources[.002])
                self.model, self.data, self.scene, self.profile, self.reference, self.manager = self.args
                initial_time = float(self.data.time)
                def contacts(model, measured, allowed):
                    if self.data.time > initial_time and ((measured is self.data) == (epoch == "applied_preintegration_solve")):
                        return (("left_shin_geom", "geom_left_foot"),)
                    return original(model, measured, allowed)
                with patch.object(ft, "_unexpected_loaded_contacts", side_effect=contacts):
                    result = ft.execute_foot_transfer(*self.args, self.request)
                self.assertEqual(result["status"], "CONTACT_LOSS")
                self.assertEqual(result["steps"], 1)
                self.assertEqual(result["guard_failure"]["force_epoch"], epoch)
                self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))

    def test_native_hand_guard_records_applied_failure_and_release(self):
        native_reaction = self.manager._reaction
        def reaction(limb, measured):
            if measured is self.data and self.data.time > 2. and limb == Limb.LEFT_HAND:
                return (851., 0., 0.)
            return native_reaction(limb, measured)
        with patch.object(self.manager, "_reaction", side_effect=reaction):
            result = ft.execute_foot_transfer(*self.args, self.request)
        self.assertEqual(result["status"], "GRIP_FAILURE")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["grip_releases"][-1]["applied_load_N"], 851.)
        self.assertEqual(result["grip_releases"][-1]["force_epoch"], "applied_preintegration_solve")
        self.assertFalse(self.manager.is_attached(Limb.LEFT_HAND))
        self.assertIsNone(result["final_reference"])

    def test_fault_cleanup_preserves_unowned_channels_after_guard_exception(self):
        original = ft._unexpected_loaded_contacts
        body = self.model.body("right_foot").id
        root = self.model.body("climber_root").id
        seen, rows = [], []
        def contacts(model, measured, allowed):
            if np.any(self.data.xfrc_applied[body, :3]):
                seen.append(self.data.xfrc_applied[body, :3].copy())
                self.data.xfrc_applied[root, 5] = .125
                self.data.qfrc_applied[5] = .25
                raise RuntimeError("unexpected guard after owned force")
            return original(model, measured, allowed)
        with patch.object(ft, "_unexpected_loaded_contacts", side_effect=contacts):
            result = ft.execute_foot_transfer(*self.args, self.request, fault="support_loss",
                                              observer=lambda row, *_: rows.append(row))
        np.testing.assert_array_equal(seen, [[1500., 0., 0.]])
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertIn("unexpected guard", result["reason"])
        self.assertFalse(np.any(self.data.xfrc_applied[body]))
        self.assertEqual(self.data.xfrc_applied[root, 5], .125)
        self.assertEqual(self.data.qfrc_applied[5], .25)
        self.assertEqual(rows[-1], result["terminal_observation"])
        np.testing.assert_array_equal(rows[-1]["external_force_world_N"], self.data.xfrc_applied)
        np.testing.assert_array_equal(rows[-1]["qfrc_applied"], self.data.qfrc_applied)
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))

    def test_observer_exception_propagates_identity_and_does_not_retry(self):
        error = ValueError("one-shot observer")
        rows = []
        def observer(row, *_):
            rows.append(row)
            raise error
        with self.assertRaises(ValueError) as caught:
            ft.execute_foot_transfer(*self.args, self.request, observer=observer)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["terminal"])
        np.testing.assert_array_equal(rows[0]["qpos"], self.data.qpos)

    def test_recovered_state_reports_applied_work_not_clock(self):
        native_step = mujoco.mj_step
        def recover(model, data):
            native_step(model, data)
            data.time = 0.
            data.ctrl[:] = 0.
            data.eq_active[:] = False
        with patch("mujoco.mj_step", side_effect=recover):
            result = ft.execute_foot_transfer(*self.args, self.request)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["duration_s"], .002)
        self.assertFalse(result["success"] or result["readiness"]["ready"])
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        self.assertFalse(result["terminal_observation"]["command"]["matches_last_request"])
        np.testing.assert_array_equal(result["terminal_observation"]["command"]["commanded_Nm"], 0.)


class FootTransferPreflightTests(unittest.TestCase):
    def test_native_weak_hands_fail_candidate_support_before_release(self):
        profile = replace(ft.make_foot_transfer_fixture()[3], name="native_weak", grip_capacity=80.)
        args = prepare(profile=profile)
        model, data, _, _, _, manager = args
        before = _integration_state(model, data)
        with patch.object(ft, "compute_pose_control", side_effect=AssertionError("weak source commanded")), \
                patch("mujoco.mj_step", side_effect=AssertionError("weak source integrated")):
            result = ft.execute_foot_transfer(*args, ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target"))
        self.assertEqual(result["status"], "SUPPORT_INFEASIBLE", result["reason"])
        self.assertIn("capacity", result["reason"])
        self.assertEqual(result["steps"], 0)
        self.assertFalse(result["released"] or result["support_admission"]["admitted"])
        np.testing.assert_array_equal(_integration_state(model, data), before)
        self.assertEqual(result["final_state"], get_state_summary(model, data, manager))

    def test_unreachable_canonical_step_returns_local_failure_without_motion(self):
        _, _, scene, profile, seed = ft.make_foot_transfer_fixture()
        target = scene.region("foot_target")
        target = replace(target, position=(*target.position[:2], target.position[2] + 2.))
        scene = replace(scene, contact_regions=tuple(target if r.id == target.id else r for r in scene.contact_regions))
        tree = ET.fromstring(build_mjcf(scene, profile))
        tree.find("option").attrib.update(timestep=".002", iterations="100", tolerance="1e-10")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        data = mujoco.MjData(model)
        reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
        initial = execute_static_hold(model, data, scene, profile, reference, manager, duration=2., settle=1.,
                                      score_window=.5, keep_samples=False)
        self.assertTrue(initial["success"], initial["reason"])
        before = _integration_state(model, data)
        with patch.object(ft, "compute_pose_control", side_effect=AssertionError("unreachable source commanded")), \
                patch("mujoco.mj_step", side_effect=AssertionError("unreachable source integrated")):
            result = ft.execute_foot_transfer(model, data, scene, profile, reference, manager,
                                              ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target"))
        self.assertEqual(result["status"], "REACH_INFEASIBLE")
        self.assertIn("not a global infeasibility proof", result["reason"])
        self.assertFalse(result["released"])
        self.assertEqual(result["steps"], 0)
        self.assertIsNone(result["final_reference"])
        np.testing.assert_array_equal(_integration_state(model, data), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
