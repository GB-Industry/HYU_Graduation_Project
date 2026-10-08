"""Stage4 boundary counterexamples, not a replacement positive physics matrix."""

import copy
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import single_hand
from boulder_v1.contact_geometry import Frame
from boulder_v1.grasp import GraspManager
from boulder_v1.locomotion import get_state_summary
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import ReadinessEvidence, initialize_static_reference


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


class SingleHandBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, data, scene, profile, seed = single_hand.make_single_hand_fixture()
        reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
        # Only prepare loaded native source contacts; no Stage4 positive is cached.
        execute_static_hold(model, data, scene, profile, reference, manager,
                            duration=.5, settle=.2, score_window=.05, keep_samples=False)
        if manager.contact_configuration() != dict(reference.contact_intent):
            raise AssertionError("Short native preparation did not establish the intended source contacts")
        cls.source = (model, data, scene, profile, reference, manager)

    def setUp(self):
        original_model, original_data, scene, profile, reference, original_manager = self.source
        model = copy.copy(original_model)
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, original_data)
        manager = GraspManager(model, data, scene, profile=profile)
        manager.synchronize_from_live()
        manager.capture_events = copy.deepcopy(original_manager.capture_events)
        manager.releases = copy.deepcopy(original_manager.releases)
        self.args = (model, data, scene, profile, reference, manager)
        self.model, self.data, self.reference, self.manager = model, data, reference, manager
        self.request = single_hand.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")

    def reject_atomically(self, *, reference=None, request=None, fault=None):
        args = list(self.args)
        if reference is not None:
            args[4] = reference
        if request is None:
            request = self.request
        self.data.ctrl[:] = .123
        before = integration_state(self.model, self.data)
        captures, releases = copy.deepcopy(self.manager.capture_events), copy.deepcopy(self.manager.releases)
        rows = []
        with patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("rejected input commanded")), \
                patch("mujoco.mj_step", side_effect=AssertionError("rejected input integrated")):
            result = single_hand.execute_single_hand(*args, request, fault=fault,
                                                     observer=lambda row, *_: rows.append(row))
        self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
        self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
        np.testing.assert_array_equal(integration_state(self.model, self.data), before)
        self.assertEqual(self.manager.capture_events, captures)
        self.assertEqual(self.manager.releases, releases)
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        self.assertEqual(rows, [result["terminal_observation"]])
        return result

    def test_external_force_and_moment_entry_rejection_is_atomic(self):
        root = self.model.body("climber_root").id
        for field, index in (("qfrc_applied", 2), ("qfrc_applied", 5),
                             ("xfrc_applied", (root, 2)), ("xfrc_applied", (root, 5))):
            with self.subTest(field=field, index=index):
                getattr(self.data, field)[index] = .25
                result = self.reject_atomically(fault="support_loss")
                self.assertEqual(result["status"], "INITIALIZATION_FAILURE")
                self.assertIn("undeclared", result["reason"])
                np.testing.assert_array_equal(result["terminal_observation"]["qfrc_applied"], self.data.qfrc_applied)
                np.testing.assert_array_equal(result["terminal_observation"]["external_force_world_N"], self.data.xfrc_applied)
                getattr(self.data, field)[index] = 0.

    def test_optional_hand_position_solve_failure_preserves_terminal_result(self):
        before = self.data.time
        rows = []
        native_fresh = single_hand.fresh_data

        def fail_after_integration(model, measured_data):
            # Stage5 adds a mandatory preflight solve; this fault targets the native endpoint.
            if measured_data is self.data and measured_data.time > before:
                raise ValueError("Nonfinite contact solve")
            return native_fresh(model, measured_data)

        with patch.object(single_hand, "fresh_data", side_effect=fail_after_integration):
            result = single_hand.execute_single_hand(*self.args, self.request,
                                                     observer=lambda row, *_: rows.append(row))
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 1)
        self.assertAlmostEqual(self.data.time - before, self.model.opt.timestep)
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        terminal = result["terminal_observation"]
        self.assertEqual(rows, [terminal])
        self.assertEqual(terminal["time_s"], self.data.time)
        self.assertIsNone(terminal["actual_hand_position_world_m"])
        self.assertIsNone(terminal["tracking_error_m"])
        self.assertIn("Nonfinite contact solve", terminal["hand_position_measurement_error"])
        self.assertFalse(terminal["pose_available"] or terminal["readiness"]["ready"])

    def test_full_reference_validation_precedes_live_control(self):
        seed = np.array(self.reference.qpos)
        nonunit = seed.copy(); nonunit[3:7] *= 2.
        nonfinite = seed.copy(); nonfinite[0] = math.nan
        outside = seed.copy()
        elbow = self.model.joint("right_elbow")
        outside[elbow.qposadr[0]] = elbow.range[1] + .01
        target = dict(self.reference.target_pose)
        target["right_elbow"] += .01
        bad_references = (
            replace(self.reference, qpos=nonunit), replace(self.reference, qpos=nonfinite),
            replace(self.reference, qpos=outside), replace(self.reference, qpos=seed[:-1]),
            replace(self.reference, contact_intent={l.value: h for l, h in self.reference.contact_intent.items()}),
            replace(self.reference, contact_intent={Limb.RIGHT_HAND: "right_hand"}),
            replace(self.reference, contact_intent={**self.reference.contact_intent, Limb.LEFT_HAND: []}),
            replace(self.reference, target_pose=target),
            SimpleNamespace(qpos=seed[:, None], contact_intent={}, target_pose={}),
            SimpleNamespace(qpos=seed, contact_intent=None, target_pose={}), object(),
        )
        for bad in bad_references:
            with self.subTest(reference=bad):
                result = self.reject_atomically(reference=bad)
                self.assertIn(result["status"], ("INITIALIZATION_FAILURE", "ROM_FAILURE"))
                row = result["terminal_observation"]
                self.assertFalse(row["source_reference_admitted"])
                self.assertIsNone(row["q_ref"])
                self.assertIsNone(row["qd_ref"])
                self.assertIsNone(row["command"])

    def test_reference_admission_cannot_be_forged_with_valid_rom(self):
        seed = np.array(self.reference.qpos)
        seed[0] += .05
        result = self.reject_atomically(reference=replace(self.reference, qpos=seed))
        self.assertEqual(result["status"], "INITIALIZATION_FAILURE")
        self.assertFalse(result["terminal_observation"]["source_reference_admitted"])

    def test_malformed_requests_and_unknown_fault_do_not_break_terminal_reporting(self):
        bad_requests = (object(), replace(self.request, limb=[]), replace(self.request, target=[]),
                        replace(self.request, target="missing"), replace(self.request, reach_s=None),
                        replace(self.request, transfer_s=True), replace(self.request, clearance_m=".02"),
                        replace(self.request, approach_normal=(0., 0., 0.)),
                        replace(self.request, approach_normal=(1., 0., 0.)))
        for bad in bad_requests:
            with self.subTest(request=bad):
                self.assertEqual(self.reject_atomically(request=bad)["status"], "REACH_INFEASIBLE")
        for fault in ("support_los", [], 1):
            with self.subTest(fault=fault):
                result = self.reject_atomically(fault=fault)
                self.assertIn("Unknown declared", result["reason"])
        for scenario in ("grip_after_captur", None, []):
            with self.subTest(scenario=scenario), \
                    patch.object(single_hand, "make_single_hand_fixture", side_effect=AssertionError("unknown scenario constructed")), \
                    self.assertRaisesRegex(ValueError, "Unknown single-hand benchmark scenario"):
                single_hand.run_single_hand_benchmark(scenario=scenario)

    def test_step_audits_caller_force_without_clearing_it_or_writing_next_command(self):
        for field, index in (("qfrc_applied", 5), ("xfrc_applied", (self.model.body("climber_root").id, 5))):
            with self.subTest(field=field):
                commands, rows = [], []
                native_control = single_hand.compute_pose_control

                def control(*args, **kwargs):
                    commands.append(float(self.data.time))
                    return native_control(*args, **kwargs)

                def observer(row, *_):
                    rows.append(row)
                    if not row["terminal"]:
                        getattr(self.data, field)[index] = .125

                with patch.object(single_hand, "compute_pose_control", side_effect=control):
                    result = single_hand.execute_single_hand(*self.args, self.request, observer=observer)
                self.assertEqual(result["status"], "CONTROL_FAILURE")
                self.assertIn("undeclared", result["reason"])
                self.assertEqual(len(commands), result["steps"])
                self.assertEqual(getattr(self.data, field)[index], .125)
                self.assertTrue(rows[-1]["terminal"])
                getattr(self.data, field)[index] = 0.

    def test_one_shot_running_callback_value_error_propagates_without_terminal_retry(self):
        rows, stopped = [], {}

        def observer(row, *_):
            rows.append(row)
            if len(rows) == 1:
                stopped["state"] = integration_state(self.model, self.data)
                raise ValueError("one-shot observer failure")

        with self.assertRaisesRegex(ValueError, "one-shot observer failure"):
            single_hand.execute_single_hand(*self.args, self.request, observer=observer)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["terminal"])
        np.testing.assert_array_equal(integration_state(self.model, self.data), stopped["state"])

    def test_recovered_control_telemetry_describes_actual_terminal_data(self):
        native_step = mujoco.mj_step
        rows = []

        def recover(model, data):
            native_step(model, data)
            data.time = 0.
            data.ctrl[:] = 0.
            data.eq_active[:] = False

        with patch("mujoco.mj_step", side_effect=recover):
            result = single_hand.execute_single_hand(*self.args, self.request,
                                                     observer=lambda row, *_: rows.append(row))
        self.assertEqual((result["status"], result["steps"]), ("CONTROL_FAILURE", 1))
        self.assertEqual(rows, [result["terminal_observation"]])
        row = rows[0]
        self.assertFalse(row["command"]["matches_last_request"])
        np.testing.assert_array_equal(row["command"]["commanded_Nm"], self.data.ctrl * self.model.actuator_gear[:, 0])
        np.testing.assert_array_equal(row["command"]["utilization"], 0.)
        np.testing.assert_array_equal(row["eq_active"], self.data.eq_active)
        self.assertEqual(row["readiness"]["time"], self.data.time)
        self.assertFalse(row["readiness"]["ready"])
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))

    @contextmanager
    def negative_epoch_harness(self):
        """Clock/task stubs isolate executor boundaries; never assert physics success."""
        class ClockOnlyTracker:
            evidence = ReadinessEvidence(ready=True, duration=.5, reason="TEST_ONLY clock stub")

            def __init__(self, *args):
                pass

            def check(self):
                return True, self.evidence.reason

            def sample_after_step(self, before):
                return self.evidence

        def clock_step(model, data):
            data.time += model.opt.timestep

        supports = {limb: hold for limb, hold in self.reference.contact_intent.items() if limb != self.request.limb}
        # Stage5 preflight consumes complete diagnostics, not just the torque map.
        # Freeze a real source estimate, retaining zero FF for this clock-only protocol.
        estimate = single_hand.estimate_support_torques(
            self.model, self.args[2], self.args[3], self.data.qpos, supports)
        estimate.update(torques_Nm={name: 0. for name in self.reference.target_pose},
                        scope="TEST_ONLY frozen production diagnostics and zero FF; clock protocol, not physical acceptance")

        with ExitStack() as stack:
            stack.enter_context(patch.object(single_hand, "ReadinessTracker", ClockOnlyTracker))
            stack.enter_context(patch.object(single_hand, "solve_hand_reference", side_effect=lambda m, q, *_:
                                             SimpleNamespace(qpos=tuple(q), converged=True)))
            stack.enter_context(patch.object(single_hand, "estimate_support_torques", return_value=estimate))
            stack.enter_context(patch.object(single_hand, "_unexpected_loaded_contacts", return_value=()))
            stack.enter_context(patch.object(single_hand, "_supported_hold_geometry", return_value=True))
            stack.enter_context(patch("mujoco.mj_step", side_effect=clock_step))
            stack.enter_context(patch.object(self.manager, "can_attach", return_value=False))
            yield replace(self.request, transfer_s=.002, support_s=.002, reach_s=.4, capture_timeout_s=.1)

    def test_all_active_grips_are_guarded_before_post_capture_command(self):
        target = self.args[2].region(self.request.target)
        attachment = self.manager.registered_attachment(Limb.RIGHT_HAND, target)
        activated, commands = [], []
        native_control, native_reaction = single_hand.compute_pose_control, self.manager._reaction
        native_measurement = self.manager._capture_measurement

        def activate(limb, region):
            self.data.eq_active[attachment.eq_id] = True
            self.manager.synchronize_from_live()
            event = copy.deepcopy(next(e for e in self.manager.capture_events if e["limb"] == limb.value))
            event.update(region_id=region.id, time_s=float(self.data.time), initial_reaction_N=170.,
                         initial_reaction_world_N=(170., 0., 0.))
            self.manager.capture_events.append(event)
            activated.append((float(self.data.time), self.data.ctrl.copy(), len(commands)))
            return True

        def reaction(limb, data):
            if self.data.eq_active[attachment.eq_id]:
                return (851., 0., 0.) if limb == Limb.LEFT_HAND else (170., 0., 0.)
            return native_reaction(limb, data)

        def measurement(limb, region, data=None):
            info = native_measurement(limb, region, data)
            if self.data.eq_active[attachment.eq_id] and region.id == target.id:
                info.update(penetration_m=0., signed_geom_distance_m=.0005)
            return info

        def control(*args, **kwargs):
            commands.append(float(self.data.time))
            return native_control(*args, **kwargs)

        with self.negative_epoch_harness() as request, \
                patch.object(self.manager, "can_attach", return_value=True), \
                patch.object(self.manager, "attach", side_effect=activate), \
                patch.object(self.manager, "_reaction", side_effect=reaction), \
                patch.object(self.manager, "_capture_measurement", side_effect=measurement), \
                patch.object(single_hand, "compute_pose_control", side_effect=control), \
                patch.object(single_hand, "solve_contact_pose", side_effect=AssertionError("overloaded capture retargeted")):
            result = single_hand.execute_single_hand(*self.args, request)
        self.assertEqual(result["status"], "GRIP_FAILURE", result["reason"])
        self.assertFalse(result["success"])
        self.assertEqual(len(activated), 1)
        self.assertEqual(len(commands), activated[0][2])
        np.testing.assert_array_equal(self.data.ctrl, activated[0][1])
        self.assertEqual(result["capture"]["initial_reaction_N"], 170.)
        decisions = result["capture"]["post_activation_decisions"]
        self.assertTrue(decisions["RIGHT_HAND"]["maintain"])
        self.assertFalse(decisions["LEFT_HAND"]["maintain"])
        self.assertEqual(result["guard_failure"]["force_epoch"], "fresh_post_activation_solve")
        self.assertEqual(result["release_events"][-1]["endpoint_load_N"], 851.)
        self.assertFalse(self.manager.is_attached(Limb.LEFT_HAND))
        self.assertTrue(self.manager.is_attached(Limb.RIGHT_HAND))

    def test_unexpected_guard_error_cleans_only_owned_force_and_reports_actual_terminal(self):
        body, caller_body = self.model.body("left_foot").id, self.model.body("climber_root").id
        native_guard = self.manager.evaluate_and_update
        seen, rows = [], []

        def guard(*args, **kwargs):
            if np.any(self.data.xfrc_applied[body, :3]):
                seen.append(self.data.xfrc_applied[body].copy())
                self.data.xfrc_applied[caller_body, 5] = .125
                self.data.qfrc_applied[5] = .25
                raise RuntimeError("unexpected guard after declared force")
            return native_guard(*args, **kwargs)

        with self.negative_epoch_harness() as request, patch.object(self.manager, "evaluate_and_update", side_effect=guard):
            result = single_hand.execute_single_hand(*self.args, request, fault="support_loss",
                                                     observer=lambda row, *_: rows.append(row))
        np.testing.assert_array_equal(seen, [[1500., 0., 0., 0., 0., 0.]])
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertIn("unexpected guard", result["reason"])
        self.assertFalse(np.any(self.data.xfrc_applied[body]))
        self.assertEqual(self.data.xfrc_applied[caller_body, 5], .125)
        self.assertEqual(self.data.qfrc_applied[5], .25)
        self.assertTrue(rows[-1]["terminal"])
        np.testing.assert_array_equal(rows[-1]["external_force_world_N"], self.data.xfrc_applied)
        np.testing.assert_array_equal(rows[-1]["qfrc_applied"], self.data.qfrc_applied)
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))

    def test_declared_force_does_not_exempt_other_channels_on_its_body(self):
        body = self.model.body("left_foot").id
        native_guard = self.manager.evaluate_and_update
        seen = []

        def guard(*args, **kwargs):
            if np.any(self.data.xfrc_applied[body, :3]):
                seen.append(float(self.data.time))
                self.data.xfrc_applied[body, 3] = .125
                return {}
            return native_guard(*args, **kwargs)

        with self.negative_epoch_harness() as request, patch.object(self.manager, "evaluate_and_update", side_effect=guard):
            result = single_hand.execute_single_hand(*self.args, request, fault="support_loss")
        self.assertEqual(seen, [self.data.time])
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertIn("undeclared", result["reason"])
        np.testing.assert_array_equal(self.data.xfrc_applied[body], [0., 0., 0., .125, 0., 0.])
        np.testing.assert_array_equal(result["terminal_observation"]["external_force_world_N"], self.data.xfrc_applied)

    def test_terminal_callback_exception_is_not_retried_after_owned_force_cleanup(self):
        body = self.model.body("left_foot").id
        native_guard = self.manager.evaluate_and_update
        terminal = []

        def guard(*args, **kwargs):
            if np.any(self.data.xfrc_applied[body]):
                raise RuntimeError("stop at declared force")
            return native_guard(*args, **kwargs)

        def observer(row, *_):
            if row["terminal"]:
                terminal.append(row)
                self.assertFalse(np.any(self.data.xfrc_applied[body]))
                if len(terminal) == 1:
                    raise ValueError("one-shot terminal observer failure")

        with self.negative_epoch_harness() as request, \
                patch.object(self.manager, "evaluate_and_update", side_effect=guard), \
                self.assertRaisesRegex(ValueError, "one-shot terminal observer failure"):
            single_hand.execute_single_hand(*self.args, request, fault="support_loss", observer=observer)
        self.assertEqual(len(terminal), 1)
        self.assertFalse(np.any(self.data.xfrc_applied[body]))

    def test_rotated_reach_endpoints_and_midpoint_use_local_quaternion_increment(self):
        a, b = .7, .4
        start_rotation = np.array([[1., 0., 0.], [0., math.cos(a), -math.sin(a)],
                                   [0., math.sin(a), math.cos(a)]])
        turn = lambda angle: np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                                       [-math.sin(angle), 0., math.cos(angle)]])
        start = Frame((.2, -.4, 1.), tuple(map(tuple, start_rotation)))
        goal = Frame((.3, -.4, 1.06), tuple(map(tuple, turn(b) @ start_rotation)))
        for time, expected in ((0., start), (1., goal)):
            actual = single_hand.reach_frame(start, goal, time, 1., .02)
            np.testing.assert_allclose(actual.rotation, expected.rotation, rtol=0, atol=1e-14)
            np.testing.assert_allclose(actual.position, expected.position, rtol=0, atol=1e-14)
        midpoint = single_hand.reach_frame(start, goal, .5, 1., .02)
        np.testing.assert_allclose(midpoint.rotation, turn(b / 2) @ start_rotation, rtol=0, atol=1e-14)

    def test_reference_shaper_bounds_phase_junctions_and_converges_without_live_writes(self):
        before = integration_state(self.model, self.data)
        joints = self.model.actuator_trnid[:, 0]
        qi, vi = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
        shoulder = self.model.joint("right_shoulder_pitch").qposadr[0]
        for dt in (.002, .001):
            with self.subTest(dt=dt):
                qref, qd = np.array(self.reference.qpos), np.zeros(self.model.nv)
                goal = qref.copy(); goal[shoulder] += .15
                for phase in ("RELEASE_CLEARANCE", "THREE_POINT", "REACH", "SETTLE"):
                    if phase == "THREE_POINT":
                        goal = qref.copy()
                    elif phase == "REACH":
                        goal[shoulder] -= .12
                    elif phase == "SETTLE":
                        goal = np.array(self.reference.qpos)
                    for _ in range(round((1.5 if phase == "SETTLE" else .08) / dt)):
                        previous, velocity = qref.copy(), qd.copy()
                        qref, qd = single_hand._shape_reference(self.model, goal, previous, velocity, dt)
                        self.assertLessEqual(np.max(np.abs(qd[vi])), .5 + 1e-12)
                        self.assertLessEqual(np.max(np.abs(qd[vi] - velocity[vi])), 2. * dt + 1e-12)
                        np.testing.assert_allclose((qref[qi] - previous[qi]) / dt, qd[vi], rtol=0, atol=1e-12)
                np.testing.assert_allclose(qref, goal, rtol=0, atol=1e-12)
                np.testing.assert_allclose(qd, 0., rtol=0, atol=1e-12)
        np.testing.assert_array_equal(integration_state(self.model, self.data), before)

    def test_reference_shaper_brakes_at_compiled_rom_without_clipping_live_pose(self):
        joint = self.model.joint("right_elbow")
        qref = np.array(self.reference.qpos)
        qref[joint.qposadr[0]] = joint.range[1] - .08
        qd = np.zeros(self.model.nv); qd[joint.dofadr[0]] = .45
        goal = qref.copy(); goal[joint.qposadr[0]] = joint.range[1]
        before = integration_state(self.model, self.data)
        for tick in range(500):
            try:
                qref, qd = single_hand._shape_reference(self.model, goal, qref, qd, .002)
            except ValueError as error:
                self.fail(f"tick={tick} room={joint.range[1] - qref[joint.qposadr[0]]!r} "
                          f"velocity={qd[joint.dofadr[0]]!r}: {error}")
            self.assertLessEqual(qref[joint.qposadr[0]], joint.range[1])
        np.testing.assert_allclose(qref, goal, rtol=0, atol=1e-12)
        np.testing.assert_allclose(qd, 0., rtol=0, atol=1e-12)
        goal[joint.qposadr[0]] += .01
        with self.assertRaisesRegex(ValueError, "compiled ROM"):
            single_hand._shape_reference(self.model, goal, qref, qd, .002)
        np.testing.assert_array_equal(integration_state(self.model, self.data), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
