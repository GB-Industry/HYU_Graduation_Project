"""Short native Stage3 runs covering failure observations and pulse boundaries."""

import copy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1.contact_benchmarks import make_mixed_fixture
from boulder_v1.locomotion import get_state_summary
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import ReadinessEvidence, ReadinessTracker, initialize_static_reference
from boulder_v1.support import FootStatus
from scripts.render_demo import _json_value


class StaticFailureTests(unittest.TestCase):
    def admitted(self, dt=.002):
        fixture = make_mixed_fixture(dt)
        intent = {limb: limb.value.lower() for limb in Limb}
        reference, manager = initialize_static_reference(fixture.model, fixture.data, fixture.scene,
                                                         fixture.profile, fixture.reference, intent)
        return fixture, reference, manager

    def run_hold(self, fixture, reference, manager, **kwargs):
        return execute_static_hold(fixture.model, fixture.data, manager.scene, fixture.profile,
                                   reference, manager, duration=.5, settle=.2, score_window=.05, **kwargs)

    def test_observer_exceptions_restore_force_without_restoring_other_state(self):
        for error_type in (RuntimeError, ValueError):
            for terminal in (False, True):
                with self.subTest(error=error_type.__name__, terminal=terminal):
                    fixture, reference, manager = self.admitted()
                    model, data = fixture.model, fixture.data
                    pulse = {"body": "climber_root", "start_s": .2, "duration_s": .1,
                             "force_world_N": [2., 0., 0.]}
                    at_failure = {}

                    def observer(row, detached_model, detached_data):
                        self.assertIsNot(detached_model, model)
                        self.assertIsNot(detached_data, data)
                        if (row["terminal"] if terminal else row["disturbance_active"]):
                            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active"):
                                at_failure[name] = getattr(data, name).copy()
                            at_failure["time"] = float(data.time)
                            if terminal:
                                self.assertFalse(np.any(data.xfrc_applied))
                            else:
                                np.testing.assert_array_equal(data.xfrc_applied[model.body("climber_root").id, :3],
                                                              pulse["force_world_N"])
                            data.xfrc_applied[model.body("right_hand").id, 0] = .25
                            raise error_type("observer failure")

                    with patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")), \
                            self.assertRaisesRegex(error_type, "observer failure"):
                        self.run_hold(fixture, reference, manager, disturbance=pulse, observer=observer)
                    self.assertTrue(at_failure)
                    self.assertFalse(np.any(data.xfrc_applied[model.body("climber_root").id]))
                    self.assertEqual(data.xfrc_applied[model.body("right_hand").id, 0], .25)
                    self.assertFalse(np.any(data.qfrc_applied))
                    self.assertEqual(data.time, at_failure["time"])
                    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active"):
                        np.testing.assert_array_equal(getattr(data, name), at_failure[name])

    def test_overload_before_onset_preserves_failure_and_json_evidence(self):
        fixture, reference, manager = self.admitted()
        model, data = fixture.model, fixture.data
        pulse = {"body": "climber_root", "start_s": .3, "duration_s": .1,
                 "force_world_N": [2., 0., 0.]}
        reaction = manager._reaction

        def overloaded(limb, measured_data):
            if limb == Limb.LEFT_HAND and measured_data.time >= 2 * model.opt.timestep:
                return (2 * fixture.profile.grip_capacity, 0., 0.)
            return reaction(limb, measured_data)

        with patch.object(manager, "_reaction", side_effect=overloaded), \
                patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")):
            result = self.run_hold(fixture, reference, manager, disturbance=pulse)
        self.assertEqual(result["status"], "GRASP_OVERLOAD")
        self.assertFalse(result["success"] or result["contact_acceptance"] or result["controller_convergence"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["duration_s"], 2 * model.opt.timestep)
        self.assertEqual(result["final_state"], get_state_summary(model, data, manager))
        self.assertEqual(len(result["release_events"]), 1)
        self.assertEqual(result["release_events"][0]["limb"], Limb.LEFT_HAND.value)
        self.assertFalse(result["final_state"].hand_states[Limb.LEFT_HAND].active)
        self.assertEqual(result["recovery"], {"ready_after_pulse_s": None, "peak_root_speed_m_s": None,
                                             "peak_joint_speed_rad_s": None})
        realization = result["disturbance_realization"]
        self.assertEqual(realization["applied_steps"], 0)
        self.assertEqual(realization["integrated_impulse_world_Ns"], [0., 0., 0.])
        self.assertIsNone(realization["realized_start_elapsed_s"])
        self.assertIsNone(realization["realized_end_elapsed_s"])
        encoded = json.loads(json.dumps(_json_value(result), allow_nan=False))
        self.assertEqual(encoded["status"], "GRASP_OVERLOAD")
        self.assertEqual(encoded["steps"], 2)
        self.assertEqual(len(encoded["release_events"]), 1)
        self.assertEqual(encoded["final_state"]["time"], data.time)

    def test_last_endpoint_slip_has_coherent_terminal_observation(self):
        fixture, reference, manager = self.admitted()
        model, data = fixture.model, fixture.data
        snapshot = manager.contact_snapshot
        observations = []

        def slipping():
            state = snapshot()
            if data.time >= .202 - 1e-12:
                foot = state.feet[Limb.LEFT_FOOT]
                state = replace(state, feet={**state.feet, Limb.LEFT_FOOT: replace(
                    foot, status=FootStatus.SLIPPING, supporting=False, slipping=True, tangential_speed=.02)})
            return state

        def observer(row, detached_model, detached_data):
            self.assertEqual(row["time_s"], detached_data.time)
            self.assertEqual(row["time_s"], data.time)
            np.testing.assert_array_equal(row["ctrl"], detached_data.ctrl)
            np.testing.assert_allclose(row["command"]["commanded_Nm"],
                                        detached_data.ctrl * detached_model.actuator_gear[:, 0], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(row["joint_velocity_rad_s"],
                                          detached_data.qvel[model.jnt_dofadr[model.actuator_trnid[:, 0]]])
            self.assertEqual(row["root_linear_m_s"], np.linalg.norm(detached_data.qvel[:3]))
            self.assertEqual(row["root_angular_rad_s"], np.linalg.norm(detached_data.qvel[3:6]))
            observations.append(row)

        with patch.object(manager, "contact_snapshot", side_effect=slipping), \
                patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")):
            result = execute_static_hold(model, data, manager.scene, fixture.profile, reference, manager,
                                         duration=.202, settle=.2, score_window=.002, observer=observer)
        self.assertEqual(result["steps"], 101)
        self.assertEqual(result["status"], "CONTACT_LOSS")
        self.assertFalse(result["success"] or result["readiness"]["ready"])
        self.assertEqual(len(result["samples"]), 100)
        terminal = observations[-1]
        self.assertTrue(terminal["terminal"])
        self.assertEqual(terminal, result["terminal_observation"])
        self.assertEqual(terminal["status"], result["status"])
        self.assertEqual(terminal["reason"], result["reason"])
        self.assertFalse(terminal["readiness"]["ready"])
        self.assertEqual(terminal["readiness"]["time"], data.time)
        self.assertGreater(terminal["time_s"], result["samples"][-1]["time_s"])
        for limb in Limb:
            measured = result["final_state"].hand_states if limb.is_hand else result["final_state"].foot_states
            self.assertEqual(terminal["hands" if limb.is_hand else "feet"][limb.value], asdict(measured[limb]))
        self.assertTrue(terminal["feet"][Limb.LEFT_FOOT.value]["slipping"])
        self.assertFalse(terminal["feet"][Limb.LEFT_FOOT.value]["supporting"])
        self.assertTrue(all(not row["terminal"] for row in result["samples"]))
        terminal["hands"][Limb.RIGHT_HAND.value]["active"] = False
        self.assertTrue(result["terminal_observation"]["hands"][Limb.RIGHT_HAND.value]["active"])

    def test_invalid_endpoint_and_recovery_observe_actual_data(self):
        for fault in ("qpos", "qvel", "ctrl", "recovered"):
            recovered = fault == "recovered"
            with self.subTest(fault=fault):
                fixture, reference, manager = self.admitted()
                model, data = fixture.model, fixture.data
                native_step = mujoco.mj_step
                endpoints, observations = [], []

                def step(live_model, live_data):
                    native_step(live_model, live_data)
                    if len(endpoints) == 1:
                        # Endpoint fault injection only; production must retain it.
                        if recovered:
                            live_data.time = 0.
                            live_data.eq_active[:] = False
                            live_data.ctrl[:] = 0.
                        else:
                            getattr(live_data, fault)[0] = np.nan
                    endpoints.append((live_data.qpos.copy(), live_data.qvel.copy(),
                                      live_data.ctrl.copy(), float(live_data.time)))

                def observer(row, detached_model, detached_data):
                    self.assertEqual(row["time_s"], detached_data.time)
                    np.testing.assert_array_equal(detached_data.qpos, data.qpos)
                    np.testing.assert_array_equal(detached_data.qvel, data.qvel)
                    np.testing.assert_array_equal(row["ctrl"], data.ctrl)
                    observations.append(row)

                with patch("mujoco.mj_step", side_effect=step), \
                        patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")):
                    result = self.run_hold(fixture, reference, manager, observer=observer)
                self.assertEqual(result["status"], "NONFINITE_STATE")
                self.assertEqual(result["steps"], 2)
                terminal = observations[-1]
                self.assertTrue(terminal["terminal"])
                self.assertEqual(terminal["status"], "NONFINITE_STATE")
                self.assertFalse(terminal["readiness"]["ready"])
                self.assertEqual(terminal["time_s"], endpoints[-1][3])
                for name, expected in zip(("qpos", "qvel", "ctrl"), endpoints[-1][:3]):
                    np.testing.assert_array_equal(getattr(data, name), expected)
                if recovered:
                    self.assertEqual(data.time, 0.)
                    self.assertTrue(terminal["pose_available"])
                    self.assertTrue(all(not hand["active"] for hand in terminal["hands"].values()))
                    self.assertFalse(terminal["command"]["matches_last_request"])
                    np.testing.assert_array_equal(terminal["command"]["commanded_Nm"], 0.)
                else:
                    self.assertFalse(terminal["pose_available"])
                    self.assertIsNone(terminal["unintended_contacts"])
                    self.assertTrue(all(not hand["measurement_valid"] for hand in terminal["hands"].values()))
                    self.assertTrue(all(not foot["measurement_valid"] for foot in terminal["feet"].values()))
                json.dumps(_json_value(result), allow_nan=False)

    def test_nonintegral_or_empty_pulse_rejected_without_mutation(self):
        fixture, reference, manager = self.admitted()
        model, data = fixture.model, fixture.data
        data.ctrl[:] = .123
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, before, spec)
        captures, releases = copy.deepcopy(manager.capture_events), copy.deepcopy(manager.releases)
        for onset, width in ((.2005, .05), (.2, .0505), (.2, .0005), (.2, .001), (.2, 0.), (0., .05)):
            with self.subTest(onset=onset, width=width):
                pulse = {"body": "climber_root", "start_s": onset, "duration_s": width,
                         "force_world_N": [2., 0., 0.]}
                with patch("boulder_v1.static_control.compute_pose_control", side_effect=AssertionError("control write")), \
                        patch("mujoco.mj_step", side_effect=AssertionError("integration")), \
                        patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")), \
                        self.assertRaises(ValueError):
                    self.run_hold(fixture, reference, manager, disturbance=pulse)
                after = np.empty_like(before)
                mujoco.mj_getState(model, data, after, spec)
                np.testing.assert_array_equal(after, before)
                self.assertEqual(manager.capture_events, captures)
                self.assertEqual(manager.releases, releases)

    def test_integer_pulse_cadence_impulse_and_postpulse_recovery_selection(self):
        for dt in (.002, .001):
            with self.subTest(dt=dt):
                fixture, reference, manager = self.admitted(dt)
                model, data = fixture.model, fixture.data
                data.time = 1000.  # Native clock drift must not select pulse intervals.
                pulse = {"body": "climber_root", "start_s": .3, "duration_s": .1,
                         "force_world_N": [2., 0., 0.]}
                forces = []
                native_step = mujoco.mj_step

                def step(live_model, live_data):
                    forces.append(live_data.xfrc_applied[model.body("climber_root").id, :3].copy())
                    native_step(live_model, live_data)

                def timing_gate(tracker, before_time):
                    # Timing-only readiness injection, not physical acceptance evidence:
                    # only applied-pulse rows are ready, including its last endpoint.
                    return ReadinessEvidence(ready=bool(np.any(tracker.data.xfrc_applied)), duration=.5,
                                             reason="TEST_ONLY pulse timing gate", time=float(tracker.data.time))

                with patch("mujoco.mj_step", side_effect=step), \
                        patch.object(ReadinessTracker, "sample_after_step", timing_gate), \
                        patch("mujoco.mj_resetData", side_effect=AssertionError("execution reset")):
                    result = self.run_hold(fixture, reference, manager, disturbance=pulse)
                self.assertEqual(result["steps"], round(.5 / dt))
                self.assertEqual(len(result["samples"]), result["steps"])
                expected = np.zeros((result["steps"], 3))
                first, count = round(.3 / dt), round(.1 / dt)
                expected[first:first + count] = pulse["force_world_N"]
                np.testing.assert_array_equal(forces, expected)
                np.testing.assert_array_equal([row["disturbance_active"] for row in result["samples"]],
                                              np.any(expected, axis=1))
                realization = result["disturbance_realization"]
                self.assertEqual(realization["applied_steps"], 50 if dt == .002 else 100)
                self.assertEqual(realization["integrated_impulse_world_Ns"], [.2, 0., 0.])
                np.testing.assert_allclose(np.sum(forces, axis=0) * dt, [.2, 0., 0.], rtol=0, atol=1e-12)
                self.assertEqual(realization["realized_start_elapsed_s"], first * dt)
                self.assertEqual(realization["realized_end_elapsed_s"], (first + count) * dt)
                self.assertIsNone(result["recovery"]["ready_after_pulse_s"])
                self.assertIsNotNone(result["recovery"]["peak_root_speed_m_s"])
                self.assertIsNotNone(result["recovery"]["peak_joint_speed_rad_s"])
                self.assertFalse(np.any(data.xfrc_applied))
                self.assertFalse(result["terminal_observation"]["disturbance_active"])

    def test_recovery_starts_on_an_unforced_interval_not_last_pulse_endpoint(self):
        fixture, reference, manager = self.admitted()
        dt = fixture.model.opt.timestep
        pulse = {"body": "climber_root", "start_s": .3, "duration_s": .1,
                 "force_world_N": [2., 0., 0.]}

        def timing_gate(tracker, before_time):
            # Timing-only gate: readiness covers the pulse and first unforced row.
            return ReadinessEvidence(ready=tracker.data.time >= pulse["start_s"], duration=.5,
                                     reason="TEST_ONLY recovery timing gate", time=float(tracker.data.time))

        with patch.object(ReadinessTracker, "sample_after_step", timing_gate):
            result = self.run_hold(fixture, reference, manager, disturbance=pulse)
        end = round((pulse["start_s"] + pulse["duration_s"]) / dt)
        last_applied, first_unforced = result["samples"][end - 1:end + 1]
        self.assertTrue(last_applied["readiness"]["ready"] and last_applied["disturbance_active"])
        self.assertGreaterEqual(last_applied["elapsed_s"], pulse["start_s"] + pulse["duration_s"])
        self.assertFalse(first_unforced["disturbance_active"])
        self.assertTrue(first_unforced["readiness"]["ready"])
        self.assertEqual(result["recovery"]["ready_after_pulse_s"], first_unforced["integrated_elapsed_s"])
        self.assertEqual(result["recovery"]["ready_after_pulse_s"], (end + 1) * dt)


if __name__ == "__main__":
    unittest.main()
