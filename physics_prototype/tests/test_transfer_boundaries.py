"""Short boundary counterexamples; clock/contact stubs never certify physics."""
import copy
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from boulder_v1 import foot_transfer as ft, single_hand as sh, transfers
from boulder_v1.contact_ik import HandReferenceResult
from boulder_v1.grasp import GraspManager
from boulder_v1.locomotion import get_state_summary
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import ReadinessEvidence, ReadinessTracker, _integration_state, initialize_static_reference
from boulder_v1.support import FootStatus, FootSupportState


class BoundaryTracker:
    evidence = ReadinessEvidence(ready=True, duration=.5, reason="TEST_ONLY boundary clock")
    _session_token = ReadinessTracker._session_token

    def __init__(self, model, data, manager):
        self.manager = manager

    def check(self):
        return True, self.evidence.reason

    def sample_after_step(self, before):
        return self.evidence


class TransferBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, data, scene, profile, seed = ft.make_foot_transfer_fixture()
        reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
        execute_static_hold(model, data, scene, profile, reference, manager,
                            duration=.5, settle=.2, score_window=.05, keep_samples=False)
        if manager.contact_configuration() != dict(reference.contact_intent):
            raise AssertionError("Short native preparation did not establish all source contacts")
        cls.source = model, data, scene, profile, reference, manager

    def setUp(self):
        original, live, self.scene, self.profile, self.reference, owner = self.source
        self.model = copy.copy(original)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, live)
        self.manager = GraspManager(self.model, self.data, self.scene, profile=self.profile)
        self.manager.synchronize_from_live()
        self.manager.capture_events = copy.deepcopy(owner.capture_events)
        self.manager.releases = copy.deepcopy(owner.releases)
        self.args = self.model, self.data, self.scene, self.profile, self.reference, self.manager
        self.foot = ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target")
        self.hand = sh.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")
        self.generic = transfers.TransferRequest(self.foot.limb, self.foot.source, self.foot.target,
                                                 self.reference.contact_intent)

    @contextmanager
    def foot_protocol(self, *, recontact_tick=None, epoch="both", target_tick=5):
        """Inject readings and advance only the clock to isolate executor guards."""
        base = self.manager.contact_snapshot()
        state = {"tick": 0}
        source = base.feet[Limb.LEFT_FOOT]
        corner = source.contacts[0]
        target = replace(corner, region_id="foot_target", surface_geom="geom_foot_target",
                         normal_force=.8, force_world=(0., 0., .8), tangential_force=0.,
                         tangential_speed=0., admissible=True)
        zero_source = replace(corner, normal_force=0., force_world=(0., 0., 0.),
                              tangential_force=0., tangential_speed=0., admissible=False)

        def feet(measured=None):
            tick = state["tick"]
            moving = source if tick < 3 else FootSupportState()
            if target_tick is not None and tick >= target_tick:
                load = .8 if tick == target_tick else 10.
                contact = replace(target, normal_force=load, force_world=(0., 0., load))
                contacts = (contact,)
                selected = epoch == "both" or (measured is self.data) == (epoch == "applied")
                if recontact_tick is not None and tick >= recontact_tick and selected:
                    contacts += (zero_source,)
                moving = FootSupportState(status=FootStatus.SUPPORTING if load > 5. else FootStatus.CONTACTING,
                                          contacting=True, supporting=load > 5., normal_force=load,
                                          contacts=contacts, support_regions=("foot_target",) if load > 5. else (),
                                          support_surfaces=("geom_foot_target",) if load > 5. else ())
            return {Limb.LEFT_FOOT: moving, Limb.RIGHT_FOOT: base.feet[Limb.RIGHT_FOOT]}

        def snapshot():
            measured = feet()
            contacts = dict(base.configuration)
            contacts.pop(Limb.LEFT_FOOT)
            if measured[Limb.LEFT_FOOT].supporting:
                contacts[Limb.LEFT_FOOT] = measured[Limb.LEFT_FOOT].primary_region
            return replace(base, time=float(self.data.time), feet=measured, configuration=contacts,
                           supporting_limbs=tuple(contacts))

        def clock(model, data):
            state["tick"] += 1
            data.time += model.opt.timestep

        request = replace(self.foot, unload_s=.002, lift_s=.002, support_s=.002, reach_s=.002,
                          contact_timeout_s=.002, load_s=.104, settle_timeout_s=.502)
        diagnostic_reference = replace(self.reference, contact_intent={**self.reference.contact_intent,
                                                                       Limb.LEFT_FOOT: "foot_target"})
        with ExitStack() as stack:
            stack.enter_context(patch.object(ft, "ReadinessTracker", BoundaryTracker))
            stack.enter_context(patch.object(ft, "solve_foot_reference", side_effect=lambda m, q, *_:
                                             HandReferenceResult(tuple(q), True, 0, 0., 0., 1., "TEST_ONLY reference")))
            stack.enter_context(patch.object(ft, "_foot_residual", return_value=SimpleNamespace(
                support_point_world=target.point)))
            stack.enter_context(patch.object(ft, "estimate_support_torques", return_value={
                "torques_Nm": {n: 0. for n in self.reference.target_pose}, "root_balance_residual": 0.}))
            stack.enter_context(patch.object(ft, "solve_contact_pose", return_value=SimpleNamespace(
                admitted=True, reference=diagnostic_reference)))
            stack.enter_context(patch.object(ft, "_unexpected_loaded_contacts", return_value=()))
            stack.enter_context(patch.object(self.manager, "contact_snapshot", side_effect=snapshot))
            stack.enter_context(patch.object(self.manager.foot_sensor, "measure", side_effect=lambda m, d, *a, **k: feet(d)))
            stack.enter_context(patch.object(self.manager, "evaluate_and_update", return_value={}))
            stack.enter_context(patch("mujoco.mj_step", side_effect=clock))
            yield request, state

    def test_terminal_observers_reject_all_live_integration_channels_without_rollback(self):
        channels = (("qpos", 0), ("qvel", 0), ("ctrl", 0), ("qacc_warmstart", 0),
                    ("eq_active", 0), ("xfrc_applied", (0, 0)), ("qfrc_applied", 0), ("time", None))
        for executor, request in ((ft.execute_foot_transfer, self.foot), (sh.execute_single_hand, self.hand)):
            for name, index in channels:
                with self.subTest(executor=executor.__name__, channel=name):
                    self.setUp()
                    rows, changed = [], []

                    def observer(row, *_):
                        rows.append(row)
                        self.assertTrue(row["terminal"])
                        if name == "time":
                            self.data.time += .125
                        elif name == "eq_active":
                            self.data.eq_active[index] = not self.data.eq_active[index]
                        else:
                            getattr(self.data, name)[index] += .125
                        changed.append(_integration_state(self.model, self.data))

                    args = (*self.args[:4], object(), self.manager)
                    with patch("mujoco.mj_step", side_effect=AssertionError("invalid reference stepped")), \
                            self.assertRaisesRegex(RuntimeError, "Observer changed the authoritative"):
                        executor(*args, request, observer=observer)
                    self.assertEqual(len(rows), 1)
                    np.testing.assert_array_equal(_integration_state(self.model, self.data), changed[0])

    def test_running_observer_mutations_fail_before_another_native_command(self):
        channels = (("qpos", 0), ("qvel", 0), ("ctrl", 0), ("qacc_warmstart", 0),
                    ("time", None), ("qfrc_applied", 5), ("xfrc_applied", (0, 0)))
        for module, executor, request in ((ft, ft.execute_foot_transfer, self.foot),
                                          (sh, sh.execute_single_hand, self.hand)):
            for name, index in channels:
                with self.subTest(executor=executor.__name__, channel=name):
                    self.setUp()
                    rows, changed, terminal_states = [], [], []

                    def observer(row, *_):
                        rows.append(row)
                        if row["terminal"]:
                            terminal_states.append(_integration_state(self.model, self.data))
                        else:
                            if name == "time":
                                self.data.time += .125
                            else:
                                getattr(self.data, name)[index] += .125
                            changed.append(_integration_state(self.model, self.data))

                    with patch.object(module, "compute_pose_control", wraps=module.compute_pose_control) as control:
                        result = executor(*self.args, request, observer=observer)
                    self.assertEqual(result["status"], "CONTROL_FAILURE")
                    self.assertFalse(result["success"] or result["readiness"]["ready"])
                    self.assertIsNone(result["final_reference"])
                    self.assertIn("undeclared" if name in ("qfrc_applied", "xfrc_applied")
                                  else "outside the authoritative", result["reason"])
                    self.assertEqual(len(rows), 2)
                    self.assertFalse(rows[0]["terminal"])
                    self.assertTrue(rows[1]["terminal"])
                    self.assertEqual(rows[1], result["terminal_observation"])
                    self.assertEqual(control.call_count, result["steps"])
                    self.assertEqual(control.call_count, rows[0]["steps"])
                    np.testing.assert_array_equal(terminal_states[0], changed[0])
                    np.testing.assert_array_equal(_integration_state(self.model, self.data), changed[0])

    def test_success_terminal_callback_cannot_return_a_reference_after_live_write(self):
        for name in ("qpos", "ctrl"):
            with self.subTest(channel=name):
                self.setUp()
                rows, changed = [], []

                def observer(row, *_):
                    rows.append(row)
                    if row["terminal"]:
                        self.assertEqual(row["status"], "SUCCESS")
                        getattr(self.data, name)[0] += .125
                        changed.append(_integration_state(self.model, self.data))

                with self.foot_protocol() as (request, _), \
                        self.assertRaisesRegex(RuntimeError, "Observer changed the authoritative"):
                    ft.execute_foot_transfer(*self.args, request, observer=observer)
                self.assertEqual(sum(row["terminal"] for row in rows), 1)
                np.testing.assert_array_equal(_integration_state(self.model, self.data), changed[0])

    def test_observer_original_exception_identity_and_detached_writes(self):
        for executor, request in ((ft.execute_foot_transfer, self.foot), (sh.execute_single_hand, self.hand)):
            with self.subTest(executor=executor.__name__):
                self.setUp()
                before = _integration_state(self.model, self.data)
                error = ValueError("original callback failure")
                rows = []

                def observer(row, model, data):
                    rows.append(row)
                    row["qpos"][0] = 999.
                    model.body_mass[:] = 999.
                    data.qpos[:] = 999.
                    raise error

                with self.assertRaises(ValueError) as caught:
                    executor(*self.args[:4], object(), self.manager, request, observer=observer)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(rows), 1)
                np.testing.assert_array_equal(_integration_state(self.model, self.data), before)

    def test_generic_observer_mutation_propagates_without_a_terminal_result(self):
        rows = []

        def observer(row, *_):
            rows.append(row)
            self.data.xfrc_applied[0, 0] = .125

        with self.assertRaisesRegex(RuntimeError, "Observer changed the authoritative"):
            transfers.execute_transfer(*self.args[:4], object(), self.manager, self.generic, observer=observer)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.data.xfrc_applied[0, 0], .125)

    def test_detached_observer_writes_are_allowed_without_live_input_changes(self):
        for executor, request in ((ft.execute_foot_transfer, self.foot), (sh.execute_single_hand, self.hand)):
            with self.subTest(executor=executor.__name__):
                before = _integration_state(self.model, self.data)
                mass = self.model.body_mass.copy()
                rows = []

                def observer(row, model, data):
                    rows.append(row)
                    row["qpos"][0] = 999.
                    model.body_mass[:] = 999.
                    model.opt.timestep = 1.
                    data.qpos[:] = 999.
                    data.ctrl[:] = 999.
                    data.xfrc_applied[:] = 999.

                result = executor(*self.args[:4], object(), self.manager, request, observer=observer)
                self.assertEqual(result["status"], "INITIALIZATION_FAILURE")
                self.assertEqual(len(rows), 1)
                np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
                np.testing.assert_array_equal(self.model.body_mass, mass)

    def test_source_zero_force_recontact_is_rejected_in_both_epochs_and_all_landing_phases(self):
        for tick, phase in ((5, "LANDING"), (6, "LOAD"), (58, "SETTLE")):
            for epoch in ("applied", "endpoint"):
                with self.subTest(phase=phase, epoch=epoch):
                    self.setUp()
                    with self.foot_protocol(recontact_tick=tick, epoch=epoch) as (request, _):
                        result = ft.execute_foot_transfer(*self.args, request)
                        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
                    self.assertEqual(result["status"], "CONTACT_LOSS", result["reason"])
                    self.assertFalse(result["success"] or result["readiness"]["ready"])
                    self.assertIsNone(result["final_reference"])
                    self.assertTrue(result["released"])
                    self.assertEqual(result["phases"][-1], phase)
                    failure = result["guard_failure"]
                    self.assertEqual(failure["force_epoch"], "applied_preintegration_solve" if epoch == "applied"
                                     else "fresh_endpoint_solve")
                    self.assertEqual(failure["source_contact"], {"contact_count": 1, "normal_force_N": 0., "supporting": False})
                    self.assertEqual(result["steps"], tick)
                    if phase != "SETTLE":
                        self.assertIsNone(result["acquisition"])
                        self.assertFalse(any(e.get("event") == "ACQUIRED" for e in result["events"]))

    def test_light_target_touchdown_is_not_misclassified_as_source_or_full_support(self):
        with self.foot_protocol() as (request, _):
            result = ft.execute_foot_transfer(*self.args, replace(request, load_s=.002))
        self.assertEqual(result["status"], "LANDING_FAILURE")
        self.assertEqual(result["touchdown"]["normal_force_N"], .8)
        self.assertFalse(result["touchdown"]["supporting"])
        self.assertIsNone(result["guard_failure"])
        self.assertIsNone(result["acquisition"])

    def test_foot_fault_cleanup_preserves_replacements_and_unowned_channels(self):
        for replacement in (None, 1234., math.nan):
            with self.subTest(replacement=replacement):
                self.setUp()
                body, root = self.model.body("right_foot").id, self.model.body("climber_root").id
                rows, seen = [], []

                def guard(*_):
                    if self.data.xfrc_applied[body, 0] == 1500.:
                        seen.append(float(self.data.time))
                        if replacement is not None:
                            self.data.xfrc_applied[body, 0] = replacement
                            self.data.xfrc_applied[body, 1] = .375
                        self.data.xfrc_applied[root, 5] = .125
                        self.data.qfrc_applied[5] = .25
                        raise RuntimeError("boundary fault guard")
                    return ()

                with self.foot_protocol(target_tick=None) as (request, _), \
                        patch.object(ft, "_unexpected_loaded_contacts", side_effect=guard):
                    result = ft.execute_foot_transfer(*self.args, replace(request, reach_s=.4), fault="support_loss",
                                                      observer=lambda row, *_: rows.append(row))
                    self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
                self.assertEqual(len(seen), 1)
                self.assertEqual(result["status"], "CONTROL_FAILURE")
                self.assertIsNone(result["final_reference"])
                np.testing.assert_array_equal(self.data.xfrc_applied[body, :3],
                                              [0. if replacement is None else replacement,
                                               0. if replacement is None else .375, 0.])
                self.assertEqual(self.data.xfrc_applied[root, 5], .125)
                self.assertEqual(self.data.qfrc_applied[5], .25)
                self.assertEqual(rows[-1], result["terminal_observation"])
                np.testing.assert_array_equal(rows[-1]["external_force_world_N"], self.data.xfrc_applied)
                if replacement is not None and math.isnan(replacement):
                    self.assertFalse(rows[-1]["pose_available"])

    def test_observer_exception_after_fault_cleans_only_still_owned_components(self):
        body = self.model.body("right_foot").id
        error, rows = RuntimeError("original observer after fault"), []

        def observer(row, *_):
            rows.append(row)
            if self.data.xfrc_applied[body, 0] == 1500.:
                self.data.xfrc_applied[body, 0] = 1234.
                raise error

        with self.foot_protocol(target_tick=None) as (request, _), self.assertRaises(RuntimeError) as caught:
            ft.execute_foot_transfer(*self.args, replace(request, reach_s=.4), fault="support_loss", observer=observer)
        self.assertIs(caught.exception, error)
        self.assertFalse(any(row["terminal"] for row in rows))
        np.testing.assert_array_equal(self.data.xfrc_applied[body, :3], [1234., 0., 0.])

    def test_hand_fault_cleanup_preserves_same_component_replacement(self):
        body, root = self.model.body("left_foot").id, self.model.body("climber_root").id
        base_feet = self.manager.contact_snapshot().feet
        seen = []

        def guard(*args, **kwargs):
            if self.data.xfrc_applied[body, 0] == 1500.:
                seen.append(float(self.data.time))
                self.data.xfrc_applied[body, :3] = (1234., .375, 0.)
                self.data.xfrc_applied[root, 5] = .125
                self.data.qfrc_applied[5] = .25
                raise RuntimeError("hand boundary fault guard")
            return {}

        def clock(model, data):
            data.time += model.opt.timestep

        with patch.object(sh, "ReadinessTracker", BoundaryTracker), \
                patch.object(sh, "solve_hand_reference", side_effect=lambda m, q, *_:
                             HandReferenceResult(tuple(q), True, 0, 0., 0., 1., "TEST_ONLY reference")), \
                patch.object(sh, "estimate_support_torques", return_value={
                    "torques_Nm": {n: 0. for n in self.reference.target_pose}, "root_balance_residual": 0.,
                    "forces_world_N": [(0., 0., 10.), (0., 0., 10.), (0., 0., 0.)]}), \
                patch.object(sh, "_unexpected_loaded_contacts", return_value=()), \
                patch.object(self.manager.foot_sensor, "measure", return_value=base_feet), \
                patch.object(self.manager, "evaluate_and_update", side_effect=guard), \
                patch.object(self.manager, "can_attach", return_value=False), \
                patch("mujoco.mj_step", side_effect=clock):
            result = sh.execute_single_hand(*self.args, replace(self.hand, transfer_s=.002, support_s=.002,
                                                               reach_s=.4, capture_timeout_s=.002), fault="support_loss")
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["status"], "CONTROL_FAILURE", result["reason"])
        self.assertIsNone(result["final_reference"])
        np.testing.assert_array_equal(self.data.xfrc_applied[body, :3], [1234., .375, 0.])
        self.assertEqual(self.data.xfrc_applied[root, 5], .125)
        self.assertEqual(self.data.qfrc_applied[5], .25)
        np.testing.assert_array_equal(result["terminal_observation"]["external_force_world_N"], self.data.xfrc_applied)

    def test_generic_mismatch_rebuilds_terminal_truth_without_mutating_underlying(self):
        initial = get_state_summary(self.model, self.data, self.manager)
        command = ft.compute_pose_control(self.model, self.data, dict(self.reference.target_pose))
        terminal = {"terminal": True, "status": "SUCCESS", "reason": "stub success", "time_s": initial.time,
                    "qpos": list(initial.qpos), "ctrl": self.data.ctrl.tolist(),
                    "readiness": {"ready": True, "duration": .5}, "tracking_error_m": .001,
                    "command": {"commanded_Nm": command.commanded_Nm, "limits_Nm": command.limits_Nm}}
        before = get_state_summary(self.model, self.data, self.manager)
        underlying = {"success": True, "status": "SUCCESS", "reason": "stub success", "initial_state": before,
                      "final_state": before, "final_reference": self.reference, "steps": 0, "duration_s": 0.,
                      "dt_s": .002, "events": [], "phases": [], "readiness": terminal["readiness"],
                      "terminal_observation": terminal, "capture": None}

        def executor(*args, **kwargs):
            self.data.qpos[0] += 1e-5
            self.data.ctrl[0] += .125
            return underlying

        observer = Mock()
        with patch.object(ft, "execute_foot_transfer", side_effect=executor):
            result = transfers.execute_transfer(*self.args, self.generic, observer=observer)
        actual = get_state_summary(self.model, self.data, self.manager)
        row = result["terminal_observation"]
        self.assertEqual(result["status"], row["status"])
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertFalse(result["success"] or row["readiness"]["ready"])
        self.assertIsNone(result["final_reference"])
        self.assertEqual(result["final_state"], actual)
        self.assertEqual(row["readiness"], result["readiness"])
        self.assertEqual(row["qpos"], list(actual.qpos))
        self.assertEqual(row["ctrl"], list(actual.ctrl))
        self.assertIsNone(row["tracking_error_m"])
        self.assertFalse(row["command"]["matches_last_request"])
        np.testing.assert_array_equal(row["command"]["commanded_Nm"], self.data.ctrl * self.model.actuator_gear[:, 0])
        self.assertEqual(terminal["status"], "SUCCESS")
        self.assertTrue(terminal["readiness"]["ready"])
        self.assertIs(result["underlying_result"], underlying)
        observer.assert_not_called()

    def test_generic_malformed_policy_and_overflow_are_atomic_for_moves_and_sequences(self):
        invalid = (replace(self.generic, hand_acquisition_policy=np.array(["endpoint_settle", "first_eligible"])),
                   replace(self.generic, hand_acquisition_policy=[]),
                   replace(self.generic, hand_reach_s=1e308), replace(self.generic, hand_reach_s=10 ** 1000),
                   replace(self.generic, hand_reach_s=math.nan), replace(self.generic, hand_reach_s=math.inf),
                   replace(self.generic, hand_reach_s=True), replace(self.generic, root_linear_max_m_s=10 ** 1000))
        observer = Mock()
        for request in invalid:
            with self.subTest(policy=type(request.hand_acquisition_policy), duration=type(request.hand_reach_s)):
                before = _integration_state(self.model, self.data)
                events = copy.deepcopy((self.manager.capture_events, self.manager.releases))
                with patch.object(ft, "execute_foot_transfer", side_effect=AssertionError("invalid dispatched")), \
                        patch.object(sh, "execute_single_hand", side_effect=AssertionError("invalid dispatched")), \
                        patch("mujoco.mj_step", side_effect=AssertionError("invalid stepped")):
                    result = transfers.execute_transfer(*self.args, request, observer=observer)
                    sequence = transfers.execute_transfer_sequence(*self.args[:4], (self.generic, request),
                                                                    self.reference, self.manager, observer=observer)
                self.assertEqual(result["status"], "INVALID_REQUEST")
                self.assertEqual((result["steps"], result["duration_s"]), (0, 0.))
                self.assertEqual(sequence["status"], "INVALID_REQUEST")
                self.assertEqual((sequence["steps"], sequence["completed_moves"], sequence["failed_index"]), (0, 0, 1))
                self.assertIsNone(result["final_reference"])
                self.assertIsNone(sequence["final_reference"])
                np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
                self.assertEqual((self.manager.capture_events, self.manager.releases), events)
        observer.assert_not_called()

    def test_foot_overflow_duration_is_rejected_before_commands(self):
        before = _integration_state(self.model, self.data)
        with patch.object(ft, "compute_pose_control", side_effect=AssertionError("invalid commanded")), \
                patch("mujoco.mj_step", side_effect=AssertionError("invalid stepped")):
            result = ft.execute_foot_transfer(*self.args, replace(self.foot, unload_s=1e308))
        self.assertEqual(result["status"], "REACH_INFEASIBLE")
        self.assertEqual(result["reason"], "Durations must be positive integral native intervals")
        self.assertEqual(result["steps"], 0)
        self.assertIsNone(result["final_reference"])
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
