"""Generic boundary contracts and two-hand native continuity at 2ms and 1ms."""
import copy
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import json
from types import MappingProxyType
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from boulder_v1 import foot_transfer, single_hand, transfers
from boulder_v1.contact import CAPTURE_DISTANCE
from boulder_v1.grasp import AttachmentStateError, GraspManager
from boulder_v1.locomotion import get_state_summary
from boulder_v1.schema import Limb
from boulder_v1.static_state import _integration_state, initialize_static_reference
from test_single_hand import NativeAudit, capture_gate, model_parameters


def json_value(value):
    # Match validate_transition._json_value without deepcopying MappingProxyType.
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(json_value(key)): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class TransferTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cls.runs.clear()

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for dt in (.002, .001):
            audits, arguments, rows, failures, boundaries = [], [], [], [], []
            native_execute = single_hand.execute_single_hand
            native_attach = GraspManager.attach

            def attach(manager, limb, region, force=False):
                accepted = native_attach(manager, limb, region, force=force)
                if audits and manager is audits[-1].manager:
                    audit = audits[-1]
                    measurement = manager.capture_events[-1] if accepted else manager._capture_measurement(limb, region)
                    audit.record(not force, "forced generic hand capture")
                    audit.record(not accepted or capture_gate(measurement), "generic capture outside unchanged gate")
                    audit.captures.append({"measurement": measurement, "accepted": accepted})
                return accepted

            def execute(model, data, scene, profile, reference, manager, request, **kwargs):
                args = (model, data, scene, profile, reference, manager)
                arguments.append(args)
                boundaries.append(_integration_state(model, data).copy())
                if request.limb == Limb.LEFT_HAND:
                    # Native negative from precisely the RH endpoint, with no
                    # replay, initialization, reset, or additional successful run.
                    contacts = manager.contact_configuration()
                    bad = transfers.TransferRequest(request.limb, request.source, "unreachable_target", contacts)
                    before = _integration_state(model, data)
                    with patch.object(single_hand, "execute_single_hand", new=native_execute), \
                            patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("negative commanded")), \
                            patch("mujoco.mj_step", side_effect=AssertionError("negative stepped")):
                        failure = transfers.execute_transfer_sequence(model, data, scene, profile, (bad, bad),
                                                                       reference, manager)
                    np.testing.assert_array_equal(_integration_state(model, data), before)
                    failures.append(failure)
                audit = NativeAudit(*args, request, None)
                audits.append(audit)
                with ExitStack() as stack:
                    audit.install(stack)
                    result = native_execute(*args, request, **kwargs)
                    audit.unchanged()
                audit.after_model = model_parameters(model)
                boundaries.append(_integration_state(model, data).copy())
                return result

            def observer(row, display_model, display_data):
                audit = audits[-1]
                if display_model is audit.model or display_data is audit.data:
                    raise AssertionError("Observer received live physics")
                np.testing.assert_array_equal(display_data.qpos, audit.data.qpos)
                np.testing.assert_array_equal(row["qpos"], audit.data.qpos)
                rows.append(copy.deepcopy(row))
                row["qpos"][0] = 999.
                row["move_index"] = 999
                display_model.body_mass[:] = 999.
                display_model.eq_data[:] = 999.
                display_model.opt.timestep = 1.
                display_data.qpos[:] = 999.
                display_data.qvel[:] = 999.
                display_data.ctrl[:] = 999.
                display_data.eq_active[:] = False
                display_data.qacc_warmstart[:] = 999.
                display_data.xfrc_applied[:] = 999.

            with patch.object(single_hand, "execute_single_hand", new=execute), \
                    patch.object(GraspManager, "attach", new=attach):
                result = transfers.run_transfer_benchmark(dt, observer=observer)
            cls.runs[dt] = (result, audits, arguments, rows, failures, boundaries)
            print("TRANSFER_SEQUENCE_JSON " + json.dumps(json_value({
                key: result[key] for key in ("status", "reason", "dt_s", "total_steps", "duration_s",
                                            "completed_moves", "failed_index", "final_time_s", "contact_trace")
            } | {"moves": [{key: move[key] for key in (
                "moving_limb", "source", "target", "support_contacts", "status", "steps", "duration_s",
                "release_time_s", "reach_start_time_s", "capture_time_s", "capture_error_m", "capture_margin_m",
                "foot_acquisition_time_s", "readiness_time_s", "readiness", "three_point")}
                for move in result.get("moves", [])]}), allow_nan=False), flush=True)

    def setUp(self):
        self.result, self.audits, self.arguments, self.rows, self.failures, self.boundaries = self.runs[.002]
        self.args = self.arguments[-1]
        self.model, self.data, self.scene, self.profile, _, self.manager = self.args
        self.reference = self.result["final_reference"]
        self.args = (self.model, self.data, self.scene, self.profile, self.reference, self.manager)
        contacts = self.manager.contact_configuration()
        self.request = transfers.TransferRequest(Limb.LEFT_HAND, contacts[Limb.LEFT_HAND], "left_hand", contacts)

    def test_two_hand_native_success_and_exact_state_reference_continuity(self):
        for dt, (result, audits, args, rows, _, boundaries) in self.runs.items():
            with self.subTest(dt=dt):
                self.assertTrue(result["success"], result["reason"])
                self.assertEqual((result["completed_moves"], result["failed_index"]), (2, None))
                self.assertEqual(len(audits), 2)
                first, second = result["moves"]
                self.assertEqual(first["moving_limb"], Limb.RIGHT_HAND)
                self.assertEqual(second["moving_limb"], Limb.LEFT_HAND)
                self.assertEqual(first["initial_state"], result["initial_static"]["final_state"])
                self.assertAlmostEqual(first["initial_state"].time, 2., places=10)
                self.assertEqual(second["initial_state"], first["final_state"])
                np.testing.assert_array_equal(boundaries[1], boundaries[2])
                self.assertIs(args[1][4], first["final_reference"])
                for index in (0, 1, 2, 3, 5):
                    self.assertIs(args[1][index], args[0][index])
                self.assertEqual(dict(args[0][2].start_configuration), first["initial_contacts"])
                self.assertEqual(dict(args[0][4].contact_intent), first["initial_contacts"])
                self.assertEqual(dict(args[1][4].contact_intent), first["final_contacts"])
                self.assertEqual(dict(second["source_contacts"]), first["final_contacts"])
                self.assertIs(result["final_reference"], second["final_reference"])
                self.assertEqual(result["final_state"], get_state_summary(args[1][0], args[1][1], args[1][5]))
                self.assertEqual(result["total_steps"], sum(len(audit.steps) for audit in audits))
                self.assertEqual(result["total_steps"], first["steps"] + second["steps"])
                self.assertEqual(result["duration_s"], result["total_steps"] * dt)
                self.assertEqual(result["total_time_s"], result["duration_s"])
                self.assertAlmostEqual(result["final_time_s"] - result["initial_time_s"], result["duration_s"], places=9)
                self.assertEqual(result["contact_trace"], [first["initial_contacts"], first["final_contacts"], second["final_contacts"]])
                self.assertEqual(result["current_contacts"], second["final_contacts"])
                self.assertFalse(any("foot" in args[0][0].equality(i).name for i in range(args[0][0].neq)))
                self.assertEqual({row["move_index"] for row in rows}, {0, 1})
                self.assertTrue(all(row["move_count"] == 2 for row in rows))
                self.assertTrue(rows[-1]["terminal"])
                for index, move in enumerate(result["moves"]):
                    terminal = {k: v for k, v in next(row for row in reversed(rows)
                                if row["move_index"] == index).items()
                                if k not in ("move_index", "move_count")}
                    self.assertEqual(terminal, move["terminal_observation"])
                json.dumps(json_value(result), allow_nan=False)

    def test_each_move_retains_native_physics_and_capture_admission(self):
        for dt, (result, audits, _, _, _, _) in self.runs.items():
            for move, audit in zip(result["moves"], audits):
                with self.subTest(dt=dt, limb=move["moving_limb"]):
                    self.assertEqual(audit.violations, set())
                    self.assertEqual((len(audit.steps), len(audit.controls), len(audit.samples)), (move["steps"],) * 3)
                    for name, original in audit.before_model.items():
                        np.testing.assert_array_equal(audit.after_model[name], original, err_msg=name)
                    self.assertEqual(audit.reference_before,
                                     (audit.reference.qpos, dict(audit.reference.target_pose), dict(audit.reference.contact_intent)))
                    self.assertIs(move["final_reference"], move["underlying_result"]["final_reference"])
                    self.assertTrue(move["admission"]["admitted"])
                    supports = {l: h for l, h in move["source_contacts"].items() if l != move["moving_limb"]}
                    self.assertEqual(dict(move["support_contacts"]), supports)
                    self.assertEqual(dict(move["admission"]["contacts"]), supports)
                    self.assertEqual(move["support_admission"]["contacts"], supports)
                    self.assertTrue(move["admission"]["source_reference_admitted"])
                    self.assertTrue(capture_gate(move["capture"]))
                    self.assertGreater(move["capture_margin_m"], .0009)
                    self.assertEqual(move["capture_margin_m"], CAPTURE_DISTANCE - move["capture_error_m"])
                    self.assertEqual(move["capture"]["acquisition_policy"], "endpoint_settle")
                    self.assertLess(move["release_time_s"], move["reach_start_time_s"])
                    self.assertLess(move["reach_start_time_s"], move["capture_time_s"])
                    self.assertLess(move["capture_time_s"], move["readiness_time_s"])
                    self.assertIsNone(move["foot_acquisition_time_s"])
                    self.assertTrue(move["readiness"]["ready"])
                    self.assertGreaterEqual(move["readiness"]["duration"], .5 - 1e-10)
                    self.assertEqual(len(audit.detaches), 1)
                    self.assertEqual(audit.detaches[0]["limb"], move["moving_limb"])
                    other = Limb.LEFT_HAND if move["moving_limb"] == Limb.RIGHT_HAND else Limb.RIGHT_HAND
                    constraint = move["initial_state"].attachment_constraints[other]
                    equality = audit.model.equality(constraint).id
                    self.assertTrue(all(row["eq_active"][equality] for row in move["samples"]))
                    for row in move["samples"]:
                        self.assertNotIn("move_index", row)
                        self.assertLessEqual(row["root_linear_m_s"], .10)
                        self.assertLessEqual(row["root_angular_rad_s"], .50)
                        self.assertLessEqual(row["joint_max_rad_s"], 1.)
                        self.assertFalse(np.any(row["external_force_world_N"]) or np.any(row["qfrc_applied"]))
                        self.assertTrue(row["hands"][other.value]["active"])
                        self.assertEqual(row["hands"][other.value]["region_id"], move["source_contacts"][other])
                    final = move["final_reference"]
                    with self.assertRaises(TypeError):
                        final.contact_intent[other] = "changed"
                    before = _integration_state(audit.model, audit.data)
                    initialize_static_reference(audit.model, mujoco.MjData(audit.model),
                                                replace(audit.scene, start_configuration=final.contact_intent),
                                                audit.profile, final.qpos, final.contact_intent)
                    np.testing.assert_array_equal(_integration_state(audit.model, audit.data), before)

    def reject(self, request):
        before = _integration_state(self.model, self.data)
        events = copy.deepcopy((self.manager.capture_events, self.manager.releases))
        observer = Mock()
        with patch.object(single_hand, "execute_single_hand", side_effect=AssertionError("invalid hand dispatch")), \
                patch.object(foot_transfer, "execute_foot_transfer", side_effect=AssertionError("invalid foot dispatch")), \
                patch.object(transfers, "initialize_static_reference", side_effect=AssertionError("invalid rebuilt source")), \
                patch("mujoco.mj_step", side_effect=AssertionError("invalid integrated")):
            result = transfers.execute_transfer(*self.args, request, observer=observer)
        self.assertFalse(result["success"] or result["released"] or result["admission"]["admitted"])
        self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
        self.assertEqual(result["initial_state"], result["final_state"])
        self.assertEqual(result["final_state"], get_state_summary(self.model, self.data, self.manager))
        self.assertIsNone(result["final_reference"])
        self.assertIsNone(result["underlying_result"])
        observer.assert_not_called()
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        self.assertEqual((self.manager.capture_events, self.manager.releases), events)
        return result

    def test_malformed_generic_requests_are_atomic_without_source_rebuild(self):
        contacts = dict(self.request.source_contacts)
        supports = {l: h for l, h in contacts.items() if l != self.request.limb}
        alias = {l.value: h for l, h in contacts.items()}
        bad = (object(), replace(self.request, limb="LEFT_HAND"), replace(self.request, source_contacts=alias),
               replace(self.request, source_contacts={Limb.LEFT_HAND: "left_reach_target"}),
               replace(self.request, source_contacts={**contacts, Limb.LEFT_FOOT: .001}),
               replace(self.request, source_contacts={**contacts, Limb.LEFT_FOOT: "floor"}),
               replace(self.request, source_contacts={**contacts, Limb.RIGHT_HAND: "right_hand"}),
               replace(self.request, source="left_hand"), replace(self.request, target=self.request.source),
               replace(self.request, target="missing"), replace(self.request, target="foot_target"),
               replace(self.request, support_contacts=contacts), replace(self.request, support_contacts={}),
               replace(self.request, support_contacts={l.value: h for l, h in supports.items()}),
               replace(self.request, root_linear_max_m_s=.101), replace(self.request, root_angular_max_rad_s=.501),
               replace(self.request, joint_max_rad_s=1.01), replace(self.request, joint_max_rad_s=.9),
               replace(self.request, hand_reach_s=.0025), replace(self.request, hand_reach_s=True),
               replace(self.request, hand_acquisition_policy="unknown"), replace(self.request, primitive=[]))
        for request in bad:
            with self.subTest(request=request):
                self.assertEqual(self.reject(request)["status"], "INVALID_REQUEST")
        for primitive in ("balance", "foot"):
            self.assertEqual(self.reject(replace(self.request, primitive=primitive))["status"], "UNSUPPORTED_PRIMITIVE")

    def test_contact_maps_are_detached_and_immutable(self):
        contacts = dict(self.request.source_contacts)
        supports = {l: h for l, h in contacts.items() if l != self.request.limb}
        request = replace(self.request, source_contacts=contacts, support_contacts=supports)
        contacts[Limb.RIGHT_HAND] = "changed"
        supports.clear()
        self.assertEqual(dict(request.source_contacts), dict(self.request.source_contacts))
        self.assertEqual(len(request.support_contacts), 3)
        for mapping in (request.source_contacts, request.support_contacts):
            self.assertIsInstance(mapping, MappingProxyType)
            with self.assertRaises(TypeError):
                mapping[Limb.RIGHT_HAND] = "changed"

    def test_native_unreachable_from_rh_endpoint_freezes_terminal_and_stops(self):
        for dt, (result, _, _, _, failures, _) in self.runs.items():
            with self.subTest(dt=dt):
                failure = failures[0]
                self.assertEqual(failure["status"], "REACH_INFEASIBLE")
                self.assertEqual((failure["completed_moves"], failure["failed_index"], failure["total_steps"]), (0, 0, 0))
                self.assertEqual(len(failure["moves"]), 1)
                self.assertEqual(failure["initial_state"], result["moves"][0]["final_state"])
                self.assertEqual(failure["final_state"], failure["initial_state"])
                self.assertIsNone(failure["final_reference"])
                move = failure["moves"][0]
                self.assertFalse(move["released"] or move["readiness"]["ready"])
                self.assertEqual(dict(move["admission"]["contacts"]), dict(result["moves"][1]["support_contacts"]))
                self.assertTrue(move["admission"]["source_reference_admitted"])
                self.assertIsNone(move["final_reference"])
                self.assertIn("not a global infeasibility proof", move["reason"])
                json.dumps(json_value(failure), allow_nan=False)

    def test_generic_dispatch_reuses_requests_sessions_defaults_and_underlying_fields(self):
        contacts = self.manager.contact_configuration()
        for limb, target in ((Limb.RIGHT_HAND, "right_hand"), (Limb.LEFT_HAND, "left_hand"),
                             (Limb.LEFT_FOOT, "foot_target")):
            with self.subTest(limb=limb):
                supports = {l: h for l, h in contacts.items() if l != limb}
                request = transfers.TransferRequest(limb, contacts[limb], target, contacts, supports)
                state = get_state_summary(self.model, self.data, self.manager)
                native = {"success": True, "status": "SUCCESS", "reason": "dispatch contract only",
                          "initial_state": state, "final_state": state, "final_reference": self.reference,
                          "steps": 0, "duration_s": 0., "events": [], "capture": None,
                          "readiness": {"ready": True, "time": state.time},
                          "support_admission": {"admitted": True, "contacts": supports},
                          "terminal_observation": {"source_reference_admitted": True}, "snapshots": ["preserved"]}
                module, name = (single_hand, "execute_single_hand") if limb.is_hand else (foot_transfer, "execute_foot_transfer")
                with patch.object(module, name, return_value=native) as executor:
                    result = transfers.execute_transfer(*self.args, request, keep_samples=False)
                call = executor.call_args
                for actual, expected in zip(call.args[:6], self.args):
                    self.assertIs(actual, expected)
                primitive_request = call.args[6]
                if limb.is_hand:
                    self.assertIsInstance(primitive_request, single_hand.SingleHandRequest)
                    self.assertEqual(primitive_request.acquisition_policy, "endpoint_settle")
                    self.assertEqual(primitive_request.reach_s, 4.)
                else:
                    self.assertEqual(primitive_request, foot_transfer.FootRequest(limb, contacts[limb], target))
                self.assertEqual((primitive_request.limb, primitive_request.source, primitive_request.target),
                                 (limb, request.source, target))
                self.assertEqual(call.kwargs, {"observer": None, "keep_samples": False})
                self.assertIs(result["underlying_result"], native)
                self.assertIs(result["snapshots"], native["snapshots"])
                self.assertIs(result["final_reference"], self.reference)
                self.assertNotIn("samples", result)

    def test_low_level_reference_rom_force_and_support_failures_are_not_success(self):
        outside = np.array(self.reference.qpos)
        outside[self.model.joint("left_knee").qposadr[0]] = 10.
        for reference, status in ((object(), "INITIALIZATION_FAILURE"),
                                  (replace(self.reference, qpos=outside), "ROM_FAILURE")):
            before = _integration_state(self.model, self.data)
            args = (*self.args[:4], reference, self.manager)
            result = transfers.execute_transfer(*args, self.request)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["steps"], 0)
            self.assertIsNone(result["final_reference"])
            np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        for field, index in (("qfrc_applied", 5), ("xfrc_applied", (0, 5))):
            channel = getattr(self.data, field)
            channel[index] = .125
            try:
                self.assertEqual(self.reject(self.request)["status"], "INITIALIZATION_FAILURE")
                self.assertEqual(channel[index], .125)
            finally:
                channel[index] = 0.
        for reason in ("Support reference root balance failed", "Support reference exceeds remaining-hand capacity",
                       "Support reference violates unilateral compression/friction", "Support reference exceeds original motor capability"):
            with self.subTest(reason=reason):
                before = _integration_state(self.model, self.data)
                with patch.object(single_hand, "estimate_support_torques", side_effect=ValueError(reason)), \
                        patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("support failure commanded")), \
                        patch("mujoco.mj_step", side_effect=AssertionError("support failure stepped")):
                    result = transfers.execute_transfer(*self.args, self.request)
                self.assertEqual(result["status"], "THREE_POINT_SUPPORT_FAILURE")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["steps"], 0)
                self.assertFalse(result["admission"]["admitted"])
                np.testing.assert_array_equal(_integration_state(self.model, self.data), before)

    def test_sequence_validates_every_request_before_any_move(self):
        observer = Mock()
        for requests in ([], None, "sequence", [self.request, object()],
                         [self.request, replace(self.request, target="missing")]):
            before = _integration_state(self.model, self.data)
            with patch.object(transfers, "execute_transfer", side_effect=AssertionError("invalid chain dispatched")):
                result = transfers.execute_transfer_sequence(*self.args[:4], requests, self.reference,
                                                             self.manager, observer=observer)
            self.assertEqual(result["status"], "INVALID_REQUEST")
            self.assertEqual((result["completed_moves"], result["total_steps"], result["moves"]), (0, 0, []))
            self.assertEqual(result["initial_state"], result["final_state"])
            np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        observer.assert_not_called()

    def test_sequence_halts_on_second_failure_and_missing_final_reference(self):
        first = self.result["moves"][0]
        second = self.failures[0]["moves"][0]
        requests = [move["request"] for move in self.result["moves"]]
        requests.append(requests[-1])
        for failed in (second, {**second, "success": True, "status": "SUCCESS"}):
            with patch.object(transfers, "execute_transfer", side_effect=[first, failed]) as executor:
                result = transfers.execute_transfer_sequence(*self.args[:4], requests, self.reference, self.manager)
            self.assertFalse(result["success"])
            self.assertEqual((result["completed_moves"], result["failed_index"]), (1, 1))
            self.assertEqual(executor.call_count, 2)
            self.assertIs(executor.call_args_list[1].args[4], first["final_reference"])
            self.assertEqual(result["total_steps"], first["steps"])
            self.assertIsNone(result["final_reference"])

    def test_changed_subsequent_contact_intent_is_not_rebased(self):
        contacts = dict(self.request.source_contacts)
        contacts[Limb.RIGHT_HAND] = "right_hand"
        bad = replace(self.request, source_contacts=contacts)
        result = transfers.execute_transfer_sequence(*self.args[:4], (bad,), self.reference, self.manager)
        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual((result["failed_index"], result["total_steps"]), (0, 0))
        self.assertEqual(result["initial_state"], result["final_state"])

    def test_session_identity_and_observer_exception_are_not_hidden(self):
        with self.assertRaises(AttachmentStateError):
            transfers.execute_transfer(self.model, mujoco.MjData(self.model), self.scene, self.profile,
                                       self.reference, self.manager, self.request)
        with self.assertRaises(AttachmentStateError):
            transfers.execute_transfer(self.model, self.data, replace(self.scene), self.profile,
                                       self.reference, self.manager, self.request)
        error = ValueError("observer exception identity")
        observer = Mock(side_effect=error)
        def execute(*args, observer, **kwargs):
            observer({"phase": "PREFLIGHT"}, object(), object())
        before = _integration_state(self.model, self.data)
        with patch.object(single_hand, "execute_single_hand", side_effect=execute):
            with self.assertRaises(ValueError) as caught:
                transfers.execute_transfer_sequence(*self.args[:4], (self.request,), self.reference,
                                                   self.manager, observer=observer)
        self.assertIs(caught.exception, error)
        self.assertEqual(observer.call_count, 1)
        self.assertEqual(observer.call_args.args[0]["move_index"], 0)
        self.assertEqual(observer.call_args.args[0]["move_count"], 1)
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)

    def test_fixture_runner_uses_common_setup_once_and_declared_second_target(self):
        # Delivery contract only: native chain acceptance is cached above.
        model, data, scene, profile, seed = transfers.make_transfer_fixture()
        fake_manager = Mock()
        fake_reference = Mock(contact_intent=scene.start_configuration)
        fake_retarget = Mock(admitted=True, qpos=seed)
        with patch.object(transfers, "make_transfer_fixture", return_value=(model, data, scene, profile, seed)) as fixture, \
                patch.object(transfers, "solve_contact_pose", return_value=fake_retarget) as solve, \
                patch.object(transfers, "initialize_static_reference", return_value=(fake_reference, fake_manager)) as initialize, \
                patch.object(transfers, "get_state_summary", return_value="source snapshot"), \
                patch.object(transfers, "execute_static_hold", return_value={"success": True}) as hold, \
                patch.object(transfers, "execute_transfer_sequence", return_value={"success": False}) as sequence:
            result = transfers.run_transfer_benchmark(negative="second_unreachable", profile=profile, keep_samples=False)
        fixture.assert_called_once_with(.002, profile=profile)
        solve.assert_called_once()
        initialize.assert_called_once()
        hold.assert_called_once()
        self.assertEqual(hold.call_args.kwargs, {"duration": 2., "settle": 1., "score_window": .5, "keep_samples": False})
        first, second = sequence.call_args.args[4]
        self.assertEqual((first.limb, first.source, first.target), (Limb.RIGHT_HAND, "right_hand", "reach_target"))
        self.assertEqual((second.limb, second.source, second.target), (Limb.LEFT_HAND, "left_hand", "unreachable_target"))
        self.assertEqual(dict(second.source_contacts), {**scene.start_configuration, Limb.RIGHT_HAND: first.target})
        self.assertIs(sequence.call_args.args[5], fake_reference)
        self.assertIs(sequence.call_args.args[6], fake_manager)
        self.assertIs(result["initial_reference"], fake_reference)
        self.assertEqual(result["setup_initial_state"], "source snapshot")
        with patch.object(transfers, "make_transfer_fixture", side_effect=AssertionError("invalid benchmark compiled")):
            for kwargs in ({"kind": "route"}, {"negative": "support_loss"},
                           {"kind": "foot", "negative": "second_unreachable"}):
                with self.assertRaises(ValueError):
                    transfers.run_transfer_benchmark(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
