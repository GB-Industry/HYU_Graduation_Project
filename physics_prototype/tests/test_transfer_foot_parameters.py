"""Parser/dispatch contracts only; mocked contacts/executors do not certify physics."""
import copy
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from boulder_v1 import foot_transfer as ft, single_hand as sh, transfers
from boulder_v1.contact_geometry import Frame
from boulder_v1.locomotion import get_state_summary
from boulder_v1.schema import Limb
from boulder_v1.static_state import _integration_state, initialize_static_reference
from boulder_v1.whole_body_motion import WholeBodyMotion
from scripts.validate_transition import _json_value


class TransferFootParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, data, scene, profile, seed = transfers.make_transfer_fixture()
        reference, manager = initialize_static_reference(model, data, scene, profile, seed,
                                                        scene.start_configuration)
        cls.source = model, data, scene, profile, reference, manager

    def setUp(self):
        self.args = self.source
        self.model, self.data, self.scene, self.profile, self.reference, self.manager = self.args
        self.contacts = dict(self.reference.contact_intent)
        self.foot = ft.FootRequest(Limb.LEFT_FOOT, self.contacts[Limb.LEFT_FOOT], "foot_target")
        self.request = transfers.TransferRequest(self.foot.limb, self.foot.source, self.foot.target,
                                                 self.contacts)
        # This unstepped static session is finite, but these four contacts are a
        # TEST_ONLY generic-boundary stub, not measured native support evidence.
        self.state = replace(get_state_summary(self.model, self.data, self.manager),
                             contact_configuration=dict(self.contacts))
        summary = patch.object(transfers, "get_state_summary", return_value=self.state)
        summary.start()
        self.addCleanup(summary.stop)

    def dispatch(self, request):
        native = {"success": True, "status": "SUCCESS", "reason": "TEST_ONLY dispatch contract",
                  "initial_state": self.state, "final_state": self.state,
                  "final_reference": self.reference, "steps": 0, "duration_s": 0.,
                  "events": [], "capture": None, "readiness": {"ready": True, "time": self.state.time},
                  "support_admission": {"admitted": False},
                  "terminal_observation": {"source_reference_admitted": False}, "snapshots": []}
        observer = Mock()
        before = _integration_state(self.model, self.data)
        with patch.object(ft, "execute_foot_transfer", return_value=native) as executor:
            result = transfers.execute_transfer(*self.args, request, observer=observer, keep_samples=False)
        self.assertTrue(result["success"], result["reason"])
        self.assertIs(result["underlying_result"], native)
        self.assertIs(result["final_reference"], self.reference)
        self.assertIs(result["snapshots"], native["snapshots"])
        self.assertEqual(result["final_state"], self.state)
        self.assertNotIn("samples", result)
        executor.assert_called_once()
        for actual, expected in zip(executor.call_args.args[:6], self.args):
            self.assertIs(actual, expected)
        expected_options = {"observer": observer, "keep_samples": False}
        if request.whole_body is not None:
            expected_options["whole_body"] = request.whole_body
            self.assertIs(executor.call_args.kwargs["whole_body"], request.whole_body)
        self.assertEqual(executor.call_args.kwargs, expected_options)
        observer.assert_not_called()
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        return executor.call_args.args[6]

    def reject(self, request):
        before = _integration_state(self.model, self.data)
        ctrl, time = self.data.ctrl.copy(), float(self.data.time)
        events = copy.deepcopy((self.manager.capture_events, self.manager.releases))
        observer = Mock()
        with ExitStack() as stack:
            forbidden = [stack.enter_context(patch.object(module, name, side_effect=AssertionError(name)))
                         for module, name in ((ft, "execute_foot_transfer"), (sh, "execute_single_hand"),
                                              (ft, "compute_pose_control"), (sh, "compute_pose_control"),
                                              (transfers, "initialize_static_reference"), (mujoco, "mj_step"))]
            result = transfers.execute_transfer(*self.args, request, observer=observer)
            with patch.object(transfers, "execute_transfer", side_effect=AssertionError("sequence moved")) as move:
                sequence = transfers.execute_transfer_sequence(*self.args[:4], (self.request, request),
                                                                self.reference, self.manager, observer=observer)
            move.assert_not_called()
            for callback in forbidden:
                callback.assert_not_called()
        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
        self.assertFalse(result["success"] or result["released"] or result["admission"]["admitted"])
        self.assertIsNone(result["underlying_result"])
        self.assertIsNone(result["final_reference"])
        self.assertEqual(sequence["status"], "INVALID_REQUEST")
        self.assertEqual((sequence["completed_moves"], sequence["failed_index"], sequence["steps"],
                          sequence["duration_s"], sequence["moves"]), (0, 1, 0, 0., []))
        self.assertIsNone(sequence["final_reference"])
        for outcome in (result, sequence):
            self.assertEqual(outcome["initial_state"], outcome["final_state"])
            self.assertEqual(outcome["initial_integration_state"], outcome["final_integration_state"])
        observer.assert_not_called()
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        np.testing.assert_array_equal(self.data.ctrl, ctrl)
        self.assertEqual(float(self.data.time), time)
        self.assertEqual((self.manager.capture_events, self.manager.releases), events)

    def test_default_dispatch_keeps_exact_three_argument_constructor(self):
        self.assertIsNone(self.request.foot_request)
        with patch.object(ft, "FootRequest", wraps=ft.FootRequest) as constructor:
            primitive = self.dispatch(self.request)
        constructor.assert_called_once_with(self.foot.limb, self.foot.source, self.foot.target)
        self.assertEqual(primitive, self.foot)

    def test_explicit_ascending_parameters_and_separate_motion_dispatch_by_identity(self):
        motion = WholeBodyMotion(Frame((0., 0., 0.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                                 {"waist_yaw": 0., "waist_pitch": .07, "waist_roll": 0.}, prepare_s=4.)
        for goal in (None, motion):
            with self.subTest(whole_body=goal is not None):
                foot = replace(self.foot, lift_m=.082 + .02, unload_s=2. if goal is None else goal.prepare_s,
                               lift_s=1.5, reach_s=3., load_s=2.)
                request = replace(self.request, foot_request=foot, whole_body=goal)
                before = asdict(foot)
                self.assertIs(self.dispatch(request), foot)
                self.assertEqual(asdict(foot), before)
                self.assertAlmostEqual(foot.lift_m, .102)
                self.assertEqual((foot.support_s, foot.contact_timeout_s, foot.settle_timeout_s, foot.clearance_m),
                                 (.5, 3., 5., .02))

    def test_all_explicit_scalar_fields_are_preserved_and_zero_distances_are_valid(self):
        foot = replace(self.foot, unload_s=4., lift_s=2., support_s=1., reach_s=5.,
                       contact_timeout_s=4., load_s=3., settle_timeout_s=6., lift_m=.102, clearance_m=.04)
        for candidate in (foot, replace(foot, lift_m=0., clearance_m=0.)):
            with self.subTest(lift=candidate.lift_m):
                self.assertIs(self.dispatch(replace(self.request, foot_request=candidate)), candidate)

    def test_explicit_request_type_and_exact_typed_intent_are_required(self):
        invalid = (object(), asdict(self.foot), replace(self.foot, limb=self.foot.limb.value),
                   replace(self.foot, limb=True), replace(self.foot, limb=Limb.RIGHT_FOOT),
                   replace(self.foot, limb=Limb.RIGHT_HAND), replace(self.foot, source="right_foot"),
                   replace(self.foot, source=1), replace(self.foot, target="left_foot"),
                   replace(self.foot, target=1), replace(self.foot, target=True))
        for foot in invalid:
            with self.subTest(foot=foot):
                self.reject(replace(self.request, foot_request=foot))
        for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            hand = transfers.TransferRequest(limb, self.contacts[limb],
                                             "left_reach_target" if limb == Limb.LEFT_HAND else "reach_target",
                                             self.contacts)
            matching_hand = ft.FootRequest(hand.limb, hand.source, hand.target)
            self.reject(replace(hand, foot_request=matching_hand))
        for changes in ({"limb": True}, {"limb": "LEFT_FOOT"}, {"source": 1}, {"target": 1}):
            self.reject(replace(self.request, foot_request=self.foot, **changes))

    def test_every_duration_rejects_non_native_nonfinite_bool_and_overflow_atomically(self):
        names = ("unload_s", "lift_s", "support_s", "reach_s", "contact_timeout_s", "load_s", "settle_timeout_s")
        for name in names:
            for value in (0., -.002, .0025, 1e-12, math.nan, math.inf, -math.inf, True, False,
                          "2", None, 1e308, 10 ** 1000):
                with self.subTest(field=name, value=value):
                    foot = replace(self.foot, **{name: value})
                    self.reject(replace(self.request, foot_request=foot))

    def test_lift_and_clearance_reject_invalid_scalars_atomically(self):
        for name in ("lift_m", "clearance_m"):
            for value in (-.001, math.nan, math.inf, -math.inf, True, False, "0.03", None, 10 ** 1000):
                with self.subTest(field=name, value=value):
                    self.reject(replace(self.request, foot_request=replace(self.foot, **{name: value})))

    def test_bad_second_object_is_atomic_before_first_move(self):
        self.reject(object())

    def test_requests_stay_immutable_and_serializer_walks_nested_dataclasses(self):
        contacts = dict(self.contacts)
        supports = {limb: hold for limb, hold in contacts.items() if limb != self.foot.limb}
        motion = WholeBodyMotion(Frame((0., 0., 0.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                                 {"waist_yaw": 0., "waist_pitch": 0., "waist_roll": 0.})
        foot = replace(self.foot, lift_m=.102)
        request = replace(self.request, source_contacts=contacts, support_contacts=supports,
                          foot_request=foot, whole_body=motion)
        contacts.clear()
        supports.clear()
        self.assertEqual(dict(request.source_contacts), self.contacts)
        self.assertEqual(len(request.support_contacts), 3)
        for mapping in (request.source_contacts, request.support_contacts):
            with self.assertRaises(TypeError):
                mapping[Limb.RIGHT_HAND] = "changed"
        with self.assertRaises(FrozenInstanceError):
            request.foot_request = None
        with self.assertRaises(FrozenInstanceError):
            foot.lift_m = .03
        encoded = json.loads(json.dumps(_json_value(request), allow_nan=False))
        self.assertEqual(encoded["foot_request"], _json_value(foot))
        self.assertEqual(encoded["foot_request"]["limb"], Limb.LEFT_FOOT.value)
        self.assertEqual(encoded["whole_body"], _json_value(motion))
        self.assertIsNone(_json_value(self.request)["foot_request"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
