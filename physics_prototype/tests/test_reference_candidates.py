"""Bounded selection with two cached, natively held morphology sources."""
from collections.abc import Mapping
import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from enum import Enum
import json
import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import reference_candidates as candidates
from boulder_v1 import transfer_feasibility as feasibility
from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.grasp import GraspManager
from boulder_v1.morphology_envelope import make_envelope_fixture, study_profiles
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import _integration_state, initialize_static_reference
from boulder_v1.transfers import TransferRequest
from boulder_v1.whole_body_motion import reference_frames


def evidence(value):
    """Walk immutable dataclasses/proxies without asdict's deepcopy."""
    if is_dataclass(value):
        return {field.name: evidence(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(evidence(key)): evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [evidence(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def model_snapshot(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


class ReferenceCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = {}
        cls.selections = {}
        for name, dt in (("baseline", .002), ("shorter", .001)):
            model, data, scene, profile, seed, _ = make_envelope_fixture(study_profiles()[name], timestep=dt)
            reference, manager = initialize_static_reference(model, data, scene, profile, seed,
                                                             scene.start_configuration)
            hold = execute_static_hold(model, data, scene, profile, reference, manager,
                                       duration=2., settle=1., score_window=.5, keep_samples=False)
            if not hold["success"]:
                raise AssertionError(hold["reason"])
            cls.sessions[name] = (model, data, scene, profile, reference, manager)

    def request(self, args):
        contacts = args[4].contact_intent
        return TransferRequest(Limb.RIGHT_HAND, contacts[Limb.RIGHT_HAND], "reach_target", contacts,
                               support_contacts={limb: hold for limb, hold in contacts.items()
                                                 if limb != Limb.RIGHT_HAND}, hand_reach_s=5.)

    def choose(self, name, **kwargs):
        args = self.sessions[name]
        model, live, scene, profile, reference, manager = args
        before = _integration_state(model, live)
        channels = {key: getattr(live, key).copy() for key in (
            "qacc", "qacc_warmstart", "ctrl", "act", "qfrc_actuator", "qfrc_constraint",
            "efc_force", "qfrc_applied", "xfrc_applied")}
        compiled = model_snapshot(model)
        logs = copy.deepcopy((manager.capture_events, manager.releases, manager.last_capture_failure))
        attachments = dict(manager._attachments)
        registry = manager._attachments
        native_assess = candidates.assess_hand_transfer
        native_admit = feasibility.initialize_static_reference
        native_attach = GraspManager.attach
        native_reset = mujoco.mj_resetData
        calls = []

        def assess(*received, **options):
            for actual, expected in zip(received, (*args, request)):
                self.assertIs(actual, expected)
            np.testing.assert_array_equal(_integration_state(model, live), before)
            calls.append(options["policy"])
            result = native_assess(*received, **options)
            np.testing.assert_array_equal(result.diagnostics["actual_integration_state"], before)
            return result

        def admit(m, d, *a, **kw):
            self.assertIsNot(d, live)
            return native_admit(m, d, *a, **kw)

        def attach(gm, *a, **kw):
            self.assertIsNot(gm.data, live)
            return native_attach(gm, *a, **kw)

        def reset(m, d):
            self.assertIsNot(d, live)
            return native_reset(m, d)

        request = self.request(args)
        with patch.object(candidates, "assess_hand_transfer", side_effect=assess), \
                patch("mujoco.mj_step", side_effect=AssertionError("selection stepped physics")), \
                patch.object(feasibility, "initialize_static_reference", side_effect=admit), \
                patch.object(GraspManager, "attach", new=attach), \
                patch("mujoco.mj_resetData", side_effect=reset):
            result = candidates.choose_hand_reference(*args, request, **kwargs)
        self.assertEqual(calls, [policy for _, policy in candidates.REFERENCE_CANDIDATES])
        np.testing.assert_array_equal(_integration_state(model, live), before)
        for key, original in channels.items():
            np.testing.assert_array_equal(getattr(live, key), original, err_msg=key)
        after = model_snapshot(model)
        for key, original in compiled.items():
            np.testing.assert_array_equal(after[key], original, err_msg=key)
        self.assertIs(manager._attachments, registry)
        self.assertEqual(manager._attachments, attachments)
        self.assertEqual((manager.capture_events, manager.releases, manager.last_capture_failure), logs)
        return result

    def selection(self, name):
        if name not in self.selections:
            self.selections[name] = self.choose(name, prepare_s=6.)
        return self.selections[name]

    def test_fixed_three_policies_preserve_stock_timing(self):
        self.assertEqual([(label, policy.rise_fraction, policy.yaw_fraction, policy.waist_pitch_fraction)
                          for label, policy in candidates.REFERENCE_CANDIDATES],
                         [("default", .28, .3, .3), ("neutral_yaw", .28, 0., .3),
                          ("conservative", .18, 0., 0.)])
        for _, policy in candidates.REFERENCE_CANDIDATES:
            self.assertEqual((policy.prepare_s, policy.sample_interval_s), (4., .05))
            self.assertEqual(policy.rise_leg_bound, .06)
        for options in ({"prepare_s": 6.}, {"sample_interval_s": .1}):
            with self.assertRaises(ValueError):
                feasibility.ReferencePolicy(**options)

    def test_baseline_first_admitted_all_three_assessed_rich_immutable_evidence(self):
        result = self.selection("baseline")
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual((result.selected_index, result.selected_name), (0, "default"))
        self.assertEqual(result.classification, "GEOMETRICALLY_FEASIBLE")
        self.assertEqual(len(result.assessments), 3)
        admitted = result.assessments[0][1]
        self.assertIsNot(admitted.source_reference, self.sessions["baseline"][4])
        self.assertEqual(result.motion, replace(admitted.motion, prepare_s=6.))
        self.assertEqual(admitted.motion.prepare_s, 4.)
        self.assertEqual(len(admitted.preparation_qpos), 81)
        self.assertTrue(admitted.endpoint.converged)
        self.assertTrue(admitted.diagnostics["source_result"]["admitted"])
        self.assertTrue(admitted.diagnostics["candidate_support"])
        self.assertTrue(admitted.diagnostics["load_plan"])
        self.assertIn("compiled_rom_rad", admitted.diagnostics)
        self.assertIn("path_collision_certificate", admitted.diagnostics)
        with self.assertRaises(FrozenInstanceError):
            result.selected_index = 2
        with self.assertRaises(TypeError):
            result.motion.waist_target["waist_pitch"] = 0.
        with self.assertRaises(TypeError):
            admitted.diagnostics["candidate_support"][0]["margins"][Limb.LEFT_FOOT]["normal_N"] = 0.
        saved = json.loads(json.dumps(evidence(result), allow_nan=False))
        self.assertEqual(len(saved["assessments"]), 3)
        self.assertEqual(saved["motion"]["prepare_s"], 6.)
        self.assertEqual(saved["assessments"][0][1]["motion"]["prepare_s"], 4.)
        self.assertEqual(saved["assessments"][0][1]["endpoint"]["qpos"], list(admitted.endpoint.qpos))

    def test_shorter_own_compiled_geometry_all_three_without_native_execution(self):
        result = self.selection("shorter")
        model, data, scene, profile, reference, manager = self.sessions["shorter"]
        baseline = self.sessions["baseline"]
        self.assertEqual(scene, baseline[2])
        source = canonical_geometry(scene.region(reference.contact_intent[Limb.RIGHT_HAND])).hand_frame
        target = canonical_geometry(scene.region("reach_target")).hand_frame
        self.assertAlmostEqual(target.position[2] - source.position[2], .145)
        self.assertNotEqual(profile.arm_reach, baseline[3].arm_reach)
        np.testing.assert_array_equal(model.jnt_range, baseline[0].jnt_range)
        self.assertEqual(len(result.assessments), 3)
        default = result.assessments[0][1]
        self.assertEqual(default.classification, "ROM_LIMITED_SEARCH", default.reason)
        self.assertTrue(any("shoulder_yaw" in joint for joint in default.diagnostics["active_rom_limits"]))
        for label, assessment in result.assessments:
            self.assertIsNotNone(assessment.motion, assessment.reason)
            self.assertEqual(assessment.motion.prepare_s, 4.)
            self.assertEqual(assessment.diagnostics["actual_qpos"], tuple(data.qpos))
            if label != "default":
                root, _ = reference_frames(model, data.qpos)
                self.assertEqual(assessment.motion.root_target.rotation, root.rotation)
            print(f"shorter candidate {label}: {assessment.classification} "
                  f"geometric={assessment.geometric_feasible} support={assessment.support_feasible} "
                  f"prep={len(assessment.preparation_qpos)} "
                  f"reach_bound_gap_m={assessment.diagnostics['reach_bound']['gap_m']:.6f} "
                  f"active_rom_limits={assessment.diagnostics.get('active_rom_limits', ())} "
                  "native_execution=False", flush=True)
        first = next((i for i, (_, assessment) in enumerate(result.assessments) if assessment.feasible), None)
        self.assertEqual(result.selected_index, first)
        if first is not None:
            self.assertEqual(result.classification, "GEOMETRICALLY_FEASIBLE")
        else:
            self.assertEqual(result.classification, "BOUNDED_CANDIDATE_SET_EXHAUSTED")
            self.assertIn("not a global", result.reason)

    def test_invalid_execution_timing_atomic_before_any_assessment(self):
        args = self.sessions["baseline"]
        before = _integration_state(*args[:2])
        for duration in (False, True, math.nan, math.inf, -math.inf, 0., -1., .003, "4", None):
            with self.subTest(duration=duration), \
                    patch.object(candidates, "assess_hand_transfer", side_effect=AssertionError("assessment called")):
                with self.assertRaises(ValueError):
                    candidates.choose_hand_reference(*args, self.request(args), prepare_s=duration)
                np.testing.assert_array_equal(_integration_state(*args[:2]), before)

    def test_execution_timing_override_does_not_change_assessment_or_order(self):
        args = self.sessions["baseline"]
        assessments = self.selection("baseline").assessments
        for duration in (.002, 4., 6., np.int64(8)):
            with patch.object(candidates, "assess_hand_transfer",
                              side_effect=[assessment for _, assessment in assessments]) as assess:
                result = candidates.choose_hand_reference(*args, self.request(args), prepare_s=duration)
            self.assertEqual(result.selected_index, 0)
            self.assertEqual(result.assessments, assessments)
            for (_, original), (_, retained) in zip(assessments, result.assessments):
                self.assertIs(retained, original)
                self.assertEqual(retained.motion.prepare_s, 4.)
            self.assertEqual(result.motion.prepare_s, float(duration))
            self.assertIsInstance(result.motion.prepare_s, float)
            self.assertEqual(result.motion.root_target, assessments[0][1].motion.root_target)
            self.assertEqual([call.kwargs["policy"] for call in assess.call_args_list],
                             [policy for _, policy in candidates.REFERENCE_CANDIDATES])

    def test_first_feasible_order_exhaustion_and_callback_errors(self):
        args = self.sessions["baseline"]
        template = self.selection("baseline").assessments[0][1]
        rejected = [replace(template, feasible=False, classification=classification, reason="local failure")
                    for classification in ("ROM_LIMITED_SEARCH", "COLLISION_INFEASIBLE", "SUPPORT_INFEASIBLE")]
        with patch.object(candidates, "assess_hand_transfer", return_value=template) as assess:
            result = candidates.choose_hand_reference(*args, self.request(args))
        self.assertEqual(assess.call_count, 3)
        self.assertEqual(result.selected_index, 0)
        for index in (0, 1, 2, None):
            assessments = list(rejected)
            if index is not None:
                assessments[index] = template
            with patch.object(candidates, "assess_hand_transfer", side_effect=assessments) as assess:
                result = candidates.choose_hand_reference(*args, self.request(args))
            self.assertEqual(assess.call_count, 3)
            self.assertEqual(result.selected_index, index)
            self.assertEqual(tuple(value for _, value in result.assessments), tuple(assessments))
            if index is None:
                self.assertIsNone(result.motion)
                self.assertIsNone(result.selected_name)
                self.assertFalse(result.feasible)
                self.assertEqual(result.classification, "BOUNDED_CANDIDATE_SET_EXHAUSTED")
                for assessment in rejected:
                    self.assertIn(assessment.classification, result.reason)
                self.assertIn("not a global", result.reason)
        for error in (TypeError("stock type error"), RuntimeError("callback nondeterminism")):
            with patch.object(candidates, "assess_hand_transfer", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    candidates.choose_hand_reference(*args, self.request(args))
                self.assertIs(caught.exception, error)

    def test_stock_source_request_and_data_errors_not_repaired_or_reclassified(self):
        args = self.sessions["baseline"]
        model, data, scene, profile, reference, manager = args
        request = self.request(args)
        before = _integration_state(model, data)
        cases = (
            (args, replace(request, source_contacts={limb.value: hold for limb, hold in request.source_contacts.items()}),
             "INVALID_REQUEST"),
            (args, replace(request, whole_body=self.selection("baseline").motion), "INVALID_REQUEST"),
            ((model, data, scene, profile, None, manager), request, "SOURCE_INFEASIBLE"),
            ((model, mujoco.MjData(model), scene, profile, reference, manager), request, "SOURCE_INFEASIBLE"),
            (args, replace(request, limb=Limb.LEFT_FOOT, source=request.source_contacts[Limb.LEFT_FOOT],
                           target="foot_target", support_contacts=None), "UNSUPPORTED_PRIMITIVE"),
        )
        with patch.object(feasibility, "solve_contact_pose", side_effect=AssertionError("invalid source repaired")), \
                patch("mujoco.mj_step", side_effect=AssertionError("invalid source stepped")):
            for received, supplied, classification in cases:
                with self.subTest(classification=classification):
                    result = candidates.choose_hand_reference(*received, supplied)
                    self.assertEqual(len(result.assessments), 3)
                    self.assertTrue(all(assessment.classification == classification
                                        for _, assessment in result.assessments))
                    self.assertEqual(result.classification, "BOUNDED_CANDIDATE_SET_EXHAUSTED")
                    self.assertIsNone(result.motion)
                    np.testing.assert_array_equal(_integration_state(model, data), before)
        with self.assertRaises(TypeError):
            candidates.choose_hand_reference(model, object(), scene, profile, reference, manager, request)
        np.testing.assert_array_equal(_integration_state(model, data), before)

    def test_renamed_profile_holds_and_measured_orientation_do_not_tune_policies(self):
        args = self.sessions["baseline"]
        model, data, scene, profile, reference, manager = args
        ids = {region.id: f"anchor_{i}" for i, region in enumerate(scene.contact_regions)}
        renamed = replace(scene, contact_regions=tuple(replace(region, id=ids[region.id])
                                                       for region in scene.contact_regions),
                          start_configuration={limb: ids[hold] for limb, hold in scene.start_configuration.items()},
                          goal_regions=tuple(ids[hold] for hold in scene.goal_regions))
        request = self.request(args)
        renamed_request = replace(request, source=ids[request.source], target=ids[request.target],
                                  source_contacts=renamed.start_configuration, support_contacts=None)
        renamed_profile = replace(profile, name="arbitrary_no_tuning")
        root, frames = reference_frames(model, data.qpos)
        for _, policy in candidates.REFERENCE_CANDIDATES:
            motion, *_ = feasibility._candidate_motion(model, data.qpos, scene, profile, request, policy)
            other, *_ = feasibility._candidate_motion(model, data.qpos, renamed, renamed_profile,
                                                      renamed_request, policy)
            self.assertEqual(motion, other)
            repeated, *_ = feasibility._candidate_motion(model, data.qpos, scene, profile, request, policy)
            self.assertEqual(motion, repeated)
            if policy.yaw_fraction == 0.:
                self.assertEqual(motion.root_target.rotation, root.rotation)
        angle = .4
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
        rotated_root = replace(root, rotation=tuple(map(tuple, rotation @ root.rotation)))
        for _, policy in candidates.REFERENCE_CANDIDATES:
            with patch.object(feasibility, "reference_frames", return_value=(rotated_root, frames)):
                motion, *_ = feasibility._candidate_motion(model, data.qpos, scene, profile, request, policy)
            ordinary, *_ = feasibility._candidate_motion(model, data.qpos, scene, profile, request, policy)
            np.testing.assert_allclose(motion.root_target.rotation, rotation @ ordinary.root_target.rotation, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
