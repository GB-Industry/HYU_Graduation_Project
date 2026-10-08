"""Scratch-only bounded morphology assessment, with native source setup once."""
from collections.abc import Mapping
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1 import transfer_feasibility as feasibility
from boulder_v1.contact_geometry import Frame, canonical_geometry
from boulder_v1.contact_ik import CollisionResidual, solve_hand_reference
from boulder_v1.grasp import GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import _integration_state, initialize_static_reference
from boulder_v1.transfers import TransferRequest
from boulder_v1.whole_body_demo import make_whole_body_fixture
from boulder_v1.whole_body_motion import reference_frames


def model_snapshot(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


def held_session(model, data, scene, profile, seed):
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    result = execute_static_hold(model, data, scene, profile, reference, manager,
                                 duration=2., settle=1., score_window=.5, keep_samples=False)
    if not result["success"]:
        raise AssertionError(result["reason"])
    return model, data, scene, profile, reference, manager


class TransferFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, data, scene, profile, seed = make_whole_body_fixture()
        cls.seed = seed
        cls.args = held_session(model, data, scene, profile, seed)
        cls.request = TransferRequest(Limb.RIGHT_HAND, scene.start_configuration[Limb.RIGHT_HAND],
                                      "reach_target", scene.start_configuration,
                                      support_contacts={l: h for l, h in scene.start_configuration.items()
                                                        if l != Limb.RIGHT_HAND})
        cls.nominal = None

    def assess(self, request=None, args=None, **kwargs):
        args = self.args if args is None else args
        model, live, scene, profile, reference, manager = args
        state = _integration_state(model, live)
        compiled = model_snapshot(model)
        channels = {n: getattr(live, n).copy() for n in (
            "qacc", "qfrc_actuator", "qfrc_constraint", "efc_force", "qfrc_applied", "xfrc_applied")}
        logs = copy.deepcopy((manager.capture_events, manager.releases, manager.last_capture_failure))
        attachments = manager._attachments
        native_admit = feasibility.initialize_static_reference
        native_attach = GraspManager.attach
        native_reset = mujoco.mj_resetData

        def admit(m, d, *a, **kw):
            self.assertIsNot(d, live)
            return native_admit(m, d, *a, **kw)

        def attach(gm, *a, **kw):
            self.assertIsNot(gm.data, live)
            return native_attach(gm, *a, **kw)

        def reset(m, d):
            self.assertIsNot(d, live)
            return native_reset(m, d)

        with patch("mujoco.mj_step", side_effect=AssertionError("assessment stepped physics")), \
                patch.object(feasibility, "initialize_static_reference", side_effect=admit), \
                patch.object(GraspManager, "attach", new=attach), \
                patch("mujoco.mj_resetData", side_effect=reset):
            result = feasibility.assess_hand_transfer(*args, self.request if request is None else request, **kwargs)
        np.testing.assert_array_equal(_integration_state(model, live), state)
        for name, value in channels.items():
            np.testing.assert_array_equal(getattr(live, name), value, err_msg=name)
        for name, value in compiled.items():
            np.testing.assert_array_equal(model_snapshot(model)[name], value, err_msg=name)
        self.assertIs(manager._attachments, attachments)
        self.assertEqual((manager.capture_events, manager.releases, manager.last_capture_failure), logs)
        self.assertEqual(result.diagnostics["actual_state_digest"], hashlib.sha256(state.tobytes()).hexdigest())
        return result

    def nominal_result(self):
        if self.__class__.nominal is None:
            self.__class__.nominal = self.assess()
        return self.__class__.nominal

    def test_nominal_native_source_geometry_support_and_endpoint_admission(self):
        result = self.nominal_result()
        model, data, scene, profile, reference, manager = self.args
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.classification, "GEOMETRICALLY_FEASIBLE")
        self.assertTrue(result.geometric_feasible and result.support_feasible)
        self.assertTrue(result.diagnostics["endpoint_admitted"])
        self.assertTrue(result.endpoint.converged)
        self.assertEqual(len(result.preparation_qpos), 81)
        self.assertEqual(len(result.diagnostics["reach_results"]), 92)
        self.assertEqual(len(result.diagnostics["candidate_support"]), 255)
        self.assertIsNot(result.source_reference, reference)
        np.testing.assert_array_equal(result.diagnostics["actual_qpos"], data.qpos)
        actual_root, actual_frames = reference_frames(model, data.qpos)
        self.assertEqual(result.diagnostics["actual_root"]["position"], actual_root.position)
        rise = .28 * max(0., np.dot(np.asarray(canonical_geometry(scene.region(self.request.target)).hand_frame.position)
                                   - actual_frames[self.request.limb].position, (0., 0., 1.)))
        self.assertAlmostEqual(result.motion.root_target.position[2] - actual_root.position[2], rise)
        self.assertAlmostEqual(result.motion.root_target.position[1], actual_root.position[1])
        self.assertLessEqual(rise, .06 * profile.leg_reach)
        self.assertNotEqual(result.motion.root_target.rotation, ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        for qpos in result.preparation_qpos:
            validate_reference_pose(model, qpos)
        for row in result.diagnostics["candidate_support"]:
            self.assertTrue(row["admitted"], row.get("reason"))
            self.assertGreaterEqual(row["rom_margin_min_rad"], 0.)
            self.assertLess(row["root_balance_residual"], 1e-8)
            self.assertGreater(row["motor_margin_min_Nm"], 0.)
            for limb, margin in row["margins"].items():
                if limb.is_foot:
                    self.assertGreater(margin["minimum_load_margin_N"], 0.)
                    self.assertGreaterEqual(margin["friction_margin_N"], 0.)
                else:
                    self.assertGreaterEqual(margin["margin_N"], 0.)
        plan = result.diagnostics["load_plan"]
        self.assertEqual((plan[0]["blend"], plan[-1]["blend"]), (0., 1.))
        self.assertEqual(plan[-1]["moving_hand_nominal_weight_share"], 0.)
        self.assertTrue(all(row["motor_margin_min_Nm"] > 0. for row in plan))
        new_intent = {**self.request.source_contacts, self.request.limb: self.request.target}
        self.assertEqual(dict(result.diagnostics["endpoint_reference"]["contact_intent"]), new_intent)
        for name in ("waist_pitch", "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee"):
            qa = int(model.joint(name).qposadr[0])
            self.assertGreater(np.ptp(np.asarray(result.preparation_qpos)[:, qa]), .01)
        minimum_motor = min(row["motor_margin_min_Nm"] for row in result.diagnostics["candidate_support"])
        print(f"bounded nominal: prep={len(result.preparation_qpos)} reach=92 root_rise_m={rise:.6f} "
              f"minimum_motor_margin_Nm={minimum_motor:.6f} endpoint_admitted=True", flush=True)

    def test_deep_immutable_contract(self):
        result = self.nominal_result()
        with self.assertRaises(FrozenInstanceError):
            result.feasible = False
        with self.assertRaises(TypeError):
            result.motion.waist_target["waist_pitch"] = 99.
        with self.assertRaises(TypeError):
            result.diagnostics["candidate_support"][0]["margins"][Limb.LEFT_FOOT]["normal_N"] = 0.
        self.assertIsInstance(result.diagnostics, Mapping)
        self.assertIsInstance(result.preparation_qpos[0], tuple)

    def test_structural_and_native_source_rejections_are_atomic(self):
        cases = ((replace(self.request, source_contacts={l.value: h for l, h in self.request.source_contacts.items()}), "INVALID_REQUEST"),
                 (replace(self.request, support_contacts=self.request.source_contacts), "INVALID_REQUEST"),
                 (replace(self.request, whole_body=self.nominal_result().motion), "INVALID_REQUEST"),
                 (replace(self.request, limb=Limb.LEFT_FOOT, source=self.request.source_contacts[Limb.LEFT_FOOT],
                          target="foot_target", support_contacts=None), "UNSUPPORTED_PRIMITIVE"),
                 (replace(self.request, source_contacts={**self.request.source_contacts, Limb.LEFT_HAND: self.request.target},
                          support_contacts=None),
                  "SOURCE_INFEASIBLE"))
        for request, classification in cases:
            with self.subTest(classification=classification):
                result = self.assess(request)
                self.assertFalse(result.feasible)
                self.assertEqual(result.classification, classification, result.reason)
        model, data, scene, profile, reference, manager = self.args
        for channel in ("qfrc_applied", "xfrc_applied"):
            old = getattr(data, channel).copy()
            try:
                getattr(data, channel).flat[0] = 1.
                result = self.assess()
                self.assertEqual(result.classification, "SOURCE_INFEASIBLE")
            finally:
                getattr(data, channel)[:] = old
        qa = int(model.joint("right_elbow").qposadr[0])
        old = float(data.qpos[qa])
        try:
            data.qpos[qa] = model.joint("right_elbow").range[1] + .001
            with patch.object(feasibility, "solve_contact_pose", side_effect=AssertionError("ROM was repaired")):
                result = self.assess()
            self.assertEqual(result.classification, "ROM_INFEASIBLE")
            self.assertEqual(result.diagnostics["invalid_actual_qpos"][qa], data.qpos[qa])
        finally:
            data.qpos[qa] = old
        cold = mujoco.MjData(model)
        cold_ref, cold_manager = initialize_static_reference(model, cold, scene, profile, self.seed, scene.start_configuration)
        cold.qpos[2] += .05
        result = self.assess(args=(model, cold, scene, profile, cold_ref, cold_manager))
        self.assertEqual(result.classification, "SOURCE_INFEASIBLE")
        self.assertIsNone(result.source_reference)

    def test_analytic_bound_and_local_failures_are_not_global_proofs(self):
        unreachable = self.assess(replace(self.request, target="unreachable_target"))
        self.assertEqual(unreachable.classification, "GEOMETRY_INFEASIBLE", unreachable.reason)
        self.assertFalse(unreachable.geometric_feasible)
        self.assertGreater(unreachable.diagnostics["reach_bound"]["gap_m"], 1e-6)
        self.assertIsNone(unreachable.endpoint)
        # Force an admission refusal, retaining all source IK evidence.
        model, data, scene, profile, reference, manager = self.args
        source_result = feasibility.solve_contact_pose(model, scene, profile, data.qpos.copy(), self.request.source_contacts)
        rejected = replace(source_result, admitted=False, reference=None, reason="Strict source admission refused")
        with patch.object(feasibility, "solve_contact_pose", return_value=rejected):
            result = self.assess()
        self.assertEqual(result.classification, "SOURCE_INFEASIBLE")
        self.assertIsNone(result.geometric_feasible)
        solution = self.nominal_result().endpoint
        stalled = replace(solution, converged=False, reason="Local bounded search exhausted; no global proof")
        limited_qpos = list(stalled.qpos)
        limited_qpos[int(model.joint("right_elbow").qposadr[0])] = model.joint("right_elbow").range[0]
        for evidence, classification in ((stalled, "LOCAL_SEARCH_UNRESOLVED"),
                                         (replace(stalled, qpos=limited_qpos), "ROM_LIMITED_SEARCH"),
                                         (replace(stalled, collisions=(CollisionResidual(("shoe", "obstacle"), -.001),)),
                                          "UNKNOWN_COLLISION_BLOCKED_SEARCH")):
            with patch.object(feasibility, "solve_whole_body_reference", return_value=evidence):
                result = self.assess()
            self.assertEqual(result.classification, classification)
            self.assertIsNone(result.geometric_feasible)
            self.assertEqual(result.diagnostics["failed_qpos"], evidence.qpos)
            self.assertEqual(result.preparation_qpos, ())

    def test_rejected_support_preserves_numeric_capacity_and_motor_margins(self):
        model, data, scene, profile, reference, manager = self.args
        weak = replace(profile, name="arbitrary", grip_capacity=.1)
        supports = dict(self.request.support_contacts)
        evidence = feasibility._support_evidence(model, scene, weak, self.nominal_result().endpoint.qpos, supports)
        self.assertFalse(evidence["admitted"])
        self.assertIn("capacity", evidence["reason"])
        held_hand = next(l for l in supports if l.is_hand)
        self.assertLess(evidence["margins"][held_hand]["margin_N"], 0.)
        self.assertTrue(np.isfinite(evidence["motor_margin_min_Nm"]))
        self.assertLess(evidence["root_balance_residual"], 1e-8)
        with patch.object(feasibility, "estimate_support_torques", side_effect=ValueError("Support reference exceeds remaining-hand capacity")):
            result = self.assess()
        self.assertEqual(result.classification, "SUPPORT_INFEASIBLE")
        self.assertTrue(result.geometric_feasible and result.diagnostics["endpoint_admitted"])
        self.assertFalse(result.support_feasible or result.feasible)
        self.assertTrue(result.diagnostics["candidate_support"])

    def test_blocked_declared_path_has_a_geometric_certificate_without_reach_ik(self):
        from boulder_v1.morphology_envelope import make_envelope_fixture, study_profiles
        model, data, scene, profile, seed, _ = make_envelope_fixture(study_profiles()["baseline"], "blocked_path")
        args = held_session(model, data, scene, profile, seed)
        request = TransferRequest(Limb.RIGHT_HAND, scene.start_configuration[Limb.RIGHT_HAND], "reach_target",
                                  scene.start_configuration, hand_reach_s=5.)
        result = self.assess(request, args=args)
        self.assertEqual(result.classification, "COLLISION_INFEASIBLE", result.reason)
        self.assertFalse(result.feasible or result.geometric_feasible)
        self.assertIsNotNone(result.source_reference)
        self.assertIsNone(result.endpoint)
        self.assertEqual(len(result.preparation_qpos), 81)
        self.assertEqual(result.diagnostics["reach_results"], ())
        self.assertNotIn("failed_qpos", result.diagnostics)
        self.assertTrue(all(row["admitted"] for row in result.diagnostics["candidate_support"]))
        certificate = result.diagnostics["path_collision_certificate"]
        self.assertEqual(certificate["static_geom"], "geom_path_obstacle")
        self.assertEqual(certificate["phase"], "reach")
        self.assertEqual(certificate["static_body_weldid"], 0)
        self.assertAlmostEqual(certificate["hand_interior_margin_m"], .006)
        self.assertGreater(certificate["obstacle_interior_margin_m"], .001)
        self.assertGreater(certificate["intersection_ball_radius_m"], .001)
        self.assertIn("specified sampled effector path only", certificate["scope"])
        with self.assertRaises(TypeError):
            certificate["collision_masks"]["hand"] = (0, 0)
        target = canonical_geometry(scene.region(request.target)).hand_frame
        samples = tuple(feasibility._declared_hand_path(scene, request, target, feasibility.ReferencePolicy()))
        sample = next(frame for phase, index, time, frame in samples
                      if phase == certificate["phase"] and index == certificate["index"])
        np.testing.assert_array_equal(sample.position, certificate["point_world_m"])
        snapshot = args[-1].contact_snapshot()
        self.assertTrue(all(hand.active and hand.valid for hand in snapshot.hands.values()))
        self.assertTrue(all(foot.normal_force > 5. and not foot.slipping for foot in snapshot.feet.values()))
        print(f"blocked_path: {result.classification} prep=81 reach_ik=0 "
              f"certificate_time_s={certificate['time_s']:.2f} "
              f"obstacle_margin_m={certificate['obstacle_interior_margin_m']:.6f} "
              "live_state_unchanged=True", flush=True)

    def test_path_certificate_honors_native_collision_filters_and_rigid_interiors(self):
        from boulder_v1.morphology_envelope import make_envelope_fixture, study_profiles
        _, _, scene, profile, seed, _ = make_envelope_fixture(study_profiles()["baseline"], "blocked_path")
        request = TransferRequest(Limb.RIGHT_HAND, scene.start_configuration[Limb.RIGHT_HAND], "reach_target",
                                  scene.start_configuration, hand_reach_s=5.)
        target = canonical_geometry(scene.region(request.target)).hand_frame
        samples = tuple(feasibility._declared_hand_path(scene, request, target, feasibility.ReferencePolicy()))
        for case, expected in (("box", True), ("sphere", True), ("rotated_hand_geom", True),
                               ("visual", False), ("incompatible_masks", False), ("excluded", False),
                               ("explicit_overrides_masks", True), ("explicit_overrides_exclusion", True),
                               ("moving_obstacle", False), ("site_outside_hand", False), ("contacts_disabled", False)):
            with self.subTest(case=case):
                tree = ET.fromstring(build_mjcf(scene, profile))
                obstacle = tree.find(".//geom[@name='geom_path_obstacle']")
                hand = tree.find(".//geom[@name='right_hand_geom']")
                if case == "sphere":
                    obstacle.attrib.update(type="sphere", size=".015")
                if case == "rotated_hand_geom":
                    hand.attrib["quat"] = f"{math.sqrt(.5)} {math.sqrt(.5)} 0 0"
                if case in ("visual", "explicit_overrides_masks"):
                    obstacle.attrib.update(contype="0", conaffinity="0")
                if case == "incompatible_masks":
                    obstacle.attrib.update(contype="8", conaffinity="8")
                if case in ("excluded", "explicit_overrides_exclusion"):
                    ET.SubElement(tree.find("contact"), "exclude", body1="right_hand", body2="contact_path_obstacle")
                if case in ("explicit_overrides_masks", "explicit_overrides_exclusion"):
                    ET.SubElement(tree.find("contact"), "pair", geom1="right_hand_geom", geom2="geom_path_obstacle")
                if case == "moving_obstacle":
                    tree.find(".//body[@name='contact_path_obstacle']").attrib["mocap"] = "true"
                if case == "site_outside_hand":
                    tree.find(".//site[@name='right_hand_site']").attrib["pos"] = "0 0 -.1"
                if case == "contacts_disabled":
                    ET.SubElement(tree.find("option"), "flag", contact="disable")
                model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                before = model_snapshot(model)
                with patch("mujoco.mj_step", side_effect=AssertionError("certificate stepped physics")):
                    certificate = feasibility._path_collision_certificate(model, seed, request, samples)
                self.assertEqual(certificate is not None, expected)
                for name, value in before.items():
                    np.testing.assert_array_equal(model_snapshot(model)[name], value, err_msg=name)
                if certificate is not None:
                    self.assertEqual(certificate["static_geom"], "geom_path_obstacle")
                    self.assertEqual(certificate["explicit_pair"], case.startswith("explicit_"))
                # Independently confirm native pair gating at a hypothetical FK
                # pose placing the rigid END site at a required interior point.
                if case not in ("moving_obstacle", "site_outside_hand"):
                    point = next(frame.position for phase, index, time, frame in samples
                                 if phase == "reach" and time == 2.5)
                    scratch = mujoco.MjData(model)
                    scratch.qpos[:] = seed
                    scratch.eq_active[:] = False
                    mujoco.mj_forward(model, scratch)
                    scratch.qpos[:3] += np.asarray(point) - scratch.site("right_hand_site").xpos
                    mujoco.mj_forward(model, scratch)
                    pair = {model.geom("right_hand_geom").id, model.geom("geom_path_obstacle").id}
                    self.assertEqual(any({int(c.geom1), int(c.geom2)} == pair and c.dist < 0.
                                         for c in scratch.contact), expected)
        # Intended moving source/target are the only HOLDs ignored. A support
        # HOLD is still an obstacle when the required moving-hand point enters it.
        tree = ET.fromstring(build_mjcf(scene, profile))
        for wall in scene.walls:
            tree.find(f".//geom[@name='{wall.id}_geom']").attrib.update(contype="0", conaffinity="0")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = seed
        scratch.eq_active[:] = False
        mujoco.mj_forward(model, scratch)
        for hold in (request.source, request.target, scene.start_configuration[Limb.LEFT_HAND]):
            point = tuple(scratch.geom(f"geom_{hold}").xpos)
            frame = Frame(point, target.rotation)
            certificate = feasibility._path_collision_certificate(model, seed, request, (("reach", 0, 0., frame),))
            if hold in (request.source, request.target):
                self.assertIsNone(certificate)
            else:
                self.assertEqual(certificate["static_geom"], f"geom_{hold}")
        obstacle = model.geom("geom_path_obstacle").id
        position = scratch.geom_xpos[obstacle]
        rotation = scratch.geom_xmat[obstacle].reshape(3, 3)
        for depth in (-.001, .0005):
            point = position + rotation @ (0., 0., model.geom_size[obstacle, 2] - depth)
            frame = Frame(tuple(point), target.rotation)
            self.assertIsNone(feasibility._path_collision_certificate(model, seed, request, (("reach", 0, 0., frame),)))

    def test_compiled_five_percent_morphologies_use_the_same_scene(self):
        model, data, scene, profile, reference, manager = self.args
        source_site = reference_frames(model, self.seed)[1][Limb.LEFT_FOOT].position
        candidates = []
        for scale in (.95, 1.05):
            changed = replace(profile, name="geometry_only", upper_arm_length=profile.upper_arm_length * scale,
                              forearm_length=profile.forearm_length * scale, thigh_length=profile.thigh_length * scale,
                              shin_length=profile.shin_length * scale)
            tree = ET.fromstring(build_mjcf(scene, changed))
            tree.find("option").attrib.update(timestep=".002", iterations="100", tolerance="1e-10")
            compiled = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
            seed = self.seed.copy()
            foot = reference_frames(compiled, seed)[1][Limb.LEFT_FOOT].position
            seed[:3] += np.asarray(source_site) - foot
            for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
                solution = solve_hand_reference(compiled, seed, limb,
                    canonical_geometry(scene.region(scene.start_configuration[limb])).hand_frame)
                self.assertTrue(solution.converged, solution.reason)
                seed = np.asarray(solution.qpos)
            args = held_session(compiled, mujoco.MjData(compiled), scene, changed, seed)
            result = self.assess(args=args)
            self.assertIsNotNone(result.source_reference, result.reason)
            self.assertIsNotNone(result.motion)
            self.assertEqual(args[2], scene)
            np.testing.assert_array_equal(compiled.jnt_range, model.jnt_range)
            np.testing.assert_array_equal(compiled.actuator_gear, model.actuator_gear)
            self.assertEqual(changed.grip_capacity, profile.grip_capacity)
            candidates.append(result)
            print(f"bounded morphology scale={scale}: {result.classification} "
                  f"prep={len(result.preparation_qpos)} endpoint_admitted={result.diagnostics['endpoint_admitted']}", flush=True)
        self.assertNotEqual(candidates[0].motion.root_target.rotation, candidates[1].motion.root_target.rotation)
        self.assertNotEqual(candidates[0].motion.waist_target["waist_pitch"], candidates[1].motion.waist_target["waist_pitch"])

    def test_renamed_profile_and_holds_produce_identical_candidate(self):
        model, data, scene, profile, reference, manager = self.args
        ids = {r.id: f"anchor_{i}" for i, r in enumerate(scene.contact_regions)}
        renamed = replace(scene, contact_regions=tuple(replace(r, id=ids[r.id]) for r in scene.contact_regions),
                          start_configuration={l: ids[h] for l, h in scene.start_configuration.items()},
                          goal_regions=tuple(ids[h] for h in scene.goal_regions))
        profile = replace(profile, name="no_semantic_profile_name")
        tree = ET.fromstring(build_mjcf(renamed, profile))
        tree.find("option").attrib.update(timestep=".002", iterations="100", tolerance="1e-10")
        compiled = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        args = held_session(compiled, mujoco.MjData(compiled), renamed, profile, self.seed)
        request = TransferRequest(self.request.limb, ids[self.request.source], ids[self.request.target], renamed.start_configuration)
        result = self.assess(request, args=args)
        baseline = self.nominal_result()
        self.assertEqual(result.classification, baseline.classification)
        self.assertEqual(result.motion, baseline.motion)
        np.testing.assert_array_equal(result.preparation_qpos, baseline.preparation_qpos)
        np.testing.assert_array_equal(result.endpoint.qpos, baseline.endpoint.qpos)

    def test_geometry_policy_responds_to_reach_target_and_world_rotation(self):
        model, data, scene, profile, reference, manager = self.args
        policy = feasibility.ReferencePolicy()
        motion, target, root, frames = feasibility._candidate_motion(model, data.qpos, scene, profile, self.request, policy)
        for scale in (.95, 1.05):
            changed = replace(profile, name="same_name_is_irrelevant", upper_arm_length=profile.upper_arm_length * scale,
                              forearm_length=profile.forearm_length * scale, thigh_length=profile.thigh_length * scale,
                              shin_length=profile.shin_length * scale)
            other, *_ = feasibility._candidate_motion(model, data.qpos, scene, changed, self.request, policy)
            self.assertNotEqual(other.waist_target["waist_pitch"], motion.waist_target["waist_pitch"])
            self.assertNotEqual(other.root_target.rotation, motion.root_target.rotation)
        for dx in (-.01, .01):
            perturbed = replace(scene, contact_regions=tuple(
                replace(r, position=tuple(np.asarray(r.position) + (dx, 0., 0.))) if r.id == self.request.target else r
                for r in scene.contact_regions))
            other, *_ = feasibility._candidate_motion(model, data.qpos, perturbed, profile, self.request, policy)
            self.assertAlmostEqual(other.root_target.position[0] - motion.root_target.position[0], .1 * dx)
        angle = .4
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
        rotated_scene = replace(scene, contact_regions=tuple(replace(r, position=tuple(rotation @ r.position),
                                                                    normal=tuple(rotation @ r.normal))
                                                           for r in scene.contact_regions))
        def rotate(frame):
            return Frame(tuple(rotation @ frame.position), tuple(map(tuple, rotation @ frame.rotation)))
        with patch.object(feasibility, "reference_frames", return_value=(rotate(root), {l: rotate(f) for l, f in frames.items()})):
            rotated, *_ = feasibility._candidate_motion(model, data.qpos, rotated_scene, profile, self.request, policy)
        np.testing.assert_allclose(rotated.root_target.position, rotation @ motion.root_target.position, atol=1e-12)
        np.testing.assert_allclose(rotated.root_target.rotation, rotation @ motion.root_target.rotation, atol=1e-12)
        self.assertEqual(rotated.waist_target, motion.waist_target)
        for kwargs in ({"rise_fraction": True}, {"yaw_fraction": math.nan}, {"rise_leg_bound": -1.},
                       {"yaw_max_rad": 1.}, {"sample_interval_s": .1}, {"prepare_s": 3.}):
            with self.assertRaises(ValueError):
                feasibility.ReferencePolicy(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
