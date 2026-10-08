"""Stage5.1 whole-body geometric references on the unchanged compiled humanoid."""
from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.contact_geometry import FOOT_SITE_OFFSET, Frame, canonical_geometry
from boulder_v1.contact_ik import solve_hand_reference
from boulder_v1.grasp import GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Affordance, ClimberProfile, Limb, SourceType
from boulder_v1.single_hand import make_single_hand_fixture
from boulder_v1.static_state import initialize_static_reference
from boulder_v1.whole_body_reference import WholeBodyReferenceResult, solve_whole_body_reference


WAIST = {"waist_yaw": 0., "waist_pitch": 0., "waist_roll": 0.}
FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
HANDS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)


def probe(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.eq_active[:] = False
    mujoco.mj_forward(model, data)
    return data


def limb_frame(data, limb):
    name = limb.value.lower()
    rotation = (data.site(f"{name}_site").xmat if limb.is_hand else data.geom(f"{name}_geom").xmat).reshape(3, 3)
    return Frame(tuple(data.site(f"{name}_site").xpos), tuple(map(tuple, rotation)))


def root_frame(model, data):
    joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    body = int(model.jnt_bodyid[joint])
    return Frame(tuple(data.xpos[body]), tuple(map(tuple, data.xmat[body].reshape(3, 3))))


def yaw_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def model_snapshot(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def crouch_fixture():
    """Keep Stage5's four source holds fixed; derive root offset from planted FK."""
    _, _, stage5_scene, profile, stage5_seed = make_single_hand_fixture()
    scene = replace(stage5_scene, contact_regions=tuple(
        r for r in stage5_scene.contact_regions if r.id in stage5_scene.start_configuration.values()))
    model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
    seed = stage5_seed.copy()
    for side in ("left", "right"):
        for name, angle in (("hip_pitch", .5), ("knee", 1.), ("ankle_pitch", .5)):
            seed[int(model.joint(f"{side}_{name}").qposadr[0])] = angle
    bent = probe(model, seed)
    root_qa = int(model.joint("root").qposadr[0])
    foot = canonical_geometry(scene.region(scene.start_configuration[Limb.LEFT_FOOT])).foot_frame
    planted = np.array(foot.position) + np.array(foot.rotation) @ [0., .01, 0.]
    seed[root_qa:root_qa + 3] += planted - bent.site("left_foot_site").xpos
    # Independently prepare the arms; no full-body solver or physical steps.
    for limb in HANDS:
        target = canonical_geometry(scene.region(scene.start_configuration[limb])).hand_frame
        arm = solve_hand_reference(model, seed, limb, target)
        if not arm.converged:
            raise AssertionError(arm.reason)
        seed = np.array(arm.qpos)
    initialize_static_reference(model, mujoco.MjData(model), scene, profile, seed, scene.start_configuration)
    return model, scene, profile, seed


class WholeBodyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.scene, cls.profile, cls.seed = crouch_fixture()
        cls.intent = dict(cls.scene.start_configuration)
        cls.source = probe(cls.model, cls.seed)
        cls.start_root = root_frame(cls.model, cls.source)
        cls.goal = replace(cls.start_root,
                           position=tuple(np.array(cls.start_root.position) + [0., 0., .04]),
                           rotation=tuple(map(tuple, yaw_rotation(math.radians(4.)))))

    def solve(self, **kwargs):
        options = dict(root_target=self.goal, waist_target={**WAIST, "waist_pitch": .07})
        options.update(kwargs)
        return solve_whole_body_reference(self.model, self.scene, self.profile, self.seed, self.intent, **options)

    def assert_evidence(self, model, result, *, converged=True, tolerance=1e-6):
        self.assertIsInstance(result, WholeBodyReferenceResult)
        self.assertEqual(result.converged, converged, result.reason)
        validate_reference_pose(model, result.qpos)
        data = probe(model, result.qpos)
        actual_root = root_frame(model, data)
        np.testing.assert_allclose(actual_root.position, result.root_target.position, rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual_root.rotation, result.root_target.rotation, rtol=0, atol=1e-15)
        self.assertLess(result.root_position_error, 1e-15)
        self.assertLess(result.root_orientation_error, 1e-15)
        self.assertFalse(hasattr(result, "admitted"))
        self.assertFalse(hasattr(result, "reference"))
        self.assertIn("no contact admission", result.reason)
        self.assertEqual(len(result.target_pose), model.nu)
        for name, angle in result.target_pose.items():
            self.assertEqual(angle, result.qpos[int(model.joint(name).qposadr[0])])
        for r in result.residuals:
            frame = limb_frame(data, r.limb)
            current, goal = np.array(frame.rotation), np.array(r.target.rotation)
            if r.limb.is_hand:
                alignment = float(np.clip(current[:, 2] @ goal[:, 2], -1., 1.))
                angle = math.atan2(float(np.linalg.norm(np.cross(current[:, 2], goal[:, 2]))), alignment)
            else:
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, (goal @ current.T).ravel())
                angle = 2. * math.atan2(float(np.linalg.norm(quat[1:])), abs(float(quat[0])))
            self.assertAlmostEqual(r.position_error, np.linalg.norm(np.array(frame.position) - r.target.position), delta=1e-12)
            self.assertAlmostEqual(r.orientation_error, angle, delta=1e-12)
            if converged:
                self.assertLessEqual(r.position_error, tolerance)
                self.assertLessEqual(r.orientation_error, tolerance)
        for name, target in result.waist_target.items():
            qa = int(model.joint(name).qposadr[0])
            self.assertAlmostEqual(result.waist_errors[name], abs(result.qpos[qa] - target), delta=1e-15)
            if converged:
                self.assertLessEqual(result.waist_errors[name], tolerance)
        if converged:
            self.assertFalse(result.collisions)
        return data

    def test_four_contact_root_rise_yaw_and_waist_generate_hip_knee_participation(self):
        result = self.solve()
        data = self.assert_evidence(self.model, result)
        self.assertGreater(result.iterations, 0)
        self.assertAlmostEqual(result.initial_root_position_error, .04, places=13)
        self.assertAlmostEqual(result.initial_root_orientation_error, math.radians(4.), places=13)
        self.assertAlmostEqual(result.initial_waist_errors["waist_pitch"], .07, places=13)
        self.assertLess(result.initial_root.position[2], 1.2)
        self.assertGreater(result.initial_root.position[2], 1.15)
        for side in ("left", "right"):
            for name in ("hip_pitch", "knee"):
                qa = int(self.model.joint(f"{side}_{name}").qposadr[0])
                self.assertGreater(abs(result.qpos[qa] - self.seed[qa]), .03)
        for limb in Limb:
            np.testing.assert_allclose(limb_frame(data, limb).position, limb_frame(self.source, limb).position,
                                       rtol=0, atol=1e-6)
            np.testing.assert_array_equal(result.source_frames[limb], limb_frame(self.source, limb))
        for r in result.residuals:
            self.assertLess(next(i.position_error for i in result.initial_residuals if i.limb == r.limb), 1e-7)
            if r.limb.is_foot:
                self.assertAlmostEqual(r.canonical_position_error, .01, delta=1e-6)
                self.assertLess(abs(r.foot_site_normal_residual), 1e-6)
                self.assertLess(abs(r.foot_minimum_sole_distance), 1e-6)
                self.assertIsNotNone(r.foot_residual, r.foot_reason)
        # The caller, not this generator, exercises the unchanged physical gate.
        reference, _ = initialize_static_reference(self.model, mujoco.MjData(self.model), self.scene,
                                                   self.profile, result.qpos, self.intent)
        self.assertEqual(int(self.model.joint("root").type[0]), mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(all("foot" not in self.model.equality(i).name for i in range(self.model.neq)))
        print("WHOLE_BODY_REFERENCE " + json.dumps({
            "source_root_z_m": result.initial_root.position[2], "root_rise_m": result.initial_root_position_error,
            "root_yaw_deg": math.degrees(result.initial_root_orientation_error), "iterations": result.iterations,
            "waist_max_rad": max(result.waist_errors.values()),
            "initial_position_max_m": max(r.position_error for r in result.initial_residuals),
            "initial_hand_errors_m": {r.limb.value: r.position_error for r in result.initial_residuals if r.limb.is_hand},
            "position_max_m": max(r.position_error for r in result.residuals),
            "orientation_max_rad": max(r.orientation_error for r in result.residuals),
            "foot_site_normal_max_m": max(abs(r.foot_site_normal_residual) for r in result.residuals if r.limb.is_foot),
            "foot_sole_max_m": max(abs(r.foot_minimum_sole_distance) for r in result.residuals if r.limb.is_foot),
            "hip_knee_changes_rad": {f"{side}_{name}": result.target_pose[f"{side}_{name}"] -
                                     self.seed[int(self.model.joint(f"{side}_{name}").qposadr[0])]
                                     for side in ("left", "right") for name in ("hip_pitch", "knee")},
            "caller_admitted": bool(reference), "hinges": len(result.target_pose)}, allow_nan=False), flush=True)

    def test_waist_only_change_does_not_preserve_an_already_admitted_seed(self):
        result = self.solve(root_target=self.start_root, waist_target={**WAIST, "waist_pitch": math.radians(6.)})
        self.assert_evidence(self.model, result)
        self.assertGreater(result.iterations, 0)
        self.assertNotEqual(result.qpos, tuple(self.seed))
        self.assertAlmostEqual(result.target_pose["waist_pitch"], math.radians(6.), delta=1e-6)
        for limb in Limb:
            np.testing.assert_allclose(limb_frame(probe(self.model, result.qpos), limb).position,
                                       limb_frame(self.source, limb).position, rtol=0, atol=1e-6)

    def test_legal_unadmitted_crouch_can_solve_original_world_hand_anchors(self):
        seed = self.seed.copy()
        for side in ("left", "right"):
            for name, angle in (("shoulder_pitch", .3), ("shoulder_roll", 0.), ("shoulder_yaw", 0.),
                                ("elbow", math.pi / 2), ("wrist", -.3)):
                seed[int(self.model.joint(f"{side}_{name}").qposadr[0])] = angle
        source = probe(self.model, seed)
        result = solve_whole_body_reference(self.model, self.scene, self.profile, seed, self.intent,
                                           root_target=self.start_root, waist_target=WAIST)
        self.assert_evidence(self.model, result)
        for r in result.initial_residuals:
            self.assertEqual(result.source_frames[r.limb], limb_frame(source, r.limb))
            if r.limb.is_hand:
                self.assertGreater(r.position_error, .08)
        self.assertGreater(result.iterations, 0)

    def test_source_capture_precedes_override_and_explicit_supports_do_not_drift(self):
        supports = {limb: limb_frame(self.source, limb) for limb in Limb}
        first = self.solve(support_frames=supports)
        self.assert_evidence(self.model, first)
        later = replace(self.goal, position=tuple(np.array(self.goal.position) + [-.01, .005, .01]))
        second = solve_whole_body_reference(self.model, self.scene, self.profile, first.qpos, self.intent,
                                           root_target=later, waist_target={**WAIST, "waist_pitch": .1},
                                           support_frames=supports)
        self.assert_evidence(self.model, second)
        for r in second.residuals:
            self.assertEqual(r.target, supports[r.limb])
        self.assertEqual(first.initial_root, self.start_root)
        self.assertNotEqual(second.initial_root, self.start_root)
        for limb in FEET:
            self.assertGreater(np.linalg.norm(np.array(supports[limb].position) -
                                             (np.array(limb_frame(self.source, limb).position) + [0., 0., .04])), .039)

    def test_transient_either_hand_145mm_reach_and_declared_source_goal_touches(self):
        for limb in HANDS:
            with self.subTest(limb=limb):
                original = self.scene.region(self.intent[limb])
                goal_hold = replace(original, id="renamed_reach_goal", position=tuple(
                    np.array(original.position) + [0., 0., .25 * self.profile.arm_reach]))
                scene = replace(self.scene, contact_regions=(*self.scene.contact_regions, goal_hold))
                model = mujoco.MjModel.from_xml_string(build_mjcf(scene, self.profile))
                target = canonical_geometry(goal_hold).hand_frame
                supports = {l: limb_frame(self.source, l) for l in Limb}
                for fraction in (0., .5, 1.):
                    start = canonical_geometry(original).hand_frame
                    moving = replace(target, position=tuple((1. - fraction) * np.array(start.position)
                                                           + fraction * np.array(target.position)))
                    result = solve_whole_body_reference(
                        model, scene, self.profile, self.seed, {**self.intent, limb: goal_hold.id},
                        root_target=self.goal, waist_target={**WAIST, "waist_pitch": .07},
                        hand_targets={limb: moving}, support_frames=supports,
                        allowed_hand_holds={limb: (original.id, goal_hold.id)})
                    self.assert_evidence(model, result)
                    for r in result.residuals:
                        self.assertEqual(r.target, moving if r.limb == limb else supports[r.limb])

    def test_hand_twist_free_but_shoe_twist_and_antiparallel_normals_are_measured(self):
        supports = {limb: limb_frame(self.source, limb) for limb in Limb}
        twist = yaw_rotation(1.1)
        twisted = {limb: replace(frame, rotation=tuple(map(tuple, np.array(frame.rotation) @ twist)))
                   if limb.is_hand else frame for limb, frame in supports.items()}
        matched = self.solve(root_target=self.start_root, waist_target=WAIST, support_frames=twisted, max_iterations=0)
        self.assert_evidence(self.model, matched)
        np.testing.assert_array_equal(matched.qpos, self.seed)
        for limb in (Limb.RIGHT_HAND, Limb.RIGHT_FOOT):
            for rotation, minimum in ((yaw_rotation(.06), .05), (np.diag([1., -1., -1.]), 3.)):
                if limb.is_hand and minimum < 1.:
                    continue
                frame = supports[limb]
                bad = replace(frame, rotation=tuple(map(tuple, np.array(frame.rotation) @ rotation)))
                result = self.solve(root_target=self.start_root, waist_target=WAIST,
                                    support_frames={**supports, limb: bad}, max_iterations=0)
                self.assert_evidence(self.model, result, converged=False)
                r = next(r for r in result.residuals if r.limb == limb)
                self.assertLess(r.position_error, 1e-12)
                self.assertGreater(r.orientation_error, minimum)

    def test_virtual_root_exact_at_every_trial_and_live_model_manager_unchanged(self):
        model = self.model
        live = mujoco.MjData(model)
        _, manager = initialize_static_reference(model, live, self.scene, self.profile, self.seed, self.intent)
        live.qvel[:] = np.linspace(-.01, .01, model.nv)
        live.ctrl[:] = .123
        live.time = .73
        live.xfrc_applied[model.body("climber_root").id] = (1., 2., 3., 4., 5., 6.)
        live.qfrc_applied[:] = .2
        before_model, before_live, before_seed = model_snapshot(model), integration_state(model, live), self.seed.copy()
        events, releases, attachments = manager.capture_events.copy(), manager.releases.copy(), manager.active_attachments()
        native_integrate, native_forward = mujoco.mj_integratePos, mujoco.mj_forward
        steps, forwarded = [], []
        qa, va = int(model.joint("root").qposadr[0]), int(model.joint("root").dofadr[0])
        virtual = np.array((*self.goal.position, *self.goal.quaternion))

        def integrate(m, qpos, tangent, dt):
            self.assertFalse(np.shares_memory(qpos, live.qpos))
            self.assertFalse(np.shares_memory(qpos, self.seed))
            np.testing.assert_array_equal(tangent[va:va + 6], 0.)
            self.assertLessEqual(np.linalg.norm(tangent), .1 + 1e-15)
            steps.append(tangent.copy())
            native_integrate(m, qpos, tangent, dt)

        def forward(m, data):
            self.assertIsNot(data, live)
            expected = self.seed[qa:qa + 7] if not forwarded else virtual
            np.testing.assert_array_equal(data.qpos[qa:qa + 7], expected)
            validate_reference_pose(m, data.qpos)
            self.assertFalse(np.any(data.eq_active))
            self.assertFalse(np.any(data.xfrc_applied) or np.any(data.qfrc_applied) or np.any(data.ctrl))
            forwarded.append(data.qpos.copy())
            native_forward(m, data)

        with patch("mujoco.mj_integratePos", side_effect=integrate), \
                patch("mujoco.mj_forward", side_effect=forward), \
                patch("mujoco.mj_step", side_effect=AssertionError("generator stepped physics")), \
                patch.object(GraspManager, "attach", side_effect=AssertionError("generator attempted capture")), \
                patch("boulder_v1.static_state.initialize_static_reference",
                      side_effect=AssertionError("generator attempted admission")):
            result = self.solve()
        self.assert_evidence(model, result)
        self.assertTrue(steps)
        self.assertEqual(len(result.target_pose), 25)
        np.testing.assert_array_equal(result.qpos[qa:qa + 7], virtual)
        np.testing.assert_array_equal(integration_state(model, live), before_live)
        np.testing.assert_array_equal(self.seed, before_seed)
        after_model = model_snapshot(model)
        for name, value in before_model.items():
            np.testing.assert_array_equal(after_model[name], value, err_msg=name)
        self.assertEqual(manager.capture_events, events)
        self.assertEqual(manager.releases, releases)
        self.assertEqual(manager.active_attachments(), attachments)
        with self.assertRaises(FrozenInstanceError):
            result.converged = False
        for mapping in (result.target_pose, result.waist_target, result.waist_errors, result.source_frames):
            with self.assertRaises(TypeError):
                mapping["bad"] = 0.
        with self.assertRaises(TypeError):
            result.qpos[0] = 0.

    def test_unreachable_hand_and_overextended_root_return_legal_measured_failure(self):
        hand = Limb.RIGHT_HAND
        frame = limb_frame(self.source, hand)
        far = replace(frame, position=tuple(np.array(frame.position) + [0., 0., 2.]))
        for kwargs in ({"hand_targets": {hand: far}},
                       {"root_target": replace(self.goal, position=tuple(np.array(self.goal.position) + [0., 0., .4]))}):
            with self.subTest(kwargs=kwargs):
                result = self.solve(max_iterations=25, **kwargs)
                self.assert_evidence(self.model, result, converged=False)
                self.assertLessEqual(result.iterations, 25)
                self.assertIn("not a global infeasibility proof", result.reason)
                self.assertGreater(max(r.position_error for r in result.residuals), .01)

    def test_native_collision_masks_exclusions_and_declared_hand_pairs(self):
        for mode in ("enabled", "visual", "excluded"):
            tree = ET.fromstring(build_mjcf(self.scene, self.profile))
            attrs = {"name": "unexpected_body_obstacle", "type": "box", "size": ".1 .09 .04"}
            body = ET.SubElement(tree.find("worldbody"), "body", name="obstacle", pos=" ".join(map(str, self.start_root.position)))
            if mode == "visual":
                attrs.update(contype="0", conaffinity="0")
            ET.SubElement(body, "geom", **attrs)
            if mode == "excluded":
                for name in ("pelvis", "abdomen", "left_thigh", "right_thigh"):
                    ET.SubElement(tree.find("contact"), "exclude", body1=name, body2="obstacle")
            model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
            with self.subTest(mode=mode):
                result = solve_whole_body_reference(model, self.scene, self.profile, self.seed, self.intent,
                                                   root_target=self.start_root, waist_target=WAIST, max_iterations=0)
                self.assert_evidence(model, result, converged=mode != "enabled")
                self.assertEqual(any("unexpected_body_obstacle" in r.geoms for r in result.collisions), mode == "enabled")
                if mode == "enabled":
                    stalled = solve_whole_body_reference(model, self.scene, self.profile, self.seed, self.intent,
                                                        root_target=self.start_root, waist_target=WAIST, max_iterations=10)
                    self.assert_evidence(model, stalled, converged=False)
                    self.assertTrue(stalled.collisions)
                    self.assertIn("not a global infeasibility proof", stalled.reason)
        # A declared extra hold is not automatically allowed to contact a palm.
        hand = Limb.RIGHT_HAND
        source_hold = self.scene.region(self.intent[hand])
        duplicate = replace(source_hold, id="explicit_hand_touch",
                            position=tuple(np.array(source_hold.position) + .004 * np.array(source_hold.normal)))
        scene = replace(self.scene, contact_regions=(*self.scene.contact_regions, duplicate))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, self.profile))
        for declared in (False, True):
            result = solve_whole_body_reference(
                model, scene, self.profile, self.seed, self.intent, root_target=self.start_root, waist_target=WAIST,
                max_iterations=0, allowed_hand_holds={hand: (duplicate.id,)} if declared else None)
            self.assert_evidence(model, result, converged=declared)
            self.assertEqual(any("geom_explicit_hand_touch" in r.geoms for r in result.collisions), not declared)

    def test_nonzero_root_addresses_reordered_motors_and_passive_hinge_are_not_slices(self):
        tree = ET.fromstring(build_mjcf(self.scene, self.profile))
        body = ET.Element("body", name="passive_body", pos="3 -3 2")
        ET.SubElement(body, "joint", name="passive_hinge", type="hinge", range="-30 30")
        ET.SubElement(body, "geom", type="sphere", size=".01", mass=".1", contype="0", conaffinity="0")
        tree.find("worldbody").insert(0, body)
        motors = tree.find("actuator")
        motors[:] = list(reversed(list(motors)))
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        seed = model.qpos0.copy()
        for jid in range(self.model.njnt):
            name = self.model.joint(jid).name
            old, new = int(self.model.jnt_qposadr[jid]), int(model.joint(name).qposadr[0])
            width = 7 if name == "root" else 1
            seed[new:new + width] = self.seed[old:old + width]
        qa, va = int(model.joint("root").qposadr[0]), int(model.joint("root").dofadr[0])
        passive_qa, passive_va = int(model.joint("passive_hinge").qposadr[0]), int(model.joint("passive_hinge").dofadr[0])
        seed[passive_qa] = .17
        self.assertGreater(qa, 0)
        native = mujoco.mj_integratePos

        def integrate(m, qpos, tangent, dt):
            np.testing.assert_array_equal(tangent[va:va + 6], 0.)
            self.assertEqual(tangent[passive_va], 0.)
            native(m, qpos, tangent, dt)

        with patch("mujoco.mj_integratePos", side_effect=integrate):
            result = solve_whole_body_reference(model, self.scene, self.profile, seed, self.intent,
                                               root_target=self.goal, waist_target={**WAIST, "waist_pitch": .07})
        self.assert_evidence(model, result)
        self.assertEqual(result.qpos[passive_qa], .17)
        self.assertNotIn("passive_hinge", result.target_pose)
        np.testing.assert_array_equal(result.qpos[qa:qa + 7], (*self.goal.position, *self.goal.quaternion))

    def test_renamed_profile_morphology_world_frames_and_mirrored_root_motion(self):
        profile = ClimberProfile("arbitrary_not_a_reference_branch", torso_length=.60, shoulder_width=.46,
                                 hip_width=.34, upper_arm_length=.34, forearm_length=.29,
                                 thigh_length=.46, shin_length=.43, rom_scale=.8)
        empty = replace(self.scene, walls=(), contact_regions=(), start_configuration={})
        bare = mujoco.MjModel.from_xml_string(build_mjcf(empty, profile))
        seed = bare.qpos0.copy()
        for side in ("left", "right"):
            for name, value in (("hip_pitch", .45), ("knee", .9), ("ankle_pitch", .45),
                                ("shoulder_pitch", .3), ("elbow", math.pi / 2), ("wrist", -.3)):
                seed[int(bare.joint(f"{side}_{name}").qposadr[0])] = value
        source = probe(bare, seed)
        rotation, translation = yaw_rotation(-.31), np.array([.08, -.04, .07])
        regions, intent = [], {}
        for index, limb in enumerate(Limb):
            rid = f"anchor_{91 + 7 * index}"
            intent[limb] = rid
            template = self.scene.region(self.intent[limb])
            offset = .026 * np.array([0., -1., 0.]) if limb.is_hand else np.array([0., .01, .03 + FOOT_SITE_OFFSET])
            center = source.site(f"{limb.value.lower()}_site").xpos - offset
            regions.append(replace(template, id=rid, position=tuple(rotation @ center + translation),
                                   normal=tuple(rotation @ template.normal)))
        scene = replace(empty, contact_regions=tuple(regions), start_configuration=intent)
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
        seed[:3] = rotation @ seed[:3] + translation
        seed[3:7] = Frame((0., 0., 0.), tuple(map(tuple, rotation))).quaternion
        start = root_frame(model, probe(model, seed))
        pitch = math.radians(3.)
        lean = np.array([[1., 0., 0.], [0., math.cos(pitch), -math.sin(pitch)],
                         [0., math.sin(pitch), math.cos(pitch)]])
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                target = replace(start, position=tuple(np.array(start.position) + rotation @ [sign * .01, .005, .04]),
                                 rotation=tuple(map(tuple, rotation @ yaw_rotation(sign * math.radians(4.)) @ lean)))
                result = solve_whole_body_reference(model, scene, profile, seed, intent, root_target=target,
                                                   waist_target={"waist_yaw": .03, "waist_pitch": .07,
                                                                 "waist_roll": sign * .02})
                self.assert_evidence(model, result)
                for r in result.residuals:
                    self.assertEqual(r.region_id, intent[r.limb])
                    geometry = canonical_geometry(scene.region(r.region_id))
                    if r.limb.is_hand:
                        self.assertEqual(r.target, geometry.hand_frame)
                    else:
                        self.assertAlmostEqual(r.canonical_position_error, .01, delta=1e-6)
                renamed = solve_whole_body_reference(model, scene, replace(profile, name="another_name"), seed,
                                                    intent, root_target=target, waist_target=result.waist_target)
                np.testing.assert_array_equal(renamed.qpos, result.qpos)

    def test_actual_shoe_full_rotation_not_rotated_end_site(self):
        tree = ET.fromstring(build_mjcf(self.scene, self.profile))
        tree.find(".//site[@name='right_foot_site']").set("quat", "0.9238795325 0 0 0.3826834324")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        source = probe(model, self.seed)
        supports = {limb: limb_frame(source, limb) for limb in Limb}
        foot = Limb.RIGHT_FOOT
        supports[foot] = replace(supports[foot], rotation=tuple(map(tuple, np.array(supports[foot].rotation) @ yaw_rotation(.06))))
        result = solve_whole_body_reference(model, self.scene, self.profile, self.seed, self.intent,
                                           root_target=self.goal, waist_target={**WAIST, "waist_pitch": .07},
                                           support_frames=supports)
        data = self.assert_evidence(model, result)
        self.assertGreater(np.linalg.norm(data.site("right_foot_site").xmat.reshape(3, 3) - supports[foot].rotation), .5)
        np.testing.assert_allclose(data.geom("right_foot_geom").xmat.reshape(3, 3), supports[foot].rotation, rtol=0, atol=1e-6)

    def test_compiled_numerics_mode_motor_and_contact_geometry_rejected(self):
        for invalid in ("numerics", "mode", "motor", "waist", "shoe", "offset", "site", "foot_geom", "hold_frame", "hold_size"):
            with self.subTest(invalid=invalid):
                tree = ET.fromstring(build_mjcf(self.scene, self.profile))
                if invalid == "site":
                    tree.find(".//site[@name='right_foot_site']").set("name", "missing_foot_site")
                elif invalid == "shoe":
                    tree.find(".//geom[@name='right_foot_geom']").set("type", "ellipsoid")
                elif invalid == "offset":
                    tree.find(".//site[@name='right_foot_site']").set("pos", "0 .09 -.024")
                elif invalid == "foot_geom":
                    tree.find(".//geom[@name='right_foot_geom']").set("name", "missing_foot_geom")
                    for pair in tree.findall("./contact/pair[@geom1='right_foot_geom']"):
                        pair.set("geom1", "missing_foot_geom")
                elif invalid == "waist":
                    tree.find(".//joint[@name='waist_roll']").set("name", "other_roll")
                    tree.find(".//motor[@joint='waist_roll']").set("joint", "other_roll")
                elif invalid == "hold_frame":
                    tree.find(".//site[@name='site_right_hand']").set("pos", "0 0 0")
                elif invalid == "hold_size":
                    tree.find(".//geom[@name='geom_right_hand']").set("size", ".03")
                model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                if invalid == "numerics":
                    model.body_mass[model.body("right_foot").id] = np.nan
                elif invalid == "mode":
                    model.numeric("contact_mode").data[0] = 1.
                elif invalid == "motor":
                    model.actuator_gear[0, 0] = 0.
                before = model_snapshot(model)
                with self.assertRaises(ValueError):
                    solve_whole_body_reference(model, self.scene, self.profile, self.seed, self.intent,
                                               root_target=self.goal, waist_target=WAIST)
                after = model_snapshot(model)
                for name, value in before.items():
                    np.testing.assert_array_equal(after[name], value, err_msg=name)

    def test_malformed_frames_pose_intent_waist_and_rom_fail_before_fk(self):
        invalid = [
            {"root_target": None}, {"root_target": replace(self.goal, position=(np.nan, 0., 0.))},
            {"root_target": replace(self.goal, rotation=((1., 0., 0.),) * 3)},
            {"root_target": replace(self.goal, rotation=tuple(map(tuple, np.diag([1., 1., -1.]))))},
            {"hand_targets": {Limb.LEFT_FOOT: self.goal}}, {"support_frames": {"LEFT_FOOT": self.goal}},
            {"support_frames": {Limb.LEFT_FOOT: replace(self.goal, position=(0., 0.))}},
            {"support_frames": {Limb.LEFT_FOOT: replace(self.goal, rotation=((1., 0.),))}},
            {"waist_target": {}}, {"waist_target": {**WAIST, "waist_pitch": np.inf}},
            {"waist_target": {**WAIST, "waist_pitch": 3.}},
            {"allowed_hand_holds": {Limb.LEFT_FOOT: ("left_foot",)}},
            {"allowed_hand_holds": {Limb.LEFT_HAND: ("left_foot",)}},
            {"allowed_hand_holds": {Limb.LEFT_HAND: "left_hand"}},
            {"max_iterations": True}, {"max_iterations": -1}, {"max_iterations": 1.5},
            {"tolerance": None}, {"tolerance": 0.}, {"tolerance": np.nan}, {"tolerance": np.inf},
        ]
        with patch("mujoco.mj_forward", side_effect=AssertionError("invalid input reached FK")):
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.solve(**kwargs)
            for name in ("waist_pitch", "left_ankle_roll", "right_elbow"):
                seed = self.seed.copy()
                joint = self.model.joint(name)
                seed[int(joint.qposadr[0])] = joint.range[1] + 1e-9
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "compiled ROM"):
                    solve_whole_body_reference(self.model, self.scene, self.profile, seed, self.intent,
                                               root_target=self.goal, waist_target=WAIST)
            for seed in (self.seed[:-1], self.seed[:, None], np.full(self.model.nq, np.nan),
                         ["bad"] * self.model.nq, mujoco.MjData(self.model)):
                with self.assertRaises(ValueError):
                    solve_whole_body_reference(self.model, self.scene, self.profile, seed, self.intent,
                                               root_target=self.goal, waist_target=WAIST)
            for quat in ((0., 0., 0., 0.), (1.000001, 0., 0., 0.)):
                seed = self.seed.copy()
                seed[3:7] = quat
                with self.assertRaisesRegex(ValueError, "unit length"):
                    solve_whole_body_reference(self.model, self.scene, self.profile, seed, self.intent,
                                               root_target=self.goal, waist_target=WAIST)
            for intent in ({Limb.LEFT_HAND: "left_hand"}, {**self.intent, Limb.LEFT_HAND: "missing"},
                           {**self.intent, Limb.LEFT_HAND: "left_foot"}, {**self.intent, Limb.LEFT_HAND: None}):
                with self.assertRaises(ValueError):
                    solve_whole_body_reference(self.model, self.scene, self.profile, self.seed, intent,
                                               root_target=self.goal, waist_target=WAIST)


if __name__ == "__main__":
    unittest.main()
