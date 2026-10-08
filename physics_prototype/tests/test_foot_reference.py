"""Six-hinge scratch foot references, not foot acquisition or dynamic tests."""

from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.contact_geometry import FOOT_SITE_OFFSET, Frame, canonical_geometry
from boulder_v1.contact_ik import HandReferenceResult, solve_foot_reference
from boulder_v1.grasp import GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Limb
from boulder_v1.single_hand import make_single_hand_fixture


FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
HINGES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


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


def probe_pose(model, qpos):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    return scratch


def foot_frame(data, limb):
    name = limb.value.lower()
    return Frame(tuple(data.site(f"{name}_site").xpos),
                 tuple(map(tuple, data.geom(f"{name}_geom").xmat.reshape(3, 3))))


class FootReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Construction only: never run the single-hand benchmark or live physics.
        cls.model, _, cls.scene, cls.profile, cls.seed = make_single_hand_fixture()

    def assert_model_unchanged(self, model, before):
        after = model_snapshot(model)
        self.assertEqual(after.keys(), before.keys())
        for name, value in before.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(after[name].tobytes(), value.tobytes(), name)
            else:
                self.assertEqual(after[name], value, name)

    def assert_reference(self, model, measured, limb, target, result, *, converged=True):
        self.assertIsInstance(result, HandReferenceResult)
        self.assertEqual(result.converged, converged, result.reason)
        validate_reference_pose(model, result.qpos)
        side = limb.value.lower().removesuffix("_foot")
        joints = [model.joint(f"{side}_{name}").id for name in HINGES]
        frozen = np.ones(model.nq, dtype=bool)
        frozen[model.jnt_qposadr[joints]] = False
        self.assertEqual(np.asarray(result.qpos)[frozen].tobytes(),
                         np.asarray(measured)[frozen].tobytes())
        scratch = probe_pose(model, result.qpos)
        frame = foot_frame(scratch, limb)
        current, goal = np.asarray(frame.rotation), np.asarray(target.rotation)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (goal @ current.T).ravel())
        angle = 2. * math.atan2(float(np.linalg.norm(quaternion[1:])), abs(float(quaternion[0])))
        alignment = float(np.clip(current[:, 2] @ goal[:, 2], -1., 1.))
        self.assertAlmostEqual(result.position_error, np.linalg.norm(np.array(frame.position) - target.position),
                               delta=1e-12)
        self.assertAlmostEqual(result.orientation_error, angle, delta=1e-12)
        self.assertAlmostEqual(result.orientation_alignment, alignment, delta=1e-12)
        self.assertFalse(hasattr(result, "admitted"))
        self.assertFalse(hasattr(result, "reference"))
        self.assertIn("no contact admission", result.reason)
        if converged:
            self.assertLessEqual(result.position_error, 1e-7)
            self.assertLessEqual(result.orientation_error, 1e-7)
            np.testing.assert_allclose(frame.position, target.position, rtol=0, atol=1e-7)
            np.testing.assert_allclose(frame.rotation, target.rotation, rtol=0, atol=1e-7)
        source = probe_pose(model, measured)
        for other in Limb:
            if other != limb:
                name = f"{other.value.lower()}_site"
                self.assertEqual(scratch.site(name).xpos.tobytes(), source.site(name).xpos.tobytes())
                self.assertEqual(scratch.site(name).xmat.tobytes(), source.site(name).xmat.tobytes())

    def test_both_legs_float_and_lateral_step_from_slightly_flexed_source(self):
        source = probe_pose(self.model, self.seed)
        before_model, before_seed = model_snapshot(self.model), self.seed.tobytes()
        for limb in FEET:
            start = foot_frame(source, limb)
            side = limb.value.lower().removesuffix("_foot")
            np.testing.assert_array_equal(self.model.joint(f"{side}_knee").axis, [-1., 0., 0.])
            np.testing.assert_array_equal(source.site(f"{limb.value.lower()}_site").xmat,
                                          source.geom(f"{limb.value.lower()}_geom").xmat)
            for dx, dz in ((0., .02), (0., .03), (0., .04), (-.04, .02), (.04, .02)):
                with self.subTest(limb=limb, dx=dx, dz=dz):
                    target = replace(start, position=tuple(np.array(start.position) + [dx, 0., dz]))
                    result = solve_foot_reference(self.model, self.seed, limb, target)
                    self.assert_reference(self.model, self.seed, limb, target, result)
                    self.assertGreater(result.iterations, 0)
                    self.assertGreater(result.qpos[int(self.model.joint(f"{side}_knee").qposadr[0])], .3)
        self.assertEqual(self.seed.tobytes(), before_seed)
        self.assert_model_unchanged(self.model, before_model)

    def test_straight_knee_boundary_failure_is_not_global_infeasibility(self):
        seed = self.seed.copy()
        for side in ("left", "right"):
            for name in HINGES:
                seed[int(self.model.joint(f"{side}_{name}").qposadr[0])] = 0.
        source = probe_pose(self.model, seed)
        for limb in FEET:
            with self.subTest(limb=limb):
                start = foot_frame(source, limb)
                target = replace(start, position=tuple(np.array(start.position) + [0., 0., .02]))
                result = solve_foot_reference(self.model, seed, limb, target)
                self.assert_reference(self.model, seed, limb, target, result, converged=False)
                self.assertIn("local search stalled", result.reason)
                self.assertIn("not a global infeasibility proof", result.reason)
                # The ankle body's 20mm offset initially makes positive knee
                # flexion LOWER a flat shoe. Monotone local DLS cannot cross that
                # boundary valley, but the requested bent source can reach it.
                recovered = solve_foot_reference(self.model, self.seed, limb, target)
                self.assert_reference(self.model, self.seed, limb, target, recovered)

    def test_supplied_sole_plane_offset_and_tangential_centering_are_exact_targets(self):
        self.assertEqual(FOOT_SITE_OFFSET, .011)
        for limb in FEET:
            with self.subTest(limb=limb):
                geometry = canonical_geometry(self.scene.region(limb.value.lower()))
                target = replace(geometry.foot_frame,
                                 position=tuple(np.array(geometry.foot_frame.position) + [.04, 0., .02]))
                result = solve_foot_reference(self.model, self.seed, limb, target)
                self.assert_reference(self.model, self.seed, limb, target, result)
                scratch = probe_pose(self.model, result.qpos)
                geom = scratch.geom(f"{limb.value.lower()}_geom")
                normal = geom.xmat.reshape(3, 3)[:, 2]
                sole = geom.xpos - self.model.geom_size[geom.id, 2] * normal
                self.assertAlmostEqual((scratch.site(f"{limb.value.lower()}_site").xpos - sole) @ normal,
                                       FOOT_SITE_OFFSET, delta=1e-12)
                self.assertAlmostEqual((sole - geometry.foot_surface_frame.position) @ normal, .02, delta=1e-7)

    def test_full_orientation_includes_tangent_twist_and_matched_pose_is_bitexact(self):
        source = probe_pose(self.model, self.seed)
        angle = .06
        twist = np.array([[math.cos(angle), -math.sin(angle), 0.],
                          [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        for limb in FEET:
            with self.subTest(limb=limb):
                start = foot_frame(source, limb)
                matched = solve_foot_reference(self.model, self.seed, limb, start, max_iterations=0)
                self.assert_reference(self.model, self.seed, limb, start, matched)
                self.assertEqual(matched.iterations, 0)
                self.assertEqual(np.asarray(matched.qpos).tobytes(), self.seed.tobytes())
                target = replace(start, rotation=tuple(map(tuple, np.array(start.rotation) @ twist)))
                rejected = solve_foot_reference(self.model, self.seed, limb, target, max_iterations=0)
                self.assert_reference(self.model, self.seed, limb, target, rejected, converged=False)
                self.assertAlmostEqual(rejected.orientation_error, angle, places=12)
                self.assertAlmostEqual(rejected.orientation_alignment, 1., places=12)
                solved = solve_foot_reference(self.model, self.seed, limb, target)
                self.assert_reference(self.model, self.seed, limb, target, solved)
                self.assertGreater(solved.iterations, 0)

    def test_measured_rotated_root_full_rotation_step_bound_and_live_model_nonmutation(self):
        model = self.model
        live = mujoco.MjData(model)
        live.qpos[:] = self.seed
        live.qpos[0] = -0.
        live.qpos[3:7] = np.array([.98, .08, -.04, .10]) / np.linalg.norm([.98, .08, -.04, .10])
        for name, value in (("waist_yaw", .04), ("waist_pitch", -.025), ("waist_roll", .03),
                            ("left_shoulder_roll", .02), ("left_hip_roll", -.03)):
            live.qpos[int(model.joint(name).qposadr[0])] = value
        live.ctrl[:] = .123
        live.qvel[:] = np.linspace(-.01, .01, model.nv)
        live.time = .73
        live.xfrc_applied[model.body("climber_root").id, :3] = (1., 2., 3.)
        mujoco.mj_forward(model, live)
        measured = live.qpos.copy()
        goal = measured.copy()
        for name, value in (("hip_pitch", .24), ("hip_roll", -.06), ("hip_yaw", .05),
                            ("knee", .48), ("ankle_pitch", .22), ("ankle_roll", .04)):
            goal[int(model.joint(f"right_{name}").qposadr[0])] = value
        target = foot_frame(probe_pose(model, goal), Limb.RIGHT_FOOT)
        before_state, before_model = integration_state(model, live), model_snapshot(model)
        joints = [model.joint(f"right_{name}").id for name in HINGES]
        frozen = np.ones(model.nv, dtype=bool)
        frozen[model.jnt_dofadr[joints]] = False
        native_integrate, native_forward = mujoco.mj_integratePos, mujoco.mj_forward
        steps = []

        def integrate(m, qpos, tangent, dt):
            self.assertIs(m, model)
            self.assertFalse(np.shares_memory(qpos, live.qpos))
            self.assertFalse(np.shares_memory(qpos, measured))
            np.testing.assert_array_equal(tangent[frozen], 0.)
            self.assertLessEqual(np.linalg.norm(tangent), .1 + 1e-15)
            steps.append(tangent.copy())
            native_integrate(m, qpos, tangent, dt)

        def forward(m, data):
            self.assertIs(m, model)
            self.assertIsNot(data, live)
            self.assertFalse(np.shares_memory(data.qpos, live.qpos))
            self.assertFalse(np.any(data.eq_active))
            validate_reference_pose(m, data.qpos)
            qfrozen = np.ones(model.nq, dtype=bool)
            qfrozen[model.jnt_qposadr[joints]] = False
            self.assertEqual(data.qpos[qfrozen].tobytes(), measured[qfrozen].tobytes())
            native_forward(m, data)

        with patch("mujoco.mj_integratePos", side_effect=integrate), \
                patch("mujoco.mj_forward", side_effect=forward), \
                patch("mujoco.mj_step", side_effect=AssertionError("leg reference stepped physics")), \
                patch.object(GraspManager, "attach", side_effect=AssertionError("leg reference attempted capture")), \
                patch("boulder_v1.contact_ik.initialize_static_reference",
                      side_effect=AssertionError("leg reference attempted admission")):
            result = solve_foot_reference(model, live.qpos, Limb.RIGHT_FOOT, target)
        self.assert_reference(model, measured, Limb.RIGHT_FOOT, target, result)
        self.assertTrue(steps)
        self.assertEqual(live.qpos.tobytes(), measured.tobytes())
        self.assertEqual(integration_state(model, live).tobytes(), before_state.tobytes())
        self.assert_model_unchanged(model, before_model)
        self.assertEqual(int(model.joint("root").type[0]), mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(all("foot" not in model.equality(i).name for i in range(model.neq)))
        with self.assertRaises(FrozenInstanceError):
            result.converged = False
        with self.assertRaises(TypeError):
            result.qpos[0] = 0.

    def test_reversed_normal_cannot_false_converge_at_zero_position_gap(self):
        source = probe_pose(self.model, self.seed)
        before_model, before_seed = model_snapshot(self.model), self.seed.tobytes()
        for limb in FEET:
            start = foot_frame(source, limb)
            target = replace(start, rotation=tuple(map(tuple, np.array(start.rotation) @ np.diag([1., -1., -1.]))))
            for budget in (0, 1, 30):
                with self.subTest(limb=limb, budget=budget):
                    result = solve_foot_reference(self.model, self.seed, limb, target, max_iterations=budget)
                    self.assert_reference(self.model, self.seed, limb, target, result, converged=False)
                    self.assertGreater(result.orientation_error, 1.)
                    if not budget:
                        self.assertLess(result.position_error, 1e-12)
                        self.assertAlmostEqual(result.orientation_error, math.pi, places=12)
                        self.assertAlmostEqual(result.orientation_alignment, -1., places=12)
        self.assertEqual(self.seed.tobytes(), before_seed)
        self.assert_model_unchanged(self.model, before_model)

    def test_unreachable_targets_return_measured_legal_uncertified_failure(self):
        source = probe_pose(self.model, self.seed)
        before_model, before_seed = model_snapshot(self.model), self.seed.tobytes()
        for limb in FEET:
            with self.subTest(limb=limb):
                start = foot_frame(source, limb)
                target = replace(start, position=tuple(np.array(start.position) + [5., -2., 3.]))
                result = solve_foot_reference(self.model, self.seed, limb, target, max_iterations=30)
                self.assert_reference(self.model, self.seed, limb, target, result, converged=False)
                self.assertGreater(result.position_error, 1.)
                self.assertLessEqual(result.iterations, 30)
                self.assertIn("not a global infeasibility proof", result.reason)
        self.assertEqual(self.seed.tobytes(), before_seed)
        self.assert_model_unchanged(self.model, before_model)

    def test_any_measured_rom_deviation_or_nonunit_quaternion_fails_atomically(self):
        target = foot_frame(probe_pose(self.model, self.seed), Limb.RIGHT_FOOT)
        before_model = model_snapshot(self.model)
        seeds = []
        for name, bound, sign in (("right_knee", 0, -1), ("right_ankle_pitch", 1, 1),
                                 ("left_ankle_roll", 0, -1), ("left_elbow", 1, 1), ("waist_yaw", 1, 1)):
            seed = self.seed.copy()
            joint = self.model.joint(name)
            seed[int(joint.qposadr[0])] = joint.range[bound] + sign * 1e-9
            seeds.append((seed, "compiled ROM"))
        for quaternion in ((0., 0., 0., 0.), (1.000001, 0., 0., 0.)):
            seed = self.seed.copy()
            seed[3:7] = quaternion
            seeds.append((seed, "unit length"))
        for seed, message in seeds:
            before = seed.tobytes()
            with self.subTest(message=message), \
                    patch("mujoco.mj_forward", side_effect=AssertionError("invalid measured pose forwarded")), \
                    patch("mujoco.mj_integratePos", side_effect=AssertionError("invalid measured pose clipped")), \
                    self.assertRaisesRegex(ValueError, message):
                solve_foot_reference(self.model, seed, Limb.RIGHT_FOOT, target)
            self.assertEqual(seed.tobytes(), before)
        self.assert_model_unchanged(self.model, before_model)

    def test_nonfinite_malformed_inputs_fail_before_forward_without_writes(self):
        target = foot_frame(probe_pose(self.model, self.seed), Limb.RIGHT_FOOT)
        before_model, before_seed = model_snapshot(self.model), self.seed.tobytes()
        invalid_seeds = [np.full(self.model.nq, value) for value in (np.nan, np.inf, -np.inf)]
        invalid_seeds.extend((self.seed[:-1], self.seed[:, None], mujoco.MjData(self.model), ["bad"] * self.model.nq))
        invalid_targets = (
            None, replace(target, position=(np.nan, 0., 0.)), replace(target, position=(0., np.inf, 0.)),
            replace(target, position=(0., 0.)), replace(target, position=("bad", 0., 0.)),
            replace(target, rotation=((np.nan, 0., 0.), (0., 1., 0.), (0., 0., 1.))),
            replace(target, rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., -1.))),
            replace(target, rotation=((1., 0., 0.),) * 3), replace(target, rotation=((1., 0.),)),
        )
        with patch("mujoco.mj_forward", side_effect=AssertionError("invalid input forwarded")), \
                patch("mujoco.mj_integratePos", side_effect=AssertionError("invalid input integrated")):
            for seed in invalid_seeds:
                before = seed.tobytes() if isinstance(seed, np.ndarray) else None
                with self.subTest(seed_type=type(seed).__name__), self.assertRaises(ValueError):
                    solve_foot_reference(self.model, seed, Limb.RIGHT_FOOT, target)
                if before is not None:
                    self.assertEqual(seed.tobytes(), before)
            for frame in invalid_targets:
                with self.subTest(frame=frame), self.assertRaises(ValueError):
                    solve_foot_reference(self.model, self.seed, Limb.RIGHT_FOOT, frame)
            for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND, "RIGHT_FOOT", None):
                with self.subTest(limb=limb), self.assertRaises(ValueError):
                    solve_foot_reference(self.model, self.seed, limb, target)
            for kwargs in ({"max_iterations": -1}, {"max_iterations": True}, {"max_iterations": 1.5},
                           {"tolerance": 0.}, {"tolerance": np.nan}, {"tolerance": np.inf}, {"tolerance": None}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    solve_foot_reference(self.model, self.seed, Limb.RIGHT_FOOT, target, **kwargs)
        self.assertEqual(self.seed.tobytes(), before_seed)
        self.assert_model_unchanged(self.model, before_model)

    def test_compiled_nonfinite_model_and_missing_leg_are_rejected_atomically(self):
        for invalid in ("numerics", "joint", "site", "shoe"):
            with self.subTest(invalid=invalid):
                tree = ET.fromstring(build_mjcf(self.scene, self.profile))
                if invalid == "joint":
                    tree.find(".//joint[@name='right_hip_yaw']").set("name", "missing_hip_yaw")
                    tree.find(".//motor[@joint='right_hip_yaw']").set("joint", "missing_hip_yaw")
                elif invalid == "site":
                    tree.find(".//site[@name='right_foot_site']").set("name", "missing_foot_site")
                elif invalid == "shoe":
                    tree.find(".//geom[@name='right_foot_geom']").set("type", "ellipsoid")
                model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                if invalid == "numerics":
                    model.body_mass[model.body("right_foot").id] = np.nan
                before_model, before_seed = model_snapshot(model), self.seed.tobytes()
                target = Frame((0., 0., 0.), tuple(map(tuple, np.eye(3))))
                with patch("mujoco.mj_forward", side_effect=AssertionError("invalid model forwarded")), \
                        self.assertRaises(ValueError):
                    solve_foot_reference(model, self.seed, Limb.RIGHT_FOOT, target)
                self.assertEqual(self.seed.tobytes(), before_seed)
                self.assert_model_unchanged(model, before_model)

    def test_actual_shoe_rotation_not_locally_rotated_site_drives_orientation(self):
        tree = ET.fromstring(build_mjcf(self.scene, self.profile))
        tree.find(".//site[@name='right_foot_site']").set("quat", "0.9238795325 0 0 0.3826834324")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        source = probe_pose(model, self.seed)
        start = foot_frame(source, Limb.RIGHT_FOOT)
        self.assertGreater(np.linalg.norm(source.site("right_foot_site").xmat.reshape(3, 3) - start.rotation), .5)
        target = replace(start, position=tuple(np.array(start.position) + [.04, 0., .02]))
        before_model = model_snapshot(model)
        result = solve_foot_reference(model, self.seed, Limb.RIGHT_FOOT, target)
        self.assert_reference(model, self.seed, Limb.RIGHT_FOOT, target, result)
        self.assert_model_unchanged(model, before_model)

    def test_compiled_morphology_and_restricted_rom_not_hardcoded_lengths(self):
        profile = replace(self.profile, name="foot_reference_morphology", thigh_length=.46, shin_length=.43,
                          hip_width=.34, rom_scale=.8)
        model = mujoco.MjModel.from_xml_string(build_mjcf(self.scene, profile))
        seed = model.qpos0.copy()
        for side in ("left", "right"):
            for name, angle in (("hip_pitch", .15), ("knee", .3), ("ankle_pitch", .15)):
                seed[int(model.joint(f"{side}_{name}").qposadr[0])] = angle
        source = probe_pose(model, seed)
        before_model, before_seed = model_snapshot(model), seed.tobytes()
        for limb in FEET:
            for dx in (-.04, .04):
                with self.subTest(limb=limb, dx=dx):
                    start = foot_frame(source, limb)
                    target = replace(start, position=tuple(np.array(start.position) + [dx, 0., .02]))
                    result = solve_foot_reference(model, seed, limb, target)
                    self.assert_reference(model, seed, limb, target, result)
        self.assertEqual(seed.tobytes(), before_seed)
        self.assert_model_unchanged(model, before_model)


if __name__ == "__main__":
    unittest.main()
