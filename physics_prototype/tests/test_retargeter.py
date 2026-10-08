import unittest

import numpy as np

from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.retargeter import (
    ACTUATED_JOINT_NAMES,
    RetargetResult,
    StanceSpecification,
    solve_retargeted_stance,
)
from boulder_v1.runtime import compile_model, mujoco_available
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import ClimberProfile, Limb


class RetargeterTests(unittest.TestCase):
    def setUp(self):
        if not mujoco_available():
            self.skipTest("MuJoCo is not available")
        import mujoco

        self.scene = make_synthetic_scene()
        self.base_profile = ClimberProfile(name="base")
        self.xml = build_mjcf(self.scene, self.base_profile)
        self.model = compile_model(self.xml)

    def test_stance_specification_defaults(self):
        spec = StanceSpecification()
        self.assertEqual(spec.hand_holds[Limb.LEFT_HAND], "H3")
        self.assertEqual(spec.hand_holds[Limb.RIGHT_HAND], "H4")
        self.assertEqual(spec.foot_holds[Limb.LEFT_FOOT], "H1")
        self.assertEqual(spec.foot_holds[Limb.RIGHT_FOOT], "H2")
        self.assertAlmostEqual(spec.pelvis_wall_distance, 0.64)
        self.assertEqual(spec.pelvis_lateral_bias, 0.0)

    def test_solve_retargeted_stance_converges_submillimeter(self):
        """Position-only IK evidence at canonical anchors, not physical acquisition."""
        import mujoco

        result = solve_retargeted_stance(self.model, self.scene, self.base_profile)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 30)

        # Check end-effector errors are all below 1.5mm
        for limb, err in result.errors.items():
            self.assertLess(err, 0.0015, f"{limb} error {err} exceeds tolerance")

        data = mujoco.MjData(self.model)
        data.qpos[:] = result.qpos
        mujoco.mj_forward(self.model, data)
        for limb, hold in self.scene.start_configuration.items():
            geometry = canonical_geometry(self.scene.region(hold))
            target = geometry.hand_frame if limb.is_hand else geometry.foot_frame
            measured = np.linalg.norm(data.site(f"{limb.value.lower()}_site").xpos - target.position)
            self.assertAlmostEqual(result.errors[limb], measured, places=14)
        self.assertFalse(np.any(data.eq_active))
        self.assertEqual(data.time, 0.)

        # Verify all 25 actuated joint angles are present in target_pose
        self.assertEqual(len(result.target_pose), len(ACTUATED_JOINT_NAMES))
        for jname in ACTUATED_JOINT_NAMES:
            self.assertIn(jname, result.target_pose)

    def test_solve_retargeted_stance_respects_joint_limits(self):
        result = solve_retargeted_stance(self.model, self.scene, self.base_profile)
        import mujoco

        for jname in ACTUATED_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if self.model.jnt_limited[jid]:
                r = self.model.jnt_range[jid]
                val = result.target_pose[jname]
                self.assertGreaterEqual(val, r[0] - 1e-4, f"{jname} below min limit")
                self.assertLessEqual(val, r[1] + 1e-4, f"{jname} above max limit")

    def test_solve_retargeted_stance_profile_variations(self):
        profiles = [
            ClimberProfile(name="base"),
            ClimberProfile(name="long_reach_lower_grip", upper_arm_length=0.36, forearm_length=0.32),
            ClimberProfile(name="compact_strong", thigh_length=0.38, shin_length=0.36),
        ]
        for prof in profiles:
            xml = build_mjcf(self.scene, prof)
            model = compile_model(xml)
            res = solve_retargeted_stance(model, self.scene, prof)
            self.assertTrue(res.converged, f"Failed to converge for {prof.name}")
            for limb, err in res.errors.items():
                self.assertLess(err, 0.002, f"{prof.name} {limb} error too high")

    def test_solve_retargeted_stance_with_free_limb(self):
        spec = StanceSpecification(
            pelvis_lateral_bias=-0.15,
            free_limbs=(Limb.RIGHT_HAND,),
        )
        res = solve_retargeted_stance(self.model, self.scene, self.base_profile, spec=spec)
        self.assertTrue(res.converged)
        self.assertNotIn(Limb.RIGHT_HAND, res.errors)
        self.assertIn(Limb.LEFT_HAND, res.errors)
        self.assertIn(Limb.LEFT_FOOT, res.errors)
        self.assertIn(Limb.RIGHT_FOOT, res.errors)
        self.assertAlmostEqual(res.pelvis_pos[0], -0.1634, delta=0.03)


if __name__ == "__main__":
    unittest.main()
