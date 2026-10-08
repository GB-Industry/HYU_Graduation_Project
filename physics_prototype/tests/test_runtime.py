import unittest

from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import (
    compile_model,
    get_end_effector_positions,
    mujoco_available,
    run_pose_control,
    smoke_step,
)
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import ClimberProfile


class RuntimeTests(unittest.TestCase):
    def test_missing_mujoco_has_actionable_error(self):
        if mujoco_available():
            self.skipTest("MuJoCo is installed; missing-dependency path not applicable")
        with self.assertRaisesRegex(RuntimeError, "pip install"):
            compile_model("<mujoco/>")

    @unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
    def test_model_compiles_and_steps(self):
        xml = build_mjcf(make_synthetic_scene(), ClimberProfile(name="base"))
        result = smoke_step(xml, steps=50)
        self.assertTrue(result.finite)
        self.assertEqual(result.steps, 50)
        self.assertGreater(result.nu, 0)
        import mujoco
        from unittest import mock

        model = compile_model(xml)
        self.assertEqual(model.numeric("contact_mode").data[0], 0.)
        self.assertEqual((model.neq, model.nexclude, model.npair), (16, 15, 20))
        self.assertTrue(all("hand" in model.equality(i).name for i in range(model.neq)))

        malformed = xml.replace('gear="90.0"', 'gear="1e154"', 1)
        self.assertNotEqual(malformed, xml)
        with mock.patch.object(mujoco, "mj_forward", side_effect=AssertionError("bad model must not be forwarded")), \
                mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("bad model must not be integrated")):
            for execute in (compile_model, smoke_step, run_pose_control):
                with self.subTest(execute=execute.__name__), self.assertRaisesRegex(ValueError, "compiled model numerics"):
                    execute(malformed)

    @unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
    def test_end_effector_sites_queryable(self):
        import mujoco
        xml = build_mjcf(make_synthetic_scene(), ClimberProfile(name="base"))
        model = compile_model(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        positions = get_end_effector_positions(model, data)
        self.assertEqual(len(positions), 4)
        for name in ("left_hand_site", "right_hand_site", "left_foot_site", "right_foot_site"):
            self.assertIn(name, positions)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.assertGreaterEqual(site_id, 0)
            x, y, z = positions[name]
            self.assertTrue(all(isinstance(v, float) and float("-inf") < v < float("inf") for v in (x, y, z)))

    @unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
    def test_deterministic_pose_control_stays_finite(self):
        xml = build_mjcf(make_synthetic_scene(), ClimberProfile(name="base"))
        result = run_pose_control(xml, steps=300, stabilize_root=False)
        self.assertTrue(result.finite)
        self.assertTrue(result.error_reduced)
        self.assertLess(result.final_error, result.initial_error)
        self.assertEqual(len(result.end_effector_positions), 4)

    @unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
    def test_deterministic_pose_control_with_stabilize_root(self):
        xml = build_mjcf(make_synthetic_scene(), ClimberProfile(name="base"))
        result = run_pose_control(xml, steps=300, stabilize_root=True)
        self.assertTrue(result.finite)
        self.assertTrue(result.error_reduced)
        self.assertLess(result.final_error, result.initial_error)
        self.assertEqual(len(result.end_effector_positions), 4)

    @unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
    def test_pose_control_preserves_profile_variations(self):
        scene = make_synthetic_scene()
        long_reach = ClimberProfile(
            name="long_reach",
            upper_arm_length=0.34,
            forearm_length=0.30,
            thigh_length=0.44,
            shin_length=0.42,
        )
        compact = ClimberProfile(
            name="compact",
            upper_arm_length=0.28,
            forearm_length=0.24,
            thigh_length=0.39,
            shin_length=0.37,
        )
        res_long = run_pose_control(build_mjcf(scene, long_reach), steps=100)
        res_compact = run_pose_control(build_mjcf(scene, compact), steps=100)
        self.assertTrue(res_long.finite)
        self.assertTrue(res_compact.finite)
        self.assertNotEqual(
            res_long.end_effector_positions["left_hand_site"],
            res_compact.end_effector_positions["left_hand_site"],
        )

    def test_debug_reference_pose_is_rejected_when_outside_profile_rom(self):
        from unittest import mock
        import mujoco
        from boulder_v1.runtime import validate_reference_pose

        xml = build_mjcf(make_synthetic_scene(), ClimberProfile(name="narrow", rom_scale=.1))
        with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("invalid reference must not be integrated")):
            with self.assertRaisesRegex(ValueError, "violates compiled ROM"):
                run_pose_control(xml)
        model = compile_model(xml)
        for quaternion in ([0, 0, 0, 0], [2, 0, 0, 0]):
            reference = model.qpos0.copy()
            reference[3:7] = quaternion
            with self.assertRaisesRegex(ValueError, "must be unit length"):
                validate_reference_pose(model, reference)



if __name__ == "__main__":
    unittest.main()
