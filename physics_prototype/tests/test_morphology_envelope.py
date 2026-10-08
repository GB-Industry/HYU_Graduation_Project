"""Fixed-task profile regeneration and native ownership for Stage5.2."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import morphology_envelope as study
from boulder_v1.static_state import _integration_state


class EnvelopeFixtureTests(unittest.TestCase):
    def test_bounded_one_factor_profiles_and_geometry_derived_neighborhood(self):
        profiles = study.study_profiles()
        base = profiles["baseline"]
        for name, scale in (("shorter", .95), ("longer", 1.05)):
            profile = profiles[name]
            for key in ("upper_arm_length", "forearm_length", "thigh_length", "shin_length"):
                self.assertAlmostEqual(getattr(profile, key), scale * getattr(base, key))
            for key in ("mass_scale", "rom_scale", "strength_scale", "grip_capacity", "torso_length",
                        "shoulder_width", "hip_width"):
                self.assertEqual(getattr(profile, key), getattr(base, key))
        self.assertEqual(replace(profiles["reduced_grip"], name=base.name, grip_capacity=base.grip_capacity), base)
        self.assertEqual(profiles["reduced_grip"].grip_capacity, 425.)
        cases = study.target_perturbations()
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases["up"], (0., .01, 0.))
        self.assertEqual(cases["down"], (0., -.01, 0.))
        self.assertEqual(cases["lateral_minus"], (-.01, 0., 0.))
        self.assertEqual(cases["lateral_plus"], (.01, 0., 0.))

    def test_fixed_scene_sources_are_independently_solved_without_steps(self):
        models, scenes, seeds = [], [], []
        with patch("mujoco.mj_step", side_effect=AssertionError("fixture stepped physics")):
            for profile in study.study_profiles().values():
                model, data, scene, returned, seed, metadata = study.make_envelope_fixture(profile)
                self.assertIs(returned, profile)
                self.assertEqual(data.time, 0.)
                self.assertEqual(metadata["mass_kg"], 78.3)
                self.assertFalse(any("foot" in model.equality(i).name for i in range(model.neq)))
                models.append(model); scenes.append(scene); seeds.append(seed)
        self.assertTrue(all(scene == scenes[0] for scene in scenes))
        for model in models[1:]:
            np.testing.assert_array_equal(model.jnt_range, models[0].jnt_range)
            np.testing.assert_array_equal(model.actuator_gear, models[0].actuator_gear)
        self.assertFalse(np.array_equal(seeds[0], seeds[1]))
        self.assertFalse(np.array_equal(seeds[0], seeds[2]))
        np.testing.assert_array_equal(seeds[0], seeds[3])

    def test_target_offsets_are_common_metres_not_profile_rescaled_tasks(self):
        for case in study.target_perturbations():
            scenes = [study.make_envelope_fixture(profile, case)[2]
                      for profile in study.study_profiles().values()]
            self.assertTrue(all(scene == scenes[0] for scene in scenes))
        with self.assertRaises(ValueError):
            study.make_envelope_fixture(study.study_profiles()["baseline"], "unknown")


class EnvelopeNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls.runs.clear()

    def run_case(self, name, dt):
        key = name, dt
        if key not in self.runs:
            native_initialize = study.initialize_static_reference
            native_step = mujoco.mj_step
            session = {}

            def initialize(model, data, *args, **kwargs):
                self.assertFalse(session, "episode was initialized more than once")
                result = native_initialize(model, data, *args, **kwargs)
                session.update(model=model, data=data, qpos=data.qpos.copy(), qvel=data.qvel.copy(), steps=0)
                return result

            def step(model, data):
                self.assertIs(model, session["model"])
                self.assertIs(data, session["data"])
                np.testing.assert_array_equal(data.qpos, session["qpos"])
                np.testing.assert_array_equal(data.qvel, session["qvel"])
                self.assertFalse(np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
                native_step(model, data)
                np.testing.assert_array_equal(data.qfrc_actuator[:6], 0.)
                session.update(qpos=data.qpos.copy(), qvel=data.qvel.copy(), steps=session["steps"] + 1)

            with patch.object(study, "initialize_static_reference", side_effect=initialize), \
                    patch("mujoco.mj_step", side_effect=step):
                result = study.run_envelope_case(study.study_profiles()[name], timestep=dt)
            np.testing.assert_array_equal(result["final_integration_state"],
                                          _integration_state(session["model"], session["data"]))
            self.assertEqual(session["steps"], result["initial_static"]["steps"] + result["steps"])
            self.runs[key] = result
        return self.runs[key]

    def assert_success(self, result):
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(result["classification"], "PHYSICAL_SUCCESS")
        self.assertTrue(result["assessment"].feasible)
        self.assertEqual(result["initial_state"], result["initial_static"]["final_state"])
        self.assertEqual(result["assessment"].diagnostics["actual_integration_state"],
                         result["initial_integration_state"])
        self.assertLess(result["capture_error_m"], .001)
        self.assertGreater(result["capture_margin_m"], 0.)
        self.assertTrue(result["readiness"]["ready"])
        self.assertGreaterEqual(result["readiness"]["duration"], .5 - 1e-10)
        self.assertEqual(len(result["samples"]), result["steps"])
        self.assertLess(result["actuator_utilization_max"], 1.)
        for row in result["samples"]:
            self.assertFalse(np.any(row["external_force_world_N"]) or np.any(row["qfrc_applied"]))
            self.assertLessEqual(row["root_linear_m_s"], .10)
            self.assertLessEqual(row["root_angular_rad_s"], .50)
            self.assertLessEqual(row["joint_max_rad_s"], 1.)
            self.assertTrue(all(foot["supporting"] and not foot["slipping"] for foot in row["feet"].values()))

    def test_baseline_recomputed_native_transfer(self):
        self.assert_success(self.run_case("baseline", .002))

    def test_longer_recomputed_native_transfer_and_geometry_response(self):
        baseline, longer = self.run_case("baseline", .002), self.run_case("longer", .002)
        self.assert_success(longer)
        self.assertNotEqual(baseline["assessment"].preparation_qpos, longer["assessment"].preparation_qpos)
        self.assertNotEqual(baseline["assessment"].motion, longer["assessment"].motion)
        self.assertGreater(np.linalg.norm(np.array(longer["final_state"].qpos[:3]) -
                                         baseline["final_state"].qpos[:3]), .02)

    def test_shorter_native_source_preserved_on_unresolved_rom_search(self):
        result = self.run_case("shorter", .001)
        self.assertTrue(result["initial_static"]["success"])
        self.assertFalse(result["success"])
        self.assertEqual(result["classification"], "ROM_LIMITED_SEARCH")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["initial_state"], result["final_state"])
        self.assertEqual(result["initial_integration_state"], result["final_integration_state"])
        self.assertIsNone(result["final_reference"])
        self.assertFalse(result["released"])
        self.assertTrue(result["assessment"].diagnostics["active_rom_limits"])
        self.assertIn("not a global infeasibility proof", result["reason"])


if __name__ == "__main__":
    unittest.main()
