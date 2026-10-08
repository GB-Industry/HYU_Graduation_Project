"""Native ascending ownership, reference identity and signed body progression."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import ascending_sequence as ascent
from boulder_v1.foot_transfer import FootRequest
from boulder_v1.morphology_envelope import study_profiles
from boulder_v1.schema import Limb
from boulder_v1.static_state import _integration_state
from boulder_v1.transfers import TransferRequest, _request_error
from scripts.transfer_motion_audit import audit_motion
from scripts.validate_whole_body import _evidence_value


class AscendingRequestTests(unittest.TestCase):
    def test_airborne_pitch_is_bounded_and_future_invalid_request_rejects(self):
        _, _, scene, _, _, _ = ascent.make_ascending_fixture()
        contacts = dict(scene.start_configuration)
        request = TransferRequest(Limb.LEFT_FOOT, contacts[Limb.LEFT_FOOT], "foot_target", contacts,
                                  foot_request=FootRequest(Limb.LEFT_FOOT, contacts[Limb.LEFT_FOOT], "foot_target"))
        self.assertIsNone(_request_error(scene, request, .002))
        self.assertEqual(request.foot_request.airborne_pitch_rad, 0.)
        for angle in (True, None, "0", np.nan, np.inf, .6, -.6):
            with self.subTest(angle=angle):
                changed = replace(request, foot_request=replace(request.foot_request, airborne_pitch_rad=angle))
                self.assertEqual(_request_error(scene, changed, .002)[0], "INVALID_REQUEST")
        valid = replace(request, foot_request=replace(request.foot_request, airborne_pitch_rad=-.25))
        self.assertIsNone(_request_error(scene, valid, .002))

    def test_common_step_is_geometry_derived_and_source_is_not_teleported_between_moves(self):
        scenes = []
        for name in ("baseline", "longer", "shorter"):
            model, data, scene, profile, seed, metadata = ascent.make_ascending_fixture(study_profiles()[name])
            self.assertEqual(data.time, 0.)
            self.assertAlmostEqual(metadata["step_height_m"], .082)
            self.assertAlmostEqual(scene.region("foot_target").position[2] - scene.region("left_foot").position[2], .082)
            self.assertFalse(any("foot" in model.equality(i).name for i in range(model.neq)))
            self.assertEqual(model.nu, 25)
            scenes.append(scene)
        self.assertTrue(all(scene == scenes[0] for scene in scenes))


class AscendingNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.cache.clear()

    def run_native(self, name, dt):
        key = name, dt
        if key not in self.cache:
            native_init, native_step, native_transfer = ascent.initialize_static_reference, mujoco.mj_step, ascent.execute_transfer
            session, references = {}, []

            def initialize(model, data, *args, **kwargs):
                self.assertFalse(session, "ascending episode reinitialized")
                reference, manager = native_init(model, data, *args, **kwargs)
                fields = ("body_mass", "body_inertia", "jnt_axis", "jnt_range", "dof_damping", "dof_armature",
                          "geom_size", "geom_friction", "geom_solref", "geom_solimp", "actuator_gear",
                          "actuator_ctrlrange", "eq_type", "eq_data")
                session.update(model=model, data=data, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                               model_arrays={f: getattr(model, f).copy() for f in fields}, steps=0)
                references.append(reference)
                return reference, manager

            def step(model, data):
                self.assertIs(model, session["model"])
                self.assertIs(data, session["data"])
                np.testing.assert_array_equal(data.qpos, session["qpos"])
                np.testing.assert_array_equal(data.qvel, session["qvel"])
                self.assertFalse(np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
                native_step(model, data)
                np.testing.assert_array_equal(data.qfrc_actuator[:6], 0.)
                session.update(qpos=data.qpos.copy(), qvel=data.qvel.copy(), steps=session["steps"] + 1)

            def transfer(model, data, scene, profile, reference, manager, request, **kwargs):
                self.assertIs(reference, references[-1])
                before = _integration_state(model, data)
                result = native_transfer(model, data, scene, profile, reference, manager, request, **kwargs)
                np.testing.assert_array_equal(result["initial_integration_state"], before)
                if result["success"]:
                    references.append(result["final_reference"])
                return result

            with patch.object(ascent, "initialize_static_reference", side_effect=initialize), \
                    patch("mujoco.mj_step", side_effect=step), \
                    patch.object(ascent, "execute_transfer", side_effect=transfer):
                result = ascent.run_ascending_sequence(study_profiles()[name], dt, timing="fast")
            self.assertEqual(session["steps"], result["initial_static"]["steps"] + result["steps"])
            np.testing.assert_array_equal(_integration_state(session["model"], session["data"]), result["final_integration_state"])
            for field, original in session["model_arrays"].items():
                np.testing.assert_array_equal(getattr(session["model"], field), original)
            metrics = audit_motion(session["model"], _evidence_value(result))
            self.cache[key] = result, metrics
        return self.cache[key]

    def assert_native_ascent(self, name, dt):
        result, metrics = self.run_native(name, dt)
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(result["completed_moves"], 3)
        self.assertEqual([m["moving_limb"] for m in result["moves"]], [Limb.RIGHT_HAND, Limb.LEFT_FOOT, Limb.LEFT_HAND])
        self.assertGreater(metrics["whole_case"]["bodies"]["pelvis"]["displacement_world_m"][2], .05)
        self.assertGreater(metrics["whole_case"]["bodies"]["climber_com"]["displacement_world_m"][2], .05)
        self.assertEqual(result["initial_static"]["final_state"], result["initial_state"])
        self.assertNotEqual(result["final_contacts"], dict(result["initial_state"].contact_configuration))
        for index, move in enumerate(result["moves"]):
            self.assertTrue(move["success"], move["reason"])
            self.assertTrue(move["readiness"]["ready"])
            self.assertGreaterEqual(move["readiness"]["duration"], .5 - 1e-10)
            self.assertLess(move["actuator_utilization_max"], 1.)
            self.assertTrue(all(speed <= .01 for speed in move["foot_slip_max_m_s"].values()))
            if index:
                previous = result["moves"][index - 1]
                self.assertIs(move["incoming_reference"], previous["final_reference"])
                self.assertEqual(move["initial_state"], previous["final_state"])
                self.assertEqual(move["initial_integration_state"], previous["final_integration_state"])
            self.assertGreater(metrics["transfers"][index]["moving_effector"]["displacement_world_m"][2], .05)
        foot = result["moves"][1]
        self.assertEqual(foot["request"].foot_request.airborne_pitch_rad, -.25)
        self.assertEqual(foot["request"].whole_body.prepare_s, 4.)
        self.assertEqual(foot["release"]["contact_count"], 0)
        self.assertGreater(foot["acquisition"]["normal_force_N"], 5.)
        self.assertGreaterEqual(foot["acquisition"]["sustained_s"], .1 - 1e-10)
        self.assertEqual([s["selection"].selected_name for s in result["candidates"]], ["default", "conservative"])
        for boundary in metrics["sequence_boundaries"]:
            self.assertTrue(boundary["full_state_equal"] and boundary["integration_state_equal"])

    def test_baseline_fast_2ms_native_ascent(self):
        self.assert_native_ascent("baseline", .002)

    def test_longer_fast_1ms_native_ascent(self):
        self.assert_native_ascent("longer", .001)


if __name__ == "__main__":
    unittest.main()
