"""Opt-in coordinated preparation with unchanged native Stage5 foot ownership."""
import copy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import foot_transfer as ft
from boulder_v1.contact_geometry import Frame
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Limb
from boulder_v1.static_control import execute_static_hold
from boulder_v1.static_state import _integration_state, initialize_static_reference
from boulder_v1.support import FORCE_TOLERANCE, MAX_SLIP_SPEED
from boulder_v1.whole_body_demo import make_whole_body_fixture
from boulder_v1.whole_body_motion import WholeBodyMotion, reference_frames


def prepare(dt):
    model, data, scene, profile, seed = make_whole_body_fixture(dt)
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager,
                                  duration=2., settle=1., score_window=.5, keep_samples=False)
    if not initial["success"]:
        raise AssertionError(initial["reason"])
    return model, data, scene, profile, reference, manager


def motion_for(model, reference):
    root, _ = reference_frames(model, reference.qpos)
    angle = math.radians(4.)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ root.rotation
    return WholeBodyMotion(
        Frame(tuple(np.array(root.position) + [.015, 0., .04]), tuple(map(tuple, rotation))),
        {"waist_yaw": 0., "waist_pitch": .07, "waist_roll": 0.}, prepare_s=4.)


class WholeBodyFootTests(unittest.TestCase):
    def run_native_case(self, dt):
        args = prepare(dt)
        model, data, scene, profile, reference, manager = args
        motion = motion_for(model, reference)
        request = ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target")
        target = scene.region("foot_target")
        source = scene.region("left_foot")
        self.assertAlmostEqual(target.position[0] - source.position[0], -.105)
        self.assertAlmostEqual(target.position[2] - source.position[2], .02)
        self.assertAlmostEqual(float(model.body_subtreemass[model.body("climber_root").id]), 78.3)
        compiled_names = (
            "body_mass", "body_inertia", "body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat",
            "geom_friction", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
            "site_pos", "site_quat", "jnt_range", "jnt_axis", "dof_damping", "dof_armature",
            "actuator_gear", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
            "eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp")
        compiled = {n: getattr(model, n).copy() for n in compiled_names}
        expected = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy()}
        native_step, native_control, native_ik = mujoco.mj_step, ft.compute_pose_control, ft.solve_foot_reference
        callbacks, measured_ik_calls = [], []

        def step(m, d):
            np.testing.assert_array_equal(d.qpos, expected["qpos"])
            np.testing.assert_array_equal(d.qvel, expected["qvel"])
            native_step(m, d)
            expected.update(qpos=d.qpos.copy(), qvel=d.qvel.copy())

        def control(m, d, *a, **kw):
            np.testing.assert_array_equal(d.qpos, expected["qpos"])
            np.testing.assert_array_equal(d.qvel, expected["qvel"])
            self.assertNotIn("kp", kw)
            self.assertNotIn("kd", kw)
            return native_control(m, d, *a, **kw)

        def ik(m, seed, limb, frame, **kw):
            if callbacks:
                np.testing.assert_array_equal(seed, data.qpos)
                measured_ik_calls.append(float(data.time))
            solution = native_ik(m, seed, limb, frame, **kw)
            leg = m.jnt_qposadr[[m.joint("left_" + n).id for n in
                               ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]]
            frozen = np.ones(m.nq, dtype=bool)
            frozen[leg] = False
            np.testing.assert_array_equal(np.array(solution.qpos)[frozen], np.array(seed)[frozen])
            return solution

        def observer(row, display_model, display_data):
            np.testing.assert_array_equal(row["qpos"], data.qpos)
            np.testing.assert_array_equal(display_data.qpos, data.qpos)
            callbacks.append(copy.deepcopy(row))
            row["qpos"][0] = 999.
            display_data.qpos[:] = 999.
            display_data.ctrl[:] = 999.
            display_data.eq_active[:] = False
            display_data.xfrc_applied[:] = 999.
            display_model.body_mass[:] = 999.
            display_model.geom_friction[:] = .0001
            display_model.opt.timestep = 1.

        with patch("mujoco.mj_step", side_effect=step), \
                patch.object(ft, "compute_pose_control", side_effect=control), \
                patch.object(ft, "solve_foot_reference", side_effect=ik):
            result = ft.execute_foot_transfer(*args, request, whole_body=motion, observer=observer)
        self.assertTrue(result["success"], (result["status"], result["reason"], result["guard_failure"]))
        for n, values in compiled.items():
            np.testing.assert_array_equal(getattr(model, n), values, err_msg=n)
        self.assertEqual(float(model.opt.timestep), dt)
        self.assertTrue(measured_ik_calls)
        self.assertEqual(callbacks[-1], result["terminal_observation"])
        self.assertEqual(result["grip_releases"], [])
        self.assertIsNone(result["guard_failure"])
        self.assertEqual(result["phases"], ["SOURCE_STABILIZE", "UNLOAD", "LIFT", "THREE_POINT",
                                            "REACH", "LANDING", "LOAD", "SETTLE"])
        rows = result["samples"]
        unload = [r for r in rows if r["phase"] == "UNLOAD"]
        self.assertAlmostEqual(len(unload) * dt, motion.prepare_s)
        self.assertLess(unload[-1]["feet"]["LEFT_FOOT"]["normal_force"], 5.)
        self.assertGreater(unload[-1]["feet"]["RIGHT_FOOT"]["normal_force"],
                           1.8 * unload[0]["feet"]["RIGHT_FOOT"]["normal_force"])
        self.assertEqual(result["release"]["contact_count"], 0)
        self.assertEqual(result["release"]["normal_force_N"], 0.)
        self.assertGreater(result["release"]["signed_source_geom_distance_m"], 0.)
        self.assertGreater(result["touchdown"]["normal_force_N"], 0.)
        self.assertGreater(result["first_support"]["normal_force_N"], 5.)
        self.assertGreaterEqual(result["acquisition"]["sustained_s"], .1 - 1e-12)
        self.assertAlmostEqual(result["load_duration_s"], request.load_s)
        self.assertGreater(result["terminal_observation"]["feet"]["LEFT_FOOT"]["normal_force"], 300.)
        self.assertTrue(result["readiness"]["ready"])
        self.assertGreaterEqual(result["readiness"]["duration"], .5 - 1e-12)
        self.assertEqual(manager.contact_configuration(), {**reference.contact_intent, Limb.LEFT_FOOT: "foot_target"})
        self.assertFalse(any("foot" in model.equality(i).name for i in range(model.neq)))
        self.assertLess(result["support_admission"]["preparation_motor_utilization_max"], 1.)
        self.assertLess(result["support_admission"]["motor_utilization_max"], 1.)
        self.assertLess(result["actuator_utilization_max"], 1.)

        roots = np.array([r["qpos"][:3] for r in rows])
        displacement = np.linalg.norm(roots - result["initial_state"].qpos[:3], axis=1)
        self.assertGreaterEqual(displacement[-1], .03)
        self.assertGreaterEqual(float(displacement.max()), .03)
        self.assertGreaterEqual(roots[-1, 2] - result["initial_state"].qpos[2], .03)
        actual = np.array([r["qpos"] for r in unload])
        for name in ("waist_pitch", "left_knee", "right_knee", "left_hip_pitch", "right_hip_pitch",
                     "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_pitch", "right_shoulder_pitch"):
            self.assertGreater(float(np.ptp(actual[:, int(model.joint(name).qposadr[0])])), .01, name)
        joints = model.actuator_trnid[:, 0]
        qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        previous, velocity = np.array(reference.qpos), np.zeros(model.nv)
        support_q = [int(model.joint(n).qposadr[0]) for n in reference.target_pose if not n.startswith("left_hip")
                     and n not in ("left_knee", "left_ankle_pitch", "left_ankle_roll")]
        support_ref = np.array(next(r for r in rows if r["phase"] == "THREE_POINT")["q_ref"])
        for r in rows:
            self.assertFalse(np.any(r["external_force_world_N"]))
            self.assertFalse(np.any(r["qfrc_applied"]))
            self.assertEqual(r["eq_active"], result["terminal_observation"]["eq_active"])
            self.assertTrue(r["feet"]["RIGHT_FOOT"]["supporting"])
            self.assertFalse(r["feet"]["RIGHT_FOOT"]["slipping"])
            for hand in r["hands"].values():
                self.assertTrue(hand["active"] and hand["valid"])
                self.assertLessEqual(hand["load"], hand["capacity"])
            for foot in r["feet"].values():
                self.assertTrue(foot["measurement_valid"])
                for contact in foot["contacts"]:
                    if contact["normal_force"] > FORCE_TOLERANCE:
                        self.assertTrue(contact["admissible"])
                        self.assertLessEqual(contact["tangential_speed"], MAX_SLIP_SPEED)
                        self.assertLessEqual(contact["friction_utilization"], 1. + 1e-8)
            self.assertLessEqual(r["root_linear_m_s"], .10)
            self.assertLessEqual(r["root_angular_rad_s"], .50)
            self.assertLessEqual(r["joint_max_rad_s"], 1.)
            q, qd = np.array(r["q_ref"]), np.array(r["qd_ref"])
            validate_reference_pose(model, q)
            self.assertLessEqual(np.max(np.abs(qd[vi])), .5 + 1e-10)
            self.assertLessEqual(np.max(np.abs(qd[vi] - velocity[vi])), 2. * dt + 1e-10)
            np.testing.assert_allclose((q[qi] - previous[qi]) / dt, qd[vi], atol=1e-10, rtol=0)
            previous, velocity = q, qd
            if r["phase"] in ("THREE_POINT", "REACH", "LANDING", "LOAD"):
                np.testing.assert_array_equal(q[support_q], support_ref[support_q])
            if r["released"]:
                self.assertEqual(r["source_contact"]["contact_count"], 0)
                self.assertEqual(r["source_contact"]["normal_force_N"], 0.)
            if r["phase"] in ("THREE_POINT", "REACH"):
                self.assertFalse(r["feet"]["LEFT_FOOT"]["contacting"])
            if r["acquisition"] is not None:
                self.assertEqual(r["feet"]["LEFT_FOOT"]["support_surfaces"], ("geom_foot_target",))
        np.testing.assert_allclose(result["terminal_observation"]["q_ref"], result["final_reference"].qpos,
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(result["terminal_observation"]["qd_ref"], 0., atol=1e-8, rtol=0)
        print(f"wholebody foot dt={dt}: steps={result['steps']} duration={result['duration_s']:.3f}s "
              f"pelvis_net/max_mm={displacement[-1] * 1000:.3f}/{displacement.max() * 1000:.3f} "
              f"unload_Fn_N={unload[-1]['feet']['LEFT_FOOT']['normal_force']:.3f}/"
              f"{unload[-1]['feet']['RIGHT_FOOT']['normal_force']:.3f} "
              f"final_Fn_N={result['terminal_observation']['feet']['LEFT_FOOT']['normal_force']:.3f} "
              f"release/touch/acquire_s={result['release_time_s']:.3f}/"
              f"{result['touchdown']['time_s']:.3f}/{result['acquisition_time_s']:.3f} "
              f"motor_max={result['actuator_utilization_max']:.6f} "
              f"slip_max={result['foot_slip_max_m_s']} hands_max={result['hand_load_max_N']} "
              f"tracking_max_m={result['max_tracking_error_m']:.6f}", flush=True)

    def test_native_whole_body_foot_2ms(self):
        self.run_native_case(.002)

    def test_native_whole_body_foot_1ms(self):
        self.run_native_case(.001)

    def test_invalid_whole_body_goals_reject_atomically(self):
        args = prepare(.002)
        model, data, _, _, reference, manager = args
        motion = motion_for(model, reference)
        invalid = (
            replace(motion, root_target=Frame((math.nan, 0., 1.), motion.root_target.rotation)),
            replace(motion, root_target=Frame(motion.root_target.position, ((1., 0., 0.),) * 3)),
            replace(motion, waist_target={**motion.waist_target, "waist_pitch": 10.}),
            replace(motion, root_target=Frame(tuple(np.array(motion.root_target.position) + [0., 0., 2.]),
                                             motion.root_target.rotation)))
        before = _integration_state(model, data)
        contacts = manager.contact_configuration()
        releases, captures = copy.deepcopy(manager.releases), copy.deepcopy(manager.capture_events)
        for goal in invalid:
            with self.subTest(goal=goal), \
                    patch.object(ft, "compute_pose_control", side_effect=AssertionError("invalid goal commanded")), \
                    patch("mujoco.mj_step", side_effect=AssertionError("invalid goal integrated")):
                result = ft.execute_foot_transfer(*args, ft.FootRequest(Limb.LEFT_FOOT, "left_foot", "foot_target"),
                                                  whole_body=goal)
            self.assertEqual(result["status"], "REACH_INFEASIBLE", result["reason"])
            self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
            self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
            self.assertIsNone(result["final_reference"])
            np.testing.assert_array_equal(_integration_state(model, data), before)
            self.assertEqual(manager.contact_configuration(), contacts)
            self.assertEqual(manager.releases, releases)
            self.assertEqual(manager.capture_events, captures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
