"""Sanity checks for the mannequin climber.  Run: python -m unittest discover tests"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from climber import BASELINE, PRESETS, actuated_joints, build_model, climb_pose, t_pose
from climber.measure import body_stats


class ModelTest(unittest.TestCase):
    def test_all_presets_compile(self):
        for p in PRESETS.values():
            m = build_model(p)
            self.assertEqual(m.nu, 25)
            self.assertEqual(m.nq, 7 + 25)

    def test_total_mass(self):
        m = build_model(BASELINE)
        root = m.body("pelvis").id
        climber_mass = sum(m.body_mass[b] for b in range(m.nbody) if m.body_rootid[b] == root)
        self.assertAlmostEqual(float(climber_mass), BASELINE.total_mass, places=6)

    def test_baseline_is_175cm_with_human_arm_span(self):
        stats = body_stats(build_model(BASELINE))
        self.assertAlmostEqual(stats["height"], 1.75, delta=0.01)
        self.assertAlmostEqual(stats["arm_span"] / stats["height"], 1.0, delta=0.03)
        self.assertAlmostEqual(stats["hip_joint_height"] / stats["height"], 0.53, delta=0.02)

    def test_total_mass_follows_profile(self):
        from dataclasses import replace
        m = build_model(replace(BASELINE, total_mass=60.0))
        root = m.body("pelvis").id
        self.assertAlmostEqual(sum(m.body_mass[b] for b in range(m.nbody) if m.body_rootid[b] == root), 60.0, places=6)

    def test_hip_joints_use_profile_width(self):
        m = build_model(BASELINE)
        x = abs(m.body("left_thigh").pos[0])
        self.assertAlmostEqual(x, 0.85 * BASELINE.hip_width / 2, places=6)

    def test_visual_balls_do_not_collide(self):
        m = build_model(BASELINE)
        root = m.body("pelvis").id
        climber = [i for i in range(m.ngeom) if m.body_rootid[m.geom_bodyid[i]] == root]
        balls = [i for i in climber if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_size[i][0] < 0.05]
        self.assertTrue(balls)
        for i in balls:
            self.assertEqual((m.geom_contype[i], m.geom_conaffinity[i]), (0, 0))
        for i in climber:  # segments touch the world, never each other
            self.assertEqual(m.geom_conaffinity[i] & m.geom_contype[i], 0)

    def test_actuators_cover_every_hinge(self):
        m = build_model(BASELINE)
        hinges = {m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        self.assertEqual(hinges, set(actuated_joints()))


class PoseAndSimTest(unittest.TestCase):
    def test_climb_pose_reaches_holds(self):
        m = build_model(BASELINE)
        for p in PRESETS.values():
            _, info = climb_pose(build_model(p))
            for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                self.assertLess(info["residual"][limb], 0.02, f"{p.name} {limb}")
            for part, gap in info["wall_gap"].items():  # torso hugs the wall, not 0.5 m out
                self.assertLess(gap, 0.40, f"{p.name} {part} is {gap:.2f} m from the wall")

    def test_t_pose_stands_on_floor(self):
        m = build_model(BASELINE)
        d = mujoco.MjData(m)
        d.qpos[:] = t_pose(m)
        mujoco.mj_forward(m, d)
        hand, shoulder = d.xpos[m.body("left_hand").id], d.xpos[m.body("left_upper_arm").id]
        self.assertGreater(abs(hand[0] - shoulder[0]), 0.5)  # arm out sideways (roll limit 80 deg)
        feet_z = min(d.geom_xpos[i][2] for i in range(m.ngeom)
                     if m.geom_bodyid[i] == m.body("left_foot").id and m.geom_contype[i])
        self.assertLess(feet_z, 0.06)

    def test_gripped_hang_is_stable(self):
        """Hands welded to holds, zero torque, 2 s: must not blow up or fall off the wall."""
        m = build_model(BASELINE)
        d = mujoco.MjData(m)
        q, info = climb_pose(m)
        d.qpos[:] = q
        for side, hold in zip(("left", "right"), info["hands"]):
            d.eq_active[m.equality(f"grasp_{side}_{hold}").id] = 1
        for _ in range(1000):
            mujoco.mj_step(m, d)
        self.assertTrue(np.all(np.isfinite(d.qpos)))
        self.assertGreater(d.xpos[m.body("pelvis").id][2], 0.3)
        self.assertEqual(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number, 0)


if __name__ == "__main__":
    unittest.main()
