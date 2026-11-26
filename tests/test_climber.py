"""Sanity checks for the mannequin climber.  Run: python -m unittest discover tests

[한국어 설명]
기준 에이전트가 약속한 사양을 지키는지 자동으로 확인하는 테스트 10개.
에이전트 코드를 고친 뒤에는 꼭 실행해서 전부 OK인지 확인할 것.
  - 모델 생성: 세 체형 모두 관절·모터 27개
  - 질량: 전체 70kg, total_mass를 바꾸면 그대로 따라감
  - 체형: 키 1.75m, 팔 너비 ≈ 키, 고관절 높이 ≈ 키의 53%
  - 충돌: 관절 볼은 충돌 없음, 몸끼리 충돌 없음
  - 자세: 벽 자세에서 손발이 홀드에 닿고 몸통이 벽에서 40cm 이내, T자 자세는 바닥에 섬
  - 물리: 힘 0으로 2초 돌려도 시뮬레이션이 터지지 않음 / 자세 유지 제어기로 3초간 스스로 매달림
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from climber import (BASELINE, PRESETS, ContactModel, PoseHold, actuated_joints, build_model, climb_pose,
                     t_pose, test_wall)
from climber.measure import body_stats


class ModelTest(unittest.TestCase):
    def test_all_presets_compile(self):
        for p in PRESETS.values():
            m = build_model(p)
            self.assertEqual(m.nu, 27)
            self.assertEqual(m.nq, 7 + 27)

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
    def test_climb_pose_puts_palms_and_soles_on_holds(self):
        """175cm 몸에 맞는 테스트 벽에서: 손바닥·발바닥이 홀드 표면에 닿고, 방향도 홀드를 향한다."""
        for p in PRESETS.values():
            _, info = climb_pose(build_model(p, test_wall()), feet=("LF", "RF"), hand_options=(("LH", "RH"),))
            r = info["residual"]
            for limb in ("left_palm", "right_palm", "left_sole", "right_sole"):
                self.assertLess(r[limb], 0.03, f"{p.name} {limb} {r[limb]:.3f} m")
                self.assertLess(r[f"{limb}_deg"], 10.0, f"{p.name} {limb} {r[limb + '_deg']:.1f} deg")
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

    def test_sim_from_climb_pose_stays_finite(self):
        """벽 자세에서 그립 규칙을 켜고 힘 0으로 2초: 시뮬레이션이 터지지 않아야 한다.
        (자세를 버티는 제어기는 아직 없으므로 몸이 처지거나 손을 놓치는 것은 정상)"""
        m = build_model(BASELINE, test_wall())
        d = mujoco.MjData(m)
        d.qpos[:], _ = climb_pose(m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
        mujoco.mj_forward(m, d)
        cm = ContactModel(m)
        cm.update(d)
        for _ in range(1000):
            mujoco.mj_step(m, d)
            cm.update(d)
        self.assertTrue(np.all(np.isfinite(d.qpos)))
        self.assertEqual(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number, 0)


    def test_hangs_on_its_own_with_pose_hold(self):
        """시작 자세 유지 제어기(PoseHold)만으로 3초 동안 양손으로 매달려 버틴다 (보조 장치 없음)."""
        for p in PRESETS.values():
            m = build_model(p, test_wall())
            d = mujoco.MjData(m)
            q0, _ = climb_pose(m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
            d.qpos[:] = q0
            mujoco.mj_forward(m, d)
            cm, ctrl = ContactModel(m, grip_capacity=p.grip_capacity), PoseHold(m, q0)
            cm.update(d)
            for _ in range(1500):
                d.ctrl[:] = ctrl(d)
                mujoco.mj_step(m, d)
                cm.update(d)
                self.assertIsNotNone(cm.hands["left"].hold, f"{p.name} left: {cm.hands['left'].last_release}")
                self.assertIsNotNone(cm.hands["right"].hold, f"{p.name} right: {cm.hands['right'].last_release}")

if __name__ == "__main__":
    unittest.main()
