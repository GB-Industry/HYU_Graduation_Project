"""Hand grip / foot support rules (climber/contact.py).  Run: python -m unittest discover tests

[한국어 설명]
손 잡기·놓치기 규칙과 발 지지 판정을 확인하는 테스트.
자세를 버티는 제어기가 아직 없으므로, 손 규칙만 따로 보기 위해 골반을 벽 앞에 '테스트용 고정(weld)'으로 묶고
손에 정해진 힘을 직접 가해서 확인한다.
  - 손바닥이 홀드를 정면으로 향하면 잡힌다 (20° 이내)
  - 손바닥이 돌아가 있으면 닿아도 안 잡힌다
  - 잡은 상태의 하중은 실제 팔 무게 수준이고, 손이 홀드를 파고들지 않는다
  - 홀드가 버틸 수 있는 방향(저그: 아래)으로 당기면 계속 잡고 있다
  - 그립 한계를 넘으면 놓친다 / 홀드에서 떼어 내는 방향으로 당기면(버틸 수 있는 방향 밖) 놓친다
  - 놓친 손은 잠깐(0.3초) 다시 못 잡는다
  - 바닥에 서 있으면 양발이 '지지 중'
  - 악력 피로: 매달리면 줄고, 쉬면 회복되고, 30% 밑으로는 안 내려간다
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from climber import BASELINE, ContactModel, PoseHold, build_model, build_xml, climb_pose, t_pose, test_wall
from climber.contact import fatigue_step

PIN = '<weld name="test_pin" body1="pelvis" active="false" solref="0.004 1" solimp="0.99 0.999 0.001"/>'


class GripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mujoco.MjModel.from_xml_string(
            build_xml(BASELINE, test_wall()).replace("</equality>", PIN + "</equality>"))
        cls.q0, _ = climb_pose(cls.m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
        cls.hand = cls.m.body("left_hand").id

    def start(self, q=None, capacity=850.0):
        """자세를 놓고, 골반을 지금 위치에 고정한 뒤, 그립 규칙을 한 번 적용."""
        m = self.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.q0 if q is None else q
        mujoco.mj_forward(m, d)
        pin, pb = m.equality("test_pin").id, m.body("pelvis").id
        p, qw, R = d.xpos[pb].copy(), d.xquat[pb].copy(), d.xmat[pb].reshape(3, 3)
        m.eq_data[pin][0:3] = 0
        m.eq_data[pin][3:6] = R.T @ (-p)                       # weld: 월드 원점을 골반 좌표로
        m.eq_data[pin][6:10] = [qw[0], -qw[1], -qw[2], -qw[3]]
        d.eq_active[pin] = 1
        cm = ContactModel(m, grip_capacity=capacity)
        cm.update(d)
        return d, cm

    def step(self, d, cm, n, push=None):
        peak = 0.0
        for _ in range(n):
            d.xfrc_applied[:] = 0
            if push is not None:
                d.xfrc_applied[self.hand, :3] = push
            mujoco.mj_step(self.m, d)
            cm.update(d)
            if cm.hands["left"].hold:
                peak = max(peak, float(np.linalg.norm(cm.hands["left"].load)))
        return peak

    def test_grip_forms_when_palm_faces_hold(self):
        _, cm = self.start()
        for s, hold in (("left", "LH"), ("right", "RH")):
            self.assertEqual(cm.hands[s].hold, hold)
            self.assertLessEqual(cm.hands[s].palm_angle_deg, 20.0)

    def test_no_grip_when_palm_turned_away(self):
        q = self.q0.copy()
        q[self.m.jnt_qposadr[self.m.joint("left_forearm_twist").id]] -= np.radians(50)
        _, cm = self.start(q)
        h = cm.hands["left"]
        self.assertEqual(h.touching, "LH")           # 닿아는 있지만
        self.assertGreater(h.palm_angle_deg, 20.0)   # 손바닥이 홀드를 향하지 않아서
        self.assertIsNone(h.hold)                    # 안 잡힌다

    def test_resting_grip_load_is_arm_weight_and_no_penetration(self):
        d, cm = self.start()
        peak = self.step(d, cm, 10)
        self.assertEqual(cm.hands["left"].hold, "LH")
        self.assertLess(peak, 100.0)
        hand = cm.hand_geoms["left"]
        pen = max([-c.dist for c in d.contact[:d.ncon] if c.geom1 in hand or c.geom2 in hand] or [0.0])
        self.assertLess(pen, 0.003)

    def test_pulling_down_on_a_jug_keeps_grip(self):
        d, cm = self.start()
        peak = self.step(d, cm, 50, push=[0, 0.1, -300])
        self.assertEqual(cm.hands["left"].hold, "LH")
        self.assertGreater(peak, 250.0)

    def test_grip_limit_releases(self):
        d, cm = self.start(capacity=60.0)
        self.step(d, cm, 50, push=[0, 0.1, -300])
        self.assertIsNone(cm.hands["left"].hold)
        self.assertTrue(cm.hands["left"].last_release.startswith("grip limit"))

    def test_pulling_off_the_hold_releases_by_pull_direction(self):
        d, cm = self.start()
        self.step(d, cm, 50, push=[0, -300, 50])
        self.assertIsNone(cm.hands["left"].hold)
        self.assertTrue(cm.hands["left"].last_release.startswith("outside pull direction"))

    def test_released_hand_waits_before_regripping(self):
        d, cm = self.start()
        cm.release(d, "left", "test")
        self.step(d, cm, 20)
        self.assertIsNone(cm.hands["left"].hold)


class FootSupportTest(unittest.TestCase):
    def test_standing_feet_are_supported(self):
        m = build_model(BASELINE)
        d = mujoco.MjData(m)
        d.qpos[:] = t_pose(m)
        mujoco.mj_forward(m, d)
        cm = ContactModel(m)
        for _ in range(5):
            mujoco.mj_step(m, d)
            cm.update(d)
        for s in ("left", "right"):
            f = cm.feet[s]
            self.assertTrue(f.supported, f)
            self.assertEqual(f.surface, "floor")
            self.assertLess(f.sole_angle_deg, 20.0)



class GripFatigueTest(unittest.TestCase):
    """악력 피로 규칙 (값: 절반 60초, 회복 30초, 최저 30%)."""
    def run_for(self, seconds, capacity, load, gripping, base=850.0, dt=0.002, load_ratio=None):
        for _ in range(int(seconds / dt)):
            current_load = capacity * load_ratio if load_ratio is not None else load
            capacity = fatigue_step(capacity, base, current_load, dt, gripping)
        return capacity

    def test_full_load_halves_strength_above_floor_in_60s(self):
        c = self.run_for(60, 850.0, None, True, load_ratio=1.0)   # 악력 한계로 계속 매달림
        floor = 850.0 * 0.3
        self.assertAlmostEqual((c - floor) / (850.0 - floor), 0.5, delta=0.02)

    def test_light_load_tires_slower(self):
        heavy = self.run_for(30, 850.0, None, True, load_ratio=1.0)
        light = self.run_for(30, 850.0, None, True, load_ratio=0.25)
        self.assertGreater(light, heavy)

    def test_never_below_30_percent(self):
        c = self.run_for(600, 850.0, None, True, load_ratio=1.0)
        self.assertGreaterEqual(c, 850.0 * 0.3 - 1e-6)
        self.assertLess(c, 850.0 * 0.3 + 5.0)

    def test_resting_hand_recovers_95_percent_in_30s(self):
        c = self.run_for(30, 400.0, 0.0, False)
        self.assertGreater((c - 400.0) / (850.0 - 400.0), 0.94)

    def test_hanging_climber_gets_tired(self):
        m = build_model(BASELINE, test_wall())
        d = mujoco.MjData(m)
        q0, _ = climb_pose(m, feet=("LF", "RF"), hand_options=(("LH", "RH"),))
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        cm, ctrl = ContactModel(m, profile=BASELINE), PoseHold(m, q0)
        cm.update(d)
        for _ in range(1000):  # 2초
            d.ctrl[:] = ctrl(d)
            mujoco.mj_step(m, d)
            cm.update(d)
        for s in ("left", "right"):
            self.assertIsNotNone(cm.hands[s].hold)
            self.assertLess(cm.hands[s].capacity, BASELINE.grip_capacity - 1.0)

if __name__ == "__main__":
    unittest.main()
