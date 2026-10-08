import unittest
from dataclasses import replace
import math

from boulder_v1.contact import (
    CAPTURE_DISTANCE, CAPTURE_ORIENTATION, CAPTURE_SPEED, GripController,
    can_attach, contact_allowed, effective_grip_capacity, orientation_compatibility,
)
from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import Affordance, ClimberProfile, Limb


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_synthetic_scene()
        self.h3 = self.scene.region("H3")
        self.target = canonical_geometry(self.h3).hand_frame.position
        self.profile = ClimberProfile(name="test", grip_capacity=500.0)

    def test_limb_affordance_compatibility(self):
        self.assertTrue(contact_allowed(Limb.LEFT_HAND, Affordance.GRASP, self.h3))
        self.assertFalse(contact_allowed(Limb.LEFT_FOOT, Affordance.GRASP, self.h3))

    def test_orientation_penalizes_wrong_facing_hand(self):
        good = orientation_compatibility((0.0, 1.0, 0.0), self.h3.normal)
        bad = orientation_compatibility((0.0, -1.0, 0.0), self.h3.normal)
        self.assertAlmostEqual(good, 1.0)
        self.assertAlmostEqual(bad, 0.0)

    def test_grip_breaks_above_capacity(self):
        ctrl = GripController()
        below = ctrl.evaluate(self.profile, self.h3, (0.0, 1.0, 0.0), required_load=250.0)
        above = ctrl.evaluate(self.profile, self.h3, (0.0, 1.0, 0.0), required_load=900.0)
        self.assertTrue(below.maintain)
        self.assertFalse(above.maintain)
        self.assertGreater(above.utilization, 1.0)

    def test_can_attach_eligible(self):
        # Acquisition uses the canonical surface anchor, not the hold center.
        ok = can_attach(
            limb=Limb.LEFT_HAND,
            region=self.h3,
            effector_pos=self.target,
            effector_normal=(0.0, 1.0, 0.0),
            affordance=Affordance.GRASP,
        )
        self.assertTrue(ok)

    def test_can_attach_rejects_excess_distance(self):
        # Position far from hold (e.g. 0.5m away)
        far_pos = (self.target[0], self.target[1] - 0.5, self.target[2])
        ok = can_attach(
            limb=Limb.LEFT_HAND,
            region=self.h3,
            effector_pos=far_pos,
            effector_normal=(0.0, 1.0, 0.0),
            affordance=Affordance.GRASP,
        )
        self.assertFalse(ok)

    def test_can_attach_rejects_incompatible_affordance(self):
        # Foot attempting to GRASP
        ok = can_attach(
            limb=Limb.LEFT_FOOT,
            region=self.h3,
            effector_pos=self.target,
            effector_normal=(0.0, 1.0, 0.0),
            affordance=Affordance.GRASP,
        )
        self.assertFalse(ok)

    def test_can_attach_rejects_incompatible_orientation(self):
        # Normal facing directly away from hold normal
        opposing_normal = (0.0, -1.0, 0.0)
        ok = can_attach(
            limb=Limb.LEFT_HAND,
            region=self.h3,
            effector_pos=self.target,
            effector_normal=opposing_normal,
            affordance=Affordance.GRASP,
        )
        self.assertFalse(ok)

    def test_physical_capture_distance_speed_and_orientation_bounds(self):
        for distance, speed, angle, expected in (
            (0., 0., 0., True), (.001, .05, 0., True),
            (.00101, 0., 0., False), (0., .05001, 0., False),
            (0., 0., 29., True), (0., 0., 30., True), (0., 0., 31., False),
            (0., math.hypot(.04, .04), 0., False),
        ):
            with self.subTest(distance=distance, speed=speed, angle=angle):
                radians = math.radians(angle)
                self.assertEqual(can_attach(
                    Limb.LEFT_HAND, self.h3,
                    (self.target[0] + distance, self.target[1], self.target[2]),
                    (math.sin(radians), math.cos(radians), 0.),
                    relative_velocity=(speed, 0., 0.),
                ), expected)
        self.assertFalse(can_attach(Limb.LEFT_HAND, self.h3, self.h3.position, (0., 1., 0.)))
        for kwargs in ({"max_distance": .15}, {"max_speed": .1}, {"min_orientation": .2}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(can_attach(Limb.LEFT_HAND, self.h3, self.target, (0., 1., 0.), **kwargs))
        self.assertEqual((CAPTURE_DISTANCE, CAPTURE_SPEED), (.001, .05))
        self.assertAlmostEqual(CAPTURE_ORIENTATION, math.cos(math.radians(30)))

    def test_capacity_is_quality_scaled_not_orientation_or_friction_scaled(self):
        for friction in (.1, 2.):
            region = replace(self.h3, friction=friction, grip_quality=.8)
            for normal in ((0., 1., 0.), (0., -1., 0.)):
                with self.subTest(friction=friction, normal=normal):
                    self.assertEqual(effective_grip_capacity(self.profile, region, normal), 400.)
                    self.assertTrue(GripController().evaluate(self.profile, region, normal, 400.).maintain)
                    self.assertFalse(GripController().evaluate(self.profile, region, normal, 400.01).maintain)


if __name__ == "__main__":
    unittest.main()
