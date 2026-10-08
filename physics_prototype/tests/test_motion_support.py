"""Estimated force points must belong to actual native contact geometry."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1.foot_transfer import make_foot_transfer_fixture
from boulder_v1.motion_support import estimate_support_torques
from boulder_v1.schema import Limb
from boulder_v1.static_state import _foot_residual, initialize_static_reference, _integration_state
from boulder_v1.contact_geometry import canonical_geometry


class MotionSupportTests(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.scene, self.profile, seed = make_foot_transfer_fixture()
        self.reference, self.manager = initialize_static_reference(
            self.model, self.data, self.scene, self.profile, seed, self.scene.start_configuration)
        self.scratch = mujoco.MjData(self.model)
        self.scratch.qpos[:] = seed
        mujoco.mj_forward(self.model, self.scratch)
        self.contacts = dict(self.reference.contact_intent)
        self.points = {limb: _foot_residual(self.model, self.scratch, limb, self.scene.region(self.contacts[limb]),
                                          canonical_geometry(self.scene.region(self.contacts[limb]))).support_point_world
                       for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)}

    def estimate(self, **options):
        return estimate_support_torques(self.model, self.scene, self.profile, self.reference.qpos,
                                        self.contacts, **options)

    def test_verified_patch_centers_preserve_original_allocation(self):
        original = self.estimate()
        measured = self.estimate(foot_points=self.points)
        self.assertEqual(original, measured)

    def test_points_outside_shoe_or_surface_rejected_without_live_mutation(self):
        before = _integration_state(self.model, self.data)
        for delta in ((1., 0., 0.), (0., 0., .01), (0., .2, 0.)):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "outside the real"):
                self.estimate(foot_points={Limb.LEFT_FOOT: np.array(self.points[Limb.LEFT_FOOT]) + delta})
        np.testing.assert_array_equal(before, _integration_state(self.model, self.data))

    def test_points_and_contact_keys_are_typed_and_finite(self):
        for points in ({Limb.RIGHT_HAND: (0., 0., 0.)}, {Limb.LEFT_FOOT: (0., np.nan, 0.)}, [],
                       {Limb.LEFT_FOOT.value: self.points[Limb.LEFT_FOOT]}):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.estimate(foot_points=points)

    def test_one_foot_two_hands_are_estimates_not_injected_forces(self):
        supports = {l: h for l, h in self.contacts.items() if l != Limb.LEFT_FOOT}
        before = _integration_state(self.model, self.data)
        with patch("mujoco.mj_step", side_effect=AssertionError("allocation cannot integrate")):
            result = estimate_support_torques(self.model, self.scene, self.profile, self.reference.qpos,
                                              supports, foot_points={Limb.RIGHT_FOOT: self.points[Limb.RIGHT_FOOT]})
        self.assertEqual(len(result["forces_world_N"]), 3)
        self.assertLess(result["root_balance_residual"], 1e-8)
        self.assertGreater(result["forces_world_N"][0][2], 5.)
        np.testing.assert_array_equal(before, _integration_state(self.model, self.data))
        self.assertFalse(np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied))


if __name__ == "__main__":
    unittest.main()
