import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from boulder_v1.contact_benchmarks import make_hand_fixture, make_mixed_fixture
from boulder_v1.contact_geometry import ContactMode
from boulder_v1.grasp import (
    AttachmentStateError,
    GraspManager,
    InitialContactError,
    initialize_episode,
    simulate_static_stance,
)
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import compile_model, mujoco_available
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import ClimberProfile, ContactRegion, Limb
from foundation_fixture_control import archived_normalized_pose_control


class GraspTests(unittest.TestCase):
    """Legacy registry fixtures are debug; physical checks own canonical fixtures."""

    def setUp(self):
        if not mujoco_available():
            self.skipTest("MuJoCo is not available")
        import mujoco

        self.scene = make_synthetic_scene()
        self.base_profile = ClimberProfile(name="base")
        self.xml = build_mjcf(self.scene, self.base_profile, contact_mode=ContactMode.IDEALIZED_DEBUG)
        self.model = compile_model(self.xml)
        self.data = mujoco.MjData(self.model)
        self.h3 = self.scene.region("H3")
        self.h4 = self.scene.region("H4")

    def test_grasp_manager_can_attach_checks_distance(self):
        gm = GraspManager(self.model, self.data, self.scene)

        # In default upright pose, left hand is far from H3
        self.assertFalse(gm.can_attach(Limb.LEFT_HAND, self.h3))

        # Feet cannot attach to GRASP holds
        self.assertFalse(gm.can_attach(Limb.LEFT_FOOT, self.h3))

    def test_grasp_manager_attach_and_detach_without_recompilation(self):
        model_id = id(self.model)
        gm = GraspManager(self.model, self.data, self.scene)

        self.assertFalse(gm.is_attached(Limb.LEFT_HAND))
        self.assertIsNone(gm.active_attachment(Limb.LEFT_HAND))

        # Attach left hand to H3 (force=True to bypass distance check for direct unit testing)
        attached = gm.attach(Limb.LEFT_HAND, self.h3, force=True)
        self.assertTrue(attached)
        self.assertTrue(gm.is_attached(Limb.LEFT_HAND))

        active_region = gm.active_attachment(Limb.LEFT_HAND)
        self.assertIsInstance(active_region, ContactRegion)
        self.assertEqual(active_region.id, "H3")

        eq_id = gm._attachments[Limb.LEFT_HAND].eq_id
        self.assertEqual(self.data.eq_active[eq_id], 1)

        # Model is NOT recompiled
        self.assertEqual(id(self.model), model_id)

        # Detach
        detached = gm.detach(Limb.LEFT_HAND)
        self.assertTrue(detached)
        self.assertFalse(gm.is_attached(Limb.LEFT_HAND))
        self.assertIsNone(gm.active_attachment(Limb.LEFT_HAND))
        self.assertEqual(self.data.eq_active[eq_id], 0)
        self.assertEqual(id(self.model), model_id)

    def test_grasp_manager_detach_all(self):
        gm = GraspManager(self.model, self.data, self.scene)
        gm.attach(Limb.LEFT_HAND, self.h3, force=True)
        gm.attach(Limb.RIGHT_HAND, self.h4, force=True)

        self.assertTrue(gm.is_attached(Limb.LEFT_HAND))
        self.assertTrue(gm.is_attached(Limb.RIGHT_HAND))

        gm.detach_all()
        self.assertFalse(gm.is_attached(Limb.LEFT_HAND))
        self.assertFalse(gm.is_attached(Limb.RIGHT_HAND))

    def test_get_grasp_load_measurement(self):
        import mujoco

        fixture = make_hand_fixture(capacity=150., gap=0., speed=0.)
        model, data, gm = fixture.model, fixture.data, fixture.manager
        self.assertEqual(gm.mode, ContactMode.PHYSICAL)
        self.assertEqual(gm.profile, fixture.profile)
        self.assertEqual(gm.get_grasp_load(Limb.LEFT_HAND), 0.)
        self.assertTrue(gm.attach(Limb.LEFT_HAND, fixture.scene.region("left_hand")))
        gm.synchronize_from_live()
        data.xfrc_applied[model.body("left_hand").id, :3] = [-60., -80., 0.]
        for _ in range(50):
            gm.evaluate_and_update()
            mujoco.mj_step(model, data)
            gm.evaluate_and_update(applied_data=data)
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        mujoco.mj_forward(model, scratch)
        eq = model.equality("grasp_left_hand_left_hand").id
        rows = np.flatnonzero((scratch.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                              & (scratch.efc_id == eq))
        self.assertEqual(len(rows), 3)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacobian, None, model.site("left_hand_site").id)
        np.testing.assert_allclose(scratch.efc_J.reshape(scratch.nefc, model.nv)[rows], jacobian, rtol=0, atol=1e-10)
        np.testing.assert_allclose(gm.get_grasp_force(Limb.LEFT_HAND), scratch.efc_force[rows], rtol=0, atol=1e-10)
        self.assertAlmostEqual(gm.get_grasp_load(Limb.LEFT_HAND), np.linalg.norm(scratch.efc_force[rows]), places=10)
        self.assertGreaterEqual(np.count_nonzero(np.abs(scratch.efc_force[rows]) > 1.), 2)
        self.assertLessEqual(gm.get_grasp_load(Limb.LEFT_HAND), fixture.profile.grip_capacity)
        for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
            with self.assertRaisesRegex(ValueError, "Foot collision loads"):
                gm.get_grasp_load(limb)
            with self.assertRaises(ValueError):
                gm.get_grasp_force(limb)

    def test_profile_conditioned_breakable_grip(self):
        """Verify identical real physical load causes weak grip to break and strong grip to hold."""
        import mujoco

        weak = make_hand_fixture(capacity=70., gap=0., speed=0.)
        strong = make_hand_fixture(capacity=150., gap=0., speed=0.)
        for field in ("body_mass", "body_inertia", "body_pos", "geom_size", "geom_friction", "eq_solref"):
            np.testing.assert_array_equal(getattr(weak.model, field), getattr(strong.model, field))
        for fixture in (weak, strong):
            model, data, gm = fixture.model, fixture.data, fixture.manager
            self.assertTrue(gm.attach(Limb.LEFT_HAND, fixture.scene.region("left_hand")))
            gm.synchronize_from_live()
            data.xfrc_applied[model.body("left_hand").id, :3] = [0., -100., 0.]
            for _ in range(50):
                gm.evaluate_and_update()
                mujoco.mj_step(model, data)
                gm.evaluate_and_update(applied_data=data)
            eq = model.equality("grasp_left_hand_left_hand").id
            expected = fixture.profile.grip_capacity == 150.
            self.assertEqual(gm.is_attached(Limb.LEFT_HAND), expected)
            self.assertEqual(bool(data.eq_active[eq]), expected)
            self.assertEqual(gm.profile, fixture.profile)
        self.assertGreater(weak.manager.releases[0]["required_load_N"], 70.)
        self.assertEqual(weak.manager.releases[0]["capacity_N"], 70.)
        self.assertFalse(strong.manager.releases)
        self.assertAlmostEqual(strong.manager.get_grasp_load(Limb.LEFT_HAND), 100., delta=.1)

    @mock.patch("boulder_v1.grasp.compute_pose_control", archived_normalized_pose_control)
    def test_simulate_static_stance(self):
        """Archived NONPHYSICAL PD suspension, no Stage3 controller/grip claim."""
        res = simulate_static_stance(
            self.model,
            self.data,
            self.scene,
            self.base_profile,
            steps=500,
            check_grip=True,
        )

        self.assertEqual(res.steps, 500)
        self.assertAlmostEqual(res.time, 1.0, places=2)
        self.assertTrue(res.finite, "Simulation state had NaN or Inf")
        self.assertTrue(res.supported, f"Climber fell! Final root z: {res.final_root_z}")
        self.assertTrue(res.left_hand_attached)
        self.assertTrue(res.right_hand_attached)
        self.assertGreater(res.left_hand_load, 10.0)
        self.assertGreater(res.right_hand_load, 10.0)
        self.assertLess(res.left_hand_load, 600.0)
        self.assertLess(res.right_hand_load, 600.0)
        self.assertGreater(res.min_root_z, 1.05, "Root dipped too low")
        self.assertGreater(res.final_root_z, 1.05, "Root finished too low")

    def test_duplicate_drift_and_wrong_constraint_identity_are_rejected(self):
        import numpy as np

        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        source = self.model.equality("grasp_right_hand_H4").id
        target = self.model.equality("grasp_right_hand_H5").id
        self.data.eq_active[target] = True
        flags = self.data.eq_active.copy()
        for action in (gm.active_attachments, gm.synchronize_from_live):
            with self.assertRaisesRegex(AttachmentStateError, "Multiple active"):
                action()
            np.testing.assert_array_equal(self.data.eq_active, flags)
        self.data.eq_active[target] = False
        self.data.eq_active[source] = False
        with self.assertRaisesRegex(AttachmentStateError, "registry disagrees"):
            gm.contact_configuration()
        gm.synchronize_from_live()
        self.assertNotIn(Limb.RIGHT_HAND, gm.contact_configuration())
        self.data.eq_active[source] = True
        self.model.eq_obj2id[source] = self.model.site("site_H5").id
        with self.assertRaisesRegex(AttachmentStateError, "incorrect type or sites"):
            gm.synchronize_from_live()
        with self.assertRaisesRegex(AttachmentStateError, "incorrect type or sites"):
            GraspManager(self.model, self.data, self.scene)

    def test_new_owner_requires_explicit_adoption_or_initialization(self):
        import mujoco
        import numpy as np

        original = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        original.attach(Limb.RIGHT_HAND, self.scene.region("H5"), force=True)
        self.data.time = 8.0
        self.data.qvel[:] = 0.1
        self.data.ctrl[:] = 0.2
        self.data.qacc_warmstart[:] = 0.3
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, before, spec)
        owner = GraspManager(self.model, self.data, self.scene)
        self.assertFalse(owner.initialized)
        with self.assertRaisesRegex(AttachmentStateError, "registry disagrees"):
            owner.contact_configuration()
        owner.synchronize_from_live()
        after = np.empty_like(before)
        mujoco.mj_getState(self.model, self.data, after, spec)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(owner.contact_configuration(), original.contact_configuration())
        self.assertEqual(owner.contact_configuration()[Limb.RIGHT_HAND], "H5")

        # Explicit reset is also able to clear a duplicated/corrupt activation set.
        self.data.eq_active[self.model.equality("grasp_right_hand_H4").id] = True
        reset_owner = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        self.assertEqual(reset_owner.contact_configuration(), self.scene.start_configuration)
        self.assertEqual(sum(self.data.eq_active), 4)
        self.assertFalse(self.data.eq_active[self.model.equality("grasp_right_hand_H5").id])
        self.assertEqual(self.data.time, 0)
        np.testing.assert_array_equal(self.data.qvel, 0)
        np.testing.assert_array_equal(self.data.ctrl, 0)
        self.assertAlmostEqual(np.linalg.norm(self.data.qpos[3:7]), 1.0, places=14)
        with self.assertRaises(AttachmentStateError):
            owner.contact_configuration()

    def test_failed_attachment_replacement_preserves_current_constraint(self):
        from dataclasses import replace
        import numpy as np

        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile)
        before = self.data.eq_active.copy()
        registry = dict(gm.active_attachments())
        missing = replace(self.h3, id="missing_equality")
        with self.assertRaises(KeyError):
            gm.attach(Limb.LEFT_HAND, missing, force=True)
        np.testing.assert_array_equal(self.data.eq_active, before)
        self.assertEqual(gm.active_attachments(), registry)

    def test_narrow_rom_initialization_rejects_without_mutating_live_episode(self):
        import mujoco
        import numpy as np
        from boulder_v1.runtime import validate_reference_pose

        for scale in (.5, .8):
            with self.subTest(scale=scale):
                profile = ClimberProfile(name="narrow", rom_scale=scale)
                model = compile_model(build_mjcf(self.scene, profile))
                data = mujoco.MjData(model)
                validate_reference_pose(model, model.qpos0)  # Neutral remains legal.
                data.time = 4
                data.ctrl[:] = .1
                data.qvel[:] = .2
                data.qacc_warmstart[:] = .3
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                before = np.empty(mujoco.mj_stateSize(model, spec))
                after = np.empty_like(before)
                mujoco.mj_getState(model, data, before, spec)
                manager = GraspManager(model, data, self.scene, profile=profile)
                with self.assertRaisesRegex(ValueError, "violates compiled ROM"):
                    initialize_episode(model, data, self.scene, profile=profile, manager=manager)
                mujoco.mj_getState(model, data, after, spec)
                np.testing.assert_array_equal(before, after)
                self.assertFalse(manager.initialized)
                self.assertEqual(manager.active_attachments(), {})

    def test_physical_legacy_seed_and_foot_attachment_reject_before_live_reset(self):
        import mujoco

        model = compile_model(build_mjcf(self.scene, self.base_profile))
        data = mujoco.MjData(model)
        gm = GraspManager(model, data, self.scene, profile=self.base_profile)
        data.time, data.ctrl[:], data.qvel[:], data.qacc_warmstart[:] = 4., .1, .2, .3
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, before, spec)
        for kwargs, error in (({}, InitialContactError), ({"attach_feet": True}, ValueError)):
            with self.subTest(kwargs=kwargs), mock.patch.object(mujoco, "mj_resetData") as reset:
                with self.assertRaises(error):
                    initialize_episode(model, data, self.scene, profile=self.base_profile, manager=gm, **kwargs)
                reset.assert_not_called()
                mujoco.mj_getState(model, data, after, spec)
                np.testing.assert_array_equal(before, after)
                self.assertFalse(gm.initialized)
                self.assertEqual(gm.active_attachments(), {})
        with self.assertRaises(ValueError):
            gm.attach(Limb.LEFT_HAND, self.h3, force=True)
        with self.assertRaises(ValueError):
            gm.attach(Limb.LEFT_FOOT, self.scene.region("H1"))

    def test_physical_explicit_initial_pose_acquires_hands_without_hidden_offsets(self):
        fixture = make_mixed_fixture()
        scene = replace(fixture.scene, start_configuration={Limb.LEFT_HAND: "left_hand", Limb.RIGHT_HAND: "right_hand"})
        gm = initialize_episode(fixture.model, fixture.data, scene, profile=fixture.profile,
                                initial_qpos=fixture.reference)
        self.assertEqual(gm.mode, ContactMode.PHYSICAL)
        self.assertEqual(gm.profile, fixture.profile)
        np.testing.assert_array_equal(fixture.data.qpos, fixture.reference)
        np.testing.assert_array_equal(fixture.data.qvel, 0.)
        self.assertEqual(fixture.data.time, 0.)
        self.assertEqual(set(gm.active_attachments()), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        self.assertEqual(sum(fixture.data.eq_active), 2)
        self.assertEqual(len(gm.capture_events), 2)
        self.assertTrue(all(e["gap_m"] <= .001 and e["penetration_m"] < .001 for e in gm.capture_events))
        snapshot = gm.contact_snapshot()
        self.assertTrue(all(state.normal_force == 0. and not state.supporting for state in snapshot.feet.values()))

    def test_idealized_debug_never_evaluates_physical_grip_capacity(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        with mock.patch.object(gm.grip_controller, "evaluate", side_effect=AssertionError("debug is not physical")):
            self.assertEqual(gm.evaluate_and_update(), {})
        self.assertEqual(gm.mode, ContactMode.IDEALIZED_DEBUG)
        self.assertEqual(gm.contact_configuration(), self.scene.start_configuration)


if __name__ == "__main__":
    unittest.main()
