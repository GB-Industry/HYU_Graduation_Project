"""Contact-boundary regressions, independent of production choreography success."""
from dataclasses import replace
import math
import unittest
from unittest import mock

import mujoco
import numpy as np

from boulder_v1 import (AttachmentStateError, GraspManager, Limb, TransitionRequest,
                         TransitionStatus, build_mjcf, compile_model, execute_transition,
                         initialize_episode, simulate_single_limb_reach, simulate_static_stance,
                         simulate_three_point_support)
from boulder_v1.contact_benchmarks import FOOT_CASES, make_foot_fixture, make_hand_fixture, make_mixed_fixture
from boulder_v1.schema import Affordance


def integration(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, value, spec)
    return value


class ContactIntegrityTests(unittest.TestCase):
    def test_profile_and_geometry_faults_fail_closed_without_state_mutation(self):
        fixture = make_mixed_fixture()
        weak = replace(fixture.profile, grip_capacity=70.)
        default_owner = GraspManager(fixture.model, fixture.data, fixture.scene)
        before = integration(fixture.model, fixture.data)
        with self.assertRaisesRegex(AttachmentStateError, "profile differs"):
            initialize_episode(fixture.model, fixture.data, fixture.scene, weak,
                               manager=default_owner, initial_qpos=fixture.reference)
        np.testing.assert_array_equal(before, integration(fixture.model, fixture.data))

        for simulate in (simulate_static_stance, simulate_three_point_support):
            with self.subTest(simulate=simulate.__name__):
                with self.assertRaisesRegex(AttachmentStateError, "profile differs"):
                    simulate(fixture.model, fixture.data, fixture.scene, weak,
                             steps=1, manager=fixture.manager)
                np.testing.assert_array_equal(before, integration(fixture.model, fixture.data))

        target = fixture.scene.region("left_hand")
        with mock.patch.object(mujoco, "mj_geomDistance", return_value=float("nan")):
            with self.assertRaisesRegex(ValueError, "signed hand/surface"):
                fixture.manager.can_attach(Limb.LEFT_HAND, target)
            state = fixture.manager.contact_snapshot().hands[Limb.LEFT_HAND]
            self.assertTrue(state.active)
            self.assertFalse(state.valid)
            self.assertFalse(state.measurement_valid)
        np.testing.assert_array_equal(before, integration(fixture.model, fixture.data))
        fixture.model.geom_size[fixture.model.geom("geom_left_hand").id, 0] = float("nan")
        state = fixture.manager.contact_snapshot().hands[Limb.LEFT_HAND]
        self.assertTrue(state.active)
        self.assertFalse(state.valid)
        self.assertFalse(state.measurement_valid)
        with self.assertRaisesRegex(ValueError, "compiled contact geometry"):
            fixture.manager.evaluate_and_update()

        moved = make_hand_fixture(gap=0., speed=0.)
        self.assertTrue(moved.manager.attach(Limb.LEFT_HAND, moved.scene.region("left_hand")))
        moved.manager.synchronize_from_live()
        body = moved.model.body("contact_left_hand").id
        site = moved.model.site("site_left_hand").id
        moved.model.body_pos[body, 0] += 1.
        moved.model.site_pos[site, 0] -= 1.
        self.assertFalse(moved.manager.contact_snapshot().hands[Limb.LEFT_HAND].valid)
        decisions = moved.manager.evaluate_and_update()
        self.assertFalse(decisions[Limb.LEFT_HAND].maintain)
        self.assertFalse(moved.manager.is_attached(Limb.LEFT_HAND))

    def test_every_rollout_helper_checks_capacity_before_native_integration(self):
        for simulate in (simulate_static_stance, simulate_three_point_support):
            with self.subTest(simulate=simulate.__name__):
                fixture = make_mixed_fixture()
                profile = replace(fixture.profile, grip_capacity=400.)
                scene = replace(fixture.scene, start_configuration={Limb.LEFT_HAND: "left_hand", Limb.RIGHT_HAND: "right_hand",
                                                                    Limb.LEFT_FOOT: "left_foot", Limb.RIGHT_FOOT: "right_foot"})
                manager = GraspManager(fixture.model, fixture.data, scene, profile=profile)
                manager.synchronize_from_live()
                # Deliberate actual hand overload, independent of controller units/gains.
                fixture.data.xfrc_applied[fixture.model.body("left_hand").id, :3] = [0., -10_000., 0.]
                qpos, qvel, time = fixture.data.qpos.copy(), fixture.data.qvel.copy(), fixture.data.time
                with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("overload must be rejected first")) as native_step:
                    result = simulate(fixture.model, fixture.data, scene, profile,
                                      steps=1, manager=manager, check_grip=False)
                native_step.assert_not_called()
                self.assertEqual(result.steps, 0)
                self.assertEqual(fixture.data.time, time)
                np.testing.assert_array_equal(fixture.data.qpos, qpos)
                np.testing.assert_array_equal(fixture.data.qvel, qvel)
                self.assertTrue(manager.releases)
                self.assertTrue(all(e["required_load_N"] > e["capacity_N"] for e in manager.releases))
                self.assertFalse(result.supported)

    def test_physical_reach_does_not_report_height_alone_as_support(self):
        fixture = make_mixed_fixture()
        result = simulate_single_limb_reach(
            fixture.model, fixture.data, fixture.scene, fixture.profile,
            from_region_id="right_hand", to_region_id="left_hand", manager=fixture.manager,
        )
        self.assertEqual(result.status, TransitionStatus.SUPPORT_FAILURE)
        self.assertGreater(result.final_root_z, .8)
        self.assertFalse(result.supported)
        self.assertEqual(result.steps, 0)

    def test_capture_windows_belong_to_one_acquisition_and_mark_early_release(self):
        fixture = make_hand_fixture(gap=0., speed=0.)
        manager, model, data = fixture.manager, fixture.model, fixture.data
        region = fixture.scene.region("left_hand")
        self.assertTrue(manager.attach(Limb.LEFT_HAND, region))
        manager.synchronize_from_live()
        first = manager.capture_events[-1]
        self.assertTrue(manager.attach(Limb.LEFT_HAND, region))
        self.assertTrue(first["window_truncated"])
        self.assertEqual(first["window_truncation_time_s"], 0.)
        closed = first.copy()
        second = manager.capture_events[-1]
        data.xfrc_applied[model.body("left_hand").id, :3] = [0., -100., 0.]
        manager.evaluate_and_update()
        mujoco.mj_step(model, data)
        manager.evaluate_and_update(applied_data=data)
        self.assertEqual(first, closed)
        self.assertGreater(second["peak_applied_reaction_N"], 90.)
        self.assertEqual(second["window_last_sample_time_s"], data.time)
        self.assertTrue(math.isfinite(second["peak_hand_kinetic_energy_J"]))
        self.assertTrue(manager.detach(Limb.LEFT_HAND))
        self.assertTrue(second["window_truncated"])
        self.assertEqual(second["window_truncation_time_s"], data.time)
        closed = second.copy()
        mujoco.mj_step(model, data)
        manager.evaluate_and_update(applied_data=data)
        self.assertEqual(second, closed)

    def test_release_force_epochs_and_episode_reset_are_truthful(self):
        fixture = make_hand_fixture(capacity=70., gap=0., speed=0.)
        region = fixture.scene.region("left_hand")
        self.assertTrue(fixture.manager.attach(Limb.LEFT_HAND, region))
        fixture.manager.synchronize_from_live()
        fixture.data.xfrc_applied[fixture.model.body("left_hand").id, :3] = [0., -100., 0.]
        mujoco.mj_step(fixture.model, fixture.data)  # Deliberately bypass guard to audit applied-force provenance.
        fixture.data.xfrc_applied[:] = 0.
        fixture.manager.evaluate_and_update(applied_data=fixture.data)
        release = fixture.manager.releases[-1]
        self.assertEqual(release["force_epoch"], "applied_preintegration_solve")
        self.assertAlmostEqual(release["force_state_time_s"], 0.)
        self.assertAlmostEqual(math.hypot(*release["force_world_N"]), release["required_load_N"])
        self.assertGreater(release["applied_load_N"], release["endpoint_load_N"])

        mixed = make_mixed_fixture()
        scene = replace(mixed.scene, start_configuration={Limb.LEFT_HAND: "left_hand", Limb.RIGHT_HAND: "right_hand"})
        manager = GraspManager(mixed.model, mixed.data, scene, profile=mixed.profile)
        manager.synchronize_from_live()
        manager.capture_events.append({"time_s": 20.})
        manager.releases.append({"time_s": 20.})
        manager.last_capture_failure[Limb.LEFT_HAND] = "previous episode"
        initialize_episode(mixed.model, mixed.data, scene, mixed.profile,
                           manager=manager, initial_qpos=mixed.reference)
        self.assertEqual(mixed.data.time, 0.)
        self.assertEqual(len(manager.capture_events), 2)
        self.assertTrue(all(event["time_s"] == 0. for event in manager.capture_events))
        self.assertEqual(manager.releases, [])
        self.assertEqual(manager.last_capture_failure, {})

    def test_floor_support_is_observed_but_not_admitted_as_a_scripted_hold(self):
        fixture = make_mixed_fixture()
        regions = tuple(replace(region, position=(region.position[0], region.position[1], region.position[2] - .345))
                        for region in fixture.scene.contact_regions if Affordance.GRASP in region.affordances)
        scene = replace(fixture.scene, contact_regions=regions)
        model = compile_model(build_mjcf(scene, fixture.profile))
        data = mujoco.MjData(model)
        reference = fixture.reference.copy()
        reference[2] -= .345
        data.qpos[:] = reference
        manager = GraspManager(model, data, scene, profile=fixture.profile)
        for limb, name in ((Limb.LEFT_HAND, "left_hand"), (Limb.RIGHT_HAND, "right_hand")):
            self.assertTrue(manager.attach(limb, scene.region(name)))
        manager.synchronize_from_live()
        for _ in range(10):
            self.assertTrue(all(d.maintain for d in manager.evaluate_and_update().values()))
            mujoco.mj_step(model, data)
            manager.evaluate_and_update(applied_data=data)
        self.assertEqual(manager.contact_configuration()[Limb.LEFT_FOOT], "floor")
        before = integration(model, data)
        result = execute_transition(model, data, scene, fixture.profile,
                                    TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand", steps=1), manager=manager)
        self.assertEqual(result.status, TransitionStatus.SUPPORT_FAILURE)
        self.assertEqual(result.steps, 0)

        # The floor and a named HOLD may share a scene label, but never load credit.
        from boulder_v1.locomotion import _supported_hold_geometry
        from boulder_v1.support import FootSupportSensor
        import xml.etree.ElementTree as ET
        foot = make_foot_fixture(FOOT_CASES[1])  # Effective HOLD friction 0.8, not automatic max mixing.
        tree = ET.fromstring(foot.xml.replace("SUPPORT", "floor"))
        surface = tree.find(".//body[@name='contact_floor']")
        surface.set("pos", "0 0 -0.05002")
        ET.SubElement(tree.find("worldbody"), "geom", name="floor", type="plane", size="5 5 0.1", friction="0.8 0.05 0.005")
        ET.SubElement(tree.find("contact"), "pair", geom1="left_foot_geom", geom2="floor",
                      condim="3", friction="1 1 0 0 0", solref="0.01 1")
        patch_scene = replace(foot.scene, contact_regions=(replace(foot.scene.contact_regions[0], id="floor", position=(0., 0., -.05002)),))
        patch_model = compile_model(ET.tostring(tree, encoding="unicode"))
        patch_data = mujoco.MjData(patch_model)
        for _ in range(300):
            mujoco.mj_step(patch_model, patch_data)
        state = FootSupportSensor().measure(patch_model, patch_data, patch_scene)[Limb.LEFT_FOOT]
        hold_load = sum(c.normal_force for c in state.contacts if c.surface_geom == "geom_floor")
        self.assertGreater(hold_load, 0.)
        self.assertLess(hold_load, 5.)
        self.assertIn("floor", state.support_surfaces)
        self.assertNotIn("geom_floor", state.support_surfaces)
        self.assertFalse(_supported_hold_geometry(state, "floor"))
        np.testing.assert_array_equal(before, integration(model, data))
        with self.assertRaisesRegex(AttachmentStateError, "native HOLD surfaces"):
            simulate_three_point_support(model, data, scene, fixture.profile, steps=1, manager=manager)
        np.testing.assert_array_equal(before, integration(model, data))

        from boulder_v1.schema import ContactRegion, SourceType
        misleading = ContactRegion("floor", SourceType.HOLD, (.7, -1.5, 2.), (0., -1., 0.), .9,
                                    frozenset({Affordance.STEP}), half_size=(.06, .05, .03))
        aliased_scene = replace(scene, contact_regions=(*scene.contact_regions, misleading))
        aliased_model = compile_model(build_mjcf(aliased_scene, fixture.profile))
        aliased_data = mujoco.MjData(aliased_model)
        for field in ("qpos", "qvel", "ctrl", "eq_active", "qacc_warmstart", "qfrc_applied"):
            getattr(aliased_data, field)[:] = getattr(data, field)
        aliased_data.time = data.time
        mujoco.mj_forward(aliased_model, aliased_data)
        owner = GraspManager(aliased_model, aliased_data, aliased_scene, profile=fixture.profile)
        owner.synchronize_from_live()
        result = execute_transition(aliased_model, aliased_data, aliased_scene, fixture.profile,
                                    TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand", steps=1), manager=owner)
        self.assertEqual(result.status, TransitionStatus.SUPPORT_FAILURE)
        self.assertEqual(result.steps, 0)
