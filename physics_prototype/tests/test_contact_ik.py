"""Stage4 scratch IK tests on real compiled, unmodified physical models."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.contact_benchmarks import make_mixed_fixture
from boulder_v1.contact_geometry import FOOT_SITE_OFFSET, Frame, canonical_geometry
from boulder_v1.contact_ik import generate_contact_reference, solve_contact_pose, solve_hand_reference
from boulder_v1.grasp import GraspManager, STATIC_STANCE_QPOS
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import Affordance, BoulderScene, ClimberProfile, ContactRegion, Limb, SourceType
from boulder_v1.static_state import initialize_static_reference


INTENT = {limb: limb.value.lower() for limb in Limb}


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def model_snapshot(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


def perturbed(model, seed, magnitude=.003):
    tangent = np.random.default_rng(417).normal(size=model.nv) * magnitude
    tangent[:3] = (.002, -.003, .004)
    tangent[3:6] = (.008, -.006, .005)
    # Do not begin on the wrong side of zero-bend hinge limits.
    for joint in range(model.njnt):
        if model.jnt_limited[joint]:
            q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            if seed[q] == model.jnt_range[joint, 0]:
                tangent[v] = abs(tangent[v])
    qpos = np.array(seed, dtype=float)
    mujoco.mj_integratePos(model, qpos, tangent, 1.)
    return qpos


def generated_fixture(profile, *, sphere_feet=False):
    """Generate scene geometry around a legal compiled profile pose, not name offsets."""
    empty = BoulderScene("MuJoCo world", 1., (), ())
    probe_model = mujoco.MjModel.from_xml_string(build_mjcf(empty, profile))
    seed = probe_model.qpos0.copy()
    for side in ("left", "right"):
        for name, value in (("shoulder_pitch", .3), ("elbow", math.pi / 2), ("wrist", -.3)):
            seed[probe_model.joint(f"{side}_{name}").qposadr[0]] = value
        if sphere_feet:
            seed[probe_model.joint(f"{side}_ankle_pitch").qposadr[0]] = math.pi / 6
    probe = mujoco.MjData(probe_model)
    probe.qpos[:] = seed
    mujoco.mj_forward(probe_model, probe)
    regions, intent = [], {}
    for index, limb in enumerate(Limb):
        rid = f"anchor_{91 + 7 * index}"
        intent[limb] = rid
        site = probe.site(f"{limb.value.lower()}_site")
        if limb.is_hand:
            center = site.xpos - .026 * np.array([0., -1., 0.])
            region = ContactRegion(rid, SourceType.HOLD, tuple(center), (0., -1., 0.),
                                   .9, frozenset({Affordance.GRASP}), radius=.02)
        elif sphere_feet:
            normal = np.array([0., -.5, math.sqrt(3) / 2])
            center = site.xpos - (.025 + FOOT_SITE_OFFSET) * normal
            region = ContactRegion(rid, SourceType.HOLD, tuple(center), (0., -1., 0.),
                                   .9, frozenset({Affordance.STEP}), radius=.025)
        else:
            center = site.xpos - np.array([0., .01, FOOT_SITE_OFFSET + .03])
            region = ContactRegion(rid, SourceType.HOLD, tuple(center), (0., -1., 0.),
                                   .9, frozenset({Affordance.STEP}), half_size=(.06, .05, .03))
        regions.append(region)
    scene = replace(empty, contact_regions=tuple(regions))
    model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
    return model, scene, seed, intent


class ContactIKTests(unittest.TestCase):
    def assert_admitted(self, model, scene, profile, intent, result):
        self.assertTrue(result.admitted, result.reason)
        self.assertIsNotNone(result.reference)
        validate_reference_pose(model, result.qpos)
        reference, _ = initialize_static_reference(model, mujoco.MjData(model),
                                                  replace(scene, start_configuration=intent),
                                                  profile, result.qpos, intent)
        # Production initialization renormalizes even an already unit quaternion.
        np.testing.assert_allclose(reference.qpos, result.qpos, rtol=0, atol=1e-15)
        for residual in result.residuals:
            if residual.limb.is_hand:
                self.assertLessEqual(residual.position_error, .001)
                self.assertLess(residual.signed_geom_distance, .001 + 1e-12)
                self.assertGreater(residual.signed_geom_distance, -.001)
            else:
                self.assertIsNotNone(residual.foot_residual, residual.foot_reason)

    def test_perturbed_stage3_mixed_seed_recovers_exact_admission(self):
        fixture = make_mixed_fixture()
        qpos = perturbed(fixture.model, fixture.reference)
        result = solve_contact_pose(fixture.model, fixture.scene, fixture.profile, qpos, INTENT)
        self.assert_admitted(fixture.model, fixture.scene, fixture.profile, INTENT, result)
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertNotEqual(tuple(qpos), result.qpos)
        for residual in result.residuals:
            self.assertLess(residual.position_error, 1e-6)
            self.assertLess(residual.orientation_error, 1e-6)
        self.assertTrue(any(r.position_error > .001 for r in result.initial_residuals))

    def test_valid_seed_preserved_exactly_including_box_offsets(self):
        fixture = make_mixed_fixture()
        result = solve_contact_pose(fixture.model, fixture.scene, fixture.profile,
                                    fixture.reference, INTENT, tolerance=1e-9)
        self.assert_admitted(fixture.model, fixture.scene, fixture.profile, INTENT, result)
        np.testing.assert_array_equal(result.qpos, fixture.reference)
        self.assertEqual(result.iterations, 0)
        for residual in result.residuals:
            if residual.limb.is_foot:
                self.assertAlmostEqual(residual.canonical_position_error, .01, places=13)
                self.assertAlmostEqual(residual.foot_residual.site_normal_residual, 0., places=13)

    def test_loose_ik_tolerance_does_not_short_circuit_exact_sole_admission(self):
        fixture = make_mixed_fixture()
        result = solve_contact_pose(fixture.model, fixture.scene, fixture.profile,
                                    perturbed(fixture.model, fixture.reference), INTENT, tolerance=.001)
        self.assert_admitted(fixture.model, fixture.scene, fixture.profile, INTENT, result)
        self.assertTrue(result.converged)
        for residual in result.residuals:
            if residual.limb.is_foot:
                self.assertLess(abs(residual.foot_site_normal_residual), 1e-6)

    def test_nonunit_seed_quaternion_normalized_only_on_scratch(self):
        fixture = make_mixed_fixture()
        seed = fixture.reference.copy()
        seed[3:7] *= 1.2
        before = seed.copy()
        result = solve_contact_pose(fixture.model, fixture.scene, fixture.profile, seed, INTENT)
        self.assert_admitted(fixture.model, fixture.scene, fixture.profile, INTENT, result)
        self.assertIn("quaternion", result.initial_reason)
        np.testing.assert_array_equal(seed, before)
        self.assertAlmostEqual(np.linalg.norm(result.qpos[3:7]), 1., places=14)

    def test_free_root_manifold_and_full_live_model_manager_nonmutation(self):
        fixture = make_mixed_fixture()
        model, data, manager = fixture.model, fixture.data, fixture.manager
        data.ctrl[:] = .123
        data.qvel[:] = np.linspace(-.01, .01, model.nv)
        data.xfrc_applied[model.body("climber_root").id, :3] = (1., 2., 3.)
        data.time = .73
        before_state, before_model = integration_state(model, data), model_snapshot(model)
        before_events = copy.deepcopy(manager.capture_events)
        before_releases = copy.deepcopy(manager.releases)
        before_attachments = manager.active_attachments().copy()
        seed = perturbed(model, fixture.reference)
        before_seed = seed.copy()
        native = mujoco.mj_integratePos
        native_attach = GraspManager.attach
        tangent_updates = []

        def integrate(m, q, tangent, dt):
            self.assertEqual(tangent.shape, (model.nv,))
            self.assertFalse(np.shares_memory(q, data.qpos))
            self.assertFalse(np.shares_memory(q, seed))
            tangent_updates.append(tangent.copy())
            native(m, q, tangent, dt)

        def attach(owner, limb, region, force=False):
            self.assertIsNot(owner.data, data)
            self.assertTrue(limb.is_hand)
            self.assertFalse(force)
            return native_attach(owner, limb, region, force=force)

        with patch("mujoco.mj_integratePos", side_effect=integrate), \
                patch.object(GraspManager, "attach", new=attach), \
                patch("mujoco.mj_step", side_effect=AssertionError("IK integrated live physics")):
            result = solve_contact_pose(model, fixture.scene, fixture.profile, seed, INTENT)
        self.assertTrue(result.admitted, result.reason)
        self.assertTrue(tangent_updates)
        self.assertTrue(any(np.any(step[:6]) for step in tangent_updates))
        self.assertGreater(np.linalg.norm(np.asarray(result.qpos)[:3] - seed[:3]), 1e-5)
        self.assertAlmostEqual(np.linalg.norm(np.asarray(result.qpos)[3:7]), 1., places=14)
        np.testing.assert_array_equal(seed, before_seed)
        np.testing.assert_array_equal(integration_state(model, data), before_state)
        after_model = model_snapshot(model)
        for field, values in before_model.items():
            np.testing.assert_array_equal(after_model[field], values, err_msg=field)
        self.assertEqual(manager.capture_events, before_events)
        self.assertEqual(manager.releases, before_releases)
        self.assertEqual(manager.active_attachments(), before_attachments)
        with self.assertRaises(FrozenInstanceError):
            result.admitted = False
        with self.assertRaises(TypeError):
            result.qpos[0] = 1.
        with self.assertRaises(TypeError):
            result.reference.contact_intent[Limb.LEFT_HAND] = "bad"

    def test_renamed_holds_and_profile_morphology_compiled_rom(self):
        profile = ClimberProfile("arbitrary_not_base", torso_length=.60, shoulder_width=.46,
                                 hip_width=.34, upper_arm_length=.34, forearm_length=.29,
                                 thigh_length=.46, shin_length=.43, rom_scale=.8)
        model, scene, seed, intent = generated_fixture(profile)
        result = solve_contact_pose(model, scene, profile, perturbed(model, seed), intent)
        self.assert_admitted(model, scene, profile, intent, result)
        self.assertGreater(result.iterations, 0)
        for residual in result.residuals:
            expected = canonical_geometry(scene.region(intent[residual.limb]))
            if residual.limb.is_hand:
                np.testing.assert_array_equal(residual.target.position, expected.hand_frame.position)
                self.assertGreater(residual.orientation_alignment, 1. - 1e-10)

    def test_selected_sphere_foot_normal_and_real_sole(self):
        profile = ClimberProfile("round_supports")
        model, scene, seed, intent = generated_fixture(profile, sphere_feet=True)
        result = solve_contact_pose(model, scene, profile, perturbed(model, seed, .001), intent)
        self.assert_admitted(model, scene, profile, intent, result)
        self.assertGreater(result.iterations, 0)
        for residual in result.residuals:
            if residual.limb.is_foot:
                expected = canonical_geometry(scene.region(intent[residual.limb])).foot_frame
                np.testing.assert_array_equal(residual.target, expected)
                np.testing.assert_allclose(residual.target.normal, [0., -.5, math.sqrt(3) / 2], atol=1e-14)
                self.assertLess(residual.orientation_error, 1e-6)
                self.assertLess(abs(residual.foot_residual.site_normal_residual), 1e-6)
                self.assertIsNone(residual.foot_residual.overlap_area)

    def test_intermediate_hand_world_frame_is_never_four_contact_admission(self):
        fixture = make_mixed_fixture()
        hand = Limb.LEFT_HAND
        canonical = canonical_geometry(fixture.scene.region(INTENT[hand])).hand_frame
        angle = .035
        swing = np.array([[math.cos(angle), -math.sin(angle), 0.],
                          [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        target = Frame(tuple(np.array(canonical.position) + [-.004, -.003, .005]),
                       tuple(map(tuple, swing @ np.array(canonical.rotation))))
        with patch("boulder_v1.contact_ik.initialize_static_reference",
                   side_effect=AssertionError("intermediate reference attempted four-contact admission")):
            result = generate_contact_reference(fixture.model, fixture.scene, fixture.profile,
                                                fixture.reference, INTENT, hand_targets={hand: target})
        self.assertTrue(result.converged, result.reason)
        self.assertFalse(result.admitted)
        self.assertIsNone(result.reference)
        validate_reference_pose(fixture.model, result.qpos)
        data = mujoco.MjData(fixture.model)
        data.qpos[:] = result.qpos
        data.eq_active[:] = False
        mujoco.mj_forward(fixture.model, data)
        np.testing.assert_allclose(data.site("left_hand_site").xpos, target.position, rtol=0, atol=1e-6)
        np.testing.assert_allclose(data.site("left_hand_site").xmat.reshape(3, 3),
                                   target.rotation, rtol=0, atol=1e-6)
        self.assertGreater(np.linalg.norm(data.site("left_hand_site").xpos - canonical.position), .001)

    def test_frozen_root_right_hand_reaches_preserve_three_supports_and_live_model(self):
        fixture = make_mixed_fixture()
        hand = Limb.RIGHT_HAND
        source_region = fixture.scene.region(INTENT[hand])
        for rise in (.04, .06):
            with self.subTest(rise=rise):
                goal = replace(source_region, id=f"right_goal_{round(1000 * rise)}",
                               position=tuple(np.array(source_region.position) + [0., -.02, rise]))
                scene = replace(fixture.scene, contact_regions=(*fixture.scene.contact_regions, goal))
                model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
                live = mujoco.MjData(model)
                # The 40mm future hold overlaps the source palm. This test
                # certifies scratch reference geometry, never static admission
                # of that starting scene or safety of the intervening path.
                live.qpos[:] = fixture.reference
                mujoco.mj_forward(model, live)
                manager = GraspManager(model, live, scene, profile=fixture.profile)
                manager.synchronize_from_live()
                live.ctrl[:] = .123
                live.qvel[:] = np.linspace(-.01, .01, model.nv)
                live.time = .73
                live.xfrc_applied[model.body("climber_root").id, :3] = (1., 2., 3.)
                before_state, before_model = integration_state(model, live), model_snapshot(model)
                before_events = copy.deepcopy(manager.capture_events)
                before_releases = copy.deepcopy(manager.releases)
                before_attachments = manager.active_attachments().copy()
                seed = fixture.reference.copy()
                before_seed = seed.copy()
                canonical = canonical_geometry(goal).hand_frame
                target = replace(canonical, position=tuple(np.array(canonical.position)
                                                          + .0004 * np.array(canonical.normal)))
                root = model.joint("root")
                qa, va = int(root.qposadr[0]), int(root.dofadr[0])
                native_integrate = mujoco.mj_integratePos
                steps = []

                def integrate(m, qpos, tangent, dt):
                    self.assertIs(m, model)
                    self.assertFalse(np.shares_memory(qpos, live.qpos))
                    self.assertFalse(np.shares_memory(qpos, seed))
                    np.testing.assert_array_equal(tangent[va:va + 6], 0.)
                    steps.append(tangent.copy())
                    native_integrate(m, qpos, tangent, dt)

                with patch("mujoco.mj_integratePos", side_effect=integrate), \
                        patch("mujoco.mj_step", side_effect=AssertionError("reference stepped live physics")), \
                        patch("boulder_v1.contact_ik.initialize_static_reference",
                              side_effect=AssertionError("frozen reference attempted capture admission")):
                    result = generate_contact_reference(
                        model, scene, fixture.profile, seed, {**INTENT, hand: goal.id},
                        hand_targets={hand: target}, freeze_root=True)
                self.assertTrue(result.converged, result.reason)
                self.assertFalse(result.admitted)
                self.assertIsNone(result.reference)
                self.assertGreater(result.iterations, 0)
                self.assertTrue(steps)
                validate_reference_pose(model, result.qpos)
                np.testing.assert_array_equal(result.qpos[qa:qa + 7], seed[qa:qa + 7])
                self.assertEqual(int(root.type[0]), mujoco.mjtJoint.mjJNT_FREE)
                probe = mujoco.MjData(model)
                probe.qpos[:] = result.qpos
                probe.eq_active[:] = False
                mujoco.mj_forward(model, probe)
                source = mujoco.MjData(model)
                source.qpos[:] = seed
                source.eq_active[:] = False
                mujoco.mj_forward(model, source)
                for limb in Limb:
                    site_name = f"{limb.value.lower()}_site"
                    expected_position = target.position if limb == hand else source.site(site_name).xpos
                    expected_rotation = target.rotation if limb == hand else source.site(site_name).xmat.reshape(3, 3)
                    np.testing.assert_allclose(probe.site(site_name).xpos, expected_position, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(probe.site(site_name).xmat.reshape(3, 3),
                                               expected_rotation, rtol=0, atol=1e-6)
                evidence = next(r for r in result.residuals if r.limb == hand)
                self.assertGreater(evidence.signed_geom_distance, .0002)
                self.assertFalse(result.collisions)
                print("CONTACT_IK_FROZEN " + json.dumps({
                    "rise_m": rise, "outward_m": .02, "goal_clearance_m": .0004,
                    "iterations": result.iterations, "position_m": evidence.position_error,
                    "orientation_rad": evidence.orientation_error,
                    "signed_distance_m": evidence.signed_geom_distance,
                    "root_exact": True, "admitted": result.admitted}, allow_nan=False), flush=True)
                np.testing.assert_array_equal(seed, before_seed)
                np.testing.assert_array_equal(integration_state(model, live), before_state)
                after_model = model_snapshot(model)
                for field, value in before_model.items():
                    np.testing.assert_array_equal(after_model[field], value, err_msg=field)
                self.assertEqual(manager.capture_events, before_events)
                self.assertEqual(manager.releases, before_releases)
                self.assertEqual(manager.active_attachments(), before_attachments)

    def test_frozen_root_infeasible_reference_returns_uncertified_measured_failure(self):
        fixture = make_mixed_fixture()
        hand = Limb.RIGHT_HAND
        canonical = canonical_geometry(fixture.scene.region(INTENT[hand])).hand_frame
        target = replace(canonical, position=tuple(np.array(canonical.position) + [0., -2., 2.]))
        result = generate_contact_reference(fixture.model, fixture.scene, fixture.profile,
                                            fixture.reference, INTENT, hand_targets={hand: target},
                                            freeze_root=True, max_iterations=25)
        self.assertFalse(result.converged or result.admitted)
        self.assertIsNone(result.reference)
        self.assertIn("not a global infeasibility proof", result.reason)
        np.testing.assert_array_equal(result.qpos[:7], fixture.reference[:7])
        validate_reference_pose(fixture.model, result.qpos)
        self.assertGreater(next(r.position_error for r in result.residuals if r.limb == hand), 1.)

    def test_frozen_nonidentity_root_survives_rom_projection_and_backtracking_exactly(self):
        fixture = make_mixed_fixture()
        seed = fixture.reference.copy()
        seed[3:7] = np.array([.98, .02, .03, .04]) / np.linalg.norm([.98, .02, .03, .04])
        elbow = fixture.model.joint("right_elbow")
        seed[int(elbow.qposadr[0])] = float(elbow.range[1]) + .1
        before = seed.copy()
        hand = Limb.RIGHT_HAND
        target = canonical_geometry(fixture.scene.region(INTENT[hand])).hand_frame
        result = generate_contact_reference(fixture.model, fixture.scene, fixture.profile, seed, INTENT,
                                            hand_targets={hand: target}, freeze_root=True, max_iterations=15)
        np.testing.assert_array_equal(result.qpos[:7], before[:7])
        np.testing.assert_array_equal(seed, before)
        validate_reference_pose(fixture.model, result.qpos)
        self.assertFalse(result.admitted)
        self.assertIsNone(result.reference)

    def test_frozen_root_requires_bool_and_unit_seed_quaternion(self):
        fixture = make_mixed_fixture()
        seed = fixture.reference.copy()
        seed[3:7] *= 1.2
        for solve, kwargs in ((generate_contact_reference, {"hand_targets": {}}), (solve_contact_pose, {})):
            with self.subTest(api=solve.__name__):
                with self.assertRaisesRegex(ValueError, "unit seed quaternion"):
                    solve(fixture.model, fixture.scene, fixture.profile, seed, INTENT,
                          freeze_root=True, **kwargs)
                with self.assertRaisesRegex(ValueError, "must be a bool"):
                    solve(fixture.model, fixture.scene, fixture.profile, fixture.reference, INTENT,
                          freeze_root="yes", **kwargs)

    def test_frozen_root_centered_left_hand_admission_with_bent_box_foot_seed(self):
        fixture = make_mixed_fixture()
        seed = fixture.reference.copy()
        for side in ("left", "right"):
            for name, angle in (("hip_pitch", .3), ("knee", .6), ("ankle_pitch", .3)):
                seed[int(fixture.model.joint(f"{side}_{name}").qposadr[0])] = angle
        source = mujoco.MjData(fixture.model)
        source.qpos[:] = seed
        source.eq_active[:] = False
        mujoco.mj_forward(fixture.model, source)
        regions = []
        for region in fixture.scene.contact_regions:
            if region.id in (INTENT[Limb.LEFT_FOOT], INTENT[Limb.RIGHT_FOOT]):
                limb = next(limb for limb in Limb if INTENT[limb] == region.id)
                geometry = canonical_geometry(region)
                offset = source.site(f"{limb.value.lower()}_site").xpos - geometry.foot_frame.position
                # Retain the Stage3 fixture's valid 10mm tangential site offset.
                offset -= np.array([0., .01, 0.])
                region = replace(region, position=tuple(np.array(region.position) + offset))
            regions.append(region)
        source_scene = replace(fixture.scene, contact_regions=tuple(regions))
        source_model = mujoco.MjModel.from_xml_string(build_mjcf(source_scene, fixture.profile))
        initialize_static_reference(source_model, mujoco.MjData(source_model), source_scene,
                                    fixture.profile, seed, INTENT)
        scene = replace(source_scene, contact_regions=tuple(
            replace(region, position=(0., *region.position[1:])) if region.id == INTENT[Limb.LEFT_HAND]
            else region for region in source_scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        live = mujoco.MjData(model)
        live.qpos[:] = seed
        live.qvel[:] = np.linspace(-.01, .01, model.nv)
        live.ctrl[:] = .123
        live.time = .73
        mujoco.mj_forward(model, live)
        before_state, before_model = integration_state(model, live), model_snapshot(model)
        before_seed = seed.copy()
        root = model.joint("root")
        qa, va = int(root.qposadr[0]), int(root.dofadr[0])
        np.testing.assert_array_equal(seed[qa:qa + 7], model.qpos0[qa:qa + 7])
        native_integrate = mujoco.mj_integratePos
        steps = []

        def integrate(m, qpos, tangent, dt):
            self.assertIs(m, model)
            self.assertFalse(np.shares_memory(qpos, live.qpos))
            self.assertFalse(np.shares_memory(qpos, seed))
            np.testing.assert_array_equal(tangent[va:va + 6], 0.)
            steps.append(tangent.copy())
            native_integrate(m, qpos, tangent, dt)

        with patch("mujoco.mj_integratePos", side_effect=integrate), \
                patch("mujoco.mj_step", side_effect=AssertionError("IK stepped native physics")):
            result = solve_contact_pose(model, scene, fixture.profile, seed, INTENT, freeze_root=True)
        print("CONTACT_IK_CENTERED " + json.dumps({
            "converged": result.converged, "admitted": result.admitted, "iterations": result.iterations,
            "reason": result.reason, "limbs": {r.limb.value: {
                "position_m": r.position_error, "normal_error_rad": r.orientation_error,
                "signed_distance_m": r.signed_geom_distance, "foot_reason": r.foot_reason}
                for r in result.residuals},
            "collisions": [(r.geoms, r.signed_distance) for r in result.collisions]}, allow_nan=False), flush=True)
        self.assert_admitted(model, scene, fixture.profile, INTENT, result)
        self.assertTrue(result.converged, result.reason)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(steps)
        self.assertFalse(result.collisions)
        np.testing.assert_array_equal(result.qpos[qa:qa + 7], before_seed[qa:qa + 7])
        for residual in result.residuals:
            self.assertLess(residual.position_error, 1e-6)
            self.assertLess(residual.orientation_error, 1e-6)
            geometry = canonical_geometry(scene.region(INTENT[residual.limb]))
            frame = geometry.hand_frame if residual.limb.is_hand else geometry.foot_frame
            np.testing.assert_array_equal(residual.target.rotation, frame.rotation)
        # Adoption on disposable episode data must not introduce a physical weld.
        adopted = mujoco.MjData(model)
        _, manager = initialize_static_reference(model, adopted, scene, fixture.profile, result.qpos, INTENT)
        self.assertEqual(int(root.type[0]), mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(set(manager.active_attachments()), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        np.testing.assert_array_equal(seed, before_seed)
        np.testing.assert_array_equal(integration_state(model, live), before_state)
        after_model = model_snapshot(model)
        for field, value in before_model.items():
            np.testing.assert_array_equal(after_model[field], value, err_msg=field)

    def test_close_orientation_invalid_hand_rejected_not_position_certified(self):
        fixture = make_mixed_fixture()
        qpos = fixture.reference.copy()
        q = int(fixture.model.joint("left_wrist").qposadr[0])
        qpos[q] -= .8
        probe = mujoco.MjData(fixture.model)
        probe.qpos[:] = qpos
        probe.eq_active[:] = False
        mujoco.mj_forward(fixture.model, probe)
        # Retarget scene anchors around this pose, leaving the wrong palm normal.
        hand = fixture.scene.region("left_hand")
        center = probe.site("left_hand_site").xpos - .026 * np.array(hand.normal)
        scene = replace(fixture.scene, contact_regions=tuple(
            replace(region, position=tuple(center)) if region.id == hand.id else region
            for region in fixture.scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        result = solve_contact_pose(model, scene, fixture.profile, qpos, INTENT, max_iterations=0)
        evidence = next(r for r in result.residuals if r.limb == Limb.LEFT_HAND)
        self.assertLess(evidence.position_error, 1e-12)
        self.assertGreater(evidence.orientation_error, .7)
        self.assertLess(evidence.orientation_alignment, math.cos(math.pi / 6))
        self.assertFalse(result.admitted or result.converged)
        self.assertIsNone(result.reference)

    def test_one_mm_components_do_not_pass_euclidean_capture(self):
        fixture = make_mixed_fixture()
        hand = fixture.scene.region("left_hand")
        # Each component is under 1mm, but the Euclidean hand gap is over 1mm.
        scene = replace(fixture.scene, contact_regions=tuple(
            replace(region, position=tuple(np.array(region.position) + [.0008, 0., .0008]))
            if region.id == hand.id else region for region in fixture.scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        result = solve_contact_pose(model, scene, fixture.profile, fixture.reference,
                                    INTENT, max_iterations=0, tolerance=.001)
        evidence = next(r for r in result.residuals if r.limb == Limb.LEFT_HAND)
        self.assertGreater(evidence.position_error, .001)
        self.assertLess(evidence.position_error, .0012)
        self.assertFalse(result.admitted or result.converged)

    def test_hand_close_colliding_candidate_and_low_capacity_rejected(self):
        fixture = make_mixed_fixture()
        # Exact position/orientation still cannot certify native reaction capacity.
        low_capacity = replace(fixture.profile, name="weak_grip", grip_capacity=.01)
        result = solve_contact_pose(fixture.model, fixture.scene, low_capacity,
                                    fixture.reference, INTENT, max_iterations=0)
        self.assertTrue(result.converged)
        self.assertFalse(result.admitted)
        self.assertIsNone(result.reference)
        self.assertIn("InitialContactError", result.reason)
        self.assertTrue(all(r.position_error < 1e-12 for r in result.residuals))

        # A very thin, wide box at the canonical hand anchor can have a palm
        # corner deeply inside despite the hand site being exactly at the anchor.
        qpos = fixture.reference.copy()
        qpos[int(fixture.model.joint("left_wrist").qposadr[0])] -= .45
        probe = mujoco.MjData(fixture.model)
        probe.qpos[:] = qpos
        mujoco.mj_forward(fixture.model, probe)
        center = probe.site("left_hand_site").xpos - .016 * np.array([0., -1., 0.])
        scene = replace(fixture.scene, contact_regions=tuple(
            replace(region, position=tuple(center), half_size=(.08, .01, .08))
            if region.id == "left_hand" else region for region in fixture.scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        result = solve_contact_pose(model, scene, fixture.profile, qpos, INTENT, max_iterations=0)
        evidence = next(r for r in result.residuals if r.limb == Limb.LEFT_HAND)
        self.assertLess(evidence.position_error, 1e-12)
        self.assertGreater(evidence.orientation_alignment, math.cos(math.pi / 6))
        self.assertLess(evidence.signed_geom_distance, -.001)
        self.assertFalse(result.admitted)

    def test_full_compiled_rom_projection_and_no_illegal_frozen_hips(self):
        fixture = make_mixed_fixture()
        qpos = fixture.reference.copy()
        joint = fixture.model.joint("left_elbow")
        qpos[int(joint.qposadr[0])] = float(joint.range[1]) + .2
        rejected = solve_contact_pose(fixture.model, fixture.scene, fixture.profile,
                                      qpos, INTENT, max_iterations=0)
        self.assertFalse(rejected.admitted)
        self.assertIn("compiled ROM", rejected.initial_reason)
        solved = solve_contact_pose(fixture.model, fixture.scene, fixture.profile, qpos, INTENT)
        validate_reference_pose(fixture.model, solved.qpos)
        self.assert_admitted(fixture.model, fixture.scene, fixture.profile, INTENT, solved)

        scene, profile = make_synthetic_scene(), ClimberProfile("restricted", rom_scale=.8)
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
        seed = np.array(STATIC_STANCE_QPOS)
        mujoco.mj_normalizeQuat(model, seed)
        solved = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
        validate_reference_pose(model, solved.qpos)
        self.assertIn("left_hip_pitch", solved.initial_reason)
        self.assertIn("right_hip_pitch", solved.initial_reason)
        for side in ("left", "right"):
            joint = model.joint(f"{side}_hip_pitch")
            self.assertLessEqual(solved.qpos[int(joint.qposadr[0])], joint.range[1])
        if solved.admitted:
            self.assert_admitted(model, scene, profile, scene.start_configuration, solved)

    def test_unreachable_intent_reports_local_failure_without_fallback(self):
        fixture = make_mixed_fixture()
        scene = replace(fixture.scene, contact_regions=tuple(
            replace(region, position=(8., -1., 7.)) if region.id == "left_hand" else region
            for region in fixture.scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        for freeze_root in (False, True):
            with self.subTest(freeze_root=freeze_root):
                result = solve_contact_pose(model, scene, fixture.profile, fixture.reference,
                                            INTENT, freeze_root=freeze_root, max_iterations=25)
                self.assertFalse(result.admitted or result.converged)
                self.assertIsNone(result.reference)
                self.assertIn("not a global infeasibility proof", result.reason)
                self.assertGreater(max(r.position_error for r in result.residuals), 1.)
                validate_reference_pose(model, result.qpos)
                if freeze_root:
                    np.testing.assert_array_equal(result.qpos[:7], fixture.reference[:7])

    def test_collision_clearances_use_native_masks_exclusions_not_visuals(self):
        fixture = make_mixed_fixture()
        for mode in ("enabled", "visual", "excluded"):
            with self.subTest(mode=mode):
                tree = ET.fromstring(fixture.xml)
                obstacle = ET.SubElement(tree.find("worldbody"), "body", name="obstacle", pos="0 -0.46 1.18")
                attrs = {"name": "obstacle_geom", "type": "box", "size": ".1 .09 .04"}
                if mode == "visual":
                    attrs.update(contype="0", conaffinity="0")
                ET.SubElement(obstacle, "geom", **attrs)
                if mode == "excluded":
                    # Exclude all affected native rigid bodies, not by geom name.
                    for body in ("pelvis", "abdomen", "left_thigh", "right_thigh"):
                        ET.SubElement(tree.find("contact"), "exclude", body1=body, body2="obstacle")
                model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                result = solve_contact_pose(model, fixture.scene, fixture.profile,
                                            fixture.reference, INTENT, max_iterations=0)
                if mode == "enabled":
                    self.assertFalse(result.admitted)
                    self.assertIn("unintended loaded body contacts", result.reason)
                    self.assertTrue(any("obstacle_geom" in r.geoms for r in result.collisions))
                else:
                    self.assertTrue(result.admitted, result.reason)
                    self.assertFalse(any("obstacle_geom" in r.geoms for r in result.collisions))

    def test_route_intent_replaces_starts_on_scratch_only(self):
        fixture = make_mixed_fixture()
        old_scene = replace(fixture.scene, start_configuration={Limb.LEFT_HAND: "right_hand"})
        before = old_scene.to_dict()
        result = solve_contact_pose(fixture.model, old_scene, fixture.profile, fixture.reference, INTENT)
        self.assertTrue(result.admitted, result.reason)
        self.assertEqual(dict(result.reference.contact_intent), INTENT)
        self.assertEqual(old_scene.to_dict(), before)

    def test_rotated_canonical_frames_and_renamed_route_anchors(self):
        profile = ClimberProfile("rotated_route")
        _, scene, seed, intent = generated_fixture(profile)
        angle = .37
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        translation = np.array([.08, -.04, .07])
        scene = replace(scene, contact_regions=tuple(
            replace(region, position=tuple(rotation @ region.position + translation),
                    normal=tuple(rotation @ region.normal)) for region in scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
        tangent = np.zeros(model.nv)
        tangent[:3] = rotation @ seed[:3] + translation - seed[:3]
        tangent[5] = angle
        mujoco.mj_integratePos(model, seed, tangent, 1.)
        result = solve_contact_pose(model, scene, profile, perturbed(model, seed), intent)
        self.assert_admitted(model, scene, profile, intent, result)
        self.assertGreater(result.iterations, 0)
        for residual in result.residuals:
            self.assertLess(residual.orientation_error, 1e-6)
            if residual.limb.is_hand:
                np.testing.assert_allclose(residual.target.normal, rotation @ [0., -1., 0.], atol=1e-14)

    def test_malformed_inputs_rejected(self):
        fixture = make_mixed_fixture()
        for seed, intent, kwargs in (
            ([0.] * fixture.model.nq, INTENT, {}),
            ([np.nan] * fixture.model.nq, INTENT, {}),
            (fixture.reference[:-1], INTENT, {}),
            (fixture.reference, {Limb.LEFT_HAND: "left_hand"}, {}),
            (fixture.reference, INTENT, {"max_iterations": -1}),
            (fixture.reference, INTENT, {"tolerance": 0.}),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve_contact_pose(fixture.model, fixture.scene, fixture.profile, seed, intent, **kwargs)
        with self.assertRaises(ValueError):
            generate_contact_reference(fixture.model, fixture.scene, fixture.profile, fixture.reference,
                                       INTENT, hand_targets={Limb.LEFT_HAND: Frame((0., 0., 0.),
                                                                                ((1., 0., 0.),) * 3)})

    def test_legacy_base_measured_diagnostics_and_honest_local_verdict(self):
        scene, profile = make_synthetic_scene(), ClimberProfile("base")
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
        seed = np.array(STATIC_STANCE_QPOS)
        mujoco.mj_normalizeQuat(model, seed)
        result = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
        rows = lambda residuals: {r.limb.value: {
            "position_m": r.position_error, "orientation_rad": r.orientation_error,
            "alignment": r.orientation_alignment, "signed_distance_m": r.signed_geom_distance,
            "foot_site_normal_residual_m": r.foot_site_normal_residual,
            "foot_minimum_sole_distance_m": r.foot_minimum_sole_distance,
            "foot_reason": r.foot_reason} for r in residuals}
        print("CONTACT_IK_LEGACY " + json.dumps({
            "admitted": result.admitted, "converged": result.converged,
            "iterations": result.iterations, "reason": result.reason,
            "initial_reason": result.initial_reason, "initial": rows(result.initial_residuals),
            "final": rows(result.residuals),
            "admitted_hand_reactions": [{"limb": r.limb.value, "load_N": r.hand_load,
                                         "capacity_N": r.hand_capacity}
                                        for r in result.reference.residuals if r.limb.is_hand]
                                       if result.reference else [],
            "collisions": [(r.geoms, r.signed_distance) for r in result.collisions]}, allow_nan=False), flush=True)
        initial = {r.limb: r for r in result.initial_residuals}
        self.assertAlmostEqual(initial[Limb.LEFT_HAND].position_error, .0708, delta=.0002)
        self.assertAlmostEqual(initial[Limb.RIGHT_HAND].position_error, .0657, delta=.0002)
        self.assertLess(initial[Limb.LEFT_HAND].signed_geom_distance, -.02)
        self.assertLess(initial[Limb.RIGHT_HAND].signed_geom_distance, -.02)
        if result.admitted:
            self.assert_admitted(model, scene, profile, scene.start_configuration, result)
        else:
            self.assertIsNone(result.reference)
            self.assertNotEqual(result.qpos, tuple(seed))
            self.assertIn("Local failure", result.reason)


class HandReferenceTests(unittest.TestCase):
    def assert_reference(self, model, measured, limb, target, result, *, converged=True):
        self.assertEqual(result.converged, converged, result.reason)
        validate_reference_pose(model, result.qpos)
        side = limb.value.lower().removesuffix("_hand")
        joints = [model.joint(f"{side}_{name}").id
                  for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
        frozen = np.ones(model.nq, dtype=bool)
        frozen[model.jnt_qposadr[joints]] = False
        self.assertEqual(np.asarray(result.qpos)[frozen].tobytes(), np.asarray(measured)[frozen].tobytes())
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = result.qpos
        scratch.eq_active[:] = False
        mujoco.mj_forward(model, scratch)
        site = scratch.site(f"{limb.value.lower()}_site")
        normal = site.xmat.reshape(3, 3)[:, 2]
        target_normal = np.asarray(target.normal)
        alignment = float(np.clip(normal @ target_normal, -1., 1.))
        angle = math.atan2(np.linalg.norm(np.cross(normal, target_normal)), alignment)
        self.assertAlmostEqual(result.position_error, np.linalg.norm(site.xpos - target.position), delta=1e-12)
        self.assertAlmostEqual(result.orientation_error, angle, delta=1e-12)
        self.assertAlmostEqual(result.orientation_alignment, alignment, delta=1e-12)
        if converged:
            self.assertLessEqual(result.position_error, 1e-7)
            self.assertLessEqual(result.orientation_error, 1e-7)
        self.assertFalse(hasattr(result, "admitted"))
        self.assertFalse(hasattr(result, "reference"))

    def test_both_compiled_arms_reach_up_60mm_outward_20mm_without_other_changes(self):
        fixture = make_mixed_fixture()
        model, live = fixture.model, fixture.data
        for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            with self.subTest(limb=limb):
                canonical = canonical_geometry(fixture.scene.region(INTENT[limb])).hand_frame
                target = replace(canonical, position=tuple(np.array(canonical.position) + [0., -.02, .06]))
                before = live.qpos.copy()
                result = solve_hand_reference(model, live.qpos, limb, target)
                self.assert_reference(model, before, limb, target, result)
                self.assertGreater(result.iterations, 0)
                np.testing.assert_array_equal(live.qpos, before)
                print("HAND_REFERENCE_METRICS " + json.dumps({
                    "limb": limb.value, "iterations": result.iterations,
                    "position_m": result.position_error, "normal_error_rad": result.orientation_error,
                    "alignment": result.orientation_alignment}, allow_nan=False), flush=True)

    def test_measured_root_torso_frame_exact_preservation_step_bound_and_nonmutation(self):
        fixture = make_mixed_fixture()
        model, live, manager = fixture.model, fixture.data, fixture.manager
        live.qpos[3:7] = np.array([.995, .04, -.02, .035]) / np.linalg.norm([.995, .04, -.02, .035])
        live.qpos[0] = -0.
        for name, value in (("waist_yaw", .04), ("waist_pitch", -.025), ("waist_roll", .03),
                            ("left_shoulder_roll", .02), ("left_hip_pitch", .15), ("left_knee", .3)):
            live.qpos[int(model.joint(name).qposadr[0])] = value
        live.ctrl[:] = .123
        live.qvel[:] = np.linspace(-.01, .01, model.nv)
        live.time = .73
        live.xfrc_applied[model.body("climber_root").id, :3] = (1., 2., 3.)
        mujoco.mj_forward(model, live)
        canonical = canonical_geometry(fixture.scene.region("right_hand")).hand_frame
        target = replace(canonical, position=tuple(live.site("right_hand_site").xpos + [0., -.02, .06]))
        measured = live.qpos.copy()
        before_state, before_model = integration_state(model, live), model_snapshot(model)
        before_events, before_releases = copy.deepcopy(manager.capture_events), copy.deepcopy(manager.releases)
        before_attachments = manager.active_attachments().copy()
        dofs = [int(model.joint(f"right_{name}").dofadr[0])
                for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
        frozen = np.ones(model.nv, dtype=bool)
        frozen[dofs] = False
        native_integrate = mujoco.mj_integratePos
        steps = []

        def integrate(m, qpos, tangent, dt):
            self.assertIs(m, model)
            self.assertFalse(np.shares_memory(qpos, live.qpos))
            self.assertFalse(np.shares_memory(qpos, measured))
            np.testing.assert_array_equal(tangent[frozen], 0.)
            self.assertLessEqual(np.max(np.abs(tangent)), .1 + 1e-15)
            self.assertLessEqual(np.linalg.norm(tangent), .1 + 1e-15)
            steps.append(tangent.copy())
            native_integrate(m, qpos, tangent, dt)

        with patch("mujoco.mj_integratePos", side_effect=integrate), \
                patch("mujoco.mj_step", side_effect=AssertionError("arm reference stepped physics")), \
                patch.object(GraspManager, "attach", side_effect=AssertionError("arm reference attempted capture")), \
                patch("boulder_v1.contact_ik.initialize_static_reference",
                      side_effect=AssertionError("arm reference attempted admission")):
            result = solve_hand_reference(model, live.qpos, Limb.RIGHT_HAND, target)
        self.assert_reference(model, measured, Limb.RIGHT_HAND, target, result)
        self.assertTrue(steps)
        self.assertEqual(live.qpos.tobytes(), measured.tobytes())
        np.testing.assert_array_equal(integration_state(model, live), before_state)
        after_model = model_snapshot(model)
        for field, value in before_model.items():
            np.testing.assert_array_equal(after_model[field], value, err_msg=field)
        self.assertEqual(manager.capture_events, before_events)
        self.assertEqual(manager.releases, before_releases)
        self.assertEqual(manager.active_attachments(), before_attachments)
        with self.assertRaises(FrozenInstanceError):
            result.converged = False
        with self.assertRaises(TypeError):
            result.qpos[0] = 0.

    def test_tangent_twist_is_not_a_sixth_task_and_matched_seed_is_exact(self):
        fixture = make_mixed_fixture()
        site = fixture.data.site("right_hand_site")
        angle = 1.1
        twist = np.array([[math.cos(angle), -math.sin(angle), 0.],
                          [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        target = Frame(tuple(site.xpos), tuple(map(tuple, site.xmat.reshape(3, 3) @ twist)))
        result = solve_hand_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target,
                                      max_iterations=0)
        self.assert_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target, result)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(np.asarray(result.qpos).tobytes(), fixture.data.qpos.tobytes())

    def test_antiparallel_normal_with_zero_position_gap_cannot_false_converge(self):
        fixture = make_mixed_fixture()
        site = fixture.data.site("right_hand_site")
        target = Frame(tuple(site.xpos), tuple(map(tuple, site.xmat.reshape(3, 3) @ np.diag([1., -1., -1.]))))
        for budget in (0, 1):
            with self.subTest(budget=budget):
                result = solve_hand_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target,
                                              max_iterations=budget)
                self.assert_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target,
                                      result, converged=False)
                self.assertGreater(result.orientation_error, 2.)
                if not budget:
                    self.assertLess(result.position_error, 1e-12)
                    self.assertAlmostEqual(result.orientation_error, math.pi, places=13)
                    self.assertAlmostEqual(result.orientation_alignment, -1., places=13)

    def test_soft_limit_deviations_anywhere_and_nonunit_root_fail_before_forward(self):
        fixture = make_mixed_fixture()
        target = canonical_geometry(fixture.scene.region("right_hand")).hand_frame
        seeds = []
        for name, bound, sign in (("right_elbow", 1, 1), ("left_ankle_roll", 0, -1), ("waist_yaw", 1, 1)):
            seed = fixture.data.qpos.copy()
            joint = fixture.model.joint(name)
            seed[int(joint.qposadr[0])] = joint.range[bound] + sign * 1e-9
            seeds.append((seed, "compiled ROM"))
        seed = fixture.data.qpos.copy()
        seed[3:7] *= 1.000001
        seeds.append((seed, "unit length"))
        for seed, message in seeds:
            before = seed.copy()
            with self.subTest(message=message), \
                    patch("mujoco.mj_forward", side_effect=AssertionError("invalid measured pose forwarded")), \
                    patch("mujoco.mj_integratePos", side_effect=AssertionError("invalid measured pose clipped")), \
                    self.assertRaisesRegex(ValueError, message):
                solve_hand_reference(fixture.model, seed, Limb.RIGHT_HAND, target)
            self.assertEqual(seed.tobytes(), before.tobytes())

    def test_nonfinite_malformed_inputs_and_invalid_target_frames_fail_before_forward(self):
        fixture = make_mixed_fixture()
        target = canonical_geometry(fixture.scene.region("right_hand")).hand_frame
        invalid_seeds = [np.full(fixture.model.nq, value) for value in (np.nan, np.inf, -np.inf)]
        invalid_seeds.extend((fixture.data.qpos[:-1], fixture.data.qpos[:, None], fixture.data))
        invalid_targets = (
            replace(target, position=(np.nan, 0., 0.)),
            replace(target, position=(0., np.inf, 0.)),
            replace(target, rotation=((np.nan, 0., 0.), (0., 1., 0.), (0., 0., 1.))),
            replace(target, rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., -1.))),
            replace(target, rotation=((1., 0., 0.),) * 3),
        )
        with patch("mujoco.mj_forward", side_effect=AssertionError("invalid input forwarded")):
            for seed in invalid_seeds:
                with self.assertRaises(ValueError):
                    solve_hand_reference(fixture.model, seed, Limb.RIGHT_HAND, target)
            for frame in invalid_targets:
                with self.assertRaises(ValueError):
                    solve_hand_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, frame)
            for limb in (Limb.LEFT_FOOT, "RIGHT_HAND"):
                with self.assertRaises(ValueError):
                    solve_hand_reference(fixture.model, fixture.data.qpos, limb, target)
            for kwargs in ({"max_iterations": -1}, {"max_iterations": True},
                           {"tolerance": 0.}, {"tolerance": np.nan}, {"tolerance": np.inf}):
                with self.assertRaises(ValueError):
                    solve_hand_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target, **kwargs)

    def test_task_match_can_converge_despite_real_palm_collision_without_claiming_admission(self):
        fixture = make_mixed_fixture()
        site = fixture.data.site("right_hand_site")
        target = Frame(tuple(site.xpos), tuple(map(tuple, site.xmat.reshape(3, 3))))
        scene = replace(fixture.scene, contact_regions=tuple(
            replace(region, position=tuple(fixture.data.geom("right_hand_geom").xpos))
            if region.id == "right_hand" else region for region in fixture.scene.contact_regions))
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = fixture.reference
        scratch.eq_active[:] = False
        mujoco.mj_forward(model, scratch)
        distance = mujoco.mj_geomDistance(model, scratch, model.geom("right_hand_geom").id,
                                         model.geom("geom_right_hand").id, math.inf, None)
        self.assertLess(distance, -.001)
        result = solve_hand_reference(model, fixture.reference, Limb.RIGHT_HAND, target, max_iterations=0)
        self.assert_reference(model, fixture.reference, Limb.RIGHT_HAND, target, result)
        self.assertIn("no contact admission", result.reason)

    def test_unreachable_arm_target_returns_measured_rom_valid_failure(self):
        fixture = make_mixed_fixture()
        canonical = canonical_geometry(fixture.scene.region("right_hand")).hand_frame
        target = replace(canonical, position=tuple(np.array(canonical.position) + [5., -2., 3.]))
        result = solve_hand_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target,
                                      max_iterations=30)
        self.assert_reference(fixture.model, fixture.data.qpos, Limb.RIGHT_HAND, target,
                              result, converged=False)
        self.assertGreater(result.position_error, 1.)
        self.assertIn("not a global infeasibility proof", result.reason)


if __name__ == "__main__":
    unittest.main()
