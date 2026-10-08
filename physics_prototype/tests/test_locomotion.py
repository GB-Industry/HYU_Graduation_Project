import unittest
from dataclasses import replace
from functools import partial
from unittest import mock

import numpy as np

from boulder_v1.grasp import AttachmentStateError, GraspManager, InitialContactError, initialize_episode, setup_static_stance
from boulder_v1.contact_benchmarks import make_mixed_fixture, verify_fixed_allocation
from boulder_v1.contact_geometry import ContactMode
from boulder_v1.locomotion import (
    SingleLimbTransitionResult,
    StateSummary,
    ThreePointSupportResult,
    TransitionPhase,
    TransitionRequest,
    TransitionResult,
    TransitionSequenceResult,
    TransitionStatus,
    check_stabilization_readiness,
    execute_transition,
    execute_transition_sequence,
    get_state_summary,
    simulate_single_limb_reach,
    simulate_three_point_support,
)
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import compile_model, compute_pose_control, mujoco_available
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import Affordance, ClimberProfile, Limb
from boulder_v1.support import FootStatus
from foundation_fixture_control import archived_normalized_pose_control


debug_request = partial(TransitionRequest, max_attach_distance=.15)


class IdealizedDebugLocomotionTests(unittest.TestCase):
    """Stage0 contracts with archived NONPHYSICAL PD, no Stage3 controller claim."""

    def setUp(self):
        if not mujoco_available():
            self.skipTest("MuJoCo is not available")
        import mujoco

        self.scene = make_synthetic_scene()
        self.base_profile = ClimberProfile(name="base")
        self.xml = build_mjcf(self.scene, self.base_profile, contact_mode=ContactMode.IDEALIZED_DEBUG)
        self.model = compile_model(self.xml)
        self.data = mujoco.MjData(self.model)
        archived_control = partial(archived_normalized_pose_control, default_kp=30., default_kd=3.)
        self.enterContext(mock.patch("boulder_v1.locomotion.compute_pose_control", archived_control))
        self.enterContext(mock.patch(f"{__name__}.compute_pose_control", archived_control))

    def snapshot(self, data, manager):
        import mujoco

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, data, state, spec)
        derived = {name: getattr(data, name).copy() for name in ("qacc", "site_xpos", "efc_force", "efc_pos")}
        return state, derived, dict(manager._attachments) if manager is not None else None

    def assert_snapshot_equal(self, before, after):
        np.testing.assert_array_equal(before[0], after[0])
        for name in before[1]:
            np.testing.assert_array_equal(before[1][name], after[1][name])
        self.assertEqual(before[2], after[2])

    def assert_actual_result(self, result, model, data, manager):
        actual = {}
        for limb in Limb:
            prefix = f"grasp_{limb.value.lower()}_"
            identities = [model.equality(i).name for i in range(model.neq)
                          if data.eq_active[i] and model.equality(i).name.startswith(prefix)]
            self.assertLessEqual(len(identities), 1)
            if identities:
                actual[limb] = identities[0][len(prefix):]
        self.assertEqual(actual, manager.contact_configuration())
        self.assertEqual(actual, {limb: att.region.id for limb, att in manager._attachments.items()})
        self.assertEqual(actual, result.final_contact_configuration)
        self.assertEqual(actual, result.final_state.contact_configuration)
        self.assertEqual(result.initial_state.contact_mode, ContactMode.IDEALIZED_DEBUG)
        self.assertEqual(result.final_state.contact_mode, ContactMode.IDEALIZED_DEBUG)
        self.assertTrue(set(result.final_state.attachment_loads).issubset({Limb.LEFT_HAND, Limb.RIGHT_HAND}))
        for limb, state in result.final_state.foot_states.items():
            if limb in actual:
                self.assertEqual(state.status, FootStatus.IDEALIZED)
                self.assertEqual(state.idealized_attachment, actual[limb])
                self.assertFalse(state.supporting)
                self.assertEqual(state.normal_force, 0.)
        for field in ("qpos", "qvel", "eq_active", "ctrl", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(result.final_state, field), getattr(data, field))
        if result.success:
            self.assertTrue(result.released)
            self.assertTrue(result.target_captured)
            self.assertTrue(result.final_state.finite)
            self.assertEqual(actual[result.limb], result.target_hold)
            source_id = model.equality(f"grasp_{result.limb.value.lower()}_{result.source_hold}").id
            target_id = model.equality(f"grasp_{result.limb.value.lower()}_{result.target_hold}").id
            self.assertTrue(result.initial_state.eq_active[source_id])
            self.assertFalse(data.eq_active[source_id])
            self.assertTrue(data.eq_active[target_id])
            self.assertEqual(result.phases_traversed, tuple(p.value for p in TransitionPhase))

    def test_three_point_support_rh_release(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile)
        result = simulate_three_point_support(
            self.model,
            self.data,
            self.scene,
            self.base_profile,
            released_limb=Limb.RIGHT_HAND,
            steps=200,
            manager=gm,
        )
        self.assertTrue(result.finite, "Simulation encountered NaN or non-finite values")
        self.assertTrue(result.supported, f"Humanoid fell below 0.8m (min_z={result.min_root_z:.3f})")
        self.assertEqual(result.released_limb, Limb.RIGHT_HAND)
        self.assertTrue(result.supporting_attached[Limb.LEFT_HAND], "Left hand failed to maintain support")
        self.assertGreater(result.supporting_loads[Limb.LEFT_HAND], 100.0, "Left hand has suspiciously low load")

    def test_three_point_support_lh_release(self):
        import mujoco

        data = mujoco.MjData(self.model)
        gm = initialize_episode(self.model, data, self.scene, profile=self.base_profile)
        result = simulate_three_point_support(
            self.model,
            data,
            self.scene,
            self.base_profile,
            released_limb=Limb.LEFT_HAND,
            steps=200,
            manager=gm,
        )
        self.assertTrue(result.finite)
        self.assertTrue(result.supported)
        self.assertEqual(result.released_limb, Limb.LEFT_HAND)
        self.assertTrue(result.supporting_attached[Limb.RIGHT_HAND])
        self.assertGreater(result.supporting_loads[Limb.RIGHT_HAND], 100.0)

    def test_three_point_support_idealized_debug_is_not_a_capacity_certificate(self):
        import mujoco

        # Debug constraints deliberately bypass physical capacity enforcement.
        weak_profile = ClimberProfile(name="weak", grip_capacity=200.0)
        xml = build_mjcf(self.scene, weak_profile, contact_mode=ContactMode.IDEALIZED_DEBUG)
        model = compile_model(xml)
        data = mujoco.MjData(model)
        gm = initialize_episode(model, data, self.scene, profile=weak_profile)

        result = simulate_three_point_support(
            model,
            data,
            self.scene,
            weak_profile,
            released_limb=Limb.RIGHT_HAND,
            steps=200,
            check_grip=True,
            manager=gm,
        )
        self.assertTrue(result.finite)
        self.assertTrue(result.supporting_attached[Limb.LEFT_HAND])
        self.assertEqual(gm.mode, ContactMode.IDEALIZED_DEBUG)
        self.assertFalse(gm.releases)

    def test_single_limb_reach_rh_transition(self):
        import numpy as np
        import mujoco

        data = mujoco.MjData(self.model)
        gm = initialize_episode(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)
        result = simulate_single_limb_reach(
            self.model,
            data,
            self.scene,
            self.base_profile,
            limb=Limb.RIGHT_HAND,
            from_region_id="H4",
            to_region_id="H5",
            steps=210,
            max_attach_distance=.15,
            manager=gm,
        )
        self.assertTrue(result.finite, "Simulation diverged")
        self.assertTrue(result.supported, f"Humanoid dropped (final_z={result.final_root_z:.3f})")
        self.assertTrue(result.eligibility_detected, "Eligibility was never detected during reach")
        self.assertTrue(result.reattached, "Limb failed to reattach to target hold")

        # Verify hold-to-hold semantics: start hold != target hold, displacement >= 0.15m
        self.assertNotEqual(result.from_region_id, result.to_region_id)
        pos_h4 = np.array(self.scene.region(result.from_region_id).position)
        pos_h5 = np.array(self.scene.region(result.to_region_id).position)
        hold_disp = float(np.linalg.norm(pos_h5 - pos_h4))
        self.assertGreaterEqual(hold_disp, 0.15, f"Hold displacement {hold_disp:.3f}m must be >= 0.15m")

        expected_phases = (
            TransitionPhase.INITIAL_STANCE.value,
            TransitionPhase.PRE_SHIFT.value,
            TransitionPhase.RELEASE_LIMB.value,
            TransitionPhase.SUPPORT_PHASE.value,
            TransitionPhase.REACH_PHASE.value,
            TransitionPhase.ATTACH_PHASE.value,
            TransitionPhase.STABILIZED_STANCE.value,
        )
        self.assertEqual(result.phase_history, expected_phases)

        # Debug constraint reactions, not physical support evidence.
        self.assertGreater(result.attachment_loads[Limb.LEFT_HAND], 50.0)
        self.assertGreater(result.attachment_loads[Limb.RIGHT_HAND], 50.0)

    def test_single_limb_reach_lh_transition(self):
        import mujoco

        data = mujoco.MjData(self.model)
        gm = initialize_episode(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)
        prerequisites = execute_transition_sequence(self.model, data, self.scene, self.base_profile, [
            debug_request(Limb.RIGHT_HAND, "H4", "H5"),
            debug_request(Limb.LEFT_FOOT, "H1", "H6"),
        ], manager=gm)
        self.assertTrue(prerequisites.success)
        result = simulate_single_limb_reach(
            self.model,
            data,
            self.scene,
            self.base_profile,
            limb=Limb.LEFT_HAND,
            from_region_id="H3",
            to_region_id="H7",
            steps=210,
            max_attach_distance=.15,
            manager=gm,
        )
        self.assertTrue(result.finite)
        self.assertTrue(result.supported)
        self.assertTrue(result.eligibility_detected)
        self.assertTrue(result.reattached)
        self.assertEqual(
            result.phase_history,
            (
                TransitionPhase.INITIAL_STANCE.value,
                TransitionPhase.PRE_SHIFT.value,
                TransitionPhase.RELEASE_LIMB.value,
                TransitionPhase.SUPPORT_PHASE.value,
                TransitionPhase.REACH_PHASE.value,
                TransitionPhase.ATTACH_PHASE.value,
                TransitionPhase.STABILIZED_STANCE.value,
            ),
        )

    def test_biomechanical_knee_hinge_semantics(self):
        """Verify knee joints have anatomical posterior flexion axis [-1, 0, 0] and positive stance flexion."""
        import mujoco
        from boulder_v1.grasp import STATIC_STANCE_TARGETS

        for kname in ("left_knee", "right_knee"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, kname)
            self.assertGreaterEqual(jid, 0)
            axis = list(self.model.jnt_axis[jid])
            self.assertEqual(axis, [-1.0, 0.0, 0.0], f"{kname} must flex posteriorly around [-1, 0, 0]")
            self.assertGreater(
                STATIC_STANCE_TARGETS[kname],
                0.8,
                f"{kname} in climbing stance must have positive flexion angle",
            )

    def test_passive_foot_demo_preshift_load_regression(self):
        """Historical passive-foot demo regression, not the generalized executor."""
        import mujoco
        from boulder_v1.grasp import setup_static_stance
        from boulder_v1.retargeter import StanceSpecification, solve_retargeted_stance

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile)

        # Initial 4-point stance settling
        spec_4pt = StanceSpecification(pelvis_wall_distance=0.52)
        ret_4pt = solve_retargeted_stance(self.model, self.scene, self.base_profile, spec=spec_4pt)
        for _ in range(24):
            compute_pose_control(self.model, data, target_pose=ret_4pt.target_pose, kp=30.0, kd=3.0)
            mujoco.mj_step(self.model, data)
        init_rh_load = gm.get_grasp_load(Limb.RIGHT_HAND)
        self.assertGreater(init_rh_load, 100.0)

        # Pre-shift execution
        spec_preshift = StanceSpecification(
            pelvis_wall_distance=0.52,
            pelvis_lateral_bias=-0.16,
            torso_orientation=(-0.08, -0.10, -0.05),
            hand_holds={Limb.LEFT_HAND: "H3", Limb.RIGHT_HAND: "H4"},
        )
        ret_preshift = solve_retargeted_stance(self.model, self.scene, self.base_profile, spec=spec_preshift)
        for step in range(32):
            alpha = step / 31.0
            t = {k: (1.0 - alpha) * ret_4pt.target_pose[k] + alpha * ret_preshift.target_pose[k] for k in ret_4pt.target_pose}
            compute_pose_control(self.model, data, target_pose=t, kp=30.0, kd=3.0)
            if alpha > 0.5:
                scale = 1.0 - (alpha - 0.5) / 0.5 * 0.8
                for aid in range(self.model.nu):
                    jid = int(self.model.actuator_trnid[aid, 0])
                    jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and "right_" in jname and ("shoulder" in jname or "elbow" in jname or "wrist" in jname):
                        data.ctrl[aid] *= scale
            mujoco.mj_step(self.model, data)

        preshift_rh_load = gm.get_grasp_load(Limb.RIGHT_HAND)
        load_reduction = (init_rh_load - preshift_rh_load) / init_rh_load

        # Assert load reduction on right hand is strictly greater than 20%
        self.assertGreater(load_reduction, 0.20, f"RH load reduction {load_reduction*100:.1f}% must be > 20%")

    def test_single_limb_reach_profile_variations(self):
        """Historical idealized fixture outcomes, not guaranteed all-profile success."""
        import mujoco

        from scripts.render_demo import PROFILES

        for profile in PROFILES.values():
            xml = build_mjcf(self.scene, profile, contact_mode=ContactMode.IDEALIZED_DEBUG)
            model = compile_model(xml)
            data = mujoco.MjData(model)
            gm = initialize_episode(model, data, self.scene, profile=profile, attach_feet=True)
            source = model.equality("grasp_right_hand_H4").id
            target = model.equality("grasp_right_hand_H5").id
            self.assertTrue(data.eq_active[source])
            self.assertFalse(data.eq_active[target])
            with mock.patch.object(mujoco, "mj_resetData", side_effect=AssertionError("Unexpected episode reset")), \
                 mock.patch.object(mujoco, "mj_step", wraps=mujoco.mj_step) as integration:
                res = simulate_single_limb_reach(
                    model,
                    data,
                    self.scene,
                    profile,
                    limb=Limb.RIGHT_HAND,
                    from_region_id="H4",
                    to_region_id="H5",
                    steps=210,
                    max_attach_distance=.15,
                    manager=gm,
                )
            self.assertTrue(res.finite, f"Profile {profile.name} diverged")
            self.assertTrue(res.supported, f"Profile {profile.name} dropped")
            self.assertEqual(gm.mode, ContactMode.IDEALIZED_DEBUG)
            self.assertEqual((res.limb, res.from_region_id, res.to_region_id), (Limb.RIGHT_HAND, "H4", "H5"))
            self.assertEqual(res.steps, 210)
            self.assertEqual(res.steps, integration.call_count)
            self.assertAlmostEqual(res.duration, res.steps * model.opt.timestep)
            self.assertEqual(res.time, data.time)
            self.assertAlmostEqual(res.duration, data.time)
            self.assertEqual(res.final_root_z, data.qpos[2])
            self.assertEqual(res.phase_history, tuple(p.value for p in TransitionPhase))
            self.assertFalse(data.eq_active[source])
            self.assertEqual(res.reattached, gm.contact_configuration().get(Limb.RIGHT_HAND) == "H5")
            self.assertGreater(res.attachment_loads[Limb.LEFT_HAND], 50.0)
            if profile.name == "long_reach_lower_grip":
                self.assertFalse(res.success)
                self.assertEqual(res.status, TransitionStatus.ATTACH_FAILURE)
                self.assertFalse(res.eligibility_detected)
                self.assertFalse(res.reattached)
                self.assertFalse(data.eq_active[target])
                expected = {Limb.LEFT_HAND: "H3", Limb.LEFT_FOOT: "H1", Limb.RIGHT_FOOT: "H2"}
                self.assertEqual(gm.contact_configuration(), expected)
                self.assertEqual({model.equality(i).name for i in range(model.neq) if data.eq_active[i]},
                                 {"grasp_left_hand_H3", "grasp_left_foot_H1", "grasp_right_foot_H2"})
                self.assertFalse(gm.is_attached(Limb.RIGHT_HAND))
                self.assertIsNone(gm.active_attachment(Limb.RIGHT_HAND))
                self.assertNotIn(Limb.RIGHT_HAND, res.attachment_loads)
                self.assertEqual(gm.get_grasp_load(Limb.RIGHT_HAND), 0.)
                snapshot = gm.contact_snapshot()
                self.assertEqual(set(snapshot.supporting_limbs), set(expected))
                self.assertFalse(snapshot.hands[Limb.RIGHT_HAND].active)
                for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
                    self.assertEqual(snapshot.feet[limb].status, FootStatus.IDEALIZED)
                    self.assertEqual(snapshot.feet[limb].idealized_attachment, expected[limb])
                    self.assertFalse(snapshot.feet[limb].supporting)
                self.assertGreater(res.final_root_z, .85)
            else:
                self.assertTrue(res.eligibility_detected, f"Profile {profile.name} eligibility failed")
                self.assertTrue(res.reattached, f"Profile {profile.name} reattachment failed")
                self.assertTrue(data.eq_active[target])
                self.assertGreater(res.attachment_loads[Limb.RIGHT_HAND], 50.0)

    def test_motion_state_changes_only_inside_live_integration(self):
        """All live qpos/qvel changes come from mj_step, not waypoint reconstruction."""
        import mujoco

        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        last = [self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time)]
        real_step = mujoco.mj_step

        def step(model, data):
            self.assertIs(model, self.model)
            self.assertIs(data, self.data)
            np.testing.assert_array_equal(data.qpos, last[0])
            np.testing.assert_array_equal(data.qvel, last[1])
            self.assertEqual(data.time, last[2])
            real_step(model, data)
            last[:] = [data.qpos.copy(), data.qvel.copy(), float(data.time)]

        with mock.patch.object(mujoco, "mj_step", side_effect=step) as integration:
            result = execute_transition(self.model, self.data, self.scene, self.base_profile,
                                        debug_request(Limb.RIGHT_HAND, "H4", "H5"), manager=gm)
        self.assertTrue(result.success)
        self.assertEqual(integration.call_count, result.steps)
        np.testing.assert_array_equal(result.final_state.qpos, last[0])
        np.testing.assert_array_equal(result.final_state.qvel, last[1])

    def test_transition_request_validation(self):
        """Verify explicit validation errors for invalid transition requests."""
        import mujoco

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)
        contacts = dict(self.scene.start_configuration)

        # 1. Source equals target
        req_same = debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H4")
        res_same = execute_transition(self.model, data, self.scene, self.base_profile, req_same, manager=gm, current_contacts=contacts)
        self.assertFalse(res_same.success)
        self.assertEqual(res_same.status, TransitionStatus.INVALID_REQUEST)

        # 2. Non-positive steps
        req_steps = debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H5", steps=0)
        res_steps = execute_transition(self.model, data, self.scene, self.base_profile, req_steps, manager=gm, current_contacts=contacts)
        self.assertFalse(res_steps.success)
        self.assertEqual(res_steps.status, TransitionStatus.INVALID_REQUEST)

        # 3. Source hold not currently attached
        req_not_att = debug_request(limb=Limb.RIGHT_HAND, source_hold="H5", target_hold="H4")
        res_not_att = execute_transition(self.model, data, self.scene, self.base_profile, req_not_att, manager=gm, current_contacts=contacts)
        self.assertFalse(res_not_att.success)
        self.assertEqual(res_not_att.status, TransitionStatus.SOURCE_NOT_ATTACHED)

        # 4. Ineligible target: unknown hold
        req_unknown = debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H_NONEXISTENT")
        res_unknown = execute_transition(self.model, data, self.scene, self.base_profile, req_unknown, manager=gm, current_contacts=contacts)
        self.assertFalse(res_unknown.success)
        self.assertEqual(res_unknown.status, TransitionStatus.INELIGIBLE_TARGET)

    def test_bilateral_hand_transitions(self):
        """Verify bilateral hand transitions: RH H4->H5 and LH H3->H7 using generalized machinery."""
        import mujoco
        import numpy as np

        # Test RH transition H4 -> H5
        data_rh = mujoco.MjData(self.model)
        gm_rh = setup_static_stance(self.model, data_rh, self.scene, profile=self.base_profile, attach_feet=True)
        req_rh = debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H5", steps=210)
        res_rh = execute_transition(
            self.model, data_rh, self.scene, self.base_profile, req_rh, manager=gm_rh, current_contacts=dict(self.scene.start_configuration)
        )
        self.assertTrue(res_rh.success, f"RH transition failed: {res_rh.reason}")
        self.assertEqual(res_rh.status, TransitionStatus.SUCCESS)
        self.assertGreaterEqual(res_rh.displacement, 0.15)
        self.assertTrue(gm_rh.is_attached(Limb.RIGHT_HAND))
        self.assertEqual(len(res_rh.phases_traversed), 7)
        self.assert_actual_result(res_rh, self.model, data_rh, gm_rh)

        # Test LH transition H3 -> H7 (using the exact same generalized execute_transition)
        data_lh = mujoco.MjData(self.model)
        gm_lh = setup_static_stance(self.model, data_lh, self.scene, profile=self.base_profile, attach_feet=True)
        prerequisites = execute_transition_sequence(self.model, data_lh, self.scene, self.base_profile, [
            debug_request(Limb.RIGHT_HAND, "H4", "H5"),
            debug_request(Limb.LEFT_FOOT, "H1", "H6"),
        ], manager=gm_lh)
        self.assertTrue(prerequisites.success)
        req_lh = debug_request(limb=Limb.LEFT_HAND, source_hold="H3", target_hold="H7", steps=210)
        res_lh = execute_transition(
            self.model, data_lh, self.scene, self.base_profile, req_lh, manager=gm_lh, current_contacts=gm_lh.contact_configuration()
        )
        self.assertTrue(res_lh.success, f"LH transition failed: {res_lh.reason}")
        self.assertEqual(res_lh.status, TransitionStatus.SUCCESS)
        self.assertGreaterEqual(res_lh.displacement, 0.15)
        self.assertTrue(gm_lh.is_attached(Limb.LEFT_HAND))
        self.assertEqual(len(res_lh.phases_traversed), 7)
        self.assert_actual_result(res_lh, self.model, data_lh, gm_lh)

    def test_foot_reposition_foundation(self):
        """Verify foot reposition transition: LF H1->H6 release, 3-point support, step site attach."""
        import mujoco

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)
        req = debug_request(limb=Limb.LEFT_FOOT, source_hold="H1", target_hold="H6", steps=210)
        res = execute_transition(
            self.model, data, self.scene, self.base_profile, req, manager=gm, current_contacts=dict(self.scene.start_configuration)
        )
        self.assertTrue(res.success, f"LF foot reposition failed: {res.reason}")
        self.assertEqual(res.status, TransitionStatus.SUCCESS)
        self.assertGreaterEqual(res.displacement, 0.15)
        self.assertTrue(gm.is_attached(Limb.LEFT_FOOT))
        self.assertGreater(res.final_state.root_pos[2], 1.15, "Climber root dropped significantly during foot reposition")
        self.assertEqual(len(res.phases_traversed), 7)
        self.assert_actual_result(res, self.model, data, gm)

    def test_stabilization_readiness_criterion(self):
        """Verify stabilization readiness checking root/joint velocities and support count."""
        import mujoco

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)
        settled = execute_transition(self.model, data, self.scene, self.base_profile,
                                     debug_request(Limb.RIGHT_HAND, "H4", "H5"), manager=gm)
        self.assertTrue(settled.success, settled.reason)

        is_stable, reason = check_stabilization_readiness(self.model, data, gm)
        self.assertTrue(is_stable, f"Settled stance should be stable: {reason}")

        # Inject artificial high root velocity
        data.qvel[0] = 2.0
        is_stable_vel, reason_vel = check_stabilization_readiness(self.model, data, gm)
        self.assertFalse(is_stable_vel)
        self.assertIn("Root linear velocity", reason_vel)

        data.qvel[:] = 0
        gm.detach(Limb.LEFT_FOOT)
        gm.detach(Limb.RIGHT_FOOT)
        is_stable_support, reason_support = check_stabilization_readiness(self.model, data, gm)
        self.assertFalse(is_stable_support)
        self.assertIn("Active support count", reason_support)
        data.eq_active[:] = False
        with self.assertRaisesRegex(AttachmentStateError, "registry disagrees"):
            check_stabilization_readiness(self.model, data, gm)

    def test_multi_move_sequence_without_reset(self):
        """Verify multi-move sequence execution without simulation reset across 3 continuous moves."""
        import mujoco
        import numpy as np

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)

        moves = [
            debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H5", steps=210),
            debug_request(limb=Limb.LEFT_FOOT, source_hold="H1", target_hold="H6", steps=210),
            debug_request(limb=Limb.LEFT_HAND, source_hold="H3", target_hold="H7", steps=210),
        ]

        boundaries = {}

        def observe(event, live_data, manager):
            self.assertIsNot(live_data, data)
            self.assertIsNot(manager, gm)
            self.assertIs(manager.data, live_data)
            if event.steps == 0 or event.result is not None:
                boundaries[event.move_index, event.result is not None] = self.snapshot(live_data, manager)
                self.assert_snapshot_equal(boundaries[event.move_index, event.result is not None], self.snapshot(data, gm))

        with mock.patch.object(mujoco, "mj_resetData", side_effect=AssertionError("Unexpected episode reset")):
            seq_res = execute_transition_sequence(
                self.model, data, self.scene, self.base_profile, moves, manager=gm, frame_callback=observe)

        self.assertTrue(seq_res.success, f"Sequence failed to execute: {seq_res.transition_results[-1].reason if seq_res.transition_results else 'empty'}")
        self.assertEqual(seq_res.completed_moves, 3)
        self.assertIsNone(seq_res.failed_move_index)
        self.assertEqual(len(seq_res.transition_results), 3)

        # Verify STRICT simulation continuity between moves (NO mj_resetData, NO teleport)
        r0 = seq_res.transition_results[0]
        r1 = seq_res.transition_results[1]
        r2 = seq_res.transition_results[2]

        for index, (previous, following) in enumerate(((r0, r1), (r1, r2))):
            self.assertEqual(previous.final_state.time, following.initial_state.time)
            for field in ("qpos", "qvel", "eq_active", "ctrl", "qacc_warmstart"):
                np.testing.assert_array_equal(getattr(previous.final_state, field), getattr(following.initial_state, field))
            self.assertEqual(previous.final_state.attachment_constraints, following.initial_state.attachment_constraints)
            self.assertEqual(previous.final_state.contact_configuration, following.initial_state.contact_configuration)
            for field in ("contact_mode", "hand_states", "foot_states", "capture_events", "release_events"):
                self.assertEqual(getattr(previous.final_state, field), getattr(following.initial_state, field))
            self.assert_snapshot_equal(boundaries[index, True], boundaries[index + 1, False])

        # Final contact configuration verification
        expected_contacts = {
            Limb.RIGHT_HAND: "H5",
            Limb.LEFT_FOOT: "H6",
            Limb.LEFT_HAND: "H7",
            Limb.RIGHT_FOOT: "H2",
        }
        self.assertEqual(seq_res.final_contact_configuration, expected_contacts)

        # Verify all 4 limbs are actively attached in GraspManager at end of sequence
        for limb in Limb:
            self.assertTrue(gm.is_attached(limb), f"Limb {limb.value} not attached at sequence end")

        # Climber remains suspended in realistic climbing zone
        self.assertGreater(seq_res.final_state_summary.root_pos[2], 1.15)
        self.assertTrue(seq_res.final_state_summary.finite)
        self.assertEqual(seq_res.total_steps, 750)
        self.assertAlmostEqual(seq_res.total_time, 1.5)
        self.assertEqual([r.steps for r in seq_res.transition_results], [250, 250, 250])
        self.assert_actual_result(r2, self.model, data, gm)

    def test_sequence_failure_propagation(self):
        """Verify that an invalid move in a sequence halts execution immediately without running subsequent moves."""
        import mujoco

        data = mujoco.MjData(self.model)
        gm = setup_static_stance(self.model, data, self.scene, profile=self.base_profile, attach_feet=True)

        # Move 2 targets a non-existent hold
        moves = [
            debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H5", steps=210),
            debug_request(limb=Limb.LEFT_FOOT, source_hold="H1", target_hold="H_INVALID", steps=210),
            debug_request(limb=Limb.LEFT_HAND, source_hold="H3", target_hold="H7", steps=210),
        ]

        seq_res = execute_transition_sequence(
            self.model,
            data,
            self.scene,
            self.base_profile,
            moves,
            manager=gm,
        )

        self.assertFalse(seq_res.success)
        self.assertEqual(seq_res.completed_moves, 1)
        self.assertEqual(seq_res.failed_move_index, 1)
        # Move 3 was NEVER executed
        self.assertEqual(len(seq_res.transition_results), 2)
        self.assertEqual(seq_res.transition_results[0].status, TransitionStatus.SUCCESS)
        self.assertEqual(seq_res.transition_results[1].status, TransitionStatus.INELIGIBLE_TARGET)

    def test_idealized_debug_profile_sequence_outcomes(self):
        """Historical idealized fixture outcomes, not guaranteed all-profile success."""
        import mujoco
        from scripts.render_demo import PROFILES

        moves = [
            debug_request(limb=Limb.RIGHT_HAND, source_hold="H4", target_hold="H5", steps=210),
            debug_request(limb=Limb.LEFT_FOOT, source_hold="H1", target_hold="H6", steps=210),
            debug_request(limb=Limb.LEFT_HAND, source_hold="H3", target_hold="H7", steps=210),
        ]

        for profile in PROFILES.values():
            xml = build_mjcf(self.scene, profile, contact_mode=ContactMode.IDEALIZED_DEBUG)
            model = compile_model(xml)
            data = mujoco.MjData(model)
            gm = setup_static_stance(model, data, self.scene, profile=profile, attach_feet=True)

            with mock.patch.object(mujoco, "mj_resetData", side_effect=AssertionError("Unexpected episode reset")), \
                 mock.patch.object(mujoco, "mj_step", wraps=mujoco.mj_step) as integration:
                seq_res = execute_transition_sequence(model, data, self.scene, profile, moves, manager=gm)
            self.assert_actual_result(seq_res.transition_results[-1], model, data, gm)
            self.assertEqual(seq_res.final_contact_configuration, gm.contact_configuration())
            self.assertEqual(seq_res.total_steps, integration.call_count)
            self.assertAlmostEqual(seq_res.total_time, seq_res.total_steps * model.opt.timestep)
            self.assertTrue(seq_res.final_state_summary.finite)
            if profile.name == "long_reach_lower_grip":
                self.assertFalse(seq_res.success)
                self.assertEqual(seq_res.completed_moves, 0)
                self.assertEqual(seq_res.failed_move_index, 0)
                self.assertEqual(len(seq_res.transition_results), 1)
                result = seq_res.transition_results[0]
                self.assertEqual(result.status, TransitionStatus.ATTACH_FAILURE)
                self.assertTrue(result.released)
                self.assertFalse(result.eligibility_detected)
                self.assertFalse(result.target_captured)
                self.assertEqual(result.phases_traversed, tuple(p.value for p in TransitionPhase))
                source = model.equality("grasp_right_hand_H4").id
                target = model.equality("grasp_right_hand_H5").id
                self.assertTrue(result.initial_state.eq_active[source])
                self.assertFalse(result.initial_state.eq_active[target])
                self.assertFalse(data.eq_active[source])
                self.assertFalse(data.eq_active[target])
                self.assertEqual(seq_res.final_contact_configuration,
                                 {Limb.LEFT_HAND: "H3", Limb.LEFT_FOOT: "H1", Limb.RIGHT_FOOT: "H2"})
                self.assertEqual(set(result.final_state.attachment_constraints),
                                 {Limb.LEFT_HAND, Limb.LEFT_FOOT, Limb.RIGHT_FOOT})
                self.assertNotIn(Limb.RIGHT_HAND, result.final_state.attachment_loads)
                self.assertEqual(result.steps, 210)
                self.assertEqual(seq_res.total_steps, result.steps)
                self.assertEqual(seq_res.total_time, result.duration)
                self.assertAlmostEqual(seq_res.total_time, .42)
                self.assertEqual(seq_res.final_state_summary, result.final_state)
                self.assertGreater(result.final_state.root_pos[2], .85)
                self.assertFalse(result.clock_discontinuity)
            else:
                self.assertTrue(seq_res.success, f"{profile.name}: {seq_res.transition_results[-1].reason}")
                self.assertEqual(seq_res.completed_moves, 3)
                self.assertEqual(seq_res.total_steps, 750)
                self.assertGreater(seq_res.final_state_summary.root_pos[2], 1.15)
                self.assertIsNone(seq_res.failed_move_index)
                self.assertEqual([r.steps for r in seq_res.transition_results], [250, 250, 250])
                for previous, following in zip(seq_res.transition_results, seq_res.transition_results[1:]):
                    self.assertEqual(previous.final_state, following.initial_state)

    def test_invalid_requests_leave_complete_live_state_unchanged(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        self.data.time = 17.25
        self.data.qpos[0] += 0.03
        self.data.qvel[:] = np.linspace(-0.1, 0.1, self.model.nv)
        self.data.ctrl[:] = np.linspace(-0.5, 0.5, self.model.nu)
        self.data.qacc_warmstart[:] = np.arange(self.model.nv)
        request = debug_request(Limb.RIGHT_HAND, "H4", "H5")
        cases = [
            replace(request, steps=0), replace(request, steps=1.5),
            replace(request, target_hold="H4"), replace(request, kp=float("nan")),
            replace(request, kd=-1), replace(request, max_attach_distance=float("inf")),
            replace(request, settle_steps_after=-1), replace(request, limb="RIGHT_HAND"),
            replace(request, source_contact_configuration={Limb.RIGHT_HAND: "H5"}),
            replace(request, target_contact_configuration=dict(self.scene.start_configuration)),
        ]
        for invalid in cases:
            with self.subTest(request=invalid):
                before = self.snapshot(self.data, gm)
                result = execute_transition(self.model, self.data, self.scene, self.base_profile, invalid, manager=gm)
                self.assert_snapshot_equal(before, self.snapshot(self.data, gm))
                self.assertEqual(result.status, TransitionStatus.INVALID_REQUEST)
                self.assertEqual(result.steps, 0)
                self.assertEqual(result.duration, 0)
                self.assertEqual(result.time, 17.25)
                self.assertEqual(result.phases_traversed, ())
                self.assertEqual(result.initial_state, result.final_state)

    def test_missing_uninitialized_or_wrong_session_never_resets(self):
        import mujoco

        request = debug_request(Limb.RIGHT_HAND, "H4", "H5")
        wrong_data = mujoco.MjData(self.model)
        for manager in (None, GraspManager(self.model, self.data, self.scene),
                        initialize_episode(self.model, wrong_data, self.scene, profile=self.base_profile, attach_feet=True)):
            for execute in (execute_transition, execute_transition_sequence, simulate_single_limb_reach):
                with self.subTest(manager=manager, execute=execute.__name__):
                    before = self.snapshot(self.data, manager)
                    with self.assertRaises(AttachmentStateError):
                        if execute is execute_transition:
                            execute(self.model, self.data, self.scene, self.base_profile, request, manager=manager)
                        elif execute is execute_transition_sequence:
                            execute(self.model, self.data, self.scene, self.base_profile, [request], manager=manager)
                        else:
                            execute(self.model, self.data, self.scene, self.base_profile, manager=manager)
                    self.assert_snapshot_equal(before, self.snapshot(self.data, manager))

        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        before = self.snapshot(self.data, gm)
        with self.assertRaisesRegex(ValueError, "at least one"):
            execute_transition_sequence(self.model, self.data, self.scene, self.base_profile, [], manager=gm)
        self.assert_snapshot_equal(before, self.snapshot(self.data, gm))

    def test_source_and_configuration_assertions_use_live_attachments(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        request = debug_request(Limb.RIGHT_HAND, "H4", "H5")
        before = self.snapshot(self.data, gm)
        stale = dict(gm.contact_configuration())
        stale[Limb.LEFT_FOOT] = "H6"
        result = execute_transition(self.model, self.data, self.scene, self.base_profile, request,
                                    manager=gm, current_contacts=stale)
        self.assertEqual(result.status, TransitionStatus.INVALID_REQUEST)
        self.assert_snapshot_equal(before, self.snapshot(self.data, gm))
        gm.detach(Limb.RIGHT_HAND)
        before = self.snapshot(self.data, gm)
        result = execute_transition(self.model, self.data, self.scene, self.base_profile, request,
                                    manager=gm, current_contacts=dict(self.scene.start_configuration))
        self.assertEqual(result.status, TransitionStatus.SOURCE_NOT_ATTACHED)
        self.assertNotIn(Limb.RIGHT_HAND, result.final_contact_configuration)
        self.assert_snapshot_equal(before, self.snapshot(self.data, gm))

    def test_inactive_target_topology_is_rejected_before_mutation(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        target = self.model.equality("grasp_right_hand_H5").id
        self.model.eq_obj2id[target] = self.model.site("site_H3").id
        before = self.snapshot(self.data, gm)
        with self.assertRaisesRegex(AttachmentStateError, "incorrect type or sites"):
            execute_transition(self.model, self.data, self.scene, self.base_profile,
                               debug_request(Limb.RIGHT_HAND, "H4", "H5"), manager=gm)
        self.assert_snapshot_equal(before, self.snapshot(self.data, gm))

    def test_observer_model_data_and_results_are_detached_from_execution(self):
        request = debug_request(Limb.RIGHT_HAND, "H4", "H5")
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        baseline = execute_transition(self.model, self.data, self.scene, self.base_profile, request, manager=gm)
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)

        def hostile_observer(event, observed_data, observed_manager):
            self.assertIsNot(observed_data, self.data)
            self.assertIsNot(observed_manager.model, self.model)
            self.assertIsNot(observed_manager.scene, self.scene)
            self.assertEqual(observed_manager.profile, gm.profile)
            self.assertIsNot(observed_manager.profile, gm.profile)
            self.assertEqual(event.state.contact_mode, ContactMode.IDEALIZED_DEBUG)
            observed_data.qacc[:] = float("nan")
            observed_data.qpos[:] = float("nan")
            observed_manager.model.opt.gravity[:] = 0
            with self.assertRaises(TypeError):
                observed_manager.scene.start_configuration[Limb.LEFT_HAND] = "H7"
            observed_manager.detach_all()
            event.state.contact_configuration.clear()
            event.state.hand_states.clear()
            event.state.foot_states.clear()
            for capture in event.state.capture_events:
                capture["mode"] = "observer corruption"
            if event.result is not None:
                event.result.final_contact_configuration[Limb.RIGHT_HAND] = "H4"
                event.result.final_state.contact_configuration[Limb.RIGHT_HAND] = "H4"

        result = execute_transition(self.model, self.data, self.scene, self.base_profile,
                                    request, manager=gm, frame_callback=hostile_observer)
        self.assertEqual(result, baseline)
        self.assertTrue(np.isfinite(self.data.qacc).all())
        self.assertEqual(self.model.opt.gravity[2], -9.81)
        self.assertEqual(len(self.scene.start_configuration), 4)
        self.assert_actual_result(result, self.model, self.data, gm)

    def test_short_budgets_cannot_claim_target_or_completion(self):
        for steps in (1, 28, 40):
            with self.subTest(steps=steps):
                gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
                result = execute_transition(self.model, self.data, self.scene, self.base_profile,
                                            debug_request(Limb.RIGHT_HAND, "H4", "H5", steps=steps), manager=gm)
                self.assertFalse(result.success)
                self.assertEqual(result.status, TransitionStatus.INCOMPLETE)
                self.assertEqual(result.steps, steps)
                self.assertAlmostEqual(result.duration, steps * self.model.opt.timestep)
                self.assertFalse(result.target_captured)
                self.assert_actual_result(result, self.model, self.data, gm)
                if steps <= 28:
                    self.assertFalse(result.released)
                    self.assertEqual(result.final_contact_configuration[Limb.RIGHT_HAND], "H4")
                else:
                    self.assertTrue(result.released)
                    self.assertNotIn(Limb.RIGHT_HAND, result.final_contact_configuration)

        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        legacy = simulate_single_limb_reach(self.model, self.data, self.scene, self.base_profile, steps=1, manager=gm)
        self.assertEqual(legacy.status, TransitionStatus.INCOMPLETE)
        self.assertFalse(legacy.reattached)
        self.assertFalse(legacy.eligibility_detected)

    def test_actual_duration_includes_settling_at_nonzero_start_time(self):
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        self.data.time = 7.0
        result = execute_transition(self.model, self.data, self.scene, self.base_profile,
                                    debug_request(Limb.RIGHT_HAND, "H4", "H5"), manager=gm)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 250)
        self.assertAlmostEqual(result.duration, 0.5)
        self.assertAlmostEqual(result.time, 7.5)
        self.assertEqual(result.duration, result.final_state.time - result.initial_state.time)
        self.assertFalse(result.clock_discontinuity)
        self.assert_actual_result(result, self.model, self.data, gm)

    def test_physics_failure_stops_sequence_and_preserves_actual_endpoint(self):
        import boulder_v1.locomotion as locomotion

        moves = [debug_request(Limb.RIGHT_HAND, "H4", "H5"),
                 debug_request(Limb.LEFT_FOOT, "H1", "H6"),
                 debug_request(Limb.LEFT_HAND, "H3", "H7")]
        gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
        with mock.patch.object(locomotion, "check_stabilization_readiness", return_value=(False, "forced endpoint rejection")):
            sequence = execute_transition_sequence(self.model, self.data, self.scene, self.base_profile, moves, manager=gm)
        self.assertFalse(sequence.success)
        self.assertEqual(sequence.failed_move_index, 0)
        self.assertEqual(len(sequence.transition_results), 1)
        result = sequence.transition_results[0]
        self.assertEqual(result.status, TransitionStatus.UNSTABLE_FINAL_STATE)
        self.assertTrue(result.target_captured)
        self.assertEqual(sequence.final_contact_configuration[Limb.RIGHT_HAND], "H5")
        self.assertEqual(sequence.final_contact_configuration, result.final_state.contact_configuration)
        self.assertEqual(sequence.total_steps, 250)
        self.assert_actual_result(result, self.model, self.data, gm)

    def test_nonfinite_main_or_settling_state_never_succeeds(self):
        import mujoco

        original_step = mujoco.mj_step
        for failure_step in (10, 211):
            with self.subTest(failure_step=failure_step):
                gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
                count = 0

                def corrupt(model, data):
                    nonlocal count
                    original_step(model, data)
                    count += 1
                    if count == failure_step:
                        data.qpos[0] = float("nan")

                with mock.patch.object(mujoco, "mj_step", side_effect=corrupt):
                    result = execute_transition(self.model, self.data, self.scene, self.base_profile,
                                                debug_request(Limb.RIGHT_HAND, "H4", "H5"), manager=gm)
                self.assertEqual(result.status, TransitionStatus.NONFINITE_STATE)
                self.assertFalse(result.success)
                self.assertFalse(result.final_state.finite)
                self.assertEqual(result.steps, failure_step)
                self.assertTrue(np.isnan(self.data.qpos[0]))

        # Native recovery resets constraints; it must yield a terminal result,
        # not raise registry drift before numerical failure is reported.
        for start_time, recovery_step in ((0.0, 10), (7.0, 1)):
            with self.subTest(start_time=start_time, recovery_step=recovery_step):
                gm = initialize_episode(self.model, self.data, self.scene, profile=self.base_profile, attach_feet=True)
                self.data.time = start_time
                count = 0
                events = []

                def recover(model, data):
                    nonlocal count
                    count += 1
                    if count == recovery_step:
                        data.qfrc_applied[0] = 1e30
                    original_step(model, data)

                moves = [debug_request(Limb.RIGHT_HAND, "H4", "H5"),
                         debug_request(Limb.LEFT_FOOT, "H1", "H6")]
                with mock.patch.object(mujoco, "mj_step", side_effect=recover):
                    sequence = execute_transition_sequence(
                        self.model, self.data, self.scene, self.base_profile, moves,
                        manager=gm, frame_callback=lambda event, data, manager: events.append(event))
                self.assertEqual(len(sequence.transition_results), 1)
                result = sequence.transition_results[0]
                self.assertEqual(result.status, TransitionStatus.NONFINITE_STATE)
                self.assertEqual(result.steps, recovery_step)
                self.assertAlmostEqual(result.duration, recovery_step * self.model.opt.timestep)
                self.assertAlmostEqual(sequence.total_time, result.duration)
                self.assertAlmostEqual(result.time, 0.002)
                self.assertTrue(result.clock_discontinuity)
                self.assertEqual(events[-1].result, result)
                self.assertEqual(result.final_contact_configuration, {})
                self.assert_actual_result(result, self.model, self.data, gm)


class PhysicalLocomotionTests(unittest.TestCase):
    def test_archived_controller_fixture_rejects_physical_models_without_writes(self):
        fixture = make_mixed_fixture()
        before = get_state_summary(fixture.model, fixture.data, fixture.manager)
        with self.assertRaisesRegex(AssertionError, "IDEALIZED_DEBUG mode 1"):
            archived_normalized_pose_control(fixture.model, fixture.data, {})
        self.assertEqual(get_state_summary(fixture.model, fixture.data, fixture.manager), before)

    def settle_fixture(self, fixture):
        """Benchmark-only native steps, not a production stance initializer."""
        import mujoco

        model, data, manager = fixture.model, fixture.data, fixture.manager
        allocation = verify_fixed_allocation(fixture)
        self.assertTrue(allocation["valid"])
        joints = model.actuator_trnid[:, 0]
        qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        tau = np.array(allocation["tau_ff_Nm"])
        for _ in range(round(1. / model.opt.timestep)):
            data.ctrl[:] = np.clip((tau + 80. * (fixture.reference[qids] - data.qpos[qids])
                                    - data.qvel[dofs]) / model.actuator_gear[:, 0], -1., 1.)
            self.assertTrue(all(d.maintain for d in manager.evaluate_and_update().values()))
            mujoco.mj_step(model, data)
            self.assertTrue(all(d.maintain for d in manager.evaluate_and_update(applied_data=data).values()))

    def test_legacy_profile_sequence_rejects_initial_contact_before_any_live_loop(self):
        import mujoco
        from scripts.render_demo import PROFILES

        scene = make_synthetic_scene()
        for profile in PROFILES.values():
            with self.subTest(profile=profile.name):
                model = compile_model(build_mjcf(scene, profile))
                data = mujoco.MjData(model)
                manager = GraspManager(model, data, scene, profile=profile)
                data.time = 7.
                data.qvel[:] = .1
                data.ctrl[:] = .2
                data.qacc_warmstart[:] = .3
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                before = np.empty(mujoco.mj_stateSize(model, spec))
                after = np.empty_like(before)
                mujoco.mj_getState(model, data, before, spec)
                with mock.patch.object(mujoco, "mj_resetData", side_effect=AssertionError("premature live reset")), \
                     mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("invalid initial contacts entered loop")):
                    with self.assertRaises(InitialContactError):
                        initialize_episode(model, data, scene, profile=profile, manager=manager)
                mujoco.mj_getState(model, data, after, spec)
                np.testing.assert_array_equal(before, after)
                self.assertFalse(manager.initialized)
                self.assertEqual(manager.active_attachments(), {})

    def test_mixed_snapshot_and_executor_use_native_feet_not_attachment_registry(self):
        import mujoco

        fixture = make_mixed_fixture()
        model, data, manager = fixture.model, fixture.data, fixture.manager
        initial = get_state_summary(model, data, manager)
        self.assertEqual(initial.contact_mode, ContactMode.PHYSICAL)
        self.assertEqual(set(initial.attachment_constraints), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        self.assertEqual(len(initial.capture_events), 2)
        self.assertEqual(initial.release_events, ())
        self.assertEqual(set(initial.contact_configuration), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        self.assertTrue(all(not foot.supporting and foot.normal_force == 0. for foot in initial.foot_states.values()))
        self.assertFalse(check_stabilization_readiness(model, data, manager)[0])

        self.settle_fixture(fixture)
        settled = get_state_summary(model, data, manager)
        self.assertEqual(set(settled.contact_configuration), set(Limb))
        self.assertEqual(set(manager.active_attachments()), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        self.assertEqual(set(settled.attachment_loads), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        for limb, state in settled.foot_states.items():
            self.assertEqual(state.status, FootStatus.SUPPORTING)
            self.assertGreater(state.normal_force, 5.)
            self.assertLessEqual(state.tangential_speed, .01)
            self.assertEqual(settled.contact_configuration[limb], state.primary_region)
            self.assertIsNone(state.idealized_attachment)
            self.assertFalse(manager.is_attached(limb))
        for limb, state in settled.hand_states.items():
            self.assertTrue(state.active and state.valid)
            self.assertAlmostEqual(state.load, np.linalg.norm(state.force_world))
            self.assertEqual(state.capacity, fixture.profile.grip_capacity)
            self.assertAlmostEqual(state.margin, state.capacity - state.load)

        request = TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand", steps=1)
        self.assertTrue(request.check_grip)
        self.assertEqual(request.max_attach_distance, .001)
        with mock.patch.object(manager, "evaluate_and_update", wraps=manager.evaluate_and_update) as capacity, \
             mock.patch.object(mujoco, "mj_step", wraps=mujoco.mj_step) as integration:
            result = execute_transition(model, data, fixture.scene, fixture.profile, request, manager=manager)
        self.assertEqual(result.status, TransitionStatus.INCOMPLETE)
        self.assertEqual(result.steps, 1)
        self.assertEqual(integration.call_count, 1)
        self.assertEqual(capacity.call_count, 2)
        self.assertNotIn("applied_data", capacity.call_args_list[0].kwargs)
        self.assertIs(capacity.call_args_list[1].kwargs["applied_data"], data)
        self.assertEqual(result.initial_state, settled)
        self.assertEqual(result.final_state.contact_mode, ContactMode.PHYSICAL)
        for field in ("qpos", "qvel", "eq_active", "ctrl", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(result.final_state, field), getattr(data, field))

        for invalid in (replace(request, check_grip=False), replace(request, max_attach_distance=.15),
                        replace(request, check_grip=1), replace(request, kp=float("nan")),
                        replace(request, settle_steps_after=-1)):
            with self.subTest(invalid=invalid):
                before = get_state_summary(model, data, manager)
                with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("invalid request integrated")):
                    rejected = execute_transition(model, data, fixture.scene, fixture.profile, invalid, manager=manager)
                self.assertEqual(rejected.status, TransitionStatus.INVALID_REQUEST)
                self.assertEqual(rejected.steps, 0)
                self.assertEqual(rejected.initial_state, before)
                self.assertEqual(rejected.final_state, before)

        before = get_state_summary(model, data, manager)
        with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("mismatched profile integrated")):
            rejected = execute_transition(model, data, fixture.scene, replace(fixture.profile, grip_capacity=70.),
                                          request, manager=manager)
        self.assertEqual(rejected.status, TransitionStatus.INVALID_REQUEST)
        self.assertEqual(rejected.final_state, before)
        self.assertEqual(manager.profile, fixture.profile)

    def test_physical_grip_failure_stops_sequence_at_actual_endpoint(self):
        import mujoco
        import boulder_v1.locomotion as locomotion

        fixture = make_mixed_fixture()
        self.settle_fixture(fixture)
        model, data, manager = fixture.model, fixture.data, fixture.manager
        initial = get_state_summary(model, data, manager)
        self.assertEqual(set(initial.contact_configuration), set(Limb))
        moves = [TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand"),
                 TransitionRequest(Limb.LEFT_FOOT, "left_foot", "right_foot")]
        native_control, native_step = locomotion.compute_pose_control, mujoco.mj_step
        control_calls = 0
        overload_call = 5
        endpoint = [data.qpos.copy(), data.qvel.copy(), float(data.time)]

        def control_with_overload(live_model, live_data, **kwargs):
            nonlocal control_calls
            self.assertIs(live_model, model)
            self.assertIs(live_data, data)
            command = native_control(live_model, live_data, **kwargs)
            control_calls += 1
            if control_calls == overload_call:
                # Actual external force on the owned hand, before the capacity guard.
                data.xfrc_applied[model.body("left_hand").id, :3] = [0., -10_000., 0.]
            return command

        def step(live_model, live_data):
            self.assertIs(live_model, model)
            self.assertIs(live_data, data)
            np.testing.assert_array_equal(data.qpos, endpoint[0])
            np.testing.assert_array_equal(data.qvel, endpoint[1])
            self.assertEqual(data.time, endpoint[2])
            native_step(live_model, live_data)
            endpoint[:] = [data.qpos.copy(), data.qvel.copy(), float(data.time)]

        with mock.patch.object(mujoco, "mj_resetData", side_effect=AssertionError("unexpected live reset")), \
             mock.patch.object(locomotion, "compute_pose_control", side_effect=control_with_overload), \
             mock.patch.object(mujoco, "mj_step", side_effect=step) as integration:
            sequence = execute_transition_sequence(model, data, fixture.scene, fixture.profile, moves, manager=manager)
        self.assertFalse(sequence.success)
        self.assertEqual((sequence.completed_moves, sequence.failed_move_index), (0, 0))
        self.assertEqual(len(sequence.transition_results), 1)
        result = sequence.transition_results[0]
        self.assertEqual(result.status, TransitionStatus.GRIP_FAILURE)
        self.assertEqual(result.initial_state, initial)
        self.assertEqual(control_calls, overload_call)
        self.assertEqual(result.steps, overload_call - 1)
        np.testing.assert_array_equal(data.qpos, endpoint[0])
        np.testing.assert_array_equal(data.qvel, endpoint[1])
        self.assertEqual(data.time, endpoint[2])
        self.assertFalse(result.clock_discontinuity)
        self.assertEqual(sequence.total_steps, integration.call_count)
        self.assertEqual(sequence.total_steps, result.steps)
        self.assertAlmostEqual(sequence.total_time, result.steps * model.opt.timestep)
        self.assertEqual(result.duration, data.time - initial.time)
        self.assertEqual(sequence.total_time, result.duration)
        self.assertEqual(result.final_state, sequence.final_state_summary)
        self.assertEqual(sequence.final_state_summary, get_state_summary(model, data, manager))
        self.assertEqual(sequence.final_contact_configuration, manager.contact_configuration())
        self.assertEqual(sequence.final_state_summary.contact_mode, ContactMode.PHYSICAL)
        self.assertTrue(manager.releases)
        self.assertTrue(any(e["limb"] == Limb.LEFT_HAND.value for e in manager.releases))
        for event in manager.releases:
            limb = Limb(event["limb"])
            self.assertTrue(limb.is_hand)
            self.assertGreater(event["required_load_N"], event["capacity_N"])
            self.assertAlmostEqual(event["required_load_N"], np.linalg.norm(event["force_world_N"]))
            self.assertEqual(event["capacity_N"], fixture.profile.grip_capacity * fixture.scene.region(event["region_id"]).grip_quality)
            self.assertFalse(manager.is_attached(limb))
            self.assertFalse(data.eq_active[model.equality(f"grasp_{limb.value.lower()}_{event['region_id']}").id])
        for limb, state in result.final_state.foot_states.items():
            self.assertNotIn(limb, manager.active_attachments())
            self.assertNotEqual(state.status, FootStatus.IDEALIZED)


if __name__ == "__main__":
    unittest.main()
