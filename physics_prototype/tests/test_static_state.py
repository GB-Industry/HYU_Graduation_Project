from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.contact_benchmarks import make_mixed_fixture, verify_fixed_allocation
from boulder_v1.contact import CAPTURE_ORIENTATION
from boulder_v1.contact_geometry import ContactMode
from boulder_v1.grasp import AttachmentStateError, GraspManager, InitialContactError, initialize_episode
from boulder_v1.locomotion import TransitionRequest, TransitionStatus, check_stabilization_readiness, execute_transition
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.schema import Limb
from boulder_v1.static_state import ReadinessTracker, initialize_static_reference
from boulder_v1.support import fresh_data


INTENT = {limb: limb.value.lower() for limb in Limb}


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, values, spec)
    return values


def controlled_step(fixture, allocation, tracker=None):
    model, data, manager = fixture.model, fixture.data, fixture.manager
    joints = model.actuator_trnid[:, 0]
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    # Test-only native Nm controller; no planned contact forces are applied.
    data.ctrl[:] = np.clip((np.array(allocation["tau_ff_Nm"])
                           + 80. * (fixture.reference[qids] - data.qpos[qids])
                           - data.qvel[dofs]) / model.actuator_gear[:, 0], -1., 1.)
    manager.evaluate_and_update()
    before = float(data.time)
    mujoco.mj_step(model, data)
    manager.evaluate_and_update(applied_data=data)
    return tracker.sample_after_step(before) if tracker is not None else None


class StaticReferenceTests(unittest.TestCase):
    def test_explicit_seed_admits_unloaded_touch_and_tangential_offset(self):
        fixture = make_mixed_fixture()
        scene = fixture.scene
        fixture.data.time = 2.
        fixture.data.qvel[:] = .01
        reference, manager = initialize_static_reference(fixture.model, fixture.data, scene, fixture.profile,
                                                         fixture.reference, INTENT)
        self.assertEqual(dict(scene.start_configuration), {})
        self.assertEqual(dict(manager.scene.start_configuration), INTENT)
        manager.require_session(fixture.model, fixture.data, manager.scene)
        np.testing.assert_array_equal(reference.qpos, fixture.reference)
        np.testing.assert_array_equal(fixture.data.qpos, fixture.reference)
        np.testing.assert_array_equal(fixture.data.qvel, 0.)
        self.assertEqual(fixture.data.time, 0.)
        self.assertEqual(set(manager.active_attachments()), {Limb.LEFT_HAND, Limb.RIGHT_HAND})
        self.assertTrue(all("foot" not in fixture.model.equality(i).name for i in range(fixture.model.neq)))
        feet = [r for r in reference.residuals if r.limb.is_foot]
        self.assertEqual(len(feet), 2)
        for residual in feet:
            self.assertAlmostEqual(abs(residual.tangential_offset[1]), .010, places=12)
            self.assertAlmostEqual(residual.signed_geom_distance, 0., places=12)
            self.assertAlmostEqual(residual.site_normal_residual, 0., places=12)
            self.assertGreater(residual.overlap_area, 0.)
        self.assertTrue(all(f.normal_force == 0. for f in manager.contact_snapshot().feet.values()))
        self.assertEqual(len(manager.capture_events), 2)
        with self.assertRaises(FrozenInstanceError):
            reference.qpos = ()
        with self.assertRaises(TypeError):
            reference.target_pose["left_elbow"] = 0.
        with self.assertRaises(TypeError):
            reference.contact_intent[Limb.LEFT_FOOT] = "other"
        fixture.reference[:] = 0.
        self.assertNotEqual(reference.qpos[2], 0.)

    def test_scene_consistency_and_profile_name_invariance(self):
        fixture = make_mixed_fixture()
        results = []
        for name in ("stage2_mixed", "compact_strong", "tall_weak", "unrelated"):
            profile = replace(fixture.profile, name=name)
            reference, _ = initialize_static_reference(fixture.model, fixture.data, fixture.scene, profile,
                                                       fixture.reference, INTENT)
            results.append(reference)
        self.assertTrue(all(r.qpos == results[0].qpos and r.residuals == results[0].residuals for r in results))
        declared = replace(fixture.scene, start_configuration=INTENT)
        _, manager = initialize_static_reference(fixture.model, fixture.data, declared, fixture.profile,
                                                 fixture.reference, INTENT)
        self.assertIs(manager.scene, declared)
        conflict = replace(fixture.scene, start_configuration={Limb.LEFT_HAND: "right_hand"})
        before = integration_state(fixture.model, fixture.data)
        with self.assertRaisesRegex(InitialContactError, "conflicts"):
            initialize_static_reference(fixture.model, fixture.data, conflict, fixture.profile, fixture.reference, INTENT)
        np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)

    def test_reference_matches_existing_reset_quaternion_normalization(self):
        fixture = make_mixed_fixture()
        seed = fixture.reference.copy()
        seed[3] += 1e-10
        caller = seed.copy()
        reference, _ = initialize_static_reference(fixture.model, fixture.data, fixture.scene, fixture.profile, seed, INTENT)
        np.testing.assert_array_equal(reference.qpos, fixture.data.qpos)
        np.testing.assert_array_equal(seed, caller)

    def test_rejects_invalid_seed_and_capacity_before_live_reset(self):
        fixture = make_mixed_fixture()
        model, data = fixture.model, fixture.data
        seeds = []
        rom = fixture.reference.copy()
        jid = model.joint("left_hip_pitch").id
        rom[int(model.jnt_qposadr[jid])] = model.jnt_range[jid, 1] + .01
        seeds.append(rom)
        far = fixture.reference.copy()
        far[0] += .002
        seeds.append(far)
        quaternion = fixture.reference.copy()
        quaternion[3] = 2.
        seeds.append(quaternion)
        seeds.append(np.full(model.nq, np.nan))
        seeds.append(fixture.reference[:-1])
        foot = fixture.reference.copy()
        foot[int(model.joint("left_ankle_pitch").qposadr[0])] = .2
        seeds.append(foot)
        data.time = 1.25
        data.ctrl[:] = .2
        data.qacc_warmstart[:] = .3
        before = integration_state(model, data)
        before_qacc = data.qacc.copy()
        for seed in seeds:
            with self.subTest(seed=seed), patch("boulder_v1.static_state.initialize_episode", wraps=initialize_episode) as reset:
                with self.assertRaises(ValueError):
                    initialize_static_reference(model, data, fixture.scene, fixture.profile, seed, INTENT)
                self.assertTrue(all(call.args[1] is not data for call in reset.call_args_list))
                np.testing.assert_array_equal(integration_state(model, data), before)
                np.testing.assert_array_equal(data.qacc, before_qacc)
        weak = replace(fixture.profile, grip_capacity=.001)
        with self.assertRaises(InitialContactError):
            initialize_static_reference(model, data, fixture.scene, weak, fixture.reference, INTENT)
        np.testing.assert_array_equal(integration_state(model, data), before)
        self.assertTrue(fixture.manager.initialized)

    def test_nonfinite_controls_and_integration_rejected_before_reset(self):
        fixture = make_mixed_fixture()
        for name in ("ctrl", "qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
            array = getattr(fixture.data, name)
            saved = array.copy()
            array.flat[0] = np.nan
            before = integration_state(fixture.model, fixture.data)
            with self.subTest(name=name), patch("boulder_v1.static_state.initialize_episode") as reset:
                with self.assertRaisesRegex(InitialContactError, "before reset"):
                    initialize_static_reference(fixture.model, fixture.data, fixture.scene, fixture.profile,
                                                fixture.reference, INTENT)
                reset.assert_not_called()
                np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)
            array[:] = saved

    def test_canonical_compiled_geometry_is_checked_not_scene_labels(self):
        fixture = make_mixed_fixture()
        before = integration_state(fixture.model, fixture.data)
        fixture.model.site("site_step_left_foot").pos[1] += .002
        with self.assertRaisesRegex(InitialContactError, "canonical Stage2"):
            initialize_static_reference(fixture.model, fixture.data, fixture.scene, fixture.profile,
                                        fixture.reference, INTENT)
        np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)

    def test_strict_hand_orientation_and_no_force_capture(self):
        fixture = make_mixed_fixture()
        model = fixture.model
        seed = fixture.reference.copy()
        wrist = model.joint("left_wrist").id
        seed[int(model.jnt_qposadr[wrist])] = model.jnt_range[wrist, 1]
        probe = mujoco.MjData(model)
        probe.qpos[:] = seed
        manager = GraspManager(model, probe, fixture.scene, profile=fixture.profile)
        measurement = manager._capture_measurement(Limb.LEFT_HAND, fixture.scene.region("left_hand"))
        self.assertLess(measurement["orientation"], CAPTURE_ORIENTATION)
        before = integration_state(model, fixture.data)
        with self.assertRaises(InitialContactError):
            initialize_static_reference(model, fixture.data, fixture.scene, fixture.profile, seed, INTENT)
        np.testing.assert_array_equal(integration_state(model, fixture.data), before)
        original = GraspManager.attach
        calls = []

        def checked_attach(manager, limb, region, force=False):
            calls.append(force)
            return original(manager, limb, region, force=force)

        with patch.object(GraspManager, "attach", checked_attach):
            initialize_static_reference(model, fixture.data, fixture.scene, fixture.profile, fixture.reference, INTENT)
        self.assertTrue(calls)
        self.assertFalse(any(calls))

    def test_even_inactive_foot_equality_rejected_by_compiled_site_identity(self):
        fixture = make_mixed_fixture()
        tree = ET.fromstring(fixture.xml)
        ET.SubElement(tree.find("equality"), "connect", name="hidden_constraint", site1="left_foot_site",
                      site2="site_step_left_foot", active="false")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        data = mujoco.MjData(model)
        before = integration_state(model, data)
        with self.assertRaisesRegex(InitialContactError, "zero foot equalities"):
            initialize_static_reference(model, data, fixture.scene, fixture.profile, fixture.reference, INTENT)
        np.testing.assert_array_equal(integration_state(model, data), before)


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = make_mixed_fixture(.001)
        cls.allocation = verify_fixed_allocation(cls.fixture)
        cls.tracker = ReadinessTracker(cls.fixture.model, cls.fixture.data, cls.fixture.manager)
        cls.history = [controlled_step(cls.fixture, cls.allocation, cls.tracker) for _ in range(3000)]
        cls.settled = mujoco.MjData(cls.fixture.model)
        mujoco.mj_copyData(cls.settled, cls.fixture.model, cls.fixture.data)

    def setUp(self):
        fixture = self.fixture
        data = mujoco.MjData(fixture.model)
        mujoco.mj_copyData(data, fixture.model, self.settled)
        manager = GraspManager(fixture.model, data, fixture.scene, profile=fixture.profile)
        manager.synchronize_from_live()
        self.live = replace(fixture, data=data, manager=manager)

    def test_real_native_rollout_rejects_intermittent_one_frame_crossings(self):
        self.assertTrue(any(row.max_hinge_speed is not None and row.max_hinge_speed > .10 for row in self.history[-500:]))
        self.assertTrue(any(row.duration > 0. for row in self.history[-500:]))
        self.assertFalse(any(row.ready for row in self.history))
        self.assertLess(max(row.duration for row in self.history), .5)
        self.assertFalse(self.history[0].ready)
        self.assertFalse(np.any(self.fixture.data.qfrc_applied))
        self.assertFalse(np.any(self.fixture.data.xfrc_applied))
        self.assertFalse(self.fixture.manager.releases)

    def test_default_half_second_window_on_native_clock(self):
        fixture = self.live
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager)
        self.assertEqual(tracker.required_duration, .5)
        # Timing unit test only: inject a valid gate, not a physical certificate.
        # The separate real rollout test retains all measured physical gates.
        with patch.object(tracker, "_measure", return_value=("", (0., 0., 0., 0.))):
            for index in range(501):
                row = controlled_step(fixture, self.allocation, tracker)
                self.assertEqual(row.ready, index == 500)
                self.assertAlmostEqual(row.duration, index * .001, places=10)
                self.assertEqual(tracker.check()[0], row.ready)

    def test_query_onset_duration_and_duplicate_endpoints(self):
        fixture = self.live
        dt = fixture.model.opt.timestep
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager, required_duration=3 * dt)
        before = integration_state(fixture.model, fixture.data)
        for _ in range(5):
            self.assertFalse(check_stabilization_readiness(fixture.model, fixture.data, fixture.manager)[0])
            self.assertFalse(check_stabilization_readiness(fixture.model, fixture.data, fixture.manager, tracker=tracker)[0])
        np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), before)
        for index in range(4):
            start = fixture.data.time
            evidence = controlled_step(fixture, self.allocation, tracker)
            self.assertAlmostEqual(evidence.duration, index * dt, places=10)
            self.assertEqual(evidence.ready, index == 3)
            sampled = integration_state(fixture.model, fixture.data)
            for _ in range(5):
                duplicate = tracker.sample_after_step(start)
                self.assertEqual(duplicate, evidence)
                self.assertEqual(tracker.check()[0], evidence.ready)
            np.testing.assert_array_equal(integration_state(fixture.model, fixture.data), sampled)

    def test_gaps_adoption_and_reset_clear_duration(self):
        fixture = self.live
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager, required_duration=fixture.model.opt.timestep)
        controlled_step(fixture, self.allocation, tracker)
        self.assertTrue(controlled_step(fixture, self.allocation, tracker).ready)
        controlled_step(fixture, self.allocation)
        self.assertFalse(tracker.check()[0])
        gap = controlled_step(fixture, self.allocation, tracker)
        self.assertEqual(gap.duration, 0.)
        self.assertIn("Gap", gap.reason)
        self.assertFalse(controlled_step(fixture, self.allocation, tracker).ready)
        self.assertTrue(controlled_step(fixture, self.allocation, tracker).ready)
        fixture.manager.synchronize_from_live()
        self.assertFalse(tracker.check()[0])
        adopted = controlled_step(fixture, self.allocation, tracker)
        self.assertEqual(adopted.duration, 0.)
        self.assertIn("adoption", adopted.reason)
        mujoco.mj_resetData(fixture.model, fixture.data)
        fixture.manager.synchronize_from_live()
        self.assertFalse(tracker.check()[0])
        self.assertFalse(tracker.sample_after_step(fixture.data.time - fixture.model.opt.timestep).ready)

    def test_max_hinge_not_rms_and_nonfinite_control_state(self):
        fixture = self.live
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager)
        jid = fixture.model.joint("waist_yaw").id
        fixture.data.qvel[int(fixture.model.jnt_dofadr[jid])] = .11
        reason, speeds = tracker._measure()
        self.assertLess(speeds[3], .10)
        self.assertGreater(speeds[2], .10)
        self.assertIn("Maximum hinge", reason)
        for name in ("ctrl", "qacc_warmstart", "qacc", "qfrc_applied", "xfrc_applied"):
            mujoco.mj_copyData(fixture.data, fixture.model, self.settled)
            getattr(fixture.data, name).flat[0] = np.nan
            with self.subTest(name=name):
                self.assertFalse(tracker.check()[0])
                self.assertIn("Nonfinite", tracker._measure()[0])

    def test_slip_contact_loss_and_hand_overload_are_physical(self):
        fixture = self.live
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager, required_duration=fixture.model.opt.timestep)
        controlled_step(fixture, self.allocation, tracker)
        self.assertTrue(controlled_step(fixture, self.allocation, tracker).ready)
        fixture.data.qvel[0] = .019  # Root threshold passes, actual sole slip fails.
        feet = fixture.manager.contact_snapshot().feet
        self.assertTrue(any(foot.slipping for foot in feet.values()))
        self.assertIn("without slip", tracker._measure()[0])
        self.assertFalse(tracker.check()[0])
        invalid = tracker.sample_after_step(fixture.data.time - fixture.model.opt.timestep)
        self.assertEqual(invalid.duration, 0.)
        fixture.manager.detach(Limb.LEFT_HAND)
        self.assertIn("hand", tracker._measure()[0])
        fixture.manager.synchronize_from_live()
        self.assertFalse(tracker.check()[0])
        overloaded_data = mujoco.MjData(fixture.model)
        mujoco.mj_copyData(overloaded_data, fixture.model, self.settled)
        weak = replace(fixture.profile, grip_capacity=.001)
        weak_manager = GraspManager(fixture.model, overloaded_data, fixture.scene, profile=weak)
        weak_manager.synchronize_from_live()
        overloaded = ReadinessTracker(fixture.model, overloaded_data, weak_manager)
        self.assertIn("overloaded", overloaded._measure()[0])
        self.assertFalse(overloaded.check()[0])

    def test_readiness_profile_names_cannot_relax_physical_limits(self):
        fixture = self.live
        for name in ("compact_strong", "tall_weak", "stage2_mixed"):
            manager = GraspManager(fixture.model, fixture.data, fixture.scene, profile=replace(fixture.profile, name=name))
            manager.synchronize_from_live()
            tracker = ReadinessTracker(fixture.model, fixture.data, manager)
            fixture.data.qvel[3] = .051
            self.assertFalse(check_stabilization_readiness(fixture.model, fixture.data, manager,
                             v_lin_thresh=100., v_ang_thresh=100., qvel_thresh=100., tracker=tracker)[0])
            self.assertTrue(tracker._measure()[0])
            fixture.data.qvel[:] = self.settled.qvel

    def test_native_foot_contact_loss_and_duplicate_invalid_state_clear_window(self):
        fixture = self.live
        tracker = ReadinessTracker(fixture.model, fixture.data, fixture.manager, required_duration=fixture.model.opt.timestep)
        controlled_step(fixture, self.allocation, tracker)
        self.assertTrue(controlled_step(fixture, self.allocation, tracker).ready)
        previous_qacc = fixture.data.qacc.copy()
        fixture.data.qacc[0] = np.nan
        duplicate = tracker.sample_after_step(fixture.data.time - fixture.model.opt.timestep)
        self.assertFalse(duplicate.ready)
        self.assertEqual(duplicate.duration, 0.)
        fixture.data.qacc[:] = previous_qacc
        self.assertFalse(controlled_step(fixture, self.allocation, tracker).ready)
        fixture.data.xfrc_applied[fixture.model.body("left_foot").id, 2] = 1000.
        for _ in range(30):
            controlled_step(fixture, self.allocation, tracker)
            if not fixture.manager.contact_snapshot().feet[Limb.LEFT_FOOT].supporting:
                break
        self.assertFalse(fixture.manager.contact_snapshot().feet[Limb.LEFT_FOOT].supporting)
        self.assertFalse(tracker.check()[0])

    def test_transition_owns_fresh_tracker_and_samples_native_steps_not_observers(self):
        fixture = self.live
        owned = []
        observations = []
        pose = {fixture.model.joint(jid).name: float(fixture.reference[int(fixture.model.jnt_qposadr[jid])])
                for jid in range(fixture.model.njnt) if fixture.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE}

        def new_tracker(model, data, manager):
            tracker = ReadinessTracker(model, data, manager)
            tracker.sample_after_step = Mock(wraps=tracker.sample_after_step)
            owned.append(tracker)
            return tracker

        def observer(observation, data, manager):
            observations.append(observation)
            self.assertIsNot(data, fixture.data)
            with self.assertRaises(AttachmentStateError):
                check_stabilization_readiness(manager.model, data, manager, tracker=owned[0])
            data.qvel[:] = 100.

        request = TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand", steps=10, settle_steps_after=0)
        before = float(fixture.data.time)
        with patch("boulder_v1.locomotion.ReadinessTracker", side_effect=new_tracker), \
                patch("boulder_v1.locomotion.solve_retargeted_stance", return_value=SimpleNamespace(target_pose=pose)):
            result = execute_transition(fixture.model, fixture.data, fixture.scene, fixture.profile, request,
                                        manager=fixture.manager, frame_callback=observer)
        self.assertEqual(len(owned), 1)
        self.assertEqual(owned[0].sample_after_step.call_count, result.steps)
        self.assertEqual(result.status, TransitionStatus.INCOMPLETE)
        self.assertEqual(result.steps, 10)
        self.assertAlmostEqual(result.duration, 10 * fixture.model.opt.timestep, places=10)
        self.assertAlmostEqual(result.time - before, result.duration, places=10)
        self.assertTrue(observations)
        self.assertLess(np.max(np.abs(fixture.data.qvel)), 100.)

    def test_legacy_instantaneous_success_is_explicitly_nonphysical(self):
        fixture = make_mixed_fixture()
        scene = replace(fixture.scene, start_configuration=INTENT)
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile, contact_mode=ContactMode.IDEALIZED_DEBUG))
        data = mujoco.MjData(model)
        manager = initialize_episode(model, data, scene, fixture.profile, attach_feet=True, initial_qpos=fixture.reference)
        ready, reason = check_stabilization_readiness(model, data, manager)
        self.assertTrue(ready)
        self.assertIn("NONPHYSICAL", reason)
        self.assertIn("software regression", reason)
        with self.assertRaises(ValueError):
            ReadinessTracker(model, data, manager)

    def test_unintended_loaded_body_contact_in_compiled_model(self):
        fixture = make_mixed_fixture()
        regions = tuple(replace(region, position=(region.position[0], -.405, .355), half_size=(.06, .095, .03))
                        if region.id == "left_foot" else region for region in fixture.scene.contact_regions)
        scene = replace(fixture.scene, contact_regions=regions)
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, fixture.profile))
        data = mujoco.MjData(model)
        data.qpos[:] = fixture.reference
        data.eq_active[:] = fixture.data.eq_active
        mujoco.mj_forward(model, data)
        manager = GraspManager(model, data, scene, profile=fixture.profile)
        manager.synchronize_from_live()
        tracker = ReadinessTracker(model, data, manager)
        from boulder_v1.static_state import _unexpected_loaded_contacts
        allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{hold}")) for limb, hold in INTENT.items()}
        unexpected = _unexpected_loaded_contacts(model, fresh_data(model, data), allowed)
        self.assertTrue(any("left_shin_collider" in pair for pair in unexpected), unexpected)
        self.assertFalse(tracker.check()[0])


if __name__ == "__main__":
    unittest.main()
