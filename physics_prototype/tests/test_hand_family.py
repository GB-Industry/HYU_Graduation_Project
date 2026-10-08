"""Hand-side capture margin and unchanged native-physics family evidence."""
import copy
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import json
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1 import hand_family, single_hand
from boulder_v1.contact import CAPTURE_DISTANCE
from boulder_v1.contact_geometry import canonical_geometry
from boulder_v1.grasp import GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.schema import ClimberProfile, Limb
from test_single_hand import NativeAudit, capture_gate, integration_state, model_parameters


def encode(value):
    # StaticReference deliberately contains immutable MappingProxy mappings.
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(encode(key)): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def metrics(result):
    return {key: encode(result.get(key)) for key in
            ("status", "reason", "dt_s", "steps", "capture", "first_eligible", "guard_failure", "readiness")}


def audited_benchmark(dt, *, family=False, limb=Limb.RIGHT_HAND, pose_perturbation_rad=0.):
    audits, final_poses = [], []
    native_execute, native_attach = single_hand.execute_single_hand, GraspManager.attach
    native_solve = single_hand.solve_contact_pose

    def attach(manager, moving, region, force=False):
        measurement = manager._capture_measurement(moving, region)
        accepted = native_attach(manager, moving, region, force=force)
        if audits and manager is audits[-1].manager:
            audit = audits[-1]
            audit.record(not force, "forced family attachment")
            audit.record(not accepted or capture_gate(measurement), "family capture outside unchanged gate")
            audit.captures.append({"measurement": measurement, "accepted": accepted, "limb": moving})
        return accepted

    def solve(*args, **kwargs):
        result = native_solve(*args, **kwargs)
        final_poses.append(result)
        return result

    def execute(model, data, scene, profile, reference, manager, request, **kwargs):
        audit = NativeAudit(model, data, scene, profile, reference, manager, request, kwargs.get("fault"))
        audits.append(audit)
        with ExitStack() as stack:
            audit.install(stack)
            stack.enter_context(patch.object(single_hand, "solve_contact_pose", new=solve))
            result = native_execute(model, data, scene, profile, reference, manager, request, **kwargs)
            audit.unchanged()
        audit.after_model = model_parameters(model)
        return result

    with patch.object(single_hand, "execute_single_hand", new=execute), \
            patch.object(hand_family, "execute_single_hand", new=execute), \
            patch.object(GraspManager, "attach", new=attach):
        result = (hand_family.run_hand_family_benchmark(dt, limb=limb, pose_perturbation_rad=pose_perturbation_rad)
                  if family else single_hand.run_single_hand_benchmark(dt))
    if len(audits) != 1:
        raise AssertionError("No authoritative hand executor: " + result["reason"])
    return result, audits[0], final_poses


class HandFamilyTests(unittest.TestCase):
    def assert_native_success(self, result, audit, final_poses):
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(audit.violations, set())
        self.assertEqual(result["steps"], len(audit.steps))
        self.assertEqual(result["steps"], len(audit.controls))
        self.assertEqual(result["steps"], len(audit.samples))
        self.assertEqual(result["steps"], len(result["samples"]))
        self.assertEqual(len(audit.guards), 2 * result["steps"] + 1)
        self.assertEqual(audit.reference_before,
                         (audit.reference.qpos, dict(audit.reference.target_pose), dict(audit.reference.contact_intent)))
        for field, value in audit.before_model.items():
            np.testing.assert_array_equal(audit.after_model[field], value, err_msg=field)
        self.assertFalse(np.any(audit.data.xfrc_applied) or np.any(audit.data.qfrc_applied))
        self.assertEqual(result["initial_contacts"], dict(audit.scene.start_configuration))
        self.assertTrue(result["initial_static"]["success"])
        self.assertAlmostEqual(result["initial_state"].time, 2., places=9)
        self.assertEqual(len(audit.detaches), 1)
        self.assertEqual(audit.detaches[0]["limb"], audit.request.limb)
        self.assertLessEqual(audit.detaches[0]["applied_palm_N"], 1e-6)
        self.assertLessEqual(audit.detaches[0]["endpoint_palm_N"], 1e-6)
        for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
            self.assertEqual(result["foot_support_fraction"][limb.value], 1.)
            self.assertLessEqual(result["foot_slip_max_m_s"][limb.value], .01)
        velocities = np.array([row["qd_ref"] for row in result["samples"]])[:, audit.vi]
        self.assertLessEqual(np.max(np.abs(velocities)), .5 + 1e-10)
        self.assertLessEqual(np.max(np.abs(np.diff(np.vstack((np.zeros((1, audit.model.nu)), velocities)), axis=0)))
                             / result["dt_s"], 2. + 1e-8)
        capture = result["capture"]
        self.assertTrue(capture_gate(capture))
        self.assertEqual(capture["capture_margin_m"], CAPTURE_DISTANCE - capture["gap_m"])
        self.assertEqual(capture["first_eligible_time_s"], result["first_eligible"]["time_s"])
        preceding = next(row["capture_measurement"] for row in result["samples"]
                         if abs(row["time_s"] - (capture["time_s"] - result["dt_s"])) < 1e-10)
        for key, value in preceding.items():
            self.assertEqual(capture["preceding_measurement"][key], value)
        for event in result["capture_events"]:
            self.assertEqual(event["capture_margin_m"], .001 - event["gap_m"])
            self.assertTrue(capture_gate(event))
            self.assertLessEqual(event["peak_window_reaction_N"], 850.)
        self.assertEqual(len(final_poses), 1)
        self.assertIs(result["final_reference"], final_poses[0].reference)
        self.assertEqual(dict(result["final_reference"].contact_intent), result["new_contacts"])
        np.testing.assert_allclose(result["samples"][-1]["q_ref"], result["final_reference"].qpos, rtol=0, atol=1e-12)
        with self.assertRaises(TypeError):
            result["final_reference"].contact_intent[audit.request.limb] = "other"
        with self.assertRaises(TypeError):
            result["final_reference"].target_pose["waist_yaw"] = 1.
        self.assertTrue(result["readiness"]["ready"])
        self.assertGreaterEqual(result["readiness"]["duration"], .5 - 1e-10)
        self.assertEqual(result["events"][-1]["event"], "FINAL_READY")
        self.assertTrue(result["support_admission"]["admitted"])
        admission = result["support_admission"]
        self.assertNotIn(audit.request.limb, admission["contacts"])
        self.assertLessEqual(admission["root_balance_residual"], 1e-8)
        self.assertLess(admission["motor_utilization_max"], 1.)
        self.assertGreater(admission["motor_margin_min_Nm"], 0.)
        self.assertTrue(admission["finite"])
        self.assertFalse(admission["applied_unintended"] or admission["endpoint_unintended"])
        for hand in admission["estimated_hands"].values():
            self.assertGreater(hand["margin_N"], 0.)
        for foot in admission["estimated_feet"].values():
            self.assertGreater(foot["minimum_load_margin_N"], 0.)
            self.assertGreaterEqual(foot["friction_margin_N"], 0.)
        json.dumps(encode(result), allow_nan=False)

    def test_stage4_default_capture_timing(self):
        expected_gaps = {.002: .000988638015618121, .001: .0009997163052341445}
        for dt in (.002, .001):
            with self.subTest(dt=dt):
                result, audit, poses = audited_benchmark(dt)
                print("HAND_DEFAULT_AUDIT " + json.dumps(metrics(result), allow_nan=False), flush=True)
                self.assert_native_success(result, audit, poses)
                self.assertAlmostEqual(result["capture"]["gap_m"], expected_gaps[dt], delta=1e-12)
                self.assertEqual(result["capture"]["time_s"], result["first_eligible"]["time_s"])
                self.assertAlmostEqual(result["capture"]["reference_time_s"], 3.76, places=12)
                self.assertEqual(result["capture"]["acquisition_policy"], "first_eligible")
                self.assertLess(result["first_eligible"]["reach_elapsed_s"], 4.)
                self.assertFalse(capture_gate(result["first_eligible"]["preceding_measurement"]))

    def test_family_endpoint_physics(self):
        for dt in (.002, .001):
            for limb in (Limb.RIGHT_HAND, Limb.LEFT_HAND):
                with self.subTest(dt=dt, limb=limb):
                    result, audit, poses = audited_benchmark(dt, family=True, limb=limb)
                    print("HAND_FAMILY_AUDIT " + json.dumps({"limb": limb.value, **metrics(result)}, allow_nan=False), flush=True)
                    self.assert_native_success(result, audit, poses)
                    capture = result["capture"]
                    self.assertEqual(capture["reference_time_s"], 4.1)
                    self.assertEqual(capture["reference_endpoint_error_m"], 0.)
                    self.assertGreater(capture["time_s"], result["first_eligible"]["time_s"])
                    self.assertGreater(capture["capture_margin_m"], result["first_eligible"]["capture_margin_m"])
                    self.assertLessEqual(capture["relative_speed_m_s"], capture["preceding_measurement"]["relative_speed_m_s"])
                    self.assertLessEqual(capture["gap_m"], capture["preceding_measurement"]["gap_m"])
                    # Useful margin is observed, not a new physical acceptance gate.
                    self.assertGreater(capture["capture_margin_m"], .0009)
                    before_capture = [row for row in result["samples"] if row["phase"] == "REACH"]
                    eligible_rows = [row for row in before_capture if capture_gate(row["capture_measurement"])]
                    self.assertEqual(eligible_rows[0]["time_s"], result["first_eligible"]["time_s"])
                    self.assertTrue(all(not row["hands"][limb.value]["active"] for row in before_capture))
                    tree = ET.fromstring(build_mjcf(audit.scene, audit.profile))
                    tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
                    builder = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                    for key, value in model_parameters(builder).items():
                        np.testing.assert_array_equal(audit.before_model[key], value, err_msg=key)
                    target = audit.scene.region(audit.request.target)
                    source = audit.scene.region(audit.request.source)
                    self.assertEqual(target.position[:2], source.position[:2])
                    self.assertAlmostEqual(target.position[2] - source.position[2], .06)
                    for side in ("left", "right"):
                        self.assertEqual(audit.reference.qpos[audit.model.joint(f"{side}_knee").qposadr[0]], .3)

    def test_perturbed_scratch_reference_native_use(self):
        for dt in (.002, .001):
            for limb, perturbation in ((Limb.RIGHT_HAND, .0001), (Limb.LEFT_HAND, -.0001)):
                with self.subTest(dt=dt, limb=limb):
                    result, audit, poses = audited_benchmark(dt, family=True, limb=limb,
                                                           pose_perturbation_rad=perturbation)
                    print("HAND_PERTURBED_AUDIT " + json.dumps({"limb": limb.value, "perturbation_rad": perturbation,
                                                               **metrics(result)}, allow_nan=False), flush=True)
                    self.assertTrue(result["retarget"].admitted, result["retarget"].reason)
                    self.assertEqual(audit.reference.qpos, result["retarget"].qpos)
                    self.assert_native_success(result, audit, poses)
                    self.assertGreater(result["capture"]["capture_margin_m"], .0009)

    def test_preflight_rejects_support_estimates_and_unreachable_without_live_change(self):
        model, data, scene, profile, seed = hand_family.make_hand_family_fixture()
        pose = single_hand.solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
        reference, manager = single_hand.initialize_static_reference(model, data, scene, profile, pose.qpos,
                                                                      scene.start_configuration)
        initial = single_hand.execute_static_hold(model, data, scene, profile, reference, manager,
                                                  duration=2., settle=1., score_window=.5, keep_samples=False)
        self.assertTrue(initial["success"], initial["reason"])
        request = single_hand.SingleHandRequest(Limb.LEFT_HAND, "left_hand", "left_reach_target",
                                                acquisition_policy="endpoint_settle")
        for reason in ("Support reference root balance failed", "Support reference exceeds remaining-hand capacity",
                       "Support reference violates unilateral compression/friction", "Support reference exceeds original motor capability"):
            with self.subTest(reason=reason):
                before = integration_state(model, data)
                with patch.object(single_hand, "estimate_support_torques", side_effect=ValueError(reason)) as estimate, \
                        patch("mujoco.mj_step", side_effect=AssertionError("support rejection stepped")), \
                        patch.object(GraspManager, "detach", side_effect=AssertionError("support rejection released")), \
                        patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("support rejection commanded")):
                    result = single_hand.execute_single_hand(model, data, scene, profile, reference, manager, request)
                self.assertEqual(result["status"], "THREE_POINT_SUPPORT_FAILURE")
                self.assertEqual(result["steps"], 0)
                self.assertFalse(result["released"] or result["success"] or result["support_admission"]["admitted"])
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["final_reference"])
                self.assertEqual(estimate.call_count, 1)
                self.assertEqual(estimate.call_args.args[-1], {l: h for l, h in reference.contact_intent.items() if l != request.limb})
                np.testing.assert_array_equal(integration_state(model, data), before)
        for bad, expected in ((replace(request, target="unreachable_target"), "REACH_INFEASIBLE"),
                              (replace(request, acquisition_policy="unknown"), "REACH_INFEASIBLE")):
            before = integration_state(model, data)
            with patch("mujoco.mj_step", side_effect=AssertionError("invalid target stepped")):
                result = single_hand.execute_single_hand(model, data, scene, profile, reference, manager, bad)
            self.assertEqual(result["status"], expected)
            self.assertIsNone(result["final_reference"])
            self.assertFalse(result["released"])
            np.testing.assert_array_equal(integration_state(model, data), before)
        captured_not_ready = single_hand.execute_single_hand(model, data, scene, profile, reference, manager,
                                                             replace(request, settle_timeout_s=.002))
        self.assertEqual(captured_not_ready["status"], "TIMEOUT", captured_not_ready["reason"])
        self.assertIsNotNone(captured_not_ready["capture"])
        self.assertIsNone(captured_not_ready["final_reference"])
        self.assertFalse(captured_not_ready["readiness"]["ready"])

    def test_fixed_scene_profile_diagnostics(self):
        profiles = (ClimberProfile("compact_strong", upper_arm_length=.28, forearm_length=.24,
                                   thigh_length=.39, shin_length=.37, rom_scale=1.08,
                                   strength_scale=1.15, grip_capacity=1000.),
                    ClimberProfile("long_reach_lower_grip", upper_arm_length=.34, forearm_length=.30,
                                   thigh_length=.44, shin_length=.42, rom_scale=.95, strength_scale=.92,
                                   grip_capacity=850.))
        fixed = hand_family.make_family_scene()
        for profile in profiles:
            with self.subTest(profile=profile.name):
                model, data, scene, bound, seed = hand_family.make_hand_family_fixture(profile=profile)
                self.assertEqual(scene.to_dict(), fixed.to_dict())
                self.assertIs(bound, profile)
                before = integration_state(model, data)
                pose = single_hand.solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
                arms = {limb.value: single_hand.solve_hand_reference(model, pose.qpos, limb,
                        canonical_geometry(scene.region(target)).hand_frame)
                        for limb, target in ((Limb.RIGHT_HAND, "reach_target"), (Limb.LEFT_HAND, "left_reach_target"))}
                print("HAND_PROFILE_DIAGNOSTIC " + json.dumps(encode({"profile": profile.to_dict(),
                       "geometric_converged": pose.converged, "admitted": pose.admitted, "reason": pose.reason,
                       "iterations": pose.iterations, "residuals": pose.residuals, "arm_reference": arms}), allow_nan=False), flush=True)
                self.assertEqual(pose.admitted, pose.reference is not None)
                self.assertTrue(pose.reason)
                self.assertTrue(all(np.isfinite(pose.qpos)))
                np.testing.assert_array_equal(integration_state(model, data), before)
                # These are local diagnostics, never a claim of physical personalization.
                for arm in arms.values():
                    self.assertTrue(arm.reason)
                    self.assertTrue(np.isfinite(arm.position_error))
                physical = hand_family.run_hand_family_benchmark(profile=profile, keep_samples=False)
                print("HAND_PROFILE_PHYSICAL " + json.dumps(encode({"profile": profile.name,
                      "status": physical["status"], "reason": physical["reason"],
                      "initial_static_status": physical.get("initial_static", {}).get("status"),
                      "capture": physical.get("capture")}),
                      allow_nan=False), flush=True)
                self.assertTrue(physical["reason"])
                if physical["success"]:
                    self.assertTrue(physical["readiness"]["ready"])
                    self.assertIsNotNone(physical["final_reference"])
                else:
                    self.assertIsNone(physical.get("final_reference"))


class HandPreflightBoundaryTests(unittest.TestCase):
    def test_fresh_measurement_failure_rejects_before_live_execution(self):
        model, data, scene, profile, seed = single_hand.make_single_hand_fixture()
        reference, manager = single_hand.initialize_static_reference(model, data, scene, profile, seed,
                                                                      scene.start_configuration)
        single_hand.execute_static_hold(model, data, scene, profile, reference, manager,
                                        duration=.5, settle=.2, score_window=.05, keep_samples=False)
        self.assertEqual(manager.contact_configuration(), dict(reference.contact_intent))
        request = single_hand.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")
        before = integration_state(model, data)
        captures, releases = copy.deepcopy(manager.capture_events), copy.deepcopy(manager.releases)
        rows = []
        with patch.object(single_hand, "fresh_data", side_effect=ValueError("Preflight contact solve failed")), \
                patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("preflight failure commanded")) as control, \
                patch("mujoco.mj_step", side_effect=AssertionError("preflight failure integrated")) as step, \
                patch.object(manager, "detach", side_effect=AssertionError("preflight failure released")) as detach:
            result = single_hand.execute_single_hand(model, data, scene, profile, reference, manager, request,
                                                      observer=lambda row, *_: rows.append(row))
        self.assertEqual(result["status"], "THREE_POINT_SUPPORT_FAILURE")
        self.assertEqual(result["reason"], "Preflight contact solve failed")
        self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
        self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
        self.assertFalse(result["support_admission"]["admitted"])
        self.assertIsNone(result["final_reference"])
        control.assert_not_called()
        step.assert_not_called()
        detach.assert_not_called()
        np.testing.assert_array_equal(integration_state(model, data), before)
        self.assertEqual(manager.capture_events, captures)
        self.assertEqual(manager.releases, releases)
        self.assertEqual(result["initial_state"], result["final_state"])
        terminal = result["terminal_observation"]
        self.assertEqual(rows, [terminal])
        self.assertEqual(terminal["time_s"], data.time)
        self.assertIsNone(terminal["actual_hand_position_world_m"])
        self.assertIsNone(terminal["tracking_error_m"])
        self.assertFalse(terminal["pose_available"])
        json.dumps(encode(result), allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
