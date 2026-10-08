import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.contact_geometry import ContactMode
from boulder_v1.model_validation import (
    JOINT_SPECS,
    actuator_diagnostic,
    climber_body_ids,
    freefall_diagnostic,
    mass_matrix,
    measure_joint_semantics,
    mirror_diagnostic,
    mirror_state,
    neutral_rollout_diagnostic,
    passive_decay_diagnostic,
    physics_numeric_arrays,
    profile_diagnostics,
    summarize_model,
    validate_physical_model,
)
from boulder_v1.runtime import compiled_numerical_issues
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import ClimberProfile


def _engine_fault(function, field, value=np.nan, at_time=0, index=-1):
    original = getattr(mujoco, function)

    def inject(model, data):
        original(model, data)
        if data.time >= at_time:
            if field == "time":
                data.time = value
            else:
                getattr(data, field).flat[index] = value

    return inject


class PhysicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_synthetic_scene()
        cls.profile = ClimberProfile(name="base")
        cls.model = cls.compile(cls.profile)
        cls.summary = summarize_model(cls.model)

    @classmethod
    def compile(cls, profile):
        return mujoco.MjModel.from_xml_string(build_mjcf(cls.scene, profile))

    def assert_diagnostic(self, evidence):
        self.assertTrue(evidence["passed"], {name: value for name, value in evidence["checks"].items() if not value})

    def test_compiled_topology_mass_inertias_com(self):
        self.assert_diagnostic(self.summary)
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu, self.model.njnt), (32, 31, 25, 26))
        self.assertEqual(self.model.ngeom, 74)
        self.assertEqual((self.model.nsite, self.model.neq, self.model.nexclude, self.model.npair), (21, 16, 15, 20))
        self.assertEqual(self.summary["climber"]["mass"], 78.3)
        self.assertAlmostEqual(self.summary["climber"]["com_world_neutral"][0], 0, places=14)
        bounds = self.summary["climber"]["primary_proxy_bounds_world_neutral"]
        np.testing.assert_allclose(bounds, [[-0.248, -0.575, 0.343], [0.248, -0.310, 1.9545]], rtol=0, atol=1e-14)
        com = np.array(self.summary["climber"]["com_world_neutral"])
        self.assertTrue(np.isfinite(com).all() and (com >= bounds[0]).all() and (com <= bounds[1]).all())
        np.testing.assert_array_equal(self.summary["mass_matrix"]["dof_armature"], [0] * 6 + [0.01] * 25)
        self.assertEqual(self.summary["mass_matrix"]["armature_units"], ["kg"] * 3 + ["kg*m^2"] * 28)
        for actuator in self.summary["actuators"]:
            self.assertFalse(actuator["force_limited"])
            self.assertEqual(actuator["force_range"], [0, 0])
        root, pelvis = self.model.body("climber_root").id, self.model.body("pelvis").id
        self.assertEqual(self.model.body_mass[root], 0)
        self.assertEqual(self.model.body_mass[pelvis], 10)
        self.assertEqual(self.model.body_weldid[root], self.model.body_weldid[pelvis])
        for name, mass in {"torso": 40, "head_neck": 5, "left_arm": 4.45, "right_arm": 4.45,
                           "left_leg": 12.2, "right_leg": 12.2}.items():
            self.assertAlmostEqual(self.summary["climber"]["groups"][name]["mass"], mass)
        for body in climber_body_ids(self.model):
            if self.model.body_mass[body] > 0:
                inertia = self.model.body_inertia[body]
                self.assertTrue(np.all(inertia > 0))
                self.assertLessEqual(2 * max(inertia), sum(inertia) + 1e-12)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        matrix = mass_matrix(self.model, data)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(matrix)[0], 0)
        self.assertGreater(self.summary["fixed_environment"]["mass"], 0)
        self.assertEqual(compiled_numerical_issues(self.model), [])
        for field, value in (("actuator_acc0", np.inf), ("dof_invweight0", np.inf), ("actuator_gainprm", np.nan)):
            with self.subTest(native_field=field):
                corrupted = copy.copy(self.model)
                getattr(corrupted, field).flat[0] = value
                with patch("boulder_v1.model_validation.mujoco.mj_forward") as forward, \
                     patch("boulder_v1.model_validation.np.linalg.eigh") as eigh, \
                     patch("boulder_v1.model_validation.np.linalg.eigvalsh") as eigvalsh:
                    summary = summarize_model(corrupted)
                    motors = actuator_diagnostic(corrupted)
                    with patch("boulder_v1.model_validation.mujoco.MjModel.from_xml_string", return_value=corrupted):
                        aggregate = validate_physical_model()
                    forward.assert_not_called()
                    eigh.assert_not_called()
                    eigvalsh.assert_not_called()
                for report in (summary, motors, aggregate["compiled_numerics"]):
                    self.assertFalse(report["passed"])
                    self.assertFalse(report["checks"]["compiled_numeric_finite"])
                    self.assertIn(f"model.{field}", report["issues"])
                    json.dumps(report, allow_nan=False)
                self.assertFalse(aggregate["passed"])
                self.assertNotIn("freefall", aggregate)
                self.assertIsNone(motors["maximum_error"])
                json.dumps(aggregate, allow_nan=False)
        for field in ("time", "qacc_warmstart", "xpos", "xquat", "xmat", "xipos", "ximat",
                      "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "subtree_com"):
            with self.subTest(sample_field=field), \
                 patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", field)), \
                 self.assertRaisesRegex(ValueError, f"sample data.{field}"):
                summarize_model(self.model)

    def test_primary_ellipsoid_dimensions_analytic_inertia(self):
        for updates in ({}, {"torso_length": 0.66}, {"hip_width": 0.36}, {"shoulder_width": 0.50}):
            profile = replace(self.profile, **updates)
            model = self.compile(profile)
            torso_scale = profile.torso_length / 0.55
            sizes = {"pelvis": [profile.hip_width / 2 * 0.90, 0.095, 0.14 * torso_scale * 0.48],
                     "abdomen": [profile.hip_width / 2 * 0.82, 0.088, 0.16 * torso_scale * 0.48],
                     "chest": [profile.shoulder_width / 2 * 0.88, 0.115, 0.25 * torso_scale * 0.48]}
            for name, dimensions in sizes.items():
                body, geom = model.body(name).id, model.geom(f"{name}_geom").id
                np.testing.assert_allclose(model.geom_size[geom], dimensions, atol=1e-14, rtol=0)
                x, y, z = dimensions
                mass = {"pelvis": 10, "abdomen": 12, "chest": 18}[name]
                expected = mass / 5 * np.array([y*y + z*z, x*x + z*z, x*x + y*y])
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
                rotation = rotation.reshape(3, 3)
                np.testing.assert_allclose(rotation @ np.diag(model.body_inertia[body]) @ rotation.T,
                                           np.diag(expected), rtol=0, atol=1e-13)
                np.testing.assert_array_equal(model.body_ipos[body], [0, 0, 0])

    def test_capsule_length_inertia_and_visual_collision_separation(self):
        profile = replace(self.profile, upper_arm_length=0.37, forearm_length=0.33,
                          thigh_length=0.49, shin_length=0.46)
        model = self.compile(profile)
        summary = summarize_model(model)
        self.assertTrue(summary["checks"]["primary_capsule_inertias"])
        for side in ("left", "right"):
            for segment, length, mass in (("upper_arm", 0.37, 2.4), ("forearm", 0.33, 1.5),
                                          ("thigh", 0.49, 7.5), ("shin", 0.46, 3.5)):
                body = model.body(f"{side}_{segment}").id
                geom = model.geom(f"{side}_{segment}_geom").id
                self.assertAlmostEqual(2 * model.geom_size[geom, 1], length)
                self.assertAlmostEqual(model.body_mass[body], mass)
                np.testing.assert_allclose(model.body_ipos[body], [0, 0, -length / 2], atol=1e-14)
            for segment, visual_radius, collision_radius in (("forearm", 0.031, 0.028), ("shin", 0.038, 0.038)):
                visual = model.geom(f"{side}_{segment}_geom").id
                self.assertEqual(model.geom_contype[visual], 0)
                self.assertEqual(model.geom_conaffinity[visual], 0)
                self.assertAlmostEqual(model.geom_size[visual, 0], visual_radius)
                body = model.body(f"{side}_{segment}").id
                colliders = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
                self.assertEqual(len(colliders), 1)
                collider = colliders[0]
                self.assertEqual(model.geom_rgba[collider, 3], 0)
                self.assertAlmostEqual(model.geom_size[collider, 0], collision_radius)
                if segment == "shin":
                    self.assertAlmostEqual(2 * model.geom_size[collider, 1], profile.shin_length - 0.003)
                    self.assertAlmostEqual(model.geom_pos[collider, 2] - model.geom_size[collider, 1]
                                           - collision_radius, -profile.shin_length - 0.035)

    def test_all_axes_order_and_independent_body_site_directions(self):
        evidence = measure_joint_semantics(self.model)
        self.assert_diagnostic(evidence)
        self.assertEqual(len(evidence["measurements"]), 25)
        for index, measurement in enumerate(evidence["measurements"], 1):
            self.assertEqual(self.model.joint(index).name, measurement["joint"])
            positive, negative = measurement["samples"]
            self.assertEqual((positive["angle_rad"], negative["angle_rad"]), (0.05, -0.05))
            self.assertAlmostEqual(positive["measured_body_angle_rad"], 0.05, places=14)
            self.assertAlmostEqual(negative["measured_body_angle_rad"], -0.05, places=14)
            name = measurement["joint"]
            # Explicit world-space sign checks, not another copy of measured axes.
            if "knee" in name:
                self.assertLess(positive["site_delta_world"][1], 0)
                self.assertGreater(negative["site_delta_world"][1], 0)
                self.assertFalse(negative["within_rom"])
                self.assertGreater(positive["body_rotation_world"][1][2], 0)
            elif "yaw" not in name and "roll" not in name:
                self.assertGreater(positive["site_delta_world"][1], 0)
                self.assertLess(negative["site_delta_world"][1], 0)
                self.assertLess(positive["body_rotation_world"][1][2], 0)
                if "elbow" in name:
                    self.assertFalse(negative["within_rom"])
                if name == "waist_pitch":
                    self.assertLess(positive["chest_delta_world"][1], 0)
                    self.assertGreater(negative["chest_delta_world"][1], 0)
                    self.assertLess(positive["head_top_delta_world"][1], 0)
                    self.assertGreater(negative["head_top_delta_world"][1], 0)
            elif "roll" in name:
                self.assertLess(positive["site_delta_world"][0], 0)
                self.assertGreater(negative["site_delta_world"][0], 0)
                self.assertGreater(positive["body_rotation_world"][0][2], 0)
            else:
                self.assertGreater(positive["off_axis_probe_delta_world"][1], 0)
                self.assertLess(negative["off_axis_probe_delta_world"][1], 0)
                self.assertGreater(positive["body_rotation_world"][1][0], 0)
                if "shoulder" in name:
                    np.testing.assert_allclose(positive["site_delta_world"], [0, 0, 0], atol=1e-12)
                elif "hip" in name:
                    self.assertLess(positive["site_delta_world"][0], 0)
                else:
                    self.assertLess(positive["site_delta_world"][1], 0)
        changed = copy.copy(self.model)
        changed.jnt_axis[changed.joint("left_elbow").id] = [-1, 0, 0]
        self.assertFalse(measure_joint_semantics(changed)["passed"])
        with patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", "site_xmat")), \
             self.assertRaisesRegex(ValueError, "sample data.site_xmat"):
            measure_joint_semantics(self.model)

    def test_ranges_sites_exclusion_signatures(self):
        for joint in self.summary["joints"][1:]:
            lo, hi = joint["range_rad"]
            self.assertTrue(np.isfinite([lo, hi]).all())
            self.assertLess(lo, hi)
            self.assertLessEqual(lo, 0)
            self.assertGreaterEqual(hi, 0)
        for side in ("left", "right"):
            for limb, position in (("hand", [0, 0, -0.04]), ("foot", [0, 0.09, -0.025])):
                site = self.model.site(f"{side}_{limb}_site").id
                self.assertEqual(self.model.site_bodyid[site], self.model.body(f"{side}_{limb}").id)
                np.testing.assert_allclose(self.model.site_pos[site], position, atol=1e-14)
        self.assertEqual(len(self.summary["exclusions"]), 15)
        for pair in self.summary["exclusions"]:
            first, second = pair["body_ids"]
            self.assertLess(first, second)
            self.assertEqual(pair["signature"], (first << 16) + second)
            self.assertEqual(pair["body_names"], [self.model.body(first).name, self.model.body(second).name])
        policy = self.summary["collision_policy"]
        self.assertEqual(policy["adjacent_overlap_exclusions"], 15)
        self.assertEqual(policy["temporary_environment_exclusions"], 0)
        self.assertEqual(policy["humanoid_enabled_geometries"], 21)
        self.assertEqual(policy["contact_mode"], ContactMode.PHYSICAL.value)
        self.assertTrue(all(e["classification"] == "adjacent_overlap_policy" for e in self.summary["exclusions"]))
        self.assertTrue(all("hand" in e["name"] for e in self.summary["equalities"]))

    def test_idealized_debug_report_classifies_legacy_debt_without_changing_human(self):
        debug = mujoco.MjModel.from_xml_string(build_mjcf(self.scene, self.profile,
                                                       contact_mode=ContactMode.IDEALIZED_DEBUG))
        report = summarize_model(debug)
        self.assert_diagnostic(report)
        self.assertEqual(report["collision_policy"]["contact_mode"], ContactMode.IDEALIZED_DEBUG.value)
        self.assertEqual(report["collision_policy"]["temporary_environment_exclusions"], 50)
        self.assertEqual(report["collision_policy"]["adjacent_overlap_exclusions"], 15)
        self.assertEqual((debug.neq, debug.nexclude), (30, 65))
        self.assertIn("not physical support", report["collision_policy"]["note"])
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                      "body_parentid", "jnt_range", "jnt_axis", "dof_damping", "dof_armature",
                      "actuator_gear", "site_pos", "site_bodyid"):
            np.testing.assert_array_equal(getattr(debug, field), getattr(self.model, field), err_msg=field)
        # Classifying legacy debt is not enough to pass a model labelled physical.
        debug.numeric("contact_mode").data[0] = 0.
        mislabeled = summarize_model(debug)
        self.assertTrue(mislabeled["checks"]["classified_exclusions"])
        self.assertFalse(mislabeled["checks"]["physical_contact_policy"])
        self.assertFalse(mislabeled["passed"])

    def test_profile_scaling_morphology_rom_and_label_only_arrays(self):
        evidence = profile_diagnostics(self.scene, self.profile)
        self.assert_diagnostic(evidence)
        self.assert_diagnostic(profile_diagnostics(self.scene, replace(self.profile, rom_scale=1.2)))
        domain = evidence["numerical_domain"]
        self.assertEqual(domain["sample_count"], 21)
        self.assertTrue(all(s["passed"] and not s["issues"] for s in domain["samples"]))
        dimensions = ("torso_length", "hip_width", "shoulder_width", "upper_arm_length",
                      "forearm_length", "thigh_length", "shin_length")
        for size, mass, strength in ((.05, .1, 1e-6), (2, 10, 10)):
            boundary = replace(self.profile, **dict.fromkeys(dimensions, size), mass_scale=mass, strength_scale=strength)
            report = profile_diagnostics(self.scene, boundary)
            self.assert_diagnostic(report)
            for requested in (0.7, 1.3):
                variant = report["variants"][f"mass_{requested}"]
                self.assertTrue(.1 <= variant["mass_scale"] <= 10)
                self.assertAlmostEqual(variant["actual_factor"], variant["mass_scale"] / mass)
                self.assertEqual(variant["requested_factor"], requested)
                motor = report["variants"][f"strength_{requested}"]
                self.assertTrue(1e-6 <= motor["strength_scale"] <= 10)
                self.assertAlmostEqual(motor["actual_factor"], motor["strength_scale"] / strength)
            for label in ("torso", "hip_width", "shoulder_width", "limbs"):
                self.assertTrue(all(.05 <= value <= 2 for value in report["variants"][label]["profile_updates"].values()))
        for scale in (1e-6, 0.5, 0.8, 1.0, 1.2):
            charts = evidence["variants"][f"rom_{scale}"]["coordinate_charts"]
            for chart in charts:
                self.assertEqual(chart["singular_angles_within_rom_rad"], [])
                if "shoulder_roll" in chart["middle_joint"]:
                    for sample in chart["rotation_jacobian_near_limits"]:
                        sine = abs(np.sin(sample["middle_angle_rad"]))
                        np.testing.assert_allclose(sample["singular_values"], [np.sqrt(1 + sine), 1, np.sqrt(1 - sine)], atol=1e-14)
                        self.assertGreater(min(sample["singular_values"]), 0.06)
                        self.assertAlmostEqual(abs(sample["middle_angle_rad"]), np.deg2rad(min(80 * scale, 85)) * (1 - 1e-6))
        other = self.compile(replace(self.profile, name="unrelated_label"))
        expected, actual = physics_numeric_arrays(self.model), physics_numeric_arrays(other)
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)

    def test_analytic_unforced_freefall(self):
        evidence = freefall_diagnostic(self.model)
        self.assert_diagnostic(evidence)
        self.assertEqual(evidence["max_contacts"], 0)
        self.assertGreater(evidence["initial_root_position"][2], 10)
        self.assertLess(evidence["velocity_error"], 1e-9)
        fine = freefall_diagnostic(self.model, timestep=0.001)
        self.assert_diagnostic(fine)
        self.assertEqual(fine["checks"], evidence["checks"])
        self.assertAlmostEqual(fine["position_error"] / evidence["position_error"], 0.5, places=9)
        np.testing.assert_allclose(fine["position_bias"], np.array(evidence["position_bias"]) / 2, atol=1e-12)
        for duration in (0, -1, np.nan, np.inf, .0001):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                freefall_diagnostic(self.model, duration=duration)
        for timestep in (0, -1, np.nan, np.inf):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                freefall_diagnostic(self.model, timestep=timestep)
        for field, at_time in (("qacc", 0), ("qvel", 0.4)):
            with self.subTest(forward_fault=field), \
                 patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", field, at_time=at_time)), \
                 self.assertRaisesRegex(ValueError, f"sample data.{field}"):
                freefall_diagnostic(self.model)
        with patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", "qvel", value=0.25, at_time=0.4)):
            terminal = freefall_diagnostic(self.model)
        self.assertFalse(terminal["passed"])
        self.assertFalse(terminal["checks"]["no_internal_motion"])
        self.assertEqual(terminal["max_internal_speed"], 0.25)
        json.dumps(terminal, allow_nan=False)

    def test_selected_and_all_joint_passive_decay_and_timestep_convergence(self):
        for selected in (None, ["waist_pitch"], ["left_elbow"], ["right_knee"], ["left_shoulder_yaw"]):
            with self.subTest(joints=selected):
                coarse = passive_decay_diagnostic(self.model, joints=selected)
                fine = passive_decay_diagnostic(self.model, joints=selected, timestep=0.001)
                self.assert_diagnostic(coarse)
                self.assert_diagnostic(fine)
                self.assertLess(coarse["energy_ratio"], 1)
                self.assertLess(fine["relative_velocity_reference_error"], coarse["relative_velocity_reference_error"] + 1e-7)
                self.assertLess(abs(coarse["energy_ratio"] - fine["energy_ratio"]), 0.01)
        for kwargs in ([{"duration": v} for v in (0, -1, np.nan, np.inf, .0001)]
                       + [{"timestep": v} for v in (0, -1, np.nan, np.inf)]
                       + [{"speed": v} for v in (0, -1, np.nan, np.inf, 1e-300)]
                       + [{"joints": []}, {"joints": ["root"]}]):
            with self.subTest(invalid=kwargs), self.assertRaises(ValueError):
                passive_decay_diagnostic(self.model, **kwargs)
        for function, field, at_time in (("mj_forward", "qacc", 0), ("mj_forward", "qacc", .2),
                                        ("mj_step", "qacc_warmstart", 0)):
            with self.subTest(engine_fault=(function, field, at_time)), \
                 patch(f"boulder_v1.model_validation.mujoco.{function}", side_effect=_engine_fault(function, field, at_time=at_time)), \
                 self.assertRaisesRegex(ValueError, f"sample data.{field}"):
                passive_decay_diagnostic(self.model)
        with np.errstate(over="ignore", invalid="ignore"), \
             patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", "qvel", value=1e200, at_time=.002)), \
             self.assertRaisesRegex(ValueError, "passive kinetic energy"):
            passive_decay_diagnostic(self.model)

    def test_mirrored_kinematics_and_unforced_coupled_dynamics(self):
        self.assert_diagnostic(mirror_diagnostic(self.model))
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0.2, -5, 20]
        data.qvel[:6] = [1, 2, 3, 4, 5, 6]
        data.qpos[self.model.joint("left_shoulder_roll").qposadr[0]] = 0.2
        data.qpos[self.model.joint("left_elbow").qposadr[0]] = 0.4
        data.qpos[self.model.joint("left_hip_yaw").qposadr[0]] = 0.1
        qpos, qvel = mirror_state(self.model, data.qpos, data.qvel)
        np.testing.assert_array_equal(qpos[:3], [-0.2, -5, 20])
        np.testing.assert_array_equal(qvel[:6], [-1, 2, 3, 4, -5, -6])
        self.assertEqual(qpos[self.model.joint("right_shoulder_roll").qposadr[0]], -0.2)
        self.assertEqual(qpos[self.model.joint("right_elbow").qposadr[0]], 0.4)
        self.assertEqual(qpos[self.model.joint("right_hip_yaw").qposadr[0]], -0.1)
        twice_pos, twice_vel = mirror_state(self.model, qpos, qvel)
        np.testing.assert_array_equal(twice_pos, data.qpos)
        np.testing.assert_array_equal(twice_vel, data.qvel)
        for duration in (0, -1, np.nan, np.inf, .0001):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                mirror_diagnostic(self.model, duration=duration)
        for field, at_time in (("qacc", 0), ("qpos", .2)):
            with self.subTest(forward_fault=field), \
                 patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", field, at_time=at_time)), \
                 self.assertRaisesRegex(ValueError, f"sample data.{field}"):
                mirror_diagnostic(self.model)
        with np.errstate(over="ignore", invalid="ignore"), \
             patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=_engine_fault("mj_forward", "qacc", value=np.finfo(float).max, index=4)), \
             self.assertRaisesRegex(ValueError, "mirror acceleration error"):
            mirror_diagnostic(self.model)
        original_difference = mujoco.mj_differentiatePos

        def corrupt_tangent(model, result, dt, first, second):
            original_difference(model, result, dt, first, second)
            result[-1] = np.nan

        with patch("boulder_v1.model_validation.mujoco.mj_differentiatePos", side_effect=corrupt_tangent), \
             self.assertRaisesRegex(ValueError, "mirror tangent position error"):
            mirror_diagnostic(self.model)

    def test_all_direct_motor_controls_strength_and_clamping(self):
        for scale in (0.7, 1.0, 1.3):
            model = self.compile(replace(self.profile, strength_scale=scale))
            evidence = actuator_diagnostic(model, strength_scale=scale)
            self.assert_diagnostic(evidence)
            self.assertEqual(len(evidence["samples"]), 150)
            for sample in evidence["samples"]:
                self.assertEqual(sample["actuator_force"], np.clip(sample["ctrl"], -1, 1))
                torque = dict((s[0], s[4]) for s in JOINT_SPECS)[sample["joint"]]
                self.assertAlmostEqual(sample["qfrc_actuator"], np.clip(sample["ctrl"], -1, 1) * torque * scale)
        original_forward = mujoco.mj_forward
        for field, value in (("actuator_force", np.nan), ("qfrc_actuator", np.inf), ("qacc", np.nan)):
            def corrupt_output(model, data):
                original_forward(model, data)
                if np.any(data.ctrl):
                    getattr(data, field)[-1] = value  # Also test faults outside the commanded motor/DOF.

            with self.subTest(output_field=field), patch("boulder_v1.model_validation.mujoco.mj_forward", side_effect=corrupt_output):
                evidence = actuator_diagnostic(self.model)
            self.assertFalse(evidence["passed"])
            self.assertTrue(evidence["checks"]["compiled_numeric_finite"])
            self.assertFalse(evidence["checks"]["finite_forward_outputs"])
            self.assertFalse(evidence["checks"]["all_150_force_torque_clamps"])
            self.assertIsNone(evidence["maximum_error"])
            self.assertTrue(all(s["full_vector_error"] is None for s in evidence["samples"]))
            self.assertEqual(evidence["nonfinite_observations"][0]["fields"][field]["values"], [None])
            json.dumps(evidence, allow_nan=False)

    def test_neutral_two_seconds_collision_enabled_finite_floor_impacts(self):
        evidence = neutral_rollout_diagnostic(self.model)
        self.assert_diagnostic(evidence)
        self.assertAlmostEqual(evidence["duration"], 2)
        self.assertGreater(evidence["floor_contact_steps"], 0)
        self.assertEqual(evidence["warnings"], {})
        for duration in (0, -1, np.nan, np.inf, .0001):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                neutral_rollout_diagnostic(self.model, duration=duration)
        with patch("boulder_v1.model_validation.mujoco.mj_step", side_effect=_engine_fault("mj_step", "qacc")), \
             self.assertRaisesRegex(ValueError, "sample data.qacc"):
            neutral_rollout_diagnostic(self.model)

    def test_diagnostics_do_not_mutate_source_model(self):
        before = physics_numeric_arrays(self.model)
        measure_joint_semantics(self.model)
        freefall_diagnostic(self.model)
        freefall_diagnostic(self.model, timestep=0.001)
        passive_decay_diagnostic(self.model)
        mirror_diagnostic(self.model)
        actuator_diagnostic(self.model)
        neutral_rollout_diagnostic(self.model)
        after = physics_numeric_arrays(self.model)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)

    def test_cli_strict_json_without_graphics_and_nonzero_on_failed_checks(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_model.py"
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MUJOCO_GL": "disable"}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), "--output", directory],
                                    env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evidence = json.loads((Path(directory) / "validation.json").read_text(),
                                  parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
            self.assertTrue(evidence["passed"])
            self.assertTrue(evidence["freefall_1ms"]["passed"])
            self.assertTrue(evidence["freefall_timestep_comparison"]["passed"])
            self.assertNotIn("render", evidence)
            self.assertFalse((Path(directory) / "neutral-com.png").exists())
        # Exercise exit status against an actual bad physical model, not mocked diagnostics.
        program = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); "
            "import re; import validate_model; import boulder_v1.model_validation as v; "
            "original=v.build_mjcf; v.build_mjcf=lambda s,p: re.sub("
            "r'(<motor name=\"act_waist_yaw\"[^>]*gear=\")[^\"]+', r'\\g<1>91.0', original(s,p)); "
            "raise SystemExit(validate_model.main(['--output',sys.argv[1]]))"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", program, directory], cwd=script.parents[1],
                                    env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            evidence = json.loads((Path(directory) / "validation.json").read_text())
            self.assertFalse(evidence["passed"])
            self.assertFalse(evidence["actuators"]["passed"])
        program = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); "
            "import validate_model; import boulder_v1.model_validation as v; "
            "original=v.mujoco.MjModel.from_xml_string\n"
            "def corrupt_compile(xml):\n"
            "    model=original(xml)\n"
            "    getattr(model,sys.argv[2]).flat[0]=float('nan')\n"
            "    return model\n"
            "def forbidden_forward(*args):\n"
            "    raise AssertionError('invalid compiled model was forwarded')\n"
            "v.mujoco.MjModel.from_xml_string=corrupt_compile\n"
            "v.mujoco.mj_forward=forbidden_forward\n"
            "raise SystemExit(validate_model.main(['--output',sys.argv[1],'--render']))"
        )
        for field in ("actuator_acc0", "dof_invweight0", "actuator_gainprm"):
            with self.subTest(cli_native_field=field), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-c", program, directory, field], cwd=script.parents[1],
                                        env=environment, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                evidence = json.loads((Path(directory) / "validation.json").read_text(),
                                      parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
                self.assertFalse(evidence["passed"])
                self.assertIn(f"model.{field}", evidence["compiled_numerics"]["issues"])
                self.assertFalse(evidence["checks"]["compiled_numerics"])
                self.assertNotIn("error", evidence)
                self.assertNotIn("render", evidence)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), "--output", directory, "--profile", ""],
                                    env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            evidence = json.loads((Path(directory) / "validation.json").read_text())
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["error"]["type"], "ValueError")
        program = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); "
            "import validate_model; import boulder_v1.model_validation as v; "
            "original=v.mujoco.mj_forward\n"
            "def corrupt_forward(model,data):\n"
            "    original(model,data)\n"
            "    data.qacc[-1]=float('nan')\n"
            "def forbidden_render(*args):\n"
            "    raise AssertionError('invalid measurement was rendered')\n"
            "v.mujoco.mj_forward=corrupt_forward\n"
            "validate_model._render_neutral=forbidden_render\n"
            "raise SystemExit(validate_model.main(['--output',sys.argv[1],'--render']))"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", program, directory], cwd=script.parents[1],
                                    env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            evidence = json.loads((Path(directory) / "validation.json").read_text(),
                                  parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["error"]["type"], "ValueError")
            self.assertIn("sample data.qacc", evidence["error"]["message"])
            self.assertNotIn("render", evidence)
            self.assertFalse((Path(directory) / "neutral-com.png").exists())
        # Raw XML bypasses the builder's strength domain. Native overflow is
        # version-dependent, but even finite huge gears must fail the motor contract.
        base_xml = build_mjcf(self.scene, self.profile)
        huge_xml = re.sub(r'gear="[^"]+"', 'gear="1e152"', base_xml)
        with patch("boulder_v1.model_validation.build_mjcf", return_value=huge_xml):
            evidence = validate_physical_model()
        self.assertFalse(evidence["passed"])
        json.dumps(evidence, allow_nan=False)
        program = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); "
            "import re; import validate_model; import boulder_v1.model_validation as v; "
            "original=v.build_mjcf; v.build_mjcf=lambda s,p: re.sub(r'gear=\"[^\"]+\"', 'gear=\"1e152\"', original(s,p)); "
            "raise SystemExit(validate_model.main(['--output',sys.argv[1]]))"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", program, directory], cwd=script.parents[1],
                                    env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            evidence = json.loads((Path(directory) / "validation.json").read_text(),
                                  parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
            self.assertFalse(evidence["passed"])
            self.assertNotIn("Out of range float values", str(evidence.get("error", {})))


if __name__ == "__main__":
    unittest.main()
