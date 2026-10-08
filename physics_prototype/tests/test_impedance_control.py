"""Stage3 controller contract and isolated dynamics, not Stage2 contact tests."""

import copy
from dataclasses import FrozenInstanceError
import json
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.controller_diagnostics import (
    LEGACY_BASELINE,
    REPRESENTATIVE_JOINTS,
    TORQUE_NOISE_FLOOR,
    VELOCITY_NOISE_FLOOR,
    _signal_metrics,
    make_isolated_fixture,
    run_isolated_case,
    run_isolated_suite,
)
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import TorqueCommand, compile_model, compute_pose_control
from boulder_v1.schema import BoulderScene, ClimberProfile


class ImpedanceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = ClimberProfile(name="production_reference")
        cls.production_xml = build_mjcf(BoulderScene("xyz_metres", 1., (), ()), cls.profile)
        # Compilation/forward checks only. Never integrate the full climber here.
        cls.production = compile_model(cls.production_xml)
        cls.suite = run_isolated_suite()
        cls.cases = {
            (case["configuration"]["joint"], case["configuration"]["timestep_s"], case["case"]): case
            for case in cls.suite["cases"]
        }

    def case(self, name, dt, label):
        return self.cases[name, dt, label]

    def test_fixtures_preserve_production_subtree_inertia_geometry_joint_motor(self):
        source = ET.fromstring(self.production_xml)
        for name in REPRESENTATIVE_JOINTS:
            with self.subTest(joint=name):
                fixture = make_isolated_fixture(name, profile=self.profile)
                model = fixture.model
                isolated = ET.fromstring(fixture.xml)
                self.assertEqual((model.nq, model.nv, model.nu, model.njnt, model.neq), (1, 1, 1, 1, 0))
                self.assertEqual((model.npair, model.nexclude), (0, 0))
                np.testing.assert_array_equal(model.opt.gravity, [0, 0, 0])
                self.assertEqual(model.opt.integrator, self.production.opt.integrator)
                self.assertEqual(ET.tostring(isolated.find("compiler")), ET.tostring(source.find("compiler")))
                self.assertEqual(ET.tostring(isolated.find("default")), ET.tostring(source.find("default")))
                self.assertEqual(isolated.find(f".//joint[@name='{name}']").attrib,
                                 source.find(f".//joint[@name='{name}']").attrib)
                self.assertEqual(isolated.find("./actuator/motor").attrib,
                                 source.find(f"./actuator/motor[@joint='{name}']").attrib)
                self.assertEqual(model.body("isolated_mount").jntnum[0], 0)
                self.assertEqual(model.body("isolated_mount").mass[0], 0)
                jid = self.production.joint(name).id
                dof = int(self.production.jnt_dofadr[jid])
                for field in ("jnt_axis", "jnt_range", "jnt_pos", "jnt_type", "jnt_limited"):
                    np.testing.assert_array_equal(getattr(model, field)[0], getattr(self.production, field)[jid])
                self.assertEqual(model.dof_damping[0], self.production.dof_damping[dof])
                self.assertEqual(model.dof_armature[0], self.production.dof_armature[dof])
                self.assertEqual(model.dof_armature[0], .01)
                self.assertGreater(fixture.effective_inertia_kg_m2, .01)
                self.assertFalse(np.any(model.geom_contype) or np.any(model.geom_conaffinity))
                for body in range(2, model.nbody):
                    body_name = model.body(body).name
                    original = self.production.body(body_name).id
                    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
                        np.testing.assert_array_equal(getattr(model, field)[body], getattr(self.production, field)[original],
                                                      err_msg=f"{body_name}.{field}")
                for geom in range(model.ngeom):
                    geom_name = model.geom(geom).name
                    original = self.production.geom(geom_name).id
                    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction"):
                        np.testing.assert_array_equal(getattr(model, field)[geom], getattr(self.production, field)[original],
                                                      err_msg=f"{geom_name}.{field}")
                    original_xml = source.find(f".//geom[@name='{geom_name}']")
                    expected = {**original_xml.attrib, "contype": "0", "conaffinity": "0"}
                    self.assertEqual(isolated.find(f".//geom[@name='{geom_name}']").attrib, expected)
                parents = {child: parent for parent in source.iter() for child in parent}
                original_joint = source.find(f".//joint[@name='{name}']")
                original_body = parents[original_joint]
                expected_removed = {j.get("name") for j in original_body.iter("joint") if j.get("name") != name}
                self.assertEqual(set(fixture.removed_joints), expected_removed)
                self.assertEqual(fixture.proximal_body, parents[original_body].get("name"))

    def test_scalar_and_joint_mapping_gains_velocity_feedforward_actual_torque(self):
        for name in REPRESENTATIVE_JOINTS:
            fixture = make_isolated_fixture(name)
            model = fixture.model
            for kp, kd in ((35., 4.), ({name: 35.}, {name: 4.})):
                with self.subTest(joint=name, mapping=isinstance(kp, dict)):
                    data = mujoco.MjData(model)
                    data.qpos[0], data.qvel[0] = .05, .25
                    command = compute_pose_control(model, data, {name: .1}, kp=kp, kd=kd,
                                                   target_velocity={name: .5}, feedforward={name: -.3})
                    expected = 35 * (.1 - .05) + 4 * (.5 - .25) - .3
                    self.assertIsInstance(command, TorqueCommand)
                    self.assertEqual(command.joint_names, (name,))
                    self.assertEqual(command.stiffness_Nm_rad, (35.,))
                    self.assertEqual(command.damping_Nms_rad, (4.,))
                    self.assertAlmostEqual(command.desired_Nm[0], expected)
                    self.assertAlmostEqual(command.commanded_Nm[0], expected)
                    gear = float(model.actuator_gear[0, 0])
                    self.assertEqual(command.limits_Nm, (gear,))
                    self.assertAlmostEqual(data.ctrl[0], expected / gear)
                    self.assertAlmostEqual(command.utilization[0], abs(expected) / gear)
                    self.assertEqual(command.saturated, (False,))
                    mujoco.mj_forward(model, data)
                    np.testing.assert_allclose(data.qfrc_actuator, [expected], rtol=0, atol=1e-12)
                    for field in command.__dataclass_fields__:
                        self.assertIsInstance(getattr(command, field), tuple)
                    with self.assertRaises(FrozenInstanceError):
                        command.desired_Nm = (0.,)

    def test_physical_torque_clamps_then_divides_by_gear_both_signs_and_boundary(self):
        for name in REPRESENTATIVE_JOINTS:
            model = make_isolated_fixture(name).model
            data = mujoco.MjData(model)
            limit = float(model.actuator_gear[0, 0])
            for fraction in (-2., -1., -.25, 0., .25, 1., 2.):
                with self.subTest(joint=name, fraction=fraction):
                    command = compute_pose_control(model, data, {name: 0.}, kp=0., kd=0.,
                                                   feedforward={name: fraction * limit})
                    torque = np.clip(fraction, -1., 1.) * limit
                    self.assertEqual(command.desired_Nm, (fraction * limit,))
                    self.assertEqual(command.commanded_Nm, (torque,))
                    self.assertEqual(command.utilization, (abs(np.clip(fraction, -1., 1.)),))
                    self.assertEqual(command.saturated, (abs(fraction) > 1.,))
                    self.assertEqual(data.ctrl[0], np.clip(fraction, -1., 1.))
                    mujoco.mj_forward(model, data)
                    self.assertAlmostEqual(data.qfrc_actuator[0], torque)

    def test_invalid_references_gains_and_state_rejected_before_any_ctrl_write(self):
        model = self.production
        first = "left_shoulder_pitch"
        last = "right_ankle_roll"
        last_id = model.joint(last).id
        above_rom = float(model.jnt_range[last_id, 1]) + .01
        bad_arguments = (
            {"target_pose": {first: .1, last: above_rom}},
            {"target_pose": {first: .1, last: np.nan}},
            {"target_pose": {first: .1, last: np.inf}},
            {"target_pose": {"not_a_joint": .1}},
            {"target_pose": {"root": 0.}},
            {"target_pose": {}, "target_velocity": {last: np.nan}},
            {"target_pose": {}, "target_velocity": {"not_a_joint": 0.}},
            {"target_pose": {}, "feedforward": {last: np.inf}},
            {"target_pose": {}, "feedforward": {"not_a_joint": 0.}},
            {"target_pose": {}, "kp": -1.},
            {"target_pose": {}, "kd": -1.},
            {"target_pose": {}, "kp": np.nan},
            {"target_pose": {}, "kd": np.inf},
            {"target_pose": {last: .1}, "kp": {model.joint(i).name: (-1. if i == last_id else 20.)
                                                       for i in range(1, model.njnt)}},
        )
        for kwargs in bad_arguments:
            with self.subTest(arguments=kwargs):
                data = mujoco.MjData(model)
                data.ctrl[:] = np.linspace(-.3, .3, model.nu)
                before = data.ctrl.copy()
                with self.assertRaises(ValueError):
                    compute_pose_control(model, data, **kwargs)
                np.testing.assert_array_equal(data.ctrl, before)
        for field, address in (("qpos", model.jnt_qposadr[last_id]), ("qvel", model.jnt_dofadr[last_id])):
            with self.subTest(state=field):
                data = mujoco.MjData(model)
                data.ctrl[:] = .123
                getattr(data, field)[address] = np.nan
                with self.assertRaises(ValueError):
                    compute_pose_control(model, data, {})
                np.testing.assert_array_equal(data.ctrl, np.full(model.nu, .123))

    def test_invalid_direct_motor_capabilities_rejected_atomically(self):
        mutations = (
            ("actuator_gear", (0, 0), 0.), ("actuator_gear", (0, 0), -1.),
            ("actuator_gear", (0, 0), np.nan), ("actuator_gear", (0, 1), 1.),
            ("actuator_ctrllimited", 0, False), ("actuator_ctrlrange", (0, 0), 0.),
            ("actuator_ctrlrange", (0, 1), 0.), ("actuator_forcelimited", 0, True),
            ("actuator_gainprm", (0, 0), 2.),
            ("actuator_dyntype", 0, mujoco.mjtDyn.mjDYN_FILTER),
            ("actuator_gaintype", 0, mujoco.mjtGain.mjGAIN_AFFINE),
            ("actuator_biastype", 0, mujoco.mjtBias.mjBIAS_AFFINE),
        )
        fixture = make_isolated_fixture("left_wrist")
        for field, index, value in mutations:
            with self.subTest(field=field, value=value):
                model = copy.copy(fixture.model)
                getattr(model, field)[index] = value
                data = mujoco.MjData(model)
                data.ctrl[:] = .123
                with self.assertRaises(ValueError):
                    compute_pose_control(model, data, {fixture.joint_name: .05})
                np.testing.assert_array_equal(data.ctrl, [.123])

    def test_class_defaults_independent_of_strength_and_profile_name(self):
        defaults = {
            "waist_pitch": (160., 25.), "left_shoulder_pitch": (80., 12.),
            "left_elbow": (60., 6.), "left_wrist": (20., 1.),
            "left_hip_pitch": (160., 25.), "left_knee": (120., 12.), "left_ankle_pitch": (40., 2.),
        }
        for name, gains in defaults.items():
            commands = []
            fixtures = []
            for label, strength in (("weak_name", .5), ("arbitrary_rename", .5), ("strong_name", 1.5)):
                with self.subTest(joint=name, profile=label):
                    fixture = make_isolated_fixture(name, profile=ClimberProfile(name=label, strength_scale=strength))
                    fixtures.append(fixture)
                    command = compute_pose_control(fixture.model, mujoco.MjData(fixture.model), {name: .05})
                    commands.append(command)
                    self.assertEqual((command.stiffness_Nm_rad[0], command.damping_Nms_rad[0]), gains)
                    self.assertEqual(command.desired_Nm, (.05 * gains[0],))
                    self.assertEqual(command.saturated, (False,))
            self.assertEqual(commands[0], commands[1])
            self.assertEqual(commands[0].commanded_Nm, commands[2].commanded_Nm)
            self.assertAlmostEqual(commands[2].limits_Nm[0] / commands[0].limits_Nm[0], 3.)
            for fixture in fixtures[1:]:
                self.assertEqual(fixture.effective_inertia_kg_m2, fixtures[0].effective_inertia_kg_m2)
                for field in ("body_mass", "body_inertia", "dof_damping", "dof_armature", "jnt_axis", "jnt_range"):
                    np.testing.assert_array_equal(getattr(fixture.model, field), getattr(fixtures[0].model, field))

    def test_unsaturated_small_response_equal_across_strengths_at_both_timesteps(self):
        for name in REPRESENTATIVE_JOINTS:
            for dt in (.002, .001):
                with self.subTest(joint=name, timestep=dt):
                    weak = run_isolated_case(make_isolated_fixture(name, profile=ClimberProfile(name="weak", strength_scale=.5),
                                                                  timestep=dt), include_traces=True)
                    strong = run_isolated_case(make_isolated_fixture(name, profile=ClimberProfile(name="strong", strength_scale=1.5),
                                                                    timestep=dt), include_traces=True)
                    self.assertEqual(weak["metrics"]["saturated_steps"], 0)
                    self.assertEqual(strong["metrics"]["saturated_steps"], 0)
                    for field in ("position_rad", "velocity_rad_s", "commanded_Nm"):
                        np.testing.assert_allclose(weak["traces"][field], strong["traces"][field], rtol=0, atol=1e-12)

    def test_demanding_saturation_capability_difference_without_gain_retuning(self):
        for name in REPRESENTATIVE_JOINTS:
            for dt in (.002, .001):
                with self.subTest(joint=name, timestep=dt):
                    label = "demanding_weak_0.1" if name == "left_ankle_pitch" else "demanding_weak"
                    weak, strong = self.case(name, dt, label), self.case(name, dt, "demanding_strong")
                    for field in ("stiffness_Nm_rad", "damping_Nms_rad", "target_rad"):
                        self.assertEqual(weak["configuration"][field], strong["configuration"][field])
                    self.assertGreater(weak["metrics"]["saturated_steps"], 0)
                    self.assertEqual(strong["metrics"]["saturated_steps"], 0)
                    self.assertEqual(weak["metrics"]["peak_utilization"], 1.)
                    self.assertLessEqual(strong["metrics"]["peak_utilization"], .8 + 1e-12)
                    self.assertEqual(weak["metrics"]["torque_peak_Nm"], weak["configuration"]["limit_Nm"])
                    self.assertGreater(strong["metrics"]["torque_peak_Nm"], weak["metrics"]["torque_peak_Nm"])
                    self.assertGreaterEqual(weak["metrics"]["settle_s"], strong["metrics"]["settle_s"] - dt)
                    if name == "left_ankle_pitch":
                        self.assertEqual(self.case(name, dt, "demanding_weak")["metrics"]["saturated_steps"], 0)
                        self.assertEqual(weak["configuration"]["strength_scale"], .1)
                    small = self.case(name, dt, "small")
                    for field in ("stiffness_Nm_rad", "damping_Nms_rad"):
                        self.assertEqual(weak["configuration"][field], small["configuration"][field])

    def test_small_and_moderate_convergence_across_classes_and_timesteps(self):
        self.assertTrue(self.suite["passed"])
        self.assertEqual((self.suite["case_count"], self.suite["physical_case_count"]), (90, 86))
        for name in REPRESENTATIVE_JOINTS:
            for dt in (.002, .001):
                for label in ("small", "moderate"):
                    with self.subTest(joint=name, timestep=dt, case=label):
                        result = self.case(name, dt, label)
                        self.assertTrue(result["passed"], result)
                        metrics = result["metrics"]
                        self.assertLess(metrics["settle_s"], 1.)
                        self.assertGreater(metrics["rise_10_90_s"], 0.)
                        self.assertLess(metrics["overshoot_fraction"], .1)
                        self.assertLess(metrics["tail_max_error_rad"], 1e-5)
                        self.assertLess(metrics["tail_rms_error_rad"], 1e-5)
                        self.assertLess(metrics["tail_max_speed_rad_s"], 1e-4)
                        self.assertLess(metrics["tail_rms_speed_rad_s"], 1e-4)
                        self.assertEqual(metrics["tail_velocity"]["sign_changes"], 0)
                        self.assertEqual(metrics["tail_torque"]["sign_changes"], 0)

    def test_small_external_torque_pulse_recovers_at_both_timesteps(self):
        for name in REPRESENTATIVE_JOINTS:
            for dt in (.002, .001):
                with self.subTest(joint=name, timestep=dt):
                    result = self.case(name, dt, "pulse")
                    self.assertEqual(result["configuration"]["pulse"], [1., .05, .5])
                    self.assertTrue(result["passed"], result)
                    self.assertGreater(result["metrics"]["pulse_peak_error_rad"], 1e-4)
                    self.assertLess(result["metrics"]["pulse_recovery_s"], .3)
                    self.assertLess(result["metrics"]["tail_max_error_rad"], 1e-5)
                    self.assertLess(result["metrics"]["tail_max_speed_rad_s"], 1e-4)

    def test_timestep_responses_approximately_agree_not_just_final_positions(self):
        for name in REPRESENTATIVE_JOINTS:
            with self.subTest(joint=name):
                coarse = run_isolated_case(make_isolated_fixture(name, timestep=.002), target=.5, include_traces=True)
                fine = run_isolated_case(make_isolated_fixture(name, timestep=.001), target=.5, include_traces=True)
                for field, tolerance in (("position_rad", .0025), ("velocity_rad_s", .04)):
                    np.testing.assert_allclose(coarse["traces"][field], fine["traces"][field][::2], rtol=0, atol=tolerance)
                for field in ("rise_10_90_s", "settle_s"):
                    self.assertLessEqual(abs(coarse["metrics"][field] - fine["metrics"][field]), .004)
                self.assertLess(abs(coarse["metrics"]["overshoot_fraction"] - fine["metrics"]["overshoot_fraction"]), .002)

    def test_distal_legacy_period_two_chatter_reduced_by_physical_defaults(self):
        for name in ("left_wrist", "left_ankle_pitch"):
            with self.subTest(joint=name):
                before = self.case(name, .002, "small_legacy_NOT_PRODUCTION")
                after = self.case(name, .002, "small")
                self.assertEqual(before["configuration"]["controller"], LEGACY_BASELINE)
                self.assertEqual(before["configuration"]["stiffness_Nm_rad"], 5 * before["configuration"]["limit_Nm"])
                self.assertEqual(before["configuration"]["damping_Nms_rad"], .5 * before["configuration"]["limit_Nm"])
                self.assertFalse(before["passed"])
                self.assertIsNone(before["metrics"]["settle_s"])
                self.assertGreater(before["metrics"]["tail_rms_speed_rad_s"], 1.)
                self.assertGreater(before["metrics"]["saturated_fraction"], .95)
                for signal in ("tail_velocity", "tail_torque"):
                    self.assertGreater(before["metrics"][signal]["sign_changes"], 200)
                    self.assertGreater(before["metrics"][signal]["successive_flip_fraction"], .99)
                    self.assertGreater(before["metrics"][signal]["dominant_frequency_Hz"], 240.)
                    self.assertGreater(before["metrics"][signal]["high_frequency_power_fraction"], .95)
                    self.assertEqual(after["metrics"][signal]["sign_changes"], 0)
                    self.assertEqual(after["metrics"][signal]["successive_flip_fraction"], 0.)
                self.assertLess(after["metrics"]["tail_rms_speed_rad_s"], 1e-5)
                self.assertEqual(after["metrics"]["saturated_steps"], 0)
                self.assertTrue(after["passed"])
                # The baseline is timestep-sensitive; do not claim chatter at 1ms.
                fine_legacy = self.case(name, .001, "small_legacy_NOT_PRODUCTION")
                self.assertTrue(fine_legacy["passed"])
                self.assertEqual(fine_legacy["metrics"]["tail_velocity"]["sign_changes"], 0)

    def test_optional_traces_strict_json_and_fixture_not_mutated(self):
        json.dumps(self.suite, allow_nan=False)
        self.assertTrue(all("traces" not in case for case in self.suite["cases"]))
        fixture = make_isolated_fixture("left_wrist")
        fields = ("body_mass", "body_inertia", "dof_damping", "dof_armature", "actuator_gear", "jnt_range")
        before = {field: getattr(fixture.model, field).copy() for field in fields}
        result = run_isolated_case(fixture, duration=.2, include_traces=True)
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["traces"]["time_s"]), 101)
        self.assertEqual(len(result["traces"]["position_rad"]), 101)
        self.assertEqual(len(result["traces"]["commanded_Nm"]), 100)
        repeat = run_isolated_case(fixture, duration=.2, include_traces=True)
        self.assertEqual(result, repeat)
        for field in fields:
            np.testing.assert_array_equal(getattr(fixture.model, field), before[field])
        # A two-step diagnostic still has a nonempty tail and valid JSON.
        json.dumps(run_isolated_case(fixture, duration=.004), allow_nan=False)

    def test_numerical_convergence_noise_not_classified_as_chatter(self):
        fixture = make_isolated_fixture("left_wrist")
        result = run_isolated_case(fixture, initial_position=.05, initial_velocity=1e-12)
        self.assertTrue(result["passed"])
        for signal in ("velocity", "torque", "tail_velocity", "tail_torque"):
            self.assertEqual(result["metrics"][signal]["sign_changes"], 0)
            self.assertIsNone(result["metrics"][signal]["dominant_frequency_Hz"])
            self.assertEqual(result["metrics"][signal]["high_frequency_power_fraction"], 0.)
        alternating = (-1.) ** np.arange(128)
        for floor in (VELOCITY_NOISE_FLOOR, TORQUE_NOISE_FLOOR):
            noise = _signal_metrics(alternating * floor / 1000, floor, .002)
            chatter = _signal_metrics(alternating * floor * 2, floor, .002)
            self.assertEqual(noise["sign_changes"], 0)
            self.assertIsNone(noise["dominant_frequency_Hz"])
            self.assertEqual(chatter["sign_changes"], 127)
            self.assertEqual(chatter["successive_flip_fraction"], 1.)
            self.assertEqual(chatter["dominant_frequency_Hz"], 250.)

    def test_diagnostic_admission_rejects_invalid_cases(self):
        for name in ("root", "unknown"):
            with self.subTest(joint=name), self.assertRaises(ValueError):
                make_isolated_fixture(name)
        for dt in (0., -1., np.nan, np.inf):
            with self.subTest(timestep=dt), self.assertRaises(ValueError):
                make_isolated_fixture(timestep=dt)
        fixture = make_isolated_fixture()
        invalid_cases = (
            {"target": np.nan}, {"target": 2.}, {"initial_position": 2.},
            {"initial_velocity": np.inf}, {"duration": .001}, {"duration": 0.},
            {"controller": "unknown"}, {"controller": LEGACY_BASELINE, "kp": 5.},
            {"pulse": (-.1, .05, .5)}, {"pulse": (1., .0001, .5)},
            {"pulse": (2.99, .05, .5)}, {"pulse": (1., .05, np.nan)},
        )
        for kwargs in invalid_cases:
            with self.subTest(arguments=kwargs), self.assertRaises(ValueError):
                run_isolated_case(fixture, **kwargs)
        with self.assertRaises(ValueError):
            run_isolated_suite(joints=())
        with self.assertRaises(ValueError):
            run_isolated_suite(timesteps=())


if __name__ == "__main__":
    unittest.main()
