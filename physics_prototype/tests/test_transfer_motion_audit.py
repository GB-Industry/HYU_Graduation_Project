"""Kinematic unit evidence only: no new physical-success cases or simulation."""
import copy
from contextlib import ExitStack
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1.transfers import make_transfer_fixture
from scripts.transfer_motion_audit import audit_motion, trajectory_rows


def manual_record(model, seed, limb="RIGHT_HAND", curve="translation", start=2.):
    """FakeManualPose: known manifold samples, explicitly not physical evidence."""
    poses = []
    for t in (0., .5, 1., .5, 0.):
        q = seed.copy()
        if curve == "translation":
            q[:3] += t * np.array([.02, -.01, .03])
        elif curve == "rotation":
            angle = np.radians(20.) * t
            q[3:7] = [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
        elif curve == "limb":
            side = limb.split("_")[0].lower()
            name = side + ("_elbow" if limb.endswith("HAND") else "_knee")
            q[model.joint(name).qposadr[0]] += .2 * t
            q[model.joint("waist_yaw").qposadr[0]] += .01 * t
            q[model.joint("right_ankle_roll").qposadr[0]] += .03 * t
        else:
            raise ValueError(curve)
        poses.append(q)

    def state(q, time):
        return {"time": time, "qpos": q.tolist(), "qvel": [0.] * model.nv, "ctrl": [0.] * model.nu,
                "qacc_warmstart": [0.] * model.nv, "eq_active": [False] * model.neq,
                "finite": True, "contact_configuration": {limb: "source"},
                "hand_states": {"LEFT_HAND": {"active": True, "valid": True, "load": 10.,
                                              "capacity": 100., "margin": 90.}},
                "foot_states": {"RIGHT_FOOT": {"normal_force": 100., "tangential_force": 3.,
                                              "tangential_speed": .001, "supporting": True}},
                "capture_events": [], "release_events": [], "contact_mode": "FakeManualPose"}

    initial = state(poses[0], start)
    samples = []
    for i, q in enumerate(poses[1:], 1):
        snapshot = state(q, start + i * .002)
        samples.append({"time_s": snapshot.pop("time"), **snapshot, "steps": i, "phase": "REACH",
                        "hands": snapshot["hand_states"], "feet": snapshot["foot_states"],
                        "q_ref": (q + 999.).tolist(), "qd_ref": [0.] * model.nv})
    return {"kind": "foot" if limb.endswith("FOOT") else limb.lower(), "moving_limb": limb,
            "source": "source", "target": "target", "dt_s": .002, "status": "FakeManualPose",
            "initial_state": initial, "final_state": state(poses[-1], start + .008), "samples": samples,
            "events": [{"phase": "REACH", "time_s": start},
                       {"event": "RELEASED", "time_s": start + .002},
                       {"event": "CAPTURED" if limb.endswith("HAND") else "ACQUIRED", "time_s": start + .004},
                       {"phase": "SETTLE", "time_s": start + .004},
                       {"event": "FINAL_READY", "time_s": start + .008}]}


class TransferMotionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.live_data, _, _, cls.seed = make_transfer_fixture()

    def meter(self, evidence):
        with ExitStack() as stack:
            for name in ("mj_forward", "mj_step", "mj_step1", "mj_step2", "mj_inverse", "mj_resetData",
                         "mj_integratePos", "mj_normalizeQuat"):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError(f"Forbidden {name}")))
            return audit_motion(self.model, evidence)

    def expected_geometry(self, evidence):
        data = mujoco.MjData(self.model)
        root = self.model.body("climber_root").id
        descendants = []
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == root:
                descendants.append(body)
        mass = self.model.body_mass[descendants]
        result = {name: [] for name in ("root", "pelvis", "pelvis_body_com", "climber_com", "effector")}
        for row in sorted(trajectory_rows(evidence), key=lambda r: r["record_index"]):
            data.qpos[:] = row["qpos"]
            mujoco.mj_kinematics(self.model, data)
            # Independently sum body inertial positions, not subtree_com.
            result["climber_com"].append(np.sum(data.xipos[descendants] * mass[:, None], axis=0) / np.sum(mass))
            result["root"].append(data.xpos[root].copy())
            result["pelvis"].append(data.body("pelvis").xpos.copy())
            result["pelvis_body_com"].append(data.body("pelvis").xipos.copy())
            result["effector"].append(data.site(row["moving_limb"].lower() + "_site").xpos.copy())
        return {name: np.array(points) for name, points in result.items()}

    def test_translation_curve_uses_all_native_rows_not_references(self):
        record = manual_record(self.model, self.seed)
        record["underlying_result"] = copy.deepcopy(record)
        record["setup_initial_state"] = {"time": 0., "qpos": [999.] * self.model.nq}
        record["initial_static"] = {"final_state": copy.deepcopy(record["initial_state"])}
        result = self.meter(record)
        transfer = result["transfers"][0]
        distance = np.linalg.norm([.02, -.01, .03])
        for metric in [transfer["moving_effector"], *transfer["bodies"].values()]:
            self.assertAlmostEqual(metric["net_displacement_m"], 0., places=12)
            self.assertAlmostEqual(metric["max_distance_from_start_m"], distance, places=12)
            self.assertAlmostEqual(metric["path_length_m"], 2 * distance, places=12)
            self.assertAlmostEqual(metric["largest_successive_step_m"], distance / 2, places=12)
            np.testing.assert_allclose(metric["xyz_peak_to_peak_m"], [.02, .01, .03], atol=1e-12)
        self.assertEqual(transfer["timing"]["native_sample_count"], 4)
        self.assertEqual(transfer["timing"]["native_row_count"], 6)
        self.assertEqual(transfer["timing"]["duplicate_time_count"], 1)
        self.assertEqual(transfer["timing"]["sample_gap_count"], 0)
        self.assertAlmostEqual(transfer["timing"]["time_span_s"], .008)
        self.assertIsNone(result["quality_verdict"])
        self.assertTrue(result["source_matches_initial_static_final_state"])
        self.assertNotIn("passed", result)
        self.assertIsNone(transfer["events"][-1]["event_after"])
        self.assertIsNone(transfer["events"][-1]["native_step"])
        json.dumps(result, allow_nan=False)

    def test_rotation_curve_and_quaternion_sign_are_measured_relatively(self):
        record = manual_record(self.model, self.seed, curve="rotation")
        record["samples"][1]["qpos"][3:7] = [-x for x in record["samples"][1]["qpos"][3:7]]
        result = self.meter(record)["transfers"][0]
        self.assertAlmostEqual(result["root_orientation"]["net_relative_angle_deg"], 0.)
        self.assertAlmostEqual(result["root_orientation"]["max_relative_angle_deg"], 20.)
        self.assertAlmostEqual(result["root_orientation"]["largest_successive_angle_deg"], 10.)
        self.assertAlmostEqual(result["bodies"]["root"]["max_distance_from_start_m"], 0.)
        self.assertGreater(result["moving_effector"]["max_distance_from_start_m"], 0.)

    def test_limb_curves_match_fk_massweighted_com_and_all_compiled_joint_axes(self):
        for limb in ("RIGHT_HAND", "LEFT_HAND", "LEFT_FOOT", "RIGHT_FOOT"):
            with self.subTest(limb=limb):
                record = manual_record(self.model, self.seed, limb, "limb")
                expected = self.expected_geometry(record)
                transfer = self.meter(record)["transfers"][0]
                for name, metric in [("effector", transfer["moving_effector"]), *transfer["bodies"].items()]:
                    points = expected[name]
                    np.testing.assert_allclose(metric["initial_world_m"], points[0], atol=1e-12)
                    np.testing.assert_allclose(metric["final_world_m"], points[-1], atol=1e-12)
                    self.assertAlmostEqual(metric["max_distance_from_start_m"],
                                           np.max(np.linalg.norm(points - points[0], axis=1)), places=12)
                    self.assertAlmostEqual(metric["path_length_m"], np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)), places=12)
                self.assertGreater(transfer["bodies"]["climber_com"]["max_distance_from_start_m"], 0.)
                self.assertEqual(transfer["bodies"]["pelvis"]["max_distance_from_start_m"], 0.)
                self.assertEqual(len(transfer["joints"]), 25)
                self.assertEqual(len(transfer["relevant_joints"]), 5 if limb.endswith("HAND") else 6)
                self.assertAlmostEqual(transfer["joints"]["waist_yaw"]["peak_to_peak_rad"], .01)
                self.assertAlmostEqual(transfer["joints"]["right_ankle_roll"]["peak_to_peak_rad"], .03)
                self.assertAlmostEqual(transfer["largest_joint_excursion"]["rad"], .2)
                self.assertAlmostEqual(transfer["largest_joint_increment"]["rad"], .1)
                for name, metric in transfer["joints"].items():
                    joint = self.model.joint(name)
                    self.assertEqual(metric["joint_id"], joint.id)
                    self.assertEqual(metric["qpos_address"], joint.qposadr[0])
                    self.assertEqual(metric["dof_address"], joint.dofadr[0])
                    np.testing.assert_array_equal(metric["axis_local"], joint.axis)
                    self.assertAlmostEqual(metric["peak_to_peak_deg"], np.degrees(metric["peak_to_peak_rad"]))

    def test_event_brackets_keep_pre_activation_row_and_do_not_fabricate_frames(self):
        record = manual_record(self.model, self.seed)
        record["samples"][1]["hands"]["RIGHT_HAND"] = {"active": False, "load": 0., "capacity": 100., "margin": 100.}
        record["samples"][2]["hands"]["RIGHT_HAND"] = {"active": True, "load": 20., "capacity": 100., "margin": 80.}
        result = self.meter(record)["transfers"][0]
        event = result["events"][2]
        self.assertEqual(event["event_before"]["row_id"], "move_0.sample[1]")
        self.assertEqual(event["event_before"]["time_s"], event["time_s"])
        self.assertFalse(event["event_before"]["moving_contact"]["active"])
        self.assertTrue(event["event_after"]["moving_contact"]["active"])
        self.assertEqual(event["event_after"]["row_id"], "move_0.sample[2]")
        self.assertAlmostEqual(event["native_step"]["end_m"], np.linalg.norm([.01, -.005, .015]))
        self.assertAlmostEqual(event["native_step"]["delta_time_s"], .002)
        self.assertAlmostEqual(event["delta_from_previous_event_s"], .002)
        self.assertAlmostEqual(result["phases"][0]["duration_s"], .004)
        self.assertAlmostEqual(result["phases"][1]["duration_s"], .004)
        record["events"][2]["time_s"] += .0005
        event = self.meter(record)["transfers"][0]["events"][2]
        self.assertLess(event["event_before"]["time_s"], event["time_s"])
        self.assertGreater(event["event_after"]["time_s"], event["time_s"])

    def sequence(self):
        first = manual_record(self.model, self.seed, curve="limb")
        second = manual_record(self.model, self.seed, "LEFT_HAND", "limb", first["final_state"]["time"])
        second["initial_state"] = copy.deepcopy(first["final_state"])
        return {"kind": "sequence", "dt_s": .002, "moves": [first, second],
                "initial_state": copy.deepcopy(first["initial_state"]),
                "final_state": copy.deepcopy(second["final_state"])}

    def test_sequence_exact_full_structure_continuity_and_separate_effectors(self):
        record = self.sequence()
        result = self.meter(record)
        self.assertEqual(set(result["whole_case"]["effectors"]), {"RIGHT_HAND", "LEFT_HAND"})
        self.assertNotIn("moving_effector", result["whole_case"])
        boundary = result["sequence_boundaries"][0]
        self.assertTrue(boundary["full_state_equal"])
        self.assertIsNone(boundary["integration_state_equal"])
        self.assertEqual(boundary["differing_fields"], [])
        self.assertEqual(boundary["max_qpos_delta"], 0.)
        self.assertEqual(boundary["delta_time_s"], 0.)
        self.assertTrue(result["case_initial_matches_first_move"])
        self.assertTrue(result["case_final_matches_last_move"])
        self.assertEqual(result["whole_case"]["timing"]["step_counter_reset_count"], 0)
        rows = trajectory_rows(record)
        boundary_time = record["moves"][0]["final_state"]["time"]
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["time_s"] for row in rows], sorted(row["time_s"] for row in rows))
        self.assertEqual([row["row_kind"] for row in rows if row["time_s"] == boundary_time],
                         ["sample", "final_state", "initial_state"])
        for field, value in (("qvel", [1.] * self.model.nv), ("contact_configuration", {"LEFT_HAND": "changed"}),
                             ("release_events", [{"time_s": 1.}]), ("time", boundary_time + 1e-15)):
            changed = copy.deepcopy(record)
            changed["moves"][1]["initial_state"][field] = value
            boundary = self.meter(changed)["sequence_boundaries"][0]
            self.assertFalse(boundary["full_state_equal"])
            self.assertIn(field, boundary["differing_fields"])
        record["moves"][0]["final_state"]["optional_observation"] = None
        boundary = self.meter(record)["sequence_boundaries"][0]
        self.assertFalse(boundary["full_state_equal"])
        self.assertEqual(boundary["differing_fields"], ["optional_observation"])
        record["moves"][0]["final_state"]["integration_state"] = [1., 2.]
        record["moves"][1]["initial_state"]["integration_state"] = [1., 2.]
        self.assertTrue(self.meter(record)["sequence_boundaries"][0]["integration_state_equal"])
        record["moves"][1]["initial_state"]["integration_state"][0] = 3.
        self.assertFalse(self.meter(record)["sequence_boundaries"][0]["integration_state_equal"])

    def test_clock_sorting_does_not_hide_native_gaps_resets_or_duplicate_pose_changes(self):
        record = manual_record(self.model, self.seed)
        record["samples"][1]["time_s"] = 2.001
        record["samples"][2]["time_s"] = record["samples"][0]["time_s"]
        record["samples"][2]["steps"] = 0
        timing = self.meter(record)["transfers"][0]["timing"]
        self.assertEqual(timing["time_reset_count"], 1)
        self.assertEqual(timing["sample_gap_count"], 1)
        self.assertEqual(timing["step_counter_reset_count"], 1)
        self.assertFalse(timing["nondecreasing_recorded_times"])
        self.assertFalse(timing["strictly_increasing_sample_times"])
        rows = trajectory_rows(record)
        self.assertEqual([r["time_s"] for r in rows], sorted(r["time_s"] for r in rows))
        self.assertNotEqual([r["record_index"] for r in rows], list(range(len(rows))))
        record["samples"][1]["time_s"] = record["samples"][0]["time_s"]
        timing = self.meter(record)["transfers"][0]["timing"]
        self.assertEqual(timing["changed_pose_at_duplicate_time_count"], 2)

    def test_actual_support_extremes_not_estimated_loads(self):
        record = manual_record(self.model, self.seed)
        record["samples"][1]["hands"]["LEFT_HAND"]["load"] = 42.
        record["samples"][2]["feet"]["RIGHT_FOOT"]["normal_force"] = 81.
        record["support_admission"] = {"estimated_load": 9999.}
        loads = self.meter(record)["transfers"][0]["support_loads"]
        self.assertEqual(loads["hands"]["LEFT_HAND"]["load"]["max_N"], 42.)
        self.assertEqual(loads["hands"]["LEFT_HAND"]["load"]["max_row_id"], "move_0.sample[1]")
        self.assertEqual(loads["feet"]["RIGHT_FOOT"]["normal_force"]["min_N"], 81.)
        self.assertIsNone(loads["feet"]["LEFT_FOOT"]["normal_force"])
        record["samples"][0]["hands"]["LEFT_HAND"] = {"active": False, "load": 0., "capacity": None, "margin": None}
        loads = self.meter(record)["transfers"][0]["support_loads"]["hands"]["LEFT_HAND"]
        self.assertEqual(loads["capacity_unavailable_count"], 1)
        self.assertEqual(loads["margin_unavailable_count"], 1)
        self.assertEqual(loads["capacity"]["min_N"], 100.)
        record["samples"][0]["hands"]["LEFT_HAND"]["active"] = True
        with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
            self.meter(record)

    def test_zero_step_record_has_actual_endpoints_and_unavailable_event_bracket(self):
        record = manual_record(self.model, self.seed)
        record["samples"] = []
        record["final_state"] = copy.deepcopy(record["initial_state"])
        record["events"] = [{"event": "PREFLIGHT_REJECTION", "time_s": 2.}]
        transfer = self.meter(record)["transfers"][0]
        self.assertEqual(transfer["timing"]["native_sample_count"], 0)
        self.assertEqual(transfer["timing"]["positive_dt_count"], 0)
        self.assertIsNone(transfer["timing"]["min_positive_dt_s"])
        self.assertEqual(transfer["moving_effector"]["path_length_m"], 0.)
        self.assertIsNone(transfer["events"][0]["native_step"])
        self.assertEqual(transfer["events"][0]["event_before"]["row_id"], "move_0.final_state")

    def test_pre_release_load_and_plateau_meter_only_saved_actual_rows_and_support_loads(self):
        record = manual_record(self.model, self.seed, curve="limb")
        record["release_time_s"] = 2.004
        record["samples"][0]["phase"] = "LOAD_TRANSFER"
        record["samples"][1]["phase"] = "LOAD"
        for row in record["samples"][2:]:
            row["phase"] = "SETTLE"
        record["samples"][2]["feet"]["RIGHT_FOOT"]["normal_force"] = 57.
        before = copy.deepcopy(record)
        transfer = self.meter(record)["transfers"][0]
        self.assertEqual(transfer["pre_release"]["timing"]["native_sample_count"], 2)
        self.assertEqual(transfer["native_load"]["timing"]["native_sample_count"], 1)
        self.assertEqual(transfer["loaded_plateau"]["timing"]["native_sample_count"], 2)
        self.assertAlmostEqual(transfer["pre_release"]["joints"]["right_elbow"]["peak_to_peak_rad"], .2)
        self.assertAlmostEqual(transfer["loaded_plateau"]["joints"]["right_elbow"]["peak_to_peak_rad"], .1)
        self.assertEqual(transfer["loaded_plateau"]["support_loads"]["feet"]["RIGHT_FOOT"]["normal_force"]["min_N"], 57.)
        self.assertEqual(record, before)

    def test_native_step_clock_and_applied_force_accounting_is_observation_not_quality_verdict(self):
        record = manual_record(self.model, self.seed)
        record.update(steps=4, duration_s=.008)
        for row in record["samples"]:
            row.update(qfrc_applied=[0.] * self.model.nv, external_force_world_N=[[0.] * 6] * self.model.nbody)
        result = self.meter(record)
        accounting = result["native_accounting"][0]
        self.assertTrue(accounting["step_clock_matches"])
        self.assertTrue(accounting["duration_matches_steps"])
        self.assertTrue(accounting["zero_recorded_applied_forces"])
        self.assertTrue(accounting["samples_have_both_force_channels"])
        self.assertIsNone(result["quality_verdict"])
        record["samples"][0]["external_force_world_N"][0][0] = 1.
        record["steps"] = 3
        accounting = self.meter(record)["native_accounting"][0]
        self.assertFalse(accounting["zero_recorded_applied_forces"])
        self.assertFalse(accounting["step_clock_matches"])
        self.assertFalse(accounting["duration_matches_steps"])

    def test_null_nonfinite_nonunit_and_invalid_actual_states_fail_explicitly(self):
        for field, value in (("qpos", None), ("qpos", [0.] * (self.model.nq - 1)),
                             ("qvel", None), ("ctrl", [np.nan] * self.model.nu),
                             ("qacc_warmstart", [np.inf] * self.model.nv), ("time_s", None),
                             ("pose_available", False), ("q_ref", [np.nan] * self.model.nq),
                             ("q_ref", None), ("qd_ref", None)):
            with self.subTest(field=field, value=value):
                record = manual_record(self.model, self.seed)
                record["samples"][0][field] = value
                with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
                    self.meter(record)
                if field == "time_s" or (field == "qpos" and value is None):
                    with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
                        trajectory_rows(record)
        for quaternion in ([0., 0., 0., 0.], [2., 0., 0., 0.], [None, 0., 0., 0.]):
            record = manual_record(self.model, self.seed)
            record["samples"][0]["qpos"][3:7] = quaternion
            with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
                self.meter(record)
        record = manual_record(self.model, self.seed)
        record["samples"][0]["hands"]["LEFT_HAND"]["load"] = None
        with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
            self.meter(record)
        record = manual_record(self.model, self.seed)
        record["samples"][1]["qpos"][0] = 1e308
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
            self.meter(record)

    def test_detached_rows_model_and_caller_data_are_unchanged_no_data_argument(self):
        record = self.sequence()
        before = copy.deepcopy(record)
        model_arrays = {name: value.copy() for name in dir(self.model)
                        if isinstance(value := getattr(self.model, name), np.ndarray)}
        self.live_data.time = 17.
        self.live_data.qpos[:] = self.seed
        self.live_data.ctrl[:] = .123
        self.live_data.xfrc_applied[:] = .456
        # Never inspect unallocated solver-arena views on an unstepped MjData.
        data_arrays = {name: getattr(self.live_data, name).copy() for name in
                       ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "eq_active", "qfrc_applied",
                        "xfrc_applied", "mocap_pos", "mocap_quat", "userdata", "xpos", "xquat", "xipos",
                        "site_xpos", "geom_xpos", "subtree_com")}
        timestep, time = self.model.opt.timestep, self.live_data.time
        self.meter(record)
        self.assertEqual(record, before)
        rows = trajectory_rows(record)
        rows[0]["qpos"][0] = 999.
        rows[0]["hands"]["LEFT_HAND"]["load"] = 999.
        self.assertEqual(record, before)
        for name, value in model_arrays.items():
            np.testing.assert_array_equal(getattr(self.model, name), value, err_msg=name)
        for name, value in data_arrays.items():
            np.testing.assert_array_equal(getattr(self.live_data, name), value, err_msg=name)
        self.assertEqual(self.model.opt.timestep, timestep)
        self.assertEqual(self.live_data.time, time)
        with self.assertRaises(TypeError):
            audit_motion(self.model, record, self.live_data)


if __name__ == "__main__":
    unittest.main()
