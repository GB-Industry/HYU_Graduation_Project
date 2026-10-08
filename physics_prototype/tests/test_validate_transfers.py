"""Short CLI/serializer/HUD contracts; no native acceptance-suite reruns."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from boulder_v1.schema import Limb
from boulder_v1.static_state import StaticReference
from scripts import validate_transfers as vt


def row():
    return {
        "status": "SUCCESS", "phase": "SETTLE", "time_s": 9.106, "elapsed_s": 7.106,
        "steps": 3553, "terminal": True, "pose_available": True,
        "qpos": [0., .1, .2], "qvel": [0., 0., 0.], "q_ref": [0., .1, .2], "qd_ref": [0., 0., 0.],
        "root_linear_m_s": .001, "root_angular_rad_s": .002, "joint_max_rad_s": .003,
        "tracking_error_m": .000082,
        "capture_measurement": {"gap_m": .000082, "orientation": 1., "relative_speed_m_s": .00001},
        "hands": {limb.value: {"active": True, "valid": True, "region_id": limb.value.lower(),
                              "load": 23.5, "capacity": 850.} for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND)},
        "feet": {limb.value: {"normal_force": 210., "tangential_force": 3., "tangential_speed": .0001,
                             "contacting": True, "supporting": True, "slipping": False}
                 for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)},
        "contacts": {limb.value: limb.value.lower() for limb in Limb},
        "readiness": {"ready": True, "duration": .5, "rms_hinge_speed": .001, "reason": "Sustained readiness"},
        "command": {"commanded_Nm": [10., -12.], "utilization": [.1, .12], "saturated": [False, False]},
        "reason": "Native transfer and sustained readiness",
    }


def physical_result():
    reference = StaticReference((0., .1, .2), {"waist_yaw": .1}, {Limb.LEFT_HAND: "left_reach_target"}, ())
    state = {"qpos": [0., .1, .2], "qvel": [0., 0., 0.], "ctrl": [0., 0.], "time": 9.106, "finite": True}
    sample = row()
    return {"success": True, "status": "SUCCESS", "reason": sample["reason"], "steps": 3553,
            "duration_s": 7.106, "final_reference": reference, "initial_state": state, "final_state": state,
            "readiness": sample["readiness"], "capture": {"gap_m": .000082, "capture_margin_m": .000918},
            "terminal_observation": sample, "samples": [sample]}


class ValidateTransfersTests(unittest.TestCase):
    def test_recursive_serializer_keeps_immutable_final_references_and_contact_maps(self):
        result = physical_result()
        contacts = MappingProxyType({Limb.LEFT_HAND: "left_reach_target", Limb.LEFT_FOOT: "foot_target"})
        wrapped = {"moves": [{**result, "support_contacts": contacts, "underlying_result": result}],
                   "final_reference": result["final_reference"], "array": np.array([1., np.nan, np.inf]),
                   "scalar": np.float64(-np.inf), "none": None, "path": Path("evidence.json")}
        encoded = json.loads(json.dumps(vt._json_value(wrapped), allow_nan=False))
        self.assertEqual(encoded["array"], [1., None, None])
        self.assertIsNone(encoded["scalar"])
        self.assertIsNone(encoded["none"])
        self.assertEqual(encoded["path"], "evidence.json")
        self.assertEqual(encoded["final_reference"]["target_pose"], {"waist_yaw": .1})
        self.assertEqual(encoded["moves"][0]["support_contacts"]["LEFT_FOOT"], "foot_target")
        self.assertEqual(encoded["moves"][0]["underlying_result"]["final_reference"], encoded["final_reference"])
        with self.assertRaises(TypeError):
            result["final_reference"].contact_intent[Limb.LEFT_HAND] = "changed"
        with self.assertRaises(TypeError):
            vt._json_value(object())

    def test_job_matrix_is_four_primitives_both_dts_and_cost_aware_diagnostics(self):
        args = argparse.Namespace(suite="all", dt=None, case=None)
        jobs = vt._jobs(args)
        self.assertEqual([job for job in jobs if job[0] == "family"],
                         [("family", case, dt) for case in vt.FAMILY for dt in (.002, .001)])
        self.assertEqual(len([job for job in jobs if job[0] == "negative"]), len(vt.NEGATIVES))
        self.assertTrue(all(dt == .002 for suite, _, dt in jobs if suite == "negative"))
        self.assertEqual(len([job for job in jobs if job[1] in vt.PROFILE_CASES]), 2)
        self.assertEqual(len([job for job in jobs if job[1] in vt.PERTURBATIONS]), 4)
        args.dt, args.case = .001, ["left_hand"]
        self.assertEqual(vt._jobs(args), [("family", "left_hand", .001)])

    def test_dispatch_preserves_generic_api_and_stage4_default_policy(self):
        from boulder_v1 import single_hand, transfers

        observer = Mock()
        with patch.object(transfers, "run_transfer_benchmark", return_value={}) as run:
            for kind in vt.FAMILY:
                vt._execute_case(kind, .001, observer)
                run.assert_called_with(timestep=.001, kind=kind, negative=None, observer=observer, keep_samples=True)
            vt._execute_case("sequence_second_unreachable", .002, None)
            run.assert_called_with(timestep=.002, kind="sequence", negative="second_unreachable",
                                   observer=None, keep_samples=True)
        with patch.object(single_hand, "run_single_hand_benchmark", return_value={}) as run:
            for scenario in vt.STAGE4_NEGATIVES:
                vt._execute_case(scenario, .002, None)
                run.assert_called_with(timestep=.002, scenario=scenario, observer=None, keep_samples=True)
        with self.assertRaisesRegex(ValueError, "Unknown transfer case"):
            vt._execute_case("made_up_scene", .002, None)

    def test_profile_dispatch_uses_existing_objects_without_retuning(self):
        from boulder_v1 import hand_family
        from scripts.view_scene import PROFILES

        with patch.object(hand_family, "run_hand_family_benchmark", side_effect=lambda **_: {}) as run:
            vt._execute_case("compact_strong", .002, None)
            self.assertIs(run.call_args.kwargs["profile"], PROFILES["compact_strong"])
            self.assertEqual(run.call_args.kwargs["pose_perturbation_rad"], 0.)
            result = vt._execute_case("perturbed_left_hand", .001, None)
            self.assertEqual(run.call_args.kwargs["limb"], Limb.LEFT_HAND)
            self.assertEqual(run.call_args.kwargs["pose_perturbation_rad"], -1e-4)
            self.assertFalse(result["diagnostic"]["targets_retuned"])

    def test_side_agnostic_context_respects_sequence_index_and_explicit_limb(self):
        self.assertEqual(vt._move_context("left_hand", {}), (Limb.LEFT_HAND, "left_hand", "left_reach_target"))
        self.assertEqual(vt._move_context("sequence", {"move_index": 1, "move_count": 2}),
                         (Limb.LEFT_HAND, "left_hand", "left_reach_target"))
        self.assertEqual(vt._move_context("foot", {"moving_limb": "RIGHT_FOOT", "target": "other_step"}),
                         (Limb.RIGHT_FOOT, "left_foot", "other_step"))
        with self.assertRaises(KeyError):
            vt._move_context("sequence", {})
        with self.assertRaises(ValueError):
            vt._move_context("sequence", {"move_index": 2, "move_count": 3})

    def test_goal_measurement_uses_selected_hand_or_step_site(self):
        for limb, goal in ((Limb.LEFT_HAND, "site_left_reach_target"), (Limb.RIGHT_FOOT, "site_step_foot_target")):
            with self.subTest(limb=limb):
                names = []
                def site(name):
                    names.append(name)
                    return SimpleNamespace(id=len(names), xpos=np.array([float(len(names)), 0., 0.]),
                                           xmat=np.eye(3).reshape(-1))
                data = SimpleNamespace(site=site, qvel=np.zeros(3))
                measured = vt._site_measurement(Mock(), SimpleNamespace(nv=3), data, limb,
                                                "left_reach_target" if limb.is_hand else "foot_target")
                self.assertEqual(names, [limb.value.lower() + "_site", goal])
                self.assertEqual(measured, {"gap_m": 1., "orientation": 1., "relative_speed_m_s": 0.})

    def test_hud_bounds_fixed_font_colors_and_all_mandatory_fields(self):
        for limb, source, target in ((Limb.RIGHT_HAND, "right_hand", "reach_target"),
                                     (Limb.LEFT_HAND, "left_hand", "left_reach_target"),
                                     (Limb.LEFT_FOOT, "left_foot", "foot_target"),
                                     (Limb.RIGHT_FOOT, "right_foot", "foot_target")):
            with self.subTest(limb=limb):
                image, sample = Image.new("RGB", (640, 480)), row()
                sample.update(move_index=1, move_count=2, reason="Long exact terminal reason " * 40)
                sample["readiness"]["reason"] = "Long readiness explanation " * 40
                sample.update(released=True, foot_distance_m=-.0001,
                              source_contact={"contact_count": 0, "normal_force_N": 0.},
                              touchdown={"normal_force_N": .21},
                              acquisition={"time_s": 10.2, "sustained_s": .1, "normal_force_N": 25.})
                drawn, rectangles = [], []
                original_draw = ImageDraw.Draw
                real = original_draw(image, "RGBA")
                def text(xy, value, **kwargs):
                    drawn.append((xy, value, kwargs))
                    bbox = real.textbbox(xy, value, font=kwargs["font"])
                    self.assertGreaterEqual(bbox[0], 0)
                    self.assertGreaterEqual(bbox[1], 0)
                    self.assertLessEqual(bbox[2], image.width)
                    self.assertLessEqual(bbox[3], image.height)
                    real.text(xy, value, **kwargs)
                def rectangle(bounds, **kwargs):
                    rectangles.append((bounds, kwargs))
                    self.assertGreaterEqual(bounds[0][1], 0)
                    self.assertLessEqual(bounds[1][1], image.height)
                    real.rectangle(bounds, **kwargs)
                proxy = Mock(wraps=real)
                proxy.text.side_effect, proxy.rectangle.side_effect = text, rectangle
                with patch.object(ImageDraw, "Draw", return_value=proxy), \
                        patch.object(ImageFont, "load_default", wraps=ImageFont.load_default) as font:
                    self.assertIs(vt._draw_hud(image, sample, .002, limb, source, target), image)
                    font.assert_called_once_with(size=10)
                contents = "\n".join(item[1] for item in drawn)
                for required in (limb.value, target, "move 2/2", "Contacts:", "hinge max/rms=", "q/ref hinge",
                                 "Ref error=", "Actual error=", "margin=", "LH:", "RH:", "load/cap=",
                                 "LF:", "RF:", "Fn=", "Ft=", "slip v=", "support=", "Motor |tau|max=",
                                 "util=", "Readiness=True sustained=0.500s", "full reason: JSON"):
                    self.assertIn(required, contents)
                if limb.is_foot:
                    self.assertIn("Native acquired=True", contents)
                    self.assertIn("source n=0", contents)
                    self.assertIn("touchdown Fn=0.21N", contents)
                self.assertTrue(all(item[2]["fill"] == (210, 230, 245) for item in drawn))
                self.assertTrue(all(item[1]["fill"] == (12, 18, 28, 220) for item in rectangles))
                self.assertTrue(all(item[1]["outline"] == (50, 90, 140, 240) for item in rectangles))
                self.assertEqual(sample["reason"], "Long exact terminal reason " * 40)

    def test_diagnostics_are_not_physical_passes_and_nonfinite_is_not_hidden(self):
        result = physical_result()
        self.assertTrue(vt._case_verdict("family", "left_hand", result))
        result.update(success=False, status="CONTROL_FAILURE", final_reference=None)
        result["readiness"]["ready"] = False
        self.assertFalse(vt._physical_success(result))
        self.assertTrue(vt._case_verdict("sensitivity", "compact_strong", result))
        self.assertFalse(vt._case_verdict("family", "left_hand", result))
        result["samples"][0]["qvel"][0] = np.nan
        self.assertFalse(vt._episode_finite(result))
        self.assertFalse(vt._case_verdict("sensitivity", "compact_strong", result))
        self.assertIsNone(vt._json_value(result)["samples"][0]["qvel"][0])

    def test_early_profile_admission_failure_is_a_finite_diagnostic_not_a_pass(self):
        result = {"success": False, "status": "INITIALIZATION_FAILURE", "reason": "Local seed admission failed",
                  "retarget": SimpleNamespace(qpos=(0., .1), reason="Exact diagnostic"), "steps": 0}
        self.assertTrue(vt._case_verdict("sensitivity", "long_reach_lower_grip", result))
        self.assertFalse(vt._physical_success(result))

    def test_preflight_negative_requires_valid_source_and_no_release_or_steps(self):
        result = physical_result()
        result.update(success=False, status="SUPPORT_INFEASIBLE", steps=0, released=False,
                      final_reference=None, initial_static={"success": True}, support_admission={"admitted": False})
        result["readiness"]["ready"] = False
        self.assertTrue(vt._case_verdict("negative", "foot_low_friction", result))
        result["initial_static"]["success"] = False
        self.assertFalse(vt._case_verdict("negative", "foot_low_friction", result))
        result["initial_static"]["success"], result["steps"] = True, 1
        self.assertFalse(vt._case_verdict("negative", "foot_low_friction", result))

    def test_existing_baseline_paths_and_hashes_are_not_claimed_as_reruns(self):
        with patch.object(Path, "is_file", return_value=False):
            missing = vt._baselines()
        self.assertEqual(set(missing), {"contacts", "controller", "transition"})
        for label, baseline in missing.items():
            self.assertEqual(baseline["report_json"], vt.ROOT / "outputs" / f"stage5-baseline-{label}" / "report.json")
            self.assertFalse(baseline["available"])
            self.assertFalse(baseline["rerun"])
            self.assertIsNone(baseline["report_sha256"])
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_bytes", return_value=b"{}"):
            present = vt._baselines()
        self.assertTrue(all(baseline["available"] and len(baseline["report_sha256"]) == 64
                            and not baseline["rerun"] for baseline in present.values()))

    def test_cli_protects_existing_and_future_preservation_directories(self):
        for destination in (vt.ROOT / "scripts", vt.ROOT / "outputs" / "stage5-baseline-transition",
                            vt.ROOT / "outputs" / "stage5-preserved-future"):
            with self.subTest(destination=destination), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(vt, "_execute_case") as execute:
                with self.assertRaises(SystemExit) as caught:
                    vt.main(["--suite", "family", "--case", "foot", "--output", str(destination)])
                self.assertEqual(caught.exception.code, 2)
                execute.assert_not_called()

    def test_sequence_negative_requires_one_success_then_actual_second_rejection(self):
        first = physical_result()
        second = {"success": False, "status": "REACH_INFEASIBLE", "steps": 0, "released": False,
                  "initial_state": first["final_state"], "final_state": first["final_state"], "final_reference": None}
        result = {"success": False, "status": "REACH_INFEASIBLE", "completed_moves": 1, "failed_index": 1,
                  "moves": [first, second], "final_state": second["final_state"]}
        self.assertTrue(vt._case_verdict("negative", "sequence_second_unreachable", result))
        result["completed_moves"] = 0
        self.assertFalse(vt._case_verdict("negative", "sequence_second_unreachable", result))

    def test_render_callback_copies_rows_and_closes_resources_on_exception(self):
        import imageio.v2 as iio
        import mujoco
        from scripts import render_demo

        sample = row()
        sample["pose_available"] = False
        renderer, writer = Mock(), Mock()
        writer.append_data.side_effect = RuntimeError("encoder failure")
        def execute(case, dt, observer):
            try:
                observer(sample, SimpleNamespace(), SimpleNamespace())
            finally:
                sample["reason"] = "changed after callback"
        with patch.object(vt, "_execute_case", side_effect=execute) as run, \
                patch.object(vt, "_write_json") as write, \
                patch.object(mujoco, "MjData"), patch.object(mujoco, "MjvCamera"), \
                patch.object(mujoco, "Renderer", return_value=renderer), \
                patch.object(iio, "get_writer", return_value=writer), \
                patch.object(render_demo, "_configure_camera"), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            summary = vt._run_case("family", "left_hand", .002, vt.ROOT / "outputs", True, {}, {})
        run.assert_called_once()
        renderer.close.assert_called_once()
        writer.close.assert_called_once()
        self.assertEqual(summary["status"], "EXECUTION_ERROR")
        self.assertFalse(summary["rendering"]["success"])
        self.assertIn("encoder failure", stderr.getvalue())
        observer_json = write.call_args_list[0].args[1]
        self.assertEqual(observer_json["observations"][0]["row"]["reason"], "Native transfer and sustained readiness")
        self.assertTrue(all(call.kwargs["compact"] for call in write.call_args_list))

    def test_cli_rejects_render_without_family_and_empty_case_selection(self):
        for arguments in (["--suite", "negative", "--render"],
                          ["--suite", "sensitivity", "--render"],
                          ["--suite", "negative", "--case", "foot"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(vt, "_execute_case") as execute:
                with self.assertRaises(SystemExit) as caught:
                    vt.main(arguments)
                self.assertEqual(caught.exception.code, 2)
                execute.assert_not_called()

    def test_requested_render_failure_returns_nonzero_without_changing_physical_verdict(self):
        result = physical_result()
        summary = {"name": "left_hand_2ms", "suite": "family", "case": "left_hand", "status": "SUCCESS",
                   "reason": "physical pass", "physical_success": True, "accepted": True, "completed_moves": 1,
                   "moves": [vt._move_summary(result)], "rendering": {"requested": True, "success": False,
                   "errors": ["export failed"]}}
        with patch.object(vt, "_run_case", return_value=summary), patch.object(vt, "_provenance", return_value={}), \
                patch.object(vt, "_baselines", return_value={}), patch.object(vt, "_write_json") as write, \
                patch.object(Path, "mkdir"), patch.dict(vt.os.environ), contextlib.redirect_stdout(io.StringIO()):
            code = vt.main(["--suite", "family", "--case", "left_hand", "--dt", ".002", "--render", "--summary"])
        self.assertEqual(code, 1)
        report = write.call_args.args[1]
        self.assertTrue(report["physical_acceptance"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["rendering_success"])

    def test_missing_family_evidence_does_not_report_partial_physical_acceptance(self):
        with patch.object(vt, "_run_case", side_effect=OSError("JSON export failed")), \
                patch.object(vt, "_provenance", return_value={}), patch.object(vt, "_baselines", return_value={}), \
                patch.object(vt, "_write_json") as write, patch.object(Path, "mkdir"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = vt.main(["--suite", "family", "--case", "foot", "--dt", ".002", "--summary"])
        self.assertEqual(code, 1)
        self.assertFalse(write.call_args.args[1]["physical_acceptance"])
        self.assertEqual(write.call_args.args[1]["physical_success_count"], 0)


if __name__ == "__main__":
    unittest.main()
