"""Read-only ascent rendering contracts; mocked exports certify no physics/video."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image

from boulder_v1 import ascending_sequence, contact_ik, runtime, transfers
from scripts import render_clean_transfers as clean
from scripts import validate_ascent as study
from test_validate_ascent import synthetic_fixture, synthetic_record


class RenderAscentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {name: synthetic_fixture(name, dt=.1) for name in ("baseline", "longer")}

    def setUp(self):
        stack = self.enterContext(ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData",
                     "mj_integratePos", "mj_normalizeQuat", "mj_fwdPosition", "mj_fwdVelocity",
                     "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
            stack.enter_context(mock.patch.object(mujoco, name, side_effect=AssertionError("Forbidden " + name)))
        for module, name in ((ascending_sequence, "make_ascending_fixture"), (ascending_sequence, "run_ascending_sequence"),
                (ascending_sequence, "execute_transfer"), (ascending_sequence, "initialize_static_reference"),
                (ascending_sequence, "execute_static_hold"), (transfers, "make_transfer_fixture"),
                (transfers, "execute_transfer"), (runtime, "compute_pose_control"), (contact_ik, "solve_contact_pose")):
            stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError("Forbidden " + name)))

    def packet(self, name="baseline"):
        fixture = self.fixtures[name]
        result = synthetic_record(fixture)
        return {**study.wb._evidence_value(result), "fixture_inputs": copy.deepcopy(fixture[4]),
                "unit_evidence": "Synthetic parser schema only, NOT a native physical episode",
                "validation": {"accepted": True, "physical_success": True},
                "reference_identity": study._reference_identity(result),
                "provenance": {"modules": {"boulder_v1." + path.stem: {"file": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (clean.ROOT / "src" / "boulder_v1").glob("*.py")}}}

    def test_case_maps_and_fixture_defaults_do_not_append_to_stage5(self):
        self.assertEqual(clean.CASES, {"right_hand": "right_hand", "left_hand": "left_hand", "left_foot": "foot", "sequence": "sequence"})
        self.assertEqual(clean.ASCENDING_CASES, {"ascending_baseline": "baseline", "ascending_longer": "longer"})
        for fixture, expected in (("stage5", list(clean.CASES)), ("whole_body", list(clean.CASES)),
                                  ("ascending", list(clean.ASCENDING_CASES))):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "outputs").mkdir()
                with mock.patch.object(clean, "ROOT", root), mock.patch.object(clean, "render_case",
                        side_effect=lambda case, *a, **kw: {"case": case, "artifacts": {}}) as render, redirect_stdout(io.StringIO()):
                    self.assertEqual(clean.main(["--fixture", fixture, "--output", str(root / "outputs" / "unit")]), 0)
                self.assertEqual([c.args[0] for c in render.call_args_list], expected)

    def test_fixture_mixed_cases_and_shorter_never_generate_a_success_video(self):
        for argv in (["--case", "ascending_baseline"], ["--fixture", "ascending", "--case", "sequence"],
                     ["--fixture", "ascending", "--case", "ascending_shorter"], ["--timing", "fast"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), \
                    mock.patch.object(clean, "render_case") as render:
                clean.main(argv)
            render.assert_not_called()

    def test_input_name_uses_report_demo_timing_and_dt_not_guessed_case_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(json.dumps({"demo_timing": "moderate"}))
            for case, profile in clean.ASCENDING_CASES.items():
                for dt in (.002, .001):
                    path, timing = clean._case_input(case, root, dt, "ascending")
                    self.assertEqual(path.name, f"{profile}_moderate_{dt * 1000:g}ms.json")
                    self.assertEqual(timing, "moderate")
            self.assertEqual(clean._case_input("ascending_baseline", root, .001, "ascending", "fast")[0].name,
                             "baseline_fast_1ms.json")
            (root / "report.json").write_text(json.dumps({"demo_timing": None}))
            with self.assertRaisesRegex(ValueError, "No certified demo_timing"):
                clean._case_input("ascending_baseline", root, .002, "ascending")

    def test_loader_requires_all_current_native_sources_and_certified_three_move_packet(self):
        packet = self.packet()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            clean._write_json(path, packet)
            original = path.read_bytes()
            self.assertEqual(clean._load_verified_evidence(path, "ascending"), packet)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "fixture"):
                clean._load_verified_evidence(path)
            for mode in ("missing_native", "stale_native", "owner_identity", "negative", "uncertified", "zero_moves"):
                changed = copy.deepcopy(packet)
                if mode == "missing_native":
                    del changed["provenance"]["modules"]["boulder_v1.ascending_sequence"]
                elif mode == "stale_native":
                    changed["provenance"]["modules"]["boulder_v1.reference_candidates"]["sha256"] = "0" * 64
                elif mode == "owner_identity":
                    changed["reference_identity"][0]["same_object"] = False
                elif mode == "negative":
                    changed["negative"] = "candidate_exhaustion"
                elif mode == "uncertified":
                    changed["validation"]["physical_success"] = False
                else:
                    changed["moves"] = []
                clean._write_json(path, changed)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    clean._load_verified_evidence(path, "ascending")

    def test_reconstruction_compiles_saved_scene_profile_seed_and_geometry_without_factory_or_reset(self):
        for name in self.fixtures:
            packet = self.packet(name)
            before = copy.deepcopy(packet)
            model, data, scene, profile, seed = clean._matching_fixture(packet, .1, "ascending")
            expected = self.fixtures[name]
            self.assertIsNone(data)
            self.assertEqual(study.wb._model_hash(model), study.wb._model_hash(expected[0]))
            self.assertEqual(scene.to_dict(), expected[1].to_dict())
            self.assertEqual(profile, expected[2])
            np.testing.assert_array_equal(seed, expected[3])
            self.assertEqual(packet, before)
        ascending_sequence.make_ascending_fixture.assert_not_called()

    def test_reconstruction_refuses_input_xml_compiled_model_geometry_and_seed_tampering(self):
        for mode in ("input_hash", "xml", "compiled", "geometry", "seed", "profile", "dt"):
            packet = self.packet()
            inputs = packet["fixture_inputs"]
            if mode == "input_hash":
                inputs["scene"]["scale"] = 2.
            elif mode == "xml":
                inputs["model_xml_sha256"] = "0" * 64
            elif mode == "compiled":
                inputs["compiled_model_sha256"] = "0" * 64
            elif mode == "geometry":
                inputs["contact_geometry"]["foot_target"]["foot_frame"]["position"][2] += .01
            elif mode == "seed":
                inputs["seed_qpos"].pop()
            elif mode == "profile":
                inputs["profile"]["strength_scale"] = 2.
                packet["profile"] = inputs["profile"]
            else:
                packet["dt_s"] = .002
            if mode not in ("input_hash", "dt"):
                inputs["input_sha256"] = study.wb._hash({k: v for k, v in inputs.items() if k != "input_sha256"})
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                clean._matching_fixture(packet, .1, "ascending")

    def test_native_fk_literal_qpos_and_all_time_bounds_cover_nine_cm_ascent(self):
        model, _, _, seed, _ = self.fixtures["baseline"]
        data = mujoco.MjData(model)
        data.ctrl[:] = .123
        data.qfrc_applied[:] = .456
        data.xfrc_applied[:] = .789
        before = {key: getattr(data, key).copy() for key in ("ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart", "eq_active")}
        rows = [{"qpos": (seed + np.r_[[0., 0., rise], np.zeros(model.nq - 3)]).tolist(),
                 "qvel": [0.] * model.nv, "q_ref": [999.] * model.nq, "time_s": 2. + i * .1}
                for i, rise in enumerate((0., .09, .085))]
        snapshots = copy.deepcopy(rows)
        with mock.patch.object(mujoco, "mj_kinematics", wraps=mujoco.mj_kinematics) as fk, \
                mock.patch.object(mujoco, "mj_comPos", wraps=mujoco.mj_comPos) as com, \
                mock.patch.object(mujoco, "mj_camlight", wraps=mujoco.mj_camlight) as light:
            clean._kinematics(mujoco, model, data, rows[1])
        for call in (fk, com, light):
            call.assert_called_once_with(model, data)
        np.testing.assert_array_equal(data.qpos, rows[1]["qpos"])
        self.assertEqual(data.time, rows[1]["time_s"])
        for key in before:
            np.testing.assert_array_equal(getattr(data, key), before[key])
        clean._kinematics(mujoco, model, data, rows[0])
        lo, hi = clean._actor_bounds(mujoco, model, data)
        cameras, bounds = clean._camera_definitions(model, rows, mujoco, "ascending_baseline")
        np.testing.assert_allclose(bounds["minimum_world_m"], lo)
        np.testing.assert_allclose(bounds["maximum_world_m"], hi + [0., 0., .09])
        self.assertTrue(all(c["fixed_over_entire_recording"] for c in cameras.values()))
        self.assertEqual(cameras["side"]["azimuth_deg"], 0.)
        self.assertEqual(rows, snapshots)

    def test_generic_foot_keyframes_read_saved_foot_request_duration(self):
        packet = {"moves": [{"primitive": "foot", "request": {"hand_reach_s": 99., "foot_request": {"reach_s": 6.}},
                             "events": [{"phase": "REACH", "time_s": 10.}]}]}
        rows = [{"move_index": 0, "time_s": t} for t in (10., 11.5, 13., 16.)]
        self.assertEqual(clean._event_keyframes(packet, rows)["move1_reach_mid"], 2)

    def test_scene_only_returns_unaltered_viewport_no_sidebar_or_telemetry(self):
        model = copy.copy(self.fixtures["baseline"][0])
        data = mujoco.MjData(model)
        row = {"qpos": self.fixtures["baseline"][3].tolist(), "time_s": 2.}
        cameras, _ = clean._camera_definitions(model, [row], mujoco, "ascending_baseline")
        renderer = mock.Mock()
        pixels = np.full((720, 1280, 3), 61, dtype=np.uint8)
        renderer.render.return_value = pixels
        with mock.patch.object(clean, "_compose", side_effect=AssertionError("No overlay")):
            image = clean._render_view(mujoco, renderer, model, data, row, cameras["rear"], [], "rear",
                                       "ascending_baseline", 1280, 720, scene_only=True)
        np.testing.assert_array_equal(np.asarray(image), pixels)
        self.assertEqual(image.size, (1280, 720))

    def test_export_three_views_shares_indices_and_only_reads_one_case_at_a_time(self):
        for name in self.fixtures:
            packet = self.packet(name)
            case = "ascending_" + name
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                output.mkdir()
                path = root / f"{name}_fast_100ms.json"
                clean._write_json(path, packet)
                original = path.read_bytes()
                renderer = mock.Mock()
                writers = []
                def writer(*a, **kw):
                    value = mock.MagicMock()
                    value.__enter__.return_value = value
                    writers.append(value)
                    return value
                with mock.patch.object(mujoco, "Renderer", return_value=renderer), \
                        mock.patch.object(iio, "get_writer", side_effect=writer), \
                        mock.patch.object(clean, "_render_view", return_value=Image.new("RGB", (1280, 720))) as render, \
                        mock.patch.object(Image.Image, "save"), \
                        mock.patch.object(clean, "_decoded_contact_sheet", return_value={}), \
                        mock.patch.object(clean, "_video_timing", return_value={"verified": True}) as proof:
                    report = clean.render_case(case, root, output, timestep=.1, fixture="ascending", timing="fast",
                                               width=1280, height=720)
                self.assertEqual(len(writers), 3)
                self.assertEqual(proof.call_count, 3)
                for view in clean.VIEW_NAMES:
                    calls = [c for c in render.call_args_list if c.args[7] == view]
                    count = len(report["rendered_native_indices"])
                    self.assertEqual([c.args[4]["record_index"] for c in calls[:count]], report["rendered_native_indices"])
                    self.assertTrue(report["artifacts"][view]["playback_timing"]["verified"])
                self.assertEqual(report["physics_steps_executed"], 0)
                self.assertFalse(report["dynamics_replayed"])
                self.assertEqual(report["motion_demonstration_verdict"], "PENDING VISUAL INSPECTION")
                self.assertEqual(path.read_bytes(), original)
                renderer.close.assert_called_once()

    def test_audit_only_and_forged_success_never_open_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "baseline_fast_100ms.json"
            packet = self.packet()
            clean._write_json(path, packet)
            with mock.patch.object(mujoco, "Renderer") as renderer, mock.patch.object(iio, "get_writer") as writer:
                report = clean.render_case("ascending_baseline", root, root, timestep=.1,
                                          fixture="ascending", timing="fast", audit_only=True)
                self.assertEqual(report["physics_steps_executed"], 0)
                self.assertIsNone(report["quality_verdict"])
                packet["moves"][1]["acquisition"]["sustained_s"] = .001
                clean._write_json(path, packet)
                with self.assertRaisesRegex(ValueError, "failed current native-contact"):
                    clean.render_case("ascending_baseline", root, root, timestep=.1, fixture="ascending", timing="fast")
            renderer.assert_not_called()
            writer.assert_not_called()

    def test_ffprobe_verifies_native_duration_frame_count_and_both_rates(self):
        stream = {"avg_frame_rate": "20/1", "r_frame_rate": "20/1", "nb_read_frames": "202", "duration": "10.1"}
        with mock.patch.object(clean.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps({"streams": [stream]}))) as probe:
            result = clean._video_timing(Path("unit-not-a-real-video.mp4"), 20., 202, 10.025)
        self.assertTrue(result["verified"])
        self.assertFalse(result["video_speedup"])
        self.assertEqual(probe.call_args.args[0][0], "ffprobe")
        for key, value in (("avg_frame_rate", "40/1"), ("r_frame_rate", "40/1"), ("nb_read_frames", "101"), ("duration", "5.05")):
            changed = {**stream, key: value}
            with self.subTest(key=key), mock.patch.object(clean.subprocess, "run",
                    return_value=SimpleNamespace(stdout=json.dumps({"streams": [changed]}))), self.assertRaises(ValueError):
                clean._video_timing(Path("unit.mp4"), 20., 202, 10.025)


if __name__ == "__main__":
    unittest.main()
