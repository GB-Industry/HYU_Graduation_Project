"""Fast saved-evidence/CLI contracts; synthetic records are not physics passes."""
import argparse
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock

import mujoco
import numpy as np

from boulder_v1 import whole_body_demo
from boulder_v1.contact_geometry import Frame
from boulder_v1.schema import Limb
from boulder_v1.transfers import TransferRequest, make_transfer_fixture
from boulder_v1.whole_body_motion import WholeBodyMotion
from scripts import render_clean_transfers as clean
from scripts import validate_whole_body as wb


class ValidateWholeBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately substitute a cheap existing geometry fixture, never simulate.
        cls.fixture = make_transfer_fixture(.002)
        cls.model, cls.data, cls.scene, cls.profile, cls.seed = cls.fixture
        with mock.patch.object(whole_body_demo, "make_whole_body_fixture", return_value=cls.fixture):
            cls.metadata_fixture = wb._fixture_metadata(.002)

    def setUp(self):
        stack = self.enterContext(contextlib.ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData"):
            stack.enter_context(mock.patch.object(mujoco, name, side_effect=AssertionError(f"Forbidden {name}")))
        for name in ("run_whole_body_benchmark", "make_whole_body_fixture", "execute_transfer",
                     "execute_transfer_sequence", "execute_static_hold"):
            stack.enter_context(mock.patch.object(whole_body_demo, name, side_effect=AssertionError(f"Forbidden {name}")))

    def record(self, *, negative=None):
        def state(time):
            return {"time": time, "qpos": self.seed.tolist(), "qvel": [0.] * self.model.nv,
                    "ctrl": [0.] * self.model.nu, "qacc_warmstart": [0.] * self.model.nv,
                    "eq_active": [False] * self.model.neq, "integration_state": [time, 7., 11.],
                    "finite": True, "contact_configuration": self.scene.to_dict()["start_configuration"],
                    "unit_only": "Synthetic parser record; no dynamics or success evidence"}

        initial = state(2.)
        final = state(2. if negative else 2.004)
        samples = [] if negative else [{"time_s": 2. + i * .002, "qpos": self.seed.tolist(),
            "qvel": [0.] * self.model.nv, "ctrl": [0.] * self.model.nu, "steps": i,
            "qfrc_applied": [0.] * self.model.nv, "external_force_world_N": [[0.] * 6] * self.model.nbody,
            "root_linear_m_s": 0., "root_angular_rad_s": 0., "joint_max_rad_s": 0., "phase": "SETTLE"}
            for i in (1, 2)]
        return {"fixture": wb.FIXTURE, "kind": "right_hand", "moving_limb": "RIGHT_HAND",
                "source": "right_hand", "target": "reach_target", "dt_s": .002,
                "profile": self.profile.to_dict(), "negative": negative, "initial_state": initial,
                "final_state": final, "initial_static": {"success": True, "final_state": copy.deepcopy(initial)},
                "success": negative is None, "status": wb.NEGATIVES[negative] if negative else "SUCCESS",
                "reason": "Synthetic unit outcome, not physics evidence", "steps": 0 if negative else 2,
                "duration_s": 0. if negative else .004, "released": negative is None, "events": [],
                "samples": samples, "final_reference": None if negative else {"qpos": self.seed.tolist()},
                "readiness": {"ready": negative is None}}

    def test_default_matrix_is_eight_positives_and_eight_atomic_negatives(self):
        args = argparse.Namespace(suite="all", dt=None, case=None)
        jobs = wb._jobs(args)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(jobs[:8], [("family", case, dt) for case in wb.FAMILY for dt in (.002, .001)])
        self.assertEqual(jobs[8:], [("negative", case, dt) for case in wb.NEGATIVES for dt in (.002, .001)])
        args.dt, args.case = .001, ["foot", "waist_rom"]
        self.assertEqual(wb._jobs(args), [("family", "foot", .001), ("negative", "waist_rom", .001)])

    def test_dispatch_retains_samples_and_forwards_observer_without_retuning(self):
        observer = mock.Mock()
        with mock.patch.object(whole_body_demo, "run_whole_body_benchmark", return_value={}) as run:
            for case in (*wb.FAMILY, *wb.NEGATIVES):
                wb._execute_case(case, .001, observer)
                run.assert_called_with(timestep=.001, kind=case if case in wb.FAMILY else "right_hand",
                                       negative=case if case in wb.NEGATIVES else None,
                                       keep_samples=True, observer=observer)
        with self.assertRaisesRegex(ValueError, "Unknown whole-body case"):
            wb._execute_case("invented", .002)

    def test_field_walk_serializes_immutable_motion_and_all_native_state_fields(self):
        motion = WholeBodyMotion(Frame((0., 0., 1.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                                 {"waist_yaw": 0., "waist_pitch": .07, "waist_roll": 0.})
        request = TransferRequest(Limb.RIGHT_HAND, "right_hand", "reach_target",
                                  {limb: limb.value.lower() for limb in Limb}, whole_body=motion)

        @dataclass(frozen=True)
        class NativeSnapshot:
            qpos: tuple = (1., 2.)
            qvel: tuple = (3., 4.)
            ctrl: tuple = (5.,)
            qacc_warmstart: tuple = (6., 7.)
            eq_active: tuple = (True, False)
            time: float = 2.
            integration_state: tuple = (11., 13.)
            extra_native_field: object = None

        state = NativeSnapshot(extra_native_field=MappingProxyType({"untouched": 19.}))
        result = {"request": request, "initial_state": state, "final_state": state,
                  "underlying_result": {"samples": [999.]},
                  "moves": [{"request": request, "underlying_result": {"samples": [999.]}}]}
        encoded = wb._evidence_value(result)
        json.dumps(encoded, allow_nan=False)
        self.assertEqual(encoded["request"]["whole_body"]["waist_target"], dict(motion.waist_target))
        self.assertEqual(encoded["initial_state"]["integration_state"], [11., 13.])
        self.assertEqual(encoded["initial_state"]["extra_native_field"], {"untouched": 19.})
        self.assertNotIn("underlying_result", encoded)
        self.assertNotIn("underlying_result", encoded["moves"][0])
        self.assertIn("underlying_result", result)
        with self.assertRaises(TypeError):
            motion.waist_target["waist_pitch"] = .9
        for bad in (np.array([np.nan]), np.float64(np.inf), {"finite": False}, {"nested": [float("nan")]}):
            self.assertFalse(wb._finite_evidence(bad))
        self.assertTrue(wb._finite_evidence({"capacity": None, "motion": motion, "state": state}))

    def test_atomic_negatives_require_valid_source_zero_steps_and_exact_unchanged_state(self):
        for negative in wb.NEGATIVES:
            record = self.record(negative=negative)
            audit = wb.audit_motion(self.model, record)
            self.assertTrue(wb._case_verdict("negative", negative, record, audit))
            for field, value in (("steps", 1), ("released", True), ("success", True),
                                 ("final_reference", {}), ("status", "SUCCESS")):
                with self.subTest(negative=negative, field=field):
                    changed = {**record, field: value}
                    self.assertFalse(wb._case_verdict("negative", negative, changed, audit))
            for field in ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active", "integration_state", "time"):
                changed = copy.deepcopy(record)
                changed["final_state"][field] = [999.] if field != "time" else 3.
                self.assertFalse(wb._case_verdict("negative", negative, changed, audit))
            record["initial_static"]["success"] = False
            self.assertFalse(wb._case_verdict("negative", negative, record, audit))

    def test_family_needs_native_readiness_clock_and_zero_forces_not_just_success(self):
        record = self.record()
        audit = wb.audit_motion(self.model, record)
        self.assertTrue(wb._case_verdict("family", "right_hand", record, audit))
        for key, value in (("root_linear_m_s", .101), ("root_angular_rad_s", .501), ("joint_max_rad_s", 1.001)):
            changed = copy.deepcopy(record)
            changed["samples"][0][key] = value
            self.assertFalse(wb._case_verdict("family", "right_hand", changed, audit))
        for change in ("force", "duration", "missing_force", "readiness", "nonfinite", "source"):
            changed = copy.deepcopy(record)
            if change == "force":
                changed["samples"][0]["qfrc_applied"][0] = 1.
            elif change == "duration":
                changed["duration_s"] = 1.
            elif change == "missing_force":
                del changed["samples"][0]["external_force_world_N"]
            elif change == "readiness":
                changed["readiness"]["ready"] = False
            elif change == "source":
                changed["initial_static"]["final_state"]["ctrl"][0] = 1.
            else:
                changed["samples"][0]["qvel"][0] = np.nan
            measured = None if change == "nonfinite" else wb.audit_motion(self.model, changed)
            self.assertFalse(wb._case_verdict("family", "right_hand", changed, measured), change)

    def test_sequence_requires_exact_handoff_including_integration_state(self):
        first = self.record()
        second = self.record()
        second.update(moving_limb="LEFT_HAND", source="left_hand", target="left_reach_target")
        second["initial_state"] = copy.deepcopy(first["final_state"])
        second["final_state"]["time"] += .004
        for row in second["samples"]:
            row["time_s"] += .004
        result = {**first, "kind": "sequence", "moves": [first, second], "completed_moves": 2,
                  "final_state": second["final_state"], "steps": 4, "duration_s": .008}
        self.assertTrue(wb._case_verdict("family", "sequence", result, wb.audit_motion(self.model, result)))
        for field in ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active", "integration_state", "time"):
            changed = copy.deepcopy(result)
            initial = changed["moves"][1]["initial_state"]
            if field == "time":
                initial[field] += 1e-10
            else:
                initial[field][0] += 1
            self.assertFalse(wb._case_verdict("family", "sequence", changed, wb.audit_motion(self.model, changed)), field)

    def test_whole_body_model_reconstruction_is_hash_bound_without_forward_or_ik(self):
        evidence = {**self.record(), "fixture_inputs": self.metadata_fixture[1]}
        before = copy.deepcopy(evidence)
        model, data, scene, profile, seed = clean._matching_fixture(evidence, .002, "whole_body")
        self.assertIsNone(data)
        self.assertEqual(wb._model_hash(model), wb._model_hash(self.model))
        self.assertEqual(scene.to_dict(), self.scene.to_dict())
        self.assertEqual(profile.to_dict(), self.profile.to_dict())
        np.testing.assert_array_equal(seed, self.seed)
        self.assertEqual(evidence, before)
        for field, value in (("factory", "unknown"), ("dt_s", .001), ("model_xml", "bad XML"),
                             ("compiled_model_sha256", "0" * 64), ("scene", {})):
            changed = copy.deepcopy(evidence)
            changed["fixture_inputs"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                clean._matching_fixture(changed, .002, "whole_body")
        changed = copy.deepcopy(evidence)
        changed["fixture_inputs"]["compiled_model_sha256"] = "0" * 64
        inputs = changed["fixture_inputs"]
        inputs["input_sha256"] = wb._hash({key: value for key, value in inputs.items() if key != "input_sha256"})
        with self.assertRaisesRegex(ValueError, "compiled model differs"):
            clean._matching_fixture(changed, .002, "whole_body")
        with self.assertRaisesRegex(ValueError, "fixture"):
            clean._matching_fixture(evidence, .002, "stage5")

    def test_provenance_covers_every_native_source_and_new_whole_body_modules(self):
        provenance = wb._provenance(["--suite", "family"])
        required = {*clean.PROVENANCE_MODULES, "boulder_v1.whole_body_demo", "boulder_v1.whole_body_motion",
                    "boulder_v1.whole_body_reference", "boulder_v1.static_control", "boulder_v1.runtime"}
        self.assertTrue(required.issubset(provenance["modules"]))
        for name, info in provenance["modules"].items():
            self.assertEqual(info["file"], wb.ROOT / "src" / (name.replace(".", "/") + ".py"))
            self.assertEqual(info["sha256"], hashlib.sha256(info["file"].read_bytes()).hexdigest())
        self.assertEqual(provenance["command"][1], str(wb.ROOT / "scripts" / "validate_whole_body.py"))

    def test_cli_writes_one_native_record_manifest_report_and_never_duplicates_primitive_samples(self):
        record = self.record()
        record["underlying_result"] = copy.deepcopy(record)
        original = copy.deepcopy(record)
        with tempfile.TemporaryDirectory(dir=wb.ROOT / "outputs") as directory, \
                mock.patch.object(wb, "_fixture_metadata", return_value=self.metadata_fixture), \
                mock.patch.object(wb, "_execute_case", return_value=record) as execute, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = wb.main(["--suite", "family", "--case", "right_hand", "--dt", ".002", "--summary",
                            "--output", directory])
            self.assertEqual(code, 0)
            execute.assert_called_once_with("right_hand", .002)
            output = Path(directory)
            saved = json.loads((output / "right_hand_2ms.json").read_text())
            self.assertNotIn("underlying_result", saved)
            self.assertEqual(saved["samples"], record["samples"])
            self.assertEqual(saved["final_state"], record["final_state"])
            self.assertEqual(saved["fixture_inputs"], self.metadata_fixture[1])
            self.assertIsNone(saved["motion_audit"]["quality_verdict"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["episodes"]), 1)
            self.assertEqual(manifest["episodes"][0]["evidence_sha256"],
                             hashlib.sha256((output / "right_hand_2ms.json").read_bytes()).hexdigest())
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertIsNone(report["visual_quality_verdict"])
            self.assertNotIn("provenance", json.loads(stdout.getvalue()))
        self.assertEqual(record, original)

    def test_missing_evidence_and_source_change_during_run_are_not_partial_acceptance(self):
        for mode in ("execution_error", "source_change"):
            provenance = {"modules": {}, "validator_sha256": "same"}
            current = {"modules": {"changed": {}}, "validator_sha256": "same"} if mode == "source_change" else provenance
            summary = {"name": "right_hand_2ms", "suite": "family", "case": "right_hand", "dt_s": .002,
                       "status": "SUCCESS", "reason": "synthetic", "physical_success": True, "accepted": True,
                       "finite_episode": True, "steps": 2, "duration_s": .004, "audit_error": None,
                       "evidence_json": "unit", "evidence_sha256": "unit"}
            with tempfile.TemporaryDirectory(dir=wb.ROOT / "outputs") as directory, \
                    mock.patch.object(wb, "_provenance", side_effect=[provenance, current]), \
                    mock.patch.object(wb, "_fixture_metadata", return_value=self.metadata_fixture), \
                    mock.patch.object(wb, "_run_case", **({"side_effect": OSError("export failed")}
                                                        if mode == "execution_error" else {"return_value": summary})), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(wb.main(["--suite", "family", "--case", "right_hand", "--dt", ".002",
                                          "--summary", "--output", directory]), 1)
                report = json.loads((Path(directory) / "report.json").read_text())
                self.assertFalse(report["physical_acceptance"])
                self.assertFalse(report["passed"])

    def test_nonfinite_real_measurement_cannot_become_an_accepted_null(self):
        record = self.record()
        record["samples"][0]["qpos"][0] = float("nan")
        with tempfile.TemporaryDirectory(dir=wb.ROOT / "outputs") as directory, \
                mock.patch.object(wb, "_execute_case", return_value=record):
            summary = wb._run_case("family", "right_hand", .002, Path(directory), {}, self.metadata_fixture)
            self.assertFalse(summary["finite_episode"])
            self.assertFalse(summary["accepted"])
            saved = json.loads((Path(directory) / "right_hand_2ms.json").read_text())
            self.assertIsNone(saved["samples"][0]["qpos"][0])
            self.assertIn("Nonfinite", summary["audit_error"])

    def test_cli_rejects_wrong_suite_and_nonowned_or_preservation_output_before_execution(self):
        options = [["--suite", "negative", "--case", "foot"], ["--output", str(wb.ROOT / "scripts")],
                   ["--output", str(wb.ROOT / "outputs")],
                   ["--output", str(wb.ROOT / "outputs" / "stage5-preserved-future")]]
        for arguments in options:
            with self.subTest(arguments=arguments), mock.patch.object(wb, "_execute_case") as execute, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                wb.main(arguments)
            self.assertEqual(caught.exception.code, 2)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
