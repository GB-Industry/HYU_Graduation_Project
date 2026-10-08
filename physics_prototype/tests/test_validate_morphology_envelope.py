"""Fast contracts only: synthetic saved states never certify native physics."""
import argparse
import contextlib
import copy
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.schema import Limb
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.transfers import make_transfer_fixture
from scripts import validate_morphology_envelope as study


class ValidateMorphologyEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.scene, _, cls.seed = make_transfer_fixture(.002)
        cls.profile = study.envelope.study_profiles()["baseline"]
        cls.inputs = study._evidence_value({"profile": cls.profile.to_dict(), "target_case": "nominal",
                      "scene": cls.scene.to_dict(), "seed_qpos": cls.seed.tolist(),
                      "model_xml_sha256": "synthetic-xml", "compiled_rom_rad": {},
                      "motor_ceiling_Nm": cls.model.actuator_gear[:, 0].tolist(), "mass_kg": 78.3,
                      "target_offset_world_m": [0., 0., 0.]})
        cls.fixture = (cls.model, {**cls.inputs, "compiled_model_sha256": study._model_hash(cls.model),
                      "compiled_geometry_sha256": "synthetic-geometry", "scene_sha256": study._hash(cls.inputs["scene"]),
                      "input_sha256": study._hash(cls.inputs),
                      "expected_hand_capacity_N": {r.id: 850. for r in cls.scene.contact_regions}}, cls.inputs)

    def setUp(self):
        stack = self.enterContext(contextlib.ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData"):
            stack.enter_context(mock.patch.object(mujoco, name, side_effect=AssertionError("Forbidden " + name)))
        stack.enter_context(mock.patch.object(study.envelope, "run_envelope_case", side_effect=AssertionError("No dynamics")))
        stack.enter_context(mock.patch.object(study.envelope, "make_envelope_fixture", side_effect=AssertionError("No source solve")))
        stack.enter_context(mock.patch.object(study.envelope, "target_perturbations", return_value={
            "nominal": (0., 0., 0.), "up": (0., .01, 0.), "down": (0., -.01, 0.),
            "lateral_minus": (-.01, 0., 0.), "lateral_plus": (.01, 0., 0.)}))

    def record(self, classification="PHYSICAL_SUCCESS", geometric=None):
        success = classification == "PHYSICAL_SUCCESS"
        source = self.scene.to_dict()["start_configuration"]
        supports = {limb: hold for limb, hold in source.items() if limb != "RIGHT_HAND"}
        goal = {**source, "RIGHT_HAND": "reach_target"}
        release_time, reach_time, capture_time, end_time = (2. + step * .002 for step in (2501, 2751, 5302, 5553))
        capture = {"time_s": capture_time, "step": 5302, "gap_m": .0002, "relative_speed_m_s": .0001,
                   "relative_velocity_world_m_s": [0., 0., .0001], "orientation": 1., "penetration_m": 0.,
                   "capture_margin_m": .0008, "initial_reaction_N": 20., "acquisition_policy": "endpoint_settle",
                   "reach_elapsed_s": capture_time - reach_time, "reference_time_s": 5.1,
                   "post_activation_decisions": {limb: {"maintain": True, "required_load": 20.,
                       "effective_capacity": 850., "utilization": 20. / 850., "reason": "within bounded point-grasp capacity"}
                       for limb in ("LEFT_HAND", "RIGHT_HAND")}}
        native_capture = {**capture, "limb": "RIGHT_HAND", "region_id": "reach_target", "mode": "physical",
                          "initial_reaction_world_N": [0., 0., 20.], "initial_reaction_epoch": "fresh_post_activation_solve"}

        def state(time, contacts=None):
            contacts = source if contacts is None else contacts
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.seed
            scratch.time = time
            scratch.eq_active[:] = False
            for limb, hold in contacts.items():
                if limb.endswith("HAND"):
                    scratch.eq_active[self.model.equality(f"grasp_{limb.lower()}_{hold}").id] = True
            vector = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(self.model, scratch, vector, mujoco.mjtState.mjSTATE_INTEGRATION)
            return {"time": time, "qpos": self.seed.tolist(), "qvel": scratch.qvel.tolist(),
                    "ctrl": scratch.ctrl.tolist(), "qacc_warmstart": scratch.qacc_warmstart.tolist(),
                    "eq_active": scratch.eq_active.tolist(), "finite": True, "contact_mode": "physical",
                    "contact_configuration": contacts,
                    "hand_states": {limb: {"active": limb in contacts, "valid": limb in contacts, "measurement_valid": True,
                        "load": 20. if limb in contacts else 0., "capacity": 850. if limb in contacts else None,
                        "margin": 830. if limb in contacts else None, "region_id": contacts.get(limb),
                        "force_world": [0., 0., 20. if limb in contacts else 0.]}
                        for limb in ("LEFT_HAND", "RIGHT_HAND")},
                    "foot_states": {limb: {"measurement_valid": True, "supporting": True, "contacting": True,
                        "slipping": False, "normal_force": 350., "tangential_force": 0., "tangential_speed": 0.,
                        "support_regions": [contacts[limb]], "support_surfaces": ["geom_" + contacts[limb]],
                        "idealized_attachment": None, "contacts": [{"surface_geom": "geom_" + contacts[limb],
                            "shoe_geom": limb.lower() + "_geom", "admissible": True, "normal_force": 350., "tangential_speed": 0.}]}
                        for limb in ("LEFT_FOOT", "RIGHT_FOOT")},
                    "capture_events": [native_capture] if contacts == goal else []}, vector.tolist()

        initial, iv = state(2.)
        final, fv = state(end_time if success else 2., goal if success else source)
        static_start, _ = state(0.)
        zero_forces = {"qfrc_applied": [0.] * self.model.nv,
                       "external_force_world_N": [[0.] * 6 for _ in range(self.model.nbody)]}
        source_readiness = {"ready": True, "duration": .5, "reason": "Sustained physical readiness", "time": 2.,
                            "root_linear_speed": 0., "root_angular_speed": 0., "max_hinge_speed": 0., "rms_hinge_speed": 0.}
        source_terminal = {"terminal": True, "pose_available": True, "status": "SUCCESS", "time_s": 2.,
                           "reason": "Synthetic static hold schema only", "ctrl": initial["ctrl"],
                           "hands": initial["hand_states"], "feet": initial["foot_states"],
                           "readiness": source_readiness, "unintended_contacts": [], "disturbance_active": False}
        readiness = {"ready": success, "duration": .5 if success else 0., "reason": "Sustained physical readiness",
                     "time": final["time"], "root_linear_speed": 0., "root_angular_speed": 0.,
                     "max_hinge_speed": 0., "rms_hinge_speed": 0.}
        samples = []
        detached, _ = state(release_time, supports)
        if success:
            for i in range(1, 5554):
                snapshot = initial if i <= 2501 else detached if i <= 5302 else final
                phase = ("SOURCE_STABILIZE" if i <= 251 else "LOAD_TRANSFER" if i <= 2251 else
                         "RELEASE_CLEARANCE" if i <= 2501 else "THREE_POINT" if i <= 2751 else "REACH" if i <= 5302 else "SETTLE")
                time = 2. + i * .002
                samples.append({"time_s": time, "qpos": snapshot["qpos"], "qvel": snapshot["qvel"],
                    "ctrl": snapshot["ctrl"], "eq_active": snapshot["eq_active"], "steps": i,
                    "q_ref": snapshot["qpos"], "qd_ref": snapshot["qvel"],
                    "root_linear_m_s": 0., "root_angular_rad_s": 0., "joint_max_rad_s": 0., "phase": phase,
                    "hands": snapshot["hand_states"], "feet": snapshot["foot_states"], "contacts": snapshot["contact_configuration"],
                    "readiness": {**readiness, "time": time, "ready": i == 5553, "duration": max(0., (i - 5303) * .002)},
                    **zero_forces})
        result = {"profile": self.profile.to_dict(), "target_case": "nominal", "dt_s": .002,
                "kind": "right_hand", "moving_limb": Limb.RIGHT_HAND, "source": "right_hand", "target": "reach_target",
                "fixture_inputs": copy.deepcopy(self.inputs), "classification": classification,
                "status": "SUCCESS" if success else classification, "reason": "Synthetic contract only, no physical evidence",
                "success": success, "steps": 5553 if success else 0, "duration_s": 11.106 if success else 0.,
                "released": success, "final_reference": {"qpos": self.seed.tolist(), "contact_intent": goal,
                    "target_pose": {self.model.joint(int(j)).name: float(self.seed[self.model.jnt_qposadr[j]])
                                    for j in self.model.actuator_trnid[:, 0]}} if success else None,
                "initial_state": initial, "final_state": final, "initial_integration_state": iv,
                "final_integration_state": fv, "samples": samples, "events": [], "readiness": readiness,
                "initial_static": {"success": True, "steps": 1000, "duration_s": 2., "initial_state": static_start,
                                   "final_state": copy.deepcopy(initial), "terminal_observation": source_terminal,
                                   "status": "SUCCESS", "reason": source_terminal["reason"], "readiness": source_readiness,
                                   "contact_acceptance": True, "controller_convergence": True,
                                   "disturbance": None, "disturbance_realization": None},
                "assessment": {"feasible": success, "classification": "GEOMETRICALLY_FEASIBLE" if success else classification,
                    "reason": "Synthetic contract only, no physical evidence", "geometric_feasible": True if success else geometric,
                    "support_feasible": True if success else None, "preparation_qpos": [self.seed.tolist()],
                    "diagnostics": {"actual_integration_state": iv, "actual_qpos": self.seed.tolist(),
                                    "candidate_support": [{"full_geometric_load_evidence": [1., 2., 3.]}]}}}
        if success:
            result.update(request={"limb": "RIGHT_HAND", "source": "right_hand", "target": "reach_target",
                "source_contacts": source, "support_contacts": supports, "hand_reach_s": 5.,
                "hand_acquisition_policy": "endpoint_settle"}, final_contacts=goal, new_contacts=goal,
                release_time_s=release_time, reach_start_time_s=reach_time, readiness_time_s=end_time,
                capture=capture, capture_time_s=capture_time, capture_error_m=.0002, capture_margin_m=.0008,
                capture_events=[native_capture], release_events=[])
            for phase, step in (("SOURCE_STABILIZE", 0), ("LOAD_TRANSFER", 251), ("RELEASE_CLEARANCE", 2251),
                                ("THREE_POINT", 2501), ("REACH", 2751), ("SETTLE", 5302)):
                result["events"].append({"phase": phase, "time_s": 2. + step * .002, "step": step})
            result["events"].extend([{"event": "RELEASED", "time_s": release_time, "step": 2501,
                "limb": "RIGHT_HAND", "source": "right_hand"}, {"event": "CAPTURED", "time_s": capture_time, "step": 5302},
                {"event": "FINAL_READY", "time_s": end_time, "step": 5553}])
            result["events"].sort(key=lambda event: event["time_s"])
            result["terminal_observation"] = {**copy.deepcopy(samples[-1]), "terminal": True, "pose_available": True,
                "status": "SUCCESS", "reason": result["reason"], "readiness": readiness, "root_pose": final["qpos"][:7]}
        return result

    def verdict(self, record, motion=None):
        # Records are already JSON-shaped; production finiteness-before-serialization
        # is tested separately. Do not duplicate full per-step serializer walks here.
        return study._limit_checks(self.model, self.fixture[1], record) and study._case_verdict(
            record, motion or study.audit_motion(self.model, record), study._integration_checks(self.model, record), self.fixture[1])

    def seal_terminal(self, record, row):
        """Encode a synthetic actual endpoint, without dynamics or invented steps."""
        final = copy.deepcopy(record["final_state"])
        final.update(time=row["time_s"], contact_configuration=copy.deepcopy(row["contacts"]),
                     hand_states=copy.deepcopy(row["hands"]), foot_states=copy.deepcopy(row["feet"]),
                     capture_events=copy.deepcopy(record.get("capture_events", [])))
        scratch = mujoco.MjData(self.model)
        for key in ("qpos", "qvel", "ctrl", "eq_active"):
            final[key] = copy.deepcopy(row[key])
            getattr(scratch, key)[:] = row[key]
        scratch.qacc_warmstart[:] = final["qacc_warmstart"]
        scratch.time = final["time"]
        vector = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, scratch, vector, mujoco.mjtState.mjSTATE_INTEGRATION)
        record.update(final_state=final, final_integration_state=vector.tolist(), steps=row["steps"],
                      duration_s=row["steps"] * record["dt_s"], final_contacts=row["contacts"], new_contacts=row["contacts"])
        velocity = np.asarray(final["qvel"])
        hinges = velocity[self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]]
        speeds = (float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:6])),
                  float(np.max(np.abs(hinges))), float(np.sqrt(np.mean(hinges ** 2))))
        record["readiness"] = {"ready": False, "duration": 0., "reason": record["reason"], "time": final["time"],
            **dict(zip(("root_linear_speed", "root_angular_speed", "max_hinge_speed", "rms_hinge_speed"), speeds))}
        record["terminal_observation"] = {**copy.deepcopy(row), "terminal": True, "pose_available": True,
            "root_pose": final["qpos"][:7], "status": record["status"], "reason": record["reason"],
            "readiness": record["readiness"], "root_linear_m_s": speeds[0], "root_angular_rad_s": speeds[1],
            "joint_max_rad_s": speeds[2]}

    def dynamic_record(self, status):
        record = self.record()
        record.update(classification="DYNAMIC_CONTACT_INFEASIBLE", status=status, success=False,
                      final_reference=None, readiness_time_s=None, reason="Synthetic witnessed " + status)
        record["events"] = [e for e in record["events"] if e.get("event") != "FINAL_READY"]
        if status in ("CAPTURE_FAILURE", "TIMEOUT"):
            phase = "REACH" if status == "CAPTURE_FAILURE" else "SETTLE"
            start = record["reach_start_time_s"] if phase == "REACH" else record["capture_time_s"]
            end = start + (7. if phase == "REACH" else 5.)
            steps = round((end - 2.) / .002)
            template = copy.deepcopy(record["samples"][3000 if phase == "REACH" else -1])
            if phase == "REACH":
                record.update(capture=None, capture_time_s=None, capture_error_m=None, capture_margin_m=None, capture_events=[])
                record["events"] = [e for e in record["events"] if e.get("event") != "CAPTURED" and e.get("phase") != "SETTLE"]
            old_rows = record["samples"]
            record["samples"] = []
            for step in range(1, steps + 1):
                time = 2. + step * .002
                row = dict(old_rows[step - 1]) if time <= start else dict(template)
                row.update(time_s=time, steps=step)
                if time > start:
                    row.update(phase=phase, readiness={"ready": False, "duration": 0.,
                               "reason": "Root linear speed exceeds 0.02 m/s" if phase == "SETTLE" else "Awaiting contacts"})
                    if phase == "SETTLE":
                        row["qpos"] = list(template["qpos"])
                        row["qpos"][0] += .03 * (time - start)
                        row["qvel"] = list(template["qvel"])
                        row["qvel"][0] = .03
                        row["root_linear_m_s"] = .03
                record["samples"].append(row)
            self.seal_terminal(record, record["samples"][-1])
        else:
            terminal = copy.deepcopy(record["samples"][-1])
            terminal.update(time_s=terminal["time_s"] + .002, steps=terminal["steps"] + 1)
            if status == "CONTACT_LOSS":
                terminal["feet"]["LEFT_FOOT"].update(supporting=False, normal_force=0., support_regions=[], support_surfaces=[], contacts=[])
                terminal["contacts"].pop("LEFT_FOOT")
                record["guard_failure"] = {"feet": terminal["feet"], "hands": terminal["hands"],
                                            "applied_unintended": [], "endpoint_unintended": []}
            if status == "GRIP_FAILURE":
                terminal["hands"]["RIGHT_HAND"].update(active=False, valid=False, capacity=None, margin=None,
                    region_id=None, load=0., force_world=[0., 0., 0.])
                terminal["contacts"].pop("RIGHT_HAND")
                terminal["eq_active"][self.model.equality("grasp_right_hand_reach_target").id] = False
                record["release_events"] = [{"limb": "RIGHT_HAND", "region_id": "reach_target",
                    "time_s": terminal["time_s"], "required_load_N": 900., "capacity_N": 850.,
                    "force_world_N": [0., 0., 900.], "force_epoch": "fresh_endpoint_solve",
                    "force_state_time_s": terminal["time_s"], "reason": "capacity exceeded"}]
            self.seal_terminal(record, terminal)
        return record

    def test_success_requires_measured_capture_goal_and_sustained_terminal_readiness(self):
        record = self.record()
        motion = study.audit_motion(self.model, record)
        self.assertTrue(self.verdict(record, motion))
        for mode in ("missing_capture", "capture_speed", "capture_gap", "capture_orientation", "capture_penetration",
                     "capture_margin", "capture_clock", "capture_event", "release_event", "ready_event",
                     "short_window", "readiness_speed", "wrong_goal", "final_ref", "terminal_state",
                     "terminal_reason", "post_activation", "initial_reaction", "fake_native_foot", "source_telemetry", "force_shape"):
            changed = copy.deepcopy(record)
            capture = changed["capture"]
            if mode == "missing_capture":
                changed["capture"] = None
            elif mode == "capture_speed":
                capture.update(relative_speed_m_s=.051, relative_velocity_world_m_s=[0., 0., .051])
            elif mode == "capture_gap":
                capture["gap_m"] = .00101
            elif mode == "capture_orientation":
                capture["orientation"] = .8
            elif mode == "capture_penetration":
                capture["penetration_m"] = .001
            elif mode == "capture_margin":
                capture["capture_margin_m"] = .001
            elif mode == "capture_clock":
                capture["time_s"] = changed["final_state"]["time"] + .002
            elif mode in ("capture_event", "release_event", "ready_event"):
                event = {"capture_event": "CAPTURED", "release_event": "RELEASED", "ready_event": "FINAL_READY"}[mode]
                changed["events"] = [e for e in changed["events"] if e.get("event") != event]
            elif mode == "short_window":
                changed["readiness"]["duration"] = .499
            elif mode == "readiness_speed":
                changed["readiness"]["root_linear_speed"] = .03
            elif mode == "wrong_goal":
                changed["final_state"]["hand_states"]["RIGHT_HAND"]["region_id"] = "right_hand"
            elif mode == "final_ref":
                changed["final_reference"]["qpos"][7] = 999.
            elif mode == "terminal_state":
                changed["terminal_observation"]["ctrl"][0] = .1
            elif mode == "terminal_reason":
                changed["terminal_observation"]["reason"] = "Not a success"
            elif mode == "post_activation":
                capture["post_activation_decisions"]["LEFT_HAND"]["maintain"] = False
            elif mode == "fake_native_foot":
                changed["final_state"]["foot_states"]["LEFT_FOOT"]["contacts"][0]["surface_geom"] = "floor"
            elif mode == "source_telemetry":
                changed["initial_static"]["readiness"]["root_linear_speed"] = .019
            elif mode == "force_shape":
                changed["terminal_observation"]["external_force_world_N"] = 0.
            else:
                capture["initial_reaction_N"] = 900.
                changed["capture_events"][0].update(initial_reaction_N=900., initial_reaction_world_N=[0., 0., 900.])
            self.assertFalse(self.verdict(changed, motion), mode)

    def test_old_two_step_ready_flag_without_capture_cannot_certify(self):
        weak = self.record()
        weak["samples"] = weak["samples"][:2]
        weak.update(capture=None, capture_events=[], events=[], final_reference={"qpos": self.seed.tolist()})
        self.seal_terminal(weak, weak["samples"][-1])
        weak.update(status="SUCCESS", success=True, classification="PHYSICAL_SUCCESS", readiness={"ready": True})
        self.assertFalse(self.verdict(weak))

    def test_witnessed_dynamic_failures_are_diagnostics_not_successes(self):
        for status in ("CAPTURE_FAILURE", "TIMEOUT", "CONTACT_LOSS", "GRIP_FAILURE"):
            with self.subTest(status=status):
                record = self.dynamic_record(status)
                motion = study.audit_motion(self.model, record)
                self.assertTrue(self.verdict(record, motion), status)
                self.assertFalse(study._physical_success(record))
                changed = copy.deepcopy(record)
                if status in ("CAPTURE_FAILURE", "TIMEOUT"):
                    event = next(e for e in changed["events"] if e.get("phase") == ("REACH" if status == "CAPTURE_FAILURE" else "SETTLE"))
                    event["time_s"] += .002
                    event["step"] += 1
                elif status == "CONTACT_LOSS":
                    changed["guard_failure"] = None
                else:
                    changed["release_events"][0]["required_load_N"] = 849.
                self.assertFalse(self.verdict(changed, motion), status)
                changed = copy.deepcopy(record)
                changed.update(status="CONTROL_FAILURE", reason="Unexpected controller exception")
                changed["terminal_observation"].update(status=changed["status"], reason=changed["reason"])
                changed["readiness"]["reason"] = changed["reason"]
                self.assertFalse(self.verdict(changed, motion))

    def saved_packet(self, directory, record=None):
        path = Path(directory) / "baseline_nominal_2ms.json"
        provenance = study._json_value(study._provenance([]))
        provenance["validator_sha256"] = "older-validator-not-an-acceptance-authority"
        packet = {**study._evidence_value(record or self.record("ROM_LIMITED_SEARCH")),
                  "fixture_inputs": {**self.fixture[1], "native_metadata": self.inputs}, "provenance": provenance,
                  "validation": {"name": path.stem, "finite_episode": True, "accepted": False},
                  "motion_audit": {"obsolete": True}, "integration_audit": {"obsolete": True}}
        study._write_json(path, packet, compact=True)
        return path, study._json_value(packet)

    def test_resume_reaudits_read_only_and_preserves_generation_provenance_without_nesting(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                mock.patch.object(study, "_execute_case", side_effect=AssertionError("No native replay")) as execute, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            path, original = self.saved_packet(directory)
            raw = path.read_bytes()
            for _ in range(2):
                self.assertEqual(study.main(["--resume", "--profile", "baseline", "--case", "nominal", "--dt", ".002",
                                             "--summary", "--output", directory]), 0)
                self.assertEqual(path.read_bytes(), raw)
                audit = json.loads(path.with_suffix(".audit.json").read_bytes())
                report = json.loads((Path(directory) / "report.json").read_bytes())
                self.assertEqual(audit["provenance"], original["provenance"])
                self.assertEqual(audit["audit_provenance"]["validator_sha256"], study._provenance([])["validator_sha256"])
                self.assertEqual(audit["validation"]["resumed_from"]["evidence_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertNotIn("obsolete", audit["motion_audit"])
                self.assertNotIn("audit_provenance", audit["provenance"])
                self.assertEqual(report["resumed_case_count"], 1)
                self.assertTrue(report["partial_scope"])
            self.assertIn("[RE-AUDIT]", stdout.getvalue())
            execute.assert_not_called()

    def test_resume_executes_only_missing_cases_once(self):
        missing = self.record("ROM_LIMITED_SEARCH")
        missing["dt_s"] = .001
        missing["initial_static"]["steps"] = 2000
        with tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                mock.patch.object(study, "_execute_case", return_value=missing) as execute, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.saved_packet(directory)
            self.assertEqual(study.main(["--resume", "--profile", "baseline", "--case", "nominal", "--summary", "--output", directory]), 0)
            execute.assert_called_once_with(self.profile, "nominal", .001)
            self.assertIn("[EXECUTE] baseline_nominal_1ms", stdout.getvalue())
            report = json.loads((Path(directory) / "report.json").read_bytes())
            self.assertEqual((report["resumed_case_count"], report["executed_case_count"]), (1, 1))

    def test_resume_rejects_stale_corrupt_or_mismatched_artifacts_without_overwrite_or_execution(self):
        for mode in ("modules", "source_path", "profile", "dt", "environment", "model_hash", "native_metadata",
                     "body_force", "sanitized_nonfinite", "json", "duplicate_fields"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                    mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                    mock.patch.object(study, "_execute_case", side_effect=AssertionError("No native rerun")) as execute, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                path, packet = self.saved_packet(directory)
                if mode == "modules":
                    packet["provenance"]["modules"]["boulder_v1.morphology_envelope"]["sha256"] = "stale"
                elif mode == "source_path":
                    packet["provenance"]["modules"]["boulder_v1.morphology_envelope"]["file"] = "/another/worktree/morphology_envelope.py"
                elif mode == "profile":
                    packet["profile"]["name"] = "longer"
                elif mode == "dt":
                    packet["dt_s"] = .001
                elif mode == "environment":
                    packet["provenance"]["package_versions"]["mujoco"] = "different"
                elif mode == "model_hash":
                    packet["fixture_inputs"]["compiled_model_sha256"] = "different"
                elif mode == "native_metadata":
                    packet["fixture_inputs"]["native_metadata"]["seed_qpos"][0] += .01
                elif mode == "body_force":
                    scratch = mujoco.MjData(self.model)
                    spec = mujoco.mjtState.mjSTATE_INTEGRATION
                    mujoco.mj_setState(self.model, scratch, np.asarray(packet["initial_integration_state"]), spec)
                    scratch.qfrc_applied[0] = 1.
                    vector = np.empty(mujoco.mj_stateSize(self.model, spec))
                    mujoco.mj_getState(self.model, scratch, vector, spec)
                    packet["initial_integration_state"] = vector.tolist()
                elif mode == "sanitized_nonfinite":
                    packet["validation"]["finite_episode"] = False
                if mode == "json":
                    with path.open("w") as stream:
                        stream.write("{")
                elif mode == "duplicate_fields":
                    raw = path.read_bytes()
                    path.write_bytes(b'{"profile":{"name":"corrupt"},' + raw[1:])
                else:
                    study._write_json(path, packet, compact=True)
                raw = path.read_bytes()
                self.assertEqual(study.main(["--resume", "--profile", "baseline", "--case", "nominal", "--summary", "--output", directory]), 1)
                execute.assert_not_called()
                self.assertEqual(path.read_bytes(), raw)
                self.assertFalse(path.with_suffix(".audit.json").exists())

    def negative_fixture(self, case):
        """Compiled solid geometry certificates only, never a native episode."""
        source = study.canonical_geometry(self.scene.region("right_hand")).hand_frame
        target = study.canonical_geometry(self.scene.region("reach_target")).hand_frame
        clear = study.Frame(tuple(np.asarray(source.position) + .01 * np.asarray(source.normal)), source.rotation)
        point = study.reach_frame(clear, target, .5, 5., .02)
        if case == "beyond_reach":
            scene = replace(self.scene, contact_regions=tuple(replace(r, position=(r.position[0] + 2., *r.position[1:]))
                            if r.id == "reach_target" else r for r in self.scene.contact_regions))
        else:
            obstacle = replace(self.scene.region("right_hand"), id="path_obstacle", position=point.position,
                               affordances=frozenset((study.Affordance.STEP,)), half_size=(.02, .02, .02))
            scene = replace(self.scene, contact_regions=(*self.scene.contact_regions, obstacle))
        tree = ET.fromstring(build_mjcf(scene, self.profile))
        if case == "blocked_path":
            ET.SubElement(tree.find("contact"), "pair", geom1="right_hand_geom", geom2="geom_path_obstacle", condim="3")
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        inputs = {**self.inputs, "scene": study._evidence_value(scene.to_dict()), "target_case": case}
        record = self.record("GEOMETRY_INFEASIBLE" if case == "beyond_reach" else "COLLISION_INFEASIBLE", geometric=False)
        record.update(target_case=case, fixture_inputs=inputs)
        assessment = record["assessment"]
        assessment["source_reference"] = {"qpos": self.seed.tolist()}
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = self.seed
        mujoco.mj_kinematics(model, scratch)
        if case == "beyond_reach":
            root = model.body("climber_root").id
            assessment["motion"] = {"root_target": {"position": scratch.xpos[root].tolist(),
                "rotation": scratch.xmat[root].reshape(3, 3).tolist()},
                "waist_target": {name: float(self.seed[int(model.joint(name).qposadr[0])])
                                 for name in ("waist_yaw", "waist_pitch", "waist_roll")}}
            shoulder, site = model.body("right_upper_arm").id, model.site("right_hand_site").id
            body, maximum = int(model.site_bodyid[site]), float(np.linalg.norm(model.site_pos[site]))
            while body != shoulder:
                maximum += float(np.linalg.norm(model.body_pos[body]))
                body = int(model.body_parentid[body])
            target = study.canonical_geometry(scene.region("reach_target")).hand_frame
            distance = float(np.linalg.norm(np.asarray(target.position) - scratch.xpos[shoulder]))
            assessment["diagnostics"]["reach_bound"] = {"distance_m": distance, "maximum_m": maximum,
                "gap_m": distance - maximum, "shoulder_world_m": scratch.xpos[shoulder].tolist(),
                "scope": "fixed policy endpoint root/waist; position-only upper bound"}
        else:
            hand, site, obstacle = model.geom("right_hand_geom").id, model.site("right_hand_site").id, model.geom("geom_path_obstacle").id
            body = int(model.geom_bodyid[obstacle])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[hand])
            local_site = rotation.reshape(3, 3).T @ (model.site_pos[site] - model.geom_pos[hand])
            local = scratch.geom_xmat[obstacle].reshape(3, 3).T @ (point.position - scratch.geom_xpos[obstacle])
            hm, om = float(np.min(model.geom_size[hand] - np.abs(local_site))), float(np.min(model.geom_size[obstacle] - np.abs(local)))
            assessment["diagnostics"]["path_collision_certificate"] = {
                "phase": "reach", "index": 10, "time_s": .5, "task_frame": study._evidence_value(point),
                "point_world_m": list(point.position), "point_obstacle_local_m": local.tolist(), "site_hand_local_m": local_site.tolist(),
                "hand_interior_margin_m": hm, "obstacle_interior_margin_m": om, "intersection_ball_radius_m": min(hm, om),
                "moving_geom": "right_hand_geom", "moving_site": "right_hand_site", "static_geom": "geom_path_obstacle",
                "static_body": model.body(body).name, "static_shape": "box", "explicit_pair": True, "static_body_weldid": 0,
                "static_geom_frame": {"position": scratch.geom_xpos[obstacle].tolist(), "rotation": scratch.geom_xmat[obstacle].reshape(3, 3).tolist()},
                "static_body_frame": {"position": scratch.xpos[body].tolist(), "rotation": scratch.xmat[body].reshape(3, 3).tolist()},
                "collision_masks": {"hand": [int(model.geom_contype[hand]), int(model.geom_conaffinity[hand])],
                    "obstacle": [int(model.geom_contype[obstacle]), int(model.geom_conaffinity[obstacle])]},
                "proof": "Required END point is strictly interior to both collidable rigid solids",
                "scope": "specified sampled effector path only; not a global transfer or native force claim"}
        return model, record, inputs

    def test_negative_certificates_are_recomputed_not_classification_labels(self):
        for case in study.NEGATIVES:
            model, record, inputs = self.negative_fixture(case)
            self.assertTrue(study._negative_certificate(model, record, case, inputs), case)
            key = "reach_bound" if case == "beyond_reach" else "path_collision_certificate"
            for mode in ("missing", "wrong_class", "zero_margin", "wrong_point", "wrong_mask"):
                changed = copy.deepcopy(record)
                certificate = changed["assessment"]["diagnostics"][key]
                if mode == "missing":
                    changed["assessment"]["diagnostics"].pop(key)
                elif mode == "wrong_class":
                    changed["classification"] = "LOCAL_SEARCH_UNRESOLVED"
                elif mode == "zero_margin":
                    certificate["gap_m" if case == "beyond_reach" else "intersection_ball_radius_m"] = 0.
                elif mode == "wrong_point":
                    certificate["shoulder_world_m" if case == "beyond_reach" else "point_world_m"][0] += .01
                elif case == "blocked_path":
                    certificate["collision_masks"]["hand"][0] ^= 1
                else:
                    certificate["maximum_m"] += .01
                self.assertFalse(study._negative_certificate(model, changed, case, inputs), (case, mode))

    def test_unresolved_primary_is_not_an_accepted_declared_counterexample(self):
        record = self.record("LOCAL_SEARCH_UNRESOLVED")
        self.assertTrue(self.verdict(record))
        for case in study.NEGATIVES:
            inputs = {**self.inputs, "target_case": case}
            record.update(target_case=case, fixture_inputs=inputs)
            fixture = (self.model, {**self.fixture[1], "target_case": case}, inputs)
            with tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                    mock.patch.object(study, "_fixture_metadata", return_value=fixture), \
                    mock.patch.object(study, "_execute_case", return_value=record), contextlib.redirect_stdout(io.StringIO()):
                summary = study._run_case("negative", "baseline", case, .002, Path(directory), {}, self.profile)
                self.assertFalse(summary["accepted"])
                self.assertFalse(summary["negative_certificate_verified"])

    def test_default_matrix_and_filters(self):
        args = argparse.Namespace(suite="all", profile=None, case=None, dt=None)
        jobs = study._jobs(args)
        self.assertEqual(len(jobs), 44)
        self.assertEqual(sum(j[0] == "matrix" for j in jobs), 40)
        self.assertEqual(jobs[-4:], [("negative", "baseline", c, dt) for c in study.NEGATIVES for dt in (.002, .001)])
        args.profile, args.case, args.dt = ["longer"], ["up"], .001
        self.assertEqual(study._jobs(args), [("matrix", "longer", "up", .001)])
        args.suite = "negative"
        self.assertEqual(study._jobs(args), [])

    def test_dispatch_passes_object_and_retains_every_sample(self):
        renamed = replace(self.profile, name="not_a_preset")
        with mock.patch.object(study.envelope, "run_envelope_case", return_value={}) as execute:
            study._execute_case(renamed, "up", .001)
            execute.assert_called_once_with(renamed, "up", .001, keep_samples=True)

    def test_study_changes_only_geometry_or_grip_not_mass_rom_strength(self):
        profiles = study.envelope.study_profiles()
        baseline = profiles["baseline"].to_dict()
        for name, profile in profiles.items():
            for key, value in profile.to_dict().items():
                if key not in (*study.LENGTHS, "name", "grip_capacity"):
                    self.assertEqual(value, baseline[key])
            scale = {"shorter": .95, "longer": 1.05}.get(name, 1.)
            for key in study.LENGTHS:
                self.assertAlmostEqual(getattr(profile, key), baseline[key] * scale)
            self.assertEqual(profile.grip_capacity, 425. if name == "reduced_grip" else 850.)

    def test_immutable_recursive_serializer_preserves_rich_diagnostics_and_unknown(self):
        @dataclass(frozen=True)
        class Assessment:
            feasible: bool = False
            geometric_feasible: object = None
            diagnostics: object = None
        raw = {"assessment": Assessment(diagnostics=MappingProxyType({Limb.RIGHT_HAND:
            MappingProxyType({"per_pose_loads": (np.array([1., 2.]),), "ROM": (0., 1.)})})),
            "underlying_result": {"samples": [999.]}}
        saved = study._evidence_value(raw)
        self.assertIsNone(saved["assessment"]["geometric_feasible"])
        self.assertEqual(saved["assessment"]["diagnostics"]["RIGHT_HAND"]["per_pose_loads"], [[1., 2.]])
        self.assertNotIn("underlying_result", saved)
        self.assertTrue(study._finite_evidence(raw))
        json.dumps(saved, allow_nan=False)
        for value in (np.nan, np.float64(np.inf), np.array([np.nan]), {"finite": False}):
            self.assertFalse(study._finite_evidence({"assessment": Assessment(diagnostics={"nested": [value]})}))

    def test_atomic_bounded_failures_accept_valid_unchanged_sources_not_global_claims(self):
        for classification in study.CLASSIFICATIONS - {"PHYSICAL_SUCCESS", "DYNAMIC_CONTACT_INFEASIBLE", "SOURCE_STATIC_FAILURE"}:
            record = self.record(classification)
            self.assertTrue(self.verdict(record), classification)
            self.assertIsNone(study._evidence_value(record)["assessment"]["geometric_feasible"])
            for field, value in (("steps", 1), ("released", True), ("final_reference", {}),
                                 ("success", True), ("status", "UNKNOWN"), ("samples", [{}])):
                changed = copy.deepcopy(record)
                changed[field] = value
                # Audit rejects malformed rows separately; these fields also fail the verdict.
                saved = study._evidence_value(changed)
                motion = study.audit_motion(self.model, study._evidence_value(record))
                self.assertFalse(study._case_verdict(saved, motion, {"initial": True, "final": True}))

    def test_physical_success_requires_live_source_forces_clock_and_integration(self):
        record = self.record()
        self.assertTrue(self.verdict(record))
        for mode in ("force", "missing_force", "clock", "source", "integration", "speed", "static_duration",
                     "invalid_hand", "invalid_foot", "assessment_replay", "nonfinite"):
            changed = copy.deepcopy(record)
            if mode == "force":
                changed["samples"][0]["qfrc_applied"][0] = 1.
            elif mode == "missing_force":
                del changed["samples"][0]["external_force_world_N"]
            elif mode == "clock":
                changed["duration_s"] = 1.
            elif mode == "source":
                changed["initial_static"]["final_state"]["ctrl"][0] = 1.
            elif mode == "integration":
                changed["final_integration_state"][0] += .1
            elif mode == "speed":
                changed["samples"][0]["root_linear_m_s"] = .101
            elif mode == "static_duration":
                changed["initial_static"]["duration_s"] = 0.
            elif mode == "invalid_hand":
                changed["initial_state"]["hand_states"]["LEFT_HAND"]["valid"] = False
            elif mode == "invalid_foot":
                changed["initial_state"]["foot_states"]["LEFT_FOOT"]["idealized_attachment"] = "fake"
            elif mode == "assessment_replay":
                changed["assessment"]["diagnostics"]["actual_qpos"][0] += .01
            else:
                changed["assessment"]["diagnostics"]["bad"] = np.nan
            self.assertFalse(self.verdict(changed), mode)

    def test_zero_step_mutation_source_failure_and_dynamic_failure_are_not_passes(self):
        record = self.record("ROM_LIMITED_SEARCH")
        for field in ("qpos", "qvel", "ctrl", "eq_active", "qacc_warmstart"):
            changed = copy.deepcopy(record)
            changed["final_state"][field][0] += 1
            self.assertFalse(self.verdict(changed), field)
        for classification in ("SOURCE_STATIC_FAILURE", "DYNAMIC_CONTACT_INFEASIBLE", "UNSUPPORTED_PRIMITIVE"):
            changed = self.record(classification)
            self.assertFalse(self.verdict(changed))

    def test_original_limits_cannot_be_relaxed(self):
        saved = study._evidence_value(self.record())
        self.assertTrue(study._limit_checks(self.model, self.fixture[1], saved))
        changed = copy.deepcopy(saved)
        changed["samples"][0]["hands"]["LEFT_HAND"]["capacity"] = 1700.
        self.assertFalse(study._limit_checks(self.model, self.fixture[1], changed))
        changed = copy.deepcopy(saved)
        changed["samples"][0]["command"] = {"limits_Nm": [999.] * self.model.nu}
        self.assertFalse(study._limit_checks(self.model, self.fixture[1], changed))

    def test_fixture_metadata_reconstruction_is_parameter_bound_without_native_steps(self):
        fixture = (self.model, self.data, self.scene, self.profile, self.seed, self.inputs)
        with mock.patch.object(study.envelope, "make_envelope_fixture", return_value=fixture) as build:
            model, inputs, native = study._fixture_metadata(self.profile, "nominal", .002)
        build.assert_called_once_with(self.profile, "nominal", .002)
        self.assertIs(model, self.model)
        self.assertEqual(native, self.inputs)
        self.assertEqual(inputs["compiled_model_sha256"], study._model_hash(self.model))
        self.assertEqual(inputs["input_sha256"], study._hash(native))
        self.assertEqual(inputs["expected_hand_capacity_N"]["right_hand"], 850.)
        changed = {**self.inputs, "seed_qpos": [999.]}
        with mock.patch.object(study.envelope, "make_envelope_fixture", return_value=(*fixture[:-1], changed)):
            with self.assertRaisesRegex(ValueError, "seed/timestep"):
                study._fixture_metadata(self.profile, "nominal", .002)

    def test_combined_presets_are_rejected_by_parameters_not_display_names(self):
        for profile in (replace(self.profile, name="looks_like_baseline", strength_scale=2.),
                        replace(self.profile, rom_scale=.9),
                        replace(study.envelope.study_profiles()["longer"], grip_capacity=425.),
                        replace(self.profile, upper_arm_length=self.profile.upper_arm_length * 1.05)):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "one-factor"):
                study._fixture_metadata(profile, "nominal", .002)

    def synthetic_summary(self, suite, name, case, dt):
        """Invented report inputs test aggregators, never saved native certification."""
        profile = study.envelope.study_profiles()[name]
        geometry = tuple(getattr(profile, key) for key in study.LENGTHS)
        if not hasattr(self, "_aggregate_metrics"):
            record = study._evidence_value(self.record())
            self._aggregate_metrics = study._metrics(self.model, record, study.audit_motion(self.model, record))
        metrics = copy.deepcopy(self._aggregate_metrics)
        metrics["com"]["final_world_m"][0] += profile.thigh_length - self.profile.thigh_length
        metrics["preparation_path_sha256"] = study._hash(geometry)
        ratio = profile.grip_capacity / 850.
        for hand in metrics["support_loads"]["hands"].values():
            for key in ("min_N", "max_N", "observation_mean_N"):
                hand["capacity"][key] *= ratio
        return {"name": f"{name}_{case}_{dt * 1000:g}ms", "suite": suite, "profile_name": name,
                "case": case, "dt_s": dt, "profile": profile.to_dict(), "classification": "PHYSICAL_SUCCESS",
                "status": "SUCCESS", "reason": "Synthetic aggregate unit only", "geometric_feasible": True,
                "physical_success": True, "accepted": True, "finite_episode": True,
                "steps": 2, "duration_s": 2 * dt, "audit_error": None, "valid_source": True,
                "seed_qpos_sha256": study._hash(geometry), "metrics": metrics,
                "fixture_fingerprints": {"compiled_geometry_sha256": study._hash(geometry),
                    "scene_sha256": "fixed-scene-for-" + case, "compiled_rom_rad": {}, "motor_ceiling_Nm": [90.],
                    "mass_kg": 78.3, "target_offset_world_m": [0., 0., 0.]},
                "evidence_json": "unit-only", "evidence_sha256": "unit-only"}

    def test_comparisons_measure_geometry_response_capacity_and_timestep_disagreements(self):
        runs = [self.synthetic_summary("matrix", name, "nominal", dt)
                for name in study.envelope.study_profiles() for dt in (.002, .001)]
        comparisons = study._comparisons(runs)
        self.assertEqual(len(comparisons["timestep_pairs"]), 4)
        self.assertTrue(all(pair["consistent"] for pair in comparisons["timestep_pairs"]))
        self.assertTrue(all(pair["consistent"] for pair in comparisons["fixed_scene_design"]))
        self.assertTrue(any(pair["observable_response"] for pair in comparisons["morphology_responses"]))
        self.assertTrue(all(pair["capacity_scales"] for pair in comparisons["grip_capacity_comparisons"]))
        self.assertEqual(len(comparisons["grip_capacity_comparisons"]), 2)
        runs[1].update(classification="DYNAMIC_CONTACT_INFEASIBLE", physical_success=False)
        self.assertFalse(study._comparisons(runs)["timestep_pairs"][0]["consistent"])
        runs[0].update(classification="ROM_LIMITED_SEARCH", status="ROM_LIMITED_SEARCH", physical_success=False,
                       geometric_feasible=None, reason="bounded local search")
        runs[1].update(classification="ROM_LIMITED_SEARCH", status="ROM_LIMITED_SEARCH", physical_success=False,
                       geometric_feasible=None, reason="different evidence")
        self.assertFalse(study._comparisons(runs)["timestep_pairs"][0]["consistent"])
        runs[1]["reason"] = runs[0]["reason"]
        self.assertTrue(study._comparisons(runs)["timestep_pairs"][0]["consistent"])

    def test_saved_pose_teleport_and_invented_safe_speed_are_rejected(self):
        record = self.record()
        record["samples"][0]["qpos"][0] += .01
        self.assertFalse(self.verdict(record))
        record = study._evidence_value(self.record())
        record["samples"][0]["qvel"][0] = 1.
        self.assertFalse(study._limit_checks(self.model, self.fixture[1], record))

    def test_full_matrix_requires_successes_from_distinct_geometries_and_observable_response(self):
        for mode in ("good", "bounded_shorter_failure", "all_source_failed", "only_one_geometry", "identical_response", "dt_regression"):
            def run(suite, name, case, dt, *unused):
                summary = self.synthetic_summary(suite, name, case, dt)
                if mode == "all_source_failed":
                    summary.update(physical_success=False, accepted=False, valid_source=False,
                                   classification="SOURCE_STATIC_FAILURE")
                elif mode == "bounded_shorter_failure" and name == "shorter":
                    summary.update(physical_success=False, classification="ROM_LIMITED_SEARCH", status="ROM_LIMITED_SEARCH",
                                   geometric_feasible=None, steps=0, duration_s=0.)
                elif mode == "only_one_geometry" and name in ("shorter", "longer"):
                    summary.update(physical_success=False, classification="ROM_LIMITED_SEARCH")
                elif mode == "identical_response":
                    summary["metrics"]["com"]["final_world_m"][0] = 0.
                elif mode == "dt_regression" and dt == .001:
                    summary.update(physical_success=False, accepted=False, classification="DYNAMIC_CONTACT_INFEASIBLE")
                return summary
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                    mock.patch.object(study, "_run_case", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                code = study.main(["--suite", "matrix", "--summary", "--output", directory])
                report = json.loads((Path(directory) / "report.json").read_text())
                self.assertTrue(report["complete_primary_matrix"])
                self.assertFalse(report["partial_scope"])
                self.assertEqual(code, 0 if mode in ("good", "bounded_shorter_failure") else 1)
                if mode == "only_one_geometry":
                    self.assertIn("INSUFFICIENT_DISTINCT_GEOMETRY_PHYSICAL_SUCCESSES", report["flags"])
                if mode == "identical_response":
                    self.assertIn("NO_MEASURED_MORPHOLOGY_RESPONSE", report["flags"])
                if mode == "dt_regression":
                    self.assertIn("TIMESTEP_QUALITATIVE_DISAGREEMENT", report["flags"])

    def test_unexpected_execution_exception_and_source_edits_cannot_be_accepted(self):
        for mode in ("exception", "source_change"):
            provenance = {"modules": {}, "validator_sha256": "same", "audit_dependencies": {}}
            current = {**provenance, "validator_sha256": "changed"} if mode == "source_change" else provenance
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                    mock.patch.object(study, "_provenance", side_effect=[provenance, current]), \
                    mock.patch.object(study, "_run_case", **({"side_effect": RuntimeError("unexpected native exception")}
                        if mode == "exception" else {"return_value": self.synthetic_summary("matrix", "baseline", "nominal", .002)})), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(study.main(["--profile", "baseline", "--case", "nominal", "--dt", ".002",
                                             "--summary", "--output", directory]), 1)
                report = json.loads((Path(directory) / "report.json").read_text())
                self.assertFalse(report["acceptance"])
                if mode == "exception":
                    self.assertEqual(report["runs"], [])
                    self.assertIn("unexpected native exception", report["errors"][0]["error"])
                else:
                    self.assertIn("SOURCE_HASHES_CHANGED", report["flags"])
    def test_cli_partial_scope_preserves_evidence_and_hashes(self):
        record = self.record()
        record["underlying_result"] = copy.deepcopy(record)
        original = copy.deepcopy(record)
        with tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                mock.patch.object(study, "_execute_case", return_value=record) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(study.main(["--profile", "baseline", "--case", "nominal", "--dt", ".002",
                                         "--summary", "--output", directory]), 0)
            execute.assert_called_once_with(self.profile, "nominal", .002)
            path = Path(directory) / "baseline_nominal_2ms.json"
            saved = json.loads(path.read_text())
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertTrue(report["partial_scope"])
            self.assertFalse(report["complete_primary_matrix"])
            self.assertEqual(saved["samples"], study._evidence_value(record["samples"]))
            self.assertEqual(saved["assessment"], study._evidence_value(record["assessment"]))
            self.assertNotIn("underlying_result", saved)
            self.assertEqual(report["runs"][0]["evidence_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIsNone(saved["motion_audit"]["quality_verdict"])
        self.assertEqual(original, record)

    def test_nonfinite_native_numbers_are_flagged_before_null_conversion(self):
        record = self.record()
        record["assessment"]["diagnostics"]["bad"] = np.nan
        with tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                mock.patch.object(study, "_execute_case", return_value=record):
            summary = study._run_case("matrix", "baseline", "nominal", .002, Path(directory), {}, self.profile)
            self.assertFalse(summary["accepted"])
            self.assertFalse(summary["finite_episode"])
            self.assertIn("Nonfinite original", summary["audit_error"])
            saved = json.loads((Path(directory) / "baseline_nominal_2ms.json").read_text())
            self.assertIsNone(saved["assessment"]["diagnostics"]["bad"])

    def test_native_metadata_drift_and_successful_counterexample_cannot_pass(self):
        for mode in ("metadata_drift", "successful_negative"):
            record = self.record()
            if mode == "metadata_drift":
                record["fixture_inputs"]["model_xml_sha256"] = "different"
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=study.ROOT / "outputs") as directory, \
                    mock.patch.object(study, "_fixture_metadata", return_value=self.fixture), \
                    mock.patch.object(study, "_execute_case", return_value=record):
                summary = study._run_case("negative" if mode == "successful_negative" else "matrix",
                                         "baseline", "nominal", .002, Path(directory), {}, self.profile)
                self.assertFalse(summary["accepted"])
                if mode == "metadata_drift":
                    self.assertIn("differs from reconstructed", summary["audit_error"])
                    self.assertFalse(summary["physical_success"])

    def test_output_and_selector_rejections_precede_native_execution(self):
        options = [["--suite", "negative", "--profile", "longer"], ["--suite", "matrix", "--case", "blocked_path"],
                   ["--suite", "negative", "--profile", "baseline", "--profile", "longer"],
                   ["--profile", "longer", "--case", "up", "--case", "blocked_path"],
                   ["--output", str(study.ROOT / "scripts")], ["--output", str(study.ROOT / "outputs")],
                   ["--output", str(study.ROOT / "outputs" / "stage5.2-baseline-future")],
                   ["--output", str(study.ROOT / "outputs" / "any-preserved-future")]]
        for arguments in options:
            with self.subTest(arguments=arguments), mock.patch.object(study, "_execute_case") as execute, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                study.main(arguments)
            self.assertEqual(caught.exception.code, 2)
            execute.assert_not_called()

    def test_provenance_covers_entire_native_package_and_audit_dependencies(self):
        provenance = study._provenance(["--summary"])
        self.assertIn("boulder_v1.morphology_envelope", provenance["modules"])
        self.assertIn("boulder_v1.transfer_feasibility", provenance["modules"])
        self.assertEqual(provenance["command"][1], str(study.ROOT / "scripts" / "validate_morphology_envelope.py"))
        for info in provenance["modules"].values():
            self.assertEqual(info["sha256"], hashlib.sha256(info["file"].read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
