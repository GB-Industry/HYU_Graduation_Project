"""Saved-state viewer fidelity; synthetic records do not certify new dynamics."""
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1 import ascending_sequence, contact_ik, runtime
from scripts import view_ascent as view
from test_validate_ascent import synthetic_fixture, synthetic_record


def packet(name="baseline", dt=.1):
    fixture = synthetic_fixture(name, dt)
    result = synthetic_record(fixture)
    # Complete native-point schema for the parser, not invented physical evidence.
    for move in result["moves"]:
        rows = [move["initial_state"], *move["samples"], move["final_state"]]
        for row in rows:
            for foot in row.get("feet", row.get("foot_states")).values():
                for contact in foot["contacts"]:
                    contact["point"] = [0., 0., .2]
    return fixture, result


class ViewAscentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.evidence = packet()
        cls.model = cls.fixture[0]

    def setUp(self):
        stack = self.enterContext(ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData", "mj_integratePos"):
            stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError("Forbidden " + name)))
        for module, name in ((ascending_sequence, "run_ascending_sequence"), (ascending_sequence, "make_ascending_fixture"),
                             (ascending_sequence, "execute_transfer"), (runtime, "compute_pose_control"),
                             (contact_ik, "solve_contact_pose")):
            stack.enter_context(patch.object(module, name, side_effect=AssertionError("Forbidden " + name)))

    def recording(self, evidence=None, model=None):
        return view.recording_from_evidence(model or self.model, self.evidence if evidence is None else evidence,
                                            Path("synthetic.json"), "unit-only")

    def test_first_pose_velocity_control_and_constraints_are_recorded_not_standing(self):
        r = self.recording()
        display = copy.copy(self.model)
        data = mujoco.MjData(display)
        self.assertFalse(np.array_equal(r.qpos[0], self.model.qpos0))
        view.apply_frame(display, data, r, 0)
        source = self.evidence["initial_state"]
        for name in ("qpos", "qvel", "ctrl", "eq_active"):
            np.testing.assert_array_equal(getattr(data, name), source[name])
        self.assertEqual(data.time, source["time"])
        self.assertGreater(np.count_nonzero(data.eq_active), 0)

    def test_literal_native_state_fidelity_all_frames_no_interpolation_or_physics(self):
        r = self.recording()
        data = mujoco.MjData(r.model)
        before = copy.deepcopy(self.evidence)
        for index in range(len(r.times)):
            view.apply_frame(r.model, data, r, index)
            for name in ("qpos", "qvel", "ctrl", "eq_active"):
                np.testing.assert_array_equal(getattr(data, name), getattr(r, name)[index])
            self.assertEqual(data.time, r.times[index])
        self.assertEqual(self.evidence, before)
        for array in (r.times, r.qpos, r.qvel, r.ctrl, r.eq_active):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 999.

    def test_capture_activation_and_foot_contact_changes_follow_each_saved_mask(self):
        r = self.recording()
        data = mujoco.MjData(r.model)
        found = {"released": False, "captured": False, "foot": False}
        for index, info in enumerate(r.frames):
            view.apply_frame(r.model, data, r, index)
            contacts = dict(info.contacts)
            active = {r.model.equality(i).name for i in np.flatnonzero(data.eq_active)}
            self.assertEqual(active, {f"grasp_{limb.lower()}_{hold}" for limb, hold in contacts.items() if limb.endswith("HAND")})
            found["released"] |= "RIGHT_HAND" not in contacts
            found["captured"] |= contacts.get("RIGHT_HAND") == "reach_target"
            found["foot"] |= contacts.get("LEFT_FOOT") == "foot_target"
        self.assertTrue(all(found.values()))

    def test_exact_move_boundaries_and_broken_warmstart_are_rejected(self):
        r = self.recording()
        for move in self.evidence["moves"][1:]:
            indices = np.flatnonzero(r.times == move["initial_state"]["time"])
            for key in ("qpos", "qvel", "ctrl", "eq_active"):
                for index in indices:
                    np.testing.assert_array_equal(getattr(r, key)[index], move["initial_state"][key])
        for key in ("qpos", "qvel", "ctrl", "eq_active", "qacc_warmstart"):
            e = copy.deepcopy(self.evidence)
            e["moves"][1]["initial_state"] = copy.deepcopy(e["moves"][1]["initial_state"])
            e["moves"][1]["initial_state"][key][0] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "boundary"):
                self.recording(e)
        e = copy.deepcopy(self.evidence)
        e["moves"][1]["initial_integration_state"] = e["moves"][1]["initial_integration_state"].copy()
        e["moves"][1]["initial_integration_state"][0] += 1
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.recording(e)

    def test_missing_native_fields_nonfinite_masks_and_reset_are_not_repaired(self):
        for key in ("qpos", "qvel", "ctrl", "eq_active", "contacts", "hands", "feet", "time_s"):
            e = copy.deepcopy(self.evidence)
            del e["moves"][0]["samples"][0][key]
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                self.recording(e)
        for mode in ("nonfinite", "mask", "identity", "clock", "steps", "points", "quaternion"):
            e = copy.deepcopy(self.evidence)
            row = e["moves"][0]["samples"][0]
            if mode == "nonfinite":
                row["qvel"][0] = np.nan
            elif mode == "mask":
                row["eq_active"][0] = .5
            elif mode == "identity":
                row["hands"]["LEFT_HAND"]["region_id"] = "wrong"
            elif mode == "clock":
                row["time_s"] = 0.
            elif mode == "steps":
                row["steps"] = 99
            elif mode == "points":
                del row["feet"]["LEFT_FOOT"]["contacts"][0]["point"]
            else:
                row["qpos"][3] = 2.
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.recording(e)

    def test_loader_verifies_bank_hash_model_profile_and_never_calls_fixture_runner(self):
        for name in ("baseline", "longer"):
            fixture, result = packet(name, .002)
            result = copy.deepcopy(result)
            from scripts.validate_ascent import _reference_identity
            result.update(fixture_inputs=fixture[4], validation={"physical_success": True, "accepted": True},
                          reference_identity=_reference_identity(result), provenance={"modules": {
                "boulder_v1." + path.stem: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (view.ROOT / "src" / "boulder_v1").glob("*.py")}})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{name}_fast_2ms.json"
                path.write_text(json.dumps(result))
                report = {"passed": True, "authority_acceptance": True, "demo_timing": "fast", "runs": [{
                    "name": path.stem, "accepted": True, "physical_success": True,
                    "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
                (Path(directory) / "report.json").write_text(json.dumps(report))
                before = path.read_bytes()
                # Synthetic rows are schema exercises; do not call them certified dynamics.
                with patch.object(view, "audit_motion", return_value={}), patch.object(view, "_case_verdict", return_value=True):
                    r = view.load_recording(directory, name)
                self.assertEqual(r.profile_name, name)
                self.assertEqual(r.model.nq, fixture[0].nq)
                np.testing.assert_array_equal(r.model.body_pos, fixture[0].body_pos)
                np.testing.assert_array_equal(r.model.eq_data, fixture[0].eq_data)
                self.assertEqual(path.read_bytes(), before)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    view.load_recording(directory, "longer" if name == "baseline" else "baseline")
                report["runs"][0]["evidence_sha256"] = "0" * 64
                (Path(directory) / "report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    view.load_recording(directory, name)

    def test_play_pause_restart_speed_and_native_step_use_monotonic_time(self):
        r = self.recording()
        p = view.Playback(r, now=0.)
        self.assertFalse(p.playing)
        p.advance(1.)
        self.assertEqual(p.index, 0)
        p.key(32, 1.)
        p.advance(2.)
        self.assertAlmostEqual(p.cursor, r.times[0] + 1.)
        p.key(49, 2.)
        p.advance(4.)
        self.assertAlmostEqual(p.cursor, r.times[0] + 2.)
        p.key(51, 4.)
        p.advance(5.)
        self.assertAlmostEqual(p.cursor, r.times[0] + 4.)
        p.key(32, 5.)
        saved = p.cursor
        p.advance(6.)
        self.assertEqual(p.cursor, saved)
        p.key(82, 6.)
        self.assertEqual(p.index, 0)
        self.assertFalse(p.playing)
        p.key(262, 6.)
        self.assertEqual(p.cursor, r.times[1])
        p.key(263, 6.)
        self.assertEqual(p.index, 0)
        with self.assertRaises(ValueError):
            p.advance(5.)
        p.key(32, 6.)
        p.advance(10000.)
        self.assertEqual(p.index, len(r.times) - 1)
        self.assertFalse(p.playing)

    def test_time_selection_is_preceding_native_pose_not_interpolated(self):
        r = self.recording()
        p = view.Playback(r, autoplay=True, now=0.)
        p.advance(.05)
        self.assertEqual(p.index, 0)  # Unit record has 0.1 s native intervals.
        self.assertAlmostEqual(p.cursor, r.times[0] + .05)
        p.advance(.1)
        self.assertEqual(r.times[p.index], r.times[1])
        for move in self.evidence["moves"][1:]:
            p.advance(move["initial_state"]["time"] - r.times[0])
            self.assertEqual(r.frames[p.index].move_index, self.evidence["moves"].index(move))

    def test_passive_window_opens_on_climbing_pose_and_gui_mutates_only_owned_copy(self):
        r = self.recording()
        original_q = r.qpos.copy()
        original_mass = r.model.body_mass.copy()
        captures, clock_value = [], [0.]
        window = SimpleNamespace(cam=mujoco.MjvCamera(), opt=mujoco.MjvOption(),
                                 user_scn=mujoco.MjvScene(r.model, maxgeom=100))
        window.lock = nullcontext
        window.set_texts = lambda texts: captures.append(texts[0][2])
        iterations = [0]

        def running():
            iterations[0] += 1
            return iterations[0] <= 3

        window.is_running = running

        def launch(model, data, key_callback):
            self.assertIsNot(model, r.model)
            np.testing.assert_array_equal(data.qpos, r.qpos[0])
            np.testing.assert_array_equal(data.eq_active, r.eq_active[0])
            self.assertEqual(data.time, r.times[0])

            def sync(state_only):
                self.assertTrue(state_only)
                if iterations[0] > 1:
                    self.assertEqual(window.cam.azimuth, 42.)
                window.cam.azimuth = 42.
                data.qpos[:] = 999.
                data.ctrl[:] = 999.
                data.eq_active[:] = False
                data.xfrc_applied[:] = 999.
                model.body_mass[:] = 999.

            window.sync = sync
            return nullcontext(window)

        with redirect_stdout(io.StringIO()):
            view.run_viewer(r, launch=launch, clock=lambda: clock_value[0], sleep=lambda _: clock_value.__setitem__(0, clock_value[0] + .01))
        np.testing.assert_array_equal(r.qpos, original_q)
        np.testing.assert_array_equal(r.model.body_mass, original_mass)
        self.assertTrue(all("Native 2.000 s" in text for text in captures))
        self.assertTrue(window.user_scn.ngeom)

    def test_headless_check_and_gui_failure_do_not_retry_or_run_physics(self):
        with patch.object(view, "load_recording", return_value=self.recording()), patch.object(view, "run_viewer") as run, redirect_stdout(io.StringIO()):
            self.assertEqual(view.main(["--check"]), 0)
            run.assert_not_called()
        error = RuntimeError("GUI unavailable")
        with patch.object(view, "load_recording", return_value=self.recording()), \
                patch.object(view, "run_viewer", side_effect=error) as run, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(view.main([]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("No legacy pose fallback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
