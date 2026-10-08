from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from boulder_v1 import ContactMode, Limb, TransitionRequest, TransitionStatus
from boulder_v1.runtime import mujoco_available
from scripts import render_demo

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MUJOCO_GL": "egl"}


def _integration_state(model, data):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def _mixed_scene():
    from boulder_v1.contact_benchmarks import make_mixed_fixture

    fixture = make_mixed_fixture()
    scene = replace(fixture.scene, start_configuration={
        Limb.LEFT_HAND: "left_hand", Limb.RIGHT_HAND: "right_hand",
        Limb.LEFT_FOOT: "left_foot", Limb.RIGHT_FOOT: "right_foot",
    })
    return scene, fixture.reference.copy()


@unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
class RenderDemoTests(unittest.TestCase):
    def test_physical_initialization_failure_preserves_actual_neutral_endpoint(self):
        import mujoco

        initialize = render_demo.initialize_episode
        render_frame = render_demo._render_frame
        for render in (render_demo.render_single_limb_reach_transition, render_demo.render_transition_sequence):
            with self.subTest(render=render.__name__), tempfile.TemporaryDirectory() as tmpdir:
                live = []

                def reject(model, data, *args, **kwargs):
                    live.append((model, data, _integration_state(model, data), model.eq_data.copy()))
                    return initialize(model, data, *args, **kwargs)

                def observe(mj, model, data, scratch, renderer, camera):
                    self.assertIs(data, live[0][1])
                    self.assertIsNot(data, scratch)
                    image = render_frame(mj, model, data, scratch, renderer, camera)
                    np.testing.assert_array_equal(_integration_state(model, data), live[0][2])
                    np.testing.assert_array_equal(model.eq_data, live[0][3])
                    return image

                with mock.patch.object(render_demo, "initialize_episode", side_effect=reject) as init, \
                        mock.patch.object(render_demo, "execute_transition_sequence") as execute, \
                        mock.patch.object(mujoco, "mj_resetData", wraps=mujoco.mj_resetData) as reset, \
                        mock.patch.object(render_demo, "_render_frame", side_effect=observe):
                    result = render("base", Path(tmpdir), width=320, height=180, save_keyframes=False)
                execute.assert_not_called()
                reset.assert_not_called()
                self.assertFalse(init.call_args.kwargs["attach_feet"])
                self.assertIsNone(init.call_args.kwargs["initial_qpos"])
                self.assertIsNone(result["physics_result"])
                self.assertFalse(result["success"])
                self.assertFalse(result["physical_success"])
                self.assertFalse(result["reattached"])
                self.assertFalse(result["eligibility_detected"])
                self.assertEqual(result["status"], "INITIALIZATION_FAILURE")
                self.assertEqual(result["frame_count"], 1)
                manifest = json.loads(result["manifest"].read_text())
                self.assertEqual(manifest, result["metadata"])
                self.assertEqual(manifest["contact_mode"], "physical")
                self.assertIn("capture rejected: gap=", manifest["initialization_error"])
                self.assertIsNone(manifest["physics_result"])
                endpoint = manifest["endpoint"]
                self.assertEqual(endpoint["status_label"], "FAILURE/INITIALIZATION_FAILURE")
                self.assertEqual((endpoint["source_hold"], endpoint["target_hold"]), ("H4", "H5"))
                self.assertEqual(endpoint["contact_configuration"], {})
                self.assertEqual(endpoint["active_attachments"], {})
                self.assertEqual(endpoint["capture_events"], [])
                self.assertEqual(endpoint["release_events"], [])
                self.assertEqual(endpoint["steps"], 0)
                self.assertEqual(endpoint["time"], 0)
                self.assertIsNone(endpoint["phase"])
                self.assertTrue(endpoint["finite"])
                self.assertTrue(endpoint["terminal"])
                self.assertEqual(endpoint["contact_mode_numeric"], 0)
                self.assertTrue(endpoint["scientific_contacts"])
                self.assertFalse(endpoint["physics_certified"])
                np.testing.assert_array_equal(endpoint["qpos"], live[0][0].qpos0)
                np.testing.assert_array_equal(endpoint["qvel"], np.zeros(live[0][0].nv))
                self.assertFalse(any(endpoint["eq_active"]))
                self.assertFalse(any("foot" in live[0][0].equality(i).name for i in range(live[0][0].neq)))
                for hand in endpoint["hand_states"].values():
                    self.assertFalse(hand["active"])
                    self.assertIsNone(hand["region_id"])
                    self.assertIsNone(hand["capacity"])
                for foot in endpoint["foot_states"].values():
                    self.assertEqual(foot["status"], "FOOT_NO_CONTACT")
                    self.assertFalse(foot["contacting"])
                    self.assertFalse(foot["supporting"])
                    self.assertEqual(foot["contacts"], [])
                self.assertEqual(manifest["frames"], [endpoint])
                self.assertEqual(result["keyframes"], {})
                self.assertIsNone(result["montage"])
                with Image.open(result["endpoint"]) as image:
                    self.assertEqual(image.size, (320, 180))

    def test_static_failure_is_observation_without_physics_certificate(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "simulate_static_stance") as simulate:
            path = render_demo.render_static_stance("base", Path(tmpdir), width=320, height=180)
            manifest = json.loads(path.with_suffix(".json").read_text())
        simulate.assert_not_called()
        self.assertIsNone(manifest["physics_result"])
        self.assertFalse(manifest["physics_certified"])
        self.assertFalse(manifest["physical_success"])
        self.assertEqual(manifest["observation"]["status_label"], "FAILURE/INITIALIZATION_FAILURE")
        self.assertEqual(manifest["observation"]["time"], 0)
        self.assertEqual(manifest["observation"]["active_attachments"], {})

    def test_explicit_physical_pose_static_observer_matches_native_simulator(self):
        import mujoco

        scene, reference = _mixed_scene()
        profile = render_demo.PROFILES["base"]
        model = render_demo.compile_model(render_demo.build_mjcf(scene, profile))
        data = mujoco.MjData(model)
        manager = render_demo.initialize_episode(model, data, scene, profile=profile, initial_qpos=reference)
        baseline = render_demo.simulate_static_stance(model, data, scene, profile, steps=40, manager=manager)
        state = render_demo.get_state_summary(model, data, manager)
        render_frame = render_demo._render_frame

        def observe(mj, live_model, live_data, scratch, renderer, camera):
            before = _integration_state(live_model, live_data)
            np.testing.assert_array_equal(before, _integration_state(model, data))
            image = render_frame(mj, live_model, live_data, scratch, renderer, camera)
            np.testing.assert_array_equal(_integration_state(live_model, live_data), before)
            return image

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "make_synthetic_scene", return_value=scene), \
                mock.patch.object(render_demo, "_render_frame", side_effect=observe), \
                mock.patch.object(render_demo, "initialize_episode", wraps=render_demo.initialize_episode) as init:
            path = render_demo.render_static_stance("base", Path(tmpdir), width=320, height=180,
                                                   initial_qpos=reference)
            manifest = json.loads(path.with_suffix(".json").read_text())
        np.testing.assert_array_equal(init.call_args.kwargs["initial_qpos"], reference)
        self.assertFalse(init.call_args.kwargs["attach_feet"])
        self.assertIsNone(manifest["initialization_error"])
        self.assertEqual(manifest["physics_result"], render_demo._json_value(baseline))
        self.assertEqual(manifest["observation"]["hand_states"], render_demo._json_value(state.hand_states))
        self.assertEqual(manifest["observation"]["foot_states"], render_demo._json_value(state.foot_states))
        self.assertEqual(manifest["observation"]["status_label"], "STATIC OBSERVATION")
        self.assertIn("NO PHYSICS CERTIFICATION", manifest["observation"]["reason"])

    def test_authoritative_observation_not_duplicate_manager_queries(self):
        executor = render_demo.execute_transition_sequence
        observations = []

        def execute(**kwargs):
            callback = kwargs["frame_callback"]

            def observe(observation, data, manager):
                manager.contact_configuration = mock.Mock(side_effect=AssertionError("duplicate contact epoch"))
                manager.active_attachments = mock.Mock(side_effect=AssertionError("duplicate attachments"))
                callback(observation, data, manager)
                observations.append(observation)
            kwargs["frame_callback"] = observe
            return executor(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "execute_transition_sequence", side_effect=execute):
            result = render_demo.render_single_limb_reach_transition(
                "base", Path(tmpdir), steps=35, width=320, height=180,
                contact_mode="idealized_debug")
        self.assertEqual(result["status"], TransitionStatus.INCOMPLETE)
        self.assertEqual(result["physics_result"].total_steps, 35)
        self.assertEqual(result["frame_count"], 37)
        self.assertFalse(result["physical_success"])
        self.assertIn("NONPHYSICAL!", result["metadata"]["contact_mode_label"])
        self.assertEqual(result["metadata"]["requests"][0]["max_attach_distance"], .15)
        for frame, observation in zip(result["metadata"]["frames"], observations, strict=True):
            state = render_demo._json_value(observation.state)
            for field in ("hand_states", "foot_states", "contact_configuration", "attachment_loads",
                          "capture_events", "release_events", "qpos", "qvel", "qacc_warmstart"):
                self.assertEqual(frame[field], state[field])
            self.assertEqual(frame["contact_mode"], "idealized_debug")
            self.assertEqual(frame["contact_mode_numeric"], 1)
            self.assertFalse(frame["scientific_contacts"])
        release = result["metadata"]["keyframes"]["transition_02_release.png"]
        self.assertFalse(release["hand_states"][Limb.RIGHT_HAND.value]["active"])
        self.assertNotIn(Limb.RIGHT_HAND.value, release["active_attachments"])

    def test_debug_full_sequence_reports_actual_outcome_not_assumed_success(self):
        import mujoco

        profile = render_demo.PROFILES["long_reach_lower_grip"]
        scene = render_demo.make_synthetic_scene()
        model = render_demo.compile_model(render_demo.build_mjcf(scene, profile, ContactMode.IDEALIZED_DEBUG))
        data = mujoco.MjData(model)
        manager = render_demo.initialize_episode(model, data, scene, profile=profile, attach_feet=True)
        requests = render_demo._transition_requests("sequence", 210, ContactMode.IDEALIZED_DEBUG)
        baseline = render_demo.execute_transition_sequence(model, data, scene, profile, requests, manager=manager)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_demo.render_transition_sequence(profile.name, Path(tmpdir), width=320, height=180,
                                                            contact_mode="idealized_debug")
            self.assertEqual(result["metadata"]["physics_result"], render_demo._json_value(baseline))
            self.assertEqual(result["success"], baseline.success)
            last = baseline.transition_results[-1]
            self.assertEqual(result["metadata"]["endpoint"]["status"], last.status.value)
            self.assertEqual(result["metadata"]["endpoint"]["reason"], last.reason)
            self.assertEqual(result["metadata"]["endpoint"]["foot_states"],
                             render_demo._json_value(last.final_state.foot_states))
            self.assertFalse(result["physical_success"])
            self.assertTrue(result["endpoint"].exists())
            self.assertTrue(result["montage"].exists())

    def test_physical_missing_source_is_not_initialization_failure(self):
        scene, reference = _mixed_scene()
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "make_synthetic_scene", return_value=scene):
            result = render_demo.render_single_limb_reach_transition(
                "base", Path(tmpdir), from_region_id="left_hand", to_region_id="right_hand",
                width=320, height=180, initial_qpos=reference, save_keyframes=False)
        self.assertIsNone(result["initialization_error"])
        self.assertEqual(result["status"], TransitionStatus.SOURCE_NOT_ATTACHED)
        self.assertIsNotNone(result["physics_result"])
        self.assertEqual(result["physics_result"].total_steps, 0)
        self.assertEqual(result["metadata"]["endpoint"]["status_label"], "FAILURE/SOURCE_NOT_ATTACHED")
        self.assertEqual(set(result["metadata"]["endpoint"]["active_attachments"]),
                         {Limb.LEFT_HAND.value, Limb.RIGHT_HAND.value})

    def test_physical_foot_request_does_not_invent_support_from_scene_start(self):
        scene, reference = _mixed_scene()
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "make_synthetic_scene", return_value=scene):
            result = render_demo.render_single_limb_reach_transition(
                "base", Path(tmpdir), limb=Limb.LEFT_FOOT, from_region_id="left_foot", to_region_id="right_foot",
                width=320, height=180, initial_qpos=reference, save_keyframes=False)
        self.assertIsNone(result["initialization_error"])
        self.assertEqual(result["status"], TransitionStatus.SOURCE_NOT_ATTACHED)
        foot = result["metadata"]["endpoint"]["foot_states"][Limb.LEFT_FOOT.value]
        self.assertFalse(foot["supporting"])
        self.assertEqual(foot["normal_force"], 0)
        self.assertNotIn(Limb.LEFT_FOOT.value, result["metadata"]["endpoint"]["contact_configuration"])
        self.assertNotIn(Limb.LEFT_FOOT.value, result["metadata"]["endpoint"]["active_attachments"])

    def test_explicit_physical_transition_matches_baseline_without_live_writes(self):
        import mujoco

        scene, reference = _mixed_scene()
        profile = render_demo.PROFILES["base"]
        model = render_demo.compile_model(render_demo.build_mjcf(scene, profile))
        data = mujoco.MjData(model)
        manager = render_demo.initialize_episode(model, data, scene, profile=profile, initial_qpos=reference)
        requests = [TransitionRequest(Limb.RIGHT_HAND, "right_hand", "left_hand", steps=4)]
        baseline = render_demo.execute_transition_sequence(model, data, scene, profile, requests, manager=manager)
        render_frame = render_demo._render_frame

        def observe(mj, model, data, scratch, renderer, camera):
            before = _integration_state(model, data)
            image = render_frame(mj, model, data, scratch, renderer, camera)
            np.testing.assert_array_equal(_integration_state(model, data), before)
            return image

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "make_synthetic_scene", return_value=scene), \
                mock.patch.object(render_demo, "_render_frame", side_effect=observe):
            result = render_demo._render_transitions("base", Path(tmpdir), requests, 30, 320, 180, False,
                                                      sequence=False, initial_qpos=reference)
        self.assertIsNone(result["initialization_error"])
        self.assertEqual(render_demo._json_value(result["physics_result"]), render_demo._json_value(baseline))
        self.assertEqual(result["metadata"]["endpoint"]["foot_states"],
                         render_demo._json_value(baseline.final_state_summary.foot_states))
        self.assertEqual(result["metadata"]["endpoint"]["hand_states"],
                         render_demo._json_value(baseline.final_state_summary.hand_states))
        self.assertTrue(result["metadata"]["endpoint"]["capture_events"])
        for event in result["metadata"]["endpoint"]["capture_events"]:
            self.assertEqual(event["mode"], "physical")
            self.assertLessEqual(event["gap_m"], .001)
            self.assertLessEqual(event["relative_speed_m_s"], .05)
            self.assertGreaterEqual(event["orientation"], np.cos(np.deg2rad(30)))
            self.assertLess(event["penetration_m"], .001)

    def test_hud_matches_real_mixed_contact_sensor_including_slip_and_separation(self):
        import mujoco
        from boulder_v1.contact_benchmarks import make_mixed_fixture

        fixture = make_mixed_fixture()
        for _ in range(10):
            mujoco.mj_step(fixture.model, fixture.data)
        for case in ("supporting", "slipping", "separated"):
            with self.subTest(case=case):
                if case == "slipping":
                    fixture.data.qvel[0] = .2
                elif case == "separated":
                    fixture.data.qpos[2] += .4
                    fixture.data.qvel[:] = 0
                before = _integration_state(fixture.model, fixture.data)
                state = render_demo.get_state_summary(fixture.model, fixture.data, fixture.manager)
                metadata = render_demo._state_metadata(state, fixture.model, fixture.scene)
                metadata.update(status_label="OBSERVATION", phase=None, steps=0)
                image = Image.new("RGB", (1280, 720))
                draw = mock.Mock(wraps=render_demo.ImageDraw.Draw(image, "RGBA"))
                with mock.patch.object(render_demo.ImageDraw, "Draw", return_value=draw):
                    render_demo._draw_hud(image, "Mixed", "base", metadata)
                texts = [call.args[1] for call in draw.text.call_args_list]
                for line in render_demo._contact_lines(metadata):
                    self.assertIn(line, texts)
                self.assertTrue(any("HAND_GRASP LH: target=left_hand active=True load=" in text for text in texts))
                foot = metadata["foot_states"][Limb.LEFT_FOOT.value]
                if case == "supporting":
                    self.assertTrue(foot["supporting"])
                    self.assertGreater(foot["normal_force"], 5)
                    self.assertEqual(metadata["contact_configuration"][Limb.LEFT_FOOT.value], "left_foot")
                    self.assertNotIn(Limb.LEFT_FOOT.value, metadata["active_attachments"])
                elif case == "slipping":
                    self.assertTrue(foot["slipping"])
                    self.assertFalse(foot["supporting"])
                    self.assertNotIn(Limb.LEFT_FOOT.value, metadata["contact_configuration"])
                elif case == "separated":
                    self.assertFalse(foot["contacting"])
                    self.assertFalse(foot["supporting"])
                    self.assertEqual(foot["normal_force"], 0)
                    self.assertEqual(foot["contacts"], [])
                self.assertTrue(any("FOOT_SUPPORT LF:" in text and "Fn=" in text and "Ft=" in text
                                    and "v=" in text and "support=" in text for text in texts))
                np.testing.assert_array_equal(_integration_state(fixture.model, fixture.data), before)

    def test_hud_contact_fields_and_caveat_fit_real_text_bounds(self):
        from boulder_v1.contact_benchmarks import make_mixed_fixture
        from boulder_v1.support import FootStatus, FootSupportState

        fixture = make_mixed_fixture()
        metadata = render_demo._state_metadata(
            render_demo.get_state_summary(fixture.model, fixture.data, fixture.manager),
            fixture.model, fixture.scene)
        metadata.update(status_label="FAILURE/INITIALIZATION_FAILURE", phase="stabilized_stance", steps=250,
                        limb=Limb.RIGHT_HAND.value, move_index=2, total_moves=3, source_hold="H4", target_hold="H5",
                        reason="Initial contact reference rejected; supply an admissible explicit pose. " * 4)
        metadata["hand_states"][Limb.LEFT_HAND.value].update(load=56.52, capacity=1000., margin=943.48)
        metadata["hand_states"][Limb.RIGHT_HAND.value].update(
            region_id=None, active=False, load=0., capacity=None, margin=None, valid=False)
        metadata["foot_states"][Limb.LEFT_FOOT.value] = render_demo._json_value(
            FootSupportState(status=FootStatus.SLIPPING, contacting=True, slipping=True,
                             normal_force=373.74, tangential_force=32.53, tangential_speed=.2))
        metadata["foot_states"][Limb.RIGHT_FOOT.value] = render_demo._json_value(
            FootSupportState(status=FootStatus.SUPPORTING, contacting=True, supporting=True,
                             normal_force=314.25, tangential_force=10.17, tangential_speed=.001))
        for mode in ContactMode:
            state = json.loads(json.dumps(metadata))
            state.update(contact_mode=mode.value, contact_mode_label=render_demo.contact_mode_label(mode),
                         nonphysical=mode == ContactMode.IDEALIZED_DEBUG)
            if mode == ContactMode.IDEALIZED_DEBUG:
                for foot in state["foot_states"].values():
                    foot.update(render_demo._json_value(
                        FootSupportState(status=FootStatus.IDEALIZED, idealized_attachment="H1")))
            for size in ((320, 180), (1280, 720)):
                with self.subTest(mode=mode, size=size):
                    image = Image.new("RGB", size)
                    real_draw = render_demo.ImageDraw.Draw(image, "RGBA")
                    draw = mock.Mock(wraps=real_draw)
                    with mock.patch.object(render_demo.ImageDraw, "Draw", return_value=draw):
                        self.assertIs(render_demo._draw_hud(
                            image, "Multi-Limb Sequence (No Reset)", "long_reach_lower_grip", state), image)
                    self.assertEqual(image.size, size)
                    texts = [call.args[1] for call in draw.text.call_args_list]
                    previous_bottom = 0
                    for call in draw.text.call_args_list:
                        bounds = real_draw.textbbox(call.args[0], call.args[1], font=call.kwargs["font"])
                        self.assertGreaterEqual(bounds[0], 0, call.args[1])
                        self.assertGreaterEqual(bounds[1], 0, call.args[1])
                        self.assertLessEqual(bounds[2], size[0], call.args[1])
                        self.assertLessEqual(bounds[3], size[1], call.args[1])
                        self.assertGreaterEqual(bounds[3] - bounds[1], 8, call.args[1])
                        self.assertGreaterEqual(bounds[1], previous_bottom, call.args[1])
                        previous_bottom = bounds[3]
                    self.assertIn("Reason: see manifest for full details", texts)
                    caveat = texts[2]
                    if mode == ContactMode.PHYSICAL:
                        self.assertIn("PHYSICAL", caveat)
                        self.assertIn("CONTROLLER NOT VALIDATED", caveat.replace("_", " "))
                    else:
                        self.assertIn("NONPHYSICAL", caveat)
                        self.assertIn("NOT SCIENTIFIC", caveat)
                    if size == (1280, 720):
                        for line in render_demo._contact_lines(state):
                            self.assertIn(line, texts)
                        continue
                    for limb, label in ((Limb.LEFT_HAND, "LH"), (Limb.RIGHT_HAND, "RH")):
                        hand = state["hand_states"][limb]
                        index = next(i for i, text in enumerate(texts) if text.startswith(f"HAND {label}:"))
                        self.assertIn(f"target={hand['region_id'] or 'none'}", texts[index])
                        self.assertIn(f"active={'Y' if hand['active'] else 'N'}", texts[index])
                        self.assertIn(f"valid={'Y' if hand['valid'] else 'N'}", texts[index])
                        for field, key in (("load", "load"), ("cap", "capacity"), ("margin", "margin")):
                            value = f"{hand[key]:.2f}N" if hand[key] is not None else "n/a"
                            self.assertIn(f"{field}={value}", texts[index + 1])
                    for limb, label in ((Limb.LEFT_FOOT, "LF"), (Limb.RIGHT_FOOT, "RF")):
                        foot = state["foot_states"][limb]
                        index = next(i for i, text in enumerate(texts) if text.startswith(f"FOOT {label}:"))
                        self.assertIn("IDEALIZED_DEBUG" if mode == ContactMode.IDEALIZED_DEBUG else
                                      foot["status"].removeprefix("FOOT_"), texts[index])
                        for field, key in (("contact", "contacting"), ("support", "supporting"), ("slip", "slipping")):
                            self.assertIn(f"{field}={'Y' if foot[key] else 'N'}", texts[index])
                        self.assertIn(f"Fn={foot['normal_force']:.2f}N", texts[index + 1])
                        self.assertIn(f"Ft={foot['tangential_force']:.2f}N", texts[index + 1])
                        self.assertIn(f"speed={foot['tangential_speed']:.3f}m/s", texts[index + 1])

    def test_export_failures_keep_initialization_endpoint_and_manifest(self):
        original_save = Image.Image.save

        def save(image, path, *args, **kwargs):
            if Path(path).suffix == ".gif":
                raise RuntimeError("GIF unavailable")
            return original_save(image, path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("imageio.v3.imwrite", side_effect=RuntimeError("MP4 unavailable")), \
                mock.patch.object(Image.Image, "save", autospec=True, side_effect=save):
            result = render_demo.render_single_limb_reach_transition(
                "base", Path(tmpdir), width=320, height=180, save_keyframes=False)
            self.assertTrue(result["endpoint"].exists())
            manifest = json.loads(result["manifest"].read_text())
        self.assertEqual(set(manifest["export_errors"]), {"mp4", "gif"})
        self.assertIsNone(result["gif"])
        self.assertIsNone(result["mp4"])
        self.assertEqual(manifest["endpoint"]["status_label"], "FAILURE/INITIALIZATION_FAILURE")

    def test_nonfinite_endpoint_preserves_actual_state_and_has_no_fabricated_pose(self):
        import mujoco

        initializer = render_demo.initialize_episode
        real_step = mujoco.mj_step
        live = []

        def initialize(*args, **kwargs):
            manager = initializer(*args, **kwargs)
            live.append(manager.data)
            return manager

        def step(model, data):
            real_step(model, data)
            data.qpos[2] = float("nan")

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(render_demo, "initialize_episode", side_effect=initialize), \
                mock.patch.object(mujoco, "mj_step", side_effect=step) as integrate:
            result = render_demo.render_transition_sequence(
                "base", Path(tmpdir), width=320, height=180, contact_mode="idealized_debug", save_keyframes=False)
            endpoint = result["metadata"]["endpoint"]
            with Image.open(result["endpoint"]) as image:
                self.assertEqual(image.getpixel((0, 0)), (10, 14, 22))
            image = Image.new("RGB", (320, 180))
            draw = mock.Mock(wraps=render_demo.ImageDraw.Draw(image, "RGBA"))
            with mock.patch.object(render_demo.ImageDraw, "Draw", return_value=draw):
                render_demo._draw_hud(image, "Failure", "base", endpoint)
        integrate.assert_called_once()
        self.assertEqual(result["status"], TransitionStatus.NONFINITE_STATE)
        self.assertFalse(endpoint["pose_available"])
        self.assertIsNone(endpoint["root_pos"][2])
        self.assertIn("POSE UNAVAILABLE: nonfinite state", [c.args[1] for c in draw.text.call_args_list])
        state = result["physics_result"].final_state_summary
        for field in ("qpos", "qvel", "ctrl", "eq_active", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(state, field), getattr(live[0], field))

    def test_cli_all_physical_init_failures_exit_one_and_generate_every_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/render_demo.py"), "--mode", "all",
                 "--width", "320", "--height", "180", "--no-keyframes", "--output", tmpdir],
                cwd=ROOT, env=ENV, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            self.assertNotIn("[SUCCESS]", process.stdout)
            self.assertIn("FAILURE/INITIALIZATION_FAILURE", process.stdout)
            for name in ("stance_base", "stance_compact_strong", "transition_base", "sequence_base"):
                manifest = json.loads((Path(tmpdir) / f"{name}.json").read_text())
                self.assertEqual(manifest["contact_mode"], "physical")
                self.assertIsNone(manifest["physics_result"])
                self.assertFalse(manifest["physical_success"])
                endpoint = manifest["endpoint"] if name.startswith("stance") else manifest["artifacts"]["endpoint"]
                self.assertTrue(Path(endpoint).exists())

    def test_cli_debug_outcome_is_nonphysical_and_matches_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/render_demo.py"), "--mode", "sequence",
                 "--contact-mode", "idealized_debug", "--width", "320", "--height", "180",
                 "--steps", "35", "--output", tmpdir],
                cwd=ROOT, env=ENV, capture_output=True, text=True, check=False)
            manifest = json.loads((Path(tmpdir) / "sequence_base.json").read_text())
        self.assertEqual(process.returncode, 0 if manifest["success"] else 1, process.stdout + process.stderr)
        self.assertIn("NONPHYSICAL!", process.stdout)
        self.assertFalse(manifest["physical_success"])
        self.assertIsNone(manifest["initialization_error"])
        self.assertEqual(manifest["success"], manifest["physics_result"]["success"])

    def test_run_demo_reports_rejection_without_executor_or_forced_grasps(self):
        from scripts import run_demo

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(sys, "argv", ["run_demo.py", "--output", tmpdir]), \
                mock.patch.object(run_demo, "render_profile_comparison"), \
                mock.patch.object(run_demo, "execute_transition_sequence") as execute:
            self.assertEqual(run_demo.main(), 1)
            manifest = json.loads((Path(tmpdir) / "demo_results.json").read_text())
        execute.assert_not_called()
        for result in manifest["profiles"].values():
            self.assertEqual(result["status"], "INITIALIZATION_FAILURE")
            self.assertEqual(result["contact_mode"], "physical")
            self.assertIsNone(result["physics_result"])
            self.assertEqual(result["actual_state"]["time"], 0)
            self.assertEqual(result["actual_state"]["contact_configuration"], {})
            self.assertFalse(any(result["actual_state"]["eq_active"]))

    def test_renderer_closed_when_observation_raises(self):
        import mujoco

        for render in (render_demo.render_static_stance, render_demo.render_single_limb_reach_transition):
            with self.subTest(render=render.__name__), tempfile.TemporaryDirectory() as tmpdir:
                renderer = mock.Mock()
                with mock.patch.object(mujoco, "Renderer", return_value=renderer), \
                        mock.patch.object(render_demo, "_render_frame", side_effect=RuntimeError("render failed")):
                    with self.assertRaisesRegex(RuntimeError, "render failed"):
                        render("base", Path(tmpdir), width=320, height=180)
                renderer.close.assert_called_once_with()
