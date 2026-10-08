from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
from io import StringIO
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from boulder_v1 import ClimberProfile, ContactMode, Limb, build_mjcf, compile_model, make_synthetic_scene
from boulder_v1.runtime import mujoco_available
from scripts import view_scene


def _integration_state(model, data):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return tuple(state)


class _FakeViewer:
    def __init__(self, close_after=None, overlay=True):
        import mujoco

        self.cam = mujoco.MjvCamera()
        self.model = self.data = self.live_model = self.live_data = self.result = None
        self.closed = False
        self.close_after = close_after
        self.snapshots = []
        self.live_snapshots = []
        self.text_frames = []
        self.frozen_frames = 0
        if overlay:
            self.set_texts = self.text_frames.append

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def is_running(self):
        return not self.closed

    def launch(self, model, data):
        import mujoco

        self.model, self.data = model, data
        mujoco.mj_forward(model, data)
        return self

    def lock(self):
        return nullcontext()

    def sync(self):
        import mujoco

        self.snapshots.append((float(self.data.time), tuple(self.data.qpos), tuple(self.data.qvel),
                               tuple(self.data.eq_active), tuple(self.data.ctrl)))
        if self.live_data is not None:
            self.live_snapshots.append(_integration_state(self.live_model, self.live_data))
        # Deliberately emulate native GUI resets, perturbations, and model edits.
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = 123
        self.data.time += 1
        self.data.qpos[0] += .5
        self.data.qvel[:] = .5
        self.data.ctrl[:] = .25
        self.data.eq_active[:] = False
        self.data.xfrc_applied[:] = 25
        self.model.opt.gravity[2] = -1
        self.model.opt.timestep = .003
        if self.result is not None:
            self.frozen_frames += 1
            if self.frozen_frames >= 3:
                self.closed = True
        if self.close_after is not None and len(self.snapshots) >= self.close_after:
            self.closed = True


@unittest.skipUnless(mujoco_available(), "MuJoCo package unavailable")
class ViewSceneTests(unittest.TestCase):
    def test_physical_default_rejects_all_old_profiles_without_reset_or_execution(self):
        import mujoco

        initializer = view_scene.initialize_episode
        for profile in view_scene.PROFILES:
            for mode in ("stance", "transition", "sequence"):
                with self.subTest(profile=profile, mode=mode):
                    live = []

                    def initialize(model, data, *args, **kwargs):
                        live.append((model, data, _integration_state(model, data)))
                        return initializer(model, data, *args, **kwargs)

                    output = StringIO()
                    with redirect_stdout(output), \
                            patch.object(view_scene, "initialize_episode", side_effect=initialize) as init, \
                            patch.object(view_scene, "execute_transition_sequence") as execute, \
                            patch.object(view_scene, "simulate_static_stance") as static, \
                            patch.object(mujoco, "mj_resetData", wraps=mujoco.mj_resetData) as reset:
                        self.assertFalse(view_scene.validate_scene_headlessly(profile, mode=mode))
                    execute.assert_not_called()
                    static.assert_not_called()
                    reset.assert_not_called()
                    self.assertFalse(init.call_args.kwargs["attach_feet"])
                    model, data, before = live[0]
                    self.assertEqual(_integration_state(model, data), before)
                    np.testing.assert_array_equal(data.qpos, model.qpos0)
                    self.assertEqual(data.time, 0)
                    self.assertFalse(any(data.eq_active))
                    self.assertIn("FAILURE/INITIALIZATION_FAILURE", output.getvalue())
                    self.assertIn("target=none active=False", output.getvalue())
                    self.assertIn("FOOT_NO_CONTACT contact=False support=False", output.getvalue())
                    self.assertIn("CONTROLLER_NOT_VALIDATED", output.getvalue())
                    self.assertNotIn("SOURCE_NOT_ATTACHED", output.getvalue())
                    self.assertNotIn("[SUCCESS]", output.getvalue())

    def test_debug_headless_outcomes_are_reported_without_weakening_criteria(self):
        for mode, steps in (("transition", 35), ("sequence", 210)):
            with self.subTest(mode=mode), redirect_stdout(StringIO()) as output, \
                    patch.object(view_scene, "_report_sequence_result", wraps=view_scene._report_sequence_result) as report, \
                    patch.object(view_scene, "initialize_episode", wraps=view_scene.initialize_episode) as init:
                success = view_scene.validate_scene_headlessly(mode=mode, steps=steps, contact_mode="idealized_debug")
                result = report.call_args.args[0]
                self.assertEqual(success, result.success)
                self.assertTrue(init.call_args.kwargs["attach_feet"])
                self.assertEqual(result.final_state_summary.contact_mode, ContactMode.IDEALIZED_DEBUG)
                self.assertIn("NONPHYSICAL! NOT SCIENTIFIC", output.getvalue())
                self.assertIn(f"status={result.transition_results[-1].status.value}", output.getvalue())
                self.assertIn(result.transition_results[-1].reason, output.getvalue())
                if steps == 35:
                    self.assertFalse(success)
                    self.assertEqual(result.total_steps, 35)
                    self.assertIn("status=INCOMPLETE", output.getvalue())

    def test_request_gates_are_mode_specific(self):
        physical = view_scene._transition_requests("sequence", 50)
        debug = view_scene._transition_requests("sequence", 50, ContactMode.IDEALIZED_DEBUG)
        self.assertEqual([request.max_attach_distance for request in physical], [.001] * 3)
        self.assertEqual([request.max_attach_distance for request in debug], [.15] * 3)
        self.assertEqual([request.steps for request in physical + debug], [50] * 6)
        self.assertTrue(all(request.check_grip for request in physical))

    def test_pose_and_free_remain_explicit_debug_checks(self):
        for mode in ("pose", "free"):
            with self.subTest(mode=mode), redirect_stdout(StringIO()) as output:
                self.assertTrue(view_scene.validate_scene_headlessly(mode=mode, steps=50))
                self.assertIn("NOT transition validation", output.getvalue())

    def test_camera_framing_derived_from_model_stat(self):
        import mujoco

        model = compile_model(build_mjcf(make_synthetic_scene(), ClimberProfile("base")))
        camera = mujoco.MjvCamera()
        view_scene.configure_camera(camera, model)
        self.assertAlmostEqual(camera.distance, float(model.stat.extent * 1.15))
        self.assertEqual(camera.lookat[0], model.stat.center[0])
        self.assertAlmostEqual(camera.lookat[1], model.stat.center[1] - .30)
        self.assertAlmostEqual(camera.lookat[2], model.stat.center[2] * .95)
        self.assertEqual(camera.azimuth, 105)
        self.assertEqual(camera.elevation, -12)

    def test_physical_gui_initialization_failure_freezes_neutral_isolated_data(self):
        import mujoco.viewer

        initializer = view_scene.initialize_episode
        for mode in ("stance", "transition", "sequence"):
            with self.subTest(mode=mode):
                viewer = _FakeViewer(close_after=4)
                original = []

                def initialize(model, data, *args, **kwargs):
                    viewer.live_model, viewer.live_data = model, data
                    original.append((_integration_state(model, data), model.eq_data.copy(), model.opt.gravity.copy()))
                    return initializer(model, data, *args, **kwargs)

                with redirect_stdout(StringIO()) as output, \
                        patch.object(mujoco.viewer, "launch_passive", side_effect=viewer.launch), \
                        patch.object(view_scene, "initialize_episode", side_effect=initialize) as init, \
                        patch.object(view_scene, "execute_transition_sequence") as execute, \
                        patch.object(view_scene, "compute_pose_control") as control, \
                        patch.object(mujoco, "mj_step") as step, patch.object(view_scene.time, "sleep"):
                    self.assertEqual(view_scene.launch_visual_viewer(mode=mode), 1)
                init.assert_called_once()
                execute.assert_not_called()
                control.assert_not_called()
                step.assert_not_called()
                self.assertIsNot(viewer.model, viewer.live_model)
                self.assertIsNot(viewer.data, viewer.live_data)
                self.assertFalse(np.shares_memory(viewer.model.eq_data, viewer.live_model.eq_data))
                self.assertEqual(_integration_state(viewer.live_model, viewer.live_data), original[0][0])
                np.testing.assert_array_equal(viewer.live_model.eq_data, original[0][1])
                np.testing.assert_array_equal(viewer.live_model.opt.gravity, original[0][2])
                self.assertEqual(viewer.snapshots, [viewer.snapshots[0]] * 4)
                self.assertEqual(viewer.live_snapshots, [original[0][0]] * 4)
                self.assertEqual(viewer.snapshots[0][0], 0)
                self.assertFalse(any(viewer.snapshots[0][3]))
                text = viewer.text_frames[-1][0][2]
                self.assertIn("FAILURE/INITIALIZATION_FAILURE", text)
                self.assertIn("Actual time=0.000s", text)
                self.assertIn("HAND_GRASP LH: target=none active=False", text)
                self.assertIn("FOOT_SUPPORT LF: FOOT_NO_CONTACT", text)
                if mode != "stance":
                    self.assertIn("Requested: RIGHT_HAND H4 -> H5", text)
                self.assertIn("Initialization failure frozen", output.getvalue())

    def test_gui_debug_executor_matches_headless_and_freezes_actual_endpoint(self):
        import mujoco.viewer

        for mode, steps in (("transition", 35), ("sequence", 210)):
            with self.subTest(mode=mode):
                viewer = _FakeViewer()
                scene = make_synthetic_scene()
                profile = view_scene.PROFILES["base"]
                model = compile_model(build_mjcf(scene, profile, ContactMode.IDEALIZED_DEBUG))
                data = mujoco.MjData(model)
                initializer = view_scene.initialize_episode
                executor = view_scene.execute_transition_sequence
                manager = initializer(model, data, scene, profile=profile, attach_feet=True)
                baseline = executor(model, data, scene, profile,
                                    view_scene._transition_requests(mode, steps, ContactMode.IDEALIZED_DEBUG),
                                    manager=manager)
                observations = []

                def initialize(model, data, *args, **kwargs):
                    viewer.live_model, viewer.live_data = model, data
                    return initializer(model, data, *args, **kwargs)

                def execute(*args, **kwargs):
                    callback = kwargs["frame_callback"]

                    def observe(observation, observed_data, manager):
                        before = _integration_state(args[0], args[1])
                        callback(observation, observed_data, manager)
                        self.assertEqual(_integration_state(args[0], args[1]), before)
                        observations.append(observation)
                    kwargs["frame_callback"] = observe
                    viewer.result = executor(*args, **kwargs)
                    return viewer.result

                with redirect_stdout(StringIO()) as output, \
                        patch.object(mujoco.viewer, "launch_passive", side_effect=viewer.launch), \
                        patch.object(view_scene, "initialize_episode", side_effect=initialize), \
                        patch.object(view_scene, "execute_transition_sequence", side_effect=execute) as run, \
                        patch.object(view_scene.time, "sleep"), \
                        patch.object(view_scene, "compute_pose_control", side_effect=AssertionError("observer control")):
                    code = view_scene.launch_visual_viewer(mode=mode, steps=steps, contact_mode="idealized_debug")
                run.assert_called_once()
                self.assertEqual(code, 0 if baseline.success else 1)
                self.assertEqual(viewer.result, baseline)
                self.assertEqual(_integration_state(viewer.live_model, viewer.live_data), _integration_state(model, data))
                self.assertEqual(viewer.live_model.opt.timestep, model.opt.timestep)
                np.testing.assert_array_equal(viewer.live_model.opt.gravity, model.opt.gravity)
                self.assertEqual(viewer.frozen_frames, 3)
                self.assertEqual(viewer.snapshots[-4:], [viewer.snapshots[-1]] * 4)
                self.assertEqual(viewer.live_snapshots[-4:], [viewer.live_snapshots[-1]] * 4)
                self.assertIn("Endpoint frozen", output.getvalue())
                last = observations[-1]
                final_text = viewer.text_frames[-1][0][2]
                self.assertIn(f"status={last.status.value}", final_text)
                self.assertIn(f"Reason: {last.result.reason}", final_text)
                self.assertIn("NONPHYSICAL!", final_text)
                for line in view_scene._contact_lines(asdict(last.state)):
                    self.assertIn(line, final_text)

    def test_gui_interruption_has_no_automatic_headless_rerun(self):
        import mujoco.viewer

        for close_after, duration in ((10, None), (None, 0)):
            with self.subTest(close_after=close_after):
                viewer = _FakeViewer(close_after=close_after)
                with redirect_stdout(StringIO()) as output, \
                        patch.object(mujoco.viewer, "launch_passive", side_effect=viewer.launch), \
                        patch.object(view_scene, "execute_transition_sequence", wraps=view_scene.execute_transition_sequence) as run, \
                        patch.object(view_scene.time, "sleep"):
                    self.assertEqual(view_scene.launch_visual_viewer(
                        mode="sequence", duration=duration, contact_mode="idealized_debug"), 1)
                run.assert_called_once()
                self.assertIn("[INTERRUPTED]", output.getvalue())
                self.assertNotIn("Endpoint frozen", output.getvalue())
                self.assertNotIn("[SUCCESS]", output.getvalue())

    def test_cli_physical_default_failures_and_explicit_debug_mode(self):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for mode, contact_mode in (("stance", "physical"), ("transition", "physical"),
                                   ("sequence", "physical"), ("transition", "idealized_debug")):
            with self.subTest(mode=mode, contact_mode=contact_mode):
                process = subprocess.run(
                    [sys.executable, str(root / "scripts/view_scene.py"), "--headless", "--mode", mode,
                     "--steps", "35", "--contact-mode", contact_mode],
                    cwd=root, env=env, capture_output=True, text=True, check=False)
                self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
                self.assertNotIn("Traceback", process.stderr)
                self.assertNotIn("[SUCCESS]", process.stdout)
                self.assertIn("left_hand_site", process.stdout)
                if contact_mode == "physical":
                    self.assertIn("FAILURE/INITIALIZATION_FAILURE", process.stdout)
                else:
                    self.assertIn("NONPHYSICAL!", process.stdout)
                    self.assertIn("status=INCOMPLETE", process.stdout)

    def test_main_passes_mode_and_preserves_failure_exit(self):
        for flags in (["--headless"], []):
            with self.subTest(flags=flags), redirect_stdout(StringIO()) as output, \
                    patch.object(sys, "argv", ["view_scene.py", "--mode", "transition", *flags]), \
                    patch.object(view_scene, "is_ssh_session", return_value=True), \
                    patch.object(view_scene, "validate_scene_headlessly", return_value=False) as validate:
                self.assertEqual(view_scene.main(), 1)
                validate.assert_called_once_with(profile_name="base", mode="transition", steps=210, contact_mode="physical")
                if not flags:
                    self.assertIn("graphical desktop terminal", output.getvalue())
                    self.assertIn("--contact-mode physical", output.getvalue())

    def test_main_gui_failure_does_not_rerun_headlessly(self):
        with redirect_stdout(StringIO()), \
                patch.object(sys, "argv", ["view_scene.py", "--gui", "--mode", "sequence",
                                           "--contact-mode", "idealized_debug"]), \
                patch.object(view_scene, "launch_visual_viewer", side_effect=RuntimeError("observer failed")) as launch, \
                patch.object(view_scene, "validate_scene_headlessly") as validate:
            self.assertEqual(view_scene.main(), 1)
        launch.assert_called_once_with(profile_name="base", mode="sequence", duration=None, steps=210,
                                       contact_mode="idealized_debug")
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
