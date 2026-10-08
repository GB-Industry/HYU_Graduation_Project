"""Clean-renderer units only; manual poses are not physical-success evidence.

Fixture construction may initialize kinematics. Every test thereafter forbids
dynamics, control and IK. Only one test opens EGL, for a single explicit pose;
export orchestration is mocked, and the codec test encodes four tiny color frames.
"""
import copy
from contextlib import contextmanager, ExitStack
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image

from boulder_v1 import contact_ik, runtime, transfers, whole_body_demo
from scripts import render_clean_transfers as clean
from scripts import validate_whole_body as wb


class RenderCleanTransfersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The canonical constructor is allowed; no transfer/hold is executed.
        cls.fixture = transfers.make_transfer_fixture()
        cls.model, cls.live_data, cls.scene, _, cls.seed = cls.fixture

    def setUp(self):
        self.model_arrays = {name: value.copy() for name in dir(self.model)
                             if isinstance(value := getattr(self.model, name), np.ndarray)}
        self.data_arrays = {name: getattr(self.live_data, name).copy() for name in
                            ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "eq_active",
                             "qfrc_applied", "xfrc_applied", "xpos", "xquat", "xipos", "geom_xpos",
                             "geom_xmat", "site_xpos", "subtree_com")}
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration_state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.live_data, self.integration_state, spec)
        self.timestep, self.live_time = self.model.opt.timestep, self.live_data.time
        stack = self.enterContext(ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData",
                     "mj_integratePos", "mj_normalizeQuat", "mj_fwdPosition", "mj_fwdVelocity",
                     "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
            stack.enter_context(mock.patch.object(mujoco, name, side_effect=AssertionError(f"Forbidden {name}")))
        for module, name in ((runtime, "compute_pose_control"), (runtime, "run_pose_control"),
                             (contact_ik, "solve_contact_pose"), (transfers, "solve_contact_pose"),
                              (transfers, "execute_transfer"), (transfers, "execute_transfer_sequence"),
                              (transfers, "run_transfer_benchmark"),
                              (whole_body_demo, "make_whole_body_fixture"),
                              (whole_body_demo, "run_whole_body_benchmark")):
            stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError(f"Forbidden {name}")))

    def tearDown(self):
        for name, value in self.model_arrays.items():
            np.testing.assert_array_equal(getattr(self.model, name), value, err_msg=name)
        for name, value in self.data_arrays.items():
            np.testing.assert_array_equal(getattr(self.live_data, name), value, err_msg=name)
        state = np.empty_like(self.integration_state)
        mujoco.mj_getState(self.model, self.live_data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(state, self.integration_state)
        self.assertEqual(self.model.opt.timestep, self.timestep)
        self.assertEqual(self.live_data.time, self.live_time)

    def manual_evidence(self, case="right_hand"):
        """Synthetic parser input with unit quaternions, not a simulated episode."""
        limbs = {"right_hand": ["RIGHT_HAND"], "left_hand": ["LEFT_HAND"],
                 "left_foot": ["LEFT_FOOT"], "sequence": ["RIGHT_HAND", "LEFT_HAND"]}[case]
        contacts = {limb: limb.lower() for limb in ("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT")}
        moves, pose, time = [], self.seed.copy(), 2.

        def state(q, t):
            return {"time": t, "qpos": q.tolist(), "qvel": [0.] * self.model.nv,
                    "ctrl": [0.] * self.model.nu, "qacc_warmstart": [0.] * self.model.nv,
                    "eq_active": [False] * self.model.neq, "finite": True,
                    "contact_configuration": dict(contacts), "contact_mode": "FakeManualPose",
                    "physics_certified": False}

        for limb in limbs:
            target = {"RIGHT_HAND": "reach_target", "LEFT_HAND": "left_reach_target",
                      "LEFT_FOOT": "foot_target"}[limb]
            initial, samples = state(pose, time), []
            for step in range(1, 57):
                q = pose.copy()
                q[:3] += math.sin(math.pi * step / 56.) ** 2 * np.array([.04, -.02, .03])
                snapshot = state(q, time + step * .002)
                samples.append({"time_s": snapshot.pop("time"), **snapshot, "steps": step,
                                "contacts": dict(contacts), "phase": "REACH", "status": "RUNNING",
                                "q_ref": [999.] * self.model.nq, "qd_ref": [999.] * self.model.nv})
            pose = np.array(samples[-1]["qpos"])
            contacts[limb] = target
            final_time = time + .112
            moves.append({"moving_limb": limb, "source": initial["contact_configuration"][limb],
                          "target": target, "primitive": "foot" if limb.endswith("FOOT") else "hand",
                          "dt_s": .002, "success": True, "status": "SUCCESS",
                          "initial_state": initial, "samples": samples, "final_state": state(pose, final_time),
                          "release_time_s": time + .008, "capture": {"time_s": time + .082},
                          "events": [{"phase": "REACH", "time_s": time + .010}]})
            time = final_time
        provenance = {name: {"file": str(source := clean.ROOT / "src" / (name.replace(".", "/") + ".py")),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                      for name in clean.PROVENANCE_MODULES}
        envelope = {"kind": clean.CASES[case], "success": True, "status": "SUCCESS", "dt_s": .002,
                    "unit_evidence": "FakeManualPose; parser acceptance only, not physical success",
                    "provenance": {"modules": provenance}}
        if case == "sequence":
            return {**envelope, "moves": moves, "initial_state": copy.deepcopy(moves[0]["initial_state"]),
                    "final_state": copy.deepcopy(moves[-1]["final_state"])}
        return {**envelope, **moves[0]}

    @contextmanager
    def mocked_export(self, case="right_hand", evidence=None):
        """Exercise real loading/auditing/reporting, never a full video render."""
        evidence = self.manual_evidence(case) if evidence is None else evidence
        with tempfile.TemporaryDirectory() as directory:
            input_dir, output = Path(directory), Path(directory) / "output"
            output.mkdir()
            input_path = input_dir / f"{clean.CASES[case]}_2ms.json"
            input_path.write_text(json.dumps(evidence), encoding="utf-8")
            original = input_path.read_bytes()
            image = Image.new("RGB", (1920, 1080), (27, 43, 61))
            renderer, writers = mock.Mock(), []

            def writer(*args, **kwargs):
                result = mock.MagicMock()
                result.__enter__.return_value = result
                writers.append(result)
                return result

            with mock.patch.object(transfers, "make_transfer_fixture", return_value=self.fixture) as fixture, \
                    mock.patch.object(mujoco, "Renderer", return_value=renderer) as create_renderer, \
                    mock.patch.object(clean, "_render_view", return_value=image) as render, \
                    mock.patch.object(iio, "get_writer", side_effect=writer) as create_writer, \
                    mock.patch.object(clean, "_decoded_contact_sheet", return_value={"unit_stub": True}) as sheet, \
                    mock.patch.object(Image.Image, "save") as save:
                yield SimpleNamespace(input=input_dir, output=output, path=input_path, original=original,
                                      renderer=renderer, create_renderer=create_renderer, fixture=fixture,
                                      render=render, writers=writers, create_writer=create_writer, sheet=sheet, save=save)
            self.assertEqual(input_path.read_bytes(), original)

    def test_compose_preserves_every_scene_pixel_and_only_draws_fitting_essentials(self):
        scene = Image.new("RGB", (1440, 1080), (71, 113, 157))
        pixels = np.asarray(scene).copy()
        draw_factory = clean.ImageDraw.Draw
        for case, limb, source, target in (
                ("right_hand", "RIGHT_HAND", "right_hand", "reach_target"),
                ("left_hand", "LEFT_HAND", "left_hand", "left_reach_target"),
                ("left_foot", "LEFT_FOOT", "left_foot", "foot_target"),
                ("sequence", "LEFT_HAND", "left_hand", "left_reach_target")):
            for view in clean.VIEW_NAMES:
                for terminal in (False, True):
                    with self.subTest(case=case, view=view, terminal=terminal):
                        row = {"moving_limb": limb, "source": source, "target": target,
                               "move_index": 1 if case == "sequence" else 0,
                               "move_count": 2 if case == "sequence" else 1,
                               "phase": "SETTLED" if terminal else "SOURCE_READY",
                               "status": "SUCCESS" if terminal else "RUNNING", "terminal": terminal,
                               "time_s": 7.114, "contacts": {"LEFT_HAND": "left_reach_target",
                                                             "RIGHT_HAND": "reach_target",
                                                             "LEFT_FOOT": "foot_target", "RIGHT_FOOT": "right_foot"}}
                        before, drawings = copy.deepcopy(row), []

                        def record_draw(image):
                            real = draw_factory(image)
                            spy = mock.Mock(wraps=real)
                            drawings.append((real, spy))
                            return spy

                        with mock.patch.object(clean.ImageDraw, "Draw", side_effect=record_draw):
                            result = clean._compose(scene, row, case, view, 1920, 1080)
                        self.assertEqual(result.size, (1920, 1080))
                        np.testing.assert_array_equal(np.asarray(result)[:, :1440], pixels)
                        np.testing.assert_array_equal(np.asarray(scene), pixels)
                        self.assertEqual(row, before)
                        real, draw = drawings[0]
                        texts, bottom = [], 0
                        for call in draw.text.call_args_list:
                            text, font = call.args[1], call.kwargs["font"]
                            bounds = real.textbbox(call.args[0], text, font=font)
                            self.assertIn(font.size, (20, 25, 30))
                            self.assertGreaterEqual(bounds[0], 1468, text)
                            self.assertLessEqual(bounds[2], 1892, text)
                            self.assertGreaterEqual(bounds[1], bottom, text)
                            self.assertLessEqual(bounds[3], 1080, text)
                            bottom = bounds[3]
                            texts.append(text)
                        text = " ".join(texts)
                        for essential in (clean.VIEW_NAMES[view], "Move 2 / 2" if case == "sequence" else "Move 1 / 1",
                                          limb.replace("_", " "), source, "-> " + target, "PHASE",
                                          row["phase"].replace("_", " "),
                                          "MOVE 2: SUCCESS" if terminal and case == "sequence" else row["status"],
                                          "ACTUAL CONTACTS", "LH: left_reach_target", "RH: reach_target",
                                          "LF: foot_target", "RF: right_foot", "Recorded time 7.114 s"):
                            self.assertIn(essential, text)
                        self.assertEqual("Wall hidden in display only" in text, view == "front")
                        self.assertNotRegex(text.lower(), r"force|speed|capacity|margin|torque|qvel|fn=|ft=")
                        for call in draw.line.call_args_list:
                            self.assertGreaterEqual(min(call.args[0][::2]), 1440)

    def test_compose_defaults_and_extra_force_fields_do_not_invent_or_draw_pose_data(self):
        row = {"time_s": 2.024, "contacts": {}, "force": "DO_NOT_DRAW_FORCE",
               "tangential_speed": "DO_NOT_DRAW_SPEED", "pose_available": False}
        real = clean.ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
        draw = mock.Mock(wraps=real)
        with mock.patch.object(clean.ImageDraw, "Draw", return_value=draw):
            clean._compose(Image.new("RGB", (1440, 1080)), row, "right_hand", "rear", 1920, 1080)
        texts = [call.args[1] for call in draw.text.call_args_list]
        for text in ("Move 1 / 1", "SOURCE", "RUNNING", "LH: none", "RH: none", "LF: none", "RF: none"):
            self.assertIn(text, texts)
        self.assertNotIn("SUCCESS", texts)
        self.assertNotIn("DO_NOT_DRAW_FORCE", texts)
        self.assertNotIn("DO_NOT_DRAW_SPEED", texts)

    def test_actor_bounds_cover_rotated_primitives_and_ignore_environment_and_invisible_geoms(self):
        s = math.sqrt(.5)
        rotation = np.array([[s, 0., s], [0., 1., 0.], [-s, 0., s]])
        center = np.array([1., -2., 3.])
        for kind, size, extent in (
                (mujoco.mjtGeom.mjGEOM_SPHERE, [.3, 0., 0.], [.3, .3, .3]),
                (mujoco.mjtGeom.mjGEOM_BOX, [1., 2., 3.], [4 * s, 2., 4 * s]),
                (mujoco.mjtGeom.mjGEOM_ELLIPSOID, [1., 2., 3.], [math.sqrt(5), 2., math.sqrt(5)]),
                (mujoco.mjtGeom.mjGEOM_CAPSULE, [.3, 2., 0.], [2 * s + .3, .3, 2 * s + .3]),
                (mujoco.mjtGeom.mjGEOM_CYLINDER, [.3, 2., 0.], [2.3 * s, .3, 2.3 * s])):
            with self.subTest(kind=kind):
                model = SimpleNamespace(ngeom=3, geom_bodyid=np.array([1, 2, 1]),
                                        geom_rgba=np.array([[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 0.]]),
                                        geom_size=np.array([size, [1000., 1000., 1000.], [1000., 1000., 1000.]]),
                                        geom_type=np.array([kind] * 3))
                data = SimpleNamespace(geom_xpos=np.array([center, center, center]),
                                       geom_xmat=np.array([rotation.ravel()] * 3))
                with mock.patch("boulder_v1.model_validation.climber_body_ids", return_value=np.array([1])):
                    lo, hi = clean._actor_bounds(mujoco, model, data)
                np.testing.assert_allclose(lo, center - extent, atol=1e-12)
                np.testing.assert_allclose(hi, center + extent, atol=1e-12)

    def test_actor_bounds_refuse_missing_or_unsupported_visible_actor_geometry(self):
        display = copy.copy(self.model)
        scratch = mujoco.MjData(display)
        clean._kinematics(mujoco, display, scratch, {"qpos": self.seed, "time_s": 2.})
        from boulder_v1.model_validation import climber_body_ids

        actor = np.flatnonzero(np.isin(display.geom_bodyid, climber_body_ids(display)))
        display.geom_rgba[actor, 3] = 0.
        with self.assertRaisesRegex(ValueError, "No finite visible climber bounds"):
            clean._actor_bounds(mujoco, display, scratch)
        display.geom_rgba[actor[0], 3] = 1.
        display.geom_type[actor[0]] = mujoco.mjtGeom.mjGEOM_MESH
        with self.assertRaisesRegex(ValueError, "Unsupported visible actor geom"):
            clean._actor_bounds(mujoco, display, scratch)

    def test_fixed_cameras_use_all_time_actor_union_and_spherical_fov_margin_for_every_case(self):
        offsets = np.array([[0., 0., 0.], [.7, -.4, .5], [.1, .05, 0.]])
        rows = [{"qpos": (self.seed + np.r_[offset, np.zeros(self.model.nq - 3)]).tolist(), "time_s": 2. + i * .002}
                for i, offset in enumerate(offsets)]
        before = copy.deepcopy(rows)
        scratch = mujoco.MjData(self.model)
        clean._kinematics(mujoco, self.model, scratch, rows[0])
        initial_lo, initial_hi = clean._actor_bounds(mujoco, self.model, scratch)
        lo, hi = initial_lo + offsets.min(axis=0), initial_hi + offsets.max(axis=0)
        center, radius = (lo + hi) / 2., np.linalg.norm((hi - lo) / 2.)
        for case in clean.CASES:
            with self.subTest(case=case), mock.patch.object(clean, "_kinematics", wraps=clean._kinematics) as refresh:
                cameras, bounds = clean._camera_definitions(self.model, rows, mujoco, case)
                self.assertEqual(refresh.call_count, len(rows))
                self.assertTrue(all(call.args[2] is not self.live_data for call in refresh.call_args_list))
                np.testing.assert_allclose(bounds["minimum_world_m"], lo, atol=1e-12)
                np.testing.assert_allclose(bounds["maximum_world_m"], hi, atol=1e-12)
                self.assertAlmostEqual(bounds["bounding_sphere_radius_m"], radius)
                self.assertEqual(bounds["framing_margin_factor"], 1.12)
                self.assertEqual(set(cameras), {"rear", "side", "front"})
                for view, camera in cameras.items():
                    np.testing.assert_allclose(camera["lookat_world_m"], center, atol=1e-12)
                    half_fov = math.radians(self.model.vis.global_.fovy / 2.)
                    self.assertAlmostEqual(camera["distance_m"], radius * 1.12 / math.sin(half_fov))
                    self.assertLess(math.asin(radius / camera["distance_m"]), half_fov)
                    self.assertEqual(camera["vertical_fov_deg"], self.model.vis.global_.fovy)
                    self.assertEqual(camera["elevation_deg"], -5.)
                    self.assertTrue(camera["fixed_over_entire_recording"])
                    self.assertEqual(camera["wall_display"], "hidden" if view == "front" else "visible")
                    self.assertEqual(camera["azimuth_deg"], {"rear": 135., "side": 180., "front": 225.}[view]
                                     if case == "right_hand" else
                                     {"rear": 135. if case == "sequence" else 45., "side": 0., "front": 315.}[view])
        self.assertEqual(rows, before)

    def test_kinematics_refreshes_only_owned_scratch_from_actual_q_state(self):
        display, row = copy.copy(self.model), {"qpos": self.seed.tolist(), "time_s": 12.5,
                                              "qvel": np.linspace(-.1, .1, self.model.nv).tolist(),
                                              "q_ref": [999.] * self.model.nq, "ctrl": [999.] * self.model.nu}
        row["qpos"][0] += .03
        before = copy.deepcopy(row)
        scratch = mujoco.MjData(display)
        scratch.ctrl[:] = .123
        scratch.qfrc_applied[:] = .456
        scratch.xfrc_applied[:] = .789
        scratch.qacc_warmstart[:] = .321
        untouched = {name: getattr(scratch, name).copy() for name in
                     ("ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart", "eq_active")}
        with mock.patch.object(mujoco, "mj_kinematics", wraps=mujoco.mj_kinematics) as fk, \
                mock.patch.object(mujoco, "mj_comPos", wraps=mujoco.mj_comPos) as com, \
                mock.patch.object(mujoco, "mj_camlight", wraps=mujoco.mj_camlight) as light:
            clean._kinematics(mujoco, display, scratch, row)
        for refresh in (fk, com, light):
            refresh.assert_called_once_with(display, scratch)
        np.testing.assert_array_equal(scratch.qpos, row["qpos"])
        np.testing.assert_array_equal(scratch.qvel, row["qvel"])
        self.assertEqual(scratch.time, 12.5)
        np.testing.assert_allclose(scratch.body("climber_root").xpos, row["qpos"][:3])
        self.assertTrue(np.isfinite(scratch.subtree_com).all())
        for name, value in untouched.items():
            np.testing.assert_array_equal(getattr(scratch, name), value, err_msg=name)
        clean._kinematics(mujoco, display, scratch, {"qpos": row["qpos"], "time_s": 12.502})
        np.testing.assert_array_equal(scratch.qvel, np.zeros(self.model.nv))
        self.assertEqual(row, before)

    def test_render_view_hides_sites_and_front_wall_on_display_copy_then_restores_even_on_errors(self):
        row = {"qpos": self.seed.tolist(), "time_s": 2.024}
        cameras, _ = clean._camera_definitions(self.model, [row], mujoco, "right_hand")
        for view in cameras:
            for error in (None, "update_scene", "render"):
                with self.subTest(view=view, error=error):
                    display = copy.copy(self.model)
                    scratch = mujoco.MjData(display)
                    arrays = {name: value.copy() for name in dir(display)
                              if isinstance(value := getattr(display, name), np.ndarray)}
                    walls = [display.geom(f"{wall.id}_geom").id for wall in self.scene.walls]
                    self.assertTrue(walls)
                    self.assertFalse(np.shares_memory(display.geom_rgba, self.model.geom_rgba))
                    self.assertFalse(np.shares_memory(display.geom_contype, self.model.geom_contype))
                    renderer = mock.Mock()

                    def update(data, *, camera, scene_option):
                        self.assertIs(data, scratch)
                        np.testing.assert_array_equal(data.qpos, row["qpos"])
                        np.testing.assert_array_equal(scene_option.sitegroup, np.zeros(6))
                        np.testing.assert_array_equal(display.geom_rgba[walls, 3],
                                                      np.zeros(len(walls)) if view == "front" else arrays["geom_rgba"][walls, 3])
                        np.testing.assert_array_equal(camera.lookat, cameras[view]["lookat_world_m"])
                        self.assertEqual(camera.distance, cameras[view]["distance_m"])
                        self.assertEqual(camera.azimuth, cameras[view]["azimuth_deg"])
                        self.assertEqual(camera.elevation, cameras[view]["elevation_deg"])
                        for name in ("geom_contype", "geom_conaffinity", "body_mass", "body_inertia", "eq_data",
                                     "geom_size", "geom_pos", "site_rgba"):
                            np.testing.assert_array_equal(getattr(display, name), arrays[name], err_msg=name)
                        if error == "update_scene":
                            raise RuntimeError("update_scene failed")

                    renderer.update_scene.side_effect = update
                    renderer.render.return_value = np.full((1080, 1440, 3), 83, dtype=np.uint8)
                    if error == "render":
                        renderer.render.side_effect = RuntimeError("render failed")
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error + " failed"):
                            clean._render_view(mujoco, renderer, display, scratch, row, cameras[view],
                                               walls, view, "right_hand", 1920, 1080)
                    else:
                        image = clean._render_view(mujoco, renderer, display, scratch, row, cameras[view],
                                                   walls, view, "right_hand", 1920, 1080)
                        np.testing.assert_array_equal(np.asarray(image)[:, :1440], renderer.render.return_value)
                    for name, value in arrays.items():
                        np.testing.assert_array_equal(getattr(display, name), value, err_msg=name)

    def test_actual_egl_pose_has_native_shoes_palms_no_sites_and_unmodified_scene_pixels(self):
        display = copy.copy(self.model)
        scratch = mujoco.MjData(display)
        row = {"qpos": self.seed.tolist(), "qvel": [0.] * self.model.nv, "time_s": 12.5,
               "moving_limb": "RIGHT_HAND", "source": "right_hand", "target": "reach_target",
               "status": "KINEMATIC UNIT ONLY", "phase": "SOURCE", "contacts": {}}
        cameras, bounds = clean._camera_definitions(self.model, [row], mujoco, "right_hand")
        walls = [display.geom(f"{wall.id}_geom").id for wall in self.scene.walls]
        original = display.geom_rgba.copy()
        self.assertGreaterEqual(display.vis.global_.offwidth, 1440)
        self.assertGreaterEqual(display.vis.global_.offheight, 1080)
        with mujoco.Renderer(display, width=1440, height=1080) as renderer:
            with mock.patch.object(renderer, "update_scene", wraps=renderer.update_scene) as update:
                image = clean._render_view(mujoco, renderer, display, scratch, row, cameras["front"],
                                           walls, "front", "right_hand", 1920, 1080)
            np.testing.assert_array_equal(update.call_args.kwargs["scene_option"].sitegroup, np.zeros(6))
            pixels = renderer.render().copy()
            self.assertEqual(pixels.shape, (1080, 1440, 3))
            self.assertGreater(float(pixels.std()), 5.)
            np.testing.assert_array_equal(np.asarray(image)[:, :1440], pixels)
            scene_geoms = renderer.scene.geoms[:renderer.scene.ngeom]
            self.assertFalse(any(geom.objtype == mujoco.mjtObj.mjOBJ_SITE for geom in scene_geoms))
            ids = {geom.objid for geom in scene_geoms if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM}
            self.assertTrue(ids.isdisjoint(walls))
            for name in ("left_hand_geom", "right_hand_geom", "left_foot_geom", "right_foot_geom",
                         "left_shoe_upper", "right_shoe_upper"):
                geom = next(geom for geom in scene_geoms
                            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == display.geom(name).id)
                np.testing.assert_allclose(geom.rgba, original[geom.objid], atol=1e-7)
            # Project the conservative union's eight corners with the actual GL
            # camera; fitting just the pelvis or projected geom centers is unsafe.
            corners = np.array(list(product(*zip(bounds["minimum_world_m"], bounds["maximum_world_m"]))))
            gl = renderer.scene.camera[0]
            delta = corners - gl.pos
            depth = delta @ gl.forward
            vertical = delta @ gl.up * gl.frustum_near / depth
            horizontal = delta @ np.cross(gl.forward, gl.up) * gl.frustum_near / depth
            half_width = (gl.frustum_top - gl.frustum_bottom) / 2. * 1440 / 1080
            self.assertTrue(np.all(depth > gl.frustum_near))
            self.assertTrue(np.all(depth < gl.frustum_far))
            self.assertTrue(np.all(vertical > gl.frustum_bottom))
            self.assertTrue(np.all(vertical < gl.frustum_top))
            self.assertTrue(np.all(np.abs(horizontal - gl.frustum_center) < half_width))
        np.testing.assert_array_equal(display.geom_rgba, original)

    def test_sample_indices_are_monotonic_twenty_fps_zero_order_native_rows_with_terminal(self):
        rows = [{"time_s": time, "qpos": [i, -i]} for i, time in
                enumerate((2., 2.024, 2.05, 2.05, 2.074, 2.099, 2.104))]
        before = copy.deepcopy(rows)
        indices = clean._sample_indices(rows, 20.)
        self.assertEqual(indices, [0, 3, 5, 6])
        self.assertEqual(indices, sorted(indices))
        self.assertEqual([rows[i]["qpos"] for i in indices], [[0, 0], [3, -3], [5, -5], [6, -6]])
        self.assertEqual(rows, before)
        self.assertEqual(clean._sample_indices(rows[:1], 20.), [0])
        self.assertEqual(clean._sample_indices([{"time_s": 2.}, {"time_s": 2.05}], 20.), [0, 1])
        self.assertEqual(clean._sample_indices([{"time_s": 2.}, {"time_s": 2.}], 20.), [1])
        with self.assertRaisesRegex(ValueError, "monotonic clock"):
            clean._sample_indices([], 20.)
        with self.assertRaisesRegex(ValueError, "monotonic clock"):
            clean._sample_indices([{"time_s": 2.}, {"time_s": 1.}], 20.)

    def test_event_keyframes_use_native_epoch_brackets_and_stay_in_the_correct_move(self):
        rows = [{"move_index": move, "time_s": time} for move, times in
                ((0, (10., 10.002, 10.004, 12.002, 12.004)),
                 (1, (12.004, 12.006, 12.008, 13.504, 13.506, 13.508))) for time in times]
        evidence = {"moves": [
            {"release_time_s": 10.002, "events": [{"phase": "REACH", "time_s": 10.002}],
             "capture": {"time_s": 12.002}, "readiness": {"time": 12.004}},
            {"primitive": "foot", "events": [{"event": "RELEASED", "time_s": 12.004},
                                              {"phase": "REACH", "time_s": 12.004}],
             "capture": {"time_s": 13.505}, "touchdown": {"time_s": 13.504},
             "acquisition": {"time_s": 13.506}, "load_complete_time_s": 13.508,
             "readiness": {"time": 13.508}}]}
        before = copy.deepcopy((evidence, rows))
        frames = clean._event_keyframes(evidence, rows)
        self.assertEqual(frames, {"start": 0, "final": 10, "move1_release_before": 1,
                                  "move1_release_after": 2, "move1_reach_mid": 3,
                                  "move1_capture_before": 3, "move1_capture_after": 4, "move1_ready": 4,
                                  "move2_release_before": 5, "move2_release_after": 6,
                                  "move2_reach_mid": 8, "move2_capture_before": 8,
                                  "move2_capture_after": 9, "move2_touchdown": 8,
                                  "move2_acquired": 9, "move2_loaded": 10, "move2_ready": 10})
        for label, index in frames.items():
            if label.startswith("move"):
                self.assertEqual(rows[index]["move_index"], int(label[4]) - 1)
        self.assertEqual((evidence, rows), before)
        self.assertEqual(clean._event_keyframes({}, [{"move_index": 0, "time_s": 2.}]), {"start": 0, "final": 0})

    def test_whole_body_reach_mid_uses_saved_five_second_hand_duration(self):
        evidence = {"request": {"hand_reach_s": 5.}, "events": [{"phase": "REACH", "time_s": 10.}]}
        rows = [{"move_index": 0, "time_s": time} for time in (10., 12., 12.5, 15.)]
        self.assertEqual(clean._event_keyframes(evidence, rows)["move1_reach_mid"], 2)

    def test_load_verified_evidence_reads_literal_file_and_checks_source_sha_without_rewriting(self):
        evidence = self.manual_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached_native_states.json"
            clean._write_json(path, evidence)
            before = path.read_bytes()
            self.assertEqual(clean._load_verified_evidence(path), evidence)
            self.assertEqual(path.read_bytes(), before)
            changed = copy.deepcopy(evidence)
            name = clean.PROVENANCE_MODULES[0]
            changed["provenance"]["modules"][name]["sha256"] = "0" * 64
            clean._write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "Recorded source differs"):
                clean._load_verified_evidence(path)
            outside = Path(directory) / "outside_source.py"
            outside.write_text("# unit source outside authorized root\n", encoding="utf-8")
            changed["provenance"]["modules"][name] = {
                "file": str(outside), "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}
            clean._write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "Recorded source differs"):
                clean._load_verified_evidence(path)

    def test_loader_rejects_incomplete_or_misbound_canonical_hash_sets(self):
        evidence = self.manual_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            for mode in ("subset", "wrong_identity"):
                changed = copy.deepcopy(evidence)
                modules = changed["provenance"]["modules"]
                if mode == "subset":
                    changed["provenance"]["modules"] = {clean.PROVENANCE_MODULES[0]: modules[clean.PROVENANCE_MODULES[0]]}
                else:
                    modules[clean.PROVENANCE_MODULES[0]] = modules[clean.PROVENANCE_MODULES[1]]
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    clean._load_verified_evidence(path)

    def whole_body_evidence(self, case="right_hand"):
        """Saved-input reconstruction contract only; no whole-body simulation."""
        evidence = self.manual_evidence(case)
        with mock.patch.object(whole_body_demo, "make_whole_body_fixture", return_value=self.fixture):
            _, inputs = wb._fixture_metadata(.002)
        evidence.update(fixture=wb.FIXTURE, fixture_inputs=inputs, profile=inputs["profile"],
                        provenance=wb._json_value(wb._provenance([])))
        moves = evidence.get("moves") or [evidence]
        for move in moves:
            move.update(steps=len(move["samples"]), duration_s=len(move["samples"]) * .002)
            for row in move["samples"]:
                row.update(qfrc_applied=[0.] * self.model.nv,
                           external_force_world_N=[[0.] * 6] * self.model.nbody)
        return evidence

    def test_whole_body_loader_requires_explicit_fixture_full_hashes_and_current_sources(self):
        evidence = self.whole_body_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "right_hand_2ms.json"
            clean._write_json(path, evidence)
            self.assertEqual(clean._load_verified_evidence(path, "whole_body"), evidence)
            with self.assertRaisesRegex(ValueError, "fixture"):
                clean._load_verified_evidence(path)
            for mode in ("bad_fixture", "old_hash", "missing_whole_body", "unknown_module"):
                changed = copy.deepcopy(evidence)
                modules = changed["provenance"]["modules"]
                if mode == "bad_fixture":
                    changed["fixture"] = "invented_fixture"
                elif mode == "old_hash":
                    modules["boulder_v1.whole_body_motion"]["sha256"] = "0" * 64
                elif mode == "missing_whole_body":
                    del modules["boulder_v1.whole_body_reference"]
                else:
                    modules["boulder_v1.unknown"] = modules["boulder_v1.whole_body_demo"]
                clean._write_json(path, changed)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    clean._load_verified_evidence(path, "whole_body")

    def test_whole_body_all_four_exports_use_matching_saved_model_and_twelve_shared_index_videos(self):
        for case in clean.CASES:
            with self.subTest(case=case), self.mocked_export(case, self.whole_body_evidence(case)) as unit:
                report = clean.render_case(case, unit.input, unit.output, fixture="whole_body")
                unit.fixture.assert_not_called()
                self.assertEqual(len(unit.writers), 3)
                self.assertEqual(report["audit_source"]["fixture"], wb.FIXTURE)
                self.assertEqual(report["audit_source"]["factory"], clean.FIXTURES["whole_body"][1])
                self.assertEqual(report["audit_source"]["input_sha256"], hashlib.sha256(unit.original).hexdigest())
                self.assertEqual(report["resolution"], [1920, 1080])
                self.assertEqual(report["physics_steps_executed"], 0)
                for view in clean.VIEW_NAMES:
                    calls = [call for call in unit.render.call_args_list if call.args[7] == view]
                    self.assertEqual([call.args[4]["record_index"] for call in calls[:len(report["rendered_native_indices"])]],
                                     report["rendered_native_indices"])
                self.assertIsNone(report["motion"]["quality_verdict"])
                self.assertEqual(report["motion_demonstration_verdict"], "PENDING VISUAL INSPECTION")

    def test_whole_body_audit_only_never_opens_renderer_writer_or_replays_fixture(self):
        with self.mocked_export(evidence=self.whole_body_evidence()) as unit:
            report = clean.render_case("right_hand", unit.input, unit.output, fixture="whole_body", audit_only=True)
            unit.fixture.assert_not_called()
            unit.create_renderer.assert_not_called()
            unit.create_writer.assert_not_called()
            unit.render.assert_not_called()
            unit.save.assert_not_called()
            self.assertEqual(report["physics_steps_executed"], 0)
            self.assertIsNone(report["quality_verdict"])
            self.assertTrue((unit.output / "audit_right_hand_report.json").is_file())

    def test_whole_body_bad_input_hash_or_handoff_is_rejected_before_export(self):
        for mode in ("input_hash", "force", "steps", "handoff"):
            evidence = self.whole_body_evidence("sequence")
            if mode == "input_hash":
                evidence["fixture_inputs"]["scene"]["scale"] = 2.
            elif mode == "handoff":
                evidence["moves"][1]["initial_state"]["qacc_warmstart"][0] = 1.
            elif mode == "force":
                evidence["moves"][0]["samples"][0]["qfrc_applied"][0] = 1.
            else:
                evidence["moves"][0]["steps"] += 1
            with self.subTest(mode=mode), self.mocked_export("sequence", evidence) as unit:
                with self.assertRaisesRegex(ValueError, "hash|handoff/clock/force"):
                    clean.render_case("sequence", unit.input, unit.output, fixture="whole_body")
                unit.create_renderer.assert_not_called()
                unit.create_writer.assert_not_called()

    def test_decoded_capture_brackets_cannot_label_a_pre_capture_frame_as_after(self):
        rows = [{"time_s": time} for time in (8.600, 8.604, 8.606, 8.650)]
        reader = mock.MagicMock()
        reader.__enter__.return_value = reader
        reader.get_data.side_effect = lambda index: np.full((108, 192, 3), index * 120, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(iio, "get_reader", return_value=reader):
            result = clean._decoded_contact_sheet(
                Path(directory) / "video.mp4", [0, 3], {"move1_capture_before": 1, "move1_capture_after": 2},
                rows, Path(directory) / "sheet.png")
        mapping = result["native_to_decoded_mapping"]
        before, after = mapping["move1_capture_before"], mapping["move1_capture_after"]
        self.assertEqual((before["decoded_frame_index"], after["decoded_frame_index"]), (0, 1))
        self.assertLessEqual(before["decoded_state_time_s"], before["native_state_time_s"])
        self.assertGreaterEqual(after["decoded_state_time_s"], after["native_state_time_s"])
        self.assertEqual(result["state_times_s"], [8.600, 8.650])

    def test_loader_rejects_unsuccessful_or_unprovenanced_records(self):
        for change in ({"success": False}, {"status": "INCOMPLETE"}, {"provenance": {}},
                       {"provenance": {"modules": {}}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                evidence = {**self.manual_evidence(), **change}
                path = Path(directory) / "evidence.json"
                clean._write_json(path, evidence)
                with self.assertRaisesRegex(ValueError, "recorded successful episode|source-module provenance"):
                    clean._load_verified_evidence(path)

    def test_success_envelope_cannot_render_malformed_nonfinite_or_unavailable_native_states(self):
        for field, value in (("qpos", None), ("qpos", [0.] * (self.model.nq - 1)),
                             ("qvel", [float("nan")] * self.model.nv), ("pose_available", False),
                             ("finite", False)):
            with self.subTest(field=field):
                evidence = self.manual_evidence()
                evidence["samples"][0][field] = value
                with self.mocked_export(evidence=evidence) as unit:
                    with self.assertRaisesRegex(ValueError, "Corrupt recorded state"):
                        clean.render_case("right_hand", unit.input, unit.output)
                    unit.create_renderer.assert_not_called()
                    unit.create_writer.assert_not_called()
                    unit.render.assert_not_called()
                    unit.save.assert_not_called()
                    self.assertEqual(list(unit.output.iterdir()), [])

    def test_discontinuous_native_clock_is_refused_before_any_render_or_video(self):
        for error in ("reset", "gap", "changed_duplicate"):
            with self.subTest(error=error):
                evidence = self.manual_evidence()
                if error == "reset":
                    evidence["samples"][1]["time_s"] = 1.
                elif error == "gap":
                    evidence["samples"][1]["time_s"] += .001
                else:
                    evidence["samples"][1]["time_s"] = evidence["samples"][0]["time_s"]
                with self.mocked_export(evidence=evidence) as unit:
                    with self.assertRaisesRegex(ValueError, "trajectory is discontinuous"):
                        clean.render_case("right_hand", unit.input, unit.output)
                    unit.create_renderer.assert_not_called()
                    unit.create_writer.assert_not_called()
                    unit.save.assert_not_called()

    def test_all_four_case_exports_share_exact_indices_and_report_actual_native_keyframes(self):
        for case in clean.CASES:
            with self.subTest(case=case), self.mocked_export(case) as unit:
                report = clean.render_case(case, unit.input, unit.output)
                evidence = json.loads(unit.original)
                rows = clean.trajectory_rows(evidence)
                indices, keyframes = clean._sample_indices(rows, 20.), clean._event_keyframes(evidence, rows)
                self.assertEqual(report["input_json"], str(unit.path))
                self.assertEqual(report["input_sha256"], hashlib.sha256(unit.original).hexdigest())
                self.assertTrue(report["source_provenance_checked"])
                self.assertEqual(report["physics_steps_executed"], 0)
                self.assertFalse(report["dynamics_replayed"])
                self.assertTrue(report["all_cameras_share_native_record_indices"])
                self.assertEqual(report["rendered_native_indices"], indices)
                self.assertEqual(report["rendered_state_times_s"], [rows[i]["time_s"] for i in indices])
                self.assertEqual(report["resolution"], [1920, 1080])
                self.assertEqual(report["scene_viewport"], [0, 0, 1440, 1080])
                self.assertEqual(report["sidebar"], [1440, 0, 480, 1080])
                self.assertEqual(report["fps"], 20.)
                self.assertFalse(report["preview_only"])
                self.assertIsNone(report["motion"]["quality_verdict"])
                self.assertEqual(report["motion_demonstration_verdict"], "PENDING VISUAL INSPECTION")
                unit.fixture.assert_called_once_with(.002)
                display = unit.create_renderer.call_args.args[0]
                self.assertIsNot(display, self.model)
                self.assertFalse(np.shares_memory(display.geom_rgba, self.model.geom_rgba))
                self.assertEqual(unit.create_renderer.call_args.kwargs, {"height": 1080, "width": 1440})
                self.assertEqual(len(unit.writers), 3)
                for view, writer in zip(("rear", "side", "front"), unit.writers):
                    calls = [call for call in unit.render.call_args_list if call.args[7] == view]
                    self.assertEqual([call.args[4]["record_index"] for call in calls], indices + list(keyframes.values()))
                    for call, native in zip(calls, indices + list(keyframes.values())):
                        row = call.args[4]
                        self.assertIs(call.args[2], display)
                        self.assertIsNot(call.args[3], self.live_data)
                        self.assertEqual(row["qpos"], rows[native]["qpos"])
                        self.assertEqual(call.args[5], report["cameras"][view])
                        self.assertEqual(call.args[9:], (1920, 1080))
                        if row["row_kind"] != "sample":
                            self.assertEqual(row["phase"], "SETTLED" if row["row_kind"] == "final_state" else "SOURCE_READY")
                            self.assertEqual(row["terminal"], row["row_kind"] == "final_state")
                    writer.__enter__.assert_called_once_with()
                    writer.__exit__.assert_called_once_with(None, None, None)
                    self.assertEqual(writer.append_data.call_count, len(indices))
                    for event, native in keyframes.items():
                        snapshot = report["artifacts"][view]["keyframes"][event]
                        self.assertEqual(snapshot["native_record_index"], native)
                        self.assertEqual(snapshot["actual_state_time_s"], rows[native]["time_s"])
                    self.assertEqual(report["artifacts"][view]["frame_count"], len(indices))
                self.assertEqual(unit.sheet.call_count, 3)
                for call in unit.sheet.call_args_list:
                    self.assertEqual(call.args[0].suffix, ".mp4")
                    self.assertEqual(call.args[1:3], (indices, keyframes))
                for call in unit.create_writer.call_args_list:
                    self.assertEqual(call.kwargs["fps"], 20.)
                    self.assertEqual(call.kwargs["codec"], "libx264")
                    self.assertEqual(call.kwargs["pixelformat"], "yuv420p")
                self.assertEqual(unit.save.call_count, 3 * len(keyframes))
                unit.renderer.close.assert_called_once_with()
                saved = json.loads((unit.output / f"clean_{case}_report.json").read_text())
                self.assertEqual(saved, report)

    def test_preview_is_not_video_evidence_and_closes_renderer(self):
        with self.mocked_export() as unit:
            report = clean.render_case("right_hand", unit.input, unit.output, preview=True)
            self.assertTrue(report["preview_only"])
            self.assertEqual(report["artifacts"], {})
            self.assertEqual(unit.render.call_count, 3)
            self.assertTrue(all(call.args[4]["record_index"] == 0 for call in unit.render.call_args_list))
            unit.create_writer.assert_not_called()
            unit.sheet.assert_not_called()
            self.assertEqual(unit.save.call_count, 3)
            unit.renderer.close.assert_called_once_with()

    def test_renderer_and_writer_are_closed_when_export_or_decode_raises(self):
        for failure in ("render", "append", "decode"):
            with self.subTest(failure=failure), self.mocked_export() as unit:
                if failure == "render":
                    unit.render.side_effect = RuntimeError("render failed")
                elif failure == "decode":
                    unit.sheet.side_effect = RuntimeError("decode failed")
                else:
                    factory = unit.create_writer.side_effect

                    def fail_append(*args, **kwargs):
                        writer = factory(*args, **kwargs)
                        writer.append_data.side_effect = RuntimeError("append failed")
                        return writer

                    unit.create_writer.side_effect = fail_append
                with self.assertRaisesRegex(RuntimeError, failure + " failed"):
                    clean.render_case("right_hand", unit.input, unit.output)
                unit.renderer.close.assert_called_once_with()
                self.assertEqual(len(unit.writers), 1)
                unit.writers[0].__exit__.assert_called_once()
                if failure == "decode":
                    unit.writers[0].__exit__.assert_called_once_with(None, None, None)
                else:
                    self.assertIs(unit.writers[0].__exit__.call_args.args[0], RuntimeError)
                self.assertFalse((unit.output / "clean_right_hand_report.json").exists())

    def test_decoded_contact_sheet_uses_actual_mp4_pixels_indices_times_and_closes_reader(self):
        rows = [{"time_s": time} for time in (2., 2.002, 2.048, 2.052, 2.098, 2.104)]
        indices = [0, 2, 4, 5]
        events = {"start": 0, "release_before": 1, "release_after": 2,
                  "capture_before": 3, "acquired": 4, "final": 5}
        with tempfile.TemporaryDirectory() as directory:
            video, output = Path(directory) / "encoded_unit.mp4", Path(directory) / "decoded.png"
            with iio.get_writer(str(video), fps=20., codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-threads", "1"]) as writer:
                for color in ((201, 35, 59), (29, 187, 71), (53, 79, 211), (181, 157, 31)):
                    writer.append_data(np.full((48, 64, 3), color, dtype=np.uint8))
            reader = iio.get_reader(video)
            with mock.patch.object(iio, "get_reader", return_value=reader) as get_reader, \
                    mock.patch.object(reader, "get_data", wraps=reader.get_data) as get_frame:
                result = clean._decoded_contact_sheet(video, indices, events, rows, output)
            get_reader.assert_called_once_with(video)
            self.assertEqual([call.args[0] for call in get_frame.call_args_list], [0, 1, 2, 3])
            self.assertTrue(reader.closed)
            self.assertEqual(result["decoded_video_frame_indices"], [0, 1, 2, 3])
            self.assertEqual(result["state_times_s"], [2., 2.048, 2.098, 2.104])
            self.assertEqual(result["path"], str(output))
            with Image.open(output) as sheet, iio.get_reader(video) as decoded:
                self.assertEqual(sheet.size, (1920, 810))
                for frame in range(4):
                    x, y = frame % 3 * 640, frame // 3 * 405 + 40
                    np.testing.assert_array_equal(np.asarray(sheet.crop((x, y, x + 64, y + 48))), decoded.get_data(frame))


if __name__ == "__main__":
    unittest.main()
