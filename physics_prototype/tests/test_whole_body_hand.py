"""Whole-body hand preparation and native RH -> LH continuity, without assistance."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1 import single_hand, static_control, static_state, transfers
from boulder_v1 import whole_body_demo as demo
from boulder_v1 import whole_body_motion as motion
from boulder_v1.contact import CAPTURE_DISTANCE, CAPTURE_ORIENTATION, CAPTURE_SPEED
from boulder_v1.contact_geometry import Frame, canonical_geometry
from boulder_v1.grasp import GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Limb
from boulder_v1.static_state import _integration_state, initialize_static_reference
from boulder_v1.support import FORCE_TOLERANCE, MAX_SLIP_SPEED


PHYSICAL_ARRAYS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_quat",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
    "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
    "site_pos", "site_quat", "jnt_type", "jnt_range", "jnt_axis", "jnt_pos",
    "dof_damping", "dof_armature", "actuator_trnid", "actuator_gear",
    "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
    "actuator_ctrllimited", "actuator_forcelimited", "pair_friction",
    "pair_solref", "pair_solimp", "exclude_signature", "eq_type", "eq_objtype",
    "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp")


def model_snapshot(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


def assert_capture(test, measurement):
    test.assertLessEqual(measurement["gap_m"], CAPTURE_DISTANCE + 1e-12)
    test.assertGreaterEqual(measurement["orientation"], CAPTURE_ORIENTATION)
    test.assertLessEqual(measurement["relative_speed_m_s"], CAPTURE_SPEED + 1e-12)
    test.assertLess(measurement["penetration_m"], .001)
    test.assertLessEqual(measurement["signed_geom_distance_m"], CAPTURE_DISTANCE + 1e-12)


class WholeBodyHandContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.scene, cls.profile, cls.seed = demo.make_whole_body_fixture(.002)
        # Explicit source admission is setup, not a simulated static hold.
        cls.reference, cls.manager = initialize_static_reference(
            cls.model, cls.data, cls.scene, cls.profile, cls.seed, cls.scene.start_configuration)
        cls.goal = demo.hand_motion(cls.model, cls.reference)

    def test_source_geometry_and_original_physical_contract(self):
        original, _, scene, profile, seed = transfers.make_transfer_fixture(.002)
        self.assertEqual(self.profile.to_dict(), profile.to_dict())
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu, self.model.njnt), (32, 31, 25, 26))
        self.assertAlmostEqual(self.model.body_subtreemass[self.model.body("climber_root").id], 78.3)
        self.assertEqual(self.model.numeric("contact_mode").data[0], 0.)
        self.assertEqual(self.model.joint("root").type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(np.all(self.model.jnt_type[self.model.actuator_trnid[:, 0]] == mujoco.mjtJoint.mjJNT_HINGE))
        self.assertFalse(any("foot" in self.model.equality(i).name for i in range(self.model.neq)))
        self.assertEqual(self.scene.start_configuration, scene.start_configuration)
        for name in PHYSICAL_ARRAYS:
            np.testing.assert_array_equal(getattr(self.model, name), getattr(original, name), err_msg=name)
        moved = {self.model.body("contact_" + name).id for name in ("reach_target", "left_reach_target")}
        fixed = [i for i in range(self.model.nbody) if i not in moved]
        np.testing.assert_array_equal(self.model.body_pos[fixed], original.body_pos[fixed])
        for limb, source in scene.start_configuration.items():
            self.assertEqual(self.scene.region(source), scene.region(source))
            geometry = canonical_geometry(self.scene.region(source))
            if limb.is_hand:
                np.testing.assert_allclose(self.data.site(limb.value.lower() + "_site").xpos,
                                           geometry.hand_frame.position, rtol=0, atol=1e-6)
                self.assertEqual(self.manager.contact_snapshot().hands[limb].capacity, 850.)
            else:
                self.assertEqual(geometry.shape, "box")
        for source, target in (("right_hand", "reach_target"), ("left_hand", "left_reach_target")):
            np.testing.assert_allclose(np.array(self.scene.region(target).position) - self.scene.region(source).position,
                                       [0., 0., .25 * profile.arm_reach], rtol=0, atol=1e-15)
            self.assertEqual(replace(self.scene.region(target), position=scene.region(target).position), scene.region(target))
        validate_reference_pose(self.model, self.seed)
        self.assertFalse(np.array_equal(self.seed, seed))
        self.assertTrue(1.15 < self.seed[2] < 1.2)

    def test_preparation_admits_every_four_contact_sample_on_scratch(self):
        before, compiled = _integration_state(self.model, self.data), model_snapshot(self.model)
        admissions = []
        native = static_state.initialize_static_reference

        def admit(model, data, scene, profile, qpos, intent):
            self.assertIsNot(data, self.data)
            self.assertEqual(dict(intent), dict(self.reference.contact_intent))
            result = native(model, data, scene, profile, qpos, intent)
            self.assertEqual(len(result[0].target_pose), model.nu)
            admissions.append(result[0])
            return result

        with patch.object(static_state, "initialize_static_reference", side_effect=admit), \
                patch("mujoco.mj_step", side_effect=AssertionError("geometry stepped physics")):
            frames, path, endpoint = motion.prepare_whole_body(
                self.model, self.scene, self.profile, self.reference, self.goal,
                hand_target=canonical_geometry(self.scene.region("reach_target")).hand_frame,
                limb=Limb.RIGHT_HAND, target="reach_target")
        self.assertEqual(len(path), 81)
        self.assertEqual(len(admissions), len(path))
        self.assertTrue(endpoint.converged, endpoint.reason)
        self.assertEqual(len(endpoint.target_pose), 25)
        for qpos, admitted in zip(path, admissions):
            np.testing.assert_array_equal(qpos, admitted.qpos)
            validate_reference_pose(self.model, qpos)
        self.assertEqual(endpoint.root_target, self.goal.root_target)
        for residual in endpoint.residuals:
            if residual.limb != Limb.RIGHT_HAND:
                self.assertEqual(residual.target, frames[residual.limb])
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        after = model_snapshot(self.model)
        for name, value in compiled.items():
            np.testing.assert_array_equal(after[name], value, err_msg=name)

    def test_invalid_goals_and_typed_contact_maps_reject_atomically(self):
        args = (self.model, self.data, self.scene, self.profile, self.reference, self.manager)
        contacts = dict(self.reference.contact_intent)
        request = transfers.TransferRequest(Limb.RIGHT_HAND, "right_hand", "reach_target", contacts,
                                            hand_reach_s=5., whole_body=self.goal)
        root = self.goal.root_target
        invalid = (
            (replace(request, whole_body=replace(self.goal, root_target=Frame(root.position, ((1., 0., 0.),) * 3))),
             "REACH_INFEASIBLE"),
            (replace(request, whole_body=replace(self.goal, root_target=replace(root, position=(math.nan, 0., 1.)))),
             "REACH_INFEASIBLE"),
            (replace(request, whole_body=replace(self.goal, waist_target={**self.goal.waist_target, "waist_pitch": 3.})),
             "REACH_INFEASIBLE"),
            (replace(request, whole_body=replace(self.goal, root_target=replace(
                root, position=tuple(np.array(root.position) + [0., 0., .4])))), "REACH_INFEASIBLE"),
            (replace(request, target="unreachable_target"), "REACH_INFEASIBLE"),
            (replace(request, source_contacts={limb.value: hold for limb, hold in contacts.items()}), "INVALID_REQUEST"),
            (replace(request, source_contacts={**contacts, Limb.LEFT_FOOT: "left_hand"}), "INVALID_REQUEST"),
            (replace(request, support_contacts=contacts), "INVALID_REQUEST"),
            (replace(request, whole_body={"root_target": root}), "REACH_INFEASIBLE"))
        before = _integration_state(self.model, self.data)
        logs = copy.deepcopy((self.manager.capture_events, self.manager.releases, self.manager.last_capture_failure))
        for bad, status in invalid:
            with self.subTest(status=status, request=bad), \
                    patch("mujoco.mj_step", side_effect=AssertionError("negative integrated")), \
                    patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("negative commanded")):
                result = transfers.execute_transfer(*args, bad)
            self.assertEqual(result["status"], status, result["reason"])
            self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
            self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
            self.assertIsNone(result["final_reference"])
            self.assertEqual(result["initial_state"], result["final_state"])
            if result["terminal_observation"] is not None:
                self.assertEqual(result["terminal_observation"]["status"], status)
                self.assertFalse(result["terminal_observation"]["readiness"]["ready"])
            np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
            self.assertEqual(self.manager.contact_configuration(), contacts)
            self.assertEqual((self.manager.capture_events, self.manager.releases, self.manager.last_capture_failure), logs)
        # Invalid seed quaternions and foot-target mappings must not reach FK.
        with patch("mujoco.mj_forward", side_effect=AssertionError("malformed input reached FK")):
            for quaternion in ((math.nan, 0., 0., 0.), (0., 0., 0., 0.)):
                seed = self.seed.copy()
                seed[3:7] = quaternion
                with self.assertRaises(ValueError):
                    motion.solve_whole_body_reference(self.model, self.scene, self.profile, seed, contacts,
                                                     root_target=root, waist_target=self.goal.waist_target)
            with self.assertRaises(ValueError):
                motion.solve_whole_body_reference(self.model, self.scene, self.profile, self.seed, contacts,
                                                 root_target=root, waist_target=self.goal.waist_target,
                                                 hand_targets={Limb.LEFT_FOOT: root})
        # A real box intersecting the pelvis is not relaxed into a legal source.
        tree = ET.fromstring(build_mjcf(self.scene, self.profile))
        ET.SubElement(tree.find("worldbody"), "geom", name="blocking_box", type="box",
                      pos=" ".join(map(str, self.seed[:3])), size=".1 .09 .04")
        blocked = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        with patch("mujoco.mj_step", side_effect=AssertionError("collision preflight integrated")), self.assertRaises(ValueError):
            motion.prepare_whole_body(blocked, self.scene, self.profile, self.reference, self.goal)
        np.testing.assert_array_equal(_integration_state(self.model, self.data), before)
        self.assertEqual((self.manager.capture_events, self.manager.releases, self.manager.last_capture_failure), logs)


class WholeBodyHandNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls.runs.clear()

    def run_native_case(self, dt):
        if dt not in self.runs:
            sessions, callbacks, captures, ik_times = {}, [], [], []
            native_factory, native_step = demo.make_whole_body_fixture, mujoco.mj_step
            native_control, native_static = single_hand.compute_pose_control, static_control.compute_pose_control
            native_solver, native_ik = motion.solve_whole_body_reference, single_hand.solve_hand_reference
            native_guard, native_attach = GraspManager.evaluate_and_update, GraspManager.attach

            def factory(timestep):
                args = native_factory(timestep)
                model, data = args[:2]
                sessions[id(model)] = {"args": args, "model": model_snapshot(model), "previous": {},
                                       "steps": 0, "controls": 0, "static_controls": 0,
                                       "guards": 0, "solves": 0}
                return args

            def unchanged(model, data):
                session = sessions[id(model)]
                self.assertIs(data, session["args"][1])
                previous = session["previous"].get(id(data))
                if previous is not None:
                    np.testing.assert_array_equal(data.qpos, previous[0])
                    np.testing.assert_array_equal(data.qvel, previous[1])
                    self.assertEqual(float(data.time), previous[2])
                self.assertFalse(np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
                return session

            def step(model, data, *args, **kwargs):
                session = unchanged(model, data)
                # The legitimate source initialization precedes the first step.
                session["previous"].setdefault(id(data), (data.qpos.copy(), data.qvel.copy(), float(data.time)))
                before = float(data.time)
                native_step(model, data, *args, **kwargs)
                self.assertAlmostEqual(float(data.time) - before, dt, delta=1e-12)
                np.testing.assert_array_equal(data.qfrc_actuator[:6], 0.)
                for equality in np.flatnonzero(data.eq_active):
                    rows = ((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                            & (data.efc_id == equality))
                    self.assertEqual(np.count_nonzero(rows), 3)
                    self.assertLessEqual(np.linalg.norm(data.efc_force[rows]), 850.)
                self.assertFalse(np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
                self.assertFalse(any(w.number for w in data.warning))
                session["previous"][id(data)] = (data.qpos.copy(), data.qvel.copy(), float(data.time))
                session["steps"] += 1

            def control(native, model, data, target, *args, static=False, **kwargs):
                session = unchanged(model, data)
                self.assertEqual(set(target), {model.joint(int(j)).name for j in model.actuator_trnid[:, 0]})
                self.assertEqual(len(target), model.nu)
                self.assertFalse({"kp", "kd"} & kwargs.keys())
                result = native(model, data, target, *args, **kwargs)
                unchanged(model, data)
                session["controls"] += 1
                session["static_controls"] += int(static)
                return result

            def solve(model, scene, profile, seed, intent, **kwargs):
                session = sessions[id(model)]
                live = session["args"][1]
                unchanged(model, live)
                self.assertFalse(np.shares_memory(np.asarray(seed), live.qpos))
                before = _integration_state(model, live)
                result = native_solver(model, scene, profile, seed, intent, **kwargs)
                np.testing.assert_array_equal(_integration_state(model, live), before)
                self.assertEqual(len(result.target_pose), model.nu)
                session["solves"] += 1
                return result

            def ik(model, seed, limb, frame, **kwargs):
                live = sessions[id(model)]["args"][1]
                unchanged(model, live)
                np.testing.assert_array_equal(seed, live.qpos)
                before = _integration_state(model, live)
                result = native_ik(model, seed, limb, frame, **kwargs)
                np.testing.assert_array_equal(_integration_state(model, live), before)
                ik_times.append(float(live.time))
                return result

            def guard(manager, *args, **kwargs):
                session = sessions.get(id(manager.model))
                if session is not None and manager.data is session["args"][1]:
                    unchanged(manager.model, manager.data)
                    session["guards"] += 1
                return native_guard(manager, *args, **kwargs)

            def attach(manager, limb, region, force=False):
                session = sessions.get(id(manager.model))
                live = session is not None and manager.data is session["args"][1]
                if live:
                    unchanged(manager.model, manager.data)
                accepted = native_attach(manager, limb, region, force=force)
                if live:
                    self.assertFalse(force)
                    unchanged(manager.model, manager.data)
                    if accepted:
                        event = copy.deepcopy(manager.capture_events[-1])
                        assert_capture(self, event)
                        self.assertLessEqual(event["initial_reaction_N"], 850.)
                        captures.append(event)
                return accepted

            def observer(row, display_model, display_data):
                session = next(iter(sessions.values()))
                model, live = session["args"][:2]
                unchanged(model, live)
                self.assertIsNot(display_model, model)
                self.assertIsNot(display_data, live)
                np.testing.assert_array_equal(row["qpos"], live.qpos)
                np.testing.assert_array_equal(row["qvel"], live.qvel)
                np.testing.assert_array_equal(display_data.qpos, live.qpos)
                np.testing.assert_array_equal(display_data.qvel, live.qvel)
                np.testing.assert_array_equal(_integration_state(display_model, display_data),
                                              _integration_state(model, live))
                callbacks.append(copy.deepcopy(row))
                row["qpos"][0] = 999.
                for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied"):
                    getattr(display_data, name)[:] = 999.
                display_data.eq_active[:] = False
                display_model.body_mass[:] = 999.
                display_model.eq_data[:] = 999.
                display_model.geom_friction[:] = .0001
                display_model.opt.timestep = 1.
                unchanged(model, live)

            with ExitStack() as stack:
                stack.enter_context(patch.object(demo, "make_whole_body_fixture", side_effect=factory))
                stack.enter_context(patch("mujoco.mj_step", side_effect=step))
                stack.enter_context(patch.object(single_hand, "compute_pose_control",
                                                 side_effect=lambda *a, **kw: control(native_control, *a, **kw)))
                stack.enter_context(patch.object(static_control, "compute_pose_control",
                                                  side_effect=lambda *a, **kw: control(native_static, *a, static=True, **kw)))
                stack.enter_context(patch.object(motion, "solve_whole_body_reference", side_effect=solve))
                stack.enter_context(patch.object(single_hand, "solve_hand_reference", side_effect=ik))
                stack.enter_context(patch.object(GraspManager, "evaluate_and_update", new=guard))
                stack.enter_context(patch.object(GraspManager, "attach", new=attach))
                result = demo.run_whole_body_benchmark(dt, kind="sequence", observer=observer)
            session = next(iter(sessions.values()))
            unchanged(*session["args"][:2])
            self.runs[dt] = result, session, callbacks, captures, ik_times

        result, session, callbacks, captures, ik_times = self.runs[dt]
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual((result["status"], result["completed_moves"], result["failed_index"]), ("SUCCESS", 2, None))
        model, data, scene, profile, _ = session["args"]
        after = model_snapshot(model)
        for name, value in session["model"].items():
            np.testing.assert_array_equal(after[name], value, err_msg=name)
        first, second = result["moves"]
        self.assertTrue(result["initial_static"]["success"])
        self.assertEqual(first["initial_state"], result["initial_static"]["final_state"])
        self.assertEqual(first["final_state"], second["initial_state"])
        self.assertEqual(first["final_integration_state"], second["initial_integration_state"])
        self.assertIs(result["final_reference"], second["final_reference"])
        self.assertTrue(np.any(second["initial_state"].qvel[:6]))
        self.assertEqual(session["steps"], result["total_steps"] + result["initial_static"]["steps"])
        # Static admission probes the controller once before the native loop.
        self.assertEqual(session["static_controls"], result["initial_static"]["steps"] + 1)
        self.assertEqual(session["controls"] - session["static_controls"], result["total_steps"])
        self.assertGreaterEqual(session["guards"], 2 * result["total_steps"])
        self.assertGreater(session["solves"], 160)
        self.assertTrue(ik_times)
        self.assertEqual(result["total_steps"], sum(move["steps"] for move in result["moves"]))
        self.assertAlmostEqual(result["duration_s"], result["total_steps"] * dt, delta=1e-12)
        self.assertAlmostEqual(data.time - result["initial_time_s"], result["duration_s"], delta=1e-9)
        self.assertEqual(len(captures), 4)
        self.assertEqual(result["initial_state"], first["initial_state"])
        self.assertEqual(result["final_state"], second["final_state"])
        np.testing.assert_array_equal(_integration_state(model, data), result["final_integration_state"])
        self.assertFalse(any("foot" in model.equality(i).name for i in range(model.neq)))
        joints = model.actuator_trnid[:, 0]
        qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        previous, velocity = np.array(result["initial_reference"].qpos), np.zeros(model.nv)
        metrics = []
        for index, move in enumerate(result["moves"]):
            self.assertTrue(move["success"], (move["status"], move["reason"]))
            self.assertEqual(move["moving_limb"], (Limb.RIGHT_HAND, Limb.LEFT_HAND)[index])
            self.assertTrue(move["released"] and move["admission"]["admitted"])
            self.assertTrue(move["readiness"]["ready"])
            self.assertGreaterEqual(move["readiness"]["duration"], .5 - 1e-10)
            self.assertIsNone(move["guard_failure"])
            self.assertEqual(move["phases"], ["SOURCE_STABILIZE", "LOAD_TRANSFER", "RELEASE_CLEARANCE",
                                             "RELEASE", "THREE_POINT", "REACH", "SETTLE"])
            self.assertEqual(move["final_contacts"], {**move["source_contacts"], move["moving_limb"]: move["target"]})
            self.assertEqual(dict(move["final_reference"].contact_intent), move["final_contacts"])
            assert_capture(self, move["capture"])
            self.assertLess(move["release_time_s"], move["reach_start_time_s"])
            self.assertLess(move["reach_start_time_s"], move["capture_time_s"])
            self.assertLess(move["capture_time_s"], move["readiness_time_s"])
            self.assertLess(move["actuator_utilization_max"], 1.)
            rows = move["samples"]
            self.assertEqual(len(rows), move["steps"])
            unload = [row for row in rows if row["phase"] == "LOAD_TRANSFER"]
            self.assertAlmostEqual(len(unload) * dt, 4.)
            moving = move["moving_limb"].value
            other = (Limb.LEFT_HAND if index == 0 else Limb.RIGHT_HAND).value
            if index == 0:
                self.assertGreater(unload[0]["hands"][moving]["load"], 30.)
                self.assertLess(unload[-1]["hands"][moving]["load"], 2.)
                self.assertGreater(unload[-1]["hands"][other]["load"], 1.8 * unload[0]["hands"][other]["load"])
            actual = np.array([row["qpos"] for row in unload])
            for name in ("waist_pitch", "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee"):
                change = np.ptp(actual[:, int(model.joint(name).qposadr[0])])
                self.assertGreater(change, math.radians(3.) if name == "waist_pitch" and index == 0 else .01, name)
            roots = np.array([row["qpos"][:3] for row in rows])
            displacement = np.linalg.norm(roots - move["initial_state"].qpos[:3], axis=1)
            if index == 0:
                self.assertGreater(displacement.max(), .03)
                rotation = np.array([row["qpos"][3:7] for row in unload])
                angle = 2. * np.arccos(np.clip(np.abs(rotation @ rotation[0]), 0., 1.))
                self.assertGreater(angle.max(), math.radians(3.))
            for row in rows:
                self.assertFalse(np.any(row["external_force_world_N"]) or np.any(row["qfrc_applied"]))
                self.assertLessEqual(row["root_linear_m_s"], .1)
                self.assertLessEqual(row["root_angular_rad_s"], .5)
                self.assertLessEqual(row["joint_max_rad_s"], 1.)
                self.assertTrue(row["hands"][other]["active"] and row["hands"][other]["valid"])
                for hand in row["hands"].values():
                    if hand["active"]:
                        self.assertTrue(hand["valid"])
                        self.assertLessEqual(hand["load"], hand["capacity"])
                for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
                    foot = row["feet"][limb.value]
                    self.assertTrue(foot["measurement_valid"] and foot["supporting"])
                    self.assertFalse(foot["slipping"])
                    self.assertGreater(foot["normal_force"], 5.)
                    self.assertEqual(foot["support_surfaces"], ("geom_" + move["source_contacts"][limb],))
                    for contact in foot["contacts"]:
                        if contact["normal_force"] > FORCE_TOLERANCE:
                            self.assertTrue(contact["admissible"])
                            self.assertLessEqual(contact["tangential_speed"], MAX_SLIP_SPEED)
                            self.assertLessEqual(contact["friction_utilization"], 1. + 1e-8)
                q, qd = np.array(row["q_ref"]), np.array(row["qd_ref"])
                validate_reference_pose(model, q)
                self.assertLessEqual(np.max(np.abs(qd[vi])), .5 + 1e-10)
                self.assertLessEqual(np.max(np.abs(qd[vi] - velocity[vi])), 2. * dt + 1e-10)
                np.testing.assert_allclose((q[qi] - previous[qi]) / dt, qd[vi], rtol=0, atol=1e-10)
                previous, velocity = q, qd
            hand_travel = np.linalg.norm(np.array(rows[-1]["actual_hand_position_world_m"]) -
                                         rows[0]["actual_hand_position_world_m"])
            self.assertAlmostEqual(hand_travel, .25 * profile.arm_reach, delta=.003)
            reach = [row for row in rows if row["phase"] == "REACH"]
            desired = np.array([row["desired_hand_frame"]["position"] for row in reach])
            normal = np.array(canonical_geometry(scene.region(move["target"])).hand_frame.normal)
            self.assertGreater(np.ptp(desired @ normal), .01)
            terminal = {key: value for key, value in next(row for row in reversed(callbacks)
                        if row["move_index"] == index).items() if key not in ("move_index", "move_count")}
            self.assertEqual(terminal, move["terminal_observation"])
            for name in ("qpos", "qvel", "ctrl", "eq_active"):
                np.testing.assert_array_equal(terminal[name], getattr(move["final_state"], name))
            np.testing.assert_allclose(terminal["q_ref"], move["final_reference"].qpos, rtol=0, atol=1e-10)
            np.testing.assert_allclose(terminal["qd_ref"], 0., rtol=0, atol=1e-8)
            self.assertGreater(min(unload[-1]["feet"][limb.value]["normal_force"]
                                   for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)), 10.)
            metrics.append(f"{moving}: travel_mm={hand_travel * 1000:.3f} "
                           f"root_max_mm={displacement.max() * 1000:.3f} "
                           f"unload_N={unload[-1]['hands'][moving]['load']:.3f}/"
                           f"{unload[-1]['hands'][other]['load']:.3f} "
                           f"capture_gap_mm={move['capture_error_m'] * 1000:.3f}")
        rise = result["final_state"].qpos[2] - result["initial_state"].qpos[2]
        self.assertGreater(rise, .06)
        roots = np.array([row["qpos"][:3] for move in result["moves"] for row in move["samples"]])
        total_max = np.linalg.norm(roots - result["initial_state"].qpos[:3], axis=1).max()
        self.assertGreater(total_max, .06)
        for index in (0, 1):
            regular = [row for row in callbacks if row["move_index"] == index and not row["terminal"]]
            np.testing.assert_allclose(np.diff([row["time_s"] for row in regular]), .05, rtol=0, atol=1e-12)
        print(f"wholebody hands dt={dt}: steps={result['total_steps']} duration={result['duration_s']:.3f}s "
              f"rise_mm={rise * 1000:.3f} total_root_max_mm={total_max * 1000:.3f}; "
              + "; ".join(metrics), flush=True)

    def test_native_sequence_2ms(self):
        self.run_native_case(.002)

    def test_native_sequence_1ms(self):
        self.run_native_case(.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
