"""Single-hand native release/reach/capture, without relaxed Stage3 physics."""

import copy
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1 import single_hand
from boulder_v1.contact import CAPTURE_DISTANCE, CAPTURE_ORIENTATION, CAPTURE_SPEED
from boulder_v1.contact_benchmarks import make_mixed_fixture
from boulder_v1.contact_geometry import ContactMode, Frame, canonical_geometry
from boulder_v1.grasp import AttachmentStateError, GraspManager
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.runtime import validate_reference_pose
from boulder_v1.schema import Limb
from boulder_v1.static_state import ReadinessTracker
from boulder_v1.support import FORCE_TOLERANCE, FootStatus, fresh_data


INTENT = {limb: limb.value.lower() for limb in Limb}
NEW_INTENT = {**INTENT, Limb.RIGHT_HAND: "reach_target"}
FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
TIMESTEPS = (.002, .001)
NEGATIVES = ("unreachable", "orientation_invalid", "support_loss", "grip_after_capture")
PHASES = ["SOURCE_STABILIZE", "LOAD_TRANSFER", "RELEASE_CLEARANCE", "RELEASE",
          "THREE_POINT", "REACH", "SETTLE"]
GAINS = {"waist": (160., 25.), "shoulder": (80., 12.), "elbow": (60., 6.),
         "wrist": (20., 1.), "hip": (160., 25.), "knee": (120., 12.), "ankle": (40., 2.)}


def integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def model_parameters(model):
    return {f"{label}.{name}": value.copy() if isinstance(value, np.ndarray) else value
            for label, owner in (("model", model), ("option", model.opt), ("stat", model.stat))
            for name in dir(owner) if not name.startswith("_")
            if isinstance(value := getattr(owner, name), (np.ndarray, float, int, bool, str))}


def capture_gate(measurement):
    return (measurement["gap_m"] <= .001 + 1e-12
            and measurement["orientation"] >= math.cos(math.radians(30))
            and measurement["relative_speed_m_s"] <= .05 + 1e-12
            and measurement["penetration_m"] < .001
            and measurement["signed_geom_distance_m"] <= .001 + 1e-12)


def collision_loads(model, data):
    loads = {}
    for index, contact in enumerate(data.contact):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        pair = frozenset(model.geom(int(g)).name for g in (contact.geom1, contact.geom2))
        loads[pair] = loads.get(pair, 0.) + float(np.linalg.norm(wrench[:3]))
    return loads


class NativeAudit:
    """Audit the authoritative data, not the deliberately detached observer data."""

    def __init__(self, model, data, scene, profile, reference, manager, request, fault):
        self.model, self.data, self.scene, self.profile = model, data, scene, profile
        self.reference, self.manager, self.request, self.fault = reference, manager, request, fault
        self.identity = (id(model), id(data), id(scene), id(profile), id(reference), id(manager))
        self.before_model = model_parameters(model)
        self.reference_before = (reference.qpos, dict(reference.target_pose), dict(reference.contact_intent))
        self.start_data = mujoco.MjData(model)
        mujoco.mj_copyData(self.start_data, model, data)
        self.start_captures = copy.deepcopy(manager.capture_events)
        self.start_releases = copy.deepcopy(manager.releases)
        self.previous = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        self.violations, self.steps, self.controls, self.guards, self.samples = set(), [], [], [], []
        self.captures, self.detaches, self.eligibility, self.ik = [], [], [], []
        self.force_guards = []
        self.shaped_goals = []
        joints = model.actuator_trnid[:, 0]
        self.qi, self.vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]

    def record(self, condition, message):
        if not condition:
            self.violations.add(message)

    def unchanged(self):
        qpos, qvel, time = self.previous
        self.record(np.array_equal(self.data.qpos, qpos), "qpos written between native steps")
        self.record(np.array_equal(self.data.qvel, qvel), "qvel written between native steps")
        self.record(self.data.time == time, "time written between native steps")

    def install(self, stack):
        native_step, native_control = mujoco.mj_step, single_hand.compute_pose_control
        native_guard, native_sample = GraspManager.evaluate_and_update, ReadinessTracker.sample_after_step
        native_detach, native_can_attach = GraspManager.detach, GraspManager.can_attach
        native_ik = single_hand.solve_hand_reference
        native_shape = single_hand._shape_reference

        def step(model, data, *args, **kwargs):
            self.record(model is self.model and data is self.data, "stepped a planning/observer episode")
            self.unchanged()
            self.record(len(self.controls) == len(self.steps) + 1, "missing hinge torque command")
            self.record(self.guards[-1:] == [("pre", float(data.time))], "missing pre-step grasp guard")
            attachments = self.manager.active_attachments()
            flags = data.eq_active.copy()
            allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{att.region.id}"))
                       for limb, att in attachments.items()}
            allowed.update(frozenset((f"{limb.value.lower()}_geom", f"geom_{INTENT[limb]}")) for limb in FEET)
            forces = [(model.body(int(body)).name, tuple(data.xfrc_applied[body]))
                      for body in np.flatnonzero(np.any(data.xfrc_applied != 0., axis=1))]
            self.record(not np.any(data.qfrc_applied), "injected generalized/root force")
            expected_force = {"support_loss": ("left_foot", (1500., 0., 0., 0., 0., 0.)),
                              "grip_after_capture": ("right_hand", (0., -2000., 0., 0., 0., 0.))}
            self.record(not forces or forces == [expected_force.get(self.fault)], "undeclared body wrench")
            before = float(data.time)
            native_step(model, data, *args, **kwargs)
            self.record(abs(data.time - before - model.opt.timestep) < 1e-12, "discontinuous native clock")
            self.record(not np.any(data.qfrc_actuator[:6]), "motor applied root wrench")
            self.record(np.allclose(data.qfrc_actuator[self.vi], data.ctrl * model.actuator_gear[:, 0],
                                    rtol=0, atol=1e-10), "native hinge torques differ from command")
            self.record(all(np.isfinite(v).all() for v in
                            (data.qpos, data.qvel, data.qacc, data.efc_force)), "nonfinite native solve")
            self.record(not any(w.number for w in data.warning), "native recovery/warning")
            loads = collision_loads(model, data)
            unexpected = {tuple(sorted(pair)): load for pair, load in loads.items()
                          if pair not in allowed and load > FORCE_TOLERANCE}
            self.record(not unexpected, "unintended applied body/contact support")
            endpoint = collision_loads(model, fresh_data(model, data))
            self.record(not any(pair not in allowed and load > FORCE_TOLERANCE
                                for pair, load in endpoint.items()), "unintended endpoint body/contact support")
            hand_loads = {}
            for limb, att in attachments.items():
                rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                      & (data.efc_id == att.eq_id))
                self.record(len(rows) == 3, "active hand lacks native three-row point grasp")
                hand_loads[limb.value] = float(np.linalg.norm(data.efc_force[rows]))
                if self.fault is None:
                    self.record(hand_loads[limb.value] <= 850., "applied grasp exceeds unchanged 850N bound")
            self.steps.append({"before_s": before, "time_s": float(data.time), "eq_active": flags.tolist(),
                               "hands_N": hand_loads, "loads": loads, "forces": forces,
                               "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy()})
            self.previous = (data.qpos.copy(), data.qvel.copy(), float(data.time))

        def control(model, data, target, *args, **kwargs):
            self.record(model is self.model and data is self.data, "controlled a nonauthoritative episode")
            self.unchanged()
            self.record(not args and not ({"kp", "kd"} & kwargs.keys()), "overrode Stage3 gains")
            self.record(set(target) == {model.joint(int(j)).name for j in model.actuator_trnid[:, 0]},
                        "command was not exactly 25 hinge references")
            qpos, qvel = data.qpos.copy(), data.qvel.copy()
            command = native_control(model, data, target, *args, **kwargs)
            self.record(np.array_equal(qpos, data.qpos) and np.array_equal(qvel, data.qvel),
                        "controller wrote generalized state")
            ff, qd = kwargs.get("feedforward", {}), kwargs.get("target_velocity", {})
            desired = []
            for index, name in enumerate(command.joint_names):
                kind = name.removeprefix("left_").removeprefix("right_").split("_")[0]
                kp, kd = GAINS[kind]
                self.record((command.stiffness_Nm_rad[index], command.damping_Nms_rad[index]) == (kp, kd),
                            "changed physical controller gains")
                desired.append(kp * (target[name] - qpos[self.qi[index]])
                               + kd * (qd.get(name, 0.) - qvel[self.vi[index]]) + ff.get(name, 0.))
            self.record(np.allclose(command.desired_Nm, desired, rtol=0, atol=1e-12), "not physical PD/FF torque")
            self.record(np.allclose(data.ctrl * model.actuator_gear[:, 0], command.commanded_Nm,
                                    rtol=0, atol=1e-12), "control/Nm normalization mismatch")
            self.controls.append({"target": dict(target), "velocity": dict(qd), "command": command})
            return command

        def guard(manager, *args, **kwargs):
            if manager is self.manager:
                epoch = ("applied" if kwargs.get("applied_data") is self.data else
                         "post_activation" if len(self.controls) == len(self.steps) and self.captures
                         and self.captures[-1]["measurement"]["time_s"] == self.data.time else "pre")
                self.guards.append((epoch, float(self.data.time)))
                for body in np.flatnonzero(np.any(self.data.xfrc_applied != 0., axis=1)):
                    self.force_guards.append({"epoch": epoch, "time_s": float(self.data.time),
                                              "body": self.model.body(int(body)).name,
                                              "force": tuple(self.data.xfrc_applied[body])})
            return native_guard(manager, *args, **kwargs)

        def shape(model, goal, previous, previous_velocity, dt):
            self.unchanged()
            self.shaped_goals.append(np.array(goal))
            result = native_shape(model, goal, previous, previous_velocity, dt)
            self.unchanged()
            return result

        def sample(tracker, before):
            if tracker.data is self.data:
                self.record(self.guards[-1:] == [("applied", float(self.data.time))],
                            "sampled readiness before applied guard")
                self.samples.append((before, float(self.data.time)))
            return native_sample(tracker, before)

        def detach(manager, limb):
            if manager is self.manager:
                self.unchanged()
                before = self.data.eq_active.copy()
                attachment = manager.active_attachments().get(limb)
                pair = frozenset((f"{limb.value.lower()}_geom", f"geom_{attachment.region.id}")) if attachment else None
                applied = collision_loads(self.model, self.data).get(pair, 0.)
                endpoint = collision_loads(self.model, fresh_data(self.model, self.data)).get(pair, 0.)
                accepted = native_detach(manager, limb)
                self.detaches.append({"limb": limb, "time_s": float(self.data.time), "before": before,
                                      "after": self.data.eq_active.copy(), "attachment": attachment,
                                      "applied_palm_N": applied, "endpoint_palm_N": endpoint, "accepted": accepted})
                self.unchanged()
                return accepted
            return native_detach(manager, limb)

        def can_attach(manager, limb, region, *args, **kwargs):
            accepted = native_can_attach(manager, limb, region, *args, **kwargs)
            if manager is self.manager and region.id == self.request.target:
                self.eligibility.append((float(self.data.time), accepted,
                                         manager._capture_measurement(limb, region)))
            return accepted

        def ik(model, measured, limb, target, *args, **kwargs):
            self.unchanged()
            self.record(np.shares_memory(measured, self.data.qpos), "IK seed was not measured live pose")
            seed = np.array(measured)
            result = native_ik(model, measured, limb, target, *args, **kwargs)
            side = limb.value.lower().removesuffix("_hand")
            arm = [model.joint(f"{side}_{name}").id
                   for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
            frozen = np.ones(model.nq, dtype=bool)
            frozen[model.jnt_qposadr[arm]] = False
            self.record(np.array_equal(np.array(result.qpos)[frozen], seed[frozen]), "arm IK rewrote root/support frame")
            validate_reference_pose(model, result.qpos)
            if result.converged:
                probe = mujoco.MjData(model)
                probe.qpos[:] = result.qpos
                mujoco.mj_forward(model, probe)
                site = probe.site(f"{limb.value.lower()}_site")
                self.record(np.linalg.norm(site.xpos - target.position) <= 1e-7, "IK fabricated positional convergence")
                self.record(np.linalg.norm(site.xmat.reshape(3, 3)[:, 2] - target.normal) <= 1e-7,
                            "IK fabricated facing convergence")
            self.ik.append((float(self.data.time), result.converged, result.iterations))
            self.unchanged()
            return result

        # Scratch admission legitimately resets disposable data even during SETTLE.
        for name in ("mj_resetData", "mj_resetDataKeyframe", "mj_forward", "mj_setState"):
            native = getattr(mujoco, name)

            def scratch_only(model, data, *args, _native=native, _name=name, **kwargs):
                self.record(data is not self.data, f"{_name} targeted live integration data")
                if data is self.data:
                    raise AssertionError(f"{_name} targeted live data after initialization")
                return _native(model, data, *args, **kwargs)

            stack.enter_context(patch.object(mujoco, name, new=scratch_only))
        native_copy, native_integrate, native_normalize = mujoco.mj_copyData, mujoco.mj_integratePos, mujoco.mj_normalizeQuat

        def copy_data(destination, model, source):
            self.record(destination is not self.data, "scratch/observer data copied into live episode")
            return native_copy(destination, model, source)

        def integrate(model, qpos, tangent, dt):
            self.record(not np.shares_memory(qpos, self.data.qpos), "integrated a virtual reference into live qpos")
            return native_integrate(model, qpos, tangent, dt)

        def normalize(model, qpos):
            self.record(not np.shares_memory(qpos, self.data.qpos), "renormalized live root outside mj_step")
            return native_normalize(model, qpos)

        stack.enter_context(patch("mujoco.mj_copyData", new=copy_data))
        stack.enter_context(patch("mujoco.mj_integratePos", new=integrate))
        stack.enter_context(patch("mujoco.mj_normalizeQuat", new=normalize))
        stack.enter_context(patch("mujoco.mj_step", new=step))
        stack.enter_context(patch.object(single_hand, "compute_pose_control", new=control))
        stack.enter_context(patch.object(single_hand, "solve_hand_reference", new=ik))
        stack.enter_context(patch.object(single_hand, "_shape_reference", new=shape))
        stack.enter_context(patch.object(GraspManager, "evaluate_and_update", new=guard))
        stack.enter_context(patch.object(GraspManager, "detach", new=detach))
        stack.enter_context(patch.object(GraspManager, "can_attach", new=can_attach))
        stack.enter_context(patch.object(ReadinessTracker, "sample_after_step", new=sample))


class SingleHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.audits, cls.run_errors = {}, {}, {}
        for dt, scenario in [(dt, "success") for dt in TIMESTEPS] + [(.002, name) for name in NEGATIVES]:
            key = (dt, scenario)
            try:
                result, audit = cls.audited_benchmark(dt, scenario=scenario)
                cls.results[key], cls.audits[key] = result, audit
                rows = result.get("samples", [])
                velocities = np.array([r["qd_ref"] for r in rows])[:, audit.vi] if rows else np.zeros((0, audit.model.nu))
                metrics = {"dt_s": dt, "scenario": scenario, "status": result["status"],
                           "steps": result.get("steps"), "duration_s": result.get("duration_s"),
                           "final_time_s": result["final_state"].time, "capture": result.get("capture"),
                           "remaining_hand_peak_measured_N": max((r["hands"]["LEFT_HAND"]["load"] for r in rows), default=0.),
                           "remaining_hand_peak_reported_N": result.get("remaining_hand_load_max_N"),
                           "actuator_utilization_max": result.get("actuator_utilization_max"),
                           "foot_support_fraction": result.get("foot_support_fraction"),
                           "foot_slip_max_m_s": result.get("foot_slip_max_m_s"),
                           "readiness": result.get("readiness"), "preflight": result.get("preflight"),
                           "reference_velocity_max_rad_s": float(np.max(np.abs(velocities), initial=0.)),
                           "reference_acceleration_max_rad_s2": float(np.max(np.abs(np.diff(
                               np.vstack((np.zeros((1, audit.model.nu)), velocities)), axis=0)), initial=0.) / dt),
                           "release_events": result.get("release_events"), "audit_violations": sorted(audit.violations)}
                print("SINGLE_HAND_METRICS " + json.dumps(metrics, allow_nan=False), flush=True)
            except Exception as error:
                cls.run_errors[key] = f"{type(error).__name__}: {error}"
                print("SINGLE_HAND_METRICS " + json.dumps({"dt_s": dt, "scenario": scenario,
                                                         "error": cls.run_errors[key]}), flush=True)

    @staticmethod
    def audited_benchmark(dt, *, scenario="success", observer=None):
        audits, captures, violations = [], [], set()
        native_execute, native_attach = single_hand.execute_single_hand, GraspManager.attach

        def attach(manager, limb, region, force=False):
            before_state = (manager.data.qpos.copy(), manager.data.qvel.copy(), float(manager.data.time))
            before = manager.data.eq_active.copy()
            measurement = manager._capture_measurement(limb, region)
            attachment = manager.registered_attachment(limb, region)
            candidate = fresh_data(manager.model, manager.data)
            previous = manager.active_attachments().get(limb)
            if previous:
                candidate.eq_active[previous.eq_id] = False
            candidate.eq_active[attachment.eq_id] = True
            mujoco.mj_forward(manager.model, candidate)
            rows = np.flatnonzero((candidate.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                  & (candidate.efc_id == attachment.eq_id))
            reaction = float(np.linalg.norm(candidate.efc_force[rows]))
            accepted = native_attach(manager, limb, region, force=force)
            unchanged = (np.array_equal(before_state[0], manager.data.qpos)
                         and np.array_equal(before_state[1], manager.data.qvel) and before_state[2] == manager.data.time)
            expected = before.copy()
            if accepted:
                if previous:
                    expected[previous.eq_id] = False
                expected[attachment.eq_id] = True
            if force or not limb.is_hand or not unchanged or not np.array_equal(expected, manager.data.eq_active):
                violations.add("attachment forced/rewrote physics/activated unintended equality")
            if accepted and (not capture_gate(measurement) or len(rows) != 3 or reaction > 850.):
                violations.add("accepted attachment outside strict geometry/candidate-capacity gate")
            record = {"owner": manager, "limb": limb, "region": region.id, "force": force,
                      "accepted": accepted, "measurement": measurement, "candidate_reaction_N": reaction,
                      "eq_id": attachment.eq_id, "before": before, "after": manager.data.eq_active.copy()}
            captures.append(record)
            if audits and manager is audits[-1].manager:
                audits[-1].captures.append(record)
            return accepted

        def execute(model, data, scene, profile, reference, manager, request, **kwargs):
            audit = NativeAudit(model, data, scene, profile, reference, manager, request, kwargs.get("fault"))
            audits.append(audit)
            with ExitStack() as stack:
                audit.install(stack)
                result = native_execute(model, data, scene, profile, reference, manager, request, **kwargs)
                audit.unchanged()
            audit.after_model = model_parameters(model)
            audit.all_captures = captures
            audit.violations.update(violations)
            return result

        with patch.object(single_hand, "execute_single_hand", new=execute), \
                patch.object(GraspManager, "attach", new=attach):
            result = single_hand.run_single_hand_benchmark(dt, scenario=scenario, observer=observer)
        if len(audits) != 1:
            raise AssertionError(f"benchmark did not enter exactly one real executor: {result['reason']}")
        return result, audits[0]

    def success_runs(self):
        for dt in TIMESTEPS:
            key = (dt, "success")
            self.assertNotIn(key, self.run_errors, self.run_errors.get(key))
            yield dt, self.results[key], self.audits[key]

    def continuation(self):
        self.assertNotIn((.002, "success"), self.run_errors)
        audit = self.audits[.002, "success"]
        model = copy.copy(audit.model)
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, audit.start_data)
        manager = GraspManager(model, data, audit.scene, profile=audit.profile)
        manager.synchronize_from_live()
        manager.capture_events = copy.deepcopy(audit.start_captures)
        manager.releases = copy.deepcopy(audit.start_releases)
        return model, data, audit.scene, audit.profile, audit.reference, manager

    def test_success_protocol_clock_counts_and_identity(self):
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                self.assertTrue(result["success"], result["reason"])
                self.assertEqual(result["status"], "SUCCESS")
                self.assertEqual(result["phases"], PHASES)
                self.assertTrue(result["initial_static"]["success"])
                self.assertTrue(result["initial_static"]["readiness"]["ready"])
                self.assertTrue(result["retarget"].admitted)
                self.assertAlmostEqual(result["initial_state"].time, 2., places=10)
                self.assertEqual(result["initial_state"], result["initial_static"]["final_state"])
                self.assertTrue(result["initial_state"].finite and result["final_state"].finite)
                self.assertEqual(result["initial_state"].contact_mode, ContactMode.PHYSICAL)
                self.assertEqual(result["initial_contacts"], INTENT)
                self.assertEqual(result["new_contacts"], NEW_INTENT)
                self.assertEqual(result["final_state"].contact_configuration, NEW_INTENT)
                self.assertEqual(result["steps"], len(result["samples"]))
                self.assertEqual(result["steps"], len(audit.steps))
                self.assertEqual(result["dt_s"], dt)
                self.assertEqual(result["duration_s"], result["steps"] * dt)
                self.assertAlmostEqual(result["final_state"].time - result["initial_state"].time,
                                       result["duration_s"], places=9)
                request = audit.request
                self.assertLessEqual(result["duration_s"], 2. + request.transfer_s + .5 + request.support_s
                                     + request.reach_s + request.capture_timeout_s + request.settle_timeout_s)
                self.assertGreaterEqual(result["final_state"].time - result["capture"]["time_s"], .5 - 1e-12)
                np.testing.assert_allclose([r["elapsed_s"] for r in result["samples"]],
                                           np.arange(1, result["steps"] + 1) * dt, rtol=0, atol=2e-9)
                self.assertEqual(audit.identity, (id(audit.model), id(audit.data), id(audit.scene),
                                                  id(audit.profile), id(audit.reference), id(audit.manager)))
                self.assertEqual(result["terminal_observation"]["qpos"], list(result["final_state"].qpos))
                self.assertEqual(result["terminal_observation"]["qvel"], list(result["final_state"].qvel))
                for phase, duration in (("SOURCE_STABILIZE", .5 + dt), ("LOAD_TRANSFER", 1.),
                                        ("RELEASE_CLEARANCE", .5), ("THREE_POINT", .5)):
                    rows = [r for r in result["samples"] if r["phase"] == phase]
                    self.assertEqual(len(rows), round(duration / dt), phase)
                self.assertFalse(any(r["phase"] == "RELEASE" for r in result["samples"]))
                source_ready = next(e for e in result["events"] if e.get("event") == "SOURCE_READY")
                ready_row = result["samples"][source_ready["step"] - 1]
                self.assertTrue(ready_row["readiness"]["ready"])
                self.assertGreaterEqual(ready_row["readiness"]["duration"], .5 - 1e-12)

    def test_only_right_hand_released_after_actual_palm_unload(self):
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                self.assertEqual(len(audit.detaches), 1)
                release = audit.detaches[0]
                self.assertEqual(release["limb"], Limb.RIGHT_HAND)
                self.assertTrue(release["accepted"])
                eq = release["attachment"].eq_id
                self.assertTrue(release["before"][eq])
                self.assertFalse(release["after"][eq])
                self.assertEqual(np.flatnonzero(release["before"] != release["after"]).tolist(), [eq])
                self.assertLessEqual(release["applied_palm_N"], FORCE_TOLERANCE)
                self.assertLessEqual(release["endpoint_palm_N"], FORCE_TOLERANCE)
                clearance = [r for r in result["samples"] if r["phase"] == "RELEASE_CLEARANCE"]
                self.assertTrue(all(r["eq_active"][eq] and r["hands"]["RIGHT_HAND"]["active"]
                                    and r["hands"]["RIGHT_HAND"]["valid"] for r in clearance))
                source_pair = frozenset(("right_hand_geom", "geom_right_hand"))
                before = [s for s in audit.steps if s["time_s"] <= clearance[0]["time_s"]]
                self.assertGreater(max(s["loads"].get(source_pair, 0.) for s in before), FORCE_TOLERANCE)
                unloaded = [s for s in audit.steps if clearance[0]["time_s"] <= s["time_s"] <= release["time_s"]
                            and s["loads"].get(source_pair, 0.) <= FORCE_TOLERANCE and s["eq_active"][eq]]
                self.assertTrue(unloaded, "source collision must unload while point grasp remains active")
                self.assertEqual(result["release_events"], [])
                capture = result["capture"]
                left = audit.model.equality("grasp_left_hand_left_hand").id
                target = audit.model.equality("grasp_right_hand_reach_target").id
                for row in result["samples"]:
                    self.assertTrue(row["eq_active"][left])
                    self.assertEqual(row["contacts"].get("LEFT_HAND"), "left_hand")
                    if row["time_s"] <= release["time_s"]:
                        self.assertEqual(sum(row["eq_active"]), 2)
                        self.assertTrue(row["eq_active"][eq])
                        self.assertFalse(row["eq_active"][target])
                    elif row["time_s"] <= capture["time_s"]:
                        self.assertEqual(sum(row["eq_active"]), 1)
                        self.assertFalse(row["eq_active"][eq] or row["eq_active"][target])
                        self.assertFalse(row["hands"]["RIGHT_HAND"]["active"])
                        self.assertNotIn("RIGHT_HAND", row["contacts"])
                    else:
                        self.assertEqual(sum(row["eq_active"]), 2)
                        self.assertFalse(row["eq_active"][eq])
                        self.assertTrue(row["eq_active"][target])
                        self.assertEqual(row["contacts"].get("RIGHT_HAND"), "reach_target")

    def test_both_feet_have_real_loaded_own_surfaces_and_no_slip(self):
        for dt, result, audit in self.success_runs():
            for limb in FEET:
                with self.subTest(dt=dt, limb=limb):
                    feet = [r["feet"][limb.value] for r in result["samples"]]
                    self.assertTrue(all(f["measurement_valid"] and f["contacting"] and f["supporting"]
                                        and not f["slipping"] and f["status"] == FootStatus.SUPPORTING for f in feet))
                    self.assertTrue(all(f["normal_force"] > 5. and f["tangential_speed"] <= .01 for f in feet))
                    self.assertTrue(all(f["support_regions"] == (INTENT[limb],)
                                        and f["support_surfaces"] == (f"geom_{INTENT[limb]}",)
                                        and f["idealized_attachment"] is None for f in feet))
                    for foot in feet:
                        for contact in foot["contacts"]:
                            if contact["normal_force"] > FORCE_TOLERANCE:
                                self.assertTrue(contact["admissible"])
                                self.assertGreaterEqual(contact["sole_alignment"], .9)
                                self.assertGreaterEqual(contact["surface_alignment"], .9)
                                self.assertLessEqual(contact["tangential_force"],
                                                     contact["friction"] * contact["normal_force"] + FORCE_TOLERANCE)
                                self.assertLessEqual(contact["tangential_speed"], .01)
                    pair = frozenset((f"{limb.value.lower()}_geom", f"geom_{INTENT[limb]}"))
                    self.assertGreater(min(s["loads"].get(pair, 0.) for s in audit.steps), 5.)
                    self.assertEqual(result["foot_support_fraction"][limb.value], 1.)
                    self.assertEqual(result["foot_slip_max_m_s"][limb.value], max(f["tangential_speed"] for f in feet))
                    self.assertLessEqual(result["foot_slip_max_m_s"][limb.value], .01)
                self.assertLessEqual(max(r["hands"]["LEFT_HAND"]["load"] for r in result["samples"]), 90.)

    def test_every_actual_attachment_uses_strict_gate_and_native_candidate_capacity(self):
        self.assertEqual((CAPTURE_DISTANCE, CAPTURE_SPEED), (.001, .05))
        self.assertEqual(CAPTURE_ORIENTATION, math.cos(math.radians(30)))
        for key, audit in self.audits.items():
            with self.subTest(dt=key[0], scenario=key[1]):
                self.assertTrue(audit.all_captures)
                for call in audit.all_captures:
                    self.assertFalse(call["force"])
                    self.assertTrue(call["limb"].is_hand)
                    if call["accepted"]:
                        self.assertTrue(capture_gate(call["measurement"]), call["measurement"])
                        self.assertLessEqual(call["candidate_reaction_N"], 850.)
                        self.assertTrue(call["after"][call["eq_id"]])
                for event in self.results[key]["capture_events"]:
                    self.assertTrue(capture_gate(event), event)
                    self.assertEqual(event["mode"], "physical")
                    self.assertEqual(event["initial_reaction_epoch"], "fresh_post_activation_solve")
                    self.assertLessEqual(event["initial_reaction_N"], 850.)
                    self.assertAlmostEqual(event["initial_reaction_N"],
                                           np.linalg.norm(event["initial_reaction_world_N"]), delta=1e-10)
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                self.assertEqual(len(audit.captures), 1)
                call = audit.captures[0]
                self.assertEqual((call["limb"], call["region"], call["accepted"]),
                                 (Limb.RIGHT_HAND, "reach_target", True))
                capture = result["capture"]
                event = result["capture_events"][-1]
                self.assertEqual(len(result["capture_events"]), 3)
                self.assertEqual((event["limb"], event["region_id"]), ("RIGHT_HAND", "reach_target"))
                self.assertEqual(capture["time_s"], event["time_s"])
                self.assertEqual(capture["time_s"], result["first_eligible"]["time_s"])
                self.assertEqual(capture["time_s"], next(t for t, eligible, _ in audit.eligibility if eligible))
                self.assertEqual(capture["gap_m"], call["measurement"]["gap_m"])
                self.assertAlmostEqual(capture["initial_reaction_N"], call["candidate_reaction_N"], delta=1e-10)
                self.assertAlmostEqual(capture["initial_reaction_N"], 170., delta=10.)
                reach_start = next(e["time_s"] for e in result["events"] if e.get("phase") == "REACH")
                self.assertGreater(capture["time_s"], reach_start)
                self.assertLessEqual(capture["time_s"] - reach_start,
                                     audit.request.reach_s + audit.request.capture_timeout_s + 1e-12)
                self.assertLessEqual(capture["gap_m"], .001)
                decisions = capture["post_activation_decisions"]
                self.assertEqual(set(decisions), {"LEFT_HAND", "RIGHT_HAND"})
                self.assertTrue(all(d["maintain"] and d["effective_capacity"] == 850.
                                    and d["required_load"] <= 850. for d in decisions.values()))
                self.assertFalse(event["window_truncated"])
                self.assertGreaterEqual(event["window_last_sample_time_s"] - event["time_s"], .02 - dt - 1e-9)
                self.assertLessEqual(event["peak_window_reaction_N"], 850.)
                self.assertGreater(event["peak_applied_reaction_N"], 0.)
                previous = [r for r in result["samples"] if r["phase"] == "REACH" and r["time_s"] < capture["time_s"]]
                self.assertTrue(previous)
                self.assertTrue(all(not capture_gate(r["capture_measurement"]) for r in previous))

    def test_native_only_state_evolution_guards_and_immutable_model_reference(self):
        for key, audit in self.audits.items():
            with self.subTest(dt=key[0], scenario=key[1]):
                self.assertEqual(audit.violations, set())
                self.assertEqual(len(audit.steps), self.results[key]["steps"])
                self.assertEqual(audit.reference_before,
                                 (audit.reference.qpos, dict(audit.reference.target_pose), dict(audit.reference.contact_intent)))
                for field, value in audit.before_model.items():
                    np.testing.assert_array_equal(audit.after_model[field], value, err_msg=field)
                self.assertFalse(np.any(audit.data.xfrc_applied) or np.any(audit.data.qfrc_applied))
                np.testing.assert_array_equal(audit.data.qpos, audit.previous[0])
                np.testing.assert_array_equal(audit.data.qvel, audit.previous[1])
                self.assertEqual(audit.data.time, audit.previous[2])
                terminal = self.results[key]["terminal_observation"]
                np.testing.assert_array_equal(terminal["ctrl"], audit.data.ctrl)
                np.testing.assert_array_equal(terminal["qfrc_applied"], audit.data.qfrc_applied)
                np.testing.assert_array_equal(terminal["external_force_world_N"], audit.data.xfrc_applied)
                self.assertEqual(terminal["readiness"]["time"], audit.data.time)
                if terminal["command"] is not None:
                    np.testing.assert_allclose(terminal["command"]["commanded_Nm"],
                                               audit.data.ctrl * audit.model.actuator_gear[:, 0], rtol=0, atol=1e-12)
                    self.assertTrue(terminal["command"]["matches_last_request"])
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                self.assertEqual(len(audit.controls), result["steps"])
                self.assertEqual(len(audit.samples), result["steps"])
                self.assertEqual(len(audit.guards), 2 * result["steps"] + 1)
                self.assertEqual([epoch for epoch, _ in audit.guards if epoch != "post_activation"],
                                 ["pre", "applied"] * result["steps"])
                self.assertEqual([(epoch, time) for epoch, time in audit.guards if epoch == "post_activation"],
                                 [("post_activation", result["capture"]["time_s"])])
                self.assertTrue(audit.ik and all(converged for _, converged, _ in audit.ik))
                self.assertTrue(any(iterations > 0 for _, _, iterations in audit.ik))
                for row, native in zip(result["samples"], audit.steps):
                    np.testing.assert_array_equal(row["qpos"], native["qpos"])
                    np.testing.assert_array_equal(row["qvel"], native["qvel"])
                    self.assertEqual(row["time_s"], native["time_s"])

    def test_fixture_is_builder_physics_with_only_hold_positions_and_extra_target(self):
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                tree = ET.fromstring(build_mjcf(audit.scene, audit.profile))
                tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
                builder = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
                expected = model_parameters(builder)
                for field, value in audit.before_model.items():
                    np.testing.assert_array_equal(value, expected[field], err_msg=field)
                model = audit.model
                self.assertEqual((model.nq, model.nv, model.nu), (32, 31, 25))
                self.assertEqual(float(model.numeric("contact_mode").data[0]), 0.)
                self.assertAlmostEqual(model.body_subtreemass[model.body("climber_root").id], 78.3, places=10)
                self.assertEqual(result["model_proof"]["equalities"], [model.equality(i).name for i in range(model.neq)])
                self.assertEqual(result["model_proof"]["nexclude"], model.nexclude)
                self.assertEqual(result["model_proof"]["mass_kg"], 78.3)
                self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
                self.assertTrue(np.all(model.jnt_type[model.actuator_trnid[:, 0]] == mujoco.mjtJoint.mjJNT_HINGE))
                stage3 = make_mixed_fixture(dt).model
                for field in ("dof_damping", "dof_armature", "dof_frictionloss", "jnt_range", "actuator_gear",
                              "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange"):
                    np.testing.assert_array_equal(getattr(model, field), getattr(stage3, field), err_msg=field)
                # The extra static hold shifts body IDs, but must not change excluded body pairs.
                exclusions = [{frozenset((compiled.body(int(signature) >> 16).name,
                                           compiled.body(int(signature) & 0xffff).name))
                               for signature in compiled.exclude_signature}
                              for compiled in (model, stage3)]
                self.assertEqual(exclusions[0], exclusions[1])
                self.assertFalse(any(any("forearm" in name for name in pair)
                                     and any(name == wall.id for wall in audit.scene.walls for name in pair)
                                     for pair in exclusions[0]))
                for body in range(stage3.nbody):
                    name = stage3.body(body).name
                    current = model.body(name).id
                    for field in ("body_mass", "body_inertia"):
                        np.testing.assert_array_equal(getattr(model, field)[current], getattr(stage3, field)[body])
                for eq in range(model.neq):
                    self.assertNotIn("foot", model.equality(eq).name)
                    self.assertEqual(model.eq_type[eq], mujoco.mjtEq.mjEQ_CONNECT)
                    self.assertEqual(model.eq_objtype[eq], mujoco.mjtObj.mjOBJ_SITE)
                    self.assertIn(model.site(int(model.eq_obj1id[eq])).name, ("left_hand_site", "right_hand_site"))
                _, _, _, _, seed = single_hand.make_single_hand_fixture(dt)
                for side in ("left", "right"):
                    self.assertEqual(seed[model.joint(f"{side}_knee").qposadr[0]], .3)

    def test_minimum_jerk_endpoints_rates_and_outward_arch(self):
        for time, expected in ((-1., (0., 0.)), (0., (0., 0.)), (4., (1., 0.)), (8., (1., 0.))):
            self.assertEqual(single_hand.minimum_jerk(time, 4.), expected)
        self.assertEqual(single_hand.minimum_jerk(2., 4.), (.5, 1.875 / 4.))
        for time, duration in ((math.nan, 4.), (math.inf, 4.), (1., 0.), (1., -1.), (1., math.inf)):
            with self.subTest(time=time, duration=duration), self.assertRaises(ValueError):
                single_hand.minimum_jerk(time, duration)
        for dt in TIMESTEPS:
            t = np.arange(0., 4. + dt / 2, dt)
            values = [single_hand.minimum_jerk(v, 4.) for v in t]
            self.assertTrue(np.all(np.diff([b for b, _ in values]) >= -1e-14))
            self.assertLessEqual(max(rate for _, rate in values), 1.875 / 4.)
        audit = self.audits[.002, "success"]
        start = canonical_geometry(audit.scene.region("right_hand")).hand_frame
        goal = canonical_geometry(audit.scene.region("reach_target")).hand_frame
        for time, expected in ((0., start), (4., goal)):
            actual = single_hand.reach_frame(start, goal, time, 4., .02)
            np.testing.assert_allclose(actual.position, expected.position, rtol=0, atol=1e-15)
            np.testing.assert_allclose(actual.rotation, expected.rotation, rtol=0, atol=1e-15)
        midpoint = single_hand.reach_frame(start, goal, 2., 4., .02)
        np.testing.assert_allclose(midpoint.position,
                                   (np.array(start.position) + goal.position) / 2 + .02 * np.array(goal.normal),
                                   rtol=0, atol=1e-15)
        for end, sign in ((0., 1), (4., -1)):
            nearby = single_hand.reach_frame(start, goal, end + sign * 1e-5, 4., .02)
            endpoint = single_hand.reach_frame(start, goal, end, 4., .02)
            self.assertLess(np.linalg.norm(np.array(nearby.position) - endpoint.position) / 1e-5, 1e-6)

    def test_reach_frame_rotated_goal_interpolates_in_correct_frame(self):
        audit = self.audits[.002, "success"]
        start = canonical_geometry(audit.scene.region("right_hand")).hand_frame
        goal = canonical_geometry(audit.scene.region("reach_target")).hand_frame
        angle = .4
        rotation = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                             [-math.sin(angle), 0., math.cos(angle)]])
        rotated = Frame(goal.position, tuple(map(tuple, rotation @ np.array(goal.rotation))))
        middle = single_hand.reach_frame(start, rotated, 2., 4., .02)
        endpoint = single_hand.reach_frame(start, rotated, 4., 4., .02)
        error = np.array(endpoint.rotation) @ np.array(rotated.rotation).T
        angular_error = math.acos(float(np.clip((np.trace(error) - 1.) / 2, -1., 1.)))
        print("SINGLE_HAND_ROTATED_FRAME " + json.dumps({"goal_rotation_rad": angle,
              "endpoint_rotation_error_rad": angular_error,
              "endpoint_normal_error": float(np.linalg.norm(np.array(endpoint.normal) - rotated.normal))}), flush=True)
        np.testing.assert_allclose(endpoint.rotation, rotated.rotation, rtol=0, atol=1e-14,
                                   err_msg="reach_frame must reach a rotated goal from a nonidentity start frame")
        np.testing.assert_allclose(np.array(middle.rotation).T @ middle.rotation, np.eye(3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(middle.rotation, np.array([[math.cos(angle / 2), 0., math.sin(angle / 2)],
                                  [0., 1., 0.], [-math.sin(angle / 2), 0., math.cos(angle / 2)]]) @ start.rotation,
                                   rtol=0, atol=1e-14)

    def test_reference_rates_rom_commands_and_genuine_reach_path(self):
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                rows = result["samples"]
                qrefs = np.array([audit.reference.qpos] + [r["q_ref"] for r in rows])
                qdrefs = np.array([r["qd_ref"] for r in rows])
                self.assertLessEqual(np.max(np.abs(np.diff(qrefs[:, audit.qi], axis=0))), .5 * dt + 1e-12)
                self.assertLessEqual(np.max(np.abs(qdrefs[:, audit.vi])), .5 + 1e-12)
                rate_changes = np.diff(np.vstack((np.zeros((1, audit.model.nv)), qdrefs)), axis=0)
                self.assertLessEqual(np.max(np.abs(rate_changes[:, audit.vi])), 2. * dt + 1e-12)
                limits = audit.model.actuator_gear[:, 0] * audit.model.actuator_ctrlrange[:, 1]
                for index, row in enumerate(rows):
                    validate_reference_pose(audit.model, row["q_ref"])
                    validate_reference_pose(audit.model, row["qpos"])
                    np.testing.assert_array_equal([audit.controls[index]["target"][name]
                                                   for name in row["command"]["joint_names"]], qrefs[index + 1, audit.qi])
                    np.testing.assert_array_equal([audit.controls[index]["velocity"].get(name, 0.)
                                                   for name in row["command"]["joint_names"]], qdrefs[index, audit.vi])
                    np.testing.assert_array_equal(row["command"]["limits_Nm"], limits)
                    np.testing.assert_allclose(row["command"]["commanded_Nm"],
                                               np.clip(row["command"]["desired_Nm"], -limits, limits), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(row["command"]["utilization"],
                                               np.abs(row["command"]["commanded_Nm"]) / limits, rtol=0, atol=1e-12)
                    self.assertFalse(any(row["command"]["saturated"]))
                    tangent = np.zeros(audit.model.nv)
                    mujoco.mj_differentiatePos(audit.model, tangent, dt, qrefs[index], qrefs[index + 1])
                    np.testing.assert_allclose(qdrefs[index], tangent, rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(row["ctrl"], audit.steps[index]["ctrl"])
                self.assertLess(result["actuator_utilization_max"], .46)
                reach = [r for r in rows if r["phase"] == "REACH"]
                begin = Frame(**reach[0]["desired_hand_frame"])
                goal = canonical_geometry(audit.scene.region("reach_target")).hand_frame
                for index, row in enumerate(reach):
                    expected = single_hand.reach_frame(begin, goal, index * dt, 4., .02)
                    np.testing.assert_allclose(row["desired_hand_frame"]["position"], expected.position, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(row["desired_hand_frame"]["rotation"], expected.rotation, rtol=0, atol=1e-12)
                self.assertGreater(np.linalg.norm(np.array(reach[0]["qpos"])[audit.qi]
                                                  - np.array(reach[-1]["qpos"])[audit.qi]), .05)
                self.assertGreater(reach[0]["capture_measurement"]["gap_m"], .05)
                self.assertLess(reach[-1]["capture_measurement"]["gap_m"], .001)
                self.assertLessEqual(result["three_point"]["root_linear_max_m_s"], .10)
                self.assertLessEqual(result["three_point"]["root_angular_max_rad_s"], .50)
                self.assertLessEqual(result["three_point"]["joint_max_rad_s"], 1.)
                self.assertAlmostEqual(result["three_point"]["duration_s"],
                                       sum(r["phase"] in ("THREE_POINT", "REACH", "CAPTURE") for r in rows) * dt)

    def test_settle_has_admitted_blended_reference_and_sustained_normal_stage3_ready(self):
        for dt, result, audit in self.success_runs():
            with self.subTest(dt=dt):
                settle = [r for r in result["samples"] if r["phase"] == "SETTLE"]
                self.assertGreaterEqual(len(settle), round(.5 / dt) + 1)
                self.assertGreaterEqual(result["readiness"]["duration"], .5 - 1e-12)
                self.assertTrue(result["readiness"]["ready"])
                self.assertFalse(settle[0]["readiness"]["ready"])
                self.assertNotEqual(settle[0]["q_ref"], settle[-1]["q_ref"])
                final_window = settle[-(round(.5 / dt) + 1):]
                for row in final_window:
                    self.assertLessEqual(row["root_linear_m_s"], .02)
                    self.assertLessEqual(row["root_angular_rad_s"], .05)
                    self.assertLessEqual(row["joint_max_rad_s"], .10)
                    self.assertEqual(row["readiness"]["root_linear_speed"], row["root_linear_m_s"])
                    self.assertEqual(row["readiness"]["root_angular_speed"], row["root_angular_rad_s"])
                    self.assertEqual(row["readiness"]["max_hinge_speed"], row["joint_max_rad_s"])
                self.assertGreaterEqual(final_window[-1]["time_s"] - final_window[0]["time_s"], .5 - 1e-12)
                ready = next(e for e in result["events"] if e.get("event") == "FINAL_READY")
                self.assertEqual((ready["time_s"], ready["step"]), (result["final_state"].time, result["steps"]))
                self.assertEqual(result["terminal_observation"]["readiness"], result["readiness"])
                # Re-admission is scratch-only and does not replace the source reference.
                pose = single_hand.solve_contact_pose(
                    audit.model, audit.scene, audit.profile,
                    result["samples"][result["capture"]["step"] - 1]["qpos"], NEW_INTENT)
                self.assertTrue(pose.admitted, pose.reason)
                self.assertEqual(dict(pose.reference.contact_intent), NEW_INTENT)
                np.testing.assert_allclose(settle[-1]["q_ref"], pose.qpos, rtol=0, atol=1e-12)
                begin = np.array(result["samples"][result["capture"]["step"] - 1]["q_ref"])
                tangent = np.zeros(audit.model.nv)
                mujoco.mj_differentiatePos(audit.model, tangent, 1., begin, np.array(pose.qpos))
                for index, row in enumerate(settle):
                    blend, _ = single_hand.minimum_jerk((index + 1) * dt, .5)
                    expected = begin.copy()
                    mujoco.mj_integratePos(audit.model, expected, tangent, blend)
                    # The admitted minimum-jerk goal precedes velocity/acceleration shaping.
                    np.testing.assert_allclose(audit.shaped_goals[result["capture"]["step"] + index],
                                               expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(np.array(settle[-1]["qd_ref"])[audit.vi], 0., rtol=0, atol=1e-8)

    def test_remaining_hand_peak_summary_is_measured_not_missing(self):
        for dt, result, _ in self.success_runs():
            with self.subTest(dt=dt):
                measured = max(r["hands"]["LEFT_HAND"]["load"] for r in result["samples"])
                self.assertIsNotNone(result["remaining_hand_load_max_N"],
                                     f"remaining_hand_load_max_N=None despite observed LEFT_HAND peak={measured:.9f}N")
                self.assertAlmostEqual(result["remaining_hand_load_max_N"], measured, delta=1e-10)

    def test_physical_negative_verdicts_and_failure_evidence(self):
        expected = {"unreachable": "REACH_INFEASIBLE", "orientation_invalid": "CAPTURE_FAILURE",
                    "support_loss": "CONTACT_LOSS", "grip_after_capture": "GRIP_FAILURE"}
        for scenario in NEGATIVES:
            key = (.002, scenario)
            with self.subTest(scenario=scenario):
                self.assertNotIn(key, self.run_errors, self.run_errors.get(key))
                result, audit = self.results[key], self.audits[key]
                self.assertEqual(result["status"], expected[scenario], result["reason"])
                self.assertFalse(result["success"] or result["readiness"]["ready"])
                self.assertEqual(result["terminal_observation"]["status"], expected[scenario])
                self.assertEqual(result["terminal_observation"]["reason"], result["reason"])
                self.assertEqual(result["terminal_observation"]["readiness"]["reason"], result["reason"])
                self.assertFalse(np.any(audit.data.xfrc_applied))
                if scenario in ("unreachable", "orientation_invalid"):
                    self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
                    self.assertFalse(result["released"])
                    self.assertEqual(result["initial_state"], result["final_state"])
                    self.assertEqual(result["new_contacts"], INTENT)
                    self.assertEqual(audit.detaches, [])
                    self.assertEqual(audit.captures, [])
                    self.assertEqual(result["release_events"], [])
                    np.testing.assert_array_equal(integration_state(audit.model, audit.data),
                                                  integration_state(audit.model, audit.start_data))
                    if scenario == "orientation_invalid":
                        proposed = result["preflight"]
                        self.assertLess(proposed["gap_m"], 1e-10)
                        self.assertAlmostEqual(proposed["orientation"], math.cos(math.radians(31)), delta=1e-7)
                        self.assertLess(proposed["orientation"], CAPTURE_ORIENTATION)
                        self.assertGreaterEqual(proposed["penetration_m"], .001)
                else:
                    self.assertTrue(result["released"])
                    self.assertGreater(result["steps"], 0)
                    self.assertEqual(result["fault"], scenario)
                    self.assertLessEqual(len(result["samples"]), result["steps"])
                    self.assertTrue(audit.force_guards, "physical negative never applied its declared force")
                    force_steps = [s for s in audit.steps if s["forces"]]
                    if scenario == "support_loss":
                        self.assertEqual(result["steps"], len(result["samples"]) + 1)
                        self.assertTrue(force_steps)
                        self.assertEqual((audit.force_guards[0]["body"], audit.force_guards[0]["force"]),
                                         ("left_foot", (1500., 0., 0., 0., 0., 0.)))
                        begin = next(e for e in result["events"] if e.get("phase") == "REACH")
                        self.assertAlmostEqual(force_steps[0]["before_s"] - begin["time_s"], .3, places=10)
                        self.assertIsNone(result["capture"])
                        self.assertEqual(audit.captures, [])
                        failure = result["guard_failure"]
                        self.assertIsNotNone(failure)
                        self.assertTrue(any(f["slipping"] or not f["supporting"]
                                            or f["support_regions"] != (INTENT[limb],)
                                            for limb in FEET for f in [failure["feet"][limb.value]]))
                    else:
                        self.assertEqual(result["steps"], len(result["samples"]))
                        self.assertIsNotNone(result["capture"])
                        self.assertEqual((audit.force_guards[0]["body"], audit.force_guards[0]["force"]),
                                         ("right_hand", (0., -2000., 0., 0., 0., 0.)))
                        self.assertAlmostEqual(audit.force_guards[0]["time_s"] - result["capture"]["time_s"],
                                               2 * .002, places=10)
                        release = next(e for e in result["release_events"] if e["region_id"] == "reach_target")
                        self.assertEqual(release["limb"], "RIGHT_HAND")
                        self.assertEqual(release["capacity_N"], 850.)
                        self.assertGreater(release["required_load_N"], 850.)
                        self.assertEqual(release["reason"], "capacity exceeded")
                        self.assertFalse(audit.manager.is_attached(Limb.RIGHT_HAND))
                        self.assertAlmostEqual(release["time_s"] - result["capture"]["time_s"], .004, places=10)
                        # A force can break the fresh pre-step solve without another mj_step.
                        self.assertEqual(result["terminal_observation"]["phase"], "SETTLE")

    def test_adversarial_observer_cannot_change_exact_full_result(self):
        observations = []

        def observer(row, model, data):
            observations.append(row["time_s"])
            row["readiness"]["ready"] = False
            row["feet"]["LEFT_FOOT"]["supporting"] = False
            row["hands"]["LEFT_HAND"]["load"] = 1e9
            row["q_ref"][:] = [1e6] * model.nq
            row["eq_active"][:] = [0] * model.neq
            if row["command"]:
                row["command"]["commanded_Nm"] = [1e9] * model.nu
            model.body_mass[:] = 1e6
            model.body_inertia[:] = 1e6
            model.geom_size[:] = 1e6
            model.pair_friction[:] = 0.
            model.eq_solref[:] = 1.
            model.actuator_gear[:] = 0.
            model.dof_damping[:] = 1e6
            model.opt.gravity[:] = 0.
            model.opt.timestep = .1
            data.qpos[:] = 1e6
            data.qvel[:] = 1e6
            data.ctrl[:] = 1e6
            data.eq_active[:] = False
            data.xfrc_applied[:] = 1e6
            data.time = -1.

        observed, audit = self.audited_benchmark(.002, observer=observer)
        self.assertGreater(len(observations), 100)
        self.assertEqual(observed, self.results[.002, "success"])
        self.assertEqual(audit.violations, set())
        for field, value in audit.before_model.items():
            np.testing.assert_array_equal(audit.after_model[field], value, err_msg=field)

    def test_malformed_requests_reject_without_steps_resets_or_command_mutation(self):
        args = self.continuation()
        model, data, scene, profile, reference, manager = args
        request = single_hand.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")
        invalid = [replace(request, limb=Limb.LEFT_FOOT), replace(request, limb="RIGHT_HAND"),
                   replace(request, source="left_hand"), replace(request, target="right_hand"),
                   replace(request, target="missing"), replace(request, transfer_s=0.),
                   replace(request, support_s=-.5), replace(request, reach_s=math.nan),
                   replace(request, capture_timeout_s=math.inf), replace(request, settle_timeout_s=.003),
                   replace(request, clearance_m=-.01), replace(request, clearance_m=math.nan),
                   replace(request, approach_normal=(math.nan, 0., 0.))]
        data.ctrl[:] = .123
        native_reset = mujoco.mj_resetData
        live_resets = []

        def reset(model, destination):
            if destination is data:
                live_resets.append(float(data.time))
                raise AssertionError("malformed request reset live data")
            return native_reset(model, destination)

        for bad in invalid:
            with self.subTest(request=bad):
                before = integration_state(model, data)
                events, releases = copy.deepcopy(manager.capture_events), copy.deepcopy(manager.releases)
                with patch("mujoco.mj_resetData", new=reset), \
                        patch("mujoco.mj_step", side_effect=AssertionError("malformed request stepped")), \
                        patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("malformed request commanded")):
                    result = single_hand.execute_single_hand(*args, bad)
                self.assertFalse(result["success"] or result["released"] or result["readiness"]["ready"])
                self.assertEqual(result["status"], "REACH_INFEASIBLE")
                self.assertEqual((result["steps"], result["duration_s"], result["samples"]), (0, 0., []))
                np.testing.assert_array_equal(integration_state(model, data), before)
                self.assertEqual(manager.capture_events, events)
                self.assertEqual(manager.releases, releases)
                self.assertEqual(live_resets, [])
        with self.assertRaises(FrozenInstanceError):
            request.target = "other"
        for index, replacement in ((0, copy.copy(model)), (1, mujoco.MjData(model)),
                                   (2, replace(scene, scale=2.))):
            mismatched = list(args)
            mismatched[index] = replacement
            before = integration_state(model, data)
            with self.subTest(argument=index), self.assertRaises(AttachmentStateError), \
                    patch("mujoco.mj_step", side_effect=AssertionError("session mismatch stepped")), \
                    patch("mujoco.mj_resetData", side_effect=AssertionError("session mismatch reset")), \
                    patch.object(single_hand, "compute_pose_control", side_effect=AssertionError("session mismatch commanded")):
                single_hand.execute_single_hand(*mismatched, request)
            np.testing.assert_array_equal(integration_state(model, data), before)

    def test_unexpected_exception_cleans_owned_force_and_returns_terminal_failure(self):
        args = self.continuation()
        model, data, _, _, _, manager = args
        request = single_hand.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")
        body = model.body("left_foot").id
        original = manager.evaluate_and_update
        force_seen, exception, result = [], None, None

        def guard(*args, **kwargs):
            if np.any(data.xfrc_applied[body]):
                force_seen.append(data.xfrc_applied[body].copy())
                raise RuntimeError("injected unexpected guard exception after owned support-loss force")
            return original(*args, **kwargs)

        with patch.object(manager, "evaluate_and_update", side_effect=guard):
            try:
                result = single_hand.execute_single_hand(*args, request, fault="support_loss")
            except RuntimeError as error:
                exception = error
        self.assertTrue(force_seen, f"counterexample did not reach declared force injection: {result}")
        np.testing.assert_array_equal(force_seen[0], [1500., 0., 0., 0., 0., 0.])
        self.assertFalse(np.any(data.xfrc_applied), "owned fault wrench leaked during unexpected exception")
        self.assertIsNone(exception, f"executor leaked {exception!r} instead of returning a terminal CONTROL_FAILURE")
        self.assertFalse(result["success"] or result["readiness"]["ready"])
        self.assertEqual(result["status"], "CONTROL_FAILURE")
        self.assertIn("injected unexpected guard exception", result["reason"])

    def test_actual_last_native_endpoint_failure_is_not_overwritten_by_success(self):
        args = self.continuation()
        model, data, _, _, _, _ = args
        end = self.results[.002, "success"]["final_state"].time
        request = single_hand.SingleHandRequest(Limb.RIGHT_HAND, "right_hand", "reach_target")
        native, pulses = mujoco.mj_step, []
        body = model.body("left_foot").id

        def step(m, d, *args, **kwargs):
            if d is data and d.time >= end - model.opt.timestep - 1e-12:
                original = data.xfrc_applied[body, :3].copy()
                data.xfrc_applied[body, :3] = [1500., 0., 0.]
                pulses.append(float(data.time))
                # This fixture owns one native-interval pulse; keep undeclared-force audits enabled.
                try:
                    return native(m, d, *args, **kwargs)
                finally:
                    data.xfrc_applied[body, :3] = original
            return native(m, d, *args, **kwargs)

        try:
            with patch("mujoco.mj_step", new=step):
                result = single_hand.execute_single_hand(*args, request)
        finally:
            data.xfrc_applied[body, :3] = 0.
        self.assertEqual(len(pulses), 1)
        self.assertEqual(result["steps"], self.results[.002, "success"]["steps"])
        self.assertEqual(result["status"], "CONTACT_LOSS", result["reason"])
        self.assertFalse(result["success"] or result["readiness"]["ready"])
        self.assertEqual(result["terminal_observation"]["status"], "CONTACT_LOSS")
        self.assertIsNotNone(result["guard_failure"])
        self.assertTrue(result["guard_failure"]["feet"]["LEFT_FOOT"]["slipping"])
        self.assertFalse(any(e.get("event") == "FINAL_READY" for e in result["events"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
