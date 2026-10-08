"""Fast saved-evidence contracts. Synthetic contact records certify no physics."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from boulder_v1.contact_geometry import Frame, canonical_geometry
from boulder_v1.schema import Affordance, BoulderScene, ContactRegion, Limb, SourceType, WallSurface
from boulder_v1.whole_body_motion import WholeBodyMotion
from scripts import validate_ascent as study

NATIVE_SEQUENCE = study.ascent.run_ascending_sequence


def synthetic_fixture(name="baseline", dt=.002):
    """Compile a declared unit scene, never solve a source or execute a hold."""
    profile = study.study_profiles()[name]
    regions = []
    for limb in Limb:
        regions.append(ContactRegion(limb.value.lower(), SourceType.HOLD,
            ((-.2 if limb.value.startswith("LEFT") else .2), 0., 1.5 if limb.is_hand else .2),
            (0., -1., 0.) if limb.is_hand else (0., 0., 1.), .9,
            frozenset((Affordance.GRASP if limb.is_hand else Affordance.STEP,)), half_size=(.06, .04, .025)))
    for limb, target in zip(study.LIMBS, study.TARGETS):
        source = next(r for r in regions if r.id == limb.lower())
        regions.append(ContactRegion(target, SourceType.HOLD,
            (source.position[0] - (.105 if limb.endswith("FOOT") else 0.), 0.,
             source.position[2] + (.082 if limb.endswith("FOOT") else .145)), source.normal, .9,
            source.affordances, half_size=source.half_size))
    scene = BoulderScene("unit", 1., (WallSurface("wall", (0., .2, 1.), (3., .1, 3.)),),
                         tuple(regions), {l: l.value.lower() for l in Limb})
    tree = ET.fromstring(study.build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(dt), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    seed = model.qpos0.copy()
    for jid in model.actuator_trnid[:, 0]:
        seed[model.jnt_qposadr[jid]] = np.mean(model.jnt_range[jid])
    native = {"fixture": study.FIXTURE, "profile": profile.to_dict(), "scene": scene.to_dict(),
        "seed_qpos": seed, "dt_s": dt, "negative": None, "step_height_m": .082,
        "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "contact_geometry": {r.id: canonical_geometry(r) for r in scene.contact_regions}}
    fixture = (model, mujoco.MjData(model), scene, profile, seed, native)
    with mock.patch.object(study.ascent, "make_ascending_fixture", return_value=fixture):
        _, inputs = study._fixture_metadata(profile, dt)
    return model, scene, profile, seed, inputs


def synthetic_record(fixture, timing="fast"):
    """A complete schema exercise, deliberately not a native physical episode."""
    model, scene, profile, seed, inputs = fixture
    dt = float(model.opt.timestep)
    contacts = scene.to_dict()["start_configuration"]
    zero = {"qfrc_applied": [0.] * model.nv, "external_force_world_N": [[0.] * 6 for _ in range(model.nbody)]}

    def readiness(time, duration=.5):
        return {"ready": True, "duration": duration, "reason": "Sustained physical readiness", "time": time,
                "root_linear_speed": 0., "root_angular_speed": 0., "max_hinge_speed": 0., "rms_hinge_speed": 0.}

    def state(time, intent, z=0.):
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = seed
        scratch.qpos[2] += z
        scratch.eq_active[:] = False
        for limb, hold in intent.items():
            if limb.endswith("HAND"):
                scratch.eq_active[model.equality("grasp_" + limb.lower() + "_" + hold).id] = True
        scratch.time = time
        vector = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, scratch, vector, mujoco.mjtState.mjSTATE_INTEGRATION)
        snapshot = {"time": time, "qpos": scratch.qpos.tolist(), "qvel": scratch.qvel.tolist(),
            "ctrl": scratch.ctrl.tolist(), "eq_active": scratch.eq_active.tolist(),
            "qacc_warmstart": scratch.qacc_warmstart.tolist(), "finite": True, "contact_mode": "physical",
            "contact_configuration": dict(intent),
            "hand_states": {l.value: {"active": l.value in intent, "valid": l.value in intent,
                "measurement_valid": True, "region_id": intent.get(l.value), "load": 20. if l.value in intent else 0.,
                "capacity": 850. if l.value in intent else None, "margin": 830. if l.value in intent else None,
                "force_world": [0., 0., 20. if l.value in intent else 0.]} for l in Limb if l.is_hand},
            "foot_states": {l.value: {"measurement_valid": True, "supporting": l.value in intent,
                "contacting": l.value in intent, "slipping": False, "normal_force": 350. if l.value in intent else 0.,
                "tangential_force": 0., "tangential_speed": 0., "idealized_attachment": None,
                "support_regions": [intent[l.value]] if l.value in intent else [],
                "support_surfaces": ["geom_" + intent[l.value]] if l.value in intent else [],
                "contacts": [{"surface_geom": "geom_" + intent[l.value], "shoe_geom": l.value.lower() + "_geom",
                    "admissible": True, "normal_force": 350., "tangential_speed": 0.}] if l.value in intent else []}
                for l in Limb if l.is_foot}}
        return snapshot, vector.tolist()

    initial, iv = state(2., contacts)
    static = {"success": True, "status": "SUCCESS", "reason": "Unit schema only", "initial_state": state(0., contacts)[0],
        "final_state": initial, "steps": round(2. / dt), "duration_s": 2., "contact_acceptance": True,
        "controller_convergence": True, "disturbance": None, "disturbance_realization": None, "readiness": readiness(2.),
        "terminal_observation": {"terminal": True, "pose_available": True, "status": "SUCCESS", "reason": "Unit schema only",
            "time_s": 2., "ctrl": initial["ctrl"], "hands": initial["hand_states"], "feet": initial["foot_states"],
            "readiness": readiness(2.), "unintended_contacts": [], "disturbance_active": False}}
    moves, candidates, references = [], [], []
    for index, (limb, target) in enumerate(zip(study.LIMBS, study.TARGETS)):
        start = initial if index == 0 else moves[-1]["final_state"]
        vector = iv if index == 0 else moves[-1]["final_integration_state"]
        source, supports = contacts[limb], {l: h for l, h in contacts.items() if l != limb}
        goal = {**contacts, limb: target}
        prepare, reach_s = study.ascent.TIMINGS[timing]
        foot = limb.endswith("FOOT")
        if foot:
            prepare = 4.
        source_step = round(.5 / dt)
        release_step = source_step + round((prepare + (.75 if foot else .5)) / dt)
        reach_step = source_step + round((prepare + (2. if foot else 1.)) / dt)
        capture_step = reach_step + round((3. if foot else reach_s + .102) / dt)
        acquired_step = capture_step + max(2, round(.102 / dt)) if foot else capture_step
        load_step = capture_step + round(2. / dt) if foot else capture_step
        steps = load_step + round((1. + dt if foot else .5 + dt) / dt)
        epoch = lambda step: start["time"] + step * dt
        phases = [("SOURCE_STABILIZE", 0), ("UNLOAD" if foot else "LOAD_TRANSFER", source_step)]
        if foot:
            phases += [("LIFT", source_step + round(prepare / dt)), ("THREE_POINT", reach_step - round(.5 / dt)),
                       ("REACH", reach_step), ("LANDING", capture_step - round(.25 / dt)), ("LOAD", capture_step), ("SETTLE", load_step)]
        else:
            phases += [("RELEASE_CLEARANCE", release_step - round(.5 / dt)), ("THREE_POINT", release_step),
                       ("REACH", reach_step), ("SETTLE", capture_step)]
        events = [{"phase": phase, "time_s": epoch(step), "step": step} for phase, step in phases]
        events += [{"event": "SOURCE_READY", "time_s": epoch(source_step), "step": source_step},
            {"event": "SEPARATION" if foot else "RELEASED", "time_s": epoch(release_step), "step": release_step,
             "limb": limb, "source": source}, {"event": "FINAL_READY", "time_s": epoch(steps), "step": steps}]
        if not foot:
            events.append({"event": "CAPTURED", "time_s": epoch(capture_step), "step": capture_step})
        events.sort(key=lambda e: (e["time_s"], e.get("event") != "SOURCE_READY"))
        final, fv = state(epoch(steps), goal, .001 * (index + 1))
        ref = {"qpos": final["qpos"], "contact_intent": goal,
               "target_pose": {model.joint(int(j)).name: final["qpos"][model.jnt_qposadr[j]] for j in model.actuator_trnid[:, 0]}}
        references.append(ref)
        samples = []
        templates = {"source": state(start["time"], contacts)[0], "supports": state(start["time"], supports)[0],
                     "goal": state(start["time"], goal)[0]}
        for step in range(1, steps + 1):
            phase = next(p for p, s in reversed(phases) if step > s)
            config = "source" if (step < release_step if foot else step <= release_step) else "supports" if step <= capture_step else "goal"
            snapshot = templates[config]
            qpos = seed.copy()
            qpos[2] += .001 * (index + step / steps)
            samples.append({"time_s": epoch(step), "qpos": qpos.tolist(), "qvel": snapshot["qvel"],
                "ctrl": snapshot["ctrl"], "eq_active": snapshot["eq_active"], "hands": snapshot["hand_states"],
                "feet": snapshot["foot_states"], "contacts": snapshot["contact_configuration"], "steps": step,
                "phase": phase, "pose_available": True, "root_linear_m_s": 0., "root_angular_rad_s": 0., "joint_max_rad_s": 0.,
                **zero})
        # Samples at the touchdown epoch precede activation; native feet become
        # loaded immediately afterwards in this synthetic schema exercise.
        if foot:
            samples[capture_step]["feet"] = templates["goal"]["foot_states"]
        terminal = {**samples[-1], "terminal": True, "status": "SUCCESS", "reason": "Unit schema only",
            "root_pose": final["qpos"][:7], "readiness": readiness(final["time"]), "qpos": final["qpos"],
            "hands": final["hand_states"], "feet": final["foot_states"], "contacts": goal}
        request = {"limb": limb, "source": source, "target": target, "source_contacts": dict(contacts),
            "support_contacts": supports, "hand_reach_s": reach_s, "hand_acquisition_policy": "endpoint_settle",
            "whole_body": {"prepare_s": prepare}, "foot_request": {"lift_s": 1.5, "support_s": .5,
                "reach_s": 3., "load_s": 2., "lift_m": .102, "airborne_pitch_rad": -.25} if foot else None}
        move = {"success": True, "status": "SUCCESS", "reason": "Unit schema only", "moving_limb": limb,
            "source": source, "target": target, "primitive": "foot" if foot else "hand", "dt_s": dt,
            "steps": steps, "duration_s": steps * dt, "initial_state": start, "final_state": final,
            "initial_integration_state": vector, "final_integration_state": fv, "samples": samples,
            "request": request, "released": True, "release_time_s": epoch(release_step),
            "capture_time_s": None if foot else epoch(capture_step), "reach_start_time_s": epoch(reach_step),
            "readiness_time_s": final["time"], "readiness": readiness(final["time"]), "final_contacts": goal,
            "events": events, "terminal_observation": terminal, "final_reference": ref,
            "incoming_reference": references[index - 1] if index else ref,
            "admission": {"source_reference_admitted": True}}
        if foot:
            move.update(release={"time_s": epoch(release_step), "contact_count": 0, "normal_force_N": 0.,
                "signed_source_geom_distance_m": .001}, touchdown={"time_s": epoch(capture_step), "normal_force_N": .2},
                acquisition={"time_s": epoch(acquired_step), "normal_force_N": 350., "sustained_s": .1},
                acquisition_time_s=epoch(acquired_step), load_start_time_s=epoch(capture_step),
                load_complete_time_s=epoch(load_step), load_duration_s=2.)
        else:
            capture = {"time_s": epoch(capture_step), "step": capture_step, "gap_m": .0002,
                "relative_speed_m_s": .0001, "relative_velocity_world_m_s": [0., 0., .0001], "orientation": 1.,
                "penetration_m": 0., "capture_margin_m": .0008, "initial_reaction_N": 20.,
                "acquisition_policy": "endpoint_settle", "reach_elapsed_s": epoch(capture_step) - epoch(reach_step),
                "reference_time_s": reach_s + .1,
                "post_activation_decisions": {l: {"maintain": True, "required_load": 20., "effective_capacity": 850.,
                    "utilization": 20. / 850.} for l in goal if l.endswith("HAND")}}
            move.update(capture=capture, capture_error_m=.0002, capture_margin_m=.0008,
                capture_events=[{**capture, "limb": limb, "region_id": target, "mode": "physical",
                    "initial_reaction_epoch": "fresh_post_activation_solve", "initial_reaction_world_N": [0., 0., 20.]}])
        moves.append(move)
        if not foot:
            diagnostics = {"actual_qpos": start["qpos"], "actual_integration_state": vector,
                "actual_state_digest": hashlib.sha256(np.asarray(vector).tobytes()).hexdigest(), "endpoint_admitted": True}
            assessments = [[label, {"feasible": True, "classification": "GEOMETRICALLY_FEASIBLE",
                "geometric_feasible": True, "support_feasible": True, "diagnostics": diagnostics}]
                for label in ("default", "neutral_yaw", "conservative")]
            if index == 2:
                for _, assessment in assessments[:2]:
                    assessment.update(feasible=False, classification="LOCAL_SEARCH_UNRESOLVED", geometric_feasible=None)
            candidates.append({"move_index": index, "selection": {"feasible": True, "selected_index": 0 if index == 0 else 2,
                "selected_name": "default" if index == 0 else "conservative", "assessments": assessments}})
        contacts = goal
    return {"success": True, "status": "SUCCESS", "reason": "SYNTHETIC UNIT SCHEMA, NOT PHYSICS", "kind": "ascending_sequence",
        "fixture": study.FIXTURE, "fixture_inputs": inputs["native_metadata"], "profile": profile.to_dict(),
        "dt_s": dt, "timing": timing, "negative": None, "initial_static": static, "initial_state": initial,
        "initial_integration_state": iv, "final_state": moves[-1]["final_state"],
        "final_integration_state": moves[-1]["final_integration_state"], "moves": moves, "candidates": candidates,
        "completed_moves": 3, "failed_index": None, "final_reference": references[-1],
        "steps": sum(m["steps"] for m in moves), "duration_s": sum(m["duration_s"] for m in moves)}


class ValidateAscentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Coarse unit-only clocks keep schema checks fast. The CLI's authority
        # matrix is separately tested to require native 2/1ms jobs.
        cls.fixture = synthetic_fixture(dt=.1)

    def setUp(self):
        stack = self.enterContext(ExitStack())
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_resetData"):
            stack.enter_context(mock.patch.object(mujoco, name, side_effect=AssertionError("Forbidden " + name)))
        for name in ("make_ascending_fixture", "run_ascending_sequence", "execute_transfer", "execute_static_hold"):
            stack.enter_context(mock.patch.object(study.ascent, name, side_effect=AssertionError("No native episode " + name)))

    def test_field_walk_preserves_immutable_nested_evidence_and_identity_is_not_json_equality(self):
        @dataclass(frozen=True)
        class Reference:
            contact_intent: object
        reference = Reference(MappingProxyType({Limb.LEFT_HAND: MappingProxyType({"nested": (1., 2.)})}))
        result = {"moves": [{"final_reference": reference}, {"incoming_reference": reference}]}
        self.assertTrue(study._reference_identity(result)[0]["same_object"])
        equal_but_distinct = Reference(reference.contact_intent)
        result["moves"][1]["incoming_reference"] = equal_but_distinct
        self.assertFalse(study._reference_identity(result)[0]["same_object"])
        self.assertEqual(study.wb._evidence_value(reference), study.wb._evidence_value(equal_but_distinct))
        self.assertIs(result["moves"][0]["final_reference"], reference)

    def test_native_orchestration_passes_exact_previous_reference_and_stops_failed_live_execution(self):
        model, scene, profile, seed, inputs = self.fixture
        motion = WholeBodyMotion(Frame((0., 0., 1.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                                 {"waist_yaw": 0., "waist_pitch": 0., "waist_roll": 0.}, prepare_s=2.)
        @dataclass(frozen=True)
        class Reference:
            contact_intent: object
        @dataclass(frozen=True)
        class State:
            contact_configuration: object
            epoch: int
        for fail in (False, True):
            initial_ref = Reference(MappingProxyType(dict(scene.start_configuration)))
            live = {"state": State(initial_ref.contact_intent, 0), "epoch": 0}
            observed = []
            def execute(model, data, scene, profile, incoming, manager, request, *, observer, keep_samples):
                before, integration = live["state"], tuple([float(live["epoch"])])
                goal = {**dict(before.contact_configuration), request.limb: request.target}
                live["epoch"] += 1
                live["state"] = State(MappingProxyType(goal), live["epoch"])
                reference = Reference(MappingProxyType(goal)) if not fail else None
                if observer:
                    observer({"terminal": True}, model, data)
                return {"success": not fail, "status": "SUCCESS" if not fail else "CONTACT_LOSS", "reason": "Mock executor",
                    "steps": 1, "duration_s": .1, "initial_state": before, "final_state": live["state"],
                    "initial_integration_state": integration, "final_integration_state": tuple([float(live["epoch"])]),
                    "final_reference": reference, "final_contacts": goal}
            with mock.patch.object(study.ascent, "make_ascending_fixture", return_value=(model, mujoco.MjData(model),
                        scene, profile, seed, inputs["native_metadata"])), \
                    mock.patch.object(study.ascent, "initialize_static_reference", return_value=(initial_ref, object())), \
                    mock.patch.object(study.ascent, "execute_static_hold", return_value={"success": True, "reason": "Mock source"}), \
                    mock.patch.object(study.ascent, "get_state_summary", side_effect=lambda *a: live["state"]), \
                    mock.patch.object(study.ascent, "_integration_state", side_effect=lambda *a: np.array([float(live["epoch"])])), \
                    mock.patch.object(study.ascent, "choose_hand_reference", return_value=SimpleNamespace(feasible=True, motion=motion)) as select, \
                    mock.patch.object(study.ascent, "foot_motion", return_value=motion), \
                    mock.patch.object(study.ascent, "execute_transfer", side_effect=execute) as stock:
                result = NATIVE_SEQUENCE(profile, .1, timing="fast",
                    observer=lambda row, *a: observed.append(dict(row)), keep_samples=True)
            self.assertEqual(stock.call_count, 1 if fail else 3)
            self.assertEqual(select.call_count, 1 if fail else 2)
            self.assertIs(result["moves"][0]["incoming_reference"], initial_ref)
            self.assertEqual(result["failed_index"], 0 if fail else None)
            if not fail:
                for i in (1, 2):
                    self.assertIs(stock.call_args_list[i].args[4], result["moves"][i - 1]["final_reference"])
                    self.assertIs(result["moves"][i]["incoming_reference"], result["moves"][i - 1]["final_reference"])
                self.assertTrue(all(b["integration_state_equal"] for b in result["sequence_boundaries"]))
            self.assertEqual([r["move_index"] for r in observed], list(range(stock.call_count)))

    def test_positive_schema_checks_all_three_native_move_contracts_without_physics(self):
        model, _, _, _, inputs = self.fixture
        result = synthetic_record(self.fixture)
        saved = study.wb._evidence_value(result)
        motion = study.audit_motion(model, saved)
        self.assertTrue(study._case_verdict(model, saved, inputs, motion, study._reference_identity(result)))

    def test_forged_success_capture_foot_persistence_and_handoff_are_rejected(self):
        model, _, _, _, inputs = self.fixture
        original = synthetic_record(self.fixture)
        motion = study.audit_motion(model, study.wb._evidence_value(original))
        for mode in ("capture", "foot_load", "foot_persistence", "foot_attachment", "force", "warmstart", "identity", "candidate", "source"):
            saved = study.wb._evidence_value(original)
            identity = study._reference_identity(original)
            if mode == "capture":
                saved["moves"][0]["capture"]["gap_m"] = .5
            elif mode == "foot_load":
                saved["moves"][1]["load_duration_s"] = .1
            elif mode == "foot_persistence":
                saved["moves"][1]["acquisition"]["sustained_s"] = .01
            elif mode == "foot_attachment":
                saved["moves"][1]["final_state"]["foot_states"]["LEFT_FOOT"]["idealized_attachment"] = "fake"
            elif mode == "force":
                saved["moves"][0]["samples"][0]["qfrc_applied"][0] = 1.
            elif mode == "warmstart":
                saved["moves"][1]["initial_integration_state"][0] += 1.
            elif mode == "identity":
                identity[0]["same_object"] = False
            elif mode == "candidate":
                saved["candidates"][0]["selection"]["assessments"].pop()
            else:
                saved["initial_static"]["readiness"]["duration"] = .01
            with self.subTest(mode=mode):
                self.assertFalse(study._case_verdict(model, saved, inputs, motion, identity))

    def test_shorter_exhaustion_requires_three_local_rom_witnesses_and_zero_native_moves(self):
        model, _, _, _, inputs = self.fixture
        saved = study.wb._evidence_value(synthetic_record(self.fixture))
        initial = saved["initial_state"]
        saved.update(success=False, status="BOUNDED_CANDIDATE_SET_EXHAUSTED", moves=[], final_state=initial,
            final_integration_state=saved["initial_integration_state"], completed_moves=0, steps=0, duration_s=0.,
            failed_index=0, final_reference=None, candidates=saved["candidates"][:1])
        selection = saved["candidates"][0]["selection"]
        selection.update(feasible=False, selected_index=None, selected_name=None)
        joint = model.joint("left_knee")
        for _, assessment in selection["assessments"]:
            qpos = list(initial["qpos"])
            qpos[int(joint.qposadr[0])] = float(joint.range[0])
            assessment.update(feasible=False, classification="ROM_LIMITED_SEARCH")
            assessment["diagnostics"].update(active_rom_limits=["left_knee"], failed_qpos=qpos,
                failed_result={"converged": False, "qpos": qpos, "collisions": []})
        self.assertTrue(study._case_verdict(model, saved, inputs, None, [], "candidate_exhaustion"))
        selection["assessments"][1][1]["diagnostics"]["active_rom_limits"] = []
        self.assertFalse(study._case_verdict(model, saved, inputs, None, [], "candidate_exhaustion"))
        # The support negative needs a numeric deficit against the actual declared
        # capacity, not merely an INFEASIBLE flag or an arbitrary negative number.
        for _, assessment in selection["assessments"]:
            capacity = inputs["expected_hand_capacity_N"]["left_hand"]
            assessment.update(classification="SUPPORT_INFEASIBLE")
            assessment["diagnostics"]["actual_candidate_support"] = {"admitted": False,
                "contacts": {l: h for l, h in initial["contact_configuration"].items() if l != "RIGHT_HAND"}, "margins": {"LEFT_HAND":
                    {"capacity_N": capacity, "load_N": capacity + 1., "margin_N": -1.}}}
        self.assertTrue(study._case_verdict(model, saved, inputs, None, [], "support_infeasible"))
        selection["assessments"][0][1]["diagnostics"]["actual_candidate_support"]["margins"]["LEFT_HAND"]["margin_N"] = -2.
        self.assertFalse(study._case_verdict(model, saved, inputs, None, [], "support_infeasible"))
        for negative, certificate in (("outside_workspace", "beyond_reach"), ("blocked_path", "blocked_path")):
            with mock.patch.object(study.envelope, "_negative_certificate", return_value=True) as verify:
                self.assertTrue(study._case_verdict(model, saved, inputs, None, [], negative))
                self.assertEqual(verify.call_count, 3)
                self.assertTrue(all(c.args[2] == certificate for c in verify.call_args_list))
            with mock.patch.object(study.envelope, "_negative_certificate", return_value=False):
                self.assertFalse(study._case_verdict(model, saved, inputs, None, [], negative))

    def test_timing_selection_is_fastest_certified_not_native_success_flag(self):
        runs = [{"profile_name": "baseline", "timing": t, "dt_s": .002, "negative": None,
                 "physical_success": t != "fast", "native_success": True} for t in study.TIMINGS]
        self.assertEqual(study._select_timing(runs), "moderate")
        self.assertIsNone(study._select_timing([]))

    def test_runner_preserves_failure_and_nonfinite_cannot_become_certified_null(self):
        model, _, _, _, inputs = self.fixture
        result = synthetic_record(self.fixture)
        result["initial_integration_state"][0] = float("nan")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(study, "_fixture_metadata", return_value=(model, inputs)), \
                mock.patch.object(study.ascent, "run_ascending_sequence", return_value=result) as native:
            summary = study._run_case("baseline", "fast", .002, Path(directory), {})
            self.assertFalse(summary["accepted"])
            self.assertFalse(summary["finite_episode"])
            packet = json.loads(Path(summary["evidence_json"]).read_text())
            self.assertEqual(packet["status"], result["status"])
            self.assertIn("Original native result", summary["audit_error"])
            native.assert_called_once_with(study.study_profiles()["baseline"], .002, timing="fast", negative=None, keep_samples=True)
            with self.assertRaises(FileExistsError):
                study._run_case("baseline", "fast", .002, Path(directory), {})

    def test_cli_matrix_reuses_selected_trial_and_leaves_failed_timings_honest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            provenance = {"modules": {}, "validator_sha256": "unit", "audit_dependencies": {}}
            def run(name, timing, dt, output, proof, negative=None):
                good = timing != "fast"
                return {"name": f"{name}_{timing}_{dt}", "profile_name": name, "timing": timing, "dt_s": dt,
                        "negative": negative, "accepted": good, "physical_success": good and negative is None,
                        "evidence_json": "unit-no-native-file", "evidence_sha256": "unit"}
            with mock.patch.object(study, "ROOT", root), mock.patch.object(study, "_provenance", return_value=provenance), \
                    mock.patch.object(study, "_run_case", side_effect=run) as execute, redirect_stdout(io.StringIO()):
                self.assertEqual(study.main(["--output", str(root / "outputs" / "unit")]), 0)
            report = json.loads((root / "outputs" / "unit" / "report.json").read_text())
            self.assertEqual(report["demo_timing"], "moderate")
            self.assertEqual(execute.call_count, 14)
            self.assertEqual(sum(c.args[:3] == ("baseline", "moderate", .002) for c in execute.call_args_list), 1)
            self.assertFalse(next(r for r in report["runs"] if r["timing"] == "fast")["accepted"])

    def test_cli_protects_preserved_outputs(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            study.main(["--output", str(study.ROOT / "outputs" / "stage5-preserved-unit")])


if __name__ == "__main__":
    unittest.main()
