"""One declared ascending chain, with fresh-state references and no route search."""
from dataclasses import replace
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_geometry import Frame, canonical_geometry
from .foot_transfer import FootRequest
from .locomotion import get_state_summary
from .mjcf_builder import build_mjcf
from .morphology_envelope import make_envelope_fixture, study_profiles
from .reference_candidates import choose_hand_reference
from .schema import ClimberProfile, Limb
from .static_control import execute_static_hold
from .static_state import _integration_state, initialize_static_reference
from .transfers import TransferRequest, _request_error, execute_transfer
from .whole_body_motion import WholeBodyMotion, reference_frames


TIMINGS = {"conservative": (4., 5.), "moderate": (3., 4.), "fast": (2., 3.)}


def make_ascending_fixture(profile=None, timestep=.002, *, negative=None):
    profile = study_profiles()["baseline"] if profile is None else profile
    case = {"outside_workspace": "beyond_reach", "blocked_path": "blocked_path"}.get(negative, "nominal")
    _, _, scene, _, seed, _ = make_envelope_fixture(profile, case, timestep)
    if negative not in (None, "outside_workspace", "blocked_path", "support_infeasible", "candidate_exhaustion"):
        raise ValueError("Unknown ascending negative")
    # Common physical task across morphologies: one tenth of baseline leg reach.
    step = .1 * ClimberProfile("geometry_reference").leg_reach
    source = scene.region(scene.start_configuration[Limb.LEFT_FOOT])
    scene = replace(scene, contact_regions=tuple(
        replace(r, position=(*r.position[:2], source.position[2] + step)) if r.id == "foot_target" else
        replace(r, grip_quality=.04) if negative == "support_infeasible" and r.id == "reach_target"
        else r for r in scene.contact_regions))
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    metadata = {"fixture": "ascending_stage5.3", "profile": profile.to_dict(), "scene": scene.to_dict(),
                "seed_qpos": tuple(seed), "dt_s": timestep, "negative": negative,
                "step_height_m": step, "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                "contact_geometry": {r.id: canonical_geometry(r) for r in scene.contact_regions}}
    return model, mujoco.MjData(model), scene, profile, seed, metadata


def foot_motion(model, data, scene, profile, request, prepare_s):
    """An explicit stance-centering/upward guide; only hinges are commanded."""
    root, frames = reference_frames(model, data.qpos)
    source = canonical_geometry(scene.region(request.source)).foot_frame
    goal = canonical_geometry(scene.region(request.target)).foot_frame
    step = max(0., goal.position[2] - source.position[2])
    lateral = np.array(source.rotation)[:, 0]
    opposite = Limb.RIGHT_FOOT if request.limb == Limb.LEFT_FOOT else Limb.LEFT_FOOT
    shift = .12 * float((np.array(frames[opposite].position) - root.position) @ lateral)
    shift = np.clip(shift, -.04 * profile.leg_reach, .04 * profile.leg_reach)
    position = np.array(root.position) + shift * lateral + [0., 0., min(.36 * step, .04 * profile.leg_reach)]
    waist = {name: float(data.qpos[int(model.joint(name).qposadr[0])])
             for name in ("waist_yaw", "waist_pitch", "waist_roll")}
    return WholeBodyMotion(Frame(tuple(position), root.rotation), waist, prepare_s)


def run_ascending_sequence(profile=None, timestep=.002, *, timing="conservative", negative=None,
                           observer=None, keep_samples=True):
    if timing not in TIMINGS:
        raise ValueError("Unknown bounded ascending timing")
    prepare_s, reach_s = TIMINGS[timing]
    if negative == "candidate_exhaustion" and profile is None:
        profile = study_profiles()["shorter"]
    model, data, scene, profile, seed, metadata = make_ascending_fixture(profile, timestep, negative=negative)
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    initial_static = execute_static_hold(model, data, scene, profile, reference, manager,
                                        duration=2., settle=1., score_window=.5, keep_samples=False)
    initial = get_state_summary(model, data, manager)
    initial_integration = tuple(_integration_state(model, data))
    declarations = ((Limb.RIGHT_HAND, "reach_target"), (Limb.LEFT_FOOT, "foot_target"),
                    (Limb.LEFT_HAND, "left_reach_target"))
    intent = dict(reference.contact_intent)
    requests = []
    for limb, target in declarations:
        foot_request = None
        if limb.is_foot:
            start = canonical_geometry(scene.region(intent[limb])).foot_frame
            goal = canonical_geometry(scene.region(target)).foot_frame
            foot_request = FootRequest(limb, intent[limb], target,
                                       lift_m=max(0., goal.position[2] - start.position[2]) + .02,
                                       airborne_pitch_rad=-.25)
        request = TransferRequest(limb, intent[limb], target, dict(intent),
                                  support_contacts={l: h for l, h in intent.items() if l != limb},
                                  hand_reach_s=reach_s, foot_request=foot_request)
        error = _request_error(scene, request, timestep)
        if error:
            raise ValueError(error[1])
        requests.append(request)
        intent[limb] = target
    moves, candidates, boundaries = [], [], []
    status, reason = "INITIALIZATION_FAILURE", initial_static["reason"]
    failed = 0 if not initial_static["success"] else None
    if initial_static["success"]:
        for index, request in enumerate(requests):
            before = get_state_summary(model, data, manager)
            before_integration = tuple(_integration_state(model, data))
            if index:
                boundary = {"before_move_index": index - 1, "after_move_index": index,
                            "full_state_equal": moves[-1]["final_state"] == before,
                            "integration_state_equal": moves[-1]["final_integration_state"] == before_integration}
                boundaries.append(boundary)
                if not all(boundary[key] for key in ("full_state_equal", "integration_state_equal")):
                    raise RuntimeError("Ascending boundary changed the exact native terminal state")
            if request.limb.is_hand:
                selection = choose_hand_reference(model, data, scene, profile, reference, manager, request,
                                                   prepare_s=prepare_s)
                candidates.append({"move_index": index, "selection": selection})
                if not selection.feasible:
                    status, reason, failed = selection.classification, selection.reason, index
                    break
                motion = selection.motion
            else:
                motion = foot_motion(model, data, scene, profile, request, 4.)
            if before_integration != tuple(_integration_state(model, data)):
                raise RuntimeError("Reference generation mutated the authoritative ascending episode")

            def bridge(row, display_model, display_data):
                row.update(move_index=index, move_count=len(requests))
                observer(row, display_model, display_data)

            # The exact preceding reference object enters the executor, not a
            # newly admitted assessment source or an adopted scratch pose.
            incoming_reference = reference
            move = execute_transfer(model, data, scene, profile, incoming_reference, manager,
                                     replace(request, whole_body=motion),
                                     observer=bridge if observer else None, keep_samples=keep_samples)
            move["incoming_reference"] = incoming_reference
            moves.append(move)
            status, reason = move["status"], move["reason"]
            if not move["success"] or move["final_reference"] is None:
                failed = index
                break
            reference = move["final_reference"]
    final = get_state_summary(model, data, manager)
    success = failed is None and len(moves) == len(requests) and all(m["success"] for m in moves)
    return {"success": success, "status": status, "reason": reason, "kind": "ascending_sequence",
            "fixture": "ascending_stage5.3", "fixture_inputs": metadata, "profile": profile.to_dict(),
            "timing": timing, "timing_parameters": {"prepare_s": prepare_s, "hand_reach_s": reach_s,
                                                       "foot_prepare_s": 4.},
            "negative": negative, "dt_s": timestep, "initial_static": initial_static,
            "initial_state": initial, "final_state": final,
            "initial_integration_state": initial_integration,
            "final_integration_state": tuple(_integration_state(model, data)),
            "steps": sum(m["steps"] for m in moves), "duration_s": sum(m["duration_s"] for m in moves),
            "completed_moves": sum(m["success"] for m in moves), "failed_index": failed,
            "moves": moves, "candidates": candidates, "sequence_boundaries": boundaries,
            "contact_trace": [dict(initial.contact_configuration), *[m["final_contacts"] for m in moves]],
            "final_contacts": dict(final.contact_configuration), "final_reference": reference if success else None}
