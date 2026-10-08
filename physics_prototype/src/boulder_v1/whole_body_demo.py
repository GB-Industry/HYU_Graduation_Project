"""Stage5.1 explicit, morphology-derived demonstrations, not a route planner."""
from dataclasses import replace
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_geometry import Frame, canonical_geometry
from .contact_ik import solve_hand_reference
from .mjcf_builder import build_mjcf
from .schema import Limb
from .static_control import execute_static_hold
from .static_state import initialize_static_reference
from .transfers import (TransferRequest, execute_transfer, execute_transfer_sequence,
                        make_transfer_fixture)
from .whole_body_motion import WholeBodyMotion, reference_frames


def make_whole_body_fixture(timestep=.002):
    """Fixed validated source holds; derive planted crouch and reach from the rig."""
    _, _, original, profile, seed = make_transfer_fixture(timestep)
    rise = .25 * profile.arm_reach
    regions = tuple(replace(region, position=(region.position[0], region.position[1],
                                              original.region("right_hand" if region.id == "reach_target"
                                                              else "left_hand").position[2] + rise))
                    if region.id in ("reach_target", "left_reach_target") else region
                    for region in original.contact_regions)
    scene = replace(original, contact_regions=regions)
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    seed = seed.copy()
    for side in ("left", "right"):
        for name, angle in (("hip_pitch", .5), ("knee", 1.), ("ankle_pitch", .5)):
            seed[int(model.joint(f"{side}_{name}").qposadr[0])] = angle
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    foot = canonical_geometry(scene.region("left_foot")).foot_frame
    planted = np.array(foot.position) + np.array(foot.rotation) @ [0., .01, 0.]
    qa = int(model.joint("root").qposadr[0])
    seed[qa:qa + 3] += planted - scratch.site("left_foot_site").xpos
    for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
        solution = solve_hand_reference(model, seed, limb,
                                        canonical_geometry(scene.region(scene.start_configuration[limb])).hand_frame)
        if not solution.converged:
            raise ValueError("Crouched source arm reference failed: " + solution.reason)
        seed = np.array(solution.qpos)
    return model, mujoco.MjData(model), scene, profile, seed


def hand_motion(model, reference, *, move_index=0, limb=Limb.RIGHT_HAND):
    root, _ = reference_frames(model, reference.qpos)
    angle = math.radians(4. if limb == Limb.RIGHT_HAND else -4.)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array(root.rotation)
    delta = [(-.008 if limb == Limb.RIGHT_HAND else .008), 0., .04 if move_index == 0 else .03]
    target = Frame(tuple(np.array(root.position) + delta), tuple(map(tuple, rotation)))
    return WholeBodyMotion(target, {"waist_yaw": 0., "waist_pitch": .07 + .03 * move_index,
                                    "waist_roll": 0.}, prepare_s=4.)


def run_whole_body_benchmark(timestep=.002, *, kind="right_hand", observer=None, keep_samples=True,
                             negative=None):
    if kind not in ("right_hand", "left_hand", "foot", "sequence"):
        raise ValueError("Unknown whole-body benchmark kind")
    if negative not in (None, "unreachable", "root_unreachable", "waist_rom", "invalid_support"):
        raise ValueError("Unknown whole-body negative")
    model, data, scene, profile, seed = make_whole_body_fixture(timestep)
    reference, manager = initialize_static_reference(model, data, scene, profile, seed, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager,
                                   duration=2., settle=1., score_window=.5, keep_samples=False)
    if not initial["success"]:
        return {"success": False, "status": "INITIALIZATION_FAILURE", "reason": initial["reason"],
                "initial_static": initial}
    limb = Limb.LEFT_FOOT if kind == "foot" else Limb.LEFT_HAND if kind == "left_hand" else Limb.RIGHT_HAND
    motion = hand_motion(model, reference, limb=limb)
    if kind == "foot":
        root, _ = reference_frames(model, reference.qpos)
        c, s = math.cos(math.radians(4.)), math.sin(math.radians(4.))
        motion = WholeBodyMotion(
            Frame(tuple(np.array(root.position) + [.015, 0., .04]),
                  ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))),
            {"waist_yaw": 0., "waist_pitch": .07, "waist_roll": 0.})
    if negative == "root_unreachable":
        motion = replace(motion, root_target=replace(motion.root_target,
                         position=tuple(np.array(motion.root_target.position) + [0., 0., .4])))
    elif negative == "waist_rom":
        motion = replace(motion, waist_target={**motion.waist_target, "waist_pitch": 3.})
    contacts = dict(reference.contact_intent)
    target = "foot_target" if kind == "foot" else "left_reach_target" if limb == Limb.LEFT_HAND else "reach_target"
    request = TransferRequest(limb, contacts[limb], "unreachable_target" if negative == "unreachable" else target,
                              contacts, hand_reach_s=5., whole_body=motion,
                              support_contacts=contacts if negative == "invalid_support" else None)
    if kind == "sequence":
        after = {**contacts, limb: target}
        second_motion = hand_motion(model, reference, move_index=1)
        second_motion = replace(second_motion, root_target=replace(
            second_motion.root_target,
            position=tuple(np.array(motion.root_target.position) + [.008, 0., .03]),
            rotation=tuple(map(tuple, np.array(motion.root_target.rotation) @
                                       np.array(second_motion.root_target.rotation)))))
        second = TransferRequest(Limb.LEFT_HAND, "left_hand", "left_reach_target", after,
                                  hand_reach_s=5., whole_body=second_motion)
        result = execute_transfer_sequence(model, data, scene, profile, (request, second), reference, manager,
                                            observer=observer, keep_samples=keep_samples)
    else:
        result = execute_transfer(model, data, scene, profile, reference, manager, request,
                                   observer=observer, keep_samples=keep_samples)
    result.update(initial_static=initial, initial_reference=reference, kind=kind, profile=profile.to_dict(),
                   negative=negative, dt_s=timestep, reference_mode="whole_body",
                   fixture="whole_body_stage5.1", hand_target_travel_m=.25 * profile.arm_reach)
    return result
