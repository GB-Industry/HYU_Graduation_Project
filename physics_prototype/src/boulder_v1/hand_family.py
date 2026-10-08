"""Fixed-scene hand-family fixture and diagnostics, not profile personalization."""
from __future__ import annotations

from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco

from .contact_ik import solve_contact_pose
from .mjcf_builder import build_mjcf
from .schema import Limb
from .single_hand import SingleHandRequest, execute_single_hand, make_single_hand_fixture
from .static_control import execute_static_hold
from .static_state import initialize_static_reference


def make_family_scene():
    """One unchanged Stage4 stance, mirrored 60mm targets, and a far diagnostic hold."""
    _, _, scene, _, _ = make_single_hand_fixture()
    left = scene.region("left_hand")
    target = replace(left, id="left_reach_target", position=(*left.position[:2], left.position[2] + .06))
    far = replace(scene.region("reach_target"), id="unreachable_target",
                  position=(*scene.region("reach_target").position[:2], left.position[2] + 2.))
    return replace(scene, contact_regions=(*scene.contact_regions, target, far))


def make_hand_family_fixture(timestep=.002, *, profile=None):
    """Compile production physics; all profiles use the same wall and hold positions."""
    _, _, _, default_profile, seed = make_single_hand_fixture(timestep)
    profile = default_profile if profile is None else profile
    scene = make_family_scene()
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    return model, mujoco.MjData(model), scene, profile, seed


def run_hand_family_benchmark(timestep=.002, *, limb=Limb.RIGHT_HAND, profile=None,
                              pose_perturbation_rad=0., acquisition_policy="endpoint_settle",
                              observer=None, keep_samples=True):
    """Scratch retarget, initial 2s static hold, then authoritative native execution."""
    if not isinstance(limb, Limb) or not limb.is_hand:
        raise ValueError("Hand family requires a hand Limb")
    model, data, scene, profile, seed = make_hand_family_fixture(timestep, profile=profile)
    seed[model.joint("waist_yaw").qposadr[0]] += pose_perturbation_rad
    retarget = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
    if not retarget.admitted:
        return {"success": False, "status": "INITIALIZATION_FAILURE", "reason": retarget.reason,
                "retarget": retarget, "steps": 0}
    reference, manager = initialize_static_reference(model, data, scene, profile, retarget.qpos, scene.start_configuration)
    initial = execute_static_hold(model, data, scene, profile, reference, manager, duration=2., settle=1.,
                                  score_window=.5, keep_samples=False)
    if not initial["success"]:
        return {"success": False, "status": "INITIALIZATION_FAILURE", "reason": initial["reason"],
                "retarget": retarget, "initial_static": initial, "steps": initial["steps"]}
    request = SingleHandRequest(limb, limb.value.lower(),
                                "reach_target" if limb == Limb.RIGHT_HAND else "left_reach_target",
                                acquisition_policy=acquisition_policy)
    result = execute_single_hand(model, data, scene, profile, reference, manager, request,
                                 observer=observer, keep_samples=keep_samples)
    result.update(retarget=retarget, initial_static=initial, profile=profile.to_dict(),
                  pose_perturbation_rad=pose_perturbation_rad)
    return result
