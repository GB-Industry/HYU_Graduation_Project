"""Explicit whole-body reference goals; no live root command or force channel."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import mujoco
import numpy as np

from .contact_geometry import Frame
from .schema import Limb
from .whole_body_reference import _validated_frame, solve_whole_body_reference


@dataclass(frozen=True)
class WholeBodyMotion:
    root_target: Frame
    waist_target: Mapping[str, float]
    prepare_s: float = 4.

    def __post_init__(self):
        if isinstance(self.waist_target, Mapping):
            object.__setattr__(self, "waist_target", MappingProxyType(dict(self.waist_target)))


def reference_frames(model, qpos):
    """Capture exact world frames from an admitted reference on isolated scratch."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.eq_active[:] = False
    mujoco.mj_forward(model, data)
    root = model.body("climber_root").id
    root_frame = Frame(tuple(data.xpos[root]), tuple(map(tuple, data.xmat[root].reshape(3, 3))))
    frames = {}
    for limb in Limb:
        name = limb.value.lower()
        site = data.site(f"{name}_site")
        rotation = site.xmat if limb.is_hand else data.geom(f"{name}_geom").xmat
        frames[limb] = Frame(tuple(site.xpos), tuple(map(tuple, rotation.reshape(3, 3))))
    return root_frame, frames


def prepare_whole_body(model, scene, profile, reference, motion, *, hand_target=None,
                       limb=None, target=None):
    """Preflight source/end geometry and sample a four-contact preparation path.

    All root goals live exclusively in scratch references. Consumers apply only
    motor hinge targets through the existing rate shaper and torque controller.
    A converged trajectory is not a dynamic support or capture certificate.
    """
    if not isinstance(motion, WholeBodyMotion):
        raise ValueError("Whole-body motion requires a WholeBodyMotion")
    _validated_frame(motion.root_target, "whole-body root target")
    dt = float(model.opt.timestep)
    if (isinstance(motion.prepare_s, bool) or not isinstance(motion.prepare_s, (float, int))
            or not math.isfinite(motion.prepare_s) or motion.prepare_s <= 0
            or abs(motion.prepare_s / dt - round(motion.prepare_s / dt)) > 1e-8):
        raise ValueError("Whole-body preparation needs positive integral native intervals")
    root, frames = reference_frames(model, reference.qpos)
    waist = {n: reference.target_pose[n] for n in ("waist_yaw", "waist_pitch", "waist_roll")}
    # Imports are local because maintained primitives own interpolation/shaping.
    from .single_hand import reach_frame
    from dataclasses import replace
    from .static_state import initialize_static_reference
    seed = np.array(reference.qpos)
    path = []
    count = max(1, math.ceil(motion.prepare_s / .05))
    for index in range(count + 1):
        time = motion.prepare_s * index / count
        s = time / motion.prepare_s
        b = s ** 3 * (10. + s * (-15. + 6. * s))
        frame = reach_frame(root, motion.root_target, time, motion.prepare_s, 0.)
        solution = solve_whole_body_reference(
            model, scene, profile, seed, reference.contact_intent, root_target=frame,
            waist_target={n: (1. - b) * waist[n] + b * motion.waist_target[n] for n in waist},
            support_frames=frames)
        if not solution.converged:
            raise ValueError("Whole-body preparation reference failed: " + solution.reason)
        seed = np.array(solution.qpos)
        initialize_static_reference(model, mujoco.MjData(model),
                                    replace(scene, start_configuration=dict(reference.contact_intent)),
                                    profile, seed, reference.contact_intent)
        path.append(seed.copy())
    endpoint = None
    if hand_target is not None:
        endpoint = solve_whole_body_reference(
            model, scene, profile, seed, {**reference.contact_intent, limb: target},
            root_target=motion.root_target, waist_target=motion.waist_target,
            support_frames=frames, hand_targets={limb: hand_target},
            allowed_hand_holds={limb: (reference.contact_intent[limb], target)})
        if not endpoint.converged:
            raise ValueError("Whole-body reach reference failed: " + endpoint.reason)
    return frames, tuple(path), endpoint


def preparation_reference(model, path, time, duration):
    """Interpolate sampled hinge references; root entries are never actuated."""
    fraction = min(1., max(0., time / duration)) * (len(path) - 1)
    index = min(len(path) - 2, int(fraction))
    b = fraction - index
    tangent = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1., path[index], path[index + 1])
    result = path[index].copy()
    mujoco.mj_integratePos(model, result, tangent, b)
    return result
