"""Local, scratch-only contact IK; admission is the unchanged Stage3 preflight.

This generates poses, not equilibrium, readiness, or collision-free trajectories.
No live data or manager is accepted, and the compiled model is read-only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from .contact_geometry import FOOT_SITE_OFFSET, Frame
from .mjcf_builder import END_EFFECTOR_SITES
from .runtime import compiled_numerical_issues, validate_reference_pose
from .schema import Affordance, BoulderScene, ClimberProfile, Limb, SourceType
from .static_state import (
    StaticReference, StaticResidual, _canonical_surface, _foot_residual,
    initialize_static_reference,
)


@dataclass(frozen=True)
class ContactIKResidual:
    limb: Limb
    region_id: str
    target: Frame
    position_error: float  # Euclidean metres, relative to the selected IK target.
    canonical_position_error: float
    orientation_error: float  # Radians; normal swing or SO(3), according to the task.
    orientation_alignment: float
    signed_geom_distance: float
    foot_residual: StaticResidual | None = None
    foot_reason: str = ""
    foot_site_normal_residual: float | None = None
    foot_minimum_sole_distance: float | None = None
    foot_tangential_offset: tuple[float, float] | None = None


@dataclass(frozen=True)
class CollisionResidual:
    geoms: tuple[str, str]
    signed_distance: float


@dataclass(frozen=True)
class ContactIKResult:
    """Immutable evidence, including measured seed diagnostics.

    ``admitted`` alone authorizes a full four-contact seed. ``converged`` means
    geometric IK convergence (or an already admitted, exactly preserved seed).
    A geometric solution can fail reaction/penetration/ROM admission. Failed
    qpos is diagnostic only, never a fallback certified reference. ``iterations``
    counts accepted tangent steps, not native physics steps. ``reference`` is
    present only after exact admission on disposable data. Reference generation
    always sets admitted=False, reference=None, even at a route endpoint.
    """

    qpos: tuple[float, ...]
    converged: bool
    admitted: bool
    iterations: int
    residuals: tuple[ContactIKResidual, ...]
    reason: str
    reference: StaticReference | None
    initial_residuals: tuple[ContactIKResidual, ...]
    initial_reason: str
    collisions: tuple[CollisionResidual, ...]


@dataclass(frozen=True)
class HandReferenceResult:
    """Arm-only task evidence, never contact admission or dynamic acceptance.

    Errors are Euclidean metres and normal-swing radians; alignment is the
    world +Z normal dot product. Iterations count accepted scratch tangent steps.
    qpos preserves every coordinate outside the selected five arm hinges.
    A failed reference is diagnostic only. No collision, capacity, trajectory,
    or readiness guarantee follows even from converged=True.
    """

    qpos: tuple[float, ...]
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float
    orientation_alignment: float
    reason: str


def _rotation_error(current, target, normal_only):
    if normal_only:
        a, b = current[:, 2], target[:, 2]
        cross = np.cross(a, b)
        sine, cosine = float(np.linalg.norm(cross)), float(np.clip(a @ b, -1., 1.))
        if sine > 1e-12:
            return cross * (math.atan2(sine, cosine) / sine)
        # A deterministic tangent axis also resolves an exactly reversed normal.
        return np.zeros(3) if cosine >= 0 else math.pi * current[:, 0]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).ravel())
    if quat[0] < 0:
        quat *= -1
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, quat, 1.)
    return error


def solve_hand_reference(
    model: mujoco.MjModel,
    measured_qpos: Sequence[float],
    limb: Limb,
    target: Frame,
    *,
    max_iterations: int = 80,
    tolerance: float = 1e-7,
) -> HandReferenceResult:
    """Generate a five-hinge hand reference in the measured root/torso frame.

    Only shoulder pitch/roll/yaw, elbow and wrist of the selected hand may change
    on disposable data. Position and outward-facing +Z normal are solved; target
    tangent-frame twist is intentionally ignored. Root, waist and all other
    coordinates remain bit-for-bit measured, including integration round-off.
    Each trial arm tangent step has Euclidean norm at most 0.1 radians.

    Malformed/nonfinite inputs, nonunit root quaternions and ANY measured hinge
    outside compiled ROM raise ValueError before forward kinematics. Native
    soft-limit deviations are not clipped or silently repaired. Local failures
    return measured task errors, not a global infeasibility verdict. No live
    MjData is accepted. This pure task generator neither steers by collisions
    nor attempts capture/admission: the executor owns physical guards, PD/FF,
    velocity-reference continuity, capture and readiness.
    """
    if not isinstance(limb, Limb) or not limb.is_hand:
        raise ValueError("Select a hand Limb for an arm-only reference")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0 or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Require nonnegative integer max_iterations and positive finite tolerance")
    try:
        seed = np.asarray(measured_qpos, dtype=float).copy()
    except (TypeError, ValueError):
        raise ValueError("Supply a complete finite measured model.nq pose") from None
    if seed.shape != (model.nq,) or not np.isfinite(seed).all():
        raise ValueError("Supply a complete finite measured model.nq pose")
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled numerics: {', '.join(issues)}")
    if (np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1
            or any(int(kind) not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
                   for kind in model.jnt_type)):
        raise ValueError("Hand references require one free root and scalar hinges")
    validate_reference_pose(model, seed)
    if not isinstance(target, Frame):
        raise ValueError("Supply a world target Frame")
    position = np.asarray(target.position, dtype=float).copy()
    rotation = np.asarray(target.rotation, dtype=float).copy()
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Hand target must have a finite position and an SO(3) rotation")
    side = limb.value.lower().removesuffix("_hand")
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
                       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITES[limb])
    if site < 0 or np.any(joints < 0) or np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Missing compiled five-hinge arm or hand site")
    qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limited = model.jnt_limited[joints].astype(bool)
    lower = np.where(limited, model.jnt_range[joints, 0], -np.inf)
    upper = np.where(limited, model.jnt_range[joints, 1], np.inf)
    frozen = np.ones(model.nq, dtype=bool)
    frozen[qadr] = False
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)

    def measure(with_jacobian=False):
        current = scratch.site_xmat[site].reshape(3, 3)
        ep = position - scratch.site_xpos[site]
        er = _rotation_error(current, rotation, True)
        jacobian = None
        if with_jacobian:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, scratch, jp, jr, site)
            projection = np.eye(3) - np.outer(current[:, 2], current[:, 2])
            jacobian = np.vstack((jp[:, dofs], .12 * (projection @ jr[:, dofs])))
        return (np.r_[ep, .12 * er], jacobian, float(np.linalg.norm(ep)),
                float(np.linalg.norm(er)), float(np.clip(current[:, 2] @ rotation[:, 2], -1., 1.)))

    iterations, damping = 0, .001
    reason = "Iteration budget exhausted"
    for _ in range(max_iterations + 1):
        error, jacobian, position_error, orientation_error, alignment = measure(True)
        if position_error <= tolerance and orientation_error <= tolerance:
            validate_reference_pose(model, scratch.qpos)
            return HandReferenceResult(tuple(float(v) for v in scratch.qpos), True, iterations,
                                       position_error, orientation_error, alignment,
                                       "Arm-only geometric reference; no contact admission attempted")
        if iterations >= max_iterations:
            break
        current = scratch.qpos.copy()
        score = float(error @ error)
        accepted = False
        for _ in range(5):
            active = np.ones(5, dtype=bool)
            step = np.zeros(5)
            for _ in range(6):
                j = jacobian[:, active]
                step[:] = 0.
                step[active] = np.linalg.solve(j.T @ j + damping ** 2 * np.eye(j.shape[1]), j.T @ error)
                outward = (((current[qadr] <= lower + 1e-10) & (step < 0))
                           | ((current[qadr] >= upper - 1e-10) & (step > 0)))
                if not np.any(outward & active):
                    break
                active[outward] = False
            step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-12))
            tangent = np.zeros(model.nv)
            tangent[dofs] = step
            for backtrack in range(12):
                candidate = current.copy()
                mujoco.mj_integratePos(model, candidate, tangent, .5 ** backtrack)
                candidate[qadr] = np.clip(candidate[qadr], lower, upper)
                candidate[frozen] = seed[frozen]
                scratch.qpos[:] = candidate
                mujoco.mj_forward(model, scratch)
                trial, _, _, _, _ = measure()
                if np.isfinite(trial).all() and float(trial @ trial) < score - 1e-24:
                    accepted = True
                    break
            if accepted:
                iterations += 1
                damping = max(.00001, damping * .5)
                break
            damping *= 10.
        if not accepted:
            scratch.qpos[:] = current
            mujoco.mj_forward(model, scratch)
            reason = "Bounded arm-only local search stalled"
            break
    _, _, position_error, orientation_error, alignment = measure()
    return HandReferenceResult(tuple(float(v) for v in scratch.qpos), False, iterations,
                               position_error, orientation_error, alignment,
                               f"{reason}; no contact admission attempted. "
                               "Local failure is not a global infeasibility proof")


def solve_foot_reference(
    model: mujoco.MjModel,
    measured_qpos: Sequence[float],
    limb: Limb,
    target: Frame,
    *,
    max_iterations: int = 80,
    tolerance: float = 1e-7,
) -> HandReferenceResult:
    """Generate a six-hinge leg reference, never physical contact admission.

    Only the selected foot's hip pitch/roll/yaw, knee and ankle pitch/roll may
    change on disposable data. Target position is the world foot END site;
    target rotation is the actual shoe geom's full SO(3), with +Z sole normal.
    Tangential centering and twist are not discarded. A sole-plane endpoint
    must already include FOOT_SITE_OFFSET (11mm); no contact compression or
    additional offset is inferred here. Geometry comes from the compiled model,
    including the knee's axis, rather than an anatomical sign convention.

    Every other qpos coordinate remains bit-for-bit measured. Each trial leg
    tangent step has Euclidean norm at most 0.1 radians. ANY measured hinge
    outside compiled ROM, a nonunit root quaternion or malformed/nonfinite
    input raises ValueError before forward kinematics, without clipping live
    soft-limit deviations. The model is read-only and no live MjData is accepted.

    The immutable HandReferenceResult layout is reused, but orientation_error
    here is full SO(3) radians and qpos may change six leg hinges, not five arm
    hinges. Alignment remains the world +Z normal dot product. Iterations count
    accepted scratch tangent steps. Failed qpos is diagnostic only; even a
    converged reference claims no collision, support, acquisition, readiness or
    dynamically trackable trajectory. The executor owns those physical checks.
    """
    if not isinstance(limb, Limb) or not limb.is_foot:
        raise ValueError("Select a foot Limb for a leg-only reference")
    try:
        valid_tolerance = math.isfinite(tolerance) and tolerance > 0
    except (TypeError, ValueError, OverflowError):
        valid_tolerance = False
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0 or not valid_tolerance):
        raise ValueError("Require nonnegative integer max_iterations and positive finite tolerance")
    try:
        seed = np.asarray(measured_qpos, dtype=float).copy()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Supply a complete finite measured model.nq pose") from None
    if seed.shape != (model.nq,) or not np.isfinite(seed).all():
        raise ValueError("Supply a complete finite measured model.nq pose")
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled numerics: {', '.join(issues)}")
    if (np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1
            or any(int(kind) not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
                   for kind in model.jnt_type)):
        raise ValueError("Foot references require one free root and scalar hinges")
    validate_reference_pose(model, seed)
    if not isinstance(target, Frame):
        raise ValueError("Supply a world target Frame")
    try:
        position = np.asarray(target.position, dtype=float).copy()
        rotation = np.asarray(target.rotation, dtype=float).copy()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Foot target must have a finite position and an SO(3) rotation") from None
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Foot target must have a finite position and an SO(3) rotation")
    side = limb.value.lower().removesuffix("_foot")
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
                       for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITES[limb])
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{limb.value.lower()}_geom")
    if (site < 0 or geom < 0 or np.any(joints < 0)
            or np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)
            or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX
            or model.site_bodyid[site] != model.geom_bodyid[geom]):
        raise ValueError("Missing compiled six-hinge leg or rigid foot site/box shoe")
    qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limited = model.jnt_limited[joints].astype(bool)
    lower = np.where(limited, model.jnt_range[joints, 0], -np.inf)
    upper = np.where(limited, model.jnt_range[joints, 1], np.inf)
    frozen = np.ones(model.nq, dtype=bool)
    frozen[qadr] = False
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)

    def measure(with_jacobian=False):
        current = scratch.geom_xmat[geom].reshape(3, 3)
        ep = position - scratch.site_xpos[site]
        er = _rotation_error(current, rotation, False)
        jacobian = None
        if with_jacobian:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            # Site and shoe are rigid on the same body; their angular Jacobians
            # coincide even if their compiled local rotations differ.
            mujoco.mj_jacSite(model, scratch, jp, jr, site)
            jacobian = np.vstack((jp[:, dofs], .12 * jr[:, dofs]))
        return (np.r_[ep, .12 * er], jacobian, float(np.linalg.norm(ep)),
                float(np.linalg.norm(er)), float(np.clip(current[:, 2] @ rotation[:, 2], -1., 1.)))

    iterations, damping = 0, .001
    reason = "Iteration budget exhausted"
    for _ in range(max_iterations + 1):
        error, jacobian, position_error, orientation_error, alignment = measure(True)
        if position_error <= tolerance and orientation_error <= tolerance:
            validate_reference_pose(model, scratch.qpos)
            return HandReferenceResult(tuple(float(v) for v in scratch.qpos), True, iterations,
                                       position_error, orientation_error, alignment,
                                       "Leg-only geometric reference; no contact admission attempted")
        if iterations >= max_iterations:
            break
        current = scratch.qpos.copy()
        score = float(error @ error)
        accepted = False
        for _ in range(5):
            active = np.ones(6, dtype=bool)
            step = np.zeros(6)
            for _ in range(7):
                j = jacobian[:, active]
                step[:] = 0.
                step[active] = np.linalg.solve(j.T @ j + damping ** 2 * np.eye(j.shape[1]), j.T @ error)
                outward = (((current[qadr] <= lower + 1e-10) & (step < 0))
                           | ((current[qadr] >= upper - 1e-10) & (step > 0)))
                if not np.any(outward & active):
                    break
                active[outward] = False
            step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-12))
            tangent = np.zeros(model.nv)
            tangent[dofs] = step
            for backtrack in range(12):
                candidate = current.copy()
                mujoco.mj_integratePos(model, candidate, tangent, .5 ** backtrack)
                candidate[qadr] = np.clip(candidate[qadr], lower, upper)
                candidate[frozen] = seed[frozen]
                scratch.qpos[:] = candidate
                mujoco.mj_forward(model, scratch)
                trial, _, _, _, _ = measure()
                if np.isfinite(trial).all() and float(trial @ trial) < score - 1e-24:
                    accepted = True
                    break
            if accepted:
                iterations += 1
                damping = max(.00001, damping * .5)
                break
            damping *= 10.
        if not accepted:
            scratch.qpos[:] = current
            mujoco.mj_forward(model, scratch)
            reason = "Bounded leg-only local search stalled"
            break
    _, _, position_error, orientation_error, alignment = measure()
    return HandReferenceResult(tuple(float(v) for v in scratch.qpos), False, iterations,
                               position_error, orientation_error, alignment,
                               f"{reason}; no contact admission attempted. "
                               "Local failure is not a global infeasibility proof")


def solve_contact_pose(
    model: mujoco.MjModel,
    scene: BoulderScene,
    profile: ClimberProfile,
    initial_qpos: Sequence[float],
    contact_intent: Mapping[Limb, str],
    *,
    freeze_root: bool = False,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> ContactIKResult:
    """Retarget a complete four-HOLD intent, with exact scratch admission.

    Route intent replaces episode starts only in a disposable scene copy. An
    already admitted seed is returned bit-for-bit, regardless of IK tolerance.
    Otherwise quaternions are normalized and scalar ROM is projected on scratch;
    all optimization updates, including the free root, use mj_integratePos.
    Local failure does not prove that the route is globally infeasible.
    Malformed input/unsupported models raise ValueError; local failure returns
    measured evidence with admitted=False. Tolerance never relaxes admission.

    freeze_root=True preserves the seed's seven root coordinates exactly and
    excludes its six tangent DOFs, requiring a unit seed quaternion. All scalar
    joints remain available; canonical orientation tasks still constrain only
    facing/sole normals, not tangent-frame twist. This is a scratch reference
    constraint only: an admitted episode retains its physical free root.
    """
    return _solve(model, scene, profile, initial_qpos, contact_intent,
                  max_iterations, tolerance, None, freeze_root)


def generate_contact_reference(
    model: mujoco.MjModel,
    scene: BoulderScene,
    profile: ClimberProfile,
    initial_qpos: Sequence[float],
    contact_intent: Mapping[Limb, str],
    *,
    hand_targets: Mapping[Limb, Frame],
    freeze_root: bool = False,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> ContactIKResult:
    """Generate intermediate tangent-reference poses, NOT four-contact admission.

    Overrides are world position/rotation Frames for moving hands. Other hands
    retain canonical anchors; both feet retain canonical sole constraints.
    Full SO(3) is solved for overrides. The caller owns release, feedforward,
    trajectory interpolation, live execution, and endpoint solve_contact_pose.
    Signed distances remain relative to declared holds, including moving hands.

    freeze_root=True excludes the six root tangent DOFs from scratch IK and
    preserves the seed's seven root coordinates exactly (requiring a unit seed
    quaternion). Nonmoving limbs retain their seed orientations with full SO(3)
    tasks, rather than unconstrained support twist. This is a virtual reference
    constraint, not a physical weld, actuator, live-state assignment, or
    certificate that a reach is trackable.
    """
    return _solve(model, scene, profile, initial_qpos, contact_intent,
                  max_iterations, tolerance, dict(hand_targets), freeze_root)


def _solve(model, scene, profile, initial_qpos, contact_intent,
           max_iterations, tolerance, hand_targets, freeze_root=False):
    if not isinstance(freeze_root, bool):
        raise ValueError("freeze_root must be a bool")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0 or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Require nonnegative integer max_iterations and positive finite tolerance")
    seed = np.asarray(initial_qpos, dtype=float).copy()
    if seed.shape != (model.nq,) or not np.isfinite(seed).all():
        raise ValueError("Supply a complete finite model.nq seed")
    intent = dict(contact_intent)
    if set(intent) != set(Limb) or any(not isinstance(limb, Limb) for limb in intent):
        raise ValueError("Supply explicit HOLD intent for all four limbs")
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled numerics: {', '.join(issues)}")
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(roots) != 1 or any(int(kind) not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
                              for kind in model.jnt_type):
        raise ValueError("Contact IK requires one free root and scalar hinges")
    root_qpos = slice(int(model.jnt_qposadr[roots[0]]), int(model.jnt_qposadr[roots[0]]) + 7)
    quaternion = root_qpos.start + 3
    if math.hypot(*seed[quaternion:quaternion + 4]) < 1e-12:
        raise ValueError("Seed root quaternion must be nonzero")
    if freeze_root and not math.isclose(math.hypot(*seed[quaternion:quaternion + 4]),
                                        1., rel_tol=0, abs_tol=1e-8):
        raise ValueError("Frozen root requires a unit seed quaternion")
    if float(model.numeric("contact_mode").data[0]) != 0:
        raise ValueError("Contact IK requires the physical compiled model")
    configured_scene = replace(scene, start_configuration=intent)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    if not freeze_root:
        mujoco.mj_normalizeQuat(model, scratch.qpos)
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    sites, geoms, holds, geometries, targets = {}, {}, {}, {}, {}
    if hand_targets is not None:
        for limb, frame in hand_targets.items():
            if not isinstance(limb, Limb) or not limb.is_hand or not isinstance(frame, Frame):
                raise ValueError("hand_targets maps hand Limb values to world Frames")
            rotation, position = np.asarray(frame.rotation), np.asarray(frame.position)
            if (position.shape != (3,) or rotation.shape != (3, 3)
                    or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                    or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-8)):
                raise ValueError("Hand target must have a finite position and an SO(3) rotation")
    for limb in Limb:
        region = scene.region(intent[limb])
        affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
        if region.source_type != SourceType.HOLD or affordance not in region.affordances:
            raise ValueError(f"{limb.value}: intent must be an eligible HOLD")
        geometry = _canonical_surface(model, scratch, region)
        sites[limb] = model.site(END_EFFECTOR_SITES[limb]).id
        geoms[limb] = model.geom(f"{limb.value.lower()}_geom").id
        if limb.is_foot and int(model.geom_type[geoms[limb]]) != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Contact IK requires the compiled box shoe")
        holds[limb] = model.geom(f"geom_{region.id}").id
        geometries[limb] = geometry
        frame = geometry.hand_frame if limb.is_hand else geometry.foot_frame
        if limb.is_foot and geometry.shape == "box":
            rotation = np.asarray(frame.rotation)
            local = rotation.T @ (scratch.site_xpos[sites[limb]] - frame.position)
            local[:2] = np.clip(local[:2], -np.asarray(geometry.size[:2]), geometry.size[:2])
            local[2] = 0.
            frame = Frame(tuple(np.asarray(frame.position) + rotation @ local), frame.rotation)
        if hand_targets is not None and limb in hand_targets:
            supplied = hand_targets[limb]
            frame = Frame(tuple(float(v) for v in supplied.position),
                          tuple(tuple(float(v) for v in row) for row in supplied.rotation))
        elif freeze_root and hand_targets is not None:
            rotation = (scratch.site_xmat[sites[limb]] if limb.is_hand else
                        scratch.geom_xmat[geoms[limb]]).reshape(3, 3)
            frame = Frame(frame.position, tuple(tuple(float(v) for v in row) for row in rotation))
        targets[limb] = frame
    allowed = {frozenset((geoms[limb], holds[limb])) for limb in Limb
               if hand_targets is None or limb not in hand_targets}
    limited = np.flatnonzero(model.jnt_limited)
    qadr, dofs = model.jnt_qposadr[limited], model.jnt_dofadr[limited]
    bounds = model.jnt_range[limited]
    root_dof = int(model.jnt_dofadr[roots[0]])

    def admit(qpos):
        try:
            reference, _ = initialize_static_reference(
                model, mujoco.MjData(model), configured_scene, profile, qpos, intent)
            return reference, "Exact Stage3 scratch admission passed"
        except (ValueError, KeyError) as error:
            return None, f"{type(error).__name__}: {error}"

    def measure(with_jacobian=False):
        errors, jacobians, evidence, collisions = [], [], [], []
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for limb in Limb:
            site, frame = sites[limb], targets[limb]
            rotation = (scratch.site_xmat[site] if limb.is_hand else
                        scratch.geom_xmat[geoms[limb]]).reshape(3, 3)
            target_rotation = np.asarray(frame.rotation)
            normal_only = hand_targets is None or (not freeze_root and limb not in hand_targets)
            error = _rotation_error(rotation, target_rotation, normal_only)
            position = np.asarray(frame.position) - scratch.site_xpos[site]
            errors.extend((position, .12 * error))
            if with_jacobian:
                mujoco.mj_jacSite(model, scratch, jp, jr, site)
                angular = (np.eye(3) - np.outer(rotation[:, 2], rotation[:, 2])) @ jr if normal_only else jr
                jacobians.extend((jp.copy(), .12 * angular))
            geometry = geometries[limb]
            canonical = geometry.hand_frame if limb.is_hand else geometry.foot_frame
            distance = float(mujoco.mj_geomDistance(model, scratch, geoms[limb], holds[limb], math.inf, None))
            foot, foot_reason = None, ""
            normal_residual, minimum_sole_distance, tangential = None, None, None
            if limb.is_foot:
                face = geometry.foot_surface_frame
                face_rotation = np.asarray(face.rotation)
                local_site = face_rotation.T @ (scratch.site_xpos[site] - face.position)
                normal_residual = float(local_site[2] - FOOT_SITE_OFFSET)
                tangential = tuple(float(v) for v in local_site[:2])
                size = model.geom_size[geoms[limb]]
                sole = scratch.geom_xpos[geoms[limb]] - size[2] * rotation[:, 2]
                corners = np.array([sole + x * size[0] * rotation[:, 0] + y * size[1] * rotation[:, 1]
                                    for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                minimum_sole_distance = float(np.min((corners - face.position) @ face_rotation[:, 2]))
                try:
                    foot = _foot_residual(model, scratch, limb, scene.region(intent[limb]), geometry)
                except ValueError as failure:
                    foot_reason = str(failure)
            evidence.append(ContactIKResidual(
                limb, intent[limb], frame, float(np.linalg.norm(position)),
                float(np.linalg.norm(np.asarray(canonical.position) - scratch.site_xpos[site])),
                float(np.linalg.norm(error)), float(rotation[:, 2] @ target_rotation[:, 2]),
                distance, foot, foot_reason, normal_residual, minimum_sole_distance, tangential))
        # Native contact enumeration honors masks, explicit pairs, welded/parent
        # filtering and exclusions. Never treat visual-only geoms as obstacles.
        for contact in scratch.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if frozenset((g1, g2)) in allowed:
                continue
            distance = float(contact.dist)
            if distance >= .0002:
                continue
            collisions.append(CollisionResidual(
                (model.geom(g1).name or f"geom#{g1}", model.geom(g2).name or f"geom#{g2}"), distance))
            errors.append(np.array([2. * (.0002 - distance)]))
            if with_jacobian:
                j1, j2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, scratch, j1, None, contact.pos, int(model.geom_bodyid[g1]))
                mujoco.mj_jac(model, scratch, j2, None, contact.pos, int(model.geom_bodyid[g2]))
                jacobians.append(2. * (contact.frame[:3] @ (j2 - j1))[None, :])
        error = np.concatenate(errors)
        return error, np.vstack(jacobians) if with_jacobian else None, tuple(evidence), tuple(collisions)

    def result(converged, reference, iterations, reason):
        _, _, residuals, collisions = measure()
        return ContactIKResult(tuple(float(v) for v in scratch.qpos), converged,
                               reference is not None, iterations, residuals, reason,
                               reference, initial_residuals, initial_reason, collisions)

    _, _, initial_residuals, _ = measure()
    original_reference, initial_reason = (admit(seed) if hand_targets is None else
                                          (None, "Reference generation: four-contact admission not requested"))
    if hand_targets is None and original_reference is not None:
        scratch.qpos[:] = seed
        mujoco.mj_forward(model, scratch)
        return result(True, original_reference, 0, "Already admitted; seed preserved exactly")
    # Project scalar ROM through tangent coordinates, never by freezing illegal
    # seed angles or treating root quaternion components as optimization DOFs.
    correction = np.zeros(model.nv)
    correction[dofs] = np.clip(scratch.qpos[qadr], bounds[:, 0], bounds[:, 1]) - scratch.qpos[qadr]
    if max_iterations and np.any(correction):
        mujoco.mj_integratePos(model, scratch.qpos, correction, 1.)
        if freeze_root:
            scratch.qpos[root_qpos] = seed[root_qpos]
        mujoco.mj_forward(model, scratch)
    iterations, damping = 0, .001
    stopped = "Iteration budget exhausted"
    for _ in range(max_iterations + 1):
        error, jacobian, residuals, collisions = measure(True)
        geometry_converged = (all(r.position_error <= min(tolerance, .001)
                                  and r.orientation_error <= min(tolerance, 1e-5) for r in residuals)
                              and not collisions)
        if geometry_converged:
            try:
                validate_reference_pose(model, scratch.qpos)
            except ValueError as failure:
                return result(False, None, iterations, str(failure))
            if hand_targets is not None:
                return result(True, None, iterations, "Geometric reference only; no four-contact admission attempted")
            reference, reason = admit(scratch.qpos)
            if reference is not None:
                return result(True, reference, iterations, reason)
            # Especially with a loose caller tolerance, sole touch may require
            # another ordinary IK step. Geometry alone is not an admission gate.
        if iterations >= max_iterations:
            break
        current = scratch.qpos.copy()
        score = float(error @ error)
        accepted = False
        for _ in range(5):
            active = np.ones(model.nv, dtype=bool)
            if freeze_root:
                active[root_dof:root_dof + 6] = False
            step = np.zeros(model.nv)
            for _ in range(len(limited) + 1):
                j = jacobian[:, active]
                step[:] = 0.
                step[active] = np.linalg.solve(j.T @ j + damping ** 2 * np.eye(j.shape[1]), j.T @ error)
                outward = (((current[qadr] <= bounds[:, 0] + 1e-10) & (step[dofs] < 0))
                           | ((current[qadr] >= bounds[:, 1] - 1e-10) & (step[dofs] > 0)))
                blocked = dofs[outward & active[dofs]]
                if not len(blocked):
                    break
                active[blocked] = False
            scale = min(1., .15 / max(float(np.max(np.abs(step))), 1e-12),
                        .03 / max(float(np.linalg.norm(step[root_dof:root_dof + 3])), 1e-12))
            step *= scale
            for backtrack in range(12):
                candidate = current.copy()
                mujoco.mj_integratePos(model, candidate, step, .5 ** backtrack)
                correction[:] = 0.
                correction[dofs] = np.clip(candidate[qadr], bounds[:, 0], bounds[:, 1]) - candidate[qadr]
                mujoco.mj_integratePos(model, candidate, correction, 1.)
                if freeze_root:
                    # Zero root tangents can still renormalize a quaternion in
                    # mj_integratePos. Preserve the virtual root on scratch only.
                    candidate[root_qpos] = seed[root_qpos]
                else:
                    mujoco.mj_normalizeQuat(model, candidate)
                scratch.qpos[:] = candidate
                mujoco.mj_forward(model, scratch)
                trial, _, _, _ = measure()
                if np.isfinite(trial).all() and float(trial @ trial) < score - 1e-20:
                    accepted = True
                    break
            if accepted:
                iterations += 1
                damping = max(.00001, damping * .5)
                break
            damping *= 10.
        if not accepted:
            scratch.qpos[:] = current
            mujoco.mj_forward(model, scratch)
            stopped = "Bounded local search stalled"
            break
    reference, reason = admit(scratch.qpos) if hand_targets is None else (None, "No four-contact admission attempted")
    if reference is not None:
        return result(False, reference, iterations,
                      f"{stopped}; {reason}; IK target tolerance not reached")
    return result(geometry_converged, reference, iterations,
                  f"{stopped}; {reason}. Local failure is not a global infeasibility proof")
