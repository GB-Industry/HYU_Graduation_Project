"""Stage5.1 scratch whole-body references, never dynamics or contact admission."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import mujoco
import numpy as np

from .contact_geometry import FOOT_SITE_OFFSET, Frame
from .contact_ik import CollisionResidual, ContactIKResidual, _rotation_error
from .mjcf_builder import END_EFFECTOR_SITES
from .runtime import compiled_numerical_issues, validate_reference_pose
from .schema import Affordance, BoulderScene, ClimberProfile, Limb, SourceType
from .static_state import _canonical_surface, _foot_residual, _no_foot_equalities


@dataclass(frozen=True)
class WholeBodyReferenceResult:
    """Immutable geometric evidence; failed qpos is diagnostic only.

    Root errors at the endpoint describe the virtual reference, NOT a measured
    physical root. Initial errors and source_frames describe the original input
    pose before any root override. Position errors are metres, angular/waist
    errors radians. Iterations count accepted bounded tangent steps. target_pose
    contains only actuated hinges and is the motor-reference interface; root
    qpos/metadata must never be assigned to live data by a controller.

    Even convergence certifies no reaction capacity, penetration of allowed
    touch pairs, equilibrium, readiness, or dynamically trackable trajectory.
    The caller owns unchanged physical guards and exact endpoint admission.
    """

    qpos: tuple[float, ...]
    converged: bool
    iterations: int
    residuals: tuple[ContactIKResidual, ...]
    initial_residuals: tuple[ContactIKResidual, ...]
    collisions: tuple[CollisionResidual, ...]
    root_target: Frame
    initial_root: Frame
    root_position_error: float
    root_orientation_error: float
    initial_root_position_error: float
    initial_root_orientation_error: float
    waist_target: Mapping[str, float]
    waist_errors: Mapping[str, float]
    initial_waist_errors: Mapping[str, float]
    source_frames: Mapping[Limb, Frame]
    target_pose: Mapping[str, float]
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "qpos", tuple(float(v) for v in self.qpos))
        for name in ("residuals", "initial_residuals", "collisions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("waist_target", "waist_errors", "initial_waist_errors", "source_frames", "target_pose"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _validated_frame(frame: Frame, label: str) -> Frame:
    try:
        if not isinstance(frame, Frame):
            raise ValueError
        position = np.asarray(frame.position, dtype=float)
        rotation = np.asarray(frame.rotation, dtype=float)
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-8)):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a finite world Frame with an SO(3) rotation") from None
    return Frame(tuple(float(v) for v in position), tuple(tuple(float(v) for v in row) for row in rotation))


def solve_whole_body_reference(
    model: mujoco.MjModel,
    scene: BoulderScene,
    profile: ClimberProfile,
    initial_qpos: Sequence[float],
    contact_intent: Mapping[Limb, str],
    *,
    root_target: Frame,
    waist_target: Mapping[str, float],
    hand_targets: Mapping[Limb, Frame] | None = None,
    support_frames: Mapping[Limb, Frame] | None = None,
    allowed_hand_holds: Mapping[Limb, tuple[str, ...]] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> WholeBodyReferenceResult:
    """Solve all motor hinges at an exactly prescribed virtual free-root frame.

    Each hand has five tasks: END-site position and world outward +Z normal;
    tangent twist is free, matching a point grasp without a fictitious moment.
    Each foot has six: END-site position and actual box shoe full SO(3), NOT
    the site's local rotation. Waist angles add three .12-weighted tasks.
    A physical 25-hinge rig thus has 25 tasks, without root tangent columns.

    Default held hands use canonical anchors. Default feet use canonical sole
    frames (already including the 11mm site offset), retaining the original
    source's box-face tangential offset. Explicit support_frames are fixed world
    targets, useful across repeated solves as the desired root moves; moving
    hand_targets take precedence. Source FK is captured BEFORE overriding root.
    No source admission is required, so the caller can prepare a legal crouch.

    Native contacts below 0.2mm clearance add weight-2 collision penalties,
    honoring masks, exclusions and explicit pairs. Intended touch pairs and
    declared additional hand HOLDs are ignored for IK collision steering only;
    intended HOLD distances remain evidence, not admission. No model, caller
    pose, live state, actuator gains, equality or physical force is modified.
    Malformed inputs, nonunit seed quaternions, ANY seed hinge outside compiled
    ROM and waist goals outside ROM raise ValueError before FK. Trial motor
    tangents have norm <=0.1rad, with backtracking and scalar ROM projection.
    Local failure is not a global infeasibility proof.
    """
    try:
        valid_tolerance = not isinstance(tolerance, bool) and math.isfinite(tolerance) and tolerance > 0
    except (TypeError, ValueError, OverflowError):
        valid_tolerance = False
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0 or not valid_tolerance):
        raise ValueError("Require nonnegative integer max_iterations and positive finite tolerance")
    if not isinstance(scene, BoulderScene) or not isinstance(profile, ClimberProfile):
        raise ValueError("Supply a BoulderScene and ClimberProfile")
    try:
        seed = np.asarray(initial_qpos, dtype=float).copy()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Supply a complete finite model.nq source pose") from None
    if seed.shape != (model.nq,) or not np.isfinite(seed).all():
        raise ValueError("Supply a complete finite model.nq source pose")
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled numerics: {', '.join(issues)}")
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (len(roots) != 1 or any(int(kind) not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
                              for kind in model.jnt_type)):
        raise ValueError("Whole-body references require one free root and scalar hinges")
    validate_reference_pose(model, seed)
    try:
        physical = model.numeric("contact_mode").data
    except KeyError:
        raise ValueError("Whole-body references require the physical compiled contact mode") from None
    if len(physical) != 1 or physical[0] != 0:
        raise ValueError("Whole-body references require the physical compiled contact mode")
    root_target = _validated_frame(root_target, "root_target")
    if (not isinstance(contact_intent, Mapping) or set(contact_intent) != set(Limb)
            or any(not isinstance(limb, Limb) for limb in contact_intent)):
        raise ValueError("Supply explicit HOLD intent for all four limbs")
    intent = dict(contact_intent)
    supplied = []
    for mapping, label, hands_only in ((hand_targets, "hand_targets", True),
                                       (support_frames, "support_frames", False)):
        if mapping is not None and not isinstance(mapping, Mapping):
            raise ValueError(f"{label} must map Limb values to world Frames")
        frames = {}
        for limb, frame in (mapping.items() if mapping is not None else ()):
            if not isinstance(limb, Limb) or (hands_only and not limb.is_hand):
                raise ValueError(f"{label} must map {'hand ' if hands_only else ''}Limb values to world Frames")
            frames[limb] = _validated_frame(frame, label)
        supplied.append(frames)
    hand_targets, support_frames = supplied
    if allowed_hand_holds is not None and not isinstance(allowed_hand_holds, Mapping):
        raise ValueError("allowed_hand_holds must map hand Limb values to tuples of HOLD IDs")
    additional = dict(allowed_hand_holds) if allowed_hand_holds is not None else {}
    for limb, ids in additional.items():
        if (not isinstance(limb, Limb) or not limb.is_hand or not isinstance(ids, tuple)
                or any(not isinstance(rid, str) or not rid.strip() for rid in ids)):
            raise ValueError("allowed_hand_holds must map hand Limb values to tuples of HOLD IDs")

    joints = []
    for motor in range(model.nu):
        jid = int(model.actuator_trnid[motor, 0])
        if (not 0 <= jid < model.njnt or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                or model.actuator_trntype[motor] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_dyntype[motor] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[motor] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[motor] != mujoco.mjtBias.mjBIAS_NONE
                or model.actuator_gainprm[motor, 0] != 1.
                or not model.actuator_ctrllimited[motor] or model.actuator_forcelimited[motor]
                or not model.actuator_ctrlrange[motor, 0] < 0 < model.actuator_ctrlrange[motor, 1]
                or model.actuator_ctrlrange[motor, 0] != -model.actuator_ctrlrange[motor, 1]
                or model.actuator_gear[motor, 0] <= 0 or np.any(model.actuator_gear[motor, 1:])):
            raise ValueError("Whole-body references require finite direct scalar hinge motor capabilities")
        capability = float(model.actuator_gear[motor, 0]) * float(model.actuator_ctrlrange[motor, 1])
        if not math.isfinite(capability) or capability <= 0:
            raise ValueError("Whole-body references require finite positive motor capabilities")
        if jid not in joints:
            joints.append(jid)
    joints = np.array(joints, dtype=int)
    qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limited = model.jnt_limited[joints].astype(bool)
    lower = np.where(limited, model.jnt_range[joints, 0], -np.inf)
    upper = np.where(limited, model.jnt_range[joints, 1], np.inf)
    waist_names = ("waist_yaw", "waist_pitch", "waist_roll")
    if not isinstance(waist_target, Mapping) or set(waist_target) != set(waist_names):
        raise ValueError("Supply all three waist_yaw/pitch/roll angles")
    waist, waist_columns, waist_qadr = {}, [], []
    for name in waist_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        try:
            angle = float(waist_target[name])
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Waist targets must be finite angles within compiled ROM") from None
        if isinstance(waist_target[name], (bool, str)) or not math.isfinite(angle) or jid not in joints:
            raise ValueError("Waist targets must identify actuated hinges with finite angles")
        column = int(np.flatnonzero(joints == jid)[0])
        if not lower[column] <= angle <= upper[column]:
            raise ValueError(f"Waist target {name!r} violates compiled ROM")
        waist[name] = angle
        waist_columns.append(column)
        waist_qadr.append(int(model.jnt_qposadr[jid]))

    sites, geoms, regions = {}, {}, {}
    for limb in Limb:
        rid = intent[limb]
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"{limb.value}: intent must name an eligible HOLD")
        affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
        try:
            region = scene.region(rid)
            sites[limb] = model.site(END_EFFECTOR_SITES[limb]).id
            geoms[limb] = model.geom(f"{limb.value.lower()}_geom").id
            model.geom(f"geom_{rid}")
            model.site(f"site_{rid}")
            if Affordance.STEP in region.affordances:
                model.site(f"site_step_{rid}")
        except KeyError:
            raise ValueError(f"{limb.value}: missing declared HOLD or compiled contact frames") from None
        if region.source_type != SourceType.HOLD or affordance not in region.affordances:
            raise ValueError(f"{limb.value}: intent must be an eligible HOLD")
        if model.site_bodyid[sites[limb]] != model.geom_bodyid[geoms[limb]]:
            raise ValueError(f"{limb.value}: END site must be rigid on its contact geom body")
        if limb.is_foot:
            geom, site = geoms[limb], sites[limb]
            local_rotation = np.empty(9)
            mujoco.mju_quat2Mat(local_rotation, model.geom_quat[geom])
            sole_offset = ((model.site_pos[site] - model.geom_pos[geom]) @ local_rotation.reshape(3, 3)[:, 2]
                           + model.geom_size[geom, 2])
            if (model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX
                    or not math.isclose(float(sole_offset), FOOT_SITE_OFFSET, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("Whole-body references require a compiled box shoe with the 11mm END-site sole offset")
        regions[limb] = region
    _no_foot_equalities(model)
    extra_regions = {}
    for limb, ids in additional.items():
        for rid in ids:
            try:
                region = scene.region(rid)
                model.geom(f"geom_{rid}")
                model.site(f"site_{rid}")
            except KeyError:
                raise ValueError(f"{rid}: missing declared allowed hand HOLD") from None
            if region.source_type != SourceType.HOLD or Affordance.GRASP not in region.affordances:
                raise ValueError(f"{rid}: allowed hand touches require an eligible GRASP HOLD")
            extra_regions[rid] = region

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    root_joint = int(roots[0])
    root_body = int(model.jnt_bodyid[root_joint])
    root_qa = int(model.jnt_qposadr[root_joint])
    root_slice = slice(root_qa, root_qa + 7)
    initial_root = Frame(tuple(float(v) for v in scratch.xpos[root_body]),
                         tuple(map(tuple, scratch.xmat[root_body].reshape(3, 3))))
    source_frames, targets, geometries, holds = {}, {}, {}, {}
    for limb in Limb:
        rotation = (scratch.site_xmat[sites[limb]] if limb.is_hand else scratch.geom_xmat[geoms[limb]]).reshape(3, 3)
        source_frames[limb] = Frame(tuple(float(v) for v in scratch.site_xpos[sites[limb]]), tuple(map(tuple, rotation)))
        geometry = _canonical_surface(model, scratch, regions[limb])
        geometries[limb] = geometry
        holds[limb] = model.geom(f"geom_{intent[limb]}").id
        frame = geometry.hand_frame if limb.is_hand else geometry.foot_frame
        if limb.is_foot and geometry.shape == "box":
            rotation = np.asarray(frame.rotation)
            offset = rotation.T @ (scratch.site_xpos[sites[limb]] - frame.position)
            offset[:2] = np.clip(offset[:2], -np.asarray(geometry.size[:2]), geometry.size[:2])
            offset[2] = 0.
            frame = Frame(tuple(float(v) for v in np.asarray(frame.position) + rotation @ offset), frame.rotation)
        targets[limb] = hand_targets.get(limb, support_frames.get(limb, frame))
    for region in extra_regions.values():
        _canonical_surface(model, scratch, region)
    allowed = {frozenset((geoms[limb], holds[limb])) for limb in Limb}
    allowed.update(frozenset((geoms[limb], model.geom(f"geom_{rid}").id))
                   for limb, ids in additional.items() for rid in ids)
    virtual_root = np.array((*root_target.position, *root_target.quaternion))
    frozen = np.ones(model.nq, dtype=bool)
    frozen[qadr] = False
    waist_jacobian = .12 * np.eye(len(joints))[waist_columns]

    def forward():
        scratch.qpos[root_slice] = virtual_root
        mujoco.mj_forward(model, scratch)

    def measure(with_jacobian=False):
        errors, jacobians, residuals, collisions = [], [], [], []
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for limb in Limb:
            site, geom, target = sites[limb], geoms[limb], targets[limb]
            rotation = (scratch.site_xmat[site] if limb.is_hand else scratch.geom_xmat[geom]).reshape(3, 3)
            goal_rotation = np.asarray(target.rotation)
            ep = np.asarray(target.position) - scratch.site_xpos[site]
            er = _rotation_error(rotation, goal_rotation, limb.is_hand)
            angular_basis = rotation[:, :2].T if limb.is_hand else np.eye(3)
            errors.extend((ep, .12 * (angular_basis @ er)))
            if with_jacobian:
                mujoco.mj_jacSite(model, scratch, jp, jr, site)
                jacobians.extend((jp[:, dofs].copy(), .12 * (angular_basis @ jr[:, dofs])))
            geometry = geometries[limb]
            canonical = geometry.hand_frame if limb.is_hand else geometry.foot_frame
            distance = float(mujoco.mj_geomDistance(model, scratch, geom, holds[limb], math.inf, None))
            foot, foot_reason, normal_residual, minimum_sole, tangential = None, "", None, None, None
            if limb.is_foot:
                face = geometry.foot_surface_frame
                face_rotation = np.asarray(face.rotation)
                local = face_rotation.T @ (scratch.site_xpos[site] - face.position)
                normal_residual, tangential = float(local[2] - FOOT_SITE_OFFSET), tuple(float(v) for v in local[:2])
                size = model.geom_size[geom]
                sole = scratch.geom_xpos[geom] - size[2] * rotation[:, 2]
                corners = np.array([sole + x * size[0] * rotation[:, 0] + y * size[1] * rotation[:, 1]
                                    for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                minimum_sole = float(np.min((corners - face.position) @ face_rotation[:, 2]))
                try:
                    foot = _foot_residual(model, scratch, limb, regions[limb], geometry)
                except ValueError as failure:
                    foot_reason = str(failure)
            residuals.append(ContactIKResidual(
                limb, intent[limb], target, float(np.linalg.norm(ep)),
                float(np.linalg.norm(np.asarray(canonical.position) - scratch.site_xpos[site])),
                float(np.linalg.norm(er)), float(np.clip(rotation[:, 2] @ goal_rotation[:, 2], -1., 1.)),
                distance, foot, foot_reason, normal_residual, minimum_sole, tangential))
        waist_error = np.array([waist[name] - scratch.qpos[qa] for name, qa in zip(waist_names, waist_qadr)])
        errors.append(.12 * waist_error)
        if with_jacobian:
            jacobians.append(waist_jacobian)
        for contact in scratch.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if frozenset((g1, g2)) in allowed or contact.dist >= .0002:
                continue
            distance = float(contact.dist)
            collisions.append(CollisionResidual(
                (model.geom(g1).name or f"geom#{g1}", model.geom(g2).name or f"geom#{g2}"), distance))
            errors.append(np.array([2. * (.0002 - distance)]))
            if with_jacobian:
                j1, j2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, scratch, j1, None, contact.pos, int(model.geom_bodyid[g1]))
                mujoco.mj_jac(model, scratch, j2, None, contact.pos, int(model.geom_bodyid[g2]))
                jacobians.append(2. * (contact.frame[:3] @ (j2 - j1)[:, dofs])[None, :])
        return (np.concatenate(errors), np.vstack(jacobians) if with_jacobian else None,
                tuple(residuals), tuple(collisions), dict(zip(waist_names, map(float, np.abs(waist_error)))))

    _, _, initial_residuals, _, initial_waist_errors = measure()
    initial_root_position_error = float(np.linalg.norm(np.asarray(root_target.position) - initial_root.position))
    initial_root_orientation_error = float(np.linalg.norm(_rotation_error(
        np.asarray(initial_root.rotation), np.asarray(root_target.rotation), False)))
    forward()
    iterations, damping, reason = 0, .001, "Iteration budget exhausted"
    converged = False
    for _ in range(max_iterations + 1):
        error, jacobian, residuals, collisions, waist_errors = measure(True)
        converged = (all(r.position_error <= tolerance and r.orientation_error <= tolerance for r in residuals)
                     and max(waist_errors.values()) <= tolerance and not collisions)
        if converged or iterations >= max_iterations:
            break
        current, score, accepted = scratch.qpos.copy(), float(error @ error), False
        for _ in range(5):
            active, step = np.ones(len(joints), dtype=bool), np.zeros(len(joints))
            for _ in range(len(joints) + 1):
                j = jacobian[:, active]
                step[:] = 0.
                step[active] = np.linalg.solve(j.T @ j + damping ** 2 * np.eye(j.shape[1]), j.T @ error)
                outward = (((current[qadr] <= lower + 1e-10) & (step < 0))
                           | ((current[qadr] >= upper - 1e-10) & (step > 0)))
                if not np.any(outward & active):
                    break
                active[outward] = False
            step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-12),
                        .15 / max(float(np.max(np.abs(step))), 1e-12))
            tangent = np.zeros(model.nv)
            tangent[dofs] = step
            for backtrack in range(12):
                candidate = current.copy()
                mujoco.mj_integratePos(model, candidate, tangent, .5 ** backtrack)
                candidate[qadr] = np.clip(candidate[qadr], lower, upper)
                candidate[frozen] = seed[frozen]
                scratch.qpos[:] = candidate
                forward()
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
            forward()
            reason = "Bounded whole-body local search stalled"
            break
    _, _, residuals, collisions, waist_errors = measure()
    validate_reference_pose(model, scratch.qpos)
    root_position_error = float(np.linalg.norm(scratch.xpos[root_body] - root_target.position))
    root_orientation_error = float(np.linalg.norm(_rotation_error(
        scratch.xmat[root_body].reshape(3, 3), np.asarray(root_target.rotation), False)))
    return WholeBodyReferenceResult(
        tuple(float(v) for v in scratch.qpos), converged, iterations, residuals, initial_residuals, collisions,
        root_target, initial_root, root_position_error, root_orientation_error,
        initial_root_position_error, initial_root_orientation_error, waist, waist_errors, initial_waist_errors,
        source_frames, {model.joint(int(jid)).name: float(scratch.qpos[qa]) for jid, qa in zip(joints, qadr)},
        ("Whole-body geometric reference; no contact admission attempted" if converged else
         f"{reason}; no contact admission attempted. Local failure is not a global infeasibility proof"))
