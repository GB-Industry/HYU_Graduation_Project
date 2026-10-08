from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .contact_geometry import canonical_geometry
from .mjcf_builder import END_EFFECTOR_SITES
from .runtime import _import_mujoco
from .schema import BoulderScene, ClimberProfile, Limb, Vec3

ACTUATED_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw", "waist_pitch", "waist_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
)


@dataclass(frozen=True)
class StanceSpecification:
    """High-level structured specification of climbing posture intent.

    Replaces brittle flat hard-coded joint arrays with structured posture semantics:
      - target holds for hands and footholds for feet,
      - pelvis distance to the wall and lateral/height biases,
      - preferred torso orientation,
      - preferred elbow and knee flexion,
      - preferred hip openness / abduction,
      - optional free limbs (limbs released from hold constraints).
    """

    hand_holds: dict[Limb, str] = field(default_factory=lambda: {
        Limb.LEFT_HAND: "H3",
        Limb.RIGHT_HAND: "H4",
    })
    foot_holds: dict[Limb, str] = field(default_factory=lambda: {
        Limb.LEFT_FOOT: "H1",
        Limb.RIGHT_FOOT: "H2",
    })
    pelvis_wall_distance: float = 0.64
    pelvis_lateral_bias: float = 0.0
    pelvis_height_bias: float = 0.0
    torso_orientation: tuple[float, float, float] = (0.0, -0.24, 0.0)  # yaw, pitch, roll
    preferred_elbow_flexion: float = 1.00
    preferred_knee_flexion: float = 1.35
    hip_openness: float = 0.28
    free_limbs: tuple[Limb, ...] = ()
    custom_limb_targets: dict[Limb, Vec3] = field(default_factory=dict)


@dataclass(frozen=True)
class RetargetResult:
    """Result of deterministic stance retargeting."""

    qpos: tuple[float, ...]
    target_pose: dict[str, float]
    errors: dict[Limb, float]
    converged: bool
    iterations: int
    pelvis_pos: tuple[float, float, float]


def solve_retargeted_stance(
    model: Any,
    scene: BoulderScene,
    profile: ClimberProfile | None = None,
    spec: StanceSpecification | None = None,
    max_iterations: int = 80,
    tolerance: float = 1e-3,
) -> RetargetResult:
    """Solve for a deterministic retargeted humanoid posture using Damped Least Squares (DLS) IK.

    Formulates a multi-objective task Jacobian:
      1. Active end-effector sites reach target hold positions (hands) and wedged foothold positions (feet).
      2. Pelvis root coordinates match wall distance, lateral bias, and height.
      3. Nullspace projection regularizes joint redundancy toward biomechanically plausible preferred angles.
      4. Joint limits are respected at every iteration.
    """
    mujoco = _import_mujoco()
    s = spec if spec is not None else StanceSpecification()

    # Determine base pelvis height and wall distance adapted to profile morphology
    delta_leg = (profile.leg_reach - 0.82) if profile is not None else 0.0
    delta_arm = (profile.arm_reach - 0.58) if profile is not None else 0.0

    pelvis_z = 1.263 + delta_leg * 0.85 + s.pelvis_height_bias
    pelvis_y = -s.pelvis_wall_distance - delta_arm * 0.70
    pelvis_x = -0.0134 + s.pelvis_lateral_bias
    target_pelvis = np.array([pelvis_x, pelvis_y, pelvis_z])

    # End-effector targets in world coordinates
    limb_targets: dict[Limb, np.ndarray] = {}
    for limb, hold_id in s.hand_holds.items():
        if limb in s.free_limbs and limb not in s.custom_limb_targets:
            continue
        if limb in s.custom_limb_targets:
            limb_targets[limb] = np.array(s.custom_limb_targets[limb])
        else:
            region = scene.region(hold_id)
            limb_targets[limb] = np.array(canonical_geometry(region).hand_frame.position)

    for limb, hold_id in s.foot_holds.items():
        if limb in s.free_limbs and limb not in s.custom_limb_targets:
            continue
        if limb in s.custom_limb_targets:
            limb_targets[limb] = np.array(s.custom_limb_targets[limb])
        else:
            region = scene.region(hold_id)
            limb_targets[limb] = np.array(canonical_geometry(region).foot_frame.position)

    for limb, tgt in s.custom_limb_targets.items():
        if limb not in limb_targets:
            limb_targets[limb] = np.array(tgt)

    # Initialize MjData
    d = mujoco.MjData(model)
    d.qpos[0:3] = target_pelvis
    d.qpos[3:7] = [0.9968, -0.0794, -0.0096, -0.0006]

    # Posture priors based on high-level intent & biomechanical realism
    y_bias = s.torso_orientation[0]
    p_bias = s.torso_orientation[1]
    r_bias = s.torso_orientation[2]
    ef = s.preferred_elbow_flexion
    kf = s.preferred_knee_flexion
    ho = s.hip_openness

    # Natural hip pitch & turnout conditioned on wall proximity:
    # Closer wall distance requires lower hip pitch (thighs angled down) and higher turnout
    hip_pitch_prior = min(1.68, max(0.85, 0.85 + (s.pelvis_wall_distance - 0.40) * 3.2))
    hip_roll_prior = max(ho, 0.42 - (s.pelvis_wall_distance - 0.40) * 0.4)

    defaults: dict[str, float] = {
        "waist_yaw": y_bias,
        "waist_pitch": p_bias,
        "waist_roll": r_bias,
        "left_shoulder_pitch": 1.0590,
        "left_shoulder_roll": 0.3250,
        "left_shoulder_yaw": 0.0027,
        "left_elbow": ef,
        "left_wrist": -0.0031,
        "right_shoulder_pitch": 1.1033,
        "right_shoulder_roll": -0.3555,
        "right_shoulder_yaw": 0.0027,
        "right_elbow": ef,
        "right_wrist": -0.0015,
        "left_hip_pitch": hip_pitch_prior,
        "left_hip_roll": hip_roll_prior,
        "left_hip_yaw": -0.0021,
        "left_knee": kf,
        "left_ankle_pitch": -0.28,
        "left_ankle_roll": -0.0001,
        "right_hip_pitch": hip_pitch_prior,
        "right_hip_roll": -hip_roll_prior,
        "right_hip_yaw": -0.0016,
        "right_knee": kf,
        "right_ankle_pitch": -0.28,
        "right_ankle_roll": -0.0001,
    }
    for name, qval in defaults.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[model.jnt_qposadr[jid]] = qval

    mujoco.mj_forward(model, d)
    q_pref = d.qpos[7:32].copy()

    site_ids = {
        limb: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITES[limb])
        for limb in END_EFFECTOR_SITES
    }

    knee_bodies = {
        "left_knee": (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shin"), "left_hip_roll", 1.0),
        "right_knee": (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shin"), "right_hip_roll", -1.0),
    }

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    converged = False
    iterations_run = 0

    for it in range(max_iterations):
        iterations_run = it + 1
        mujoco.mj_forward(model, d)

        errs: list[np.ndarray] = []
        jacs: list[np.ndarray] = []

        # 1. End-effector position constraints
        for limb, target in limb_targets.items():
            sid = site_ids[limb]
            pos = d.site_xpos[sid]
            errs.append(target - pos)
            mujoco.mj_jacSite(model, d, jacp, jacr, sid)
            jacs.append(jacp.copy())

        # 2. Pelvis root coordinates constraints
        errs.append(np.array([target_pelvis[0] - d.qpos[0]]))
        j_px = np.zeros((1, model.nv)); j_px[0, 0] = 1.0; jacs.append(j_px)
        errs.append(np.array([target_pelvis[1] - d.qpos[1]]))
        j_py = np.zeros((1, model.nv)); j_py[0, 1] = 1.0; jacs.append(j_py)
        errs.append(np.array([target_pelvis[2] - d.qpos[2]]))
        j_pz = np.zeros((1, model.nv)); j_pz[0, 2] = 1.0; jacs.append(j_pz)

        err = np.concatenate(errs)
        J = np.vstack(jacs)

        num_effector_constraints = len(limb_targets) * 3
        if np.max(np.abs(err[:num_effector_constraints])) < tolerance:
            converged = True
            break

        # Task weighting: end-effectors have priority 1.0, pelvis position has balanced priority 0.35
        m = J.shape[0]
        W = np.ones(m)
        W[num_effector_constraints:] = 0.35

        # Damped Least-Squares solve: J^T (J J^T + lambda^2 I)^-1 err
        lambda_sq = 1e-4
        JJT = J @ J.T + lambda_sq * np.eye(m)
        dq_task = J.T @ np.linalg.solve(JJT, err * W)

        # Nullspace projection towards biomechanically preferred angles
        dq_pref = np.zeros(model.nv)
        dq_pref[6:31] = q_pref - d.qpos[7:32]

        # Bend-plane & pole-vector preference: penalize knee/elbow extension near singular limits
        for kname in ("left_knee", "right_knee"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, kname)
            if jid >= 0:
                qadr = model.jnt_qposadr[jid]
                dof_adr = model.jnt_dofadr[jid]
                if d.qpos[qadr] < 0.60:
                    dq_pref[dof_adr] += (1.00 - d.qpos[qadr]) * 1.5

        # Knee wall-clearance preference: prevent knees pressing flat into wall
        for kname, (bid, roll_name, sgn) in knee_bodies.items():
            if bid >= 0:
                knee_y = d.xpos[bid][1]
                if knee_y > -0.08:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, roll_name)
                    if jid >= 0:
                        dof = model.jnt_dofadr[jid]
                        dq_pref[dof] += sgn * (knee_y - (-0.08)) * 4.0

        for ename in ("left_elbow", "right_elbow"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ename)
            if jid >= 0:
                qadr = model.jnt_qposadr[jid]
                dof_adr = model.jnt_dofadr[jid]
                if d.qpos[qadr] < 0.40:
                    dq_pref[dof_adr] += (0.80 - d.qpos[qadr]) * 1.5

        dq_null = dq_pref - J.T @ np.linalg.solve(JJT, J @ dq_pref)

        # Update step
        dq = (dq_task + 0.4 * dq_null) * 0.5
        qpos_new = d.qpos.copy()
        mujoco.mj_integratePos(model, qpos_new, dq, 1.0)

        # Clamp joint limits
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                adr = model.jnt_qposadr[j]
                r = model.jnt_range[j]
                qpos_new[adr] = np.clip(qpos_new[adr], r[0], r[1])
        d.qpos[:] = qpos_new

    mujoco.mj_forward(model, d)

    errors = {}
    for limb, target in limb_targets.items():
        sid = site_ids[limb]
        errors[limb] = float(np.linalg.norm(target - d.site_xpos[sid]))

    target_pose = {
        name: float(d.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
        for name in ACTUATED_JOINT_NAMES
    }

    return RetargetResult(
        qpos=tuple(float(v) for v in d.qpos[: len(ACTUATED_JOINT_NAMES) + 7]),
        target_pose=target_pose,
        errors=errors,
        converged=converged,
        iterations=iterations_run,
        pelvis_pos=(float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])),
    )
