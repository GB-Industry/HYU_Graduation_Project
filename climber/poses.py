"""Kinematic poses for viewing and as episode starting states (no physics stepping)."""
from __future__ import annotations

import mujoco
import numpy as np


def ik(model: mujoco.MjModel, qpos: np.ndarray, targets: dict[str, np.ndarray],
       weights: dict[str, float] | None = None, dof_mask: np.ndarray | None = None,
       iters: int = 300) -> tuple[np.ndarray, dict[str, float]]:
    """Weighted damped-least-squares IK on body origins, clamped to joint limits.

    Returns the solved qpos and the remaining distance (m) per target body.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    weights = weights or {}
    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        rows, errs = [], []
        for body, target in targets.items():
            jac = np.zeros((3, model.nv))
            bid = model.body(body).id
            mujoco.mj_jacBody(model, d, jac, None, bid)
            w = np.broadcast_to(np.asarray(weights.get(body, 1.0), dtype=float), (3,))  # per-axis weight
            rows.append(w[:, None] * jac)
            errs.append(w * (target - d.xpos[bid]))
        J, e = np.vstack(rows), np.concatenate(errs)
        if dof_mask is not None:
            J = J * dof_mask
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
        mujoco.mj_integratePos(model, d.qpos, 0.5 * dq, 1)
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
                a = model.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *model.jnt_range[j])
    mujoco.mj_kinematics(model, d)
    residual = {b: float(np.linalg.norm(t - d.xpos[model.body(b).id])) for b, t in targets.items()}
    return d.qpos.copy(), residual


def _drop_to_floor(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)
    feet = {model.body("left_foot").id, model.body("right_foot").id}
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    lowest = np.inf
    for i in range(model.ngeom):
        if model.geom_bodyid[i] in feet and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = d.geom_xpos[i] + (signs * model.geom_size[i]) @ d.geom_xmat[i].reshape(3, 3).T
            lowest = min(lowest, corners[:, 2].min())
    q = qpos.copy()
    q[2] -= lowest  # sole box touches z = 0
    return q


def t_pose(model: mujoco.MjModel, y: float = -1.6) -> np.ndarray:
    """Standing, arms out sideways, facing away from the wall (toward a camera at -y)."""
    q = model.qpos0.copy()
    q[0:3] = [0.0, y, 1.25]
    q[3:7] = [0, 0, 0, 1]
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    mask = np.zeros(model.nv)
    for j in range(model.njnt):
        if "shoulder" in model.joint(j).name or "elbow" in model.joint(j).name:
            mask[model.jnt_dofadr[j]] = 1
    targets = {}
    chest_x = d.xpos[model.body("chest").id][0]
    for s in ("left", "right"):
        sh = d.xpos[model.body(f"{s}_upper_arm").id].copy()
        out = np.sign(sh[0] - chest_x)
        reach = float(np.linalg.norm(d.xpos[model.body(f"{s}_hand").id] - sh))
        targets[f"{s}_hand"] = sh + [out * reach, 0, 0]
        targets[f"{s}_forearm"] = sh + [out * reach * 0.53, 0, 0]
    q, _ = ik(model, q, targets, dof_mask=mask)
    return _drop_to_floor(model, q)


def climb_pose(model: mujoco.MjModel, feet=("H1", "H2"),
               hand_options=(("H3", "H4"), ("H7", "H4"), ("H3", "H5"), ("H7", "H5"))) -> tuple[np.ndarray, dict]:
    """Four-point stance on the wall. Tries a few hand-hold pairs / hip placements and
    keeps the one whose hands and feet land closest to their holds."""
    hold = lambda n: model.body(f"hold_{n}").pos.copy()
    limbs = ("left_hand", "right_hand", "left_foot", "right_foot")
    # hip-joint-to-sole length of this body, so the pelvis search scales with the profile
    leg = float(-model.body("left_shin").pos[2] - model.body("left_foot").pos[2] + 0.06)
    foot_z = max(hold(f)[2] for f in feet)
    chest_up = float(model.body("abdomen").pos[2] + model.body("chest").pos[2])  # pelvis -> chest origin
    best = None
    for lh, rh in hand_options:
        for pz in foot_z + leg * np.array([0.35, 0.5, 0.65]):
            for py in (-0.30, -0.38):
                targets = {"left_hand": hold(lh) + [0, -0.06, -0.02], "right_hand": hold(rh) + [0, -0.06, -0.02],
                           "left_foot": hold(feet[0]) + [0, -0.09, 0.03], "right_foot": hold(feet[1]) + [0, -0.09, 0.03],
                           "pelvis": np.array([0.0, py, pz]), "chest": np.array([0.0, py + 0.06, pz + chest_up])}
                q0 = model.qpos0.copy()
                q0[0:3] = [0.0, py, pz]
                q, err = ik(model, q0, targets, weights={"pelvis": 0.3, "chest": 0.3})
                score = max(err[k] for k in limbs) + 0.2 * (err["pelvis"] + err["chest"])
                if best is None or score < best[0]:
                    best = (score, q, targets, (lh, rh))
    # refine: keep the torso close to the wall (y) but let its height float, so hands and
    # feet settle onto their holds instead of the hips swinging out from the wall
    _, q, targets, hands = best
    torso_w = np.array([0.05, 0.6, 0.02])
    q, err = ik(model, q, targets, weights={"pelvis": torso_w, "chest": torso_w}, iters=400)
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    wall_gap = {b: float(-d.xpos[model.body(b).id][1]) for b in ("pelvis", "chest")}
    return q, {"hands": hands, "feet": feet, "residual": err, "wall_gap": wall_gap}
