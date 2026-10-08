"""Reference support allocation for one-hand release; only motor torque is applied."""
from __future__ import annotations

import mujoco
import numpy as np
from collections.abc import Mapping

from .contact import effective_grip_capacity
from .contact_geometry import canonical_geometry
from .runtime import validate_reference_pose
from .schema import Limb
from .static_state import _foot_residual


def estimate_support_torques(model, scene, profile, qpos, contacts, *, foot_points=None):
    """Project nominal gravity shares onto six free-root balance rows.

    One or two ordinary feet and bounded hands are required. Real sole/face
    overlap centers define foot force points. This estimate is not measured
    support, and its forces are never injected into native physics.
    """
    feet = tuple(limb for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT) if limb in contacts)
    hands = tuple(limb for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND) if limb in contacts)
    if (not isinstance(contacts, Mapping) or any(not isinstance(l, Limb) for l in contacts)
            or not feet or not hands or len(feet) + len(hands) < 3 or set(contacts) != set((*feet, *hands))):
        raise ValueError("Support allocation requires at least three declared foot/hand contacts")
    if foot_points is not None and (not isinstance(foot_points, Mapping)
            or any(not isinstance(l, Limb) or l not in feet for l in foot_points)):
        raise ValueError("Measured force points must identify declared supporting feet")
    validate_reference_pose(model, qpos)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.eq_active[:] = False
    mujoco.mj_forward(model, data)
    weight = -model.body_subtreemass[model.body("climber_root").id] * model.opt.gravity
    blocks, points, nominal = [], [], []
    for limb in (*feet, *hands):
        region = scene.region(contacts[limb])
        if limb.is_foot:
            geometry = canonical_geometry(region)
            if foot_points is not None and limb in foot_points:
                point = np.array(foot_points[limb], dtype=float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError("Measured foot force point must be a finite world vector")
                face = geometry.foot_surface_frame
                local = np.array(face.rotation).T @ (point - face.position)
                shoe = model.geom(f"{limb.value.lower()}_geom")
                shoe_local = data.geom_xmat[shoe.id].reshape(3, 3).T @ (point - data.geom_xpos[shoe.id])
                if (np.any(np.abs(shoe_local) > shoe.size + .001)
                        or geometry.shape == "box" and (abs(local[2]) > .001
                           or np.any(np.abs(local[:2]) > np.array(geometry.size[:2]) + .001))
                        or geometry.shape == "sphere" and abs(np.linalg.norm(point - region.position) - region.radius) > .001):
                    raise ValueError("Estimated native force point lies outside the real shoe/surface patch")
            else:
                evidence = _foot_residual(model, data, limb, region, geometry)
                point = np.array(evidence.support_point_world)
            body = int(model.geom(f"{limb.value.lower()}_geom").bodyid[0])
            share = .9 / len(feet)
        else:
            site = data.site(f"{limb.value.lower()}_site")
            point, body, share = site.xpos.copy(), int(model.site_bodyid[site.id]), .1 / len(hands)
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, body)
        blocks.append(jac.T)
        points.append(point.tolist())
        nominal.extend(share * weight)
    jacobian = np.concatenate(blocks, axis=1)
    root = jacobian[:6]
    nominal = np.array(nominal)
    force = nominal + np.linalg.lstsq(root, data.qfrc_bias[:6] - root @ nominal, rcond=None)[0]
    if not np.isfinite(force).all() or np.max(np.abs(root @ force - data.qfrc_bias[:6])) > 1e-8:
        raise ValueError("Support reference root balance failed")
    for index, limb in enumerate((*feet, *hands)):
        region = scene.region(contacts[limb])
        vector = force[3 * index:3 * index + 3]
        if limb.is_foot:
            local = np.array(canonical_geometry(region).foot_surface_frame.rotation).T @ vector
            if local[2] <= 5. or np.abs(local[:2]).sum() > min(1.8, region.friction) * local[2] + 1e-8:
                raise ValueError("Support reference violates unilateral compression/friction")
        elif np.linalg.norm(vector) > effective_grip_capacity(profile, region, region.normal):
            raise ValueError("Support reference exceeds remaining-hand capacity")
    joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joints]
    torque = data.qfrc_bias[dofs] - (jacobian @ force)[dofs]
    ceiling = model.actuator_gear[:, 0] * np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    if not np.isfinite(torque).all() or np.any(np.abs(torque) > ceiling):
        raise ValueError("Support reference exceeds original motor capability")
    return {"torques_Nm": {model.joint(int(j)).name: float(v) for j, v in zip(joints, torque)},
            "contacts": dict(contacts), "forces_world_N": force.reshape(-1, 3).tolist(), "points_world_m": points,
            "root_balance_residual": float(np.max(np.abs(root @ force - data.qfrc_bias[:6]))),
            "scope": "estimated support reactions; only derived hinge motor torque is applied"}
