"""Read-only native contact measurements; feet are never grasp attachments."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import mujoco
import numpy as np

from .contact_geometry import canonical_geometry
from .schema import Affordance, BoulderScene, Limb, SourceType

MIN_FOOT_LOAD = 5.0
MAX_SLIP_SPEED = 0.01
MIN_NORMAL_ALIGNMENT = 0.9
FORCE_TOLERANCE = 1e-6


def fresh_data(model, data):
    """An endpoint solve, never copied back into live integration state."""
    inputs = (model.geom_size, model.geom_pos, model.geom_quat, model.geom_friction,
              model.site_pos, model.site_quat, model.body_mass, model.body_inertia,
              model.eq_solref, model.eq_solimp)
    if not all(np.isfinite(v).all() for v in inputs):
        raise ValueError("Nonfinite compiled contact geometry or parameters")
    if not math.isfinite(data.time) or not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl, data.qacc_warmstart)):
        raise ValueError("Nonfinite state cannot be sampled as a physical contact")
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)
    if not all(np.isfinite(v).all() for v in (scratch.qacc, scratch.efc_force, scratch.site_xpos,
                                              scratch.site_xmat, scratch.geom_xpos, scratch.geom_xmat)):
        raise ValueError("Nonfinite contact solve")
    return scratch


class FootStatus(str, Enum):
    NO_CONTACT = "FOOT_NO_CONTACT"
    CONTACTING = "FOOT_CONTACTING"
    SUPPORTING = "FOOT_SUPPORTING"
    SLIPPING = "FOOT_SLIPPING"
    INVALID = "FOOT_INVALID_MEASUREMENT"
    IDEALIZED = "FOOT_IDEALIZED_DEBUG_ATTACHMENT"


@dataclass(frozen=True)
class FootContact:
    region_id: str
    shoe_geom: str
    surface_geom: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    force_world: tuple[float, float, float]
    normal_force: float
    tangential_force: float
    tangential_speed: float
    friction: float
    friction_utilization: float
    sole_alignment: float
    surface_alignment: float
    admissible: bool


@dataclass(frozen=True)
class FootSupportState:
    status: FootStatus = FootStatus.NO_CONTACT
    contacting: bool = False
    supporting: bool = False
    slipping: bool = False
    normal_force: float = 0.0
    tangential_force: float = 0.0
    tangential_speed: float = 0.0
    friction_utilization: float = 0.0
    support_regions: tuple[str, ...] = ()
    contacts: tuple[FootContact, ...] = ()
    measurement_valid: bool = True
    reason: str = ""
    idealized_attachment: str | None = None
    support_surfaces: tuple[str, ...] = ()

    @property
    def primary_surface(self):
        if not self.support_surfaces:
            return None
        return max(self.support_surfaces, key=lambda surface: sum(c.normal_force for c in self.contacts
                                                                  if c.surface_geom == surface and c.admissible))

    @property
    def primary_region(self):
        surface = self.primary_surface
        if surface is None:
            return None
        return next(c.region_id for c in self.contacts if c.surface_geom == surface)


class FootSupportSensor:
    def measure(self, model, data, scene: BoulderScene, *, fresh=True):
        feet = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
        try:
            d = fresh_data(model, data) if fresh else data
        except ValueError as error:
            return {limb: FootSupportState(status=FootStatus.INVALID, measurement_valid=False, reason=str(error))
                    for limb in feet}
        surfaces = {}
        for region in scene.contact_regions:
            if region.source_type == SourceType.HOLD:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{region.id}")
                if gid >= 0:
                    surfaces[gid] = (region.id, np.array(canonical_geometry(region).foot_surface_frame.normal),
                                     Affordance.STEP in region.affordances)
        for wall in scene.walls:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{wall.id}_geom")
            if gid >= 0:
                surfaces[gid] = (wall.id, np.array(wall.normal), False)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor >= 0:
            surfaces[floor] = ("floor", np.array([0., 0., 1.]), True)
        result = {}
        for limb in feet:
            name = f"{limb.value.lower()}_geom"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                result[limb] = FootSupportState()
                continue
            shoe_body = int(model.geom_bodyid[gid])
            records = []
            for index, contact in enumerate(d.contact):
                if gid not in (int(contact.geom1), int(contact.geom2)):
                    continue
                other = int(contact.geom1) if contact.geom2 == gid else int(contact.geom2)
                if other not in surfaces:
                    continue
                region, expected_normal, eligible = surfaces[other]
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, d, index, wrench)
                sign = 1 if contact.geom2 == gid else -1
                frame = contact.frame.reshape(3, 3)
                normal = sign * frame[0]
                force = sign * frame.T @ wrench[:3]
                fn, ft = float(wrench[0]), float(np.linalg.norm(wrench[1:3]))
                jp, jr, op, ori = (np.zeros((3, model.nv)) for _ in range(4))
                mujoco.mj_jac(model, d, jp, jr, contact.pos, shoe_body)
                mujoco.mj_jac(model, d, op, ori, contact.pos, int(model.geom_bodyid[other]))
                velocity = (jp - op) @ d.qvel
                slip = float(np.linalg.norm(velocity - normal * (velocity @ normal)))
                mu = float(min(contact.friction[:2]))
                utilization = ft / max(mu * fn, 1e-12)
                sole = float(d.geom_xmat[gid].reshape(3, 3)[:, 2] @ normal)
                alignment = float(expected_normal @ normal)
                finite = np.isfinite([fn, ft, slip, mu, utilization, sole, alignment, *force, *normal]).all()
                if not finite or fn < -FORCE_TOLERANCE:
                    raise ValueError("Invalid native foot force or contact frame")
                admissible = bool(eligible and contact.efc_address >= 0 and fn > 0
                                  and sole >= MIN_NORMAL_ALIGNMENT and alignment >= MIN_NORMAL_ALIGNMENT
                                  and ft <= mu * fn + FORCE_TOLERANCE)
                records.append(FootContact(region, name, model.geom(other).name,
                                            tuple(float(v) for v in contact.pos), tuple(float(v) for v in normal),
                                            tuple(float(v) for v in force), fn, ft, slip, mu, utilization,
                                            sole, alignment, admissible))
            loaded = [c for c in records if c.normal_force > FORCE_TOLERANCE]
            slipping = any(c.tangential_speed > MAX_SLIP_SPEED for c in loaded)
            supported_surfaces = tuple(dict.fromkeys(c.surface_geom for c in records if c.admissible
                                          and sum(k.normal_force for k in records
                                                  if k.surface_geom == c.surface_geom and k.admissible) > MIN_FOOT_LOAD
                                          and not any(k.tangential_speed > MAX_SLIP_SPEED for k in loaded
                                                      if k.surface_geom == c.surface_geom)))
            regions = tuple(dict.fromkeys(c.region_id for c in records if c.surface_geom in supported_surfaces))
            support = bool(supported_surfaces)
            status = (FootStatus.SLIPPING if slipping else FootStatus.SUPPORTING if support
                      else FootStatus.CONTACTING if records else FootStatus.NO_CONTACT)
            result[limb] = FootSupportState(status, bool(records), support, slipping,
                                           sum(c.normal_force for c in records), sum(c.tangential_force for c in records),
                                           max((c.tangential_speed for c in loaded), default=0.),
                                           max((c.friction_utilization for c in loaded), default=0.), regions, tuple(records),
                                           support_surfaces=supported_surfaces)
        return result
