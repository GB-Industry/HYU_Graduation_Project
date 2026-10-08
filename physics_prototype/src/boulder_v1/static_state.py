"""Explicit Stage3 seed admission and owner-sampled, sustained physical readiness."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from .contact_geometry import FOOT_SITE_OFFSET, ContactMode, canonical_geometry
from .grasp import AttachmentStateError, GraspManager, InitialContactError, initialize_episode
from .runtime import compiled_numerical_issues, validate_reference_pose
from .schema import Affordance, BoulderScene, ClimberProfile, Limb, SourceType
from .support import FORCE_TOLERANCE, MAX_SLIP_SPEED, MIN_FOOT_LOAD, MIN_NORMAL_ALIGNMENT, fresh_data

HANDS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)
FEET = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)
TOUCH_TOLERANCE = 1e-6


def _integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def _finite_state(model, data):
    return all(np.isfinite(value).all() for value in
               (_integration_state(model, data), data.qacc, data.qacc_warmstart,
                data.qfrc_actuator, data.qfrc_constraint, data.efc_force))


def _no_foot_equalities(model):
    foot_bodies = {int(model.geom(f"{limb.value.lower()}_geom").bodyid[0]) for limb in FEET}
    for eq in range(model.neq):
        if model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_SITE:
            bodies = {int(model.site_bodyid[int(s)]) for s in (model.eq_obj1id[eq], model.eq_obj2id[eq]) if s >= 0}
        elif model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_BODY:
            bodies = {int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])}
        else:
            bodies = set()
        if bodies & foot_bodies:
            raise InitialContactError("Physical static references require zero foot equalities")


def _unexpected_loaded_contacts(model, data, allowed):
    """Inspect real compiled collision pairs, including body/self contacts."""
    unexpected = []
    for index, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise ValueError("Nonfinite native contact force")
        pair = frozenset(model.geom(gid).name or f"geom#{gid}"
                         for gid in (int(contact.geom1), int(contact.geom2)))
        if pair not in allowed and np.linalg.norm(force[:3]) > FORCE_TOLERANCE:
            unexpected.append(tuple(sorted(pair)))
    return tuple(unexpected)


@dataclass(frozen=True)
class StaticResidual:
    limb: Limb
    region_id: str
    signed_geom_distance: float
    site_normal_residual: float
    orientation_alignment: float
    tangential_offset: tuple[float, float]
    overlap_area: float | None
    native_adjacency: bool
    hand_load: float | None = None
    hand_capacity: float | None = None
    support_point_world: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class StaticReference:
    """Admitted seed, not a certificate of equilibrium or controller convergence."""

    qpos: tuple[float, ...]
    target_pose: Mapping[str, float]
    contact_intent: Mapping[Limb, str]
    residuals: tuple[StaticResidual, ...]

    def __post_init__(self):
        object.__setattr__(self, "qpos", tuple(float(v) for v in self.qpos))
        object.__setattr__(self, "target_pose", MappingProxyType(dict(self.target_pose)))
        object.__setattr__(self, "contact_intent", MappingProxyType(dict(self.contact_intent)))
        object.__setattr__(self, "residuals", tuple(self.residuals))


def _canonical_surface(model, data, region):
    geometry = canonical_geometry(region)
    geom = model.geom(f"geom_{region.id}")
    kind = mujoco.mjtGeom.mjGEOM_BOX if geometry.shape == "box" else mujoco.mjtGeom.mjGEOM_SPHERE
    if (int(geom.type[0]) != kind
            or not np.allclose(geom.size[:len(geometry.size)], geometry.size, rtol=0, atol=1e-9)
            or model.body_weldid[int(geom.bodyid[0])] != 0
            or not np.allclose(data.geom_xpos[geom.id], geometry.body_frame.position, rtol=0, atol=1e-9)
            or not np.allclose(data.geom_xmat[geom.id].reshape(3, 3), geometry.body_frame.rotation, rtol=0, atol=1e-9)):
        raise InitialContactError(f"{region.id}: compiled hold differs from canonical Stage2 geometry")
    for prefix, frame in (("site_", geometry.hand_frame), ("site_step_", geometry.foot_frame)):
        if prefix == "site_step_" and Affordance.STEP not in region.affordances:
            continue
        site = data.site(prefix + region.id)
        if (model.site_bodyid[site.id] != geom.bodyid[0]
                or not np.allclose(site.xpos, frame.position, rtol=0, atol=1e-9)
                or not np.allclose(site.xmat.reshape(3, 3), frame.rotation, rtol=0, atol=1e-9)):
            raise InitialContactError(f"{region.id}: compiled target frame differs from canonical Stage2 geometry")
    return geometry


def _foot_residual(model, data, limb, region, geometry):
    shoe = model.geom(f"{limb.value.lower()}_geom")
    hold = model.geom(f"geom_{region.id}")
    if int(shoe.type[0]) != mujoco.mjtGeom.mjGEOM_BOX:
        raise InitialContactError("Static sole validation requires the compiled box shoe")
    rotation = data.geom_xmat[shoe.id].reshape(3, 3)
    face = geometry.foot_surface_frame
    frame = np.array(face.rotation)
    normal = frame[:, 2]
    alignment = float(rotation[:, 2] @ normal)
    size = shoe.size
    sole_center = data.geom_xpos[shoe.id] - size[2] * rotation[:, 2]
    corners = np.array([sole_center + x * size[0] * rotation[:, 0] + y * size[1] * rotation[:, 1]
                        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    local = (corners - face.position) @ frame
    site_local = frame.T @ (data.site(f"{limb.value.lower()}_site").xpos - face.position)
    distance = float(mujoco.mj_geomDistance(model, data, shoe.id, hold.id, 1., None))
    adjacency = any({int(c.geom1), int(c.geom2)} == {shoe.id, hold.id} for c in data.contact)
    overlap_area = None
    if geometry.shape == "box":
        # Clip the actual sole quadrilateral to the actual support face. The site
        # need not coincide with its center, and the whole shoe need not fit.
        polygon = list(local[:, :2])
        for axis, bound in enumerate(geometry.size[:2]):
            for sign in (-1, 1):
                clipped = []
                for index, current in enumerate(polygon):
                    previous = polygon[index - 1]
                    a, b = bound - sign * previous[axis], bound - sign * current[axis]
                    if (a >= 0) != (b >= 0):
                        clipped.append(previous + a / (a - b) * (current - previous))
                    if b >= 0:
                        clipped.append(current)
                polygon = clipped
        overlap_area = abs(sum(float(polygon[i - 1][0] * p[1] - polygon[i - 1][1] * p[0])
                                for i, p in enumerate(polygon))) / 2
        support_point = np.array(face.position) + frame[:, :2] @ np.mean(polygon, axis=0) if polygon else np.array(face.position)
        inside = all(abs(site_local[i]) <= geometry.size[i] + TOUCH_TOLERANCE for i in (0, 1))
        overlap = overlap_area > 1e-12
    else:
        # A sphere has a selected tangent point, not a rectangular support face.
        point = rotation.T @ (np.array(face.position) - sole_center)
        inside = overlap = bool(np.all(np.abs(point[:2]) <= size[:2] + TOUCH_TOLERANCE))
        support_point = np.array(face.position)
    residual = float(site_local[2] - FOOT_SITE_OFFSET)
    if (not np.isfinite([distance, alignment, residual, *site_local, *local.ravel()]).all()
            or alignment < MIN_NORMAL_ALIGNMENT or not inside or not overlap
            or not -.001 < distance <= TOUCH_TOLERANCE
            or not -.001 < float(np.min(local[:, 2])) <= TOUCH_TOLERANCE
            or abs(residual) > .001):
        raise InitialContactError(f"{limb.value}: sole is not touching the intended canonical support face with real overlap")
    return StaticResidual(limb, region.id, distance, residual, alignment,
                          tuple(float(v) for v in site_local[:2]), overlap_area, adjacency,
                          support_point_world=tuple(float(v) for v in support_point))


def initialize_static_reference(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    initial_qpos: Sequence[float],
    contact_intent: Mapping[Limb, str],
) -> tuple[StaticReference, GraspManager]:
    """Preflight a full explicit seed, then reset via initialize_episode.

    No IK, profile-name offsets, or force captures are used. Conflicting declared
    starts are rejected. Missing starts are completed in a frozen scene copy;
    use the returned manager.scene for subsequent session-bound execution.
    Invalid admission never changes live MjData or the caller's scene/manager.
    """
    if not _finite_state(model, data):
        raise InitialContactError("Nonfinite live integration/control state before reset")
    issues = compiled_numerical_issues(model)
    if issues:
        raise InitialContactError(f"Invalid compiled numerics: {', '.join(issues)}")
    seed = np.asarray(initial_qpos, dtype=float)
    if seed.shape != (model.nq,):
        raise InitialContactError("Supply a complete one-dimensional model.nq reference")
    validate_reference_pose(model, seed)
    intent = dict(contact_intent)
    if set(intent) != set(Limb) or any(not isinstance(limb, Limb) for limb in intent):
        raise InitialContactError("Supply explicit HOLD intent for all four limbs")
    if any(intent.get(limb) != hold for limb, hold in scene.start_configuration.items()):
        raise InitialContactError("Contact intent conflicts with scene.start_configuration")
    configured_scene = scene if dict(scene.start_configuration) == intent else replace(scene, start_configuration=intent)
    scratch = mujoco.MjData(model)
    proposed = GraspManager(model, scratch, configured_scene, profile=profile)
    if proposed.mode != ContactMode.PHYSICAL:
        raise InitialContactError("StaticReference is physical only, not idealized_debug")
    _no_foot_equalities(model)
    scratch.qpos[:] = seed
    scratch.eq_active[:] = False
    mujoco.mj_forward(model, scratch)
    residuals = []
    for limb in Limb:
        region = configured_scene.region(intent[limb])
        affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
        if region.source_type != SourceType.HOLD or affordance not in region.affordances:
            raise InitialContactError(f"{limb.value}: intent must be an eligible HOLD")
        geometry = _canonical_surface(model, scratch, region)
        if limb.is_foot:
            residuals.append(_foot_residual(model, scratch, limb, region, geometry))
    # Exercise the exact production reset/capture path on scratch first, including
    # sequential acquisition and the final simultaneous bounded reaction solve.
    proposed = initialize_episode(model, scratch, configured_scene, profile, manager=proposed, initial_qpos=seed)
    snapshot = proposed.contact_snapshot()
    for limb in HANDS:
        hand = snapshot.hands[limb]
        if not hand.active or not hand.valid or not hand.measurement_valid or hand.load > hand.capacity:
            raise InitialContactError(f"{limb.value}: initial hand reaction/geometry is inadmissible")
        event = next(e for e in proposed.capture_events if e["limb"] == limb.value)
        residuals.append(StaticResidual(limb, intent[limb], event["signed_geom_distance_m"], event["gap_m"],
                                        event["orientation"], (0., 0.), None, False, hand.load, hand.capacity))
    allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{hold}")) for limb, hold in intent.items()}
    endpoint = fresh_data(model, scratch)
    if not _finite_state(model, scratch) or _unexpected_loaded_contacts(model, endpoint, allowed):
        raise InitialContactError("Initial seed has nonfinite state or unintended loaded body contacts")
    targets = {model.joint(jid).name: float(seed[int(model.jnt_qposadr[jid])])
               for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE}
    reference = StaticReference(tuple(scratch.qpos), targets, intent, tuple(residuals))
    manager = initialize_episode(model, data, configured_scene, profile, initial_qpos=seed)
    return reference, manager


@dataclass(frozen=True)
class ReadinessEvidence:
    ready: bool = False
    duration: float = 0.
    reason: str = "No authoritative endpoint sampled"
    time: float | None = None
    root_linear_speed: float | None = None
    root_angular_speed: float | None = None
    max_hinge_speed: float | None = None
    rms_hinge_speed: float | None = None


class ReadinessTracker:
    """One owner calls sample_after_step(before_time) after each native mj_step.

    Construction/reset is not a sample. A valid first endpoint starts the window
    at that endpoint, never at its unsampled interval start. Queries cannot add
    duration. Bind this tracker only to authoritative live data, not observers.
    """

    def __init__(self, model: Any, data: Any, manager: GraspManager, *, required_duration: float = .5):
        if not math.isfinite(required_duration) or required_duration <= 0:
            raise ValueError("Readiness requires a finite positive sustained duration")
        manager.require_session(model, data, manager.scene)
        if manager.mode != ContactMode.PHYSICAL:
            raise ValueError("ReadinessTracker is physical only")
        _no_foot_equalities(model)
        self.model, self.data, self.manager = model, data, manager
        self.required_duration = float(required_duration)
        self._hinges = model.jnt_dofadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE].copy()
        roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(roots) != 1:
            raise ValueError("Readiness requires a single free root")
        self._root = int(model.jnt_dofadr[roots[0]])
        self.reset()

    def _session_token(self):
        # Adoption replaces this registry; reset recaptures new attachment/event
        # objects. Their identities distinguish even a same-clock reset/adoption.
        gm = self.manager
        return (id(gm.scene), id(gm.profile), id(gm._attachments),
                tuple((limb, id(att)) for limb, att in gm._attachments.items()),
                tuple(id(event) for event in gm.capture_events), len(gm.releases))

    def reset(self):
        self._last_time = float(self.data.time)
        self._last_state = _integration_state(self.model, self.data)
        self._token = self._session_token()
        self._lifecycle_objects = (self.manager._attachments, *self.manager._attachments.values(), *self.manager.capture_events)
        self._warnings = tuple(int(w.number) for w in self.data.warning)
        self._timestep = float(self.model.opt.timestep)
        self._start = None
        self._sampled = False
        self._evidence = ReadinessEvidence()

    def invalidate(self, reason: str):
        self._start = None
        self._evidence = replace(self._evidence, ready=False, duration=0., reason=reason)

    def _measure(self):
        model, data, gm = self.model, self.data, self.manager
        gm.require_session(model, data, gm.scene)
        if not _finite_state(model, data):
            return "Nonfinite integration/control state", (None,) * 4
        root = data.qvel[self._root:self._root + 6]
        hinges = data.qvel[self._hinges]
        linear, angular = float(np.linalg.norm(root[:3])), float(np.linalg.norm(root[3:]))
        maximum = float(np.max(np.abs(hinges), initial=0.))
        rms = float(np.sqrt(np.mean(hinges ** 2))) if len(hinges) else 0.
        speeds = (linear, angular, maximum, rms)
        snapshot = gm.contact_snapshot()
        for limb in HANDS:
            hand = snapshot.hands[limb]
            if (not hand.active or not hand.valid or not hand.measurement_valid or hand.capacity is None
                    or not np.isfinite([hand.load, hand.capacity, *hand.force_world]).all()
                    or hand.load > hand.capacity):
                return f"{limb.value}: missing, invalid, or overloaded hand", speeds
            if math.hypot(*gm._reaction(limb, data)) > hand.capacity:
                return f"{limb.value}: applied native hand reaction exceeds capacity", speeds
        for limb in FEET:
            foot = snapshot.feet[limb]
            if (not foot.measurement_valid or not foot.supporting or foot.slipping
                    or not np.isfinite([foot.normal_force, foot.tangential_speed, foot.friction_utilization]).all()
                    or foot.normal_force <= MIN_FOOT_LOAD or foot.tangential_speed > MAX_SLIP_SPEED
                    or any(c.normal_force > FORCE_TOLERANCE and not c.admissible for c in foot.contacts)):
                return f"{limb.value}: both feet must support without slip", speeds
        allowed = {frozenset((f"{limb.value.lower()}_geom", f"geom_{snapshot.hands[limb].region_id}")) for limb in HANDS}
        for limb in FEET:
            allowed.update(frozenset((f"{limb.value.lower()}_geom", surface))
                           for surface in snapshot.feet[limb].support_surfaces)
        if (_unexpected_loaded_contacts(model, data, allowed)
                or _unexpected_loaded_contacts(model, fresh_data(model, data), allowed)):
            return "Unintended loaded body contact", speeds
        if linear > .02:
            return "Root linear speed exceeds 0.02 m/s", speeds
        if angular > .05:
            return "Root angular speed exceeds 0.05 rad/s", speeds
        if maximum > .10:
            return "Maximum hinge speed exceeds 0.10 rad/s", speeds
        return "", speeds

    def sample_after_step(self, before_time: float) -> ReadinessEvidence:
        """Record exactly one authoritative endpoint; skipped intervals invalidate."""
        now = float(self.data.time)
        state = _integration_state(self.model, self.data)
        token = self._session_token()
        warnings = tuple(int(w.number) for w in self.data.warning)
        timestep = float(self.model.opt.timestep)
        same_time = now == self._last_time
        try:
            reason, speeds = self._measure()
        except (AttachmentStateError, ValueError, KeyError) as error:
            reason, speeds = str(error), (None,) * 4
        if (same_time and self._sampled and np.array_equal(state, self._last_state)
                and token == self._token and warnings == self._warnings and timestep == self._timestep
                and math.isfinite(before_time)
                and math.isclose(now - before_time, timestep, rel_tol=1e-8, abs_tol=1e-12)):
            if reason:
                self._start = None
                self._evidence = ReadinessEvidence(False, 0., reason, now, *speeds)
            return self._evidence
        continuous = (math.isfinite(now) and math.isfinite(before_time) and math.isfinite(timestep) and timestep > 0
                      and math.isclose(before_time, self._last_time, rel_tol=0, abs_tol=1e-12)
                      and math.isclose(now - before_time, timestep, rel_tol=1e-8, abs_tol=1e-12)
                      and timestep == self._timestep and token == self._token and warnings == self._warnings)
        if not continuous:
            reason = "Gap, reset, adoption, or invalid native integration endpoint"
        if reason:
            self._start = None
        elif self._start is None:
            self._start = now
        duration = 0. if self._start is None else now - self._start
        ready = not reason and duration + 1e-12 >= self.required_duration
        self._evidence = ReadinessEvidence(ready, duration, reason or ("Sustained physical readiness" if ready
                                           else "Awaiting sustained physical readiness"), now, *speeds)
        self._last_time, self._last_state, self._token = now, state, token
        self._lifecycle_objects = (self.manager._attachments, *self.manager._attachments.values(), *self.manager.capture_events)
        self._warnings, self._timestep, self._sampled = warnings, timestep, True
        return self._evidence

    @property
    def evidence(self) -> ReadinessEvidence:
        """Read-only query; reject stale evidence rather than certify a crossing."""
        if (float(self.data.time) != self._last_time or self._session_token() != self._token
                or tuple(int(w.number) for w in self.data.warning) != self._warnings
                or float(self.model.opt.timestep) != self._timestep
                or not np.array_equal(_integration_state(self.model, self.data), self._last_state)):
            return ReadinessEvidence(reason="Live state changed since the authoritative readiness sample")
        try:
            reason, speeds = self._measure()
        except (AttachmentStateError, ValueError, KeyError) as error:
            return ReadinessEvidence(reason=str(error))
        if reason:
            return ReadinessEvidence(reason=reason, time=float(self.data.time), root_linear_speed=speeds[0],
                                     root_angular_speed=speeds[1], max_hinge_speed=speeds[2], rms_hinge_speed=speeds[3])
        return self._evidence

    def check(self) -> tuple[bool, str]:
        evidence = self.evidence
        return evidence.ready, evidence.reason
