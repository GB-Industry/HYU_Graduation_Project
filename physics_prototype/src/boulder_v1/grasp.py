from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .contact import (CAPTURE_DISTANCE, CAPTURE_SPEED, CAPTURE_ORIENTATION,
                      GripController, GripDecision, can_attach as pure_can_attach,
                      effective_grip_capacity, orientation_compatibility)
from .contact_geometry import ContactMode
from .support import FootSupportSensor, FootSupportState, FootStatus, fresh_data
from .mjcf_builder import END_EFFECTOR_SITES, get_grasp_equality_name
from .runtime import _import_mujoco, compute_pose_control, validate_reference_pose
from .schema import Affordance, BoulderScene, ClimberProfile, ContactRegion, Limb, Vec3

MAX_HELD_SURFACE_DISTANCE = 0.01  # Retained compliant grip stretch, not an acquisition radius.

# Existing synthetic demo reference, not a validated physical support stance.
# Joint angles are radians. Root translation is (-0.0134, -0.6419, 1.2631)
# before morphology offsets; initialization normalizes the rounded quaternion
# and rejects joint angles outside the actual compiled ROM.
STATIC_STANCE_QPOS: tuple[float, ...] = (
    -0.0134, -0.6419, 1.2631, 0.9973, -0.0729, -0.006, 0.0002,
    -0.0061, -0.2482, -0.0156,
    1.1248, 0.3217, 0.0009, 0.9919, -0.0036,
    1.1731, -0.3564, 0.0021, 1.0684, -0.0028,
    1.768, 0.3104, -0.0244, 1.3863, -0.2831, 0.0012,
    1.7445, -0.3282, 0.0322, 1.3526, -0.2899, -0.0021,
)

STATIC_STANCE_TARGETS: dict[str, float] = {
    "waist_yaw": -0.0061,
    "waist_pitch": -0.2482,
    "waist_roll": -0.0156,
    "left_shoulder_pitch": 1.1248,
    "left_shoulder_roll": 0.3217,
    "left_shoulder_yaw": 0.0009,
    "left_elbow": 0.9919,
    "left_wrist": -0.0036,
    "right_shoulder_pitch": 1.1731,
    "right_shoulder_roll": -0.3564,
    "right_shoulder_yaw": 0.0021,
    "right_elbow": 1.0684,
    "right_wrist": -0.0028,
    "left_hip_pitch": 1.768,
    "left_hip_roll": 0.3104,
    "left_hip_yaw": -0.0244,
    "left_knee": 1.3863,
    "left_ankle_pitch": -0.2831,
    "left_ankle_roll": 0.0012,
    "right_hip_pitch": 1.7445,
    "right_hip_roll": -0.3282,
    "right_hip_yaw": 0.0322,
    "right_knee": 1.3526,
    "right_ankle_pitch": -0.2899,
    "right_ankle_roll": -0.0021,
}


@dataclass(frozen=True)
class GraspAttachment:
    limb: Limb
    region: ContactRegion
    eq_id: int
    eq_name: str


class AttachmentStateError(RuntimeError):
    """The live equality state and its attachment owner disagree."""


class InitialContactError(ValueError):
    """An explicit proposed initial pose cannot establish physical hand grasps."""


@dataclass(frozen=True)
class HandGraspState:
    active: bool
    region_id: str | None
    force_world: tuple[float, float, float]
    load: float
    capacity: float | None
    margin: float | None
    valid: bool
    loaded: bool
    reason: str = ""
    gap: float | None = None
    penetration: float | None = None
    measurement_valid: bool = True


@dataclass(frozen=True)
class ContactSnapshot:
    time: float
    mode: ContactMode
    hands: dict[Limb, HandGraspState]
    feet: dict[Limb, FootSupportState]
    configuration: dict[Limb, str]
    supporting_limbs: tuple[Limb, ...]


class GraspManager:
    """Owns bounded hand point grasps and separately observes native foot support.

    Foot equalities exist only in the compiled IDEALIZED_DEBUG model. They never
    enter the hand capacity evaluator or become physical support measurements.
    """

    def __init__(self, model: Any, data: Any, scene: BoulderScene, profile: ClimberProfile | None = None):
        self.model = model
        self.data = data
        self.scene = scene
        self.profile = profile if profile is not None else ClimberProfile(name="base")
        try:
            flag = float(model.numeric("contact_mode").data[0])
        except KeyError as error:
            raise AttachmentStateError("Compiled model must declare contact_mode") from error
        if flag not in (0, 1):
            raise AttachmentStateError("Invalid compiled contact mode")
        self.mode = ContactMode.PHYSICAL if flag == 0 else ContactMode.IDEALIZED_DEBUG
        self.capture_events: list[dict] = []
        self.releases: list[dict] = []
        self.last_capture_failure: dict[Limb, str] = {}
        self.foot_sensor = FootSupportSensor()
        self.grip_controller = GripController()
        self._attachments: dict[Limb, GraspAttachment] = {}
        self._mujoco = _import_mujoco()
        self.initialized = False
        self._constraints: dict[int, GraspAttachment] = {}
        for region in scene.contact_regions:
            for limb in Limb:
                if limb.is_foot and self.mode == ContactMode.PHYSICAL:
                    continue
                affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
                if affordance not in region.affordances:
                    continue
                name = get_grasp_equality_name(limb, region.id)
                eq_id = self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_EQUALITY, name)
                if eq_id < 0:
                    continue
                att = GraspAttachment(limb, region, eq_id, name)
                self._validate_constraint(att)
                self._constraints[eq_id] = att

    def _validate_constraint(self, att: GraspAttachment) -> None:
        target = ("site_" if att.limb.is_hand else "site_step_") + att.region.id
        site1 = self.model.site(END_EFFECTOR_SITES[att.limb]).id
        site2 = self.model.site(target).id
        eq_id = att.eq_id
        if (self.model.equality(eq_id).name != att.eq_name
                or self.model.eq_type[eq_id] != self._mujoco.mjtEq.mjEQ_CONNECT
                or self.model.eq_objtype[eq_id] != self._mujoco.mjtObj.mjOBJ_SITE
                or self.model.eq_obj1id[eq_id] != site1 or self.model.eq_obj2id[eq_id] != site2):
            raise AttachmentStateError(f"Attachment equality {att.eq_name!r} has incorrect type or sites")

    def _read_live_attachments(self) -> dict[Limb, GraspAttachment]:
        active: dict[Limb, GraspAttachment] = {}
        for eq_id in range(self.model.neq):
            if not self.data.eq_active[eq_id]:
                continue
            att = self._constraints.get(eq_id)
            if att is None:
                name = self.model.equality(eq_id).name
                if name.startswith("grasp_"):
                    raise AttachmentStateError(f"Unregistered active attachment equality {name!r}")
                continue
            if att.limb in active:
                raise AttachmentStateError(f"Multiple active attachment equalities for {att.limb.value}")
            self._validate_constraint(att)
            active[att.limb] = att
        return active

    def active_attachments(self) -> dict[Limb, GraspAttachment]:
        """Derive live identities and reject registry drift without mutating state."""
        active = self._read_live_attachments()
        if active != self._attachments:
            raise AttachmentStateError("Attachment registry disagrees with live eq_active; explicitly initialize or synchronize")
        return active

    def contact_configuration(self) -> dict[Limb, str]:
        """Convenience projection; structured grasp/support state remains authoritative."""
        return self.contact_snapshot().configuration

    def registered_attachment(self, limb: Limb, region: ContactRegion) -> GraspAttachment:
        """Validate even an inactive target before the executor mutates its source."""
        name = get_grasp_equality_name(limb, region.id)
        eq_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, name)
        att = self._constraints.get(eq_id)
        if att is None or att.region != region:
            raise AttachmentStateError(f"Attachment {name!r} does not match the registered scene")
        self._validate_constraint(att)
        return att

    def synchronize_from_live(self) -> None:
        """Explicitly adopt an existing, unambiguous live attachment state."""
        active = self._read_live_attachments()
        self._attachments = active
        self.initialized = True

    def require_session(self, model: Any, data: Any, scene: BoulderScene) -> None:
        if self.model is not model or self.data is not data or self.scene is not scene:
            raise AttachmentStateError("Manager is bound to a different model, data, or scene")
        if not self.initialized:
            raise AttachmentStateError("Episode is not initialized; call initialize_episode or synchronize_from_live")
        self.active_attachments()

    def _capture_measurement(self, limb, region, data=None):
        d = fresh_data(self.model, self.data) if data is None else data
        source = d.site(END_EFFECTOR_SITES[limb])
        target = d.site(("site_" if limb.is_hand else "site_step_") + region.id)
        jp, jr, tp, tr = (np.zeros((3, self.model.nv)) for _ in range(4))
        self._mujoco.mj_jacSite(self.model, d, jp, jr, source.id)
        self._mujoco.mj_jacSite(self.model, d, tp, tr, target.id)
        relative = (jp - tp) @ d.qvel
        outward = target.xmat.reshape(3, 3)[:, 2]
        normal = -source.xmat.reshape(3, 3)[:, 2] if limb.is_hand else source.xmat.reshape(3, 3)[:, 1]
        orientation = orientation_compatibility(tuple(normal), tuple(outward)) if limb.is_hand else 1.
        penetration = 0.
        distance = 0.
        if limb.is_hand:
            distance = float(self._mujoco.mj_geomDistance(
                self.model, d, self.model.geom(f"{limb.value.lower()}_geom").id,
                self.model.geom(f"geom_{region.id}").id, 1., np.zeros(6)))
            if not math.isfinite(distance):
                raise ValueError("Nonfinite signed hand/surface distance")
            penetration = max(0., -distance)
        velocity = np.zeros(self.model.nv)
        self._mujoco.mj_mulM(self.model, d, velocity, d.qvel)
        kinetic = float(d.qvel @ velocity / 2)
        body = int(self.model.site_bodyid[source.id])
        cp, cr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self._mujoco.mj_jacBodyCom(self.model, d, cp, cr, body)
        linear, angular = cp @ d.qvel, cr @ d.qvel
        principal = d.ximat[body].reshape(3, 3)
        tensor = principal @ np.diag(self.model.body_inertia[body]) @ principal.T
        hand_kinetic = float(self.model.body_mass[body] * (linear @ linear) / 2 + angular @ tensor @ angular / 2)
        return {"time_s": float(d.time), "gap_m": float(np.linalg.norm(source.xpos - target.xpos)),
                "relative_velocity_world_m_s": tuple(float(v) for v in relative),
                "relative_speed_m_s": float(np.linalg.norm(relative)), "orientation": float(orientation),
                "penetration_m": penetration, "kinetic_energy_J": kinetic,
                "hand_kinetic_energy_J": hand_kinetic,
                "signed_geom_distance_m": distance,
                "target_position_world_m": tuple(float(v) for v in target.xpos)}

    def _reaction(self, limb, d):
        att = self.active_attachments().get(limb)
        if att is None:
            return (0., 0., 0.)
        rows = [i for i in range(d.nefc) if d.efc_type[i] == self._mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and d.efc_id[i] == att.eq_id]
        if len(rows) != 3:
            raise AttachmentStateError("Active point-grasp force must have three Cartesian rows")
        force = d.efc_force[rows]
        if not np.isfinite(force).all():
            raise ValueError("Nonfinite grasp reaction")
        return tuple(float(v) for v in force)

    def get_grasp_force(self, limb):
        if not limb.is_hand:
            raise ValueError("Foot collision loads are not hand grip forces")
        return self._reaction(limb, fresh_data(self.model, self.data))

    def contact_snapshot(self):
        attachments = self.active_attachments()
        try:
            d = fresh_data(self.model, self.data)
            measurement_valid = True
        except ValueError:
            d, measurement_valid = None, False
        hands = {}
        configuration = {}
        supports = []
        for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            att = attachments.get(limb)
            if att is None:
                hands[limb] = HandGraspState(False, None, (0., 0., 0.), 0., None, None, False, False,
                                            self.last_capture_failure.get(limb, "detached"))
                continue
            force = self._reaction(limb, d) if measurement_valid else (0., 0., 0.)
            load = math.hypot(*force)
            cap = effective_grip_capacity(self.profile, att.region, (0., 0., 1.))
            hand_measurement_valid = measurement_valid
            geometry = None
            if measurement_valid:
                try:
                    geometry = self._capture_measurement(limb, att.region, d)
                except ValueError:
                    hand_measurement_valid = False
            valid = bool(hand_measurement_valid and (self.mode == ContactMode.IDEALIZED_DEBUG
                         or load <= cap and geometry["penetration_m"] < .001
                         and geometry["signed_geom_distance_m"] <= MAX_HELD_SURFACE_DISTANCE))
            hands[limb] = HandGraspState(True, att.region.id, force, load, cap, cap - load, valid,
                                        load > 5., "valid" if valid else "invalid/overloaded grasp",
                                        geometry["gap_m"] if geometry else None,
                                        geometry["penetration_m"] if geometry else None, hand_measurement_valid)
            configuration[limb] = att.region.id
            if valid:
                supports.append(limb)
        feet = self.foot_sensor.measure(self.model, d if measurement_valid else self.data, self.scene,
                                       fresh=not measurement_valid)
        if self.mode == ContactMode.IDEALIZED_DEBUG:
            for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
                att = attachments.get(limb)
                if att is not None:
                    feet[limb] = FootSupportState(status=FootStatus.IDEALIZED, reason="not physical support",
                                                  idealized_attachment=att.region.id)
                    configuration[limb] = att.region.id
                    supports.append(limb)
        else:
            for limb, state in feet.items():
                if state.supporting and not state.slipping:
                    configuration[limb] = state.primary_region
                    supports.append(limb)
        return ContactSnapshot(float(self.data.time), self.mode, hands, feet, configuration, tuple(supports))

    def _get_site_pos_and_normal(self, limb: Limb) -> tuple[Vec3, Vec3]:
        site_name = END_EFFECTOR_SITES[limb]
        site_data = self.data.site(site_name)
        pos = (float(site_data.xpos[0]), float(site_data.xpos[1]), float(site_data.xpos[2]))
        xmat = site_data.xmat.reshape((3, 3))
        if limb.is_hand:
            # Arm reaches along local -Z towards hand site
            normal = (-float(xmat[0, 2]), -float(xmat[1, 2]), -float(xmat[2, 2]))
        else:
            # Foot toe points along local +Y towards wall
            normal = (float(xmat[0, 1]), float(xmat[1, 1]), float(xmat[2, 1]))
        return pos, normal

    def can_attach(
        self,
        limb: Limb,
        region: ContactRegion,
        max_distance: float = CAPTURE_DISTANCE,
        min_orientation: float = CAPTURE_ORIENTATION,
    ) -> bool:
        """Check whether limb end-effector is currently eligible to attach to region."""
        if limb.is_foot and self.mode == ContactMode.PHYSICAL:
            return False
        affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
        if affordance not in region.affordances:
            return False
        info = self._capture_measurement(limb, region)
        if self.mode == ContactMode.IDEALIZED_DEBUG:
            allowed = info["gap_m"] <= max_distance and info["orientation"] >= .2
        else:
            allowed = bool(0 < max_distance <= CAPTURE_DISTANCE
                           and CAPTURE_ORIENTATION <= min_orientation <= 1
                           and info["gap_m"] <= max_distance + 1e-12
                           and info["relative_speed_m_s"] <= CAPTURE_SPEED + 1e-12
                           and info["orientation"] >= min_orientation and info["penetration_m"] < .001
                           and info["signed_geom_distance_m"] <= CAPTURE_DISTANCE + 1e-12)
        if not allowed:
            self.last_capture_failure[limb] = f"capture rejected: gap={info['gap_m']:.6f}m speed={info['relative_speed_m_s']:.6f}m/s orientation={info['orientation']:.3f} penetration={info['penetration_m']:.6f}m"
        return allowed

    def attach(self, limb: Limb, region: ContactRegion, force: bool = False) -> bool:
        """Attach limb to contact region via predeclared connect equality constraint."""
        self.active_attachments()
        if (limb.is_foot or force) and self.mode == ContactMode.PHYSICAL:
            raise ValueError("Physical mode cannot force a grasp or attach a foot")
        affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
        if affordance not in region.affordances:
            raise ValueError(f"Limb {limb} cannot form an attachment on {region.id} with affordance {affordance}")
        if not force and not self.can_attach(limb, region, max_distance=.15 if self.mode == ContactMode.IDEALIZED_DEBUG else CAPTURE_DISTANCE):
            return False

        eq_name = get_grasp_equality_name(limb, region.id)
        eq_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        if eq_id < 0:
            raise KeyError(f"Equality constraint '{eq_name}' not found in compiled model")
        att = self.registered_attachment(limb, region)

        if self.mode == ContactMode.PHYSICAL:
            candidate = fresh_data(self.model, self.data)
            previous = self._attachments.get(limb)
            if previous is not None:
                candidate.eq_active[previous.eq_id] = False
            candidate.eq_active[att.eq_id] = True
            self._mujoco.mj_forward(self.model, candidate)
            rows = [i for i in range(candidate.nefc)
                    if candidate.efc_type[i] == self._mujoco.mjtConstraint.mjCNSTR_EQUALITY
                    and candidate.efc_id[i] == att.eq_id]
            force = candidate.efc_force[rows]
            cap = effective_grip_capacity(self.profile, region, (0., 0., 1.))
            if len(rows) != 3 or not np.isfinite(force).all() or math.hypot(*force) > cap:
                self.last_capture_failure[limb] = "capture reaction exceeds bounded hand capacity"
                return False

        if limb in self._attachments:
            self.detach(limb)

        self.data.eq_active[att.eq_id] = True
        self._attachments[limb] = att
        if limb.is_hand:
            info = self._capture_measurement(limb, region)
            force_vector = self.get_grasp_force(limb)
            self.capture_events.append({"limb": limb.value, "region_id": region.id, "mode": self.mode.value,
                                        **info, "initial_reaction_world_N": force_vector,
                                        "initial_reaction_N": math.hypot(*force_vector),
                                        "peak_window_reaction_N": math.hypot(*force_vector),
                                        "initial_reaction_epoch": "fresh_post_activation_solve",
                                        "peak_endpoint_reaction_N": math.hypot(*force_vector),
                                        "peak_applied_reaction_N": 0.,
                                        "peak_system_kinetic_energy_J": info["kinetic_energy_J"],
                                         "peak_hand_kinetic_energy_J": info["hand_kinetic_energy_J"],
                                         "post_hand_kinetic_energy_J": info["hand_kinetic_energy_J"],
                                         "window_truncated": False, "window_last_sample_time_s": info["time_s"],
                                         "window_s": .02, "post_kinetic_energy_J": info["kinetic_energy_J"]})
        return True

    def detach(self, limb: Limb) -> bool:
        """Detach limb from its current hold."""
        self.active_attachments()
        if limb not in self._attachments:
            return False
        att = self._attachments.pop(limb)
        self.data.eq_active[att.eq_id] = False
        for event in self.capture_events:
            if (event["limb"] == limb.value and not event["window_truncated"]
                    and 0 <= self.data.time - event["time_s"] < event["window_s"] - 1e-12):
                event["window_truncated"] = True
                event["window_truncation_time_s"] = float(self.data.time)
                event["window_truncation_reason"] = "grasp detached"
        return True

    def detach_all(self) -> None:
        """Detach all currently attached limbs."""
        self.active_attachments()
        for limb in list(self._attachments.keys()):
            self.detach(limb)

    def is_attached(self, limb: Limb) -> bool:
        """Return True if limb is currently attached to a hold."""
        return limb in self.active_attachments()

    def active_attachment(self, limb: Limb) -> ContactRegion | None:
        """Return ContactRegion attached to limb, or None if unattached."""
        att = self.active_attachments().get(limb)
        return att.region if att is not None else None

    def get_grasp_load(self, limb: Limb) -> float:
        """Current endpoint-solve hand equality reaction magnitude, in Newtons."""
        return math.hypot(*self.get_grasp_force(limb))

    def evaluate_and_update(self, profile: ClimberProfile | None = None, applied_data=None) -> dict[Limb, GripDecision]:
        """Evaluate active grasp loads against profile grip capacity; break attachments if overloaded."""
        self.active_attachments()
        profile = self.profile if profile is None else profile
        if self.mode == ContactMode.IDEALIZED_DEBUG:
            return {}
        if profile != self.profile:
            raise AttachmentStateError("Capacity evaluation must use the bound physical profile")
        current = fresh_data(self.model, self.data)
        decisions: dict[Limb, GripDecision] = {}
        for limb in list(self._attachments.keys()):
            if not limb.is_hand:
                raise AttachmentStateError("A physical foot must never enter the hand evaluator")
            att = self._attachments[limb]
            force = self._reaction(limb, current)
            endpoint_force = force
            endpoint_load = math.hypot(*force)
            load = endpoint_load
            epoch = "fresh_endpoint_solve"
            force_time = float(self.data.time)
            applied_force, applied_load = None, None
            if applied_data is not None:
                applied_force = self._reaction(limb, applied_data)
                applied_load = math.hypot(*applied_force)
                if applied_load > load:
                    load, force = applied_load, applied_force
                    epoch = "applied_preintegration_solve"
                    force_time = float(applied_data.time - self.model.opt.timestep)
            geometry = self._capture_measurement(limb, att.region, current)
            decision = self.grip_controller.evaluate(
                profile=profile,
                region=att.region,
                hand_normal=(0., 0., 1.),
                required_load=load,
            )
            decisions[limb] = decision
            if (geometry["penetration_m"] >= .001
                    or geometry["signed_geom_distance_m"] > MAX_HELD_SURFACE_DISTANCE):
                decision = GripDecision(False, load, decision.effective_capacity, decision.utilization, "inadmissible palm/surface geometry")
                decisions[limb] = decision
            for event in reversed(self.capture_events):
                if event["limb"] != limb.value:
                    continue
                if (event["limb"] == limb.value and event["region_id"] == att.region.id
                        and not event["window_truncated"]
                        and 0 <= self.data.time - event["time_s"] <= event["window_s"] + 1e-12):
                    event["peak_window_reaction_N"] = max(event["peak_window_reaction_N"], load)
                    event["peak_endpoint_reaction_N"] = max(event["peak_endpoint_reaction_N"], endpoint_load)
                    if applied_load is not None:
                        event["peak_applied_reaction_N"] = max(event["peak_applied_reaction_N"], applied_load)
                    event["post_kinetic_energy_J"] = geometry["kinetic_energy_J"]
                    event["post_hand_kinetic_energy_J"] = geometry["hand_kinetic_energy_J"]
                    event["peak_system_kinetic_energy_J"] = max(event["peak_system_kinetic_energy_J"], geometry["kinetic_energy_J"])
                    event["peak_hand_kinetic_energy_J"] = max(event["peak_hand_kinetic_energy_J"], geometry["hand_kinetic_energy_J"])
                    event["window_last_sample_time_s"] = float(self.data.time)
                break
            if not decision.maintain:
                self.releases.append({"limb": limb.value, "region_id": att.region.id, "time_s": float(self.data.time),
                                      "required_load_N": load, "capacity_N": decision.effective_capacity,
                                      "reason": decision.reason, "force_world_N": force,
                                      "force_epoch": epoch, "force_state_time_s": force_time,
                                      "endpoint_force_world_N": endpoint_force,
                                      "endpoint_load_N": endpoint_load,
                                      "applied_force_world_N": applied_force, "applied_load_N": applied_load})
                self.detach(limb)
        return decisions


def initialize_episode(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile | None = None,
    manager: GraspManager | None = None,
    attach_feet: bool = False,
    initial_qpos=None,
) -> GraspManager:
    """Explicitly reset an episode to the existing demo stance and attachments.

    This operation is not a physical feasibility validation. Transition executors
    never call it. Resetting clears all old equality activations and solver state.
    """
    mujoco = _import_mujoco()
    gm = manager if manager is not None else GraspManager(model, data, scene, profile=profile)
    if gm.model is not model or gm.data is not data or gm.scene is not scene:
        raise AttachmentStateError("Cannot initialize a manager bound to a different session")
    if profile is not None and profile != gm.profile:
        raise AttachmentStateError("Initialization profile differs from the bound contact profile")
    if attach_feet and gm.mode == ContactMode.PHYSICAL:
        raise ValueError("attach_feet is idealized_debug only; physical STEP is collision support")
    initial = {limb: scene.region(hold) for limb, hold in scene.start_configuration.items()
               if limb.is_hand or attach_feet}
    for limb, region in initial.items():
        gm.registered_attachment(limb, region)

    reference = list(STATIC_STANCE_QPOS if initial_qpos is None else initial_qpos)
    if profile is not None and initial_qpos is None:
        reference[1] -= (profile.arm_reach - 0.58) * 0.70
        reference[2] += (profile.leg_reach - 0.82) * 0.85
    norm = math.hypot(*reference[3:7])
    reference[3:7] = [value / norm for value in reference[3:7]]
    # Reject rather than clip: clipping would silently change attachment geometry.
    validate_reference_pose(model, reference)
    if gm.mode == ContactMode.PHYSICAL:
        candidate = mujoco.MjData(model)
        candidate.qpos[:] = reference
        candidate.eq_active[:] = False
        proposed = GraspManager(model, candidate, scene, profile=gm.profile)
        rejected = []
        for limb, region in initial.items():
            if limb.is_hand and not proposed.attach(limb, region):
                rejected.append(f"{limb.value}: {proposed.last_capture_failure.get(limb, 'capture unavailable')}")
        if rejected:
            raise InitialContactError("Initial contact reference rejected; supply an admissible explicit pose. " + "; ".join(rejected))
        decisions = proposed.evaluate_and_update()
        if any(not decision.maintain for decision in decisions.values()):
            raise InitialContactError("Initial hand load/geometry exceeds physical grasp bounds")

    mujoco.mj_resetData(model, data)
    data.eq_active[:] = False
    gm._attachments.clear()
    gm.capture_events.clear()
    gm.releases.clear()
    gm.last_capture_failure.clear()
    gm.initialized = False
    data.qpos[:] = reference
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for limb, region in initial.items():
        if not gm.attach(limb, region, force=gm.mode == ContactMode.IDEALIZED_DEBUG):
            raise InitialContactError("Initial capture unexpectedly rejected")
    mujoco.mj_forward(model, data)
    gm.initialized = True
    gm.active_attachments()
    return gm


def setup_static_stance(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile | None = None,
    manager: GraspManager | None = None,
    attach_feet: bool = False,
    initial_qpos=None,
) -> GraspManager:
    """Existing explicit initialization entry point; resets the whole episode."""
    return initialize_episode(model, data, scene, profile, manager, attach_feet, initial_qpos)


@dataclass(frozen=True)
class StanceResult:
    steps: int
    time: float
    finite: bool
    supported: bool
    left_hand_attached: bool
    right_hand_attached: bool
    left_hand_load: float
    right_hand_load: float
    min_root_z: float
    final_root_z: float


def simulate_static_stance(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    steps: int = 500,
    check_grip: bool = True,
    settle_steps: int = 30,
    kp: float | None = None,
    kd: float | None = None,
    manager: GraspManager | None = None,
    target_pose=None,
    feedforward=None,
) -> StanceResult:
    """Simulate the static four-point stance under full gravity with a free root (NO root stabilization)."""
    mujoco = _import_mujoco()
    manager = manager if manager is not None else initialize_episode(model, data, scene, profile=profile)
    manager.require_session(model, data, scene)
    if manager.mode == ContactMode.PHYSICAL and profile != manager.profile:
        raise AttachmentStateError("Execution profile differs from the bound contact profile")
    targets = target_pose
    if targets is None:
        targets = (STATIC_STANCE_TARGETS if manager.mode == ContactMode.IDEALIZED_DEBUG else
                   {model.joint(int(jid)).name: float(data.qpos[model.jnt_qposadr[jid]])
                    for jid in model.actuator_trnid[:, 0]})

    min_root_z = float(data.qpos[2])
    actual_steps = 0

    for step_idx in range(steps):
        # Joint hold controller holding the static stance angles against gravity
        compute_pose_control(model, data, target_pose=targets, kp=kp, kd=kd, feedforward=feedforward)
        if manager.mode == ContactMode.PHYSICAL:
            decisions = manager.evaluate_and_update(profile)
            if any(not decision.maintain for decision in decisions.values()):
                break
        mujoco.mj_step(model, data)
        actual_steps += 1

        curr_z = float(data.qpos[2])
        if curr_z < min_root_z:
            min_root_z = curr_z

        if manager.mode == ContactMode.PHYSICAL or check_grip:
            decisions = manager.evaluate_and_update(profile, applied_data=data)
            if any(not decision.maintain for decision in decisions.values()):
                break

    values = list(data.qpos) + list(data.qvel)
    finite = all(math.isfinite(float(v)) for v in values)
    supported = float(data.qpos[2]) > 0.8  # Remained supported on wall above 0.8m (did not fall to floor)
    if manager.mode == ContactMode.PHYSICAL:
        supported = supported and len(manager.contact_snapshot().supporting_limbs) >= 3

    return StanceResult(
        steps=actual_steps,
        time=float(data.time),
        finite=finite,
        supported=supported,
        left_hand_attached=manager.is_attached(Limb.LEFT_HAND),
        right_hand_attached=manager.is_attached(Limb.RIGHT_HAND),
        left_hand_load=manager.get_grasp_load(Limb.LEFT_HAND),
        right_hand_load=manager.get_grasp_load(Limb.RIGHT_HAND),
        min_root_z=min_root_z,
        final_root_z=float(data.qpos[2]),
    )
