from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any

import numpy as np

from .contact import Affordance, CAPTURE_DISTANCE
from .contact_geometry import ContactMode
from .grasp import AttachmentStateError, GraspManager
from .retargeter import StanceSpecification, solve_retargeted_stance
from .runtime import _import_mujoco, compute_pose_control
from .schema import BoulderScene, ClimberProfile, Limb
from .support import MIN_FOOT_LOAD, MAX_SLIP_SPEED
from .static_state import ReadinessTracker


class TransitionPhase(str, Enum):
    INITIAL_STANCE = "initial_stance"
    PRE_SHIFT = "pre_shift"
    RELEASE_LIMB = "release_limb"
    SUPPORT_PHASE = "support_phase"
    REACH_PHASE = "reach_phase"
    ATTACH_PHASE = "attach_phase"
    STABILIZED_STANCE = "stabilized_stance"


class TransitionStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_NOT_ATTACHED = "SOURCE_NOT_ATTACHED"
    INELIGIBLE_TARGET = "INELIGIBLE_TARGET"
    SUPPORT_FAILURE = "SUPPORT_FAILURE"
    GRIP_FAILURE = "GRIP_FAILURE"
    INCOMPLETE = "INCOMPLETE"
    NONFINITE_STATE = "NONFINITE_STATE"
    ATTACH_FAILURE = "ATTACH_FAILURE"
    UNSTABLE_FINAL_STATE = "UNSTABLE_FINAL_STATE"


@dataclass(frozen=True)
class StateSummary:
    """Snapshot summary of character physical and contact state."""

    time: float
    root_pos: tuple[float, float, float]
    root_vel: tuple[float, float, float]
    qvel_norm: float
    finite: bool
    contact_configuration: dict[Limb, str]
    attachment_loads: dict[Limb, float]
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    eq_active: tuple[bool, ...]
    ctrl: tuple[float, ...]
    qacc_warmstart: tuple[float, ...]
    attachment_constraints: dict[Limb, str]
    hand_states: dict
    foot_states: dict
    contact_mode: ContactMode
    capture_events: tuple[dict, ...]
    release_events: tuple[dict, ...]


@dataclass(frozen=True)
class TransitionRequest:
    """Structured request defining a single-limb transition between holds."""

    limb: Limb
    source_hold: str
    target_hold: str
    source_contact_configuration: dict[Limb, str] | None = None
    target_contact_configuration: dict[Limb, str] | None = None
    steps: int = 210
    kp: float | None = None  # Nm/rad; None selects physical joint-class impedance.
    kd: float | None = None  # Nm*s/rad, independent of motor gear/capability.
    max_attach_distance: float = CAPTURE_DISTANCE
    check_grip: bool = True
    settle_steps_after: int = 40


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of a single-limb transition."""

    success: bool
    status: TransitionStatus
    limb: Limb
    source_hold: str
    target_hold: str
    phases_traversed: tuple[str, ...]
    initial_state: StateSummary
    final_state: StateSummary
    final_contact_configuration: dict[Limb, str]
    steps: int
    time: float
    duration: float
    displacement: float = 0.0
    reason: str = ""
    released: bool = False
    target_captured: bool = False
    eligibility_detected: bool = False
    clock_discontinuity: bool = False


@dataclass(frozen=True)
class TransitionObservation:
    """Read-only execution event; RUNNING is never a successful outcome."""

    request: TransitionRequest
    phase: TransitionPhase | None
    steps: int
    state: StateSummary
    result: TransitionResult | None = None
    move_index: int = 0
    total_moves: int = 1

    @property
    def status(self) -> TransitionStatus:
        return self.result.status if self.result is not None else TransitionStatus.RUNNING


@dataclass(frozen=True)
class TransitionSequenceResult:
    """Structured result of executing a multi-move sequential transition chain without reset."""

    success: bool
    completed_moves: int
    failed_move_index: int | None
    transition_results: tuple[TransitionResult, ...]
    final_contact_configuration: dict[Limb, str]
    final_state_summary: StateSummary
    total_steps: int
    total_time: float


@dataclass(frozen=True)
class ThreePointSupportResult:
    """Outcome of deterministic 3-point wall support test with one released limb."""

    steps: int
    time: float
    finite: bool
    supported: bool
    released_limb: Limb
    supporting_attached: dict[Limb, bool]
    supporting_loads: dict[Limb, float]
    min_root_z: float
    final_root_z: float


@dataclass(frozen=True)
class SingleLimbTransitionResult:
    """Backward-compatible outcome of deterministic single-limb reach-and-reattach transition."""

    steps: int
    time: float
    finite: bool
    supported: bool
    limb: Limb
    from_region_id: str
    to_region_id: str
    reattached: bool
    eligibility_detected: bool
    final_root_z: float
    phase_history: tuple[str, ...]
    attachment_loads: dict[Limb, float]
    success: bool
    status: TransitionStatus
    duration: float


def _supported_hold_geometry(foot, hold_id):
    """Scene labels cannot turn the native floor geom into a named hold."""
    surface = f"geom_{hold_id}"
    matching = [c for c in foot.contacts if c.surface_geom == surface and c.admissible]
    return (hold_id is not None and surface in foot.support_surfaces
            and sum(c.normal_force for c in matching) > MIN_FOOT_LOAD
            and all(c.tangential_speed <= MAX_SLIP_SPEED for c in matching))


def get_state_summary(
    model: Any,
    data: Any,
    manager: GraspManager,
) -> StateSummary:
    """Extract a physical state and load summary snapshot from current MjData."""
    manager.require_session(model, data, manager.scene)
    attachments = manager.active_attachments()
    contact_state = manager.contact_snapshot()
    contacts = contact_state.configuration
    root_pos = (float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]))
    root_vel = (float(data.qvel[0]), float(data.qvel[1]), float(data.qvel[2]))
    qvel_norm = float(np.linalg.norm(data.qvel))
    values = [data.time, *data.qpos, *data.qvel, *data.ctrl, *data.qacc, *data.qacc_warmstart]
    finite = all(math.isfinite(float(v)) for v in values)
    loads = {limb: state.load for limb, state in contact_state.hands.items() if state.active}
    return StateSummary(
        time=float(data.time),
        root_pos=root_pos,
        root_vel=root_vel,
        qvel_norm=qvel_norm,
        finite=finite,
        contact_configuration=dict(contacts),
        attachment_loads=loads,
        qpos=tuple(float(v) for v in data.qpos),
        qvel=tuple(float(v) for v in data.qvel),
        eq_active=tuple(bool(v) for v in data.eq_active),
        ctrl=tuple(float(v) for v in data.ctrl),
        qacc_warmstart=tuple(float(v) for v in data.qacc_warmstart),
        attachment_constraints={limb: att.eq_name for limb, att in attachments.items()},
        hand_states=contact_state.hands,
        foot_states=contact_state.feet,
        contact_mode=contact_state.mode,
        capture_events=tuple(copy.deepcopy(manager.capture_events)),
        release_events=tuple(copy.deepcopy(manager.releases)),
    )


def check_stabilization_readiness(
    model: Any,
    data: Any,
    manager: GraspManager,
    min_root_z: float = 0.85,
    min_supports: int = 3,
    v_lin_thresh: float = 0.60,
    v_ang_thresh: float = 3.50,
    qvel_thresh: float = 4.00,
    *,
    tracker: ReadinessTracker | None = None,
) -> tuple[bool, str]:
    """Query owned sustained history; legacy thresholds are debug-mode only.

    A physical query never records an endpoint or relaxes the Stage3 limits.
    IDEALIZED_DEBUG retains NONPHYSICAL instantaneous software regression checks.
    """
    manager.require_session(model, data, manager.scene)
    if manager.mode == ContactMode.PHYSICAL:
        if tracker is None:
            return False, "Sustained physical readiness requires authoritative endpoint history"
        if tracker.model is not model or tracker.data is not data or tracker.manager is not manager:
            raise AttachmentStateError("Readiness tracker is bound to a different session")
        return tracker.check()
    values = list(data.qpos) + list(data.qvel)
    if not all(math.isfinite(float(v)) for v in values):
        return False, "Non-finite values detected in simulation state"

    curr_z = float(data.qpos[2])
    if curr_z < min_root_z:
        return False, f"Root height {curr_z:.3f}m dropped below minimum threshold {min_root_z:.3f}m"

    active_supports = len(manager.contact_snapshot().supporting_limbs)
    if active_supports < min_supports:
        return False, f"Active support count {active_supports} is less than minimum {min_supports}"

    v_lin = float(np.linalg.norm(data.qvel[:3]))
    if v_lin > v_lin_thresh:
        return False, f"Root linear velocity {v_lin:.3f} m/s exceeds threshold {v_lin_thresh:.3f} m/s"

    v_ang = float(np.linalg.norm(data.qvel[3:6]))
    if v_ang > v_ang_thresh:
        return False, f"Root angular velocity {v_ang:.3f} rad/s exceeds threshold {v_ang_thresh:.3f} rad/s"

    joint_qvel = data.qvel[6:]
    mean_joint_vel = float(np.linalg.norm(joint_qvel) / math.sqrt(max(1, len(joint_qvel))))
    if mean_joint_vel > qvel_thresh:
        return False, f"Mean joint velocity {mean_joint_vel:.3f} rad/s exceeds threshold {qvel_thresh:.3f} rad/s"

    return True, "NONPHYSICAL idealized_debug instantaneous readiness (software regression only)"


def simulate_three_point_support(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    released_limb: Limb = Limb.RIGHT_HAND,
    steps: int = 300,
    settle_steps: int = 40,
    check_grip: bool = True,
    kp: float | None = None,
    kd: float | None = None,
    manager: GraspManager | None = None,
) -> ThreePointSupportResult:
    """Simulate and validate deterministic 3-point support under free-root physics."""
    mujoco = _import_mujoco()
    if manager is None:
        raise AttachmentStateError("Support execution requires an explicitly initialized manager")
    gm = manager
    gm.require_session(model, data, scene)
    if gm.mode == ContactMode.PHYSICAL and profile != gm.profile:
        raise AttachmentStateError("Execution profile differs from the bound contact profile")

    contacts = gm.contact_configuration()
    measured = gm.contact_snapshot()
    if gm.mode == ContactMode.PHYSICAL and any(limb in contacts and not _supported_hold_geometry(measured.feet[limb], contacts[limb])
                                              for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)):
        raise AttachmentStateError("Support references require native HOLD surfaces, not floor/wall contacts")
    hands = {limb: hold for limb, hold in contacts.items() if limb.is_hand}
    feet = {limb: contacts.get(limb, scene.start_configuration.get(limb))
            for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)}
    if any(hold is None for hold in feet.values()):
        raise AttachmentStateError("Support reference requires explicit foot-surface intent")
    # References use declared intent; only the native sensor establishes support.
    retarget_4pt = solve_retargeted_stance(model, scene, profile,
                                          spec=StanceSpecification(hand_holds=hands, foot_holds=feet))

    # Solve 3-point supporting posture with lateral shift over supporting limbs
    lateral_shift = -0.16 if released_limb == Limb.RIGHT_HAND else 0.16
    torso_yaw = -0.08 if released_limb == Limb.RIGHT_HAND else 0.08
    spec_3pt = StanceSpecification(
        pelvis_lateral_bias=lateral_shift,
        torso_orientation=(torso_yaw, -0.10, -0.05 if released_limb == Limb.RIGHT_HAND else 0.05),
        free_limbs=(released_limb,),
        hand_holds=hands,
        foot_holds=feet,
    )
    retarget_3pt = solve_retargeted_stance(model, scene, profile, spec=spec_3pt)

    min_root_z = float(data.qpos[2])
    actual_steps = 0
    supporting_limbs = [Limb.LEFT_HAND] if released_limb == Limb.RIGHT_HAND else [Limb.RIGHT_HAND]

    shift_steps = 40
    shift_start = settle_steps
    release_step = shift_start + shift_steps

    for step_idx in range(steps):
        if step_idx < shift_start:
            current_targets = retarget_4pt.target_pose
        elif step_idx < release_step:
            alpha = (step_idx - shift_start) / float(shift_steps)
            current_targets = {
                k: (1.0 - alpha) * retarget_4pt.target_pose[k] + alpha * retarget_3pt.target_pose[k]
                for k in retarget_4pt.target_pose
            }
        else:
            if step_idx == release_step:
                gm.detach(released_limb)
            current_targets = retarget_3pt.target_pose

        compute_pose_control(model, data, target_pose=current_targets, kp=kp, kd=kd)
        if gm.mode == ContactMode.PHYSICAL:
            decisions = gm.evaluate_and_update(profile)
            if any(not decision.maintain for decision in decisions.values()):
                break
        mujoco.mj_step(model, data)
        actual_steps += 1

        curr_z = float(data.qpos[2])
        if curr_z < min_root_z:
            min_root_z = curr_z

        if gm.mode == ContactMode.PHYSICAL or check_grip:
            decisions = gm.evaluate_and_update(profile, applied_data=data)
            if any(not decision.maintain for decision in decisions.values()):
                break

    values = list(data.qpos) + list(data.qvel)
    finite = all(math.isfinite(float(v)) for v in values)
    supported = float(data.qpos[2]) > 0.8
    if gm.mode == ContactMode.PHYSICAL:
        supported = supported and len(gm.contact_snapshot().supporting_limbs) >= 3

    supporting_attached = {limb: gm.is_attached(limb) for limb in supporting_limbs}
    supporting_loads = {limb: gm.get_grasp_load(limb) for limb in supporting_limbs}

    return ThreePointSupportResult(
        steps=actual_steps,
        time=float(data.time),
        finite=finite,
        supported=supported,
        released_limb=released_limb,
        supporting_attached=supporting_attached,
        supporting_loads=supporting_loads,
        min_root_z=min_root_z,
        final_root_z=float(data.qpos[2]),
    )


def execute_transition(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    request: TransitionRequest,
    manager: GraspManager | None = None,
    current_contacts: dict[Limb, str] | None = None,
    frame_callback: Any = None,
) -> TransitionResult:
    """Continue an explicitly initialized episode; never reset or reconstruct live pose.

    Supports bilateral hands (LEFT_HAND, RIGHT_HAND) and feet (LEFT_FOOT, RIGHT_FOOT) through
    continuous 7-stage cadence:
      1. INITIAL_STANCE: Settle in current multi-point support stance.
      2. PRE_SHIFT: Shift pelvis, torso, and center of mass toward supporting limbs, reducing load on moving limb.
      3. RELEASE_LIMB: Detach target limb via GraspManager.detach() event.
      4. SUPPORT_PHASE: Maintain 3-point free-root support against gravity while moving limb assumes clearance posture.
      5. REACH_PHASE: Coordinated whole-body reach / stepping toward target hold.
      6. ATTACH_PHASE: Move limb into target position, evaluate can_attach() eligibility, and attach.
      7. STABILIZED_STANCE: Settle into stabilized stance with updated contact configuration.
    """
    mujoco = _import_mujoco()

    if manager is None:
        raise AttachmentStateError("Transition execution requires an explicitly initialized manager")
    gm = manager
    gm.require_session(model, data, scene)
    readiness = ReadinessTracker(model, data, gm) if gm.mode == ContactMode.PHYSICAL else None
    contacts = gm.contact_configuration()
    init_state = get_state_summary(model, data, gm)
    phase_history: list[str] = []
    actual_steps = 0
    elapsed_duration = 0.0
    clock_discontinuity = False
    released = False
    target_captured = False
    eligibility_detected = False
    phase: TransitionPhase | None = None
    displacement = 0.0

    def observe(result: TransitionResult | None = None) -> None:
        if frame_callback is not None:
            # Native rendering/viewer sync may write derived data or model options.
            # Observers receive detached copies, including mutable result mappings.
            observed_model = copy.copy(model)
            observed_data = mujoco.MjData(observed_model)
            mujoco.mj_copyData(observed_data, observed_model, data)
            observed_manager = GraspManager(observed_model, observed_data, copy.deepcopy(scene), profile=copy.deepcopy(gm.profile))
            observed_manager.synchronize_from_live()
            observed_manager.capture_events = copy.deepcopy(gm.capture_events)
            observed_manager.releases = copy.deepcopy(gm.releases)
            observed_manager.last_capture_failure = copy.deepcopy(gm.last_capture_failure)
            frame_callback(TransitionObservation(copy.deepcopy(request), phase, actual_steps,
                                                  get_state_summary(model, data, gm), copy.deepcopy(result)),
                           observed_data, observed_manager)

    def finish(status: TransitionStatus, reason: str) -> TransitionResult:
        final = get_state_summary(model, data, gm)
        if status == TransitionStatus.SUCCESS and not final.finite:
            status = TransitionStatus.NONFINITE_STATE
            reason = "Final simulation state is non-finite"
        result = TransitionResult(
            success=status == TransitionStatus.SUCCESS, status=status,
            limb=request.limb, source_hold=request.source_hold, target_hold=request.target_hold,
            phases_traversed=tuple(phase_history), initial_state=init_state, final_state=final,
            final_contact_configuration=dict(final.contact_configuration), steps=actual_steps,
            time=final.time, duration=elapsed_duration, displacement=displacement,
            reason=reason, released=released, target_captured=target_captured,
            eligibility_detected=eligibility_detected,
            clock_discontinuity=clock_discontinuity,
        )
        observe(result)
        return result

    # Caller configurations are assertions only. No validation below writes MjData.
    if (not isinstance(request.limb, Limb)
            or not isinstance(request.source_hold, str) or not isinstance(request.target_hold, str)
            or type(request.steps) is not int or request.steps <= 0
            or type(request.settle_steps_after) is not int or request.settle_steps_after < 0
            or type(request.check_grip) is not bool
            or any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0)
                   for v in (request.kp, request.kd))
            or not isinstance(request.max_attach_distance, (int, float))
            or not math.isfinite(request.max_attach_distance) or request.max_attach_distance < 0
            or request.max_attach_distance == 0
            or gm.mode == ContactMode.PHYSICAL and (request.max_attach_distance > CAPTURE_DISTANCE or not request.check_grip)
            or request.source_hold == request.target_hold):
        return finish(TransitionStatus.INVALID_REQUEST, "Invalid limb, holds, budgets, gains, or capture distance")
    known_hold_ids = {r.id for r in scene.contact_regions}
    if request.source_hold not in known_hold_ids:
        return finish(TransitionStatus.INVALID_REQUEST, f"Unknown source hold {request.source_hold!r}")
    if request.target_hold not in known_hold_ids:
        return finish(TransitionStatus.INELIGIBLE_TARGET, f"Unknown target hold {request.target_hold!r}")
    if contacts.get(request.limb) != request.source_hold:
        return finish(TransitionStatus.SOURCE_NOT_ATTACHED, "Requested source is not the actual active attachment")
    if (request.limb.is_foot and gm.mode == ContactMode.PHYSICAL
            and not _supported_hold_geometry(gm.contact_snapshot().feet[request.limb], request.source_hold)):
        return finish(TransitionStatus.SOURCE_NOT_ATTACHED, "Native source shoe/hold geometry does not match the requested hold")
    expected_contacts = dict(contacts)
    expected_contacts[request.limb] = request.target_hold
    if (current_contacts is not None and current_contacts != contacts
            or request.source_contact_configuration is not None and request.source_contact_configuration != contacts
            or request.target_contact_configuration is not None and request.target_contact_configuration != expected_contacts):
        return finish(TransitionStatus.INVALID_REQUEST, "Contact configuration assertion disagrees with live state or requested move")
    target_region = scene.region(request.target_hold)
    expected_affordance = Affordance.GRASP if request.limb.is_hand else Affordance.STEP
    if expected_affordance not in target_region.affordances:
        return finish(TransitionStatus.INELIGIBLE_TARGET, "Target lacks the required limb affordance")
    if profile != gm.profile:
        return finish(TransitionStatus.INVALID_REQUEST, "Execution profile differs from the bound contact profile")
    equality_move = request.limb.is_hand or gm.mode == ContactMode.IDEALIZED_DEBUG
    target_eq, source_eq = None, None
    if equality_move:
        target_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                     f"grasp_{request.limb.value.lower()}_{request.target_hold}")
        if target_eq < 0:
            return finish(TransitionStatus.INELIGIBLE_TARGET, "Target attachment is not registered in the compiled model")
        gm.registered_attachment(request.limb, target_region)
        source_eq = gm.active_attachments()[request.limb].eq_id
    else:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{target_region.id}") < 0:
            return finish(TransitionStatus.INELIGIBLE_TARGET, "Target has no physical support geometry")
    if set(contacts) != set(Limb):
        return finish(TransitionStatus.SUPPORT_FAILURE, "Current scripted references require two valid hands and two measured feet")
    if any(contacts[limb] not in known_hold_ids for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)):
        return finish(TransitionStatus.SUPPORT_FAILURE, "Scripted references do not support floor/wall starting surfaces")
    if (gm.mode == ContactMode.PHYSICAL
            and any(not _supported_hold_geometry(gm.contact_snapshot().feet[limb], contacts[limb])
                    for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT))):
        return finish(TransitionStatus.SUPPORT_FAILURE, "Scripted references require the native named-hold geom")
    if not init_state.finite:
        return finish(TransitionStatus.NONFINITE_STATE, "Initial simulation state is non-finite")
    observe()

    # All pre-conditions passed! Now compute retargeted stances
    source_region = scene.region(request.source_hold)
    displacement = float(np.linalg.norm(np.array(target_region.position) - np.array(source_region.position)))

    d_wall = 0.48 if profile.name == "compact_strong" else 0.52
    h_holds = {Limb.LEFT_HAND: contacts[Limb.LEFT_HAND], Limb.RIGHT_HAND: contacts[Limb.RIGHT_HAND]}
    f_holds = {Limb.LEFT_FOOT: contacts[Limb.LEFT_FOOT], Limb.RIGHT_FOOT: contacts[Limb.RIGHT_FOOT]}

    # Check foot level for natural pitch bias
    p_bias = -0.10 if (contacts[Limb.LEFT_FOOT] == "H6" or contacts[Limb.RIGHT_FOOT] == "H6") else -0.24
    spec_init = StanceSpecification(
        pelvis_wall_distance=d_wall,
        hand_holds=h_holds.copy(),
        foot_holds=f_holds.copy(),
        torso_orientation=(0.0, p_bias, 0.0),
    )
    ret_init = solve_retargeted_stance(model, scene, profile, spec=spec_init)

    # Pre-shift posture: shift weight AWAY from releasing limb
    if request.limb == Limb.RIGHT_HAND:
        lat_shift, yaw, roll = -0.16, -0.08, -0.05
    elif request.limb == Limb.LEFT_HAND:
        lat_shift, yaw, roll = 0.14, 0.08, 0.05
    elif request.limb == Limb.LEFT_FOOT:
        lat_shift, yaw, roll = 0.12, 0.04, 0.02
    else:
        lat_shift, yaw, roll = -0.12, -0.04, -0.02

    spec_preshift = StanceSpecification(
        pelvis_wall_distance=d_wall,
        pelvis_lateral_bias=lat_shift,
        torso_orientation=(yaw, p_bias, roll),
        hand_holds=h_holds.copy(),
        foot_holds=f_holds.copy(),
    )
    ret_preshift = solve_retargeted_stance(model, scene, profile, spec=spec_preshift)

    # Support posture: moving limb assumes cocked / clearance position
    if request.limb.is_hand:
        support_tgt = (0.24, -0.12, 1.48) if request.limb == Limb.RIGHT_HAND else (-0.24, -0.12, 1.48)
    else:
        spos = scene.region(request.source_hold).position
        support_tgt = (spos[0] + (0.05 if request.limb == Limb.LEFT_FOOT else -0.05), -0.12, spos[2] + 0.06)

    spec_support = StanceSpecification(
        pelvis_wall_distance=d_wall,
        pelvis_lateral_bias=lat_shift,
        torso_orientation=(yaw, p_bias, roll),
        hand_holds=h_holds.copy(),
        foot_holds=f_holds.copy(),
        free_limbs=(request.limb,),
        custom_limb_targets={request.limb: support_tgt},
    )
    ret_support = solve_retargeted_stance(model, scene, profile, spec=spec_support)

    # Final stance: target configuration
    target_h = h_holds.copy()
    target_f = f_holds.copy()
    if request.limb.is_hand:
        target_h[request.limb] = request.target_hold
    else:
        target_f[request.limb] = request.target_hold

    target_p_bias = -0.10 if (target_f[Limb.LEFT_FOOT] == "H6" or target_f[Limb.RIGHT_FOOT] == "H6") else -0.24
    spec_final = StanceSpecification(
        pelvis_wall_distance=d_wall,
        hand_holds=target_h,
        foot_holds=target_f,
        torso_orientation=(0.0, target_p_bias, 0.0),
    )
    ret_final = solve_retargeted_stance(model, scene, profile, spec=spec_final)

    reach_target_pose = ret_final.target_pose.copy()
    if request.limb == Limb.RIGHT_HAND and request.target_hold == "H5":
        reach_target_pose["right_shoulder_pitch"] = max(ret_final.target_pose["right_shoulder_pitch"], 1.55)
        reach_target_pose["right_elbow"] = 0.50

    # Cadence budgets
    steps = request.steps
    s_initial_end = min(24, max(12, int(steps * 0.12)))
    s_preshift_end = s_initial_end + min(32, max(16, int(steps * 0.16)))
    s_release = s_preshift_end
    s_support_end = s_release + min(20, max(10, int(steps * 0.10)))
    s_reach_end = s_support_end + min(40, max(20, int(steps * 0.20)))
    s_attach_end = s_reach_end + min(40, max(20, int(steps * 0.20)))

    reattached = False
    status = TransitionStatus.RUNNING
    reason = "Transition is running"
    limb_prefix = "right_" if request.limb in (Limb.RIGHT_HAND, Limb.RIGHT_FOOT) else "left_"
    numerical_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                          mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
    warning_counts = tuple(int(data.warning[w].number) for w in numerical_warnings)

    def integrate() -> bool:
        nonlocal actual_steps, elapsed_duration, clock_discontinuity, status, reason
        before = float(data.time)
        timestep = float(model.opt.timestep)
        if gm.mode == ContactMode.PHYSICAL:
            decisions = gm.evaluate_and_update(profile)
            if any(not decision.maintain for decision in decisions.values()):
                readiness.invalidate("Hand grasp failed before integration")
                status = TransitionStatus.GRIP_FAILURE
                reason = "Hand grasp capacity/geometry failed before integration"
                return False
        mujoco.mj_step(model, data)
        actual_steps += 1
        delta = float(data.time) - before
        continuous_clock = math.isclose(delta, timestep, rel_tol=1e-8, abs_tol=1e-12)
        # Recovery may reset data.time. Preserve completed integration cadence as
        # work duration while keeping the actual reset clock in the state snapshot.
        elapsed_duration += delta if continuous_clock else timestep
        clock_discontinuity = clock_discontinuity or not continuous_clock
        if readiness is not None:
            readiness.sample_after_step(before)
        if (tuple(int(data.warning[w].number) for w in numerical_warnings) != warning_counts
                or not continuous_clock):
            # MuJoCo numerical recovery may reset eq_active. Adopt that actual state
            # explicitly before reporting failure; never restore old attachments.
            gm.synchronize_from_live()
            status = TransitionStatus.NONFINITE_STATE
            reason = "Non-finite state or numerical recovery detected during integration"
            return False
        if not get_state_summary(model, data, gm).finite:
            status = TransitionStatus.NONFINITE_STATE
            reason = "Non-finite simulation state after integration"
            return False
        if gm.mode == ContactMode.PHYSICAL:
            decisions = gm.evaluate_and_update(profile, applied_data=data)
            if any(not decision.maintain for decision in decisions.values()):
                status = TransitionStatus.GRIP_FAILURE
                reason = "Hand grasp capacity/geometry failed after integration"
                return False
        if data.qpos[2] < 0.70:
            status = TransitionStatus.SUPPORT_FAILURE
            reason = "Root collapsed below the existing 0.70m threshold"
            return False
        return True

    for step_idx in range(steps):
        if not get_state_summary(model, data, gm).finite:
            status = TransitionStatus.NONFINITE_STATE
            reason = "Non-finite state before integration"
            break
        ctrl_kp, ctrl_kd = request.kp, request.kd
        if step_idx < s_initial_end:
            phase = TransitionPhase.INITIAL_STANCE
            targets = ret_init.target_pose
        elif step_idx < s_preshift_end:
            phase = TransitionPhase.PRE_SHIFT
            alpha = (step_idx - s_initial_end) / float(s_preshift_end - s_initial_end)
            targets = {k: (1 - alpha) * ret_init.target_pose[k] + alpha * ret_preshift.target_pose[k] for k in ret_init.target_pose}
        elif step_idx == s_release:
            phase = TransitionPhase.RELEASE_LIMB
            if equality_move:
                released = gm.detach(request.limb)
                if not released or data.eq_active[source_eq]:
                    raise AttachmentStateError("Requested source attachment was not released")
            targets = ret_support.target_pose
        elif step_idx < s_support_end:
            phase = TransitionPhase.SUPPORT_PHASE
            targets = ret_support.target_pose
            ctrl_kp, ctrl_kd = (45.0 if request.limb.is_hand else 35.0), (2.0 if request.limb.is_hand else 3.0)
        elif step_idx < s_reach_end:
            phase = TransitionPhase.REACH_PHASE
            alpha = (step_idx - s_support_end) / float(s_reach_end - s_support_end)
            targets = {k: (1 - alpha) * ret_support.target_pose[k] + alpha * reach_target_pose[k] for k in reach_target_pose}
            ctrl_kp, ctrl_kd = (45.0 if request.limb.is_hand else 35.0), (2.0 if request.limb.is_hand else 3.0)
        elif step_idx < s_attach_end:
            phase = TransitionPhase.ATTACH_PHASE
            alpha = (step_idx - s_reach_end) / float(s_attach_end - s_reach_end)
            targets = {k: (1 - alpha) * reach_target_pose[k] + alpha * ret_final.target_pose[k] for k in reach_target_pose}
            ctrl_kp, ctrl_kd = (45.0 if request.limb.is_hand else 35.0), (2.0 if request.limb.is_hand else 3.0)

            if equality_move:
                can_att = gm.can_attach(request.limb, target_region, max_distance=request.max_attach_distance)
                eligibility_detected = eligibility_detected or can_att
                if can_att and not reattached:
                    reattached = gm.attach(request.limb, target_region)
                    target_captured = target_captured or reattached
        else:
            phase = TransitionPhase.STABILIZED_STANCE
            targets = ret_final.target_pose
            if equality_move and not reattached:
                can_att = gm.can_attach(request.limb, target_region, max_distance=request.max_attach_distance)
                eligibility_detected = eligibility_detected or can_att
                if can_att:
                    reattached = gm.attach(request.limb, target_region)
                    target_captured = target_captured or reattached

        if not phase_history or phase_history[-1] != phase.value:
            phase_history.append(phase.value)

        compute_pose_control(model, data, target_pose=targets, kp=ctrl_kp, kd=ctrl_kd)

        # Relax releasing limb actuators during pre-shift
        if phase == TransitionPhase.PRE_SHIFT:
            alpha = (step_idx - s_initial_end) / float(s_preshift_end - s_initial_end)
            if alpha > 0.5:
                scale = 1.0 - (alpha - 0.5) / 0.5 * 0.8
                for aid in range(model.nu):
                    jid = int(model.actuator_trnid[aid, 0])
                    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and limb_prefix in jname and ("shoulder" in jname or "elbow" in jname or "wrist" in jname or "knee" in jname or "ankle" in jname):
                        data.ctrl[aid] *= scale

        if not integrate():
            break
        if not equality_move and step_idx >= s_release:
            foot = gm.contact_snapshot().feet[request.limb]
            released = released or not any(contact.surface_geom == f"geom_{request.source_hold}" for contact in foot.contacts)
            acquired = released and _supported_hold_geometry(foot, request.target_hold) and foot.supporting and not foot.slipping
            eligibility_detected = eligibility_detected or acquired
            target_captured = target_captured or acquired

        # Failure check: Breakable grip
        if gm.mode == ContactMode.IDEALIZED_DEBUG and request.check_grip and step_idx >= s_release:
            decisions = gm.evaluate_and_update(profile)
            if any(not d.maintain for d in decisions.values()):
                status = TransitionStatus.GRIP_FAILURE
                reason = "Grip failure: supporting limb grip capacity exceeded"
                break

        observe()

    # Post-transition verification & stabilization
    if status == TransitionStatus.RUNNING:
        if tuple(phase_history) != tuple(p.value for p in TransitionPhase):
            status = TransitionStatus.INCOMPLETE
            reason = "Execution budget ended before the full phase progression completed"
        elif (not released or not target_captured
              or gm.contact_configuration().get(request.limb) != request.target_hold
              or equality_move and (not data.eq_active[target_eq] or data.eq_active[source_eq])):
            status = TransitionStatus.ATTACH_FAILURE
            reason = f"Requested target {request.target_hold!r} was not acquired after releasing the source"
        else:
            # Settle period
            for settle_idx in range(request.settle_steps_after):
                if not get_state_summary(model, data, gm).finite:
                    status = TransitionStatus.NONFINITE_STATE
                    reason = "Non-finite state before settling integration"
                    break
                compute_pose_control(model, data, target_pose=ret_final.target_pose, kp=request.kp, kd=request.kd)
                if not integrate():
                    break
                observe()

            # Check stabilization criterion
            if status == TransitionStatus.RUNNING:
                foot_source_contact = (not equality_move and any(c.surface_geom == f"geom_{request.source_hold}"
                                      for c in gm.contact_snapshot().feet[request.limb].contacts))
                feet_valid = (gm.mode == ContactMode.IDEALIZED_DEBUG
                              or all(_supported_hold_geometry(gm.contact_snapshot().feet[limb], expected_contacts[limb])
                                     for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT)))
                if (gm.contact_configuration() != expected_contacts or foot_source_contact
                        or not feet_valid
                        or equality_move and (not data.eq_active[target_eq] or data.eq_active[source_eq])):
                    status = TransitionStatus.ATTACH_FAILURE
                    reason = "Actual final attachments differ from the requested transition"
                else:
                    is_stable, stab_reason = check_stabilization_readiness(model, data, gm, tracker=readiness)
                    status = TransitionStatus.SUCCESS if is_stable else TransitionStatus.UNSTABLE_FINAL_STATE
                    reason = (f"Source released, target captured, phase progression completed: {stab_reason}"
                              if is_stable else f"Readiness rejected final state: {stab_reason}")

    return finish(status, reason)


def execute_transition_sequence(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    requests: list[TransitionRequest] | tuple[TransitionRequest, ...],
    manager: GraspManager | None = None,
    initial_contacts: dict[Limb, str] | None = None,
    frame_callback: Any = None,
) -> TransitionSequenceResult:
    """Execute an ordered sequence of transitions strictly from the accumulated physical state of each previous move.

    Critical invariant: The simulation MUST NOT reset between moves. Simulation time, qpos, qvel,
    active equality constraints, and contact configurations carry forward continuously.
    If any transition fails, execution halts immediately and returns failure information.
    """
    if manager is None:
        raise AttachmentStateError("Sequence execution requires an explicitly initialized manager")
    gm = manager
    gm.require_session(model, data, scene)
    if not requests:
        raise ValueError("Sequence execution requires at least one transition request")
    if initial_contacts is not None and initial_contacts != gm.contact_configuration():
        raise AttachmentStateError("Initial sequence contact assertion disagrees with live attachments")

    results: list[TransitionResult] = []
    total_steps = 0

    for idx, req in enumerate(requests):
        def on_frame(observation: TransitionObservation, d: Any, manager: GraspManager) -> None:
            if frame_callback is not None:
                frame_callback(replace(observation, move_index=idx, total_moves=len(requests)), d, manager)

        res = execute_transition(
            model=model,
            data=data,
            scene=scene,
            profile=profile,
            request=req,
            manager=gm,
            frame_callback=on_frame if frame_callback is not None else None,
        )
        results.append(res)
        total_steps += res.steps

        if not res.success:
            # Halt immediately on failure
            return TransitionSequenceResult(
                success=False,
                completed_moves=idx,
                failed_move_index=idx,
                transition_results=tuple(results),
                final_contact_configuration=dict(res.final_contact_configuration),
                final_state_summary=res.final_state,
                total_steps=total_steps,
                total_time=sum(r.duration for r in results),
            )

    final_state = results[-1].final_state
    return TransitionSequenceResult(
        success=True,
        completed_moves=len(requests),
        failed_move_index=None,
        transition_results=tuple(results),
        final_contact_configuration=dict(final_state.contact_configuration),
        final_state_summary=final_state,
        total_steps=total_steps,
        total_time=sum(r.duration for r in results),
    )


def simulate_single_limb_reach(
    model: Any,
    data: Any,
    scene: BoulderScene,
    profile: ClimberProfile,
    limb: Limb = Limb.RIGHT_HAND,
    from_region_id: str = "H4",
    to_region_id: str = "H5",
    steps: int = 210,
    kp: float | None = None,
    kd: float | None = None,
    max_attach_distance: float = CAPTURE_DISTANCE,
    manager: GraspManager | None = None,
) -> SingleLimbTransitionResult:
    """Existing reach-result adapter; requires the same explicit episode lifecycle."""
    if manager is None:
        raise AttachmentStateError("Reach execution requires an explicitly initialized manager")
    gm = manager

    req = TransitionRequest(
        limb=limb,
        source_hold=from_region_id,
        target_hold=to_region_id,
        steps=steps,
        kp=kp,
        kd=kd,
        max_attach_distance=max_attach_distance,
        settle_steps_after=0,
    )

    res = execute_transition(
        model=model,
        data=data,
        scene=scene,
        profile=profile,
        request=req,
        manager=gm,
    )

    return SingleLimbTransitionResult(
        steps=res.steps,
        time=res.time,
        finite=res.final_state.finite,
        supported=(res.final_state.root_pos[2] > 0.8
                   and (gm.mode == ContactMode.IDEALIZED_DEBUG
                        or len(gm.contact_snapshot().supporting_limbs) >= 3)),
        limb=limb,
        from_region_id=from_region_id,
        to_region_id=to_region_id,
        reattached=(res.target_captured and res.final_contact_configuration.get(limb) == to_region_id),
        eligibility_detected=res.eligibility_detected,
        final_root_z=res.final_state.root_pos[2],
        phase_history=res.phases_traversed,
        attachment_loads=res.final_state.attachment_loads,
        success=res.success,
        status=res.status,
        duration=res.duration,
    )
