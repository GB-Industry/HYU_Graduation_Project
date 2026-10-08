"""Declared single-limb transfers and sequential execution, not route planning.

Only the fixture runner initializes an episode. Execution retains the bound live
session and delegates admission, control, capture and readiness to the maintained
primitives. Their existing safety limits are fixed, not tuning parameters.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real
from types import MappingProxyType

from . import foot_transfer, single_hand
from .contact import CAPTURE_DISTANCE, CAPTURE_ORIENTATION, CAPTURE_SPEED
from .contact_geometry import ContactMode
from .contact_ik import solve_contact_pose
from .locomotion import get_state_summary
from .schema import Affordance, Limb, SourceType
from .static_control import execute_static_hold
from .static_state import _finite_state, _integration_state, initialize_static_reference
from .whole_body_motion import WholeBodyMotion


@dataclass(frozen=True)
class TransferRequest:
    limb: Limb
    source: str
    target: str
    source_contacts: Mapping[Limb, str]
    support_contacts: Mapping[Limb, str] | None = None
    primitive: str | None = None
    root_linear_max_m_s: float = .10
    root_angular_max_rad_s: float = .50
    joint_max_rad_s: float = 1.
    hand_acquisition_policy: str = "endpoint_settle"
    hand_reach_s: float = 4.
    whole_body: WholeBodyMotion | None = None
    foot_request: foot_transfer.FootRequest | None = None

    def __post_init__(self):
        # Leave malformed inputs for atomic execution-time rejection.
        for name in ("source_contacts", "support_contacts"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, MappingProxyType(dict(value)))


def _request_error(scene, request, dt):
    """Structural validation is also safe for future moves before integration."""
    if not isinstance(request, TransferRequest) or not isinstance(request.limb, Limb):
        return "INVALID_REQUEST", "Transfer requires a TransferRequest and an actual Limb"
    contacts = request.source_contacts
    if (not isinstance(contacts, Mapping) or set(contacts) != set(Limb)
            or any(not isinstance(limb, Limb) for limb in contacts)
            or any(not isinstance(hold, str) or not hold.strip() for hold in contacts.values())):
        return "INVALID_REQUEST", "source_contacts requires all four actual Limb keys and HOLD ids"
    if (any(not isinstance(v, str) or not v.strip() for v in (request.source, request.target))
            or contacts[request.limb] != request.source or request.source == request.target):
        return "INVALID_REQUEST", "Moving source must match source_contacts and differ from target"
    supports = {limb: hold for limb, hold in contacts.items() if limb != request.limb}
    if request.support_contacts is not None and (
            not isinstance(request.support_contacts, Mapping)
            or any(not isinstance(limb, Limb) for limb in request.support_contacts)
            or dict(request.support_contacts) != supports):
        return "INVALID_REQUEST", "support_contacts must exactly exclude the moving limb"
    bounds = (request.root_linear_max_m_s, request.root_angular_max_rad_s, request.joint_max_rad_s)
    if any(isinstance(v, bool) or not isinstance(v, Real) or v != limit or not math.isfinite(v)
           for v, limit in zip(bounds, (.10, .50, 1.))):
        return "INVALID_REQUEST", "Maintained primitive safety bounds are fixed at 0.10/0.50/1.0; no override"
    foot = request.foot_request
    if foot is not None:
        foot_error = "INVALID_REQUEST", "Explicit foot request requires matching foot intent and valid native parameters"
        if (not isinstance(foot, foot_transfer.FootRequest) or not request.limb.is_foot
                or not isinstance(foot.limb, Limb)
                or any(not isinstance(v, str) for v in (foot.source, foot.target))
                or (foot.limb, foot.source, foot.target) != (request.limb, request.source, request.target)):
            return foot_error
        try:
            durations = (foot.unload_s, foot.lift_s, foot.support_s, foot.reach_s,
                         foot.contact_timeout_s, foot.load_s, foot.settle_timeout_s)
            if (not math.isfinite(dt) or dt <= 0
                    or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
                           or not math.isfinite(intervals := v / dt)
                           or round(intervals) < 1 or abs(intervals - round(intervals)) > 1e-8 for v in durations)
                    or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
                            for v in (foot.lift_m, foot.clearance_m))
                    or isinstance(foot.airborne_pitch_rad, bool) or not isinstance(foot.airborne_pitch_rad, Real)
                    or not math.isfinite(foot.airborne_pitch_rad) or abs(foot.airborne_pitch_rad) > math.pi / 6):
                return foot_error
        except (OverflowError, TypeError, ValueError):
            return foot_error
    primitive = "hand" if request.limb.is_hand else "foot"
    if request.primitive is not None:
        if not isinstance(request.primitive, str):
            return "INVALID_REQUEST", "primitive must be a string or None"
        if request.primitive != primitive:
            return "UNSUPPORTED_PRIMITIVE", "Declared primitive is not supported for this moving limb"
    hand_error = "INVALID_REQUEST", "Hand policy/duration must use a maintained policy and positive native intervals"
    try:
        if (not isinstance(request.hand_acquisition_policy, str)
                or request.hand_acquisition_policy not in ("first_eligible", "endpoint_settle")
                or isinstance(request.hand_reach_s, bool) or not isinstance(request.hand_reach_s, Real)
                or not math.isfinite(dt) or dt <= 0
                or not math.isfinite(request.hand_reach_s) or request.hand_reach_s <= 0
                or not math.isfinite(intervals := request.hand_reach_s / dt)
                or abs(intervals - round(intervals)) > 1e-8):
            return hand_error
    except (OverflowError, TypeError, ValueError):
        return hand_error
    try:
        for limb, hold in (*contacts.items(), (request.limb, request.target)):
            region = scene.region(hold)
            affordance = Affordance.GRASP if limb.is_hand else Affordance.STEP
            if region.id != hold or region.source_type != SourceType.HOLD or affordance not in region.affordances:
                return "INVALID_REQUEST", "Contacts and target must be eligible canonical HOLD ids"
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        return "INVALID_REQUEST", str(error)
    return None


def execute_transfer(model, data, scene, profile, reference, manager, request, *, observer=None,
                     keep_samples=True) -> dict:
    """Execute one declared transfer without resetting or adopting live state.

    Bad generic requests are rejected without callbacks, commands or scratch
    source admission. Full reference/ROM and native support checks remain owned
    by the selected primitive. Session binding errors propagate as in primitives.
    """
    manager.require_session(model, data, scene)
    initial = get_state_summary(model, data, manager)
    initial_integration = tuple(float(v) for v in _integration_state(model, data))
    dt = float(model.opt.timestep)
    error = _request_error(scene, request, dt)
    limb = request.limb if isinstance(request, TransferRequest) else None
    contacts = request.source_contacts if isinstance(request, TransferRequest) else None
    supports = ({l: h for l, h in contacts.items() if l != limb}
                if isinstance(contacts, Mapping) and isinstance(limb, Limb) else {})
    if error is None and dict(contacts) != initial.contact_configuration:
        error = "INVALID_REQUEST", "source_contacts differs from the actual native four-contact source"
    if error is None and (manager.mode != ContactMode.PHYSICAL or manager.profile is not profile
                          or not _finite_state(model, data)
                          or data.qfrc_applied.any() or data.xfrc_applied.any()):
        error = "INITIALIZATION_FAILURE", "Invalid bound physical profile/full live state or undeclared forces"

    if error is not None:
        status, reason = error
        underlying = None
        result = {"success": False, "status": status, "reason": reason,
                  "initial_state": initial, "final_state": initial, "steps": 0, "duration_s": 0.,
                  "dt_s": dt, "initial_contacts": dict(initial.contact_configuration),
                  "new_contacts": dict(initial.contact_configuration), "released": False,
                  "final_reference": None, "phases": [], "events": [], "capture": None,
                  "readiness": {"ready": False, "duration": 0., "reason": reason},
                  "terminal_observation": None, "support_admission": None}
        if keep_samples:
            result["samples"] = []
    else:
        if limb.is_hand:
            primitive_request = single_hand.SingleHandRequest(
                limb, request.source, request.target, reach_s=request.hand_reach_s,
                acquisition_policy=request.hand_acquisition_policy)
            executor = single_hand.execute_single_hand
        else:
            primitive_request = (foot_transfer.FootRequest(limb, request.source, request.target)
                                 if request.foot_request is None else request.foot_request)
            executor = foot_transfer.execute_foot_transfer
        options = {"whole_body": request.whole_body} if request.whole_body is not None else {}
        underlying = executor(model, data, scene, profile, reference, manager, primitive_request,
                               observer=observer, keep_samples=keep_samples, **options)
        # Shallow wrapping preserves the exact immutable final reference.
        result = dict(underlying)
        actual = get_state_summary(model, data, manager)
        if actual != result["final_state"] or (result["success"] and result["final_reference"] is None):
            result.update(success=False, status="CONTROL_FAILURE",
                          reason="Primitive did not deliver an unchanged terminal state and admitted final reference",
                          final_state=actual, new_contacts=dict(actual.contact_configuration), final_reference=None)
            hinges = data.qvel[model.jnt_dofadr[model.actuator_trnid[:, 0]]]
            linear, angular = math.hypot(*actual.qvel[:3]), math.hypot(*actual.qvel[3:6])
            maximum = float(max(abs(v) for v in hinges))
            result["readiness"] = {**result["readiness"], "ready": False, "duration": 0.,
                                   "reason": result["reason"], "time": actual.time,
                                   "root_linear_speed": linear, "root_angular_speed": angular,
                                   "max_hinge_speed": maximum,
                                   "rms_hinge_speed": math.hypot(*hinges) / math.sqrt(len(hinges))}
            terminal = copy.deepcopy(result.get("terminal_observation") or {})
            terminal.update(terminal=True, status=result["status"], reason=result["reason"],
                            time_s=actual.time, elapsed_s=actual.time - initial.time,
                            steps=result["steps"], integrated_elapsed_s=result["duration_s"],
                            qpos=list(actual.qpos), qvel=list(actual.qvel), ctrl=list(actual.ctrl),
                            eq_active=list(actual.eq_active), qacc_warmstart=list(actual.qacc_warmstart),
                            root_pose=list(actual.qpos[:7]), root_linear_m_s=linear,
                            root_angular_rad_s=angular, joint_max_rad_s=maximum,
                            hands={l.value: asdict(s) for l, s in actual.hand_states.items()},
                            feet={l.value: asdict(s) for l, s in actual.foot_states.items()},
                            contacts={l.value: h for l, h in actual.contact_configuration.items()},
                            external_force_world_N=data.xfrc_applied.tolist(),
                            qfrc_applied=data.qfrc_applied.tolist(), readiness=dict(result["readiness"]),
                            pose_available=_finite_state(model, data))
            if terminal.get("command") is not None:
                command = terminal["command"]
                torque = data.ctrl * model.actuator_gear[:, 0]
                command.update(matches_last_request=all(v == old / gear for v, old, gear in zip(
                                   data.ctrl, command["commanded_Nm"], model.actuator_gear[:, 0])),
                               commanded_Nm=tuple(float(v) for v in torque),
                               utilization=tuple(float(abs(v) / limit) for v, limit in zip(torque, command["limits_Nm"])))
            # Task measurements from the inconsistent primitive snapshot are stale.
            for key in ("actual_hand_position_world_m", "actual_foot_position_world_m", "foot_distance_m",
                        "capture_measurement", "tracking_error_m", "source_contact"):
                if key in terminal:
                    terminal[key] = None
            result["terminal_observation"] = terminal

    events = result["events"]
    release = next((e["time_s"] for e in events if e.get("event") == "RELEASED"), None)
    reach = next((e["time_s"] for e in events if e.get("phase") == "REACH"), None)
    capture = result.get("capture")
    native_admission = result.get("support_admission") or {}
    result.update(
        initial_integration_state=initial_integration,
        final_integration_state=tuple(float(v) for v in _integration_state(model, data)),
        request=request if isinstance(request, TransferRequest) else None,
        primitive="hand" if isinstance(limb, Limb) and limb.is_hand else
                  "foot" if isinstance(limb, Limb) and limb.is_foot else None,
        moving_limb=limb, source=request.source if isinstance(request, TransferRequest) else None,
        target=request.target if isinstance(request, TransferRequest) else None,
        source_contacts=contacts, support_contacts=MappingProxyType(supports),
        final_contacts=dict(result["final_state"].contact_configuration), underlying_result=underlying,
        release_time_s=result.get("release_time_s", release), reach_start_time_s=reach,
        capture_time_s=capture["time_s"] if capture else None,
        foot_acquisition_time_s=result.get("acquisition_time_s"),
        capture_error_m=capture["gap_m"] if capture else None,
        capture_margin_m=capture["capture_margin_m"] if capture else None,
        readiness_time_s=result["readiness"].get("time") if result["success"] else None,
        admission={**native_admission, "contacts": MappingProxyType(supports),
                   "admitted": bool(native_admission.get("admitted", False)),
                   "source_reference_admitted": bool((result.get("terminal_observation") or {}).get(
                       "source_reference_admitted", False))},
        requirements={"motion_safety": {"root_linear_max_m_s": .10, "root_angular_max_rad_s": .50,
                                         "joint_max_rad_s": 1.},
                      "reference": {"speed_max_rad_s": .5, "acceleration_max_rad_s2": 2.,
                                    "admitted_static_only": True},
                      "hand_capture": {"distance_m": CAPTURE_DISTANCE, "orientation": CAPTURE_ORIENTATION,
                                       "speed_m_s": CAPTURE_SPEED},
                      "readiness": "Owned sustained Stage3 readiness; unchanged primitive criteria"})
    return result


def execute_transfer_sequence(model, data, scene, profile, requests, reference, manager, *, observer=None,
                              keep_samples=True) -> dict:
    """Execute a nonempty declared chain on one episode, stopping at first failure.

    Each next move uses precisely the previous success's final_reference. Contact
    identities are checked against native state just before each move. Observer
    metadata is added only to a row copy; primitive display data is already detached.
    """
    manager.require_session(model, data, scene)
    initial = get_state_summary(model, data, manager)
    initial_integration = tuple(float(v) for v in _integration_state(model, data))
    moves, trace, completed, failed = [], [dict(initial.contact_configuration)], 0, None
    status, reason = "INVALID_REQUEST", "Sequence requires a nonempty sequence of TransferRequest values"
    valid = isinstance(requests, Sequence) and not isinstance(requests, (str, bytes)) and len(requests) > 0
    if valid:
        requests = tuple(requests)
        for index, request in enumerate(requests):
            error = _request_error(scene, request, float(model.opt.timestep))
            if error is not None:
                status, reason = error
                failed, valid = index, False
                break
    if valid:
        for index, request in enumerate(requests):
            def bridge(row, display_model, display_data):
                tagged = copy.deepcopy(row)
                tagged.update(move_index=index, move_count=len(requests))
                observer(tagged, display_model, display_data)

            move = execute_transfer(model, data, scene, profile, reference, manager, request,
                                    observer=bridge if observer is not None else None, keep_samples=keep_samples)
            moves.append(move)
            trace.append(dict(move["final_contacts"]))
            status, reason = move["status"], move["reason"]
            if not move["success"] or move["final_reference"] is None:
                failed = index
                if move["success"]:
                    status, reason = "CONTROL_FAILURE", "Successful primitive omitted its admitted final reference"
                break
            completed += 1
            reference = move["final_reference"]
    final = get_state_summary(model, data, manager)
    steps = sum(move["steps"] for move in moves)
    duration = sum(move["duration_s"] for move in moves)
    success = bool(valid and failed is None and completed == len(requests))
    return {"success": success, "status": status, "reason": reason, "completed_moves": completed,
            "failed_index": failed, "total_steps": steps, "steps": steps, "duration_s": duration,
            "total_time_s": duration, "initial_time_s": initial.time, "final_time_s": final.time,
            "dt_s": float(model.opt.timestep), "initial_state": initial, "final_state": final,
            "initial_integration_state": initial_integration,
            "final_integration_state": tuple(float(v) for v in _integration_state(model, data)),
            "current_contacts": dict(final.contact_configuration), "final_contacts": dict(final.contact_configuration),
            "contact_trace": trace, "moves": moves, "final_reference": reference if success else None}


def make_transfer_fixture(timestep=.002, *, profile=None):
    """Maintained hand-family targets and native foot target, unchanged physics."""
    return foot_transfer.make_foot_transfer_fixture(timestep, profile=profile)


def run_transfer_benchmark(timestep=.002, *, kind="sequence", profile=None, negative=None,
                           observer=None, keep_samples=True) -> dict:
    """Admit/hold the common source once, then run a primitive or RH -> LH chain.

    second_unreachable is a declared target substitution, not a force/state fault.
    Initial setup is separate evidence and is not included in transfer step totals.
    """
    if kind not in ("right_hand", "left_hand", "foot", "sequence"):
        raise ValueError("Unknown transfer benchmark kind")
    if negative is not None and (negative != "second_unreachable" or kind != "sequence"):
        raise ValueError("Only sequence second_unreachable is a declared benchmark negative")
    model, data, scene, profile, seed = make_transfer_fixture(timestep, profile=profile)
    retarget = solve_contact_pose(model, scene, profile, seed, scene.start_configuration)
    if not retarget.admitted:
        # No manager can own this uninitialized episode; report the untouched
        # integration state directly rather than adopting it for a summary.
        state = {"time": float(data.time), "qpos": tuple(data.qpos), "qvel": tuple(data.qvel),
                 "ctrl": tuple(data.ctrl), "qacc_warmstart": tuple(data.qacc_warmstart),
                 "eq_active": tuple(bool(v) for v in data.eq_active),
                 "integration_state": tuple(_integration_state(model, data))}
        return {"success": False, "status": "INITIALIZATION_FAILURE", "reason": retarget.reason,
                "retarget": retarget, "steps": 0, "duration_s": 0., "final_reference": None,
                "initial_state": state, "final_state": state, "initial_static": None, "initial_reference": None,
                "kind": kind, "negative": negative, "profile": profile.to_dict(), "dt_s": float(model.opt.timestep)}
    reference, manager = initialize_static_reference(model, data, scene, profile, retarget.qpos, scene.start_configuration)
    setup_initial = get_state_summary(model, data, manager)
    initial = execute_static_hold(model, data, scene, profile, reference, manager, duration=2., settle=1.,
                                  score_window=.5, keep_samples=False)
    contacts = dict(reference.contact_intent)
    right = TransferRequest(Limb.RIGHT_HAND, contacts[Limb.RIGHT_HAND], "reach_target", contacts)
    left = TransferRequest(Limb.LEFT_HAND, contacts[Limb.LEFT_HAND], "left_reach_target", contacts)
    foot = TransferRequest(Limb.LEFT_FOOT, contacts[Limb.LEFT_FOOT], "foot_target", contacts)
    if not initial["success"]:
        result = {"success": False, "status": "INITIALIZATION_FAILURE", "reason": initial["reason"],
                  "initial_state": initial["initial_state"], "final_state": initial["final_state"],
                  "steps": 0, "duration_s": 0., "final_reference": None}
    elif kind == "sequence":
        second_contacts = {**contacts, Limb.RIGHT_HAND: right.target}
        second = TransferRequest(Limb.LEFT_HAND, second_contacts[Limb.LEFT_HAND],
                                 "unreachable_target" if negative else left.target, second_contacts)
        result = execute_transfer_sequence(model, data, scene, profile, (right, second), reference, manager,
                                           observer=observer, keep_samples=keep_samples)
    else:
        request = {"right_hand": right, "left_hand": left, "foot": foot}[kind]
        result = execute_transfer(model, data, scene, profile, reference, manager, request,
                                  observer=observer, keep_samples=keep_samples)
    result.update(initial_static=initial, retarget=retarget, initial_reference=reference,
                  setup_initial_state=setup_initial, kind=kind, negative=negative, profile=profile.to_dict())
    return result
