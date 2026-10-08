from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import math
from typing import Any

import numpy as np
from .mjcf_builder import END_EFFECTOR_SITE_NAMES

DEFAULT_TARGET_POSE: dict[str, float] = {
    # Torso / Waist
    "waist_yaw": 0.0,
    "waist_pitch": 0.087266,   # ~5 deg backward lean (upright torso moves toward -Y)
    "waist_roll": 0.0,
    # Arms
    "left_shoulder_pitch": 0.785398,   # +45 deg forward reach
    "left_shoulder_roll": 0.261799,    # +15 deg abduction
    "left_shoulder_yaw": 0.0,
    "left_elbow": 1.047198,            # +60 deg flexion
    "left_wrist": 0.0,
    "right_shoulder_pitch": 0.785398,  # +45 deg forward reach
    "right_shoulder_roll": -0.261799,  # Negative angle abducts the right arm toward +X
    "right_shoulder_yaw": 0.0,
    "right_elbow": 1.047198,           # +60 deg flexion
    "right_wrist": 0.0,
    # Legs
    "left_hip_pitch": 0.523599,        # +30 deg flexion
    "left_hip_roll": 0.174533,         # +10 deg abduction
    "left_hip_yaw": 0.0,
    "left_knee": 0.785398,             # +45 deg flexion
    "left_ankle_pitch": -0.261799,     # -15 deg plantarflexion (positive raises toe)
    "left_ankle_roll": 0.0,
    "right_hip_pitch": 0.523599,       # +30 deg flexion
    "right_hip_roll": -0.174533,       # -10 deg abduction
    "right_hip_yaw": 0.0,
    "right_knee": 0.785398,            # +45 deg flexion
    "right_ankle_pitch": -0.261799,    # -15 deg plantarflexion (positive raises toe)
    "right_ankle_roll": 0.0,
}


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Install the optional runtime with: "
            "pip install 'mujoco>=3.2,<4'"
        ) from exc
    return mujoco


def compile_model(xml: str):
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_string(xml)
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled model numerics: {', '.join(issues)}")
    return model


def compiled_numerical_issues(model: Any) -> list[str]:
    """Check engine-derived arrays too, before forwarding or integrating data."""
    issues = []
    for owner_name, owner in (("model", model), ("option", model.opt), ("stat", model.stat)):
        for name in dir(owner):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray) and value.dtype.kind in "fci":
                if not np.isfinite(value).all():
                    issues.append(f"{owner_name}.{name}")
            elif isinstance(value, float) and not math.isfinite(value):
                issues.append(f"{owner_name}.{name}")
    if not model.opt.timestep > 0:
        issues.append("option.timestep must be positive")
    return issues


def validate_reference_pose(model: Any, qpos: Any) -> None:
    """Validate a complete reference against this prototype's scalar joint ROM."""
    if len(qpos) != model.nq or not all(math.isfinite(float(v)) for v in qpos):
        raise ValueError("Reference qpos must have model.nq finite values")
    violations = []
    mujoco = _import_mujoco()
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE:
            start = int(model.jnt_qposadr[jid]) + 3
            if not math.isclose(math.hypot(*qpos[start:start + 4]), 1.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Reference quaternion for {model.joint(jid).name!r} must be unit length")
        if model.jnt_limited[jid]:
            if int(model.jnt_type[jid]) != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Prototype reference ROM validation supports scalar hinges only")
            value = float(qpos[int(model.jnt_qposadr[jid])])
            lo, hi = model.jnt_range[jid]
            if not lo <= value <= hi:
                violations.append(model.joint(jid).name)
    if violations:
        raise ValueError(f"Reference pose violates compiled ROM: {', '.join(violations)}")


@dataclass(frozen=True)
class SmokeResult:
    steps: int
    time: float
    nq: int
    nv: int
    nu: int
    finite: bool


def smoke_step(xml: str, steps: int = 200) -> SmokeResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mujoco = _import_mujoco()
    model = compile_model(xml)
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    values = list(data.qpos) + list(data.qvel)
    finite = all(float("-inf") < float(v) < float("inf") for v in values)
    return SmokeResult(steps=steps, time=float(data.time), nq=model.nq, nv=model.nv, nu=model.nu, finite=finite)


def get_end_effector_positions(model: Any, data: Any) -> dict[str, tuple[float, float, float]]:
    _import_mujoco()
    positions: dict[str, tuple[float, float, float]] = {}
    for name in END_EFFECTOR_SITE_NAMES:
        pos = data.site(name).xpos
        positions[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return positions


# Physical impedance, independent of motor gear/profile strength. Nm/rad, Nm*s/rad.
IMPEDANCE_GAINS = MappingProxyType({
    "waist": (160., 25.), "shoulder": (80., 12.), "elbow": (60., 6.),
    "wrist": (20., 1.), "hip": (160., 25.), "knee": (120., 12.), "ankle": (40., 2.),
})


@dataclass(frozen=True)
class TorqueCommand:
    joint_names: tuple[str, ...]
    stiffness_Nm_rad: tuple[float, ...]
    damping_Nms_rad: tuple[float, ...]
    desired_Nm: tuple[float, ...]
    commanded_Nm: tuple[float, ...]
    limits_Nm: tuple[float, ...]
    utilization: tuple[float, ...]
    saturated: tuple[bool, ...]


def compute_pose_control(
    model: Any,
    data: Any,
    target_pose: dict[str, float] | None = None,
    kp: float | Mapping[str, float] | None = None,
    kd: float | Mapping[str, float] | None = None,
    *,
    target_velocity: Mapping[str, float] | None = None,
    feedforward: Mapping[str, float] | None = None,
) -> TorqueCommand:
    """Torque impedance: tau=Kp*(q_ref-q)+Kd*(qd_ref-qd)+tau_ff, then capability clamp.

    Kp is Nm/rad, Kd is Nm*s/rad, positions rad, velocities rad/s, torque Nm.
    Feedforward is explicit joint torque, never a root wrench/contact force.
    All admission precedes live control writes. Motor gear sets capability only.
    """
    mujoco = _import_mujoco()
    targets = target_pose if target_pose is not None else DEFAULT_TARGET_POSE
    velocities, ff = target_velocity or {}, feedforward or {}
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("Physical control requires finite generalized state")
    names, stiffness, damping, desired, limits = [], [], [], [], []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if (not jname or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                or model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE
                or model.actuator_gainprm[i, 0] != 1.):
            raise ValueError("Torque control requires direct scalar hinge motors")
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        target_val = targets.get(jname, 0.0)
        curr_pos = float(data.qpos[qadr])
        curr_vel = float(data.qvel[vadr])
        kind = jname.removeprefix("left_").removeprefix("right_").split("_")[0]
        gains = IMPEDANCE_GAINS.get(kind)
        if gains is None and (kp is None or kd is None):
            raise ValueError(f"Supply physical gains for unclassified joint {jname!r}")
        k = kp[jname] if isinstance(kp, Mapping) else kp if kp is not None else gains[0]
        b = kd[jname] if isinstance(kd, Mapping) else kd if kd is not None else gains[1]
        gear = float(model.actuator_gear[i, 0])
        lo, hi = model.actuator_ctrlrange[i]
        values = (k, b, target_val, curr_pos, curr_vel, velocities.get(jname, 0.), ff.get(jname, 0.), gear, lo, hi)
        if (not all(math.isfinite(float(v)) for v in values) or k < 0 or b < 0 or gear <= 0
                or not model.actuator_ctrllimited[i] or lo >= 0 or hi <= 0 or lo != -hi
                or model.actuator_forcelimited[i] or np.any(model.actuator_gear[i, 1:])):
            raise ValueError("Invalid physical gains, state, reference, or direct-motor capability")
        if model.jnt_limited[jid] and not model.jnt_range[jid, 0] <= target_val <= model.jnt_range[jid, 1]:
            raise ValueError(f"Reference target {jname!r} violates compiled ROM")
        names.append(jname)
        stiffness.append(float(k))
        damping.append(float(b))
        desired.append(float(k * (target_val - curr_pos) + b * (velocities.get(jname, 0.) - curr_vel) + ff.get(jname, 0.)))
        limits.append(float(gear * min(-lo, hi)))
    known = set(names)
    if any(set(mapping) - known for mapping in (targets, velocities, ff)):
        raise ValueError("Reference/velocity/feedforward must identify actuated hinges")
    requested, ceiling = np.array(desired), np.array(limits)
    if not np.isfinite(requested).all() or not np.isfinite(ceiling).all() or np.any(ceiling <= 0):
        raise ValueError("Nonfinite requested torque")
    commanded = np.clip(requested, -ceiling, ceiling)
    data.ctrl[:] = commanded / model.actuator_gear[:, 0]
    return TorqueCommand(tuple(names), tuple(stiffness), tuple(damping), tuple(desired), tuple(commanded),
                         tuple(limits), tuple(np.abs(commanded) / ceiling), tuple(np.abs(requested) > ceiling))


@dataclass(frozen=True)
class PoseControlResult:
    steps: int
    time: float
    initial_error: float
    final_error: float
    error_reduced: bool
    finite: bool
    end_effector_positions: dict[str, tuple[float, float, float]]


def run_pose_control(
    xml: str,
    target_pose: dict[str, float] | None = None,
    steps: int = 400,
    kp: float | None = None,
    kd: float | None = None,
    stabilize_root: bool = False,
) -> PoseControlResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mujoco = _import_mujoco()
    model = compile_model(xml)
    data = mujoco.MjData(model)
    targets = target_pose if target_pose is not None else DEFAULT_TARGET_POSE
    reference = model.qpos0.copy()
    for name, value in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if (jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                or not np.any(model.actuator_trnid[:, 0] == jid)):
            raise ValueError(f"Reference target {name!r} must identify an actuated hinge")
        reference[int(model.jnt_qposadr[jid])] = value
    validate_reference_pose(model, reference)

    mujoco.mj_forward(model, data)
    init_root_qpos = data.qpos[0:7].copy() if stabilize_root else None

    # Initial joint tracking error
    initial_errors: list[float] = []
    for jname, target_val in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            initial_errors.append(abs(target_val - float(data.qpos[qadr])))
    initial_error = float(sum(initial_errors) / len(initial_errors)) if initial_errors else 0.0

    # Simulate steps under deterministic PD pose control
    for _ in range(steps):
        compute_pose_control(model, data, targets, kp=kp, kd=kd)
        if stabilize_root and init_root_qpos is not None:
            data.qpos[0:7] = init_root_qpos
            data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)
        if stabilize_root and init_root_qpos is not None:
            data.qpos[0:7] = init_root_qpos
            data.qvel[0:6] = 0.0

    # Final joint tracking error
    final_errors: list[float] = []
    for jname, target_val in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            final_errors.append(abs(target_val - float(data.qpos[qadr])))
    final_error = float(sum(final_errors) / len(final_errors)) if final_errors else 0.0

    values = list(data.qpos) + list(data.qvel)
    ee_positions = get_end_effector_positions(model, data)
    ee_finite = all(math.isfinite(c) for pos in ee_positions.values() for c in pos)
    finite = all(math.isfinite(float(v)) for v in values) and ee_finite

    return PoseControlResult(
        steps=steps,
        time=float(data.time),
        initial_error=initial_error,
        final_error=final_error,
        error_reduced=final_error < initial_error,
        finite=finite,
        end_effector_positions=ee_positions,
    )
