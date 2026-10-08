"""Stage3 impedance evidence on production-derived, fixed-mount single hinges.

These fixtures are not human-physics replacements or contact benchmarks. Other
joints in the selected subtree are welded by removing their joint declarations;
mass, geometry, selected-joint damping/armature, ROM and motor remain unchanged.
The legacy normalized controller below exists ONLY as a diagnostic baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .mjcf_builder import build_mjcf
from .runtime import compile_model, compute_pose_control
from .schema import BoulderScene, ClimberProfile


REPRESENTATIVE_JOINTS = (
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "left_hip_pitch", "left_knee", "left_ankle_pitch",
)
LEGACY_BASELINE = "legacy_normalized_kp5_kd0.5_NOT_PRODUCTION"
VELOCITY_NOISE_FLOOR = 1e-5  # rad/s, well above numerical convergence noise
TORQUE_NOISE_FLOOR = 1e-4  # Nm


@dataclass(frozen=True)
class IsolatedFixture:
    model: Any
    xml: str
    joint_name: str
    profile: ClimberProfile
    proximal_body: str
    removed_joints: tuple[str, ...]
    effective_inertia_kg_m2: float


def make_isolated_fixture(
    joint_name: str = "left_wrist",
    *,
    profile: ClimberProfile | None = None,
    timestep: float = .002,
) -> IsolatedFixture:
    """Compile one production motor/hinge and its rigid downstream subtree.

    The joint's original parent is represented by a fixed, massless mount. The
    subtree's local body transforms and inertial/geom attributes are preserved;
    only collision masks and other joint declarations change. Inherited joint
    defaults are copied rather than inventing extra damping or armature.
    """
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    profile = profile if profile is not None else ClimberProfile(name="isolated")
    scene = BoulderScene("xyz_metres", 1., (), ())
    source = ET.fromstring(build_mjcf(scene, profile))
    parents = {child: parent for parent in source.iter() for child in parent}
    selected = source.find(f".//joint[@name='{joint_name}']")
    motor = source.find(f"./actuator/motor[@joint='{joint_name}']")
    if selected is None or motor is None:
        raise ValueError(f"Expected a production actuated hinge, got {joint_name!r}")
    body = parents[selected]
    proximal_body = parents[body].get("name", "world")
    subtree = deepcopy(body)
    removed = []
    for owner in subtree.iter("body"):
        for joint in list(owner.findall("joint")) + list(owner.findall("freejoint")):
            if joint.get("name") != joint_name:
                removed.append(joint.get("name", "unnamed"))
                owner.remove(joint)
    for geom in subtree.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    root = ET.Element("mujoco", model=f"stage3_isolated_{joint_name}_{profile.name}")
    for tag in ("compiler", "option", "default"):
        root.append(deepcopy(source.find(tag)))
    option = root.find("option")
    option.set("timestep", repr(timestep))
    option.set("gravity", "0 0 0")
    mount = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="isolated_mount")
    mount.append(subtree)
    ET.SubElement(root, "actuator").append(deepcopy(motor))
    xml = ET.tostring(root, encoding="unicode")
    model = compile_model(xml)
    if (model.nq, model.nv, model.nu, model.njnt, model.neq) != (1, 1, 1, 1, 0):
        raise ValueError("Isolation did not produce exactly one actuated hinge")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inertia = np.empty(1)
    mujoco.mj_mulM(model, data, inertia, np.ones(1))
    return IsolatedFixture(model, xml, joint_name, profile, proximal_body, tuple(removed), float(inertia[0]))


def _signal_metrics(values: np.ndarray, floor: float, timestep: float) -> dict[str, Any]:
    """Only adjacent, above-floor samples can count as successive sign flips."""
    active = np.abs(values) > floor
    pairs = active[:-1] & active[1:]
    flips = pairs & (np.signbit(values[:-1]) != np.signbit(values[1:]))
    centered = values - np.mean(values)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    dominant = None
    high_fraction = 0.
    if rms > floor and len(values) > 2:
        # No window: a true alternating sequence lies exactly at Nyquist.
        power = np.abs(np.fft.rfft(centered)) ** 2
        power[0] = 0.
        frequencies = np.fft.rfftfreq(len(values), timestep)
        dominant = float(frequencies[int(np.argmax(power))])
        high_fraction = float(np.sum(power[frequencies >= .4 / timestep]) / np.sum(power))
    return {
        "sign_changes": int(np.count_nonzero(flips)),
        "successive_flip_fraction": float(np.count_nonzero(flips) / max(1, np.count_nonzero(pairs))),
        "dominant_frequency_Hz": dominant,
        "high_frequency_power_fraction": high_fraction,
    }


def run_isolated_case(
    fixture: IsolatedFixture,
    *,
    target: float = .05,
    duration: float = 3.,
    initial_position: float = 0.,
    initial_velocity: float = 0.,
    pulse: tuple[float, float, float] | None = None,
    controller: str = "physical",
    kp: float | Mapping[str, float] | None = None,
    kd: float | Mapping[str, float] | None = None,
    include_traces: bool = False,
) -> dict[str, Any]:
    """Measure a step, optionally disturbed by (start_s, duration_s, torque_Nm).

    Rise time is first 10%-to-90% crossing; settling is entry into a 2% position
    band (minimum .001 rad) for the remainder of the run. Pulse recovery starts
    at pulse end. Convergence additionally requires tail max speed <= .01 rad/s.
    The tail is the final min(.5 s, duration/4). FFT uses demeaned tail velocity,
    with high frequency defined as >=80% of Nyquist. Results are strict-JSON
    compatible, including optional traces. No source fixture state is mutated.
    """
    model, name = fixture.model, fixture.joint_name
    dt = float(model.opt.timestep)
    if (not all(math.isfinite(v) for v in (target, duration, initial_position, initial_velocity))
            or duration < 2 * dt):
        raise ValueError("Case needs finite references/state and at least two timesteps")
    if not model.jnt_range[0, 0] <= initial_position <= model.jnt_range[0, 1]:
        raise ValueError("Initial position violates compiled ROM")
    if controller not in ("physical", LEGACY_BASELINE):
        raise ValueError("Unknown diagnostic controller")
    if controller == LEGACY_BASELINE and (kp is not None or kd is not None):
        raise ValueError("Legacy baseline is fixed normalized kp=5, kd=.5")
    steps = int(math.ceil(duration / dt))
    if pulse is not None:
        start, width, torque = pulse
        if (not all(math.isfinite(v) for v in pulse) or start < 0 or width < dt
                or start + width >= steps * dt):
            raise ValueError("Pulse must last at least one timestep and end before the run")

    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0] = initial_position, initial_velocity
    mujoco.mj_forward(model, data)
    # Admission uses the production contract even for the explicitly legacy law.
    first = compute_pose_control(model, data, {name: target}, kp=kp, kd=kd)
    gear = float(model.actuator_gear[0, 0])
    limit = first.limits_Nm[0]
    k, b = first.stiffness_Nm_rad[0], first.damping_Nms_rad[0]
    if controller == LEGACY_BASELINE:
        k, b = 5 * gear, .5 * gear
    times = np.arange(steps + 1, dtype=float) * dt
    q, v = np.empty(steps + 1), np.empty(steps + 1)
    desired, commanded, external = np.empty(steps), np.empty(steps), np.zeros(steps)
    saturated = np.empty(steps, dtype=bool)
    q[0], v[0] = initial_position, initial_velocity
    max_contacts = 0
    torque_mapping_error = 0.
    for i in range(steps):
        if controller == "physical":
            command = compute_pose_control(model, data, {name: target}, kp=kp, kd=kd)
            desired[i], commanded[i], saturated[i] = command.desired_Nm[0], command.commanded_Nm[0], command.saturated[0]
        else:
            # Historical normalized controls, NOT a production controller path.
            normalized = 5 * (target - data.qpos[0]) - .5 * data.qvel[0]
            data.ctrl[0] = np.clip(normalized, -1., 1.)
            desired[i], commanded[i] = normalized * gear, float(data.ctrl[0]) * gear
            saturated[i] = abs(normalized) > 1.
        if pulse is not None and pulse[0] <= times[i] < pulse[0] + pulse[1]:
            external[i] = pulse[2]
        data.qfrc_applied[0] = external[i]
        mujoco.mj_step(model, data)
        q[i + 1], v[i + 1] = data.qpos[0], data.qvel[0]
        if not np.isfinite([q[i + 1], v[i + 1], data.time, *data.qacc]).all():
            raise ValueError(f"Nonfinite isolated state at step {i + 1}")
        max_contacts = max(max_contacts, int(data.ncon))
        torque_mapping_error = max(torque_mapping_error, abs(float(data.qfrc_actuator[0]) - float(commanded[i])))

    error = target - q
    amplitude = abs(target - initial_position)
    band = max(.001, .02 * amplitude)
    direction = 1. if target >= initial_position else -1.
    progress = direction * (q - initial_position)
    rise = None
    if amplitude > 0:
        ten, ninety = np.flatnonzero(progress >= .1 * amplitude), np.flatnonzero(progress >= .9 * amplitude)
        if len(ten) and len(ninety):
            rise = float(times[ninety[0]] - times[ten[0]])
    outside = np.flatnonzero(np.abs(error) > band)
    settled_index = int(outside[-1] + 1) if len(outside) else 0
    settle = float(times[settled_index]) if settled_index <= steps else None
    recovery = None
    disturbance_peak = None
    if pulse is not None:
        end_index = int(np.searchsorted(times, pulse[0] + pulse[1] - dt * 1e-8))
        post_outside = np.flatnonzero(np.abs(error[end_index:]) > band)
        recovery_index = end_index + int(post_outside[-1] + 1) if len(post_outside) else end_index
        if recovery_index <= steps:
            recovery = max(0., float(times[recovery_index] - times[end_index]))
        disturbance_peak = float(np.max(np.abs(error[times >= pulse[0]])))
    tail_steps = max(1, int(round(min(.5, steps * dt / 4) / dt)))
    tail_error, tail_speed = error[-tail_steps - 1:], v[-tail_steps - 1:]
    velocity_signal = _signal_metrics(v, VELOCITY_NOISE_FLOOR, dt)
    torque_signal = _signal_metrics(commanded, TORQUE_NOISE_FLOOR, dt)
    tail_velocity_signal = _signal_metrics(tail_speed, VELOCITY_NOISE_FLOOR, dt)
    tail_torque_signal = _signal_metrics(commanded[-tail_steps:], TORQUE_NOISE_FLOOR, dt)
    utilization = np.abs(commanded) / limit
    warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
    metrics = {
        "rise_10_90_s": rise,
        "settle_s": settle,
        "settle_band_rad": band,
        "overshoot_fraction": float(max(0., np.max(direction * (q - target))) / amplitude) if amplitude else 0.,
        "pulse_recovery_s": recovery,
        "pulse_peak_error_rad": disturbance_peak,
        "final_error_rad": float(error[-1]),
        "tail_max_error_rad": float(np.max(np.abs(tail_error))),
        "tail_rms_error_rad": float(np.sqrt(np.mean(tail_error ** 2))),
        "peak_speed_rad_s": float(np.max(np.abs(v))),
        "tail_max_speed_rad_s": float(np.max(np.abs(tail_speed))),
        "tail_rms_speed_rad_s": float(np.sqrt(np.mean(tail_speed ** 2))),
        "desired_peak_Nm": float(np.max(np.abs(desired))),
        "torque_peak_Nm": float(np.max(np.abs(commanded))),
        "peak_utilization": float(np.max(utilization)),
        "saturated_steps": int(np.count_nonzero(saturated)),
        "saturated_fraction": float(np.mean(saturated)),
        "saturated_time_s": float(np.count_nonzero(saturated) * dt),
        "torque_mapping_max_error_Nm": torque_mapping_error,
        "velocity": velocity_signal,
        "torque": torque_signal,
        "tail_velocity": tail_velocity_signal,
        "tail_torque": tail_torque_signal,
        "max_contacts": max_contacts,
        "warnings": warnings,
    }
    checks = {
        "finite": bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(desired).all()),
        "physical_torque_bound": bool(np.max(np.abs(commanded)) <= limit + 1e-10),
        "actual_torque_mapping": torque_mapping_error < 1e-10,
        "no_contacts_or_warnings": max_contacts == 0 and not warnings,
        "tail_position_converged": metrics["tail_max_error_rad"] <= band,
        "tail_velocity_converged": metrics["tail_max_speed_rad_s"] <= .01,
        "no_successive_tail_chatter": tail_velocity_signal["successive_flip_fraction"] <= .1,
    }
    result = {
        "configuration": {
            "joint": name, "profile": fixture.profile.name, "strength_scale": fixture.profile.strength_scale,
            "timestep_s": dt, "duration_s": steps * dt, "target_rad": target,
            "initial_position_rad": initial_position, "initial_velocity_rad_s": initial_velocity,
            "pulse": list(pulse) if pulse is not None else None, "controller": controller,
            "stiffness_Nm_rad": k, "damping_Nms_rad": b, "limit_Nm": limit,
            "effective_inertia_kg_m2": fixture.effective_inertia_kg_m2,
            "passive_damping_Nms_rad": float(model.dof_damping[0]),
            "armature_kg_m2": float(model.dof_armature[0]),
            "proximal_fixed_body": fixture.proximal_body, "removed_joints": list(fixture.removed_joints),
            "velocity_noise_floor_rad_s": VELOCITY_NOISE_FLOOR, "torque_noise_floor_Nm": TORQUE_NOISE_FLOOR,
        },
        "metrics": metrics, "checks": checks, "passed": all(checks.values()),
    }
    if include_traces:
        result["traces"] = {
            "time_s": times.tolist(), "position_rad": q.tolist(), "velocity_rad_s": v.tolist(),
            "error_rad": error.tolist(), "desired_Nm": desired.tolist(), "commanded_Nm": commanded.tolist(),
            "external_Nm": external.tolist(), "utilization": utilization.tolist(), "saturated": saturated.tolist(),
        }
    return result


def run_isolated_suite(
    *,
    joints: Sequence[str] = REPRESENTATIVE_JOINTS,
    timesteps: Sequence[float] = (.002, .001),
    include_traces: bool = False,
) -> dict[str, Any]:
    """Small/moderate steps, pulse, strength/capability, and distal legacy cases.

    Identical production class gains apply to ALL physical cases. Demanding
    references are ROM-admitted and below the strong motor's initial ceiling.
    If strength .5 cannot saturate at that reference (notably ankle), an extra
    .1 demanding case is included, without retuning the small-case gains.
    """
    if not joints or not timesteps:
        raise ValueError("Suite needs joints and timesteps")
    cases = []
    for name in joints:
        for dt in timesteps:
            base = make_isolated_fixture(name, timestep=dt)
            for label, kwargs in (
                ("small", {"target": .05}), ("moderate", {"target": .5}),
                ("pulse", {"target": .05, "pulse": (1., .05, .5)}),
            ):
                result = run_isolated_case(base, include_traces=include_traces, **kwargs)
                result["case"] = label
                cases.append(result)
            weak = make_isolated_fixture(name, profile=ClimberProfile(name="weak_label", strength_scale=.5), timestep=dt)
            strong = make_isolated_fixture(name, profile=ClimberProfile(name="unrelated_strong_label", strength_scale=1.5), timestep=dt)
            command = compute_pose_control(base.model, mujoco.MjData(base.model), {name: .05})
            target = min(.9 * float(base.model.jnt_range[0, 1]), .8 * float(strong.model.actuator_gear[0, 0]) / command.stiffness_Nm_rad[0])
            for fixture, strength_label in ((weak, "weak"), (strong, "strong")):
                for label, reference in (("small", .05), ("demanding", target)):
                    result = run_isolated_case(fixture, target=reference, include_traces=include_traces)
                    result["case"] = f"{label}_{strength_label}"
                    cases.append(result)
            if command.stiffness_Nm_rad[0] * target <= weak.model.actuator_gear[0, 0]:
                very_weak = make_isolated_fixture(name, profile=ClimberProfile(name="weak_demand_only", strength_scale=.1), timestep=dt)
                result = run_isolated_case(very_weak, target=target, include_traces=include_traces)
                result["case"] = "demanding_weak_0.1"
                cases.append(result)
            if name in ("left_wrist", "left_ankle_pitch"):
                result = run_isolated_case(base, controller=LEGACY_BASELINE, include_traces=include_traces)
                result["case"] = "small_legacy_NOT_PRODUCTION"
                cases.append(result)
    physical = [case for case in cases if case["configuration"]["controller"] == "physical"]
    return {
        "scope": "Stage3 fixed-mount rigid-subtree hinges; zero gravity, no collisions/equalities; not full-climber evidence",
        "legacy_note": "Normalized kp=5, kd=.5 baseline ONLY; its effective physical gains scale with motor gear",
        "weak_demand_note": "Strength .1 only where .5 cannot saturate; unchanged physical gains in every small/demanding case",
        "passed": all(case["passed"] for case in physical),
        "physical_case_count": len(physical), "case_count": len(cases), "cases": cases,
    }
