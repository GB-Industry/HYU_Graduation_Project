"""Read-only geometry/continuity metering of saved Stage 5 native evidence.

No files, renderer, live MjData, controller, IK, or physics execution are owned
here. The caller supplies the matching compiled model and decoded result JSON.
These measurements are not a physical-success or visual motion-quality verdict.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import mujoco
import numpy as np


def _numbers(value, label, shape=None):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Corrupt recorded state: {label} is not numeric") from exc
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError(f"Corrupt recorded state: {label} must be finite with shape {shape}")
    return array


def trajectory_rows(evidence) -> list[dict]:
    """Return detached native rows, stably sorted by actual recorded time.

    Includes each move's actual initial/final StateSummary, not setup/references
    or 20 Hz observer callbacks. Duplicate endpoint/boundary times are retained.
    `record_index` preserves storage order so sorting cannot conceal a reset.
    Native sample metadata is retained; snapshots alias hand_states/foot_states
    to hands/feet for renderer use. No pose interpolation or resampling occurs.
    """
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence must be a decoded result mapping")
    moves = evidence.get("moves") if "moves" in evidence else [evidence]
    if not isinstance(moves, (list, tuple)) or not moves:
        raise ValueError("Evidence requires at least one recorded transfer")
    rows = []
    for index, move in enumerate(moves):
        if not isinstance(move, Mapping):
            raise ValueError(f"Move {index} must be a result mapping")
        limb = move.get("moving_limb")
        if limb not in ("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"):
            raise ValueError(f"Move {index} requires an explicit moving_limb")
        if any(not isinstance(move.get(k), str) or not move[k] for k in ("source", "target")):
            raise ValueError(f"Move {index} requires source and target names")
        samples = move.get("samples")
        if not isinstance(samples, (list, tuple)):
            raise ValueError(f"Move {index} requires retained native samples")
        states = [("initial_state", None, move.get("initial_state"))]
        states.extend(("sample", i, sample) for i, sample in enumerate(samples))
        states.append(("final_state", None, move.get("final_state")))
        for kind, sample_index, state in states:
            row_id = f"move_{index}.{kind}" + (f"[{sample_index}]" if sample_index is not None else "")
            if not isinstance(state, Mapping) or "qpos" not in state:
                raise ValueError(f"Corrupt recorded state: {row_id} requires an actual qpos snapshot")
            row = copy.deepcopy(dict(state))
            pose = _numbers(row["qpos"], row_id + ".qpos")
            if pose.ndim != 1 or not len(pose):
                raise ValueError(f"Corrupt recorded state: {row_id}.qpos must be a nonempty vector")
            time = _numbers(row.get("time_s" if kind == "sample" else "time"), row_id + ".time", ())
            if kind != "sample":
                for source, target in (("hand_states", "hands"), ("foot_states", "feet"),
                                       ("contact_configuration", "contacts")):
                    if source in row:
                        row[target] = row[source]
            row.update(time_s=float(time), row_id=row_id, row_kind=kind, sample_index=sample_index,
                       record_index=len(rows), move_index=index, move_count=len(moves),
                       moving_limb=limb, source=move["source"], target=move["target"])
            rows.append(row)
    rows.sort(key=lambda row: row["time_s"])
    return rows


def _pair(rows, before, after):
    return {"before_row_id": rows[before]["row_id"], "after_row_id": rows[after]["row_id"],
            "before_time_s": rows[before]["time_s"], "after_time_s": rows[after]["time_s"],
            "delta_time_s": rows[after]["time_s"] - rows[before]["time_s"]}


def _position_metrics(points, rows):
    displacement = points[-1] - points[0]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    index = int(np.argmax(steps)) if len(steps) else None
    return {"initial_world_m": points[0].tolist(), "final_world_m": points[-1].tolist(),
            "displacement_world_m": displacement.tolist(), "net_displacement_m": float(np.linalg.norm(displacement)),
            "max_distance_from_start_m": float(np.max(np.linalg.norm(points - points[0], axis=1))),
            "path_length_m": float(np.sum(steps)), "xyz_peak_to_peak_m": np.ptp(points, axis=0).tolist(),
            "min_world_m": np.min(points, axis=0).tolist(), "max_world_m": np.max(points, axis=0).tolist(),
            "largest_successive_step_m": float(steps[index]) if index is not None else 0.,
            "largest_step_pair": _pair(rows, index, index + 1) if index is not None else None}


def _angles(quaternions, reference):
    # Inputs have already passed finite/unit validation. Normalize only roundoff
    # for acos; q and -q represent the same orientation.
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.minimum(1., np.abs(np.sum(q * r, axis=-1)))))


def _orientation_metrics(quaternions, rows):
    relative = _angles(quaternions, quaternions[0])
    steps = _angles(quaternions[1:], quaternions[:-1])
    index = int(np.argmax(steps)) if len(steps) else None
    return {"initial_quaternion_wxyz": quaternions[0].tolist(),
            "final_quaternion_wxyz": quaternions[-1].tolist(),
            "net_relative_angle_deg": float(relative[-1]), "max_relative_angle_deg": float(np.max(relative)),
            "largest_successive_angle_deg": float(steps[index]) if index is not None else 0.,
            "largest_step_pair": _pair(rows, index, index + 1) if index is not None else None}


def _joint_metrics(values, joints, rows):
    metrics = {}
    for column, joint in enumerate(joints):
        q = values[:, column]
        increments = np.abs(np.diff(q))
        index = int(np.argmax(increments)) if len(increments) else None
        item = {**joint, "initial_rad": float(q[0]), "final_rad": float(q[-1]),
                "net_rad": float(q[-1] - q[0]), "min_rad": float(np.min(q)), "max_rad": float(np.max(q)),
                "peak_to_peak_rad": float(np.ptp(q)),
                "max_abs_from_start_rad": float(np.max(np.abs(q - q[0]))),
                "largest_successive_step_rad": float(increments[index]) if index is not None else 0.,
                "largest_step_pair": _pair(rows, index, index + 1) if index is not None else None}
        for key, value in list(item.items()):
            if key.endswith("_rad"):
                item[key.removesuffix("_rad") + "_deg"] = np.degrees(value).tolist()
        metrics[joint["name"]] = item
    return metrics


def _clock_metrics(rows, qpos, expected_dt):
    times = np.array([row["time_s"] for row in rows])
    delta = np.diff(times)
    positive = delta[delta > 0]
    # Clock tolerance only, not a motion-quality acceptance threshold.
    gaps = np.flatnonzero(delta > expected_dt + max(1e-12, expected_dt * 1e-6))
    resets = np.flatnonzero(delta < 0)
    changed_duplicates = np.flatnonzero((delta == 0) & np.any(np.diff(qpos, axis=0) != 0, axis=1))
    samples = [row for row in rows if row["row_kind"] == "sample"]
    sample_times = np.array([row["time_s"] for row in samples])
    step_resets = sum(a["move_index"] == b["move_index"] and b["steps"] <= a["steps"]
                      for a, b in zip(samples, samples[1:]) if "steps" in a and "steps" in b)
    return {"initial_time_s": float(times[0]), "final_time_s": float(times[-1]),
            "time_span_s": float(times[-1] - times[0]), "min_time_s": float(np.min(times)),
            "max_time_s": float(np.max(times)), "expected_dt_s": expected_dt,
            "native_row_count": len(rows), "native_sample_count": len(samples),
            "nondecreasing_recorded_times": bool(np.all(delta >= 0)),
            "strictly_increasing_sample_times": bool(np.all(np.diff(sample_times) > 0)),
            "positive_dt_count": int(np.sum(delta > 0)), "duplicate_time_count": int(np.sum(delta == 0)),
            "time_reset_count": len(resets), "sample_gap_count": len(gaps),
            "changed_pose_at_duplicate_time_count": len(changed_duplicates),
            "step_counter_reset_count": int(step_resets),
            "min_positive_dt_s": float(np.min(positive)) if len(positive) else None,
            "max_positive_dt_s": float(np.max(positive)) if len(positive) else None,
            "max_dt_error_s": float(np.max(np.abs(positive - expected_dt))) if len(positive) else None,
            "gaps": [_pair(rows, int(i), int(i + 1)) for i in gaps],
            "resets": [_pair(rows, int(i), int(i + 1)) for i in resets],
            "changed_duplicate_pairs": [_pair(rows, int(i), int(i + 1)) for i in changed_duplicates]}


def _support_metrics(rows):
    result = {"hands": {}, "feet": {}}
    for group, limbs, fields in (
        ("hands", ("LEFT_HAND", "RIGHT_HAND"), ("load", "capacity", "margin")),
        ("feet", ("LEFT_FOOT", "RIGHT_FOOT"), ("normal_force", "tangential_force", "tangential_speed")),
    ):
        for limb in limbs:
            observed = [(row, row[group][limb]) for row in rows if limb in row.get(group, {})]
            item = {"observation_count": len(observed)}
            for field in fields:
                values = []
                unavailable = 0
                for row, state in observed:
                    if field not in state:
                        unavailable += 1
                        continue
                    # Detached hand capacity/margin are undefined in native JSON.
                    if (group == "hands" and field in ("capacity", "margin")
                            and not state.get("active") and state[field] is None):
                        unavailable += 1
                        continue
                    values.append((row, state[field]))
                item[field + "_unavailable_count"] = unavailable
                unit = "m_s" if field == "tangential_speed" else "N"
                if values:
                    numbers = _numbers([value for _, value in values], f"{group}.{limb}.{field}", (len(values),))
                    minimum, maximum = int(np.argmin(numbers)), int(np.argmax(numbers))
                    item[field] = {"min_" + unit: float(numbers[minimum]), "max_" + unit: float(numbers[maximum]),
                                   "min_row_id": values[minimum][0]["row_id"], "max_row_id": values[maximum][0]["row_id"],
                                   "observation_mean_" + unit: float(np.mean(numbers))}
                else:
                    item[field] = None
            for flag in (("active", "valid") if group == "hands" else ("contacting", "supporting", "slipping")):
                item[flag + "_row_count"] = sum(bool(state.get(flag)) for _, state in observed)
            result[group][limb] = item
    return result


def audit_motion(model, evidence) -> dict:
    """Meter every saved actual pose using owned MjData and FK/COM only.

    Motion/path/increment metrics follow original record order, even if corrupt
    clocks reset; trajectory_rows independently exposes time-sorted render rows.
    All actuated hinges are located through actual compiled actuator_trnid,
    joint qpos/dof addresses, names and axes, never assumed pose offsets.
    """
    rows = sorted(trajectory_rows(evidence), key=lambda row: row["record_index"])
    moves = evidence["moves"] if "moves" in evidence else [evidence]
    root = model.body("climber_root").id
    pelvis = model.body("pelvis").id
    root_joint = int(model.body_jntadr[root])
    if model.body_jntnum[root] != 1 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Compiled climber_root must have its actual free joint")
    root_qadr = int(model.jnt_qposadr[root_joint])
    joints = []
    for actuator, joint_id in enumerate(model.actuator_trnid[:, 0]):
        joint_id = int(joint_id)
        if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Motion audit requires joint-transmitted hinge actuators")
        if any(j["joint_id"] == joint_id for j in joints):
            raise ValueError("Motion audit requires unique actuated hinges")
        joints.append({"name": model.joint(joint_id).name, "joint_id": joint_id, "actuator_id": actuator,
                       "qpos_address": int(model.jnt_qposadr[joint_id]),
                       "dof_address": int(model.jnt_dofadr[joint_id]),
                       "axis_local": model.jnt_axis[joint_id].tolist(),
                       "compiled_range_rad": model.jnt_range[joint_id].tolist()})
    qpos = np.empty((len(rows), model.nq))
    for i, row in enumerate(rows):
        label = row["row_id"]
        if row.get("finite") is False or row.get("pose_available") is False:
            raise ValueError(f"Corrupt recorded state: {label} explicitly marks its pose invalid")
        qpos[i] = _numbers(row["qpos"], label + ".qpos", (model.nq,))
        for field, size in (("qvel", model.nv), ("ctrl", model.nu), ("qacc_warmstart", model.nv),
                            ("qfrc_applied", model.nv), ("root_pose", 7)):
            if field in row:
                _numbers(row[field], label + "." + field, (size,))
        for field in ("external_force_world_N", "integration_state"):
            if field in row:
                _numbers(row[field], label + "." + field)
        for field in ("q_ref", "qd_ref"):
            if field in row:
                _numbers(row[field], label + "." + field, (model.nq if field == "q_ref" else model.nv,))
    quaternions = qpos[:, root_qadr + 3:root_qadr + 7]
    norms = np.linalg.norm(quaternions, axis=1)
    bad = np.flatnonzero(~np.isclose(norms, 1., rtol=0., atol=1e-6))
    if len(bad):
        raise ValueError(f"Corrupt recorded state: {rows[int(bad[0])]['row_id']}.root quaternion must be unit")
    limbs = list(dict.fromkeys(move["moving_limb"] for move in moves))
    sites = {limb: model.site(limb.lower() + "_site").id for limb in limbs}
    positions = {name: np.empty((len(rows), 3)) for name in
                 ("root", "pelvis", "pelvis_body_com", "climber_com", *limbs)}
    scratch = mujoco.MjData(model)
    for i, row in enumerate(rows):
        scratch.qpos[:] = qpos[i]
        scratch.time = row["time_s"]
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        positions["root"][i] = scratch.xpos[root]
        positions["pelvis"][i] = scratch.xpos[pelvis]
        positions["pelvis_body_com"][i] = scratch.xipos[pelvis]
        positions["climber_com"][i] = scratch.subtree_com[root]
        for limb, site in sites.items():
            positions[limb][i] = scratch.site_xpos[site]
    if any(not np.isfinite(points).all() for points in positions.values()):
        raise ValueError("Corrupt recorded state: nonfinite derived FK/COM geometry")
    hinge_values = qpos[:, [j["qpos_address"] for j in joints]]

    def summary(indices, dt, selected_limbs):
        selected_rows = [rows[i] for i in indices]
        joint_metrics = _joint_metrics(hinge_values[indices], joints, selected_rows)
        excursion = max(joint_metrics.values(), key=lambda j: j["peak_to_peak_rad"])
        increment = max(joint_metrics.values(), key=lambda j: j["largest_successive_step_rad"])
        return {"timing": _clock_metrics(selected_rows, qpos[indices], dt),
                "bodies": {name: _position_metrics(positions[name][indices], selected_rows)
                           for name in ("root", "pelvis", "pelvis_body_com", "climber_com")},
                "effectors": {limb: _position_metrics(positions[limb][indices], selected_rows) for limb in selected_limbs},
                "root_orientation": _orientation_metrics(quaternions[indices], selected_rows),
                "joints": joint_metrics,
                "largest_joint_excursion": {"name": excursion["name"], "rad": excursion["peak_to_peak_rad"],
                                            "deg": excursion["peak_to_peak_deg"]},
                "largest_joint_increment": {"name": increment["name"], "rad": increment["largest_successive_step_rad"],
                                            "deg": increment["largest_successive_step_deg"],
                                            "pair": increment["largest_step_pair"]},
                "support_loads": _support_metrics(selected_rows)}

    dt = float(_numbers(evidence.get("dt_s", model.opt.timestep), "dt_s", ()))
    if dt <= 0:
        raise ValueError("Corrupt recorded state: dt_s must be positive")
    transfers = []
    accounting = []
    boundaries = []
    for index, move in enumerate(moves):
        indices = np.array([i for i, row in enumerate(rows) if row["move_index"] == index])
        move_dt = float(_numbers(move.get("dt_s", dt), f"move_{index}.dt_s", ()))
        if move_dt <= 0:
            raise ValueError(f"Corrupt recorded state: move_{index}.dt_s must be positive")
        limb = move["moving_limb"]
        item = summary(indices, move_dt, [limb])
        suffixes = (("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist") if limb.endswith("HAND")
                    else ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
        relevant = [limb.split("_")[0].lower() + "_" + suffix for suffix in suffixes]
        if any(name not in item["joints"] for name in relevant):
            raise ValueError(f"Compiled model is missing relevant {limb} hinges")
        item.update(move_index=index, moving_limb=limb, source=move["source"], target=move["target"],
                    relevant_joints=relevant, moving_effector=item["effectors"][limb])
        time_order = indices[np.argsort([rows[i]["time_s"] for i in indices], kind="stable")]
        times = np.array([rows[i]["time_s"] for i in time_order])
        event_metrics, phases = [], []
        previous_time = None
        for event_index, event in enumerate(move.get("events", [])):
            time = float(_numbers(event.get("time_s"), f"move_{index}.events[{event_index}].time_s", ()))
            insertion = int(np.searchsorted(times, time, side="right"))
            before = int(time_order[insertion - 1]) if insertion else None
            after = int(time_order[insertion]) if insertion < len(time_order) else None

            def event_state(i):
                if i is None:
                    return None
                row = rows[i]
                group = "hands" if limb.endswith("HAND") else "feet"
                return {"row_id": row["row_id"], "time_s": row["time_s"], "row_kind": row["row_kind"],
                        "sample_index": row["sample_index"], "steps": row.get("steps"),
                        "moving_contact": copy.deepcopy(row.get(group, {}).get(limb)),
                        "eq_active": copy.deepcopy(row.get("eq_active"))}

            step = None
            if before is not None and after is not None:
                joint_delta = hinge_values[after] - hinge_values[before]
                maximum = int(np.argmax(np.abs(joint_delta)))
                step = {**_pair(rows, before, after),
                        "moving_effector_delta_world_m": (positions[limb][after] - positions[limb][before]).tolist(),
                        "end_m": float(np.linalg.norm(positions[limb][after] - positions[limb][before])),
                        "body_displacement_m": {name: float(np.linalg.norm(positions[name][after] - positions[name][before]))
                                                for name in ("root", "pelvis", "pelvis_body_com", "climber_com")},
                        "root_angle_deg": float(_angles(quaternions[after], quaternions[before])),
                        "joint_delta_rad": dict(zip((j["name"] for j in joints), joint_delta.tolist())),
                        "joint_max_step_rad": float(abs(joint_delta[maximum])),
                        "joint_max_step_deg": float(np.degrees(abs(joint_delta[maximum]))),
                        "joint_max_step_name": joints[maximum]["name"]}
            event_metrics.append({"event_index": event_index, "event": event.get("event"), "phase": event.get("phase"),
                                  "time_s": time, "offset_from_initial_s": time - rows[int(indices[0])]["time_s"],
                                  "delta_from_previous_event_s": time - previous_time if previous_time is not None else None,
                                  "event_before": event_state(before), "event_after": event_state(after), "native_step": step})
            previous_time = time
            if "phase" in event:
                phases.append({"phase": event["phase"], "start_time_s": time})
        for i, phase in enumerate(phases):
            phase["end_time_s"] = phases[i + 1]["start_time_s"] if i + 1 < len(phases) else move["final_state"]["time"]
            phase["duration_s"] = phase["end_time_s"] - phase["start_time_s"]
        release = move.get("release_time_s")
        if release is None:
            release = next((e["time_s"] for e in move.get("events", []) if e.get("event") == "RELEASED"), None)
        pre_release = indices[[rows[i]["time_s"] <= release + 1e-10 for i in indices]] if release is not None else []
        plateau = indices[[rows[i].get("phase") == "SETTLE" for i in indices]]
        load = indices[[rows[i].get("phase") == "LOAD" for i in indices]]
        item.update(events=event_metrics, phases=phases,
                    pre_release=summary(pre_release, move_dt, [limb]) if len(pre_release) else None,
                    native_load=summary(load, move_dt, [limb]) if len(load) else None,
                    loaded_plateau=summary(plateau, move_dt, [limb]) if len(plateau) else None)
        observed = [rows[i] for i in indices]
        applied = [row[field] for row in observed for field in ("qfrc_applied", "external_force_world_N")
                   if field in row]
        span = move["final_state"]["time"] - move["initial_state"]["time"]
        steps, duration = move.get("steps"), move.get("duration_s")
        tolerance = max(1e-9, abs(span) * 1e-9)
        accounting.append({"move_index": index, "recorded_steps": steps, "recorded_duration_s": duration,
                           "native_clock_span_s": span,
                           "step_clock_matches": steps is not None and abs(span - steps * move_dt) <= tolerance,
                           "duration_matches_steps": steps is not None and duration is not None
                                                     and abs(duration - steps * move_dt) <= tolerance,
                           "applied_force_observation_count": len(applied),
                           "samples_have_both_force_channels": all("qfrc_applied" in row
                               and "external_force_world_N" in row for row in observed if row["row_kind"] == "sample"),
                           "zero_recorded_applied_forces": all(not np.any(value) for value in applied)})
        transfers.append(item)
        if index:
            old, new = moves[index - 1]["final_state"], move["initial_state"]
            old_integration = moves[index - 1].get("final_integration_state", old.get("integration_state"))
            new_integration = move.get("initial_integration_state", new.get("integration_state"))
            a, b = int(indices[0] - 1), int(indices[0])
            boundaries.append({"before_move_index": index - 1, "after_move_index": index,
                               "full_state_equal": old == new,
                               "integration_state_equal": (old_integration == new_integration
                                    if old_integration is not None and new_integration is not None else None),
                               "differing_fields": sorted(k for k in old.keys() | new.keys()
                                                          if k not in old or k not in new or old[k] != new[k]),
                               **_pair(rows, a, b), "max_qpos_delta": float(np.max(np.abs(qpos[b] - qpos[a]))),
                               "pelvis_displacement_m": float(np.linalg.norm(positions["pelvis"][b] - positions["pelvis"][a])),
                               "com_displacement_m": float(np.linalg.norm(positions["climber_com"][b] - positions["climber_com"][a]))})
    whole = summary(np.arange(len(rows)), dt, limbs)
    result = {"schema_version": 1, "kind": evidence.get("kind"), "moving_limbs": limbs,
            "method": "Recorded actual qpos only; mj_kinematics + mj_comPos on owned scratch; no dynamics/references",
            "quality_verdict": None,
            "limitations": ["Motion quality requires the main renderer's manual multi-camera visual inspection.",
                            "Event brackets use closest recorded time <= event and next > event, not an invented activation frame.",
                            "Small native increments/full-state equality are observations, not universal no-snap or autonomy proofs.",
                            "Sequence effectors are separate full-case trajectories; paths are never joined across different limbs.",
                             "Support loads are recorded observations, not forces recomputed by this audit.",
                             "Detached hand capacity/margin nulls are unavailable observations, never zero capacity/margin.",
                             "Full-state equality covers saved snapshot fields; unrecorded integration_state equality is unavailable."],
            "geometry": {"root_body": "climber_root", "pelvis_body": "pelvis", "com_subtree": "climber_root",
                         "climber_mass_kg": float(model.body_subtreemass[root]),
                         "pelvis_position_source": "xpos", "pelvis_body_com_source": "xipos",
                         "climber_com_source": "subtree_com",
                         "pelvis_parent_body": model.body(int(model.body_parentid[pelvis])).name,
                         "pelvis_local_body_offset_m": model.body_pos[pelvis].tolist(),
                         "initial_root_to_pelvis_world_m": (positions["pelvis"][0] - positions["root"][0]).tolist()},
            "source_matches_initial_static_final_state": (
                evidence["initial_static"]["final_state"] == moves[0]["initial_state"]
                if isinstance(evidence.get("initial_static"), Mapping) and "final_state" in evidence["initial_static"] else None),
            "case_initial_matches_first_move": evidence["initial_state"] == moves[0]["initial_state"],
            "case_final_matches_last_move": evidence["final_state"] == moves[-1]["final_state"],
             "sequence_boundaries": boundaries, "native_accounting": accounting,
             "whole_case": whole, "transfers": transfers}
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Corrupt recorded state: nonfinite motion metric or event observation")
    return result
