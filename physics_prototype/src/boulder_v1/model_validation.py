"""Stage 1 evidence for the compiled physical model, not a climbing benchmark.

All probes own their model/data. No controller, stance initializer, attachments,
or trajectory state is used. Dimensions and mass fractions are engineering
proxies; these checks make no anthropometric or performance claims.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from itertools import product

import mujoco
import numpy as np

from .contact_geometry import ContactMode
from .mjcf_builder import build_mjcf
from .runtime import compile_model, compiled_numerical_issues
from .scene_factory import make_synthetic_scene
from .schema import BoulderScene, ClimberProfile


# Independent model contract: never infer expected directions/gears from MJCF.
# name, moving body, observed site, local axis, unit-control torque (Nm).
JOINT_SPECS = (
    ("waist_yaw", "abdomen", "left_hand_site", (0, 0, 1), 90),
    ("waist_pitch", "abdomen", "left_hand_site", (1, 0, 0), 110),
    ("waist_roll", "abdomen", "left_hand_site", (0, 1, 0), 90),
    ("left_shoulder_pitch", "left_upper_arm", "left_hand_site", (1, 0, 0), 85),
    ("left_shoulder_roll", "left_upper_arm", "left_hand_site", (0, 1, 0), 85),
    ("left_shoulder_yaw", "left_upper_arm", "left_hand_site", (0, 0, 1), 75),
    ("left_elbow", "left_forearm", "left_hand_site", (1, 0, 0), 65),
    ("left_wrist", "left_hand", "left_hand_site", (1, 0, 0), 35),
    ("right_shoulder_pitch", "right_upper_arm", "right_hand_site", (1, 0, 0), 85),
    ("right_shoulder_roll", "right_upper_arm", "right_hand_site", (0, 1, 0), 85),
    ("right_shoulder_yaw", "right_upper_arm", "right_hand_site", (0, 0, 1), 75),
    ("right_elbow", "right_forearm", "right_hand_site", (1, 0, 0), 65),
    ("right_wrist", "right_hand", "right_hand_site", (1, 0, 0), 35),
    ("left_hip_pitch", "left_thigh", "left_foot_site", (1, 0, 0), 140),
    ("left_hip_roll", "left_thigh", "left_foot_site", (0, 1, 0), 110),
    ("left_hip_yaw", "left_thigh", "left_foot_site", (0, 0, 1), 80),
    ("left_knee", "left_shin", "left_foot_site", (-1, 0, 0), 130),
    ("left_ankle_pitch", "left_foot", "left_foot_site", (1, 0, 0), 55),
    ("left_ankle_roll", "left_foot", "left_foot_site", (0, 1, 0), 40),
    ("right_hip_pitch", "right_thigh", "right_foot_site", (1, 0, 0), 140),
    ("right_hip_roll", "right_thigh", "right_foot_site", (0, 1, 0), 110),
    ("right_hip_yaw", "right_thigh", "right_foot_site", (0, 0, 1), 80),
    ("right_knee", "right_shin", "right_foot_site", (-1, 0, 0), 130),
    ("right_ankle_pitch", "right_foot", "right_foot_site", (1, 0, 0), 55),
    ("right_ankle_roll", "right_foot", "right_foot_site", (0, 1, 0), 40),
)
_REFLECT = np.diag([-1.0, 1.0, 1.0])
_AXIAL = np.diag([1.0, -1.0, -1.0])


def _compiled_numerics(model):
    issues = compiled_numerical_issues(model)
    return {"issues": issues, "checks": {"compiled_numeric_finite": not issues}, "passed": not issues}


def _require_finite(value, label):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite diagnostic {label}")
    return value


def _require_finite_sample(data):
    for name in ("time", "qpos", "qvel", "qacc", "qacc_warmstart", "xpos", "xquat", "xmat",
                 "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "subtree_com"):
        _require_finite(getattr(data, name), f"sample data.{name}")


def _integration_steps(duration, timestep):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    if duration < timestep or not np.isfinite(duration / timestep):
        raise ValueError("duration must cover at least one finite integration step")
    return round(duration / timestep)


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed_{index}"


def _id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Required compiled {kind.name}: {name}")
    return index


def climber_body_ids(model) -> np.ndarray:
    """IDs in the root subtree, excluding all fixed environment bodies."""
    root = _id(model, mujoco.mjtObj.mjOBJ_BODY, "climber_root")
    ids = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in ids:
            ids.add(body)
    return np.array(sorted(ids), dtype=int)


def _isolated_state(model, *, contacts=True, gravity=None, midpoint=False):
    issues = compiled_numerical_issues(model)
    if issues:
        raise ValueError(f"Invalid compiled model numerics: {', '.join(issues)}")
    model = copy.copy(model)
    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if contacts:
        model.opt.disableflags &= ~contact_bit
    else:
        model.opt.disableflags |= contact_bit
    if gravity is not None:
        model.opt.gravity[:] = gravity
    data = mujoco.MjData(model)
    data.eq_active[:] = 0
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    if midpoint:
        for joint in range(model.njnt):
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[model.jnt_qposadr[joint]] = np.mean(model.jnt_range[joint])
    mujoco.mj_forward(model, data)
    return model, data


def mass_matrix(model, data) -> np.ndarray:
    """Dense coupled generalized inertia via the stable public multiply API."""
    result = np.empty((model.nv, model.nv))
    for column, unit in enumerate(np.eye(model.nv)):
        vector = np.empty(model.nv)
        mujoco.mj_mulM(model, data, vector, unit)
        result[:, column] = vector
    return _require_finite(result, "mass matrix")


def _inertia_tensor(model, body):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def physics_numeric_arrays(model) -> dict[str, np.ndarray]:
    """Physics-only arrays for label invariance; no name/visual/string buffers."""
    prefixes = ("body_", "jnt_", "dof_", "geom_", "site_", "actuator_", "eq_", "exclude_", "pair_")
    visual = ("rgba", "matid", "group", "user", "texcoord", "dataid")
    arrays = {}
    for name in dir(model):
        if name.startswith(prefixes) and not any(token in name for token in visual):
            value = getattr(model, name)
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                arrays[name] = value.copy()
    for name in ("gravity", "timestep", "integrator", "disableflags", "enableflags",
                 "iterations", "tolerance", "solver", "cone", "impratio"):
        arrays[f"option_{name}"] = np.asarray(getattr(model.opt, name)).copy()
    arrays["counts"] = np.array([model.nq, model.nv, model.nu, model.nbody, model.njnt,
                                  model.ngeom, model.nsite, model.neq, model.nexclude])
    return arrays


def _proxy_inertia_errors(model):
    """Independent uniform primary-ellipsoid/capsule formulas in body frames."""
    ellipsoid, capsule = {}, {}
    for name in ("pelvis", "abdomen", "chest"):
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        x, y, z = model.geom_size[geom]
        expected = model.body_mass[body] / 5 * np.array([y*y + z*z, x*x + z*z, x*x + y*y])
        ellipsoid[name] = float(np.max(np.abs(_inertia_tensor(model, body) - np.diag(expected))))
    for side in ("left", "right"):
        for segment in ("upper_arm", "forearm", "thigh", "shin"):
            name = f"{side}_{segment}"
            body = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            radius, half_length = model.geom_size[geom, :2]
            length = 2 * half_length
            mass = model.body_mass[body]
            cylinder_mass = mass * length / (length + 4 * radius / 3)
            sphere_mass = mass - cylinder_mass
            axial = cylinder_mass * radius**2 / 2 + sphere_mass * 2 * radius**2 / 5
            transverse = (cylinder_mass * (3 * radius**2 + length**2) / 12
                          + sphere_mass * (2 * radius**2 / 5 + length**2 / 4
                                           + 3 * length * radius / 8))
            capsule[name] = {
                "tensor_max_error": float(np.max(np.abs(
                    _inertia_tensor(model, body) - np.diag([transverse, transverse, axial])))),
                "com_max_error": float(np.max(np.abs(model.body_ipos[body] - [0, 0, -half_length]))),
                "primary_length": float(length),
            }
    return {"ellipsoid_tensor_errors": ellipsoid, "capsule_errors": capsule}


def summarize_model(model, *, mass_scale=1.0) -> dict:
    """Read topology, mass/COM/inertia, collision policy and mappings from MjModel."""
    numerics = _compiled_numerics(model)
    if not numerics["passed"]:
        return numerics
    model, data = _isolated_state(model)
    _require_finite_sample(data)
    ids = climber_body_ids(model)
    id_set = set(ids.tolist())
    root = int(ids[0])
    env = np.array([i for i in range(model.nbody) if i not in id_set], dtype=int)
    total = float(np.sum(model.body_mass[ids]))
    environment_mass = float(np.sum(model.body_mass[env]))
    weighted_com = np.sum(model.body_mass[ids, None] * data.xipos[ids], axis=0) / total
    _require_finite(weighted_com, "weighted COM")
    proxy_bounds = np.array([[np.inf] * 3, [-np.inf] * 3])
    primary_names = {f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, body)}_geom" for body in ids} | {"neck_geom"}
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in id_set or _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) not in primary_names:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.sqrt(rotation**2 @ size**2)
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * np.abs(rotation[:, 2])
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation) @ size
        else:
            raise ValueError("Unsupported primary inertia-proxy geometry")
        proxy_bounds[0] = np.minimum(proxy_bounds[0], data.geom_xpos[geom] - extent)
        proxy_bounds[1] = np.maximum(proxy_bounds[1], data.geom_xpos[geom] + extent)
    mirror_center_error = float(abs(weighted_com[0] - data.xpos[root, 0]))
    _require_finite(proxy_bounds, "primary proxy bounds")
    bodies = []
    for body in range(model.nbody):
        bodies.append({
            "id": body, "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "parent_id": int(model.body_parentid[body]), "weld_id": int(model.body_weldid[body]),
            "subtree": "climber" if body in id_set else "fixed_environment",
            "mass": float(model.body_mass[body]), "inertia_principal": model.body_inertia[body].tolist(),
            "inertia_quaternion": model.body_iquat[body].tolist(),
            "inertia_body_tensor": _inertia_tensor(model, body).tolist(),
            "com_local": model.body_ipos[body].tolist(), "com_world_neutral": data.xipos[body].tolist(),
            "position_local": model.body_pos[body].tolist(),
        })
    groups = {"torso": [], "head_neck": [], "left_arm": [], "right_arm": [],
              "left_leg": [], "right_leg": []}
    for body in ids:
        name = bodies[body]["name"]
        if name in ("climber_root", "pelvis", "abdomen", "chest"):
            group = "torso"
        elif name == "head":
            group = "head_neck"
        else:
            side = name.split("_", 1)[0]
            group = f"{side}_{'arm' if any(s in name for s in ('arm', 'hand')) else 'leg'}"
        groups[group].append(int(body))
    group_summary = {}
    for name, members in groups.items():
        mass = float(np.sum(model.body_mass[members]))
        com = np.sum(model.body_mass[members, None] * data.xipos[members], axis=0) / mass
        _require_finite(com, f"{name} group COM")
        group_summary[name] = {"body_ids": members, "mass": mass, "com_world_neutral": com.tolist()}
    weld_groups = [{"weld_id": int(weld), "body_ids": ids[model.body_weldid[ids] == weld].tolist(),
                    "mass": float(np.sum(model.body_mass[ids[model.body_weldid[ids] == weld]]))}
                   for weld in np.unique(model.body_weldid[ids])]
    joints = [{"id": joint, "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
               "body_id": int(model.jnt_bodyid[joint]), "type": int(model.jnt_type[joint]),
               "axis_local": model.jnt_axis[joint].tolist(), "position_local": model.jnt_pos[joint].tolist(),
               "qpos_address": int(model.jnt_qposadr[joint]), "dof_address": int(model.jnt_dofadr[joint]),
               "limited": bool(model.jnt_limited[joint]), "range_rad": model.jnt_range[joint].tolist(),
               "range_deg": np.rad2deg(model.jnt_range[joint]).tolist(),
               "damping": float(model.dof_damping[model.jnt_dofadr[joint]])}
              for joint in range(model.njnt)]
    charts = []
    for body in ids:
        ordered = [joint for joint in joints if joint["body_id"] == body]
        if len(ordered) == 3:
            middle = ordered[1]
            lo, hi = middle["range_rad"]
            singularities = [angle for angle in (-np.pi / 2, np.pi / 2) if lo <= angle <= hi]
            jacobian_samples = []
            probe = mujoco.MjData(model)
            probe.eq_active[:] = 0
            for limit in (lo, hi):
                angle = limit * (1 - 1e-6)
                probe.qpos[:] = model.qpos0
                probe.qpos[middle["qpos_address"]] = angle
                mujoco.mj_forward(model, probe)
                _require_finite_sample(probe)
                rotation_jacobian = np.empty((3, model.nv))
                mujoco.mj_jacBody(model, probe, None, rotation_jacobian, int(body))
                _require_finite(rotation_jacobian, "rotation Jacobian")
                singular_values = np.linalg.svd(rotation_jacobian[:, [j["dof_address"] for j in ordered]], compute_uv=False)
                _require_finite(singular_values, "rotation Jacobian singular values")
                jacobian_samples.append({"middle_angle_rad": float(angle),
                                         "singular_values": singular_values.tolist()})
            charts.append({"body": bodies[body]["name"], "joint_order": [j["name"] for j in ordered],
                           "middle_joint": middle["name"], "singular_angles_within_rom_rad": singularities,
                           "rotation_jacobian_near_limits": jacobian_samples,
                           "note": "Three serial hinge coordinates have a chart singularity at middle angle +/-pi/2. "
                                   "The supported prototype ROM avoids these; shoulder roll scales from +/-80 deg "
                                   "then caps at +/-85 deg. This is a technical chart policy, not biological ROM."})
    actuators = [{"id": act, "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act),
                  "joint_id": int(model.actuator_trnid[act, 0]),
                  "joint_name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[act, 0]),
                  "gear": model.actuator_gear[act].tolist(),
                  "ctrl_range": model.actuator_ctrlrange[act].tolist(),
                  "ctrl_limited": bool(model.actuator_ctrllimited[act]),
                  "force_range": model.actuator_forcerange[act].tolist(),
                  "force_limited": bool(model.actuator_forcelimited[act])}
                 for act in range(model.nu)]
    equalities = [{"id": eq, "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq),
                   "type": int(model.eq_type[eq]), "object_type": int(model.eq_objtype[eq]),
                   "object_ids": [int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])],
                   "active_by_default": bool(model.eq_active0[eq]), "parameters": model.eq_data[eq].tolist()}
                  for eq in range(model.neq)]
    sites = [{"id": site, "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, site),
              "body_id": int(model.site_bodyid[site]), "position_local": model.site_pos[site].tolist(),
              "position_world_neutral": data.site_xpos[site].tolist()}
             for site in range(model.nsite)]
    foot_sites = {site["id"] for site in sites if bodies[site["body_id"]]["name"] in ("left_foot", "right_foot")}
    geoms = [{"id": geom, "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
              "body_id": int(model.geom_bodyid[geom]), "type": int(model.geom_type[geom]),
              "size": model.geom_size[geom].tolist(), "position_local": model.geom_pos[geom].tolist(),
              "quaternion_local": model.geom_quat[geom].tolist(),
              "contype": int(model.geom_contype[geom]), "conaffinity": int(model.geom_conaffinity[geom]),
              "friction": model.geom_friction[geom].tolist(),
              "invisible": bool(model.geom_rgba[geom, 3] == 0)} for geom in range(model.ngeom)]
    exclusions = []
    contact_mode_flag = float(model.numeric("contact_mode").data[0])
    if contact_mode_flag not in (0., 1.):
        raise ValueError("Invalid compiled contact mode")
    contact_mode = (ContactMode.PHYSICAL if contact_mode_flag == 0
                    else ContactMode.IDEALIZED_DEBUG)
    for signature in model.exclude_signature:
        signature = int(signature)
        first, second = signature >> 16, signature & 0xffff
        adjacent = (model.body_parentid[first] == second or model.body_parentid[second] == first)
        welded = model.body_weldid[first] == model.body_weldid[second]
        self_pair = first in id_set and second in id_set
        exclusions.append({
            "signature": signature, "body_ids": [first, second],
            "body_names": [bodies[first]["name"], bodies[second]["name"]],
            "classification": ("adjacent_overlap_policy" if self_pair and (adjacent or welded)
                               else "temporary_environment_workaround_stage2_debt" if not self_pair
                               else "unclassified_self_exclusion"),
            "purpose": "necessary articulated-adjacency overlap policy" if self_pair else "temporary environment handover workaround",
            "redundant_default_parent_filter": bool(adjacent and not welded),
            "redundant_weld_filter": bool(welded),
        })
    carrying = ids[model.body_mass[ids] > 0]
    inertia = model.body_inertia[carrying]
    eigenvalues = np.linalg.eigvalsh(mass_matrix(model, data))
    _require_finite(eigenvalues, "mass matrix eigenvalues")
    proxy = _proxy_inertia_errors(model)
    hinge_names = [j["name"] for j in joints if j["type"] == int(mujoco.mjtJoint.mjJNT_HINGE)]
    mirror_errors = []
    for body in ids:
        name = bodies[body]["name"]
        if name.startswith("left_"):
            other = _id(model, mujoco.mjtObj.mjOBJ_BODY, name.replace("left_", "right_", 1))
            mirror_errors.extend([abs(model.body_mass[body] - model.body_mass[other]),
                                  np.max(np.abs(data.xipos[other] - _REFLECT @ data.xipos[body])),
                                  np.max(np.abs(_inertia_tensor(model, other)
                                                - _REFLECT @ _inertia_tensor(model, body) @ _REFLECT))])
    collision_count = sum(bool(g["contype"] or g["conaffinity"]) for g in geoms if g["body_id"] in id_set)
    expected_masses = {"climber_root": 0, "pelvis": 10, "abdomen": 12, "chest": 18, "head": 5}
    expected_masses.update({f"{side}_{segment}": mass for side in ("left", "right")
                            for segment, mass in (("upper_arm", 2.4), ("forearm", 1.5), ("hand", 0.55),
                                                  ("thigh", 7.5), ("shin", 3.5), ("foot", 1.2))})
    checks = {
        "compiled_numeric_finite": True,
        "topology_32_31_25_one_free": bool((model.nq, model.nv, model.nu) == (32, 31, 25)
                                          and sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
                                          and model.njnt == 26),
        "joint_order": hinge_names == [s[0] for s in JOINT_SPECS],
        "actuator_order_and_joint_map": ([a["name"] for a in actuators] == [f"act_{s[0]}" for s in JOINT_SPECS]
                                         and [a["joint_name"] for a in actuators] == hinge_names),
        "finite_legal_neutral_ranges": all(j["limited"] and np.isfinite(j["range_rad"]).all()
                                            and j["range_rad"][0] < j["range_rad"][1]
                                            and j["range_rad"][0] <= 0 <= j["range_rad"][1]
                                            for j in joints if j["type"] == int(mujoco.mjtJoint.mjJNT_HINGE)),
        "climber_mass_78_3_scaled": bool(abs(total - 78.3 * mass_scale) < 1e-10),
        "fixed_segment_mass_fractions": (set(expected_masses) == {bodies[body]["name"] for body in ids}
            and all(abs(bodies[body]["mass"] - expected_masses[bodies[body]["name"]] * mass_scale) < 1e-12 for body in ids)),
        "massless_root_positive_pelvis_weld": bool(model.body_mass[root] == 0
            and model.body_weldid[root] == model.body_weldid[_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]),
        "mass_carrying_inertias_positive_triangle": bool(np.isfinite(inertia).all()
            and (inertia > 0).all() and (2 * np.max(inertia, axis=1) <= np.sum(inertia, axis=1) + 1e-12).all()),
        "positive_moving_weld_groups": all(g["weld_id"] > 0 and g["mass"] > 0 for g in weld_groups),
        "mass_matrix_spd": bool(np.isfinite(eigenvalues).all() and eigenvalues[0] > 0),
        "weighted_com_matches_subtree": bool(np.max(np.abs(weighted_com - data.subtree_com[root])) < 1e-12),
        "finite_com_within_primary_proxy_bounds": bool(np.isfinite(weighted_com).all() and np.isfinite(proxy_bounds).all()
            and (weighted_com >= proxy_bounds[0]).all() and (weighted_com <= proxy_bounds[1]).all()),
        "neutral_com_on_mirror_plane": mirror_center_error < 1e-12,
        "coordinate_charts_no_singularities": all(not c["singular_angles_within_rom_rad"] for c in charts),
        "rotation_chart_jacobians_nonsingular_near_limits": all(min(s["singular_values"]) > 0.06
            for c in charts for s in c["rotation_jacobian_near_limits"]),
        "primary_ellipsoid_inertias": max(proxy["ellipsoid_tensor_errors"].values()) < 1e-12,
        "primary_capsule_inertias": all(max(v["tensor_max_error"], v["com_max_error"]) < 1e-12
                                         for v in proxy["capsule_errors"].values()),
        "bilateral_mass_com_inertia": bool(max(mirror_errors) < 1e-12),
        "humanoid_geom_count_64": sum(g["body_id"] in id_set for g in geoms) == 64,
        "humanoid_collision_geom_count_21": collision_count == 21,
        "neutral_no_contacts": data.ncon == 0,
        "default_equalities_inactive": not bool(np.any(model.eq_active0)),
        "classified_exclusions": all(e["classification"] != "unclassified_self_exclusion" for e in exclusions),
        "physical_contact_policy": (contact_mode != ContactMode.PHYSICAL
            or (all(e["classification"] == "adjacent_overlap_policy" for e in exclusions)
                and all(e["object_type"] != int(mujoco.mjtObj.mjOBJ_SITE)
                        or foot_sites.isdisjoint(e["object_ids"]) for e in equalities))),
    }
    return {
        "topology": {key: int(getattr(model, key)) for key in
                     ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nsite", "neq", "nexclude")},
        "climber": {"root_body_id": root, "body_ids": ids.tolist(), "mass": total,
                    "com_world_neutral": weighted_com.tolist(), "groups": group_summary,
                    "primary_proxy_bounds_world_neutral": proxy_bounds.tolist(),
                    "mirror_plane_x": float(data.xpos[root, 0]), "com_mirror_center_error": mirror_center_error,
                    "moving_weld_groups": weld_groups},
        "fixed_environment": {"body_ids": env.tolist(), "mass": environment_mass,
            "com_world": (np.sum(model.body_mass[env, None] * data.xipos[env], axis=0)
                          / environment_mass).tolist() if environment_mass else None},
        "mass_matrix": {"minimum_eigenvalue": float(eigenvalues[0]),
                        "maximum_eigenvalue": float(eigenvalues[-1]),
                        "dof_armature": model.dof_armature.tolist(),
                        "armature_units": ["kg" if i < 3 else "kg*m^2" for i in range(model.nv)]},
        "units": {"mass": "kg", "inertia": "kg*m^2", "position": "m", "joint_angle": "rad",
                  "motor_ctrl_and_force_range": "dimensionless normalized motor output (force before gear)",
                  "motor_gear": "N*m per normalized output", "hinge_actuator_torque": "N*m"},
        "bodies": bodies, "joints": joints, "coordinate_charts": charts, "actuators": actuators,
        "equalities": equalities, "sites": sites,
        "geometries": geoms, "exclusions": exclusions,
        "collision_policy": {"contact_mode": contact_mode.value, "humanoid_enabled_geometries": collision_count,
            "neutral_contacts": [{"geom_ids": [int(c.geom1), int(c.geom2)], "distance": float(c.dist)}
                                 for c in data.contact],
            "adjacent_overlap_exclusions": sum(e["classification"] == "adjacent_overlap_policy" for e in exclusions),
            "temporary_environment_exclusions": sum(e["classification"].startswith("temporary_") for e in exclusions),
            "note": "Primary visual capsules define inertia; invisible forearm/shin envelopes only define collision. "
                    "Adjacent overlap exclusions are redundant with default parent/weld filters. "
                    + ("Physical mode requires no environment exclusions or foot equalities."
                        if contact_mode == ContactMode.PHYSICAL else
                        "Idealized debug retains legacy environment exclusion debt and foot equalities; not physical support.")},
        "inertia_proxies": proxy, "checks": checks, "passed": all(checks.values()),
    }


def measure_joint_semantics(model, angle=0.05) -> dict:
    """Signed forward-only neutral probes, including diagnostic out-of-ROM poses."""
    if not np.isfinite(angle) or angle <= 0:
        raise ValueError("probe angle must be finite and positive")
    model, neutral = _isolated_state(model, contacts=False, gravity=(0, 0, 0))
    _require_finite_sample(neutral)
    measurements = []
    for name, body_name, site_name, expected_axis, _ in JOINT_SPECS:
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        site = _id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        axis = np.array(expected_axis, dtype=float)
        probe = np.array([0.1, 0, 0]) if axis[2] else np.array([0, 0, -0.1])
        samples = []
        for signed_angle in (angle, -angle):
            data = mujoco.MjData(model)
            data.qpos[:] = neutral.qpos
            data.eq_active[:] = 0
            data.qpos[model.jnt_qposadr[joint]] = signed_angle
            mujoco.mj_forward(model, data)  # Never step the negative elbow/knee diagnostic.
            _require_finite_sample(data)
            rotation = data.xmat[body].reshape(3, 3)
            rotation_delta = rotation @ neutral.xmat[body].reshape(3, 3).T
            sine_vector = np.array([rotation_delta[2, 1] - rotation_delta[1, 2],
                                    rotation_delta[0, 2] - rotation_delta[2, 0],
                                    rotation_delta[1, 0] - rotation_delta[0, 1]]) / 2
            measured_angle = float(np.arctan2(axis @ sine_vector, (np.trace(rotation_delta) - 1) / 2))
            _require_finite(measured_angle, "measured joint angle")
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
            expected_rotation = np.eye(3) + np.sin(signed_angle) * skew + (1 - np.cos(signed_angle)) * (skew @ skew)
            site_delta = data.site_xpos[site] - neutral.site_xpos[site]
            probe_delta = rotation @ probe - probe
            _require_finite(site_delta, "joint site displacement")
            _require_finite(probe_delta, "joint off-axis displacement")
            # World conventions are independently specified, not derived from jnt_axis.
            if name == "waist_pitch":
                site_direction = (1, 1)
                description = "+X waist pitch moves upright chest/head top toward -Y (away from wall); the below-axis hand site moves +Y"
            elif axis[0]:
                site_direction = (1, int(axis[0]))
                description = "+X pitch moves the downward hand/foot site +Y; the knee's -X axis moves it -Y"
            elif axis[1]:
                site_direction = (0, -1)
                description = "+Y roll moves the below-axis observed hand/foot site -X"
            elif "shoulder" in name:
                site_direction = None
                description = "+Z yaw rotates local +X toward +Y; straight centerline is stationary"
            elif "hip" in name:
                site_direction = (0, -1)
                description = "+Z yaw rotates the forward (+Y) foot site toward -X"
            else:
                site_direction = (1, -1)
                description = "+Z waist yaw rotates the left (-X) hand toward -Y"
            direction_ok = (_require_finite(np.linalg.norm(site_delta), "joint site displacement norm") < 1e-12 if site_direction is None else
                            site_delta[site_direction[0]] * site_direction[1] * np.sign(signed_angle) > 1e-7)
            probe_component, probe_sign = ((1, 1) if axis[2] else (1, int(axis[0])) if axis[0] else (0, -1))
            samples.append({
                "angle_rad": float(signed_angle), "measured_body_angle_rad": measured_angle,
                "within_rom": bool(model.jnt_range[joint, 0]
                    <= signed_angle <= model.jnt_range[joint, 1]),
                "body_rotation_world": rotation.tolist(), "site_rotation_world": data.site_xmat[site].reshape(3, 3).tolist(),
                "site_position_world": data.site_xpos[site].tolist(), "site_delta_world": site_delta.tolist(),
                "off_axis_probe_local": probe.tolist(), "off_axis_probe_delta_world": probe_delta.tolist(),
                "rotation_error": _require_finite(float(np.max(np.abs(rotation - expected_rotation))), "joint rotation error"),
                "site_rotation_error": _require_finite(float(np.max(np.abs(data.site_xmat[site].reshape(3, 3) - expected_rotation))), "joint site rotation error"),
                "direction_passed": bool(direction_ok and probe_delta[probe_component] * probe_sign * np.sign(signed_angle) > 1e-7),
            })
            if name == "waist_pitch":
                chest = _id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
                head = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")
                head_tops = [state.geom_xpos[head] + state.geom_xmat[head].reshape(3, 3)
                             @ [0, 0, model.geom_size[head, 2]] for state in (neutral, data)]
                chest_delta = data.xpos[chest] - neutral.xpos[chest]
                head_delta = head_tops[1] - head_tops[0]
                _require_finite(chest_delta, "chest displacement")
                _require_finite(head_tops, "head top positions")
                _require_finite(head_delta, "head top displacement")
                samples[-1].update({"chest_delta_world": chest_delta.tolist(),
                                    "head_top_position_world": head_tops[1].tolist(),
                                    "head_top_delta_world": head_delta.tolist()})
                samples[-1]["direction_passed"] &= bool(chest_delta[1] * np.sign(signed_angle) < -1e-7
                                                       and head_delta[1] * np.sign(signed_angle) < -1e-7)
        measurements.append({"joint": name, "body": body_name, "site": site_name,
                             "compiled_axis": model.jnt_axis[joint].tolist(), "expected_axis": list(expected_axis),
                             "positive_semantics": description, "samples": samples})
    checks = {
        "all_25_axes": all(np.array_equal(m["compiled_axis"], m["expected_axis"]) for m in measurements),
        "signed_body_site_rotation": all(s["rotation_error"] < 1e-12 and s["site_rotation_error"] < 1e-12
                                         for m in measurements for s in m["samples"]),
        "independent_signed_directions": all(s["direction_passed"] for m in measurements for s in m["samples"]),
    }
    return {"mode": "kinematic_only_mj_forward_no_steps", "angle_rad": float(angle),
            "note": "Negative elbow/knee angles are intentionally outside ROM; they diagnose sign only, never dynamics.",
            "measurements": measurements, "checks": checks, "passed": all(checks.values())}


def _warnings(data):
    return {name: int(data.warning[int(value)].number) for name, value in mujoco.mjtWarning.__members__.items()
            if name != "mjNWARNING" and data.warning[int(value)].number}


def _unforced(data):
    return bool(not np.any(data.eq_active) and not np.any(data.ctrl)
                and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
                and not np.any(data.qfrc_actuator))


def freefall_diagnostic(model, duration=0.4, *, timestep=None) -> dict:
    """Contact-enabled ballistic COM at altitude, with explicit timestep error."""
    timestep = model.opt.timestep if timestep is None else timestep
    steps = _integration_steps(duration, timestep)
    model, data = _isolated_state(model)
    _require_finite_sample(data)
    model.opt.timestep = timestep
    root_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    address = int(model.jnt_qposadr[root_joint])
    dof = int(model.jnt_dofadr[root_joint])
    data.qpos[address:address + 3] = [0, -5, 20]
    mujoco.mj_forward(model, data)
    _require_finite_sample(data)
    root = int(climber_body_ids(model)[0])
    initial_com = data.subtree_com[root].copy()
    initial_qpos = data.qpos.copy()
    max_contacts, max_internal_speed = data.ncon, float(np.max(np.abs(data.qvel[dof + 3:])))
    unforced = _unforced(data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        _require_finite_sample(data)
        max_contacts = max(max_contacts, data.ncon)
        max_internal_speed = max(max_internal_speed, float(np.max(np.abs(data.qvel[dof + 3:]))))
        unforced &= _unforced(data)
    mujoco.mj_forward(model, data)
    _require_finite_sample(data)
    max_internal_speed = max(max_internal_speed, float(np.max(np.abs(data.qvel[dof + 3:]))))
    max_contacts = max(max_contacts, data.ncon)
    unforced &= _unforced(data)
    time = float(data.time)
    analytic_com = initial_com + model.opt.gravity * time**2 / 2
    discrete_com = initial_com + model.opt.gravity * time * (time + model.opt.timestep) / 2
    position_error = float(np.linalg.norm(data.subtree_com[root] - analytic_com))
    discrete_error = float(np.linalg.norm(data.subtree_com[root] - discrete_com))
    velocity_error = float(np.linalg.norm(data.qvel[dof:dof + 3] - model.opt.gravity * time))
    tolerance = float(np.linalg.norm(model.opt.gravity) * time * model.opt.timestep / 2 + 1e-9)
    _require_finite([position_error, discrete_error, velocity_error, tolerance], "freefall comparison errors")
    checks = {"unforced": unforced, "zero_contacts": max_contacts == 0,
              "ballistic_velocity": velocity_error < 1e-9, "ballistic_com": position_error <= tolerance,
              "semi_implicit_discrete_com": discrete_error < 1e-9,
              "no_internal_motion": max_internal_speed < 1e-9,
              "no_warnings": not _warnings(data), "elapsed_time": abs(time - steps * model.opt.timestep) < 1e-10}
    return {"duration": time, "timestep": float(model.opt.timestep), "gravity": model.opt.gravity.tolist(),
            "initial_root_position": initial_qpos[address:address + 3].tolist(),
            "initial_com": initial_com.tolist(), "final_com": data.subtree_com[root].tolist(),
            "analytic_com": analytic_com.tolist(), "position_error": position_error,
            "position_bias": (data.subtree_com[root] - analytic_com).tolist(),
            "position_tolerance": tolerance, "discrete_position_error": discrete_error,
            "velocity_error": velocity_error, "max_contacts": int(max_contacts),
            "max_internal_speed": max_internal_speed, "warnings": _warnings(data),
            "checks": checks, "passed": all(checks.values())}


def passive_decay_diagnostic(model, *, joints=None, timestep=0.002, duration=0.2, speed=1e-5) -> dict:
    """Tiny-velocity damping against a full coupled constant-M exponential.

    M^-1 D is not symmetric. Diagonalize M^-1/2 D M^-1/2 instead, including
    the undamped free-root modes, using only NumPy. Joint limits remain enabled
    at valid ROM midpoints; contact alone is disabled with mjDSBL_CONTACT.
    """
    steps = _integration_steps(duration, timestep)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("initial velocity speed must be finite and positive")
    selected = [s[0] for s in JOINT_SPECS] if joints is None else list(joints)
    if not selected:
        raise ValueError("at least one passive joint is required")
    model, data = _isolated_state(model, contacts=False, gravity=(0, 0, 0), midpoint=True)
    _require_finite_sample(data)
    model.opt.timestep = timestep
    for index, name in enumerate(selected):
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("passive velocity probes require hinge joints")
        data.qvel[model.jnt_dofadr[joint]] = speed * np.cos(index + 0.3)
    mujoco.mj_forward(model, data)
    _require_finite_sample(data)
    initial_qpos, initial_velocity = data.qpos.copy(), data.qvel.copy()
    inertia = mass_matrix(model, data)
    initial_energy = float(initial_velocity @ inertia @ initial_velocity / 2)
    scale = float(np.linalg.norm(initial_velocity))
    if not np.isfinite(initial_energy) or initial_energy <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("passive probe requires finite positive initial energy and velocity norm")
    values, vectors = np.linalg.eigh(inertia)
    sqrt_m = (vectors * np.sqrt(values)) @ vectors.T
    inverse_sqrt_m = (vectors / np.sqrt(values)) @ vectors.T
    damping = np.diag(model.dof_damping)
    rates, modes = np.linalg.eigh(inverse_sqrt_m @ damping @ inverse_sqrt_m)
    rates = np.maximum(rates, 0)
    transformed = modes.T @ sqrt_m @ initial_velocity
    time = steps * timestep
    reference = inverse_sqrt_m @ modes @ (np.exp(-rates * time) * transformed)
    discrete_reference = inverse_sqrt_m @ modes @ ((1 + timestep * rates)**(-steps) * transformed)
    _require_finite(reference, "passive velocity reference")
    _require_finite(discrete_reference, "passive discrete velocity reference")
    _require_finite(rates, "passive damping rates")
    energies = [initial_energy]
    unforced, contacts, constraints = _unforced(data), data.ncon, data.nefc
    valid_rom = True
    for _ in range(steps):
        mujoco.mj_step(model, data)
        _require_finite_sample(data)
        mujoco.mj_forward(model, data)
        _require_finite_sample(data)
        energies.append(_require_finite(float(data.qvel @ mass_matrix(model, data) @ data.qvel / 2), "passive kinetic energy"))
        contacts = max(contacts, data.ncon)
        constraints = max(constraints, data.nefc)
        unforced &= _unforced(data)
        for joint in range(model.njnt):
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                valid_rom &= bool(model.jnt_range[joint, 0] < data.qpos[model.jnt_qposadr[joint]] < model.jnt_range[joint, 1])
    reference_error = float(np.linalg.norm(data.qvel - reference) / scale)
    discrete_error = float(np.linalg.norm(data.qvel - discrete_reference) / scale)
    reference_energy = float(reference @ inertia @ reference / 2)
    _require_finite([reference_error, discrete_error, reference_energy, energies[-1] / initial_energy,
                     reference_energy / initial_energy, abs(energies[-1] - reference_energy) / initial_energy],
                    "passive comparison errors and energy ratios")
    checks = {"unforced": unforced, "contacts_disabled_zero": contacts == 0,
              "joint_limits_enabled_interior": bool(valid_rom and not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_LIMIT)),
              "no_constraint_forces": constraints == 0,
              "energy_decreased": energies[-1] < initial_energy,
              "energy_monotone": bool(np.max(np.diff(energies)) <= initial_energy * 1e-8),
              "coupled_exponential_reference": reference_error < 0.03,
              "reference_energy_decay": abs(energies[-1] - reference_energy) / initial_energy < 0.01,
              "implicit_discrete_reference": discrete_error < 5e-4,
              "finite": True,
              "no_warnings": not _warnings(data), "elapsed_time": abs(data.time - time) < 1e-10}
    return {"joints": selected, "timestep": timestep, "duration": time, "initial_velocity_scale": speed,
            "initial_qpos": initial_qpos.tolist(), "initial_qvel": initial_velocity.tolist(),
            "final_qvel": data.qvel.tolist(), "reference_qvel": reference.tolist(),
            "initial_energy": initial_energy, "final_energy": energies[-1],
            "reference_final_energy": reference_energy, "energy_ratio": energies[-1] / initial_energy,
            "reference_energy_ratio": reference_energy / initial_energy,
            "relative_velocity_reference_error": reference_error,
            "relative_velocity_discrete_error": discrete_error,
            "damping_rates": rates.tolist(), "max_contacts": int(contacts),
            "max_constraints": int(constraints), "warnings": _warnings(data),
            "checks": checks, "passed": all(checks.values())}


def mirror_state(model, qpos, qvel):
    """Reflect X, swap limbs, and reflect axial vectors (X same; Y/Z negative)."""
    reflected_pos, reflected_vel = qpos.copy(), qvel.copy()
    root = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    address, dof = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    reflected_pos[address:address + 3] = _REFLECT @ qpos[address:address + 3]
    reflected_pos[address + 3:address + 7] = qpos[address + 3:address + 7] * [1, 1, -1, -1]
    reflected_vel[dof:dof + 3] = _REFLECT @ qvel[dof:dof + 3]
    reflected_vel[dof + 3:dof + 6] = _AXIAL @ qvel[dof + 3:dof + 6]
    for name, _, _, axis, _ in JOINT_SPECS:
        other = name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
        source = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        target = _id(model, mujoco.mjtObj.mjOBJ_JOINT, other)
        sign = 1 if axis[0] else -1
        reflected_pos[model.jnt_qposadr[target]] = sign * qpos[model.jnt_qposadr[source]]
        reflected_vel[model.jnt_dofadr[target]] = sign * qvel[model.jnt_dofadr[source]]
    return reflected_pos, reflected_vel


def mirror_diagnostic(model, duration=0.2) -> dict:
    steps = _integration_steps(duration, model.opt.timestep)
    model, data = _isolated_state(model, contacts=False, gravity=(0, 0, 0), midpoint=True)
    _require_finite_sample(data)
    for index, spec in enumerate(JOINT_SPECS):
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, spec[0])
        lo, hi = model.jnt_range[joint]
        data.qpos[model.jnt_qposadr[joint]] += 0.08 * (hi - lo) * np.sin(index + 0.4)
        data.qvel[model.jnt_dofadr[joint]] = 0.01 * np.cos(index + 0.2)
    root = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_address, root_dof = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    data.qpos[root_address:root_address + 3] = [0.2, -5, 20]
    data.qpos[root_address + 3:root_address + 7] = [1, 0, 0, 0]
    data.qvel[root_dof:root_dof + 6] = [0.02, -0.03, 0.01, 0.01, -0.02, 0.03]
    mirrored = mujoco.MjData(model)
    mirrored.eq_active[:] = 0
    mirrored.qpos[:], mirrored.qvel[:] = mirror_state(model, data.qpos, data.qvel)
    transform = np.column_stack([mirror_state(model, data.qpos, unit)[1] for unit in np.eye(model.nv)])
    body_error, rotation_error, site_error, velocity_error, acceleration_error, matrix_error = [0.0] * 6
    unforced = True
    max_contacts, max_constraints = 0, 0
    for step in range(steps + 1):
        mujoco.mj_forward(model, data)
        _require_finite_sample(data)
        mujoco.mj_forward(model, mirrored)
        _require_finite_sample(mirrored)
        for body in climber_body_ids(model):
            name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            other = name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
            target = _id(model, mujoco.mjtObj.mjOBJ_BODY, other)
            body_error = max(body_error, _require_finite(float(np.max(np.abs(mirrored.xpos[target] - _REFLECT @ data.xpos[body]))), "mirror body position error"))
            rotation_error = max(rotation_error, _require_finite(float(np.max(np.abs(mirrored.xmat[target].reshape(3, 3)
                - _REFLECT @ data.xmat[body].reshape(3, 3) @ _REFLECT))), "mirror body rotation error"))
        for side in ("left", "right"):
            for limb in ("hand", "foot"):
                source = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{limb}_site")
                target = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{'right' if side == 'left' else 'left'}_{limb}_site")
                site_error = max(site_error, _require_finite(float(np.max(np.abs(mirrored.site_xpos[target] - _REFLECT @ data.site_xpos[source]))), "mirror site position error"))
        velocity_error = max(velocity_error, _require_finite(float(np.max(np.abs(mirrored.qvel - transform @ data.qvel))), "mirror velocity error"))
        acceleration_error = max(acceleration_error, _require_finite(float(np.max(np.abs(mirrored.qacc - transform @ data.qacc))), "mirror acceleration error"))
        if step == 0:
            matrix_error = float(np.max(np.abs(mass_matrix(model, mirrored) - transform @ mass_matrix(model, data) @ transform.T)))
            _require_finite(matrix_error, "mirror mass matrix error")
        unforced &= _unforced(data) and _unforced(mirrored)
        max_contacts = max(max_contacts, data.ncon, mirrored.ncon)
        max_constraints = max(max_constraints, data.nefc, mirrored.nefc)
        if step < steps:
            mujoco.mj_step(model, data)
            _require_finite_sample(data)
            mujoco.mj_step(model, mirrored)
            _require_finite_sample(mirrored)
    expected_pos, _ = mirror_state(model, data.qpos, data.qvel)
    position_difference = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, position_difference, 1, expected_pos, mirrored.qpos)
    _require_finite(position_difference, "mirror tangent position error")
    checks = {"unforced": unforced, "no_contacts_constraints": max_contacts == max_constraints == 0,
              "body_site_kinematics": max(body_error, rotation_error, site_error) < 1e-10,
              "coupled_mass_matrix": matrix_error < 1e-10,
              "physical_dynamics": max(velocity_error, acceleration_error, float(np.max(np.abs(position_difference)))) < 1e-9,
              "no_warnings": not _warnings(data) and not _warnings(mirrored)}
    return {"duration": float(data.time), "root_initial_orientation": [1, 0, 0, 0],
            "reflection": "polar X negative; axial X same, Y/Z negative; swap left/right",
            "body_position_error": body_error, "body_rotation_error": rotation_error,
            "site_position_error": site_error, "velocity_error": velocity_error,
            "acceleration_error": acceleration_error, "mass_matrix_error": matrix_error,
            "qpos_tangent_error": float(np.max(np.abs(position_difference))),
            "checks": checks, "passed": all(checks.values())}


def actuator_diagnostic(model, *, strength_scale=1.0) -> dict:
    """All 25 direct motors: force=clipped ctrl, torque=force*known gear."""
    numerics = _compiled_numerics(model)
    if not numerics["passed"]:
        return {**numerics, "samples": [], "maximum_error": None, "nonfinite_observations": []}
    if not np.isfinite(strength_scale) or not 1e-6 <= strength_scale <= 10:
        raise ValueError("supported diagnostic strength_scale is [1e-6, 10]")
    model, data = _isolated_state(model, contacts=False, gravity=(0, 0, 0), midpoint=True)
    samples, mapping_ok, max_error = [], True, 0.0
    nonfinite_observations = []

    def nonfinite_outputs():
        fields = {}
        for name in ("actuator_force", "qfrc_actuator", "qacc"):
            indices = np.flatnonzero(~np.isfinite(getattr(data, name))).tolist()
            if indices:
                fields[name] = {"indices": indices, "values": [None] * len(indices)}
        return fields

    initial_faults = nonfinite_outputs()
    if initial_faults:
        nonfinite_observations.append({"joint": None, "ctrl": 0, "fields": initial_faults})
    for name, _, _, _, torque in JOINT_SPECS:
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
        dof = int(model.jnt_dofadr[joint])
        expected_gear = torque * strength_scale
        mapping_ok &= bool(model.actuator_trnid[actuator, 0] == joint
                           and model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                           and model.actuator_ctrllimited[actuator]
                           and np.array_equal(model.actuator_ctrlrange[actuator], [-1, 1])
                           and np.max(np.abs(model.actuator_gear[actuator] - [expected_gear, 0, 0, 0, 0, 0])) < 1e-12)
        for control in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
            data.ctrl[:] = 0
            data.ctrl[actuator] = control  # Deliberately not pre-clamped by Python.
            mujoco.mj_forward(model, data)
            expected_force = np.zeros(model.nu)
            expected_force[actuator] = np.clip(control, -1, 1)
            expected_qfrc = np.zeros(model.nv)
            expected_qfrc[dof] = expected_force[actuator] * expected_gear
            faults = nonfinite_outputs()
            error = None
            if faults:
                nonfinite_observations.append({"joint": name, "ctrl": control, "fields": faults})
            else:
                error = float(max(np.max(np.abs(data.actuator_force - expected_force)),
                                  np.max(np.abs(data.qfrc_actuator - expected_qfrc))))
                max_error = max(max_error, error)
            samples.append({"joint": name, "actuator_id": actuator, "dof_address": dof,
                            "ctrl": control, "clipped_ctrl": float(expected_force[actuator]),
                            "expected_gear": expected_gear,
                            "actuator_force": float(data.actuator_force[actuator]) if np.isfinite(data.actuator_force[actuator]) else None,
                            "qfrc_actuator": float(data.qfrc_actuator[dof]) if np.isfinite(data.qfrc_actuator[dof]) else None,
                            "full_vector_error": error})
    checks = {"compiled_numeric_finite": True, "finite_forward_outputs": not nonfinite_observations,
              "mapping_and_known_gears": mapping_ok,
              "all_150_force_torque_clamps": not nonfinite_observations and max_error < 1e-12,
              "no_equalities_external_forces_contacts": bool(not np.any(data.eq_active)
                  and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied) and data.ncon == 0),
              "no_warnings": not _warnings(data)}
    return {"strength_scale": strength_scale, "samples": samples,
            "maximum_error": None if nonfinite_observations else max_error,
            "nonfinite_observations": nonfinite_observations,
            "checks": checks, "passed": all(checks.values())}


def neutral_rollout_diagnostic(model, duration=2.0) -> dict:
    """Uncontrolled, collision-enabled floor impacts, not balance/stability."""
    steps = _integration_steps(duration, model.opt.timestep)
    model, data = _isolated_state(model)
    _require_finite_sample(data)
    floor = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    max_speed, max_acceleration, max_contacts, floor_steps = 0.0, 0.0, data.ncon, 0
    unforced = _unforced(data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        _require_finite_sample(data)
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))
        max_acceleration = max(max_acceleration, float(np.max(np.abs(data.qacc))))
        max_contacts = max(max_contacts, data.ncon)
        floor_steps += any(c.geom1 == floor or c.geom2 == floor for c in data.contact)
        unforced &= _unforced(data)
    checks = {"unforced": unforced, "finite": True, "floor_impacts_observed": floor_steps > 0,
              "no_warnings": not _warnings(data), "bounded_speed_acceleration": max_speed < 100 and max_acceleration < 1e6,
              "elapsed_time": abs(data.time - steps * model.opt.timestep) < 1e-10,
              "collisions_enabled": not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))}
    return {"duration": float(data.time), "steps": steps, "floor_contact_steps": int(floor_steps),
            "max_contacts": int(max_contacts), "max_abs_qvel": max_speed, "max_abs_qacc": max_acceleration,
            "thresholds": {"max_abs_qvel": 100, "max_abs_qacc": 1e6, "warning_count": 0},
            "warnings": _warnings(data), "checks": checks, "passed": all(checks.values())}


def profile_diagnostics(scene: BoulderScene, profile: ClimberProfile) -> dict:
    """Compile physical variants, comparing arrays rather than model labels."""
    def compile_profile(candidate):
        return compile_model(build_mjcf(scene, candidate))

    def supported_value(value, factor, lo, hi):
        target = value * factor
        return target if lo <= target <= hi else value / factor

    base = compile_profile(profile)
    ids = climber_body_ids(base)
    _, base_data = _isolated_state(base)
    _require_finite_sample(base_data)
    base_arrays = physics_numeric_arrays(base)
    variants, checks = {}, {}
    for label, updates in (("renamed", {"name": "stage1_label_only"}),
                           ("power_0.7", {"power_scale": 0.7}), ("power_1.3", {"power_scale": 1.3})):
        candidate = compile_profile(replace(profile, **updates))
        arrays = physics_numeric_arrays(candidate)
        differences = [name for name in base_arrays if not np.array_equal(base_arrays[name], arrays[name])]
        variants[label] = {"different_physics_arrays": differences, "compared_array_count": len(base_arrays)}
        checks[label] = not differences
    for requested in (0.7, 1.3):
        mass_scale = supported_value(profile.mass_scale, requested, .1, 10)
        scale = mass_scale / profile.mass_scale
        candidate = compile_profile(replace(profile, mass_scale=mass_scale))
        _, data = _isolated_state(candidate)
        _require_finite_sample(data)
        root = int(ids[0])
        mass_error = float(np.max(np.abs(candidate.body_mass[ids] - base.body_mass[ids] * scale)))
        inertia_error = float(np.max(np.abs(candidate.body_inertia[ids] - base.body_inertia[ids] * scale)))
        com_error = float(np.max(np.abs(data.subtree_com[root] - base_data.subtree_com[root])))
        # Armature is intentionally not scaled with body mass.
        matrix_error = float(np.max(np.abs(mass_matrix(candidate, data) - np.diag(candidate.dof_armature)
                            - scale * (mass_matrix(base, base_data) - np.diag(base.dof_armature)))))
        environment_error = float(np.max(np.abs(candidate.body_mass[np.setdiff1d(np.arange(base.nbody), ids)]
                                                - base.body_mass[np.setdiff1d(np.arange(base.nbody), ids)])))
        variants[f"mass_{requested}"] = {"requested_factor": requested, "actual_factor": scale,
            "mass_scale": mass_scale, "climber_mass": float(np.sum(candidate.body_mass[ids])),
            "expected_mass": 78.3 * profile.mass_scale * scale, "mass_error": mass_error,
            "inertia_error": inertia_error, "com_error": com_error,
            "mass_matrix_without_armature_error": matrix_error, "fixed_environment_mass_error": environment_error}
        checks[f"mass_{requested}"] = bool(max(mass_error, inertia_error, com_error, matrix_error, environment_error) < 1e-10
            and abs(np.sum(candidate.body_mass[ids]) - 78.3 * profile.mass_scale * scale) < 1e-10)
    morphology = (
        ("torso", {"torso_length": supported_value(profile.torso_length, 1.2, .05, 2)}, ("pelvis", "abdomen", "chest")),
        ("hip_width", {"hip_width": supported_value(profile.hip_width, 1.2, .05, 2)}, ("pelvis", "abdomen")),
        ("shoulder_width", {"shoulder_width": supported_value(profile.shoulder_width, 1.2, .05, 2)}, ("chest",)),
        ("limbs", {name: supported_value(getattr(profile, name), 1.2, .05, 2) for name in
                    ("upper_arm_length", "forearm_length", "thigh_length", "shin_length")},
         tuple(f"{side}_{segment}" for side in ("left", "right") for segment in ("upper_arm", "forearm", "thigh", "shin"))),
    )
    for label, updates, affected in morphology:
        candidate = compile_profile(replace(profile, **updates))
        errors = _proxy_inertia_errors(candidate)
        changes = {name: float(np.max(np.abs(_inertia_tensor(candidate, _id(candidate, mujoco.mjtObj.mjOBJ_BODY, name))
                                           - _inertia_tensor(base, _id(base, mujoco.mjtObj.mjOBJ_BODY, name))))) for name in affected}
        mass_error = float(np.max(np.abs(candidate.body_mass[ids] - base.body_mass[ids])))
        variants[label] = {"profile_updates": updates, "requested_factor": 1.2,
                           "actual_factors": {name: value / getattr(profile, name) for name, value in updates.items()},
                           "inertia_changes": changes,
                           "mass_error": mass_error, **errors}
        checks[label] = bool(mass_error < 1e-12 and min(changes.values()) > 1e-6
                            and max(errors["ellipsoid_tensor_errors"].values()) < 1e-12
                            and all(max(v["tensor_max_error"], v["com_max_error"]) < 1e-12 for v in errors["capsule_errors"].values()))
    for scale in (1e-6, 0.5, 0.8, 1.0, 1.2):
        candidate = compile_profile(replace(profile, rom_scale=scale))
        expected = base.jnt_range[1:] * (scale / profile.rom_scale)
        for side in ("left", "right"):
            joint = _id(candidate, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_shoulder_roll")
            expected[joint - 1] = np.deg2rad(np.clip(np.array([-80, 80]) * scale, -85, 85))
        range_error = float(np.max(np.abs(candidate.jnt_range[1:] - expected)))
        charts = summarize_model(candidate)["coordinate_charts"]
        chart_minimum = min(min(s["singular_values"]) for c in charts for s in c["rotation_jacobian_near_limits"])
        mirror_error = 0.0
        for name, _, _, axis, _ in JOINT_SPECS:
            if name.startswith("left_"):
                left = _id(candidate, mujoco.mjtObj.mjOBJ_JOINT, name)
                right = _id(candidate, mujoco.mjtObj.mjOBJ_JOINT, name.replace("left_", "right_", 1))
                mirrored_range = candidate.jnt_range[left] if axis[0] else -candidate.jnt_range[left, ::-1]
                mirror_error = max(mirror_error, float(np.max(np.abs(candidate.jnt_range[right] - mirrored_range))))
        variants[f"rom_{scale}"] = {"range_scaling_error": range_error, "mirror_range_error": mirror_error,
                                    "actual_ranges_rad": candidate.jnt_range[1:].tolist(), "coordinate_charts": charts,
                                    "minimum_rotation_jacobian_singular_value": chart_minimum}
        checks[f"rom_{scale}"] = (range_error < 1e-12 and mirror_error < 1e-12 and chart_minimum > 0.06
                                  and all(not c["singular_angles_within_rom_rad"] for c in charts))
    checks["coordinate_charts_safe_supported_rom"] = all(checks[f"rom_{scale}"] for scale in (1e-6, 0.5, 0.8, 1.0, 1.2))
    for requested in (0.7, 1.3):
        strength_scale = supported_value(profile.strength_scale, requested, 1e-6, 10)
        candidate = compile_profile(replace(profile, strength_scale=strength_scale))
        evidence = actuator_diagnostic(candidate, strength_scale=strength_scale)
        variants[f"strength_{requested}"] = {**evidence, "requested_factor": requested,
                                            "actual_factor": strength_scale / profile.strength_scale}
        checks[f"strength_{requested}"] = evidence["passed"]

    # Compile-only acceptance samples: numeric support does not certify poses,
    # collision separation, anatomy, or dynamics for arbitrary morphologies.
    domain_samples = [{"case": "baseline", **_compiled_numerics(base)}]
    dimensions = ("torso_length", "hip_width", "shoulder_width", "upper_arm_length",
                  "forearm_length", "thigh_length", "shin_length")
    for index, (size, mass, strength, rom) in enumerate(product((.05, 2), (.1, 10), (1e-6, 10), (1e-6, 1.2))):
        updates = {**dict.fromkeys(dimensions, size), "mass_scale": mass, "strength_scale": strength, "rom_scale": rom}
        candidate = compile_profile(replace(profile, **updates))
        domain_samples.append({"case": f"profile_corner_{index}", "profile_updates": updates, **_compiled_numerics(candidate)})
    for label, value in (("geometry_friction_lower", 1e-6), ("geometry_friction_upper", 100),
                         ("coordinates_lower", -1000), ("coordinates_upper", 1000)):
        if label.startswith("geometry"):
            friction = 1e-6 if value == 1e-6 else 10
            walls = tuple(replace(w, size=(value,) * 3, friction=friction) for w in scene.walls)
            regions = tuple(replace(r, radius=value, friction=friction) for r in scene.contact_regions)
            updates = {"wall_size_and_region_radius": value, "friction": friction}
        else:
            walls = tuple(replace(w, center=(value,) * 3) for w in scene.walls)
            regions = tuple(replace(r, position=(value,) * 3) for r in scene.contact_regions)
            updates = {"coordinate_components": value}
        candidate = compile_model(build_mjcf(replace(scene, walls=walls, contact_regions=regions), profile))
        domain_samples.append({"case": label, "scene_updates": updates, **_compiled_numerics(candidate)})
    checks["compiled_numerical_domain_samples"] = all(s["passed"] for s in domain_samples)
    return {"variants": variants, "rom_policy": {"supported_scale": [1e-6, 1.2],
            "shoulder_roll_base_deg": [-80, 80], "shoulder_roll_absolute_cap_deg": 85,
            "coverage": "All intermediate scaled chart intervals are contained in the tested ROM=1.2 interval; technical policy, not biological ROM."},
            "numerical_domain": {"samples": domain_samples, "sample_count": len(domain_samples),
                "scope": "Compile-only native-array/option/stat finiteness, not collision-free or anatomical validity.",
                "variant_policy": "Use reciprocal factors when requested scaling leaves the backend domain; actual factors are reported."},
            "checks": checks, "passed": all(checks.values())}


def validate_physical_model(scene=None, profile=None) -> dict:
    """Run the Stage 1 suite and return strict-JSON-compatible evidence."""
    scene = make_synthetic_scene() if scene is None else scene
    profile = ClimberProfile(name="base") if profile is None else profile
    model = mujoco.MjModel.from_xml_string(build_mjcf(scene, profile))
    numerics = _compiled_numerics(model)
    metadata = {"stage": 1, "scope": "compiled physical model only; no climbing or anthropometric claims",
                "mujoco_version": mujoco.__version__, "profile": profile.to_dict()}
    if not numerics["passed"]:
        return {**metadata, "compiled_numerics": numerics, "checks": {"compiled_numerics": False}, "passed": False}
    passive = {}
    for label, joints in (("all", None), ("waist_pitch", ["waist_pitch"]),
                          ("left_elbow", ["left_elbow"]), ("right_knee", ["right_knee"]),
                          ("left_shoulder_yaw", ["left_shoulder_yaw"])):
        passive[label] = passive_decay_diagnostic(model, joints=joints)
    finer = passive_decay_diagnostic(model, timestep=0.001)
    coarse = passive["all"]
    timestep_comparison = {
        "coarse_timestep": 0.002, "fine_timestep": 0.001,
        "qvel_difference_norm": float(np.linalg.norm(np.array(coarse["final_qvel"]) - finer["final_qvel"])),
        "energy_ratio_difference": abs(coarse["energy_ratio"] - finer["energy_ratio"]),
        "coarse_reference_error": coarse["relative_velocity_reference_error"],
        "fine_reference_error": finer["relative_velocity_reference_error"],
        "note": "Numerical convergence, not bitwise equality; full coupled damping reference uses NumPy eigh.",
        "passed": bool(finer["passed"] and finer["relative_velocity_reference_error"]
                       <= coarse["relative_velocity_reference_error"] + 1e-7
                       and abs(coarse["energy_ratio"] - finer["energy_ratio"]) < 0.01),
    }
    _require_finite(timestep_comparison["qvel_difference_norm"], "passive timestep velocity difference")
    fall_coarse = freefall_diagnostic(model, timestep=0.002)
    fall_fine = freefall_diagnostic(model, timestep=0.001)
    bias_ratio = fall_fine["timestep"] / fall_coarse["timestep"]
    bias_error = float(np.linalg.norm(np.array(fall_fine["position_bias"]) - bias_ratio * np.array(fall_coarse["position_bias"])))
    _require_finite(bias_error, "freefall timestep bias error")
    fall_comparison = {
        "coarse_timestep": 0.002, "fine_timestep": 0.001,
        "coarse_position_error": fall_coarse["position_error"], "fine_position_error": fall_fine["position_error"],
        "expected_bias_ratio": bias_ratio,
        "measured_bias_ratio": fall_fine["position_error"] / fall_coarse["position_error"] if fall_coarse["position_error"] > 1e-12 else None,
        "scaled_position_bias_error": bias_error,
        "same_qualitative_checks": fall_coarse["checks"] == fall_fine["checks"],
        "note": "Semi-implicit Euler ballistic position bias halves at 1 ms; both remain unforced, contact-free and warning-free.",
        "passed": bool(fall_coarse["passed"] and fall_fine["passed"] and bias_error < 1e-9
                       and fall_coarse["checks"] == fall_fine["checks"]),
    }
    sections = {
        "compiled_numerics": numerics,
        "compiled_model": summarize_model(model, mass_scale=profile.mass_scale),
        "joint_semantics": measure_joint_semantics(model),
        "freefall": fall_coarse,
        "freefall_1ms": fall_fine,
        "freefall_timestep_comparison": fall_comparison,
        "passive_all_1ms": finer,
        "passive_timestep_comparison": timestep_comparison,
        "mirror": mirror_diagnostic(model),
        "actuators": actuator_diagnostic(model, strength_scale=profile.strength_scale),
        "neutral_rollout": neutral_rollout_diagnostic(model),
        "profile_variants": profile_diagnostics(scene, profile),
    }
    checks = {name: section["passed"] for name, section in sections.items()}
    checks.update({f"passive_{name}": section["passed"] for name, section in passive.items()})
    return {**metadata, **sections, "passive_2ms": passive, "checks": checks, "passed": all(checks.values())}
