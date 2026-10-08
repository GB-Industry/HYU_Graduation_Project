#!/usr/bin/env python3
"""Clean multi-camera inspection of saved, authoritative Stage 5 native states.

No dynamics, controller or reference generation is replayed. All camera videos
use the same recorded state indices, with only display-owned kinematics refreshed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.transfer_motion_audit import audit_motion, trajectory_rows
from scripts.validate_transition import _json_value

CASES = {"right_hand": "right_hand", "left_hand": "left_hand", "left_foot": "foot", "sequence": "sequence"}
ASCENDING_CASES = {"ascending_baseline": "baseline", "ascending_longer": "longer"}
VIEW_NAMES = {"rear": "Rear Three-Quarter", "side": "Side", "front": "Front Three-Quarter"}
PROVENANCE_MODULES = tuple("boulder_v1." + name for name in
                           ("contact_ik", "foot_transfer", "hand_family", "mjcf_builder", "single_hand", "transfers"))
FIXTURES = {"stage5": (None, "boulder_v1.transfers.make_transfer_fixture"),
            "whole_body": ("whole_body_stage5.1", "boulder_v1.whole_body_demo.make_whole_body_fixture"),
            "ascending": ("ascending_stage5.3", "boulder_v1.ascending_sequence.make_ascending_fixture")}


def _file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _case_input(case, input_dir, timestep, fixture, timing=None):
    if fixture == "ascending":
        if case not in ASCENDING_CASES:
            raise ValueError("Ascending fixture requires an ascending profile case")
        if timing is None:
            with (input_dir / "report.json").open() as file:
                timing = json.load(file).get("demo_timing")
        if timing not in ("conservative", "moderate", "fast"):
            raise ValueError("No certified demo_timing in report; select an explicit recorded timing")
        return input_dir / f"{ASCENDING_CASES[case]}_{timing}_{timestep * 1000:g}ms.json", timing
    if case not in CASES or timing is not None:
        raise ValueError("Legacy fixture requires one of its original four cases and no ascent timing")
    return input_dir / f"{CASES[case]}_{timestep * 1000:g}ms.json", None


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _kinematics(mujoco, model, scratch, row):
    scratch.qpos[:] = row["qpos"]
    scratch.qvel[:] = row.get("qvel", 0.)
    scratch.time = float(row["time_s"])
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    mujoco.mj_camlight(model, scratch)


def _actor_bounds(mujoco, model, data):
    from boulder_v1.model_validation import climber_body_ids

    bodies = set(climber_body_ids(model))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in bodies or model.geom_rgba[geom, 3] <= 0:
            continue
        size, kind = model.geom_size[geom], int(model.geom_type[geom])
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = np.full(3, size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.sqrt(rotation ** 2 @ size ** 2)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = np.abs(rotation[:, 2]) * size[1] + size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis = rotation[:, 2]
            extent = np.abs(axis) * size[1] + size[0] * np.sqrt(np.maximum(0., 1. - axis ** 2))
        else:
            raise ValueError(f"Unsupported visible actor geom: {model.geom(geom).name}")
        lo = np.minimum(lo, data.geom_xpos[geom] - extent)
        hi = np.maximum(hi, data.geom_xpos[geom] + extent)
    if not np.isfinite([*lo, *hi]).all():
        raise ValueError("No finite visible climber bounds")
    return lo, hi


def _camera_definitions(model, rows, mujoco, case):
    scratch = mujoco.MjData(model)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for row in rows:
        _kinematics(mujoco, model, scratch, row)
        low, high = _actor_bounds(mujoco, model, scratch)
        lo, hi = np.minimum(lo, low), np.maximum(hi, high)
    center = .5 * (lo + hi)
    radius = float(np.linalg.norm(.5 * (hi - lo)))
    fovy = float(model.vis.global_.fovy)
    distance = radius * 1.12 / math.sin(math.radians(fovy / 2.))
    # The side view faces the isolated moving limb; the sequence uses the LH side.
    left = case in ("left_hand", "left_foot", "sequence") or case in ASCENDING_CASES
    azimuths = {"rear": 45. if case in ("left_hand", "left_foot") else 135.,
                "side": 0. if left else 180., "front": 315. if left else 225.}
    definitions = {name: {"lookat_world_m": center.tolist(), "distance_m": distance,
                           "azimuth_deg": azimuth, "elevation_deg": -5., "vertical_fov_deg": fovy,
                           "fixed_over_entire_recording": True, "wall_display": "hidden" if name == "front" else "visible"}
                   for name, azimuth in azimuths.items()}
    return definitions, {"minimum_world_m": lo.tolist(), "maximum_world_m": hi.tolist(),
                         "bounding_sphere_radius_m": radius, "framing_margin_factor": 1.12}


def _sample_indices(rows, fps):
    times = np.array([r["time_s"] for r in rows], dtype=float)
    if len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError("Recorded rows must have a monotonic clock")
    grid = times[0] + np.arange(math.floor((times[-1] - times[0]) * fps) + 1) / fps
    indices = np.searchsorted(times, grid + 1e-10, side="right") - 1
    selected = [int(i) for i in indices]
    if selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)
    return selected


def _event_keyframes(evidence, rows):
    frames = {"start": 0, "final": len(rows) - 1}
    moves = evidence.get("moves") or [evidence]
    for move_index, move in enumerate(moves):
        prefix = f"move{move_index + 1}"
        eligible = [i for i, row in enumerate(rows) if row["move_index"] == move_index]
        times = np.array([rows[i]["time_s"] for i in eligible])
        if "initial_state" in move:
            frames[prefix + "_start"] = eligible[0]
        events = move.get("events", [])
        release = move.get("release_time_s")
        if release is None:
            release = next((e["time_s"] for e in events if e.get("event") == "RELEASED"), None)
        reach = next((e["time_s"] for e in events if e.get("phase") == "REACH"), None)
        request = move.get("request") or {}
        reach_s = (request.get("foot_request") or {}).get("reach_s", 3.) if move.get("primitive") == "foot" else request.get("hand_reach_s", 4.)
        capture = (move.get("capture") or {}).get("time_s")
        for label, time, after in (("release_before", release, False), ("release_after", release, True),
                                   ("reach_mid", reach + .5 * reach_s if reach is not None else None, False),
                                   ("capture_before", capture, False), ("capture_after", capture, True),
                                   ("touchdown", (move.get("touchdown") or {}).get("time_s"), False),
                                   ("acquired", (move.get("acquisition") or {}).get("time_s"), False),
                                   ("loaded", move.get("load_complete_time_s"), False),
                                   ("ready", (move.get("readiness") or {}).get("time"), False)):
            if time is not None:
                index = int(np.searchsorted(times, time + 1e-10, side="right"))
                if not after:
                    index -= 1
                frames[f"{prefix}_{label}"] = eligible[min(len(eligible) - 1, max(0, index))]
    return frames


def _compose(scene_image, row, case, view, width, height):
    """Only essentials in an independent sidebar; no pixels drawn over the actor."""
    scene_width = scene_image.width
    image = Image.new("RGB", (width, height), (15, 22, 32))
    image.paste(scene_image, (0, 0))
    draw = ImageDraw.Draw(image)
    x, room = scene_width + 28, width - scene_width - 56
    title, font, small = (ImageFont.load_default(size=s) for s in (30, 25, 20))
    draw.line((scene_width, 0, scene_width, height), fill=(69, 89, 111), width=2)
    y = 40

    def text(value, color=(219, 228, 238), selected=font, gap=12):
        nonlocal y
        words = str(value).split()
        line = ""
        for word in words:
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=selected) > room and line:
                draw.text((x, y), line, font=selected, fill=color)
                y += selected.size + 8
                line = word
            else:
                line = trial
        draw.text((x, y), line, font=selected, fill=color)
        y += selected.size + gap

    text(VIEW_NAMES[view], selected=title, gap=22)
    count, index = row.get("move_count", 1), row.get("move_index", 0)
    text(f"Move {index + 1} / {count}")
    text(row.get("moving_limb", "").replace("_", " "), selected=title, gap=22)
    text(row.get("source", ""), selected=small)
    text("-> " + row.get("target", ""), gap=30)
    text("PHASE", (139, 157, 177), small, gap=5)
    text(row.get("phase", "SOURCE").replace("_", " "), gap=26)
    status = row.get("status", "RUNNING")
    if row.get("terminal") and count > 1:
        status = f"MOVE {index + 1}: {status}"
    text(status, (129, 217, 168) if "SUCCESS" in status else
         (255, 156, 122) if row.get("terminal") else (221, 229, 239), gap=28)
    text("ACTUAL CONTACTS", (139, 157, 177), small, gap=10)
    contacts = row.get("contacts", {})
    for limb, label in (("LEFT_HAND", "LH"), ("RIGHT_HAND", "RH"), ("LEFT_FOOT", "LF"), ("RIGHT_FOOT", "RF")):
        text(f"{label}: {contacts.get(limb, 'none')}", selected=small, gap=12)
    if y > height - 115:
        raise ValueError("Essential sidebar text exceeds the reserved panel")
    draw.text((x, height - 95), f"Recorded time {row['time_s']:.3f} s", font=small, fill=(170, 185, 202))
    if view == "front":
        draw.text((x, height - 59), "Wall hidden in display only", font=small, fill=(170, 185, 202))
    return image


def _render_view(mujoco, renderer, display_model, data, row, definition, walls, view, case, width, height,
                 *, scene_only=False):
    _kinematics(mujoco, display_model, data, row)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = definition["lookat_world_m"]
    camera.distance = definition["distance_m"]
    camera.azimuth, camera.elevation = definition["azimuth_deg"], definition["elevation_deg"]
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    original = display_model.geom_rgba[walls, 3].copy()
    try:
        if view == "front":
            display_model.geom_rgba[walls, 3] = 0.
        renderer.update_scene(data, camera=camera, scene_option=options)
        scene_image = Image.fromarray(renderer.render().copy())
    finally:
        display_model.geom_rgba[walls, 3] = original
    return scene_image if scene_only else _compose(scene_image, row, case, view, width, height)


def _load_verified_evidence(path, fixture="stage5"):
    if fixture not in FIXTURES:
        raise ValueError("Unknown evidence fixture")
    with path.open() as file:
        evidence = json.load(file)
    if evidence.get("fixture") != FIXTURES[fixture][0]:
        raise ValueError("Recorded fixture does not match the explicitly selected fixture")
    if not evidence.get("success") or evidence.get("status") != "SUCCESS":
        raise ValueError("Clean motion demonstration requires a recorded successful episode")
    modules = evidence.get("provenance", {}).get("modules", {})
    allowed = {"boulder_v1." + path.stem for path in (ROOT / "src" / "boulder_v1").glob("*.py")}
    required = allowed if fixture in ("whole_body", "ascending") else set(PROVENANCE_MODULES)
    if not isinstance(modules, dict) or not required.issubset(modules):
        raise ValueError("Recorded episode lacks complete canonical source-module provenance")
    for name, info in modules.items():
        if name not in allowed or not isinstance(info, dict):
            raise ValueError(f"Invalid recorded source-module identity: {name}")
        source = Path(info["file"]).resolve()
        expected = (ROOT / "src" / (name.replace(".", "/") + ".py")).resolve()
        if (source != expected or not source.is_relative_to(ROOT)
                or hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]):
            raise ValueError(f"Recorded source differs from current source: {name}")
    if fixture == "ascending":
        identity = evidence.get("reference_identity") or []
        if (evidence.get("kind") != "ascending_sequence" or evidence.get("negative") is not None
                or (evidence.get("validation") or {}).get("physical_success") is not True
                or (evidence.get("validation") or {}).get("accepted") is not True
                or len(evidence.get("moves", [])) != 3 or evidence.get("completed_moves") != 3
                or len(identity) != 2 or not all(item.get("same_object") is True for item in identity)):
            raise ValueError("Ascending rendering requires certified three-move native evidence and owner identity observations")
    return evidence


def _matching_fixture(evidence, timestep, fixture):
    """Compile recorded whole-body inputs without fixture IK, reset or forward."""
    if fixture not in FIXTURES or evidence.get("fixture") != FIXTURES[fixture][0]:
        raise ValueError("Recorded fixture does not match the explicitly selected fixture")
    if fixture == "stage5":
        from boulder_v1.transfers import make_transfer_fixture

        return make_transfer_fixture(timestep)
    import mujoco
    from boulder_v1.mjcf_builder import build_mjcf
    from boulder_v1.schema import Affordance, BoulderScene, ClimberProfile, ContactRegion, Limb, SourceType, WallSurface
    from scripts.validate_whole_body import _hash, _model_hash

    inputs = evidence.get("fixture_inputs")
    if not isinstance(inputs, dict) or inputs.get("factory") != FIXTURES[fixture][1]:
        raise ValueError("Missing or invalid whole-body fixture inputs")
    if (inputs.get("fixture") != FIXTURES[fixture][0] or inputs.get("dt_s") != timestep
            or evidence.get("dt_s") != timestep or evidence.get("profile") != inputs.get("profile")
            or inputs.get("input_sha256") != _hash({key: value for key, value in inputs.items() if key != "input_sha256"})):
        raise ValueError("Recorded whole-body fixture input hash/timestep/profile mismatch")
    raw_scene = inputs["scene"]
    scene = BoulderScene(**{**raw_scene,
        "walls": tuple(WallSurface(**wall) for wall in raw_scene["walls"]),
        "contact_regions": tuple(ContactRegion(**{**region, "source_type": SourceType(region["source_type"]),
            "affordances": frozenset(Affordance(value) for value in region["affordances"])})
                                 for region in raw_scene["contact_regions"]),
        "start_configuration": {Limb(limb): hold for limb, hold in raw_scene["start_configuration"].items()}})
    profile = ClimberProfile(**inputs["profile"])
    if fixture == "ascending":
        from boulder_v1.contact_geometry import canonical_geometry
        from boulder_v1.morphology_envelope import study_profiles
        from scripts.validate_whole_body import _evidence_value

        if (profile not in tuple(study_profiles()[name] for name in ("baseline", "longer"))
                or inputs.get("negative") is not None
                or inputs.get("contact_geometry") != _evidence_value({r.id: canonical_geometry(r) for r in scene.contact_regions})):
            raise ValueError("Recorded ascending profile/typed contact geometry mismatch")
    tree = ET.fromstring(build_mjcf(scene, profile))
    tree.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10")
    xml = ET.tostring(tree, encoding="unicode")
    if xml != inputs.get("model_xml") or hashlib.sha256(xml.encode()).hexdigest() != inputs.get("model_xml_sha256"):
        raise ValueError("Recorded model XML differs from the saved scene/profile")
    model = mujoco.MjModel.from_xml_string(xml)
    dimensions = {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq")}
    if dimensions != inputs.get("model_dimensions") or _model_hash(model) != inputs.get("compiled_model_sha256"):
        raise ValueError("Recorded compiled model differs from the saved scene/profile")
    seed = np.asarray(inputs["seed_qpos"], dtype=float)
    if seed.shape != (model.nq,) or not np.isfinite(seed).all():
        raise ValueError("Invalid recorded fixture seed")
    return model, None, scene, profile, seed


def _decoded_contact_sheet(video_path, indices, event_frames, rows, output_path):
    """Inspect encoded video frames, not merely pre-encoding source pictures."""
    import imageio.v2 as iio

    selected = {}
    mapping = {}
    times = np.array([rows[i]["time_s"] for i in indices])
    for name, native in event_frames.items():
        native_time = rows[native]["time_s"]
        if name.endswith("_before"):
            frame = max(0, int(np.searchsorted(times, native_time + 1e-10, side="right")) - 1)
            bracket = "at_or_before"
        elif name.endswith(("_after", "_touchdown", "_acquired", "_loaded", "_ready")):
            frame = min(len(times) - 1, int(np.searchsorted(times, native_time - 1e-10, side="left")))
            bracket = "at_or_after"
        else:
            frame = int(np.argmin(np.abs(times - native_time)))
            bracket = "nearest"
        mapping[name] = {"native_record_index": native, "native_state_time_s": native_time,
                         "decoded_frame_index": frame, "decoded_state_time_s": float(times[frame]),
                         "bracket_policy": bracket}
        selected.setdefault(frame, []).append(name)
    columns, cell_width, cell_height = 3, 640, 405
    sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(selected) / columns) * cell_height), (15, 22, 32))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=17)
    with iio.get_reader(video_path) as reader:
        for position, (frame, labels) in enumerate(sorted(selected.items())):
            image = Image.fromarray(reader.get_data(frame))
            image.thumbnail((cell_width, 360))
            x, y = position % columns * cell_width, position // columns * cell_height
            sheet.paste(image, (x, y + 40))
            draw.text((x + 8, y + 4), f"{', '.join(labels)} | {times[frame]:.3f}s", font=font, fill=(221, 230, 241))
    sheet.save(output_path)
    return {"path": str(output_path), "decoded_video_frame_indices": list(sorted(selected)),
            "state_times_s": [float(times[i]) for i in sorted(selected)], "native_to_decoded_mapping": mapping,
            "timing_note": "Decoded sheets use directed 20Hz brackets; exact native event states are in the separate PNGs."}


def _video_timing(path, fps, frame_count, native_duration):
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_read_frames,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    rates = [float(a) / float(b) for a, b in (stream[key].split("/") for key in ("avg_frame_rate", "r_frame_rate"))]
    duration, decoded_count = float(stream["duration"]), int(stream["nb_read_frames"])
    if (decoded_count != frame_count or not all(math.isclose(rate, fps, abs_tol=1e-8) for rate in rates)
            or not math.isclose(duration, frame_count / fps, abs_tol=1e-5)
            or not native_duration < duration <= native_duration + 2. / fps + 1e-5):
        raise ValueError("Encoded ascending video does not preserve native 1x duration/frame rate")
    return {"ffprobe": stream, "native_duration_s": native_duration, "encoded_duration_s": duration,
            "encoded_to_native_duration_ratio": duration / native_duration, "video_speedup": False,
            "terminal_frame_padding_max_s": 2. / fps, "verified": True}


def render_case(case, input_dir, output, *, timestep=.002, width=1920, height=1080, fps=20., preview=False,
                fixture="stage5", audit_only=False, timing=None, scene_only=False):
    import mujoco
    import imageio.v2 as iio

    input_path, selected_timing = _case_input(case, input_dir, timestep, fixture, timing)
    evidence = _load_verified_evidence(input_path, fixture)
    model, _, scene, _, _ = _matching_fixture(evidence, timestep, fixture)
    if evidence.get("kind") != ("ascending_sequence" if fixture == "ascending" else CASES[case]):
        raise ValueError("Recorded kind does not match selected case")
    if fixture == "ascending" and (evidence.get("profile", {}).get("name") != ASCENDING_CASES[case]
            or evidence.get("timing") != selected_timing):
        raise ValueError("Recorded ascending profile/timing does not match selected case")
    measurements = audit_motion(model, evidence)
    if (measurements["whole_case"]["timing"]["time_reset_count"]
            or measurements["whole_case"]["timing"]["sample_gap_count"]
            or measurements["whole_case"]["timing"]["changed_pose_at_duplicate_time_count"]):
        raise ValueError("Recorded trajectory is discontinuous; refuse a cleaned success video")
    if fixture in ("whole_body", "ascending") and (not all(boundary["full_state_equal"] for boundary in measurements["sequence_boundaries"])
            or not all(item["step_clock_matches"] and item["duration_matches_steps"]
                       and item["zero_recorded_applied_forces"] and item["samples_have_both_force_channels"]
                       for item in measurements["native_accounting"])):
        raise ValueError("Recorded whole-body state handoff/clock/force accounting mismatch")
    if fixture == "ascending":
        from scripts.validate_ascent import _case_verdict

        if not _case_verdict(model, evidence, evidence["fixture_inputs"], measurements, evidence["reference_identity"]):
            raise ValueError("Recorded ascending episode failed current native-contact/integration certification")
    input_digest = _file_sha256(input_path)
    audit_source = {"fixture": FIXTURES[fixture][0] or "stage5", "factory": FIXTURES[fixture][1],
                    "fixture_input_sha256": (evidence.get("fixture_inputs") or {}).get("input_sha256"),
                    "input_json": str(input_path), "input_sha256": input_digest,
                    "source_modules": evidence["provenance"]["modules"], "physics_steps_executed": 0}
    if audit_only:
        report = {"case": case, "audit_source": audit_source, "motion": measurements,
                  "physics_steps_executed": 0, "dynamics_replayed": False, "quality_verdict": None}
        _write_json(output / f"audit_{case}_report.json", report)
        return report
    rows = trajectory_rows(evidence)
    moves = evidence.get("moves") or [evidence]
    for row in rows:
        if row["row_kind"] != "sample":
            terminal = row["row_kind"] == "final_state"
            row.update(phase="SETTLED" if terminal else "SOURCE_READY", terminal=terminal,
                       status=moves[row["move_index"]]["status"] if terminal else "RUNNING")
    cameras, framing = _camera_definitions(model, rows, mujoco, case)
    scene_width = width if scene_only else width * 3 // 4
    if (scene_width > model.vis.global_.offwidth or height > model.vis.global_.offheight
            or width < 1280 or height < 720 or width % 2 or height % 2):
        raise ValueError("Use a framebuffer-supported even resolution of at least 1280x720")
    display_model = copy.copy(model)
    display_data = mujoco.MjData(display_model)
    walls = [display_model.geom(f"{wall.id}_geom").id for wall in scene.walls]
    indices = _sample_indices(rows, fps)
    keyframes = _event_keyframes(evidence, rows)
    artifacts = {}
    renderer = mujoco.Renderer(display_model, height=height, width=scene_width)
    display_options = {"scene_only": True} if scene_only else {}
    try:
        for view, definition in cameras.items():
            prefix = output / f"clean_{case}_{view}"
            if preview:
                image = _render_view(mujoco, renderer, display_model, display_data, rows[0], definition,
                                     walls, view, case, width, height, **display_options)
                image.save(prefix.with_suffix(".png"))
                continue
            with iio.get_writer(str(prefix.with_suffix(".mp4")), fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=8, macro_block_size=1,
                                ffmpeg_params=["-threads", "2"]) as writer:
                for native in indices:
                    image = _render_view(mujoco, renderer, display_model, display_data, rows[native], definition,
                                         walls, view, case, width, height, **display_options)
                    writer.append_data(np.asarray(image))
            snapshots = {}
            for event, native in keyframes.items():
                path = output / f"clean_{case}_{view}_{event}.png"
                image = _render_view(mujoco, renderer, display_model, display_data, rows[native], definition,
                                     walls, view, case, width, height, **display_options)
                image.save(path)
                snapshots[event] = {"path": str(path), "actual_state_time_s": rows[native]["time_s"],
                                    "native_record_index": native}
            sheet_path = output / f"audit_{case}_{view}_decoded.png"
            decoded = _decoded_contact_sheet(prefix.with_suffix(".mp4"), indices, keyframes, rows, sheet_path)
            artifacts[view] = {"video": str(prefix.with_suffix(".mp4")), "keyframes": snapshots,
                               "decoded_contact_sheet": decoded, "frame_count": len(indices)}
            if fixture == "ascending":
                artifacts[view]["playback_timing"] = _video_timing(prefix.with_suffix(".mp4"), fps, len(indices),
                                                                  rows[-1]["time_s"] - rows[0]["time_s"])
    finally:
        renderer.close()
    report = {"case": case, "input_json": str(input_path), "input_sha256": input_digest,
              "source_provenance_checked": True, "audit_source": audit_source,
              "physics_steps_executed": 0, "dynamics_replayed": False,
              "resolution": [width, height], "scene_viewport": [0, 0, scene_width, height],
              "sidebar": None if scene_only else [scene_width, 0, width - scene_width, height], "fps": fps,
              "all_cameras_share_native_record_indices": True, "rendered_native_indices": indices,
              "rendered_state_times_s": [rows[i]["time_s"] for i in indices], "cameras": cameras,
              "framing": framing, "motion": measurements, "artifacts": artifacts, "preview_only": preview,
              "display_changes": ["sites hidden to expose real hands/shoes", "wall alpha hidden only for front display"],
              "layout_verdict": "PASS", "motion_demonstration_verdict": "PENDING VISUAL INSPECTION"}
    _write_json(output / f"clean_{case}_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=(*CASES, *ASCENDING_CASES),
                        help="default: original four cases, or the two ascending positives with --fixture ascending")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "transfers-stage5")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage5-clean")
    parser.add_argument("--fixture", choices=tuple(FIXTURES), default="stage5")
    parser.add_argument("--timing", choices=("conservative", "moderate", "fast"), help="ascending only; default: report demo_timing")
    parser.add_argument("--scene-only", action="store_true", help="native actor viewport only; no sidebar or telemetry")
    parser.add_argument("--dt", "--timestep", type=float, choices=(.002, .001), default=.002)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=20.)
    parser.add_argument("--preview", action="store_true", help="initial-state camera PNGs only; not completed video evidence")
    parser.add_argument("--audit-only", action="store_true", help="save read-only native-motion audits without opening a renderer")
    args = parser.parse_args(argv)
    selected_cases = args.case or (ASCENDING_CASES if args.fixture == "ascending" else CASES)
    if (any(case not in (ASCENDING_CASES if args.fixture == "ascending" else CASES) for case in selected_cases)
            or args.timing is not None and args.fixture != "ascending"):
        parser.error("Cases/timing must match the explicitly selected fixture")
    if args.width < 1280 or args.height < 720 or args.width % 2 or args.height % 2:
        parser.error("Resolution must be even and at least 1280x720")
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    output = args.output.resolve()
    outputs = ROOT / "outputs"
    if (not output.parent.is_dir() or output == args.input.resolve() or output == outputs
            or not output.is_relative_to(outputs) or output.exists() and not output.is_dir()
            or any(part.startswith(("stage4-baseline-", "stage4-preserved-", "stage5-baseline-", "stage5-preserved-"))
                   for part in output.relative_to(outputs).parts)):
        parser.error("Use a separate output directory beneath this worktree's existing outputs")
    output.mkdir(exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    options = {"timing": args.timing, "scene_only": args.scene_only} if args.fixture == "ascending" or args.scene_only else {}
    cases = [render_case(case, args.input.resolve(), output, timestep=args.dt, width=args.width,
                         height=args.height, fps=args.fps, preview=args.preview, fixture=args.fixture,
                         audit_only=args.audit_only, **options) for case in selected_cases]
    summary = {"resolution": [args.width, args.height], "physics_steps_executed": 0,
               "dynamics_replayed": False, "fixture": args.fixture, "audit_only": args.audit_only,
               "cases": [{"case": r["case"], "report": str(output / f"{'audit' if args.audit_only else 'clean'}_{r['case']}_report.json"),
                          "artifacts": r.get("artifacts", {})} for r in cases]}
    _write_json(output / "render_manifest.json", summary)
    print(json.dumps({"resolution": summary["resolution"], "cases": [r["case"] for r in cases],
                      "physics_steps_executed": 0, "output": str(output), "preview_only": args.preview}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
