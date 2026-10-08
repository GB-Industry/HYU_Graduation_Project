#!/usr/bin/env python3
"""Offscreen observation of authoritative static and transition simulations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from boulder_v1 import (
    ClimberProfile,
    ContactMode,
    GraspManager,
    InitialContactError,
    Limb,
    TransitionObservation,
    TransitionPhase,
    TransitionRequest,
    build_mjcf,
    compile_model,
    execute_transition_sequence,
    get_state_summary,
    initialize_episode,
    make_synthetic_scene,
    simulate_static_stance,
)
from boulder_v1.runtime import _import_mujoco
from scripts.view_scene import _contact_lines, _initialization_state, _transition_requests, contact_mode_label

PROFILES: dict[str, ClimberProfile] = {
    "base": ClimberProfile(name="base"),
    "compact_strong": ClimberProfile(
        name="compact_strong",
        upper_arm_length=0.28,
        forearm_length=0.24,
        thigh_length=0.39,
        shin_length=0.37,
        rom_scale=1.08,
        strength_scale=1.15,
        grip_capacity=1000.0,
    ),
    "long_reach_lower_grip": ClimberProfile(
        name="long_reach_lower_grip",
        upper_arm_length=0.34,
        forearm_length=0.30,
        thigh_length=0.44,
        shin_length=0.42,
        rom_scale=0.95,
        strength_scale=0.92,
        grip_capacity=850.0,
    ),
}


def _ensure_offscreen_framebuffer(xml: str) -> str:
    """Ensure MJCF declares a framebuffer capable of the existing image sizes."""
    if "<visual>" not in xml:
        return xml.replace(
            "</mujoco>",
            '  <visual><global offwidth="1920" offheight="1080"/></visual>\n</mujoco>',
        )
    return xml


def _configure_camera(cam: Any, model: Any) -> None:
    center = model.stat.center
    cam.lookat[0] = float(center[0])
    cam.lookat[1] = float(center[1] - 0.20)
    cam.lookat[2] = float(center[2] * 0.94)
    cam.distance = float(model.stat.extent * 0.96)
    cam.elevation = -9.0
    cam.azimuth = 108.0


def _render_frame(mujoco: Any, model: Any, data: Any, scratch: Any,
                  renderer: Any, camera: Any) -> Image.Image:
    # Forward refreshes derived rendering state, but must not change live solver state.
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)
    renderer.update_scene(scratch, camera=camera)
    return Image.fromarray(renderer.render())


def _draw_hud(img: Image.Image, title: str, profile_name: str, metadata: dict[str, Any]) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    compact = img.width < 800 or img.height < 240
    x, y = (12, 8) if compact else (32, 26)
    text_width = img.width - 2 * x
    status = metadata["status_label"]
    color = (255, 140, 100) if status.startswith("FAILURE/") else (240, 245, 255)
    if "limb" in metadata:
        move = (f"Requested move {metadata['move_index'] + 1}/{metadata['total_moves']}: "
                f"{metadata['limb']} {metadata['source_hold']} -> {metadata['target_hold']}")
    else:
        move = "Observation only; no stability certificate"
    sim_time, root_z = metadata["time"], metadata["root_pos"][2]
    time_text = f"{sim_time:.3f}s" if sim_time is not None and math.isfinite(sim_time) else "nonfinite"
    root_text = f"{root_z:+.3f}m" if root_z is not None and math.isfinite(root_z) else "nonfinite"
    lines = [f"{title} | {profile_name}", status, metadata["contact_mode_label"], move,
             f"Phase: {metadata['phase'] or 'not_started'}", *_contact_lines(metadata),
             f"Step {metadata['steps']} | t={time_text} | z={root_text}"]
    if compact:
        # Keep all contact fields at the normal font size, rather than shrinking the HUD.
        mode = ("DEBUG: NONPHYSICAL / NOT SCIENTIFIC" if metadata["nonphysical"] else
                "PHYSICAL | CONTROLLER NOT VALIDATED")
        progress = f"Phase: {metadata['phase'] or 'not_started'}"
        if "limb" in metadata:
            limb_label = {"LEFT_HAND": "LH", "RIGHT_HAND": "RH", "LEFT_FOOT": "LF", "RIGHT_FOOT": "RF"}.get(
                metadata["limb"], metadata["limb"])
            progress = (f"{metadata['move_index'] + 1}/{metadata['total_moves']} "
                        f"{limb_label} {metadata['source_hold']} -> {metadata['target_hold']} | "
                        f"{metadata['phase'] or 'not_started'}")
        lines = [lines[0], status, mode, progress]

        def number(value: Any, unit: str, precision: int = 2) -> str:
            return f"{value:.{precision}f}{unit}" if value is not None and math.isfinite(value) else "n/a"

        for limb, label in ((Limb.LEFT_HAND, "LH"), (Limb.RIGHT_HAND, "RH")):
            hand = metadata["hand_states"][limb]
            lines.extend([
                f"HAND {label}: target={hand['region_id'] or 'none'} "
                f"active={'Y' if hand['active'] else 'N'} valid={'Y' if hand['valid'] else 'N'}",
                f"load={number(hand['load'], 'N')} cap={number(hand['capacity'], 'N')} "
                f"margin={number(hand['margin'], 'N')}",
            ])
        for limb, label in ((Limb.LEFT_FOOT, "LF"), (Limb.RIGHT_FOOT, "RF")):
            foot = metadata["foot_states"][limb]
            state = foot["status"].removeprefix("FOOT_").replace("IDEALIZED_DEBUG_ATTACHMENT", "IDEALIZED_DEBUG")
            lines.extend([
                f"FOOT {label}: {state} contact={'Y' if foot['contacting'] else 'N'} "
                f"support={'Y' if foot['supporting'] else 'N'} slip={'Y' if foot['slipping'] else 'N'}",
                f"Fn={number(foot['normal_force'], 'N')} Ft={number(foot['tangential_force'], 'N')} "
                f"speed={number(foot['tangential_speed'], 'm/s', 3)}",
            ])
        lines.append(f"Step {metadata['steps']} | t={time_text} | z={root_text}")
    if metadata.get("pose_available") is False:
        lines.append("POSE UNAVAILABLE: nonfinite state")
    elif metadata.get("reason"):
        reason = metadata["reason"]
        if draw.textbbox((0, 0), reason, font=font)[2] > text_width:
            reason = "Reason: see manifest for full details"
        lines.append(reason)
    spacing = 12 if compact else 16
    inset = 6 if compact else 18
    draw.rectangle([(inset, inset), (img.width - inset, min(img.height - inset, y + spacing * len(lines)))],
                   fill=(12, 18, 28, 220), outline=(50, 90, 140, 240), width=2)
    for index, line in enumerate(lines):
        line_color = (255, 220, 80) if index == 2 else color if index == 1 or index >= 10 else (210, 230, 245)
        draw.text((x, y + index * spacing), line, fill=line_color, font=font)
    return img


def _json_value(value: Any) -> Any:
    """Serialize snapshots, including nonfinite failures, as portable strict JSON."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {_json_value(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(v) for v in value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _state_metadata(state: Any, model: Any, scene: Any) -> dict[str, Any]:
    metadata = _json_value(state)
    mode = ContactMode(metadata["contact_mode"])
    metadata.update({
        "pose_available": metadata["finite"], "contact_mode_label": contact_mode_label(mode),
        "contact_mode_numeric": 0 if mode == ContactMode.PHYSICAL else 1,
        "scientific_contacts": mode == ContactMode.PHYSICAL,
        "nonphysical": mode == ContactMode.IDEALIZED_DEBUG,
        "physics_certified": False,
        "active_attachments": {
            limb: {"limb": limb, "region": _json_value(scene.region(metadata["contact_configuration"][limb])),
                   "eq_name": name, "eq_id": model.equality(name).id}
            for limb, name in metadata["attachment_constraints"].items()
        },
    })
    return metadata


def render_static_stance(
    profile_name: str = "compact_strong",
    output_dir: Path | None = None,
    width: int = 1280,
    height: int = 720,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> Path:
    """Observe the authoritative static simulator without certifying stability."""
    mujoco = _import_mujoco()
    profile = PROFILES.get(profile_name, PROFILES["base"])
    contact_mode = ContactMode(contact_mode)
    out_dir = output_dir or (HERE / "outputs" / "visual")
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_synthetic_scene()
    model = compile_model(_ensure_offscreen_framebuffer(build_mjcf(scene, profile, contact_mode=contact_mode)))
    data = mujoco.MjData(model)
    result = None
    initialization_error = None
    try:
        manager = initialize_episode(model, data, scene, profile=profile,
                                     attach_feet=contact_mode == ContactMode.IDEALIZED_DEBUG,
                                     initial_qpos=initial_qpos)
    except InitialContactError as exc:
        initialization_error = str(exc)
        state = _initialization_state(model, data, scene, profile)
    else:
        result = simulate_static_stance(model, data, scene, profile, steps=40, manager=manager)
        state = get_state_summary(model, data, manager)
    metadata = _state_metadata(state, model, scene)
    metadata.update({
        "status_label": "FAILURE/INITIALIZATION_FAILURE" if initialization_error else "STATIC OBSERVATION",
        "status": "INITIALIZATION_FAILURE" if initialization_error else "STATIC_OBSERVATION",
        "phase": None, "steps": result.steps if result is not None else 0,
        "reason": initialization_error or "Static image only; NO PHYSICS CERTIFICATION",
        "initialization_error": initialization_error,
    })
    camera = mujoco.MjvCamera()
    _configure_camera(camera, model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        img = (_render_frame(mujoco, model, data, mujoco.MjData(model), renderer, camera)
               if metadata["finite"] else Image.new("RGB", (width, height), color=(10, 14, 22)))
        _draw_hud(img, "Static Stance", profile.name, metadata)
        out_file = out_dir / f"stance_{profile.name}.png"
        img.save(out_file)
    finally:
        renderer.close()
    manifest = {"profile": profile.name, "resolution": (width, height),
                "rendering_mode": "static", "contact_mode": contact_mode,
                "contact_mode_label": contact_mode_label(contact_mode), "physics_certified": False,
                "scientific_contacts": contact_mode == ContactMode.PHYSICAL,
                "nonphysical": contact_mode == ContactMode.IDEALIZED_DEBUG,
                "initialization_error": initialization_error, "physical_success": False,
                "endpoint": out_file, "observation": metadata, "physics_result": result}
    (out_dir / f"stance_{profile.name}.json").write_text(
        json.dumps(_json_value(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{metadata['status_label']}] {metadata['reason']} | {metadata['contact_mode_label']}")
    print(f"[ARTIFACT] Static observation: {out_file} ({width}x{height})")
    return out_file


def generate_transition_montage(
    keyframe_paths: dict[str, Path],
    output_path: Path,
    annotations: dict[str, str] | None = None,
    *,
    columns: int = 4,
) -> Path | None:
    """Arrange observed keyframes, including partial runs and actual terminal outcomes."""
    if not keyframe_paths:
        return None
    out_path = Path(output_path)
    if out_path.is_dir() or out_path.suffix == "":
        out_path = out_path / "transition_montage.png"
    cell_w, cell_h = 640, 360
    rows = max(2, math.ceil(len(keyframe_paths) / columns))
    montage = Image.new("RGB", (cell_w * columns, cell_h * rows), color=(10, 14, 22))
    for idx, (name, path) in enumerate(keyframe_paths.items()):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile, "RGBA")
        draw.rectangle([(8, cell_h - 36), (cell_w - 8, cell_h - 8)], fill=(12, 18, 28, 210),
                       outline=(50, 90, 140, 200), width=1)
        draw.text((16, cell_h - 30), (annotations or {}).get(name, name), fill=(255, 220, 80))
        montage.paste(tile, ((idx % columns) * cell_w, (idx // columns) * cell_h))
    montage.save(out_path)
    return out_path


def create_sequence_montage(
    keyframe_paths: dict[str, Path],
    out_path: Path,
    annotations: dict[str, str] | None = None,
) -> Path | None:
    return generate_transition_montage(keyframe_paths, out_path, annotations, columns=2)


def _render_transitions(
    profile_name: str,
    output_dir: Path | None,
    requests: list[TransitionRequest],
    fps: int,
    width: int,
    height: int,
    save_keyframes: bool,
    *,
    sequence: bool,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> dict[str, Any]:
    """Run one authoritative sequence; the callback only observes its state and events."""
    mujoco = _import_mujoco()
    profile = PROFILES.get(profile_name, PROFILES["base"])
    contact_mode = ContactMode(contact_mode)
    out_dir = output_dir or (HERE / "outputs" / "visual")
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_synthetic_scene()
    model = compile_model(_ensure_offscreen_framebuffer(build_mjcf(scene, profile, contact_mode=contact_mode)))
    data = mujoco.MjData(model)
    initialization_error = None
    physics_result = None
    try:
        manager = initialize_episode(model, data, scene, profile=profile,
                                     attach_feet=contact_mode == ContactMode.IDEALIZED_DEBUG,
                                     initial_qpos=initial_qpos)
    except InitialContactError as exc:
        initialization_error = str(exc)
        initial_state = _initialization_state(model, data, scene, profile)
    scratch = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    _configure_camera(camera, model)
    prefix = "sequence" if sequence else "transition"
    title = "Multi-Limb Sequence (No Reset)" if sequence else "Single-Limb Transition"
    endpoint_path = out_dir / f"{prefix}_{profile.name}_endpoint.png"
    pil_frames: list[Image.Image] = []
    frame_metadata: list[dict[str, Any]] = []
    saved_keyframes: dict[str, Path] = {}
    keyframe_metadata: dict[str, dict[str, Any]] = {}
    annotations: dict[str, str] = {}
    last_phases: dict[int, TransitionPhase | None] = {}
    phase_names = {
        TransitionPhase.INITIAL_STANCE: "transition_00_initial.png",
        TransitionPhase.PRE_SHIFT: "transition_01_pre_shift.png",
        TransitionPhase.RELEASE_LIMB: "transition_02_release.png",
        TransitionPhase.SUPPORT_PHASE: "transition_03_support.png",
        TransitionPhase.REACH_PHASE: "transition_04_reach.png",
        TransitionPhase.ATTACH_PHASE: "transition_05_attach.png",
        TransitionPhase.STABILIZED_STANCE: "transition_06_stabilized.png",
    }
    renderer = mujoco.Renderer(model, height=height, width=width)

    def on_frame(observation: TransitionObservation, observed_data: Any, gm: GraspManager) -> None:
        terminal = observation.result is not None
        phase_changed = last_phases.get(observation.move_index) != observation.phase
        last_phases[observation.move_index] = observation.phase
        # Retain sequence subsampling, but never omit initial, phase-event, or terminal frames.
        if sequence and observation.steps % 2 and not terminal and not phase_changed:
            return
        label = observation.status.value
        if terminal and not observation.result.success:
            label = f"FAILURE/{label}"
        metadata = _state_metadata(observation.state, model, scene)
        metadata.update(_json_value({
            "move_index": observation.move_index,
            "total_moves": observation.total_moves,
            "limb": observation.request.limb,
            "source_hold": observation.request.source_hold,
            "target_hold": observation.request.target_hold,
            "phase": observation.phase,
            "steps": observation.steps,
            "time": observation.state.time,
            "root_pos": observation.state.root_pos,
            "finite": observation.state.finite,
            "pose_available": observation.state.finite,
            "status": observation.status,
            "status_label": label,
            "terminal": terminal,
            "reason": observation.result.reason if terminal else "",
        }))
        # Never forward invalid geometry into a recovered or fabricated pose.
        img = (_render_frame(mujoco, model, observed_data, scratch, renderer, camera)
               if observation.state.finite else Image.new("RGB", (width, height), color=(10, 14, 22)))
        _draw_hud(img, title, profile.name, metadata)
        pil_frames.append(img)
        frame_metadata.append(metadata)
        key = None
        if terminal:
            img.save(endpoint_path)
            if sequence:
                abbreviations = {Limb.RIGHT_HAND: "rh", Limb.LEFT_HAND: "lh",
                                 Limb.LEFT_FOOT: "lf", Limb.RIGHT_FOOT: "rf"}
                index = observation.move_index + 1
                key = f"{index:02d}_move{index}_{abbreviations[observation.request.limb]}"
            else:
                key = "transition_endpoint.png"
        elif observation.phase is None:
            if observation.move_index == 0:
                key = "00_initial" if sequence else "transition_00_initial.png"
        elif not sequence:
            key = phase_names[observation.phase]
        if save_keyframes and key is not None and key not in saved_keyframes:
            name = f"sequence_{key}.png" if sequence else key
            path = out_dir / name
            img.save(path)
            saved_keyframes[key] = path
            keyframe_metadata[key] = metadata
            annotations[key] = (f"{label} | {metadata['limb']} {metadata['source_hold']} -> "
                                f"{metadata['target_hold']} | {metadata['phase'] or 'not_started'}")

    try:
        if initialization_error:
            request = requests[0]
            metadata = _state_metadata(initial_state, model, scene)
            metadata.update(_json_value({
                "move_index": 0, "total_moves": len(requests), "limb": request.limb,
                "source_hold": request.source_hold, "target_hold": request.target_hold,
                "phase": None, "steps": 0, "status": "INITIALIZATION_FAILURE",
                "status_label": "FAILURE/INITIALIZATION_FAILURE", "terminal": True,
                "reason": initialization_error, "initialization_error": initialization_error,
            }))
            img = (_render_frame(mujoco, model, data, scratch, renderer, camera)
                   if metadata["finite"] else Image.new("RGB", (width, height), color=(10, 14, 22)))
            _draw_hud(img, title, profile.name, metadata)
            img.save(endpoint_path)
            pil_frames.append(img)
            frame_metadata.append(metadata)
        else:
            physics_result = execute_transition_sequence(
                model=model, data=data, scene=scene, profile=profile, requests=requests,
                manager=manager, frame_callback=on_frame,
            )
    finally:
        renderer.close()

    mp4_path = out_dir / f"{prefix}_{profile.name}.mp4"
    gif_path = out_dir / f"{prefix}_{profile.name}.gif"
    export_errors: dict[str, str] = {}
    try:
        import imageio.v3 as iio
        iio.imwrite(str(mp4_path), np.stack([np.asarray(img) for img in pil_frames]), fps=fps)
    except Exception as exc:
        export_errors["mp4"] = str(exc)
        print(f"[WARNING] MP4 export failed: {exc}")
        mp4_path = None
    try:
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=int(1000.0 / fps), loop=0)
    except Exception as exc:
        export_errors["gif"] = str(exc)
        print(f"[WARNING] GIF export failed: {exc}")
        gif_path = None
    montage_name = ("sequence_montage.png" if sequence else
                    "transition_montage.png" if profile.name == "base" else
                    f"transition_montage_{profile.name}.png")
    montage_path = generate_transition_montage(
        saved_keyframes, out_dir / montage_name, annotations, columns=2 if sequence else 4)
    last_result = physics_result.transition_results[-1] if physics_result is not None else None
    success = physics_result.success if physics_result is not None else False
    status = last_result.status if last_result is not None else "INITIALIZATION_FAILURE"
    physical_success = success and contact_mode == ContactMode.PHYSICAL
    metadata = _json_value({
        "profile": profile.name,
        "rendering_mode": prefix, "contact_mode": contact_mode,
        "contact_mode_label": contact_mode_label(contact_mode), "physics_certified": False,
        "scientific_contacts": contact_mode == ContactMode.PHYSICAL,
        "nonphysical": contact_mode == ContactMode.IDEALIZED_DEBUG,
        "physical_success": physical_success, "initialization_error": initialization_error,
        "resolution": (width, height),
        "fps": fps,
        "requests": [{**_json_value(request), "contact_mode": contact_mode} for request in requests],
        "physics_result": physics_result,
        "success": success,
        "status": status,
        "frame_count": len(pil_frames),
        "frames": frame_metadata,
        "keyframes": keyframe_metadata,
        "endpoint": frame_metadata[-1],
        "artifacts": {"endpoint": endpoint_path, "mp4": mp4_path, "gif": gif_path,
                      "montage": montage_path, "keyframes": saved_keyframes},
        "export_errors": export_errors,
    })
    manifest_path = out_dir / f"{prefix}_{profile.name}.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{frame_metadata[-1]['status_label']}] {frame_metadata[-1]['reason']} | {contact_mode_label(contact_mode)}")
    print(f"[ARTIFACT] Endpoint: {endpoint_path}; manifest: {manifest_path}")
    return {
        "mp4": mp4_path,
        "gif": gif_path,
        "keyframes": saved_keyframes,
        "montage": montage_path,
        "endpoint": endpoint_path,
        "manifest": manifest_path,
        "metadata": metadata,
        "physics_result": physics_result,
        "success": success, "physical_success": physical_success,
        "status": status, "contact_mode": contact_mode,
        "initialization_error": initialization_error,
        "frame_count": len(pil_frames),
        "resolution": (width, height),
        "reattached": (last_result is not None and last_result.target_captured
                       and physics_result.final_contact_configuration.get(last_result.limb) == last_result.target_hold),
        "eligibility_detected": last_result.eligibility_detected if last_result is not None else False,
    }


def render_single_limb_reach_transition(
    profile_name: str = "base",
    output_dir: Path | None = None,
    limb: Limb = Limb.RIGHT_HAND,
    from_region_id: str = "H4",
    to_region_id: str = "H5",
    steps: int = 210,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    save_keyframes: bool = True,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> dict[str, Any]:
    contact_mode = ContactMode(contact_mode)
    options = {"max_attach_distance": .15} if contact_mode == ContactMode.IDEALIZED_DEBUG else {}
    request = TransitionRequest(limb=limb, source_hold=from_region_id,
                                 target_hold=to_region_id, steps=steps, **options)
    return _render_transitions(profile_name, output_dir, [request], fps, width, height,
                               save_keyframes, sequence=False, contact_mode=contact_mode, initial_qpos=initial_qpos)


def render_transition_sequence(
    profile_name: str = "base",
    output_dir: Path | None = None,
    steps_per_move: int = 210,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    save_keyframes: bool = True,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> dict[str, Any]:
    contact_mode = ContactMode(contact_mode)
    moves = _transition_requests("sequence", steps_per_move, contact_mode)
    return _render_transitions(profile_name, output_dir, moves, fps, width, height,
                               save_keyframes, sequence=True, contact_mode=contact_mode, initial_qpos=initial_qpos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless observation of Boulder Prototype physics")
    parser.add_argument("--profile", choices=list(PROFILES), default="base")
    parser.add_argument("--mode", choices=["all", "stance", "transition", "sequence"], default="all")
    parser.add_argument("--contact-mode", choices=[mode.value for mode in ContactMode], default="physical",
                        help="idealized_debug is explicitly NONPHYSICAL, not scientific validation")
    parser.add_argument("--output", default=str(HERE / "outputs" / "visual"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--steps", type=int, default=210, help="Physics step budget per transition")
    parser.add_argument("--no-keyframes", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    if args.mode in ("all", "stance"):
        profiles = ["compact_strong"] + ([args.profile] if args.profile != "compact_strong" else [])
        for profile in profiles:
            path = render_static_stance(profile, output_dir=out_dir, width=args.width, height=args.height,
                                        contact_mode=args.contact_mode)
            failed |= json.loads(path.with_suffix(".json").read_text())["initialization_error"] is not None
    for mode, render in (("transition", render_single_limb_reach_transition),
                         ("sequence", render_transition_sequence)):
        if args.mode not in ("all", mode):
            continue
        options = {"steps" if mode == "transition" else "steps_per_move": args.steps}
        result = render(profile_name=args.profile, output_dir=out_dir, fps=args.fps,
                        width=args.width, height=args.height, save_keyframes=not args.no_keyframes,
                        contact_mode=args.contact_mode,
                        **options)
        failed |= not result["success"]
    print("[ARTIFACTS] Rendering complete; physics outcomes are recorded in the manifests.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
