from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from boulder_v1 import (
    ClimberProfile,
    ContactMode,
    END_EFFECTOR_SITE_NAMES,
    GraspManager,
    InitialContactError,
    Limb,
    STATIC_STANCE_TARGETS,
    build_mjcf,
    compile_model,
    compute_pose_control,
    execute_transition_sequence,
    get_end_effector_positions,
    get_state_summary,
    initialize_episode,
    make_synthetic_scene,
    mujoco_available,
    run_pose_control,
    simulate_static_stance,
    TransitionObservation,
    TransitionRequest,
    TransitionSequenceResult,
)
from boulder_v1.runtime import _import_mujoco

PROFILES: dict[str, ClimberProfile] = {
    "base": ClimberProfile(name="base"),
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
}


def is_ssh_session() -> bool:
    return bool(
        os.environ.get("SSH_CLIENT")
        or os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
    )


def configure_camera(cam: Any, model: Any) -> None:
    """Configure default viewer camera framing wall, holds, and humanoid.

    Derives lookat point and viewing distance dynamically from model.stat
    (scene bounding center and extent) rather than hard-coded screen coordinates.
    """
    center = model.stat.center
    extent = model.stat.extent

    # Center camera between wall (y ~ 0.05) and humanoid climber (y ~ -0.55)
    cam.lookat[0] = float(center[0])
    cam.lookat[1] = float(center[1] - 0.30)
    cam.lookat[2] = float(center[2] * 0.95)

    # Frame the full extent with comfortable margin
    cam.distance = float(extent * 1.15)
    cam.elevation = -12.0
    cam.azimuth = 105.0


def contact_mode_label(contact_mode: ContactMode | str) -> str:
    return ("IDEALIZED_DEBUG: NONPHYSICAL! NOT SCIENTIFIC"
            if ContactMode(contact_mode) == ContactMode.IDEALIZED_DEBUG else
            "PHYSICAL_DEFAULT: scientific contacts; CONTROLLER_NOT_VALIDATED")


def _initialization_state(model: Any, data: Any, scene: Any, profile: ClimberProfile) -> dict[str, Any]:
    """Observe a rejected episode without adopting it as an initialized session."""
    manager = GraspManager(model, data, scene, profile=profile)
    snapshot = manager.contact_snapshot()
    return {
        "time": float(data.time), "root_pos": tuple(float(v) for v in data.qpos[:3]),
        "root_vel": tuple(float(v) for v in data.qvel[:3]),
        "qvel_norm": math.sqrt(sum(float(v) ** 2 for v in data.qvel)),
        "finite": all(math.isfinite(float(v)) for values in
                      (data.qpos, data.qvel, data.ctrl, data.qacc, data.qacc_warmstart)
                      for v in values) and math.isfinite(float(data.time)),
        "qpos": tuple(float(v) for v in data.qpos),
        "qvel": tuple(float(v) for v in data.qvel),
        "ctrl": tuple(float(v) for v in data.ctrl),
        "eq_active": tuple(bool(v) for v in data.eq_active),
        "qacc_warmstart": tuple(float(v) for v in data.qacc_warmstart),
        "contact_configuration": snapshot.configuration,
        "attachment_constraints": {}, "attachment_loads": {},
        "hand_states": {limb: asdict(state) for limb, state in snapshot.hands.items()},
        "foot_states": {limb: asdict(state) for limb, state in snapshot.feet.items()},
        "contact_mode": snapshot.mode, "capture_events": (), "release_events": (),
    }


def _contact_lines(state: dict[str, Any]) -> list[str]:
    def number(value: Any, unit: str, precision: int = 2) -> str:
        return f"{value:.{precision}f}{unit}" if value is not None and math.isfinite(value) else "n/a"

    lines = []
    for limb, label in ((Limb.LEFT_HAND, "LH"), (Limb.RIGHT_HAND, "RH")):
        hand = state["hand_states"][limb]
        lines.append(f"HAND_GRASP {label}: target={hand['region_id'] or 'none'} active={hand['active']} "
                     f"load={number(hand['load'], 'N')} cap={number(hand['capacity'], 'N')} "
                     f"margin={number(hand['margin'], 'N')} valid={hand['valid']}")
    for limb, label in ((Limb.LEFT_FOOT, "LF"), (Limb.RIGHT_FOOT, "RF")):
        foot = state["foot_states"][limb]
        status = foot["status"]
        status = status.value if hasattr(status, "value") else status
        surfaces = ','.join(foot['support_regions']) or 'none'
        debug = (f" NONPHYSICAL eq={foot['idealized_attachment']}"
                 if foot['idealized_attachment'] else "")
        lines.append(f"FOOT_SUPPORT {label}: {status} contact={foot['contacting']} "
                     f"support={foot['supporting']} slip={foot['slipping']} "
                     f"Fn={number(foot['normal_force'], 'N')} Ft={number(foot['tangential_force'], 'N')} "
                     f"v={number(foot['tangential_speed'], 'm/s', 3)} surfaces={surfaces} "
                     f"n={len(foot['contacts'])} measured={foot['measurement_valid']}{debug}")
    return lines


def print_mode_banner(mode: str, contact_mode: ContactMode = ContactMode.PHYSICAL) -> None:
    print("=" * 60)
    print(f"Contact mode:   {contact_mode_label(contact_mode)}")
    if contact_mode == ContactMode.PHYSICAL:
        print("HAND_GRASP:     Capture <=1mm, speed <=0.05m/s, cos(30deg), penetration <1mm.")
        print("FOOT_SUPPORT:   STEP collision support, NO foot EQUALITY; Fn/Ft and slip measured.")
    else:
        print("Debug:          Hand/foot equalities and expanded capture gate; NONPHYSICAL!")
    if mode == "stance":
        print("[Viewer Mode: STANCE (Deterministic Static 4-Point Climbing Stance)]")
        print("Selection Note: Selected via CLI flag '--mode stance'.")
        print("                (This cannot be toggled from within the MuJoCo GUI window)")
        print("Physics:        REAL free-root physics under full gravity (NO root weld).")
        print("Requested:      LH H3, RH H4, LF H1, RF H2; actual contacts are sensor-derived.")
        print("Actuation:      Deterministic joint hold controller maintaining climbing pose.")
    elif mode == "sequence":
        print("[Viewer Mode: SEQUENCE (Deterministic Multi-Limb Sequence Without Reset)]")
        print("Selection Note: Selected via CLI flag '--mode sequence'.")
        print("Physics:        REAL continuous physical simulation without resets between moves.")
        print("Sequence:       Move 1: RH H4 -> H5 | Move 2: LF H1 -> H6 | Move 3: LH H3 -> H7")
        print("Contact roles:  Hand grasps and foot support reported separately.")
    elif mode == "transition":
        print("[Viewer Mode: TRANSITION (Deterministic Single-Limb Reach Transition)]")
        print("Selection Note: Selected via CLI flag '--mode transition'.")
        print("                (This cannot be toggled from within the MuJoCo GUI window)")
        print("Physics:        REAL free-root physics under full gravity (NO root weld).")
        print("Cadence:        Authoritative executor phases, including pre-shift and settling.")
        print("Grasp:          Dynamic detach and deterministic reattachment via GraspManager.")
    elif mode == "pose":
        print("[Viewer Mode: POSE (Actuator & Morphology Visualization)]")
        print("Selection Note: Selected via CLI flag '--mode pose'.")
        print("                (This cannot be toggled from within the MuJoCo GUI window)")
        print("Stabilization:  Temporary root support is ACTIVE for debug visualization.")
        print("                Enables inspecting joint actuation, morphology, and end-effector")
        print("                sites without falling under gravity.")
        print("                (NOTE: This is NOT a balance controller; hold grasp constraints")
        print("                 are demonstrated in '--mode stance').")
        print("Validation:     Debug visualization only, NOT transition validation.")
    else:
        print("[Viewer Mode: FREE (Real Free-Root Physics)]")
        print("Selection Note: Selected via CLI flag '--mode free'.")
        print("                (This cannot be toggled from within the MuJoCo GUI window)")
        print("Stabilization:  NONE (Unconstrained free-root physics under gravity).")
        print("                Falling to the floor is expected as the character has no")
        print("                standing balance policy or hold attachments yet.")
        print("Validation:     Free-physics debug only, NOT transition validation.")
    if mode in ("transition", "sequence"):
        print("Playback:       One execution only; success or failure endpoint stays frozen.")
    print("=" * 60)


def _transition_requests(mode: str, steps: int,
                         contact_mode: ContactMode = ContactMode.PHYSICAL) -> list[TransitionRequest]:
    options = {"steps": steps}
    if contact_mode == ContactMode.IDEALIZED_DEBUG:
        options["max_attach_distance"] = .15
    moves = [TransitionRequest(Limb.RIGHT_HAND, "H4", "H5", **options)]
    if mode == "sequence":
        moves.extend([
            TransitionRequest(Limb.LEFT_FOOT, "H1", "H6", **options),
            TransitionRequest(Limb.LEFT_HAND, "H3", "H7", **options),
        ])
    return moves


def _format_contacts(contacts: dict[Limb, str]) -> str:
    return ", ".join(f"{limb.value}:{hold}" for limb, hold in contacts.items()) or "none"


def _report_sequence_result(
    result: TransitionSequenceResult,
    total_moves: int,
    log: Callable[[str], None] = print,
) -> bool:
    log(f"  Contact mode: {contact_mode_label(result.final_state_summary.contact_mode)}")
    for index, move in enumerate(result.transition_results):
        phase = move.phases_traversed[-1] if move.phases_traversed else "none"
        log(
            f"  Move {index + 1}/{total_moves}: {move.limb.value} "
            f"{move.source_hold} -> {move.target_hold} "
            f"status={move.status.value} phase={phase}"
        )
        log(f"    Phases: {' -> '.join(move.phases_traversed)}")
        log(f"    Steps: {move.steps}; duration: {move.duration:.3f} s; time: {move.time:.3f} s")
        log(f"    Finite: {move.final_state.finite}; root z: {move.final_state.root_pos[2]:+.3f} m")
        log(f"    Contacts: {_format_contacts(move.final_contact_configuration)}")
        for line in _contact_lines(asdict(move.final_state)):
            log(f"    {line}")
        log(f"    Reason: {move.reason}")
    log(f"  Completed moves: {result.completed_moves}/{total_moves}")
    if result.failed_move_index is not None:
        log(f"  Failed move: {result.failed_move_index + 1}")
    log(f"  Total steps: {result.total_steps}; duration: {result.total_time:.3f} s")
    log(f"  Final time: {result.final_state_summary.time:.3f} s; finite: {result.final_state_summary.finite}")
    log(f"  Final contacts: {_format_contacts(result.final_contact_configuration)}")
    if result.success:
        log("[SUCCESS] Requested movement completed and settled.")
    else:
        log("[FAILURE] Requested movement did not complete.")
    return result.success


class _ViewerInterrupted(RuntimeError):
    """Cancel executor playback when its observer can no longer display it."""


def validate_scene_headlessly(
    profile_name: str = "base",
    mode: str = "stance",
    steps: int = 210,
    verbose: bool = True,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> bool:
    def _log(msg: str = "") -> None:
        if verbose:
            print(msg)

    profile = PROFILES.get(profile_name, PROFILES["base"])
    contact_mode = ContactMode(contact_mode)
    scene = make_synthetic_scene()
    xml = build_mjcf(scene, profile, contact_mode=contact_mode)
    model = compile_model(xml)
    mujoco = _import_mujoco()
    data = mujoco.MjData(model)

    _log("=" * 60)
    _log(f"=== Boulder Prototype Scene & Model Validation ({profile.name}) ===")
    _log(f"Mode: {mode.upper()}")
    _log(f"Contact mode: {contact_mode_label(contact_mode)}")
    _log("=" * 60)
    _log(f"Scene: {scene.metadata.get('name', 'boulder_scene')}")
    _log(f"  Walls: {len(scene.walls)} ({', '.join(w.id for w in scene.walls)})")
    _log(
        f"  Contact Regions: {len(scene.contact_regions)} "
        f"({', '.join(r.id for r in scene.contact_regions)})"
    )
    _log(f"  Actuated Joints: nu={model.nu}, nq={model.nq}, nv={model.nv}")

    _log("\n--- End-Effector Sites Inspection ---")
    inspection = mujoco.MjData(model)
    mujoco.mj_copyData(inspection, model, data)
    mujoco.mj_forward(model, inspection)
    ee_positions = get_end_effector_positions(model, inspection)
    for site_name in END_EFFECTOR_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise RuntimeError(f"Required end-effector site '{site_name}' missing from model!")
        pos = ee_positions[site_name]
        _log(
            f"  [OK] Site {site_name:18s} (id={site_id:2d}): "
            f"world pos = ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m"
        )

    if mode in ("transition", "sequence", "stance"):
        try:
            gm = initialize_episode(model, data, scene, profile=profile,
                                    attach_feet=contact_mode == ContactMode.IDEALIZED_DEBUG,
                                    initial_qpos=initial_qpos)
        except InitialContactError as exc:
            state = _initialization_state(model, data, scene, profile)
            _log(f"[FAILURE/INITIALIZATION_FAILURE] {exc}")
            if mode in ("transition", "sequence"):
                request = _transition_requests(mode, steps, contact_mode)[0]
                _log(f"  Requested: {request.limb.value} {request.source_hold} -> {request.target_hold}")
            _log(f"  Actual time={state['time']:.3f}s root={state['root_pos']}; no execution or reset.")
            for line in _contact_lines(state):
                _log(f"  {line}")
            return False
    if mode in ("transition", "sequence"):
        _log(f"\n--- Authoritative {mode.title()} Execution (No Reset) ---")
        moves = _transition_requests(mode, steps, contact_mode)
        result = execute_transition_sequence(model, data, scene, profile, moves, manager=gm)
        return _report_sequence_result(result, len(moves), _log)
    elif mode == "stance":
        _log("\n--- Deterministic Static Four-Point Stance Verification ---")
        result = simulate_static_stance(model, data, scene, profile, steps=steps, manager=gm)
        _log(f"  Steps simulated:     {result.steps} ({result.time:.3f} s)")
        _log(f"  Numerically finite:  {result.finite}")
        _log(f"  Supported on wall:   {result.supported} (final root z = {result.final_root_z:+.3f} m)")
        _log(f"  Left hand attached:  {result.left_hand_attached} (load = {result.left_hand_load:.1f} N)")
        _log(f"  Right hand attached: {result.right_hand_attached} (load = {result.right_hand_load:.1f} N)")
        for line in _contact_lines(asdict(get_state_summary(model, data, gm))):
            _log(f"  {line}")

        if not result.finite:
            raise RuntimeError("Static stance simulation resulted in non-finite values!")
        if not result.supported:
            raise RuntimeError("Climber fell from wall in static stance simulation!")
    else:
        _log("\n--- Pose/Free Debug Sanity Check (NOT Transition Validation) ---")
        stabilize_root = (mode == "pose")
        res = run_pose_control(xml, steps=steps, stabilize_root=stabilize_root)
        _log(f"  Root stabilization:  {stabilize_root}")
        _log(f"  Simulation steps:    {res.steps} ({res.time:.3f} s simulated)")
        _log(f"  Initial mean error:  {math.degrees(res.initial_error):.2f} deg")
        _log(f"  Final mean error:    {math.degrees(res.final_error):.2f} deg")
        _log(f"  Error reduced:       {res.error_reduced}")
        _log(f"  Numerically finite:  {res.finite}")

        if not res.finite:
            raise RuntimeError("Pose control sanity check resulted in non-finite values!")
        if not res.error_reduced:
            raise RuntimeError("Pose control failed to reduce error toward target pose!")

    if mode == "stance":
        _log("\n[SUCCESS] Static stance sanity check passed; no transition was requested.\n")
    else:
        _log("\n[SUCCESS] Debug sanity check passed; NOT transition validation.\n")
    return True


def launch_visual_viewer(
    profile_name: str = "base",
    mode: str = "stance",
    duration: float | None = None,
    steps: int = 210,
    contact_mode: ContactMode | str = ContactMode.PHYSICAL,
    initial_qpos: Any = None,
) -> int:
    profile = PROFILES.get(profile_name, PROFILES["base"])
    contact_mode = ContactMode(contact_mode)
    scene = make_synthetic_scene()
    xml = build_mjcf(scene, profile, contact_mode=contact_mode)
    model = compile_model(xml)
    mujoco = _import_mujoco()
    data = mujoco.MjData(model)

    import mujoco.viewer

    print_mode_banner(mode, contact_mode)
    print("Elements displayed:")
    print(f"  - Wall ({scene.walls[0].id})")
    print(f"  - Holds / Contact Regions: {len(scene.contact_regions)} holds with visual markers")
    print("  - Humanoid Climber with 4 high-contrast end-effector sites:")
    for name in END_EFFECTOR_SITE_NAMES:
        print(f"      * {name}")
    print("\nControls:")
    print("  - Left click + drag: Rotate camera")
    print("  - Right click + drag: Pan camera")
    print("  - Scroll wheel: Zoom")
    print("  - Close window or Ctrl+C to exit")
    print("=" * 60)

    initialization_error = None
    if mode in ("transition", "sequence", "stance"):
        try:
            gm = initialize_episode(model, data, scene, profile=profile,
                                    attach_feet=contact_mode == ContactMode.IDEALIZED_DEBUG,
                                    initial_qpos=initial_qpos)
        except InitialContactError as exc:
            initialization_error = str(exc)
            state = _initialization_state(model, data, scene, profile)
            failure_text = (f"FAILURE/INITIALIZATION_FAILURE: {exc}\n"
                            f"{contact_mode_label(contact_mode)}\n"
                            f"Actual time={state['time']:.3f}s root={state['root_pos']}\n"
                            + '\n'.join(_contact_lines(state)))
            if mode in ("transition", "sequence"):
                request = _transition_requests(mode, steps, contact_mode)[0]
                failure_text += f"\nRequested: {request.limb.value} {request.source_hold} -> {request.target_hold}"
            print(failure_text)
    if mode == "pose":
        init_root_qpos = data.qpos[0:7].copy()
    if mode == "stance" and not initialization_error:
        stance_targets = (STATIC_STANCE_TARGETS if contact_mode == ContactMode.IDEALIZED_DEBUG else
                          {model.joint(int(jid)).name: float(data.qpos[model.jnt_qposadr[jid]])
                           for jid in model.actuator_trnid[:, 0]})

    exit_code = 1 if initialization_error else 0
    display_model, display_data = model, data
    if mode in ("transition", "sequence", "stance"):
        # Passive launch/GUI sync can forward, reset, perturb, and edit both inputs.
        display_model = copy.copy(model)
        display_data = mujoco.MjData(display_model)
        mujoco.mj_copyData(display_data, display_model, data)
    with mujoco.viewer.launch_passive(display_model, display_data) as viewer:
        configure_camera(viewer.cam, display_model)

        def sync_display(source: Any = data) -> None:
            if display_data is not data:
                with viewer.lock():
                    mujoco.mj_copyData(display_data, display_model, source)
                    mujoco.mj_forward(display_model, display_data)
            viewer.sync()

        sync_display()

        start_time = time.monotonic()
        if initialization_error:
            if hasattr(viewer, "set_texts"):
                viewer.set_texts([(None, None, failure_text, "")])
            print("[Viewer] Initialization failure frozen; no physics steps or automatic replay.")
            while viewer.is_running():
                if duration is not None and time.monotonic() - start_time >= duration:
                    break
                sync_display()
                time.sleep(model.opt.timestep)
        elif mode in ("transition", "sequence"):
            moves = _transition_requests(mode, steps, contact_mode)
            last_progress = None
            next_frame_at = start_time
            last_sim_time = float(data.time)

            def observe(observation: TransitionObservation, observed_data: Any, manager: GraspManager) -> None:
                nonlocal last_progress, next_frame_at, last_sim_time
                if not viewer.is_running():
                    raise _ViewerInterrupted("viewer closed")
                if duration is not None and time.monotonic() - start_time >= duration:
                    raise _ViewerInterrupted("viewer duration expired")
                request = observation.request
                status = observation.status.value
                label = status
                if observation.result is not None and not observation.result.success:
                    label = f"FAILURE/{status}"
                phase = observation.phase.value if observation.phase is not None else "none"
                state = observation.state
                move_text = (
                    f"Move {observation.move_index + 1}/{observation.total_moves}: "
                    f"{request.limb.value} {request.source_hold} -> {request.target_hold} "
                    f"{label} status={status} phase={phase}"
                )
                state_text = (
                    f"steps={observation.steps} time={state.time:.3f} s finite={state.finite}\n"
                    f"{contact_mode_label(state.contact_mode)}\n"
                    + '\n'.join(_contact_lines(asdict(state)))
                )
                if observation.result is not None:
                    state_text += f"\nReason: {observation.result.reason}"
                progress = (observation.move_index, phase, status)
                if progress != last_progress:
                    print(f"[Viewer] {move_text}\n  {state_text}")
                    last_progress = progress
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts([(None, None, f"{move_text}\n{state_text}", "")])
                sync_display(observed_data)
                next_frame_at += max(0.0, state.time - last_sim_time)
                last_sim_time = state.time
                wait = next_frame_at - time.monotonic()
                if duration is not None:
                    wait = min(wait, duration - (time.monotonic() - start_time))
                if wait > 0:
                    time.sleep(wait)
                if not viewer.is_running():
                    raise _ViewerInterrupted("viewer closed")
                if duration is not None and time.monotonic() - start_time >= duration:
                    raise _ViewerInterrupted("viewer duration expired")

            try:
                result = execute_transition_sequence(
                    model, data, scene, profile, moves, manager=gm, frame_callback=observe
                )
            except (_ViewerInterrupted, KeyboardInterrupt) as exc:
                print(f"[INTERRUPTED] Movement not complete: {exc or 'Ctrl+C'}; time={data.time:.3f} s.")
                return 1
            exit_code = 0 if _report_sequence_result(result, len(moves)) else 1
            print("[Viewer] Endpoint frozen; no further physics steps or automatic replay.")
            while viewer.is_running():
                if duration is not None and time.monotonic() - start_time >= duration:
                    break
                sync_display()
                time.sleep(model.opt.timestep)
        else:
            while viewer.is_running():
                step_start = time.monotonic()
                if duration is not None and step_start - start_time >= duration:
                    break
                if mode == "stance":
                    compute_pose_control(model, data, target_pose=stance_targets)
                    if any(not d.maintain for d in gm.evaluate_and_update(profile).values()):
                        print("[FAILURE/GRASP_OVERLOAD] Static grasp guard rejected the next interval.")
                        sync_display()
                        exit_code = 1
                        break
                elif mode == "pose":
                    compute_pose_control(model, data)
                    # Temporary root support is confined to pose debug visualization.
                    data.qpos[0:7] = init_root_qpos
                    data.qvel[0:6] = 0.0

                mujoco.mj_step(model, data)
                if mode == "stance":
                    if any(not d.maintain for d in gm.evaluate_and_update(profile, applied_data=data).values()):
                        print("[FAILURE/GRASP_OVERLOAD] Static grasp guard rejected the applied interval.")
                        sync_display()
                        exit_code = 1
                        break

                if mode == "pose":
                    data.qpos[0:7] = init_root_qpos
                    data.qvel[0:6] = 0.0

                sync_display()
                wait = model.opt.timestep - (time.monotonic() - step_start)
                if wait > 0:
                    time.sleep(wait)

    print("[Viewer] Window closed cleanly.")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MuJoCo 3D Scene Viewer for Boulder Prototype v1.5: Retargeting, 3-Point Support & Reach Transition"
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="base",
        help="Climber profile preset to visualize (default: base)",
    )
    parser.add_argument(
        "--mode",
        choices=["stance", "transition", "sequence", "pose", "free", "passive"],
        default="stance",
        help=(
            "Simulation mode: "
            "'stance' (deterministic static 4-point climbing stance with active grasps; real free-root physics under gravity), "
            "'transition' (deterministic single-limb reach and reattach transition under free-root physics), "
            "'sequence' (deterministic multi-limb transition sequence without simulation reset), "
            "'pose' (actuator/morphology visualization with temporary root support), or "
            "'free' (unconstrained free-root physics under gravity; falling expected)"
        ),
    )
    parser.add_argument(
        "--contact-mode", choices=[mode.value for mode in ContactMode], default="physical",
        help="Physical contacts by default; idealized_debug is NONPHYSICAL, not scientific validation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=210,
        help="Main physics-step budget per transition, excluding executor settling (default: 210); headless debug/stance steps",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to run viewer before closing (default: run until closed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless verification without attempting to open any graphical window",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force visual viewer launch even if SSH session is detected",
    )

    args = parser.parse_args()
    if args.mode in ("free", "passive"):
        normalized_mode = "free"
    elif args.mode == "pose":
        normalized_mode = "pose"
    elif args.mode == "transition":
        normalized_mode = "transition"
    elif args.mode == "sequence":
        normalized_mode = "sequence"
    else:
        normalized_mode = "stance"

    if not mujoco_available():
        print("Error: MuJoCo is not installed in this environment.")
        print("Install with: pip install 'mujoco>=3.2,<4'")
        return 1

    if args.headless:
        return 0 if validate_scene_headlessly(
            profile_name=args.profile, mode=normalized_mode, steps=args.steps, contact_mode=args.contact_mode
        ) else 1

    if is_ssh_session() and not args.gui:
        print("[Notice] SSH remote session detected.")
        print("A graphical desktop is not assumed over SSH; use --gui only when a display is available.")
        print(f"Performing headless {normalized_mode} checks...\n")
        success = validate_scene_headlessly(profile_name=args.profile, mode=normalized_mode, steps=args.steps,
                                            contact_mode=args.contact_mode)
        print("=" * 60)
        print("To launch the viewer, run from a graphical desktop terminal:")
        print(
            f"   {sys.executable!r} {str(HERE / 'scripts' / 'view_scene.py')!r} "
            f"--gui --profile {args.profile} --mode {normalized_mode} --steps {args.steps} "
            f"--contact-mode {args.contact_mode}"
        )
        print("=" * 60)
        return 0 if success else 1

    try:
        return launch_visual_viewer(
            profile_name=args.profile,
            mode=normalized_mode,
            duration=args.duration,
            steps=args.steps,
            contact_mode=args.contact_mode,
        )
    except Exception as exc:
        print(f"\n[ERROR] Viewer unavailable or execution failed: {exc}")
        print("No automatic rerun was attempted. Use --headless for an explicit headless run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
