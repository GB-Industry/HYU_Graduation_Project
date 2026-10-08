#!/usr/bin/env python3
"""Interactive, read-only playback of certified Stage 5.3 native observations."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from scripts.render_clean_transfers import (_actor_bounds, _case_input, _file_sha256,
                                          _load_verified_evidence, _matching_fixture)
from scripts.transfer_motion_audit import audit_motion
from scripts.validate_ascent import _case_verdict


@dataclass(frozen=True)
class FrameInfo:
    move_index: int
    phase: str
    contacts: tuple[tuple[str, str], ...]
    feet: tuple[tuple[str, str, float, bool], ...]
    points: tuple[tuple[tuple[float, float, float], float], ...]


@dataclass(frozen=True)
class Recording:
    model: mujoco.MjModel
    profile_name: str
    path: Path
    sha256: str
    dt: float
    times: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    eq_active: np.ndarray
    frames: tuple[FrameInfo, ...]


def _vector(row, key, length, label):
    if key not in row:
        raise ValueError(f"Missing native field {label}.{key}; capture new validated evidence, do not synthesize it")
    value = np.asarray(row[key], dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid native field {label}.{key}: expected finite shape {(length,)}")
    return value


def recording_from_evidence(model, evidence, path, digest):
    """Compact immutable display state, without copying the large reference traces.

    Warmstarts/full integration vectors are not present at every native sample.
    They are not needed for literal pose/constraint/contact observation playback;
    this stream must never be used to continue dynamics or recompute native loads.
    """
    if model.na or model.nmocap or model.nplugin:
        raise ValueError("Unsupported activation/mocap/plugin state: a separate full native capture is required")
    moves = evidence.get("moves")
    if not isinstance(moves, list) or len(moves) != 3:
        raise ValueError("A recorded three-move ascending sequence is required")
    rows, infos = [], []
    for index, move in enumerate(moves):
        if index and (move["initial_state"] != moves[index - 1]["final_state"]
                      or move.get("initial_integration_state") is None
                      or move["initial_integration_state"] != moves[index - 1].get("final_integration_state")):
            raise ValueError(f"Move {index + 1}: exact native boundary continuity is missing or broken")
        samples = move.get("samples")
        if not isinstance(samples, list) or len(samples) != move.get("steps"):
            raise ValueError(f"Move {index + 1}: complete native samples are required")
        states = [(move["initial_state"], False, "SOURCE_READY"),
                  *((row, True, row.get("phase", "UNKNOWN")) for row in samples),
                  (move["final_state"], False, "FINAL_READY")]
        for number, (state, sample, phase) in enumerate(states):
            label = f"move_{index + 1}.state_{number}"
            timestamp_key = "time_s" if sample else "time"
            if timestamp_key not in state or not isinstance(state[timestamp_key], (float, int)):
                raise ValueError(f"Missing native timestamp {label}.{timestamp_key}")
            timestamp = float(state[timestamp_key])
            if not math.isfinite(timestamp) or rows and timestamp < rows[-1][0]:
                raise ValueError(f"Nonfinite or reversed native clock at {label}; never sort away resets")
            if sample and (state.get("steps") != number or abs(timestamp - move["initial_state"]["time"]
                             - number * evidence["dt_s"]) > 1e-8):
                raise ValueError(f"Native sample cadence is missing or inconsistent at {label}")
            qpos = _vector(state, "qpos", model.nq, label)
            qvel = _vector(state, "qvel", model.nv, label)
            ctrl = _vector(state, "ctrl", model.nu, label)
            mask = _vector(state, "eq_active", model.neq, label)
            if np.any((mask != 0.) & (mask != 1.)):
                raise ValueError(f"Invalid native equality mask at {label}")
            contacts = state.get("contacts" if sample else "contact_configuration")
            hands = state.get("hands" if sample else "hand_states")
            feet = state.get("feet" if sample else "foot_states")
            if not all(isinstance(value, dict) for value in (contacts, hands, feet)):
                raise ValueError(f"Missing native contact/hand/foot observations at {label}")
            if set(hands) != {"LEFT_HAND", "RIGHT_HAND"} or set(feet) != {"LEFT_FOOT", "RIGHT_FOOT"}:
                raise ValueError(f"Incomplete native limb observations at {label}")
            if set(contacts) - set(hands) - set(feet) or any(not isinstance(h, str) for h in contacts.values()):
                raise ValueError(f"Invalid native contact identities at {label}")
            expected = np.zeros(model.neq, dtype=bool)
            for limb, hand in hands.items():
                active = hand.get("active")
                if not isinstance(active, bool) or active != (limb in contacts):
                    raise ValueError(f"Hand observation/configuration mismatch at {label}.{limb}")
                if active:
                    if hand.get("region_id") != contacts[limb]:
                        raise ValueError(f"Hand contact identity mismatch at {label}.{limb}")
                    expected[model.equality(f"grasp_{limb.lower()}_{contacts[limb]}").id] = True
            if not np.array_equal(mask.astype(bool), expected):
                raise ValueError(f"Native equality/contact activation mismatch at {label}")
            points, foot_info = [], []
            for limb, foot in feet.items():
                if not isinstance(foot.get("contacts"), (list, tuple)):
                    raise ValueError(f"Missing native foot contact list at {label}.{limb}")
                for contact in foot["contacts"]:
                    point = _vector(contact, "point", 3, label + "." + limb)
                    force = contact.get("normal_force")
                    if not isinstance(force, (float, int)) or not math.isfinite(force) or force < 0:
                        raise ValueError(f"Missing/invalid recorded foot force at {label}.{limb}")
                    points.append((tuple(float(v) for v in point), float(force)))
                force = foot.get("normal_force")
                if not isinstance(force, (float, int)) or not math.isfinite(force) or force < 0:
                    raise ValueError(f"Missing/invalid native foot load at {label}.{limb}")
                foot_info.append((limb, str(foot.get("status", "UNKNOWN")), float(force), bool(foot.get("slipping"))))
            rows.append((timestamp, qpos, qvel, ctrl, mask.astype(bool)))
            infos.append(FrameInfo(index, phase, tuple(sorted(contacts.items())), tuple(foot_info), tuple(points)))
    if evidence["initial_state"] != moves[0]["initial_state"] or evidence["final_state"] != moves[-1]["final_state"]:
        raise ValueError("Sequence endpoints differ from recorded native move endpoints")
    arrays = [np.asarray([row[column] for row in rows]) for column in range(5)]
    for array in arrays:
        array.flags.writeable = False
    # Checking unit quaternions does not normalize or repair the recorded poses.
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        qa = model.jnt_qposadr[joint]
        if np.any(np.abs(np.linalg.norm(arrays[1][:, qa + 3:qa + 7], axis=1) - 1.) > 1e-8):
            raise ValueError("Recorded root quaternion is not unit length")
    return Recording(model, evidence["profile"]["name"], Path(path), digest, float(evidence["dt_s"]),
                     *arrays, tuple(infos))


def load_recording(input_dir, profile="baseline", dt=.002):
    """Load certified saved inputs; no fixture runner, initializer, IK or control."""
    if profile not in ("baseline", "longer") or dt not in (.002, .001):
        raise ValueError("Select baseline/longer and native 2 ms/1 ms evidence")
    input_dir = Path(input_dir).resolve()
    with (input_dir / "report.json").open() as file:
        report = json.load(file)
    if report.get("passed") is not True or report.get("authority_acceptance") is not True:
        raise ValueError("Input bank is not a certified Stage 5.3 authority report")
    path, _ = _case_input("ascending_" + profile, input_dir, dt, "ascending")
    entry = next((r for r in report.get("runs", []) if r.get("name") == path.stem), None)
    digest = _file_sha256(path)
    if (entry is None or entry.get("physical_success") is not True or entry.get("accepted") is not True
            or entry.get("evidence_sha256") != digest):
        raise ValueError("Native recording is missing from the certified bank or its SHA-256 differs")
    evidence = _load_verified_evidence(path, "ascending")
    model, _, _, loaded_profile, _ = _matching_fixture(evidence, dt, "ascending")
    if loaded_profile.name != profile:
        raise ValueError("Recorded profile differs from the requested climber")
    recording = recording_from_evidence(model, evidence, path, digest)
    motion = audit_motion(model, evidence)
    if not _case_verdict(model, evidence, evidence["fixture_inputs"], motion, evidence["reference_identity"]):
        raise ValueError("Saved native ascent fails current contact/state/boundary certification")
    if _file_sha256(path) != digest:
        raise ValueError("Recording changed during loading")
    return recording


class Playback:
    """Monotonic wall-clock scheduling, choosing preceding literal native rows."""

    def __init__(self, recording, *, speed=1., autoplay=False, now=None):
        if speed not in (.5, 1., 2.):
            raise ValueError("Playback speed must be 0.5x, 1x or 2x")
        self.recording, self.speed, self.playing = recording, float(speed), bool(autoplay)
        self.index, self.cursor = 0, float(recording.times[0])
        self.last_wall = time.monotonic() if now is None else now

    def advance(self, now):
        elapsed = now - self.last_wall
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Playback requires a monotonic finite wall clock")
        self.last_wall = now
        if self.playing:
            self.cursor = min(float(self.recording.times[-1]), self.cursor + elapsed * self.speed)
            self.index = max(0, int(np.searchsorted(self.recording.times, self.cursor + 1e-10, side="right")) - 1)
            if self.cursor >= self.recording.times[-1]:
                self.playing = False

    def key(self, key, now):
        self.advance(now)
        if key == 32:  # Space
            self.playing = not self.playing
        elif key in (82, 268):  # R / Home: return to the native first pose, paused.
            self.index, self.cursor, self.playing = 0, float(self.recording.times[0]), False
        elif key in (49, 50, 51):
            self.speed = {49: .5, 50: 1., 51: 2.}[key]
        elif key in (262, 263, 44, 46):  # Arrow or comma/period: unique native time.
            self.playing = False
            current = self.recording.times[self.index]
            if key in (262, 46):
                next_index = int(np.searchsorted(self.recording.times, current, side="right"))
                if next_index < len(self.recording.times):
                    self.index = int(np.searchsorted(self.recording.times, self.recording.times[next_index], side="right")) - 1
            else:
                self.index = max(0, int(np.searchsorted(self.recording.times, current, side="left")) - 1)
            self.cursor = float(self.recording.times[self.index])


def apply_frame(model, data, recording, index):
    """Copy recorded display state only. No integration, normalization or solver."""
    data.qpos[:] = recording.qpos[index]
    data.qvel[:] = recording.qvel[index]
    data.ctrl[:] = recording.ctrl[index]
    data.eq_active[:] = recording.eq_active[index]
    data.time = recording.times[index]
    # Clear only display-owned mouse perturbations. They are never simulated.
    data.xfrc_applied[:] = 0.
    data.qfrc_applied[:] = 0.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def _contact_geoms(handle, model, data, info):
    """Draw measured foot points and recorded active hand connects, not new loads."""
    scene = handle.user_scn
    scene.ngeom = 0
    if len(info.points) + np.count_nonzero(data.eq_active) > scene.maxgeom:
        raise ValueError("Display scene cannot hold the recorded contact markers")
    for point, force in info.points:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.004, 0., 0.]),
                           np.array(point), np.eye(3).ravel(), np.array([.2, .9, .3, 1.] if force > 5. else [1., .6, .1, 1.]))
        scene.ngeom += 1
    for eid in np.flatnonzero(data.eq_active):
        a, b = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([.3, .6, 1., 1.]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .003, data.site_xpos[a], data.site_xpos[b])
        scene.ngeom += 1


def run_viewer(recording, *, speed=1., autoplay=False, duration=None, launch=None, clock=time.monotonic,
               sleep=time.sleep):
    """The viewer receives its own model and data, never a live simulation owner."""
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Viewer duration must be finite and positive")
    if launch is None:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("A graphical desktop is required (DISPLAY/WAYLAND_DISPLAY missing); use --check for headless validation")
        from mujoco import viewer
        launch = viewer.launch_passive
    display_model = copy.copy(recording.model)
    display_data = mujoco.MjData(display_model)
    apply_frame(display_model, display_data, recording, 0)  # Before the window opens.
    commands = SimpleQueue()
    started = clock()
    player = Playback(recording, speed=speed, autoplay=autoplay, now=started)
    markers = True
    with launch(display_model, display_data, key_callback=commands.put) as handle:
        with handle.lock():
            lo, hi = _actor_bounds(mujoco, display_model, display_data)
            handle.cam.lookat[:] = .5 * (lo + hi)
            handle.cam.distance = float(np.linalg.norm(.5 * (hi - lo))) * 1.2 / math.sin(math.radians(display_model.vis.global_.fovy / 2.))
            handle.cam.azimuth, handle.cam.elevation = 135., -5.
            handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        print("Space: play/pause | R/Home: restart paused | 1:0.5x 2:1x 3:2x | arrows or ,/.: native step | C: contact markers")
        print("Mouse: native MuJoCo rotate/zoom/pan. Recorded observations only; no physics or reference generation.")
        player.last_wall = clock()  # Window creation is not playback time.
        while handle.is_running():
            now = clock()
            if duration is not None and now - started >= duration:
                break
            player.advance(now)
            while True:
                try:
                    key = commands.get_nowait()
                except Empty:
                    break
                if key == 67:
                    markers = not markers
                else:
                    player.key(key, now)
            info = recording.frames[player.index]
            with handle.lock():
                apply_frame(display_model, display_data, recording, player.index)
                if markers:
                    _contact_geoms(handle, display_model, display_data, info)
                else:
                    handle.user_scn.ngeom = 0
            contacts = dict(info.contacts)
            text = (f"Stage 5.3 {recording.profile_name} | Native {display_data.time:.3f} s | "
                    f"Move {info.move_index + 1}/3 {info.phase}\n"
                    f"{'PLAY' if player.playing else 'PAUSED'} {player.speed:g}x | saved row {player.index + 1}/{len(recording.times)}\n"
                    + "  ".join(f"{l}: {contacts.get(name, 'none')}" for l, name in
                                (("LH", "LEFT_HAND"), ("RH", "RIGHT_HAND"), ("LF", "LEFT_FOOT"), ("RF", "RIGHT_FOOT")))
                    + "\nRecorded feet: " + "  ".join(f"{l}: {status.removeprefix('FOOT_')} {force:.1f} N{' SLIP' if slip else ''}"
                                                        for l, status, force, slip in info.feet)
                    + "\nSpace play/pause | R restart | 1/2/3 speed | arrows step | C markers")
            handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, "")])
            handle.sync(state_only=True)
            sleep(1. / 60.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "ascent-stage5.3-final")
    parser.add_argument("--profile", choices=("baseline", "longer"), default="baseline")
    parser.add_argument("--dt", type=float, choices=(.002, .001), default=.002)
    parser.add_argument("--speed", type=float, choices=(.5, 1., 2.), default=1.)
    parser.add_argument("--autoplay", action="store_true", help="default: paused at the actual climbing source")
    parser.add_argument("--duration", type=float, help="optional GUI wall-clock duration, never simulation steps")
    parser.add_argument("--check", action="store_true", help="validate saved playback state without opening a GUI")
    args = parser.parse_args(argv)
    try:
        recording = load_recording(args.input, args.profile, args.dt)
        print(f"Verified {recording.profile_name}: {len(recording.times)} saved rows, "
              f"native {recording.times[0]:.3f}..{recording.times[-1]:.3f} s; dt={recording.dt:g}")
        print("Exact recorded poses/equality masks/contacts; no interpolated event frames or native force recomputation.")
        if not args.check:
            run_viewer(recording, speed=args.speed, autoplay=args.autoplay, duration=args.duration)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Ascent viewer failed: {type(error).__name__}: {error}", file=sys.stderr)
        print("No legacy pose fallback, native execution, or evidence recapture was attempted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
