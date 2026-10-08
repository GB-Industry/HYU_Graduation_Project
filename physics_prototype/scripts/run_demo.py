from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from boulder_v1 import (
    ClimberProfile,
    ContactMode,
    GripController,
    InitialContactError,
    build_mjcf,
    can_attach,
    compile_model,
    execute_transition_sequence,
    get_state_summary,
    initialize_episode,
    make_synthetic_scene,
)
from boulder_v1.runtime import _import_mujoco, mujoco_available, run_pose_control, smoke_step
from boulder_v1.schema import Limb
from boulder_v1.visualize import render_profile_comparison
from scripts.render_demo import _json_value
from scripts.view_scene import _contact_lines, _initialization_state, _transition_requests, contact_mode_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Software demo, not a physics feasibility certificate")
    parser.add_argument("--contact-mode", choices=[mode.value for mode in ContactMode], default="physical",
                        help="idealized_debug is explicitly NONPHYSICAL, not scientific validation")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--steps", type=int, default=210)
    args = parser.parse_args()
    contact_mode = ContactMode(args.contact_mode)
    print(contact_mode_label(contact_mode))
    scene = make_synthetic_scene()
    long_reach = ClimberProfile(
        name="long_reach_lower_grip",
        upper_arm_length=0.34,
        forearm_length=0.30,
        thigh_length=0.44,
        shin_length=0.42,
        rom_scale=0.95,
        strength_scale=0.92,
        grip_capacity=850.0,
    )
    compact_strong = ClimberProfile(
        name="compact_strong",
        upper_arm_length=0.28,
        forearm_length=0.24,
        thigh_length=0.39,
        shin_length=0.37,
        rom_scale=1.08,
        strength_scale=1.15,
        grip_capacity=1000.0,
    )

    outputs = args.output.resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    scene_artifact = {**scene.to_dict(), "contact_mode": contact_mode.value,
                      "contact_mode_label": contact_mode_label(contact_mode)}
    (outputs / "scene.json").write_text(json.dumps(scene_artifact, indent=2), encoding="utf-8")

    for profile in (long_reach, compact_strong):
        xml = build_mjcf(scene, profile, contact_mode=contact_mode)
        (outputs / f"model_{profile.name}.xml").write_text(xml, encoding="utf-8")

    render_profile_comparison(scene, [long_reach, compact_strong], outputs / "profile_binding.png")

    grip_controller = GripController()
    grip_region = scene.region("H3")
    grip_results = {}
    for profile in (long_reach, compact_strong):
        decision = grip_controller.evaluate(
            profile, grip_region, (0.0, 1.0, 0.0), required_load=500.0
        )
        grip_results[profile.name] = {
            "contact_mode": contact_mode.value, "contact_mode_label": contact_mode_label(contact_mode),
            "scope": "offline capacity calculation; not an actual grasp",
            "required_load": decision.required_load,
            "effective_capacity": decision.effective_capacity,
            "maintain": decision.maintain,
            "utilization": decision.utilization,
        }
    (outputs / "grip_demo.json").write_text(json.dumps(grip_results, indent=2), encoding="utf-8")

    print("Generated canonical scene, two profile-conditioned MJCF models, and schematic image.")
    failed = False
    observations = {}
    if mujoco_available():
        mujoco = _import_mujoco()
        for profile in (long_reach, compact_strong):
            xml = build_mjcf(scene, profile, contact_mode=contact_mode)
            result = smoke_step(xml, steps=200)
            print(profile.name, result)
            ctrl_result = run_pose_control(xml, steps=300)
            print(
                f"{profile.name} pose control: steps={ctrl_result.steps} "
                f"initial_err={ctrl_result.initial_error:.3f}rad final_err={ctrl_result.final_error:.3f}rad "
                f"error_reduced={ctrl_result.error_reduced} finite={ctrl_result.finite} "
                f"sites={list(ctrl_result.end_effector_positions.keys())}"
            )

        print("\n--- Geometric Eligibility Only (No Capture Certificate) ---")
        h3 = scene.region("H3")
        eligible = can_attach(Limb.LEFT_HAND, h3, h3.position, (0.0, 1.0, 0.0))
        ineligible_limb = can_attach(Limb.LEFT_FOOT, h3, h3.position, (0.0, 1.0, 0.0))
        print(f"Contact eligibility check (H3): Hand={eligible}, Foot={ineligible_limb}")

        for profile in (long_reach, compact_strong):
            xml = build_mjcf(scene, profile, contact_mode=contact_mode)
            model = compile_model(xml)
            data = mujoco.MjData(model)
            result = None
            initialization_error = None
            try:
                manager = initialize_episode(model, data, scene, profile=profile,
                                             attach_feet=contact_mode == ContactMode.IDEALIZED_DEBUG)
            except InitialContactError as exc:
                initialization_error = str(exc)
                state = _initialization_state(model, data, scene, profile)
                status = "INITIALIZATION_FAILURE"
                print(f"[FAILURE/{status}] {profile.name}: {exc}; no contact execution or reset.")
            else:
                requests = _transition_requests("sequence", args.steps, contact_mode)
                result = execute_transition_sequence(model, data, scene, profile, requests, manager=manager)
                state = _json_value(get_state_summary(model, data, manager))
                status = result.transition_results[-1].status.value
                print(f"[{'SUCCESS' if result.success else 'FAILURE/' + status}] {profile.name}: "
                      f"{result.transition_results[-1].reason}")
            success = result.success if result is not None else False
            failed |= not success
            for line in _contact_lines(state):
                print(f"  {line}")
            observations[profile.name] = {
                "contact_mode": contact_mode, "contact_mode_label": contact_mode_label(contact_mode),
                "nonphysical": contact_mode == ContactMode.IDEALIZED_DEBUG,
                "scientific_contacts": contact_mode == ContactMode.PHYSICAL,
                "success": success, "physical_success": success and contact_mode == ContactMode.PHYSICAL,
                "physics_certified": False, "status": status,
                "initialization_error": initialization_error, "actual_state": state, "physics_result": result,
            }
    else:
        print("MuJoCo package is not installed in this environment; runtime compile/step smoke test skipped.")
        print("Install with: pip install 'mujoco>=3.2,<4'")
        failed = True
    (outputs / "demo_results.json").write_text(
        json.dumps(_json_value({"contact_mode": contact_mode, "contact_mode_label": contact_mode_label(contact_mode),
                                "nonphysical": contact_mode == ContactMode.IDEALIZED_DEBUG,
                                "scientific_contacts": contact_mode == ContactMode.PHYSICAL,
                                "physics_certified": False, "profiles": observations}),
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
