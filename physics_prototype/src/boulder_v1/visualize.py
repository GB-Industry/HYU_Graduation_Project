from __future__ import annotations

from pathlib import Path

from .schema import BoulderScene, ClimberProfile, SourceType


def render_profile_comparison(scene: BoulderScene, profiles: list[ClimberProfile], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 7), squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        wall = scene.walls[0]
        ax.set_xlim(-wall.size[0], wall.size[0])
        ax.set_ylim(0, wall.center[2] + wall.size[2])
        ax.set_aspect("equal")
        ax.set_title(profile.name)
        ax.set_xlabel(
            f"arm reach={profile.arm_reach:.2f}m | ROM x{profile.rom_scale:.2f}\n"
            f"strength x{profile.strength_scale:.2f} | grip={profile.grip_capacity:.0f}N"
        )

        for region in scene.contact_regions:
            if region.source_type == SourceType.WALL:
                continue
            x, _, z = region.position
            ax.scatter([x], [z], s=max(40, 500 * region.radius))
            ax.text(x + 0.025, z + 0.025, region.id, fontsize=8)

        # Simple front-view morphology sketch; not simulated motion.
        torso_top = 1.55 + profile.torso_length / 2
        torso_bottom = 1.55 - profile.torso_length / 2
        sx = profile.shoulder_width / 2
        hx = profile.hip_width / 2
        ax.plot([0, 0], [torso_bottom, torso_top], linewidth=5)
        ax.plot([-sx, sx], [torso_top * 0.97, torso_top * 0.97], linewidth=3)
        arm_y = torso_top * 0.97
        for sign in (-1, 1):
            elbow_x = sign * (sx + profile.upper_arm_length * 0.72)
            elbow_z = arm_y - profile.upper_arm_length * 0.69
            hand_x = elbow_x + sign * profile.forearm_length * 0.70
            hand_z = elbow_z - profile.forearm_length * 0.71
            ax.plot([sign * sx, elbow_x, hand_x], [arm_y, elbow_z, hand_z], linewidth=2)

            knee_x = sign * (hx + 0.05)
            knee_z = torso_bottom - profile.thigh_length
            foot_x = knee_x + sign * 0.04
            foot_z = knee_z - profile.shin_length
            ax.plot([sign * hx, knee_x, foot_x], [torso_bottom, knee_z, foot_z], linewidth=2)

        ax.text(
            0.02,
            0.02,
            "schematic only\nMuJoCo geometry is generated from the same profile",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
