from __future__ import annotations

from .schema import Affordance, BoulderScene, ContactRegion, Limb, SourceType, WallSurface, affordance_set


def make_synthetic_scene() -> BoulderScene:
    wall = WallSurface(
        id="wall_main",
        center=(0.0, 0.05, 1.5),
        size=(1.15, 0.05, 1.50),
        normal=(0.0, -1.0, 0.0),
        friction=0.95,
    )

    holds = (
        ContactRegion("H1", SourceType.HOLD, (-0.33, -0.02, 0.72), (0.0, -1.0, 0.0), 0.90,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.075, 1.15),
        ContactRegion("H2", SourceType.HOLD, (0.33, -0.02, 0.72), (0.0, -1.0, 0.0), 0.90,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.075, 1.15),
        ContactRegion("H3", SourceType.HOLD, (-0.38, -0.02, 1.48), (0.0, -1.0, 0.0), 0.82,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.065, 0.92),
        ContactRegion("H4", SourceType.HOLD, (0.35, -0.02, 1.55), (0.0, -1.0, 0.0), 0.80,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.060, 0.86),
        ContactRegion("H5", SourceType.HOLD, (0.20, -0.02, 1.62), (0.0, -1.0, 0.0), 0.85,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.065, 0.90),
        # Prototype v1.6 additions for bilateral transitions and foot repositioning
        ContactRegion("H6", SourceType.HOLD, (-0.20, -0.02, 0.82), (0.0, -1.0, 0.0), 0.90,
                      affordance_set([Affordance.STEP, Affordance.GRASP]), 0.070, 1.10),
        ContactRegion("H7", SourceType.HOLD, (-0.22, -0.02, 1.68), (0.0, -1.0, 0.0), 0.85,
                      affordance_set([Affordance.GRASP, Affordance.STEP]), 0.065, 0.90),
        ContactRegion("TOP", SourceType.HOLD, (0.24, -0.02, 2.60), (0.0, -1.0, 0.0), 0.88,
                      affordance_set([Affordance.GRASP]), 0.075, 1.05),
        ContactRegion("WALL_PATCH", SourceType.WALL, (0.60, 0.0, 1.05), (0.0, -1.0, 0.0), 1.05,
                      affordance_set([Affordance.SMEAR, Affordance.PRESS]), 0.14, 1.0),
    )

    return BoulderScene(
        coordinate_system="MuJoCo world: x horizontal, y wall-normal, z up",
        scale=1.0,
        walls=(wall,),
        contact_regions=holds,
        start_configuration={
            Limb.LEFT_FOOT: "H1",
            Limb.RIGHT_FOOT: "H2",
            Limb.LEFT_HAND: "H3",
            Limb.RIGHT_HAND: "H4",
        },
        goal_regions=("TOP",),
        metadata={"name": "prototype-v1-synthetic-wall", "purpose": "M0 physics foundation"},
    )
