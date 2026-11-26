"""Bouldering wall and hold layout.

MuJoCo world frame: x = horizontal along the wall, y = wall normal (climber stands at -y),
z = up. The wall's climbing face is the plane y = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hold:
    name: str
    x: float
    z: float
    radius: float = 0.065
    friction: float = 0.85


@dataclass(frozen=True)
class Wall:
    width: float = 2.3
    height: float = 3.0
    thickness: float = 0.10
    friction: float = 0.95
    holds: tuple[Hold, ...] = field(default_factory=tuple)

    def hold(self, name: str) -> Hold:
        for h in self.holds:
            if h.name == name:
                return h
        raise KeyError(name)


def default_wall() -> Wall:
    """Layout inherited from the first physics prototype so results stay comparable."""
    return Wall(holds=(
        Hold("H1", -0.33, 0.72, 0.075, 0.90),
        Hold("H2", 0.33, 0.72, 0.075, 0.90),
        Hold("H3", -0.38, 1.48, 0.065, 0.82),
        Hold("H4", 0.35, 1.55, 0.060, 0.80),
        Hold("H5", 0.20, 1.62, 0.065, 0.85),
        Hold("H6", -0.20, 0.82, 0.070, 0.90),
        Hold("H7", -0.22, 1.68, 0.065, 0.85),
        Hold("TOP", 0.24, 2.60, 0.075, 0.88),
    ))
