from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .schema import ContactRegion, Vec3

HAND_SITE_OFFSET = 0.006
FOOT_SITE_OFFSET = 0.011
SHOE_FRICTION = 1.8

Rotation = tuple[Vec3, Vec3, Vec3]


class ContactMode(str, Enum):
    PHYSICAL = "physical"
    IDEALIZED_DEBUG = "idealized_debug"


@dataclass(frozen=True)
class Frame:
    """World position and right-handed rotation stored as three matrix rows."""

    position: Vec3
    rotation: Rotation

    @property
    def normal(self) -> Vec3:
        return tuple(row[2] for row in self.rotation)

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Unit quaternion in MuJoCo's (w, x, y, z) order, without engine imports."""
        r = self.rotation
        trace = sum(r[i][i] for i in range(3))
        if trace > 0:
            s = 2 * math.sqrt(1 + trace)
            q = (s / 4, (r[2][1] - r[1][2]) / s,
                 (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
        else:
            i = max(range(3), key=lambda axis: r[axis][axis])
            j, k = (i + 1) % 3, (i + 2) % 3
            s = 2 * math.sqrt(1 + r[i][i] - r[j][j] - r[k][k])
            xyz = [0.0, 0.0, 0.0]
            xyz[i] = s / 4
            xyz[j] = (r[j][i] + r[i][j]) / s
            xyz[k] = (r[k][i] + r[i][k]) / s
            q = ((r[k][j] - r[j][k]) / s, *xyz)
        norm = math.hypot(*q)
        return tuple(value / norm for value in q)


@dataclass(frozen=True)
class ContactGeometry:
    body_frame: Frame
    hand_frame: Frame
    foot_surface_frame: Frame
    foot_frame: Frame
    shape: str
    size: tuple[float, ...]


def canonical_geometry(region: ContactRegion) -> ContactGeometry:
    """Canonical hold geometry, hand anchor, sole touch and foot-site reference.

    Body columns are (right, into hold, projected up). The hand's third
    column is outward; the foot's third column is the selected sole normal.
    Sphere size is (radius,); box size is (right, depth, up) half extents.
    """
    n = region.normal
    into = tuple(-value for value in n)
    up = tuple((1.0 if i == 2 else 0.0) - n[2] * n[i] for i in range(3))
    if math.hypot(*up) < 1e-12:
        up = tuple((1.0 if i == 1 else 0.0) - n[1] * n[i] for i in range(3))
    norm = math.hypot(*up)
    up = tuple(value / norm for value in up)
    right = (into[1] * up[2] - into[2] * up[1],
             into[2] * up[0] - into[0] * up[2],
             into[0] * up[1] - into[1] * up[0])
    norm = math.hypot(*right)
    right = tuple(value / norm for value in right)
    up = (right[1] * into[2] - right[2] * into[1],
          right[2] * into[0] - right[0] * into[2],
          right[0] * into[1] - right[1] * into[0])
    body_rotation = tuple(zip(right, into, up))
    hand_rotation = tuple(zip(right, up, n))

    if region.half_size is None:
        shape, size = "sphere", (region.radius,)
        depth = region.radius
        foot_normal = tuple(0.5 * n[i] + math.sqrt(3) / 2 * up[i] for i in range(3))
        foot_into = tuple(math.sqrt(3) / 2 * into[i] + 0.5 * up[i] for i in range(3))
        foot_rotation = tuple(zip(right, foot_into, foot_normal))
        foot_surface = tuple(region.position[i] + region.radius * foot_normal[i] for i in range(3))
    else:
        shape, size = "box", region.half_size
        depth = size[1]
        foot_normal = up
        foot_rotation = body_rotation
        foot_surface = tuple(region.position[i] + size[2] * up[i] for i in range(3))

    hand_position = tuple(region.position[i] + (depth + HAND_SITE_OFFSET) * n[i] for i in range(3))
    foot_position = tuple(foot_surface[i] + FOOT_SITE_OFFSET * foot_normal[i] for i in range(3))
    return ContactGeometry(
        body_frame=Frame(region.position, body_rotation),
        hand_frame=Frame(hand_position, hand_rotation),
        foot_surface_frame=Frame(foot_surface, foot_rotation),
        foot_frame=Frame(foot_position, foot_rotation),
        shape=shape,
        size=size,
    )
