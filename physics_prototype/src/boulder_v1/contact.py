from __future__ import annotations

from dataclasses import dataclass
import math

from .schema import Affordance, ClimberProfile, ContactRegion, Limb, Vec3
from .contact_geometry import canonical_geometry

CAPTURE_DISTANCE = 0.001
CAPTURE_SPEED = 0.05
CAPTURE_ORIENTATION = math.cos(math.radians(30))


def _unit(v: Vec3) -> Vec3:
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError("normal must have three finite components")
    n = math.hypot(*v)
    if n <= 1e-12:
        raise ValueError("normal vector must be non-zero")
    return tuple(x / n for x in v)  # type: ignore[return-value]


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def contact_allowed(limb: Limb, affordance: Affordance, region: ContactRegion) -> bool:
    if affordance not in region.affordances:
        return False
    if affordance in {Affordance.GRASP, Affordance.PRESS}:
        return limb.is_hand
    if affordance in {Affordance.STEP, Affordance.SMEAR}:
        return limb.is_foot
    return False


def orientation_compatibility(effector_normal: Vec3, region_normal: Vec3) -> float:
    """1 means the effector faces the surface; 0 means incompatible."""
    e = _unit(effector_normal)
    r = _unit(region_normal)
    return max(0.0, min(1.0, _dot(e, (-r[0], -r[1], -r[2]))))


def can_attach(
    limb: Limb,
    region: ContactRegion,
    effector_pos: Vec3,
    effector_normal: Vec3,
    affordance: Affordance = Affordance.GRASP,
    max_distance: float = CAPTURE_DISTANCE,
    min_orientation: float = CAPTURE_ORIENTATION,
    target_pos: Vec3 | None = None,
    relative_velocity: Vec3 = (0., 0., 0.),
    max_speed: float = CAPTURE_SPEED,
) -> bool:
    """Evaluate whether a limb end-effector is eligible to attach to a contact region.

    Reuses existing contact_allowed(...) and orientation_compatibility(...).
    Conditions:
    1. Limb and affordance must be compatible with the target region.
    2. End-effector must be within max_distance of the target position (defaulting to region.position).
    3. Effector normal must have compatible orientation (>= min_orientation).
    4. Target region must have positive friction and radius.
    """
    if not contact_allowed(limb, affordance, region):
        return False
    if not limb.is_hand or affordance != Affordance.GRASP:
        return False
    if region.friction <= 0 or region.radius <= 0:
        return False

    if (len(effector_pos) != 3 or len(relative_velocity) != 3
            or not all(math.isfinite(v) for v in (*effector_pos, *relative_velocity, max_distance, max_speed, min_orientation))
            or not 0 < max_distance <= CAPTURE_DISTANCE or not 0 < max_speed <= CAPTURE_SPEED
            or not CAPTURE_ORIENTATION <= min_orientation <= 1):
        return False
    ref_pos = target_pos if target_pos is not None else canonical_geometry(region).hand_frame.position
    dx = effector_pos[0] - ref_pos[0]
    dy = effector_pos[1] - ref_pos[1]
    dz = effector_pos[2] - ref_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(dist) or dist > max_distance + 1e-12 or math.hypot(*relative_velocity) > max_speed + 1e-12:
        return False

    orient = orientation_compatibility(effector_normal, region.normal)
    if orient < min_orientation:
        return False

    return True


def effective_grip_capacity(
    profile: ClimberProfile,
    region: ContactRegion,
    hand_normal: Vec3,
) -> float:
    capacity = profile.grip_capacity * region.grip_quality
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Hand capacity must be finite and positive in Newtons")
    return capacity


@dataclass(frozen=True)
class GripDecision:
    maintain: bool
    required_load: float
    effective_capacity: float
    utilization: float
    reason: str = ""


class GripController:
    def evaluate(
        self,
        profile: ClimberProfile,
        region: ContactRegion,
        hand_normal: Vec3,
        required_load: float,
    ) -> GripDecision:
        if not math.isfinite(required_load) or required_load < 0:
            raise ValueError("required load cannot be negative")
        capacity = effective_grip_capacity(profile, region, hand_normal)
        utilization = math.inf if capacity <= 1e-9 else required_load / capacity
        return GripDecision(
            maintain=required_load <= capacity,
            required_load=required_load,
            effective_capacity=capacity,
            utilization=utilization,
            reason="capacity exceeded" if required_load > capacity else "within bounded point-grasp capacity",
        )
