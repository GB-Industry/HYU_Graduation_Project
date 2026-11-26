"""Body profile: the physical traits that differ between climbers."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BodyProfile:
    """Segment lengths are metres; *_scale values are multipliers on the baseline body.

    The baseline is an adult male, 1.75 m (sole to head top) and 70 kg. Segment lengths
    follow Winter's height ratios (upper arm 0.186 H, forearm 0.146 H, thigh/shank ~0.245 H,
    shoulder height ~0.82 H, hip joint ~0.53 H). total_mass is split over the segments with
    de Leva (1996) male mass fractions (see model.MASS_FRACTION).

    torso_length spans hip joint level to the base of the neck. hip_width is the outer
    pelvis width; hip joint centres sit at +/- 0.85 * hip_width / 2. shoulder_width is
    the distance between the two shoulder joints.
    """

    name: str = "baseline"
    torso_length: float = 0.645
    shoulder_width: float = 0.40
    hip_width: float = 0.22
    upper_arm_length: float = 0.325
    forearm_length: float = 0.255
    thigh_length: float = 0.43
    shin_length: float = 0.43
    total_mass: float = 70.0     # kg
    rom_scale: float = 1.0       # joint range of motion multiplier
    strength_scale: float = 1.0  # joint torque ceiling multiplier
    grip_capacity: float = 850.0  # N, max hand load before a grasp breaks (not enforced yet)

    def __post_init__(self) -> None:
        for field in ("torso_length", "shoulder_width", "hip_width", "upper_arm_length",
                      "forearm_length", "thigh_length", "shin_length"):
            v = getattr(self, field)
            if not 0.05 <= v <= 2.0:
                raise ValueError(f"{field}={v} is outside [0.05, 2.0] m")
        for field in ("total_mass", "rom_scale", "strength_scale", "grip_capacity"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")

    def with_limbs(self, scale: float, name: str | None = None) -> "BodyProfile":
        """Scale all four limb segments together (e.g. 0.95 = 5% shorter arms and legs)."""
        return replace(self, name=name or f"{self.name}_limbs{scale:g}",
                       upper_arm_length=self.upper_arm_length * scale,
                       forearm_length=self.forearm_length * scale,
                       thigh_length=self.thigh_length * scale,
                       shin_length=self.shin_length * scale)

    @property
    def arm_reach(self) -> float:
        return self.upper_arm_length + self.forearm_length

    @property
    def leg_reach(self) -> float:
        return self.thigh_length + self.shin_length


BASELINE = BodyProfile()
SHORT_LIMBS = BASELINE.with_limbs(0.95, "short_limbs")
LONG_LIMBS = BASELINE.with_limbs(1.05, "long_limbs")
PRESETS = {p.name: p for p in (BASELINE, SHORT_LIMBS, LONG_LIMBS)}
