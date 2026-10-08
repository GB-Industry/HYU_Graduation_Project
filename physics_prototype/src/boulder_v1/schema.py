from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from numbers import Real

Vec3 = tuple[float, float, float]


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _positive_real(value: object, name: str) -> float:
    number = _finite_real(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _tuple(value: object, name: str) -> tuple:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a collection")
    return tuple(value)


def _vec3(value: object, name: str) -> Vec3:
    if isinstance(value, (set, frozenset)):
        raise ValueError(f"{name} must be an ordered vector")
    components = _tuple(value, name)
    if len(components) != 3:
        raise ValueError(f"{name} must have exactly three components")
    x, y, z = (_finite_real(v, name) for v in components)
    return x, y, z


def _unit_normal(value: object, name: str) -> Vec3:
    normal = _vec3(value, name)
    norm = math.hypot(*normal)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError(f"{name} must have a finite nonzero norm")
    # Scaling keeps even subnormal directions accurate during normalization.
    scale = max(abs(v) for v in normal)
    x, y, z = (v / scale for v in normal)
    norm = math.hypot(x, y, z)
    return x / norm, y / norm, z / norm


class _ImmutableMapping(Mapping):
    __slots__ = ("_items",)

    def __init__(self, values: Mapping):
        if hasattr(self, "_items"):
            raise TypeError("mapping is immutable")
        object.__setattr__(self, "_items", tuple(values.items()))

    def __setattr__(self, name, value):
        raise TypeError("mapping is immutable")

    def __delattr__(self, name):
        raise TypeError("mapping is immutable")

    def __getitem__(self, key):
        for existing_key, value in self._items:
            if existing_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, value in self._items)

    def __len__(self):
        return len(self._items)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Scene validation limits contents to immutable strings and Limb keys.
        return self


class SourceType(str, Enum):
    HOLD = "HOLD"
    WALL = "WALL"
    VOLUME = "VOLUME"
    EDGE = "EDGE"


class Affordance(str, Enum):
    GRASP = "GRASP"
    STEP = "STEP"
    SMEAR = "SMEAR"
    PRESS = "PRESS"


class Limb(str, Enum):
    LEFT_HAND = "LEFT_HAND"
    RIGHT_HAND = "RIGHT_HAND"
    LEFT_FOOT = "LEFT_FOOT"
    RIGHT_FOOT = "RIGHT_FOOT"

    @property
    def is_hand(self) -> bool:
        return self in {Limb.LEFT_HAND, Limb.RIGHT_HAND}

    @property
    def is_foot(self) -> bool:
        return self in {Limb.LEFT_FOOT, Limb.RIGHT_FOOT}


@dataclass(frozen=True)
class ClimberProfile:
    name: str
    torso_length: float = 0.55
    shoulder_width: float = 0.42
    hip_width: float = 0.30
    upper_arm_length: float = 0.31
    forearm_length: float = 0.27
    thigh_length: float = 0.42
    shin_length: float = 0.40
    mass_scale: float = 1.0
    rom_scale: float = 1.0
    strength_scale: float = 1.0
    power_scale: float = 1.0  # Reserved metadata; unused by the current dynamics.
    grip_capacity: float = 850.0

    def __post_init__(self) -> None:
        _nonempty_string(self.name, "profile name")
        positive = {
            "torso_length": self.torso_length,
            "shoulder_width": self.shoulder_width,
            "hip_width": self.hip_width,
            "upper_arm_length": self.upper_arm_length,
            "forearm_length": self.forearm_length,
            "thigh_length": self.thigh_length,
            "shin_length": self.shin_length,
            "mass_scale": self.mass_scale,
            "rom_scale": self.rom_scale,
            "strength_scale": self.strength_scale,
            "power_scale": self.power_scale,
            "grip_capacity": self.grip_capacity,
        }
        for name, value in positive.items():
            object.__setattr__(self, name, _positive_real(value, f"profile {name}"))

    @property
    def arm_reach(self) -> float:
        return self.upper_arm_length + self.forearm_length

    @property
    def leg_reach(self) -> float:
        return self.thigh_length + self.shin_length

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WallSurface:
    id: str
    center: Vec3
    size: Vec3
    normal: Vec3 = (0.0, -1.0, 0.0)
    friction: float = 0.9

    def __post_init__(self) -> None:
        _nonempty_string(self.id, "wall id")
        object.__setattr__(self, "center", _vec3(self.center, "wall center"))
        object.__setattr__(self, "size", _vec3(self.size, "wall size"))
        object.__setattr__(self, "normal", _unit_normal(self.normal, "wall normal"))
        if any(v <= 0 for v in self.size):
            raise ValueError("wall size values must be positive")
        object.__setattr__(self, "friction", _positive_real(self.friction, "wall friction"))


@dataclass(frozen=True)
class ContactRegion:
    id: str
    source_type: SourceType
    position: Vec3
    normal: Vec3
    friction: float
    affordances: frozenset[Affordance]
    radius: float = 0.06
    grip_quality: float = 1.0
    half_size: Vec3 | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.id, "contact region id")
        if not isinstance(self.source_type, SourceType):
            raise ValueError("contact source_type must be a SourceType")
        object.__setattr__(self, "position", _vec3(self.position, "contact position"))
        object.__setattr__(self, "normal", _unit_normal(self.normal, "contact normal"))
        for name in ("friction", "radius", "grip_quality"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), f"contact {name}"))
        if self.half_size is not None:
            half_size = _vec3(self.half_size, "contact half_size")
            if any(v <= 0 for v in half_size):
                raise ValueError("contact half_size values must be positive")
            object.__setattr__(self, "half_size", half_size)
        affordances = _tuple(self.affordances, "contact affordances")
        if not affordances:
            raise ValueError("at least one affordance is required")
        if any(not isinstance(value, Affordance) for value in affordances):
            raise ValueError("contact affordances must contain Affordance values")
        object.__setattr__(self, "affordances", frozenset(affordances))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["affordances"] = sorted(a.value for a in self.affordances)
        return d


@dataclass(frozen=True)
class BoulderScene:
    coordinate_system: str
    scale: float
    walls: tuple[WallSurface, ...]
    contact_regions: tuple[ContactRegion, ...]
    start_configuration: Mapping[Limb, str] = field(default_factory=dict)
    goal_regions: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.coordinate_system, "coordinate_system")
        object.__setattr__(self, "scale", _positive_real(self.scale, "scene scale"))
        walls = _tuple(self.walls, "scene walls")
        regions = _tuple(self.contact_regions, "scene contact_regions")
        goals = _tuple(self.goal_regions, "scene goal_regions")
        if any(not isinstance(wall, WallSurface) for wall in walls):
            raise ValueError("scene walls must contain WallSurface values")
        if any(not isinstance(region, ContactRegion) for region in regions):
            raise ValueError("scene contact_regions must contain ContactRegion values")
        wall_ids = [wall.id for wall in walls]
        if len(wall_ids) != len(set(wall_ids)):
            raise ValueError("wall ids must be unique")
        ids = [region.id for region in regions]
        if len(ids) != len(set(ids)):
            raise ValueError("contact region ids must be unique")
        if not isinstance(self.start_configuration, Mapping):
            raise ValueError("start_configuration must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        starts = dict(self.start_configuration)
        metadata = dict(self.metadata)
        for limb, region_id in starts.items():
            if not isinstance(limb, Limb):
                raise ValueError("start_configuration keys must be Limb values")
            _nonempty_string(region_id, "start_configuration region id")
        for region_id in goals:
            _nonempty_string(region_id, "goal region id")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items()):
            raise ValueError("metadata keys and values must be strings")
        known = set(ids)
        unknown = [rid for rid in starts.values() if rid not in known]
        unknown += [rid for rid in goals if rid not in known]
        if unknown:
            raise ValueError(f"unknown contact region ids: {unknown}")
        object.__setattr__(self, "walls", walls)
        object.__setattr__(self, "contact_regions", regions)
        object.__setattr__(self, "goal_regions", goals)
        object.__setattr__(self, "start_configuration", _ImmutableMapping(starts))
        object.__setattr__(self, "metadata", _ImmutableMapping(metadata))

    def region(self, region_id: str) -> ContactRegion:
        for region in self.contact_regions:
            if region.id == region_id:
                return region
        raise KeyError(region_id)

    def to_dict(self) -> dict:
        return {
            "coordinate_system": self.coordinate_system,
            "scale": self.scale,
            "walls": [asdict(w) for w in self.walls],
            "contact_regions": [r.to_dict() for r in self.contact_regions],
            "start_configuration": {k.value: v for k, v in self.start_configuration.items()},
            "goal_regions": list(self.goal_regions),
            "metadata": dict(self.metadata),
        }


def affordance_set(values: Iterable[Affordance]) -> frozenset[Affordance]:
    return frozenset(values)
