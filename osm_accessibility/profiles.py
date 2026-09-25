"""Explicit preference weights, hard requirements and tunable cost parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .errors import DataError
from .models import _finite_number

CRITERIA = (
    "avoid_bicycles", "lit_roads", "surface_type", "smoothness", "short_kerbs",
    "supervised_crossings", "road_width", "road_incline", "avoid_stairs", "wheelchair_accessible",
)
IMPORTANCE = {"Not Important": 0.0, "A Little Important": 0.25, "Important": 0.5,
              "Very Important": 0.75, "Essential": 1.0}


def finite_number(value: object, name: str, low: float, high: float) -> float:
    number = _finite_number(value, name)
    if not low <= number <= high:
        raise DataError(f"{name} must be between {low} and {high}")
    return number


@dataclass(frozen=True, slots=True)
class Profile:
    name: str = "custom"
    weights: Mapping[str, float] = field(default_factory=dict)
    wheelchair_required: bool = False
    allow_ramps: bool = True
    prefer_handrail: bool = True
    min_width_m: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataError("Profile name must be nonempty")
        if not isinstance(self.weights, Mapping):
            raise DataError("Profile weights must be a mapping")
        unknown = set(self.weights) - set(CRITERIA)
        if unknown:
            raise DataError(f"Unknown profile criteria: {sorted(unknown)}")
        weights = {name: finite_number(self.weights.get(name, 0), name, 0, 1) for name in CRITERIA}
        object.__setattr__(self, "weights", MappingProxyType(weights))
        finite_number(self.min_width_m, "min_width_m", 0.01, 10)
        for name in ("wheelchair_required", "allow_ramps", "prefer_handrail"):
            if not isinstance(getattr(self, name), bool):
                raise DataError(f"{name} must be a boolean")

    @classmethod
    def from_dict(cls, raw: Mapping) -> Profile:
        if not isinstance(raw, Mapping):
            raise DataError("A profile must be an object")
        allowed = {"name", "weights", "wheelchair_required", "allow_ramps", "prefer_handrail", "min_width_m"}
        if set(raw) - allowed:
            raise DataError(f"Unknown profile fields: {sorted(set(raw) - allowed)}")
        values = dict(raw)
        weights = values.get("weights", {})
        if not isinstance(weights, Mapping):
            raise DataError("Profile weights must be an object")
        values["weights"] = {name: IMPORTANCE.get(value, value) if isinstance(value, str) else value
                             for name, value in weights.items()}
        return cls(**values)

    def as_dict(self) -> dict:
        return {"name": self.name, "weights": dict(self.weights), "wheelchair_required": self.wheelchair_required,
                "allow_ramps": self.allow_ramps, "prefer_handrail": self.prefer_handrail, "min_width_m": self.min_width_m}


@dataclass(frozen=True, slots=True)
class CostParameters:
    accessibility_factor: float = 1.5
    unknown_risk: float = 0.5
    event_equivalent_m: float = 20.0
    incline_full_risk_percent: float = 12.0
    kerb_full_risk_m: float = 0.1

    def __post_init__(self) -> None:
        finite_number(self.accessibility_factor, "accessibility_factor", 0, 1_000)
        finite_number(self.unknown_risk, "unknown_risk", 0, 1)
        finite_number(self.event_equivalent_m, "event_equivalent_m", 0, 10_000)
        finite_number(self.incline_full_risk_percent, "incline_full_risk_percent", .01, 1_000)
        finite_number(self.kerb_full_risk_m, "kerb_full_risk_m", .001, 10)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, raw: Mapping) -> CostParameters:
        if not isinstance(raw, Mapping) or set(raw) - set(cls.__dataclass_fields__):
            raise DataError("Invalid cost parameter fields")
        return cls(**raw)


def wheelchair_profile() -> Profile:
    return Profile(name="wheelchair", wheelchair_required=True, weights={
        "surface_type": .75, "smoothness": .75, "short_kerbs": 1, "supervised_crossings": .5,
        "road_width": .5, "road_incline": .5, "avoid_stairs": 1, "wheelchair_accessible": 1,
        "lit_roads": .25, "avoid_bicycles": .25,
    })


def walking_profile() -> Profile:
    return Profile(name="walking", weights={"surface_type": .5, "smoothness": .5, "lit_roads": .5})


def profile_named(name: str) -> Profile:
    if name == "wheelchair":
        return wheelchair_profile()
    if name == "walking":
        return walking_profile()
    if name == "distance":
        return Profile(name="distance")
    raise DataError(f"Unknown built-in profile: {name}")