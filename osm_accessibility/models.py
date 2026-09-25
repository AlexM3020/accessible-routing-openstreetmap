"""Immutable graph records with WGS84 latitude/longitude coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose, isfinite

from .errors import DataError
from .tags import normalize_tags


def _normalize_id(value: object, name: str) -> str:
    """Canonicalize string and integer identifiers without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DataError(f"{name} must be a nonempty string or an integer")
    try:
        identifier = str(value).strip()
    except ValueError as exc:
        raise DataError(f"{name} cannot be represented as a string") from exc
    if not identifier:
        raise DataError(f"{name} must not be empty")
    return identifier


def _finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise DataError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise DataError(f"{name} must be a finite number")
    if nonnegative and number < 0:
        raise DataError(f"{name} must be nonnegative")
    return number


@dataclass(frozen=True, slots=True)
class Coordinate:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = _finite_number(self.latitude, "Latitude")
        longitude = _finite_number(self.longitude, "Longitude")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise DataError("Latitude or longitude is outside WGS84 bounds")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)

    @classmethod
    def from_lat_lon(cls, value: list[float] | tuple[float, float]) -> Coordinate:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise DataError("A coordinate must contain latitude and longitude")
        return cls(value[0], value[1])

    @classmethod
    def from_geojson(cls, value: list[float] | tuple[float, float]) -> Coordinate:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise DataError("A GeoJSON coordinate must contain longitude and latitude")
        longitude, latitude = value
        return cls(latitude, longitude)

    def as_lat_lon(self) -> list[float]:
        return [self.latitude, self.longitude]

    def as_geojson(self) -> list[float]:
        return [self.longitude, self.latitude]


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    coordinate: Coordinate
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_id(self.id, "Node ID"))
        if not isinstance(self.coordinate, Coordinate):
            raise DataError("Node coordinate must be a Coordinate")
        object.__setattr__(self, "tags", normalize_tags(self.tags))


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    source: str
    target: str
    way_id: str
    distance_m: float
    tags: Mapping[str, str] = field(default_factory=dict)
    crossing_tags: Mapping[str, str] = field(default_factory=dict)
    source_tags: Mapping[str, str] = field(default_factory=dict)
    target_tags: Mapping[str, str] = field(default_factory=dict)
    way_has_crossing_nodes: bool = False
    way_has_kerb_nodes: bool = False

    def __post_init__(self) -> None:
        for name in ("id", "source", "target", "way_id"):
            object.__setattr__(
                self, name, _normalize_id(getattr(self, name), f"Edge {name}")
            )
        object.__setattr__(
            self,
            "distance_m",
            _finite_number(self.distance_m, "Edge distance", nonnegative=True),
        )
        for name in ("tags", "crossing_tags", "source_tags", "target_tags"):
            object.__setattr__(self, name, normalize_tags(getattr(self, name)))
        for name in ("way_has_crossing_nodes", "way_has_kerb_nodes"):
            if not isinstance(getattr(self, name), bool):
                raise DataError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class Route:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    distance_m: float
    cost: float
    expanded_nodes: int = 0
    settled_nodes: int = 0
    elapsed_seconds: float = 0.0
    algorithm: str = "astar"

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, (list, tuple)) or not isinstance(
            self.edges, (list, tuple)
        ):
            raise DataError("Route nodes and edges must be ordered sequences")
        object.__setattr__(
            self,
            "nodes",
            tuple(_normalize_id(node, "Route node ID") for node in self.nodes),
        )
        if any(not isinstance(edge, Edge) for edge in self.edges):
            raise DataError("Route edges must be Edge records")
        object.__setattr__(self, "edges", tuple(self.edges))
        for name in ("distance_m", "cost", "elapsed_seconds"):
            object.__setattr__(
                self, name, _finite_number(getattr(self, name), name, nonnegative=True)
            )
        for name in ("expanded_nodes", "settled_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DataError(f"{name} must be a nonnegative integer")
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise DataError("Route algorithm must be a nonempty string")
        object.__setattr__(self, "algorithm", self.algorithm.strip())
        if not self.nodes or len(self.nodes) != len(self.edges) + 1:
            raise DataError("Route must have exactly one more node than edges")
        if any((edge.source, edge.target) != (self.nodes[index], self.nodes[index + 1])
               for index, edge in enumerate(self.edges)):
            raise DataError("Route edges must form the directed node sequence")
        if not isclose(self.distance_m, sum(edge.distance_m for edge in self.edges), rel_tol=1e-12, abs_tol=1e-6):
            raise DataError("Route distance must equal its summed edge lengths")
        if self.cost + 1e-6 < self.distance_m:
            raise DataError("Route cost cannot be smaller than its physical distance")
