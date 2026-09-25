"""Deterministic graph snapshots and a bounded nearest-node spatial grid."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from math import asin, cos, degrees, floor, pi, radians, sin
from types import MappingProxyType

from .errors import DataError
from .geo import EARTH_RADIUS_M, haversine_m
from .models import Coordinate, Edge, Node, _finite_number

_MIN_CELL_SIZE_DEG = 1e-6
_MAX_CELL_SIZE_DEG = 180.0
_MAX_QUERY_CELLS = 10_000
_DISTANCE_TOLERANCE_M = 1e-6


class InMemoryGraph:
    """Keep every node, but index only nodes incident to an explicit edge.

    IDs and edge lists are sorted lexically. Cell sizes must lie between
    1e-6 and 180 degrees; large search envelopes scan connected nodes instead
    of enumerating an unbounded number of cells.
    """

    def __init__(
        self, nodes: Iterable[Node], edges: Iterable[Edge], cell_size_deg: float = 0.005
    ) -> None:
        cell_size = _finite_number(cell_size_deg, "Spatial cell size")
        if not _MIN_CELL_SIZE_DEG <= cell_size <= _MAX_CELL_SIZE_DEG:
            raise DataError(
                "Spatial cell size must be between 0.000001 and 180 degrees"
            )
        self._cell_size = cell_size

        node_map: dict[str, Node] = {}
        for node in nodes:
            if not isinstance(node, Node):
                raise DataError("Graph nodes must be Node records")
            if node.id in node_map:
                raise DataError(f"Duplicate node ID: {node.id}")
            node_map[node.id] = node
        self.nodes: Mapping[str, Node] = MappingProxyType(
            {key: node_map[key] for key in sorted(node_map)}
        )

        edge_list: list[Edge] = []
        edge_ids: set[str] = set()
        for edge in edges:
            if not isinstance(edge, Edge):
                raise DataError("Graph edges must be Edge records")
            if edge.id in edge_ids:
                raise DataError(f"Duplicate edge ID: {edge.id}")
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise DataError(f"Edge {edge.id} references a missing node")
            distance = _finite_number(
                edge.distance_m, "Edge distance", nonnegative=True
            )
            direct_distance = haversine_m(
                self.nodes[edge.source].coordinate, self.nodes[edge.target].coordinate
            )
            if distance + _DISTANCE_TOLERANCE_M < direct_distance:
                raise DataError(
                    f"Edge {edge.id} is shorter than its geographic endpoints"
                )
            edge_ids.add(edge.id)
            edge_list.append(edge)

        self.edges: tuple[Edge, ...] = tuple(
            sorted(edge_list, key=lambda edge: edge.id)
        )
        adjacency: dict[str, list[Edge]] = defaultdict(list)
        incoming: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source].append(edge)
            incoming[edge.target].append(edge)
        self.adjacency: Mapping[str, tuple[Edge, ...]] = MappingProxyType(
            {key: tuple(adjacency[key]) for key in sorted(adjacency)}
        )
        self.incoming: Mapping[str, tuple[Edge, ...]] = MappingProxyType(
            {key: tuple(incoming[key]) for key in sorted(incoming)}
        )
        self._connected_node_ids = tuple(
            node_id
            for node_id in self.nodes
            if node_id in adjacency or node_id in incoming
        )
        grid: dict[tuple[int, int], list[str]] = defaultdict(list)
        for node_id in self._connected_node_ids:
            grid[self._cell(self.nodes[node_id].coordinate)].append(node_id)
        self._grid: Mapping[tuple[int, int], tuple[str, ...]] = MappingProxyType(
            {key: tuple(grid[key]) for key in sorted(grid)}
        )

    def _cell(self, coordinate: Coordinate) -> tuple[int, int]:
        return (
            floor(coordinate.latitude / self._cell_size),
            floor(coordinate.longitude / self._cell_size),
        )

    def nearest_node(
        self, coord: Coordinate, max_distance_m: float = 150,
        eligible: Callable[[str], bool] | None = None,
    ) -> tuple[str, float] | None:
        """Return the nearest eligible connected node, breaking ties by ID."""
        if not isinstance(coord, Coordinate):
            raise DataError("Nearest-node coordinate must be a Coordinate")
        max_distance = _finite_number(max_distance_m, "Snap distance")
        if max_distance <= 0:
            raise DataError("Snap distance must be positive")
        if eligible is not None and not callable(eligible):
            raise DataError("Node eligibility must be callable")
        if not self._connected_node_ids:
            return None

        candidates = self._connected_node_ids
        radius = min(pi, max_distance / EARTH_RADIUS_M)
        latitude = radians(coord.latitude)
        lower_latitude = latitude - radius
        upper_latitude = latitude + radius

        # A spherical cap touching a pole can include every longitude.
        if lower_latitude > -pi / 2 and upper_latitude < pi / 2:
            lon_delta = degrees(asin(max(0.0, min(1.0, sin(radius) / cos(latitude)))))
            lower_longitude = coord.longitude - lon_delta
            upper_longitude = coord.longitude + lon_delta
            # Scanning also handles the two representations of the antimeridian.
            if lower_longitude > -180 and upper_longitude < 180:
                y_min = floor(degrees(lower_latitude) / self._cell_size) - 1
                y_max = floor(degrees(upper_latitude) / self._cell_size) + 1
                x_min = floor(lower_longitude / self._cell_size) - 1
                x_max = floor(upper_longitude / self._cell_size) + 1
                cell_count = (y_max - y_min + 1) * (x_max - x_min + 1)
                if cell_count <= min(_MAX_QUERY_CELLS, len(self._grid)):
                    candidates = tuple(
                        node_id
                        for y in range(y_min, y_max + 1)
                        for x in range(x_min, x_max + 1)
                        for node_id in self._grid.get((y, x), ())
                    )

        ranked = sorted(
            (haversine_m(coord, self.nodes[node_id].coordinate), node_id)
            for node_id in candidates
        )
        for distance, node_id in ranked:
            if distance > max_distance:
                break
            if eligible is None or eligible(node_id):
                return node_id, distance
        return None
