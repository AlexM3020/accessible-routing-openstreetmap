"""Deterministic shortest paths on a directed multigraph, with traceable snapping."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count
from time import perf_counter

from .errors import DataError, NoRouteFound
from .evaluation import Evaluator
from .geo import haversine_m
from .graph import InMemoryGraph
from .models import Coordinate, Edge, Route


@dataclass(frozen=True, slots=True)
class Snap:
    requested: Coordinate
    node_id: str
    coordinate: Coordinate
    distance_m: float

    def as_dict(self) -> dict:
        return {"requested_lat_lon": self.requested.as_lat_lon(), "node_id": self.node_id,
                "snapped_lat_lon": self.coordinate.as_lat_lon(), "distance_m": self.distance_m,
                "connection_verified": False}


def snap_endpoints(graph: InMemoryGraph, start: Coordinate, end: Coordinate, evaluator: Evaluator,
                   max_distance_m: float = 150) -> tuple[Snap, Snap]:
    snaps = []
    for point, adjacency in ((start, graph.adjacency), (end, graph.incoming)):
        match = graph.nearest_node(point, max_distance_m,
                                   lambda node_id: any(not evaluator(edge).blocked for edge in adjacency.get(node_id, ())))
        if match is None:
            raise NoRouteFound("No connected, permitted path node within the selected snap distance")
        snaps.append(Snap(point, match[0], graph.nodes[match[0]].coordinate, match[1]))
    if snaps[0].node_id == snaps[1].node_id:
        raise NoRouteFound("Origin and destination snap to the same node; select distinct path nodes")
    return snaps[0], snaps[1]


def find_route(graph: InMemoryGraph, start: str, goal: str, evaluator: Evaluator,
               algorithm: str = "astar", distance_only: bool = False) -> Route:
    """Distance-only still respects all hard restrictions of the selected profile."""
    if algorithm not in {"astar", "dijkstra"}:
        raise DataError("algorithm must be astar or dijkstra")
    if start not in graph.nodes or goal not in graph.nodes:
        raise NoRouteFound("Origin or destination node is absent from the graph")
    started = perf_counter()
    if start == goal:
        return Route((start,), (), 0.0, 0.0, algorithm=algorithm)
    sequence = count()
    queue = [(0.0, next(sequence), 0.0, start)]
    distances = {start: 0.0}
    predecessors: dict[str, Edge] = {}
    expanded = 0
    settled: set[str] = set()
    while queue:
        _, _, queued_cost, current = heapq.heappop(queue)
        if queued_cost != distances[current]:
            continue
        settled.add(current)
        if current == goal:
            break
        expanded += 1
        for edge in graph.adjacency.get(current, ()):
            evaluation = evaluator(edge)
            if evaluation.blocked:
                continue
            candidate = queued_cost + (edge.distance_m if distance_only else evaluation.cost_m)
            if candidate < distances.get(edge.target, float("inf")):
                distances[edge.target] = candidate
                predecessors[edge.target] = edge
                heuristic = haversine_m(graph.nodes[edge.target].coordinate, graph.nodes[goal].coordinate) if algorithm == "astar" else 0
                heapq.heappush(queue, (candidate + heuristic, next(sequence), candidate, edge.target))
    if goal not in predecessors:
        raise NoRouteFound("No path satisfies the selected hard requirements in this graph")
    edges = []
    node = goal
    while node != start:
        edge = predecessors[node]
        edges.append(edge)
        node = edge.source
    edges.reverse()
    return Route((start, *(edge.target for edge in edges)), tuple(edges), sum(e.distance_m for e in edges),
                 distances[goal], expanded, len(settled), perf_counter() - started, algorithm)