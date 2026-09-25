"""Construct explicit OSM topology from complete node and way documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

from .errors import DataError
from .geo import haversine_m
from .graph import InMemoryGraph
from .models import Coordinate, Edge, Node, _normalize_id
from .tags import is_crossing, normalize_tags, reverse_incline


def build_graph(
    node_documents: Iterable[Mapping[str, Any]],
    way_documents: Iterable[Mapping[str, Any]],
) -> InMemoryGraph:
    """Read GeoJSON point coordinates and ordered OSM node references.

    Keep every raw node and every highway type, including access-restricted
    ways for obstacle plots. Only explicit pedestrian direction tags suppress
    directed edges. Geometric intersections never create connections.
    """
    nodes: dict[str, Node] = {}
    for document in node_documents:
        if not isinstance(document, Mapping):
            raise DataError("Node documents must be mappings")
        node_id = _normalize_id(document.get("id"), "Node ID")
        if node_id in nodes:
            raise DataError(f"Duplicate node ID: {node_id}")
        location = document.get("location")
        if not isinstance(location, Mapping) or location.get(
            "type", "Point"
        ) != "Point":
            raise DataError(f"Node {node_id} must have a GeoJSON point location")
        nodes[node_id] = Node(
            node_id,
            Coordinate.from_geojson(location.get("coordinates")),
            document.get("tags", {}),
        )

    ways: dict[str, tuple[tuple[str, ...], Mapping[str, str]]] = {}
    way_ids: set[str] = set()
    for document in way_documents:
        if not isinstance(document, Mapping):
            raise DataError("Way documents must be mappings")
        way_id = _normalize_id(document.get("id"), "Way ID")
        if way_id in way_ids:
            raise DataError(f"Duplicate way ID: {way_id}")
        way_ids.add(way_id)
        tags = normalize_tags(document.get("tags", {}))
        if not tags.get("highway"):
            continue
        references = document.get("nodes", ())
        if not isinstance(references, (list, tuple)):
            raise DataError(f"Way {way_id} nodes must be an ordered sequence")
        node_ids = tuple(_normalize_id(value, "Way node ID") for value in references)
        missing = set(node_ids) - nodes.keys()
        if missing:
            missing_ids = ", ".join(sorted(missing)[:5])
            raise DataError(f"Way {way_id} references missing nodes: {missing_ids}")
        ways[way_id] = (node_ids, tags)

    edges: list[Edge] = []
    empty_tags = normalize_tags(None)
    for way_id in sorted(ways):
        node_ids, tags = ways[way_id]
        crossing_nodes: set[str] = set()
        way_has_kerb_nodes = False
        # Inspect all references once, not just the endpoints of each segment.
        for node_id in node_ids:
            point_tags = nodes[node_id].tags
            if is_crossing(point_tags):
                crossing_nodes.add(node_id)
                # A crossing also contributes a kerb observation, possibly
                # unknown. Suppress redundant way-level kerb scoring throughout
                # this way, not just on the segment arriving at the crossing.
                way_has_kerb_nodes = True
            if (
                "kerb" in point_tags
                or "kerb:height" in point_tags
                or point_tags.get("barrier") == "kerb"
            ):
                way_has_kerb_nodes = True
        way_has_crossing_nodes = bool(crossing_nodes)

        foot_oneway = tags.get("oneway:foot")
        forward_allowed = (
            foot_oneway != "-1" and tags.get("foot:forward") not in {"no", "private"}
        )
        backward_allowed = (
            foot_oneway not in {"yes", "1", "true"}
            and tags.get("foot:backward") not in {"no", "private"}
        )
        reverse_tags = tags
        if "incline" in tags:
            reversed_incline = reverse_incline(tags["incline"])
            if reversed_incline != tags["incline"]:
                reverse_tags = normalize_tags({**tags, "incline": reversed_incline})

        for index, (first, second) in enumerate(pairwise(node_ids)):
            if first == second:
                continue
            distance = haversine_m(nodes[first].coordinate, nodes[second].coordinate)
            for reverse, allowed in (
                (False, forward_allowed), (True, backward_allowed)
            ):
                if not allowed:
                    continue
                source, target = (second, first) if reverse else (first, second)
                edges.append(
                    Edge(
                        id=f"{way_id}:{index}:{'r' if reverse else 'f'}",
                        source=source,
                        target=target,
                        way_id=way_id,
                        distance_m=distance,
                        tags=reverse_tags if reverse else tags,
                        crossing_tags=(
                            nodes[target].tags
                            if target in crossing_nodes else empty_tags
                        ),
                        source_tags=nodes[source].tags,
                        target_tags=nodes[target].tags,
                        way_has_crossing_nodes=way_has_crossing_nodes,
                        way_has_kerb_nodes=way_has_kerb_nodes,
                    )
                )
    return InMemoryGraph(nodes.values(), edges)
