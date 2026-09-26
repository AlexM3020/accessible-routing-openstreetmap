"""Scientific figures from supplied graph geometry and profile diagnostics.

No data are fetched and no scores are recomputed. ``score`` includes transition
events; ``way_score`` is not substituted for it. Geometry consists of the stored
node-to-node segments: missing intermediate shapes are not inferred. Both routes
remain unshifted; background geometry is clipped to their metric corridor union.
Coincident background score directions use bounded offsets before re-clipping.
Marker filtering, cropping and coalescing affect drawings, not the full report.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, fsum, hypot, isfinite, radians, sin
from numbers import Real
from pathlib import Path
from textwrap import shorten, wrap

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely import STRtree
from shapely import points as spatial_points
from shapely.geometry import LineString
from shapely.geometry import Point as SpatialPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .graph import InMemoryGraph
from .models import Edge, Route
from .outputs import OBSTACLE_LABELS

Point = tuple[float, float]
Segment = tuple[Point, Point]
Extent = tuple[float, float, float, float]

_BLUE = "#1464b4"
_ORANGE = "#df7c16"
_NAVY = "#182b49"
_RED = "#c52b2f"
_GREY = "#8c8c8c"
_GRAPH_GREY = "#e7e9ec"
_OFFSET_M = 1.0
_COALESCE_M = 12.0
_MAX_LABELS = 12
_ROUTE_SIZE = (16, 10)
_ROUTE_RECT = (.065, .205, .54, .685)
_ROUTE_RATIO = _ROUTE_RECT[2] * _ROUTE_SIZE[0] / (_ROUTE_RECT[3] * _ROUTE_SIZE[1])
_MARKERS = {
    "stairs": "^", "ramp": ">", "barrier": "X", "kerb": "s", "crossing": "P",
    "incline": "D", "wheelchair_restriction": "v", "surface": "h", "smoothness": "p",
    "width": "s", "lighting": "o", "bicycles": "d",
}


@dataclass(frozen=True)
class _Score:
    blocked: bool
    value: float | None


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _project(transformer: Transformer, longitude: float, latitude: float) -> Point:
    try:
        x, y = transformer.transform(longitude, latitude, errcheck=True)
    except ProjError as exc:
        raise ValueError("Coordinates cannot be projected into the local metre CRS") from exc
    if not isfinite(x) or not isfinite(y):
        raise ValueError("Projected coordinates must be finite")
    return float(x), float(y)


def _project_graph(graph: InMemoryGraph, centre_nodes: Sequence[str] | None = None
                   ) -> tuple[dict[str, Point], Transformer]:
    if not graph.nodes:
        raise ValueError("A local projection requires at least one graph node")
    coordinates = [graph.nodes[key].coordinate for key in sorted(set(centre_nodes) if centre_nodes is not None else graph.nodes)]
    # A spherical mean keeps local extracts crossing the antimeridian local.
    latitudes = [radians(point.latitude) for point in coordinates]
    longitudes = [radians(point.longitude) for point in coordinates]
    x = fsum(cos(lat) * cos(lon) for lat, lon in zip(latitudes, longitudes))
    y = fsum(cos(lat) * sin(lon) for lat, lon in zip(latitudes, longitudes))
    z = fsum(sin(lat) for lat in latitudes)
    if hypot(hypot(x, y), z) <= len(coordinates) * 1e-12:
        raise ValueError("Graph coordinates do not define a local projection centre")
    latitude = degrees(atan2(z, hypot(x, y)))
    longitude = degrees(atan2(y, x))
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude:.15g} +lon_0={longitude:.15g} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    positions = {
        key: _project(transformer, graph.nodes[key].coordinate.longitude,
                      graph.nodes[key].coordinate.latitude)
        for key in sorted(graph.nodes)
    }
    return positions, transformer


def projected_positions(graph: InMemoryGraph) -> dict[str, Point]:
    """Return fresh node positions in local WGS84 azimuthal equidistant metres.

    The origin is the spherical mean of all graph nodes, independent of their
    insertion order. Coordinates are always passed to pyproj as longitude,
    latitude. These positions are never shifted for plotting convenience.
    """
    return _project_graph(graph)[0]


def _true_segments(edges: Sequence[Edge], positions: Mapping[str, Point]) -> dict[str, Segment]:
    return {edge.id: (positions[edge.source], positions[edge.target]) for edge in edges}


def score_segments(
    graph: InMemoryGraph,
    positions: Mapping[str, Point] | None = None,
    *,
    offset_m: float = _OFFSET_M,
) -> dict[str, Segment]:
    """Return directed drawing segments for the *graph score panel only*.

    Exactly coincident endpoint pairs, including reverse edges and co-located
    node IDs, are grouped geometrically. Lexical edge IDs determine symmetric
    perpendicular offsets bounded by ``offset_m``. The normal uses canonical
    endpoint order, so reverse directions cannot cancel their offsets. Unique
    segments remain exact. Coincident zero-length edges use a northing offset.
    No supplied coordinates or graph records are changed.
    """
    distance = _number(offset_m, "offset_m")
    if distance < 0:
        raise ValueError("offset_m must be nonnegative")
    positions = projected_positions(graph) if positions is None else positions
    segments = _true_segments(graph.edges, positions)
    groups: dict[Segment, list[str]] = defaultdict(list)
    for edge_id, (start, end) in segments.items():
        groups[(min(start, end), max(start, end))].append(edge_id)
    for (start, end), edge_ids in groups.items():
        if len(edge_ids) < 2:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        nx, ny = (-dy / length, dx / length) if length else (0.0, 1.0)
        for index, edge_id in enumerate(sorted(edge_ids)):
            shift = distance * (2 * index / (len(edge_ids) - 1) - 1)
            a, b = segments[edge_id]
            segments[edge_id] = (
                (a[0] + nx * shift, a[1] + ny * shift),
                (b[0] + nx * shift, b[1] + ny * shift),
            )
    return segments


def _validated_scores(graph: InMemoryGraph, diagnostics: Mapping[str, dict]) -> dict[str, _Score]:
    scores = {}
    for edge in graph.edges:
        if edge.id not in diagnostics:
            raise ValueError(f"Missing diagnostics for edge {edge.id!r}")
        record = diagnostics[edge.id]
        if not isinstance(record, Mapping) or "blocked" not in record or "score" not in record:
            raise ValueError(f"Diagnostics for edge {edge.id!r} require blocked and score")
        if not isinstance(record["blocked"], bool):
            raise ValueError(f"blocked for edge {edge.id!r} must be a boolean")
        # Validate even blocked edges; invalid values must never disappear behind a style.
        for field in ("score", "way_score", "coverage"):
            value = record.get(field)
            if value is not None:
                value = _number(value, f"{field} for edge {edge.id!r}")
                if not 0 <= value <= 100:
                    raise ValueError(f"{field} for edge {edge.id!r} must be in [0, 100]")
        scores[edge.id] = _Score(
            record["blocked"], None if record["score"] is None else float(record["score"])
        )
    return scores


def _validate_route(graph: InMemoryGraph, route: Route) -> None:
    if not route.nodes or len(route.nodes) != len(route.edges) + 1:
        raise ValueError("Route must contain one more node than edges, including a start node")
    if any(node_id not in graph.nodes for node_id in route.nodes):
        raise ValueError("Route references a node outside the graph")
    edge_map = {edge.id: edge for edge in graph.edges}
    for index, edge in enumerate(route.edges):
        if edge_map.get(edge.id) != edge:
            raise ValueError(f"Route edge {edge.id!r} does not match the graph")
        if (edge.source, edge.target) != (route.nodes[index], route.nodes[index + 1]):
            raise ValueError(f"Route edge {edge.id!r} disagrees with the directed node sequence")


def _extent(points: Sequence[Point], ratio: float) -> Extent:
    xs, ys = zip(*points)
    padding = max(max(max(xs) - min(xs), max(ys) - min(ys)) * .10, 12.0)
    extent_x, extent_y = max(xs) - min(xs) + 2 * padding, max(ys) - min(ys) + 2 * padding
    if extent_x / extent_y < ratio:
        extent_x = extent_y * ratio
    else:
        extent_y = extent_x / ratio
    center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return (center_x - extent_x / 2, center_x + extent_x / 2,
            center_y - extent_y / 2, center_y + extent_y / 2)


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _path_geometries(paths: Sequence[Sequence[Point]]) -> list[BaseGeometry]:
    """Separate paths, never a fictitious connection between route endpoints."""
    return [LineString(path) if len(path) > 1 and len(set(path)) > 1 else SpatialPoint(path[0])
            for path in paths if path]


def _indexed_path_distances(candidates: Sequence[Point], paths: Sequence[Sequence[Point]]) -> list[float]:
    """Batched exact point-to-segment distances, not nearest vertex distances."""
    if not candidates:
        return []
    segments: list[BaseGeometry] = []
    for path in paths:
        if len(path) == 1:
            segments.append(SpatialPoint(path[0]))
        segments.extend(LineString((a, b)) if a != b else SpatialPoint(a)
                        for a, b in zip(path, path[1:]))
    tree = STRtree(segments)
    indices, distances = tree.query_nearest(spatial_points(candidates), return_distance=True, all_matches=False)
    result = [float("inf")] * len(candidates)
    for index, distance in zip(indices[0], distances):
        result[int(index)] = float(distance)
    return result


@dataclass(frozen=True)
class _Corridor:
    geometry: BaseGeometry
    segments: Mapping[str, list[Segment]]
    extent: Extent
    radius_m: float


def _geometry_parts(shape: BaseGeometry) -> list[Segment]:
    if shape.is_empty:
        return []
    if shape.geom_type == "LineString":
        coordinates = list(shape.coords)
        return [(tuple(a), tuple(b)) for a, b in zip(coordinates, coordinates[1:])]
    if shape.geom_type == "Point":
        point = tuple(shape.coords[0])
        return [(point, point)]
    return [segment for part in shape.geoms for segment in _geometry_parts(part)]


def _corridor_scores(graph: InMemoryGraph, positions: Mapping[str, Point],
                     corridor: _Corridor) -> dict[str, list[Segment]]:
    """Offset only original corridor members, then clip to the original buffer.

    Unchanged segments reuse their intersection. No nearest-path scans, new
    corridor selection, or graph mutations are needed for the score layer.
    """
    if not corridor.segments:
        return {}
    shifted = score_segments(graph, positions)
    result = {}
    for edge in graph.edges:
        if edge.id not in corridor.segments:
            continue
        segment = shifted[edge.id]
        if segment == (positions[edge.source], positions[edge.target]):
            result[edge.id] = corridor.segments[edge.id]
        else:
            shape = LineString(segment) if segment[0] != segment[1] else SpatialPoint(segment[0])
            result[edge.id] = _geometry_parts(shape.intersection(corridor.geometry))
    return result


def _corridor(graph: InMemoryGraph, positions: Mapping[str, Point],
              paths: Sequence[Sequence[Point]], radius_m: float, ratio: float = _ROUTE_RATIO) -> _Corridor:
    """Clip intersecting graph edges in AEQD metres; retain each edge's identity.

    GEOS buffers approximate round arcs (64 segments per quadrant); the spatial
    prefilter uses exact dwithin distances, including crossings whose endpoints
    are outside the corridor. Route paths are rendered separately, in full.
    ``ratio`` is retained for caller compatibility, not used to expand bounds:
    each original buffer dimension gets small independent padding. Equal metric
    aspect is enforced by the axes box, never by adding empty map territory.
    """
    routes = _path_geometries(paths)
    geometry = unary_union([path.buffer(radius_m, quad_segs=64) for path in routes])
    xmin, ymin, xmax, ymax = geometry.bounds
    padx, pady = max((xmax - xmin) * .04, 2.), max((ymax - ymin) * .04, 2.)
    extent = (xmin - padx, xmax + padx, ymin - pady, ymax + pady)
    edges = list(graph.edges)
    raw = _true_segments(edges, positions)
    lines = [LineString(raw[edge.id]) if raw[edge.id][0] != raw[edge.id][1]
             else SpatialPoint(raw[edge.id][0]) for edge in edges]
    tree = STRtree(lines)
    indices = tree.query(routes, predicate="dwithin", distance=radius_m)
    clipped: dict[str, list[Segment]] = {}

    for index in sorted(set(indices[1])):
        edge = edges[index]
        result = _geometry_parts(lines[index].intersection(geometry))
        if result:
            clipped[edge.id] = result
    return _Corridor(geometry, clipped, extent, radius_m)


def _comparison_markers(presentation: dict | None, standard_route: Route | None) -> dict | None:
    """Fresh display-only union; complete comparison encounter counts stay intact."""
    if presentation is None or standard_route is None:
        return presentation
    comparison = presentation.get("comparison", {})
    standard = comparison.get("standard", {}).get("obstacles", [])
    options = presentation.get("options", {})
    kinds = options.get("obstacle_kinds", presentation.get("filter_summary", {}).get("selected_kinds"))
    standard_rows = [dict(row, route_affiliation="standard", on_route=True) for row in standard if (
        row["kind"] in kinds if kinds is not None else
        row["kind"] in {"stairs", "barrier", "wheelchair_restriction"}
        or bool(row.get("preference_labels")) or row["status"] in {"blocked", "conflict"}
    )]
    rows = [dict(row, route_affiliation="selected" if row["on_route"] else "context")
            for row in presentation.get("map_obstacles", [])]
    # Keep candidate accounting intact; sharing happens after AUTO consolidation.
    rows.extend(standard_rows)
    summary = dict(presentation.get("filter_summary", {}))
    omitted = summary.get("omitted_by_filter", 0) + len(standard) - len(standard_rows)
    summary.update(selected_candidates=len(rows), total_candidates=len(rows) + omitted, omitted_by_filter=omitted)
    return dict(presentation, map_obstacles=rows, filter_summary=summary)


def _marker_candidates(pois: list[dict], presentation: dict | None,
                       transformer: Transformer) -> tuple[list[dict], dict[str, int]]:
    if presentation is not None and not isinstance(presentation, Mapping):
        raise ValueError("presentation must be a mapping or None")
    legacy = presentation is None
    view = presentation or {}
    options, summary = view.get("options", {}), view.get("filter_summary", {})
    if not isinstance(options, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("presentation options and filter_summary must be mappings")
    limit = _count(options.get("max_markers", summary.get("max_markers", 12)), "max_markers")
    if limit > 100:
        raise ValueError("max_markers must be between 0 and 100")
    raw = pois if legacy else view.get("map_obstacles", [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("POIs / map_obstacles must be a sequence of records")
    flag = "blocked" if legacy else "status"
    required = {"id", "kind", "latitude", "longitude", "on_route", flag, "description"}
    result = []
    for record in raw:
        if not isinstance(record, Mapping) or not required.issubset(record):
            raise ValueError(f"Each POI / map obstacle requires id, kind, coordinates, on_route, {flag} and description")
        identifier = record["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError("POI id must be a nonempty string or integer")
        if not isinstance(record["kind"], str) or record["kind"] not in _MARKERS:
            raise ValueError(f"Unsupported POI kind for {identifier!r}")
        if not isinstance(record["on_route"], bool):
            raise ValueError(f"POI on_route for {identifier!r} must be a boolean")
        if not isinstance(record["description"], str):
            raise ValueError(f"POI description for {identifier!r} must be a string")
        latitude = _number(record["latitude"], f"POI latitude for {identifier!r}")
        longitude = _number(record["longitude"], f"POI longitude for {identifier!r}")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"POI coordinates for {identifier!r} are outside WGS84 bounds")
        position = _project(transformer, longitude, latitude)
        if legacy:
            if record["blocked"] is not None and not isinstance(record["blocked"], bool):
                raise ValueError(f"POI blocked for {identifier!r} must be a boolean or None")
            # A mapped stair is a known feature even if its passability is unknown.
            # Legacy barrier records provide no criticality beyond the blocked flag.
            if record["kind"] != "stairs" and not (record["kind"] == "barrier" and record["blocked"] is True):
                continue
            status = "blocked" if record["blocked"] else "encountered" if record["on_route"] else "context"
        else:
            status = record["status"]
            if not isinstance(status, str) or status not in {"conflict", "blocked", "encountered", "context", "unknown"}:
                raise ValueError(f"Invalid map obstacle status for {identifier!r}")
        row = dict(record, id=str(identifier), position=position, status=status)
        for key, old_key in (("way_id", "osm_way_id"), ("node_id", "osm_node_id")):
            owner = record.get(key, record.get(old_key))
            if owner is not None and (isinstance(owner, bool) or not isinstance(owner, (str, int))):
                raise ValueError(f"POI {key} must be a string, integer or None")
            row[key] = None if owner is None else str(owner)
        row["first_segment"] = record.get("first_segment")
        if row["first_segment"] is not None:
            _count(row["first_segment"], "first_segment")
        for key in ("edge_ids", "preference_labels"):
            values = record.get(key, [])
            if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"POI {key} must be a sequence of strings")
            row[key] = list(values)
        result.append(row)
    selected = len(result)
    total = len(raw) if legacy else _count(summary.get("total_candidates", selected), "total_candidates")
    reported_selected = _count(summary.get("selected_candidates", selected), "selected_candidates")
    if total < selected or reported_selected != selected:
        raise ValueError("filter_summary counts disagree with map_obstacles")
    omitted = total - selected
    if _count(summary.get("omitted_by_filter", omitted), "omitted_by_filter") != omitted:
        raise ValueError("filter_summary omitted_by_filter disagrees with candidate counts")
    return result, {"total_candidates": total, "selected_candidates": selected,
                    "omitted_by_filter": omitted, "max_markers": limit}


def _distance_to_path(point: Point, points: Sequence[Point]) -> float:
    distances = [hypot(point[0] - points[0][0], point[1] - points[0][1])]
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        fraction = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2)) if length2 else 0.
        distances.append(hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy))
    return min(distances)


def _select_obstacles(pois: list[dict], presentation: dict | None, transformer: Transformer,
                      points: Sequence[Point], *, paths: Sequence[Sequence[Point]] | None = None,
                      corridor: _Corridor | None = None) -> tuple[list[dict], dict[str, int], Extent]:
    candidates, counts = _marker_candidates(pois, presentation, transformer)
    paths = paths if paths is not None else [points]
    extent = corridor.extent if corridor is not None else _extent(points, _ROUTE_RATIO)
    xmin, xmax, ymin, ymax = extent
    visible = [row for row in candidates if xmin <= row["position"][0] <= xmax
               and ymin <= row["position"][1] <= ymax]
    if corridor is not None:
        visible = [row for row in visible if corridor.geometry.covers(SpatialPoint(row["position"]))]
    counts["cropped"] = len(candidates) - len(visible)
    counts["unknown_suppressed"] = sum(row["status"] == "unknown" for row in visible)
    known = [row for row in visible if row["status"] != "unknown"]
    # AUTO only: consolidate a wheelchair restriction with its physical stairs /
    # barrier encounter. Exact owners, directed edges, visit and affiliation are
    # required; proximity alone is never evidence of a shared observation.
    options = (presentation or {}).get("options", {})
    summary = (presentation or {}).get("filter_summary", {})
    automatic = presentation is not None and options.get("obstacle_kinds", summary.get("selected_kinds")) is None
    if automatic:
        def physical_identity(row):
            return (row["node_id"], row["way_id"], tuple(row["edge_ids"]), row["first_segment"],
                    row["position"], row["on_route"], row.get("route_affiliation"))
        physical = {physical_identity(row): row for row in known
                    if row["kind"] in {"stairs", "barrier"}
                    and (row["node_id"] is not None or row["way_id"] is not None)}
        retained = []
        for row in known:
            primary = physical.get(physical_identity(row)) if row["kind"] == "wheelchair_restriction" else None
            if primary is None:
                retained.append(row)
                continue
            primary["preference_labels"] = list(dict.fromkeys(primary["preference_labels"] + row["preference_labels"]))
            primary["description"] += "; " + row["description"]
            primary["display_label"] = OBSTACLE_LABELS[primary["kind"]] + " + wheelchair restriction"
            primary.setdefault("combined_ids", []).append(row["id"])
            rank = {"context": 0, "encountered": 1, "conflict": 2, "blocked": 3}
            primary["status"] = max((primary["status"], row["status"]), key=rank.__getitem__)
        known = retained
    def shared_identity(row):
        return (row["kind"], row["node_id"], row["way_id"], tuple(row["edge_ids"]),
                row["position"], row["status"], row["description"], tuple(row["preference_labels"]))
    selected_rows = {shared_identity(row): row for row in known
                     if row.get("route_affiliation") == "selected" and row["edge_ids"]
                     and (row["node_id"] is not None or row["way_id"] is not None)}
    retained = []
    for row in known:
        primary = selected_rows.get(shared_identity(row)) if row.get("route_affiliation") == "standard" else None
        if primary is None:
            retained.append(row)
        else:
            primary["route_affiliation"] = "both"
            primary.setdefault("combined_ids", []).extend([row["id"], *row.get("combined_ids", [])])
            if "display_label" in row:
                primary["display_label"] = row["display_label"]
    known = retained
    # No marker budget means no nearest-distance work, including tree creation.
    distances = (_indexed_path_distances([row["position"] for row in known], paths)
                 if counts["max_markers"] else [0.] * len(known))
    by_identity = {id(row): distance for row, distance in zip(known, distances)}

    def priority(row):
        rank = (0 if row["on_route"] and row["status"] == "conflict" else
                1 if row["status"] == "blocked" or row["kind"] in {"stairs", "barrier", "wheelchair_restriction"} else
                2 if row["on_route"] else 3)
        segment = row["first_segment"] if row["on_route"] and row["first_segment"] is not None else float("inf")
        return (rank, not row["on_route"], segment, by_identity[id(row)],
            row["kind"], row["id"], row["position"], row["node_id"] or "", row["way_id"] or "",
            tuple(row["edge_ids"]), tuple(row["preference_labels"]), row["description"])

    groups: list[dict] = []
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in sorted(known, key=priority):
        owner = ("node", row["node_id"]) if row["node_id"] is not None else (
            ("way", row["way_id"]) if row["way_id"] is not None else ("point", row["position"])
        )
        key = (row["kind"], row["on_route"], row["status"], row.get("route_affiliation"), owner,
               tuple(row["edge_ids"]) if row.get("route_affiliation") else None)
        match = next((group for group in buckets[key] if hypot(
            group["position"][0] - row["position"][0], group["position"][1] - row["position"][1]
        ) <= _COALESCE_M), None)
        if match is None:
            members = [row["id"], *row.get("combined_ids", [])]
            match = dict(row, member_ids=members, candidate_count=len(members))
            groups.append(match)
            buckets[key].append(match)
        else:
            members = [row["id"], *row.get("combined_ids", [])]
            match["candidate_count"] += len(members)
            match["member_ids"].extend(members)
            for field in ("edge_ids", "preference_labels"):
                match[field] = list(dict.fromkeys([*match[field], *row[field]]))
    displayed = [dict(row, marker_label=str(index + 1) if index < _MAX_LABELS else None)
                 for index, row in enumerate(groups[:counts["max_markers"]])]
    counts.update(displayed=len(displayed), labelled=min(len(displayed), _MAX_LABELS),
                  coalesced=len(visible) - counts["unknown_suppressed"] - len(groups),
                  omitted_by_limit=len(groups) - len(displayed),
                  suppressed=len(visible) - len(displayed))
    return displayed, counts, extent


def select_map_obstacles(graph: InMemoryGraph, route: Route, pois: list[dict], *,
                         presentation: dict | None = None, standard_route: Route | None = None,
                         corridor_radius_m: float = 150.0) -> tuple[list[dict], dict[str, int], Extent]:
    """Return fresh display rows, accounting counts and (xmin, xmax, ymin, ymax).

    Selected map_obstacles are already type-filtered; never supplement them from
    POIs, conflicts or unknowns. When a standard trace is supplied, its comparison
    encounters use the same display filters and shared cap. Legacy calls select
    stairs / blocked barriers. Both route buffers define the metric viewport.
    Within the route viewport, prioritize on-route conflicts, important features,
    other encounters, then proximity. Merge only nearby same-type/same-owner rows
    with the same status and route membership, retaining a real representative
    point, member IDs and counts. Without an owner, only exact coincidences merge.
    AUTO combines wheelchair restrictions with exact same-owner, same-directed-
    edge stairs/barrier encounters; explicit type filters retain separate symbols.
    Shared-route symbols require matching directed identities and observations.
    At most 12 markers are numbered, even for an explicitly larger marker limit.
    selected_candidates = cropped + displayed + suppressed; suppressed includes
    unknown observations, coalesced rows and distinct markers beyond the limit.
    """
    _validate_route(graph, route)
    radius = _validated_radius(corridor_radius_m)
    if standard_route is not None:
        _validate_route(graph, standard_route)
    routes = [route] + ([standard_route] if standard_route is not None else [])
    positions, transformer = _project_graph(graph, [node for path in routes for node in path.nodes])
    paths = [[positions[key] for key in path.nodes] for path in routes]
    corridor = _corridor(graph, positions, paths, radius)
    return _select_obstacles(pois, _comparison_markers(presentation, standard_route), transformer,
                             paths[0], paths=paths, corridor=corridor)


def _map_axes(ax: Axes, points: Sequence[Point], title: str, *, extent: Extent | None = None) -> None:
    position = ax.get_position(original=True)
    fig_width, fig_height = ax.figure.get_size_inches()
    extent = extent or _extent(points, position.width * fig_width / (position.height * fig_height))
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=9, width=.6)
    ax.set_axisbelow(True)
    ax.grid(color="#eef0f2", linewidth=.5)
    for spine in ax.spines.values():
        spine.set_color("#c6cbd1")
        spine.set_linewidth(.6)


def _directions(ax: Axes, segments: Sequence[Segment], colours: Sequence[str], gid: str) -> None:
    moving = [(a, b, colour) for (a, b), colour in zip(segments, colours) if a != b]
    if not moving:
        return
    # A deterministic sparse sample for either route, never the background.
    moving = moving[::max(1, (len(moving) + 5) // 6)][:6]
    fraction = .68 if gid == "standard-directions" else .4
    arrows = ax.quiver(
        [a[0] + fraction * (b[0] - a[0]) for a, b, _ in moving],
        [a[1] + fraction * (b[1] - a[1]) for a, b, _ in moving],
        [0.22 * (b[0] - a[0]) for a, b, _ in moving],
        [0.22 * (b[1] - a[1]) for a, b, _ in moving],
        color=[colour for _, _, colour in moving], angles="xy", scale_units="xy", scale=1,
        units="inches", width=.024, headwidth=5, headlength=6, minlength=0,
        edgecolors="white", linewidths=.4, zorder=20,
    )
    arrows.set_gid(gid)


def _lines(ax: Axes, segments: Sequence[Segment], colour: str, gid: str, *,
           dashed: bool = False, width: float = 1.5, zorder: int = 2) -> LineCollection:
    collection = LineCollection(segments, colors=colour, linewidths=width,
                                linestyles="dashed" if dashed else "solid", zorder=zorder)
    collection.set_gid(gid)
    ax.add_collection(collection)
    stationary = [start for start, end in segments if start == end]
    if stationary:
        dots = ax.scatter(*zip(*stationary), c=colour, s=22, marker="x" if dashed else "o", zorder=zorder)
        dots.set_gid(f"{gid}-points")
    return collection


def _endpoints(ax: Axes, route: Route, positions: Mapping[str, Point]) -> None:
    start, end = positions[route.nodes[0]], positions[route.nodes[-1]]
    ax.scatter(*start, s=110, marker="o", facecolors="none", edgecolors="#195f45", linewidths=2, zorder=17)
    ax.scatter(*end, s=85, marker="s", facecolors="none", edgecolors=_NAVY, linewidths=2, zorder=17)
    labels = [("Start / end", start, (6, 8))] if start == end else [
        ("Start", start, (6, 8)), ("End", end, (6, -14))
    ]
    for text, point, offset in labels:
        label = ax.annotate(text, point, xytext=offset, textcoords="offset points", fontsize=10, zorder=30,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
        label.set_gid("route-endpoint-label")


def _route_line(ax: Axes, route: Route, positions: Mapping[str, Point], *,
                colour: str = _BLUE, contrast: bool = False, width: float = 4.6) -> None:
    points = [positions[node_id] for node_id in route.nodes]
    halo, = ax.plot(*zip(*points), color="white", linewidth=width + 5, zorder=10,
                    solid_capstyle="round", solid_joinstyle="round")
    halo.set_gid("route-halo")
    if contrast:
        outline, = ax.plot(*zip(*points), color=_NAVY, linewidth=width + 2.5, zorder=11,
                           solid_capstyle="round", solid_joinstyle="round")
        outline.set_gid("route-outline")
    line, = ax.plot(*zip(*points), color=colour, linewidth=width, zorder=12,
                    linestyle=(0, (5, 2)) if contrast else "solid",
                    solid_capstyle="round", solid_joinstyle="round")
    line.set_gid("selected-route")


def _standard_line(ax: Axes, route: Route, positions: Mapping[str, Point]) -> None:
    points = [positions[key] for key in route.nodes]
    # A dashed white halo leaves the selected solid blue visible through gaps,
    # including on shared sections. Neither path is shifted off its geometry.
    for colour, width, zorder, gid in (("white", 5.8, 14, "standard-halo"),
                                       (_ORANGE, 3.2, 15, "standard-route")):
        line, = ax.plot(*zip(*points), color=colour, linewidth=width, linestyle=(0, (5, 4)),
                        zorder=zorder, dash_capstyle="butt")
        # Use an identical physical dash period for both widths (mpl scales dashes).
        line.set_dashes((16 / width, 12 / width))
        line.set_gid(gid)
    segments = [(positions[edge.source], positions[edge.target]) for edge in route.edges]
    _directions(ax, segments, [_ORANGE] * len(segments), "standard-directions")


def _finish_route(ax: Axes, route: Route, positions: Mapping[str, Point], colour: str) -> None:
    _endpoints(ax, route, positions)
    segments = [(positions[edge.source], positions[edge.target]) for edge in route.edges]
    _directions(ax, segments, [colour] * len(segments), "route-directions")


def _score_lines(ax: Axes, edges: Sequence[Edge], segments: Mapping[str, Segment],
                 scores: Mapping[str, _Score], scale: ScalarMappable, prefix: str, *,
                 width: float = 2.2, zorder: int = 2,
                 clipped: Mapping[str, list[Segment]] | None = None) -> None:
    def geometry_for(edge):
        return clipped.get(edge.id, []) if clipped is not None else [segments[edge.id]]

    active = [edge for edge in edges if not scores[edge.id].blocked and scores[edge.id].value is not None]
    unscored = [part for edge in edges if not scores[edge.id].blocked and scores[edge.id].value is None
                for part in geometry_for(edge)]
    geometry = [part for edge in active for part in geometry_for(edge)]
    if geometry:
        values = [scores[edge.id].value for edge in active for _ in geometry_for(edge)]
        collection = LineCollection(geometry, cmap=scale.cmap, norm=scale.norm, linewidths=width, zorder=zorder)
        collection.set_array(values)
        collection.set_gid(f"{prefix}-score")
        ax.add_collection(collection)
        for (start, end), value in zip(geometry, values):
            if start == end:
                ax.scatter(*start, color=scale.to_rgba(value), s=22, zorder=zorder)
    if unscored:
        _lines(ax, unscored, _GREY, f"{prefix}-unscored", width=width, zorder=zorder)
    blocked = [part for edge in edges if scores[edge.id].blocked for part in geometry_for(edge)]
    if blocked:
        _lines(ax, blocked, _RED, f"{prefix}-blocked", dashed=True, width=width, zorder=zorder + 1)
    if not edges:
        ax.text(0.5, 0.92, "No edges to score", transform=ax.transAxes, ha="center", fontsize=9)


def _marker_colour(row: dict) -> str:
    return _RED if row["status"] in {"conflict", "blocked"} else "#a26812" if row["on_route"] else "#637184"


def _obstacles(ax: Axes, rows: Sequence[dict], endpoints: Sequence[Point]) -> None:
    for row in rows:
        # The hollow symbol is wider than the route halo, leaving its outline visible.
        symbol = ax.scatter(*row["position"], marker=_MARKERS[row["kind"]], s=196,
                            facecolors="none", edgecolors=_marker_colour(row), linewidths=1.6, zorder=5)
        symbol.set_gid(f"obstacle-{row['kind']}-{row['status']}")
    # Test label rectangles in display coordinates. Text stays >=9pt and never
    # becomes a raw ID/description; the small numbered key carries the names.
    ax.apply_aspect()
    unit = ax.figure.dpi / 72
    occupied = [Bbox.from_bounds(x - 8 * unit, y - 8 * unit, 16 * unit, 16 * unit)
                for x, y in ax.transData.transform([*endpoints, *(row["position"] for row in rows)])]
    bounds = ax.get_window_extent()
    offsets = ((10, 12), (10, -18), (-10, 12), (-10, -18), (22, 0), (-22, 0),
               (0, 26), (0, -28), (30, 26), (-30, 26), (30, -28), (-30, -28))
    for row in rows[:_MAX_LABELS]:
        x, y = ax.transData.transform(row["position"])
        width, height = (6 * len(row["marker_label"]) + 6) * unit, 15 * unit
        choices = []
        for dx, dy in offsets:
            left = x + dx * unit - (width if dx < 0 else width / 2 if dx == 0 else 0)
            box = Bbox.from_bounds(left, y + dy * unit - height / 2, width, height)
            outside = not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))
            choices.append((100 * outside + sum(box.overlaps(other) for other in occupied), (dx, dy), box))
        _, offset, box = min(choices, key=lambda choice: choice[0])
        occupied.append(box)
        label = ax.annotate(row["marker_label"], row["position"], xytext=offset, textcoords="offset points",
                            ha="left" if offset[0] > 0 else "right" if offset[0] < 0 else "center",
                            va="center", fontsize=9, color=_NAVY, zorder=30,
                            bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": "#c6cbd1"},
                            arrowprops={"arrowstyle": "-", "color": "#8c8c8c", "linewidth": .5})
        label.set_gid("obstacle-label")


def _marker_key(ax: Axes, rows: Sequence[dict]) -> None:
    ax.set_axis_off()
    comparison = ax.get_position().width > .2
    height = ax.get_position().height * ax.figure.get_size_inches()[1]
    heading = ax.text(0, .99, "Numbered map key", fontsize=10, weight="bold", va="top")
    heading.set_gid("key-heading")
    if comparison:
        affiliation = ax.text(0, 1 - .20 / height, "S selected · N standard · S+N shared", fontsize=9, va="top")
        affiliation.set_gid("key-affiliation")
    for index, row in enumerate(rows[:_MAX_LABELS]):
        count = f" ×{row['candidate_count']}" if row["candidate_count"] > 1 else ""
        affiliation = {"selected": "S", "standard": "N", "both": "S+N", "context": "context"}.get(row.get("route_affiliation"))
        suffix = f" [{affiliation}]" if affiliation else ""
        text = f"{row['marker_label']}  {row.get('display_label', OBSTACLE_LABELS[row['kind']])}{count}{suffix}"
        x, y = ((index % 2) * .51, 1 - (.40 + (index // 2) * .26) / height) if comparison else (0, .88 - index * .071)
        if comparison:
            text = "\n".join(wrap(text, width=32))
        label = ax.text(x, y, text, va="top", fontsize=9, color=_marker_colour(row), linespacing=1.05)
        label.set_gid("obstacle-key")
    if not rows:
        ax.text(0, .86, "No markers in view.", fontsize=9)


def _short(value: object, width: int) -> str:
    return shorten(" ".join(str(value).split()), width=width, placeholder="…")


def _metric(value: object, suffix: str = "", *, precision: int = 1) -> str:
    return "Not supplied" if value is None else f"{_number(value, 'presentation metric'):.{precision}f}{suffix}"


def _summary_panel(ax: Axes, route: Route, presentation: dict | None) -> None:
    ax.set_axis_off()
    view = presentation or {}
    metrics, validation = view.get("metrics", {}), view.get("validation", {})
    y, step = .99, 12 / (ax.get_position().height * ax.figure.get_size_inches()[1] * 72)

    def put(text, *, bold=False, colour=_NAVY, size=9.5):
        nonlocal y
        ax.text(0, y, text, transform=ax.transAxes, va="top", fontsize=size,
                weight="bold" if bold else "normal", color=colour, parse_math=False, linespacing=1.2)
        y -= step * (text.count("\n") + 1) + (.004 if bold else 0)

    put("Profile: " + _short(view.get("profile_name", "not supplied"), 52), bold=True, size=12)
    put(f"Distance {_metric(metrics.get('distance_m', route.distance_m), ' m')}   ·   "
        f"Cost {_metric(metrics.get('cost_m', route.cost), ' m')}")
    score = "No active score" if "score_percent" in metrics and metrics["score_percent"] is None else _metric(metrics.get("score_percent"), "%")
    coverage = "No active score" if "coverage_percent" in metrics and metrics["coverage_percent"] is None else _metric(metrics.get("coverage_percent"), "%")
    put(f"Preference match {score}   ·   Coverage {coverage}")
    y -= .01
    put("Selected preferences · weights", bold=True)
    preferences = view.get("selected_preferences", [])
    for index, item in enumerate(preferences):
        text = f"{_short(item.get('label', item.get('name', 'Preference')), 27)}  {_number(item['weight'], 'weight'):g}"
        ax.text((index % 2) * .52, y - (index // 2) * step, text, transform=ax.transAxes,
                va="top", fontsize=9, parse_math=False)
    if not preferences:
        put("None selected" if presentation is not None else "Not supplied")
    else:
        y -= ((len(preferences) + 1) // 2) * step
    y -= .01
    requirements = view.get("requirements", [])
    wheelchair = next((item for item in requirements if item.lower().startswith("wheelchair required:")), "")
    flag = ("YES · nonnegotiable model rule" if wheelchair.lower().startswith("wheelchair required: true") else
            "NO · wheelchair weight is a soft preference" if wheelchair.lower().startswith("wheelchair required: false") else
            "not supplied")
    put("Wheelchair required: " + flag, bold=True)
    settings = [item.split(". ", 1)[0] for item in requirements
                if item.startswith(("Ramps allowed:", "Handrail preferred:", "Minimum width:"))]
    if settings:
        put("\n".join(wrap(" · ".join(settings), width=77)))
    conflicts, unknowns = view.get("conflicts", []), view.get("unknowns", [])
    put(f"Known conflicts: {len(conflicts)}   ·   Unknown observations: {len(unknowns)}" if presentation is not None else
        "Conflicts / unknown observations: not supplied", bold=True)
    if conflicts:
        put(f"Top conflicts ({min(3, len(conflicts))} of {len(conflicts)})", bold=True)
        for row in conflicts[:3]:
            label = row.get("label", OBSTACLE_LABELS.get(row.get("kind"), "Observation"))
            reason = row.get("reason", "").split("; model observation:", 1)[0]
            text = reason if reason.lower().startswith(label.lower()) else f"{label}: {reason}"
            put("\n".join(wrap(_short(f"• {text}", 110), width=59,
                               max_lines=2, placeholder="…")))
    put("Complete preferences / encounters: route_summary.md", size=9)
    y -= .01
    put("Supplied model checks", bold=True)
    hard = validation.get("hard_constraint_violations")
    put("Hard-block violations: " + (str(_count(hard, "hard_constraint_violations")) if hard is not None else
                                    "Not compared"), colour=_RED if hard else _NAVY)
    agree, difference = validation.get("astar_dijkstra_agree"), validation.get("optimal_cost_difference_m")
    if isinstance(agree, bool) and difference is not None:
        put(f"A* cost {'=' if agree else '≠'} Dijkstra (tolerance check); |Δ| = "
            f"{_number(difference, 'optimal_cost_difference_m'):.6g} m", colour=_NAVY if agree else _RED)
    else:
        put("A*/Dijkstra: Not compared")
    baseline = validation.get("distance_baseline_m")
    put("Distance baseline: Not compared" if baseline is None else
        f"Distance baseline: {_metric(baseline, ' m')} · overhead {_metric(validation.get('distance_overhead_percent'), '%')}")


def _legend_line(label: str, colour: str, *, dashed: bool = False) -> Line2D:
    return Line2D([], [], color=colour, linewidth=2, linestyle="--" if dashed else "-", label=label)


def _route_key(colour: str, *, dashed: bool = False) -> Line2D:
    return Line2D([], [], color=colour, linewidth=3, linestyle="--" if dashed else "-",
                  marker=">", markersize=7, label="Selected route / travel direction")


def _caption(synthetic: bool) -> str:
    data = ("Synthetic example; not observed OSM data." if synthetic else
            "© OpenStreetMap contributors, ODbL 1.0 — openstreetmap.org/copyright.")
    return data + "  WGS84 AEQD (metres). Model scores are not safety guarantees; unverified snap links excluded."


def _figure(title: str, heading: str, synthetic: bool, size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, facecolor="white")
    FigureCanvasAgg(figure)
    figure.text(.065, .97, heading + " — " + _short(title, 70), fontsize=15, weight="bold", parse_math=False)
    if synthetic:
        badge = figure.text(.975, .935, "SYNTHETIC DATA", ha="right", fontsize=10, weight="bold", color="#6b440b",
                            bbox={"boxstyle": "round,pad=.35", "facecolor": "#fff1cf", "edgecolor": "none"})
        badge.set_gid("synthetic-badge")
    return figure


def _comparison_panel(ax: Axes, presentation: dict | None, route: Route, standard: Route) -> None:
    ax.set_axis_off()
    view = presentation or {}
    comparison = view.get("comparison", {})
    height = ax.get_position().height * ax.figure.get_size_inches()[1]
    y = 0.

    def put(text, *, size=9, bold=False, colour=_NAVY, gid="comparison-profile", advance=.18):
        nonlocal y
        artist = ax.text(0, 1 - y / height, text, fontsize=size, weight="bold" if bold else "normal",
                         color=colour, va="top", linespacing=1.1, parse_math=False)
        artist.set_gid(gid)
        y += advance

    put("Selected vs no-preference standard", size=11, bold=True, advance=.26)
    put("Profile: " + _short(view.get("profile_name", "not supplied"), 55), advance=.22)
    put("Selected preferences · weights", bold=True)
    preferences = view.get("selected_preferences", [])
    for index, item in enumerate(preferences):
        text = f"{item.get('label', item.get('name', 'Preference'))}  {item['weight']:g}"
        artist = ax.text((index % 2) * .51, 1 - (y + index // 2 * .18) / height,
                         text, fontsize=9, va="top", parse_math=False)
        artist.set_gid("comparison-preference")
    y += max(1, (len(preferences) + 1) // 2) * .18
    if not preferences:
        artist = ax.text(0, 1 - (y - .18) / height, "None selected" if presentation is not None else "Not supplied",
                         fontsize=9, va="top")
        artist.set_gid("comparison-preference")
    settings = [item.split(". ", 1)[0] for item in view.get("requirements", [])
                if item.startswith(("Wheelchair required:", "Ramps allowed:", "Handrail preferred:", "Minimum width:"))]
    # Two short lines retain the explicit hard/soft distinction without policy prose.
    for offset in (0, 2):
        if settings[offset:offset + 2]:
            put(" · ".join(settings[offset:offset + 2]))
    y = _comparison_profile_height(view)
    if comparison.get("status") != "available":
        put("Comparison assessment not supplied.\nDistances (m): "
                f"{route.distance_m:.1f} / {standard.distance_m:.1f}\n"
                "No obstacle counts or feasibility inferred.")
        return
    selected, baseline = comparison["selected"], comparison["standard"]
    records = []
    worse = []
    def add(label, key, suffix="", precision=0):
        values = [stats.get(key) for stats in (selected, baseline)]
        delta = values[0] - values[1] if all(value is not None for value in values) else None
        rendered = [("No active score" if key in {"score_percent", "coverage_percent"} and key in stats
                     else "Not supplied") if value is None else _metric(value, suffix, precision=precision)
                    for stats, value in zip((selected, baseline), values)]
        records.append([label, *rendered, "—" if delta is None else f"{delta:+.{precision}f}" + (" pp" if suffix == "%" else "")])
        worse.append(delta is not None and (delta < 0 if suffix == "%" else delta > 0))
    add("Distance (m)", "distance_m", precision=1)
    records.append(["Selected requirements", *["Infeasible" if stats["hard_block_violations"] else "Feasible"
                                               for stats in (selected, baseline)], "—"])
    worse.append(False)
    add("Profile match", "score_percent", "%", 1)
    for column, stats in enumerate((selected, baseline), 1):
        if stats["hard_block_violations"]:
            records[-1][column] = "Infeasible"
            records[-1][3] = "—"
            worse[-1] = False
    add("Data coverage", "coverage_percent", "%", 1)
    add("Known soft conflicts", "conflict_count")
    add("Missing information", "unknown_count")
    add("Blocked traversals", "hard_block_violations")
    for row in comparison["obstacle_rows"]:
        if row["selected"] or row["standard"]:
            delta = row["selected"] - row["standard"]
            records.append([row["label"], str(row["selected"]), str(row["standard"]), f"{delta:+d}"])
            worse.append(delta > 0)
    table_height = (len(records) + 1) * .21
    table = ax.table(cellText=records, colLabels=["Measure / encounters", "Selected", "Standard", "Change"],
                     colWidths=[.43, .20, .20, .17], cellLoc="left", bbox=(0, 1 - (y + table_height) / height, 1, table_height / height))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.set_gid("route-comparison-table")
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e9ee")
        cell.set_linewidth(.4)
        cell.set_facecolor("#eef3f8" if row == 0 else "white" if row % 2 else "#f7f9fb")
        if cell.get_text().get_text() == "No active score":
            cell.PAD = .03  # Retain the complete label at 9pt within the numeric column.
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(_BLUE if column == 1 else _ORANGE if column == 2 else _NAVY)
        elif cell.get_text().get_text() == "Infeasible" or column == 3 and worse[row - 1]:
            cell.set_facecolor("#fce8e6")
            cell.get_text().set_color(_RED)
    y += table_height + .10
    if baseline["hard_block_violations"]:
        put("STANDARD INFEASIBLE under selected hard rules", bold=True, colour=_RED, gid="comparison-warning")
    else:
        put("Feasibility is under selected model rules, not a safety guarantee.", gid="comparison-warning")
    put("Change = selected − standard; red marks worse, not all changes better.\n"
        "All grouped encounters (not stair risers); full detail: route_summary.md",
        gid="comparison-footer", advance=.32)
    put("Standard: shortest mapped pedestrian distance; no soft weights,\n"
        "wheelchair hard requirement or handrail preference. Access/direction rules\n"
        "remain. Differences reflect hard rules as well as weights.", gid="comparison-baseline")


def _comparison_profile_height(view: dict) -> float:
    return .66 + max(1, (len(view.get("selected_preferences", [])) + 1) // 2) * .18 + .40


def _comparison_height(view: dict) -> float:
    obstacles = view.get("comparison", {}).get("obstacle_rows", [])
    return _comparison_profile_height(view) + (8 + sum(bool(row["selected"] or row["standard"]) for row in obstacles)) * .21 + 1.02


def _same_path(route: Route, standard: Route | None) -> bool:
    return standard is not None and route.nodes == standard.nodes and tuple(
        edge.id for edge in route.edges) == tuple(edge.id for edge in standard.edges)


def _route_figure(graph: InMemoryGraph, route: Route, positions: Mapping[str, Point], rows: list[dict],
                  counts: dict[str, int], extent: Extent, presentation: dict | None,
                  title: str, synthetic: bool, corridor: _Corridor,
                  standard_route: Route | None = None) -> Figure:
    summary_height = _comparison_height(presentation or {})
    key_height = .40 + .26 * max(1, ceil(min(len(rows), _MAX_LABELS) / 2))
    height = max(10, 2 * ceil((summary_height + key_height + .08) / (.77 * 2))) if standard_route is not None else 10
    figure = _figure(title, "Route & obstacles", synthetic, (16, height))
    ax = figure.add_axes(_ROUTE_RECT, label="route-focus")
    if standard_route is not None:
        bottom = .92 - summary_height / height
        summary = figure.add_axes((.64, bottom, .34, summary_height / height), label="route-summary")
        key = figure.add_axes((.64, bottom - (key_height + .08) / height, .34, key_height / height), label="obstacle-key")
    else:
        summary = figure.add_axes((.64, .43, .34, .49), label="route-summary")
        key = figure.add_axes((.82, .205, .16, .20), label="obstacle-key")
    title_note = " · same path" if _same_path(route, standard_route) else ""
    _map_axes(ax, [positions[node] for node in route.nodes],
              f"Route union · {corridor.radius_m:g} m corridor{title_note}", extent=extent)
    ax.set_anchor("N")
    segments = [segment for parts in corridor.segments.values() for segment in parts]
    _lines(ax, segments, _GRAPH_GREY, "graph-edges", width=1.1, zorder=1)
    _obstacles(ax, rows, [positions[route.nodes[0]], positions[route.nodes[-1]]])
    if standard_route is None:
        _summary_panel(summary, route, presentation)
    else:
        _comparison_panel(summary, presentation, route, standard_route)
    _marker_key(key, rows)
    if standard_route is None:
        inset = figure.add_axes((.665, .205, .13, .19), label="area-context")
        _map_axes(inset, [], "Corridor context", extent=extent)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=2))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=2))
        _lines(inset, segments, _GRAPH_GREY, "graph-edges", width=.8, zorder=1)
        _route_line(inset, route, positions, width=1.8)
    # Foreground route is deliberately added after every background and marker.
    _route_line(ax, route, positions)
    if standard_route is not None:
        _standard_line(ax, standard_route, positions)
    _finish_route(ax, route, positions, _BLUE)
    handles = [_route_key(_BLUE), _legend_line("Corridor context", _GRAPH_GREY)]
    if standard_route is not None:
        handles.append(Line2D([], [], color=_ORANGE, linestyle="--", marker=">",
                              label="Standard (N) / travel direction"))
    for label, colour in (("Conflict / hard block", _RED), ("Other encounter", "#a26812"), ("Off-route feature", "#637184")):
        handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                              markeredgecolor=colour, label=label))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .105), ncols=3, frameon=False, fontsize=9)
    figure.text(.065, .076,
                f"Markers: {counts['selected_candidates']} selected / {counts['total_candidates']} candidates; "
                f"{counts['cropped']} cropped; {counts['displayed']} displayed ({counts['labelled']} numbered); "
                f"{counts['suppressed']} suppressed. Type filter omitted {counts['omitted_by_filter']}.", fontsize=9)
    figure.text(.065, .055,
                f"Suppressed: {counts['coalesced']} coalesced (× in key), {counts['omitted_by_limit']} beyond marker limit "
                f"{counts['max_markers']}, {counts['unknown_suppressed']} unknown. Complete report unaffected: route_summary.md.", fontsize=9)
    figure.text(.065, .026, _caption(synthetic), fontsize=9)
    return figure


def _accessibility_figure(graph: InMemoryGraph, route: Route, positions: Mapping[str, Point],
                          scores: Mapping[str, _Score], title: str, synthetic: bool,
                          corridor: _Corridor, standard_route: Route | None = None) -> Figure:
    xmin, ymin, xmax, ymax = corridor.geometry.bounds
    stacked = (xmax - xmin) / (ymax - ymin) > 1.8
    height = 10.
    left_rect, right_rect = (.065, .215, .40, .675), (.57, .215, .40, .675)
    if stacked:
        # Full-width equal panels with physical-inch gutters for titles, ticks
        # and the footer. Derive height from the *padded* metric union: no empty
        # square axes, stretching, or rotation of either route is necessary.
        x0, x1, y0, y1 = corridor.extent
        panel_height = 16 * .905 * (y1 - y0) / (x1 - x0)
        height = 2 * panel_height + 4.2
        left_rect = (.065, (2.15 + panel_height + .85) / height, .905, panel_height / height)
        right_rect = (.065, 2.15 / height, .905, panel_height / height)
    figure = _figure(title, "Area accessibility", synthetic, (16, height))
    if stacked:
        for text in figure.texts:
            x, y = text.get_position()
            text.set_position((x, 1 - (1 - y) * 10 / height))
    left = figure.add_axes(left_rect, label="area-scores")
    right = figure.add_axes(right_rect, label="route-detail")
    _map_axes(left, [], "Selected route · selected-profile edge scores", extent=corridor.extent)
    blocked_standard = standard_route is not None and any(scores[edge.id].blocked for edge in standard_route.edges)
    right_title = ("No-preference standard" + (" · infeasible under selected rules" if blocked_standard else "")
                   if standard_route is not None else "Route scores · standard route not supplied")
    if _same_path(route, standard_route):
        right_title += " · same path"
    _map_axes(right, [], right_title, extent=corridor.extent)
    scale = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap="cividis")
    true = _true_segments(graph.edges, positions)
    shifted = _corridor_scores(graph, positions, corridor)
    for ax in ((left, right) if standard_route is not None else (left,)):
        _score_lines(ax, graph.edges, true, scores, scale, "graph", width=.8, clipped=shifted)
    _route_line(left, route, positions, width=6)
    if standard_route is not None:
        _score_lines(left, route.edges, true, scores, scale, "path", width=3.2, zorder=13)
    _finish_route(left, route, positions, _BLUE)
    if standard_route is not None:
        _standard_line(right, standard_route, positions)
        blocked = [true[edge.id] for edge in standard_route.edges if scores[edge.id].blocked]
        if blocked:
            _lines(right, blocked, _RED, "standard-blocked", dashed=True, width=3.2, zorder=16)
        _endpoints(right, standard_route, positions)
    else:
        _route_line(right, route, positions, width=6)
        _score_lines(right, route.edges, true, scores, scale, "path", width=3.2, zorder=13)
        _finish_route(right, route, positions, _BLUE)
    colour_axis = figure.add_axes((.30, 1.39 / height, .40, .19 / height), label="edge-score-scale")
    colourbar = figure.colorbar(scale, cax=colour_axis, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    if colourbar.solids is not None:
        colourbar.solids.set_rasterized(False)  # Matplotlib otherwise rasterizes continuous colourbars in PDF.
    colourbar.set_ticklabels(["0 (low)", "25", "50", "75", "100 (high)"])
    colourbar.set_label("Event-inclusive edge score (0-100)", fontsize=9)
    colourbar.ax.tick_params(labelsize=9)
    handles = [_route_key(_BLUE), _legend_line("Hard-blocked direction", _RED, dashed=True),
               _legend_line("No active scored exposure", _GREY)]
    if standard_route is not None:
        handles.insert(1, _legend_line("No-preference standard", _ORANGE, dashed=True))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .64 / height), ncols=4, frameon=False, fontsize=9)
    figure.text(.065, .48 / height,
                f"{len(corridor.segments)} of {len(graph.edges)} directed edges intersect the {corridor.radius_m:g} m route-union corridor. "
                "Full diagnostics retained; identical selected-profile score scale in both panels.", fontsize=9)
    figure.text(.065, .30 / height, "Coincident background directions offset by ≤1 m for visibility, then clipped to the same corridor; "
                "selection uses original geometry. Route traces stay exact.", fontsize=9)
    figure.text(.065, .12 / height, _caption(synthetic), fontsize=9)
    return figure


def _validated_radius(value: object) -> float:
    radius = _number(value, "corridor_radius_m")
    if not 0 < radius <= 5000:
        raise ValueError("corridor_radius_m must be > 0 and <= 5000 metres")
    return radius


def create_figures(
    graph: InMemoryGraph,
    route: Route,
    diagnostics: Mapping[str, dict],
    pois: list[dict],
    *,
    title: str,
    synthetic: bool,
    output_dir: Path,
    presentation: dict | None = None,
    standard_route: Route | None = None,
    corridor_radius_m: float = 150.0,
    file_formats: tuple[str, ...] = ("png",),
    dpi: int = 240,
) -> list[Path]:
    """Write two publication maps; PNG by default, vector PDF on request.

    All graph edges require explicit ``blocked`` and ``score`` diagnostics.
    Numeric score, way_score and coverage values must be finite and in [0, 100];
    None is allowed. Hard blocks take precedence over numeric colours, and an
    unblocked None score is grey (no active criteria or scored exposure).

    presentation is the output-only build_route_presentation dictionary. Its
    already-filtered map_obstacles and relevant standard encounters share a cap;
    no diagnostics, metrics, geometry or complete encounter lists are changed.
    Without it, old POI callers get stairs / blocked barriers, at most 12 markers.
    standard_route is the supplied no-preference trace, never reconstructed or
    shifted. None supports archived selected-only runs without a fake comparison.
    Both maps clip context to the union of route buffers in a route-centred AEQD
    projection. corridor_radius_m must be finite, positive and at most 5000 m.
    Only coincident background score segments are offset (at most 1 m), then
    clipped back to that same corridor; membership uses original geometry.
    No observations are inferred from unknowns. Only PNG and PDF are accepted.
    Inputs are not mutated. Existing targets (including symbolic links) fail
    before writing; exclusive file creation also protects against races. Files
    created by a failed call are removed after their handles have been closed.
    Return order is route_obstacles, area_accessibility, each in file_formats order.
    """
    if not isinstance(title, str) or not isinstance(synthetic, bool):
        raise ValueError("title must be a string and synthetic must be a boolean")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if (not isinstance(file_formats, tuple) or not file_formats
            or any(fmt not in ("png", "pdf") for fmt in file_formats)
            or len(set(file_formats)) != len(file_formats)):
        raise ValueError("file_formats must be a nonempty tuple of distinct 'png' and/or 'pdf'")
    output_dir = Path(output_dir)
    names = ("route_obstacles", "area_accessibility")
    paths = [output_dir / f"{name}.{fmt}" for name in names for fmt in file_formats]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing plot: {path}")
    scores = _validated_scores(graph, diagnostics)
    _validate_route(graph, route)
    radius = _validated_radius(corridor_radius_m)
    if standard_route is not None:
        _validate_route(graph, standard_route)
    routes = [route] + ([standard_route] if standard_route is not None else [])
    positions, transformer = _project_graph(graph, [node for path in routes for node in path.nodes])
    paths_in_metres = [[positions[key] for key in path.nodes] for path in routes]
    corridor = _corridor(graph, positions, paths_in_metres, radius)
    rows, counts, extent = _select_obstacles(pois, _comparison_markers(presentation, standard_route), transformer,
                                             paths_in_metres[0], paths=paths_in_metres, corridor=corridor)
    if presentation is not None and "synthetic" in presentation:
        if not isinstance(presentation["synthetic"], bool):
            raise ValueError("presentation synthetic must be a boolean")
        synthetic = synthetic or presentation["synthetic"]
    figures: list[Figure] = []
    created: list[Path] = []
    try:
        with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "text.usetex": False}):
            figures.append(_route_figure(graph, route, positions, rows, counts, extent, presentation, title, synthetic,
                                         corridor, standard_route))
            figures.append(_accessibility_figure(graph, route, positions, scores, title, synthetic, corridor,
                                                 standard_route))
            output_dir.mkdir(parents=True, exist_ok=True)
            with ExitStack() as stack:
                files = {}
                # Reserve every target before writing any bytes. 'xb' never truncates.
                for path in paths:
                    files[path] = stack.enter_context(path.open("xb"))
                    created.append(path)
                for figure, name in zip(figures, names):
                    for fmt in file_formats:
                        metadata = {"Title": title}
                        if fmt == "pdf":
                            metadata.update({"Creator": "osm_accessibility.visualization",
                                             "CreationDate": None, "ModDate": None})
                        figure.savefig(files[output_dir / f"{name}.{fmt}"], format=fmt, dpi=dpi,
                                       bbox_inches="tight", facecolor="white", metadata=metadata)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for figure in figures:
            figure.clear()
    return paths
