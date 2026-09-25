"""Scientific figures from supplied graph geometry and profile diagnostics.

No data are fetched and no scores are recomputed. ``score`` includes transition
events; ``way_score`` is not substituted for it. Geometry consists of the stored
node-to-node segments: missing intermediate shapes are not inferred. Coincident
edges are separated only in the graph score panel, never in the selected path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from math import atan2, cos, degrees, fsum, hypot, isfinite, radians, sin
from numbers import Real
from pathlib import Path
from textwrap import shorten, wrap

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError

from .graph import InMemoryGraph
from .models import Edge, Route

Point = tuple[float, float]
Segment = tuple[Point, Point]

_BLUE = "#1464b4"
_RED = "#c52b2f"
_GREY = "#8c8c8c"
_GRAPH_GREY = "#c2c2c2"
_OFFSET_M = 1.0
_MAX_POI_LABELS = 25
_POI_SYMBOLS = {
    "stairs": ("^", "Stairs"),
    "ramp": (">", "Ramp"),
    "barrier": ("X", "Barrier"),
    "kerb": ("s", "Kerb"),
    "crossing": ("P", "Crossing"),
    "incline": ("D", "Incline"),
    "wheelchair_restriction": ("v", "Wheelchair restriction"),
}


@dataclass(frozen=True)
class _Score:
    blocked: bool
    value: float | None


@dataclass(frozen=True)
class _POI:
    identifier: str
    kind: str
    position: Point
    on_route: bool
    blocked: bool | None
    description: str


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


def _project_graph(graph: InMemoryGraph) -> tuple[dict[str, Point], Transformer]:
    if not graph.nodes:
        raise ValueError("A local projection requires at least one graph node")
    coordinates = [graph.nodes[key].coordinate for key in sorted(graph.nodes)]
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


def _project_pois(pois: list[dict], transformer: Transformer) -> list[_POI]:
    result = []
    required = {"id", "kind", "latitude", "longitude", "on_route", "blocked", "description"}
    for record in pois:
        if not isinstance(record, Mapping) or not required.issubset(record):
            raise ValueError("Each POI requires id, kind, coordinates, on_route, blocked and description")
        identifier = record["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError("POI id must be a nonempty string or integer")
        if not isinstance(record["kind"], str) or record["kind"] not in _POI_SYMBOLS:
            raise ValueError(f"Unsupported POI kind for {identifier!r}")
        if not isinstance(record["on_route"], bool):
            raise ValueError(f"POI on_route for {identifier!r} must be a boolean")
        if record["blocked"] is not None and not isinstance(record["blocked"], bool):
            raise ValueError(f"POI blocked for {identifier!r} must be a boolean or None")
        if not isinstance(record["description"], str):
            raise ValueError(f"POI description for {identifier!r} must be a string")
        latitude = _number(record["latitude"], f"POI latitude for {identifier!r}")
        longitude = _number(record["longitude"], f"POI longitude for {identifier!r}")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"POI coordinates for {identifier!r} are outside WGS84 bounds")
        result.append(_POI(str(identifier), record["kind"], _project(transformer, longitude, latitude),
                           record["on_route"], record["blocked"], record["description"]))
    return sorted(result, key=lambda poi: (
        poi.kind, poi.identifier, poi.position, poi.on_route, str(poi.blocked), poi.description
    ))


def _map_axes(ax: Axes, points: Sequence[Point], title: str) -> None:
    xs, ys = zip(*points)
    padding = max(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.07, 5.0)
    extent_x, extent_y = max(xs) - min(xs) + 2 * padding, max(ys) - min(ys) + 2 * padding
    # Match the plot rectangle by expanding the shorter extent. Equal metre
    # scaling is retained without squeezing a tall graph into a tiny subplot.
    position = ax.get_position(original=True)
    fig_width, fig_height = ax.figure.get_size_inches()
    ratio = position.width * fig_width / (position.height * fig_height)
    if extent_x / extent_y < ratio:
        extent_x = extent_y * ratio
    else:
        extent_y = extent_x / ratio
    center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    ax.set_xlim(center_x - extent_x / 2, center_x + extent_x / 2)
    ax.set_ylim(center_y - extent_y / 2, center_y + extent_y / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, fontsize=11, pad=12)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(color="#e6e6e6", linewidth=0.6)


def _directions(ax: Axes, segments: Sequence[Segment], colours: Sequence[str], gid: str) -> None:
    moving = [(a, b, colour) for (a, b), colour in zip(segments, colours) if a != b]
    if not moving:
        return
    arrows = ax.quiver(
        [a[0] + 0.4 * (b[0] - a[0]) for a, b, _ in moving],
        [a[1] + 0.4 * (b[1] - a[1]) for a, b, _ in moving],
        [0.22 * (b[0] - a[0]) for a, b, _ in moving],
        [0.22 * (b[1] - a[1]) for a, b, _ in moving],
        color=[colour for _, _, colour in moving], angles="xy", scale_units="xy", scale=1,
        units="inches", width=0.022, headwidth=5, headlength=6, minlength=0, zorder=5,
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
        ax.scatter(*zip(*stationary), c=colour, s=22, marker="x" if dashed else "o", zorder=zorder)
    return collection


def _endpoints(ax: Axes, route: Route, positions: Mapping[str, Point]) -> None:
    start, end = positions[route.nodes[0]], positions[route.nodes[-1]]
    ax.scatter(*start, s=90, marker="o", facecolors="white", edgecolors="#195f45", linewidths=2, zorder=8)
    ax.scatter(*end, s=35, marker="s", c="#222222", zorder=9)
    labels = [("Start / end", start, (6, 8))] if start == end else [
        ("Start", start, (6, 8)), ("End", end, (6, -14))
    ]
    for text, point, offset in labels:
        label = ax.annotate(text, point, xytext=offset, textcoords="offset points", fontsize=9, zorder=10)
        label.set_gid("route-endpoint-label")


def _route_line(ax: Axes, route: Route, positions: Mapping[str, Point]) -> None:
    points = [positions[node_id] for node_id in route.nodes]
    line, = ax.plot(*zip(*points), color=_BLUE, linewidth=2.8, zorder=3)
    line.set_gid("selected-route")
    segments = [(positions[edge.source], positions[edge.target]) for edge in route.edges]
    _directions(ax, segments, [_BLUE] * len(segments), "route-directions")
    _endpoints(ax, route, positions)


def _blocked_lines(ax: Axes, edges: Sequence[Edge], segments: Mapping[str, Segment],
                   scores: Mapping[str, _Score], gid: str) -> None:
    blocked = [segments[edge.id] for edge in edges if scores[edge.id].blocked]
    if blocked:
        _lines(ax, blocked, _RED, gid, dashed=True, width=2.2, zorder=4)
        _directions(ax, blocked, [_RED] * len(blocked), f"{gid}-directions")


def _score_lines(ax: Axes, edges: Sequence[Edge], segments: Mapping[str, Segment],
                 scores: Mapping[str, _Score], scale: ScalarMappable, prefix: str) -> None:
    active = [edge for edge in edges if not scores[edge.id].blocked and scores[edge.id].value is not None]
    unscored = [segments[edge.id] for edge in edges if not scores[edge.id].blocked and scores[edge.id].value is None]
    if active:
        geometry = [segments[edge.id] for edge in active]
        values = [scores[edge.id].value for edge in active]
        collection = LineCollection(geometry, cmap=scale.cmap, norm=scale.norm, linewidths=2.6, zorder=3)
        collection.set_array(values)
        collection.set_gid(f"{prefix}-score")
        ax.add_collection(collection)
        colours = [to_hex(scale.to_rgba(value)) for value in values]
        _directions(ax, geometry, colours, f"{prefix}-score-directions")
        for (start, end), colour in zip(geometry, colours):
            if start == end:
                ax.scatter(*start, c=colour, s=22, zorder=3)
    if unscored:
        _lines(ax, unscored, _GREY, f"{prefix}-unscored", width=2.2)
        _directions(ax, unscored, [_GREY] * len(unscored), f"{prefix}-unscored-directions")
    _blocked_lines(ax, edges, segments, scores, f"{prefix}-blocked")
    if not edges:
        ax.text(0.5, 0.92, "No edges to score", transform=ax.transAxes, ha="center", fontsize=9)


def _pois(ax: Axes, pois: Sequence[_POI]) -> None:
    groups: dict[tuple[str, bool, str], list[_POI]] = defaultdict(list)
    for poi in pois:
        status = "blocked" if poi.blocked is True else "unknown" if poi.blocked is None else "not-blocked"
        groups[(poi.kind, poi.on_route, status)].append(poi)
    for (kind, on_route, status), group in sorted(groups.items()):
        points = [poi.position for poi in group]
        symbols = ax.scatter(
            *zip(*points), marker=_POI_SYMBOLS[kind][0], s=68,
            facecolors=_BLUE if on_route else "none",
            edgecolors=_RED if status == "blocked" else "#444444", linewidths=1.5, zorder=7,
        )
        symbols.set_gid(f"poi-{kind}-{'on' if on_route else 'off'}-{status}")
    offsets = ((9, 9), (9, -15), (-9, 9), (-9, -15), (18, 0), (-18, 0), (0, 20), (0, -22))
    co_located: dict[Point, int] = defaultdict(int)
    for index, poi in enumerate(pois[:_MAX_POI_LABELS]):
        ordinal = co_located[poi.position]
        offset = offsets[ordinal % len(offsets)]
        co_located[poi.position] += 1
        label = ax.annotate(
            str(index + 1), poi.position, xytext=offset, textcoords="offset points", fontsize=7,
            ha="left" if offset[0] > 0 else "right" if offset[0] < 0 else "center", color="#333333", zorder=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#c8c8c8", "alpha": .9},
            arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": .5},
        )
        label.set_gid("poi-label")


def _poi_key(ax: Axes, pois: Sequence[_POI]) -> None:
    ax.set_axis_off()
    ax.set_title("POI lookup (full details in pois.csv)", fontsize=9, loc="left", pad=12)
    selected = pois[:_MAX_POI_LABELS]
    step = min(.065, .98 / max(len(selected), 1))
    for index, poi in enumerate(selected):
        text = f"{index + 1}. {_POI_SYMBOLS[poi.kind][1]}: {poi.identifier}"
        lines = wrap(shorten(text, width=86, placeholder="..."), width=47)
        label = ax.text(0, .97 - index * step, "\n".join(lines[:2]), transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, linespacing=1.15,
                        color=_RED if poi.blocked else "#333333")
        label.set_gid("poi-key")
    if not selected:
        ax.text(0, .95, "No recorded POIs in this extract.", transform=ax.transAxes, fontsize=8)


def _legend_line(label: str, colour: str, *, dashed: bool = False) -> Line2D:
    return Line2D([], [], color=colour, linewidth=2, linestyle="--" if dashed else "-", label=label)


def _endpoint_key() -> list[Line2D]:
    return [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor="#195f45", label="Start"),
        Line2D([], [], linestyle="none", marker="s", color="#222222", label="End"),
    ]


def _caption(synthetic: bool, detail: str) -> str:
    attribution = ("Synthetic example; not observed OpenStreetMap data." if synthetic else
                   "Map data: © OpenStreetMap contributors, ODbL 1.0 — https://www.openstreetmap.org/copyright")
    return "\n".join((
        attribution,
        "Plots reflect the chosen profile and supplied data; they are not an accuracy proof.",
        "Event-inclusive edge score combines way criteria and transition events; not a pure continuous score.",
        "Local WGS84 azimuthal equidistant: metre axes give distance scale; northing increases upwards.",
        detail,
    ))


def _figure(title: str, size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, facecolor="white")
    FigureCanvasAgg(figure)
    figure.suptitle(title, fontsize=14, y=0.98)
    return figure


def _route_figure(graph: InMemoryGraph, route: Route, positions: Mapping[str, Point],
                  scores: Mapping[str, _Score], title: str, synthetic: bool) -> Figure:
    figure = _figure(title, (9, 7))
    ax = figure.add_subplot(111)
    figure.subplots_adjust(left=0.11, right=0.96, bottom=0.29, top=0.87)
    _map_axes(ax, [positions[key] for key in route.nodes], "Selected path: exact stored geometry")
    _route_line(ax, route, positions)
    _blocked_lines(ax, route.edges, _true_segments(graph.edges, positions), scores, "route-blocked")
    figure.legend(handles=[_legend_line("Selected route / arrows: travel direction", _BLUE),
                           _legend_line("Hard-blocked direction", _RED, dashed=True), *_endpoint_key()],
                  loc="lower center", bbox_to_anchor=(0.5, 0.17), ncols=2, frameon=False, fontsize=8)
    figure.text(0.04, 0.025, _caption(synthetic,
                "Every stored path vertex is retained; no simplification or lateral displacement."), fontsize=8)
    return figure


def _graph_figure(graph: InMemoryGraph, route: Route, positions: Mapping[str, Point],
                  scores: Mapping[str, _Score], pois: Sequence[_POI], title: str, synthetic: bool) -> Figure:
    figure = _figure(title, (12, 9))
    ax = figure.add_axes((.08, .35, .55, .53))
    lookup = figure.add_axes((.67, .35, .30, .53))
    _map_axes(ax, [*positions.values(), *(poi.position for poi in pois)], "Graph, selected path and all supplied POIs")
    segments = _true_segments(graph.edges, positions)
    _lines(ax, list(segments.values()), _GRAPH_GREY, "graph-edges", width=1.3, zorder=1)
    _route_line(ax, route, positions)
    _blocked_lines(ax, graph.edges, segments, scores, "graph-blocked")
    _pois(ax, pois)
    _poi_key(lookup, pois)
    handles = [_legend_line("All graph edges", _GRAPH_GREY), _legend_line("Selected route", _BLUE),
               _legend_line("Hard-blocked direction", _RED, dashed=True), *_endpoint_key()]
    for kind in sorted({poi.kind for poi in pois}):
        marker, label = _POI_SYMBOLS[kind]
        handles.append(Line2D([], [], linestyle="none", marker=marker, markerfacecolor="none",
                              markeredgecolor="#444444", label=label))
    for label, face, edge in (("On-path POI (filled)", _BLUE, "#444444"),
                              ("Off-path POI (hollow)", "none", "#444444"),
                              ("Known blocked POI (red outline)", "none", _RED)):
        handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor=face,
                              markeredgecolor=edge, label=label))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.17),
                  ncols=3, frameon=False, fontsize=8)
    detail = (f"All {len(pois)} supplied POIs shown; {min(len(pois), _MAX_POI_LABELS)} numbered labels sorted by kind/ID/position."
              " Other POI outlines do not assert passability.\n"
              "Coincident directions share true geometry here; a blocked reverse edge may overlap the blue path.")
    figure.text(0.04, 0.025, _caption(synthetic, detail), fontsize=8)
    return figure


def _accessibility_figure(graph: InMemoryGraph, route: Route, positions: Mapping[str, Point],
                          scores: Mapping[str, _Score], title: str, synthetic: bool) -> Figure:
    figure = _figure(title, (13, 8))
    left, right = figure.add_subplot(121), figure.add_subplot(122)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.40, top=0.87, wspace=0.27)
    shifted = score_segments(graph, positions)
    extent = [*positions.values(), *(point for segment in shifted.values() for point in segment)]
    _map_axes(left, extent, "All directed graph edge scores")
    _map_axes(right, [positions[key] for key in route.nodes], "Selected path edge scores: true geometry")
    scale = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap="cividis")
    _score_lines(left, graph.edges, shifted, scores, scale, "graph")
    _score_lines(right, route.edges, _true_segments(graph.edges, positions), scores, scale, "path")
    _endpoints(right, route, positions)
    colour_axis = figure.add_axes((0.30, 0.28, 0.40, 0.024))
    colourbar = figure.colorbar(scale, cax=colour_axis, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colourbar.set_ticklabels(["0 (low)", "25", "50", "75", "100 (high)"])
    colourbar.set_label("Event-inclusive edge score (0-100)", fontsize=9)
    colourbar.ax.tick_params(labelsize=8)
    figure.legend(handles=[_legend_line("Hard-blocked direction", _RED, dashed=True),
                           _legend_line("No active criteria / no scored exposure", _GREY), *_endpoint_key()],
                  loc="lower center", bbox_to_anchor=(0.5, 0.15), ncols=4, frameon=False, fontsize=8)
    detail = (f"Graph score panel only: coincident-edge offsets (at most {_OFFSET_M:g} m) are not real geometry.\n"
              "Selected path edges are unshifted. Arrows indicate direction; zero-length edges are points.")
    figure.text(0.04, 0.025, _caption(synthetic, detail), fontsize=8)
    return figure


def create_figures(
    graph: InMemoryGraph,
    route: Route,
    diagnostics: Mapping[str, dict],
    pois: list[dict],
    *,
    title: str,
    synthetic: bool,
    output_dir: Path,
    file_formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 180,
) -> list[Path]:
    """Write route, graph/POI and two-panel accessibility figures without overwrite.

    All graph edges require explicit ``blocked`` and ``score`` diagnostics.
    Numeric score, way_score and coverage values must be finite and in [0, 100];
    None is allowed. Hard blocks take precedence over numeric colours, and an
    unblocked None score is grey (no active criteria or scored exposure).

    All supplied POIs are plotted, using their supplied on_route/blocked flags;
    at most 25 receive deterministic annotations. Only PNG and PDF are accepted.
    Inputs are not mutated. Existing targets (including symbolic links) fail
    before writing; exclusive file creation also protects against races. Files
    created by a failed call are removed after their handles have been closed.
    Return order is route, graph_pois, accessibility, each in file_formats order.
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
    paths = [output_dir / f"{name}.{fmt}" for name in ("route", "graph_pois", "accessibility") for fmt in file_formats]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing plot: {path}")
    scores = _validated_scores(graph, diagnostics)
    _validate_route(graph, route)
    positions, transformer = _project_graph(graph)
    projected_pois = _project_pois(pois, transformer)
    figures: list[Figure] = []
    created: list[Path] = []
    try:
        figures.append(_route_figure(graph, route, positions, scores, title, synthetic))
        figures.append(_graph_figure(graph, route, positions, scores, projected_pois, title, synthetic))
        figures.append(_accessibility_figure(graph, route, positions, scores, title, synthetic))
        output_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            files = {}
            # Reserve every target before writing any bytes. 'xb' never truncates.
            for path in paths:
                files[path] = stack.enter_context(path.open("xb"))
                created.append(path)
            for figure, name in zip(figures, ("route", "graph_pois", "accessibility")):
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
