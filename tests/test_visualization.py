"""Small, manually specified scientific fixtures; no image golden files."""

from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, hypot, radians, sin, sqrt
from pathlib import Path
from struct import unpack

import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.quiver import Quiver
from pyproj import CRS, Transformer

from osm_accessibility import visualization as viz
from osm_accessibility.evaluation import Evaluator
from osm_accessibility.graph import InMemoryGraph
from osm_accessibility.models import Coordinate, Edge, Node, Route
from osm_accessibility.outputs import (
    PREFERENCE_LABELS,
    PresentationOptions,
    build_route_presentation,
    collect_pois,
    edge_diagnostics,
)
from osm_accessibility.profiles import CostParameters, Profile
from osm_accessibility.visualization import (
    _project_graph,
    _validated_scores,
    create_figures,
    projected_positions,
    score_segments,
    select_map_obstacles,
)


def _distance(a: Node, b: Node) -> float:
    """Independent haversine calculation, without the production distance helper."""
    lat1, lat2 = radians(a.coordinate.latitude), radians(b.coordinate.latitude)
    delta_lon = radians(b.coordinate.longitude - a.coordinate.longitude)
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    return 2 * 6_371_008.8 * atan2(sqrt(h), sqrt(max(0.0, 1 - h)))


@pytest.fixture
def example():
    nodes = [Node(key, Coordinate(lat, lon)) for key, lat, lon in (
        ("a", 49.5900, 11.0000), ("b", 49.5900, 11.0010),
        ("c", 49.5910, 11.0010), ("d", 49.5910, 11.0000),
        ("e", 49.5905, 11.0020), ("b-copy", 49.5900, 11.0010),
    )]
    by_id = {node.id: node for node in nodes}
    edges = [Edge(key, source, target, key, _distance(by_id[source], by_id[target]), tags)
             for key, source, target, tags in (
                 ("ab", "a", "b", {"highway": "footway"}),
                 ("ab-parallel", "a", "b-copy", {"highway": "footway"}),
                 ("ba-blocked", "b", "a", {"highway": "footway", "access": "no"}),
                 ("bc", "b", "c", {"highway": "footway", "incline": "6%"}),
                 ("be", "b", "e", {"highway": "footway"}),
                 ("cd-stairs", "c", "d", {"highway": "steps", "wheelchair": "no"}),
                 ("da-unscored", "d", "a", {"highway": "footway"}),
             )]
    graph = InMemoryGraph(nodes, edges)
    route = Route(("a", "b", "c"), (edges[0], edges[3]),
                  edges[0].distance_m + edges[3].distance_m,
                  edges[0].distance_m + edges[3].distance_m)
    values = {"ab": 70.0, "ab-parallel": 100.0, "bc": 35.0, "be": 0.0}
    diagnostics = {}
    for edge in edges:
        blocked = edge.id in {"ba-blocked", "cd-stairs"}
        score = values.get(edge.id)
        diagnostics[edge.id] = {
            "blocked": blocked, "score": score, "way_score": 95.0 if score is not None else None,
            "coverage": 60.0 if score is not None else None, "risk": 0.05,
            "transition_risk": 0.8, "penalty_m": 20.0,
            "cost_m": None if blocked else edge.distance_m + 20.0,
            "hard_blocks": ["explicit restriction"] if blocked else [],
        }
    kinds = ("stairs", "ramp", "barrier", "kerb", "crossing", "incline", "wheelchair_restriction")
    pois = [
        {"id": f"off-{index}", "kind": kind, "latitude": 49.5915,
         "longitude": 11.0000 + index * 0.0004, "on_route": False,
         "blocked": True if kind in {"stairs", "barrier", "wheelchair_restriction"} else None,
         "description": f"Mapped {kind}"}
        for index, kind in enumerate(kinds)
    ]
    pois.append({"id": "on-crossing", "kind": "crossing", "latitude": 49.5900,
                 "longitude": 11.0010, "on_route": True, "blocked": False,
                 "description": "Selected path crossing"})
    return graph, route, diagnostics, pois


def _segments(collection: LineCollection):
    return [tuple(tuple(float(value) for value in point) for point in segment)
            for segment in collection.get_segments()]


def _collection(ax, gid: str):
    return next(artist for artist in ax.collections if artist.get_gid() == gid)


def _axis(figure, label):
    return next(ax for ax in figure.axes if ax.get_label() == label)


def _line(ax, gid):
    return next(line for line in ax.lines if line.get_gid() == gid)


def _text(figure):
    return "\n".join(text.get_text() for text in [*figure.texts, *(text for ax in figure.axes for text in ax.texts)])


def _candidate(identifier, kind="stairs", *, latitude=49.5903, longitude=11.0006,
               on_route=False, status="context", way_id=None, node_id=None, first_segment=None):
    return {"id": identifier, "kind": kind, "latitude": latitude, "longitude": longitude,
            "on_route": on_route, "status": status, "description": f"Mapped {kind}",
            "preference_labels": ["Selected preference"] if status == "conflict" else [],
            "edge_ids": ["ab"] if on_route else [], "way_id": way_id, "node_id": node_id,
            "first_segment": first_segment}


def _presentation(route, candidates, *, limit=12, omitted=7):
    """Only the documented builder fields; no renderer-specific input keys."""
    return {
        "profile_name": "Paper profile",
        "selected_preferences": [{"name": name, "label": label, "weight": .75, "importance": "Very high"}
                                 for name, label in PREFERENCE_LABELS.items()],
        "requirements": ["Wheelchair required: true (hard model rule). " + "Verbose policy. " * 20,
                         "Ramps allowed: true. The model accepts explicitly tagged concessions only."],
        "metrics": {"distance_m": route.distance_m, "cost_m": route.cost,
                    "score_percent": 78.25, "coverage_percent": 62.5},
        "conflicts": [{"kind": "stairs", "label": "Stairs", "location": "An OSM way; route segment 1",
                       "reason": f"Known concern {index}; " + "A long but readable observation. " * 8,
                       "edge_ids": ["ab"], "status": "conflict"} for index in range(4)],
        "unknowns": [{"kind": "width", "label": "Width", "reason": "Not recorded", "status": "unknown"}],
        "encountered_obstacles": [{"kind": "stairs", "status": "encountered"}],
        "map_obstacles": candidates,
        "filter_summary": {"selected_kinds": None, "total_candidates": len(candidates) + omitted,
                           "selected_candidates": len(candidates), "omitted_by_filter": omitted, "max_markers": limit},
        "validation": {"hard_constraint_violations": 0, "astar_dijkstra_agree": True,
                       "optimal_cost_difference_m": 0.0, "distance_baseline_m": 150.0,
                       "distance_overhead_percent": 22.0},
        "synthetic": True, "snapping": {}, "options": {"max_markers": limit, "obstacle_kinds": None},
    }


def _input_snapshot(graph, route, diagnostics, pois):
    # Immutable records contain read-only tag mappings that deepcopy cannot copy.
    return repr(graph.nodes), repr(graph.edges), repr(route), deepcopy(diagnostics), deepcopy(pois)


def test_projection_uses_metres_correct_axes_and_antimeridian():
    nodes = [Node("west", Coordinate(0, -0.001)), Node("origin", Coordinate(0, 0)),
             Node("east", Coordinate(0, 0.001))]
    graph = InMemoryGraph(nodes, [])
    positions = projected_positions(graph)
    expected = Transformer.from_crs("EPSG:4326", CRS.from_proj4(
        "+proj=aeqd +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
    ), always_xy=True)
    for node in nodes:
        assert positions[node.id] == pytest.approx(expected.transform(
            node.coordinate.longitude, node.coordinate.latitude), abs=1e-6)
    assert positions["east"][0] - positions["west"][0] == pytest.approx(222.639, abs=0.01)
    meridian = InMemoryGraph([Node("south", Coordinate(-0.001, 0)),
                              Node("north", Coordinate(0.001, 0))], [])
    northings = projected_positions(meridian)
    assert northings["north"][1] > northings["south"][1]
    assert abs(northings["north"][0]) < 1e-6
    dateline = InMemoryGraph([Node("a", Coordinate(0, 179.999)),
                              Node("b", Coordinate(0, -179.999))], [])
    points = projected_positions(dateline)
    assert hypot(points["a"][0] - points["b"][0], points["a"][1] - points["b"][1]) < 224


def test_offsets_are_small_deterministic_and_only_for_coincident_edges(example):
    graph, _, _, _ = example
    positions = projected_positions(graph)
    original = deepcopy(positions)
    shifted = score_segments(graph, positions, offset_m=0.75)
    repeated = InMemoryGraph(reversed(tuple(graph.nodes.values())), reversed(graph.edges))
    assert shifted == score_segments(repeated, projected_positions(repeated), offset_m=0.75)
    assert positions == original
    assert set(shifted) == {edge.id for edge in graph.edges}
    coincident = {"ab", "ab-parallel", "ba-blocked"}
    midpoints = []
    for edge in graph.edges:
        a, b = positions[edge.source], positions[edge.target]
        start, end = shifted[edge.id]
        if edge.id not in coincident:
            assert (start, end) == (a, b)
        assert hypot(start[0] - a[0], start[1] - a[1]) <= 0.750001
        assert (end[0] - start[0], end[1] - start[1]) == pytest.approx((b[0] - a[0], b[1] - a[1]))
        if edge.id in coincident:
            midpoints.append(((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
    assert len(set(midpoints)) == 3
    assert score_segments(graph, positions, offset_m=0) == {
        edge.id: (positions[edge.source], positions[edge.target]) for edge in graph.edges
    }


def test_two_default_high_resolution_pngs(example, tmp_path, monkeypatch):
    original_save = Figure.savefig
    figures = []

    def inspect(figure, destination, **kwargs):
        assert kwargs["format"] == "png"
        assert kwargs["dpi"] >= 220
        figures.append(figure)
        return original_save(figure, destination, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    paths = create_figures(*example, title="Default maps", synthetic=True, output_dir=tmp_path)
    assert paths == [tmp_path / "route_obstacles.png", tmp_path / "area_accessibility.png"]
    assert set(tmp_path.iterdir()) == set(paths)
    for path in paths:
        content = path.read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert min(unpack(">II", content[16:24])) >= 2000
        assert len(content) > 1000
    assert all(not figure.axes for figure in figures)  # No live figures retained after saving.


def test_four_artifacts_exact_foreground_geometry_scores_and_nonmutation(example, tmp_path, monkeypatch):
    graph, route, diagnostics, _ = example
    before = _input_snapshot(*example)
    candidates = [
        _candidate("conflict-" + "x" * 120, "crossing", on_route=True, status="conflict",
                   latitude=49.5900, longitude=11.0010, node_id="b", first_segment=1),
        _candidate("hard-barrier", "barrier", status="blocked"),
        _candidate("cropped", latitude=49.6),
    ]
    presentation = _presentation(route, candidates)
    before_presentation = deepcopy(presentation)
    positions = _project_graph(graph, route.nodes)[0]
    true = {edge.id: (positions[edge.source], positions[edge.target]) for edge in graph.edges}
    shifted = score_segments(graph, positions)
    original_save = Figure.savefig
    inspected = set()

    def foreground(ax, colour):
        halo, line = _line(ax, "route-halo"), _line(ax, "selected-route")
        expected = [positions[key] for key in route.nodes]
        for artist in (halo, line):
            assert list(zip(artist.get_xdata(), artist.get_ydata())) == expected
        assert halo.get_color() == "white" and halo.get_linewidth() > line.get_linewidth()
        assert line.get_color() == colour
        background = [artist for artist in ax.collections if (artist.get_gid() or "").startswith(("graph-", "obstacle-"))]
        assert all(halo.get_zorder() > artist.get_zorder() for artist in background)
        assert line.get_zorder() > halo.get_zorder()
        assert all(ax.get_children().index(line) > ax.get_children().index(artist) for artist in background)
        assert {text.get_text() for text in ax.texts if text.get_gid() == "route-endpoint-label"} == {"Start", "End"}
        arrows = _collection(ax, "route-directions")
        assert arrows.get_zorder() > line.get_zorder()
        assert all(arrows.get_zorder() > artist.get_zorder() for artist in background)
        assert ax.collections[-1] is arrows
        assert list(arrows.U) == pytest.approx([.22 * (true[e.id][1][0] - true[e.id][0][0]) for e in route.edges])
        assert list(arrows.V) == pytest.approx([.22 * (true[e.id][1][1] - true[e.id][0][1]) for e in route.edges])

    def inspect(figure, destination, **kwargs):
        name = Path(destination.name).stem
        inspected.add(name)
        maps = [ax for ax in figure.axes if ax.get_label() in {"route-focus", "area-context", "area-scores", "route-detail"}]
        for ax in maps:
            assert ax.get_aspect() == 1
            assert ax.get_xlabel() == "Easting (m)"
            assert ax.get_ylabel() == "Northing (m)"
            assert ax.get_axisbelow()
            assert all(text.get_fontsize() >= 9 for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts])
            if ax.get_label() != "area-context":
                assert ax.get_position(original=True).height > .60
        caption = "\n".join(text.get_text() for text in figure.texts)
        assert "Synthetic example" in caption
        assert "not safety guarantees" in caption
        assert "SYNTHETIC DATA" in caption
        assert all(text.get_fontsize() >= 9 for text in figure.texts)
        assert any("travel direction" in text.get_text() for legend in figure.legends for text in legend.get_texts())
        if name == "route_obstacles":
            ax = _axis(figure, "route-focus")
            foreground(ax, "#1464b4")
            assert _segments(_collection(ax, "graph-edges")) == [true[edge.id] for edge in graph.edges]
            assert min(_collection(ax, "graph-edges").get_colors()[0][:3]) > .9
            symbols = [artist for artist in ax.collections if isinstance(artist, PathCollection)
                       and (artist.get_gid() or "").startswith("obstacle-")]
            assert sum(len(artist.get_offsets()) for artist in symbols) == 2
            on_path = _collection(ax, "obstacle-crossing-conflict")
            assert tuple(on_path.get_offsets()[0]) == pytest.approx(positions["b"])
            for artist in symbols:
                assert len(artist.get_facecolors()) == 0  # Hollow even for on-route conflicts.
                for x, y in artist.get_offsets():
                    assert ax.get_xlim()[0] < x < ax.get_xlim()[1]
                    assert ax.get_ylim()[0] < y < ax.get_ylim()[1]
            assert len([text for text in ax.texts if text.get_gid() == "obstacle-label"]) == 2
            assert "x" * 100 not in _text(figure)
            assert "3 selected / 10 candidates; 1 cropped; 2 displayed" in caption
            summary = "\n".join(text.get_text() for text in _axis(figure, "route-summary").texts)
            assert "78.2%" in summary and "62.5%" in summary  # Supplied aggregate, not mean edge score.
            assert "Wheelchair required: YES" in summary and "nonnegotiable" in summary
            assert "Known conflicts: 4" in summary and "Unknown observations: 1" in summary
            assert "Top conflicts (3 of 4)" in summary and "Known concern 3" not in summary
            assert "route_summary.md" in summary and "Verbose policy" not in summary
            assert "Hard-block violations: 0" in summary and "A* cost = Dijkstra" in summary and "|Δ| = 0 m" in summary
            assert "150.0 m" in summary and "22.0%" in summary
            assert all(text.get_position()[1] >= 0 and text.get_fontsize() >= 9 for text in _axis(figure, "route-summary").texts)
            inset = _axis(figure, "area-context")
            assert _segments(_collection(inset, "graph-edges")) == [true[e.id] for e in graph.edges]
            assert inset.get_xlim() == ax.get_xlim() and inset.get_ylim() == ax.get_ylim()
        if name == "area_accessibility":
            assert "150 m route-union corridor" in caption and "Full diagnostics retained" in caption
            left, right = _axis(figure, "area-scores"), _axis(figure, "route-detail")
            foreground(left, "#1464b4")
            foreground(right, "#1464b4")
            assert left.get_xlim() == right.get_xlim() and left.get_ylim() == right.get_ylim()
            assert not _line(left, "selected-route").is_dashed()
            for ax, prefix, edges, geometry in ((left, "graph", graph.edges, shifted), (right, "path", route.edges, true)):
                collection = _collection(ax, f"{prefix}-score")
                active = [edge for edge in edges if not diagnostics[edge.id]["blocked"] and diagnostics[edge.id]["score"] is not None]
                assert _segments(collection) == [geometry[edge.id] for edge in active]
                assert list(collection.get_array()) == [diagnostics[edge.id]["score"] for edge in active]
                assert collection.norm.vmin == 0 and collection.norm.vmax == 100
                assert collection.cmap.name == "cividis"
                assert sum(len(artist.get_segments()) for artist in ax.collections
                           if isinstance(artist, LineCollection)) == len(edges)
                assert [artist.get_gid() for artist in ax.collections if isinstance(artist, Quiver)] == ["route-directions"]
            assert _collection(right, "path-score").norm is _collection(left, "graph-score").norm
            assert _collection(right, "path-score").get_zorder() > _line(right, "selected-route").get_zorder()
            blocked = _collection(left, "graph-blocked")
            assert _segments(blocked) == [shifted[e.id] for e in graph.edges if diagnostics[e.id]["blocked"]]
            assert all(dashes is not None for _, dashes in blocked.get_linestyles())
            assert tuple(blocked.get_colors()[0]) == pytest.approx(to_rgba("#c52b2f"))
            assert blocked.get_zorder() < _line(left, "route-halo").get_zorder()  # Blocked reverse cannot hide route.
            unscored = _collection(left, "graph-unscored")
            assert _segments(unscored) == [true["da-unscored"]]
            assert tuple(unscored.get_colors()[0]) == pytest.approx(to_rgba("#8c8c8c"))
            colour_axis = _axis(figure, "edge-score-scale")
            assert colour_axis.get_xlabel() == "Event-inclusive edge score (0-100)"
            assert [tick.get_text() for tick in colour_axis.get_xticklabels()] == ["0 (low)", "25", "50", "75", "100 (high)"]
            assert all(not artist.get_rasterized() for artist in colour_axis.collections)
        return original_save(figure, destination, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    paths = create_figures(*example, title="Manual directed graph", synthetic=True, output_dir=tmp_path, dpi=60,
                           presentation=presentation, file_formats=("png", "pdf"))
    assert paths == [tmp_path / f"{name}.{fmt}" for name in ("route_obstacles", "area_accessibility") for fmt in ("png", "pdf")]
    assert inspected == {"route_obstacles", "area_accessibility"}
    assert set(tmp_path.iterdir()) == set(paths)
    for path in paths:
        content = path.read_bytes()
        assert len(content) > 1000
        if path.suffix == ".png":
            assert content.startswith(b"\x89PNG\r\n\x1a\n")
            pixels = imread(path)
            assert min(pixels.shape[:2]) > 100
            assert pixels[..., :3].min() < 0.2 and pixels[..., :3].max() > 0.95
        else:
            assert content.startswith(b"%PDF-")
            assert content.rstrip().endswith(b"%%EOF")
            assert b"/Type /Page" in content
            assert b"/Subtype /Image" not in content  # Maps/text remain vector, not a raster screenshot.
    assert _input_snapshot(*example) == before
    assert presentation == before_presentation


def test_marker_priority_coalescing_crop_accounting_and_stability(example):
    graph, route, _, pois = example
    contexts = [_candidate(f"context-{i:03}", "crossing", node_id=str(i)) for i in range(300)]
    candidates = contexts + [
        _candidate("conflict", "surface", on_route=True, status="conflict", way_id="route-way", first_segment=1),
        _candidate("duplicate-conflict", "surface", on_route=True, status="conflict", way_id="route-way",
                   latitude=49.59031, first_segment=2),
        _candidate("hard", "barrier", status="blocked"),
        _candidate("encounter", "ramp", on_route=True, status="encountered", first_segment=2),
        _candidate("outside", latitude=49.7),
        _candidate("unknown-measurement", "width", status="unknown"),
    ]
    presentation = _presentation(route, candidates, limit=3)
    before = deepcopy(presentation)
    selected, counts, extent = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert [row["id"] for row in selected] == ["conflict", "hard", "encounter"]
    assert selected[0]["candidate_count"] == 2
    assert set(selected[0]["member_ids"]) == {"conflict", "duplicate-conflict"}
    assert [row["marker_label"] for row in selected] == ["1", "2", "3"]
    assert counts == {"total_candidates": 313, "selected_candidates": 306, "omitted_by_filter": 7,
                      "max_markers": 3, "cropped": 1, "unknown_suppressed": 1, "displayed": 3, "labelled": 3,
                      "coalesced": 1, "omitted_by_limit": 300, "suppressed": 302}
    assert counts["selected_candidates"] == counts["cropped"] + counts["displayed"] + counts["suppressed"]
    assert presentation == before
    reversed_view = dict(presentation, map_obstacles=list(reversed(candidates)))
    assert select_map_obstacles(graph, route, pois, presentation=reversed_view) == (selected, counts, extent)
    selected[0]["edge_ids"].append("display-only")
    assert presentation == before


def test_coalescing_never_moves_points_or_merges_different_types_ways_statuses(example):
    graph, route, _, pois = example
    candidates = [
        _candidate("a", way_id="same"), _candidate("a-near", way_id="same", latitude=49.59031),
        _candidate("a-far", way_id="same", latitude=49.5908),
        _candidate("b", way_id="different"), _candidate("c", "barrier", way_id="same"),
        _candidate("d", way_id="same", status="blocked"),
        _candidate("e", way_id="same", on_route=True, status="encountered"),
    ]
    presentation = _presentation(route, candidates)
    selected, counts, _ = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert counts["coalesced"] == 1 and counts["displayed"] == 6
    combined = next(row for row in selected if row["candidate_count"] == 2)
    assert combined["id"] in {"a", "a-near"}
    original = next(row for row in candidates if row["id"] == combined["id"])
    assert combined["latitude"] == original["latitude"] and combined["longitude"] == original["longitude"]
    assert len({row["kind"] for row in selected}) == 2


@pytest.mark.parametrize("limit", [0, 1, 12, 100])
def test_total_marker_limit_and_only_twelve_labels(example, tmp_path, monkeypatch, limit):
    graph, route, _, pois = example
    candidates = [_candidate(f"stairs-{i:03}", node_id=str(i)) for i in range(120)]
    presentation = _presentation(route, candidates, limit=limit)
    selected, counts, _ = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert len(selected) == limit
    assert counts["omitted_by_limit"] == 120 - limit
    assert [row["marker_label"] for row in selected[:12]] == [str(i) for i in range(1, min(limit, 12) + 1)]
    assert all(row["marker_label"] is None for row in selected[12:])

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            ax = _axis(figure, "route-focus")
            assert len([artist for artist in ax.collections if (artist.get_gid() or "").startswith("obstacle-")]) == limit
            assert len([text for text in ax.texts if text.get_gid() == "obstacle-label"]) == min(limit, 12)
            assert len([text for text in _axis(figure, "obstacle-key").texts if text.get_gid() == "obstacle-key"]) == min(limit, 12)
        else:
            assert sum(len(artist.get_segments()) for artist in _axis(figure, "area-scores").collections
                       if isinstance(artist, LineCollection)) == len(graph.edges)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(*example, presentation=presentation, title="Marker cap", synthetic=True, output_dir=tmp_path)


def test_legacy_fallback_filters_critical_features_and_reports_missing_checks(example, tmp_path, monkeypatch):
    graph, route, diagnostics, pois = example
    pois = [dict(poi, latitude=49.5904, longitude=11.0003) for poi in pois]
    pois += [dict(pois[0], id=f"stairs-{index:02}", osm_way_id=str(index)) for index in range(30)]
    selected, counts, extent = select_map_obstacles(graph, route, pois)
    assert len(selected) == 12
    assert {row["kind"] for row in selected} <= {"stairs", "barrier"}
    assert counts["omitted_by_filter"] == 6
    assert counts["selected_candidates"] == 32
    assert counts["max_markers"] == 12
    assert select_map_obstacles(graph, route, list(reversed(pois))) == (selected, counts, extent)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            text = _text(figure)
            assert "A*/Dijkstra: Not compared" in text and "Hard-block violations: Not compared" in text
            assert "Conflicts / unknown observations: not supplied" in text
        assert "© OpenStreetMap contributors, ODbL 1.0" in _text(figure)
        assert "SYNTHETIC DATA" not in _text(figure)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, pois, title="Legacy caller", synthetic=False, output_dir=tmp_path)


@pytest.mark.parametrize("kinds", [(), ("incline",)])
def test_actual_presentation_filter_respected_without_supplementing_pois_or_conflicts(example, tmp_path, monkeypatch, kinds):
    graph, route, _, _ = example
    profile, parameters = Profile(weights={"road_incline": 1, "lit_roads": .5}), CostParameters()
    diagnostics = edge_diagnostics(graph, route, Evaluator(profile, parameters))
    pois = collect_pois(graph, route, diagnostics)
    presentation = build_route_presentation(graph, route, diagnostics, profile, parameters,
                                           options=PresentationOptions(kinds, 12), synthetic=True)
    before = deepcopy(presentation), deepcopy(diagnostics)
    assert presentation["conflicts"] and presentation["unknowns"]
    selected, _, _ = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert {row["kind"] for row in selected} == set(kinds)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            markers = [artist.get_gid() for artist in _axis(figure, "route-focus").collections
                       if (artist.get_gid() or "").startswith("obstacle-")]
            assert len(markers) == len(selected)
            assert all(gid.startswith("obstacle-incline-") for gid in markers)
            assert "Unknown observations:" in _text(figure)
        else:
            assert sum(len(artist.get_segments()) for artist in _axis(figure, "area-scores").collections
                       if isinstance(artist, LineCollection)) == len(graph.edges)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, pois, presentation=presentation,
                   title="Builder contract", synthetic=True, output_dir=tmp_path)
    assert (presentation, diagnostics) == before


def test_all_maps_exclude_disconnected_island_but_keep_full_diagnostics(example, tmp_path, monkeypatch):
    original, route, diagnostics, pois = example
    far_a, far_b = Node("far-a", Coordinate(49.7, 11.1)), Node("far-b", Coordinate(49.71, 11.1))
    far_edge = Edge("far-edge", far_a.id, far_b.id, "far-way", _distance(far_a, far_b))
    graph = InMemoryGraph([*original.nodes.values(), far_a, far_b], [*original.edges, far_edge])
    diagnostics[far_edge.id] = {"score": 50., "blocked": False}
    positions = _project_graph(graph, route.nodes)[0]
    candidates = [_candidate("near"), _candidate("far", latitude=49.705, longitude=11.1)]
    presentation = _presentation(route, candidates, limit=1)
    selected, counts, extent = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert [row["id"] for row in selected] == ["near"] and counts["cropped"] == 1
    assert extent[1] - extent[0] < 700 and extent[3] - extent[2] < 700
    before = deepcopy(diagnostics)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            focus = _axis(figure, "route-focus")
            assert (*focus.get_xlim(), *focus.get_ylim()) == extent
            assert len(_collection(focus, "graph-edges").get_segments()) == len(original.edges)
            full = _axis(figure, "area-context")
        else:
            full = _axis(figure, "area-scores")
            assert sum(len(artist.get_segments()) for artist in full.collections
                       if isinstance(artist, LineCollection)) == len(original.edges)
        assert full.get_xlim() == extent[:2] and full.get_ylim() == extent[2:]
        for node in (far_a, far_b):
            x, y = positions[node.id]
            assert not (full.get_xlim()[0] < x < full.get_xlim()[1]
                        and full.get_ylim()[0] < y < full.get_ylim()[1])

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, pois, presentation=presentation, title="Disconnected context",
                   synthetic=True, output_dir=tmp_path)
    assert diagnostics == before and far_edge.id in diagnostics


def test_unknown_candidates_are_suppressed_not_drawn_as_obstacles(example, tmp_path, monkeypatch):
    graph, route, _, pois = example
    presentation = _presentation(route, [_candidate("unknown", "incline", status="unknown")])
    selected, counts, _ = select_map_obstacles(graph, route, pois, presentation=presentation)
    assert not selected and counts["unknown_suppressed"] == counts["suppressed"] == 1

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            assert not any((artist.get_gid() or "").startswith("obstacle-") for artist in _axis(figure, "route-focus").collections)
            assert "1 unknown" in _text(figure)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(*example, presentation=presentation, title="Unknown is not observed", synthetic=True, output_dir=tmp_path)


def test_narrow_score_range_is_not_rescaled_and_uses_supplied_score_not_way_score(example, tmp_path, monkeypatch):
    for record in example[2].values():
        if not record["blocked"]:
            record.update(score=63.25, way_score=99.0)
    scores = _validated_scores(example[0], example[2])
    assert scores["ab"].value == 63.25

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "area_accessibility":
            for label, prefix in (("area-scores", "graph"), ("route-detail", "path")):
                collection = _collection(_axis(figure, label), f"{prefix}-score")
                assert collection.norm.vmin == 0 and collection.norm.vmax == 100
                assert collection.cmap.name == "cividis"
                assert set(collection.get_array()) == {63.25}

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(*example, title="Fixed scale", synthetic=True, output_dir=tmp_path)


def test_only_sparse_route_arrows_and_all_segments_retained(tmp_path, monkeypatch):
    nodes = [Node(str(i), Coordinate(49.59, 11 + i * .00002)) for i in range(41)]
    edges = [Edge(str(i), a.id, b.id, "path", _distance(a, b)) for i, (a, b) in enumerate(zip(nodes, nodes[1:]))]
    graph = InMemoryGraph(nodes, edges)
    distance = sum(edge.distance_m for edge in edges)
    route = Route(tuple(node.id for node in nodes), tuple(edges), distance, distance)
    diagnostics = {edge.id: {"score": 40 + i % 10, "blocked": False} for i, edge in enumerate(edges)}
    positions = _project_graph(graph, route.nodes)[0]
    sampled = edges[::7]  # ceil(40/6); at most six arrows, not forty city-edge arrows.

    def inspect(figure, destination, **kwargs):
        for ax in figure.axes:
            if ax.get_label() not in {"route-focus", "area-scores", "route-detail"}:
                continue
            arrows = _collection(ax, "route-directions")
            assert len(arrows.U) == 6
            assert list(arrows.U) == pytest.approx([
                .22 * (positions[edge.target][0] - positions[edge.source][0]) for edge in sampled
            ])
            assert len(_line(ax, "selected-route").get_xdata()) == len(route.nodes)
            assert len([artist for artist in ax.collections if isinstance(artist, Quiver)]) == 1
        if Path(destination.name).stem == "area_accessibility":
            assert len(_collection(_axis(figure, "area-scores"), "graph-score").get_segments()) == 40
            assert len(_collection(_axis(figure, "route-detail"), "path-score").get_segments()) == 40

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, [], title="Long path", synthetic=True, output_dir=tmp_path)


@pytest.mark.parametrize("vertical", [False, True])
def test_two_km_corridor_renderer_readable_equal_scale_layout(tmp_path, monkeypatch, vertical):
    local = CRS.from_proj4("+proj=aeqd +lat_0=49.59 +lon_0=11 +datum=WGS84 +units=m +no_defs")
    inverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
    xy = {f"{x},{y}": (50. * x, 50. * y) for x in range(41) for y in range(-4, 5)}
    xy.update({"island-a": (9000., 9000.), "island-b": (9100., 9000.)})
    nodes = []
    for key, point in xy.items():
        lon, lat = inverse.transform(*(point[::-1] if vertical else point))
        nodes.append(Node(key, Coordinate(lat, lon)))
    edges = []
    for x in range(41):
        for y in range(-4, 5):
            a = f"{x},{y}"
            for b in (f"{x + 1},{y}", f"{x},{y + 1}"):
                if b in xy:
                    edges.extend(Edge(f"{s}>{t}", s, t, "grid", 50., {"highway": "footway"})
                                 for s, t in ((a, b), (b, a)))
    edges.append(Edge("island", "island-a", "island-b", "island", 100.))
    graph = InMemoryGraph(nodes, edges)
    by_id = {edge.id: edge for edge in edges}

    def trace(keys):
        path = tuple(by_id[f"{a}>{b}"] for a, b in zip(keys, keys[1:]))
        return Route(tuple(keys), path, len(path) * 50., len(path) * 50.)

    standard = trace([f"{x},0" for x in range(41)])
    route = trace(["0,0", *[f"{x},1" for x in range(41)], "40,0"])
    diagnostics = {edge.id: {"score": 100., "blocked": edge.id == "20,0>21,0"} for edge in edges}
    before = repr(graph.nodes), repr(graph.edges), repr(route), repr(standard), deepcopy(diagnostics)
    positions, _ = viz._project_graph(graph, [*route.nodes, *standard.nodes])
    paths = [[positions[key] for key in item.nodes] for item in (route, standard)]
    corridor = viz._corridor(graph, positions, paths, 150.)
    nearby = InMemoryGraph(nodes[:-2], edges[:-1])
    near_positions, _ = viz._project_graph(nearby, [*route.nodes, *standard.nodes])
    assert corridor.extent == viz._corridor(nearby, near_positions, paths, 150.).extent
    assert "island" not in corridor.segments
    assert min(corridor.extent[1] - corridor.extent[0], corridor.extent[3] - corridor.extent[2]) < 600
    seen = []

    def inspect(figure, destination, **kwargs):
        seen.append(Path(destination.name).stem)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if seen[-1] == "route_obstacles":
            ax = _axis(figure, "route-focus")
            assert (*ax.get_xlim(), *ax.get_ylim()) == corridor.extent
            assert ax.get_position().y1 == pytest.approx(ax.get_position(original=True).y1)
            assert ax.get_anchor() == "N"
            for gid, expected in (("selected-route", paths[0]), ("standard-route", paths[1])):
                line = _line(ax, gid)
                assert list(zip(line.get_xdata(), line.get_ydata())) == expected
            return
        selected, baseline = _axis(figure, "area-scores"), _axis(figure, "route-detail")
        a, b = selected.get_position(), baseline.get_position()
        assert a.width == pytest.approx(b.width) and a.height == pytest.approx(b.height)
        if vertical:
            assert a.x1 < b.x0 and a.y0 == pytest.approx(b.y0)
        else:
            assert a.y0 > b.y1 and a.x0 == pytest.approx(b.x0)
            assert a.width > .85  # More than twice the former .40-wide side-by-side pane.
            assert figure.get_size_inches()[1] < 10
        scales = []
        for ax, gid, expected in ((selected, "selected-route", paths[0]), (baseline, "standard-route", paths[1])):
            assert (*ax.get_xlim(), *ax.get_ylim()) == corridor.extent
            assert ax.get_aspect() == 1 and ax.get_adjustable() == "box"
            origin, east, north = ax.transData.transform([(0, 0), (1, 0), (0, 1)])
            scale_x, scale_y = east[0] - origin[0], north[1] - origin[1]
            assert scale_x == pytest.approx(scale_y)
            assert east[1] == origin[1] and north[0] == origin[0]  # No rotation/shear.
            scales.append(scale_x)
            line = _line(ax, gid)
            assert list(zip(line.get_xdata(), line.get_ydata())) == expected  # No invented connectors.
            for suffix in ("score", "blocked"):
                assert max(_collection(ax, "graph-" + suffix).get_linewidths()) <= 1.
            # Neither the distance of an island nor the score offset expands the original buffer.
            assert all(corridor.geometry.buffer(1e-7).covers(viz.LineString(segment))
                       for segment in _collection(ax, "graph-score").get_segments())
        assert scales[0] == pytest.approx(scales[1])
        assert _line(selected, "selected-route").get_linewidth() == 6
        assert max(_collection(selected, "path-score").get_linewidths()) == 3.2
        # Renderer-space bounds include tick labels, axis labels and pane titles.
        boxes = [ax.get_tightbbox(renderer) for ax in (selected, baseline, _axis(figure, "edge-score-scale"))]
        boxes += [text.get_window_extent(renderer) for text in figure.texts]
        boxes += [legend.get_window_extent(renderer) for legend in figure.legends]
        for index, box in enumerate(boxes):
            assert figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1)
            assert all(not box.overlaps(previous) for previous in boxes[:index])

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, [], standard_route=standard, corridor_radius_m=150.,
                   title="Two kilometre grid", synthetic=True, output_dir=tmp_path)
    assert seen == ["route_obstacles", "area_accessibility"]
    assert (repr(graph.nodes), repr(graph.edges), repr(route), repr(standard), diagnostics) == before


def test_supplied_failed_checks_remain_visible_and_synthetic_flag_cannot_be_hidden(example, tmp_path, monkeypatch):
    presentation = _presentation(example[1], [])
    presentation["validation"].update(hard_constraint_violations=1, astar_dijkstra_agree=False,
                                      optimal_cost_difference_m=2.5)
    presentation["requirements"][0] = "Wheelchair required: false. Separate from soft preferences."
    example[2]["bc"]["blocked"] = True

    def inspect(figure, destination, **kwargs):
        assert "SYNTHETIC DATA" in _text(figure)
        if Path(destination.name).stem == "route_obstacles":
            text = _text(figure)
            assert "Hard-block violations: 1" in text and "Hard-block violations: 0" not in text
            assert "A* cost ≠ Dijkstra" in text and "|Δ| = 2.5 m" in text
            assert "Wheelchair required: NO" in text

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(*example, presentation=presentation, title="Failed supplied checks", synthetic=False, output_dir=tmp_path)


def test_missing_off_path_diagnostics_fail_before_creating_files(example, tmp_path):
    graph, route, diagnostics, pois = example
    del diagnostics["cd-stairs"]
    with pytest.raises(ValueError, match="Missing diagnostics.*cd-stairs"):
        create_figures(graph, route, diagnostics, pois, title="Missing", synthetic=True, output_dir=tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("field,value", [
    ("score", -0.1), ("score", 100.1), ("score", float("nan")),
    ("score", float("inf")), ("score", float("-inf")), ("score", True),
    ("score", "75"), ("way_score", 101), ("coverage", float("nan")),
])
@pytest.mark.parametrize("edge_id", ["ab", "cd-stairs"])
def test_invalid_scores_fail_even_for_blocked_edges(example, tmp_path, field, value, edge_id):
    graph, route, diagnostics, pois = example
    diagnostics[edge_id][field] = value
    with pytest.raises(ValueError, match=field):
        create_figures(graph, route, diagnostics, pois, title="Invalid", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["blocked", "score"])
def test_incomplete_diagnostic_record_is_not_silently_grey(example, tmp_path, field):
    del example[2]["be"][field]
    with pytest.raises(ValueError, match="require blocked and score"):
        create_figures(*example, title="Incomplete", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_invalid_blocked_flag_is_not_truthiness_coerced(example, tmp_path):
    example[2]["be"]["blocked"] = 1
    with pytest.raises(ValueError, match="blocked.*boolean"):
        create_figures(*example, title="Invalid flag", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [
    ("latitude", 91), ("latitude", float("nan")), ("longitude", "11"), ("longitude", 181),
    ("id", ""), ("id", True), ("kind", "unrecognised"), ("on_route", 1),
    ("blocked", "no"), ("description", {}),
])
def test_invalid_legacy_pois_fail_even_if_they_would_be_cropped(example, tmp_path, field, value):
    example[3][0][field] = value
    with pytest.raises(ValueError, match="POI"):
        create_figures(*example, title="Invalid POI", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("limit", [True, -1, 101, 1.5, "12", None])
def test_invalid_marker_limits_fail_before_output(example, tmp_path, limit):
    presentation = _presentation(example[1], [], limit=limit)
    with pytest.raises(ValueError, match="max_markers"):
        create_figures(*example, presentation=presentation, title="Invalid cap", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [("status", "bogus"), ("latitude", float("inf")), ("on_route", None)])
def test_invalid_presentation_markers_are_not_silently_omitted(example, tmp_path, field, value):
    presentation = _presentation(example[1], [dict(_candidate("invalid"), **{field: value})], limit=0)
    with pytest.raises(ValueError):
        create_figures(*example, presentation=presentation, title="Invalid candidate", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_route_must_still_belong_to_supplied_graph(example, tmp_path):
    graph, route, diagnostics, pois = example
    first = route.edges[0]
    other = Edge(first.id, first.source, first.target, "wrong-way", first.distance_m)
    changed = Route(route.nodes, (other, *route.edges[1:]), route.distance_m, route.cost)
    with pytest.raises(ValueError, match="does not match the graph"):
        create_figures(graph, changed, diagnostics, pois, title="Wrong route", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_later_artifact_is_never_overwritten(example, tmp_path):
    existing = tmp_path / "area_accessibility.pdf"
    existing.write_bytes(b"preserve exactly")
    with pytest.raises(FileExistsError, match="area_accessibility.pdf"):
        create_figures(*example, title="Collision", synthetic=True, output_dir=tmp_path, file_formats=("png", "pdf"))
    assert existing.read_bytes() == b"preserve exactly"
    assert list(tmp_path.iterdir()) == [existing]


def test_exclusive_creation_handles_race_without_overwrite(example, tmp_path, monkeypatch):
    original_open = Path.open
    raced = tmp_path / "area_accessibility.pdf"

    def race(path, mode="r", *args, **kwargs):
        if path == raced and mode == "xb":
            with original_open(path, "wb") as stream:
                stream.write(b"concurrent plot")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        create_figures(*example, title="Concurrent creation", synthetic=True, output_dir=tmp_path, file_formats=("png", "pdf"))
    assert raced.read_bytes() == b"concurrent plot"
    assert list(tmp_path.iterdir()) == [raced]


@pytest.mark.parametrize("fail_at", [1, 3])
def test_failed_write_removes_only_its_new_artifacts(example, tmp_path, monkeypatch, fail_at):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    figures = []

    def fail(figure, destination, **kwargs):
        figures.append(figure)
        destination.write(b"partial")
        if len(figures) == fail_at:
            raise OSError("simulated write failure")

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        create_figures(*example, title="Failure", synthetic=True, output_dir=tmp_path, file_formats=("png", "pdf"))
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert all(not figure.axes for figure in figures)


def test_dangling_symlink_is_never_overwritten(example, tmp_path):
    target = tmp_path / "area_accessibility.png"
    try:
        target.symlink_to(tmp_path / "missing.png")
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks requires platform support / permission")
    with pytest.raises(FileExistsError, match="area_accessibility.png"):
        create_figures(*example, title="Symlink collision", synthetic=True, output_dir=tmp_path)
    assert target.is_symlink() and not target.exists()
    assert list(tmp_path.iterdir()) == [target]


def test_no_active_criteria_and_blocked_selected_edge_styles(example, tmp_path, monkeypatch):
    graph, route, diagnostics, pois = example
    for record in diagnostics.values():
        record.update(score=None, way_score=None, coverage=None)
    diagnostics["bc"].update(blocked=True, hard_blocks=["selected edge restriction"], score=0.0)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "area_accessibility":
            for label, prefix in (("area-scores", "graph"), ("route-detail", "path")):
                ax = _axis(figure, label)
                assert not any((artist.get_gid() or "").endswith("-score") for artist in ax.collections)
                assert tuple(_collection(ax, f"{prefix}-unscored").get_colors()[0]) == pytest.approx(to_rgba("#8c8c8c"))
            ax = _axis(figure, "route-detail")
            assert len(_collection(ax, "path-unscored").get_segments()) == 1
            assert len(_collection(ax, "path-blocked").get_segments()) == 1
            assert all(dashes is not None for _, dashes in _collection(ax, "path-blocked").get_linestyles())

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, pois, title="No active scores", synthetic=True,
                   output_dir=tmp_path, file_formats=("pdf",))


def test_single_node_route_and_zero_length_edges(tmp_path, monkeypatch):
    node = Node("a", Coordinate(49.59, 11.0))
    edge = Edge("aa", "a", "a", "point", 0)
    graph = InMemoryGraph([node], [edge])
    route = Route(("a",), (), 0, 0)
    diagnostics = {"aa": {"blocked": False, "score": None}}

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "route_obstacles":
            ax = _axis(figure, "route-focus")
            assert "Start / end" in [text.get_text() for text in ax.texts]
            assert ax.get_xlim()[1] - ax.get_xlim()[0] >= 10
        if Path(destination.name).stem == "area_accessibility":
            assert isinstance(_collection(_axis(figure, "area-scores"), "graph-unscored-points"), PathCollection)
        assert not any(isinstance(artist, Quiver) for ax in figure.axes for artist in ax.collections)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, [], title="Stationary", synthetic=True,
                   output_dir=tmp_path, file_formats=("png",))


@pytest.mark.parametrize("formats", [(), ("svg",), ("png", "png"), ["png"]])
def test_invalid_formats_have_no_output(example, tmp_path, formats):
    with pytest.raises(ValueError, match="file_formats"):
        create_figures(*example, title="Formats", synthetic=True, output_dir=tmp_path, file_formats=formats)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [{"dpi": 0}, {"dpi": True}, {"dpi": 1.5}, {"title": None}, {"synthetic": 1}])
def test_invalid_output_options_leave_no_files(example, tmp_path, kwargs):
    options = {"title": "Invalid options", "synthetic": True, "output_dir": tmp_path} | kwargs
    with pytest.raises(ValueError):
        create_figures(*example, **options)
    assert not list(tmp_path.iterdir())


def test_explicit_format_order_preserved(example, tmp_path, monkeypatch):
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    paths = create_figures(*example, title="Format order", synthetic=True, output_dir=tmp_path, file_formats=("pdf", "png"))
    assert paths == [tmp_path / f"{name}.{fmt}" for name in ("route_obstacles", "area_accessibility") for fmt in ("pdf", "png")]
