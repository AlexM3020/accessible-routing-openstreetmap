"""Metric clipping, route comparison and indexed-distance regression contracts."""

from copy import deepcopy
from math import hypot
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point

from osm_accessibility import visualization as viz
from osm_accessibility.evaluation import Evaluator
from osm_accessibility.graph import InMemoryGraph
from osm_accessibility.models import Coordinate, Edge, Node, Route
from osm_accessibility.outputs import (
    OBSTACLE_LABELS,
    PREFERENCE_LABELS,
    PresentationOptions,
    build_route_comparison,
    build_route_presentation,
    edge_diagnostics,
)
from osm_accessibility.profiles import CostParameters, Profile


def _transforms():
    local = CRS.from_proj4("+proj=aeqd +lat_0=49.59 +lon_0=11 +datum=WGS84 +units=m +no_defs")
    return (Transformer.from_crs("EPSG:4326", local, always_xy=True),
            Transformer.from_crs(local, "EPSG:4326", always_xy=True))


@pytest.fixture
def divergent():
    _, inverse = _transforms()
    coordinates = {"a": (0, 0), "b": (0, 600), "c": (800, 600), "d": (800, 0),
                   "cross-a": (400, 200), "cross-b": (400, 1000),
                   "gap-a": (350, 300), "gap-b": (450, 300),
                   "island-a": (9000, 9000), "island-b": (9100, 9000)}
    nodes = []
    for key, xy in coordinates.items():
        lon, lat = inverse.transform(*xy)
        nodes.append(Node(key, Coordinate(lat, lon)))
    specifications = [("ab", "a", "b"), ("bc", "b", "c"), ("cd", "c", "d"),
                      ("ad", "a", "d"), ("cross", "cross-a", "cross-b"),
                      ("gap", "gap-a", "gap-b"), ("island", "island-a", "island-b")]
    edges = [Edge(key, a, b, key, hypot(coordinates[a][0] - coordinates[b][0],
                                      coordinates[a][1] - coordinates[b][1]),
                  {"highway": "steps", "wheelchair": "no"} if key == "ad" else {"highway": "footway"})
             for key, a, b in specifications]
    graph = InMemoryGraph(nodes, edges)
    route = Route(("a", "b", "c", "d"), tuple(edges[:3]), 2000., 2000.)
    standard = Route(("a", "d"), (edges[3],), 800., 800.)
    profile = Profile(weights={"avoid_stairs": 1.0}, wheelchair_required=True)
    parameters = CostParameters()
    diagnostics = edge_diagnostics(graph, route, Evaluator(profile, parameters))
    view = build_route_presentation(graph, route, diagnostics, profile, parameters,
                                    options=PresentationOptions(None, 12), synthetic=True)
    view["comparison"] = build_route_comparison(graph, route, standard, diagnostics, profile, parameters,
                                                 selected_presentation=view)
    return graph, route, standard, diagnostics, view


def _axis(figure, label):
    return next(ax for ax in figure.axes if ax.get_label() == label)


def _collection(ax, gid):
    return next(item for item in ax.collections if item.get_gid() == gid)


def _line(ax, gid):
    return next(item for item in ax.lines if item.get_gid() == gid)


def test_corridor_union_crossing_clip_radius_and_no_mutation(divergent):
    graph, route, standard, diagnostics, view = divergent
    before = repr(graph.nodes), repr(graph.edges), repr(route), repr(standard), deepcopy(diagnostics), deepcopy(view)
    positions, _ = viz._project_graph(graph, [*route.nodes, *standard.nodes])
    paths = [[positions[key] for key in trace.nodes] for trace in (route, standard)]
    corridor = viz._corridor(graph, positions, paths, 100.)
    assert set(corridor.segments) == {"ab", "bc", "cd", "ad", "cross"}
    crossing = next(edge for edge in graph.edges if edge.id == "cross")
    assert all(not corridor.geometry.covers(Point(positions[key])) for key in (crossing.source, crossing.target))
    pieces = corridor.segments["cross"]
    assert sum(LineString(segment).length for segment in pieces) == pytest.approx(200., abs=.02)
    assert all(corridor.geometry.buffer(1e-8).covers(LineString(segment)) for segment in pieces)
    for edge in (*route.edges, *standard.edges):
        assert corridor.segments[edge.id] == [(positions[edge.source], positions[edge.target])]
    wide = viz._corridor(graph, positions, paths, 350.)
    assert "gap" in wide.segments and "island" not in wide.segments
    assert wide.extent[1] - wide.extent[0] > corridor.extent[1] - corridor.extent[0]
    selected_only = viz._corridor(graph, positions, paths[:1], 100.)
    assert sum(LineString(part).length for part in selected_only.segments["ad"]) < 201
    assert sum(LineString(part).length for part in corridor.segments["ad"]) == pytest.approx(800., abs=.02)
    assert (repr(graph.nodes), repr(graph.edges), repr(route), repr(standard), diagnostics, view) == before


def test_no_fictitious_connector_in_corridor_or_nearest_distance():
    paths = [[(0., 0.), (10., 0.)], [(100., 100.), (110., 100.)]]
    middle = (55., 50.)
    expected = min(viz._distance_to_path(middle, path) for path in paths)
    assert viz._indexed_path_distances([middle], paths) == pytest.approx([expected])
    assert expected > 60  # A concatenated path would put the point on its invented connector.
    nodes = [Node("x", Coordinate(0, 0)), Node("y", Coordinate(0, .001))]
    graph = InMemoryGraph(nodes, [Edge("gap", "x", "y", "gap", 112.)])
    corridor = viz._corridor(graph, {"x": (54., 50.), "y": (56., 50.)}, paths, 5.)
    assert not corridor.segments
    assert not corridor.geometry.covers(Point(middle))


@pytest.mark.parametrize("ratio", [.25, viz._ROUTE_RATIO, 4.])
@pytest.mark.parametrize("vertical", [False, True])
def test_corridor_framing_pads_each_original_dimension_without_aspect_expansion(ratio, vertical):
    graph = InMemoryGraph([Node("a", Coordinate(0, 0)), Node("b", Coordinate(0, .01))],
                           [Edge("ab", "a", "b", "w", 2000.)])
    positions = {"a": (0., 0.), "b": (0., 2000.) if vertical else (2000., 0.)}
    corridor = viz._corridor(graph, positions, [list(positions.values())], 150., ratio)
    xmin, ymin, xmax, ymax = corridor.geometry.bounds
    x0, x1, y0, y1 = corridor.extent
    # Padding follows each buffer dimension, never the longest axis or a pane ratio.
    assert 0 < xmin - x0 <= (xmax - xmin) * .06
    assert 0 < ymin - y0 <= (ymax - ymin) * .06
    assert x1 - xmax == pytest.approx(xmin - x0)
    assert y1 - ymax == pytest.approx(ymin - y0)
    assert min(x1 - x0, y1 - y0) < 600
    assert corridor.segments == {"ab": [(positions["a"], positions["b"])]}
    assert corridor.extent == viz._corridor(graph, positions, [list(positions.values())], 150.).extent


def test_indexed_distances_equal_brute_segments_including_stationary_paths():
    paths = [[(0., 0.), (1000., 0.), (1000., 0.), (1000., 900.)], [(400., 400.)]]
    candidates = [(500., 1.), (400., 400.), (900., 850.), (999., 300.), (-5., -6.)]
    expected = [min(viz._distance_to_path(point, path) for path in paths) for point in candidates]
    assert viz._indexed_path_distances(candidates, paths) == pytest.approx(expected, abs=1e-10)


def test_thousands_of_candidates_batch_segment_index_and_zero_cap_skip(monkeypatch):
    transformer, inverse = _transforms()
    path = [(i * 2.5, 0.) for i in range(801)]  # 2 km, 800 segments.
    candidates = []
    for index in range(4000):
        owner = index % 2000
        lon, lat = inverse.transform(float(owner), float(owner % 7 + 1))
        candidates.append({"id": str(index), "kind": "stairs", "latitude": lat, "longitude": lon,
                           "on_route": False, "status": "context", "description": "Mapped stairs",
                           "node_id": str(owner), "way_id": None, "edge_ids": [], "preference_labels": []})
    presentation = {"map_obstacles": candidates, "options": {"max_markers": 12}}
    real_tree = viz.STRtree
    calls = []

    class ObservedTree:
        def __init__(self, geometries):
            self.tree = real_tree(geometries)
            assert len(geometries) == 800

        def query_nearest(self, geometry, **kwargs):
            calls.append((len(geometry), kwargs))
            return self.tree.query_nearest(geometry, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Per-candidate linear path scan / nearest work at cap zero")

    monkeypatch.setattr(viz, "STRtree", ObservedTree)
    monkeypatch.setattr(viz, "_distance_to_path", forbidden)
    rows, counts, extent = viz._select_obstacles([], presentation, transformer, path)
    assert calls == [(4000, {"return_distance": True, "all_matches": False})]
    assert counts["selected_candidates"] == 4000 and counts["coalesced"] == 2000
    assert counts["displayed"] == 12 and counts["omitted_by_limit"] == 1988
    assert all(row["candidate_count"] == 2 for row in rows)
    reverse = dict(presentation, map_obstacles=list(reversed(candidates)))
    assert viz._select_obstacles([], reverse, transformer, path) == (rows, counts, extent)
    monkeypatch.setattr(viz, "_indexed_path_distances", forbidden)
    zero = dict(presentation, options={"max_markers": 0})
    hidden, hidden_counts, _ = viz._select_obstacles([], zero, transformer, path)
    assert hidden == [] and hidden_counts["coalesced"] == counts["coalesced"]
    assert hidden_counts["omitted_by_limit"] == 2000 and hidden_counts["displayed"] == 0
    assert hidden_counts["selected_candidates"] == counts["selected_candidates"]


@pytest.mark.parametrize("radius", [0, -1, 5000.01, float("nan"), float("inf"), True, "150", None])
def test_invalid_radius_rejected_without_outputs(divergent, tmp_path, radius):
    graph, route, standard, diagnostics, view = divergent
    with pytest.raises(ValueError, match="corridor_radius_m"):
        viz.create_figures(graph, route, diagnostics, [], presentation=view, standard_route=standard,
                           corridor_radius_m=radius, title="Invalid", synthetic=True, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_two_panel_same_extent_scores_exact_traces_and_full_comparison(divergent, tmp_path, monkeypatch):
    graph, route, standard, diagnostics, view = divergent
    before = deepcopy(view), deepcopy(diagnostics), repr(route), repr(standard)
    positions, _ = viz._project_graph(graph, [*route.nodes, *standard.nodes])
    corridor = viz._corridor(graph, positions,
                             [[positions[key] for key in trace.nodes] for trace in (route, standard)], 100.)
    captured = []

    def check(figure, destination, **kwargs):
        captured.append(Path(destination.name).stem)
        if captured[-1] == "route_obstacles":
            ax = _axis(figure, "route-focus")
            for gid, trace in (("selected-route", route), ("standard-route", standard), ("standard-halo", standard)):
                line = _line(ax, gid)
                assert list(zip(line.get_xdata(), line.get_ydata())) == [positions[key] for key in trace.nodes]
            assert not _line(ax, "selected-route").is_dashed()
            assert _line(ax, "standard-route").is_dashed()
            table = next(iter(_axis(figure, "route-summary").tables))
            cells = [cell.get_text().get_text() for cell in table.get_celld().values()]
            assert "Infeasible" in cells and "Stairs" in cells and "Missing information" in cells
            assert str(view["comparison"]["standard"]["obstacle_counts"]["stairs"]) in cells
            key = " ".join(text.get_text() for text in _axis(figure, "obstacle-key").texts)
            assert "[N]" in key
            text = " ".join(t.get_text() for axis in figure.axes for t in axis.texts)
            assert "avoided" not in text.lower()
        else:
            left, right = _axis(figure, "area-scores"), _axis(figure, "route-detail")
            assert (*left.get_xlim(), *left.get_ylim()) == corridor.extent
            assert left.get_xlim() == right.get_xlim() and left.get_ylim() == right.get_ylim()
            assert "infeasible" in right.get_title()
            for ax in (left, right):
                collection = _collection(ax, "graph-score")
                expected = [diagnostics[edge.id]["score"] for edge in graph.edges
                            if not diagnostics[edge.id]["blocked"] and diagnostics[edge.id]["score"] is not None
                            for _ in corridor.segments.get(edge.id, [])]
                assert list(collection.get_array()) == expected
                assert all(corridor.geometry.buffer(1e-8).covers(LineString(segment))
                           for segment in collection.get_segments())
            assert _collection(left, "graph-score").norm is _collection(right, "graph-score").norm
            assert _collection(left, "path-score").norm is _collection(left, "graph-score").norm
            path_scores = _collection(left, "path-score")
            assert list(path_scores.get_array()) == [diagnostics[edge.id]["score"] for edge in route.edges]
            assert [tuple(map(tuple, segment)) for segment in path_scores.get_segments()] == [
                (positions[edge.source], positions[edge.target]) for edge in route.edges]
            arrows = _collection(right, "standard-directions")
            assert list(arrows.U) == pytest.approx([.22 * (positions["d"][0] - positions["a"][0])])
            assert len(_collection(right, "standard-blocked").get_segments()) == 1
            assert all(dashes for _, dashes in _collection(right, "standard-blocked").get_linestyles())
            for ax, gid, trace in ((left, "selected-route", route), (right, "standard-route", standard)):
                line = _line(ax, gid)
                assert list(zip(line.get_xdata(), line.get_ydata())) == [positions[key] for key in trace.nodes]

    monkeypatch.setattr(Figure, "savefig", check)
    viz.create_figures(graph, route, diagnostics, [], standard_route=standard, presentation=view,
                       corridor_radius_m=100., title="Comparison", synthetic=True, output_dir=tmp_path)
    assert captured == ["route_obstacles", "area_accessibility"]
    assert (view, diagnostics, repr(route), repr(standard)) == before


@pytest.mark.parametrize("kinds,limit", [((), 12), (("stairs",), 1), (None, 0)])
def test_standard_markers_obey_type_filters_shared_cap_and_preserve_counts(divergent, kinds, limit):
    graph, route, standard, _, view = divergent
    view["options"] = {"obstacle_kinds": kinds, "max_markers": limit}
    view["map_obstacles"] = [row for row in view["map_obstacles"] if kinds is None or row["kind"] in kinds]
    view["filter_summary"] = {"selected_candidates": len(view["map_obstacles"]),
                               "total_candidates": len(view["map_obstacles"]), "omitted_by_filter": 0}
    before = deepcopy(view)
    rows, counts, _ = viz.select_map_obstacles(graph, route, [], standard_route=standard, presentation=view)
    assert len(rows) <= limit
    if kinds == () or limit == 0:
        assert not rows
    else:
        assert rows and {row["kind"] for row in rows} == {"stairs"}
        assert all(row["route_affiliation"] == "standard" for row in rows)
    assert counts["selected_candidates"] == counts["cropped"] + counts["displayed"] + counts["suppressed"]
    assert view == before and view["comparison"]["standard"]["obstacle_counts"]["stairs"] == 1


def test_same_path_label_shared_geometry_and_missing_comparison_not_fabricated(divergent, tmp_path, monkeypatch):
    graph, route, _, diagnostics, _ = divergent
    def check(figure, destination, **kwargs):
        titles = " ".join(ax.get_title() for ax in figure.axes)
        assert "same path" in titles
        if Path(destination.name).stem == "route_obstacles":
            ax = _axis(figure, "route-focus")
            selected, standard = _line(ax, "selected-route"), _line(ax, "standard-route")
            assert list(selected.get_xdata()) == list(standard.get_xdata())
            assert list(selected.get_ydata()) == list(standard.get_ydata())
            assert standard.get_zorder() > selected.get_zorder()
            text = " ".join(t.get_text() for t in _axis(figure, "route-summary").texts)
            assert "not supplied" in text and "No obstacle counts or feasibility inferred" in text
    monkeypatch.setattr(Figure, "savefig", check)
    viz.create_figures(graph, route, diagnostics, [], standard_route=route,
                       title="Same trace", synthetic=True, output_dir=tmp_path)


def test_island_does_not_change_projection_or_map_extent(divergent):
    graph, route, standard, _, _ = divergent
    local = InMemoryGraph([node for node in graph.nodes.values() if not node.id.startswith("island")],
                           [edge for edge in graph.edges if edge.id != "island"])
    extents = []
    for source in (graph, local):
        positions, _ = viz._project_graph(source, [*route.nodes, *standard.nodes])
        extents.append(viz._corridor(source, positions,
                                     [[positions[key] for key in trace.nodes] for trace in (route, standard)], 150.).extent)
    assert extents[0] == extents[1]


def test_auto_standard_physical_marker_preserves_full_counts_and_details(divergent):
    graph, route, standard, _, view = divergent
    obstacles = view["comparison"]["standard"]["obstacles"]
    stairs = next(row for row in obstacles if row["kind"] == "stairs")
    wheelchair = next(row for row in obstacles if row["kind"] == "wheelchair_restriction")
    stairs["status"] = "encountered"
    stairs["preference_labels"] = ["Avoid stairs"]
    wheelchair["status"] = "blocked"
    wheelchair["preference_labels"] = ["Wheelchair access"]
    before = deepcopy(view)
    rows, counts, _ = viz.select_map_obstacles(graph, route, [], standard_route=standard, presentation=view)
    combined = [row for row in rows if row.get("route_affiliation") == "standard"]
    assert len(combined) == 1
    marker = combined[0]
    assert marker["kind"] == "stairs" and marker["status"] == "blocked"
    assert set(marker["preference_labels"]) == {"Avoid stairs", "Wheelchair access"}
    assert stairs["description"] in marker["description"] and wheelchair["description"] in marker["description"]
    assert marker["candidate_count"] == 2 and counts["coalesced"] == 1
    assert "wheelchair restriction" in marker["display_label"]
    assert counts["selected_candidates"] == counts["cropped"] + counts["displayed"] + counts["suppressed"]
    assert view == before
    explicit = deepcopy(view)
    explicit["options"]["obstacle_kinds"] = ["stairs", "wheelchair_restriction"]
    rows, _, _ = viz.select_map_obstacles(graph, route, [], standard_route=standard, presentation=explicit)
    assert {row["kind"] for row in rows} == {"stairs", "wheelchair_restriction"}


@pytest.mark.parametrize("change", ["edge_ids", "way_id", "node_id", "first_segment", "longitude"])
def test_auto_never_combines_different_owners_directions_or_visits(divergent, change):
    graph, route, standard, _, view = divergent
    wheelchair = next(row for row in view["comparison"]["standard"]["obstacles"]
                      if row["kind"] == "wheelchair_restriction")
    wheelchair[change] = {"edge_ids": ["reverse-ad"], "way_id": "other", "node_id": "other",
                          "first_segment": 8, "longitude": wheelchair["longitude"] + .000001}[change]
    rows, _, _ = viz.select_map_obstacles(graph, route, [], standard_route=standard, presentation=view)
    standard_rows = [row for row in rows if row["route_affiliation"] == "standard"]
    assert len(standard_rows) == 2 and all(row["candidate_count"] == 1 for row in standard_rows)


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_marker_requires_same_directed_identity_and_preserves_revisits(divergent, reverse):
    graph, route, standard, _, view = divergent
    original = next(row for row in view["comparison"]["standard"]["obstacles"] if row["kind"] == "stairs")
    view["options"]["obstacle_kinds"] = ["stairs"]
    selected = deepcopy(original)
    if reverse:
        selected["edge_ids"] = ["reverse-ad"]
    view["map_obstacles"] = [selected]
    view["filter_summary"] = {"selected_candidates": 1, "total_candidates": 1, "omitted_by_filter": 0}
    repeated = dict(original, id="second-visit", first_segment=9)
    view["comparison"]["standard"]["obstacles"].append(repeated)
    before = deepcopy(view)
    rows, counts, _ = viz.select_map_obstacles(graph, route, [], standard_route=standard, presentation=view)
    assert len(rows) == (2 if reverse else 1)
    assert {row["route_affiliation"] for row in rows} == ({"selected", "standard"} if reverse else {"both"})
    assert counts["selected_candidates"] == 3
    assert sum(row["candidate_count"] for row in rows) == 3
    assert view == before  # Nonconsecutive report encounters remain separate.


def test_offset_directions_keep_values_original_selection_and_clip():
    # Metric positions below are deliberately supplied independently of projection.
    nodes = [Node(str(i), Coordinate(0, 0)) for i in range(8)]
    edges = [Edge("forward", "0", "1", "w", 30.), Edge("reverse", "1", "0", "w", 30.),
             Edge("outside-a", "2", "3", "outside", 30.), Edge("outside-b", "3", "2", "outside", 30.),
             Edge("unique", "4", "5", "unique", 1.), Edge("point-a", "6", "6", "point", 0.),
             Edge("point-b", "6", "6", "point", 0.)]
    graph = InMemoryGraph(nodes, edges)
    positions = {"0": (-20., -2.), "1": (20., -2.), "2": (-20., 2.1), "3": (20., 2.1),
                 "4": (-1., 0.), "5": (1., 0.), "6": (0., 0.), "7": (100., 100.)}
    corridor = viz._corridor(graph, positions, [[(-10., 0.), (10., 0.)]], 2.)
    before = deepcopy(positions), repr(graph.edges), deepcopy(corridor.segments)
    shifted = viz._corridor_scores(graph, positions, corridor)
    assert set(shifted) == set(corridor.segments)
    assert "outside-a" not in shifted and "outside-b" not in shifted
    assert shifted["unique"] is corridor.segments["unique"]
    assert shifted["forward"] != shifted["reverse"]
    assert not shifted["forward"]  # Its outward offset is clipped away, not a fabricated edge.
    assert all(corridor.geometry.buffer(1e-9).covers(LineString(part))
               for pieces in shifted.values() for part in pieces)
    assert (positions, repr(graph.edges), corridor.segments) == before
    # With a wider corridor both reciprocal scores must be drawn distinctly.
    corridor = viz._corridor(graph, positions, [[(-10., 0.), (10., 0.)]], 4.)
    shifted = viz._corridor_scores(graph, positions, corridor)
    fig = Figure()
    ax = fig.add_subplot()
    scores = {edge.id: viz._Score(False, 20. if edge.id == "forward" else 85.) for edge in edges}
    viz._score_lines(ax, edges, viz._true_segments(edges, positions), scores,
                     viz.ScalarMappable(norm=viz.Normalize(0, 100), cmap="cividis"), "graph", clipped=shifted)
    collection = _collection(ax, "graph-score")
    assert set(collection.get_array()) == {20., 85.}
    assert len(set(tuple(map(tuple, part)) for part in collection.get_segments()[:2])) == 2
    assert all(corridor.geometry.buffer(1e-9).covers(LineString(part)) for part in collection.get_segments())


@pytest.mark.parametrize("dense", [False, True])
def test_rendered_comparison_bounds_full_preferences_table_footer_and_key(divergent, tmp_path, monkeypatch, dense):
    graph, route, standard, diagnostics, view = divergent
    view["selected_preferences"] = [{"name": name, "label": label, "weight": (i + 1) / 10}
                                    for i, (name, label) in enumerate(PREFERENCE_LABELS.items())]
    if dense:
        view["comparison"]["obstacle_rows"] = [
            {"kind": kind, "label": label, "selected": 2, "standard": 1}
            for kind, label in OBSTACLE_LABELS.items()]
        source = view["comparison"]["standard"]["obstacles"][0]
        view["comparison"]["standard"]["obstacles"] = [
            dict(source, id=f"key-{i}", kind=kind, node_id=f"owner-{i}", description=label)
            for i, (kind, label) in enumerate(OBSTACLE_LABELS.items())]
        view["options"]["obstacle_kinds"] = list(OBSTACLE_LABELS)
    before = deepcopy(view)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem != "route_obstacles":
            return
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        summary, key = _axis(figure, "route-summary"), _axis(figure, "obstacle-key")
        assert tuple(figure.get_size_inches()) == (16, 12 if dense else 10)
        preferences = [text for text in summary.texts if text.get_gid() == "comparison-preference"]
        assert len(preferences) == 10
        for text, item in zip(preferences, view["selected_preferences"]):
            assert text.get_text() == f"{item['label']}  {item['weight']:g}"
        table = next(iter(summary.tables))
        texts = [*summary.texts, *key.texts]
        boxes = [text.get_window_extent(renderer) for text in texts]
        table_box = table.get_window_extent(renderer)
        # Real renderer extents, not merely anchor positions: catches the old
        # multiline footer / affiliation overlap and long preference clipping.
        for index, box in enumerate(boxes):
            assert figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1), texts[index].get_text()
            assert not box.overlaps(table_box), texts[index].get_text()
            for previous in range(index):
                assert not box.overlaps(boxes[previous]), (texts[index].get_text(), texts[previous].get_text())
        for cell in table.get_celld().values():
            box = cell.get_text().get_window_extent(renderer)
            bounds = cell.get_window_extent(renderer)
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), cell.get_text().get_text()
        labels = [text for text in key.texts if text.get_gid() == "obstacle-key"]
        if dense:
            assert len(labels) == 12
            assert len({text.get_position()[0] for text in labels}) == 2
        content = " ".join(text.get_text() for text in summary.texts)
        assert "STANDARD INFEASIBLE" in content and "hard rules as well as weights" in content
        assert "wheelchair hard requirement" in content and "route_summary.md" in content
        assert "Handrail preferred:" in content and "Minimum width:" in content
        cells = table.get_celld()
        assert cells[0, 3].get_text().get_text() == "Change"
        assert cells[1, 3].get_text().get_text() == "+1200.0"
        assert cells[1, 3].get_text().get_color() == viz._RED
        arrows = _collection(_axis(figure, "route-focus"), "standard-directions")
        assert len(arrows.U) == 1

    monkeypatch.setattr(Figure, "savefig", inspect)
    viz.create_figures(graph, route, diagnostics, [], standard_route=standard, presentation=view,
                       title="Layout stress", synthetic=True, output_dir=tmp_path)
    assert view == before


def test_no_active_comparison_score_is_not_missing_input(divergent):
    _, route, standard, _, view = divergent
    view["selected_preferences"] = []
    for stats in (view["comparison"]["selected"], view["comparison"]["standard"]):
        stats.update(score_percent=None, coverage_percent=None, hard_block_violations=0)
    fig = Figure(figsize=(16, 10))
    viz.FigureCanvasAgg(fig)
    ax = fig.add_axes((.64, .2, .34, .7))
    viz._comparison_panel(ax, view, route, standard)
    cells = next(iter(ax.tables)).get_celld()
    assert cells[3, 1].get_text().get_text() == "No active score"
    assert cells[3, 2].get_text().get_text() == "No active score"
    assert cells[3, 3].get_text().get_text() == "—"
    assert cells[4, 1].get_text().get_text() == "No active score"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for cell in cells.values():
        box, bounds = cell.get_text().get_window_extent(renderer), cell.get_window_extent(renderer)
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)


def test_empty_and_dense_offset_layers_do_not_rescan_paths(monkeypatch):
    nodes = [Node(str(i), Coordinate(0, 0)) for i in range(601)]
    edges = [Edge(f"{i}-{suffix}", str(a), str(b), str(i), 2.)
             for i in range(600) for suffix, a, b in (("f", i, i + 1), ("r", i + 1, i))]
    graph = InMemoryGraph(nodes, edges)
    positions = {str(i): (i * 2., 0.) for i in range(601)}
    corridor = viz._corridor(graph, positions, [[(0., 0.), (1200., 0.)]], 5.)
    def forbidden(*args, **kwargs):
        pytest.fail("Offset rendering must not rescan paths or rebuild the corridor")
    monkeypatch.setattr(viz, "_corridor", forbidden)
    monkeypatch.setattr(viz, "_distance_to_path", forbidden)
    shifted = viz._corridor_scores(graph, positions, corridor)
    assert len(shifted) == 1200
    assert sum(len(parts) for parts in shifted.values()) == 1200
    assert all(corridor.geometry.buffer(1e-8).covers(LineString(part))
               for parts in shifted.values() for part in parts)
    monkeypatch.setattr(viz, "score_segments", forbidden)
    empty = viz._Corridor(corridor.geometry, {}, corridor.extent, corridor.radius_m)
    assert viz._corridor_scores(graph, positions, empty) == {}