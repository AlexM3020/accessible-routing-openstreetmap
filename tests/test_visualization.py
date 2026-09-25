"""Small, manually specified scientific fixtures; no image golden files."""

from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, hypot, radians, sin, sqrt
from pathlib import Path

import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.image import imread
from pyproj import CRS, Transformer

from osm_accessibility.graph import InMemoryGraph
from osm_accessibility.models import Coordinate, Edge, Node, Route
from osm_accessibility.visualization import create_figures, projected_positions, score_segments


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


def test_artifacts_geometry_scores_blocked_edges_and_nonmutation(example, tmp_path, monkeypatch):
    graph, route, diagnostics, pois = example
    before = _input_snapshot(*example)
    positions = projected_positions(graph)
    true = {edge.id: (positions[edge.source], positions[edge.target]) for edge in graph.edges}
    shifted = score_segments(graph, positions)
    original_save = Figure.savefig
    inspected = set()

    def inspect(figure, destination, **kwargs):
        name = Path(destination.name).stem
        inspected.add(name)
        for ax in figure.axes[:2 if name == "accessibility" else 1]:
            assert ax.get_aspect() == 1
            assert ax.get_xlabel() == "Easting (m)"
            assert ax.get_ylabel() == "Northing (m)"
        caption = "\n".join(text.get_text() for text in figure.texts)
        assert "Synthetic example" in caption
        assert "not an accuracy proof" in caption
        assert "not a pure continuous score" in caption
        if name in {"route", "graph_pois"}:
            ax = figure.axes[0]
            line = next(line for line in ax.lines if line.get_gid() == "selected-route")
            assert list(zip(line.get_xdata(), line.get_ydata())) == [positions[key] for key in route.nodes]
            assert line.get_color() == "#1464b4"
            assert {text.get_text() for text in ax.texts if text.get_gid() == "route-endpoint-label"} == {"Start", "End"}
            arrows = _collection(ax, "route-directions")
            assert list(arrows.U) == pytest.approx([0.22 * (true[e.id][1][0] - true[e.id][0][0]) for e in route.edges])
            assert list(arrows.V) == pytest.approx([0.22 * (true[e.id][1][1] - true[e.id][0][1]) for e in route.edges])
        if name == "graph_pois":
            assert _segments(_collection(ax, "graph-edges")) == [true[edge.id] for edge in graph.edges]
            blocked = _collection(ax, "graph-blocked")
            assert _segments(blocked) == [true[edge.id] for edge in graph.edges if diagnostics[edge.id]["blocked"]]
            assert tuple(blocked.get_colors()[0]) == pytest.approx(to_rgba("#c52b2f"))
            assert all(dashes is not None for _, dashes in blocked.get_linestyles())
            symbols = [artist for artist in ax.collections if isinstance(artist, PathCollection)
                       and (artist.get_gid() or "").startswith("poi-")]
            assert sum(len(artist.get_offsets()) for artist in symbols) == len(pois)
            for kind in {poi["kind"] for poi in pois}:
                assert any(artist.get_gid().startswith(f"poi-{kind}-off-") for artist in symbols)
            on_path = _collection(ax, "poi-crossing-on-not-blocked")
            assert tuple(on_path.get_offsets()[0]) == pytest.approx(positions["b"])
            assert tuple(on_path.get_facecolors()[0]) == pytest.approx(to_rgba("#1464b4"))
            assert len(_collection(ax, "poi-crossing-off-unknown").get_facecolors()) == 0
            for artist in symbols:
                for x, y in artist.get_offsets():
                    assert ax.get_xlim()[0] < x < ax.get_xlim()[1]
                    assert ax.get_ylim()[0] < y < ax.get_ylim()[1]
        if name == "accessibility":
            assert "offsets (at most 1 m) are not real geometry" in caption
            for ax, prefix, edges, geometry in ((figure.axes[0], "graph", graph.edges, shifted),
                                                 (figure.axes[1], "path", route.edges, true)):
                collection = _collection(ax, f"{prefix}-score")
                active = [edge for edge in edges if diagnostics[edge.id]["score"] is not None]
                assert _segments(collection) == [geometry[edge.id] for edge in active]
                assert list(collection.get_array()) == [diagnostics[edge.id]["score"] for edge in active]
                assert collection.norm.vmin == 0 and collection.norm.vmax == 100
                assert collection.cmap.name == "cividis"
                assert sum(len(artist.get_segments()) for artist in ax.collections
                           if isinstance(artist, LineCollection)) == len(edges)
            blocked = _collection(figure.axes[0], "graph-blocked")
            assert _segments(blocked) == [shifted[e.id] for e in graph.edges if diagnostics[e.id]["blocked"]]
            assert all(dashes is not None for _, dashes in blocked.get_linestyles())
            assert tuple(blocked.get_colors()[0]) == pytest.approx(to_rgba("#c52b2f"))
            unscored = _collection(figure.axes[0], "graph-unscored")
            assert _segments(unscored) == [true["da-unscored"]]
            assert tuple(unscored.get_colors()[0]) == pytest.approx(to_rgba("#8c8c8c"))
            colour_axis = figure.axes[2]
            assert colour_axis.get_xlabel() == "Event-inclusive edge score (0-100)"
            assert [tick.get_text() for tick in colour_axis.get_xticklabels()] == ["0 (low)", "25", "50", "75", "100 (high)"]
        return original_save(figure, destination, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    paths = create_figures(*example, title="Manual directed graph", synthetic=True, output_dir=tmp_path, dpi=60)
    assert paths == [tmp_path / f"{name}.{fmt}" for name in ("route", "graph_pois", "accessibility")
                     for fmt in ("png", "pdf")]
    assert inspected == {"route", "graph_pois", "accessibility"}
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
    assert _input_snapshot(*example) == before


def test_all_pois_kept_but_only_25_deterministic_annotations(example, tmp_path, monkeypatch):
    graph, route, diagnostics, pois = example
    pois = pois + [dict(pois[0], id=f"extra-{index:02}", longitude=11.002 + index * 0.00001)
                   for index in range(30)]
    labels = []

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "graph_pois":
            ax = figure.axes[0]
            symbols = [artist for artist in ax.collections if (artist.get_gid() or "").startswith("poi-")]
            assert sum(len(artist.get_offsets()) for artist in symbols) == len(pois)
            labels.append([text.get_text() for text in ax.texts if text.get_gid() == "poi-label"])
            assert len(labels[-1]) == 25
            assert labels[-1] == [str(index) for index in range(1, 26)]
            assert len([text for text in figure.axes[1].texts if text.get_gid() == "poi-key"]) == 25
        assert "© OpenStreetMap contributors, ODbL 1.0" in "\n".join(text.get_text() for text in figure.texts)

    monkeypatch.setattr(Figure, "savefig", inspect)
    for index, ordered in enumerate((pois, list(reversed(pois)))):
        create_figures(graph, route, diagnostics, ordered, title="POI inventory", synthetic=False,
                       output_dir=tmp_path / str(index), file_formats=("png",))
    assert labels[0] == labels[1]


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


def test_existing_later_artifact_is_never_overwritten(example, tmp_path):
    existing = tmp_path / "accessibility.pdf"
    existing.write_bytes(b"preserve exactly")
    with pytest.raises(FileExistsError, match="accessibility.pdf"):
        create_figures(*example, title="Collision", synthetic=True, output_dir=tmp_path)
    assert existing.read_bytes() == b"preserve exactly"
    assert list(tmp_path.iterdir()) == [existing]


def test_exclusive_creation_handles_race_without_overwrite(example, tmp_path, monkeypatch):
    original_open = Path.open
    raced = tmp_path / "accessibility.pdf"

    def race(path, mode="r", *args, **kwargs):
        if path == raced and mode == "xb":
            with original_open(path, "wb") as stream:
                stream.write(b"concurrent plot")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        create_figures(*example, title="Concurrent creation", synthetic=True, output_dir=tmp_path)
    assert raced.read_bytes() == b"concurrent plot"
    assert list(tmp_path.iterdir()) == [raced]


def test_failed_write_removes_only_its_new_artifacts(example, tmp_path, monkeypatch):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    def fail(figure, destination, **kwargs):
        destination.write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        create_figures(*example, title="Failure", synthetic=True, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def test_no_active_criteria_and_blocked_selected_edge_styles(example, tmp_path, monkeypatch):
    graph, route, diagnostics, pois = example
    for record in diagnostics.values():
        record.update(score=None, way_score=None, coverage=None)
    diagnostics["bc"].update(blocked=True, hard_blocks=["selected edge restriction"], score=0.0)

    def inspect(figure, destination, **kwargs):
        if Path(destination.name).stem == "accessibility":
            for ax in figure.axes[:2]:
                assert not any((artist.get_gid() or "").endswith("-score") for artist in ax.collections)
            ax = figure.axes[1]
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
        if Path(destination.name).stem == "route":
            ax = figure.axes[0]
            assert "Start / end" in [text.get_text() for text in ax.texts]
            assert ax.get_xlim()[1] - ax.get_xlim()[0] >= 10
        if Path(destination.name).stem == "accessibility":
            assert any(isinstance(artist, PathCollection) for artist in figure.axes[0].collections)

    monkeypatch.setattr(Figure, "savefig", inspect)
    create_figures(graph, route, diagnostics, [], title="Stationary", synthetic=True,
                   output_dir=tmp_path, file_formats=("png",))


@pytest.mark.parametrize("formats", [(), ("svg",), ("png", "png")])
def test_invalid_formats_have_no_output(example, tmp_path, formats):
    with pytest.raises(ValueError, match="file_formats"):
        create_figures(*example, title="Formats", synthetic=True, output_dir=tmp_path, file_formats=formats)
    assert not list(tmp_path.iterdir())
