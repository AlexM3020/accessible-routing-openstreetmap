import heapq
import math
import random
from dataclasses import replace

import pytest

from osm_accessibility.builder import build_graph
from osm_accessibility.data import Dataset
from osm_accessibility.errors import DataError, NoRouteFound
from osm_accessibility.evaluation import Evaluator, evaluate_edge
from osm_accessibility.geo import haversine_m
from osm_accessibility.graph import InMemoryGraph
from osm_accessibility.models import Coordinate, Edge, Node
from osm_accessibility.outputs import graph_fingerprint, route_summary
from osm_accessibility.profiles import CostParameters, Profile, profile_named, wheelchair_profile
from osm_accessibility.routing import find_route, snap_endpoints
from osm_accessibility.tags import incline_percent, metres


def test_artificial_obstacles_are_avoided_but_preserved_in_graph(graph, evaluator):
    route = find_route(graph, "s", "t", evaluator)
    assert all(edge.tags.get("highway") != "steps" for edge in route.edges)
    assert "gate" not in route.nodes
    assert any(edge.tags.get("highway") == "steps" for edge in graph.edges)
    assert route.cost == pytest.approx(sum(evaluator(edge).cost_m for edge in route.edges))
    assert route_summary(route, evaluator)["hard_block_violations"] == 0


def test_profile_hard_constraint_is_not_inferred_from_weight():
    edge = Edge("e", "a", "b", "stairs", 10, {"highway": "steps", "wheelchair": "no"})
    soft = Profile(weights={"wheelchair_accessible": 1}, wheelchair_required=False)
    hard = Profile(weights={}, wheelchair_required=True)
    assert not evaluate_edge(edge, soft, CostParameters()).blocked
    assert evaluate_edge(edge, hard, CostParameters()).blocked


@pytest.mark.parametrize("tags,blocked", [
    ({"highway": "footway", "access": "private"}, True),
    ({"highway": "footway", "access": "no", "foot": "yes"}, False),
    ({"highway": "motorway", "foot": "yes"}, True),
    ({"highway": "cycleway"}, True),
    ({"highway": "cycleway", "foot": "designated"}, False),
    ({"highway": "steps", "ramp": "separate"}, True),
    ({"highway": "steps", "ramp": "yes", "ramp:bicycle": "yes"}, True),
    ({"highway": "steps", "ramp:wheelchair": "yes"}, False),
    ({"highway": "steps", "ramp": "yes", "wheelchair": "yes", "ramp:wheelchair": "no"}, True),
    ({"highway": "steps", "ramp": "yes", "wheelchair": "yes"}, False),
])
def test_hard_policy(tags, blocked):
    result = evaluate_edge(Edge("e", "a", "b", "w", 10, tags), wheelchair_profile(), CostParameters())
    assert result.blocked is blocked
    assert (result.cost_m is None) is blocked
    if blocked:
        assert result.hard_blocks and result.score is None


@pytest.mark.parametrize("node_tags", [{"barrier": "stile"}, {"wheelchair": "no"}, {"barrier": "wall"}])
def test_node_obstacles_cannot_be_reset_by_positive_way_tag(node_tags):
    edge = Edge("e", "a", "b", "w", 10, {"highway": "footway", "wheelchair": "yes"}, target_tags=node_tags)
    assert evaluate_edge(edge, wheelchair_profile(), CostParameters()).blocked


def test_edge_diagnostics_decompose_cost_and_score():
    edge = Edge("e", "a", "b", "w", 100, {"highway": "footway", "surface": "asphalt", "lit": "no"},
                target_tags={"highway": "crossing", "kerb": "raised"}, way_has_crossing_nodes=True, way_has_kerb_nodes=True)
    profile = Profile(weights={"surface_type": .75, "lit_roads": .25, "short_kerbs": 1})
    parameters = CostParameters(accessibility_factor=2, event_equivalent_m=20)
    evaluation = evaluate_edge(edge, profile, parameters)
    assert evaluation.way_risk == .25
    assert evaluation.event_risk == 1
    assert evaluation.penalty_m == 90
    assert evaluation.cost_m == 190
    assert evaluation.score == pytest.approx(100 * (1 - 45 / 120))
    assert evaluation.coverage == 100
    assert len(evaluation.criteria) == 3


def test_unknown_data_and_inactive_criteria_are_not_claimed_perfect():
    edge = Edge("e", "a", "b", "w", 10, {"highway": "footway"})
    unknown = evaluate_edge(edge, Profile(weights={"surface_type": 1}), CostParameters(unknown_risk=.75))
    assert unknown.score == 25 and unknown.coverage == 0
    inactive = evaluate_edge(edge, Profile(), CostParameters())
    assert inactive.score is None and inactive.coverage is None
    assert inactive.cost_m == 10


def test_fixed_crossing_cost_and_pooled_score_survive_geometry_subdivision():
    profile = Profile(weights={"surface_type": 1, "short_kerbs": 1, "supervised_crossings": 1})
    parameters = CostParameters()
    tags = {"highway": "footway", "surface": "asphalt", "footway": "crossing", "kerb": "raised"}
    crossing = {"highway": "crossing", "crossing": "unmarked", "kerb": "raised"}
    whole = Edge("whole", "a", "c", "w", 100, tags, target_tags=crossing, way_has_crossing_nodes=True, way_has_kerb_nodes=True)
    first = Edge("first", "a", "b", "w", 90, tags, way_has_crossing_nodes=True, way_has_kerb_nodes=True)
    last = replace(whole, id="last", source="b", distance_m=10)
    single = evaluate_edge(whole, profile, parameters)
    split = [evaluate_edge(edge, profile, parameters) for edge in (first, last)]
    assert single.cost_m == pytest.approx(sum(item.cost_m for item in split))
    assert single.exposure_m == pytest.approx(sum(item.exposure_m for item in split))
    assert single.score == pytest.approx(100 * (1 - sum(item.way_risk_metres + item.event_risk_metres for item in split) / sum(item.exposure_m for item in split)))


def test_events_only_dont_dilute_scores_with_unscored_lengths():
    profile = Profile(weights={"short_kerbs": 1})
    ordinary = Edge("plain", "a", "b", "w", 1000, {"highway": "footway"})
    event = Edge("event", "b", "c", "w", 5, {"highway": "footway"}, target_tags={"kerb": "raised"})
    plain_eval, event_eval = [evaluate_edge(edge, profile, CostParameters()) for edge in (ordinary, event)]
    assert plain_eval.score is None and plain_eval.exposure_m == 0
    assert event_eval.score == 0 and event_eval.exposure_m == 20


def test_builder_marks_unknown_kerb_at_crossing_for_whole_way():
    documents = [{"id": node, "location": {"coordinates": coord}, "tags": tags} for node, coord, tags in (
        ("a", [11, 49.58], {}), ("b", [11.0005, 49.58], {}),
        ("c", [11.001, 49.58], {"highway": "crossing", "crossing": "unmarked"}),
    )]
    tags = {"highway": "footway", "surface": "asphalt", "footway": "crossing", "kerb": "raised"}
    whole = build_graph(documents, [{"id": "w", "nodes": ["a", "c"], "tags": tags}])
    divided = build_graph(documents, [{"id": "w", "nodes": ["a", "b", "c"], "tags": tags}])
    profile = Profile(weights={"surface_type": 1, "short_kerbs": 1, "supervised_crossings": 1})
    summaries = []
    for graph in (whole, divided):
        assert all(edge.way_has_kerb_nodes for edge in graph.edges)
        evaluator = Evaluator(profile, CostParameters())
        summaries.append(route_summary(find_route(graph, "a", "c", evaluator), evaluator))
    # Haversine between latitude-parallel subdivision points has a tiny spherical
    # geometry difference; the 20 m event contribution must remain identical.
    assert summaries[0]["event_risk_metres"] == summaries[1]["event_risk_metres"]
    assert summaries[0]["profile_cost_m"] == pytest.approx(summaries[1]["profile_cost_m"], abs=1e-6)
    assert summaries[0]["score_percent"] == pytest.approx(summaries[1]["score_percent"], abs=1e-6)


def test_coverage_percent_never_rounds_above_one_hundred(graph, evaluator):
    for edge in graph.edges:
        value = evaluator(edge).coverage
        assert value is None or 0 <= value <= 100


@pytest.mark.parametrize("seed", range(10))
def test_astar_matches_independent_shortest_path_oracle(seed):
    rng = random.Random(seed)
    nodes = [Node(str(i), Coordinate(49.58 + i // 4 * .001, 11 + i % 4 * .001)) for i in range(16)]
    edges = []
    for index in range(75):
        source, target = rng.sample(nodes, 2)
        tags = rng.choice([{"highway": "footway", "surface": "asphalt"},
                           {"highway": "path", "surface": "gravel"}, {"highway": "steps"}])
        edges.append(Edge(str(index), source.id, target.id, str(index), haversine_m(source.coordinate, target.coordinate), tags))
    graph = InMemoryGraph(nodes, edges)
    evaluator = Evaluator(wheelchair_profile(), CostParameters(accessibility_factor=6))
    costs = {edge.id: evaluator(edge).cost_m for edge in graph.edges}
    queue, best = [(0.0, "0")], {"0": 0.0}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        for edge in graph.adjacency.get(node, ()):
            if costs[edge.id] is None:
                continue
            candidate = cost + costs[edge.id]
            if candidate < best.get(edge.target, math.inf):
                best[edge.target] = candidate
                heapq.heappush(queue, (candidate, edge.target))
    if "15" not in best:
        with pytest.raises(NoRouteFound):
            find_route(graph, "0", "15", evaluator)
    else:
        route = find_route(graph, "0", "15", evaluator)
        assert route.cost == pytest.approx(best["15"])
        assert route.nodes == ("0", *(edge.target for edge in route.edges))


def test_distance_baseline_preserves_wheelchair_requirements(graph, evaluator):
    route = find_route(graph, "s", "t", evaluator, "dijkstra", distance_only=True)
    assert all(not evaluator(edge).blocked for edge in route.edges)
    assert route.cost == pytest.approx(route.distance_m)


def test_graph_and_routing_are_input_order_invariant(dataset, graph, evaluator):
    shuffled = Dataset.from_elements(list(reversed(list(dataset.elements()))), dataset.metadata).graph()
    assert graph_fingerprint(graph) == graph_fingerprint(shuffled)
    assert find_route(graph, "s", "t", evaluator).nodes == find_route(shuffled, "s", "t", evaluator).nodes


def test_disconnected_and_identical_snapping_are_explicit(graph, evaluator):
    with pytest.raises(NoRouteFound):
        find_route(graph, "s", "island_a", evaluator)
    with pytest.raises(NoRouteFound):
        snap_endpoints(graph, Coordinate(49.58, 11), Coordinate(49.58, 11), evaluator)
    with pytest.raises(NoRouteFound):
        snap_endpoints(graph, Coordinate(0, 0), Coordinate(49.58, 11.002), evaluator)


def test_snapping_retains_requested_and_mapped_points(graph, evaluator):
    start, end = snap_endpoints(graph, Coordinate(49.58001, 11), Coordinate(49.58, 11.002), evaluator)
    assert start.node_id == "s" and start.distance_m > 0
    assert start.requested != start.coordinate
    assert end.node_id == "t"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_weight(value):
    with pytest.raises(DataError):
        Profile(weights={"surface_type": value})


@pytest.mark.parametrize("value,expected", [("120 cm", 1.2), ("3 ft", .9144), ("1,2 m", 1.2), ("2;3", None), ("-1", None)])
def test_osm_widths(value, expected):
    assert metres(value) == pytest.approx(expected) if expected is not None else metres(value) is None


def test_incline_units_and_profiles():
    assert incline_percent("45°") == pytest.approx(100)
    assert incline_percent("-6%") == -6
    assert incline_percent("up") is None
    assert Profile.from_dict({"weights": {"surface_type": "Essential"}}).weights["surface_type"] == 1
    assert profile_named("walking").wheelchair_required is False
    with pytest.raises(DataError):
        CostParameters(unknown_risk=2)


def test_direction_and_graph_integrity():
    nodes = [{"id": name, "location": {"coordinates": coordinate}} for name, coordinate in
             (("a", [11, 49.58]), ("b", [11.001, 49.58]))]
    graph = build_graph(nodes, [{"id": "way", "nodes": ["a", "b"], "tags": {"highway": "path", "oneway:foot": "-1", "incline": "6%"}}])
    assert len(graph.edges) == 1 and graph.edges[0].source == "b"
    assert graph.edges[0].tags["incline"] == "-6%"
    with pytest.raises(DataError):
        InMemoryGraph(graph.nodes.values(), [replace(graph.edges[0], distance_m=1)])
    with pytest.raises(DataError):
        build_graph(nodes, [{"id": "bad", "nodes": ["a", "absent"], "tags": {"highway": "path"}}])
    assert all(edge.source != edge.target for edge in graph.edges)
