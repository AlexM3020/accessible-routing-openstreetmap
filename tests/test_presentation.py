"""Output-only contracts using real evaluator diagnostics and explicit route traces.

The manual trace helper deliberately permits a model-blocked trace so the report's
independent hard-violation audit can be tested without changing the router.
"""

import json
import re
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from osm_accessibility import outputs
from osm_accessibility.builder import build_graph
from osm_accessibility.errors import DataError
from osm_accessibility.evaluation import Evaluator
from osm_accessibility.graph import InMemoryGraph
from osm_accessibility.models import Coordinate, Node, Route
from osm_accessibility.outputs import (
    OBSTACLE_KINDS,
    PresentationOptions,
    build_route_comparison,
    build_route_presentation,
    edge_diagnostics,
    gpx_text,
    graph_fingerprint,
    render_route_summary,
    route_summary,
)
from osm_accessibility.profiles import CostParameters, Profile
from osm_accessibility.routing import find_route


def _network(ways, *, node_tags=None, coordinates=None):
    names = list(dict.fromkeys(node for _, nodes, _ in ways for node in nodes))
    node_tags, coordinates = node_tags or {}, coordinates or {}
    documents = []
    for index, name in enumerate(names):
        latitude, longitude = coordinates.get(name, (49.58, 11 + index * .0001))
        documents.append({"id": name, "location": {"coordinates": [longitude, latitude]},
                          "tags": node_tags.get(name, {})})
    return build_graph(documents, [{"id": name, "nodes": nodes, "tags": tags} for name, nodes, tags in ways])


def _trace(graph, edge_ids, profile, parameters=None):
    evaluator = Evaluator(profile, parameters or CostParameters())
    edges_by_id = {edge.id: edge for edge in graph.edges}
    edges = tuple(edges_by_id[identifier] for identifier in edge_ids)
    evaluations = [evaluator(edge) for edge in edges]
    route = Route((edges[0].source, *(edge.target for edge in edges)), edges,
                  sum(edge.distance_m for edge in edges),
                  sum(item.length_m + item.penalty_m for item in evaluations))
    return route, edge_diagnostics(graph, route, evaluator), evaluator


def _markers(presentation, kind):
    return [row for row in presentation["map_obstacles"] if row["kind"] == kind]


def _body(presentation):
    """The only report section permitted to change with marker configuration."""
    return render_route_summary(presentation).split("## Map markers", 1)[0]


def test_options_contract_round_trip_and_frozen_state():
    assert OBSTACLE_KINDS == (
        "stairs", "barrier", "wheelchair_restriction", "kerb", "crossing", "incline",
        "ramp", "surface", "smoothness", "width", "lighting", "bicycles",
    )
    assert PresentationOptions().as_dict() == {"obstacle_kinds": None, "max_markers": 12}
    assert PresentationOptions.from_dict({}) == PresentationOptions()
    for kinds in (None, (), ("stairs",), OBSTACLE_KINDS):
        for limit in (0, 12, 100):
            options = PresentationOptions(kinds, limit)
            assert PresentationOptions.from_dict(json.loads(json.dumps(options.as_dict()))) == options
            with pytest.raises(FrozenInstanceError):
                setattr(options, "max_markers", 7)
    options = PresentationOptions(("stairs", "ramp"))
    exported = options.as_dict()
    exported["obstacle_kinds"].append("kerb")
    assert options.obstacle_kinds == ("stairs", "ramp")


@pytest.mark.parametrize("kinds", ["stairs", ["stairs"], {"stairs"}, ("stairs", "stairs"),
                                   ("unknown",), ("Stairs",), (None,), (True,), ([],)])
def test_options_reject_invalid_constructor_kinds(kinds):
    with pytest.raises(DataError):
        PresentationOptions(obstacle_kinds=kinds)


@pytest.mark.parametrize("limit", [True, False, -1, 101, 1.0, "12", None, float("nan")])
def test_options_reject_noninteger_or_out_of_bounds_limits(limit):
    with pytest.raises(DataError):
        PresentationOptions(max_markers=limit)


@pytest.mark.parametrize("raw", [None, [], "stairs", {"extra": True}, {"obstacle_kinds": "stairs"},
                                 {"obstacle_kinds": ["stairs", "stairs"]}, {"obstacle_kinds": [4]},
                                 {"max_markers": False}, {"max_markers": 101}])
def test_options_reject_invalid_serialized_forms(raw):
    with pytest.raises(DataError):
        PresentationOptions.from_dict(raw)


def test_known_conflict_and_zero_risk_unknown_have_distinct_complete_records():
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no"})])
    profile = Profile(name="Lighting and surface", weights={"lit_roads": 1, "surface_type": .75})
    parameters = CostParameters(unknown_risk=0)
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile, parameters)
    result = build_route_presentation(graph, route, diagnostics, profile, parameters, synthetic=True)
    expected = route_summary(route, evaluator)
    assert set(result) == {
        "profile_name", "selected_preferences", "requirements", "metrics", "conflicts", "unknowns",
        "encountered_obstacles", "map_obstacles", "filter_summary", "validation", "synthetic", "snapping", "options",
    }
    assert result["metrics"] == {"distance_m": route.distance_m, "cost_m": route.cost,
                                 "score_percent": expected["score_percent"],
                                 "coverage_percent": expected["coverage_percent"]}
    assert len(result["conflicts"]) == len(result["unknowns"]) == 1
    conflict, unknown = result["conflicts"][0], result["unknowns"][0]
    required = {"id", "kind", "label", "preference", "weight", "status", "location", "way_id", "node_id",
                "edge_ids", "distance_m", "risk", "reason"}
    assert set(conflict) == set(unknown) == required
    assert conflict["status"] == "conflict" and conflict["preference"] == "lit_roads"
    assert "lit=no" in conflict["reason"]
    assert unknown["status"] == "unknown" and unknown["preference"] == "surface_type"
    assert unknown["risk"] == 0
    assert "surface/tracktype not recorded" in unknown["reason"]
    assert unknown["edge_ids"] == ["w:0:f"]
    assert unknown["distance_m"] == route.distance_m
    assert "OSM way w" in unknown["location"] and "route segment 1" in unknown["location"]
    assert not _markers(result, "surface")
    assert {row["kind"] for row in result["encountered_obstacles"]} == {"lighting"}
    assert result["validation"]["hard_constraint_violations"] == 0
    assert [item["importance"] for item in result["selected_preferences"]] == ["Essential", "Very high"]
    report = render_route_summary(result)
    assert "Synthetic example" in report and "**PREFERENCE CONFLICT**" in report
    assert "**Essential**" in report and "**Very high**" in report
    assert "## Missing data (1 grouped entries)" in report
    assert "not run or not supplied" in report
    json.dumps(result, allow_nan=False)


def test_stairway_subdivision_groups_distance_count_and_true_length_midpoint():
    graph = _network(
        [("stairs", ["a", "b", "c"], {"highway": "steps", "step_count": "12"})],
        coordinates={"a": (49.58, 11), "b": (49.58, 11.0002), "c": (49.581, 11.0002)},
    )
    profile = Profile(weights={"avoid_stairs": 1})
    route, diagnostics, evaluator = _trace(graph, ["stairs:0:f", "stairs:1:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert len(result["conflicts"]) == len(result["encountered_obstacles"]) == 1
    row = result["conflicts"][0]
    assert row["edge_ids"] == [edge.id for edge in route.edges]
    assert row["distance_m"] == pytest.approx(route.distance_m)
    assert row["risk"] == 1
    assert "route segments 1–2" in row["location"]
    assert row["reason"].count("step_count=12") == 1
    assert "step_count=24" not in render_route_summary(result)
    markers = _markers(result, "stairs")
    assert len(markers) == 1 and markers[0]["first_segment"] == 1
    assert markers[0]["edge_ids"] == ["stairs:0:f", "stairs:1:f"]
    fraction = (route.distance_m / 2 - route.edges[0].distance_m) / route.edges[1].distance_m
    assert 0 < fraction < 1
    assert markers[0]["longitude"] == pytest.approx(11.0002, rel=0, abs=1e-10)
    assert markers[0]["latitude"] == pytest.approx(49.58 + .001 * fraction, rel=0, abs=1e-10)
    assert markers[0]["longitude"] != pytest.approx((11 + 11.0002 + 11.0002) / 3, abs=1e-8)


@pytest.mark.parametrize("criterion,kind,tags,observation", [
    ("avoid_bicycles", "bicycles", {"bicycle": "yes"}, "bicycle=yes"),
    ("lit_roads", "lighting", {"lit": "no"}, "lit=no"),
    ("surface_type", "surface", {"surface": "gravel"}, "surface=gravel"),
    ("smoothness", "smoothness", {"smoothness": "bad"}, "smoothness=bad"),
    ("short_kerbs", "kerb", {"kerb": "raised"}, "kerb=raised"),
    ("supervised_crossings", "crossing", {"footway": "crossing", "crossing": "unmarked"}, "crossing=unmarked"),
    ("road_width", "width", {"width": "70 cm"}, "width=70 cm"),
    ("road_incline", "incline", {"incline": "5°"}, "incline=5°"),
    ("avoid_stairs", "stairs", {"highway": "steps"}, "highway=steps"),
    ("wheelchair_accessible", "wheelchair_restriction", {"wheelchair": "limited"}, "wheelchair=limited"),
])
def test_each_selected_known_criterion_maps_to_its_own_obstacle_kind(criterion, kind, tags, observation):
    graph = _network([("w", ["a", "b"], {"highway": "footway", **tags})])
    profile = Profile(weights={criterion: 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    conflict, = result["conflicts"]
    assert conflict["preference"] == criterion and conflict["kind"] == kind
    assert conflict["status"] == "conflict" and conflict["risk"] > 0
    assert observation in conflict["reason"]
    assert _markers(result, kind)[0]["status"] == "conflict"


def test_positive_unknown_risk_is_not_misreported_as_an_obstacle_or_conflict():
    graph = _network([("w", ["a", "b"], {"highway": "footway"})])
    profile = Profile(weights={"lit_roads": 1})
    parameters = CostParameters(unknown_risk=1)
    route, diagnostics, _ = _trace(graph, ["w:0:f"], profile, parameters)
    result = build_route_presentation(graph, route, diagnostics, profile, parameters)
    assert result["unknowns"][0]["risk"] == 1
    assert result["metrics"]["score_percent"] == result["metrics"]["coverage_percent"] == 0
    assert result["map_obstacles"] == result["encountered_obstacles"] == result["conflicts"] == []
    assert result["validation"]["hard_constraint_violations"] == 0


def test_adjacent_backtracking_is_one_way_encounter_with_the_exact_traversal_length():
    graph = _network([("w", ["a", "b", "c"], {"highway": "steps"})])
    profile = Profile(weights={"avoid_stairs": 1})
    sequence = ["w:0:f", "w:0:r", "w:0:f", "w:1:f"]
    route, diagnostics, evaluator = _trace(graph, sequence, profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    conflict, = result["conflicts"]
    marker, = _markers(result, "stairs")
    assert conflict["edge_ids"] == marker["edge_ids"] == sequence
    assert conflict["distance_m"] == route.distance_m
    assert "route segments 1–4" in conflict["location"]


@pytest.mark.parametrize("step_count", ["0", "00", "-1", "3;4", ""])
def test_zero_or_ambiguous_step_count_is_not_an_exact_observed_count(step_count):
    graph = _network([("w", ["a", "b"], {"highway": "steps", "step_count": step_count})])
    profile = Profile(weights={"avoid_stairs": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert "step count unknown" in result["conflicts"][0]["reason"]
    assert result["conflicts"][0]["status"] == "conflict"  # Stairs themselves are known.


def test_nonconsecutive_way_visits_and_repeated_edges_keep_separate_encounters():
    graph = _network([("stairs", ["a", "b", "c"], {"highway": "steps"}),
                      ("detour", ["c", "d", "b"], {"highway": "footway"})])
    profile = Profile(weights={"avoid_stairs": 1})
    sequence = ["stairs:0:f", "stairs:1:f", "detour:0:f", "detour:1:f", "stairs:1:f"]
    route, diagnostics, evaluator = _trace(graph, sequence, profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert len(result["conflicts"]) == len(_markers(result, "stairs")) == 2
    first, second = result["conflicts"]
    assert first["edge_ids"] == sequence[:2] and second["edge_ids"] == sequence[-1:]
    assert "route segments 1–2" in first["location"]
    assert "route segment 5" in second["location"]
    assert first["id"] != second["id"]
    assert second["distance_m"] == route.edges[-1].distance_m
    assert [row["first_segment"] for row in _markers(result, "stairs")] == [1, 5]
    assert result["metrics"]["score_percent"] == pytest.approx(route_summary(route, evaluator)["score_percent"])


def test_missing_or_nonconflicting_middle_segment_breaks_a_way_criterion_group():
    original = _network([("w", ["a", "b", "c", "d"], {"highway": "footway", "lit": "no"})])
    edges = [replace(edge, tags={"highway": "footway"}) if edge.id == "w:1:f" else edge
             for edge in original.edges]
    graph = InMemoryGraph(original.nodes.values(), edges)
    profile = Profile(weights={"lit_roads": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f", "w:1:f", "w:2:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert [row["edge_ids"] for row in result["conflicts"]] == [["w:0:f"], ["w:2:f"]]
    assert [row["edge_ids"] for row in result["unknowns"]] == [["w:1:f"]]
    assert [row["first_segment"] for row in _markers(result, "lighting")] == [1, 3]


def test_node_events_use_arrivals_once_and_preserve_nonconsecutive_visits():
    graph = _network([("w", ["a", "b", "c"], {"highway": "footway"})], node_tags={
        "b": {"highway": "crossing", "crossing": "unmarked", "kerb": "raised", "kerb:height": "12 cm"},
    })
    profile = Profile(weights={"short_kerbs": 1, "supervised_crossings": .75})
    for sequence, arrivals in ((["w:0:f", "w:1:f"], [1]),
                               (["w:0:f", "w:1:f", "w:1:r", "w:1:f"], [1, 3])):
        route, diagnostics, evaluator = _trace(graph, sequence, profile)
        result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
        assert len(result["conflicts"]) == len(result["encountered_obstacles"]) == 2 * len(arrivals)
        assert all(row["node_id"] == "b" and row["way_id"] is None for row in result["conflicts"])
        assert all(row["distance_m"] == 0 for row in result["conflicts"])
        assert all(len(row["edge_ids"]) == 1 for row in result["conflicts"])
        assert [row["first_segment"] for row in _markers(result, "kerb")] == arrivals
        assert all(row["edge_ids"] != ["w:1:f"] for row in result["conflicts"])
        assert "kerb:height=12 cm (0.12 m height)" in next(
            row["reason"] for row in result["conflicts"] if row["kind"] == "kerb"
        )
        assert result["metrics"]["score_percent"] == 0  # Unscored lengths cannot dilute node events.


def test_crossing_fallback_tags_and_unknown_kerb_do_not_create_an_observed_kerb():
    original = _network([("w", ["a", "b"], {"highway": "footway"})])
    edges = [replace(edge, crossing_tags={"highway": "crossing", "crossing": "unmarked"})
             if edge.id == "w:0:f" else edge for edge in original.edges]
    graph = InMemoryGraph(original.nodes.values(), edges)
    profile = Profile(weights={"short_kerbs": 1, "supervised_crossings": 1})
    parameters = CostParameters(unknown_risk=0)
    route, diagnostics, _ = _trace(graph, ["w:0:f"], profile, parameters)
    result = build_route_presentation(graph, route, diagnostics, profile, parameters)
    assert [row["kind"] for row in result["conflicts"]] == ["crossing"]
    assert [row["kind"] for row in result["unknowns"]] == ["kerb"]
    assert result["unknowns"][0]["node_id"] == "b"
    assert not _markers(result, "kerb")
    assert {row["kind"] for row in result["encountered_obstacles"]} == {"crossing"}


def test_all_zero_profile_has_no_score_or_soft_conflicts_but_reports_stairs():
    graph = _network([("w", ["a", "b"], {"highway": "steps"})])
    profile = Profile(name="Distance only")
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert result["selected_preferences"] == result["conflicts"] == result["unknowns"] == []
    assert result["metrics"]["score_percent"] is None
    assert result["metrics"]["coverage_percent"] is None
    assert result["metrics"]["cost_m"] == result["metrics"]["distance_m"]
    assert _markers(result, "stairs")[0]["status"] == "encountered"
    assert result["encountered_obstacles"][0]["classification"] == "encountered"
    report = render_route_summary(result)
    assert "None selected" in report and "**PREFERENCE CONFLICT**" not in report
    assert "Real-data run" in report and "not field-verified" in report


def test_inactive_stair_preference_is_not_a_conflict_for_another_selected_criterion():
    graph = _network([("w", ["a", "b"], {"highway": "steps", "lit": "yes", "wheelchair": "no"})])
    for weights, expected in (({"lit_roads": 1}, []),
                              ({"wheelchair_accessible": 1}, ["wheelchair_accessible"])):
        profile = Profile(weights=weights)
        route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
        # Explicit types keep each criterion's own marker; automatic mode may
        # combine co-located physical features while preserving their warnings.
        result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                          options=PresentationOptions(OBSTACLE_KINDS))
        assert [row["preference"] for row in result["conflicts"]] == expected
        assert _markers(result, "stairs")[0]["status"] == "encountered"
        assert _markers(result, "stairs")[0]["preference_labels"] == []
        assert result["validation"]["hard_constraint_violations"] == 0


@pytest.mark.parametrize("weights", [{}, {"wheelchair_accessible": 1},
                                     {"avoid_stairs": 1, "wheelchair_accessible": 1}])
def test_automatic_redundant_marker_retains_all_selected_conflicts(weights):
    graph = _network([("stairs", ["a", "b", "c"], {"highway": "steps", "wheelchair": "no"})])
    profile = Profile(weights=weights)
    route, diagnostics, evaluator = _trace(graph, ["stairs:0:f", "stairs:1:f"], profile)
    auto = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    explicit = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                        options=PresentationOptions(OBSTACLE_KINDS))
    marker, = auto["map_obstacles"]
    assert marker["kind"] == "stairs"
    assert marker["status"] == ("conflict" if weights else "encountered")
    expected_labels = {label for row in explicit["map_obstacles"] for label in row["preference_labels"]}
    assert set(marker["preference_labels"]) == expected_labels
    assert "wheelchair=no" in marker["description"]
    assert {row["kind"] for row in explicit["map_obstacles"]} == {"stairs", "wheelchair_restriction"}
    assert auto["encountered_obstacles"] == explicit["encountered_obstacles"]
    assert auto["conflicts"] == explicit["conflicts"]
    assert _body(auto) == _body(explicit)


def test_automatic_dedup_does_not_move_a_partial_way_restriction():
    original = _network([("stairs", ["a", "b", "c"], {"highway": "steps", "wheelchair": "yes"})])
    graph = InMemoryGraph(original.nodes.values(), [
        replace(edge, tags={"highway": "steps", "wheelchair": "no"}) if edge.id == "stairs:0:f" else edge
        for edge in original.edges
    ])
    profile = Profile(weights={"wheelchair_accessible": 1})
    route, diagnostics, evaluator = _trace(graph, ["stairs:0:f", "stairs:1:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    marker, = _markers(result, "wheelchair_restriction")
    assert marker["status"] == "conflict" and marker["edge_ids"] == ["stairs:0:f"]
    assert _markers(result, "stairs")[0]["edge_ids"] == ["stairs:0:f", "stairs:1:f"]


def test_positive_profile_and_diagnostic_weights_are_both_needed_for_conflicts():
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no"})])
    profile = Profile(weights={"lit_roads": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    # A manually supplied inactive criterion must not override the selected profile.
    diagnostics["w:0:f"]["criteria"].append({"name": "avoid_stairs", "scope": "way", "weight": 1,
                                            "risk": 1, "known": True, "reason": "inactive preference"})
    diagnostics["w:0:f"]["criteria"][0]["weight"] = 0
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert result["conflicts"] == []
    assert all(row["classification"] == "encountered" for row in result["encountered_obstacles"])


def test_marker_filters_and_limits_do_not_change_route_report_metrics_or_inputs(graph, evaluator):
    route = find_route(graph, "s", "t", evaluator)
    diagnostics = edge_diagnostics(graph, route, evaluator)
    saved = deepcopy(diagnostics)
    fingerprint, gpx, summary = graph_fingerprint(graph), gpx_text(graph, route), route_summary(route, evaluator)
    snapping = {"start": {"distance_m": 3.25, "requested_lat_lon": [49.58, 11], "connection_verified": False}}
    configurations = [PresentationOptions(), PresentationOptions((), 0),
                      PresentationOptions(("stairs",), 0), PresentationOptions(OBSTACLE_KINDS, 100)]
    results = [build_route_presentation(graph, route, diagnostics, evaluator.profile, evaluator.parameters,
                                       options=options, snapping=snapping, synthetic=True)
               for options in configurations]
    display_fields = {"options", "filter_summary", "map_obstacles"}
    for result in results[1:]:
        assert {key: value for key, value in result.items() if key not in display_fields} == {
            key: value for key, value in results[0].items() if key not in display_fields
        }
        assert _body(result) == _body(results[0])
    assert results[1]["map_obstacles"] == []
    assert len(results[2]["map_obstacles"]) > 0  # A zero marker limit does not cap the candidate list.
    assert {row["kind"] for row in results[2]["map_obstacles"]} == {"stairs"}
    assert results[1]["filter_summary"]["omitted_by_filter"] == results[1]["filter_summary"]["total_candidates"]
    assert results[3]["filter_summary"]["omitted_by_filter"] == 0
    assert diagnostics == saved
    assert graph_fingerprint(graph) == fingerprint
    assert gpx_text(graph, route) == gpx
    assert route_summary(route, evaluator) == summary
    results[0]["snapping"]["start"]["requested_lat_lon"][0] = 0
    assert snapping["start"]["requested_lat_lon"][0] == 49.58


def test_full_report_and_candidate_list_have_no_hidden_marker_cap():
    count = 18
    graph = _network([(f"stairs{index:02}", [str(index), str(index + 1)], {"highway": "steps"})
                      for index in range(count)])
    profile = Profile(weights={"avoid_stairs": 1})
    route, diagnostics, evaluator = _trace(graph, [f"stairs{index:02}:0:f" for index in range(count)], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                      options=PresentationOptions(("stairs",), 3))
    assert len(result["map_obstacles"]) == len(result["conflicts"]) == len(result["encountered_obstacles"]) == count
    assert result["filter_summary"] == {"selected_kinds": ["stairs"], "total_candidates": count,
                                        "selected_candidates": count, "omitted_by_filter": 0, "max_markers": 3}
    report = render_route_summary(result)
    assert report.count("| Avoid stairs |") == count
    assert "18 grouped encounters" in report
    assert "Marker limit: 3" in report and "18 of 18" in report
    assert "before viewport cropping and coalescing" in report
    assert "OSM way stairs17" in report
    assert "not truncated or changed" in report


def test_explicit_selection_can_show_neutral_ramp_crossing_and_kerb_without_conflicts():
    graph = _network([("w", ["a", "b"], {"highway": "steps", "ramp:wheelchair": "yes"})], node_tags={
        "b": {"highway": "crossing", "crossing": "traffic_signals", "kerb": "lowered"},
    })
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    auto = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    selected = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                        options=PresentationOptions(("ramp", "crossing", "kerb")))
    assert {row["kind"] for row in auto["map_obstacles"]} == {"stairs"}
    assert {row["kind"] for row in selected["map_obstacles"]} == {"ramp", "crossing", "kerb"}
    assert all(row["on_route"] and row["status"] == "encountered" for row in selected["map_obstacles"])
    assert selected["encountered_obstacles"] == auto["encountered_obstacles"]
    assert selected["conflicts"] == selected["unknowns"] == []
    assert _body(selected) == _body(auto)


def test_explicit_types_are_not_also_subject_to_automatic_preference_filtering():
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no", "width": "70 cm"})])
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    auto = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    explicit = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                        options=PresentationOptions(("lighting", "width")))
    assert auto["map_obstacles"] == []
    assert {row["kind"] for row in explicit["map_obstacles"]} == {"lighting", "width"}
    assert explicit["conflicts"] == []
    assert len(auto["encountered_obstacles"]) == 2  # Known unselected obstacles remain encounters.
    assert explicit["encountered_obstacles"] == auto["encountered_obstacles"]


@pytest.mark.parametrize("tags,ramps,handrail,wheelchair,risk,blocked", [
    ({"ramp:wheelchair": "yes", "handrail": "yes"}, True, True, True, .25, False),
    ({"ramp:wheelchair": "yes"}, True, True, True, .5, False),
    ({"ramp:wheelchair": "yes", "handrail": "no"}, True, False, True, .25, False),
    ({"ramp": "yes"}, True, True, True, 1, True),
    ({"ramp": "separate"}, True, True, True, 1, True),
    ({"ramp": "yes", "wheelchair": "yes", "ramp:wheelchair": "no"}, True, True, True, 1, True),
    ({"ramp:wheelchair": "yes", "handrail": "yes"}, False, True, True, 1, True),
    ({"ramp:wheelchair": "yes"}, False, True, False, 1, False),
    ({}, True, True, False, 1, False),
])
def test_handrail_and_ramp_classification_matches_evaluator_not_invented_hard_rules(
    tags, ramps, handrail, wheelchair, risk, blocked,
):
    graph = _network([("w", ["a", "b"], {"highway": "steps", **tags})])
    profile = Profile(weights={"avoid_stairs": 1}, allow_ramps=ramps,
                      prefer_handrail=handrail, wheelchair_required=wheelchair)
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                      options=PresentationOptions(("stairs", "ramp")))
    criterion = diagnostics["w:0:f"]["criteria"][0]
    assert criterion["known"] is True and criterion["risk"] == risk
    assert diagnostics["w:0:f"]["blocked"] is blocked
    assert result["validation"]["hard_constraint_violations"] == int(blocked)
    assert len(result["conflicts"]) == 1 and result["conflicts"][0]["risk"] == risk
    assert result["conflicts"][0]["kind"] == "stairs"
    assert result["unknowns"] == []
    assert _markers(result, "stairs")[0]["status"] == ("blocked" if blocked else "conflict")
    assert all(row["preference_labels"] == [] for row in _markers(result, "ramp"))
    requirements = " ".join(result["requirements"])
    assert f"Wheelchair required: {str(wheelchair).lower()}" in requirements
    assert f"Ramps allowed: {str(ramps).lower()}" in requirements
    assert f"Handrail preferred: {str(handrail).lower()}" in requirements
    assert "handrails are not a hard requirement" in requirements
    if blocked:
        with pytest.raises(DataError, match="blocked"):
            route_summary(route, evaluator)  # Existing strict output behavior is preserved.


def test_hard_violations_are_independent_of_soft_selection_and_on_route_flags():
    graph = _network([("w", ["a", "b"], {"highway": "footway"})], node_tags={"b": {"barrier": "stile"}})
    profile = Profile(wheelchair_required=True)
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    for record in diagnostics.values():
        record["on_route"] = False  # Only the exact Route sequence establishes selection.
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert result["validation"]["hard_constraint_violations"] == 1
    assert result["conflicts"] == result["unknowns"] == []
    assert _markers(result, "barrier")[0]["status"] == "blocked"
    assert result["encountered_obstacles"][0]["classification"] == "hard_constraint_violation"
    assert "**HARD CONSTRAINT VIOLATION**" in render_route_summary(result)


def test_selected_direction_is_not_flagged_by_its_blocked_reverse():
    original = _network([("w", ["a", "b"], {"highway": "steps", "ramp:wheelchair": "yes", "handrail": "yes"})])
    graph = InMemoryGraph(original.nodes.values(), [
        replace(edge, tags={"highway": "steps", "wheelchair": "no"}) if edge.id.endswith(":r") else edge
        for edge in original.edges
    ])
    profile = Profile(weights={"avoid_stairs": 1}, wheelchair_required=True)
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    assert diagnostics["w:0:r"]["blocked"] is True and diagnostics["w:0:f"]["blocked"] is False
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert result["validation"]["hard_constraint_violations"] == 0
    assert len(_markers(result, "stairs")) == 1
    assert _markers(result, "stairs")[0]["status"] == "conflict"
    assert _markers(result, "stairs")[0]["edge_ids"] == ["w:0:f"]
    assert not _markers(result, "wheelchair_restriction")
    assert result["conflicts"][0]["risk"] == .25


@pytest.mark.parametrize("both_blocked", [False, True])
def test_offroute_equivalent_directions_coalesce_and_only_all_blocked_means_blocked(both_blocked):
    original = _network([("main", ["a", "b"], {"highway": "footway"}),
                         ("other", ["x", "y", "z"], {"highway": "steps", "ramp:wheelchair": "yes"})])
    graph = InMemoryGraph(original.nodes.values(), [
        replace(edge, tags={"highway": "steps", "wheelchair": "no"})
        if edge.way_id == "other" and (edge.id.endswith(":r") or both_blocked) else edge
        for edge in original.edges
    ])
    profile = Profile(weights={"avoid_stairs": 1}, wheelchair_required=True)
    route, diagnostics, evaluator = _trace(graph, ["main:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    markers = _markers(result, "stairs")
    assert len(markers) == 1  # Not one per segment and direction.
    assert markers[0]["on_route"] is False and markers[0]["first_segment"] is None
    assert markers[0]["status"] == ("blocked" if both_blocked else "context")
    assert set(markers[0]["edge_ids"]) == {"other:0:f", "other:0:r", "other:1:f", "other:1:r"}
    assert result["encountered_obstacles"] == result["conflicts"] == []
    assert "not claimed to have been successfully avoided" in render_route_summary(result)


def test_offroute_adverse_direction_is_not_lost_when_the_first_direction_is_neutral():
    original = _network([("main", ["a", "b"], {"highway": "footway", "lit": "yes"}),
                         ("other", ["x", "y"], {"highway": "footway", "lit": "yes"})])
    graph = InMemoryGraph(original.nodes.values(), [
        replace(edge, tags={"highway": "footway", "lit": "no"}) if edge.id == "other:0:r" else edge
        for edge in original.edges
    ])
    profile = Profile(weights={"lit_roads": 1})
    route, diagnostics, evaluator = _trace(graph, ["main:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    marker, = _markers(result, "lighting")
    assert marker["status"] == "context" and marker["on_route"] is False
    assert "lit=no" in marker["description"]
    assert set(marker["edge_ids"]) == {"other:0:f", "other:0:r"}


def test_supplied_risk_exposures_not_edge_scores_are_used_without_reevaluation(monkeypatch):
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no"})])
    profile = Profile(weights={"lit_roads": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    record = diagnostics["w:0:f"]
    record.update({"exposure_m": 40.0, "way_risk_metres": 5.0, "event_risk_metres": 5.0,
                   "known_exposure_m": 20.0, "score": 99.0, "way_score": 99.0, "coverage": 99.0})

    def unexpected(*args, **kwargs):
        raise AssertionError("Presentation must not calculate a route or reevaluate costs")

    monkeypatch.setattr("osm_accessibility.evaluation.evaluate_edge", unexpected)
    monkeypatch.setattr("osm_accessibility.routing.find_route", unexpected)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    assert result["metrics"] == {"distance_m": route.distance_m, "cost_m": route.cost,
                                 "score_percent": 75.0, "coverage_percent": 50.0}


@pytest.mark.parametrize("value", [None, True, "10", float("nan"), float("inf"), -1])
def test_missing_or_invalid_baseline_cost_is_never_claimed_verified(value):
    graph = _network([("w", ["a", "b"], {"highway": "footway"})])
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    for baselines in (None, {}, {"astar_dijkstra_agree": True},
                      {"astar": {"optimized_cost_m": route.cost}, "dijkstra": {"optimized_cost_m": value}}):
        result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters, baselines=baselines)
        assert result["validation"]["astar_dijkstra_agree"] is None
        assert result["validation"]["optimal_cost_difference_m"] is None
        assert "A*/Dijkstra cost comparison: not run or not supplied" in render_route_summary(result)


@pytest.mark.parametrize("difference,agree", [(0, True), (1e-7, True), (5, False)])
def test_baseline_comparison_difference_and_distance_overhead(difference, agree):
    graph = _network([("w", ["a", "b"], {"highway": "footway"})])
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    baselines = {"astar": {"optimized_cost_m": route.cost},
                 "dijkstra": {"optimized_cost_m": route.cost + difference},
                 "distance_only_same_constraints": {"distance_m": route.distance_m / 2}}
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters, baselines=baselines)
    validation = result["validation"]
    assert validation["astar_dijkstra_agree"] is agree
    assert validation["optimal_cost_difference_m"] == pytest.approx(difference, abs=1e-9)
    assert validation["distance_baseline_m"] == route.distance_m / 2
    assert validation["distance_overhead_percent"] == pytest.approx(100)
    report = render_route_summary(result)
    assert ("verified: supplied costs agree" if agree else "NOT verified: supplied costs disagree") in report
    assert "same graph, cost function and hard constraints" in report
    assert "not proof of physical safety" in report


def test_evaluator_backed_synthetic_baselines_and_all_public_marker_fields(graph, evaluator):
    route = find_route(graph, "s", "t", evaluator)
    baselines = {}
    for label, algorithm, distance_only in (("astar", "astar", False), ("dijkstra", "dijkstra", False),
                                           ("distance_only_same_constraints", "dijkstra", True)):
        local = Evaluator(evaluator.profile, evaluator.parameters)
        baseline = find_route(graph, "s", "t", local, algorithm, distance_only)
        baselines[label] = route_summary(baseline, local)
    result = build_route_presentation(graph, route, edge_diagnostics(graph, route, evaluator),
                                      evaluator.profile, evaluator.parameters, baselines=baselines, synthetic=True)
    assert result["validation"]["astar_dijkstra_agree"] is True
    assert result["validation"]["optimal_cost_difference_m"] == pytest.approx(0)
    required = {"id", "kind", "latitude", "longitude", "on_route", "status", "description",
                "preference_labels", "edge_ids", "way_id", "node_id", "first_segment"}
    assert result["map_obstacles"]
    for marker in result["map_obstacles"]:
        assert set(marker) == required
        assert marker["kind"] in OBSTACLE_KINDS
        assert marker["status"] in {"conflict", "encountered", "blocked", "context"}
        assert isinstance(marker["latitude"], float) and isinstance(marker["longitude"], float)
        assert isinstance(marker["id"], str) and isinstance(marker["edge_ids"], list)
        assert all(isinstance(label, str) for label in marker["preference_labels"])
        assert marker["first_segment"] is None or marker["first_segment"] >= 1
    assert len({marker["id"] for marker in result["map_obstacles"]}) == len(result["map_obstacles"])
    json.dumps(result, allow_nan=False)


def test_empty_route_zero_exposure_and_zero_distance_baseline_are_explicit():
    graph = InMemoryGraph([Node("a", Coordinate(49.58, 11))], [])
    route = Route(("a",), (), 0, 0)
    result = build_route_presentation(graph, route, {}, Profile(), CostParameters(), baselines={
        "astar": {"optimized_cost_m": 0}, "dijkstra": {"optimized_cost_m": 0},
        "distance_only_same_constraints": {"distance_m": 0},
    })
    assert result["metrics"] == {"distance_m": 0.0, "cost_m": 0.0, "score_percent": None, "coverage_percent": None}
    assert result["validation"]["distance_overhead_percent"] == 0
    assert result["validation"]["astar_dijkstra_agree"] is True
    assert result["map_obstacles"] == result["encountered_obstacles"] == []
    json.dumps(result, allow_nan=False)


def test_zero_event_equivalent_does_not_hide_known_conflicts_or_invent_a_score():
    graph = _network([("w", ["a", "b"], {"highway": "footway"})], node_tags={"b": {"kerb": "raised"}})
    profile = Profile(weights={"short_kerbs": 1})
    parameters = CostParameters(event_equivalent_m=0)
    route, diagnostics, _ = _trace(graph, ["w:0:f"], profile, parameters)
    result = build_route_presentation(graph, route, diagnostics, profile, parameters, baselines={
        "distance_only_same_constraints": {"distance_m": 0},
    })
    assert result["metrics"]["score_percent"] is result["metrics"]["coverage_percent"] is None
    assert len(result["conflicts"]) == len(_markers(result, "kerb")) == 1
    assert result["validation"]["distance_overhead_percent"] is None


def test_measurement_keys_units_and_qualitative_incline_unknown_are_preserved():
    graph = _network([("w", ["a", "b"], {"highway": "footway", "width": "80 cm", "incline": "up"})])
    profile = Profile(weights={"road_width": 1, "road_incline": 1}, min_width_m=1.2)
    parameters = CostParameters(unknown_risk=0)
    route, diagnostics, _ = _trace(graph, ["w:0:f"], profile, parameters)
    result = build_route_presentation(graph, route, diagnostics, profile, parameters,
                                      options=PresentationOptions(("incline", "width")))
    assert "width=80 cm (0.8 m)" in result["conflicts"][0]["reason"]
    assert "selected minimum 1.2 m" in result["conflicts"][0]["reason"]
    assert "incline=up" in result["unknowns"][0]["reason"]
    assert "slope magnitude unknown" in result["unknowns"][0]["reason"]
    assert not _markers(result, "incline")
    assert result["validation"]["hard_constraint_violations"] == 0


def test_markdown_escapes_osm_and_profile_text_without_html_or_table_injection():
    hostile = '<img src=x onerror="alert(1)"> | [open](javascript:alert(1))\n## forged **bold** `code` \\'
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no", "name": hostile,
                                         "surface": "<script>bad</script>|new\n| forged row |"})])
    profile = Profile(name=hostile, weights={"lit_roads": 1, "surface_type": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    report = render_route_summary(result)
    assert "<img" not in report and "<script>" not in report
    assert "&lt;img" in report and "&lt;script&gt;" in report
    assert "\n## forged" not in report and "[open](javascript:" not in report
    assert "\\|" in report and "\\*\\*bold\\*\\*" in report
    table_lines = [line for line in report.splitlines() if line.startswith("|")]
    assert len(table_lines) == 6  # Header, separator and one row in each of two sections.
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 6 for line in table_lines)
    assert graph_fingerprint(graph) not in report
    assert str(list(route.nodes)) not in report
    assert '"surface":' not in report


def test_incomplete_diagnostics_or_foreign_route_is_not_silently_summarized():
    graph = _network([("w", ["a", "b"], {"highway": "footway"})])
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    missing = {key: value for key, value in diagnostics.items() if key != "w:0:r"}
    with pytest.raises(DataError, match="Complete graph diagnostics"):
        build_route_presentation(graph, route, missing, profile, evaluator.parameters)
    foreign = replace(route, edges=(replace(route.edges[0], tags={"highway": "steps"}),))
    with pytest.raises(DataError, match="supplied graph"):
        build_route_presentation(graph, foreign, diagnostics, profile, evaluator.parameters)


def _standard_route(graph, start, end, parameters):
    return find_route(graph, start, end, Evaluator(
        Profile(name="No preferences", weights={}, wheelchair_required=False, prefer_handrail=False), parameters
    ), distance_only=True)


def test_demo_comparison_has_exact_contract_and_stair_encounters_not_risers(graph, evaluator):
    selected = find_route(graph, "s", "t", evaluator)
    standard = _standard_route(graph, "s", "t", evaluator.parameters)
    diagnostics = edge_diagnostics(graph, selected, evaluator)
    result = build_route_comparison(graph, selected, standard, diagnostics, evaluator.profile, evaluator.parameters)
    assert set(result) == {"status", "same_endpoints", "same_path", "definition", "selected", "standard",
                           "obstacle_rows", "distance_difference_m", "distance_difference_percent",
                           "score_difference_pp", "limitations"}
    stat_keys = {"distance_m", "score_percent", "coverage_percent", "hard_block_violations", "conflict_count",
                 "unknown_count", "encounter_count", "obstacle_counts", "obstacles", "conflicts", "unknowns"}
    assert set(result["selected"]) == set(result["standard"]) == stat_keys
    assert result["status"] == "available" and result["same_endpoints"] and not result["same_path"]
    assert all(edge.way_id.startswith("smooth_") for edge in selected.edges)
    assert [edge.way_id for edge in standard.edges] == ["stairs", "stairs"]
    assert result["selected"]["obstacle_counts"]["stairs"] == 0
    assert result["standard"]["obstacle_counts"]["stairs"] == 1  # Not step_count=12 or two edges.
    assert result["standard"]["hard_block_violations"] == 2
    assert result["standard"]["score_percent"] is result["score_difference_pp"] is None
    assert result["standard"]["coverage_percent"] is not None
    assert [row["kind"] for row in result["obstacle_rows"]] == list(OBSTACLE_KINDS)
    for row in result["obstacle_rows"]:
        assert set(row) == {"kind", "label", "selected", "standard", "difference"}
        assert row["difference"] == row["selected"] - row["standard"]
    for stats in (result["selected"], result["standard"]):
        assert set(stats["obstacle_counts"]) == set(OBSTACLE_KINDS)
        assert stats["encounter_count"] == len(stats["obstacles"]) == sum(stats["obstacle_counts"].values())
        assert stats["conflict_count"] == len(stats["conflicts"])
        assert stats["unknown_count"] == len(stats["unknowns"])
    assert result["distance_difference_m"] == pytest.approx(selected.distance_m - standard.distance_m)
    assert result["distance_difference_percent"] == pytest.approx(
        100 * (selected.distance_m - standard.distance_m) / standard.distance_m)
    caveats = " ".join(result["limitations"])
    assert "multiple type rows" in caveats and "not globally invariant" in caveats
    assert "wheelchair hard requirement" in caveats and "matched hard constraints" in caveats
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("required", [False, True])
def test_comparison_wheelchair_only_soft_conflict_and_separate_hard_requirement(required):
    graph = _network([("stairs", ["s", "t"], {"highway": "steps", "wheelchair": "no"}),
                      ("smooth", ["s", "b", "t"], {"highway": "footway", "wheelchair": "yes"})])
    profile = Profile(weights={"wheelchair_accessible": 1}, wheelchair_required=required)
    selected, diagnostics, evaluator = _trace(graph, ["smooth:0:f", "smooth:1:f"], profile)
    standard = _standard_route(graph, "s", "t", evaluator.parameters)
    result = build_route_comparison(graph, selected, standard, diagnostics, profile, evaluator.parameters)
    baseline = result["standard"]
    assert [row["preference"] for row in baseline["conflicts"]] == ["wheelchair_accessible"]
    assert baseline["obstacle_counts"]["stairs"] == baseline["obstacle_counts"]["wheelchair_restriction"] == 1
    assert baseline["hard_block_violations"] == int(required)
    assert baseline["score_percent"] == (None if required else 0)
    assert baseline["coverage_percent"] == 100
    assert result["score_difference_pp"] == (None if required else 100)


@pytest.mark.parametrize("unknown_risk", [0, .65])
def test_feasible_comparison_uses_same_selected_weights_exposure_and_missingness(unknown_risk):
    graph = _network([("a", ["s", "a", "t"], {"highway": "footway", "lit": "no", "surface": "asphalt"}),
                      ("b", ["s", "b", "t"], {"highway": "footway", "lit": "yes"})],
                     node_tags={"b": {"kerb": "raised"}})
    profile = Profile(weights={"lit_roads": .25, "surface_type": 1, "short_kerbs": .75})
    parameters = CostParameters(unknown_risk=unknown_risk, event_equivalent_m=37)
    selected, diagnostics, evaluator = _trace(graph, ["a:0:f", "a:1:f"], profile, parameters)
    standard, _, no_preferences = _trace(graph, ["b:0:f", "b:1:f"], Profile(), parameters)
    assert route_summary(standard, no_preferences)["score_percent"] is None
    result = build_route_comparison(graph, selected, standard, diagnostics, profile, parameters)
    for key, route in (("selected", selected), ("standard", standard)):
        expected = route_summary(route, evaluator)
        assert result[key]["score_percent"] == pytest.approx(expected["score_percent"])
        assert result[key]["coverage_percent"] == pytest.approx(expected["coverage_percent"])
        view = build_route_presentation(graph, route, diagnostics, profile, parameters)
        assert result[key]["obstacles"] == view["encountered_obstacles"]
        assert result[key]["conflicts"] == view["conflicts"]
        assert result[key]["unknowns"] == view["unknowns"]
    assert result["standard"]["unknown_count"] == 1
    assert result["standard"]["conflict_count"] == 1
    assert result["score_difference_pp"] == pytest.approx(
        result["selected"]["score_percent"] - result["standard"]["score_percent"])


def test_comparison_same_path_zero_weights_is_not_perfect_accessibility():
    graph = _network([("w", ["a", "b"], {"highway": "steps"})])
    profile = Profile(weights={})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_comparison(graph, route, route, diagnostics, profile, evaluator.parameters)
    assert result["same_path"] and result["selected"] == result["standard"]
    assert result["selected"]["score_percent"] is result["selected"]["coverage_percent"] is None
    assert result["score_difference_pp"] is None
    assert result["distance_difference_m"] == result["distance_difference_percent"] == 0
    view = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    view["comparison"] = result
    report = render_route_summary(view)
    assert "**Same path:**" in report and "**No selected soft preferences:**" in report
    assert "unavailable, not 100% accessibility" in report


def test_comparison_identical_feasible_path_and_zero_distance_denominator():
    graph = _network([("w", ["a", "b"], {"highway": "footway", "lit": "no"})])
    profile = Profile(weights={"lit_roads": 1})
    route, diagnostics, evaluator = _trace(graph, ["w:0:f"], profile)
    result = build_route_comparison(graph, route, route, diagnostics, profile, evaluator.parameters)
    assert result["same_path"] and result["score_difference_pp"] == 0
    assert all(row["difference"] == 0 for row in result["obstacle_rows"])
    empty = Route(("a",), (), 0, 0)
    zero = build_route_comparison(graph, empty, empty, diagnostics, profile, evaluator.parameters)
    assert zero["distance_difference_m"] == 0
    assert zero["distance_difference_percent"] is zero["score_difference_pp"] is None
    view = build_route_presentation(graph, empty, diagnostics, profile, evaluator.parameters)
    view["comparison"] = zero
    assert "percentage not defined: zero standard distance" in render_route_summary(view)
    json.dumps(zero, allow_nan=False)


def test_comparison_same_path_requires_exact_directed_sequence_and_endpoints():
    graph = _network([("w", ["a", "b", "c"], {"highway": "steps"})])
    profile = Profile()
    route, diagnostics, evaluator = _trace(graph, ["w:0:f", "w:1:f"], profile)
    extra_visit, _, _ = _trace(graph, ["w:0:f", "w:0:r", "w:0:f", "w:1:f"], profile)
    result = build_route_comparison(graph, route, extra_visit, diagnostics, profile, evaluator.parameters)
    assert not result["same_path"]
    for sequence in (["w:1:r", "w:0:r"], ["w:0:f"]):
        other, _, _ = _trace(graph, sequence, profile)
        with pytest.raises(DataError, match="same snapped endpoints"):
            build_route_comparison(graph, route, other, diagnostics, profile, evaluator.parameters)


def test_comparison_displays_worsening_lighting_as_well_as_reduced_stairs():
    graph = _network([("stairs", ["s", "t"], {"highway": "steps", "lit": "yes"}),
                      ("smooth", ["s", "b", "t"], {"highway": "footway", "lit": "no"})])
    profile = Profile(weights={"avoid_stairs": 1, "lit_roads": .25})
    route, diagnostics, evaluator = _trace(graph, ["smooth:0:f", "smooth:1:f"], profile)
    standard = _standard_route(graph, "s", "t", evaluator.parameters)
    view = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                    baselines={"distance_only_same_constraints": {"distance_m": standard.distance_m}})
    view["comparison"] = build_route_comparison(graph, route, standard, diagnostics, profile, evaluator.parameters)
    rows = {row["kind"]: row for row in view["comparison"]["obstacle_rows"]}
    assert rows["stairs"]["difference"] == -1 and rows["lighting"]["difference"] == 1
    report = render_route_summary(view)
    assert report.index("## Selected vs") < report.index("## Selected preferences")
    assert "| Stairs encounters | 0 | 1 | -1 |" in report
    assert "| Lighting encounters | 1 | 0 | +1 |" in report
    assert "| Ramp encounters |" not in report
    assert "## Standard route — full selected-profile assessment" in report
    assert "Distance-only baseline under the same hard constraints:" in report
    assert "route distance overhead:" in report
    assert "Zero tagged stairs is not proof of no stairs" in report


@pytest.mark.parametrize("kinds", [None, (), ("stairs",), OBSTACLE_KINDS])
def test_comparison_is_independent_of_markers_and_does_not_copy_nested_comparison(graph, evaluator, kinds, monkeypatch):
    route = find_route(graph, "s", "t", evaluator)
    standard = _standard_route(graph, "s", "t", evaluator.parameters)
    diagnostics = edge_diagnostics(graph, route, evaluator)
    reference = build_route_comparison(graph, route, standard, diagnostics, evaluator.profile, evaluator.parameters)
    supplied = build_route_presentation(graph, route, diagnostics, evaluator.profile, evaluator.parameters,
                                        options=PresentationOptions(kinds, 0))
    supplied["comparison"] = supplied  # A recursive old value must never be traversed/copied.
    original = outputs.build_route_presentation
    calls = []

    def capture(*args, **kwargs):
        calls.append((args[1], kwargs["include_context"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(outputs, "build_route_presentation", capture)
    result = build_route_comparison(graph, route, standard, diagnostics, evaluator.profile, evaluator.parameters,
                                    selected_presentation=supplied)
    assert calls == [(standard, False)]
    assert result == reference
    assert supplied["comparison"] is supplied
    json.dumps(result, allow_nan=False)
    result["selected"]["obstacles"].clear()
    assert supplied["encountered_obstacles"]  # Returned records own their lists.


def test_comparison_node_visits_group_by_type_not_unique_physical_feature():
    graph = _network([("w", ["a", "b", "c"], {"highway": "footway"})], node_tags={
        "b": {"highway": "crossing", "crossing": "unmarked", "kerb": "raised"}})
    profile = Profile(weights={"short_kerbs": 1, "supervised_crossings": 1})
    selected, diagnostics, evaluator = _trace(graph, ["w:0:f", "w:1:f", "w:1:r", "w:1:f"], profile)
    standard, _, _ = _trace(graph, ["w:0:f", "w:1:f"], profile)
    result = build_route_comparison(graph, selected, standard, diagnostics, profile, evaluator.parameters)
    for key, count in (("selected", 2), ("standard", 1)):
        assert result[key]["obstacle_counts"]["kerb"] == result[key]["obstacle_counts"]["crossing"] == count
        assert result[key]["encounter_count"] == result[key]["conflict_count"] == count * 2
        assert {row["node_id"] for row in result[key]["obstacles"]} == {"b"}


@pytest.mark.parametrize("kinds", [None, (), OBSTACLE_KINDS])
def test_without_context_preserves_every_onroute_record_and_skips_offroute_analysis(kinds, monkeypatch):
    graph = _network([("on", ["a", "b", "c"], {"highway": "steps", "lit": "no"}),
                      ("off", ["c", "d", "e"], {"highway": "steps", "name": "OFF_ROUTE_ONLY"})],
                     node_tags={"a": {"barrier": "stile"}, "b": {"kerb": "raised"},
                                "d": {"barrier": "stile", "name": "OFF_ROUTE_ONLY"}})
    profile = Profile(weights={"lit_roads": 1, "surface_type": 1, "short_kerbs": 1}, wheelchair_required=True)
    route, diagnostics, evaluator = _trace(graph, ["on:0:f", "on:1:f"], profile)
    options = PresentationOptions(kinds, 0)
    full = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters, options=options)
    original = outputs._feature_details
    calls = []

    def guard(tags, record, profile, parameters, scope, edge=None):
        assert tags.get("name") != "OFF_ROUTE_ONLY"
        if edge is not None:
            assert edge.id in {item.id for item in route.edges}
        calls.append(scope)
        return original(tags, record, profile, parameters, scope, edge)

    monkeypatch.setattr(outputs, "_feature_details", guard)
    local = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters,
                                     options=options, include_context=False)
    assert calls == ["way", "way", "event", "event", "event"]
    for key in full.keys() - {"map_obstacles", "filter_summary"}:
        assert local[key] == full[key]
    assert local["map_obstacles"] == [row for row in full["map_obstacles"] if row["on_route"]]
    assert local["filter_summary"]["total_candidates"] < full["filter_summary"]["total_candidates"]
    assert local["validation"]["hard_constraint_violations"] == 2
    assert local["conflicts"] and local["unknowns"] and local["encountered_obstacles"]


def test_comparison_never_searches_or_reevaluates_and_masks_blocked_selected_score(monkeypatch):
    graph = _network([("w", ["a", "b", "c"], {"highway": "steps", "wheelchair": "no"})])
    profile = Profile(weights={"avoid_stairs": 1}, wheelchair_required=True)
    route, diagnostics, evaluator = _trace(graph, ["w:0:f", "w:1:f"], profile)
    view = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)

    def forbidden(*args, **kwargs):
        pytest.fail("Comparison must not search or evaluate")

    monkeypatch.setattr("osm_accessibility.routing.find_route", forbidden)
    monkeypatch.setattr(outputs, "Evaluator", forbidden)
    view["comparison"] = build_route_comparison(graph, route, route, diagnostics, profile, evaluator.parameters,
                                               selected_presentation=view)
    assert view["comparison"]["selected"]["score_percent"] is None
    report = render_route_summary(view)
    assert "infeasible under selected requirements" in report
    assert "**Optimized cost:** not available" in report
    assert "**The standard path is not recommended:**" in report
    assert "| Blocked directed traversals | 2 | 2 | +0 |" in report


def test_standard_route_full_tables_escape_osm_text_and_keep_unknowns_separate():
    hostile = '<script>alert(1)</script> | [link](javascript:evil)\n## injected'
    graph = _network([("good", ["s", "b", "t"], {"highway": "footway", "lit": "yes", "surface": "asphalt"}),
                      ("bad", ["s", "t"], {"highway": "footway", "lit": "no", "name": hostile, "surface": hostile})])
    profile = Profile(weights={"lit_roads": 1, "surface_type": 1})
    route, diagnostics, evaluator = _trace(graph, ["good:0:f", "good:1:f"], profile)
    standard = _standard_route(graph, "s", "t", evaluator.parameters)
    view = build_route_presentation(graph, route, diagnostics, profile, evaluator.parameters)
    view["comparison"] = build_route_comparison(graph, route, standard, diagnostics, profile, evaluator.parameters)
    report = render_route_summary(view)
    lower = report.split("## Standard route — full selected-profile assessment", 1)[1]
    assert "### Encountered obstacles (1 grouped encounters)" in lower
    assert "### Known selected soft conflicts (1 grouped entries)" in lower
    assert "### Missing data (1 grouped entries)" in lower
    assert "&lt;script&gt;" in lower and "<script>" not in report
    assert "[link](javascript:" not in report and "\n## injected" not in report
    assert "\\|" in lower
    obstacle_table = lower.split("### Known selected soft conflicts", 1)[0]
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 7
               for line in obstacle_table.splitlines() if line.startswith("|"))
