"""Reproducible route, diagnostic and provenance artifacts for research runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from .data import canonical_json
from .errors import DataError
from .evaluation import Evaluator, wheelchair_ramp
from .graph import InMemoryGraph
from .models import Route
from .tags import incline_percent, is_crossing


def graph_fingerprint(graph: InMemoryGraph) -> str:
    digest = hashlib.sha256()
    for node in graph.nodes.values():
        digest.update(canonical_json({"id": node.id, "coordinate": node.coordinate.as_lat_lon(), "tags": dict(node.tags)}).encode() + b"\n")
    for edge in graph.edges:
        record = {"id": edge.id, "source": edge.source, "target": edge.target, "way_id": edge.way_id,
                  "distance_m": edge.distance_m, "tags": dict(edge.tags), "source_tags": dict(edge.source_tags),
                  "target_tags": dict(edge.target_tags), "crossing_tags": dict(edge.crossing_tags),
                  "way_has_crossing_nodes": edge.way_has_crossing_nodes, "way_has_kerb_nodes": edge.way_has_kerb_nodes}
        digest.update(canonical_json(record).encode() + b"\n")
    return digest.hexdigest()


def route_summary(route: Route, evaluator: Evaluator) -> dict:
    evaluations = [evaluator(edge) for edge in route.edges]
    if any(item.blocked for item in evaluations):
        raise DataError("Route contains an edge blocked by the selected profile")
    exposure = sum(item.exposure_m for item in evaluations)
    way_risk = sum(item.way_risk_metres for item in evaluations)
    event_risk = sum(item.event_risk_metres for item in evaluations)
    known = sum(item.known_exposure_m for item in evaluations)
    score = 100 * max(0.0, min(1.0, 1 - (way_risk + event_risk) / exposure)) if exposure > 0 else None
    component = defaultdict(lambda: {"exposure_m": 0.0, "risk_metres": 0.0, "known_exposure_m": 0.0})
    for evaluation in evaluations:
        for criterion in evaluation.criteria:
            length = evaluation.length_m if criterion.scope == "way" else evaluation.event_equivalent_m
            item = component[f"{criterion.scope}:{criterion.name}"]
            item["exposure_m"] += length
            item["risk_metres"] += length * criterion.risk
            item["known_exposure_m"] += length if criterion.known else 0.0
    return {
        "node_ids": list(route.nodes), "edge_ids": [edge.id for edge in route.edges],
        "distance_m": route.distance_m, "optimized_cost_m": route.cost,
        "profile_cost_m": sum(evaluation.cost_m for evaluation in evaluations),
        "way_risk_metres": way_risk, "event_risk_metres": event_risk, "score_exposure_m": exposure,
        "known_exposure_m": known, "score_percent": score,
        "coverage_percent": 100 * min(1.0, max(0.0, known / exposure)) if exposure else None,
        "event_count": sum(item.event_weight > 0 for item in evaluations),
        "hard_block_violations": 0, "expanded_nodes": route.expanded_nodes, "settled_nodes": route.settled_nodes,
        "search_seconds": route.elapsed_seconds, "algorithm": route.algorithm,
        "criterion_exposure": dict(component),
    }


def edge_diagnostics(graph: InMemoryGraph, route: Route, evaluator: Evaluator) -> dict[str, dict]:
    route_edges = {edge.id for edge in route.edges}
    return {edge.id: evaluator(edge).as_dict() | {"source": edge.source, "target": edge.target,
            "way_id": edge.way_id, "on_route": edge.id in route_edges} for edge in graph.edges}


def collect_pois(graph: InMemoryGraph, route: Route, diagnostics: dict[str, dict]) -> list[dict]:
    """Marked obstacles are candidates to inspect, not necessarily avoidable POIs."""
    pois = []
    selected_nodes = set(route.nodes)
    selected_edges = {edge.id for edge in route.edges}
    for node in graph.nodes.values():
        tags = node.tags
        kinds = []
        if is_crossing(tags):
            kinds.append("crossing")
        if "kerb" in tags or "kerb:height" in tags:
            kinds.append("kerb")
        if tags.get("barrier") not in {None, "no"}:
            kinds.append("barrier")
        if tags.get("wheelchair") == "no":
            kinds.append("wheelchair_restriction")
        if tags.get("highway") == "steps":
            kinds.append("stairs")
        if tags.get("ramp") not in {None, "no"} or tags.get("ramp:wheelchair") == "yes":
            kinds.append("ramp")
        incident = (*graph.adjacency.get(node.id, ()), *graph.incoming.get(node.id, ()))
        blocked = all(diagnostics[edge.id]["blocked"] for edge in incident) if incident else None
        for kind in kinds:
            pois.append({"id": f"node:{node.id}:{kind}", "kind": kind,
                         "latitude": node.coordinate.latitude, "longitude": node.coordinate.longitude,
                         "on_route": node.id in selected_nodes, "blocked": blocked,
                         "description": canonical_json(dict(tags)), "osm_node_id": node.id, "osm_way_id": None})
    # Show every staircase/inclined/ramp segment, not a misleading average position
    # that could fall off a bent way. Reciprocal directions share a plotted POI.
    grouped: dict[tuple, list] = defaultdict(list)
    for edge in graph.edges:
        grouped[(edge.way_id, *sorted((edge.source, edge.target)))].append(edge)
    for _, edges in sorted(grouped.items()):
        edge = edges[0]
        tags = edge.tags
        kinds = []
        if tags.get("highway") == "steps":
            kinds.append("stairs")
        if wheelchair_ramp(tags) or tags.get("ramp") not in {None, "no"}:
            kinds.append("ramp")
        slope = incline_percent(tags.get("incline"))
        if (slope is not None and slope != 0) or tags.get("incline") in {"up", "down"}:
            kinds.append("incline")
        if tags.get("wheelchair") == "no":
            kinds.append("wheelchair_restriction")
        first, second = graph.nodes[edge.source].coordinate, graph.nodes[edge.target].coordinate
        for kind in kinds:
            pois.append({"id": f"way:{edge.id}:{kind}", "kind": kind,
                         "latitude": (first.latitude + second.latitude) / 2,
                         "longitude": (first.longitude + second.longitude) / 2,
                         "on_route": any(item.id in selected_edges for item in edges),
                         "blocked": all(diagnostics[item.id]["blocked"] for item in edges),
                         "description": canonical_json(dict(tags)), "osm_node_id": None, "osm_way_id": edge.way_id})
    return sorted(pois, key=lambda poi: poi["id"])


def gpx_text(graph: InMemoryGraph, route: Route, title: str = "Accessibility-aware route") -> str:
    root = Element("gpx", {"version": "1.1", "creator": "osm-accessibility-research",
                           "xmlns": "http://www.topografix.com/GPX/1/1"})
    metadata = SubElement(root, "metadata")
    SubElement(metadata, "name").text = title
    SubElement(metadata, "desc").text = "Mapped graph geometry only; endpoint snapping connections are not verified."
    segment = SubElement(SubElement(root, "trk"), "trkseg")
    for node_id in route.nodes:
        point = graph.nodes[node_id].coordinate
        SubElement(segment, "trkpt", {"lat": repr(point.latitude), "lon": repr(point.longitude)})
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="unicode")


def write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            # CSV annotations may be opened in spreadsheet programs; stop formula
            # interpretation without changing the raw JSON diagnostic artifact.
            row = {key: ("'" + value if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"} else value)
                   for key, value in raw.items()}
            writer.writerow(row)


def save_diagnostics(output: Path, graph: InMemoryGraph, route: Route, diagnostics: dict[str, dict], pois: list[dict]) -> None:
    write_json(output / "edge_diagnostics.json", diagnostics)
    write_json(output / "pois.json", pois)
    rows = [{**item, "hard_blocks": "; ".join(item["hard_blocks"])} for item in diagnostics.values()]
    fields = ["edge_id", "source", "target", "way_id", "on_route", "blocked", "length_m", "score", "way_score",
              "coverage", "risk", "transition_risk", "way_weight", "event_weight", "way_known_weight",
              "event_known_weight", "way_risk_metres", "event_risk_metres", "exposure_m", "penalty_m", "cost_m", "hard_blocks"]
    _write_csv(output / "all_edges.csv", rows, fields)
    segments = [{"sequence": i, **diagnostics[edge.id], "hard_blocks": "; ".join(diagnostics[edge.id]["hard_blocks"])}
                for i, edge in enumerate(route.edges, 1)]
    _write_csv(output / "route_segments.csv", segments, ["sequence", *fields])
    component_rows = [{"edge_id": edge_id, "on_route": item["on_route"], **criterion}
                      for edge_id, item in diagnostics.items() for criterion in item["criteria"]]
    _write_csv(output / "criteria.csv", component_rows, ["edge_id", "on_route", "name", "scope", "weight", "risk", "known", "reason"])
    _write_csv(output / "pois.csv", pois, ["id", "kind", "latitude", "longitude", "on_route", "blocked", "osm_node_id", "osm_way_id", "description"])
    features = []
    for edge in route.edges:
        properties = {key: value for key, value in diagnostics[edge.id].items() if key != "criteria"}
        features.append({"type": "Feature", "id": edge.id, "properties": properties,
                         "geometry": {"type": "LineString", "coordinates": [graph.nodes[edge.source].coordinate.as_geojson(), graph.nodes[edge.target].coordinate.as_geojson()]}})
    write_json(output / "route.geojson", {"type": "FeatureCollection", "features": features})
    with (output / "route.gpx").open("x", encoding="utf-8") as stream:
        stream.write(gpx_text(graph, route))