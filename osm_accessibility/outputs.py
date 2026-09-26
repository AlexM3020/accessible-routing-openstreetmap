"""Reproducible route, diagnostic and provenance artifacts for research runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from html import escape
from math import isclose, isfinite
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from .data import canonical_json
from .errors import DataError
from .evaluation import POOR_SMOOTHNESS, POOR_SURFACES, Evaluator, wheelchair_ramp
from .graph import InMemoryGraph
from .models import Edge, Route
from .profiles import CostParameters, Profile
from .tags import incline_percent, is_crossing, metres

OBSTACLE_KINDS = (
    "stairs", "barrier", "wheelchair_restriction", "kerb", "crossing", "incline",
    "ramp", "surface", "smoothness", "width", "lighting", "bicycles",
)
OBSTACLE_LABELS = {
    "stairs": "Stairs", "barrier": "Barrier", "wheelchair_restriction": "Wheelchair restriction",
    "kerb": "Kerb", "crossing": "Crossing", "incline": "Incline", "ramp": "Ramp",
    "surface": "Surface", "smoothness": "Smoothness", "width": "Width",
    "lighting": "Lighting", "bicycles": "Bicycle traffic",
}
PREFERENCE_LABELS = {
    "avoid_bicycles": "Avoid bicycle traffic", "lit_roads": "Lit paths",
    "surface_type": "Suitable surfaces", "smoothness": "Smooth paths",
    "short_kerbs": "Low kerbs", "supervised_crossings": "Controlled crossings",
    "road_width": "Minimum path width", "road_incline": "Gentle inclines",
    "avoid_stairs": "Avoid stairs", "wheelchair_accessible": "Mapped wheelchair access",
}
_CRITERION_KINDS = {
    "avoid_bicycles": "bicycles", "lit_roads": "lighting", "surface_type": "surface",
    "smoothness": "smoothness", "short_kerbs": "kerb", "supervised_crossings": "crossing",
    "road_width": "width", "road_incline": "incline", "avoid_stairs": "stairs",
    "wheelchair_accessible": "wheelchair_restriction",
}
_PHYSICAL_BARRIERS = {"wall", "fence", "hedge", "retaining_wall", "block"}
_WHEELCHAIR_BARRIERS = {"stile", "turnstile", "kissing_gate", "cycle_barrier"}
_NODE_KINDS = {"stairs", "barrier", "wheelchair_restriction", "kerb", "crossing", "ramp"}
_MAP_FIELDS = (
    "id", "kind", "latitude", "longitude", "on_route", "status", "description",
    "preference_labels", "edge_ids", "way_id", "node_id", "first_segment",
)


@dataclass(frozen=True)
class PresentationOptions:
    """Marker-only settings; an empty tuple hides all types, unlike automatic None."""

    obstacle_kinds: tuple[str, ...] | None = None
    max_markers: int = 12

    def __post_init__(self) -> None:
        kinds = self.obstacle_kinds
        if kinds is not None:
            if not isinstance(kinds, tuple) or any(
                not isinstance(kind, str) or kind not in OBSTACLE_KINDS for kind in kinds
            ):
                raise DataError("obstacle_kinds must be a tuple of supported obstacle kind strings or None")
            if len(set(kinds)) != len(kinds):
                raise DataError("obstacle_kinds must not contain duplicates")
        if (isinstance(self.max_markers, bool) or not isinstance(self.max_markers, int)
                or not 0 <= self.max_markers <= 100):
            raise DataError("max_markers must be an integer between 0 and 100")

    def as_dict(self) -> dict:
        return {"obstacle_kinds": None if self.obstacle_kinds is None else list(self.obstacle_kinds),
                "max_markers": self.max_markers}

    @classmethod
    def from_dict(cls, raw: Mapping) -> PresentationOptions:
        if not isinstance(raw, Mapping) or set(raw) - {"obstacle_kinds", "max_markers"}:
            raise DataError("Invalid presentation option fields; expected an object")
        kinds = raw.get("obstacle_kinds")
        if kinds is not None:
            if not isinstance(kinds, (list, tuple)):
                raise DataError("obstacle_kinds must be an array or None")
            kinds = tuple(kinds)
        return cls(obstacle_kinds=kinds, max_markers=raw.get("max_markers", 12))


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


def _importance(weight: float) -> str:
    if weight >= 1:
        return "Essential"
    if weight >= .75:
        return "Very high"
    if weight >= .5:
        return "Important"
    return "A little important" if weight > 0 else "Not selected"


def _requirements(profile: Profile) -> list[str]:
    wheelchair = (
        "Wheelchair required: true (hard model rule). Rejects wheelchair=no on ways or endpoints, "
        "wheelchair barriers without wheelchair=yes, and steps without an accepted wheelchair ramp."
        if profile.wheelchair_required else
        "Wheelchair required: false. No wheelchair-specific hard exclusions; the wheelchair preference "
        "weight, if selected, is a separate soft criterion."
    )
    ramps = (
        "Ramps allowed: true. On steps, the model accepts ramp:wheelchair=yes, or ramp=yes or "
        "ramp=wheelchair together with wheelchair=yes; ramp:wheelchair=no overrides these tags. This can relax the "
        "wheelchair steps exclusion and reduce selected stair risk, not certify a usable ramp."
        if profile.allow_ramps else
        "Ramps allowed: false. The ramp concession for stairs is disabled; this is not a blanket "
        "hard ban on all paths with ramps."
    )
    return [
        wheelchair, ramps,
        f"Handrail preferred: {str(profile.prefer_handrail).lower()}. Soft stair rule only: when this "
        "preference is enabled and avoid-stairs is selected, unconfirmed handrails can increase stair risk "
        "(capped at full risk); "
        "handrails are not a hard requirement.",
        f"Minimum width: {profile.min_width_m:g} m. A soft threshold only when the width preference "
        "is selected; missing width remains unknown, not a verified clearance or a hard exclusion.",
        "Pedestrian policy (hard): only supported pedestrian highways, or cycleways with explicit "
        "permitted foot access. Excluded highway types remain blocked. Restricted foot access, or "
        "restricted access without a permitted foot override, blocks ways and endpoints.",
        "Barrier policy (hard): wall, fence, hedge, retaining_wall and block are excluded. With the "
        "wheelchair requirement, stile, turnstile, kissing_gate and cycle_barrier also need "
        "wheelchair=yes. All these rules describe the model, not physical safety guarantees.",
    ]


def _tag_text(tags: Mapping[str, str], keys: tuple[str, ...]) -> str:
    return "; ".join(f"{key}={tags[key]}" for key in keys if key in tags) or (
        "/".join(keys) + " not recorded"
    )


def _observation(kind: str, tags: Mapping[str, str], profile: Profile,
                 parameters: CostParameters) -> str:
    """Plain text, not markup or raw JSON; keep measurement keys and original units."""
    if kind == "stairs":
        parts = ["Mapped stairs (highway=steps)"]
        count = tags.get("step_count", "")
        # A zero, range, noninteger or missing count is not an exact observed count.
        if count.isascii() and count.isdigit() and any(digit != "0" for digit in count):
            parts.append(f"step_count={count} for the tagged feature, not per route segment")
        else:
            parts.append("step count unknown" + (f" (step_count={count})" if count else ""))
        parts.append(_tag_text(tags, ("ramp", "ramp:wheelchair", "handrail")))
        if not wheelchair_ramp(tags):
            parts.append("an explicitly usable wheelchair ramp is not confirmed by these tags")
        elif not profile.allow_ramps:
            parts.append("the profile disables the ramp concession for stairs")
        if (profile.prefer_handrail and profile.weights["avoid_stairs"] > 0
                and tags.get("handrail") not in {"yes", "left", "right", "both"}):
            parts.append("the handrail preference is not satisfied by the tags; this is a soft stair rule")
        return "; ".join(parts)
    if kind == "width":
        width = metres(tags.get("width"))
        measurement = f" ({width:g} m)" if width is not None else " (width in metres unknown)"
        return (f"Width: {_tag_text(tags, ('width',))}{measurement}; "
                f"selected minimum {profile.min_width_m:g} m is a soft threshold")
    if kind == "incline":
        incline = incline_percent(tags.get("incline"))
        measurement = (f" ({incline:g}% grade); model uses absolute grade, full risk at "
                       f"{parameters.incline_full_risk_percent:g}%" if incline is not None else
                       "; slope magnitude unknown, not evidence of a measured steep incline")
        return "Incline: " + _tag_text(tags, ("incline",)) + measurement
    if kind == "kerb":
        height = metres(tags.get("kerb:height"))
        measurement = f" ({height:g} m height)" if height is not None else ""
        return ("Kerb: " + _tag_text(tags, ("kerb", "kerb:height")) + measurement
                + f"; model full risk at {parameters.kerb_full_risk_m:g} m")
    keys = {
        "barrier": ("barrier", "wheelchair", "access", "foot"),
        "wheelchair_restriction": ("wheelchair",),
        "crossing": ("highway", "footway", "crossing", "crossing:signals"),
        "ramp": ("ramp", "ramp:wheelchair", "ramp:bicycle", "wheelchair"),
        "surface": ("surface", "tracktype"), "smoothness": ("smoothness",),
        "lighting": ("lit",), "bicycles": ("bicycle", "segregated"),
    }
    text = f"{OBSTACLE_LABELS[kind]}: {_tag_text(tags, keys[kind])}"
    if kind == "ramp":
        text += "; a mapped ramp is not a guarantee of wheelchair usability"
    return text


def _selected_criteria(record: dict, profile: Profile, scope: str | None = None) -> list[dict]:
    return [criterion for criterion in record["criteria"]
            if profile.weights.get(criterion["name"], 0) > 0 and criterion["weight"] > 0
            and (scope is None or criterion["scope"] == scope)]


def _segment_text(segments: list[int]) -> str:
    numbers = sorted(set(segments))
    if not numbers:
        return "route start (no traversed segment)"
    ranges = []
    first = last = numbers[0]
    for number in numbers[1:]:
        if number == last + 1:
            last = number
        else:
            ranges.append(str(first) if first == last else f"{first}–{last}")
            first = last = number
    ranges.append(str(first) if first == last else f"{first}–{last}")
    return f"route segment{'s' if len(numbers) != 1 else ''} " + ", ".join(ranges)


def _location(way_id: str | None, node_id: str | None, name: str | None,
              segments: list[int], on_route: bool = True) -> str:
    owner = f"OSM node {node_id}" if node_id is not None else f"OSM way {way_id}"
    if name:
        owner += f" ({name})"
    return owner + "; " + (_segment_text(segments) if on_route else "off route")


def _route_issues(route: Route, diagnostics: dict[str, dict], profile: Profile,
                  parameters: CostParameters) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    previous: dict[tuple, dict] = {}
    for segment, edge in enumerate(route.edges, 1):
        record = diagnostics[edge.id]
        for criterion in _selected_criteria(record, profile):
            if criterion["known"] is False:
                status = "unknown"  # Even when the configured unknown risk is zero.
            elif criterion["known"] is True and criterion["risk"] > 0:
                status = "conflict"
            else:
                continue
            event = criterion["scope"] == "event"
            tags = (edge.target_tags or edge.crossing_tags) if event else edge.tags
            owner = edge.target if event else edge.way_id
            key = (criterion["scope"], owner, criterion["name"], status)
            row = previous.get(key)
            if row is None or row["_segments"][-1] != segment - 1:
                row = {
                    "id": f"{criterion['scope']}:{owner}:{criterion['name']}:{status}:{segment}",
                    "kind": _CRITERION_KINDS[criterion["name"]],
                    "label": OBSTACLE_LABELS[_CRITERION_KINDS[criterion["name"]]],
                    "preference": criterion["name"], "weight": float(criterion["weight"]),
                    "status": status, "way_id": None if event else edge.way_id,
                    "node_id": edge.target if event else None, "edge_ids": [], "distance_m": 0.0,
                    "_segments": [], "_risk_metres": 0.0, "_max_risk": 0.0, "_reasons": [],
                    "_name": tags.get("name"),
                }
                rows.append(row)
                previous[key] = row
            length = 0.0 if event else float(record["length_m"])
            risk = float(criterion["risk"])
            row["edge_ids"].append(edge.id)  # Preserve traversal order, including repeated edges.
            row["_segments"].append(segment)
            row["distance_m"] += length
            row["_risk_metres"] += length * risk
            row["_max_risk"] = max(row["_max_risk"], risk)
            reason = _observation(row["kind"], tags, profile, parameters)
            if status == "unknown":
                reason += (f"; missing or uninterpretable data, configured unknown risk={risk:g}; "
                           "not an observed obstacle or a hard violation")
            if criterion.get("reason"):
                reason += f"; model observation: {criterion['reason']}"
            row["_reasons"].append(reason)
    result = []
    for row in rows:
        result.append({key: value for key, value in row.items() if not key.startswith("_")} | {
            "location": _location(row["way_id"], row["node_id"], row["_name"], row["_segments"]),
            "risk": (row["_risk_metres"] / row["distance_m"] if row["distance_m"] else row["_max_risk"]),
            "reason": "; ".join(dict.fromkeys(row["_reasons"])),
        })
    return ([row for row in result if row["status"] == "conflict"],
            [row for row in result if row["status"] == "unknown"])


def _tag_features(tags: Mapping[str, str], profile: Profile) -> dict[str, tuple[bool, bool]]:
    """Known features -> (automatically important, adverse); not an evaluator.

    Neutral, explicitly selectable features stay out of the obstacle report.
    Qualitative/unknown measurements never manufacture an adverse feature.
    """
    features = {}
    if tags.get("highway") == "steps":
        features["stairs"] = (True, True)
    barrier = tags.get("barrier")
    if barrier not in {None, "", "no", "none", "unknown", "kerb"}:
        important = barrier in _PHYSICAL_BARRIERS or (
            profile.wheelchair_required and barrier in _WHEELCHAIR_BARRIERS
            and tags.get("wheelchair") != "yes"
        )
        features["barrier"] = (important, important)
    wheelchair = tags.get("wheelchair")
    if wheelchair in {"no", "limited"}:
        features["wheelchair_restriction"] = (wheelchair == "no", True)
    if tags.get("ramp") in {"yes", "wheelchair", "bicycle", "stroller", "separate"} or wheelchair_ramp(tags):
        features["ramp"] = (False, False)
    height = metres(tags.get("kerb:height"))
    kerb = tags.get("kerb")
    if height is not None or kerb in {"raised", "rolled", "stepped", "lowered", "flush"}:
        adverse = height > 0 if height is not None else kerb in {"raised", "rolled", "stepped"}
        features["kerb"] = (False, adverse)
    if is_crossing(tags):
        adverse = tags.get("crossing:signals") != "yes" and tags.get("crossing") in {
            "marked", "zebra", "uncontrolled", "unmarked", "unsupervised",
        }
        features["crossing"] = (False, adverse)
    slope = incline_percent(tags.get("incline"))
    if slope is not None and slope != 0:
        features["incline"] = (False, True)
    if tags.get("surface") in POOR_SURFACES or tags.get("tracktype") in {"grade3", "grade4", "grade5"}:
        features["surface"] = (False, True)
    if tags.get("smoothness") in POOR_SMOOTHNESS | {"intermediate"}:
        features["smoothness"] = (False, True)
    width = metres(tags.get("width"))
    if width is not None and width < profile.min_width_m:
        features["width"] = (False, True)
    if tags.get("lit") == "no":
        features["lighting"] = (False, True)
    if (tags.get("bicycle") in {"yes", "designated", "permissive", "destination"}
            and tags.get("segregated") != "yes"):
        features["bicycles"] = (False, True)
    return features


def _feature_details(tags: Mapping[str, str], record: dict, profile: Profile,
                     parameters: CostParameters, scope: str, edge: Edge | None = None) -> dict[str, dict]:
    features = _tag_features(tags, profile)
    if scope == "event":
        features = {kind: flags for kind, flags in features.items() if kind in _NODE_KINDS}
    elif edge is not None:
        event = edge.target_tags or edge.crossing_tags
        if edge.way_has_crossing_nodes or is_crossing(event):
            features.pop("crossing", None)
        if edge.way_has_kerb_nodes or is_crossing(event) or "kerb" in event or "kerb:height" in event:
            features.pop("kerb", None)
    labels: dict[str, list[str]] = defaultdict(list)
    for criterion in _selected_criteria(record, profile, scope):
        if criterion["known"] is True and criterion["risk"] > 0:
            kind = _CRITERION_KINDS[criterion["name"]]
            features[kind] = (True, True)
            labels[kind].append(
                f"{PREFERENCE_LABELS[criterion['name']]} ({_importance(criterion['weight'])})"
            )
    return {kind: {"automatic": flags[0], "adverse": flags[1],
                   "description": _observation(kind, tags, profile, parameters),
                   "preference_labels": labels[kind]}
            for kind, flags in features.items()}


def _new_obstacle(kind: str, way_id: str | None, node_id: str | None,
                  on_route: bool, name: str | None) -> dict:
    return {"kind": kind, "way_id": way_id, "node_id": node_id, "on_route": on_route,
            "_name": name, "_edges": [], "_geometry": [], "_segments": [], "_last_index": -2,
            "_descriptions": [], "_preferences": [], "_automatic": False, "_adverse": False}


def _include_feature(group: dict, feature: dict, edges: list[Edge], geometry: list[Edge],
                     segment: int | None = None, visit: int = -2) -> None:
    group["_edges"].extend(edges)
    group["_geometry"].extend(geometry)
    if segment is not None:
        group["_segments"].append(segment)
    group["_last_index"] = visit
    group["_descriptions"].append(feature["description"])
    group["_preferences"].extend(feature["preference_labels"])
    group["_automatic"] |= feature["automatic"]
    group["_adverse"] |= feature["adverse"]


def _geometry_key(edge: Edge) -> tuple[str, str, str]:
    return edge.way_id, min(edge.source, edge.target), max(edge.source, edge.target)


def _representative_point(graph: InMemoryGraph, edges: list[Edge]) -> tuple[float, float]:
    """Length midpoint on a real segment, never a centroid of a bent way.

    On-route edges are the exact ordered encounter, including repeated traversals.
    Off-route geometry is a deterministic list with reciprocal directions removed;
    no connecting geometry between branches is invented.
    """
    remaining = sum(edge.distance_m for edge in edges) / 2
    for edge in edges:
        if remaining <= edge.distance_m:
            break
        remaining -= edge.distance_m
    else:
        edge = edges[-1]
        remaining = edge.distance_m
    fraction = remaining / edge.distance_m if edge.distance_m else .5
    first, second = graph.nodes[edge.source].coordinate, graph.nodes[edge.target].coordinate
    delta_lon = (second.longitude - first.longitude + 180) % 360 - 180
    return (float(first.latitude + fraction * (second.latitude - first.latitude)),
            float((first.longitude + fraction * delta_lon + 180) % 360 - 180))


def _finish_obstacle(graph: InMemoryGraph, group: dict, diagnostics: dict[str, dict]) -> dict:
    edges = group["_edges"]
    preferences = list(dict.fromkeys(group["_preferences"]))
    if group["on_route"]:
        blocked = any(diagnostics[edge.id]["blocked"] for edge in edges)
        status = "blocked" if blocked else "conflict" if preferences else "encountered"
    else:
        blocked = bool(edges) and all(diagnostics[edge.id]["blocked"] for edge in edges)
        status = "blocked" if blocked else "context"
    if group["node_id"] is not None:
        point = graph.nodes[group["node_id"]].coordinate
        latitude, longitude = point.latitude, point.longitude
    else:
        latitude, longitude = _representative_point(graph, group["_geometry"])
    first_segment = min(group["_segments"]) if group["_segments"] else None
    owner = f"node:{group['node_id']}" if group["node_id"] is not None else f"way:{group['way_id']}"
    encounter = f"route:{first_segment or 0}" if group["on_route"] else "offroute"
    edge_ids = [edge.id for edge in edges]
    if group["node_id"] is not None or not group["on_route"]:
        edge_ids = list(dict.fromkeys(edge_ids))
    return {
        "id": f"{owner}:{group['kind']}:{encounter}", "kind": group["kind"],
        "latitude": float(latitude), "longitude": float(longitude), "on_route": group["on_route"],
        "status": status, "description": "; ".join(dict.fromkeys(group["_descriptions"])),
        "preference_labels": preferences, "edge_ids": edge_ids,
        "way_id": group["way_id"], "node_id": group["node_id"], "first_segment": first_segment,
        "label": OBSTACLE_LABELS[group["kind"]],
        "location": _location(group["way_id"], group["node_id"], group["_name"],
                              group["_segments"], group["on_route"]),
        "distance_m": float(sum(edge.distance_m for edge in group["_geometry"])),
        "_automatic": group["_automatic"], "_adverse": group["_adverse"],
    }


def _obstacle_candidates(graph: InMemoryGraph, route: Route, diagnostics: dict[str, dict],
                         profile: Profile, parameters: CostParameters, *,
                         include_context: bool = True) -> list[dict]:
    groups: list[dict] = []
    previous: dict[tuple, dict] = {}
    for segment, edge in enumerate(route.edges, 1):
        for kind, feature in _feature_details(
            edge.tags, diagnostics[edge.id], profile, parameters, "way", edge
        ).items():
            key = (edge.way_id, kind)
            group = previous.get(key)
            if group is None or group["_last_index"] != segment - 1:
                group = _new_obstacle(kind, edge.way_id, None, True, edge.tags.get("name"))
                previous[key] = group
                groups.append(group)
            _include_feature(group, feature, [edge], [edge], segment, segment)

    # A node criterion belongs to arrival, never all incident/reverse directions.
    # Include physical start-node features without inventing a starting event cost.
    previous = {}
    for visit, node_id in enumerate(route.nodes):
        arrival = route.edges[visit - 1] if visit else None
        source = route.edges[0] if route.edges and not visit else None
        tags = ((arrival.target_tags or arrival.crossing_tags) if arrival else
                source.source_tags if source else {}) or graph.nodes[node_id].tags
        record = diagnostics[arrival.id] if arrival else {"criteria": []}
        incident = [arrival] if arrival else [source] if source else []
        segment = visit if arrival else 1 if source else None
        for kind, feature in _feature_details(tags, record, profile, parameters, "event").items():
            key = (node_id, kind)
            group = previous.get(key)
            if group is None or group["_last_index"] != visit - 1:
                group = _new_obstacle(kind, None, node_id, True, tags.get("name"))
                previous[key] = group
                groups.append(group)
            _include_feature(group, feature, incident, [], segment, visit)

    if not include_context:
        return _finish_candidates(graph, groups, diagnostics)

    # Do not turn a blocked reciprocal of a selected segment into an on-route
    # violation or a second obstacle. Context covers only untraversed geometry.
    used_geometry = {_geometry_key(edge) for edge in route.edges}
    equivalent: dict[tuple, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        if _geometry_key(edge) not in used_geometry:
            equivalent[_geometry_key(edge)].append(edge)
    offroute: dict[tuple, dict] = {}
    for _, edges in sorted(equivalent.items()):
        features: dict[str, list[tuple[Edge, dict]]] = defaultdict(list)
        for edge in edges:
            for kind, feature in _feature_details(
                edge.tags, diagnostics[edge.id], profile, parameters, "way", edge
            ).items():
                features[kind].append((edge, feature))
        for kind, matches in features.items():
            representative = matches[0][0]
            key = (representative.way_id, kind)
            if key not in offroute:
                offroute[key] = _new_obstacle(kind, representative.way_id, None, False,
                                              representative.tags.get("name"))
                groups.append(offroute[key])
            for index, (edge, feature) in enumerate(matches):
                # All equivalent directions affect blocked status, even when only
                # one direction contains an adverse selected-criterion observation.
                _include_feature(offroute[key], feature, edges if index == 0 else [],
                                 [edge] if index == 0 else [])

    route_nodes = set(route.nodes)
    for node in graph.nodes.values():
        if node.id in route_nodes:
            continue
        incoming = graph.incoming.get(node.id, ())
        outgoing = graph.adjacency.get(node.id, ())
        incident = list({edge.id: edge for edge in (*incoming, *outgoing)}.values())
        observations = [(node.tags, {"criteria": []})]
        observations.extend((edge.target_tags or edge.crossing_tags or node.tags, diagnostics[edge.id])
                            for edge in incoming)
        observations.extend((edge.source_tags or node.tags, {"criteria": []}) for edge in outgoing)
        node_groups: dict[str, dict] = {}
        for tags, record in observations:
            for kind, feature in _feature_details(tags, record, profile, parameters, "event").items():
                if kind not in node_groups:
                    node_groups[kind] = _new_obstacle(kind, None, node.id, False, tags.get("name"))
                    groups.append(node_groups[kind])
                _include_feature(node_groups[kind], feature, incident if not node_groups[kind]["_edges"] else [], [])
    return _finish_candidates(graph, groups, diagnostics)


def _finish_candidates(graph: InMemoryGraph, groups: list[dict],
                       diagnostics: dict[str, dict]) -> list[dict]:
    candidates = [_finish_obstacle(graph, group, diagnostics) for group in groups]
    return sorted(candidates, key=lambda row: (
        not row["on_route"], {"blocked": 0, "conflict": 1, "encountered": 2, "context": 3}[row["status"]],
        row["first_segment"] if row["first_segment"] is not None else float("inf"),
        OBSTACLE_KINDS.index(row["kind"]), row["id"],
    ))


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if isfinite(number) and number >= 0 else None


def _baseline_number(baselines: dict, label: str, field: str) -> float | None:
    record = baselines.get(label)
    return _nonnegative_number(record.get(field)) if isinstance(record, Mapping) else None


def build_route_presentation(
    graph: InMemoryGraph, route: Route, diagnostics: dict[str, dict], profile: Profile,
    parameters: CostParameters, *, options: PresentationOptions | None = None,
    baselines: dict | None = None, snapping: dict | None = None, synthetic: bool = False,
    include_context: bool = True,
) -> dict:
    """Build a complete, output-only view; never reroute or reevaluate the graph.

    Diagnostics must be evaluator.as_dict() records for this graph/profile/cost
    configuration. Risk/exposure pooling matches route_summary; cost_m is the
    supplied route's optimized cost. Blocked submitted routes are reported, not
    silently dropped. Row locations use 1-based segment ranges. Way-row risk is
    length-weighted; contiguous repeated node arrivals count as one event, using
    their maximum risk. Nonconsecutive visits remain separate encounters.

    Baselines use compare_baselines' astar/dijkstra.optimized_cost_m and
    distance_only_same_constraints.distance_m. The optimal-cost difference is
    absolute A* minus Dijkstra; no searches or provenance checks are performed.

    encountered_obstacles contains every known adverse on-route feature, including
    inactive-preference encounters, independent of display settings. Neutral ramps,
    controlled crossings and flush/lowered kerbs are explicit-only map context.
    map_obstacles is sorted but NOT capped: the drawing consumer applies max_markers.
    selected_kinds=None means automatic important/selected-adverse markers; a tuple
    is solely a type filter over known features, not an additional relevance filter.
    include_context=False skips off-route edge/node feature analysis, preserving
    all on-route records and markers. Complete graph diagnostics are still required.
    """
    if options is None:
        options = PresentationOptions()
    if not isinstance(options, PresentationOptions):
        raise DataError("options must be PresentationOptions or None")
    edge_map = {edge.id: edge for edge in graph.edges}
    if any(node not in graph.nodes for node in route.nodes) or any(
        edge_map.get(edge.id) != edge for edge in route.edges
    ):
        raise DataError("The selected route must belong to the supplied graph")
    if any(edge.id not in diagnostics for edge in graph.edges):
        raise DataError("Complete graph diagnostics are required for the presentation")
    records = [diagnostics[edge.id] for edge in route.edges]
    exposure = sum(float(record["exposure_m"]) for record in records)
    risk = sum(float(record["way_risk_metres"]) for record in records) + sum(
        float(record["event_risk_metres"]) for record in records
    )
    known = sum(float(record["known_exposure_m"]) for record in records)
    conflicts, unknowns = _route_issues(route, diagnostics, profile, parameters)
    candidates = _obstacle_candidates(graph, route, diagnostics, profile, parameters,
                                      include_context=include_context)
    selected = [row for row in candidates if (
        row["_automatic"] if options.obstacle_kinds is None else row["kind"] in options.obstacle_kinds
    )]
    if options.obstacle_kinds is None:
        # Share a physical symbol only for the exact same encounter/geometry.
        # Keep its strongest warning and all selected preference labels. Copies
        # ensure this visual consolidation never changes the full report below.
        def owner(row):
            return (row["way_id"], row["node_id"], row["on_route"], row["first_segment"], tuple(row["edge_ids"]))

        selected = [dict(row) for row in selected]
        physical = {owner(row): row for row in selected if row["kind"] in {"stairs", "barrier"}}
        retained = []
        for row in selected:
            primary = physical.get(owner(row)) if row["kind"] == "wheelchair_restriction" else None
            if primary is None:
                retained.append(row)
                continue
            primary["preference_labels"] = list(dict.fromkeys(primary["preference_labels"] + row["preference_labels"]))
            primary["description"] += "; " + row["description"]
            if row["status"] == "blocked" or row["status"] == "conflict" and primary["status"] != "blocked":
                primary["status"] = row["status"]
        selected = retained
    encountered = []
    for row in candidates:
        if row["on_route"] and row["_adverse"]:
            classification = ("hard_constraint_violation" if row["status"] == "blocked" else
                              "preference_conflict" if row["status"] == "conflict" else "encountered")
            encountered.append(deepcopy({key: value for key, value in row.items() if not key.startswith("_")})
                               | {"classification": classification})
    baselines = baselines or {}
    astar = _baseline_number(baselines, "astar", "optimized_cost_m")
    dijkstra = _baseline_number(baselines, "dijkstra", "optimized_cost_m")
    if astar is not None and dijkstra is not None:
        difference = abs(astar - dijkstra)
        agree = isclose(astar, dijkstra, rel_tol=1e-10, abs_tol=1e-6)
    else:
        difference = agree = None
    baseline_distance = _baseline_number(baselines, "distance_only_same_constraints", "distance_m")
    overhead = None
    if baseline_distance is not None:
        if baseline_distance > 0:
            overhead = 100 * (route.distance_m - baseline_distance) / baseline_distance
        elif route.distance_m == 0:
            overhead = 0.0
    return {
        "profile_name": profile.name,
        "selected_preferences": [{"name": name, "label": PREFERENCE_LABELS[name], "weight": float(weight),
                                  "importance": _importance(weight)}
                                 for name, weight in profile.weights.items() if weight > 0],
        "requirements": _requirements(profile),
        "metrics": {"distance_m": float(route.distance_m),
                    "score_percent": 100 * max(0.0, min(1.0, 1 - risk / exposure)) if exposure > 0 else None,
                    "coverage_percent": 100 * min(1.0, max(0.0, known / exposure)) if exposure else None,
                    "cost_m": float(route.cost)},
        "conflicts": conflicts, "unknowns": unknowns, "encountered_obstacles": encountered,
        "map_obstacles": [deepcopy({key: row[key] for key in _MAP_FIELDS}) for row in selected],
        "filter_summary": {"selected_kinds": options.as_dict()["obstacle_kinds"],
                           "total_candidates": len(candidates), "selected_candidates": len(selected),
                           "omitted_by_filter": len(candidates) - len(selected), "max_markers": options.max_markers},
        "validation": {"hard_constraint_violations": sum(record["blocked"] is True for record in records),
                       "astar_dijkstra_agree": agree, "optimal_cost_difference_m": difference,
                       "distance_baseline_m": baseline_distance, "distance_overhead_percent": overhead},
        "synthetic": bool(synthetic), "snapping": deepcopy(snapping) if snapping is not None else {},
        "options": options.as_dict(),
    }


def build_route_comparison(
    graph: InMemoryGraph, selected_route: Route, standard_route: Route,
    diagnostics: dict[str, dict], profile: Profile, parameters: CostParameters, *,
    selected_presentation: dict | None = None,
) -> dict:
    """Compare supplied traces without searching or changing the evaluator.

    Both routes' diagnostics MUST come from the selected profile/parameters.
    The caller supplies a distance-only standard route found with no soft weights,
    wheelchair_required=False and prefer_handrail=False at the same snapped nodes.
    Its shortest-path provenance is the caller's responsibility, not verified here.
    A supplied selected presentation must describe these same inputs; only its
    full route records are copied, never marker selections or a nested comparison.
    Search costs are deliberately not compared/exported: their objectives differ,
    and a blocked trace has no feasible selected-profile optimized cost or score.
    """
    if (selected_route.nodes[0], selected_route.nodes[-1]) != (
        standard_route.nodes[0], standard_route.nodes[-1]
    ):
        raise DataError("Route comparison requires the same snapped endpoints in the same direction")

    def stats(route: Route, view: dict | None = None) -> dict:
        if view is None:
            view = build_route_presentation(graph, route, diagnostics, profile, parameters,
                                            include_context=False)
        records = [diagnostics[edge.id] for edge in route.edges]
        blocked = sum(record["blocked"] is True for record in records)
        # Pool selected-profile diagnostics, never the standard search's cost or
        # score and never averages of edge scores. Repeated traversals count again.
        exposure = sum(float(record["exposure_m"]) for record in records)
        risk = sum(float(record["way_risk_metres"]) + float(record["event_risk_metres"])
                   for record in records)
        known = sum(float(record["known_exposure_m"]) for record in records)
        obstacles = deepcopy(view["encountered_obstacles"])
        conflicts, unknowns = deepcopy(view["conflicts"]), deepcopy(view["unknowns"])
        counts = {kind: sum(row["kind"] == kind for row in obstacles) for kind in OBSTACLE_KINDS}
        return {
            "distance_m": float(route.distance_m),
            "score_percent": (100 * max(0.0, min(1.0, 1 - risk / exposure))
                              if exposure > 0 and not blocked else None),
            "coverage_percent": 100 * min(1.0, max(0.0, known / exposure)) if exposure else None,
            "hard_block_violations": blocked, "conflict_count": len(conflicts),
            "unknown_count": len(unknowns), "encounter_count": len(obstacles),
            "obstacle_counts": counts, "obstacles": obstacles, "conflicts": conflicts, "unknowns": unknowns,
        }

    selected = stats(selected_route, selected_presentation)
    standard = stats(standard_route)
    distance_difference = selected["distance_m"] - standard["distance_m"]
    selected_score, standard_score = selected["score_percent"], standard["score_percent"]
    limitations = [
        "Both routes are assessed using the same selected-profile observations, weights, risk, "
        "event exposure and missing-data rule. Profile match is a model summary, not a safety probability; "
        "score changes are percentage points, not percent overall accessibility improvement.",
        "Counts are grouped encounters by type, not stair risers or unique physical objects. "
        "Contiguous way encounters and node visits are grouped; nonconsecutive revisits count again. "
        "The same physical feature may count in multiple type rows. Counts can change with OSM "
        "way segmentation; they are not globally invariant under resegmentation.",
        "Known selected soft conflicts and missing information are reported separately. Unknown data "
        "are not observed obstacles, even when assigned risk. Zero tagged stairs is not proof of no stairs; "
        "only recorded features are visible and neither route is field-verified.",
        "A route violating selected hard requirements is infeasible under those requirements, not a "
        "valid scored alternative: its profile-match score and comparable optimized cost are unavailable. "
        "The standard path is not recommended if it violates selected requirements.",
        "The no-preference baseline differs from the distance-only baseline under matched hard constraints. "
        "It does not replace matched-constraint distance overhead or A*/Dijkstra optimality validation. "
        "Search costs from different objectives are not compared.",
        "Distances cover the same snapped graph endpoints, not unverified connections to requested "
        "locations. Signed changes are selected minus standard and may show trade-offs in either direction.",
        "No selected soft preferences means no preference-match score, not 100% accessibility.",
    ]
    if profile.wheelchair_required:
        limitations.append(
            "The standard search disables the selected wheelchair hard requirement. Differences may "
            "therefore reflect that changed feasibility constraint, not just soft-preference optimization."
        )
    return {
        "status": "available", "same_endpoints": True,
        "same_path": tuple(edge.id for edge in selected_route.edges) == tuple(edge.id for edge in standard_route.edges),
        "definition": "Standard route: supplied shortest pedestrian route by mapped distance between the "
                      "same snapped endpoints, with no soft preferences, no wheelchair hard requirement "
                      "and no handrail preference. Basic pedestrian, access and directional hard rules "
                      "remain enforced. Both routes are assessed here under the selected profile.",
        "selected": selected, "standard": standard,
        "obstacle_rows": [{"kind": kind, "label": OBSTACLE_LABELS[kind],
                           "selected": selected["obstacle_counts"][kind],
                           "standard": standard["obstacle_counts"][kind],
                           "difference": selected["obstacle_counts"][kind] - standard["obstacle_counts"][kind]}
                          for kind in OBSTACLE_KINDS],
        "distance_difference_m": distance_difference,
        "distance_difference_percent": (100 * distance_difference / standard["distance_m"]
                                        if standard["distance_m"] else None),
        "score_difference_pp": (selected_score - standard_score
                                if selected_score is not None and standard_score is not None else None),
        "limitations": limitations,
    }


def _markdown_text(value: object) -> str:
    """Make untrusted OSM/profile text a single literal Markdown cell/inline span."""
    text = " ".join(str(value).split())
    text = "".join("\\" + character if character in "\\`*_{}[]()#+-.!|~" else character for character in text)
    return escape(text, quote=True)


def _issue_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["None."]
    lines = ["| Preference | Weight / importance | Location | Extent | Observation |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        extent = "One node event" if row["node_id"] is not None else f"{row['distance_m']:.1f} m"
        cells = [PREFERENCE_LABELS[row["preference"]], f"{row['weight']:g} / {_importance(row['weight'])}",
                 row["location"], extent, row["reason"]]
        lines.append("| " + " | ".join(_markdown_text(cell) for cell in cells) + " |")
    return lines


def _comparison_table(comparison: dict, *, no_preferences: bool) -> list[str]:
    selected, standard = comparison["selected"], comparison["standard"]

    def percent(value: float | None) -> str:
        return "not available" if value is None else f"{value:.1f}%"

    def match(stats: dict) -> str:
        return ("infeasible under selected requirements" if stats["hard_block_violations"]
                else percent(stats["score_percent"]))

    distance_change = f"{comparison['distance_difference_m']:+.1f} m"
    distance_percent = comparison["distance_difference_percent"]
    distance_change += (f" ({distance_percent:+.1f}%)" if distance_percent is not None
                        else " (percentage not defined: zero standard distance)")
    score_change = comparison["score_difference_pp"]
    coverage_change = (selected["coverage_percent"] - standard["coverage_percent"]
                       if selected["coverage_percent"] is not None and standard["coverage_percent"] is not None
                       else None)
    lines = ["## Selected vs no-preference shortest pedestrian route", "",
             _markdown_text(comparison["definition"]), "",
             "Changes are selected minus standard; fewer obstacles can coexist with worse conditions of another type.", "",
             "| Measure | Selected route | Standard route | Change |",
             "| --- | --- | --- | --- |",
             f"| Distance | {selected['distance_m']:.1f} m | {standard['distance_m']:.1f} m | {distance_change} |",
             f"| Profile match | {match(selected)} | {match(standard)} | "
             + ("not available" if score_change is None else f"{score_change:+.1f} pp") + " |",
             f"| Data coverage | {percent(selected['coverage_percent'])} | {percent(standard['coverage_percent'])} | "
             + ("not available" if coverage_change is None else f"{coverage_change:+.1f} pp") + " |"]
    for label, key in (("Known selected soft conflicts", "conflict_count"),
                       ("Unknowns (missing information)", "unknown_count"),
                       ("Blocked directed traversals", "hard_block_violations")):
        lines.append(f"| {label} | {selected[key]} | {standard[key]} | {selected[key] - standard[key]:+d} |")
    for row in comparison["obstacle_rows"]:
        if row["selected"] or row["standard"]:
            lines.append(f"| {_markdown_text(row['label'])} encounters | {row['selected']} | "
                         f"{row['standard']} | {row['difference']:+d} |")
    if comparison["same_path"]:
        lines.extend(["", "**Same path:** both routes use the exact same directed edge sequence."])
    if no_preferences:
        lines.extend(["", "**No selected soft preferences:** profile match is unavailable, not 100% accessibility."])
    if standard["hard_block_violations"]:
        lines.extend(["", "**The standard path is not recommended:** it violates selected hard requirements."])
    return lines


def _standard_route_details(comparison: dict) -> list[str]:
    standard = comparison["standard"]
    lines = ["", "## Standard route — full selected-profile assessment", "",
             "These are all grouped records, independent of marker filters and limits.", "",
             f"### Encountered obstacles ({standard['encounter_count']} grouped encounters)", ""]
    if standard["obstacles"]:
        lines.extend(["| Type | Location | Extent | Status | Selected preferences | Observation |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for row in standard["obstacles"]:
            extent = "Node encounter" if row["node_id"] is not None else f"{row['distance_m']:.1f} m"
            cells = [row["label"], row["location"], extent, row["classification"],
                     "; ".join(row["preference_labels"]) or "None", row["description"]]
            lines.append("| " + " | ".join(_markdown_text(cell) for cell in cells) + " |")
    else:
        lines.append("No known adverse encounters identified; this is not evidence of an obstacle-free route.")
    lines.extend(["", f"### Known selected soft conflicts ({standard['conflict_count']} grouped entries)", ""])
    lines.extend(_issue_table(standard["conflicts"]))
    lines.extend(["", f"### Missing data ({standard['unknown_count']} grouped entries)", ""])
    lines.extend(_issue_table(standard["unknowns"]))
    lines.extend(["", "### Comparison limitations", ""])
    lines.extend("- " + _markdown_text(item) for item in comparison["limitations"])
    return lines


def render_route_summary(presentation: dict) -> str:
    """Human-first Markdown of the full route, never the marker-limited subset."""
    metrics = presentation["metrics"]
    score = "not available (no selected-criterion exposure)" if metrics["score_percent"] is None else (
        f"{metrics['score_percent']:.1f}%"
    )
    coverage = "not available" if metrics["coverage_percent"] is None else f"{metrics['coverage_percent']:.1f}%"
    comparison = presentation.get("comparison")
    cost = f"{metrics['cost_m']:.1f} m"
    if comparison is not None and comparison["selected"]["hard_block_violations"]:
        score = "infeasible under selected requirements"
        cost = "not available (infeasible under selected requirements)"
    data = ("Synthetic example — not observed OpenStreetMap conditions." if presentation["synthetic"] else
            "Real-data run (OpenStreetMap input, not flagged synthetic) — not field-verified.")
    lines = [f"# Route summary — {_markdown_text(presentation['profile_name'])}", "", data, "",
             f"**Distance:** {metrics['distance_m']:.1f} m · **Preference match:** {score} · "
             f"**Data coverage:** {coverage} · **Optimized cost:** {cost}", "",
             "Match and coverage pool the supplied risk/exposure diagnostics, not averages of edge scores. "
             "They are model summaries, not safety probabilities.", ""]
    if comparison is not None:
        lines.extend(_comparison_table(comparison, no_preferences=not presentation["selected_preferences"]))
        lines.append("")
    lines.extend(["## Selected preferences", ""])
    for preference in presentation["selected_preferences"]:
        lines.append(f"- {_markdown_text(preference['label'])}: weight {preference['weight']:g} "
                     f"(**{_markdown_text(preference['importance'])}**).")
    if not presentation["selected_preferences"]:
        lines.append("None selected; no preference-match score is available.")
    lines.extend(["", "## Requirements and model policy", ""])
    lines.extend(f"- {_markdown_text(requirement)}" for requirement in presentation["requirements"])
    encountered = presentation["encountered_obstacles"]
    lines.extend(["", f"## Encountered obstacles ({len(encountered)} grouped encounters)", ""])
    for obstacle in encountered:
        classification = "**PREFERENCE CONFLICT**" if obstacle["preference_labels"] else "Encountered"
        if obstacle["status"] == "blocked":
            classification += " — **HARD CONSTRAINT VIOLATION** on a selected traversal"
        preferences = (". Selected preferences: " + "; ".join(obstacle["preference_labels"]) + "."
                   if obstacle["preference_labels"] else ". No known selected soft-preference conflict for this feature.")
        lines.append(f"- {classification}: {_markdown_text(obstacle['label'])} — "
                     f"{_markdown_text(obstacle['location'])}. "
                     f"{_markdown_text(obstacle['description'] + preferences)}")
    if not encountered:
        lines.append("No known adverse encounters identified; this is not evidence that the route is obstacle-free.")
    lines.extend(["", f"## Known selected soft conflicts ({len(presentation['conflicts'])} grouped entries)", ""])
    lines.extend(_issue_table(presentation["conflicts"]))
    lines.extend(["", f"## Missing data ({len(presentation['unknowns'])} grouped entries)", "",
                  "Unknown data are not observed obstacles or hard violations. They remain here even when "
                  "the configured unknown risk is zero; they still reduce data coverage.", ""])
    lines.extend(_issue_table(presentation["unknowns"]))
    if comparison is not None:
        lines.extend(_standard_route_details(comparison))
    validation = presentation["validation"]
    lines.extend(["", "## Validation and limitations", "",
                  f"- Hard-constraint violations on selected directed edges: {validation['hard_constraint_violations']}."])
    agree = validation["astar_dijkstra_agree"]
    if agree is None:
        lines.append("- A*/Dijkstra cost comparison: not run or not supplied (both baseline costs are required).")
    else:
        result = "verified: supplied costs agree within tolerance" if agree else "NOT verified: supplied costs disagree"
        lines.append(f"- A*/Dijkstra cost comparison: {result}; absolute difference "
                     f"{validation['optimal_cost_difference_m']:.6g} m.")
    distance = validation["distance_baseline_m"]
    if distance is None:
        lines.append("- Distance-only baseline under the same hard constraints: not run or not supplied.")
    else:
        overhead = validation["distance_overhead_percent"]
        detail = "not defined for a zero-distance baseline" if overhead is None else f"{overhead:.1f}%"
        lines.append(f"- Distance-only baseline under the same hard constraints: {distance:.1f} m; "
                     f"route distance overhead: {detail}.")
    lines.extend([
        "- Cost optimality is conditional on the same graph, cost function and hard constraints. "
        "Supplied baseline agreement is not proof of physical safety, complete mapping, or general algorithm correctness.",
        "- Only recorded features are visible; conditions and missing tags need independent checking. "
        "Off-route features are context, not claimed to have been successfully avoided.",
    ])
    snapping = presentation["snapping"]
    gaps = []
    for label in ("start", "end"):
        snap = snapping.get(label)
        gap = _nonnegative_number(snap.get("distance_m")) if isinstance(snap, Mapping) else None
        if gap is not None:
            gaps.append(f"{label} {gap:.1f} m")
    lines.append("- Snapping: " + (", ".join(gaps) if gaps else "gap distances not supplied")
                 + ". Connections to mapped endpoints are unverified and excluded from route distance and GPX.")
    summary = presentation["filter_summary"]
    kinds = summary["selected_kinds"]
    mode = ("Automatic: important obstacles and known selected-preference concerns" if kinds is None else
            "Explicit types: " + (", ".join(OBSTACLE_LABELS[kind] for kind in kinds) if kinds else "none"))
    count, limit = summary["selected_candidates"], summary["max_markers"]
    lines.extend(["", "## Map markers", "", mode + ".",
                  f"{count} of {summary['total_candidates']} known-feature candidates remain after selection; "
                  f"{summary['omitted_by_filter']} omitted by the marker filter. Marker limit: {limit} "
                  f"(at most {min(count, limit)} markers before viewport cropping and coalescing).",
                  "The figure caption reports actual displayed/cropped/suppressed counts; a hidden marker is not a hidden route conflict.",
                  "Type selection and the marker limit affect map markers only. The complete encounter, "
                  "conflict and missing-data lists, route metrics and calculations above are not truncated or changed."])
    return "\n".join(lines) + "\n"