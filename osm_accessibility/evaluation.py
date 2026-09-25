"""Transparent profile-dependent edge evaluation; scores are not safety probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .models import Edge
from .profiles import CostParameters, Profile
from .tags import incline_percent, is_crossing, metres

MODEL_VERSION = "osm-accessibility-0.1"
PEDESTRIAN_HIGHWAYS = {"footway", "pedestrian", "living_street", "path", "track", "residential",
                      "service", "unclassified", "tertiary", "secondary", "primary", "steps"}
FORBIDDEN_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "raceway", "construction"}
DENIED_ACCESS = {"no", "private", "customers", "delivery", "agricultural", "forestry"}
ALLOWED_FOOT = {"yes", "designated", "permissive", "destination"}
GOOD_SURFACES = {"asphalt", "concrete", "concrete:plates", "paved", "paving_stones"}
POOR_SURFACES = {"cobblestone", "unhewn_cobblestone", "unpaved", "ground", "sand", "mud", "gravel", "sett"}
GOOD_SMOOTHNESS = {"excellent", "good"}
POOR_SMOOTHNESS = {"bad", "very_bad", "horrible", "very_horrible", "impassable"}


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    scope: str
    weight: float
    risk: float
    known: bool
    reason: str

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EdgeEvaluation:
    edge_id: str
    length_m: float
    hard_blocks: tuple[str, ...]
    criteria: tuple[Criterion, ...]
    way_risk: float
    event_risk: float
    way_weight: float
    event_weight: float
    way_known_weight: float
    event_known_weight: float
    event_equivalent_m: float
    accessibility_factor: float

    @property
    def blocked(self) -> bool:
        return bool(self.hard_blocks)

    @property
    def way_risk_metres(self) -> float:
        return self.length_m * self.way_risk

    @property
    def event_risk_metres(self) -> float:
        return self.event_equivalent_m * self.event_risk if self.event_weight else 0.0

    @property
    def penalty_m(self) -> float:
        return self.accessibility_factor * (self.way_risk_metres + self.event_risk_metres)

    @property
    def cost_m(self) -> float | None:
        return None if self.blocked else self.length_m + self.penalty_m

    @property
    def exposure_m(self) -> float:
        return (self.length_m if self.way_weight else 0.0) + (
            self.event_equivalent_m if self.event_weight else 0.0
        )

    @property
    def known_exposure_m(self) -> float:
        return (self.length_m * self.way_known_weight / self.way_weight if self.way_weight else 0.0) + (
            self.event_equivalent_m * self.event_known_weight / self.event_weight if self.event_weight else 0.0
        )

    @property
    def score(self) -> float | None:
        if self.blocked or self.exposure_m <= 0:
            return None
        return 100 * max(0.0, min(1.0, 1 - (self.way_risk_metres + self.event_risk_metres) / self.exposure_m))

    @property
    def coverage(self) -> float | None:
        return 100 * min(1.0, max(0.0, self.known_exposure_m / self.exposure_m)) if self.exposure_m else None

    def as_dict(self) -> dict:
        return {
            "edge_id": self.edge_id, "blocked": self.blocked, "hard_blocks": list(self.hard_blocks),
            "length_m": self.length_m, "risk": self.way_risk, "transition_risk": self.event_risk,
            "way_weight": self.way_weight, "event_weight": self.event_weight,
            "way_known_weight": self.way_known_weight, "event_known_weight": self.event_known_weight,
            "way_risk_metres": self.way_risk_metres, "event_risk_metres": self.event_risk_metres,
            "event_equivalent_m": self.event_equivalent_m, "exposure_m": self.exposure_m,
            "known_exposure_m": self.known_exposure_m, "penalty_m": self.penalty_m, "cost_m": self.cost_m,
            "score": self.score, "way_score": 100 * (1 - self.way_risk) if self.way_weight and not self.blocked else None,
            "coverage": self.coverage, "criteria": [criterion.as_dict() for criterion in self.criteria],
        }


def wheelchair_ramp(tags: Mapping[str, str]) -> bool:
    if tags.get("ramp:wheelchair") == "no":
        return False
    return tags.get("ramp:wheelchair") == "yes" or (
        tags.get("ramp") in {"yes", "wheelchair"} and tags.get("wheelchair") == "yes"
    )


def hard_restrictions(edge: Edge, profile: Profile) -> tuple[str, ...]:
    reasons = []
    highway = edge.tags.get("highway", "")
    if highway in FORBIDDEN_HIGHWAYS:
        reasons.append(f"excluded highway: {highway}")
    elif highway not in PEDESTRIAN_HIGHWAYS and not (
        highway == "cycleway" and edge.tags.get("foot") in ALLOWED_FOOT
    ):
        reasons.append("no supported pedestrian highway/permission")
    for source, tags in (("way", edge.tags), ("source node", edge.source_tags), ("target node", edge.target_tags)):
        foot = tags.get("foot")
        if foot in DENIED_ACCESS or (foot not in ALLOWED_FOOT and tags.get("access") in DENIED_ACCESS):
            reasons.append(f"{source}: pedestrian access restricted")
        if tags.get("barrier") in {"wall", "fence", "hedge", "retaining_wall", "block"}:
            reasons.append(f"{source}: physical barrier")
        if profile.wheelchair_required:
            if tags.get("wheelchair") == "no":
                reasons.append(f"{source}: wheelchair=no")
            if tags.get("barrier") in {"stile", "turnstile", "kissing_gate", "cycle_barrier"} and tags.get("wheelchair") != "yes":
                reasons.append(f"{source}: barrier not verified for wheelchair use")
    if profile.wheelchair_required and highway == "steps" and not (profile.allow_ramps and wheelchair_ramp(edge.tags)):
        reasons.append("steps without an explicitly usable wheelchair ramp")
    return tuple(dict.fromkeys(reasons))


def evaluate_edge(edge: Edge, profile: Profile, parameters: CostParameters) -> EdgeEvaluation:
    tags = edge.tags
    criteria: list[Criterion] = []
    unknown = parameters.unknown_risk

    def add(name: str, risk: float | None, reason: str, scope: str = "way") -> None:
        weight = profile.weights[name]
        if weight > 0:
            criteria.append(Criterion(name, scope, weight, unknown if risk is None else min(1.0, max(0.0, risk)),
                                       risk is not None, reason))

    bike = tags.get("bicycle")
    bicycle_risk = (0.0 if bike in {"no", "dismount"} or tags.get("segregated") == "yes" else
                    1.0 if bike in {"yes", "designated", "permissive", "destination"} else None)
    add("avoid_bicycles", bicycle_risk, "explicit cycling permission and segregation")
    add("lit_roads", {"yes": 0.0, "no": 1.0}.get(tags.get("lit")), "mapped lighting")
    surface = tags.get("surface")
    surface_risk = 0.0 if surface in GOOD_SURFACES else 1.0 if surface in POOR_SURFACES else None
    track_risk = {"grade3": .6, "grade4": .8, "grade5": 1.0}.get(tags.get("tracktype"))
    if track_risk is not None:
        surface_risk = max(surface_risk if surface_risk is not None else unknown, track_risk)
    add("surface_type", surface_risk, "surface and track grade")
    smoothness = tags.get("smoothness")
    add("smoothness", 0.0 if smoothness in GOOD_SMOOTHNESS else 1.0 if smoothness in POOR_SMOOTHNESS
        else .5 if smoothness == "intermediate" else None, "mapped smoothness")
    width = metres(tags.get("width"))
    add("road_width", None if width is None else float(width < profile.min_width_m), "minimum selected width")
    incline = incline_percent(tags.get("incline"))
    add("road_incline", None if incline is None else abs(incline) / parameters.incline_full_risk_percent,
        "absolute incline; qualitative up/down has unknown magnitude")
    stairs = tags.get("highway") == "steps"
    stair_risk = .25 if stairs and profile.allow_ramps and wheelchair_ramp(tags) else 1.0 if stairs else 0.0
    if stairs and profile.prefer_handrail and tags.get("handrail") not in {"yes", "left", "right", "both"}:
        stair_risk = min(1.0, stair_risk + .25)
    add("avoid_stairs", stair_risk, "steps, ramp and handrail")
    add("wheelchair_accessible", {"yes": 0.0, "no": 1.0, "limited": .5}.get(tags.get("wheelchair")),
        "explicit wheelchair tag (separate from hard requirement)")

    def add_kerb(values: Mapping[str, str], scope: str) -> None:
        height = metres(values.get("kerb:height"))
        risk = (height / parameters.kerb_full_risk_m if height is not None else
                {"flush": 0.0, "lowered": 0.0, "no": 0.0, "raised": 1.0, "rolled": 1.0, "stepped": 1.0}.get(values.get("kerb")))
        add("short_kerbs", risk, "kerb height/type at a mapped event or crossing", scope)

    def add_crossing(values: Mapping[str, str], scope: str) -> None:
        risk = (0.0 if values.get("crossing:signals") == "yes" else
                {"traffic_signals": 0.0, "supervised": 0.0, "marked": .35, "zebra": .35,
                 "uncontrolled": .35, "unmarked": 1.0, "unsupervised": 1.0}.get(values.get("crossing")))
        add("supervised_crossings", risk, "mapped crossing control", scope)

    event = edge.target_tags or edge.crossing_tags
    crossing_event = is_crossing(event)
    kerb_event = crossing_event or "kerb" in event or "kerb:height" in event
    if crossing_event:
        add_crossing(event, "event")
    if kerb_event:
        add_kerb(event, "event")
    # Do not charge the same feature as both a way attribute and a node event.
    # These flags cover the whole way, so subdividing its geometry cannot change
    # which continuous criteria enter the denominator.
    if is_crossing(tags) and not (edge.way_has_crossing_nodes or crossing_event):
        add_crossing(tags, "way")
    if (is_crossing(tags) or "kerb" in tags or "kerb:height" in tags) and not (edge.way_has_kerb_nodes or kerb_event):
        add_kerb(tags, "way")

    def aggregate(scope: str) -> tuple[float, float, float]:
        subset = [criterion for criterion in criteria if criterion.scope == scope]
        weight = sum(criterion.weight for criterion in subset)
        known = sum(criterion.weight for criterion in subset if criterion.known)
        return sum(criterion.weight * criterion.risk for criterion in subset) / weight if weight else 0.0, weight, known

    way_risk, way_weight, way_known = aggregate("way")
    event_risk, event_weight, event_known = aggregate("event")
    return EdgeEvaluation(edge.id, edge.distance_m, hard_restrictions(edge, profile), tuple(criteria),
                          way_risk, event_risk, way_weight, event_weight, way_known, event_known,
                          parameters.event_equivalent_m, parameters.accessibility_factor)


class Evaluator:
    """Request/run-local cache; cost, exported scores and plots share evaluations."""

    def __init__(self, profile: Profile, parameters: CostParameters):
        self.profile = profile
        self.parameters = parameters
        self.cache: dict[str, EdgeEvaluation] = {}

    def __call__(self, edge: Edge) -> EdgeEvaluation:
        if edge.id not in self.cache:
            self.cache[edge.id] = evaluate_edge(edge, self.profile, self.parameters)
        return self.cache[edge.id]