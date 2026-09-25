"""Read OSM extracts and validate complete references before any database write."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import ijson
from defusedxml import ElementTree

from .builder import build_graph
from .errors import DataError
from .models import Coordinate, _normalize_id
from .tags import normalize_tags

OSM_ATTRIBUTION = "© OpenStreetMap contributors — https://www.openstreetmap.org/copyright (ODbL 1.0)"


def canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_element(raw: Mapping) -> dict:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("type"), str) or raw["type"] not in {"node", "way", "relation"}:
        raise DataError("Expected an OSM node, way or relation object")
    kind = raw["type"]
    result = {"type": kind, "id": _normalize_id(raw.get("id"), "OSM ID"), "tags": dict(normalize_tags(raw.get("tags", {})))}
    if kind == "node":
        if "location" in raw:
            location = raw["location"]
            if not isinstance(location, Mapping) or location.get("type", "Point") != "Point":
                raise DataError("A node must have Point geometry")
            coordinate = Coordinate.from_geojson(location.get("coordinates"))
        else:
            coordinate = Coordinate.from_lat_lon([raw.get("lat"), raw.get("lon")])
        result["location"] = {"type": "Point", "coordinates": coordinate.as_geojson()}
    elif kind == "way":
        if not isinstance(raw.get("nodes"), (list, tuple)):
            raise DataError("OSM ways require ordered node references; use a complete extract")
        result["nodes"] = [_normalize_id(value, "Way node ID") for value in raw["nodes"]]
    else:
        if not isinstance(raw.get("members", []), (list, tuple)):
            raise DataError("Relation members must be an array")
        members = []
        for member in raw.get("members", []):
            if not isinstance(member, Mapping) or member.get("type") not in {"node", "way", "relation"}:
                raise DataError("Invalid relation member")
            members.append({"type": member["type"], "ref": _normalize_id(member.get("ref"), "Relation member ID"),
                            "role": str(member.get("role", ""))})
        result["members"] = members
    for key in ("version", "timestamp"):
        if key in raw:
            result[key] = str(raw[key])
    # Usernames/UIDs and changeset personal metadata are not needed for routing.
    return result


@dataclass(frozen=True)
class Dataset:
    nodes: tuple[dict, ...]
    ways: tuple[dict, ...]
    relations: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_elements(cls, elements: Iterable[Mapping], metadata: dict | None = None) -> Dataset:
        groups: dict[str, dict[str, dict]] = {"node": {}, "way": {}, "relation": {}}
        for raw in elements:
            document = normalize_element(raw)
            group = groups[document["type"]]
            if document["id"] in group:
                if group[document["id"]] != document:
                    raise DataError(f"Conflicting duplicate {document['type']} ID {document['id']}")
                continue
            group[document["id"]] = document
        dataset = cls(*(tuple(group[key] for key in sorted(group)) for group in groups.values()),
                      metadata=dict(metadata or {}))
        dataset.validate()
        return dataset

    def validate(self) -> None:
        if not self.nodes or not self.ways:
            raise DataError("Extract must contain nodes and ways, including referenced geometry")
        ids = {"node": {item["id"] for item in self.nodes}, "way": {item["id"] for item in self.ways},
               "relation": {item["id"] for item in self.relations}}
        for way in self.ways:
            missing = set(way["nodes"]) - ids["node"]
            if missing:
                raise DataError(f"Way {way['id']} has missing node references: {sorted(missing)[:5]}")
        for relation in self.relations:
            for member in relation["members"]:
                if member["ref"] not in ids[member["type"]]:
                    raise DataError(f"Relation {relation['id']} has a missing {member['type']} reference: {member['ref']}")

    def elements(self) -> Iterable[dict]:
        yield from self.nodes
        yield from self.ways
        yield from self.relations

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for document in self.elements():
            digest.update(canonical_json(document).encode("utf-8") + b"\n")
        return digest.hexdigest()

    def counts(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "ways": len(self.ways), "relations": len(self.relations)}

    def graph(self):
        ways = {way["id"]: {**way, "tags": dict(way["tags"])} for way in self.ways}
        inherited: dict[str, dict] = {}
        for relation in self.relations:
            if relation["tags"].get("highway") != "pedestrian":
                continue
            for member in relation["members"]:
                if member["type"] == "way" and member["role"] in {"", "outer"}:
                    tags = inherited.setdefault(member["ref"], {})
                    for key, value in relation["tags"].items():
                        if key in tags and tags[key] != value and key not in ways[member["ref"]]["tags"]:
                            raise DataError(f"Conflicting inherited pedestrian relation tag {key} for way {member['ref']}")
                        tags[key] = value
        for way_id, tags in inherited.items():
            ways[way_id]["tags"] = {**tags, **ways[way_id]["tags"]}
        return build_graph(self.nodes, ways.values())


def _xml_elements(path: Path) -> Iterable[dict]:
    with path.open("rb") as stream:
        context = ElementTree.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        if root.tag != "osm":
            raise DataError("Expected OSM XML root; change files and GPX are not OSM extracts")
        for event, element in context:
            if event == "end" and element.tag == "remark" and (element.text or "").strip():
                raise DataError("OSM XML contains an error/remark; export a complete dataset before importing")
            if event != "end" or element.tag not in {"node", "way", "relation"}:
                continue
            raw = {"type": element.tag, **element.attrib,
                   "tags": {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}}
            if element.tag == "node":
                raw["lat"], raw["lon"] = float(raw["lat"]), float(raw["lon"])
            elif element.tag == "way":
                raw["nodes"] = [node.attrib["ref"] for node in element.findall("nd")]
            else:
                raw["members"] = [dict(member.attrib) for member in element.findall("member")]
            yield raw
            element.clear()
            root.clear()


def load_osm(path: str | Path) -> Dataset:
    path = Path(path)
    try:
        metadata = {"source_file": path.name, "source_sha256": sha256_file(path),
                    "synthetic": False, "attribution": OSM_ATTRIBUTION}
        if path.suffix.lower() in {".osm", ".xml"}:
            return Dataset.from_elements(_xml_elements(path), metadata)
        if path.suffix.lower() != ".json":
            raise DataError("Use a complete .osm/.xml extract or Overpass .json (not GeoJSON or PBF)")
        with path.open("rb") as stream:
            for event in ijson.items(stream, "remark", use_float=True):
                if event:
                    raise DataError("Overpass reported an error/remark; export a complete dataset before importing")
        with path.open("rb") as stream:
            meta = next(ijson.items(stream, "osm3s", use_float=True), {})
            metadata["osm_base_timestamp"] = meta.get("timestamp_osm_base") if isinstance(meta, dict) else None
        with path.open("rb") as stream:
            return Dataset.from_elements(ijson.items(stream, "elements.item", use_float=True), metadata)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Cannot parse complete OSM extract ({type(exc).__name__})") from None