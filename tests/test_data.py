import json

import mongomock
import pytest
from pymongo.errors import OperationFailure

from osm_accessibility.data import Dataset, load_osm, normalize_element
from osm_accessibility.errors import DatabaseError, DataError
from osm_accessibility.mongo import import_dataset, load_dataset


def extract():
    return {"osm3s": {"timestamp_osm_base": "2026-09-01T00:00:00Z"}, "elements": [
        {"type": "node", "id": 1, "lat": 49.58, "lon": 11.0, "tags": {"highway": "crossing", "kerb": "lowered"}},
        {"type": "node", "id": 2, "lat": 49.58, "lon": 11.001},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway"}},
    ]}


def test_overpass_json_normalizes_ids_and_preserves_node_tags(tmp_path):
    path = tmp_path / "extract.json"
    path.write_text(json.dumps(extract()))
    dataset = load_osm(path)
    assert dataset.nodes[0]["id"] == "1"
    assert dataset.nodes[0]["location"]["coordinates"] == [11, 49.58]
    assert dataset.nodes[0]["tags"]["kerb"] == "lowered"
    assert dataset.metadata["osm_base_timestamp"] == "2026-09-01T00:00:00Z"
    assert dataset.fingerprint() == Dataset.from_elements(extract()["elements"]).fingerprint()


def test_osm_xml_reads_complete_topology_and_metadata(tmp_path):
    path = tmp_path / "extract.osm"
    path.write_text('<osm version="0.6"><node id="1" lat="49.58" lon="11" version="2"><tag k="kerb" v="raised"/></node>'
                    '<node id="2" lat="49.58" lon="11.001"/><way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="path"/></way></osm>')
    dataset = load_osm(path)
    assert dataset.nodes[0]["version"] == "2"
    assert dataset.ways[0]["nodes"] == ["1", "2"]


@pytest.mark.parametrize("bad", [
    {"remark": "runtime error: timeout", **extract()},
    {"type": "FeatureCollection", "features": []},
    {"elements": [{"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "path"}}]},
])
def test_incomplete_exports_are_rejected(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(DataError):
        load_osm(path)


def test_malformed_and_entity_xml_is_rejected(tmp_path):
    path = tmp_path / "unsafe.osm"
    path.write_text('<!DOCTYPE osm [<!ENTITY malicious "expansion">]><osm>&malicious;</osm>')
    with pytest.raises(DataError):
        load_osm(path)


def test_xml_error_remark_is_not_treated_as_valid_partial_data(tmp_path):
    path = tmp_path / "partial.osm"
    path.write_text('<osm><node id="1" lat="49.58" lon="11"/><node id="2" lat="49.58" lon="11.001"/>'
                    '<way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="path"/></way>'
                    '<remark>runtime error: Query timed out</remark></osm>')
    with pytest.raises(DataError, match="remark"):
        load_osm(path)


@pytest.mark.parametrize("kind", [None, [], {}, 5, "unknown"])
def test_invalid_osm_element_type_is_a_data_error(kind):
    with pytest.raises(DataError):
        normalize_element({"type": kind})


def test_conflicting_duplicates_fail_and_identical_duplicates_deduplicate():
    elements = extract()["elements"]
    assert len(Dataset.from_elements([*elements, elements[0]]).nodes) == 2
    with pytest.raises(DataError, match="Conflicting duplicate"):
        Dataset.from_elements([*elements, elements[0] | {"lat": 0}])


def test_pedestrian_relations_inherit_tags_without_shortcuts():
    elements = extract()["elements"]
    elements[-1]["tags"] = {"surface": "asphalt"}
    elements.append({"type": "relation", "id": 100, "members": [{"type": "way", "ref": 10, "role": "outer"}],
                     "tags": {"highway": "pedestrian", "surface": "gravel"}})
    graph = Dataset.from_elements(elements).graph()
    assert len(graph.edges) == 2
    assert graph.edges[0].tags["highway"] == "pedestrian"
    assert graph.edges[0].tags["surface"] == "asphalt"


def test_relation_missing_members_fail():
    elements = extract()["elements"]
    elements.append({"type": "relation", "id": 100, "members": [{"type": "way", "ref": 99}], "tags": {}})
    with pytest.raises(DataError, match="missing way"):
        Dataset.from_elements(elements)


def test_dry_run_never_touches_database(dataset):
    class ForbiddenDatabase:
        def __getattr__(self, name):
            raise AssertionError("Dry run accessed database")
    result = import_dataset(dataset, ForbiddenDatabase(), "experiment_1")
    assert result["write"] is False and result["counts"] == dataset.counts()


def test_snapshot_upload_roundtrip_and_idempotency_protection(dataset):
    database = mongomock.MongoClient().OSMResearch
    result = import_dataset(dataset, database, "sample", write=True, batch_size=2)
    assert result["status"] == "ready"
    restored = load_dataset(database, "sample")
    assert restored.fingerprint() == dataset.fingerprint()
    assert restored.nodes == dataset.nodes
    before = database.Nodes.count_documents({})
    with pytest.raises(DataError, match="already exists"):
        import_dataset(dataset, database, "sample", write=True)
    assert database.Nodes.count_documents({}) == before
    import_dataset(dataset, database, "sample_second", write=True)
    assert database.Nodes.count_documents({}) == before * 2


def test_partial_snapshot_never_becomes_routable(dataset, monkeypatch):
    database = mongomock.MongoClient().OSMResearch
    monkeypatch.setattr(database.Ways, "insert_many", lambda *a, **k: (_ for _ in ()).throw(OperationFailure("private-driver-details")))
    with pytest.raises(DatabaseError) as error:
        import_dataset(dataset, database, "partial", write=True)
    assert "private-driver-details" not in str(error.value)
    assert database.Datasets.find_one({"_id": "partial"})["status"] == "failed"
    assert database.Nodes.count_documents({"dataset_id": "partial"}) > 0
    with pytest.raises(DataError, match="not ready"):
        load_dataset(database, "partial")


def test_mutated_snapshot_is_detected(dataset):
    database = mongomock.MongoClient().OSMResearch
    import_dataset(dataset, database, "sample", write=True)
    database.Nodes.update_one({"dataset_id": "sample", "id": "gate"}, {"$set": {"tags.wheelchair": "yes"}})
    with pytest.raises(DataError, match="fingerprint"):
        load_dataset(database, "sample")


def test_interrupted_loading_snapshot_is_not_readable_or_overwritten(dataset):
    database = mongomock.MongoClient().OSMResearch
    database.Datasets.insert_one({"_id": "interrupted", "status": "loading"})
    database.Nodes.insert_one({**dataset.nodes[0], "dataset_id": "interrupted"})
    with pytest.raises(DataError, match="not ready"):
        load_dataset(database, "interrupted")
    with pytest.raises(DataError, match="already exists"):
        import_dataset(dataset, database, "interrupted", write=True)
    assert database.Nodes.count_documents({"dataset_id": "interrupted"}) == 1


def test_personal_osm_metadata_not_retained():
    result = normalize_element({"type": "node", "id": 1, "lat": 0, "lon": 0, "user": "mapper", "uid": 123})
    assert "user" not in result and "uid" not in result