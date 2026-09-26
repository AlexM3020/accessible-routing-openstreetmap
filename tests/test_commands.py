import json
from contextlib import contextmanager

import mongomock
import pytest
from pymongo.errors import OperationFailure

from osm_accessibility.cli import main
from osm_accessibility.errors import DatabaseError, DataError
from osm_accessibility.experiment import load_run
from osm_accessibility.mongo import connect


def write_extract(path):
    raw = {"elements": [
        {"type": "node", "id": 1, "lat": 49.58, "lon": 11.0},
        {"type": "node", "id": 2, "lat": 49.58, "lon": 11.001},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway", "surface": "asphalt"}},
    ]}
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_command_pipeline_local_upload_and_snapshot_routing(tmp_path, monkeypatch, capsys):
    path = tmp_path / "extract.json"
    write_extract(path)
    calls = []
    database = mongomock.MongoClient().OSMResearch

    @contextmanager
    def fake_connection(name):
        calls.append(name)
        yield database

    monkeypatch.setattr("osm_accessibility.cli.connect", fake_connection)
    assert main(["import-osm", str(path), "--dataset", "study"]) == 0
    assert not calls
    assert json.loads(capsys.readouterr().out)["write"] is False
    assert main(["import-osm", str(path), "--dataset", "study", "--write"]) == 0
    assert calls == ["OSMResearch"]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"name": "custom", "weights": {"surface_type": "Essential"}}))
    route_args = ["--start", "49.58", "11", "--end", "49.58", "11.001", "--profile-file", str(profile), "--no-plots"]
    assert main(["route", "--osm", str(path), *route_args, "--output", str(tmp_path / "local")]) == 0
    assert main(["route", "--dataset", "study", *route_args, "--output", str(tmp_path / "mongo")]) == 0
    local = load_run(tmp_path / "local")[-1]
    remote = load_run(tmp_path / "mongo")[-1]
    assert local["graph_sha256"] == remote["graph_sha256"]
    assert local["route"]["edge_ids"] == remote["route"]["edge_ids"]
    assert main(["plot", str(tmp_path / "local"), "--output", str(tmp_path / "plots"), "--format", "png"]) == 0
    assert main(["sweep", str(tmp_path / "local"), "--output", str(tmp_path / "sweep"), "--factors", "0", "1.5", "--unknown-risks", ".5", "--repeats", "1"]) == 0


def test_database_connection_closes_and_sanitizes_errors(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://invalid.example/")

    class FakeClient:
        closed = False
        admin = None

        def __init__(self):
            self.admin = self

        def command(self, name):
            assert name == "ping"

        def __getitem__(self, name):
            assert name == "OSMResearch"
            return self

        def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr("osm_accessibility.mongo.MongoClient", lambda *a, **kw: client)
    with connect() as database:
        assert database is client
    assert client.closed
    client.closed = False
    with pytest.raises(DatabaseError) as result:
        with connect():
            raise OperationFailure("example-private-driver-detail")
    assert client.closed
    assert "example-private-driver-detail" not in str(result.value)


def test_no_missing_credentials_or_system_database_writes(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(DatabaseError, match="MONGODB_URI"):
        with connect():
            pass
    for name in ("admin", "config", "local", "unsafe.name"):
        with pytest.raises(DataError):
            with connect(name):
                pass


def test_marker_controls_json_and_replot_inheritance(tmp_path, capsys):
    output = tmp_path / "controlled"
    assert main(["demo", "--output", str(output), "--no-plots", "--obstacles", "stairs",
                 "--max-markers", "4", "--details", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["output"]["options"] == {"obstacle_kinds": ["stairs"], "max_markers": 4}
    assert (output / "diagnostics" / "manifest.json").exists()
    assert main(["plot", str(output), "--output", str(tmp_path / "redrawn"), "--max-markers", "2"]) == 0
    summary = (tmp_path / "redrawn" / "route_summary.md").read_text(encoding="utf-8")
    assert "Explicit types: Stairs" in summary and "Marker limit: 2" in summary
    assert main(["demo", "--output", str(tmp_path / "invalid"), "--no-plots",
                 "--obstacles", "auto", "stairs"]) == 1
    assert not (tmp_path / "invalid").exists()