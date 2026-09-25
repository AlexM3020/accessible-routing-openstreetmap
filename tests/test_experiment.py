import csv
import json
from xml.etree import ElementTree

import pytest

from osm_accessibility.cli import main
from osm_accessibility.errors import DataError
from osm_accessibility.experiment import load_run, parameter_sweep, replot, run_experiment
from osm_accessibility.models import Coordinate
from osm_accessibility.profiles import CostParameters, wheelchair_profile


def run(dataset, output, **kwargs):
    return run_experiment(dataset, Coordinate(49.58, 11), Coordinate(49.58, 11.002),
                          wheelchair_profile(), CostParameters(accessibility_factor=6), output, **kwargs)


def test_run_outputs_preserve_route_and_cost(dataset, tmp_path):
    directory = tmp_path / "run"
    manifest = run(dataset, directory, plots=False)
    assert manifest["dataset"]["synthetic"]
    assert manifest["route"]["hard_block_violations"] == 0
    assert manifest["baselines"]["astar"]["optimized_cost_m"] == pytest.approx(manifest["baselines"]["dijkstra"]["optimized_cost_m"])
    assert (directory / "manifest.json").exists()
    _, graph, route, evaluator, _ = load_run(directory)
    points = ElementTree.parse(directory / "route.gpx").findall(".//{http://www.topografix.com/GPX/1/1}trkpt")
    coordinates = [[float(p.attrib["lat"]), float(p.attrib["lon"])] for p in points]
    assert coordinates == [graph.nodes[node].coordinate.as_lat_lon() for node in route.nodes]
    rows = list(csv.DictReader((directory / "route_segments.csv").open(encoding="utf-8")))
    assert [row["edge_id"] for row in rows] == [edge.id for edge in route.edges]
    assert sum(float(row["cost_m"]) for row in rows) == pytest.approx(route.cost)
    assert route.cost == pytest.approx(sum(evaluator(edge).cost_m for edge in route.edges))
    pois = json.loads((directory / "pois.json").read_text(encoding="utf-8"))
    assert any(poi["kind"] == "stairs" and not poi["on_route"] for poi in pois)
    assert any(poi["kind"] == "barrier" and not poi["on_route"] for poi in pois)
    assert any(poi["kind"] == "crossing" and poi["on_route"] for poi in pois)


def test_no_overwrite_and_tamper_detection(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    with pytest.raises(DataError, match="already exists"):
        run(dataset, directory, plots=False)
    (directory / "route.gpx").write_text("modified")
    with pytest.raises(DataError, match="checksum"):
        load_run(directory)


def test_missing_manifest_rejects_incomplete_run(tmp_path):
    with pytest.raises(DataError):
        load_run(tmp_path)


@pytest.mark.parametrize("field", ["distance_m", "optimized_cost_m", "score_percent", "event_risk_metres"])
def test_manifest_numeric_alterations_are_detected(dataset, tmp_path, field):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["route"][field] += 10
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError):
        load_run(directory)


def test_manifest_cannot_discard_required_artifact_hashes(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError):
        load_run(directory)


def test_manifest_profile_hash_is_verified(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["profile"]["name"] = "a different study profile"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError):
        load_run(directory)


def test_parameter_sweep_retains_constraints_and_repeats(dataset, tmp_path):
    directory = tmp_path / "original"
    run(dataset, directory, plots=False)
    result = parameter_sweep(directory, tmp_path / "sweep", [0, 6], [.25, .75], repeats=2)
    assert len(result["rows"]) == 24
    for factor in [0, 6]:
        for unknown in [.25, .75]:
            rows = [row for row in result["rows"] if row["factor"] == factor and row["unknown_risk"] == unknown and row["repeat"] == 0]
            assert rows[0]["optimized_cost_m"] == pytest.approx(rows[1]["optimized_cost_m"])
    assert (tmp_path / "sweep" / "sensitivity.csv").exists()


def test_replot_uses_archived_graph_and_generates_real_files(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    paths = replot(directory, tmp_path / "plots", ("png",))
    assert len(paths) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_cli_demo_and_error_handling(tmp_path, capsys):
    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output), "--no-plots"]) == 0
    assert main(["demo", "--output", str(output), "--no-plots"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["query", "--bbox", "49.57", "10.995", "49.61", "11.025"]) == 0
    assert ">>" in capsys.readouterr().out
    assert main(["query", "--bbox", "49.61", "10.995", "49.57", "11.025"]) == 1