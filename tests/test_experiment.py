import csv
import io
import json
import shutil
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from osm_accessibility.cli import main
from osm_accessibility.errors import DataError
from osm_accessibility.experiment import (
    ARCHIVE_NAME,
    LEGACY_IMPLEMENTATIONS,
    REQUIRED_ARTIFACTS,
    load_run,
    parameter_sweep,
    replot,
    run_experiment,
)
from osm_accessibility.models import Coordinate
from osm_accessibility.outputs import PresentationOptions
from osm_accessibility.profiles import CostParameters, Profile, wheelchair_profile


def run(dataset, output, **kwargs):
    return run_experiment(dataset, Coordinate(49.58, 11), Coordinate(49.58, 11.002),
                          wheelchair_profile(), CostParameters(accessibility_factor=6), output, **kwargs)


def read_manifest(directory):
    with ZipFile(directory / ARCHIVE_NAME) as archive:
        return json.loads(archive.read("manifest.json"))


def alter_archive(directory, change):
    path = directory / ARCHIVE_NAME
    with ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    change(contents)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)


def alter_manifest(directory, change):
    def update(contents):
        manifest = json.loads(contents["manifest.json"])
        change(manifest)
        contents["manifest.json"] = json.dumps(manifest).encode()
    alter_archive(directory, update)


def test_run_outputs_preserve_route_and_cost(dataset, tmp_path):
    directory = tmp_path / "run"
    manifest = run(dataset, directory, plots=False)
    assert manifest["dataset"]["synthetic"]
    assert manifest["route"]["hard_block_violations"] == 0
    assert manifest["baselines"]["astar"]["optimized_cost_m"] == pytest.approx(manifest["baselines"]["dijkstra"]["optimized_cost_m"])
    assert (directory / ARCHIVE_NAME).exists()
    _, graph, route, evaluator, _ = load_run(directory)
    points = ElementTree.parse(directory / "route.gpx").findall(".//{http://www.topografix.com/GPX/1/1}trkpt")
    coordinates = [[float(p.attrib["lat"]), float(p.attrib["lon"])] for p in points]
    assert coordinates == [graph.nodes[node].coordinate.as_lat_lon() for node in route.nodes]
    with ZipFile(directory / ARCHIVE_NAME) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("route_segments.csv").decode("utf-8"))))
        pois = json.loads(archive.read("pois.json"))
    assert [row["edge_id"] for row in rows] == [edge.id for edge in route.edges]
    assert sum(float(row["cost_m"]) for row in rows) == pytest.approx(route.cost)
    assert route.cost == pytest.approx(sum(evaluator(edge).cost_m for edge in route.edges))
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
    alter_manifest(directory, lambda manifest: manifest["route"].__setitem__(field, manifest["route"][field] + 10))
    with pytest.raises(DataError):
        load_run(directory)


def test_manifest_cannot_discard_required_artifact_hashes(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    alter_manifest(directory, lambda manifest: manifest.__setitem__("artifact_sha256", {}))
    with pytest.raises(DataError):
        load_run(directory)


def test_manifest_profile_hash_is_verified(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    alter_manifest(directory, lambda manifest: manifest["profile"].__setitem__("name", "a different study profile"))
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
    assert len(paths) == 2
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert (tmp_path / "plots" / "route_summary.md").exists()


def test_five_file_default_and_complete_evidence_archive(dataset, tmp_path):
    directory = tmp_path / "compact"
    manifest = run(dataset, directory)
    assert {path.name for path in directory.iterdir()} == {
        "route.gpx", "route_summary.md", "route_obstacles.png", "area_accessibility.png", ARCHIVE_NAME,
    }
    with ZipFile(directory / ARCHIVE_NAME) as archive:
        assert REQUIRED_ARTIFACTS.issubset(archive.namelist())
        assert "presentation.json" in archive.namelist()
        assert json.loads(archive.read("manifest.json")) == manifest
    text = (directory / "route_summary.md").read_text(encoding="utf-8")
    assert "Selected preferences" in text
    assert "PREFERENCE CONFLICT" in text
    assert "Missing data" in text
    assert "supplied costs agree" in text
    assert "sha256" not in text


def test_details_opt_in_and_png_pdf_are_separate_controls(dataset, tmp_path):
    directory = tmp_path / "detailed"
    run(dataset, directory, detailed=True, file_formats=("pdf",))
    assert (directory / "route_obstacles.pdf").read_bytes().startswith(b"%PDF")
    assert (directory / "area_accessibility.pdf").read_bytes().startswith(b"%PDF")
    assert not list(directory.glob("*.png"))
    assert (directory / "diagnostics" / "criteria.csv").is_file()
    assert (directory / "diagnostics" / "dataset.json").is_file()
    assert not (directory / "dataset.json").exists()
    load_run(directory)


def test_portable_archive_loads_without_visible_deliverables(dataset, tmp_path):
    directory = tmp_path / "original"
    run(dataset, directory, plots=False)
    standalone = tmp_path / "study.zip"
    shutil.copyfile(directory / ARCHIVE_NAME, standalone)
    _, _, route, _, manifest = load_run(standalone)
    assert list(route.nodes) == manifest["route"]["node_ids"]


def test_display_filters_never_change_algorithm_outputs(dataset, tmp_path):
    first = run(dataset, tmp_path / "a", plots=False)
    second = run(dataset, tmp_path / "b", plots=False,
                 presentation_options=PresentationOptions(("stairs",), 1))
    for key in ("node_ids", "edge_ids", "distance_m", "optimized_cost_m", "score_percent", "coverage_percent"):
        assert first["route"][key] == second["route"][key]
    assert first["presentation"]["conflicts"] == second["presentation"]["conflicts"]
    assert {row["kind"] for row in second["presentation"]["map_obstacles"]} <= {"stairs"}


def test_old_full_run_accepts_only_verified_unchanged_scientific_core(dataset, tmp_path):
    current = tmp_path / "new"
    run(dataset, current, plots=False)
    legacy = tmp_path / "old"
    legacy.mkdir()
    with ZipFile(current / ARCHIVE_NAME) as archive:
        for name in REQUIRED_ARTIFACTS:
            (legacy / name).write_bytes(archive.read(name))
    manifest = read_manifest(current)
    manifest["format_version"] = 1
    manifest["environment"].pop("scientific_sha256")
    manifest["environment"]["implementation_sha256"] = next(iter(LEGACY_IMPLEMENTATIONS))
    manifest["artifact_sha256"] = {key: value for key, value in manifest["artifact_sha256"].items() if key in REQUIRED_ARTIFACTS}
    (legacy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    load_run(legacy)
    manifest["environment"]["implementation_sha256"] = "unknown-old-code"
    (legacy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match="Scientific implementation differs"):
        load_run(legacy)


def test_corrupt_zip_reports_domain_error(dataset, tmp_path):
    directory = tmp_path / "damaged"
    run(dataset, directory, plots=False)
    (directory / ARCHIVE_NAME).write_bytes(b"not a zip")
    with pytest.raises(DataError):
        load_run(directory)


@pytest.mark.parametrize("name", ["unexpected.json", "../outside.json", "route.gpx"])
def test_unexpected_duplicate_or_traversal_archive_members_are_rejected(dataset, tmp_path, name):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    with ZipFile(directory / ARCHIVE_NAME, "a") as archive:
        if name == "route.gpx":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(name, b"changed")
        else:
            archive.writestr(name, b"unexpected")
    with pytest.raises(DataError, match="missing, duplicate or unexpected members"):
        load_run(directory)
    assert not (tmp_path / "outside.json").exists()


def test_missing_archive_member_is_not_accepted(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    alter_archive(directory, lambda contents: contents.pop("criteria.csv"))
    with pytest.raises(DataError, match="missing, duplicate or unexpected members"):
        load_run(directory)


@pytest.mark.parametrize("name", ["dataset.json", "presentation.json", "criteria.csv"])
def test_archived_evidence_checksums_are_verified(dataset, tmp_path, name):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    alter_archive(directory, lambda contents: contents.__setitem__(name, contents[name] + b" "))
    with pytest.raises(DataError, match="archived artifact checksum changed"):
        load_run(directory / ARCHIVE_NAME)


def test_manifest_and_archived_presentation_must_match(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory, plots=False)
    alter_manifest(directory, lambda manifest: manifest["presentation"].__setitem__("conflicts", []))
    with pytest.raises(DataError, match="presentation differs"):
        load_run(directory)


def test_missing_visible_map_does_not_invalidate_portable_archive(dataset, tmp_path):
    directory = tmp_path / "run"
    run(dataset, directory)
    (directory / "area_accessibility.png").unlink()
    with pytest.raises(DataError, match="visible result file"):
        load_run(directory)
    load_run(directory / ARCHIVE_NAME)


def test_interrupted_archive_has_no_completion_marker_or_leftover_staging(dataset, tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise OSError("simulated archive write interruption")

    monkeypatch.setattr(ZipFile, "write", interrupted)
    directory = tmp_path / "interrupted"
    with pytest.raises(OSError, match="simulated archive"):
        run(dataset, directory, plots=False)
    assert {path.name for path in directory.iterdir()} == {"route.gpx", "route_summary.md"}
    with pytest.raises(DataError):
        load_run(directory)
    with pytest.raises(DataError, match="already exists"):
        run(dataset, directory, plots=False)


def test_replot_display_overrides_preserve_saved_route(dataset, tmp_path):
    directory = tmp_path / "source"
    original = run(dataset, directory, plots=False)
    paths = replot(directory, tmp_path / "redrawn", presentation_options=PresentationOptions(("stairs",), 2))
    assert len(paths) == 2
    assert "Explicit types: Stairs" in (tmp_path / "redrawn" / "route_summary.md").read_text(encoding="utf-8")
    assert read_manifest(directory) == original


def test_cli_demo_and_error_handling(tmp_path, capsys):
    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output), "--no-plots"]) == 0
    console = capsys.readouterr().out
    assert "Selected preferences:" in console and "PREFERENCE CONFLICT" in console
    assert "graph_sha256" not in console
    assert main(["demo", "--output", str(output), "--no-plots"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["query", "--bbox", "49.57", "10.995", "49.61", "11.025"]) == 0
    assert ">>" in capsys.readouterr().out
    assert main(["query", "--bbox", "49.61", "10.995", "49.57", "11.025"]) == 1


def test_standard_comparison_same_endpoints_and_selected_profile_assessment(dataset, tmp_path):
    directory = tmp_path / "paired"
    result = run(dataset, directory, plots=False, corridor_radius_m=75)
    assert result["format_version"] == 3
    assert result["standard_route"]["objective"] == "distance_only"
    standard = result["standard_route"]["route"]
    assert standard["node_ids"] == ["s", "stairs_mid", "t"]
    assert standard["node_ids"][::len(standard["node_ids"]) - 1] == [
        result["route"]["node_ids"][0], result["route"]["node_ids"][-1],
    ]
    assert standard["optimized_cost_m"] == standard["distance_m"]
    assert standard["score_percent"] is None
    assert not result["standard_route"]["profile"]["wheelchair_required"]
    assert not any(result["standard_route"]["profile"]["weights"].values())
    compared = result["presentation"]["comparison"]
    assert compared["selected"]["obstacle_counts"]["stairs"] == 0
    assert compared["standard"]["obstacle_counts"]["stairs"] == 1
    assert compared["standard"]["hard_block_violations"] == 2
    assert compared["standard"]["score_percent"] is None
    assert compared["score_difference_pp"] is None
    assert compared["standard"]["unknown_count"] > 0
    assert result["output"]["corridor_radius_m"] == 75
    assert result["baselines"]["distance_only_same_constraints"]["distance_m"] > standard["distance_m"]
    load_run(directory)
    load_run(directory / ARCHIVE_NAME)


def test_no_baselines_still_generates_standard_comparison(dataset, tmp_path):
    result = run(dataset, tmp_path / "no-oracles", plots=False, baselines=False)
    assert result["baselines"] == {}
    assert result["presentation"]["validation"]["astar_dijkstra_agree"] is None
    assert result["presentation"]["comparison"]["status"] == "available"


def test_no_preferences_same_path_is_not_perfect_accessibility(dataset, tmp_path):
    result = run_experiment(dataset, Coordinate(49.58, 11), Coordinate(49.58, 11.002),
                            Profile(name="No preferences", prefer_handrail=False), CostParameters(),
                            tmp_path / "unweighted", plots=False)
    comparison = result["presentation"]["comparison"]
    assert comparison["same_path"]
    assert comparison["selected"] == comparison["standard"]
    assert comparison["selected"]["score_percent"] is None
    assert all(row["difference"] == 0 for row in comparison["obstacle_rows"])


@pytest.mark.parametrize("radius", [0, -1, 5001, True, float("nan"), float("inf")])
def test_invalid_corridor_rejected_before_creating_output(dataset, tmp_path, radius):
    output = tmp_path / "invalid"
    with pytest.raises(DataError, match="corridor_radius_m"):
        run(dataset, output, plots=False, corridor_radius_m=radius)
    assert not output.exists()


def test_display_controls_do_not_change_either_trace_or_comparison(dataset, tmp_path):
    first = run(dataset, tmp_path / "one", plots=False, corridor_radius_m=25)
    second = run(dataset, tmp_path / "two", plots=False, corridor_radius_m=500,
                 presentation_options=PresentationOptions((), 0))
    assert first["presentation"]["comparison"] == second["presentation"]["comparison"]
    for key in ("node_ids", "edge_ids", "distance_m", "optimized_cost_m", "score_percent"):
        assert first["route"][key] == second["route"][key]
        assert first["standard_route"]["route"][key] == second["standard_route"]["route"][key]


@pytest.mark.parametrize("corruption", ["profile", "objective", "cost", "path", "comparison", "radius"])
def test_standard_comparison_corruption_is_detected(dataset, tmp_path, corruption):
    output = tmp_path / "source"
    run(dataset, output, plots=False)

    def corrupt(contents):
        manifest = json.loads(contents["manifest.json"])
        standard = manifest["standard_route"]
        if corruption == "profile":
            standard["profile"]["wheelchair_required"] = True
        elif corruption == "objective":
            standard["objective"] = "preferences"
        elif corruption == "cost":
            standard["route"]["optimized_cost_m"] += 10
        elif corruption == "path":
            standard["route"]["edge_ids"].reverse()
        elif corruption == "radius":
            manifest["output"]["corridor_radius_m"] = -2
        else:
            # Rewrite both stored copies and their hash: the semantic comparison
            # check must detect a false count even when byte checksums agree.
            import hashlib
            manifest["presentation"]["comparison"]["standard"]["obstacle_counts"]["stairs"] = 0
            contents["presentation.json"] = json.dumps(manifest["presentation"]).encode()
            manifest["artifact_sha256"]["presentation.json"] = hashlib.sha256(contents["presentation.json"]).hexdigest()
        contents["manifest.json"] = json.dumps(manifest).encode()

    alter_archive(output, corrupt)
    with pytest.raises(DataError):
        load_run(output)


def test_replot_uses_recorded_standard_without_search_and_inherits_radius(dataset, tmp_path, monkeypatch):
    import osm_accessibility.experiment as experiment
    import osm_accessibility.visualization as visualization

    source = tmp_path / "source"
    saved = run(dataset, source, plots=False, corridor_radius_m=85)
    captured = []
    real_figures = visualization.create_figures

    def observe(*args, **kwargs):
        captured.append((kwargs["standard_route"], kwargs["corridor_radius_m"], kwargs["presentation"]))
        return real_figures(*args, **kwargs)

    def no_search(*args, **kwargs):
        pytest.fail("Replot must not reroute a recorded comparison")

    monkeypatch.setattr(experiment, "find_route", no_search)
    monkeypatch.setattr(visualization, "create_figures", observe)
    replot(source, tmp_path / "inherited")
    replot(source, tmp_path / "override", corridor_radius_m=30)
    assert [row[1] for row in captured] == [85, 30]
    assert list(captured[0][0].nodes) == saved["standard_route"]["route"]["node_ids"]
    assert captured[0][2]["comparison"] == captured[1][2]["comparison"]
    assert read_manifest(source) == saved


def test_version_two_archive_replots_without_inventing_a_standard_route(dataset, tmp_path, monkeypatch):
    import hashlib

    import osm_accessibility.experiment as experiment

    source = tmp_path / "old"
    run(dataset, source, plots=False)

    def old_format(contents):
        manifest = json.loads(contents["manifest.json"])
        manifest["format_version"] = 2
        manifest.pop("standard_route")
        manifest["presentation"].pop("comparison")
        contents["presentation.json"] = json.dumps(manifest["presentation"]).encode()
        manifest["artifact_sha256"]["presentation.json"] = hashlib.sha256(contents["presentation.json"]).hexdigest()
        contents["manifest.json"] = json.dumps(manifest).encode()

    alter_archive(source, old_format)
    monkeypatch.setattr(experiment, "find_route", lambda *args, **kwargs: pytest.fail("No invented old baseline"))
    replot(source, tmp_path / "old-figures")
    text = (tmp_path / "old-figures" / "route_summary.md").read_text(encoding="utf-8")
    assert "Selected vs no-preference shortest pedestrian route" not in text