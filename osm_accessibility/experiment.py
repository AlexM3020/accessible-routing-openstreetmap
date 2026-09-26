"""Reproducible experiments, archived inputs and post-hoc parameter sweeps."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from . import __version__
from .data import Dataset, canonical_json, sha256_file
from .errors import DataError, NoRouteFound
from .evaluation import MODEL_VERSION, Evaluator
from .models import Coordinate, Route
from .outputs import (
    PresentationOptions,
    build_route_comparison,
    build_route_presentation,
    collect_pois,
    edge_diagnostics,
    graph_fingerprint,
    render_route_summary,
    route_summary,
    save_diagnostics,
    write_json,
)
from .profiles import CostParameters, Profile
from .routing import find_route, snap_endpoints

REQUIRED_ARTIFACTS = frozenset({
    "dataset.json", "route.gpx", "route.geojson", "route_segments.csv", "all_edges.csv",
    "criteria.csv", "edge_diagnostics.json", "pois.csv", "pois.json",
})
ARCHIVE_NAME = "reproducibility.zip"
ARCHIVE_ARTIFACTS = REQUIRED_ARTIFACTS | {"route_summary.md", "presentation.json"}
SCIENCE_FILES = frozenset({
    "builder.py", "data.py", "evaluation.py", "geo.py", "graph.py", "models.py", "profiles.py",
    "routing.py", "tags.py",
})
# Version 0.1.0's core was verified unchanged during this presentation migration.
# Unknown older implementations still fail instead of silently changing a study.
LEGACY_IMPLEMENTATIONS = {
    "48c3d801de7ad72924c28fde02d5c5f8194ce996bfe2810b0bb3daad7c3823e6":
        "ef69ad5f613476c4e92a60e718a9cf564ac6c3cf9847c7947fd21bf2e36d6362",
}


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def scientific_fingerprint() -> str:
    """Presentation edits may change; routing/data/model code must remain identical."""
    digest = hashlib.sha256()
    for name in sorted(SCIENCE_FILES):
        path = Path(__file__).parent / name
        digest.update(name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_record() -> dict:
    dependencies = {}
    for name in ("pymongo", "matplotlib", "pyproj", "shapely", "defusedxml", "ijson"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "package_version": __version__,
            "dependencies": dependencies,
            "installed_distributions": dict(sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name")
            )),
            "implementation_sha256": implementation_fingerprint(),
            "scientific_sha256": scientific_fingerprint()}


def revision_record() -> dict:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return {"commit": None, "uncommitted_changes": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
        return {"commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "uncommitted_changes": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "uncommitted_changes": None}


def compare_baselines(graph, start: str, goal: str, profile: Profile, parameters: CostParameters) -> dict:
    runs = {}
    for label, algorithm, distance_only in (("astar", "astar", False), ("dijkstra", "dijkstra", False),
                                            ("distance_only_same_constraints", "dijkstra", True)):
        evaluator = Evaluator(profile, parameters)  # Fresh evaluation cache per search.
        route = find_route(graph, start, goal, evaluator, algorithm, distance_only)
        runs[label] = route_summary(route, evaluator)
    if not math.isclose(runs["astar"]["optimized_cost_m"], runs["dijkstra"]["optimized_cost_m"], rel_tol=1e-10, abs_tol=1e-6):
        raise DataError("A* and Dijkstra disagree on optimal cost; inspect graph/evaluator")
    return runs


def _standard_profile() -> Profile:
    return Profile(name="No preferences", weights={}, wheelchair_required=False, prefer_handrail=False)


def _corridor_radius(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 5000:
        raise DataError("corridor_radius_m must be finite, greater than 0 and at most 5000 metres")
    return float(value)


def _stored_standard(graph, selected: Route, record: dict, parameters: CostParameters) -> Route:
    """Reconstruct and check the archived trace, not a new route on replot."""
    profile = _standard_profile()
    if record["profile"] != profile.as_dict() or record["objective"] != "distance_only":
        raise DataError("Invalid no-preference baseline definition")
    summary = record["route"]
    if summary["algorithm"] not in {"astar", "dijkstra"}:
        raise DataError("Invalid standard search algorithm")
    edges = {edge.id: edge for edge in graph.edges}
    route = Route(tuple(summary["node_ids"]), tuple(edges[key] for key in summary["edge_ids"]),
                  summary["distance_m"], summary["optimized_cost_m"], algorithm=summary["algorithm"])
    if (route.nodes[0], route.nodes[-1]) != (selected.nodes[0], selected.nodes[-1]):
        raise DataError("Standard route must use the same snapped endpoints")
    recomputed = route_summary(route, Evaluator(profile, parameters))
    for key in ("profile_cost_m", "score_percent", "coverage_percent", "hard_block_violations"):
        if recomputed[key] != summary[key]:
            raise DataError("Standard route metrics do not match the recorded graph")
    if not math.isclose(route.cost, route.distance_m, rel_tol=1e-10, abs_tol=1e-6):
        raise DataError("Standard route objective must equal mapped distance")
    return route


def run_experiment(dataset: Dataset, start: Coordinate, end: Coordinate, profile: Profile,
                   parameters: CostParameters, output_dir: str | Path, *, algorithm: str = "astar",
                   snap_distance_m: float = 150, plots: bool = True, baselines: bool = True,
                   file_formats: tuple[str, ...] = ("png",),
                   presentation_options: PresentationOptions | None = None, detailed: bool = False,
                   corridor_radius_m: float = 150.0) -> dict:
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise DataError("Output directory already exists. Use a new run directory to preserve previous results")
    options = presentation_options or PresentationOptions()
    if not isinstance(options, PresentationOptions):
        raise DataError("presentation_options must be PresentationOptions")
    radius = _corridor_radius(corridor_radius_m)
    graph = dataset.graph()
    if not graph.edges:
        raise DataError("Dataset produced no routing graph")
    evaluator = Evaluator(profile, parameters)
    start_snap, end_snap = snap_endpoints(graph, start, end, evaluator, snap_distance_m)
    # Exclude snapping and full-graph diagnostics from the reported search time.
    evaluator = Evaluator(profile, parameters)
    route = find_route(graph, start_snap.node_id, end_snap.node_id, evaluator, algorithm)
    standard_profile = _standard_profile()
    standard_evaluator = Evaluator(standard_profile, parameters)
    standard_route = find_route(graph, start_snap.node_id, end_snap.node_id, standard_evaluator,
                                algorithm, distance_only=True)
    comparisons = compare_baselines(graph, start_snap.node_id, end_snap.node_id, profile, parameters) if baselines else {}
    diagnostics = edge_diagnostics(graph, route, evaluator)
    pois = collect_pois(graph, route, diagnostics)
    manifest = {
        "format_version": 3, "model_version": MODEL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_record(), "source_revision": revision_record(),
        "dataset": dataset.metadata | {"sha256": dataset.fingerprint(), "counts": dataset.counts()},
        "graph_sha256": graph_fingerprint(graph), "graph_counts": {"nodes": len(graph.nodes), "directed_edges": len(graph.edges)},
        "profile": profile.as_dict(), "cost_parameters": parameters.as_dict(),
        "profile_sha256": hashlib.sha256(canonical_json(profile.as_dict()).encode()).hexdigest(),
        "snapping": {"start": start_snap.as_dict(), "end": end_snap.as_dict(), "max_distance_m": snap_distance_m},
        "route": route_summary(route, evaluator), "baselines": comparisons,
        "standard_route": {"objective": "distance_only", "profile": standard_profile.as_dict(),
                   "route": route_summary(standard_route, standard_evaluator)},
        "graph_blocked_directed_edges": sum(item["blocked"] for item in diagnostics.values()),
        "pois_on_route": [poi["id"] for poi in pois if poi["on_route"]],
        "limitations": ["Scores are heuristic preference matches, not measured accessibility or safety probabilities.",
                        "Only recorded obstacles are visible; absence of a POI is not proof of absence.",
                        "Snapping gaps are not verified and are excluded from GPX and distance.",
                        "Search timing excludes ingestion, snapping, diagnostics and plotting."],
    }
    presentation = build_route_presentation(
        graph, route, diagnostics, profile, parameters, options=options, baselines=comparisons,
        snapping=manifest["snapping"], synthetic=bool(dataset.metadata.get("synthetic", False)),
        include_context=False,
    )
    presentation["comparison"] = build_route_comparison(
        graph, route, standard_route, diagnostics, profile, parameters, selected_presentation=presentation,
    )
    manifest["presentation"] = presentation
    manifest["output"] = {"mode": "compact", "detailed": detailed, "file_formats": list(file_formats) if plots else [],
                          "options": options.as_dict(), "archive": ARCHIVE_NAME, "corridor_radius_m": radius,
                          "marker_scope": "selected_and_standard_routes"}
    output.mkdir(parents=True, exist_ok=False)
    # Keep all reproducibility evidence together, not spread across the result folder.
    with TemporaryDirectory(prefix=".archive-work-", dir=output) as staging_name:
        staging = Path(staging_name)
        write_json(staging / "dataset.json", {"format_version": 1, "metadata": dataset.metadata,
                                             "elements": list(dataset.elements())})
        save_diagnostics(staging, graph, route, diagnostics, pois)
        write_json(staging / "presentation.json", presentation)
        with (staging / "route_summary.md").open("x", encoding="utf-8") as stream:
            stream.write(render_route_summary(presentation))
        for name in ("route.gpx", "route_summary.md"):
            _copy_new(staging / name, output / name)
        if plots:
            from .visualization import create_figures
            create_figures(graph, route, diagnostics, pois, title=f"{profile.name}; factor={parameters.accessibility_factor:g}",
                           synthetic=bool(dataset.metadata.get("synthetic", False)), output_dir=output,
                           file_formats=file_formats, presentation=presentation,
                           standard_route=standard_route, corridor_radius_m=radius)
        manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in sorted(staging.iterdir())}
        manifest["deliverable_sha256"] = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
        write_json(staging / "manifest.json", manifest)
        if detailed:
            details = output / "diagnostics"
            details.mkdir()
            for path in sorted(staging.iterdir()):
                _copy_new(path, details / path.name)
        # Archive is the completion marker. Exclusive creation never replaces a run.
        archive_path = output / ARCHIVE_NAME
        created = False
        try:
            with archive_path.open("xb") as stream:
                created = True
                with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                    for path in sorted(staging.iterdir()):
                        archive.write(path, arcname=path.name)
        except BaseException:
            if created:
                archive_path.unlink(missing_ok=True)
            raise
    return manifest


def _copy_new(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def _validate_hashes(hashes: object, required: frozenset[str]) -> dict:
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise DataError("Saved manifest is missing required artifact checksums")
    for name, expected in hashes.items():
        if not re_safe_name(name) or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise DataError("Invalid artifact name or checksum")
    return hashes


def _read_saved_inputs(root: Path) -> tuple[dict, dict]:
    standalone = root.is_file()
    archive_path = root if standalone else root / ARCHIVE_NAME
    if archive_path.is_file():
        if archive_path.is_symlink():
            raise DataError("Research archive cannot be a symbolic link")
        with ZipFile(archive_path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != ARCHIVE_ARTIFACTS | {"manifest.json"}:
                raise DataError("Archive has missing, duplicate or unexpected members")
            if any(entry.flag_bits & 1 or entry.file_size > 1024**3 for entry in entries) or sum(
                entry.file_size for entry in entries
            ) > 2 * 1024**3:
                raise DataError("Encrypted or excessively large research archive")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format_version") not in {2, 3}:
                raise DataError("Unsupported compact archive version")
            hashes = _validate_hashes(manifest["artifact_sha256"], ARCHIVE_ARTIFACTS)
            if set(hashes) != ARCHIVE_ARTIFACTS:
                raise DataError("Unexpected archived artifact declaration")
            for name, expected in hashes.items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise DataError("An archived artifact checksum changed")
            raw = json.loads(archive.read("dataset.json"))
            if json.loads(archive.read("presentation.json")) != manifest["presentation"]:
                raise DataError("Archived route presentation differs from its manifest")
            deliverables = _validate_hashes(manifest["deliverable_sha256"], frozenset({"route.gpx", "route_summary.md"}))
            if not standalone:
                for name, expected in deliverables.items():
                    path = root / name
                    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                        raise DataError("A visible result file is missing or its checksum changed")
        return manifest, raw
    # Legacy 0.1 results remain readable after verifying their known scientific core.
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise DataError("Unsupported saved run version or missing completion archive")
    hashes = _validate_hashes(manifest["artifact_sha256"], REQUIRED_ARTIFACTS)
    for name, expected in hashes.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise DataError("A recorded artifact is missing or its checksum changed")
    return manifest, json.loads((root / "dataset.json").read_text(encoding="utf-8"))


def load_run(run_dir: str | Path):
    root = Path(run_dir)
    try:
        manifest, raw = _read_saved_inputs(root)
        if raw["format_version"] != 1 or manifest["model_version"] != MODEL_VERSION:
            raise DataError("Unsupported saved run or scoring-model version")
        recorded_core = manifest["environment"].get("scientific_sha256")
        if recorded_core is None:
            recorded_core = LEGACY_IMPLEMENTATIONS.get(manifest["environment"]["implementation_sha256"])
        if recorded_core != scientific_fingerprint():
            raise DataError("Scientific implementation differs from the recorded run; use the recorded revision")
        dataset = Dataset.from_elements(raw["elements"], raw["metadata"])
        if dataset.fingerprint() != manifest["dataset"]["sha256"] or dataset.counts() != manifest["dataset"]["counts"]:
            raise DataError("Saved input fingerprint mismatch")
        graph = dataset.graph()
        if graph_fingerprint(graph) != manifest["graph_sha256"]:
            raise DataError("Saved graph fingerprint mismatch")
        profile = Profile.from_dict(manifest["profile"])
        if hashlib.sha256(canonical_json(profile.as_dict()).encode()).hexdigest() != manifest["profile_sha256"]:
            raise DataError("Saved profile fingerprint mismatch")
        evaluator = Evaluator(profile, CostParameters.from_dict(manifest["cost_parameters"]))
        route_data = manifest["route"]
        if route_data["algorithm"] not in {"astar", "dijkstra"}:
            raise DataError("Unsupported saved search algorithm")
        edges = {edge.id: edge for edge in graph.edges}
        route = Route(tuple(route_data["node_ids"]), tuple(edges[key] for key in route_data["edge_ids"]),
                      route_data["distance_m"], route_data["optimized_cost_m"], algorithm=route_data["algorithm"])
        if route.nodes != (route.nodes[0], *(edge.target for edge in route.edges)) or any(
            edge.source != node for edge, node in zip(route.edges, route.nodes)
        ):
            raise DataError("Saved route is not a connected sequence")
        recomputed = route_summary(route, evaluator)
        for key in ("profile_cost_m", "way_risk_metres", "event_risk_metres", "score_exposure_m",
                    "known_exposure_m", "score_percent", "coverage_percent", "event_count", "hard_block_violations"):
            actual, recorded = recomputed[key], route_data[key]
            matches = (recorded is None if actual is None else
                       not isinstance(recorded, bool) and isinstance(recorded, (int, float))
                       and math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-6))
            if not matches:
                raise DataError(f"Saved route metric {key} does not match its recorded graph/profile")
        if not math.isclose(route.cost, recomputed["profile_cost_m"], rel_tol=1e-10, abs_tol=1e-6):
            raise DataError("Saved optimized cost does not match the selected path")
        for label, node_id in (("start", route.nodes[0]), ("end", route.nodes[-1])):
            snap = manifest["snapping"][label]
            if snap["node_id"] != node_id or snap["snapped_lat_lon"] != graph.nodes[node_id].coordinate.as_lat_lon():
                raise DataError("Saved snap endpoint does not match the route")
        if manifest["format_version"] == 3:
            standard = _stored_standard(graph, route, manifest["standard_route"], evaluator.parameters)
            _corridor_radius(manifest["output"]["corridor_radius_m"])
            diagnostics = edge_diagnostics(graph, route, evaluator)
            comparison = build_route_comparison(graph, route, standard, diagnostics, profile, evaluator.parameters)
            if comparison != manifest["presentation"]["comparison"]:
                raise DataError("Saved route comparison does not match its recorded paths/profile")
        return dataset, graph, route, evaluator, manifest
    except DataError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError, BadZipFile):
        raise DataError("Cannot read a complete, valid saved research run") from None


def re_safe_name(name: str) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None


def replot(run_dir: str | Path, output_dir: str | Path, file_formats: tuple[str, ...] = ("png",), *,
           presentation_options: PresentationOptions | None = None,
           presentation_overrides: dict | None = None, corridor_radius_m: float | None = None) -> list[Path]:
    from .visualization import create_figures
    dataset, graph, route, evaluator, manifest = load_run(run_dir)
    if presentation_options is not None and presentation_overrides:
        raise DataError("Provide full presentation options or partial overrides, not both")
    options = presentation_options or PresentationOptions.from_dict(
        manifest.get("output", {}).get("options", {}) | (presentation_overrides or {})
    )
    radius = _corridor_radius(corridor_radius_m if corridor_radius_m is not None else
                              manifest.get("output", {}).get("corridor_radius_m", 150.0))
    standard = (_stored_standard(graph, route, manifest["standard_route"], evaluator.parameters)
                if manifest.get("format_version") == 3 else None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = edge_diagnostics(graph, route, evaluator)
    presentation = build_route_presentation(
        graph, route, diagnostics, evaluator.profile, evaluator.parameters, options=options,
        baselines=manifest["baselines"], snapping=manifest["snapping"], synthetic=bool(dataset.metadata.get("synthetic", False)),
        include_context=standard is None,
    )
    if standard is not None:
        presentation["comparison"] = build_route_comparison(
            graph, route, standard, diagnostics, evaluator.profile, evaluator.parameters,
            selected_presentation=presentation,
        )
    with (output / "route_summary.md").open("x", encoding="utf-8") as stream:
        stream.write(render_route_summary(presentation))
    return create_figures(graph, route, diagnostics, collect_pois(graph, route, diagnostics), title=evaluator.profile.name,
                          synthetic=bool(dataset.metadata.get("synthetic", False)), output_dir=output,
                          file_formats=file_formats, presentation=presentation,
                          standard_route=standard, corridor_radius_m=radius)


def parameter_sweep(run_dir: str | Path, output_dir: str | Path, factors: list[float], unknown_risks: list[float], repeats: int = 3) -> dict:
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 100:
        raise DataError("repeats must be between 1 and 100")
    output = Path(output_dir)
    if output.exists():
        raise DataError("Output directory already exists; use a new sweep directory")
    _, graph, original, evaluator, manifest = load_run(run_dir)
    rows = []
    for factor in factors:
        for unknown in unknown_risks:
            parameters = replace(evaluator.parameters, accessibility_factor=factor, unknown_risk=unknown)
            for repeat in range(repeats):
                compared = {}
                for algorithm, distance_only in (("astar", False), ("dijkstra", False), ("distance_only", True)):
                    local = Evaluator(evaluator.profile, parameters)
                    try:
                        route = find_route(graph, original.nodes[0], original.nodes[-1], local,
                                           "dijkstra" if distance_only else algorithm, distance_only)
                        summary = route_summary(route, local)
                        compared[algorithm] = route.cost
                        rows.append({"factor": factor, "unknown_risk": unknown, "repeat": repeat, "algorithm": algorithm,
                                     "status": "ok", **{key: summary[key] for key in ("distance_m", "optimized_cost_m", "profile_cost_m", "score_percent", "coverage_percent", "expanded_nodes", "search_seconds")}})
                    except NoRouteFound:
                        compared[algorithm] = None
                        rows.append({"factor": factor, "unknown_risk": unknown, "repeat": repeat, "algorithm": algorithm, "status": "no_route"})
                if (compared["astar"] is None) != (compared["dijkstra"] is None) or (
                    compared["astar"] is not None and not math.isclose(compared["astar"], compared["dijkstra"], rel_tol=1e-10, abs_tol=1e-6)
                ):
                    raise DataError("A* and Dijkstra disagreed during the parameter sweep")
    if not rows:
        raise DataError("Provide at least one cost factor and unknown-risk value")
    output.mkdir(parents=True, exist_ok=False)
    fields = ["factor", "unknown_risk", "repeat", "algorithm", "status", "distance_m", "optimized_cost_m", "profile_cost_m",
              "score_percent", "coverage_percent", "expanded_nodes", "search_seconds"]
    with (output / "sensitivity.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    record = {"source_graph_sha256": manifest["graph_sha256"], "profile": evaluator.profile.as_dict(),
              "environment": environment_record(), "rows": rows,
              "note": "Same snapped endpoints and hard constraints. Fresh caches; search times are not end-to-end latency. Scores across profiles/parameter settings are not empirical accuracy."}
    write_json(output / "sensitivity.json", record)
    return record