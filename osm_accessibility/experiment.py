"""Reproducible experiments, archived inputs and post-hoc parameter sweeps."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .data import Dataset, canonical_json, sha256_file
from .errors import DataError, NoRouteFound
from .evaluation import MODEL_VERSION, Evaluator
from .models import Coordinate, Route
from .outputs import (
    collect_pois,
    edge_diagnostics,
    graph_fingerprint,
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


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_record() -> dict:
    dependencies = {}
    for name in ("pymongo", "matplotlib", "pyproj", "defusedxml", "ijson"):
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
            "implementation_sha256": implementation_fingerprint()}


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


def run_experiment(dataset: Dataset, start: Coordinate, end: Coordinate, profile: Profile,
                   parameters: CostParameters, output_dir: str | Path, *, algorithm: str = "astar",
                   snap_distance_m: float = 150, plots: bool = True, baselines: bool = True,
                   file_formats: tuple[str, ...] = ("png", "pdf")) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise DataError("Output directory already exists. Use a new run directory to preserve previous results")
    graph = dataset.graph()
    if not graph.edges:
        raise DataError("Dataset produced no routing graph")
    evaluator = Evaluator(profile, parameters)
    start_snap, end_snap = snap_endpoints(graph, start, end, evaluator, snap_distance_m)
    # Exclude snapping and full-graph diagnostics from the reported search time.
    evaluator = Evaluator(profile, parameters)
    route = find_route(graph, start_snap.node_id, end_snap.node_id, evaluator, algorithm)
    comparisons = compare_baselines(graph, start_snap.node_id, end_snap.node_id, profile, parameters) if baselines else {}
    diagnostics = edge_diagnostics(graph, route, evaluator)
    pois = collect_pois(graph, route, diagnostics)
    manifest = {
        "format_version": 1, "model_version": MODEL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_record(), "source_revision": revision_record(),
        "dataset": dataset.metadata | {"sha256": dataset.fingerprint(), "counts": dataset.counts()},
        "graph_sha256": graph_fingerprint(graph), "graph_counts": {"nodes": len(graph.nodes), "directed_edges": len(graph.edges)},
        "profile": profile.as_dict(), "cost_parameters": parameters.as_dict(),
        "profile_sha256": hashlib.sha256(canonical_json(profile.as_dict()).encode()).hexdigest(),
        "snapping": {"start": start_snap.as_dict(), "end": end_snap.as_dict(), "max_distance_m": snap_distance_m},
        "route": route_summary(route, evaluator), "baselines": comparisons,
        "graph_blocked_directed_edges": sum(item["blocked"] for item in diagnostics.values()),
        "pois_on_route": [poi["id"] for poi in pois if poi["on_route"]],
        "limitations": ["Scores are heuristic preference matches, not measured accessibility or safety probabilities.",
                        "Only recorded obstacles are visible; absence of a POI is not proof of absence.",
                        "Snapping gaps are not verified and are excluded from GPX and distance.",
                        "Search timing excludes ingestion, snapping, diagnostics and plotting."],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset.json", {"format_version": 1, "metadata": dataset.metadata,
                                         "elements": list(dataset.elements())})
    save_diagnostics(output, graph, route, diagnostics, pois)
    if plots:
        from .visualization import create_figures
        create_figures(graph, route, diagnostics, pois, title=f"{profile.name}; factor={parameters.accessibility_factor:g}",
                       synthetic=bool(dataset.metadata.get("synthetic", False)), output_dir=output, file_formats=file_formats)
    manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "manifest.json", manifest)  # Written last: only a completed run has this file.
    return manifest


def load_run(run_dir: str | Path):
    root = Path(run_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        raw = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != 1 or raw["format_version"] != 1 or manifest["model_version"] != MODEL_VERSION:
            raise DataError("Unsupported saved run or scoring-model version")
        if manifest["environment"]["implementation_sha256"] != implementation_fingerprint():
            raise DataError("Implementation differs from the recorded run; use the recorded revision to reproduce it")
        hashes = manifest["artifact_sha256"]
        if not isinstance(hashes, dict) or not REQUIRED_ARTIFACTS.issubset(hashes):
            raise DataError("Saved manifest is missing required artifact checksums")
        for name, expected in hashes.items():
            if (not re_safe_name(name) or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
                    or not (root / name).is_file() or (root / name).is_symlink() or sha256_file(root / name) != expected):
                raise DataError("A recorded artifact is missing or its checksum changed")
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
        return dataset, graph, route, evaluator, manifest
    except DataError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise DataError("Cannot read a complete, valid saved research run") from None


def re_safe_name(name: str) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None


def replot(run_dir: str | Path, output_dir: str | Path, file_formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    from .visualization import create_figures
    dataset, graph, route, evaluator, _ = load_run(run_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = edge_diagnostics(graph, route, evaluator)
    return create_figures(graph, route, diagnostics, collect_pois(graph, route, diagnostics), title=evaluator.profile.name,
                          synthetic=bool(dataset.metadata.get("synthetic", False)), output_dir=output, file_formats=file_formats)


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