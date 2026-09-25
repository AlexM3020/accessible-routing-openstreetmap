"""Command-line entry points for import, route generation and offline diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_osm
from .demo import synthetic_dataset
from .errors import ResearchError
from .experiment import parameter_sweep, replot, run_experiment
from .models import Coordinate
from .mongo import connect, import_dataset, load_dataset
from .profiles import CostParameters, Profile, profile_named


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Accessibility-aware OpenStreetMap routing and scientific diagnostics")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an artificial graph without MongoDB or downloads")
    route = commands.add_parser("route", help="Route from a complete local OSM extract or named MongoDB snapshot")
    for command in (demo, route):
        command.add_argument("--output", type=Path, required=True, help="New directory; never overwrite a prior run")
        command.add_argument("--profile", choices=("wheelchair", "walking", "distance"), default="wheelchair")
        command.add_argument("--profile-file", type=Path, help="JSON profile overrides the built-in profile")
        command.add_argument("--factor", type=float, default=1.5, help="Nonnegative distance/risk trade-off (default 1.5)")
        command.add_argument("--unknown-risk", type=float, default=.5)
        command.add_argument("--event-metres", type=float, default=20)
        command.add_argument("--snap-distance", type=float, default=150)
        command.add_argument("--algorithm", choices=("astar", "dijkstra"), default="astar")
        command.add_argument("--no-plots", action="store_true")
        command.add_argument("--no-baselines", action="store_true")
        command.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    source = route.add_mutually_exclusive_group(required=True)
    source.add_argument("--osm", type=Path)
    source.add_argument("--dataset", help="Named ready snapshot in MongoDB")
    route.add_argument("--database", default="OSMResearch")
    route.add_argument("--start", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    route.add_argument("--end", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    importer = commands.add_parser("import-osm", help="Validate an extract; writes only when --write is supplied")
    importer.add_argument("input", type=Path)
    importer.add_argument("--dataset", required=True)
    importer.add_argument("--database", default="OSMResearch")
    importer.add_argument("--write", action="store_true")
    plotting = commands.add_parser("plot", help="Replot an archived run after fingerprint verification")
    plotting.add_argument("run", type=Path)
    plotting.add_argument("--output", type=Path, required=True)
    plotting.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    sweep = commands.add_parser("sweep", help="Compare algorithms, trade-off factors and unknown-data assumptions")
    sweep.add_argument("run", type=Path)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--factors", type=float, nargs="+", default=[0, 1.5, 6])
    sweep.add_argument("--unknown-risks", type=float, nargs="+", default=[.25, .5, .75])
    sweep.add_argument("--repeats", type=int, default=3)
    query = commands.add_parser("query", help="Print a complete-geometry Overpass QL query; makes no network request")
    query.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        formats = ("png", "pdf") if getattr(args, "format", "both") == "both" else (args.format,)
        if args.command == "query":
            south, west, north, east = args.bbox
            Coordinate(south, west)
            Coordinate(north, east)
            if south >= north or west >= east:
                raise ValueError("Use a non-antimeridian bounding box with south<north and west<east")
            bbox = ",".join(f"{value:g}" for value in args.bbox)
            print(f'[out:json][timeout:180];\n(way["highway"]({bbox}); relation["highway"="pedestrian"]({bbox}););\n(._; >>;);\nout body;')
            return 0
        if args.command == "import-osm":
            dataset = load_osm(args.input)
            if args.write:
                with connect(args.database) as database:
                    report = import_dataset(dataset, database, args.dataset, write=True)
            else:
                report = import_dataset(dataset, None, args.dataset)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "plot":
            paths = replot(args.run, args.output, formats)
            print(f"Saved {len(paths)} figures to {args.output}")
            return 0
        if args.command == "sweep":
            record = parameter_sweep(args.run, args.output, args.factors, args.unknown_risks, args.repeats)
            print(f"Saved {len(record['rows'])} benchmark observations to {args.output}")
            return 0
        profile = (Profile.from_dict(json.loads(args.profile_file.read_text(encoding="utf-8")))
                   if args.profile_file else profile_named(args.profile))
        parameters = CostParameters(args.factor, args.unknown_risk, args.event_metres)
        if args.command == "demo":
            dataset = synthetic_dataset()
            start, end = Coordinate(49.58, 11), Coordinate(49.58, 11.002)
        else:
            if args.osm:
                dataset = load_osm(args.osm)
            else:
                with connect(args.database) as database:
                    dataset = load_dataset(database, args.dataset)
            start, end = Coordinate.from_lat_lon(args.start), Coordinate.from_lat_lon(args.end)
        manifest = run_experiment(dataset, start, end, profile, parameters, args.output,
                                  algorithm=args.algorithm, snap_distance_m=args.snap_distance,
                                  plots=not args.no_plots, baselines=not args.no_baselines, file_formats=formats)
        print(json.dumps({"output": str(args.output), "synthetic": dataset.metadata.get("synthetic", False),
                          "graph_sha256": manifest["graph_sha256"], "route": manifest["route"]}, indent=2))
        return 0
    except (ResearchError, OSError, ValueError) as exc:
        # DatabaseError already sanitizes driver errors. No URI command-line option.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())