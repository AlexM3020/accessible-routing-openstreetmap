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
from .outputs import OBSTACLE_KINDS, PresentationOptions
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
        command.add_argument("--no-baselines", action="store_true", help="Skip numerical oracle checks; retain the standard-route comparison")
        command.add_argument("--format", choices=("png", "pdf", "both"), default="png")
        command.add_argument("--details", action="store_true", help="Also expand raw CSV/JSON evidence into diagnostics/")
        command.add_argument("--json", action="store_true", help="Print full run metadata instead of a brief summary")
        _presentation_arguments(command)
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
    plotting.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    _presentation_arguments(plotting)
    sweep = commands.add_parser("sweep", help="Compare algorithms, trade-off factors and unknown-data assumptions")
    sweep.add_argument("run", type=Path)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--factors", type=float, nargs="+", default=[0, 1.5, 6])
    sweep.add_argument("--unknown-risks", type=float, nargs="+", default=[.25, .5, .75])
    sweep.add_argument("--repeats", type=int, default=3)
    query = commands.add_parser("query", help="Print a complete-geometry Overpass QL query; makes no network request")
    query.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    return root


def _presentation_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--corridor-radius", type=float, default=None, metavar="METRES",
                         help="Map buffer around both routes, default 150 m (0 < radius <= 5000); does not restrict routing")
    command.add_argument("--obstacles", nargs="+", choices=("auto", "all", "none", *OBSTACLE_KINDS),
                         help="Marker types only, e.g. stairs barrier kerb; auto is important/preference-relevant features")
    command.add_argument("--max-markers", type=int, default=None, help="Map marker budget, default 12 (0-100); complete report is never truncated")


def _presentation_overrides(args) -> dict:
    kinds = args.obstacles
    values = {}
    if kinds is None:
        pass
    elif kinds == ["auto"]:
        selected = None
        values["obstacle_kinds"] = selected
    elif kinds == ["all"]:
        values["obstacle_kinds"] = list(OBSTACLE_KINDS)
    elif kinds == ["none"]:
        values["obstacle_kinds"] = []
    else:
        if any(kind in {"all", "auto", "none"} for kind in kinds):
            raise ValueError("Use auto/all/none alone, or select individual obstacle types")
        values["obstacle_kinds"] = kinds
    if args.max_markers is not None:
        values["max_markers"] = args.max_markers
    return values


def _presentation_options(args) -> PresentationOptions:
    return PresentationOptions.from_dict(_presentation_overrides(args))


def _print_result(manifest: dict, output: Path) -> None:
    view = manifest["presentation"]
    metrics = view["metrics"]
    score = "not defined" if metrics["score_percent"] is None else f"{metrics['score_percent']:.1f}%"
    coverage = "not defined" if metrics["coverage_percent"] is None else f"{metrics['coverage_percent']:.1f}%"
    print("SYNTHETIC EXAMPLE — not observed map conditions" if view["synthetic"] else "Route result — based on recorded map data")
    print(f"Profile: {view['profile_name']} | Distance: {metrics['distance_m']:.1f} m | Match: {score} | Coverage: {coverage}")
    paired = view.get("comparison")
    if paired:
        baseline = paired["standard"]
        print(f"Standard no-preference route: {baseline['distance_m']:.1f} m; "
              f"selected distance change {paired['distance_difference_m']:+.1f} m")
        print("Obstacle encounters              Selected  Standard  Change")
        for row in paired["obstacle_rows"]:
            if row["selected"] or row["standard"]:
                print(f"  {row['label']:<29} {row['selected']:>6} {row['standard']:>9} {row['difference']:>+7}")
        print(f"Known preference conflicts: {paired['selected']['conflict_count']} selected / "
              f"{baseline['conflict_count']} standard; missing observations: "
              f"{paired['selected']['unknown_count']} / {baseline['unknown_count']}.")
        if baseline["hard_block_violations"]:
            print("STANDARD ROUTE INFEASIBLE under selected hard requirements; not a recommended alternative.")
        if paired["same_path"]:
            print("Same path: preferences did not change the selected directed route.")
        print("Both paths assessed under your profile; counts are grouped encounters, not unique physical obstacles.")
    selected = "; ".join(f"{item['label']} ({item['weight']:g})" for item in view["selected_preferences"])
    print("Selected preferences: " + (selected or "none"))
    print(f"Known preference conflicts: {len(view['conflicts'])}; unknown observations: {len(view['unknowns'])}")
    encounters = view["encountered_obstacles"]
    for item in encounters[:8]:
        flag = "PREFERENCE CONFLICT" if item["preference_labels"] else "Encountered"
        print(f"  [{flag}] {item['label']} — {item['location']}")
    if len(encounters) > 8:
        print(f"  {len(encounters) - 8} further encounters are listed in route_summary.md")
    checks = view["validation"]
    agreement = checks["astar_dijkstra_agree"]
    comparison = "costs agree" if agreement is True else "COSTS DISAGREE" if agreement is False else "not compared"
    print(f"Model checks: {checks['hard_constraint_violations']} hard-block violations; A*/Dijkstra {comparison}.")
    print("These checks concern the graph and cost model, not physical safety or mapping completeness.")
    print(f"Read: {output / 'route_summary.md'}\nGPX: {output / 'route.gpx'}\nReproduction: {output / 'reproducibility.zip'}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        format_name = getattr(args, "format", "png")
        formats = ("png", "pdf") if format_name == "both" else (format_name,)
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
            paths = replot(args.run, args.output, formats, presentation_overrides=_presentation_overrides(args),
                           corridor_radius_m=args.corridor_radius)
            print(f"Saved {len(paths)} maps and route_summary.md to {args.output}")
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
                                  plots=not args.no_plots, baselines=not args.no_baselines, file_formats=formats,
                                  presentation_options=_presentation_options(args), detailed=args.details,
                                  corridor_radius_m=150.0 if args.corridor_radius is None else args.corridor_radius)
        if args.json:
            print(json.dumps(manifest, indent=2))
        else:
            _print_result(manifest, args.output)
        return 0
    except (ResearchError, OSError, ValueError) as exc:
        # DatabaseError already sanitizes driver errors. No URI command-line option.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())