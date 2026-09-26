# Accessibility-aware routing on OpenStreetMap graphs

A Python research toolkit for reproducible, preference-dependent pedestrian pathfinding using OpenStreetMap (OSM) data. It provides a directed graph model, configurable accessibility costs, A* and Dijkstra search, safe MongoDB snapshot ingestion, GPX export, and static diagnostic figures.

**Research status:** the model implements explicit hypotheses about accessibility-related tags. Its scores are heuristic preference matches, **not measured accessibility, probabilities, or safety guarantees**. Synthetic examples demonstrate computations only; neither route is field-verified.

All executable code and examples are Python. Besides this README, only minimal Python dependency/packaging metadata and ignore rules are included. Generated data, figures, credentials and run outputs are not committed.

**Version 0.3.0 adds a no-preference standard-route comparison and corridor-focused maps.** Experiment orchestration computes an additional baseline; the underlying routing algorithms, graph construction, cost model, profiles and ingestion remain unchanged. The default result is still exactly five files, with complete reproducibility evidence in one archive. Corridor geometry and spatial indexing require **Shapely >=2.0,<3**, installed with the project dependencies.

## Quick start — no database required

Python **3.11 or newer** is required. The initial local verification uses Python 3.12. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m osm_accessibility demo --output results/demo --factor 6
```

On Linux/macOS, create the environment with `python3 -m venv .venv` and activate with `source .venv/bin/activate`.

The demo constructs a **clearly artificial** network with stairs, a ramp, a barrier, controlled/uncontrolled crossings, raised/lowered kerbs, rough/smooth surfaces, slopes, missing tags and a disconnected component. The coordinate labels are illustrative; these are not surveyed paths in Erlangen.

Every output directory must be new. Re-running against the same directory fails rather than silently replacing an experiment. PNG is the default; use `--format pdf` for two vector figures, or `--format both` when both image formats are useful. `--no-plots` produces just the GPX, summary and archive. `--no-baselines` skips only the matched-hard-constraint distance and A*/Dijkstra checks, marking them as not compared; **the standard route is always computed**, including with `--no-plots`.

## Outputs and graphical checks

### Five files by default

| Artifact | Contents |
| --- | --- |
| `route.gpx` | GPX 1.1 containing every selected graph node, in route order |
| `route_summary.md` | Selected preferences/requirements, both routes' complete obstacle/conflict/missing-data assessment, signed differences and numerical checks |
| `route_obstacles.png` | Selected and standard routes with relevant markers, comparison metrics and clipped corridor context |
| `area_accessibility.png` | Selected and standard route panels with the same corridor extent and selected-profile edge-score scale |
| `reproducibility.zip` | Complete normalized inputs, manifest, diagnostics, raw tables, both route traces and presentation settings for offline verification/replotting |

### Selected versus standard route

The standard route minimizes mapped distance between the **same snapped endpoints** as the selected route. It uses `Profile(name="No preferences", weights={}, wheelchair_required=False, prefer_handrail=False)`, `distance_only=True` and the selected search algorithm (`--algorithm astar`, the default, or `dijkstra`). Basic pedestrian-access, physical-barrier and direction hard rules still apply; the selected wheelchair hard requirement does not.

Both traces are then assessed under the **same selected profile and cost parameters**, including weights, event exposure and missing-data policy. If the standard route violates selected hard requirements, it has **no comparable profile-match score or optimized cost**, carries an infeasibility warning and is **not a recommended alternative**. No selected soft criteria means an undefined match, not 100% accessibility. Search costs with different objectives are not compared.

Reported changes are **selected minus standard**: distance in metres and percent, match in percentage points (pp), and grouped obstacle/conflict/missing-data count differences. Worsening and trade-offs remain visible. Differences can reflect **relaxing the wheelchair hard requirement as well as soft preferences**, not solely a preference-cost improvement. The separate matched-hard-constraint distance baseline and A*/Dijkstra checks remain available.

The terminal prints a short result with selected preferences and explicit **PREFERENCE CONFLICT** labels, not large raw node lists or hashes. Use `--json` only when you want the full machine-readable record.

The archive is the completion marker and is written **last**. A missing or corrupt archive means the new-format run did not complete; the program does not overwrite an earlier result. No pickle or executable object is stored in the archive. It is read without extracting files and validated for expected members, hashes and size limits (1 GiB/member, 2 GiB total uncompressed).

Detailed CSV/JSON remains available **inside the archive**, not scattered over the result directory. `--details` additionally expands the evidence into a `diagnostics/` subdirectory: `manifest.json`, `dataset.json`, `edge_diagnostics.json`, `all_edges.csv`, `route_segments.csv`, `criteria.csv`, `pois.csv`, `pois.json`, `route.geojson`, `presentation.json`, GPX and summary. This flag does not alter route selection, scores or figure contents.

### Choose the obstacles to show

By default, markers focus on **both routes**, not every off-route POI: stairs, important blocking barriers/restrictions, and known concerns associated with selected preferences. Neutral mapped ramps, lowered kerbs and controlled crossings do not all become markers automatically. A wheelchair restriction on the exact same stairway/barrier encounter shares its physical marker in automatic mode, retaining all selected-preference warnings. Restrictions affecting a different portion of the way keep separate markers. All original raw POIs remain archived.

Choose exact types when checking a specific issue:

```powershell
python -m osm_accessibility demo --output results/stairs-check --factor 6 --obstacles stairs barrier kerb --max-markers 8
python -m osm_accessibility plot results/stairs-check --output results/comparison-figures --obstacles stairs barrier --max-markers 6 --corridor-radius 100 --format pdf
```

Available types: `stairs`, `barrier`, `wheelchair_restriction`, `kerb`, `crossing`, `incline`, `ramp`, `surface`, `smoothness`, `width`, `lighting`, `bicycles`. `--obstacles auto` restores automatic selection; `all` includes all recorded candidate types; `none` hides marker types. The special values must be used alone.

The shared marker budget defaults to **12** (valid range 0–100), prioritizing conflicts and important obstacles on either route. Repeated geometry on the same way is grouped rather than putting a staircase marker on every edge and reverse direction. Candidates outside the corridor are not drawn; nearby same-type/same-owner markers can be coalesced. Captions disclose type-filter, cropping, coalescing and marker-limit counts. **Filtering and marker caps change display only:** complete encounter/conflict/missing-data records and counts for both routes, graph diagnostics, scores and selected-route GPX remain unchanged.

The area map scores **mapped directed segments within the displayed corridor**, not every possible complete origin/destination route. Its panels highlight the selected and standard paths. Changing `--max-markers` during replotting retains the previously selected obstacle types unless `--obstacles` is also supplied.

### Read the preference/obstacle report

The summary distinguishes four different facts:

- **Encountered:** an observed feature on the selected path, such as mapped stairs. It is not necessarily forbidden under the selected profile.
- **PREFERENCE CONFLICT:** a selected criterion has a known, nonzero modeled risk on a traversed direction—for example stairs when Avoid stairs was selected, a path narrower than the chosen minimum, or `lit=no` when lighting matters. These are soft-cost trade-offs, not automatically hard violations or unavoidable obstacles.
- **Missing information:** a selected attribute was not recorded or could not be parsed. This is kept separate from observed obstacles, including when the configured unknown-risk penalty is zero.
- **Hard-constraint violation:** a selected directed edge is actually blocked under the model. This is checked separately and should be zero for a route produced by the algorithm. A blocked reverse edge is not a violation of the chosen direction.

Repeated contiguous way observations are grouped into one encounter with route segment ranges and length. Node events are associated with arrival at that node. Nonconsecutive returns remain separate encounters. The selected weight/importance and observed OSM measurement are shown, so a conflict with an Essential preference is explicit. All selected preferences and hard/soft requirement settings are listed, including wheelchair requirement, ramp use, handrail preference and minimum width.

Comparison counts are **grouped encounters by type**, not stair risers, unique physical objects or counts invariant to OSM way segmentation. A physical feature can appear in multiple type rows; nonconsecutive revisits count again. Both routes' full on-route obstacles, selected soft conflicts and missing information are retained regardless of marker limits. Unknown data are not observed obstacles, and fewer encounters of one type can coexist with more of another.

### How to read the figures

- Figures use a local WGS84 azimuthal-equidistant projection with metre axes and equal aspect. The coordinate conversion always passes longitude, latitude to pyproj. Graph distances used in optimization are spherical Haversine distances; plotted projection distances are for local inspection, not a replacement for those costs.
- `--corridor-radius METRES` defaults to **150**, must be finite and satisfy **0 < radius <= 5000**, and controls both maps. Background edges are geometrically **clipped to the union of buffers around both route polylines in metric AEQD coordinates**, not merely selected by a bounding box. This is display-only: routing, normalized inputs and full archived diagnostics are not cropped. Both score panels use the same extent and color scale.
- Long horizontal corridors use two stacked full-width score panels; other shapes use side-by-side panels. Bounds follow the corridor with small independent padding rather than expanding its narrow dimension into an empty square. Equal metric aspect is preserved, without stretching or rotating either route.
- OSM figures carry attribution; synthetic figures explicitly state that they are artificial. No map tiles, geocoder, online basemap or map API key is needed.
- Selected route geometry is not simplified or shifted. Every stored point appears in GPX. **Snapping gaps are not routes**, are not added to GPX, and are recorded separately in the manifest.
- Hard-blocked directions are dashed red on the area score map, **not assigned an accessibility score of zero**. No scored exposure/active criteria is gray. Unknown attributes normally have a modeled risk and therefore a numeric score; they are not the same as no scored exposure. Scored segments use the same fixed **0–100 `cividis` scale**, with a colorbar; the range is not rescaled to make a route look better.
- Opposite and parallel directions may have different event costs. Only the background score layer offsets coincident segments by at most 1 metre to expose both directions, then **re-clips them to the same corridor**. Corridor membership uses original geometry; these color-layer offsets do not change either route trace, graph or GPX.
- The route-focus map uses a blue selected route and dashed orange standard route, with white halos above pale context edges and hollow obstacle symbols. The score map places the paths in separate panels and marks standard-route directions blocked under the selected profile in dashed red. Arrows, line styles and marker labels (`S`, `N`, `S+N`) supplement color; coincident paths are identified explicitly.
- All route conflicts remain in the summary even when marker types or their count are limited. Numbered markers point to a concise map key, not long raw tags over the path. Off-route features are **not claimed to be successfully avoided merely because they are outside a route**.
- A 100% score can occur on fully tagged ideal synthetic segments. It does not establish that the model, tags or real paths are accurate. An absent stair/kerb marker is not proof that the obstacle is absent.

Recreate figures offline from saved traces:

```powershell
python -m osm_accessibility plot results/demo --output results/demo-figures
python -m osm_accessibility plot results/demo --output results/demo-wide --corridor-radius 250
```

Replotting writes the two maps and an updated readable summary. An archive can also be supplied directly instead of its containing folder. Saved marker settings and corridor radius are reused unless overridden; PNG remains the CLI default unless `--format` is supplied. Format-3 runs use **stored selected and standard traces**, not fresh searches. Formats 1/2 remain loadable subject to provenance checks, but have no stored standard comparison: the replot labels the standard route as not supplied, without rerouting or fabricating one. Display selection never rewrites the original archive.

### Figure export and model checks

The PNGs use 240 DPI; `--format pdf` produces vector maps with embedded fonts. The selected preferences, route metrics, remaining conflicts and model checks give readers enough context to interpret the decision. The maps demonstrate the route choice and trade-offs **in the supplied graph**, not that all physical obstacles are known.

The numerical panel/report states A*/Dijkstra cost agreement only when those comparisons were actually run, along with the absolute difference, hard-block violations and distance overhead against the matched-constraint distance baseline. These are useful case-specific checks of computational behavior; they are not proof of overall algorithm correctness, field accessibility, or safety. A high model-generated score cannot independently validate the model that generated it.

## Python modules

```text
osm_accessibility/
	cli.py, __main__.py       Command-line commands
	models.py, errors.py      Validated records and scientific domain errors
	tags.py, profiles.py      OSM units, preference weights and cost parameters
	data.py                  Complete OSM XML/JSON ingestion and content fingerprints
	mongo.py                 Explicit upload, indexed snapshots and read-only loading
	builder.py, graph.py      Deterministic directed topology and nearest-node lookup
	geo.py                   Haversine distance
	evaluation.py            Criterion-level explanations, hard blocks and edge scores
	routing.py               A* / Dijkstra with traceable snapping
	outputs.py               GPX, readable preferences/conflicts and optional raw diagnostics
	experiment.py            Compact archives, provenance checks, baselines and sweeps
	visualization.py         Route/obstacle and area-accessibility publication maps
	demo.py                  Artificial test network (not empirical data)
examples/run_research.py    Editable Python experiment/profile example
tests/                     Algorithm, ingestion, snapshot, archive and figure tests
```

## Getting complete OSM data

Supported input formats are **OSM XML** (`.osm`/`.xml`) and **Overpass JSON** (`.json` with an `elements` array). GeoJSON, GPX, `.osm.pbf`, OSM change files and arrays without topology are not accepted. Do not rename those files to make the importer accept them.

Use a small, complete city/region extract. Select an area larger than the origin/destination corridor so legitimate detours are not excluded. The program loads the resulting normalized graph into memory; it is **not a planet-scale streaming router**. Parsing is incremental, but normalized records, graph and full diagnostics remain resident.

For example, print a complete-node Overpass query for Erlangen:

```powershell
python -m osm_accessibility query --bbox 49.57 10.995 49.61 11.025
```

The output is:

```text
[out:json][timeout:180];
(way["highway"](49.57,10.995,49.61,11.025); relation["highway"="pedestrian"](49.57,10.995,49.61,11.025););
(._; >>;);
out body;
```

Execute it using an Overpass-compatible tool and save the returned **raw JSON data**, not its GeoJSON conversion, as `data/erlangen.json`. The query-printing command itself makes no network request. Respect public Overpass usage limits; do not run repeated bulk extractions for each route. Use a consistent archived extract for experiments.

The recursive step preserves referenced members and nodes. `out body` retains node tags, unlike tag-free skeleton output. If using `out meta`, this importer keeps element timestamps/versions but not contributor usernames/UIDs. Overpass error `remark` fields, invalid coordinates, conflicting duplicates and missing referenced nodes/members cause validation failure. Do not use clipped geometry or `out center` as a substitute for complete way topology. For a dated study use an archived extract or a supported Overpass date query; record its source, extraction date and spatial boundary.

Route directly from a local extract:

```powershell
python -m osm_accessibility route --osm data/erlangen.json --start 49.585 11.005 --end 49.595 11.005 --profile wheelchair --factor 6 --output results/erlangen-file
```

These coordinates are **examples, not verified successful routes**. The selected route may be unavailable under its requirements or data coverage; its hard constraints are never silently relaxed. The separately labelled standard comparison deliberately omits the wheelchair hard requirement as described above.

Once loaded, routing, scoring, plots and GPX generation operate entirely on the in-memory graph. There are no database writes in `route`, `plot`, `sweep` or `demo`.

## Preference model and algorithm

### Graph construction

The graph is a directed multigraph: vertices are OSM nodes; an edge connects consecutive references on a mapped way and retains the exact way ID, endpoint tags and Haversine length. Parallel edges remain distinct. Edges and node IDs are deterministically ordered for tie-breaking. Only a **shared node ID** creates an intersection; crossing drawn lines do not imply connectivity across bridges or different levels.

The builder retains restricted highways for diagnostics, but routing omits edges prohibited by the profile/policy. It honors `oneway:foot`, `foot:forward` and `foot:backward`; vehicle `oneway` is not automatically applied to pedestrians. Reverse edges reverse signed incline tags. The current policy penalizes absolute steepness equally uphill and downhill. Selected pedestrian relations can provide missing outer-way tags, with explicit way tags taking precedence. No shortcut across a pedestrian polygon, building or hole is invented. Missing geometry references and conflicting inherited tags fail explicitly.

Origin/destination snapping is nearest **eligible connected node**, not edge projection. Starts require a usable outgoing edge and destinations an incoming edge. Default maximum snap distance is 150 m, changeable with `--snap-distance`. If both coordinates map to the same node or no permitted path exists, the run fails rather than reporting a perfect empty route. A snap gap is a localization distance, **not an error estimate or proof of traversability**.

### Explicit hard constraints versus soft preferences

`Profile.wheelchair_required` is a separate boolean; it is **not silently inferred from a score weight**. Known pedestrian-access restrictions, excluded major highways and physical barriers remain hard exclusions. Specific allowed `foot` tags can override generic access restrictions. With `wheelchair_required=True`, explicit `wheelchair=no`, unverified restrictive barriers and steps without a usable wheelchair ramp are excluded. `ramp:wheelchair=yes` or `ramp=yes` plus `wheelchair=yes` can identify a usable ramp; `ramp:wheelchair=no` overrides. Bicycle-only or separately mapped ramps are not assumed to make the stair edge traversable.

The ten soft criteria are:

| Criterion | Risk rule (before weighting) |
| --- | --- |
| `avoid_bicycles` | Explicit shared cycling without segregation: 1; prohibited/dismount/segregated: 0; otherwise unknown |
| `lit_roads` | `yes`: 0; `no`: 1; otherwise unknown |
| `surface_type` | Recognized firm paving: 0; recognized loose/rough surfaces: 1; poor track grades can impose a floor |
| `smoothness` | excellent/good: 0; intermediate: 0.5; bad or worse: 1 |
| `road_width` | Recognized width below chosen minimum: 1; otherwise 0 |
| `road_incline` | `min(abs(grade)/incline_full_risk_percent, 1)`; flat is 0; qualitative up/down magnitude is unknown |
| `avoid_stairs` | No steps: 0; usable ramp with steps: 0.25; other steps: 1; missing requested handrail adds 0.25 up to 1 |
| `wheelchair_accessible` | yes: 0; limited: 0.5; no: 1; otherwise unknown (hard requirement evaluated separately) |
| `short_kerbs` | Recognized flush/lowered: 0; raised/rolled/stepped: 1; numeric height normalized by selected kerb scale |
| `supervised_crossings` | Signals: 0; marked/zebra/uncontrolled: 0.35; unmarked/unsupervised: 1; otherwise unknown |

Unrecognized/missing observations use `unknown_risk` (default 0.5), **not a claimed probability**. Known-data coverage is reported separately. Weights are numeric in `[0,1]`, or the five labels mapped to 0, .25, .5, .75, 1: Not Important, A Little Important, Important, Very Important, Essential. Zero-weight criteria are excluded from normalization.

Use built-in profiles `wheelchair`, `walking` or `distance`, or edit [examples/run_research.py](examples/run_research.py) and use the Python classes. An optional JSON profile supplied via `--profile-file` has this shape:

```json
{"name":"custom","wheelchair_required":true,"allow_ramps":true,"prefer_handrail":true,"min_width_m":1.1,"weights":{"surface_type":1,"smoothness":0.75,"short_kerbs":1,"avoid_stairs":1,"road_width":0.5,"road_incline":0.75}}
```

Missing weights are zero. Unknown names and non-finite/out-of-range parameters fail validation. Complete normalized profiles and parameters are archived. The `distance` built-in has no soft criteria and no wheelchair requirement: do **not** treat it as enforcing the same constraints. Likewise, the new **standard comparison** relaxes the wheelchair requirement. The separate **matched-constraint distance baseline** retains the selected profile's hard requirements and measures distance overhead without that confound.

### Cost and A* correctness

For continuous way risk $R_e$, event risk $T_e$, edge length $d_e$, trade-off factor $\lambda$ and event exposure $q$:

$$ C(e)=d_e+\lambda\left(d_e R_e+qT_e\right). $$

Each risk is the weighted average over its active, applicable criteria. Crossings/kerbs represented as nodes are charged once on **arrival**, not multiplied by the approach segment length. Where a way already contains node events, corresponding way-level event tags are not charged again. For way-only crossing data without node events, those criteria are treated as continuous way attributes. This prevents geometry subdivision from diluting a mapped node event or double-counting its tags.

Default $\lambda=1.5$, $q=20$ equivalent metres; `--factor 6` places more weight on accessibility. Defaults are hypotheses to calibrate, **not empirical findings**. `CostParameters` also exposes unknown risk, 12% full-risk incline and 0.1 m full-risk kerb scales.

All edge lengths are finite, nonnegative and no shorter than endpoint Haversine distance (1 µm numerical tolerance). All penalties are nonnegative. Therefore the straight-line Haversine heuristic is admissible/consistent to numerical precision for the constructed graph. A* and Dijkstra minimize the **same scalar cost**. This is not a multi-objective Pareto search and not a novel A* algorithm. Stored predecessor edges preserve geometry and way attribution. Disconnected/forbidden routes return `NoRouteFound`; no constraints are secretly relaxed.

### Scores, exposure and missingness

For an edge, scored exposure $E_e$ is its length **only if continuous criteria are active**, plus $q$ if an event criterion is active. The event-inclusive match is:

$$ S_e=100\left(1-\frac{d_eR_e+qT_e}{E_e}\right). $$

If no scored exposure is present, the score is **undefined (`null`)**, not 100%. Blocked edges also have `null` scores and explicit reasons. `way_score` excludes the node-event component. The route score pools numerators/exposures across edges; it is not the unweighted average of segment scores. Unselected criteria/segments cannot falsely improve it by adding zero-risk denominator length. Coverage likewise aggregates recognized selected-criterion weights within each scope/exposure.

Route cost is invariant to equivalent subdivision of an approach edge with the same tags and node event. The **local event-inclusive segment score need not be**: subdividing that segment concentrates the fixed event exposure in the incoming piece. Inspect `way_score`, `transition_risk`, `event_risk_metres`, `criteria.csv` and pooled route score together. Compare scores only for the same profile/parameters; changes in weighting or missingness policy change the measuring rule.

## Reproducibility and evaluation

Each run's archive records normalized input fingerprint, graph fingerprint, full implementation-source fingerprint, a separate scientific-core fingerprint, model/package/dependency versions, all installed Python distribution versions, Git revision when available and whether source changes were uncommitted. Dataset metadata includes the input file checksum and Overpass base timestamp when present. Direct/MongoDB loading of the same normalized snapshot yields the same graph hash and deterministic path under the same parameters. Runtime and creation timestamps are expected to differ. Run in an isolated environment and retain that recorded package list when reproducing results.

Saved-run validation checks mandatory artifact hashes and recomputes route length, score, costs and exposure from the recorded graph/profile. Hashes detect accidental changes and aid reproduction; the manifest is not digitally signed and is not proof of authenticity against someone deliberately rewriting the entire experiment archive.

New archives use **format 3**. The manifest stores the standard route's node/edge trace, search algorithm, no-preference profile and distance-only objective; presentation data stores the full selected-profile comparison. This adds **no extra visible file or second GPX**. Format-2 archives and recognized format-1 full-directory runs remain loadable with the existing provenance/scientific-core checks, without inventing a missing standard comparison. Unknown older implementations or a changed scientific core still fail with an instruction to use the original revision. The original run is never rewritten during replotting. Archive files are checksummed internally; loading a whole result directory also checks the visible GPX, summary and figures. A standalone archive does not require its external figures to travel with it.

In addition to the always-computed standard route, optional baseline checks compare A*, Dijkstra with the same accessibility objective, and distance-only Dijkstra with the **same selected hard requirements and snapped endpoints**. Only these checks are skipped by `--no-baselines`. A*/Dijkstra optimal-cost disagreement raises an error. Tied optimal paths can differ in independent implementations; cost equality matters more than one arbitrary node sequence. Search time excludes loading, graph construction, snapping, exports and plots; fresh cost caches are used for each timed search. It is not an end-to-end latency measurement.

Run a parameter sensitivity experiment on archived inputs:

```powershell
python -m osm_accessibility sweep results/demo --output results/sensitivity --factors 0 1.5 6 --unknown-risks 0.25 0.5 0.75 --repeats 3
```

This writes CSV/JSON observations for factor, unknown risk, algorithm, distance, cost, score, coverage, nodes expanded and search time. It retains the original snapped endpoints and profile constraints. Repeated timings on a tiny example are not evidence of large-scale performance. Full-graph diagnostics are retained even though the maps show only the corridor; diagnostics, presentation and exports must not be included in claims about A* query time.

Version 0.3.0 skips context-only obstacle candidate generation for paired-route presentations, uses indexed **exact nearest-segment** distances instead of the former candidate-by-route-segment (`P × R`) scan, and writes ZIP evidence with fast DEFLATE compression level **1**. These changes retain the same raw evidence and complete on-route reports. No real-data 2 km end-to-end runtime or measured speedup is claimed here; those require representative benchmarks.

### Tests

```powershell
python -m pip install ".[test]"
python -m pytest --cov=osm_accessibility
python -m ruff check .
```

The suite uses artificial graphs, OSM fixtures generated in Python and a mock MongoDB database. It checks cost decomposition, subdivision invariance, constraints, missingness, exact GPX order, import safety/fingerprints, route optimality, saved-run validation and the actual plotting artifacts. It does **not** certify a live MongoDB deployment, geospatial index behavior on a real server, true wheelchair accessibility or an entire city dataset.

The original scientific baseline is extended with comparison, orchestration and presentation checks: same-endpoint standard routing, selected-profile assessment and infeasibility, complete counts despite marker caps, corridor clipping and indexed distances, exact foreground traces, archive integrity and old-run replot compatibility. Display-setting checks cover unchanged selected-route GPX geometry and route costs. No empirical accessibility results or live database tests are implied by these checks.

**Version 0.3 verification:** Python 3.12.10 recorded **343 passed, 1 skipped**, approximately **93% branch-aware coverage**; lint and dependency consistency passed. The skip requires unavailable Windows symbolic-link permissions. PNG/PDF layouts were inspected for the small example and a synthetic ~2 km route.

One synthetic scalability check used **6,561 nodes and 25,920 directed edges**, identical inputs/interpreter and PNG-only output. The selected ~2.1 km trace, distance, cost and score were unchanged: version 0.2 took **12.19 s**, and the comparison implementation took **10.55 s** including the extra standard search (before final framing polish). Separately, exact distances for 4,000 candidates against 800 segments of a 2 km path took **1.447 s** by scanning versus **0.017 s** with the spatial index (~84× for that component, **not** the whole workflow). These are single local synthetic measurements, not a promise about a particular real extract, machine, or field accessibility.

## Data attribution and references

OpenStreetMap data is licensed under **ODbL 1.0**, separately from program code. Retain attribution in plotted maps and comply with derivative-database obligations when sharing extracts/derived databases. Do not commit credentials, private participant coordinates or unlicensed map imagery. Generated OSM plots include credit; synthetic plots are labelled instead.

- OpenStreetMap contributors, [copyright and license](https://www.openstreetmap.org/copyright).
- [Overpass QL documentation](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL): complete topology and recursive extraction.
- [MongoDB geospatial queries](https://www.mongodb.com/docs/manual/geospatial-queries/): GeoJSON order and indexes.
- Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs*. Numerische Mathematik 1, 269–271. DOI: [10.1007/BF01386390](https://doi.org/10.1007/BF01386390).
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics 4(2), 100–107. DOI: [10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136).

These foundational references are not a substitute for a systematic review of accessibility-routing literature.
