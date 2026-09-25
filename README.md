# Accessibility-aware routing on OpenStreetMap graphs

A Python research toolkit for reproducible, preference-dependent pedestrian pathfinding using OpenStreetMap (OSM) data. It provides a directed graph model, configurable accessibility costs, A* and Dijkstra search, safe MongoDB snapshot ingestion, GPX export, and static diagnostic figures.

**Research status:** the model implements explicit hypotheses about accessibility-related tags. Its scores are heuristic preference matches, **not measured accessibility, probabilities, or safety guarantees**. Synthetic examples demonstrate computations only. A manuscript should report real-data and independent ground-truth evaluation before claiming improved accessibility.

All executable code and examples are Python. Besides this README, only minimal Python dependency/packaging metadata and ignore rules are included. Generated data, figures, credentials and run outputs are not committed.

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

Every output directory must be new. Re-running against the same directory fails rather than silently replacing an experiment. Use `--no-plots` to run only computation/export, `--format png` or `--format pdf` for a single figure format, and `--no-baselines` to skip baseline searches.

## Outputs and graphical checks

Each successful run produces:

| Artifact | Contents |
| --- | --- |
| `route.gpx` | GPX 1.1 containing every selected graph node, in route order |
| `route.geojson` | Selected directed segments, geographic geometry and evaluation properties |
| `route.png`, `route.pdf` | Exact selected path with origin, destination and direction arrows |
| `graph_pois.png`, `graph_pois.pdf` | Whole extracted graph, highlighted route, blocked directions and relevant obstacle/feature POIs |
| `accessibility.png`, `accessibility.pdf` | Two panels: all directed graph segments colored by score, and selected route segments colored by the same score |
| `route_segments.csv` | Ordered edge IDs, OSM way IDs, length, cost, risk, score, coverage and hard-block reasons |
| `all_edges.csv` | The same diagnostics for all graph edges, including unselected and blocked alternatives |
| `criteria.csv` | Per-edge, per-criterion weight/risk/known-data/scope decomposition |
| `edge_diagnostics.json` | Full precision diagnostic values, reasons and component records |
| `pois.csv`, `pois.json` | Stairs, ramps, barriers, kerbs, crossings, inclines and explicit wheelchair restrictions, with on-route flags |
| `dataset.json` | Normalized input snapshot needed for offline reproduction |
| `manifest.json` | Preferences, parameters, input/graph/code/artifact SHA-256 hashes, versions, snap records, score decomposition and baselines |

The final manifest is written **last**. Its absence means the run did not complete. Partial output is retained for inspection; rerun into a fresh directory after resolving errors.

### How to read the figures

- Figures use a local WGS84 azimuthal-equidistant projection with metre axes and equal aspect. The coordinate conversion always passes longitude, latitude to pyproj. Graph distances used in optimization are spherical Haversine distances; plotted projection distances are for local inspection, not a replacement for those costs.
- OSM figures carry attribution; synthetic figures explicitly state that they are artificial. No map tiles, geocoder, online basemap or map API key is needed.
- Selected route geometry is not simplified or shifted. Every stored point appears in GPX. **Snapping gaps are not routes**, are not added to GPX, and are recorded separately in the manifest.
- Hard-blocked directions are dashed red, **not assigned an accessibility score of zero**. No scored exposure/active criteria is gray. Scored segments use the same fixed **0–100 `cividis` scale**, with a colorbar; the range is not rescaled to make a route look better.
- Opposite and parallel directions may have different event costs. Only the all-edge score panel offsets coincident segments by at most 1 metre to expose both directions. The caption identifies this diagnostic displacement; it does not change the graph, route or GPX.
- All detected POIs are plotted, including those **not on the route**. Hollow/filled symbols distinguish off-route/on-route features. At most 25 POIs are annotated to limit clutter; the tables retain all IDs/descriptions. A POI is not necessarily forbidden: a ramp or signalized crossing can be desirable. A node's blocked flag concerns its incident graph edges, not independent field verification of that physical object.
- A 100% score can occur on fully tagged ideal synthetic segments. It does not establish that the model, tags or real paths are accurate. An absent stair/kerb marker is not proof that the obstacle is absent.

Recreate figures without reconnecting to MongoDB:

```powershell
python -m osm_accessibility plot results/demo --output results/demo-figures
```

Saved inputs, graph/model/code fingerprint and artifact hashes are checked before replotting. Use the original implementation revision when reproducing a saved run.

POI annotations are short numeric labels linked to an adjacent ID lookup panel, rather than long tag strings covering the route. The full raw tags remain in `pois.csv`/`pois.json`. These figures support fast inspection; overlapping features in a dense city still need examination of the tabular diagnostics.

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
	outputs.py               GPX, GeoJSON, CSV, score/POI diagnostics
	experiment.py            Saved runs, provenance checks, baselines and sensitivity sweeps
	visualization.py         Static route, graph/POI and accessibility figures
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

Route directly from a local extract before uploading it:

```powershell
python -m osm_accessibility route --osm data/erlangen.json --start 49.585 11.005 --end 49.595 11.005 --profile wheelchair --factor 6 --output results/erlangen-file
```

These coordinates are **examples, not verified successful routes**. The route may be unavailable under the selected requirements or data coverage. No hidden relaxation of hard constraints occurs.

## MongoDB connection and upload

### 1. Prepare a dedicated research database

Use a local MongoDB instance or MongoDB Atlas. Prefer a dedicated database such as `OSMResearch`, separate from other operational data. This toolkit stores named snapshots with `dataset_id`; it does not assume that unrelated existing collections follow its snapshot conventions.

- For import, use an account allowed to insert documents and create indexes in the dedicated research database.
- For routing, use a read-only account once import is complete.
- In Atlas allow only the required client network addresses. Keep TLS verification enabled and never put passwords in a repository or paper artifact.
- If an existing collection has incompatible indexes or unmanaged data, choose a new dedicated research database. The importer never drops indexes, collections or databases to repair it.

Only **`MONGODB_URI`** is a credential setting. Database and snapshot names are separate command arguments. Local unauthenticated MongoDB example (for a trusted local instance only):

```powershell
$env:MONGODB_URI = "mongodb://localhost:27017/"
```

For Atlas, set the URI through your environment/secret mechanism:

```text
mongodb+srv://<username>:<password>@<cluster-host>/?retryWrites=true&w=majority
```

Percent-encode reserved characters in credentials. Use a secret manager or enter the value privately; do not paste it into this README, a profile file or command argument. The program accepts no URI command-line option and does not save the URI to outputs. Database-driver errors are sanitized. Environment files are not loaded automatically.

### 2. Validate first: no credentials or database writes required

```powershell
python -m osm_accessibility import-osm data/erlangen.json --dataset erlangen_2026_09
```

Without `--write`, the importer performs a **dry run**: normalizes data, checks references, constructs the graph and prints counts/fingerprint. It does not connect to MongoDB.

### 3. Upload explicitly

```powershell
python -m osm_accessibility import-osm data/erlangen.json --database OSMResearch --dataset erlangen_2026_09 --write
```

This creates these collections and indexes when needed:

| Collection | Contents / indexes |
| --- | --- |
| `Nodes` | WGS84 GeoJSON `Point` location, tags, OSM ID, dataset ID; unique `(dataset_id,id)` and `2dsphere(location)` |
| `Ways` | Ordered node IDs and tags; unique `(dataset_id,id)` and `(dataset_id,nodes)` |
| `Relations` | Member IDs/types/roles and tags; unique `(dataset_id,id)` and `(dataset_id,members.ref)` |
| `Datasets` | Snapshot name, normalized content hash, source metadata, counts and import status |

Example normalized node:

```json
{"dataset_id":"erlangen_2026_09","type":"node","id":"123","location":{"type":"Point","coordinates":[11.005,49.585]},"tags":{"highway":"crossing","kerb":"lowered"}}
```

**Coordinate order:** CLI/Python coordinates are latitude, longitude. MongoDB GeoJSON and GeoJSON exports are **longitude, latitude**. IDs are normalized to strings consistently.

Imports are batched, but they are **not one multi-collection transaction**. A unique manifest is first marked `loading`; only a fully written, count-verified snapshot becomes `ready`. A failed import retains partial records and is marked `failed` when possible. Readers refuse non-ready snapshots. Inspect failed records administratively, fix the cause and use a **new snapshot name**. Reusing any existing snapshot ID fails instead of replacing it. There is no `--force`, truncate or implicit overwrite behavior. Concurrent imports to one snapshot name are prevented by its unique manifest key.

Routing rereads the complete named snapshot and verifies its normalized content hash and counts, so subsequent changes are detected. For a paper, record the snapshot ID and hash; archive the source extract separately under its data license. MongoDB is a storage layer here, not a per-route mutable source of truth.

### 4. Run from MongoDB

```powershell
python -m osm_accessibility route --database OSMResearch --dataset erlangen_2026_09 --start 49.585 11.005 --end 49.595 11.005 --profile wheelchair --factor 6 --output results/erlangen-mongo
```

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

Missing weights are zero. Unknown names and non-finite/out-of-range parameters fail validation. Complete normalized profiles and parameters are archived. The `distance` built-in has no soft criteria and no wheelchair requirement: do **not** compare it as if it enforced the same constraints. The automatic **distance-only baseline uses the selected profile's hard constraints**, avoiding that confound.

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

Each run records normalized input fingerprint, graph fingerprint, implementation-source fingerprint, model/package/dependency versions, all installed Python distribution versions, Git revision when available and whether source changes were uncommitted. Dataset metadata includes the input file checksum and Overpass base timestamp when present. Direct/MongoDB loading of the same normalized snapshot yields the same graph hash and deterministic path under the same parameters. Runtime and creation timestamps are expected to differ. Run in an isolated environment and retain that recorded package list when reproducing results.

Saved-run validation checks mandatory artifact hashes and recomputes route length, score, costs and exposure from the recorded graph/profile. Hashes detect accidental changes and aid reproduction; the manifest is not digitally signed and is not proof of authenticity against someone deliberately rewriting the entire experiment archive.

Baseline output compares A*, Dijkstra with the same accessibility objective, and distance-only Dijkstra with the **same hard requirements and snapped endpoints**. A*/Dijkstra optimal-cost disagreement raises an error. Tied optimal paths can differ in independent implementations; cost equality matters more than one arbitrary node sequence. Search time excludes loading, graph construction, snapping, exports and plots; fresh cost caches are used for each timed search. It is not an end-to-end latency measurement.

Run a parameter sensitivity experiment on archived inputs:

```powershell
python -m osm_accessibility sweep results/demo --output results/sensitivity --factors 0 1.5 6 --unknown-risks 0.25 0.5 0.75 --repeats 3
```

This writes CSV/JSON observations for factor, unknown risk, algorithm, distance, cost, score, coverage, nodes expanded and search time. It retains the original snapped endpoints and profile constraints. Repeated timings on a tiny example are not evidence of large-scale performance. Plot computation deliberately evaluates the whole graph and must not be included in claims about A* query time.

### Tests

```powershell
python -m pip install ".[test]"
python -m pytest --cov=osm_accessibility
python -m ruff check .
```

The suite uses artificial graphs, OSM fixtures generated in Python and a mock MongoDB database. It checks cost decomposition, subdivision invariance, constraints, missingness, exact GPX order, import safety/fingerprints, route optimality, saved-run validation and the actual plotting artifacts. It does **not** certify a live MongoDB deployment, geospatial index behavior on a real server, true wheelchair accessibility or an entire city dataset.

Local verification (Python 3.12.10): **115 tests passed**, with approximately **89% branch-aware coverage**; lint, dependency consistency and source compilation also passed. The complete synthetic command-line workflow generated the three PNG/PDF views, GPX, tabular diagnostics, offline replotting and 81 parameter-sweep observations. Figures were visually inspected for route geometry, obstacle visibility, direction arrows, score legends and readable POI labels. No empirical accessibility results or live database tests are implied by these checks.


## Data attribution and references

OpenStreetMap data is licensed under **ODbL 1.0**, separately from program code. Retain attribution in plotted maps and comply with derivative-database obligations when sharing extracts/derived databases. Do not commit credentials, private participant coordinates or unlicensed map imagery. Generated OSM plots include credit; synthetic plots are labelled instead.

- OpenStreetMap contributors, [copyright and license](https://www.openstreetmap.org/copyright).
- [Overpass QL documentation](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL): complete topology and recursive extraction.
- [MongoDB geospatial queries](https://www.mongodb.com/docs/manual/geospatial-queries/): GeoJSON order and indexes.
- Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs*. Numerische Mathematik 1, 269–271. DOI: [10.1007/BF01386390](https://doi.org/10.1007/BF01386390).
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics 4(2), 100–107. DOI: [10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136).

These foundational references are not a substitute for a systematic review of accessibility-routing literature.
