"""Deterministic artificial network for demonstrations, never empirical map evidence."""

from .data import Dataset


def synthetic_dataset() -> Dataset:
    coordinates = {
        "s": (49.58, 11.0), "stairs_mid": (49.58, 11.001), "t": (49.58, 11.002),
        "smooth_a": (49.5815, 11.0), "smooth_b": (49.5815, 11.001), "smooth_c": (49.5815, 11.002),
        "rough_a": (49.579, 11.0005), "rough_b": (49.579, 11.0015),
        "gate": (49.5805, 11.001), "unknown": (49.5822, 11.001),
        "island_a": (49.5825, 11.0025), "island_b": (49.583, 11.003),
    }
    node_tags = {
        "smooth_b": {"highway": "crossing", "crossing": "traffic_signals", "kerb": "lowered"},
        "rough_b": {"highway": "crossing", "crossing": "unmarked", "kerb": "raised", "kerb:height": "0.12"},
        "gate": {"barrier": "stile", "wheelchair": "no"},
    }
    nodes = [{"type": "node", "id": name, "lat": lat, "lon": lon, "tags": node_tags.get(name, {})}
             for name, (lat, lon) in coordinates.items()]
    good = {"highway": "footway", "surface": "asphalt", "smoothness": "good", "width": "2 m",
            "incline": "0%", "lit": "yes", "wheelchair": "yes", "bicycle": "no"}
    definitions = [
        ("stairs", ["s", "stairs_mid", "t"], {"highway": "steps", "step_count": "12", "wheelchair": "no", "ramp": "no"}),
        ("smooth_first", ["s", "smooth_a", "smooth_b"], good),
        ("smooth_last", ["smooth_b", "smooth_c", "t"], good | {"width": "1.3", "lit": "no"}),
        ("rough", ["s", "rough_a", "rough_b", "t"], good | {"surface": "gravel", "smoothness": "bad", "incline": "10%", "width": "0.8"}),
        ("barrier_path", ["s", "gate", "t"], good),
        ("unknown", ["smooth_a", "unknown", "smooth_c"], {"highway": "path"}),
        ("ramped_stairs", ["smooth_b", "unknown"], good | {"highway": "steps", "ramp:wheelchair": "yes", "handrail": "yes", "step_count": "4"}),
        ("island", ["island_a", "island_b"], good),
    ]
    ways = [{"type": "way", "id": name, "nodes": refs, "tags": tags} for name, refs, tags in definitions]
    return Dataset.from_elements([*nodes, *ways], {"synthetic": True, "source": "deterministic artificial fixture v1",
                                                 "attribution": "Synthetic network — not observed OpenStreetMap data"})