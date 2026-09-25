"""Edit a Python profile and produce a complete reproducible artificial example."""

from osm_accessibility.demo import synthetic_dataset
from osm_accessibility.experiment import run_experiment
from osm_accessibility.models import Coordinate
from osm_accessibility.profiles import CostParameters, Profile

if __name__ == "__main__":
    profile = Profile(name="custom-wheelchair", wheelchair_required=True, min_width_m=1.1, weights={
        "surface_type": 1, "smoothness": .75, "short_kerbs": 1, "supervised_crossings": .5,
        "avoid_stairs": 1, "road_incline": 1, "road_width": .75, "wheelchair_accessible": 1,
    })
    run_experiment(synthetic_dataset(), Coordinate(49.58, 11), Coordinate(49.58, 11.002),
                   profile, CostParameters(accessibility_factor=6), "results/custom-example")