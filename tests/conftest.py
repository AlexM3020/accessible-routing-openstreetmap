import pytest

from osm_accessibility.demo import synthetic_dataset
from osm_accessibility.evaluation import Evaluator
from osm_accessibility.profiles import CostParameters, wheelchair_profile


@pytest.fixture
def dataset():
    return synthetic_dataset()


@pytest.fixture
def graph(dataset):
    return dataset.graph()


@pytest.fixture
def evaluator():
    return Evaluator(wheelchair_profile(), CostParameters(accessibility_factor=6))