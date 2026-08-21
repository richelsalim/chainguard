"""Shared fixtures.

The whole suite runs against a *small* synthetic workbook generated at collection
time. That means the tests need no data files in the repository, run in seconds,
and exercise the real loader and the real Excel round-trip rather than a
hand-built DataFrame that quietly diverges from what production actually reads.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chainguard.config import SCENARIOS
from chainguard.feasibility import build_candidates
from chainguard.loader import load
from chainguard.synth import SynthConfig, write


@pytest.fixture(scope="session")
def small_config() -> SynthConfig:
    return SynthConfig(
        n_hubs=60,
        n_materials=40,
        n_families=12,
        n_internal=40,
        n_external=30,
        alternatives_per_lane=6,
        seed=7,
    )


@pytest.fixture(scope="session")
def workbook(tmp_path_factory, small_config):
    path = tmp_path_factory.mktemp("data") / "synthetic.xlsx"
    return write(path, small_config)


@pytest.fixture(scope="session")
def dataset(workbook):
    return load(workbook)


@pytest.fixture(scope="session")
def scenario():
    return SCENARIOS["baseline"]


@pytest.fixture(scope="session")
def candidates(dataset, scenario):
    return build_candidates(dataset, scenario)


@pytest.fixture
def toy_candidates() -> pd.DataFrame:
    """Two shipments with hand-checkable numbers, for exact scoring assertions."""
    return pd.DataFrame(
        {
            "shipment_id": ["S1", "S1", "S1", "S2", "S2"],
            "route_id": ["R1", "R2", "R3", "R4", "R5"],
            "lead_days": [2.0, 4.0, 6.0, 5.0, 5.0],
            "cost_per_kg": [10.0, 5.0, 1.0, 3.0, 3.0],
            "risk": [1.0, 2.0, 3.0, 4.0, 4.0],
            "co2_kg": [100.0, 150.0, 200.0, 120.0, 120.0],
            "qty": [100.0, 100.0, 100.0, 50.0, 50.0],
            "weight_kg": [10.0, 10.0, 10.0, 5.0, 5.0],
            "total_cost_eur": [100.0, 50.0, 10.0, 15.0, 15.0],
            "route_capacity": [1e6] * 5,
            "from_hub": ["A", "A", "A", "B", "B"],
            "to_hub": ["C", "D", "E", "F", "F"],
            "is_primary": [True, False, False, True, False],
            "mode": ["Air", "Road", "Ocean", "Air", "Air"],
            "material_family": ["F1"] * 5,
        }
    )
