"""Hard gates must be hard. Every one of these is a correctness guarantee."""

from __future__ import annotations

import pandas as pd
import pytest

from chainguard.config import SCENARIOS, CapacityPolicy
from chainguard.feasibility import build_candidates, hub_headroom


def test_headroom_is_linear_in_the_utilisation_ceiling():
    """Regression: an earlier model applied MaxUtilizationPct twice.

    headroom = C x (u_max - r - u_cur). With C=10000, u_max=0.9, u_cur=0.5 and
    no disruption that is exactly 4000. The buggy form (C x u_max x u_max - used)
    gives 3100 and silently rejects feasible routes.
    """
    hubs = pd.DataFrame(
        {
            "HubID": ["H1"],
            "WeeklyCapacityUnits": [10_000.0],
            "MaxUtilizationPct": [0.9],
            "CurrentUtilizationPct": [0.5],
            "CapacityReductionPct": [0.0],
            "DisruptionScenario": ["None"],
        }
    )
    assert hub_headroom(hubs, SCENARIOS["baseline"])["H1"] == pytest.approx(4000.0)


def test_headroom_applies_disruption_reduction_only_to_disrupted_hubs():
    hubs = pd.DataFrame(
        {
            "HubID": ["DOWN", "FINE"],
            "WeeklyCapacityUnits": [10_000.0, 10_000.0],
            "MaxUtilizationPct": [0.9, 0.9],
            "CurrentUtilizationPct": [0.5, 0.5],
            "CapacityReductionPct": [0.2, 0.2],
            "DisruptionScenario": ["Port congestion", "None"],
        }
    )
    head = hub_headroom(hubs, SCENARIOS["port_congestion"])
    assert head["DOWN"] == pytest.approx(2000.0)  # 10000 * (0.9 - 0.2 - 0.5)
    assert head["FINE"] == pytest.approx(4000.0)  # untagged hub keeps full headroom


def test_headroom_never_goes_negative():
    hubs = pd.DataFrame(
        {
            "HubID": ["OVER"],
            "WeeklyCapacityUnits": [10_000.0],
            "MaxUtilizationPct": [0.9],
            "CurrentUtilizationPct": [0.99],
            "CapacityReductionPct": [0.0],
            "DisruptionScenario": ["None"],
        }
    )
    assert hub_headroom(hubs, SCENARIOS["baseline"])["OVER"] == 0.0


def test_safety_buffer_reduces_headroom():
    hubs = pd.DataFrame(
        {
            "HubID": ["H1"],
            "WeeklyCapacityUnits": [10_000.0],
            "MaxUtilizationPct": [0.9],
            "CurrentUtilizationPct": [0.5],
            "CapacityReductionPct": [0.0],
            "DisruptionScenario": ["None"],
        }
    )
    conservative = hub_headroom(hubs, SCENARIOS["baseline"], CapacityPolicy(safety_buffer_pct=0.1))
    assert conservative["H1"] == pytest.approx(3000.0)


# ---------------------------------------------------------------------------
# Gate behaviour on the real candidate pipeline
# ---------------------------------------------------------------------------


def test_cold_chain_scenario_only_scopes_cold_chain_materials(dataset):
    cs = build_candidates(dataset, SCENARIOS["cold_chain"])
    assert (cs.shipments["TempRequirement"] == "Cold Chain").all()
    assert len(cs.shipments) < len(dataset.internal)


def test_cold_chain_candidates_only_use_cold_capable_hubs(dataset):
    """The single most safety-critical gate: no warm hub may ever appear."""
    cs = build_candidates(dataset, SCENARIOS["cold_chain"])
    cold_hubs = set(
        dataset.hubs.loc[
            dataset.hubs["ColdChainAvailable"].str.casefold() == "yes", "HubID"
        ]
    )
    if cs.n_candidates:
        assert set(cs.candidates["from_hub"]) <= cold_hubs
        assert set(cs.candidates["to_hub"]) <= cold_hubs


def test_hazard_gate_is_respected(dataset, scenario):
    cs = build_candidates(dataset, scenario)
    hubs = dataset.hubs.set_index("HubID")["SupportedHazardClasses"].fillna("")
    mats = dataset.materials.set_index("MaterialNo_Anon")["HazardClass"]
    ship_hazard = dataset.internal.set_index("ShipmentID")["MaterialNo_Anon"].map(mats)

    sample = cs.candidates.head(200)
    for _, row in sample.iterrows():
        hazard = ship_hazard.get(row["shipment_id"], "None")
        if hazard and hazard != "None":
            assert hazard in hubs.get(row["from_hub"], "")
            assert hazard in hubs.get(row["to_hub"], "")


def test_no_candidate_exceeds_hub_headroom(dataset, scenario):
    cs = build_candidates(dataset, scenario)
    qty = cs.candidates["qty"]
    assert (cs.candidates["from_hub"].map(cs.headroom).fillna(0) >= qty).all()
    assert (cs.candidates["to_hub"].map(cs.headroom).fillna(0) >= qty).all()


def test_no_candidate_exceeds_route_capacity(candidates):
    assert (candidates.candidates["route_capacity"] >= candidates.candidates["qty"]).all()


def test_unavailable_routes_are_excluded(dataset, scenario):
    cs = build_candidates(dataset, scenario)
    unavailable = set(
        dataset.routes.loc[
            dataset.routes["AvailableFlag"].str.casefold() != "yes", "RouteOptionID"
        ]
    )
    assert not (set(cs.candidates["route_id"]) & unavailable)


def test_primary_hub_down_excludes_every_primary_route(dataset):
    cs = build_candidates(dataset, SCENARIOS["primary_hub_down"])
    if cs.n_candidates:
        assert not cs.candidates["is_primary"].any()


def test_splitting_can_only_widen_the_candidate_set(dataset, scenario):
    """Relaxing an all-or-nothing capacity gate must never remove an option."""
    strict = build_candidates(dataset, scenario, max_splits=1)
    relaxed = build_candidates(dataset, scenario, max_splits=4)
    assert relaxed.n_candidates >= strict.n_candidates
    assert relaxed.coverage >= strict.coverage


def test_ledger_accounts_for_every_gate(candidates):
    from chainguard.feasibility import GATES

    assert set(candidates.ledger["gate"]) == set(GATES)
    assert (candidates.ledger["rejected_pairs"] >= 0).all()


def test_coverage_is_a_fraction(candidates):
    assert 0.0 <= candidates.coverage <= 1.0
    assert candidates.summary()["shipments_in_scope"] == candidates.n_shipments


def test_cost_per_kg_is_finite_and_positive(candidates):
    cpk = candidates.candidates["cost_per_kg"]
    assert (cpk > 0).all()
    assert cpk.notna().all()
    assert (cpk != float("inf")).all()
