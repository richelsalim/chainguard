"""Data contract, synthetic generator and network graph."""

from __future__ import annotations

import pandas as pd
import pytest

from chainguard.config import REQUIRED_SHEETS, SCENARIOS, STAGE_ORDER
from chainguard.loader import load
from chainguard.network import build_network, haversine_km
from chainguard.schema import CONTRACTS, SchemaError, validate_all
from chainguard.synth import SynthConfig, generate, write

# ---------------------------------------------------------------------------
# Loader / schema
# ---------------------------------------------------------------------------


def test_all_required_sheets_load(dataset):
    for sheet in REQUIRED_SHEETS:
        assert not dataset.frame(sheet).empty


def test_loader_rejects_a_workbook_missing_a_required_column(tmp_path, small_config):
    frames = generate(small_config)
    frames["Route_Options"] = frames["Route_Options"].drop(columns=["RiskScore"])
    path = tmp_path / "broken.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        for sheet, df in frames.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    with pytest.raises(SchemaError, match="RiskScore"):
        load(path)


def test_loader_gives_a_helpful_error_for_a_missing_file():
    with pytest.raises(FileNotFoundError, match="make synth"):
        load("data/definitely_not_here.xlsx")


def test_validation_reports_every_problem_at_once():
    problems = validate_all({}, strict=False)
    assert len(problems) == len(CONTRACTS)


def test_unnamed_excel_padding_columns_are_dropped(dataset):
    for sheet in REQUIRED_SHEETS:
        assert not any(str(c).startswith("Unnamed") for c in dataset.frame(sheet).columns)


def test_numeric_columns_are_actually_numeric(dataset):
    assert pd.api.types.is_numeric_dtype(dataset.routes["BaseCostEUR"])
    assert pd.api.types.is_numeric_dtype(dataset.hubs["WeeklyCapacityUnits"])
    assert pd.api.types.is_numeric_dtype(dataset.internal["Qty"])


def test_profile_reports_full_required_coverage(dataset):
    profile = dataset.profile()
    assert (profile["required_present"] == profile["required_total"]).all()


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def test_generator_is_deterministic_under_a_seed(small_config):
    a = generate(small_config)["Route_Options"]
    b = generate(small_config)["Route_Options"]
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data(small_config):
    from dataclasses import replace

    a = generate(small_config)["Hub_Constraints"]
    b = generate(replace(small_config, seed=small_config.seed + 1))["Hub_Constraints"]
    assert not a.equals(b)


def test_generated_workbook_round_trips_through_excel(tmp_path, small_config):
    path = write(tmp_path / "rt.xlsx", small_config)
    reloaded = load(path)
    assert reloaded.warnings == []


def test_every_shipment_references_a_real_material(dataset):
    known = set(dataset.materials["MaterialNo_Anon"])
    assert set(dataset.internal["MaterialNo_Anon"]) <= known


def test_every_route_references_real_hubs(dataset):
    known = set(dataset.hubs["HubID"])
    assert set(dataset.routes["FromHub"]) <= known
    assert set(dataset.routes["ToHub"]) <= known


def test_no_route_starts_and_ends_at_the_same_hub(dataset):
    """A leg from a facility to itself is not a movement."""
    assert not (dataset.routes["FromHub"] == dataset.routes["ToHub"]).any()


def test_external_shipments_link_to_real_internal_shipments(dataset):
    known = set(dataset.internal["ShipmentID"])
    assert set(dataset.external["InternalShipmentID_Link"]) <= known


def test_utilisation_percentages_are_fractions(dataset):
    for col in ("CurrentUtilizationPct", "MaxUtilizationPct", "CapacityReductionPct"):
        assert dataset.hubs[col].between(0.0, 1.0).all()


def test_coordinates_are_on_earth(dataset):
    assert dataset.hubs["Latitude"].between(-90, 90).all()
    assert dataset.hubs["Longitude"].between(-180, 180).all()


def test_generator_scales_to_the_requested_dimensions():
    cfg = SynthConfig(n_hubs=40, n_materials=20, n_families=6, n_internal=15, n_external=10,
                      alternatives_per_lane=4, seed=5)
    frames = generate(cfg)
    assert len(frames["Hub_Constraints"]) == 40
    assert len(frames["Material_Families"]) == 20
    assert len(frames["Internal_Shipments"]) == 15


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_haversine_matches_a_known_distance():
    # Singapore -> Frankfurt is about 10,200 km great-circle.
    d = haversine_km(1.3521, 103.8198, 50.1109, 8.6821)
    assert 9_800 < d < 10_600


def test_haversine_is_symmetric_and_zero_on_identity():
    assert haversine_km(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0)
    assert haversine_km(1.0, 2.0, 3.0, 4.0) == pytest.approx(haversine_km(3.0, 4.0, 1.0, 2.0))


def test_network_contains_every_hub_as_a_node(dataset, scenario):
    net = build_network(dataset, scenario)
    assert net.n_nodes == len(dataset.hubs)


def test_network_edges_carry_positive_weights(dataset, scenario):
    net = build_network(dataset, scenario)
    for _, _, data in net.graph.edges(data=True):
        assert data["weight"] > 0


def test_network_excludes_unavailable_routes(dataset, scenario):
    net = build_network(dataset, scenario)
    unavailable = set(
        dataset.routes.loc[dataset.routes["AvailableFlag"].str.casefold() != "yes", "RouteOptionID"]
    )
    edge_ids = {d["route_id"] for _, _, d in net.graph.edges(data=True)}
    assert not (edge_ids & unavailable)


def test_primary_hub_down_network_has_no_primary_edges(dataset):
    net = build_network(dataset, SCENARIOS["primary_hub_down"])
    assert not any(d["is_primary"] for _, _, d in net.graph.edges(data=True))


def test_end_to_end_paths_exist_for_most_families(dataset, scenario):
    """The synthetic network must be traversable, or the graph layer proves nothing."""
    net = build_network(dataset, scenario)
    families = dataset.routes["MaterialFamily"].unique()[:10]
    found = sum(net.best_stage_path(f) is not None for f in families)
    assert found >= len(families) * 0.5


def test_path_metrics_are_internally_consistent(dataset, scenario):
    net = build_network(dataset, scenario)
    family = dataset.routes["MaterialFamily"].iloc[0]
    path = net.best_stage_path(family, STAGE_ORDER)
    if path is None:
        pytest.skip("no end-to-end path for this family under this scenario")
    assert path["n_legs"] == len(path["legs"]) == len(path["hubs"]) - 1
    assert path["total_lead_days"] == pytest.approx(sum(leg["lead_days"] for leg in path["legs"]))
    assert 0.0 <= path["path_risk"] <= 1.0


def test_compounded_path_risk_exceeds_any_single_leg(dataset, scenario):
    """Chained risk must compound, not average — a long path is riskier than its legs."""
    net = build_network(dataset, scenario)
    for family in dataset.routes["MaterialFamily"].unique()[:20]:
        path = net.best_stage_path(family)
        if path and path["n_legs"] > 1:
            worst_leg = max(leg["risk"] / 5.0 for leg in path["legs"])
            assert path["path_risk"] >= worst_leg - 1e-9
            return
    pytest.skip("no multi-leg path available in this fixture")


def test_k_best_paths_are_returned_in_increasing_cost(dataset, scenario):
    net = build_network(dataset, scenario)
    edges = list(net.graph.edges())
    if not edges:
        pytest.skip("empty network")
    source, target = edges[0]
    paths = net.k_best_paths(source, target, k=4)
    weights = [p["total_weight"] for p in paths]
    assert weights == sorted(weights)


def test_critical_hubs_are_ranked_by_betweenness(dataset, scenario):
    net = build_network(dataset, scenario)
    top = net.critical_hubs(10)
    assert top["betweenness"].is_monotonic_decreasing
    assert (top["betweenness"] >= 0).all()
