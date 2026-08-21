"""Hard feasibility gates: which routes a shipment is *allowed* to take.

Optimisation is only as trustworthy as its constraint set. A cheap route that
cannot legally carry the material is not a saving, it is a defect. So every gate
here is a hard filter, applied before any score is computed, and every rejection
is *counted by reason* — the gate ledger is a first-class output, not a debug
print. If a shipment ends up with no candidates, the ledger says exactly which
gate closed the last door.

Gates
-----
1. **Lane**          route must serve the shipment's StageFrom -> StageTo
2. **Family**        route must be published for the shipment's MaterialFamily
3. **Availability**  ``AvailableFlag`` must be Yes
4. **Scenario**      route's ``DisruptionScenario`` must be active in this run
5. **Primary**       excluded when the scenario is a primary-hub-down drill
6. **Cold chain**    cold-chain materials need cold-capable hubs at both ends
7. **Hazard**        both hubs must declare support for the material's hazard class
8. **Route capacity** route's weekly capacity must cover the shipment quantity
9. **Hub headroom**  both hubs must have residual weekly headroom for the quantity
10. **Shelf life**   lead time must fit inside the material's shelf life
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    COLD_CHAIN_LABEL,
    DEFAULT_CAPACITY_POLICY,
    CapacityPolicy,
    DisruptionScenario,
)
from .loader import Dataset

GATES: tuple[str, ...] = (
    "lane",
    "family",
    "availability",
    "scenario",
    "primary_excluded",
    "cold_chain",
    "hazard",
    "route_capacity",
    "hub_headroom",
    "shelf_life",
)


# ---------------------------------------------------------------------------
# Hub capacity
# ---------------------------------------------------------------------------


def hub_headroom(
    hubs: pd.DataFrame,
    scenario: DisruptionScenario,
    policy: CapacityPolicy = DEFAULT_CAPACITY_POLICY,
) -> pd.Series:
    """Residual weekly units each hub can still absorb, indexed by ``HubID``.

    .. math::

        \\text{headroom}_h = C_h \\cdot
            \\bigl(u^{\\max}_h - r_h \\cdot \\mathbb{1}[\\text{disrupted}]
                   - u^{\\text{cur}}_h - b \\bigr)^{+}

    where :math:`C_h` is weekly capacity, :math:`u^{\\max}` the utilisation
    ceiling, :math:`r_h` the scenario capacity reduction, :math:`u^{\\text{cur}}`
    current utilisation and :math:`b` the planning safety buffer.

    Note this is *linear* in the utilisation ceiling. An earlier iteration of
    this model applied the ceiling twice (``C·(u_max−r)·u_max − used``), which
    silently understated every hub's headroom by ~10% and made the capacity
    gate reject feasible routes. See ``docs/METHODOLOGY.md`` for the correction.
    """
    capacity = hubs["WeeklyCapacityUnits"].astype(float)
    ceiling = hubs["MaxUtilizationPct"].astype(float)
    current = hubs["CurrentUtilizationPct"].astype(float)
    reduction = hubs["CapacityReductionPct"].astype(float).fillna(0.0)

    if policy.apply_disruption_reduction and scenario.hub_disruptions:
        disrupted = hubs["DisruptionScenario"].astype(str).isin(scenario.hub_disruptions)
        reduction = reduction.where(disrupted, 0.0)
    else:
        reduction = pd.Series(0.0, index=hubs.index)

    headroom = capacity * (ceiling - reduction - current - policy.safety_buffer_pct)
    if policy.floor_at_zero:
        headroom = headroom.clip(lower=0.0)
    return pd.Series(headroom.to_numpy(), index=hubs["HubID"].to_numpy(), name="headroom")


def _hazard_supported(supported: pd.Series, hazard: pd.Series) -> pd.Series:
    """Vectorised 'does this hub declare support for this hazard class'."""
    sup = supported.fillna("").astype(str).str.casefold()
    haz = hazard.fillna("").astype(str).str.casefold().str.strip()
    no_hazard = haz.isin({"", "none", "nan"})
    contains = pd.Series(
        [h in s for s, h in zip(sup, haz, strict=False)], index=supported.index
    )
    return no_hazard | contains


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """Every feasible (shipment, route) pair, plus the ledger of what was cut."""

    candidates: pd.DataFrame
    ledger: pd.DataFrame
    shipments: pd.DataFrame
    headroom: pd.Series
    scenario: DisruptionScenario
    unplaceable: list[str] = field(default_factory=list)

    @property
    def n_shipments(self) -> int:
        return len(self.shipments)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def coverage(self) -> float:
        """Share of in-scope shipments with at least one feasible route."""
        if not self.n_shipments:
            return 0.0
        return 1.0 - len(self.unplaceable) / self.n_shipments

    def summary(self) -> dict:
        per_ship = self.candidates.groupby("shipment_id").size() if self.n_candidates else pd.Series(dtype=int)
        return {
            "scenario": self.scenario.key,
            "shipments_in_scope": self.n_shipments,
            "feasible_pairs": self.n_candidates,
            "shipments_with_options": int(per_ship.size),
            "unplaceable": len(self.unplaceable),
            "coverage": round(self.coverage, 4),
            "median_options_per_shipment": float(per_ship.median()) if per_ship.size else 0.0,
        }


def _kg_per_unit(dataset: Dataset) -> tuple[pd.Series, float]:
    """Average kilograms per unit by material family, learned from the external sheet.

    The internal sheet counts pieces; the objective is denominated in cost per
    kilogram. External deliveries carry both a chargeable weight and a piece
    count, so they are the bridge between the two. Families never seen externally
    fall back to the global median rather than to 1.0, which would otherwise
    silently inflate their cost per kg by orders of magnitude.
    """
    ext = dataset.external.copy()
    if "MaterialFamily_Link" not in ext.columns or "Pieces" not in ext.columns:
        return pd.Series(dtype=float), 1.0
    ext = ext[(ext["Pieces"].fillna(0) > 0) & (ext["ChargeableWeight_KG"].fillna(0) > 0)]
    if ext.empty:
        return pd.Series(dtype=float), 1.0
    ext = ext.assign(kg_per_unit=ext["ChargeableWeight_KG"] / ext["Pieces"])
    by_family = ext.groupby("MaterialFamily_Link")["kg_per_unit"].median()
    return by_family, float(ext["kg_per_unit"].median())


def _in_scope(dataset: Dataset, scenario: DisruptionScenario) -> pd.DataFrame:
    """Join shipments to their material attributes and apply scenario scoping."""
    mats = dataset.materials[
        ["MaterialNo_Anon", "HazardClass", "TempRequirement", "PriorityClass", "ShelfLifeDays"]
    ]
    ship = dataset.internal.merge(mats, on="MaterialNo_Anon", how="left")
    ship["HazardClass"] = ship["HazardClass"].fillna("None")
    ship["TempRequirement"] = ship["TempRequirement"].fillna("Ambient")
    ship["PriorityClass"] = ship["PriorityClass"].fillna("Standard")
    ship["ShelfLifeDays"] = pd.to_numeric(ship["ShelfLifeDays"], errors="coerce").fillna(9_999)

    if scenario.only_cold_chain:
        ship = ship[ship["TempRequirement"] == COLD_CHAIN_LABEL]
    if scenario.only_expedite:
        ship = ship[ship["PriorityClass"].isin({"Expedite", "Critical"})]
    return ship.reset_index(drop=True)


def build_candidates(
    dataset: Dataset,
    scenario: DisruptionScenario,
    policy: CapacityPolicy = DEFAULT_CAPACITY_POLICY,
    max_splits: int = 1,
) -> CandidateSet:
    """Cross shipments with routes, apply all ten gates, return survivors + ledger.

    ``max_splits`` must match the value later passed to the MILP. The two
    capacity gates are all-or-nothing against the *whole* shipment quantity, so
    if the optimiser is allowed to divide a shipment into N lots, the gates must
    test one lot's worth of volume — otherwise the gate throws away routes the
    optimiser could legally have used, and the split relaxation silently does
    nothing.
    """
    ship = _in_scope(dataset, scenario)
    routes = dataset.routes.copy()
    hubs = dataset.hubs.set_index("HubID")
    headroom = hub_headroom(dataset.hubs, scenario, policy)

    kg_by_family, kg_median = _kg_per_unit(dataset)
    ship["kg_per_unit"] = (
        ship["MaterialFamily"].map(kg_by_family).fillna(kg_median).clip(lower=1e-6)
    )
    ship["weight_kg"] = (ship["Qty"].astype(float) * ship["kg_per_unit"]).clip(lower=1e-6)

    ledger: dict[str, int] = dict.fromkeys(GATES, 0)

    # ---- Gates 3-5 act on the route table alone: apply once, not per shipment.
    n0 = len(routes)
    routes = routes[routes["AvailableFlag"].astype(str).str.casefold() == "yes"]
    ledger["availability"] = n0 - len(routes)

    n0 = len(routes)
    routes = routes[routes["DisruptionScenario"].astype(str).isin(scenario.route_scenarios)]
    ledger["scenario"] = n0 - len(routes)

    if scenario.exclude_primary:
        n0 = len(routes)
        routes = routes[routes["IsPrimary"].astype(str).str.casefold() != "yes"]
        ledger["primary_excluded"] = n0 - len(routes)

    # ---- Gates 1-2 are the join keys, so the join itself enforces them.
    pairs = ship.merge(
        routes,
        left_on=["MaterialFamily", "StageFrom", "StageTo"],
        right_on=["MaterialFamily", "StageFrom", "StageTo"],
        how="inner",
        suffixes=("", "_route"),
    )
    ledger["lane"] = ledger["family"] = 0  # accounted for by construction

    if pairs.empty:
        return CandidateSet(
            candidates=_empty_candidates(),
            ledger=_ledger_frame(ledger),
            shipments=ship,
            headroom=headroom,
            scenario=scenario,
            unplaceable=ship["ShipmentID"].tolist(),
        )

    # ---- Attach hub attributes for both ends of the leg.
    for side, hub_col in (("from", "FromHub"), ("to", "ToHub")):
        attrs = hubs[["ColdChainAvailable", "SupportedHazardClasses", "Stage"]]
        attrs = attrs.rename(columns={c: f"{side}_{c}" for c in attrs.columns})
        pairs = pairs.join(attrs, on=hub_col)
        pairs[f"{side}_headroom"] = pairs[hub_col].map(headroom).fillna(0.0)
        for col in (f"{side}_ColdChainAvailable", f"{side}_SupportedHazardClasses"):
            pairs[col] = pairs[col].fillna("")

    # ---- Gate 6: cold chain
    needs_cold = pairs["TempRequirement"].eq(COLD_CHAIN_LABEL)
    cold_ok = (
        pairs["from_ColdChainAvailable"].astype(str).str.casefold().eq("yes")
        & pairs["to_ColdChainAvailable"].astype(str).str.casefold().eq("yes")
    )
    g_cold = (~needs_cold) | cold_ok

    # ---- Gate 7: hazard handling
    g_hazard = _hazard_supported(pairs["from_SupportedHazardClasses"], pairs["HazardClass"]) & (
        _hazard_supported(pairs["to_SupportedHazardClasses"], pairs["HazardClass"])
    )

    # ---- Gate 8: route weekly capacity, scaled by any mode-level rationing
    n_lots = max(1, int(max_splits))
    lot_qty = pairs["Qty"].astype(float) / n_lots
    mode_mult = pairs["TransportMode"].map(scenario.mode_capacity_multiplier).fillna(1.0)
    effective_route_capacity = pairs["CapacityUnitsPerWeek"].astype(float) * mode_mult
    g_route_cap = effective_route_capacity >= lot_qty

    # ---- Gate 9: hub headroom at both ends
    g_headroom = (pairs["from_headroom"] >= lot_qty) & (pairs["to_headroom"] >= lot_qty)

    # ---- Gate 10: shelf life
    g_shelf = pairs["BaseLeadTimeDays"].astype(float) <= pairs["ShelfLifeDays"].astype(float)

    # Ledger counts rejections attributable to each gate *in isolation*, which is
    # what a planner wants to know ("what is blocking me"), rather than a
    # sequential funnel that hides overlapping causes.
    ledger["cold_chain"] = int((~g_cold).sum())
    ledger["hazard"] = int((~g_hazard).sum())
    ledger["route_capacity"] = int((~g_route_cap).sum())
    ledger["hub_headroom"] = int((~g_headroom).sum())
    ledger["shelf_life"] = int((~g_shelf).sum())

    feasible = g_cold & g_hazard & g_route_cap & g_headroom & g_shelf
    kept = pairs[feasible].copy()

    # ---- Economics: freight + handling at both hubs.
    fixed = (
        kept["FromHub"].map(hubs.get("FixedHandlingCost_EUR", pd.Series(dtype=float))).fillna(0.0)
        + kept["ToHub"].map(hubs.get("FixedHandlingCost_EUR", pd.Series(dtype=float))).fillna(0.0)
    )
    variable_rate = (
        kept["FromHub"].map(hubs.get("VariableHandlingCostPerUnit_EUR", pd.Series(dtype=float))).fillna(0.0)
        + kept["ToHub"].map(hubs.get("VariableHandlingCostPerUnit_EUR", pd.Series(dtype=float))).fillna(0.0)
    )
    total_cost = kept["BaseCostEUR"].astype(float) + fixed + variable_rate * kept["Qty"].astype(float)

    out = pd.DataFrame(
        {
            "shipment_id": kept["ShipmentID"].to_numpy(),
            "route_id": kept["RouteOptionID"].to_numpy(),
            "material_family": kept["MaterialFamily"].to_numpy(),
            "stage_from": kept["StageFrom"].to_numpy(),
            "stage_to": kept["StageTo"].to_numpy(),
            "from_hub": kept["FromHub"].to_numpy(),
            "to_hub": kept["ToHub"].to_numpy(),
            "mode": kept["TransportMode"].to_numpy(),
            "is_primary": kept["IsPrimary"].astype(str).str.casefold().eq("yes").to_numpy(),
            "qty": kept["Qty"].astype(float).to_numpy(),
            "weight_kg": kept["weight_kg"].astype(float).to_numpy(),
            "lead_days": kept["BaseLeadTimeDays"].astype(float).to_numpy(),
            "freight_eur": kept["BaseCostEUR"].astype(float).to_numpy(),
            "total_cost_eur": total_cost.to_numpy(),
            "cost_per_kg": (total_cost.to_numpy() / kept["weight_kg"].to_numpy()),
            "risk": kept["RiskScore"].astype(float).to_numpy(),
            "co2_kg": kept["CO2Kg"].astype(float).to_numpy(),
            "route_capacity": effective_route_capacity[feasible].to_numpy(),
            "priority_class": kept["PriorityClass"].to_numpy(),
            "temp_requirement": kept["TempRequirement"].to_numpy(),
        }
    )

    with_options = set(out["shipment_id"].unique())
    unplaceable = [s for s in ship["ShipmentID"] if s not in with_options]

    return CandidateSet(
        candidates=out.reset_index(drop=True),
        ledger=_ledger_frame(ledger),
        shipments=ship,
        headroom=headroom,
        scenario=scenario,
        unplaceable=unplaceable,
    )


def _empty_candidates() -> pd.DataFrame:
    cols = [
        "shipment_id", "route_id", "material_family", "stage_from", "stage_to",
        "from_hub", "to_hub", "mode", "is_primary", "qty", "weight_kg",
        "lead_days", "freight_eur", "total_cost_eur", "cost_per_kg", "risk",
        "co2_kg", "route_capacity", "priority_class", "temp_requirement",
    ]
    return pd.DataFrame({c: pd.Series(dtype="float64" if c not in {
        "shipment_id", "route_id", "material_family", "stage_from", "stage_to",
        "from_hub", "to_hub", "mode", "priority_class", "temp_requirement",
    } else "object") for c in cols})


def _ledger_frame(ledger: dict[str, int]) -> pd.DataFrame:
    return (
        pd.DataFrame({"gate": list(ledger), "rejected_pairs": [ledger[g] for g in ledger]})
        .sort_values("rejected_pairs", ascending=False)
        .reset_index(drop=True)
    )


__all__ = ["GATES", "CandidateSet", "build_candidates", "hub_headroom"]
