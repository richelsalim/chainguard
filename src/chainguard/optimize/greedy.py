"""Greedy per-shipment selection — the baseline this project exists to beat.

This is the standard hackathon answer, and it is a *reasonable* one: score every
feasible route for a shipment, take the argmin, move on. It is fast, trivially
explainable, and per-shipment optimal.

It is also wrong at the network level, for one specific reason: **hub capacity
is a shared resource and greedy treats it as private**. Every shipment
independently picks the same handful of cheap, fast, low-risk hubs, and nothing
in the loop notices that their combined quantity has blown through those hubs'
weekly headroom. The plan looks excellent on paper and cannot be executed.

Quantifying that gap is the point of :func:`capacity_violations` — it is what
turns "we built an optimiser" into "here is what the naive answer costs you".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..feasibility import CandidateSet
from ..scoring import score_candidates, summarise_plan


@dataclass
class Plan:
    """A completed assignment of shipments to routes."""

    assignments: pd.DataFrame
    method: str
    scenario: str
    in_scope: int
    unassigned: list[str] = field(default_factory=list)
    solver_status: str = "n/a"
    solve_seconds: float = 0.0
    objective_bound: float | None = None
    meta: dict = field(default_factory=dict)

    def per_shipment(self) -> pd.DataFrame:
        """Collapse to one row per shipment, volume-weighting any split legs.

        With ``max_splits > 1`` a shipment can occupy several assignment rows.
        Every headline metric must be per *shipment*, not per row, or a split
        shipment would be double-counted and the mean score would drift.
        """
        df = self.assignments
        if df.empty:
            return df
        if "volume_share" not in df.columns or (df.groupby("shipment_id").size() <= 1).all():
            return df.reset_index(drop=True)

        w = df["volume_share"].to_numpy()
        weighted = df.assign(
            _score_w=df["score"].to_numpy() * w,
            _lead_w=df["lead_days"].to_numpy() * w,
            _risk_w=df["risk"].to_numpy() * w,
            _cpk_w=df["cost_per_kg"].to_numpy() * w,
            _co2_w=df["co2_kg"].to_numpy() * w,
        )
        agg = weighted.groupby("shipment_id", as_index=False).agg(
            score=("_score_w", "sum"),
            lead_days=("_lead_w", "sum"),
            risk=("_risk_w", "sum"),
            cost_per_kg=("_cpk_w", "sum"),
            co2_kg=("_co2_w", "sum"),
            total_cost_eur=("total_cost_eur", "sum"),
            qty=("qty", "sum"),
            weight_kg=("weight_kg", "sum"),
            is_primary=("is_primary", "max"),
            legs=("route_id", "count"),
            # A split shipment has no single route; join the ids so downstream
            # consumers (simulation, exports) still have a stable identifier.
            route_id=("route_id", lambda col: "+".join(map(str, col))),
            material_family=("material_family", "first"),
            mode=("mode", "first"),
            from_hub=("from_hub", "first"),
            to_hub=("to_hub", "first"),
        )
        return agg

    @property
    def mean_score(self) -> float:
        per = self.per_shipment()
        return float(per["score"].mean()) if len(per) else float("nan")

    def summary(self) -> dict:
        base = summarise_plan(self.per_shipment(), n_in_scope=self.in_scope)
        base.update(
            {
                "method": self.method,
                "scenario": self.scenario,
                "unassigned": len(self.unassigned),
                "solver_status": self.solver_status,
                "solve_seconds": round(self.solve_seconds, 3),
                "optimality_gap": self.optimality_gap,
            }
        )
        return base

    @property
    def optimality_gap(self) -> float | None:
        """Relative gap between the achieved objective and the proven bound."""
        if self.objective_bound is None or not len(self.assignments):
            return None
        achieved = float(self.assignments["score"].sum())
        if achieved <= 0:
            return 0.0
        return round(max(0.0, (achieved - self.objective_bound) / achieved), 6)


def solve(
    candidate_set: CandidateSet,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> Plan:
    """Pick each shipment's lowest-scoring feasible route, independently."""
    scored = score_candidates(candidate_set.candidates, weights)
    if scored.empty:
        return Plan(
            assignments=scored,
            method="greedy",
            scenario=candidate_set.scenario.key,
            in_scope=candidate_set.n_shipments,
            unassigned=list(candidate_set.unplaceable),
        )

    # Deterministic tie-breaking: score, then cost, then route id. Without this a
    # rerun can silently produce a different plan with an identical objective,
    # which makes the greedy-vs-MILP diff unreproducible.
    ordered = scored.sort_values(
        ["shipment_id", "score", "cost_per_kg", "route_id"], kind="mergesort"
    )
    best = ordered.groupby("shipment_id", as_index=False, sort=False).first()

    return Plan(
        assignments=best.reset_index(drop=True),
        method="greedy",
        scenario=candidate_set.scenario.key,
        in_scope=candidate_set.n_shipments,
        unassigned=list(candidate_set.unplaceable),
        solver_status="greedy-argmin",
    )


def capacity_violations(plan: Plan, candidate_set: CandidateSet) -> pd.DataFrame:
    """Where does this plan exceed shared hub headroom, and by how much?

    An assignment loads each **distinct** hub it touches, once. The distinction
    matters on intra-stage lanes (Backend -> Backend), where origin and
    destination can be the same facility: counting it twice would invent
    violations the solver correctly did not create, and the audit would
    contradict the constraint it is meant to verify.
    """
    if plan.assignments.empty:
        return pd.DataFrame(columns=["hub", "committed_units", "headroom_units", "excess_units", "overload_pct"])

    committed = hub_load(plan.assignments)
    headroom = candidate_set.headroom.reindex(committed.index).fillna(0.0)

    frame = pd.DataFrame(
        {
            "hub": committed.index,
            "committed_units": committed.to_numpy(),
            "headroom_units": headroom.to_numpy(),
        }
    )
    frame["excess_units"] = (frame["committed_units"] - frame["headroom_units"]).clip(lower=0.0)
    frame["overload_pct"] = (
        frame["committed_units"] / frame["headroom_units"].replace(0.0, float("nan"))
    )
    return (
        frame[frame["excess_units"] > 0]
        .sort_values("excess_units", ascending=False)
        .reset_index(drop=True)
    )


def hub_load(assignments: pd.DataFrame) -> pd.Series:
    """Units committed to each hub by a plan, counting distinct hubs per leg."""
    origin = assignments[["from_hub", "qty"]].rename(columns={"from_hub": "hub"})
    dest = assignments.loc[
        assignments["to_hub"] != assignments["from_hub"], ["to_hub", "qty"]
    ].rename(columns={"to_hub": "hub"})
    return pd.concat([origin, dest]).groupby("hub")["qty"].sum()


def route_capacity_violations(plan: Plan) -> pd.DataFrame:
    """Route options booked beyond their own weekly capacity by the whole plan."""
    if plan.assignments.empty:
        return pd.DataFrame(columns=["route_id", "committed_units", "route_capacity", "excess_units"])

    grouped = plan.assignments.groupby("route_id").agg(
        committed_units=("qty", "sum"), route_capacity=("route_capacity", "first")
    )
    grouped["excess_units"] = (grouped["committed_units"] - grouped["route_capacity"]).clip(lower=0.0)
    return (
        grouped[grouped["excess_units"] > 0]
        .reset_index()
        .sort_values("excess_units", ascending=False)
        .reset_index(drop=True)
    )
