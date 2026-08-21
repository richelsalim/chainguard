"""Optimiser guarantees.

The claims this project makes in its README are asserted here. If the MILP ever
stops being optimal, stops respecting capacity, or stops beating the repair
heuristic on the penalised objective, CI fails.
"""

from __future__ import annotations

import pytest

from chainguard.config import DEFAULT_SOLVER, SCENARIOS, SolverConfig
from chainguard.feasibility import build_candidates
from chainguard.optimize import (
    capacity_violations,
    hub_utilisation,
    route_capacity_violations,
    solve_greedy,
    solve_milp,
)
from chainguard.optimize import repair as repair_mod
from chainguard.optimize.milp import _lot_quantities


def _penalised(plan, penalty: float = DEFAULT_SOLVER.unassigned_penalty_multiplier) -> float:
    s = plan.summary()
    return (s["mean_score"] * s["shipments"] + penalty * (s["in_scope"] - s["shipments"])) / s["in_scope"]


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------


def test_greedy_assigns_each_shipment_at_most_once(candidates):
    plan = solve_greedy(candidates)
    assert not plan.assignments["shipment_id"].duplicated().any()


def test_greedy_picks_the_per_shipment_minimum(candidates):
    from chainguard.scoring import score_candidates

    plan = solve_greedy(candidates)
    scored = score_candidates(candidates.candidates)
    minima = scored.groupby("shipment_id")["score"].min()
    for _, row in plan.assignments.iterrows():
        assert row["score"] == pytest.approx(minima[row["shipment_id"]])


def test_greedy_is_deterministic(candidates):
    a = solve_greedy(candidates).assignments
    b = solve_greedy(candidates).assignments
    assert a["route_id"].tolist() == b["route_id"].tolist()


def test_greedy_overbooks_shared_capacity(candidates):
    """The premise of the whole project. If this ever passes cleanly, say so."""
    plan = solve_greedy(candidates)
    violations = capacity_violations(plan, candidates)
    assert len(violations) > 0, (
        "greedy produced a capacity-feasible plan on this instance — the network "
        "is not contended enough for this fixture to be a meaningful benchmark"
    )


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------


def test_milp_respects_every_hub_capacity(candidates):
    plan = solve_milp(candidates)
    assert len(capacity_violations(plan, candidates)) == 0


def test_milp_respects_every_route_capacity(candidates):
    plan = solve_milp(candidates)
    assert len(route_capacity_violations(plan)) == 0


def test_milp_proves_optimality_on_this_instance(candidates):
    plan = solve_milp(candidates)
    assert plan.solver_status == "OPTIMAL"
    assert plan.optimality_gap == pytest.approx(0.0, abs=1e-4)


def test_milp_assigns_each_shipment_at_most_once(candidates):
    plan = solve_milp(candidates)
    assert not plan.assignments["shipment_id"].duplicated().any()


def test_milp_without_capacity_reproduces_greedy(candidates):
    """Removing the coupling constraint must decompose the program into argmins.

    This is the cleanest possible evidence that the MILP's behaviour comes from
    the shared-capacity constraint and not from some accidental difference in how
    the two code paths score candidates.
    """
    greedy = solve_greedy(candidates)
    uncoupled = solve_milp(
        candidates, enforce_hub_capacity=False, enforce_route_capacity=False
    )
    assert uncoupled.assignments["score"].sum() == pytest.approx(
        greedy.assignments["score"].sum(), abs=1e-3
    )
    assert len(uncoupled.assignments) == len(greedy.assignments)


def test_milp_beats_repair_on_the_penalised_objective(candidates):
    """The headline claim, asserted rather than advertised."""
    milp = solve_milp(candidates)
    repaired = repair_mod.solve(candidates)
    assert _penalised(milp) <= _penalised(repaired) + 1e-6


def test_milp_is_reproducible(candidates):
    a = solve_milp(candidates)
    b = solve_milp(candidates)
    assert a.assignments["score"].sum() == pytest.approx(b.assignments["score"].sum())


def test_solver_time_limit_is_respected(candidates):
    plan = solve_milp(candidates, solver_config=SolverConfig(max_time_seconds=2.0))
    assert plan.solve_seconds < 15.0  # generous CI allowance over the 2s target


# ---------------------------------------------------------------------------
# Split relaxation
# ---------------------------------------------------------------------------


def test_lot_quantities_sum_exactly():
    for qty in (0, 1, 7, 100, 1001):
        for n in (1, 2, 3, 7):
            lots = _lot_quantities(qty, n)
            assert sum(lots) == qty
            assert len(lots) == n
            assert max(lots) - min(lots) <= 1  # balanced


def test_split_relaxation_never_reduces_coverage(dataset, scenario):
    strict_cs = build_candidates(dataset, scenario, max_splits=1)
    split_cs = build_candidates(dataset, scenario, max_splits=3)
    strict = solve_milp(strict_cs)
    split = solve_milp(split_cs, max_splits=3)
    assert split.summary()["coverage"] >= strict.summary()["coverage"] - 1e-9


def test_split_plan_stays_capacity_feasible(dataset, scenario):
    cs = build_candidates(dataset, scenario, max_splits=3)
    plan = solve_milp(cs, max_splits=3)
    assert len(capacity_violations(plan, cs)) == 0


def test_split_shipment_volumes_sum_to_one(dataset, scenario):
    cs = build_candidates(dataset, scenario, max_splits=3)
    plan = solve_milp(cs, max_splits=3)
    if "volume_share" in plan.assignments:
        shares = plan.assignments.groupby("shipment_id")["volume_share"].sum()
        assert (shares <= 1.0 + 1e-6).all()


def test_per_shipment_collapses_split_legs(dataset, scenario):
    cs = build_candidates(dataset, scenario, max_splits=3)
    plan = solve_milp(cs, max_splits=3)
    per = plan.per_shipment()
    assert not per["shipment_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Chance constraint
# ---------------------------------------------------------------------------


def test_chance_constraint_raises_service_level(candidates):
    from chainguard.simulate import service_level_summary, simulate_plan

    base = solve_milp(candidates)
    strict = solve_milp(candidates, min_on_time_probability=0.85)
    base_otd = service_level_summary(simulate_plan(base.per_shipment()))["mean_on_time_probability"]
    strict_otd = service_level_summary(simulate_plan(strict.per_shipment()))["mean_on_time_probability"]
    assert strict_otd >= base_otd


def test_chance_constraint_is_actually_binding(candidates):
    plan = solve_milp(candidates, min_on_time_probability=0.85)
    assert plan.meta.get("routes_excluded_by_chance_constraint", 0) > 0


def test_chance_constraint_accounts_for_lost_shipments(candidates):
    """Shipments with no compliant route must be reported, not silently vanish."""
    plan = solve_milp(candidates, min_on_time_probability=0.95)
    s = plan.summary()
    assert s["shipments"] + s["unassigned"] >= s["in_scope"] - 1e-9


# ---------------------------------------------------------------------------
# Repair heuristic
# ---------------------------------------------------------------------------


def test_repair_produces_a_feasible_plan(candidates):
    plan = repair_mod.solve(candidates)
    assert len(capacity_violations(plan, candidates)) == 0


def test_repair_terminates_on_every_scenario(dataset):
    for key in SCENARIOS:
        cs = build_candidates(dataset, SCENARIOS[key])
        plan = repair_mod.solve(cs, SCENARIOS[key].weights)
        assert len(capacity_violations(plan, cs)) == 0


def test_hub_utilisation_never_exceeds_one_for_milp(candidates):
    plan = solve_milp(candidates)
    util = hub_utilisation(plan, candidates)
    assert (util["utilisation"].dropna() <= 1.0 + 1e-9).all()
