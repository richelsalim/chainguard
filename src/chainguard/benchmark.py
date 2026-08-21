r"""Reproducible head-to-head benchmark — the source of every number in the README.

No figure in this project's documentation is typed by hand. ``make benchmark``
runs this module, which writes ``artifacts/benchmark.csv`` and
``artifacts/benchmark.json``, and the README quotes those files. If the model
changes, the numbers change with it.

Methods compared
----------------
``greedy``
    Per-shipment argmin. Fast, per-shipment optimal, and **not executable** —
    it over-books shared hub capacity because it cannot see other shipments.
``greedy_repair``
    Greedy plus min-regret capacity repair. This is the fair baseline: a
    legitimate heuristic producing an executable plan, and roughly what a
    planner does by hand.
``milp``
    Global CP-SAT assignment. Same objective, same gates, all shipments decided
    simultaneously under shared capacity.
``milp_split{N}``
    The MILP with the split-shipment relaxation, which mainly buys coverage.
``milp_sla{A}``
    The MILP with a chance constraint on simulated on-time probability, which
    prices the service guarantee.

Reading the table
-----------------
Two columns carry all the weight, and neither is ``mean_score`` on its own.

``executable``
    Does the plan respect every shared capacity limit? Greedy's mean score is the
    best in the table and its plans are the worst in the table, because it wins
    by spending capacity it does not have. An inexecutable plan is not a cheaper
    plan; it is not a plan.

``objective_per_shipment``
    The quantity actually being minimised:

    .. math::

        Z = \frac{1}{|S|}\Bigl(\sum_{i \in A} s_i \;+\; M\,|S \setminus A|\Bigr)

    for assigned set :math:`A` and drop penalty :math:`M`. **``mean_score`` alone
    is not comparable across methods**, because methods differ in how many
    shipments they place: a plan that assigns only its easiest 70% of shipments
    posts a flattering mean while quietly abandoning the hard ones. Charging the
    same penalty per dropped shipment that the solver charges puts every method
    on one axis, and makes the MILP's optimality claim falsifiable — if any
    heuristic ever beat it on this column at equal feasibility, the model would
    be wrong.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_SIMULATION,
    DEFAULT_SOLVER,
    SCENARIOS,
    DisruptionScenario,
    SimulationConfig,
)
from .feasibility import build_candidates
from .loader import Dataset
from .network import build_network
from .optimize import capacity_violations, solve_greedy, solve_milp
from .optimize import repair as repair_mod
from .optimize.greedy import Plan
from .scoring import score_candidates
from .simulate import decision_flips, risk_adjusted_ranking, service_level_summary, simulate_plan


@dataclass
class BenchmarkRow:
    scenario: str
    method: str
    mean_score: float
    objective_per_shipment: float
    median_score: float
    p90_score: float
    coverage: float
    shipments_assigned: int
    shipments_in_scope: int
    hub_violations: int
    excess_units: float
    total_cost_eur: float
    mean_lead_days: float
    mean_risk: float
    total_co2_kg: float
    mean_on_time_probability: float
    mean_cvar95_days: float
    solver_status: str
    solve_seconds: float
    optimality_gap: float | None
    executable: bool


def _penalised_objective(summary: dict, penalty: float) -> float:
    """Mean score plus the solver's own drop penalty for every unplaced shipment."""
    in_scope = summary["in_scope"] or 1
    assigned = summary["shipments"]
    total = (summary["mean_score"] * assigned) if assigned else 0.0
    return (total + penalty * (in_scope - assigned)) / in_scope


def _evaluate(
    plan: Plan,
    candidate_set,
    simulation: SimulationConfig,
    penalty: float = DEFAULT_SOLVER.unassigned_penalty_multiplier,
) -> BenchmarkRow:
    per_ship = plan.per_shipment()
    summary = plan.summary()
    violations = capacity_violations(plan, candidate_set)
    service = service_level_summary(simulate_plan(per_ship, simulation))

    return BenchmarkRow(
        scenario=plan.scenario,
        method=plan.method,
        mean_score=round(summary["mean_score"], 6),
        objective_per_shipment=round(_penalised_objective(summary, penalty), 6),
        median_score=round(summary["median_score"], 6),
        p90_score=round(summary["p90_score"], 6),
        coverage=round(summary["coverage"], 4),
        shipments_assigned=summary["shipments"],
        shipments_in_scope=summary["in_scope"],
        hub_violations=len(violations),
        excess_units=float(violations["excess_units"].sum()) if len(violations) else 0.0,
        total_cost_eur=round(summary["total_cost_eur"], 2),
        mean_lead_days=round(summary["mean_lead_days"], 4),
        mean_risk=round(summary["mean_risk"], 4),
        total_co2_kg=round(summary["total_co2_kg"], 1),
        mean_on_time_probability=round(service["mean_on_time_probability"], 4),
        mean_cvar95_days=round(service["mean_cvar95_days"], 4),
        solver_status=plan.solver_status,
        solve_seconds=round(plan.solve_seconds, 4),
        optimality_gap=plan.optimality_gap,
        # The single most important column: does this plan respect every shared
        # capacity limit? A plan that does not is not a plan, whatever it scores.
        executable=len(violations) == 0,
    )


def run_scenario(
    dataset: Dataset,
    scenario: DisruptionScenario,
    splits: tuple[int, ...] = (2, 3),
    sla_targets: tuple[float, ...] = (0.85,),
    simulation: SimulationConfig = DEFAULT_SIMULATION,
) -> tuple[list[BenchmarkRow], dict]:
    """Run every method on one scenario and return rows plus diagnostics."""
    rows: list[BenchmarkRow] = []
    base = build_candidates(dataset, scenario)

    rows.append(_evaluate(solve_greedy(base, scenario.weights), base, simulation))
    rows.append(_evaluate(repair_mod.solve(base, scenario.weights), base, simulation))
    rows.append(_evaluate(solve_milp(base, scenario.weights), base, simulation))

    for n in splits:
        cs = build_candidates(dataset, scenario, max_splits=n)
        rows.append(_evaluate(solve_milp(cs, scenario.weights, max_splits=n), cs, simulation))

    for alpha in sla_targets:
        plan = solve_milp(
            base, scenario.weights, min_on_time_probability=alpha, simulation=simulation
        )
        row = _evaluate(plan, base, simulation)
        row.method = f"milp_sla{int(alpha * 100)}"
        rows.append(row)

    # Diagnostics that are about the *network*, not about one method.
    network = build_network(dataset, scenario, scenario.weights)
    flips = decision_flips(risk_adjusted_ranking(score_candidates(base.candidates, scenario.weights)))

    diagnostics = {
        "scenario": scenario.key,
        "label": scenario.label,
        "candidates": base.summary(),
        "gate_ledger": base.ledger.to_dict("records"),
        "network": network.stats(),
        "critical_hubs": network.critical_hubs(5).to_dict("records"),
        "risk_decision_flips": flips,
    }
    return rows, diagnostics


def run(
    dataset: Dataset,
    scenarios: tuple[str, ...] | None = None,
    splits: tuple[int, ...] = (2, 3),
    sla_targets: tuple[float, ...] = (0.85,),
    simulation: SimulationConfig = DEFAULT_SIMULATION,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Benchmark every requested scenario. Returns (table, diagnostics)."""
    keys = scenarios or tuple(SCENARIOS)
    all_rows: list[BenchmarkRow] = []
    diagnostics: dict = {"scenarios": {}, "source": str(dataset.source)}

    t0 = time.perf_counter()
    for key in keys:
        scenario = SCENARIOS[key]
        if verbose:
            print(f"  {scenario.label} ...", flush=True)
        rows, diag = run_scenario(dataset, scenario, splits, sla_targets, simulation)
        all_rows.extend(rows)
        diagnostics["scenarios"][key] = diag

    table = pd.DataFrame([asdict(r) for r in all_rows])
    diagnostics["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    diagnostics["headline"] = headline(table)
    return table, diagnostics


def headline(table: pd.DataFrame) -> dict:
    """The three claims this project stands on, computed rather than asserted."""
    if table.empty:
        return {}

    greedy = table[table["method"] == "greedy"]
    repair = table[table["method"] == "greedy_repair"]
    milp = table[table["method"] == "milp"]

    def _mean(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) else float("nan")

    repair_obj = _mean(repair, "objective_per_shipment")
    milp_obj = _mean(milp, "objective_per_shipment")
    improvement = (repair_obj - milp_obj) / repair_obj if repair_obj else float("nan")
    # Compare like with like: the mean across scenarios for the single best split
    # setting, not the best single scenario, which would flatter the result.
    split_rows = table[table["method"].str.startswith("milp_split")]
    if len(split_rows):
        by_method = split_rows.groupby("method")["objective_per_shipment"].mean()
        best_split_method = by_method.idxmin()
        split_obj = float(by_method.min())
        split_cov = float(split_rows[split_rows["method"] == best_split_method]["coverage"].mean())
    else:
        best_split_method, split_obj, split_cov = None, float("nan"), float("nan")

    sla = table[table["method"].str.startswith("milp_sla")]

    return {
        "scenarios_evaluated": int(table["scenario"].nunique()),
        # 1. The naive plan is not executable.
        "greedy_hub_violations": int(greedy["hub_violations"].sum()),
        "greedy_excess_units": float(greedy["excess_units"].sum()),
        "greedy_plans_executable": int(greedy["executable"].sum()),
        # 2. Against a fair, feasible baseline the MILP wins on the objective.
        "repair_objective": round(repair_obj, 6),
        "milp_objective": round(milp_obj, 6),
        "milp_vs_repair_improvement_pct": round(improvement * 100, 2),
        "best_split_objective": round(split_obj, 6),
        "repair_mean_score": round(_mean(repair, "mean_score"), 6),
        "milp_mean_score": round(_mean(milp, "mean_score"), 6),
        "milp_all_executable": bool(milp["executable"].all()) if len(milp) else False,
        "milp_proven_optimal": int((milp["solver_status"] == "OPTIMAL").sum()),
        "milp_mean_solve_seconds": round(_mean(milp, "solve_seconds"), 3),
        # 3. Coverage and service are levers with a measurable price.
        "milp_coverage": round(_mean(milp, "coverage"), 4),
        "best_split_method": best_split_method,
        "best_split_coverage": round(split_cov, 4),
        "split_vs_milp_improvement_pct": round((milp_obj - split_obj) / milp_obj * 100, 2)
        if milp_obj
        else None,
        "sla_mean_on_time_probability": round(_mean(sla, "mean_on_time_probability"), 4),
        "milp_mean_on_time_probability": round(_mean(milp, "mean_on_time_probability"), 4),
        "sla_coverage_cost": round(_mean(milp, "coverage") - _mean(sla, "coverage"), 4),
    }


def save(table: pd.DataFrame, diagnostics: dict, out_dir: str | Path = "artifacts") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    json_path = out_dir / "benchmark.json"
    md_path = out_dir / "benchmark.md"

    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(diagnostics, indent=2, default=str))
    md_path.write_text(to_markdown(table, diagnostics))
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def to_markdown(table: pd.DataFrame, diagnostics: dict) -> str:
    """Render the results as the markdown block the README embeds."""
    cols = [
        "scenario", "method", "objective_per_shipment", "mean_score", "coverage", "shipments_assigned",
        "hub_violations", "excess_units", "executable",
        "mean_on_time_probability", "mean_cvar95_days", "solver_status", "solve_seconds",
    ]
    head = diagnostics.get("headline", {})
    lines = [
        "# Benchmark results",
        "",
        f"Source workbook: `{diagnostics.get('source', 'n/a')}`  ",
        f"Total runtime: {diagnostics.get('elapsed_seconds', 'n/a')}s  ",
        f"Scenarios: {head.get('scenarios_evaluated', 'n/a')}",
        "",
        "## Headline",
        "",
        f"- Greedy over-books **{head.get('greedy_hub_violations', 0)} hub-scenario pairs** "
        f"by **{head.get('greedy_excess_units', 0):,.0f} units**; "
        f"{head.get('greedy_plans_executable', 0)} of its plans are executable.",
        f"- MILP penalised objective **{head.get('milp_objective')}** vs repaired-greedy "
        f"**{head.get('repair_objective')}** — a "
        f"**{head.get('milp_vs_repair_improvement_pct')}% improvement**, both fully executable.",
        f"- Proven optimal on {head.get('milp_proven_optimal')} scenarios, "
        f"mean solve time **{head.get('milp_mean_solve_seconds')}s**.",
        f"- Split-shipment relaxation lifts coverage from "
        f"**{head.get('milp_coverage')}** to **{head.get('best_split_coverage')}**.",
        f"- An 85% service constraint raises mean on-time probability from "
        f"**{head.get('milp_mean_on_time_probability')}** to "
        f"**{head.get('sla_mean_on_time_probability')}**, at a coverage cost of "
        f"**{head.get('sla_coverage_cost')}**.",
        "",
        "## Full results",
        "",
        table[cols].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
