"""Global route assignment as a mixed-integer program, solved with CP-SAT.

Formulation
-----------
Let :math:`S` be the in-scope shipments, :math:`J_i` the feasible routes for
shipment :math:`i`, :math:`H` the hubs and :math:`R` the route options.

**Decision variables**

.. math::

    x_{ij} \\in \\{0,1\\} \\quad \\text{route } j \\text{ carries shipment } i \\\\
    z_i \\in \\{0,1\\} \\quad \\text{shipment } i \\text{ is left unassigned}

**Objective** — minimise total normalised score plus a drop penalty:

.. math::

    \\min \\; \\sum_{i \\in S} \\sum_{j \\in J_i} s_{ij}\\, x_{ij}
            \\;+\\; M \\sum_{i \\in S} z_i

**Constraints**

.. math::

    \\sum_{j \\in J_i} x_{ij} + z_i = 1
        &\\qquad \\forall i \\in S \\quad \\text{(assign exactly once, or drop)} \\\\
    \\sum_{(i,j) \\,:\\, h \\in \\{o_j, d_j\\}} q_i \\, x_{ij} \\;\\le\\; K_h
        &\\qquad \\forall h \\in H \\quad \\text{(shared hub headroom)} \\\\
    \\sum_{i \\,:\\, j \\in J_i} q_i \\, x_{ij} \\;\\le\\; U_j
        &\\qquad \\forall j \\in R \\quad \\text{(route weekly capacity)}

The hub constraint is the whole point. It is the only thing coupling shipments
to each other, and it is precisely what per-shipment greedy cannot see. Drop the
hub constraints and this program decomposes into |S| independent argmins — that
is, it *becomes* greedy. Keep them and the solver has to trade a slightly worse
route for one shipment against a much better one for another, which is a genuine
combinatorial problem (a multi-dimensional generalised assignment problem, and
NP-hard in general).

**Why CP-SAT rather than a classical LP/MIP solver?** The model is pure integer
with knapsack-style side constraints and no continuous relaxation of interest.
CP-SAT's portfolio search handles this shape well, ships in a permissively
licensed wheel with no separate solver install, proves optimality on instances
of this size in seconds, and — importantly for a public repo — anyone can `pip
install ortools` and reproduce the exact numbers.

**On the penalty M.** It is set to ``unassigned_penalty_multiplier`` times the
worst achievable assigned score, so dropping a shipment is always worse than any
placement. Modelling drops explicitly (rather than declaring the whole instance
infeasible) means a genuinely over-subscribed network still returns the best
partial plan *and tells you what it could not place* — which is the answer a
planner actually needs during a disruption.
"""

from __future__ import annotations

import math
import time

import pandas as pd
from ortools.sat.python import cp_model

from ..config import (
    DEFAULT_SIMULATION,
    DEFAULT_SOLVER,
    DEFAULT_WEIGHTS,
    ObjectiveWeights,
    SimulationConfig,
    SolverConfig,
)
from ..feasibility import CandidateSet
from ..scoring import score_candidates
from ..simulate import attach_service_metrics
from .greedy import Plan, hub_load


def _lot_quantities(qty: int, n_lots: int) -> list[int]:
    """Split an integer quantity into ``n_lots`` parts that sum back exactly."""
    base, remainder = divmod(qty, n_lots)
    return [base + (1 if k < remainder else 0) for k in range(n_lots)]


def solve(
    candidate_set: CandidateSet,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    solver_config: SolverConfig = DEFAULT_SOLVER,
    enforce_hub_capacity: bool = True,
    enforce_route_capacity: bool = True,
    max_splits: int = 1,
    min_on_time_probability: float | None = None,
    simulation: SimulationConfig = DEFAULT_SIMULATION,
) -> Plan:
    """Solve the global assignment. Returns the same :class:`Plan` type as greedy.

    Parameters
    ----------
    enforce_hub_capacity
        Deliberately exposed. Switching it off decomposes the program into |S|
        independent argmins and reproduces the greedy objective exactly — the
        cleanest possible demonstration that the improvement comes from the
        coupling constraint and not from an incidental scoring difference.
    max_splits
        Maximum number of distinct routes a single shipment's quantity may be
        divided across. ``1`` is the strict challenge rule (one shipment, one
        route). Values above 1 enable the **split-shipment relaxation**: the
        quantity is divided into equal lots that may travel separately.

        This matters because the single biggest cause of unplaceable shipments
        is a large quantity meeting a small residual headroom — an all-or-nothing
        constraint that no route choice can satisfy. Splitting is also what a
        real planner does (two trucks instead of one), so the relaxation is
        operationally meaningful rather than a modelling convenience. The
        shipment's score becomes the volume-weighted mean of its lots' scores,
        so a split is only chosen when it genuinely helps.
    min_on_time_probability
        Chance constraint. When set, a route may only carry a shipment if its
        Monte Carlo on-time probability meets this threshold, turning the service
        level into part of the feasible region rather than a term in the
        objective.

        This is the right place for a service target. Blended into the objective,
        a 40% cost term can outvote it and the "safer" plan silently gets less
        safe (see :func:`chainguard.simulate.risk_adjusted_ranking`). As a
        constraint it is guaranteed by construction, and what it costs — in
        objective value and in shipments that can no longer be placed at all —
        becomes directly measurable. That price *is* the answer to "what does a
        90% service guarantee cost us".
    """
    scored = score_candidates(candidate_set.candidates, weights)

    chance_meta: dict = {}
    chance_excluded: list[str] = []
    if min_on_time_probability is not None and not scored.empty:
        scored = attach_service_metrics(scored, simulation)
        before_rows = len(scored)
        before_ships = set(scored["shipment_id"].unique())
        scored = scored[scored["on_time_probability"] >= min_on_time_probability]
        after_ships = set(scored["shipment_id"].unique()) if not scored.empty else set()
        # Shipments with no route clearing the service bar are unassigned by the
        # constraint itself, before the solver ever sees them. They must still be
        # reported, or coverage would look better than it is.
        chance_excluded = sorted(before_ships - after_ships)
        chance_meta = {
            "min_on_time_probability": min_on_time_probability,
            "routes_excluded_by_chance_constraint": before_rows - len(scored),
            "shipments_lost_to_chance_constraint": len(chance_excluded),
        }
        # Re-normalise: the candidate pool changed, so the per-shipment min-max
        # baseline must be recomputed or scores would reference routes that are
        # no longer selectable.
        scored = score_candidates(scored, weights)

    if scored.empty:
        return Plan(
            assignments=scored,
            method="milp",
            scenario=candidate_set.scenario.key,
            in_scope=candidate_set.n_shipments,
            unassigned=list(candidate_set.unplaceable) + chance_excluded,
            solver_status="EMPTY",
            meta=chance_meta,
        )

    n_lots = max(1, int(max_splits))
    scale = solver_config.score_scale
    model = cp_model.CpModel()

    scored = scored.reset_index(drop=True)
    shipments = scored["shipment_id"].unique().tolist()

    rows_by_shipment: dict[str, list[int]] = {s: [] for s in shipments}
    for idx, sid in enumerate(scored["shipment_id"]):
        rows_by_shipment[sid].append(idx)

    ship_qty = {
        sid: int(round(float(scored.loc[rows[0], "qty"])))
        for sid, rows in rows_by_shipment.items()
    }
    lot_qty = {sid: _lot_quantities(q, n_lots) for sid, q in ship_qty.items()}

    # ---- Variables: one boolean per (candidate row, lot) ------------------
    # x[(row, lot)] = lot travels on that route.  z[(shipment, lot)] = lot dropped.
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for idx in range(len(scored)):
        for lot in range(n_lots):
            x[(idx, lot)] = model.NewBoolVar(f"x_{idx}_{lot}")
    z: dict[tuple[str, int], cp_model.IntVar] = {
        (sid, lot): model.NewBoolVar(f"z_{sid}_{lot}")
        for sid in shipments
        for lot in range(n_lots)
    }

    # ---- Each lot goes on exactly one route, or is explicitly dropped -----
    for sid, rows in rows_by_shipment.items():
        for lot in range(n_lots):
            model.Add(sum(x[(i, lot)] for i in rows) + z[(sid, lot)] == 1)
        # Symmetry breaking: lots of a shipment are interchangeable, so require
        # that lot L is only dropped if lot L-1 already was. Without this the
        # solver wastes search on n_lots! equivalent permutations.
        for lot in range(1, n_lots):
            model.Add(z[(sid, lot)] >= z[(sid, lot - 1)])
        # The number of distinct routes a shipment can use is bounded by its lot
        # count by construction, so `max_splits` needs no separate constraint.

    # ---- Shared hub headroom ---------------------------------------------
    hub_terms: dict[str, list[tuple[tuple[int, int], int]]] = {}
    if enforce_hub_capacity:
        for idx, (sid, o, d) in enumerate(
            zip(scored["shipment_id"], scored["from_hub"], scored["to_hub"], strict=False)
        ):
            for lot in range(n_lots):
                q = lot_qty[sid][lot]
                hub_terms.setdefault(o, []).append(((idx, lot), q))
                if d != o:
                    hub_terms.setdefault(d, []).append(((idx, lot), q))
        for hub, entries in hub_terms.items():
            cap = int(math.floor(float(candidate_set.headroom.get(hub, 0.0))))
            model.Add(sum(q * x[key] for key, q in entries) <= cap)

    # ---- Route weekly capacity -------------------------------------------
    if enforce_route_capacity:
        by_route: dict[str, list[int]] = {}
        for idx, rid in enumerate(scored["route_id"]):
            by_route.setdefault(rid, []).append(idx)
        caps = scored["route_capacity"].astype(float).tolist()
        sids = scored["shipment_id"].tolist()
        for _rid, rows in by_route.items():
            cap = int(math.floor(caps[rows[0]]))
            terms = [
                (lot_qty[sids[i]][lot], (i, lot)) for i in rows for lot in range(n_lots)
            ]
            if sum(q for q, _ in terms) > cap:  # constraint can actually bind
                model.Add(sum(q * x[key] for q, key in terms) <= cap)

    # ---- Objective --------------------------------------------------------
    # A lot carries 1/n_lots of its shipment's score contribution, so a shipment
    # split across routes is charged the mean of the routes it uses.
    drop_penalty = int(round(scale * solver_config.unassigned_penalty_multiplier / n_lots))
    obj_terms = []
    for idx, score_value in enumerate(scored["score"]):
        coeff = int(round(float(score_value) * scale / n_lots))
        for lot in range(n_lots):
            obj_terms.append(coeff * x[(idx, lot)])
    model.Minimize(sum(obj_terms) + drop_penalty * sum(z.values()))

    # ---- Solve ------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_config.max_time_seconds
    solver.parameters.num_search_workers = solver_config.num_workers
    solver.parameters.log_search_progress = solver_config.log_search_progress

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan(
            assignments=scored.iloc[0:0],
            method=_method_name(n_lots),
            scenario=candidate_set.scenario.key,
            in_scope=candidate_set.n_shipments,
            unassigned=list(candidate_set.unplaceable) + chance_excluded + shipments,
            solver_status=status_name,
            solve_seconds=elapsed,
            meta={"note": "no feasible assignment under the capacity constraints"},
        )

    # ---- Extract: collapse lots back to one row per (shipment, route) ------
    picks: list[dict] = []
    for idx in range(len(scored)):
        sid = scored.at[idx, "shipment_id"]
        lots_on_route = [lot for lot in range(n_lots) if solver.Value(x[(idx, lot)])]
        if not lots_on_route:
            continue
        row = scored.iloc[idx].to_dict()
        row["qty"] = sum(lot_qty[sid][lot] for lot in lots_on_route)
        row["lots"] = len(lots_on_route)
        row["volume_share"] = row["qty"] / ship_qty[sid] if ship_qty[sid] else 1.0
        picks.append(row)

    dropped_units = {
        sid: sum(lot_qty[sid][lot] for lot in range(n_lots) if solver.Value(z[(sid, lot)]))
        for sid in shipments
    }
    fully_dropped = [sid for sid, units in dropped_units.items() if units >= ship_qty[sid]]

    assignments = (
        pd.DataFrame(picks).sort_values(["shipment_id", "route_id"]).reset_index(drop=True)
        if picks
        else scored.iloc[0:0]
    )
    if not assignments.empty:
        # Re-derive per-shipment economics on the actually-shipped quantity.
        assignments["total_cost_eur"] = assignments["total_cost_eur"] * assignments["volume_share"]
        assignments["weight_kg"] = assignments["weight_kg"] * assignments["volume_share"]

    bound_scaled = solver.BestObjectiveBound()
    dropped_lots = sum(1 for key in z if solver.Value(z[key]))
    bound = max(0.0, (bound_scaled - drop_penalty * dropped_lots) / scale)

    return Plan(
        assignments=assignments,
        method=_method_name(n_lots),
        scenario=candidate_set.scenario.key,
        in_scope=candidate_set.n_shipments,
        unassigned=list(candidate_set.unplaceable) + chance_excluded + fully_dropped,
        solver_status=status_name,
        solve_seconds=elapsed,
        objective_bound=bound,
        meta={
            "variables": len(x) + len(z),
            "hub_constraints": len(hub_terms),
            "branches": solver.NumBranches(),
            "conflicts": solver.NumConflicts(),
            "enforce_hub_capacity": enforce_hub_capacity,
            "enforce_route_capacity": enforce_route_capacity,
            "max_splits": n_lots,
            "split_shipments": int(
                (assignments.groupby("shipment_id").size() > 1).sum()
            ) if not assignments.empty else 0,
            **chance_meta,
            "partially_served": int(
                sum(
                    1
                    for sid, units in dropped_units.items()
                    if 0 < units < ship_qty[sid]
                )
            ),
        },
    )


def _method_name(n_lots: int) -> str:
    return "milp" if n_lots == 1 else f"milp_split{n_lots}"


def hub_utilisation(plan: Plan, candidate_set: CandidateSet) -> pd.DataFrame:
    """Committed units vs headroom for every hub the plan touches."""
    if plan.assignments.empty:
        return pd.DataFrame(columns=["hub", "committed_units", "headroom_units", "utilisation"])

    committed = hub_load(plan.assignments)
    headroom = candidate_set.headroom.reindex(committed.index).fillna(0.0)
    return pd.DataFrame(
        {
            "hub": committed.index,
            "committed_units": committed.to_numpy(),
            "headroom_units": headroom.to_numpy(),
            "utilisation": committed.to_numpy() / headroom.replace(0.0, float("nan")).to_numpy(),
        }
    ).sort_values("utilisation", ascending=False).reset_index(drop=True)
