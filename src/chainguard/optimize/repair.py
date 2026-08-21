"""Capacity repair — the honest baseline to benchmark the MILP against.

Comparing a raw greedy plan to the MILP is not a fair fight, and reporting it as
one would be a modelling sin: greedy's plan is *cheaper because it is illegal*.
It books hubs past their headroom, so of course its mean score looks good.

What a real planner does when the naive plan blows a capacity limit is repair it:
find the worst-overloaded hub, move the least-painful shipment off it, repeat.
That is a legitimate, widely used heuristic (greedy construction + local repair),
and it is the baseline the MILP has to beat to justify its existence.

The loop is a min-regret one. At each step it considers every shipment currently
loading an over-committed hub, prices the score penalty of moving it to its best
alternative route that does not itself overload anything, and executes the
cheapest such move. When a shipment has no legal alternative it is dropped —
which is exactly what happens on a real planning board.
"""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..feasibility import CandidateSet
from ..scoring import score_candidates
from .greedy import Plan
from .greedy import solve as solve_greedy

MAX_ITERATIONS = 10_000


def _hub_load(assignments: dict[str, dict]) -> dict[str, float]:
    load: dict[str, float] = {}
    for row in assignments.values():
        load[row["from_hub"]] = load.get(row["from_hub"], 0.0) + row["qty"]
        if row["to_hub"] != row["from_hub"]:
            load[row["to_hub"]] = load.get(row["to_hub"], 0.0) + row["qty"]
    return load


def solve(
    candidate_set: CandidateSet,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    max_iterations: int = MAX_ITERATIONS,
) -> Plan:
    """Greedy construction followed by min-regret capacity repair."""
    seed = solve_greedy(candidate_set, weights)
    if seed.assignments.empty:
        return Plan(
            assignments=seed.assignments,
            method="greedy_repair",
            scenario=candidate_set.scenario.key,
            in_scope=candidate_set.n_shipments,
            unassigned=list(candidate_set.unplaceable),
            solver_status="empty",
        )

    scored = score_candidates(candidate_set.candidates, weights)
    options: dict[str, list[dict]] = {
        sid: grp.sort_values(["score", "cost_per_kg", "route_id"], kind="mergesort").to_dict("records")
        for sid, grp in scored.groupby("shipment_id")
    }
    headroom = candidate_set.headroom.to_dict()

    current: dict[str, dict] = {r["shipment_id"]: r for r in seed.assignments.to_dict("records")}
    dropped: list[str] = []
    moves = 0

    for _ in range(max_iterations):
        load = _hub_load(current)
        overloaded = {
            hub: qty - headroom.get(hub, 0.0)
            for hub, qty in load.items()
            if qty > headroom.get(hub, 0.0) + 1e-9
        }
        if not overloaded:
            break

        worst_hub = max(overloaded, key=overloaded.get)
        occupants = [
            sid for sid, row in current.items()
            if worst_hub in (row["from_hub"], row["to_hub"])
        ]

        best_move: tuple[float, str, dict | None] | None = None
        for sid in occupants:
            row = current[sid]
            # Load the network would have without this shipment.
            trial_load = dict(load)
            trial_load[row["from_hub"]] -= row["qty"]
            if row["to_hub"] != row["from_hub"]:
                trial_load[row["to_hub"]] -= row["qty"]

            for alt in options.get(sid, []):
                if alt["route_id"] == row["route_id"]:
                    continue
                if worst_hub in (alt["from_hub"], alt["to_hub"]):
                    continue
                fits = all(
                    trial_load.get(h, 0.0) + alt["qty"] <= headroom.get(h, 0.0) + 1e-9
                    for h in {alt["from_hub"], alt["to_hub"]}
                )
                if not fits:
                    continue
                regret = alt["score"] - row["score"]
                if best_move is None or regret < best_move[0]:
                    best_move = (regret, sid, alt)
                break  # options are score-sorted: the first legal one is that shipment's best

        if best_move is not None:
            _, sid, alt = best_move
            current[sid] = alt
            moves += 1
            continue

        # Nothing can move: drop the largest occupant, which relieves the most
        # pressure per unit of lost coverage.
        victim = max(occupants, key=lambda s: current[s]["qty"])
        current.pop(victim)
        dropped.append(victim)

    assignments = (
        pd.DataFrame(list(current.values())).sort_values("shipment_id").reset_index(drop=True)
        if current
        else scored.iloc[0:0]
    )

    return Plan(
        assignments=assignments,
        method="greedy_repair",
        scenario=candidate_set.scenario.key,
        in_scope=candidate_set.n_shipments,
        unassigned=list(candidate_set.unplaceable) + dropped,
        solver_status="repaired",
        meta={"repair_moves": moves, "repair_drops": len(dropped)},
    )
