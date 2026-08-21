"""Chainguard — resilient semiconductor supply-chain route optimisation.

Typical use::

    from chainguard import load, build_candidates, solve_milp, SCENARIOS

    dataset = load("data/synthetic.xlsx")
    candidates = build_candidates(dataset, SCENARIOS["port_congestion"])
    plan = solve_milp(candidates)
    print(plan.summary())

The layers, in the order data flows through them:

``loader``       typed frames, schema contract enforced at the boundary
``feasibility``  ten hard gates plus a ledger of what each one rejected
``scoring``      the 40/40/20 objective, min-max normalised per shipment
``optimize``     greedy baseline, capacity repair, and the global CP-SAT MILP
``simulate``     Monte Carlo lead times -> P90, on-time probability, CVaR
``network``      NetworkX multi-leg paths and structural chokepoints
``benchmark``    the reproducible head-to-head behind every documented number
"""

from __future__ import annotations

from .config import (
    DEFAULT_SIMULATION,
    DEFAULT_SOLVER,
    DEFAULT_WEIGHTS,
    SCENARIOS,
    DisruptionScenario,
    ObjectiveWeights,
    SimulationConfig,
    SolverConfig,
)
from .feasibility import CandidateSet, build_candidates, hub_headroom
from .loader import Dataset, load
from .network import RouteNetwork, build_network
from .optimize import Plan, capacity_violations, solve_greedy, solve_milp
from .scoring import objective_value, score_candidates, summarise_plan
from .simulate import service_level_summary, simulate_plan
from .synth import SynthConfig
from .synth import write as write_synthetic

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_SIMULATION",
    "DEFAULT_SOLVER",
    "DEFAULT_WEIGHTS",
    "SCENARIOS",
    "CandidateSet",
    "Dataset",
    "DisruptionScenario",
    "ObjectiveWeights",
    "Plan",
    "RouteNetwork",
    "SimulationConfig",
    "SolverConfig",
    "SynthConfig",
    "__version__",
    "build_candidates",
    "build_network",
    "capacity_violations",
    "hub_headroom",
    "load",
    "objective_value",
    "score_candidates",
    "service_level_summary",
    "simulate_plan",
    "solve_greedy",
    "solve_milp",
    "summarise_plan",
    "write_synthetic",
]
