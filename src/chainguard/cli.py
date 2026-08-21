"""Command-line interface: ``chainguard <command>``.

Every capability in the library is reachable from here, so the whole project can
be reproduced end to end without writing a line of Python:

.. code-block:: bash

    chainguard synth                        # fabricate a runnable dataset
    chainguard profile                      # inspect a workbook against the schema
    chainguard optimize --scenario baseline # solve one scenario, compare methods
    chainguard simulate --scenario baseline # Monte Carlo service levels
    chainguard network  --scenario baseline # graph stats and chokepoints
    chainguard benchmark                    # full head-to-head, writes artifacts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import benchmark as bench
from .config import DEFAULT_SIMULATION, SCENARIOS, SimulationConfig
from .feasibility import build_candidates
from .loader import load
from .network import build_network
from .optimize import capacity_violations, solve_greedy, solve_milp
from .optimize import repair as repair_mod
from .scoring import score_candidates
from .simulate import (
    decision_flips,
    risk_adjusted_ranking,
    service_level_summary,
    simulate_plan,
)
from .synth import SynthConfig
from .synth import write as write_synth

DEFAULT_DATA = "data/synthetic.xlsx"


def _print_df(df: pd.DataFrame, max_rows: int = 40) -> None:
    if df.empty:
        print("  (no rows)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows} more rows")


def _sim_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_draws=getattr(args, "draws", DEFAULT_SIMULATION.n_draws),
        seed=getattr(args, "seed", DEFAULT_SIMULATION.seed),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    cfg = SynthConfig(seed=args.seed, n_internal=args.shipments, n_hubs=args.hubs)
    path = write_synth(args.out, cfg)
    size_kb = path.stat().st_size / 1024
    print(f"Wrote synthetic workbook -> {path} ({size_kb:,.0f} KB, seed={args.seed})")
    print("Every value in it is fabricated. No real supply-chain data is used anywhere.")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    dataset = load(args.data, strict=not args.lenient)
    print(f"Source: {dataset.source}\n")
    _print_df(dataset.profile())
    if dataset.warnings:
        print("\nValidation warnings:")
        for w in dataset.warnings:
            print(f"  - {w}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    dataset = load(args.data)
    scenario = SCENARIOS[args.scenario]
    print(f"Scenario: {scenario.label} — {scenario.tagline}\n")

    cs = build_candidates(dataset, scenario, max_splits=args.max_splits)
    print("Feasible candidate set:")
    print(json.dumps(cs.summary(), indent=2))
    print("\nGate rejection ledger (pairs cut by each hard gate, counted independently):")
    _print_df(cs.ledger)

    rows = []
    plans = {
        "greedy": solve_greedy(cs, scenario.weights),
        "greedy_repair": repair_mod.solve(cs, scenario.weights),
        "milp": solve_milp(
            cs,
            scenario.weights,
            max_splits=args.max_splits,
            min_on_time_probability=args.min_otd,
        ),
    }
    for name, plan in plans.items():
        v = capacity_violations(plan, cs)
        s = plan.summary()
        rows.append(
            {
                "method": name,
                "mean_score": round(s["mean_score"], 5),
                "coverage": s["coverage"],
                "assigned": s["shipments"],
                "hub_violations": len(v),
                "excess_units": int(v["excess_units"].sum()) if len(v) else 0,
                "executable": len(v) == 0,
                "status": plan.solver_status,
                "seconds": round(plan.solve_seconds, 3),
            }
        )
    print("\nMethod comparison:")
    _print_df(pd.DataFrame(rows))

    if args.out:
        best = plans["milp"]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        best.assignments.to_csv(args.out, index=False)
        print(f"\nWrote MILP plan -> {args.out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    dataset = load(args.data)
    scenario = SCENARIOS[args.scenario]
    sim_cfg = _sim_config(args)

    cs = build_candidates(dataset, scenario)
    plan = solve_milp(cs, scenario.weights)
    result = simulate_plan(plan.per_shipment(), sim_cfg)

    print(f"Scenario: {scenario.label}")
    print(f"Monte Carlo: {sim_cfg.n_draws:,} draws per leg, seed {sim_cfg.seed}\n")
    print("Plan service level:")
    print(json.dumps(service_level_summary(result), indent=2, default=float))

    print("\nWorst legs by CVaR95 (mean lead time in the worst 5% of outcomes):")
    _print_df(
        result.worst(10)[
            ["shipment_id", "route_id", "mode", "deterministic_lead_days", "p90", "cvar95", "on_time_probability"]
        ].round(3)
    )

    flips = decision_flips(risk_adjusted_ranking(score_candidates(cs.candidates, scenario.weights), scenario.weights, sim_cfg))
    print("\nCVaR-substitution diagnostic (see simulate.risk_adjusted_ranking):")
    print(json.dumps(flips, indent=2, default=float))
    if flips.get("tail_days_saved", 0) < 0:
        print(
            "\n  Note: tail_days_saved is negative — substituting CVaR into the objective\n"
            "  makes the tail WORSE. This is why the service target belongs in the\n"
            "  constraint set (`--min-otd`), not in the objective."
        )
    return 0


def cmd_network(args: argparse.Namespace) -> int:
    dataset = load(args.data)
    scenario = SCENARIOS[args.scenario]
    net = build_network(dataset, scenario, scenario.weights)

    print(f"Scenario: {scenario.label}\n")
    print("Graph:")
    print(json.dumps(net.stats(), indent=2, default=float))

    print("\nStructural chokepoints (highest betweenness — most paths must cross them):")
    _print_df(net.critical_hubs(args.top))

    family = args.family or (
        dataset.routes["MaterialFamily"].iloc[0] if len(dataset.routes) else None
    )
    if family:
        print(f"\nBest end-to-end path for material family {family}:")
        path = net.best_stage_path(family)
        if path is None:
            print("  no end-to-end path exists under this scenario's constraints")
        else:
            print(f"  {' -> '.join(path['hubs'])}")
            print(
                f"  legs={path['n_legs']}  lead={path['total_lead_days']}d  "
                f"cost=EUR {path['total_cost_eur']:,.0f}  CO2={path['total_co2_kg']}kg  "
                f"compounded path risk={path['path_risk']}  distance={path['total_distance_km']:,.0f}km"
            )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    dataset = load(args.data)
    scenarios = tuple(args.scenarios) if args.scenarios else None
    print(f"Benchmarking {dataset.source} ...")
    table, diagnostics = bench.run(
        dataset,
        scenarios=scenarios,
        splits=tuple(args.splits),
        sla_targets=tuple(args.sla),
        simulation=_sim_config(args),
    )
    paths = bench.save(table, diagnostics, args.out_dir)

    print("\nHeadline:")
    print(json.dumps(diagnostics["headline"], indent=2, default=float))
    print("\nResults:")
    _print_df(
        table[
            ["scenario", "method", "mean_score", "coverage", "hub_violations", "executable", "solve_seconds"]
        ]
    )
    print("\nWrote:")
    for kind, path in paths.items():
        print(f"  {kind:9s} {path}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chainguard",
        description="Resilient semiconductor supply-chain route optimiser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default=DEFAULT_DATA, help=f"source workbook (default: {DEFAULT_DATA})")
        p.add_argument(
            "--scenario",
            default="baseline",
            choices=sorted(SCENARIOS),
            help="disruption scenario to evaluate",
        )

    p_synth = sub.add_parser("synth", help="generate a synthetic workbook (no real data)")
    p_synth.add_argument("--out", default=DEFAULT_DATA)
    p_synth.add_argument("--seed", type=int, default=42)
    p_synth.add_argument("--shipments", type=int, default=240)
    p_synth.add_argument("--hubs", type=int, default=488)
    p_synth.set_defaults(func=cmd_synth)

    p_profile = sub.add_parser("profile", help="validate a workbook against the schema")
    p_profile.add_argument("--data", default=DEFAULT_DATA)
    p_profile.add_argument("--lenient", action="store_true", help="warn instead of raising")
    p_profile.set_defaults(func=cmd_profile)

    p_opt = sub.add_parser("optimize", help="solve one scenario and compare methods")
    add_common(p_opt)
    p_opt.add_argument("--max-splits", type=int, default=1, help="lots a shipment may be split into")
    p_opt.add_argument("--min-otd", type=float, default=None, help="chance constraint on on-time probability, e.g. 0.85")
    p_opt.add_argument("--out", default=None, help="write the MILP plan to this CSV")
    p_opt.set_defaults(func=cmd_optimize)

    p_sim = sub.add_parser("simulate", help="Monte Carlo service levels for the optimal plan")
    add_common(p_sim)
    p_sim.add_argument("--draws", type=int, default=DEFAULT_SIMULATION.n_draws)
    p_sim.add_argument("--seed", type=int, default=DEFAULT_SIMULATION.seed)
    p_sim.set_defaults(func=cmd_simulate)

    p_net = sub.add_parser("network", help="graph statistics, chokepoints, end-to-end paths")
    add_common(p_net)
    p_net.add_argument("--top", type=int, default=10)
    p_net.add_argument("--family", default=None, help="material family for the path search")
    p_net.set_defaults(func=cmd_network)

    p_bench = sub.add_parser("benchmark", help="full head-to-head across scenarios")
    p_bench.add_argument("--data", default=DEFAULT_DATA)
    p_bench.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS), default=None)
    p_bench.add_argument("--splits", nargs="*", type=int, default=[2, 3])
    p_bench.add_argument("--sla", nargs="*", type=float, default=[0.85])
    p_bench.add_argument("--draws", type=int, default=DEFAULT_SIMULATION.n_draws)
    p_bench.add_argument("--seed", type=int, default=DEFAULT_SIMULATION.seed)
    p_bench.add_argument("--out-dir", default="artifacts")
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
