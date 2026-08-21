"""Statistical properties of the Monte Carlo layer.

Simulation code is uniquely easy to get subtly wrong and uniquely hard to notice:
it always returns plausible-looking numbers. These tests pin the distributional
behaviour to things that must be true analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from chainguard.config import SimulationConfig
from chainguard.simulate import (
    _gamma_params,
    attach_service_metrics,
    decision_flips,
    risk_adjusted_ranking,
    service_level_summary,
    simulate_lead_times,
    simulate_plan,
)


def test_gamma_params_recover_mean_and_cv():
    mean = np.array([5.0, 12.0])
    cv = np.array([0.2, 0.5])
    shape, scale = _gamma_params(mean, cv)
    assert np.allclose(shape * scale, mean)                 # E[X] = k*theta
    assert np.allclose(np.sqrt(shape) * scale / mean, cv)   # CV  = 1/sqrt(k)


def test_simulated_mean_converges_to_the_deterministic_lead_time():
    """With no disruption the Gamma must be unbiased around the promise."""
    cfg = SimulationConfig(n_draws=40_000, seed=1, disruption_base=0.0, disruption_per_risk_point=0.0)
    metrics, _ = simulate_lead_times(np.array([6.0]), np.array([2.0]), cfg)
    assert metrics["mean_lead_days"].iloc[0] == pytest.approx(6.0, rel=0.02)


def test_disruptions_shift_the_mean_upward_by_the_expected_amount():
    """E[extra] = P(disruption) x E[Exp(lambda)] — an analytic check on the shock term."""
    cfg = SimulationConfig(
        n_draws=80_000, seed=3, cv_base=0.01, cv_per_risk_point=0.0,
        disruption_base=0.5, disruption_per_risk_point=0.0, disruption_delay_mean_days=4.0,
    )
    metrics, _ = simulate_lead_times(np.array([10.0]), np.array([0.0]), cfg)
    assert metrics["mean_lead_days"].iloc[0] == pytest.approx(10.0 + 0.5 * 4.0, rel=0.03)


def test_percentiles_are_ordered():
    metrics, _ = simulate_lead_times(np.array([4.0, 9.0]), np.array([1.0, 4.0]))
    assert (metrics["p50"] <= metrics["p90"]).all()
    assert (metrics["p90"] <= metrics["p95"]).all()


def test_cvar_dominates_the_percentile_it_is_taken_from():
    """CVaR95 is the mean of the worst 5%, so it must exceed P95."""
    metrics, _ = simulate_lead_times(np.array([4.0, 9.0, 2.0]), np.array([1.0, 4.0, 3.0]))
    assert (metrics["cvar95"] >= metrics["p95"] - 1e-9).all()


def test_higher_risk_means_worse_service_at_equal_lead_time():
    """The core monotonicity the risk score is supposed to encode."""
    metrics, _ = simulate_lead_times(np.array([5.0, 5.0]), np.array([0.5, 4.5]))
    assert metrics["on_time_probability"].iloc[0] > metrics["on_time_probability"].iloc[1]
    assert metrics["cvar95"].iloc[0] < metrics["cvar95"].iloc[1]


def test_lead_times_are_never_negative():
    metrics, draws = simulate_lead_times(
        np.array([0.5]), np.array([5.0]), SimulationConfig(n_draws=5_000), keep_draws=True
    )
    assert draws.min() >= 0.0


def test_simulation_is_reproducible_under_a_fixed_seed():
    a, _ = simulate_lead_times(np.array([3.0, 7.0]), np.array([1.0, 3.0]), SimulationConfig(seed=99))
    b, _ = simulate_lead_times(np.array([3.0, 7.0]), np.array([1.0, 3.0]), SimulationConfig(seed=99))
    assert np.allclose(a["cvar95"], b["cvar95"])


def test_different_seeds_give_different_draws_but_similar_moments():
    a, _ = simulate_lead_times(np.array([6.0]), np.array([2.0]), SimulationConfig(seed=1, n_draws=20_000))
    b, _ = simulate_lead_times(np.array([6.0]), np.array([2.0]), SimulationConfig(seed=2, n_draws=20_000))
    assert a["cvar95"].iloc[0] != b["cvar95"].iloc[0]
    assert a["mean_lead_days"].iloc[0] == pytest.approx(b["mean_lead_days"].iloc[0], rel=0.05)


def test_empty_input_is_handled():
    metrics, draws = simulate_lead_times(np.array([]), np.array([]))
    assert metrics.empty and draws is None


def test_identical_routes_get_identical_service_metrics(candidates):
    """The dedup cache must not leak different draws to identical routes."""
    enriched = attach_service_metrics(candidates.candidates)
    grouped = enriched.groupby([enriched["lead_days"].round(3), enriched["risk"].round(3)])
    assert (grouped["cvar95_days"].nunique() == 1).all()


def test_service_summary_on_a_real_plan(candidates):
    from chainguard.optimize import solve_milp

    plan = solve_milp(candidates)
    summary = service_level_summary(simulate_plan(plan.per_shipment()))
    assert summary["legs"] == len(plan.per_shipment())
    assert 0.0 <= summary["mean_on_time_probability"] <= 1.0
    assert summary["mean_cvar95_days"] > 0


def test_cvar_substitution_flips_some_decisions(candidates):
    """Documented finding: substituting CVaR into the objective changes choices."""
    from chainguard.scoring import score_candidates

    ranked = risk_adjusted_ranking(score_candidates(candidates.candidates))
    flips = decision_flips(ranked)
    assert flips["shipments"] > 0
    assert 0.0 <= flips["flip_rate"] <= 1.0
    # And it is NOT guaranteed to help — that is precisely the documented result,
    # and why the service target lives in the constraint set instead.
    assert "tail_days_saved" in flips
