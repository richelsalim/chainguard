"""Monte Carlo lead-time simulation — turning a risk *score* into a service level.

``RiskScore`` is an ordinal number between 0 and 5. It ranks routes, which is
useful, but it does not answer any question a planner can act on. "Route A has
risk 3.2" has no operational meaning. "Route A delivers on time 78% of the time,
and in the worst 5% of weeks it is 6.1 days late" does.

This module closes that gap by treating lead time as a random variable instead of
a number.

Model
-----
For a route with deterministic lead time :math:`\\mu` and risk score :math:`r`:

* **Baseline variability** — :math:`T_0 \\sim \\text{Gamma}(k, \\theta)` with
  :math:`\\mathbb{E}[T_0]=\\mu` and coefficient of variation
  :math:`c(r) = c_0 + c_1 r`. Gamma is the natural choice: strictly positive
  (transit time cannot be negative), right-skewed (delays have a long tail while
  early arrivals do not), and closed under addition, so summing legs of a
  multi-leg path stays inside the family.
* **Disruption events** — a Bernoulli trial with
  :math:`p(r) = p_0 + p_1 r` fires an additional
  :math:`\\text{Exp}(\\lambda)` delay. This is what produces the heavy tail that
  a pure Gamma understates, and it is where the risk score does its real work.

Outputs
-------
* ``p50`` / ``p90`` / ``p95`` lead time — the planning numbers.
* ``on_time_probability`` against the deterministic promise plus an SLA slack.
* ``expected_delay_days`` — mean overrun beyond the promise.
* **CVaR₉₅** — the mean lead time *conditional on being in the worst 5% of
  outcomes*. This is the number that separates a route which is usually fine and
  occasionally catastrophic from one that is reliably mediocre. Expected value
  cannot distinguish them; CVaR can, and it is coherent as a risk measure in a
  way that a raw percentile is not.

Risk-aware selection
--------------------
:func:`risk_adjusted_ranking` re-scores a candidate set using simulated CVaR in
place of the deterministic lead time. Comparing the two rankings shows how many
routing decisions actually *flip* once uncertainty is priced in — which is the
entire argument for simulating in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_SIMULATION, DEFAULT_WEIGHTS, ObjectiveWeights, SimulationConfig


@dataclass
class SimulationResult:
    """Per-route simulated service metrics, plus the raw draws for charting."""

    metrics: pd.DataFrame
    draws: np.ndarray | None = None
    config: SimulationConfig = DEFAULT_SIMULATION

    def worst(self, n: int = 10, by: str = "cvar95") -> pd.DataFrame:
        return self.metrics.sort_values(by, ascending=False).head(n).reset_index(drop=True)


def _gamma_params(mean: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (mean, coefficient of variation) to Gamma (shape, scale)."""
    cv = np.clip(cv, 1e-6, None)
    shape = 1.0 / cv**2
    scale = np.clip(mean, 1e-9, None) / shape
    return shape, scale


def simulate_lead_times(
    lead_days: np.ndarray,
    risk: np.ndarray,
    config: SimulationConfig = DEFAULT_SIMULATION,
    keep_draws: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Draw ``config.n_draws`` lead times for each (lead_days, risk) pair.

    Returns a metrics frame aligned to the inputs, and optionally the raw
    ``(n_routes, n_draws)`` sample matrix.
    """
    rng = np.random.default_rng(config.seed)
    mu = np.asarray(lead_days, dtype=float)
    r = np.asarray(risk, dtype=float)
    n, m = mu.shape[0], config.n_draws

    if n == 0:
        return pd.DataFrame(), None

    cv = config.cv_base + config.cv_per_risk_point * r
    shape, scale = _gamma_params(mu, cv)

    base = rng.gamma(shape[:, None], scale[:, None], size=(n, m))

    p_disrupt = np.clip(config.disruption_base + config.disruption_per_risk_point * r, 0.0, 0.95)
    fired = rng.random((n, m)) < p_disrupt[:, None]
    shock = rng.exponential(config.disruption_delay_mean_days, size=(n, m))
    draws = base + fired * shock

    promise = mu + config.sla_slack_days
    overrun = np.clip(draws - promise[:, None], 0.0, None)

    # CVaR: mean of the worst (1-alpha) tail. Computed from the sorted sample,
    # which is the standard non-parametric estimator and needs no distributional
    # assumption beyond the draws themselves.
    tail_start = int(np.floor(config.cvar_alpha * m))
    tail_start = min(max(tail_start, 0), m - 1)
    sorted_draws = np.sort(draws, axis=1)
    cvar = sorted_draws[:, tail_start:].mean(axis=1)

    metrics = pd.DataFrame(
        {
            "deterministic_lead_days": mu,
            "risk": r,
            "mean_lead_days": draws.mean(axis=1),
            "p50": np.percentile(draws, 50, axis=1),
            "p90": np.percentile(draws, 90, axis=1),
            "p95": np.percentile(draws, 95, axis=1),
            "std_lead_days": draws.std(axis=1, ddof=1),
            "on_time_probability": (draws <= promise[:, None]).mean(axis=1),
            "expected_delay_days": overrun.mean(axis=1),
            "cvar95": cvar,
            "disruption_probability": p_disrupt,
        }
    )
    metrics["tail_penalty_days"] = metrics["cvar95"] - metrics["p50"]
    return metrics, (draws if keep_draws else None)


def simulate_plan(
    plan_assignments: pd.DataFrame,
    config: SimulationConfig = DEFAULT_SIMULATION,
    keep_draws: bool = False,
) -> SimulationResult:
    """Simulate every leg of a completed plan."""
    if plan_assignments.empty:
        return SimulationResult(metrics=pd.DataFrame(), config=config)

    metrics, draws = simulate_lead_times(
        plan_assignments["lead_days"].to_numpy(),
        plan_assignments["risk"].to_numpy(),
        config=config,
        keep_draws=keep_draws,
    )
    metrics.insert(0, "shipment_id", plan_assignments["shipment_id"].to_numpy())
    if "route_id" in plan_assignments:
        metrics.insert(1, "route_id", plan_assignments["route_id"].to_numpy())
    if "mode" in plan_assignments:
        metrics.insert(2, "mode", plan_assignments["mode"].to_numpy())
    return SimulationResult(metrics=metrics, draws=draws, config=config)


def service_level_summary(result: SimulationResult) -> dict:
    """Plan-level service metrics — the numbers that belong on a slide."""
    m = result.metrics
    if m.empty:
        return {
            "legs": 0,
            "mean_on_time_probability": float("nan"),
            "legs_below_90pct_otd": 0,
            "legs_below_75pct_otd": 0,
            "mean_p90_lead_days": float("nan"),
            "mean_cvar95_days": float("nan"),
            "mean_expected_delay_days": float("nan"),
            "plan_all_on_time_probability": float("nan"),
        }
    return {
        "legs": int(len(m)),
        "mean_on_time_probability": float(m["on_time_probability"].mean()),
        "legs_below_90pct_otd": int((m["on_time_probability"] < 0.90).sum()),
        "legs_below_75pct_otd": int((m["on_time_probability"] < 0.75).sum()),
        "mean_p90_lead_days": float(m["p90"].mean()),
        "mean_cvar95_days": float(m["cvar95"].mean()),
        "mean_expected_delay_days": float(m["expected_delay_days"].mean()),
        # Probability that no leg in the plan misses — the product of per-leg
        # on-time probabilities under an independence assumption. It is
        # deliberately sobering: independent 95% legs compound fast.
        "plan_all_on_time_probability": float(np.exp(np.log(
            m["on_time_probability"].clip(lower=1e-9)).sum())),
    }


def attach_service_metrics(
    candidates: pd.DataFrame,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> pd.DataFrame:
    """Add simulated ``cvar95_days`` and ``on_time_probability`` to a candidate table.

    Distinct ``(lead_days, risk)`` pairs are simulated once and reused. A
    candidate set of 100k rows typically contains only a few hundred unique
    pairs, so this is orders of magnitude cheaper than simulating row-wise, and
    it guarantees identical routes receive identical service metrics.
    """
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()
    key = pd.MultiIndex.from_arrays([df["lead_days"].round(3), df["risk"].round(3)])
    unique = key.unique()
    lead_u = np.array([k[0] for k in unique], dtype=float)
    risk_u = np.array([k[1] for k in unique], dtype=float)
    metrics, _ = simulate_lead_times(lead_u, risk_u, config=config)

    cvar_lookup = dict(zip(unique, metrics["cvar95"].to_numpy(), strict=False))
    otd_lookup = dict(zip(unique, metrics["on_time_probability"].to_numpy(), strict=False))
    df["cvar95_days"] = [cvar_lookup[k] for k in key]
    df["on_time_probability"] = [otd_lookup[k] for k in key]
    return df


def risk_adjusted_ranking(
    scored_candidates: pd.DataFrame,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> pd.DataFrame:
    """Re-score candidates on simulated CVaR in place of deterministic lead time.

    .. warning::

        This is a **diagnostic, not a recommended policy**, and the distinction
        is the most interesting result in this module.

        The intuition — "swap the lead-time term for a tail-risk term and the
        plan gets safer" — does not survive contact with the arithmetic. Both
        terms are min-max normalised inside each shipment's candidate pool, and
        CVaR compresses differently from mean lead time at the low end. A route
        that is second-best on lead time can be *relatively* closer to the front
        on normalised CVaR, and the 40% cost term then tips the choice to it. The
        result is that some decisions flip toward routes with a **worse** tail.

        Measured on the bundled synthetic network, this substitution flips ~6% of
        routing decisions and moves mean CVaR₉₅ in the wrong direction.

        The lesson generalises: a service level is a **constraint, not a
        preference**. If a plan must deliver on time 90% of the time, that
        belongs in the feasible region — see ``min_on_time_probability`` in
        :func:`chainguard.optimize.milp.solve`, which enforces it directly and
        prices the guarantee — not blended into an objective where a cost term
        can outvote it.

    :func:`decision_flips` quantifies the effect so the claim above is
    reproducible rather than asserted.
    """
    if scored_candidates.empty:
        return scored_candidates.copy()

    df = attach_service_metrics(scored_candidates, config)

    from .scoring import minmax_within_group  # local import avoids a cycle

    group = df["shipment_id"]
    df["norm_cvar"] = minmax_within_group(df["cvar95_days"], group)
    df["risk_adjusted_score"] = (
        weights.lead_time * df["norm_cvar"]
        + weights.cost * df["norm_cost"]
        + weights.risk * df["norm_risk"]
        + weights.co2 * df.get("norm_co2", pd.Series(0.0, index=df.index))
    )
    return df


def decision_flips(risk_adjusted: pd.DataFrame) -> dict:
    """How many shipments get routed differently when CVaR replaces lead time?

    Reports ``tail_days_saved`` as *deterministic minus risk-adjusted* mean
    CVaR₉₅, so a **negative** value means the substitution made the tail worse.
    On the bundled synthetic network it is negative — see the warning on
    :func:`risk_adjusted_ranking` for why, and why the fix is a chance
    constraint rather than a reweighted objective.
    """
    if risk_adjusted.empty:
        return {"shipments": 0, "flipped": 0, "flip_rate": float("nan")}

    det = (
        risk_adjusted.sort_values(["shipment_id", "score", "route_id"], kind="mergesort")
        .groupby("shipment_id")["route_id"]
        .first()
    )
    rob = (
        risk_adjusted.sort_values(["shipment_id", "risk_adjusted_score", "route_id"], kind="mergesort")
        .groupby("shipment_id")["route_id"]
        .first()
    )
    joined = pd.DataFrame({"deterministic": det, "risk_adjusted": rob}).dropna()
    flipped = (joined["deterministic"] != joined["risk_adjusted"]).sum()

    # What the flip buys, in days of tail exposure avoided.
    cvar = risk_adjusted.set_index(["shipment_id", "route_id"])["cvar95_days"]
    det_cvar = cvar.reindex(list(zip(joined.index, joined["deterministic"], strict=False))).to_numpy()
    rob_cvar = cvar.reindex(list(zip(joined.index, joined["risk_adjusted"], strict=False))).to_numpy()

    return {
        "shipments": int(len(joined)),
        "flipped": int(flipped),
        "flip_rate": float(flipped / len(joined)) if len(joined) else float("nan"),
        "mean_cvar_deterministic": float(np.nanmean(det_cvar)),
        "mean_cvar_risk_adjusted": float(np.nanmean(rob_cvar)),
        "tail_days_saved": float(np.nanmean(det_cvar) - np.nanmean(rob_cvar)),
    }
