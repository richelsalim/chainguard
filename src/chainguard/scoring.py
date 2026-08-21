"""The challenge objective, implemented once and used by every optimiser.

.. math::

    s_{ij} \\;=\\; w_L \\,\\hat L_{ij} \\;+\\; w_C \\,\\hat C_{ij}
                \\;+\\; w_R \\,\\hat R_{ij} \\;+\\; w_G \\,\\hat G_{ij}

for shipment :math:`i` and candidate route :math:`j`, where each :math:`\\hat{\\cdot}`
is the min-max normalisation of that attribute **within shipment i's own feasible
candidate pool**:

.. math::

    \\hat X_{ij} = \\frac{X_{ij} - \\min_k X_{ik}}{\\max_k X_{ik} - \\min_k X_{ik}}

Why normalise per shipment rather than globally? Because the score has to answer
"how good is this option *for this shipment*". A shipment whose only routes are
all slow should not be penalised for the network's geography — it should be
judged on whether the plan picked the best of what it actually had. It also
makes 0.0 mean "best available" and 1.0 "worst available" for every shipment,
which is what makes averaging scores across a heterogeneous shipment population
meaningful at all.

Degenerate pools (one candidate, or all candidates identical on an attribute)
normalise to 0.0 rather than NaN: with nothing to choose between, no option is
worse than another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHTS, ObjectiveWeights

ATTRIBUTES: dict[str, str] = {
    "lead_time": "lead_days",
    "cost": "cost_per_kg",
    "risk": "risk",
    "co2": "co2_kg",
}


def minmax_within_group(values: pd.Series, group: pd.Series) -> pd.Series:
    """Min-max normalise ``values`` inside each ``group``; flat groups -> 0.0."""
    grouped = values.groupby(group)
    lo = grouped.transform("min")
    hi = grouped.transform("max")
    span = hi - lo
    out = np.where(span > 0, (values - lo) / span.replace(0, np.nan), 0.0)
    return pd.Series(np.nan_to_num(out, nan=0.0), index=values.index)


def score_candidates(
    candidates: pd.DataFrame,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    group_col: str = "shipment_id",
) -> pd.DataFrame:
    """Add ``norm_*`` columns and the weighted ``score`` to a candidate table.

    Returns a copy; the input is never mutated.
    """
    if candidates.empty:
        out = candidates.copy()
        for name in ATTRIBUTES:
            out[f"norm_{name}"] = pd.Series(dtype=float)
        out["score"] = pd.Series(dtype=float)
        return out

    out = candidates.copy()
    group = out[group_col]
    score = pd.Series(0.0, index=out.index)

    for name, column in ATTRIBUTES.items():
        weight = getattr(weights, name)
        if column not in out.columns:
            out[f"norm_{name}"] = 0.0
            continue
        normalised = minmax_within_group(out[column].astype(float), group)
        out[f"norm_{name}"] = normalised
        if weight:
            score = score + weight * normalised

    out["score"] = score
    return out


def objective_value(plan: pd.DataFrame) -> float:
    """Mean weighted score of a plan — the number the challenge ranks on."""
    if plan.empty:
        return float("nan")
    return float(plan["score"].mean())


def summarise_plan(plan: pd.DataFrame, n_in_scope: int | None = None) -> dict:
    """Headline metrics for a completed plan.

    ``n_in_scope`` lets the summary report coverage honestly when some shipments
    could not be placed at all: a plan that solves 80% of shipments beautifully
    is not the same as one that solves 100% adequately, and the mean score alone
    cannot tell them apart.
    """
    if plan.empty:
        return {
            "shipments": 0,
            "in_scope": n_in_scope or 0,
            "coverage": 0.0,
            "mean_score": float("nan"),
            "median_score": float("nan"),
            "p90_score": float("nan"),
            "std_score": float("nan"),
            "best_score": float("nan"),
            "worst_score": float("nan"),
            "total_cost_eur": 0.0,
            "mean_cost_per_kg": float("nan"),
            "mean_lead_days": float("nan"),
            "total_co2_kg": 0.0,
            "mean_risk": float("nan"),
            "primary_share": float("nan"),
        }

    n_scope = n_in_scope if n_in_scope is not None else len(plan)
    return {
        "shipments": int(len(plan)),
        "in_scope": int(n_scope),
        "coverage": round(len(plan) / n_scope, 4) if n_scope else 0.0,
        "mean_score": float(plan["score"].mean()),
        "median_score": float(plan["score"].median()),
        "p90_score": float(plan["score"].quantile(0.90)),
        "std_score": float(plan["score"].std(ddof=1)) if len(plan) > 1 else 0.0,
        "best_score": float(plan["score"].min()),
        "worst_score": float(plan["score"].max()),
        "total_cost_eur": float(plan["total_cost_eur"].sum()),
        "mean_cost_per_kg": float(plan["cost_per_kg"].mean()),
        "mean_lead_days": float(plan["lead_days"].mean()),
        "total_co2_kg": float(plan["co2_kg"].sum()),
        "mean_risk": float(plan["risk"].mean()),
        "primary_share": float(plan["is_primary"].mean()) if "is_primary" in plan else float("nan"),
    }
