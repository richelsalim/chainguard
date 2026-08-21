"""The objective must be exactly the challenge objective — verified by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chainguard.config import DEFAULT_WEIGHTS, ObjectiveWeights
from chainguard.scoring import minmax_within_group, objective_value, score_candidates


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ObjectiveWeights(lead_time=0.5, cost=0.5, risk=0.5)


def test_default_weights_are_the_challenge_weights():
    assert (DEFAULT_WEIGHTS.lead_time, DEFAULT_WEIGHTS.cost, DEFAULT_WEIGHTS.risk) == (0.40, 0.40, 0.20)


def test_minmax_is_per_group_not_global():
    values = pd.Series([0.0, 10.0, 100.0, 200.0])
    groups = pd.Series(["a", "a", "b", "b"])
    out = minmax_within_group(values, groups)
    # Each group independently spans 0..1; the global max does not leak across.
    assert list(out) == [0.0, 1.0, 0.0, 1.0]


def test_flat_group_normalises_to_zero_not_nan():
    """A shipment with identical options has no worse choice — score 0, never NaN."""
    values = pd.Series([5.0, 5.0, 5.0])
    out = minmax_within_group(values, pd.Series(["x", "x", "x"]))
    assert list(out) == [0.0, 0.0, 0.0]
    assert not out.isna().any()


def test_single_candidate_group_scores_zero(toy_candidates):
    single = toy_candidates[toy_candidates["route_id"] == "R1"]
    scored = score_candidates(single)
    assert scored["score"].iloc[0] == 0.0


def test_score_matches_hand_computation(toy_candidates):
    scored = score_candidates(toy_candidates, DEFAULT_WEIGHTS).set_index("route_id")

    # S1: lead 2/4/6 -> norm 0.0/0.5/1.0 ; cost 10/5/1 -> norm 1.0/(4/9)/0.0
    #     risk 1/2/3 -> norm 0.0/0.5/1.0
    # R1 = .4*0.0 + .4*1.0 + .2*0.0 = 0.40
    # R2 = .4*0.5 + .4*(4/9) + .2*0.5 = 0.2 + 0.177... + 0.1
    # R3 = .4*1.0 + .4*0.0 + .2*1.0 = 0.60
    assert scored.loc["R1", "score"] == pytest.approx(0.40)
    assert scored.loc["R2", "score"] == pytest.approx(0.4 * 0.5 + 0.4 * (4 / 9) + 0.2 * 0.5)
    assert scored.loc["R3", "score"] == pytest.approx(0.60)


def test_identical_candidates_score_identically(toy_candidates):
    scored = score_candidates(toy_candidates).set_index("route_id")
    assert scored.loc["R4", "score"] == pytest.approx(scored.loc["R5", "score"])


def test_scoring_never_mutates_input(toy_candidates):
    before = toy_candidates.copy(deep=True)
    score_candidates(toy_candidates)
    pd.testing.assert_frame_equal(toy_candidates, before)


def test_co2_weight_changes_the_ranking(toy_candidates):
    """Sustainability weights must actually move the answer, or they are decoration.

    The toy fixture's cheapest-on-CO2 route is also its fastest, so re-weighting
    alone cannot flip it. Invert the CO2 ordering to isolate the carbon term.
    """
    frame = toy_candidates.copy()
    frame["co2_kg"] = [300.0, 200.0, 50.0, 120.0, 120.0]  # now R3 is greenest

    default = score_candidates(frame, DEFAULT_WEIGHTS)
    green = score_candidates(frame, ObjectiveWeights(lead_time=0.2, cost=0.2, risk=0.1, co2=0.5))

    best_default = default.loc[default.groupby("shipment_id")["score"].idxmin(), "route_id"].tolist()
    best_green = green.loc[green.groupby("shipment_id")["score"].idxmin(), "route_id"].tolist()
    assert best_default == ["R1", "R4"]
    assert best_green == ["R3", "R4"]


def test_co2_is_ignored_at_zero_weight(toy_candidates):
    """The default objective must be blind to CO2, however extreme the values."""
    frame = toy_candidates.copy()
    baseline = score_candidates(frame, DEFAULT_WEIGHTS)["score"].tolist()
    frame["co2_kg"] = [9_999.0, 1.0, 5_000.0, 42.0, 42.0]
    assert score_candidates(frame, DEFAULT_WEIGHTS)["score"].tolist() == baseline


def test_scores_are_bounded_in_unit_interval(candidates):
    scored = score_candidates(candidates.candidates)
    assert scored["score"].min() >= -1e-9
    assert scored["score"].max() <= 1.0 + 1e-9


def test_each_attribute_has_a_zero_and_a_one_within_every_shipment(candidates):
    """Min-max is applied per attribute, per shipment.

    Note what this does *not* imply: that some route scores exactly 0. That would
    require one route to be simultaneously best on lead time, cost and risk, and
    real candidate pools rarely contain such a dominating option. The blended
    minimum is strictly positive whenever the attributes disagree — which is the
    normal case, and the reason the problem is interesting at all.
    """
    scored = score_candidates(candidates.candidates)
    multi = scored.groupby("shipment_id").filter(lambda g: len(g) > 1)
    for column in ("norm_lead_time", "norm_cost", "norm_risk"):
        grouped = multi.groupby("shipment_id")[column]
        assert np.allclose(grouped.min().to_numpy(), 0.0, atol=1e-9)
        assert np.allclose(grouped.max().to_numpy(), 1.0, atol=1e-9)


def test_blended_minimum_is_zero_only_when_one_route_dominates(toy_candidates):
    dominating = toy_candidates[toy_candidates["shipment_id"] == "S1"].copy()
    # Make R1 best on every attribute at once.
    dominating[["lead_days", "cost_per_kg", "risk"]] = [
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 3.0],
        [9.0, 9.0, 5.0],
    ]
    scored = score_candidates(dominating)
    assert scored["score"].min() == pytest.approx(0.0)


def test_objective_value_of_empty_plan_is_nan():
    assert np.isnan(objective_value(pd.DataFrame(columns=["score"])))
