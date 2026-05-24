"""Regression guard for the rupee-cost story — fixed probability vector."""

import numpy as np

from retainiq.threshold import find_cost_optimal_threshold


# Hand-picked: two clear churners, four non-churners with spread probabilities
GOLDEN_Y = np.array([1, 1, 0, 0, 0, 0], dtype=int)
GOLDEN_P = np.array([0.95, 0.85, 0.40, 0.20, 0.05, 0.01], dtype=float)


def test_golden_cost_optimal_threshold_below_naive():
    best = find_cost_optimal_threshold(GOLDEN_Y, GOLDEN_P, fn_cost=40_000, fp_cost=500)
    assert best["threshold"] < 0.5
    assert best["total_cost_inr"] <= best["naive_threshold_cost_inr"]
    assert best["savings_vs_naive_inr"] >= 0


def test_golden_cost_optimal_reproducible():
    a = find_cost_optimal_threshold(GOLDEN_Y, GOLDEN_P)
    b = find_cost_optimal_threshold(GOLDEN_Y, GOLDEN_P)
    assert a["threshold"] == b["threshold"]
    assert a["total_cost_inr"] == b["total_cost_inr"]
