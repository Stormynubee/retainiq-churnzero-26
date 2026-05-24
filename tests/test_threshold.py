"""Cost threshold logic — pure numpy, no models."""

import numpy as np

from retainiq import config
from retainiq.threshold import (
    cost_curve,
    expected_business_cost,
    find_cost_optimal_threshold,
)


def test_expected_business_cost_counts_fn_and_fp():
  y_true = np.array([1, 1, 0, 0])
  y_pred = np.array([0, 1, 1, 0])
  # 1 FN @ 40k + 1 FP @ 500
  assert expected_business_cost(y_true, y_pred, fn_cost=40_000, fp_cost=500) == 40_500


def test_zero_cost_when_perfect_predictions():
  y = np.array([1, 0, 1, 0])
  assert expected_business_cost(y, y) == 0.0


def test_find_cost_optimal_prefers_catching_churners_when_fn_expensive(
  y_true_small, y_proba_small
):
  best = find_cost_optimal_threshold(y_true_small, y_proba_small, fn_cost=40_000, fp_cost=500)
  assert best["threshold"] < 0.5
  assert best["total_cost_inr"] <= best["naive_threshold_cost_inr"]
  assert best["savings_vs_naive_inr"] >= 0


def test_theoretical_threshold_matches_config():
  assert abs(config.THEORETICAL_OPTIMAL_THRESHOLD - 500 / 40_500) < 1e-9


def test_cost_curve_monotonic_threshold_grid(y_true_small, y_proba_small):
  curve = cost_curve(
    y_true_small,
    y_proba_small,
    thresholds=np.array([0.0, 0.5, 1.0]),
    fn_cost=40_000,
    fp_cost=500,
  )
  assert len(curve) == 3
  assert curve["threshold"].tolist() == [0.0, 0.5, 1.0]
  # t=1.0 predicts nobody churns -> 2 FN
  row_high = curve.loc[curve["threshold"] == 1.0].iloc[0]
  assert row_high["fn"] == 2
  assert row_high["total_cost_inr"] == 80_000
