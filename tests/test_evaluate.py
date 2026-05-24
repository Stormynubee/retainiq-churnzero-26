"""Metric bundle at a threshold."""

import numpy as np

from retainiq.evaluate import evaluate_at_threshold, summarize


def test_evaluate_at_threshold_perfect_classifier():
  y = np.array([1, 1, 0, 0])
  p = np.array([0.99, 0.95, 0.05, 0.01])
  m = evaluate_at_threshold(y, p, threshold=0.5)
  assert m["pr_auc"] == 1.0
  assert m["recall"] == 1.0
  assert m["fn"] == 0
  assert m["fp"] == 0
  assert m["total_cost_inr"] == 0.0


def test_summarize_includes_pr_auc_and_cost():
  m = evaluate_at_threshold(
    np.array([1, 0]), np.array([0.8, 0.2]), threshold=0.5
  )
  text = summarize(m)
  assert "PR-AUC" in text
  assert "cost INR" in text
