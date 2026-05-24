"""Fairness tables."""

from retainiq.fairness import fairness_report


def test_fairness_report_returns_per_group_and_summary(fairness_eval_frame):
  report = fairness_report(fairness_eval_frame)
  assert not report["per_group"].empty
  assert "gender" in report["summary"]["attribute"].values
  assert "demographic_parity_diff" in report["summary"].columns


def test_fairness_skips_missing_sensitive_columns(fairness_eval_frame):
  df = fairness_eval_frame.drop(columns=["region"])
  report = fairness_report(df, sensitive_cols=["gender", "region"])
  assert set(report["summary"]["attribute"]) == {"gender"}
