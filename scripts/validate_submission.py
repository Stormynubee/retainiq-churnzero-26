"""Pre-upload checks for ChurnZero_RetainIQ_Predictions.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config
from retainiq.submit import validate_submission


def main() -> None:
    sub_path = config.SUBMISSION_CSV
    test_path = config.TEST_CSV
    if not sub_path.is_file():
        raise SystemExit(f"Missing submission: {sub_path}. Run: python -m scripts.predict")
    if not test_path.is_file():
        raise SystemExit(f"Missing test CSV: {test_path}")

    sub = pd.read_csv(sub_path)
    test_df = pd.read_csv(test_path)
    validate_submission(sub, expected_rows=len(test_df))

    if test_df[config.ID_COL].tolist() != sub[config.ID_COL].tolist():
        raise SystemExit("customer_id order does not match test file")

    pos_rate = sub["churn_prediction"].mean()
    print(f"OK: {len(sub)} rows, positive rate {pos_rate:.4f}")
    print("  threshold used at predict time: see data/processed/cost_optimal_threshold.json")


if __name__ == "__main__":
    main()
