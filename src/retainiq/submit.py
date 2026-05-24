"""Write the submission CSV exactly per the official spec.

Spec:
    File: ChurnZero_<TeamName>_Predictions.csv
    Columns: customer_id, churn_prediction (0/1), churn_probability (float, 0-1)
    Rows: exactly 2,026 (all test customer_ids, no nulls)

Any deviation = auto-rejection. We assert before writing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def build_submission(
    test_ids: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build the 3-column dataframe that will be written to disk."""
    y_proba = np.clip(np.asarray(y_proba), 0.0, 1.0)
    y_pred = (y_proba >= threshold).astype(int)

    df = pd.DataFrame(
        {
            config.ID_COL: test_ids.values,
            "churn_prediction": y_pred,
            "churn_probability": np.round(y_proba, 6),
        }
    )
    return df


def validate_submission(df: pd.DataFrame, expected_rows: int = 2026) -> None:
    """Hard assertions before writing."""
    assert list(df.columns) == [
        config.ID_COL,
        "churn_prediction",
        "churn_probability",
    ], f"Wrong column order/names: {list(df.columns)!r}"

    assert len(df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(df)}"
    )

    assert df.isna().sum().sum() == 0, "Submission has nulls — fix before writing"

    assert set(df["churn_prediction"].unique()).issubset({0, 1}), (
        f"churn_prediction must be 0 or 1, got {df['churn_prediction'].unique()}"
    )

    assert df["churn_probability"].between(0.0, 1.0).all(), (
        "churn_probability must be in [0, 1]"
    )

    # Defensive: customer_ids should be unique
    assert df[config.ID_COL].is_unique, "Duplicate customer_id in submission"


def write_submission(
    df: pd.DataFrame,
    path: Path | str = config.SUBMISSION_CSV,
) -> Path:
    """Validate and write."""
    validate_submission(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
