"""Submission file builder.

The official spec wants a CSV with exactly these columns:
    customer_id, churn_prediction (0/1), churn_probability (float in [0, 1])
and exactly 2,026 rows. Anything else = auto-rejection, so we sanity-check
before writing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def build_submission(
    test_ids: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_proba = np.clip(np.asarray(y_proba), 0.0, 1.0)
    y_pred = (y_proba >= threshold).astype(int)
    return pd.DataFrame({
        config.ID_COL: test_ids.values,
        "churn_prediction": y_pred,
        "churn_probability": np.round(y_proba, 6),
    })


def validate_submission(df: pd.DataFrame, expected_rows: int = 2026) -> None:
    expected_cols = [config.ID_COL, "churn_prediction", "churn_probability"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Wrong columns: got {list(df.columns)!r}, expected {expected_cols!r}")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if df.isna().sum().sum() > 0:
        raise ValueError("Submission has nulls; refusing to write")
    bad_preds = set(df["churn_prediction"].unique()) - {0, 1}
    if bad_preds:
        raise ValueError(f"churn_prediction must be 0/1; saw {bad_preds}")
    if not df["churn_probability"].between(0.0, 1.0).all():
        raise ValueError("churn_probability must lie in [0, 1]")
    if not df[config.ID_COL].is_unique:
        raise ValueError("Duplicate customer_id in submission")


def write_submission(
    df: pd.DataFrame,
    path: Path | str = config.SUBMISSION_CSV,
) -> Path:
    validate_submission(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
