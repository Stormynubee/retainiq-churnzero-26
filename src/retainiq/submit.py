"""Build and validate the competition CSV."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def build_submission(
    test_ids: pd.Series,
    y_proba_rank: np.ndarray,
    threshold: float,
    *,
    y_proba_calibrated: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rank probabilities for PR-AUC column; calibrated stream for cost-optimal binary preds."""
    p_rank = np.clip(np.asarray(y_proba_rank, dtype=float), 0.0, 1.0)
    p_cal = np.clip(
        np.asarray(y_proba_calibrated if y_proba_calibrated is not None else p_rank, dtype=float),
        0.0,
        1.0,
    )
    y_pred = (p_cal >= threshold).astype(int)
    return pd.DataFrame(
        {
            config.ID_COL: test_ids.values,
            "churn_prediction": y_pred,
            "churn_probability": np.round(p_rank, 6),
        }
    )


def validate_submission(df: pd.DataFrame, expected_rows: int = 2026) -> None:
    expected_cols = [config.ID_COL, "churn_prediction", "churn_probability"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"wrong columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")
    if df.isna().sum().sum() > 0:
        raise ValueError("nulls in submission")
    bad_preds = set(df["churn_prediction"].unique()) - {0, 1}
    if bad_preds:
        raise ValueError(f"churn_prediction must be 0/1, saw {bad_preds}")
    if not df["churn_probability"].between(0.0, 1.0).all():
        raise ValueError("probabilities out of [0,1]")
    if not df[config.ID_COL].is_unique:
        raise ValueError("duplicate customer_id")


def write_submission(
    df: pd.DataFrame,
    path: Path | str = config.SUBMISSION_CSV,
) -> Path:
    validate_submission(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
