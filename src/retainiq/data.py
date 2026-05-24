"""Load, validate, and split the ChurnZero datasets.

Contracts enforced here:
- Train has TARGET_COL ("churn"); test does not.
- ID_COL is preserved on test for the submission file but never enters features.
- Missingness in `app_rating_given` (and any other numeric col) is captured
  via an explicit *_isna flag before imputation downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_train(path: Path | str = config.TRAIN_CSV) -> pd.DataFrame:
    """Load the training set; assert basic shape and presence of target."""
    df = pd.read_csv(path)
    assert config.TARGET_COL in df.columns, (
        f"Train file is missing target column {config.TARGET_COL!r}"
    )
    assert config.ID_COL in df.columns, (
        f"Train file is missing id column {config.ID_COL!r}"
    )
    return df


def load_test(path: Path | str = config.TEST_CSV) -> pd.DataFrame:
    """Load the test set; assert no target and presence of id."""
    df = pd.read_csv(path)
    assert config.TARGET_COL not in df.columns, (
        "Test file unexpectedly contains the target column."
    )
    assert config.ID_COL in df.columns, (
        f"Test file is missing id column {config.ID_COL!r}"
    )
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with id + target removed from X.

    Note: this is the LAST step before features.py touches the data. We
    drop id explicitly so it cannot leak into any encoder.
    """
    y = df[config.TARGET_COL].astype(int)
    X = df.drop(columns=[config.TARGET_COL] + config.DROP_BEFORE_FEATURES)
    return X, y


def stratified_folds(
    y: pd.Series,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> StratifiedKFold:
    """Return a fitted StratifiedKFold splitter over (X, y)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Sanity / quick stats (for EDA + slide 2)
# ---------------------------------------------------------------------------
def quick_stats(df: pd.DataFrame, target: str = config.TARGET_COL) -> dict:
    """Tiny summary used in the EDA notebook / slide 2."""
    out = {
        "rows": len(df),
        "cols": df.shape[1],
        "n_missing_total": int(df.isna().sum().sum()),
        "cols_with_missing": int((df.isna().sum() > 0).sum()),
    }
    if target in df.columns:
        out["churn_rate"] = float(df[target].mean())
        out["churn_count"] = int(df[target].sum())
    return out


def write_processed(df: pd.DataFrame, name: str) -> Path:
    """Persist a processed dataframe under data/processed/."""
    out = config.DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out
