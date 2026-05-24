"""IO + splits for the ChurnZero datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config


def load_train(path: Path | str = config.TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Train file missing target column {config.TARGET_COL!r}")
    if config.ID_COL not in df.columns:
        raise ValueError(f"Train file missing id column {config.ID_COL!r}")
    return df


def load_test(path: Path | str = config.TEST_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config.TARGET_COL in df.columns:
        raise ValueError("Test file unexpectedly contains the target column")
    if config.ID_COL not in df.columns:
        raise ValueError(f"Test file missing id column {config.ID_COL!r}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop id + target from X. Anything that touches the data later
    fits on this X, so customer_id never enters an encoder."""
    y = df[config.TARGET_COL].astype(int)
    X = df.drop(columns=[config.TARGET_COL] + config.DROP_BEFORE_FEATURES)
    return X, y


def stratified_folds(
    y: pd.Series,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def quick_stats(df: pd.DataFrame, target: str = config.TARGET_COL) -> dict:
    """Tiny EDA summary used in the notebook and on slide 2."""
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
    out = config.DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out
