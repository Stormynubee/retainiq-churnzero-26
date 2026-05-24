"""Load CSVs and build stratified folds."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config


def load_train(path: Path | str = config.TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"missing {config.TARGET_COL}")
    if config.ID_COL not in df.columns:
        raise ValueError(f"missing {config.ID_COL}")
    return df


def load_test(path: Path | str = config.TEST_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config.TARGET_COL in df.columns:
        raise ValueError("test file should not have churn column")
    if config.ID_COL not in df.columns:
        raise ValueError(f"missing {config.ID_COL}")
    return df


def drop_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and post-treatment cols; safe when test schema omits optional columns."""
    return df.drop(columns=config.DROP_BEFORE_FEATURES, errors="ignore")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[config.TARGET_COL].astype(int)
    drop_cols = [config.TARGET_COL] + [
        c for c in config.DROP_BEFORE_FEATURES if c in df.columns
    ]
    X = df.drop(columns=drop_cols)
    return X, y


def stratified_folds(
    y: pd.Series,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def quick_stats(df: pd.DataFrame, target: str = config.TARGET_COL) -> dict:
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
