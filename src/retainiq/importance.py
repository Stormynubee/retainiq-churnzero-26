"""LightGBM gain table from a trained stacked bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import artifacts
from .models import StackedBundle


def compute_importances(bundle: StackedBundle, X_fe: pd.DataFrame) -> pd.DataFrame:
    importance = np.zeros(X_fe.shape[1], dtype=float)
    for model in bundle.lgb_models:
        importance += model.feature_importance(importance_type="gain")
    importance /= max(len(bundle.lgb_models), 1)

    imp_df = pd.DataFrame({"feature": X_fe.columns, "lgb_gain": importance}).sort_values(
        "lgb_gain", ascending=False
    )
    imp_df["lgb_gain_pct"] = 100.0 * imp_df["lgb_gain"] / imp_df["lgb_gain"].sum()
    return imp_df


def export_feature_importances(bundle: StackedBundle, X_fe: pd.DataFrame) -> pd.DataFrame:
    imp_df = compute_importances(bundle, X_fe)
    imp_df.to_csv(artifacts.feature_importances_path(), index=False)
    return imp_df
