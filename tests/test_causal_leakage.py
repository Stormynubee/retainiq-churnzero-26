"""Post-treatment leakage guards and causal feature policy."""

import pandas as pd

from retainiq import config
from retainiq.data import split_features_target
from retainiq.features import fit_transform
from retainiq.uplift import split_treatment


def test_post_treatment_cols_not_in_churn_model_features():
    df = pd.DataFrame(
        {
            config.ID_COL: ["a", "b"],
            config.TARGET_COL: [0, 1],
            config.TREATMENT_COL_PRIMARY: [1, 0],
            config.TREATMENT_COL_COMPLIANCE: [1, 0],
            config.TREATMENT_COL_SECONDARY: [0, 1],
            "gender": ["M", "F"],
            "annual_income": [100.0, 200.0],
        }
    )
    X, _ = split_features_target(df)
    assert config.TREATMENT_COL_COMPLIANCE not in X.columns
    assert config.TREATMENT_COL_SECONDARY not in X.columns


def test_retention_offer_effective_not_engineered():
    df = pd.DataFrame(
        {
            config.ID_COL: ["a", "b", "c"],
            config.TARGET_COL: [0, 1, 0],
            config.TREATMENT_COL_PRIMARY: [1, 0, 0],
            "gender": ["M", "F", "M"],
            "region": ["N", "S", "E"],
            "annual_income": [500_000.0, 400_000.0, 300_000.0],
            "avg_monthly_balance": [50_000.0, 30_000.0, 20_000.0],
            "last_login_days": [10, 120, 5],
            "tenure_months": [24, 6, 48],
        }
    )
    X, _ = split_features_target(df)
    X_fe, _ = fit_transform(X)
    assert "retention_offer_effective" not in X_fe.columns


def test_uplift_splits_drop_post_treatment_compliance():
    X = pd.DataFrame(
        {
            config.TREATMENT_COL_PRIMARY: [1, 0, 1, 0],
            config.TREATMENT_COL_COMPLIANCE: [1, 0, 0, 1],
            config.TREATMENT_COL_SECONDARY: [0, 1, 0, 0],
            "annual_income": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    X_t, _, X_c, _ = split_treatment(X, y)
    for frame in (X_t, X_c):
        assert config.TREATMENT_COL_PRIMARY not in frame.columns
        assert config.TREATMENT_COL_COMPLIANCE not in frame.columns
        assert config.TREATMENT_COL_SECONDARY not in frame.columns
