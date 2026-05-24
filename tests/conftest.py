"""Shared fixtures — synthetic data only; no mocks of business logic."""

import numpy as np
import pandas as pd
import pytest

from retainiq import artifacts, config


@pytest.fixture
def y_true_small() -> np.ndarray:
    return np.array([1, 1, 0, 0, 0, 0], dtype=int)


@pytest.fixture
def y_proba_small() -> np.ndarray:
    return np.array([0.9, 0.8, 0.3, 0.2, 0.05, 0.01], dtype=float)


@pytest.fixture
def minimal_feature_frame() -> pd.DataFrame:
    """Tiny frame with columns needed for ratio + categorical FE."""
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "region": ["North", "South", "East"],
            "annual_income": [500_000.0, 400_000.0, np.nan],
            "avg_monthly_balance": [50_000.0, 30_000.0, 20_000.0],
            "last_login_days": [10, 120, 5],
            "tenure_months": [24, 6, 48],
            "retention_offer_received": [1, 0, 0],
            "retention_offer_accepted": [1, 0, 0],
        }
    )


@pytest.fixture
def fairness_eval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [1, 1, 0, 0, 1, 0],
            "y_pred": [1, 0, 0, 1, 1, 0],
            "gender": ["M", "M", "F", "F", "M", "F"],
            "region": ["N", "S", "N", "S", "N", "S"],
        }
    )


@pytest.fixture
def raw_data_available() -> bool:
    return config.TRAIN_CSV.is_file() and config.TEST_CSV.is_file()


@pytest.fixture
def trained_artifacts_available() -> bool:
    return artifacts.trained_bundle_exists()
