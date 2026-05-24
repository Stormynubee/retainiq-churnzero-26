"""Stacking alignment, OOF meta calibration, and Platt scaling."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from retainiq.models import (
    calibrate_platt,
    ensemble_base_predictions,
    fit_meta,
    fit_meta_oof,
    predict_platt,
    stacked_oof,
    train_lightgbm_oof,
)


def test_fit_meta_oof_differs_from_insample_stacked():
    rng = np.random.default_rng(42)
    n = 80
    y = pd.Series(rng.integers(0, 2, n))
    oof_lgb = rng.uniform(0.05, 0.95, n)
    oof_cat = rng.uniform(0.05, 0.95, n)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_meta = fit_meta_oof(oof_lgb, oof_cat, y, splitter)
    meta = fit_meta(oof_lgb, oof_cat, y)
    in_sample = stacked_oof(oof_lgb, oof_cat, meta)

    assert not np.allclose(oof_meta, in_sample, atol=1e-6)


def test_platt_calibration_is_logistic_regression():
    y = pd.Series([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
    calibrator = calibrate_platt(p, y)
    assert isinstance(calibrator, LogisticRegression)


def test_platt_produces_many_unique_calibrated_values():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.001, 0.99, n)
    y = pd.Series((p + rng.normal(0, 0.08, n) > 0.5).astype(int))
    calibrator = calibrate_platt(p, y)
    calibrated = predict_platt(calibrator, p)
    assert calibrated.min() > 0.0
    assert calibrated.max() < 1.0
    assert len(np.unique(np.round(calibrated, 4))) > 15


def test_ensemble_base_predictions_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=20), "f2": rng.normal(size=20)})
    y = pd.Series(rng.integers(0, 2, 20))
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    _, lgb_models = train_lightgbm_oof(
        X,
        y,
        splitter,
        params={"num_boost_round": 20, "early_stopping": 5},
    )
    avg_lgb, avg_cat = ensemble_base_predictions(X, lgb_models, [], [])
    assert avg_lgb.shape == (20,)
    assert avg_cat.shape == (20,)
