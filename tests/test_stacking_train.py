"""Stacker must be fit on OOF base predictions, not ensemble training averages."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from retainiq.models import ensemble_base_predictions, fit_meta, fit_meta_oof


def test_meta_on_oof_differs_from_meta_on_ensemble_train_averages():
    rng = np.random.default_rng(7)
    n = 60
    y = pd.Series(rng.integers(0, 2, n))
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    from retainiq.models import train_lightgbm_oof

    oof_lgb, lgb_models = train_lightgbm_oof(
        X, y, splitter, params={"num_boost_round": 30, "early_stopping": 10}
    )
    oof_cat = oof_lgb * 0.9 + rng.normal(0, 0.02, n)
    avg_lgb, avg_cat = ensemble_base_predictions(X, lgb_models, [], [])

    meta_oof = fit_meta(oof_lgb, oof_cat, y)
    meta_avg = fit_meta(avg_lgb, avg_cat, y)

    Z_oof = np.column_stack([oof_lgb, oof_cat])
    Z_avg = np.column_stack([avg_lgb, avg_cat])
    assert not np.allclose(
        meta_oof.predict_proba(Z_oof)[:, 1],
        meta_avg.predict_proba(Z_avg)[:, 1],
        atol=1e-5,
    )


def test_oof_meta_is_out_of_fold():
    rng = np.random.default_rng(1)
    n = 80
    y = pd.Series(rng.integers(0, 2, n))
    oof_lgb = rng.uniform(0.05, 0.95, n)
    oof_cat = rng.uniform(0.05, 0.95, n)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

    oof_meta = fit_meta_oof(oof_lgb, oof_cat, y, splitter)
    meta = fit_meta(oof_lgb, oof_cat, y)
    in_sample = meta.predict_proba(np.column_stack([oof_lgb, oof_cat]))[:, 1]
    assert not np.allclose(oof_meta, in_sample, atol=1e-6)
