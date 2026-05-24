"""Rank-average stacking for PR-AUC leaderboard column."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from retainiq.models import rank_average_probabilities, tune_rank_stack_weights


def test_rank_average_in_unit_interval():
    p_lgb = np.array([0.1, 0.5, 0.9, 0.3])
    p_cat = np.array([0.2, 0.4, 0.8, 0.35])
    out = rank_average_probabilities(p_lgb, p_cat)
    assert out.min() > 0.0
    assert out.max() < 1.0


def test_rank_preserves_order_when_one_model_dominates():
    p_lgb = np.linspace(0.01, 0.99, 20)
    p_cat = np.full(20, 0.5)
    out = rank_average_probabilities(p_lgb, p_cat, w_lgb=1.0, w_cat=0.0)
    assert np.all(np.diff(out) > 0)


def test_tune_rank_stack_weights_improves_or_matches_default():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    oof_lgb = np.clip(rng.normal(0.3, 0.2, n) + y * 0.3, 0, 1)
    oof_cat = np.clip(oof_lgb + rng.normal(0, 0.05, n), 0, 1)
    best = tune_rank_stack_weights(oof_lgb, oof_cat, y)
    default_pr = average_precision_score(
        y, rank_average_probabilities(oof_lgb, oof_cat, 0.5, 0.5)
    )
    assert best["oof_pr_auc"] >= default_pr - 1e-9
    assert abs(best["w_lgb"] + best["w_cat"] - 1.0) < 1e-6


def test_rank_and_calibrated_streams_can_diverge_for_binary():
    from retainiq.submit import build_submission
    import pandas as pd

    ids = pd.Series([1, 2, 3])
    rank = np.array([0.1, 0.9, 0.5])
    cal = np.array([0.8, 0.2, 0.6])
    sub = build_submission(ids, rank, 0.5, y_proba_calibrated=cal)
    assert sub["churn_prediction"].tolist() == [1, 0, 1]
    assert np.allclose(sub["churn_probability"].values, rank)
    assert spearmanr(sub["churn_probability"], rank).correlation > 0.99
