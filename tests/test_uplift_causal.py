"""IPTW + propensity trimming for observational uplift."""

import numpy as np
import pandas as pd

from retainiq import config
from retainiq.uplift import (
    estimate_propensity,
    estimate_propensity_raw,
    fit_t_learner,
    iptw_weights,
    propensity_overlap_mask,
    propensity_trim_summary,
)


def _synthetic_uplift_frame(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=n)
    t = (x1 + rng.normal(0, 0.5, n) > 0).astype(int)
    y = ((x1 < 0) & (t == 0)).astype(int) | ((x1 > 0.5) & (t == 1)).astype(int)
    y = rng.integers(0, 2, n)  # ensure both classes
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.normal(size=n),
            config.TREATMENT_COL_PRIMARY: t,
        }
    )
    return X, pd.Series(y)


def test_iptw_weights_positive():
    t = np.array([0, 1, 1, 0])
    e = np.array([0.4, 0.6, 0.7, 0.3])
    w = iptw_weights(t, e)
    assert (w > 0).all()


def test_propensity_clipped_to_bounds():
    X, _ = _synthetic_uplift_frame(80)
    e = estimate_propensity(X)
    assert e.min() >= config.PROPENSITY_MIN
    assert e.max() <= config.PROPENSITY_MAX


def test_overlap_mask_excludes_extreme_raw_scores():
    e_raw = np.array([0.01, 0.5, 0.99, 0.2])
    mask = propensity_overlap_mask(e_raw)
    assert mask.tolist() == [False, True, False, True]


def test_fit_t_learner_returns_models_and_summary():
    X, y = _synthetic_uplift_frame(150)
    out = fit_t_learner(X, y)
    assert "mu_1" in out and "mu_0" in out
    summary = out["propensity_summary"]
    assert summary["n_after_trim"] > 0
    assert summary["n_after_trim"] <= summary["n_total"]


def test_propensity_trim_summary_fields():
    X, _ = _synthetic_uplift_frame(100)
    summary = propensity_trim_summary(X)
    assert "n_trimmed" in summary
    assert "mean_propensity_raw" in summary


def test_estimate_propensity_raw_pair():
    X, _ = _synthetic_uplift_frame(60)
    raw, clipped = estimate_propensity_raw(X)
    assert len(raw) == len(clipped) == len(X)
    assert (clipped >= config.PROPENSITY_MIN).all()
