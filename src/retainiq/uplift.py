"""T-learner uplift on retention_offer_received.

We model P(stay | offer) - P(stay | no offer). Positive CATE = offer helps retention.
Offers weren't randomised in the data — mention that on the deck.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def _drop_uplift_leakage_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Remove treatment assignment and post-treatment compliance from uplift features."""
    drop = [config.TREATMENT_COL_PRIMARY, *config.POST_TREATMENT_COLS]
    return X.drop(columns=[c for c in drop if c in X.columns])


def split_treatment(
    X: pd.DataFrame, y: pd.Series, treatment_col: str = config.TREATMENT_COL_PRIMARY
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if treatment_col not in X.columns:
        raise KeyError(f"no column {treatment_col!r}")

    t = X[treatment_col].astype(int).values
    X_no_t = _drop_uplift_leakage_cols(X)
    treated_mask = t == 1
    return (
        X_no_t.loc[treated_mask].copy(),
        y.loc[treated_mask].copy(),
        X_no_t.loc[~treated_mask].copy(),
        y.loc[~treated_mask].copy(),
    )


def estimate_propensity(X: pd.DataFrame) -> np.ndarray:
    """P(offer | X), clipped to [PROPENSITY_MIN, PROPENSITY_MAX] for IPTW stability."""
    _, e_clipped = estimate_propensity_raw(X)
    return e_clipped


def estimate_propensity_raw(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and clipped propensity scores."""
    if config.TREATMENT_COL_PRIMARY not in X.columns:
        raise KeyError("treatment column missing")
    t = X[config.TREATMENT_COL_PRIMARY].astype(int).values
    X_num = _drop_uplift_leakage_cols(X).select_dtypes(include=[np.number])
    lr = LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)
    lr.fit(X_num, t)
    e_raw = lr.predict_proba(X_num)[:, 1]
    e_clipped = np.clip(e_raw, config.PROPENSITY_MIN, config.PROPENSITY_MAX)
    return e_raw, e_clipped


def propensity_overlap_mask(e_raw: np.ndarray) -> np.ndarray:
    """Rows with common support (exclude extreme propensities before clipping)."""
    return (e_raw >= config.PROPENSITY_MIN) & (e_raw <= config.PROPENSITY_MAX)


def iptw_weights(treatment: np.ndarray, e_clipped: np.ndarray) -> np.ndarray:
    t = np.asarray(treatment).astype(int)
    e = np.clip(np.asarray(e_clipped), config.PROPENSITY_MIN, config.PROPENSITY_MAX)
    return np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))


def propensity_trim_summary(X: pd.DataFrame) -> dict:
    """Counts and mean propensity before/after overlap trim."""
    e_raw, e_clipped = estimate_propensity_raw(X)
    mask = propensity_overlap_mask(e_raw)
    n = len(e_raw)
    n_kept = int(mask.sum())
    return {
        "n_total": n,
        "n_after_trim": n_kept,
        "n_trimmed": n - n_kept,
        "trim_fraction": float(1.0 - n_kept / n) if n else 0.0,
        "mean_propensity_raw": float(e_raw.mean()),
        "mean_propensity_clipped": float(e_clipped[mask].mean()) if n_kept else None,
        "propensity_min": config.PROPENSITY_MIN,
        "propensity_max": config.PROPENSITY_MAX,
    }


def fit_t_learner(X: pd.DataFrame, y: pd.Series) -> dict:
    if config.TREATMENT_COL_PRIMARY not in X.columns:
        raise KeyError(f"no column {config.TREATMENT_COL_PRIMARY!r}")

    e_raw, e_clipped = estimate_propensity_raw(X)
    mask = propensity_overlap_mask(e_raw)
    if not mask.any():
        raise ValueError(
            "no rows in propensity overlap region; widen PROPENSITY_MIN/MAX or check data"
        )

    X_trim = X.loc[mask]
    y_trim = y.loc[mask]
    t_trim = X_trim[config.TREATMENT_COL_PRIMARY].astype(int).values
    w = iptw_weights(t_trim, e_clipped[mask])

    X_t, y_t, X_c, y_c = split_treatment(X_trim, y_trim)
    treated_mask = t_trim == 1
    w_t = w[treated_mask]
    w_c = w[~treated_mask]

    stay_t = 1 - y_t.astype(int)
    stay_c = 1 - y_c.astype(int)

    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        random_state=config.RANDOM_SEED,
        verbose=-1,
    )
    mu_1 = LGBMClassifier(**common).fit(X_t, stay_t, sample_weight=w_t)
    mu_0 = LGBMClassifier(**common).fit(X_c, stay_c, sample_weight=w_c)
    return {
        "mu_1": mu_1,
        "mu_0": mu_0,
        "propensity_summary": propensity_trim_summary(X),
    }


def estimate_cate(model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    X_no_t = _drop_uplift_leakage_cols(X)
    p_stay_t = model_dict["mu_1"].predict_proba(X_no_t)[:, 1]
    p_stay_c = model_dict["mu_0"].predict_proba(X_no_t)[:, 1]
    return p_stay_t - p_stay_c


def segment_customers(cate: np.ndarray, churn_proba: np.ndarray) -> pd.Series:
    cate = np.asarray(cate)
    churn_proba = np.asarray(churn_proba)
    median_p = np.median(churn_proba)
    labels = np.full(len(cate), fill_value="other", dtype=object)
    labels[(cate > 0.05) & (churn_proba > median_p)] = "persuadable"
    labels[(np.abs(cate) <= 0.05) & (churn_proba < 0.10)] = "sure-thing"
    labels[(np.abs(cate) <= 0.05) & (churn_proba >= 0.50)] = "lost-cause"
    labels[cate < -0.02] = "sleeping-dog"
    return pd.Series(labels)


def segmentation_report(cate: np.ndarray, churn_proba: np.ndarray) -> pd.DataFrame:
    seg = segment_customers(cate, churn_proba)
    df = pd.DataFrame({"segment": seg, "cate": cate, "churn_proba": churn_proba})
    return (
        df.groupby("segment")
        .agg(
            n=("segment", "size"),
            avg_cate=("cate", "mean"),
            avg_churn_proba=("churn_proba", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )


def write_propensity_summary(X: pd.DataFrame, path: Path) -> Path:
    path.write_text(json.dumps(propensity_trim_summary(X), indent=2, default=float))
    return path
