"""Causal / uplift layer — Layer 3 of RetainIQ.

The dataset includes `retention_offer_received` (treatment) and
`retention_offer_accepted` (compliance). We use the T-learner pattern:

    mu_1(x) = E[Y | X=x, T=1]   <- model trained on treated subset
    mu_0(x) = E[Y | X=x, T=0]   <- model trained on control subset
    CATE(x) = mu_1(x) - mu_0(x)

In churn, the "treatment effect" we want is on **non-churn**, not churn,
because the offer is supposed to *retain*. We therefore flip Y to (1 - churn)
when computing CATE, so positive CATE = "offer makes them stay more".

This produces the four-cell segmentation:
    persuadable: high CATE, would have churned without offer
    sure-thing: low CATE, would have stayed anyway (don't waste offer)
    lost-cause: ~0 CATE, will leave regardless (don't waste offer)
    sleeping-dog: negative CATE — contacting them makes things WORSE

CAVEAT: `retention_offer_received` was not randomly assigned in the raw
dataset; banks targeted whom they thought was at risk. The CATE estimate
therefore has selection bias. We acknowledge this in the deck and report
both raw T-learner and a propensity-stratified variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def split_treatment(
    X: pd.DataFrame, y: pd.Series, treatment_col: str = config.TREATMENT_COL_PRIMARY
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_treated, y_treated, X_control, y_control)."""
    if treatment_col not in X.columns:
        raise KeyError(f"Treatment column {treatment_col!r} not in features")

    t = X[treatment_col].astype(int).values
    X_no_t = X.drop(columns=[treatment_col])

    treated_mask = t == 1
    return (
        X_no_t.loc[treated_mask].copy(),
        y.loc[treated_mask].copy(),
        X_no_t.loc[~treated_mask].copy(),
        y.loc[~treated_mask].copy(),
    )


def fit_t_learner(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit two LGBM models — one on treated, one on control.

    The target inside the learners is "stay" = (1 - churn), so positive
    CATE means the offer reduces churn.
    """
    X_t, y_t, X_c, y_c = split_treatment(X, y)
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
    mu_1 = LGBMClassifier(**common).fit(X_t, stay_t)
    mu_0 = LGBMClassifier(**common).fit(X_c, stay_c)
    return {"mu_1": mu_1, "mu_0": mu_0}


def estimate_cate(model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """Return per-customer CATE estimates."""
    X_no_t = X.drop(columns=[config.TREATMENT_COL_PRIMARY], errors="ignore")
    p_stay_t = model_dict["mu_1"].predict_proba(X_no_t)[:, 1]
    p_stay_c = model_dict["mu_0"].predict_proba(X_no_t)[:, 1]
    return p_stay_t - p_stay_c


def estimate_propensity(X: pd.DataFrame) -> np.ndarray:
    """Crude propensity model: P(treatment | X). Used for bias adjustment."""
    if config.TREATMENT_COL_PRIMARY not in X.columns:
        raise KeyError("Treatment column missing — cannot fit propensity model")
    t = X[config.TREATMENT_COL_PRIMARY].astype(int).values
    X_no_t = X.drop(columns=[config.TREATMENT_COL_PRIMARY])

    # Numeric-only LR for transparency; categorical-aware LGBM is fine too.
    X_num = X_no_t.select_dtypes(include=[np.number])
    lr = LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)
    lr.fit(X_num, t)
    return lr.predict_proba(X_num)[:, 1]


def segment_customers(
    cate: np.ndarray, churn_proba: np.ndarray
) -> pd.Series:
    """Map (CATE, churn_proba) -> {persuadable, sure-thing, lost-cause, sleeping-dog}.

    Heuristic boundaries:
        persuadable:   CATE > 0.05  AND  churn_proba > median
        sure-thing:    CATE < 0.05  AND  churn_proba < 0.10
        lost-cause:    CATE < 0.05  AND  churn_proba >= 0.50
        sleeping-dog:  CATE < -0.02
    """
    cate = np.asarray(cate)
    churn_proba = np.asarray(churn_proba)
    median_p = np.median(churn_proba)
    labels = np.full(len(cate), fill_value="other", dtype=object)
    labels[(cate > 0.05) & (churn_proba > median_p)] = "persuadable"
    labels[(cate.__abs__() <= 0.05) & (churn_proba < 0.10)] = "sure-thing"
    labels[(cate.__abs__() <= 0.05) & (churn_proba >= 0.50)] = "lost-cause"
    labels[cate < -0.02] = "sleeping-dog"
    return pd.Series(labels)


def segmentation_report(
    cate: np.ndarray, churn_proba: np.ndarray
) -> pd.DataFrame:
    """Per-segment summary for the deck slide."""
    seg = segment_customers(cate, churn_proba)
    df = pd.DataFrame(
        {"segment": seg, "cate": cate, "churn_proba": churn_proba}
    )
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
