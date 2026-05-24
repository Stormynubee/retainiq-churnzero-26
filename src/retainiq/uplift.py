"""T-learner uplift on retention_offer_received.

We model P(stay | offer) - P(stay | no offer). Positive CATE = offer helps retention.
Offers weren't randomised in the data — mention that on the deck.
"""

from __future__ import annotations

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


def fit_t_learner(X: pd.DataFrame, y: pd.Series) -> dict:
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
    X_no_t = _drop_uplift_leakage_cols(X)
    p_stay_t = model_dict["mu_1"].predict_proba(X_no_t)[:, 1]
    p_stay_c = model_dict["mu_0"].predict_proba(X_no_t)[:, 1]
    return p_stay_t - p_stay_c


def estimate_propensity(X: pd.DataFrame) -> np.ndarray:
    """P(offer | X) — rough check for selection bias."""
    if config.TREATMENT_COL_PRIMARY not in X.columns:
        raise KeyError("treatment column missing")
    t = X[config.TREATMENT_COL_PRIMARY].astype(int).values
    X_num = X.drop(columns=[config.TREATMENT_COL_PRIMARY]).select_dtypes(include=[np.number])
    lr = LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)
    lr.fit(X_num, t)
    return lr.predict_proba(X_num)[:, 1]


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
