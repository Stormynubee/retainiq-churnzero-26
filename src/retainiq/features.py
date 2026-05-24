"""Feature engineering — leak-safe by construction.

Public API:
    fit_transform(X_train) -> (X_train_fe, fitter_state)
    transform(X_test, fitter_state) -> X_test_fe

The "fitter state" is just the median values + observed categories captured
on the training set; nothing about the test set ever flows back into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# State carried from fit -> transform (so test can't leak into train)
# ---------------------------------------------------------------------------
@dataclass
class FeatureFitState:
    """Everything we learned on train that we need to apply to test."""

    numeric_medians: dict[str, float] = field(default_factory=dict)
    isna_flag_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    final_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engineered features (domain knowledge → strong predictors)
# ---------------------------------------------------------------------------
def _engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven ratios / deltas. Idempotent."""
    df = df.copy()

    safe = lambda x: x.replace(0, np.nan)  # avoid div-by-zero  # noqa: E731

    # Wallet share / liquidity
    if {"avg_monthly_balance", "annual_income"}.issubset(df.columns):
        df["balance_to_income"] = df["avg_monthly_balance"] / safe(
            df["annual_income"]
        )

    # Engagement decay — high last_login_days vs short tenure = trouble
    if {"last_login_days", "tenure_months"}.issubset(df.columns):
        df["engagement_decay"] = df["last_login_days"] / (1 + df["tenure_months"])

    # Quarterly behaviour drop (combines amt change + count change)
    if {"total_amt_chng_q4_q1", "total_ct_chng_q4_q1"}.issubset(df.columns):
        df["q4_q1_combined_drop"] = (1 - df["total_amt_chng_q4_q1"]) + (
            1 - df["total_ct_chng_q4_q1"]
        )

    # Product density (multi-product but new = stickier)
    if {"number_of_products", "tenure_months"}.issubset(df.columns):
        df["product_density"] = df["number_of_products"] / (
            1 + df["tenure_months"]
        )

    # Complaint pressure index
    if {"total_complaints", "escalation_count", "complaint_resolution_time"}.issubset(
        df.columns
    ):
        df["complaint_pressure"] = (
            df["total_complaints"] * (1 + df["escalation_count"])
        ) / (1 + df["complaint_resolution_time"])

    # Credit stress
    if {"credit_utilization_ratio", "late_credit_card_payment_count"}.issubset(
        df.columns
    ):
        df["credit_stress"] = df["credit_utilization_ratio"] * (
            1 + df["late_credit_card_payment_count"]
        )

    # Digital vs branch preference
    if {"branch_visit_count", "total_digital_logins"}.issubset(df.columns):
        df["digital_vs_branch"] = df["total_digital_logins"] / (
            1 + df["branch_visit_count"]
        )

    # Retention offer effectiveness flag (was offer received AND accepted?)
    if {"retention_offer_received", "retention_offer_accepted"}.issubset(df.columns):
        df["retention_offer_effective"] = (
            df["retention_offer_received"] * df["retention_offer_accepted"]
        )

    return df


# ---------------------------------------------------------------------------
# Missingness handling
# ---------------------------------------------------------------------------
def _add_missingness_flags(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """For numeric cols with NaNs, add an *_isna flag BEFORE imputation."""
    df = df.copy()
    flag_cols: list[str] = []
    for col in numeric_cols:
        if df[col].isna().any():
            flag = f"{col}_isna"
            df[flag] = df[col].isna().astype(int)
            flag_cols.append(flag)
    return df, flag_cols


def _resolve_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce categoricals to category dtype with 'Unknown' as a real level."""
    df = df.copy()
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype("category")
    return df


# ---------------------------------------------------------------------------
# Public fit/transform API
# ---------------------------------------------------------------------------
def fit_transform(X: pd.DataFrame) -> tuple[pd.DataFrame, FeatureFitState]:
    """Fit on training X; return engineered frame and fitter state."""
    state = FeatureFitState()

    # 1. Engineered features (no fitting — pure functions)
    X_fe = _engineer_ratios(X)

    # 2. Resolve categoricals
    X_fe = _resolve_categoricals(X_fe)
    state.categorical_cols = [
        c for c in config.CATEGORICAL_COLS if c in X_fe.columns
    ]

    # 3. Missingness flags + numeric imputation (medians from TRAIN ONLY)
    numeric_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    X_fe, flag_cols = _add_missingness_flags(X_fe, numeric_cols)
    state.isna_flag_cols = flag_cols

    medians = {col: float(X_fe[col].median()) for col in numeric_cols}
    state.numeric_medians = medians
    X_fe[numeric_cols] = X_fe[numeric_cols].fillna(value=medians)

    # 4. Replace any inf produced by ratios with median
    for col in numeric_cols:
        if np.isinf(X_fe[col]).any():
            X_fe[col] = X_fe[col].replace([np.inf, -np.inf], medians.get(col, 0.0))

    state.final_columns = X_fe.columns.tolist()
    return X_fe, state


def transform(X: pd.DataFrame, state: FeatureFitState) -> pd.DataFrame:
    """Apply the fitted state to a new dataframe (e.g. test)."""
    X_fe = _engineer_ratios(X)
    X_fe = _resolve_categoricals(X_fe)

    # Missingness flags — must match training shape
    for flag in state.isna_flag_cols:
        base = flag.removesuffix("_isna")
        if base in X_fe.columns:
            X_fe[flag] = X_fe[base].isna().astype(int)
        else:
            X_fe[flag] = 0

    # Apply training medians for imputation (NEVER recompute on test)
    for col, median in state.numeric_medians.items():
        if col in X_fe.columns:
            X_fe[col] = X_fe[col].fillna(median).replace(
                [np.inf, -np.inf], median
            )

    # Re-order to training column order (creates any missing as 0)
    for col in state.final_columns:
        if col not in X_fe.columns:
            X_fe[col] = 0
    X_fe = X_fe[state.final_columns]

    return X_fe


def list_categorical_indices(
    X: pd.DataFrame, state: FeatureFitState
) -> list[int]:
    """Return positional indices of categorical columns (for CatBoost)."""
    return [
        X.columns.get_loc(c)
        for c in state.categorical_cols
        if c in X.columns
    ]
