"""Feature engineering — fit on train, transform test with saved state."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class FeatureFitState:
    numeric_medians: dict[str, float] = field(default_factory=dict)
    isna_flag_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    final_columns: list[str] = field(default_factory=list)


def _engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe = lambda x: x.replace(0, np.nan)  # noqa: E731

    if {"avg_monthly_balance", "annual_income"}.issubset(df.columns):
        df["balance_to_income"] = df["avg_monthly_balance"] / safe(df["annual_income"])

    if {"last_login_days", "tenure_months"}.issubset(df.columns):
        df["engagement_decay"] = df["last_login_days"] / (1 + df["tenure_months"])

    if {"total_amt_chng_q4_q1", "total_ct_chng_q4_q1"}.issubset(df.columns):
        df["q4_q1_combined_drop"] = (1 - df["total_amt_chng_q4_q1"]) + (
            1 - df["total_ct_chng_q4_q1"]
        )

    if {"number_of_products", "tenure_months"}.issubset(df.columns):
        df["product_density"] = df["number_of_products"] / (1 + df["tenure_months"])

    if {"total_complaints", "escalation_count", "complaint_resolution_time"}.issubset(
        df.columns
    ):
        df["complaint_pressure"] = (
            df["total_complaints"] * (1 + df["escalation_count"])
        ) / (1 + df["complaint_resolution_time"])

    if {"credit_utilization_ratio", "late_credit_card_payment_count"}.issubset(df.columns):
        df["credit_stress"] = df["credit_utilization_ratio"] * (
            1 + df["late_credit_card_payment_count"]
        )

    if {"branch_visit_count", "total_digital_logins"}.issubset(df.columns):
        df["digital_vs_branch"] = df["total_digital_logins"] / (
            1 + df["branch_visit_count"]
        )

    if {"retention_offer_received", "retention_offer_accepted"}.issubset(df.columns):
        df["retention_offer_effective"] = (
            df["retention_offer_received"] * df["retention_offer_accepted"]
        )

    return df


def _add_missingness_flags(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    flag_cols: list[str] = []
    for col in numeric_cols:
        if df[col].isna().any():
            flag = f"{col}_isna"
            df[flag] = df[col].isna().astype(int)
            flag_cols.append(flag)
    return df, flag_cols


def _resolve_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype("category")
    return df


def fit_transform(X: pd.DataFrame) -> tuple[pd.DataFrame, FeatureFitState]:
    state = FeatureFitState()

    X_fe = _engineer_ratios(X)
    X_fe = _resolve_categoricals(X_fe)
    state.categorical_cols = [c for c in config.CATEGORICAL_COLS if c in X_fe.columns]

    numeric_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    X_fe, flag_cols = _add_missingness_flags(X_fe, numeric_cols)
    state.isna_flag_cols = flag_cols

    medians = {col: float(X_fe[col].median()) for col in numeric_cols}
    state.numeric_medians = medians
    X_fe[numeric_cols] = X_fe[numeric_cols].fillna(value=medians)

    for col in numeric_cols:
        if np.isinf(X_fe[col]).any():
            X_fe[col] = X_fe[col].replace([np.inf, -np.inf], medians.get(col, 0.0))

    state.final_columns = X_fe.columns.tolist()
    return X_fe, state


def transform(X: pd.DataFrame, state: FeatureFitState) -> pd.DataFrame:
    X_fe = _engineer_ratios(X)
    X_fe = _resolve_categoricals(X_fe)

    for flag in state.isna_flag_cols:
        base = flag.removesuffix("_isna")
        X_fe[flag] = X_fe[base].isna().astype(int) if base in X_fe.columns else 0

    for col, median in state.numeric_medians.items():
        if col in X_fe.columns:
            X_fe[col] = X_fe[col].fillna(median).replace([np.inf, -np.inf], median)

    for col in state.final_columns:
        if col not in X_fe.columns:
            X_fe[col] = 0
    return X_fe[state.final_columns]


def list_categorical_indices(X: pd.DataFrame, state: FeatureFitState) -> list[int]:
    return [X.columns.get_loc(c) for c in state.categorical_cols if c in X.columns]
