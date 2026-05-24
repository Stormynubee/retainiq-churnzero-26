"""Generate synthetic mini CSVs for CI smoke tests — not competition data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

CATEGORICAL_DEFAULTS = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education_level": ["Graduate", "Postgraduate", "High School"],
    "occupation_type": ["Salaried", "Self-employed"],
    "income_band": ["Low", "Mid", "High"],
    "income_category": ["A", "B", "C"],
    "city_tier": ["Tier1", "Tier2", "Tier3"],
    "region": ["North", "South", "East", "West"],
    "customer_segment": ["Mass", "Affluent", "Premium"],
    "onboarding_channel": ["Branch", "Digital"],
    "relationship_type": ["Individual", "Joint"],
    "primary_account_type": ["Savings", "Current"],
    "card_category": ["Classic", "Gold", "Platinum"],
    "competitor_bank_offer_awareness": ["Yes", "No", "Unknown"],
    "customer_feedback_sentiment": ["Positive", "Neutral", "Negative"],
}


def _build(n: int, rng: np.random.Generator, churn_rate: float = 0.16) -> pd.DataFrame:
    rows: dict[str, list] = {
        "customer_id": [f"MINI_{i:05d}" for i in range(n)],
        "churn": (rng.random(n) < churn_rate).astype(int).tolist(),
    }

    for col, choices in CATEGORICAL_DEFAULTS.items():
        rows[col] = rng.choice(choices, n).tolist()

    for col in [
        "dependent_count",
        "referral_count",
        "number_of_products",
        "tenure_months",
        "last_login_days",
        "total_digital_logins",
        "total_trans_count",
        "branch_visit_count",
        "relationship_manager_interaction_count",
        "campaign_response_count",
        "retention_offer_received",
        "retention_offer_accepted",
        "total_complaints",
        "escalation_count",
    ]:
        rows[col] = rng.integers(0, 20, n).tolist()

    rows["annual_income"] = rng.integers(200_000, 2_000_000, n).tolist()
    rows["avg_monthly_balance"] = rng.integers(5_000, 200_000, n).tolist()
    rows["balance_decline_percentage"] = rng.uniform(0, 0.5, n).round(4).tolist()
    rows["total_amt_chng_q4_q1"] = rng.uniform(0.5, 1.2, n).round(4).tolist()
    rows["total_ct_chng_q4_q1"] = rng.uniform(0.5, 1.2, n).round(4).tolist()
    rows["credit_utilization_ratio"] = rng.uniform(0, 1, n).round(4).tolist()
    rows["complaint_resolution_time"] = rng.integers(1, 30, n).tolist()

    # remaining numeric / flag columns default to small random values
    base_cols = set(rows) | {"churn", "customer_id"}
    extra_numeric = [
        "age",
        "customer_lifetime_value",
        "loyalty_program_member",
        "last_contacted_days",
        "relationship_manager_assigned",
        "current_balance",
        "monthly_transaction_count",
        "monthly_transaction_value",
        "cash_withdrawal_count",
        "upi_transaction_count",
        "debit_card_transaction_count",
        "net_banking_transaction_count",
        "account_inactive_days",
        "total_trans_amt",
        "avg_open_to_buy",
        "total_revolving_bal",
        "savings_account_flag",
        "current_account_flag",
        "credit_card_flag",
        "personal_loan_flag",
        "home_loan_flag",
        "auto_loan_flag",
        "fixed_deposit_flag",
        "investment_product_flag",
        "insurance_product_flag",
        "demat_account_flag",
        "credit_card_limit",
        "credit_card_spend",
        "minimum_due_paid_flag",
        "late_credit_card_payment_count",
        "loan_outstanding_amount",
        "emi_amount",
        "emi_payment_delay_count",
        "loan_default_risk_score",
        "prepayment_flag",
        "mobile_app_login_count",
        "website_login_count",
        "digital_transaction_ratio",
        "failed_login_count",
        "app_rating_given",
        "paperless_statement_enabled",
        "digital_service_usage_score",
        "mobile_banking_active_flag",
        "email_open_rate",
        "unresolved_complaint_count",
        "call_center_interaction_count",
        "service_request_count",
        "satisfaction_score",
        "nps_score",
        "campaign_received_count",
        "cross_sell_offer_count",
        "upsell_offer_count",
        "last_campaign_response_days",
        "discount_or_fee_waiver_received",
        "monthly_income_estimate",
        "credit_utilization_3m_avg",
        "credit_utilization_6m_avg",
        "avg_quarterly_balance",
        "debt_to_income_ratio",
        "digital_engagement_index",
    ]
    for col in extra_numeric:
        if col not in base_cols:
            rows[col] = rng.integers(0, 100, n).tolist()

    df = pd.DataFrame(rows)
    # churn signal: higher login decline -> more churn
    signal = (
        df["balance_decline_percentage"] * 2
        + (df["last_login_days"] / 100)
        - (df["total_digital_logins"] / 50)
    )
    df["churn"] = (signal > signal.quantile(1 - churn_rate)).astype(int)
    return df


def main() -> None:
    rng = np.random.default_rng(42)
    train = _build(160, rng)
    test = _build(40, rng, churn_rate=0.16).drop(columns=["churn"])
    test["customer_id"] = [f"MINI_T_{i:05d}" for i in range(len(test))]

    train.to_csv(OUT / "mini_train.csv", index=False)
    test.to_csv(OUT / "mini_test.csv", index=False)
    print(f"wrote {OUT / 'mini_train.csv'} ({len(train)} rows)")
    print(f"wrote {OUT / 'mini_test.csv'} ({len(test)} rows)")


if __name__ == "__main__":
    main()
