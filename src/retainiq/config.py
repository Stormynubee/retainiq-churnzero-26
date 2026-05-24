"""Single source of truth for paths, costs, seeds, and column groups.

Every other module imports from here. If a constant is hardcoded
elsewhere, that is a bug — fix the caller, not by adding a duplicate.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TRAIN_CSV = DATA_RAW / "ChurnZero_dataset_v1.csv"
TEST_CSV = DATA_RAW / "ChurnZero_test_v1.csv"

# Submission file naming follows the official ZIP contract.
TEAM_NAME = "RetainIQ"
SUBMISSION_CSV = SUBMISSION_DIR / f"ChurnZero_{TEAM_NAME}_Predictions.csv"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5  # stratified k-fold

# ---------------------------------------------------------------------------
# Business cost matrix (from the official problem statement)
# ---------------------------------------------------------------------------
# FN: predicted not-churn but actually churned -> lost CLV
# FP: predicted churn but actually loyal -> wasted retention spend
FN_COST = 40_000  # rupees
FP_COST = 500  # rupees

# Closed-form cost-optimal threshold for perfectly calibrated probabilities:
#   predict 1 iff p > FP_COST / (FN_COST + FP_COST)
THEORETICAL_OPTIMAL_THRESHOLD = FP_COST / (FN_COST + FP_COST)  # ≈ 0.01235

# ---------------------------------------------------------------------------
# Target / id
# ---------------------------------------------------------------------------
TARGET_COL = "churn"
ID_COL = "customer_id"

# ---------------------------------------------------------------------------
# Treatment columns for uplift / causal layer
# ---------------------------------------------------------------------------
TREATMENT_COL_PRIMARY = "retention_offer_received"
TREATMENT_COL_COMPLIANCE = "retention_offer_accepted"
TREATMENT_COL_SECONDARY = "discount_or_fee_waiver_received"

# ---------------------------------------------------------------------------
# Categorical columns (verified against the actual training file header).
# ---------------------------------------------------------------------------
CATEGORICAL_COLS: list[str] = [
    "gender",
    "marital_status",
    "education_level",
    "occupation_type",
    "income_band",
    "income_category",
    "city_tier",
    "region",
    "customer_segment",
    "onboarding_channel",
    "relationship_type",
    "primary_account_type",
    "card_category",
    "competitor_bank_offer_awareness",
    "customer_feedback_sentiment",
]

# Sensitive attributes for the fairness audit (Layer 5).
SENSITIVE_COLS: list[str] = ["gender", "region"]

# Columns we never feed to the predictive model (id only — leak guard).
DROP_BEFORE_FEATURES: list[str] = [ID_COL]
