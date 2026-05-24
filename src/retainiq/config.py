"""Paths, costs, column lists. Edit here if file locations change."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TRAIN_CSV = DATA_RAW / "ChurnZero_dataset_v1.csv"
TEST_CSV = DATA_RAW / "ChurnZero_test_v1.csv"

TEAM_NAME = "RetainIQ"
SUBMISSION_CSV = SUBMISSION_DIR / f"ChurnZero_{TEAM_NAME}_Predictions.csv"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

# problem statement cost matrix
FN_COST = 40_000
FP_COST = 500
THEORETICAL_OPTIMAL_THRESHOLD = FP_COST / (FN_COST + FP_COST)  # ~0.01235

TARGET_COL = "churn"
ID_COL = "customer_id"

TREATMENT_COL_PRIMARY = "retention_offer_received"
TREATMENT_COL_COMPLIANCE = "retention_offer_accepted"
TREATMENT_COL_SECONDARY = "discount_or_fee_waiver_received"

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

SENSITIVE_COLS: list[str] = ["gender", "region"]
DROP_BEFORE_FEATURES: list[str] = [ID_COL]
