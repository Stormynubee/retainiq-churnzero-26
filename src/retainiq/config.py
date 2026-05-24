"""Project-wide constants: paths, costs, seeds, column groups."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TRAIN_CSV = DATA_RAW / "ChurnZero_dataset_v1.csv"
TEST_CSV = DATA_RAW / "ChurnZero_test_v1.csv"

# Naming follows the official ZIP contract from the problem PDF.
TEAM_NAME = "RetainIQ"
SUBMISSION_CSV = SUBMISSION_DIR / f"ChurnZero_{TEAM_NAME}_Predictions.csv"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

# From the problem statement:
#   FN cost = INR 40,000 (lost CLV from a missed churner)
#   FP cost = INR 500    (wasted retention call)
FN_COST = 40_000
FP_COST = 500

# Closed-form optimum for calibrated probabilities (Bayes risk minimiser).
THEORETICAL_OPTIMAL_THRESHOLD = FP_COST / (FN_COST + FP_COST)  # ~0.01235

TARGET_COL = "churn"
ID_COL = "customer_id"

# Treatment indicators for the uplift / causal layer
TREATMENT_COL_PRIMARY = "retention_offer_received"
TREATMENT_COL_COMPLIANCE = "retention_offer_accepted"
TREATMENT_COL_SECONDARY = "discount_or_fee_waiver_received"

# Categoricals verified by inspecting the training file header.
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

# Sensitive attributes for the fairness audit.
SENSITIVE_COLS: list[str] = ["gender", "region"]

# Never feed these to the model. customer_id is just an identifier.
DROP_BEFORE_FEATURES: list[str] = [ID_COL]
