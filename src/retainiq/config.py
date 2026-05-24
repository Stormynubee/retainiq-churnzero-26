"""Paths, costs, column lists. Edit here if file locations change."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TRAIN_CSV = DATA_RAW / "ChurnZero_dataset_v1.csv"
TEST_CSV = DATA_RAW / "ChurnZero_test_v1.csv"

# Unstop team registration (filename slug: no spaces/underscores)
TEAM_NAME = "TeamVortex"
TEAM_DISPLAY_NAME = "Team Vortex"
# Project / deck product name (RetainIQ pipeline and slides)
PRODUCT_NAME = "RetainIQ"

SUBMISSION_CSV = SUBMISSION_DIR / f"ChurnZero_{TEAM_NAME}_Predictions.csv"
CODE_SUBMISSION = PROJECT_ROOT / f"ChurnZero_{TEAM_NAME}_Code.py"
SUBMISSION_ZIP = SUBMISSION_DIR / f"ChurnZero_{TEAM_NAME}.zip"
DECK_PPTX = PROJECT_ROOT / "deck" / f"ChurnZero_{TEAM_NAME}_Presentation.pptx"
DECK_PDF = PROJECT_ROOT / "deck" / f"ChurnZero_{TEAM_NAME}_Presentation.pdf"

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

# Post-treatment outcomes — never use to predict pre-intervention churn
POST_TREATMENT_COLS: list[str] = [
    TREATMENT_COL_COMPLIANCE,
    TREATMENT_COL_SECONDARY,
]

# propensity overlap for IPTW uplift
PROPENSITY_MIN = 0.05
PROPENSITY_MAX = 0.95

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
DROP_BEFORE_FEATURES: list[str] = [ID_COL, *POST_TREATMENT_COLS]
