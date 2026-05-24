# %% [markdown]
# # RetainIQ — From Churn Prediction to a Persuadable-Customer Engine
#
# **ChurnZero 26 · IIT Kharagpur · Round 2 submission**
#
# This notebook is the readable companion to the RetainIQ codebase. It loads
# the artefacts produced by `python -m scripts.train` and `python -m
# scripts.build_artifacts` and walks through the full story end-to-end.
#
# > **One-line claim**: We don't just predict churn. We predict who is *worth
# > saving*, prove it in rupees, and stop the bank from accidentally
# > triggering churn by calling the wrong people.
#
# Anchors for the judges:
#
# 1. **Cost-aware threshold** — the rubric's FN=₹40,000 vs FP=₹500 cost
#    matrix means the right decision threshold is ~0.001, not 0.5. Our model
#    cuts business cost by **68.8%** vs the naive baseline.
# 2. **Causal uplift** — using `retention_offer_received` as a real
#    treatment, we find that only **4 of 8,101 customers** are genuinely
#    persuadable, while **141 are sleeping-dogs** that the bank actively
#    harms by contacting.
# 3. **Fairness audit** — the model passes the 4/5ths-style fairness rule on
#    both Gender (DP diff 2.7%) and Region (DP diff 1.3%).

# %%
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

# Make the project root importable
ROOT = Path.cwd()
if (ROOT / "src" / "retainiq").exists():
    PROJECT = ROOT
else:
    PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from retainiq import config  # noqa: E402

PROC = PROJECT / "data" / "processed"
CHARTS = PROJECT / "deck" / "charts"

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1. Dataset glance

# %%
df_train = pd.read_csv(PROJECT / "data" / "raw" / "ChurnZero_dataset_v1.csv")
df_test = pd.read_csv(PROJECT / "data" / "raw" / "ChurnZero_test_v1.csv")
print(f"Train: {df_train.shape}  ·  Churn rate: {df_train['churn'].mean():.4f}")
print(f"Test:  {df_test.shape}   ·  no target")
df_train["churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).to_frame("count")

# %% [markdown]
# ~16% churn — typical for retail banking. The dataset has 97 features
# spanning 8 categories: customer profile, relationship/tenure, account &
# transactions, product holdings, credit/loan, digital engagement, service &
# complaints, and marketing/retention. We treat `Unknown` as a real category
# (it's a signal, not noise) and add explicit *_isna flags for numeric
# columns with missing values before imputing the median.

# %% [markdown]
# ## 2. Headline model performance (PR-AUC, F1, business cost)
#
# All numbers below are out-of-fold across 5 stratified folds — never
# in-fold, never test-set leakage.

# %%
metrics = json.loads((PROC / "training_metrics.json").read_text())
optimal = metrics["metrics_optimal"]
naive = metrics["metrics_naive"]

results = pd.DataFrame({
    "Cost-optimal (t=0.001)": optimal,
    "Naive (t=0.5)": naive,
}).T[[
    "pr_auc", "roc_auc", "precision", "recall", "f1",
    "tp", "fp", "fn", "tn",
    "total_cost_inr", "predicted_positive_rate",
]]
results.columns = [
    "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1",
    "TP", "FP", "FN", "TN",
    "Total Cost ₹", "Predicted Positive Rate",
]
results

# %% [markdown]
# Two observations the deck must drive home:
#
# - **PR-AUC is essentially 1.0**. The dataset is highly separable. Every
#   competing team will likely hit ≥0.95. AUC alone is therefore *not* the
#   differentiator — the rest of the rubric is.
# - **Recall flips from 99.6% to 100%** when we move from threshold 0.5 to
#   the cost-optimal 0.001. Five churners worth ₹2 lakh in lost CLV are
#   recovered — at the cost of 124 extra false alarms worth ₹62k.

# %% [markdown]
# ## 3. The cost slide — same model, different decision
#
# The official cost matrix is asymmetric: a False Negative (a missed
# churner) costs the bank ₹40,000 in lost CLV, while a False Positive (an
# unnecessary retention call) costs only ₹500. That's an **80:1 ratio**.
#
# For perfectly calibrated probabilities, the closed-form cost-optimal
# threshold is:
#
# $$t^* = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{500}{40500} \approx 0.0123$$
#
# Most teams will leave their classifier at the default 0.5. We don't.

# %%
display(Image(filename=str(CHARTS / "08_cost_curve.png")))

# %%
threshold_info = json.loads((PROC / "cost_optimal_threshold.json").read_text())
print(f"Empirical cost-optimal threshold : {threshold_info['threshold']:.4f}")
print(f"Theoretical optimum (closed form): {threshold_info['theoretical_optimal_threshold']:.4f}")
print(f"Cost @ naive  0.5 threshold      : ₹{threshold_info['naive_threshold_cost_inr']:,.0f}")
print(f"Cost @ optimal threshold         : ₹{threshold_info['total_cost_inr']:,.0f}")
print(f"Saving                           : ₹{threshold_info['savings_vs_naive_inr']:,.0f}  "
      f"({threshold_info['savings_pct_vs_naive']:.1f}%)")

# %% [markdown]
# ## 4. Top churn drivers — actionable vs not-actionable
#
# We separate features the bank can *change* (engagement, complaints,
# campaigns) from those it cannot (age, tenure). Predictive importance
# without that distinction is useless on Monday morning.

# %%
display(Image(filename=str(CHARTS / "09_feature_importance.png")))

# %%
imp = pd.read_csv(PROC / "feature_importances.csv").head(8)
actionable_lookup = {
    "total_digital_logins": "actionable",
    "balance_decline_percentage": "leading indicator",
    "relationship_manager_interaction_count": "actionable",
    "total_trans_count": "leading indicator",
    "campaign_response_count": "actionable",
    "cash_withdrawal_count": "leading indicator",
    "unresolved_complaint_count": "actionable",
    "monthly_transaction_value": "leading indicator",
}
imp["lever_type"] = imp["feature"].map(actionable_lookup).fillna("contextual")
imp[["feature", "lgb_gain_pct", "lever_type"]]

# %% [markdown]
# ## 5. The causal layer — predicted ≠ savable
#
# This is the slide that wins or loses Round 2 for us.
#
# A standard churn model ranks customers by **risk**. The bank then calls
# the top-N. But "risk" doesn't mean "savable". We split the population
# into four groups based on the **causal effect** of the retention offer
# (CATE = P(stay | offered) − P(stay | not offered)), estimated with a
# T-learner trained on the 33% of customers who actually received an offer.

# %%
display(Image(filename=str(CHARTS / "10_uplift_quadrant.png")))

# %%
seg = pd.read_csv(PROC / "uplift_segmentation.csv")
seg

# %% [markdown]
# ### What the segmentation actually says
#
# Out of 8,101 customers:
#
# - **6,707 sure-things** — won't churn whether we call them or not. Don't
#   waste the offer.
# - **1,248 lost-causes** — will churn whether we call or not. The offer
#   doesn't help. Don't waste the offer.
# - **141 sleeping-dogs** — *negative* CATE of −0.27. Calling them
#   **causes** them to churn. The bank's existing retention program is
#   actively hurting these customers.
# - **4 persuadables** — the only group where the offer creates real
#   value. CATE > +0.05 and high churn risk.
#
# **Operational implication**: the bank should *shrink* its retention
# campaign — not grow it. Call only the persuadables, never the
# sleeping-dogs. This is a fundamentally different conclusion from "send
# everyone above threshold a retention offer".
#
# **Caveat we own up to**: `retention_offer_received` was not randomly
# assigned. The bank likely targeted whom they thought were at-risk. We
# acknowledge this selection bias on slide 10 of the deck.

# %% [markdown]
# ## 6. Per-customer recourse (counterfactual sketch)
#
# For one persuadable customer, the model gives a clear, *individual* set
# of levers the bank can pull. (Full DiCE counterfactual generation lives
# in `src/retainiq/uplift.py`; the snippet below is the headline.)

# %%
per_cust = pd.read_csv(PROC / "uplift_per_customer.csv")
persuadables = per_cust[per_cust["segment"] == "persuadable"]
persuadables.head()

# %% [markdown]
# Even with only 4 persuadables, the lesson holds: each has a different
# combination of weak engagement signals (low logins, recent complaints,
# competitor offer awareness) that — if reversed — flips the prediction.

# %% [markdown]
# ## 7. Fairness audit
#
# Industry deployment of a churn-treatment model is a regulated decision.
# We check both demographic parity (similar predicted-positive rate across
# groups) and equal opportunity (similar TPR across groups). Both attributes
# pass the 10% (4/5ths-style) rule.

# %%
display(Image(filename=str(CHARTS / "12_fairness.png")))

# %%
pd.read_csv(PROC / "fairness_summary.csv")

# %%
pd.read_csv(PROC / "fairness_per_group.csv")

# %% [markdown]
# ## 8. The retention playbook (operational hand-off)
#
# This is what the retention team does on Monday morning, derived from the
# above three layers:

# %%
playbook = pd.DataFrame([
    {
        "Segment": "Persuadable",
        "Trigger Rule": "CATE > 0.05  AND  P(churn) > median",
        "Action": "Dedicated RM call + 12-month fee waiver",
        "Expected ROI": "High — every saved customer = ₹40k retained CLV",
        "Est. monthly contacts": 4,
    },
    {
        "Segment": "Sleeping-dog",
        "Trigger Rule": "CATE < -0.02",
        "Action": "DO NOT contact (calling worsens churn)",
        "Expected ROI": "Pure savings — every skipped contact avoids triggering churn",
        "Est. monthly contacts": 0,
    },
    {
        "Segment": "Lost-cause",
        "Trigger Rule": "CATE ≈ 0  AND  P(churn) > 0.5",
        "Action": "Skip retention; offer graceful exit + re-engagement in 12 months",
        "Expected ROI": "Cost saving + brand value",
        "Est. monthly contacts": 0,
    },
    {
        "Segment": "Sure-thing",
        "Trigger Rule": "CATE ≈ 0  AND  P(churn) < 0.10",
        "Action": "No retention action; route to cross-sell flow",
        "Expected ROI": "Avoid annoyance; potential cross-sell upside",
        "Est. monthly contacts": 0,
    },
])
playbook

# %% [markdown]
# ## 9. Submission preview & format check

# %%
sub = pd.read_csv(PROJECT / "submission" / "ChurnZero_RetainIQ_Predictions.csv")
print(f"Rows                     : {len(sub)} (expected 2,026)")
print(f"Columns                  : {list(sub.columns)}")
print(f"Predicted positive rate  : {sub['churn_prediction'].mean():.4f}")
print(f"Mean probability         : {sub['churn_probability'].mean():.4f}")
print(f"Nulls                    : {sub.isna().sum().sum()}")
print(f"Unique customer_id check : {sub['customer_id'].is_unique}")
sub.head()

# %% [markdown]
# ## 10. Risks & honest limitations
#
# - **Selection bias in the treatment column** — the bank's existing
#   retention process targeted whom they thought were at-risk. Our T-learner
#   inherits some of that bias. Mitigation: we check the propensity model in
#   `src/retainiq/uplift.py` and report the result; in production we would
#   recommend a randomised holdout in future quarters.
# - **Saturated PR-AUC** — the dataset is unusually separable; we cannot
#   demonstrate model improvements past the third decimal place. We compete
#   on the *other 60%* of the rubric instead.
# - **Concept drift** — banking behaviour shifts seasonally and after
#   product launches. We recommend monthly retraining and a PSI-based drift
#   monitor on the top 10 features.
# - **Regulatory disclosure** — automated decisions on retail banking
#   customers in India are increasingly under RBI scrutiny. The model card
#   that accompanies this submission is the starting point.

# %% [markdown]
# ## 11. Reproducing this notebook from scratch
#
# ```powershell
# pip install -r requirements.txt
# python -m scripts.train             # ~4 min
# python -m scripts.predict           # ~10 sec → submission CSV
# python -m scripts.build_artifacts   # uplift + fairness + 4 PNG charts
# ```
#
# All artefacts are deterministic (seed pinned at 42).

# %% [markdown]
# ---
#
# *Thank you for reading. Questions and Q&A in the slide deck.*
