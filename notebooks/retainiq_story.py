# %% [markdown]
# # RetainIQ — ChurnZero 26 walkthrough
#
# Hansraj Tiwari & swayangjeet nayak
#
# Reads outputs from `scripts/train` and `scripts/build_artifacts`.
# Good enough for judges to follow without running the full training loop.

# %%
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

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
# ## 1. Data

# %%
df_train = pd.read_csv(PROJECT / "data" / "raw" / "ChurnZero_dataset_v1.csv")
df_test = pd.read_csv(PROJECT / "data" / "raw" / "ChurnZero_test_v1.csv")
print(f"Train {df_train.shape}, churn rate {df_train['churn'].mean():.4f}")
print(f"Test  {df_test.shape}")
df_train["churn"].value_counts()

# %% [markdown]
# ~16% churn. 97 features. We keep `Unknown` categories and add missingness flags before imputing.

# %% [markdown]
# ## 2. Model performance (OOF)

# %%
metrics = json.loads((PROC / "training_metrics.json").read_text())
optimal = metrics["metrics_optimal"]
naive = metrics["metrics_naive"]

pd.DataFrame({"cost-optimal": optimal, "t=0.5": naive}).T[
    ["pr_auc", "recall", "f1", "total_cost_inr", "predicted_positive_rate"]
]

# %% [markdown]
# PR-AUC is basically 1.0 — the data separates easily. The interesting bit is cost at different thresholds.

# %% [markdown]
# ## 3. Cost threshold

# %%
display(Image(filename=str(CHARTS / "08_cost_curve.png")))

threshold_info = json.loads((PROC / "cost_optimal_threshold.json").read_text())
print(f"optimal t     : {threshold_info['threshold']:.4f}")
print(f"theory t      : {threshold_info['theoretical_optimal_threshold']:.4f}")
print(f"cost @ 0.5    : INR {threshold_info['naive_threshold_cost_inr']:,.0f}")
print(f"cost @ optimal: INR {threshold_info['total_cost_inr']:,.0f}")
print(f"saved         : INR {threshold_info['savings_vs_naive_inr']:,.0f}")

# %% [markdown]
# ## 4. Feature drivers

# %%
display(Image(filename=str(CHARTS / "09_feature_importance.png")))

imp = pd.read_csv(PROC / "feature_importances.csv").head(8)
imp

# %% [markdown]
# ## 5. Uplift — who is worth calling?

# %%
display(Image(filename=str(CHARTS / "10_uplift_quadrant.png")))

uplift_seg = pd.read_csv(PROC / "uplift_segmentation.csv")
uplift_seg

# %% [markdown]
# Most customers are sure-things or lost-causes. Only a handful are persuadable.
# Offers in the data were not randomised — we say that on slide 10.

# %%
n_pers = int(uplift_seg.loc[uplift_seg["segment"] == "persuadable", "n"].iloc[0])
n_sleep = int(uplift_seg.loc[uplift_seg["segment"] == "sleeping-dog", "n"].iloc[0])
print(f"{n_pers} persuadables; {n_sleep} sleeping-dogs (negative CATE on average)")

per_cust = pd.read_csv(PROC / "uplift_per_customer.csv")
per_cust[per_cust["segment"] == "persuadable"].head()

# %% [markdown]
# ## 6. Fairness

# %%
display(Image(filename=str(CHARTS / "12_fairness.png")))
pd.read_csv(PROC / "fairness_summary.csv")

# %% [markdown]
# ## 7. Playbook (what we'd tell the bank)

# %%
pd.DataFrame(
    [
        ("Persuadable", "Call + waiver", "Only group where offer clearly helps"),
        ("Sleeping-dog", "Do not call", "Negative CATE"),
        ("Lost-cause", "Skip retention", "Won't stay anyway"),
        ("Sure-thing", "Cross-sell", "Don't annoy them"),
    ],
    columns=["segment", "action", "note"],
)

# %% [markdown]
# ## 8. Submission check

# %%
sub = pd.read_csv(PROJECT / "submission" / "ChurnZero_RetainIQ_Predictions.csv")
print(len(sub), "rows, positive rate", sub["churn_prediction"].mean())
sub.head()

# %% [markdown]
# ## Reproduce from scratch
#
# ```
# pip install -r requirements.txt
# python -m scripts.train
# python -m scripts.predict
# python -m scripts.build_artifacts
# ```
