"""Load canonical deck numbers from processed training artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from retainiq import config


@dataclass(frozen=True)
class DeckMetrics:
    threshold: float
    pr_auc: float
    cost_optimal_inr: int
    cost_naive_inr: int
    savings_inr: int
    savings_pct: float
    persuadable_n: int
    sleeping_dog_n: int
    persuadable_avg_cate: float
    sleeping_dog_avg_cate: float
    gender_dp_gap_pct: float
    region_dp_gap_pct: float


def load_deck_metrics() -> DeckMetrics:
    tm_path = config.DATA_PROCESSED / "training_metrics.json"
    if not tm_path.is_file():
        raise FileNotFoundError("Run: python -m scripts.train")

    tm = json.loads(tm_path.read_text(encoding="utf-8"))
    opt = tm["metrics_optimal"]
    naive = tm["metrics_naive"]
    savings = int(naive["total_cost_inr"] - opt["total_cost_inr"])
    pct = 100.0 * savings / naive["total_cost_inr"]

    seg_path = config.DATA_PROCESSED / "uplift_segmentation.csv"
    if not seg_path.is_file():
        raise FileNotFoundError("Run: python -m scripts.build_artifacts")
    seg = pd.read_csv(seg_path).set_index("segment")

    fair_path = config.DATA_PROCESSED / "fairness_summary.csv"
    if not fair_path.is_file():
        raise FileNotFoundError("Run: python -m scripts.build_artifacts")
    fair = pd.read_csv(fair_path)
    gender_row = fair[fair["attribute"] == "gender"].iloc[0]
    region_row = fair[fair["attribute"] == "region"].iloc[0]

    dp_col = "demographic_parity_diff"
    if dp_col not in fair.columns:
        dp_col = "demographic_parity_gap"

    return DeckMetrics(
        threshold=float(opt["threshold"]),
        pr_auc=round(float(opt["pr_auc"]), 4),
        cost_optimal_inr=int(opt["total_cost_inr"]),
        cost_naive_inr=int(naive["total_cost_inr"]),
        savings_inr=savings,
        savings_pct=round(pct, 1),
        persuadable_n=int(seg.loc["persuadable", "n"]),
        sleeping_dog_n=int(seg.loc["sleeping-dog", "n"]),
        persuadable_avg_cate=round(float(seg.loc["persuadable", "avg_cate"]), 2),
        sleeping_dog_avg_cate=round(float(seg.loc["sleeping-dog", "avg_cate"]), 2),
        gender_dp_gap_pct=round(100.0 * float(gender_row[dp_col]), 1),
        region_dp_gap_pct=round(100.0 * float(region_row[dp_col]), 1),
    )
