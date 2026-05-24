"""Pick a decision threshold that minimises expected rupee cost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def expected_business_cost(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred_binary).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(fn * fn_cost + fp * fp_cost)


def cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.unique(
            np.concatenate(
                [
                    np.linspace(0.001, 0.05, 50),
                    np.linspace(0.05, 0.3, 50),
                    np.linspace(0.3, 0.95, 30),
                ]
            )
        )

    rows = []
    y_true = np.asarray(y_true).astype(int)
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        tn = int(((y_true == 0) & (y_hat == 0)).sum())
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total_cost_inr": float(fn * fn_cost + fp * fp_cost),
                "predicted_positive_rate": float((y_hat == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
) -> dict:
    curve = cost_curve(y_true, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    best = curve.loc[curve["total_cost_inr"].idxmin()].to_dict()

    y_hat_default = (y_proba >= 0.5).astype(int)
    naive_cost = expected_business_cost(y_true, y_hat_default, fn_cost, fp_cost)

    best["naive_threshold_cost_inr"] = naive_cost
    best["savings_vs_naive_inr"] = naive_cost - best["total_cost_inr"]
    best["savings_pct_vs_naive"] = (
        100.0 * best["savings_vs_naive_inr"] / max(naive_cost, 1.0)
    )
    best["theoretical_optimal_threshold"] = config.THEORETICAL_OPTIMAL_THRESHOLD
    return best
