"""PR-AUC, F1, and rupee cost at a given threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .threshold import expected_business_cost


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_hat = (np.asarray(y_proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_cost_inr": expected_business_cost(y_true, y_hat),
        "predicted_positive_rate": float(y_hat.mean()),
    }


def summarize(metrics: dict) -> str:
    return (
        f"PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"P={metrics['precision']:.3f} | "
        f"R={metrics['recall']:.3f} | "
        f"cost INR {metrics['total_cost_inr']:,.0f} @ t={metrics['threshold']:.4f}"
    )
