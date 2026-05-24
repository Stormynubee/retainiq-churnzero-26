"""Fairness audit — Layer 5 of RetainIQ.

We compute two industry-standard fairness diagnostics across the
sensitive attributes declared in `config.SENSITIVE_COLS`:

    1. Demographic parity difference
       max_g  P(yhat=1 | group=g)  -  min_g  P(yhat=1 | group=g)

    2. Equal opportunity difference
       max_g  TPR(group=g)  -  min_g  TPR(group=g)

Both should be small (close to 0). Threshold of 0.10 is the common
"4/5ths-style" rule of thumb from US-EEOC tradition; we report numbers
and let the deck contextualize.

Implementation note: we deliberately avoid `fairlearn` so the audit
runs even on environments where fairlearn fails to install. If
fairlearn is available, the same metrics line up with their API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _per_group(
    df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Helper: per-group counts of (yhat positive rate, TPR)."""
    rows = []
    for value, sub in df.groupby(group_col):
        n = len(sub)
        if n == 0:
            continue
        positives_mask = sub["y_true"] == 1
        n_pos = int(positives_mask.sum())
        tpr = (
            float((sub.loc[positives_mask, "y_pred"] == 1).mean())
            if n_pos > 0
            else float("nan")
        )
        rows.append(
            {
                "group_attribute": group_col,
                "group_value": str(value),
                "n": n,
                "predicted_positive_rate": float((sub["y_pred"] == 1).mean()),
                "tpr": tpr,
                "actual_churn_rate": float(positives_mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def fairness_report(
    df_eval: pd.DataFrame,
    sensitive_cols: list[str] = config.SENSITIVE_COLS,
) -> dict:
    """Return per-group + summary fairness metrics.

    `df_eval` must contain columns: y_true, y_pred (0/1), and each col in
    `sensitive_cols`. The function tolerates missing sensitive columns.
    """
    per_group_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for col in sensitive_cols:
        if col not in df_eval.columns:
            continue
        g = _per_group(df_eval, col)
        if g.empty:
            continue
        per_group_frames.append(g)

        ppr_diff = float(
            g["predicted_positive_rate"].max() - g["predicted_positive_rate"].min()
        )
        tpr_diff = float(g["tpr"].max() - g["tpr"].min())
        summaries.append(
            {
                "attribute": col,
                "demographic_parity_diff": ppr_diff,
                "equal_opportunity_diff": tpr_diff,
                "passes_dp_10pct_rule": bool(ppr_diff <= 0.10),
                "passes_eo_10pct_rule": bool(tpr_diff <= 0.10),
            }
        )

    return {
        "per_group": (
            pd.concat(per_group_frames, ignore_index=True)
            if per_group_frames
            else pd.DataFrame()
        ),
        "summary": pd.DataFrame(summaries),
    }
