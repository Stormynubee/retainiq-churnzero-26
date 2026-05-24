"""Investigate why PR-AUC is suspiciously high.

Run after `scripts.train`. Loads the saved bundle, asks both base learners
for feature importance, and prints the top 25 + a single-feature ablation
that drops the most-suspicious column to see how badly PR-AUC degrades.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:  # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from retainiq import config, data, features  # noqa: E402
from retainiq.models import StackedBundle  # noqa: E402


def main() -> int:
    df_train = data.load_train()
    X_raw, y = data.split_features_target(df_train)
    fe_state = joblib.load(config.DATA_PROCESSED / "fe_state.joblib")
    X_fe = features.transform(X_raw, fe_state)

    bundle: StackedBundle = joblib.load(
        config.DATA_PROCESSED / "stacked_bundle.joblib"
    )

    # ---- LightGBM feature importance (gain, averaged across folds) ----
    importance = np.zeros(X_fe.shape[1], dtype=float)
    for m in bundle.lgb_models:
        importance += m.feature_importance(importance_type="gain")
    importance /= max(len(bundle.lgb_models), 1)

    imp_df = pd.DataFrame(
        {"feature": X_fe.columns, "lgb_gain": importance}
    ).sort_values("lgb_gain", ascending=False)
    imp_df["lgb_gain_pct"] = (
        100.0 * imp_df["lgb_gain"] / imp_df["lgb_gain"].sum()
    )

    out = config.DATA_PROCESSED / "feature_importances.csv"
    imp_df.to_csv(out, index=False)
    print(f"\nFeature importances saved to {out}\n")
    print("Top 25 features by mean LightGBM gain:")
    print(imp_df.head(25).to_string(index=False))

    # ---- Single-feature ablation: drop the top feature, refit one LGBM,
    #      compare PR-AUC. Big drop -> that feature is doing all the work
    #      (likely leakage). Small drop -> ensemble is healthy.
    import lightgbm as lgb

    top_feature = imp_df.iloc[0]["feature"]
    print(f"\nAblation: drop '{top_feature}' and refit a single LightGBM...")

    X_minus = X_fe.drop(columns=[top_feature])
    splitter = data.stratified_folds(y)
    oof = np.zeros(len(y))
    for tr, va in splitter.split(X_minus, y):
        train_set = lgb.Dataset(
            X_minus.iloc[tr], label=y.iloc[tr], free_raw_data=False
        )
        val_set = lgb.Dataset(
            X_minus.iloc[va],
            label=y.iloc[va],
            reference=train_set,
            free_raw_data=False,
        )
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "average_precision",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "verbose": -1,
                "seed": config.RANDOM_SEED,
                "force_col_wise": True,
            },
            train_set,
            num_boost_round=800,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(
            X_minus.iloc[va], num_iteration=model.best_iteration
        )

    pr_auc = average_precision_score(y, oof)
    roc_auc = roc_auc_score(y, oof)
    print(
        f"\nAblation result (without '{top_feature}'): "
        f"PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}"
    )

    summary = {
        "top_feature_removed": str(top_feature),
        "pr_auc_without_top_feature": float(pr_auc),
        "roc_auc_without_top_feature": float(roc_auc),
        "verdict": (
            "likely_leakage" if pr_auc < 0.85 else "feature_is_just_strong"
        ),
    }
    (config.DATA_PROCESSED / "ablation_audit.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nAudit verdict: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
