"""Drop the top LGB feature and see if PR-AUC collapses — leakage check."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from retainiq import artifacts, config, data, features  # noqa: E402
from retainiq.importance import export_feature_importances  # noqa: E402
from retainiq.models import StackedBundle  # noqa: E402


def main():
    df_train = data.load_train()
    X_raw, y = data.split_features_target(df_train)
    fe_state = joblib.load(artifacts.fe_state_path())
    X_fe = features.transform(X_raw, fe_state)

    bundle: StackedBundle = joblib.load(artifacts.bundle_path())

    imp_df = export_feature_importances(bundle, X_fe)
    out = artifacts.feature_importances_path()
    print(f"saved {out}\n")
    print(imp_df.head(25).to_string(index=False))

    import lightgbm as lgb

    top_feature = imp_df.iloc[0]["feature"]
    print(f"\nablation: drop '{top_feature}', refit one LGBM...")

    X_minus = X_fe.drop(columns=[top_feature])
    splitter = data.stratified_folds(y)
    oof = np.zeros(len(y))
    for tr, va in splitter.split(X_minus, y):
        train_set = lgb.Dataset(X_minus.iloc[tr], label=y.iloc[tr], free_raw_data=False)
        val_set = lgb.Dataset(
            X_minus.iloc[va], label=y.iloc[va], reference=train_set, free_raw_data=False
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
        oof[va] = model.predict(X_minus.iloc[va], num_iteration=model.best_iteration)

    pr_auc = average_precision_score(y, oof)
    roc_auc = roc_auc_score(y, oof)
    print(f"without '{top_feature}': PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")

    summary = {
        "top_feature_removed": str(top_feature),
        "pr_auc_without_top_feature": float(pr_auc),
        "roc_auc_without_top_feature": float(roc_auc),
        "verdict": "likely_leakage" if pr_auc < 0.85 else "feature_is_just_strong",
    }
    (artifacts.feature_importances_path().parent / "ablation_audit.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
