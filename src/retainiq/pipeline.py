"""Train and predict — called from scripts/train.py and scripts/predict.py."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import artifacts, config, data, evaluate, features, importance, models, submit
from .threshold import cost_curve, find_cost_optimal_threshold


@dataclass
class TrainOptions:
    train_path: Path | str | None = None
    n_splits: int = config.N_SPLITS
    lgb_num_boost_round: int = 2000
    lgb_early_stopping: int = 100
    cat_iterations: int = 2000
    cat_od_wait: int = 100


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def train(options: TrainOptions | None = None) -> dict:
    opts = options or TrainOptions()
    artifacts.ensure_dirs()

    print("loading train...")
    df_train = data.load_train(opts.train_path or config.TRAIN_CSV)
    stats = data.quick_stats(df_train)
    print(f"  {stats['rows']} rows, churn rate {stats['churn_rate']:.4f}")

    X_raw, y = data.split_features_target(df_train)

    print("feature engineering (fit on train)...")
    X_fe, fe_state = features.fit_transform(X_raw)
    cat_idx = features.list_categorical_indices(X_fe, fe_state)
    print(f"  {X_fe.shape[1]} columns, {len(cat_idx)} categorical")

    splitter = data.stratified_folds(y, n_splits=opts.n_splits)

    lgb_params = {
        "num_boost_round": opts.lgb_num_boost_round,
        "early_stopping": opts.lgb_early_stopping,
    }
    cat_params = {"iterations": opts.cat_iterations, "od_wait": opts.cat_od_wait}

    print(f"lightgbm {opts.n_splits}-fold oof...")
    oof_lgb, lgb_models = models.train_lightgbm_oof(
        X_fe, y, splitter, params=lgb_params
    )

    print(f"catboost {opts.n_splits}-fold oof...")
    oof_cat, cat_models = models.train_catboost_oof(
        X_fe, y, splitter, cat_idx, params=cat_params
    )

    print("stacker + calibration...")
    avg_lgb, avg_cat = models.ensemble_base_predictions(
        X_fe, lgb_models, cat_models, cat_idx
    )
    meta = models.fit_meta(avg_lgb, avg_cat, y)

    oof_meta = models.fit_meta_oof(oof_lgb, oof_cat, y, splitter)
    calibrator = models.calibrate_platt(oof_meta, y)
    oof_calibrated = models.predict_platt(calibrator, oof_meta)

    oof_stacked = models.stacked_oof(avg_lgb, avg_cat, meta)

    print("threshold sweep...")
    best = find_cost_optimal_threshold(y.values, oof_calibrated)
    threshold = float(best["threshold"])
    print(f"  optimal t={threshold:.4f} (theory ~{best['theoretical_optimal_threshold']:.4f})")
    print(
        f"  saves INR {best['savings_vs_naive_inr']:,.0f} vs t=0.5 "
        f"({best['savings_pct_vs_naive']:.1f}%)"
    )

    metrics_optimal = evaluate.evaluate_at_threshold(y.values, oof_calibrated, threshold)
    metrics_naive = evaluate.evaluate_at_threshold(y.values, oof_calibrated, 0.5)
    print(f"  OOF optimal: {evaluate.summarize(metrics_optimal)}")
    print(f"  OOF t=0.5:   {evaluate.summarize(metrics_naive)}")

    bundle = models.StackedBundle(
        lgb_models=lgb_models,
        cat_models=cat_models,
        meta_model=meta,
        calibrator=calibrator,
        cat_feature_indices=cat_idx,
        feature_columns=X_fe.columns.tolist(),
    )
    joblib.dump(bundle, artifacts.bundle_path())
    joblib.dump(fe_state, artifacts.fe_state_path())

    importance.export_feature_importances(bundle, X_fe)

    artifacts.threshold_path().write_text(json.dumps(best, indent=2))
    cost_curve(y.values, oof_calibrated).to_csv(artifacts.cost_curve_path(), index=False)

    pd.DataFrame(
        {
            "customer_index": np.arange(len(y)),
            "y_true": y.values,
            "p_lgb": oof_lgb,
            "p_cat": oof_cat,
            "p_stacked": oof_stacked,
            "p_calibrated": oof_calibrated,
        }
    ).to_parquet(artifacts.oof_path(), index=False)

    summary = {
        "train_stats": stats,
        "metrics_optimal": metrics_optimal,
        "metrics_naive": metrics_naive,
        "threshold_info": best,
    }
    artifacts.metrics_path().write_text(json.dumps(summary, indent=2, default=float))
    artifacts.write_manifest(
        {
            "git_commit": _git_commit(),
            "train_rows": stats["rows"],
            "n_splits": opts.n_splits,
            "optimal_threshold": threshold,
            "pr_auc": metrics_optimal["pr_auc"],
            "cost_optimal_inr": metrics_optimal["total_cost_inr"],
            "cost_naive_inr": metrics_naive["total_cost_inr"],
        }
    )
    print(f"saved -> {artifacts.bundle_path()}")
    return summary


def predict(test_path: Path | str | None = None) -> Path:
    artifacts.require_trained()

    print("loading test...")
    df_test = data.load_test(test_path or config.TEST_CSV)
    test_ids = df_test[config.ID_COL].copy()
    X_raw = df_test.drop(columns=config.DROP_BEFORE_FEATURES)

    fe_state = joblib.load(artifacts.fe_state_path())
    bundle: models.StackedBundle = joblib.load(artifacts.bundle_path())
    threshold_info = json.loads(artifacts.threshold_path().read_text())
    threshold = float(threshold_info["threshold"])

    X_fe = features.transform(X_raw, fe_state)
    y_proba = models.predict_stacked(bundle, X_fe)

    sub = submit.build_submission(test_ids, y_proba, threshold)
    out = submit.write_submission(sub)
    print(f"wrote {out}")
    print(f"  threshold {threshold:.4f}, positive rate {sub['churn_prediction'].mean():.4f}")
    return out


def train_cli() -> None:
    train()


def predict_cli() -> None:
    predict()
