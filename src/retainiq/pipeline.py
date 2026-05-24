"""End-to-end glue: train and predict entry points used by the CLI scripts."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Windows powershell defaults to cp1252 and dies on the rupee glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config, data, evaluate, features, models, submit
from .threshold import cost_curve, find_cost_optimal_threshold

BUNDLE_PATH = config.DATA_PROCESSED / "stacked_bundle.joblib"
THRESHOLD_PATH = config.DATA_PROCESSED / "cost_optimal_threshold.json"
METRICS_PATH = config.DATA_PROCESSED / "training_metrics.json"
COST_CURVE_PATH = config.DATA_PROCESSED / "cost_curve.csv"
OOF_PATH = config.DATA_PROCESSED / "oof_predictions.parquet"
FE_STATE_PATH = config.DATA_PROCESSED / "fe_state.joblib"


def train() -> dict:
    """Fit the full RetainIQ pipeline; return the headline metrics."""
    print("[RetainIQ] Loading training data...")
    df_train = data.load_train()
    stats = data.quick_stats(df_train)
    print(f"  Train: {stats['rows']} rows | churn_rate={stats['churn_rate']:.4f}")

    X_raw, y = data.split_features_target(df_train)

    print("[RetainIQ] Engineering features (fit on train)...")
    X_fe, fe_state = features.fit_transform(X_raw)
    cat_idx = features.list_categorical_indices(X_fe, fe_state)
    print(f"  After FE: {X_fe.shape[1]} cols | {len(cat_idx)} categorical")

    splitter = data.stratified_folds(y)

    print("[RetainIQ] Training LightGBM (5-fold OOF)...")
    oof_lgb, lgb_models = models.train_lightgbm_oof(X_fe, y, splitter)

    print("[RetainIQ] Training CatBoost (5-fold OOF)...")
    oof_cat, cat_models = models.train_catboost_oof(X_fe, y, splitter, cat_idx)

    print("[RetainIQ] Fitting logistic meta-learner...")
    meta = models.fit_meta(oof_lgb, oof_cat, y)
    oof_stacked = models.stacked_oof(oof_lgb, oof_cat, meta)

    print("[RetainIQ] Calibrating with isotonic regression...")
    calibrator = models.calibrate_isotonic(oof_stacked, y)
    oof_calibrated = calibrator.predict(oof_stacked)

    print("[RetainIQ] Searching cost-optimal threshold...")
    best = find_cost_optimal_threshold(y.values, oof_calibrated)
    threshold = float(best["threshold"])
    print(
        f"  Cost-optimal threshold: {threshold:.4f}  "
        f"(theoretical: {best['theoretical_optimal_threshold']:.4f})"
    )
    print(
        f"  INR saved vs naive 0.5: INR {best['savings_vs_naive_inr']:,.0f} "
        f"({best['savings_pct_vs_naive']:.1f}% reduction)"
    )

    # Headline metrics @ cost-optimal threshold
    metrics_optimal = evaluate.evaluate_at_threshold(y.values, oof_calibrated, threshold)
    metrics_naive = evaluate.evaluate_at_threshold(y.values, oof_calibrated, 0.5)
    print(f"  OOF @ optimal:  {evaluate.summarize(metrics_optimal)}")
    print(f"  OOF @ t=0.5  :  {evaluate.summarize(metrics_naive)}")

    # Persist artifacts so predict.py can run independently
    bundle = models.StackedBundle(
        lgb_models=lgb_models,
        cat_models=cat_models,
        meta_model=meta,
        calibrator=calibrator,
        cat_feature_indices=cat_idx,
        feature_columns=X_fe.columns.tolist(),
    )
    joblib.dump(bundle, BUNDLE_PATH)
    joblib.dump(fe_state, FE_STATE_PATH)

    THRESHOLD_PATH.write_text(json.dumps(best, indent=2))
    cost_curve(y.values, oof_calibrated).to_csv(COST_CURVE_PATH, index=False)

    pd.DataFrame(
        {
            "customer_index": np.arange(len(y)),
            "y_true": y.values,
            "p_lgb": oof_lgb,
            "p_cat": oof_cat,
            "p_stacked": oof_stacked,
            "p_calibrated": oof_calibrated,
        }
    ).to_parquet(OOF_PATH, index=False)

    summary = {
        "train_stats": stats,
        "metrics_optimal": metrics_optimal,
        "metrics_naive": metrics_naive,
        "threshold_info": best,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[RetainIQ] Saved bundle -> {BUNDLE_PATH}")
    print(f"[RetainIQ] Saved metrics -> {METRICS_PATH}")
    return summary


def predict() -> Path:
    """Score the test set and write the submission CSV."""
    print("[RetainIQ] Loading test set...")
    df_test = data.load_test()
    test_ids = df_test[config.ID_COL].copy()
    X_raw = df_test.drop(columns=config.DROP_BEFORE_FEATURES)

    print("[RetainIQ] Loading fitted FE state + bundle...")
    fe_state = joblib.load(FE_STATE_PATH)
    bundle: models.StackedBundle = joblib.load(BUNDLE_PATH)
    threshold_info = json.loads(THRESHOLD_PATH.read_text())
    threshold = float(threshold_info["threshold"])

    X_fe = features.transform(X_raw, fe_state)
    print(f"  Test FE shape: {X_fe.shape}")

    print("[RetainIQ] Scoring stacked model...")
    y_proba = models.predict_stacked(bundle, X_fe)

    print("[RetainIQ] Building submission CSV...")
    sub = submit.build_submission(test_ids, y_proba, threshold)
    out = submit.write_submission(sub)
    print(f"[RetainIQ] Wrote {out}")
    print(f"  Threshold used: {threshold:.4f}")
    print(
        f"  Predicted positive rate: "
        f"{sub['churn_prediction'].mean():.4f}  "
        f"(probability mean: {sub['churn_probability'].mean():.4f})"
    )
    return out
