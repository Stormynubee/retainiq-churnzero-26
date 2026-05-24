"""Train and predict — called from scripts/train.py and scripts/predict.py."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
    print("loading train...")
    df_train = data.load_train()
    stats = data.quick_stats(df_train)
    print(f"  {stats['rows']} rows, churn rate {stats['churn_rate']:.4f}")

    X_raw, y = data.split_features_target(df_train)

    print("feature engineering (fit on train)...")
    X_fe, fe_state = features.fit_transform(X_raw)
    cat_idx = features.list_categorical_indices(X_fe, fe_state)
    print(f"  {X_fe.shape[1]} columns, {len(cat_idx)} categorical")

    splitter = data.stratified_folds(y)

    print("lightgbm 5-fold oof...")
    oof_lgb, lgb_models = models.train_lightgbm_oof(X_fe, y, splitter)

    print("catboost 5-fold oof...")
    oof_cat, cat_models = models.train_catboost_oof(X_fe, y, splitter, cat_idx)

    print("stacker + calibration...")
    meta = models.fit_meta(oof_lgb, oof_cat, y)
    oof_stacked = models.stacked_oof(oof_lgb, oof_cat, meta)
    calibrator = models.calibrate_isotonic(oof_stacked, y)
    oof_calibrated = calibrator.predict(oof_stacked)

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
    print(f"saved -> {BUNDLE_PATH}")
    return summary


def predict() -> Path:
    print("loading test...")
    df_test = data.load_test()
    test_ids = df_test[config.ID_COL].copy()
    X_raw = df_test.drop(columns=config.DROP_BEFORE_FEATURES)

    fe_state = joblib.load(FE_STATE_PATH)
    bundle: models.StackedBundle = joblib.load(BUNDLE_PATH)
    threshold_info = json.loads(THRESHOLD_PATH.read_text())
    threshold = float(threshold_info["threshold"])

    X_fe = features.transform(X_raw, fe_state)
    y_proba = models.predict_stacked(bundle, X_fe)

    sub = submit.build_submission(test_ids, y_proba, threshold)
    out = submit.write_submission(sub)
    print(f"wrote {out}")
    print(f"  threshold {threshold:.4f}, positive rate {sub['churn_prediction'].mean():.4f}")
    return out
