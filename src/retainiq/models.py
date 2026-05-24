"""LightGBM + CatBoost stack with isotonic calibration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import config


@dataclass
class StackedBundle:
    lgb_models: list
    cat_models: list
    meta_model: LogisticRegression
    calibrator: IsotonicRegression
    cat_feature_indices: list[int]
    feature_columns: list[str]


def train_lightgbm_oof(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: StratifiedKFold,
    params: dict | None = None,
) -> tuple[np.ndarray, list]:
    import lightgbm as lgb

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    base_params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "seed": config.RANDOM_SEED,
        "force_col_wise": True,
    }
    if params:
        base_params.update({k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping")})

    num_boost_round = int((params or {}).get("num_boost_round", 2000))
    early_stopping = int((params or {}).get("early_stopping", 100))

    oof = np.zeros(len(y))
    models: list = []

    for tr, va in splitter.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        val_set = lgb.Dataset(X_va, label=y_va, reference=train_set, free_raw_data=False)

        model = lgb.train(
            base_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X_va, num_iteration=model.best_iteration)
        models.append(model)

    return oof, models


def train_catboost_oof(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: StratifiedKFold,
    cat_features: list[int],
    params: dict | None = None,
) -> tuple[np.ndarray, list]:
    from catboost import CatBoostClassifier, Pool

    base_params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5,
        "random_seed": config.RANDOM_SEED,
        "verbose": False,
        "auto_class_weights": "Balanced",
        "od_type": "Iter",
        "od_wait": 100,
    }
    if params:
        base_params.update(params)

    oof = np.zeros(len(y))
    models: list = []

    X_str = X.copy()
    for idx in cat_features:
        col = X.columns[idx]
        X_str[col] = X_str[col].astype(str)

    for tr, va in splitter.split(X_str, y):
        X_tr, X_va = X_str.iloc[tr], X_str.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)

        model = CatBoostClassifier(**base_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va] = model.predict_proba(val_pool)[:, 1]
        models.append(model)

    return oof, models


def fit_meta(oof_lgb: np.ndarray, oof_cat: np.ndarray, y: pd.Series) -> LogisticRegression:
    Z = np.column_stack([oof_lgb, oof_cat])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_SEED)
    meta.fit(Z, y)
    return meta


def stacked_oof(
    oof_lgb: np.ndarray,
    oof_cat: np.ndarray,
    meta: LogisticRegression,
) -> np.ndarray:
    Z = np.column_stack([oof_lgb, oof_cat])
    return meta.predict_proba(Z)[:, 1]


def calibrate_isotonic(p_uncal: np.ndarray, y_true: pd.Series) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_uncal, y_true)
    return iso


def predict_stacked(bundle: StackedBundle, X_test: pd.DataFrame) -> np.ndarray:
    X_test = X_test[bundle.feature_columns]

    lgb_preds = np.mean(
        [m.predict(X_test, num_iteration=m.best_iteration) for m in bundle.lgb_models],
        axis=0,
    )

    X_str = X_test.copy()
    for idx in bundle.cat_feature_indices:
        col = X_str.columns[idx]
        X_str[col] = X_str[col].astype(str)
    cat_preds = np.mean(
        [m.predict_proba(X_str)[:, 1] for m in bundle.cat_models],
        axis=0,
    )

    Z = np.column_stack([lgb_preds, cat_preds])
    p_uncal = bundle.meta_model.predict_proba(Z)[:, 1]
    return bundle.calibrator.predict(p_uncal)
