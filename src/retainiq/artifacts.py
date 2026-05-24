"""Canonical artifact paths and guards — single source for train/predict/deck."""

from __future__ import annotations

import json
from pathlib import Path

from . import config

# --- paths (resolved at call time so tests can monkeypatch config.DATA_PROCESSED) ---


def bundle_path() -> Path:
    return config.DATA_PROCESSED / "stacked_bundle.joblib"


def fe_state_path() -> Path:
    return config.DATA_PROCESSED / "fe_state.joblib"


def threshold_path() -> Path:
    return config.DATA_PROCESSED / "cost_optimal_threshold.json"


def metrics_path() -> Path:
    return config.DATA_PROCESSED / "training_metrics.json"


def cost_curve_path() -> Path:
    return config.DATA_PROCESSED / "cost_curve.csv"


def oof_path() -> Path:
    return config.DATA_PROCESSED / "oof_predictions.parquet"


def feature_importances_path() -> Path:
    return config.DATA_PROCESSED / "feature_importances.csv"


def manifest_path() -> Path:
    return config.DATA_PROCESSED / "artifacts_manifest.json"


def rank_stack_weights_path() -> Path:
    return config.DATA_PROCESSED / "rank_stack_weights.json"


def uplift_propensity_summary_path() -> Path:
    return config.DATA_PROCESSED / "uplift_propensity_summary.json"


def chart_dir() -> Path:
    return config.PROJECT_ROOT / "deck" / "charts"


def ensure_dirs() -> None:
    """Create output dirs from entrypoints — not on config import."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    chart_dir().mkdir(parents=True, exist_ok=True)


def trained_bundle_exists() -> bool:
    return (
        bundle_path().is_file()
        and fe_state_path().is_file()
        and threshold_path().is_file()
    )


def require_trained() -> None:
    missing = [
        p.name
        for p in (bundle_path(), fe_state_path(), threshold_path())
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing trained artifacts: {', '.join(missing)}. Run: python -m scripts.train"
        )


def write_manifest(extra: dict | None = None) -> Path:
    payload = {
        "team": config.TEAM_NAME,
        "seed": config.RANDOM_SEED,
        "fn_cost_inr": config.FN_COST,
        "fp_cost_inr": config.FP_COST,
    }
    if extra:
        payload.update(extra)
    out = manifest_path()
    out.write_text(json.dumps(payload, indent=2, default=float))
    return out
