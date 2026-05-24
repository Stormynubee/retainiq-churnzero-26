"""Fast end-to-end train on synthetic fixture — no competition CSVs."""

from pathlib import Path

import pytest

from retainiq import artifacts, config
from retainiq.pipeline import TrainOptions, train

FIXTURE_TRAIN = Path(__file__).parent / "fixtures" / "mini_train.csv"


@pytest.mark.slow
def test_smoke_train_writes_core_artifacts(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    submission = tmp_path / "submission"
    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    monkeypatch.setattr(config, "SUBMISSION_DIR", submission)

    summary = train(
        TrainOptions(
            train_path=FIXTURE_TRAIN,
            n_splits=2,
            lgb_num_boost_round=60,
            lgb_early_stopping=15,
            cat_iterations=60,
            cat_od_wait=15,
        )
    )

    assert summary["metrics_optimal"]["pr_auc"] > 0.35
    assert summary["threshold_info"]["total_cost_inr"] <= summary["threshold_info"]["naive_threshold_cost_inr"]
    assert artifacts.bundle_path().is_file()
    assert artifacts.fe_state_path().is_file()
    assert artifacts.feature_importances_path().is_file()
    assert artifacts.manifest_path().is_file()
    assert artifacts.rank_stack_weights_path().is_file()
    assert artifacts.metrics_path().is_file()
