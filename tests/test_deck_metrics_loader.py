"""Deck metrics loader reads processed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retainiq import config
from retainiq.deck_metrics import load_deck_metrics


@pytest.fixture
def processed_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    (tmp_path / "training_metrics.json").write_text(
        json.dumps(
            {
                "metrics_optimal": {
                    "threshold": 0.002,
                    "pr_auc": 0.9999,
                    "total_cost_inr": 65000.0,
                    "recall": 0.999,
                },
                "metrics_naive": {
                    "threshold": 0.5,
                    "pr_auc": 0.9999,
                    "total_cost_inr": 202500.0,
                    "recall": 0.996,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "uplift_segmentation.csv").write_text(
        "segment,n,avg_cate,avg_churn_proba\n"
        "persuadable,6,0.40,0.64\n"
        "sleeping-dog,90,-0.30,0.53\n",
        encoding="utf-8",
    )
    (tmp_path / "fairness_summary.csv").write_text(
        "attribute,demographic_parity_diff,equal_opportunity_diff\n"
        "gender,0.031,0.001\n"
        "region,0.017,0.003\n",
        encoding="utf-8",
    )


def test_load_deck_metrics(processed_metrics: None) -> None:
    m = load_deck_metrics()
    assert m.threshold == 0.002
    assert m.savings_inr == 137500
    assert m.persuadable_n == 6
    assert m.sleeping_dog_n == 90
    assert m.gender_dp_gap_pct == 3.1
