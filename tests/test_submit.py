"""Submission CSV contract."""

import numpy as np
import pandas as pd
import pytest

from retainiq import config
from retainiq.submit import build_submission, validate_submission


def test_build_submission_shape_and_columns():
    ids = pd.Series([101, 102, 103])
    proba = np.array([0.9, 0.1, 0.55])
    sub = build_submission(ids, proba, threshold=0.5)
    assert list(sub.columns) == [
        config.ID_COL,
        "churn_prediction",
        "churn_probability",
    ]
    assert sub["churn_prediction"].tolist() == [1, 0, 1]
    assert sub["churn_probability"].max() <= 1.0


def test_build_submission_dual_track_uses_calibrated_for_binary():
    ids = pd.Series([1, 2, 3])
    rank = np.array([0.1, 0.9, 0.5])
    cal = np.array([0.8, 0.2, 0.6])
    sub = build_submission(ids, rank, threshold=0.5, y_proba_calibrated=cal)
    assert sub["churn_prediction"].tolist() == [1, 0, 1]
    assert np.allclose(sub["churn_probability"].values, rank)


def test_build_submission_clips_out_of_range_probabilities():
    ids = pd.Series([1, 2])
    sub = build_submission(ids, np.array([-0.2, 1.5]), threshold=0.5)
    assert sub["churn_probability"].min() >= 0.0
    assert sub["churn_probability"].max() <= 1.0


def test_validate_submission_rejects_wrong_row_count():
    df = build_submission(pd.Series([1]), np.array([0.5]), 0.5)
    with pytest.raises(ValueError, match="expected 2026"):
        validate_submission(df, expected_rows=2026)


def test_validate_submission_rejects_duplicate_ids():
    df = pd.DataFrame(
        {
            config.ID_COL: [1, 1],
            "churn_prediction": [0, 1],
            "churn_probability": [0.1, 0.9],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(df, expected_rows=2)


def test_validate_submission_rejects_invalid_prediction_values():
    df = pd.DataFrame(
        {
            config.ID_COL: [1],
            "churn_prediction": [2],
            "churn_probability": [0.5],
        }
    )
    with pytest.raises(ValueError, match="0/1"):
        validate_submission(df, expected_rows=1)
