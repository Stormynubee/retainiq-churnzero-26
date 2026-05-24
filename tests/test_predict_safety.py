"""Predict path must not crash when post-treatment columns are absent from test CSV."""

import pandas as pd

from retainiq import config
from retainiq.data import drop_inference_features


def test_drop_inference_features_ignores_missing_post_treatment_cols():
    df = pd.DataFrame(
        {
            config.ID_COL: ["t1", "t2"],
            "gender": ["M", "F"],
            "annual_income": [100.0, 200.0],
        }
    )
    X = drop_inference_features(df)
    assert config.ID_COL not in X.columns
    assert config.TREATMENT_COL_COMPLIANCE not in X.columns
    assert "gender" in X.columns
