"""Data loading and splits."""

import pandas as pd
import pytest

from retainiq import config
from retainiq.data import load_test, load_train, quick_stats, split_features_target


def test_split_features_target_drops_id_and_target():
  df = pd.DataFrame(
    {config.ID_COL: [1, 2], config.TARGET_COL: [0, 1], "annual_income": [1.0, 2.0]}
  )
  X, y = split_features_target(df)
  assert config.ID_COL not in X.columns
  assert config.TARGET_COL not in X.columns
  assert y.tolist() == [0, 1]


@pytest.mark.integration
def test_load_train_shape_and_churn_rate(raw_data_available):
  if not raw_data_available:
    pytest.skip("raw CSVs not on disk")
  df = load_train()
  stats = quick_stats(df)
  assert stats["rows"] == 8101
  assert 0.10 < stats["churn_rate"] < 0.25


@pytest.mark.integration
def test_load_test_no_target_column(raw_data_available):
  if not raw_data_available:
    pytest.skip("raw CSVs not on disk")
  df = load_test()
  assert len(df) == 2026
  assert config.TARGET_COL not in df.columns
