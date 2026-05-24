"""Uplift segmentation rules (no model fitting)."""

import numpy as np
import pandas as pd
import pytest

from retainiq import config
from retainiq.uplift import segment_customers, split_treatment


def test_segment_sleeping_dog_when_cate_strongly_negative():
  cate = np.array([-0.1])
  churn = np.array([0.3])
  assert segment_customers(cate, churn).iloc[0] == "sleeping-dog"


def test_segment_persuadable_high_cate_and_above_median_risk():
  cate = np.array([0.2, 0.01])
  churn = np.array([0.9, 0.1])
  seg = segment_customers(cate, churn)
  assert seg.iloc[0] == "persuadable"


def test_segment_sure_thing_low_churn_low_cate():
  cate = np.array([0.0])
  churn = np.array([0.05])
  assert segment_customers(cate, churn).iloc[0] == "sure-thing"


def test_segment_lost_cause_high_churn_flat_cate():
  cate = np.array([0.0])
  churn = np.array([0.8])
  assert segment_customers(cate, churn).iloc[0] == "lost-cause"


def test_split_treatment_requires_column():
  X = pd.DataFrame({"x": [1, 2]})
  y = pd.Series([0, 1])
  with pytest.raises(KeyError):
    split_treatment(X, y, treatment_col=config.TREATMENT_COL_PRIMARY)
