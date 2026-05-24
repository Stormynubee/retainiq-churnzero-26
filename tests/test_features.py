"""Feature engineering fit/transform contract."""

import pandas as pd

from retainiq.features import fit_transform, transform


def test_fit_transform_then_transform_same_columns(minimal_feature_frame):
  X_train = minimal_feature_frame.iloc[:2]
  X_test = minimal_feature_frame.iloc[2:]

  X_tr_fe, state = fit_transform(X_train)
  X_te_fe = transform(X_test, state)

  assert list(X_te_fe.columns) == state.final_columns
  assert len(X_tr_fe.columns) == len(state.final_columns)


def test_unknown_categorical_becomes_category_not_nan(minimal_feature_frame):
  X_fe, _ = fit_transform(minimal_feature_frame)
  assert X_fe["gender"].isna().sum() == 0


def test_missing_numeric_gets_median_and_isna_flag(minimal_feature_frame):
  X_fe, state = fit_transform(minimal_feature_frame)
  assert "annual_income_isna" in state.isna_flag_cols
  assert X_fe["annual_income"].isna().sum() == 0
