"""End-to-end checks on real artifacts and submission file."""

from pathlib import Path

import pandas as pd
import pytest

from retainiq import config
from retainiq.submit import validate_submission


@pytest.mark.integration
def test_submission_file_matches_competition_contract(raw_data_available):
  path = config.SUBMISSION_CSV
  if not path.is_file():
    pytest.skip("submission CSV not generated yet")
  sub = pd.read_csv(path)
  validate_submission(sub, expected_rows=2026)


@pytest.mark.integration
def test_test_ids_align_with_submission(raw_data_available):
  if not raw_data_available or not config.SUBMISSION_CSV.is_file():
    pytest.skip("need raw test + submission")
  test_df = pd.read_csv(config.TEST_CSV)
  sub = pd.read_csv(config.SUBMISSION_CSV)
  assert test_df[config.ID_COL].tolist() == sub[config.ID_COL].tolist()


@pytest.mark.integration
def test_training_metrics_json_sane_if_present():
  path = config.DATA_PROCESSED / "training_metrics.json"
  if not path.is_file():
    pytest.skip("run scripts.train first")
  import json

  metrics = json.loads(path.read_text())
  opt = metrics["metrics_optimal"]
  assert opt["pr_auc"] > 0.9
  assert opt["total_cost_inr"] <= metrics["metrics_naive"]["total_cost_inr"]


@pytest.mark.slow
def test_predict_script_writes_valid_submission(trained_artifacts_available, tmp_path):
  if not trained_artifacts_available:
    pytest.skip("trained artifacts missing")
  if not config.TEST_CSV.is_file():
    pytest.skip("test CSV missing")

  import json
  import joblib

  from retainiq import data, features, models, submit

  df_test = data.load_test()
  test_ids = df_test[config.ID_COL]
  X_raw = df_test.drop(columns=config.DROP_BEFORE_FEATURES)
  fe_state = joblib.load(config.DATA_PROCESSED / "fe_state.joblib")
  bundle = joblib.load(config.DATA_PROCESSED / "stacked_bundle.joblib")
  threshold = float(
    json.loads((config.DATA_PROCESSED / "cost_optimal_threshold.json").read_text())[
      "threshold"
    ]
  )

  X_fe = features.transform(X_raw, fe_state)
  proba = models.predict_stacked(bundle, X_fe)
  out = tmp_path / "pred.csv"
  submit.write_submission(
    submit.build_submission(test_ids, proba, threshold),
    path=out,
  )
  validate_submission(pd.read_csv(out), expected_rows=2026)
