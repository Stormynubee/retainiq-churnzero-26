# RetainIQ Finalist Push — executed 2026-05-24

Implementation record for the finalist plan (source plan in Cursor, not committed).

## Shipped

| Phase | Deliverable |
|-------|-------------|
| P0 | Meta on OOF; `drop_inference_features`; OOF `p_stacked` = OOF meta |
| P1 | `rank_average_probabilities`, OOF weight tune, dual-track `build_submission` |
| P2 | IPTW + propensity trim in `fit_t_learner`; `uplift_propensity_summary.json` |
| P3 | Tests (57+), docs, `scipy` dependency |
| P4 | Fairness threshold shifting **skipped** (DP gaps under 10%) |

## Predict flow

1. Ensemble-average LGB + CatBoost on test  
2. Calibrated stack → cost-optimal threshold → `churn_prediction`  
3. Rank stack → `churn_probability`

## Re-train

Place `ChurnZero_dataset_v1.csv` and `ChurnZero_test_v1.csv` under `data/raw/`, then:

```powershell
python -m scripts.train
python -m scripts.build_artifacts
python -m scripts.predict
```
