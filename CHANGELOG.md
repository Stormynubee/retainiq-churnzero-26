# Changelog

All notable changes to RetainIQ for ChurnZero 26.

## v1.0-submission — 2026-05-24

- Deck PDF, strict Unstop ZIP, pre-upload gate
- Metrics-driven PPTX from `training_metrics.json`
- Submission CSV: 2,026 rows, dual-track columns

## v0.3-finalist — 2026-05-24

- Meta learner on OOF LightGBM/CatBoost predictions (not in-sample ensemble means)
- `churn_probability` = rank blend; `churn_prediction` = calibrated + cost threshold
- IPTW T-learner uplift with propensity overlap trim

## v0.2-causal-fixes — 2026-05-24

- Removed post-treatment leakage (`retention_offer_accepted`, waiver cols)
- Platt calibration on OOF meta predictions
- Cost threshold tuned on OOF calibrated probabilities

## v0.1-baseline — 2026-05-24

- Initial pipeline: LGB + CatBoost, stacking, pytest suite, CI smoke train
- Feature engineering with train-only fit
