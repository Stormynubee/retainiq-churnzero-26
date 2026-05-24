# Causal & calibration fixes (2026-05-24)

Review findings from external ML audit — all four issues were **valid** and are now fixed.

## FLAW 1 — Post-treatment leakage ✅

**Problem:** `retention_offer_accepted` and `discount_or_fee_waiver_received` are post-treatment outcomes. Using them in the churn model or T-learner features introduces collider bias.

**Fix:**
- `POST_TREATMENT_COLS` added to `config.py` → dropped before feature engineering
- Removed `retention_offer_effective` ratio from `features.py`
- `uplift.py` drops treatment + all post-treatment cols from `mu_1` / `mu_0` features

**Note:** `retention_offer_received` remains in the **churn model** as an observed historical assignment variable (not used in uplift feature space). Uplift still uses it only to split treated/control arms.

## FLAW 2 — In-sample calibration leakage ✅

**Problem:** Isotonic calibrator was fit on in-sample meta-model predictions.

**Fix:**
- `fit_meta_oof()` produces truly out-of-fold stacked probabilities
- Platt calibrator fit on **OOF meta** predictions only
- Threshold tuning uses `oof_calibrated` from OOF path

## FLAW 3 — Stacking distribution shift ✅

**Problem:** Meta trained on single-fold OOF base preds but inference averaged all K fold models.

**Fix:**
- `ensemble_base_predictions()` averages all fold models (train + inference)
- Final `meta_model` fit on ensemble-averaged base predictions
- `predict_stacked()` uses same averaging before meta + Platt

## FLAW 4 — Isotonic step resolution ✅

**Problem:** Isotonic regression creates flat steps; unstable at t ≈ 0.001 cost-optimal threshold.

**Fix:**
- Replaced with **Platt scaling** (logistic on logit probabilities)
- Legacy isotonic bundles still load via `apply_calibration()` backward compat

## Re-train required

After pulling these changes, re-run:

```powershell
python -m scripts.train
python -m scripts.build_artifacts
python -m scripts.predict
```

OOF metrics and optimal threshold may shift slightly — update deck numbers from `docs/RUN_NOTES.md`.
