# Causal & calibration fixes

Review findings from external ML audits — tracked in two waves.

## Wave 1 (2026-05-24) — four flaws ✅

### FLAW 1 — Post-treatment leakage ✅

**Problem:** `retention_offer_accepted` and `discount_or_fee_waiver_received` are post-treatment outcomes. Using them in the churn model or T-learner features introduces collider bias.

**Fix:**
- `POST_TREATMENT_COLS` added to `config.py` → dropped before feature engineering
- Removed `retention_offer_effective` ratio from `features.py`
- `uplift.py` drops treatment + all post-treatment cols from `mu_1` / `mu_0` features

**Note:** `retention_offer_received` remains in the **churn model** as an observed historical assignment variable (not used in uplift feature space). Uplift still uses it only to split treated/control arms. Test CSV keeps observed offer flags (no forced zeroing).

### FLAW 2 — In-sample calibration leakage ✅

**Problem:** Isotonic calibrator was fit on in-sample meta-model predictions.

**Fix:**
- `fit_meta_oof()` produces truly out-of-fold stacked probabilities
- Platt calibrator fit on **OOF meta** predictions only
- Threshold tuning uses `oof_calibrated` from OOF path

### FLAW 3 — Stacking distribution shift ✅

**Problem:** Meta trained on single-fold OOF base preds but inference averaged all K fold models.

**Fix:**
- `meta_model` fit on **OOF** `p_lgb` / `p_cat` (not in-sample ensemble averages)
- `ensemble_base_predictions()` averages all fold models at inference
- `predict_stacked()` uses same averaging before meta + Platt
- OOF parquet `p_stacked` = OOF meta (not leaky ensemble stack)

### FLAW 4 — Isotonic step resolution ✅

**Problem:** Isotonic regression creates flat steps; unstable at t ≈ 0.001 cost-optimal threshold.

**Fix:**
- Replaced with **Platt scaling** (logistic on logit probabilities)
- Legacy isotonic bundles still load via `apply_calibration()` backward compat

---

## Wave 2 — Finalist push (2026-05-24) ✅

### Predict crash on missing post-treatment columns ✅

**Problem:** `predict()` dropped `DROP_BEFORE_FEATURES` without `errors="ignore"` → `KeyError` when test CSV omits optional columns.

**Fix:** `data.drop_inference_features()` used in `pipeline.predict()`.

### Meta trained on leaky ensemble averages (regression from wave 1) ✅

**Problem:** `fit_meta(avg_lgb, avg_cat, y)` used in-sample ensemble averages on full train.

**Fix:** `fit_meta(oof_lgb, oof_cat, y)`; inference still pairs OOF-trained meta with `ensemble_base_predictions()`.

### Dual-track submission for PR-AUC ✅

**Problem:** Calibrated probabilities optimize rupee cost at extreme threshold; rank blending can help PR-AUC (40% rubric weight).

**Fix:**
- `rank_average_probabilities()` (scipy rank blend) with weights tuned on OOF PR-AUC → `rank_stack_weights.json`
- `churn_probability` = rank stack; `churn_prediction` = threshold on **calibrated** stack

### IPTW uplift for observational offers ✅

**Problem:** T-learner treated all rows equally despite selection into offers.

**Fix:**
- Propensity `P(offer|X)` clipped to [0.05, 0.95]
- Overlap trim drops extreme propensities before fitting
- IPTW `sample_weight` on both `mu_1` and `mu_0`
- Summary → `uplift_propensity_summary.json` (deck slide 10)

### Fairness threshold shifting — skipped

OOF demographic parity gaps remain **under 10%** (gender ~3.1%, region ~1.7%). No per-group threshold adjustment on submission.

---

## Re-train required

After pulling finalist changes, re-run on competition CSVs:

```powershell
python -m scripts.train
python -m scripts.build_artifacts
python -m scripts.predict
```

Update deck numbers from `docs/RUN_NOTES.md` and `data/processed/artifacts_manifest.json`.
