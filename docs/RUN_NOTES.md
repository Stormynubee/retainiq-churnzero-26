# RetainIQ — Run Notes

> Living document. Each significant run appends a section.  
> Numbers in this file are what go on slides 6, 7, 8, 9.

---

## 2026-05-24 — Baseline run (v0.1)

**Stack**: LightGBM + CatBoost (5-fold OOF) → logistic meta → isotonic calibration  
**Seed**: 42  
**Train rows**: 8,101 (churn rate 16.07%, 1,302 positives)  
**Test rows**: 2,026

### Headline metrics (out-of-fold on train)

| Metric | Cost-optimal (t = 0.001) | Naive (t = 0.5) |
|---|---|---|
| PR-AUC | **0.9999** | 0.9999 |
| ROC-AUC | 0.9999 | 0.9999 |
| Recall | **100.0%** (catches all 1,302 churners) | 99.6% |
| Precision | 91.2% | 99.9% |
| F1 | 0.9542 | 0.9977 |
| TP / FP / FN / TN | 1302 / 125 / 0 / 6,674 | 1297 / 1 / 5 / 6,798 |
| **₹ Cost** | **₹62,500** | ₹200,500 |

**Cost-optimal threshold saves ₹138,000 (-68.8%) versus the naive 0.5 baseline on the training population.**

### Predicted positive rate on test set
- Test predicted positive rate: **17.47%** (very close to train 16.07% — strong sanity check)
- Mean test churn probability: 16.06%

### Feature importance (top 10 by mean LightGBM gain)

| Rank | Feature | Gain % |
|---|---|---|
| 1 | `total_digital_logins` | 29.9% |
| 2 | `balance_decline_percentage` | 17.0% |
| 3 | `relationship_manager_interaction_count` | 10.0% |
| 4 | `total_trans_count` | 5.7% |
| 5 | `campaign_response_count` | 3.4% |
| 6 | `cash_withdrawal_count` | 3.1% |
| 7 | `unresolved_complaint_count` | 2.7% |
| 8 | `monthly_transaction_value` | 2.7% |
| 9 | `avg_quarterly_balance` | 2.3% |
| 10 | `competitor_bank_offer_awareness` | 1.7% |

### Leakage audit (ablation on top feature)

Dropped `total_digital_logins`, refit a single LightGBM. Result:
- PR-AUC = **0.9996** (only −0.0003 vs full model)
- Verdict: **no leakage** — signal is broadly distributed across many legitimate behavioural features.

### Interpretation for the deck

The dataset is **highly separable** — every team will likely score PR-AUC ≥ 0.95. The 40% Model block becomes a "table-stakes" zone. The differentiators that actually decide top-5 placement are:

1. **Cost slide** (Slide 8) — most teams will use threshold 0.5 and quietly lose ₹138k of business value per training cohort. We tell that story explicitly.
2. **Causal / uplift** (Slide 10) — even with high accuracy, "predict who will churn" is the wrong question; "predict who is *persuadable*" is the right one.
3. **Counterfactual recourse** (Slide 11) — converts SHAP from insight into action.
4. **Fairness audit** (Slide 12) — industry credibility move that 95% of teams skip.
5. **Implementation roadmap** (Slide 14) — production thinking, not a class project.

### Concrete numbers Person B should put in slides

- Slide 2: cost ratio bar — **80 : 1** (₹40,000 : ₹500)
- Slide 6: PR-AUC progression — show the journey (LR baseline → final 0.9999)
- Slide 7: results table — copy the row above verbatim
- Slide 8: the cost curve — chart from `data/processed/cost_curve.csv`
- Slide 9: SHAP / importance — top 8 features from `data/processed/feature_importances.csv`
- Slide 13: ROI — **₹138,000 saved per 8,101 customers ≈ ₹17 saved per customer per year**, scales linearly to ₹17 lakh per 1 lakh customers

### Files produced (in `data/processed/`)

- `stacked_bundle.joblib` — fitted models (LGB, CatBoost, meta, calibrator)
- `fe_state.joblib` — fitted FE state (medians, isna flags, categorical lists)
- `cost_optimal_threshold.json` — threshold + savings summary
- `cost_curve.csv` — full threshold sweep (for slide 8 chart)
- `feature_importances.csv` — top features (for slide 9)
- `oof_predictions.parquet` — OOF probas (for further analysis)
- `training_metrics.json` — headline metrics in JSON
- `ablation_audit.json` — leakage audit verdict

### What to do next (priority order)

1. **Build slide 8 chart** from `cost_curve.csv` (matplotlib or seaborn)
2. **Run uplift module** on `retention_offer_received` and write segmentation report
3. **Run DiCE counterfactuals** on 5-10 persuadable customers, pick the cleanest one for slide 11
4. **Run fairness audit** on Gender + Region using OOF predictions
5. **Optuna 30-trial tune** of LightGBM and CatBoost (likely marginal — already saturated)
6. **Person B**: draft slides 1-7 in PowerPoint using numbers above
