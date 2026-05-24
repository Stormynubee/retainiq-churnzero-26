# RetainIQ — ChurnZero 26 (IIT Kharagpur)

> A causal, cost-aware retention engine for banking churn.  
> Round 2 submission for [ChurnZero 26](https://unstop.com/competitions/churnzero-26-iit-kharagpur-1686181) (IIT Kharagpur, 2026).

---

## TL;DR

We don't just predict churn — we predict **who is worth saving**.

Five layers:
1. **Predictive**: LightGBM + CatBoost stacked ensemble
2. **Cost-aware**: threshold tuned for FN=₹40,000 vs FP=₹500 (80:1 cost ratio)
3. **Causal**: uplift modelling using `retention_offer_received` as a real treatment
4. **Prescriptive**: per-customer counterfactual recourse with DiCE
5. **Governance**: fairness audit on Gender + Region

Primary metric: **PR-AUC** (not ROC-AUC — read the rubric carefully).

---

## Quickstart

```powershell
# 1. (one time) install Python deps
pip install -r requirements.txt

# 2. train + write submission
python -m scripts.train
python -m scripts.predict

# Outputs:
#   submission/ChurnZero_RetainIQ_Predictions.csv   <- the deliverable
#   data/processed/oof_predictions.parquet          <- for stacking / debugging
#   data/processed/feature_importances.csv          <- for slide 9
```

## Repo layout

```
churnzero-26/
├── README.md                    <- you are here
├── requirements.txt             <- pinned deps
├── data/
│   ├── raw/                     <- original CSVs + problem PDF
│   └── processed/               <- generated artifacts (not committed)
├── docs/specs/                  <- the design spec (the source of truth)
├── src/retainiq/                <- importable modules
│   ├── config.py                <- paths, cost matrix, seeds
│   ├── data.py                  <- load + split + leakage guards
│   ├── features.py              <- feature engineering
│   ├── models.py                <- LightGBM, CatBoost, stacked ensemble
│   ├── threshold.py             <- cost-aware threshold sweep
│   ├── evaluate.py              <- PR-AUC, F1, business cost
│   ├── uplift.py                <- T-learner / X-learner (Layer 3)
│   ├── fairness.py              <- demographic parity (Layer 5)
│   ├── submit.py                <- writes submission.csv per spec
│   └── pipeline.py              <- end-to-end orchestrator
├── scripts/
│   ├── train.py                 <- python -m scripts.train
│   └── predict.py               <- python -m scripts.predict
├── notebooks/
│   └── retainiq_main.ipynb      <- the polished narrative for judges
├── deck/
│   └── retainiq_outline.md      <- 15-slide outline + speaker notes
├── reviews/
│   └── qa_checklist.md          <- pre-submission QA gate
└── submission/                  <- final ZIP target
```

## Team-split (2 people)

| Person | Owns | Files |
|---|---|---|
| **A (ML lead)** | Model + submission CSV + reproducibility | `src/`, `scripts/`, `notebooks/` |
| **B (Story lead)** | Deck + business case + QA + ZIP packaging | `deck/`, `reviews/`, `submission/` |

Daily 15-min sync: A shares latest val PR-AUC; B shares latest slide draft. End-of-day commit on both branches.

## Key design choices (read these before changing code)

1. **PR-AUC, not ROC-AUC** — rubric is explicit. We never report ROC-AUC as primary.
2. **Cost matrix** — `FN_COST=40000`, `FP_COST=500` lives in `src/retainiq/config.py`. Single source of truth.
3. **No leakage** — `customer_id` dropped before any FE. All transformers `.fit()` on train only, `.transform()` on test.
4. **Calibrated probabilities** — `submission.csv` reports calibrated probabilities so any post-hoc cost analysis is honest.
5. **Reproducibility** — `RANDOM_SEED=42` everywhere. Pinned `requirements.txt`. Submission is byte-identical across runs.

## Status / progress

See `docs/specs/2026-05-24-retainiq-design.md` for the locked spec, day-by-day plan, and definition of done.

## License

Educational submission for ChurnZero 26. Not for redistribution.
