# RetainIQ — Design Spec

> Status: locked  
> Date: 2026-05-24  
> Authors: ChurnZero 26 Round 2 team (2 members)  
> Competition: ChurnZero 26 (IIT Kharagpur, Unstop)

---

## 1. Mission (one sentence)

Beat the median submission by reframing the problem from *"who will churn"* to *"who can we save profitably"* — using a cost-aware ensemble, real causal uplift modelling on `retention_offer_received`, counterfactual recourse, and a fairness audit, packaged as a 15-slide exec-grade deck.

## 2. Hard constraints (from the official problem statement)

| Constraint | Value |
|---|---|
| Train rows | 8,101 (target included) |
| Test rows  | 2,026 (target hidden) |
| Features   | 97 across 8 categories |
| **Primary metric** | **PR-AUC** on held-out test labels |
| Secondary metric | F1 on the positive class |
| **Cost matrix** | **FN = ₹40,000**, **FP = ₹500** (80:1 ratio) |
| Slide cap | 15 slides total |
| Submission ZIP | `ChurnZero_<TeamName>.zip` containing exactly: |
|  | `ChurnZero_<TeamName>_Predictions.csv` (2,026 rows; cols `customer_id`, `churn_prediction` ∈ {0,1}, `churn_probability` ∈ [0,1]; no nulls) |
|  | `ChurnZero_<TeamName>_Presentation.pptx` (or PDF) — exec audience |
|  | `ChurnZero_<TeamName>_Code` — `.ipynb` / `.py` / `.Rmd` / `.R` runnable end-to-end |

### Scoring weights
- Model performance — **40%** (PR-AUC + F1 + business cost)
- Presentation quality — **25%**
- Methodology & rigour — **20%** (FE decisions, no leakage)
- Code quality — **15%** (reproducibility, readability, comments)

## 3. The single biggest scoring lever judges will miss

The **cost ratio is 80:1**. For perfectly calibrated probabilities, the cost-optimal decision threshold is:

```
predict churn  ⇔  P(churn) > 500 / (40000 + 500) ≈ 0.0123
```

The default 0.5 threshold every other team will use minimises **accuracy**, not **expected business cost**. We will:

1. Calibrate probabilities (isotonic on validation fold).
2. Sweep thresholds in [0.005, 0.95] and pick the one that minimises `(FP × ₹500) + (FN × ₹40,000)` on validation.
3. Show a single "cost slide" with rupees saved per 1,000 customers vs the naive baseline.

This single slide is the most defensible 25% of the Presentation block and bleeds into the 40% Model block.

## 4. Five-layer architecture (RetainIQ)

| Layer | Question | Method | Library |
|---|---|---|---|
| 1. Predictive | Who is at risk? | LightGBM + CatBoost stacked w/ logistic meta-learner; isotonic calibration | `lightgbm`, `catboost`, `scikit-learn` |
| 2. Cost-aware decisioning | What is the *correct* threshold? | Threshold sweep minimising expected ₹ cost | `scikit-learn` |
| 3. Causal | Who is *persuadable* (vs sure-thing / lost-cause / sleeping-dog)? | T-learner & X-learner using `retention_offer_received` as treatment | `causalml` or `econml` |
| 4. Prescriptive | What action would actually flip them? | DiCE counterfactuals on the predictive model | `dice-ml` |
| 5. Governance | Is it fair across Gender / Region? | Demographic parity + equal opportunity diff | `fairlearn` |

All five layers wrap into `src/retainiq/pipeline.py`. Layers 1 + 2 produce the **submission CSV**. Layers 3 + 4 + 5 power slides 10-12 (the differentiators).

## 5. Pipeline data flow

```
data/raw/ChurnZero_dataset_v1.csv ──┐
                                    ├─► data.py (load, schema, split) ──► features.py (FE) ──► models.py (train ensemble) ──► threshold.py (cost-optimal cutoff) ──┐
data/raw/ChurnZero_test_v1.csv  ────┘                                                                                                                              ├─► submit.py ──► submission/ChurnZero_<Team>_Predictions.csv
                                                                                                                                                                   │
                                                                                                                              uplift.py (T/X-learner) ──┬──► slides
                                                                                                                              counterfactuals.py        ├──► slides
                                                                                                                              fairness.py               └──► slides
```

## 6. Feature engineering plan (no-leakage discipline)

- **Drop pre-split**: `customer_id` (identifier only).
- **Redundancy resolution**: keep `income_band` *or* `income_category`, not both. Keep numeric `annual_income` raw + binned.
- **Unknown handling**: keep `Unknown` as its own category for `marital_status`, `education_level`, `customer_feedback_sentiment`, `competitor_bank_offer_awareness`. Do not impute as mode — it is itself a signal.
- **Missingness flag**: `app_rating_given` has many NaNs. Add `app_rating_given_isna` as a feature, then median-impute.
- **Engineered ratios** (all train-fit / test-transform safe):
  - `balance_to_income = avg_monthly_balance / max(annual_income, 1)`
  - `digital_share = digital_transaction_ratio` (already present; verify)
  - `complaint_pressure = total_complaints * (1 + escalation_count) / (1 + complaint_resolution_time)`
  - `engagement_decay = last_login_days / (1 + tenure_months)`
  - `q4_q1_combined_drop = (1 - total_amt_chng_q4_q1) + (1 - total_ct_chng_q4_q1)` (positive = decline)
  - `product_density = number_of_products / (1 + tenure_months)`
- **Categorical encoding**: pass categoricals natively to CatBoost; use one-hot or target-encode (with K-fold) for LightGBM.
- **No future leakage**: train/test are a customer hold-out (different ID ranges); no time-based features cross the split.

## 7. Modelling plan

| Step | Choice |
|---|---|
| Validation | Stratified 5-fold on `churn` |
| Base models | LightGBM (`scale_pos_weight = neg/pos`), CatBoost (categorical-aware) |
| Meta-learner | Logistic regression on out-of-fold base predictions |
| Calibration | Isotonic regression on val fold (post-stack) |
| Tuning | Optuna 30-trial budget per base model on PR-AUC |
| Seed | Pinned at 42 across all stochastic steps |

Target: **PR-AUC ≥ 0.55** on a held-out val fold (stretch: ≥0.65). Baseline LR PR-AUC will likely sit ~0.40.

## 8. Submission CSV contract

```csv
customer_id,churn_prediction,churn_probability
767114958,0,0.0123
708123033,1,0.4421
...
```

- 2,026 rows, no nulls
- `churn_prediction` ∈ {0, 1} at cost-optimal threshold
- `churn_probability` is the calibrated probability (4 decimal places)
- `customer_id` order matches test file order

## 9. Day-by-day plan (2-person team)

| Day | Person A (model) | Person B (deck + business) |
|---|---|---|
| **D1 (today)** | Run EDA notebook; baseline LR PR-AUC | Read problem statement; draft slides 1-5 outline |
| **D2** | LightGBM + CatBoost solo CV scores | Slides 1-5 first draft + EDA charts |
| **D3** | Stacked ensemble + isotonic calibration | Slide 8 (cost) + ROI math; slide 13 playbook |
| **D4** | Cost-optimal threshold; first valid submission CSV | Slides 9 (drivers), 13 (playbook) draft |
| **D5** | Uplift T/X learner on `retention_offer_received` | Slide 10 (uplift) writeup; rehearse |
| **D6** | DiCE counterfactual on 1 persuadable; fairness audit | Slides 11, 12, 14, 15; QA pass on whole deck |
| **D7** | Final hyperparameter tune + submission rebuild | Final QA + ZIP packaging + submit |

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Calibration is poor at extreme low probabilities | Use isotonic instead of Platt; validate calibration curve |
| `retention_offer_received` may have selection bias (banks target who they thought was at risk) | Acknowledge in slide 10; report Qini AND a propensity-stratified version |
| Optuna takes too long | Cap trials at 30; use `pruner=MedianPruner` |
| LightGBM categorical bug with many `Unknown` rows | Convert to `category` dtype explicitly; CatBoost as fallback |
| Submission format wrong → auto-rejected | Add `tests/test_submission_format.py`; assert before zipping |

## 11. Out of scope (YAGNI)

- Deep learning (no benefit on tabular ~10k rows)
- Real-time API
- Streamlit demo (Round 2 is PPT-only; we'll add it for Round 3 prep separately)
- Hyperparameter tuning beyond Optuna 30 trials
- Multiple uplift models — pick one (T-learner) and one validation (X-learner)

## 12. Definition of done (Round 2)

- [ ] `ChurnZero_<Team>_Predictions.csv` produced, format-validated, no nulls, 2,026 rows
- [ ] PR-AUC on internal val ≥ 0.55
- [ ] Cost-optimal threshold reduces expected ₹ cost ≥ 30% vs threshold=0.5
- [ ] 15-slide PDF, every slide named, every chart labelled
- [ ] Code zip runs end-to-end on a fresh checkout in < 10 min
- [ ] QA checklist (`reviews/qa_checklist.md`) signed by both team members
