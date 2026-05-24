# Run notes

Scratch pad for numbers we paste into slides. Regenerate by running `scripts/train` and `scripts/build_artifacts`.

---

## 2026-05-24 — finalist push (code shipped)

**Stacking fix:** meta trained on OOF `p_lgb`/`p_cat` (not in-sample ensemble averages).  
**Submission:** `churn_probability` = rank-averaged LGB/Cat (OOF-tuned weights in `rank_stack_weights.json`); `churn_prediction` = cost threshold on calibrated stack.  
**Uplift:** IPTW + propensity overlap trim → see `uplift_propensity_summary.json` after `build_artifacts`.  
**Fairness:** no per-group threshold shift (DP gaps still under 10%).

Re-run on full data when CSVs are in `data/raw/`:

```powershell
python -m scripts.train
python -m scripts.build_artifacts
python -m scripts.predict
```

Then paste fresh PR-AUC, threshold, rank weights, and uplift counts below.

---

## 2026-05-24 — local retrain (final numbers)

Stack: LightGBM + CatBoost (5-fold OOF) → meta on OOF → Platt calibration. Rank blend (LGB **0.4** / Cat **0.6**). Seed 42.

Train: 8,101 rows, 16.07% churn. Test: 2,026 rows.

### OOF metrics (train)

| | t = 0.002 (cost-optimal) | t = 0.5 |
|---|---|---|
| PR-AUC | 0.9999 | 0.9999 |
| Recall | 99.9% | 99.6% |
| F1 | 0.9808 | 0.9962 |
| Cost | **INR 65,000** | INR 202,500 |

Saves **INR 137,500 (~67.9%)** vs default threshold on the training set.

Test submission positive rate: **16.63%** (threshold 0.0020; train churn 16.07%).

### Uplift (IPTW T-learner)

| segment | n | avg CATE |
|---|---|---|
| sure-thing | 6,755 | ~0 |
| lost-cause | 1,250 | ~0 |
| sleeping-dog | 90 | **-0.30** |
| persuadable | 6 | **+0.40** |

Deck angle: 6 people where the offer clearly helps; 90 where it might hurt.

### Fairness @ t=0.002

| attribute | DP diff | EO diff |
|---|---|---|
| gender | 0.031 | 0.001 |
| region | 0.017 | 0.003 |

Both under 10% on demographic parity.

### Pre-upload

```powershell
python -m scripts.validate_submission
python -m scripts.package_submission
```

---

## 2026-05-24 — first full pipeline run

Stack: LightGBM + CatBoost (5-fold OOF) → logistic meta → isotonic calibration. Seed 42.

Train: 8,101 rows, 16.07% churn (1,302 positives). Test: 2,026 rows.

### OOF metrics (train)

| | t = 0.001 (cost-optimal) | t = 0.5 |
|---|---|---|
| PR-AUC | 0.9999 | 0.9999 |
| Recall | 100% | 99.6% |
| F1 | 0.9542 | 0.9977 |
| Cost | **INR 62,500** | INR 200,500 |

Saves **INR 138,000 (~69%)** vs default threshold on the training set.

Test submission positive rate: **17.47%** (train churn 16.07% — looks sane).

### Top features (LGB gain %)

1. total_digital_logins — 29.9  
2. balance_decline_percentage — 17.0  
3. relationship_manager_interaction_count — 10.0  
4. total_trans_count — 5.7  
5. campaign_response_count — 3.4  

### Leakage check

Dropped `total_digital_logins`, refit single LGBM: PR-AUC still 0.9996. Not leakage — dataset is just easy.

### Uplift (T-learner on `retention_offer_received`)

| segment | n | avg CATE |
|---|---|---|
| sure-thing | 6,707 | ~0 |
| lost-cause | 1,248 | ~0 |
| sleeping-dog | 141 | **-0.27** |
| persuadable | 4 | +0.37 |

Deck angle: only 4 people where the offer clearly helps; 141 where it might hurt.

### Fairness @ t=0.001

| attribute | DP diff | EO diff |
|---|---|---|
| gender | 0.027 | 0.000 |
| region | 0.013 | 0.000 |

Both under 10% on demographic parity.

### Charts

`deck/charts/08_cost_curve.png`, `09_feature_importance.png`, `10_uplift_quadrant.png`, `12_fairness.png`

### Still todo

- swayangjeet: PowerPoint/PDF → `deck/ChurnZero_RetainIQ_Presentation.pdf` then re-run `package_submission`
- optional: DiCE counterfactual for slide 11 (6 persuadables in training)
- Upload `submission/ChurnZero_RetainIQ.zip` to Unstop

### Slide copy-paste (swayangjeet)

- Slide 2: cost ratio **80:1** (40k vs 500)
- Slide 8: use cost curve PNG; **INR 137.5k saved**, threshold **0.002**
- Slide 10: **6** persuadables, **90** sleeping-dogs
- Slide 13: **INR 137.5k saved per 8,101 customers**
