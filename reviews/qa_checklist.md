# Pre-submit QA

Both of us sign each section before uploading to Unstop.

---

## CSV

- [ ] `ChurnZero_RetainIQ_Predictions.csv`
- [ ] 2,026 rows
- [ ] columns: `customer_id`, `churn_prediction`, `churn_probability` (in that order)
- [ ] same `customer_id` order as test file
- [ ] no nulls
- [ ] predictions 0/1 only, probs in [0, 1]
- [ ] threshold is cost-optimal (not 0.5)
- [ ] positive rate roughly 10–45%

Hansraj: ______  swayangjeet: ______

---

## Deck (15 slides max)

- [ ] team name on slide 1
- [ ] PR-AUC called out as primary metric (slide 7)
- [ ] cost curve with INR (slide 8)
- [ ] at least 3 actionable drivers (slide 9)
- [ ] uplift explained in plain English (slide 10)
- [ ] one persuadable example on slide 11 (even if hand-written)
- [ ] fairness numbers filled in (slide 12)
- [ ] ROI in rupees on slide 13
- [ ] spell-check + PDF export looks OK

Hansraj: ______  swayangjeet: ______

---

## Code

- [ ] `pip install -e ".[dev]"` works
- [ ] `python -m scripts.train` then `build_artifacts` then `predict` reproduces the CSV
- [ ] `feature_importances.csv` exists after train (chart 09)
- [ ] CI unit + smoke tests pass
- [ ] no secrets in repo
- [ ] raw competition CSVs not committed

Hansraj: ______  swayangjeet: ______

---

## ZIP contents

`ChurnZero_RetainIQ.zip`:

1. predictions CSV  
2. presentation PDF/pptx  
3. code (notebook or this repo subset)

Final check: unzip locally and open everything once.

Hansraj: ______  swayangjeet: ______  Date: ______
