# Pre-submit QA

Both of us sign each section before uploading to Unstop.

**Automated (2026-05-24):** `validate_submission` OK (2026 rows, ~16.63% positive); integration tests pass locally; `package_submission` builds ZIP (add deck PDF before final upload).

---

## CSV

- [x] `ChurnZero_RetainIQ_Predictions.csv`
- [x] 2,026 rows
- [x] columns: `customer_id`, `churn_prediction`, `churn_probability` (in that order)
- [x] same `customer_id` order as test file (`python -m scripts.validate_submission`)
- [x] no nulls
- [x] predictions 0/1 only, probs in [0, 1]
- [x] threshold is cost-optimal on **calibrated** probs (t ≈ 0.002, not 0.5)
- [x] `churn_probability` is rank stack; `churn_prediction` from calibrated + cost threshold
- [x] positive rate roughly 10–45% (16.63%)

Hansraj: ______  swayangjeet: ______

---

## Deck (15 slides max)

- [ ] team name on slide 1
- [ ] PR-AUC called out as primary metric (slide 7)
- [ ] cost curve with INR (slide 8) — numbers: **137.5k saved**, t=**0.002**
- [ ] at least 3 actionable drivers (slide 9)
- [ ] uplift explained in plain English (slide 10) — **6** persuadables, **90** sleeping-dogs
- [ ] one persuadable example on slide 11 (even if hand-written)
- [ ] fairness numbers filled in (slide 12) — gender DP **0.031**, region **0.017**
- [ ] ROI in rupees on slide 13 — **INR 137.5k**
- [ ] spell-check + PDF export looks OK
- [ ] polish starter PPTX from `python -m scripts.build_deck_pptx` (see `deck/PPT_BUILD.md`; metrics auto-loaded from `training_metrics.json`)
- [ ] slide 11: customer **140348** (P(churn) 0.96, CATE +0.35) — rehearse persuadable band explanation
- [ ] save as `deck/ChurnZero_RetainIQ_Presentation.pdf` and re-run `package_submission`

Hansraj: ______  swayangjeet: ______

---

## Code

- [x] `pip install -e ".[dev]"` works
- [x] `python -m scripts.train` then `build_artifacts` then `predict` reproduces the CSV
- [x] `feature_importances.csv` exists after train (chart 09)
- [x] CI unit + smoke tests pass (`python -m pytest -m "not slow and not integration"`)
- [x] no secrets in repo
- [x] raw competition CSVs not committed

Hansraj: ______  swayangjeet: ______

---

## ZIP contents

`ChurnZero_RetainIQ.zip` (from `python -m scripts.package_submission`):

1. predictions CSV  
2. presentation PDF/pptx (add to `deck/` before final pack)  
3. code (`retainiq-code/` subset in ZIP)

Final check: unzip locally and open everything once.

Pitch prep: [`docs/JUDGE_QA.md`](../docs/JUDGE_QA.md) (15 min rehearsal).

Hansraj: ______  swayangjeet: ______  Date: ______

---

## Upload

- [ ] Upload `submission/ChurnZero_RetainIQ.zip` to Unstop
