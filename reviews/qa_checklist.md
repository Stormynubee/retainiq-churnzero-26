# RetainIQ — Pre-submission QA Checklist

> Both team members must sign every section before zipping the submission.  
> If any item is unchecked, **do not submit**.

---

## Section 1 — Submission CSV (the deliverable)

- [ ] File name is exactly `ChurnZero_RetainIQ_Predictions.csv`
- [ ] Exactly 2,026 rows (matches test file row count)
- [ ] Three columns in this order: `customer_id`, `churn_prediction`, `churn_probability`
- [ ] `customer_id` order matches the order of `ChurnZero_test_v1.csv`
- [ ] No NaN / null / empty cells (verified by `submit.validate_submission`)
- [ ] `churn_prediction` ∈ {0, 1} only
- [ ] `churn_probability` ∈ [0.0, 1.0] only, calibrated
- [ ] Threshold used to derive `churn_prediction` is the **cost-optimal** one (NOT 0.5)
- [ ] Sanity: predicted positive rate is in a reasonable range (0.10 – 0.45)

**Reviewer A (ML)**: ____________  **Reviewer B (Story)**: ____________

---

## Section 2 — Presentation PDF

- [ ] Exactly 15 slides (count title and final slide)
- [ ] Slide 1 has team name and "ChurnZero 26"
- [ ] Slide 7 reports **PR-AUC** as the primary metric (not ROC-AUC)
- [ ] Slide 8 (Cost Slide) shows the threshold curve with rupee values
- [ ] Slide 9 highlights at least 3 *actionable* features (not just `age` etc.)
- [ ] Slide 10 explains the uplift quadrant in plain English
- [ ] Slide 11 shows one named persuadable customer with concrete counterfactual
- [ ] Slide 12 fairness table is filled in with actual numbers
- [ ] Slide 13 shows ₹ROI per segment, not generic "saves money"
- [ ] No slide has > 30 words of body text
- [ ] Every chart has a title, axis labels, and units
- [ ] Spelling pass done (use Word/Google Docs grammar check)
- [ ] PDF export verified — no formatting glitches

**Reviewer A (ML)**: ____________  **Reviewer B (Story)**: ____________

---

## Section 3 — Code package

- [ ] Repo is committed (git status clean before zip)
- [ ] `README.md` quickstart section runs from a fresh checkout
- [ ] `requirements.txt` is pinned and complete
- [ ] `python -m scripts.train` completes without error in < 15 min
- [ ] `python -m scripts.predict` produces a valid submission CSV
- [ ] No hardcoded paths outside `src/retainiq/config.py`
- [ ] No Jupyter notebook with stale outputs from someone's local run
- [ ] No raw data in commits (verified by `.gitignore`)
- [ ] No API keys, no `.env` files committed
- [ ] All `print` statements are intentional (no debug prints left)

**Reviewer A (ML)**: ____________  **Reviewer B (Story)**: ____________

---

## Section 4 — Methodology / leakage audit

- [ ] `customer_id` is dropped before any feature engineering or encoder fitting
- [ ] Train/val/test splits stratified on the target
- [ ] Imputers and encoders are `.fit()` on train only, `.transform()` on test
- [ ] No use of test target anywhere in the pipeline
- [ ] OOF predictions (not in-fold) used for stacking and calibration
- [ ] Random seed pinned to 42 in every stochastic step
- [ ] Cross-validation results reported with mean ± std

**Reviewer A (ML)**: ____________  **Reviewer B (Story)**: ____________

---

## Section 5 — Final ZIP

The ZIP must be named `ChurnZero_RetainIQ.zip` and contain exactly three items:

1. `ChurnZero_RetainIQ_Predictions.csv`
2. `ChurnZero_RetainIQ_Presentation.pdf`  (or `.pptx`)
3. `ChurnZero_RetainIQ_Code.ipynb`        (or `.zip` of the repo if multi-file)

- [ ] ZIP opens cleanly on Windows AND Mac
- [ ] No extra files (no `__pycache__`, no `.DS_Store`, no `_MACOSX`)
- [ ] Total ZIP size < 50 MB
- [ ] One team member has tested unzipping on a clean machine

**Reviewer A (ML)**: ____________  **Reviewer B (Story)**: ____________

---

## Section 6 — Last-mile sanity check

- [ ] Reread the official problem statement once more (the PDF in `data/raw/`)
- [ ] Submission deadline confirmed and there are at least 4 hours of buffer
- [ ] Both team members agree the deck tells one coherent story end-to-end
- [ ] If anything in the QA failed and was "fixed quickly", **rerun the full ZIP build**

**Final sign-off**

- Person A signature: _________________________  Date/time: _____________
- Person B signature: _________________________  Date/time: _____________

> Once both signatures are present and every box ticked, submit on Unstop.
