# Judge Q&A — RetainIQ (internal prep)

Use this before live pitch or technical Q&A. Not marketing copy — talking points with file pointers.

**Team:** Hansraj Tiwari (ML / code) · swayangjeet nayak (deck / business)

---

## If asked: “Did you use AI?”

> We used coding assistants for boilerplate, tests, and docs. Design choices, external ML audits, and every number on the slides are ours — we can re-run the pipeline and explain each step live.

Do not claim “100% human-written.” Do not lead the deck with AI.

---

## Live demo (60–90 seconds)

**Fast path** (no competition CSVs):

```powershell
pip install -e ".[dev]"
python -m pytest tests/test_smoke_pipeline.py -q
```

**Full path** (with `data/raw/` CSVs):

```powershell
pip install -e .
python -m scripts.train
python -m scripts.predict
```

**Fallback** if train is too slow on stage laptop:

- Open `data/processed/training_metrics.json` (after a prior local train)
- Mention `tests/test_causal_leakage.py` and `tests/test_stacking_train.py` — methodology encoded as tests
- Show submission: `submission/ChurnZero_RetainIQ_Predictions.csv` — 2,026 rows, dual columns

---

## Three stories you must own (5 min each)

### 1. Cost threshold (Hansraj → slide 8)

- FN **₹40,000**, FP **₹500** → ratio 80:1 → naive t=0.5 is wrong for this cost matrix.
- We sweep thresholds on **OOF calibrated** probabilities and pick minimum rupee cost (~t ≈ 0.002).
- **Files:** `src/retainiq/threshold.py`, `data/processed/cost_curve.csv`, `deck/charts/08_cost_curve.png`

### 2. No leakage + honest stacking (Hansraj → slide 5 / appendix)

- Drop post-treatment cols (`retention_offer_accepted`, `discount_or_fee_waiver_received`) before features.
- Meta learner fit on **OOF** LGB/Cat preds, not in-sample ensemble averages.
- **Dual CSV:** rank blend → `churn_probability`; calibrated stack + cost threshold → `churn_prediction`.
- **Files:** `docs/CAUSAL_FIXES.md`, `src/retainiq/pipeline.py`, `src/retainiq/submit.py`

### 3. Uplift honesty (swayangjeet → slide 10)

- Offers were **not** randomized — observational T-learner.
- IPTW + propensity overlap trim; still directional, not RCT-grade.
- Only **6** persuadables in training — use segments to **order** actions, not as exact headcount.
- **Files:** `src/retainiq/uplift.py`, `data/processed/uplift_propensity_summary.json`, `deck/charts/10_uplift_quadrant.png`

---

## Ten likely judge questions

| # | Question | Short answer | Where to point |
|---|----------|--------------|----------------|
| 1 | Why is PR-AUC 0.9999? Isn’t that suspicious? | Dataset is very separable; we say that upfront. We differentiated on **cost** and **who to call**, not another 0.001 AUC. | `docs/RUN_NOTES.md`, ablation in run notes |
| 2 | Why threshold 0.002 instead of 0.5? | Cost-optimal for FN/FP weights; same model, ~68% lower OOF cost vs t=0.5. | `src/retainiq/threshold.py`, slide 8 |
| 3 | What’s in the submission CSV? | `churn_probability` = rank stack (PR-AUC column); `churn_prediction` = calibrated + cost threshold. | `src/retainiq/submit.py` |
| 4 | How do you avoid leakage? | Post-treatment cols dropped; FE fit on train only; meta on OOF; documented audits. | `docs/CAUSAL_FIXES.md`, `tests/test_causal_leakage.py` |
| 5 | Why keep `retention_offer_received` in the churn model? | Observed pre-decision assignment in train; dropped from uplift **features**; not forced to zero on test. | `docs/CAUSAL_FIXES.md` |
| 6 | How does uplift work? | T-learner: P(stay given offer) minus P(stay given no offer); IPTW for selection; segments for playbook. | `src/retainiq/uplift.py`, slide 10 |
| 7 | Only 6 persuadables — is uplift useless? | No — it tells you **not** to blast offers (90 sleeping-dogs). Small N is a **limitation** we state. | `docs/PLAYBOOK.md` |
| 8 | Fairness? | Gender DP gap ~3.1%, region ~1.7% at operating threshold; both under 10%. | `data/processed/fairness_summary.csv`, slide 12 |
| 9 | Can we reproduce your code? | `pip install -e .` → `python -m scripts.train` → `predict`; 58 pytest tests + CI. | `README.md`, `.github/workflows/test.yml` |
| 10 | What would you do in production? | Weekly scores, quarterly retrain, PSI on drivers, fairness gate before campaigns. | Slide 14, `docs/PLAYBOOK.md` |

---

## Speaker split (who talks when)

| Slides | Hansraj | swayangjeet |
|--------|---------|-------------|
| 1–2 | Intro + cost framing (together) | Cost “why it matters” emphasis |
| 3–4 | Data / segments if probed | Segment business meaning |
| 5–7 | Approach, model journey, metrics table | Tie metrics to rubric (PR-AUC primary) |
| 8 | **Own** cost curve | INR savings line for judges |
| 9 | Feature drivers (technical) | Which levers bank can pull |
| 10–11 | IPTW method if asked | Uplift quadrants + customer story |
| 12 | Fairness metrics if asked | Policy / gate before rollout |
| 13–15 | Repro command if asked | Playbook, ROI, close |

---

## Pre-pitch checklist

- [ ] Each person explains **one** fix from `docs/CAUSAL_FIXES.md` in their own words
- [ ] `python -m pytest -q` passes on pitch machine
- [ ] Deck numbers match `docs/RUN_NOTES.md` (re-train if not)
- [ ] `reviews/qa_checklist.md` signed for CSV + deck
- [ ] Know the honest AI-tool one-liner (top of this doc)

---

## Rubric reminder

| Block | Weight | Our lead |
|-------|--------|----------|
| Model | 40% | PR-AUC + cost @ optimal threshold |
| Deck | 25% | Slide 8 cost + slide 10 uplift |
| Methodology | 20% | CAUSAL_FIXES + tests |
| Code | 15% | Reproducible `scripts/train` |

See also: [`docs/specs/2026-05-24-retainiq-design.md`](specs/2026-05-24-retainiq-design.md)

---

## If they say it looks AI-generated

Do not argue or get defensive. Pivot to evidence:

1. Open [`docs/CAUSAL_FIXES.md`](CAUSAL_FIXES.md) — walk **FLAW 1** (post-treatment leakage) in your own words.
2. Run `python -m pytest tests/test_stacking_train.py tests/test_causal_leakage.py -q` (60s).
3. Show `data/processed/cost_curve.csv` + slide 8 chart — numbers come from OOF sweep, not slide copy.

**Roles:** Hansraj runs the demo path above. swayangjeet owns slide 8 INR line without reading bullets verbatim.

**RetainIQ name (15 sec):** *We called it RetainIQ because the product is retention intelligence—who to call and what it costs—not because the AUC is special.*

**Sparse git history:** Point reviewers to [`docs/ITERATION.md`](ITERATION.md) and [`docs/PROVENANCE.md`](PROVENANCE.md) — history was rewritten with `filter-repo`; phased work is in causal audits and tests.

**PR-AUC ~1.0:** Same as Q1 in the table — separable data, leakage ablation in RUN_NOTES, compete on cost and uplift.
