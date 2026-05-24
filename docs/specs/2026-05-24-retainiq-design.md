# RetainIQ — design notes

Hansraj Tiwari & swayangjeet nayak · ChurnZero 26 · 2026-05-24

Working notes from when we locked the approach. Not gospel — change if the data disagrees.

---

## What we're submitting

ZIP name: `ChurnZero_RetainIQ.zip`

1. `ChurnZero_RetainIQ_Predictions.csv` — 2,026 rows, cols `customer_id`, `churn_prediction`, `churn_probability`
2. `ChurnZero_RetainIQ_Presentation.pdf` (or pptx) — max 15 slides
3. Code — this repo / `notebooks/retainiq_story.py`

## Rubric (what judges care about)

| Block | Weight | Our angle |
|---|---|---|
| Model | 40% | PR-AUC + F1 + **business cost** |
| Deck | 25% | cost slide + uplift story |
| Methodology | 20% | no leakage, calibrated probs |
| Code | 15% | `python -m scripts.train` reproduces everything |

Primary metric: **PR-AUC**. Secondary: F1. Cost: FN ₹40,000, FP ₹500.

## The threshold thing

For calibrated scores, the Bayes-optimal cutoff is roughly:

```
t* ≈ 500 / (40000 + 500) ≈ 0.0123
```

Everyone defaults to 0.5. We sweep and pick the minimum-cost threshold on OOF predictions. That's the main "business" differentiator — the model is already saturated.

## Pipeline (actual code)

```
raw CSVs → data.py → features.py (fit train only)
         → models.py (LGB + CatBoost OOF, stack, calibrate)
         → threshold.py (cost sweep)
         → submit.py → predictions CSV

Side paths (deck only):
         → uplift.py (T-learner)
         → fairness.py (gender, region)
         → build_artifacts.py (PNG charts)
```

We skipped DiCE / fairlearn / causalml in code — uplift and fairness are hand-rolled so `pip install` stays painless on Windows.

## Feature engineering choices

- Drop `customer_id` before any model sees the data
- Keep `Unknown` as a real category (not imputed away)
- Add `*_isna` flags, median-fill numerics
- A few ratio features (engagement decay, complaint pressure, etc.) — see `features.py`

## Day plan (rough)

| Day | Hansraj | swayangjeet |
|---|---|---|
| D1 | baseline + EDA | slides 1–5 outline |
| D2 | LGB + CatBoost CV | draft slides 1–5 |
| D3 | stack + calibration | cost slide + playbook |
| D4 | threshold + CSV | drivers + playbook |
| D5 | uplift | slide 10 writeup |
| D6 | fairness + optional DiCE | slides 11–15, QA |
| D7 | retune if needed | ZIP + upload |

## Honest limitations (say these in the deck)

- PR-AUC ~1.0 — can't win on model score alone
- `retention_offer_received` is not a randomised experiment
- Only 6 persuadables in training — uplift is directionally right, not precise counts
- No production deployment in Round 2 — that's Round 3 if we get there
