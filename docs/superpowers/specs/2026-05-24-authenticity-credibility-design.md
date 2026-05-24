# Authenticity and credibility — design spec

Hansraj Tiwari & swayangjeet nayak · ChurnZero 26 · 2026-05-24

## Principles

1. **Credibility over camouflage** — show audits, tests, and file paths; do not claim "100% human."
2. **Cost-first narrative** — PR-AUC ~1.0 is reported once with context (separable data + leakage checks), not as the headline hedge.
3. **Internal vs public** — assistant disclosure stays in [`docs/JUDGE_QA.md`](../JUDGE_QA.md) only; not on slides.
4. **No git theater** — document `filter-repo` and phased work; do not fabricate commits.

## Banned phrases (public surfaces)

- `the model is saturated`
- `PR-AUC saturated` (as slide title)
- `hero moment`
- `You can't win Round 2 on AUC alone`
- `game-changer`, `leverage`, `robust`, `comprehensive solution`

Enforced by `tests/test_authenticity_phrases.py`.

## Approved voice

**README lead:** PR-AUC is ~1.0 on this dataset (see `tests/test_causal_leakage.py` and leakage ablation in RUN_NOTES). We competed on rupee cost at the operating point and who should get an offer.

**Slide 6 title:** Model scores plateau — cutoff still matters

**Slide 6 speaker:** We tried logistic → LGB → stack; PR-AUC barely moved. The bank still picks a threshold—we optimized that.

## If asked: "PR-AUC is 0.9999 — suspicious?"

> The dataset separates easily—we show that in RUN_NOTES with a leakage ablation. We still report PR-AUC for the rubric, but our differentiation is the cost curve at t=0.002 and the uplift segments.

Point to: `data/processed/cost_curve.csv`, `tests/test_causal_leakage.py`, slide 8.

## If asked: "Did you use AI?"

Use the existing one-liner in JUDGE_QA. Then demo: `pytest tests/test_stacking_train.py -q` or open CAUSAL_FIXES flaw 1.

## If they say it looks AI-generated

Do not argue. Show external audit fixes in CAUSAL_FIXES and run a 60s validate/demo path. swayangjeet owns slide 8 INR line in own words.

## RetainIQ name (15 sec)

> We called it RetainIQ for retention intelligence—who to call and what it costs—not because the AUC is special.

## Speaker overlap

| Topic | Lead | Backup |
|-------|------|--------|
| Stacking / leakage | Hansraj | swayangjeet reads CAUSAL_FIXES |
| Slide 8 INR | swayangjeet | Hansraj explains axes |
| Uplift caveat | swayangjeet | Hansraj explains IPTW |
