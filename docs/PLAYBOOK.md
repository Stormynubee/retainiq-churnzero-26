# Retention playbook

What RetainIQ recommends **Monday morning** if this model were deployed — tied to our OOF numbers on 8,101 customers.

## 1. Score everyone

Run `python -m scripts.predict` (or `retainiq-predict` after `pip install -e .`).

Output per customer:

- `churn_probability` — rank-averaged LGB/Cat scores (OOF-tuned weights; optimizes PR-AUC ranking)
- `churn_prediction` — binary flag from **calibrated** stack at **cost-optimal threshold** (0.002), not 0.5

## 2. Prioritize by rupee impact

| Action | Who | Why |
|---|---|---|
| **Call first** | High `churn_probability` + **persuadable** uplift segment | Offer actually helps (6 in training) |
| **Standard retention** | High churn score, not sleeping-dog | FN costs ₹40,000 each |
| **Do not push offers** | **Sleeping-dog** segment (90 in training) | CATE ≈ −0.30 — outreach may increase churn |
| **Low touch** | Sure-thing / lost-cause | Model says churn risk low or offer won't move them |

## 3. Operating threshold

Cost matrix: **FN ₹40,000** · **FP ₹500** (80:1).

At t = 0.002 on OOF: **₹65,000** total cost vs **₹202,500** at t = 0.5 — **~68% savings** with same model.

## 4. Fairness gate

Before rolling out campaigns by segment, check `data/processed/fairness_summary.csv`:

- Gender demographic parity gap: **3.1%**
- Region demographic parity gap: **1.7%**

Both under our **10%** policy line at the operating threshold.

## 5. Honest limits (say these out loud)

- PR-AUC ≈ 0.9999 — you cannot win Round 2 on model score alone.
- `retention_offer_received` is **not** a randomized experiment; uplift uses IPTW + overlap trim but counts remain directional.
- Only **6** persuadables in training — use uplift for **policy ordering**, not precise headcount.
- OOF threshold is optimistic vs a true holdout; recalibrate in production.

## Commands

```powershell
pip install -e ".[dev]"
python -m scripts.train
python -m scripts.build_artifacts   # uplift + fairness + deck charts
python -m scripts.audit_features    # optional top-feature ablation
python -m scripts.predict
```

See [`docs/RESULTS.md`](RESULTS.md) for headline metrics and [`docs/RUN_NOTES.md`](RUN_NOTES.md) for slide copy-paste values.
