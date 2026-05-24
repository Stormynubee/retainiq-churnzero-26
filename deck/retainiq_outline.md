# Deck outline — swayangjeet owns this

15 slides max. 16:9. Big font. Numbers in INR where possible.

**Arc:** we predict churn, but the interesting part is *who to call* and *what it costs*.

**Prep doc:** [`docs/JUDGE_QA.md`](../docs/JUDGE_QA.md) — demo script, Q&A, speaker split.

## Speaker ownership

| Person | Owns on stage |
|--------|----------------|
| **Hansraj Tiwari** | Pipeline (slides 5–7), cost curve mechanics (8), leakage/stacking if challenged, live repro |
| **swayangjeet nayak** | INR story (2, 8), uplift quadrants (10), customer story (11), fairness policy (12), playbook + ROI (13), close |

---

## Slide 1 — Title

**RetainIQ** — churn prediction with a retention playbook  
Hansraj Tiwari & swayangjeet nayak · ChurnZero 26 · IIT Kharagpur

Speaker: *"We didn't stop at AUC — we built the rupee math for who to call."*

---

## Slide 2 — Why cost matters

- Missed churner (FN): **INR 40,000**
- Unnecessary call (FP): **INR 500**
- Ratio **80:1** → default threshold 0.5 is wrong for this problem

Visual: simple 80:1 bar chart

---

## Slide 3 — Data snapshot

- 8,101 train, 97 features, ~16% churn
- Engagement / inactivity separates churners (violin or box on `last_login_days`)

Fill Y× from EDA when you build slides.

---

## Slide 4 — Segments

Churn clusters in low-tenure, low-engagement, complaint-heavy customers.  
Bar chart: churn rate by segment × tenure bucket.

---

## Slide 5 — Approach (one slide)

**Speaker: Hansraj**

- LightGBM + CatBoost, 5-fold OOF, logistic meta on OOF, Platt calibration
- Rank stack for `churn_probability`; calibrated stack + cost threshold for `churn_prediction`
- T-learner uplift on `retention_offer_received` with IPTW + overlap trim
- Fairness on gender + region
- Footer line: *Validated with pytest + external ML audit → fixes in `docs/CAUSAL_FIXES.md`*

---

## Slide 6 — Model journey

Show progression (even if everything ends near 0.99):  
logistic baseline → single LGB → stack → calibrated.

---

## Slide 7 — Results table

Copy from `docs/RUN_NOTES.md`:

| | t = 0.001 | t = 0.5 |
|---|---|---|
| PR-AUC | 0.9999 | 0.9999 |
| Recall | 100% | 99.6% |
| Cost | INR 62,500 | INR 200,500 |

Lead with PR-AUC as primary metric per rubric.

---

## Slide 8 — Cost curve (main slide)

**Speaker: Hansraj (chart) → swayangjeet (INR line)**

PNG: `deck/charts/08_cost_curve.png`

- Vertical line at 0.5 vs our threshold (~0.001)
- Caption: same model, **INR 138k saved** on 8,101 customers vs naive cutoff

Pause here — judges asked for business cost.

---

## Slide 9 — Drivers

PNG: `deck/charts/09_feature_importance.png`

Mark which levers the bank can actually change (logins, complaints, campaigns) vs structural (tenure).

---

## Slide 10 — Uplift

**Speaker: swayangjeet** (Hansraj covers IPTW only if asked)

PNG: `deck/charts/10_uplift_quadrant.png`

| Quadrant | What to do |
|---|---|
| Persuadable | Call — offer helps |
| Sure-thing | Don't waste budget |
| Lost-cause | Don't waste budget |
| Sleeping-dog | **Do not call** — CATE negative |

Numbers: **4** persuadables, **141** sleeping-dogs (from run notes).

Caveat: offers weren't randomised — say that out loud.  
Method: propensity scores clipped to [0.05, 0.95], overlap trim, IPTW-weighted T-learner (`uplift_propensity_summary.json` for n trimmed).

---

## Slide 11 — One customer story

Pick one persuadable from `data/processed/uplift_per_customer.csv`.  
Hand-write 2–3 changes that would lower their risk (login, complaint fix, offer accepted).  
DiCE optional — we didn't wire it into the repo yet.

---

## Slide 12 — Fairness

PNG: `deck/charts/12_fairness.png`

| | DP diff | EO diff |
|---|---|---|
| Gender | 0.027 | 0.000 |
| Region | 0.013 | 0.000 |

---

## Slide 13 — Playbook + ROI

| Segment | Action |
|---|---|
| Persuadable | RM call + waiver |
| Sleeping-dog | No contact |
| Lost-cause | Graceful exit, re-engage later |
| Sure-thing | Cross-sell instead |

ROI line: **INR 138k saved** on training cohort (scale to per-1L customers if asked).

---

## Slide 14 — If we shipped this

- Weekly batch scores to CRM
- Retrain quarterly; PSI on top features
- Risks: selection bias in offers, saturated AUC, RBI-style disclosure

---

## Slide 15 — Close

RetainIQ · questions

Contact emails if you want them on the slide.

---

## Assets

| Slide | File |
|---|---|
| 8 | `deck/charts/08_cost_curve.png` |
| 9 | `deck/charts/09_feature_importance.png` |
| 10 | `deck/charts/10_uplift_quadrant.png` |
| 12 | `deck/charts/12_fairness.png` |

Regenerate: `python -m scripts.build_artifacts`

---

## Appendix — Methodology backup (optional, not in 15-slide count)

Use only if a judge challenges leakage, calibration, or uplift.

**Speaker: Hansraj**

| Topic | One line |
|-------|----------|
| Post-treatment leakage | Dropped `retention_offer_accepted` and waiver cols before features |
| Calibration | Platt fit on **OOF meta** only — not in-sample |
| Stacking | Meta trained on OOF base preds; inference uses ensemble-averaged folds |
| Uplift | IPTW + propensity trim [0.05, 0.95]; offers still observational |

Source: [`docs/CAUSAL_FIXES.md`](../docs/CAUSAL_FIXES.md)
