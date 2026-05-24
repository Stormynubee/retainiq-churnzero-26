# RetainIQ — Slide copy (ChurnZero 26)

Paste into PowerPoint or generate via `python -m scripts.build_deck_pptx`.  
Metrics: local retrain 2026-05-24 ([`docs/RUN_NOTES.md`](../docs/RUN_NOTES.md)).

Design: primary `#0b5394`, save `#2e7d32`, risk `#cc0000`, footnotes `#666666`.  
Font 28–32 pt titles, 18–22 pt body, max 5 bullets per slide.

---

## Slide 1 — Title

**On-slide title:** RetainIQ — Who to call, and what it costs

**Bullets:**
- Hansraj Tiwari & swayangjeet nayak
- ChurnZero 26 · IIT Kharagpur

**Visual:** none (title slide)

**Speaker notes:** We didn't stop at AUC. The pitch is rupee-optimal retention: who to call, who to leave alone, and what each mistake costs.

**Owner:** both

---

## Slide 2 — Why cost matters

**On-slide title:** The 80:1 problem

**Bullets:**
- Missed churner (false negative): **₹40,000**
- Unnecessary retention call (false positive): **₹500**
- Cost ratio **80:1** — default threshold 0.5 optimizes the wrong objective

**Visual:** text callout or simple two-bar graphic (80:1)

**Speaker notes:** Everyone ships 0.99 AUC here. What matters is whether your cutoff matches the bank's cost ratio. One missed churner costs eighty times one wasted call.

**Owner:** swayangjeet

---

## Slide 3 — Data

**On-slide title:** Dataset snapshot

**Bullets:**
- **8,101** training customers · **97** features · **16.1%** churn
- **2,026** test customers — submission rate **16.6%** (aligned with train)
- Inactivity and balance stress (`last_login_days`, balance decline) separate churners

**Visual:** optional violin/box on `last_login_days`

**Speaker notes:** One EDA point is enough. Don't overclaim. The data is separable; our edge is the operating point and playbook, not a magic feature.

**Owner:** Hansraj

---

## Slide 4 — Who churns

**On-slide title:** Where churn concentrates

**Bullets:**
- Higher churn: **low tenure**, **low digital engagement**, **open complaints**
- Lower churn: stable tenure with active digital footprint
- Segments guide playbook priority—not blanket campaigns

**Visual:** bar chart tenure bucket × churn rate (optional)

**Speaker notes:** These are the levers the bank can influence versus structural factors you monitor but cannot fix in one quarter.

**Owner:** swayangjeet

---

## Slide 5 — Approach

**On-slide title:** RetainIQ pipeline

**Bullets:**
- LightGBM + CatBoost, 5-fold OOF → meta on OOF → Platt calibration
- Dual CSV: rank blend → `churn_probability`; calibrated + cost threshold → `churn_prediction`
- IPTW T-learner uplift · fairness audit (gender, region)

**Visual:** simple flow diagram (optional)

**Speaker notes:** Post-treatment columns dropped before features. Meta trained on OOF preds, not in-sample averages. We documented fixes in CAUSAL_FIXES and can re-run the pipeline live.

**Owner:** Hansraj

---

## Slide 6 — Model journey

**On-slide title:** Model scores plateau — cutoff still matters

**Bullets:**
- Logistic baseline → LightGBM → stack → calibrated: PR-AUC ≈ **0.99** at each step
- Gains on AUC are flat on this dataset
- **Where we win:** cost-optimal cutoff + uplift + playbook

**Visual:** small progression table

**Speaker notes:** We tried logistic → LGB → stack; PR-AUC barely moved. The bank still picks a threshold—we optimized that. Lead with cost and who gets an offer.

**Owner:** Hansraj

---

## Slide 7 — Results

**On-slide title:** Out-of-fold results (primary: PR-AUC)

**Bullets:**
- PR-AUC **0.9999** (5-fold OOF, seed 42) — same at t=0.002 and t=0.5
- Business cost @ **t = 0.002:** **₹65,000** | @ **t = 0.5:** **₹202,500**
- Recall **99.9%** vs **99.6%** — F1 tradeoff acceptable for cost goal

**Visual:** table (on-slide)

**Speaker notes:** Lead with PR-AUC for the rubric, then pivot to cost. That's what moves P&L.

**Owner:** Hansraj

---

## Slide 8 — Hero: cost curve

**On-slide title:** Same model, different cutoff

**Bullets:**
- Threshold **0.002** (cost-optimal) vs **0.5** (naive default)
- **₹137,500 saved** on 8,101 customers (**~68%**) — same predictions, different operating point
- Pause here — this is the business slide

**Visual:** `deck/charts/08_cost_curve.png` (full width)

**Speaker notes:** Hansraj explains axes and optimal line. swayangjeet delivers the INR savings line. Pause ~20s on the INR line.

**Owner:** Hansraj + swayangjeet

---

## Slide 9 — Drivers

**On-slide title:** What drives churn (and what we can change)

**Bullets:**
- **Actionable:** digital logins, complaints, campaigns, RM interactions
- **Structural:** tenure — monitor, do not promise instant fixes
- Top gain: `total_digital_logins` (~30% of LGB importance)

**Visual:** `deck/charts/09_feature_importance.png`

**Speaker notes:** Tie each driver to a bank action. Judges want a playbook, not a feature list.

**Owner:** swayangjeet

---

## Slide 10 — Uplift

**On-slide title:** Who is worth calling?

**Bullets:**
- **6** persuadables — offer associated with higher stay (CATE **+0.40** avg)
- **90** sleeping-dogs — outreach may **increase** churn (CATE **−0.30** avg)
- Offers were **not** randomized — IPTW + overlap trim; use for **ordering**, not headcount

**Visual:** `deck/charts/10_uplift_quadrant.png`

**Speaker notes:** Say the caveat out loud: offers weren't randomized. Six persuadables is a small N—we use uplift to order actions, not as a census.

**Owner:** swayangjeet

---

## Slide 11 — One customer

**On-slide title:** Case study — persuadable segment

**Bullets:**
- Customer **140348** · P(churn) **0.96** · CATE **+0.35** (persuadable)
- High uplift in persuadable band — not a mass promotional blast
- Actions: targeted RM call · resolve complaints · timed waiver

**Visual:** none or simple card layout

**Speaker notes:** Pick one persuadable row from uplift_per_customer.csv. If P(churn) looks low on slide, explain: persuadable means high CATE in the risk band, not necessarily the highest churn score. Regenerate story JSON with pick_persuadable_story.

**Owner:** swayangjeet

---

## Slide 12 — Fairness

**On-slide title:** Fairness at operating threshold

**Bullets:**
- Demographic parity gap — gender: **3.1%** · region: **1.7%** (policy limit 10%)
- Equal opportunity gaps near zero
- Gate campaigns by segment **after** fairness check

**Visual:** `deck/charts/12_fairness.png`

**Speaker notes:** We flag at the cost-optimal threshold, not 0.5. Both groups under policy line.

**Owner:** swayangjeet

---

## Slide 13 — Playbook + ROI

**On-slide title:** Monday-morning playbook

**Bullets:**
- Persuadable → RM call + waiver
- Sleeping-dog → **no** promotional contact
- Lost-cause / sure-thing → low-touch or cross-sell
- ROI: **₹137.5k** saved on training cohort (~**₹17** per customer → ~**₹17L** per 1L customers)

**Visual:** segment → action table

**Speaker notes:** This is the slide judges remember if they forget everything else. Tie back to slide 8 INR number.

**Owner:** swayangjeet

---

## Slide 14 — If we shipped this

**On-slide title:** Production path

**Bullets:**
- Weekly batch scores to CRM · quarterly retrain · PSI on top features
- Reproducible: `python -m scripts.train` → **58** pytest tests + CI
- Limits: observational uplift, PR-AUC ~1.0 on this dataset, model governance / RBI disclosure

**Visual:** none

**Speaker notes:** Brief code credibility for 15% rubric weight—do not demo unless asked.

**Owner:** Hansraj

---

## Slide 15 — Close

**On-slide title:** RetainIQ

**Bullets:**
- Churn prediction + cost-optimal cutoff + uplift playbook
- Questions?
- github.com/Stormynubee/retainiq-churnzero-26

**Visual:** none

**Speaker notes:** Thank judges. Offer to show `validate_submission` or cost curve source if time.

**Owner:** both

---

## Appendix — Methodology backup (optional slide 16, not counted in 15)

**On-slide title:** Methodology (if challenged)

**Bullets:**
- Post-treatment leakage: dropped `retention_offer_accepted` and waiver cols before features
- Calibration: Platt on **OOF meta** only
- Stacking: meta on OOF base preds; inference uses ensemble-averaged folds
- Uplift: IPTW + propensity trim [0.05, 0.95]; still observational

**Owner:** Hansraj
