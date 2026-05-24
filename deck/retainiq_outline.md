# RetainIQ — 15-Slide Deck Outline (Person B owns)

> Audience: non-technical executives + IIT KGP DS faculty + industry mentors  
> Total slides: 15 (cover + back matter included; problem statement says max 15)  
> Format: 16:9, large type (≥18pt), 1 chart per slide where possible  
> Tone: confident, business-first, jargon-free; numbers always in ₹

**The narrative arc:**
> *We didn't just predict churn. We built the rupee math that tells a bank exactly **whom to call, when, what to say, and what it will save them**.*

---

## Slide 1 — Title

**Heading**: RetainIQ — From Churn Prediction to a Persuadable-Customer Engine
**Subheading**: ChurnZero 26 · IIT Kharagpur · Team RetainIQ · 2026
**Visual**: clean cover with team names + minimal logo
**Speaker note (15 sec)**: *"In one line: most banks predict who will churn. We predict who is worth saving — and prove it in rupees."*

---

## Slide 2 — The ₹ problem this deck answers

**Heading**: Banks lose 15-25% of customers a year. The fix is not "more accuracy"; it's "better decisions."

| Bullet | One-line |
|---|---|
| ₹ at stake | Each missed churner costs the bank ₹40,000 (FN). Each unnecessary call costs ₹500 (FP). |
| Cost ratio | 80 : 1. The default classifier threshold of 0.5 is **catastrophically wrong** for this asymmetry. |
| Our claim | A cost-aware ensemble + causal targeting saves >₹X lakh per 1,000 customers vs the naive baseline. |

**Visual**: 80:1 cost-ratio bar (one tall red bar vs one tiny green bar)
**Speaker note**: *"Read the rubric: judges are explicitly evaluating us on rupee cost, not accuracy. Most teams will optimize for AUC and quietly lose this slide."*

---

## Slide 3 — Dataset & EDA Insight #1

**Heading**: 8,101 customers, 97 features, ~X% churn — and a clear behavioural fingerprint
**Visual**: violin or box plot — `last_login_days` and `account_inactive_days` by churn class
**Insight bullet (1 line)**: Inactive customers (>90 days) churn at Y× the base rate
**Speaker note**: *"Behaviour beats demographics — engagement decay is the single strongest churn signal."*

---

## Slide 4 — EDA Insight #2 — the segment view

**Heading**: Churn is concentrated in 3 segments: low-tenure, low-engagement, complaint-heavy
**Visual**: horizontal bar — churn rate by `customer_segment` × `tenure_bucket`
**Insight**: ~Z% of all churn comes from ~Y% of the customer base — targeting matters more than scale
**Speaker note**: *"This is why uplift modelling matters: the people most likely to churn are not the same as those most likely to be saved."*

---

## Slide 5 — Feature engineering decisions

**Heading**: From 97 raw features to ~110 leak-safe signals
**Bullets (4)**:
- Engineered ratios: `balance_to_income`, `engagement_decay`, `complaint_pressure`, `q4_q1_combined_drop`
- "Unknown" treated as a real category (it's a signal, not noise)
- Missingness flags: `app_rating_given_isna` (its absence is itself meaningful)
- All transformers `.fit()` on train only — zero leakage
**Visual**: a clean before/after schematic (raw → engineered)
**Speaker note**: *"Methodology = 20% of the score. Every transform is fit on train only. Reproducible end-to-end in <10 minutes."*

---

## Slide 6 — Model journey: from baseline to stacked ensemble

**Heading**: We tested 4 model classes; stacking + calibration won.
**Visual**: small horizontal bar chart of CV PR-AUC: LR → RF → LGBM → CatBoost → **Stacked + Calibrated**
**Bullet**: 5-fold stratified CV, isotonic calibration, seed pinned at 42
**Speaker note**: *"PR-AUC is the rubric's primary metric — not ROC-AUC. We optimize for the right thing."*

---

## Slide 7 — Final model performance

**Heading**: PR-AUC = X.XX · F1 = X.XX · ₹Cost reduced by X% vs naive baseline

**Table**:
| Metric | Cost-optimal threshold | Naive (t=0.5) | Δ |
|---|---|---|---|
| PR-AUC | 0.XX | 0.XX | – |
| F1     | 0.XX | 0.XX | +X% |
| Recall | 0.XX | 0.XX | +X% |
| ₹ cost (per 8,101 cust.) | ₹X.XX cr | ₹X.XX cr | **–X%** |

**Speaker note**: *"This is the 40% Model block. Cost-optimal threshold isn't a hyperparameter — it's a business decision derived from the rubric's own cost matrix."*

---

## Slide 8 — 🔑 The Cost Slide (the one judges will remember)

**Heading**: Default 0.5 threshold = ₹X cr wasted. Cost-optimal threshold = ₹X cr saved.
**Visual** (full-width): cost curve — x-axis = threshold (0 to 1), y-axis = ₹ cost. Vertical lines at:
  - threshold = 0.5 (the trap)
  - threshold = our cost-optimal value
  - threshold = theoretical optimum (₹500 / ₹40,500 ≈ 0.0123)
**Caption**: *Same model. Different threshold. ₹X.XX lakh saved per 1,000 customers.*
**Speaker note**: *"This is the single most important slide. Most teams won't even calculate it. Pause here."*

---

## Slide 9 — Top churn drivers (SHAP, plain English)

**Heading**: 5 drivers explain most churn — and 3 of them are intervenable
**Visual**: SHAP global summary, top 8 features
**Annotation per feature** (one line each):
1. `last_login_days` — engagement decay
2. `unresolved_complaint_count` — service failure
3. `digital_engagement_index` — sticky behaviour
4. `tenure_months` — early-life vulnerability
5. `competitor_bank_offer_awareness` — flight risk signal
**Marked as "intervenable"**: 1, 2, 3 (you can change these; 4 and 5 you cannot)
**Speaker note**: *"This separates predictive importance from actionable importance — the difference between insight and decision."*

---

## Slide 10 — 🔑 The causal layer (uplift)

**Heading**: Predictive ≠ savable. We segment customers by *causal effect* of intervention.
**Visual**: 2×2 quadrant chart — x = P(churn), y = CATE (causal effect of retention offer)

| Quadrant | Action |
|---|---|
| Persuadable (high P, high CATE) | Call → high ROI |
| Sure-thing (low P, low CATE)    | Skip — they'll stay anyway |
| Lost cause (high P, ~0 CATE)    | Skip — they'll leave anyway |
| Sleeping dog (any P, neg CATE)  | **Definitely skip** — calling makes them WORSE |

**Speaker note**: *"`retention_offer_received` is in the dataset, so we can do real causal modelling, not synthetic. T-learner. Note: offers were not randomly assigned, so we report propensity-stratified results too."*

---

## Slide 11 — 🔑 Per-customer recourse (counterfactual)

**Heading**: For one persuadable customer, what would actually save them?
**Visual**: a single customer card
> Customer 716574033, P(churn) = 0.61
> Counterfactual: if `last_login_days` ≤ 7 AND `retention_offer_accepted` = 1 AND `unresolved_complaint_count` = 0 → P(churn) drops to 0.14

**Bullet**: Three concrete, individual-level levers — not "encourage digital usage in Germany"
**Speaker note**: *"DiCE counterfactuals turn SHAP from insight into action."*

---

## Slide 12 — Fairness audit

**Heading**: The model is fair across Gender and Region (within 4/5ths rule)
**Table**:
| Attribute | Demographic-parity Δ | Equal-opportunity Δ | Pass? |
|---|---|---|---|
| Gender | 0.0X | 0.0X | ✅ |
| Region | 0.0X | 0.0X | ✅ |

**Speaker note**: *"Industry deployment requires this. Most students skip it. We don't."*

---

## Slide 13 — The Retention Playbook

**Heading**: 3 segments × 3 actions × ₹ROI

| Segment | Trigger rule | Action | Expected ROI |
|---|---|---|---|
| **High-value persuadables** | P(churn) > 0.5 AND CLV > P75 AND CATE > 0.10 | Dedicated RM call + 1-yr fee waiver | ₹X saved per ₹1 spent |
| **Silent churners** | last_login_days > 60 AND tenure > 24 mo | Re-engagement campaign + UPI cashback | ₹X saved per ₹1 spent |
| **Cross-sell defectors** | products = 1 AND digital_engagement > median | Targeted product bundle | ₹X new revenue per call |

**Speaker note**: *"This slide answers the question: 'On Monday morning, what does the retention team actually do?'"*

---

## Slide 14 — Implementation & risks

**Heading**: 6-week deployment path · 3 named risks · drift monitor on day 0
**Bullets**:
- Batch scoring weekly; export to CRM as next-best-action queue
- Drift monitor: PSI on top 10 features, alarm at PSI > 0.2
- Risks: (1) treatment selection bias — mitigated via propensity strat; (2) seasonality — flagged for re-train every quarter; (3) regulatory disclosure on automated decisions — model card delivered
**Speaker note**: *"Production-grade thinking — that's the difference between a class project and a hackathon winner."*

---

## Slide 15 — Closing

**Heading**: Same data. Same dataset. ₹X.XX lakh better outcome — because we asked the right question.
**Three takeaways**:
1. We predict **who can be saved**, not just who will churn
2. Cost-optimal threshold = the single biggest scoring lever
3. Causal + counterfactual = actionable, not just explainable
**Visual**: side-by-side card — *baseline approach* vs *RetainIQ*, in ₹
**Closing line**: *Thank you. Questions?*

---

## Visual specs (Person B → designer)

| Slide | Chart | Library / data file |
|---|---|---|
| 2 | Cost ratio bar | manual |
| 3 | Violin: last_login_days × churn | seaborn from `data/raw` |
| 4 | Bar: churn rate by segment × tenure | groupby + seaborn |
| 6 | Bar: PR-AUC progression | from `data/processed/training_metrics.json` |
| 8 | Cost curve (the killer slide) | from `data/processed/cost_curve.csv` |
| 9 | SHAP summary (top 8) | from `notebooks/retainiq_main.ipynb` |
| 10 | 2×2 uplift quadrant | from uplift `segmentation_report` |
| 11 | Counterfactual card | DiCE output |
| 12 | Fairness table | from `fairness.fairness_report` |

## Tone rules (do not violate)

- No "AUC". Every metric is **PR-AUC** unless we name ROC explicitly as a secondary.
- Every threshold/decision shows the **rupee impact**.
- "Customer", not "user".
- "Indian Rupee ₹" (not USD).
- 1 chart per slide max.
- No code on any slide.
