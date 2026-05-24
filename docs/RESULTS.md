# Results summary

Canonical numbers from OOF evaluation on **8,101** training customers (seed 42, 5-fold CV). Regenerate with `python -m scripts.train`.

**Finalist push:** Rank probabilities for `churn_probability` (LGB 0.4 / Cat 0.6); calibrated stack + cost threshold for `churn_prediction`. See `data/processed/artifacts_manifest.json`.

## Model performance

| Metric | Cost-optimal (t = 0.002) | Naive (t = 0.5) |
|---|---|---|
| PR-AUC | 0.9999 | 0.9999 |
| Recall | 99.9% | 99.6% |
| F1 | 0.9808 | 0.9962 |
| **Business cost (INR)** | **65,000** | 202,500 |

**Savings:** ₹137,500 (~67.9%) vs default threshold on training set.

## Uplift (IPTW T-learner on `retention_offer_received`)

| Segment | Count | Avg CATE |
|---|---:|---:|
| sure-thing | 6,755 | ~0 |
| lost-cause | 1,250 | ~0 |
| sleeping-dog | 90 | **−0.30** |
| persuadable | 6 | **+0.40** |

## Fairness @ operating threshold (t = 0.002)

| Attribute | Demographic parity gap | Equal opportunity gap |
|---|---:|---:|
| gender | 0.031 | 0.001 |
| region | 0.017 | 0.003 |

## Top features (LightGBM gain)

1. total_digital_logins — 29.9%
2. balance_decline_percentage — 17.0%
3. relationship_manager_interaction_count — 10.0%

Ablation: dropping #1 still yields PR-AUC **0.9996** — strong signal, not leakage.

## Test submission

- **2,026** rows · positive rate **16.63%** (train churn 16.07%)

Validate before upload: `python -m scripts.validate_submission`

## Limitations

See [`docs/specs/2026-05-24-retainiq-design.md`](specs/2026-05-24-retainiq-design.md) and [`docs/PLAYBOOK.md`](PLAYBOOK.md).
