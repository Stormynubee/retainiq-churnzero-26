# Results summary

Canonical numbers from OOF evaluation on **8,101** training customers (seed 42, 5-fold CV). Regenerate with `python -m scripts.train`.

## Model performance

| Metric | Cost-optimal (t ≈ 0.001) | Naive (t = 0.5) |
|---|---|---|
| PR-AUC | 0.9999 | 0.9999 |
| Recall | 100% | 99.6% |
| F1 | 0.9542 | 0.9977 |
| **Business cost (INR)** | **62,500** | 200,500 |

**Savings:** ₹138,000 (~69%) vs default threshold on training set.

## Uplift (T-learner on `retention_offer_received`)

| Segment | Count | Avg CATE |
|---|---:|---:|
| sure-thing | 6,707 | ~0 |
| lost-cause | 1,248 | ~0 |
| sleeping-dog | 141 | **−0.27** |
| persuadable | 4 | **+0.37** |

## Fairness @ operating threshold

| Attribute | Demographic parity gap | Equal opportunity gap |
|---|---:|---:|
| gender | 0.027 | 0.000 |
| region | 0.013 | 0.000 |

## Top features (LightGBM gain)

1. total_digital_logins — 29.9%
2. balance_decline_percentage — 17.0%
3. relationship_manager_interaction_count — 10.0%

Ablation: dropping #1 still yields PR-AUC **0.9996** — strong signal, not leakage.

## Test submission

- **2,026** rows · positive rate **17.47%** (train churn 16.07%)

## Limitations

See [`docs/specs/2026-05-24-retainiq-design.md`](specs/2026-05-24-retainiq-design.md) and [`docs/PLAYBOOK.md`](PLAYBOOK.md).
