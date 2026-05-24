# Architecture

One-page map of RetainIQ. Full design rationale: [`docs/specs/2026-05-24-retainiq-design.md`](specs/2026-05-24-retainiq-design.md).

## Pipeline

```
raw CSVs
  → data.py          load, stratified folds
  → features.py      ratios, categoricals, missingness (train-only fit)
  → models.py        LightGBM + CatBoost OOF → logistic stack → isotonic
  → threshold.py     rupee cost sweep → optimal t
  → submit.py        competition CSV
  → importance.py    LGB gain table (during train)
  → uplift.py        T-learner CATE + segments (deck)
  → fairness.py      gender / region audit (deck)
  → build_artifacts  PNG charts for slides
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Paths, costs, column lists |
| `artifacts.py` | Output paths, `ensure_dirs()`, manifest |
| `pipeline.py` | `train()` / `predict()` orchestration |
| `scripts/` | CLI wrappers for judges |

## Artifact contract

After `train()`:

| File | Purpose |
|---|---|
| `stacked_bundle.joblib` | LGB + CatBoost + meta + calibrator |
| `fe_state.joblib` | Feature engineering state |
| `cost_optimal_threshold.json` | Optimal t + cost comparison |
| `training_metrics.json` | OOF metrics @ optimal vs 0.5 |
| `feature_importances.csv` | Deck chart 09 |
| `artifacts_manifest.json` | Seed, git commit, headline numbers |

## Tests

- **Unit** — threshold, submit, features, fairness rules (~30 tests, <5s)
- **Smoke** — synthetic 160-row train, 2-fold, ~10s (`@pytest.mark.slow`)
- **Integration** — real competition CSVs when present (`@pytest.mark.integration`)

## Team ownership

| Area | Owner |
|---|---|
| `src/`, `scripts/`, `tests/` | Hansraj Tiwari |
| `deck/`, `reviews/`, submission ZIP | swayangjeet nayak |
