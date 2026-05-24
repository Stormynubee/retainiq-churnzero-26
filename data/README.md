# Data directory

Competition CSVs are **not** committed to this repository.

## Expected files

Place these in `data/raw/` (from the [ChurnZero 26 portal](https://unstop.com/competitions/churnzero-26-iit-kharagpur-1686181)):

| File | Rows | Notes |
|---|---:|---|
| `ChurnZero_dataset_v1.csv` | 8,101 | Includes `churn` target |
| `ChurnZero_test_v1.csv` | 2,026 | No `churn` column |

## Generated outputs (`data/processed/`)

Created by `python -m scripts.train` — gitignored, reproducible locally.

## CI / tests without competition data

Synthetic fixtures live in `tests/fixtures/`:

- `mini_train.csv` — 160 rows for smoke tests
- `mini_test.csv` — 40 rows

Regenerate: `python tests/fixtures/build_fixtures.py`

**Do not redistribute** official competition datasets in forks or PRs.
