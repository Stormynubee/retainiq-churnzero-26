# Contributing

## Setup

```powershell
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---|---|
| `python -m pytest -m "not slow and not integration"` | Fast unit tests (~5s) |
| `python -m pytest -m slow` | Smoke train on synthetic data (~10s) |
| `python -m pytest -m integration` | Needs `data/raw/` CSVs |
| `python -m scripts.train` | Full train (~4 min) |
| `python -m scripts.build_artifacts` | Deck charts (after train) |
| `python -m scripts.audit_features` | Optional ablation audit |
| `python -m scripts.predict` | Submission CSV |

## Test markers

- `slow` — trains models on `tests/fixtures/mini_train.csv`
- `integration` — requires competition CSVs locally

## Pull requests

1. Run fast tests before pushing.
2. Do **not** commit `data/raw/*.csv`, `submission/*.csv`, or `.cursor/`.
3. Update `docs/RUN_NOTES.md` if headline metrics change.

## Owners

- ML / code: **Hansraj Tiwari**
- Deck / QA / ZIP: **swayangjeet nayak**
