# Contributing

## Setup

```powershell
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---|---|
| `python -m pytest -m "not slow and not integration"` | Fast unit tests (~5s, ~50 tests) |
| `python -m pytest` | Full suite (~58 tests) |
| `python -m pytest -m slow` | Smoke train on synthetic data (~10s) |
| `python -m pytest -m integration` | Needs `data/raw/` CSVs |
| `python -m scripts.train` | Full train (~4 min) |
| `python -m scripts.build_artifacts` | Deck charts (after train) |
| `python -m scripts.audit_features` | Optional ablation audit |
| `python -m scripts.predict` | Submission CSV |
| `python -m scripts.validate_submission` | Pre-upload CSV checks |
| `python -m scripts.pre_upload_check` | Full gate before Unstop (add `--require-pdf` when ready) |
| `python -m scripts.package_submission` | Build `submission/ChurnZero_RetainIQ.zip` |
| `python -m scripts.package_submission --strict` | ZIP requires deck PDF |
| `python -m scripts.pick_persuadable_story` | Slide 11 customer story JSON |
| `python -m scripts.build_deck_pptx` | Starter `deck/ChurnZero_RetainIQ_Presentation.pptx` (needs `.[deck]`) |
| `python -m scripts.export_deck_pdf` | PPTX → PDF (Windows + PowerPoint) |

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
