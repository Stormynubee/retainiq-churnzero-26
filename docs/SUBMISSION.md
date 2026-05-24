# Unstop upload checklist

Hansraj Tiwari & swayangjeet nayak · ChurnZero 26 · **Team Vortex** · RetainIQ

## Before you zip

```powershell
cd C:\Users\storm\Projects\churnzero-26
pip install -e ".[dev]"
python -m scripts.pre_upload_check
```

Add `--require-pdf` once the deck PDF exists:

```powershell
python -m scripts.pre_upload_check --require-pdf
```

## Deck (swayangjeet)

1. Polish [`deck/ChurnZero_TeamVortex_Presentation.pptx`](../deck/ChurnZero_TeamVortex_Presentation.pptx) per [`deck/PPT_BUILD.md`](../deck/PPT_BUILD.md).
2. Export **`deck/ChurnZero_TeamVortex_Presentation.pdf`** (File → Save as PDF), or `python -m scripts.export_deck_pdf` on Windows.
3. Rehearse slides **8**, **10**, **11** using [`docs/JUDGE_QA.md`](JUDGE_QA.md).

Regenerate starter deck after any retrain:

```powershell
python -m scripts.pick_persuadable_story
python -m scripts.build_deck_pptx
```

## Package and verify

```powershell
python -m scripts.package_submission --strict
```

Unzip `ChurnZero_TeamVortex.zip` once and confirm **exactly three files** at the zip root:

1. `ChurnZero_TeamVortex_Predictions.csv` (2,026 rows)
2. `ChurnZero_TeamVortex_Presentation.pptx` (or `.pdf` if you exported PDF)
3. `ChurnZero_TeamVortex_Code.py`

Build the zip:

```powershell
python -m scripts.package_submission
```

Or from your Downloads assets:

```powershell
python -m scripts.package_submission `
  --csv "$env:USERPROFILE\Downloads\ChurnZero_TeamVortex_Predictions.csv" `
  --deck "$env:USERPROFILE\Downloads\ChurnZero_TeamVortex_Presentation.pptx" `
  --output "$env:USERPROFILE\Downloads\ChurnZero_TeamVortex.zip"
```

## Upload

Upload the ZIP to the ChurnZero 26 portal on Unstop.

Sign [`reviews/qa_checklist.md`](../reviews/qa_checklist.md) when done.

## After results

Record shortlist / round outcome in [`docs/RUN_NOTES.md`](RUN_NOTES.md) under `## Competition outcome` (do not guess on slides).
