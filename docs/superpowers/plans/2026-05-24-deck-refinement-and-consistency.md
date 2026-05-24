# Deck refinement and consistency — executed 2026-05-24

## Done in repo

| Item | Path |
|------|------|
| Metrics loader | `src/retainiq/deck_metrics.py` |
| Doc drift tests | `tests/test_deck_metrics_consistency.py` |
| Synced judge docs | `JUDGE_QA.md`, `PLAYBOOK.md`, `CAUSAL_FIXES.md`, spec, `RUN_NOTES.md` (Historical section) |
| Humanized slide copy | `deck/retainiq_slides_content.md` |
| Metrics-driven PPTX | `scripts/build_deck_pptx.py` |
| Stronger slide 11 | `scripts/pick_persuadable_story.py` → customer **140348** |
| Chart PNGs committed | `deck/charts/*.png` |
| Checklist updates | `deck/PPT_BUILD.md` |

## Human remaining (swayangjeet)

1. Polish `deck/ChurnZero_RetainIQ_Presentation.pptx` (slides 2, 4, 8 visuals).
2. Export `deck/ChurnZero_RetainIQ_Presentation.pdf`.
3. `python -m scripts.package_submission` and upload ZIP.
4. Sign `reviews/qa_checklist.md` deck section.

## Anti-drift command

```powershell
python -m scripts.pick_persuadable_story
python -m scripts.build_deck_pptx
python -m pytest tests/test_deck_metrics_consistency.py -q
```
