# ChurnZero deck content — executed 2026-05-24

## Shipped

| Deliverable | Path |
|-------------|------|
| Slide copy + speaker notes | `deck/retainiq_slides_content.md` |
| PPTX generator | `scripts/build_deck_pptx.py` |
| Slide 11 story | `scripts/pick_persuadable_story.py` |
| Build guide | `deck/PPT_BUILD.md` |
| Tests | `tests/test_deck_build.py` |

## Commands

```powershell
pip install -e ".[deck]"
python -m scripts.pick_persuadable_story
python -m scripts.build_deck_pptx
```

## Human remaining

Export PDF from polished PPTX → `deck/ChurnZero_RetainIQ_Presentation.pdf` → `package_submission` → Unstop upload.
