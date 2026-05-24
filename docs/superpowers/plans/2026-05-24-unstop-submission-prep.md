# ChurnZero Unstop Submission Prep — executed 2026-05-24

Implementation record. Source plan in Cursor (not edited in repo).

## Shipped

| Task | Deliverable |
|------|-------------|
| Validate | [`scripts/validate_submission.py`](../../scripts/validate_submission.py) |
| Package | [`scripts/package_submission.py`](../../scripts/package_submission.py) |
| Docs | [`RUN_NOTES.md`](../../RUN_NOTES.md), [`RESULTS.md`](../../RESULTS.md), [`README.md`](../../README.md) |
| Deck | [`deck/retainiq_outline.md`](../../../deck/retainiq_outline.md) slides 7–13 |
| QA | [`reviews/qa_checklist.md`](../../../reviews/qa_checklist.md) |

## Canonical metrics (local retrain)

- Threshold **0.002** · OOF cost **₹65,000** vs **₹202,500** @ t=0.5 · savings **₹137,500 (~67.9%)**
- Test positive rate **16.63%**
- Uplift: **6** persuadable, **90** sleeping-dog
- Rank weights: LGB **0.4**, Cat **0.6**

## Commands

```powershell
python -m scripts.validate_submission
python -m scripts.package_submission
python -m pytest tests/test_integration.py -m integration -v
```

## Human remaining

1. swayangjeet: `deck/ChurnZero_RetainIQ_Presentation.pdf` from outline + `deck/charts/*.png`
2. Re-run `package_submission` after PDF exists
3. Upload `submission/ChurnZero_RetainIQ.zip` to Unstop
4. Rehearse [`JUDGE_QA.md`](../../JUDGE_QA.md)
