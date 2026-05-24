# PowerPoint build checklist (swayangjeet)

## Generate starter deck

```powershell
cd C:\Users\storm\Projects\churnzero-26
pip install -e ".[deck]"
python -m scripts.pick_persuadable_story
python -m scripts.build_deck_pptx
```

Output: `deck/ChurnZero_RetainIQ_Presentation.pptx` (15 slides).

Copy reference: [`retainiq_slides_content.md`](retainiq_slides_content.md) · speaker notes · [`docs/JUDGE_QA.md`](../docs/JUDGE_QA.md)

## Manual polish (30–60 min)

1. Open PPTX — fix alignment, team branding (IIT logo optional).
2. **Slide 2:** add simple 80:1 bar visual if not already clear.
3. **Slide 4:** optional tenure × churn bar chart from EDA.
4. **Slide 8:** ensure cost curve is large; verify caption **₹137.5k saved**, threshold **0.002**.
5. Spell-check all slides.
6. Export PDF: `deck/ChurnZero_RetainIQ_Presentation.pdf` (File → Save as PDF).

## Package for Unstop

```powershell
python -m scripts.validate_submission
python -m scripts.package_submission
```

Unzip `submission/ChurnZero_RetainIQ.zip` once — confirm CSV, PDF, and `retainiq-code/` folder.

## Rehearsal (15 min with Hansraj)

- Slide **8** — hero cost curve (pause 20 seconds)
- Slide **10** — uplift + “offers not randomized”
- Slide **11** — one persuadable story

Sign [`reviews/qa_checklist.md`](../reviews/qa_checklist.md) when done.
