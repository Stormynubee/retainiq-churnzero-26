# How this repo was built

Hansraj Tiwari & swayangjeet nayak · ChurnZero 26 · RetainIQ

## Team

| Person | Primary ownership | Also comfortable explaining |
|--------|-------------------|---------------------------|
| **Hansraj Tiwari** | `src/retainiq/`, `scripts/train`, stacking, calibration, threshold, tests | Slide 8 axes, uplift IPTW if asked |
| **swayangjeet nayak** | Deck, INR story, Unstop ZIP, rehearsal, QA checklist | Cost framing, segment playbook |

**Overlap:** Both can walk through [`docs/CAUSAL_FIXES.md`](CAUSAL_FIXES.md) flaws 1–4 (post-treatment leakage, OOF calibration, meta-on-OOF, Platt). Both reviewed submission CSV before upload.

## Tools

- **Coding assistants:** boilerplate tests, doc drafts, repetitive script wiring—the same scope described in [`docs/JUDGE_QA.md`](JUDGE_QA.md).
- **External ML review:** causal and calibration issues documented in [`docs/CAUSAL_FIXES.md`](CAUSAL_FIXES.md); fixes are in code and pytest, not generated narrative alone.
- **No** fairlearn / causalml in production deps—we hand-rolled uplift and fairness to keep Windows installs simple.

## What we can show live (without retraining)

```powershell
python -m pytest tests/test_stacking_train.py tests/test_causal_leakage.py -q
python -m scripts.validate_submission
```

Then open:

- `data/processed/cost_curve.csv` and `deck/charts/08_cost_curve.png` (slide 8)
- `data/processed/training_metrics.json` (OOF costs @ t=0.002 vs 0.5)

## Reviewers

- Development timeline: [`docs/ITERATION.md`](ITERATION.md)
- Release notes: [`CHANGELOG.md`](../CHANGELOG.md)
- Methodology fixes: [`docs/CAUSAL_FIXES.md`](CAUSAL_FIXES.md)
