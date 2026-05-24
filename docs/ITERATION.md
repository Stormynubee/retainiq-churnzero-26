# Development iteration

How RetainIQ evolved during ChurnZero 26. Commit SHAs are on `main` as of submission freeze.

## Why commit count looks small

This repository history was rewritten with `git filter-repo` (metadata under `.git/filter-repo/`). The visible history is **~15 commits**, not a day-by-day log. The work landed in **phases** below; tests and [`docs/CAUSAL_FIXES.md`](CAUSAL_FIXES.md) preserve the iteration story.

## Phases

| Phase | Commits (approx) | What changed |
|-------|------------------|--------------|
| **Baseline** | `bc783b0` … `37ee63a` | Package layout, pytest + smoke CI, README, initial pipeline |
| **Causal + calibration** | `5bf5e84` | Drop post-treatment leakage; OOF meta calibration; Platt scaling; stacking alignment |
| **Finalist push** | `0e4d159` | Meta trained on OOF base preds; dual-track CSV; IPTW uplift trim |
| **Judge prep** | `dc0793a`, `603a998` | JUDGE_QA, validate/package scripts, metrics sync |
| **Deck + submission** | `c3dbc5f` … `383d722` | Slide copy, PPTX builder, chart PNGs, PDF export, strict ZIP |

## Evidence besides git

| Artifact | What it proves |
|----------|----------------|
| `tests/test_causal_leakage.py` | Post-treatment cols stay out of features |
| `tests/test_stacking_train.py` | Meta uses OOF predictions, not in-sample averages |
| `docs/CAUSAL_FIXES.md` | External audit findings and fixes |
| `data/processed/cost_optimal_threshold.json` | Threshold chosen on OOF calibrated probs |

## Tagged releases (optional on GitHub)

If tags are present: `v0.2-causal-fixes`, `v0.3-finalist`, `v1.0-submission` mark the phases above.
