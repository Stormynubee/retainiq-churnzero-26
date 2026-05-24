#!/usr/bin/env python3
"""
ChurnZero 26 — Team Vortex — reproducible code submission

RetainIQ churn pipeline: train → predict → validate submission CSV.

Setup
-----
1. Python 3.11+
2. Competition CSVs in ``data/raw/``:
   - ChurnZero_dataset_v1.csv (train, 8,101 rows)
   - ChurnZero_test_v1.csv (test, 2,026 rows)
3. From repo root::

       pip install -e .

Run end-to-end
--------------
::

    python ChurnZero_TeamVortex_Code.py

Writes ``submission/ChurnZero_TeamVortex_Predictions.csv`` with columns
``customer_id``, ``churn_prediction`` (0/1), ``churn_probability`` ([0, 1]).

Optional (deck charts / uplift, not required for CSV)::

    python -m scripts.build_artifacts

Narrative walkthrough (after train): ``notebooks/retainiq_story.py`` in Jupyter or VS Code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(argv: list[str]) -> None:
    print(f"\n>>> {' '.join(argv)}\n", flush=True)
    subprocess.check_call(argv, cwd=ROOT)


def _ensure_package() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import retainiq  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-e", "."])


def _check_data() -> None:
    raw = ROOT / "data" / "raw"
    needed = ("ChurnZero_dataset_v1.csv", "ChurnZero_test_v1.csv")
    missing = [name for name in needed if not (raw / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing in data/raw/: "
            + ", ".join(missing)
            + "\nObtain files from the ChurnZero 26 portal (not redistributed in this repo)."
        )


def main() -> None:
    _check_data()
    _ensure_package()
    _run([sys.executable, "-m", "scripts.train"])
    _run([sys.executable, "-m", "scripts.predict"])
    _run([sys.executable, "-m", "scripts.validate_submission"])
    from retainiq import config

    print(f"\nOK — wrote {config.SUBMISSION_CSV}")
    print("Upload ChurnZero_TeamVortex.zip to Unstop with your GitHub repo link.")


if __name__ == "__main__":
    main()
