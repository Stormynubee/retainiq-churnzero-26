"""Run before Unstop upload. Exit 1 on blockers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-upload QA gate for ChurnZero 26")
    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Fail if deck/ChurnZero_RetainIQ_Presentation.pdf is missing",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not slow and not integration", "-q"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        errors.append("unit tests failed")

    r = subprocess.run(
        [sys.executable, "-m", "scripts.validate_submission"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        errors.append("validate_submission failed")

    if not (ROOT / "data/processed/training_metrics.json").is_file():
        errors.append("missing data/processed/training_metrics.json — run train")

    pdf = ROOT / "deck/ChurnZero_RetainIQ_Presentation.pdf"
    if not pdf.is_file():
        msg = "missing deck/ChurnZero_RetainIQ_Presentation.pdf — export before final ZIP"
        if args.require_pdf:
            errors.append(msg)
        else:
            warnings.append(msg)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_deck_metrics_consistency.py", "-q"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        errors.append("doc metrics consistency tests failed")

    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)
    print("pre_upload_check: OK")


if __name__ == "__main__":
    main()
