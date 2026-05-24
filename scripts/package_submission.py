"""Build ChurnZero_<TeamName>.zip for Unstop upload."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config

OUT = config.SUBMISSION_ZIP
REQUIRED = [config.SUBMISSION_CSV]
OPTIONAL_DECK = [config.DECK_PDF, config.DECK_PPTX]


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build {OUT.name}")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=f"Require {config.DECK_PDF.name}",
    )
    args = parser.parse_args()

    missing = [p for p in REQUIRED if not p.is_file()]
    if missing:
        raise SystemExit("Missing: " + ", ".join(p.name for p in missing))

    pdf = config.DECK_PDF
    if args.strict and not pdf.is_file():
        raise SystemExit(f"Missing {pdf} (export PDF first)")

    staging = ROOT / "submission" / "_zip_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(REQUIRED[0], staging / REQUIRED[0].name)
    if args.strict and pdf.is_file():
        shutil.copy2(pdf, staging / pdf.name)
    else:
        deck = next((p for p in OPTIONAL_DECK if p.is_file()), None)
        if deck:
            shutil.copy2(deck, staging / deck.name)
        else:
            print("WARN: no deck PDF/pptx in deck/ — add before final upload")

    code_dir = staging / "retainiq-code"
    shutil.copytree(
        ROOT / "src",
        code_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for rel in ("scripts", "requirements.txt", "pyproject.toml", "README.md"):
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, code_dir / rel)
        elif src.is_dir():
            shutil.copytree(src, code_dir / rel, ignore=shutil.ignore_patterns("__pycache__"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file():
        OUT.unlink()
    shutil.make_archive(str(OUT.with_suffix("")), "zip", staging)
    shutil.rmtree(staging)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
