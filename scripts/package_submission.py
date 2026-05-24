"""Build ChurnZero_<TeamName>.zip for Unstop (exactly three files at zip root)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config


def _expected_names() -> tuple[str, str, str]:
    base = f"ChurnZero_{config.TEAM_NAME}"
    return (
        f"{base}_Predictions.csv",
        f"{base}_Presentation.pptx",
        f"{base}_Code.py",
    )


def build_unstop_zip(
    *,
    csv: Path,
    deck: Path,
    code: Path,
    out: Path,
) -> Path:
    csv_name, deck_name, code_name = _expected_names()
    for path, label in ((csv, "predictions CSV"), (deck, "presentation"), (code, "code")):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    staging = ROOT / "submission" / "_zip_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(csv, staging / csv_name)
    shutil.copy2(deck, staging / deck_name)
    shutil.copy2(code, staging / code_name)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    shutil.make_archive(str(out.with_suffix("")), "zip", staging)
    shutil.rmtree(staging)
    return out


def main() -> None:
    csv_n, deck_n, code_n = _expected_names()
    parser = argparse.ArgumentParser(description=f"Build ChurnZero_{config.TEAM_NAME}.zip")
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.SUBMISSION_CSV,
        help=f"Predictions CSV (default: submission/{csv_n})",
    )
    parser.add_argument(
        "--deck",
        type=Path,
        default=config.DECK_PPTX,
        help=f"Presentation (default: deck/{deck_n})",
    )
    parser.add_argument(
        "--code",
        type=Path,
        default=config.CODE_SUBMISSION,
        help=f"Code file (default: {code_n})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.SUBMISSION_ZIP,
        help=f"Output zip path (default: submission/ChurnZero_{config.TEAM_NAME}.zip)",
    )
    args = parser.parse_args()

    out = build_unstop_zip(csv=args.csv, deck=args.deck, code=args.code, out=args.output)
    print(f"Wrote {out}")
    print("Zip contains exactly 3 files:")
    for name in _expected_names():
        print(f"  - {name}")


if __name__ == "__main__":
    main()
