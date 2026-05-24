"""CLI entry point: score the test set and write submission CSV.

Usage:
    python -m scripts.predict
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from retainiq import pipeline  # noqa: E402


def main() -> int:
    pipeline.predict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
