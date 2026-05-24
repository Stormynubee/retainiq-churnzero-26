"""CLI entry point: train the RetainIQ stacked ensemble.

Usage:
    python -m scripts.train
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/retainiq` importable as `retainiq` when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from retainiq import pipeline  # noqa: E402  (path injection above)


def main() -> int:
    pipeline.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
