"""python -m scripts.train"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retainiq import pipeline


if __name__ == "__main__":
    pipeline.train()
