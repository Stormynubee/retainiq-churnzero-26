"""Export top persuadable customer for deck slide 11 (run after build_artifacts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config


def main() -> None:
    path = config.DATA_PROCESSED / "uplift_per_customer.csv"
    if not path.is_file():
        raise SystemExit("Run: python -m scripts.build_artifacts")

    df = pd.read_csv(path)
    persuadable = df[df["segment"] == "persuadable"]
    if persuadable.empty:
        raise SystemExit("No persuadable segment rows")

    median_p = persuadable["churn_proba"].median()
    candidates = persuadable[persuadable["churn_proba"] >= median_p]
    pool = candidates if not candidates.empty else persuadable
    row = pool.sort_values("cate", ascending=False).iloc[0]

    p_churn = round(float(row["churn_proba"]), 3)
    cate = round(float(row["cate"]), 3)
    story = {
        "customer_id": str(row["customer_id"]),
        "churn_proba": p_churn,
        "cate": cate,
        "segment": str(row["segment"]),
        "talking_points": [
            f"Uplift CATE = +{cate}: offer strongly associated with staying for this profile.",
            f"Churn score = {p_churn} (persuadable band: high CATE + above-median risk in segment).",
            "Actions: targeted RM call, resolve complaints, timed waiver—not mass campaigns.",
        ],
    }
    out = config.DATA_PROCESSED / "slide11_customer_story.json"
    out.write_text(json.dumps(story, indent=2))
    print(json.dumps(story, indent=2))


if __name__ == "__main__":
    main()
