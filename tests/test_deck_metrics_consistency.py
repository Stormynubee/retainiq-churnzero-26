"""Guard judge-facing docs against stale threshold / uplift numbers."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STALE_SNIPPETS = (
    "t ≈ 0.001",
    "threshold 0.001",
    "~0.001),",
    "At t ≈ 0.001",
    "Only **4** persuadables",
    "~4 in training",
    "141 sleeping-dogs",
    "141 in training",
    "Gender DP gap ~2.7%",
    "gender ~2.7%",
    "₹62,500",
)


@pytest.mark.parametrize(
    "rel_path,needle",
    [
        ("docs/JUDGE_QA.md", "0.002"),
        ("docs/PLAYBOOK.md", "0.002"),
        ("docs/PLAYBOOK.md", "6 in training"),
        ("deck/retainiq_slides_content.md", "137,500"),
        ("notebooks/retainiq_story.py", "sleeping-dog"),
    ],
)
def test_docs_reference_current_threshold_and_savings(rel_path: str, needle: str) -> None:
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    assert needle in text, f"{rel_path} missing {needle!r}"
    for snippet in STALE_SNIPPETS:
        assert snippet not in text, f"{rel_path} still contains stale {snippet!r}"
