"""Public docs must not use template hedge phrases (see authenticity spec)."""

from __future__ import annotations

from pathlib import Path

PUBLIC_PATHS = [
    "README.md",
    "deck/retainiq_slides_content.md",
    "docs/specs/2026-05-24-retainiq-design.md",
]

BANNED = [
    "the model is saturated",
    "PR-AUC saturated",
    "hero moment",
    "You can't win Round 2 on AUC alone",
    "game-changer",
    "leverage",
    "robust",
    "comprehensive solution",
]


def test_public_docs_avoid_hedge_phrases() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in PUBLIC_PATHS:
        text = (root / rel).read_text(encoding="utf-8")
        lower = text.lower()
        for phrase in BANNED:
            assert phrase.lower() not in lower, f"{rel}: banned phrase {phrase!r}"
