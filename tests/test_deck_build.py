"""Deck PPTX generation."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "deck" / "charts"


@pytest.mark.slow
def test_build_deck_creates_pptx_with_15_slides(tmp_path, monkeypatch):
    pytest.importorskip("pptx")
    if not (CHARTS / "08_cost_curve.png").is_file():
        pytest.skip("charts missing — run: python -m scripts.build_artifacts")

    out = tmp_path / "deck.pptx"
    monkeypatch.setenv("RETAINIQ_DECK_OUT", str(out))

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_deck_pptx", ROOT / "scripts" / "build_deck_pptx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    mod.build_deck(out)
    assert out.is_file()

    from pptx import Presentation

    prs = Presentation(str(out))
    assert len(prs.slides) == 15
