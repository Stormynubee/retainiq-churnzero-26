"""Export deck PPTX to PDF via PowerPoint on Windows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "deck" / "ChurnZero_RetainIQ_Presentation.pptx"
PDF = ROOT / "deck" / "ChurnZero_RetainIQ_Presentation.pdf"


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("PDF export requires Windows with Microsoft PowerPoint installed")
    if not PPTX.is_file():
        raise SystemExit(f"Missing {PPTX} — run: python -m scripts.build_deck_pptx")

    ps = f"""
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open('{PPTX.resolve()}')
$pres.SaveAs('{PDF.resolve()}', 32)
$pres.Close()
$ppt.Quit()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=True,
    )
    if not PDF.is_file():
        raise SystemExit("PDF export failed")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
