"""Render docs/assets/banner.png from the SVG design (Pillow; Windows-safe)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/banner.png"
W, H = 1600, 400  # 2x for GitHub retina; displayed at 800px wide


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), "#111827")
    draw = ImageDraw.Draw(img)

    # Border
    draw.rounded_rectangle((2, 2, W - 3, H - 3), radius=30, outline="#1E293B", width=4)

    # Soft accent blobs
    draw.ellipse((W - 380, -180, W + 60, 260), fill="#6366F1")
    draw.ellipse((-60, H - 220, 260, H + 60), fill="#22C55E")
    overlay = Image.new("RGBA", (W, H), (17, 24, 39, 200))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)

    title_font = _font(84, bold=True)
    accent_font = _font(36, bold=True)
    tag_font = _font(32)
    pill_font = _font(24, bold=True)

    draw.text((96, 72), "RetainIQ", fill="#F8FAFC", font=title_font)
    draw.text((96, 168), "CHURNZERO 26 \u00b7 IIT KHARAGPUR", fill="#818CF8", font=accent_font)

    # Tagline backing pill (high contrast)
    draw.rounded_rectangle((80, 228, 1120, 308), radius=16, fill="#0F172A")
    draw.text(
        (96, 248),
        "Cost-aware churn prediction \u00b7 Uplift \u00b7 Fairness audit",
        fill="#F8FAFC",
        font=tag_font,
    )

    pills = [
        (96, 328, 316, 388, "#22C55E", "PR-AUC 0.9999"),
        (336, 328, 596, 388, "#818CF8", "~68% cost savings"),
        (616, 328, 896, 388, "#F59E0B", "6 persuadables"),
    ]
    for x0, y0, x1, y1, color, label in pills:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill="#1E293B")
        draw.text((x0 + 24, y0 + 14), label, fill=color, font=pill_font)

    img.convert("RGB").resize((800, 200), Image.Resampling.LANCZOS).save(OUT, optimize=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
