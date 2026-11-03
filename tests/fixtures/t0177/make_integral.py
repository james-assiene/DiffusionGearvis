"""Render T-0177's integral fixture: a 1568x590 JPEG (quality 85) of a definite integral, as a shared screen shows it.

The picture is the shape of James's t14 case (logs/sessions/20261027-230926-966483.jsonl: a 1568x590 shared screen
holding an integral the model never saw): a white page, a small heading, and one large definite integral

    the integral from 0 to pi of x sin(x) dx

drawn with DejaVuSans (the integral sign, the limits, the integrand ``x sin(x)`` and ``dx``). Its value is pi.

Usage (Pillow and DejaVuSans required; nothing else is read)::

    uv run python tests/fixtures/t0177/make_integral.py tests/fixtures/t0177/integral_1568x590.jpg

The output is deterministic for one Pillow / libjpeg build; MANIFEST.md records the sha256 of the committed file and
the versions it was made with. Tests pin the committed bytes (sha256), never a regeneration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1568, 590
QUALITY = 85
FONT_DIRS = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/TTF"),
    Path("/usr/share/fonts/dejavu"),
)
HEADING = "Calculus II - worksheet 3, problem 2"
INTEGRAND = "x sin(x) dx"


def font(size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        path = d / "DejaVuSans.ttf"
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise SystemExit(f"DejaVuSans.ttf not found under {[str(d) for d in FONT_DIRS]}")


def render() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # a thin browser-ish bar, so the picture reads as a shared screen
    draw.rectangle((0, 0, WIDTH, 44), fill=(232, 234, 237))
    draw.text((24, 10), "worksheet-3.pdf", font=font(20), fill=(60, 64, 67))
    draw.text((80, 76), HEADING, font=font(30), fill=(32, 33, 36))
    draw.text((80, 126), "Evaluate:", font=font(30), fill=(32, 33, 36))
    # the integral: sign, limits, integrand
    sign_font = font(250)
    draw.text((420, 150), "∫", font=sign_font, fill=(0, 0, 0))
    draw.text((560, 160), "π", font=font(64), fill=(0, 0, 0))
    draw.text((470, 470), "0", font=font(64), fill=(0, 0, 0))
    draw.text((620, 290), INTEGRAND, font=font(110), fill=(0, 0, 0))
    return img


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1])
    render().save(out, format="JPEG", quality=QUALITY, optimize=False, progressive=False)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
