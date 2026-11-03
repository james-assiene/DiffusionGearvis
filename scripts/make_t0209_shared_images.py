#!/usr/bin/env python3
"""Draw the T-0209 shared images (A1.8) into ``local_astra/pages/rtx_shared/``.

Usage: ``uv run python scripts/make_t0209_shared_images.py``

Each image is 1280x800 RGB, saved as JPEG at quality 85, drawn with Pillow from shapes and plain text only (no
screenshot, no logo, no copied markup). The committed bytes are the source of truth (``rtx_tasks.RTX`` pins their
sha256); this script records how they were made and writes nowhere but ``local_astra/pages/rtx_shared/``.

- R1 ``r1_blender_error.jpg``: a Blender-like main window titled "Blender 4.2.3 LTS" behind a "Blender Preferences"
  window with "System" highlighted in its left list, a "Cycles Render Devices" heading with the device tabs (CUDA
  selected), the body "No compatible GPUs found for Cycles", and a red status bar with the CUDA kernel error. No GPU
  name and no logo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "local_astra" / "pages" / "rtx_shared"
W, H = 1280, 800
QUALITY = 85
BG = (48, 48, 48)
PANEL = (61, 61, 61)
HEADER = (35, 35, 35)
TEXT = (220, 220, 220)
MUTED = (150, 150, 150)
SELECT = (71, 114, 179)
RED = (176, 40, 40)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def r1() -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # the main window behind: title bar, menu bar, a 3D viewport and side panels
    d.rectangle((0, 0, W, 34), fill=HEADER)
    d.text((16, 7), "Blender 4.2.3 LTS", fill=TEXT, font=font(20))
    d.rectangle((0, 34, W, 64), fill=(42, 42, 42))
    for i, word in enumerate(("File", "Edit", "Render", "Window", "Help")):
        d.text((16 + i * 86, 40), word, fill=TEXT, font=font(18))
    d.rectangle((0, 64, 960, 740), fill=(57, 57, 57))
    for x in range(40, 960, 80):
        d.line((x, 64, x, 740), fill=(66, 66, 66), width=1)
    for y in range(104, 740, 80):
        d.line((0, y, 960, y), fill=(66, 66, 66), width=1)
    d.rectangle((960, 64, W, 740), fill=PANEL)
    d.polygon([(420, 420), (520, 360), (620, 420), (520, 480)], outline=(120, 120, 120), width=3)
    d.line((420, 420, 420, 540), fill=(120, 120, 120), width=3)
    d.line((620, 420, 620, 540), fill=(120, 120, 120), width=3)
    d.line((520, 480, 520, 600), fill=(120, 120, 120), width=3)
    d.line((420, 540, 520, 600), fill=(120, 120, 120), width=3)
    d.line((620, 540, 520, 600), fill=(120, 120, 120), width=3)

    # the preferences window in front
    x0, y0, x1, y1 = 180, 110, 1100, 640
    d.rectangle((x0 + 8, y0 + 8, x1 + 8, y1 + 8), fill=(20, 20, 20))
    d.rectangle((x0, y0, x1, y1), fill=PANEL, outline=(24, 24, 24), width=2)
    d.rectangle((x0, y0, x1, y0 + 36), fill=HEADER)
    d.text((x0 + 16, y0 + 8), "Blender Preferences", fill=TEXT, font=font(20))
    # left list, "System" highlighted
    items = ("Interface", "Themes", "Viewport", "Lights", "Editing", "Animation", "Add-ons", "Input",
             "Navigation", "Keymap", "System", "Save & Load", "File Paths")  # fmt: skip
    ly = y0 + 52
    for item in items:
        if item == "System":
            d.rectangle((x0 + 10, ly - 4, x0 + 200, ly + 26), fill=SELECT)
        d.text((x0 + 22, ly), item, fill=TEXT, font=font(18))
        ly += 34
    d.line((x0 + 212, y0 + 36, x0 + 212, y1), fill=(40, 40, 40), width=2)
    # right side: Cycles Render Devices
    rx = x0 + 240
    d.text((rx, y0 + 56), "Cycles Render Devices", fill=TEXT, font=font(24))
    tabs = ("None", "CUDA", "OptiX", "HIP", "oneAPI")
    tx = rx
    for tab in tabs:
        w = 110
        fill = SELECT if tab == "CUDA" else (84, 84, 84)
        d.rectangle((tx, y0 + 100, tx + w, y0 + 136), fill=fill, outline=(40, 40, 40))
        d.text((tx + 16, y0 + 107), tab, fill=TEXT, font=font(18))
        tx += w + 4
    d.rectangle((rx, y0 + 152, x1 - 30, y0 + 230), fill=(52, 52, 52), outline=(40, 40, 40))
    d.text((rx + 18, y0 + 178), "No compatible GPUs found for Cycles", fill=TEXT, font=font(20))
    d.text((rx, y0 + 270), "Memory & Limits", fill=MUTED, font=font(20))
    d.text((rx, y0 + 310), "Undo Steps            32", fill=MUTED, font=font(18))
    d.text((rx, y0 + 344), "Undo Memory Limit     0", fill=MUTED, font=font(18))

    # the red status bar at the bottom of the main window
    d.rectangle((0, 748, W, H), fill=RED)
    d.text(
        (16, 762),
        "CUDA binary kernel for this graphics card compute capability (12.0) not found.",
        fill=(255, 255, 255),
        font=font(22),
    )
    return im


IMAGES = {"r1_blender_error.jpg": r1}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, draw in IMAGES.items():
        draw().save(OUT_DIR / name, "JPEG", quality=QUALITY)
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
