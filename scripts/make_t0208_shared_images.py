#!/usr/bin/env python3
"""Draw the three T-0208 shared images (B2) into ``local_astra/pages/handoff/``.

Usage: ``uv run python scripts/make_t0208_shared_images.py``

Each image is 1280x800 RGB, saved as JPEG at quality 85, drawn with Pillow from shapes and plain text only (no
photo, no logo, no copied markup). The committed bytes are the source of truth (``handoff_tasks.image_sha256`` pins
them); this script records how they were made and writes nowhere but ``local_astra/pages/handoff/``.

- H1 ``h1_steam_wand.jpg``: a line-drawn espresso machine with a label plate "Barista Express BES870" and the steam
  wand drawn snapped, inside a red circle (the only red in the picture).
- H2 ``h2_stove_box.jpg``: a box front reading "Trailhead Duo Stove" and "2-person", with a stove outline.
- H3 ``h3_chat.jpg``: a chat window with one message from "Sam".
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "local_astra" / "pages" / "handoff"
W, H = 1280, 800
QUALITY = 85
INK = (40, 40, 48)
PAPER = (246, 244, 238)
RED = (220, 30, 30)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def h1() -> Image.Image:
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    # body
    d.rounded_rectangle((330, 140, 900, 700), radius=24, outline=INK, width=6)
    # bean hopper on top
    d.polygon([(420, 140), (480, 70), (620, 70), (680, 140)], outline=INK, width=5)
    # gauge
    d.ellipse((560, 190, 680, 310), outline=INK, width=5)
    d.line((620, 250, 655, 215), fill=INK, width=4)
    # label plate
    d.rectangle((400, 340, 840, 420), outline=INK, width=4)
    d.text((420, 355), "Barista Express BES870", fill=INK, font=font(38))
    # group head and portafilter
    d.rectangle((520, 470, 680, 510), outline=INK, width=5)
    d.rounded_rectangle((500, 510, 700, 560), radius=10, outline=INK, width=5)
    d.line((700, 535, 860, 535), fill=INK, width=12)
    # drip tray
    d.rectangle((380, 640, 850, 690), outline=INK, width=5)
    # steam wand: mount on the left side, then the snapped stub and the broken-off piece
    d.rectangle((300, 430, 330, 470), outline=INK, width=4)
    d.line((300, 450, 250, 500), fill=INK, width=10)  # stub still attached
    d.line((228, 540, 205, 620), fill=INK, width=10)  # broken-off piece below
    d.line((248, 502, 240, 512), fill=INK, width=3)  # jagged break
    d.line((232, 528, 240, 520), fill=INK, width=3)
    # the red circle around the snapped wand
    d.ellipse((150, 410, 340, 640), outline=RED, width=12)
    return im


def h2() -> Image.Image:
    im = Image.new("RGB", (W, H), (225, 230, 236))
    d = ImageDraw.Draw(im)
    # box front
    d.rectangle((240, 80, 1040, 720), fill=(250, 248, 240), outline=INK, width=6)
    d.text((300, 120), "Trailhead Duo Stove", fill=INK, font=font(72))
    d.text((300, 220), "2-person", fill=INK, font=font(48))
    d.text((300, 280), "two-burner camping stove", fill=INK, font=font(30))
    # stove outline
    d.rectangle((360, 420, 920, 620), outline=INK, width=6)
    d.ellipse((420, 380, 600, 460), outline=INK, width=5)
    d.ellipse((680, 380, 860, 460), outline=INK, width=5)
    d.ellipse((480, 400, 540, 440), outline=INK, width=4)
    d.ellipse((740, 400, 800, 440), outline=INK, width=4)
    d.ellipse((420, 560, 470, 610), outline=INK, width=4)
    d.ellipse((810, 560, 860, 610), outline=INK, width=4)
    d.line((380, 620, 360, 680), fill=INK, width=6)
    d.line((900, 620, 920, 680), fill=INK, width=6)
    return im


def h3() -> Image.Image:
    im = Image.new("RGB", (W, H), (210, 214, 220))
    d = ImageDraw.Draw(im)
    # window
    d.rectangle((120, 60, 1160, 740), fill=(255, 255, 255), outline=INK, width=4)
    d.rectangle((120, 60, 1160, 130), fill=(235, 237, 241), outline=INK, width=4)
    d.text((160, 78), "Chat with Sam", fill=INK, font=font(34))
    # avatar and message bubble
    d.ellipse((160, 200, 240, 280), outline=INK, width=4)
    d.text((184, 220), "S", fill=INK, font=font(36))
    d.text((270, 180), "Sam", fill=INK, font=font(30))
    d.rounded_rectangle(
        (260, 225, 1120, 370), radius=24, fill=(232, 240, 250), outline=INK, width=3
    )
    d.text((290, 250), "can you get me a large pepperoni from Domino's?", fill=INK, font=font(34))
    d.text((290, 305), "I'll pick it up at the W 44th St store", fill=INK, font=font(34))
    # input box
    d.rectangle((160, 640, 1120, 700), outline=INK, width=3)
    d.text((180, 655), "Type a message", fill=(150, 150, 160), font=font(28))
    return im


IMAGES = {"h1_steam_wand.jpg": h1, "h2_stove_box.jpg": h2, "h3_chat.jpg": h3}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, draw in IMAGES.items():
        draw().save(OUT_DIR / name, "JPEG", quality=QUALITY)
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
