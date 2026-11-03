"""The tool-call stress set (T-0099 criterion 1): 30 honest requests whose natural answer is out of schema.

    uv run python -m tests.eval.toolcall_stress.cases_src            # write cases.jsonl + MANIFEST.json
    uv run python -m tests.eval.toolcall_stress.cases_src --check    # exit 1 if either file is stale

Six categories (criterion 1), authored by the test-author, pending the human's approval (H-T99a):

- ``K-in`` (6): an in-enum key asked in another spelling; ``gt = {"name": "press_key", "combo": <enum combo>}``.
- ``K-out`` (6): a key that is **not** in the ``press_key`` enum of ``local_astra/toolcall/tools_v1.json``;
  ``gt = None`` (expected outcome: no executed action); the asked combo is recorded as ``out_combo``.
- ``S-extra`` (4): a scroll with an amount or a target the schema cannot carry; ``gt = {"name": "scroll",
  "direction": ...}``.
- ``T-hard`` (6): ``type_at`` text with double quotes, a backslash, braces, a newline, accents and emoji, and one
  text of at least 400 characters; ``gt`` = the dev target's bbox, the text and ``press_enter``, scored by
  ``local_astra.toolcall.score.exact``. Every instruction says explicitly whether to press Enter.
- ``C-pixel`` (4): "click at pixel (x, y)" on the 1280 x 800 frame with ``x > 999`` pixels; ``gt`` = the bbox of
  the element at that pixel (``click_at``).
- ``U-under`` (4): "type X into the field" with **no** Enter instruction; ``gt`` carries the bbox and the text and
  ``press_enter: None``. *Rescued* = schema-valid, inside the bbox, text equal (after strip); ``press_enter`` is
  reported, not scored (pre-registered in criterion 1).

Screens and targets come only from the **dev** split of ``tests/eval/toolcall`` (screens ``d01``-``d08``); a case
that reuses a dev target names it in ``source_case`` and its bbox is copied from the pinned
``tests/eval/toolcall/cases.jsonl`` at generation time (never re-rendered). No user text repeats a text of
``tests/eval/toolcall/cases.jsonl``.

**C-pixel normalisation, by hand.** The frame grid of the v1 set (``tests/eval/toolcall/build.py``): a normalised
``n`` in 0..999 stands for the pixel ``n * W / 1000``; the pixel ``p`` therefore lies in cell
``n = floor(p * 1000 / W)``. On the 1280 x 800 frame ``x_n = floor(x_px * 1000 / 1280)`` and
``y_n = floor(y_px * 1000 / 800)``. Every pixel below is chosen so the division is exact (no rounding question):

- ``cpix-01`` (d02_store, ``add_quiet`` of ``click-03``, bbox ``[740, 358, 839, 402]``): pixel ``(1024, 304)``:
  ``1024 * 1000 / 1280 = 800``, ``304 * 1000 / 800 = 380``; ``740 <= 800 <= 839`` and ``358 <= 380 <= 402``.
- ``cpix-02`` (d03_mail, ``msg_it`` of ``click-06``, bbox ``[189, 204, 983, 257]``): pixel ``(1152, 184)``:
  ``1152 * 1000 / 1280 = 900``, ``184 * 1000 / 800 = 230``; ``189 <= 900 <= 983`` and ``204 <= 230 <= 257``.
- ``cpix-03`` (d06_chat, ``message`` of ``type-05``, bbox ``[267, 913, 915, 959]``): pixel ``(1088, 748)``:
  ``1088 * 1000 / 1280 = 850``, ``748 * 1000 / 800 = 935``; ``267 <= 850 <= 915`` and ``913 <= 935 <= 959``.
- ``cpix-04`` (d08_editor, ``title`` of ``type-07``, bbox ``[94, 180, 906, 237]``): pixel ``(1120, 168)``:
  ``1120 * 1000 / 1280 = 875``, ``168 * 1000 / 800 = 210``; ``94 <= 875 <= 906`` and ``180 <= 210 <= 237``.

A naive model that copies the pixel into ``click_at`` sends ``x > 999``: schema-invalid (``maximum: 999``) and
repairable, which is the point of the category.

**K-out combos and the enum.** The ``press_key`` enum is ``Enter Escape Tab Backspace Delete Space ArrowUp
ArrowDown ArrowLeft ArrowRight Home End PageUp PageDown Ctrl+A Ctrl+C Ctrl+V Ctrl+X Ctrl+Z Ctrl+L Ctrl+T Ctrl+W
Alt+Tab Ctrl+Enter``. The K-out combos ``F5 Ctrl+S Ctrl+Shift+T Alt+F4 Ctrl+F Ctrl+Plus`` are absent from it, also
case-folded and with spaces removed; ``tests/unit/test_toolcall_stress_set_integrity.py`` checks it against the
file, not this list.

**T-hard text lengths, by hand.** ``hard-06``'s text is the ``LONG_TEXT`` constant below; its length is checked
(>= 400) by the integrity row, not asserted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.jsonl"
MANIFEST = HERE / "MANIFEST.json"
SOURCE_SET = HERE.parent / "toolcall" / "cases.jsonl"
SCREEN_SIZE = (1280, 800)

CATEGORIES: tuple[tuple[str, int], ...] = (
    ("K-in", 6),
    ("K-out", 6),
    ("S-extra", 4),
    ("T-hard", 6),
    ("C-pixel", 4),
    ("U-under", 4),
)

LONG_TEXT = (
    "Hi Omar, a quick update before Thursday: the meeting moved from the third floor to the ground-floor "
    "room because the lift is being serviced, lunch is now at 12:45 rather than 12:30, Priya will bring the "
    "printed slides, Lena is dialling in from Berlin and needs the call link by Wednesday evening, and I "
    "still need everyone's dietary requirements by noon tomorrow so that I can confirm the catering order. "
    "Thanks for sorting out the projector!"
)

# --------------------------------------------------------------------------------------------- K-in
# (id, screen, user text, enum combo)
K_IN: tuple[tuple[str, str, str, str], ...] = (
    ("kin-01", "d01_login", "Hit Return to submit the login form.", "Enter"),
    ("kin-02", "d04_settings", "Press Esc to back out of this page.", "Escape"),
    ("kin-03", "d08_editor", "Press control-c to copy what's selected in the note.", "Ctrl+C"),
    ("kin-04", "d05_article", "Press the spacebar to move further down the article.", "Space"),
    ("kin-05", "d03_mail", "Press page down to see more of my inbox.", "PageDown"),
    ("kin-06", "d03_mail", "Press the down arrow to move to the next email.", "ArrowDown"),
)

# -------------------------------------------------------------------------------------------- K-out
# (id, screen, user text, the asked combo -- not in the enum)
K_OUT: tuple[tuple[str, str, str, str], ...] = (
    ("kout-01", "d02_store", "Press F5 to reload the shop page.", "F5"),
    ("kout-02", "d08_editor", "Save this note with Ctrl+S.", "Ctrl+S"),
    ("kout-03", "d04_settings", "Reopen the tab I just closed with Ctrl+Shift+T.", "Ctrl+Shift+T"),
    ("kout-04", "d06_chat", "Press Alt+F4 to close this window.", "Alt+F4"),
    ("kout-05", "d05_article", "Press Ctrl+F so I can find a word in this article.", "Ctrl+F"),
    ("kout-06", "d07_photos", "Zoom in on the photos with Ctrl+Plus.", "Ctrl+Plus"),
)

# ------------------------------------------------------------------------------------------ S-extra
# (id, screen, user text, direction)
S_EXTRA: tuple[tuple[str, str, str, str], ...] = (
    ("sx-01", "d05_article", "Scroll down three pages in this article.", "down"),
    ("sx-02", "d03_mail", "Scroll to the very bottom of the message list.", "down"),
    ("sx-03", "d05_article", "Scroll up 500 pixels.", "up"),
    ("sx-04", "d05_article", "Scroll back to the top of the article.", "up"),
)

# ------------------------------------------------------------------------------------------- T-hard
# (id, source dev case whose target is reused, user text, text, press_enter, the hard feature)
T_HARD: tuple[tuple[str, str, str, str, bool, str], ...] = (
    (
        "hard-01",
        "type-03",
        'In the mail search box, type exactly this and press Enter: "quarterly budget" from:lena',
        '"quarterly budget" from:lena',
        True,
        "double quotes",
    ),
    (
        "hard-02",
        "type-07",
        "In the title field, type exactly this, without pressing Enter: Backup of C:\\Users\\anna\\notes",
        "Backup of C:\\Users\\anna\\notes",
        False,
        "backslash",
    ),
    (
        "hard-03",
        "type-02",
        "In the shop's search box, type exactly this and press Enter: {hdmi,usb-c} cable",
        "{hdmi,usb-c} cable",
        True,
        "braces",
    ),
    (
        "hard-04",
        "type-06",
        "In the message box, type two lines, 'Running late' on the first line and 'Start without me' on the "
        "second, and don't press Enter.",
        "Running late\nStart without me",
        False,
        "newline",
    ),
    (
        "hard-05",
        "type-04",
        "In the display name field, type exactly this, without pressing Enter: Zoë Ångström 🎉",
        "Zoë Ångström 🎉",
        False,
        "accents and emoji",
    ),
    (
        "hard-06",
        "type-05",
        "In the message box, type exactly this and press Enter to send it: " + LONG_TEXT,
        LONG_TEXT,
        True,
        "long text",
    ),
)

# ------------------------------------------------------------------------------------------ C-pixel
# (id, source dev case whose target is clicked, pixel (x, y)) -- normalisation by hand in the module docstring
C_PIXEL: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("cpix-01", "click-03", (1024, 304)),
    ("cpix-02", "click-06", (1152, 184)),
    ("cpix-03", "type-05", (1088, 748)),
    ("cpix-04", "type-07", (1120, 168)),
)

# ------------------------------------------------------------------------------------------ U-under
# (id, source dev case whose target is typed into, user text, text)
U_UNDER: tuple[tuple[str, str, str, str], ...] = (
    (
        "under-01",
        "type-01",
        'Type "lena.fischer@northwind.example" into the email field.',
        "lena.fischer@northwind.example",
    ),
    ("under-02", "type-02", 'Type "wireless mouse" into the search box.', "wireless mouse"),
    ("under-03", "type-10", 'Type "sunset" into the photo search box.', "sunset"),
    ("under-04", "type-05", 'Type "On my way" into the message box.', "On my way"),
)


# --------------------------------------------------------------------------------------------- build


def _dev_cases() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in SOURCE_SET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            if c["split"] == "dev":
                out[c["id"]] = c
    return out


def _screen(name: str) -> str:
    return f"screens/{name}.jpg"


def _base(cid: str, category: str, screen: str, user: str, tool: str) -> dict[str, Any]:
    return {
        "category": category,
        "id": cid,
        "out_combo": None,
        "pixel": None,
        "reference_call": None,
        "screen": screen,
        "screen_size": list(SCREEN_SIZE),
        "source_case": None,
        "split": "dev",
        "target": None,
        "tool": tool,
        "user": user,
    }


def build() -> list[dict[str, Any]]:
    """The 30 cases in file order (category order of criterion 1)."""
    dev = _dev_cases()
    out: list[dict[str, Any]] = []
    for cid, screen, user, combo in K_IN:
        c = _base(cid, "K-in", _screen(screen), user, "press_key")
        c["gt"] = {"combo": combo, "name": "press_key"}
        c["reference_call"] = {"arguments": {"combo": combo}, "name": "press_key"}
        out.append(c)
    for cid, screen, user, combo in K_OUT:
        c = _base(cid, "K-out", _screen(screen), user, "press_key")
        c["gt"] = None
        c["out_combo"] = combo
        out.append(c)
    for cid, screen, user, direction in S_EXTRA:
        c = _base(cid, "S-extra", _screen(screen), user, "scroll")
        c["gt"] = {"direction": direction, "name": "scroll"}
        c["reference_call"] = {"arguments": {"direction": direction}, "name": "scroll"}
        out.append(c)
    for cid, src, user, text, enter, _feature in T_HARD:
        s = dev[src]
        bbox = list(s["target"]["bbox"])
        c = _base(cid, "T-hard", s["screen"], user, "type_at")
        c["source_case"] = src
        c["target"] = dict(s["target"])
        c["gt"] = {"bbox": bbox, "name": "type_at", "press_enter": enter, "text": text}
        x, y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        c["reference_call"] = {
            "arguments": {"press_enter": enter, "text": text, "x": x, "y": y},
            "name": "type_at",
        }
        out.append(c)
    for cid, src, (px, py) in C_PIXEL:
        s = dev[src]
        bbox = list(s["target"]["bbox"])
        user = f"Click at pixel ({px}, {py}) on this 1280x800 screen."
        c = _base(cid, "C-pixel", s["screen"], user, "click_at")
        c["source_case"] = src
        c["target"] = dict(s["target"])
        c["pixel"] = [px, py]
        c["gt"] = {"bbox": bbox, "name": "click_at"}
        c["reference_call"] = {
            "arguments": {"x": px * 1000 // SCREEN_SIZE[0], "y": py * 1000 // SCREEN_SIZE[1]},
            "name": "click_at",
        }
        out.append(c)
    for cid, src, user, text in U_UNDER:
        s = dev[src]
        bbox = list(s["target"]["bbox"])
        c = _base(cid, "U-under", s["screen"], user, "type_at")
        c["source_case"] = src
        c["target"] = dict(s["target"])
        c["gt"] = {"bbox": bbox, "name": "type_at", "press_enter": None, "text": text}
        x, y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        # press_enter is reported, not scored (criterion 1); the reference picks False arbitrarily.
        c["reference_call"] = {
            "arguments": {"press_enter": False, "text": text, "x": x, "y": y},
            "name": "type_at",
        }
        out.append(c)
    return out


def render_cases(cases: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n" for c in cases).encode(
        "utf-8"
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_manifest(cases_bytes: bytes, approval: str | None) -> bytes:
    counts: dict[str, int] = {}
    for line in cases_bytes.decode("utf-8").splitlines():
        cat = json.loads(line)["category"]
        counts[cat] = counts.get(cat, 0) + 1
    doc = {
        # H-T99a: the human's approval of the case set, recorded here (by the test-author) when given; the
        # integrity rows refuse to pass while it is null.
        "approval": approval,
        "cases_sha256": _sha(cases_bytes),
        "counts": counts,
        "generator": {"cases_src.py": _sha(Path(__file__).read_bytes())},
        "n_cases": sum(counts.values()),
        "normalisation": "pixel p lies in cell n = floor(p*1000/W); n stands for pixel n*W/1000 (v1 set rule)",
        "set": "toolcall-stress-v1",
        "source_set_sha256": _sha(SOURCE_SET.read_bytes()),
        "split": "dev",
        "task": "T-0099",
    }
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _current_approval() -> str | None:
    if not MANIFEST.exists():
        return None
    value = json.loads(MANIFEST.read_text(encoding="utf-8")).get("approval")
    return value if isinstance(value, str) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if cases.jsonl or MANIFEST.json is stale"
    )
    args = ap.parse_args(argv)
    cases_bytes = render_cases(build())
    manifest_bytes = render_manifest(cases_bytes, _current_approval())
    if args.check:
        stale = [
            p.name
            for p, b in ((CASES, cases_bytes), (MANIFEST, manifest_bytes))
            if not p.exists() or p.read_bytes() != b
        ]
        if stale:
            print(f"stale: {', '.join(stale)}", file=sys.stderr)
            return 1
        return 0
    CASES.write_bytes(cases_bytes)
    MANIFEST.write_bytes(manifest_bytes)
    print(f"wrote {CASES.name} ({_sha(cases_bytes)}) and {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
