#!/usr/bin/env python3
"""Screenshot the HUD in both themes at four moments of a real session (T-0128 criterion 15).

    uv run --extra browser python scripts/demo/hud_screens.py --out logs/hud/T-0128-<sha>/

What it does, in order:

1. serves ``hud/`` with the production ``HudStaticServer`` on a loopback ephemeral port and starts
   a loopback ``HudStub`` (``tests/fakes/hud_server.py``) for the page's socket;
2. opens the page in Playwright's Chromium at 1440x900, DPR 1, with ``?source=canvas`` (the
   labelled synthetic capture: no screen is shared);
3. replays the committed fixture ``tests/fixtures/hud/session_20261020_t3_t11.jsonl`` (turns t3-t11
   of the 2026-10-20 desk session, recorded ``wire.out`` messages, MANIFEST.md has its SHA-256)
   through ``HudStub.push`` in fixture order, stopping after the last message of t9 (the
   shopping-list card), t10 (the search turn and its action card) and t11 (the multi-utterance
   turn, after ``astra.spoken.played``), plus once before any message (idle);
3a. every card is the one the *current* gate makes (T-0128 ruling on f6b7d98, T-0154 C5): the
   wire's ``astra.ui.surface`` lines predate T-0127 (t9's carry an empty root Column, t3's and
   t8's no root at all), so for every turn that has raw ``ui`` commits in
   ``tests/fixtures/a2ui/desk_20261020_card_children.jsonl`` (t3, t5, t6, t8, t9, t10; read with
   ``scripts/replay_ui_gate.py``'s ``ui_commits``) the wire surfaces are withheld (the page's
   bridge would skip any later canvas at or below their ``canvas_idx``). Instead that turn's
   commits go through one fresh ``UiGate`` per turn as ``Session._gate_surface`` does, and right
   after the wire replay has pushed the turn's last message every emitted surface is pushed, then
   the last one again with ``final=True``, as ``Session._publish_final_surface`` does. Turns the
   UI fixture lacks (t4) keep their wire surfaces;
4. at each stop (the thread keeps itself at its bottom, T-0154 B9; nothing is scrolled by hand,
   B12), shoots the page once per theme. The theme is chosen the way it is for a human with nothing stored: the system
   colour scheme is emulated as dark, then light, and ``hud/src/theme.js`` follows it live; the
   script checks ``<html data-theme>`` before each shot.

It writes ``{theme}-{n}-{state}.png`` (eight PNGs) and ``requests.txt``: every URL the page
requested, HTTP and WebSocket, one per line, so a reviewer can check that only 127.0.0.1 appears.
It exits non-zero if any request left loopback.

With ``--compare DIR`` (T-0154 A8; DIR is the committed mockup, ``tests/fixtures/hud/
mockup_20261021/``) it also serves DIR from its own loopback static server, renders its
``hud.html?theme={theme}`` at the same 1440x900, DPR 1, in the same Chromium, and writes
``compare-{theme}.png``: the mockup on the left, the HUD's t9 shot on the right, composed by that
Chromium (a page of two ``data:`` images, so Pillow is not needed), under a one-line label strip.
It writes ``fonts.txt`` too: ``fc-match`` for ``system-ui`` and ``sans-serif``, the faces both
pages actually render in. The mockup's requests are logged in ``requests.txt`` with the HUD's.
It also writes ``compare-cards-{theme}.png`` (T-0154 C6): the mockup's shopping-list card from
``cards.html?theme={theme}`` on the left and the HUD's t9 card (taken at the t9 stop, inside the
surface's shadow root) on the right, each cropped to the card plus a 16 px margin.
And (T-0164 C5) ``compare-genui-{theme}.png``: for each of the four payloads the option-A mock
draws (trial-omelette, desk-t8, desk-t9, trial-flight; final canvases from
``tests/fixtures/a2ui/genui_census_20261027.jsonl``), the mock's card from
``tests/fixtures/hud/genui_20261027/cards.html?theme={theme}`` on the left and the HUD's card on
the right, one row per payload. Each payload is the only turn of a fresh HUD page (the production
``HudStaticServer`` and a loopback ``HudStub``); its crops are also written as
``genui-cards/{key}-{theme}.png``. The mock page is rendered 3000 px tall so its four cards are in
view unscrolled.

With ``--genui-v2`` (T-0184 B10) it writes only ``genui-v2-{dark,light}.png`` (and
``requests.txt``): every payload of ``tests/fixtures/a2ui/cards_v2_exemplars.jsonl`` (every v2 kind,
then the approved Main, Compare and Chart boards rebuilt from their v2 payloads ``board-main``,
``board-compare`` and ``board-chart``), pushed as the cards of one fresh HUD page (the production
``HudStaticServer`` and a loopback ``HudStub``) one at a time, each cropped to its box plus a 16 px
margin in both themes while it is the newest card at the thread's bottom (B9; nothing is scrolled,
B12), one labelled row per payload. The session replay above is skipped.

Nothing here spends GPU time and nothing is written under ``artifacts/``; ``logs/`` is gitignored.
The fixture carries no ``astra.canvas.*`` message, so the Thinking pane shows its empty state.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import importlib.util
import json
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # so `python scripts/demo/hud_screens.py` works from anywhere
    sys.path.insert(0, str(REPO))

from local_astra.conductor.ui_gate import UiGate  # noqa: E402
from local_astra.contracts.protocol import AstraUiSurface  # noqa: E402
from local_astra.hud_static import HudStaticServer  # noqa: E402
from tests.fakes.hud_server import HudStub  # noqa: E402

HUD = REPO / "hud"
FIXTURE = REPO / "tests" / "fixtures" / "hud" / "session_20261020_t3_t11.jsonl"
UI_FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "desk_20261020_card_children.jsonl"
REPLAY_UI_GATE = REPO / "scripts" / "replay_ui_gate.py"
#: The turn whose card is re-gated, and the surface id the session gives it.
CARD_TURN = "t9"
CARD_SURFACE_ID = f"turn-{CARD_TURN}"
VIEWPORT = {"width": 1440, "height": 900}
THEMES = ("dark", "light")
#: (file stem, the turn whose last fixture message precedes the shot); None is the idle shot.
STATES: tuple[tuple[str, str | None], ...] = (
    ("1-idle", None),
    ("2-t9-card", "t9"),
    ("3-t10-search", "t10"),
    ("4-t11-played", "t11"),
)
#: Pause between two pushed messages, so the page renders canvases one at a time as it would live.
PUSH_GAP_S = 0.01
SETTLE_S = 1.0
TWO_FRAMES_JS = (
    "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
)
#: The panes that mount: the shell's four (conversation, What I see, Thinking, Diagnostics — T-0154
#: B1 made the three conversation panes one) plus the assists and session-log panels.
PANES = 6
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
#: For the card shot: whether the card turn's gated surface rendered in its card slot. Nothing is
#: scrolled by hand (T-0154 B12): the thread keeps itself at its bottom (B9).
CARD_RENDERED_JS = f"!!document.querySelector('[data-thread] [data-ui-surface=\"{CARD_SURFACE_ID}\"] [data-a2ui-root]')"


def turn_of(msg: dict[str, Any]) -> str | None:
    turn = msg.get("turn_id")
    if turn is None and str(msg.get("surface_id") or "").startswith("turn-"):
        turn = str(msg["surface_id"])[len("turn-") :]
    return None if turn is None else str(turn)


def last_index(lines: list[str], turn: str) -> int:
    hits = [i for i, line in enumerate(lines) if turn_of(json.loads(line)) == turn]
    if not hits:
        raise SystemExit(f"the fixture has no message for turn {turn}")
    return hits[-1]


def is_regated_surface(line: str, turns: dict[str, Any]) -> bool:
    """A wire ``astra.ui.surface`` of a re-gated turn: pre-T-0127, withheld from the replay."""
    msg = json.loads(line)
    sid = str(msg.get("surface_id") or "")
    return bool(
        msg.get("type") == "astra.ui.surface"
        and sid.startswith("turn-")
        and sid[len("turn-") :] in turns
    )


def _ui_commits() -> dict[str, list[dict[str, Any]]]:
    spec = importlib.util.spec_from_file_location("_hud_screens_replay_ui_gate", REPLAY_UI_GATE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {REPLAY_UI_GATE}")
    replay_ui_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_ui_gate)
    commits: dict[str, list[dict[str, Any]]] = replay_ui_gate.ui_commits(UI_FIXTURE)
    return commits


def gated_surfaces(turn: str, commits: list[dict[str, Any]]) -> list[AstraUiSurface]:
    """What the session publishes for one turn today: its ``ui`` commits through one fresh
    ``UiGate`` (as ``Session._gate_surface``), each emitted surface, then the last as final (as
    ``Session._publish_final_surface``)."""
    gate = UiGate()
    surfaces = [
        r.surface
        for r in (
            gate.gate(
                c.get("text", ""),
                turn_id=turn,
                canvas_idx=int(c.get("canvas_idx", 0)),
                final=False,
            )
            for c in commits
        )
        if r.surface is not None
    ]
    if not surfaces:
        raise SystemExit(f"the gate made no surface from {turn}'s ui commits")
    if surfaces[-1].surface_id != f"turn-{turn}":
        raise SystemExit(f"the gated surface is {surfaces[-1].surface_id!r}, not turn-{turn}")
    return [*surfaces, surfaces[-1].model_copy(update={"final": True})]


def gated_turns() -> dict[str, list[AstraUiSurface]]:
    """T-0154 C5: every turn the UI fixture carries ``ui`` commits for (t3, t5, t6, t8, t9, t10),
    each through its own fresh gate, in the fixture's turn order."""
    turns = {turn: gated_surfaces(turn, commits) for turn, commits in _ui_commits().items()}
    if CARD_TURN not in turns:
        raise SystemExit(f"{UI_FIXTURE.name} has no ui commits for {CARD_TURN}")
    return turns


#: The HUD shot the composites put beside the mockup (the mockup draws the t9 shopping-list turn).
COMPARE_STEM = "2-t9-card"
LABEL_H = 28


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def serve_dir(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    """A loopback static server for the mockup directory (never ``HudStaticServer``: A1 keeps the
    mockup out of everything ``astra hud`` serves)."""
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def composite_html(left: bytes, right: bytes, left_label: str, right_label: str) -> str:
    """Two 1440x900 PNGs side by side under a label strip, as a page with no external request."""
    w, h = VIEWPORT["width"], VIEWPORT["height"]

    def img(data: bytes) -> str:
        b64 = base64.b64encode(data).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'

    return (
        "<!doctype html><html><head><style>"
        "html,body{margin:0;background:#202020;color:#f0f0f0;font:14px/28px sans-serif}"
        f".row{{display:flex;width:{2 * w}px}} .row>div{{width:{w}px;padding:0 12px;"
        f"box-sizing:border-box;height:{LABEL_H}px}} img{{display:block}}"
        "</style></head><body>"
        f'<div class="row"><div>{left_label}</div><div>{right_label}</div></div>'
        f'<div class="row">{img(left)}{img(right)}</div>'
        "</body></html>"
    )


def compare(
    browser: Any, mockup: Path, hud_shots: dict[str, Path], out: Path, requests: list[str]
) -> list[Path]:
    """T-0154 A8: ``compare-{theme}.png``, the mockup (left) and the HUD's t9 shot (right)."""
    if not (mockup / "hud.html").is_file():
        raise SystemExit(f"--compare: {mockup} has no hud.html")
    server, port = serve_dir(mockup)
    written: list[Path] = []
    try:
        for theme in THEMES:
            context = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=1, color_scheme=theme
            )
            try:
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.goto(f"http://127.0.0.1:{port}/hud.html?theme={theme}")
                page.evaluate("document.fonts.ready.then(() => " + TWO_FRAMES_JS + ")")
                got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                if got != theme:
                    raise SystemExit(f"the mockup did not take ?theme={theme}: {got!r}")
                mock_png = page.screenshot()
                out_page = context.new_page()
                out_page.set_viewport_size(
                    {"width": 2 * VIEWPORT["width"], "height": VIEWPORT["height"] + LABEL_H}
                )
                out_page.set_content(
                    composite_html(
                        mock_png,
                        hud_shots[theme].read_bytes(),
                        f"mockup_20261021/hud.html ({theme})",
                        f"HUD {hud_shots[theme].name}",
                    )
                )
                out_page.evaluate(TWO_FRAMES_JS)
                path = out / f"compare-{theme}.png"
                out_page.screenshot(path=str(path))
                written.append(path)
            finally:
                context.close()
    finally:
        server.shutdown()
        server.server_close()
    return written


#: T-0154 C6: the rectangles the card composites are cropped to (the card's own box). The HUD's
#: card is the renderer's light-DOM card root (T-0164 B); the mockup's is its shopping-list card.
HUD_CARD_RECT_JS = (
    "(() => { const c = document.querySelector("
    f"'[data-thread] [data-ui-surface=\"{CARD_SURFACE_ID}\"] [data-a2ui-root]');"
    " if (!c) return null; const r = c.getBoundingClientRect();"
    " return {x: r.x, y: r.y, width: r.width, height: r.height}; })()"
)
MOCK_CARD_RECT_JS = (
    "(() => { const c = [...document.querySelectorAll('.a2-card')]"
    ".find((e) => e.querySelector('.a2-list'));"
    " if (!c) return null; const r = c.getBoundingClientRect();"
    " return {x: r.x, y: r.y, width: r.width, height: r.height}; })()"
)
#: Room around the card in its crop, so its border and shadow are in the picture.
CARD_PAD = 16
CARD_GAP = 24


def crop(page: Any, rect_js: str, what: str, view: dict[str, int] = VIEWPORT) -> bytes:
    """A screenshot of ``what``'s box (plus ``CARD_PAD``), clipped to the viewport ``view``.
    Nothing is scrolled: the HUD's thread keeps itself at its bottom (B9, B12), where the card is."""
    rect = page.evaluate(rect_js)
    if not rect:
        raise SystemExit(f"C6: {what} was not found")
    x0 = max(0.0, float(rect["x"]) - CARD_PAD)
    y0 = max(0.0, float(rect["y"]) - CARD_PAD)
    x1 = min(float(view["width"]), float(rect["x"]) + float(rect["width"]) + CARD_PAD)
    y1 = min(float(view["height"]), float(rect["y"]) + float(rect["height"]) + CARD_PAD)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise SystemExit(f"C6: {what} is outside the viewport: {rect}")
    if float(rect["y"]) < 0 or float(rect["y"]) + float(rect["height"]) > view["height"]:
        print(f"note: {what} is only partly in view; its crop is clipped to the viewport: {rect}")
    data: bytes = page.screenshot(clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    return data


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk."""
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def cards_html(
    left: bytes, right: bytes, left_label: str, right_label: str
) -> tuple[str, int, int]:
    """Two card crops side by side, top-aligned, under a label strip; no external request.
    Returns the page and the size it needs."""
    (lw, lh), (rw, rh) = png_size(left), png_size(right)

    def img(data: bytes, w: int, h: int) -> str:
        b64 = base64.b64encode(data).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'

    return (
        (
            "<!doctype html><html><head><style>"
            "html,body{margin:0;background:#202020;color:#f0f0f0;font:14px/28px sans-serif}"
            f".row{{display:flex;gap:{CARD_GAP}px;align-items:flex-start}}"
            f".row>div{{padding:0 4px;box-sizing:border-box;height:{LABEL_H}px;white-space:nowrap;"
            "overflow:hidden} img{display:block}"
            "</style></head><body>"
            f'<div class="row"><div style="width:{lw}px">{left_label}</div>'
            f'<div style="width:{rw}px">{right_label}</div></div>'
            f'<div class="row">{img(left, lw, lh)}{img(right, rw, rh)}</div>'
            "</body></html>"
        ),
        lw + CARD_GAP + rw,
        LABEL_H + max(lh, rh),
    )


def compare_cards(
    browser: Any, mockup: Path, hud_cards: dict[str, bytes], out: Path, requests: list[str]
) -> list[Path]:
    """T-0154 C6: ``compare-cards-{theme}.png``, the mockup's shopping-list card (left, from
    ``cards.html?theme={theme}``) beside the HUD's t9 card (right), each cropped to the card."""
    if not (mockup / "cards.html").is_file():
        raise SystemExit(f"--compare: {mockup} has no cards.html")
    server, port = serve_dir(mockup)
    written: list[Path] = []
    try:
        for theme in THEMES:
            if theme not in hud_cards:
                raise SystemExit(f"C6: the HUD's t9 card was not captured in {theme}")
            context = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=1, color_scheme=theme
            )
            try:
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.goto(f"http://127.0.0.1:{port}/cards.html?theme={theme}")
                page.evaluate("document.fonts.ready.then(() => " + TWO_FRAMES_JS + ")")
                got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                if got != theme:
                    raise SystemExit(f"the mockup did not take ?theme={theme}: {got!r}")
                mock_png = crop(page, MOCK_CARD_RECT_JS, "the mockup's shopping-list card")
                html, width, height = cards_html(
                    mock_png,
                    hud_cards[theme],
                    f"mockup_20261021/cards.html ({theme})",
                    f"HUD {CARD_SURFACE_ID} card ({theme})",
                )
                out_page = context.new_page()
                out_page.set_viewport_size({"width": width, "height": height})
                out_page.set_content(html)
                out_page.evaluate(TWO_FRAMES_JS)
                path = out / f"compare-cards-{theme}.png"
                out_page.screenshot(path=str(path))
                written.append(path)
            finally:
                context.close()
    finally:
        server.shutdown()
        server.server_close()
    return written


#: T-0164 C5: the committed option-A mock (the test-author's fixture), the four payloads it draws,
#: and where their gated canvases live (T-0164 group A's census fixture).
GENUI_MOCK = REPO / "tests" / "fixtures" / "hud" / "genui_20261027"
GENUI_CENSUS = REPO / "tests" / "fixtures" / "a2ui" / "genui_census_20261027.jsonl"
GENUI_KEYS = ("trial-omelette", "desk-t8", "desk-t9", "trial-flight")
#: The mock's card for one payload (``cards.html`` draws each payload once, keyed by its key).
MOCK_GENUI_CARD = '[data-payload="{key}"] .ga-card'
#: The mock page draws the four cards one under another; it is rendered this tall so all four are
#: in view without scrolling it (B12: the script scrolls nothing).
GENUI_MOCK_VIEW = {"width": VIEWPORT["width"], "height": 3000}
#: The HUD card crops go here, apart from the eight full-page PNGs (T-0128 criterion 15).
GENUI_CARDS_DIR = "genui-cards"


def genui_surfaces() -> dict[str, AstraUiSurface]:
    """Each C5 payload's final canvas as the session publishes it (one ``final`` surface)."""
    found: dict[str, AstraUiSurface] = {}
    for line in GENUI_CENSUS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else {}
        key = row.get("key")
        if key in GENUI_KEYS:
            found[key] = AstraUiSurface(
                surface_id=row["surface_id"],
                canvas_idx=int(row["canvas_idxs"][-1]),
                final=True,
                messages=tuple(row["canvases"][-1]),
            )
    missing = [k for k in GENUI_KEYS if k not in found]
    if missing:
        raise SystemExit(f"C5: {GENUI_CENSUS.name} has no payload {missing}")
    return found


def card_rect_js(selector: str) -> str:
    return (
        f"(() => {{ const c = document.querySelector({json.dumps(selector)});"
        " if (!c) return null; const r = c.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height}; })()"
    )


def hud_genui_cards(browser: Any, requests: list[str]) -> dict[str, dict[str, bytes]]:
    """C5: each payload pushed as the only turn of a fresh HUD page (``HudStaticServer`` and a
    ``HudStub``, loopback), its card cropped in both themes. ``{theme: {key: png}}``."""
    shots: dict[str, dict[str, bytes]] = {t: {} for t in THEMES}
    server = HudStaticServer(HUD, 0)
    try:
        for key, surface in genui_surfaces().items():
            # The surface is the stub's script, not a HudStub.push: this page is its own one-turn
            # session, apart from the replay above. It is sent on the page's first astra.frame
            # (the synthetic capture), which comes after the panels have mounted; a surface sent
            # on astra.hello arrives before the thread can take it.
            stub = HudStub([surface], send_after="astra.frame").start()
            context = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=1, color_scheme=THEMES[0]
            )
            try:
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.on("websocket", lambda ws: requests.append(ws.url))
                page.goto(
                    f"http://127.0.0.1:{server.port}/?source=canvas&ws="
                    + urllib.parse.quote(stub.url, safe="")
                )
                if not stub.script_sent.wait(20):
                    raise SystemExit(f"C5: the stub never sent {key}'s surface")
                page.wait_for_function(
                    f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
                )
                root = f'[data-thread] [data-ui-surface="{surface.surface_id}"] [data-a2ui-root]'
                page.wait_for_function(
                    f"!!document.querySelector({json.dumps(root)})", timeout=10_000
                )
                time.sleep(SETTLE_S)
                for theme in THEMES:
                    page.emulate_media(color_scheme=theme)
                    page.evaluate(TWO_FRAMES_JS)
                    got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                    if got != theme:
                        raise SystemExit(f"emulated {theme}, but the page shows {got!r}")
                    shots[theme][key] = crop(page, card_rect_js(root), f"the HUD's {key} card")
            finally:
                context.close()
                stub.stop()
    finally:
        server.close()
    return shots


def genui_html(rows: list[tuple[str, bytes, bytes]], theme: str) -> tuple[str, int, int]:
    """One labelled row per payload: the mock's option-A card (left) beside the HUD's (right)."""
    parts: list[str] = []
    width = height = 0
    for key, left, right in rows:
        html, w, h = cards_html(
            left, right, f"{key}: genui_20261027 option A ({theme})", f"{key}: HUD ({theme})"
        )
        parts.append(html.split("<body>", 1)[1].rsplit("</body>", 1)[0])
        width, height = max(width, w), height + h + CARD_GAP
    head = cards_html(rows[0][1], rows[0][2], "", "")[0].split("<body>", 1)[0]
    return head + "<body>" + "".join(parts) + "</body></html>", width, height


def compare_genui(
    browser: Any, hud_cards: dict[str, dict[str, bytes]], out: Path, requests: list[str]
) -> list[Path]:
    """T-0164 C5: ``compare-genui-{theme}.png``, the four payloads, each the mock's option-A card
    (``genui_20261027/cards.html?theme={theme}``) beside the HUD's card, cropped to the cards."""
    if not (GENUI_MOCK / "cards.html").is_file():
        raise SystemExit(f"C5: {GENUI_MOCK} has no cards.html")
    server, port = serve_dir(GENUI_MOCK)
    written: list[Path] = []
    try:
        for theme in THEMES:
            context = browser.new_context(
                viewport=GENUI_MOCK_VIEW, device_scale_factor=1, color_scheme=theme
            )
            try:
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.goto(f"http://127.0.0.1:{port}/cards.html?theme={theme}")
                page.evaluate("document.fonts.ready.then(() => " + TWO_FRAMES_JS + ")")
                got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                if got != theme:
                    raise SystemExit(f"the mock did not take ?theme={theme}: {got!r}")
                rows = []
                for key in GENUI_KEYS:
                    mock_png = crop(
                        page,
                        card_rect_js(MOCK_GENUI_CARD.format(key=key)),
                        f"the mock's {key}",
                        GENUI_MOCK_VIEW,
                    )
                    rows.append((key, mock_png, hud_cards[theme][key]))
                html, width, height = genui_html(rows, theme)
                out_page = context.new_page()
                out_page.set_viewport_size({"width": width, "height": height})
                out_page.set_content(html)
                out_page.evaluate(TWO_FRAMES_JS)
                path = out / f"compare-genui-{theme}.png"
                out_page.screenshot(path=str(path), full_page=True)
                written.append(path)
            finally:
                context.close()
    finally:
        server.shutdown()
        server.server_close()
    return written


#: T-0184 B10: group A's v2 payloads, one per kind and edge case, then the three boards.
GENUI_V2_EXEMPLARS = REPO / "tests" / "fixtures" / "a2ui" / "cards_v2_exemplars.jsonl"
#: Tall enough that the tallest exemplar card (nine Stats) is in view whole at the thread's bottom.
GENUI_V2_VIEW = {"width": VIEWPORT["width"], "height": 2400}


def genui_v2_surfaces() -> list[tuple[str, str | None, AstraUiSurface]]:
    """``(key, board, surface)`` per exemplar, in the fixture's order, as one ``final`` surface."""
    out: list[tuple[str, str | None, AstraUiSurface]] = []
    for line in GENUI_V2_EXEMPLARS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        sid = next(m["createSurface"]["surfaceId"] for m in messages if "createSurface" in m)
        surface = AstraUiSurface(surface_id=sid, canvas_idx=0, final=True, messages=tuple(messages))
        out.append((str(row["key"]), row.get("board"), surface))
    if not out:
        raise SystemExit(f"B10: {GENUI_V2_EXEMPLARS.name} has no payload")
    return out


def v2_card_settled_js(surface_id: str) -> str:
    """Whether the card root of ``surface_id`` is wholly in the viewport and kept its box over two
    frames. Nothing is scrolled (B12): the thread keeps itself at its bottom (B9), where the card
    just pushed is."""
    sel = f'[data-thread] [data-ui-surface="{surface_id}"] [data-a2ui-root]'
    return (
        f"(() => {{ const box = () => {{ const c = document.querySelector({json.dumps(sel)});"
        " if (!c) return null; const r = c.getBoundingClientRect();"
        " return [r.x, r.y, r.width, r.height]; };"
        " const a = box();"
        " return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => {"
        " const b = box();"
        " res(!!a && !!b && a.every((v, i) => v === b[i]) && b[1] >= 0"
        " && b[1] + b[3] <= innerHeight); }))); })()"
    )


def wait_settled(page: Any, js: str, what: str, timeout_s: float = 10.0) -> None:
    """Polls ``js`` (a promise of a boolean) until it is true, or exits naming ``what``."""
    deadline = time.monotonic() + timeout_s
    while not page.evaluate(js):
        if time.monotonic() > deadline:
            raise SystemExit(f"B10: {what} never settled wholly in view (nothing is scrolled)")


def genui_v2_html(rows: list[tuple[str, bytes]]) -> tuple[str, int, int]:
    """One labelled row per card crop, top to bottom; a page with no external request."""
    parts: list[str] = []
    width = height = 0
    for label, png in rows:
        w, h = png_size(png)
        b64 = base64.b64encode(png).decode("ascii")
        parts.append(
            f'<div class="label" style="width:{w}px">{label}</div>'
            f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'
        )
        width, height = max(width, w), height + LABEL_H + h + CARD_GAP
    html = (
        "<!doctype html><html><head><style>"
        "html,body{margin:0;background:#202020;color:#f0f0f0;font:14px/28px sans-serif}"
        f".label{{padding:0 4px;box-sizing:border-box;height:{LABEL_H}px;white-space:nowrap;"
        f"overflow:hidden}} img{{display:block;margin-bottom:{CARD_GAP}px}}"
        "</style></head><body>" + "".join(parts) + "</body></html>"
    )
    return html, width, height


def genui_v2(out: Path, requests: list[str]) -> list[Path]:
    """T-0184 B10: ``genui-v2-{theme}.png``, every v2 exemplar card cropped, one row each."""
    from playwright.sync_api import sync_playwright

    surfaces = genui_v2_surfaces()
    written: list[Path] = []
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(
            stub.url, safe=""
        )
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(
                    viewport=GENUI_V2_VIEW, device_scale_factor=1, color_scheme=THEMES[0]
                )
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.on("websocket", lambda ws: requests.append(ws.url))
                page.goto(url)
                if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
                    raise SystemExit("B10: the page never said astra.hello to the stub")
                stub.wait_for_client(10)
                page.wait_for_function(
                    f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
                )
                # One card at a time: each is pushed, and the thread keeps itself at its bottom
                # (T-0154 B9), so the newest card is in view; it is cropped there in both themes
                # before the next one is pushed. Nothing is scrolled by hand (B12).
                rows_of: dict[str, list[tuple[str, bytes]]] = {t: [] for t in THEMES}
                for key, board, surface in surfaces:
                    stub.push(surface)
                    root = (
                        f'[data-thread] [data-ui-surface="{surface.surface_id}"] [data-a2ui-root]'
                    )
                    try:
                        page.wait_for_function(
                            f"!!document.querySelector({json.dumps(root)})", timeout=10_000
                        )
                    except Exception as exc:
                        raise SystemExit(f"B10: the {key} card did not render: {exc}") from exc
                    if not rows_of[THEMES[0]]:
                        time.sleep(SETTLE_S)
                    settled_js = v2_card_settled_js(surface.surface_id)
                    rect_js = card_rect_js(root)
                    what = f"board {board}, rebuilt from the v2 payload {key}" if board else key
                    for theme in THEMES:
                        page.emulate_media(color_scheme=theme)
                        page.evaluate(TWO_FRAMES_JS)
                        got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                        if got != theme:
                            raise SystemExit(f"B10: emulated {theme}, but the page shows {got!r}")
                        wait_settled(page, settled_js, f"the HUD's {key} card")
                        png = crop(page, rect_js, f"the HUD's {key} card", GENUI_V2_VIEW)
                        rows_of[theme].append((f"{what} ({theme})", png))
                for theme in THEMES:
                    rows = rows_of[theme]
                    html, width, height = genui_v2_html(rows)
                    out_page = context.new_page()
                    out_page.set_viewport_size({"width": width, "height": min(height, 16_000)})
                    out_page.set_content(html)
                    out_page.evaluate(TWO_FRAMES_JS)
                    path = out / f"genui-v2-{theme}.png"
                    out_page.screenshot(path=str(path), full_page=True)
                    out_page.close()
                    written.append(path)
                context.close()
            finally:
                browser.close()
    finally:
        server.close()
        stub.stop()
    return written


def write_fonts(out: Path) -> Path:
    """``fonts.txt``: the faces ``system-ui`` and ``sans-serif`` resolve to on this box."""
    lines = []
    for family in ("system-ui", "sans-serif"):
        try:
            got = subprocess.run(
                ["fc-match", family], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            got = f"(fc-match failed: {exc})"
        lines.append(f"fc-match {family}: {got}")
    path = out / "fonts.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def shoot(
    lines: list[str], out: Path, requests: list[str], mockup: Path | None = None
) -> list[Path]:
    from playwright.sync_api import sync_playwright

    shots: list[Path] = []
    regated = gated_turns()
    # each re-gated turn's surfaces go out right after that turn's last wire message (O-C (6))
    push_after = {last_index(lines, turn): turn for turn in regated}
    card_crops: dict[str, bytes] = {}
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(
            stub.url, safe=""
        )
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(
                    viewport=VIEWPORT, device_scale_factor=1, color_scheme=THEMES[0]
                )
                page = context.new_page()
                page.on("request", lambda r: requests.append(r.url))
                page.on("websocket", lambda ws: requests.append(ws.url))
                page.goto(url)
                if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
                    raise SystemExit("the page never said astra.hello to the stub")
                stub.wait_for_client(10)
                page.wait_for_function(
                    f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
                )
                cursor = 0
                for stem, turn in STATES:
                    if turn is not None:
                        end = last_index(lines, turn) + 1
                        for i in range(cursor, end):
                            if not is_regated_surface(lines[i], regated):
                                stub.push(lines[i])
                                time.sleep(PUSH_GAP_S)
                            for surface in regated.get(push_after.get(i, ""), []):
                                stub.push(surface)
                                time.sleep(PUSH_GAP_S)
                        cursor = end
                    time.sleep(SETTLE_S)
                    # The thread keeps the whole session and sticks to its bottom on its own
                    # (T-0154 B9), so the shot shows the turn it is named after unscrolled.
                    if turn == CARD_TURN and not page.evaluate(CARD_RENDERED_JS):
                        raise SystemExit(f"{CARD_TURN}'s gated card did not render")
                    for theme in THEMES:
                        page.emulate_media(color_scheme=theme)
                        page.evaluate(TWO_FRAMES_JS)
                        got = page.evaluate("document.documentElement.getAttribute('data-theme')")
                        if got != theme:
                            raise SystemExit(f"emulated {theme}, but the page shows {got!r}")
                        path = out / f"{theme}-{stem}.png"
                        page.screenshot(path=str(path))
                        shots.append(path)
                        if turn == CARD_TURN and mockup is not None:
                            card_crops[theme] = crop(page, HUD_CARD_RECT_JS, "the HUD's t9 card")
                    page.emulate_media(color_scheme=THEMES[0])
                if mockup is not None:
                    hud_shots = {t: out / f"{t}-{COMPARE_STEM}.png" for t in THEMES}
                    shots.extend(compare(browser, mockup, hud_shots, out, requests))
                    shots.extend(compare_cards(browser, mockup, card_crops, out, requests))
                    genui = hud_genui_cards(browser, requests)
                    (out / GENUI_CARDS_DIR).mkdir(exist_ok=True)
                    for theme, cards in genui.items():
                        for key, png in cards.items():
                            path = out / GENUI_CARDS_DIR / f"{key}-{theme}.png"
                            path.write_bytes(png)
                            shots.append(path)
                    shots.extend(compare_genui(browser, genui, out, requests))
            finally:
                browser.close()
    finally:
        server.close()
        stub.stop()
    return sorted(shots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="directory for the PNGs")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="the mockup directory (hud.html): also write compare-{dark,light}.png and fonts.txt",
    )
    parser.add_argument(
        "--genui-v2",
        action="store_true",
        help="T-0184 B10: write only genui-v2-{dark,light}.png, every v2 exemplar card",
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    if (REPO / "artifacts") in [out.resolve(), *out.resolve().parents]:
        raise SystemExit("criterion 15: the screenshots never go under artifacts/")
    out.mkdir(parents=True, exist_ok=True)
    requests: list[str] = []
    if args.genui_v2:
        if args.compare is not None:
            raise SystemExit("--genui-v2 writes the v2 composites only; drop --compare")
        shots = genui_v2(out, requests)
    else:
        lines = FIXTURE.read_text(encoding="utf-8").splitlines()
        shots = shoot(lines, out, requests, args.compare.resolve() if args.compare else None)
    if args.compare is not None:
        print(write_fonts(out))
    (out / "requests.txt").write_text("".join(u + "\n" for u in requests), encoding="utf-8")
    off_loopback = [
        u
        for u in requests
        if urllib.parse.urlsplit(u).scheme in {"http", "https", "ws", "wss"}
        and urllib.parse.urlsplit(u).hostname not in LOOPBACK
    ]
    for path in shots:
        print(path)
    print(f"{out / 'requests.txt'}: {len(requests)} requests, {len(off_loopback)} off loopback")
    for u in off_loopback:
        print(f"  NOT LOOPBACK: {u}")
    return 1 if off_loopback else 0


if __name__ == "__main__":
    raise SystemExit(main())
