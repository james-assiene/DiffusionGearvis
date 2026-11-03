"""T-0111c criterion 8 probe: what every ``press_key`` combo does in the pinned Chromium.

Run (no network; the page is served on 127.0.0.1 only)::

    uv run python tests/fixtures/browser/probe_press_key.py

Plain Playwright, no ``local_astra.computer``: headless Chromium (versions.lock ``[browser]``), a fresh context per
row at 1280x800, DSF 1, ``keys.html`` from this directory. For each combo and each setup the probe prints the
``keydown`` events (key, code, ctrlKey, altKey), the non-key events, the change in ``__state()`` and the page count.
The output recorded on 2026-10-20 is in ``MANIFEST.md``; the expectations of
``tests/integration/test_computer_browser_e2e.py::test_every_press_key_combo_effect_or_refusal`` are read from it.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import threading
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
COMBOS = [
    "Enter", "Escape", "Tab", "Backspace", "Delete", "Space", "ArrowUp", "ArrowDown", "ArrowLeft",
    "ArrowRight", "Home", "End", "PageUp", "PageDown", "Ctrl+A", "Ctrl+C", "Ctrl+V", "Ctrl+X", "Ctrl+Z",
    "Ctrl+L", "Ctrl+T", "Ctrl+W", "Alt+Tab", "Ctrl+Enter",
]  # fmt: skip
SETUPS = (
    "inp_typed",
    "ta_caret",
    "ta_selected",
    "body_scrolled",
    "btn_focused",
    "dialog",
    "copied",
)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def pw_key(combo: str) -> str:
    if combo == "Space":
        return " "
    return combo.replace("Ctrl+", "Control+")


async def setup(page: Any, name: str) -> None:
    if name == "inp_typed":  # typed (so there is an undo step), caret before the last character
        await page.click("#inp")
        await page.keyboard.type("abc")
        await page.keyboard.press("ArrowLeft")
    elif name == "ta_caret":
        await page.evaluate("document.getElementById('ta').focus(); ta.setSelectionRange(12, 12)")
    elif name == "ta_selected":
        await page.evaluate("document.getElementById('ta').focus(); ta.setSelectionRange(0, 8)")
    elif name == "body_scrolled":
        await page.evaluate("scrollTo(0, 1000); document.activeElement.blur()")
    elif name == "btn_focused":
        await page.evaluate("document.getElementById('btn').focus()")
    elif name == "dialog":
        await page.evaluate("document.getElementById('dlg').showModal()")
    elif name == "copied":  # 'line one' copied with a real Ctrl+C, then #b focused
        await page.evaluate("document.getElementById('ta').focus(); ta.setSelectionRange(0, 8)")
        await page.keyboard.press("Control+c")
        await page.evaluate("document.getElementById('b').focus()")


async def main() -> None:
    handler = functools.partial(_Quiet, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/keys.html"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        print("chromium", browser.version, p.chromium.executable_path)
        for combo in COMBOS:
            for name in SETUPS:
                ctx = await browser.new_context(
                    viewport={"width": 1280, "height": 800}, device_scale_factor=1
                )
                page = await ctx.new_page()
                await page.goto(url)
                await setup(page, name)
                await page.evaluate("window.__log.length = 0")
                before = await page.evaluate("__state()")
                await page.keyboard.press(pw_key(combo))
                await page.evaluate(
                    "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
                )
                after = await page.evaluate("__state()")
                log = await page.evaluate("window.__log")
                diff = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
                kd = [
                    (e["key"], e["code"], e["ctrlKey"], e["altKey"])
                    for e in log
                    if e["type"] == "keydown"
                ]
                other = sorted(
                    {e["type"] + (f"={e['clip']!r}" if "clip" in e else "") for e in log}
                    - {"keydown", "keyup"}
                )
                print(
                    f"{combo:10s} {name:13s} keydown={kd} other={other} diff={json.dumps(diff)} pages={len(ctx.pages)}"
                )
                await ctx.close()
        await browser.close()
    srv.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
