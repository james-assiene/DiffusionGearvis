#!/usr/bin/env python3
"""Render the quiz fixture in the real HUD page and leave a PNG of it (T-0024 criterion 6).

    uv run python scripts/demo/hud_ui_demo.py --shot logs/hud/ui-quiz.png

What it does, in the order it does it:

1. takes ``tests/fixtures/a2ui/quiz_example.jsonl`` through the **real** ``UiGate``, canvas by
   canvas (prefix-cumulative at message boundaries: createSurface, then the 38-component
   updateComponents, then the updateDataModel — the same split as
   ``tests/integration/test_hud_ui_surface.py``);
2. serves ``hud/`` on loopback and opens it in headless Chrome, pointed at a loopback
   ``HudStub`` with ``?ws=``;
3. pushes each resulting ``AstraUiSurface`` over the socket, one canvas at a time, so the
   progressive morph really happens;
4. screenshots the settled page with CDP ``Page.captureScreenshot``.

If ``gate()`` returns no surface it prints the ``SurfaceReport`` and exits non-zero. It never
falls back to pushing the raw fixture: a demo that renders unvalidated JSON is exactly the lie
criterion 4 forbids (the ``astra.canvas.synthetic`` rule, docs/12 §2).

``logs/`` is gitignored and is not ``artifacts/``, and nothing here spends GPU time, so this
script needs no launch token. The CDP driver and the stub are the ones the integration tests use
(``tests/integration/test_hud_headless.py``, ``tests/fakes/hud_server.py``) — there is no second
implementation of either.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # so `python scripts/demo/hud_ui_demo.py` works from anywhere
    sys.path.insert(0, str(REPO))

from local_astra.conductor.ui_gate import UiGate  # noqa: E402
from local_astra.contracts.protocol import AstraUiSurface  # noqa: E402
from tests.fakes.hud_server import HudStub  # noqa: E402
from tests.integration.test_hud_headless import CHROME, Chrome, serve_hud  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"
LAST_PROMPT = "Which is larger: 5/6 or 7/9?"

DEEP_TEXT_JS = """
(() => {
  const out = [];
  const walk = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const el of root.querySelectorAll('*')) {
      out.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document);
  return out
    .filter((el) => el.children.length === 0)
    .map((el) => (el.textContent || '').trim())
    .filter((t) => t.length > 0)
    .join('\\n');
})()
"""


def gated_canvases(turn_id: str = "t1") -> list[AstraUiSurface]:
    """The fixture through the real gate. Exits non-zero if the gate rejects a canvas."""
    lines = FIXTURE.read_text().splitlines(keepends=True)
    boundaries = [1, 41, len(lines)]
    gate = UiGate()
    out: list[AstraUiSurface] = []
    for idx, end in enumerate(boundaries):
        result = gate.gate(
            "".join(lines[:end]),
            turn_id=turn_id,
            canvas_idx=idx,
            final=(idx == len(boundaries) - 1),
        )
        if result.surface is None:
            print(f"the gate produced no surface for canvas {idx}:", file=sys.stderr)
            print(f"  {result.report}", file=sys.stderr)
            raise SystemExit(2)
        out.append(result.surface)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", default="logs/hud/ui-quiz.png", help="where to write the PNG")
    parser.add_argument("--settle-s", type=float, default=1.0, help="pause between canvases")
    parser.add_argument("--size", default="1280,900", help="window size, WxH or W,H")
    args = parser.parse_args()

    if CHROME is None:
        print("no google-chrome/chromium on PATH", file=sys.stderr)
        return 2

    shot = Path(args.shot)
    if not shot.is_absolute():
        shot = REPO / shot
    shot.parent.mkdir(parents=True, exist_ok=True)

    canvases = gated_canvases()
    print(f"gate: {len(canvases)} canvases, surface_id={canvases[0].surface_id}")
    for surface in canvases:
        print(
            f"  canvas {surface.canvas_idx}: {len(surface.messages)} A2UI messages, "
            f"final={surface.final}, repaired={surface.repaired}"
        )

    stub = HudStub([]).start()
    server, port = serve_hud()
    try:
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with Chrome(CHROME) as chrome:
            chrome.navigate(url)
            if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
                print("the page never said astra.hello", file=sys.stderr)
                return 1
            for surface in canvases:
                stub.push(surface)
                time.sleep(args.settle_s)

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if LAST_PROMPT in str(chrome.evaluate(DEEP_TEXT_JS) or ""):
                    break
                time.sleep(0.15)
            else:
                print(f"the page never showed {LAST_PROMPT!r}", file=sys.stderr)
                print(str(chrome.evaluate(DEEP_TEXT_JS) or "")[:2000], file=sys.stderr)
                return 1

            time.sleep(0.5)  # one more frame for the catalog's styles to settle
            data = chrome._cmd("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            shot.write_bytes(base64.b64decode(data["data"]))
    finally:
        server.shutdown()
        stub.stop()

    print(f"{shot} ({shot.stat().st_size} bytes) — captured with CDP Page.captureScreenshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
