"""Generator of the tool-call eval set v1 (T-0080b criterion 1; Q4 synthetic screens). Deterministic, offline.

    uv run python -m tests.eval.toolcall.build            # render screens, write cases.jsonl + MANIFEST.json
    uv run python -m tests.eval.toolcall.build --check    # re-render into a temp dir and compare the bboxes

What it does:

1. writes every page of ``screens.py`` to ``screens/html/<id>.html`` (inline CSS only);
2. renders each with headless Chrome over the DevTools Protocol (loopback pipe-free websocket to a browser
   started here with ``--remote-debugging-port=0``), viewport 1280 x 800, device scale 1, network disabled
   (``Network.emulateNetworkConditions(offline=True)`` and every non-``file:`` request blocked), so a page can only
   load itself from ``file://``;
3. applies the screen's ``scroll_y``/``hscroll``, reads ``getBoundingClientRect()`` of every ``[data-t]``
   element, and captures the viewport as JPEG (quality 85) into ``screens/<id>.jpg``;
4. normalises each bbox to the 0-999 frame grid (below), resolves every case of ``cases_src.py`` and writes
   ``cases.jsonl`` and ``MANIFEST.json`` (sha256 of every file, the split's sha256, counts, the renderer).

**Normalisation.** A normalised coordinate ``n`` in 0..999 stands for the pixel ``n * W / 1000`` (``W`` the frame
width; ``H`` likewise). The inclusive bbox of an element whose box covers pixels ``[left, right)`` is
``x0 = ceil(left * 1000 / W)``, ``x1 = ceil(right * 1000 / W) - 1``, clipped to 0..999 -- exactly the ``n`` whose
pixel lies inside the element. An element that is not fully inside the viewport is an error, never clipped
silently.

The rendered JPEG bytes depend on this machine's Chrome and fonts; the committed images and the MANIFEST are
the set. ``--check`` re-renders and compares the bboxes (the geometry the scorer uses), not the bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.eval.toolcall import cases_src as src
from tests.eval.toolcall import screens as scr

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.jsonl"
MANIFEST = HERE / "MANIFEST.json"
SCREENS_DIR = HERE / "screens"
JPEG_QUALITY = 85
TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")
SPLITS = ("dev", "test")


# --------------------------------------------------------------------------------------- geometry


def normalise(rect: Mapping[str, float], w: int, h: int, *, clip: bool = False) -> list[int]:
    """The inclusive 0-999 bbox of a pixel rect ``{left, top, right, bottom}`` (module docstring).

    ``clip=True`` (scroll/press_key regions only, never a ground-truth bbox) first clips the rect to the viewport."""
    left, top, right, bottom = rect["left"], rect["top"], rect["right"], rect["bottom"]
    if clip:
        left, top, right, bottom = (
            max(0.0, left),
            max(0.0, top),
            min(float(w), right),
            min(float(h), bottom),
        )
    if left < 0 or top < 0 or right > w or bottom > h or right <= left or bottom <= top:
        raise ValueError(f"element not fully inside the {w}x{h} viewport: {dict(rect)}")
    x0 = math.ceil(left * 1000 / w)
    y0 = math.ceil(top * 1000 / h)
    x1 = min(999, math.ceil(right * 1000 / w) - 1)
    y1 = min(999, math.ceil(bottom * 1000 / h) - 1)
    if x1 < x0 or y1 < y0:
        raise ValueError(f"element too small for the 0-999 grid: {dict(rect)}")
    return [x0, y0, x1, y1]


def centre(bbox: Sequence[int]) -> tuple[int, int]:
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2


# ---------------------------------------------------------------------------------------- renderer


class Chrome:
    """Headless Chrome over the DevTools Protocol: navigate a file:// page, read the DOM, capture JPEG."""

    def __init__(self) -> None:
        binary = (
            shutil.which("google-chrome")
            or shutil.which("chromium")
            or shutil.which("chromium-browser")
        )
        if binary is None:
            raise SystemExit("no google-chrome/chromium on PATH")
        self.binary = binary
        self.profile = tempfile.mkdtemp(prefix="astra-toolcall-screens-")
        self.proc: subprocess.Popen[bytes] | None = None
        self.ws: Any = None
        self._id = 0

    def __enter__(self) -> Chrome:
        from websockets.sync.client import connect

        self.proc = subprocess.Popen(
            [
                self.binary,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--no-first-run",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--force-color-profile=srgb",
                "--font-render-hinting=none",
                "--window-size=1280,800",
                f"--user-data-dir={self.profile}",
                "--remote-debugging-address=127.0.0.1",
                "--remote-debugging-port=0",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_file = Path(self.profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        port = None
        while time.monotonic() < deadline:
            if port_file.exists() and port_file.read_text().splitlines():
                port = int(port_file.read_text().splitlines()[0])
                break
            time.sleep(0.05)
        if port is None:
            raise SystemExit("headless Chrome never wrote DevToolsActivePort")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
            targets = json.loads(r.read())
        page = next(t for t in targets if t["type"] == "page")
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024, open_timeout=15)
        self.cmd("Page.enable")
        self.cmd("Network.enable")
        self.cmd(
            "Network.emulateNetworkConditions",
            offline=True,
            latency=0,
            downloadThroughput=-1,
            uploadThroughput=-1,
        )
        self.cmd(
            "Network.setBlockedURLs", urls=["http://*", "https://*", "ws://*", "wss://*", "data:*"]
        )
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        finally:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)

    def cmd(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return dict(msg["result"])

    def evaluate(self, expression: str) -> Any:
        r = self.cmd(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if "exceptionDetails" in r:
            raise RuntimeError(f"page exception: {r['exceptionDetails']}")
        return r["result"].get("value")

    def render(self, html_path: Path, screen: scr.Screen) -> tuple[bytes, dict[str, list[int]]]:
        self.cmd(
            "Emulation.setDeviceMetricsOverride",
            width=scr.W,
            height=scr.H,
            deviceScaleFactor=1,
            mobile=False,
        )
        self.cmd("Page.navigate", url=html_path.resolve().as_uri())
        deadline = time.monotonic() + 15
        while self.evaluate("document.readyState") != "complete":
            if time.monotonic() > deadline:
                raise RuntimeError(f"{html_path} never finished loading")
            time.sleep(0.02)
        self.evaluate("document.fonts.ready.then(() => true)")
        hs = json.dumps(dict(screen.hscroll))
        vs = json.dumps(dict(screen.vscroll))
        self.evaluate(
            f"(() => {{ window.scrollTo(0, {int(screen.scroll_y)});"
            f' const h = {hs}; for (const k in h) {{ document.querySelector(`[data-t="${{k}}"]`).scrollLeft = h[k]; }}'
            f' const v = {vs}; for (const k in v) {{ document.querySelector(`[data-t="${{k}}"]`).scrollTop = v[k]; }}'
            " return true; })()"
        )
        for axis, pairs in (("scrollLeft", screen.hscroll), ("scrollTop", screen.vscroll)):
            for k, want in pairs:
                got = self.evaluate(f'document.querySelector(`[data-t="{k}"]`).{axis}')
                if got != want:
                    raise RuntimeError(f"{screen.id}: {k}.{axis} is {got}, not {want}")
        # two animation frames so layout and paint settle after the scroll
        self.evaluate(
            "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )
        if screen.scroll_y and self.evaluate("window.scrollY") != screen.scroll_y:
            raise RuntimeError(f"{screen.id}: page did not scroll to {screen.scroll_y}")
        rects = self.evaluate(
            "Object.fromEntries([...document.querySelectorAll('[data-t]')].map(e => {"
            " const r = e.getBoundingClientRect();"
            " return [e.dataset.t, {left: r.left, top: r.top, right: r.right, bottom: r.bottom}]; }))"
        )
        n = self.evaluate("document.querySelectorAll('[data-t]').length")
        if n != len(rects):
            raise RuntimeError(f"{screen.id}: duplicate data-t names")
        shot = self.cmd(
            "Page.captureScreenshot",
            format="jpeg",
            quality=JPEG_QUALITY,
        )
        import base64

        return base64.b64decode(shot["data"]), rects

    def version(self) -> str:
        return str(self.cmd("Browser.getVersion").get("product", "unknown"))


def render_all(out_dir: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Render every screen into ``out_dir``; return ``({screen id: {element: raw rect}}, browser version)``."""
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    rects: dict[str, dict[str, Any]] = {}
    with Chrome() as chrome:
        version = chrome.version()
        for screen in scr.SCREENS:
            page = html_dir / f"{screen.id}.html"
            page.write_text(scr.page(screen))
            jpeg, r = chrome.render(page, screen)
            (out_dir / f"{screen.id}.jpg").write_bytes(jpeg)
            rects[screen.id] = r
    return rects, version


# ------------------------------------------------------------------------------------------- cases


def _screen_ref(screen_id: str) -> str:
    return f"screens/{screen_id}.jpg"


def _target(
    bboxes: Mapping[str, Mapping[str, list[int]]], screen_id: str, element: str | None
) -> dict[str, Any] | None:
    if element is None:
        return None
    try:
        return {"element": element, "bbox": list(bboxes[screen_id][element])}
    except KeyError:
        raise ValueError(f"{screen_id} has no element data-t={element!r}") from None


def _check_split(case_id: str, split: str, screen_id: str | None) -> None:
    if screen_id is not None and scr.BY_ID[screen_id].split != split:
        raise ValueError(
            f"{case_id} ({split}) uses {screen_id}, a {scr.BY_ID[screen_id].split} screen"
        )


def build_cases(rects: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every case resolved against the rendered rects: click/type targets must lie wholly in the frame
    (strict); scroll/press_key regions are clipped to it (they are not ground truth)."""
    bboxes = _bboxes(rects, clip=False)
    regions = _bboxes(rects, clip=True)
    out: list[dict[str, Any]] = []
    size = [scr.W, scr.H]

    def base(
        cid: str, split: str, kind: str, tool: str | None, user: str, screen: str | None
    ) -> dict[str, Any]:
        _check_split(cid, split, screen)
        return {
            "id": cid,
            "split": split,
            "kind": kind,
            "tool": tool,
            "user": user,
            "screen": _screen_ref(screen) if screen else None,
            "screen_size": size if screen else None,
        }

    for cid, split, user, kws in src.SEARCH:
        out.append(
            {
                **base(cid, split, "tool", "google_search", user, None),
                "target": None,
                "gt": {"name": "google_search", "keywords": list(kws)},
                "reference_call": {"name": "google_search", "arguments": {"query": " ".join(kws)}},
            }
        )
    for cid, split, screen, el, user in src.CLICK:
        t = _target(bboxes, screen, el)
        assert t is not None
        x, y = centre(t["bbox"])
        out.append(
            {
                **base(cid, split, "tool", "click_at", user, screen),
                "target": t,
                "gt": {"name": "click_at", "bbox": t["bbox"]},
                "reference_call": {"name": "click_at", "arguments": {"x": x, "y": y}},
            }
        )
    for cid, split, screen, el, user, text, enter in src.TYPE:
        t = _target(bboxes, screen, el)
        assert t is not None
        x, y = centre(t["bbox"])
        out.append(
            {
                **base(cid, split, "tool", "type_at", user, screen),
                "target": t,
                "gt": {"name": "type_at", "bbox": t["bbox"], "text": text, "press_enter": enter},
                "reference_call": {
                    "name": "type_at",
                    "arguments": {"x": x, "y": y, "text": text, "press_enter": enter},
                },
            }
        )
    for cid, split, screen, el, user, direction in src.SCROLL:
        out.append(
            {
                **base(cid, split, "tool", "scroll", user, screen),
                "target": _target(regions, screen, el),
                "gt": {"name": "scroll", "direction": direction},
                "reference_call": {"name": "scroll", "arguments": {"direction": direction}},
            }
        )
    for cid, split, screen, el, user, combo in src.KEY:
        out.append(
            {
                **base(cid, split, "tool", "press_key", user, screen),
                "target": _target(regions, screen, el),
                "gt": {"name": "press_key", "combo": combo},
                "reference_call": {"name": "press_key", "arguments": {"combo": combo}},
            }
        )
    for cid, split, screen, user in src.CHAT:
        out.append(
            {
                **base(cid, split, "no_tool", None, user, screen),
                "target": None,
                "gt": None,
                "reference_call": None,
            }
        )
    return out


# ---------------------------------------------------------------------------------------- manifest


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cases_bytes(cases: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n").encode() for c in cases
    )


def split_canonical(cases: Sequence[Mapping[str, Any]]) -> bytes:
    """The split as canonical JSON: ``{"dev": [sorted ids], "test": [sorted ids]}``; its sha256 is pinned."""
    split = {s: sorted(c["id"] for c in cases if c["split"] == s) for s in SPLITS}
    return json.dumps(split, sort_keys=True, separators=(",", ":")).encode()


def counts(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for s in SPLITS:
        row = {t: sum(1 for c in cases if c["split"] == s and c["tool"] == t) for t in TOOLS}
        row["no_tool"] = sum(1 for c in cases if c["split"] == s and c["kind"] == "no_tool")
        out[s] = row
    return out


def manifest(
    cases: Sequence[Mapping[str, Any]], screens_dir: Path, renderer: str, cases_file: bytes
) -> dict[str, Any]:
    files = {}
    for p in sorted(screens_dir.rglob("*")):
        if p.is_file():
            files[p.relative_to(screens_dir.parent).as_posix()] = sha256_bytes(p.read_bytes())
    return {
        "set": "toolcall-v1",
        "task": "T-0080b",
        "cases_sha256": sha256_bytes(cases_file),
        "split_sha256": sha256_bytes(split_canonical(cases)),
        "counts": counts(cases),
        "files": files,
        "generator": {
            p: sha256_bytes((HERE / p).read_bytes())
            for p in ("build.py", "screens.py", "cases_src.py")
        },
        "renderer": renderer,
        "viewport": [scr.W, scr.H],
        "jpeg_quality": JPEG_QUALITY,
        "normalisation": "n stands for pixel n*W/1000; bbox = [ceil(l*1000/W), ceil(t*1000/H), ceil(r*1000/W)-1, ceil(b*1000/H)-1]",
    }


def _bboxes(
    rects: Mapping[str, Mapping[str, Any]], *, clip: bool
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for sid, r in rects.items():
        out[sid] = {}
        for el, rect in r.items():
            try:
                out[sid][el] = normalise(rect, scr.W, scr.H, clip=clip)
            except ValueError:
                continue  # an off-screen element is an error only if a case targets it (_target)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--check", action="store_true", help="re-render into a temp dir; compare bboxes"
    )
    args = ap.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="toolcall-check-") as tmp:
            rects, _ = render_all(Path(tmp))
            fresh = build_cases(rects)
        committed = [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]
        diff = [
            (a["id"], a.get("target"), b.get("target"))
            for a, b in zip(committed, fresh, strict=True)
            if a.get("target") != b.get("target")
        ]
        for d in diff:
            print("bbox differs:", d)
        print("check:", "OK" if not diff else f"{len(diff)} differ")
        return 0 if not diff else 1
    if SCREENS_DIR.exists():
        shutil.rmtree(SCREENS_DIR)
    SCREENS_DIR.mkdir()
    rects, version = render_all(SCREENS_DIR)
    cases = build_cases(rects)
    data = cases_bytes(cases)
    CASES.write_bytes(data)
    MANIFEST.write_text(
        json.dumps(manifest(cases, SCREENS_DIR, version, data), indent=2, sort_keys=True) + "\n"
    )
    print(f"{len(cases)} cases, {len(scr.SCREENS)} screens, renderer {version}")
    print(json.dumps(counts(cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
