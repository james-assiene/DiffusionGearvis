"""T-0111c: the browser executor against a real Chromium and loopback fixture pages (criteria 2-11, 13-15).

**Gate.** The module is ``integration`` + ``browser`` and runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser``);
without it every row is a named skip. With it, a missing ``playwright`` or a missing Chromium is a *failure* naming
P1 of docs/tasks/T-0111c.md, never a skip (precedent: ``tests/integration/test_hud_headless.py``
``test_chrome_and_the_page_are_present_for_the_integration_layer``). Until the implementer registers the ``browser``
marker in ``pyproject.toml`` the module carries ``integration`` only, so ``--strict-markers`` cannot turn the whole
module into a collection error; ``tests/unit/test_computer_imports.py::test_browser_marker_registered`` pins the
registration.

**Fixtures.** ``tests/fixtures/browser/`` served by ``http.server`` on 127.0.0.1 (MANIFEST.md lists the pages). Each
page loads ``recorder.js``, which appends every event it receives to ``window.__log``. Expected values are computed
by hand from the fixture geometry (see each page's CSS) or come from the probes ``probe_press_key.py`` and
``probe_sandbox.py`` (plain Playwright, no executor; output in MANIFEST.md).

**The API these rows pin** (the task fixes behaviour, not names; these are the hooks a row needs, and the
implementer must provide them; ``tests/unit/test_computer_executor.py`` pins the driver seam):

=========================================  ============================================================
``BrowserToolExecutor(*, now_ms, viewport=(1280, 800), guard=None, user_data_dir=None, driver=None)``
                                           keyword-only; ``driver=None`` means the Playwright runtime
``ex.mode``                                ``"live"``
``await ex.execute(inv)``                  a ``ToolOutcome`` (criterion 1)
``await ex.frame()``                       an own ``Frame`` of the active page (launches if needed)
``ex.latest_frame``                        the newest own ``Frame`` (after an executed action: the
                                           post-action frame whose phash is ``resolved["frame_after"]``)
``await ex.open(url)``                     the harness's navigation of the active page (loopback only;
                                           c3 brings the ``open_url`` tool)
``ex.egress_log``                          the blocked off-loopback attempts, each ``{"host", "scheme"}``
``ex.driver.page``                         the active Playwright ``Page`` (the rows read ``__log`` from it)
``await ex.close()``                       closes the browser
=========================================  ============================================================

**Regime.** Playwright 1.64.0, Chromium 156.0.8078.4 headless (versions.lock ``[browser]``), 1280x800 at DSF 1 unless a
row says otherwise, n = 1. One executor, and so one browser, per row: the kill/restart row and the frame-generation
rule make a browser shared between rows unsafe (a deviation from the task's "one browser per module", reported).

**Network.** Everything is loopback. The egress row first proves the fence against a canary on this box (the
hostname, which /etc/hosts maps to 127.0.1.1, outside the fence's allowlist) and only then makes the attempts the
criterion names against 198.51.100.7 (TEST-NET-2) and example.invalid; against an *unfenced* executor those would
leave the box, so the canary phase stops the row first.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import importlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.parse
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.tool_executor import DryRunToolExecutor, to_pixels
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocols import ToolExecutor
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "browser"
GATE_ENV = "ASTRA_BROWSER_TESTS"
GATED = os.environ.get(GATE_ENV) == "1"
P1 = (
    "P1 of docs/tasks/T-0111c.md: `uv sync --all-extras` (the optional extra 'browser', playwright==1.64.0) and "
    "`uv run playwright install chromium` (versions.lock [browser])"
)


def _browser_marker_registered() -> bool:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    markers = data["tool"]["pytest"]["ini_options"].get("markers", [])
    return any(m.split(":", 1)[0].strip() == "browser" for m in markers)


pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

W, H = 1280, 800
CLOCK_MS = 1_700_000_004_242
#: Every input event the recorder logs that an action dispatches. ``pointermove`` and ``scroll`` are left out of the
#: "dispatched nothing" comparisons: Chromium synthesises both after layout changes (probe_press_key.py output).
INPUT_EVENTS = frozenset(
    {
        "pointerdown",
        "pointerup",
        "click",
        "dblclick",
        "contextmenu",
        "auxclick",
        "keydown",
        "keyup",
        "input",
        "submit",
        "copy",
        "cut",
        "paste",
    }
)
MODIFIER_KEYS = frozenset({"Control", "Alt", "Shift", "Meta"})


# --------------------------------------------------------------------------------------------- the gate


@pytest.fixture(autouse=True)
def _browser_layer() -> None:
    if not GATED:
        pytest.skip(
            f"{GATE_ENV} is not 1: the T-0111c browser e2e layer drives a real Chromium; run `make browser` "
            "(docs/tasks/T-0111c.md, Regime)"
        )
    try:
        importlib.import_module("playwright.async_api")
    except ImportError as exc:
        pytest.fail(f"{GATE_ENV}=1 but playwright does not import ({exc}); {P1}")
    if not _bundled_chromium_dirs():
        pytest.fail(f"{GATE_ENV}=1 but no Playwright Chromium under {_browsers_path()}; {P1}")


def _browsers_path() -> Path:
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    return Path(env) if env and env != "0" else Path.home() / ".cache" / "ms-playwright"


def _bundled_chromium_dirs() -> list[Path]:
    root = _browsers_path()
    return sorted(p for p in root.glob("chromium*-*") if p.is_dir()) if root.is_dir() else []


def _lock() -> dict[str, str]:
    data = tomllib.loads((REPO / "services" / "versions.lock").read_text())
    return {k: str(v) for k, v in data["browser"].items()}


def test_the_browser_layer_is_installed() -> None:
    """P1/P2 present: the pinned Playwright imports and its Chromium is in Playwright's cache (green at BASE)."""
    from importlib.metadata import version

    assert version("playwright") == _lock()["playwright"], P1
    assert _bundled_chromium_dirs(), P1


# --------------------------------------------------------------------------------------------- helpers


def computer() -> Any:
    """``local_astra.computer``, or a named AssertionError while it does not exist (never an ImportError)."""
    try:
        return importlib.import_module("local_astra.computer")
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "local_astra.computer" or exc.name.startswith("local_astra.computer.")
        ):
            raise AssertionError(f"T-0111c: {exc.name} does not exist yet") from exc
        if exc.name == "playwright" or (exc.name or "").startswith("playwright."):
            raise AssertionError(f"local_astra.computer needs playwright: {P1}") from exc
        raise


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        # ``/t0111c-redirect?to=<url>``: a loopback 302 to any target (criterion 14's redirect vector).
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/t0111c-redirect":
            self.send_response(302)
            self.send_header("Location", urllib.parse.parse_qs(u.query)["to"][0])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    handler = functools.partial(_Quiet, directory=str(FIXTURES))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0111c-fixture-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


@contextlib.asynccontextmanager
async def executor(**kw: Any) -> AsyncIterator[Any]:
    cls = computer().BrowserToolExecutor
    kw.setdefault("now_ms", lambda: CLOCK_MS)
    ex = cls(**kw)
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)


def geom(f: Frame) -> FrameGeometry:
    return FrameGeometry(phash=f.phash, source=f.source, width=f.width, height=f.height)


_N = iter(range(1_000_000))


def inv(name: str, frame: FrameGeometry | Frame | None = None, **arguments: Any) -> ToolInvocation:
    g = geom(frame) if isinstance(frame, Frame) else frame
    return ToolInvocation(
        call_id=f"c{next(_N)}",
        turn_id="t1",
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


def coord(px: int, size: int) -> int:
    """The 0-999 coordinate whose pixel is ``px`` (an inverse for *targeting* only; checked against to_pixels)."""
    c = round(px * 999 / (size - 1))
    assert to_pixels(c, size) == px, (px, size, c)
    return c


async def run(aw: Awaitable[Any], timeout: float = 30.0) -> Any:
    return await asyncio.wait_for(aw, timeout)


async def execute(ex: Any, i: ToolInvocation, timeout: float = 30.0) -> ToolOutcome:
    out = await run(ex.execute(i), timeout)
    assert isinstance(out, ToolOutcome), type(out)
    assert out.call_id == i.call_id
    return out


async def page_log(ex: Any) -> list[dict[str, Any]]:
    return list(await ex.driver.page.evaluate("window.__log"))


async def clear_log(ex: Any) -> None:
    await ex.driver.page.evaluate("window.__log.length = 0")


async def state(ex: Any) -> dict[str, Any]:
    return dict(await ex.driver.page.evaluate("window.__state()"))


def of_type(log: list[dict[str, Any]], *types: str) -> list[dict[str, Any]]:
    return [e for e in log if e.get("type") in types]


def inputs(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in log if e.get("type") in INPUT_EVENTS]


async def poll(
    fn: Callable[[], Awaitable[Any]], pred: Callable[[Any], bool], timeout: float = 2.0
) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        v = await fn()
        if pred(v) or time.monotonic() > deadline:
            return v
        await asyncio.sleep(0.05)


async def settled_scroll(ex: Any) -> tuple[int, int]:
    """(scrollX, scrollY) once two reads 150 ms apart agree (bounded at 3 s): "once settled" in criterion 7."""
    deadline = time.monotonic() + 3.0
    last = None
    while True:
        s = await state(ex)
        cur = (int(s["scrollX"]), int(s["scrollY"]))
        if cur == last or time.monotonic() > deadline:
            return cur
        last = cur
        await asyncio.sleep(0.15)


def rgb_at(f: Frame, x: int, y: int) -> tuple[int, int, int]:
    from PIL import Image

    img = Image.open(io.BytesIO(f.jpeg)).convert("RGB")
    r, g, b = img.getpixel((x, y))
    return int(r), int(g), int(b)


def near(c: tuple[int, int, int], want: tuple[int, int, int], tol: int = 40) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(c, want, strict=True))


def mean_abs_diff(a: Frame, b: Frame) -> float:
    import numpy as np
    from PIL import Image

    x = np.asarray(Image.open(io.BytesIO(a.jpeg)).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(io.BytesIO(b.jpeg)).convert("RGB"), dtype=np.float64)
    assert x.shape == y.shape, (x.shape, y.shape)
    return float(np.abs(x - y).mean())


def _exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def chromium_mains() -> list[tuple[int, str, str]]:
    """``(pid, exe, cmdline)`` of every main browser process descended from this pytest process: an executable
    under Playwright's cache (or any chrome/chromium) whose parent's executable is not. Chromium rewrites its argv,
    so the cmdline is joined with spaces."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        children.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(d))

    def browserish(exe: str) -> bool:
        base = os.path.basename(exe)
        return "ms-playwright" in exe or "chrom" in base or base == "google-chrome"

    out, todo = [], [os.getpid()]
    while todo:
        p = todo.pop()
        for c in children.get(p, []):
            todo.append(c)
            if browserish(_exe(c)) and not browserish(_exe(p)):
                try:
                    cmd = Path(f"/proc/{c}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    cmd = ""
                out.append((c, _exe(c), cmd))
    return out


class Canary:
    """A TCP listener on ``ip``: counts the connections it accepted (any = an egress the fence let through)."""

    def __init__(self, ip: str) -> None:
        self.sock = socket.socket()
        self.sock.bind((ip, 0))
        self.sock.listen(64)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.count = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="t0111c-canary", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.count += 1
            with contextlib.suppress(OSError):
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            c.close()

    def close(self) -> None:
        self._stop.set()
        self._t.join(2)
        self.sock.close()


#: The canaries of the redirect and WebRTC rows: an IP literal on loopback outside the fence's allowlist. Linux
#: routes all of 127/8 to ``lo``, so binding it needs no /etc/hosts entry and nothing leaves the box.
CANARY_IP = "127.0.1.1"


class UdpCanary:
    """A UDP socket on ``ip``: counts the datagrams it received (any = a STUN/TURN packet the fence let out)."""

    def __init__(self, ip: str) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, 0))
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.count = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="t0111c-udp-canary", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            self.count += 1

    def close(self) -> None:
        self._stop.set()
        self._t.join(2)
        self.sock.close()


def descendant_pids() -> set[int]:
    """Every process descended from this pytest process (the Playwright driver, the browser and its children)."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        children.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(d))
    out: set[int] = set()
    todo = [os.getpid()]
    while todo:
        for c in children.get(todo.pop(), []):
            out.add(c)
            todo.append(c)
    return out


def browser_udp_sockets() -> list[str]:
    """``ss -uanp`` lines owned by a process this row started (another browser on the desk is not counted)."""
    proc = subprocess.run(["ss", "-uanp"], capture_output=True, text=True, timeout=10, check=False)
    assert proc.returncode == 0, f"ss -uanp failed: {proc.stderr}"
    pids = descendant_pids()
    return [
        line.split()[0] + " " + line.split()[3]
        for line in proc.stdout.splitlines()
        if any(int(p) in pids for p in re.findall(r"pid=(\d+),", line))
    ]


def canary_host() -> tuple[str, str] | None:
    """``(hostname, ip)`` when this box's hostname resolves to a loopback address other than 127.0.0.1."""
    name = socket.gethostname()
    try:
        ip = socket.gethostbyname(name)
    except OSError:
        return None
    if not ip.startswith("127.") or ip == "127.0.0.1" or name.lower() == "localhost":
        return None
    return name, ip


EXC_NAME = re.compile(r"\b[A-Z][A-Za-z0-9_]*(Error|Exception)\b")


def names_an_exception(reason: str | None) -> bool:
    return bool(reason) and EXC_NAME.search(reason or "") is not None


# --------------------------------------------------------------------------------------------- criterion 1, 2


async def test_frame_shape_and_determinism(server: str) -> None:
    """Criterion 2. ``source`` WINDOW; size = decoded size = the 1280x800 viewport (``screen_max`` is 1568x1176,
    so ``target_size`` leaves it unscaled: scale = min(1, 1568/1280, 1176/800) = 1); the JPEG decodes; phash is 16
    lowercase hex, equal for identical pixels and different for a different page; ``ts_ms`` is the injected clock's."""
    from PIL import Image

    async with executor() as ex:
        assert isinstance(ex, ToolExecutor) and ex.mode == "live"
        await run(ex.open(f"{server}/color.html"))
        f1 = await run(ex.frame())
        f2 = await run(ex.frame())
        assert isinstance(f1, Frame), type(f1)
        assert f1.source == Source.WINDOW
        img = Image.open(io.BytesIO(f1.jpeg))
        assert img.format == "JPEG"
        img.load()
        assert img.size == (f1.width, f1.height) == (W, H)
        assert re.fullmatch(r"[0-9a-f]{16}", f1.phash), f1.phash
        assert f1.phash == f2.phash, "identical pixels, different phash"
        assert f1.ts_ms == CLOCK_MS, "ts_ms comes from the injected clock"
        await run(ex.open(f"{server}/color.html"))
        f3 = await run(ex.frame())
        assert f3.phash == f1.phash, "the same page re-opened is the same pixels"
        await run(ex.open(f"{server}/scroll.html"))
        f4 = await run(ex.frame())
        assert f4.phash != f1.phash, "a constant phash would make every frame 'own'"


# --------------------------------------------------------------------------------------------- criterion 3


async def test_mapping_corners_and_centre(server: str) -> None:
    """Criterion 3: (0,0) -> (0,0), (999,999) -> (1279,799), (500,500) -> (640,400), by hand:
    (2*500*1279 + 999) // 1998 = 640 and (2*500*799 + 999) // 1998 = 400."""
    async with executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        for (x, y), want in (((0, 0), (0, 0)), ((999, 999), (1279, 799)), ((500, 500), (640, 400))):
            f = await run(ex.frame())
            await clear_log(ex)
            out = await execute(ex, inv("click_at", f, x=x, y=y))
            assert out.status == "executed", out
            assert (out.resolved["x_px"], out.resolved["y_px"]) == want, out.resolved
            downs = of_type(await page_log(ex), "pointerdown")
            assert [(d["x"], d["y"]) for d in downs] == [want], downs
            assert downs[0]["target"] == "grid"


async def test_mapping_uses_viewport_not_image_when_downscaled(server: str) -> None:
    """Criterion 3 adversarial (reviewer mutation 1). At 1600x1000 the frame is downscaled by ``screen_max``:
    scale = min(1568/1600, 1176/1000) = 0.98 -> 1568x980. (999,999) must reach (1599,999) and (500,500) must reach
    (800,500); mapping against the image would give (1567,979) and (784,490)."""
    async with executor(viewport=(1600, 1000)) as ex:
        await run(ex.open(f"{server}/grid.html"))
        f = await run(ex.frame())
        assert (f.width, f.height) == (1568, 980), "the 1600x1000 frame is downscaled to screen_max"
        for (x, y), want in (((999, 999), (1599, 999)), ((500, 500), (800, 500))):
            f = await run(ex.frame())
            await clear_log(ex)
            out = await execute(ex, inv("click_at", f, x=x, y=y))
            assert out.status == "executed", out
            assert (out.resolved["x_px"], out.resolved["y_px"]) == want, out.resolved
            downs = of_type(await page_log(ex), "pointerdown")
            assert [(d["x"], d["y"]) for d in downs] == [want], downs


# --------------------------------------------------------------------------------------------- criterion 4


async def test_foreign_frame_dispatches_nothing(server: str) -> None:
    """Criterion 4 adversarial (reviewer mutation 2): a HUD-shaped ``FrameGeometry(source=SCREEN)`` of the very
    size of the viewport, with an invented phash, is refused and dispatches nothing; so are the other refusals."""
    async with executor() as ex:
        await run(ex.open(f"{server}/form.html"))
        own = await run(ex.frame())
        invented = "0123456789abcdef" if own.phash != "0123456789abcdef" else "fedcba9876543210"
        hud = FrameGeometry(phash=invented, source=Source.SCREEN, width=W, height=H)
        window_invented = FrameGeometry(phash=invented, source=Source.WINDOW, width=W, height=H)
        cases: list[tuple[ToolInvocation, str]] = [
            (inv("click_at", None, x=500, y=500), "no frame"),
            (inv("type_at", None, x=234, y=150, text="hi", press_enter=True), "no frame"),
            (inv("click_at", hud, x=500, y=500), "frame not from this browser"),
            (inv("click_at", window_invented, x=500, y=500), "frame not from this browser"),
            (
                inv("type_at", hud, x=234, y=150, text="hi", press_enter=True),
                "frame not from this browser",
            ),
            (inv("click_at", own, x=1000, y=500), "bad coordinates"),
            (inv("click_at", own, x=-1, y=500), "bad coordinates"),
            (inv("click_at", own, x=True, y=500), "bad coordinates"),
            (inv("click_at", own, x="500", y=500), "bad coordinates"),
            (inv("click_at", own, x=500), "bad coordinates"),
            (inv("type_at", own, x=234, y=150, text=5, press_enter=True), "bad arguments"),
            (inv("type_at", own, x=234, y=150, text="hi", press_enter="yes"), "bad arguments"),
        ]
        before = await state(ex)
        await clear_log(ex)
        for i, reason in cases:
            out = await execute(ex, i)
            assert (out.status, out.reason) == ("refused", reason), (i.arguments, i.frame, out)
        await asyncio.sleep(0.2)
        assert inputs(await page_log(ex)) == [], "a refused call dispatched input"
        assert (await state(ex))["values"] == before["values"]


# --------------------------------------------------------------------------------------------- criterion 5


async def test_click_is_one_primary_click(server: str) -> None:
    """Criterion 5: exactly one primary pointerdown/pointerup/click at the target; no dblclick, contextmenu or
    auxclick."""
    async with executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        f = await run(ex.frame())
        await clear_log(ex)
        out = await execute(ex, inv("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        await asyncio.sleep(0.3)  # a late dblclick would arrive here
        log = await page_log(ex)
        for t in ("pointerdown", "pointerup", "click"):
            evs = of_type(log, t)
            assert len(evs) == 1, (t, evs)
            assert (evs[0]["x"], evs[0]["y"], evs[0]["button"], evs[0]["target"]) == (
                640,
                400,
                0,
                "grid",
            ), evs
        assert of_type(log, "pointerdown")[0]["buttons"] == 1, "primary button held at pointerdown"
        assert of_type(log, "pointerup")[0]["buttons"] == 0
        assert of_type(log, "dblclick", "contextmenu", "auxclick") == []


# --------------------------------------------------------------------------------------------- criterion 6


async def test_type_enter_and_no_enter(server: str) -> None:
    """Criterion 6. ``form.html`` focuses ``#other`` on load, so a ``type_at`` that relied on focus instead of
    clicking types there (cua #8). (234,150) -> (300,120), inside ``#q`` (x 100..499, y 100..139). ``#q`` starts as
    ``pre-``; it is a single-line ``<input>`` with no ``type``, so ``type_at`` replaces its text (T-0189 criterion 1,
    superseding H-T111c-6 for single-line inputs): the value is exactly the typed text."""
    async with executor() as ex:
        for text, enter in (("hello", True), ("world", False)):
            await run(ex.open(f"{server}/form.html"))
            f = await run(ex.frame())
            await clear_log(ex)
            out = await execute(ex, inv("type_at", f, x=234, y=150, text=text, press_enter=enter))
            assert out.status == "executed", out
            assert (out.resolved["x_px"], out.resolved["y_px"]) == (300, 120), out.resolved
            await asyncio.sleep(0.2)
            log = await page_log(ex)
            values = (await state(ex))["values"]
            assert values["q"] == text, f"T-0189 criterion 1: #q must hold only {text!r}: {values}"
            assert values["other"] == "", "typed into the element that had focus, not the target"
            downs = [i for i, e in enumerate(log) if e["type"] == "pointerdown"]
            first_key = next(i for i, e in enumerate(log) if e["type"] in ("keydown", "input"))
            assert downs and downs[0] < first_key, "type_at clicks the target before typing"
            assert (log[downs[0]]["x"], log[downs[0]]["y"], log[downs[0]]["target"]) == (
                300,
                120,
                "q",
            )
            enters = [e for e in of_type(log, "keydown") if e["key"] == "Enter"]
            submits = of_type(log, "submit")
            if enter:
                assert len(submits) == 1 and len(enters) == 1, (submits, enters)
            else:
                assert submits == [] and enters == [], (submits, enters)


async def test_newline_in_text_types_nothing(server: str) -> None:
    """Criterion 6 adversarial (reviewer mutation 5; cua ``carl\\enter``): text with a control character is refused
    as ``bad arguments: control character`` and nothing is clicked or typed (``keyboard.type`` would press Enter)."""
    async with executor() as ex:
        await run(ex.open(f"{server}/form.html"))
        f = await run(ex.frame())
        before = await state(ex)
        await clear_log(ex)
        for text in (
            "carl\nsmith",
            "carl\r",
            "a\r\nb",
            "tab\there",
            "nul\x00",
            "del\x7f",
            "esc\x1b[A",
            "\x1f",
        ):
            for enter in (False, True):
                out = await execute(
                    ex, inv("type_at", f, x=234, y=150, text=text, press_enter=enter)
                )
                assert (out.status, out.reason) == (
                    "refused",
                    "bad arguments: control character",
                ), (text, out)
        await asyncio.sleep(0.2)
        assert inputs(await page_log(ex)) == []
        assert (await state(ex))["values"] == before["values"]


async def test_non_ascii_text(server: str) -> None:
    """Criterion 6: ``café 東京`` arrives exactly; ``#q`` (prefilled ``pre-``) holds only it (T-0189 criterion 1)."""
    async with executor() as ex:
        await run(ex.open(f"{server}/form.html"))
        f = await run(ex.frame())
        out = await execute(
            ex, inv("type_at", f, x=234, y=150, text="café 東京", press_enter=False)
        )
        assert out.status == "executed", out
        await asyncio.sleep(0.2)
        got = (await state(ex))["values"]["q"]
        assert got == "café 東京", f"T-0189 criterion 1: #q must hold only the typed text: {got!r}"


# --------------------------------------------------------------------------------------------- criterion 7


async def test_scroll_moves_one_screen(server: str) -> None:
    """Criterion 7: on a page three viewports tall, ``down`` from the top moves scrollY into [400, 800] once
    settled; ``up`` reduces it; ``right``/``left`` likewise on a page three viewports wide ([640, 1280]). The
    post-action frame differs from the pre-action frame in pixels (not by phash, cua §0.2)."""
    async with executor() as ex:
        await run(ex.open(f"{server}/scroll.html"))
        pre = await run(ex.frame())
        out = await execute(ex, inv("scroll", direction="down"))
        assert out.status == "executed", out
        _, y1 = await settled_scroll(ex)
        assert H // 2 <= y1 <= H, y1
        post = ex.latest_frame
        assert post is not None and post.phash == out.resolved["frame_after"]
        assert mean_abs_diff(pre, post) > 5.0, "the post-action frame shows the same pixels"
        out = await execute(ex, inv("scroll", direction="up"))
        assert out.status == "executed", out
        _, y2 = await settled_scroll(ex)
        assert y2 < y1, (y1, y2)

        await run(ex.open(f"{server}/scroll_wide.html"))
        pre = await run(ex.frame())
        out = await execute(ex, inv("scroll", direction="right"))
        assert out.status == "executed", out
        x1, _ = await settled_scroll(ex)
        assert W // 2 <= x1 <= W, x1
        post = ex.latest_frame
        assert post is not None and mean_abs_diff(pre, post) > 5.0
        out = await execute(ex, inv("scroll", direction="left"))
        assert out.status == "executed", out
        x2, _ = await settled_scroll(ex)
        assert x2 < x1, (x1, x2)


# --------------------------------------------------------------------------------------------- criterion 8
#
# Established with tests/fixtures/browser/probe_press_key.py on Chromium 156.0.8078.4 headless (MANIFEST.md). Each
# executed combo has: a setup on keys.html done by the row through Playwright, the keydown it must produce
# (key, code, ctrlKey, altKey) and a check of the effect. The four browser-chrome combos are refused (H-T111c-7);
# in the probe none of them had any effect in the page either (Ctrl+T opened no page, Ctrl+W closed none).

TA = "line one\nline two\nline three"  # keys.html #ta: 28 characters, caret 12 is in "line two"
REFUSED_COMBOS = frozenset({"Alt+Tab", "Ctrl+T", "Ctrl+W", "Ctrl+L"})


async def _inp_abc(page: Any, token: str) -> None:
    await page.click("#inp")
    await page.keyboard.type("abc")
    await page.keyboard.press("ArrowLeft")


async def _inp_focus(page: Any, token: str) -> None:
    await page.evaluate("document.getElementById('inp').focus()")


async def _ta_caret(page: Any, token: str) -> None:
    await page.evaluate(
        "const t = document.getElementById('ta'); t.focus(); t.setSelectionRange(12, 12)"
    )


async def _body_scrolled(page: Any, token: str) -> None:
    await page.evaluate("window.scrollTo(0, 1000); document.activeElement.blur()")
    await asyncio.sleep(0.2)


async def _dialog(page: Any, token: str) -> None:
    await page.evaluate("document.getElementById('dlg').showModal()")


async def _token_selected(page: Any, token: str) -> None:
    await page.evaluate(
        "tok => { const t = document.getElementById('ta'); t.value = tok + ' rest'; t.focus();"
        " t.setSelectionRange(0, tok.length); }",
        token,
    )


async def _token_on_clipboard(page: Any, token: str) -> None:
    await _token_selected(page, token)
    await page.keyboard.press("Control+c")
    await page.evaluate("document.getElementById('b').focus()")


def _sel(s: dict[str, Any]) -> tuple[Any, Any]:
    return s["selStart"], s["selEnd"]


#: combo -> (setup, (key, code, ctrlKey, altKey), check(state, log, token) -> bool, what the check means)
EXECUTED: dict[str, tuple[Any, tuple[Any, str, bool, bool], Callable[..., bool], str]] = {
    "Enter": (_inp_abc, ("Enter", "Enter", False, False), lambda s, log, t: len(of_type(log, "submit")) == 1, "submits the form once"),
    "Escape": (_dialog, ("Escape", "Escape", False, False), lambda s, log, t: s["dialogOpen"] is False, "closes the modal dialog"),
    "Tab": (_inp_focus, ("Tab", "Tab", False, False), lambda s, log, t: s["active"] == "ta", "moves focus #inp -> #ta"),
    "Backspace": (_inp_abc, ("Backspace", "Backspace", False, False), lambda s, log, t: s["values"]["inp"] == "ac", "abc -> ac"),
    "Delete": (_inp_abc, ("Delete", "Delete", False, False), lambda s, log, t: s["values"]["inp"] == "ab", "abc -> ab"),
    "Space": (_inp_abc, (" ", "Space", False, False), lambda s, log, t: s["values"]["inp"] == "ab c", "abc -> ab c"),
    "ArrowUp": (_ta_caret, ("ArrowUp", "ArrowUp", False, False), lambda s, log, t: _sel(s) == (3, 3), "caret 12 -> 3"),
    "ArrowDown": (_ta_caret, ("ArrowDown", "ArrowDown", False, False), lambda s, log, t: _sel(s) == (21, 21), "caret 12 -> 21"),
    "ArrowLeft": (_ta_caret, ("ArrowLeft", "ArrowLeft", False, False), lambda s, log, t: _sel(s) == (11, 11), "caret 12 -> 11"),
    "ArrowRight": (_ta_caret, ("ArrowRight", "ArrowRight", False, False), lambda s, log, t: _sel(s) == (13, 13), "caret 12 -> 13"),
    "Home": (_ta_caret, ("Home", "Home", False, False), lambda s, log, t: _sel(s) == (9, 9), "caret 12 -> 9 (line start)"),
    "End": (_ta_caret, ("End", "End", False, False), lambda s, log, t: _sel(s) == (17, 17), "caret 12 -> 17 (line end)"),
    "PageUp": (_body_scrolled, ("PageUp", "PageUp", False, False), lambda s, log, t: s["scrollY"] < 1000, "scrollY below 1000"),
    "PageDown": (_body_scrolled, ("PageDown", "PageDown", False, False), lambda s, log, t: s["scrollY"] > 1000, "scrollY above 1000"),
    "Ctrl+A": (_ta_caret, ({"a", "A"}, "KeyA", True, False), lambda s, log, t: _sel(s) == (0, len(TA)), "selects all 28 characters"),
    "Ctrl+C": (_token_selected, ({"c", "C"}, "KeyC", True, False), lambda s, log, t: len(of_type(log, "copy")) == 1, "fires copy (the paste-back check follows)"),
    "Ctrl+V": (_token_on_clipboard, ({"v", "V"}, "KeyV", True, False), lambda s, log, t: s["values"]["b"] == t, "pastes the token into #b"),
    "Ctrl+X": (_token_selected, ({"x", "X"}, "KeyX", True, False), lambda s, log, t: s["values"]["ta"] == " rest", "cuts the token"),
    "Ctrl+Z": (_inp_abc, ({"z", "Z"}, "KeyZ", True, False), lambda s, log, t: s["values"]["inp"] == "", "undoes the typing"),
    "Ctrl+Enter": (_inp_focus, ("Enter", "Enter", True, False), lambda s, log, t: len(of_type(log, "ctrl-enter-handled")) == 1, "the page's Ctrl+Enter binding fires"),
}  # fmt: skip


def _combo_enum() -> list[str]:
    tools = json.loads((REPO / "local_astra" / "toolcall" / "tools_v1.json").read_text())
    (press,) = [t for t in tools if t["name"] == "press_key"]
    return list(press["parameters"]["properties"]["combo"]["enum"])


async def test_every_press_key_combo_effect_or_refusal(server: str) -> None:
    """Criterion 8 (reviewer mutation 9): every enum value is executed with its keydown and its established effect,
    or refused as ``not available in the browser`` dispatching nothing. The table covers the enum exactly."""
    combos = _combo_enum()
    assert set(combos) == set(EXECUTED) | REFUSED_COMBOS and not set(EXECUTED) & REFUSED_COMBOS, (
        combos
    )
    problems: list[str] = []
    async with executor() as ex:
        for n, combo in enumerate(combos):
            token = f"t0111c-token-{n}-{os.getpid()}"
            await run(ex.open(f"{server}/keys.html"))
            page = ex.driver.page
            pages_before = len(page.context.pages)
            if combo in REFUSED_COMBOS:
                await _inp_focus(page, token)
                before = await state(ex)
                await clear_log(ex)
                out = await execute(ex, inv("press_key", combo=combo))
                await asyncio.sleep(0.2)
                log = await page_log(ex)
                if (out.status, out.reason) != ("refused", "not available in the browser"):
                    problems.append(
                        f"{combo}: expected refused/'not available in the browser', got {out.status}/{out.reason}"
                    )
                if inputs(log):
                    problems.append(f"{combo}: refused but dispatched {inputs(log)}")
                if (
                    len(ex.driver.page.context.pages) != pages_before
                    or (await state(ex))["values"] != before["values"]
                ):
                    problems.append(f"{combo}: refused but the page changed")
                continue
            setup, (key, code, ctrl, alt), check, meaning = EXECUTED[combo]
            await setup(page, token)
            await clear_log(ex)
            out = await execute(ex, inv("press_key", combo=combo))
            if out.status != "executed":
                problems.append(f"{combo}: expected executed, got {out.status}/{out.reason}")
                continue

            async def snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
                return await state(ex), await page_log(ex)

            s, log = await poll(snapshot, lambda v: check(v[0], v[1], token), timeout=2.0)  # noqa: B023
            keys = [e for e in of_type(log, "keydown") if e["key"] not in MODIFIER_KEYS]
            if len(keys) != 1:
                problems.append(f"{combo}: expected one non-modifier keydown, got {keys}")
            else:
                k = keys[0]
                key_ok = k["key"] in key if isinstance(key, set) else k["key"] == key
                if not (
                    key_ok and k["code"] == code and k["ctrlKey"] is ctrl and k["altKey"] is alt
                ):
                    problems.append(
                        f"{combo}: keydown {k} is not key={key} code={code} ctrl={ctrl} alt={alt}"
                    )
            if not check(s, log, token):
                problems.append(f"{combo}: no effect ({meaning}); state {s}")
            if combo == "Ctrl+C":  # paste it back, as the row (not the executor) presses Ctrl+V
                await page.evaluate("document.getElementById('b').focus()")
                await page.keyboard.press("Control+v")
                if (await state(ex))["values"]["b"] != token:
                    problems.append(f"{combo}: the token did not reach the clipboard")
            if "frame_after" not in out.resolved:
                problems.append(f"{combo}: executed without resolved['frame_after']")
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------------------------- criterion 9


async def test_post_action_frame_shows_the_effect(server: str) -> None:
    """Criterion 9 adversarial (reviewer mutation 11). ``color.html``'s 400x400 block at (440,200) turns blue two
    animation frames after the first click and toggles on each later one; (500,500) -> (640,400), its centre. A
    frame taken before the settle shows the old colour in ~3 of 10 trials (probe_sandbox.py), so the row clicks
    eight times: a pre-settle screenshot survives all eight with probability ~0.7**8 = 0.06. The action text names
    the pixels and no URL path or query (the page URL carries both)."""
    want = [(0, 0, 255), (255, 0, 0)]
    async with executor() as ex:
        await run(ex.open(f"{server}/color.html?t0111c=secret-query"))
        f: Frame = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), (255, 0, 0)), rgb_at(f, 640, 400)
        for i in range(8):
            out = await execute(ex, inv("click_at", f, x=500, y=500))
            assert out.status == "executed", out
            latest = ex.latest_frame
            assert isinstance(latest, Frame)
            assert out.resolved["frame_after"] == latest.phash, (
                "frame_after is the newest own frame"
            )
            assert near(rgb_at(latest, 640, 400), want[i % 2]), (i, rgb_at(latest, 640, 400))
            assert "640" in out.action and "400" in out.action, out.action
            for leak in ("color.html", "secret", "?", "http", "127.0.0.1"):
                assert leak not in out.action, (leak, out.action)
            f = latest  # the post-action frame is an own frame: the next click may use it


# --------------------------------------------------------------------------------------------- criterion 10


async def test_page_closed_mid_action_is_failed(server: str) -> None:
    """Criterion 10 (reviewer mutation 3): ``busy.html`` blocks its renderer for 3 s on pointerdown, so the click is
    still in flight when the row closes the page. The outcome is ``failed`` with the exception type named."""
    async with executor() as ex:
        await run(ex.open(f"{server}/busy.html?ms=3000"))
        f = await run(ex.frame())
        task = asyncio.ensure_future(ex.execute(inv("click_at", f, x=500, y=500)))
        await asyncio.sleep(0.5)
        assert not task.done(), (
            "the click finished before the page was closed; the row tested nothing"
        )
        await run(ex.driver.page.close())
        out = await run(task)
        assert out.status == "failed", out
        assert names_an_exception(out.reason), out.reason


async def test_browser_killed_restart_once_no_retry(server: str) -> None:
    """Criterion 10 (reviewer mutation 4): SIGKILL the browser (only processes descended from this pytest process)
    mid-click. That call is ``failed`` naming the exception; the next call relaunches once, does not perform its
    action and is ``failed: browser restarted``; a pre-kill frame is refused; nothing is replayed."""
    async with executor() as ex:
        await run(ex.open(f"{server}/busy.html?ms=3000"))
        f0 = await run(ex.frame())
        mains = chromium_mains()
        assert len(mains) == 1, mains
        task = asyncio.ensure_future(ex.execute(inv("click_at", f0, x=500, y=500)))
        await asyncio.sleep(0.5)
        assert not task.done(), "the click finished before the kill; the row tested nothing"
        os.kill(mains[0][0], signal.SIGKILL)
        out = await run(task)
        assert out.status == "failed", out
        assert names_an_exception(out.reason), out.reason

        out2 = await execute(ex, inv("press_key", combo="Tab"))
        assert (out2.status, out2.reason) == ("failed", "browser restarted"), out2
        after = chromium_mains()
        assert len(after) == 1 and after[0][0] != mains[0][0], (mains, after)

        out3 = await execute(ex, inv("click_at", f0, x=500, y=500))
        assert (out3.status, out3.reason) == ("refused", "frame from before a browser restart"), (
            out3
        )

        await run(ex.open(f"{server}/grid.html"))
        f1 = await run(ex.frame())
        await asyncio.sleep(0.3)
        log = await page_log(ex)
        assert inputs(log) == [], f"something was replayed after the restart: {inputs(log)}"
        out4 = await execute(ex, inv("click_at", f1, x=500, y=500))
        assert out4.status == "executed", out4
        log = await page_log(ex)
        assert len(of_type(log, "click")) == 1 and of_type(log, "keydown") == [], log
        assert [p for p, _, _ in chromium_mains()] == [after[0][0]], "relaunched more than once"


# --------------------------------------------------------------------------------------------- criterion 11


async def test_cancel_mid_type_leaves_nothing_held(server: str) -> None:
    """Criterion 11 (reviewer mutation 6). Cancel ``execute`` while typing 2 000 characters: ``CancelledError``
    propagates, the ``input`` count is stable for 500 ms after the cancel returns, a following Tab shows every
    modifier false, and a pointermove shows ``buttons == 0``. Then the same for a cancel while a click's pointerdown
    is held up by ``busy.html``'s 800 ms handler."""
    async with executor() as ex:
        await run(ex.open(f"{server}/form.html"))
        f = await run(ex.frame())
        await clear_log(ex)
        task = asyncio.ensure_future(
            ex.execute(inv("type_at", f, x=234, y=150, text="x" * 2000, press_enter=False))
        )

        async def n_inputs() -> int:
            return int(
                await ex.driver.page.evaluate("window.__log.filter(e => e.type === 'input').length")
            )

        await poll(n_inputs, lambda n: n >= 1, timeout=10.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        n0 = await n_inputs()
        await asyncio.sleep(0.5)
        n1 = await n_inputs()
        assert n1 == n0, f"{n1 - n0} keystrokes arrived after the cancel returned"
        assert n0 < 2000, "the typing finished before the cancel; the row tested nothing"

        await clear_log(ex)
        out = await execute(ex, inv("press_key", combo="Tab"))
        assert out.status == "executed", out
        (tab,) = [e for e in of_type(await page_log(ex), "keydown") if e["key"] == "Tab"]
        assert not (tab["ctrlKey"] or tab["altKey"] or tab["shiftKey"] or tab["metaKey"]), tab
        await clear_log(ex)
        await ex.driver.page.mouse.move(700, 700)
        await ex.driver.page.mouse.move(710, 712)
        moves = of_type(await page_log(ex), "pointermove")
        assert moves and moves[-1]["buttons"] == 0, moves

        await run(ex.open(f"{server}/busy.html?ms=800"))
        f = await run(ex.frame())
        task = asyncio.ensure_future(ex.execute(inv("click_at", f, x=500, y=500)))
        await asyncio.sleep(0.3)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1.0)  # the busy handler has returned
        await clear_log(ex)
        await ex.driver.page.mouse.move(100, 100)
        await ex.driver.page.mouse.move(110, 112)
        moves = of_type(await page_log(ex), "pointermove")
        assert moves and moves[-1]["buttons"] == 0, f"the primary button is still down: {moves}"


# --------------------------------------------------------------------------------------------- criterion 13


def _artifact_files() -> set[Path]:
    tmp = Path(tempfile.gettempdir())
    return {p for d in tmp.glob("playwright-artifacts-*") for p in d.rglob("*") if p.is_file()}


async def test_downloads_uploads_dialogs_permissions_closed(server: str) -> None:
    """Criterion 13. On ``sandbox.html`` (centres in its CSS): a download saves nothing and is reported in
    ``resolved["blocked"]``; the file input gets no file; alert/confirm/prompt are dismissed (confirm returns false,
    prompt null) and reported; a beforeunload prompt is dismissed (the page stays) and reported; geolocation,
    notifications, camera and microphone are ``denied`` (Playwright's default is ``prompt``, probe_sandbox.py)."""
    payload = (FIXTURES / "t0111c-payload.txt").read_bytes()
    before_artifacts = _artifact_files()
    async with executor() as ex:
        await run(ex.open(f"{server}/sandbox.html"))
        f = await run(ex.frame())

        def at(px: int, py: int, frame: Frame) -> ToolInvocation:
            return inv("click_at", frame, x=coord(px, W), y=coord(py, H))

        out = await execute(ex, at(250, 150, f))
        assert out.status == "executed", out
        await asyncio.sleep(0.5)
        blocked = out.resolved.get("blocked")
        assert blocked, f"the download is not reported in resolved['blocked']: {out.resolved}"
        assert "download" in json.dumps(blocked).lower(), blocked
        new = [p for p in _artifact_files() - before_artifacts if p.read_bytes() == payload]
        assert new == [], f"the download was saved: {new}"
        assert not (Path.home() / "Downloads" / "t0111c-payload.txt").exists()

        out = await execute(ex, at(650, 150, f))
        assert out.status != "failed", out
        await asyncio.sleep(0.3)
        n_files = await ex.driver.page.evaluate("document.getElementById('file').files.length")
        assert n_files == 0
        assert of_type(await page_log(ex), "file-change") == []

        for (px, py), kind in (
            ((250, 350), "alert"),
            ((650, 350), "confirm"),
            ((1050, 350), "prompt"),
        ):
            out = await execute(ex, at(px, py, f))
            assert out.status == "executed", (kind, out)
            assert kind in json.dumps(out.resolved).lower(), (
                f"the {kind} dialog is not reported: {out.resolved}"
            )
        returned = {
            e["kind"]: e.get("value") for e in of_type(await page_log(ex), "dialog-returned")
        }
        assert returned == {"alert": None, "confirm": False, "prompt": None}, returned

        await ex.driver.page.evaluate("window.__armBeforeUnload = true")
        out = await execute(ex, at(250, 550, f))
        assert out.status == "executed", out
        await asyncio.sleep(0.5)
        assert "beforeunload" in json.dumps(out.resolved).lower(), out.resolved
        assert ex.driver.page.url.split("?")[0].endswith("/sandbox.html"), ex.driver.page.url

        states = await ex.driver.page.evaluate(
            """async () => { const o = {};
                 for (const n of ['geolocation', 'notifications', 'camera', 'microphone']) {
                   try { o[n] = (await navigator.permissions.query({name: n})).state; }
                   catch (e) { o[n] = 'error ' + e.name; } }
                 return o; }"""
        )
        assert states == dict.fromkeys(
            ("geolocation", "notifications", "camera", "microphone"), "denied"
        ), states


async def test_uses_bundled_chromium_not_system_chrome(server: str) -> None:
    """Criterion 13: the one browser process this row started is Playwright's Chromium (under its cache, the
    version in versions.lock), never ``/usr/bin/google-chrome``; the context is not persistent (``context.browser``
    is ``None`` for a persistent context); the profile is a temporary directory outside ``~/.config`` and
    ``~/.mozilla`` and is gone after ``close()``."""
    async with executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        await run(ex.frame())
        mains = chromium_mains()
        assert len(mains) == 1, mains
        _pid, exe, cmd = mains[0]
        real = Path(exe).resolve()
        assert real.is_relative_to(_browsers_path().resolve()), real
        assert not str(real).startswith(("/usr/", "/opt/", "/snap/")), real
        ctx = ex.driver.page.context
        assert ctx.browser is not None, "a persistent context (user_data_dir) was launched"
        assert ctx.browser.version == _lock()["chromium"], ctx.browser.version
        dirs = re.findall(r"--user-data-dir=(\S+)", cmd)
        assert len(dirs) == 1, cmd
        profile = Path(dirs[0]).resolve()
        assert profile.is_relative_to(Path(tempfile.gettempdir()).resolve()), profile
        for user in (".config", ".mozilla"):
            assert not profile.is_relative_to((Path.home() / user).resolve()), profile
        await run(ex.close())
    assert not profile.exists(), f"the profile {profile} outlived the browser"


# --------------------------------------------------------------------------------------------- criterion 14


async def test_egress_attempts_all_fail_and_are_logged(server: str) -> None:
    """Criterion 14 adversarial (reviewer mutations 7, 8). Phase 1, on this box: img/fetch/beacon/iframe/WebSocket,
    resource hints, a service worker that fetches on install, and a top-level navigation, all against the canary
    (this host's name -> 127.0.1.1, outside the allowlist). Unfenced Chromium connects to it 5 times for the
    attempts and once for the service worker (probe_sandbox.py); here the count must stay 0 and no service worker
    may register. (Unfenced headless Chromium made 0 preconnect connections in the probe, so the hint part has no
    power on this build; it is kept for the day it does.) Phase 2: the criterion's attempts against 198.51.100.7
    and example.invalid fail. Every attempt is in ``egress_log`` as exactly ``{host, scheme}``; neither the secret
    path nor the query appears in the log or in any action. Loopback fixture requests succeed."""
    hosts = canary_host()
    async with executor() as ex:
        await run(ex.open(f"{server}/egress.html"))
        page = ex.driver.page
        loop_ok = await run(page.evaluate("window.__loopback()"))
        assert loop_ok == {"img": {"ok": True}, "fetch": {"ok": True}}, loop_ok

        canary = None
        if hosts is not None:
            name, ip = hosts
            canary = Canary(ip)
            try:
                auth = f"{name}:{canary.port}"
                res = await run(page.evaluate("a => window.__attempt(a)", auth))
                await run(page.evaluate("a => window.__hints(a)", auth))
                sw = await run(
                    page.evaluate("t => window.__registerSW(t)", f"http://{auth}/sw-install")
                )
                await asyncio.sleep(1.0)
                reg = await run(
                    page.evaluate(
                        "async () => { const r = await navigator.serviceWorker?.getRegistration();"
                        " return r ? {active: !!r.active, installing: !!r.installing, waiting: !!r.waiting} : null; }"
                    )
                )
                await page.evaluate(
                    "u => { document.getElementById('nav').href = u; }",
                    f"http://{auth}/t0111c-secret-path/nav?token=t0111c-secret-query",
                )
                f = await run(ex.frame())
                nav = await execute(ex, inv("click_at", f, x=500, y=500))
                await asyncio.sleep(1.0)
                assert canary.count == 0, (
                    f"{canary.count} connection(s) reached {auth} outside the fence: attempts {res}, sw {sw}, "
                    f"registration {reg}"
                )
                assert reg is None or not any(reg.values()), f"a service worker registered: {reg}"
                # In-page, only fetch can tell: unfenced, img/iframe/WebSocket already read as failed against the
                # canary (it answers an empty 200, not an image or a WebSocket; a cross-origin iframe is
                # unreadable), so the canary count and the egress log are the oracles for those.
                assert res["fetch"]["ok"] is False, res
                for leak in ("secret", "?", "/t0111c"):
                    assert leak not in nav.action, nav.action
            finally:
                canary.close()
            await run(ex.open(f"{server}/egress.html"))
            page = ex.driver.page

        for host in ("198.51.100.7", "example.invalid"):
            res = await run(page.evaluate("a => window.__attempt(a)", host))
            for kind in ("img", "fetch", "websocket"):
                assert res[kind]["ok"] is False, (host, kind, res)
            await run(page.evaluate("a => window.__hints(a)", host))
        for host in ("198.51.100.7", "example.invalid"):
            await run(ex.open(f"{server}/egress.html"))
            page = ex.driver.page
            await page.evaluate(
                "u => { document.getElementById('nav').href = u; }",
                f"http://{host}/t0111c-secret-path/nav?token=t0111c-secret-query",
            )
            f = await run(ex.frame())
            nav = await execute(ex, inv("click_at", f, x=500, y=500))
            await asyncio.sleep(0.5)
            for leak in ("secret", "?", "/t0111c"):
                assert leak not in nav.action, nav.action

        log = list(ex.egress_log)
        for rec in log:
            assert set(rec) == {"host", "scheme"}, rec
            assert "t0111c-secret" not in json.dumps(rec) and "/" not in rec["host"], rec
        got = {(r["host"], r["scheme"]) for r in log}
        want = {
            ("198.51.100.7", "http"),
            ("198.51.100.7", "ws"),
            ("example.invalid", "http"),
            ("example.invalid", "ws"),
        }
        if hosts is not None:
            want |= {(hosts[0].lower(), "http"), (hosts[0].lower(), "ws")}
        got_lower = {(h.lower(), s) for h, s in got}
        assert want <= got_lower, (
            f"attempts missing from egress_log: {sorted(want - got_lower)}; log {sorted(got)}"
        )
        assert all(
            not h.startswith("127.0.0.1") and h not in ("localhost", "[::1]", "::1") for h, _ in got
        ), "a loopback request was logged as egress"
    if hosts is None:
        pytest.skip(
            "the rest of the row passed, but no hostname resolves to a loopback address other than 127.0.0.1 "
            "(/etc/hosts), so the canary phase -- the only phase that can see a connection -- did not run"
        )


async def test_redirects_to_off_loopback_reach_nothing(server: str) -> None:
    """Criterion 14, redirect vector (ruling a918d74; pins ``--host-resolver-rules``). A loopback 302 to the TCP
    canary on 127.0.1.1, for fetch, img, iframe and a top-level navigation: the canary accepts 0 connections for
    each. Probe on 785e88e (in-process, the executor with the resolver rule removed): 4 connections for these four
    vectors, both for an IP-literal and a hostname target; with the rule, 0. The route allowlist alone does not see a
    redirect's target."""
    async with executor() as ex:
        await run(ex.open(f"{server}/egress.html"))
        counts: dict[str, int] = {}
        for kind in ("fetch", "img", "iframe", "toplevel"):
            canary = Canary(CANARY_IP)
            try:
                target = f"http://{CANARY_IP}:{canary.port}/t0111c-secret-path/{kind}?token=t0111c-secret-query"
                hop = f"{server}/t0111c-redirect?to={urllib.parse.quote(target, safe='')}"
                page = ex.driver.page
                if kind == "fetch":
                    await page.evaluate("u => fetch(u, {mode: 'no-cors'}).catch(() => 0)", hop)
                elif kind == "img":
                    await page.evaluate("u => { const i = new Image(); i.src = u; }", hop)
                elif kind == "iframe":
                    await page.evaluate(
                        "u => { const f = document.createElement('iframe'); f.src = u; document.body.appendChild(f); }",
                        hop,
                    )
                else:
                    await page.evaluate("u => { document.getElementById('nav').href = u; }", hop)
                    f = await run(ex.frame())
                    out = await execute(ex, inv("click_at", f, x=500, y=500))
                    assert "secret" not in out.action, out.action
                await asyncio.sleep(1.5)
                counts[kind] = canary.count
            finally:
                canary.close()
            await run(ex.open(f"{server}/egress.html"))
        assert counts == dict.fromkeys(("fetch", "img", "iframe", "toplevel"), 0), (
            f"a loopback redirect reached {CANARY_IP}: {counts}"
        )


async def test_webrtc_absent_and_no_udp(server: str) -> None:
    """Criterion 14, WebRTC vector (ruling a918d74: WebRTC is disabled outright). In ``webrtc.html`` the four
    constructors are absent from the top frame, a same-origin child and a script-made about:blank child; a STUN and
    a TURN-UDP attempt at a UDP canary on 127.0.1.1 deliver 0 datagrams; and no process this row started has a UDP
    socket (``ss -uanp`` by PID). Probe on 785e88e: both attempts were made, the canary got 15 datagrams, and the
    browser held UDP sockets on 0.0.0.0, *, and 224.0.0.251:5353 (mDNS)."""
    async with executor() as ex:
        await run(ex.open(f"{server}/webrtc.html"))
        page = ex.driver.page
        await asyncio.sleep(0.3)  # the child frame loads
        types = await run(page.evaluate("window.__rtcTypes()"))
        canary = UdpCanary(CANARY_IP)
        try:
            tried = await run(
                page.evaluate(
                    "([s, t]) => window.__rtcAttempt(s, t)",
                    [
                        f"stun:{CANARY_IP}:{canary.port}",
                        f"turn:{CANARY_IP}:{canary.port}?transport=udp",
                    ],
                )
            )
            await asyncio.sleep(4.0)  # ICE gathering sends its STUN binding requests within this
            datagrams = canary.count
        finally:
            canary.close()
        udp = browser_udp_sockets()
    present = {w: [n for n, t in d.items() if t != "undefined"] for w, d in types.items()}
    assert all(v == [] for v in present.values()), f"WebRTC constructors present: {present}"
    assert datagrams == 0, f"{datagrams} datagram(s) reached the UDP canary; attempts {tried}"
    assert udp == [], f"the browser holds UDP sockets: {udp}"
    assert all(t != "made" for t in tried), tried


# --------------------------------------------------------------------------------------------- criterion 15

#: popup.html's PALETTE (page n has PALETTE[n]).
PALETTE = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 200, 0), (160, 0, 160), (0, 200, 200)]


async def test_popup_becomes_active_page_and_cap(server: str) -> None:
    """Criterion 15. ``popup.html?n=0`` is one ``target=_blank`` link to page n+1 in the colour PALETTE[n]. Each
    click makes the new page active, and the post-action frame and the next own frame show its colour. After the
    4th click five pages have been opened: at most four stay open, the oldest (n=0) is the one closed, and that is
    reported in the outcome (a ``resolved`` key naming "closed")."""
    async with executor() as ex:
        await run(ex.open(f"{server}/popup.html?n=0"))
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), PALETTE[0]), rgb_at(f, 640, 400)
        out = None
        for k in range(1, 5):
            out = await execute(ex, inv("click_at", f, x=500, y=500))
            assert out.status == "executed", (k, out)
            assert await ex.driver.page.evaluate("window.__n") == k, (
                f"page {k} is not the active page"
            )
            latest = ex.latest_frame
            assert latest is not None and near(rgb_at(latest, 640, 400), PALETTE[k]), (
                k,
                rgb_at(latest, 640, 400),
            )
            f = await run(ex.frame())
            assert near(rgb_at(f, 640, 400), PALETTE[k]), (k, rgb_at(f, 640, 400))
            assert len(ex.driver.page.context.pages) <= 4, (k, len(ex.driver.page.context.pages))
        open_ns = sorted([await p.evaluate("window.__n") for p in ex.driver.page.context.pages])
        assert open_ns == [1, 2, 3, 4], open_ns
        assert out is not None
        closed = [k for k in out.resolved if "closed" in k.lower()]
        assert closed and out.resolved[closed[0]], (
            f"closing the oldest page is not reported: {out.resolved}"
        )


async def test_late_popup_is_shown_by_next_frame(server: str) -> None:
    """Criterion 15 tightened (ruling a918d74): ``late_popup.html`` opens ``popup.html?n=2`` (blue) 300 ms after
    the click, after the executor's settle has returned. The next ``frame()`` shows the popup, which is the active
    page. The red opener is what a frame of the old page would show."""
    async with executor() as ex:
        await run(ex.open(f"{server}/late_popup.html"))
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), (255, 0, 0)), rgb_at(f, 640, 400)
        out = await execute(ex, inv("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        await asyncio.sleep(1.0)  # the popup opens at +300 ms
        nxt = await run(ex.frame())
        assert near(rgb_at(nxt, 640, 400), PALETTE[2]), (
            f"the next frame shows {rgb_at(nxt, 640, 400)}, not the popup"
        )
        assert await ex.driver.page.evaluate("window.__n") == 2, (
            "the late popup is not the active page"
        )


POINTER_EVENTS = ("pointerdown", "pointerup", "click", "dblclick", "contextmenu", "auxclick")


async def test_late_popup_refuses_the_openers_stale_frame(server: str) -> None:
    """Criterion 15 as ruled at 683cc1e (the late-popup misdirect). ``late_popup_misdirect.html``: a click on
    ``#open`` (left half, red) opens ``popup.html?n=2`` (blue) 400 ms later, after the settle. The row waits for
    the popup, then calls ``execute(click_at)`` carrying the opener's frame and aimed at ``#second`` (right half,
    green). Before the ruling the click landed on the popup at the opener's coordinates. Now it is ``refused`` with
    ``the active page changed; take a new frame``; neither the opener's ``#second`` nor the popup records a
    pointer event; and the next ``frame()`` shows the popup."""
    async with executor() as ex:
        await run(ex.open(f"{server}/late_popup_misdirect.html"))
        f = await run(ex.frame())
        assert near(rgb_at(f, 320, 400), (255, 0, 0)) and near(rgb_at(f, 960, 400), (0, 160, 0)), (
            rgb_at(f, 320, 400),
            rgb_at(f, 960, 400),
        )
        opener = ex.driver.page
        out = await execute(ex, inv("click_at", f, x=coord(320, W), y=coord(400, H)))
        assert out.status == "executed", out
        opener_frame = ex.latest_frame
        assert opener_frame is not None and out.resolved["frame_after"] == opener_frame.phash
        assert near(rgb_at(opener_frame, 960, 400), (0, 160, 0)), (
            "the post-action frame is not the opener's: the popup came before the settle returned and the row "
            "tests nothing"
        )
        pages = await poll(
            lambda: asyncio.sleep(0, list(opener.context.pages)),
            lambda ps: len(ps) == 2,
            timeout=5.0,
        )
        assert len(pages) == 2, f"the popup never opened: {pages}"
        popup = next(p for p in pages if p is not opener)
        await run(popup.wait_for_load_state("domcontentloaded"))
        await asyncio.sleep(0.3)
        await opener.evaluate("window.__log.length = 0")

        out = await execute(ex, inv("click_at", opener_frame, x=coord(960, W), y=coord(400, H)))
        await asyncio.sleep(0.3)
        opener_log = of_type(await opener.evaluate("window.__log"), *POINTER_EVENTS)
        popup_log = of_type(await popup.evaluate("window.__log"), *POINTER_EVENTS)
        assert (out.status, out.reason) == (
            "refused",
            "the active page changed; take a new frame",
        ), (out, popup_log)
        assert [e for e in opener_log if e.get("target") == "second"] == [], opener_log
        assert opener_log == [], f"the opener received pointer events: {opener_log}"
        assert popup_log == [], f"the stale-frame click reached the popup: {popup_log}"

        nxt = await run(ex.frame())
        assert near(rgb_at(nxt, 640, 400), PALETTE[2]), (
            f"the next frame shows {rgb_at(nxt, 640, 400)}, not the popup"
        )
        assert await ex.driver.page.evaluate("window.__n") == 2, "the popup is not the active page"


async def test_self_closed_popup_refuses_its_own_frame(server: str) -> None:
    """Criteria 4 and 15 as ruled on the round-3 review of e345ef2 (kills S9, no ``page_serial`` bump when a closed
    active page is replaced). ``self_close_popup.html``: a click on the red opener opens the blue popup, which the
    click's settle adopts; the row takes an own frame of the popup, then makes the popup call ``window.close()`` on
    itself. The next ``execute(click_at)`` carries the popup's frame; ``ensure()`` falls back to the opener, so the
    click is ``refused`` (``the active page changed; take a new frame``) and no pointer event reaches the opener
    (before the ruling it landed there, at the popup frame's coordinates). The guard's too-wide side: the next own
    frame shows the opener and a click on it executes."""
    async with executor() as ex:
        await run(ex.open(f"{server}/self_close_popup.html"))
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), (255, 0, 0)), rgb_at(f, 640, 400)
        opener = ex.driver.page
        out = await execute(ex, inv("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        popup = ex.driver.page
        assert popup is not opener and await popup.evaluate("window.__isPopup === true"), (
            "the premise failed: the popup is not the active page after the click's settle"
        )
        await run(popup.wait_for_function("window.__ready === true"))
        pf = await run(ex.frame())
        assert near(rgb_at(pf, 640, 400), (0, 0, 255)), (
            f"the premise failed: the popup's frame shows {rgb_at(pf, 640, 400)}"
        )

        await opener.evaluate("new BroadcastChannel('t0111c-self-close').postMessage('close')")
        closed = await poll(
            lambda: asyncio.sleep(0, popup.is_closed()), lambda c: bool(c), timeout=5.0
        )
        assert closed, "the premise failed: the popup did not close itself"
        await asyncio.sleep(0.2)
        await opener.evaluate("window.__log.length = 0")

        out = await execute(ex, inv("click_at", pf, x=500, y=500))
        await asyncio.sleep(0.3)
        opener_log = of_type(await opener.evaluate("window.__log"), *POINTER_EVENTS)
        assert (out.status, out.reason) == (
            "refused",
            "the active page changed; take a new frame",
        ), (out, opener_log)
        assert opener_log == [], f"the popup's frame was clicked onto the opener: {opener_log}"
        assert len(opener.context.pages) == 1, "the refused click opened another popup"

        nxt = await run(ex.frame())
        assert ex.driver.page is opener, "the opener is not the active page after the popup closed"
        assert near(rgb_at(nxt, 640, 400), (255, 0, 0)), rgb_at(nxt, 640, 400)
        out = await execute(ex, inv("click_at", nxt, x=500, y=500))
        assert out.status == "executed", out
        assert len(of_type(await opener.evaluate("window.__log"), "click")) == 1


#: Launch steps a cancelled first launch is made to hang in (the ruled row, and the first fence step).
LAUNCH_HANGS = ("deny_permissions", "context_route")


@pytest.mark.parametrize("hang_in", LAUNCH_HANGS)
async def test_launch_cancelled_before_alive_relaunches_with_the_fence(
    server: str, hang_in: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criteria 10 and 14 as ruled on the round-3 review of e345ef2 (kills S7, ``_alive = True`` set before the
    launch has installed the fence). ``ACTION_TIMEOUT_S`` is patched to 4 s (a cold launch here takes about 1 s)
    and the first launch only is made to sleep 60 s in one step: ``PlaywrightDriver._deny_permissions``, the last
    launch step (the ruled row), or Playwright's ``BrowserContext.route``, the first fence step (added: there a
    browser marked alive too early has no fence at all). The first ``execute`` launches the browser inside its
    timeout and fails naming ``TimeoutError``. The next ``execute`` relaunches (the main browser process is a new
    one; the spec gives no status for a call after a launch that never finished, so ``executed`` and ``failed:
    browser restarted`` are both accepted, and it must not time out again). On the relaunched browser
    ``RTCPeerConnection`` is undefined, and a fetch to a TCP canary on 127.0.1.1 (loopback, outside the allowlist)
    reaches nothing and is logged by the route as ``{"host": "127.0.1.1", "scheme": "http"}`` (the resolver rule
    alone would also stop the connection, so the log entry is what shows the route is installed)."""
    from playwright.async_api import BrowserContext

    mod = computer()
    executor_mod = importlib.import_module("local_astra.computer.executor")
    runtime_mod = importlib.import_module("local_astra.computer.runtime")
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 4.0)
    hit: list[str] = []

    def first_only(orig: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kw: Any) -> Any:
            if not hit:
                hit.append(hang_in)
                await asyncio.sleep(60)
            return await orig(*args, **kw)

        return wrapper

    if hang_in == "deny_permissions":
        cls = runtime_mod.PlaywrightDriver
        monkeypatch.setattr(cls, "_deny_permissions", first_only(cls._deny_permissions))
    else:
        monkeypatch.setattr(BrowserContext, "route", first_only(BrowserContext.route))

    async with executor() as ex:
        assert isinstance(ex, mod.BrowserToolExecutor)
        out = await execute(ex, inv("press_key", combo="Tab"), timeout=20.0)
        assert hit == [hang_in], f"the premise failed: the launch never reached {hang_in}"
        assert out.status == "failed" and "TimeoutError" in (out.reason or ""), out
        first = chromium_mains()

        out2 = await execute(ex, inv("press_key", combo="Tab"), timeout=20.0)
        assert out2.status == "executed" or (out2.status, out2.reason) == (
            "failed",
            "browser restarted",
        ), out2
        after = chromium_mains()
        assert len(after) == 1, after
        assert [p for p, _, _ in first] != [after[0][0]], (
            f"no relaunch: the browser whose launch was cancelled is still the one in use ({first}); it was "
            "marked alive before its fence was installed"
        )

        await run(ex.open(f"{server}/webrtc.html"))
        await asyncio.sleep(0.3)  # the child frames load
        page = ex.driver.page
        assert await page.evaluate("typeof RTCPeerConnection") == "undefined", (
            "WebRTC is present on the relaunched page: the init script is not installed"
        )
        types = await run(page.evaluate("window.__rtcTypes()"))
        present = {w: [n for n, t in d.items() if t != "undefined"] for w, d in types.items()}
        assert all(v == [] for v in present.values()), present

        canary = Canary(CANARY_IP)
        try:
            res = await run(
                page.evaluate(
                    "u => fetch(u, {mode: 'no-cors'}).then(() => 'ok', e => 'error ' + e.name)",
                    f"http://{CANARY_IP}:{canary.port}/t0111c-launch-canary",
                )
            )
            await asyncio.sleep(0.5)
            assert canary.count == 0, f"{canary.count} connection(s) reached the canary ({res})"
        finally:
            canary.close()
        assert res != "ok", res
        assert {"host": CANARY_IP, "scheme": "http"} in list(ex.egress_log), (
            f"the off-loopback fetch is not in egress_log, so the route is not installed: {ex.egress_log}"
        )


async def test_reopened_page_after_restart_is_clickable(server: str) -> None:
    """Criterion 2/4 as ruled at 683cc1e (the reviewer's livelock, live): ``grid.html`` is one uniform element, so
    re-opening it after a browser restart draws the pre-restart pixels. The pre-restart frame is still refused;
    the fresh frame carries a different phash, is accepted, and its click reaches the page once."""
    async with executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        f0 = await run(ex.frame())
        mains = chromium_mains()
        assert len(mains) == 1, mains
        os.kill(mains[0][0], signal.SIGKILL)
        await asyncio.sleep(0.5)
        out = await execute(ex, inv("press_key", combo="Tab"))
        assert (out.status, out.reason) == ("failed", "browser restarted"), out
        await run(ex.open(f"{server}/grid.html"))
        f1 = await run(ex.frame())
        assert mean_abs_diff(f0, f1) < 2.0, "the re-opened page does not draw the same pixels"
        assert f1.phash != f0.phash, "identical pixels after a restart carry the pre-restart phash"
        out = await execute(ex, inv("click_at", f0, x=500, y=500))
        assert (out.status, out.reason) == ("refused", "frame from before a browser restart"), out
        await asyncio.sleep(0.2)
        assert inputs(await page_log(ex)) == [], "the refused click dispatched something"
        out = await execute(ex, inv("click_at", f1, x=500, y=500))
        assert out.status == "executed", out
        assert len(of_type(await page_log(ex), "click")) == 1


# --------------------------------------------------------------------------------------------- criterion 1 (live)


async def test_search_is_the_dry_run_description_live(server: str) -> None:
    """Criterion 1 against the real runtime: ``google_search`` equals ``DryRunToolExecutor`` field by field."""
    async with executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        for args in ({"query": "omelette recipe"}, {}):
            i = inv("google_search", **args)
            assert await execute(ex, i) == await DryRunToolExecutor().execute(i)
