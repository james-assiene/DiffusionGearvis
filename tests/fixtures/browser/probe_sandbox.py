"""T-0111c probe: the facts the e2e module's oracles rest on, measured on plain Playwright (no executor).

Run (loopback only; the "canary" is a TCP listener on the address this box's hostname resolves to, 127.0.1.1)::

    uv run python tests/fixtures/browser/probe_sandbox.py

Each line is one fact; the output recorded on 2026-10-20 is in ``MANIFEST.md``. What each fact is for:

- ``settle``: ``color.html`` turns blue two animation frames after the click. A screenshot taken straight after the
  click shows red; after two frames, blue. So a post-action frame taken before the settle is caught (mutation 11).
- ``canary``: *unfenced* Chromium does connect to the canary for img/fetch/beacon/iframe/WebSocket, for a preconnect
  hint and for a service worker's fetch; so a zero count under the executor is evidence, not a vacuous pass.
- ``download``/``dialog``/``permissions``/``filechooser``/``beforeunload``: the defaults the executor must close.
- ``cancel``: cancelling ``keyboard.type`` stops the keystrokes (the count is stable after the cancel).
- ``kill``: what a pending action raises when the browser process is SIGKILLed.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import io
import os
import signal
import socket
import threading
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


class Canary:
    """Counts TCP connections accepted on ``(host, port)``."""

    def __init__(self, host: str) -> None:
        self.sock = socket.socket()
        self.sock.bind((host, 0))
        self.sock.listen(64)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.count = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, name="t0111c-probe-canary", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop:
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
        self._stop = True
        self._t.join(1)
        self.sock.close()


def chromium_pids() -> list[int]:
    """Main browser processes descended from this one: an executable under ``ms-playwright`` whose parent's is not
    (Chromium rewrites its argv, so ``--type=`` cannot be read reliably from ``/proc/<pid>/cmdline``)."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(d))

    def exe(pid: int) -> str:
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return ""

    out, todo = [], [os.getpid()]
    while todo:
        p = todo.pop()
        for c in children.get(p, []):
            todo.append(c)
            if "ms-playwright" in exe(c) and "ms-playwright" not in exe(p):
                out.append(c)
    return out


def centre_rgb(png: bytes, x: int, y: int) -> tuple[int, ...]:
    return tuple(Image.open(io.BytesIO(png)).convert("RGB").getpixel((x, y)))


async def main() -> None:
    handler = functools.partial(_Quiet, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    hostname = socket.gethostname()
    canary_ip = socket.gethostbyname(hostname)
    print("hostname", hostname, "->", canary_ip)
    raf2 = "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        print("chromium", browser.version, "pids", chromium_pids())

        async def fresh(**kw: Any) -> Any:
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 800}, device_scale_factor=1, **kw
            )
            return ctx, await ctx.new_page()

        # settle
        for wait in (False, True):
            ctx, page = await fresh()
            await page.goto(base + "/color.html")
            await page.mouse.click(640, 400)
            if wait:
                await page.evaluate(raf2)
            shot = await page.screenshot(type="png")
            print("settle wait_two_frames=", wait, "centre rgb", centre_rgb(shot, 640, 400))
            await ctx.close()

        # canary, unfenced
        for what in ("attempt", "hints", "sw"):
            canary = Canary(canary_ip)
            auth = f"{hostname}:{canary.port}"
            ctx, page = await fresh()
            await page.goto(base + "/egress.html")
            if what == "attempt":
                res = await page.evaluate("a => window.__attempt(a)", auth)
            elif what == "hints":
                res = await page.evaluate("a => window.__hints(a)", auth)
                await asyncio.sleep(1.0)
            else:
                res = await page.evaluate("t => window.__registerSW(t)", f"http://{auth}/sw-fetch")
                await asyncio.sleep(1.0)
            await asyncio.sleep(0.3)
            print("canary unfenced", what, "connections", canary.count, "page", res)
            await ctx.close()
            canary.close()
        sw_state = """async () => { const r = await navigator.serviceWorker.getRegistration();
            return r ? {installing: !!r.installing, waiting: !!r.waiting, active: !!r.active} : null; }"""
        for sw in ("allow", "block"):
            canary = Canary(canary_ip)
            ctx, page = await fresh(service_workers=sw)
            await page.goto(base + "/egress.html")
            res = await page.evaluate(
                "t => window.__registerSW(t)", f"http://{hostname}:{canary.port}/sw-fetch"
            )
            await asyncio.sleep(1.0)
            state = await page.evaluate(sw_state)
            print(
                f"canary sw service_workers={sw} register",
                res,
                "registration",
                state,
                "connections",
                canary.count,
            )
            await ctx.close()
            canary.close()
        canary = Canary(canary_ip)
        ctx, page = await fresh()
        await page.goto(base + f"/hints.html?a={hostname}:{canary.port}")
        await asyncio.sleep(1.5)
        print("canary unfenced static hints connections", canary.count)
        await ctx.close()
        canary.close()
        ctx, page = await fresh()
        await page.goto(base + "/egress.html")
        print("loopback controls", await page.evaluate("window.__loopback()"))
        await ctx.close()

        # download
        ctx, page = await fresh(accept_downloads=False)
        await page.goto(base + "/sandbox.html")
        events: list[Any] = []
        page.on("download", lambda d: events.append(("download", d.suggested_filename)))
        page.on("dialog", lambda d: events.append(("dialog", d.type)))
        await page.mouse.click(250, 150)
        await asyncio.sleep(0.5)
        print("download accept_downloads=False events", events)
        # file chooser, no handler
        await page.mouse.click(650, 150)
        await asyncio.sleep(0.3)
        print(
            "filechooser files.length",
            await page.evaluate("document.getElementById('file').files.length"),
        )
        await ctx.close()

        # dialogs: no handler at all, then with a handler
        for handled in (False, True):
            ctx, page = await fresh()
            await page.goto(base + "/sandbox.html")
            seen: list[str] = []
            if handled:
                page.on(
                    "dialog",
                    lambda d, seen=seen: (seen.append(d.type), asyncio.ensure_future(d.dismiss())),
                )
            for x, y in ((250, 350), (650, 350), (1050, 350)):
                await page.mouse.click(x, y)
                await asyncio.sleep(0.2)
            log = await page.evaluate("window.__log.filter(e => e.type === 'dialog-returned')")
            print("dialog handled=", handled, "seen", seen, "page", log)
            await ctx.close()

        # beforeunload on a click-initiated navigation
        ctx, page = await fresh()
        await page.goto(base + "/sandbox.html")
        seen = []
        page.on("dialog", lambda d: (seen.append(d.type), asyncio.ensure_future(d.dismiss())))
        await page.evaluate("window.__armBeforeUnload = true")
        await page.mouse.click(250, 550)
        await asyncio.sleep(1.0)
        print("beforeunload seen", seen, "url ends", page.url.rsplit("/", 1)[-1])
        await ctx.close()

        # permissions
        ctx, page = await fresh()
        await page.goto(base + "/grid.html")
        states = await page.evaluate(
            """async () => { const o = {}; for (const n of ['geolocation','notifications','camera','microphone'])
                 { try { o[n] = (await navigator.permissions.query({name: n})).state } catch (e) { o[n] = 'err ' + e.name } }
                 o.Notification = Notification.permission; return o; }"""
        )
        print("permissions default", states)
        await ctx.close()

        # cancel mid-type
        ctx, page = await fresh()
        await page.goto(base + "/form.html")
        await page.click("#q")
        task = asyncio.ensure_future(page.keyboard.type("x" * 2000))
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        n0 = await page.evaluate("window.__log.filter(e => e.type === 'input').length")
        await asyncio.sleep(0.5)
        n1 = await page.evaluate("window.__log.filter(e => e.type === 'input').length")
        print("cancel keyboard.type inputs at cancel", n0, "after 500 ms", n1)
        await ctx.close()

        # kill mid-type
        ctx, page = await fresh()
        await page.goto(base + "/form.html")
        await page.click("#q")
        task = asyncio.ensure_future(page.keyboard.type("y" * 2000))
        await asyncio.sleep(0.3)
        pids = chromium_pids()
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        try:
            await task
            print("kill: type returned normally")
        except Exception as exc:
            print("kill: pids", pids, "raised", type(exc).__module__, type(exc).__name__)
        print("kill: browser.is_connected", browser.is_connected())
    srv.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
