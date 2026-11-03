"""T-0111g C-E1 (and B13's e2e half): a real ``Session`` over the fakes driving the real headless browser on the real
start pages through the tool loop (docs/tasks/T-0111g.md; ``make browser-live``).

**Gate.** As ``tests/integration/test_computer_browser_e2e.py`` (its autouse gate imported): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; a missing ``playwright`` or Chromium is then a failure naming
P1. Every row is ``pending(task="T-0111g")``.

**The rig.** ``tests/unit/_tool_turn_support.tool_rig`` with the loop on and a live kit
(``_live_browser_support.live_kit``) whose executor is a real ``BrowserToolExecutor(headless=True,
end_when_closed=False, allowed_origins={start-page origin})`` behind ``Watch``, a proxy that **delegates every name**
to the real executor and only counts the calls in flight and records each execute's outcome and post-action frame.
The session runs on the ``ScriptedClock`` while the browser runs on wall time, so ``drive`` advances virtual time
only while no browser call is in flight (otherwise the session's virtual 10 s deadline would cut a real action).
Each step's coordinates are computed from the live DOM just before that step's request is released
(``FakeLlm.hold_steps``): the page is the browser's, the script is the test's.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless, 1280x800 at DSF 1, the start-page server on 127.0.0.1 at
an ephemeral port, FakeLlm with one script per step, ScriptedClock, fake STT/TTS/audio, n = 1 per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    chromium_mains,
    computer,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import data_url, live_kit
from tests.unit._tool_turn_support import answer, tool, tool_rig
from tests.unit.test_start_pages import new_server

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

W, H = 1280, 800
Q = '<|"|>'


# --------------------------------------------------------------------------------------------- the rig


class Watch:
    """Delegates every name to the real executor; counts in-flight calls; records outcomes and post frames."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.inflight = 0
        self.outcomes: list[Any] = []
        self.invocations: list[Any] = []
        self.after: list[Any] = []
        #: (browser generation when the frame returned, the frame) for every ``frame()`` the session took
        self.frames: list[tuple[int, Any]] = []
        self.before_execute: Callable[[], None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def frame(self) -> Any:
        self.inflight += 1
        try:
            out = await self._inner.frame()
        finally:
            self.inflight -= 1
        self.frames.append((self._inner.driver.generation, out))
        return out

    async def execute(self, inv: Any) -> Any:
        if self.before_execute is not None:
            hook, self.before_execute = self.before_execute, None
            hook()
        self.inflight += 1
        try:
            out = await self._inner.execute(inv)
        finally:
            self.inflight -= 1
        self.invocations.append(inv)
        self.outcomes.append(out)
        self.after.append(self._inner.latest_frame)
        return out


@contextlib.asynccontextmanager
async def live_session(tmp_path: Path, start: str = "/") -> AsyncIterator[tuple[Any, Watch, str]]:
    srv = new_server()
    srv.start()
    origin = srv.origin
    ex = computer().BrowserToolExecutor(
        now_ms=lambda: 0,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({origin}),
    )
    watch = Watch(ex)
    try:
        await ex.open(origin + start)
        async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.LOOP) as rig:
            ex._now_ms = lambda: rig.clock.now_ms
            yield rig, watch, origin
    finally:
        with contextlib.suppress(Exception):
            await ex.close()
        srv.stop()


async def drive(
    rig: Any, watch: Watch, done: Callable[[], bool], what: str, limit_s: float = 90
) -> None:
    t0 = time.monotonic()
    await rig.pump()
    while not done():
        assert time.monotonic() - t0 < limit_s, (
            f"{what} (state {rig.state.state}, calls {len(watch.outcomes)})"
        )
        if watch.inflight or rig.state.awaiting_result is not None:
            await asyncio.sleep(0.05)
            await rig.pump(rounds=1)
        else:
            await rig.advance(50)


async def ask(rig: Any, watch: Watch, text: str) -> None:
    await rig.onset()
    await drive(rig, watch, lambda: watch.inflight == 0, "the onset frame never came")
    await rig.words(text)
    await rig.end_of_turn()


def coord(px: float, size: int) -> int:
    """The 0-999 value ``tool_executor.to_pixels`` maps to ``px`` (its inverse, rounded)."""
    return max(0, min(999, round(px * 999 / (size - 1))))


async def center(ex: Any, js: str) -> tuple[int, int]:
    """The 0-999 centre of the element the expression ``js`` returns, on the active page."""
    box = await ex.driver.page.evaluate(
        f"(() => {{ const e = ({js}); if (!e) return null; const r = e.getBoundingClientRect();"
        " return [r.left + r.width / 2, r.top + r.height / 2]; })()"
    )
    assert box is not None, f"no element for {js}"
    return coord(box[0], W), coord(box[1], H)


async def state(ex: Any, key: str) -> Any:
    return await ex.driver.page.evaluate(f"window.__state && window.__state.{key}")


SEARCH_BOX = "document.querySelector('input')"
BLUE_RESULT = "[...document.querySelectorAll('a')].find(a => /blue mug/i.test(a.textContent))"
ADD_BLUE = (
    "[...document.querySelectorAll('button,a,input[type=button],input[type=submit]')]"
    ".filter(b => /^add to cart$/i.test((b.textContent || b.value || '').trim()))"
    ".find(b => { let c = b; while (c.parentElement && c.parentElement !== document.body"
    " && c.parentElement.querySelectorAll('button').length === 1) c = c.parentElement;"
    " return /blue mug/i.test(c.textContent); })"
)
CHEAPEST_ADD = (
    "(() => { const bs = [...document.querySelectorAll('button')].filter(b => /^add to cart$/i.test("
    "b.textContent.trim())); const price = b => { let c = b; while (c.parentElement && c.parentElement !== "
    "document.body && c.parentElement.querySelectorAll('button').length === 1) c = c.parentElement; "
    "const m = c.textContent.match(/(\\d+[.,]\\d{2})/); return m ? parseFloat(m[1].replace(',', '.')) : 1e9; };"
    " return bs.reduce((a, b) => price(b) < price(a) ? b : a); })()"
)


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


def type_at(x: int, y: int, text: str, enter: bool) -> str:
    e = "true" if enter else "false"
    return f"<|tool_call>call:type_at{{press_enter:{e},text:{Q}{text}{Q},x:{x},y:{y}}}<tool_call|>"


SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"


async def release_step(rig: Any, watch: Watch, n: int, script: list[Any]) -> None:
    """Wait for result ``n`` (the step-n request is held), give step n+1 its script, release it."""
    await drive(
        rig, watch, lambda: len(watch.outcomes) >= n and watch.inflight == 0, f"step {n} never ran"
    )
    rig.llm.step_scripts = [list(script)]
    rig.llm.release_steps()
    await drive(
        rig, watch, lambda: len(tsl.steps_of(rig)) >= n, f"step {n + 1} was never requested"
    )
    rig.llm.hold_steps()


# --------------------------------------------------------------------------------------------- sockets


def _addr(hex_addr: str) -> bytes:
    ip, _port = hex_addr.split(":")
    raw = bytes.fromhex(ip)
    if len(raw) == 4:
        return raw[::-1]
    return b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))


def _loopback(raw: bytes) -> bool:
    if len(raw) == 4:
        return raw[0] == 127 or raw == b"\0\0\0\0"
    if raw == socket.inet_pton(socket.AF_INET6, "::1") or raw == bytes(16):
        return True
    return raw[:12] == b"\0" * 10 + b"\xff\xff" and raw[12] == 127


def non_loopback_sockets(pids: list[int]) -> list[str]:
    """Sockets of ``pids`` (from ``/proc/<pid>/fd`` inodes) whose local bind or peer is off loopback. A listener's
    peer ``0.0.0.0:0`` counts as no peer; a bind to ``0.0.0.0``/``::`` counts as off loopback."""
    inodes: set[str] = set()
    for pid in pids:
        with contextlib.suppress(OSError):
            for fd in os.listdir(f"/proc/{pid}/fd"):
                with contextlib.suppress(OSError):
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                    if target.startswith("socket:["):
                        inodes.add(target[8:-1])
    bad: list[str] = []
    for table in ("tcp", "tcp6", "udp", "udp6"):
        with contextlib.suppress(OSError):
            for line in Path(f"/proc/{os.getpid()}/net/{table}").read_text().splitlines()[1:]:
                f = line.split()
                if f[9] not in inodes:
                    continue
                local, peer = _addr(f[1]), _addr(f[2])
                if local in (b"\0\0\0\0", bytes(16)) or not _loopback(local):
                    bad.append(f"{table} local {f[1]}")
                if not _loopback(peer):
                    bad.append(f"{table} peer {f[2]}")
    return bad


def descendants() -> list[int]:
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            with contextlib.suppress(OSError):
                stat = Path(f"/proc/{d}/stat").read_text()
                children.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(d))
    out, todo = [], [os.getpid()]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def per_turn_checks(rig: Any, turn: str) -> None:
    assert rig.record(turn).extras.get("browser_egress_blocked") == [], rig.record(turn).extras
    assert non_loopback_sockets(descendants()) == [], "C-E1: a socket off loopback"


def newest_jpeg_is(rig: Any, k: int, jpeg: bytes) -> None:
    body = tsl.step_request(rig, k)
    urls = [
        p.url for m in body.messages for p in m.content if getattr(p, "type", "") == "image_url"
    ]
    assert urls and urls[-1] == data_url(jpeg), (
        f"C-E1: step {k}'s newest image is the post-action frame"
    )


# --------------------------------------------------------------------------------------------- rows


async def test_search_shop_add_blue_mug(tmp_path: Path) -> None:
    """C-E1 turn 1: type "mug" + Enter on the start page, click the Blue mug result, click its Add to cart. Pass:
    ``cart == ["blue mug"]``; each step k+1's newest image is step k's post-action JPEG; each call's geometry phash
    is the previous ``frame_after``."""
    async with live_session(tmp_path) as (rig, watch, _origin):
        ex = watch._inner
        x, y = await center(ex, SEARCH_BOX)
        rig.llm.script = [answer("I will search for mugs. "), tool(type_at(x, y, "mug", True))]
        rig.llm.hold_steps()
        await ask(rig, watch, "search for mugs and add the blue mug to the cart")
        await drive(rig, watch, lambda: len(watch.outcomes) >= 1 and watch.inflight == 0, "step 1")
        bx, by = await center(ex, BLUE_RESULT)
        await release_step(rig, watch, 1, [answer(""), tool(click(bx, by))])
        await drive(rig, watch, lambda: len(watch.outcomes) >= 2 and watch.inflight == 0, "step 2")
        ax, ay = await center(ex, ADD_BLUE)
        await release_step(rig, watch, 2, [answer(""), tool(click(ax, ay))])
        await release_step(rig, watch, 3, [answer("The blue mug is in the cart. ")])
        rig.llm.release_steps()
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        assert [o.status for o in watch.outcomes] == ["executed"] * 3, watch.outcomes
        assert await state(ex, "cart") == ["blue mug"]
        for k in (2, 3, 4):
            newest_jpeg_is(rig, k, watch.after[k - 2].jpeg)
        for k in range(1, 3):
            assert watch.outcomes[k - 1].resolved["frame_after"] == watch.after[k - 1].phash
            assert (
                watch.invocations[k].frame.phash == watch.outcomes[k - 1].resolved["frame_after"]
            ), f"C-E1: step {k + 1}'s geometry is step {k}'s frame_after"
        per_turn_checks(rig, "t1")


async def test_moving_target_resolves_on_new_frame(tmp_path: Path) -> None:
    """C-E1 turn 2 (adversarial): scroll down on deals, then click the cheapest item's button at its post-scroll
    position; the cart holds that item (without the swap, B9 refuses step 2 and the cart stays empty)."""
    async with live_session(tmp_path, "/demo/deals.html") as (rig, watch, _origin):
        ex = watch._inner
        rig.llm.script = [answer("I will look further down. "), tool(SCROLL)]
        rig.llm.hold_steps()
        await ask(rig, watch, "add the cheapest item to the cart")
        await drive(rig, watch, lambda: len(watch.outcomes) >= 1 and watch.inflight == 0, "step 1")
        cx, cy = await center(ex, CHEAPEST_ADD)
        await release_step(rig, watch, 1, [answer(""), tool(click(cx, cy))])
        await release_step(rig, watch, 2, [answer("Added the cheapest item. ")])
        rig.llm.release_steps()
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        assert [o.status for o in watch.outcomes] == ["executed", "executed"], watch.outcomes
        cart = await state(ex, "cart")
        assert isinstance(cart, list) and len(cart) == 1, cart
        per_turn_checks(rig, "t1")


async def test_kill_then_fresh_frame_click_executes(tmp_path: Path) -> None:
    """C-E1 turn 3: the browser is killed right before step 1's execute; step 1 is ``failed`` with a reason naming the
    browser's death, and step 2 carries a fresh own frame of the new browser generation (not the pre-kill frame) and
    its click executes.

    T-0111g ruling "on the group B implementation" (C-E1 iii): Playwright may not yet have seen the disconnect, so
    the click is dispatched to a dead target and step 1 reads ``failed: TargetClosedError during click_at`` rather
    than ``failed: browser restarted``; both are accepted. What the row protects is the recovery: if the session
    kept the previous image on ``failed`` (B8), step 2 would read the pre-kill frame and this row goes red."""
    async with live_session(tmp_path, "/demo/shop.html") as (rig, watch, _origin):
        ex = watch._inner
        pre: dict[str, Any] = {}

        def kill() -> None:
            pre["generation"] = ex.driver.generation
            pre["frame"] = ex.latest_frame
            for pid, _exe, _cmd in chromium_mains():
                os.kill(pid, signal.SIGKILL)

        watch.before_execute = kill
        x, y = await center(ex, ADD_BLUE)
        rig.llm.script = [answer("I will add it. "), tool(click(x, y))]
        rig.llm.hold_steps()
        await ask(rig, watch, "add the blue mug to the cart")
        await release_step(rig, watch, 1, [answer(""), tool(click(500, 500))])
        await release_step(rig, watch, 2, [answer("Done. ")])
        rig.llm.release_steps()
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        first = watch.outcomes[0]
        assert first.status == "failed" and (
            first.reason == "browser restarted" or "TargetClosedError" in (first.reason or "")
        ), (
            f"C-E1 (iii): step 1 fails naming the browser's death; got {first.status!r} {first.reason!r}"
        )
        assert pre.get("frame") is not None, "premise: an own frame existed before the kill"
        fresh = [f for g, f in watch.frames if g > pre["generation"] and f is not None]
        assert fresh, (
            f"C-E1 (iii): the session took a frame of the new browser generation "
            f"(pre-kill generation {pre['generation']}, frames at {[g for g, _ in watch.frames]})"
        )
        # the request's own parts (as ``newest_jpeg_is``): ``image_urls`` reads the wire dict shape
        # ``{"image_url": {"url": ...}}``, which this contract's ``model_dump`` does not produce
        urls = [
            p.url
            for m in tsl.step_request(rig, 2).messages
            for p in m.content
            if getattr(p, "type", "") == "image_url"
        ]
        assert urls, "step 2 carries an image"
        assert len(watch.invocations) >= 2, watch.invocations
        step2 = watch.invocations[1].frame
        assert step2 is not None and step2.phash != pre["frame"].phash, (
            "C-E1 (iii): step 2's geometry is not the pre-kill frame (the previous image was kept on failed)"
        )
        assert step2.phash in {f.phash for f in fresh}, (
            "C-E1 (iii): step 2's geometry is a fresh own frame of the new browser generation"
        )
        shown = next(f for f in fresh if f.phash == step2.phash)
        assert urls[-1] == data_url(shown.jpeg), (
            "C-E1 (iii): step 2's newest image is that fresh new-generation frame"
        )
        assert watch.outcomes[1].status == "executed", watch.outcomes[1]
        per_turn_checks(rig, "t1")


async def test_cancel_mid_type_through_session(tmp_path: Path) -> None:
    """B13 e2e: a takeover while ``type_at`` is typing cancels it; no keystroke arrives after the turn closes."""
    from local_astra.contracts import events as E

    async with live_session(tmp_path) as (rig, watch, _origin):
        ex = watch._inner
        x, y = await center(ex, SEARCH_BOX)
        rig.llm.script = [answer("Typing it. "), tool(type_at(x, y, "m" * 400, False))]
        await ask(rig, watch, "type a lot")
        await drive(
            rig,
            watch,
            lambda: watch.inflight > 0 and rig.state.awaiting_result is not None,
            "typing",
        )
        await asyncio.sleep(0.05)
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        await drive(rig, watch, lambda: watch.inflight == 0, "the cancel never settled")
        typed = await ex.driver.page.evaluate("document.querySelector('input').value.length")
        await asyncio.sleep(1.5)
        later = await ex.driver.page.evaluate("document.querySelector('input').value.length")
        assert later == typed, f"T-0111g B13: keystrokes after the turn closed ({typed} -> {later})"


async def test_sockets_loopback_only(tmp_path: Path) -> None:
    """C-E1: zero sockets of this process and its Chromium children with a non-loopback peer or bind, read from
    ``/proc`` (no dependency added); the parser's power: a socket bound to 0.0.0.0 is reported."""
    async with live_session(tmp_path) as (rig, watch, _origin):
        rig.llm.script = [answer("Hello. ")]
        await ask(rig, watch, "hello")
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        assert chromium_mains(), "premise: a browser is running"
        assert non_loopback_sockets(descendants()) == []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            s.listen(1)
            assert non_loopback_sockets([os.getpid()]), "power: a 0.0.0.0 bind is seen"
