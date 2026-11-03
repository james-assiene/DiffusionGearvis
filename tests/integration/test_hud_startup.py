"""T-0165: the HUD page opens its socket only once its panels have mounted, and loads quickly.

``docs/tasks/T-0165.md`` C1-C5 and C7, in headless Chrome against the real ``HudServer`` and the
real ``Session`` over ``tests/unit/_hud_rig.py`` (the ``Live`` rig of
``tests/integration/test_hud_rehearsal.py``, imported, not edited), with the page served by the
**production** ``HudStaticServer`` (its keep-alive behaviour is part of what C3 depends on).

**Regime.** Headless Chrome as found by ``requires_chrome`` (version in the C7 failure message),
``HudStaticServer(HUD_DIR, 0)`` on 127.0.0.1, a fresh profile per ``Chrome`` (no cache), DSF 1,
real ``SessionClock`` and ``RECONNECT_GRACE_MS`` (4000 since T-0166), n=1 per row except C7 (n=3 loads, every
load must pass), no seed. Wall-clock waits are generous ceilings, never latency claims; C7 asserts
counts and orderings only, so it holds at any load average.

**Seams.** ``Page.addScriptToEvaluateOnNewDocument`` installs ``INIT_JS`` before any page script:
it wraps ``window.WebSocket`` (a ``Proxy``, so the statics stay) to record what the page looked like
at the first construction, and a ``MutationObserver`` records ``[data-ask-input].disabled`` at the
first moment ``#conn-status`` reads ``connected`` (C2: an observer, not polling).
``<html data-panels>`` is the task's own test seam (C1; implementation notes: ``"mounted"`` when
mounting settles, something else, e.g. ``"timeout"``, on the C5 ceiling path).

Activated with T-0165's implementation (2026-10-27): no ``pending`` marker.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import re
import shutil
import time
import typing
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.hud_server import RECONNECT_GRACE_MS
from local_astra.contracts import events as E
from local_astra.contracts.protocol import AstraSessionCleared, AstraToolCall
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HUD_DIR, HudStaticServer
from tests.integration.test_hud_headless import CHROME, Chrome, requires_chrome
from tests.integration.test_hud_rehearsal import (
    PAGE_TIMEOUT_S,
    Live,
    Page,
    answer,
    page_url,
    thought,
    ui_commit,
)
from tests.integration.test_hud_restart import cleared_on_wire, restart, state, turn_ids

pytestmark = [pytest.mark.integration]

#: C5 ruling Q1 (orchestrator, 2026-10-27, provisional): the mount ceiling. Only the never-loading
#: panel row reads it, as the earliest moment that page may open its socket.
MOUNT_CEILING_MS = 3000
#: Upper-bound slack for that row (review, 2026-10-27): ``at_ms`` counts from navigation start, and
#: the ceiling timer is armed only once main.js has loaded, so the open lands at the ceiling plus
#: the page's load time (~0.3 s idle). A 15000 ms ceiling is far outside it.
CEILING_SLACK_MS = 2000

#: C9(g): PANEL_SPECS keeps its content and order; the seven panel entry modules C7(b) watches.
PANEL_SPECS = (
    "./ui/index.js",
    "./denoise/index.js",
    "./assists/index.js",
    "./tools/index.js",
    "./ask/index.js",
    "./log/index.js",
    "./session/index.js",
)

#: Distinctive strings, so "no pre-reset text" is a substring search.
Q1 = "please list the walrus colonies of svalbard"
Q2 = "what is the capital of the moon base"
ANSWER = "Walruses gather on the ice floes. "
SECRET_THOUGHT = "SECRETTHOUGHT startup rows never show this"


def script() -> list[Any]:
    return [thought(SECRET_THOUGHT), answer(ANSWER), ui_commit()]


def tool_call(turn_id: str, call_id: str, status: str) -> AstraToolCall:
    return AstraToolCall(
        turn_id=turn_id,
        call_id=call_id,
        name="scroll",
        arguments={"direction": "down"},
        status=status,  # type: ignore[arg-type]
        source="draft",
        action="scrolled down" if status == "executed" else None,
    )


# --- the browser side ---------------------------------------------------------------------------

INIT_JS = """
(() => {
  const rec = (window.__t0165 = { constructions: [], connected: null });
  const Real = window.WebSocket;
  window.WebSocket = new Proxy(Real, {
    construct(target, args, newTarget) {
      const root = document.documentElement;
      const ask = document.querySelector('[data-ask-input]');
      rec.constructions.push({
        panels: root && root.dataset.panels !== undefined ? root.dataset.panels : null,
        tools_list: Boolean(document.querySelector('[data-tools-root]')),
        ask_input: Boolean(ask),
        ask_disabled: ask ? ask.disabled : null,
        restart: Boolean(document.querySelector('[data-restart-button]')),
        at_ms: performance.now(),
      });
      return Reflect.construct(target, args, newTarget);
    },
  });
  const watch = () => {
    if (rec.connected) return;
    const status = document.getElementById('conn-status');
    if (!status || status.textContent !== 'connected') return;
    const ask = document.querySelector('[data-ask-input]');
    rec.connected = {
      ask_input: Boolean(ask),
      ask_disabled: ask ? ask.disabled : null,
      at_ms: performance.now(),
    };
  };
  new MutationObserver(watch).observe(document, { childList: true, subtree: true, characterData: true });
})();
"""

#: The live-page premises publish only once the pane is listening: before that, a live message is
#: missed by design of the subscription seam, which is not what these rows test.
TOOLS_MOUNTED_JS = "Boolean(document.querySelector('[data-tools-root]'))"

#: The Tools pane's calls in DOM order, each with the status the pane shows.
TOOLS_JS = """
[...document.querySelectorAll('[data-tools-call]')].map((d) => {
  const s = d.querySelector('[data-tools-status]');
  return [d.getAttribute('data-tools-call'), s ? s.getAttribute('data-tools-status') : null];
})"""


class CdpChrome(Chrome):
    """``Chrome`` that keeps every CDP event it reads (the ``_RecordingChrome`` pattern of
    ``test_hud_headless``, extended to all methods), plus the two calls this module adds."""

    def __init__(self, binary: str) -> None:
        super().__init__(binary)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _note(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if isinstance(method, str):
            self.events.append((method, dict(msg.get("params") or {})))

    def _cmd(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            self._note(msg)
            if msg.get("id") == self._id:
                assert "error" not in msg, f"{method}: {msg['error']}"
                return dict(msg["result"])

    def drain(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self._note(json.loads(self.ws.recv(timeout=0.2)))
            except TimeoutError:
                continue
            except Exception:
                return

    def add_init_script(self, source: str) -> None:
        self._cmd("Page.enable")
        self._cmd("Page.addScriptToEvaluateOnNewDocument", source=source)

    def reload(self) -> None:
        self._cmd("Page.reload", ignoreCache=True)


@contextlib.asynccontextmanager
async def tab(
    url: str, *, init_js: str | None = INIT_JS, network: bool = False
) -> typing.AsyncIterator[tuple[CdpChrome, Page]]:
    """One headless tab on ``url`` (not waited on: each row says what it waits for)."""
    assert CHROME is not None
    chrome = CdpChrome(CHROME).__enter__()
    try:
        if network:
            await asyncio.to_thread(chrome._cmd, "Network.enable")
        if init_js is not None:
            await asyncio.to_thread(chrome.add_init_script, init_js)
        page = Page(chrome)
        await page.navigate(url)
        yield chrome, page
    finally:
        with contextlib.suppress(Exception):
            chrome.__exit__()


async def recorded(page: Page) -> dict[str, Any]:
    value = await page.evaluate("window.__t0165")
    assert isinstance(value, dict), f"INIT_JS did not run before the page's scripts: {value!r}"
    return value


async def tools(page: Page) -> list[list[Any]]:
    value = await page.evaluate(TOOLS_JS)
    assert isinstance(value, list), f"the Tools probe threw: {value!r}"
    return value


async def settle_tools(page: Page, expected: list[list[str]], timeout_s: float) -> list[list[Any]]:
    """Poll the Tools pane until it reads ``expected`` or the ceiling passes; return the last read."""
    deadline = time.monotonic() + timeout_s
    last: list[list[Any]] = []
    while time.monotonic() < deadline:
        last = await tools(page)
        if last == expected:
            return last
        await asyncio.sleep(0.1)
    return last


@contextlib.asynccontextmanager
async def started(tmp_path: Path, root: Path = HUD_DIR) -> typing.AsyncIterator[tuple[Live, str]]:
    """A started ``Live`` and the page URL, served by the production ``HudStaticServer``."""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    static = HudStaticServer(root, 0, poll_interval=0.05)
    try:
        yield live, page_url(static.port, live.server.url)
    finally:
        static.close()
        await live.aclose()


async def hold_an_open_turn(live: Live, question: str) -> tuple[asyncio.Task[None], str]:
    """Ask with the LLM held, so the turn stays open; returns the asking task and the turn id."""
    known = set(turn_ids(live))
    live.llm.hold()
    asking = live.asking(question)
    await live.until(
        lambda: (
            live.session.state.state is TurnState.THINKING
            and any(t not in known for t in turn_ids(live))
        ),
        f"an open turn for {question!r} (the LLM held)",
    )
    return asking, [t for t in turn_ids(live) if t not in known][-1]


async def release(live: Live, asking: asyncio.Task[None]) -> None:
    live.llm.release()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(asking, timeout=10)


# --- C1 and C2 ----------------------------------------------------------------------------------


@requires_chrome
async def test_the_socket_opens_after_every_panel_mounted(tmp_path):
    """C1: the first ``WebSocket`` is constructed only after ``mountPanels`` settled.

    At that moment ``<html data-panels="mounted">`` and the Tools pane list, ``[data-ask-input]``
    and the Restart button are in the DOM. ``[data-ask-input]`` is markup in ``index.html`` (so its
    presence alone says little); the ask panel enables it at mount from IDLE alone (implementation
    notes), so it is also recorded as enabled. C9(b): one ``astra.hello``, one socket.
    """
    async with started(tmp_path) as (live, url):
        async with tab(url) as (_chrome, page):
            await page.connected()
            await asyncio.sleep(0.5)
            rec = await recorded(page)
        constructions = rec["constructions"]
        assert constructions, f"C1: the page never constructed a WebSocket: {rec}"
        first = constructions[0]
        assert first["panels"] == "mounted", (
            "C1: the page opens its socket only once every installed panel has mounted, and marks "
            f"that on <html data-panels>; at the first WebSocket construction it read {first}"
        )
        missing = [k for k in ("tools_list", "ask_input", "restart") if not first[k]]
        assert not missing, f"C1: absent at the first WebSocket construction: {missing}; {first}"
        assert first["ask_disabled"] is False, (
            f"C1/C2: the ask panel had mounted (IDLE enables the box) before the socket; {first}"
        )
        assert len(constructions) == 1 and live.hellos == 1, (
            f"C9(b): one socket, one astra.hello; constructions={constructions}, hellos={live.hellos}"
        )


@requires_chrome
async def test_the_ask_box_is_enabled_when_connected_reads(tmp_path):
    """C2 (T-0039 criterion 13): at the first moment ``#conn-status`` reads ``connected`` on an IDLE
    session, ``[data-ask-input]`` is enabled. A ``MutationObserver`` records it; polling would let
    the box catch up before the read."""
    async with started(tmp_path) as (live, url):
        assert live.session.state.state is TurnState.IDLE, "premise: the session is IDLE"
        async with tab(url) as (_chrome, page):
            await page.connected()
            rec = await recorded(page)
        seen = rec["connected"]
        assert seen is not None, f"the observer never saw 'connected': {rec}"
        assert seen["ask_input"], f"premise: [data-ask-input] is in the page; {seen}"
        assert seen["ask_disabled"] is False, (
            "C2: the page read 'connected' while the ask box was still disabled (the ask panel "
            f"had not mounted yet); {seen}"
        )


# --- C3: the Tools pane replays the journal -------------------------------------------------------

#: c-1 proposed, c-2 proposed, then c-1 executed: first-seen order c-1, c-2; latest statuses.
EXPECTED_CALLS = [["c-1", "executed"], ["c-2", "proposed"]]


def publish_two_calls(live: Live, turn_id: str) -> None:
    live.server.publish(tool_call(turn_id, "c-1", "proposed"))
    live.server.publish(tool_call(turn_id, "c-2", "proposed"))
    live.server.publish(tool_call(turn_id, "c-1", "executed"))


@requires_chrome
async def test_reload_replays_the_tools_pane_journal(tmp_path):
    """C3 (docs/02 §5 T-0090 U10): after ``Page.reload`` during an open turn, the Tools pane lists
    the latest status of every journalled call, in first-seen order.

    **Depends on ``RECONNECT_GRACE_MS`` (4000 ms since T-0166):** the journal is cleared once the
    last greeted page has been gone that long, so a reload slower than the grace replays nothing. The failure message
    prints the measured reload-to-hello time and whether the grace expired.
    """
    async with started(tmp_path) as (live, url):
        asking, turn = await hold_an_open_turn(live, Q1)
        try:
            async with tab(url) as (chrome, page):
                await page.connected()
                await page.until(TOOLS_MOUNTED_JS, "premise: the Tools pane is mounted")
                publish_two_calls(live, turn)
                shown = await settle_tools(page, EXPECTED_CALLS, PAGE_TIMEOUT_S)
                assert shown == EXPECTED_CALLS, f"premise: the live page lists both calls; {shown}"

                hellos = live.hellos
                t0 = time.monotonic()
                await asyncio.to_thread(chrome.reload)
                await live.until(lambda: live.hellos > hellos, "the reloaded page said hello")
                reload_to_hello_ms = (time.monotonic() - t0) * 1000
                # the regime record (docs/tasks/T-0165.md Regime): visible with `pytest -s`
                print(f"T-0165 reload-to-hello {reload_to_hello_ms:.0f} ms")
                await page.connected()
                after = await settle_tools(page, EXPECTED_CALLS, 5.0)
            departed = live.machine.events(E.ClientDisconnected)
            assert after == EXPECTED_CALLS, (
                "C3: after a reload the Tools pane replays the open turn's calls, latest status, "
                f"first-seen order; expected {EXPECTED_CALLS}, the pane lists {after}. "
                f"reload-to-hello {reload_to_hello_ms:.0f} ms (RECONNECT_GRACE_MS "
                f"{RECONNECT_GRACE_MS}); grace expired: {bool(departed)}"
            )
        finally:
            await release(live, asking)


@requires_chrome
async def test_a_second_tab_replays_the_tools_pane_journal(tmp_path):
    """C3, second-tab variant: no grace dependence (the first tab stays connected)."""
    async with started(tmp_path) as (live, url):
        asking, turn = await hold_an_open_turn(live, Q1)
        try:
            async with tab(url) as (_c1, first):
                await first.connected()
                await first.until(
                    TOOLS_MOUNTED_JS, "premise: the first tab's Tools pane is mounted"
                )
                publish_two_calls(live, turn)
                shown = await settle_tools(first, EXPECTED_CALLS, PAGE_TIMEOUT_S)
                assert shown == EXPECTED_CALLS, f"premise: the first tab lists both calls; {shown}"
                async with tab(url) as (_c2, second):
                    await second.connected()
                    listed = await settle_tools(second, EXPECTED_CALLS, 5.0)
            assert listed == EXPECTED_CALLS, (
                "C3: a second tab opened during the open turn lists the journalled calls; "
                f"expected {EXPECTED_CALLS}, it lists {listed}"
            )
        finally:
            await release(live, asking)


# --- C4: the reset replay order -------------------------------------------------------------------


@requires_chrome
async def test_reload_after_restart_shows_only_the_new_session(tmp_path):
    """C4, adversarial: the reset replay and the journal reaching a panel in the wrong order, or a
    panel that misses the cleared echo.

    Turn 1 (spoken, with a card) gets two tool calls; Restart -> confirm; a new open turn gets one
    call; reload. The reloaded page's hello reply begins with ``astra.session.cleared`` replayed
    (T-0162 A-5). The Tools pane lists only the post-reset call, the thread holds no pre-reset text
    and no card. The reload is a new page whose first message ever is that replayed reset, so
    ``onNewReset`` keeps its ``?source=canvas`` capture (D-3's exception).
    """
    async with started(tmp_path) as (live, url):
        asking: asyncio.Task[None] | None = None
        try:
            async with tab(url) as (chrome, page):
                await page.connected()
                await page.until(TOOLS_MOUNTED_JS, "premise: the Tools pane is mounted")
                await live.ask(Q1)
                await live.idle()
                (old,) = turn_ids(live)
                live.server.publish(tool_call(old, "c-old-1", "executed"))
                live.server.publish(tool_call(old, "c-old-2", "proposed"))
                shown = await settle_tools(
                    page, [["c-old-1", "executed"], ["c-old-2", "proposed"]], PAGE_TIMEOUT_S
                )
                assert len(shown) == 2, f"premise: two pre-reset calls on the page; {shown}"

                await restart(page, live)
                await live.until(lambda: len(cleared_on_wire(live)) == 1, "astra.session.cleared")
                asking, new = await hold_an_open_turn(live, Q2)
                live.server.publish(tool_call(new, "c-post", "executed"))
                shown = await settle_tools(page, [["c-post", "executed"]], PAGE_TIMEOUT_S)
                assert shown == [["c-post", "executed"]], f"premise: the live page; {shown}"

                hellos = live.hellos
                t0 = time.monotonic()
                await asyncio.to_thread(chrome.reload)
                await live.until(lambda: live.hellos > hellos, "the reloaded page said hello")
                reload_to_hello_ms = (time.monotonic() - t0) * 1000
                # the regime record (docs/tasks/T-0165.md Regime): visible with `pytest -s`
                print(f"T-0165 reload-to-hello {reload_to_hello_ms:.0f} ms")
                await page.connected()
                await page.until(
                    "Number(document.querySelector('[data-frames-sent]')"
                    ".getAttribute('data-frames-sent')) > 0",
                    "the reloaded page's capture sends frames",
                )
                listed = await settle_tools(page, [["c-post", "executed"]], 5.0)
                await asyncio.sleep(1.5)  # a wrong stopCapture or a late clear lands by now
                listed = await tools(page)
                after = await state(page)
            head = live.hello_replies[-1][0] if live.hello_replies[-1] else None
        finally:
            if asking is not None:
                await release(live, asking)

        assert isinstance(head, AstraSessionCleared) and head.replay, (
            f"premise (A-5): the reload's hello reply begins with the reset, replayed; {head!r}"
        )
        assert listed == [["c-post", "executed"]], (
            "C4: after Restart, a new open turn and a reload, the Tools pane lists the post-reset "
            f"call and nothing from before the reset; it lists {listed}. reload-to-hello "
            f"{reload_to_hello_ms:.0f} ms (RECONNECT_GRACE_MS {RECONNECT_GRACE_MS})"
        )
        thread = after["thread_text"]
        assert Q1 not in thread and ANSWER.strip() not in thread, (
            f"C4: the thread holds pre-reset text after the reload: {thread!r}"
        )
        assert after["card_wraps"] == 0, f"C4: a pre-reset card is back; {after}"
        assert Q2 in thread, f"C4: the open post-reset turn is shown (anti-vacuity): {thread!r}"
        assert after["banner_visible"] and not after["mute_disabled"], (
            "C4 / D-3: a reloaded page whose first message is the replayed reset is a page opened "
            f"after the reset; its capture is kept; {after}"
        )


# --- C7: load work, counted --------------------------------------------------------------------------

_STATIC_IMPORT = re.compile(
    r"^[ \t]*(?:import|export)\b(?:[^'\";]*?\bfrom)?\s*[\"']([^\"']+)[\"']", re.M | re.S
)


def static_imports(path: Path) -> list[Path]:
    """The relative static ``import … from "…"`` / ``import "…"`` / ``export … from "…"`` targets
    of one module. Statement-anchored at a line start, so ``//`` and ``*`` comment lines and the
    dynamic ``import(…)`` are not matched (prose that begins a line with ``import`` would be)."""
    out = []
    for spec in _STATIC_IMPORT.findall(path.read_text(encoding="utf-8")):
        if spec.startswith("."):
            out.append((path.parent / spec).resolve())
    return out


def graph_depth(path: Path, memo: dict[Path, int], stack: frozenset[Path] = frozenset()) -> int:
    """Longest chain of static imports from ``path``, counting ``path`` itself (a leaf is 1)."""
    path = path.resolve()
    if path in memo:
        return memo[path]
    assert path not in stack, f"a static import cycle through {path}"
    children = [c for c in static_imports(path) if c.is_file()]
    depth = 1 + max((graph_depth(c, memo, stack | {path}) for c in children), default=0)
    memo[path] = depth
    return depth


def depth_bound() -> tuple[int, dict[str, int]]:
    """C7(c)'s K from the source tree: main.js's graph + 1 (panels.js) + the deepest panel + 1."""
    src = HUD_DIR / "src"
    memo: dict[Path, int] = {}
    main = graph_depth(src / "main.js", memo)
    panels = {spec: graph_depth(src / spec[2:], memo) for spec in PANEL_SPECS}
    return main + 1 + max(panels.values()) + 1, {"main.js": main, **panels}


def requests_of(events: list[tuple[str, dict[str, Any]]], origin: str) -> list[dict[str, Any]]:
    """One record per request under ``origin``: url, requestWillBeSent and loadingFinished times."""
    by_id: dict[str, dict[str, Any]] = {}
    for method, params in events:
        rid = params.get("requestId")
        if method == "Network.requestWillBeSent":
            url = (params.get("request") or {}).get("url", "")
            if url.startswith(origin):
                by_id.setdefault(rid, {"url": url, "start": params["timestamp"], "end": None})
        elif method == "Network.loadingFinished" and rid in by_id:
            by_id[rid]["end"] = params["timestamp"]
    return list(by_id.values())


def serial_depth(reqs: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """Longest chain in which each request starts after its predecessor's ``loadingFinished``."""
    ordered = sorted(reqs, key=lambda r: r["start"])
    best: dict[int, tuple[int, list[str]]] = {}
    for i, r in enumerate(ordered):
        chain: tuple[int, list[str]] = (1, [r["url"]])
        for j in range(i):
            p = ordered[j]
            if p["end"] is not None and p["end"] <= r["start"] and best[j][0] + 1 > chain[0]:
                chain = (best[j][0] + 1, [*best[j][1], r["url"]])
        best[i] = chain
    return max(best.values(), default=(0, []))


@requires_chrome
async def test_the_page_loads_in_one_wave_of_panels(tmp_path):
    """C7(a-c), from CDP ``Network`` events over one load from ``HudStaticServer``, n=3 loads:

    (a) every URL under the page's origin is requested exactly once;
    (b) the seven panel entry modules are requested in one wave: every ``requestWillBeSent`` comes
        before the ``loadingFinished`` of all seven;
    (c) the serial depth of the document + module requests is at most K, computed from the static
        imports under ``hud/src/`` (so it moves with the code).

    Counts and orderings only: they must hold at any load average.
    """
    k, depths = depth_bound()
    async with started(tmp_path) as (_live, url):
        origin = url.split("?", 1)[0]
        loads: list[dict[str, Any]] = []
        for n in range(3):
            async with tab(url, init_js=None, network=True) as (chrome, page):
                version = await asyncio.to_thread(chrome._cmd, "Browser.getVersion")
                await page.connected()
                await page.until(
                    "Boolean(document.querySelector('[data-restart-button]'))"
                    " && Boolean(document.querySelector('[data-tools-root]'))",
                    "the last panels mounted",
                )
                await asyncio.to_thread(chrome.drain, 0.5)
                reqs = requests_of(chrome.events, origin)
            regime = f"load {n + 1}/3, {version.get('product')}, K={k} from {depths}"

            problems: list[str] = []
            counts = collections.Counter(r["url"] for r in reqs)
            dupes = {u: c for u, c in counts.items() if c > 1}
            if dupes:
                problems.append(f"C7(a): a URL fetched more than once: {dupes}")

            panel_urls = [origin + "src/" + spec[2:] for spec in PANEL_SPECS]
            panel_reqs = [r for r in reqs if r["url"] in panel_urls]
            assert sorted(r["url"] for r in panel_reqs) == sorted(panel_urls), (
                f"C7(b) premise: each panel entry requested once; {[r['url'] for r in panel_reqs]}"
            )
            assert all(r["end"] is not None for r in panel_reqs), f"C7(b): unfinished; {panel_reqs}"
            first_end = min(r["end"] for r in panel_reqs)
            late = [r["url"].rsplit("/src/", 1)[1] for r in panel_reqs if r["start"] >= first_end]
            if late:
                problems.append(
                    "C7(b): the seven panel entry modules are requested in one wave; these were "
                    f"requested only after another panel's module had finished loading: {late}"
                )

            modules = [
                r for r in reqs if r["url"].split("?", 1)[0].endswith(".js") or r["url"] == url
            ]
            depth, chain = serial_depth(modules)
            if depth > k:
                problems.append(
                    f"C7(c): serial depth {depth} > K={k}; the chain: "
                    f"{[c.replace(origin, '/') for c in chain]}"
                )
            loads.append(
                {"load": n + 1, "requests": len(reqs), "depth": depth, "problems": problems}
            )
            assert not problems, f"{regime}: " + " | ".join(problems)
        assert len(loads) == 3, loads


# --- C5 end to end: a panel whose import never settles ----------------------------------------------


@requires_chrome
async def test_a_panel_that_never_loads_still_lets_the_page_connect(tmp_path):
    """C5 (T-0008 criterion 10 extended to a loader that never settles), through the page.

    A copy of ``hud/`` in which the *last* panel (``./session/index.js``) statically imports a
    module whose top-level ``await`` never settles, so its ``import()`` never settles either. The
    last one, because mounts stay in PANEL_SPECS order (C6): every panel before it can mount. The
    page must still connect, after the ceiling and not before (``MOUNT_CEILING_MS``, ruling Q1),
    with ``<html data-panels>`` set to something other than ``"mounted"`` (implementation notes),
    the other panels mounted, and exactly one hello.
    """
    root = tmp_path / "hud-with-a-hanging-panel"
    shutil.copytree(HUD_DIR, root)
    (root / "src" / "never.js").write_text(
        "// T-0165 C5: top-level await that never settles\nawait new Promise(() => {});\nexport {};\n"
    )
    session_panel = root / "src" / "session" / "index.js"
    session_panel.write_text('import "../never.js";\n' + session_panel.read_text())

    async with started(tmp_path, root) as (live, url):
        async with tab(url) as (_chrome, page):
            await page.until(
                "document.querySelector('#conn-status').textContent === 'connected'",
                "#conn-status === 'connected' with one panel import hanging",
                timeout_s=MOUNT_CEILING_MS / 1000 + 20,
            )
            await asyncio.sleep(1.0)
            rec = await recorded(page)
            dom = await page.evaluate(
                "({ tools: Boolean(document.querySelector('[data-tools-root]')),"
                " log: Boolean(document.querySelector('[data-log-toggle]')),"
                " ask_disabled: (document.querySelector('[data-ask-input]') || {}).disabled,"
                " restart: Boolean(document.querySelector('[data-restart-button]')),"
                " panels: document.documentElement.dataset.panels ?? null })"
            )
        assert not dom["restart"], f"premise: the hanging panel never mounted; {dom}"
        constructions = rec["constructions"]
        assert len(constructions) == 1 and live.hellos == 1, (
            f"C5/C9(b): one socket and exactly one hello; constructions={constructions}, "
            f"hellos={live.hellos}"
        )
        first = constructions[0]
        assert isinstance(first["panels"], str) and first["panels"] not in ("", "mounted"), (
            "C5: the socket opened on the ceiling path, which marks <html data-panels> with "
            f"something other than 'mounted' (so C1 cannot pass vacuously); {first}"
        )
        assert first["at_ms"] >= MOUNT_CEILING_MS, (
            f"C5: open is never called before mounting settles or the {MOUNT_CEILING_MS} ms "
            f"ceiling elapses; the socket was constructed at {first['at_ms']:.0f} ms after "
            "navigation start"
        )
        assert first["at_ms"] <= MOUNT_CEILING_MS + CEILING_SLACK_MS, (
            f"C5/Q1: the ceiling path opens at about {MOUNT_CEILING_MS} ms, not later (a page with "
            f"a hung import must not stay dead for longer); the socket was constructed at "
            f"{first['at_ms']:.0f} ms after navigation start (bound {MOUNT_CEILING_MS} + "
            f"{CEILING_SLACK_MS} ms of page load before the timer is armed)"
        )
        assert dom["tools"] and dom["log"] and dom["ask_disabled"] is False, (
            f"C5/C6: every panel before the hanging one is mounted; {dom}"
        )
