"""T-0169 criteria 6 and 7: a live session built by ``cli`` opens ``<origin>/web.html`` in web mode and the catalogue
``<origin>/`` with ``--no-web``, in a real headless Chromium, and Restart keeps each there (docs/tasks/T-0169.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser-web``: web mode needs
Chromium's sandbox, H0). Without it every row is a named skip; with it a missing Playwright is a failure.

**The rig.** ``astra session --tools live --browser headless --silent --no-mic --no-log --turns <tmp>`` through
``cli.main``, with the LLM, STT, TTS and memory replaced by ``tests/fakes`` (T-0092b's ``fake_clients``, as the
``make browser-live`` rows' sessions are), repair off (payloads pointed nowhere: no warm-up). The real
``StartPageServer`` and the real ``BrowserToolExecutor`` are built by ``cli._start_live``; each is a thin subclass that
only records (the executor: every ``frame()`` with the page URL at capture, and ``reset_page()`` calls; in web mode it
is handed an injected proxy resolver and connector that refuse and record everything, so nothing can leave the box).
The HUD server is a stub whose ``start()`` runs the row's scenario on the session's loop, then sends SIGINT, which is
how ``cli`` ends a session; a scenario failure is re-raised after ``cli.main`` returns.

**Turn 1.** A typed question (``UserText``) answered by one scripted answer commit; the request's image must be the
bytes of a frame the executor captured while the page was ``<origin>/web.html`` (live prompts carry the captured JPEG
as is, ``test_session_reset.py``). With the T-0152 default the turn line's ``overview_frame`` counts are
``{"shown": 0, "fits": 1, "too_tall": 0, "failed": 0, "refused": 0}``: ``/web.html`` fits on one 1280x800 screen
(height at most 1.05 x 800 = 840 CSS px, criterion 6).

``test_live_web_restart_stays_on_web_html`` and ``test_live_no_web_session_starts_on_catalogue`` are green at b372edaa
(Restart already used ``/web.html`` in web mode; loopback is unchanged), so they are active, not pending.

Regime: Playwright 1.64.0, Chromium 1248 headless at 1280x800, DSF 1, sandbox on (H0), the start-page server on
127.0.0.1 at an ephemeral port, FakeLlm with one answer, wall clock (``SessionClock``), n = 1 per row, no network.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import signal
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.computer import executor as executor_mod
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, ImagePart
from local_astra.pages import server as server_mod
from tests.fakes import FakeLlm

# T-0057: ``cli.main --silent`` builds a real ``PipeWireAudioIO`` whose first ``play()`` starts ``astra-audio-writer``;
# importing the tracker lets the conftest's per-row stopper find and stop it (the leak guard itself is unchanged)
from tests.fakes import audio_device as _audio_tracker  # noqa: F401
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli

GATE_ENV = "ASTRA_BROWSER_TESTS"
GATED = os.environ.get(GATE_ENV) == "1"
pytestmark = [pytest.mark.integration, pytest.mark.browser]

W, H = 1280, 800
#: criterion 6: "fits on one 1280x800 screen" is T-0152's lower bound, 1.05 screens
FITS_PX = 1.05 * H
FITS = {"shown": 0, "fits": 1, "too_tall": 0, "failed": 0, "refused": 0}
SCENARIO_S = 120.0
ANSWER = "It is the page with the web address field. "

REAL_EXECUTOR = executor_mod.BrowserToolExecutor
REAL_PAGES = server_mod.StartPageServer


@pytest.fixture(autouse=True)
def _browser_layer() -> None:
    if not GATED:
        pytest.skip(
            f"{GATE_ENV} is not 1: the T-0169 rows drive a real Chromium (docs/tasks/T-0169.md)"
        )
    try:
        importlib.import_module("playwright.async_api")
    except ImportError as exc:
        pytest.fail(f"{GATE_ENV}=1 but playwright does not import ({exc}): `uv sync --all-extras`")


# =====================================================================================================
# the rig
# =====================================================================================================


class DenyNet:
    """The web-mode proxy's seams: every resolve fails and every connect is refused, each recorded."""

    def __init__(self) -> None:
        self.resolves: list[str] = []
        self.connects: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        self.resolves.append(host)
        raise socket.gaierror(socket.EAI_NONAME, "T-0169 rig: no network")

    async def connect(self, ip: str, port: int) -> Any:
        self.connects.append((ip, port))
        raise ConnectionRefusedError("T-0169 rig: no network")


class Live:
    """What one ``cli`` run built and saw."""

    def __init__(self) -> None:
        self.executors: list[Any] = []
        self.pages: list[Any] = []
        self.net = DenyNet()
        self.llm = FakeLlm([Commit(request_id="?", canvas_idx=0, segment="answer", text=ANSWER)])
        self.errors: list[BaseException] = []
        self.out: dict[str, Any] = {}
        self.turns = Path()

    @property
    def ex(self) -> Any:
        assert len(self.executors) == 1, f"premise: cli built one executor; {len(self.executors)}"
        return self.executors[0]

    @property
    def origin(self) -> str:
        assert len(self.pages) == 1, f"premise: cli built one start-page server; {len(self.pages)}"
        return str(self.pages[0].origin)


Scenario = Callable[[Any, Live], Awaitable[None]]


def run_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: Scenario, *extra: str
) -> Live:
    live = Live()
    live.turns = tmp_path / "turns.jsonl"
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    monkeypatch.setattr(cli, "VllmClient", lambda *a, **k: live.llm)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json")

    class RecordingExecutor(REAL_EXECUTOR):  # type: ignore[misc, valid-type]
        def __init__(self, **kw: Any) -> None:
            if kw.get("web"):
                kw.setdefault("proxy_resolve", live.net.resolve)
                kw.setdefault("proxy_connect", live.net.connect)
            super().__init__(**kw)
            self.seen: list[tuple[str, Any]] = []
            self.resets = 0
            live.executors.append(self)

        async def frame(self) -> Any:
            f = await super().frame()
            self.seen.append((str(self.driver.page.url), f))
            return f

        async def reset_page(self) -> None:
            await super().reset_page()
            self.resets += 1

    class RecordingPages(REAL_PAGES):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            live.pages.append(self)

    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", RecordingExecutor)
    monkeypatch.setattr(server_mod, "StartPageServer", RecordingPages)

    class ScenarioHud:
        def __init__(self, session: Any, **kwargs: Any) -> None:
            self.session, self.port, self.url = session, 0, "ws://127.0.0.1:0/v1/realtime"

        async def start(self) -> None:
            asyncio.get_running_loop().create_task(self._run())

        async def _run(self) -> None:
            try:
                await asyncio.wait_for(scenario(self.session, live), SCENARIO_S)
            except BaseException as exc:  # re-raised after cli returns
                live.errors.append(exc)
            finally:
                os.kill(os.getpid(), signal.SIGINT)

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(cli, "HudServer", ScenarioHud)
    config = T.write_config(tmp_path)
    args = T.session_argv(
        config,
        tmp_path,
        "--tools",
        "live",
        "--browser",
        "headless",
        "--turns",
        str(live.turns),
        *extra,
    )
    code = run_cli(args)
    if live.errors:
        raise live.errors[0]
    assert code == 0, f"the session exited {code}"
    assert live.out.get("done") is True, "the scenario never ran to its end"
    return live


async def wait_for(what: str, ok: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not ok():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def wait_url(page: Any, want: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while page.url != want and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return str(page.url)


async def page_facts(page: Any) -> dict[str, Any]:
    facts = await page.evaluate(
        "() => ({ state: window.__state ? window.__state.page : null,"
        " height: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0),"
        " inner: [window.innerWidth, window.innerHeight] })"
    )
    return {"url": str(page.url), **facts}


def turn_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


async def reset(session: Any, ex: Any) -> None:
    """The HUD's Restart (``astra.session.reset`` → ``SessionReset``), then wait for ``reset_page`` to have run."""
    before = ex.resets
    session.submit(E.SessionReset(t_ms=session.clock.now_ms))
    await wait_for("reset_page after the Restart", lambda: ex.resets > before)


# =====================================================================================================
# criterion 6: web mode
# =====================================================================================================


def test_live_web_session_starts_on_web_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6: after start the agent page is ``<origin>/web.html`` with ``__state.page == "web"``; it fits one 1280x800
    screen; turn 1's image is that page's own frame and its overview-frame outcome is ``fits``; nothing left
    loopback (no egress attempt logged, nothing resolved or connected by the proxy)."""

    async def scenario(session: Any, live: Live) -> None:
        ex, web_url = live.ex, live.origin + "/web.html"
        assert ex.web is True, "premise: a default live start is in web mode (T-0160)"
        start = await page_facts(ex.driver.page)
        live.out["start"] = start
        assert start["url"] == web_url, (
            f"T-0169 C6: the session started on {start['url']}, not {web_url}"
        )
        assert start["state"] == "web", f"T-0169 C6: window.__state.page is {start['state']!r}"
        assert start["inner"] == [W, H], f"premise: a 1280x800 viewport; {start['inner']}"
        assert start["height"] <= FITS_PX, f"T-0169 C6: /web.html is {start['height']} px tall"
        session.submit(E.UserText(t_ms=session.clock.now_ms, text="what is on the screen"))
        await wait_for("turn 1's line", lambda: len(turn_lines(live.turns)) >= 1, 60)
        assert live.llm.generates, "premise: turn 1 sent a request"
        prompt = live.llm.generates[0][1]
        users = [m for m in prompt.messages if m.role == "user"]
        images = [p.url for p in users[-1].content if isinstance(p, ImagePart)] if users else []
        assert len(images) == 1, f"T-0169 C6: turn 1 carries one image; {len(images)}"
        on_web = [base64.b64encode(f.jpeg).decode() for url, f in ex.seen if url == web_url]
        assert any(b in images[0] for b in on_web), (
            f"T-0169 C6: turn 1's image is not a frame captured on {web_url}; captured on "
            f"{[url for url, _ in ex.seen]}"
        )
        line = turn_lines(live.turns)[0]
        assert line.get("extras", {}).get("overview_frame") == FITS, (
            f"T-0169 C6 / T-0152: step 1's overview-frame outcome; {line.get('extras', {}).get('overview_frame')}"
        )
        assert ex.driver.page.url == web_url, f"turn 1 took no action; now on {ex.driver.page.url}"
        assert ex.egress_log == (), f"T-0169 C6: an egress attempt was logged: {ex.egress_log}"
        assert live.net.resolves == [] and live.net.connects == [], (
            f"T-0169 C6: the proxy resolved {live.net.resolves} / connected {live.net.connects}"
        )
        live.out["done"] = True

    run_live(tmp_path, monkeypatch, scenario)


def test_live_web_restart_stays_on_web_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6: in web mode, after the page went to the catalogue ``<origin>/``, Restart brings it back to
    ``<origin>/web.html`` (``__state.page == "web"``); nothing left loopback. Green at b372edaa (reset_page already
    used ``/web.html`` in web mode): a regression row."""

    async def scenario(session: Any, live: Live) -> None:
        ex, web_url = live.ex, live.origin + "/web.html"
        assert ex.web is True, "premise: a default live start is in web mode (T-0160)"
        page = ex.driver.page
        await page.goto(live.origin + "/", wait_until="domcontentloaded")
        assert await wait_url(page, live.origin + "/") == live.origin + "/", (
            "premise: on the catalogue"
        )
        await reset(session, ex)
        got = await wait_url(page, web_url)
        assert got == web_url, f"T-0169 C6: after Restart the page is {got}, not {web_url}"
        assert (await page_facts(page))["state"] == "web"
        assert ex.egress_log == () and live.net.resolves == [] and live.net.connects == []
        live.out["done"] = True

    run_live(tmp_path, monkeypatch, scenario)


# =====================================================================================================
# criterion 7: loopback mode (negative)
# =====================================================================================================


def test_live_no_web_session_starts_on_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C7 (negative, green at b372edaa): with ``--no-web`` the session starts on the catalogue ``<origin>/``
    (``__state.page == "start"``) and, after the page went to ``<origin>/search.html``, Restart returns there."""

    async def scenario(session: Any, live: Live) -> None:
        ex, catalogue = live.ex, live.origin + "/"
        assert not ex.web, "premise: --no-web builds a loopback executor"
        page = ex.driver.page
        start = await page_facts(page)
        assert start["url"] == catalogue, f"T-0169 C7: started on {start['url']}, not {catalogue}"
        assert start["state"] == "start", start
        await page.goto(live.origin + "/search.html", wait_until="domcontentloaded")
        assert await wait_url(page, live.origin + "/search.html") == live.origin + "/search.html"
        await reset(session, ex)
        got = await wait_url(page, catalogue)
        assert got == catalogue, f"T-0169 C7: after Restart the page is {got}, not {catalogue}"
        assert ex.egress_log == (), ex.egress_log
        live.out["done"] = True

    run_live(tmp_path, monkeypatch, scenario, "--no-web")
