"""T-0160 group B, launch rows (B-1, B-2, B-5, B-6, B-8): ``PlaywrightDriver`` in web mode over a fake Playwright, with
the real ``EgressProxy`` recorded (docs/tasks/T-0160.md).

**The surface these rows pin.** ``PlaywrightDriver(*, viewport, on_egress, user_data_dir=None, headless=True,
allowed_origins=None, end_when_closed=False, web=False, start_origin=None, proxy_resolve=None, proxy_connect=None)``
(B-1's names), and ``BrowserToolExecutor(..., web=True, start_origin=..., proxy_resolve=..., proxy_connect=...)``
forwarding them when it builds its own driver (*reading*, the module docstring of
``tests/unit/test_computer_web_executor.py``). In web mode the driver:

- builds one ``egress_proxy.EgressProxy(start_origin=<start>, on_blocked=..., on_host=..., resolve=<proxy_resolve>,
  connect=<proxy_connect>, tunnel_start_origin=True)`` per launch (O-A2 (2), P1 (d)) and ``start()``s it before
  ``chromium.launch``; it is stopped at teardown and close, and when a launch fails or is cancelled;
- launches with ``proxy={"server": proxy.origin, "bypass": "<-loopback>"}`` (P1 (b)), ``chromium_sandbox=True``, and
  ``args`` holding every ``CHROMIUM_ARGS`` flag (P1 (a)) plus ``--disable-quic``;
- routes the start origin through ``route.fetch(max_redirects=0)`` (T-0111g), ``continue_()``s public URLs, and
  aborts and logs everything else as BASE; public ``wss`` is left to Chromium (no WebSocket route matches it, so it
  goes through the proxy; a route-level ``connect_to_server`` would run in Playwright's process, outside the fence);
- deduplicates the proxy's ``on_blocked`` and ``on_host`` per action on ``{host, scheme}`` (O-A2 (4)); *reading*:
  a host reaches the executor as a ``take_events()`` event ``{"kind": "host", "host": ...}``.

**The fakes.** ``FakePlaywright`` records every ``launch`` / ``launch_persistent_context`` / ``new_context`` call and
the routes; a route registered with a callable, a glob or a regex is matched as Playwright does (the last registered
matching handler first; ``fallback()`` passes to the next). Any Playwright name not deliberately faked is a named
``AssertionError``. The proxy is the real ``EgressProxy`` (loopback, port 0) under a recording subclass patched into
``egress_proxy`` (and ``runtime``, if it binds the name), so "stopped" is observable: its port refuses a connect.
Nothing here opens a browser or a non-loopback socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import inspect
import re
import socket
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import egress_proxy as proxy_mod
from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod

REPO = Path(__file__).resolve().parents[2]
START = "http://127.0.0.1:8123"
VIEWPORT = (1280, 800)
BASE_OPTIONS = {
    "viewport": {"width": 1280, "height": 800},
    "screen": {"width": 1280, "height": 800},
    "device_scale_factor": 1,
    "accept_downloads": False,
    "service_workers": "block",
    "permissions": [],
}


# =====================================================================================================
# the fake Playwright
# =====================================================================================================


class _Strict:
    """Any public name not deliberately faked is a named failure (.claude/rules/tests.md)."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(
            f"the runtime used {type(self).__name__}.{name}, which the T-0160 launch fake does not provide "
            "(tests/unit/test_computer_web_launch.py); extend the fake if the call is legitimate"
        )


class FakeResponse(_Strict):
    def __init__(self, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}


class FakeRequest(_Strict):
    def __init__(self, url: str) -> None:
        self.url = url


class FakeRoute(_Strict):
    def __init__(self, url: str, response: FakeResponse) -> None:
        self.request = FakeRequest(url)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._response = response
        self.fell_back = False

    async def fetch(self, **kw: Any) -> FakeResponse:
        self.calls.append(("fetch", kw))
        return self._response

    async def fulfill(self, **kw: Any) -> None:
        self.calls.append(("fulfill", kw))

    async def continue_(self, **kw: Any) -> None:
        self.calls.append(("continue", kw))

    async def abort(self, *a: Any, **kw: Any) -> None:
        self.calls.append(("abort", {"args": a, **kw}))

    async def fallback(self, **kw: Any) -> None:
        self.fell_back = True

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeWebSocket(_Strict):
    def __init__(self, url: str) -> None:
        self.url = url
        self.calls: list[str] = []

    async def close(self, *a: Any, **kw: Any) -> None:
        self.calls.append("close")

    def connect_to_server(self) -> Any:
        self.calls.append("connect_to_server")
        return self


def _matches(pred: Any, url: str) -> bool:
    if callable(pred):
        return bool(pred(url))
    if isinstance(pred, re.Pattern):
        return pred.search(url) is not None
    if isinstance(pred, str):
        return pred in ("**", "**/*", "*") or fnmatch.fnmatch(url, pred.replace("**", "*"))
    raise AssertionError(f"unsupported route matcher {pred!r}")


class FakeCDP(_Strict):
    async def send(self, method: str, params: Any = None) -> Any:
        if method == "Target.getTargetInfo":
            return {"targetInfo": {"browserContextId": "ctx-1"}}
        return {}

    async def detach(self) -> None:
        pass


class FakePage(_Strict):
    def __init__(self) -> None:
        self._closed = False

    def on(self, event: str, cb: Callable[..., Any]) -> None:
        pass

    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True


class FakeContext(_Strict):
    def __init__(self, owner: FakePlaywright) -> None:
        self.owner = owner
        self.routes: list[tuple[Any, Callable[..., Any]]] = []
        self.ws_routes: list[tuple[Any, Callable[..., Any]]] = []
        self.pages: list[FakePage] = []
        self.closed = False
        self.init_scripts: list[str] = []

    def on(self, event: str, cb: Callable[..., Any]) -> None:
        pass

    def set_default_timeout(self, ms: float) -> None:
        pass

    async def route(self, pred: Any, handler: Callable[..., Any], **kw: Any) -> None:
        self.routes.append((pred, handler))

    async def route_web_socket(self, pred: Any, handler: Callable[..., Any]) -> None:
        self.ws_routes.append((pred, handler))

    async def add_init_script(self, script: str | None = None, **kw: Any) -> None:
        self.init_scripts.append(script or "")

    async def new_page(self) -> FakePage:
        p = FakePage()
        self.pages.append(p)
        return p

    async def new_cdp_session(self, page: Any) -> FakeCDP:
        return FakeCDP()

    async def close(self, **kw: Any) -> None:
        self.closed = True

    async def dispatch(self, url: str, response: FakeResponse | None = None) -> FakeRoute | None:
        """Playwright's order: the last registered matching handler first; ``fallback()`` passes on."""
        route = FakeRoute(url, response or FakeResponse())
        for pred, handler in reversed(self.routes):
            if _matches(pred, url):
                route.fell_back = False
                await handler(route)
                if not route.fell_back:
                    return route
        return None if not route.calls else route

    async def dispatch_ws(self, url: str) -> FakeWebSocket | None:
        for pred, handler in reversed(self.ws_routes):
            if _matches(pred, url):
                ws = FakeWebSocket(url)
                await handler(ws)
                return ws
        return None


class FakeBrowser(_Strict):
    def __init__(self, owner: FakePlaywright) -> None:
        self.owner = owner
        self.connected = True
        self.contexts: list[FakeContext] = []

    async def new_context(self, **kw: Any) -> FakeContext:
        self.owner.new_context_calls.append(kw)
        c = FakeContext(self.owner)
        self.contexts.append(c)
        return c

    async def new_browser_cdp_session(self) -> FakeCDP:
        return FakeCDP()

    def is_connected(self) -> bool:
        return self.connected

    async def close(self, **kw: Any) -> None:
        self.connected = False


class FakeChromium(_Strict):
    def __init__(self, owner: FakePlaywright) -> None:
        self.owner = owner

    async def launch(self, **kw: Any) -> FakeBrowser:
        self.owner.launch_calls.append(kw)
        self.owner.running_at_launch.append([p for p in RecordingProxy.instances if p.running])
        if self.owner.fail_launch:
            raise RuntimeError("synthetic launch failure")
        if self.owner.hang_launch is not None:
            self.owner.hang_launch.set()
            await asyncio.Event().wait()
        b = FakeBrowser(self.owner)
        self.owner.browsers.append(b)
        return b

    async def launch_persistent_context(self, *a: Any, **kw: Any) -> FakeContext:
        self.owner.persistent_calls.append((a, kw))
        return FakeContext(self.owner)


class FakePlaywright(_Strict):
    def __init__(self) -> None:
        self.chromium = FakeChromium(self)
        self.launch_calls: list[dict[str, Any]] = []
        self.persistent_calls: list[Any] = []
        self.new_context_calls: list[dict[str, Any]] = []
        self.browsers: list[FakeBrowser] = []
        self.running_at_launch: list[list[RecordingProxy]] = []
        self.fail_launch = False
        self.hang_launch: asyncio.Event | None = None

    async def stop(self) -> None:
        pass

    @property
    def context(self) -> FakeContext:
        return self.browsers[-1].contexts[-1]


class RecordingProxy(proxy_mod.EgressProxy):
    """The real proxy, recorded: its constructor keywords, and whether it is listening."""

    instances: list[RecordingProxy] = []

    def __init__(self, *a: Any, **kw: Any) -> None:
        self.args, self.kwargs = a, kw
        self.running = False
        RecordingProxy.instances.append(self)
        super().__init__(*a, **kw)

    def start(self) -> None:
        super().start()
        self.running = True
        self.port = int(self.origin.rsplit(":", 1)[1])

    def stop(self) -> None:
        super().stop()
        self.running = False

    def listening(self) -> bool:
        if not hasattr(self, "port"):
            return False
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", self.port), 0.5):
            return True
        return False

    def kw(self, name: str, index: int) -> Any:
        return (
            self.kwargs[name]
            if name in self.kwargs
            else (self.args[index] if len(self.args) > index else None)
        )


@pytest.fixture(autouse=True)
def recording_proxy(monkeypatch: pytest.MonkeyPatch) -> Any:
    RecordingProxy.instances = []
    monkeypatch.setattr(proxy_mod, "EgressProxy", RecordingProxy)
    if hasattr(runtime_mod, "EgressProxy"):
        monkeypatch.setattr(runtime_mod, "EgressProxy", RecordingProxy)
    yield RecordingProxy
    for p in RecordingProxy.instances:  # never leak a listener past a row
        with contextlib.suppress(Exception):
            p.stop()


async def fake_resolve(host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
    raise AssertionError("the launch rows never resolve")


async def fake_connect(ip: str, port: int) -> Any:
    raise AssertionError("the launch rows never connect")


def require_keywords(cls: type, *names: str) -> None:
    """A missing T-0160 keyword is a named assertion, never a ``TypeError`` (the pending convention)."""
    params = inspect.signature(cls.__init__).parameters
    missing = [n for n in names if n not in params]
    assert not missing, f"T-0160 B-1: {cls.__name__} has no keyword(s) {missing} yet"


def driver(
    *, web: bool, egress: list[dict[str, str]] | None = None, **kw: Any
) -> tuple[Any, FakePlaywright]:
    log = egress if egress is not None else []
    extra: dict[str, Any] = (
        {
            "web": True,
            "start_origin": START,
            "proxy_resolve": fake_resolve,
            "proxy_connect": fake_connect,
        }
        if web
        else {"web": False}
    )
    require_keywords(runtime_mod.PlaywrightDriver, *extra)
    d = runtime_mod.PlaywrightDriver(
        viewport=VIEWPORT,
        on_egress=log.append,
        allowed_origins=frozenset({START}),
        **extra,
        **kw,
    )
    pw = FakePlaywright()
    d._pw = pw  # the lazy ``async_playwright().start()`` is skipped; ``chromium`` is the fake
    return d, pw


def running() -> list[RecordingProxy]:
    return [p for p in RecordingProxy.instances if p.running]


# =====================================================================================================
# B-1: the web launch
# =====================================================================================================


async def test_web_launch_kwargs_and_proxy() -> None:
    """B-1: one proxy, started before ``chromium.launch``, built with the start origin, the two seams and
    ``tunnel_start_origin=True``; the launch has the proxy with the explicit ``<-loopback>`` bypass, the sandbox
    on, every BASE flag and ``--disable-quic``; a non-persistent context with BASE's options."""
    d, pw = driver(web=True)
    await asyncio.wait_for(d.ensure(), 10)
    try:
        assert len(RecordingProxy.instances) == 1, RecordingProxy.instances
        proxy = RecordingProxy.instances[0]
        assert pw.running_at_launch == [[proxy]], (
            "B-1: the proxy was not listening when Chromium launched"
        )
        assert proxy.kw("start_origin", 0) == START
        assert proxy.kw("resolve", 3) is fake_resolve and proxy.kw("connect", 4) is fake_connect
        assert proxy.kwargs.get("tunnel_start_origin") is True, proxy.kwargs
        assert callable(proxy.kw("on_blocked", 1)) and callable(proxy.kw("on_host", 2))
        (launch,) = pw.launch_calls
        assert launch.get("proxy") == {"server": proxy.origin, "bypass": "<-loopback>"}, launch
        assert launch.get("chromium_sandbox") is True, launch
        assert launch.get("headless") is True, launch
        args = list(launch.get("args", []))
        missing = [a for a in runtime_mod.CHROMIUM_ARGS if a not in args]
        assert missing == [], f"B-1 / P1 (a): BASE flags dropped in web mode: {missing}"
        assert "--disable-quic" in args, args
        assert pw.persistent_calls == []
        assert pw.new_context_calls == [BASE_OPTIONS], pw.new_context_calls
    finally:
        await d.close()


async def test_loopback_launch_equals_base() -> None:
    """B-1 / B-8: ``web=False`` launches exactly as BASE: no proxy built, ``launch(headless=True,
    args=CHROMIUM_ARGS)`` and nothing else, BASE's context options, and BASE's route verdicts."""
    log: list[dict[str, str]] = []
    d, pw = driver(web=False, egress=log)
    await asyncio.wait_for(d.ensure(), 10)
    try:
        assert RecordingProxy.instances == []
        assert pw.launch_calls == [{"headless": True, "args": list(runtime_mod.CHROMIUM_ARGS)}], (
            pw.launch_calls
        )
        assert pw.new_context_calls == [BASE_OPTIONS]
        ctx = pw.context
        r = await ctx.dispatch(f"{START}/index.html")
        assert r is not None and r.kinds() == ["fetch", "fulfill"], r and r.calls
        r = await ctx.dispatch("https://shop.example/")
        assert r is not None and r.kinds() == ["abort"], r and r.calls
        assert log == [{"host": "shop.example", "scheme": "https"}], log
        ws = await ctx.dispatch_ws("wss://shop.example/s")
        assert ws is not None and ws.calls == ["close"]
    finally:
        await d.close()


async def test_web_route_split() -> None:
    """B-1 route split, adversarial (SSRF by name at the route layer): the start origin keeps the no-redirect fetch;
    public https URLs are ``continue_()``d (never fetched in Playwright's process); everything else is aborted and
    logged ``{host, scheme}`` only, with no path or query (B-6). Public ``wss`` meets no WebSocket route (Chromium
    takes it through the proxy); loopback ``ws`` is closed and logged."""
    log: list[dict[str, str]] = []
    d, pw = driver(web=True, egress=log)
    await asyncio.wait_for(d.ensure(), 10)
    try:
        ctx = pw.context
        r = await ctx.dispatch(f"{START}/web.html")
        assert r is not None and r.kinds() == ["fetch", "fulfill"], r and r.calls
        assert r.calls[0][1].get("max_redirects") == 0, r.calls
        for url in (
            "https://shop.example/a/b?q=secret#frag",
            "https://93.184.215.14/",
            "https://en.wikipedia.org:443/",
        ):
            r = await ctx.dispatch(url)
            assert r is not None and r.kinds() == ["continue"], (url, r and r.calls)
        assert log == []
        refused = {
            "http://shop.example/cart?id=7": ("shop.example", "http"),
            "https://shop.example:8443/x": ("shop.example", "https"),
            "http://127.0.0.1:9/hud?t=1": ("127.0.0.1", "http"),
            "https://10.0.0.5/admin": ("10.0.0.5", "https"),
            "https://2130706433/": ("2130706433", "https"),
            "https://intranet/": ("intranet", "https"),
            "https://foo.localhost/": ("foo.localhost", "https"),
        }
        for url, (host, scheme) in refused.items():
            mark = len(log)
            r = await ctx.dispatch(url)
            assert r is not None and "abort" in r.kinds(), (url, r and r.calls)
            assert not {"continue", "fetch"} & set(r.kinds()), (url, r.calls)
            new = log[mark:]
            assert new and all(set(e) == {"host", "scheme"} for e in new), (url, new)
            assert {"host": host, "scheme": scheme} in new, (url, new)
        blob = repr(log)
        for leak in ("/cart", "id=7", "/hud", "t=1", "/admin", "secret", "frag"):
            assert leak not in blob, f"B-6: a URL path, query or fragment is logged: {log}"
        assert await ctx.dispatch_ws("wss://shop.example/socket") is None, (
            "a WebSocket route matched public wss: it would run outside the proxy"
        )
        mark = len(log)
        ws = await ctx.dispatch_ws("ws://127.0.0.1:9/")
        assert ws is not None and ws.calls == ["close"], ws and ws.calls
        assert log[mark:] == [{"host": "127.0.0.1", "scheme": "ws"}], log[mark:]
    finally:
        await d.close()


@pytest.mark.parametrize("how", ["fails", "cancelled"])
async def test_proxy_stopped_when_launch_fails_or_is_cancelled(how: str) -> None:
    """B-1, adversarial (a listener left behind): a launch that raises, or is cancelled mid-launch, leaves no proxy
    listening once ``ensure()`` has returned; the port refuses a connect."""
    d, pw = driver(web=True)
    if how == "fails":
        pw.fail_launch = True
        with pytest.raises(RuntimeError, match="synthetic launch failure"):
            await asyncio.wait_for(d.ensure(), 10)
    else:
        pw.hang_launch = asyncio.Event()
        task = asyncio.create_task(d.ensure())
        await asyncio.wait_for(pw.hang_launch.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(RecordingProxy.instances) == 1, "premise: the proxy was started before the launch"
    proxy = RecordingProxy.instances[0]
    assert not proxy.running and not proxy.listening(), (
        f"B-1: the proxy is still listening after a launch {how}"
    )
    await d.close()


async def test_relaunch_is_fresh_and_close_stops_the_proxy() -> None:
    """B-1 / B-2: after a crash the relaunch is a new non-persistent context (never ``launch_persistent_context``),
    the old proxy is stopped (at most one listening at any time), and ``close()`` stops the last one."""
    d, pw = driver(web=True)
    await asyncio.wait_for(d.ensure(), 10)
    pw.browsers[-1].connected = False  # the browser died
    await asyncio.wait_for(d.ensure(), 10)
    assert d.generation == 2
    assert len(pw.launch_calls) == 2 and pw.persistent_calls == []
    assert pw.new_context_calls == [BASE_OPTIONS, BASE_OPTIONS]
    assert len(running()) == 1, running()
    for launched in pw.running_at_launch:
        assert len(launched) == 1, f"two proxies listening at a launch: {pw.running_at_launch}"
    first = pw.running_at_launch[0][0]
    if first is not running()[0]:
        assert not first.listening(), "the first launch's proxy outlived its browser"
    await d.close()
    assert running() == [] and not any(p.listening() for p in RecordingProxy.instances)


# =====================================================================================================
# B-6: the proxy's callbacks, deduplicated per action
# =====================================================================================================


async def test_proxy_callbacks_deduplicated_per_action() -> None:
    """B-6 / O-A2 (4): Chromium preconnects, so the proxy reports a host twice. Fired from another thread (the proxy
    thread), two ``on_blocked`` for one ``{host, scheme}`` reach ``on_egress`` once and two ``on_host`` give one host
    event; after the action boundary (``take_events``) the same host counts again."""
    log: list[dict[str, str]] = []
    d, _pw = driver(web=True, egress=log)
    await asyncio.wait_for(d.ensure(), 10)
    try:
        assert len(RecordingProxy.instances) == 1, "B-1: web mode built no egress proxy"
        proxy = RecordingProxy.instances[0]
        on_blocked, on_host = proxy.kw("on_blocked", 1), proxy.kw("on_host", 2)

        def fire() -> None:
            for _ in range(2):
                on_blocked({"host": "private.example", "scheme": "https"})
                on_host("shop.example")

        t = threading.Thread(target=fire)
        t.start()
        t.join(5)
        await asyncio.sleep(0.05)  # a callback marshalled onto the loop has run
        assert log == [{"host": "private.example", "scheme": "https"}], log
        hosts = [e.get("host") for e in d.take_events() if e.get("kind") == "host"]
        assert hosts == ["shop.example"], hosts
        on_host("shop.example")
        on_blocked({"host": "private.example", "scheme": "https"})
        await asyncio.sleep(0.05)
        hosts = [e.get("host") for e in d.take_events() if e.get("kind") == "host"]
        assert hosts == ["shop.example"], f"the next action does not see the host again: {hosts}"
        assert len(log) == 2, log
    finally:
        await d.close()


# =====================================================================================================
# the executor builds its driver in web mode; the real driver always has focused_field
# =====================================================================================================


async def test_executor_builds_web_driver() -> None:
    """B-1 / B-3 (e): ``BrowserToolExecutor(web=True, start_origin=..., proxy_resolve=..., proxy_connect=...)``
    builds a web-mode ``PlaywrightDriver`` whose proxy carries the injected seams; the real driver class has
    ``focused_field``."""
    require_keywords(
        executor_mod.BrowserToolExecutor, "web", "start_origin", "proxy_resolve", "proxy_connect"
    )
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: 0,
        web=True,
        start_origin=START,
        allowed_origins=frozenset({START}),
        proxy_resolve=fake_resolve,
        proxy_connect=fake_connect,
    )
    d = ex.driver
    assert isinstance(d, runtime_mod.PlaywrightDriver)
    assert callable(getattr(runtime_mod.PlaywrightDriver, "focused_field", None))
    pw = FakePlaywright()
    d._pw = pw
    await asyncio.wait_for(d.ensure(), 10)
    try:
        (proxy,) = RecordingProxy.instances
        assert proxy.kw("resolve", 3) is fake_resolve and proxy.kw("connect", 4) is fake_connect
        assert pw.launch_calls[0].get("chromium_sandbox") is True
    finally:
        await ex.close()


def test_web_mode_without_start_origin_is_a_value_error() -> None:
    """B-1, adversarial (a web launch with no start origin has nothing to tunnel to and nothing to route as the
    start page): ``web=True`` without ``start_origin`` is a ``ValueError`` at construction, in the driver and in the
    executor, before any proxy is built. The too-wide side: the same constructions with the start origin, and
    ``web=False`` without one, do not raise. The message wording is not pinned (the task names none)."""
    require_keywords(runtime_mod.PlaywrightDriver, "web", "start_origin")
    require_keywords(executor_mod.BrowserToolExecutor, "web", "start_origin")
    origins = frozenset({START})
    for omitted in ({}, {"start_origin": None}):
        with pytest.raises(ValueError):
            runtime_mod.PlaywrightDriver(
                viewport=VIEWPORT,
                on_egress=lambda e: None,
                allowed_origins=origins,
                web=True,
                **omitted,
            )
        with pytest.raises(ValueError):
            executor_mod.BrowserToolExecutor(
                now_ms=lambda: 0, allowed_origins=origins, web=True, **omitted
            )
    assert RecordingProxy.instances == [], "a proxy was built for a refused web construction"
    runtime_mod.PlaywrightDriver(
        viewport=VIEWPORT,
        on_egress=lambda e: None,
        allowed_origins=origins,
        web=True,
        start_origin=START,
    )
    runtime_mod.PlaywrightDriver(
        viewport=VIEWPORT, on_egress=lambda e: None, allowed_origins=origins, web=False
    )
    executor_mod.BrowserToolExecutor(
        now_ms=lambda: 0, allowed_origins=origins, web=True, start_origin=START
    )
    executor_mod.BrowserToolExecutor(now_ms=lambda: 0, allowed_origins=origins, web=False)


def test_web_html_exists() -> None:
    """B-5: the new start page."""
    assert (REPO / "local_astra" / "pages" / "static" / "web.html").is_file()


# =====================================================================================================
# B-5 / B-8 negatives (green at BASE by design)
# =====================================================================================================

#: group B's base: main at "Merge T-0160 group A" (a literal, never a moving ref; .claude/rules/tests.md)
BASE = "a2d0ebde"
#: group B's tip (T-0094, two-ended): ``None`` until group B merges, meaning "the working tree"; whoever merges group
#: B sets it to the literal ``M^2`` of the merge in the same merge, so group C's legitimate edits to ``cli.py`` and
#: ``config.py`` do not redden these rows.
TIP: str | None = (
    "e73b2b56f5c977e92b6909fab168eab0b419c681"  # M^2 of "Merge T-0160 group B impl" (set at the merge)
)
FROZEN = [
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/contracts",
    "local_astra/pages/server.py",
    "tests/fakes",
]


def _have(rev: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", rev], capture_output=True
        ).returncode
        == 0
    )


def _range() -> list[str]:
    """``BASE TIP`` once TIP is set, else ``BASE`` (against the working tree, committed or not); skips with the
    reason when either commit is absent from this clone."""
    for rev in (BASE, TIP):
        if rev is not None and not _have(rev):
            pytest.skip(
                f"{rev} is not in this clone (T-0160 group B scope rows need {BASE}..{TIP})"
            )
    return [BASE] if TIP is None else [BASE, TIP]


@pytest.mark.parametrize("path", FROZEN)
def test_frozen_paths_unchanged(path: str) -> None:
    """B-8 / B-4: byte-identical over group B (``BASE..TIP``; the working tree until TIP is set). Green at BASE by
    design: a negative row."""
    revs = _range()
    r = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", *revs, "--", path], capture_output=True
    )
    assert r.returncode == 0, f"T-0160 B-8: {path} changed over {BASE}..{TIP or 'working tree'}"


def test_other_start_pages_unchanged() -> None:
    """B-5: over group B, ``index.html`` and every other start page and asset are byte-identical; only ``web.html``
    is new."""
    revs = _range()
    static = "local_astra/pages/static"
    r = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", *revs, "--", static],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    changed = [p for p in r.stdout.split() if p != f"{static}/web.html"]
    assert changed == [], f"T-0160 B-5: start pages changed: {changed}"
