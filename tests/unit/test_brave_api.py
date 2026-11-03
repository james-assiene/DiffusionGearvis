"""T-0163 criteria 4-6: ``BraveApiSearch`` (``local_astra/search/brave_api.py``) over ``tests/fakes/brave_api.py``.

No network: every row hands the backend ``FakeBrave(...).transport()`` (an ``httpx.MockTransport``), except the
direct-transport rows, which replace ``httpx.HTTPTransport.handle_request`` and
``httpx.AsyncHTTPTransport.handle_async_request`` with recorders (nothing below them runs, so nothing connects). The
key comes from the production loader over a tmp 0600 file holding a generated canary (never a real key).

Expected values are written out by hand from criterion 5 and the outcome table of criterion 6: host
``api.search.brave.com`` on 443 over https, path ``/res/v1/web/search`` (Brave Search API v1 web search), the key
only in ``X-Subscription-Token``, parameters ``q`` (cut to 500), ``count=5``, ``safesearch=moderate``,
``text_decorations`` off and ``result_filter=web``; ``SearchFailed`` text ``"<kind>: <detail>"`` with the table's
closed detail; result bounds title 200, url 500, snippet 600, 5 results.

The surface these rows assume (listed in the test-author's report): ``BraveApiSearch(key, *, transport=None)`` where
``key`` is the loader's ``load_key(path).key``; ``transport`` is an ``httpx`` transport used instead of the default;
``search(query)``, ``aclose()``; module-level ``ALLOWED_HOSTS``. The parameter names for "plain-text snippets" and
"web results only" are Brave's documented ``text_decorations`` and ``result_filter`` (the task asks the implementer
to confirm them).

Adversarial: the key leaking anywhere (``test_canary_never_leaks``: bodies and transport errors that echo it);
``follow_redirects=True`` (``test_redirect_makes_no_second_request``); ``trust_env`` left at its default
(``test_default_transport_ignores_the_environment``: ``SSL_CERT_FILE`` at a missing file, a mixed-case
``Https_Proxy`` that only ``trust_env`` reads); a body parsed before it is bounded (``oversize``, also chunked with no
``Content-Length``); the key in a query parameter (``test_request_shape``).

Regime: CPU, no network, httpx 0.28.1.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import secrets
import ssl
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fakes import brave_api as fb
from tests.fakes.brave_api import BLOCK, FakeBrave, brave_body, brave_json, brave_row

TASK = "T-0163"
API_MOD = "local_astra.search.brave_api"
KEY_MOD = "local_astra.search.brave_key"
HOST = "api.search.brave.com"
PATH = "/res/v1/web/search"
#: T-0111b's ``PROXY_ENV`` (criterion 5 names it), written out
PROXY_VARS = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
_ALPHABET = "GHJKMNPQRSTVWXYZghjkmnpqrstvwxyz"


def canary() -> str:
    return "BSAzq" + "".join(secrets.choice(_ALPHABET) for _ in range(27))


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.search"):
            raise NotImplementedError(f"{TASK}: {name} does not exist yet") from exc
        raise


def _api() -> Any:
    return _import(API_MOD)


def _failed() -> type[Exception]:
    from local_astra.contracts.protocols import SearchFailed

    return SearchFailed


def load(tmp_path: Path, key: str) -> Any:
    """The loader's key for ``key`` (a tmp 0600 file)."""
    m = _import(KEY_MOD)
    path = tmp_path / "keydir" / "brave_api_key"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n")
    path.chmod(0o600)
    result = m.load_key(path)
    assert result.status == "configured", (result.status, result.reason)
    return result.key


def make(tmp_path: Path, fake: FakeBrave, key: str | None = None) -> Any:
    return _api().BraveApiSearch(load(tmp_path, key or canary()), transport=fake.transport())


async def outcome(backend: Any, query: str = "Eiffel Tower height", timeout: float = 3.0) -> Any:
    """The search's result or the exception it raised (hang guard ``timeout``)."""
    task = asyncio.ensure_future(backend.search(query))
    done, _ = await asyncio.wait({task}, timeout=timeout)
    assert task in done, f"the search did not finish within {timeout} s (hang)"
    if task.cancelled():
        return asyncio.CancelledError()
    exc = task.exception()
    return exc if exc is not None else task.result()


def triples(results: Any) -> list[tuple[str, str, str]]:
    return [(r.title, r.url, r.snippet) for r in results.results]


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


# --- what is sent ---------------------------------------------------------------------------------------------------


def test_is_a_search_backend(tmp_path: Path) -> None:
    from local_astra.contracts.protocols import SearchBackend

    assert isinstance(make(tmp_path, FakeBrave()), SearchBackend)


def test_allowed_hosts() -> None:
    assert frozenset({HOST}) == _api().ALLOWED_HOSTS


async def test_request_shape(tmp_path: Path) -> None:
    key = canary()
    fake = FakeBrave()
    backend = make(tmp_path, fake, key)
    got = await outcome(backend, "weather in Paris")
    assert not isinstance(got, BaseException), repr(got)
    assert len(fake.requests) == 1
    req = fake.requests[0]
    assert req.method == "GET"
    assert req.url.scheme == "https"
    assert req.url.host == HOST
    assert req.url.port in (None, 443), req.url.port
    assert req.url.path == PATH
    params = req.url.params
    assert sorted(params.keys()) == sorted(
        {"q", "count", "safesearch", "text_decorations", "result_filter"}
    ), f"criterion 5: only these parameters; {sorted(params.multi_items())}"
    assert all(len(params.get_list(k)) == 1 for k in params), params.multi_items()
    assert params["q"] == "weather in Paris"
    assert params["count"] == "5"
    assert params["safesearch"] == "moderate"
    assert params["text_decorations"].lower() in ("false", "0"), "plain-text snippets"
    assert params["result_filter"] == "web", "web results only"
    # the key: exactly once, in X-Subscription-Token, nowhere else
    assert req.headers.get_list("x-subscription-token") == [key]
    assert "authorization" not in req.headers and "cookie" not in req.headers, dict(req.headers)
    others = [v for k, v in req.headers.multi_items() if k.lower() != "x-subscription-token"]
    assert not any(key in v for v in others), "the key in another header"
    assert key not in str(req.url), "the key in the URL"
    assert req.content == b"", "a GET with no body"
    # the client's own timeout bounds the request (5 s budget, within the session's 10 s tool deadline)
    timeouts = req.extensions.get("timeout", {})
    assert set(timeouts) >= {"connect", "read", "write", "pool"}, timeouts
    assert all(v is not None and 0 < v <= 5 for v in timeouts.values()), timeouts
    assert got.query == "weather in Paris"


async def test_long_query_cut_to_500(tmp_path: Path) -> None:
    fake = FakeBrave()
    q = "".join(chr(ord("a") + i % 26) for i in range(731))
    got = await outcome(make(tmp_path, fake), q)
    assert not isinstance(got, BaseException), repr(got)
    assert [r.url.params["q"] for r in fake.requests] == [q[:500]]


async def test_key_read_at_construction_not_per_search(tmp_path: Path) -> None:
    """Criterion 2: the key is held from construction; the file can be gone by the first search."""
    key = canary()
    fake = FakeBrave()
    backend = make(tmp_path, fake, key)
    (tmp_path / "keydir" / "brave_api_key").unlink()
    for q in ("one", "two"):
        got = await outcome(backend, q)
        assert not isinstance(got, BaseException), repr(got)
    assert [r.headers.get("x-subscription-token") for r in fake.requests] == [key, key]


# --- the outcome table ----------------------------------------------------------------------------------------------

#: id -> (reply, expected): expected is ("results", n) | ("timeout",) | ("failed", kind, exact str(exc))
TABLE: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "200": (brave_json(brave_body(fb.THREE_ROWS)), ("results", 3)),
    "200_empty": (brave_json(brave_body([])), ("results", 0)),
    "429": (fb.brave_status(429), ("failed", "rate_limited", "rate_limited: HTTP 429")),
    "401": (fb.brave_status(401), ("failed", "error", "error: HTTP 401 (key refused)")),
    "403": (fb.brave_status(403), ("failed", "error", "error: HTTP 403 (key refused)")),
    "404": (fb.brave_status(404), ("failed", "error", "error: HTTP 404")),
    "422": (fb.brave_status(422), ("failed", "error", "error: HTTP 422")),
    "500": (fb.brave_status(500), ("failed", "error", "error: HTTP 500")),
    "503": (fb.brave_status(503), ("failed", "error", "error: HTTP 503")),
    "302": (fb.brave_redirect(302), ("failed", "error", "error: redirect refused (HTTP 302)")),
    "oversize": (
        fb.brave_bytes(fb.oversize_body()),
        ("failed", "error", "error: malformed response"),
    ),
    "oversize_chunked": (
        fb.chunked(fb.oversize_body()),
        ("failed", "error", "error: malformed response"),
    ),
    "bad_json": (
        fb.brave_bytes(b'{"web": {"results": [}'),
        ("failed", "error", "error: malformed response"),
    ),
    "wrong_shape": (brave_json([1, 2, 3]), ("failed", "error", "error: malformed response")),
    "results_not_a_list": (
        brave_json({"type": "search", "web": {"results": "x"}}),
        ("failed", "error", "error: malformed response"),
    ),
    "timeout": (httpx.ReadTimeout("read timed out"), ("timeout",)),
    "connect_timeout": (httpx.ConnectTimeout("connect timed out"), ("timeout",)),
    "pool_timeout": (httpx.PoolTimeout("pool timed out"), ("timeout",)),
    "connect_error": (
        httpx.ConnectError("[Errno 111] Connection refused"),
        ("failed", "network", "network: ConnectError"),
    ),
    "read_error": (httpx.ReadError("reset"), ("failed", "network", "network: ReadError")),
    "remote_protocol_error": (
        httpx.RemoteProtocolError("bad"),
        ("failed", "network", "network: RemoteProtocolError"),
    ),
}


@pytest.mark.parametrize("case", [*TABLE, "closed"])
async def test_outcome_table(tmp_path: Path, case: str) -> None:
    failed = _failed()
    if case == "closed":
        fake = FakeBrave()
        backend = make(tmp_path, fake)
        await backend.aclose()
        got = await outcome(backend)
        assert isinstance(got, failed) and got.kind == "closed", repr(got)  # type: ignore[attr-defined]
        assert fake.requests == [], "no request after aclose"
        return
    reply, want = TABLE[case]
    fake = FakeBrave(reply)
    got = await outcome(make(tmp_path, fake))
    assert len(fake.requests) == 1, f"one request; {len(fake.requests)}"
    if want[0] == "results":
        from local_astra.contracts.toolcall import SearchResults

        assert isinstance(got, SearchResults), repr(got)
        assert len(got.results) == want[1]
        if want[1]:
            assert triples(got) == list(fb.THREE_TRIPLES)
    elif want[0] == "timeout":
        assert isinstance(got, TimeoutError) and not isinstance(got, failed), repr(got)
    else:
        _, kind, text = want
        assert isinstance(got, failed), repr(got)
        assert got.kind == kind, repr(got)  # type: ignore[attr-defined]
        assert str(got) == text, f"the closed detail; got {str(got)!r}"


# --- T-0176 criterion 8 (T-0163b): ``last_attempts``, one entry ``brave_api`` ---------------------------------------


def _attempts(backend: Any) -> Any:
    assert hasattr(backend, "last_attempts"), "T-0176 criterion 8: BraveApiSearch.last_attempts"
    return backend.last_attempts


def _want_attempts(want: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    """The outcome table's row as ``last_attempts``: ``()`` for results, else one ``("brave_api", kind)``."""
    if want[0] == "results":
        return ()
    if want[0] == "timeout":
        return (("brave_api", "timeout"),)
    return (("brave_api", want[1]),)


@pytest.mark.parametrize("case", [*TABLE, "closed"])
async def test_last_attempts_per_outcome(tmp_path: Path, case: str) -> None:
    """Criterion 8: after each outcome of the table, ``last_attempts`` is ``()`` (answered) or one
    ``("brave_api", kind)``; the key never appears in it."""
    key = canary()
    if case == "closed":
        backend = make(tmp_path, FakeBrave(), key)
        await backend.aclose()
        await outcome(backend)
        want: tuple[tuple[str, str], ...] = (("brave_api", "closed"),)
    else:
        reply, row = TABLE[case]
        backend = make(tmp_path, FakeBrave(reply), key)
        await outcome(backend)
        want = _want_attempts(row)
    got = _attempts(backend)
    assert got == want, f"T-0176 criterion 8 [{case}]: want {want}, got {got!r}"
    assert key not in repr(got)


async def test_last_attempts_429_then_success(tmp_path: Path) -> None:
    """A 429 gives ``(("brave_api", "rate_limited"),)``; ``()`` before any search, while the next search runs, and
    after it answers."""
    fake = FakeBrave(fb.brave_status(429), BLOCK)
    backend = make(tmp_path, fake)
    assert _attempts(backend) == ()
    await outcome(backend)
    assert _attempts(backend) == (("brave_api", "rate_limited"),)
    fake.entered.clear()
    task = asyncio.ensure_future(backend.search("again"))
    try:
        await _until_entered(fake)
        assert _attempts(backend) == (), f"() while a search runs; {_attempts(backend)!r}"
    finally:
        fake.release.set()
    done, _ = await asyncio.wait({task}, timeout=3)
    assert task in done and task.exception() is None, "premise: BLOCK answers 200 once released"
    assert _attempts(backend) == ()
    await backend.aclose()


async def test_body_at_the_bound_is_parsed(tmp_path: Path) -> None:
    """The other side of the 2 MiB bound: a body of exactly 2 MiB is not over it, so it is parsed (a guard too wide
    would refuse it)."""
    for reply in (fb.brave_bytes(fb.at_bound_body()), fb.chunked(fb.at_bound_body())):
        fake = FakeBrave(reply)
        got = await outcome(make(tmp_path, fake))
        assert not isinstance(got, BaseException), repr(got)
        assert triples(got) == list(fb.THREE_TRIPLES)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirect_makes_no_second_request(tmp_path: Path, status: int) -> None:
    """A 3xx is never followed, whatever its ``Location`` (a different host here, and the same host with another
    path below): one request, ``SearchFailed("error")``, ``redirect refused (HTTP <code>)``."""
    failed = _failed()
    for location in (
        "https://elsewhere.example.org/x",
        f"https://{HOST}/res/v1/web/search?q=other",
    ):
        fake = FakeBrave(fb.brave_redirect(status, location), brave_json(brave_body(fb.THREE_ROWS)))
        got = await outcome(make(tmp_path, fake))
        assert len(fake.requests) == 1, [str(r.url) for r in fake.requests]
        assert isinstance(got, failed) and got.kind == "error", repr(got)  # type: ignore[attr-defined]
        assert str(got) == f"error: redirect refused (HTTP {status})", str(got)


# --- result mapping -------------------------------------------------------------------------------------------------


async def test_result_mapping_bounds_tags_entities_and_scheme_filter(tmp_path: Path) -> None:
    long_url = "https://e.example.org/" + "p" * 600
    rows = [
        brave_row(
            "<strong>Paris</strong>  weather\n today",
            "https://w.example.org/p",
            "14&deg;C, light rain &amp; wind <b>12</b> km/h, &quot;showers&quot; &#39;later&#39;",
        ),
        {"title": "no url", "description": "x"},
        {"url": "https://no-title.example.org/", "description": "x"},
        brave_row("", "https://empty-title.example.org/", "x"),
        brave_row("js", "javascript:alert(1)", "x"),
        brave_row("data", "data:text/html,<b>x</b>", "x"),
        brave_row("ftp", "ftp://f.example.org/f", "x"),
        brave_row("no scheme", "f.example.org/path", "x"),
        ["not", "a", "mapping"],
        brave_row("T" * 250, long_url, "B" * 700),
        brave_row("Plain http", "http://h.example.org/x", ""),
        brave_row("Third", "https://3.example.org/", "three"),
        brave_row("Fourth", "https://4.example.org/", "four"),
        brave_row("Fifth", "https://5.example.org/", "five"),
        brave_row("Sixth (cut)", "https://6.example.org/", "six"),
    ]
    fake = FakeBrave(brave_json(brave_body(rows)))
    got = await outcome(make(tmp_path, fake), "Paris weather")
    assert not isinstance(got, BaseException), repr(got)
    assert got.query == "Paris weather"
    assert triples(got) == [
        (
            "Paris weather today",
            "https://w.example.org/p",
            "14°C, light rain & wind 12 km/h, \"showers\" 'later'",
        ),
        ("T" * 200, long_url[:500], "B" * 600),
        ("Plain http", "http://h.example.org/x", ""),
        ("Third", "https://3.example.org/", "three"),
        ("Fourth", "https://4.example.org/", "four"),
    ]


async def test_fields_under_the_bounds_are_kept_whole(tmp_path: Path) -> None:
    """A guard too wide cuts too much: fields just under the bounds come back whole."""
    title, url, snippet = "t" * 199, "https://k.example.org/" + "u" * 470, "s" * 599
    fake = FakeBrave(brave_json(brave_body([brave_row(title, url, snippet)])))
    got = await outcome(make(tmp_path, fake))
    assert triples(got) == [(title, url, snippet)]


# --- the environment --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("var", PROXY_VARS)
def test_proxy_env_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    """Criterion 5: a proxy variable (T-0111b's ``PROXY_ENV``) is refused at construction with ``ValueError``, and
    the refusal does not show the key."""
    key = canary()
    loaded = load(tmp_path, key)
    monkeypatch.setenv(var, "http://127.0.0.1:3128")
    with pytest.raises(ValueError) as ei:
        _api().BraveApiSearch(loaded)
    assert key not in str(ei.value) and key not in repr(ei.value)


async def test_netrc_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials from the environment: a ``.netrc`` naming the API host adds no ``Authorization`` header."""
    home = tmp_path / "home"
    home.mkdir()
    netrc = home / ".netrc"
    netrc.write_text(f"machine {HOST}\n  login someone\n  password hunter2hunter2\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NETRC", str(netrc))
    fake = FakeBrave()
    got = await outcome(make(tmp_path, fake))
    assert not isinstance(got, BaseException), repr(got)
    assert "authorization" not in fake.requests[0].headers


def _record_default_transports(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, httpx.Request]]:
    """Replace the default transports' send with a recorder answering three results; nothing below runs."""
    seen: list[tuple[Any, httpx.Request]] = []

    def sync_send(self: Any, request: httpx.Request) -> httpx.Response:
        seen.append((self, request))
        return httpx.Response(200, json=brave_body(fb.THREE_ROWS), request=request)

    async def async_send(self: Any, request: httpx.Request) -> httpx.Response:
        seen.append((self, request))
        return httpx.Response(200, json=brave_body(fb.THREE_ROWS), request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_send)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    return seen


def _assert_direct_and_verified(seen: list[tuple[Any, httpx.Request]]) -> None:
    assert len(seen) == 1, f"one request through the default transport; {len(seen)}"
    transport, request = seen[0]
    pool = transport._pool
    assert type(pool).__name__ in ("ConnectionPool", "AsyncConnectionPool"), (
        f"a direct connection, not through a proxy; pool {type(pool).__name__}"
    )
    ctx = pool._ssl_context
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname, "TLS verification is on"
    assert request.url.host == HOST


async def test_default_transport_ignores_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trust_env=False`` (criterion 5), observed: with ``SSL_CERT_FILE``/``SSL_CERT_DIR`` naming files that do not
    exist, a client that trusts the environment fails to build its TLS context; this one searches, directly, with
    TLS verification on."""
    seen = _record_default_transports(monkeypatch)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing-ca-dir"))
    backend = _api().BraveApiSearch(load(tmp_path, canary()))
    try:
        got = await outcome(backend)
    finally:
        await backend.aclose()
    assert not isinstance(got, BaseException), repr(got)
    _assert_direct_and_verified(seen)


async def test_no_proxy_from_a_variable_outside_proxy_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """httpx (through ``urllib.request.getproxies``) reads any ``*_proxy`` variable case-insensitively, so a
    mixed-case ``Https_Proxy`` is not in ``PROXY_ENV`` yet routes a ``trust_env`` client through a proxy. Either the
    backend refuses it at construction (``ValueError``) or it connects directly; never through the proxy."""
    seen = _record_default_transports(monkeypatch)
    monkeypatch.setenv("Https_Proxy", "http://127.0.0.1:9")
    try:
        backend = _api().BraveApiSearch(load(tmp_path, canary()))
    except ValueError:
        return  # refused: safe
    try:
        got = await outcome(backend)
    finally:
        await backend.aclose()
    assert not isinstance(got, BaseException), repr(got)
    _assert_direct_and_verified(seen)


# --- cancel and close -------------------------------------------------------------------------------------------------


async def _until_entered(fake: FakeBrave, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not fake.entered.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    assert fake.entered.is_set(), "the search never reached the transport"


async def test_cancel_with_blocked_transport_returns_promptly(tmp_path: Path) -> None:
    fake = FakeBrave(BLOCK)
    backend = make(tmp_path, fake)
    task = asyncio.ensure_future(backend.search("q"))
    try:
        await _until_entered(fake)
        t0 = time.monotonic()
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=1.0)
        assert task in done, (
            "the cancelled search did not return within 1 s (the transport is still blocked)"
        )
        assert time.monotonic() - t0 < 1.0
        assert task.cancelled(), repr(task.exception())
    finally:
        fake.release.set()
        await backend.aclose()


# --- criterion 14 (amendment 2026-10-27): nothing is sent after a cancel or close; identity encoding -----------------


def _hold_before_send(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Pause every search between its start and ``client.send``: ``httpx.Client.build_request`` builds the request
    and then waits (bounded, 5 s) until ``release`` is set. The check criterion 14 asks for ("immediately before
    sending") comes after the request is built, so the cancel or close lands in the window it must catch."""
    entered, release = threading.Event(), threading.Event()
    real = httpx.Client.build_request

    def held(self: Any, *a: Any, **k: Any) -> httpx.Request:
        request = real(self, *a, **k)
        entered.set()
        release.wait(timeout=5)
        return request

    monkeypatch.setattr(httpx.Client, "build_request", held)
    return entered, release


async def _until(event: threading.Event, what: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    assert event.is_set(), f"premise: {what}"


async def _search_threads_done(timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(t.name.startswith("astra-brave-") for t in threading.enumerate()):
            return
        await asyncio.sleep(0.005)
    raise AssertionError("a Brave search thread is still running")


async def test_cancel_before_send_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 14 (a): a search cancelled after it started but before the request is sent sends nothing: no keyed
    request leaves the box."""
    fake = FakeBrave()
    backend = make(tmp_path, fake)
    entered, release = _hold_before_send(monkeypatch)
    task = asyncio.ensure_future(backend.search("weather in Paris"))
    try:
        await _until(entered, "the search built its request (httpx.Client.build_request)")
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=1.0)
        assert task in done and task.cancelled(), "the cancel propagates at once"
    finally:
        release.set()
    await _search_threads_done()
    assert fake.requests == [], f"a cancelled search sent {len(fake.requests)} request(s)"
    await backend.aclose()


async def test_close_before_send_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 14 (a): the backend closed after a search started but before its request is sent: nothing is sent
    and the search ends ``SearchFailed("closed")``."""
    failed = _failed()
    fake = FakeBrave()
    backend = make(tmp_path, fake)
    entered, release = _hold_before_send(monkeypatch)
    task = asyncio.ensure_future(backend.search("weather in Paris"))
    try:
        await _until(entered, "the search built its request (httpx.Client.build_request)")
        await backend.aclose()
    finally:
        release.set()
    done, _ = await asyncio.wait({task}, timeout=3.0)
    assert task in done, "hang"
    got = task.exception() if not task.cancelled() else asyncio.CancelledError()
    await _search_threads_done()
    assert fake.requests == [], f"a closed backend sent {len(fake.requests)} request(s)"
    assert isinstance(got, failed) and got.kind == "closed", repr(got)  # type: ignore[attr-defined]


async def test_identity_encoding(tmp_path: Path) -> None:
    """Criterion 14 (b): every request carries ``Accept-Encoding: identity`` (no compressed body is decoded, so the
    2 MiB bound bounds memory); a success and a failure both."""
    fake = FakeBrave(brave_json(brave_body(fb.THREE_ROWS)), fb.brave_status(500))
    backend = make(tmp_path, fake)
    for _ in range(2):
        await outcome(backend)
    assert [r.headers.get_list("accept-encoding") for r in fake.requests] == [["identity"]] * 2


async def test_canary_never_leaks_through_httpcore_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 4, reviewer B F3: the 200 path through a real ``httpx.HTTPTransport`` over a real
    ``httpcore.ConnectionPool`` (its network replaced by ``httpcore.MockBackend``: no socket), with httpx and httpcore
    logging at DEBUG: the results come back and no log record holds the key."""
    import json

    import httpcore

    for name in ("httpx", "httpcore", "local_astra"):
        caplog.set_level(logging.DEBUG, logger=name)
    body = json.dumps(brave_body(fb.THREE_ROWS)).encode()
    wire = [
        b"HTTP/1.1 200 OK\r\n",
        b"Content-Type: application/json\r\n",
        b"Content-Length: %d\r\n" % len(body),
        b"\r\n",
        body,
    ]
    transport = httpx.HTTPTransport()
    transport._pool = httpcore.ConnectionPool(network_backend=httpcore.MockBackend(wire))
    key = canary()
    backend = _api().BraveApiSearch(load(tmp_path, key), transport=transport)
    got = await outcome(backend)
    await backend.aclose()
    assert not isinstance(got, BaseException), repr(got)
    assert triples(got) == list(fb.THREE_TRIPLES)
    core = [r for r in caplog.records if r.name.startswith("httpcore")]
    assert core, "premise: httpcore logged at DEBUG"
    shown = [r.getMessage() for r in caplog.records] + [repr(r.args) for r in caplog.records]
    leaked = [s[:160] for s in shown if key in s or key[5:17] in s]
    assert leaked == [], f"the key appears in a log record: {leaked}"


async def test_aclose_then_search_is_closed(tmp_path: Path) -> None:
    failed = _failed()
    fake = FakeBrave()
    backend = make(tmp_path, fake)
    first = await outcome(backend)
    assert not isinstance(first, BaseException), repr(first)
    await backend.aclose()
    got = await outcome(backend)
    assert isinstance(got, failed) and got.kind == "closed", repr(got)  # type: ignore[attr-defined]
    assert len(fake.requests) == 1, "nothing is sent after aclose"
    await backend.aclose()  # idempotent


# --- the key is never shown (adversarial) ------------------------------------------------------------------------------


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and every exception a traceback would print with it (``__cause__``, and ``__context__`` unless
    suppressed)."""
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out:
        out.append(cur)
        nxt = cur.__cause__
        if nxt is None and not cur.__suppress_context__:
            nxt = cur.__context__
        cur = nxt
    return out


def _shown(exc: BaseException) -> list[str]:
    out: list[str] = []
    for e in _chain(exc):
        out += [str(e), repr(e), repr(e.args)]
    return out


async def test_canary_never_leaks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Criterion 4 (adversarial): every failure where the API or the transport echoes the key (error bodies, a
    redirect ``Location``, a malformed body, transport errors whose message carries it) surfaces with no byte of it in
    any exception's ``str``/``repr``/``args`` (the whole printed chain), in ``repr``/``str`` of the backend, in
    anything printed, or in any log record at DEBUG (httpx and httpcore included)."""
    caplog.set_level(logging.DEBUG)
    for name in ("httpx", "httpcore", "local_astra"):
        caplog.set_level(logging.DEBUG, logger=name)
    key = canary()
    echo = f'{{"type": "ErrorResponse", "error": {{"detail": "token {key} refused"}}}}'.encode()
    replies = [
        fb.brave_status(401, echo),
        fb.brave_status(403, echo),
        fb.brave_status(429, echo),
        fb.brave_status(500, echo),
        fb.brave_redirect(302, f"https://elsewhere.example.org/?token={key}"),
        fb.brave_bytes(b"{" + key.encode() + b"}"),
        brave_json({"web": {"results": key}}),
        httpx.ConnectError(f"connect failed, X-Subscription-Token: {key}"),
        httpx.ReadTimeout(f"read timed out, X-Subscription-Token: {key}"),
        httpx.RemoteProtocolError(f"bad response for {key}"),
    ]
    shown: list[str] = []
    for reply in replies:
        fake = FakeBrave(reply)
        backend = make(tmp_path, fake, key)
        got = await outcome(backend)
        assert isinstance(got, Exception), f"premise: {reply!r} fails the search; got {got!r}"
        shown += _shown(got)
        shown += [repr(backend), str(backend)]
        await backend.aclose()
        assert fake.requests[0].headers.get("x-subscription-token") == key, (
            "premise: the key was sent"
        )
    out, err = capsys.readouterr()
    shown += [out, err]
    shown += [r.getMessage() for r in caplog.records]
    shown += [repr(r.args) for r in caplog.records]
    leaked = [s[:160] for s in shown if key in s or key[5:17] in s]
    assert leaked == [], f"the key appears in: {leaked}"
