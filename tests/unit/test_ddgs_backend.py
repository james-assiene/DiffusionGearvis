"""T-0111b criteria 1-6: ``DdgsSearch``, the ddgs/DuckDuckGo ``SearchBackend``, over ``tests/fakes/ddgs_client.py``.

No network and no ``sleep``: the fake blocks on a ``threading.Event``; ``block_timeout=2`` and ``_settle``'s 2 s are
hang guards only (a row that hits one has already failed). Expected values are written out by hand from the task's
numbers (5 results, 200/500/600-char cuts, a 500-char query, engine ``duckduckgo``, region ``wt-wt``, safesearch
``moderate``, timeout 5) and from the error table; the exception classes are the installed ``ddgs`` 9.16.1's.

Adversarial (the failure mode the task names): ``asyncio.to_thread``, ``run_in_executor`` or a ``ThreadPoolExecutor``
make a fresh interpreter hang at exit behind a search that never returns
(``test_blocked_search_does_not_hold_process_exit``); an engine other than ``duckduckgo`` is refused (7c).
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import ddgs_client as fake
from tests.fakes.search import LITERAL_ROWS

REPO = Path(__file__).resolve().parents[2]
BACKEND = "local_astra.search.ddgs_backend"
#: criterion 6, plus ``PRIMP_PROXY`` (the reviewer, 2026-10-26: primp, ddgs's HTTP layer, reads it too)
PROXY_VARS = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
TASK = "T-0111b"


def _mod() -> Any:
    try:
        return importlib.import_module(BACKEND)
    except ModuleNotFoundError as exc:
        if exc.name in (BACKEND, "local_astra.search"):
            raise NotImplementedError(f"{TASK}: {BACKEND} does not exist yet") from exc
        raise


def _search_failed() -> type[Exception]:
    from local_astra.contracts import protocols

    cls = getattr(protocols, "SearchFailed", None)
    if cls is None:
        raise NotImplementedError(f"{TASK}: contracts.protocols.SearchFailed does not exist yet")
    assert isinstance(cls, type) and issubclass(cls, Exception)
    return cls


def _make(factory: Any, **kw: Any) -> Any:
    return _mod().DdgsSearch(client_factory=factory, **kw)


async def _settle(task: asyncio.Future[Any], timeout: float = 2.0) -> None:
    """Hang guard: the task must be done within ``timeout`` (real seconds, but only a guard)."""
    done, _ = await asyncio.wait({task}, timeout=timeout)
    assert task in done, f"the search did not finish within {timeout} s (hang)"


async def _until_entered(f: fake.FakeDdgsFactory, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not f.entered.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0)
    assert f.entered.is_set(), "the search never reached the client's text()"


def _search_threads() -> set[int]:
    return {t.ident or 0 for t in threading.enumerate() if t.name.startswith("astra-search-")}


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _join_finished_search_threads() -> Any:
    """A search thread that has delivered its answer may still be unwinding when the row ends; give it 2 s to
    finish before the leak guard looks (one that is still alive after that is a real leak and fails the row)."""
    yield
    for t in threading.enumerate():
        if t.name.startswith("astra-search-") and t is not threading.current_thread():
            t.join(timeout=2)


@pytest.fixture
def release() -> Any:
    """Events set at teardown, then the search threads joined (bounded), so no blocked fake thread outlives its row
    (``tests/conftest.py::astra_thread_leak_guard`` fails a row that leaves an ``astra-`` thread running)."""
    events: list[threading.Event] = []
    yield events.append
    for e in events:
        e.set()
    for t in threading.enumerate():
        if t.name.startswith("astra-search-") and t is not threading.current_thread():
            t.join(timeout=2)


def _rows(n: int) -> list[dict[str, str]]:
    return [
        {"title": f"title {i}", "href": f"https://r{i}.example.org/p", "body": f"body {i}"}
        for i in range(n)
    ]


def _triples(results: Any) -> list[tuple[str, str, str]]:
    return [(r.title, r.url, r.snippet) for r in results.results]


# --- the fake is honest (no network: the installed package is only inspected) ------------------------------------


def test_the_fake_mirrors_the_pinned_ddgs() -> None:
    import importlib.metadata as md
    import inspect

    pytest.importorskip("ddgs", reason="the `search` extra (uv sync --all-extras)")
    assert md.version("ddgs") == "9.16.1", "the pin moved: re-read DDGS's signatures and exceptions"
    # ``ddgs.DDGS`` is a lazy proxy class (``ddgs._DDGSProxy``) whose call builds ``ddgs.ddgs.DDGS``
    real = importlib.import_module("ddgs.ddgs").DDGS

    def shape(sig: inspect.Signature) -> list[tuple[str, object, object]]:
        # name, kind, default (annotations differ only in being strings under ``from __future__``)
        return [(p.name, p.kind, p.default) for p in sig.parameters.values() if p.name != "self"]

    assert shape(inspect.signature(real.__init__)) == shape(fake.INIT_SIGNATURE)
    assert shape(inspect.signature(real.text)) == shape(fake.TEXT_SIGNATURE)
    # text() forwards to _search_sync, which takes the four keywords the backend sends
    sync = inspect.signature(real._search_sync).parameters
    for kw in ("region", "safesearch", "max_results", "backend"):
        assert sync[kw].kind is inspect.Parameter.KEYWORD_ONLY, kw
    E = fake.ddgs_exceptions()
    assert issubclass(E.RatelimitException, E.DDGSException)
    assert issubclass(E.TimeoutException, E.DDGSException)
    assert not issubclass(E.TimeoutException, TimeoutError)  # so the backend must translate it


# --- criterion 1 ----------------------------------------------------------------------------------------------


def test_is_a_search_backend() -> None:
    from local_astra.contracts.protocols import SearchBackend

    assert isinstance(_make(fake.client_factory()), SearchBackend)


def test_bound_constants() -> None:
    from local_astra.conductor.session import MAX_ARG_TEXT
    from local_astra.toolcall.loop import MAX_RESULTS as LOOP_MAX_RESULTS

    m = _mod()
    assert m.CLIENT_TIMEOUT_S == 5
    assert m.MAX_RESULTS == 5 == LOOP_MAX_RESULTS
    assert (m.MAX_TITLE, m.MAX_URL, m.MAX_SNIPPET_RAW) == (200, 500, 600)
    assert m.MAX_QUERY == 500 == MAX_ARG_TEXT
    # ruling (4a): the single ``ENGINE`` assertion moved to test_search_egress_scope.py::test_engines_are_pinned


# --- criterion 2 ----------------------------------------------------------------------------------------------


async def test_client_built_and_called_with_pinned_engine_and_bounds() -> None:
    f = fake.client_factory()
    await _make(f).search("weather in Paris")
    assert len(f.clients) == 1
    client = f.clients[0]
    assert client.explicit.get("timeout") == 5, (
        client.explicit
    )  # passed, not left to ddgs's default
    assert client.effective["proxy"] is None, client.effective
    assert client.effective["verify"] is True, client.effective  # TLS verification never turned off
    assert len(f.text_calls) == 1
    call = f.text_calls[0]
    assert call.query == "weather in Paris"
    assert call.kwargs.get("backend") == "duckduckgo"
    assert call.kwargs.get("max_results") == 5
    assert call.kwargs.get("region") == "wt-wt"
    assert call.kwargs.get("safesearch") == "moderate"


async def test_a_new_client_per_call() -> None:
    f = fake.client_factory()
    b = _make(f)
    await b.search("one")
    await b.search("two")
    assert len(f.clients) == 2 and f.clients[0] is not f.clients[1]
    assert [c.query for c in f.text_calls] == ["one", "two"]


async def test_rows_map_in_rank_order() -> None:
    from local_astra.contracts.toolcall import SearchResults

    f = fake.client_factory(_rows(3))
    got = await _make(f).search("Eiffel Tower height")
    assert isinstance(got, SearchResults)
    assert got.query == "Eiffel Tower height"
    assert _triples(got) == [
        ("title 0", "https://r0.example.org/p", "body 0"),
        ("title 1", "https://r1.example.org/p", "body 1"),
        ("title 2", "https://r2.example.org/p", "body 2"),
    ]


async def test_more_than_five_rows_keep_five() -> None:
    f = fake.client_factory(_rows(8))
    got = await _make(f).search("q")
    assert [r.title for r in got.results] == ["title 0", "title 1", "title 2", "title 3", "title 4"]


_GOOD_A = {"title": "A", "href": "https://a.example.org/", "body": "a"}
_GOOD_B = {"title": "B", "href": "http://b.example.org/x", "body": "b"}
_BAD = {
    "no_title": {"href": "https://bad.example.org/", "body": "x"},
    "no_href": {"title": "bad", "body": "x"},
    "javascript_scheme": {"title": "bad", "href": "javascript:alert(1)", "body": "x"},
    "data_scheme": {"title": "bad", "href": "data:text/html,<b>x</b>", "body": "x"},
    "not_a_mapping": ["bad", "https://bad.example.org/", "x"],
    "ftp_scheme": {"title": "bad", "href": "ftp://bad.example.org/f", "body": "x"},
    "no_scheme": {"title": "bad", "href": "bad.example.org/path", "body": "x"},
}


@pytest.mark.parametrize("case", list(_BAD))
async def test_bad_rows_dropped(case: str) -> None:
    f = fake.client_factory([_GOOD_A, _BAD[case], _GOOD_B])
    got = await _make(f).search("q")
    assert _triples(got) == [
        ("A", "https://a.example.org/", "a"),
        ("B", "http://b.example.org/x", "b"),
    ]


async def test_fields_cut_and_whitespace_collapsed() -> None:
    long_href = "https://e.example.org/" + "p" * 600
    f = fake.client_factory(
        [
            {"title": "T" * 250, "href": long_href, "body": "B" * 700},
            {
                "title": "Paris \n\t  weather",
                "href": "https://w.example.org/",
                "body": "14C,\n\n  light   rain",
            },
        ]
    )
    got = await _make(f).search("q")
    assert _triples(got) == [
        ("T" * 200, long_href[:500], "B" * 600),
        ("Paris weather", "https://w.example.org/", "14C, light rain"),
    ]


@pytest.mark.parametrize("query", ["", "   ", " \n\t "])
async def test_empty_query_makes_no_call(query: str) -> None:
    f = fake.client_factory()
    got = await _make(f).search(query)
    assert got.query == query and got.results == ()
    assert f.text_calls == []


async def test_long_query_cut_to_500() -> None:
    f = fake.client_factory()
    q = "".join(chr(ord("a") + i % 26) for i in range(731))
    got = await _make(f).search(q)
    assert [c.query for c in f.text_calls] == [q[:500]]
    assert got.query == q  # criterion 2: SearchResults.query is the query as asked, not as sent


async def test_proxy_set_after_construction_refused_per_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 6 re-checked at each search: a proxy variable set after construction still refuses, and no client
    is built (the environment can change between building the backend and its first search)."""
    f = fake.client_factory()
    b = _make(f)
    monkeypatch.setenv("HTTPS_PROXY", "socks5://127.0.0.1:9")
    with pytest.raises(ValueError) as ei:
        await b.search("weather in Paris")
    assert "HTTPS_PROXY" in str(ei.value)
    assert f.clients == [] and f.text_calls == []


class _StrRaises(Exception):
    def __str__(self) -> str:
        raise RuntimeError("__str__ of this exception raises")


async def test_exception_whose_str_raises_still_settles_as_error() -> None:
    """The error table is total: a client exception whose ``__str__`` raises settles as ``SearchFailed("error")``
    within the hang guard, instead of killing the hand-back and leaving the search unsettled."""
    failed = _search_failed()
    f = fake.client_factory(raise_=_StrRaises())
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    exc = task.exception()
    assert isinstance(exc, failed), repr(type(exc))
    assert exc.kind == "error"  # type: ignore[attr-defined]


async def test_special_literals_pass_through_untouched() -> None:
    """Sanitising is the renderer's job (T-0113 criterion 11): the backend hands the literals on as they came."""
    bodies = [row["snippet"] for row in LITERAL_ROWS]
    assert bodies, "fixture"
    for start in range(0, len(bodies), 5):
        chunk = bodies[start : start + 5]
        rows = [
            {"title": f"t{start + i}", "href": f"https://inj.example.org/{start + i}", "body": body}
            for i, body in enumerate(chunk)
        ]
        b2 = _make(fake.client_factory(rows))
        got = await b2.search("q")
        assert [r.snippet for r in got.results] == chunk


# --- criterion 3: errors ----------------------------------------------------------------------------------------


def _error(case: str) -> BaseException:
    E = fake.ddgs_exceptions()
    return {
        "timeout_exception": lambda: E.TimeoutException("https://... timed out"),
        "builtin_timeout": lambda: TimeoutError("timed out"),
        "ratelimit": lambda: E.RatelimitException("202 Ratelimit"),
        "ratelimit_saying_timeout": lambda: E.RatelimitException("timed out"),
        "oserror": lambda: OSError(101, "Network is unreachable"),
        "connection_refused": lambda: ConnectionRefusedError(111, "Connection refused"),
        "ddgs_generic": lambda: E.DDGSException("No results found."),
        "ddgs_generic_saying_ratelimit": lambda: E.DDGSException("Ratelimit"),
        "other": lambda: ValueError("boom"),
    }[case]()


#: what each surfaces as: "timeout" -> builtin TimeoutError (not SearchFailed), else SearchFailed(kind)
_ERROR_TABLE = {
    "timeout_exception": "timeout",
    "builtin_timeout": "timeout",
    "ratelimit": "rate_limited",
    "ratelimit_saying_timeout": "rate_limited",  # the class decides, never the message
    "oserror": "network",
    "connection_refused": "network",
    "ddgs_generic": "error",
    "ddgs_generic_saying_ratelimit": "error",  # the class decides, never the message
    "other": "error",
}


@pytest.mark.parametrize("case", list(_ERROR_TABLE))
async def test_error_mapping(case: str) -> None:
    failed = _search_failed()
    f = fake.client_factory(raise_=_error(case))
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    exc = task.exception()
    want = _ERROR_TABLE[case]
    if want == "timeout":
        assert isinstance(exc, TimeoutError) and not isinstance(exc, failed), repr(exc)
    else:
        assert isinstance(exc, failed), repr(exc)
        assert exc.kind == want, repr(exc)  # type: ignore[attr-defined]
        assert want in str(exc), str(exc)


def test_search_failed_carries_its_kind() -> None:
    failed = _search_failed()
    for kind in ("rate_limited", "network", "error", "closed"):
        exc = failed(kind)
        assert exc.kind == kind and kind in str(exc)  # type: ignore[attr-defined]


async def test_cancel_never_swallowed(release: Any) -> None:
    # (a) a client that raises CancelledError on its thread: it propagates, it is not "error"
    f = fake.client_factory(raise_=asyncio.CancelledError())
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    assert task.cancelled() or isinstance(task.exception(), asyncio.CancelledError), (
        task.exception() if not task.cancelled() else None
    )
    # (b) the awaiting task cancelled while the client blocks: the task ends cancelled, not with SearchFailed
    gate = threading.Event()
    release(gate)
    f2 = fake.client_factory(block=gate, block_timeout=2)
    task2 = asyncio.ensure_future(_make(f2).search("q"))
    await _until_entered(f2)
    task2.cancel()
    await _settle(task2)
    assert task2.cancelled(), repr(task2.exception())


# --- criterion 4: off the event loop ------------------------------------------------------------------------------


async def test_loop_not_blocked_while_client_blocks(release: Any) -> None:
    gate = threading.Event()
    release(gate)
    f = fake.client_factory(block=gate, block_timeout=2)
    ticks = 0
    ticks_while_blocked = 0

    async def ticker() -> None:
        nonlocal ticks, ticks_while_blocked
        while True:
            ticks += 1
            if f.blocked:
                ticks_while_blocked += 1
            await asyncio.sleep(0)

    t = asyncio.ensure_future(ticker())
    task = asyncio.ensure_future(_make(f).search("q"))
    await _until_entered(f)
    for _ in range(50):
        await asyncio.sleep(0)
    assert f.blocked and ticks_while_blocked >= 50, (f.blocked, ticks_while_blocked, ticks)
    assert not f.guard_expired
    gate.set()
    await _settle(task)
    assert task.exception() is None
    t.cancel()


async def test_runs_on_named_daemon_thread() -> None:
    f = fake.client_factory()
    await _make(f).search("q")
    (call,) = f.text_calls
    assert call.thread is not threading.main_thread()
    assert call.thread.name.startswith("astra-search-"), call.thread.name
    assert call.thread.daemon


# --- criterion 5: cancellation and close ----------------------------------------------------------------------------


async def test_cancel_returns_without_the_thread(release: Any) -> None:
    loop = asyncio.get_running_loop()
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))
    gate = threading.Event()
    release(gate)
    f = fake.client_factory(block=gate, block_timeout=2)
    task = asyncio.ensure_future(_make(f).search("q"))
    await _until_entered(f)
    task.cancel()
    await _settle(task, timeout=1.0)
    assert task.cancelled()
    assert f.blocked, "the task waited for the client's thread"
    # the thread finishing later hands a result to a cancelled future: ignored, nothing reaches the loop
    gate.set()
    f.text_calls[0].thread.join(timeout=2)
    for _ in range(20):
        await asyncio.sleep(0)
    assert errors == []


async def test_aclose_does_not_join(release: Any) -> None:
    gate = threading.Event()
    release(gate)
    f = fake.client_factory(block=gate, block_timeout=2)
    b = _make(f)
    task = asyncio.ensure_future(b.search("q"))
    await _until_entered(f)
    closing = asyncio.ensure_future(b.aclose())
    await _settle(closing, timeout=1.0)
    assert closing.exception() is None
    assert f.blocked, "aclose waited for the search thread"
    task.cancel()
    gate.set()
    await asyncio.wait({task}, timeout=2)


async def test_search_after_aclose_refused_without_thread() -> None:
    failed = _search_failed()
    f = fake.client_factory()
    b = _make(f)
    await b.aclose()
    before = _search_threads()
    with pytest.raises(failed) as ei:
        await b.search("weather in Paris")
    assert ei.value.kind == "closed"  # type: ignore[attr-defined]
    assert "closed" in str(ei.value)
    assert f.clients == [] and f.text_calls == []
    assert _search_threads() <= before


_EXIT_SCRIPT = r"""
import asyncio, sys, threading, time
from tests.fakes.ddgs_client import client_factory
from local_astra.search.ddgs_backend import DdgsSearch

async def main():
    f = client_factory(block=threading.Event())  # never set, no guard: blocks forever
    b = DdgsSearch(client_factory=f)
    t = asyncio.ensure_future(b.search("weather in Paris"))
    deadline = time.monotonic() + 1.5
    while not f.entered.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0)
    if not f.entered.is_set():
        sys.exit("the search never reached the client")
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    else:
        sys.exit("the search was not cancelled")
    await b.aclose()

asyncio.run(main())
print("returned", flush=True)
"""


def _clean_env() -> dict[str, str]:
    import os

    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env["PYTHONPATH"] = str(REPO)
    return env


def test_blocked_search_does_not_hold_process_exit() -> None:
    """A fresh interpreter: search on a client that blocks forever, cancel, aclose, return. It must exit in 3 s.
    ``asyncio.to_thread`` / ``run_in_executor`` / a ``ThreadPoolExecutor`` join their worker at shutdown and hang."""
    _mod()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _EXIT_SCRIPT],
            cwd=REPO,
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"the interpreter did not exit within 3 s behind a blocked search: {exc.stderr!r}"
        ) from exc
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr
    assert "returned" in proc.stdout
    assert elapsed < 3.0, elapsed


def test_import_and_construction_with_a_fake_never_import_ddgs() -> None:
    """Criterion 3 and the lazy import: importing the backend and building it with a fake leave ``ddgs`` and ``primp``
    out of ``sys.modules``, and raise nothing."""
    _mod()
    script = (
        "import sys\n"
        "from tests.fakes.ddgs_client import client_factory\n"
        "from local_astra.search.ddgs_backend import DdgsSearch\n"
        "DdgsSearch(client_factory=client_factory())\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('ddgs', 'primp')))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


# --- criterion 6: no proxy from the environment ---------------------------------------------------------------------


@pytest.mark.parametrize("var", PROXY_VARS)
def test_proxy_env_refused(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mod()
    monkeypatch.setenv(var, "socks5://127.0.0.1:9")
    f = fake.client_factory()
    with pytest.raises(ValueError) as ei:
        m.DdgsSearch(client_factory=f)
    assert var in str(ei.value), str(ei.value)
    assert f.clients == []


def test_empty_proxy_env_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_astra.contracts.protocols import SearchBackend

    m = _mod()
    for var in PROXY_VARS:
        monkeypatch.setenv(var, "")
    monkeypatch.setenv(
        "HTTP_PROXY", "socks5://127.0.0.1:9"
    )  # not in the list: plain-http proxies are not used
    assert isinstance(m.DdgsSearch(client_factory=fake.client_factory()), SearchBackend)


# --- criterion 7c (adversarial) -------------------------------------------------------------------------------------

_ENGINES = {
    "auto": "auto",
    "bing": "bing",
    "google": "google",
    "empty": "",
    "comma_list": "duckduckgo,bing",
}


@pytest.mark.parametrize("case", list(_ENGINES))
def test_engine_other_than_duckduckgo_refused(case: str) -> None:
    m = _mod()
    # ruling (4a): the control is the default construction (was ``engine="duckduckgo"``, single-engine era)
    assert m.DdgsSearch(client_factory=fake.client_factory()) is not None
    f = fake.client_factory()
    with pytest.raises(ValueError):
        m.DdgsSearch(client_factory=f, engine=_ENGINES[case])
    assert f.clients == []


# --- ruling (4a): keyless engine fallbacks ------------------------------------------------------------------------
#
# ``ENGINES`` is read from the module (the implementer picks the fallbacks from the pinned source); its structure is
# pinned in test_search_egress_scope.py::test_engines_are_pinned. The fake answers per engine (``by_backend``).
# Fall through only on SearchFailed("error") / ("rate_limited"); stop on a timeout, "network", a cancel, a proxy
# refusal or aclose.


def _engines() -> tuple[str, ...]:
    engines = getattr(_mod(), "ENGINES", None)
    if engines is None:
        raise NotImplementedError(f"{TASK} (4a): {BACKEND}.ENGINES does not exist yet")
    assert isinstance(engines, tuple) and len(engines) >= 2, engines
    return engines


def _answered_by(backend: Any, results: Any) -> object:
    """Ruling (4a) "the result": the engine that answered, carried on the return (``results.engine``) or recorded
    on the backend (``backend.last_engine``). Only these two names are read, and neither is called."""
    on_result = getattr(results, "engine", None)
    if on_result is not None:
        return on_result
    return getattr(backend, "last_engine", None)


def _no_results() -> BaseException:
    return fake.ddgs_exceptions().DDGSException("No results found.")


def _one_row(tag: str) -> list[dict[str, str]]:
    return [{"title": f"from {tag}", "href": f"https://{tag}.example.org/", "body": tag}]


async def test_first_engine_answering_tries_no_other() -> None:
    engines = _engines()
    f = fake.client_factory(by_backend={e: _one_row(e) for e in engines})
    b = _make(f)
    got = await b.search("weather in Paris")
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]
    assert [r.title for r in got.results] == [f"from {engines[0]}"]
    assert _answered_by(b, got) == engines[0]


async def test_engines_tried_in_order_one_call_each_until_one_answers() -> None:
    engines = _engines()
    last = engines[-1]
    answers: dict[str, Any] = {e: _no_results() for e in engines[:-1]}
    answers[last] = _one_row(last)
    f = fake.client_factory(by_backend=answers)
    b = _make(f)
    got = await b.search("weather in Paris")
    assert [c.kwargs.get("backend") for c in f.text_calls] == list(engines)
    assert all(c.query == "weather in Paris" for c in f.text_calls)
    assert [r.title for r in got.results] == [f"from {last}"]
    assert got.query == "weather in Paris"
    assert _answered_by(b, got) == last, (
        "the answering engine is recorded as results.engine or backend.last_engine"
    )


_FALL_THROUGH = {
    "no_results": lambda: fake.ddgs_exceptions().DDGSException("No results found."),
    "ratelimit": lambda: fake.ddgs_exceptions().RatelimitException("202 Ratelimit"),
    "other_error": lambda: ValueError("boom"),
}


@pytest.mark.parametrize("case", list(_FALL_THROUGH))
async def test_falls_through_on_error_or_rate_limit(case: str) -> None:
    engines = _engines()
    f = fake.client_factory(
        by_backend={engines[0]: _FALL_THROUGH[case](), engines[1]: _one_row(engines[1])}
    )
    b = _make(f)
    got = await b.search("q")
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0], engines[1]]
    assert [r.title for r in got.results] == [f"from {engines[1]}"]
    assert _answered_by(b, got) == engines[1]


_STOPS = {
    "timeout_exception": (lambda: fake.ddgs_exceptions().TimeoutException("timed out"), "timeout"),
    "builtin_timeout": (lambda: TimeoutError("timed out"), "timeout"),
    "network": (lambda: OSError(101, "Network is unreachable"), "network"),
    "cancelled": (lambda: asyncio.CancelledError(), "cancelled"),
}


@pytest.mark.parametrize("case", list(_STOPS))
async def test_stops_on_timeout_network_or_cancel(case: str) -> None:
    """The second engine is never called after a timeout (the 5 s client bound is spent), a network failure (the
    next engine shares the network) or a cancel. The row for being too wide is test_falls_through_on_..."""
    failed = _search_failed()
    engines = _engines()
    make_exc, want = _STOPS[case]
    f = fake.client_factory(
        by_backend={engines[0]: make_exc(), **{e: _one_row(e) for e in engines[1:]}}
    )
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]
    if want == "cancelled":
        assert task.cancelled() or isinstance(task.exception(), asyncio.CancelledError)
        return
    exc = task.exception()
    if want == "timeout":
        assert isinstance(exc, TimeoutError) and not isinstance(exc, failed), repr(exc)
    else:
        assert isinstance(exc, failed) and exc.kind == want, repr(exc)  # type: ignore[attr-defined]


async def test_task_cancelled_during_first_engine_tries_no_other(release: Any) -> None:
    engines = _engines()
    gate = threading.Event()
    release(gate)
    # engine 1 would fall through once released; the cancel must stop the loop first
    f = fake.client_factory(
        block=gate,
        block_timeout=2,
        by_backend={engines[0]: _no_results(), **{e: _one_row(e) for e in engines[1:]}},
    )
    task = asyncio.ensure_future(_make(f).search("q"))
    await _until_entered(f)
    task.cancel()
    await _settle(task)
    assert task.cancelled()
    gate.set()
    for t in threading.enumerate():
        if t.name.startswith("astra-search-"):
            t.join(timeout=2)
    for _ in range(20):
        await asyncio.sleep(0)
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]


async def test_aclose_during_first_engine_tries_no_other(release: Any) -> None:
    engines = _engines()
    gate = threading.Event()
    release(gate)
    f = fake.client_factory(
        block=gate,
        block_timeout=2,
        by_backend={engines[0]: _no_results(), **{e: _one_row(e) for e in engines[1:]}},
    )
    b = _make(f)
    task = asyncio.ensure_future(b.search("q"))
    await _until_entered(f)
    await b.aclose()
    gate.set()
    await _settle(task)
    for t in threading.enumerate():
        if t.name.startswith("astra-search-"):
            t.join(timeout=2)
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]
    # what the search settles as after aclose is unspecified, except that it is not a later engine's answer
    assert task.exception() is not None, "a later engine answered after aclose"


async def test_proxy_set_during_first_engine_stops_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "A proxy refusal stops the loop": a proxy variable that appears while engine 1 runs refuses engine 2."""
    engines = _engines()

    def first() -> BaseException:
        monkeypatch.setenv("HTTPS_PROXY", "socks5://127.0.0.1:9")
        return _no_results()

    f = fake.client_factory(by_backend={engines[0]: first, **{e: _one_row(e) for e in engines[1:]}})
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]
    assert isinstance(task.exception(), ValueError), repr(task.exception())
    assert "HTTPS_PROXY" in str(task.exception())


@pytest.mark.parametrize("kind", ["no_results", "ratelimit"])
async def test_every_engine_refusing_is_bounded_and_fails(kind: str) -> None:
    failed = _search_failed()
    engines = _engines()
    f = fake.client_factory(by_backend={e: _FALL_THROUGH[kind]() for e in engines})
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    assert [c.kwargs.get("backend") for c in f.text_calls] == list(
        engines
    )  # each once, never twice
    assert len(f.clients) <= len(engines)
    exc = task.exception()
    assert isinstance(exc, failed), repr(exc)
    assert exc.kind in ("error", "rate_limited"), exc.kind  # type: ignore[attr-defined]


async def test_every_call_names_one_concrete_engine() -> None:
    engines = _engines()
    failed = _search_failed()
    f = fake.client_factory(by_backend={e: _no_results() for e in engines})
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    assert isinstance(task.exception(), failed), repr(task.exception())  # retrieved: no GC warning
    for c in f.text_calls:
        name = c.kwargs.get("backend")
        assert isinstance(name, str) and name in engines, name
        assert "," not in name and name not in ("auto", "all"), name
        assert c.kwargs.get("max_results") == 5 and c.kwargs.get("region") == "wt-wt"
        assert c.kwargs.get("safesearch") == "moderate"
    assert all(c.explicit.get("timeout") == 5 for c in f.clients)


# --- (4a) review, 2026-10-27 -----------------------------------------------------------------------------------------


async def test_every_engine_refusing_raises_the_last_refusal() -> None:
    """ "After the last, it is raised": each engine refuses with its own message; the search fails with the *last*
    engine's refusal (its kind and its message), not the first's."""
    failed = _search_failed()
    engines = _engines()
    E = fake.ddgs_exceptions()
    answers: dict[str, Any] = {e: E.DDGSException(f"refusal-from-{e}") for e in engines[:-1]}
    answers[engines[-1]] = E.RatelimitException(f"refusal-from-{engines[-1]}")
    f = fake.client_factory(by_backend=answers)
    task = asyncio.ensure_future(_make(f).search("q"))
    await _settle(task)
    exc = task.exception()
    assert isinstance(exc, failed), repr(exc)
    assert exc.kind == "rate_limited", exc.kind  # type: ignore[attr-defined]
    assert f"refusal-from-{engines[-1]}" in str(exc), str(exc)
    for e in engines[:-1]:
        assert f"refusal-from-{e}" not in str(exc), str(exc)


async def test_answer_arriving_after_aclose_settles_closed(release: Any) -> None:
    """Engine 1 blocks, ``aclose`` runs, then engine 1 answers: the search settles as ``SearchFailed("closed")``,
    never as the answer, and no later engine is asked."""
    failed = _search_failed()
    engines = _engines()
    gate = threading.Event()
    release(gate)
    f = fake.client_factory(
        block=gate, block_timeout=2, by_backend={e: _one_row(e) for e in engines}
    )
    b = _make(f)
    task = asyncio.ensure_future(b.search("q"))
    await _until_entered(f)
    await b.aclose()
    gate.set()
    await _settle(task)
    exc = task.exception()
    assert isinstance(exc, failed) and exc.kind == "closed", repr(exc)  # type: ignore[attr-defined]
    assert [c.kwargs.get("backend") for c in f.text_calls] == [engines[0]]


# --- T-0176 criterion 8 (T-0163b): ``last_attempts`` -----------------------------------------------------------------

#: the failure kinds an entry may carry (criterion 8)
ATTEMPT_KINDS = frozenset({"error", "rate_limited", "network", "timeout", "closed"})


def _attempts(backend: Any) -> Any:
    assert hasattr(backend, "last_attempts"), "T-0176 criterion 8: DdgsSearch.last_attempts"
    return backend.last_attempts


def _three_engines() -> tuple[str, str, str]:
    engines = _engines()
    assert engines == ("duckduckgo", "brave", "mojeek"), f"premise: the pinned order; {engines}"
    return engines  # type: ignore[return-value]


def test_last_attempts_empty_before_any_search() -> None:
    assert _attempts(_make(fake.client_factory())) == ()


async def test_last_attempts_lists_refusals_in_order() -> None:
    """duckduckgo ``error``, brave ``rate_limited``, mojeek answers: the two refusals, in order; the answering
    engine is omitted."""
    _three_engines()
    E = fake.ddgs_exceptions()
    f = fake.client_factory(
        by_backend={
            "duckduckgo": E.DDGSException("No results found."),
            "brave": E.RatelimitException("202 Ratelimit"),
            "mojeek": _one_row("mojeek"),
        }
    )
    b = _make(f)
    got = await b.search("q")
    assert [r.title for r in got.results] == ["from mojeek"], "premise: mojeek answered"
    assert _attempts(b) == (("duckduckgo", "error"), ("brave", "rate_limited"))


async def test_last_attempts_first_engine_answering_is_empty() -> None:
    _three_engines()
    b = _make(fake.client_factory(by_backend={e: _one_row(e) for e in _engines()}))
    await b.search("q")
    assert _attempts(b) == ()


async def test_last_attempts_every_engine_refusing() -> None:
    """All three refuse: three entries, in order, and the raised exception is still the last refusal (the pin of
    ``test_every_engine_refusing_raises_the_last_refusal``, unchanged)."""
    failed = _search_failed()
    _three_engines()
    E = fake.ddgs_exceptions()
    f = fake.client_factory(
        by_backend={
            "duckduckgo": E.DDGSException("refusal-from-duckduckgo"),
            "brave": E.DDGSException("refusal-from-brave"),
            "mojeek": E.RatelimitException("refusal-from-mojeek"),
        }
    )
    b = _make(f)
    task = asyncio.ensure_future(b.search("q"))
    await _settle(task)
    exc = task.exception()
    assert isinstance(exc, failed) and exc.kind == "rate_limited", repr(exc)  # type: ignore[attr-defined]
    assert "refusal-from-mojeek" in str(exc) and "refusal-from-duckduckgo" not in str(exc)
    assert _attempts(b) == (
        ("duckduckgo", "error"),
        ("brave", "error"),
        ("mojeek", "rate_limited"),
    )


@pytest.mark.parametrize(
    ("case", "kind"),
    [("timeout_exception", "timeout"), ("builtin_timeout", "timeout"), ("network", "network")],
)
async def test_last_attempts_on_a_stop(case: str, kind: str) -> None:
    """A timeout or a network failure stops the loop (T-0111b 4a): one entry, its kind."""
    _three_engines()
    make_exc, _ = _STOPS[case]
    f = fake.client_factory(
        by_backend={"duckduckgo": make_exc(), "brave": _one_row("brave"), "mojeek": _one_row("m")}
    )
    b = _make(f)
    task = asyncio.ensure_future(b.search("q"))
    await _settle(task)
    assert task.exception() is not None
    assert _attempts(b) == (("duckduckgo", kind),)


async def test_last_attempts_is_the_most_recent_search_and_empty_while_running(
    release: Any,
) -> None:
    """Criterion 8: ``()`` while a search runs (not the previous search's attempts), then that search's own; a later
    search that the first engine answers leaves ``()`` (never accumulated); an empty query tries no engine."""
    _three_engines()
    E = fake.ddgs_exceptions()
    f = fake.client_factory(
        by_backend={"duckduckgo": E.DDGSException("No results found."), "brave": _one_row("b")}
    )
    b = _make(f)
    await b.search("first")
    assert _attempts(b) == (("duckduckgo", "error"),), "premise: one refusal recorded"
    gate = threading.Event()
    release(gate)
    f.block, f.block_timeout = gate, 2
    f.entered.clear()
    task = asyncio.ensure_future(b.search("second"))
    await _until_entered(f)
    assert _attempts(b) == (), f"T-0176 criterion 8: () while a search runs; {_attempts(b)!r}"
    gate.set()
    await _settle(task)
    assert _attempts(b) == (("duckduckgo", "error"),)
    f.block = None
    f.by_backend = {e: _one_row(e) for e in _engines()}
    await b.search("third")
    assert _attempts(b) == (), "the most recent search only"
    f.by_backend = {"duckduckgo": E.DDGSException("No results found."), "brave": _one_row("b")}
    await b.search("fourth")
    assert _attempts(b) == (("duckduckgo", "error"),)
    await b.search("   ")
    assert _attempts(b) == (), "an empty query tries no engine"


async def test_last_attempts_never_carry_message_query_or_url() -> None:
    """Adversarial (criterion 8: "never ``str(exc)``"): every engine refuses with a message holding a URL and the
    query (and one with a non-ddgs exception whose message does); ``last_attempts`` holds only engine names and
    kinds."""
    _three_engines()
    E = fake.ddgs_exceptions()
    query = "zqxsecret vinland deluxe"
    url = "https://html.duckduckgo.com/html?q=zqxsecret+vinland+deluxe"
    f = fake.client_factory(
        by_backend={
            "duckduckgo": E.DDGSException(f"{url} returned 202 for {query}"),
            "brave": ValueError(f"error at {url}: {query}"),
            "mojeek": E.RatelimitException(f"{query} {url}"),
        }
    )
    b = _make(f)
    task = asyncio.ensure_future(b.search(query))
    await _settle(task)
    assert task.exception() is not None
    got = _attempts(b)
    assert isinstance(got, tuple) and len(got) == 3, repr(got)
    for entry in got:
        assert isinstance(entry, tuple) and len(entry) == 2, repr(entry)
        engine, kind = entry
        assert type(engine) is str and engine in _engines(), repr(entry)
        assert type(kind) is str and kind in ATTEMPT_KINDS, repr(entry)
    shown = repr(got)
    for leak in ("zqxsecret", "vinland", "https", "html.", "202"):
        assert leak not in shown, f"T-0176 criterion 8: {leak!r} in last_attempts {shown}"


def _redirect_servers() -> tuple[Any, Any, list[str]]:
    """Two loopback HTTP servers: 127.0.0.1 answers every GET/POST with ``302 Location: http://127.0.0.2:<p>/``,
    127.0.0.2 answers 200 with an empty page. ``hits`` records the host of every request either receives."""
    import http.server

    hits: list[str] = []
    ports: dict[str, int] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self) -> None:
            host = self.server.server_address[0]
            hits.append(str(host))
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if host == "127.0.0.1":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.2:{ports['b']}/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b"<html><body></body></html>"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        do_GET = _answer
        do_POST = _answer

        def log_message(self, *args: Any) -> None:
            pass

    a = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    try:
        b = http.server.ThreadingHTTPServer(("127.0.0.2", 0), Handler)
    except OSError as exc:
        a.server_close()
        pytest.skip(f"cannot bind 127.0.0.2 on this box: {exc}")
    ports["b"] = b.server_address[1]
    return a, b, hits


async def test_engine_redirect_to_another_host_is_not_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(4a) review: the allowlist is the hosts each engine *searches*; a 302 to another host must not be followed
    (primp follows redirects by default). Offline: every pinned engine's ``search_url`` points at a loopback server
    that redirects to 127.0.0.2. The real default client factory (real ddgs + primp) must contact only the first
    host, and the search falls through to a failure."""
    pytest.importorskip("ddgs", reason="the `search` extra (uv sync --all-extras)")
    from ddgs.engines import ENGINES as REGISTRY

    failed = _search_failed()
    engines = _engines()
    for var in ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    a, b, hits = _redirect_servers()
    servers = [threading.Thread(target=s.serve_forever, daemon=True) for s in (a, b)]
    for t in servers:
        t.start()
    try:
        for name in engines:
            monkeypatch.setattr(
                REGISTRY["text"][name],
                "search_url",
                f"http://127.0.0.1:{a.server_address[1]}/search",
            )
        task = asyncio.ensure_future(_mod().DdgsSearch().search("weather in Paris"))
        await _settle(task, timeout=len(engines) * 5 + 2)
        exc = task.exception()
    finally:
        for s in (a, b):
            s.shutdown()
            s.server_close()
    assert hits, "no engine reached the loopback server (the search_url patch missed)"
    assert "127.0.0.2" not in hits, f"a redirect to another host was followed: {hits}"
    assert isinstance(exc, failed), repr(exc)


def test_default_factory_clients_do_not_follow_redirects() -> None:
    """Guard for the row above: every engine the default factory's client would use, for each name in ``ENGINES``,
    has ``follow_redirects`` False on its HTTP client. Built offline (constructing a ddgs engine opens no socket).
    Reads ddgs 9.16.1 internals (``DDGS._get_engines`` -> engine ``.http_client.client``); if the pin moves, this row
    is where it shows."""
    pytest.importorskip("ddgs", reason="the `search` extra (uv sync --all-extras)")
    m = _mod()
    client = m.default_client_factory(timeout=m.CLIENT_TIMEOUT_S, proxy=None)
    get_engines = getattr(client, "_get_engines", None)
    assert callable(get_engines), f"{type(client).__name__} has no ddgs _get_engines"
    for name in _engines():
        instances = get_engines("text", name)
        assert len(instances) == 1, (name, instances)
        http = getattr(getattr(instances[0], "http_client", None), "client", None)
        assert http is not None, f"{name}: no http_client.client"
        assert getattr(http, "follow_redirects", None) is False, (
            name,
            getattr(http, "follow_redirects", None),
        )
