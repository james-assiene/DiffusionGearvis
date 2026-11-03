"""T-0163 criterion 7: ``FallbackSearch`` (``local_astra/search/fallback.py``), a ``SearchBackend`` that asks a
fallback only after the primary refuses.

Over two scripted in-memory backends (below): the primary stands for ``DdgsSearch`` (its ``last_engine`` is the
keyless engine that answered), the fallback for ``BraveApiSearch``. One row chains a real ``BraveApiSearch`` over
``tests/fakes/brave_api.py`` (no network) to pin ``last_engine == "brave_api"`` end to end.

The table is criterion 7 written out: the fallback is called on ``SearchFailed("error")`` and
``SearchFailed("rate_limited")`` only; never on results (zero included), ``TimeoutError``,
``SearchFailed("network"|"closed")``, ``ValueError`` or ``CancelledError``; its outcome is final.

The surface these rows assume: ``FallbackSearch(primary, fallback)`` (positional), ``search``, ``aclose``,
``last_engine``.

Reviewer mutations aimed at here: the fallback fires on ``TimeoutError`` (``test_fallback_not_called_on[timeout]``);
it does not fire on ``rate_limited`` (``test_fallback_called_on[rate_limited]``).

Regime: CPU, in-memory, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import re
import secrets
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocols import SearchBackend, SearchFailed
from local_astra.contracts.toolcall import SearchResult, SearchResults
from tests.fakes.brave_api import THREE_TRIPLES, FakeBrave

TASK = "T-0163"
MOD = "local_astra.search.fallback"
#: T-0125's engine-name rule for ``last_engine``
ENGINE_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.search"):
            raise NotImplementedError(f"{TASK}: {name} does not exist yet") from exc
        raise


def chain(primary: Any, fallback: Any) -> Any:
    return _import(MOD).FallbackSearch(primary, fallback)


def results(query: str, n: int, tag: str) -> SearchResults:
    return SearchResults(
        query=query,
        results=tuple(
            SearchResult(title=f"{tag} {i}", url=f"https://{tag}{i}.example.org/", snippet=tag)
            for i in range(n)
        ),
    )


class Scripted:
    """A ``SearchBackend`` that answers each search from ``script`` in order (the last repeats): a ``SearchResults``
    (returned), an exception instance (raised), or ``"block"`` (waits on ``gate``). ``engine`` is what it puts in
    ``last_engine`` on success (``None`` while a search runs, like ``DdgsSearch``)."""

    def __init__(self, *script: Any, engine: str | None = None) -> None:
        self.script = list(script)
        self.engine = engine
        self.last_engine: str | None = None
        self.queries: list[str] = []
        self.closed = 0
        self.close_raises: BaseException | None = None
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()

    async def search(self, query: str) -> SearchResults:
        self.queries.append(query)
        self.last_engine = None
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        self.entered.set()
        if isinstance(item, str) and item == "block":
            await self.gate.wait()
            item = results(query, 1, "late")
        if isinstance(item, BaseException):
            raise item
        self.last_engine = self.engine
        return item  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        self.closed += 1
        if self.close_raises is not None:
            raise self.close_raises


def primary(*script: Any) -> Scripted:
    return Scripted(*script, engine="mojeek")


def fallback(*script: Any) -> Scripted:
    return Scripted(*script, engine="brave_api")


async def outcome(backend: Any, query: str = "q") -> Any:
    task = asyncio.ensure_future(backend.search(query))
    done, _ = await asyncio.wait({task}, timeout=3)
    assert task in done, "hang"
    if task.cancelled():
        return asyncio.CancelledError()
    exc = task.exception()
    return exc if exc is not None else task.result()


def test_is_a_search_backend() -> None:
    assert isinstance(
        chain(primary(results("q", 1, "p")), fallback(results("q", 1, "f"))), SearchBackend
    )


@pytest.mark.parametrize("kind", ["error", "rate_limited"])
async def test_fallback_called_on(kind: str) -> None:
    p = primary(SearchFailed(kind, "No results found."))  # type: ignore[arg-type]
    want = results("q", 2, "brave")
    f = fallback(want)
    got = await outcome(chain(p, f), "Eiffel Tower height")
    assert got is want, repr(got)
    assert p.queries == ["Eiffel Tower height"] and f.queries == ["Eiffel Tower height"], (
        "the fallback is asked the same query, once"
    )


def _not_called(case: str) -> Any:
    return {
        "results": results("q", 3, "p"),
        "zero_results": results("q", 0, "p"),
        "timeout": TimeoutError("search timed out"),
        "network": SearchFailed("network", "ConnectError"),
        "closed": SearchFailed("closed"),
        "value_error": ValueError("HTTPS_PROXY is set"),
        "cancel": asyncio.CancelledError(),
    }[case]


@pytest.mark.parametrize(
    "case", ["results", "zero_results", "timeout", "network", "closed", "value_error", "cancel"]
)
async def test_fallback_not_called_on(case: str) -> None:
    item = _not_called(case)
    p = primary(item)
    f = fallback(results("q", 1, "brave"))
    got = await outcome(chain(p, f))
    assert f.queries == [], f"[{case}] the fallback was asked"
    if isinstance(item, SearchResults):
        assert got is item, repr(got)
    elif case == "cancel":
        assert isinstance(got, asyncio.CancelledError), repr(got)
    else:
        assert got is item, f"[{case}] the primary's outcome propagates unchanged; {got!r}"


@pytest.mark.parametrize("case", ["results", "failure", "timeout", "zero_results"])
async def test_fallback_outcome_is_final(case: str) -> None:
    """Whatever the fallback does is the search's outcome (no retry of the primary, no swap of the error)."""
    p = primary(SearchFailed("error", "No results found."), results("q", 3, "p"))
    item = {
        "results": results("q", 2, "brave"),
        "zero_results": results("q", 0, "brave"),
        "failure": SearchFailed("error", "HTTP 500"),
        "timeout": TimeoutError("brave timed out"),
    }[case]
    f = fallback(item)
    got = await outcome(chain(p, f))
    assert got is item, repr(got)
    assert len(p.queries) == 1 and len(f.queries) == 1


async def test_fallback_rate_limited_is_final() -> None:
    p = primary(SearchFailed("rate_limited", "202 Ratelimit"))
    err = SearchFailed("rate_limited", "HTTP 429")
    f = fallback(err)
    got = await outcome(chain(p, f))
    assert got is err


@pytest.mark.parametrize(
    "case", ["primary", "brave_api", "failed", "during_search", "zero_results"]
)
async def test_last_engine(case: str) -> None:
    if case == "primary":
        c = chain(primary(results("q", 1, "p")), fallback(results("q", 1, "f")))
        await outcome(c)
        assert c.last_engine == "mojeek"
    elif case == "zero_results":
        c = chain(primary(results("q", 0, "p")), fallback(results("q", 1, "f")))
        await outcome(c)
        assert c.last_engine == "mojeek", "zero results is a primary success"
    elif case == "brave_api":
        c = chain(primary(SearchFailed("error")), fallback(results("q", 1, "f")))
        await outcome(c)
        assert c.last_engine == "brave_api"
    elif case == "failed":
        c = chain(
            primary(results("q", 1, "p"), SearchFailed("error")), fallback(SearchFailed("error"))
        )
        await outcome(c)
        assert c.last_engine == "mojeek", "premise: a success first"
        got = await outcome(c)
        assert isinstance(got, SearchFailed)
        assert c.last_engine is None, "a failed search leaves no engine (not the previous one)"
    else:  # during_search: None while a search runs, even after a success
        p = primary(results("q", 1, "p"), SearchFailed("error"))
        f = fallback("block")
        c = chain(p, f)
        await outcome(c)
        assert c.last_engine == "mojeek"
        task = asyncio.ensure_future(c.search("again"))
        await asyncio.wait_for(f.entered.wait(), 2)
        assert c.last_engine is None, "None while the search runs (during the fallback)"
        f.gate.set()
        await asyncio.wait_for(task, 2)
        assert c.last_engine == "brave_api"
    if c.last_engine is not None:
        assert ENGINE_RE.match(c.last_engine), c.last_engine


async def test_last_engine_none_while_primary_runs() -> None:
    p = primary(results("q", 1, "p"), "block")
    c = chain(p, fallback(results("q", 1, "f")))
    await outcome(c)
    assert c.last_engine == "mojeek", "premise: a success first"
    p.entered.clear()
    task = asyncio.ensure_future(c.search("q"))
    await asyncio.wait_for(p.entered.wait(), 2)
    assert c.last_engine is None, "None while the primary runs (not the previous search's engine)"
    p.gate.set()
    await asyncio.wait_for(task, 2)
    assert c.last_engine == "mojeek"


@pytest.mark.parametrize("which", ["primary_raises", "fallback_raises", "both_raise", "neither"])
async def test_aclose_closes_both_even_if_one_raises(which: str) -> None:
    p, f = primary(results("q", 1, "p")), fallback(results("q", 1, "f"))
    if which in ("primary_raises", "both_raise"):
        p.close_raises = RuntimeError("primary close boom")
    if which in ("fallback_raises", "both_raise"):
        f.close_raises = RuntimeError("fallback close boom")
    c = chain(p, f)
    # whether a close failure is re-raised is not specified; both must have been closed
    with contextlib.suppress(Exception):
        await c.aclose()
    assert (p.closed, f.closed) == (1, 1), (p.closed, f.closed)


async def test_cancel_during_primary_never_reaches_fallback() -> None:
    p = primary("block")
    f = fallback(results("q", 1, "f"))
    c = chain(p, f)
    task = asyncio.ensure_future(c.search("q"))
    await asyncio.wait_for(p.entered.wait(), 2)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=1)
    assert task in done and task.cancelled(), "the cancel propagates promptly"
    p.gate.set()
    await asyncio.sleep(0.01)
    assert f.queries == [], "the fallback was asked after a cancel"


# --- T-0176 criterion 8 (T-0163b): ``last_attempts`` concatenates the primary's and the fallback's ------------------


class Attempting(Scripted):
    """``Scripted`` that also holds ``last_attempts``: ``()`` while a search runs, then ``attempts`` (one value per
    search, in order; the last repeats) once it settles, as ``DdgsSearch``/``BraveApiSearch`` do."""

    def __init__(self, *script: Any, engine: str | None, attempts: Any) -> None:
        super().__init__(*script, engine=engine)
        self.attempts = list(attempts)
        self.last_attempts: tuple[tuple[str, str], ...] = ()

    async def search(self, query: str) -> SearchResults:
        mine = self.attempts.pop(0) if len(self.attempts) > 1 else self.attempts[0]
        self.last_attempts = ()
        try:
            return await super().search(query)
        finally:
            self.last_attempts = mine


KEYLESS3 = (("duckduckgo", "error"), ("brave", "error"), ("mojeek", "rate_limited"))


def _chain_attempts(c: Any) -> Any:
    assert hasattr(c, "last_attempts"), "T-0176 criterion 8: FallbackSearch.last_attempts"
    return c.last_attempts


async def test_last_attempts_fallback_failure_concatenates() -> None:
    p = Attempting(SearchFailed("rate_limited", "x"), engine="mojeek", attempts=[KEYLESS3])
    f = Attempting(
        SearchFailed("error", "HTTP 500"), engine="brave_api", attempts=[(("brave_api", "error"),)]
    )
    c = chain(p, f)
    assert _chain_attempts(c) == (), "() before any search"
    got = await outcome(c)
    assert isinstance(got, SearchFailed), "premise: the fallback failed"
    assert _chain_attempts(c) == (*KEYLESS3, ("brave_api", "error"))


async def test_last_attempts_fallback_answers_gives_the_primarys() -> None:
    p = Attempting(SearchFailed("error", "x"), engine="mojeek", attempts=[KEYLESS3])
    f = Attempting(results("q", 2, "brave"), engine="brave_api", attempts=[()])
    c = chain(p, f)
    await outcome(c)
    assert _chain_attempts(c) == KEYLESS3


@pytest.mark.parametrize(
    ("case", "item", "attempts"),
    [
        ("success_after_partial_refusal", results("q", 3, "p"), (("duckduckgo", "error"),)),
        ("network_stop", SearchFailed("network", "ConnectError"), (("duckduckgo", "network"),)),
        ("timeout_stop", TimeoutError("slow"), (("duckduckgo", "timeout"),)),
    ],
)
async def test_last_attempts_primary_only(case: str, item: Any, attempts: Any) -> None:
    """The fallback is not asked: the chain's attempts are the primary's alone."""
    p = Attempting(item, engine="mojeek", attempts=[attempts])
    f = Attempting(results("q", 1, "f"), engine="brave_api", attempts=[(("brave_api", "error"),)])
    c = chain(p, f)
    await outcome(c)
    assert f.queries == [], f"premise [{case}]: the fallback was not asked"
    assert _chain_attempts(c) == attempts, f"[{case}] {_chain_attempts(c)!r}"


async def test_last_attempts_empty_while_the_fallback_runs() -> None:
    """Criterion 8 ("``()`` while one is running"): the primary has refused (its attempts are known) but the search
    is still running in the fallback; once the fallback answers, the primary's attempts."""
    p = Attempting(SearchFailed("error", "x"), engine="mojeek", attempts=[KEYLESS3])
    f = Attempting("block", engine="brave_api", attempts=[()])
    c = chain(p, f)
    task = asyncio.ensure_future(c.search("q"))
    await asyncio.wait_for(f.entered.wait(), 2)
    assert _chain_attempts(c) == (), f"() while the search runs; {_chain_attempts(c)!r}"
    f.gate.set()
    await asyncio.wait_for(task, 2)
    assert _chain_attempts(c) == KEYLESS3


async def test_last_attempts_chain_with_real_brave_429(tmp_path: Path) -> None:
    """End to end over the real ``BraveApiSearch`` (fake API): the keyless refusals, then ``("brave_api",
    "rate_limited")`` from a 429."""
    from tests.fakes.brave_api import brave_status

    p = Attempting(SearchFailed("error", "x"), engine="mojeek", attempts=[KEYLESS3])
    c = chain(p, _brave(tmp_path, FakeBrave(brave_status(429))))
    got = await outcome(c)
    assert isinstance(got, SearchFailed) and got.kind == "rate_limited", repr(got)
    assert _chain_attempts(c) == (*KEYLESS3, ("brave_api", "rate_limited"))
    await c.aclose()


def test_fallback_module_names_no_backend() -> None:
    """Criterion 7: it wraps a primary and a fallback and names neither concretely (no import of either backend
    module, no class name)."""
    mod = _import(MOD)
    text = Path(mod.__file__).read_text(encoding="utf-8")
    for name in (
        "DdgsSearch",
        "BraveApiSearch",
        "ddgs_backend",
        "import brave_api",
        "search.brave_api",
    ):
        assert name not in text, f"fallback.py names {name!r}"


# --- end to end with the real Brave backend over the fake API ---------------------------------------------------------


def _brave(tmp_path: Path, fake: FakeBrave) -> Any:
    key_mod = _import("local_astra.search.brave_key")
    path = tmp_path / "brave_api_key"
    path.write_text("BSAzq" + "".join(secrets.choice("GHJKMNPQRSTVWXYZ") for _ in range(27)))
    path.chmod(0o600)
    loaded = key_mod.load_key(path)
    assert loaded.status == "configured"
    return _import("local_astra.search.brave_api").BraveApiSearch(
        loaded.key, transport=fake.transport()
    )


async def test_chain_with_real_brave_reports_brave_api(tmp_path: Path) -> None:
    """``last_engine == "brave_api"`` when the real ``BraveApiSearch`` answered after the primary refused (what the
    session's ``web search`` note will name), and the results are Brave's."""
    fake = FakeBrave()
    c = chain(primary(SearchFailed("error", "No results found.")), _brave(tmp_path, fake))
    got = await outcome(c, "Eiffel Tower height")
    assert isinstance(got, SearchResults), repr(got)
    assert [(r.title, r.url, r.snippet) for r in got.results] == list(THREE_TRIPLES)
    assert c.last_engine == "brave_api"
    assert len(fake.requests) == 1
    await c.aclose()
