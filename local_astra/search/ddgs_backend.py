"""``DdgsSearch``: a ``SearchBackend`` over keyless web search engines through the ``ddgs`` package (T-0111b).

This is the only module in ``local_astra`` that may reach a host off this machine (the scoped exception to the
loopback rule, H-T111c). The fence around it:

- the engines are pinned to ``ENGINES``, tried in that order, one concrete ``ddgs`` engine name per call (never
  ``auto`` or a comma list); any ``engine=`` argument is refused at construction;
- the hosts it may reach are ``ALLOWED_HOSTS``, each engine's search host (the live row checks every ``connect``);
- a proxy taken from the environment is refused, before every engine, because ``ddgs`` and ``primp`` honour
  several (``PROXY_ENV``);
- ``ddgs`` is imported lazily, inside the default client factory, so importing this module, or building it with
  a fake factory, never loads it.

Each search runs on its own daemon thread (``astra-search-N``), never joined, and hands its outcome back with
``loop.call_soon_threadsafe``. There is no ``asyncio.to_thread`` and no executor: ``asyncio.run`` waits on the
default executor at shutdown without a bound, and ``concurrent.futures`` joins its workers at exit, so a search
blocked in the HTTP layer would hold the process. The client ``timeout`` bounds each engine, so a search is bounded
by ``len(ENGINES) * CLIENT_TIMEOUT_S``; the session's tool timeout is the outer bound.

Fallback (ruling 4a): the next engine is tried only after ``SearchFailed("error")`` (including "No results found.",
which is how an anti-bot page surfaces) or ``SearchFailed("rate_limited")``. A timeout, ``SearchFailed("network")``,
a cancel, a proxy in the environment or ``aclose`` stops the loop. When every engine refuses, the last engine's
``SearchFailed`` is raised. The engine that answered is ``DdgsSearch.last_engine``.

Failures: ``TimeoutError`` for a timeout, ``SearchFailed(kind)`` otherwise (see ``_failure``); a cancelled search
raises ``asyncio.CancelledError`` at once and its thread's late result is dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol

from local_astra.contracts.protocols import SearchFailed
from local_astra.contracts.toolcall import SearchResult, SearchResults

__all__ = [
    "ALLOWED_HOSTS",
    "CLIENT_TIMEOUT_S",
    "ENGINES",
    "MAX_QUERY",
    "MAX_RESULTS",
    "MAX_SNIPPET_RAW",
    "MAX_TITLE",
    "MAX_URL",
    "PROXY_ENV",
    "REGION",
    "SAFESEARCH",
    "DdgsSearch",
    "default_client_factory",
]

#: The ``ddgs`` 9.16.1 text engines used, in the order tried (ruling 4a). Each is keyless, has a fixed search host
#: and its own index: ``duckduckgo`` (Bing's index), then ``brave`` (Brave's) and ``mojeek`` (Mojeek's), so one
#: provider's anti-bot block does not take the others with it. Every name must be a concrete engine: ``ddgs`` falls
#: back to all engines (``auto``) on a name it does not know.
ENGINES: Final = ("duckduckgo", "brave", "mojeek")
#: The search host of each engine in ``ENGINES`` (bare hostnames, from each engine's ``search_url`` at the pin):
#: duckduckgo POSTs to the first, brave and mojeek GET the second and third.
ALLOWED_HOSTS: Final[frozenset[str]] = frozenset(
    {"html.duckduckgo.com", "search.brave.com", "www.mojeek.com"}
)

#: The client's own timeout in seconds: the wall bound on one engine.
CLIENT_TIMEOUT_S: Final = 5
#: Results kept per search (equal to ``toolcall.loop.MAX_RESULTS``).
MAX_RESULTS: Final = 5
#: Field cuts that bound memory and the log; the 300-char snippet cut stays the renderer's.
MAX_TITLE: Final = 200
MAX_URL: Final = 500
MAX_SNIPPET_RAW: Final = 600
#: A longer query is cut before it is sent (equal to the session's ``MAX_ARG_TEXT``).
MAX_QUERY: Final = 500
#: The "no region" code; ``ddgs`` 9.16.1 sends it unvalidated (duckduckgo's ``l`` field; brave and mojeek split it
#: into a country/language cookie).
REGION: Final = "wt-wt"
#: Accepted by ``ddgs`` 9.16.1; duckduckgo ignores it, brave and mojeek treat it as their default.
SAFESEARCH: Final = "moderate"
#: Environment variables that route the client through a proxy: ``DDGS_PROXY`` (read by ``ddgs`` when
#: ``proxy=None``), ``PRIMP_PROXY``, ``HTTPS_PROXY``/``https_proxy`` and ``ALL_PROXY``/``all_proxy`` (read by
#: ``primp`` for an https request; checked on 2026-10-26 against a loopback listener).
PROXY_ENV: Final = (
    "DDGS_PROXY",
    "PRIMP_PROXY",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)

#: The failure kinds after which the next engine is tried (ruling 4a); every other outcome stops the loop.
_FALL_THROUGH: Final = frozenset({"error", "rate_limited"})
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_THREAD_IDS = itertools.count(1)


class _Client(Protocol):
    def text(self, query: str, **kwargs: Any) -> object: ...


ClientFactory = Callable[..., _Client]


def default_client_factory(*, timeout: int, proxy: str | None) -> _Client:
    """A real ``ddgs.DDGS`` client whose engines never follow a redirect. ``ddgs`` is imported here and nowhere
    else, so nothing loads it until a search (or the construction check of a default ``DdgsSearch``).

    primp follows up to 20 redirects by default and ddgs 9.16.1 never turns that off, so a 3xx from an engine
    would connect to whatever host its ``Location`` names, outside ``ALLOWED_HOSTS``. Each engine instance the
    client builds (``DDGS._get_engines``) gets ``http_client.client.follow_redirects = False``; a 3xx then reads as
    "No results found." and the next engine is tried. If those internals are not where 9.16.1 has them, building
    the engines raises instead of silently following redirects."""
    from ddgs import DDGS

    client = DDGS(proxy=proxy, timeout=timeout)
    get_engines: Callable[[str, str], list[Any]] = client._get_engines

    def redirect_free_engines(category: str, backend: str) -> list[Any]:
        engines = get_engines(category, backend)
        for engine in engines:
            _refuse_redirects(engine)
        return engines

    client._get_engines = redirect_free_engines  # type: ignore[method-assign]
    return client


def _refuse_redirects(engine: object) -> None:
    """Turn off redirect following on one ddgs engine's primp client, or raise (fail closed)."""
    http = getattr(engine, "http_client", None)
    inner = getattr(http, "client", None)
    if inner is None or not hasattr(inner, "follow_redirects"):
        raise RuntimeError(
            f"{type(engine).__name__}: no http_client.client.follow_redirects (ddgs internals moved since 9.16.1); "
            "refusing to search with redirects possibly followed"
        )
    inner.follow_redirects = False
    if inner.follow_redirects is not False:
        raise RuntimeError(
            f"{type(engine).__name__}: follow_redirects did not turn off; refusing to search"
        )


def _check_default_client() -> None:
    """At construction of a default ``DdgsSearch``: every name in ``ENGINES`` builds exactly that one ddgs engine,
    with redirects off. Builds clients only; opens no connection."""
    client: Any = default_client_factory(timeout=CLIENT_TIMEOUT_S, proxy=None)
    for name in ENGINES:
        engines = client._get_engines("text", name)
        if [getattr(e, "name", None) for e in engines] != [name]:
            raise RuntimeError(
                f"ddgs did not build exactly the engine {name!r} (got {len(engines)} engines)"
            )
        for engine in engines:
            if engine.http_client.client.follow_redirects is not False:
                raise RuntimeError(f"{name}: redirects are still followed")


def _refuse_proxy_env() -> None:
    for name in PROXY_ENV:
        if os.environ.get(name):
            raise ValueError(
                f"{name} is set: DdgsSearch connects directly to the allowed hosts only and refuses a proxy "
                "from the environment; unset it"
            )


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _result(row: object) -> SearchResult | None:
    """One client row as a ``SearchResult``, or ``None`` when it is not a mapping, lacks a title or an href, or the
    href's scheme is not http(s)."""
    if not isinstance(row, Mapping):
        return None
    title, href, body = row.get("title"), row.get("href"), row.get("body")
    if not isinstance(title, str) or not isinstance(href, str):
        return None
    title, href = _collapse(title), _collapse(href)
    scheme, colon, _ = href.partition(":")
    if not title or not colon or scheme.lower() not in _ALLOWED_SCHEMES:
        return None
    snippet = _collapse(body) if isinstance(body, str) else ""
    return SearchResult(
        title=title[:MAX_TITLE], url=href[:MAX_URL], snippet=snippet[:MAX_SNIPPET_RAW]
    )


def _results(query: str, rows: object) -> SearchResults:
    if not isinstance(rows, list | tuple):
        raise TypeError(f"the client answered {type(rows).__name__}, not a list of rows")
    kept: list[SearchResult] = []
    for row in rows:
        if (result := _result(row)) is not None:
            kept.append(result)
            if len(kept) == MAX_RESULTS:
                break
    return SearchResults(query=query, results=tuple(kept))


def _failure(exc: BaseException) -> BaseException:
    """The error table (T-0111b): what a client exception surfaces as. Never matches on messages."""
    if isinstance(exc, asyncio.CancelledError | SearchFailed):
        return exc
    try:
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    except ImportError:  # the search extra is absent: only builtin classes can have been raised
        ddgs_timeout: tuple[type[BaseException], ...] = ()
        ddgs_ratelimit: tuple[type[BaseException], ...] = ()
        ddgs_error: tuple[type[BaseException], ...] = ()
    else:
        ddgs_timeout, ddgs_ratelimit, ddgs_error = (
            (TimeoutException,),
            (RatelimitException,),
            (DDGSException,),
        )
    failure: BaseException
    if isinstance(exc, (TimeoutError, *ddgs_timeout)):
        failure = TimeoutError(f"search timed out: {exc}")
    elif isinstance(exc, ddgs_ratelimit):
        failure = SearchFailed("rate_limited", str(exc))
    elif isinstance(exc, OSError):
        failure = SearchFailed("network", str(exc))
    elif isinstance(exc, ddgs_error):
        failure = SearchFailed("error", str(exc))
    else:
        failure = SearchFailed("error", f"{type(exc).__name__}: {exc}")
    failure.__cause__ = exc
    return failure


def _failure_kind(failure: BaseException) -> str:
    """T-0176 criterion 8: an engine's translated failure (``_failure``'s output) as its ``last_attempts`` kind,
    from its type and ``SearchFailed.kind`` only (never a message)."""
    if isinstance(failure, TimeoutError):
        return "timeout"
    if isinstance(failure, SearchFailed) and failure.kind in ("rate_limited", "network", "closed"):
        return failure.kind
    return "error"


class DdgsSearch:
    """A ``SearchBackend`` over ``ddgs`` with the engines in ``ENGINES``; see the module docstring for the fence.

    ``client_factory`` is called once per engine tried, with ``timeout=CLIENT_TIMEOUT_S`` and ``proxy=None`` (a new
    client per call: sharing one across threads is unverified). Raises ``ValueError`` for any ``engine`` argument
    (the engines are pinned) or a proxy in the environment (``PROXY_ENV``).

    ``last_engine`` is the engine that answered the most recent successful search, ``None`` before one or while a
    search runs."""

    def __init__(
        self, *, client_factory: ClientFactory | None = None, engine: str | None = None
    ) -> None:
        if engine is not None:
            raise ValueError(f"engine {engine!r} refused: DdgsSearch is pinned to {ENGINES!r}")
        _refuse_proxy_env()
        if client_factory is None:
            _check_default_client()  # fail closed at construction if redirects cannot be turned off
        self._client_factory: ClientFactory = client_factory or default_client_factory
        self._closed = False
        self.last_engine: str | None = None
        #: T-0176 criterion 8: each engine the most recent search tried and refused, in order, with its failure
        #: kind (never a message); the answering engine is omitted; ``()`` before any search and while one runs
        self.last_attempts: tuple[tuple[str, str], ...] = ()

    async def search(self, query: str) -> SearchResults:
        self.last_attempts = ()
        if self._closed:
            raise SearchFailed("closed", "the search backend is closed")
        _refuse_proxy_env()
        self.last_engine = None
        if not query.strip():
            return SearchResults(query=query, results=())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SearchResults] = loop.create_future()
        stop = (
            threading.Event()
        )  # set on the loop once the caller stops waiting (cancel): try no further engine
        future.add_done_callback(lambda _f: stop.set())
        thread = threading.Thread(
            target=self._run,
            args=(loop, future, stop, query),
            name=f"astra-search-{next(_THREAD_IDS)}",
            daemon=True,
        )
        thread.start()
        return await future

    async def aclose(self) -> None:
        """Refuse further searches, and further engines of a search in progress. Returns at once: a search thread
        still running is a daemon and is never joined; it settles its search with ``SearchFailed("closed")``."""
        self._closed = True

    def _is_closed(self) -> bool:
        """Read on the search thread; ``aclose`` may set it between two reads."""
        return self._closed

    def _settle(
        self,
        future: asyncio.Future[SearchResults],
        outcome: SearchResults | BaseException,
        engine: str | None,
        attempts: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Runs on the loop. A future already cancelled or done (the caller gave up) ignores the late outcome."""
        if future.done():
            return
        self.last_attempts = attempts
        if isinstance(outcome, asyncio.CancelledError):
            future.cancel()
        elif isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            self.last_engine = engine
            future.set_result(outcome)

    def _attempt(self, engine: str, query: str) -> SearchResults | BaseException:
        """One engine, once. Total: a translator that raises (e.g. a raising ``__str__``) becomes
        ``SearchFailed("error")``, never an escape that would leave the search unsettled."""
        try:
            client = self._client_factory(timeout=CLIENT_TIMEOUT_S, proxy=None)
            rows = client.text(
                query[:MAX_QUERY],
                backend=engine,
                max_results=MAX_RESULTS,
                region=REGION,
                safesearch=SAFESEARCH,
            )
            return _results(query, rows)
        except BaseException as exc:  # handed to the awaiting task, translated by the error table
            try:
                return _failure(exc)
            except BaseException:  # the translator itself failed
                failure = SearchFailed("error", type(exc).__name__)
                failure.__cause__ = exc
                return failure

    def _run(
        self,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future[SearchResults],
        stop: threading.Event,
        query: str,
    ) -> None:
        # The future settles exactly once, whatever happens below (the ``finally``).
        outcome: SearchResults | BaseException = SearchFailed(
            "error", "the search thread gave no outcome"
        )
        answered: str | None = None
        attempts: list[
            tuple[str, str]
        ] = []  # T-0176 criterion 8: the engines tried that refused, and how
        try:
            for engine in ENGINES:
                if stop.is_set():  # cancelled: nobody is waiting
                    break
                if self._is_closed():
                    outcome = SearchFailed(
                        "closed", "the search backend was closed during the search"
                    )
                    break
                try:
                    _refuse_proxy_env()
                except ValueError as exc:
                    outcome = exc
                    break
                outcome = self._attempt(engine, query)
                if self._is_closed() and not isinstance(outcome, asyncio.CancelledError):
                    outcome = SearchFailed(
                        "closed", "the search backend was closed during the search"
                    )
                    attempts.append((engine, "closed"))
                    break
                if isinstance(outcome, SearchResults):
                    answered = engine
                    break
                if not isinstance(outcome, asyncio.CancelledError):
                    attempts.append((engine, _failure_kind(outcome)))
                if not (isinstance(outcome, SearchFailed) and outcome.kind in _FALL_THROUGH):
                    break
                # an "error" or "rate_limited" refusal: try the next engine; after the last, it is raised
        finally:
            with contextlib.suppress(RuntimeError):  # the loop closed while this thread ran
                loop.call_soon_threadsafe(self._settle, future, outcome, answered, tuple(attempts))
