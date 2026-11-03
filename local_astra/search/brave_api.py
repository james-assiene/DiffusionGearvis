"""``BraveApiSearch``: a ``SearchBackend`` over the Brave Search API v1 web search (T-0163), asked only after every
keyless engine has refused (the chain is built in ``cli._search_backend``).

The fence (criteria 5 and 9):

- every request goes to ``_SCHEME`` on ``_HOST`` (the one host in ``ALLOWED_HOSTS``), the default https port, path
  ``_PATH``, built from these constants (no URL literal in this module);
- through ``httpx`` only, with redirects off (a 3xx is a failure, never followed), ``trust_env=False`` (no proxy, no
  ``.netrc``, no CA override from the environment) and TLS verification on;
- a proxy variable in the environment (``PROXY_ENV``, the keyless backend's list) is refused at construction;
- the key goes only in the ``X-Subscription-Token`` header; it is held in the loader's opaque value and read only
  when a request is built; no request, header object, body or ``httpx`` error text is ever logged, kept or put in an
  exception (the details of the outcome table are a closed set);
- the parameters are only ``q`` (cut to ``MAX_QUERY``), ``count``, ``safesearch``, ``text_decorations`` (off: plain
  text snippets) and ``result_filter`` (``web`` results only);
- a response body over ``MAX_BODY`` bytes is refused before it is parsed.

Each search runs on its own daemon thread (``astra-brave-N``) with its own sync ``httpx.Client``, never joined, and
hands its outcome back with ``loop.call_soon_threadsafe``: an ``httpx.AsyncClient`` resolves names on the loop's
default executor, which ``asyncio.run`` waits for at exit, so a blocked lookup could hold the process. The client
timeout (``CLIENT_TIMEOUT_S``) bounds each request. A cancelled search raises ``asyncio.CancelledError`` at once and
its thread's late outcome is dropped. A fresh client per search also keeps no cookie between searches.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import itertools
import json
import os
import re
import threading
from collections.abc import Mapping
from typing import Final, Protocol

import httpx

from local_astra.contracts.protocols import SearchFailed
from local_astra.contracts.toolcall import SearchResult, SearchResults

__all__ = [
    "ALLOWED_HOSTS",
    "CLIENT_TIMEOUT_S",
    "MAX_BODY",
    "MAX_QUERY",
    "MAX_RESULTS",
    "PROXY_ENV",
    "BraveApiSearch",
]

#: The only host this module connects to.
_HOST: Final = "api.search.brave.com"
ALLOWED_HOSTS: Final[frozenset[str]] = frozenset({_HOST})
_SCHEME: Final = "https"
#: Brave Search API v1 web search.
_PATH: Final = "/res/v1/web/search"
_TOKEN_HEADER: Final = "X-Subscription-Token"

#: The client's own timeout in seconds (each of connect, read, write, pool), within the session's tool deadline.
CLIENT_TIMEOUT_S: Final = 5.0
#: Results asked for and kept (equal to the keyless backend's and ``toolcall.loop.MAX_RESULTS``).
MAX_RESULTS: Final = 5
#: A longer query is cut before it is sent (equal to the session's ``MAX_ARG_TEXT``).
MAX_QUERY: Final = 500
#: Field cuts (the keyless backend's bounds).
_MAX_TITLE: Final = 200
_MAX_URL: Final = 500
_MAX_SNIPPET_RAW: Final = 600
#: A response body over this many bytes is refused unparsed.
MAX_BODY: Final = 2 * 1024 * 1024
_SAFESEARCH: Final = "moderate"
#: Environment variables that would route a client through a proxy (the keyless backend's ``PROXY_ENV``, repeated
#: here so this module imports nothing of it). With ``trust_env=False`` httpx reads none of them; refusing them keeps
#: the two backends' rule the same.
PROXY_ENV: Final = (
    "DDGS_PROXY",
    "PRIMP_PROXY",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)

_MALFORMED: Final = "malformed response"
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_TAG: Final = re.compile(r"<[^>]*>")
_THREAD_IDS = itertools.count(1)


class _Key(Protocol):
    """The loader's opaque key value."""

    def reveal(self) -> str: ...


def _refuse_proxy_env() -> None:
    for name in PROXY_ENV:
        if os.environ.get(name):
            raise ValueError(
                f"{name} is set: the Brave Search API backend connects directly and refuses a proxy from the "
                "environment; unset it"
            )


def _clean(text: str) -> str:
    """Tags stripped, entities unescaped, whitespace collapsed."""
    return " ".join(html.unescape(_TAG.sub("", text)).split())


def _result(row: object) -> SearchResult | None:
    """One ``web.results`` row, or ``None`` when it is not a mapping, lacks a title or a url, or the url's scheme is
    not http(s)."""
    if not isinstance(row, Mapping):
        return None
    title, url, description = row.get("title"), row.get("url"), row.get("description")
    if not isinstance(title, str) or not isinstance(url, str):
        return None
    title, url = _clean(title), " ".join(url.split())
    scheme, colon, _ = url.partition(":")
    if not title or not colon or scheme.lower() not in _ALLOWED_SCHEMES:
        return None
    snippet = _clean(description) if isinstance(description, str) else ""
    return SearchResult(
        title=title[:_MAX_TITLE], url=url[:_MAX_URL], snippet=snippet[:_MAX_SNIPPET_RAW]
    )


def _results(query: str, body: bytes) -> SearchResults | SearchFailed:
    """A 200 body as results, or ``SearchFailed("error", "malformed response")``. A body with no ``web`` section
    (the API leaves it out when there are no web results) is zero results."""
    try:
        doc = json.loads(body)
    except (ValueError, RecursionError):
        return SearchFailed("error", _MALFORMED)
    if not isinstance(doc, Mapping):
        return SearchFailed("error", _MALFORMED)
    web = doc.get("web", {})
    if not isinstance(web, Mapping):
        return SearchFailed("error", _MALFORMED)
    rows = web.get("results", [])
    if not isinstance(rows, list):
        return SearchFailed("error", _MALFORMED)
    kept: list[SearchResult] = []
    for row in rows:
        if (result := _result(row)) is not None:
            kept.append(result)
            if len(kept) == MAX_RESULTS:
                break
    return SearchResults(query=query, results=tuple(kept))


#: T-0176 criterion 8: this backend's one engine label in ``last_attempts`` (and its ``last_engine``)
_ENGINE = "brave_api"


def _failure_kind(exc: BaseException) -> str:
    """T-0176 criterion 8: a failed search's kind, from its type and ``SearchFailed.kind`` only (never a message)."""
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, SearchFailed) and exc.kind in ("rate_limited", "network", "closed"):
        return exc.kind
    return "error"


def _status_failure(code: int) -> SearchFailed:
    """The outcome table (criterion 6) for a status other than 200."""
    if 300 <= code < 400:
        return SearchFailed("error", f"redirect refused (HTTP {code})")
    if code == 429:
        return SearchFailed("rate_limited", "HTTP 429")
    if code in (401, 403):
        return SearchFailed("error", f"HTTP {code} (key refused)")
    return SearchFailed("error", f"HTTP {code}")


class BraveApiSearch:
    """A ``SearchBackend`` over the Brave Search API; see the module docstring for the fence.

    ``key`` is the loader's key value (held, never shown); ``transport`` replaces the default ``httpx`` transport
    (tests). Raises ``ValueError`` for a proxy in the environment (``PROXY_ENV``). ``last_engine`` is ``"brave_api"``
    after a successful search, ``None`` before one or while a search runs."""

    def __init__(self, key: _Key, *, transport: httpx.BaseTransport | None = None) -> None:
        _refuse_proxy_env()
        self._key = key
        self._transport = transport
        self._closed = False
        self.last_engine: str | None = None
        #: T-0176 criterion 8: ``(("brave_api", kind),)`` after a failed search, ``()`` before any search, while
        #: one runs and after one answered; the kind only, never the message
        self.last_attempts: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        return f"BraveApiSearch(closed={self._closed})"

    async def search(self, query: str) -> SearchResults:
        self.last_attempts = ()
        if self._closed:
            self.last_attempts = ((_ENGINE, "closed"),)
            raise SearchFailed("closed", "the search backend is closed")
        self.last_engine = None
        if not query.strip():
            return SearchResults(query=query, results=())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SearchResults] = loop.create_future()
        # set on the loop once the caller stops waiting (cancel): the thread then sends nothing (criterion 14 (a))
        stop = threading.Event()
        future.add_done_callback(lambda _f: stop.set())
        thread = threading.Thread(
            target=self._run,
            args=(loop, future, stop, query),
            name=f"astra-brave-{next(_THREAD_IDS)}",
            daemon=True,
        )
        thread.start()
        return await future

    async def aclose(self) -> None:
        """Refuse further searches. Returns at once: a search thread still running is a daemon and is never joined;
        it settles its search with ``SearchFailed("closed")``."""
        self._closed = True

    def _settle(
        self, future: asyncio.Future[SearchResults], outcome: SearchResults | BaseException
    ) -> None:
        """Runs on the loop. A future already done (the caller gave up) ignores the late outcome."""
        if future.done():
            return
        if isinstance(outcome, BaseException):
            self.last_attempts = ((_ENGINE, _failure_kind(outcome)),)
            future.set_exception(outcome)
        else:
            self.last_engine = _ENGINE
            future.set_result(outcome)

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
        try:
            outcome = self._attempt(query, stop)
            if self._closed:
                outcome = SearchFailed("closed", "the search backend was closed during the search")
        finally:
            with contextlib.suppress(RuntimeError):  # the loop closed while this thread ran
                loop.call_soon_threadsafe(self._settle, future, outcome)

    def _attempt(self, query: str, stop: threading.Event) -> SearchResults | BaseException:
        """One request. Total: every failure becomes an outcome of the table, built fresh (no cause, no message
        from httpx, no body), so nothing the server or the transport echoed can reach a traceback."""
        try:
            return self._request(query, stop)
        except httpx.TimeoutException:
            failure: BaseException = TimeoutError("search timed out")
        except httpx.TransportError as exc:
            failure = SearchFailed("network", type(exc).__name__)
        except httpx.DecodingError:
            failure = SearchFailed("error", _MALFORMED)
        except Exception as exc:
            failure = SearchFailed("error", type(exc).__name__)
        failure.__cause__ = None
        failure.__suppress_context__ = True
        return failure

    def _request(self, query: str, stop: threading.Event) -> SearchResults | SearchFailed:
        url = httpx.URL(
            scheme=_SCHEME,
            host=_HOST,
            path=_PATH,
            params={
                "q": query[:MAX_QUERY],
                "count": str(MAX_RESULTS),
                "safesearch": _SAFESEARCH,
                "text_decorations": "false",
                "result_filter": "web",
            },
        )
        if url.scheme != _SCHEME or url.host not in ALLOWED_HOSTS or url.port not in (None, 443):
            return SearchFailed("error", "request outside the allowed host")
        with httpx.Client(
            transport=self._transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(CLIENT_TIMEOUT_S),
        ) as client:
            # the header is built here, at send time, and dropped with the request
            request = client.build_request(
                "GET",
                url,
                headers={
                    "Accept": "application/json",
                    # criterion 14 (b): no compressed body, so the body bound bounds memory
                    "Accept-Encoding": "identity",
                    _TOKEN_HEADER: self._key.reveal(),
                },
            )
            # criterion 14 (a): immediately before sending, a cancelled search or a closed backend sends nothing
            if stop.is_set() or self._closed:
                return SearchFailed("closed", "the search ended before its request was sent")
            response = client.send(request, stream=True, follow_redirects=False)
            try:
                if response.status_code != 200:
                    return _status_failure(response.status_code)
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit() and int(declared) > MAX_BODY:
                    return SearchFailed("error", _MALFORMED)
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BODY:
                        return SearchFailed("error", _MALFORMED)
                return _results(query, bytes(body))
            finally:
                response.close()
