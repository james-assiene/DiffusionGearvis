"""A fake Brave Search API for ``local_astra/search/brave_api.py`` (T-0163): canned JSON bodies and an
``httpx.MockTransport`` that records every request.

No socket is ever opened. ``FakeBrave(replies).transport()`` is handed to ``BraveApiSearch(key, transport=...)``. The
transport works under both a sync ``httpx.Client`` (a search on a worker thread) and an ``httpx.AsyncClient``: the
implementation chooses (docs/tasks/T-0163.md, implementation notes), so the fake does not.

A reply is one of:

- an ``httpx.Response`` (returned as is; ``brave_json`` / ``brave_status`` / ``brave_redirect`` build them);
- an exception instance (raised from the transport, as httpx raises a transport error);
- ``BLOCK``: the request waits until ``release`` is set (bounded by ``block_timeout``), on a ``threading.Event`` for
  the sync path and with ``asyncio.sleep`` polling for the async path, then answers 200 with no results. The
  request is recorded and ``entered`` is set before it waits.

Replies are consumed in order; the last one repeats. Every request is recorded in ``requests`` (``httpx.Request``
objects, whose ``url``, ``headers`` and ``extensions`` the rows read).

The body shapes are the Brave Search API v1 web-search response as far as the backend reads it
(``{"type": "search", "web": {"type": "search", "results": [{"title", "url", "description"}, ...]}}``); fields the
backend ignores are included so a row cannot pass only because the body is minimal.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Final

import httpx

#: Sentinel reply: hold the request until ``FakeBrave.release`` is set.
BLOCK: Final = "BLOCK"

#: The response-size bound the backend enforces (criterion 5: a body over 2 MiB is refused unparsed).
MAX_BODY: Final = 2 * 1024 * 1024


def brave_row(title: str, url: str, description: str = "", **extra: Any) -> dict[str, Any]:
    """One ``web.results`` entry as the API returns it (plus fields the backend must ignore)."""
    row: dict[str, Any] = {
        "type": "search_result",
        "title": title,
        "url": url,
        "description": description,
        "is_source_local": False,
        "is_source_both": False,
        "language": "en",
        "family_friendly": True,
        "meta_url": {"scheme": "https", "netloc": "example.org", "path": "› page"},
    }
    row.update(extra)
    return row


def brave_body(rows: Sequence[dict[str, Any]], query: str = "q") -> dict[str, Any]:
    """A whole 200 body around ``rows``."""
    return {
        "type": "search",
        "query": {"original": query, "show_strict_warning": False, "is_navigational": False},
        "mixed": {"type": "mixed", "main": [{"type": "web", "index": 0, "all": False}]},
        "web": {"type": "search", "results": list(rows), "family_friendly": True},
    }


#: Three plain results in rank order (titles, urls and snippets written out by hand).
THREE_ROWS: Final = (
    brave_row("Eiffel Tower - height", "https://a.example.org/eiffel", "The tower is 330 m tall."),
    brave_row("Eiffel Tower facts", "https://b.example.org/facts", "Built in 1889 for the fair."),
    brave_row("Paris landmarks", "http://c.example.org/paris", "The Eiffel Tower and more."),
)
#: What ``THREE_ROWS`` must map to: ``(title, url, snippet)`` in rank order.
THREE_TRIPLES: Final = (
    ("Eiffel Tower - height", "https://a.example.org/eiffel", "The tower is 330 m tall."),
    ("Eiffel Tower facts", "https://b.example.org/facts", "Built in 1889 for the fair."),
    ("Paris landmarks", "http://c.example.org/paris", "The Eiffel Tower and more."),
)


def brave_json(body: Any, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or None)


def brave_bytes(content: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=content, headers={"content-type": "application/json"})


def brave_status(status: int, body: bytes = b'{"type":"ErrorResponse"}') -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "application/json"})


def brave_redirect(status: int = 302, location: str = "") -> httpx.Response:
    loc = location or "https://elsewhere.example.org/res/v1/web/search?q=x"
    return httpx.Response(status, headers={"location": loc}, content=b"")


def oversize_body() -> bytes:
    """Valid JSON (three results) padded with whitespace to one byte over ``MAX_BODY``: a backend that parses
    before it bounds would accept it."""
    core = json.dumps(brave_body(THREE_ROWS)).encode()
    return core + b" " * (MAX_BODY + 1 - len(core))


def at_bound_body() -> bytes:
    """Valid JSON (three results) padded to exactly ``MAX_BODY`` bytes: not over the bound, so it is parsed."""
    core = json.dumps(brave_body(THREE_ROWS)).encode()
    return core + b" " * (MAX_BODY - len(core))


class _BothStreams(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body sent in chunks with no ``Content-Length``, readable by a sync or an async client."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = list(chunks)

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def chunked(content: bytes, chunk: int = 64 * 1024) -> httpx.Response:
    """``content`` streamed in chunks without a ``Content-Length`` header."""
    parts = [content[i : i + chunk] for i in range(0, len(content), chunk)]
    return httpx.Response(
        200, stream=_BothStreams(parts), headers={"content-type": "application/json"}
    )


class _Transport(httpx.MockTransport):
    """``MockTransport`` with both halves overridden, so ``BLOCK`` blocks correctly on either path."""

    def __init__(self, fake: FakeBrave) -> None:
        super().__init__(lambda request: httpx.Response(500))
        self._fake = fake

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        reply = self._fake._take(request)
        if isinstance(reply, str) and reply == BLOCK:
            self._fake.release.wait(timeout=self._fake.block_timeout)
            return brave_json(brave_body([]))
        return self._fake._answer(reply, request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        reply = self._fake._take(request)
        if isinstance(reply, str) and reply == BLOCK:
            deadline = time.monotonic() + self._fake.block_timeout
            while not self._fake.release.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            return brave_json(brave_body([]))
        return self._fake._answer(reply, request)


class FakeBrave:
    """The scripted API: ``replies`` in order (the last repeats); every request recorded."""

    def __init__(self, *replies: Any, block_timeout: float = 5.0) -> None:
        self.replies: list[Any] = list(replies) or [brave_json(brave_body(THREE_ROWS))]
        self.requests: list[httpx.Request] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_timeout = block_timeout
        self._lock = threading.Lock()

    def transport(self) -> httpx.MockTransport:
        return _Transport(self)

    def _take(self, request: httpx.Request) -> Any:
        with self._lock:
            self.requests.append(request)
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        self.entered.set()
        return reply

    @staticmethod
    def _answer(reply: Any, request: httpx.Request) -> httpx.Response:
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, httpx.Response):
            # a fresh copy per request (a Response's stream can be read once)
            if isinstance(reply.stream, _BothStreams):
                return httpx.Response(
                    reply.status_code,
                    stream=_BothStreams(reply.stream._chunks),
                    headers=reply.headers,
                    request=request,
                )
            return httpx.Response(
                reply.status_code, content=reply.content, headers=reply.headers, request=request
            )
        raise TypeError(f"FakeBrave: unknown reply {reply!r}")
