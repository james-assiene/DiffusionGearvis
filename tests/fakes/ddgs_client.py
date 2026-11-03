"""A stand-in for ``ddgs.DDGS`` (T-0111b): canned rows, a kwargs log, real ``ddgs.exceptions``, a blocking switch.

``DdgsSearch(client_factory=...)`` builds its client by calling the factory the way it would call ``ddgs.DDGS``
(``DDGS(proxy=None, timeout=5, *, verify=True)`` at the pin, 9.16.1), then calls ``client.text(query, **kwargs)``
(``region``, ``safesearch``, ``max_results``, ``backend``, ...). The fake mirrors both signatures, so a call the
real class would refuse is refused here too; ``tests/unit/test_ddgs_backend.py::test_the_fake_mirrors_the_pinned_ddgs``
checks the mirror against the installed package (no network).

- ``rows``: what ``text()`` returns (a fresh list each call; rows are passed through as given, bad ones included);
- ``raise_``: an exception instance ``text()`` raises instead (use the real ``ddgs.exceptions`` classes, see
  ``ddgs_exceptions()``);
- ``block``: a ``threading.Event``; ``text()`` waits for it before answering. ``block_timeout`` is a hang guard
  for unit tests (``None`` = wait forever, the subprocess rows);
- ``entered`` / ``exited``: set when a ``text()`` call starts and when it returns or raises;
- every constructor call and every ``text()`` call is logged, with the thread it ran on.

The fake opens no socket, starts no thread and touches no file.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

#: Two plain rows, in rank order, in ddgs's text-result shape (``title``, ``href``, ``body``).
DEFAULT_DDGS_ROWS: tuple[dict[str, str], ...] = (
    {
        "title": "Paris weather today",
        "href": "https://weather.example.org/paris",
        "body": "Paris: 14C, light rain, wind 12 km/h from the west.",
    },
    {
        "title": "Paris forecast",
        "href": "https://forecast.example.org/fr/paris",
        "body": "Showers clearing by evening; 9C overnight.",
    },
)


def _ddgs_init_mirror(
    proxy: str | None = None, timeout: int | None = 5, *, verify: bool | str = True
) -> None:
    """``ddgs.DDGS.__init__`` at 9.16.1, without ``self``."""


def _ddgs_text_mirror(query: str, **kwargs: Any) -> None:
    """``ddgs.DDGS.text`` at 9.16.1, without ``self``."""


INIT_SIGNATURE = inspect.signature(_ddgs_init_mirror)
TEXT_SIGNATURE = inspect.signature(_ddgs_text_mirror)


def ddgs_exceptions() -> ModuleType:
    """The installed ``ddgs.exceptions`` (the ``search`` extra). Imported lazily: importing this fake never imports
    ``ddgs``."""
    import importlib

    return importlib.import_module("ddgs.exceptions")


@dataclass(frozen=True)
class TextCall:
    query: str
    kwargs: dict[str, Any]
    thread: threading.Thread


class FakeDdgsClient:
    """One client; built by ``FakeDdgsFactory.__call__`` with ``DDGS``'s arguments."""

    def __init__(self, factory: FakeDdgsFactory, args: tuple[Any, ...], kwargs: dict[str, Any]):
        #: as passed (no defaults applied); ``TypeError`` if ``DDGS`` would refuse them
        self.explicit: dict[str, Any] = dict(INIT_SIGNATURE.bind(*args, **kwargs).arguments)
        bound = INIT_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        #: with ``DDGS``'s defaults applied
        self.effective: dict[str, Any] = dict(bound.arguments)
        self._factory = factory
        self.text_calls: list[TextCall] = []

    def text(self, *args: Any, **kwargs: Any) -> list[Any]:
        bound = TEXT_SIGNATURE.bind(*args, **kwargs)
        f = self._factory
        call = TextCall(
            query=bound.arguments["query"],
            kwargs=dict(bound.arguments.get("kwargs", {})),
            thread=threading.current_thread(),
        )
        self.text_calls.append(call)
        f.text_calls.append(call)
        f.entered.set()
        try:
            if f.block is not None and not f.block.wait(timeout=f.block_timeout):
                f.guard_expired = True
            backend = call.kwargs.get("backend")
            if f.by_backend is not None and backend in f.by_backend:
                answer = f.by_backend[backend]
                if callable(answer):
                    answer = answer()
                if isinstance(answer, BaseException):
                    raise answer
                return [dict(r) if isinstance(r, dict) else r for r in answer]
            if f.raise_ is not None:
                raise f.raise_
            return [dict(r) if isinstance(r, dict) else r for r in f.rows]
        finally:
            f.exited.set()


@dataclass
class FakeDdgsFactory:
    """Callable like ``ddgs.DDGS``; every client it builds shares its configuration and logs."""

    rows: Sequence[Any] = DEFAULT_DDGS_ROWS
    raise_: BaseException | None = None
    block: threading.Event | None = None
    block_timeout: float | None = None
    #: per-engine answers, keyed by ``text(backend=...)``: rows, an exception instance to raise, or a zero-argument
    #: callable returning either (called on the search thread). Engines not listed get ``raise_`` / ``rows``.
    by_backend: Mapping[str, Any] | None = None
    clients: list[FakeDdgsClient] = field(default_factory=list)
    text_calls: list[TextCall] = field(default_factory=list)
    entered: threading.Event = field(default_factory=threading.Event)
    exited: threading.Event = field(default_factory=threading.Event)
    guard_expired: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> FakeDdgsClient:
        client = FakeDdgsClient(self, args, kwargs)
        self.clients.append(client)
        return client

    @property
    def blocked(self) -> bool:
        """A ``text()`` call has started and not yet returned."""
        return self.entered.is_set() and not self.exited.is_set()


def client_factory(
    rows: Sequence[Any] = DEFAULT_DDGS_ROWS,
    *,
    raise_: BaseException | None = None,
    block: threading.Event | None = None,
    block_timeout: float | None = None,
    by_backend: Mapping[str, Any] | None = None,
) -> FakeDdgsFactory:
    """A fresh factory to pass as ``DdgsSearch(client_factory=...)``."""
    return FakeDdgsFactory(
        rows=rows, raise_=raise_, block=block, block_timeout=block_timeout, by_backend=by_backend
    )
