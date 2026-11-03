"""``FallbackSearch``: a ``SearchBackend`` that asks a fallback backend only after its primary refuses (T-0163
criterion 7). It names neither backend; ``cli._search_backend`` builds the chain.

The fallback is asked **only** when the primary raises ``SearchFailed`` of kind ``error`` or ``rate_limited`` (a
provider refusal, including "no results" from an anti-bot page). Results (zero included), ``TimeoutError``,
``SearchFailed("network"|"closed")``, ``ValueError``, ``asyncio.CancelledError`` and any other exception propagate
unchanged, so a timeout, a dead network, a close or a cancel never reaches the fallback. The fallback's outcome is
final.

``last_engine`` is the answering backend's own ``last_engine`` (the primary's engine, or the fallback's), ``None``
after a failure and while a search runs.

``last_attempts`` (T-0176 criterion 8) is the primary's ``last_attempts`` followed by the fallback's when it was
asked: the engines that refused the most recent search, with their failure kinds; ``()`` before any search and
while one runs (the fallback included).
"""

from __future__ import annotations

from typing import Final

from local_astra.contracts.protocols import SearchBackend, SearchFailed
from local_astra.contracts.toolcall import SearchResults

__all__ = ["FallbackSearch"]

#: The refusals after which the fallback is asked.
_FALL_THROUGH: Final = frozenset({"error", "rate_limited"})


def _engine(backend: SearchBackend) -> str | None:
    engine = getattr(backend, "last_engine", None)
    return engine if isinstance(engine, str) else None


def _attempts(backend: SearchBackend) -> tuple[tuple[str, str], ...]:
    attempts = getattr(backend, "last_attempts", None)
    return tuple(attempts) if isinstance(attempts, tuple) else ()


class FallbackSearch:
    """``primary`` first; ``fallback`` only after a refusal (see the module docstring)."""

    def __init__(self, primary: SearchBackend, fallback: SearchBackend) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_engine: str | None = None
        self.last_attempts: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        return f"FallbackSearch({type(self._primary).__name__}, {type(self._fallback).__name__})"

    async def search(self, query: str) -> SearchResults:
        self.last_engine = None
        self.last_attempts = ()
        try:
            results = await self._primary.search(query)
        except SearchFailed as exc:
            primary = _attempts(self._primary)
            if exc.kind not in _FALL_THROUGH:
                self.last_attempts = primary
                raise
        except Exception:
            self.last_attempts = _attempts(self._primary)
            raise
        else:
            self.last_engine = _engine(self._primary)
            self.last_attempts = _attempts(self._primary)
            return results
        # outside the handler, so the fallback's own failure carries no context from the primary's
        try:
            results = await self._fallback.search(query)
        except Exception:
            self.last_attempts = primary + _attempts(self._fallback)
            raise
        self.last_engine = _engine(self._fallback)
        self.last_attempts = primary + _attempts(self._fallback)
        return results

    async def aclose(self) -> None:
        """Close both (each that has an ``aclose``), the fallback even if closing the primary raises."""
        try:
            await _aclose(self._primary)
        finally:
            await _aclose(self._fallback)


async def _aclose(backend: SearchBackend) -> None:
    closer = getattr(backend, "aclose", None)
    if closer is not None:
        await closer()
