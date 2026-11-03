"""Tool execution contracts (T-0091 criterion 7; T-0080d settled default 1).

A ``ToolInvocation`` is one v1 call the turn decided to run, with the geometry of the newest image attached to the
turn's prompt (``None`` when no image was attached); a ``ToolOutcome`` is what a ``ToolExecutor``
(``contracts/protocols.py``) made of it. In v1 the only executor is the dry run, so the status is ``dry_run`` or
``refused``; ``executed`` and ``failed`` are reserved for a later, separately designed executor.

``SearchResults`` is what a ``SearchBackend`` (``contracts/protocols.py``; T-0113) answered for one ``google_search``
query: every result it returned, in rank order, unbounded and unsanitised. The loop's renderer
(``toolcall/loop.py``) applies the bounds and the special-literal sanitiser.

Pydantic only: importing this module never pulls ``local_astra.decoding``, the eval, numpy, torch, outlines or
tokenizers (``tests/unit/test_toolcall_imports.py``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from local_astra.contracts.base import Contract
from local_astra.contracts.perception import Source

__all__ = [
    "FrameGeometry",
    "InvocationSource",
    "OutcomeStatus",
    "SearchResult",
    "SearchResults",
    "ToolInvocation",
    "ToolOutcome",
]

#: ``draft``: pass 1's own call; ``repair``: the H-T84-2 seeded repair's call.
InvocationSource = Literal["draft", "repair"]
OutcomeStatus = Literal["dry_run", "executed", "refused", "failed"]


class FrameGeometry(Contract):
    """The image a call's 0-999 coordinates are normalised to: its perceptual hash, source and pixel size."""

    phash: str | None
    source: Source
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ToolInvocation(Contract):
    call_id: str
    turn_id: str
    name: str
    arguments: dict[str, Any]
    source: InvocationSource
    frame: FrameGeometry | None


class ToolOutcome(Contract):
    call_id: str
    status: OutcomeStatus
    #: a human-readable description of what was (or would have been) done
    action: str
    #: why it was refused or failed; ``None`` otherwise
    reason: str | None
    #: what the executor resolved the arguments to (``x_px``/``y_px`` for a click or a type)
    resolved: dict[str, Any]


class SearchResult(Contract):
    """One search hit, as the backend gave it."""

    title: str
    url: str
    snippet: str


class SearchResults(Contract):
    """A backend's answer to ``query``: as many results as it returned, best first (the renderer keeps five)."""

    query: str
    results: tuple[SearchResult, ...]
