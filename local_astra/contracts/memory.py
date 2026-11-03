"""Memory-lite contracts (docs/01 D8): SQLite + FTS5, captions written by the brain when idle."""

from __future__ import annotations

from pydantic import Field

from local_astra.contracts.base import Contract
from local_astra.contracts.perception import Source


class Episode(Contract):
    id: int | None = None
    ts_ms: int = Field(ge=0)
    source: Source
    thumbnail_path: str
    phash: str
    window_title: str | None = None
    app: str | None = None
    #: written later by the idle captioner; FTS5 indexes this column
    caption: str | None = None
    #: which conversation saw this episode (T-0053 criterion 17, D1 "session identity, recorded").
    #: It *records*; it never filters and never ranks — recall stays cross-session — so that a
    #: recollection can be labelled "earlier in this conversation" rather than dated only by age.
    #: ``None`` for every episode written before the column existed, and for any writer that does
    #: not know its session.
    session_id: str | None = None


class MemoryHit(Contract):
    episode: Episode
    score: float
    #: which query terms matched (for the HUD "why this memory" tooltip)
    matched: tuple[str, ...] = ()


class MemoryQuery(Contract):
    text: str
    t0_ms: int | None = None
    t1_ms: int | None = None
    k: int = Field(default=3, ge=1, le=10)
