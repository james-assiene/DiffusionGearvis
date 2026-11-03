"""Fake ``MemoryClient``: in-memory episodes with word-overlap search (stand-in for FTS5).

The selectivity rule (T-0053 criteria 12-14, 18) lives here too: a term carried by
:data:`MAX_TERM_DOC_FRACTION` or more of the stored episodes carries no information about which
episode is wanted, and cannot on its own justify a hit. Without it the fake reproduces the origin
bug — on 2026-09-16 ``"Make a flat looking UI."`` matched three calculus captions on the word
``"a"`` and the ranking fell through to recency, so ``search`` returned the three newest episodes
for any query at all.

The fake mirrors the *rule*, not the number: the real store's constant is
``local_astra.memory.store.MAX_TERM_DOC_FRACTION`` and the two are asserted to satisfy the same
criteria — not to be equal — by ``tests/contract/test_memory_recall_contract.py``, which runs both
implementations in the fast suite (there is no memory *service*, so no ``gpu`` marker; see that
module's docstring).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery

#: Document-frequency ceiling: a term in this fraction of the episodes or more is a stopword *for
#: this store*, whatever the language. Deliberately not a hardcoded English stop list.
MAX_TERM_DOC_FRACTION = 0.9

#: …with the same floor the real store carries (``store._MIN_UNINFORMATIVE_DOCS``, T-0053 ruling
#: A16): a term carried by a single episode names that episode and excludes every other, so it is
#: the most selective a term can be and is never dropped. Without it a one-episode fake store
#: recalls nothing while the real store recalls its episode — the fake and the store answering
#: differently in the corner the session hits first, which is the one divergence a fake may not
#: have. Asserted against the real store by ``tests/contract/test_memory_recall_contract.py``.
_MIN_UNINFORMATIVE_DOCS = 2

_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}


class FakeMemory:
    def __init__(self) -> None:
        self.episodes: dict[int, Episode] = {}
        self._next = 1

    async def ingest(self, episode: Episode) -> int:
        eid = self._next
        self._next += 1
        self.episodes[eid] = episode.model_copy(update={"id": eid})
        return eid

    async def set_caption(self, episode_id: int, caption: str) -> None:
        self.episodes[episode_id] = self.episodes[episode_id].model_copy(
            update={"caption": caption}
        )

    def _document(self, episode: Episode) -> set[str]:
        return _terms(f"{episode.caption or ''} {episode.window_title or ''}")

    def _informative(self, terms: set[str]) -> set[str]:
        """The query terms that still distinguish one episode from another (T-0053).

        Document frequency is counted over the **whole** store — the time filter selects rows, it
        does not change what a word is worth — but only over the rows a frequency can be *measured*
        on (ruling A30). ``Session.commit_history`` ingests one episode per turn-with-a-frame with
        no caption and no title; the idle captioner fills them in later, so that backlog is the
        store's normal state. Such a row indexes nothing and can match no term, so counting it in
        the denominator only raises the bar the numerator must clear: with U uncaptioned rows
        against C captioned, a term every caption carries survives at U > C/9 and the rule stops
        existing. The denominator is the population the numerator is drawn from.
        """
        documents = [
            self._document(ep)
            for ep in self.episodes.values()
            if ep.caption is not None or ep.window_title is not None
        ]
        if not documents:
            return set()
        ceiling = max(MAX_TERM_DOC_FRACTION * len(documents), _MIN_UNINFORMATIVE_DOCS)
        return {term for term in terms if sum(1 for doc in documents if term in doc) < ceiling}

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]:
        terms = self._informative(_terms(query.text))
        hits: list[MemoryHit] = []
        for ep in self.episodes.values():
            if query.t0_ms is not None and ep.ts_ms < query.t0_ms:
                continue
            if query.t1_ms is not None and ep.ts_ms > query.t1_ms:
                continue
            matched = tuple(sorted(terms & self._document(ep)))
            if matched:
                hits.append(MemoryHit(episode=ep, score=len(matched), matched=matched))
        hits.sort(key=lambda h: (-h.score, -h.episode.ts_ms))
        return hits[: query.k]

    async def next_uncaptioned(self) -> Episode | None:
        """The newest episode with no caption (``CaptionStore``, T-0009 ruling A1).

        ``SqliteMemory.next_uncaptioned`` orders ``ts_ms DESC, id DESC`` and the ``IdleCaptioner``
        is written against that order, so the fake owes the same one: the scene the user just
        looked at is the one they are about to ask about. Added by T-0007's test-author because
        the session hands the *concrete* store to ``IdleCaptioner`` and the fake is what the fast
        suite hands it (docs/tasks/T-0007.md, Ownership conflicts).
        """
        pending = [e for e in self.episodes.values() if e.caption is None]
        if not pending:
            return None
        return max(pending, key=lambda e: (e.ts_ms, e.id or 0))

    async def forget_since(self, t0_ms: int) -> int:
        doomed = [k for k, e in self.episodes.items() if e.ts_ms >= t0_ms]
        for k in doomed:
            del self.episodes[k]
        return len(doomed)
