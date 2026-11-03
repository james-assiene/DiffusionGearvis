"""T-0053 criterion 18 — the fake obeys the same recall rule as the real store.

**Deviation from the ``backend=["fake", "real"]`` convention, on purpose.** There is no memory
*service*: the "real" implementation is ``SqliteMemory``, a file on this box, so both arms run in
the fast suite and neither is marked ``gpu``. ``.claude/rules/tests.md`` asks that "a behaviour
added to a fake needs a real-service test in the same PR" — that is satisfied by running both arms
here, and the parametrisation is spelled ``store=["fake", "sqlite"]`` so nobody reads a missing
``gpu`` marker as an oversight.

The two implementations are held to the same *criteria*, not to the same numbers: FTS5 stems with
``porter`` and the fake does not, so their scores differ and their constants may differ. What may
not differ is the answer to "does this query recall anything, and is it the right episode"
(criteria 12 and 13) — because the session hands the fake to the fast suite and the real store to
the human, and a rule present in only one of them is a rule that is not tested where it matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import Source
from local_astra.memory import store as store_mod
from local_astra.memory.store import SqliteMemory
from tests.fakes import FakeMemory
from tests.fakes import memory as fake_mod

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "memory"
ORIGIN = json.loads((FIXTURES / "calculus_session.json").read_text())
CORPUS = json.loads((FIXTURES / "recall_corpus.json").read_text())


@pytest.fixture(params=["fake", "sqlite"])
def store(request, tmp_path):
    if request.param == "fake":
        return FakeMemory()
    return SqliteMemory(tmp_path / "memory.sqlite")


async def _ingest(store, rows) -> dict[str, int]:
    ids = {}
    for row in rows:
        ids[row["key"]] = await store.ingest(
            Episode(
                ts_ms=row["ts_ms"],
                source=Source.SCREEN,
                thumbnail_path=f"{row['key']}.jpg",
                phash="0" * 16,
                caption=row["caption"],
            )
        )
    return ids


async def test_a_query_matching_only_a_universal_term_recalls_nothing(store):
    """Criterion 12 and 18, against both implementations, on the origin captions."""
    rows = [
        {"key": f"origin{row['db_id']}", "ts_ms": row["ts_ms"], "caption": row["caption"]}
        for row in ORIGIN["episodes"]
    ]
    await _ingest(store, rows)
    for text in ("Make a flat looking UI.", "what is the capital of France"):
        hits = await store.search(MemoryQuery(text=text))
        assert list(hits) == [], (
            f"criterion 12/18: {type(store).__name__} recalled {len(hits)} episodes for {text!r} "
            f"on {[h.matched for h in hits]}"
        )


async def test_the_captioner_backlog_does_not_dilute_the_denominator_in_either_implementation(
    store,
):
    """Ruling A30, on both arms: criterion 12 on the store's normal state.

    ``Session.commit_history`` ingests one caption-less, title-less episode per turn that had a
    frame, and the idle captioner drains that backlog only while the session is idle. Such a row
    indexes nothing and can match no term, so it is not part of the population a document frequency
    is measured over — but both implementations counted it in the denominator. Hand-counted: ``a``
    is in 3 of the 3 captioned episodes and in neither uncaptioned one; a denominator of 4 gives a
    ceiling of ``0.9 * 4 = 3.6``, which keeps ``a`` and returns the 2026-09-16 failure verbatim,
    and a denominator of 3 gives ``2.7``, which drops it. The session hands the *fake* to the fast
    suite, so a denominator flaw present in only one of them is a flaw nothing would catch.
    """
    rows = [
        {"key": f"origin{row['db_id']}", "ts_ms": row["ts_ms"], "caption": row["caption"]}
        for row in ORIGIN["episodes"]
    ]
    await _ingest(store, rows)
    newest = max(int(row["ts_ms"]) for row in rows)
    for n in range(10):
        await store.ingest(
            Episode(
                ts_ms=newest + 1_000 + n,
                source=Source.SCREEN,
                thumbnail_path=f"thumbnails/pending{n}.jpg",
                phash=f"{n:016x}",
                window_title=None,
                app="chromium",
            )
        )
    hits = await store.search(MemoryQuery(text="Make a flat looking UI."))
    assert list(hits) == [], (
        f"ruling A30: {type(store).__name__} let a captioner backlog put 'a' back in the query. "
        f"A caption-less row indexes nothing and can match nothing, so it does not belong in the "
        f"denominator. Got {len(hits)} hits on {[h.matched for h in hits]}"
    )


async def test_a_single_content_term_still_recalls_the_right_episode(store):
    """Criterion 13 and 18: the guard must not be so wide that it silences memory."""
    ids = await _ingest(store, CORPUS["episodes"])
    rejected = await store.search(MemoryQuery(text=CORPUS["queries"]["rejected"]))
    assert list(rejected) == [], (
        f"criterion 12/18: {type(store).__name__} recalled on 'a' alone: {[h.matched for h in rejected]}"
    )
    flight = await store.search(MemoryQuery(text=CORPUS["queries"]["flight"]))
    assert flight and flight[0].episode.id == ids[CORPUS["flight_key"]], (
        f"criterion 13/18: {type(store).__name__} lost the flight episode"
    )
    glasses = await store.search(MemoryQuery(text=CORPUS["queries"]["glasses"]))
    assert glasses and glasses[0].episode.id == ids[CORPUS["glasses_key"]]


async def test_a_one_episode_store_recalls_its_episode_in_both_implementations(store):
    """Ruling A17 and A16, on both arms: the n = 1 corner, where the two used to disagree.

    Every term the single episode carries is carried by 1 of 1 documents, so the fraction on its
    own would drop all of them and a one-episode store would answer nothing until a second episode
    arrived. Both implementations owe the same floor under the fraction. The fake had none, so on
    2026-09-17 this corner answered ``[]`` in the fast suite and the episode on the human's disk —
    a divergence no row looked at, in the state every fresh store passes through.
    """
    ids = await _ingest(
        store,
        [
            {
                "key": "boarding",
                "ts_ms": 1_000,
                "caption": "A boarding pass for the 18:40 flight to Lisbon.",
            }
        ],
    )
    hits = await store.search(MemoryQuery(text="what time was that flight"))
    assert [h.episode.id for h in hits] == [ids["boarding"]], (
        f"ruling A16/A17: {type(store).__name__} dropped a term carried by the only episode in "
        f"the store, which is the most selective a term can be; got {[h.matched for h in hits]}"
    )
    assert "flight" in hits[0].matched, (
        f"ruling A16/A17: {type(store).__name__} recalled, but not on the one-episode term"
    )


def test_both_implementations_state_the_rule_as_a_document_frequency():
    """Criterion 14/18: both carry a named fraction, and neither is a hardcoded stop list.

    The numbers may differ (the tokenizers do); what may not differ is the shape of the rule.
    """
    fake_constant = getattr(fake_mod, "MAX_TERM_DOC_FRACTION", None)
    real_constant = getattr(store_mod, "MAX_TERM_DOC_FRACTION", None)
    assert fake_constant is not None, "tests/fakes/memory.py owes the constant (test-author's)"
    assert real_constant is not None, (
        "T-0053 criterion 14: local_astra/memory/store.py owes MAX_TERM_DOC_FRACTION"
    )
    for name, value in (("fake", fake_constant), ("sqlite", real_constant)):
        assert 0.5 < float(value) <= 1.0, f"{name}: {value} is not a corpus fraction"
    for name, module in (("fake", fake_mod), ("sqlite", store_mod)):
        floor = getattr(module, "_MIN_UNINFORMATIVE_DOCS", None)
        assert floor is not None and int(floor) >= 2, (
            f"{name}: ruling A16/A17 — under the fraction sits a document-count floor, so that a "
            f"term carried by a single episode is never dropped. Got {floor}"
        )
    for module in (fake_mod, store_mod):
        source = Path(module.__file__).read_text().lower()
        for stopword in (' "the"', " 'the'", ' "and"', " 'and'"):
            assert stopword not in source, (
                f"{module.__name__}: the rule is document frequency, never an English stop list "
                "(the tokenizer is porter unicode61 and the design is not English-only)"
            )
