"""T-0053 group B — the store does not recall on a stopword (criteria 12-17).

The origin. On 2026-09-16 a fresh session with nothing shared asked "Make a flat looking UI." and
the prompt carried three captions of an hour-old calculus textbook. They were *retrieved*: the
query matched the store on the word **"a"**, every hit scored the same six-decimal-places-of-zero,
and the ``(score, ts_ms, id)`` tie-break then handed back the three newest episodes. For a query
like that, ``search`` had degenerated into "the three newest episodes", whatever was asked.

Measured before the change (this box, ``gpu-box``, python 3.11.16, sqlite 3.53.1,
``SqliteMemory`` at ``d12ddca``, fresh store per corpus, ``MemoryQuery(k=3)``, n=1 per query, no
second writer):

===========================  ====================  ==========  ===============
query                        corpus                hits        matched
===========================  ====================  ==========  ===============
``Make a flat looking UI.``  3 origin captions     3 @ ~1.6e-6  ``('a',)``
``what is the capital of…``  3 origin captions     3 @ ~1.0e-6  ``('of',)``
``Make a flat looking UI.``  400 fixture captions  3 @ ~1.6e-6  ``('a',)``
``desk``                     3 fixture captions    3 @ ~1.1e-6  ``('desk',)``
``desk``                     400 fixture captions  3 @ 4.49      ``('desk',)``
===========================  ====================  ==========  ===============

Two things in that table matter for how the rule is written. The task file says those scores are
"exactly 0.0"; they are not, they are ~1.6e-6 — bm25 clamps the IDF of a term in every document to
a hair above zero rather than to zero, so **a rule that rejected a hit at score 0.0 would not have
prevented the origin failure at all**. And the last two rows are the same three episodes, four
million times apart in score, with nothing changed but the size of the corpus: a score floor means
a different thing at n=3 and n=400, which is why the rule has to be document-frequency based
(criterion 14) and why :func:`test_a_term_is_dropped_where_it_is_universal_and_kept_where_it_is_rare`
is the row a score floor cannot pass.

Names this module fixes (see ``tests/unit/test_prompt_recollection.py`` for why the test-author
fixes them): ``local_astra.memory.store.MAX_TERM_DOC_FRACTION`` — the selectivity constant, a
fraction of the corpus above which a matched term carries no information and cannot, on its own,
justify a hit. A different spelling is a finding for the reviewer, not a silent rename. Ruling A16
adds a second name the rule cannot be stated without: ``store._MIN_UNINFORMATIVE_DOCS`` — the
document-count floor under the fraction, so that the whole condition reads
``drop iff df >= max(MAX_TERM_DOC_FRACTION * N, _MIN_UNINFORMATIVE_DOCS)``. It is a statement about
the *term* and not an exemption for small stores: a term one episode carries is the most selective
a term can be — it names that episode and excludes every other — so it survives at n=1 and at
n=400 alike, while a term every episode carries survives at neither.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import Source
from local_astra.memory import store as store_mod
from local_astra.memory.store import SqliteMemory

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "memory"
ORIGIN = json.loads((FIXTURES / "calculus_session.json").read_text())
CORPUS = json.loads((FIXTURES / "recall_corpus.json").read_text())

#: Criterion 15's stated bound: statements the *store* issues per ``search`` call (FTS5's own
#: internal statements, which sqlite marks with a leading ``--`` in the trace, are not the store's
#: and are not counted). Today's implementation issues 1. The bound exists to forbid the obvious
#: wrong shape — one document-frequency probe per query term — which an eight-word query would
#: turn into at least eight.
MAX_STATEMENTS_PER_SEARCH = 4


def _selectivity_constant() -> float:
    value = getattr(store_mod, "MAX_TERM_DOC_FRACTION", None)
    assert value is not None, (
        "T-0053 criterion 14: store.MAX_TERM_DOC_FRACTION does not exist. The selectivity rule is "
        "document-frequency based and its constant is a module-level name the tests read."
    )
    return float(value)


def _floor_constant() -> int:
    value = getattr(store_mod, "_MIN_UNINFORMATIVE_DOCS", None)
    assert value is not None, (
        "T-0053 ruling A16: store._MIN_UNINFORMATIVE_DOCS does not exist. The drop condition is "
        "df >= max(MAX_TERM_DOC_FRACTION * N, _MIN_UNINFORMATIVE_DOCS) and both constants are "
        "module-level names the tests read; a different spelling is a finding, not a rename."
    )
    return int(value)


async def _ingest(store: SqliteMemory, rows, session_id: str | None = None) -> dict[str, int]:
    """Ingest ``{"key", "ts_ms", "caption"}`` rows; returns key → episode id."""
    ids = {}
    for row in rows:
        fields = dict(
            ts_ms=row["ts_ms"],
            source=Source.SCREEN,
            thumbnail_path=f"{row['key']}.jpg",
            phash=f"{abs(hash(row['key'])):016x}"[:16],
            caption=row["caption"],
        )
        if session_id is not None:
            fields["session_id"] = session_id
        ids[row["key"]] = await store.ingest(Episode(**fields))
    return ids


def _origin_rows() -> list[dict[str, object]]:
    return [
        {"key": f"origin{row['db_id']}", "ts_ms": row["ts_ms"], "caption": row["caption"]}
        for row in ORIGIN["episodes"]
    ]


def _corpus_rows(size: int) -> list[dict[str, object]]:
    """The hand-written 40, optionally extended with filler; ``size=3`` is the ``small_subset``."""
    if size == 3:
        return [row for row in CORPUS["episodes"] if row["key"] in CORPUS["small_subset"]]
    rows = list(CORPUS["episodes"]) + list(CORPUS["filler"])
    assert size <= len(rows)
    return rows[:size]


@pytest.fixture
def store(tmp_path):
    return SqliteMemory(tmp_path / "memory.sqlite")


# --- criterion 12: the failing query recalls nothing ---------------------------------------------


async def test_a_query_matching_only_a_universal_term_recalls_nothing(store):
    """Criterion 12, on the origin fixture. Both queries return three hits today."""
    await _ingest(store, _origin_rows())
    for text in ("Make a flat looking UI.", "what is the capital of France"):
        hits = await store.search(MemoryQuery(text=text))
        assert list(hits) == [], (
            f"criterion 12: {text!r} shares only a term every episode carries; before the change "
            f"it returned {len(hits)} hits at scores {[h.score for h in hits]} on "
            f"{[h.matched for h in hits]}"
        )


def _uncaptioned(ts_ms: int, phash: str) -> Episode:
    """Exactly what ``Session.commit_history`` writes at the end of a turn that had a frame.

    The pixels and the perceptual hash are stored immediately; the caption is written *later* by
    the idle captioner, which is why ``next_uncaptioned`` exists at all. ``window_title`` comes
    from ``Frame.title``, which is ``str | None`` and is ``None`` for a pixel-first capture with
    the DOM assists off — the default. Such a row indexes the empty string in both FTS columns and
    can therefore never match any term, in any query, ever.
    """
    return Episode(
        ts_ms=ts_ms,
        source=Source.SCREEN,
        thumbnail_path=f"thumbnails/{phash}.jpg",
        phash=phash,
        window_title=None,
        app="chromium",
    )


async def test_the_captioner_backlog_does_not_dilute_the_denominator(store):
    """Ruling A30 (blocker): criterion 12 on the store's *normal* state, not a synthetic one.

    Every fixture in this task captions every episode it ingests. The real store never looks like
    that: ``commit_history`` ingests one caption-less episode per turn-with-a-frame and the idle
    captioner drains that backlog only while the session is idle, so a store in use always holds
    some. Those rows index the empty string and can match no term — they are not part of the
    population a document frequency is measured over — but the rule's denominator counted them, so
    each one raised the bar the numerator had to clear. Hand-counted below: ``a`` is carried by 3
    of the 3 captioned episodes and by neither uncaptioned row, so with the denominator at 4 the
    ceiling is ``0.9 * 4 = 3.6`` and ``df = 3`` *keeps* the term — the 2026-09-16 failure returns
    intact, three hits on ``('a',)``. With the denominator restricted to the population the
    numerator is drawn from it is ``0.9 * 3 = 2.7`` and the term goes. Measured on a copy of the
    human's own database (15 rows, 11 captioned, 4 uncaptioned): 3 hits, ids 7, 4, 11 — the three
    origin captions, on the exact database the bug came from.

    The second assertion is the one that says the denominator *ignores* the backlog rather than
    happening to survive one row of it: ten uncaptioned episodes are ten times the dilution and
    must change nothing at all.
    """
    await _ingest(store, _origin_rows())
    newest = max(int(row["ts_ms"]) for row in _origin_rows())
    pending = await store.ingest(_uncaptioned(newest + 1_000, "f" * 16))
    backlog = await store.next_uncaptioned()
    assert backlog is not None and backlog.id == pending and backlog.caption is None, (
        "the fixture must be a captioner backlog row — an episode the idle captioner has yet to "
        "describe — or this row is not testing ruling A30"
    )

    hits = await store.search(MemoryQuery(text="Make a flat looking UI."))
    assert list(hits) == [], (
        "ruling A30: one caption-less episode — what every turn with a frame writes — must not "
        "put 'a' back in the query. A caption-less row indexes nothing and can match nothing, so "
        f"it is not part of the population the frequency is measured over. Got {len(hits)} hits "
        f"on {[h.matched for h in hits]}"
    )

    for n in range(9):
        await store.ingest(_uncaptioned(newest + 2_000 + n, f"{n:016x}"))
    hits = await store.search(MemoryQuery(text="Make a flat looking UI."))
    assert list(hits) == [], (
        "ruling A30: ten uncaptioned episodes against three captioned ones is the backlog after a "
        "long session. The denominator must ignore them entirely, not merely tolerate one: above "
        f"one uncaptioned row per nine captioned no term can ever be dropped. Got {len(hits)} "
        f"hits on {[h.matched for h in hits]}"
    )


# --- criterion 13: the guard is not too wide -----------------------------------------------------


async def test_a_single_content_term_still_recalls_the_right_episode(store):
    """Criterion 13, the discriminating row: a rule phrased as "at least two matched terms" passes
    criterion 12 and fails here. Both queries match on exactly **one** content term — the fixture's
    generator asserts that no other word of either query occurs anywhere in the 40 captions.

    The first assertion is the one that is red today (40 captions, all containing "a", so the
    stopword query returns the three newest); the flight and glasses assertions pass today and are
    the direction that catches a guard silencing memory — docs/01 S4, docs/03 E4, the Astra moment.
    """
    ids = await _ingest(store, _corpus_rows(40))
    rejected = await store.search(MemoryQuery(text=CORPUS["queries"]["rejected"]))
    assert list(rejected) == [], (
        f"criterion 13: {CORPUS['queries']['rejected']!r} matches all 40 captions on 'a' alone; "
        f"it returned {[h.matched for h in rejected]}"
    )

    flight = await store.search(MemoryQuery(text=CORPUS["queries"]["flight"]))
    assert flight, "criterion 13: 'what time was that flight' must still recall the flight episode"
    assert flight[0].episode.id == ids[CORPUS["flight_key"]], (
        "criterion 13: the flight confirmation ranks first; it matched on the single term 'flight'"
    )
    glasses = await store.search(MemoryQuery(text=CORPUS["queries"]["glasses"]))
    assert glasses, "criterion 13: 'where did I leave my glasses' must still recall the glasses"
    assert glasses[0].episode.id == ids[CORPUS["glasses_key"]]


# --- criterion 14: document frequency, not a score floor -----------------------------------------


@pytest.mark.parametrize("size", [3, 400])
async def test_the_rule_holds_at_two_corpus_sizes(tmp_path, size):
    """Criterion 14: the same query, the same verdict, two corpus sizes two orders apart.

    ``a`` is in 3/3 and in 400/400 of the fixture (hand-counted in the MANIFEST), ``flight`` in 1/3
    and 1/400. A bm25 score floor cannot state this rule: the *same three episodes* score 1.1e-6 at
    n=3 and 4.49 at n=400 (module docstring).
    """
    store = SqliteMemory(tmp_path / f"memory-{size}.sqlite")
    ids = await _ingest(store, _corpus_rows(size))
    rejected = await store.search(MemoryQuery(text=CORPUS["queries"]["rejected"]))
    assert list(rejected) == [], (
        f"criterion 14: the stopword query is rejected at every corpus size; at n={size} it "
        f"returned {[h.matched for h in rejected]}"
    )
    flight = await store.search(MemoryQuery(text=CORPUS["queries"]["flight"]))
    assert flight and flight[0].episode.id == ids[CORPUS["flight_key"]], (
        f"criterion 14: a single rare content term still recalls at n={size}"
    )
    constant = _selectivity_constant()
    assert 0.5 < constant <= 1.0, (
        "criterion 14: the constant is a fraction of the corpus; at or below one half it would "
        f"reject a term carried by half the episodes, which silences recall in a small store "
        f"(docs/01 S4). Got {constant}"
    )
    floor = _floor_constant()
    assert floor >= 2, (
        "ruling A16: under the fraction sits a document-count floor, so that a term carried by a "
        "single episode — the most selective a term can be — is never dropped. At or below 1 the "
        f"fraction alone would drop a one-episode store's only term (0.9 * 1 = 0.9). Got {floor}"
    )


async def test_a_term_is_dropped_where_it_is_universal_and_kept_where_it_is_rare(tmp_path):
    """Criterion 14, adversarial — the row a score floor cannot pass.

    One word, ``desk``: in the three-episode subset it is in 3/3 and carries no information; in the
    400-episode corpus the same three episodes are the only ones that carry it, 3/400. Measured
    today: 1.1e-6 and 4.49 for the *same rows*. Any rule keyed on the score must either reject both
    (and lose the Astra moment at small corpus sizes) or accept both (and reproduce tonight's bug);
    only a document-frequency rule gives the two different answers this row demands.
    """
    small = SqliteMemory(tmp_path / "small.sqlite")
    await _ingest(small, _corpus_rows(3))
    assert list(await small.search(MemoryQuery(text="desk"))) == [], (
        "criterion 14: in this corpus every episode is a desk; the word distinguishes nothing"
    )

    big = SqliteMemory(tmp_path / "big.sqlite")
    ids = await _ingest(big, _corpus_rows(400))
    hits = await big.search(MemoryQuery(text="desk", k=3))
    assert {h.episode.id for h in hits} == {ids[key] for key in CORPUS["small_subset"]}, (
        "criterion 14: at 400 episodes 'desk' is rare and must recall exactly the three desks"
    )


async def test_a_one_episode_store_recalls_its_episode_on_a_term_only_it_carries(tmp_path):
    """Ruling A16, the row that pins ``_MIN_UNINFORMATIVE_DOCS``.

    The fraction alone says the wrong thing at the bottom of the range: in a store holding one
    episode every term that episode carries is carried by 1/1 = 100 % of the corpus, so a rule that
    were only ``df >= 0.9 * N`` would drop every one of them and the store would answer nothing,
    ever, until a second episode arrived. That is backwards — a term one episode carries names that
    episode and excludes every other, which is the most selective a term can be. Hand-counted here:
    "flight" is in 1 of 1 documents and "lisbon" in 1 of 1, and the correct verdict for both is
    *keep*. The query's other words ("what", "time", "was", "that") are in 0 of 1 and are kept by
    either reading, so the hit below can only come from the floor.
    """
    store = SqliteMemory(tmp_path / "one.sqlite")
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
        "ruling A16: a term carried by the single episode in the store is the most selective term "
        f"there is and must not be dropped; the store recalled {[h.matched for h in hits]}"
    )
    assert "flight" in hits[0].matched, (
        "ruling A16: the recall must be *on* the one-episode term, not on some other word"
    )


async def test_a_universal_term_is_dropped_from_a_query_that_also_has_a_rare_one(tmp_path):
    """Criterion 12/14: the rule drops the *term*, not the whole query.

    Four captions, hand-counted document frequencies: ``a`` 4/4, ``photograph`` 2/4, ``aquarium``
    2/4. Measured today, this box: ``"a photograph"`` returns all four rows — the two that carry
    ``photograph`` at 2.4e-6 and the two that carry only ``a`` at 1.0e-6 — because bm25 clamps the
    IDF of *both* terms to ~0 once a term is in half the documents. The score cannot separate them;
    the document frequency can, and the answer must be the two photographs.
    """
    store = SqliteMemory(tmp_path / "four.sqlite")
    rows = [
        {
            "key": "p1",
            "ts_ms": 1_000,
            "caption": "A photograph of an aquarium with three angelfish.",
        },
        {"key": "p2", "ts_ms": 2_000, "caption": "A photograph of an aquarium being cleaned."},
        {"key": "r1", "ts_ms": 3_000, "caption": "A receipt for groceries with bread and cider."},
        {"key": "w1", "ts_ms": 4_000, "caption": "A weather warning for strong wind overnight."},
    ]
    ids = await _ingest(store, rows)
    hits = await store.search(MemoryQuery(text="a photograph", k=10))
    assert {h.episode.id for h in hits} == {ids["p1"], ids["p2"]}, (
        "criterion 12/14: 'a' is in 4/4 and justifies nothing; 'photograph' is in 2/4 and justifies "
        f"two hits. Got {[h.matched for h in hits]}"
    )


# --- criterion 15: search stays cheap ------------------------------------------------------------


async def test_search_executes_a_bounded_number_of_statements(tmp_path, monkeypatch):
    """Criterion 15, counted deterministically — never timed in the fast suite.

    The eight-word query is the point: a document-frequency probe issued once per term would cost
    at least eight statements, and B10's "≤ 50 ms at 100k episodes" would then be a promise made
    per word rather than per turn. (The 100k measurement itself is a one-off for the PR body.)
    """
    statements: list[str] = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = SqliteMemory(tmp_path / "counted.sqlite")
    await _ingest(store, _corpus_rows(40))

    query = CORPUS["queries"]["rejected"]
    assert len(query.split()) >= 4
    statements.clear()
    hits = await store.search(MemoryQuery(text=query))
    issued = [s for s in statements if not s.lstrip().startswith("--")]
    assert list(hits) == [], "criterion 12 again, so this row cannot pass on today's behaviour"
    assert 1 <= len(issued) <= MAX_STATEMENTS_PER_SEARCH, (
        f"criterion 15: {len(issued)} statements for one search: {issued}"
    )


# --- criterion 16: the migration -----------------------------------------------------------------

#: The schema as it exists on the human's disk today (``d12ddca``), copied here on purpose: this is
#: a *legacy* artefact, so it must not track future edits of ``store._SCHEMA``.
LEGACY_SCHEMA = (
    """
    CREATE TABLE episodes (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_ms          INTEGER NOT NULL,
        source         TEXT    NOT NULL,
        thumbnail_path TEXT    NOT NULL,
        phash          TEXT    NOT NULL,
        window_title   TEXT,
        app            TEXT,
        caption        TEXT
    )
    """,
    "CREATE INDEX episodes_ts_ms ON episodes(ts_ms)",
    "CREATE VIRTUAL TABLE episodes_fts USING fts5(caption, window_title, tokenize='porter unicode61')",
)

#: Non-contiguous ids, as the human's store has them: a migration that copies rows into a new table
#: can renumber them, and contiguous ids would hide it.
LEGACY_ROWS = (
    (7, 1_757_000_000_000, "screen", "/thumbs/7.jpg", "a" * 16, None, None, "A calculus page."),
    (
        4,
        1_757_000_060_000,
        "screen",
        "/thumbs/4.jpg",
        "b" * 16,
        None,
        None,
        "A harbour photograph.",
    ),
    (11, 1_757_000_120_000, "screen", "/thumbs/11.jpg", "c" * 16, None, None, "A soup recipe."),
)


def _legacy_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    for statement in LEGACY_SCHEMA:
        conn.execute(statement)
    for row in LEGACY_ROWS:
        conn.execute(
            "INSERT INTO episodes"
            " (id, ts_ms, source, thumbnail_path, phash, window_title, app, caption)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        conn.execute(
            "INSERT INTO episodes_fts (rowid, caption, window_title) VALUES (?, ?, ?)",
            (row[0], row[7], ""),
        )
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()


def _rows(path: Path) -> list[tuple]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT id, ts_ms, source, thumbnail_path, phash, window_title, app, caption"
            " FROM episodes ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _user_version(path: Path) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


async def test_an_existing_database_migrates_and_keeps_every_row(tmp_path):
    """Criterion 16, adversarially. This is the human's real database.

    A migration that drops ``episodes`` and recreates it with the new column passes a naive
    "does the column exist" check. It fails here: the row identities, the AUTOINCREMENT sequence,
    the FTS index and a row written between two opens all have to survive.
    """
    path = tmp_path / "memory.sqlite"
    _legacy_database(path)
    before = _rows(path)
    assert len(before) == 3

    store = SqliteMemory(path)
    assert "session_id" in Episode.model_fields, (
        "criterion 16/17: Episode gains a nullable session_id"
    )
    columns = {row[1]: row for row in sqlite3.connect(path).execute("PRAGMA table_info(episodes)")}
    assert "session_id" in columns, "criterion 16: the column exists after the first open"
    assert _rows(path) == before, "criterion 16: every legacy row survives, byte for byte"
    assert _user_version(path) == 1, "criterion 16: PRAGMA user_version moves 0 -> 1"

    found = await store.search(MemoryQuery(text="harbour"))
    assert [h.episode.id for h in found] == [4], (
        "criterion 16: a migrated row is still findable through the FTS index"
    )
    assert found[0].episode.session_id is None, "criterion 17: legacy rows have no session"

    new_id = await store.ingest(
        Episode(
            ts_ms=1_757_000_180_000,
            source=Source.SCREEN,
            thumbnail_path="/thumbs/new.jpg",
            phash="d" * 16,
            caption="A crossword puzzle half completed.",
        )
    )
    assert new_id == 12, (
        "criterion 16: the AUTOINCREMENT sequence survives the migration (max legacy id is 11); "
        f"a rebuilt table restarts it, and this store got {new_id}"
    )
    store.close()

    second = SqliteMemory(path)
    assert _user_version(path) == 1 and len(_rows(path)) == 4
    assert [row[0] for row in _rows(path)] == [4, 7, 11, 12], "criterion 16: no row is renumbered"
    second.close()

    # A row written between two opens: a second migration pass would drop or rebuild it.
    sentinel = sqlite3.connect(path)
    sentinel.execute(
        "INSERT INTO episodes (id, ts_ms, source, thumbnail_path, phash, caption)"
        " VALUES (99, 1757000240000, 'screen', '/thumbs/99.jpg', 'eeeeeeeeeeeeeeee', 'A sentinel bicycle.')"
    )
    sentinel.execute(
        "INSERT INTO episodes_fts (rowid, caption, window_title) VALUES (99, 'A sentinel bicycle.', '')"
    )
    sentinel.commit()
    sentinel.close()

    third = SqliteMemory(path)
    assert _user_version(path) == 1, "criterion 16: a database already at 1 is not re-migrated"
    assert [row[0] for row in _rows(path)] == [4, 7, 11, 12, 99]
    again = await third.search(MemoryQuery(text="sentinel"))
    assert [h.episode.id for h in again] == [99], "criterion 16: the third open is a no-op"
    third.close()


# --- criterion 17: session_id records, it does not rank ------------------------------------------


async def test_session_id_round_trips_and_does_not_change_ranking(tmp_path):
    """Criterion 17: the column records which conversation saw the episode; it never filters."""
    assert "session_id" in Episode.model_fields, "criterion 17: Episode gains a nullable session_id"

    with_ids = SqliteMemory(tmp_path / "with.sqlite")
    without = SqliteMemory(tmp_path / "without.sqlite")
    rows = _corpus_rows(40)
    keyed = await _ingest(with_ids, rows, session_id="s-2026-09-16")
    bare = await _ingest(without, rows)

    hits = await with_ids.search(MemoryQuery(text=CORPUS["queries"]["flight"]))
    assert hits and hits[0].episode.session_id == "s-2026-09-16", "criterion 17: it round-trips"
    plain = await without.search(MemoryQuery(text=CORPUS["queries"]["flight"]))
    assert plain and plain[0].episode.session_id is None, "criterion 17: absent stays None"

    inverse = {v: k for k, v in keyed.items()}
    inverse_bare = {v: k for k, v in bare.items()}
    for query in (CORPUS["queries"]["flight"], CORPUS["queries"]["glasses"], "a photograph"):
        a = await with_ids.search(MemoryQuery(text=query, k=10))
        b = await without.search(MemoryQuery(text=query, k=10))
        assert [inverse[h.episode.id] for h in a] == [inverse_bare[h.episode.id] for h in b], (
            f"criterion 17: {query!r} ranks differently once the episodes carry a session id"
        )
        assert [h.score for h in a] == [h.score for h in b]
