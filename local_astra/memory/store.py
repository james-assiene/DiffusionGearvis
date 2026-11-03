"""SQLite + FTS5 episode store (docs/01 D8, docs/03 §4 "Memory"). Task T-0009.

Schema (``episodes`` + ``episodes_fts`` over ``caption``, ``window_title``): retrieval is FTS5
with a time filter and recency tie-break, top-3 thumbnails attached to the prompt after the
frame. Everything lives under ``~/.local/share/local_astra/`` and ``forget_since`` deletes rows
*and* thumbnails.

Three things worth knowing before changing this module:

- **The query text is a spoken transcript, never FTS5 syntax.** It arrives from Kyutai STT with
  apostrophes, quotes, hyphens and the occasional bare ``AND``; interpolating it into ``MATCH``
  either raises ``sqlite3.OperationalError`` or silently means something else. Queries are split
  on :data:`_WORD`, which keeps word characters only, and built by :func:`_match_expression`,
  which quotes every term — including the term a document-frequency probe asks about.
- **Ranking is decided here, not by SQLite's row order.** Rows are scored with ``bm25`` and then
  ordered in Python by ``(score, ts_ms, id)`` descending, so equal-scoring episodes (the common
  case: several screenshots of the same document) come back newest first, identically on every
  call, after a reopen, and whatever order they were ingested in (T-0009 A4).
- **The FTS row is not a copy that can go stale.** ``ingest``, ``set_caption`` and
  ``forget_since`` write both tables in the same transaction; a caption that is in ``episodes``
  but not in ``episodes_fts`` is invisible to search.
- **A term nearly every episode carries cannot justify a hit** (T-0053, :data:`MAX_TERM_DOC_FRACTION`).
  On 2026-09-16 ``"Make a flat looking UI."`` matched three calculus captions on the word ``"a"``
  and the prompt presented them as the current screen. The rule is document frequency, not a score
  floor and not a stop list: bm25 clamps the IDF of a universal term to a hair above zero rather
  than to zero (~1.6e-06 measured, not the 0.0 first reported), so the *same* three rows score
  1.1e-06 in a 3-episode store and 4.49 in a 400-episode one, and a score floor would mean a
  different thing at every corpus size.

The schema is versioned with ``PRAGMA user_version``: :data:`_SCHEMA` is the original v0 shape and
every later column arrives through :data:`_MIGRATIONS`, applied in one transaction in ``__init__``
before the first query, so the store on the user's disk is altered in place and never rebuilt.

The connection is used synchronously inside ``async`` methods, which is only defensible because
the write path is fast, and it is only fast with the PRAGMAs :class:`SqliteMemory` sets in
``__init__`` (T-0009 A5). Measured 2026-09-13 on this box — Linux 7.0.0-31-generic, ext4 on LVM,
python 3.11.16, sqlite 3.53.1 — as n=100 ``ingest`` calls into a fresh database, one episode each
(window title, app, thumbnail *path*; the store never reads the image), no second writer:

- sqlite's defaults (``journal_mode=delete``, ``synchronous=FULL``): median 15.45 ms, p95 20.08 ms
- as configured here (``journal_mode=wal``, ``synchronous=NORMAL``): median 0.015 ms, p95 0.039 ms

``ingest`` fires on ``scene.changed`` *inside a live turn*, so the default two-fsyncs-per-commit
regime would blow the 20 ms reflex-lane cap (``.claude/rules/conductor.md``) on its own. Under
WAL it does not, and a thread hop buys nothing. ``busy_timeout`` is shortened from sqlite's 5 s
default to 250 ms for the same reason: if a second writer ever holds the database, failing with
``OperationalError`` after 250 ms is right and blocking the event loop for five seconds is not.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery
from local_astra.contracts.perception import Source

DEFAULT_DIR = Path.home() / ".local" / "share" / "local_astra"

#: FTS5 tokenizer. ``porter`` stems (``fraction`` finds ``fractions``), ``unicode61`` folds case
#: and diacritics and splits on punctuation — captions and window titles are prose, not code.
TOKENIZER = "porter unicode61"

#: Document-frequency ceiling (T-0053 criteria 12-14). A query term carried by this fraction of
#: the episodes or more distinguishes nothing in *this* store, whatever the language, and is
#: dropped from the query before it runs — so a question whose only match is such a term recalls
#: nothing at all rather than the k newest episodes. Deliberately a fraction of the corpus and not
#: a hardcoded English stop list: the tokenizer is ``porter unicode61`` and the design is not
#: English-only. Above one half by ruling A8 (at one half it would reject a term two episodes in
#: four carry, which bm25 already discounts and which recall needs). The cost, accepted and
#: recorded in docs/tasks/T-0053.md ruling A5: in a store of a handful of episodes all about one
#: subject, a question about that subject recalls nothing.
MAX_TERM_DOC_FRACTION = 0.9

#: …with one floor: a term carried by a single episode is the most selective a term can be — it
#: names that episode and excludes every other — so it is informative at any corpus size, and a
#: store holding one episode still answers. This is a statement about the *term*, not an
#: exemption for small stores: at n=3 and at n=400 alike, a term in one episode survives and a
#: term in every episode does not (T-0053 criteria 13 and 14).
_MIN_UNINFORMATIVE_DOCS = 2

#: ``PRAGMA user_version`` of the schema this module writes.
SCHEMA_VERSION = 1

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS episodes (
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
    "CREATE INDEX IF NOT EXISTS episodes_ts_ms ON episodes(ts_ms)",
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
        USING fts5(caption, window_title, tokenize='{TOKENIZER}')
    """,
)

#: Applied in order to a database at a lower ``user_version``. The statements are ``ALTER``s, never
#: a rebuild: the user's store holds rows whose ids the rest of the system (and their thumbnails)
#: refer to, and a copy-into-a-new-table migration renumbers them and restarts the AUTOINCREMENT
#: sequence. ``_SCHEMA`` above stays frozen at the v0 shape so that a fresh database and a
#: migrated one travel the same path and are the same file afterwards.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    # v1 (T-0053): which conversation saw the episode. Nullable, unindexed, never read by search.
    ("ALTER TABLE episodes ADD COLUMN session_id TEXT",),
)

_ROW = (
    "e.id, e.ts_ms, e.source, e.thumbnail_path, e.phash, e.window_title, e.app, e.caption,"
    " e.session_id"
)

#: Markers wrapped around matched terms by ``highlight()``. Control characters, because they are
#: vanishingly unlikely in model prose or a compositor's window title — but nothing stops a
#: caption from containing them, and one that does contributes a spurious term to ``matched``.
#: That is cosmetic (the HUD's "why this memory" tooltip); ``score``, ranking and the rows
#: returned are untouched, so captions are stored verbatim rather than sanitised (T-0009).
_MARK_OPEN, _MARK_CLOSE = "\x02", "\x03"
_MARKED = re.compile(f"{_MARK_OPEN}(.*?){_MARK_CLOSE}", re.DOTALL)

_WORD = re.compile(r"\w+", re.UNICODE)


def _match_expression(terms: Sequence[str]) -> str | None:
    """Turn the words of a spoken transcript into an FTS5 ``MATCH`` expression, or ``None``.

    Every word becomes a quoted term and the terms are OR-ed: recall matters more than precision
    for a memory whose top-3 thumbnails are re-read by the model anyway, and a question such as
    "where did I leave my glasses?" shares only its content words with the caption. Quoting is
    what makes ``AND``, ``NOT``, ``*``, ``^`` and ``-`` inert; the word split (:data:`_WORD`, done
    by the caller) is what keeps a stray quote or apostrophe out of the expression in the first
    place. ``None`` means there is nothing left to ask for — the transcript had no words, or none
    of them distinguished one episode from another (:data:`MAX_TERM_DOC_FRACTION`).
    """
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def _matched_terms(*highlighted: str) -> tuple[str, ...]:
    """The document words FTS5 actually matched (post-stemming), for the HUD tooltip."""
    found = {m.lower() for text in highlighted for m in _MARKED.findall(text)}
    return tuple(sorted(found))


class SqliteMemory:
    """``MemoryClient`` backed by one SQLite file. See the module docstring."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_DIR / "memory.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        # T-0009 A5, see the module docstring. WAL turns the two fsyncs per commit into one
        # append and NORMAL drops that one for everything but a checkpoint, which is what takes
        # `ingest` from 15 ms to microseconds; the price is that a caption or two can be lost to
        # a power cut, which is the right trade for a screenshot index. WAL keeps its `-wal` and
        # `-shm` files beside the database, i.e. still under `DEFAULT_DIR` (criterion 4).
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "busy_timeout=250"):
            self._conn.execute(f"PRAGMA {pragma}")
        self._migrate()

    def _migrate(self) -> None:
        """Bring the file up to :data:`SCHEMA_VERSION`, in one transaction, before any query.

        The database this runs against is the user's own: it has real rows in it, referred to by
        id and by thumbnail path, so every version step is an ``ALTER`` and nothing is copied or
        recreated. ``BEGIN IMMEDIATE`` takes the write lock before ``user_version`` is read, so
        two processes opening the same store at once cannot both decide to migrate it; a version
        this code does not know about (a file written by a newer build) is left alone rather than
        downgraded. The transaction is not wrapped in ``try``: a migration that raises leaves the
        file untouched — sqlite rolls the transaction back when the connection goes — and the
        constructor fails loudly instead of handing back a half-migrated store.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA:
            self._conn.execute(statement)
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version < SCHEMA_VERSION:
            for step in _MIGRATIONS[version:SCHEMA_VERSION]:
                for statement in step:
                    self._conn.execute(statement)
            # Not a bound parameter: PRAGMA does not take one, and the value is an int literal.
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    async def ingest(self, episode: Episode) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes"
            " (ts_ms, source, thumbnail_path, phash, window_title, app, caption, session_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.ts_ms,
                str(episode.source),
                episode.thumbnail_path,
                episode.phash,
                episode.window_title,
                episode.app,
                episode.caption,
                episode.session_id,
            ),
        )
        episode_id = int(cur.lastrowid or 0)
        self._conn.execute(
            "INSERT INTO episodes_fts (rowid, caption, window_title) VALUES (?, ?, ?)",
            (episode_id, episode.caption or "", episode.window_title or ""),
        )
        self._conn.commit()
        return episode_id

    async def set_caption(self, episode_id: int, caption: str) -> None:
        """Replace the caption *and* the indexed text; a second FTS row would make both findable."""
        self._conn.execute("UPDATE episodes SET caption = ? WHERE id = ?", (caption, episode_id))
        self._conn.execute(
            "UPDATE episodes_fts SET caption = ? WHERE rowid = ?", (caption, episode_id)
        )
        self._conn.commit()

    def _informative(self, words: Sequence[str]) -> set[str]:
        """The lowercased query words that still distinguish one episode from another (T-0053).

        One statement for the whole query, whatever its length: a probe per term would make an
        eight-word question eight extra round trips and turn B10's "≤ 50 ms at 100 k episodes"
        into a promise made per word. Each term is counted *post-stemming*, by asking the index
        the same question the search itself asks — ``fts5vocab`` would be one lookup cheaper but
        stores stems only, so ``glasses`` (stem ``glass``), ``looking`` (``look``) and ``was``
        (``wa``) are absent from it and would every one of them be scored as a term no episode
        carries. The term reaches SQL as a bound parameter and is quoted there exactly as
        :func:`_match_expression` quotes it, so a transcript is still never FTS5 syntax.

        Document frequency is counted over the **whole** store — the time filter selects rows, it
        does not change what a word is worth — but only over the episodes the numerator can be
        drawn from: the ones that carry text. ``commit_history`` ingests an episode per
        turn-with-a-frame with no caption and the idle captioner fills it in later, so a backlog
        of caption-less rows is the store's *normal* state (that is what ``next_uncaptioned``
        exists for) and each of those rows carries an empty FTS document that can never match any
        term. Counting them in the total only inflates it: at more than one uncaptioned row per
        nine captioned, nothing could ever reach the ceiling and this function would be a no-op.
        Measured on a copy of the user's own store (15 rows, 11 captioned) it *was* one — the
        origin query returned the same three captions on the same ``"a"`` (T-0053 ruling A30).

        A store with nothing captioned in it therefore leaves every term informative, and the
        search that follows finds nothing because there is no text to find. The population is
        "carries text", not "carries a *token*": a caption of punctuation indexes nothing and is
        still counted, which errs towards keeping a term — the direction that recalls too much is
        the one this rule is for, so if that ever shows up it is a defect, not a margin.
        """
        terms = sorted({word.lower() for word in words})
        rows = self._conn.execute(
            "WITH q(term) AS (VALUES " + ",".join(["(?)"] * len(terms)) + ")"
            " SELECT q.term,"
            " (SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH '\"' || q.term || '\"'),"
            " (SELECT count(*) FROM episodes"
            "  WHERE coalesce(caption, '') <> '' OR coalesce(window_title, '') <> '')"
            " FROM q",
            terms,
        ).fetchall()
        return {
            str(term)
            for term, documents, total in rows
            if documents < max(MAX_TERM_DOC_FRACTION * total, _MIN_UNINFORMATIVE_DOCS)
        }

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]:
        words = _WORD.findall(query.text)
        if not words:
            return []
        informative = self._informative(words)
        expression = _match_expression([w for w in words if w.lower() in informative])
        if expression is None:
            return []
        sql = [
            f"SELECT {_ROW}, bm25(episodes_fts),"
            " highlight(episodes_fts, 0, ?, ?), highlight(episodes_fts, 1, ?, ?)"
            " FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.rowid"
            " WHERE episodes_fts MATCH ?"
        ]
        params: list[object] = [_MARK_OPEN, _MARK_CLOSE, _MARK_OPEN, _MARK_CLOSE, expression]
        if query.t0_ms is not None:
            sql.append(" AND e.ts_ms >= ?")
            params.append(query.t0_ms)
        if query.t1_ms is not None:
            sql.append(" AND e.ts_ms <= ?")
            params.append(query.t1_ms)
        rows = self._conn.execute("".join(sql), params).fetchall()

        hits = [
            MemoryHit(
                episode=_episode(row),
                # bm25() is "smaller is better"; a hit's score reads better the other way up.
                # Rounded so that floating-point noise cannot pre-empt the recency tie-break.
                score=round(-float(row[9]), 9),
                matched=_matched_terms(row[10], row[11]),
            )
            for row in rows
        ]
        hits.sort(key=lambda h: (h.score, h.episode.ts_ms, h.episode.id or 0), reverse=True)
        return hits[: query.k]

    async def forget_since(self, t0_ms: int) -> int:
        """Delete every episode at or after ``t0_ms`` (inclusive, T-0009 A2) and its thumbnail."""
        doomed = self._conn.execute(
            "SELECT id, thumbnail_path FROM episodes WHERE ts_ms >= ?", (t0_ms,)
        ).fetchall()
        for _, thumbnail_path in doomed:
            # The thumbnail goes first: if unlinking fails the row survives and `astra forget`
            # can be run again, rather than leaving an orphan JPEG nothing points at.
            Path(thumbnail_path).unlink(missing_ok=True)
        for episode_id, _ in doomed:
            self._conn.execute("DELETE FROM episodes_fts WHERE rowid = ?", (episode_id,))
            self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._conn.commit()
        return len(doomed)

    async def next_uncaptioned(self) -> Episode | None:
        """The newest episode with no caption — what the idle captioner works on next (A1).

        Newest first: the scene the user just looked at is the one they are about to ask about,
        and captioning walks backwards from there whenever the session stays idle.
        """
        row = self._conn.execute(
            f"SELECT {_ROW} FROM episodes e WHERE e.caption IS NULL"
            " ORDER BY e.ts_ms DESC, e.id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else _episode(row)


def _episode(row: tuple[Any, ...]) -> Episode:
    """Build an ``Episode`` from the ``_ROW`` columns (a hit carries the whole row back)."""
    return Episode(
        id=int(row[0]),
        ts_ms=int(row[1]),
        source=Source(row[2]),
        thumbnail_path=str(row[3]),
        phash=str(row[4]),
        window_title=None if row[5] is None else str(row[5]),
        app=None if row[6] is None else str(row[6]),
        caption=None if row[7] is None else str(row[7]),
        session_id=None if row[8] is None else str(row[8]),
    )
