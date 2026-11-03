"""Memory-lite: SQLite + FTS5 store and the idle captioner (docs/01 D8, docs/02 §2.6). Task T-0009.

Activated for ``/implement T-0009`` (the ``pending`` markers are gone; these must go green).

What the specs pin down, and where:

- docs/02 §2.6 — one SQLite file under ``~/.local/share/local_astra/``; ingest on scene change;
  ``search(text, time_range, k)`` is FTS5 over ``caption`` with a time filter and recency decay;
  the top-3 thumbnails are attached to the prompt.
- ``.claude/rules/perception-memory.md`` — nothing under ``memory/`` opens a non-loopback socket
  or writes outside ``~/.local/share/local_astra/``; captions are written by DiffusionGemma when
  idle > 10 s, the request is aborted on speech onset and never pre-empts a live turn.
- CLAUDE.md / ``.claude/rules/conductor.md`` — assistant *thinking* is never stored.

The captioner section below fixes an API the task file does not name (T-0009 criterion 5 states
behaviour only): ``IdleCaptioner(llm=…, memory=…)`` with ``run_once()`` and ``abort()``. The
behaviour asserted is from the docs; the spelling is this module's choice — see the test-author's
report. Everything else is asserted only where the docs are precise (time-filter *boundary*
inclusivity and ``MemoryHit.matched`` are deliberately not asserted: they are unspecified).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.llm import Commit, LlmOptions, PrefillHandle, Prompt
from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import Source
from local_astra.memory import captioner as captioner_mod
from local_astra.memory.store import SqliteMemory

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path):
    return SqliteMemory(tmp_path / "memory.sqlite")


def ep(tmp_path, ts: int, title: str | None = None) -> Episode:
    thumb = tmp_path / f"{ts}.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xd9")
    return Episode(
        ts_ms=ts,
        source=Source.SCREEN,
        thumbnail_path=str(thumb),
        phash="0" * 16,
        window_title=title,
    )


# --- criterion 1: ingest → search, FTS5 over caption and window_title, time filter, top-k -------


async def test_ingest_search_round_trip_over_caption_and_title(store, tmp_path):
    a = await store.ingest(ep(tmp_path, 1000, title="budget.xlsx — LibreOffice"))
    b = await store.ingest(ep(tmp_path, 2000))
    await store.set_caption(b, "chapter three on fractions, worksheet with four questions")
    hits = await store.search(MemoryQuery(text="fractions worksheet"))
    assert [h.episode.id for h in hits] == [b]
    hits = await store.search(MemoryQuery(text="budget"))
    assert [h.episode.id for h in hits] == [a]


async def test_time_filter_and_k(store, tmp_path):
    # The fifth episode is about something else on purpose: a term *every* episode carries
    # distinguishes no episode from another and the store drops it (T-0053 criteria 12-14), so a
    # store in which "fractions" is universal cannot answer "fractions" at all. It sits outside
    # the time window and matches neither query, so it changes no expectation below.
    await store.set_caption(await store.ingest(ep(tmp_path, 500)), "the kitchen window at dusk")
    ids = []
    for t in (1000, 2000, 3000, 4000):
        i = await store.ingest(ep(tmp_path, t))
        await store.set_caption(i, "fractions")
        ids.append(i)
    hits = await store.search(MemoryQuery(text="fractions", t0_ms=1500, t1_ms=3500, k=3))
    assert {h.episode.id for h in hits} == {ids[1], ids[2]}
    hits = await store.search(MemoryQuery(text="fractions", k=2))
    assert [h.episode.id for h in hits] == [ids[3], ids[2]], "ties broken by recency"


async def test_tokenizer_choice_is_explicit(store, tmp_path):
    """State what the FTS tokenizer does with inflection so prompts can rely on it."""
    i = await store.ingest(ep(tmp_path, 1000))
    await store.set_caption(i, "fractions")
    assert [h.episode.id for h in await store.search(MemoryQuery(text="fraction"))] == [i], (
        "porter stemming expected"
    )


async def test_search_returns_the_full_episode_row_it_ingested(store, tmp_path):
    """A hit carries the row back (the prompt builder needs ``thumbnail_path`` and ``ts_ms``)."""
    e = ep(tmp_path, 1000, title="budget.xlsx — LibreOffice")
    eid = await store.ingest(e)
    (hit,) = await store.search(MemoryQuery(text="budget"))
    assert hit.episode.id == eid
    assert hit.episode.ts_ms == 1000
    assert hit.episode.source is Source.SCREEN
    assert hit.episode.thumbnail_path == e.thumbnail_path
    assert hit.episode.phash == e.phash
    assert hit.episode.window_title == "budget.xlsx — LibreOffice"


async def test_episodes_survive_reopening_the_database_file(tmp_path):
    """One file under the directory, not process state (docs/02 §2.6)."""
    path = tmp_path / "memory.sqlite"
    first = SqliteMemory(path)
    eid = await first.ingest(ep(tmp_path, 1000))
    await first.set_caption(eid, "quarterly budget spreadsheet")
    hits = await SqliteMemory(path).search(MemoryQuery(text="spreadsheet"))
    assert [h.episode.id for h in hits] == [eid]


async def test_no_match_returns_no_hits(store, tmp_path):
    """Empty-result behaviour (docs/03 §3 "Memory ranking"): a value, not an exception."""
    eid = await store.ingest(ep(tmp_path, 1000))
    await store.set_caption(eid, "chapter three on fractions")
    assert list(await store.search(MemoryQuery(text="helicopter"))) == []


async def test_query_text_is_a_spoken_transcript_not_fts_syntax(store, tmp_path):
    """Adversarial: queries come from STT, so punctuation must not reach FTS5 as syntax.

    Naive interpolation into ``MATCH '<text>'`` raises ``sqlite3.OperationalError`` on the quote
    and on the bare ``-``/``*``; the store must still find the episode by its words.
    """
    eid = await store.ingest(ep(tmp_path, 1000))
    await store.set_caption(eid, "reading glasses on the kitchen table")
    for text in (
        'where did I leave my "glasses"?',
        "glasses - kitchen",
        "glasses AND (NOT table)",
        "glasses*^",
        "my glasses' case",
    ):
        hits = await store.search(MemoryQuery(text=text))
        assert [h.episode.id for h in hits] == [eid], text


# --- criterion 1 + "top-3 thumbnail attachment": selection is deterministic ---------------------


async def test_default_k_is_three_most_recent_matches(store, tmp_path):
    """``MemoryQuery.k`` defaults to 3 (contract) and the three are the newest of five equals."""
    ids = {}
    # …plus one episode about something else, so "fractions" is not carried by every episode in
    # the store and therefore not dropped as uninformative (T-0053 criteria 12-14). It never
    # matches the query, so the five equals below are still the only candidates.
    await store.set_caption(await store.ingest(ep(tmp_path, 500)), "the kitchen window at dusk")
    for t in (1000, 2000, 3000, 4000, 5000):
        i = await store.ingest(ep(tmp_path, t))
        await store.set_caption(i, "a worksheet about fractions")
        ids[t] = i
    hits = await store.search(MemoryQuery(text="fractions"))
    assert [h.episode.id for h in hits] == [ids[5000], ids[4000], ids[3000]]


async def test_top3_selection_is_deterministic_for_the_same_query(tmp_path):
    """Same episodes, same query ⇒ same three thumbnails in the same order.

    Three independent runs of the same inputs: repeated calls, a reopened file, and a database
    built by ingesting the episodes in the opposite order (which forbids ranking by rowid —
    equal-text captions tie on bm25, so the documented recency tie-break must decide).
    """
    stamps = [1000, 2000, 3000, 4000, 5000]
    caption = "a worksheet about fractions"
    # Both stores also hold one episode carrying neither query term, so neither "fractions" nor
    # "worksheet" is universal and both survive the document-frequency rule (T-0053 criteria
    # 12-14). It matches nothing, so it never enters the top three.
    other, other_caption = 6000, "the kitchen window at dusk"

    async def build(path: Path, order: list[int]) -> SqliteMemory:
        s = SqliteMemory(path)
        for t in order:
            await s.set_caption(await s.ingest(ep(tmp_path, t)), caption)
        await s.set_caption(await s.ingest(ep(tmp_path, other)), other_caption)
        return s

    forward = await build(tmp_path / "a.sqlite", stamps)
    backward = await build(tmp_path / "b.sqlite", list(reversed(stamps)))
    q = MemoryQuery(text="fractions worksheet", k=3)

    def shape(hits):
        return [(h.episode.ts_ms, h.episode.thumbnail_path) for h in hits]

    first = shape(await forward.search(q))
    assert [ts for ts, _ in first] == [5000, 4000, 3000]
    assert shape(await forward.search(q)) == first, "repeated query"
    assert shape(await SqliteMemory(tmp_path / "a.sqlite").search(q)) == first, "reopened file"
    assert shape(await backward.search(q)) == first, "insertion order must not decide the order"


async def test_set_caption_replaces_the_indexed_text(store, tmp_path):
    """Re-captioning must update the FTS row, not add a second one (stale-index defect)."""
    eid = await store.ingest(ep(tmp_path, 1000))
    await store.set_caption(eid, "chapter three on fractions")
    await store.set_caption(eid, "chapter four on decimals")
    assert [h.episode.id for h in await store.search(MemoryQuery(text="decimals"))] == [eid]
    assert list(await store.search(MemoryQuery(text="fractions"))) == []
    hits = await store.search(MemoryQuery(text="decimals"))
    assert hits[0].episode.caption == "chapter four on decimals"


# --- criterion 3: forget_since deletes rows and thumbnails and returns the count ----------------


async def test_forget_since_removes_rows_and_thumbnails(store, tmp_path):
    e1, e2 = ep(tmp_path, 1000), ep(tmp_path, 5000)
    await store.ingest(e1)
    await store.ingest(e2)
    assert await store.forget_since(3000) == 1
    assert (tmp_path / "1000.jpg").exists() and not (tmp_path / "5000.jpg").exists()


async def test_forget_since_cutoff_is_inclusive_and_spares_everything_earlier(store, tmp_path):
    """``astra forget --since T`` removes the episodes at or after T — and nothing before it."""
    ids = {}
    # Earlier than the cutoff and about something else, so it survives the forget *and* keeps
    # "fractions" out of every surviving caption — a term two of two survivors carried would be
    # dropped as uninformative and the search below would recall nothing (T-0053 criteria 12-14).
    await store.set_caption(await store.ingest(ep(tmp_path, 500)), "the kitchen window at dusk")
    for t in (1000, 2000, 3000, 4000):
        ids[t] = await store.ingest(ep(tmp_path, t))
        await store.set_caption(ids[t], "a worksheet about fractions")

    assert await store.forget_since(3000) == 2, "3000 and 4000 (the cutoff itself goes)"
    hits = await store.search(MemoryQuery(text="fractions", k=10))
    assert [h.episode.id for h in hits] == [ids[2000], ids[1000]]
    assert (tmp_path / "1000.jpg").exists() and (tmp_path / "2000.jpg").exists()
    assert not (tmp_path / "3000.jpg").exists() and not (tmp_path / "4000.jpg").exists()
    assert await store.forget_since(3000) == 0, "nothing left at or after the cutoff"


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    stamps=st.lists(st.integers(min_value=0, max_value=9), min_size=0, max_size=8, unique=True),
    cutoff=st.integers(min_value=0, max_value=10),
)
def test_forget_since_count_matches_a_brute_force_oracle(stamps: list[int], cutoff: int) -> None:
    """Oracle: the count is ``len([t for t in stamps if t >= cutoff])``, computed here by hand.

    Every drawn episode is ingested, deleted and recalled, so all three oracles below range over
    the whole of ``stamps``. What varies with the draw is *which* term each caption carries: the
    earliest episode is captioned "window", every later one "fractions", and the query asks for
    both. This is not decoration. A term carried by every episode in the store distinguishes no
    episode from another and is dropped before the search runs (T-0053 criteria 12-14), so a
    corpus of N identical captions recalls nothing at all and the ``left`` oracle would be
    measuring the rule rather than ``forget_since``. Making the odd episode the *earliest* drawn
    stamp is what keeps the oracle exact at every N: it is the last episode any cutoff deletes, so
    whenever k of the N survivors carry "fractions" the survivor carrying "window" is among them —
    document frequencies N-1 and 1 out of N, both under the ceiling for every N this strategy can
    reach (max 9) and both above the one-episode floor. Every survivor therefore matches the
    query, whatever the draw.
    """
    quiet = min(stamps, default=None)

    async def run() -> tuple[int, list[int]]:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            s = SqliteMemory(tmp / "memory.sqlite")
            for t in stamps:
                caption = "window" if t == quiet else "fractions"
                await s.set_caption(await s.ingest(ep(tmp, t)), caption)
            removed = await s.forget_since(cutoff)
            query = MemoryQuery(text="fractions window", k=10)
            left = [h.episode.ts_ms for h in await s.search(query)]
            survivors = sorted(str(p.name) for p in tmp.glob("*.jpg"))
            assert survivors == sorted(f"{t}.jpg" for t in stamps if t < cutoff)
            return removed, sorted(left)

    removed, left = asyncio.run(run())
    assert removed == len([t for t in stamps if t >= cutoff])
    assert left == sorted(t for t in stamps if t < cutoff)


# --- criterion 4: only its own directory, no sockets --------------------------------------------


async def test_store_writes_only_under_its_directory(store, tmp_path):
    await store.ingest(ep(tmp_path, 1000))
    created = {p.name for p in tmp_path.iterdir()}
    assert "memory.sqlite" in created
    assert all(n.startswith("memory.sqlite") or n.endswith(".jpg") for n in created), created


CHILD = r"""
import asyncio, json, os, socket, sys
from pathlib import Path

# Audit hook: anything that reaches the network. The asyncio event loop makes an AF_UNIX
# socketpair for its self-pipe, which is not a network call; an AF_INET socket, a connect(),
# a bind() or a DNS lookup is.
net = []
NETCALLS = ("socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname")

def audit(name, args):
    if name in NETCALLS:
        net.append(name)
    elif name == "socket.__new__" and args[1] != socket.AF_UNIX:
        net.append("socket.__new__ family=%r" % (args[1],))

sys.addaudithook(audit)

from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import Source
from local_astra.memory import store as store_mod

async def main():
    d = store_mod.DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    thumb = d / "shot.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xd9")
    s = store_mod.SqliteMemory()
    eid = await s.ingest(Episode(ts_ms=1000, source=Source.SCREEN, thumbnail_path=str(thumb),
                                 phash="0" * 16, window_title="budget.xlsx"))
    await s.set_caption(eid, "quarterly budget spreadsheet")
    await s.search(MemoryQuery(text="budget"))
    await s.forget_since(0)
    return str(s.path)

path = asyncio.run(main())
home = Path(os.environ["HOME"])
print(json.dumps({
    "path": path,
    "default_dir": str(store_mod.DEFAULT_DIR),
    "home_files": sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file()),
    "cwd_entries": sorted(p.name for p in Path.cwd().iterdir()),
    "net": sorted(set(net)),
}))
"""


def test_default_location_is_under_the_home_data_dir_and_nothing_else_is_touched(tmp_path):
    """Criterion 4 end to end, in a child process with ``HOME`` pointed at ``tmp_path``.

    Asserts the default path resolves to ``$HOME/.local/share/local_astra/`` *for the running
    user* (so the real home is never touched by the suite), that ingest/search/forget create no
    file outside that directory or in the working directory, and — via ``sys.addaudithook`` —
    that nothing touches the network (``.claude/rules/perception-memory.md``: nothing under
    ``memory/`` opens a non-loopback socket; the store is in-process SQLite, so there is no
    ``connect``, ``bind``, DNS lookup or AF_INET socket at all).
    """
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", CHILD], cwd=work, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout)

    astra = home / ".local" / "share" / "local_astra"
    assert out["default_dir"] == str(astra)
    assert out["path"] == str(astra / "memory.sqlite")
    assert out["net"] == [], out["net"]
    assert out["cwd_entries"] == [], "nothing may be written to the working directory"
    stray = [f for f in out["home_files"] if not f.startswith(".local/share/local_astra/")]
    assert stray == [], stray


# --- criterion 5: the idle captioner ------------------------------------------------------------


class _StallingLlm:
    """``LlmBackend`` stub that stalls mid-caption and keeps streaming *after* being aborted.

    Deliberately rogue: it records ``abort`` but does not stop. A captioner that leaves the
    cancellation to the backend, or that writes whatever the stream produced once it ends, is
    caught by ``test_abort_on_onset_drops_the_in_flight_caption``.
    """

    def __init__(self, commits: Sequence[Commit], *, stall_after: int = 1) -> None:
        self._commits = list(commits)
        self._stall_after = stall_after
        self.gate = asyncio.Event()
        self.stalled = asyncio.Event()
        self.finished = asyncio.Event()
        self.generates: list[str] = []
        self.aborted: list[str] = []

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        return PrefillHandle(request_id=request_id, prefix_tokens=0)

    async def generate(
        self, prompt: Prompt, request_id: str, options: LlmOptions | None = None
    ) -> AsyncIterator[Commit]:
        self.generates.append(request_id)
        for n, c in enumerate(self._commits):
            if n == self._stall_after:
                self.stalled.set()
                await self.gate.wait()
            yield c.model_copy(update={"request_id": request_id})
        self.finished.set()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _commit(idx: int, segment: str, text: str, *, thought_open: bool = False) -> Commit:
    return Commit(
        request_id="caption",
        canvas_idx=idx,
        segment=segment,
        text=text,
        thought_open=thought_open,
    )


def _captioner(llm: object, memory: SqliteMemory) -> object:
    cls = getattr(captioner_mod, "IdleCaptioner", None)
    if cls is None:  # pragma: no cover - the stub state
        raise NotImplementedError(
            "T-0009: local_astra.memory.captioner.IdleCaptioner(llm=…, memory=…) with "
            "await run_once() and await abort() (docs/02 §2.6, .claude/rules/conductor.md)"
        )
    return cls(llm=llm, memory=memory)


async def test_idle_captioner_writes_a_caption_that_makes_the_episode_findable(store, tmp_path):
    """Happy path: one pending episode, one request, the caption lands and FTS finds it."""
    llm = _StallingLlm([_commit(0, "answer", "LibreOffice Calc showing a quarterly budget")])
    llm.gate.set()
    eid = await store.ingest(ep(tmp_path, 1000))
    cap = _captioner(llm, store)

    await cap.run_once()

    assert len(llm.generates) == 1, llm.generates
    hits = await store.search(MemoryQuery(text="quarterly budget"))
    assert [h.episode.id for h in hits] == [eid]
    assert "budget" in (hits[0].episode.caption or "")


async def test_idle_captioner_never_stores_thinking(store, tmp_path):
    """CLAUDE.md is absolute: assistant thinking is never stored. Only the answer is a caption."""
    llm = _StallingLlm(
        [
            _commit(0, "thought", "SECRETTHOUGHT the user seems to be avoiding their taxes"),
            _commit(1, "answer", "LibreOffice Calc showing a quarterly budget"),
        ],
        stall_after=99,
    )
    eid = await store.ingest(ep(tmp_path, 1000))
    cap = _captioner(llm, store)

    await cap.run_once()

    (hit,) = await store.search(MemoryQuery(text="budget"))
    caption = hit.episode.caption or ""
    assert "SECRETTHOUGHT" not in caption and "taxes" not in caption, caption
    assert "budget" in caption
    assert list(await store.search(MemoryQuery(text="SECRETTHOUGHT"))) == []
    assert eid == hit.episode.id


async def test_abort_on_onset_drops_the_in_flight_caption(store, tmp_path):
    """Adversarial: speech onset arrives mid-caption; the result must never land in the store.

    ``.claude/rules/conductor.md``: the captioner is aborted on speech onset (the turn machine
    emits ``CaptionerAbort``) and never pre-empts a live turn. Aborting means the request is
    abandoned — a captioner that only sets a flag and writes when the stream finishes leaves a
    caption behind from a turn the user interrupted, and keeps the third ``max-num-seqs`` slot.
    """
    llm = _StallingLlm(
        [
            _commit(0, "answer", "LibreOffice Calc"),
            _commit(1, "answer", " showing a quarterly budget"),
        ]
    )
    await store.ingest(ep(tmp_path, 1000))
    cap = _captioner(llm, store)

    task = asyncio.create_task(cap.run_once())
    await asyncio.wait_for(llm.stalled.wait(), timeout=2)

    await cap.abort()  # ← UserSpeechOnset: the conductor's CaptionerAbort command

    assert llm.aborted == llm.generates, "abort must reach the backend to free the sequence"

    llm.gate.set()  # the rogue backend delivers the rest of the canvas anyway
    done, _ = await asyncio.wait({task}, timeout=2)
    assert done, "run_once must return (or be cancelled) after abort"
    with contextlib.suppress(asyncio.CancelledError):
        task.result()
    await asyncio.sleep(0.05)  # …and any late writer gets its chance to misbehave

    for term in ("LibreOffice", "Calc", "quarterly", "budget"):
        hits = await store.search(MemoryQuery(text=term, k=10))
        assert list(hits) == [], (
            f"an aborted caption landed anyway ({term}): {[h.episode.caption for h in hits]}"
        )


async def test_abort_without_an_in_flight_caption_is_idempotent(store, tmp_path):
    """``CaptionerAbort`` is emitted on every onset, live captioner or not (T-0003 criterion 1)."""
    llm = _StallingLlm([_commit(0, "answer", "LibreOffice Calc showing a quarterly budget")])
    llm.gate.set()
    await store.ingest(ep(tmp_path, 1000))
    cap = _captioner(llm, store)

    await cap.abort()
    await cap.abort()
    assert llm.generates == []

    await cap.run_once()  # a later idle period still works
    assert len(llm.generates) == 1
    assert [h.episode.ts_ms for h in await store.search(MemoryQuery(text="budget"))] == [1000]


async def test_only_one_caption_request_at_a_time(store, tmp_path):
    """``--max-num-seqs 3``: the captioner owns one slot, so it never overlaps itself."""
    llm = _StallingLlm(
        [_commit(0, "answer", "LibreOffice Calc showing a quarterly budget")], stall_after=0
    )
    await store.ingest(ep(tmp_path, 1000))
    await store.ingest(ep(tmp_path, 2000))
    cap = _captioner(llm, store)

    first = asyncio.create_task(cap.run_once())
    await asyncio.wait_for(llm.stalled.wait(), timeout=2)
    second = asyncio.create_task(cap.run_once())
    await asyncio.sleep(0.05)

    assert len(llm.generates) == 1, f"second request while one is in flight: {llm.generates}"

    llm.gate.set()
    await asyncio.wait({first, second}, timeout=2)
    for t in (first, second):
        with contextlib.suppress(asyncio.CancelledError):
            t.result()


# --- gap T1 (review): relevance decides the order, recency only breaks ties ---------------------


#: Okapi BM25 as FTS5 implements it (``fts5_aux.c``): ``k1 = 1.2``, ``b = 0.75``, and the
#: Robertson/Sparck-Jones idf ``log((N - n + 0.5) / (n + 0.5))``, clamped to ``1e-6`` when a term
#: is in more than half the corpus. Written from the published formula, not from the store —
#: the store does not compute bm25, it asks SQLite for it and decides the sign and the sort key.
_BM25_K1, _BM25_B = 1.2, 0.75


def _bm25_oracle(lengths: list[int], postings: list[list[int]], doc: int) -> float:
    """Score of ``doc`` for a query whose terms have the given ``postings`` (lists of doc ids).

    ``lengths`` are the token counts of every document in the corpus; every term occurs at most
    once per document in the fixture below, so ``tf`` is 1 whenever ``doc`` is in the postings.
    Returns the "bigger is better" score, i.e. ``-bm25()``.
    """
    n_docs = len(lengths)
    avgdl = sum(lengths) / n_docs
    total = 0.0
    for docs_with_term in postings:
        if doc not in docs_with_term:
            continue
        n = len(docs_with_term)
        idf = math.log((n_docs - n + 0.5) / (n + 0.5))
        if idf <= 0.0:
            idf = 1e-6
        norm = 1.0 - _BM25_B + _BM25_B * lengths[doc] / avgdl
        total += idf * (1.0 * (_BM25_K1 + 1.0)) / (1.0 + _BM25_K1 * norm)
    return total


#: Six episodes that share no word with the query. They exist so that "fractions" and "decimals"
#: are *rare* (2 of 9 documents): with only the three probe episodes in the corpus every term is
#: in the majority, FTS5 clamps the idf to 1e-6, and length normalisation alone would rank the
#: long, doubly-relevant caption last. Real sessions have hundreds of unrelated episodes.
_FILLER = (
    (10_000, "a kitchen table with reading glasses"),
    (11_000, "a terminal running a long build"),
    (12_000, "an email client with an unread inbox"),
    (13_000, "a browser tab showing a train timetable"),
    (14_000, "a photo of a bicycle in the hallway"),
    (15_000, "a calendar with three meetings"),
)


async def test_relevance_outranks_recency_when_the_bm25_scores_differ(store, tmp_path):
    """T-0009 A4: recency is only the *tie*-break — a better bm25 score wins outright.

    Query "fractions decimals" against three captions with genuinely different scores: the
    oldest episode matches both terms, the two newer ones match one each. Hand oracle
    (``_bm25_oracle``, tokens counted by hand below):

    ======  =====================================  ======  =======  ===============
    ts_ms   caption                                tokens  terms    expected score
    ======  =====================================  ======  =======  ===============
     1000   a worksheet about fractions and dec…        6  both     2 × 1.03552 = 2.07104
     3000   fractions                                   1  one      1.64157
     4000   decimals                                    1  one      1.64157
    ======  =====================================  ======  =======  ===============

    with ``N = 9``, ``avgdl = 47/9``, ``idf = ln 3`` for both terms. So the order is
    ``[1000, 4000, 3000]``: relevance first, then recency among the two equal scores. The
    store's ``score`` is ``-bm25()`` ("bigger is better"); flipping that sign inverts this list,
    which is the mutation this test exists to catch.
    """
    ids = {}
    for ts, caption in (
        (1000, "a worksheet about fractions and decimals"),
        (3000, "fractions"),
        (4000, "decimals"),
        *_FILLER,
    ):
        ids[ts] = await store.ingest(ep(tmp_path, ts))
        await store.set_caption(ids[ts], caption)

    hits = await store.search(MemoryQuery(text="fractions decimals", k=10))

    assert [h.episode.ts_ms for h in hits] == [1000, 4000, 3000], (
        "the oldest episode matches both query terms and must rank first"
    )
    assert hits[0].score > hits[1].score, "this ordering must come from bm25, not from a tie-break"
    assert hits[1].score == hits[2].score, "…and 3000/4000 must be the tie the recency rule breaks"

    # The scores themselves, against the hand oracle. Document order is the ingest order above:
    # 0 = ts 1000, 1 = ts 3000, 2 = ts 4000, 3.. = the filler captions.
    lengths = [6, 1, 1] + [len(caption.split()) for _, caption in _FILLER]
    postings = [[0, 1], [0, 2]]  # "fractions" in docs 0 and 1; "decimals" in docs 0 and 2
    expected = {1000: _bm25_oracle(lengths, postings, 0), 3000: _bm25_oracle(lengths, postings, 1)}
    assert expected[1000] == pytest.approx(2.07104, rel=1e-4), "oracle sanity: hand value above"
    assert expected[3000] == pytest.approx(1.64157, rel=1e-4)
    assert hits[0].score == pytest.approx(expected[1000], rel=1e-6)
    assert hits[1].score == pytest.approx(expected[3000], rel=1e-6)


# --- gap T2 (review): an unclosed thought never reaches the store -------------------------------


async def test_a_commit_that_ends_inside_an_unclosed_thought_is_never_captioned(store, tmp_path):
    """CLAUDE.md is absolute: never store a canvas that ends inside an unclosed thought.

    ``segment="answer"`` is not enough — the reasoning parser labels the whole canvas ``answer``
    while ``thought_open`` says the ``<|think|>`` block never closed, so the "answer" text is
    still reasoning that the next step may rewrite. Dropping the ``thought_open`` guard makes
    this text both part of the caption and findable by search.
    """
    llm = _StallingLlm(
        [
            _commit(
                0, "answer", "UNCLOSEDTHOUGHT they are avoiding their taxes", thought_open=True
            ),
            _commit(1, "answer", "LibreOffice Calc showing a quarterly budget"),
        ],
        stall_after=99,
    )
    eid = await store.ingest(ep(tmp_path, 1000))

    await _captioner(llm, store).run_once()

    (hit,) = await store.search(MemoryQuery(text="budget"))
    assert hit.episode.id == eid
    assert hit.episode.caption == "LibreOffice Calc showing a quarterly budget", (
        "only the closed-thought answer is a caption"
    )
    for term in ("UNCLOSEDTHOUGHT", "avoiding", "taxes"):
        assert list(await store.search(MemoryQuery(text=term, k=10))) == [], term


# --- gap T3 (review): abort before the request starts -------------------------------------------


class _SlowLookupStore:
    """Wraps the real store and holds ``next_uncaptioned()`` open until the test lets it finish.

    The race it stages is the ordinary one: ``run_once()`` is scheduled by the idle timer, the
    user starts talking a millisecond later, and the abort lands while the captioner is still
    deciding *which* episode to caption.
    """

    def __init__(self, inner: SqliteMemory) -> None:
        self._inner = inner
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def next_uncaptioned(self) -> Episode | None:
        self.entered.set()
        await self.gate.wait()
        return await self._inner.next_uncaptioned()

    async def set_caption(self, episode_id: int, caption: str) -> None:
        await self._inner.set_caption(episode_id, caption)


async def test_abort_during_the_lookup_never_starts_the_request(store, tmp_path):
    """Adversarial: onset lands between ``run_once()`` and the first token.

    The captioner is the third of ``--max-num-seqs 3``; a request submitted *after* the user has
    started talking takes a slot away from the turn that is starting and has to be aborted again
    a moment later. Nothing may reach the backend, and the episode must stay uncaptioned so a
    later idle period still gets to caption it.
    """
    llm = _StallingLlm([_commit(0, "answer", "LibreOffice Calc showing a quarterly budget")])
    llm.gate.set()  # nothing here stalls: only the *lookup* is slow
    eid = await store.ingest(ep(tmp_path, 1000))
    slow = _SlowLookupStore(store)
    cap = _captioner(llm, slow)

    task = asyncio.create_task(cap.run_once())
    await asyncio.wait_for(slow.entered.wait(), timeout=2)

    await cap.abort()  # ← UserSpeechOnset, before a request_id exists

    slow.gate.set()
    done, _ = await asyncio.wait({task}, timeout=2)
    assert done, "run_once must finish after an abort during the lookup"
    with contextlib.suppress(asyncio.CancelledError):
        task.result()

    assert llm.generates == [], "a pre-empted run must not submit a request"
    assert llm.aborted == [], "…and there is no request id to abort"
    pending = await store.next_uncaptioned()
    assert pending is not None and pending.id == eid, "the episode is still waiting for a caption"
    assert list(await store.search(MemoryQuery(text="budget", k=10))) == []


# --- gap T4a (review, ruling A5): the durability PRAGMAs are policy, so pin them -----------------


def _connection(store: SqliteMemory) -> sqlite3.Connection:
    """The store's own connection. ``synchronous`` and ``busy_timeout`` are per-connection
    settings, so they can only be read back from the connection the store actually writes on."""
    conns = [v for v in vars(store).values() if isinstance(v, sqlite3.Connection)]
    assert len(conns) == 1, f"expected one sqlite3.Connection on {type(store).__name__}: {conns}"
    return conns[0]


async def test_store_opens_in_wal_with_synchronous_normal_and_a_short_busy_timeout(tmp_path):
    """T-0009 A5: ``ingest`` fires on ``scene.changed`` inside a live turn, under the 20 ms
    reflex cap, so the default ``journal_mode=delete`` + ``synchronous=FULL`` (two fsyncs per
    commit, measured at 15.1 ms median on this box at ef64a43) is not an option. ``busy_timeout``
    is shortened from sqlite's 5 s default because a second writer must fail fast rather than
    hold the event loop.

    The values *are* the policy, so they are pinned: ``wal``, ``synchronous == 1`` (NORMAL) and
    ``busy_timeout == 250`` ms, on a fresh store before anything is written.
    """
    store = SqliteMemory(tmp_path / "memory.sqlite")
    conn = _connection(store)

    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1, "NORMAL, not FULL(2)/OFF(0)"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 250

    # Independent of the store's connection: WAL is recorded in the database header, so a
    # connection opened by this test — which sets none of these pragmas — must also see it.
    assert (
        sqlite3.connect(tmp_path / "memory.sqlite").execute("PRAGMA journal_mode").fetchone()[0]
        == "wal"
    )

    # Criterion 4 still holds: the WAL sidecars live beside the database, inside the directory.
    await store.ingest(ep(tmp_path, 1000))
    created = {p.name for p in tmp_path.iterdir()}
    assert {"memory.sqlite-wal", "memory.sqlite-shm"} <= created, created
    assert created <= {"memory.sqlite", "memory.sqlite-wal", "memory.sqlite-shm", "1000.jpg"}


# --- gap T4b (review, ruling A6): abort() is exception-safe -------------------------------------


class _RaisingAbortLlm:
    """A backend whose ``abort`` fails (the vLLM engine socket is gone) and which keeps streaming.

    Scripted: the first ``generate`` stalls mid-canvas, later ones run to completion, and every
    generation has its own commits so a late write from the aborted stream is distinguishable
    from the caption of the retry.
    """

    def __init__(self, scripts: Sequence[Sequence[Commit]], *, stall_after: int = 1) -> None:
        self._scripts = [list(s) for s in scripts]
        self._stall_after = stall_after
        self.gate = asyncio.Event()
        self.stalled = asyncio.Event()
        self.generates: list[str] = []
        self.aborted: list[str] = []

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        return PrefillHandle(request_id=request_id, prefix_tokens=0)

    async def generate(
        self, prompt: Prompt, request_id: str, options: LlmOptions | None = None
    ) -> AsyncIterator[Commit]:
        nth = len(self.generates)
        self.generates.append(request_id)
        for n, c in enumerate(self._scripts[min(nth, len(self._scripts) - 1)]):
            if nth == 0 and n == self._stall_after:
                self.stalled.set()
                await self.gate.wait()
            yield c.model_copy(update={"request_id": request_id})

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        raise RuntimeError("vLLM abort failed: engine socket gone")


async def test_a_raising_llm_abort_still_cancels_the_consumer_and_leaves_the_captioner_usable(
    store, tmp_path
):
    """T-0009 A6: the exception propagates, but only *after* the local state is unwound.

    If ``llm.abort`` is awaited before the consumer is cancelled with no ``finally``, the raise
    skips the cancellation: the consumer stays parked on the backend's stream, ``run_once`` never
    returns, and the captioner is wedged single-flight for the rest of the session — every later
    idle period silently captions nothing. The conductor is allowed to see the error (the engine
    really did fail), but not to lose the captioner because of it.
    """
    llm = _RaisingAbortLlm(
        [
            [
                _commit(0, "answer", "LibreOffice Calc"),
                _commit(1, "answer", " showing an INTERRUPTEDBUDGET"),
            ],
            [_commit(0, "answer", "a quarterly budget in Calc")],
        ]
    )
    eid = await store.ingest(ep(tmp_path, 1000))
    cap = _captioner(llm, store)

    task = asyncio.create_task(cap.run_once())
    await asyncio.wait_for(llm.stalled.wait(), timeout=2)

    with pytest.raises(RuntimeError, match="engine socket gone"):
        await cap.abort()
    assert llm.aborted == llm.generates, "the backend abort must still have been attempted"

    done, _ = await asyncio.wait({task}, timeout=2)
    assert done, "a raising llm.abort must not leave run_once awaiting an uncancelled consumer"
    with contextlib.suppress(asyncio.CancelledError):
        task.result()

    llm.gate.set()  # the rogue backend finishes the aborted canvas anyway
    await asyncio.sleep(0.05)
    assert list(await store.search(MemoryQuery(text="INTERRUPTEDBUDGET", k=10))) == [], (
        "the late caption from the aborted stream must be discarded"
    )

    await cap.run_once()  # the next idle period

    assert len(llm.generates) == 2, f"the captioner is wedged: {llm.generates}"
    (hit,) = await store.search(MemoryQuery(text="quarterly budget", k=10))
    assert hit.episode.id == eid
    assert hit.episode.caption == "a quarterly budget in Calc"
