"""T-0142 group A (A1, A2 on synthetic logs): the pure degeneracy detector and its replay script (docs/tasks/T-0142.md).

**The interfaces these rows pin** (A1's, plus the readings listed in the test-author's report):

- ``local_astra.conductor.degeneracy``: ``degenerate(text: str) -> Trip | None``; ``Trip`` a frozen dataclass with
  ``unit: str``, ``repeats: int``, ``kind: Literal["ngram", "char_run"]``; ``MAX_UNIT_TOKENS = 3``,
  ``MIN_REPEATS = 8`` (H-T142-2), ``MIN_CHAR_RUN = 64``. Tokens are ``re.findall(r"\\w+|[^\\w\\s]+", text.casefold())``.
- **R-1 (``Trip.unit``):** the rows never pin the unit's spelling, only that its tokens (the same regex over
  ``unit.casefold()``) are the repeated unit, and that ``repeats`` is the back-to-back count of that unit (at least
  ``MIN_REPEATS``; exactly the count where the input has one run). For ``char_run`` the unit is the one character.
- **R-2 (replay script):** ``uv run python scripts/replay_degeneracy.py <log-dir>... [--expect N]``. A log dir is the
  A/B layout: ``trials.jsonl`` (``session_log`` per trial) and ``sessions/<file>.jsonl``; requests are the
  ``llm.request`` records, their thought text the ``llm.commit`` records with that ``attempt_id`` and
  ``segment == "thought"``, in log order. One stdout line per request with at least one thought commit, holding its
  ``attempt_id`` as a field (separated by whitespace or ``=``) and exactly one of the words ``tripped``/``clean``; a ``tripped`` line also holds ``commit=<k>``, the
  1-based ordinal of the thought commit at which ``degenerate`` first trips on the concatenation so far (the
  planner's "thought commit 2-4 of 3-19"). Exit 0 iff no request trips and ``--expect`` is absent, or ``--expect N``
  equals the number tripped; otherwise non-zero. Read-only.

The adversarial rows: ``test_clean_rows_never_trip`` kills ``MIN_REPEATS = 3`` with hand rows of 3 to 7 repeats (the
logged clean requests never repeat a unit more than twice, so the fixture alone cannot kill it, see the report);
``test_straddles_canvases`` kills a detector fed one commit at a time; ``test_boundaries`` kills ``MIN_REPEATS`` off
by one and ``lower()`` in place of ``casefold()``. The Hypothesis rows check the detector against a brute-force
oracle on tiny token sequences, and that a trip survives appended text.

Activated at write time (no ``pending`` marker): every row fails at BASE e1ae76e on a named ``AssertionError`` (the
module or the script does not exist yet), never on an import error.

Regime: CPU, pure; fixture ``tests/fixtures/t0142/thoughts.jsonl`` (T-0134 at e59aa69, T-0132 at 2ccff41; see its
``build_thoughts.py``); n = 1, deterministic.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.conductor.degeneracy"
MODULE_PATH = REPO / "local_astra" / "conductor" / "degeneracy.py"
SCRIPT = REPO / "scripts" / "replay_degeneracy.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0142" / "thoughts.jsonl"
TOK = re.compile(r"\w+|[^\w\s]+")


# --------------------------------------------------------------------------------------------- seams


def det() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0142 A1: {MODULE} (degenerate, Trip, MAX_UNIT_TOKENS, MIN_REPEATS, MIN_CHAR_RUN) does not exist yet"
        ) from exc


def degenerate(text: str) -> Any:
    fn = getattr(det(), "degenerate", None)
    assert callable(fn), "T-0142 A1: degeneracy.degenerate(text) does not exist yet"
    return fn(text)


def fixture_rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]


def texts(row: dict[str, Any]) -> list[str]:
    return [c["text"] for c in row["thoughts"]]


def first_trip(commit_texts: list[str]) -> int | None:
    """The 1-based ordinal of the commit at which the cumulative text first trips (``None``: never)."""
    acc = ""
    for i, t in enumerate(commit_texts, 1):
        acc += t
        if degenerate(acc) is not None:
            return i
    return None


def unit_tokens(trip: Any) -> list[str]:
    return TOK.findall(str(trip.unit).casefold())


def run_present(text: str, unit: list[str], repeats: int) -> bool:
    """Whether ``unit`` occurs ``repeats`` times back to back in the tokens of ``text`` (a check, not a detector)."""
    toks = TOK.findall(text.casefold())
    block = unit * repeats
    n = len(block)
    return any(toks[i : i + n] == block for i in range(len(toks) - n + 1))


# --------------------------------------------------------------------------------------------- A1: constants, Trip


def test_constants_and_trip_shape() -> None:
    """A1: the three constants at their ruled values (H-T142-2: ``MIN_REPEATS = 8``) and ``Trip`` a frozen dataclass
    with exactly ``unit``, ``repeats``, ``kind``."""
    m = det()
    assert (
        getattr(m, "MAX_UNIT_TOKENS", None),
        getattr(m, "MIN_REPEATS", None),
        getattr(m, "MIN_CHAR_RUN", None),
    ) == (3, 8, 64), "T-0142 A1: MAX_UNIT_TOKENS = 3, MIN_REPEATS = 8, MIN_CHAR_RUN = 64"
    trip_cls = getattr(m, "Trip", None)
    assert isinstance(trip_cls, type) and dataclasses.is_dataclass(trip_cls), (
        "T-0142 A1: degeneracy.Trip is a dataclass"
    )
    assert [f.name for f in dataclasses.fields(trip_cls)] == ["unit", "repeats", "kind"]
    trip = degenerate("the " * 8)
    assert isinstance(trip, trip_cls), trip
    with pytest.raises(dataclasses.FrozenInstanceError):
        trip.repeats = 1  # type: ignore[misc]


# --------------------------------------------------------------------------------------------- A1 on the logs


def test_trips_on_fixture_collapses() -> None:
    """A1/A2: every one of the 10 logged collapses trips on its cumulative thought text, at or before the thought
    commit where the K = 12 rule trips (recorded in the fixture by the test-author's own census)."""
    rows = [r for r in fixture_rows() if r["class"] == "collapsed"]
    assert len(rows) == 10, f"premise: the fixture holds the 10 collapses; got {len(rows)}"
    late = []
    for r in rows:
        k = first_trip(texts(r))
        if k is None or k > r["k12_commit"]:
            late.append((r["source"], r["request_id"], k, r["k12_commit"]))
    assert not late, f"T-0142 A1: collapses not tripped by the K = 12 commit: {late}"
    whole = degenerate("".join(texts(rows[0])))
    assert whole is not None and whole.kind in ("ngram", "char_run"), whole


def test_clean_fixture_rows_never_trip() -> None:
    """A1 (adversarial): none of the 40 most repetitive clean logged requests trips: the deal lists ``- X: $..``,
    ``Actually, ... Actually, ...`` and every request whose longest back-to-back unit repeat is 2."""
    rows = [r for r in fixture_rows() if r["class"] == "clean"]
    assert len(rows) == 40, f"premise: 40 clean rows; got {len(rows)}"
    joined = ["".join(texts(r)) for r in rows]
    assert sum(t.count("Actually,") >= 2 for t in joined) >= 10, "premise: the Actually rows"
    assert sum(bool(re.search(r"^\s*- [^:\n]+: \$", t, re.M)) for t in joined) >= 10, (
        "premise: the deal lists"
    )
    assert sum(r["max_repeat"] == 2 for r in rows) == 20, "premise: the repeat-2 rows"
    tripped = [
        (r["source"], r["request_id"], i)
        for r in rows
        for i in [first_trip(texts(r))]
        if i is not None
    ]
    assert not tripped, f"T-0142 A1: clean logged requests tripped: {tripped}"


#: by hand: real-looking text with a unit repeated 2 to 7 times back to back (never >= MIN_REPEATS), each a reading a
#: person would call normal or merely emphatic. Kills ``MIN_REPEATS`` in 3..7.
CLEAN_BY_HAND = (
    "I think the the button is on the left.",  # a "the the" pair (repeat 2)
    "very very very good, the cart is right.",  # 3
    "ha ha ha ha, that was the wrong page.",  # 4
    "Actually, the mug. Actually, the towel. Actually, the bowl.",  # a phrase, not back to back
    "- Ceramic teapot: $32.00\n- Oak board: $24.00\n- Linen napkins: $18.00\n- Glass carafe: $16.00\n",
    "no, no, no, no, no, not that one.",  # ", no" x 4 (2-token unit, repeat 4)
    "go on go on go on go on go on now",  # 2-token unit x 5
    "wait wait wait wait wait wait wait, I see it.",  # 7: one short of the rule
    "one two three one two three one two three one two three one two three one two three one two three end",  # 3-token x 7
    "Loading" + "." * 63 + " done",  # a 63-character run
    "a b c d " * 20,  # a 4-token unit never trips
    " " * 200 + "\n" * 100 + "ok",  # whitespace runs never count
)


@pytest.mark.parametrize(
    "text", CLEAN_BY_HAND, ids=[f"clean{i}" for i in range(len(CLEAN_BY_HAND))]
)
def test_clean_rows_never_trip(text: str) -> None:
    """A1 (adversarial; mutant ``MIN_REPEATS = 3``, and any value in 3..7): text below the rule does not trip."""
    assert degenerate(text) is None, f"T-0142 A1: {text[:60]!r} tripped: {degenerate(text)}"


#: (text, expected kind or None, expected unit tokens, expected repeats): written out by hand from A1's rule
BOUNDARIES: tuple[tuple[str, str | None, list[str], int], ...] = (
    ("the " * 7, None, [], 0),
    ("the " * 8, "ngram", ["the"], 8),
    ("the. " * 7, None, [], 0),
    ("the. " * 8, "ngram", ["the", "."], 8),
    (", the" * 8, "ngram", [",", "the"], 8),
    ("the to " * 7, None, [], 0),
    ("the to " * 8, "ngram", ["the", "to"], 8),
    ("is the to " * 7, None, [], 0),
    ("is the to " * 8, "ngram", ["is", "the", "to"], 8),
    ("is the to it " * 8, None, [], 0),  # a 4-token unit: never
    ("is the to it " * 40, None, [], 0),
    ("x" + "." * 63, None, [], 0),
    ("x" + "." * 64, "char_run", ["."], 64),
    ("Wait " + "-" * 64 + " ok", "char_run", ["-"], 64),
    ("THE The the tHe THE the The THE", "ngram", ["the"], 8),  # casefold equality
    # casefold, not lower(): "STRASSE".casefold() == "straße".casefold() == "strasse"; lower() keeps "straße"
    ("STRASSE straße " * 4, "ngram", ["strasse"], 8),
)


@pytest.mark.parametrize(
    ("text", "kind", "unit", "repeats"),
    BOUNDARIES,
    ids=[
        f"{k or 'none'}-{'_'.join(u) or 'x'}-{n}-{i}" for i, (_, k, u, n) in enumerate(BOUNDARIES)
    ],
)
def test_boundaries(text: str, kind: str | None, unit: list[str], repeats: int) -> None:
    """A1 (mutations 3, off by one): 7 vs 8 repeats of a 1-, 2- and 3-token unit, a 4-token unit never, a 63 vs 64
    character run, casefold equality (``ß``), each with the expected kind, unit and count."""
    trip = degenerate(text)
    if kind is None:
        assert trip is None, f"T-0142 A1: {text[:60]!r} must not trip; got {trip}"
        return
    assert trip is not None, f"T-0142 A1: {text[:60]!r} must trip ({kind}, {unit} x {repeats})"
    assert trip.kind == kind, trip
    assert trip.repeats == repeats, trip
    if kind == "ngram":
        assert unit_tokens(trip) == unit, trip
    else:
        assert str(trip.unit) == unit[0], trip


def test_straddles_canvases() -> None:
    """A1 (adversarial; mutant: each commit judged alone): 5 repeats closing commit 1 and 3 opening commit 2 trip on
    the concatenation, while neither commit trips alone."""
    c1 = "I clicked the button and now I see" + " the" * 5
    c2 = " the" * 3 + " page has changed."
    assert degenerate(c1) is None and degenerate(c2) is None, "premise: neither commit alone trips"
    trip = degenerate(c1 + c2)
    assert trip is not None and trip.kind == "ngram" and unit_tokens(trip) == ["the"], trip
    assert trip.repeats == 8, trip
    assert first_trip([c1, c2]) == 2


def test_pure() -> None:
    """A1: stdlib-only imports and no I/O names in the module (an AST check), and deterministic (two calls agree,
    a ``Trip`` is hashable and compares by value)."""
    det()
    tree = ast.parse(MODULE_PATH.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "T-0142 A1: no relative (project) imports"
            imported.add((node.module or "").split(".")[0])
    imported.discard("__future__")
    non_std = sorted(m for m in imported if m not in sys.stdlib_module_names)
    assert not non_std, f"T-0142 A1: stdlib-only; imports {non_std}"
    io_mods = {
        "os",
        "io",
        "pathlib",
        "socket",
        "subprocess",
        "time",
        "random",
        "asyncio",
        "logging",
        "shutil",
        "tempfile",
        "urllib",
        "http",
        "threading",
        "datetime",
        "sys",
    }
    assert not (imported & io_mods), f"T-0142 A1: pure; imports {sorted(imported & io_mods)}"
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not (called & {"open", "print", "input", "eval", "exec"}), sorted(called)
    row = next(r for r in fixture_rows() if r["class"] == "collapsed")
    text = "".join(texts(row))
    a, b = degenerate(text), degenerate(text)
    assert a == b and hash(a) == hash(b), (a, b)


# --------------------------------------------------------------------------------------------- Hypothesis

ALPHABET = ("the", ".", ",", "to", "a", "The")


def oracle_max_run(tokens: list[str]) -> int:
    """Brute force: the longest back-to-back run of any 1-3 token unit, over every start position."""
    toks = [t.casefold() for t in tokens]
    best = 0
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            unit = toks[i : i + n]
            k = 1
            while toks[i + k * n : i + (k + 1) * n] == unit:
                k += 1
            best = max(best, k)
    return best


#: a chunk is a unit of 1-3 tokens repeated 1-10 times, so runs near the threshold (7, 8, 9) are common
CHUNKS = st.tuples(
    st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=3), st.integers(1, 10)
).map(lambda c: c[0] * c[1])
TOKENS = st.lists(CHUNKS, max_size=5).map(lambda cs: [t for c in cs for t in c])


@settings(max_examples=400, deadline=None, database=None)
@given(TOKENS, st.sampled_from([" ", "\n", "  "]))
def test_matches_brute_force_oracle(tokens: list[str], sep: str) -> None:
    """A1 against an independent oracle: space-separated tiny token sequences (no character run can reach 64) trip
    iff some 1-3 token unit repeats at least 8 times back to back; whitespace never matters."""
    text = sep.join(tokens)
    expected = oracle_max_run(tokens) >= 8
    trip = degenerate(text)
    assert (trip is not None) == expected, (text, trip, oracle_max_run(tokens))
    if trip is not None:
        assert trip.kind == "ngram" and trip.repeats >= 8, trip
        assert run_present(text, unit_tokens(trip), trip.repeats), trip


@settings(max_examples=200, deadline=None, database=None)
@given(
    st.lists(st.sampled_from(ALPHABET), max_size=30),
    st.integers(min_value=8, max_value=20),
    st.lists(st.sampled_from(ALPHABET), max_size=30),
)
def test_trip_survives_appended_text(before: list[str], k: int, after: list[str]) -> None:
    """A1/A3: the session judges the growing concatenation, so a trip must stay a trip when text is appended after a
    token boundary."""
    text = " ".join([*before, *(["the"] * k)])
    assert degenerate(text) is not None, text
    assert degenerate(text + " " + " ".join(after)) is not None, (text, after)


# --------------------------------------------------------------------------------------------- A2 on synthetic logs


def require_script() -> None:
    assert SCRIPT.exists(), "T-0142 A2: scripts/replay_degeneracy.py does not exist yet"


def write_dir(root: Path, sessions: dict[str, list[dict[str, Any]]]) -> Path:
    """A synthetic A/B log dir: ``trials.jsonl`` plus ``sessions/<name>`` with the given records."""
    (root / "sessions").mkdir(parents=True)
    trials = []
    for i, (name, records) in enumerate(sessions.items()):
        lines = [
            json.dumps({"seq": n, "kind": r["kind"], "data": r["data"]})
            for n, r in enumerate(records)
        ]
        (root / "sessions" / name).write_text("\n".join(lines) + "\n")
        trials.append(json.dumps({"task": "S9", "repeat": i, "session_log": name}))
    (root / "trials.jsonl").write_text("\n".join(trials) + "\n")
    return root


def request(attempt: str, phase: str = "tool_step") -> dict[str, Any]:
    return {
        "kind": "llm.request",
        "data": {"phase": phase, "request_id": attempt.split("@")[0], "attempt_id": attempt},
    }


def commit(attempt: str, text: str, segment: str = "thought", idx: int = 0) -> dict[str, Any]:
    return {
        "kind": "llm.commit",
        "data": {
            "request_id": attempt.split("@")[0],
            "attempt_id": attempt,
            "canvas_idx": idx,
            "segment": segment,
            "text": text,
            "thought_open": segment == "thought",
        },
    }


def replay(*args: str) -> subprocess.CompletedProcess[str]:
    require_script()
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=120
    )


def line_of(out: str, attempt: str) -> str:
    found = [ln for ln in out.splitlines() if attempt in re.split(r"[\s=]+", ln)]
    assert len(found) == 1, f"R-2: one line naming {attempt}; got {found}\n{out}"
    return found[0]


def status_of(line: str) -> str:
    words = set(re.findall(r"[a-z]+", line))
    got = words & {"tripped", "clean"}
    assert len(got) == 1, f"R-2: a line says exactly one of tripped/clean: {line!r}"
    return got.pop()


def commit_of(line: str) -> int:
    m = re.search(r"commit=(\d+)", line)
    assert m, f"R-2: a tripped line carries commit=<k>: {line!r}"
    return int(m.group(1))


def synthetic(tmp_path: Path) -> Path:
    """One session: a straddling collapse (5 + 3 + more, trips at thought commit 2), a clean request with a
    degenerate *answer* (never judged), a request with no thought commit, a ReGround pass 2 that collapses at its
    first commit, and a prefill (no commits)."""
    return write_dir(
        tmp_path / "run",
        {
            "s1.jsonl": [
                request("t1-prefill@p1", "prefill"),
                request("t1-gen@g1", "generate"),
                commit("t1-gen@g1", "Clean plan: click the cart."),
                commit("t1-gen@g1", "the " * 30, segment="answer"),
                request("t1-gen-s2@s2"),
                commit("t1-gen-s2@s2", "I look at the page and" + " the" * 5),
                commit("t1-gen-s2@s2", " the" * 3 + " then", idx=1),
                commit("t1-gen-s2@s2", " the" * 40, idx=2),
                request("t1-gen-s3@s3"),
                commit("t1-gen-s3@s3", "Done.", segment="answer"),
                request("t1-call-s2-reground@r1", "reground"),
                commit("t1-call-s2-reground@r1", "ok " + "." * 70),
            ]
        },
    )


def test_replay_reports_trips_on_synthetic_dir(tmp_path: Path) -> None:
    """A2 (R-2): per request with thought commits, ``tripped`` with the commit index or ``clean``; the straddling
    collapse trips at commit 2 (5 + 3), a pass-2 collapse at commit 1, a degenerate answer is not judged; exit
    non-zero because requests tripped."""
    run = synthetic(tmp_path)
    proc = replay(str(run))
    assert proc.returncode != 0, (
        f"A2: trips without --expect exit non-zero\n{proc.stdout}{proc.stderr}"
    )
    out = proc.stdout
    s2 = line_of(out, "t1-gen-s2@s2")
    assert status_of(s2) == "tripped" and commit_of(s2) == 2, s2
    r1 = line_of(out, "t1-call-s2-reground@r1")
    assert status_of(r1) == "tripped" and commit_of(r1) == 1, r1
    assert status_of(line_of(out, "t1-gen@g1")) == "clean", "R-o: an answer is never judged"
    tripped = [ln for ln in out.splitlines() if re.search(r"\btripped\b", ln)]
    assert len([ln for ln in tripped if "@" in ln]) == 2, out


def test_replay_expect(tmp_path: Path) -> None:
    """A2: ``--expect N`` exits 0 when exactly N requests tripped, non-zero otherwise; a clean directory exits 0."""
    run = synthetic(tmp_path)
    assert replay(str(run), "--expect", "2").returncode == 0
    assert replay(str(run), "--expect", "1").returncode != 0
    assert replay(str(run), "--expect", "3").returncode != 0
    clean = write_dir(
        tmp_path / "quiet",
        {"c.jsonl": [request("t1-gen@g1", "generate"), commit("t1-gen@g1", "the the cart.")]},
    )
    proc = replay(str(clean))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert status_of(line_of(proc.stdout, "t1-gen@g1")) == "clean"
    assert replay(str(clean), "--expect", "1").returncode != 0


def test_replay_many_dirs_and_read_only(tmp_path: Path) -> None:
    """A2: several directories in one run (their trips summed against ``--expect``), and nothing in them changes."""
    a = synthetic(tmp_path / "a")
    b = synthetic(tmp_path / "b")

    def snapshot() -> dict[str, bytes]:
        return {str(p): p.read_bytes() for d in (a, b) for p in sorted(d.rglob("*")) if p.is_file()}

    before = snapshot()
    proc = replay(str(a), str(b), "--expect", "4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert snapshot() == before, "A2: the replay is read-only"
