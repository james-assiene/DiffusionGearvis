"""T-0142b A1 and A2 (synthetic logs): the detector's answer threshold and the replay script's ``--segment``
(docs/tasks/T-0142b.md).

**The interfaces these rows pin** (A1/A2, plus the readings listed in the test-author's report):

- R-1: ``degenerate(text: str, *, min_repeats: int = MIN_REPEATS) -> Trip | None``: ``min_repeats`` is keyword-only
  and its default is the module's ``MIN_REPEATS`` (8). ``MIN_REPEATS_ANSWER = 6`` (H-T142b-2) is a module constant of
  ``local_astra.conductor.degeneracy``. ``min_repeats`` replaces ``MIN_REPEATS`` in the n-gram rule only; the character
  rule stays ``MIN_CHAR_RUN = 64`` whatever ``min_repeats`` is (A1: "``MIN_CHAR_RUN = 64`` ... unchanged").
- R-2: ``scripts/replay_degeneracy.py <log-dir>... [--segment {thought,answer}] [--expect N]``. With ``--segment
  answer`` the script prints one line per request with at least one ``answer`` commit, in T-0142's R-2 format
  (``attempt=<attempt_id>``, exactly one of ``tripped``/``clean``; a ``tripped`` line holds ``commit=<k>``, the 1-based
  ordinal of the *answer* commit at which ``degenerate(..., min_repeats=MIN_REPEATS_ANSWER)`` first trips on the
  raw answer text so far, concatenated with no separator). The task's "answer commit 0" is ``commit=1`` here.
  ``--segment thought`` and no ``--segment`` print exactly what the BASE script prints (byte for byte, stdout and exit
  status); an unknown segment value exits non-zero.

Every row here fails at BASE 0248cfd on a named ``AssertionError`` (no ``min_repeats`` keyword, no
``MIN_REPEATS_ANSWER``, no ``--segment``), never on an import error or a ``TypeError``, except
``test_default_matches_base_script`` (green at BASE and after: the default is T-0142's output byte for byte).

Adversarial rows: ``test_clean_answers_never_trip`` (mutant ``MIN_REPEATS_ANSWER = 2``, and any value in 2..5, by the
hand table's 2-, 4- and 5-repeat rows), ``test_answer_threshold_boundaries`` (6 vs 7/8, off by one),
``test_default_is_thought_threshold`` (the default lowered to 6), ``test_straddles_answer_commits`` (each commit judged
alone), and a Hypothesis row against a brute-force oracle for every ``min_repeats`` in 2..10.

Regime: CPU, pure; fixture ``tests/fixtures/t0142b/answers.jsonl`` (T-0132 at 2ccff41, T-0134 at e59aa69, T-0135 at
e1ae76e; see ``build_answers.py``); n = 1, deterministic.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.conductor.degeneracy"
SCRIPT = REPO / "scripts" / "replay_degeneracy.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0142b" / "answers.jsonl"
TOK = re.compile(r"\w+|[^\w\s]+")
#: T-0142b's BASE: main at 0248cfd (after ``Merge T-0142``). A literal.
BASE_SHA = "0248cfdaa56792836f2b2e7e7c72a9c5d9c68af3"
COLLAPSED_ATTEMPT = "5008519a4296-t1-gen-s2-4"


# --------------------------------------------------------------------------------------------- seams


def det() -> Any:
    return importlib.import_module(MODULE)


def answer_k() -> int:
    value = getattr(det(), "MIN_REPEATS_ANSWER", None)
    assert isinstance(value, int) and not isinstance(value, bool), (
        "T-0142b A1: degeneracy.MIN_REPEATS_ANSWER (an int module constant) does not exist yet"
    )
    return value


def require_keyword() -> None:
    params = inspect.signature(det().degenerate).parameters
    assert "min_repeats" in params, (
        "T-0142b A1: degenerate(text, *, min_repeats: int = MIN_REPEATS) does not exist yet"
    )
    p = params["min_repeats"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, f"T-0142b A1 (R-1): keyword-only; got {p.kind}"
    assert p.default == 8 == det().MIN_REPEATS, (
        f"T-0142b A1 (R-1): the default is MIN_REPEATS (8); got {p.default!r}"
    )


def degenerate(text: str, **kw: Any) -> Any:
    require_keyword()
    return det().degenerate(text, **kw)


def judge_answer(text: str) -> Any:
    return degenerate(text, min_repeats=answer_k())


def fixture_rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]


def texts(row: dict[str, Any]) -> list[str]:
    return [c["text"] for c in row["answers"]]


def first_trip(commit_texts: list[str], k: int) -> int | None:
    """The 1-based ordinal of the commit at which the cumulative text first trips at ``k`` (``None``: never)."""
    acc = ""
    for i, t in enumerate(commit_texts, 1):
        acc += t
        if degenerate(acc, min_repeats=k) is not None:
            return i
    return None


def unit_tokens(trip: Any) -> list[str]:
    return TOK.findall(str(trip.unit).casefold())


# --------------------------------------------------------------------------------------------- A1: constants


def test_constants() -> None:
    """A1 / H-T142b-2: ``MIN_REPEATS_ANSWER = 6``; T-0142's constants unchanged (3, 8, 64); the keyword is
    keyword-only with the thought default (R-1)."""
    assert answer_k() == 6, f"T-0142b A1 (H-T142b-2): MIN_REPEATS_ANSWER = 6; got {answer_k()}"
    require_keyword()
    m = det()
    assert (m.MAX_UNIT_TOKENS, m.MIN_REPEATS, m.MIN_CHAR_RUN) == (3, 8, 64)


#: (text, expected kind or None, expected unit tokens, expected repeats) at min_repeats = 6, by hand from A1's rule
BOUNDARIES: tuple[tuple[str, str | None, list[str], int], ...] = (
    ("the " * 5, None, [], 0),
    ("the " * 6, "ngram", ["the"], 6),
    ("the. " * 5, None, [], 0),
    ("the. " * 6, "ngram", ["the", "."], 6),
    ("is the to " * 5, None, [], 0),
    ("is the to " * 6, "ngram", ["is", "the", "to"], 6),
    ("is the to it " * 6, None, [], 0),  # a 4-token unit: never
    ("is the to it " * 40, None, [], 0),
    ("x" + "." * 63, None, [], 0),  # the character rule is not lowered with min_repeats
    ("x" + "." * 64, "char_run", ["."], 64),
)


@pytest.mark.parametrize(
    ("text", "kind", "unit", "repeats"),
    BOUNDARIES,
    ids=[
        f"{k or 'none'}-{'_'.join(u) or 'x'}-{n}-{i}" for i, (_, k, u, n) in enumerate(BOUNDARIES)
    ],
)
def test_answer_threshold_boundaries(
    text: str, kind: str | None, unit: list[str], repeats: int
) -> None:
    """A1 (mutation 3, off by one): a 1-, 2- and 3-token unit repeated 5 times is clean and 6 times trips at
    ``min_repeats=MIN_REPEATS_ANSWER``; a 4-token unit never trips; a 63-character run is clean and 64 trips."""
    trip = judge_answer(text)
    if kind is None:
        assert trip is None, f"T-0142b A1: {text[:60]!r} must not trip at 6; got {trip}"
        return
    assert trip is not None, (
        f"T-0142b A1: {text[:60]!r} must trip at 6 ({kind}, {unit} x {repeats})"
    )
    assert (trip.kind, trip.repeats) == (kind, repeats), trip
    if kind == "ngram":
        assert unit_tokens(trip) == unit, trip
    else:
        assert str(trip.unit) == unit[0], trip


@pytest.mark.parametrize(("n", "expected"), [(6, False), (7, False), (8, True)])
def test_default_is_thought_threshold(n: int, expected: bool) -> None:
    """A1 / A8 (mutation 4, the default lowered to 6): without ``min_repeats`` the detector is T-0142's: 6 and 7
    repeats are clean and 8 trips, for a 1- and a 2-token unit; passing ``MIN_REPEATS`` explicitly is the same."""
    for text in ("the " * n, "Wait," + " the." * n):
        got = degenerate(text)
        assert (got is not None) is expected, (
            f"T-0142b A1: degenerate({text[:40]!r}) default; got {got}"
        )
        assert degenerate(text, min_repeats=det().MIN_REPEATS) == got


def test_logged_collapse_trips_at_commit_0() -> None:
    """A1/A2 (the regression; mutant ``MIN_REPEATS_ANSWER = 7`` or ``8``): the logged S2#5 attempt trips on its first
    answer commit alone, ``. the`` x 6; at the thought default the same commit is clean (the voiced-cost argument)."""
    (row,) = [r for r in fixture_rows() if r["class"] == "collapsed"]
    assert row["attempt_id"] == COLLAPSED_ATTEMPT and len(row["answers"]) == 3, (
        "premise: the fixture"
    )
    first = texts(row)[0]
    assert first == "Wait, let me check the cart. the. the. the. the. the. the.", (
        "premise: commit 0"
    )
    trip = judge_answer(first)
    assert trip is not None, (
        "T-0142b A1: the logged collapse's first answer commit trips at MIN_REPEATS_ANSWER"
    )
    assert trip.kind == "ngram" and trip.repeats == 6, trip
    assert sorted(unit_tokens(trip)) == [".", "the"], trip
    assert first_trip(texts(row), answer_k()) == 1
    assert degenerate(first) is None, "premise: at the thought threshold commit 0 would be voiced"
    assert first_trip(texts(row), 8) == 2


#: by hand: conversational prose (H-T142b-3's gap), each a reading a person would call normal or merely emphatic
CHAT_BY_HAND = (
    "red, blue, green",
    "- Red mug: $8\n- Blue mug: $9\n- Green mug: $10",
    "1. Open the page. 2. Click add. 3. Check the cart.",
    "No, no, no, no, no.",  # ", no" x 4 as tokenised
    "very, very, very good",
    "bye bye",
    "the the",
    "Well... I think so.",
    "Ha ha ha ha ha, that is a good one.",  # 5: one short of the answer rule
    "Thanks! Thanks! Thanks!",
    "It is 10:10:10 on the clock, and the total is $1,111.11.",
)
#: a 58-token answer (the longest clean logged answer is 58 tokens): counted below, not trusted
LONG_58 = (
    "I have filled out the contact form with your name, Ada Lovelace, and your email address, "
    "ada@example.com, and then I clicked the Send button. The page now shows a confirmation that "
    "reads thanks, we will be in touch soon. Is there anything else you need today?"
)


def test_clean_answers_never_trip() -> None:
    """A1 (adversarial; mutant ``MIN_REPEATS_ANSWER = 2``, and any value in 2..5): none of the 40 highest-scoring
    clean logged answers trips at ``MIN_REPEATS_ANSWER``, nor any row of the hand table of conversational prose (lists,
    numbers, repeated words, punctuation, a 58-token answer)."""
    rows = [r for r in fixture_rows() if r["class"] == "clean"]
    assert len(rows) == 40, f"premise: 40 clean rows; got {len(rows)}"
    assert len(TOK.findall(LONG_58.casefold())) == 58, "premise: the long answer is 58 tokens"
    tripped = [
        (r["source"], r["attempt_id"]) for r in rows if first_trip(texts(r), answer_k()) is not None
    ]
    assert not tripped, f"T-0142b A1: clean logged answers tripped: {tripped}"
    for text in (*CHAT_BY_HAND, LONG_58):
        trip = judge_answer(text)
        assert trip is None, f"T-0142b A1: {text[:60]!r} tripped at MIN_REPEATS_ANSWER: {trip}"


def test_straddles_answer_commits() -> None:
    """A1/A3 (adversarial; mutant: each commit judged alone): 3 repeats closing answer commit 0 and 3 opening commit 1
    trip at commit 1 on the concatenation; neither commit trips alone."""
    k = answer_k()
    c0 = "Let me check the cart. the. the. the."
    c1 = " the. the. the. Done."
    assert judge_answer(c0) is None and judge_answer(c1) is None, (
        "premise: neither commit alone trips"
    )
    trip = judge_answer(c0 + c1)
    assert trip is not None and trip.repeats == 6 and sorted(unit_tokens(trip)) == [".", "the"], (
        trip
    )
    assert first_trip([c0, c1], k) == 2


# --------------------------------------------------------------------------------------------- Hypothesis

ALPHABET = ("the", ".", ",", "ha", "The")


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


CHUNKS = st.tuples(
    st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=3), st.integers(1, 9)
).map(lambda c: c[0] * c[1])
TOKENS = st.lists(CHUNKS, max_size=5).map(lambda cs: [t for c in cs for t in c])


@settings(max_examples=300, deadline=None, database=None)
@given(TOKENS, st.integers(min_value=2, max_value=10))
def test_min_repeats_matches_oracle(tokens: list[str], k: int) -> None:
    """A1 against an independent oracle: space-separated tiny token sequences trip at ``min_repeats=k`` iff some 1-3
    token unit repeats at least ``k`` times back to back (no character run can reach 64 here)."""
    text = " ".join(tokens)
    expected = oracle_max_run(tokens) >= k
    trip = degenerate(text, min_repeats=k)
    assert (trip is not None) == expected, (text, k, trip, oracle_max_run(tokens))
    if trip is not None:
        assert trip.kind == "ngram" and trip.repeats >= k, trip


# --------------------------------------------------------------------------------------------- A2 on synthetic logs


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


def commit(attempt: str, text: str, segment: str = "answer", idx: int = 0) -> dict[str, Any]:
    return {
        "kind": "llm.commit",
        "data": {
            "request_id": attempt.split("@")[0],
            "attempt_id": attempt,
            "canvas_idx": idx,
            "segment": segment,
            "text": text,
            "thought_open": False,
        },
    }


def synthetic(tmp_path: Path) -> Path:
    """One session, by hand:

    - ``g1``: a clean thought, then an answer collapsing at its first commit (``the.`` x 6);
    - ``s2``: a straddle, 3 + 3 repeats over two answer commits (trips at answer commit 2), then more;
    - ``s3``: a thought with 10 repeats (a thought trip) and a clean answer;
    - ``s4``: an answer with 5 repeats (clean at 6), and a thought+answer pair that would be 8 if shared (3 + 5);
    - ``p1``: a prefill, no commits; ``s5``: thought commits only."""
    return write_dir(
        tmp_path / "run",
        {
            "s1.jsonl": [
                request("t1-prefill@p1", "prefill"),
                request("t1-gen@g1", "generate"),
                commit("t1-gen@g1", "Clean plan.", segment="thought"),
                commit("t1-gen@g1", "Okay." + " the." * 6),
                commit("t1-gen@g1", " the." * 30, idx=1),
                request("t1-gen-s2@s2"),
                commit("t1-gen-s2@s2", "Let me see. the. the. the."),
                commit("t1-gen-s2@s2", " the. the. the. ok.", idx=1),
                commit("t1-gen-s2@s2", " more.", idx=2),
                request("t1-gen-s3@s3"),
                commit("t1-gen-s3@s3", "plan" + " the" * 10, segment="thought"),
                commit("t1-gen-s3@s3", "All done."),
                request("t1-gen-s4@s4"),
                commit("t1-gen-s4@s4", "I look at" + " ha" * 3, segment="thought"),
                commit("t1-gen-s4@s4", " ha" * 5 + ", fine."),
                request("t1-gen-s5@s5"),
                commit("t1-gen-s5@s5", "only a thought", segment="thought"),
            ]
        },
    )


def replay(*args: str, script: Path = SCRIPT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, timeout=120
    )


def line_of(out: str, attempt: str) -> str:
    found = [ln for ln in out.splitlines() if attempt in re.split(r"[\s=]+", ln)]
    assert len(found) == 1, f"R-2: one line naming {attempt}; got {found}\n{out}"
    return found[0]


def names(out: str) -> set[str]:
    return {w for ln in out.splitlines() for w in re.split(r"[\s=]+", ln) if "@" in w}


def status_of(line: str) -> str:
    got = set(re.findall(r"[a-z]+", line)) & {"tripped", "clean"}
    assert len(got) == 1, f"R-2: a line says exactly one of tripped/clean: {line!r}"
    return got.pop()


def commit_of(line: str) -> int:
    m = re.search(r"commit=(\d+)", line)
    assert m, f"R-2: a tripped line carries commit=<k>: {line!r}"
    return int(m.group(1))


def answer_replay(*args: str) -> subprocess.CompletedProcess[str]:
    proc = replay(*args, "--segment", "answer")
    assert "usage" not in proc.stderr.lower(), (
        f"T-0142b A2: scripts/replay_degeneracy.py has no --segment answer yet\n{proc.stderr}"
    )
    return proc


def test_replay_segment_answer(tmp_path: Path) -> None:
    """A2 (R-2): ``--segment answer`` judges each request's raw answer commits, accumulated per attempt, at
    ``MIN_REPEATS_ANSWER``: ``g1`` trips at answer commit 1, ``s2`` at 2 (the straddle), ``s3`` (a thought trip, a
    clean answer) and ``s4`` (5 repeats; thought and answer never shared) are clean, and requests without answer
    commits get no line; exit non-zero because requests tripped."""
    run = synthetic(tmp_path)
    proc = answer_replay(str(run))
    out = proc.stdout
    assert proc.returncode != 0, f"A2: trips without --expect exit non-zero\n{out}{proc.stderr}"
    g1, s2 = line_of(out, "t1-gen@g1"), line_of(out, "t1-gen-s2@s2")
    assert status_of(g1) == "tripped" and commit_of(g1) == 1, g1
    assert status_of(s2) == "tripped" and commit_of(s2) == 2, s2
    assert status_of(line_of(out, "t1-gen-s3@s3")) == "clean", (
        "A2: a thought trip is not an answer trip"
    )
    assert status_of(line_of(out, "t1-gen-s4@s4")) == "clean", (
        "A2: 5 repeats, and no shared accumulator"
    )
    assert names(out) == {"t1-gen@g1", "t1-gen-s2@s2", "t1-gen-s3@s3", "t1-gen-s4@s4"}, (
        f"R-2: one line per request with answer commits, and only those; {sorted(names(out))}"
    )


def test_replay_segment_answer_expect(tmp_path: Path) -> None:
    """A2: with ``--segment answer``, ``--expect 2`` exits 0 and ``--expect 1``/``3`` do not."""
    run = synthetic(tmp_path)
    assert answer_replay(str(run), "--expect", "2").returncode == 0
    assert answer_replay(str(run), "--expect", "1").returncode != 0
    assert answer_replay(str(run), "--expect", "3").returncode != 0


def test_replay_segment_thought_is_default(tmp_path: Path) -> None:
    """A2: ``--segment thought`` prints exactly what no ``--segment`` prints (stdout, exit status): over the same
    directory the thought trip is ``s3`` alone; an unknown segment exits non-zero."""
    run = synthetic(tmp_path)
    default = replay(str(run))
    explicit = replay(str(run), "--segment", "thought")
    assert explicit.returncode == default.returncode and explicit.stdout == default.stdout, (
        f"T-0142b A2: --segment thought is the default byte for byte\n{explicit.stderr}"
    )
    assert status_of(line_of(default.stdout, "t1-gen-s3@s3")) == "tripped"
    assert replay(str(run), "--segment", "draft").returncode != 0


def test_default_matches_base_script(tmp_path: Path) -> None:
    """A2 ("keeps T-0142's behaviour and output byte for byte"): without ``--segment`` the script's stdout, stderr
    and exit status equal BASE's script on the synthetic directory, with and without ``--expect``. Green at BASE."""
    gr.require_commits(BASE_SHA)
    proc = gr._git("show", f"{BASE_SHA}:scripts/replay_degeneracy.py")
    assert proc.returncode == 0, proc.stderr
    base_script = tmp_path / "replay_degeneracy_base.py"
    base_script.write_text(proc.stdout)
    run = synthetic(tmp_path)
    for args in ((str(run),), (str(run), "--expect", "1"), (str(run), "--expect", "2")):
        want, got = replay(*args, script=base_script), replay(*args)
        assert (got.returncode, got.stdout, got.stderr) == (
            want.returncode,
            want.stdout,
            want.stderr,
        ), args
