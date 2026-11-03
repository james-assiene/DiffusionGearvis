"""T-0142b A2 on the real logs: ``scripts/replay_degeneracy.py --segment answer`` over the three A/B run directories
reports exactly 1 request tripped, attempt ``5008519a4296-t1-gen-s2-4`` at its first answer commit, and the 1,665
others clean (docs/tasks/T-0142b.md, Goal "Evidence" and A2); plus the chat logs of H-T142b-3.

The expected values are the planner's, cross-checked here by this module's own reading of ``trials.jsonl`` and the
session files (never from the script's output), with the test-author's census in
``tests/fixtures/t0142b/build_answers.py`` (brute force, independent of ``degeneracy.py``). The output format is R-2 of
``tests/unit/test_t0142b_answer_detector.py``: the task's "answer commit 0" is ``commit=1``.

**Chat logs (H-T142b-3, read-only):** the 7 session files under ``logs/sessions/`` that carry answer commits
(ordinary voice chats from 2026-09-16 to 2026-10-20; 30 requests, 22 ``generate`` and 8 ``tool_step``). They are read
here only, never copied into the repo. No other directory under ``logs/`` was used.

The logs are untracked and live in the main checkout. Each directory is looked up under this checkout's ``logs/``,
then under the main checkout's (the git common dir's parent). Every row skips, naming the paths tried, when one is
absent.

Regime: T-0132 at 2ccff41 (3 arms x 100, two with ReGround on), T-0134 at e59aa69 and T-0135 at e1ae76e (lean, CL
256, 48 steps, NVFP4 rev 2ea8372, vLLM 0.29.0, 1120 visual tokens; T-0135 2 arms x 100, order seed 0, engine
unseeded). CPU only, no GPU, no network.

Run (CPU; the ``gpu`` marker only gates selection, as T-0142's and T-0135's evals)::

    uv run pytest tests/eval/test_t0142b_replay_answers.py -q -m gpu -rA
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_degeneracy.py"
BUILDER = REPO / "tests" / "fixtures" / "t0142b" / "build_answers.py"
T0132 = "t0132-crop-budget-ab-2ccff41"
T0134 = "t0134-feedback-ab-e59aa69"
T0135 = "t0135-thought-call-ab-e1ae76e"
PER_DIR = {T0132: 713, T0134: 584, T0135: 369}
COLLAPSED = "5008519a4296-t1-gen-s2-4"
CHAT_SESSIONS = (
    "20260916-170934-3380576.jsonl",
    "20260916-181929-3900425.jsonl",
    "20260916-202852-987340.jsonl",
    "20260916-211716-1250570.jsonl",
    "20260916-215207-2034852.jsonl",
    "20260917-165104-374798.jsonl",
    "20261020-165855-1581291.jsonl",
)


def builder() -> Any:
    spec = importlib.util.spec_from_file_location("t0142b_build_answers", BUILDER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidates(rel: str) -> list[Path]:
    out = [REPO / "logs" / rel]
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        out.append(Path(proc.stdout.strip()).parent / "logs" / rel)
    return out


def run_dir(name: str) -> Path:
    tried = _candidates(name)
    for p in tried:
        if (p / "trials.jsonl").is_file() and (p / "sessions").is_dir():
            return p
    pytest.skip(f"T-0142b A2 eval: {name} is absent (tried {[str(p) for p in tried]})")


def three() -> tuple[Path, Path, Path]:
    return run_dir(T0132), run_dir(T0134), run_dir(T0135)


def chat_dir() -> Path:
    tried = _candidates("sessions")
    for p in tried:
        if all((p / n).is_file() for n in CHAT_SESSIONS):
            return p
    pytest.skip(f"T-0142b H-T142b-3: the chat logs are absent (tried {[str(p) for p in tried]})")


def replay(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=600
    )


def answer_replay(*args: str) -> subprocess.CompletedProcess[str]:
    proc = replay(*args, "--segment", "answer")
    assert "usage" not in proc.stderr.lower(), (
        f"T-0142b A2: scripts/replay_degeneracy.py has no --segment answer yet\n{proc.stderr}"
    )
    return proc


def answer_requests(run: Path) -> dict[str, list[str]]:
    """attempt_id -> raw answer commit texts, for every request with answer commits in ``run`` (read here)."""
    b = builder()
    out: dict[str, list[str]] = {}
    for line in (run / "trials.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        name = Path(json.loads(line)["session_log"]).name
        for r in b.answers_by_request(run / "sessions" / name):
            out[r["attempt_id"]] = [c["text"] for c in r["answers"]]
    return out


def parse(out: str, attempts: set[str]) -> dict[str, tuple[str, int | None]]:
    """attempt_id -> (status, commit) for every line naming one of ``attempts`` (R-2)."""
    got: dict[str, tuple[str, int | None]] = {}
    for line in out.splitlines():
        named = [w for w in re.split(r"[\s=]+", line) if w in attempts]
        if not named:
            continue
        assert len(named) == 1, f"R-2: one request per line: {line!r}"
        words = set(re.findall(r"[a-z]+", line)) & {"tripped", "clean"}
        assert len(words) == 1, f"R-2: exactly one of tripped/clean: {line!r}"
        status = words.pop()
        m = re.search(r"commit=(\d+)", line)
        assert named[0] not in got, f"R-2: one line per request; {named[0]} twice"
        got[named[0]] = (status, int(m.group(1)) if m and status == "tripped" else None)
    return got


def answer_judge() -> Any:
    det = importlib.import_module("local_astra.conductor.degeneracy")
    k = getattr(det, "MIN_REPEATS_ANSWER", None)
    assert k == 6, f"T-0142b A1: degeneracy.MIN_REPEATS_ANSWER = 6 does not exist yet (got {k!r})"
    assert "min_repeats" in inspect.signature(det.degenerate).parameters, (
        "T-0142b A1: degenerate(text, *, min_repeats=...) does not exist yet"
    )
    return lambda text: det.degenerate(text, min_repeats=k)


def test_premise_answer_census() -> None:
    """The three directories hold 1,666 requests with answer commits (713 / 584 / 369); the collapse is there with
    3 answer commits; the 1,665 others repeat no 1-3 token unit twice back to back, run no non-whitespace character
    more than twice, and have a median of 11 tokens and a maximum of 58 (this module's reading)."""
    b = builder()
    reqs: dict[str, list[str]] = {}
    for run in three():
        got = answer_requests(run)
        assert len(got) == PER_DIR[run.name], (run.name, len(got))
        reqs.update(got)
    assert len(reqs) == 1666
    assert len(reqs[COLLAPSED]) == 3
    clean = [("".join(t)) for a, t in reqs.items() if a != COLLAPSED]
    assert len(clean) == 1665
    assert max(b.max_unit_repeat(t) for t in clean) == 1
    assert max(b.max_char_run(t) for t in clean) == 2
    lengths = [len(b.tokens(t)) for t in clean]
    assert statistics.median(lengths) == 11 and max(lengths) == 58


def test_exactly_one_answer_trips() -> None:
    """A2: with ``--segment answer`` over the three directories exactly ``5008519a4296-t1-gen-s2-4`` is ``tripped``,
    at answer commit 1 (the task's "commit 0"); every other request with answer commits is ``clean``; exit non-zero
    without ``--expect``."""
    runs = three()
    reqs: dict[str, list[str]] = {}
    for run in runs:
        reqs.update(answer_requests(run))
    proc = answer_replay(*map(str, runs))
    assert proc.returncode != 0, "A2: a trip without --expect exits non-zero"
    got = parse(proc.stdout, set(reqs))
    missing = sorted(set(reqs) - set(got))
    assert not missing, (
        f"R-2: {len(missing)} requests with answer commits have no line, e.g. {missing[:3]}"
    )
    tripped = {k: v[1] for k, v in got.items() if v[0] == "tripped"}
    assert tripped == {COLLAPSED: 1}, f"T-0142b A2: tripped {tripped}"
    assert sum(1 for v in got.values() if v[0] == "clean") == 1665


def test_expect_answer() -> None:
    """A2: ``--segment answer --expect 1`` exits 0 over the three directories; ``--expect 0`` does not; per
    directory T-0135 holds the 1 and T-0132 / T-0134 none."""
    a, b, c = three()
    assert answer_replay(str(a), str(b), str(c), "--expect", "1").returncode == 0
    assert answer_replay(str(a), str(b), str(c), "--expect", "0").returncode != 0
    assert answer_replay(str(c), "--expect", "1").returncode == 0
    assert answer_replay(str(a), str(b)).returncode == 0, "no answer trip in T-0132 / T-0134"


def test_thought_segment_unchanged_on_t0135() -> None:
    """A2 (the default kept): ``--segment thought`` over the T-0135 directory reports the planner's 6 thought collapses
    in 562 requests with thought commits (``--expect 6`` exits 0), exactly as no ``--segment`` does."""
    run = run_dir(T0135)
    default = replay(str(run), "--expect", "6")
    explicit = replay(str(run), "--segment", "thought", "--expect", "6")
    assert default.returncode == 0, default.stderr
    assert (explicit.returncode, explicit.stdout) == (default.returncode, default.stdout), (
        explicit.stderr
    )
    assert "6 tripped, 556 clean" in default.stdout, default.stdout[-300:]


def test_clean_logged_answers_never_trip() -> None:
    """A1 on all 1,665 clean logged answers: ``degenerate(text, min_repeats=MIN_REPEATS_ANSWER)`` is ``None`` after
    every answer commit; the collapse trips at its first."""
    judge = answer_judge()
    for run in three():
        for attempt, texts in answer_requests(run).items():
            acc, first = "", None
            for i, t in enumerate(texts, 1):
                acc += t
                if judge(acc) is not None:
                    first = i
                    break
            assert first == (1 if attempt == COLLAPSED else None), (run.name, attempt, first)


def test_chat_answers_never_trip() -> None:
    """H-T142b-3: the 30 answer-bearing requests of the 7 named chat sessions under ``logs/sessions/`` never trip at
    ``MIN_REPEATS_ANSWER``, after any commit (census: longest unit repeat 1, longest non-whitespace character run 4)."""
    judge = answer_judge()
    b = builder()
    d = chat_dir()
    reqs = [r for name in CHAT_SESSIONS for r in b.answers_by_request(d / name)]
    assert len(reqs) == 30, f"premise: 30 chat requests with answer commits; got {len(reqs)}"
    full = ["".join(c["text"] for c in r["answers"]) for r in reqs]
    assert max(b.max_unit_repeat(t) for t in full) == 1
    assert max(b.max_char_run(t) for t in full) == 4
    tripped = []
    for r in reqs:
        acc = ""
        for c in r["answers"]:
            acc += c["text"]
            if judge(acc) is not None:
                tripped.append(r["attempt_id"])
                break
    assert tripped == [], f"T-0142b H-T142b-3: chat answers tripped: {tripped}"
