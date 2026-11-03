"""T-0142 A2 on the real logs: ``scripts/replay_degeneracy.py`` over both A/B run directories reports exactly the 10
logged collapses as tripped, each at or before the commit where the K = 12 rule trips, and 0 others
(docs/tasks/T-0142.md, Goal "Evidence" and A2).

The expected set is the planner's, written out by hand below by (run, arm, task#repeat, phase), and cross-checked
here against the run directories by this module's own reading of ``trials.jsonl`` and the session files (never from
the script's output): each named trial has exactly one request whose thought commits the fixture
(``tests/fixtures/t0142/thoughts.jsonl``) lists as ``collapsed``, by ``attempt_id``. The K = 12 commit of each is the
fixture's ``k12_commit`` (the test-author's census, ``build_thoughts.py``). The output format is R-2 of
``tests/unit/test_t0142_degeneracy.py``.

The logs are untracked and live in the main checkout. Each directory is looked up under this checkout's ``logs/``,
then under the main checkout's (the git common dir's parent). Every row skips, naming the paths tried, when either is
absent.

Regime: T-0134 at e59aa69 (lean, CL 256, 48 steps, NVFP4 rev 2ea8372, vLLM 0.29.0, 1120 visual tokens, ReGround
off, 3 arms x 100) and T-0132 at 2ccff41 (3 arms x 100, two with ReGround on). CPU only, no GPU, no network.

Run (CPU; the ``gpu`` marker only gates selection, as T-0135's eval)::

    uv run pytest tests/eval/test_t0142_replay_logs.py -q -m gpu -rA
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_degeneracy.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0142" / "thoughts.jsonl"
T0134 = "t0134-feedback-ab-e59aa69"
T0132 = "t0132-crop-budget-ab-2ccff41"
#: the planner's ten (Goal, "Evidence"): (run, arm, task, repeat, phase); T-0132's trials carry no arm
EXPECTED = (
    (T0134, "B", "S5", 8, "tool_step"),
    (T0134, "B", "S5", 2, "tool_step"),
    (T0134, "B", "S5", 9, "tool_step"),
    (T0134, "B", "S5", 1, "tool_step"),
    (T0134, "C", "S2", 10, "tool_step"),
    (T0134, "C", "S2", 15, "tool_step"),
    (T0132, None, "S5", 6, "tool_step"),
    (T0132, None, "S1", 5, "tool_step"),
    (T0132, None, "S5", 8, "reground"),
    (T0132, None, "S5", 6, "reground"),
)
#: requests with at least one thought commit, collapsed ones included (Goal: 10 + 1,976)
WITH_THOUGHTS = 1986


def _candidates(name: str) -> list[Path]:
    out = [REPO / "logs" / name]
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        out.append(Path(proc.stdout.strip()).parent / "logs" / name)
    return out


def run_dir(name: str) -> Path:
    tried = _candidates(name)
    for p in tried:
        if (p / "trials.jsonl").is_file() and (p / "sessions").is_dir():
            return p
    pytest.skip(f"T-0142 A2 eval: {name} is absent (tried {[str(p) for p in tried]})")


def both() -> tuple[Path, Path]:
    return run_dir(T0134), run_dir(T0132)


def replay(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.exists(), "T-0142 A2: scripts/replay_degeneracy.py does not exist yet"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=600
    )


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def thought_requests(run: Path) -> dict[str, dict[str, Any]]:
    """attempt_id -> {trial, phase, n thought commits}, read from the run directory here."""
    out: dict[str, dict[str, Any]] = {}
    for t in jsonl(run / "trials.jsonl"):
        name = Path(t["session_log"]).name
        phases: dict[str, str] = {}
        counts: dict[str, int] = {}
        for rec in jsonl(run / "sessions" / name):
            data = rec.get("data") or {}
            if rec["kind"] == "llm.request":
                phases[str(data["attempt_id"])] = str(data["phase"])
            elif rec["kind"] == "llm.commit" and data.get("segment") == "thought":
                a = str(data.get("attempt_id"))
                counts[a] = counts.get(a, 0) + 1
        for a, n in counts.items():
            if a in phases:
                out[a] = {"trial": t, "phase": phases[a], "thoughts": n, "run": run.name}
    return out


def expected_attempts() -> dict[str, int]:
    """attempt_id -> K = 12 commit, for the 10 named collapses; each named trial resolved through the fixture."""
    rows = [r for r in jsonl(FIXTURE) if r["class"] == "collapsed"]
    found: dict[str, int] = {}
    for run_name, arm, task, repeat, phase in EXPECTED:
        hits = [
            r
            for r in rows
            if r["source"].startswith(run_name + "/")
            and r["arm"] == arm
            and r["task"] == task
            and r["repeat"] == repeat
            and r["phase"] == phase
        ]
        assert len(hits) == 1, (
            f"premise: one fixture row for {run_name} {arm} {task}#{repeat} {phase}"
        )
        found[hits[0]["attempt_id"]] = int(hits[0]["k12_commit"])
    assert len(found) == 10
    return found


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


def test_premise_census_matches_logs() -> None:
    """The run directories hold 1,986 requests with thought commits, and each named trial exists with the named
    phase (this module's reading, independent of the script)."""
    a, b = both()
    reqs = {**thought_requests(a), **thought_requests(b)}
    assert len(reqs) == WITH_THOUGHTS, len(reqs)
    for attempt in expected_attempts():
        assert attempt in reqs, f"premise: {attempt} is a logged request with thought commits"
        info = reqs[attempt]
        assert 3 <= info["thoughts"] <= 19, info


def test_exactly_the_ten_trip() -> None:
    """A2: over both directories, exactly the 10 named requests are ``tripped``, each at or before its K = 12 commit;
    every other request with thought commits is ``clean``; exit non-zero without ``--expect``."""
    a, b = both()
    reqs = {**thought_requests(a), **thought_requests(b)}
    want = expected_attempts()
    proc = replay(str(a), str(b))
    assert proc.returncode != 0, "A2: trips without --expect exit non-zero"
    got = parse(proc.stdout, set(reqs))
    missing = sorted(set(reqs) - set(got))
    assert not missing, (
        f"R-2: {len(missing)} requests with thought commits have no line, e.g. {missing[:3]}"
    )
    tripped = {k: v[1] for k, v in got.items() if v[0] == "tripped"}
    assert set(tripped) == set(want), (
        f"A2: tripped {sorted(tripped)} != the planner's ten {sorted(want)}"
    )
    late = {k: (tripped[k], want[k]) for k in want if tripped[k] is None or tripped[k] > want[k]}
    assert not late, f"A2: tripped after the K = 12 commit (got, k12): {late}"
    assert sum(1 for v in got.values() if v[0] == "clean") == WITH_THOUGHTS - 10


def test_expect_matches_per_directory() -> None:
    """A2: ``--expect`` with the right count exits 0 (10 over both; 6 for T-0134, 4 for T-0132), a wrong one does
    not."""
    a, b = both()
    assert replay(str(a), str(b), "--expect", "10").returncode == 0
    assert replay(str(a), "--expect", "6").returncode == 0
    assert replay(str(b), "--expect", "4").returncode == 0
    assert replay(str(a), str(b), "--expect", "9").returncode != 0
