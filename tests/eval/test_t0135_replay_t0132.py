"""T-0135 A2's exact thresholds on the real T-0132 run directory (CPU, read-only, deterministic).

``scripts/replay_thought_calls.py logs/t0132-crop-budget-ab-2ccff41 --json`` must report: 860 eligible
``generate``/``tool_step`` requests scanned; 29 firings (S3 18, S1 7, S5 4); 28 ``valid`` and 1 ``repairable``
(``20261022-210810-1975518.jsonl``, ``t1-gen-s2``); 0 firings on the other 831; and every firing is the last
``generate``/``tool_step`` request of its turn, in a trial whose ``success`` is false and whose ``tool_loop_end`` is
``final``. The pass-2 count is 10 (informational). The ``--json`` shape is the one fixed in
``tests/unit/test_t0135_replay.py``'s docstring.

The "last request of a turn" and trial checks are read from the logs here, by this module (an independent reading of
``trials.jsonl`` and the session files), never from the script's output. The firing set must also equal the
committed fixture's (``tests/fixtures/t0135/thought_tails.jsonl``), which was extracted from the same directory.

The logs are untracked and live in the main checkout. The directory is looked up at ``$T0135_RUN_DIR``, then under
this checkout's ``logs/``, then under the main checkout's (the git common dir's parent). Every row skips, naming the
paths tried, when none exists.

Regime: logs of the T-0132 run at 2ccff41 (docs/tasks/T-0135.md "Regime"); ``trials.jsonl`` sha256 prefix
``3baf06102d33f2f9``. CPU only, no GPU, no network.

Run (CPU; the ``gpu`` marker only gates selection)::

    uv run pytest tests/eval/test_t0135_replay_t0132.py -q -m gpu -rA

The rows spend no GPU time but are marked ``gpu``, as T-0132's eval is: the data exists only after a
``/launch``-reviewed run, and every row under ``tests/eval`` must be deselected by a path-only run (T-0046,
``tests/unit/test_pytest_config.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_thought_calls.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0135" / "thought_tails.jsonl"
RUN_NAME = "t0132-crop-budget-ab-2ccff41"
TRIALS_SHA_PREFIX = "3baf06102d33f2f9"
REPAIRABLE = ("20261022-210810-1975518.jsonl", "t1-gen-s2")


def _candidates() -> list[Path]:
    out = []
    env = os.environ.get("T0135_RUN_DIR")
    if env:
        out.append(Path(env))
    out.append(REPO / "logs" / RUN_NAME)
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        out.append(Path(proc.stdout.strip()).parent / "logs" / RUN_NAME)
    return out


def run_dir() -> Path:
    tried = _candidates()
    for p in tried:
        if (p / "trials.jsonl").is_file() and (p / "sessions").is_dir():
            return p
    pytest.skip(
        f"T-0135 A2 eval: the T-0132 run directory is absent (tried {[str(p) for p in tried]})"
    )


def require_script() -> None:
    if not SCRIPT.exists():
        raise AssertionError("T-0135 A2: scripts/replay_thought_calls.py does not exist yet")


def replay(run: Path, *args: str) -> subprocess.CompletedProcess[str]:
    require_script()
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )


def replay_json(run: Path) -> dict[str, Any]:
    proc = replay(run, "--json")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr}"
    doc = json.loads(proc.stdout)
    assert isinstance(doc, dict)
    return doc


def key(f: dict[str, Any]) -> tuple[str, str]:
    return (Path(str(f["session"])).name, str(f["request_id"]))


def test_premise_trials_sha() -> None:
    """The directory is the T-0132 run the task's census was taken on."""
    run = run_dir()
    sha = hashlib.sha256((run / "trials.jsonl").read_bytes()).hexdigest()
    assert sha.startswith(TRIALS_SHA_PREFIX), sha


def test_a2_thresholds_on_t0132() -> None:
    """A2, exact: 860 scanned; 29 firings (S3 18, S1 7, S5 4); 28 valid + 1 repairable (``…-210810-…``
    ``t1-gen-s2``); 0 on the other 831; pass-2 count 10; the firing set equals the fixture's."""
    run = run_dir()
    doc = replay_json(run)
    assert doc["scanned"] == 860
    firings = doc["firings"]
    keys = [key(f) for f in firings]
    assert len(keys) == len(set(keys)) == 29, f"A2: {len(keys)} firings"
    assert Counter(f["task"] for f in firings) == {"S3": 18, "S1": 7, "S5": 4}
    assert Counter(f["status"] for f in firings) == {"valid": 28, "repairable": 1}
    assert [key(f) for f in firings if f["status"] == "repairable"] == [REPAIRABLE]
    assert doc["scanned"] - len(firings) == 831
    assert doc["pass2_same_shape"] == 10
    fixture = {
        (r["source"], r["request_id"])
        for r in (json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip())
        if r["class"] == "firing"
    }
    assert set(keys) == fixture


def test_every_firing_ends_a_failed_final_turn() -> None:
    """A2: every firing is the last ``generate``/``tool_step`` request of its session's turn, and that trial has
    ``success`` false and ``tool_loop_end`` ``final`` (read from the logs here, not from the script)."""
    run = run_dir()
    doc = replay_json(run)
    trials = {
        t["session_log"]: t
        for t in (
            json.loads(x) for x in (run / "trials.jsonl").read_text().splitlines() if x.strip()
        )
    }
    for f in doc["firings"]:
        session, request_id = key(f)
        t = trials[session]
        assert t["success"] is False, (session, request_id)
        assert t["tool_loop_end"] == "final", (session, request_id, t["tool_loop_end"])
        live = [
            json.loads(x)
            for x in (run / "sessions" / session).read_text().splitlines()
            if x.strip()
        ]
        reqs = [
            r["data"]["request_id"]
            for r in live
            if r["kind"] == "llm.request" and r["data"]["phase"] in ("generate", "tool_step")
        ]
        assert reqs and reqs[-1] == request_id, (session, request_id, reqs)


def test_text_mode_exits_0_on_t0132() -> None:
    """A2: the plain run (the command the CPU regime pastes) exits 0 and prints a line per firing."""
    run = run_dir()
    proc = replay(run)
    assert proc.returncode == 0, proc.stderr
    assert "20261022-210810-1975518" in proc.stdout and "repairable" in proc.stdout
