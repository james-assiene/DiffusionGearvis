"""T-0107e criterion 9: the CPU re-read of recorded GPU trials (read-only).

Run, once the predicate exists, with the directory that holds the launch worktrees::

    T0107E_TRIALS_ROOT=/home/user/demos/local_astra/.claude/worktrees \
        uv run pytest tests/eval/test_t0107e_reread.py -q -rA -s

**CPU only, not ``gpu``.** The rows spend no GPU time: they read ``trials.jsonl`` files and call ``user_declines``.
Without ``T0107E_TRIALS_ROOT`` the module skips at collection (``allow_module_level``), so it adds **no row** to a
path-only ``pytest tests/eval`` and T-0046's ``test_a_path_only_run_of_the_eval_directory_selects_nothing`` stays at 0
selected. (Setting the variable for a fast-suite run would make that T-0046 row count these rows; do not.)

Per trial, the user text is read from the case files, never from the trial or a report:

- T-0107b (``t0107b-launch/logs/t0107b-high-stakes-ab-8387447``): ``tests/eval/toolcall_q13/cases.jsonl`` by
  ``case_id``: S1 the case's ``user``; S2 the case's ``s2[script]`` (the script's text, ``No, leave it.`` /
  ``Yes, go ahead.``); S3 the case's ``s3["user"]``;
- the 5,890 earlier trials (``{t0098,t0098c,t0103,t0108a}-launch/logs/*/trials.jsonl``): ``kind`` ``tool`` /
  ``no_tool`` from ``tests/eval/toolcall/cases.jsonl`` (``user``), ``q12`` from
  ``tests/eval/surface_placement_cases.jsonl`` (``request``), ``plain`` from T-0093's ``PLAIN_QUESTIONS``, pinned
  here as strings (and checked equal to T-0098's copy).

"A trial with a tool commit" is T-0098's ``read_commits(...)["routed"]`` (a ``tool`` commit outside an open thought),
the reading T-0107b's ``s2_no_acted`` uses; "refuses" means ``user_declines(user text)`` is true for such a trial.
Every flagged trial is printed verbatim (run with ``-s``).

Expected counts, from the task (criterion 9) and the T-0107b write-up: S2-no 62/62 routed trials refused (A0 40,
H1 22); S1 (290 routed), S2-yes (160), S3 (58): 0; the earlier 5,890 trials: 0. File line counts are pinned (740;
1590 + 2100 + 880 + 1320 = 5890) so a partial copy cannot pass.

Regime: CPU re-read of recorded trials, n = every trial line, deterministic. The trials' own regimes are in
``docs/cd/toolcall-high-stakes-ab.md`` (T-0107b at 8387447: lean, vLLM v0.29.0, DiffusionGemma NVFP4 rev 2ea8372,
CL 256, 48 steps, thinking on, arms A0/H1, typed text, no Session) and T-0107a's Goal (the earlier runs).
"""

from __future__ import annotations

import importlib
import json
import os
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ENV = "T0107E_TRIALS_ROOT"
if not os.environ.get(ENV):
    pytest.skip(
        f"{ENV} is not set: the criterion-9 re-read reads recorded trials (CPU, read-only)",
        allow_module_level=True,
    )

from tests.eval.test_toolcall_prompt_ab import read_commits  # noqa: E402

pytestmark = [pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get(ENV, "."))
MODULE = "local_astra.toolcall.user_declines"

T0107B = "t0107b-launch/logs/t0107b-high-stakes-ab-8387447/trials.jsonl"
T0107B_LINES = 740
EARLIER = {
    "t0098-launch/logs/t0098-prompt-ab-09d1c50/trials.jsonl": 1590,
    "t0098c-launch/logs/t0098c-prompt-ab-3ac00a1/trials.jsonl": 2100,
    "t0103-launch/logs/t0103-c3-confirm-6720e56/trials.jsonl": 880,
    "t0108a-launch/logs/t0108a-ui-fields-ab-e6c3a81/trials.jsonl": 1320,
}
EARLIER_DIRS = ("t0098-launch", "t0098c-launch", "t0103-launch", "t0108a-launch")

#: T-0093 J10's five plain questions, pinned (T-0098 reuses them unchanged)
PLAIN = {
    "plain-01": "What is the capital of Japan?",
    "plain-02": "Tell me a fun fact about octopuses.",
    "plain-03": "How many minutes are there in a day?",
    "plain-04": "Who painted the Mona Lisa?",
    "plain-05": "How are you doing today?",
}


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(f"T-0107e owned files: {MODULE} does not exist: {exc}") from None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(c["id"]): c for c in _jsonl(path)}


def _trials(rel: str, lines: int) -> list[dict[str, Any]]:
    path = ROOT / rel
    assert path.is_file(), f"T-0107e criterion 9: {path} is missing under {ENV}={ROOT}"
    rows = _jsonl(path)
    assert len(rows) == lines, f"premise: {rel} holds {lines} trials; got {len(rows)}"
    return rows


def _show(flagged: list[tuple[str, str, str]], what: str) -> None:
    print(f"\n[T-0107e criterion 9] {what}: {len(flagged)} flagged trial(s)")
    for rid, text, tool in flagged:
        print(f"  {rid}\tuser={text!r}\ttool={tool!r}")


def test_t0107b_strata() -> None:
    """Criterion 9, first two items: on T-0107b's 740 trials, ``user_declines`` refuses all 62 routed S2-no trials
    (A0 40, H1 22) and no routed trial of S1, S2-yes or S3."""
    declines = _mod().user_declines
    cases = _by_id(REPO / "tests/eval/toolcall_q13/cases.jsonl")
    routed: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    flagged: dict[str, list[tuple[str, str, str]]] = {}
    for t in _trials(T0107B, T0107B_LINES):
        case = cases[str(t["case_id"])]
        stratum = str(t["stratum"])
        if stratum == "S1":
            text, key = str(case["user"]), "S1"
        elif stratum == "S2":
            text, key = str(case["s2"][t["script"]]), f"S2-{t['script']}"
        else:
            assert stratum == "S3", t["stratum"]
            text, key = str(case["s3"]["user"]), "S3"
        rc = read_commits(t["commits"])
        if not rc["routed"]:
            continue
        routed[key] += 1
        if declines(text):
            refused[key] += 1
            tool = next(c["text"] for c in t["commits"] if c.get("segment") == "tool")
            flagged.setdefault(key, []).append((str(t["request_id"]), text, str(tool)))
    for key in sorted(flagged):
        _show(flagged[key], f"T-0107b {key}")
    assert dict(routed) == {"S1": 290, "S2-yes": 160, "S2-no": 62, "S3": 58}, (
        f"premise: routed trials per stratum (docs/cd/toolcall-high-stakes-ab.md); got {dict(routed)}"
    )
    assert refused["S2-no"] == 62, (
        f"T-0107e criterion 9: S2-no refuses 62/62; got {refused['S2-no']}/62"
    )
    others = {k: v for k, v in refused.items() if k != "S2-no"}
    assert not others, f"T-0107e criterion 9: every other T-0107b stratum refuses 0; got {others}"


def _earlier_text(t: dict[str, Any], tools: dict[str, Any], q12: dict[str, Any]) -> str:
    kind, cid = str(t["kind"]), str(t["case_id"])
    if kind in ("tool", "no_tool"):
        return str(tools[cid]["user"])
    if kind == "q12":
        return str(q12[cid]["request"])
    assert kind == "plain", f"unknown trial kind {kind!r}"
    return PLAIN[cid]


def test_earlier_trials_refuse_none() -> None:
    """Criterion 9, third item: the 5,890 earlier trials (T-0098, T-0098c, T-0103, T-0108a launches) -- ``user_declines``
    refuses none of them (every routed trial's user text is not a clear no); the glob of the criterion finds exactly
    the four pinned files."""
    from tests.unit.test_t0098_prompt_ab import PLAIN_QUESTIONS

    assert dict(PLAIN_QUESTIONS) == PLAIN, "premise: the pinned plain questions are T-0093's"
    declines = _mod().user_declines
    found = sorted(
        str(p.relative_to(ROOT))
        for d in EARLIER_DIRS
        for p in (ROOT / d).glob("logs/*/trials.jsonl")
    )
    assert found == sorted(EARLIER), f"premise: the criterion's glob finds {found}"
    tools = _by_id(REPO / "tests/eval/toolcall/cases.jsonl")
    q12 = _by_id(REPO / "tests/eval/surface_placement_cases.jsonl")
    total = routed = 0
    flagged: list[tuple[str, str, str]] = []
    unrouted_flagged: list[tuple[str, str, str]] = []
    for rel, lines in EARLIER.items():
        for t in _trials(rel, lines):
            total += 1
            text = _earlier_text(t, tools, q12)
            is_routed = bool(read_commits(t["commits"])["routed"])
            routed += is_routed
            if declines(text):
                (flagged if is_routed else unrouted_flagged).append(
                    (str(t["request_id"]), text, "routed" if is_routed else "-")
                )
    _show(flagged, "earlier trials, routed")
    _show(unrouted_flagged, "earlier trials, not routed (nothing to refuse; reported only)")
    assert total == 5890, f"premise: 5,890 earlier trials; got {total}"
    print(f"[T-0107e criterion 9] earlier trials: {total}, routed {routed}")
    assert routed == 3570, (
        f"premise: 3,570 of the earlier trials are routed (docs/tasks/T-0107e.md, Evidence); got {routed}"
    )
    assert not flagged, f"T-0107e criterion 9: the earlier trials refuse 0; flagged {flagged[:20]}"
