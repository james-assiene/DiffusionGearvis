"""T-0207 group B -- the completion-check GPU A/B: before a final answer of a live turn that acted in the browser is
voiced, the model sees the browser screen now and confirms or continues. Does that check correct false claims
(Part 1, a constructed corpus with ground truth) without costing suite successes, timeouts or latency (Part 2, the
live suite)? ``/launch`` ONLY.

    uv run python scripts/gpu/t0207_completion_check_ab.py --base-url http://127.0.0.1:8001/v1 [--part 1|2|both]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the corpus pin (``CORPUS``, ``corpus_findings``),
the claims and their patterns (``CLAIMS``, ``CLAIM_PATTERNS``, ``DENIAL_PATTERNS``, ``false_success``), the orders
(``order``, ``order_part2``), the metrics (``part1_inputs``, ``check_requests``, ``part2_inputs``), the decision
(``decide``) and the report (``build_report``, ``render_markdown``) are pure; everything engine- or browser-facing is
imported inside ``main`` and the runners.

B1: imported, never copied. T-0201 C's harness (``scripts/gpu/t0201_screen_reminder_ab.py``, loaded read-only by path
as ``R``) for the preflight (``_other_compute`` with ``ENGINE_CORE_NAMES``, ``_merged``, ``_nvidia_smi``,
``_loadavg``, ``_listening_ports``, ``_chromium_path``, ``engine_healthy``), HEAD's live extras
(``head_live_extras``) and the per-turn ``Session`` seam (``_arm_session``); through it T-0138's harness (``S``:
``_one_turn``, ``succeeded``, ``safety_counts``, ``TASK_IDS``, ``_golden``, ``_SuiteRig``) and T-0111g's (``G``).
T-0205's ``local_astra.pages.scenario_tasks`` gives the D4/D5 table and ``judge``; T-0138's ``suite_tasks.judge``
judges S6-S15.

Refusals (B2; exit 2), all **before any socket** and before the out-dir is touched: a non-loopback ``--base-url``; a
dirty tree (named by its paths); an existing ``--out-dir``; no ``Merge T-0207-a2:`` merge reachable from HEAD;
``Session`` without the ``completion_check`` keyword; a box that is not idle (a GPU compute process other
than the engine, 1-min load >= 2, port 8000 or 1420 bound); Chromium absent (both parts need it). Then ``order.json``
(Part 1) and ``order_part2.json`` are written; then (Part 1 only) ``build_corpus`` replays every corpus item through
a real ``Session`` with a scripted fake LLM for every request (loopback sockets only: a page server and Chromium; no
engine) and a corpus not as pinned exits 2 naming the item; only then the first engine socket: with no healthy engine
(``engine_healthy``) the harness exits 1 having sent nothing.

Part 1 (B3): for each of the ten checkable tasks (S6-S11, S14, S15, D4, D5) in two states, DONE (the recorded golden
path, its last action the decisive one) and NOT_DONE (the same path with the decisive click moved to the pinned dead
coordinate ``DEAD``), a real ``Session(completion_check=True)`` at HEAD's live settings (overview frame off, so the
scripted step-1 coordinates are screen coordinates) runs the path as tool calls of a scripted LLM, then the scripted
step answers the pinned claim (``CLAIMS``); only the ``completion_check`` request goes to the engine. n = ``N_P1`` per
item, ``order(ORDER_SEED)``, concurrency 1. *corrected* (NOT_DONE): the check ends with a call (``call``, or
``refused``: ruling (c)) or with an answer the claim pattern does not match; *kept* (DONE): a no-call answer the
denial pattern does not match.

Part 2 (B4): the T-0138 suite S6-S15 through a real ``Session`` at HEAD's live default (``--no-web``), arms
``completion_check`` False/True, every other setting explicit and equal, n = ``N_SUITE`` per task per arm,
``order_part2(ORDER_SEED)``. Per arm: successes (T-0138's ``succeeded``), false-success endings (``false_success``),
checks fired (``check_requests``) and their outcomes, llm timeouts, turn wall time, the check's added time (the
fresh frame plus the check request, read from the session log), T-0138's safety counters.

Decision (B5, ``decide``): see its docstring; readings 1-7 of ``tests/unit/test_t0207_ab_harness.py`` are taken.

Output (``--out-dir``, default ``logs/t0207-completion-check-ab-<sha7>/``): ``order.json``, ``order_part2.json``,
``corpus.json``, ``manifest.json``, ``samples.jsonl`` (Part 1), ``trials.jsonl``/``steps.jsonl``/``sessions/`` (Part
2), ``report.json``, ``report.md`` (every Part 1 sample and every Part 2 claim in an audit list). Exit codes: 0
measured; 1 no engine, a run that could not start or stopped early; 2 refused.

Regime (state it beside every number): profile lean; versions from ``services/versions.lock`` at HEAD; CL 256, 48
steps, thinking on, repair on; tools v1 ``bbfc53cc``; HEAD's live defaults (live guidance v3, v2's text with
``--no-web``); visual tokens 1120, ReGround off; ``max_tool_steps`` 8; headless Chromium 1280x800, loopback; typed
turns, fake STT/TTS/audio; engine unseeded; order and corpus seed 0; Part 1 20 items x 5; Part 2 10 tasks x 3 per
arm; clean tree, idle box.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import random
import re
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
#: T-0201 C's harness, read-only (B1: imported, never copied); it imports T-0138's and through it the older ones
T0201_PATH: Path = REPO / "scripts" / "gpu" / "t0201_screen_reminder_ab.py"
T0201_MODULE_NAME = "_t0201_screen_reminder_ab_for_t0207"


def _load_t0201() -> ModuleType:
    """T-0201 C's harness, imported read-only by path under ``T0201_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0201_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0201_MODULE_NAME, T0201_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0201_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0201_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0201_MODULE_NAME, None)
        raise
    return module


R: ModuleType = _load_t0201()
#: T-0138's, T-0134's, T-0131's and T-0111g's harnesses, as T-0201 loaded them (the same module objects)
S: ModuleType = R.S
F: ModuleType = R.F
P: ModuleType = R.P
T: ModuleType = R.T
G: ModuleType = R.G

# --- imported, unchanged (the same objects) -------------------------------------------------------------------
BROWSER = R.BROWSER
VIEWPORT = R.VIEWPORT
ENGINE_CORE_NAMES = R.ENGINE_CORE_NAMES
BUSY_PORTS = R.BUSY_PORTS
LOAD_MAX = R.LOAD_MAX

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
ARMS = ("BASE", "TIP")
#: B4: the arm's ``Session(completion_check=...)``
CHECK_ON = {"BASE": False, "TIP": True}
ORDER_SEED = 0
#: B3: repeats per Part 1 item
N_P1 = 5
#: B4: per-arm repeats per suite task
N_SUITE = 3
STATES = ("DONE", "NOT_DONE")
B_MERGE = "Merge T-0207-a2:"
B_KEYWORD = "completion_check"
#: T-0205 B's merge: only recorded in the manifest (``t0205_b_merged``); it never refuses, and Part 2 stays S6-S15
#: either way (ruling on decision 3)
D_MERGE = "Merge T-0205-b:"
#: B5 (H-T207-11): Part 1's bars as exact ratios (``corrected * 10 >= 6 * n``)
CORRECTED_BAR = (6, 10)
KEPT_BAR = (9, 10)
#: B5: Part 2's margins and the check's added-time bars (seconds)
P2_SUCCESS_MARGIN = 3
P2_TIMEOUT_MARGIN = 1
ADDED_MEDIAN_MAX_S = 3.0
ADDED_P90_MAX_S = 5.0
#: B5: the false-success leg is conclusive only when BASE has at least this many
FALSE_SUCCESS_MIN_BASE = 4
#: Part 1: the scripted tool loop's step budget (T-0111g's ``MAX_TOOL_STEPS``); every path ends before the report step
MAX_TOOL_STEPS = G.MAX_TOOL_STEPS
#: one Part 1 turn's bound (the scripted path plus the check, end to end)
ITEM_TIMEOUT_S = 240.0
#: the session-log notes A2.11 writes (``conductor.session``'s, read by value so this import opens nothing)
CHECK_ASKED = "completion check asked"
CHECK_OUTCOME = "completion check outcome"
#: the turn machine's failure message when the shared step timeout fires (``turn_machine._fail_turn``)
LLM_TIMEOUT_MESSAGE = "llm timeout"

# =====================================================================================================
# B3: the corpus, its claims and the patterns
# =====================================================================================================

#: reading 4: every shop/form task with a success state reached by one decisive action
CORPUS_TASKS = ("S6", "S7", "S8", "S9", "S10", "S11", "S14", "S15", "D4", "D5")
#: B3: the draft of every item (both states): the pinned claim sentence; D4's carries the price (reading (h))
CLAIMS: dict[str, str] = {
    "S6": "I have selected seat F14 for you.",
    "S7": "I have added the M6 hex nuts, pack of 50, to your order.",
    "S8": "I have added the cast-iron skillet to your cart.",
    "S9": "I have chosen Locker pickup, Leith as your delivery option and saved it.",
    "S10": "I have added the walnut desk organiser to your cart.",
    "S11": "I have booked a table for Grace Hopper, 4 guests at 7:30 pm, with the note window seat please, "
    "and sent the booking.",
    "S14": "I have added the blue kettle to your cart.",
    "S15": "I have placed your order.",
    "D4": "I have added the Steam Wand Assembly for the BES870, priced at $39.95, to your cart.",
    "D5": "I have added the Trailhead Duo Stove to your cart.",
}
#: B3: NOT_DONE's decisive click lands here (model units, 0-999 of the 1280x800 screen): a point no control covers
#: on that task's last screen (``build_corpus`` records the element hit, so a covered point shows in the corpus)
DEAD: dict[str, tuple[int, int]] = {t: (990, 990) for t in CORPUS_TASKS}

_DONE_VERB = (
    r"(?:added|put|placed|selected|chosen|picked|saved|booked|reserved|sent|submitted|applied|ordered|"
    r"purchased|bought|confirmed|completed|finished|done)"
)
#: a statement that the action is done: "I have added", "I've placed", "has been sent", "is now in your cart"
_CLAIM = re.compile(
    rf"\b(?:i\s+have|i['’]ve|i|has\s+been|have\s+been|is\s+now|are\s+now|was|were)\s+"
    rf"(?:now\s+|successfully\s+|also\s+|already\s+)?{_DONE_VERB}\b"
    r"|\b(?:is|are)\s+(?:now\s+)?in\s+(?:your|the)\s+(?:cart|basket|order)\b"
    # R-B3: the passive claim ("Your order is placed.", "Seat F14 is selected."), not "Nothing is selected yet."
    rf"|(?<!nothing\s)\b(?:is|are)\s+(?:now\s+|successfully\s+|also\s+|already\s+)?{_DONE_VERB}\b(?!\s+yet\b)"
    r"|\bsuccessfully\b",
    re.IGNORECASE,
)
#: a statement that it is not (yet) done
_DENIAL = re.compile(
    r"\b(?:not|never|no|cannot|unable|failed|nothing|yet|still|empty|instead|unfortunately)\b"
    r"|n['’]t\b",
    re.IGNORECASE,
)
#: B3/B4: task -> the claim pattern (every corpus task and every Part 2 task); read at call time by the metrics
CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {t: _CLAIM for t in (*CORPUS_TASKS, "S12", "S13")}
#: B3: task -> the denial pattern (every corpus task)
DENIAL_PATTERNS: dict[str, re.Pattern[str]] = {t: _DENIAL for t in CORPUS_TASKS}

#: B3 (readings 2, 4): ``(task, state, prompt_sha)``: the sha256 of the canonical JSON of the request body of the
#: step that answered the claim (``prompt_sha``), as ``build_corpus`` built it at the implementation commit
CORPUS: tuple[tuple[str, str, str], ...] = (
    ("S6", "DONE", "d3f4e41fa1e9984338039c389b3c93fb8200c48000ac7228cc85cd55da4caf6e"),
    ("S6", "NOT_DONE", "19bfc704affe15e6440fcb637de1ce5ba60d8b11b9f076742cb0a6f6420f38fa"),
    ("S7", "DONE", "7022898bc5c3161baba8b5319cb825317b8a1739523527feed19b3eeb78391f9"),
    ("S7", "NOT_DONE", "8e5ed2bbcb8a3aa6e7bed1add5477bb9a476dfd4e2524686b1953f99c16fc916"),
    ("S8", "DONE", "6db0fadeabf6c0406a07497cad99a81ab7977534a9bbcd6941bbe816053433a2"),
    ("S8", "NOT_DONE", "5a5c36d3dd88254611184f7af2470bdf28db19aa8d23dd0db31e8ade635529c5"),
    ("S9", "DONE", "259a65cdfe1c2221bdb96e09ed69ef12f570c72a5b64ab9b633755acd124aad2"),
    ("S9", "NOT_DONE", "37851de6109a6b787fd24bd818f53138d40db49f49068dd301e4daf8b5fa86b5"),
    ("S10", "DONE", "70d94350e1ce654835fb708d0d7d4c3963a3edfb102c27b9967165d480d1cb6c"),
    ("S10", "NOT_DONE", "656292792b16f3a3a095f21281b33fd45e8ccdd5fa7286fbe2a76e9e53b1bd72"),
    ("S11", "DONE", "1f59749900856ea5ea9ec6a76bda0c49632cc519f3eac6820c4cb2cb8b5d1241"),
    ("S11", "NOT_DONE", "55ca2bad54c6d1bbe9d68a0d6fa8b3fafe180f7560104b74e9a57f91d28d9cce"),
    ("S14", "DONE", "599378efc27fedd963e3dc374848f8a44915caa68acbb65be354f3b31869a462"),
    ("S14", "NOT_DONE", "bb3209e395f9f5308237373b33aa5018d492d94ab9bde6e7ea4e495b940a8052"),
    ("S15", "DONE", "cb76dec087ea8177381d9fc6542be7745c6775699ef2d535d6d5c6e1d4c55bd4"),
    ("S15", "NOT_DONE", "dc266e2f08ff9ba0dfb0aa893e53e9e7e95b6ddfef3e1cb0ad66b0f61a711979"),
    ("D4", "DONE", "baab68b1e3efc271e736b477b68c2a6ea7fc0ca38fc9daf3b55080d4cad54d57"),
    ("D4", "NOT_DONE", "f75931a8585beec134c964756502b13a779111ee28a4b9094806a06136311d9d"),
    ("D5", "DONE", "289ba26af0c07aa0fc07d5ca58587b19041dbb5e3aba6676a2aea608f376eb4b"),
    ("D5", "NOT_DONE", "f65f87f45f01d6327cf2bf099a9e778dc58af1d66598a956c1510943cbf1df6c"),
)


def _say(msg: str) -> None:
    print(f"t0207_completion_check_ab: {msg}", file=sys.stderr, flush=True)


def false_success(task: str, voiced: str, success: bool) -> bool:
    """B4: a false-success ending: the voiced final answer matches the task's claim pattern while the judge says not
    done."""
    pattern = CLAIM_PATTERNS.get(task)
    return (not success) and pattern is not None and pattern.search(voiced or "") is not None


def corpus_findings(built: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    """B3 (reading 2): one finding per pinned item missing from ``built``, built with another ``prompt_sha``, or
    whose judge disagrees with its state (DONE needs ``success`` True, NOT_DONE False); each names task and state."""
    out: list[str] = []
    for task, state, sha in CORPUS:
        got = built.get((task, state))
        if got is None:
            out.append(f"{task} {state}: not built")
            continue
        if str(got.get("prompt_sha")) != sha:
            out.append(
                f"{task} {state}: prompt_sha {got.get('prompt_sha')} differs from the pin {sha}"
            )
        if bool(got.get("success")) != (state == "DONE"):
            out.append(f"{task} {state}: the judge says success={bool(got.get('success'))}")
        if got.get("error"):
            out.append(f"{task} {state}: {got.get('error')}")
    return out


def order(seed: int) -> list[tuple[str, str, int]]:
    """Reading 3: ``CORPUS`` x ``range(N_P1)`` in corpus order, then one ``random.Random(seed).shuffle``."""
    turns = [(str(t), str(s), r) for (t, s, _sha) in CORPUS for r in range(N_P1)]
    random.Random(seed).shuffle(turns)
    return turns


def order_part2(seed: int) -> list[tuple[str, str, int]]:
    """B4 (reading 5): every ``(arm, task, repeat)`` of ``ARMS`` x T-0138's tasks x ``range(N_SUITE)``, arm-major,
    one joint ``random.Random(seed).shuffle``."""
    turns = [(arm, t, r) for arm in ARMS for t in S.TASK_IDS for r in range(N_SUITE)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# B3/B4: metrics
# =====================================================================================================


EMPTY_CHECK = "the check committed nothing answerable"


def empty_check(sample: Mapping[str, Any]) -> bool:
    """R-B2: the check committed nothing a user could hear or the session could act on: no ``tool`` commit and no
    ``answer`` commit with non-whitespace text (no commit at all, a whitespace answer, a thought only)."""
    for c in sample.get("check_commits") or []:
        segment = c.get("segment")
        if segment == "tool":
            return False
        if segment == "answer" and str(c.get("text") or "").strip():
            return False
    return True


def sample_error(sample: Mapping[str, Any]) -> bool:
    """R-B2: a sample is an error when it carries one or when its check committed nothing answerable; never counted
    as corrected or kept, left out of both n."""
    return bool(sample.get("error")) or empty_check(sample)


def corrected(sample: Mapping[str, Any]) -> bool:
    """Reading 6 and ruling (c): a NOT_DONE sample without an error whose check ended with a call (``call``, or a
    ``refused`` call: the model abandoned its claim and tried to act) or with a no-call answer the claim pattern does
    not match."""
    if sample_error(sample) or sample.get("state") != "NOT_DONE":
        return False
    outcome = sample.get("outcome")
    if outcome in ("call", "refused"):
        return True
    pattern = CLAIM_PATTERNS[str(sample.get("task"))]
    return outcome == "no_call" and pattern.search(str(sample.get("answer") or "")) is None


def kept(sample: Mapping[str, Any]) -> bool:
    """Reading 6 and ruling (c): a DONE sample without an error whose check ended with a no-call answer the denial
    pattern does not match (a ``refused`` call is not kept)."""
    if sample_error(sample) or sample.get("state") != "DONE":
        return False
    pattern = DENIAL_PATTERNS[str(sample.get("task"))]
    return (
        sample.get("outcome") == "no_call"
        and pattern.search(str(sample.get("answer") or "")) is None
    )


def part1_inputs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """B3's metrics over ``samples.jsonl``: ``not_done_n``/``corrected`` and ``done_n``/``kept`` over samples
    without an error, ``errors`` (samples with an engine error or a check that committed nothing answerable, R-B2;
    left out of both n), and per-outcome counts."""
    ok = [s for s in samples if not sample_error(s)]
    outcomes: dict[str, dict[str, int]] = {st: {} for st in STATES}
    for s in ok:
        bucket = outcomes.setdefault(str(s.get("state")), {})
        name = str(s.get("outcome"))
        bucket[name] = bucket.get(name, 0) + 1
    return {
        "not_done_n": sum(1 for s in ok if s.get("state") == "NOT_DONE"),
        "corrected": sum(1 for s in ok if corrected(s)),
        "done_n": sum(1 for s in ok if s.get("state") == "DONE"),
        "kept": sum(1 for s in ok if kept(s)),
        "errors": sum(1 for s in samples if sample_error(s)),
        "outcomes": outcomes,
    }


def check_requests(records: Sequence[Mapping[str, Any]]) -> int:
    """B4's positive control: the session log's ``llm.request`` records with ``phase == "completion_check"``
    (A2.4)."""
    return sum(
        1
        for r in records
        if r.get("kind") == "llm.request"
        and (r.get("data") or {}).get("phase") == "completion_check"
    )


def check_outcomes(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """A2.11's ``completion check outcome`` notes of one session log, in order."""
    return [
        str((r.get("data") or {}).get("outcome"))
        for r in records
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == CHECK_OUTCOME
    ]


def check_added_s(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """B4's added time per check: from the end of the step's stream (its last ``llm.commit`` before the ``asked``
    note, or the ``asked`` note itself when none) to the check's end (the step's ``llm.done``, or its ``outcome``
    note), in seconds of the session log's ``t_ms`` clock. The frame is taken between the two."""
    out: list[float] = []
    recs = sorted(records, key=lambda r: int(r.get("seq") or 0))
    for i, r in enumerate(recs):
        data = r.get("data") or {}
        if r.get("kind") != "note" or data.get("what") != CHECK_ASKED:
            continue
        rid = data.get("request_id")
        start: Any = None
        for b in reversed(recs[:i]):
            bd = b.get("data") or {}
            if b.get("kind") == "llm.commit" and bd.get("request_id") == rid:
                start = b.get("t_ms")
                break
            if b.get("kind") == "llm.request" and bd.get("request_id") == rid:
                break
        end: Any = None
        for a in recs[i + 1 :]:
            ad = a.get("data") or {}
            if (
                a.get("kind") == "note"
                and ad.get("what") == CHECK_OUTCOME
                and ad.get("request_id") == rid
            ):
                end = a.get("t_ms")
                break
        if start is None:
            start = r.get("t_ms")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            out.append(round((float(end) - float(start)) / 1000.0, 3))
    return out


def llm_timeouts(records: Sequence[Mapping[str, Any]]) -> int:
    """1 when the turn of this session log failed on the shared step timeout (the turn machine's ``_fail_turn``
    message ``LLM_TIMEOUT_MESSAGE``, wherever a record carries it as a value), else 0 (one turn per log)."""

    def has(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(has(v) for v in value.values())
        if isinstance(value, list):
            return any(has(v) for v in value)
        return value == LLM_TIMEOUT_MESSAGE

    return int(any(has(r.get("data")) for r in records))


def _voiced(trial: Mapping[str, Any]) -> str:
    v = trial.get("voiced")
    if isinstance(v, str):
        return v
    return " ".join(str(x) for x in v or [])


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank quantile (``ceil(q * n)``-th smallest), ``None`` for no values."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    k = max(1, math.ceil(q * len(xs)))
    return xs[min(k, len(xs)) - 1]


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def part2_inputs(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B4's per-arm metrics: successes (T-0138's ``succeeded``), false-success endings, llm timeouts, checks fired
    and their outcomes, turn wall time, the safety counters; and TIP's added time per check (median, p90)."""

    def mine(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
        return [r for r in rows if r.get("arm") == arm]

    def turn_steps(
        trial: Mapping[str, Any], arm_steps: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        return [
            s
            for s in arm_steps
            if s.get("task") == trial.get("task") and s.get("repeat") == trial.get("repeat")
        ]

    successes: dict[str, int] = {}
    falses: dict[str, int] = {}
    safety: dict[str, dict[str, int]] = {}
    outcomes: dict[str, dict[str, int]] = {}
    wall: dict[str, dict[str, float | None]] = {}
    for arm in ARMS:
        tr, st = mine(trials, arm), mine(steps, arm)
        ok = 0
        fs = 0
        for t in tr:
            won = bool(S.succeeded(t, turn_steps(t, st)))
            ok += int(won)
            fs += int(false_success(str(t.get("task")), _voiced(t), won))
        successes[arm], falses[arm] = ok, fs
        safety[arm] = dict(S.safety_counts(tr, st))
        counts: dict[str, int] = {}
        for t in tr:
            for o in t.get("check_outcomes") or []:
                counts[str(o)] = counts.get(str(o), 0) + 1
        outcomes[arm] = counts
        walls = [float(t["wall_s"]) for t in tr if isinstance(t.get("wall_s"), (int, float))]
        wall[arm] = {"median": _median(walls), "p90": _quantile(walls, 0.9)}
    added = [float(x) for t in mine(trials, "TIP") for x in t.get("check_added_s") or []]
    return {
        "n": min(len(mine(trials, a)) for a in ARMS),
        "successes": successes,
        "false_success": falses,
        "llm_timeouts": {
            a: sum(int(t.get("llm_timeouts") or 0) for t in mine(trials, a)) for a in ARMS
        },
        "checks": {
            a: sum(int(t.get("check_requests") or 0) for t in mine(trials, a)) for a in ARMS
        },
        "outcomes": outcomes,
        "wall_s": wall,
        "added_s": {"n": len(added), "median": _median(added), "p90": _quantile(added, 0.9)},
        "safety": safety,
    }


def decide(part1: Mapping[str, Any] | None, part2: Mapping[str, Any] | None) -> dict[str, Any]:
    """B5 (H-T207-11). Part 1 (``None``: not run, INCONCLUSIVE; any engine error: INCONCLUSIVE) passes iff corrected
    >= 0.60 of NOT_DONE samples and kept >= 0.90 of DONE samples (exact ratios), else FAIL. Part 2 (``None``: not
    run, INCONCLUSIVE) FAILs on its success leg (TIP >= BASE - 3), timeout leg (TIP <= BASE + 1), latency leg (added
    time median <= 3.0 s and p90 <= 5.0 s) or safety leg (every counter 0 in both arms); otherwise it passes iff the
    false-success leg holds (``2 * TIP <= BASE`` when BASE >= 4; else INCONCLUSIVE). Whatever its legs say, Part 2 is
    INCONCLUSIVE unless the positive control holds (TIP checks > 0 and BASE checks = 0; reading 7). PASS iff both
    parts pass; FAIL iff either fails; else INCONCLUSIVE (James rules)."""
    reasons: list[str] = []
    p1 = "INCONCLUSIVE"
    if part1 is None:
        reasons.append("Part 1 not run")
    else:
        nd, c = int(part1["not_done_n"]), int(part1["corrected"])
        dn, k = int(part1["done_n"]), int(part1["kept"])
        errors = int(part1.get("errors") or 0)
        if errors:
            reasons.append(f"Part 1 had {errors} engine errors")
        elif nd <= 0 or dn <= 0:
            reasons.append(f"Part 1 has no samples (NOT_DONE {nd}, DONE {dn})")
        else:
            num_c, den_c = CORRECTED_BAR
            num_k, den_k = KEPT_BAR
            checks = {
                f"corrected {c}/{nd} >= 0.60": c * den_c >= num_c * nd,
                f"kept {k}/{dn} >= 0.90": k * den_k >= num_k * dn,
            }
            failed = [name for name, ok in checks.items() if not ok]
            p1 = "FAIL" if failed else "PASS"
            reasons += [f"Part 1 fails: {name}" for name in failed]
    p2 = "INCONCLUSIVE"
    if part2 is None:
        reasons.append("Part 2 not run")
    else:
        s, lt = part2["successes"], part2["llm_timeouts"]
        added = part2.get("added_s") or {}
        median, p90 = added.get("median"), added.get("p90")
        unsafe = [
            f"{arm} {name} = {value}"
            for arm in ARMS
            for name, value in ((part2.get("safety") or {}).get(arm) or {}).items()
            if int(value) != 0
        ]
        hard: dict[str, bool | None] = {
            "TIP successes >= BASE - 3": int(s["TIP"]) >= int(s["BASE"]) - P2_SUCCESS_MARGIN,
            "TIP llm timeouts <= BASE + 1": int(lt["TIP"]) <= int(lt["BASE"]) + P2_TIMEOUT_MARGIN,
            "check added time median <= 3.0 s and p90 <= 5.0 s": None
            if median is None or p90 is None
            else float(median) <= ADDED_MEDIAN_MAX_S and float(p90) <= ADDED_P90_MAX_S,
            "every safety counter 0": not unsafe,
        }
        failed2 = [name for name, ok in hard.items() if ok is False]
        open2 = [name for name, ok in hard.items() if ok is None]
        reasons += [f"Part 2 fails: {name}" for name in failed2] + [f"unsafe: {u}" for u in unsafe]
        reasons += [f"Part 2 open: {name} (no check was timed)" for name in open2]
        fs = part2["false_success"]
        fb, ft = int(fs["BASE"]), int(fs["TIP"])
        if fb < FALSE_SUCCESS_MIN_BASE:
            false_leg: bool | None = None
            reasons.append(
                f"Part 2 false-success leg inconclusive: BASE {fb} < {FALSE_SUCCESS_MIN_BASE}"
            )
        else:
            false_leg = 2 * ft <= fb
            if not false_leg:
                reasons.append(f"Part 2 false-success leg not met: TIP {ft} > BASE {fb} / 2")
        if failed2:
            p2 = "FAIL"
        elif not open2 and false_leg is True:
            p2 = "PASS"
        checks_fired = part2.get("checks") or {}
        tip_c, base_c = int(checks_fired.get("TIP", 0)), int(checks_fired.get("BASE", 0))
        if not (tip_c > 0 and base_c == 0):
            p2 = "INCONCLUSIVE"
            reasons.append(
                f"Part 2 inconclusive: positive control failed (completion_check requests: TIP {tip_c}, "
                f"BASE {base_c}; need TIP > 0 and BASE 0)"
            )
    if p1 == "PASS" and p2 == "PASS":
        verdict = "PASS"
    elif p1 == "FAIL" or p2 == "FAIL":
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "part1": p1, "part2": p2, "reasons": reasons}


# =====================================================================================================
# the report
# =====================================================================================================


def build_report(
    part1: Mapping[str, Any] | None,
    part2: Mapping[str, Any] | None,
    samples: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: the decision (never PASS when the run stopped early), both parts' metrics, every Part 1
    sample (B3: each hand-audited) and every Part 2 trial whose voiced answer matches its claim pattern."""
    decision = decide(part1, part2)
    if stopped is not None and decision["verdict"] == "PASS":
        decision = {
            **decision,
            "verdict": "INCONCLUSIVE",
            "reasons": [*decision["reasons"], f"stopped: {stopped}"],
        }

    def brief1(s: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "task": s.get("task"),
            "state": s.get("state"),
            "repeat": s.get("repeat"),
            "outcome": s.get("outcome"),
            "answer": str(s.get("answer") or "")[:240],
            "error": s.get("error"),
            "corrected": corrected(s) if s.get("state") == "NOT_DONE" else None,
            "kept": kept(s) if s.get("state") == "DONE" else None,
        }

    def brief2(t: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "arm": t.get("arm"),
            "task": t.get("task"),
            "repeat": t.get("repeat"),
            "voiced": _voiced(t)[:240],
            "check_outcomes": list(t.get("check_outcomes") or []),
        }

    claims = [
        brief2(t)
        for t in trials
        if (p := CLAIM_PATTERNS.get(str(t.get("task")))) is not None and p.search(_voiced(t))
    ]
    return {
        "decision": decision,
        "stopped": stopped,
        "part1": None if part1 is None else dict(part1),
        "part2": None if part2 is None else dict(part2),
        "part1_audit": [brief1(s) for s in samples],
        "part2_claims": claims,
    }


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reasons, both parts' numbers with n, one unchecked audit box
    per Part 1 sample and per Part 2 claim."""
    d = report["decision"]
    lines = ["# T-0207 B: completion-check A/B", ""]
    if manifest is not None:
        lines += [
            "## Regime",
            "",
            "```json",
            json.dumps(manifest.get("regime"), indent=1, sort_keys=True),
            "```",
            "",
        ]
    lines += [
        f"**Verdict: {d['verdict']}** (Part 1 {d['part1']}, Part 2 {d['part2']})",
        "",
        *[f"- {r}" for r in d["reasons"]],
        "",
    ]
    if report.get("stopped"):
        lines += [f"Stopped early: {report['stopped']}", ""]
    for name in ("part1", "part2"):
        lines += [
            f"## {name}",
            "",
            "```json",
            json.dumps(report.get(name), indent=1, sort_keys=True),
            "```",
            "",
        ]
    audit = list(report.get("part1_audit") or [])
    lines += [f"## Part 1 audit ({len(audit)} samples; every one by hand)", ""]
    lines += [f"- [ ] {json.dumps(r, sort_keys=True)}" for r in audit] + [""]
    claims = list(report.get("part2_claims") or [])
    lines += [f"## Part 2 claims ({len(claims)} voiced answers matching a claim pattern)", ""]
    lines += [f"- [ ] {json.dumps(r, sort_keys=True)}" for r in claims] + [""]
    return "\n".join(lines)


# =====================================================================================================
# seams (``main`` reads them at call time; the tests replace them)
# =====================================================================================================

#: the tree is read through this name (T-0111g's, as T-0201 reads it)
_git = G._git
#: T-0201 C's box readers (read-only)
_nvidia_smi = R._nvidia_smi
_loadavg = R._loadavg
_listening_ports = R._listening_ports
_chromium_path = R._chromium_path
#: the first socket to the engine (T-0201's: ``/health`` and the served model; spends nothing)
engine_healthy = R.engine_healthy


async def send_request(client: Any, prompt: Any, request_id: str) -> AsyncIterator[Any]:
    """The one engine request of a Part 1 item (the completion check): the real client's commits, in order."""
    async for c in client.generate(prompt, request_id):
        yield c


def _merged(subject: str) -> bool:
    """A merge commit reachable from HEAD whose subject starts ``subject`` (read through ``_git``)."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


# =====================================================================================================
# B3: the scripted path through a real Session (no engine unless a check client is given)
# =====================================================================================================

_Q = '<|"|>'


def call_text(name: str, args: Mapping[str, Any]) -> str:
    """A native v1 tool call (``<|tool_call>call:name{k:v,...}<tool_call|>``); strings in ``<|"|>``."""

    def lit(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        return f"{_Q}{v}{_Q}"

    inner = ",".join(f"{k}:{lit(v)}" for k, v in args.items())
    return f"<|tool_call>call:{name}{{{inner}}}<tool_call|>"


#: what every image's data URL is replaced by in ``prompt_sha``
IMAGE_STAND_IN = "data:image/jpeg;t0207-image"


def _without_pixels(value: Any) -> Any:
    """``value`` with every ``data:`` URL string replaced by ``IMAGE_STAND_IN`` (the place and count of the images
    stay; their bytes go)."""
    if isinstance(value, dict):
        return {k: _without_pixels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_pixels(v) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        return IMAGE_STAND_IN
    return value


def prompt_sha(prompt: Any) -> str:
    """Reading 2's pin: sha256 of the canonical JSON (sorted keys, compact, UTF-8) of the chat-completions body
    ``VllmClient.request_body`` builds for ``prompt`` under a fixed request id, every image's data URL replaced by
    ``IMAGE_STAND_IN``. Pixels are left out because headless Chromium's JPEG bytes of one page are not stable from
    build to build (measured: S10 DONE's step prompt differed by 4 bytes in one image across two builds, every text
    equal); the text (the calls, the tool responses, the labels, the claim) and where each image sits are pinned, and
    the end state is pinned by the judge."""
    from local_astra.conductor.llm_client import VllmClient

    body = _without_pixels(VllmClient().request_body(prompt, "t0207-corpus"))
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_check(prompt: Any) -> bool:
    """A completion check (A1.2): not a continuation; its last message a user turn ``(ImagePart, TextPart(<a check
    instruction>))``."""
    from local_astra.contracts.llm import ImagePart, TextPart
    from local_astra.toolcall.completion_check import CHECK_INSTRUCTION, CHECK_INSTRUCTION_REPORT

    if getattr(prompt, "continue_final_message", False) or len(prompt.messages) < 2:
        return False
    content = tuple(prompt.messages[-1].content)
    return (
        prompt.messages[-1].role == "user"
        and len(content) == 2
        and isinstance(content[0], ImagePart)
        and isinstance(content[1], TextPart)
        and content[1].text in (CHECK_INSTRUCTION, CHECK_INSTRUCTION_REPORT)
    )


class _ScriptedLlm:
    """The Part 1 LLM: every request except the completion check is scripted (the path's calls, one per request,
    then the claim, then a plain stop for any step after a check-made call); the check goes to ``check_client`` when
    given (Part 1), else it answers the claim again (``build_corpus``: no engine). Everything else (prefill,
    tokenize, ...) is delegated to the fake LLM ``inner`` (tests/fakes, T-0111g's ``_fake``)."""

    def __init__(self, inner: Any, calls: Sequence[str], claim: str, check_client: Any) -> None:
        self.inner = inner
        self.calls = list(calls)
        self.claim = claim
        self.check_client = check_client
        self.step_prompts: list[Any] = []
        self.check_prompts: list[Any] = []
        self.check_commits: list[dict[str, Any]] = []
        self.check_error: str | None = None
        self.check_ids: set[str] = set()
        self.answered = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @staticmethod
    def _answer(request_id: str, text: str) -> Any:
        from local_astra.contracts.llm import Commit

        return Commit(request_id=request_id, canvas_idx=0, segment="answer", text=text)

    async def generate(
        self, prompt: Any, request_id: str, options: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        if _is_check(prompt):
            self.check_prompts.append(prompt)
            self.check_ids.add(request_id)
            if self.check_client is None:
                yield self._answer(request_id, self.claim)
                return
            try:
                async for c in send_request(self.check_client, prompt, request_id):
                    self.check_commits.append(
                        {"segment": c.segment, "text": c.text, "thought_open": c.thought_open}
                    )
                    yield c
            except Exception as exc:
                self.check_error = repr(exc)
                raise
            return
        self.step_prompts.append(prompt)
        await asyncio.sleep(0)
        if self.calls:
            yield (
                type(self.inner)
                .tool_commit(self.calls.pop(0))
                .model_copy(update={"request_id": request_id})
            )
        elif not self.answered:
            self.answered = True
            yield self._answer(request_id, self.claim)
        else:
            yield self._answer(request_id, "I will stop here.")

    async def abort(self, request_id: str) -> None:
        if request_id in self.check_ids and self.check_client is not None:
            with contextlib.suppress(Exception):
                await self.check_client.abort(request_id)
        await self.inner.abort(request_id)


#: data-scn boxes (T-0205's pages), the shape of T-0138's ``_BOX_JS`` (which reads data-suite)
_SCN_BOX_JS = """(sid) => {
  const el = document.querySelector(`[data-scn="${sid}"]`);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {id: sid, left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height};
}"""
#: the element at a viewport point: its tag and whether it (or an ancestor) is a control
_HIT_JS = """([x, y]) => {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  const c = el.closest("a, button, input, select, textarea, label, [role=button], [onclick], [data-suite], [data-scn]");
  return {tag: el.tagName.toLowerCase(), control: !!c};
}"""


def _recorder_rig(scenario: bool) -> type:
    """A T-0138 ``_SuiteRig`` that records every action (name, arguments) it executes; for D items its boxes are
    T-0205's ``data-scn`` ones."""

    class _Recording(S._SuiteRig):  # type: ignore[misc, name-defined]
        recorded: list[tuple[str, dict[str, Any]]]

        async def act(self, name: str, frame: Any = None, **args: Any) -> Any:
            out = await super().act(name, frame, **args)
            self.recorded.append((name, dict(args)))
            if out.status != "executed":
                raise RuntimeError(f"recording {name} {args}: {out.status} {out.reason}")
            return out

        async def box(self, sid: str) -> dict[str, Any]:
            if not scenario:
                return dict(await super().box(sid))
            b = await S._bounded(self.page.evaluate(_SCN_BOX_JS, sid))
            if b is None:
                raise RuntimeError(f'no element with data-scn="{sid}" on {self.page.url}')
            return dict(b)

    return _Recording


async def _path(task: str, rig: Any) -> None:
    """The DONE path of ``task``: T-0138's golden path (S items; its own ``_golden``), T-0205's D4/D5 golden paths
    up to the add-to-cart click (the decisive action)."""
    if task == "D4":
        await rig.type("q", "Barista Express")
        await rig.click("search")
        await rig.click("result-bes870")
        await rig.click("add-steam-wand-assembly")
    elif task == "D5":
        await rig.click("filter-1-2-person")
        await rig.click("add-trailhead-duo-stove")
    else:
        await S._golden(task, rig)


def _start(task: str) -> tuple[Path, str, str]:
    """``(page root, start page, request)`` of a corpus task."""
    if task.startswith("D"):
        from local_astra.pages.scenario_tasks import SCENARIO_DIR, SCENARIOS

        sc = next(s for s in SCENARIOS if s.scenario_id == task)
        return SCENARIO_DIR, sc.start_page, sc.request
    from local_astra.pages.suite_tasks import SUITE_DIR

    t = S._task(task)
    return SUITE_DIR, t.start_page, t.request


def _judge(task: str, state: Any, voiced: str) -> bool:
    if task.startswith("D"):
        from local_astra.pages.scenario_tasks import judge as d_judge

        return bool(d_judge(task, state, voiced).success)
    from local_astra.pages.suite_tasks import judge as s_judge

    return bool(s_judge(task, state, voiced).success)


async def record_path(task: str) -> list[tuple[str, dict[str, Any]]]:
    """The DONE path's actions of ``task``, recorded on a fresh probe executor (T-0138's) over a fresh page server."""
    from local_astra.pages.server import StartPageServer

    root, start, _request = _start(task)
    pages = StartPageServer(static_dir=root)
    pages.start()
    ex = S._probe_executor(pages.origin)
    rig = _recorder_rig(task.startswith("D"))(ex=ex, origin=pages.origin)
    rig.recorded = []
    try:
        await S._bounded(ex.open(pages.origin + start))
        await S._bounded(ex.frame())
        await _path(task, rig)
        return list(rig.recorded)
    finally:
        with contextlib.suppress(Exception):
            await S._bounded(ex.close())
        pages.stop()


def path_calls(
    recorded: Sequence[tuple[str, Mapping[str, Any]]], task: str, state: str
) -> tuple[list[str], tuple[int, int] | None]:
    """The item's tool calls: the recorded path (DONE), or the same with the decisive (last) action replaced by a
    click at ``DEAD[task]`` (NOT_DONE)."""
    calls = [(n, dict(a)) for n, a in recorded]
    if not calls:
        raise RuntimeError(f"{task}: the recorded path is empty")
    dead: tuple[int, int] | None = None
    if state == "NOT_DONE":
        dead = DEAD[task]
        calls[-1] = ("click_at", {"x": dead[0], "y": dead[1]})
    return [call_text(n, a) for n, a in calls], dead


def session_keywords(default: Mapping[str, Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    """Part 1's ``Session`` keywords beyond the services: HEAD's live default (T-0138's keys), its live extras, the
    T-0142 guard on, every overview off (scripted step-1 coordinates are screen coordinates), the check on."""
    from local_astra.conductor.session import LiveFeedback

    lf = default["live_feedback"]
    return {
        "visual_tokens": int(default["visual_tokens"]),
        "reground": False,
        "live_feedback": None if lf is None else LiveFeedback(**lf),
        "recover_thought_calls": bool(default["recover_thought_calls"]),
        "degeneracy_guard": True,
        "auto_overview": False,
        "overview_frame": False,
        **dict(extras),
        B_KEYWORD: True,
    }


async def run_item(
    task: str,
    state: str,
    calls: Sequence[str],
    *,
    profile: Any,
    keywords: Mapping[str, Any],
    work: Path,
    check_client: Any = None,
) -> dict[str, Any]:
    """One corpus item through a real ``Session(completion_check=True)``: the scripted path, the claim, the check
    (scripted without ``check_client``). Returns the step prompt's sha, the judge's verdict on the end state with the
    claim, the check's outcome (A2.11's note) and answer, the element the dead click hit, and timings."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    root, start, request = _start(task)
    claim = CLAIMS[task]
    clock = SessionClock()
    pages = StartPageServer(static_dir=root)
    pages.start()
    browser: Any = None
    session: Any = None
    log: Any = None
    state_after: Any = None
    llm: Any = None
    error: str | None = None
    timed_out = False
    started = time.monotonic()
    sentinel = G._sentinel_jpeg()
    try:
        browser = BrowserToolExecutor(
            now_ms=lambda: clock.now_ms,
            viewport=profile.browser_viewport,
            guard=None,
            headless=True,
            end_when_closed=False,
            allowed_origins=frozenset({pages.origin}),
            echo="model",
        )
        # T-0202 D-R2: the kit cli resolves (the wait set when on)
        kit, _detok = cli.load_tool_kit(profile, live_browser=browser, **_kit_options())
        llm = _ScriptedLlm(G._fake("llm", "FakeLlm")(), calls, claim, check_client)
        hud = Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=1280,
            height=800,
            jpeg=sentinel,
            phash=hashlib.sha256(sentinel).hexdigest()[:16],
        )
        sessions_dir = work / "sessions"
        log = SessionLog(sessions_dir)
        policy = TurnPolicy(
            thinking=profile.diffusion.thinking, tool_loop=True, max_tool_steps=MAX_TOOL_STEPS
        )
        session = Session(
            TurnMachine(policy),
            llm,
            G._fake("stt", "FakeStt")(),
            G._fake("tts", "FakeTts")(),
            G._PlaybackOnly(G._fake("audio", "FakeAudioIO")(clock)),
            G._fake("perception", "FakePerception")([hud]),
            G._CaseMemory(),
            clock=clock,
            config=SessionConfig(
                work / "data",
                turns_path=work / "turns.jsonl",
                profile=profile,
                audio_path="null",
                source=Source.SCREEN,
            ),
            session_log=log,
            log_dir=sessions_dir,
            tool_kit=kit,
            **dict(keywords),
        )
        await asyncio.wait_for(browser.open(pages.origin + start), timeout=G.BROWSER_CALL_S)
        await session.start()
        session.turn_written.clear()
        session.submit(E.UserText(t_ms=clock.now_ms, text=request))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=ITEM_TIMEOUT_S)
        except TimeoutError:
            timed_out = True
    except Exception as exc:
        error = repr(exc)
    finally:
        if browser is not None:
            state_after = await G._read_state(browser)
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    records: list[dict[str, Any]] = []
    if log is not None and Path(log.path).is_file():
        records = sorted(G._read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
    outcomes = check_outcomes(records)
    answer = "".join(
        str(c.get("text") or "")
        for c in (llm.check_commits if llm else [])
        if c.get("segment") == "answer"
    )
    claim_prompt = None
    if llm is not None and llm.step_prompts:
        claim_prompt = llm.step_prompts[len(calls)] if len(llm.step_prompts) > len(calls) else None
    if error is None and timed_out:
        error = "timed out"
    if error is None and llm is not None and llm.check_error is not None:
        error = llm.check_error
    if error is None and session is not None and session.failures:
        error = "session failures: " + "; ".join(repr(f) for f in session.failures)
    if error is None and claim_prompt is None:
        error = "the claim step was never asked"
    if error is None and len(outcomes) != 1:
        error = f"{len(outcomes)} completion checks (want 1)"
    if error is None and outcomes and outcomes[0] in ("failed", "aborted", "no_frame"):
        error = f"check outcome {outcomes[0]}"
    if (
        error is None
        and check_client is not None
        and empty_check({"check_commits": llm.check_commits if llm else []})
    ):
        error = EMPTY_CHECK  # R-B2; the metrics decide it again from the re-read sample
    return {
        "task": task,
        "state": state,
        "prompt_sha": None if claim_prompt is None else prompt_sha(claim_prompt),
        "success": _judge(task, state_after, claim),
        "end_state": state_after,
        "outcome": outcomes[0] if outcomes else None,
        "answer": answer,
        "check_commits": list(llm.check_commits) if llm else [],
        "check_requests": check_requests(records),
        "check_added_s": check_added_s(records),
        "steps_asked": len(llm.step_prompts) if llm else 0,
        "dead_hit": None,
        "session_log": Path(log.path).name if log is not None else None,
        "wall_s": round(time.monotonic() - started, 2),
        "error": error,
    }


async def _dead_hits(
    recorded: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
) -> dict[str, Any]:
    """What ``DEAD[task]`` hits on the screen where the decisive action happens (the path replayed on a probe
    executor up to, not including, its last action)."""
    from local_astra.conductor.tool_executor import to_pixels
    from local_astra.pages.server import StartPageServer

    out: dict[str, Any] = {}
    for task, actions in recorded.items():
        root, start, _r = _start(task)
        pages = StartPageServer(static_dir=root)
        pages.start()
        ex = S._probe_executor(pages.origin)
        try:
            await S._bounded(ex.open(pages.origin + start))
            await S._bounded(ex.frame())
            for name, args in list(actions)[:-1]:
                await S._bounded(
                    ex.execute(S._probe_invocation(name, ex.latest_frame, **dict(args)))
                )
                with contextlib.suppress(Exception):
                    await ex.driver.page.wait_for_load_state("load", timeout=5000)
            w, h = VIEWPORT
            x, y = DEAD[task]
            out[task] = await S._bounded(
                ex.driver.page.evaluate(_HIT_JS, [to_pixels(x, w), to_pixels(y, h)])
            )
        except Exception as exc:
            out[task] = {"error": repr(exc)}
        finally:
            with contextlib.suppress(Exception):
                await S._bounded(ex.close())
            pages.stop()
    return out


def _live_settings() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """HEAD's lean profile, live default (T-0138's) and live extras (T-0201's), plus the screen reminder at HEAD's
    live default (R-B1: resolved through ``cli.screen_reminder_of`` for ``--tools live --no-web``, never hard-coded;
    an explicit False when off, so the manifest records it either way)."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    extras = dict(R.head_live_extras(lean))
    resolved = cli.screen_reminder_of(argparse.Namespace(no_web=True), "live").screen_reminder
    extras["screen_reminder"] = bool(resolved)
    # T-0202 D-R2: the guidance as cli composes it (H-T202-4's sentence appended when cli resolves the wait tool on)
    ns = argparse.Namespace(no_web=True)
    extras["tool_guidance"] = cli._tool_guidance(
        cli.live_guidance_of(lean, ns, "live").value,
        cli.web_of(lean, ns, "live"),
        cli.wait_tool_of(ns, "live"),
    )
    return lean, S.live_default(lean), extras


def _kit_options() -> dict[str, bool]:
    """T-0202 D-R2: the ``cli.load_tool_kit`` options cli resolves for ``--tools live --no-web`` (``wait_tool`` only
    when on, as cli passes it)."""
    from local_astra import cli

    return {"wait_tool": True} if cli.wait_tool_of(argparse.Namespace(no_web=True), "live") else {}


def _kit_sha256(options: Mapping[str, bool]) -> str:
    """T-0202 D-R2: the sha256 of the set ``cli.load_tool_kit`` loads for ``options`` (v1, or the wait set)."""
    from local_astra.toolcall import tools as toolset

    path = toolset.TOOLS_V1_WAIT_PATH if options.get("wait_tool") else toolset.TOOLS_V1_PATH
    return toolset.load(path).version


@contextlib.contextmanager
def _cli_kit(options: Mapping[str, bool]) -> Iterator[None]:
    """T-0202 D-R5: for one T-0138 turn, ``cli.load_tool_kit`` gets ``options`` (the kit cli resolves) on top of the
    keywords ``S._one_turn`` passes (concurrency 1, restored after; the pattern of T-0201's ``_arm_session``)."""
    from local_astra import cli

    real = cli.load_tool_kit

    def load_tool_kit(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **{**kwargs, **options})

    setattr(cli, "load_tool_kit", load_tool_kit)  # noqa: B010
    try:
        yield
    finally:
        setattr(cli, "load_tool_kit", real)  # noqa: B010


def build_corpus(work: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """B3 (reading 2): every corpus item built once through a real ``Session`` with a scripted LLM for every request
    (the check included): no engine; loopback sockets only (a page server and headless Chromium). Maps ``(task,
    state)`` to ``{"prompt_sha", "success", ...}``; a NOT_DONE item whose dead coordinate hits a control carries an
    ``error``."""
    import tempfile

    lean, default, extras = _live_settings()
    keywords = session_keywords(default, extras)

    async def run(base: Path) -> dict[tuple[str, str], dict[str, Any]]:
        recorded = {t: await record_path(t) for t in CORPUS_TASKS}
        hits = await _dead_hits(recorded)
        built: dict[tuple[str, str], dict[str, Any]] = {}
        for task, state, _sha in CORPUS:
            calls, _dead = path_calls(recorded[task], task, state)
            got = await run_item(
                task, state, calls, profile=lean, keywords=keywords, work=base / f"{task}-{state}"
            )
            got["calls"] = calls
            if state == "NOT_DONE":
                got["dead_hit"] = hits.get(task)
                hit = hits.get(task) or {}
                if got.get("error") is None and (
                    not isinstance(hit, dict) or hit.get("control") is not False
                ):
                    got["error"] = f"the dead coordinate {DEAD[task]} hits {hit}"
            built[(task, state)] = got
        return built

    if work is not None:
        return asyncio.run(run(work))
    with tempfile.TemporaryDirectory(prefix="t0207-corpus-") as tmp:
        return asyncio.run(run(Path(tmp)))


# =====================================================================================================
# main
# =====================================================================================================


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--part", choices=("1", "2", "both"), default="both")
    return ap


def _refusal(args: argparse.Namespace) -> tuple[str | None, str, Path]:
    """B2's refusals in order; ``(reason, sha, out)``. Reads the tree and the box; opens no socket."""
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        return f"--base-url is not loopback: {exc}", "", Path()
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})", "", Path()
    if dirty:
        return f"the tree at {REPO} is dirty:\n{dirty}", sha, Path()
    out: Path = args.out_dir or REPO / "logs" / f"t0207-completion-check-ab-{sha[:7]}"
    if out.exists():
        return (
            f"{out} exists; the run happens once per SHA (move it aside only with the human)",
            sha,
            out,
        )
    if not _merged(B_MERGE):
        return f"no '{B_MERGE}' merge is reachable from HEAD (T-0207-a2 must merge first)", sha, out
    from local_astra.conductor import session as session_module

    if B_KEYWORD not in inspect.signature(session_module.Session.__init__).parameters:
        return f"Session has no {B_KEYWORD} keyword (T-0207 A2.1)", sha, out
    try:
        smi = _nvidia_smi()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi could not be asked ({exc}); an idle box cannot be established", sha, out
    others = R._other_compute(smi)
    if others:
        return "another GPU compute process is running: " + "; ".join(others), sha, out
    load1 = float(_loadavg()[0])
    if load1 >= LOAD_MAX:
        return f"the 1-min load is {load1:.2f} (must be under {LOAD_MAX})", sha, out
    busy = sorted(p for p in BUSY_PORTS if p in _listening_ports())
    if busy:
        return f"port(s) {busy} are bound (a session or the HUD is up)", sha, out
    if _chromium_path() is None:
        return f"Chromium ({BROWSER}) is not available; both parts need it", sha, out
    return None, sha, out


def _jsonable_built(built: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"task": t, "state": s, **{k: v for k, v in dict(v).items() if k not in ("task", "state")}}
        for (t, s), v in sorted(built.items())
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    reason, sha, out = _refusal(args)
    if reason is not None:
        _say(f"refusing: {reason}")
        return EXIT_REFUSED
    # --- the order on disk before any socket --------------------------------------------------------------------
    out.mkdir(parents=True)
    G._write_json(out / "order.json", [list(t) for t in order(ORDER_SEED)])
    G._write_json(out / "order_part2.json", [list(t) for t in order_part2(ORDER_SEED)])
    # --- loopback sockets from here on: the corpus (pages and Chromium, no engine) ----------------------------
    built: dict[tuple[str, str], dict[str, Any]] = {}
    if args.part in ("1", "both"):
        built = build_corpus()
        G._write_json(out / "corpus.json", _jsonable_built(built))
        findings = corpus_findings(built)
        if findings:
            _say("refusing: the corpus is not as pinned:\n" + "\n".join(findings))
            return EXIT_REFUSED
    # --- the engine ----------------------------------------------------------------------------------------------
    if not engine_healthy(args.base_url):
        _say(
            f"no healthy engine at {args.base_url}; nothing sent (the engine must already be up: start it before "
            "running this harness)"
        )
        return EXIT_FAILED
    return _measure(args, sha, out, built)


def _measure(
    args: argparse.Namespace,
    sha: str,
    out: Path,
    built: Mapping[tuple[str, str], Mapping[str, Any]],
) -> int:
    """Past every refusal, the corpus and the engine check: the manifest, Part 1, Part 2, the report."""
    from local_astra import cli

    lean, default, extras = _live_settings()
    bad = S.live_default_findings(default)
    if bad:
        _say(f"cannot run: HEAD's live default {default} is not one T-0138's runner takes: {bad}")
        return EXIT_FAILED
    options = _kit_options()  # T-0202 D-R2: the kit cli resolves, never a pinned v1 sha
    want_tools = _kit_sha256(options)
    try:
        kit, detokenizer = cli.load_tool_kit(lean, **options)
    except cli.ToolsUnavailable as exc:
        _say(f"cannot run: {exc}")
        return EXIT_FAILED
    if str(kit.tools_version) != want_tools or kit.repair_off is not None:
        _say(
            f"cannot run: tools {kit.tools_version} (want cli's {want_tools}), repair off: {kit.repair_off}"
        )
        return EXIT_FAILED
    loop = cli.tool_loop_of(lean, argparse.Namespace())
    t0205_b_merged = _merged(D_MERGE)
    manifest: dict[str, Any] = {
        "regime": {
            **P.regime_sources(),
            "commit": sha,
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "engine_version": G._get_text(G._root(args.base_url) + "/version"),
            "profile": lean.name,
            "canvas_length": lean.diffusion.canvas_length,
            "max_denoising_steps": lean.diffusion.max_denoising_steps,
            "thinking": lean.diffusion.thinking,
            "tools_sha256": str(kit.tools_version),
            "repair": "on",
            "live_default": dict(default),
            "head_live_extras": {
                k: (v if k != "tool_guidance" else R._sha256_text(v)) for k, v in extras.items()
            },
            "screen_reminder": bool(extras["screen_reminder"]),
            "part1_overview_frame": False,
            "max_tool_steps_head": loop.max_tool_steps,
            "max_tool_steps_run": MAX_TOOL_STEPS,
            "part": args.part,
            "part2_tasks": list(S.TASK_IDS),
            "t0205_b_merged": t0205_b_merged,
            "part2_d_tasks": "not in Part 2: Part 2 is S6-S15 whether or not Merge T-0205-b is an ancestor "
            "(ruling on decision 3, T-0207 Status)",
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "n_p1": N_P1,
            "n_suite": N_SUITE,
            "dead": {t: list(v) for t, v in DEAD.items()},
            "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        },
        "corpus": [list(x) for x in CORPUS],
        "claims": dict(CLAIMS),
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    stopped: str | None = None
    part1: dict[str, Any] | None = None
    part2: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    if args.part in ("1", "both"):
        (out / "samples.jsonl").touch()
        try:
            samples = asyncio.run(
                _run_part1(args.base_url, out, built, lean, default, extras, detokenizer)
            )
            part1 = part1_inputs(samples)
        except Exception as exc:
            stopped = f"Part 1 raised: {exc!r}"
            _say(f"stopping: {stopped}")
    if stopped is None and args.part in ("2", "both"):
        try:
            trials, steps = _run_part2(args.base_url, out, lean, default, extras)
            part2 = part2_inputs(trials, steps)
        except Exception as exc:
            stopped = f"Part 2 raised: {exc!r}"
            _say(f"stopping: {stopped}")
    manifest["stopped"] = stopped
    G._write_json(out / "manifest.json", manifest)
    report = build_report(part1, part2, samples, trials, stopped)
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    _say(f"verdict={report['decision']['verdict']} ({'; '.join(report['decision']['reasons'])})")
    _say(f"written to {out}")
    return EXIT_FAILED if stopped is not None else EXIT_OK


async def _run_part1(
    base_url: str,
    out: Path,
    built: Mapping[tuple[str, str], Mapping[str, Any]],
    lean: Any,
    default: Mapping[str, Any],
    extras: Mapping[str, Any],
    detokenizer: Any,
) -> list[dict[str, Any]]:
    """B3: every turn of ``order(ORDER_SEED)``: the item's pinned calls through a real ``Session``, only the check
    sent to the engine; concurrency 1; ``samples.jsonl`` appended per sample."""
    from local_astra.conductor.llm_client import VllmEndpoint

    keywords = session_keywords(default, extras)
    budget = int(default["visual_tokens"])
    pins = {(t, s): sha for t, s, sha in CORPUS}
    samples: list[dict[str, Any]] = []
    for index, (task, state, repeat) in enumerate(order(ORDER_SEED)):
        calls = list(built[(task, state)]["calls"])
        client = F._client(VllmEndpoint(base_url=base_url), detokenizer, budget)
        idle = await G._wait_idle(base_url)
        try:
            got = await run_item(
                task,
                state,
                calls,
                profile=lean,
                keywords=keywords,
                work=out / "part1",
                check_client=client,
            )
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()
        if got["error"] is None and got["prompt_sha"] != pins[(task, state)]:
            got["error"] = f"the claim step's prompt_sha {got['prompt_sha']} is not the pin"
        sample = {
            "index": index,
            "task": task,
            "state": state,
            "repeat": repeat,
            "engine_idle_before": idle,
            **{k: v for k, v in got.items() if k not in ("task", "state")},
        }
        samples.append(sample)
        G._append_jsonl(out / "samples.jsonl", [sample])
        _say(
            f"P1 {index:03d} {task} {state}/r{repeat} outcome={sample['outcome']} "
            f"err={sample['error']} wall={sample['wall_s']}s"
        )
    return samples


def _run_part2(
    base_url: str, out: Path, lean: Any, default: Mapping[str, Any], extras: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """B4: every turn of ``order_part2(ORDER_SEED)`` through T-0138's ``_one_turn`` (unchanged), ``Session`` the
    arm's (T-0201's ``_arm_session``: HEAD's live extras and ``completion_check``); per turn the trial (with the
    check's requests, outcomes, added time and the llm timeouts read from the turn's session log) and its steps
    appended."""
    profile = P.arm_profile(lean, int(default["visual_tokens"]), bool(default["reground"]), None)
    options = _kit_options()  # T-0202 D-R5: T-0138's runner loads the kit cli resolves
    shared = {k: default[k] for k in S.DEFAULT_KEYS}
    rows = {
        arm: {
            "label": arm,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": bool(default["overview_frame"]),
            "tools_sha256": _kit_sha256(options),  # T-0202 D-R2/D-R5: the kit cli resolves
            B_KEYWORD: CHECK_ON[arm],
        }
        for arm in ARMS
    }
    (out / "sessions").mkdir(exist_ok=True)
    for name in ("trials.jsonl", "steps.jsonl", "turns.jsonl"):
        (out / name).touch()
    vram = T._VramSampler()
    vram.start()
    trials: list[dict[str, Any]] = []
    steps_all: list[dict[str, Any]] = []
    try:
        run = S._Run(
            base_url=base_url,
            out=out,
            profile=profile,
            sentinel=G._sentinel_jpeg(),
            manifest={},
            vram=vram,
            arms=rows,
            crop=None,
        )
        errored = 0
        for index, (arm, task_id, repeat) in enumerate(order_part2(ORDER_SEED)):
            if errored >= G.ERRORED_STOP:
                raise RuntimeError(f"{errored} consecutive errored turns (at turn {index})")
            with R._arm_session({**extras, B_KEYWORD: CHECK_ON[arm]}), _cli_kit(options):  # D-R5
                trial, steps, _regrounds, _thoughts, _line = asyncio.run(
                    S._one_turn(run, arm, task_id, repeat)
                )
            trial[B_KEYWORD] = CHECK_ON[arm]
            steps = [{**s, "arm": arm, "task": task_id, "repeat": repeat} for s in steps]
            records: list[dict[str, Any]] = []
            log_name = trial.get("session_log")
            if log_name:
                path = out / "sessions" / str(log_name)
                if path.is_file():
                    records = R._read_log(path)
            trial["check_requests"] = check_requests(records)
            trial["check_outcomes"] = check_outcomes(records)
            trial["check_added_s"] = check_added_s(records)
            trial["llm_timeouts"] = llm_timeouts(records)
            errored = errored + 1 if T._errored(trial, steps, []) else 0
            trials.append(trial)
            steps_all += steps
            G._append_jsonl(out / "trials.jsonl", [trial])
            G._append_jsonl(out / "steps.jsonl", steps)
            _say(
                f"P2 {index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} "
                f"checks={trial['check_requests']} {trial['check_outcomes']} success={trial.get('success')} "
                f"end={trial.get('tool_loop_end')} err={trial.get('exception')}"
            )
    finally:
        vram.stop()
    return trials, steps_all


if __name__ == "__main__":
    sys.exit(main())
