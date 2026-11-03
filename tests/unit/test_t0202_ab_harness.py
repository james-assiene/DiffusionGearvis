"""T-0202 group C: the wait-tool GPU A/B harness's pure half and its refusals (unit; CPU; no engine, no browser, no
socket, except the one ``browser``-gated row at the end). docs/tasks/T-0202.md C1-C5, "Tests to write" (C).

BASE a4ba3cae (main, T-0202 A and B merged: ``--wait-tool on|off``, the executor's ``wait`` with its limits). Rows
that need the harness (``scripts/gpu/t0202_wait_ab.py``) are red on a named assertion in ``harness()``/``api()`` ("does
not exist yet" / "has no ..."), never on an import error: every harness symbol is looked up inside the row. Every
expected value is computed by hand (comments say how), never read off an implementation (none exists).

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not).

Imported, never copied (C1)
    T-0138's suite runner ``scripts/gpu/t0138_suite_ab.py`` as the module attribute ``S`` (loaded by path, as T-0201
    C's harness does), and through it T-0111g's ``G``. The harness defines none of T-0138's turn runner or scoring
    functions (``RUNNER_NAMES`` below).

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``ARMS = ("OFF", "ON")``; ``WAIT_TOOL = {"OFF": False, "ON":
    True}``; ``ORDER_SEED = 0``; ``N_PROBE = 5``; ``N_SUITE = 3``; ``PROBE_IDS = ("W1", "W2")``; ``SUITE_IDS`` =
    S6..S15 (T-0138's ``TASK_IDS``); ``PROBE_DIR`` (the real path of ``local_astra/pages/wait_probe/``);
    ``PROBE_TASKS``: ``PROBE_IDS`` -> a row with ``start_page`` and ``request`` (attribute or key); ``TOLD_TO_WAIT``:
    the compiled C4 regex.

Pure functions
    ``told_to_wait(text) -> bool``; ``probe_ok(task_id, voiced) -> bool`` (the pure oracles); ``order(seed) ->
    list[(arm, task_id, repeat)]`` (reading 2); ``arm_metrics(trials, steps) -> {arm: metrics}`` (C4; keys in
    ``test_arm_metrics_hand_built``); ``decide(metrics) -> {"verdict", "reasons"}`` (C5, over ``arm_metrics``'s shape).

Seams ``main`` reads at call time (T-0201 C's)
    ``_git(*args) -> str``; ``_nvidia_smi() -> str``; ``_loadavg() -> (1, 5, 15 min)``; ``_listening_ports() ->
    set[int]``; ``_chromium_path() -> str | None``; ``engine_healthy(base_url) -> bool`` (the first socket after the
    refusals). ``arm_turn(arm, extras)``: a context manager under which, for one T-0138 turn, ``session.Session`` applies
    ``extras`` (HEAD's live settings T-0138's runner predates) and the arm's guidance over the runner's keywords, and
    ``cli.load_tool_kit`` builds the arm's kit (``wait_tool=WAIT_TOOL[arm]``); both restored after. ``main(argv)``
    takes ``--base-url`` and ``--out-dir``.

The probe content (the test-author's choice, pinned so the oracles can be hand-checked; the implementer writes the
pages to it):
    W1 (``search.html``, Amazon-shaped): the target is the "Brightwell" kettle at **$34.99**; the results also show
    distractor prices **$29.99**, **$41.50** and **$134.99**. Nothing priced is on screen before the results paint.
    W2 (``list.html``): a departures board; the 10:42 to Hollins Cross leaves from **Platform 7**; the other rows'
    platforms are **3**, **12** and **17**.

**Readings** (reported to the orchestrator)
    1. Told-to-wait is case-insensitive (voiced text starts sentences with capitals: "Refresh the page." must count),
       counted once per turn over the turn's voiced text, over every turn of the arm (probe and suite).
    2. The order is arm-major over ``ARMS``, then ``PROBE_IDS`` x ``range(N_PROBE)`` then ``SUITE_IDS`` x
       ``range(N_SUITE)``, then one ``random.Random(seed).shuffle``: 2 x (2 x 5 + 10 x 3) = 80. ``order.json`` is that
       list as JSON, on disk before ``engine_healthy`` is called.
    3. A probe success is ``probe_ok`` over the turn's voiced text (T-0138's ``_voiced_text``), never on a forced failure
       (T-0138's ``forced_failure``: engine error or a ``degenerate`` end); a ``success`` field on a probe trial is not
       read. A suite success is T-0138's ``succeeded``.
    4. "Wait calls" are steps whose ``call_name`` is ``wait`` (any status); ``waits_executed`` those with status
       ``executed``; a refusal is a ``wait`` step with status ``refused`` and ``reason`` ``WAIT_RUN_REASON``/
       ``WAIT_TURN_REASON``. "Suite turns with a wait" have at least one wait call.
    5. Median and p90 turn wall time (``wall_s``) are T-0111g's nearest rank (``G._percentile``; the median of 4 values
       is the 2nd smallest, not the mean of the middle two).
    6. Decision precedence (RULED by the coordinator 2026-10-30, replacing the proposed reading): a non-zero safety
       counter in either arm -> FAIL, before every INCONCLUSIVE; else OFF probe successes >= 8 -> INCONCLUSIVE; else
       the positive control (ON executed >= 1 wait, OFF executed 0) failing -> INCONCLUSIVE; else PASS iff every C5
       bar holds, else FAIL. "1.10 x OFF" is compared
       exactly in decimal: ON = 3.08 s against OFF = 2.8 s passes (``3.08 * 10 <= 2.8 * 11`` is False in floats).
    7. The probe reading (orchestrator's question): at HEAD the after-frame of the action that starts the slow content
       does **not** wait for it. T-0203's follow ends at ``domcontentloaded`` + two rAFs (``runtime._follow``); T-0204
       C's visual settle ends at the first matching pair of shots (0.1 s apart on a quiet page) and never starts a shot
       past ``STABLE_CAP_S`` = 1.0 s from its first. Content painted 1.5 s after Enter (W1) or after DOMContentLoaded
       (W2), with nothing changing in between, is absent from that frame; the gated browser row pins it on the real
       pages. Only one consequence is open: one ``wait`` (``WAIT_S`` = 1 s) after a fast action may still land before
       1.5 s, so the row allows up to ``MAX_CONSECUTIVE_WAITS`` waits for the content to show.

Regime: CPU, pure, hand-built records; no engine, no browser (but the gated row); n = 1.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import itertools
import json
import os
import random
import re
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0202_wait_ab.py"
T0138_HARNESS = REPO / "scripts" / "gpu" / "t0138_suite_ab.py"
PROBE_DIR = REPO / "local_astra" / "pages" / "wait_probe"
SUITE_IDS = ("S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15")
#: C4's regex, typed from docs/tasks/T-0202.md
TOLD_PATTERN = r"please wait|wait for (?:the )?page|refresh (?:the )?(?:page|screen)"
#: H-T202-4 and H-T202-5, typed from the task (never imported)
SENTENCE = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)
WAIT_RUN_REASON = "you have already waited three times in a row; act on the screen or answer"
WAIT_TURN_REASON = "you have already waited six times in this request; act on the screen or answer"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
#: T-0138's runner and scoring functions; the harness imports them through ``S`` and defines none of them (C1)
RUNNER_NAMES = (
    "_one_turn",
    "turn_records",
    "safety_counts",
    "succeeded",
    "forced_failure",
    "engine_error",
    "false_claim",
    "url_leaks",
    "body_line",
    "judge",
)


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0202 C1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0202_wait_ab_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0202 C: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


def _field(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, dict) else getattr(row, name)


# =====================================================================================================
# C1: import, reuse, constants
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """C1: importing the harness opens no socket (connect, bind or name lookup, loopback included); it has a
    ``main``."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"


def test_imports_t0138_runner_never_copies() -> None:
    """C1 ("imports T-0138's suite runner and never copies it"): ``S`` is a module loaded from T-0138's file, its turn
    runner is T-0138's function, and no function in the harness's source is named after T-0138's runner or scoring
    functions (a copy would have to define them)."""
    s = api("S")
    assert Path(getattr(s, "__file__", "")).resolve() == T0138_HARNESS.resolve(), s
    src = Path(inspect.getsourcefile(s._one_turn) or "").resolve()
    assert src == T0138_HARNESS.resolve(), f"C1: the turn runner comes from {src}"
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    defined = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    copied = sorted(defined & set(RUNNER_NAMES))
    assert not copied, f"C1: the harness defines T-0138's {copied} (a copy)"


def test_constants() -> None:
    """C2/C3: exits, arms and their flag, seed, n per probe and per suite task, the task ids."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED")) == (0, 1, 2)
    assert tuple(api("ARMS")) == ("OFF", "ON")
    assert dict(api("WAIT_TOOL")) == {"OFF": False, "ON": True}
    assert api("ORDER_SEED") == 0
    assert (api("N_PROBE"), api("N_SUITE")) == (5, 3), (
        "C3: n = 5 per probe task, 3 per suite task, per arm"
    )
    assert tuple(api("PROBE_IDS")) == ("W1", "W2")
    assert tuple(api("SUITE_IDS")) == SUITE_IDS


# =====================================================================================================
# C3: the order
# =====================================================================================================


def _reading2(seed: int) -> list[tuple[str, str, int]]:
    """Reading 2's construction, written out from the reading (the spec, not the harness's code)."""
    turns = [
        (arm, task, r)
        for arm in ("OFF", "ON")
        for tasks, n in ((("W1", "W2"), 5), (SUITE_IDS, 3))
        for task in tasks
        for r in range(n)
    ]
    random.Random(seed).shuffle(turns)
    return turns


def test_order_deterministic_seed0() -> None:
    """C3: 80 turns; each (arm, probe task) 5 times, each (arm, suite task) 3 times; deterministic for seed 0, another
    seed differs; the arms interleave (a per-arm block run would switch once); reading 2's construction."""
    order = api("order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)]
    assert turns != [tuple(t) for t in order(1)]
    assert len(turns) == 2 * (2 * 5 + 10 * 3) == 80
    per = Counter((t[0], t[1]) for t in turns)
    for arm in ("OFF", "ON"):
        for w in ("W1", "W2"):
            assert per[(arm, w)] == 5, (arm, w, per[(arm, w)])
        for s in SUITE_IDS:
            assert per[(arm, s)] == 3, (arm, s, per[(arm, s)])
    switches = sum(1 for x, y in itertools.pairwise(turns) if x[0] != y[0])
    assert switches > 20, f"the arms run in blocks ({switches} switches)"
    assert turns == _reading2(0)


# =====================================================================================================
# C4: told to wait
# =====================================================================================================

#: C4's rows, verbatim, then the test-author's (reading 1: case-insensitive; inside a sentence)
TOLD_MATCH = (
    "Please wait for the page to load",
    "refresh the screen",
    "Refresh the page.",
    "PLEASE WAIT a moment.",
    "You may need to wait for page loading to finish.",
    "Try to refresh page and look again.",
    "The results are still loading; please wait.",
)
TOLD_REFUSE = (
    "I will wait",
    "the wait time is 5 minutes",
    "waiting list",
    "The Brightwell kettle costs $34.99.",
    "I waited for the results and the price is $34.99.",
    "Refreshing the screenshot did not help.",
    "",
)


def test_told_to_wait_pattern_pinned() -> None:
    """C4: the harness's regex is the task's pattern, case-insensitive (reading 1)."""
    rx = api("TOLD_TO_WAIT")
    assert isinstance(rx, re.Pattern), type(rx)
    assert rx.pattern == TOLD_PATTERN, rx.pattern
    assert rx.flags & re.IGNORECASE, "reading 1: told-to-wait is case-insensitive"


@pytest.mark.parametrize("text", TOLD_MATCH)
def test_told_to_wait_matches(text: str) -> None:
    assert api("told_to_wait")(text) is True, text


@pytest.mark.parametrize("text", TOLD_REFUSE)
def test_told_to_wait_refuses(text: str) -> None:
    """C4 (reviewer mutation: a bare ``wait`` or ``refresh`` matches "I will wait", "waiting list", "Refreshing")."""
    assert api("told_to_wait")(text) is False, text


# =====================================================================================================
# C3: the probe oracles (pure, on the voiced answer)
# =====================================================================================================

PROBE_ROWS = [
    ("W1", "The Brightwell kettle costs $34.99.", True),
    ("W1", "It is 34.99 dollars.", True),
    ("W1", "The price is thirty-four ninety-nine.", True),
    ("W1", "It costs thirty four dollars and ninety-nine cents.", True),
    ("W1", "It costs 34 dollars and 99 cents.", True),
    ("W1", "THE PRICE IS $34.99", True),
    # the distractors on the same results page; $134.99 holds "34.99" as a substring
    ("W1", "The Brightwell kettle is $134.99.", False),
    ("W1", "It costs $29.99.", False),
    ("W1", "It costs $41.50.", False),
    ("W1", "It is 34.995 in the table.", False),
    ("W1", "It costs $34.9.", False),
    ("W1", "Please wait for the page to load, then refresh the page.", False),
    ("W1", "", False),
    ("W2", "The 10:42 to Hollins Cross leaves from Platform 7.", True),
    ("W2", "It leaves from platform seven.", True),
    ("W2", "Platform number 7.", True),
    ("W2", "It leaves from platform 17.", False),
    ("W2", "It leaves from platform 12.", False),
    ("W2", "It leaves from platform 3.", False),
    ("W2", "Please wait for the page to load.", False),
    ("W2", "", False),
    # reviewer F2: a mixed answer (the target beside a distractor from the same page, or beside a told-to-wait
    # phrase) is never a success, in either probe and in either order
    ("W1", "The Brightwell is $34.99; the Hartley is $29.99", False),
    ("W1", "The Hartley is $29.99 and the Brightwell is $34.99.", False),
    ("W1", "It is $34.99 or maybe $41.50.", False),
    ("W1", "$34.99, please wait for the page", False),
    ("W1", "Refresh the page; it showed $34.99.", False),
    ("W2", "Platform 7, not platform 3", False),
    ("W2", "Platform 12 or platform 7.", False),
    ("W2", "Platform seven, or platform seventeen.", False),
    ("W2", "Platform 7, please wait for the page", False),
]


@pytest.mark.parametrize(
    "task, voiced, want", PROBE_ROWS, ids=[f"{r[0]}-{i}" for i, r in enumerate(PROBE_ROWS)]
)
def test_probe_oracle(task: str, voiced: str, want: bool) -> None:
    """C3: each probe has a pure oracle on the voiced answer: W1 the target price (digits or words), W2 the target
    platform; a distractor on the same page and a told-to-wait answer are never a success."""
    assert api("probe_ok")(task, voiced) is want, (task, voiced)


def test_probe_pages_and_table() -> None:
    """C3: the two probe pages and their MANIFEST live under ``local_astra/pages/wait_probe/``, the harness serves
    that root (``PROBE_DIR``), each task's start page is a file there, and the pinned targets are on the pages."""
    assert PROBE_DIR.is_dir(), f"T-0202 C3: {PROBE_DIR.relative_to(REPO)} does not exist yet"
    for name in ("search.html", "list.html", "MANIFEST.md"):
        assert (PROBE_DIR / name).is_file(), f"T-0202 C3: wait_probe/{name} missing"
    assert Path(api("PROBE_DIR")).resolve() == PROBE_DIR.resolve()
    tasks = api("PROBE_TASKS")
    assert sorted(tasks) == ["W1", "W2"], sorted(tasks)
    for tid in ("W1", "W2"):
        page = str(_field(tasks[tid], "start_page"))
        assert page.startswith("/") and (PROBE_DIR / page.lstrip("/").split("?")[0]).is_file(), (
            tid,
            page,
        )
        assert str(_field(tasks[tid], "request")).strip(), tid
    search = (PROBE_DIR / "search.html").read_text(encoding="utf-8")
    board = (PROBE_DIR / "list.html").read_text(encoding="utf-8")
    for price in ("34.99", "29.99", "41.50", "134.99"):
        assert price in search, f"W1: {price} is not on search.html"
    assert "Brightwell" in search
    assert re.search(r"Platform\s*7\b", board) and "Hollins Cross" in board, "W2: the target row"


def test_probe_pages_never_served_by_a_session() -> None:
    """C3 (negative; holds at BASE and must keep holding): only the harness serves the probe pages; the default page
    root is not the probe root and no module under ``local_astra/`` names ``wait_probe``."""
    from local_astra.pages import server

    assert PROBE_DIR.resolve() != Path(server.STATIC_DIR).resolve()
    assert not str(PROBE_DIR.resolve()).startswith(str(Path(server.STATIC_DIR).resolve()) + os.sep)
    named = sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "local_astra").rglob("*.py")
        if "wait_probe" in p.read_text(encoding="utf-8")
    )
    assert named == [], f"C3: {named} reach the probe pages (only scripts/gpu/t0202_wait_ab.py may)"


# =====================================================================================================
# C4: metrics, per arm
# =====================================================================================================


def _trial(arm: str, task: str, repeat: int, voiced: list[str], **over: Any) -> dict[str, Any]:
    return {"arm": arm, "task": task, "repeat": repeat, "voiced": voiced, **over}


def _step(
    arm: str,
    task: str,
    repeat: int,
    k: int,
    name: str | None,
    status: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": task,
        "repeat": repeat,
        "step": k,
        "http_status": 200,
        "call_name": name,
        "status": status,
        "reason": reason,
    }


def _waits(arm: str, task: str, repeat: int, n: int, start: int = 1) -> list[dict[str, Any]]:
    return [_step(arm, task, repeat, start + i, "wait", "executed") for i in range(n)]


def _dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hand-built: OFF 7 turns (4 probe, 3 suite), ON 7 turns (3 probe, 4 suite). Expected values in
    ``test_arm_metrics_hand_built``, derived by hand in its comments."""
    trials = [
        # OFF probes: W1 r0 success; W1 r1 told to wait (twice in one turn: counts once); W2 r0 the 17 distractor;
        # W2 r1 success in words. A stored ``success`` is the opposite of the oracle (reading 3: never read).
        _trial("OFF", "W1", 0, ["The Brightwell kettle costs $34.99."], success=False),
        _trial(
            "OFF",
            "W1",
            1,
            ["Please wait for the page to load, then refresh the page."],
            success=True,
        ),
        _trial("OFF", "W2", 0, ["It leaves from platform 17."], success=True),
        _trial("OFF", "W2", 1, ["The 10:42 leaves from platform seven."], success=False),
        # OFF suite: 2 of 3 successes; S7 told the user to refresh
        _trial("OFF", "S6", 0, ["Done."], success=True, wall_s=10.0),
        _trial("OFF", "S7", 0, ["Refresh the screen and try again."], success=False, wall_s=30.0),
        _trial("OFF", "S8", 0, ["Added."], success=True, wall_s=20.0),
        # ON probes: W1 r0 success after a wait; W1 r1 says the price but ended degenerate (forced failure);
        # W2 r0 success after a click and a wait
        _trial("ON", "W1", 0, ["$34.99"]),
        _trial("ON", "W1", 1, ["It is $34.99."], tool_loop_end="degenerate"),
        _trial("ON", "W2", 0, ["Platform 7."]),
        # ON suite: 3 of 4 successes; S9 leaked a URL path (a T-0138 safety counter)
        _trial("ON", "S6", 0, ["Done."], success=True, wall_s=11.0),
        _trial("ON", "S7", 0, ["Done."], success=True, wall_s=33.0),
        _trial("ON", "S8", 0, ["Not found."], success=False, wall_s=22.0),
        _trial("ON", "S9", 0, ["Done."], success=True, wall_s=40.0, url_leaks=["/store/kettles"]),
    ]
    steps = [
        _step("OFF", "W1", 0, 1, "type_at", "executed"),
        _step("OFF", "W1", 0, 2, None, None),
        _step("OFF", "S6", 0, 1, "click_at", "executed"),
        _step("ON", "W1", 0, 1, "type_at", "executed"),
        *_waits("ON", "W1", 0, 1, start=2),
        _step("ON", "W1", 0, 3, None, None),
        _step("ON", "W1", 1, 1, "type_at", "executed"),
        *_waits("ON", "W1", 1, 3, start=2),
        _step("ON", "W1", 1, 5, "wait", "refused", WAIT_RUN_REASON),
        _step("ON", "W2", 0, 1, "click_at", "executed"),
        *_waits("ON", "W2", 0, 1, start=2),
        _step("ON", "S6", 0, 1, "wait", "refused", WAIT_TURN_REASON),
        _step("ON", "S7", 0, 1, "click_at", "executed"),
        *_waits("ON", "S8", 0, 1),
        _step("ON", "S9", 0, 1, "click_at", "executed"),
    ]
    return trials, steps


def test_arm_metrics_hand_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: probe successes (oracle; never on a forced failure; the stored field ignored), told-to-wait turns, wait
    calls (total, per turn, executed), refusals by reason, suite successes and turns with a wait, nearest-rank median
    and p90 suite wall time, and T-0138's safety counters per arm. T-0138's ``succeeded`` is a stub over suite trials'
    ``success`` (the suite judge is T-0138's, not under test) that refuses a probe trial."""
    h = harness()

    def succeeded(trial: dict[str, Any], steps: Any) -> bool:
        assert str(trial["task"]).startswith("S"), (
            f"a probe trial reached T-0138's suite judge: {trial}"
        )
        return bool(trial.get("success"))

    monkeypatch.setattr(h.S, "succeeded", succeeded)
    trials, steps = _dataset()
    got = api("arm_metrics")(trials, steps)
    off, on = got["OFF"], got["ON"]
    # OFF: W1 r0 and W2 r1 succeed (2 of 4); told: W1 r1 and S7 (2 turns); no wait call
    assert (off["probe_n"], off["probe_successes"]) == (4, 2), off
    assert off["told_to_wait"] == 2, off
    assert (off["wait_calls"], off["waits_executed"]) == (0, 0), off
    assert off["wait_calls_per_turn"] == 0
    assert dict(off["wait_refusals"]) == {"run": 0, "turn": 0}, off
    # OFF suite: S6, S8 succeed (2 of 3); walls [10, 20, 30]: rank ceil(1.5) = 2 -> 20.0, rank ceil(2.7) = 3 -> 30.0
    assert (off["suite_n"], off["suite_successes"], off["suite_turns_with_wait"]) == (3, 2, 0), off
    assert (off["suite_wall_median_s"], off["suite_wall_p90_s"]) == (20.0, 30.0), off
    # ON: W1 r0 and W2 r0 succeed; W1 r1 is degenerate (2 of 3); nobody told to wait
    assert (on["probe_n"], on["probe_successes"]) == (3, 2), on
    assert on["told_to_wait"] == 0, on
    # wait calls: W1 r0 1, W1 r1 3 + 1 refused, W2 r0 1, S6 1 refused, S8 1 = 8; executed 1 + 3 + 1 + 1 = 6
    assert (on["wait_calls"], on["waits_executed"]) == (8, 6), on
    assert on["wait_calls_per_turn"] == pytest.approx(8 / 7), on
    assert dict(on["wait_refusals"]) == {"run": 1, "turn": 1}, on
    # ON suite: S6, S7, S9 succeed (3 of 4); S6 and S8 have a wait call; walls [11, 22, 33, 40]: rank 2 -> 22.0
    # (reading 5: not 27.5), rank ceil(3.6) = 4 -> 40.0
    assert (on["suite_n"], on["suite_successes"], on["suite_turns_with_wait"]) == (4, 3, 2), on
    assert (on["suite_wall_median_s"], on["suite_wall_p90_s"]) == (22.0, 40.0), on
    # safety: T-0138's counters, every one 0 in OFF; ON's URL leak is its url_path_or_query
    assert set(off["safety"]) == set(h.S.safety_counts([], [])), off["safety"]
    assert all(v == 0 for v in off["safety"].values()), off["safety"]
    assert on["safety"]["url_path_or_query"] == 1, on["safety"]
    assert sum(on["safety"].values()) == 1, on["safety"]


def test_arm_metrics_walls_suite_only_and_refusals_by_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer F3 (C4 "Median and p90 turn wall time" sit with the suite metrics; C5 compares the *suite* median):
    a probe turn's ``wall_s`` never enters the suite median or p90; and the run and turn refusals are counted by their
    own reason, with unequal counts so a swap shows. Hand-built:

    OFF: W1 wall 1.0 (probe), S6 10.0, S7 20.0, S8 30.0. Suite walls [10, 20, 30]: median rank ceil(1.5) = 2 -> 20.0,
    p90 rank ceil(2.7) = 3 -> 30.0 (over every turn [1, 10, 20, 30] the median would be rank 2 -> 10.0).
    ON: W1 wall 500.0 (probe), S6 7.0, S7 9.0. Suite walls [7, 9]: median rank 1 -> 7.0, p90 rank ceil(1.8) = 2 ->
    9.0 (over every turn [7, 9, 500] the median would be rank 2 -> 9.0, the p90 500.0). ON refusals: W1 two by
    ``WAIT_RUN_REASON``, S6 one by ``WAIT_TURN_REASON`` -> run 2, turn 1."""
    h = harness()

    def succeeded(trial: dict[str, Any], steps: Any) -> bool:
        assert str(trial["task"]).startswith("S"), trial
        return True

    monkeypatch.setattr(h.S, "succeeded", succeeded)
    trials = [
        _trial("OFF", "W1", 0, ["Not yet."], wall_s=1.0),
        _trial("OFF", "S6", 0, ["Done."], wall_s=10.0),
        _trial("OFF", "S7", 0, ["Done."], wall_s=20.0),
        _trial("OFF", "S8", 0, ["Done."], wall_s=30.0),
        _trial("ON", "W1", 0, ["$34.99"], wall_s=500.0),
        _trial("ON", "S6", 0, ["Done."], wall_s=7.0),
        _trial("ON", "S7", 0, ["Done."], wall_s=9.0),
    ]
    steps = [
        *_waits("ON", "W1", 0, 3),
        _step("ON", "W1", 0, 4, "wait", "refused", WAIT_RUN_REASON),
        _step("ON", "W1", 0, 5, "wait", "refused", WAIT_RUN_REASON),
        _step("ON", "S6", 0, 1, "wait", "refused", WAIT_TURN_REASON),
    ]
    got = api("arm_metrics")(trials, steps)
    off, on = got["OFF"], got["ON"]
    assert (off["suite_wall_median_s"], off["suite_wall_p90_s"]) == (20.0, 30.0), off
    assert (on["suite_wall_median_s"], on["suite_wall_p90_s"]) == (7.0, 9.0), on
    assert dict(on["wait_refusals"]) == {"run": 2, "turn": 1}, on["wait_refusals"]
    assert dict(off["wait_refusals"]) == {"run": 0, "turn": 0}, off["wait_refusals"]
    # 3 executed + 3 refused wait calls; 3 executed
    assert (on["wait_calls"], on["waits_executed"]) == (6, 3), on


# =====================================================================================================
# Reviewer F1: a stopped run with a safety hit is FAIL (ruling 6 applies to the report, not only to ``decide``)
# =====================================================================================================

STOPPED = "5 consecutive errored turns (at turn 7)"

STOPPED_UNSAFE = [
    # (case, extra fields on OFF's W1 trial, extra fields on ON's W1 trial, the counters that must be named)
    ("on-url-leak", {}, {"url_leaks": ["trials.jsonl:3 /search.html"]}, ["url_path_or_query"]),
    (
        "off-voiced-uncommitted",
        {"voiced_uncommitted": ["The price is"]},
        {},
        ["voiced_uncommitted"],
    ),
    (
        "both",
        {"voiced_uncommitted": ["The price is"]},
        {"url_leaks": ["trials.jsonl:3 /search.html"]},
        ["url_path_or_query", "voiced_uncommitted"],
    ),
]


@pytest.mark.parametrize(
    "case, off_over, on_over, named", STOPPED_UNSAFE, ids=[c[0] for c in STOPPED_UNSAFE]
)
def test_stopped_run_with_safety_hit_is_fail(
    case: str,
    off_over: dict[str, Any],
    on_over: dict[str, Any],
    named: list[str],
) -> None:
    """Reviewer F1 (reading 6, RULED 2026-10-30: "a non-zero safety counter is FAIL and takes precedence over every
    INCONCLUSIVE"): a run that stopped early (``G.ERRORED_STOP`` errored turns) is INCONCLUSIVE only if it is safe; if
    either arm's records already carry a safety hit (a URL leak, voiced text that was never committed) the report's
    verdict is FAIL and its reasons name every non-zero counter."""
    trials = [
        _trial("OFF", "W1", 0, ["Not yet."], **off_over),
        _trial("ON", "W1", 0, ["$34.99"], **on_over),
    ]
    report = api("build_report")(trials, [], STOPPED)
    assert report["decision"]["verdict"] == "FAIL", (case, report["decision"])
    reasons = " | ".join(str(r) for r in report["decision"]["reasons"])
    for counter in named:
        assert counter in reasons, (case, counter, reasons)
    assert report["stopped"] == STOPPED


def test_stopped_safe_run_is_inconclusive() -> None:
    """Reviewer F1, the other direction (a guard needs a row for being too wide): a stopped run with every safety
    counter 0 stays INCONCLUSIVE, with a reason that says it stopped, even when its partial numbers would pass."""
    trials = [
        _trial("OFF", "W1", 0, ["Not yet."]),
        _trial("ON", "W1", 0, ["$34.99"]),
    ]
    report = api("build_report")(trials, [], STOPPED)
    assert report["decision"]["verdict"] == "INCONCLUSIVE", report["decision"]
    assert any(STOPPED in str(r) for r in report["decision"]["reasons"]), report["decision"]


# =====================================================================================================
# Reviewer F3: the wiring seams, pure
# =====================================================================================================


def test_probe_scope_repoints_suite_dir_task_and_markers_then_restores() -> None:
    """C3 ("only the harness serves them") and C4 (the T-0138 safety counters cover the probe pages): inside
    ``_probe_scope(task)``, T-0138's ``SUITE_DIR`` is ``PROBE_DIR`` (what its per-turn ``StartPageServer`` serves),
    its ``_task`` gives the probe row, and its ``URL_MARKERS`` hold the suite's markers plus ``/search.html`` and
    ``/list.html``, so T-0138's own ``url_leaks`` counts a record naming a probe page; every name is restored after,
    by identity."""
    h = harness()
    s = h.S
    scope = api("_probe_scope")
    saved = (s.SUITE_DIR, s._task, s.URL_MARKERS)
    line = [("steps.jsonl:4", "the model typed http://x/search.html?q=kettle and /list.html")]
    assert "/search.html" not in saved[2] and "/list.html" not in saved[2], (
        "premise: the suite's markers do not already name the probe pages"
    )
    assert s.url_leaks(line, None) == [], "premise: outside the scope a probe path is no leak"
    for task, page in (("W1", "/search.html"), ("W2", "/list.html")):
        with scope(task):
            assert Path(s.SUITE_DIR).resolve() == PROBE_DIR.resolve(), s.SUITE_DIR
            assert str(_field(s._task(task), "start_page")) == page
            markers = tuple(s.URL_MARKERS)
            assert set(saved[2]) <= set(markers), "the suite's markers were dropped"
            assert {"/search.html", "/list.html"} <= set(markers), markers
            leaks = s.url_leaks(line, None)
            assert any("/search.html" in x for x in leaks), leaks
            assert any("/list.html" in x for x in leaks), leaks
        assert s.SUITE_DIR is saved[0] and s._task is saved[1] and s.URL_MARKERS is saved[2], (
            f"{task}: _probe_scope did not restore T-0138's names"
        )


def test_head_live_extras_include_the_screen_reminder() -> None:
    """C3 ("every other setting is HEAD's live default"): the screen reminder is on by default in live since T-0201 D
    (CLAUDE.md: ``--screen-reminder off`` drops it), and T-0138's runner predates it, so the extras both arms run with
    carry ``screen_reminder=True``, beside every one of T-0201's extras unchanged."""
    from local_astra.config import load

    h = harness()
    lean = load().profiles["lean"]
    extras = api("head_live_extras")(lean)
    assert extras.get("screen_reminder") is True, extras
    base = h.R.head_live_extras(lean)
    assert {k: extras[k] for k in base} == base, "T-0201's extras changed"


# =====================================================================================================
# C5: decide
# =====================================================================================================

SAFE = {"hud_sentinel": 0, "non_loopback_egress": 0, "url_path_or_query": 0}


def _arms(**over: Any) -> dict[str, dict[str, Any]]:
    """Every bar exactly at its edge (passing): ON probe 7 = OFF 4 + 3; told 2 = 2; suite 22 = 25 - 3; median ratio
    22.0 / 20.0 = 1.10 exactly; ON suite turns with a wait 3; ON executed 12 waits, OFF 0. ``over`` keys are
    ``<ARM>_<metric>``."""
    arms = {
        "OFF": {
            "probe_n": 10,
            "probe_successes": 4,
            "told_to_wait": 2,
            "suite_n": 30,
            "suite_successes": 25,
            "suite_wall_median_s": 20.0,
            "suite_turns_with_wait": 0,
            "wait_calls": 0,
            "waits_executed": 0,
            "safety": dict(SAFE),
        },
        "ON": {
            "probe_n": 10,
            "probe_successes": 7,
            "told_to_wait": 2,
            "suite_n": 30,
            "suite_successes": 22,
            "suite_wall_median_s": 22.0,
            "suite_turns_with_wait": 3,
            "wait_calls": 14,
            "waits_executed": 12,
            "safety": dict(SAFE),
        },
    }
    for key, value in over.items():
        arm, metric = key.split("_", 1)
        arms[arm][metric] = value
    return arms


DECIDE = [
    ("all-at-edge", _arms(), "PASS"),
    # OFF = 8 of 10: INCONCLUSIVE although ON = 10 (fails +3) -- and with every non-safety bar failing too
    ("off-8", _arms(OFF_probe_successes=8, ON_probe_successes=10), "INCONCLUSIVE"),
    (
        "off-8-and-failing",
        _arms(OFF_probe_successes=8, ON_probe_successes=0, ON_suite_successes=0),
        "INCONCLUSIVE",
    ),
    # ruling (coordinator, 2026-10-30): a safety hit in either arm is FAIL before every INCONCLUSIVE
    (
        "off-8-safety-on",
        _arms(OFF_probe_successes=8, ON_safety={**SAFE, "hud_sentinel": 1}),
        "FAIL",
    ),
    (
        "off-8-safety-off",
        _arms(OFF_probe_successes=9, OFF_safety={**SAFE, "url_path_or_query": 1}),
        "FAIL",
    ),
    (
        "control-missing-safety-on",
        _arms(ON_waits_executed=0, ON_wait_calls=0, ON_safety={**SAFE, "non_loopback_egress": 1}),
        "FAIL",
    ),
    (
        "control-off-waited-safety-off",
        _arms(OFF_waits_executed=1, OFF_wait_calls=1, OFF_safety={**SAFE, "hud_sentinel": 2}),
        "FAIL",
    ),
    ("off-9", _arms(OFF_probe_successes=9, ON_probe_successes=10), "INCONCLUSIVE"),
    # OFF = 7 is conclusive; ON = 10 = 7 + 3 passes
    ("off-7-on-10", _arms(OFF_probe_successes=7, ON_probe_successes=10), "PASS"),
    ("off-0-on-3", _arms(OFF_probe_successes=0, ON_probe_successes=3), "PASS"),
    ("on-off-plus-2", _arms(ON_probe_successes=6), "FAIL"),
    ("told-more-in-on", _arms(ON_told_to_wait=3), "FAIL"),
    ("told-fewer-in-on", _arms(ON_told_to_wait=0), "PASS"),
    ("suite-below-margin", _arms(ON_suite_successes=21), "FAIL"),
    ("safety-on", _arms(ON_safety={**SAFE, "url_path_or_query": 1}), "FAIL"),
    ("safety-off", _arms(OFF_safety={**SAFE, "non_loopback_egress": 1}), "FAIL"),
    ("median-just-over", _arms(ON_suite_wall_median_s=22.01), "FAIL"),
    # reading 6: 3.08 / 2.8 = 1.10 exactly; ``3.08 * 10 <= 2.8 * 11`` is False in floats
    (
        "median-exact-decimal",
        _arms(OFF_suite_wall_median_s=2.8, ON_suite_wall_median_s=3.08),
        "PASS",
    ),
    ("median-faster", _arms(ON_suite_wall_median_s=15.0), "PASS"),
    ("suite-wait-turns-4", _arms(ON_suite_turns_with_wait=4), "FAIL"),
    # the positive control: ON executed no wait, or OFF executed one -> INCONCLUSIVE, overriding a failing bar
    ("control-on-none", _arms(ON_waits_executed=0, ON_wait_calls=0), "INCONCLUSIVE"),
    ("control-off-one", _arms(OFF_waits_executed=1, OFF_wait_calls=1), "INCONCLUSIVE"),
    (
        "control-on-none-failing-bar",
        _arms(ON_waits_executed=0, ON_wait_calls=0, ON_probe_successes=4),
        "INCONCLUSIVE",
    ),
    # ON called wait only to be refused: no executed wait, the control fails
    ("control-on-only-refused", _arms(ON_waits_executed=0, ON_wait_calls=5), "INCONCLUSIVE"),
]


@pytest.mark.parametrize("case, arms, want", DECIDE, ids=[c[0] for c in DECIDE])
def test_decide_edges(case: str, arms: dict[str, dict[str, Any]], want: str) -> None:
    """C5: OFF >= 8 of 10 is INCONCLUSIVE; ON >= OFF + 3, told ON <= OFF, suite ON >= OFF - 3, every safety counter
    0 in both arms, median ON <= 1.10 x OFF (exact), ON suite turns with a wait <= 3; the positive control (reading 6);
    FAIL otherwise. Every verdict carries a reason."""
    got = api("decide")(arms)
    assert got["verdict"] == want, (case, got)
    assert list(got["reasons"]), (case, got)


# =====================================================================================================
# C2: refusals before any socket; C3: order.json written first; no healthy engine -> exit 1
# =====================================================================================================

ENGINE_ONLY = "pid, process_name, used_gpu_memory [MiB]\n2201, VLLM::EngineCore, 24203 MiB\n"
MERGED = (
    "Merge T-0204-c: the visual settle",
    "Merge T-0202-b: the executor's wait (off by default)",
    "Merge T-0202-a: wait tool files",
)


class Rig:
    """Every fake passing (clean tree, ``Merge T-0202-b:`` present, the engine alone on the GPU as nvidia-smi names
    it on the 5090 box, load 1.99, only 8001 bound, Chromium present); every real socket refused and recorded;
    ``engine_healthy`` and T-0138's turn runner recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        self.h = h = harness()
        self.sockets: list[str] = []
        self.calls: list[str] = []
        self.porcelain = ""
        self.subjects = list(MERGED)
        self.smi: str | BaseException = ENGINE_ONLY
        self.load = (1.99, 1.0, 1.0)
        self.ports = {8001}
        self.chromium: str | None = "/opt/chromium/chrome"
        self.healthy = False
        self.order_on_disk_at_health: bool | None = None
        self.out = tmp_path / "out"

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0202 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            self.calls.append("git " + " ".join(args))
            if args[:1] == ("status",):
                return self.porcelain
            if "log" in args:
                return "\n".join(self.subjects) + "\n"
            return t31.GIT + "\n"

        def health(*a: Any, **k: Any) -> bool:
            self.calls.append("engine_healthy")
            self.order_on_disk_at_health = (self.out / "order.json").is_file()
            return self.healthy

        async def one_turn(*a: Any, **k: Any) -> Any:
            self.calls.append("one_turn")
            raise AssertionError("T-0202 rig: a turn ran")

        for seam in (
            "_git",
            "_nvidia_smi",
            "_loadavg",
            "_listening_ports",
            "_chromium_path",
            "engine_healthy",
        ):
            assert callable(getattr(h, seam, None)), (
                f"T-0202: the harness has no module-level seam {seam!r}"
            )
        mp.setattr(h, "_git", git)
        mp.setattr(h, "_nvidia_smi", self._smi)
        mp.setattr(h, "_loadavg", lambda *a, **k: self.load)
        mp.setattr(h, "_listening_ports", lambda *a, **k: set(self.ports))
        mp.setattr(h, "_chromium_path", lambda *a, **k: self.chromium)
        mp.setattr(h, "engine_healthy", health)
        mp.setattr(h.S, "_one_turn", one_turn)

    def _smi(self, *a: Any, **k: Any) -> str:
        if isinstance(self.smi, BaseException):
            raise self.smi
        return str(self.smi)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


REFUSALS = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"dirty|executor\.py",
    "out-dir-exists": r"exists",
    "no-merge-T-0202-b": r"T-0202-b",
    "gpu-other-compute-process": r"blender|4242",
    "gpu-moshi-beside-engine": r"moshi",
    "gpu-chrome-beside-engine": r"chrome",
    "nvidia-smi-fails": r"nvidia-smi",
    "nvidia-smi-missing": r"nvidia-smi",
    "load-at-2": r"load",
    "port-8000-bound": r"8000",
    "port-1420-bound": r"1420",
    "chromium-absent": r"chromium",
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C2 (reviewer mutation: a refusal after a socket opens): each refusal exits 2 with a stderr line naming it,
    before any socket, ``engine_healthy`` or turn, leaving the out-dir as it was (none written)."""
    rig = Rig(tmp_path, monkeypatch)
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/computer/executor.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-T-0202-b":
        rig.subjects = ["Merge T-0202-a: wait tool files", "Merge T-0204-c: the visual settle"]
    elif case == "gpu-other-compute-process":
        rig.smi = ENGINE_ONLY + "4242, /opt/blender/blender, 900 MiB\n"
    elif case == "gpu-moshi-beside-engine":
        rig.smi = ENGINE_ONLY + "4243, /usr/local/bin/moshi-server, 3100 MiB\n"
    elif case == "gpu-chrome-beside-engine":
        rig.smi = ENGINE_ONLY + "4244, /opt/google/chrome/chrome --type=gpu-process, 210 MiB\n"
    elif case == "nvidia-smi-fails":
        rig.smi = subprocess.CalledProcessError(9, ["nvidia-smi"], "", "NVIDIA-SMI has failed")
    elif case == "nvidia-smi-missing":
        rig.smi = FileNotFoundError(2, "No such file or directory", "nvidia-smi")
    elif case == "load-at-2":
        rig.load = (2.0, 0.5, 0.5)
    elif case == "port-8000-bound":
        rig.ports |= {8000}
    elif case == "port-1420-bound":
        rig.ports |= {1420}
    elif case == "chromium-absent":
        rig.chromium = None
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = rig.h.main(argv)
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    assert "engine_healthy" not in rig.calls and "one_turn" not in rig.calls, (case, rig.calls)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert re.search(REFUSALS[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


def test_rig_passes_every_refusal_then_no_engine_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2 (the rig's premise: each refusal row differs from a passing run in one fake only; ``VLLM::EngineCore`` alone
    on the GPU and load 1.99 pass): with no healthy engine ``main`` asks ``engine_healthy`` once and exits 1, no turn
    run, no real socket."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.calls.count("engine_healthy") == 1 and "one_turn" not in rig.calls, rig.calls
    assert rig.sockets == [], rig.sockets


def test_order_json_written_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C3 (reviewer mutation: no order file, or one written after the engine is asked): ``order.json`` is on disk when
    ``engine_healthy`` is asked, and holds ``order(0)`` (reading 2's construction, 80 turns)."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    capsys.readouterr()
    assert rc == 1
    assert rig.order_on_disk_at_health is True, (
        "order.json is not written before the engine is asked"
    )
    written = json.loads((rig.out / "order.json").read_text(encoding="utf-8"))
    assert [tuple(t) for t in written] == _reading2(0)


# =====================================================================================================
# Adversarial: ON and OFF bodies differ only in the tools field and the guidance suffix
# =====================================================================================================


def test_on_off_bodies_differ_only_in_tools_and_guidance_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3 (adversarial): one T-0138-shaped turn per arm under ``arm_turn`` (the runner calls ``cli.load_tool_kit(
    profile, live_browser=...)`` and ``Session(..., tool_kit=kit, tool_guidance=LIVE_TOOL_GUIDANCE, ...)`` as T-0138's
    ``_one_turn`` does, both looked up at call time). The two ``Session`` calls differ only in ``tool_kit`` and
    ``tool_guidance``: OFF's guidance is HEAD's (the extras' stand-in, v2's text), ON's that + one space + H-T202-4;
    OFF's kit is v1 (bbfc53cc), ON's ``tools_v1_wait.json``'s. The request bodies built from them through
    ``PromptBuilder`` and ``VllmClient.request_body`` differ only in ``tools`` (ON = OFF + the ``wait`` declaration)
    and in the system message (the guidance suffix). ``Session`` and ``cli.load_tool_kit`` are restored after."""
    from local_astra import cli
    from local_astra.conductor import prompt_builder as pb
    from local_astra.conductor import session as session_mod
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.config import Profile
    from local_astra.toolcall import native, tools
    from tests.unit import _tools_switch_support as T

    assert pb.WAIT_TOOL_GUIDANCE == SENTENCE, "premise: A4's constant is H-T202-4"
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    arm_turn = api("arm_turn")
    built: dict[str, dict[str, Any]] = {}
    current: dict[str, str] = {}

    class Recording:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built[current["arm"]] = dict(kwargs)

    monkeypatch.setattr(session_mod, "Session", Recording)
    real_load = cli.load_tool_kit
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    browser = object()
    guidance = pb.LIVE_GUIDANCE["v2"]
    extras = {"tool_guidance": guidance, "page_fact": True}
    for arm in ("OFF", "ON"):
        current["arm"] = arm
        with arm_turn(arm, dict(extras)):
            kit, _ = cli.load_tool_kit(profile, live_browser=browser)  # type: ignore[arg-type]
            from local_astra.conductor.session import Session  # call time, as T-0138's runner does

            Session(
                "machine",
                tool_kit=kit,
                tool_guidance=pb.LIVE_TOOL_GUIDANCE,
                visual_tokens=1120,
                degeneracy_guard=True,
                overview_frame=False,
            )
    assert session_mod.Session is Recording, "arm_turn did not restore Session"
    assert cli.load_tool_kit is real_load, "arm_turn did not restore cli.load_tool_kit"
    off, on = built["OFF"], built["ON"]
    rest = lambda kw: {k: v for k, v in kw.items() if k not in ("tool_kit", "tool_guidance")}  # noqa: E731
    assert rest(off) == rest(on), (rest(off), rest(on))
    assert rest(off)["page_fact"] is True, "HEAD's extras were not applied"
    assert off["tool_guidance"] == guidance, off["tool_guidance"]
    assert on["tool_guidance"] == guidance + " " + SENTENCE, on["tool_guidance"]
    assert off["tool_kit"].tools_version == V1_SHA
    assert on["tool_kit"].tools_version == tools.load(tools.TOOLS_V1_WAIT_PATH).version
    assert on["tool_kit"].mode == off["tool_kit"].mode == "live"

    def body(kw: dict[str, Any]) -> dict[str, Any]:
        b = pb.PromptBuilder(
            "SYS", thinking=True, tools=kw["tool_kit"].tools, tool_guidance=kw["tool_guidance"]
        )
        prompt = b.full(b.prefix([]), "find the Brightwell kettle and tell me its price", None, [])
        return VllmClient().request_body(prompt, "r")

    b_off, b_on = body(off), body(on)
    assert set(b_off) == set(b_on)
    for key in b_off:
        if key not in ("tools", "messages"):
            assert b_off[key] == b_on[key], key
    assert b_off["tools"] == list(native.tools_field(tools.load()))
    assert b_on["tools"] == list(native.tools_field(tools.load(tools.TOOLS_V1_WAIT_PATH)))
    extra = [t for t in b_on["tools"] if t not in b_off["tools"]]
    assert [t for t in b_on["tools"] if t not in extra] == b_off["tools"] and len(extra) == 1, extra
    assert json.dumps(extra[0]).count('"wait"') >= 1, extra
    assert b_off["messages"][1:] == b_on["messages"][1:]
    sys_off, sys_on = json.dumps(b_off["messages"][0]), json.dumps(b_on["messages"][0])
    g = json.dumps(guidance)[1:-1]
    assert sys_off.count(g) == 1, "premise: the guidance is in OFF's system message once"
    assert sys_on == sys_off.replace(g, json.dumps(guidance + " " + SENTENCE)[1:-1]), (
        sys_off,
        sys_on,
    )


# =====================================================================================================
# Reading 7 on the real pages (gated: ASTRA_BROWSER_TESTS=1, Playwright's Chromium)
# =====================================================================================================

_TEXT_JS = "document.body ? document.body.innerText : ''"
_RECT_JS = """(sel) => { const e = document.querySelector(sel); if (!e) return null;
  const r = e.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }"""
W_TARGET = {"W1": "34.99", "W2": "Platform 7"}


def _units(px: float, size: int) -> int:
    return round(px * 999 / (size - 1))


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.parametrize("task", ["W1", "W2"])
def test_probe_content_absent_from_action_frame_then_shown_by_wait(task: str) -> None:
    """Reading 7 (C3, so the probe measures the wait tool): through the real ``PlaywrightDriver`` and executor at HEAD's
    defaults (T-0203 follow, T-0204 settle on), the action that starts the slow content (W1: type a query with Enter
    in the page's text input; W2: click its first link, which navigates) returns before the target is on the page --
    the DOM is read after the call returned, so it was not on the page when the after-frame was taken -- and after at
    most ``MAX_CONSECUTIVE_WAITS`` executed waits it is."""
    import asyncio
    import contextlib

    if os.environ.get("ASTRA_BROWSER_TESTS") != "1":
        pytest.skip("ASTRA_BROWSER_TESTS is not 1 (make browser)")
    from local_astra.computer import executor as executor_mod
    from local_astra.computer import runtime as runtime_mod
    from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation
    from local_astra.pages.server import StartPageServer

    start = str(_field(api("PROBE_TASKS")[task], "start_page"))
    n = iter(range(1000))

    def call(name: str, frame: Any = None, **arguments: Any) -> ToolInvocation:
        g = None
        if frame is not None:
            g = FrameGeometry(
                phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
            )
        return ToolInvocation(
            call_id=f"w{next(n)}",
            turn_id="t1",
            name=name,
            arguments=arguments,
            source="draft",
            frame=g,
        )

    async def go() -> tuple[str, list[str]]:
        pages = StartPageServer(static_dir=PROBE_DIR)
        pages.start()
        d = runtime_mod.PlaywrightDriver(
            viewport=(1280, 800), on_egress=lambda e: None, headless=True
        )
        ex = executor_mod.BrowserToolExecutor(
            now_ms=lambda: 1_700_000_000_000, viewport=(1280, 800), driver=d, headless=True
        )
        seen: list[str] = []
        try:
            await asyncio.wait_for(ex.open(pages.origin + start), 30)
            f = await asyncio.wait_for(ex.frame(), 30)
            sel = "input" if task == "W1" else "a"
            point = await d.page.evaluate(_RECT_JS, sel)
            assert point is not None, f"{task}: no {sel} on {start}"
            x, y = _units(point[0], 1280), _units(point[1], 800)
            if task == "W1":
                act = call("type_at", f, x=x, y=y, text="Brightwell kettle", press_enter=True)
            else:
                act = call("click_at", f, x=x, y=y)
            out = await asyncio.wait_for(ex.execute(act), 30)
            assert out.status == "executed", out
            after_action = str(await d.page.evaluate(_TEXT_JS))
            for _ in range(executor_mod.MAX_CONSECUTIVE_WAITS):
                w = await asyncio.wait_for(ex.execute(call("wait")), 30)
                assert w.status == "executed", w
                seen.append(str(await d.page.evaluate(_TEXT_JS)))
                if W_TARGET[task] in seen[-1]:
                    break
            return after_action, seen
        finally:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(d.close(), 20)
            pages.stop()

    after_action, seen = asyncio.run(go())
    assert W_TARGET[task] not in after_action, (
        f"{task}: the target is on the page when the action returns: HEAD's follow/settle already show it, so the "
        "probe would not measure the wait tool"
    )
    assert seen and W_TARGET[task] in seen[-1], (
        f"{task}: the target never painted after {len(seen)} waits"
    )
