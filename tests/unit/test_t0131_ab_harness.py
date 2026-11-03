"""T-0131 group D: the 6-arm grounding A/B harness's pure parts and its locked script (unit; CPU; no engine, no browser).

Rows that need the harness are ``pending(task="T-0131")`` and red on the named assertion in ``harness()`` / ``api()``
(the file or a name is absent) or on the locked-script rig's ``_script()``, never on an import error. The harness is
loaded lazily, by path, as ``tests/unit/test_t0111g_smoke_harness.py`` loads T-0111g's; ``main(argv)`` is called
in-process with every socket refused and recorded. Rows named ``*eval_oracle*`` are not pending: they check
``tests/eval/test_t0131_ab.py``'s oracle against the same hand tables, so a wrong oracle is caught on CPU.

**The assumed harness surface** (``scripts/gpu/t0131_grounding_ab.py``). The task names the behaviour; these names and
shapes are the test-author's, stated once so the implementer can implement to them (or say why not).

Imported from T-0111g, never copied (D1)
    ``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``, ``non_loopback_sockets``,
    ``turn_records`` are module attributes of the harness whose source file is ``scripts/gpu/t0111g_live_smoke.py``
    (an alias such as ``TASKS = g.TASKS`` is fine; a ``def``, ``class`` or literal of those names is not).
    ``TASK_TABLE_SHA256`` is T-0111g's value. ``main`` reads the tree through the module-level ``_git`` name.

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``.
    ``ARMS = ((280, False), (280, True), (560, False), (560, True), (1120, False), (1120, True))``, read by ``main`` at
    call time. ``arm_table_sha(arms=None) -> str``: the sha256 hex of ``json.dumps([{"visual_tokens": b, "reground":
    r} for b, r in arms], sort_keys=True, separators=(",", ":"))`` (``None`` means ``ARMS``); ``ARM_TABLE_SHA256`` is
    that of the table above (this module's ``ARM_TABLE_SHA256``). ``ORDER_SEED = 0``.
    ``INTENDED_TARGETS: Mapping[str, Sequence]``: per task id, the intended click targets (D3; their shape is the
    implementer's): S1 three, S2 one, S3 four, S4 none, S5 one.

Pure functions
    ``arm_label(visual_tokens, reground) -> str``: ``"280/off"``, ``"1120/on"``.
    ``order(seed) -> list[tuple[int, bool, str, int]]``: the 150 ``(visual_tokens, reground, task_id, repeat)`` turns,
    shuffled jointly (one shuffle over every arm, task and repeat), the same in every call and process.
    ``arm_profile(profile, visual_tokens, reground) -> Profile``: a copy with ``tools = "live"``, ``browser =
    "headless"``, ``visual_tokens`` set explicitly (280 too, never ``None``) and ``reground`` per arm; the argument is
    not changed.
    ``p1_ok(budget, delta) -> bool``: D2 P1, ``expected <= delta <= expected + 4`` with 273/522/1092.
    ``click_score(point, boxes) -> tuple[bool, float | None]``: ``point`` is ``(x, y)`` in viewport CSS px, a box is
    ``{"x", "y", "width", "height"}`` (a ``DOMRect``); a hit is inside a box, edges included; the error is the
    Euclidean distance to the nearest box (0.0 inside); ``(False, None)`` with no box.
    ``false_claim(trial, steps) -> bool`` (``steps``: the turn's own step lines): not a success (by ``score``), no
    exception, not timed out, and the last step made no call (``status`` and ``call_name`` None) and voiced text with
    no ``?``.
    ``reground_voiced_findings(pass2_answers, voiced, history, step_texts) -> list[str]``: each pass-2 answer text
    (whitespace-collapsed, non-empty) inside a voiced source or the history and inside none of the turn's own step
    texts.
    ``decide(inputs, *, p1_passed) -> str``: D5's label, ``"inconclusive"``, ``"no change"`` or the chosen arm's
    ``arm_label``. ``inputs`` maps each of the six labels to ``{"accepted": bool, "safety": int (the sum of the
    arm's eight counts), "engine_up": bool, "success": int (of 25), "hit_rate": float | None, "done_p95_ms": number |
    None, "reground_p50_ms": number | None}``.
    ``build_report(trials, steps, regrounds, probes) -> dict`` and ``render_markdown(report, manifest=None) -> str``.

Records (``trials.jsonl``, ``steps.jsonl``, ``regrounds.jsonl``, ``probes.json``)
    Trial and step lines are T-0111g's (``turn_records``), plus ``visual_tokens`` and ``reground`` on every line. A
    trial adds ``wall_s``, ``vram_min_free_mib`` (the minimum free MiB sampled during the turn, or ``None``),
    ``engine_up_after`` (``bool | None``) and ``reground_voiced`` (``reground_voiced_findings``' list). A step adds
    ``executed_name`` (the tool the browser executed, from ``_Watch``; ``None`` if none), and ``click`` and
    ``click_p1`` (``None``, or ``{"point": [x, y], "boxes": [box, ...], "hit", "error_px"}``): ``click`` on every
    executed ``click_at``/``type_at``, the boxes read in ``_Watch.execute`` before dispatch; ``click_p1`` in
    ReGround-on arms, pass 1's point scored against the same boxes. ``call_name`` is T-0111g's (pass 1's proposed
    name). A ``regrounds.jsonl`` line is one ``tool.reground`` record: ``visual_tokens``, ``reground``, ``task``,
    ``repeat``, ``step``, ``call_id``, ``call_name``, ``outcome`` (B3), ``ms``, ``attempt_id``, ``http_status``,
    ``first_commit_ms``, ``done_ms`` (the last four ``None`` for ``skipped_no_point``; ``http_status`` 0 for an
    aborted request), ``answer_text``, ``tool_text``. ``probes.json``: ``{"p1": {"ok", "budgets": {"280": {"frame_delta",
    "crop_delta", "expected", "ok"}, "560": ..., "1120": ...}}, "p2": {"ok", "http_status", "healthy_after"}, "p3":
    {"ok", "http_status"}}``.

``build_report`` (``report.json``)
    ``{"void": any probe failed, "probes": probes, "arms": {label: section} for all six labels, "decision": {"label",
    "eligible", "inputs"}}``. A section has the keys of ``tests/eval/test_t0131_ab.py``'s ``ARM_SECTIONS``, built as
    its ``derive_arm`` builds them (T-0111g's ``build_report`` sections per arm, plus ``clicks``, ``clicks_p1``,
    ``steps.scrolls``, ``false_claims``, ``reground_cost``, ``wall_s``, ``vram_min_free_mib``, ``engine_up`` and the
    two new safety counts ``reground_voiced`` and ``reground_other_tool_executed``). Step latency is read from the step
    lines only, so reground requests never enter it. ``report.md`` names every arm, the decision label and the T-0111g
    run at ca9576f beside the results (not like-for-like).

``main(argv) -> int``
    ``--base-url`` and ``--out-dir`` (default ``logs/t0131-grounding-ab-<sha7>``). T-0111g's refusals plus an
    ``arm_table_sha()`` other than ``ARM_TABLE_SHA256``, each exit 2 before any socket, browser, start-page server or
    ``nvidia-smi``; with no engine on the loopback URL, exit 1 (or 3) having built no browser.

The locked script ``scripts/locked/t0131_grounding_ab.sh`` (the T-0111g rig)
    No arguments; ``REFUSED`` and exit 2 before the boot on an argument, a dirty tree, a bound Astra port, running
    containers, ``services/vllm.sh`` changed since 806da12, an existing ``logs/t0131-grounding-ab-<SHA>``, and no
    ``Merge T-0131`` merge reachable from HEAD (another task's merge, ``Merge T-0131b`` included, does not count).
    T-0111g's boot lines verbatim, one ``setsid`` of ``services/vllm.sh`` with nothing added; the harness under
    ``timeout 90m`` with a loopback ``--base-url`` and ``--out-dir logs/t0131-grounding-ab-<SHA>``; then
    ``tests/eval/test_t0131_ab.py -m gpu`` with ``T0131_RUN_DIR`` set to it; the six proxy variables unset; the
    engine container stopped on exit, after a failing step too; exit 1 when a step fails.

**Readings where the task is open** (named in the test-author report): "the best arm" in D5 is the best of all six
(the hand tables give the same label under "best of the gated arms"); the no-change clause compares the chosen arm
with 280/off; S4's
clicks are reported as ``unscored`` and left out of the hit rate; a "question asked" is a ``?`` in the last step's
voiced text; the HUD spoken pane carries the voiced text, so ``reground_voiced`` reads the voiced sources and the
history; a void report whose P1 passed is not labelled by these rows.

**The aborted pass 2 (orchestrator ruling at 23c6c77, reading 3 overruled).** A pass 2 aborted by
``REGROUND_TIMEOUT_MS`` (outcome ``pass2_timeout``, ``http_status`` 0) is the harness's own timeout, not an engine
rejection: it is reported under ``reground_cost`` (its outcome count, its ``ms``) and never in ``acceptance``. The
acceptance section's ``regrounds``, ``regrounds_ok`` and ``reground_non_200`` count the other reground requests
(``skipped_no_point`` is never a request), so a pass 2 that got a non-200 response (``error`` at HTTP 500) still fails
the gate. Keyed on the outcome, not on the status: a non-timeout request recorded with ``http_status`` 0 still counts
as non-200 (the test-author's reading, reported; T-0111g's rule for cancelled step generates).

**The group D review round (orchestrator rulings at 4337125, and its acceptance of the implementer's choices).**
``REGROUND_TIMEOUT_MS`` (4000, the session's; ``main`` refuses with exit 2 when they differ): a ``pass2_timeout``
counts as 4000.0 in ``reground_cost.latency_ms`` and so in D5's p50 gate. ``decide(inputs, *, p1_passed, void=False,
stopped=None)`` labels "inconclusive", then "void" (P2/P3), then "incomplete" (stopped early), then D5's;
``build_report(..., *, stopped=None)`` adds a top-level ``stopped``; ``decision.eligible`` is ``[]`` under the first
three labels. Click records carry ``read_error`` (None, or the failure's repr) and both click sections
``read_errors``. ``config_evidence(bodies, guidance) -> (budgets_sent, guidance_sent)`` over the scan's ``{label:
body}``; each trial carries both. ``report.md``: the void parenthetical only when void; click-error p50/p95/mean to two
decimals. The failed-probe and turn rows drive ``main`` on CPU: T-0111g's ``_get_text``/``read_metrics`` seams stand
in for the engine, ``VllmClient`` keeps its real ``request_body`` over a fake, ``order``, ``run_probes`` and
``probe_p1`` are patched by name, and the browser side is group B's ``FakeLiveBrowser``.

Regime: CPU, pure, hand-built records and stub commands; no engine, no browser, no measurement. The two ``main``
turn rows run real ``Session`` turns over fakes on the session's wall clock (two turns, about a second each).
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import types
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_t0131_ab as oracle
from tests.unit import test_t0111g_smoke_harness as g_tests

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0111G = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
LOCKED = REPO / "scripts" / "locked" / "t0131_grounding_ab.sh"
LOCKED_T0111G = REPO / "scripts" / "locked" / "t0111g_live_smoke.sh"

#: D1, typed from docs/tasks/T-0131.md
SPEC_ARMS: tuple[tuple[int, bool], ...] = (
    (280, False),
    (280, True),
    (560, False),
    (560, True),
    (1120, False),
    (1120, True),
)
#: the arm-table pin: ``arm_sha(SPEC_ARMS)``, computed 2026-10-22 by the test-author from the table above
ARM_TABLE_SHA256 = "5cc0169fe0cf4612f982c68fc20c2feb63ae803d6f70ceff372ed4d6209f32e2"
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
LABELS = ("280/off", "280/on", "560/off", "560/on", "1120/off", "1120/on")
#: D1: the T-0111g names the harness imports
IMPORTED = (
    "TASKS",
    "score",
    "LoopMeter",
    "sentinel_hits",
    "url_leaks",
    "log_complete",
    "non_loopback_sockets",
    "turn_records",
)
OUTCOMES = (
    "ok",
    "pass2_unparseable",
    "pass2_no_point",
    "pass2_other_tool",
    "pass2_timeout",
    "error",
    "skipped_no_point",
)
SAFETY8 = (
    "hud_sentinel",
    "non_loopback_egress",
    "browser_egress_blocked",
    "url_path_or_query",
    "executed_but_raised",
    "voiced_uncommitted",
    "reground_voiced",
    "reground_other_tool_executed",
)


def arm_sha(arms: Sequence[tuple[int, bool]]) -> str:
    blob = json.dumps(
        [{"visual_tokens": b, "reground": r} for b, r in arms],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# =====================================================================================================
# Loading
# =====================================================================================================


def load_harness(name: str) -> ModuleType:
    """The harness, executed by path under ``name``. Red here, by name, until group D is implemented."""
    assert HARNESS.is_file(), (
        f"T-0131 D1: {HARNESS.relative_to(REPO)} does not exist yet "
        "(its assumed surface is in tests/unit/test_t0131_ab_harness.py's docstring)"
    )
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None, f"{HARNESS} is not importable by path"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return module


@functools.cache
def harness() -> ModuleType:
    return load_harness("_t0131_grounding_ab_unit")


def api(name: str) -> Any:
    """A name of the assumed surface; its absence is an AssertionError naming it."""
    h = harness()
    assert hasattr(h, name), (
        f"T-0131 D1: {HARNESS.name} has no {name!r} (see tests/unit/test_t0131_ab_harness.py's docstring)"
    )
    return getattr(h, name)


#: a row run against the harness (pending) and against the eval oracle (not pending)
IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]


def impl(which: str, name: str) -> Any:
    return api(name) if which == "harness" else getattr(oracle, name)


@contextmanager
def no_sockets() -> Iterator[list[str]]:
    """Every socket attempt (connect, bind, name lookup), loopback included, is recorded and refused."""
    attempts: list[str] = []
    mp = pytest.MonkeyPatch()

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{what} {args[1:] if what.startswith('socket.') else args}")
            raise AssertionError(f"T-0131: a {what} where none is allowed")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        mp.setattr(socket.socket, name, refuse(f"socket.{name}"))
    mp.setattr(socket, "create_connection", refuse("create_connection"))
    mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    try:
        yield attempts
    finally:
        mp.undo()


def _top_level(tree: ast.Module) -> tuple[set[str], set[str]]:
    """``(imported modules, names defined by def/class/literal assignment)`` at module level (``if`` blocks
    skipped: ``if TYPE_CHECKING:`` imports nothing at run time)."""
    modules: set[str] = set()
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or isinstance(value, (ast.Attribute, ast.Name)):
                continue  # an alias of an imported name
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defined |= {t.id for t in targets if isinstance(t, ast.Name)}
    return modules, defined


# =====================================================================================================
# D1: import, the task table, the arm table, the order, the arm profile
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """D1 / Implementation notes (Harness): importing the harness (and the T-0111g harness it imports) opens no
    socket, and everything engine- or browser-facing is imported inside ``main``."""
    with no_sockets() as attempts:
        h = load_harness("_t0131_grounding_ab_import_check")
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    modules, _ = _top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    facing = {
        "httpx",
        "playwright",
        "playwright.async_api",
        "local_astra.conductor.llm_client",
        "local_astra.conductor.session",
        "local_astra.computer",
        "local_astra.computer.executor",
        "local_astra.computer.runtime",
        "local_astra.pages",
        "local_astra.pages.server",
    }
    assert modules & facing == set(), f"imported at module level: {sorted(modules & facing)}"


def test_task_table_is_t0111g() -> None:
    """D1: the harness imports T-0111g's ``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``,
    ``log_complete``, ``non_loopback_sockets`` and ``turn_records`` (each defined in
    ``scripts/gpu/t0111g_live_smoke.py``; none defined again in the harness), so ``TASK_TABLE_SHA256`` is T-0111g's
    pin and the table is C-G1's, field by field."""
    h = harness()
    want = T0111G.resolve()
    for name in IMPORTED:
        obj = api(name)
        target = type(obj[0]) if name == "TASKS" else obj
        src = inspect.getsourcefile(target)
        assert src is not None and Path(src).resolve() == want, (
            f"D1: the harness's {name} comes from {src}, not {T0111G.relative_to(REPO)} (imported, not copied)"
        )
    _, defined = _top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    assert defined & set(IMPORTED) == set(), (
        f"D1: defined again in the harness: {sorted(defined & set(IMPORTED))}"
    )
    assert api("TASK_TABLE_SHA256") == g_tests.TASK_TABLE_SHA256, (
        "the task-table pin is not T-0111g's"
    )
    assert g_tests._rows_of(h.TASKS) == g_tests.SPEC_ROWS, g_tests._rows_of(h.TASKS)
    assert g_tests.canonical_sha(g_tests._rows_of(h.TASKS)) == g_tests.TASK_TABLE_SHA256


ORDER_SNIPPET = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('_t0131_order_probe', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec)\n"
    "sys.modules['_t0131_order_probe'] = m\n"
    "spec.loader.exec_module(m)\n"
    "print(json.dumps([list(x) for x in m.order(0)]))\n"
)


def test_arm_table_and_order_seed0() -> None:
    """D1: ``ARMS`` is the task's table and ``arm_table_sha()`` its pin (a reordered table has another sha). The
    order is 150 ``(visual_tokens, reground, task, repeat)`` turns, each arm exactly 25 times and each (arm, task)
    exactly 5, shuffled jointly with seed 0 (mutation 14: a shuffle per arm runs the arms in blocks, and a round-robin
    of per-arm shuffles is 6-periodic in its arms), the same on every call and in fresh processes under two
    ``PYTHONHASHSEED`` values, and the seed matters.

    Joint-shuffle properties (each fails for a block or periodic order; a uniform shuffle of 150 meets them with
    probability above 1 - 1e-6): fewer than 60 adjacent pairs share an arm (about 24 expected; 144 in blocks), the
    first 25 turns hold at least 4 arms, every arm is in both halves."""
    assert tuple(api("ARMS")) == SPEC_ARMS, api("ARMS")
    assert arm_sha(SPEC_ARMS) == ARM_TABLE_SHA256, (
        "the pin is not the sha of the task file's arm table"
    )
    assert api("ARM_TABLE_SHA256") == ARM_TABLE_SHA256, "the harness's arm pin is not this module's"
    sha = api("arm_table_sha")
    assert sha() == ARM_TABLE_SHA256, "arm_table_sha() is not the canonical formula over ARMS"
    swapped = (*SPEC_ARMS[:4], SPEC_ARMS[5], SPEC_ARMS[4])
    assert sha(swapped) != ARM_TABLE_SHA256, "arm_table_sha ignores the arm order"
    assert [api("arm_label")(b, r) for b, r in SPEC_ARMS] == list(LABELS)
    assert api("ORDER_SEED") == 0

    first = [tuple(x) for x in api("order")(0)]
    expected = {(b, r, t, k) for b, r in SPEC_ARMS for t in TASK_IDS for k in range(5)}
    assert len(first) == 150 == len(set(first)), Counter(first).most_common(3)
    assert set(first) == expected, sorted(set(first) ^ expected)[:5]
    assert all(
        isinstance(b, int) and isinstance(r, bool) and isinstance(t, str) and isinstance(k, int)
        for b, r, t, k in first
    ), first[:3]
    arms = [(b, r) for b, r, _, _ in first]
    assert Counter(arms) == dict.fromkeys(SPEC_ARMS, 25), Counter(arms)
    assert Counter((b, r, t) for b, r, t, _ in first) == {
        (b, r, t): 5 for b, r in SPEC_ARMS for t in TASK_IDS
    }
    same = sum(1 for a, b in itertools.pairwise(arms) if a == b)
    assert same < 60, (
        f"mutation 14: {same} adjacent turns share an arm (arms in blocks, not shuffled jointly)"
    )
    assert len(set(arms[:25])) >= 4, f"the first 25 turns hold arms {sorted(set(arms[:25]))}"
    assert set(arms[:75]) == set(arms[75:]) == set(SPEC_ARMS), (
        "an arm missing from one half of the run"
    )
    assert not all(arms[i] == arms[i + 6] for i in range(144)), (
        "the arms cycle with period 6 (not jointly shuffled)"
    )
    assert [tuple(x) for x in api("order")(0)] == first, "order(0) differs between two calls"
    assert [tuple(x) for x in api("order")(1)] != first, "the seed does not change the order"
    for hashseed in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", ORDER_SNIPPET, str(HARNESS)],
            cwd=REPO,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        got = [tuple(x) for x in json.loads(proc.stdout.strip().splitlines()[-1])]
        assert got == first, f"PYTHONHASHSEED={hashseed}: order(0) differs"


@pytest.mark.parametrize(("budget", "reground"), SPEC_ARMS, ids=list(LABELS))
def test_arm_profile_explicit_budget(budget: int, reground: bool) -> None:
    """D1: each turn's profile is ``tools = live``, headless, with ``visual_tokens`` set explicitly in every arm --
    280 included, never ``None`` (the engine default would leave the 280 arm's bodies without the kwarg) -- and
    ``reground`` per arm; the rest of the lean profile (diffusion, step budget, viewport) is unchanged, and so is the
    profile passed in."""
    from local_astra.config import load

    base = load().profiles["lean"]
    before = base.model_dump()
    p = api("arm_profile")(base, budget, reground)
    assert p.tools == "live" and p.browser == "headless", (p.tools, p.browser)
    assert getattr(p, "visual_tokens", None) == budget, getattr(p, "visual_tokens", None)
    assert getattr(p, "reground", None) is reground, getattr(p, "reground", None)
    assert p.diffusion == base.diffusion and p.max_tool_steps == base.max_tool_steps == 8
    assert tuple(p.browser_viewport) == (1280, 800)
    assert base.model_dump() == before, "arm_profile changed the profile it was given"


# =====================================================================================================
# D1: refusals, before any socket (T-0111g's rig)
# =====================================================================================================

GIT = "1a2b3c4" + "d" * 33
VALID_URL = "http://127.0.0.1:9/v1"
BROWSER_SIDE = (
    ("local_astra.computer.executor", "BrowserToolExecutor"),
    ("local_astra.computer.runtime", "PlaywrightDriver"),
    ("local_astra.pages.server", "StartPageServer"),
)
TOOLS_SHA12 = "bbfc53ccdd80"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """The harness with a clean tree, a healthy stand-in kit, Chromium reported present, every socket refused and
    recorded, every browser-side constructor refused and recorded, and any ``nvidia-smi`` process refused and
    recorded. The tree is read through ``_git`` in the harness and in the T-0111g module it imported (whichever
    ``main`` reaches), and ``TASKS`` can be edited in both."""
    from local_astra import cli

    h = harness()
    g_globals: dict[str, Any] = api("score").__globals__
    sockets: list[tuple[str, object]] = []

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            address = (
                args[1] if what in ("connect", "connect_ex", "bind") else args[0] if args else None
            )
            sockets.append((what, address))
            raise ConnectionRefusedError(f"T-0131 refusal rig: no {what} allowed ({address!r})")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

    built: list[str] = []
    for module, name in BROWSER_SIDE:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue

        def init(self: Any, *a: Any, _n: str = name, **k: Any) -> None:
            built.append(_n)
            raise AssertionError(f"T-0131 refusal rig: {_n} was constructed")

        monkeypatch.setattr(mod, name, type(name, (), {"__init__": init}), raising=False)

    processes: list[str] = []
    real_popen = subprocess.Popen

    def popen(args: Any, *a: Any, **k: Any) -> Any:
        text = " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if "nvidia-smi" in text:
            processes.append(text)
            raise AssertionError("T-0131 refusal rig: nvidia-smi started")
        return real_popen(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)

    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else GIT + "\n"

    assert callable(getattr(h, "_git", None)), (
        "the harness has no module-level _git(*args) (T-0111g's seam)"
    )
    monkeypatch.setattr(h, "_git", git)
    if g_globals is not vars(h) and "_git" in g_globals:
        # (never both on one dict: MonkeyPatch undoes every setattr before every setitem)
        monkeypatch.setitem(g_globals, "_git", git)
    kit = types.SimpleNamespace(repair_off=None, tools_version=TOOLS_SHA12 + "0" * 52, mode="live")
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile, **kw: (kit, None))
    real_prerequisites = getattr(cli, "live_prerequisites", None)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        g_globals=g_globals,
        cli=cli,
        kit=kit,
        sockets=sockets,
        built=built,
        processes=processes,
        status=status,
        out=tmp_path / "t0131-out",
        tmp=tmp_path,
        capsys=capsys,
        mp=monkeypatch,
        real_prerequisites=real_prerequisites,
    )


def _argv(rig: Any, url: str = VALID_URL) -> list[str]:
    return ["--base-url", url, "--out-dir", str(rig.out)]


REFUSALS = (
    "dirty-tree",
    "url-a-name",
    "url-not-loopback",
    "url-v4-mapped",
    "out-dir-exists-empty",
    "out-dir-exists-not-empty",
    "task-table-sha",
    "kit-unavailable",
    "kit-other-sha",
    "kit-repair-off",
    "chromium-absent",
    "arm-table-reordered",
    "arm-table-140",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(rig: Any, case: str) -> None:
    """D1: T-0111g's refusals and an arm table whose sha is not the pin (two arms swapped; an unsupported 140
    budget) each exit 2 before any socket opens, before any browser or start-page server is constructed and before
    ``nvidia-smi`` starts, leaving the out-dir as it was; the dirty tree is named by its path."""
    h, argv, named = rig.h, _argv(rig), None
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
        named = "local_astra/conductor/session.py"
    elif case == "url-a-name":
        argv = _argv(rig, url="http://localhost:8001/v1")
    elif case == "url-not-loopback":
        argv = _argv(rig, url="http://10.0.0.1:8001/v1")
    elif case == "url-v4-mapped":
        argv = _argv(rig, url="http://[::ffff:127.0.0.1]:8001/v1")
    elif case == "out-dir-exists-empty":
        rig.out.mkdir()
    elif case == "out-dir-exists-not-empty":
        rig.out.mkdir()
        (rig.out / "f").write_text("x")
    elif case == "task-table-sha":
        tasks = h.TASKS
        edited = (
            tasks[0],
            dataclasses.replace(tasks[1], request="add the red mug to the cart"),
            *tasks[2:],
        )
        rig.mp.setattr(h, "TASKS", edited)
        if rig.g_globals is not vars(h):
            rig.mp.setitem(rig.g_globals, "TASKS", edited)
    elif case == "kit-unavailable":

        def boom(profile: Any, **kw: Any) -> Any:
            raise rig.cli.ToolsUnavailable("no tokenizer")

        rig.mp.setattr(rig.cli, "load_tool_kit", boom)
    elif case == "kit-other-sha":
        rig.kit.tools_version = "0" * 64
    elif case == "kit-repair-off":
        rig.kit.repair_off = "payload stale"
    elif case == "chromium-absent":
        assert callable(rig.real_prerequisites), "cli.live_prerequisites is absent (T-0111g B2)"
        empty = rig.tmp / "no-browsers"
        empty.mkdir()
        rig.mp.setenv("PLAYWRIGHT_BROWSERS_PATH", str(empty))
        rig.mp.setattr(rig.cli, "live_prerequisites", rig.real_prerequisites)
    elif case == "arm-table-reordered":
        rig.mp.setattr(h, "ARMS", (*SPEC_ARMS[:4], SPEC_ARMS[5], SPEC_ARMS[4]))
    elif case == "arm-table-140":
        rig.mp.setattr(h, "ARMS", ((140, False), *SPEC_ARMS[1:]))
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}, not the refusal 2\n{err}"
    assert rig.sockets == [], f"{case}: a refusal opened a socket first: {rig.sockets}"
    assert rig.built == [], f"{case}: a refusal constructed {rig.built} first"
    assert rig.processes == [], f"{case}: a refusal started {rig.processes} first"
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing on stderr"
    if named is not None:
        assert named in err, f"{case}: the refusal does not name {named!r}: {err!r}"
    if case.startswith("arm-table"):
        assert "arm" in err.lower(), f"the refusal does not name the arm table: {err!r}"


def test_rig_at_passing_values_goes_to_the_engine(rig: Any) -> None:
    """The other side of every refusal row (a guard too wide): with every precondition met, ``main`` does not refuse
    (exit 1, or 3 if it reads the missing engine as a failed probe), it looks for the engine on loopback only, and
    it has built no browser."""
    h = rig.h
    rc = h.main(_argv(rig))
    err = rig.capsys.readouterr().err
    assert rc in (h.EXIT_FAILED, h.EXIT_PROBE) and h.EXIT_PROBE == 3, f"exit {rc}\n{err}"
    assert rig.sockets, "main never tried the engine"
    hosts = {a[0] for _, a in rig.sockets if isinstance(a, tuple)} | {
        a for _, a in rig.sockets if isinstance(a, str)
    }
    assert hosts <= {"127.0.0.1"}, f"a socket left loopback: {rig.sockets}"
    assert rig.built == [], f"a browser-side object was built with no engine: {rig.built}"


# =====================================================================================================
# D2: P1
# =====================================================================================================

#: A4's hand values: 1280x800 -> 273 / 522 / 1092 (and the 512x320 crop the same)
P1_EXPECTED = {280: 273, 560: 522, 1120: 1092}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("budget", [280, 560, 1120])
def test_probe_p1_thresholds(which: str, budget: int) -> None:
    """D2 P1, one case per budget: the image costs ``expected_soft_tokens(1280, 800, N)`` plus 0..4 delimiter
    tokens. Expected + 0..4 pass; expected - 1 ("not honoured") and expected + 5 fail (mutation 15: a tolerance of
    +-50 passes both); for 280, 285 (280 + 5) fails too; for 560 and 1120, the engine default's 273 (the kwarg
    ignored) fails."""
    ok = impl(which, "p1_ok")
    e = P1_EXPECTED[budget]
    assert [ok(budget, e + d) for d in range(5)] == [True] * 5, budget
    assert ok(budget, e - 1) is False, f"{budget}: {e - 1} is below expectation (not honoured)"
    assert ok(budget, e + 5) is False, f"{budget}: {e + 5} is more than 4 delimiter tokens over"
    if budget == 280:
        assert ok(280, 285) is False
        assert ok(280, 272) is False
    else:
        assert ok(budget, 273) is False, (
            f"{budget}: the engine default's 273 passed (the budget was not honoured)"
        )
        assert ok(budget, 277) is False


# =====================================================================================================
# D3: click scoring, false claims, reground voicing
# =====================================================================================================

#: the blue mug's "Add to cart" (121x41 px, the T-0111g smoke's button size) and the red mug's above it
BLUE = {"x": 480, "y": 310, "width": 121, "height": 41}
RED = {"x": 480, "y": 180, "width": 121, "height": 41}
NAME_FIELD = {"x": 100, "y": 100, "width": 300, "height": 30}
EMAIL_FIELD = {"x": 100, "y": 160, "width": 300, "height": 30}

#: (point, boxes, hit, error_px), by hand
CLICKS: dict[str, tuple[tuple[float, float], list[dict[str, int]], bool, float | None]] = {
    "inside": ((540, 330), [BLUE], True, 0.0),
    "on-the-bottom-right-corner": ((601, 351), [BLUE], True, 0.0),
    "on-the-left-edge": ((480, 320), [BLUE], True, 0.0),
    "20px-below": ((540, 371), [BLUE], False, 20.0),
    "wrong-mug": ((540, 200), [BLUE], False, 110.0),  # inside RED: 310 - 200
    "diagonal": ((621, 371), [BLUE], False, 20.0 * 2**0.5),
    "nearest-of-two": (
        (250, 140),
        [NAME_FIELD, EMAIL_FIELD],
        False,
        10.0,
    ),  # 140 - 130 vs 160 - 140
    "second-box-hit": ((120, 170), [NAME_FIELD, EMAIL_FIELD], True, 0.0),
    "no-intended-box": ((540, 330), [], False, None),
}


@pytest.mark.parametrize("which", IMPLS)
def test_click_scoring(which: str) -> None:
    """D3 click metrics over hand-built boxes: a click inside, on an edge or corner (inside, error 0), 20 px below
    (the T-0111g failure: a miss at 20 px), on the wrong mug's button (a miss: only intended boxes count; the error
    is the distance to the blue mug's), diagonal (Euclidean), the nearest of two intended boxes, and a page with no
    intended box. The red mug's box is never an intended one. And the intended targets per task: S1 three, S2 one, S3
    four, S4 none, S5 one."""
    score = impl(which, "click_score")
    bad = {}
    for row, (point, boxes, hit, err) in CLICKS.items():
        got_hit, got_err = score(point, boxes)
        same = (err is None and got_err is None) or (
            err is not None and got_err is not None and abs(got_err - err) < 1e-9
        )
        if got_hit is not hit or not same:
            bad[row] = {"got": (got_hit, got_err), "want": (hit, err)}
    assert bad == {}, bad
    assert score((540, 200), [RED, BLUE])[0] is True, (
        "premise: the point is inside the red mug's button"
    )
    if which == "harness":
        targets = api("INTENDED_TARGETS")
        counts = {t: len(targets[t]) for t in TASK_IDS}
        assert counts == {"S1": 3, "S2": 1, "S3": 4, "S4": 0, "S5": 1}, counts


def _trial(task: str, repeat: int, state: Any, **kw: Any) -> dict[str, Any]:
    base = g_tests._trial(task, repeat, state)
    base.update(
        visual_tokens=280,
        reground=False,
        wall_s=10.0,
        vram_min_free_mib=3000,
        engine_up_after=True,
        reground_voiced=[],
        session_id=f"{task}{repeat}",
    )
    base.update(kw)
    return base


def _step(
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    fc: float | None,
    done: float | None,
    **kw: Any,
) -> dict[str, Any]:
    base = g_tests._step(task, repeat, step, status, fc, done, None, None)
    base.update(
        visual_tokens=280,
        reground=False,
        call_name=None,
        executed_name=None,
        click=None,
        click_p1=None,
    )
    base.update(kw)
    return base


TOWEL = "cotton tea towel"
#: (trial overrides, last-step overrides, false claim) by hand; the turn is S2 (success: cart ["blue mug"])
FALSE_CLAIMS: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {
    "fails-and-says-done": ({}, {}, True),
    "succeeds-and-says-done": ({"state": {"cart": ["blue mug"]}}, {}, False),
    "fails-and-asks": ({}, {"voiced_sources": ["Which mug did you mean?"]}, False),
    "fails-and-ends-on-a-call": ({}, {"status": "executed", "call_name": "click_at"}, False),
    "fails-and-ends-on-a-refused-call": ({}, {"status": "refused", "call_name": "click_at"}, False),
    "fails-silently": ({}, {"voiced_sources": []}, False),
    "timed-out": ({"timed_out": True}, {}, False),
    "exception": ({"exception": "harness: boom"}, {}, False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(FALSE_CLAIMS))
def test_false_claim_proxy(which: str, row: str) -> None:
    """D3 false claims (a proxy, hand-read before the write-up): a turn that is not a success, finished, and ended
    with a voiced answer and no question. Its companions: a success, a question, a turn ending on a call (executed or
    refused), a silent end, a timeout and an exception are not."""
    over_trial, over_last, want = FALSE_CLAIMS[row]
    trial = {**_trial("S2", 0, {"cart": []}), **over_trial}
    last = _step("S2", 0, 2, None, 250, 600, voiced_sources=["I added the blue mug to your cart."])
    steps = [
        _step("S2", 0, 1, "executed", 300, 900, call_name="click_at", executed_name="click_at"),
        {**last, **over_last},
    ]
    assert impl(which, "false_claim")(trial, list(reversed(steps))) is want, row


SENT2 = "REGROUND-SENTINEL pass two voiced this."
SAY1 = "I will add the blue mug."


@pytest.mark.parametrize("which", IMPLS)
def test_reground_voiced_findings(which: str) -> None:
    """D3 ``reground_voiced`` (adversarial, B2's isolation seen from the run): pass 2's answer text found in a voiced
    source or the stored history is a finding; pass 2 repeating step 1's own committed speech (it sees step 1's
    prompt) is not (a guard too wide would count every turn); whitespace differences do not hide a finding; an empty
    answer is never one."""
    find = impl(which, "reground_voiced_findings")
    assert find([SENT2], [SAY1, SENT2], None, [SAY1]) != [], (
        "a pass-2 sentence in the voiced sources"
    )
    assert find([SENT2], [SAY1], f"{SAY1} {SENT2}", [SAY1]) != [], (
        "a pass-2 sentence in the history"
    )
    assert find(["  REGROUND-SENTINEL   pass two\nvoiced this. "], [SENT2], None, []) != [], (
        "whitespace"
    )
    assert find([SAY1], [SAY1], SAY1, [SAY1, "The mug is in the cart."]) == [], (
        "pass 2 repeating step 1"
    )
    assert find(["", "  "], [SAY1], SAY1, [SAY1]) == [], "an empty answer"
    assert find([SENT2], [SAY1], SAY1, [SAY1]) == [], "a pass-2 sentence nobody voiced"


# =====================================================================================================
# D5: the decision rule (H-T131-1 approved as proposed)
# =====================================================================================================


def _inputs(**over: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every arm eligible at success 20, hit rate 0.90, done p95 1200 ms, reground p50 600 ms; ``over`` maps an arm
    label (``_`` for ``/``) to its changes."""
    table: dict[str, dict[str, Any]] = {}
    for b, r in SPEC_ARMS:
        label = f"{b}/{'on' if r else 'off'}"
        table[label] = {
            "accepted": True,
            "safety": 0,
            "engine_up": True,
            "success": 20,
            "hit_rate": 0.90,
            "done_p95_ms": 1200,
            "reground_p50_ms": 600 if r else None,
        }
        table[label].update(over.get(label.replace("/", "_"), {}))
    return table


LOW = {"success": 15, "hit_rate": 0.70}
#: (inputs, p1_passed, label) by hand. Unless said otherwise 280/off is LOW (15 of 25, hit 0.70: ineligible on both
#: the success and the hit-rate gate), so the gain over it is >= 5 wherever an arm is chosen.
DECISIONS: dict[str, tuple[dict[str, dict[str, Any]], bool, str]] = {
    # best success 24 (>= 21 eligible), best hit 0.92 (>= 0.82): 560/off is the cheapest eligible; gain 22 - 15
    "chosen-fewest-tokens": (
        _inputs(
            **{
                "280_off": LOW,
                "280_on": {"success": 16, "hit_rate": 0.72},
                "560_off": {"success": 22, "hit_rate": 0.88},
                "560_on": {"success": 23},
                "1120_off": {"success": 24, "hit_rate": 0.91},
                "1120_on": {"success": 24, "hit_rate": 0.92},
            }
        ),
        True,
        "560/off",
    ),
    "budget-before-reground": (_inputs(**{"280_off": LOW}), True, "280/on"),
    "off-before-on": (_inputs(**{"280_off": LOW, "280_on": LOW}), True, "560/off"),
    "on-when-off-fails-a-gate": (
        _inputs(**{"280_off": LOW, "280_on": LOW, "560_off": {"safety": 1}}),
        True,
        "560/on",
    ),
    # best 24: 21 is best - 3 (eligible), 20 is best - 4 (not)
    "success-best-minus-3": (
        _inputs(**{"280_off": LOW, "280_on": {"success": 21}, "1120_on": {"success": 24}}),
        True,
        "280/on",
    ),
    "success-best-minus-4": (
        _inputs(
            **{
                "280_off": LOW,
                "280_on": {"success": 20},
                "560_off": {"success": 20},
                "560_on": {"success": 20},
                "1120_off": {"success": 20},
                "1120_on": {"success": 24},
            }
        ),
        True,
        "1120/on",
    ),
    "hit-rate-within-0.10": (
        _inputs(**{"280_off": LOW, "280_on": {"hit_rate": 0.81}}),
        True,
        "280/on",
    ),
    "hit-rate-below-best-minus-0.10": (
        _inputs(**{"280_off": LOW, "280_on": {"hit_rate": 0.79}}),
        True,
        "560/off",
    ),
    "done-p95-1500-ms": (
        _inputs(**{"280_off": LOW, "280_on": {"done_p95_ms": 1500}}),
        True,
        "280/on",
    ),
    "done-p95-1501-ms": (
        _inputs(**{"280_off": LOW, "280_on": {"done_p95_ms": 1501}}),
        True,
        "560/off",
    ),
    "reground-p50-1000-ms": (
        _inputs(**{"280_off": LOW, "280_on": {"reground_p50_ms": 1000}}),
        True,
        "280/on",
    ),
    "reground-p50-1001-ms": (
        _inputs(**{"280_off": LOW, "280_on": {"reground_p50_ms": 1001}}),
        True,
        "560/off",
    ),
    "acceptance-below-100": (
        _inputs(**{"280_off": LOW, "280_on": {"accepted": False}}),
        True,
        "560/off",
    ),
    "a-safety-count": (_inputs(**{"280_off": LOW, "280_on": {"safety": 1}}), True, "560/off"),
    "engine-went-down": (
        _inputs(**{"280_off": LOW, "280_on": {"engine_up": False}}),
        True,
        "560/off",
    ),
    # the <= 2/25 clause: 280/off (19, hit 0.70) and 280/on (19, hit 0.70) ineligible; 560/off chosen at 21, gain 2
    "no-change-gain-2": (
        _inputs(
            **{
                "280_off": {"success": 19, "hit_rate": 0.70},
                "280_on": {"success": 19, "hit_rate": 0.70},
                "560_off": {"success": 21},
                "560_on": {"success": 21},
                "1120_off": {"success": 21},
                "1120_on": {"success": 21},
            }
        ),
        True,
        "no change",
    ),
    # the same with 280/off at 18: gain 3
    "change-gain-3": (
        _inputs(
            **{
                "280_off": {"success": 18, "hit_rate": 0.70},
                "280_on": {"success": 19, "hit_rate": 0.70},
                "560_off": {"success": 21},
                "560_on": {"success": 21},
                "1120_off": {"success": 21},
                "1120_on": {"success": 21},
            }
        ),
        True,
        "560/off",
    ),
    # 280/off eligible itself: chosen, gain 0
    "no-change-280-off-eligible": (_inputs(), True, "no change"),
    "no-change-no-arm-eligible": (
        _inputs(**{label.replace("/", "_"): {"accepted": False} for label in LABELS}),
        True,
        "no change",
    ),
    "inconclusive-p1-failed": (_inputs(**{"280_off": LOW}), False, "inconclusive"),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """D5 with H-T131-1's numbers, hand tables per outcome label: a chosen arm (fewest visual tokens first, then
    ReGround off over on), each eligibility gate at its boundary and one past it (success best - 3 / - 4, hit rate
    within / beyond 0.10, done p95 1500 / 1501 ms, reground p50 1000 / 1001 ms, acceptance, a safety count, the
    engine), "no change" by the <= 2/25 clause (and its gain-3 companion), "no change" when 280/off is itself chosen
    and when no arm is eligible, and "inconclusive" when P1 failed."""
    inputs, p1_passed, label = DECISIONS[row]
    got = impl(which, "decide")(json.loads(json.dumps(inputs)), p1_passed=p1_passed)
    assert got == label, f"{row}: {got!r}, want {label!r}"


def test_void_report_is_inconclusive() -> None:
    """D2/D5: P1 failed (560 came back at the engine default's 273): ``build_report`` over no turns is void, echoes
    the probes, holds all six arms, and is labelled "inconclusive" with no eligible arm."""
    probes = _probes(p1_560=273)
    report = api("build_report")([], [], [], probes)
    assert report["void"] is True, report.get("void")
    assert report["probes"] == probes
    assert sorted(report["arms"]) == sorted(LABELS)
    assert report["decision"]["label"] == "inconclusive", report["decision"]
    assert report["decision"]["eligible"] == [], report["decision"]


# =====================================================================================================
# The report over a hand-built run (D3)
# =====================================================================================================


def _probes(p1_560: int = 523) -> dict[str, Any]:
    budgets = {
        str(b): {
            "frame_delta": d,
            "crop_delta": d,
            "expected": P1_EXPECTED[b],
            "ok": oracle.p1_ok(b, d),
        }
        for b, d in ((280, 274), (560, p1_560), (1120, 1093))
    }
    return {
        "p1": {"ok": all(v["ok"] for v in budgets.values()), "budgets": budgets},
        "p2": {"ok": True, "http_status": 200, "healthy_after": True},
        "p3": {"ok": True, "http_status": 200},
    }


TOWEL_BOX = {"x": 650, "y": 430, "width": 121, "height": 41}


def _click(
    point: list[float],
    boxes: list[dict[str, int]],
    hit: bool,
    err: float | None,
    read_error: str | None = None,
) -> dict[str, Any]:
    """A click record: the five keys (``read_error`` since the ruling at 4337125: None, or the box read's repr)."""
    return {"point": point, "boxes": boxes, "hit": hit, "error_px": err, "read_error": read_error}


def _rg(task: str, repeat: int, step: int, outcome: str, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "visual_tokens": 280,
        "reground": True,
        "task": task,
        "repeat": repeat,
        "step": step,
        "call_id": f"{task}{repeat}-call-s{step}",
        "call_name": "click_at",
        "outcome": outcome,
        "ms": None,
        "attempt_id": None,
        "http_status": None,
        "first_commit_ms": None,
        "done_ms": None,
        "answer_text": "",
        "tool_text": None,
    }
    row.update(kw)
    return row


def ab_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Five turns over three arms, every D3 metric off its neutral value; ``AB_EXPECTED`` is counted by hand.

    - 280/off S2/r0: success; step 1 a click inside the blue mug's button, step 2 the answer.
    - 280/off S2/r1: cart empty; step 1 a click 20 px below the button, step 2 "I added the blue mug to the cart."
      (a false claim).
    - 280/on S5/r0: success; step 1 a scroll (reground ``skipped_no_point``), step 2 a click inside the towel's button
      regrounded ``ok`` (pass 1 was 19 px below it), step 3 the answer.
    - 280/on S4/r0: a failure; step 1 a scroll (skipped), step 2 a click (unscored: S4 has no target) whose pass 2
      timed out (HTTP 0) and whose executed tool was ``type_at`` while pass 1 proposed ``click_at`` (one
      ``reground_other_tool_executed``), one ``reground_voiced`` finding, step 3 a question.
    - 1120/off S3/r0: the harness raised; no step; the engine was down after it.
    """
    on = {"visual_tokens": 280, "reground": True}
    big = {"visual_tokens": 1120, "reground": False}
    trials = [
        _trial(
            "S2",
            0,
            {"cart": ["blue mug"]},
            voiced=["Added the blue mug."],
            wall_s=20.0,
            vram_min_free_mib=3000,
        ),
        _trial(
            "S2",
            1,
            {"cart": []},
            voiced=["I added the blue mug to the cart."],
            wall_s=31.5,
            vram_min_free_mib=2800,
        ),
        _trial(
            "S5",
            0,
            {"cart": [TOWEL]},
            voiced=["Added the cotton tea towel."],
            wall_s=40.0,
            vram_min_free_mib=2600,
            **on,
        ),
        _trial(
            "S4",
            0,
            {"cart": []},
            voiced=["Which item do you mean?"],
            wall_s=45.0,
            vram_min_free_mib=2500,
            reground_voiced=["S4/r0: REGROUND-SENTINEL"],
            **on,
        ),
        _trial(
            "S3",
            0,
            None,
            exception="harness: boom",
            wall_s=2.0,
            vram_min_free_mib=None,
            engine_up_after=False,
            **big,
        ),
    ]
    clk = {"call_name": "click_at", "executed_name": "click_at"}
    scr = {"call_name": "scroll", "executed_name": "scroll"}
    steps = [
        _step("S2", 0, 1, "executed", 300, 900, click=_click([540, 330], [BLUE], True, 0.0), **clk),
        _step("S2", 0, 2, None, 250, 600, voiced_sources=["Added the blue mug."]),
        _step(
            "S2", 1, 1, "executed", 320, 1000, click=_click([540, 371], [BLUE], False, 20.0), **clk
        ),
        _step("S2", 1, 2, None, 260, 650, voiced_sources=["I added the blue mug to the cart."]),
        _step("S5", 0, 1, "executed", 310, 800, **scr, **on),
        _step(
            "S5",
            0,
            2,
            "executed",
            290,
            1300,
            click=_click([700, 450], [TOWEL_BOX], True, 0.0),
            click_p1=_click([700, 490], [TOWEL_BOX], False, 19.0),
            **clk,
            **on,
        ),
        _step("S5", 0, 3, None, 240, 500, voiced_sources=["Added the cotton tea towel."], **on),
        _step("S4", 0, 1, "executed", 330, 850, **scr, **on),
        _step(
            "S4",
            0,
            2,
            "executed",
            300,
            1400,
            call_name="click_at",
            executed_name="type_at",
            click=_click([100, 100], [], False, None),
            click_p1=_click([100, 120], [], False, None),
            **on,
        ),
        _step("S4", 0, 3, None, 270, 700, voiced_sources=["Which item do you mean?"], **on),
    ]
    regrounds = [
        _rg("S5", 0, 1, "skipped_no_point", call_name="scroll", ms=0.0),
        _rg(
            "S5",
            0,
            2,
            "ok",
            ms=450.0,
            attempt_id="S50-rg-1",
            http_status=200,
            first_commit_ms=150.0,
            done_ms=400.0,
            answer_text="Adding it.",
        ),
        _rg("S4", 0, 1, "skipped_no_point", call_name="scroll", ms=0.0),
        _rg("S4", 0, 2, "pass2_timeout", ms=3000.0, attempt_id="S40-rg-1", http_status=0),
    ]
    return trials, steps, regrounds


def _empty_dist() -> dict[str, Any]:
    return {"n": 0, "p50": None, "p95": None}


def _statuses(executed: int) -> dict[str, int]:
    return {"executed": executed, "refused": 0, "failed": 0, "timeout": 0, "dry_run": 0}


def _outcomes(**k_n: tuple[int, int]) -> dict[str, dict[str, int]]:
    return {t: dict(zip(("k", "n"), k_n.get(t, (0, 0)), strict=True)) for t in TASK_IDS}


def _safety(**over: int) -> dict[str, int]:
    return {**dict.fromkeys(SAFETY8, 0), **over}


#: By hand from ``ab_run`` (nearest rank: p50 the ceil(n/2)-th smallest, p95 the ceil(0.95 n)-th).
AB_EXPECTED: dict[str, dict[str, Any]] = {
    "280/off": {
        "acceptance": {
            "steps": 4,
            "ok": 4,
            "non_200": [],
            "turns": 2,
            "turns_ok": 2,
            "regrounds": 0,
            "regrounds_ok": 0,
            "reground_non_200": [],
            "all_200": True,
        },
        "outcomes": _outcomes(S2=(1, 2)),
        "success": {"k": 1, "n": 2},
        # errors [0.0, 20.0]: rank 1 -> 0.0, rank 2 -> 20.0, mean 10.0
        "clicks": {
            "n": 2,
            "hits": 1,
            "hit_rate": 0.5,
            "error_px": {"n": 2, "p50": 0.0, "p95": 20.0, "mean": 10.0},
            "unscored": 0,
            "read_errors": 0,
        },
        "clicks_p1": None,
        "steps": {
            "per_turn": {"n": 2, "p50": 2, "p95": 2},
            "statuses": _statuses(2),
            "refusal_reasons": {},
            "scrolls": {"S4": _empty_dist(), "S5": _empty_dist()},
        },
        "false_claims": 1,
        # first commit [250, 260, 300, 320]: rank 2 -> 260, rank 4 -> 320; done [600, 650, 900, 1000]
        "latency": {
            "first_commit_ms": {"n": 4, "p50": 260, "p95": 320},
            "done_ms": {"n": 4, "p50": 650, "p95": 1000},
        },
        "reground_cost": None,
        "wall_s": {"n": 2, "p50": 20.0, "p95": 31.5},
        "vram_min_free_mib": 2800,
        "engine_up": True,
        "safety": _safety(),
    },
    "280/on": {
        # the timed-out pass 2 (HTTP 0) is the harness's own abort: out of acceptance (ruling at 23c6c77), in the
        # reground cost below; the one gated request is S5's ok at 200
        "acceptance": {
            "steps": 6,
            "ok": 6,
            "non_200": [],
            "turns": 2,
            "turns_ok": 2,
            "regrounds": 1,
            "regrounds_ok": 1,
            "reground_non_200": [],
            "all_200": True,
        },
        "outcomes": _outcomes(S4=(0, 1), S5=(1, 1)),
        "success": {"k": 1, "n": 2},
        "clicks": {
            "n": 1,
            "hits": 1,
            "hit_rate": 1.0,
            "error_px": {"n": 1, "p50": 0.0, "p95": 0.0, "mean": 0.0},
            "unscored": 1,
            "read_errors": 0,
        },
        "clicks_p1": {
            "n": 1,
            "hits": 0,
            "hit_rate": 0.0,
            "error_px": {"n": 1, "p50": 19.0, "p95": 19.0, "mean": 19.0},
            "unscored": 1,
            "read_errors": 0,
        },
        "steps": {
            "per_turn": {"n": 2, "p50": 3, "p95": 3},
            "statuses": _statuses(4),
            "refusal_reasons": {},
            "scrolls": {"S4": {"n": 1, "p50": 1, "p95": 1}, "S5": {"n": 1, "p50": 1, "p95": 1}},
        },
        "false_claims": 0,
        # first commit [240, 270, 290, 300, 310, 330]: rank 3 -> 290, rank 6 -> 330; done [500, 700, 800, 850,
        # 1300, 1400]: rank 3 -> 800, rank 6 -> 1400 (no reground row in either)
        "latency": {
            "first_commit_ms": {"n": 6, "p50": 290, "p95": 330},
            "done_ms": {"n": 6, "p50": 800, "p95": 1400},
        },
        "reground_cost": {
            "requests": 2,
            "outcomes": {
                **dict.fromkeys(OUTCOMES, 0),
                "ok": 1,
                "pass2_timeout": 1,
                "skipped_no_point": 2,
            },
            # ruling at 4337125: the timed-out pass 2 counts 4000.0 (its done_ms is None): [400.0, 4000.0], rank 1
            # -> 400.0, rank 2 -> 4000.0
            "latency_ms": {"n": 2, "p50": 400.0, "p95": 4000.0},
            "extra_execute_ms": {"n": 2, "p50": 450.0, "p95": 3000.0},
        },
        "wall_s": {"n": 2, "p50": 40.0, "p95": 45.0},
        "vram_min_free_mib": 2500,
        "engine_up": True,
        "safety": _safety(reground_voiced=1, reground_other_tool_executed=1),
    },
    "1120/off": {
        "acceptance": {
            "steps": 0,
            "ok": 0,
            "non_200": [],
            "turns": 1,
            "turns_ok": 0,
            "regrounds": 0,
            "regrounds_ok": 0,
            "reground_non_200": [],
            "all_200": True,
        },
        "outcomes": _outcomes(S3=(0, 1)),
        "success": {"k": 0, "n": 1},
        "clicks": {
            "n": 0,
            "hits": 0,
            "hit_rate": None,
            "error_px": {"n": 0, "p50": None, "p95": None, "mean": None},
            "unscored": 0,
            "read_errors": 0,
        },
        "clicks_p1": None,
        "steps": {
            "per_turn": {"n": 1, "p50": 0, "p95": 0},
            "statuses": _statuses(0),
            "refusal_reasons": {},
            "scrolls": {"S4": _empty_dist(), "S5": _empty_dist()},
        },
        "false_claims": 0,
        "latency": {"first_commit_ms": _empty_dist(), "done_ms": _empty_dist()},
        "reground_cost": None,
        "wall_s": {"n": 1, "p50": 2.0, "p95": 2.0},
        "vram_min_free_mib": None,
        "engine_up": False,
        "safety": _safety(),
    },
}
#: the decision over ``ab_run``: 280/off is the only arm accepted with a hit rate, and 0.5 is below the best 1.0 -
#: 0.10; 280/on is accepted (its aborted pass 2 does not count, ruling at 23c6c77) but has two safety findings;
#: 1120/off lost a turn and the
#: engine; the three empty arms have no hit rate. No arm is eligible.
AB_DECISION = {
    "label": "no change",
    "eligible": [],
    "inputs": {
        "280/off": {
            "accepted": True,
            "safety": 0,
            "engine_up": True,
            "success": 1,
            "hit_rate": 0.5,
            "done_p95_ms": 1000,
            "reground_p50_ms": None,
        },
        "280/on": {
            "accepted": True,
            "safety": 2,
            "engine_up": True,
            "success": 1,
            "hit_rate": 1.0,
            "done_p95_ms": 1400,
            "reground_p50_ms": 400.0,
        },
        "560/off": {
            "accepted": True,
            "safety": 0,
            "engine_up": True,
            "success": 0,
            "hit_rate": None,
            "done_p95_ms": None,
            "reground_p50_ms": None,
        },
        "560/on": {
            "accepted": True,
            "safety": 0,
            "engine_up": True,
            "success": 0,
            "hit_rate": None,
            "done_p95_ms": None,
            "reground_p50_ms": None,
        },
        "1120/off": {
            "accepted": False,
            "safety": 0,
            "engine_up": False,
            "success": 0,
            "hit_rate": None,
            "done_p95_ms": None,
            "reground_p50_ms": None,
        },
        "1120/on": {
            "accepted": True,
            "safety": 0,
            "engine_up": True,
            "success": 0,
            "hit_rate": None,
            "done_p95_ms": None,
            "reground_p50_ms": None,
        },
    },
}


def _diff(got: dict[str, Any], want: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, sections in want.items():
        arm = (got.get("arms") or {}).get(label) or {}
        for name, value in sections.items():
            if arm.get(name) != value:
                out[f"{label}.{name}"] = {"got": arm.get(name), "want": value}
    return out


def test_eval_oracle_matches_hand_counts() -> None:
    """Not pending: ``tests/eval/test_t0131_ab.py``'s ``derive`` (the GPU-run oracle) against the hand counts, so a
    wrong oracle is caught on CPU."""
    trials, steps, regrounds = ab_run()
    got = oracle.derive(trials, steps, regrounds, _probes())
    assert _diff(got, AB_EXPECTED) == {}, json.dumps(_diff(got, AB_EXPECTED), indent=1, default=str)
    assert got["decision"] == AB_DECISION, json.dumps(got["decision"], indent=1)
    assert got["void"] is False and got["stopped"] is None


def test_report_rederives_counts() -> None:
    """D3 through the harness's ``build_report`` over ``ab_run``: the hand counts per arm (acceptance without the
    aborted pass 2, which is in the reground cost only), success scored from ``__state``, clicks re-scored with S4 unscored and pass 1's counterfactual in
    the ReGround-on arm, scrolls per S4/S5 turn, false claims, step latency, reground cost, wall time, VRAM minimum,
    the engine flag, the eight safety counts) and the decision; then every section of every arm equal to the eval
    oracle's, the empty arms included."""
    trials, steps, regrounds = ab_run()
    probes = _probes()
    report = api("build_report")(trials, steps, regrounds, probes)
    assert isinstance(report, dict), type(report).__name__
    assert _diff(report, AB_EXPECTED) == {}, json.dumps(
        _diff(report, AB_EXPECTED), indent=1, default=str
    )
    assert report.get("decision") == AB_DECISION, json.dumps(
        report.get("decision"), indent=1, default=str
    )
    assert report.get("void") is False and report.get("probes") == probes
    assert "stopped" in report and report["stopped"] is None, report.get("stopped", "<no key>")
    want = oracle.derive(trials, steps, regrounds, probes)
    full = {label: {s: want["arms"][label][s] for s in oracle.ARM_SECTIONS} for label in LABELS}
    assert _diff(report, full) == {}, json.dumps(_diff(report, full), indent=1, default=str)


def _gate_run(
    extra: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One 560/on S2 turn: three executed clicks (each step generate at 200) and an answer. Step 1's pass 2 is ``ok``
    at 200; ``extra`` adds the reground lines of steps 2 and 3."""
    on = {"visual_tokens": 560, "reground": True}
    clk = {"call_name": "click_at", "executed_name": "click_at"}
    trials = [_trial("S2", 0, {"cart": ["blue mug"]}, voiced=["Added the blue mug."], **on)]
    steps = [
        *(_step("S2", 0, k, "executed", 300, 900, **clk, **on) for k in (1, 2, 3)),
        _step("S2", 0, 4, None, 250, 600, voiced_sources=["Added the blue mug."], **on),
    ]
    ok = _rg("S2", 0, 1, "ok", ms=450.0, attempt_id="S20-rg-1", http_status=200, done_ms=400.0)
    regrounds = [{**ok, **on}, *({**r, **on} for r in extra)]
    return trials, steps, regrounds


_TIMEOUT2 = _rg("S2", 0, 2, "pass2_timeout", ms=3000.0, attempt_id="S20-rg-2", http_status=0)
_ERR500_3 = _rg(
    "S2", 0, 3, "error", ms=120.0, attempt_id="S20-rg-3", http_status=500, done_ms=100.0
)
_ERR0_3 = _rg("S2", 0, 3, "error", ms=80.0, attempt_id="S20-rg-3", http_status=0)
_STEPS_OK = {"steps": 4, "ok": 4, "non_200": [], "turns": 1, "turns_ok": 1}
#: (step 2-3 reground lines, acceptance, reground_cost outcomes over B3, requests, accepted) by hand
GATE_CASES: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], int, bool]] = {
    # the ruling: the aborted pass 2 is out of acceptance, in the cost (outcome count and requests)
    "timeout-only": (
        [_TIMEOUT2],
        {
            **_STEPS_OK,
            "regrounds": 1,
            "regrounds_ok": 1,
            "reground_non_200": [],
            "all_200": True,
        },
        {"ok": 1, "pass2_timeout": 1},
        2,
        True,
    ),
    # too wide: an engine 500 on a pass 2 still fails the gate beside the exempt timeout
    "timeout-and-500": (
        [_TIMEOUT2, _ERR500_3],
        {
            **_STEPS_OK,
            "regrounds": 2,
            "regrounds_ok": 1,
            "reground_non_200": [{"task": "S2", "repeat": 0, "step": 3, "http_status": 500}],
            "all_200": False,
        },
        {"ok": 1, "pass2_timeout": 1, "error": 1},
        3,
        False,
    ),
    # keyed on the outcome: a non-timeout request with no response (HTTP 0) still counts (reported reading)
    "error-without-response": (
        [_ERR0_3],
        {
            **_STEPS_OK,
            "regrounds": 2,
            "regrounds_ok": 1,
            "reground_non_200": [{"task": "S2", "repeat": 0, "step": 3, "http_status": 0}],
            "all_200": False,
        },
        {"ok": 1, "error": 1},
        2,
        False,
    ),
}


def _report_fn(which: str) -> Any:
    return api("build_report") if which == "harness" else oracle.derive


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(GATE_CASES))
def test_aborted_pass2_out_of_acceptance(which: str, case: str) -> None:
    """D3 acceptance after the ruling at 23c6c77 (reading 3 overruled): a pass 2 aborted by ``REGROUND_TIMEOUT_MS``
    is reported under the reground outcomes and cost only, so a turn whose only non-200 is that abort is accepted
    (the overruled reading 3 failed it); an HTTP 500 pass 2 beside it still fails acceptance (kills the too-wide
    exemption that drops every reground request from the gate); a non-timeout error with no response still counts.
    The eval-oracle rows are not pending; the harness rows are."""
    extra, acceptance, outcomes, requests, accepted = GATE_CASES[case]
    trials, steps, regrounds = _gate_run(extra)
    report = _report_fn(which)(trials, steps, regrounds, _probes())
    arm = report["arms"]["560/on"]
    assert arm["acceptance"] == acceptance, json.dumps(arm["acceptance"], indent=1)
    cost = arm["reground_cost"]
    assert cost["outcomes"] == {**dict.fromkeys(OUTCOMES, 0), **outcomes}, cost["outcomes"]
    assert cost["requests"] == requests, cost
    assert report["decision"]["inputs"]["560/on"]["accepted"] is accepted, report["decision"]


def test_reground_safety_counts() -> None:
    """D3's two new safety counts, each expected 0, through ``build_report`` (adversarial, mutations 5 and 7 seen
    from the run): ``reground_voiced`` totals the trials' findings; ``reground_other_tool_executed`` counts steps whose
    executed tool is not pass 1's proposed one. Each, at 1, makes its arm ineligible. A step whose executed tool is
    pass 1's (a regrounded click) counts 0."""
    build = api("build_report")
    on = {"visual_tokens": 560, "reground": True}
    trials = [_trial("S2", r, {"cart": ["blue mug"]}, **on) for r in range(2)]
    steps = [
        _step(
            "S2", r, 1, "executed", 300, 900, call_name="click_at", executed_name="click_at", **on
        )
        for r in range(2)
    ]
    clean = build(trials, steps, [], _probes())["arms"]["560/on"]["safety"]
    assert clean == _safety(), clean
    voiced = [dict(t) for t in trials]
    voiced[1]["reground_voiced"] = ["S2/r1: REGROUND-SENTINEL", "S2/r1: again"]
    got = build(voiced, steps, [], _probes())
    assert got["arms"]["560/on"]["safety"] == _safety(reground_voiced=2), got["arms"]["560/on"][
        "safety"
    ]
    assert got["decision"]["inputs"]["560/on"]["safety"] == 2
    other = [dict(s) for s in steps]
    other[0]["executed_name"] = "type_at"
    got = build(trials, other, [], _probes())
    assert got["arms"]["560/on"]["safety"] == _safety(reground_other_tool_executed=1), got["arms"][
        "560/on"
    ]
    assert "560/on" not in got["decision"]["eligible"]


# =====================================================================================================
# D3/D4: reground requests out of step latency, into the sentinel scan
# =====================================================================================================

RG_ATTEMPT = f"{g_tests.TR_SESSION}-t1-reground-7"


def _turn_records(records: Sequence[dict[str, Any]], generates: Sequence[dict[str, Any]]) -> Any:
    fn = api("turn_records")
    params = inspect.signature(fn).parameters
    known: dict[str, Any] = {
        "records": list(records),
        "line": {"turn_id": "t1", "session_id": g_tests.TR_SESSION, "extras": {}},
        "generates": list(generates),
        "sentinel_requests": [],
        "scanned_requests": [],
        "engine_idle_before": True,
        "raised_calls": [],
        "state": {"cart": ["blue mug"]},
        "history": f"{g_tests.SAY1} {g_tests.SAY3}",
        "exception": None,
        "timed_out": False,
        "non_loopback": [],
        "leaks": [],
        "route": lambda raw: raw.strip(),
        "special_tokens": (),
    }
    return fn("S1", 0, **{n: v for n, v in known.items() if n in params})


def test_reground_rows_excluded_from_step_latency() -> None:
    """D3 latency excludes reground requests. (a) ``turn_records`` over T-0111g's hand-built three-step turn plus a
    pass-2 request (``llm.request`` phase ``reground`` at step 1, and its ``LoopMeter`` row, not a repair, at 9999 ms):
    the steps are still 1-3, none is the pass-2 attempt or holds it as a retry, and step 1 keeps its own latency.
    (b) ``build_report``: a turn whose pass 2 took 9999 ms has step latency from its two steps only, and the 9999 ms
    in the reground cost."""
    records = g_tests.turn_log()
    at = next(i for i, r in enumerate(records) if r["kind"] == "tool.call") + 1
    records.insert(
        at,
        g_tests._rec(
            30,
            "llm.request",
            {
                "phase": "reground",
                "request_id": "t1-reground",
                "attempt_id": RG_ATTEMPT,
                "step": 1,
                "prompt": {},
            },
        ),
    )
    slow = {**g_tests._gen(RG_ATTEMPT, 200, base=0), "first_commit_ms": 8888.0, "done_ms": 9999.0}
    generates = [*g_tests.GENERATES[:2], slow, *g_tests.GENERATES[2:]]
    _trial_line, steps = _turn_records(records, generates)
    assert [s["step"] for s in steps] == [1, 2, 3], [
        (s.get("step"), s.get("attempt_id")) for s in steps
    ]
    assert all(s.get("attempt_id") != RG_ATTEMPT for s in steps), steps
    assert all(RG_ATTEMPT not in json.dumps(s.get("retries")) for s in steps), [
        s.get("retries") for s in steps
    ]
    one = steps[0]
    assert (one["first_commit_ms"], one["done_ms"]) == (302.0, 802.0), one

    on = {"visual_tokens": 280, "reground": True}
    trials = [_trial("S2", 0, {"cart": ["blue mug"]}, **on)]
    steps2 = [
        _step(
            "S2", 0, 1, "executed", 200, 600, call_name="click_at", executed_name="click_at", **on
        ),
        _step("S2", 0, 2, None, 250, 900, **on),
    ]
    rows = [
        _rg(
            "S2",
            0,
            1,
            "ok",
            ms=9999.0,
            attempt_id="S20-rg-1",
            http_status=200,
            first_commit_ms=9000.0,
            done_ms=9999.0,
        )
    ]
    for which, build in (("harness", api("build_report")), ("eval oracle", oracle.derive)):
        arm = build(trials, steps2, rows, _probes())["arms"]["280/on"]
        assert arm["latency"] == {
            "first_commit_ms": {"n": 2, "p50": 200, "p95": 250},
            "done_ms": {"n": 2, "p50": 600, "p95": 900},
        }, (which, arm["latency"])
        assert arm["reground_cost"]["latency_ms"] == {"n": 1, "p50": 9999.0, "p95": 9999.0}, (
            which,
            arm,
        )


SENTINEL = b"\xff\xd8HUD-SENTINEL-T0131-D\xff\xd9"


async def test_sentinel_covers_reground() -> None:
    """D4: a pass-2 request is a generate without ``cd``, so the harness's ``LoopMeter`` keeps its prompt under
    ``generate:<attempt_id>`` (mutation 13: a reground body labelled outside the scan), and ``sentinel_hits`` reports
    the HUD sentinel in it; the eval's coverage row maps the ``reground`` phase to ``generate:``."""
    from local_astra.conductor.prompt_builder import PromptBuilder

    class Inner:
        async def prefill(self, prefix: Any, request_id: str) -> Any:
            return types.SimpleNamespace(request_id=request_id)

        async def generate(
            self, prompt: Any, request_id: str, options: Any, xargs: Any, cd: Any = None
        ) -> Any:
            if False:
                yield None

    meter = api("LoopMeter")(Inner(), metrics=lambda: None, clock=time.monotonic)
    prompt = PromptBuilder("system reground").prefix([])
    async for _ in meter.generate(prompt, RG_ATTEMPT, None):
        pass
    assert meter.requests == {f"generate:{RG_ATTEMPT}": prompt}, sorted(meter.requests)
    body = g_tests._body_with(SENTINEL)
    assert api("sentinel_hits")({f"generate:{RG_ATTEMPT}": body}, SENTINEL) == [
        f"generate:{RG_ATTEMPT}"
    ]
    assert oracle.SCAN_PHASE["reground"] == "generate"


def test_eval_scan_coverage_maps_reground() -> None:
    """Not pending: the eval's D4 coverage row on T-0111g's hand-built turn plus a ``reground`` request. Listed as
    ``generate:<attempt>`` there is no gap; listed as ``reground:<attempt>`` or missing, exactly that attempt is a gap;
    and the mapping is D4's (prefill, generate/tool_step/reground, tool_repair)."""
    records = g_tests.turn_log()
    records.insert(
        6,
        g_tests._rec(
            30,
            "llm.request",
            {
                "phase": "reground",
                "request_id": "t1-reground",
                "attempt_id": RG_ATTEMPT,
                "step": 1,
                "prompt": {},
            },
        ),
    )
    sessions = {"s.jsonl": records}
    scanned = sorted([*g_tests.TR_SCANNED, f"generate:{RG_ATTEMPT}"])
    trial = {
        "task": "S1",
        "repeat": 0,
        "session_id": g_tests.TR_SESSION,
        "scanned_requests": scanned,
    }
    assert oracle.scan_coverage_gaps([trial], sessions) == []
    misnamed = {
        **trial,
        "scanned_requests": sorted([*g_tests.TR_SCANNED, f"reground:{RG_ATTEMPT}"]),
    }
    gaps = oracle.scan_coverage_gaps([misnamed], sessions)
    assert len(gaps) == 1 and f"generate:{RG_ATTEMPT}" in gaps[0], gaps
    missing = {**trial, "scanned_requests": list(g_tests.TR_SCANNED)}
    gaps = oracle.scan_coverage_gaps([missing], sessions)
    assert len(gaps) == 1 and RG_ATTEMPT in gaps[0], gaps
    assert oracle.SCAN_PHASE == {
        "prefill": "prefill",
        "generate": "generate",
        "tool_step": "generate",
        "reground": "generate",
        "tool_repair": "repair",
    }


def test_render_markdown_names_arms_label_and_history() -> None:
    """D5 "the report states the outcome label", and the *Regime*'s history line: ``report.md`` names every arm, the
    decision label, and T-0111g's run at ca9576f beside the results."""
    trials, steps, regrounds = ab_run()
    report = api("build_report")(trials, steps, regrounds, _probes())
    text = api("render_markdown")(report)
    missing = [label for label in LABELS if label not in text]
    assert missing == [], f"report.md does not name {missing}"
    assert report["decision"]["label"] in text, "report.md does not state the outcome label"
    assert "ca9576f" in text, "report.md does not set T-0111g's run (ca9576f) beside the results"


# =====================================================================================================
# The group D review round (orchestrator rulings at 4337125, and its acceptance of the implementer's choices)
# =====================================================================================================

REASON = "3 consecutive errored turns (last: S3/r0)"


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)``, with a keyword ``fn`` does not take yet a named failure instead of a ``TypeError``
    (the accepted signatures: ``decide(inputs, *, p1_passed, void=False, stopped=None)``, ``build_report(trials,
    steps, regrounds, probes, *, stopped=None)``)."""
    params = inspect.signature(fn).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [k for k in kwargs if k not in params and not takes_any]
    assert not missing, (
        f"T-0131 (ruling at 4337125): {fn.__name__}() takes no {missing} keyword yet"
    )
    return fn(*args, **kwargs)


def _probes_with(*, p1_560: int | None = 523, p2: bool = True, p3: bool = True) -> dict[str, Any]:
    """``_probes`` with P2 and/or P3 failed (HTTP 500), and P1's 560 frame delta set (``None``: no count)."""
    probes = _probes(p1_560=523)
    row = probes["p1"]["budgets"]["560"]
    row["frame_delta"] = p1_560
    row["ok"] = p1_560 is not None and oracle.p1_ok(560, p1_560) and row["crop_delta"] is not None
    probes["p1"]["ok"] = all(v["ok"] for v in probes["p1"]["budgets"].values())
    if not p2:
        probes["p2"] = {"ok": False, "http_status": 500, "healthy_after": True}
    if not p3:
        probes["p3"] = {"ok": False, "http_status": 500}
    return probes


# --- the latency rule: a pass2_timeout counts as 4000 ms in the reground p50 ----------------------------------------


def _latency_run(
    extra: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Three 280/on S2 turns, every gate of D5 met except (possibly) the reground p50: each a success with one
    executed click inside the blue mug's button (hit rate 1.0) and an answer (step done [900, 900, 900, 600, 600,
    600]: p95 900 ms), every step generate at 200, no safety finding, the engine up. Turn 0's pass 2 is ``ok`` at
    200 (done 600 ms); ``extra`` holds turns 1 and 2's reground lines. Every other arm is empty (no hit rate: not
    eligible), so 280/on is the only arm that can be chosen, and 280/off's success is 0 (a gain of 3)."""
    on = {"visual_tokens": 280, "reground": True}
    clk = {"call_name": "click_at", "executed_name": "click_at"}
    trials = [
        _trial("S2", r, {"cart": ["blue mug"]}, voiced=["Added the blue mug."], **on)
        for r in range(3)
    ]
    steps = []
    for r in range(3):
        steps += [
            _step(
                "S2",
                r,
                1,
                "executed",
                300,
                900,
                click=_click([540, 330], [BLUE], True, 0.0),
                **clk,
                **on,
            ),
            _step("S2", r, 2, None, 250, 600, voiced_sources=["Added the blue mug."], **on),
        ]
    ok = _rg(
        "S2",
        0,
        1,
        "ok",
        ms=650.0,
        attempt_id="S20-rg-1",
        http_status=200,
        first_commit_ms=200.0,
        done_ms=600.0,
    )
    regrounds = [{**ok, **on}, *({**g, **on} for g in extra)]
    return trials, steps, regrounds


_T1 = _rg("S2", 1, 1, "pass2_timeout", ms=4000.0, attempt_id="S21-rg-1", http_status=0)
#: a timed-out pass 2 whose aborted stream still reported a done time: it counts 4000.0 all the same
_T2 = _rg(
    "S2", 2, 1, "pass2_timeout", ms=4000.0, attempt_id="S22-rg-1", http_status=0, done_ms=3990.0
)
_OK1 = _rg("S2", 1, 1, "ok", ms=750.0, attempt_id="S21-rg-1", http_status=200, done_ms=700.0)
_OK2 = _rg("S2", 2, 1, "ok", ms=850.0, attempt_id="S22-rg-1", http_status=200, done_ms=800.0)
_GATES_MET = {
    "accepted": True,
    "safety": 0,
    "engine_up": True,
    "success": 3,
    "hit_rate": 1.0,
    "done_p95_ms": 900,
}


@pytest.mark.parametrize("which", IMPLS)
def test_timeout_counts_4000_in_reground_p50(which: str) -> None:
    """Ruling at 4337125 (the approved "reground p50 <= 1.0 s" read over every pass-2 attempt): a ``pass2_timeout``
    counts as ``REGROUND_TIMEOUT_MS`` = 4000.0 ms in ``reground_cost.latency_ms``, whatever its ``done_ms``, and so in
    D5's p50 gate. Hand table: 280/on meets every other gate (acceptance -- the timeouts are exempt from it --,
    safety, engine, success 3 of the best 3, hit rate 1.0, done p95 900 ms) with a gain of 3 over 280/off.

    - Two of its three pass 2s timed out: latency [600.0, 4000.0, 4000.0], p50 the 2nd = 4000.0 > 1000, so 280/on is
      ineligible **only** by this rule and the label is "no change" (read the timeouts out, as the tip did, and the
      p50 is 600.0 and the label "280/on").
    - The control: the same run with those two pass 2s ``ok`` at 700/800 ms is eligible and chosen, "280/on".
    - A non-timeout ``error`` with no ``done_ms`` contributes no latency."""
    build = _report_fn(which)
    report = build(*_latency_run([_T1, _T2]), _probes())
    arm = report["arms"]["280/on"]
    assert arm["reground_cost"]["latency_ms"] == {"n": 3, "p50": 4000.0, "p95": 4000.0}, arm[
        "reground_cost"
    ]
    assert arm["reground_cost"]["requests"] == 3, arm["reground_cost"]
    assert arm["acceptance"]["all_200"] is True and arm["acceptance"]["regrounds"] == 1, arm[
        "acceptance"
    ]
    assert report["decision"]["inputs"]["280/on"] == {**_GATES_MET, "reground_p50_ms": 4000.0}, (
        report["decision"]
    )
    assert report["decision"]["eligible"] == [], report["decision"]
    assert report["decision"]["label"] == "no change", report["decision"]

    control = build(*_latency_run([_OK1, _OK2]), _probes())
    assert control["arms"]["280/on"]["reground_cost"]["latency_ms"] == {
        "n": 3,
        "p50": 700.0,
        "p95": 800.0,
    }
    assert control["decision"]["inputs"]["280/on"] == {**_GATES_MET, "reground_p50_ms": 700.0}, (
        control["decision"]
    )
    assert control["decision"]["eligible"] == ["280/on"], control["decision"]
    assert control["decision"]["label"] == "280/on", control["decision"]

    err = _rg("S2", 1, 1, "error", ms=80.0, attempt_id="S21-rg-1", http_status=0)
    with_err = build(*_latency_run([err]), _probes())["arms"]["280/on"]["reground_cost"]
    assert with_err["latency_ms"] == {"n": 1, "p50": 600.0, "p95": 600.0}, with_err


def test_eval_oracle_timeout_is_the_session_constant() -> None:
    """Not pending: the 4000 ms the eval counts per timeout is the session's ``REGROUND_TIMEOUT_MS`` (implementer
    decision 1, accepted), so the oracle cannot drift from the timeout the run used."""
    from local_astra.conductor import session

    assert oracle.REGROUND_TIMEOUT_MS == session.REGROUND_TIMEOUT_MS == 4000, (
        oracle.REGROUND_TIMEOUT_MS,
        getattr(session, "REGROUND_TIMEOUT_MS", None),
    )


# --- the labels: inconclusive, then void, then incomplete, then D5 -------------------------------------------------

#: (decide's keywords, label) by hand over DECISIONS["chosen-fewest-tokens"] (D5 alone: "560/off")
LABEL_ORDER: dict[str, tuple[dict[str, Any], str]] = {
    "d5-by-default": ({"p1_passed": True}, "560/off"),
    "d5-keywords-neutral": ({"p1_passed": True, "void": False, "stopped": None}, "560/off"),
    "void": ({"p1_passed": True, "void": True}, "void"),
    "incomplete": ({"p1_passed": True, "stopped": REASON}, "incomplete"),
    "void-before-incomplete": ({"p1_passed": True, "void": True, "stopped": REASON}, "void"),
    "inconclusive-before-void": ({"p1_passed": False, "void": True}, "inconclusive"),
    "inconclusive-before-incomplete": ({"p1_passed": False, "stopped": REASON}, "inconclusive"),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(LABEL_ORDER))
def test_decide_label_order(which: str, row: str) -> None:
    """Ruling at 4337125 with the accepted ``decide(inputs, *, p1_passed, void=False, stopped=None)``: a P1 failure is
    "inconclusive" whatever else failed; else a void run (P2 or P3) is "void"; else a run stopped early is
    "incomplete"; else D5's label. The keywords at their defaults change nothing."""
    kwargs, label = LABEL_ORDER[row]
    inputs = json.loads(json.dumps(DECISIONS["chosen-fewest-tokens"][0]))
    got = _call(impl(which, "decide"), inputs, **kwargs)
    assert got == label, f"{row}: {got!r}, want {label!r}"


#: (probes, stopped, the run, label) by hand; the run is ``ab_run`` (D5: "no change") or ``_latency_run`` with its
#: control pass 2s (D5: "280/on"), so an "incomplete" never coincides with D5's own label
REPORT_LABELS: dict[str, tuple[dict[str, Any], str | None, str, str]] = {
    "void-p2": (_probes_with(p2=False), None, "none", "void"),
    "void-p3": (_probes_with(p3=False), None, "none", "void"),
    "inconclusive-p1-and-p2": (_probes_with(p1_560=273, p2=False), None, "none", "inconclusive"),
    "inconclusive-p1-none-delta": (_probes_with(p1_560=None), None, "none", "inconclusive"),
    "incomplete-ab-run": (_probes(), REASON, "ab", "incomplete"),
    "incomplete-would-choose-280-on": (_probes(), REASON, "latency-control", "incomplete"),
    "not-stopped-would-choose-280-on": (_probes(), None, "latency-control", "280/on"),
}


def _run_named(
    name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if name == "ab":
        return ab_run()
    if name == "latency-control":
        return _latency_run([_OK1, _OK2])
    return [], [], []


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(REPORT_LABELS))
def test_report_labels_void_incomplete(which: str, row: str) -> None:
    """Ruling at 4337125 through ``build_report(..., *, stopped=None)`` (and the eval's ``derive``): the report's
    label, ``void``, the top-level ``stopped`` (the reason, or None) and ``decision.eligible`` (``[]`` for
    inconclusive, void and incomplete; D5's otherwise). A stopped run keeps its D5 inputs (report-only)."""
    probes, stopped, name, label = REPORT_LABELS[row]
    trials, steps, regrounds = _run_named(name)
    report = _call(_report_fn(which), trials, steps, regrounds, probes, stopped=stopped)
    assert report["decision"]["label"] == label, (row, report["decision"]["label"])
    assert report.get("stopped", "<no key>") == stopped, (row, report.get("stopped", "<no key>"))
    assert report["void"] is (not all(probes[p]["ok"] for p in ("p1", "p2", "p3"))), (
        row,
        report["void"],
    )
    want_eligible = ["280/on"] if label == "280/on" else []
    assert report["decision"]["eligible"] == want_eligible, (row, report["decision"])
    if name == "ab":
        assert report["decision"]["inputs"] == AB_DECISION["inputs"], report["decision"]["inputs"]


# --- read_errors ----------------------------------------------------------------------------------------------------

READ_FAILED = "TimeoutError('the box read did not answer')"


@pytest.mark.parametrize("which", IMPLS)
def test_click_read_errors(which: str) -> None:
    """Ruling at 4337125 (shape accepted 2026-10-22): each click record carries ``read_error`` (None, or the repr of
    the failure); each arm's ``clicks`` and ``clicks_p1`` sections carry ``read_errors``, the count of records whose
    ``read_error`` is not None, S4's included. Hand table, 560/on: an S2 click whose box read failed (no boxes: a
    miss with no error), an S2 click inside the blue mug's button read fine, an S4 click whose read failed
    (unscored, but counted), and pass 1's counterfactual beside each with the same read."""
    on = {"visual_tokens": 560, "reground": True}
    clk = {"call_name": "click_at", "executed_name": "click_at"}
    trials = [_trial("S2", r, {"cart": ["blue mug"]}, **on) for r in range(2)] + [
        _trial("S4", 0, {"cart": []}, **on)
    ]
    steps = [
        _step(
            "S2",
            0,
            1,
            "executed",
            300,
            900,
            click=_click([540, 330], [], False, None, READ_FAILED),
            click_p1=_click([540, 371], [], False, None, READ_FAILED),
            **clk,
            **on,
        ),
        _step(
            "S2",
            1,
            1,
            "executed",
            300,
            900,
            click=_click([540, 330], [BLUE], True, 0.0),
            click_p1=_click([540, 371], [BLUE], False, 20.0),
            **clk,
            **on,
        ),
        _step(
            "S4",
            0,
            1,
            "executed",
            300,
            900,
            click=_click([100, 100], [], False, None, READ_FAILED),
            click_p1=_click([100, 120], [], False, None, READ_FAILED),
            **clk,
            **on,
        ),
    ]
    arm = _report_fn(which)(trials, steps, [], _probes())["arms"]["560/on"]
    assert arm["clicks"] == {
        "n": 2,
        "hits": 1,
        "hit_rate": 0.5,
        "error_px": {"n": 1, "p50": 0.0, "p95": 0.0, "mean": 0.0},
        "unscored": 1,
        "read_errors": 2,
    }, arm["clicks"]
    assert arm["clicks_p1"] == {
        "n": 2,
        "hits": 0,
        "hit_rate": 0.0,
        "error_px": {"n": 1, "p50": 20.0, "p95": 20.0, "mean": 20.0},
        "unscored": 1,
        "read_errors": 2,
    }, arm["clicks_p1"]


# --- probe_findings with a None delta (eval) ----------------------------------------------------------------------


def test_eval_oracle_probe_findings_none_delta() -> None:
    """Not pending (ruling at 4337125): the eval's ``probe_findings`` reads a ``None`` P1 delta (``/tokenize`` gave no
    count) as a failed delta, never a ``TypeError``: recorded as failed, no finding; recorded as passed, a finding
    naming the budget; the same for a ``None`` crop delta."""
    probes = _probes_with(p1_560=None)
    assert probes["p1"]["ok"] is False and probes["p1"]["budgets"]["560"]["ok"] is False
    assert oracle.probe_findings(probes) == []
    lying = json.loads(json.dumps(probes))
    lying["p1"]["budgets"]["560"]["ok"] = True
    lying["p1"]["ok"] = True
    found = oracle.probe_findings(lying)
    assert found and any("560" in f for f in found), found
    crop = _probes()
    crop["p1"]["budgets"]["1120"]["crop_delta"] = None
    crop["p1"]["budgets"]["1120"]["ok"] = False
    crop["p1"]["ok"] = False
    assert oracle.probe_findings(crop) == []


# --- the configuration evidence -----------------------------------------------------------------------------------


def _bodies(budget: int | None, *, guidance: bool) -> dict[str, dict[str, Any]]:
    """A prefill and a generate body as the real ``VllmClient(visual_tokens=budget)`` builds them for a prompt whose
    system text holds the live guidance (or not)."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE, PromptBuilder

    system = "You are Astra." + (f"\n{LIVE_TOOL_GUIDANCE}" if guidance else "")
    prompt = PromptBuilder(system).prefix([])
    client = VllmClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(599)), visual_tokens=budget
    )
    return {
        f"{phase}:t1-{phase}": client.request_body(prompt, f"t1-{phase}")
        for phase in ("prefill", "generate")
    }


def test_config_evidence_from_bodies() -> None:
    """Ruling at 4337125 with the accepted ``config_evidence(bodies, guidance) -> (budgets_sent, guidance_sent)`` over
    the bodies built for the sentinel scan (label -> body): the sorted distinct ``mm_processor_kwargs.max_soft_tokens``
    values, and whether the guidance is in the system text of **every** body. Real bodies (``VllmClient.request_body``):
    560 with the guidance -> ([560], True); one body without it -> False; a 280 and a 560 body -> [280, 560]; an
    unset budget sends no kwarg -> []; the guidance only in a user message is not in the system text; no body at all
    -> ([], False)."""
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE

    ev = api("config_evidence")
    assert ev(_bodies(560, guidance=True), LIVE_TOOL_GUIDANCE) == ([560], True)
    mixed = {
        **_bodies(560, guidance=True),
        "generate:t1-gen-2": _bodies(560, guidance=False)["generate:t1-generate"],
    }
    assert ev(mixed, LIVE_TOOL_GUIDANCE) == ([560], False)
    two = {
        "prefill:a": _bodies(280, guidance=True)["prefill:t1-prefill"],
        **_bodies(560, guidance=True),
    }
    assert ev(two, LIVE_TOOL_GUIDANCE) == ([280, 560], True)
    assert ev(_bodies(None, guidance=True), LIVE_TOOL_GUIDANCE) == ([], True)
    in_user = _bodies(1120, guidance=False)
    for body in in_user.values():
        body["messages"] = [*body["messages"], {"role": "user", "content": LIVE_TOOL_GUIDANCE}]
    assert ev(in_user, LIVE_TOOL_GUIDANCE) == ([1120], False)
    assert ev({}, LIVE_TOOL_GUIDANCE) == ([], False)


def _cfg_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One S2 turn per arm, each sending its own budget with the guidance; each executes one click (step 1) and in
    the on arms that click has its reground line; plus a 560/off turn that never ran (no body)."""
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    regrounds: list[dict[str, Any]] = []
    for b, r in SPEC_ARMS:
        arm = {"visual_tokens": b, "reground": r}
        trials.append(
            _trial(
                "S2",
                0,
                {"cart": ["blue mug"]},
                scanned_requests=["generate:x", "prefill:x"],
                budgets_sent=[b],
                guidance_sent=True,
                **arm,
            )
        )
        steps.append(
            _step(
                "S2",
                0,
                1,
                "executed",
                300,
                900,
                call_name="click_at",
                executed_name="click_at",
                click=_click([540, 330], [BLUE], True, 0.0),
                **arm,
            )
        )
        if r:
            regrounds.append({**_rg("S2", 0, 1, "ok", http_status=200, done_ms=400.0), **arm})
    trials.append(
        _trial(
            "S2",
            1,
            None,
            exception="not run: 3 consecutive errored turns",
            scanned_requests=[],
            budgets_sent=[],
            guidance_sent=False,
            visual_tokens=560,
            reground=False,
        )
    )
    return trials, steps, regrounds


def _mutate(rows: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    return [{**row, **kw} for row in rows]


def test_eval_oracle_config_findings() -> None:
    """Not pending: the eval's configuration-evidence check against hand tables shaped like the reviewer's mutants.
    Clean (one turn per arm, and a turn that never ran) -> no finding. ``f_client_budget_280`` (every trial sent 280)
    -> a finding for each of the four arms above 280 and none for the 280 arms; ``f3_no_guidance`` -> one per ran
    trial (six); ``f2_session_reground_false`` (no reground line) -> one per on arm (three); a reground line in an off
    arm -> a finding; a turn that sent nothing is never judged."""
    trials, steps, regrounds = _cfg_run()
    assert oracle.config_findings(trials, steps, regrounds) == []

    ran = [t for t in trials if t["scanned_requests"]]
    rest = [t for t in trials if not t["scanned_requests"]]
    f280 = oracle.config_findings([*_mutate(ran, budgets_sent=[280]), *rest], steps, regrounds)
    assert len(f280) == 4 and all(
        f.split(" ", 1)[0] in ("560/off", "560/on", "1120/off", "1120/on") for f in f280
    ), f280
    no_guidance = oracle.config_findings(
        [*_mutate(ran, guidance_sent=False), *rest], steps, regrounds
    )
    assert len(no_guidance) == 6, no_guidance
    no_reground = oracle.config_findings(trials, steps, [])
    assert sorted(f.split(" ", 1)[0] for f in no_reground) == ["1120/on", "280/on", "560/on"], (
        no_reground
    )
    stray = {**regrounds[0], "visual_tokens": 280, "reground": False}
    off_line = oracle.config_findings(trials, steps, [*regrounds, stray])
    assert len(off_line) == 1 and off_line[0].startswith("280/off"), off_line
    missing = oracle.config_findings(
        [{k: v for k, v in t.items() if k != "budgets_sent"} for t in trials], steps, regrounds
    )
    assert len(missing) == 6, missing


# --- the turn path and the failed-probe path through main(), over CPU stand-ins ----------------------------------


def _engine(rig: Any, *, health: bool = True, p2_status: int = 200, p3_status: int = 200) -> Any:
    """Past every refusal, a stand-in engine on CPU: ``/v1/models`` serves the model, ``/health`` answers (or not),
    ``/metrics`` shows an idle engine (T-0111g's seams ``_get_text`` / ``read_metrics``), ``nvidia-smi`` is absent
    (the sampler records the error and goes on), and every ``VllmClient`` the harness builds keeps the real
    ``request_body`` (the budget it was built with) while its requests go to a fake: the P2/P3 probes (told apart by
    the image: P2 has one) answer ``p2_status``/``p3_status``; a turn's requests go to the T-0131 group B pass-2 fake
    LLM over the turn's fake browser. Returns the recording namespace."""
    import httpx

    from local_astra.conductor import llm_client
    from local_astra.contracts.llm import ImagePart
    from tests.fakes.llm import http_error

    h = rig.h
    g = h.G
    model = llm_client.VllmEndpoint().model
    real_client = llm_client.VllmClient
    st = types.SimpleNamespace(
        health=health,
        probes=[],
        clients=[],
        turn_llms=[],
        browsers=[],
        pending_browser=None,
        nvidia=[],
        fail_box_read=set(),
    )

    def get_text(url: str, timeout_s: float = 10.0) -> str | None:
        path = url.rstrip("/")
        if path.endswith("/models"):
            return json.dumps({"data": [{"id": model}]})
        if path.endswith("/health"):
            return "" if st.health else None
        if path.endswith("/version"):
            return json.dumps({"version": "0.29.0"})
        return None

    rig.mp.setattr(g, "_get_text", get_text)
    idle = {
        name: 0 for name in (g.METRIC_QUERIES, g.METRIC_HITS, g.METRIC_RUNNING, g.METRIC_WAITING)
    }
    rig.mp.setattr(g, "read_metrics", lambda base_url: dict(idle))

    prev_popen = subprocess.Popen

    def popen(args: Any, *a: Any, **k: Any) -> Any:
        text = " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if "nvidia-smi" in text:
            st.nvidia.append(text)
            raise FileNotFoundError("nvidia-smi (absent in the CPU rig)")
        return prev_popen(args, *a, **k)

    rig.mp.setattr(subprocess, "Popen", popen)

    class ProbeLlm:
        def __init__(self, visual_tokens: Any) -> None:
            self.visual_tokens = visual_tokens

        async def generate(
            self,
            prompt: Any,
            request_id: str,
            options: Any = None,
            *,
            xargs: Any = None,
            cd: Any = None,
        ) -> Any:
            image = any(isinstance(p, ImagePart) for m in prompt.messages for p in m.content)
            st.probes.append({"image": image, "visual_tokens": self.visual_tokens})
            status = p2_status if image else p3_status
            if status != 200:
                raise http_error(status, detail="CPU rig")
            for _ in ():
                yield _

        async def prefill(self, prefix: Any, request_id: str) -> Any:
            raise AssertionError("a probe prefilled")

        async def abort(self, request_id: str) -> None:
            return None

    def refuse(request: Any) -> Any:
        raise AssertionError(
            f"the CPU rig's real client tried to send {request.method} {request.url}"
        )

    class StubClient:
        """The real client's ``request_body`` (built with the harness's own arguments); requests to a fake."""

        def __init__(self, endpoint: Any = None, **kwargs: Any) -> None:
            kwargs.pop("transport", None)
            self.kwargs = kwargs
            self._real = real_client(endpoint, transport=httpx.MockTransport(refuse), **kwargs)
            browser, st.pending_browser = st.pending_browser, None
            if browser is None:
                self._fake: Any = ProbeLlm(kwargs.get("visual_tokens"))
            else:
                self._fake = _turn_llm(browser)
                st.turn_llms.append(self._fake)
            st.clients.append(self)

        def request_body(self, *a: Any, **k: Any) -> Any:
            return self._real.request_body(*a, **k)

        def generate(
            self,
            prompt: Any,
            request_id: str,
            options: Any = None,
            xargs: Any = None,
            *,
            cd: Any = None,
        ) -> Any:
            return self._fake.generate(prompt, request_id, options, xargs=xargs, cd=cd)

        async def aclose(self) -> None:
            await self._real.aclose()

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self._fake, name)

    rig.mp.setattr(llm_client, "VllmClient", StubClient)
    return st


def _turn_llm(browser: Any) -> Any:
    """The group B pass-2 fake LLM for one S2 turn: step 1 says a sentence and clicks (500, 250), pass 2 answers at
    (900, 100) in the crop, step 2 answers."""
    from tests.unit._live_browser_support import make_reground_llm
    from tests.unit._tool_turn_support import VALID_CLICK, answer, tool

    # one spoken word per step: the turn runs on the session's wall clock, and the fake voice takes time per word
    llm = make_reground_llm(browser, [answer("Adding. "), tool(VALID_CLICK)])
    llm.script_pass2([tool("<|tool_call>call:click_at{x:900,y:100}<tool_call|>")])
    llm.script_steps([answer("Added.")])
    return llm


def _turn_side(rig: Any, st: Any) -> None:
    """The browser side of a turn on CPU: a start-page server that serves nothing, and in place of the
    ``BrowserToolExecutor`` the group B ``FakeLiveBrowser`` (real 1280x800 JPEG frames) with a ``driver`` whose page
    answers the box read (the blue mug's "Add to cart"; or fails, for the turns in ``st.fail_box_read``) and
    ``window.__state`` (the blue mug in the cart); the tool kit is the live kit over the harness's watch."""
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit

    class Pages:
        origin = "http://127.0.0.1:9"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Page:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"cart": ["blue mug"]}
            if self.fail:
                raise RuntimeError("CPU rig: the box read failed")
            return {"boxes": [dict(BLUE)], "element": 'button[data-item="blue mug"] Add to cart'}

    class Browser(FakeLiveBrowser):
        def __init__(self, *, now_ms: Any = None, viewport: Any = (1280, 800), **kw: Any) -> None:
            n = len(st.browsers)
            super().__init__(
                [jpeg_frame(f"onset-{n}")],
                post_frames=[jpeg_frame(f"after-{n}-1"), jpeg_frame(f"after-{n}-2")],
                now_ms=now_ms or (lambda: 0),
            )
            self.driver = types.SimpleNamespace(
                viewport=tuple(viewport), page=Page(n in st.fail_box_read)
            )
            st.browsers.append(self)
            st.pending_browser = self

    rig.mp.setattr(server_mod, "StartPageServer", Pages)
    rig.mp.setattr(executor_mod, "BrowserToolExecutor", Browser)

    def load(profile: Any, **kw: Any) -> Any:
        lb = kw.get("live_browser")
        return (rig.kit, None) if lb is None else (live_kit(lb), None)

    rig.mp.setattr(rig.cli, "load_tool_kit", load)


def _p1_passing() -> dict[str, Any]:
    return _probes()["p1"]


def _run_main(rig: Any, turns: Sequence[tuple[int, bool, str, int]]) -> tuple[int, str]:
    rig.mp.setattr(rig.h, "order", lambda seed: [tuple(t) for t in turns])
    rc = rig.h.main(_argv(rig))
    return rc, rig.capsys.readouterr().err


def _out_jsonl(rig: Any, name: str) -> list[dict[str, Any]]:
    p = rig.out / name
    assert p.is_file(), f"the run wrote no {name}"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


#: two turns, the order fixed for the row: a ReGround-on arm above 280 and a ReGround-off arm above 280
CFG_TURNS = ((560, True, "S2", 0), (1120, False, "S2", 0))


def test_turns_carry_configuration_evidence(rig: Any) -> None:
    """Ruling at 4337125, end to end through ``main`` on CPU (the engine, the browser and the start pages stood in
    by fakes; the ``Session``, ``LoopMeter``, the record extraction and the report are the harness's own), two turns:
    560/on and 1120/off, each S2 clicking once. Each trial's ``budgets_sent`` is its arm's budget alone and
    ``guidance_sent`` is True; the 560/on click has its reground line (``ok``) and the 1120/off turn none; the eval's
    ``config_findings`` over the written files finds nothing. This is the CPU kill of the reviewer's
    ``f_client_budget_280`` (the client built at 280: ``budgets_sent == [280]`` in both arms),
    ``f2_session_reground_false`` (no reground line in the on arm) and ``f3_no_guidance`` (``guidance_sent`` False).

    The 1120/off turn's box read fails: its click record carries the failure's repr in ``read_error`` and the arm's
    ``clicks.read_errors`` is 1, while 560/on's read is clean (``read_error`` None, ``read_errors`` 0). The probes all
    pass, so this is also the other side of the failed-probe rows: turns run, exit 0, not void, not stopped."""
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE

    st = _engine(rig)
    _turn_side(rig, st)
    st.fail_box_read = {1}
    rig.mp.setattr(rig.h, "probe_p1", lambda base_url, model: _p1_passing())
    rc, err = _run_main(rig, CFG_TURNS)
    assert rc == rig.h.EXIT_OK == 0, f"exit {rc}\n{err[-3000:]}"
    trials = _out_jsonl(rig, "trials.jsonl")
    steps = _out_jsonl(rig, "steps.jsonl")
    regrounds = _out_jsonl(rig, "regrounds.jsonl")
    assert [(t["visual_tokens"], t["reground"]) for t in trials] == [(560, True), (1120, False)], (
        trials
    )
    assert [t.get("exception") for t in trials] == [None, None], [
        t.get("exception") for t in trials
    ]
    assert all(t["scanned_requests"] for t in trials), [t["scanned_requests"] for t in trials]
    assert [t.get("budgets_sent", "<absent>") for t in trials] == [[560], [1120]], (
        "budgets_sent: the max_soft_tokens of every forwarded body is the arm's budget alone"
    )
    assert [t.get("guidance_sent", "<absent>") for t in trials] == [True, True], (
        "guidance_sent: LIVE_TOOL_GUIDANCE in the system text of every forwarded body"
    )
    assert [(g["visual_tokens"], g["reground"], g["step"], g["outcome"]) for g in regrounds] == [
        (560, True, 1, "ok")
    ], regrounds
    clicks = [
        (s["visual_tokens"], s["step"], s.get("executed_name"))
        for s in steps
        if s.get("click") is not None
    ]
    assert clicks == [(560, 1, "click_at"), (1120, 1, "click_at")], clicks
    assert oracle.config_findings(trials, steps, regrounds) == []
    by_arm = {s["visual_tokens"]: s["click"] for s in steps if s.get("click") is not None}
    assert set(by_arm[560]) == oracle.CLICK_KEYS and by_arm[560]["read_error"] is None, by_arm[560]
    assert (
        isinstance(by_arm[1120]["read_error"], str)
        and "the box read failed" in by_arm[1120]["read_error"]
    ), by_arm[1120]
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    assert report["void"] is False and report.get("stopped", "<no key>") is None, report.get(
        "stopped", "<no key>"
    )
    assert report["arms"]["560/on"]["clicks"]["read_errors"] == 0
    assert report["arms"]["1120/off"]["clicks"]["read_errors"] == 1, report["arms"]["1120/off"][
        "clicks"
    ]
    assert report["decision"]["label"] not in ("void", "inconclusive", "incomplete"), report[
        "decision"
    ]
    # the GPU eval's re-derivation over these very files agrees with the harness's report (its D3/D5 rows would pass)
    manifest = json.loads((rig.out / "manifest.json").read_text(encoding="utf-8"))
    probes = json.loads((rig.out / "probes.json").read_text(encoding="utf-8"))
    want = oracle.derive(
        trials,
        steps,
        regrounds,
        probes,
        _out_jsonl(rig, "turns.jsonl"),
        stopped=manifest.get("stopped"),
    )
    diff = {
        f"{label}.{s}": {"report": report["arms"][label].get(s), "derived": want["arms"][label][s]}
        for label in LABELS
        for s in oracle.ARM_SECTIONS
        if report["arms"][label].get(s) != want["arms"][label][s]
    }
    assert diff == {}, json.dumps(diff, indent=1, default=str)
    assert report["decision"] == want["decision"], (report["decision"], want["decision"])
    # the guidance the bodies were checked against is the live constant itself (premise)
    assert LIVE_TOOL_GUIDANCE and len(st.turn_llms) == 2 and len(st.browsers) == 2


#: (run_probes' return, label) for the failed-probe rows with ``run_probes`` stubbed
FAILED_PROBES: dict[str, tuple[dict[str, Any], str]] = {
    "p1-below": (_probes_with(p1_560=273), "inconclusive"),
    "p1-none-delta": (_probes_with(p1_560=None), "inconclusive"),
    "p2-500": (_probes_with(p2=False), "void"),
    "p3-500": (_probes_with(p3=False), "void"),
}


def _assert_void_run(rig: Any, rc: int, err: str, label: str, case: str) -> dict[str, Any]:
    h = rig.h
    assert rc == h.EXIT_PROBE == 3, f"{case}: exit {rc}, not the failed probe's 3\n{err[-3000:]}"
    assert rig.built == [], f"{case}: a turn started after a failed probe: {rig.built}"
    for name in ("trials.jsonl", "steps.jsonl", "regrounds.jsonl"):
        assert _out_jsonl(rig, name) == [], f"{case}: {name} is not empty after a failed probe"
    assert (rig.out / "report.json").is_file(), f"{case}: no report.json after a failed probe"
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    assert report["void"] is True, (case, report["void"])
    assert report["decision"]["label"] == label, (case, report["decision"])
    assert report["decision"]["eligible"] == [], (case, report["decision"])
    assert report.get("stopped", "<no key>") is None, (case, report.get("stopped", "<no key>"))
    probes = json.loads((rig.out / "probes.json").read_text(encoding="utf-8"))
    assert report["probes"] == probes, case
    assert oracle.probe_findings(probes) == [], case
    md = (rig.out / "report.md").read_text(encoding="utf-8")
    assert label in md, f"{case}: report.md does not state {label!r}"
    return probes


@pytest.mark.parametrize("case", list(FAILED_PROBES))
def test_failed_probe_is_void_exit_3_no_turns(rig: Any, case: str) -> None:
    """D2 through ``main`` with ``run_probes`` stubbed (the engine stood in on CPU, the browser side refused and
    recorded): a failed probe exits 3, writes ``probes.json`` and a void ``report.json``/``report.md`` labelled
    "inconclusive" (P1: below expectation, or no count at all) or "void" (P2 or P3 at HTTP 500), with no eligible arm
    and ``stopped`` null, and runs no turn (no start-page server or browser built; ``trials.jsonl``,
    ``steps.jsonl`` and ``regrounds.jsonl`` empty). The eval's ``probe_findings`` reads the written probes without
    a finding (and without a ``TypeError`` on the None delta). Kills ``h_probe_fail_exit0`` (exit),
    ``h2_probe_fail_no_report`` (no report.json) and ``h3_probe_fail_runs_turns`` (a turn's server constructed and a
    trial written)."""
    probes, label = FAILED_PROBES[case]
    _engine(rig)
    rig.mp.setattr(rig.h, "run_probes", lambda base_url, model: json.loads(json.dumps(probes)))
    rc, err = _run_main(rig, CFG_TURNS)
    got = _assert_void_run(rig, rc, err, label, case)
    assert got == probes, case


@pytest.mark.parametrize(
    ("case", "health", "p2_status", "p3_status", "label"),
    [
        ("p2-200-engine-unhealthy-after", False, 200, 200, "void"),
        ("p2-500", True, 500, 200, "void"),
        ("p3-500", True, 200, 500, "void"),
    ],
)
def test_run_probes_p2_needs_a_healthy_engine(
    rig: Any, case: str, health: bool, p2_status: int, p3_status: int, label: str
) -> None:
    """D2 P2 "returns 200 **and leaves the engine healthy**", with the real ``run_probes`` (only P1's ``/tokenize``
    stubbed, passing): a 200 generate at 1120 after which ``/health`` does not answer is a failed P2 (``ok`` False,
    ``healthy_after`` False), so the run is void, exit 3, no turn (kills ``g_p2_ignores_health``); a P2 or P3 at 500
    is void too. P2 sends one image at 1120 and P3 is text-only and still carries a budget."""
    st = _engine(rig, health=health, p2_status=p2_status, p3_status=p3_status)
    rig.mp.setattr(rig.h, "probe_p1", lambda base_url, model: _p1_passing())
    rc, err = _run_main(rig, CFG_TURNS)
    probes = _assert_void_run(rig, rc, err, label, case)
    assert probes["p2"]["healthy_after"] is health, probes["p2"]
    assert probes["p2"]["ok"] is (p2_status == 200 and health), probes["p2"]
    assert probes["p2"]["http_status"] == p2_status and probes["p3"]["http_status"] == p3_status, (
        probes
    )
    assert probes["p3"]["ok"] is (p3_status == 200), probes["p3"]
    image = [p for p in st.probes if p["image"]]
    text = [p for p in st.probes if not p["image"]]
    assert len(image) == 1 and image[0]["visual_tokens"] == 1120, st.probes
    assert len(text) == 1 and text[0]["visual_tokens"] in (280, 560, 1120), st.probes


def test_refuses_reground_timeout_mismatch(rig: Any) -> None:
    """The accepted choice (2026-10-22): the harness's ``REGROUND_TIMEOUT_MS`` -- the 4000 ms it counts per timed-out
    pass 2 -- is checked against the session's in ``main``; a mismatch exits 2 having written no out-dir and built no
    browser. The engine is stood in, so only the mismatch can refuse; at the matching value the same rig runs."""
    from local_astra.conductor import session

    h = rig.h
    assert api("REGROUND_TIMEOUT_MS") == session.REGROUND_TIMEOUT_MS == 4000
    _engine(rig)
    rig.mp.setattr(h, "probe_p1", lambda base_url, model: _p1_passing())
    rig.mp.setattr(h, "REGROUND_TIMEOUT_MS", 3000)
    rc, err = _run_main(rig, ())
    assert rc == h.EXIT_REFUSED == 2, f"exit {rc}\n{err[-3000:]}"
    assert not rig.out.exists(), "the refusal wrote the out-dir"
    assert rig.built == [], rig.built
    assert "REGROUND_TIMEOUT_MS" in err or "4000" in err, err


# --- report.md ------------------------------------------------------------------------------------------------------


def _void_lines(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if re.search(r"\bvoid\b", ln, re.IGNORECASE)]


def test_render_markdown_void_parenthetical_only_when_void() -> None:
    """Ruling at 4337125: ``report.md``'s void line carries its parenthetical (why: a probe failed) only when the run is
    void. Over ``ab_run`` (not void) no line naming "void" has a parenthesis; over the same with P2 failed one does,
    and it names the probe."""
    render, build = api("render_markdown"), api("build_report")
    trials, steps, regrounds = ab_run()
    md = render(build(trials, steps, regrounds, _probes()))
    lines = _void_lines(md)
    assert lines, "report.md has no void line"
    assert [ln for ln in lines if "(" in ln] == [], (
        f"a void parenthetical on a run that is not void: {lines}"
    )
    void_md = render(build([], [], [], _probes_with(p2=False)))
    with_paren = [ln for ln in _void_lines(void_md) if "(" in ln and "probe" in ln.lower()]
    assert with_paren, f"the void run's report.md does not say why: {_void_lines(void_md)}"


def test_render_markdown_stopped_and_incomplete() -> None:
    """Ruling at 4337125: a run stopped early has its reason in ``report.md`` beside the "incomplete" label."""
    trials, steps, regrounds = ab_run()
    report = _call(api("build_report"), trials, steps, regrounds, _probes(), stopped=REASON)
    md = api("render_markdown")(report)
    assert "incomplete" in md and REASON in md, md[:2000]


def test_render_markdown_click_error_two_decimals() -> None:
    """Ruling at 4337125 (and the accepted choice): ``report.md`` prints click-error p50, p95 and mean to two
    decimals, while ``report.json``'s mean keeps its one-decimal rounding. Hand values, 280/off S2: a click 20 px
    below the blue mug's button and one diagonal from its corner (20 sqrt 2 = 28.2842712...): p50 20.0 -> "20.00",
    p95 28.2842712 -> "28.28", the JSON mean round((20 + 28.2842712) / 2, 1) = 24.1 -> "24.10"."""
    clk = {"call_name": "click_at", "executed_name": "click_at"}
    trials = [_trial("S2", r, {"cart": []}) for r in range(2)]
    steps = [
        _step(
            "S2", 0, 1, "executed", 300, 900, click=_click([540, 371], [BLUE], False, 20.0), **clk
        ),
        _step(
            "S2",
            1,
            1,
            "executed",
            300,
            900,
            click=_click([621, 371], [BLUE], False, 20.0 * 2**0.5),
            **clk,
        ),
    ]
    report = api("build_report")(trials, steps, [], _probes())
    err = report["arms"]["280/off"]["clicks"]["error_px"]
    assert err["n"] == 2 and err["p50"] == 20.0 and abs(err["p95"] - 20.0 * 2**0.5) < 1e-9, err
    assert err["mean"] == 24.1, err
    md = api("render_markdown")(report)
    for want in ("20.00", "28.28", "24.10"):
        assert want in md, f"report.md does not print {want}"
    assert "28.284" not in md, "report.md prints the p95 past two decimals"


# =====================================================================================================
# D6: the locked script (the T-0111g rig)
# =====================================================================================================

L_HEAD = "abc1234abc1234abc1234abc1234abc1234abc1"
L_SHORT = "abc1234"
L_OUT = f"logs/t0131-grounding-ab-{L_SHORT}"
L_HARNESS = "scripts/gpu/t0131_grounding_ab.py"
L_EVAL = "tests/eval/test_t0131_ab.py"
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0111g_live_smoke.py",
    "tests/eval/test_live_smoke.py",
)
PROXIES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
STUBBED = (
    "timeout",
    "git",
    "ss",
    "docker",
    "sg",
    "setsid",
    "sleep",
    "uptime",
    "make",
    "curl",
    "nvidia-smi",
    "uv",
)
FORBIDDEN = ("sudo", "systemctl", "podman", "vllm", "wget", "astra")
MERGED_LINE = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0131: tools = live grounding (A-D)"
OTHER_MERGES = (
    "fbb97b5fbb97b5fbb97b5fbb97b5fbb97b5fbb97 Merge T-0111g: tools = live, the browser executor wired",
    "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a Merge T-0131b: ASTRA_VLLM_MM_KWARGS passthrough",
)
ENGINE_CONTAINER = "e0e0e0e0e0e0"

UV_STUB = r"""
for a in "$@"; do printf '%s\n' "$a" >> "$T_UV"; done
printf 'ENV' >> "$T_UV"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY T0131_RUN_DIR; do
  printf ' %s=%s' "$v" "${!v-}" >> "$T_UV"
done
printf '\n--end--\n' >> "$T_UV"
prev=""
for a in "$@"; do
  if [ "$prev" = "--out-dir" ]; then
    mkdir -p "$a" "$a/sessions"
    for f in report.json report.md trials.jsonl steps.jsonl regrounds.jsonl turns.jsonl manifest.json probes.json; do
      : > "$a/$f"
    done
  fi
  prev="$a"
done
if [ -n "$T_FAIL_MATCH" ] && [[ "$*" == *"$T_FAIL_MATCH"* ]]; then exit "$T_FAIL_CODE"; fi
exit 0
"""

GIT_STUB = (
    'case "$*" in\n'
    '  *"merge-base --is-ancestor"*) if [ -n "$T_NOT_MERGED" ]; then exit 1; fi; exit 0 ;;\n'
    "  *--merges*) "
    + "".join(f'echo "{m}"; ' for m in OTHER_MERGES)
    + f'if [ -z "$T_NOT_MERGED" ]; then echo "{MERGED_LINE}"; fi ;;\n'
    '  *--show-toplevel*) echo "$T_TOP" ;;\n'
    f'  *"rev-parse --short"*) echo {L_SHORT} ;;\n'
    f'  *"rev-parse HEAD"*) echo {L_HEAD} ;;\n'
    '  *"status --porcelain"*) if [ -n "$T_DIRTY" ]; then echo " M x.py"; fi ;;\n'
    '  *diff*services/vllm.sh*) if [ -n "$T_VLLM_DRIFT" ]; then echo "+--max-model-len 4096"; fi ;;\n'
    "  *) ;;\nesac\nexit 0"
)

#: ``docker ps`` lists a container before the boot only when the row says so, and the engine's after ``setsid``
DOCKER_STUB = (
    'case "$*" in\n'
    '  *ps*) if [ -n "$T_CONTAINERS" ]; then echo "deadbeef0001 vllm/vllm-openai:v0.29.0";\n'
    f'        elif [ -f "$T_BOOTED" ]; then echo "{ENGINE_CONTAINER}"; fi ;;\n'
    "esac\nexit 0"
)


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0131 D6: {LOCKED.relative_to(REPO)} does not exist yet"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


@dataclass
class UvCall:
    argv: list[str]
    env: dict[str, str]

    def names(self, path: str) -> bool:
        return any(a == path or a.endswith("/" + path) for a in self.argv)

    def flag(self, name: str) -> str | None:
        for i, a in enumerate(self.argv):
            if a == name and i + 1 < len(self.argv):
                return self.argv[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return None


@dataclass
class Result:
    rc: int
    output: str
    uv: list[UvCall]
    calls: list[str]

    def steps(self) -> list[UvCall]:
        return [c for c in self.uv if c.names(L_HARNESS) or c.names(L_EVAL)]

    def started(self) -> list[str]:
        boot = [c for c in self.calls if c.split(" ", 1)[0] in ("make", "setsid", "curl")]
        return boot + [" ".join(c.argv) for c in self.steps()]


def run_locked(
    tmp_path: Path,
    args: tuple[str, ...] = (),
    *,
    dirty: bool = False,
    port_bound: bool = False,
    containers: bool = False,
    vllm_drift: bool = False,
    pre_dirs: tuple[str, ...] = (),
    not_merged: bool = False,
    fail_match: str = "",
    fail_code: int = 1,
) -> Result:
    text = _script().read_text()
    names = "|".join(re.escape(n) for n in STUBBED + FORBIDDEN)
    absolute = re.findall(r"/(?:usr/)?(?:local/)?s?bin/(" + names + r")\b", text)
    if absolute:
        pytest.fail(
            f"the locked script calls {sorted(set(absolute))} by absolute path; it cannot be stubbed"
        )
    top = tmp_path / "top"
    (top / "scripts" / "locked").mkdir(parents=True)
    shutil.copy(LOCKED, top / "scripts" / "locked" / LOCKED.name)
    for rel in L_PLACEHOLDERS:
        (top / rel).parent.mkdir(parents=True, exist_ok=True)
        (top / rel).write_text("# placeholder\n")
    (top / "services").mkdir()
    (top / "services" / "vllm.sh").write_text("# placeholder\n")
    for d in pre_dirs:
        (top / d).mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    calls_log = tmp_path / "calls.log"
    booted = tmp_path / "booted"
    _stub(bin_dir, "git", GIT_STUB)
    _stub(bin_dir, "docker", DOCKER_STUB)
    for quiet in ("sg", "sleep", "uptime", "make", "curl"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(bin_dir, "setsid", ': > "$T_BOOTED"\nexit 0')
    _stub(
        bin_dir,
        "ss",
        'if [ -n "$T_PORT_BOUND" ]; then echo "LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:*"; fi\nexit 0',
    )
    _stub(bin_dir, "nvidia-smi", 'echo "NVIDIA GeForce RTX 5090, 595.84, 1000 MiB, 31000 MiB"')
    _stub(bin_dir, "uv", UV_STUB)
    _stub(bin_dir, "timeout", g_tests.TIMEOUT_STUB)
    for loud in FORBIDDEN:
        _stub(bin_dir, loud, 'echo "unexpected command in the locked-script harness" >&2; exit 97')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "T_CALLS": str(calls_log),
        "T_TOP": str(top),
        "T_UV": str(uv_log),
        "T_BOOTED": str(booted),
        "T_DIRTY": "1" if dirty else "",
        "T_PORT_BOUND": "1" if port_bound else "",
        "T_CONTAINERS": "1" if containers else "",
        "T_VLLM_DRIFT": "1" if vllm_drift else "",
        "T_NOT_MERGED": "1" if not_merged else "",
        "T_FAIL_MATCH": fail_match,
        "T_FAIL_CODE": str(fail_code),
        **dict.fromkeys(PROXIES, "http://proxy.invalid:3128"),
    }
    proc = subprocess.run(
        ["bash", str(top / "scripts" / "locked" / LOCKED.name), *args],
        cwd=top,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    uv: list[UvCall] = []
    if uv_log.is_file():
        argv: list[str] = []
        seen: dict[str, str] = {}
        for line in uv_log.read_text().splitlines():
            if line == "--end--":
                uv.append(UvCall(argv, seen))
                argv, seen = [], {}
            elif line.startswith("ENV"):
                seen = dict(kv.split("=", 1) for kv in line[3:].split())
            else:
                argv.append(line)
    calls = calls_log.read_text().splitlines() if calls_log.is_file() else []
    return Result(proc.returncode, proc.stdout + proc.stderr, uv, calls)


def _code_lines(text: str) -> list[str]:
    return [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(tmp_path: Path, case: str) -> None:
    """D6: each refusal prints ``REFUSED`` and exits 2 before the wheel build, the boot, the health check or any
    step: an argument, a dirty tree (named), an Astra port bound (STT's 8080 here: STT must be down), running
    containers, ``services/vllm.sh`` changed since 806da12, the run's out-dir already there (one run per SHA), and no
    ``Merge T-0131`` merge reachable from HEAD (adversarial: ``Merge T-0111g`` and ``Merge T-0131b`` are in the
    log)."""
    res = run_locked(tmp_path, **LOCKED_REFUSALS[case])  # type: ignore[arg-type]
    assert res.rc == 2, (res.rc, res.output)
    assert "REFUSED" in res.output, res.output
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"
    if case == "dirty-tree":
        assert "x.py" in res.output, f"the dirty-tree refusal does not name the path: {res.output}"


#: T-0111g's boot-regime lines, which D6 reuses verbatim (found in its script by these anchors)
BOOT_ANCHORS = (
    "make plugin-wheel",
    'setsid bash "$PWD/services/vllm.sh"',
    "git diff 806da12..HEAD -- services/vllm.sh",
    "ss -ltn",
    "dock ps -q)",
    "curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then break",
    "NOPROXY=(env",
    "trap stop_engine EXIT",
)


def test_locked_script_boot_regime_verbatim(tmp_path: Path) -> None:
    """D6: T-0111g's boot regime verbatim -- each of its boot lines (wheel, ``setsid`` of ``services/vllm.sh``, the
    806da12 pin, the port and container checks, the health loop, the proxy unset, the stop trap) appears unchanged
    -- with ``services/vllm.sh`` unchanged: started once, with nothing added (no extra argument, no mm-kwargs
    variable: that is T-0131b's contingency), never ``astra up``; ``bash -n`` clean."""
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    mine = _code_lines(_script().read_text())
    theirs = _code_lines(LOCKED_T0111G.read_text())
    missing: list[str] = []
    for anchor in BOOT_ANCHORS:
        lines = [ln for ln in theirs if anchor in ln]
        assert lines, f"premise: T-0111g's script has a line with {anchor!r}"
        missing += [ln for ln in lines if ln not in mine]
    assert missing == [], "T-0111g boot lines not reused verbatim:\n" + "\n".join(missing)
    code = "\n".join(mine)
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"
    assert not re.search(r"MM_KWARGS|mm[-_]processor", code, re.IGNORECASE), (
        "the boot passes mm kwargs"
    )
    assert len([ln for ln in mine if re.search(r"\bsetsid\b", ln)]) == 1, (
        "more than one engine start"
    )
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    boots = [c for c in res.calls if c.startswith("setsid ")]
    assert len(boots) == 1, f"one boot: {boots}"
    words = boots[0].split()
    assert words[-1].endswith("services/vllm.sh"), (
        f"services/vllm.sh started with something added: {boots[0]}"
    )
    missing_unset = [v for v in PROXIES if not re.search(rf"-u\s+{v}\b", code)]
    assert missing_unset == [], f"proxy variables not unset: {missing_unset}"


def test_locked_script_steps_in_order(tmp_path: Path) -> None:
    """D6, the other side of every refusal (merged, clean, nothing bound): boot (wheel, ``setsid``, health), then the
    harness under ``timeout 90m`` with a loopback ``--base-url`` and ``--out-dir logs/t0131-grounding-ab-<SHA>``,
    then the eval ``-m gpu`` with ``T0131_RUN_DIR`` set to that directory; no proxy variable reaches a step; the merge
    check asked of git before the boot; the engine container stopped after the last step; exit 0."""
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    steps = res.steps()

    def first(path: str) -> int:
        found = [i for i, c in enumerate(steps) if c.names(path)]
        assert found, f"no step runs {path}: {[c.argv for c in steps]}\n{res.output}"
        return found[0]

    assert first(L_HARNESS) < first(L_EVAL), [c.argv for c in steps]
    run = steps[first(L_HARNESS)]
    out = run.flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), run.argv
    assert re.match(r"^https?://(127\.0\.0\.1|\[::1\])(:\d+)?/", run.flag("--base-url") or ""), (
        run.argv
    )
    ev = steps[first(L_EVAL)]
    assert "gpu" in (ev.flag("-m") or ""), ev.argv
    assert ev.env.get("T0131_RUN_DIR", "").rstrip("/").endswith(L_OUT), ev.env
    leaked = [(c.argv[-1], v) for c in steps for v in PROXIES if c.env.get(v)]
    assert leaked == [], f"proxy variables reached a step: {leaked}"
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 90 * 60, [
        c for c in res.calls if c.startswith("timeout")
    ]

    calls = res.calls

    def at(pred: str) -> int:
        found = [i for i, c in enumerate(calls) if re.search(pred, c)]
        assert found, f"no call matches {pred!r}: {calls}"
        return found[0]

    uv_steps = [i for i, c in enumerate(calls) if c.startswith("uv ")]
    assert at(r"^make .*plugin-wheel") < at(r"^setsid ") < at(r"^curl .*/health") < uv_steps[0], (
        calls
    )
    asked = at(r"^git (-C \S+ )?(merge-base --is-ancestor|log .*--merges)")
    assert asked < at(r"^make .*plugin-wheel"), "the merge check comes after the boot"
    _assert_engine_stopped(res, after=uv_steps[-1])


def _assert_engine_stopped(res: Result, *, after: int) -> None:
    stops = [
        c
        for c in res.calls[after + 1 :]
        if c.startswith("docker ") and ENGINE_CONTAINER in c and re.search(r"\b(stop|kill|rm)\b", c)
    ]
    assert stops, f"the engine container is not stopped after the last step: {res.calls[after:]}"


@pytest.mark.parametrize(
    ("failing", "code"),
    [(L_HARNESS, 1), (L_HARNESS, 3), (L_EVAL, 1)],
    ids=["harness-1", "harness-probe-3", "eval-1"],
)
def test_locked_script_failing_step_exits_1_and_stops_the_engine(
    tmp_path: Path, failing: str, code: int
) -> None:
    """D6: a failing step (the harness, a failed probe's exit 3, or the eval) makes the script exit 1, never 0, and
    the engine is still stopped on exit."""
    res = run_locked(tmp_path, fail_match=failing, fail_code=code)
    assert res.rc == 1, (res.rc, res.output)
    uv_steps = [i for i, c in enumerate(res.calls) if c.startswith("uv ")]
    assert uv_steps, res.output
    _assert_engine_stopped(res, after=uv_steps[-1])


def test_locked_script_other_run_dirs_do_not_refuse(tmp_path: Path) -> None:
    """The once-per-SHA refusal is not too wide: this A/B's dir of another SHA and T-0111g's dir of this SHA do not
    refuse; the harness runs and the script exits 0."""
    pre = ("logs/t0131-grounding-ab-def5678", f"logs/t0111g-live-smoke-{L_SHORT}")
    res = run_locked(tmp_path, pre_dirs=pre)
    assert res.rc == 0, res.output
    assert any(c.names(L_HARNESS) for c in res.steps()), res.output
