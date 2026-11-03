"""T-0137 group C: the locked 2-arm GPU A/B of ``--repeat-guard`` -- the harness's pure parts, its ``main`` on CPU, its
locked script and C-8 (unit; CPU; no engine, no browser). docs/tasks/T-0137.md C-1...C-8, "Tests to write (Group
C)", reviewer mutation 14, the Rulings (H-T137-6/7).

**Re-targeted onto T-0138's suite** (orchestrator, 2026-10-25). H-T137-6's default foresaw forking the A/B onto T-0138's
suite once it existed; it exists (``scripts/gpu/t0138_suite_ab.py``, S6-S15), and its baseline
(``docs/cd/suite-baseline.md``) shows S9's silent scrolls, where a repeat stop can fire. So these rows pin a fork of
**T-0138's** harness, not T-0151's: arm A HEAD's live default, arm B the same plus ``repeat_guard=True``; 2 x 10 tasks x
10 = 200 turns, joint seed-0 order; paired exact McNemar by ``(task, repeat)``, report-only; C-3's W1 and C-4's voids
kept; C-5 re-expressed on the suite (the eval module's reading 6). The task file still names T-0151's harness, S1-S5,
20 repeats and the S4 clause; the orchestrator owns that file.

BASE_C 1a41145 (the merge of main into ``T-0137-c-ab`` after ``Merge T-0137 group B`` d0f1f02, O-C). **Activated
2026-10-25**: no row is ``pending``; every harness row is red on a named assertion in ``harness()``/``api()``/
``_locked()`` ("does not exist yet" / "has no ..."), never on an import error, until the implementer's fork lands.
Rows parametrised on ``which`` run once against the harness and once against the eval oracle
(``tests/eval/test_t0137_ab.py``, green), so a wrong oracle is caught on CPU. The step-line fields are pinned to group
B's merged output (d0f1f02): the ``repeat_of_step`` mapping (``repeat_fact``) on an executed repeat and on a
``"repeat"`` stop alike (S-R1, reading 2 accepted), ``no_progress``, the turn line's five ``repeat_guard`` counts as
``Session._guard_counts`` defines them, and T-0136's ``outcome`` key in the fact when T-0136 measured step j. Expected values are computed by hand (the comments say how) or by a scratch script of the test-author's
(Wilson by the closed form, sha256 of hand-written JSON), never read off an implementation (none exists). The ported
rule in the oracle is checked against group A's ``progress.judge`` on random histories (green), so the oracle and the
reviewed rule cannot drift apart.

**The assumed harness surface** (the names and shapes are the test-author's, stated once so the implementer can
implement to them or say why not). Unless stated, a name keeps T-0138's harness's shape.

Imported, never edited (C-1)
    As T-0138's harness: T-0134's harness and through it T-0132's, T-0131's and T-0111g's (``LoopMeter``,
    ``sentinel_hits``, ``log_complete``, ``non_loopback_sockets``, ``run_probes``, ``probe_p1``, ``config_evidence``,
    ``wilson``, ``newcombe``); ``TASKS``, ``SUITE_DIR``, ``judge`` from ``local_astra.pages.suite_tasks``;
    ``mcnemar_exact`` from ``local_astra.eval.cd.stats``. T-0138's and T-0151's harnesses are neither imported nor
    loaded nor named in code (fork by copy). The harness may use ``local_astra.conductor.progress`` (the rule itself);
    the eval may not (C-7).

Ported, not imported
    ``turn_records`` (T-0138's port, plus the step-line fields below), ``false_claim``, ``url_leaks``.

Seams ``main`` reads at call time (as T-0138's)
    ``_git``, ``order``, ``run_probes``, ``probe_ps1``, ``probe_ps2``, ``probe_ps3``, ``PROCESS_START_S``,
    ``RUN_BUDGET_S``; ``cli.grounding_of``, ``feedback_of``, ``recovery_of``, ``page_overview_of``,
    ``auto_overview_of``, ``overview_frame_of``, ``action_outcome_of`` (T-0136 C-2) and ``repeat_guard_of`` (T-0137
    B-9), through ``live_default``; ``cli.load_tool_kit``; the attribute ``G``.

The arms (C-1; never hard-coded)
    ``live_default(profile)`` -> T-0138's eight keys plus ``"action_outcome"`` and ``"repeat_guard"``, read from
    ``cli``; ``arm_rows(default) -> [row_A, row_B]``, each ``{"label", <the five defaults>, "degeneracy_guard": True,
    "page_overview": False, "auto_overview": False, "overview_frame": <default's>, "action_outcome": <default's>,
    "repeat_guard": A False / B True, "tools_sha256": V1}``; ``arm_table_sha(rows)``; ``live_default_findings(default)``
    (a default with the guard on names ``repeat_guard`` and "identical arms").

Constants
    ``EXIT_OK/FAILED/REFUSED/PROBE`` 0-3, ``REPEATS = 10``, ``ORDER_SEED = 0``, ``RUN_BUDGET_S = 85 * 60`` (O-C),
    ``PROCESS_START_S``, ``SUITE_TABLE_SHA256``, ``REPEAT_GUARD_ON = {"A": False, "B": True}``,
    ``W1_EXERCISED_MIN = 6``.

Pure functions (each also in the eval oracle, same name and shape; the harness's ``arm_metrics`` is the oracle's
``derive_arm``)
    ``order``, ``arm_rows``, ``arm_table_sha``, ``live_default_findings``; ``ps1_ok``, ``ps2_ok``, ``ps3_ok``;
    ``configuration_findings(trials, steps, regrounds, bodies, arms, *, guidance_sha256)`` (C-4); ``arm_metrics(trials,
    steps)`` (C-3, keys in ``HAND``); ``mcnemar_sections(trials, steps)`` (``{"pooled", "per_task", "unpaired"}``);
    ``decision_inputs(trials, steps, regrounds)``; ``decide_with_reason(inputs, *, p1_passed, void=False,
    stopped=None, configuration_findings=()) -> (label, reason)`` (``inputs`` ``{"A", "B"}``); ``build_report(trials,
    steps, regrounds, bodies, probes, *, arms, guidance_sha256, stopped=None)``; ``render_markdown(report, manifest)``.

Records ``main`` writes: the eval's docstring has every field (T-0138's, plus a step line's ``call_arguments``,
``screen``, ``repeat_of_step``, ``repeat_fact``, ``no_progress``, ``no_change``, ``no_change_source``,
``guard_outcome``, a trial's ``action_outcome``,
``repeat_guard``, ``action_outcome_logged``, ``repeat_guard_logged``, and the manifest's ``guidance_sha256``).

Refusals (exit 2, before any socket, leaving the out-dir as it was)
    T-0138's that still apply (dirty tree, non-loopback URL, out-dir present, repair off, ``LIVE_TOOL_GUIDANCE`` off
    its sha, a kit off v1, the suite table off its pin, ``cli`` without ``overview_frame_of``, a default with
    ReGround, the page overview or the auto overview on), and C-1's: a default with the guard on (named
    ``repeat_guard``, "identical arms"), ``cli`` without ``repeat_guard_of`` or ``action_outcome_of``, ``Session``
    without the ``repeat_guard`` keyword. T-0138's "frame off by default" refusal is gone: the frame follows the
    default (C-1).

**Readings** (reported to the orchestrator; the eval's docstring has the record-level ones)
    1. The re-targeting above (the orchestrator's instruction). C-2's T-0151 probes P6-P9 become T-0138's P-S1/P-S2/
       P-S3; T-0151's S4/S5 metrics and C-5's S4 clause are dropped; over-clicks become T-0138's over-actions.
    2. C-6's bounds are T-0138's, as O-C raised them (``timeout -k 60s 100m``, ``RUN_BUDGET_S = 85*60``; parity with
       T-0136 D).
    3. C-6's "the boot is verbatim from T-0151's script" is read as T-0138 reading 4 (the code lines from ``make
       plugin-wheel`` to ``device memory after boot``, less the lines naming a task), compared with T-0151's script.
    4. C-6's merge check accepts a ``Merge T-0137 group C`` subject only; ``Merge T-0137 group A``/``group B``, the
       ``-impl``/``-fix`` merges, branch merges, a bare ``Merge T-0137:`` and ``Merge T-0137c`` do not count.
    5. C-8 also pins T-0138's harness, locked script and eval by content (the fork's parent and the oracle this eval
       imports), beside T-0151's, T-0150's and T-0140's.
    6. BASE_C is 1a41145, the merge that took group B in (O-C). If main is merged into this branch again, BASE_C
       moves to that merge, or the merged ``local_astra`` changes read as group C's. That applies to a merge
       *before* TIP_C; main merged after the implementation tip (4adafae, after TIP_C 9b9bc0a) is outside
       ``BASE_C..TIP_C`` already, so BASE_C stays.

Regime: CPU, pure, hand-built records; the dress row runs ``main`` on CPU with T-0131's engine rig and the
``FakeLiveBrowser`` (fake LLM, one S9 turn per arm, four identical clicks on one unchanging screen); no engine, no
browser, no measurement, n = 1.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import itertools
import random
import re
import socket
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.conductor import progress
from local_astra.pages import suite_tasks
from tests import _git_range as gr
from tests.eval import test_t0137_ab as oracle
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0138_ab_harness as t38

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0137_repeat_guard_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0137_repeat_guard_ab.sh"
EVAL = REPO / "tests" / "eval" / "test_t0137_ab.py"
T0138_HARNESS = REPO / "scripts" / "gpu" / "t0138_suite_ab.py"
T0151_HARNESS = REPO / "scripts" / "gpu" / "t0151_overview_frame_ab.py"
T0151_LOCKED = REPO / "scripts" / "locked" / "t0151_overview_frame_ab.sh"
T0111G_HARNESS = t38.T0111G_HARNESS
T0131_HARNESS = t38.T0131_HARNESS
T0132_HARNESS = t38.T0132_HARNESS
#: group C's BASE (O-C, 2026-10-25): 1a41145, the merge of main (``Merge T-0137 group B`` d0f1f02, bookkeeping 0e2e0fc)
#: into this branch. A literal, never a moving ref.
BASE_C = "1a41145"
#: group C's tip: **bumped by the test-author to the implementation tip before review**; until then it is BASE_C and
#: the scope row is red on its bump assertion. The row compares ``BASE_C..TIP_C``, two literals (T-0094).
#: Bumped 2026-10-25 to 9b9bc0a (harness; locked script a5a5c4a). The later merge of main 4adafae (T-0136 D) sits
#: after TIP_C, so BASE_C stays 1a41145: 9b9bc0a is not a descendant of 4adafae, and the closed range
#: 1a41145..9b9bc0a holds group C's own commits only (13fd9b5, 0db6846, a5a5c4a, 9b9bc0a), no merged main change.
TIP_C = "9b9bc0a"

#: activated 2026-10-25 (O-C): no row carries ``pending`` any more; the harness rows are red on "does not exist yet"
IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

V1 = t38.V1
OV_SHA = t38.OV_SHA
GUIDANCE_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
SUITE_TABLE_SHA256 = t38.SUITE_TABLE_SHA256
TASK_IDS = t38.TASK_IDS
REQUESTS = t38.REQUESTS
CENSUS = t38.CENSUS
LABEL_FB = t38.LABEL_FB
SCREEN_NOW = t38.SCREEN_NOW
#: A-1's constants and reasons, typed from docs/tasks/T-0137.md (never imported)
REPEAT_REASON = (
    "the same action did not change this screen the last two times; it was not done again"
)
RECURRING_REASON = (
    "this screen has come back for the third time in this request; the action was not done"
)
FRESHNESS_REASON = "the screen changed since this frame; take a new frame"
RG_KEYS = ("repeats", "no_change_repeats", "stops_repeat", "stops_recurring", "max_screen_visits")

D_HEAD = t38.D_HEAD
D_OLD = t38.D_OLD
#: the followed and overview keys of HEAD's default after T-0152 (frame on) with T-0136's label off and the guard off
HEAD_REST = {
    "page_overview": False,
    "auto_overview": False,
    "overview_frame": True,
    "action_outcome": False,
    "repeat_guard": False,
}
#: another runnable default: the frame off and T-0136's label on (both follow the default, C-1)
OTHER_REST = {**HEAD_REST, "overview_frame": False, "action_outcome": True}


def _row(label: str, default: dict[str, Any], rest: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        **default,
        "degeneracy_guard": True,
        "page_overview": False,
        "auto_overview": False,
        "overview_frame": rest["overview_frame"],
        "action_outcome": rest["action_outcome"],
        "repeat_guard": label == "B",
        "tools_sha256": V1,
    }


#: ``arm_table_sha`` of each default's two rows (written out in ``_row``), by the test-author 2026-10-25 (hashlib over
#: the canonical JSON of the hand-written rows, in a scratch script)
ARM_PINS = {
    "head": (D_HEAD, HEAD_REST, "9cdd85bcff321cfc92b28eb1634cf76e8929188073e29f66323443107a7df521"),
    "other": (
        D_OLD,
        OTHER_REST,
        "09265bb983f4d379dc39099e0d9ad67b4c372018c3c9b5ec5cde7f8e8655299c",
    ),
}
#: the arms the hand-built records ran at
HAND_ARMS = [_row("A", D_HEAD, HEAD_REST), _row("B", D_HEAD, HEAD_REST)]


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0137 C-1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0137_repeat_guard_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0137: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api("arm_metrics" if name == "derive_arm" else name)
    return getattr(oracle, name)


def _cli_has(cli: Any, *names: str) -> None:
    for name in names:
        assert hasattr(cli, name), f"T-0137 B-9 / T-0136 C-2: cli.{name} does not exist yet"


# =====================================================================================================
# C-1: imports and constants
# =====================================================================================================

FROM = t38.FROM
PORTED = ("turn_records", "false_claim", "url_leaks")


def test_imports_kept_and_ports_defined() -> None:
    """C-1: importing the harness opens no socket; T-0134's chain is imported read-only; ``TASKS``, ``SUITE_DIR``,
    ``judge`` and ``mcnemar_exact`` are the package's objects; ``turn_records``, ``false_claim`` and ``url_leaks`` are
    defined in the harness file; T-0138's and T-0151's harnesses are neither loaded nor named in code (a fork by copy,
    C-1 "not edited")."""
    from local_astra.eval.cd import stats

    before = set(sys.modules)
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    for origin, names in FROM.items():
        for name in names:
            src = Path(inspect.getsourcefile(api(name)) or "").resolve()
            assert src == origin.resolve(), (
                f"T-0137 C-1: {name} comes from {src}, not {origin.name}"
            )
    assert api("TASKS") is suite_tasks.TASKS and api("judge") is suite_tasks.judge
    assert Path(api("SUITE_DIR")) == suite_tasks.SUITE_DIR
    assert api("mcnemar_exact") is stats.mcnemar_exact
    for name in PORTED:
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == HARNESS.resolve(), (
            f"T-0137 C-1: {name} is not ported into the harness ({src})"
        )
    added = [m for k in set(sys.modules) - before if (m := sys.modules.get(k)) is not None]
    for m in [*added, *(v for v in vars(h).values() if isinstance(v, ModuleType))]:
        f = getattr(m, "__file__", None)
        if f is None:
            continue
        assert Path(f).resolve() not in (T0138_HARNESS.resolve(), T0151_HARNESS.resolve()), (
            f"T-0137 C-1: {m.__name__} is a frozen harness ({f})"
        )
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    named = [
        s
        for s in t38._non_docstring_strings(tree)
        if "t0138_suite_ab" in s or "t0151_overview_frame_ab" in s
    ]
    assert named == [], f"T-0137 C-1: the harness names a frozen harness in code: {named}"


def test_constants() -> None:
    """C-1/C-5/C-6 (re-targeted, readings 1-2): the constants."""
    assert api("SUITE_TABLE_SHA256") == SUITE_TABLE_SHA256 == suite_tasks.task_table_sha()
    assert api("REPEATS") == 10 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 85 * 60, "T-0137 C-6 (O-C): RUN_BUDGET_S = 85*60"
    assert isinstance(api("PROCESS_START_S"), float) and api("PROCESS_START_S") <= time.monotonic()
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )
    assert dict(api("REPEAT_GUARD_ON")) == {"A": False, "B": True}, "C-1: B is A plus the guard"
    assert api("W1_EXERCISED_MIN") == 6, "C-5 (H-T137-7): W1(A) >= 6 to be exercised"


def test_oracle_constants() -> None:
    """The oracle's typed copies (green): the rule's constants and reasons are A-1's and equal group A's module (the
    port and the rule agree on their inputs), the guidance sha is B-OFF's, the margins are C-5's (reading 6)."""
    assert (oracle.REPEAT_TOLERANCE, oracle.MAX_NO_CHANGE_REPEATS, oracle.MAX_SCREEN_VISITS) == (
        10,
        2,
        3,
    )
    assert (
        oracle.ACTIONS
        == frozenset({"click_at", "type_at", "scroll", "press_key"})
        == progress.ACTIONS
    )
    assert oracle.NO_PROGRESS_REPEAT_REASON == REPEAT_REASON == progress.NO_PROGRESS_REPEAT_REASON
    assert (
        oracle.NO_PROGRESS_RECURRING_REASON
        == RECURRING_REASON
        == progress.NO_PROGRESS_RECURRING_REASON
    )
    from local_astra.conductor import prompt_builder
    from local_astra.conductor import session as session_mod

    assert oracle.FRESHNESS_REASON == FRESHNESS_REASON == session_mod.FRESHNESS_REASON
    assert (
        oracle.LIVE_TOOL_GUIDANCE_SHA256
        == GUIDANCE_SHA
        == hashlib.sha256(prompt_builder.LIVE_TOOL_GUIDANCE.encode()).hexdigest()
    )
    assert (oracle.W1_EXERCISED_MIN, oracle.W1_RATIO) == (6, 0.5)
    assert (oracle.SUCCESS_MARGIN, oracle.FALSE_CLAIM_MARGIN, oracle.OVER_ACTION_MARGIN) == (
        0.03,
        2,
        2,
    )
    assert oracle.ENV == "T0137_RUN_DIR"


# =====================================================================================================
# C-1: arms (read from cli), refusals, order
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("pin", list(ARM_PINS))
def test_arm_rows_and_sha(which: str, pin: str) -> None:
    """C-1: over a live default, the two rows carry its five settings, its overview frame and its T-0136 label, the
    T-0142 guard on, the page and auto overviews off and tools v1; A without the repeat guard, B with it; the sha is
    the pin; swapping the rows or flipping either arm's guard gives another sha."""
    default, rest, want_sha = ARM_PINS[pin]
    rows = impl(which, "arm_rows")({**default, **rest})
    assert [dict(r) for r in rows] == [_row("A", default, rest), _row("B", default, rest)], rows
    sha = impl(which, "arm_table_sha")
    assert sha(rows) == want_sha
    assert sha([rows[1], rows[0]]) != want_sha
    assert sha([rows[0], {**rows[1], "repeat_guard": False}]) != want_sha
    assert sha([{**rows[0], "repeat_guard": True}, rows[1]]) != want_sha


def _heads_default(cli: Any, lean: Any) -> dict[str, Any]:
    """HEAD's live default as ``cli`` gives it now (never a literal), with T-0136's and T-0137's resolvers."""
    _cli_has(cli, "repeat_guard_of", "action_outcome_of")
    d = t38._heads_default(cli, lean)
    ns = argparse.Namespace()
    d["action_outcome"] = cli.action_outcome_of(lean, ns, "live").action_outcome
    d["repeat_guard"] = cli.repeat_guard_of(lean, ns, "live").repeat_guard
    return d


def test_heads_default_has_the_guard_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise of C-1 (green once group B merged): HEAD's live default read from ``cli`` has the guard off and
    ReGround, the page and the auto overview off, and the oracle accepts it; with ``cli``'s guard default set on the
    oracle refuses it, naming ``repeat_guard`` and "identical arms"."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    # T-0179 A-5: the live default is on since T-0179 A; the C-1 premise is the off-arm default the A/B ran under
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", False)
    d = _heads_default(cli, lean)
    assert d["repeat_guard"] is False, d
    assert oracle.live_default_findings(d) == [], oracle.live_default_findings(d)
    _cli_has(cli, "LIVE_DEFAULT_REPEAT_GUARD")
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", True)
    on = _heads_default(cli, lean)
    assert on["repeat_guard"] is True, "premise: cli reads its guard default at call time (B-9)"
    found = oracle.live_default_findings(on)
    assert any("repeat_guard" in f and "identical arms" in f for f in found), found


@pytest.mark.parametrize(
    "over, needle",
    [
        ({}, None),
        ({"overview_frame": False}, None),
        ({"action_outcome": True}, None),
        ({"overview_frame": False, "action_outcome": True}, None),
        ({"repeat_guard": True}, "repeat_guard"),
        ({"repeat_guard": None}, "repeat_guard"),
        ({"reground": True}, "reground"),
        ({"page_overview": True}, "page_overview"),
        ({"auto_overview": True}, "auto_overview"),
        ({"overview_frame": None}, "overview_frame"),
        ({"action_outcome": "on"}, "action_outcome"),
    ],
)
@pytest.mark.parametrize("which", IMPLS)
def test_live_default_findings(which: str, over: dict[str, Any], needle: str | None) -> None:
    """C-1: a default with the guard on (or not a boolean) is refused as "identical arms"; ReGround, the page or the
    auto overview on is refused (tools v1, T-0138 B2); the overview frame and T-0136's label are accepted either way
    (they follow HEAD's default; **a T-0138 copy that refuses a frame-off default is too wide**) but must be
    booleans; a default without ``repeat_guard`` or ``action_outcome`` is malformed."""
    fn = impl(which, "live_default_findings")
    found = fn({**D_HEAD, **HEAD_REST, **over})
    if needle is None:
        assert found == [], found
    else:
        assert found and any(needle in f for f in found), found
    if needle == "repeat_guard":
        assert any("identical arms" in f for f in found), found
    for missing in ("repeat_guard", "action_outcome"):
        d = {k: v for k, v in {**D_HEAD, **HEAD_REST}.items() if k != missing}
        assert fn(d) == [f"live_default lacks {missing}"], (missing, fn(d))


def test_live_default_follows_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-1 "read from cli at run time": ``live_default`` equals ``cli``'s HEAD default, then follows a monkeypatched
    ``cli.LIVE_DEFAULT_REPEAT_GUARD``, ``cli.repeat_guard_of``, ``cli.action_outcome_of`` and the frame's default
    (each read at call time)."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    fn = api("live_default")
    assert dict(fn(lean)) == _heads_default(cli, lean)
    _cli_has(cli, "LIVE_DEFAULT_REPEAT_GUARD", "RepeatGuard", "ActionOutcomeChoice")
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", True)
    assert dict(fn(lean))["repeat_guard"] is True, (
        "live_default must read the guard default at call time"
    )
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", False)
    monkeypatch.setattr(cli, "repeat_guard_of", lambda p, a, m, **k: cli.RepeatGuard(True, "flag"))
    assert dict(fn(lean))["repeat_guard"] is True, "live_default must read cli.repeat_guard_of"
    monkeypatch.setattr(
        cli, "action_outcome_of", lambda p, a, m, **k: cli.ActionOutcomeChoice(True, "flag")
    )
    assert dict(fn(lean))["action_outcome"] is True, "live_default must read cli.action_outcome_of"
    monkeypatch.setattr(cli, "LIVE_DEFAULT_OVERVIEW_FRAME", False)
    assert dict(fn(lean))["overview_frame"] is False, (
        "live_default must read the frame at call time"
    )


@pytest.mark.parametrize("which", IMPLS)
def test_order_200_joint_seed0(which: str) -> None:
    """C-1 (re-targeted): 200 turns, each arm 100, each (arm, task) 10 with repeats 0-9; deterministic; seed-dependent;
    the arms interleave; T-0138's construction, so the same list as T-0138's order (its eval's)."""
    order = impl(which, "order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)] and turns != [tuple(t) for t in order(1)]
    assert len(turns) == 200
    assert Counter(t[0] for t in turns) == {"A": 100, "B": 100}
    per = Counter((t[0], t[1]) for t in turns)
    assert set(per) == {(a, k) for a in "AB" for k in TASK_IDS} and set(per.values()) == {10}
    assert {t[2] for t in turns} == set(range(10))
    switches = sum(1 for x, y in itertools.pairwise(turns) if x[0] != y[0])
    assert switches > 60, f"the arms run in blocks ({switches} switches)"
    assert turns == t38.oracle.order(0)


def test_harness_order_equals_oracle() -> None:
    assert [tuple(t) for t in api("order")(0)] == oracle.order(0)


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """T-0138's refusal rig over this harness: every socket refused and recorded; a clean tree through ``_git``;
    stand-in kits (v1 unless ``page_overview=True``, repair on); Chromium reported present; ``cli`` at HEAD's real
    default."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0137 refusal rig: no {what}")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else t31.GIT + "\n"

    assert callable(getattr(h, "_git", None)), "the harness has no module-level _git"
    monkeypatch.setattr(h, "_git", git)
    kits = t38._stand_in_kits()

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        return (kits[bool(page_overview)], None)

    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kits=kits,
        sockets=sockets,
        status=status,
        out=tmp_path / "out",
        mp=monkeypatch,
    )


REFUSALS = (
    "dirty-tree",
    "url-not-loopback",
    "out-dir-exists",
    "kit-repair-off",
    "guidance-off-its-sha",
    "kit-not-v1",
    "suite-table-off-its-pin",
    "guard-default-on",
    "guard-resolver-on",
    "reground-on-by-default",
    "page-overview-on-by-default",
    "auto-overview-on-by-default",
    "cli-without-overview-frame-of",
    "cli-without-repeat-guard-of",
    "cli-without-action-outcome-of",
    "session-without-repeat-guard-keyword",
)
REFUSAL_NAMES = {
    "guard-default-on": r"repeat_guard.*identical arms|identical arms.*repeat_guard",
    "guard-resolver-on": r"repeat_guard.*identical arms|identical arms.*repeat_guard",
    "reground-on-by-default": r"reground",
    "page-overview-on-by-default": r"page[ _]overview",
    "auto-overview-on-by-default": r"auto[ _]overview",
    "cli-without-overview-frame-of": r"overview_frame_of",
    "cli-without-repeat-guard-of": r"repeat_guard_of",
    "cli-without-action-outcome-of": r"action_outcome_of",
    "session-without-repeat-guard-keyword": r"repeat_guard",
    "suite-table-off-its-pin": r"table|sha",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, case: str
) -> None:
    """C-1 ("refused if that default has the guard on"; the ``cli`` reads at run time): T-0138's refusals that still
    apply, and a default with the guard on -- ``cli.LIVE_DEFAULT_REPEAT_GUARD`` True, or the resolver giving on --
    refused as "identical arms" naming ``repeat_guard``; ``cli`` without either new resolver and ``Session`` without
    the keyword refused naming it. Each exits 2 before any socket, leaving the out-dir as it was."""
    from local_astra.conductor import prompt_builder
    from local_astra.conductor import session as session_mod

    rig = _rig(tmp_path, monkeypatch)
    h, cli = rig.h, rig.cli
    argv = ["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)]
    real_g = cli.grounding_of
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
    elif case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "out-dir-exists":
        rig.out.mkdir()
    elif case == "kit-repair-off":
        rig.kits[False].repair_off = "payload stale"
        rig.kits[True].repair_off = "payload stale"
    elif case == "guidance-off-its-sha":
        rig.mp.setattr(
            prompt_builder, "LIVE_TOOL_GUIDANCE", prompt_builder.LIVE_TOOL_GUIDANCE + " "
        )
    elif case == "kit-not-v1":
        rig.kits[False].tools_version = OV_SHA
    elif case == "suite-table-off-its-pin":
        moved = (
            dataclasses.replace(suite_tasks.TASKS[0], request="select seat F15"),
            *suite_tasks.TASKS[1:],
        )
        rig.mp.setattr(suite_tasks, "TASKS", moved)
        if hasattr(h, "TASKS"):
            rig.mp.setattr(h, "TASKS", moved)
        assert suite_tasks.task_table_sha() != SUITE_TABLE_SHA256, "premise: the table moved"
    elif case == "guard-default-on":
        _cli_has(cli, "LIVE_DEFAULT_REPEAT_GUARD")
        rig.mp.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", True)
    elif case == "guard-resolver-on":
        _cli_has(cli, "RepeatGuard")
        rig.mp.setattr(
            cli, "repeat_guard_of", lambda p, a, m, **k: cli.RepeatGuard(True, "default")
        )
    elif case == "reground-on-by-default":
        rig.mp.setattr(
            cli, "grounding_of", lambda p, a, m: dataclasses.replace(real_g(p, a, m), reground=True)
        )
    elif case == "page-overview-on-by-default":
        rig.mp.setattr(cli, "page_overview_of", lambda p, a, m: cli.PageOverview(True, "default"))
    elif case == "auto-overview-on-by-default":
        rig.mp.setattr(cli, "auto_overview_of", lambda p, a, m: cli.AutoOverview(True, "default"))
    elif case == "cli-without-overview-frame-of":
        rig.mp.delattr(cli, "overview_frame_of")
    elif case == "cli-without-repeat-guard-of":
        _cli_has(cli, "repeat_guard_of")
        rig.mp.delattr(cli, "repeat_guard_of")
    elif case == "cli-without-action-outcome-of":
        _cli_has(cli, "action_outcome_of")
        rig.mp.delattr(cli, "action_outcome_of")
    elif case == "session-without-repeat-guard-keyword":
        params = inspect.signature(session_mod.Session.__init__).parameters
        assert "repeat_guard" in params, "T-0137 B-1: Session has no repeat_guard keyword yet"
        rig.mp.setattr(session_mod, "Session", t38._session_without(session_mod, "repeat_guard"))
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], rig.sockets
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in REFUSAL_NAMES:
        assert re.search(REFUSAL_NAMES[case], err, re.IGNORECASE | re.DOTALL), (
            f"{case}: the refusal does not name {REFUSAL_NAMES[case]!r}: {err!r}"
        )


@pytest.mark.parametrize("frame", [True, False])
def test_rig_at_heads_default_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, frame: bool
) -> None:
    """The other side of every refusal (a guard too wide): at HEAD's live default -- and with the overview frame's
    default off, which T-0138's harness refuses by design and this one must not (C-1: the frame follows the default)
    -- ``main`` does not refuse; it looks for the engine (a socket attempt) and, finding none, exits 1."""
    rig = _rig(tmp_path, monkeypatch)
    rig.mp.setattr(
        rig.cli, "LIVE_DEFAULT_REPEAT_GUARD", False
    )  # T-0179 A-5: the harness's own refusal of "on"
    if not frame:
        rig.mp.setattr(rig.cli, "LIVE_DEFAULT_OVERVIEW_FRAME", False)
    rc = rig.h.main(["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)])
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


# =====================================================================================================
# C-2: probes (T-0138's, re-targeted)
# =====================================================================================================

PROBE_ROWS = {
    "ps1-ok": ("ps1_ok", t38.PS1_OK, True),
    "ps1-s7-fits": ("ps1_ok", t38.PS1_ROWS["s7-fits"][0], False),
    "ps1-s8-too-tall-2.49": ("ps1_ok", t38.PS1_ROWS["s8-too-tall-2.49"][0], False),
    "ps2-ok": ("ps2_ok", t38.PS2_OK, True),
    "ps2-s14-trap-passes": ("ps2_ok", t38.PS2_ROWS["s14-trap-passes"][0], False),
    "ps2-s9-golden-fails": ("ps2_ok", t38.PS2_ROWS["s9-golden-fails"][0], False),
    "ps2-s9-trap-raised": ("ps2_ok", t38.PS2_ROWS["s9-trap-raised"][0], False),
    "ps3-ok": ("ps3_ok", t38.PS3_OK, True),
    "ps3-egress": ("ps3_ok", t38.PS3_ROWS["one-egress-record"][0], False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(PROBE_ROWS))
def test_probe_rules(which: str, row: str) -> None:
    """C-2 (re-targeted, reading 1): P-S1, P-S2 and P-S3 judged by T-0138's rules from T-0138's hand records."""
    name, record, want = PROBE_ROWS[row]
    assert impl(which, name)(copy.deepcopy(record)) is want, row


@pytest.mark.parametrize(
    "probe, record",
    [
        ("p2", {"http_status": 500, "healthy_after": True}),
        ("ps1", t38.PS1_ROWS["s7-fits"][0]),
        ("ps2", t38.PS2_ROWS["s14-trap-passes"][0]),
        ("ps3", t38.PS3_ROWS["one-egress-record"][0]),
    ],
)
def test_probe_verdicts_void_on_each_probe(probe: str, record: dict[str, Any]) -> None:
    """C-2 "any other probe failing voids the run" (oracle, green): over passing probes the run is not void; one
    failing record fails that probe alone and voids the run."""
    ok = {**t31._probes(), "ps1": t38.PS1_OK, "ps2": t38.PS2_OK, "ps3": t38.PS3_OK}
    names = ("p1", "p2", "p3", "ps1", "ps2", "ps3")
    assert oracle.probe_verdicts(copy.deepcopy(ok)) == dict.fromkeys(names, True) | {"void": False}
    got = oracle.probe_verdicts(copy.deepcopy({**ok, probe: record}))
    assert got[probe] is False and got["void"] is True, got
    assert all(got[p] for p in names if p != probe), got


# =====================================================================================================
# The ported rule (C-3 W1, C-4, C-7)
# =====================================================================================================


def _c(
    step: int,
    name: str,
    args: dict[str, Any],
    screen: str | None,
    status: str = "executed",
    *,
    overview: bool = False,
    no_change: bool | None = True,
) -> dict[str, Any]:
    return {
        "step": step,
        "name": name,
        "arguments": dict(args),
        "screen": screen,
        "status": status,
        "overview": overview,
        "no_change": no_change,
    }


def _xy(x: int, y: int) -> dict[str, int]:
    return {"x": x, "y": y}


DOWN = {"direction": "down"}
#: (history, proposed, (repeat_of, stop, no_change_repeats, screen_visits)), each by hand from A-3...A-6 and O-A2
RULE_TABLE: dict[str, tuple[list[dict[str, Any]], dict[str, Any], tuple[Any, Any, int, int]]] = {
    "click-within-10-each-axis": (
        [_c(1, "click_at", _xy(100, 100), "s")],
        _c(2, "click_at", _xy(110, 90), "s"),
        (1, None, 0, 1),
    ),
    "click-11-on-x-is-another-action": (
        [_c(1, "click_at", _xy(100, 100), "s")],
        _c(2, "click_at", _xy(111, 100), "s"),
        (None, None, 0, 1),
    ),
    "click-11-on-y-is-another-action": (
        [_c(1, "click_at", _xy(100, 100), "s")],
        _c(2, "click_at", _xy(100, 89), "s"),
        (None, None, 0, 1),
    ),
    "click-on-another-screen": (
        [_c(1, "click_at", _xy(100, 100), "s")],
        _c(2, "click_at", _xy(100, 100), "t"),
        (None, None, 0, 1),
    ),
    "click-on-a-blind-screen": (
        [_c(1, "click_at", _xy(100, 100), None)],
        _c(2, "click_at", _xy(100, 100), None),
        (None, None, 0, 0),
    ),
    "overview-step-is-never-an-instance": (
        [_c(1, "click_at", _xy(160, 933), "q", overview=True)],
        _c(2, "click_at", _xy(160, 933), "q"),
        (None, None, 0, 1),
    ),
    "type-press-enter-missing-equals-false": (
        [_c(1, "type_at", {"x": 300, "y": 200, "text": "Grace Hopper"}, "f")],
        _c(2, "type_at", {"x": 300, "y": 200, "text": "Grace Hopper", "press_enter": False}, "f"),
        (1, None, 0, 1),
    ),
    "type-press-enter-true-differs": (
        [_c(1, "type_at", {"x": 300, "y": 200, "text": "Grace Hopper"}, "f")],
        _c(2, "type_at", {"x": 300, "y": 200, "text": "Grace Hopper", "press_enter": True}, "f"),
        (None, None, 0, 1),
    ),
    "type-other-text-differs": (
        [_c(1, "type_at", {"x": 300, "y": 200, "text": "Grace"}, "f")],
        _c(2, "type_at", {"x": 300, "y": 200, "text": "Hopper"}, "f"),
        (None, None, 0, 1),
    ),
    "scroll-other-direction-differs": (
        [_c(1, "scroll", DOWN, "p")],
        _c(2, "scroll", {"direction": "up"}, "p"),
        (None, None, 0, 1),
    ),
    "press-key-other-combo-differs": (
        [_c(1, "press_key", {"combo": "Tab"}, "p")],
        _c(2, "press_key", {"combo": "Enter"}, "p"),
        (None, None, 0, 1),
    ),
    "search-is-never-a-repeat-nor-stopped": (
        [_c(1, "google_search", {"query": "kettle"}, "p")] * 1
        + [_c(k, "scroll", DOWN, "p") for k in (2, 3, 4)],
        _c(5, "google_search", {"query": "kettle"}, "p"),
        (None, None, 0, 1),
    ),
    # A-6: three executed no-change instances; the fourth is refused (the original never counts)
    "fourth-identical-no-change-stops": (
        [_c(k, "scroll", DOWN, "p") for k in (1, 2, 3)],
        _c(4, "scroll", DOWN, "p"),
        (3, "repeat", 2, 1),
    ),
    "third-identical-executes": (
        [_c(k, "scroll", DOWN, "p") for k in (1, 2)],
        _c(3, "scroll", DOWN, "p"),
        (2, None, 1, 1),
    ),
    "refused-and-failed-are-never-instances": (
        [
            _c(1, "scroll", DOWN, "p"),
            _c(2, "scroll", DOWN, "p", "refused"),
            _c(3, "scroll", DOWN, "p", "failed"),
            _c(4, "scroll", DOWN, "p"),
        ],
        _c(5, "scroll", DOWN, "p"),
        (4, None, 1, 1),
    ),
    # O-A2: the trailing run; a changed repeat resets it
    "changed-last-resets": (
        [
            _c(1, "scroll", DOWN, "p"),
            _c(2, "scroll", DOWN, "p"),
            _c(3, "scroll", DOWN, "p"),
            _c(4, "scroll", DOWN, "p", no_change=False),
        ],
        _c(5, "scroll", DOWN, "p"),
        (4, None, 0, 1),
    ),
    "changed-first-then-two-no-change-stops": (
        [
            _c(1, "scroll", DOWN, "p"),
            _c(2, "scroll", DOWN, "p", no_change=False),
            _c(3, "scroll", DOWN, "p"),
            _c(4, "scroll", DOWN, "p"),
        ],
        _c(5, "scroll", DOWN, "p"),
        (4, "repeat", 2, 1),
    ),
    "unknown-resets": (
        [
            _c(1, "scroll", DOWN, "p"),
            _c(2, "scroll", DOWN, "p"),
            _c(3, "scroll", DOWN, "p", no_change=None),
        ],
        _c(4, "scroll", DOWN, "p"),
        (3, None, 0, 1),
    ),
    # A-5: runs, not occurrences
    "third-visit-recurs": (
        [
            _c(1, "click_at", _xy(100, 100), "a"),
            _c(2, "click_at", _xy(200, 200), "b"),
            _c(3, "click_at", _xy(300, 300), "a"),
            _c(4, "click_at", _xy(400, 400), "b"),
        ],
        _c(5, "click_at", _xy(500, 500), "a"),
        (None, "recurring", 0, 3),
    ),
    "second-visit-does-not": (
        [
            _c(1, "click_at", _xy(100, 100), "a"),
            _c(2, "click_at", _xy(200, 200), "b"),
            _c(3, "click_at", _xy(300, 300), "a"),
        ],
        _c(4, "click_at", _xy(500, 500), "a"),
        (None, None, 0, 2),
    ),
    "form-fill-is-one-visit": (
        [
            _c(1, "type_at", {"x": 300, "y": 200, "text": "Grace Hopper"}, "f", no_change=False),
            _c(2, "type_at", {"x": 300, "y": 260, "text": "4"}, "f", no_change=False),
            _c(3, "type_at", {"x": 300, "y": 320, "text": "19:30"}, "f", no_change=False),
        ],
        _c(4, "click_at", _xy(300, 400), "f"),
        (None, None, 0, 1),
    ),
    "refused-steps-still-visit": (
        [
            _c(1, "click_at", _xy(100, 100), "a"),
            _c(2, "click_at", _xy(200, 200), "b"),
            _c(3, "click_at", _xy(300, 300), "a", "refused"),
            _c(4, "click_at", _xy(400, 400), "b"),
        ],
        _c(5, "click_at", _xy(600, 600), "a"),
        (None, "recurring", 0, 3),
    ),
    "repeat-takes-precedence": (
        [
            _c(1, "scroll", DOWN, "a"),
            _c(2, "click_at", _xy(1, 1), "b"),
            _c(3, "scroll", DOWN, "a"),
            _c(4, "click_at", _xy(2, 2), "b"),
            _c(5, "scroll", DOWN, "a"),
        ],
        _c(6, "scroll", DOWN, "a"),
        (5, "repeat", 2, 3),
    ),
}


@pytest.mark.parametrize("row", list(RULE_TABLE))
def test_ported_rule_table(row: str) -> None:
    """C-3/C-7 (oracle, green): the eval's ported rule, by hand from A-3...A-6 and O-A2 -- the tolerance at 10 vs 11
    on each axis, screens, blind, overview, ``press_enter`` missing as ``False``, text, direction, combo, non-actions,
    the stop on the fourth (not the third), refused/failed never instances, the trailing run, visits by runs (a form
    fill is one visit; a refused step still visits), and ``repeat`` over ``recurring``."""
    history, call, (repeat_of, stop, ncr, visits) = RULE_TABLE[row]
    got = oracle.rule_verdict(copy.deepcopy(history), dict(call))
    assert (got["repeat_of"], got["stop"], got["no_change_repeats"], got["screen_visits"]) == (
        repeat_of,
        stop,
        ncr,
        visits,
    ), (row, got)


def _to_step(c: dict[str, Any]) -> progress.Step:
    return progress.Step(
        step=c["step"],
        name=c["name"],
        arguments=c["arguments"],
        screen=c["screen"],
        status=c["status"],
        overview=c["overview"],
        no_change=c["no_change"],
        outcome={},
        action=None,
    )


_SCREENS = st.sampled_from(["a", "b", "c", None])
_NAMES = st.sampled_from(["click_at", "type_at", "scroll", "press_key", "google_search"])


@st.composite
def _calls(draw: Any, step: int) -> dict[str, Any]:
    name = draw(_NAMES)
    if name in ("click_at", "type_at"):
        args: dict[str, Any] = {
            "x": draw(st.sampled_from([100, 105, 110, 111, 89])),
            "y": draw(st.sampled_from([200, 190, 211])),
        }
        if name == "type_at":
            args["text"] = draw(st.sampled_from(["a", "b"]))
            enter = draw(st.sampled_from([None, False, True]))
            if enter is not None:
                args["press_enter"] = enter
    elif name == "scroll":
        args = {"direction": draw(st.sampled_from(["down", "up"]))}
    elif name == "press_key":
        args = {"combo": draw(st.sampled_from(["Tab", "Enter"]))}
    else:
        args = {"query": "q"}
    return _c(
        step,
        name,
        args,
        draw(_SCREENS),
        draw(st.sampled_from(["executed", "executed", "refused", "failed"])),
        overview=draw(st.booleans()) and step == 1,
        no_change=draw(st.sampled_from([True, True, False, None])),
    )


@st.composite
def _histories(draw: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = draw(st.integers(min_value=0, max_value=7))
    calls = [draw(_calls(k + 1)) for k in range(n + 1)]
    return calls[:-1], calls[-1]


@settings(max_examples=400, deadline=None)
@given(_histories())
def test_ported_rule_equals_group_a(case: tuple[list[dict[str, Any]], dict[str, Any]]) -> None:
    """C-7's port against group A's reviewed rule (green; an independent implementation of the same spec): on random
    histories over few screens, coordinates at the tolerance's edges, every status and ``no_change`` value,
    ``press_enter`` absent/False/True and an overview step 1, the eval's ``rule_verdict`` gives
    ``progress.judge``'s verdict."""
    history, call = case
    want = progress.judge(
        [_to_step(c) for c in history],
        call["name"],
        call["arguments"],
        call["screen"],
        overview=call["overview"],
    )
    got = oracle.rule_verdict(history, call)
    assert (got["repeat_of"], got["stop"], got["no_change_repeats"]) == (
        want.repeat_of,
        want.stop,
        want.no_change_repeats,
    ), (history, call, got, want)
    if call["screen"] is not None:
        assert got["screen_visits"] == want.screen_visits, (history, call, got, want)


# =====================================================================================================
# Hand-built trials and steps
# =====================================================================================================


def _rg(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in RG_KEYS}


def _tr(
    arm: str, task: str, repeat: int, state: Any, voiced: list[str], **kw: Any
) -> dict[str, Any]:
    """A trial at HEAD's default (frame on, T-0136 off) with the arm's guard; arm B's turn line has the key."""
    p = f"{arm}{task}{repeat}"
    base: dict[str, Any] = {
        "arm": arm,
        "voiced": list(voiced),
        "visual_tokens": 1120,
        "reground": False,
        "reground_visual_tokens": None,
        "live_feedback": dict(LABEL_FB),
        "echo": "model",
        "recover_thought_calls": True,
        "degeneracy_guard": True,
        "page_overview": False,
        "auto_overview": False,
        "overview_frame": True,
        "action_outcome": False,
        "repeat_guard": arm == "B",
        "tools_version": V1,
        "overview_frame_logged": t38._counts(**{CENSUS[task]: 1}),
        "auto_overview_logged": None,
        "page_overview_logged": None,
        "action_outcome_logged": None,
        "repeat_guard_logged": _rg() if arm == "B" else None,
        "degeneracy_guard_logged": {"on": True, "tripped": []},
        "live_feedback_logged": {**LABEL_FB, "echo": "model"},
        "recover_thought_calls_logged": True,
        "visual_tokens_logged": 1120,
        "budgets_sent": [1120],
        "guidance_sent": True,
        "scanned_requests": [f"prefill:{p}-0", f"generate:{p}-1"],
        "tool_loop_end": "final",
        "session_id": f"s-{p}",
        "history": "",
    }
    return t31._trial(task, repeat, state, **(base | kw))


def _st(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    name: str | None = None,
    args: dict[str, Any] | None = None,
    screen: str | None = None,
    *,
    shot: str | None = None,
    frame_kind: str | None = None,
    fact: int | None = None,
    fact_shot: str = "unchanged",
    stop: str | None = None,
    reason: str | None = None,
    voiced: tuple[str, ...] = (),
    **kw: Any,
) -> dict[str, Any]:
    """A step line (done 300 ms, first commit 200 ms); a call's ``no_change`` is its screenshot's (source
    ``screenshot``) and its ``guard_outcome`` ``{"screenshot": <it>}`` (S-OUT-2), an answer's are ``None``. ``fact``
    j gives ``repeat_of_step`` j and the mapping ``{"step": j, "action": ..., "screenshot": fact_shot}`` (written by
    hand: ``fact_shot`` is step j's screenshot, B-6 as merged)."""
    p = f"{arm}{task}{repeat}"
    mapping = None if fact is None else {"step": fact, "action": "done", "screenshot": fact_shot}
    fields: dict[str, Any] = {
        "arm": arm,
        "visual_tokens": 1120,
        "reground": False,
        "reground_visual_tokens": None,
        "call_name": name,
        "executed_name": name if status == "executed" else None,
        "voiced_sources": list(voiced),
        "attempt_id": f"{p}-{step}",
        "call_id": f"{p}-call-{step}" if name else None,
        "overview_shown": frame_kind == "overview",
        "frame_kind": frame_kind,
        "screenshot": shot,
        "call_arguments": None if name is None else dict(args or {}),
        "screen": screen,
        "repeat_of_step": fact,
        "repeat_fact": mapping,
        "no_progress": stop,
        "no_change": None if shot is None else shot == "unchanged",
        "no_change_source": None if shot is None else "screenshot",
        "guard_outcome": None if shot is None or name is None else {"screenshot": shot},
        "reason": reason,
    }
    return t31._step(task, repeat, step, status, 200.0, 300.0, **(fields | kw))


def _ex(arm: str, task: str, r: int, k: int, name: str, args: dict[str, Any], screen: str | None,
        shot: str = "updated", **kw: Any) -> dict[str, Any]:  # fmt: skip
    return _st(arm, task, r, k, "executed", name, args, screen, shot=shot, **kw)


def _ref(arm: str, task: str, r: int, k: int, name: str, args: dict[str, Any], screen: str | None,
         reason: str, **kw: Any) -> dict[str, Any]:  # fmt: skip
    """A refused call: B8 adds ``screenshot: "unchanged"`` as for any refusal (B-5)."""
    return _st(
        arm, task, r, k, "refused", name, args, screen, shot="unchanged", reason=reason, **kw
    )


def _answer(arm: str, task: str, r: int, k: int, voiced: list[str], **kw: Any) -> dict[str, Any]:
    return _st(arm, task, r, k, None, voiced=tuple(voiced), **kw)


def _metric_records(arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every count is written out in ``HAND``'s comment. ``scanned_requests`` is empty (no bodies: C-4 does not judge
    these records).

    Arm A (guard off; no fact, no stop logged): A1 S9#0 three silent scrolls down on screen p, "Saved." (fails); A2
    S6#0 a click at (100, 200) on s, the same click on s2, success; A3 S7#0 a step-1 overview click at (160, 933) and the
    same click on the live frame q, success; A4 S10#0 eight clicks, the third at (505, 495) on a again (a repeat of step
    1 at (500, 500)), a step-cap end; A5 S12#0 the report; A6 S15#1 the same click twice on o (two orders), "Your order is
    placed."; A7 S11#0 type "Grace Hopper" at (300, 200), the same with ``press_enter`` False, then with Enter,
    success; A8 S8#0 a click on g, the same click refused by B9, the same click again, "Added." (fails).

    Arm B (guard on; the facts and stops the rule gives): B1 S9#0 scrolls down on p: facts at steps 2 (of 1) and 3 (of
    2), step 4 stopped (``repeat``, of 3), step 5 asks "... which way should I scroll?"; B2 S10#0 clicks on a, b, a, b,
    then a fifth on a stopped (``recurring``), a sixth on c, success; B3 S6#0 clicks on x, y, x, c, d, e, f, and an
    eighth on x stopped (``recurring``) at the step cap; B4 S7#0 the overview click, the live click (changed), the
    same click again (a fact of 2), success; B5 S12#0 the report; B6 S9#1 a silent scroll, the same refused by B9
    (no fact), the same again (a fact of 1), "Saved." (fails)."""
    ok = t38._ok
    same = "unchanged"
    if arm == "A":
        trials = [
            _tr("A", "S9", 0, ok("S9", False)[0], ["Saved."]),
            _tr("A", "S6", 0, *ok("S6", True)),
            _tr("A", "S7", 0, ok("S7", True)[0], ["Added the M6 hex nuts, pack of 50."]),
            _tr("A", "S10", 0, {"page": "store", "cart": []}, [], tool_loop_end="budget"),
            _tr("A", "S12", 0, *ok("S12", True)),
            _tr(
                "A",
                "S15",
                1,
                {"page": "order", "orders": [{"n": 1}, {"n": 2}]},
                ["Your order is placed."],
            ),
            _tr("A", "S11", 0, ok("S11", True)[0], ["Booked."]),
            _tr("A", "S8", 0, ok("S8", False)[0], ["Added."]),
        ]
        g = {"x": 700, "y": 700}
        name = {"x": 300, "y": 200, "text": "Grace Hopper"}
        steps = [
            *[_ex("A", "S9", 0, k, "scroll", DOWN, "p", same) for k in (1, 2, 3)],
            _answer("A", "S9", 0, 4, ["Saved."]),
            _ex("A", "S6", 0, 1, "click_at", _xy(100, 200), "s"),
            _ex("A", "S6", 0, 2, "click_at", _xy(100, 200), "s2"),
            _answer("A", "S6", 0, 3, ["Seat F14 is selected."]),
            _ex("A", "S7", 0, 1, "click_at", _xy(160, 933), "q", frame_kind="overview"),
            _ex("A", "S7", 0, 2, "click_at", _xy(160, 933), "q", frame_kind="live"),
            _answer("A", "S7", 0, 3, ["Added the M6 hex nuts, pack of 50."]),
            _ex("A", "S10", 0, 1, "click_at", _xy(500, 500), "a"),
            _ex("A", "S10", 0, 2, "click_at", _xy(300, 300), "b"),
            _ex("A", "S10", 0, 3, "click_at", _xy(505, 495), "a"),
            *[_ex("A", "S10", 0, k, "click_at", _xy(40 * k, 40 * k), f"c{k}") for k in range(4, 9)],
            _answer("A", "S12", 0, 1, ["There is no yellow mug in this shop."]),
            _ex("A", "S15", 1, 1, "click_at", _xy(400, 600), "o"),
            _ex("A", "S15", 1, 2, "click_at", _xy(400, 600), "o"),
            _answer("A", "S15", 1, 3, ["Your order is placed."]),
            _ex("A", "S11", 0, 1, "type_at", name, "f"),
            _ex("A", "S11", 0, 2, "type_at", {**name, "press_enter": False}, "f"),
            _ex("A", "S11", 0, 3, "type_at", {**name, "press_enter": True}, "f"),
            _answer("A", "S11", 0, 4, ["Booked."]),
            _ex("A", "S8", 0, 1, "click_at", g, "g"),
            _ref("A", "S8", 0, 2, "click_at", g, "g", FRESHNESS_REASON),
            _ex("A", "S8", 0, 3, "click_at", g, "g"),
            _answer("A", "S8", 0, 4, ["Added."]),
        ]
    else:
        ask = "I can't get the panel to move; which way should I scroll?"
        trials = [
            _tr("B", "S9", 0, ok("S9", False)[0], [ask]),
            _tr("B", "S10", 0, *ok("S10", True)),
            _tr("B", "S6", 0, ok("S6", False)[0], [], tool_loop_end="budget"),
            _tr("B", "S7", 0, ok("S7", True)[0], ["Added the M6 hex nuts, pack of 50."]),
            _tr("B", "S12", 0, *ok("S12", True)),
            _tr("B", "S9", 1, ok("S9", False)[0], ["Saved."]),
        ]
        b3 = ["x", "y", "x", "c", "d", "e", "f"]
        steps = [
            _ex("B", "S9", 0, 1, "scroll", DOWN, "p", same),
            _ex("B", "S9", 0, 2, "scroll", DOWN, "p", same, fact=1),
            _ex("B", "S9", 0, 3, "scroll", DOWN, "p", same, fact=2),
            _ref("B", "S9", 0, 4, "scroll", DOWN, "p", REPEAT_REASON, stop="repeat", fact=3),
            _answer("B", "S9", 0, 5, [ask]),
            *[
                _ex("B", "S10", 0, k, "click_at", _xy(100 * k, 100 * k), s)
                for k, s in ((1, "a"), (2, "b"), (3, "a"), (4, "b"))
            ],
            _ref(
                "B", "S10", 0, 5, "click_at", _xy(500, 500), "a", RECURRING_REASON, stop="recurring"
            ),
            _ex("B", "S10", 0, 6, "click_at", _xy(600, 600), "c"),
            _answer("B", "S10", 0, 7, ["Added."]),
            *[_ex("B", "S6", 0, k, "click_at", _xy(100 * k, 100), s) for k, s in enumerate(b3, 1)],
            _ref(
                "B", "S6", 0, 8, "click_at", _xy(800, 100), "x", RECURRING_REASON, stop="recurring"
            ),
            _ex("B", "S7", 0, 1, "click_at", _xy(160, 933), "q", frame_kind="overview"),
            _ex("B", "S7", 0, 2, "click_at", _xy(160, 933), "q", frame_kind="live"),
            _ex(
                "B",
                "S7",
                0,
                3,
                "click_at",
                _xy(160, 933),
                "q",
                frame_kind="live",
                fact=2,
                fact_shot="updated",
            ),
            _answer("B", "S7", 0, 4, ["Added the M6 hex nuts, pack of 50."]),
            _answer("B", "S12", 0, 1, ["There is no yellow mug in this shop."]),
            _ex("B", "S9", 1, 1, "scroll", DOWN, "p", same),
            _ref("B", "S9", 1, 2, "scroll", DOWN, "p", FRESHNESS_REASON),
            _ex("B", "S9", 1, 3, "scroll", DOWN, "p", same, fact=1),
            _answer("B", "S9", 1, 4, ["Saved."]),
        ]
        # B-8 as merged, by hand: B1 two facts whose own steps showed no change, one stop, every judged call on its
        # first visit; B2/B3 a stop on a third visit; B4 one fact whose own step changed; B6 one fact, no change
        trials[0]["repeat_guard_logged"] = _rg(
            repeats=2, no_change_repeats=2, stops_repeat=1, max_screen_visits=1
        )
        trials[1]["repeat_guard_logged"] = _rg(stops_recurring=1, max_screen_visits=3)
        trials[2]["repeat_guard_logged"] = _rg(stops_recurring=1, max_screen_visits=3)
        trials[3]["repeat_guard_logged"] = _rg(repeats=1, max_screen_visits=1)
        trials[5]["repeat_guard_logged"] = _rg(repeats=1, no_change_repeats=1, max_screen_visits=1)
    for t in trials:
        t["scanned_requests"] = []
    return trials, steps


def _zero(tasks: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(tasks, 0)


OVER_TASKS = ("S6", "S7", "S8", "S10", "S14", "S15")
#: by hand from ``_metric_records`` (its docstring); nearest-rank percentiles (ceil(p/100 * n)-th smallest):
#: A -- success A2 (S6), A3 (S7), A5 (S12), A7 (S11) = 4 of 8; over-action A6 (two orders, S15); false claims A1
#:      ("Saved."), A6, A8 = 3 (A4 ends on a call); steps [4, 3, 3, 8, 1, 3, 4, 4] = 30 over 8 = 3.75, sorted
#:      [1,3,3,3,4,4,4,8]: p50 the 4th = 3, p95 the 8th = 8; step cap A4; W1 = A1 2 (steps 2, 3) + A4 1 (step 3,
#:      within 5 units of step 1 on a) + A6 1 + A7 1 (``press_enter`` False = missing) + A8 2 (the B9-refused step 2
#:      and step 3, whose only instance is step 1) = 7, of which refused 1; A2 (another screen), A3 (overview) and
#:      A7 step 3 (Enter) are none; no fact, no stop logged.
#: B -- success B2 (S10), B4 (S7), B5 (S12) = 3 of 6; no over-action; false claim B6 (B1 asks, B3 ends on a call);
#:      steps [5, 7, 8, 4, 1, 4] = 29 over 6, sorted [1,4,4,5,7,8]: p50 the 3rd = 4, p95 the 6th = 8; step cap B3; W1 =
#:      B1 3 (steps 2, 3 and the stopped 4) + B4 1 (step 3) + B6 2 (the B9-refused 2 and 3) = 6, of which refused 2
#:      (B2's and B3's stops are on new coordinates); facts B1 step 2 (of 1, no change), step 3 (of 2, no change), B4
#:      step 3 (of 2, changed), B6 step 3 (of 1, no change); stops repeat 1 (B1), recurring 2 (B2, B3); after them
#:      B1 asked (answered and voiced), B2 clicked c (called again), B3 hit the cap (budget); B2 succeeded, B1 and B3
#:      failed.
HAND: dict[str, dict[str, Any]] = {
    "A": {
        "n": 8,
        "success": 4,
        "per_task": _zero(TASK_IDS) | {"S6": 1, "S7": 1, "S11": 1, "S12": 1},
        "per_task_n": _zero(TASK_IDS)
        | {"S6": 1, "S7": 1, "S8": 1, "S9": 1, "S10": 1, "S11": 1, "S12": 1, "S15": 1},
        "over_action": _zero(OVER_TASKS) | {"S15": 1},
        "over_actions": 1,
        "false_claims": 3,
        "steps_per_turn": {"n": 8, "p50": 3, "p95": 8},
        "step_cap_ends": 1,
        "timeouts": 0,
        "engine_errors": 0,
        "degenerate": 0,
        "w1": 7,
        "w1_by_status": {"executed": 6, "refused": 1},
        "repeat_facts": {"no change": 0, "changed": 0, "unknown": 0},
        "stops": {"repeat": 0, "recurring": 0},
        "after_stop": {"answered_voiced": 0, "called_again": 0, "budget": 0, "other": 0},
        "stops_in": {"success": 0, "failure": 0},
    },
    "B": {
        "n": 6,
        "success": 3,
        "per_task": _zero(TASK_IDS) | {"S7": 1, "S10": 1, "S12": 1},
        "per_task_n": _zero(TASK_IDS) | {"S6": 1, "S7": 1, "S9": 2, "S10": 1, "S12": 1},
        "over_action": _zero(OVER_TASKS),
        "over_actions": 0,
        "false_claims": 1,
        "steps_per_turn": {"n": 6, "p50": 4, "p95": 8},
        "step_cap_ends": 1,
        "timeouts": 0,
        "engine_errors": 0,
        "degenerate": 0,
        "w1": 6,
        "w1_by_status": {"executed": 4, "refused": 2},
        "repeat_facts": {"no change": 3, "changed": 1, "unknown": 0},
        "stops": {"repeat": 1, "recurring": 2},
        "after_stop": {"answered_voiced": 1, "called_again": 1, "budget": 1, "other": 0},
        "stops_in": {"success": 1, "failure": 2},
    },
}
#: mean steps per turn (30/8, 29/6) and Wilson 95 % by the closed form (z = 1.959964, a scratch script, 2026-10-25)
HAND_MEAN = {"A": 30 / 8, "B": 29 / 6}
HAND_CI = {"A": (0.21522, 0.78478), "B": (0.18762, 0.81238)}


def test_hand_states_premise() -> None:
    """Premise (green): the judge's verdicts on the hand end states the counts rest on (success, ``over_action``)."""
    j = suite_tasks.judge
    assert j("S15", {"page": "order", "orders": [{"n": 1}, {"n": 2}]}, "x").over_action
    assert not j("S10", {"page": "store", "cart": []}, "").success
    assert j("S10", *t38._ok("S10", True)[:1], "Added.").success
    assert not j("S6", t38._ok("S6", False)[0], "").over_action
    assert j("S11", t38._ok("S11", True)[0], "Booked.").success
    assert not j("S8", t38._ok("S8", False)[0], "Added.").success


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_hand_counts(which: str, arm: str) -> None:
    """C-3: success per task and pooled with its Wilson interval, over-actions (T-0138's six tasks, reading 1), false
    claims, steps per turn and their mean, step-cap ends, W1 (recomputed by the ported rule in **both** arms; a
    B9-refused repeat and a stopped repeat counted, an overview click or another screen not), and arm B's
    mechanism counts (facts by the referenced step's outcome, stops by kind, what followed each stop, stops in
    successful and failing turns)."""
    trials, steps = _metric_records(arm)
    got = impl(which, "derive_arm")(trials, steps)
    for key, want in HAND[arm].items():
        assert got[key] == want, (arm, key, got[key], want)
    assert abs(float(got["mean_steps"]) - HAND_MEAN[arm]) < 1e-9, got["mean_steps"]
    lo, hi = got["success_ci"]
    assert abs(lo - HAND_CI[arm][0]) < 5e-5 and abs(hi - HAND_CI[arm][1]) < 5e-5, got["success_ci"]


@pytest.mark.parametrize("which", IMPLS)
def test_w1_counts_a_refused_repeat_in_arm_b(which: str) -> None:
    """C-3 "executed or refused" (**adversarial**; the mutant counts executed repeats only, or reads the logged
    ``repeat_of_step`` instead of recomputing): arm B's stopped fourth scroll is a W1 step though it never executed and
    the guard's own refusal is no instance; with every logged fact and stop erased from the step lines W1 is
    unchanged, because W1 is recomputed from the calls and screens."""
    trials, steps = _metric_records("B")
    s9 = [s for s in steps if (s["task"], s["repeat"]) == ("S9", 0)]
    got = impl(which, "derive_arm")(
        [t for t in trials if (t["task"], t["repeat"]) == ("S9", 0)], s9
    )
    assert (got["w1"], got["w1_by_status"]) == (3, {"executed": 2, "refused": 1}), got[
        "w1_by_status"
    ]
    erased = [{**s, "repeat_of_step": None, "no_progress": None} for s in steps]
    assert impl(which, "derive_arm")(trials, erased)["w1"] == 6


@pytest.mark.parametrize("which", IMPLS)
def test_empty_arm(which: str) -> None:
    arm = impl(which, "derive_arm")([], [])
    assert arm["success_ci"] is None and arm["n"] == 0 and arm["w1"] == 0, arm
    assert arm["mean_steps"] is None and all(v is None for v in arm["per_task_ci"].values()), arm


# =====================================================================================================
# Paired McNemar by (task, repeat), report-only
# =====================================================================================================


def _pair_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """T-0138's hand pairs (its ``PAIRS``), as this A/B's trials."""
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for task, repeat, a, b in t38.PAIRS:
        for arm, ok in (("A", a), ("B", b)):
            if ok is None:
                continue
            state, voiced = t38._ok(task, ok in (True, "forced"))
            extra = {"tool_loop_end": "degenerate"} if ok == "forced" else {}
            trials.append(_tr(arm, task, repeat, state, voiced, **extra))
            steps.append(_answer(arm, task, repeat, 1, voiced))
    for t in trials:
        t["scanned_requests"] = []
    return trials, steps


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("seed", [None, 0, 1])
def test_mcnemar_sections(which: str, seed: int | None) -> None:
    """Paired by (task, repeat) (**adversarial**: arm B's trials reversed, or both lists shuffled, so pairing by position
    cannot give these counts): the pooled section and each task by hand from T-0138's ``PAIRS`` (its comment: 15 pairs,
    b 7, c 4, p 281/512; per task its ``PER_TASK``), a ``degenerate`` arm-A turn a failure, the unpaired listed."""
    trials, steps = _pair_records()
    if seed is None:
        a = [t for t in trials if t["arm"] == "A"]
        b = [t for t in trials if t["arm"] == "B"]
        trials = a + b[::-1]
    else:
        rng = random.Random(seed)
        rng.shuffle(trials)
        rng.shuffle(steps)
    got = impl(which, "mcnemar_sections")(trials, steps)
    n, b_, c_, p = t38.SECTIONS["pooled"]
    assert (got["pooled"]["n"], got["pooled"]["b"], got["pooled"]["c"]) == (n, b_, c_), got[
        "pooled"
    ]
    assert abs(float(got["pooled"]["p"]) - p) < 1e-12
    for task, (n, b_, c_, p) in t38.PER_TASK.items():
        s = got["per_task"][task]
        assert (s["n"], s["b"], s["c"]) == (n, b_, c_) and abs(float(s["p"]) - p) < 1e-12, (task, s)
    assert sorted(got["unpaired"]) == t38.UNPAIRED


# =====================================================================================================
# C-5: the decision rule (re-expressed on the suite; reading 6 of the eval)
# =====================================================================================================


def _arm(
    w1: int, success: int, false_claims: int = 5, over: int = 4, n: int = 100, **kw: Any
) -> dict[str, Any]:
    base = {
        "accepted": True,
        "safety": 0,
        "engine_errors": 0,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
        "n": n,
        "success": success,
        "false_claims": false_claims,
        "over_actions": over,
        "w1": w1,
    }
    return base | kw


def _inputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {"A": a, "B": b}


#: (A, B, label), by hand from reading 6; boundaries at W1(A) 5 vs 6 (reviewer mutation 14), W1(B) = W1(A)/2 vs
#: W1(A)/2 + 1 (even and odd W1(A)), success -3 vs -4 of 100, false claims +2 vs +3, over-actions +2 vs +3
DECISIONS: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
    "w1a-5-not-exercised": (_arm(5, 70), _arm(0, 90, 0, 0), "not exercised"),
    "w1a-6-exercised-adopt": (_arm(6, 70), _arm(3, 70), "adopt"),
    "w1a-0-not-exercised": (_arm(0, 70), _arm(0, 70), "not exercised"),
    "w1a-5-every-clause-failing": (_arm(5, 70, 0, 0), _arm(5, 50, 9, 9), "not exercised"),
    "w1a-6-w1b-4-no-change": (_arm(6, 70), _arm(4, 70), "no change"),
    "w1a-7-w1b-3-adopt": (_arm(7, 70), _arm(3, 70), "adopt"),
    "w1a-7-w1b-4-no-change": (_arm(7, 70), _arm(4, 70), "no change"),
    "w1a-8-w1b-4-adopt": (_arm(8, 70), _arm(4, 70), "adopt"),
    "w1a-8-w1b-5-no-change": (_arm(8, 70), _arm(5, 70), "no change"),
    "success-minus-3-adopt": (_arm(6, 70), _arm(3, 67), "adopt"),
    "success-minus-4-no-change": (_arm(6, 70), _arm(3, 66), "no change"),
    "success-better-adopt": (_arm(6, 70), _arm(0, 80), "adopt"),
    "false-plus-2-adopt": (_arm(6, 70, 5), _arm(3, 70, 7), "adopt"),
    "false-plus-3-no-change": (_arm(6, 70, 5), _arm(3, 70, 8), "no change"),
    "over-plus-2-adopt": (_arm(6, 70, 5, 4), _arm(3, 70, 5, 6), "adopt"),
    "over-plus-3-no-change": (_arm(6, 70, 5, 4), _arm(3, 70, 5, 7), "no change"),
    "w1b-0-adopt": (_arm(9, 60), _arm(0, 60), "adopt"),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """C-5 (H-T137-7, re-expressed on the suite; reviewer mutation 14 -- the ``not exercised`` threshold off by one):
    every label at its boundary, by hand."""
    a, b, want = DECISIONS[row]
    got = impl(which, "decide_with_reason")(copy.deepcopy(_inputs(a, b)), p1_passed=True)
    assert got[0] == want, (row, got)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_arm(which: str, arm: str, clause: str) -> None:
    """C-5 "order as in T-0151": an ineligible arm gives ``inconclusive`` before the rule, naming the arm, the clause
    and the first failing trial; an adopting run does not save it; 2 engine errors are eligible, 3 are not."""
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 1},
        "engine": {"engine_errors": 3},
    }[clause]
    first = {c: ("S9#7" if c == clause else None) for c in ("acceptance", "safety", "engine")}
    inputs = _inputs(_arm(6, 70), _arm(3, 70))
    inputs[arm] |= flag | {"first_failing": first}
    got = impl(which, "decide_with_reason")(inputs, p1_passed=True)
    assert tuple(got) == (
        "inconclusive",
        f"arm {arm} ineligible: {clause} (first failing trial S9#7)",
    ), got
    if clause == "engine":
        inputs[arm]["engine_errors"] = 2
        assert impl(which, "decide_with_reason")(inputs, p1_passed=True)[0] == "adopt"


@pytest.mark.parametrize("which", IMPLS)
def test_label_order(which: str) -> None:
    """C-5's order: P1 failed -> ``inconclusive`` over everything; a configuration finding -> ``void``; a probe void
    -> ``void`` over a stop; stopped -> ``incomplete`` over an ineligible arm; an ineligible arm over ``not
    exercised``."""
    why = impl(which, "decide_with_reason")
    bad = _inputs(
        _arm(0, 70),
        _arm(
            0,
            70,
            accepted=False,
            first_failing={"acceptance": "S6#0", "safety": None, "engine": None},
        ),
    )
    everything = {"void": True, "stopped": "RUN_BUDGET_S", "configuration_findings": ["X", "Y"]}
    assert tuple(why(bad, p1_passed=False, **everything)) == ("inconclusive", "P1 failed")
    assert tuple(why(bad, p1_passed=True, **everything)) == ("void", "configuration: X")
    assert why(bad, p1_passed=True, void=True, stopped="RUN_BUDGET_S")[0] == "void"
    assert tuple(why(bad, p1_passed=True, stopped="RUN_BUDGET_S")) == ("incomplete", "RUN_BUDGET_S")
    assert why(bad, p1_passed=True)[0] == "inconclusive"
    assert (
        why(_inputs(_arm(0, 70), _arm(0, 70)), p1_passed=True, configuration_findings=[])[0]
        == "not exercised"
    )


@pytest.mark.parametrize("which", IMPLS)
def test_decision_inputs_from_the_hand_records(which: str) -> None:
    """C-5's per-arm inputs over the metric records, by hand (``HAND``): A accepted, n 8, success 4, false claims 3,
    over-actions 1, W1 7; B accepted, n 6, success 3, false claims 1, over-actions 0, W1 6. Over these two arms the
    rule says ``no change``: W1(A) 7 is exercised, W1(B) 6 > 3.5."""
    fn = impl(which, "decision_inputs")
    got = {arm: dict(fn(*_metric_records(arm), [])) for arm in ("A", "B")}
    keys = (
        "accepted",
        "safety",
        "engine_errors",
        "n",
        "success",
        "false_claims",
        "over_actions",
        "w1",
    )
    assert tuple(got["A"][k] for k in keys) == (True, 0, 0, 8, 4, 3, 1, 7), got["A"]
    assert tuple(got["B"][k] for k in keys) == (True, 0, 0, 6, 3, 1, 0, 6), got["B"]
    assert impl(which, "decide_with_reason")(got, p1_passed=True)[0] == "no change"


# =====================================================================================================
# C-4: configuration and guard findings, one hand trial per finding
# =====================================================================================================


def _i(tag: str) -> dict[str, str]:
    return {"image": t38._h(tag)}


def _t(text: str) -> dict[str, str]:
    return {"text": text}


def _asst(text: str) -> dict[str, Any]:
    return {"role": "assistant", "parts": [_t(text)]}


def _ov(arm: str, task: str) -> dict[str, Any]:
    return t38._ov(f"{arm}{task}0-ov", 1.5)


def _gen(arm: str, task: str, repeat: int, ov: dict[str, Any] | None) -> dict[str, Any]:
    p = f"{arm}{task}{repeat}"
    if ov is not None:
        parts = [
            {"image": ov["sha"]},
            _t(t38.frame_label(float(ov["screens"]))),
            _t(REQUESTS[task]),
        ]
    else:
        parts = [_i(f"{task}{repeat}-f0"), _t(SCREEN_NOW), _t(REQUESTS[task])]
    return t38._req(f"generate:{p}-1", "generate", 1, t38.V1_TOOLS, t38._sys(), t38._user(*parts))


def _response(name: str, payload: str) -> str:
    return f"<|tool_response>response:{name}{{{payload}}}<tool_response|>"


SCROLL_RESPONSE = _response(
    "scroll", 'action:<|"|>scrolled down<|"|>,screenshot:<|"|>unchanged<|"|>'
)
FACT_RESPONSE = _response(
    "scroll",
    'action:<|"|>scrolled down<|"|>,repeat_of_step:{action:<|"|>scrolled down<|"|>,screenshot:<|"|>unchanged'
    '<|"|>,step:1},screenshot:<|"|>unchanged<|"|>',
)


def _step2(arm: str, task: str, text: str) -> dict[str, Any]:
    p = f"{arm}{task}0"
    return t38._req(
        f"generate:{p}-2",
        "tool_step",
        2,
        t38.V1_TOOLS,
        t38._sys(),
        t38._user(_i(f"{p}-f1"), _t(t38.SCREEN_AFTER), _t(REQUESTS[task])),
        _asst("Scrolling. " + text),
    )


CFG_TASKS = ("S7", "S9", "S8")
ASK = "Which way should I scroll?"


def _cfg_clean() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Both arms at HEAD's default (the frame on): S7 shows the overview frame at step 1 in **both** arms (1.5
    screens; each arm's own capture), S9 and S8 the live frame. S9: arm A scrolls down twice on p and answers (no
    fact); arm B scrolls down three times (facts of 1 and 2) and its fourth is stopped (``repeat``, of 3), then it asks.
    S7 and S8 click once and answer. Arm A's S9 step-2 body holds a plain response, arm B's a repeat fact."""
    ok = t38._ok
    trials = []
    for task in CFG_TASKS:
        for arm in ("A", "B"):
            voiced = [ASK] if (arm, task) == ("B", "S9") else ok(task, True)[1]
            trials.append(_tr(arm, task, 0, ok(task, True)[0], voiced))
    # B-8 as merged, by hand: two facts whose own steps showed no change, one stop, every judged call on visit 1
    t38._find(trials, "B", "S9", 0)["repeat_guard_logged"] = _rg(
        repeats=2, no_change_repeats=2, stops_repeat=1, max_screen_visits=1
    )
    same = "unchanged"
    steps = [
        _ex("A", "S9", 0, 1, "scroll", DOWN, "p", same),
        _ex("A", "S9", 0, 2, "scroll", DOWN, "p", same),
        _answer("A", "S9", 0, 3, ["Saved."]),
        _ex("B", "S9", 0, 1, "scroll", DOWN, "p", same),
        _ex("B", "S9", 0, 2, "scroll", DOWN, "p", same, fact=1),
        _ex("B", "S9", 0, 3, "scroll", DOWN, "p", same, fact=2),
        _ref("B", "S9", 0, 4, "scroll", DOWN, "p", REPEAT_REASON, stop="repeat", fact=3),
        _answer("B", "S9", 0, 5, [ASK]),
    ]
    for arm in ("A", "B"):
        steps += [
            _ex(arm, "S7", 0, 1, "click_at", _xy(160, 933), "q", frame_kind="overview"),
            _answer(arm, "S7", 0, 2, ok("S7", True)[1]),
            _ex(arm, "S8", 0, 1, "click_at", _xy(500, 500), "g"),
            _answer(arm, "S8", 0, 2, ok("S8", True)[1]),
        ]
    bodies = []
    for task in CFG_TASKS:
        for arm in ("A", "B"):
            ov = _ov(arm, task) if task == "S7" else None
            requests = [t38._prefill(arm, task, 0), _gen(arm, task, 0, ov)]
            if task == "S9":
                requests.append(_step2(arm, task, SCROLL_RESPONSE if arm == "A" else FACT_RESPONSE))
            bodies.append(t38._line(arm, task, 0, requests, [dict(ov)] if ov else []))
    return trials, steps, [], bodies


#: T-0136's no-change literal (typed) and another of its labels (any text but the literal reads as a change)
NVC = "no visible change"
CHANGED_LABEL = "the page changed"
OUTCOME_ARMS = [{**r, "action_outcome": True} for r in HAND_ARMS]


def _cfg_outcome() -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]:
    """The clean records with T-0136's label on by default (both arms): every turn line has T-0136's key; arm B's S8
    clicks (500, 500) on g twice. T-0136 measured both clicks: step 1 ``no visible change`` though B8 says ``updated``
    (a hover tint: ``no_change`` True from ``t0136``, S-OUT-1), step 2 a change. Step 2 executes with the fact ``{"step":
    1, "action", "outcome": "no visible change", "screenshot": "updated"}``. Returns (trials, steps, regrounds, bodies,
    arms)."""
    trials, steps, regrounds, bodies = _cfg_clean()
    for t in trials:
        t["action_outcome"] = True
        t["action_outcome_logged"] = {
            "none": 1,
            "near": 0,
            "elsewhere": 0,
            "page": 0,
            "unmeasured": 0,
        }
    t38._find(trials, "B", "S8", 0)["repeat_guard_logged"] = _rg(repeats=1, max_screen_visits=1)
    steps = [s for s in steps if (s["arm"], s["task"]) != ("B", "S8")]
    measured = {"no_change_source": "t0136"}
    steps += [
        _ex("B", "S8", 0, 1, "click_at", _xy(500, 500), "g", no_change=True, **measured,
            guard_outcome={"outcome": NVC, "screenshot": "updated"}),
        _ex("B", "S8", 0, 2, "click_at", _xy(500, 500), "g", no_change=False, **measured,
            guard_outcome={"outcome": CHANGED_LABEL, "screenshot": "updated"}, fact=1,
            repeat_fact={"step": 1, "action": "clicked", "outcome": NVC, "screenshot": "updated"}),
        _answer("B", "S8", 0, 3, t38._ok("S8", True)[1]),
    ]  # fmt: skip
    return trials, steps, regrounds, bodies, [copy.deepcopy(r) for r in OUTCOME_ARMS]


def _step_of(steps: list[dict[str, Any]], arm: str, task: str, k: int) -> dict[str, Any]:
    (s,) = [s for s in steps if (s["arm"], s["task"], s["repeat"], s["step"]) == (arm, task, 0, k)]
    return s


def _cfg_fault(name: str) -> tuple[Any, ...]:
    """One C-4 fault on the clean records; returns (trials, steps, regrounds, bodies, arms, guidance sha, what a
    finding must name)."""
    trials, steps, regrounds, bodies = _cfg_clean()
    arms = [copy.deepcopy(r) for r in HAND_ARMS]
    sha = GUIDANCE_SHA
    a9 = lambda k: _step_of(steps, "A", "S9", k)  # noqa: E731
    b9 = lambda k: _step_of(steps, "B", "S9", k)  # noqa: E731
    out = lambda named: (trials, steps, regrounds, bodies, arms, sha, named)  # noqa: E731
    # the step fields as group B merged them (d0f1f02): the fact mapping, the step outcome, the turn line's counts
    if name == "b-fact-mapping-names-another-step":
        b9(3)["repeat_fact"] = {**b9(3)["repeat_fact"], "step": 1}
        return out("B S9#0 step 3")
    if name == "b-fact-mapping-missing-on-the-stop":
        b9(4)["repeat_fact"] = None
        return out("B S9#0 step 4")
    if name == "b-fact-mapping-on-the-recurring-shape":
        b9(4).update(repeat_of_step=None)
        return out("B S9#0 step 4")
    if name == "b-fact-screenshot-off-step-j":
        b9(2)["repeat_fact"] = {**b9(2)["repeat_fact"], "screenshot": "updated"}
        return out("B S9#0 step 2")
    if name == "b-fact-without-its-action":
        b9(2)["repeat_fact"] = {k: v for k, v in b9(2)["repeat_fact"].items() if k != "action"}
        return out("B S9#0 step 2")
    if name == "b-fact-with-an-outcome-key-t0136-never-gave":
        b9(3)["repeat_fact"] = {**b9(3)["repeat_fact"], "outcome": NVC}
        return out("B S9#0 step 3")
    if name == "guard-outcome-off-its-screenshot":
        a9(1)["guard_outcome"] = {"screenshot": "updated"}
        return out("A S9#0 step 1")
    if name == "b-turn-line-repeats-off":
        t38._find(trials, "B", "S9", 0)["repeat_guard_logged"]["repeats"] = 3
        return out("B S9#0")
    if name == "b-turn-line-no-change-repeats-off":
        t38._find(trials, "B", "S9", 0)["repeat_guard_logged"]["no_change_repeats"] = 1
        return out("B S9#0")
    if name == "b-turn-line-max-visits-off":
        t38._find(trials, "B", "S9", 0)["repeat_guard_logged"]["max_screen_visits"] = 2
        return out("B S9#0")
    if name.startswith("outcome-on-"):
        trials, steps, regrounds, bodies, arms = _cfg_outcome()
        b8 = lambda k: _step_of(steps, "B", "S8", k)  # noqa: E731
        if name == "outcome-on-fact-without-the-outcome-key":
            b8(2)["repeat_fact"] = {"step": 1, "action": "clicked", "screenshot": "updated"}
        elif name == "outcome-on-fact-outcome-off-step-j":
            b8(2)["repeat_fact"] = {**b8(2)["repeat_fact"], "outcome": CHANGED_LABEL}
        elif name == "outcome-on-no-change-off-its-label":
            b8(1)["no_change"] = False
        elif name == "outcome-on-guard-outcome-without-the-screenshot":
            b8(1)["guard_outcome"] = {"outcome": NVC}
        elif name == "outcome-on-turn-line-without-its-key":
            t38._find(trials, "B", "S8", 0)["action_outcome_logged"] = None
        else:
            raise KeyError(name)
        return (trials, steps, regrounds, bodies, arms, sha, "B S8#0")
    if name == "a-turn-line-with-the-key":
        t38._find(trials, "A", "S8", 0)["repeat_guard_logged"] = _rg()
        return out("A S8#0")
    if name == "a-step-with-a-fact":
        a9(2)["repeat_of_step"] = 1
        return out("A S9#0 step 2")
    if name == "a-step-with-a-stop":
        a9(2).update(
            status="refused", executed_name=None, no_progress="repeat", reason=REPEAT_REASON
        )
        return out("A S9#0 step 2")
    if name == "a-step-with-a-guard-reason":
        a9(2).update(status="refused", executed_name=None, reason=RECURRING_REASON)
        return out("A S9#0 step 2")
    if name == "a-body-with-a-fact":
        line = t38._find(bodies, "A", "S9", 0)
        line["requests"][2] = _step2("A", "S9", FACT_RESPONSE)
        return out("A S9#0")
    if name == "a-body-with-a-guard-reason":
        line = t38._find(bodies, "A", "S9", 0)
        line["requests"][2] = _step2(
            "A", "S9", _response("scroll", f'reason:<|"|>{RECURRING_REASON}<|"|>')
        )
        return out("A S9#0")
    if name == "b-turn-line-without-the-key":
        t38._find(trials, "B", "S8", 0)["repeat_guard_logged"] = None
        return out("B S8#0")
    if name == "b-turn-line-malformed-key":
        t38._find(trials, "B", "S7", 0)["repeat_guard_logged"] = {"repeats": 0, "stops_repeat": 0}
        return out("B S7#0")
    if name == "b-turn-line-stops-disagree":
        t38._find(trials, "B", "S9", 0)["repeat_guard_logged"] = _rg(max_screen_visits=1)
        return out("B S9#0")
    if name == "b-stop-the-rule-does-not-justify":
        b9(3).update(
            status="refused",
            executed_name=None,
            no_progress="repeat",
            reason=REPEAT_REASON,
            shot="unchanged",
        )
        return out("B S9#0 step 3")
    if name == "b-stop-missing":
        b9(4).update(
            status="executed",
            executed_name="scroll",
            no_progress=None,
            reason=None,
            repeat_of_step=3,
        )
        return out("B S9#0 step 4")
    if name == "b-stop-of-the-wrong-kind":
        b9(4).update(no_progress="recurring", reason=RECURRING_REASON)
        return out("B S9#0 step 4")
    if name == "b-stop-with-the-other-reason":
        b9(4)["reason"] = RECURRING_REASON
        return out("B S9#0 step 4")
    if name == "b-stop-not-refused":
        b9(4).update(status="executed", executed_name="scroll", reason=None)
        return out("B S9#0 step 4")
    if name == "b-fact-the-rule-does-not-justify":
        b9(3)["repeat_of_step"] = 1
        return out("B S9#0 step 3")
    if name == "b-fact-missing":
        b9(2)["repeat_of_step"] = None
        return out("B S9#0 step 2")
    if name == "b-fact-on-a-non-repeat":
        _step_of(steps, "B", "S8", 1)["repeat_of_step"] = 1
        return out("B S8#0 step 1")
    if name == "b-fact-on-a-b9-refusal":
        b9(4).update(no_progress=None, reason=FRESHNESS_REASON, repeat_of_step=3)
        return out("B S9#0 step 4")
    if name == "b-a-guard-reason-without-a-stop":
        _step_of(steps, "B", "S8", 1).update(
            status="refused", executed_name=None, reason=REPEAT_REASON
        )
        return out("B S8#0 step 1")
    if name == "reason-in-the-voiced-sources":
        t38._find(trials, "B", "S9", 0)["voiced"] = [REPEAT_REASON]
        return out("B S9#0")
    if name == "reason-in-a-steps-voiced-sources":
        b9(5)["voiced_sources"] = [f"Sorry: {REPEAT_REASON}."]
        return out("B S9#0")
    if name == "reason-in-the-history":
        t38._find(trials, "B", "S9", 0)["history"] = f"assistant: {RECURRING_REASON}"
        return out("B S9#0")
    if name == "no-change-off-its-screenshot":
        a9(1)["no_change"] = False
        return out("A S9#0 step 1")
    if name == "no-change-without-a-source":
        a9(1)["no_change_source"] = None
        return out("A S9#0 step 1")
    if name == "t0136-source-with-the-label-off":
        a9(1)["no_change_source"] = "t0136"
        return out("A S9#0 step 1")
    if name == "guidance-off-its-sha":
        sha = "0" * 64
        return out("LIVE_TOOL_GUIDANCE")
    if name == "trial-tools-off-v1":
        t38._find(trials, "B", "S8", 0)["tools_version"] = OV_SHA
        return out("B S8#0")
    if name == "trial-guard-off-its-row":
        t38._find(trials, "B", "S7", 0)["repeat_guard"] = False
        return out("B S7#0")
    if name == "trial-outcome-label-off-its-row":
        t38._find(trials, "A", "S8", 0)["action_outcome"] = True
        return out("A S8#0")
    if name == "arm-a-row-guard-on":
        arms[0]["repeat_guard"] = True
        return out("arm A")
    if name == "arm-b-row-guard-off":
        arms[1]["repeat_guard"] = False
        return out("arm B")
    if name == "arms-differ-in-the-frame":
        arms[1]["overview_frame"] = False
        return out("overview_frame")
    if name == "arms-differ-in-the-outcome-label":
        arms[0]["action_outcome"] = True
        return out("action_outcome")
    if name == "arm-rows-on-the-overview-set":
        for r in arms:
            r["tools_sha256"] = OV_SHA
        return out("tools")
    if name == "outcome-label-key-with-the-label-off":
        t38._find(trials, "B", "S8", 0)["action_outcome_logged"] = {"none": 0}
        return out("B S8#0")
    # T-0138 B7's findings that apply at HEAD's default (reading 7)
    if name == "b-shown-overview-beside-the-live-frame":
        ov = _ov("B", "S7")
        t38._final_user(t38._gen_of(bodies, "B", "S7"))["parts"] = [
            {"image": ov["sha"]},
            _t(t38.frame_label(1.5)),
            _i("S70-f0"),
            _t(SCREEN_NOW),
            _t(REQUESTS["S7"]),
        ]
        return out("B S7#0")
    if name == "frame-on-turn-line-without-the-overview-key":
        t38._find(trials, "B", "S9", 0)["overview_frame_logged"] = None
        return out("B S9#0")
    if name == "overview-on-a-fits-trial":
        t38._find(trials, "A", "S9", 0)["overview_frame_logged"] = t38._counts(shown=1)
        return out("A S9#0")
    if name == "pair-step1-request-text-differs":
        t38._final_user(t38._gen_of(bodies, "B", "S9"))["parts"][2] = _t("choose the locker")
        return out("S9#0")
    if name == "pair-step1-tools-differ":
        t38._gen_of(bodies, "A", "S7")["tools"] = t38.V1_TOOLS[:-1]
        return out("S7#0")
    if name == "reground-line":
        regrounds = [{"arm": "B", "task": "S8", "repeat": 0, "step": 1, "outcome": "ok"}]
        return (trials, steps, regrounds, bodies, arms, sha, "B S8#0")
    if name == "page-overview-key":
        t38._find(trials, "A", "S7", 0)["page_overview_logged"] = {"calls": 0}
        return out("A S7#0")
    if name == "non-1120-body":
        t38._find(trials, "B", "S9", 0)["budgets_sent"] = [1120, 560]
        return out("B S9#0")
    raise KeyError(name)


CFG_FAULTS = (
    "b-fact-mapping-names-another-step",
    "b-fact-mapping-missing-on-the-stop",
    "b-fact-mapping-on-the-recurring-shape",
    "b-fact-screenshot-off-step-j",
    "b-fact-without-its-action",
    "b-fact-with-an-outcome-key-t0136-never-gave",
    "guard-outcome-off-its-screenshot",
    "b-turn-line-repeats-off",
    "b-turn-line-no-change-repeats-off",
    "b-turn-line-max-visits-off",
    "outcome-on-fact-without-the-outcome-key",
    "outcome-on-fact-outcome-off-step-j",
    "outcome-on-no-change-off-its-label",
    "outcome-on-guard-outcome-without-the-screenshot",
    "outcome-on-turn-line-without-its-key",
    "a-turn-line-with-the-key",
    "a-step-with-a-fact",
    "a-step-with-a-stop",
    "a-step-with-a-guard-reason",
    "a-body-with-a-fact",
    "a-body-with-a-guard-reason",
    "b-turn-line-without-the-key",
    "b-turn-line-malformed-key",
    "b-turn-line-stops-disagree",
    "b-stop-the-rule-does-not-justify",
    "b-stop-missing",
    "b-stop-of-the-wrong-kind",
    "b-stop-with-the-other-reason",
    "b-stop-not-refused",
    "b-fact-the-rule-does-not-justify",
    "b-fact-missing",
    "b-fact-on-a-non-repeat",
    "b-fact-on-a-b9-refusal",
    "b-a-guard-reason-without-a-stop",
    "reason-in-the-voiced-sources",
    "reason-in-a-steps-voiced-sources",
    "reason-in-the-history",
    "no-change-off-its-screenshot",
    "no-change-without-a-source",
    "t0136-source-with-the-label-off",
    "guidance-off-its-sha",
    "trial-tools-off-v1",
    "trial-guard-off-its-row",
    "trial-outcome-label-off-its-row",
    "arm-a-row-guard-on",
    "arm-b-row-guard-off",
    "arms-differ-in-the-frame",
    "arms-differ-in-the-outcome-label",
    "arm-rows-on-the-overview-set",
    "outcome-label-key-with-the-label-off",
    "b-shown-overview-beside-the-live-frame",
    "frame-on-turn-line-without-the-overview-key",
    "overview-on-a-fits-trial",
    "pair-step1-request-text-differs",
    "pair-step1-tools-differ",
    "reground-line",
    "page-overview-key",
    "non-1120-body",
)


def _findings(which: str, *args: Any, sha: str) -> list[str]:
    return list(impl(which, "configuration_findings")(*args, guidance_sha256=sha))


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", CFG_FAULTS)
def test_configuration_findings(which: str, fault: str) -> None:
    """C-4, one hand trial per finding (reading 7 of the eval): the clean records have none; each fault gives at least
    one finding naming the trial (and the step), the pair or the setting -- a ``repeat_guard`` key, a fact, a stop or
    a guard reason in arm A's step lines or bodies; an arm-B turn line without a well-formed key or with stop counts
    off its step lines; a stop or a fact the ported rule does not justify, or the reverse (missing, the wrong kind, the
    other reason, not refused, on a non-repeat, on a B9 refusal); a guard reason in a voiced source or the history; a
    ``no_change`` off its source; ``LIVE_TOOL_GUIDANCE`` or the tools off their pins; a trial or the arm rows off C-1;
    and T-0138 B7's findings at HEAD's default (an overview beside the live frame, a frame-on turn line without its
    key, an overview on a fits task, a step-1 pair that differs, a reground line, a page-overview key, a non-1120
    body)."""
    if fault.startswith("outcome-on-"):
        trials, steps, regrounds, bodies, arms = _cfg_outcome()
    else:
        (trials, steps, regrounds, bodies), arms = _cfg_clean(), HAND_ARMS
    clean = _findings(which, trials, steps, regrounds, bodies, arms, sha=GUIDANCE_SHA)
    assert clean == [], f"premise: the clean run has no finding: {clean}"
    trials, steps, regrounds, bodies, arms, sha, named = _cfg_fault(fault)
    found = _findings(which, trials, steps, regrounds, bodies, arms, sha=sha)
    assert found and any(named in f for f in found), (fault, found)


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """C-4's guard-too-wide side (**adversarial**; the failure mode is a fork that keeps T-0138's frame-off arm B, or a
    guard check that voids every run): arm B's S7 step-1 body shows the overview frame, as arm A's does -- no finding;
    the S9 pair differs after step 1 (arm B's fact and stop) -- none; re-captured step-1 image bytes -- none; a form
    fill in arm B (three ``type_at`` on one screen and a click) with no fact or stop -- none; a B9-refused repeat with no
    fact -- none; an arm-B trial that timed out with no turn line is not judged on the key; a trial that sent nothing
    is not judged at all; with T-0136 on, a click T-0136 read as ``no visible change`` while B8 said ``updated`` counts
    as no change and its fact carries T-0136's ``outcome`` -- none (B-3's S-OUT-1, the consistency rule)."""
    o_trials, o_steps, o_regrounds, o_bodies, o_arms = _cfg_outcome()
    found = _findings(which, o_trials, o_steps, o_regrounds, o_bodies, o_arms, sha=GUIDANCE_SHA)
    assert found == [], ("T-0136 on", found)
    trials, steps, regrounds, bodies = _cfg_clean()
    b7 = t38._final_user(t38._gen_of(bodies, "B", "S7"))["parts"]
    assert t38.frame_label(1.5) in [p.get("text") for p in b7], "premise: arm B shows the frame too"
    assert len(t38._find(bodies, "B", "S9", 0)["requests"]) == 3

    def check(label: str) -> None:
        found = _findings(which, trials, steps, regrounds, bodies, HAND_ARMS, sha=GUIDANCE_SHA)
        assert found == [], (label, found)

    check("clean")
    for arm, task in (("A", "S9"), ("B", "S8")):
        t38._final_user(t38._gen_of(bodies, arm, task))["parts"][0] = _i(f"{arm}{task}0-recaptured")
    check("image bytes")
    name = {"x": 300, "y": 200, "text": "Grace Hopper"}
    steps = [s for s in steps if (s["arm"], s["task"]) != ("B", "S8")] + [
        _ex("B", "S8", 0, 1, "type_at", name, "f"),
        _ex("B", "S8", 0, 2, "type_at", {**name, "y": 260, "text": "4"}, "f"),
        _ex("B", "S8", 0, 3, "type_at", {**name, "y": 320, "text": "19:30"}, "f"),
        _ex("B", "S8", 0, 4, "click_at", _xy(300, 400), "f"),
        _answer("B", "S8", 0, 5, ["Booked."]),
    ]
    # B-8 by hand: no fact, no stop; the judged steps 2-4 are all on f's first visit
    t38._find(trials, "B", "S8", 0)["repeat_guard_logged"] = _rg(max_screen_visits=1)
    check("form fill")
    steps = [s for s in steps if (s["arm"], s["task"]) != ("B", "S7")] + [
        _ex("B", "S7", 0, 1, "click_at", _xy(160, 933), "q", frame_kind="overview"),
        _ex("B", "S7", 0, 2, "click_at", _xy(500, 500), "r"),
        _ref("B", "S7", 0, 3, "click_at", _xy(500, 500), "r", FRESHNESS_REASON),
        # the refused step 3 is no instance: step 4's fact names step 2 (reviewer mutation 2 at the finding)
        _ex("B", "S7", 0, 4, "click_at", _xy(500, 500), "r", fact=2, fact_shot="updated"),
        _answer("B", "S7", 0, 5, ["Added."]),
    ]
    # B-8 by hand: one fact (step 4) whose own step changed; the judged steps 2 and 4 on r's first visit (the B9
    # refusal is never judged)
    t38._find(trials, "B", "S7", 0)["repeat_guard_logged"] = _rg(repeats=1, max_screen_visits=1)
    check("a B9-refused repeat")
    t38._find(trials, "B", "S8", 0).update(
        timed_out=True, session_id=None, repeat_guard_logged=None
    )
    check("no turn line")
    t38._find(trials, "B", "S9", 0).update(scanned_requests=[], repeat_guard_logged=None)
    check("sent nothing")


# =====================================================================================================
# build_report and render_markdown (harness only)
# =====================================================================================================


def _metric_run() -> tuple[list[Any], list[Any]]:
    ta, sa = _metric_records("A")
    tb, sb = _metric_records("B")
    return ta + tb, sa + sb


PROBES_OK = t38.PROBES_OK


def test_build_report_counts_equal_the_hand_counts() -> None:
    """``build_report`` over both arms' metric records: each arm's C-3 counts are ``HAND``'s, McNemar and the decision
    inputs are the oracle's, and the label is ``no change`` (W1(A) 7, W1(B) 6), as the oracle decides."""
    trials, steps = _metric_run()
    report = api("build_report")(
        trials, steps, [], [], PROBES_OK, arms=HAND_ARMS, guidance_sha256=GUIDANCE_SHA
    )
    assert report["void"] is False and report["configuration_findings"] == [], report[
        "configuration_findings"
    ]
    for arm in ("A", "B"):
        for key, want in HAND[arm].items():
            assert report["arms"][arm][key] == want, (arm, key, report["arms"][arm][key], want)
    want_mc = oracle.mcnemar_sections(trials, steps)
    assert {k: report["mcnemar"]["pooled"][k] for k in ("n", "b", "c")} == {
        k: want_mc["pooled"][k] for k in ("n", "b", "c")
    }
    for arm in ("A", "B"):
        want = oracle.decision_inputs(
            [t for t in trials if t["arm"] == arm], [s for s in steps if s["arm"] == arm], []
        )
        assert report["decision"]["inputs"][arm] == want, (arm, report["decision"]["inputs"][arm])
    assert report["decision"]["label"] == "no change", report["decision"]


@pytest.mark.parametrize("fault", ["ps2-trap-passes", "ps3-egress", "guidance", "reground"])
def test_build_report_voids(fault: str) -> None:
    """C-2/C-4 wiring: a P-S2 trap that passes or a P-S3 egress record makes the run ``void``; the guidance sha off its
    pin or a reground line makes it ``void`` with ``configuration: ...``."""
    trials, steps = _metric_run()
    probes = copy.deepcopy(PROBES_OK)
    regrounds: list[dict[str, Any]] = []
    sha = GUIDANCE_SHA
    if fault == "ps2-trap-passes":
        probes["ps2"] = {**t38.PS2_ROWS["s14-trap-passes"][0], "passed": True}
    elif fault == "ps3-egress":
        probes["ps3"] = {**t38.PS3_ROWS["one-egress-record"][0], "passed": False}
    elif fault == "guidance":
        sha = "0" * 64
    else:
        regrounds = [{"arm": "A", "task": "S6", "repeat": 0, "step": 1, "outcome": "ok"}]
    report = api("build_report")(
        trials, steps, regrounds, [], probes, arms=HAND_ARMS, guidance_sha256=sha
    )
    d = report["decision"]
    assert d["label"] == "void", d
    if fault in ("guidance", "reground"):
        assert str(d["reason"]).startswith("configuration: "), d
    else:
        assert report["void"] is True, report["void"]


def test_render_markdown_states_w1_and_not_exercised() -> None:
    """C-5 "stated in the report" (re-targeted): both arms' W1, that ``not exercised`` is a legitimate result, and the
    suite's baseline context (T-0138's S9 silent scrolls)."""
    trials, steps = _metric_run()
    report = api("build_report")(
        trials, steps, [], [], PROBES_OK, arms=HAND_ARMS, guidance_sha256=GUIDANCE_SHA
    )
    text = api("render_markdown")(report, {"live_default": {**D_HEAD, **HEAD_REST}})
    assert isinstance(text, str) and text
    assert "W1" in text and re.search(r"\b7\b", text) and re.search(r"\b6\b", text), (
        "C-3: W1 per arm"
    )
    assert "not exercised" in text, "C-5: not exercised is stated as a legitimate result"
    assert "S9" in text and re.search(r"T-0138|suite-baseline", text), "the suite baseline context"


# =====================================================================================================
# main on CPU: the dress (T-0131's engine rig, the FakeLiveBrowser, a fake LLM clicking four times)
# =====================================================================================================

DRESS_CALL = "<|tool_call>call:click_at{x:500,y:300}<tool_call|>"
#: a 1.0-screen page: the overview frame (on by default) gives way, ``fits``
DRESS_METRICS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 800.0}


def _dress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, Path, Any]:
    """A CPU run of ``main`` (T-0138's dress, one S9 turn per arm): every capture is the same live frame (same bytes,
    same phash), so B8 says ``unchanged`` after each click and the screen key never moves; the model clicks (500,
    300) at steps 1-4 and then says "Saved.". Returns (exit code, out-dir, the engine rig's state)."""
    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_kit
    from tests.unit._tool_turn_support import ToolLlm, answer, tool
    from tests.unit.test_t0140_session import overview_kit

    h = harness()

    def git(*args: str) -> str:
        return "" if args[:1] == ("status",) else t31.GIT + "\n"

    monkeypatch.setattr(h, "_git", git)
    for value in list(vars(h).values()):
        if isinstance(value, ModuleType) and hasattr(value, "_git") and value is not h:
            path = Path(getattr(value, "__file__", "") or "").resolve()
            if path.parent == HARNESS.parent.resolve():
                monkeypatch.setattr(value, "_git", git)
    kits = t38._stand_in_kits()
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(h=h, cli=cli, kit=kits[False], mp=monkeypatch, out=out)
    st_ = t31._engine(rig)
    st_.pages = []

    class Pages:
        origin = "http://127.0.0.1:9"

        def __init__(self, *a: Any, static_dir: Any = None, **kw: Any) -> None:
            st_.pages.append(static_dir if static_dir is not None else (a[0] if a else None))

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class Page:
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"page": "delivery", "saved": None, "selected": None}
            return {"boxes": [], "element": None}

    class Browser(FakeLiveBrowser):
        def __init__(
            self,
            *,
            now_ms: Any = None,
            viewport: Any = (1280, 800),
            echo: str = "pixels",
            **kw: Any,
        ):
            super().__init__(
                [live_frame("dress-same")],
                post_frames=[live_frame("dress-same") for _ in range(6)],
                now_ms=now_ms or (lambda: 0),
                echo=echo,
            )
            self.page_metrics = dict(DRESS_METRICS)
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st_.browsers.append(self)
            st_.pending_browser = self

    def turn_llm(browser: Any) -> Any:
        llm = ToolLlm([answer("Let me pick it. "), tool(DRESS_CALL)])
        llm.script_steps(
            [tool(DRESS_CALL)], [tool(DRESS_CALL)], [tool(DRESS_CALL)], [answer("Saved.")]
        )
        return llm

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        if live_browser is None:
            return (kits[bool(page_overview)], None)
        return (overview_kit(live_browser) if page_overview else live_kit(live_browser), None)

    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(t31, "_turn_llm", turn_llm)
    monkeypatch.setattr(h, "run_probes", lambda base_url, model: t31._probes())
    monkeypatch.setattr(h, "probe_ps1", lambda *a, **k: copy.deepcopy(t38.PS1_OK))
    monkeypatch.setattr(h, "probe_ps2", lambda *a, **k: copy.deepcopy(t38.PS2_OK))
    monkeypatch.setattr(h, "probe_ps3", lambda *a, **k: copy.deepcopy(t38.PS3_OK))
    monkeypatch.setattr(h, "order", lambda seed: [("A", "S9", 0), ("B", "S9", 0)])
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    return rc, out, st_


def test_main_dress_repeat_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """C-1/C-3/C-4 from a real ``main`` over group B's session: four identical clicks on one unchanging screen. Arm A
    (guard off) dispatches all four and logs no fact, no stop and no turn-line key; arm B dispatches three, its steps
    2 and 3 carry facts of 1 and 2, its step 4 is refused with ``NO_PROGRESS_REPEAT_REASON`` (``no_progress``
    ``repeat``, a fact of 3), and its turn line counts the stop. Every call's ``screen`` is the same non-``None``
    executor phash (never the capture identity), ``call_arguments`` are the model's, ``no_change`` is ``True`` from
    ``screenshot``. W1 is 3 in each arm (arm B's refused fourth counted), and the eval oracle's re-derivations agree
    with ``report.json``."""
    from local_astra import cli

    monkeypatch.setattr(
        cli, "LIVE_DEFAULT_REPEAT_GUARD", False
    )  # T-0179 A-5: the A/B's off-arm default
    rc, out, st_ = _dress(tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = t38._read(out)
    assert len(st_.browsers) == 2, "a fresh browser per turn"
    dispatched = {
        arm: sum(1 for inv in b.invocations if inv.name == "click_at")
        for arm, b in zip(("A", "B"), st_.browsers, strict=True)
    }
    assert dispatched == {"A": 4, "B": 3}, f"B-5: the stop dispatches nothing: {dispatched}"
    by = {t["arm"]: t for t in run["trials"]}
    assert (by["A"]["repeat_guard"], by["B"]["repeat_guard"]) == (False, True), by
    assert by["A"]["repeat_guard_logged"] is None, by["A"]["repeat_guard_logged"]
    logged = by["B"]["repeat_guard_logged"]
    assert isinstance(logged, dict) and set(logged) == set(RG_KEYS), logged
    # B-8 as merged, by hand: steps 2 and 3 carried facts and their own steps showed no change; one repeat stop; every
    # judged call on the screen's first visit
    assert logged == {
        "repeats": 2,
        "no_change_repeats": 2,
        "stops_repeat": 1,
        "stops_recurring": 0,
        "max_screen_visits": 1,
    }, logged
    for arm in ("A", "B"):
        mine = sorted((s for s in run["steps"] if s["arm"] == arm), key=lambda s: int(s["step"]))
        calls = [s for s in mine if s["call_name"] is not None]
        assert [s["call_name"] for s in calls] == ["click_at"] * 4, (arm, mine)
        assert all(s["call_arguments"] == {"x": 500, "y": 300} for s in calls), calls
        screens = {s["screen"] for s in calls}
        assert len(screens) == 1 and None not in screens, (arm, screens)
        (screen,) = screens
        assert not str(screen).startswith("capture:"), (
            "B-2: the screen key is never the capture identity"
        )
        executed = [s for s in calls if s["status"] == "executed"]
        assert all(
            s["no_change"] is True and s["no_change_source"] == "screenshot" for s in executed
        ), executed
        if arm == "A":
            assert [s["status"] for s in calls] == ["executed"] * 4, calls
            assert [s["repeat_of_step"] for s in mine] == [None] * len(mine), mine
            assert [s["repeat_fact"] for s in mine] == [None] * len(mine), mine
            assert [s["no_progress"] for s in mine] == [None] * len(mine), mine
        else:
            assert [s["status"] for s in calls] == ["executed"] * 3 + ["refused"], calls
            assert [s["repeat_of_step"] for s in calls] == [None, 1, 2, 3], calls
            assert [s["no_progress"] for s in calls] == [None, None, None, "repeat"], calls
            assert calls[3]["reason"] == REPEAT_REASON, calls[3]
            # B-6 / S-R1 as merged: the mapping on both executed repeats and on the stop, step j's outcome being B8's
            # (T-0136 off by default): {"step": j, "action": <step j's action string>, "screenshot": "unchanged"}
            assert calls[0]["repeat_fact"] is None, calls[0]
            for s, j in zip(calls[1:], (1, 2, 3), strict=True):
                fact = s["repeat_fact"]
                assert isinstance(fact, dict) and set(fact) == {"step", "action", "screenshot"}, (
                    fact
                )
                assert (fact["step"], fact["screenshot"]) == (j, "unchanged"), fact
                assert isinstance(fact["action"], str) and fact["action"], fact
            assert all(s["guard_outcome"] == {"screenshot": "unchanged"} for s in executed), (
                executed
            )
            answer_step = mine[-1]
            assert answer_step["call_name"] is None and answer_step["voiced_sources"] == [
                "Saved."
            ], answer_step
    arms = run["report"]["arms"]
    assert (arms["A"]["w1"], arms["B"]["w1"]) == (3, 3), (arms["A"]["w1"], arms["B"]["w1"])
    assert arms["B"]["stops"] == {"repeat": 1, "recurring": 0}
    assert arms["B"]["after_stop"]["answered_voiced"] == 1, arms["B"]["after_stop"]
    assert run["manifest"]["guidance_sha256"] == GUIDANCE_SHA
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]
    for arm in ("A", "B"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_safety_and_latency_re_derived(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)


def test_main_dress_repeat_stop_with_outcome_label_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The same dress with T-0136's action-outcome label on as HEAD's default (it may be by the /launch run). Every
    capture is the same frame, so T-0136 measures ``no visible change`` after every dispatched click (its ``none``
    bucket): arm A dispatches four, arm B three (its fourth is refused, nothing dispatched, nothing measured).

    Hand-worked per step: arm A's steps 1-4 and arm B's steps 1-3 carry ``guard_outcome`` ``{"outcome": "no visible
    change", "screenshot": "unchanged"}``, ``no_change`` ``True``, ``no_change_source`` ``"t0136"`` (S-OUT-2: the
    label's text, when on, is the outcome) -- arm A's recomputed from the log, arm B's held by the session, and equal
    on the matching steps. Arm B's refused step 4 has no measured label (no after-screen), so its source is never
    ``t0136``. Arm B's repeat facts at steps 2-4 carry step j's outcome text. Both arms' rows and trials show
    ``action_outcome`` ``True`` and the configuration check finds nothing."""
    from local_astra import cli

    monkeypatch.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    monkeypatch.setattr(
        cli, "LIVE_DEFAULT_REPEAT_GUARD", False
    )  # T-0179 A-5: the A/B's off-arm default
    rc, out, st_ = _dress(tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = t38._read(out)
    dispatched = [sum(1 for inv in b.invocations if inv.name == "click_at") for b in st_.browsers]
    assert dispatched == [4, 3], f"arm A then arm B, a fresh browser each: {dispatched}"
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]
    rows = {r["label"]: r for r in run["manifest"]["regime"]["arms"]}
    assert (rows["A"]["action_outcome"], rows["B"]["action_outcome"]) == (True, True), rows
    assert (rows["A"]["repeat_guard"], rows["B"]["repeat_guard"]) == (False, True), rows
    by = {t["arm"]: t for t in run["trials"]}
    assert (by["A"]["action_outcome"], by["B"]["action_outcome"]) == (True, True), by
    # T-0136's per-turn counts: four dispatched clicks in A, three in B, every one on an unchanged frame
    none_only = {"near": 0, "elsewhere": 0, "page": 0, "unmeasured": 0}
    assert by["A"]["action_outcome_logged"] == {"none": 4, **none_only}, by["A"]
    assert by["B"]["action_outcome_logged"] == {"none": 3, **none_only}, by["B"]
    measured = {"outcome": NVC, "screenshot": "unchanged"}
    calls = {
        arm: sorted(
            (s for s in run["steps"] if s["arm"] == arm and s["call_name"] is not None),
            key=lambda s: int(s["step"]),
        )
        for arm in ("A", "B")
    }
    assert [s["status"] for s in calls["A"]] == ["executed"] * 4, calls["A"]
    assert [s["status"] for s in calls["B"]] == ["executed"] * 3 + ["refused"], calls["B"]
    for arm, n in (("A", 4), ("B", 3)):
        for s in calls[arm][:n]:
            assert (s["guard_outcome"], s["no_change"], s["no_change_source"]) == (
                measured,
                True,
                "t0136",
            ), (arm, s)
    for a, b in zip(calls["A"][:3], calls["B"][:3], strict=True):
        assert (a["guard_outcome"], a["no_change"], a["no_change_source"]) == (
            b["guard_outcome"],
            b["no_change"],
            b["no_change_source"],
        ), (a, b)
    stop = calls["B"][3]
    assert stop["no_progress"] == "repeat" and stop["no_change_source"] != "t0136", stop
    assert "outcome" not in (stop["guard_outcome"] or {}), stop
    for s, j in zip(calls["B"][1:], (1, 2, 3), strict=True):
        fact = s["repeat_fact"]
        assert isinstance(fact, dict), s
        assert (fact["step"], fact.get("outcome"), fact["screenshot"]) == (j, NVC, "unchanged"), (
            fact
        )
    assert [s["repeat_fact"] for s in calls["A"]] == [None] * 4, calls["A"]
    arms = run["report"]["arms"]
    assert (arms["A"]["w1"], arms["B"]["w1"]) == (3, 3), (arms["A"]["w1"], arms["B"]["w1"])
    for arm in ("A", "B"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_safety_and_latency_re_derived(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)
    oracle.test_configuration_findings_re_derived(run)
    oracle.test_probes_re_derived(run)
    oracle.test_mcnemar_re_derived(run)
    oracle.test_decision_re_derived(run)
    oracle.test_live_default_recorded(run)


# =====================================================================================================
# C-6: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_OUT = f"logs/t0137-repeat-guard-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0137_repeat_guard_ab.py"
L_EVAL = "tests/eval/test_t0137_ab.py"
GROUP_C = (
    "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0137 group C: the locked repeat-guard A/B"
)
#: merges that do not count (reading 4): verbatim subjects on main at BASE_C, then made-up ones
OTHERS = (
    "1ed23fa2977f814c4d47a198ca15fc83a31cd903 Merge T-0137 group A: the pure repeat/no-progress rule "
    "(local_astra/conductor/progress.py; nothing calls it yet)",
    "c2c5291c437dd66257b578370a81fd0655df2c36 Merge T-0137-a-fix (O-A2: trailing no-change run)",
    "1f9894a6638c9ee9f13a477e4f92aeaf11627b4e Merge T-0137-a-impl (progress rule module)",
    "cd0929a54ced9b82e345019b575ac24382b0f03e Merge T-0138 group B: locked baseline A/B on the S6-S15 suite",
    "eca4d88605b410f8cb751f2060cd82270eb5eed1 Merge T-0151 group B: locked 2-arm GPU A/B of --overview-frame on S1-S5",
    "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a Merge T-0137 group B: Session(repeat_guard), cli, config, docs",
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge T-0137-c-impl (the repeat-guard A/B harness)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0137c: a follow-up",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge T-0137: the repeat guard",
    "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge branch 'T-0137-c-ab' into T-0137-c-impl",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge main into T-0137-c-ab",
)
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "tests/eval/test_t0138_ab.py",
    "tests/eval/test_live_smoke.py",
    "tests/eval/test_t0131_ab.py",
    "tests/eval/test_t0132_ab.py",
    "scripts/gpu/t0135_thought_call_ab.py",
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0132_crop_budget_ab.py",
    "scripts/gpu/t0131_grounding_ab.py",
    "scripts/gpu/t0111g_live_smoke.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/thought_call.py",
    "local_astra/pages/suite_tasks.py",
    "local_astra/eval/cd/stats.py",
    "local_astra/conductor/progress.py",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch, merges: tuple[str, ...] = (*OTHERS, GROUP_C), filler: int = 0
) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0137 script, harness, eval and merges."""
    assert LOCKED.is_file(), f"T-0137 C-6: {LOCKED.relative_to(REPO)} does not exist yet"
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", t38._git_stub(merges, filler))
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0137_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json bodies.jsonl; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


LOCKED_REFUSALS: dict[str, dict[str, Any]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """C-6: ``REFUSED`` and exit 2 before the wheel build, the boot or any step: an argument, a dirty tree, an Astra
    port bound (STT 8080/8081 must be down), running containers, ``services/vllm.sh`` changed since 806da12, this
    SHA's out-dir present."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_needs_the_group_c_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C-6 (**adversarial**, reading 4): without ``Merge T-0137 group C`` -- the log holding groups A's and B's merges,
    the ``-impl``/``-fix`` and branch merges, a bare ``Merge T-0137:``, ``Merge T-0137c`` and T-0138's and T-0151's
    group merges -- the script refuses before anything starts."""
    res = _locked(monkeypatch, OTHERS)(tmp_path)
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """C-6, the other side of every refusal: the harness under ``timeout -k 60s 100m`` (O-C) with a loopback ``--base-url`` and
    ``--out-dir logs/t0137-repeat-guard-ab-<sha7>``, then ``tests/eval/test_t0137_ab.py -m gpu`` with
    ``T0137_RUN_DIR`` set to it; every ``uv run`` uses ``--locked``; exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    res = _locked(monkeypatch)(tmp_path)
    assert res.rc == 0, res.output[-3000:]
    steps = res.steps()
    runs = [c for c in steps if c.names(L_HARNESS)]
    evals = [c for c in steps if c.names(L_EVAL)]
    assert runs and evals and steps.index(runs[0]) < steps.index(evals[0]), [c.argv for c in steps]
    out = runs[0].flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), runs[0].argv
    assert (runs[0].flag("--base-url") or "").startswith(("http://127.0.0.1", "http://[::1]")), (
        runs[0].argv
    )
    assert "gpu" in (evals[0].flag("-m") or ""), evals[0].argv
    assert evals[0].env.get("T0137_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv) and len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 100 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_long_merge_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """C-6 (the here-string safeguard): a merge log past the 64 KiB pipe buffer after the group C line still passes."""
    res = _locked(monkeypatch, (*OTHERS, GROUP_C), filler=3000)(tmp_path)
    assert res.rc == 0 and "REFUSED" not in res.output, (res.rc, res.output[-2000:])
    assert [c for c in res.steps() if c.names(L_EVAL)], "the eval did not run"


def test_locked_script_eval_after_a_failed_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C-6 "then runs the eval if report.json exists": the harness failing after it wrote ``report.json`` still runs the
    eval over that report; the script exits 1."""
    res = _locked(monkeypatch)(tmp_path, fail_match=L_HARNESS, fail_code=1)
    assert [c for c in res.steps() if c.names(L_EVAL)], (
        f"the eval did not run: {res.output[-2000:]}"
    )
    assert res.rc == 1, (res.rc, res.output[-2000:])


def _boot(lines: list[str]) -> list[str]:
    """Reading 3: T-0138 reading 4's boot block, less the lines naming T-0151, T-0138 or T-0137."""
    starts = [k for k, ln in enumerate(lines) if ln.startswith("make plugin-wheel")]
    ends = [k for k, ln in enumerate(lines) if "device memory after boot" in ln]
    assert starts and ends and starts[0] < ends[0], "the script has no boot block"
    block = lines[starts[0] : ends[0] + 1]
    return [
        ln for ln in block if not re.search(r"t015[01]|T-015[01]|t0138|T-0138|t0137|T-0137", ln)
    ]


def test_t0151_boot_block_premise() -> None:
    """Premise of reading 3 (green): T-0151's locked script has a boot block of the shape the row below compares."""
    boot = _boot(t38._code(T0151_LOCKED))
    assert len(boot) > 20 and any("services/vllm.sh" in ln for ln in boot), boot


def test_locked_script_shape() -> None:
    """C-6 by text: the boot verbatim from T-0151's (reading 3); the ``Merge T-0137 group C`` check through a
    here-string, never a pipe into ``grep``, and no T-0138/T-0151 merge check left from the fork; ``timeout -k 60s
    100m`` (O-C); ``--locked`` on every ``uv run``; STT ports refused; the out-dir, the eval only if ``report.json`` exists;
    ``bash -n`` clean; the frozen harnesses never named."""
    assert LOCKED.is_file(), f"T-0137 C-6: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    code = t38._code(LOCKED)
    body = "\n".join(code)
    assert _boot(code) == _boot(t38._code(T0151_LOCKED)), "C-6: the boot is not T-0151's verbatim"
    assert "timeout -k 60s 100m" in body, body[:600]
    checks = [ln for ln in code if "Merge T-0137" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("group C" in ln for ln in checks), checks
    assert not [ln for ln in code if re.search(r"Merge T-01(38|5[01])", ln)], (
        "the fork kept another merge check"
    )
    assert not [
        ln
        for ln in code
        if "grep" in ln and ("MERGES" in ln or "--merges" in ln) and "|" in ln.split("grep")[0]
    ], "the merge log is piped into grep"
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert any("8080" in ln and "8081" in ln for ln in code), "STT ports refused"
    assert (
        "t0137-repeat-guard-ab-" in body
        and L_EVAL in body
        and L_HARNESS in body
        and "T0137_RUN_DIR" in body
    )
    for old in (
        "t0138_suite_ab",
        "t0151_overview_frame_ab",
        "test_t0138_ab.py -m",
        "T0138_RUN_DIR",
        "T0151_RUN_DIR",
    ):
        assert old not in body, f"the script names {old}"
    assert "-m gpu" in body and "report.json" in body
    assert "806da12" in body and "services/vllm.sh" in body


def test_harness_run_budget_fits_the_locked_bound() -> None:
    """C-6 (O-C): ``RUN_BUDGET_S`` (85 min) sits inside ``timeout -k 60s 100m`` with room for the report and the eval."""
    assert LOCKED.is_file(), f"T-0137 C-6: {LOCKED.relative_to(REPO)} does not exist yet"
    assert api("RUN_BUDGET_S") == 85 * 60 < 100 * 60
    assert "timeout -k 60s 100m" in "\n".join(t38._code(LOCKED))


# =====================================================================================================
# C-7 and C-8
# =====================================================================================================


def test_eval_never_imports_the_harness_or_the_rule() -> None:
    """C-7 (green): ``tests/eval/test_t0137_ab.py`` imports no script, no ``importlib``, not
    ``local_astra.conductor.progress`` (the rule is ported, never imported) and nothing of ``local_astra`` at all; no
    code string names the harness."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    mods: list[str] = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
            names += [a.name for a in node.names]
    assert not [m for m in mods if m.startswith(("scripts", "local_astra")) or "importlib" in m], (
        mods
    )
    assert "progress" not in names, names
    strings = t38._non_docstring_strings(tree)
    assert not [s for s in strings if "t0137_repeat_guard_ab" in s or "scripts/gpu" in s], (
        "the eval names the harness"
    )


#: sha256 of each file at BASE_C (d4adf4f), by the test-author 2026-10-25 (``git show d4adf4f:<path> | sha256sum``).
#: C-8 names T-0151's, T-0150's and T-0140's B files; T-0138's harness, locked script and eval are pinned too
#: (reading 5: the fork's parent, and the oracle this eval imports).
FROZEN_SHA256 = {
    "scripts/gpu/t0151_overview_frame_ab.py": "1d54358962c807d1eb3e2e3624044a51c36c0ef0b536119043448a6b33cb8adb",
    "scripts/locked/t0151_overview_frame_ab.sh": "e658f26687bcf1c3da5e458aca0ea0b405d5922425e22c6f5c0382c016334d91",
    "tests/eval/test_t0151_ab.py": "3052db6e25a982dd6d5bbd0444d62187d04048fa5554e627e965d5659587e0e0",
    "tests/unit/test_t0151_ab_harness.py": "bee120bcdf2c781483d00e9c22dfdb196c5ee5d590b3fd692698f443b883d3ef",
    "scripts/gpu/t0150_auto_overview_ab.py": "580b88e0152a11a30d508090ef8669926ea28addb662be135aa09c66d823d4ed",
    "scripts/locked/t0150_auto_overview_ab.sh": "0f58e5ff2ae08e1bdf0ba2d845949b1ba5da74aa7505278fc7ee47d3fc51eb44",
    "tests/eval/test_t0150_ab.py": "cf262e89c518b52f58285ee7c3b20f7c457c8a6ae63132bf60b4c671fc7b7e2e",
    "tests/unit/test_t0150_ab_harness.py": "530f5c9d4f520bcbb52cdd993c1abaa1292b889eaecbafb22cf75bf43a28e019",
    "scripts/gpu/t0140_overview_ab.py": "675c75d9eb8c0f4c0b4ade7a636086b2904c30fe1687061da27b52f4b91647ec",
    "scripts/locked/t0140_overview_ab.sh": "95679378f785b7a2ea02e91c1bf8bfc23843879c0ec1921ece40b179d802dc60",
    "tests/eval/test_t0140_ab.py": "4ad5fccf3ec4844597cb49ebe6e0114b604fe34b33bafdd5a476fdcadad0058e",
    "tests/unit/test_t0140_ab_harness.py": "d9224d98d3f12f8e17e9e88abd008d039e2179ba891d72842c8a3aae9a785f89",
    "scripts/gpu/t0138_suite_ab.py": "cfcc27f38f96f3cf8e41b0f2132d94b47c9962ec43bd6602ee4cf9b0f2b6ccc2",
    "scripts/locked/t0138_suite_ab.sh": "7b458c9d860b3b8fea68a59dc4b22d43085b84ebf05dfb12c0567b402c5e9a05",
    "tests/eval/test_t0138_ab.py": "a840863b0c9762c0ca58f92a120d74900ad048176d52e9fdfb7d4956f673bce8",
}


@pytest.mark.parametrize("rel", list(FROZEN_SHA256))
def test_frozen_files_unchanged(rel: str) -> None:
    """C-8 negative (content pins, green now and forever)."""
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == FROZEN_SHA256[rel], f"T-0137 C-8: {rel} changed ({got})"


#: group C's owned files (docs/tasks/T-0137.md "Owned files", Group C) and the orchestrator's
GROUP_C_FILES = {
    "scripts/gpu/t0137_repeat_guard_ab.py",
    "scripts/locked/t0137_repeat_guard_ab.sh",
    "tests/unit/test_t0137_ab_harness.py",
    "tests/eval/test_t0137_ab.py",
    "docs/cd/repeat-guard-ab.md",
    "docs/tasks/T-0137.md",
    "PROGRESS.md",
    "docs/14-computer-use-harness.md",
    "tests/unit/test_guards_use_fixed_refs.py",
}


def test_no_local_astra_change_in_group_c() -> None:
    """C-8 negative: no ``local_astra/`` file changes in group C, and nothing outside group C's owned files, over the
    closed range ``BASE_C..TIP_C`` (two literals, T-0094), BASE_C an ancestor of TIP_C. Power: the same predicate sees
    group A's new ``progress.py`` over its merge's range (1f9894a^1..1f9894a^2), and group C's own harness over
    BASE_C..TIP_C (red until TIP_C is bumped to the implementation tip; reading 6 on BASE_C)."""
    gr.require_commits(BASE_C, TIP_C, "1f9894a")
    assert gr.is_ancestor(BASE_C, TIP_C), f"{BASE_C} is not an ancestor of {TIP_C}"
    assert "local_astra/conductor/progress.py" in gr.changed(
        "1f9894a^1", "1f9894a^2", "local_astra"
    )
    assert "scripts/gpu/t0137_repeat_guard_ab.py" in gr.changed(BASE_C, TIP_C, "scripts"), (
        f"T-0137 C-8: TIP_C {TIP_C} is not bumped to group C's implementation tip yet"
    )
    changed = gr.changed(BASE_C, TIP_C, "local_astra")
    assert changed == [], f"T-0137 C-8: group C changed {changed}"
    outside = sorted(set(gr.changed(BASE_C, TIP_C, ".")) - GROUP_C_FILES)
    assert outside == [], f"T-0137 C-8: group C changed files it does not own: {outside}"
