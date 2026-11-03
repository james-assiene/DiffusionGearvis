"""T-0150 group B: the locked 2-arm auto-overview A/B harness's pure parts, its ``main`` on CPU and its locked script
(unit; CPU; no engine, no browser). docs/tasks/T-0150.md B1-B8, H-T150-1 (approved 2026-10-24: T-0140's margins, S5
report-only, ``overview not shown`` below 18/20, the P8 census a void), H-T150-3 (step 1 only) and R-b (a copy of
T-0140's harness, never a parameterisation). BASE_B 1bed93f (``Merge T-0150 group A`` at 55db4e2, then the parking
commit).

Rows that need the harness (``scripts/gpu/t0150_auto_overview_ab.py``) or the locked script
(``scripts/locked/t0150_auto_overview_ab.sh``) were activated 2026-10-24 (rulings O-B) and are red on a named assertion
in ``harness()``/``api()``/``_locked()`` ("does not exist yet" / "has no ..."), never on an import error: every harness
symbol is looked up inside the row. Rows parametrised on ``which`` run once against the harness and once
against the eval oracle (``tests/eval/test_t0150_ab.py``, green now), so a wrong oracle is caught on CPU. Nothing here
imports T-0140's harness, its eval or its unit module (R-b; B8 pins them byte-identical).

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not). Unless stated, a name keeps T-0140's harness's shape.

Imported, never edited (B1)
    T-0134's harness (``F``) and through it T-0132's (``P``), T-0131's (``T``) and T-0111g's (``G``), as T-0140's
    harness imports them: ``TASKS``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
    ``non_loopback_sockets``, ``turn_records``; ``probe_p1``, ``run_probes``, ``false_claim``, ``click_score``,
    ``config_evidence``; ``wilson``, ``newcombe``; ``score_v2``, ``over_click``, ``no_see``, ``coord_mismatch``. Module
    attributes whose objects come from those files. ``scripts/gpu/t0140_overview_ab.py`` is neither imported nor
    loaded (no module object or function of it reachable from the harness, no string constant naming it).

Seams ``main`` reads at call time (as T-0140's)
    ``_git``, ``order``, ``run_probes`` (P1-P3), ``probe_p6``, ``probe_p7`` and ``probe_p8`` (synchronous callables
    returning the ``probes.json`` records, shapes in the oracle's ``p6_ok``/``p7_ok``/``p8_ok``; the harness adds
    ``passed``), ``PROCESS_START_S``; ``cli.grounding_of``, ``cli.feedback_of``, ``cli.recovery_of``,
    ``cli.page_overview_of`` and ``cli.auto_overview_of`` (through ``live_default``, read once per ``main``);
    ``cli.load_tool_kit`` (both arms ``cli.load_tool_kit(profile, live_browser=...)``, never ``page_overview=True``).

The arms (B1; never hard-coded)
    ``live_default(profile) -> {"visual_tokens", "reground", "live_feedback": None | {"screen_label", "page_extent"},
    "echo", "recover_thought_calls", "page_overview", "auto_overview"}`` (the last two must be ``False``).
    ``arm_rows(default) -> [row_A, row_B]``, each ``{"label", "visual_tokens", "reground", "live_feedback", "echo",
    "recover_thought_calls", "degeneracy_guard": True, "page_overview": False, "auto_overview", "tools_sha256": V1}``
    with A's ``auto_overview`` False and B's True. ``arm_table_sha(rows)`` as T-0140's. ``main`` passes every setting
    explicitly in both arms (``Session(..., visual_tokens=, reground=, live_feedback=, recover_thought_calls=,
    degeneracy_guard=True, auto_overview=<the arm's>)``, ``BrowserToolExecutor(echo=)``).

Constants
    T-0140's (``EXIT_*``, ``REPEATS = 20``, ``ORDER_SEED = 0``, ``RUN_BUDGET_S = 62 * 60``, ``PROCESS_START_S``), plus
    ``AUTO_OVERVIEW_ON = {"A": False, "B": True}`` and ``NOT_SHOWN_BELOW = 18``.

Pure functions (each also in the eval oracle, same name and shape; the harness's ``arm_metrics`` is the oracle's
``derive_arm``)
    ``order``, ``arm_rows``, ``arm_table_sha``, ``live_default_findings``; ``p6_ok``, ``p7_ok`` (the step-1 shape: 2
    images), ``p8_ok``; ``configuration_findings(trials, steps, regrounds, bodies, arms)`` (B4);
    ``arm_metrics(trials, steps)`` (B3; keys in ``HAND``); ``decision_inputs`` (keys in ``_arm``);
    ``decide``/``decide_with_reason`` with labels ``"auto overview"``, ``"overview not shown"``, ``"no change"``,
    ``"inconclusive"``, ``"void"``, ``"incomplete"`` and T-0140's reasons; ``wilson``/``newcombe``;
    ``build_report(trials, steps, regrounds, bodies, probes, *, arms, stopped=None)`` (``void`` over P1-P3, P6, P7
    and P8; ``differences`` keys ``"s4 B-A"``, ``"success B-A"``, ``"s5 B-A"``); ``body_line(arm, task, repeat, raw,
    records, overviews)`` (T-0140's: a ``tool_repair`` takes its pass-1 step, the generate and its retry step 1).

Records ``main`` writes (the eval's docstring has every field)
    T-0140's files; ``probes.json`` adds ``p8``; a trial adds ``auto_overview`` and ``auto_overview_logged``; a bodies
    line's ``overviews`` holds the turn's auto capture (executor call id ending ``-auto-overview``) at step 1.

Refusals (exit 2, before any socket, leaving no out-dir)
    T-0140's that still apply (dirty tree, non-loopback URL, out-dir present, repair off, ``LIVE_TOOL_GUIDANCE`` off
    its sha, ``Session`` without the guard's keywords, ``cli`` without ``page_overview_of``, a default with the page
    overview on), the per-arm tools check requiring v1 in both arms, a default with the auto overview on, ``Session``
    without the ``auto_overview`` keyword and ``cli`` without ``auto_overview_of``.

**Readings** (reported to the orchestrator; the oracle's docstring states the record-level ones)
    1. T-0140's readings 1 and 2 stand ("counts as a failed trial"), with O-B-8: a 400/413 attempt the session's retry
       recovers (a later 200 of the same step) is not an engine error, and not an acceptance failure either (the
       test-author's consequence); every other non-200 (5xx, connection failure, HTTP 0 not the guard's) is one.
    2. B4's placement is step 1 only (H-T150-3): the ``generate`` request, its over-long retry (also ``generate``) and
       the ``tool_repair`` of the step-1 call; anything else carrying the overview is out of place.
    3. B4's "unlogged absence" and "S1-S3 overview" as the oracle's docstring states them; the S1-S3 finding is raised
       by an overview record, a shown count, or an overview label/image in a body (any of them).
    4. ``overview not shown`` reads arm B's S4 turns whose turn line logged ``shown`` = 1 (A5's count, never the
       arm); the adoption clauses are checked first, as T-0140's.
    5. The locked script's check accepts ``Merge T-0150 group B`` only; ``Merge T-0150 group A`` (55db4e2, on main at
       BASE_B), a bare ``Merge T-0150:`` (O-B-4), ``Merge T-0140 group B``, ``Merge T-0150b`` and branch merges do
       not count.
    7. O-B-2: an arm-B turn line's ``shown`` and its step-1 ``generate`` bodies (the generate or its retry) must agree
       in both directions; a mismatch is a B4 finding.

Rulings O-B (docs/tasks/T-0150.md) accept the readings above as amended by O-B-2, O-B-4 and O-B-8.
    6. T-0140's refusal of a ``load_tool_kit`` without the ``page_overview`` keyword no longer applies (both arms are
       v1); not tested either way.

Regime: CPU, pure, hand-built records; the dress rows run ``main`` on CPU with T-0131's engine rig and the
``FakeLiveBrowser`` (fake LLM, one S4 turn per arm, a scripted 1.9-screen overview); no engine, no browser, no
measurement.
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
import json
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

from tests import _git_range as gr
from tests.eval import test_t0150_ab as oracle
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0150_auto_overview_ab.py"
T0140_HARNESS = REPO / "scripts" / "gpu" / "t0140_overview_ab.py"
T0111G_HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
T0131_HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0132_HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
T0134_HARNESS = REPO / "scripts" / "gpu" / "t0134_feedback_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0150_auto_overview_ab.sh"
#: T-0150 group B's BASE: main at 1bed93f (``Merge T-0150 group A`` 55db4e2 plus the parking commit). A literal,
#: never a moving ref.
BASE_B = "1bed93f"
#: T-0150 group B's tip: the last group B commit before the merge (8c83a09, the harness and locked script). B8's
#: scope row compares ``BASE_B..TIP_B``, two literals (T-0094), never the working tree.
TIP_B = "8c83a09"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

#: tools_v1.json, the tool set of both arms (A1/A10 of T-0140 pin the same literal); the overview set, which no arm uses
V1 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
V1_TOOLS = ["google_search", "click_at", "type_at", "scroll", "press_key"]
OV_TOOLS = [*V1_TOOLS, "page_overview"]
LABEL_FB = {"screen_label": True, "page_extent": False}
#: two hand-chosen live defaults (inputs, not HEAD's): HEAD's at BASE_B as cli gave it on 2026-10-24 (1120, ReGround
#: off, label, model echo, recovery on, both overviews off), and the settings before T-0134c/T-0135b
D_HEAD = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": dict(LABEL_FB),
    "echo": "model",
    "recover_thought_calls": True,
}
D_OLD = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": None,
    "echo": "pixels",
    "recover_thought_calls": False,
}
#: ``arm_table_sha`` of each default's two rows (written out in ``test_arm_rows_and_sha``), by the test-author
#: 2026-10-24 (hashlib over the canonical JSON of the hand-written rows, in a scratch script)
ARM_PINS = {
    "head": (D_HEAD, "81779b4acce15a94fee4cec1e1a0ab810fa9f1bbb692764a62cf8147c216ae8b"),
    "old": (D_OLD, "fcd00ce2f1c1b383fd4bedbb912e44cb637bb771a3bdc66f72e759ef821cfe53"),
}
#: the arms the hand-built records ran at (B1's rows over ``D_HEAD``, written out)
HAND_ARMS = [
    {
        "label": "A",
        **D_HEAD,
        "degeneracy_guard": True,
        "page_overview": False,
        "auto_overview": False,
        "tools_sha256": V1,
    },
    {
        "label": "B",
        **D_HEAD,
        "degeneracy_guard": True,
        "page_overview": False,
        "auto_overview": True,
        "tools_sha256": V1,
    },
]
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
FROM = {
    T0111G_HARNESS: (
        "LoopMeter",
        "sentinel_hits",
        "url_leaks",
        "log_complete",
        "non_loopback_sockets",
        "turn_records",
    ),
    T0131_HARNESS: ("probe_p1", "run_probes", "false_claim", "click_score", "config_evidence"),
    T0132_HARNESS: ("wilson", "newcombe"),
    T0134_HARNESS: ("score_v2", "over_click", "no_see", "coord_mismatch"),
}
DATA = ("TASKS",)
SCROLL_CALL = '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
#: T-0140 A3's fits string and T-0150 A5's counts, typed from the criteria
FITS = "the whole page fits on the browser screen; no overview was taken"
A5_KEYS = ("shown", "fits", "failed", "refused")


def ov_action(screens: float, clipped: bool) -> str:
    c = ", top 4.0 screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def ov_label(screens: float, clipped: bool) -> str:
    c = ", top 4.0 screens shown" if clipped else ""
    return (
        f"[Whole page, scaled to fit one screen: about {screens:.1f} screens tall{c}. "
        "The next image is the browser screen.]"
    )


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0150 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0150_auto_overview_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0150: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api("arm_metrics" if name == "derive_arm" else name)
    return getattr(oracle, name)


# =====================================================================================================
# B1: imports, the arm table, the order
# =====================================================================================================


def test_imports_never_copied() -> None:
    """B1: importing the harness opens no socket; the T-0111g, T-0131, T-0132 and T-0134 names are those files'
    objects, none is defined in the T-0150 harness; T-0140's harness is neither imported nor named in code (R-b: the
    fork is a copy)."""
    before = set(sys.modules)
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    _, defined = t31._top_level(tree)
    names = {n for ns in FROM.values() for n in ns} | set(DATA)
    assert sorted(defined & names) == [], (
        f"T-0150 B1: defined, not imported: {sorted(defined & names)}"
    )
    for origin, ns in FROM.items():
        for name in ns:
            src = Path(inspect.getsourcefile(api(name)) or "").resolve()
            assert src == origin.resolve(), f"T-0150 B1: {name} comes from {src}, not {origin.name}"
    assert [t.task_id for t in api("TASKS")] == list(TASK_IDS)
    # R-b: no module object, function or class of T-0140's harness is reachable from this one
    t0140 = T0140_HARNESS.resolve()
    added = [m for k in set(sys.modules) - before if (m := sys.modules.get(k)) is not None]
    for m in [*added, *(v for v in vars(h).values() if isinstance(v, ModuleType))]:
        f = getattr(m, "__file__", None)
        assert f is None or Path(f).resolve() != t0140, (
            f"T-0150 B1: {m.__name__} is T-0140's harness"
        )
    for name, value in vars(h).items():
        if inspect.isfunction(value) or inspect.isclass(value):
            try:
                src = inspect.getsourcefile(value)
            except TypeError:  # a built-in class (``ModuleType``, ...)
                continue
            assert src is None or Path(src).resolve() != t0140, (
                f"{name} comes from T-0140's harness"
            )
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    named = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
        and "t0140_overview_ab" in n.value
    ]
    assert named == [], f"T-0150 B1/R-b: the harness names T-0140's harness in code: {named}"


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("pin", list(ARM_PINS))
def test_arm_rows_and_sha(which: str, pin: str) -> None:
    """B1: over a live default (both overviews off), the two rows carry the default's five settings, the guard on, the
    page overview off and tools v1 in both arms, A without the auto overview, B with it (written out here), and
    ``arm_table_sha`` of them is the pin; swapping the rows, B on the overview set, or B without the auto overview
    gives another sha."""
    default, want_sha = ARM_PINS[pin]
    rows = impl(which, "arm_rows")({**default, "page_overview": False, "auto_overview": False})
    want = [
        {
            "label": "A",
            **default,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "tools_sha256": V1,
        },
        {
            "label": "B",
            **default,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": True,
            "tools_sha256": V1,
        },
    ]
    assert [dict(r) for r in rows] == want, rows
    sha = impl(which, "arm_table_sha")
    assert sha(rows) == want_sha
    assert sha([rows[1], rows[0]]) != want_sha
    assert sha([rows[0], {**rows[1], "tools_sha256": OV_SHA}]) != want_sha
    assert sha([rows[0], {**rows[1], "auto_overview": False}]) != want_sha


def test_constants() -> None:
    """B1/B5/B6: the constants (``RUN_BUDGET_S = 62*60`` from process start, ``NOT_SHOWN_BELOW = 18``)."""
    assert api("REPEATS") == 20 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 62 * 60, "T-0150 B6: RUN_BUDGET_S = 62*60"
    assert isinstance(api("PROCESS_START_S"), float), "B6: the budget runs from process start"
    assert api("PROCESS_START_S") <= time.monotonic()
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )
    assert api("NOT_SHOWN_BELOW") == 18, "H-T150-1 (c)"
    assert dict(api("AUTO_OVERVIEW_ON")) == {"A": False, "B": True}


#: ``cli.Feedback.live_feedback``'s choices as ``live_default`` records them (written out by hand, as T-0135 O-16)
FEEDBACK_DICT = {
    "off": {"screen_label": False, "page_extent": False},
    "label": {"screen_label": True, "page_extent": False},
    "label+extent": {"screen_label": True, "page_extent": True},
}


def _heads_default(cli: Any, lean: Any) -> dict[str, Any]:
    """HEAD's live default as ``cli`` gives it now (never a literal, so the rows survive the next default change)."""
    ns = argparse.Namespace()
    g = cli.grounding_of(lean, ns, "live")
    f = cli.feedback_of(lean, ns, "live")
    r = cli.recovery_of(lean, ns, "live")
    o = cli.page_overview_of(lean, ns, "live")
    a = cli.auto_overview_of(lean, ns, "live")
    return {
        "visual_tokens": g.visual_tokens,
        "reground": g.reground,
        "live_feedback": FEEDBACK_DICT[f.live_feedback],
        "echo": f.action_echo,
        "recover_thought_calls": r.thought_call_recovery,
        "page_overview": o.page_overview,
        "auto_overview": a.auto_overview,
    }


def test_live_default_follows_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 (the T-0135 B pattern): ``live_default`` is read from ``cli`` at call time and follows a monkeypatched
    default in each of its resolvers (560 with ReGround on; ``label+extent`` with pixel echo; recovery off), with both
    overviews off (``cli.page_overview_of``'s and ``cli.auto_overview_of``'s defaults)."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    fn = api("live_default")
    want = _heads_default(cli, lean)
    assert want["page_overview"] is False and want["auto_overview"] is False, (
        "premise: both overviews are off by default at HEAD (T-0140 A5, T-0150 A6)"
    )
    assert dict(fn(lean)) == want, (fn(lean), want)
    assert oracle.live_default_findings(fn(lean)) == []
    real_g = cli.grounding_of
    monkeypatch.setattr(
        cli,
        "grounding_of",
        lambda p, a, m: dataclasses.replace(real_g(p, a, m), visual_tokens=560, reground=True),
    )
    monkeypatch.setattr(
        cli, "feedback_of", lambda p, a, m: cli.Feedback("label+extent", "flag", "pixels", "flag")
    )
    monkeypatch.setattr(cli, "recovery_of", lambda p, a, m: cli.Recovery(False, "flag"))
    moved = dict(fn(lean))
    assert moved == {
        "visual_tokens": 560,
        "reground": True,
        "live_feedback": FEEDBACK_DICT["label+extent"],
        "echo": "pixels",
        "recover_thought_calls": False,
        "page_overview": False,
        "auto_overview": False,
    }, moved
    monkeypatch.setattr(cli, "auto_overview_of", lambda p, a, m: cli.AutoOverview(True, "default"))
    assert dict(fn(lean))["auto_overview"] is True, "live_default must read cli.auto_overview_of"


@pytest.mark.parametrize(
    "over, needle",
    [
        ({}, None),
        ({"auto_overview": True}, "auto_overview"),
        ({"page_overview": True}, "page_overview"),
        ({"auto_overview": None}, "auto_overview"),
    ],
)
def test_live_default_findings_oracle(over: dict[str, Any], needle: str | None) -> None:
    """B1 (oracle): a recorded live default with either overview on (or not ``False``) is not one this A/B runs."""
    found = oracle.live_default_findings(
        {**D_HEAD, "page_overview": False, "auto_overview": False, **over}
    )
    if needle is None:
        assert found == [], found
    else:
        assert found and any(needle in f for f in found), found
    missing = {**D_HEAD, "page_overview": False}
    assert oracle.live_default_findings(missing) == ["live_default lacks auto_overview"]


@pytest.mark.parametrize("which", IMPLS)
def test_order_200_joint_seed0(which: str) -> None:
    """B1: 200 turns, each arm 100 times, each (arm, task) 20 times with repeats 0-19; deterministic;
    seed-dependent; the arms interleave (a per-arm shuffle would run them in blocks)."""
    order = impl(which, "order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)] and turns != [tuple(t) for t in order(1)]
    assert len(turns) == 200
    assert Counter(t[0] for t in turns) == {"A": 100, "B": 100}
    per = Counter((t[0], t[1]) for t in turns)
    assert set(per) == {(a, k) for a in "AB" for k in TASK_IDS} and set(per.values()) == {20}
    assert {t[2] for t in turns} == set(range(20))
    switches = sum(1 for x, y in itertools.pairwise(turns) if x[0] != y[0])
    assert switches > 60, f"the arms run in blocks ({switches} switches)"


def test_harness_order_equals_oracle() -> None:
    """B1/B7: the harness's seed-0 order is the oracle's (the eval checks the run against it)."""
    assert [tuple(t) for t in api("order")(0)] == oracle.order(0)


# =====================================================================================================
# B1: refusals before any socket
# =====================================================================================================


def _stand_in_kits() -> dict[bool, Any]:
    """Keyed by the ``page_overview`` keyword ``load_tool_kit`` got: v1 without it (what both arms must use), the
    overview set with it (what a T-0140-shaped arm B would get)."""
    return {
        False: types.SimpleNamespace(repair_off=None, tools_version=V1, mode="live"),
        True: types.SimpleNamespace(repair_off=None, tools_version=OV_SHA, mode="live"),
    }


def _rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Any:
    """Every socket refused and recorded; a clean tree through the harness's ``_git``; stand-in kits (v1 unless
    ``page_overview=True``, repair on); Chromium reported present."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0150 refusal rig: no {what}")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else t31.GIT + "\n"

    assert callable(getattr(h, "_git", None)), (
        "the harness has no module-level _git (T-0111g's seam)"
    )
    monkeypatch.setattr(h, "_git", git)
    kits = _stand_in_kits()

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        return (kits[bool(page_overview)], None)

    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kits=kits,
        kit=kits[False],
        sockets=sockets,
        status=status,
        out=tmp_path / "out",
        capsys=capsys,
        mp=monkeypatch,
    )


def _session_without(session_mod: Any, *missing: str) -> Any:
    """A ``Session`` whose signature (of the class and of ``__init__``) is the real one without ``missing``."""

    class Old(session_mod.Session):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    sig = inspect.signature(session_mod.Session.__init__)
    kept = [p for p in sig.parameters.values() if p.name not in missing]
    Old.__init__.__signature__ = sig.replace(parameters=kept)  # type: ignore[attr-defined]
    Old.__signature__ = sig.replace(parameters=kept[1:])  # type: ignore[attr-defined]
    return Old


REFUSALS = (
    "dirty-tree",
    "url-not-loopback",
    "out-dir-exists",
    "kit-repair-off",
    "guidance-off-its-sha",
    "kit-not-v1",
    "auto-overview-already-the-default",
    "page-overview-on-by-default",
    "cli-without-auto-overview-of",
    "cli-without-page-overview-of",
    "session-without-auto-overview-keyword",
    "session-without-guard-keywords",
)
#: what the refusal must name (B1), where the case has one name to give
REFUSAL_NAMES = {
    "auto-overview-already-the-default": "auto",
    "page-overview-on-by-default": "page",
    "cli-without-auto-overview-of": "auto_overview_of",
    "cli-without-page-overview-of": "page_overview_of",
    "session-without-auto-overview-keyword": "auto_overview",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, case: str
) -> None:
    """B1: T-0140's refusals that still apply stand (dirty tree, non-loopback URL, an existing out-dir, the tool
    repair off, the guidance off its sha, a ``Session`` without the guard's keywords, a ``cli`` without
    ``page_overview_of``, a default with the page overview on); the tools check requires v1 for both arms (a kit off
    v1 is refused); a ``cli`` default with the auto overview already on is refused; so are a ``Session`` without the
    ``auto_overview`` keyword (every other keyword present) and a ``cli`` without ``auto_overview_of``. Each exits 2
    before any socket, leaving the out-dir as it was, and names what it refused."""
    from local_astra.conductor import prompt_builder
    from local_astra.conductor import session as session_mod

    rig = _rig(tmp_path, monkeypatch, capsys)
    h = rig.h
    argv = ["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)]
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
    elif case == "auto-overview-already-the-default":
        rig.mp.setattr(
            rig.cli, "auto_overview_of", lambda p, a, m: rig.cli.AutoOverview(True, "default")
        )
    elif case == "page-overview-on-by-default":
        rig.mp.setattr(
            rig.cli, "page_overview_of", lambda p, a, m: rig.cli.PageOverview(True, "default")
        )
    elif case == "cli-without-auto-overview-of":
        rig.mp.delattr(rig.cli, "auto_overview_of")
    elif case == "cli-without-page-overview-of":
        rig.mp.delattr(rig.cli, "page_overview_of")
    elif case == "session-without-auto-overview-keyword":
        rig.mp.setattr(session_mod, "Session", _session_without(session_mod, "auto_overview"))
    else:
        rig.mp.setattr(
            session_mod,
            "Session",
            _session_without(session_mod, "degeneracy_guard", "recover_thought_calls"),
        )
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], rig.sockets
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in REFUSAL_NAMES:
        assert REFUSAL_NAMES[case] in err and "overview" in err, (
            f"{case}: the refusal does not name {REFUSAL_NAMES[case]!r}: {err!r}"
        )


def test_rig_at_passing_values_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The other side of every refusal row (a guard too wide): at HEAD's live default, a clean tree, the real
    ``Session`` (its ``auto_overview`` keyword present) and a v1 kit, ``main`` does not refuse; it looks for the
    engine (a socket attempt) and, finding none, exits 1."""
    rig = _rig(tmp_path, monkeypatch, capsys)
    h = rig.h
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)])
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


def test_probes_p1_to_p3_are_t0131s() -> None:
    """B2: P1-P3 are T-0131's own objects; P6, P7 and P8 are the harness's seams."""
    for name in ("probe_p1", "run_probes"):
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == T0131_HARNESS.resolve(), (name, src)
    assert callable(api("probe_p6")) and callable(api("probe_p7")) and callable(api("probe_p8"))


# =====================================================================================================
# B2: P6, P7 and P8 from hand records
# =====================================================================================================

#: deals.html at 1280x800 (O-A1-11: a 838x798 overview), 1.52 screens; a fitting page; T-0134's hostile fixture
#: (2800 CSS px = 3.5 screens)
P6_OK: dict[str, Any] = {
    "viewport": [1280, 800],
    "deals": {
        "outcome": "shown",
        "action": ov_action(1.52, False),
        "screens": 1.52,
        "clipped": False,
        "width": 838,
        "height": 798,
        "scroll_y_before": 0.0,
        "scroll_y_after": 0.0,
    },
    "fits": {"outcome": "fits", "action": FITS},
    "hostile": {
        "outcome": "shown",
        "action": ov_action(3.5, False),
        "screens": 3.5,
        "clipped": False,
    },
}


def _p6(**over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(P6_OK)
    for part, fields in over.items():
        out[part] = {**out[part], **fields}
    return out


#: (record, passes), by hand from B2 ("P6 as T-0140 P6")
P6_ROWS: dict[str, tuple[dict[str, Any], bool]] = {
    "ok": (P6_OK, True),
    "deals-exactly-1.3": (_p6(deals={"screens": 1.3, "action": ov_action(1.3, False)}), True),
    "deals-1.29": (_p6(deals={"screens": 1.29, "action": ov_action(1.29, False)}), False),
    "deals-fits": (_p6(deals={"outcome": "fits", "action": FITS}), False),
    "deals-image-too-wide": (_p6(deals={"width": 1281}), False),
    "deals-image-too-tall": (_p6(deals={"height": 801}), False),
    "deals-scrolled": (_p6(deals={"scroll_y_after": 2.0}), False),
    "fits-showed-an-overview": (
        _p6(fits={"outcome": "shown", "action": ov_action(1.0, False)}),
        False,
    ),
    "hostile-off-by-2px": (
        _p6(hostile={"screens": 2802 / 800, "action": ov_action(2802 / 800, False)}),
        False,
    ),
    "hostile-within-1px": (
        _p6(hostile={"screens": 2800.5 / 800, "action": ov_action(2800.5 / 800, False)}),
        True,
    ),
    "wrong-viewport": ({**P6_OK, "viewport": [1280, 720]}, False),
    "no-hostile": ({k: v for k, v in P6_OK.items() if k != "hostile"}, False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(P6_ROWS))
def test_p6_rule(which: str, row: str) -> None:
    """B2 P6 (T-0140 P6's rule, kept): deals.html gives an overview of at least 1.3 screens within 1280x800 with A3's
    action string and ``scroll_y`` unchanged; a fitting page gives ``fits``; T-0134's hostile fixture its true height
    (1 px). Each clause failing alone fails P6; a missing part fails, never raises."""
    record, want = P6_ROWS[row]
    assert impl(which, "p6_ok")(copy.deepcopy(record)) is want, row


#: an S4-like step-1 request of a fresh session: the overview and the current frame (T-0140 measured 1,116 overview
#: tokens; these counts are hand-chosen inputs)
P7_OK: dict[str, Any] = {
    "tokens": 9_400,
    "limit": 16_128,
    "http_status": 200,
    "healthy_after": True,
    "images": 2,
    "overview_images": 1,
    "overview_tokens": 1_116,
}
P7_ROWS: dict[str, tuple[dict[str, Any], bool]] = {
    "ok": (P7_OK, True),
    "exactly-16128": ({**P7_OK, "tokens": 16_128}, True),
    "16129": ({**P7_OK, "tokens": 16_129}, False),
    "no-count": ({**P7_OK, "tokens": None}, False),
    "http-500": ({**P7_OK, "http_status": 500}, False),
    "engine-down-after": ({**P7_OK, "healthy_after": False}, False),
    "no-overview-in-the-request": ({**P7_OK, "overview_images": 0}, False),
    "two-overviews": ({**P7_OK, "overview_images": 2}, False),
    # T-0140's P7 shape (a step-2 request with a history image: 3 images) is not the step-1 shape B2 asks for
    "t0140s-three-image-step-2-shape": ({**P7_OK, "images": 3}, False),
    "one-image": ({**P7_OK, "images": 1}, False),
    "another-limit": ({**P7_OK, "limit": 16_384, "tokens": 16_200}, False),
    "overview-tokens-report-only": ({**P7_OK, "overview_tokens": 4_000}, True),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(P7_ROWS))
def test_p7_rule(which: str, row: str) -> None:
    """B2 P7 (T-0132 P4's rule over the step-1 shape): a ``/tokenize`` count of an S4-like step-1 request -- 2
    images, one of them the overview -- within 16128 tokens, the generate answering 200 and the engine healthy after;
    ``overview_tokens`` is report-only."""
    record, want = P7_ROWS[row]
    assert impl(which, "p7_ok")(dict(record)) is want, row


#: B2 P8: the census at 1280x800, the start pages of T-0111g's ``TASKS``; by hand from docs/tasks/T-0150.md's
#: measured heights (deals 1220 CSS px = 1.525 screens; every other start page 800)
P8_OK: dict[str, Any] = {
    "viewport": [1280, 800],
    "tasks": {
        "S1": {"start_page": "/", "outcome": "fits"},
        "S2": {"start_page": "/demo/shop.html", "outcome": "fits"},
        "S3": {"start_page": "/demo/contact.html", "outcome": "fits"},
        "S4": {"start_page": "/demo/deals.html", "outcome": "shown", "screens": 1.525},
        "S5": {"start_page": "/demo/deals.html", "outcome": "shown", "screens": 1.525},
    },
}


def _p8(**over: Any) -> dict[str, Any]:
    out = copy.deepcopy(P8_OK)
    for task, outcome in over.items():
        if outcome is None:
            del out["tasks"][task]
        else:
            out["tasks"].setdefault(task, {})["outcome"] = outcome
    return out


P8_ROWS: dict[str, tuple[dict[str, Any], bool]] = {
    "ok": (P8_OK, True),
    # the two named in Tests to write
    "s2-shown": (_p8(S2="shown"), False),
    "s5-fits": (_p8(S5="fits"), False),
    "s1-shown": (_p8(S1="shown"), False),
    "s3-shown": (_p8(S3="shown"), False),
    "s4-fits": (_p8(S4="fits"), False),
    "s4-failed": (_p8(S4="failed"), False),
    "s5-refused": (_p8(S5="refused"), False),
    "s3-missing": (_p8(S3=None), False),
    "an-extra-task": (_p8(S6="fits"), False),
    "wrong-viewport": ({**P8_OK, "viewport": [1280, 720]}, False),
    "no-tasks": ({"viewport": [1280, 800]}, False),
    "an-outcome-missing": (
        {**P8_OK, "tasks": {**P8_OK["tasks"], "S2": {"start_page": "/demo/shop.html"}}},
        False,
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(P8_ROWS))
def test_p8_rule(which: str, row: str) -> None:
    """B2 P8 (R-d; reviewer mutation 15 at the rule): the start-page census gives exactly S4 and S5 ``shown`` and S1,
    S2 and S3 ``fits``, every task present and none other, at 1280x800; any mismatch fails P8 (and so voids the
    run); a missing part fails, never raises."""
    record, want = P8_ROWS[row]
    assert impl(which, "p8_ok")(copy.deepcopy(record)) is want, row


# =====================================================================================================
# Hand-built records: five turns in arm A, six in arm B
# =====================================================================================================

REQUEST = "Find the cheapest item."
SCREEN_NOW = "[This screenshot is the browser screen now.]"
SCREEN_AFTER = "[This screenshot is the browser screen now, after your last action.]"
TOWEL_SAID = "The cheapest is the cotton tea towel at $4.75."
MUG_SAID = "The cheapest is the enamel mug at $11.50."
NARRATED = "The whole page shows the enamel mug is the cheapest."
DEALS_EMPTY = {"page": "deals", "cart": []}


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _i(tag: str) -> dict[str, str]:
    return {"image": _h(tag)}


def _t(text: str) -> dict[str, str]:
    return {"text": text}


def _sys() -> dict[str, Any]:
    return {"role": "system", "parts": [_t("You are Astra.")]}


def _user(*parts: dict[str, str]) -> dict[str, Any]:
    return {"role": "user", "parts": list(parts)}


def _asst(text: str) -> dict[str, Any]:
    return {"role": "assistant", "parts": [_t(text)]}


def _req(
    label: str, phase: str, step: int | None, tools: list[str] | None, *messages: dict[str, Any]
) -> dict[str, Any]:
    return {
        "label": label,
        "phase": phase,
        "step": step,
        "tools": None if tools is None else list(tools),
        "messages": [copy.deepcopy(m) for m in messages],
    }


def _counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in A5_KEYS}


def _tr(
    arm: str, task: str, repeat: int, state: Any, voiced: list[str], **kw: Any
) -> dict[str, Any]:
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
        "auto_overview": arm == "B",
        "tools_version": V1,
        "auto_overview_logged": None if arm == "A" else _counts(),
        "page_overview_logged": None,
        "degeneracy_guard_logged": {"on": True, "tripped": []},
        # the turn line as the session logs it at D_HEAD (T-0134 A5 adds the browser's echo)
        "live_feedback_logged": {**LABEL_FB, "echo": "model"},
        "recover_thought_calls_logged": True,
        "visual_tokens_logged": 1120,
        "budgets_sent": [1120],
        "guidance_sent": True,
        "scanned_requests": [f"prefill:{p}-0", f"generate:{p}-1"],
        "tool_loop_end": "final",
        "session_id": f"s-{p}",
    }
    return t31._trial(task, repeat, state, **(base | kw))


def _st(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    done: float | None,
    name: str | None = None,
    *,
    voiced: tuple[str, ...] = (),
    shown: bool = False,
    **kw: Any,
) -> dict[str, Any]:
    """A step line; its first commit 100 ms before ``done``."""
    p = f"{arm}{task}{repeat}"
    return t31._step(
        task,
        repeat,
        step,
        status,
        None if done is None else done - 100.0,
        done,
        arm=arm,
        visual_tokens=1120,
        reground=False,
        reground_visual_tokens=None,
        call_name=name,
        executed_name=name if status == "executed" else None,
        voiced_sources=list(voiced),
        attempt_id=f"{p}-{step}",
        call_id=f"{p}-call-{step}" if name else None,
        overview_shown=shown,
        **kw,
    )


def _ov(tag: str, screens: float = 1.5, clipped: bool = False) -> dict[str, Any]:
    """One auto capture's overview record (step 1, B1); deals.html's is about 1.5 screens, 839x800 scaled."""
    return {
        "step": 1,
        "sha": _h(tag),
        "screens": screens,
        "clipped": clipped,
        "width": 839,
        "height": 800,
    }


def _ov_parts(ov: dict[str, Any]) -> list[dict[str, str]]:
    return [{"image": ov["sha"]}, _t(ov_label(float(ov["screens"]), bool(ov["clipped"])))]


def _gen(
    p: str,
    ov: dict[str, Any] | None = None,
    history: tuple[dict[str, Any], ...] = (),
    label: str | None = None,
) -> dict[str, Any]:
    """The step-1 generate: the final user message ``[request, (overview, its label,) frame, screen label]``."""
    parts = [_t(REQUEST), *(_ov_parts(ov) if ov else []), _i(f"{p}-f0"), _t(SCREEN_NOW)]
    return _req(
        label or f"generate:{p}-1", "generate", 1, V1_TOOLS, _sys(), *history, _user(*parts)
    )


def _step_req(p: str, k: int) -> dict[str, Any]:
    return _req(
        f"generate:{p}-{k}",
        "tool_step",
        k,
        V1_TOOLS,
        _sys(),
        _user(_t(REQUEST), _i(f"{p}-f{k - 1}"), _t(SCREEN_AFTER)),
        _asst(f"Working. step {k - 1} done."),
    )


def _plain_requests(
    arm: str,
    task: str,
    repeat: int,
    n: int,
    ov: dict[str, Any] | None = None,
    history: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    """A prefill, the step-1 generate (with ``ov`` before the current frame when given) and ``n - 1`` tool steps,
    all on tools v1."""
    p = f"{arm}{task}{repeat}"
    out = [
        _req(f"prefill:{p}-0", "prefill", None, V1_TOOLS, _sys(), *history),
        _gen(p, ov, history),
    ]
    out += [_step_req(p, k) for k in range(2, n + 1)]
    return out


def _line(
    arm: str, task: str, repeat: int, requests: list[Any], overviews: list[Any]
) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": task,
        "repeat": repeat,
        "requests": requests,
        "overviews": overviews,
    }


#: the arm-B overviews of the hand records: B1 S4#0, B2 S4#1 (a clipped one, so the label's clip text is exercised),
#: B3 S5#0
OV1 = _ov("BS40-ov")
OV2 = _ov("BS41-ov", 4.4, True)
OV3 = _ov("BS50-ov")
HISTORY = (
    _user(_t("What is on sale?"), _i("BS41-h1")),
    _asst("Six deals."),
    _user(_t("And the mugs?"), _i("BS41-h2")),
    _asst("Two mugs."),
)


def _repair_of_gen(gen: dict[str, Any], label: str, step: int) -> dict[str, Any]:
    """A ``tool_repair`` body: the step's pass-1 prompt continued with the draft (the overview, if any, included, as
    ``session._run_repair`` re-sends ``pass1_prompt``; A2 "Repair")."""
    r = copy.deepcopy(gen)
    r.update(label=label, phase="tool_repair", step=step)
    r["messages"].append(_asst("Let me click. <|tool_call>call:click_at{x:"))
    return r


def _hand_records(arm: str) -> tuple[list[Any], list[Any], list[Any]]:
    """Every count is written out in ``HAND``'s comment.

    Arm A (auto overview off): A1 S4#0 a scroll, then the towel and $4.75 (success); A2 S4#1 names the mug (false
    claim); A3 S2#0 two blue mugs, "I have added the blue mug." (over-click; S1-S3 false claim); A4 S1#0 success; A5
    S5#0 two enamel mugs, "I added the cheapest item." (S5 over-click and S5 false claim).

    Arm B (auto overview on): B1 S4#0 overview shown at step 1 (1.5 screens), a scroll, the towel (success); B2 S4#1
    overview shown (4.4 screens, clipped) with two history images (4 images), an over-long retry re-sending it with
    one history turn dropped, narrates the whole page and names the mug (false claim); B3 S5#0 overview shown, its
    step-1 call repaired (the repair re-sends the overview), the towel in the cart (success); B4 S1#0 ``fits``, a
    click, success; B5 S2#0 ``fits`` and a ``degenerate`` end with the right cart (a forced failure; its aborted step
    2 answered HTTP 0, not an engine error); B6 S4#2 the capture ``failed`` (logged; no overview), names the mug
    (false claim)."""
    if arm == "A":
        trials = [
            _tr("A", "S4", 0, DEALS_EMPTY, [TOWEL_SAID]),
            _tr("A", "S4", 1, DEALS_EMPTY, [MUG_SAID]),
            _tr(
                "A",
                "S2",
                0,
                {"page": "shop", "cart": ["blue mug", "blue mug"]},
                ["I have added the blue mug."],
            ),
            _tr("A", "S1", 0, {"page": "shop", "cart": ["blue mug"]}, ["Added."]),
            _tr(
                "A",
                "S5",
                0,
                {"page": "deals", "cart": ["enamel mug", "enamel mug"]},
                ["I added the cheapest item."],
            ),
        ]
        steps = [
            _st("A", "S4", 0, 1, "executed", 300.0, "scroll"),
            _st("A", "S4", 0, 2, None, 350.0, voiced=(TOWEL_SAID,)),
            _st("A", "S4", 1, 1, None, 360.0, voiced=(MUG_SAID,)),
            _st("A", "S2", 0, 1, "executed", 310.0, "click_at"),
            _st("A", "S2", 0, 2, "executed", 320.0, "click_at"),
            _st("A", "S2", 0, 3, None, 330.0, voiced=("I have added the blue mug.",)),
            _st("A", "S1", 0, 1, "executed", 340.0, "click_at"),
            _st("A", "S1", 0, 2, None, 370.0, voiced=("Added.",)),
            _st("A", "S5", 0, 1, "executed", 380.0, "click_at"),
            _st("A", "S5", 0, 2, "executed", 390.0, "click_at"),
            _st("A", "S5", 0, 3, None, 395.0, voiced=("I added the cheapest item.",)),
        ]
        bodies = [
            _line("A", "S4", 0, _plain_requests("A", "S4", 0, 2), []),
            _line("A", "S4", 1, _plain_requests("A", "S4", 1, 1), []),
            _line("A", "S2", 0, _plain_requests("A", "S2", 0, 3), []),
            _line("A", "S1", 0, _plain_requests("A", "S1", 0, 2), []),
            _line("A", "S5", 0, _plain_requests("A", "S5", 0, 3), []),
        ]
        return trials, steps, bodies
    trials = [
        _tr("B", "S4", 0, DEALS_EMPTY, [TOWEL_SAID], auto_overview_logged=_counts(shown=1)),
        _tr("B", "S4", 1, DEALS_EMPTY, [NARRATED], auto_overview_logged=_counts(shown=1)),
        _tr(
            "B",
            "S5",
            0,
            {"page": "deals", "cart": ["cotton tea towel"]},
            ["Added the cotton tea towel."],
            auto_overview_logged=_counts(shown=1),
        ),
        _tr(
            "B",
            "S1",
            0,
            {"page": "shop", "cart": ["blue mug"]},
            ["Added."],
            auto_overview_logged=_counts(fits=1),
        ),
        _tr(
            "B",
            "S2",
            0,
            {"page": "shop", "cart": ["blue mug"]},
            ["Adding the blue mug."],
            tool_loop_end="degenerate",
            auto_overview_logged=_counts(fits=1),
        ),
        _tr("B", "S4", 2, DEALS_EMPTY, [MUG_SAID], auto_overview_logged=_counts(failed=1)),
    ]
    steps = [
        _st("B", "S4", 0, 1, "executed", 900.0, "scroll", shown=True),
        _st("B", "S4", 0, 2, None, 305.0, voiced=(TOWEL_SAID,)),
        _st("B", "S4", 1, 1, None, 1100.0, voiced=(NARRATED,), shown=True),
        _st("B", "S5", 0, 1, "executed", 1000.0, "click_at", shown=True),
        _st("B", "S5", 0, 2, None, 410.0, voiced=("Added the cotton tea towel.",)),
        _st("B", "S1", 0, 1, "executed", 330.0, "click_at"),
        _st("B", "S1", 0, 2, None, 420.0, voiced=("Added.",)),
        _st("B", "S2", 0, 1, "executed", 340.0, "click_at", voiced=("Adding the blue mug.",)),
        _st("B", "S2", 0, 2, None, None, http_status=0, error="cancelled"),
        _st("B", "S4", 2, 1, None, 360.0, voiced=(MUG_SAID,)),
    ]
    b2 = _plain_requests("B", "S4", 1, 1, OV2, HISTORY)
    # the over-long retry (400/413): one history turn dropped, the same overview re-placed (A2 "Retry")
    b2.append(_gen("BS41", OV2, HISTORY[2:], label="generate:BS41-1r"))
    b3 = _plain_requests("B", "S5", 0, 2, OV3)
    b3.insert(2, _repair_of_gen(b3[1], "repair:BS50-1r", 1))
    bodies = [
        _line("B", "S4", 0, _plain_requests("B", "S4", 0, 2, OV1), [dict(OV1)]),
        _line("B", "S4", 1, b2, [dict(OV2)]),
        _line("B", "S5", 0, b3, [dict(OV3)]),
        _line("B", "S1", 0, _plain_requests("B", "S1", 0, 2), []),
        _line("B", "S2", 0, _plain_requests("B", "S2", 0, 2), []),
        _line("B", "S4", 2, _plain_requests("B", "S4", 2, 1), []),
    ]
    return trials, steps, bodies


def _clean() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    ta, sa, ba = _hand_records("A")
    tb, sb, bb = _hand_records("B")
    return ta + tb, sa + sb, [], ba + bb


def _find(rows: list[dict[str, Any]], arm: str, task: str, repeat: int) -> dict[str, Any]:
    (r,) = [x for x in rows if (x["arm"], x["task"], x["repeat"]) == (arm, task, repeat)]
    return r


def _request(
    bodies: list[dict[str, Any]], arm: str, task: str, repeat: int, label: str
) -> dict[str, Any]:
    (r,) = [r for r in _find(bodies, arm, task, repeat)["requests"] if r["label"] == label]
    return r


def _final_user(request: dict[str, Any]) -> dict[str, Any]:
    return [m for m in request["messages"] if m["role"] == "user"][-1]


# =====================================================================================================
# B4: configuration findings, one hand trial per finding
# =====================================================================================================


def _placed_in(p: str, request: dict[str, Any], ov: dict[str, Any], frame: str) -> None:
    """Rewrite ``request``'s final user message as A2 places an overview: ``[request, ov, its label, frame, ...]``."""
    user = _final_user(request)
    user["parts"] = [_t(REQUEST), *_ov_parts(ov), _i(frame), _t(SCREEN_AFTER)]


def _shown_on(trials: Any, bodies: Any, arm: str, task: str, repeat: int) -> None:
    """An overview shown at step 1 of ``(arm, task, repeat)``, placed and logged exactly as on S4 (record, image and
    label, ``shown`` 1)."""
    p = f"{arm}{task}{repeat}"
    ov = _ov(f"{p}-ov")
    _find(trials, arm, task, repeat)["auto_overview_logged"] = _counts(shown=1)
    line = _find(bodies, arm, task, repeat)
    line["overviews"] = [dict(ov)]
    line["requests"][1] = _gen(p, ov)


def _unlogged(trials: Any, bodies: Any, task: str, repeat: int) -> None:
    """Arm B ``task#repeat`` sent step 1 with its current frame and no overview anywhere, and logged all four counts
    0."""
    _find(trials, "B", task, repeat)["auto_overview_logged"] = _counts()
    line = _find(bodies, "B", task, repeat)
    line["overviews"] = []
    line["requests"] = _plain_requests("B", task, repeat, 2)


def _fault(name: str) -> tuple[Any, ...]:
    """One B4 fault applied to the clean records; returns (trials, steps, regrounds, bodies, arms, the name a
    finding must carry)."""
    trials, steps, regrounds, bodies = _clean()
    arms: list[dict[str, Any]] = [copy.deepcopy(r) for r in HAND_ARMS]
    if name == "non-1120-body":
        _find(trials, "A", "S4", 0)["budgets_sent"] = [1120, 560]
        return trials, steps, regrounds, bodies, arms, "A S4#0"
    if name == "reground-line":
        regrounds = [{"arm": "B", "task": "S4", "repeat": 0, "step": 2, "outcome": "ok"}]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "arm-a-body-declares-page-overview":
        _request(bodies, "A", "S4", 1, "generate:AS41-1")["tools"] = list(OV_TOOLS)
        return trials, steps, regrounds, bodies, arms, "A S4#1"
    if name == "arm-b-body-declares-page-overview":
        _request(bodies, "B", "S5", 0, "prefill:BS50-0")["tools"] = list(OV_TOOLS)
        return trials, steps, regrounds, bodies, arms, "B S5#0"
    if name == "arm-a-body-carries-an-overview":
        # the arm-A turn's step 1 placed exactly as arm B's (record and image and label): only the arm rule can see it
        ov = _ov("AS40-ov")
        line = _find(bodies, "A", "S4", 0)
        line["overviews"] = [dict(ov)]
        line["requests"][1] = _gen("AS40", ov)
        return trials, steps, regrounds, bodies, arms, "A S4#0"
    if name == "arm-a-overview-label-in-a-body":
        user = _final_user(_request(bodies, "A", "S4", 1, "generate:AS41-1"))
        user["parts"].insert(1, _t(ov_label(1.5, False)))
        return trials, steps, regrounds, bodies, arms, "A S4#1"
    if name == "overview-after-current-frame":
        # group A's mutation 1: [frame, screen label, overview, its label]
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        user["parts"] = [_t(REQUEST), _i("BS40-f0"), _t(SCREEN_NOW), *_ov_parts(OV1)]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-before-an-older-image":
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        user["parts"] = [
            _t(REQUEST),
            *_ov_parts(OV1),
            _i("BS40-older"),
            _i("BS40-f0"),
            _t(SCREEN_NOW),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-kept-in-step-2":
        # group A's mutations 2/3: the step-2 request carries it, placed as A2 places it
        _placed_in("BS40", _request(bodies, "B", "S4", 0, "generate:BS40-2"), OV1, "BS40-f1")
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-the-repair-of-step-2":
        # (sha, step) placement: a repair of the step-2 call carrying the step-1 overview is out of place
        two = copy.deepcopy(_request(bodies, "B", "S4", 0, "generate:BS40-2"))
        _placed_in("BS40", two, OV1, "BS40-f1")
        two.update(label="repair:BS40-2r", phase="tool_repair", step=2)
        _find(bodies, "B", "S4", 0)["requests"].append(two)
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-prefill":
        r = _request(bodies, "B", "S4", 0, "prefill:BS40-0")
        r["messages"].append(_user(_t("x"), {"image": OV1["sha"]}))
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-a-pass2-body":
        r = copy.deepcopy(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        r.update(label="generate:BS40-1-rg", phase="reground", step=1)
        _find(bodies, "B", "S4", 0)["requests"].append(r)
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-a-caption-body":
        r = copy.deepcopy(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        r.update(label="generate:caption-1", phase="caption", step=None, tools=None)
        _find(bodies, "B", "S4", 0)["requests"].append(r)
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-a-history-message":
        r = _request(bodies, "B", "S4", 0, "generate:BS40-2")
        r["messages"][1:1] = [_user(_t("Earlier."), {"image": OV1["sha"]}), _asst("Earlier.")]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-placed-in-a-history-message-of-step-1":
        # the right request, the right [overview, label, frame] order, but in an earlier user message
        r = _request(bodies, "B", "S4", 0, "generate:BS40-1")
        r["messages"] = [
            _sys(),
            _user(_t("Earlier."), *_ov_parts(OV1), _i("BS40-h")),
            _asst("Earlier."),
            _user(_t(REQUEST), _i("BS40-f0"), _t(SCREEN_NOW)),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "two-overview-images":
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        user["parts"].insert(0, {"image": OV1["sha"]})
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "five-images":
        r = _request(bodies, "B", "S4", 1, "generate:BS41-1")
        r["messages"][1:1] = [_user(_t("Before."), _i("BS41-h0")), _asst("Before.")]
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name == "overview-without-its-label":
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-1"))
        user["parts"] = [_t(REQUEST), {"image": OV1["sha"]}, _i("BS40-f0"), _t(SCREEN_NOW)]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-label-of-another-height":
        user = _final_user(_request(bodies, "B", "S4", 1, "generate:BS41-1"))
        user["parts"][2] = _t(ov_label(4.4, False))
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name == "overview-label-in-a-history-user-message":
        r = _request(bodies, "B", "S4", 1, "generate:BS41-1r")
        r["messages"][1]["parts"].append(_t(ov_label(4.4, True)))
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name in ("overview-on-s1", "overview-on-s2"):
        # reviewer mutation 16: placed, logged and recorded exactly as on S4; only the S1-S3 rule can see it
        task = "S1" if name.endswith("s1") else "S2"
        _shown_on(trials, bodies, "B", task, 0)
        return trials, steps, regrounds, bodies, arms, f"B {task}#0"
    if name in ("unlogged-absence-s4", "unlogged-absence-s5"):
        task = "S4" if name.endswith("s4") else "S5"
        _unlogged(trials, bodies, task, 0)
        return trials, steps, regrounds, bodies, arms, f"B {task}#0"
    if name == "overview-sent-but-shown-0":
        # O-B-2: placed exactly as A2 places it, but the turn line counts ``fits`` (no other rule can see it)
        _find(trials, "B", "S4", 0)["auto_overview_logged"] = _counts(fits=1)
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "shown-1-but-no-overview":
        # O-B-2: the turn line counts ``shown`` 1 while step 1 went out without the overview
        _find(trials, "B", "S4", 2)["auto_overview_logged"] = _counts(shown=1)
        return trials, steps, regrounds, bodies, arms, "B S4#2"
    if name == "feedback-off-its-row":
        _find(trials, "A", "S1", 0)["live_feedback"] = None
        return trials, steps, regrounds, bodies, arms, "A S1#0"
    if name == "echo-off-its-row":
        _find(trials, "B", "S5", 0)["echo"] = "pixels"
        return trials, steps, regrounds, bodies, arms, "B S5#0"
    if name == "recovery-off-its-row":
        _find(trials, "A", "S2", 0)["recover_thought_calls"] = False
        return trials, steps, regrounds, bodies, arms, "A S2#0"
    if name == "guard-off-its-row":
        _find(trials, "B", "S2", 0)["degeneracy_guard"] = False
        return trials, steps, regrounds, bodies, arms, "B S2#0"
    if name == "auto-overview-off-its-row":
        _find(trials, "B", "S1", 0)["auto_overview"] = False
        return trials, steps, regrounds, bodies, arms, "B S1#0"
    if name == "arms-differ-in-budget":
        arms[1]["visual_tokens"] = 560
        return trials, steps, regrounds, bodies, arms, "visual_tokens"
    if name == "arms-differ-in-echo":
        arms[1]["echo"] = "pixels"
        return trials, steps, regrounds, bodies, arms, "echo"
    if name == "arm-b-row-without-auto-overview":
        arms[1]["auto_overview"] = False
        return trials, steps, regrounds, bodies, arms, "arm B"
    if name == "arm-a-row-with-auto-overview":
        arms[0]["auto_overview"] = True
        return trials, steps, regrounds, bodies, arms, "arm A"
    if name == "arm-b-row-on-the-overview-set":
        arms[1]["tools_sha256"] = OV_SHA
        return trials, steps, regrounds, bodies, arms, "arm B"
    if name == "arm-a-row-with-page-overview":
        arms[0]["page_overview"] = True
        return trials, steps, regrounds, bodies, arms, "arm A"
    if name == "arm-rows-guard-off":
        for r in arms:
            r["degeneracy_guard"] = False
        return trials, steps, regrounds, bodies, arms, "degeneracy_guard"
    raise KeyError(name)


FAULTS = (
    "non-1120-body",
    "reground-line",
    "arm-a-body-declares-page-overview",
    "arm-b-body-declares-page-overview",
    "arm-a-body-carries-an-overview",
    "arm-a-overview-label-in-a-body",
    "overview-after-current-frame",
    "overview-before-an-older-image",
    "overview-kept-in-step-2",
    "overview-in-the-repair-of-step-2",
    "overview-in-prefill",
    "overview-in-a-pass2-body",
    "overview-in-a-caption-body",
    "overview-in-a-history-message",
    "overview-placed-in-a-history-message-of-step-1",
    "two-overview-images",
    "five-images",
    "overview-without-its-label",
    "overview-label-of-another-height",
    "overview-label-in-a-history-user-message",
    "overview-on-s1",
    "overview-on-s2",
    "unlogged-absence-s4",
    "unlogged-absence-s5",
    "overview-sent-but-shown-0",
    "shown-1-but-no-overview",
    "feedback-off-its-row",
    "echo-off-its-row",
    "recovery-off-its-row",
    "guard-off-its-row",
    "auto-overview-off-its-row",
    "arms-differ-in-budget",
    "arms-differ-in-echo",
    "arm-b-row-without-auto-overview",
    "arm-a-row-with-auto-overview",
    "arm-b-row-on-the-overview-set",
    "arm-a-row-with-page-overview",
    "arm-rows-guard-off",
)


def _cf(which: str) -> Any:
    fn = impl(which, "configuration_findings")
    return lambda trials, steps, regrounds, bodies, arms: fn(trials, steps, regrounds, bodies, arms)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", FAULTS)
def test_configuration_findings(which: str, fault: str) -> None:
    """B4, one hand trial per finding (reviewer mutation 16 among them, and group A's mutations 1-3 as records): the
    clean records have none; each fault gives at least one finding naming the trial or the setting -- a non-1120
    body, a reground line, any body declaring ``page_overview``, an arm-A body carrying an overview (image or label),
    an arm-B overview anywhere but immediately before the current frame of the step-1 generate, its retry or the repair
    of its call (after the frame, before an older image, in step 2, in the repair of step 2, in a prefill, a pass-2, a
    caption or a history message), two overviews in a request, a fifth image, the overview without its label or with
    another height's, an overview on an S1-S3 trial, an unlogged absence on S4/S5, a trial's feedback/echo/recovery/
    guard/auto overview off its row, the arm rows differing or off B1."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], (
        "premise: the clean run has no finding"
    )
    trials, steps, regrounds, bodies, arms, named = _fault(fault)
    found = cf(trials, steps, regrounds, bodies, arms)
    assert found and any(named in f for f in found), (fault, found)


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """B4 companions, the guard-too-wide side: the clean records hold an overview right before the current frame of
    the step-1 generate with its label (B1 S4#0), a clipped one in a four-image request with two history images and
    its over-long retry re-sending it with one history turn dropped (B2 S4#1), the repair of a step-1 call re-sending
    it (B3 S5#0), T-0134's screen label after the current frame, ``fits`` on S1/S2 with no overview, a ``failed``
    capture on S4 with no overview (a logged absence, B6) and a ``degenerate`` end -- none is a finding. A trial that
    sent nothing is not judged (O-11)."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    raised = _find(trials, "A", "S2", 0)
    raised.update(scanned_requests=[], exception="RuntimeError('no engine')", budgets_sent=None)
    _request(bodies, "A", "S2", 0, "generate:AS20-1")["tools"] = list(OV_TOOLS)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "O-11: a trial that sent nothing"


@pytest.mark.parametrize("which", IMPLS)
def test_unlogged_absence_not_too_wide(which: str) -> None:
    """B4's unlogged absence, its edges: all four counts 0 on an S4 turn that never sent step 1 (only a prefill went
    out, as a barge-in before ``LlmStart`` gives, A5) is no finding; neither is a turn with no turn line (timed out,
    T-0135b A7), nor a logged ``failed``/``refused`` absence, nor an arm-A turn without an overview."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    t = _find(trials, "B", "S4", 2)
    t["auto_overview_logged"] = _counts()
    line = _find(bodies, "B", "S4", 2)
    line["requests"] = line["requests"][:1]  # the prefill only
    t["scanned_requests"] = ["prefill:BS42-0"]
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "step 1 never went out"
    trials, steps, regrounds, bodies = _clean()
    _unlogged(trials, bodies, "S4", 0)
    _find(trials, "B", "S4", 0).update(
        timed_out=True,
        session_id=None,
        auto_overview_logged=None,
        live_feedback_logged=None,
        recover_thought_calls_logged=None,
        visual_tokens_logged=None,
        degeneracy_guard_logged=None,
    )
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "no turn line, nothing to judge"
    for logged in (_counts(refused=1), _counts(failed=1)):
        trials, steps, regrounds, bodies = _clean()
        _unlogged(trials, bodies, "S5", 0)
        _find(trials, "B", "S5", 0)["auto_overview_logged"] = logged
        assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], logged


@pytest.mark.parametrize("which", IMPLS)
def test_shown_and_body_agreement_not_too_wide(which: str) -> None:
    """O-B-2's companions, the guard-too-wide side: ``shown`` 1 with the overview only in the over-long retry (the
    first generate without it) is no finding -- "the step-1 generate body (or its retry)"; ``shown`` 0 with an
    overview in the body on a trial that wrote no turn line (timed out, T-0135b A7) is not judged; ``shown`` 0 and
    ``fits`` 1 with no overview anywhere (B4 S1#0) is the clean record."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    first = _request(bodies, "B", "S4", 1, "generate:BS41-1")
    _final_user(first)["parts"] = [_t(REQUEST), _i("BS41-f0"), _t(SCREEN_NOW)]
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "shown 1, the retry carries it"
    trials, steps, regrounds, bodies = _clean()
    _find(trials, "B", "S4", 0).update(
        timed_out=True,
        session_id=None,
        auto_overview_logged=None,
        live_feedback_logged=None,
        recover_thought_calls_logged=None,
        visual_tokens_logged=None,
        degeneracy_guard_logged=None,
    )
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "no turn line, nothing to judge"


@pytest.mark.parametrize("which", IMPLS)
def test_spoken_overview_words_are_not_a_void(which: str) -> None:
    """O-17's lesson for B4 (mutant: the label judged in every message): the model speaks the label's own words and
    they reach the next request as assistant text -- no finding; the same text in a history *user* message (the
    session storing the label) is one."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    spoken = "Whole page, scaled to fit one screen: the towel is at the bottom."
    _find(trials, "B", "S4", 0)["voiced"] = [spoken, TOWEL_SAID]
    r = _request(bodies, "B", "S4", 0, "generate:BS40-2")
    r["messages"][-1]["parts"] = [_t(spoken + " " + SCROLL_CALL)]
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    r["messages"][1:1] = [_user(_t(spoken)), _asst("Earlier.")]
    found = cf(trials, steps, regrounds, bodies, HAND_ARMS)
    assert found and all("B S4#0" in f for f in found), found


def _wire_image(data: bytes) -> dict[str, Any]:
    import base64

    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _wire_body(*, overview: bool, draft: str | None = None) -> dict[str, Any]:
    """A forwarded body on the wire: the final user message ``[request, (overview, its label,) frame, screen label]``
    on tools v1, then (a repair) the draft."""
    content: list[dict[str, Any]] = [{"type": "text", "text": REQUEST}]
    if overview:
        content += [_wire_image(b"OVERVIEW-JPEG"), {"type": "text", "text": ov_label(1.5, False)}]
    content += [_wire_image(b"FRAME-JPEG"), {"type": "text", "text": SCREEN_NOW}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are Astra."},
        {"role": "user", "content": content},
    ]
    if draft is not None:
        messages.append({"role": "assistant", "content": draft})
    return {
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": n}} for n in V1_TOOLS],
    }


def _request_record(
    phase: str, request_id: str, attempt_id: str, step: int | None = None
) -> dict[str, Any]:
    data: dict[str, Any] = {"phase": phase, "request_id": request_id, "attempt_id": attempt_id}
    if step is not None:
        data["step"] = step
    return {"kind": "llm.request", "data": data}


#: (llm.request records in log order, the repair body's (phase, step), whether the repair carrying the step-1
#: overview is in place), by hand from the turn machine's ids (pass 1 ``{turn}-gen`` / ``{turn}-gen-s{k}``, the
#: turn's one repair ``{turn}-repair``) and A2 "Repair"
REPAIR_STEPS: dict[str, tuple[list[dict[str, Any]], tuple[str, int | None], bool]] = {
    "repair-of-step-1": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_repair", "t1-repair", "e-rep-3"),
        ],
        ("tool_repair", 1),
        True,
    ),
    "repair-of-step-2": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_step", "t1-gen-s2", "e-gen-s2-2", 2),
            _request_record("tool_repair", "t1-repair", "e-rep-3"),
        ],
        ("tool_repair", 2),
        False,
    ),
    "repair-of-another-turns-step": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_repair", "t2-repair", "e-rep-3"),
        ],
        ("tool_repair", None),
        False,
    ),
    "oddly-named-repair": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_repair", "t1-fix", "e-rep-3"),
        ],
        ("tool_repair", None),
        False,
    ),
}


@pytest.mark.parametrize("case", list(REPAIR_STEPS))
def test_body_line_gives_a_repair_its_pass1_step(case: str) -> None:
    """T-0140's review F2, kept by B1 ("a repair body takes its pass-1 step, here step 1"): ``body_line`` over session
    records gives a ``tool_repair`` body the step of the pass-1 request it repairs -- after ``t1-gen`` alone, step 1;
    after ``t1-gen`` and ``t1-gen-s2``, step 2 -- and ``None`` for a repair it cannot place. End to end through the
    oracle: the step-1 repair re-sending the overview right before the current frame is no finding (A2 "Repair");
    any other repair carrying it is an out-of-place finding, never a false pass."""
    records, want, in_place = REPAIR_STEPS[case]
    raw = {"generate:e-gen-1": _wire_body(overview=True)}
    if any(r["data"]["phase"] == "tool_step" for r in records):
        raw["generate:e-gen-s2-2"] = _wire_body(overview=False, draft="Scrolled. ")
    raw["repair:e-rep-3"] = _wire_body(overview=True, draft="Let me click. <|tool_call>call:")
    ov = {**_ov("unused"), "sha": hashlib.sha256(b"OVERVIEW-JPEG").hexdigest()[:16]}
    line = api("body_line")("B", "S4", 0, raw, records, [ov])
    by = {r["label"]: r for r in line["requests"]}
    assert (by["generate:e-gen-1"]["phase"], by["generate:e-gen-1"]["step"]) == ("generate", 1)
    repair = by["repair:e-rep-3"]
    assert (repair["phase"], repair["step"]) == want, (case, repair["phase"], repair["step"])
    trial = _tr("B", "S4", 0, DEALS_EMPTY, [TOWEL_SAID], auto_overview_logged=_counts(shown=1))
    found = oracle.configuration_findings([trial], [], [], [line], HAND_ARMS)
    if in_place:
        assert found == [], found
    else:
        assert any("repair:e-rep-3" in f and "out of place" in f for f in found), found


def test_body_line_retry_is_step_1() -> None:
    """A2 "Retry": the over-long retry is a second ``generate`` attempt of the same request id; ``body_line`` gives it
    step 1, so its re-sent overview is in place (no finding)."""
    records = [
        _request_record("generate", "t1-gen", "e-gen-1"),
        _request_record("generate", "t1-gen", "e-gen-2"),
    ]
    raw = {
        "generate:e-gen-1": _wire_body(overview=True),
        "generate:e-gen-2": _wire_body(overview=True),
    }
    ov = {**_ov("unused"), "sha": hashlib.sha256(b"OVERVIEW-JPEG").hexdigest()[:16]}
    line = api("body_line")("B", "S4", 0, raw, records, [ov])
    assert [(r["phase"], r["step"]) for r in line["requests"]] == [("generate", 1)] * 2
    trial = _tr("B", "S4", 0, DEALS_EMPTY, [TOWEL_SAID], auto_overview_logged=_counts(shown=1))
    assert oracle.configuration_findings([trial], [], [], [line], HAND_ARMS) == []


#: one turn-line key off its arm row per case: T-0140's review F3 cases, then T-0150's ``auto_overview`` key (an
#: arm-A turn line carrying it, an arm-B one without it or malformed); (trial, field, value)
LOGGED_FAULTS: dict[str, tuple[tuple[str, str, int], str, Any]] = {
    "live-feedback-not-passed": (("B", "S4", 0), "live_feedback_logged", None),
    "echo-pixels": (("A", "S1", 0), "live_feedback_logged", {**LABEL_FB, "echo": "pixels"}),
    "extent-on": (
        ("B", "S5", 0),
        "live_feedback_logged",
        {"screen_label": True, "page_extent": True, "echo": "model"},
    ),
    "recovery-flipped": (("A", "S2", 0), "recover_thought_calls_logged", False),
    "recovery-not-passed": (("B", "S4", 1), "recover_thought_calls_logged", None),
    "visual-tokens-560": (("B", "S2", 0), "visual_tokens_logged", 560),
    "guard-not-logged": (("A", "S4", 1), "degeneracy_guard_logged", None),
    "arm-a-turn-line-with-auto-overview": (
        ("A", "S4", 0),
        "auto_overview_logged",
        _counts(fits=1),
    ),
    "arm-a-turn-line-with-zero-counts": (("A", "S1", 0), "auto_overview_logged", _counts()),
    "arm-b-turn-line-without-auto-overview": (("B", "S1", 0), "auto_overview_logged", None),
    "arm-b-turn-line-missing-a-count": (
        ("B", "S2", 0),
        "auto_overview_logged",
        {"fits": 1, "failed": 0, "refused": 0},
    ),
    "arm-b-turn-line-bool-count": (
        ("B", "S4", 2),
        "auto_overview_logged",
        {"shown": False, "fits": False, "failed": True, "refused": False},
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(LOGGED_FAULTS))
def test_logged_settings_off_their_rows(which: str, case: str) -> None:
    """B4 "the arms' feedback, echo, recovery or guard settings off their rows" and "an arm-A turn line carrying
    ``auto_overview``, or an arm-B turn line without it", judged on the turn line (T-0140 review F3): each key the
    session logs off its arm row is a finding naming the trial; the same value on a trial that wrote no turn line
    (timed out, T-0135b A7) is not judged."""
    cf = _cf(which)
    (arm, task, repeat), field, value = LOGGED_FAULTS[case]
    trials, steps, regrounds, bodies = _clean()
    _find(trials, arm, task, repeat)[field] = value
    found = cf(trials, steps, regrounds, bodies, HAND_ARMS)
    tid = f"{arm} {task}#{repeat}"
    assert found and all(tid in f for f in found), (case, found)
    if field != "auto_overview_logged":
        assert any(field in f for f in found), (case, found)
    t = _find(trials, arm, task, repeat)
    t.update(timed_out=True, session_id=None)
    for k in (
        "live_feedback_logged",
        "recover_thought_calls_logged",
        "visual_tokens_logged",
        "degeneracy_guard_logged",
        "auto_overview_logged",
    ):
        t[k] = None
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], (
        "T-0135b A7: no turn line, nothing to judge"
    )


PROBES_OK: dict[str, Any] = {
    **{p: {"passed": True} for p in ("p1", "p2", "p3")},
    "p6": {**P6_OK, "passed": True},
    "p7": {**P7_OK, "passed": True},
    "p8": {**P8_OK, "passed": True},
}
#: a turn that raised before any body went out (O-11): nothing scanned
RAISED = {"scanned_requests": [], "exception": "RuntimeError('harness')", "budgets_sent": None}


@pytest.mark.parametrize("which", IMPLS)
def test_errored_turn_is_acceptance_not_configuration(which: str) -> None:
    """O-11 (T-0134 O-11, the run-validity safeguard kept): a trial that raised before sending any body is not judged
    by B4, even with a body declaring ``page_overview`` in its record (no finding, so not ``void``); it is an
    acceptance failure, so the run is ``inconclusive`` naming ``acceptance`` and the trial. Companion: the same body on
    a trial that did send is a finding."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    _find(trials, "A", "S1", 0).update(RAISED)
    _request(bodies, "A", "S1", 0, "generate:AS10-1")["tools"] = list(OV_TOOLS)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    sent = _clean()
    _request(sent[3], "A", "S1", 0, "generate:AS10-1")["tools"] = list(OV_TOOLS)
    assert cf(*sent, HAND_ARMS), "a sending trial is judged"
    if which == "harness":
        report = api("build_report")(trials, steps, regrounds, bodies, PROBES_OK, arms=HAND_ARMS)
        label, reason = report["decision"]["label"], report["decision"]["reason"]
        assert report["configuration_findings"] == [], report["configuration_findings"]
    else:
        inputs = {
            a: oracle.decision_inputs(
                [t for t in trials if t["arm"] == a], [s for s in steps if s["arm"] == a], []
            )
            for a in "AB"
        }
        label, reason = oracle.decide_with_reason(
            inputs,
            p1_passed=True,
            configuration_findings=cf(trials, steps, regrounds, bodies, HAND_ARMS),
        )
    assert (label, reason) == (
        "inconclusive",
        "arm A ineligible: acceptance (first failing trial S1#0)",
    ), (label, reason)


# =====================================================================================================
# B1/A5 turn-line evidence (the eval's ``arm_setting_findings``; T-0135b A7's exemption)
# =====================================================================================================


def test_arm_setting_findings() -> None:
    """B1/A5 evidence (eval oracle): the clean records have none and no exemption. A trial with a turn line but no
    A5 key in arm B, an A5 key in arm A, a ``page_overview`` key in either, or the guard not logged on, is a finding
    whatever its success; a timed-out trial (no turn line) is exempted from the turn-line keys only -- its
    ``page_overview``, ``auto_overview`` and ``tools_version`` are still judged."""
    trials, *_ = _clean()
    assert oracle.arm_setting_findings(trials) == ([], [])
    for success in (False, True):
        t = {**_find(trials, "B", "S4", 0), "auto_overview_logged": None, "success": success}
        got, ex = oracle.arm_setting_findings([t])
        assert got == ["B S4#0: auto_overview_logged None"] and ex == [], (success, got, ex)
    t = {**_find(trials, "A", "S4", 0), "auto_overview_logged": _counts()}
    got, _ = oracle.arm_setting_findings([t])
    assert got and "A S4#0" in got[0], got
    t = {**_find(trials, "B", "S1", 0), "page_overview_logged": {"calls": 0}}
    got, _ = oracle.arm_setting_findings([t])
    assert got and "B S1#0" in got[0] and "page_overview_logged" in got[0], got
    t = {**_find(trials, "A", "S4", 0), "degeneracy_guard_logged": None}
    assert oracle.arm_setting_findings([t])[0] == ["A S4#0: degeneracy_guard_logged None"]
    timed_out = {
        **_find(trials, "B", "S4", 1),
        "timed_out": True,
        "session_id": None,
        "auto_overview_logged": None,
        "degeneracy_guard_logged": None,
    }
    assert oracle.arm_setting_findings([timed_out]) == ([], ["B S4#1"])
    got, ex = oracle.arm_setting_findings([{**timed_out, "tools_version": OV_SHA}])
    assert got == [f"B S4#1: tools_version {OV_SHA!r}"] and ex == ["B S4#1"], (got, ex)
    got, ex = oracle.arm_setting_findings([{**timed_out, "auto_overview": False}])
    assert got == ["B S4#1: auto_overview False"] and ex == ["B S4#1"], (got, ex)


# =====================================================================================================
# B3: per-arm counts by hand
# =====================================================================================================


def _dist(n: int, p50: float | None, p95: float | None) -> dict[str, Any]:
    return {"n": n, "p50": p50, "p95": p95}


ZERO = dict.fromkeys(A5_KEYS, 0)
#: by hand from ``_hand_records`` (its docstring); nearest-rank percentiles (ceil(p/100 * n)-th smallest):
#: A -- success A1 (S4) and A4 (S1); false claims A2 (S4), A3 (S2) and A5 (S5): non-S4 2 = S5 1 + S1-S3 1;
#:      over-clicks A3 (S2) and A5 (S5); no A5 key; S4 scrolls A1; S4 naming the towel A1, the mug A2; step 1
#:      latency, none shown: done [300, 310, 340, 360, 380] -> p50 340 (3rd of 5), p95 380 (5th); first commit
#:      100 ms earlier [200, 210, 240, 260, 280] -> p50 240, p95 280.
#: B -- success B1 (S4), B3 (S5), B4 (S1); B5 has the right cart but ended ``degenerate`` (a failure, not an engine
#:      error though its aborted step 2 answered HTTP 0); false claims B2 and B6 (both S4): non-S4 0; no over-click;
#:      S4 shown turns B1, B2 (B6 failed); S5 shown B3; A5 sums S1 fits 1, S2 fits 1, S4 shown 2 failed 1, S5 shown
#:      1; S4 scrolls B1; naming the towel B1, the mug B2 (narrated) and B6; narration B2 ("whole page"); step-1
#:      latency shown: done [900, 1000, 1100] -> p50 1000, p95 1100; first commit [800, 900, 1000] -> 900, 1000;
#:      not shown: done [330, 340, 360] -> 340, 360; first commit [230, 240, 260] -> 240, 260.
HAND: dict[str, dict[str, Any]] = {
    "A": {
        "success": 2,
        "per_task": {"S1": 1, "S2": 0, "S3": 0, "S4": 1, "S5": 0},
        "s4": 1,
        "success_s5": 0,
        "success_s1_s3": 1,
        "false_claims": 3,
        "false_claims_non_s4": 2,
        "false_claims_s5": 1,
        "false_claims_s1_s3": 1,
        "over_clicks": 2,
        "over_clicks_s5": 1,
        "s4_shown_turns": 0,
        "s5_shown_turns": 0,
        "auto_overview_counts": {t: dict(ZERO) for t in TASK_IDS},
        "s4_scrolls": 1,
        "s4_named_cheapest": 1,
        "s4_named_mug": 1,
        "narration": 0,
        "degenerate": 0,
        "engine_errors": 0,
        "latency": {
            "shown": {"first_commit_ms": _dist(0, None, None), "done_ms": _dist(0, None, None)},
            "not_shown": {
                "first_commit_ms": _dist(5, 240.0, 280.0),
                "done_ms": _dist(5, 340.0, 380.0),
            },
        },
    },
    "B": {
        "success": 3,
        "per_task": {"S1": 1, "S2": 0, "S3": 0, "S4": 1, "S5": 1},
        "s4": 1,
        "success_s5": 1,
        "success_s1_s3": 1,
        "false_claims": 2,
        "false_claims_non_s4": 0,
        "false_claims_s5": 0,
        "false_claims_s1_s3": 0,
        "over_clicks": 0,
        "over_clicks_s5": 0,
        "s4_shown_turns": 2,
        "s5_shown_turns": 1,
        "auto_overview_counts": {
            "S1": {**ZERO, "fits": 1},
            "S2": {**ZERO, "fits": 1},
            "S3": dict(ZERO),
            "S4": {**ZERO, "shown": 2, "failed": 1},
            "S5": {**ZERO, "shown": 1},
        },
        "s4_scrolls": 1,
        "s4_named_cheapest": 1,
        "s4_named_mug": 2,
        "narration": 1,
        "degenerate": 1,
        "engine_errors": 0,
        "latency": {
            "shown": {
                "first_commit_ms": _dist(3, 900.0, 1000.0),
                "done_ms": _dist(3, 1000.0, 1100.0),
            },
            "not_shown": {
                "first_commit_ms": _dist(3, 240.0, 260.0),
                "done_ms": _dist(3, 340.0, 360.0),
            },
        },
    },
}
#: Wilson 95 % by the closed form (z = 1.959964, computed in a scratch script, 2026-10-24): A success 2/5, S4 1/2, S5
#: 0/1; B success 3/6, S4 1/3, S5 1/1
HAND_CI = {
    "A": {
        "success_ci": (0.11762, 0.76928),
        "s4_ci": (0.09453, 0.90547),
        "s5_ci": (0.0, 0.79345),
    },
    "B": {
        "success_ci": (0.18762, 0.81238),
        "s4_ci": (0.06149, 0.79234),
        "s5_ci": (0.20655, 1.0),
    },
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_hand_counts(which: str, arm: str) -> None:
    """B3: every count of the hand-built turns (S4 by ``score_v2``; success, per task, S5 and the S1-S3 sum; false
    claims, total, non-S4 and its S5 / S1-S3 split; over-clicks and S5's; the S4/S5 turns with ``shown`` = 1; A5's
    sums per task; S4 scrolls; S4 naming the cheapest vs the mug; narration; ``degenerate`` ends; engine errors; the
    step-1 first-commit and done latency split by overview shown) and the Wilson intervals of success, S4 and S5."""
    trials, steps, _ = _hand_records(arm)
    got = impl(which, "derive_arm")(trials, steps)
    for key, want in HAND[arm].items():
        assert got[key] == want, (arm, key, got[key], want)
    for key, (want_lo, want_hi) in HAND_CI[arm].items():
        lo, hi = got[key]
        assert abs(lo - want_lo) < 5e-5 and abs(hi - want_hi) < 5e-5, (key, got[key])


@pytest.mark.parametrize("which", IMPLS)
def test_split_metrics_follow_their_tasks(which: str) -> None:
    """B3's splits (mutant: a split read from the wrong task, or S5 left out of non-S4): moving arm A's S5 turn to S3
    moves its false claim from ``false_claims_s5`` to ``false_claims_s1_s3``; two blue mugs on A4 S1#0 make it a
    failure, a false claim ("Added.") and an over-click in ``over_clicks`` but not ``over_clicks_s5``; an S3 cart of
    two is never an over-click (S1/S2/S5 only). By hand: false claims A2 (S4), A3 (S2), A4 (S1), A5 (now S3) -> S5 0,
    S1-S3 3, non-S4 3; over-clicks A3 and A4 -> 2, S5 0; success S5 0, S1-S3 0."""
    trials, steps, _ = _hand_records("A")
    s5 = _find(trials, "A", "S5", 0)
    s5.update(task="S3", state={"page": "contact", "cart": ["a", "b"]})
    for s in steps:
        if (s["task"], s["repeat"]) == ("S5", 0):
            s["task"] = "S3"
    _find(trials, "A", "S1", 0)["state"] = {"page": "shop", "cart": ["blue mug", "blue mug"]}
    got = impl(which, "derive_arm")(trials, steps)
    assert (got["false_claims_s5"], got["false_claims_s1_s3"], got["false_claims_non_s4"]) == (
        0,
        3,
        3,
    ), got
    assert (got["over_clicks"], got["over_clicks_s5"]) == (2, 0), got
    assert (got["success_s5"], got["success_s1_s3"]) == (0, 0), got


def _s4(arm: str, repeat: int, **kw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One S4 turn that names the towel with an empty cart (a ``score_v2`` success unless forced)."""
    t = _tr(arm, "S4", repeat, DEALS_EMPTY, [TOWEL_SAID], **kw)
    return t, [_st(arm, "S4", repeat, 1, None, 300.0, voiced=(TOWEL_SAID,))]


FORCED: dict[str, tuple[dict[str, Any], dict[str, Any], int, int]] = {
    # (trial fields, step fields, success, engine errors), by hand from B5 (B-1 kept)
    "plain": ({}, {}, 1, 0),
    "degenerate": ({"tool_loop_end": "degenerate"}, {}, 0, 0),
    "degenerate-with-an-aborted-attempt": (
        {"tool_loop_end": "degenerate"},
        {"http_status": 0},
        0,
        0,
    ),
    "timed-out": ({"timed_out": True}, {}, 0, 1),
    "http-500": ({}, {"http_status": 500}, 0, 1),
    "a-503-retry": ({}, {"retries": [{"http_status": 503}]}, 0, 1),
    "engine-down-after": ({"engine_up_after": False}, {}, 0, 1),
    # O-B-8: a 400/413 over-long response the session's retry recovers is not an engine error; any other non-200 is
    "a-recovered-413-retry": ({}, {"retries": [{"http_status": 413}]}, 1, 0),
    "a-recovered-400-retry": ({}, {"retries": [{"http_status": 400}]}, 1, 0),
    "an-unrecovered-413": ({}, {"retries": [{"http_status": 413}], "http_status": 413}, 0, 1),
    "a-502-after-a-recovered-413": (
        {},
        {"retries": [{"http_status": 413}, {"http_status": 502}]},
        0,
        1,
    ),
    "a-connection-failure": ({}, {"http_status": None}, 0, 1),
    "an-http-0-abort-not-the-guards": ({}, {"http_status": 0}, 0, 1),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(FORCED))
def test_forced_failures(which: str, row: str) -> None:
    """B5 (B-1 kept): an engine-error trial (timeout, HTTP, a non-200 retry, the engine down after) and a T-0142
    ``degenerate`` end count as failed trials in every metric -- an S4 turn whose cart and words are right is not a
    success -- and only the first kind is an engine error."""
    t_over, s_over, success, errors = FORCED[row]
    t, steps = _s4("B", 0, **t_over)
    steps = [{**steps[0], **s_over}]
    got = impl(which, "derive_arm")([t], steps)
    assert (got["success"], got["s4"], got["engine_errors"]) == (success, success, errors), (
        row,
        got,
    )
    assert got["degenerate"] == (1 if "degenerate" in row else 0), got
    assert score_v2_passes(t), "premise: score_v2 passes the trial on its own"
    want_fc = 0 if success == 1 or row == "timed-out" else 1
    assert got["false_claims"] == want_fc, (row, got["false_claims"])


def score_v2_passes(t: dict[str, Any]) -> bool:
    return bool(oracle.score_v2(str(t["task"]), t.get("state"), " ".join(t.get("voiced") or [])))


@pytest.mark.parametrize("which", IMPLS)
def test_empty_arm_has_no_interval(which: str) -> None:
    """O-11: an arm with no trials (a probe-void run writes none) has ``None`` intervals and zero counts."""
    if which == "harness":
        probes = {**PROBES_OK, "p8": {**P8_OK, "passed": False}}
        report = api("build_report")(
            [], [], [], [], probes, arms=HAND_ARMS, stopped="not run: a pre-flight probe failed"
        )
        arm = report["arms"]["A"]
        assert report["void"] is True and report["decision"]["label"] == "void", report["decision"]
    else:
        arm = oracle.derive_arm([], [])
    assert arm["success_ci"] is None and arm["s4_ci"] is None and arm["s5_ci"] is None, arm
    assert arm["success"] == 0 and arm["engine_errors"] == 0, arm


#: Newcombe (1998) published values (as T-0132's rows) and Wilson's
WILSON = {(81, 263): (0.2553, 0.3662), (0, 20): (0.0, 0.1611), (1, 29): (0.0061, 0.1718)}
NEWCOMBE = {(56, 70, 48, 80): (0.0524, 0.3339), (9, 10, 3, 10): (0.1705, 0.8090)}


@pytest.mark.parametrize("which", IMPLS)
def test_wilson_and_newcombe_hand_values(which: str) -> None:
    """B3: Wilson 95 % and Newcombe's method 10 equal the published values to four decimals."""
    w, nc = impl(which, "wilson"), impl(which, "newcombe")
    for args, want in WILSON.items():
        assert all(abs(a - b) < 5e-5 for a, b in zip(w(*args), want, strict=True)), (args, w(*args))
    for args, want in NEWCOMBE.items():
        got = nc(*args)
        assert all(abs(a - b) < 5e-5 for a, b in zip(got, want, strict=True)), (args, got)


# =====================================================================================================
# B5: the decision rule (H-T150-1)
# =====================================================================================================


def _arm(**over: Any) -> dict[str, Any]:
    base = {
        "accepted": True,
        "safety": 0,
        "engine_errors": 0,
        "success": 65,
        "s4": 0,
        "s5": 14,
        "false_claims_non_s4": 5,
        "over_clicks": 3,
        "s4_shown_turns": 20,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
    }
    return base | over


def _inputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {"A": _arm(**a), "B": _arm(**b)}


#: (inputs, label) by hand from B5 (H-T150-1, T-0140's margins): every margin exactly (adopt) and one past (not);
#: ``overview not shown`` at 17 vs 18 of B's S4 turns shown
DECISIONS: dict[str, tuple[dict[str, Any], str]] = {
    "s4-exactly-plus-5": (_inputs({}, {"s4": 5}), "auto overview"),
    "s4-plus-4": (_inputs({}, {"s4": 4}), "no change"),
    "s4-margin-is-relative": (_inputs({"s4": 3}, {"s4": 8}), "auto overview"),
    "s4-margin-relative-minus-1": (_inputs({"s4": 3}, {"s4": 7}), "no change"),
    "success-exactly-minus-3": (_inputs({}, {"s4": 5, "success": 62}), "auto overview"),
    "success-minus-4": (_inputs({}, {"s4": 5, "success": 61}), "no change"),
    "success-within-the-tolerance": (
        _inputs({}, {"s4": 5, "success": 62 - 5e-10}),
        "auto overview",
    ),
    "false-claims-exactly-plus-2": (
        _inputs({}, {"s4": 5, "false_claims_non_s4": 7}),
        "auto overview",
    ),
    "false-claims-plus-3": (_inputs({}, {"s4": 5, "false_claims_non_s4": 8}), "no change"),
    "over-clicks-exactly-plus-2": (_inputs({}, {"s4": 5, "over_clicks": 5}), "auto overview"),
    "over-clicks-plus-3": (_inputs({}, {"s4": 5, "over_clicks": 6}), "no change"),
    "every-margin-at-its-boundary": (
        _inputs({}, {"s4": 5, "success": 62, "false_claims_non_s4": 7, "over_clicks": 5}),
        "auto overview",
    ),
    # the margins are B's allowance over A: A worse than B never blocks
    "a-worse-on-every-count": (
        _inputs({"success": 40, "false_claims_non_s4": 20, "over_clicks": 15}, {"s4": 5}),
        "auto overview",
    ),
    # H-T150-1 (b): S5 is report-only -- no S5 guard, however far S5 falls
    "s5-report-only": (_inputs({}, {"s4": 5, "s5": 0}), "auto overview"),
    # H-T150-1 (c): fewer than 18 of 20 S4 turns shown in B -> the mechanism failed
    "not-shown-at-17": (_inputs({}, {"s4": 4, "s4_shown_turns": 17}), "overview not shown"),
    "shown-at-18": (_inputs({}, {"s4": 4, "s4_shown_turns": 18}), "no change"),
    "not-shown-at-0": (_inputs({}, {"s4_shown_turns": 0}), "overview not shown"),
    # the adoption clauses are checked first (T-0140's order)
    "adopted-with-17-shown": (_inputs({}, {"s4": 5, "s4_shown_turns": 17}), "auto overview"),
    # arm A's shown count is never read (it has no auto overview)
    "arm-a-shown-not-read-low": (
        _inputs({"s4_shown_turns": 0}, {"s4_shown_turns": 18}),
        "no change",
    ),
    "arm-a-shown-not-read-high": (
        _inputs({"s4_shown_turns": 20}, {"s4_shown_turns": 17}),
        "overview not shown",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """B5 (H-T150-1; reviewer mutations 13 and 14): hand tables at every boundary -- S4 +5/20, success -3/100, non-S4
    false claims +2/80, over-clicks +2/60 -- each exactly (adopt) and one past (not), with the 1e-9 tolerance; S5
    report-only; ``overview not shown`` at 17 vs 18 shown S4 turns in B; adoption checked first."""
    inputs, want = DECISIONS[row]
    assert impl(which, "decide_with_reason")(inputs, p1_passed=True) == (want, None), row
    assert impl(which, "decide")(inputs, p1_passed=True) == want


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_engine_eligibility_edge(which: str, arm: str) -> None:
    """B5's engine clause (B-1 kept; the engine-clause mutant, back to "any error"): 2 engine errors in an arm are
    eligible (the rule decides: ``auto overview``), 3 are ``inconclusive`` naming the arm, ``engine`` and the first
    failing trial."""
    why = impl(which, "decide_with_reason")
    inputs = json.loads(json.dumps(DECISIONS["s4-exactly-plus-5"][0]))
    inputs[arm] |= {
        "engine_errors": 2,
        "first_failing": {"acceptance": None, "safety": None, "engine": "S2#4"},
    }
    assert why(inputs, p1_passed=True) == ("auto overview", None)
    inputs[arm]["engine_errors"] = 3
    assert why(inputs, p1_passed=True) == (
        "inconclusive",
        f"arm {arm} ineligible: engine (first failing trial S2#4)",
    )


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_arm(which: str, arm: str, clause: str) -> None:
    """B5 (T-0132 C6b): any arm ineligible -> ``inconclusive``, the reason naming the arm, the clause and the first
    failing trial, exactly; every margin met does not save it."""
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 1},
        "engine": {"engine_errors": 3},
    }[clause]
    bad = json.loads(json.dumps(DECISIONS["every-margin-at-its-boundary"][0]))
    bad[arm] |= flag | {
        "first_failing": {
            c: ("S4#7" if c == clause else None) for c in ("acceptance", "safety", "engine")
        }
    }
    got = impl(which, "decide_with_reason")(bad, p1_passed=True)
    assert got == ("inconclusive", f"arm {arm} ineligible: {clause} (first failing trial S4#7)"), (
        got
    )


@pytest.mark.parametrize("which", IMPLS)
def test_decision_order(which: str) -> None:
    """B5's order, each step against every later one: P1 failed -> ``inconclusive`` (``P1 failed``) over everything;
    a configuration finding -> ``void`` (``configuration: <first>``); a probe void (P2, P3, P6, P7 or P8) -> ``void``
    over a stop and an ineligible arm; stopped -> ``incomplete`` (its reason) over an ineligible arm; an ineligible
    arm over the rule."""
    why = impl(which, "decide_with_reason")
    good = DECISIONS["s4-exactly-plus-5"][0]
    bad = json.loads(json.dumps(good))
    bad["B"] |= {
        "accepted": False,
        "first_failing": {"acceptance": "S1#0", "safety": None, "engine": None},
    }
    everything = {"void": True, "stopped": "RUN_BUDGET_S", "configuration_findings": ["X", "Y"]}
    assert why(bad, p1_passed=False, **everything) == ("inconclusive", "P1 failed")
    assert why(bad, p1_passed=True, **everything) == ("void", "configuration: X")
    assert why(bad, p1_passed=True, void=True, stopped="RUN_BUDGET_S")[0] == "void"
    assert why(bad, p1_passed=True, stopped="RUN_BUDGET_S") == ("incomplete", "RUN_BUDGET_S")
    assert why(bad, p1_passed=True)[0] == "inconclusive"
    assert why(good, p1_passed=True, configuration_findings=[]) == ("auto overview", None)


def _engine_run(errors: int) -> tuple[list[Any], list[Any]]:
    """Arm B: five S4 successes, then ``errors`` engine-error trials (a timeout, an HTTP 500, the engine down after,
    another timeout ...), then a ``degenerate`` end whose aborted attempt answered HTTP 0, then a trial whose session
    log is incomplete when ``errors`` is negative (an acceptance failure)."""
    trials: list[Any] = []
    steps: list[Any] = []
    for r in range(5):
        t, s = _s4("B", r, auto_overview_logged=_counts(shown=1))
        trials.append(t)
        steps += s
    kinds = [
        ({"timed_out": True}, {}),
        ({}, {"http_status": 500}),
        ({"engine_up_after": False}, {}),
    ]
    for k in range(max(errors, 0)):
        t_over, s_over = kinds[k % len(kinds)]
        t, s = _s4("B", 10 + k, **t_over)
        trials.append(t)
        steps += [{**s[0], **s_over}]
    t, s = _s4("B", 19, tool_loop_end="degenerate")
    trials.append(t)
    steps += [{**s[0], "http_status": 0}]
    if errors < 0:
        t, s = _s4("B", 18, log_complete=False)
        trials.append(t)
        steps += s
    return trials, steps


def _di(which: str, trials: list[Any], steps: list[Any]) -> dict[str, Any]:
    if which == "harness":
        return dict(api("decision_inputs")(trials, steps, []))
    return oracle.decision_inputs(trials, steps, [])


@pytest.mark.parametrize("which", IMPLS)
def test_decision_inputs_engine_errors_are_not_acceptance(which: str) -> None:
    """B5 from records (B-1 kept): two engine errors (a timeout, an HTTP 500) and a ``degenerate`` end leave the arm
    accepted, with ``engine_errors`` 2 and the first engine-error trial named; a third (the engine down after) makes
    3; an incomplete session log is still an acceptance failure, named. The forced failures count as failures:
    success 5 of the 5 plain turns only; the 5 shown turns are B's ``s4_shown_turns``."""
    two = _di(which, *_engine_run(2))
    assert (two["accepted"], two["engine_errors"], two["success"], two["s4"]) == (True, 2, 5, 5), (
        two
    )
    assert two["s4_shown_turns"] == 5, two
    assert two["first_failing"] == {"acceptance": None, "safety": None, "engine": "S4#10"}, two
    three = _di(which, *_engine_run(3))
    assert (three["accepted"], three["engine_errors"]) == (True, 3), three
    log = _di(which, *_engine_run(-1))
    assert log["accepted"] is False and log["first_failing"]["acceptance"] == "S4#18", log
    assert log["engine_errors"] == 0, log


@pytest.mark.parametrize("which", IMPLS)
def test_recovered_over_long_retry_is_neither_engine_error_nor_acceptance_failure(
    which: str,
) -> None:
    """O-B-8 from records: three S4 turns whose step 1 answered 413 and then 200 on the session's retry leave the arm
    accepted with no engine error (test-author's consequence: an over-long prompt the session recovered is not a
    failed attempt either); the same turns with a 503 on the first attempt are three engine errors, the first named,
    and the arm ineligible on the engine clause."""
    trials: list[Any] = []
    steps: list[Any] = []
    for r in range(3):
        t, s = _s4("B", r, auto_overview_logged=_counts(shown=1))
        trials.append(t)
        steps += [{**s[0], "retries": [{"http_status": 413}]}]
    got = _di(which, trials, steps)
    assert (got["accepted"], got["engine_errors"], got["success"]) == (True, 0, 3), got
    steps = [{**s, "retries": [{"http_status": 503}]} for s in steps]
    got = _di(which, trials, steps)
    assert (got["engine_errors"], got["first_failing"]["engine"]) == (3, "S4#0"), got


# =====================================================================================================
# build_report wiring (harness only)
# =====================================================================================================


@pytest.mark.parametrize("fault", ["none", "configuration", "p8"])
def test_build_report_wiring(fault: str) -> None:
    """``build_report`` (mutants: findings dropped; the engine clause back to "any error"; P8 left out of ``void``,
    reviewer mutation 15): the hand records plus three engine-error S4 trials in arm B are ``inconclusive`` with
    exactly ``arm B ineligible: engine (first failing trial S4#10)``; with an S1 trial showing an overview the run is
    ``void`` with a reason ``configuration: B S1#0 ...``; with P8 failed it is ``void`` (``void`` true). The
    differences are S4, success and S5 B-A."""
    trials, steps, regrounds, bodies = _clean()
    more_t, more_s = _engine_run(3)
    trials += more_t[5:-1]
    steps += more_s[5:-1]
    probes = copy.deepcopy(PROBES_OK)
    if fault == "configuration":
        _shown_on(trials, bodies, "B", "S1", 0)
    if fault == "p8":
        probes["p8"] = {**_p8(S2="shown"), "passed": False}
    report = api("build_report")(trials, steps, regrounds, bodies, probes, arms=HAND_ARMS)
    d = report["decision"]
    if fault == "configuration":
        assert d["label"] == "void" and str(d["reason"]).startswith("configuration: B S1#0"), d
    elif fault == "p8":
        assert report["void"] is True and d["label"] == "void", (report["void"], d)
        assert report["configuration_findings"] == []
    else:
        assert report["void"] is False
        assert (d["label"], d["reason"]) == (
            "inconclusive",
            "arm B ineligible: engine (first failing trial S4#10)",
        ), d
        assert report["configuration_findings"] == []
    assert d["inputs"]["B"]["engine_errors"] == 3, d["inputs"]["B"]
    assert d["inputs"]["A"]["first_failing"] == {"acceptance": None, "safety": None, "engine": None}
    assert set(report["differences"]) == {"s4 B-A", "success B-A", "s5 B-A"}, report["differences"]


def test_build_report_counts_equal_the_oracle() -> None:
    """``build_report`` over the hand records of both arms: each arm's B3 counts are ``HAND``'s, the decision inputs
    are the oracle's re-derivation, the label is the oracle's, and the differences are the oracle's Newcombe
    intervals at B3's fixed denominators."""
    trials, steps, regrounds, bodies = _clean()
    report = api("build_report")(trials, steps, regrounds, bodies, PROBES_OK, arms=HAND_ARMS)
    inputs = {}
    for arm in ("A", "B"):
        for key, want in HAND[arm].items():
            assert report["arms"][arm][key] == want, (arm, key, report["arms"][arm][key], want)
        mine = [[r for r in rows if r["arm"] == arm] for rows in (trials, steps)]
        inputs[arm] = oracle.decision_inputs(mine[0], mine[1], [])
        assert report["decision"]["inputs"][arm] == inputs[arm], (
            arm,
            report["decision"]["inputs"][arm],
        )
    assert (report["decision"]["label"], report["decision"]["reason"]) == oracle.decide_with_reason(
        inputs, p1_passed=True
    )
    want_diff = oracle.differences({a: HAND[a] for a in ("A", "B")})
    for key, (lo, hi) in want_diff.items():
        got = report["differences"][key]
        assert abs(got[0] - lo) < 1e-9 and abs(got[1] - hi) < 1e-9, (key, got, (lo, hi))


# =====================================================================================================
# main on CPU: the dress (T-0131's engine rig, the FakeLiveBrowser with a scripted overview, a fake LLM)
# =====================================================================================================

#: deals-like metrics: 1520 CSS px over an 800 px viewport, 1.9 screens, not clipped
DRESS_METRICS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1520.0}


def _dress_overview() -> Any:
    from local_astra.contracts.perception import Frame, Source
    from tests.unit._live_browser_support import jpeg_frame

    return Frame(
        source=Source.WINDOW,
        ts_ms=0,
        width=838,
        height=798,
        jpeg=jpeg_frame("dress-overview").jpeg,
        phash="0d0d0d0d0d0d0d0d",
    )


def _dress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    *,
    turns: tuple[tuple[str, str, int], ...] = (("A", "S4", 0), ("B", "S4", 0)),
    probes: Any = None,
    p6: Any = None,
    p7: Any = None,
    p8: Any = None,
    past_budget: bool = False,
) -> tuple[int, Path, Any]:
    """A CPU run of ``main`` (T-0140's dress, adapted): S4 in each arm of ``turns`` over a 1.9-screen page; step 1
    says "Let me look." and scrolls; step 2 names the towel and $4.75. Arm B's session captures the overview itself
    (A1). A ``load_tool_kit(..., page_overview=True)`` would give the overview kit (a T-0140-shaped arm B). Returns
    (exit code, out-dir, the engine rig's state)."""
    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit
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
    kits = _stand_in_kits()
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(
        h=h, cli=cli, kit=kits[False], mp=monkeypatch, out=out, capsys=capsys
    )
    st = t31._engine(rig)
    overview = _dress_overview()

    class Pages:
        origin = "http://127.0.0.1:9"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Page:
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"page": "deals", "cart": []}
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
            n = len(st.browsers)
            super().__init__(
                [jpeg_frame(f"onset-{n}")],
                post_frames=[jpeg_frame(f"after-{n}-{k}") for k in (1, 2, 3)],
                now_ms=now_ms or (lambda: 0),
                echo=echo,
            )
            self.overview_image = overview
            self.page_metrics = dict(DRESS_METRICS)
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st.browsers.append(self)
            st.pending_browser = self

    def turn_llm(browser: Any) -> Any:
        llm = ToolLlm([answer("Let me look. "), tool(SCROLL_CALL)])
        llm.script_steps([answer(TOWEL_SAID)])
        return llm

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        if live_browser is None:
            return (kits[bool(page_overview)], None)
        return (overview_kit(live_browser) if page_overview else live_kit(live_browser), None)

    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(t31, "_turn_llm", turn_llm)
    monkeypatch.setattr(h, "run_probes", lambda base_url, model: probes or t31._probes())
    monkeypatch.setattr(
        h, "probe_p6", lambda *a, **k: copy.deepcopy(p6 if p6 is not None else P6_OK)
    )
    monkeypatch.setattr(h, "probe_p7", lambda *a, **k: dict(p7 if p7 is not None else P7_OK))
    monkeypatch.setattr(
        h, "probe_p8", lambda *a, **k: copy.deepcopy(p8 if p8 is not None else P8_OK)
    )
    monkeypatch.setattr(h, "order", lambda seed: list(turns))
    if past_budget:
        monkeypatch.setattr(h, "PROCESS_START_S", time.monotonic() - h.RUN_BUDGET_S - 1)
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    return rc, out, st


def _read(out: Path) -> dict[str, Any]:
    return {
        "report": json.loads((out / "report.json").read_text()),
        "manifest": json.loads((out / "manifest.json").read_text()),
        "probes": json.loads((out / "probes.json").read_text()),
        **{
            k: oracle._jsonl(out / f"{k}.jsonl") for k in ("trials", "steps", "regrounds", "bodies")
        },
    }


def _dress_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> dict[str, Any]:
    rc, out, st = _dress(tmp_path, monkeypatch, capsys)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = _read(out)
    run["browsers"] = list(st.browsers)
    return run


def _dress_default_is_heads(manifest: dict[str, Any]) -> None:
    """The manifest's live default is HEAD's as ``cli`` gives it, with both overviews off; its arm rows and sha are
    the oracle's over it."""
    from local_astra import cli
    from local_astra.config import load

    d = manifest["live_default"]
    assert d == _heads_default(cli, load().profiles["lean"]), d
    assert oracle.live_default_findings(d) == [], oracle.live_default_findings(d)
    rows = oracle.arm_rows(d)
    assert manifest["regime"]["arms"] == rows, manifest["regime"]["arms"]
    assert manifest["arm_table_sha256"] == oracle.arm_table_sha(rows)


def test_eval_oracle_re_derives_the_dress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B7 on CPU: over the files a dress of ``main`` wrote, the eval oracle's re-derivations agree with
    ``report.json`` -- per arm the B3 counts, T-0131's sections, the safety counts (all 0) and the decision inputs;
    the findings (none: the dress is clean); the probes (P8 included) and ``void``; the label and reason; the
    differences; the turn-line evidence."""
    run = _dress_run(tmp_path, monkeypatch, capsys)
    assert len(run["trials"]) == 2, "premise: two turns ran"
    for arm in ("A", "B"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_t0131_sections_re_derived(run, arm)
        oracle.test_safety_counts_zero(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)
    oracle.test_configuration_findings_re_derived(run)
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]
    oracle.test_probes_re_derived(run)
    oracle.test_p8_census(run)
    oracle.test_decision_re_derived(run)
    oracle.test_differences_re_derived(run)
    oracle.test_live_default_in_both_arms(run)
    oracle.test_arm_setting_reached_every_turn(run)
    _dress_default_is_heads(run["manifest"])


def test_main_dress_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """B1/B3/B4 record fields from a real ``main`` (mutants: arm B built with ``page_overview=True``; the
    ``auto_overview`` keyword not passed; ``overview_shown`` read from the arm; the bodies' image identities not the
    JPEG's): both arms' kits are v1 and no body declares ``page_overview``; arm A's browser saw no capture and its turn
    line has no ``auto_overview`` key; arm B's browser saw exactly one capture, its turn line counts ``shown`` 1, its
    step-1 generate carries the overview (sha16 of the overview frame's JPEG) right before the current frame with
    T-0140's label, and its step-2 request does not; the guard is on and logged in both; every arm setting is HEAD's."""
    run = _dress_run(tmp_path, monkeypatch, capsys)
    by = {t["arm"]: t for t in run["trials"]}
    steps = {
        a: sorted((s for s in run["steps"] if s["arm"] == a), key=lambda s: int(s["step"]))
        for a in "AB"
    }
    bodies = {b["arm"]: b for b in run["bodies"]}
    assert (by["A"]["tools_version"], by["B"]["tools_version"]) == (V1, V1), by
    assert (by["A"]["auto_overview"], by["B"]["auto_overview"]) == (False, True)
    assert (by["A"]["page_overview"], by["B"]["page_overview"]) == (False, False)
    assert by["A"]["auto_overview_logged"] is None, by["A"]["auto_overview_logged"]
    assert by["B"]["auto_overview_logged"] == _counts(shown=1), by["B"]["auto_overview_logged"]
    assert by["A"]["page_overview_logged"] is None and by["B"]["page_overview_logged"] is None
    browsers = run["browsers"]
    assert len(browsers) == 2, browsers
    assert [b.overview_captures() for b in browsers] == [0, 1], "A1: one capture, arm B only"
    for arm in "AB":
        assert by[arm]["degeneracy_guard"] is True, by[arm]
        assert (by[arm]["degeneracy_guard_logged"] or {}).get("on") is True, by[arm]
        for k in oracle.DEFAULT_KEYS:
            assert by[arm][k] == run["manifest"]["live_default"][k], (arm, k)
        d = run["manifest"]["live_default"]
        lf = d["live_feedback"]
        want_lf = None if lf is None else {**lf, "echo": d["echo"]}
        assert by[arm]["live_feedback_logged"] == want_lf, (arm, by[arm]["live_feedback_logged"])
        assert by[arm]["recover_thought_calls_logged"] is d["recover_thought_calls"], (arm, by[arm])
        assert by[arm]["visual_tokens_logged"] == d["visual_tokens"], (arm, by[arm])
        for r in bodies[arm]["requests"]:
            assert r["tools"] is None or "page_overview" not in r["tools"], (arm, r["label"])
        assert any(
            r["phase"] == "generate" and r["tools"] == V1_TOOLS for r in bodies[arm]["requests"]
        )
    assert [s["overview_shown"] for s in steps["B"]] == [True, False], steps["B"]
    assert not any(s["overview_shown"] for s in steps["A"]), steps["A"]
    want_sha = hashlib.sha256(_dress_overview().jpeg).hexdigest()[:16]
    (ov,) = bodies["B"]["overviews"]
    assert ov["step"] == 1 and ov["sha"] == want_sha and abs(float(ov["screens"]) - 1.9) < 1e-9, ov
    assert ov["clipped"] is False and (ov["width"], ov["height"]) == (838, 798), ov
    (gen,) = [r for r in bodies["B"]["requests"] if r["phase"] == "generate"]
    assert gen["step"] == 1, gen["step"]
    parts = _final_user(gen)["parts"]
    k = parts.index({"image": want_sha})
    assert parts[k + 1] == {"text": ov_label(1.9, False)}, parts
    assert "image" in parts[k + 2] and not any("image" in p for p in parts[k + 3 :]), parts
    for r in bodies["B"]["requests"]:
        if r is not gen:
            flat = [p for m in r["messages"] for p in m["parts"]]
            assert {"image": want_sha} not in flat, f"A2: the overview in {r['label']}"
    flat_a = json.dumps(bodies["A"]["requests"])
    assert bodies["A"]["overviews"] == [] and oracle.OVERVIEW_STEM not in flat_a
    _dress_default_is_heads(run["manifest"])


def test_main_reads_the_arms_from_cli_at_run_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B1 ("read once from cli at run time, never hard-coded"): with ``cli.feedback_of`` and ``cli.recovery_of``
    monkeypatched (``label+extent`` with pixel echo, recovery off), a dress records that default in the manifest, its
    rows carry it, and both arms' sessions logged it on their turn lines -- with no configuration finding."""
    from local_astra import cli

    monkeypatch.setattr(
        cli, "feedback_of", lambda p, a, m: cli.Feedback("label+extent", "flag", "pixels", "flag")
    )
    monkeypatch.setattr(cli, "recovery_of", lambda p, a, m: cli.Recovery(False, "flag"))
    rc, out, _ = _dress(tmp_path, monkeypatch, capsys)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = _read(out)
    d = run["manifest"]["live_default"]
    assert (d["live_feedback"], d["echo"], d["recover_thought_calls"]) == (
        FEEDBACK_DICT["label+extent"],
        "pixels",
        False,
    ), d
    assert run["manifest"]["regime"]["arms"] == oracle.arm_rows(d)
    for t in run["trials"]:
        assert t["live_feedback_logged"] == {**FEEDBACK_DICT["label+extent"], "echo": "pixels"}, t
        assert t["recover_thought_calls_logged"] is False, t
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]


def test_run_budget_from_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B6 (``RUN_BUDGET_S`` from process start, probes included): with the process started more than
    ``RUN_BUDGET_S`` ago, ``main`` runs no turn, still writes ``report.json``, and the run is ``incomplete``
    (exit 1)."""
    rc, out, st = _dress(tmp_path, monkeypatch, capsys, past_budget=True)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    assert report.get("stopped") and report["decision"]["label"] == "incomplete", report["decision"]
    assert st.browsers == [], f"a turn ran past the budget: {len(st.browsers)} browser(s)"
    assert rc == harness().EXIT_FAILED, rc


PROBE_FAILURES: dict[str, tuple[dict[str, Any], str, str]] = {
    "p1": ({"probes": t31._probes_with(p1_560=9999)}, "inconclusive", "p1"),
    "p2": ({"probes": t31._probes_with(p2=False)}, "void", "p2"),
    "p3": ({"probes": t31._probes_with(p3=False)}, "void", "p3"),
    "p6": ({"p6": _p6(deals={"scroll_y_after": 2.0})}, "void", "p6"),
    "p7-t0140s-3-image-shape": ({"p7": {**P7_OK, "images": 3}}, "void", "p7"),
    "p7-one-token-over": ({"p7": {**P7_OK, "tokens": 16_129}}, "void", "p7"),
    # reviewer mutation 15 at main: a census of S2 shown, or S5 fits, voids the run
    "p8-s2-shown": ({"p8": _p8(S2="shown")}, "void", "p8"),
    "p8-s5-fits": ({"p8": _p8(S5="fits")}, "void", "p8"),
}


@pytest.mark.parametrize("probe", list(PROBE_FAILURES))
def test_probe_failure_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, probe: str
) -> None:
    """B2: a failed probe runs no turn, exits 3 and still writes ``report.json``: P1 -> ``inconclusive``; P2, P3, P6
    (the page scrolled), P7 (T-0140's 3-image shape, or one token over the window) or P8 (S2 shown, or S5 fits) ->
    ``void``; the recorded ``passed`` is the oracle's."""
    over, want, name = PROBE_FAILURES[probe]
    rc, out, st = _dress(tmp_path, monkeypatch, capsys, **over)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    probes = json.loads((out / "probes.json").read_text())
    assert report["decision"]["label"] == want, report["decision"]
    assert probes[name]["passed"] is False and oracle.probe_verdicts(probes)[name] is False, probes[
        name
    ]
    assert st.browsers == [], "a turn ran after a failed probe"
    assert rc == harness().EXIT_PROBE == 3, rc


# =====================================================================================================
# B6: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_OUT = f"logs/t0150-auto-overview-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0150_auto_overview_ab.py"
L_EVAL = "tests/eval/test_t0150_ab.py"
#: the merge subject that satisfies the check (reading 5)
MERGED_B = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0150 group B: the locked 2-arm auto-overview A/B"
#: merges that do not: group A's merge on main at BASE_B and T-0140 group B's (verbatim subjects), T-0150b, branch
#: merges
OTHERS = (
    "55db4e255db4e255db4e255db4e255db4e255db4 Merge T-0150 group A: --auto-overview (live, off by default) shows "
    "the whole-page picture at step 1 when the page is taller than the screen; no new tool; byte-identical when off",
    "e3c01c0e3c01c0e3c01c0e3c01c0e3c01c0e3c01 Merge T-0140 group B: locked 2-arm GPU A/B of page_overview on S4 "
    "(harness, locked script, eval oracle; CPU-verified, GPU run awaits /launch)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0150b: adopt the auto overview default",
    # O-B-4: a bare task merge does not count either; only "Merge T-0150 group B"
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge T-0150: auto overview + locked 2-arm A/B",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0150-b-ab' into T-0150-auto-overview",
    "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge main into T-0150-b-ab (T-0151)",
)
#: the only repository paths the locked script may require to exist (its imports, its eval and the v1 tool file)
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0135_thought_call_ab.py",
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0132_crop_budget_ab.py",
    "scripts/gpu/t0131_grounding_ab.py",
    "scripts/gpu/t0111g_live_smoke.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/thought_call.py",
)


def _locked(monkeypatch: pytest.MonkeyPatch, merged: str = MERGED_B, filler: int = 0) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0150 script, harness, eval and merge;
    ``filler`` other merge lines after the merge (a log past the 64 KiB pipe buffer)."""
    assert LOCKED.is_file(), f"T-0150 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    git_stub = t31.GIT_STUB.replace(t31.MERGED_LINE, merged)
    for other in t31.OTHER_MERGES:
        git_stub = git_stub.replace(f'echo "{other}"; ', "")
    git_stub = git_stub.replace(
        "*--merges*) ", "*--merges*) " + "".join(f'echo "{m}"; ' for m in OTHERS)
    )
    if filler:
        fill = (
            f'for i in $(seq 1 {filler}); do echo "9999999999999999999999999999999999999999 Merge T-0999: '
            'filler line $i of the merge log, long enough to pass the pipe buffer"; done'
        )
        before = f'echo "{merged}"; fi ;;'
        assert before in git_stub, "premise: T-0131's git stub prints the merge line this way"
        git_stub = git_stub.replace(before, f'echo "{merged}"; fi; {fill} ;;')
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", git_stub)
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0150_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json bodies.jsonl; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged-only-group-a-and-t0140-merges": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """B6: ``REFUSED`` and exit 2 before the wheel build, the boot or any step: an argument, a dirty tree, an Astra
    port bound (STT must be down), running containers, ``services/vllm.sh`` changed since 806da12, this SHA's out-dir
    present, and no ``Merge T-0150 group B`` merge reachable from HEAD -- adversarial: the log holds ``Merge T-0150
    group A`` (on main at BASE_B, verbatim), ``Merge T-0140 group B`` (verbatim), ``Merge T-0150b`` and branch merges
    (reading 5)."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B6, the other side of every refusal (``Merge T-0150 group B`` reachable): the harness under ``timeout -k 60s
    75m`` with a loopback ``--base-url`` and ``--out-dir logs/t0150-auto-overview-ab-<sha7>``, then
    ``tests/eval/test_t0150_ab.py -m gpu`` with ``T0150_RUN_DIR`` set to it; every ``uv run`` uses ``--locked``;
    exit 0."""
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
    assert evals[0].env.get("T0150_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv), [c.argv for c in res.uv]
    assert len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 75 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_long_merge_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B6 (the T-0135 here-string safeguard; mutant ``printf '%s' "$MERGES" | grep -q`` under pipefail): a merged HEAD
    whose merge log runs past the 64 KiB pipe buffer after the ``Merge T-0150 group B`` line (3000 filler merges,
    about 380 KB) still passes the check and runs both steps."""
    res = _locked(monkeypatch, MERGED_B, filler=3000)(tmp_path)
    assert res.rc == 0 and "REFUSED" not in res.output, (res.rc, res.output[-2000:])
    assert [c for c in res.steps() if c.names(L_EVAL)], "the eval did not run"


def test_locked_script_eval_after_a_failed_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B6: the harness failing after it wrote ``report.json`` (a late failure: exit 1) still runs the eval over that
    report, and the script exits 1."""
    res = _locked(monkeypatch)(tmp_path, fail_match=L_HARNESS, fail_code=1)
    evals = [c for c in res.steps() if c.names(L_EVAL)]
    assert evals, f"the eval did not run after the harness failed: {res.output[-2000:]}"
    assert res.rc == 1, (res.rc, res.output[-2000:])


def test_locked_script_shape() -> None:
    """B6 by text: the T-0134 boot regime (``services/vllm.sh`` against 806da12), the ``Merge T-0150 group B`` check
    through a here-string (never a pipe into ``grep``), no ``Merge T-0140`` check left over from the fork, the
    ``timeout -k 60s 75m`` bound, ``--locked`` on both ``uv run`` steps, the out-dir and the eval only if
    ``report.json`` exists; ``bash -n`` clean."""
    assert LOCKED.is_file(), f"T-0150 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    text = LOCKED.read_text(encoding="utf-8")
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    assert "timeout -k 60s 75m" in body, body[:600]
    checks = [ln for ln in code if "Merge T-0150" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("group B" in ln for ln in checks), checks
    assert not [ln for ln in code if "Merge T-0140" in ln], "the fork kept T-0140's merge check"
    assert not [
        ln
        for ln in code
        if "grep" in ln and ("MERGES" in ln or "--merges" in ln) and "|" in ln.split("grep")[0]
    ], "the merge log is piped into grep"
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert "t0150-auto-overview-ab-" in body and "tests/eval/test_t0150_ab.py" in body
    assert "scripts/gpu/t0150_auto_overview_ab.py" in body
    assert "t0140_overview_ab" not in body and "test_t0140_ab" not in body, "T-0140's files"
    assert "-m gpu" in body and "report.json" in body
    assert "806da12" in body and "services/vllm.sh" in body, "the T-0134 boot regime"


# =====================================================================================================
# B8: negative -- the T-0140 B files and the other harnesses are untouched; no local_astra/ change in group B
# =====================================================================================================

#: sha256 of each file at BASE_B (1bed93f), by the test-author 2026-10-24 (``sha256sum``). Content pins: these hold
#: forever (R-b: the T-0140 files are the record of a completed locked run; the T-0134/T-0135 harnesses are imported).
FROZEN_SHA256 = {
    "scripts/gpu/t0140_overview_ab.py": "675c75d9eb8c0f4c0b4ade7a636086b2904c30fe1687061da27b52f4b91647ec",
    "scripts/locked/t0140_overview_ab.sh": "95679378f785b7a2ea02e91c1bf8bfc23843879c0ec1921ece40b179d802dc60",
    "tests/unit/test_t0140_ab_harness.py": "d9224d98d3f12f8e17e9e88abd008d039e2179ba891d72842c8a3aae9a785f89",
    "tests/eval/test_t0140_ab.py": "4ad5fccf3ec4844597cb49ebe6e0114b604fe34b33bafdd5a476fdcadad0058e",
    "scripts/gpu/t0134_feedback_ab.py": "e70261f8066dd5805f1b213f9a94ebb8518ab18dc7e683820883663d09f80719",
    "scripts/gpu/t0135_thought_call_ab.py": "f06c4a06581f33c2140b1bd76e6ed035e23c3be6724ab1cad9f729dff2127e40",
}


@pytest.mark.parametrize("rel", list(FROZEN_SHA256))
def test_t0140_b_files_and_imported_harnesses_unchanged(rel: str) -> None:
    """B8 (and R-b): the T-0140 B files and the T-0134/T-0135 harnesses keep their BASE_B bytes (content pins, green
    now and forever)."""
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == FROZEN_SHA256[rel], f"T-0150 B8: {rel} changed ({got})"


def test_no_local_astra_change_in_group_b() -> None:
    """B8: no ``local_astra/`` file changes in group B, over the closed range ``BASE_B..TIP_B`` (1bed93f..8c83a09,
    two literals, T-0094), with BASE_B an ancestor of TIP_B. Power: the same predicate sees group A's change of
    ``session.py`` over 55db4e2^1..1bed93f, and group B's own new harness over BASE_B..TIP_B under ``scripts``."""
    gr.require_commits(BASE_B, TIP_B, "55db4e2")
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B} is not an ancestor of {TIP_B}"
    changed = gr.changed(BASE_B, TIP_B, "local_astra")
    assert changed == [], f"T-0150 B8: group B changed {changed}"
    assert "local_astra/conductor/session.py" in gr.changed("55db4e2^1", BASE_B, "local_astra")
    assert "scripts/gpu/t0150_auto_overview_ab.py" in gr.changed(BASE_B, TIP_B, "scripts")
