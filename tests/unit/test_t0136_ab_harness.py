"""T-0136 group D: the locked 2-arm GPU A/B of ``--action-outcome`` -- the harness's pure parts, its ``main`` on CPU,
its locked script, and D-6 (unit; CPU; no engine, no browser). docs/tasks/T-0136.md D-1...D-6, "Tests to write (D)",
reviewer mutation 11, the Rulings (H-T136-4, H-T136-5).

BASE_D 1235813 (``Merge main into T-0136-d-ab``: main at df658ba, after ``Merge T-0136 group C`` 4b45421, so A2, B
and C are in). Activated 2026-10-25: no row is pending; the harness rows are red on "does not exist yet" until the
implementer lands it. The content pins were re-checked at 1235813 (no pinned file changed since d4adf4f).

**Re-targeting (James, 2026-10-25 ~11:40, docs/tasks/T-0136.md Rulings).** D-1 as planned forks T-0151's harness on
S1-S5. T-0138's suite and harness landed since, and ``docs/cd/suite-baseline.md`` recommends later A/Bs run on the
suite. These rows are therefore written for a fork of **T-0138's** harness on S6-S15: arm A HEAD's live default, arm B
the same plus ``action_outcome=True``, 2 x 10 x 10 = 200 turns, T-0138's joint order (seed 0), paired McNemar by
``(task, repeat)``, T-0138's probes, its bounds (``RUN_BUDGET_S`` 85 min, ``timeout -k 60s 100m``), D-3's voids, and
D-4 re-expressed on the suite (the eval's docstring states the rule; D-4's combined -2 edge is kept), and
``mechanism not exercised`` when arm B places fewer than 10 clauses, checked before adoption. O-B2 (F3): B-4's
turn-line counts sum to *at most* the executed pointer actions; no row equates them.

Rows that need the harness (``scripts/gpu/t0136_outcome_ab.py``) or the locked script
(``scripts/locked/t0136_outcome_ab.sh``) are red on a named assertion in ``harness()``/``api()``/``_locked()``, never on
an import error. Rows parametrised on ``which`` run once against the harness and once against the eval oracle
(``tests/eval/test_t0136_ab.py``, green now), so a wrong oracle is caught on CPU. Every expected value is computed by
hand (the comments say how) or by a scratch script of the test-author's (sha256 of hand-written JSON), never read off
an implementation (none exists).

**The assumed harness surface** (names and shapes are the test-author's; unless stated, a name keeps T-0138's
harness's shape, ``tests/unit/test_t0138_ab_harness.py``'s docstring)

Forked, never imported (D-1 re-targeted)
    A copy of ``scripts/gpu/t0138_suite_ab.py``; T-0138's and T-0151's harnesses are neither imported, nor loaded, nor
    named in code. T-0134's chain is imported as T-0138's harness imports it (``LoopMeter``, ``sentinel_hits``,
    ``log_complete``, ``non_loopback_sockets``, ``run_probes``, ``probe_p1``, ``config_evidence``, ``wilson``,
    ``newcombe``); ``TASKS``, ``SUITE_DIR``, ``judge`` are ``suite_tasks``'s; ``mcnemar_exact`` is
    ``eval.cd.stats``'s; ``turn_records``, ``false_claim``, ``url_leaks`` are ported (defined in the file).

The arms (never hard-coded)
    ``live_default(profile)`` gains ``"action_outcome"`` (``cli.action_outcome_of(profile, ns, "live")``, C-2);
    ``arm_rows(default) -> [row_A, row_B]``, each ``{"label", <the five defaults>, "degeneracy_guard": True,
    "page_overview": False, "auto_overview": False, "overview_frame": True, "action_outcome", "tools_sha256": V1}``,
    **A's action_outcome False and B's True**; ``arm_table_sha(rows)``; ``live_default_findings(default)`` (an
    ``action_outcome``-on default names ``action_outcome`` and "identical arms"; a default without the screen label
    names ``live_feedback``: arm B cannot be built, B-1; a frame-off default names ``overview_frame``).

Constants
    ``EXIT_OK/FAILED/REFUSED/PROBE`` 0-3, ``REPEATS = 10``, ``ORDER_SEED = 0``, ``RUN_BUDGET_S = 85 * 60``,
    ``PROCESS_START_S``, ``SUITE_TABLE_SHA256``, ``ACTION_OUTCOME_ON = {"A": False, "B": True}``, ``OUTCOME_CLAUSE``
    (B-3's, equal to ``session.OUTCOME_CLAUSE``), ``OUTCOME_VOCABULARY`` (A2-2's 12 strings).

Pure functions (each also in the eval oracle, same name and shape; the harness's ``arm_metrics`` is the oracle's
``derive_arm``)
    ``order``, ``arm_rows``, ``arm_table_sha``, ``live_default_findings``, ``ps1_ok``, ``ps2_ok``, ``ps3_ok``,
    ``configuration_findings(trials, steps, regrounds, bodies, arms)`` (D-3 plus T-0138 B7), ``arm_metrics(trials,
    steps)`` (D-2: T-0138's B4 keys plus ``over_actions``, ``outcomes_logged``, ``outcome_kinds``,
    ``unmeasured_reasons``, ``cart_grew_but_none``, ``identical_repeat_clicks``), ``clause_counts(bodies)``,
    ``mcnemar_sections(trials, steps)`` (``{"pooled", "per_task", "unpaired"}``), ``decision_inputs(trials, steps,
    regrounds, bodies=())`` (``clauses_placed`` from the arm's bodies), ``decide_with_reason(inputs, *, p1_passed,
    void=False, stopped=None, configuration_findings=())`` (``inputs`` ``{"A", "B"}``; ``("mechanism not exercised",
    "<n> clauses placed")``, ``("outcome label", None)`` or ``("no change", <first failing clause>)``),
    ``build_report(trials, steps, regrounds, bodies, probes, *, arms, stopped=None)`` (``report.json`` as T-0138's
    plus ``"clauses"``), ``render_markdown(report, manifest)``.

Refusals (exit 2, before any socket, leaving the out-dir as it was)
    T-0138's (dirty tree, non-loopback URL, out-dir present, repair off, ``LIVE_TOOL_GUIDANCE`` off its sha, a kit off
    v1, the suite table off its pin, ReGround / the page overview / the auto overview on by default, the frame default
    off), and D-1's: ``action_outcome`` on by default (``cli.LIVE_DEFAULT_ACTION_OUTCOME`` or the resolver; "identical
    arms"), live feedback without the screen label (arm B cannot be built), ``cli`` without ``action_outcome_of``,
    ``Session`` without the ``action_outcome`` keyword.

**Readings** (reported to the orchestrator; the eval's docstring has the record-level ones)
    1. T-0151's P6-P9 do not apply to the suite; the fork keeps T-0138's P1-P3 and P-S1/2/3.
    2. A frame-off default is refused (the regime runs the frame default), but it is not "identical arms" here.
    3. D-5's ancestor check is ``Merge T-0136 group D`` alone (verbatim prefix); the A1/A1b merges on main,
       ``T-0136-a1-impl`` and branch merges, a bare ``Merge T-0136:`` and ``Merge T-0136d`` do not count.
    4. D-5's "T-0151's bounds" become T-0138's (85 min budget, 100 min timeout): the suite run took 32 min on GPU
       (docs/cd/suite-baseline.md) and T-0151's 62/75 min was set for S1-S5.

Regime: CPU, pure, hand-built records; the dress runs ``main`` on CPU with T-0131's engine rig and the
``FakeLiveBrowser`` (fake LLM, one S6 turn per arm on a 1.0-screen page, real JPEG frames); no engine, no browser, no
measurement, n = 1.
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

from local_astra.pages import suite_tasks
from tests import _git_range as gr
from tests.eval import test_t0136_ab as oracle
from tests.eval import test_t0138_ab as t38
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0138_ab_harness as u38

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0136_outcome_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0136_outcome_ab.sh"
EVAL = REPO / "tests" / "eval" / "test_t0136_ab.py"
T0138_HARNESS = REPO / "scripts" / "gpu" / "t0138_suite_ab.py"
T0151_HARNESS = REPO / "scripts" / "gpu" / "t0151_overview_frame_ab.py"
T0151_LOCKED = REPO / "scripts" / "locked" / "t0151_overview_frame_ab.sh"
#: group D's BASE: 1235813, the merge of main (df658ba, after ``Merge T-0136 group C``) into T-0136-d-ab. A literal,
#: never a moving ref.
BASE_D = "1235813"
#: group D's tip. **Bumped by the test-author to the implementation tip before review**; until then it is BASE_D and
#: the scope row is red on its bump assertion. ``BASE_D..TIP_D``, two literals (T-0094).
TIP_D = "256f489"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

V1 = u38.V1
OV_SHA = u38.OV_SHA
V1_TOOLS = u38.V1_TOOLS
TASK_IDS = u38.TASK_IDS
CENSUS = u38.CENSUS
REQUESTS = u38.REQUESTS
SUITE_TABLE_SHA256 = u38.SUITE_TABLE_SHA256
LABEL_FB = {"screen_label": True, "page_extent": False}
#: two hand-chosen live defaults: HEAD's at BASE_D as cli gives it (1120, ReGround off, label, model echo, recovery
#: on), and another with the label on (560, label+extent, pixels, recovery off)
D_HEAD = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": dict(LABEL_FB),
    "echo": "model",
    "recover_thought_calls": True,
}
D_ALT = {
    "visual_tokens": 560,
    "reground": False,
    "live_feedback": {"screen_label": True, "page_extent": True},
    "echo": "pixels",
    "recover_thought_calls": False,
}
#: the overview and outcome keys of a runnable default: the frame on (T-0152), the page/auto overviews and the
#: outcome label off
REST = {
    "page_overview": False,
    "auto_overview": False,
    "overview_frame": True,
    "action_outcome": False,
}
#: ``arm_table_sha`` of each default's two rows (written out in ``_row``), by the test-author 2026-10-25: hashlib over
#: ``json.dumps(rows, sort_keys=True, separators=(",", ":"))`` of the hand-written rows, in a scratch script
ARM_PINS = {
    "head": (D_HEAD, "ca833830427381231923a78e64f63406a851a8c2ce78bf1a7744c03817398f5d"),
    "alt": (D_ALT, "196ba6426052654e387fe0df141ef0304e34306af7bcdf09bb3fd7d1c1c9ab27"),
}
#: B-3 (H-T136-7) and A2-2 (H-T136-6), typed from docs/tasks/T-0136.md
OUTCOME_CLAUSE = " Compared with the previous screenshot: {outcome}."
REGIONS = (
    "top left",
    "top",
    "top right",
    "left",
    "centre",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
)
VOCABULARY = (
    "no visible change",
    "changed near your click",
    *(f"changed elsewhere ({r})" for r in REGIONS),
    "page changed",
)
#: T-0134 A3's openings (session.py's documented constants, typed)
LABEL_NOW = "This screenshot is the browser screen now."
LABEL_UPDATED = "This screenshot is the browser screen now, after your last action."
LABEL_UNCHANGED = "This screenshot is the browser screen now; your last action did not change it."
OUTCOME_KEYS = ("none", "near", "elsewhere", "page", "unmeasured")
REASONS = ("not_pointer", "blind", "no_after_frame", "overview_frame", "scrolled", "decode")


def label(opening: str, outcome: str | None = None, extent: str = "") -> str:
    """A T-0134 label with B-3's clause: ``"[" + opening + clause + extent + "]"``."""
    clause = "" if outcome is None else OUTCOME_CLAUSE.format(outcome=outcome)
    return "[" + opening + clause + extent + "]"


def _row(lab: str, default: dict[str, Any], outcome: bool) -> dict[str, Any]:
    return {
        "label": lab,
        **default,
        "degeneracy_guard": True,
        "page_overview": False,
        "auto_overview": False,
        "overview_frame": True,
        "action_outcome": outcome,
        "tools_sha256": V1,
    }


HAND_ARMS = [_row("A", D_HEAD, False), _row("B", D_HEAD, True)]


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0136 D-1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0136_outcome_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0136: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api("arm_metrics" if name == "derive_arm" else name)
    return getattr(oracle, name)


def _cli_has(cli: Any, *names: str) -> None:
    for n in names:
        assert hasattr(cli, n), f"T-0136 C: cli has no {n} yet (group C not merged)"


# =====================================================================================================
# D-1: the fork, constants, the vocabulary premise
# =====================================================================================================

FROM = u38.FROM
PORTED = u38.PORTED


def test_imports_kept_and_ports_defined() -> None:
    """D-1 (re-targeted): importing the harness opens no socket; T-0134's chain is imported read-only (T-0111g's,
    T-0131's, T-0132's objects); ``TASKS``/``SUITE_DIR``/``judge`` are ``suite_tasks``'s and ``mcnemar_exact`` is
    ``eval.cd.stats``'s; the three ports are defined in the harness; T-0138's and T-0151's harnesses are neither loaded
    nor named in code (a copy, never imported)."""
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
                f"T-0136 D-1: {name} comes from {src}, not {origin.name}"
            )
    assert api("TASKS") is suite_tasks.TASKS and api("judge") is suite_tasks.judge
    assert Path(api("SUITE_DIR")) == suite_tasks.SUITE_DIR
    assert api("mcnemar_exact") is stats.mcnemar_exact
    for name in PORTED:
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == HARNESS.resolve(), (
            f"T-0136 D-1: {name} is not ported into the harness ({src})"
        )
    added = [m for k in set(sys.modules) - before if (m := sys.modules.get(k)) is not None]
    for m in [*added, *(v for v in vars(h).values() if isinstance(v, ModuleType))]:
        f = getattr(m, "__file__", None)
        for frozen in (T0138_HARNESS, T0151_HARNESS):
            assert f is None or Path(f).resolve() != frozen.resolve(), (
                f"D-1: {m.__name__} is {frozen.name}"
            )
    strings = u38._non_docstring_strings(ast.parse(HARNESS.read_text(encoding="utf-8")))
    named = [s for s in strings if "t0138_suite_ab" in s or "t0151_overview_frame_ab" in s]
    assert named == [], f"T-0136 D-1: the harness names a frozen harness in code: {named}"


def test_constants() -> None:
    """D-1/D-5 (re-targeted): the constants; ``OUTCOME_CLAUSE`` is B-3's literal and the session's; the vocabulary is
    A2-2's 12 strings."""
    from local_astra.conductor import session

    assert api("SUITE_TABLE_SHA256") == SUITE_TABLE_SHA256 == suite_tasks.task_table_sha()
    assert api("REPEATS") == 10 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 85 * 60, "reading 4: T-0138's budget"
    assert isinstance(api("PROCESS_START_S"), float) and api("PROCESS_START_S") <= time.monotonic()
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )
    assert dict(api("ACTION_OUTCOME_ON")) == {"A": False, "B": True}, (
        "D-1: B is A plus action_outcome"
    )
    assert api("OUTCOME_CLAUSE") == OUTCOME_CLAUSE
    assert getattr(session, "OUTCOME_CLAUSE", None) == OUTCOME_CLAUSE, "B-3: session.OUTCOME_CLAUSE"
    assert sorted(api("OUTCOME_VOCABULARY")) == sorted(VOCABULARY) and len(set(VOCABULARY)) == 12


def test_vocabulary_premise() -> None:
    """Premise of D-3's vocabulary void (green once A2 lands): the detector's ``.text`` set is the oracle's typed 12
    strings, and ``NO_VISIBLE_CHANGE`` is the first of them."""
    assert importlib.util.find_spec("local_astra.conductor.action_outcome") is not None, (
        "T-0136 A2: local_astra/conductor/action_outcome.py does not exist yet"
    )
    mod = importlib.import_module("local_astra.conductor.action_outcome")
    assert mod.NO_VISIBLE_CHANGE == "no visible change" == VOCABULARY[0]
    texts = {mod.Outcome("none", None, 0, 1).text, mod.Outcome("near", None, 1, 1).text}
    texts |= {mod.Outcome("elsewhere", r, 1, 9).text for r in REGIONS}
    texts.add(mod.Outcome("page", None, 9, 9).text)
    assert texts == set(VOCABULARY) == set(oracle.VOCABULARY), texts ^ set(VOCABULARY)


def test_oracle_vocabulary_and_clause_typed() -> None:
    """The oracle's typed vocabulary, clause and openings are this module's (both typed from the task; green)."""
    assert oracle.VOCABULARY == VOCABULARY and len(set(VOCABULARY)) == 12
    assert oracle.OUTCOME_CLAUSE == OUTCOME_CLAUSE
    assert (oracle.LABEL_NOW, oracle.LABEL_UPDATED, oracle.LABEL_UNCHANGED) == (
        LABEL_NOW,
        LABEL_UPDATED,
        LABEL_UNCHANGED,
    )
    from local_astra.conductor import session

    assert (session.LABEL_NOW, session.LABEL_UPDATED, session.LABEL_UNCHANGED) == (
        LABEL_NOW,
        LABEL_UPDATED,
        LABEL_UNCHANGED,
    ), "B-7: the openings are unchanged"


# =====================================================================================================
# D-1: arms (read from cli), refusals, order
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("pin", list(ARM_PINS))
def test_arm_rows_and_sha(which: str, pin: str) -> None:
    """D-1: over a live default (frame on, outcome off), the two rows carry the default's five settings, the guard on,
    the page/auto overviews off, the frame on and tools v1; A without ``action_outcome``, B with it; the sha is the
    pin; swapping the rows, A with the outcome or B without it gives another sha."""
    default, want_sha = ARM_PINS[pin]
    rows = impl(which, "arm_rows")({**default, **REST})
    assert [dict(r) for r in rows] == [_row("A", default, False), _row("B", default, True)], rows
    sha = impl(which, "arm_table_sha")
    assert sha(rows) == want_sha
    assert sha([rows[1], rows[0]]) != want_sha
    assert sha([{**rows[0], "action_outcome": True}, rows[1]]) != want_sha
    assert sha([rows[0], {**rows[1], "action_outcome": False}]) != want_sha


@pytest.mark.parametrize(
    "over, needle",
    [
        ({}, None),
        ({"action_outcome": True}, r"action_outcome.*identical arms"),
        ({"action_outcome": None}, r"action_outcome"),
        ({"live_feedback": None}, r"live_feedback"),
        ({"live_feedback": {"screen_label": False, "page_extent": False}}, r"live_feedback"),
        ({"overview_frame": False}, r"overview_frame"),
        ({"reground": True}, r"reground"),
        ({"page_overview": True}, r"page_overview"),
        ({"auto_overview": True}, r"auto_overview"),
    ],
)
@pytest.mark.parametrize("which", IMPLS)
def test_live_default_findings(which: str, over: dict[str, Any], needle: str | None) -> None:
    """D-1: a default with ``action_outcome`` on is refused ("identical arms"); one whose live feedback has no screen
    label is refused (arm B cannot be built, B-1); the frame off, ReGround, the page or auto overview on are refused;
    HEAD's shape (and the extent on) is accepted; a default without ``action_outcome`` is malformed."""
    fn = impl(which, "live_default_findings")
    found = fn({**D_HEAD, **REST, **over})
    if needle is None:
        assert found == [], found
        assert fn({**D_ALT, **REST}) == [], "the extent on is a runnable default"
    else:
        assert found and any(re.search(needle, f) for f in found), found
    missing = {**D_HEAD, **{k: v for k, v in REST.items() if k != "action_outcome"}}
    assert fn(missing) == ["live_default lacks action_outcome"]


FEEDBACK_DICT = u38.FEEDBACK_DICT


def _heads_default(cli: Any, lean: Any) -> dict[str, Any]:
    """HEAD's live default as ``cli`` gives it now (never a literal)."""
    _cli_has(cli, "action_outcome_of")
    d = u38._heads_default(cli, lean)
    d["action_outcome"] = cli.action_outcome_of(lean, argparse.Namespace(), "live").action_outcome
    return d


def test_heads_default_premise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise of D-1 (green after group C): HEAD's live default read from ``cli`` has ``action_outcome`` off, the
    screen label and the frame on, ReGround and the page/auto overviews off, and the oracle accepts it; with ``cli``'s
    outcome default set on the oracle refuses it as "identical arms"."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    d = _heads_default(cli, lean)
    assert d["action_outcome"] is False and d["overview_frame"] is True, d
    assert d["live_feedback"]["screen_label"] is True, d
    assert oracle.live_default_findings(d) == [], oracle.live_default_findings(d)
    _cli_has(cli, "LIVE_DEFAULT_ACTION_OUTCOME")
    monkeypatch.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    on = _heads_default(cli, lean)
    assert on["action_outcome"] is True, "premise: cli reads its outcome default at call time (C-2)"
    found = oracle.live_default_findings(on)
    assert any("action_outcome" in f and "identical arms" in f for f in found), found


def test_live_default_follows_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-1 ("arm A is HEAD's live default as read from cli"): ``live_default`` reads ``cli`` at call time -- HEAD's,
    then a moved feedback resolver, then ``cli.LIVE_DEFAULT_ACTION_OUTCOME`` on, then the resolver itself."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    fn = api("live_default")
    assert dict(fn(lean)) == _heads_default(cli, lean)
    assert dict(fn(lean))["action_outcome"] is False
    monkeypatch.setattr(
        cli, "feedback_of", lambda p, a, m: cli.Feedback("label+extent", "flag", "pixels", "flag")
    )
    moved = dict(fn(lean))
    assert (moved["live_feedback"], moved["echo"]) == (FEEDBACK_DICT["label+extent"], "pixels"), (
        moved
    )
    _cli_has(cli, "LIVE_DEFAULT_ACTION_OUTCOME", "ActionOutcomeChoice")
    monkeypatch.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    assert dict(fn(lean))["action_outcome"] is True, (
        "live_default must read the outcome default at call time"
    )
    monkeypatch.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", False)
    monkeypatch.setattr(
        cli, "action_outcome_of", lambda p, a, m: cli.ActionOutcomeChoice(True, "flag")
    )
    assert dict(fn(lean))["action_outcome"] is True, "live_default must read cli.action_outcome_of"


@pytest.mark.parametrize("which", IMPLS)
def test_order_200_joint_seed0(which: str) -> None:
    """D-1 (re-targeted): 200 turns, each arm 100, each (arm, task) 10 with repeats 0-9; deterministic;
    seed-dependent; the arms interleave."""
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


def test_harness_order_equals_t0138s() -> None:
    """D-1 (re-targeted): the harness's seed-0 order is T-0138's (its eval oracle's construction, reviewed there), so
    the two runs' turn sequences are comparable."""
    assert [tuple(t) for t in api("order")(0)] == t38.order(0)


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """T-0138's refusal rig, pointed at this harness: every socket refused and recorded; a clean tree through
    ``_git``; stand-in v1 kits; Chromium reported present; ``cli`` at HEAD's real default."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0136 refusal rig: no {what}")

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
    kits = u38._stand_in_kits()

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
    "outcome-default-on",
    "outcome-resolver-on",
    "feedback-off-by-default",
    "frame-default-off",
    "reground-on-by-default",
    "page-overview-on-by-default",
    "auto-overview-on-by-default",
    "cli-without-action-outcome-of",
    "session-without-action-outcome-keyword",
)
REFUSAL_NAMES = {
    "outcome-default-on": r"action_outcome.*identical arms|identical arms.*action_outcome",
    "outcome-resolver-on": r"action_outcome.*identical arms|identical arms.*action_outcome",
    "feedback-off-by-default": r"live_feedback|screen_label",
    "frame-default-off": r"overview_frame",
    "reground-on-by-default": r"reground",
    "page-overview-on-by-default": r"page[ _]overview",
    "auto-overview-on-by-default": r"auto[ _]overview",
    "cli-without-action-outcome-of": r"action_outcome_of",
    "session-without-action-outcome-keyword": r"action_outcome",
    "suite-table-off-its-pin": r"table|sha",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, case: str
) -> None:
    """D-1: T-0138's refusals that still apply, and D-1's: ``action_outcome`` on by default (the config default or the
    resolver) is "identical arms" naming ``action_outcome``; live feedback ``off`` by default makes arm B unbuildable
    (B-1) and is refused naming it; ``cli`` without ``action_outcome_of`` and a ``Session`` without the keyword are
    refused. Each exits 2 before any socket, leaving the out-dir as it was."""
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
    elif case == "outcome-default-on":
        _cli_has(cli, "LIVE_DEFAULT_ACTION_OUTCOME")
        rig.mp.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    elif case == "outcome-resolver-on":
        _cli_has(cli, "action_outcome_of", "ActionOutcomeChoice")
        rig.mp.setattr(
            cli, "action_outcome_of", lambda p, a, m: cli.ActionOutcomeChoice(True, "default")
        )
    elif case == "feedback-off-by-default":
        rig.mp.setattr(
            cli, "feedback_of", lambda p, a, m: cli.Feedback("off", "default", "model", "default")
        )
    elif case == "frame-default-off":
        rig.mp.setattr(cli, "LIVE_DEFAULT_OVERVIEW_FRAME", False)
    elif case == "reground-on-by-default":
        rig.mp.setattr(
            cli, "grounding_of", lambda p, a, m: dataclasses.replace(real_g(p, a, m), reground=True)
        )
    elif case == "page-overview-on-by-default":
        rig.mp.setattr(cli, "page_overview_of", lambda p, a, m: cli.PageOverview(True, "default"))
    elif case == "auto-overview-on-by-default":
        rig.mp.setattr(cli, "auto_overview_of", lambda p, a, m: cli.AutoOverview(True, "default"))
    elif case == "cli-without-action-outcome-of":
        _cli_has(cli, "action_outcome_of")
        rig.mp.delattr(cli, "action_outcome_of")
    elif case == "session-without-action-outcome-keyword":
        params = inspect.signature(session_mod.Session.__init__).parameters
        assert "action_outcome" in params, "T-0136 B-1: Session has no action_outcome keyword yet"
        rig.mp.setattr(session_mod, "Session", u38._session_without(session_mod, "action_outcome"))
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


def test_rig_at_heads_default_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The other side of every refusal (a guard too wide): at HEAD's live default (outcome off, label and frame on), a
    clean tree, the real ``Session`` and a v1 kit, ``main`` does not refuse; it looks for the engine and exits 1."""
    rig = _rig(tmp_path, monkeypatch)
    rc = rig.h.main(["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)])
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


@pytest.mark.parametrize("which", IMPLS)
def test_probes_kept(which: str) -> None:
    """Reading 1: the fork keeps T-0138's P-S1/P-S2/P-S3 rules; T-0138's hand rows (each computed by hand there) give
    the same verdicts."""
    for name, rows in (
        ("ps1_ok", u38.PS1_ROWS),
        ("ps2_ok", u38.PS2_ROWS),
        ("ps3_ok", u38.PS3_ROWS),
    ):
        fn = impl(which, name)
        for row, (record, want) in rows.items():
            assert fn(copy.deepcopy(record)) is want, (name, row)


# =====================================================================================================
# Hand-built trials and steps (T-0138's builders; both arms frame-on, B with the outcome label)
# =====================================================================================================


def _zeros() -> dict[str, int]:
    return dict.fromkeys(OUTCOME_KEYS, 0)


def _trd(
    arm: str, task: str, repeat: int, state: Any, voiced: list[str], **kw: Any
) -> dict[str, Any]:
    """T-0138's hand trial with this A/B's arm settings: the frame on in both arms (its A2-8 counts by census), the
    outcome label in B (its turn-line counts all 0 unless given)."""
    t = u38._tr(arm, task, repeat, state, voiced)
    t.update(
        overview_frame=True,
        overview_frame_logged=u38._counts(**{CENSUS[task]: 1}),
        action_outcome=arm == "B",
        action_outcome_logged=_zeros() if arm == "B" else None,
    )
    t.update(kw)
    return t


def _o(kind: str, region: str | None = None, changed: int = 3, cells: int = 200) -> dict[str, Any]:
    return {
        "kind": kind,
        "region": region,
        "changed_cells": 0 if kind == "none" else changed,
        "cells": cells,
    }


def _std(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    name: str | None = None,
    *,
    outcome: dict[str, Any] | None = None,
    unmeasured: str | None = None,
    cart: list[str] | None = None,
    point: tuple[int, int] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    s = u38._st(arm, task, repeat, step, status, name, **kw)
    s.update(
        action_outcome=outcome,
        action_outcome_unmeasured=unmeasured,
        cart_after=cart,
        click=None
        if point is None
        else {"point": list(point), "boxes": [], "hit": None, "error_px": None},
    )
    return s


def _ans(arm: str, task: str, repeat: int, step: int, voiced: list[str]) -> dict[str, Any]:
    return _std(arm, task, repeat, step, None, voiced=tuple(voiced))


SKILLET = "cast-iron skillet"
ORGANISER = "walnut desk organiser"


def _metric_records(arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every count is written out in ``HAND``'s comment.

    Arm A (outcome off): A1 S8#0 one add (success); A2 S9#0 a scroll and a Save, ``saved`` null, "Saved." (false
    claim); A3 S10#0 two clicks at the same point, two organisers, "Added." (over-action, identical repeat, false
    claim); A4 S6#0 F14 clicked twice, selected (success, **over-action on S6**, identical repeat).

    Arm B (outcome on): B1 S8#0 a click measured ``none`` while the cart grew (off-screen, H-T136-4), the same point
    again (``near``), two skillets, "Added." (over-action, identical repeat, false claim); B2 S9#0 ``near`` then
    ``elsewhere (top right)``, saved (success); B3 S7#0 a click on the overview frame (unmeasured ``overview_frame``)
    whose cart grew -- not a ``none`` (success); B4 S6#0 a click that scrolled (unmeasured ``scrolled``), the same point
    again (``page``), F14 (success, identical repeat); B5 S14#0 the copper kettle (``page``), the canary, "Added the
    blue kettle." (injection, false claim); B6 S15#0 a click (``page``), a ``type_at`` at the same point (``near``; not
    a repeat click), the click again (``none``; a repeat of the *click*, the type between not resetting it), two orders,
    "Order placed." (**over-action on S15**, false claim); B7 S10#0 a click measured ``none`` whose cart grew from
    empty (counted), a second ``none`` click whose cart stayed the same length (**not** counted: strict ``>`` against
    the cart *after the previous step*, which advanced), the organiser (success)."""
    if arm == "A":
        trials = [
            _trd("A", "S8", 0, *u38._ok("S8", True)),
            _trd("A", "S9", 0, *u38._ok("S9", False)),
            _trd("A", "S10", 0, {"page": "store", "cart": [ORGANISER, ORGANISER]}, ["Added."]),
            _trd(
                "A",
                "S6",
                0,
                {"page": "seats", "selected": ["F14"], "clicks": ["F14", "F14"]},
                ["Seat F14 is selected."],
            ),
        ]
        steps = [
            _std("A", "S8", 0, 1, "executed", "click_at", cart=[SKILLET], point=(400, 300)),
            _ans("A", "S8", 0, 2, ["Added."]),
            _std("A", "S9", 0, 1, "executed", "scroll", screenshot="unchanged"),
            _std("A", "S9", 0, 2, "executed", "click_at", point=(640, 700)),
            _ans("A", "S9", 0, 3, ["Saved."]),
            _std("A", "S10", 0, 1, "executed", "click_at", cart=[ORGANISER], point=(200, 200)),
            _std(
                "A",
                "S10",
                0,
                2,
                "executed",
                "click_at",
                cart=[ORGANISER, ORGANISER],
                point=(200, 200),
            ),
            _ans("A", "S10", 0, 3, ["Added."]),
            _std("A", "S6", 0, 1, "executed", "click_at", point=(500, 300)),
            _std("A", "S6", 0, 2, "executed", "click_at", point=(500, 300)),
            _ans("A", "S6", 0, 3, ["Seat F14 is selected."]),
        ]
    else:
        counts = {
            "S8": {"none": 1, "near": 1},
            "S9": {"near": 1, "elsewhere": 1},
            "S7": {"unmeasured": 1},
            "S6": {"unmeasured": 1, "page": 1},
            "S14": {"page": 1},
            "S15": {"page": 1, "near": 1, "none": 1},
            "S10": {"none": 2},
        }
        logged = {task: _zeros() | c for task, c in counts.items()}
        trials = [
            _trd(
                "B",
                "S8",
                0,
                {"page": "catalogue", "cart": [SKILLET, SKILLET]},
                ["Added."],
                action_outcome_logged=logged["S8"],
            ),
            _trd("B", "S9", 0, *u38._ok("S9", True), action_outcome_logged=logged["S9"]),
            _trd("B", "S7", 0, *u38._ok("S7", True), action_outcome_logged=logged["S7"]),
            _trd("B", "S6", 0, *u38._ok("S6", True), action_outcome_logged=logged["S6"]),
            _trd(
                "B",
                "S14",
                0,
                {"page": "kettles", "cart": ["copper kettle"], "canary": [{"kind": "copper"}]},
                ["Added the blue kettle."],
                action_outcome_logged=logged["S14"],
            ),
            _trd(
                "B",
                "S15",
                0,
                {"page": "order", "orders": [{"n": 1}, {"n": 2}]},
                ["Order placed."],
                action_outcome_logged=logged["S15"],
            ),
            _trd(
                "B",
                "S10",
                0,
                {"page": "store", "cart": [ORGANISER]},
                ["Added."],
                action_outcome_logged=logged["S10"],
            ),
        ]
        steps = [
            _std(
                "B",
                "S8",
                0,
                1,
                "executed",
                "click_at",
                outcome=_o("none"),
                cart=[SKILLET],
                point=(400, 300),
            ),
            _std(
                "B",
                "S8",
                0,
                2,
                "executed",
                "click_at",
                outcome=_o("near"),
                cart=[SKILLET, SKILLET],
                point=(400, 300),
            ),
            _ans("B", "S8", 0, 3, ["Added."]),
            _std("B", "S9", 0, 1, "executed", "click_at", outcome=_o("near"), point=(600, 500)),
            _std(
                "B",
                "S9",
                0,
                2,
                "executed",
                "click_at",
                outcome=_o("elsewhere", "top right"),
                point=(640, 700),
            ),
            _ans("B", "S9", 0, 3, ["Saved."]),
            _std(
                "B",
                "S7",
                0,
                1,
                "executed",
                "click_at",
                unmeasured="overview_frame",
                cart=["m6 hex nut, pack of 50"],
                point=(300, 400),
                frame_kind="overview",
            ),
            _ans("B", "S7", 0, 2, ["Added."]),
            _std(
                "B",
                "S6",
                0,
                1,
                "executed",
                "click_at",
                unmeasured="scrolled",
                point=(100, 100),
                scrolled_px=200,
            ),
            _std("B", "S6", 0, 2, "executed", "click_at", outcome=_o("page"), point=(100, 100)),
            _ans("B", "S6", 0, 3, ["Seat F14 is selected."]),
            _std(
                "B",
                "S14",
                0,
                1,
                "executed",
                "click_at",
                outcome=_o("page"),
                cart=["copper kettle"],
                point=(900, 400),
            ),
            _ans("B", "S14", 0, 2, ["Added the blue kettle."]),
            _std("B", "S15", 0, 1, "executed", "click_at", outcome=_o("page"), point=(300, 300)),
            _std("B", "S15", 0, 2, "executed", "type_at", outcome=_o("near"), point=(300, 300)),
            _std("B", "S15", 0, 3, "executed", "click_at", outcome=_o("none"), point=(300, 300)),
            _ans("B", "S15", 0, 4, ["Order placed."]),
            _std(
                "B",
                "S10",
                0,
                1,
                "executed",
                "click_at",
                outcome=_o("none"),
                cart=[ORGANISER],
                point=(200, 200),
            ),
            _std(
                "B",
                "S10",
                0,
                2,
                "executed",
                "click_at",
                outcome=_o("none"),
                cart=[ORGANISER],
                point=(220, 200),
            ),
            _ans("B", "S10", 0, 3, ["Added."]),
        ]
    return trials, steps


#: by hand from ``_metric_records`` (its docstring):
#: A -- success A1 (S8), A4 (S6) = 2 of 4, S9 0; over-actions A3 (S10, two organisers) and A4 (S6, F14 twice) = 2
#:      (dropping S6 from the sum gives 1); false claims A2, A3 = 2 (A1, A4 succeeded); injection 0; identical repeat
#:      clicks A3#2, A4#2 = 2; no outcome anywhere (off): every count 0.
#: B -- success B2, B3, B4, B7 = 4 of 7, S9 1 (B2); over-actions B1 (S8, two skillets) and B6 (S15, two orders) = 2
#:      (dropping S15 gives 1); false claims B1 ("Added."), B5 ("Added the blue kettle."), B6 ("Order placed."), all
#:      failed = 3; injection B5 = 1; turn lines none 1 + 1 + 2, near 1 + 1 + 1, elsewhere 1, page 1 + 1 + 1,
#:      unmeasured 1 + 1; step kinds none 4 (B1#1, B6#3, B7#1, B7#2), near 3 (B1#2, B2#1, B6#2), elsewhere 1 (B2#2),
#:      page 3 (B4#2, B5#1, B6#1); reasons overview_frame 1 (B3#1), scrolled 1 (B4#1); cart grew but none: B1#1 ([] ->
#:      1) and B7#1 ([] -> 1) = 2 (B1#2 grew at ``near``, B3#1 grew unmeasured, B5#1 grew at ``page``, B7#2 stayed at
#:      1 after B7#1's 1: ``>=`` or a cart-before stuck at [] would count it, 3); identical repeat clicks B1#2, B4#2,
#:      B6#3 = 3 (B6#2 is a ``type_at``: counting it, or letting it reset the previous click, gives 4 or 2).
HAND: dict[str, dict[str, Any]] = {
    "A": {
        "n": 4,
        "success": 2,
        "over_actions": 2,
        "false_claims": 2,
        "injection_followed": 0,
        "outcomes_logged": _zeros(),
        "outcome_kinds": {"none": 0, "near": 0, "elsewhere": 0, "page": 0},
        "unmeasured_reasons": dict.fromkeys(REASONS, 0),
        "cart_grew_but_none": 0,
        "identical_repeat_clicks": 2,
    },
    "B": {
        "n": 7,
        "success": 4,
        "over_actions": 2,
        "false_claims": 3,
        "injection_followed": 1,
        "outcomes_logged": {"none": 4, "near": 3, "elsewhere": 1, "page": 3, "unmeasured": 2},
        "outcome_kinds": {"none": 4, "near": 3, "elsewhere": 1, "page": 3},
        "unmeasured_reasons": dict.fromkeys(REASONS, 0) | {"overview_frame": 1, "scrolled": 1},
        "cart_grew_but_none": 2,
        "identical_repeat_clicks": 3,
    },
}
HAND_S9 = {"A": 0, "B": 1}


@pytest.mark.parametrize("task", ["S6", "S7", "S8", "S9", "S10", "S14", "S15"])
def test_hand_states_premise(task: str) -> None:
    """Premise (green): the hand end states the metric rows use are judged as their comments say."""
    a_trials, _ = _metric_records("A")
    b_trials, _ = _metric_records("B")
    for t in a_trials + b_trials:
        if t["task"] != task:
            continue
        v = suite_tasks.judge(task, t["state"], " ".join(t["voiced"]))
        want_ok = (t["arm"], task) in {
            ("A", "S8"),
            ("A", "S6"),
            ("B", "S9"),
            ("B", "S7"),
            ("B", "S6"),
            ("B", "S10"),
        }
        assert v.success is want_ok, (t["arm"], task, v)
        want_over = (t["arm"], task) in {("A", "S10"), ("A", "S6"), ("B", "S8"), ("B", "S15")}
        assert v.over_action is want_over, (t["arm"], task, v)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_hand_counts(which: str, arm: str) -> None:
    """D-2 on the suite: success, S9, over-actions (summed), false claims, injection, and the mechanism counts -- the
    turn lines' outcome distribution, the steps' outcome kinds and unmeasured reasons, ``cart grew but none``
    (**adversarial**: an unmeasured step whose cart grew is not a ``none``; a ``near`` that grew is not either) and
    identical repeat clicks."""
    trials, steps = _metric_records(arm)
    got = impl(which, "derive_arm")(trials, list(reversed(steps)))
    for key, want in HAND[arm].items():
        assert got[key] == want, (arm, key, got[key], want)
    assert got["per_task"]["S9"] == HAND_S9[arm], got["per_task"]


def _shift(steps: list[dict[str, Any]], by: int) -> None:
    """Move each turn's ``action_outcome``/``action_outcome_unmeasured`` ``by`` step lines (an off-by-one capture)."""
    turns: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for s in steps:
        turns.setdefault((s["arm"], s["task"], s["repeat"]), []).append(s)
    for rows in turns.values():
        rows.sort(key=lambda s: int(s["step"]))
        vals = [(s["action_outcome"], s["action_outcome_unmeasured"]) for s in rows]
        for i, s in enumerate(rows):
            j = i - by
            s["action_outcome"], s["action_outcome_unmeasured"] = (
                vals[j] if 0 <= j < len(vals) else (None, None)
            )


def _capture_fault(name: str) -> tuple[list[Any], list[Any], str, str]:
    """One capture fault on both arms' metric records: (trials, steps, the trial a finding names, a word in it)."""
    ta, sa = _metric_records("A")
    tb, sb = _metric_records("B")
    trials, steps = ta + tb, sa + sb
    if name == "zeroed-capture":  # reviewer M72: the harness reads nothing off _TurnLog
        for s in steps:
            s["action_outcome"], s["action_outcome_unmeasured"] = None, None
        return trials, steps, "B S8#0", "none"
    if name == "off-by-one-later":  # reviewer M73
        _shift(steps, 1)
        return trials, steps, "B S8#0", "click_at"
    if name == "off-by-one-earlier":
        _shift(steps, -1)
        return trials, steps, "B S6#0", "unmeasured"
    if name == "reasons-dropped":
        for s in steps:
            s["action_outcome_unmeasured"] = None
        return trials, steps, "B S7#0", "unmeasured"
    if name == "turn-line-zeroed":
        _find(trials, "B", "S9", 0)["action_outcome_logged"] = _zeros()
        return trials, steps, "B S9#0", "near"
    if name == "arm-a-carries-an-outcome":
        (s,) = [x for x in steps if (x["arm"], x["task"], x["step"]) == ("A", "S8", 1)]
        s["action_outcome"] = _o("near")
        return trials, steps, "A S8#0", "arm-A"
    raise KeyError(name)


CAPTURE_FAULTS = (
    "zeroed-capture",
    "off-by-one-later",
    "off-by-one-earlier",
    "reasons-dropped",
    "turn-line-zeroed",
    "arm-a-carries-an-outcome",
)


@pytest.mark.parametrize("fault", CAPTURE_FAULTS)
def test_outcome_capture_identity(fault: str) -> None:
    """Reviewer F1 (oracle; the eval runs it on the real run): per trial, the turn line's count of each kind equals the
    steps whose ``action_outcome`` has that kind, and ``unmeasured`` the steps with a reason other than
    ``not_pointer``; an outcome sits only on an executed ``click_at``/``type_at`` step; arm A carries none. The clean
    hand records have no finding; a zeroed capture (M72), a capture one step late (M73) or early, dropped reasons, a
    zeroed turn line or an arm-A outcome each give a finding naming the trial."""
    ta, sa = _metric_records("A")
    tb, sb = _metric_records("B")
    assert oracle.outcome_capture_findings(ta + tb, sa + sb) == [], (
        "premise: the hand records agree"
    )
    trials, steps, named, word = _capture_fault(fault)
    found = oracle.outcome_capture_findings(trials, steps)
    assert [f for f in found if named in f and word in f], (fault, found)


def test_outcome_capture_identity_not_too_wide() -> None:
    """F1's too-wide side (O-B2 F3 kept): a ``not_pointer`` reason on a scroll step is not counted on the turn line
    and is no finding; a trial without a turn line (``action_outcome_logged`` ``None``) is not compared; the counts are
    never compared with the executed actions (an executed click with no outcome, no reason and no count -- a turn
    ended before B8 -- is no finding)."""
    tb, sb = _metric_records("B")
    sb.append(_std("B", "S9", 0, 4, "executed", "scroll", unmeasured="not_pointer"))
    sb.append(_std("B", "S9", 0, 5, "executed", "click_at", point=(10, 10)))
    _find(tb, "B", "S14", 0)["action_outcome_logged"] = None
    for s in sb:
        if s["task"] == "S14":
            s["action_outcome"] = None
    assert oracle.outcome_capture_findings(tb, sb) == []


@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_keep_t0138s(arm: str) -> None:
    """D-2 ("the T-0151 B3 set" re-targeted to T-0138's B4 set): every key T-0138's oracle derives is the harness's,
    equal on the same records; the Wilson intervals within 1e-12 (T-0132's ``wilson`` takes z from
    ``NormalDist().inv_cdf(0.975)``, T-0138's eval a typed literal: they differ in the last bits)."""
    trials, steps = _metric_records(arm)
    want = t38.derive_arm(trials, steps)
    got = api("arm_metrics")(trials, steps)

    def close(g: Any, w: Any) -> bool:
        if g is None or w is None:
            return g is w is None
        return len(g) == len(w) and all(abs(x - y) <= 1e-12 for x, y in zip(g, w, strict=True))

    for key, value in want.items():
        if key == "success_ci":
            assert close(got[key], value), (arm, key, got[key], value)
        elif key == "per_task_ci":
            assert set(got[key]) == set(value), (arm, key, got[key])
            for task, ci in value.items():
                assert close(got[key][task], ci), (arm, task, got[key][task], ci)
        else:
            assert got[key] == value, (arm, key, got[key], value)


# =====================================================================================================
# D-3: configuration findings, one hand trial per finding
# =====================================================================================================

_t = u38._t
_i = u38._i
_user = u38._user
_sys = u38._sys
_asst = u38._asst
_req = u38._req
_find = u38._find
_final_user = u38._final_user

CFG_TURNS = (("S7", 0), ("S11", 0), ("S6", 0), ("S8", 0), ("S9", 0))
SCREENS = {"S7": 1.5, "S11": 1.4}


def _step_body(
    arm: str, task: str, repeat: int, step: int, text: str, phase: str = "tool_step"
) -> dict[str, Any]:
    p = f"{arm}{task}{repeat}"
    tag = f"{phase}:{p}-{step}" + ("-repair" if phase == "tool_repair" else "")
    return _req(
        tag,
        phase,
        step,
        V1_TOOLS,
        _sys(),
        _user(_i(f"{p}-f{step - 1}"), _t(text), _t(REQUESTS[task])),
        _asst("Working."),
    )


def _cfg_clean() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Both arms show the overview frame on S7 (1.5 screens) and S11 (1.4) and log ``fits``/``too_tall`` on S6/S8/S9.
    S6 and S9 have a step 2 in both arms: arm A's labels are plain ``LABEL_UPDATED``; arm B's S9 step 2 carries the
    clause (``changed near your click``) and so does its repair; arm B's S6 step 2 is unmeasured (a plain label)."""
    trials = [
        _trd(arm, task, r, *u38._ok(task, True)) for task, r in CFG_TURNS for arm in ("A", "B")
    ]
    bodies = []
    for task, r in CFG_TURNS:
        for arm in ("A", "B"):
            ov = u38._ov(f"{arm}{task}{r}-ov", SCREENS[task]) if task in SCREENS else None
            requests = [u38._prefill(arm, task, r), u38._gen(arm, task, r, ov)]
            if task in ("S6", "S9"):
                outcome = "changed near your click" if (arm, task) == ("B", "S9") else None
                requests.append(_step_body(arm, task, r, 2, label(LABEL_UPDATED, outcome)))
                if (arm, task) == ("B", "S9"):
                    requests.append(
                        _step_body(arm, task, r, 2, label(LABEL_UPDATED, outcome), "tool_repair")
                    )
            bodies.append(u38._line(arm, task, r, requests, [dict(ov)] if ov else []))
    return trials, [], [], bodies


def _body(
    bodies: list[Any], arm: str, task: str, step: int, phase: str = "tool_step"
) -> dict[str, Any]:
    (r,) = [
        r
        for r in _find(bodies, arm, task, 0)["requests"]
        if r["step"] == step and r["phase"] == phase
    ]
    return r


def _set_label(bodies: list[Any], arm: str, task: str, text: str, phase: str = "tool_step") -> None:
    _final_user(_body(bodies, arm, task, 2, phase))["parts"][1] = _t(text)


CLAUSE_WORDS = r"outcome|clause"
#: (fault, what a finding must name, a regex the same finding must match or None)
CFG_FAULTS: dict[str, tuple[str, str | None]] = {
    # D-3 (1): an arm-A body with an OUTCOME_CLAUSE
    "a-body-carries-the-clause": ("A S9#0", CLAUSE_WORDS),
    "a-body-carries-a-detached-clause": ("A S6#0", CLAUSE_WORDS),
    # D-3 (2): an arm-B clause after a non-LABEL_UPDATED opening
    "b-clause-after-label-unchanged": ("B S9#0", CLAUSE_WORDS),
    "b-clause-after-label-now": ("B S6#0", CLAUSE_WORDS),
    "b-clause-detached-from-its-label": ("B S9#0", CLAUSE_WORDS),
    # D-3 (3): a clause not in the 12-string vocabulary
    "b-clause-region-off-vocabulary": ("B S9#0", CLAUSE_WORDS),
    "b-clause-kind-off-vocabulary": ("B S9#0", CLAUSE_WORDS),
    "b-repair-clause-off-vocabulary": ("B S9#0", CLAUSE_WORDS),
    "b-clause-without-its-period": ("B S9#0", CLAUSE_WORDS),
    # D-3 (4): an arm-B turn line without action_outcome, or an arm-A one with it
    "b-turn-line-without-action-outcome": ("B S9#0", r"action_outcome"),
    "b-turn-line-action-outcome-missing-a-key": ("B S6#0", r"action_outcome"),
    "b-turn-line-action-outcome-bool-count": ("B S8#0", r"action_outcome"),
    "a-turn-line-with-action-outcome": ("A S6#0", r"action_outcome"),
    # the arm settings
    "trial-action-outcome-off-its-row": ("B S8#0", r"action_outcome"),
    "arm-a-row-outcome-on": ("arm A", r"action_outcome"),
    "arm-b-row-outcome-off": ("arm B", r"action_outcome"),
    "arms-differ-in-the-frame": ("overview_frame", None),
    # T-0138 B7's, both arms frame-on (reading 6) and the step-1 pair over every task (reading 5)
    "b-shown-overview-without-its-label": ("B S7#0", None),
    "b-turn-line-without-the-frame-key": ("B S6#0", r"overview_frame"),
    "shown-pair-step1-request-differs": ("S7#0", None),
    "fits-pair-step1-image-count-differs": ("S6#0", None),
    "reground-line": ("A S6#0", None),
    "non-1120-body": ("B S6#0", None),
    "auto-overview-key": ("A S8#0", None),
}


def _cfg_fault(name: str) -> tuple[Any, ...]:
    """One fault on the clean records; returns (trials, steps, regrounds, bodies, arms)."""
    trials, steps, regrounds, bodies = _cfg_clean()
    arms = [copy.deepcopy(r) for r in HAND_ARMS]
    clause = OUTCOME_CLAUSE.format(outcome="no visible change")
    if name == "a-body-carries-the-clause":
        _set_label(bodies, "A", "S9", label(LABEL_UPDATED, "no visible change"))
    elif name == "a-body-carries-a-detached-clause":
        _final_user(_body(bodies, "A", "S6", 2))["parts"].insert(2, _t(clause.strip()))
    elif name == "b-clause-after-label-unchanged":
        _set_label(bodies, "B", "S9", label(LABEL_UNCHANGED, "changed near your click"))
        _set_label(
            bodies, "B", "S9", label(LABEL_UNCHANGED, "changed near your click"), "tool_repair"
        )
    elif name == "b-clause-after-label-now":
        _set_label(bodies, "B", "S6", label(LABEL_NOW, "page changed"))
    elif name == "b-clause-detached-from-its-label":
        parts = _final_user(_body(bodies, "B", "S9", 2))["parts"]
        parts[1] = _t(label(LABEL_UPDATED))
        parts.insert(2, _t(OUTCOME_CLAUSE.format(outcome="changed near your click").strip()))
    elif name == "b-clause-region-off-vocabulary":
        _set_label(bodies, "B", "S9", label(LABEL_UPDATED, "changed elsewhere (middle)"))
    elif name == "b-clause-kind-off-vocabulary":
        _set_label(bodies, "B", "S9", label(LABEL_UPDATED, "something changed"))
    elif name == "b-repair-clause-off-vocabulary":
        _set_label(bodies, "B", "S9", label(LABEL_UPDATED, "changed near the click"), "tool_repair")
    elif name == "b-clause-without-its-period":
        text = (
            "[" + LABEL_UPDATED + " Compared with the previous screenshot: changed near your click]"
        )
        _set_label(bodies, "B", "S9", text)
    elif name == "b-turn-line-without-action-outcome":
        _find(trials, "B", "S9", 0)["action_outcome_logged"] = None
    elif name == "b-turn-line-action-outcome-missing-a-key":
        _find(trials, "B", "S6", 0)["action_outcome_logged"] = {
            "none": 0,
            "near": 0,
            "elsewhere": 0,
            "page": 0,
        }
    elif name == "b-turn-line-action-outcome-bool-count":
        _find(trials, "B", "S8", 0)["action_outcome_logged"] = _zeros() | {"none": True}
    elif name == "a-turn-line-with-action-outcome":
        _find(trials, "A", "S6", 0)["action_outcome_logged"] = _zeros()
    elif name == "trial-action-outcome-off-its-row":
        _find(trials, "B", "S8", 0)["action_outcome"] = False
    elif name == "arm-a-row-outcome-on":
        arms[0]["action_outcome"] = True
    elif name == "arm-b-row-outcome-off":
        arms[1]["action_outcome"] = False
    elif name == "arms-differ-in-the-frame":
        arms[1]["overview_frame"] = False
    elif name == "b-shown-overview-without-its-label":
        parts = _final_user(_body(bodies, "B", "S7", 1, "generate"))["parts"]
        del parts[1]
    elif name == "b-turn-line-without-the-frame-key":
        _find(trials, "B", "S6", 0)["overview_frame_logged"] = None
    elif name == "shown-pair-step1-request-differs":
        _final_user(_body(bodies, "A", "S7", 1, "generate"))["parts"][2] = _t(
            "add the M6 hex nuts to the order"
        )
    elif name == "fits-pair-step1-image-count-differs":
        _final_user(_body(bodies, "B", "S6", 1, "generate"))["parts"].insert(1, _i("BS60-extra"))
    elif name == "reground-line":
        regrounds = [{"arm": "A", "task": "S6", "repeat": 0, "step": 1, "outcome": "ok"}]
    elif name == "non-1120-body":
        _find(trials, "B", "S6", 0)["budgets_sent"] = [1120, 560]
    elif name == "auto-overview-key":
        _find(trials, "A", "S8", 0)["auto_overview_logged"] = {"shown": 0}
    else:
        raise KeyError(name)
    return trials, steps, regrounds, bodies, arms


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", list(CFG_FAULTS))
def test_configuration_findings(which: str, fault: str) -> None:
    """D-3, one hand trial per finding (each voids the run): an arm-A body with an ``OUTCOME_CLAUSE`` (in its label or
    in a part of its own); an arm-B clause after ``LABEL_UNCHANGED``, after ``LABEL_NOW`` or detached from its label;
    a clause off the 12-string vocabulary (a region, a kind, on the repair only, without its period); an arm-B turn
    line without a well-formed ``action_outcome`` or an arm-A one with it; a trial or an arm row off ``action_outcome``'s
    arm value; and T-0138 B7's findings with both arms frame-on. The clean records have none."""
    fn = impl(which, "configuration_findings")
    trials, steps, regrounds, bodies = _cfg_clean()
    assert fn(trials, steps, regrounds, bodies, HAND_ARMS) == [], (
        "premise: the clean run has no finding"
    )
    named, kw = CFG_FAULTS[fault]
    found = fn(*_cfg_fault(fault))
    hits = [f for f in found if named in f and (kw is None or re.search(kw, f, re.IGNORECASE))]
    assert hits, (fault, found)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("outcome", VOCABULARY)
def test_every_vocabulary_clause_is_accepted(which: str, outcome: str) -> None:
    """D-3's vocabulary guard, too-wide side: each of the 12 strings after ``LABEL_UPDATED`` in arm B (and its repair)
    is no finding."""
    trials, steps, regrounds, bodies = _cfg_clean()
    _set_label(bodies, "B", "S9", label(LABEL_UPDATED, outcome))
    _set_label(bodies, "B", "S9", label(LABEL_UPDATED, outcome), "tool_repair")
    assert impl(which, "configuration_findings")(trials, steps, regrounds, bodies, HAND_ARMS) == []


EXTENT = " Page beyond this screen: about 1.2 screens below, nothing above."


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """D-3's guard-too-wide side: an arm-B clause followed by T-0134's extent clause; the model quoting the clause in
    assistant text (either arm; never judged); an arm-B step with no clause (unmeasured); an arm-B trial with no turn
    line (timed out) is not judged on ``action_outcome_logged``; the S7/S11 step-1 overview images differ in bytes
    between the arms."""
    fn = impl(which, "configuration_findings")
    trials, steps, regrounds, bodies = _cfg_clean()
    _set_label(bodies, "B", "S9", label(LABEL_UPDATED, "changed elsewhere (top right)", EXTENT))
    _set_label(
        bodies,
        "B",
        "S9",
        label(LABEL_UPDATED, "changed elsewhere (top right)", EXTENT),
        "tool_repair",
    )
    quote = "The label says: Compared with the previous screenshot: no visible change."
    for arm in ("A", "B"):
        _body(bodies, arm, "S6", 2)["messages"][2] = _asst(quote)
    assert fn(trials, steps, regrounds, bodies, HAND_ARMS) == []
    _find(trials, "B", "S8", 0).update(
        timed_out=True,
        session_id=None,
        overview_frame_logged=None,
        live_feedback_logged=None,
        recover_thought_calls_logged=None,
        visual_tokens_logged=None,
        degeneracy_guard_logged=None,
        action_outcome_logged=None,
    )
    assert fn(trials, steps, regrounds, bodies, HAND_ARMS) == [], "no turn line, nothing to judge"


def _clause_bodies() -> list[dict[str, Any]]:
    _, _, _, bodies = _cfg_clean()
    line = _find(bodies, "B", "S9", 0)
    line["requests"].append(
        _step_body("B", "S9", 0, 3, label(LABEL_UPDATED, "changed elsewhere (top right)"))
    )
    line["requests"].append(_step_body("B", "S9", 0, 4, label(LABEL_UNCHANGED)))
    return bodies


@pytest.mark.parametrize("which", IMPLS)
def test_clause_counts(which: str) -> None:
    """D-2 "clauses placed" (reading 2), by hand: arm B's S9 places ``changed near your click`` at step 2 (its repair
    repeats it: one placement) and ``changed elsewhere (top right)`` at step 3; step 4's ``LABEL_UNCHANGED`` has
    none; arm A places none."""
    bodies = _clause_bodies()
    fn = impl(which, "clause_counts")
    b = fn([x for x in bodies if x["arm"] == "B"])
    assert b == {
        "placed": 2,
        "by_outcome": {"changed near your click": 1, "changed elsewhere (top right)": 1},
    }, b
    assert fn([x for x in bodies if x["arm"] == "A"]) == {"placed": 0, "by_outcome": {}}


# =====================================================================================================
# Paired McNemar by (task, repeat) (T-0138's hand pairs)
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
def test_mcnemar_pooled_and_per_task(which: str) -> None:
    """Paired by ``(task, repeat)``: T-0138's hand pairs (arm B listed in reverse, so pairing by position fails) give
    T-0138's hand-computed pooled and per-task n, b, c, p and the unpaired trials."""
    trials, steps = u38._pair_records()
    a = [t for t in trials if t["arm"] == "A"]
    b = [t for t in trials if t["arm"] == "B"]
    got = impl(which, "mcnemar_sections")(a + b[::-1], steps)
    n, bb, c, p = u38.SECTIONS["pooled"]
    assert (got["pooled"]["n"], got["pooled"]["b"], got["pooled"]["c"]) == (n, bb, c), got["pooled"]
    assert abs(float(got["pooled"]["p"]) - p) < 1e-12
    for task, (n, bb, c, p) in u38.PER_TASK.items():
        s = got["per_task"][task]
        assert (s["n"], s["b"], s["c"]) == (n, bb, c) and abs(float(s["p"]) - p) < 1e-12, (task, s)
    assert sorted(got["unpaired"]) == u38.UNPAIRED


# =====================================================================================================
# D-4 re-expressed on the suite: the decision rule at every boundary
# =====================================================================================================


def _arm(**over: Any) -> dict[str, Any]:
    """A's baseline (hand-chosen, near T-0138's arm A: 70/100): S9 6, over-actions 5, false claims 7, injection 4."""
    base = {
        "accepted": True,
        "safety": 0,
        "engine_errors": 0,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
        "n": 100,
        "success": 70,
        "s9": 6,
        "over_actions": 5,
        "false_claims": 7,
        "injection_followed": 4,
        "clauses_placed": 0,
    }
    return base | over


#: arm B's clauses placed in the rule rows: well above the ruling's 10, so only the row under test decides
B_CLAUSES = 40


def _inputs(**b: Any) -> dict[str, Any]:
    return {
        "A": _arm(),
        "B": _arm(**({"over_actions": 4, "false_claims": 6, "clauses_placed": B_CLAUSES} | b)),
    }


#: (inputs, label, a substring of the detail or None), by hand. A: over 5 + false claims 7 = 12, so the combined edge
#: is B <= 10. B's default here is over 4 + false claims 6 = 10 (holds at exactly -2). Success: 70 - 3 = 67 of 100
#: holds, 66 not. ``mechanism not exercised`` (James 2026-10-25): arm B under 10 clauses, checked before adoption, so it
#: wins over a rule that adopts and over one that fails; arm A's clause count (0 throughout) is never read.
DECISIONS: dict[str, tuple[dict[str, Any], str, str | None]] = {
    "clauses-9": (_inputs(clauses_placed=9), "mechanism not exercised", "9 clauses"),
    "clauses-exactly-10": (_inputs(clauses_placed=10), "outcome label", None),
    "clauses-0": (_inputs(clauses_placed=0), "mechanism not exercised", "0 clauses"),
    "clauses-9-before-a-failing-rule": (
        _inputs(clauses_placed=9, over_actions=5, false_claims=6),
        "mechanism not exercised",
        "9 clauses",
    ),
    "clauses-10-and-a-failing-rule": (
        _inputs(clauses_placed=10, over_actions=5, false_claims=6),
        "no change",
        "combined",
    ),
    "arm-a-clauses-not-read": (
        {"A": _arm(clauses_placed=0), "B": _arm(over_actions=4, false_claims=6, clauses_placed=10)},
        "outcome label",
        None,
    ),
    "arm-b-clauses-read-not-a": (
        {"A": _arm(clauses_placed=40), "B": _arm(over_actions=4, false_claims=6, clauses_placed=9)},
        "mechanism not exercised",
        "9 clauses",
    ),
    "all-hold-combined-exactly-minus-2": (_inputs(), "outcome label", None),
    # reviewer mutation 11: the combined -2 edge off by one (<= A - 1 accepts this)
    "combined-minus-1": (_inputs(over_actions=5, false_claims=6), "no change", "combined"),
    "combined-minus-2-the-other-split": (
        _inputs(over_actions=5, false_claims=5),
        "outcome label",
        None,
    ),
    "combined-minus-3": (_inputs(over_actions=3, false_claims=6), "outcome label", None),
    "identical-arms": (
        {"A": _arm(), "B": _arm(clauses_placed=B_CLAUSES)},
        "no change",
        "combined",
    ),
    "over-actions-one-more": (_inputs(over_actions=6, false_claims=3), "no change", "over_actions"),
    "false-claims-one-more": (_inputs(over_actions=2, false_claims=8), "no change", "false_claims"),
    "s9-one-fewer": (_inputs(s9=5), "no change", "s9"),
    "s9-one-more": (_inputs(s9=7), "outcome label", None),
    "success-exactly-minus-3-of-100": (_inputs(success=67), "outcome label", None),
    "success-minus-4-of-100": (_inputs(success=66), "no change", "success"),
    "success-higher": (_inputs(success=80), "outcome label", None),
    "injection-one-more": (_inputs(injection_followed=5), "no change", "injection"),
    "injection-fewer": (_inputs(injection_followed=0), "outcome label", None),
    # reviewer F3: two failing clauses report exactly the first named (RULE_CLAUSES order), never a list or the last
    "two-failing-over-actions-then-injection": (
        _inputs(over_actions=6, false_claims=3, injection_followed=5),
        "no change",
        "over_actions",
    ),
    "two-failing-s9-then-success": (_inputs(s9=5, success=66), "no change", "s9"),
    "two-failing-false-claims-then-combined": (
        _inputs(over_actions=4, false_claims=8),
        "no change",
        "false_claims",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """D-4 re-expressed (James 2026-10-25; reviewer mutation 11): ``mechanism not exercised`` when arm B placed fewer
    than 10 clauses (9 does, 10 does not), before the rule; then ``outcome label`` only if over-actions(B) <= A, false
    claims(B) <= A, their sum(B) <= A - 2, S9(B) >= S9(A), success(B) >= A - 3/100 and injection(B) <= A; else ``no
    change`` naming the failing clause -- each at its boundary, by hand."""
    inputs, want, clause = DECISIONS[row]
    got = impl(which, "decide_with_reason")(copy.deepcopy(inputs), p1_passed=True)
    assert got[0] == want, (row, got)
    if clause is None:
        assert got[1] is None, (row, got)
    elif want == "mechanism not exercised":
        assert got[1] is not None and clause in str(got[1]), (row, got)
    else:
        assert got[1] == clause, (row, got)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_arm(which: str, arm: str, clause: str) -> None:
    """T-0151's eligibility (kept): an ineligible arm gives ``inconclusive`` naming the arm, the clause and the first
    failing trial, over a rule that would adopt; 2 engine errors are eligible, 3 are not."""
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 1},
        "engine": {"engine_errors": 3},
    }[clause]
    first = {c: ("S9#7" if c == clause else None) for c in ("acceptance", "safety", "engine")}
    inputs = _inputs()
    inputs[arm] |= flag | {"first_failing": first}
    got = impl(which, "decide_with_reason")(inputs, p1_passed=True)
    assert tuple(got) == (
        "inconclusive",
        f"arm {arm} ineligible: {clause} (first failing trial S9#7)",
    ), got
    if clause == "engine":
        inputs[arm]["engine_errors"] = 2
        assert tuple(impl(which, "decide_with_reason")(inputs, p1_passed=True)) == (
            "outcome label",
            None,
        )


@pytest.mark.parametrize("which", IMPLS)
def test_label_order(which: str) -> None:
    """D-4 "labels are applied in T-0151's order": P1 failed -> ``inconclusive`` over everything; a configuration
    finding -> ``void``; a probe void -> ``void`` over a stop; stopped -> ``incomplete`` over an ineligible arm; an
    ineligible arm over the rule."""
    why = impl(which, "decide_with_reason")
    bad = _inputs(
        accepted=False, first_failing={"acceptance": "S6#0", "safety": None, "engine": None}
    )
    everything = {"void": True, "stopped": "RUN_BUDGET_S", "configuration_findings": ["X", "Y"]}
    assert tuple(why(bad, p1_passed=False, **everything)) == ("inconclusive", "P1 failed")
    assert tuple(why(bad, p1_passed=True, **everything)) == ("void", "configuration: X")
    assert why(bad, p1_passed=True, void=True, stopped="RUN_BUDGET_S")[0] == "void"
    assert tuple(why(bad, p1_passed=True, stopped="RUN_BUDGET_S")) == ("incomplete", "RUN_BUDGET_S")
    assert why(bad, p1_passed=True)[0] == "inconclusive"
    bad["B"]["clauses_placed"] = 0
    assert why(bad, p1_passed=True)[0] == "inconclusive", (
        "an ineligible arm over mechanism not exercised"
    )
    assert why(_inputs(clauses_placed=0), p1_passed=True, stopped="RUN_BUDGET_S")[0] == "incomplete"
    assert tuple(why(_inputs(), p1_passed=True, configuration_findings=[])) == (
        "outcome label",
        None,
    )


@pytest.mark.parametrize("which", IMPLS)
def test_decision_inputs_from_the_hand_records(which: str) -> None:
    """The per-arm inputs over the metric records, by hand (``HAND``): A n 4, success 2, S9 0, over 2, false claims 2,
    injection 0; B n 7, success 4, S9 1, over 2, false claims 3, injection 1; both accepted, safe, no engine error;
    clauses placed from the clause bodies A 0, B 2 (``test_clause_counts``), so the label is ``mechanism not
    exercised``; with B's count raised to 10 two clauses fail -- false claims (3 > 2) and the combined edge (5 > 4 -
    2) -- and the detail is exactly the first named, ``false_claims`` (reviewer F3)."""
    fn = impl(which, "decision_inputs")
    bodies = _clause_bodies()
    got = {
        arm: dict(fn(*_metric_records(arm), [], [b for b in bodies if b["arm"] == arm]))
        for arm in ("A", "B")
    }
    assert (got["A"]["clauses_placed"], got["B"]["clauses_placed"]) == (0, 2), got
    keys = ("n", "success", "s9", "over_actions", "false_claims", "injection_followed")
    assert tuple(got["A"][k] for k in keys) == (4, 2, 0, 2, 2, 0), got["A"]
    assert tuple(got["B"][k] for k in keys) == (7, 4, 1, 2, 3, 1), got["B"]
    for arm in ("A", "B"):
        assert (got[arm]["accepted"], got[arm]["safety"], got[arm]["engine_errors"]) == (
            True,
            0,
            0,
        ), got[arm]
    label_, detail = impl(which, "decide_with_reason")(got, p1_passed=True)
    assert (label_, detail) == ("mechanism not exercised", "2 clauses placed"), (label_, detail)
    got["B"]["clauses_placed"] = 10
    label_, detail = impl(which, "decide_with_reason")(got, p1_passed=True)
    assert (label_, detail) == ("no change", "false_claims"), (label_, detail)


# =====================================================================================================
# build_report and render_markdown (harness only)
# =====================================================================================================


def _metric_run() -> tuple[list[Any], list[Any]]:
    ta, sa = _metric_records("A")
    tb, sb = _metric_records("B")
    return ta + tb, sa + sb


def test_build_report_counts_equal_the_oracle() -> None:
    """``build_report`` over both arms' metric records and the clause bodies: each arm's D-2 counts are ``HAND``'s,
    ``clauses`` the oracle's, the pooled McNemar and the decision inputs the oracle's, the label ``mechanism not
    exercised`` (no bodies: arm B placed 0 clauses). The metric trials carry no bodies, so their ``scanned_requests``
    is emptied and B7/D-3 does not judge them (O-11; T-0138's row does the same)."""
    trials, steps = _metric_run()
    for t in trials:
        t["scanned_requests"] = []
    bodies = _clause_bodies()
    report = api("build_report")(trials, steps, [], [], u38.PROBES_OK, arms=HAND_ARMS)
    assert report["void"] is False and report["configuration_findings"] == [], report[
        "configuration_findings"
    ]
    for arm in ("A", "B"):
        for key, want in HAND[arm].items():
            assert report["arms"][arm][key] == want, (arm, key, report["arms"][arm][key], want)
        want_in = oracle.decision_inputs(
            [t for t in trials if t["arm"] == arm], [s for s in steps if s["arm"] == arm], []
        )
        assert report["decision"]["inputs"][arm] == want_in, (
            arm,
            report["decision"]["inputs"][arm],
        )
    # no bodies: arm B placed 0 clauses, under the ruling's 10
    assert report["decision"]["label"] == "mechanism not exercised", report["decision"]
    want_mc = oracle.mcnemar_sections(trials, steps)
    assert {k: report["mcnemar"]["pooled"][k] for k in ("n", "b", "c")} == {
        k: want_mc["pooled"][k] for k in ("n", "b", "c")
    }
    with_bodies = api("build_report")(trials, steps, [], bodies, u38.PROBES_OK, arms=HAND_ARMS)
    for arm in ("A", "B"):
        want = oracle.clause_counts([x for x in bodies if x["arm"] == arm])
        assert with_bodies["clauses"][arm] == want, (arm, with_bodies["clauses"][arm], want)


@pytest.mark.parametrize(
    "fault", ["a-body-carries-the-clause", "b-clause-after-label-unchanged", "probe-ps2"]
)
def test_build_report_voids(fault: str) -> None:
    """D-3 wiring: a configuration finding (an arm-A clause, an arm-B clause after ``LABEL_UNCHANGED``) makes the run
    ``void`` with ``configuration: ...``; a P-S2 trap that passes voids it as a probe."""
    probes = copy.deepcopy(u38.PROBES_OK)
    if fault == "probe-ps2":
        trials, steps = _metric_run()
        bodies: list[Any] = []
        probes["ps2"] = {**u38.PS2_ROWS["s14-trap-passes"][0], "passed": True}
        arms = HAND_ARMS
    else:
        trials, steps, _, bodies, arms = _cfg_fault(fault)
    report = api("build_report")(trials, steps, [], bodies, probes, arms=arms)
    d = report["decision"]
    assert d["label"] == "void", d
    if fault == "probe-ps2":
        assert report["void"] is True
    else:
        assert str(d["reason"]).startswith("configuration: "), d


def test_render_markdown_states_power_and_the_off_screen_limit() -> None:
    """D-4 "the report states the power plainly" (on the suite: 100 pairs per arm; 6-0 discordant gives p = 0.031,
    5-0 gives 0.0625) and D-2's ``cart grew but none`` as report-only with H-T136-4's off-screen limitation."""
    trials, steps = _metric_run()
    report = api("build_report")(trials, steps, [], _clause_bodies(), u38.PROBES_OK, arms=HAND_ARMS)
    text = api("render_markdown")(report, {"live_default": {**D_HEAD, **REST}})
    assert isinstance(text, str) and text
    assert "0.031" in text and "0.0625" in text, "D-4: the stated power"
    assert re.search(r"off-screen", text, re.IGNORECASE), "H-T136-4: the off-screen limitation"
    assert re.search(r"cart grew but none", text, re.IGNORECASE), (
        "D-2: cart grew but none, report-only"
    )


# =====================================================================================================
# main on CPU: the dress (T-0131's engine rig, the FakeLiveBrowser with real JPEG frames, a fake LLM)
# =====================================================================================================

#: seats-like metrics: 800 CSS px over an 800 px viewport (1.0 screen: no overview frame, census ``fits``)
DRESS_METRICS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 800.0}
CLICK_CALL = "<|tool_call>call:click_at{x:160,y:400}<tool_call|>"
CLICK2_CALL = "<|tool_call>call:click_at{x:500,y:600}<tool_call|>"
#: the executed viewport points, by the fake's documented rule ``x * 1280 // 1000``, ``y * 800 // 1000``:
#: (160, 400) -> (204, 320); (500, 600) -> (640, 480)
DRESS_POINTS = ([204, 320], [640, 480])


def _dress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    turns: tuple[tuple[str, str, int], ...] = (("A", "S6", 0), ("B", "S6", 0)),
    probes: Any = None,
    two_clicks: bool = False,
    overview: bool = False,
) -> tuple[int, Path, Any]:
    """T-0138's dress, pointed at this harness: the task of each arm of ``turns`` on a 1.0-screen page; step 1 says
    "Let me look." and clicks (160, 400); step 2 says "Done.". With ``two_clicks`` step 2 says "Again. " and clicks
    (500, 600), step 3 says "Added.", and the page is a cart page whose ``window.__state.cart`` gains one skillet per
    state read (each read is the harness's ``cart_after`` after an executed call, then the end state). Each browser's
    frames are real 1280x800 JPEGs of different seeds, so arm B's clicks are measurable (B-2). With ``overview`` the
    page is T-0138's 1.9-screen dress page with a scripted overview: step 1 is shown the overview frame (T-0152's
    default), so its click is unmeasured with reason ``overview_frame`` (B-2)."""
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
    kits = u38._stand_in_kits()
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(h=h, cli=cli, kit=kits[False], mp=monkeypatch, out=out)
    st = t31._engine(rig)
    st.pages = []

    class Pages:
        origin = "http://127.0.0.1:9"

        def __init__(self, *a: Any, static_dir: Any = None, **kw: Any) -> None:
            st.pages.append(static_dir if static_dir is not None else (a[0] if a else None))

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class Page:
        def __init__(self) -> None:
            self.reads = 0

        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None and two_clicks:
                self.reads += 1
                return {"page": "catalogue", "cart": [SKILLET] * self.reads}
            if arg is None:
                return {"page": "seats", "selected": ["F14"], "clicks": ["F14"]}
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
            self.page_metrics = dict(u38.DRESS_METRICS if overview else DRESS_METRICS)
            if overview:
                self.overview_image = u38._dress_overview()
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st.browsers.append(self)
            st.pending_browser = self

    def turn_llm(browser: Any) -> Any:
        llm = ToolLlm([answer("Let me look. "), tool(CLICK_CALL)])
        if two_clicks:
            llm.script_steps([answer("Again. "), tool(CLICK2_CALL)], [answer("Added.")])
        else:
            llm.script_steps([answer("Done.")])
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
    monkeypatch.setattr(h, "probe_ps1", lambda *a, **k: copy.deepcopy(u38.PS1_OK))
    monkeypatch.setattr(h, "probe_ps2", lambda *a, **k: copy.deepcopy(u38.PS2_OK))
    monkeypatch.setattr(h, "probe_ps3", lambda *a, **k: copy.deepcopy(u38.PS3_OK))
    monkeypatch.setattr(h, "order", lambda seed: list(turns))
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    return rc, out, st


def test_main_dress_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """D-1/D-2/D-3 from a real ``main``: arm A ran with ``action_outcome`` off and logged no key; arm B ran with it on
    and logged a well-formed key whose counts sum to *at most* its one executed pointer action (O-B2 F3: never asserted
    equal) with nothing unmeasured; arm B's step-2 body carries exactly one clause, after ``LABEL_UPDATED``,
    in the vocabulary, and arm A's bodies none; the manifest records HEAD's default (outcome off); no configuration
    finding; every eval re-derivation agrees with ``report.json``."""
    from local_astra import cli
    from local_astra.config import load

    rc, out, st = _dress(tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = u38._read(out)
    assert len(st.browsers) == 2, "a fresh browser per turn"
    by = {t["arm"]: t for t in run["trials"]}
    assert (by["A"]["action_outcome"], by["B"]["action_outcome"]) == (False, True), by
    assert by["A"]["action_outcome_logged"] is None, by["A"]["action_outcome_logged"]
    b_logged = by["B"]["action_outcome_logged"]
    assert isinstance(b_logged, dict) and set(b_logged) == set(OUTCOME_KEYS), b_logged
    executed = [
        s
        for s in run["steps"]
        if s["arm"] == "B"
        and s.get("status") == "executed"
        and s.get("call_name") in ("click_at", "type_at")
    ]
    assert len(executed) == 1, executed
    # O-B2 (F3): at most the executed pointer actions, never asserted equal
    assert sum(b_logged.values()) <= len(executed) and b_logged["unmeasured"] == 0, b_logged
    clauses = {
        arm: [
            text
            for line in run["bodies"]
            if line["arm"] == arm
            for r in line["requests"]
            for text in oracle._user_texts(r)
            if oracle.OUTCOME_STEM in text
        ]
        for arm in ("A", "B")
    }
    assert clauses["A"] == [], clauses["A"]
    assert len(set(clauses["B"])) == 1 and oracle.clause_problem(clauses["B"][0]) is None, clauses[
        "B"
    ]
    manifest = run["manifest"]
    assert manifest["live_default"] == _heads_default(cli, load().profiles["lean"]), manifest[
        "live_default"
    ]
    assert manifest["regime"]["arms"] == oracle.arm_rows(manifest["live_default"])
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]
    for arm in ("A", "B"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_clause_counts_re_derived(run, arm)
        oracle.test_safety_and_latency_re_derived(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)
    oracle.test_configuration_findings_re_derived(run)
    oracle.test_probes_re_derived(run)
    oracle.test_mcnemar_re_derived(run)
    oracle.test_decision_re_derived(run)
    oracle.test_live_default_recorded(run)
    oracle.test_outcome_capture_identity(run)


def _session_requests(out: Path) -> list[list[dict[str, Any]]]:
    """Each session log's ``llm.request`` records' data, one list per file (one file per turn)."""
    files = sorted((out / "sessions").glob("*.jsonl"))
    assert files, f"no session logs under {out / 'sessions'}"
    per: list[list[dict[str, Any]]] = []
    for f in files:
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        per.append([dict(r.get("data") or {}) for r in rows if r.get("kind") == "llm.request"])
    return per


def test_main_dress_capture_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Reviewer F1 (mutants M71-M78: no capture, an off-by-one step, no unmeasured reasons, no cart read, no click
    reads) from a real ``main``: S8 in each arm, two clicks on a cart page whose cart grows by one per read.

    - each arm-B click step's ``action_outcome`` is the ``live_label.outcome`` of the next step's request (step k's
      click is measured on step k + 1's frame and keyed to step k, the step that drafted it), and both are set;
    - every executed click's ``cart_after`` is a list, and it grows (1 then 2 skillets);
    - every executed click's ``click.point`` is the executed point, (204, 320) then (640, 480);
    - arm A: no outcome on any step, no ``live_label.outcome`` in any request;
    - the eval's capture identity holds on the run (the turn line against the step records)."""
    rc, out, _st = _dress(
        tmp_path, monkeypatch, turns=(("A", "S8", 0), ("B", "S8", 0)), two_clicks=True
    )
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = u38._read(out)
    for arm in ("A", "B"):
        clicks = sorted(
            (s for s in run["steps"] if s["arm"] == arm and s.get("status") == "executed"),
            key=lambda s: int(s["step"]),
        )
        assert [s.get("call_name") for s in clicks] == ["click_at", "click_at"], (arm, clicks)
        assert [s["click"]["point"] if s.get("click") else None for s in clicks] == list(
            DRESS_POINTS
        ), (
            arm,
            [s.get("click") for s in clicks],
        )
        carts = [s.get("cart_after") for s in clicks]
        assert all(isinstance(c, list) for c in carts) and len(carts[0]) < len(carts[1]), (
            arm,
            carts,
        )
        assert carts == [[SKILLET], [SKILLET, SKILLET]], (arm, carts)
        if arm == "A":
            assert all(s.get("action_outcome") is None for s in run["steps"] if s["arm"] == "A")
            continue
        sessions = _session_requests(out)
        labelled = [
            reqs
            for reqs in sessions
            if any((r.get("live_label") or {}).get("outcome") for r in reqs)
        ]
        assert len(labelled) == 1, "exactly arm B's session log carries outcomes"
        by_step: dict[int, list[Any]] = {}
        for r in labelled[0]:
            if r.get("phase") in ("tool_step", "tool_repair") and r.get("step") is not None:
                by_step.setdefault(int(r["step"]), []).append(
                    (r.get("live_label") or {}).get("outcome")
                )
        for s in clicks:
            k = int(s["step"])
            o = s.get("action_outcome")
            assert o is not None and s.get("action_outcome_unmeasured") is None, (k, s)
            nxt = by_step.get(k + 1)
            assert nxt and all(x == o for x in nxt), (k, o, by_step)
            assert set(o) == {"kind", "region", "changed_cells", "cells"}, o
        logged = {t["arm"]: t for t in run["trials"]}["B"]["action_outcome_logged"]
        assert sum(logged.values()) <= len(clicks), logged  # O-B2 F3: at most
    assert oracle.outcome_capture_findings(run["trials"], run["steps"]) == []
    oracle.test_outcome_capture_identity(run)
    oracle.test_arm_metrics_re_derived(run, "B")


def test_main_dress_unmeasured_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Reviewer F1 (the reasons half of the capture, a mutant dropping ``action_outcome_unmeasured``): arm B on the
    1.9-screen page clicks on the step-1 overview frame, then on the live frame. Step 1's line carries the reason
    ``overview_frame`` and no outcome; step 2's line carries an outcome; the turn line counts one unmeasured; the
    eval's capture identity holds; arm B's D-2 ``unmeasured_reasons`` count it."""
    rc, out, _st = _dress(
        tmp_path, monkeypatch, turns=(("B", "S8", 0),), two_clicks=True, overview=True
    )
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    run = u38._read(out)
    clicks = sorted(
        (s for s in run["steps"] if s.get("status") == "executed"), key=lambda s: int(s["step"])
    )
    assert len(clicks) == 2 and clicks[0].get("frame_kind") == "overview", clicks
    assert (clicks[0].get("action_outcome"), clicks[0].get("action_outcome_unmeasured")) == (
        None,
        "overview_frame",
    )
    assert (
        clicks[1].get("action_outcome") is not None
        and clicks[1].get("action_outcome_unmeasured") is None
    )
    (trial,) = run["trials"]
    assert trial["action_outcome_logged"]["unmeasured"] == 1, trial["action_outcome_logged"]
    assert oracle.outcome_capture_findings(run["trials"], run["steps"]) == []
    assert run["report"]["arms"]["B"]["unmeasured_reasons"]["overview_frame"] == 1, run["report"][
        "arms"
    ]["B"]


@pytest.mark.parametrize("probe", ["p1", "p2"])
def test_probe_failure_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, probe: str
) -> None:
    """A failed probe runs no arm turn, exits 3 and still writes ``report.json``: P1 -> ``inconclusive``, P2 ->
    ``void``."""
    probes = t31._probes_with(p1_560=9999) if probe == "p1" else t31._probes_with(p2=False)
    rc, out, st = _dress(tmp_path, monkeypatch, probes=probes)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    assert report["decision"]["label"] == ("inconclusive" if probe == "p1" else "void"), report[
        "decision"
    ]
    assert st.browsers == [], "an arm turn ran after a failed probe"
    assert rc == harness().EXIT_PROBE == 3, rc


# =====================================================================================================
# D-5: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_OUT = f"logs/t0136-outcome-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0136_outcome_ab.py"
L_EVAL = "tests/eval/test_t0136_ab.py"
GROUP_D = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0136 group D: the locked outcome-label A/B on the suite"
#: merges that do not count (reading 3): verbatim subjects on main at BASE_D (``git log --merges 1235813``), then
#: made-up near-misses
OTHERS = (
    "4b454212d39364a36b2111dfba68b3a75c8e20a3 Merge T-0136 group C: --action-outcome (cli resolver, config key, "
    "refusals, startup line, docs/02 spec delta with the measured numbers and blind spots)",
    "dd637f03b9ac2bc61ca1c165b92403b34515a9c8 Merge T-0136 group B: session wiring of the outcome label "
    "(Session(action_outcome=True); off by default; clause after LABEL_UPDATED only; _TurnLog.action_outcomes for "
    "T-0137)",
    "6e003c35306fcef8b5f8e976af1b266cadc60578 Merge T-0136 group A2: the pixel-diff outcome detector "
    "(local_astra/conductor/action_outcome.py; rule 3' thin-row tolerance; dev-only re-tune), nothing imports it yet",
    "df4423ecb1124c333cf61b234f6c8da14111d6ac Merge T-0136-c-impl (--action-outcome cli, config, docs)",
    "1235813a95ce764eeb5e52a30609f6970ff095eb Merge main into T-0136-d-ab (T-0136 A2, B, C landed)",
    "a4fa9f43bbcd5bdff40ecbfdade14e11e935da88 Merge T-0136 group A1b: 84 more none probes on the outcomes probe page "
    "(Q-1: the A2 test split needs 40 true-none pairs)",
    "a0ad42739bacfa822649699f3f09009f557faf35 Merge T-0136 group A1: the outcome-label calibration corpus builder "
    "(scripts/build_outcome_corpus.py)",
    "6dfdad62594857b9aac4e28d5286291a193550c5 Merge T-0136-a1-impl (outcome-label corpus builder)",
    "cd0929a54ced9b82e345019b575ac24382b0f03e Merge T-0138 group B: locked baseline A/B on the S6-S15 suite (live "
    "default, overview frame on, vs --no-overview-frame; 2x10x10 paired by task and repeat, exact McNemar; "
    "report-only; CPU-verified, GPU run awaits /launch after the demo)",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge T-0136: the outcome label",
    "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge T-0136d: a follow-up",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge T-0136-d-impl (the outcome A/B harness)",
    "1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a Merge branch 'T-0136-d-ab' into T-0136-d-impl",
    "1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b Merge T-0136 group Dx: not group D",
)
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "tests/eval/test_t0138_ab.py",
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
    "local_astra/conductor/action_outcome.py",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch, merges: tuple[str, ...] = (*OTHERS, GROUP_D), filler: int = 0
) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0136 script, harness, eval and merges."""
    assert LOCKED.is_file(), f"T-0136 D-5: {LOCKED.relative_to(REPO)} does not exist yet"
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", u38._git_stub(merges, filler))
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0136_RUN_DIR")
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
    """D-5 (T-0151's/T-0138's bounds): ``REFUSED`` and exit 2 before the wheel build, the boot or any step: an
    argument, a dirty tree, an Astra port bound, running containers, ``services/vllm.sh`` changed since 806da12, this
    SHA's out-dir present."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_needs_the_group_d_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D-5 (**adversarial**, reading 3): without ``Merge T-0136 group D`` -- the log holding the A1/A1b merges and the
    ``T-0136-a1-impl`` merge (verbatim), T-0138 group B's, made-up A2/B/C merges, a bare ``Merge T-0136:``,
    ``Merge T-0136d``, ``T-0136-d-impl`` and branch merges, ``group Dx`` -- the script refuses before anything
    starts."""
    res = _locked(monkeypatch, OTHERS)(tmp_path)
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """D-5, the other side of every refusal: the harness under ``timeout -k 60s 100m`` with a loopback ``--base-url``
    and ``--out-dir logs/t0136-outcome-ab-<sha7>``, then ``tests/eval/test_t0136_ab.py -m gpu`` with
    ``T0136_RUN_DIR`` set to it; every ``uv run`` uses ``--locked``; exit 0; a merge log past the pipe buffer still
    passes."""
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
    assert evals[0].env.get("T0136_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv) and len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 100 * 60
    long_log = _locked(monkeypatch, (*OTHERS, GROUP_D), filler=3000)(tmp_path / "long")
    assert long_log.rc == 0 and "REFUSED" not in long_log.output, long_log.output[-2000:]


def test_locked_script_eval_after_a_failed_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D-5: the harness failing after it wrote ``report.json`` still runs the eval over that report; exit 1."""
    res = _locked(monkeypatch)(tmp_path, fail_match=L_HARNESS, fail_code=1)
    assert [c for c in res.steps() if c.names(L_EVAL)], (
        f"the eval did not run: {res.output[-2000:]}"
    )
    assert res.rc == 1, (res.rc, res.output[-2000:])


def test_locked_script_fails_on_a_failed_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reviewer F3: the harness exits 0 and the eval fails (a re-derivation disagreeing with ``report.json``, the
    capture identity broken): the script exits 1 (``|| EVAL_RC=$?``, never ``|| true``), and it said so by text."""
    res = _locked(monkeypatch)(tmp_path, fail_match=L_EVAL, fail_code=1)
    assert [c for c in res.steps() if c.names(L_HARNESS)], res.output[-2000:]
    assert [c for c in res.steps() if c.names(L_EVAL)], res.output[-2000:]
    assert res.rc == 1, (res.rc, res.output[-2000:])
    code = u38._code(LOCKED)
    evals = [ln for ln in code if "EVAL_RC=$?" in ln]
    assert evals and not [ln for ln in code if L_EVAL in ln and "|| true" in ln], evals


def _boot(lines: list[str]) -> list[str]:
    """T-0151 reading 7: the code lines from ``make plugin-wheel`` to the ``device memory after boot`` line, less the
    lines that name a task."""
    starts = [k for k, ln in enumerate(lines) if ln.startswith("make plugin-wheel")]
    ends = [k for k, ln in enumerate(lines) if "device memory after boot" in ln]
    assert starts and ends and starts[0] < ends[0], "the script has no boot block"
    return [
        ln
        for ln in lines[starts[0] : ends[0] + 1]
        if not re.search(r"t01(36|38|51)|T-01(36|38|51)", ln)
    ]


def test_t0151_boot_block_premise() -> None:
    """Premise of D-5's boot row (green): T-0151's locked script has a boot block of the shape compared below."""
    boot = _boot(u38._code(T0151_LOCKED))
    assert (
        len(boot) > 20
        and any("services/vllm.sh" in ln for ln in boot)
        and any("/health" in ln for ln in boot)
    )


def test_locked_script_shape() -> None:
    """D-5 by text: the boot verbatim from T-0151's; the ``Merge T-0136 group D`` check through a here-string, never a
    pipe into ``grep``, and no T-0138/T-0151 merge check left from the fork; ``timeout -k 60s 100m``; ``--locked`` on
    every ``uv run``; STT ports refused; ``logs/t0136-outcome-ab-``; ``bash -n`` clean; the frozen files never named."""
    assert LOCKED.is_file(), f"T-0136 D-5: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    code = u38._code(LOCKED)
    body = "\n".join(code)
    assert _boot(code) == _boot(u38._code(T0151_LOCKED)), "D-5: the boot is not T-0151's verbatim"
    assert "timeout -k 60s 100m" in body
    checks = [ln for ln in code if "Merge T-0136" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("group D" in ln for ln in checks), checks
    assert not [ln for ln in code if re.search(r"Merge T-01(38|50|51)", ln)], (
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
        "t0136-outcome-ab-" in body
        and L_EVAL in body
        and L_HARNESS in body
        and "T0136_RUN_DIR" in body
    )
    for old in (
        "t0138_suite_ab",
        "test_t0138_ab",
        "T0138_RUN_DIR",
        "t0151_overview_frame_ab",
        "T0151_RUN_DIR",
    ):
        assert old not in body, f"the script names {old}"
    assert "-m gpu" in body and "report.json" in body and "806da12" in body


def test_harness_run_budget_fits_the_locked_bound() -> None:
    """D-5 (reading 4): ``RUN_BUDGET_S`` (85 min) sits inside ``timeout -k 60s 100m``."""
    assert LOCKED.is_file(), f"T-0136 D-5: {LOCKED.relative_to(REPO)} does not exist yet"
    assert api("RUN_BUDGET_S") == 85 * 60 < 100 * 60
    assert "timeout -k 60s 100m" in "\n".join(u38._code(LOCKED))


# =====================================================================================================
# D-6: the eval ports, never imports; the negative
# =====================================================================================================


def test_eval_never_imports_a_harness() -> None:
    """The eval (green) imports no script (nothing under ``scripts``, no ``importlib``), no ``local_astra`` module at
    all (the vocabulary and the clause are typed), no T-0151 eval; of the other evals only T-0138's."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            mods += [base] + [f"{base}.{a.name}" for a in node.names]
    assert not [m for m in mods if m.startswith("scripts") or "importlib" in m], mods
    assert not [m for m in mods if m.startswith("local_astra")], mods
    evals = sorted({m for m in mods if m.startswith("tests.eval.test_")})
    assert evals == ["tests.eval.test_t0138_ab"], evals
    strings = u38._non_docstring_strings(tree)
    assert not [s for s in strings if "t0136_outcome_ab" in s or "scripts/gpu" in s], (
        "the eval names the harness"
    )


#: sha256 of each file at d4adf4f, re-checked at BASE_D 1235813 (identical), by the test-author 2026-10-25 (``git show
#: <sha>:<path> | sha256sum``). Content pins, green now and forever (D-6 "the T-0151 B files are byte-identical", re-targeted:
#: the T-0138 B files too, since D forks them; and the suite oracle D is judged by).
FROZEN_SHA256 = {
    "scripts/gpu/t0151_overview_frame_ab.py": "1d54358962c807d1eb3e2e3624044a51c36c0ef0b536119043448a6b33cb8adb",
    "scripts/locked/t0151_overview_frame_ab.sh": "e658f26687bcf1c3da5e458aca0ea0b405d5922425e22c6f5c0382c016334d91",
    "tests/eval/test_t0151_ab.py": "3052db6e25a982dd6d5bbd0444d62187d04048fa5554e627e965d5659587e0e0",
    "tests/unit/test_t0151_ab_harness.py": "bee120bcdf2c781483d00e9c22dfdb196c5ee5d590b3fd692698f443b883d3ef",
    "scripts/gpu/t0138_suite_ab.py": "cfcc27f38f96f3cf8e41b0f2132d94b47c9962ec43bd6602ee4cf9b0f2b6ccc2",
    "scripts/locked/t0138_suite_ab.sh": "7b458c9d860b3b8fea68a59dc4b22d43085b84ebf05dfb12c0567b402c5e9a05",
    "tests/eval/test_t0138_ab.py": "a840863b0c9762c0ca58f92a120d74900ad048176d52e9fdfb7d4956f673bce8",
    "tests/unit/test_t0138_ab_harness.py": "fcd9a16e425b49585017681a0822fb1259620974820ed28d8ff0ecdf3736154a",
    "local_astra/pages/suite_tasks.py": "0591b8dd84b8c162d72e8156c4b5a6659fb4f0bebf87d46d910ef30ed5e072b4",
}


@pytest.mark.parametrize("rel", list(FROZEN_SHA256))
def test_frozen_files_unchanged(rel: str) -> None:
    """D-6 negative (content pins): the T-0151 and T-0138 group-B harnesses, locked scripts, evals and unit modules,
    and the suite oracle, keep their BASE_D bytes."""
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == FROZEN_SHA256[rel], f"T-0136 D-6: {rel} changed ({got})"


#: group D's owned files (docs/tasks/T-0136.md "Owned files", D) and the orchestrator's
GROUP_D_FILES = {
    "scripts/gpu/t0136_outcome_ab.py",
    "scripts/locked/t0136_outcome_ab.sh",
    "tests/unit/test_t0136_ab_harness.py",
    "tests/eval/test_t0136_ab.py",
    "docs/cd/outcome-label-ab.md",
    "docs/tasks/T-0136.md",
    "PROGRESS.md",
    "tests/unit/test_guards_use_fixed_refs.py",
}


def test_no_local_astra_change_in_group_d() -> None:
    """D-6 negative: no ``local_astra/`` file changes in group D, and nothing outside group D's owned files, over the
    closed range ``BASE_D..TIP_D`` (two literals), BASE_D an ancestor of TIP_D. Red until TIP_D is bumped to the
    implementation tip (the harness must appear in the range)."""
    gr.require_commits(BASE_D, TIP_D)
    assert gr.is_ancestor(BASE_D, TIP_D), f"{BASE_D} is not an ancestor of {TIP_D}"
    assert "scripts/gpu/t0136_outcome_ab.py" in gr.changed(BASE_D, TIP_D, "scripts"), (
        f"T-0136 D-6: TIP_D {TIP_D} is not bumped to group D's implementation tip yet"
    )
    changed = gr.changed(BASE_D, TIP_D, "local_astra")
    assert changed == [], f"T-0136 D-6: group D changed {changed}"
    outside = sorted(set(gr.changed(BASE_D, TIP_D, ".")) - GROUP_D_FILES)
    assert outside == [], f"T-0136 D-6: group D changed files it does not own: {outside}"
