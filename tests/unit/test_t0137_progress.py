"""T-0137 group A (A-1 to A-7): the pure progress rule ``local_astra/conductor/progress.py`` (docs/tasks/T-0137.md).

**The interface these rows pin** (A-1, A-2; H-T137-1 as ruled: tolerance 10 Chebyshev, 2 no-change repeats, 3 visits):

- constants ``REPEAT_TOLERANCE = 10``, ``MAX_NO_CHANGE_REPEATS = 2``, ``MAX_SCREEN_VISITS = 3``,
  ``ACTIONS = frozenset({"click_at", "type_at", "scroll", "press_key"})`` and the two reason strings, verbatim;
- ``Step`` and ``Verdict``, frozen dataclasses with (at least) the A-2 fields. The rows build a ``Step`` with exactly
  the nine A-2 fields as keywords, so a field the implementer adds must have a default (reading R-1 below);
- ``judge(history, name, arguments, screen, *, overview) -> Verdict``.

**Readings** (flagged in the test-author's report, not resolved here):

- **R-1:** extra ``Step`` fields (A-2 "may add fields") must have defaults, or every row breaks on construction.
- **R-2:** A-3's "same action" is exercised through ``judge`` only (a one-step executed history; same iff
  ``repeat_of`` is that step). A-3 names ``same_action`` but gives no signature, so no row calls it directly.
- **R-3:** "latest" in A-4 is history order. The rows keep step numbers increasing, except the Hypothesis
  insertion row (ii), whose inserted ``google_search`` steps carry fresh numbers above the rest, which no rule
  reading may pick as an instance anyway (A-3: never the same action).
- **R-4:** A-5 counts the screens of *every* history step whatever its status (refused and failed included), as
  written; ``test_recurring_by_visits`` pins that.
- **R-5:** A-7 "no file other than progress.py changes in group A" is checked as: over ``BASE_A..TIP_A`` the only
  changed paths are progress.py, this module, ``docs/tasks/T-0137.md`` and ``PROGRESS.md`` (the orchestrator's).

Activated for ``/implement T-0137`` group A (the readings are ruled as O-A in docs/tasks/T-0137.md, 57988e7): every
row fails at BASE_A 72b7a7b on a named ``AssertionError`` from ``prog()`` (the module does not exist), never on an
import error at collection. ``test_frozen_paths_over_closed_range`` is a byte-identity row and is green at BASE
(``TIP_A`` starts at BASE_A and is bumped to the implementation tip by whoever merges).

Adversarial rows (reviewer mutations): 1 ``test_original_not_counted``/``test_stop_on_fourth``; 2
``test_repeat_of_latest_executed_only``; 3 ``test_stop_on_fourth``; 4 ``test_form_fill_never_recurs``/
``test_recurring_by_visits``; 10 and 13 ``test_same_action_table``. O-A2 (reviewer F1, 2048441: the trailing no-change
run, not a cumulative count) ``test_no_change_run_resets`` and the O-A2 bound in ``test_properties``.

Regime: CPU, pure, hand-built ``Step``s and Hypothesis (the suite's profile); n = 1, deterministic.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.conductor.progress"
MODULE_PATH = REPO / "local_astra" / "conductor" / "progress.py"
REL_MODULE = "local_astra/conductor/progress.py"

#: T-0137 group A's BASE: main at 72b7a7b, the branch point of ``T-0137-a-progress-rule``. A literal.
BASE_A = "72b7a7b4d3b2a3416e2e79d2ba93af8b7858646d"
#: The implementation tip: 89fdb4b on T-0137-a-fix (the O-A2 count; merged as c2c5291). Was f08d2d6 (1f9894a), BASE_A.
TIP_A = "89fdb4b69b6517f139be0c079048978febc9144c"
#: Power for the frozen-path predicate: T-0142's range, known to change ``local_astra/`` (degeneracy.py and others).
POWER_BASE = "e1ae76e36f70aa5404be7dbb14566296ac3fbc5a"
POWER_TIP = "712fea3c45eb2244688aa0ee426e324e225ec5cb"

REPEAT_REASON = (
    "the same action did not change this screen the last two times; it was not done again"
)
RECURRING_REASON = (
    "this screen has come back for the third time in this request; the action was not done"
)


# --------------------------------------------------------------------------------------------- seams


def prog() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name not in (MODULE, "local_astra.conductor"):
            raise
        raise AssertionError(
            f"T-0137 A-1: {MODULE} (Step, Verdict, judge, REPEAT_TOLERANCE, MAX_NO_CHANGE_REPEATS, "
            "MAX_SCREEN_VISITS, ACTIONS, the two reasons) does not exist yet"
        ) from exc


def S(
    step: int,
    name: str,
    arguments: Mapping[str, Any],
    screen: str | None,
    status: str = "executed",
    *,
    overview: bool = False,
    no_change: bool | None = False,
    action: str | None = None,
) -> Any:
    """A hand-built ``progress.Step`` with the nine A-2 fields."""
    cls = getattr(prog(), "Step", None)
    assert isinstance(cls, type), "T-0137 A-2: progress.Step does not exist yet"
    outcome = (
        {"screenshot": "unchanged" if no_change else "updated"} if no_change is not None else {}
    )
    return cls(
        step=step,
        name=name,
        arguments=dict(arguments),
        screen=screen,
        status=status,
        overview=overview,
        no_change=no_change,
        outcome=outcome,
        action=action if action is not None else f"{name} {sorted(arguments.items())}",
    )


def judge(
    history: list[Any],
    name: str,
    arguments: Mapping[str, Any],
    screen: str | None,
    *,
    overview: bool = False,
) -> Any:
    fn = getattr(prog(), "judge", None)
    assert callable(fn), "T-0137 A-2: progress.judge does not exist yet"
    return fn(history, name, dict(arguments), screen, overview=overview)


def vt(v: Any) -> tuple[Any, Any, Any, Any]:
    return (v.repeat_of, v.stop, v.no_change_repeats, v.screen_visits)


def click(x: int, y: int) -> dict[str, Any]:
    return {"x": x, "y": y}


C = click(423, 631)


# --------------------------------------------------------------------------------------------- A-1, A-2


def test_constants_and_reasons() -> None:
    """A-1: the constants at their ruled values (H-T137-1) and the two reason strings byte for byte; A-2: ``Step`` and
    ``Verdict`` are frozen dataclasses carrying the named fields, and ``judge`` returns a ``Verdict``."""
    m = prog()
    assert getattr(m, "REPEAT_TOLERANCE", None) == 10, "T-0137 A-1: REPEAT_TOLERANCE = 10"
    assert getattr(m, "MAX_NO_CHANGE_REPEATS", None) == 2, "T-0137 A-1: MAX_NO_CHANGE_REPEATS = 2"
    assert getattr(m, "MAX_SCREEN_VISITS", None) == 3, "T-0137 A-1: MAX_SCREEN_VISITS = 3"
    actions = getattr(m, "ACTIONS", None)
    assert isinstance(actions, frozenset), "T-0137 A-1: ACTIONS is a frozenset"
    assert actions == {"click_at", "type_at", "scroll", "press_key"}, actions
    assert getattr(m, "NO_PROGRESS_REPEAT_REASON", None) == REPEAT_REASON
    assert getattr(m, "NO_PROGRESS_RECURRING_REASON", None) == RECURRING_REASON
    step_fields = {
        "step",
        "name",
        "arguments",
        "screen",
        "status",
        "overview",
        "no_change",
        "outcome",
        "action",
    }
    verdict_fields = {"repeat_of", "stop", "no_change_repeats", "screen_visits"}
    for cname, need in (("Step", step_fields), ("Verdict", verdict_fields)):
        cls = getattr(m, cname, None)
        assert isinstance(cls, type) and dataclasses.is_dataclass(cls), (
            f"T-0137 A-2: {cname} is a dataclass"
        )
        assert cls.__dataclass_params__.frozen, f"T-0137 A-2: {cname} is frozen"  # type: ignore[attr-defined]
        have = {f.name for f in dataclasses.fields(cls)}
        assert need <= have, f"T-0137 A-2: {cname} lacks {sorted(need - have)}"
    s = S(1, "click_at", C, "A")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.screen = "B"
    v = judge([], "click_at", C, "A")
    assert isinstance(v, m.Verdict), type(v)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.stop = "repeat"
    assert vt(v) == (None, None, 0, 1), (
        "an empty history: no repeat, no stop, the current visit only"
    )


# --------------------------------------------------------------------------------------------- A-3


# (id, earlier (name, args, screen, overview), proposed (name, args, screen, overview), same?)
SAME_TABLE: list[tuple[str, tuple[Any, ...], tuple[Any, ...], bool]] = [
    (
        "click_dx10",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(110, 200), "A", False),
        True,
    ),
    (
        "click_dx11",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(111, 200), "A", False),
        False,
    ),
    (
        "click_dx-10",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(90, 200), "A", False),
        True,
    ),
    (
        "click_dx-11",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(89, 200), "A", False),
        False,
    ),
    (
        "click_dy10",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(100, 210), "A", False),
        True,
    ),
    (
        "click_dy11",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(100, 211), "A", False),
        False,
    ),
    (
        "click_dy-11",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(100, 189), "A", False),
        False,
    ),
    # Chebyshev, not Euclidean: (10, 10) is ~14.1 units away (mutation 13).
    (
        "click_d10_10",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(110, 210), "A", False),
        True,
    ),
    (
        "click_d-10_10",
        ("click_at", click(100, 200), "A", False),
        ("click_at", click(90, 210), "A", False),
        True,
    ),
    (
        "click_d11_0_vs_10_10",
        ("click_at", click(500, 500), "A", False),
        ("click_at", click(511, 501), "A", False),
        False,
    ),
    ("click_exact", ("click_at", C, "A", False), ("click_at", C, "A", False), True),
    (
        "type_same",
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": True}, "A", False),
        ("type_at", {"x": 505, "y": 295, "text": "milk", "press_enter": True}, "A", False),
        True,
    ),
    (
        "type_dx11",
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        ("type_at", {"x": 511, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        False,
    ),
    (
        "type_text_differs",
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "eggs", "press_enter": False}, "A", False),
        False,
    ),
    (
        "type_text_case_differs",
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "Milk", "press_enter": False}, "A", False),
        False,
    ),
    (
        "type_enter_missing_vs_false",
        ("type_at", {"x": 500, "y": 300, "text": "milk"}, "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        True,
    ),
    (
        "type_enter_false_vs_missing",
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": False}, "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "milk"}, "A", False),
        True,
    ),
    (
        "type_enter_missing_vs_true",
        ("type_at", {"x": 500, "y": 300, "text": "milk"}, "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "milk", "press_enter": True}, "A", False),
        False,
    ),
    (
        "scroll_same",
        ("scroll", {"direction": "down"}, "A", False),
        ("scroll", {"direction": "down"}, "A", False),
        True,
    ),
    (
        "scroll_dir",
        ("scroll", {"direction": "down"}, "A", False),
        ("scroll", {"direction": "up"}, "A", False),
        False,
    ),
    (
        "key_same",
        ("press_key", {"combo": "Enter"}, "A", False),
        ("press_key", {"combo": "Enter"}, "A", False),
        True,
    ),
    (
        "key_combo",
        ("press_key", {"combo": "Enter"}, "A", False),
        ("press_key", {"combo": "Tab"}, "A", False),
        False,
    ),
    (
        "key_case",
        ("press_key", {"combo": "Ctrl+A"}, "A", False),
        ("press_key", {"combo": "ctrl+a"}, "A", False),
        False,
    ),
    (
        "search_never",
        ("google_search", {"query": "oat milk"}, "A", False),
        ("google_search", {"query": "oat milk"}, "A", False),
        False,
    ),
    (
        "overview_tool_never",
        ("page_overview", {}, "A", False),
        ("page_overview", {}, "A", False),
        False,
    ),
    (
        "names_differ",
        ("click_at", click(500, 300), "A", False),
        ("type_at", {"x": 500, "y": 300, "text": "a"}, "A", False),
        False,
    ),
    ("screens_differ", ("click_at", C, "A", False), ("click_at", C, "B", False), False),
    ("earlier_screen_none", ("click_at", C, None, False), ("click_at", C, "A", False), False),
    ("proposed_screen_none", ("click_at", C, "A", False), ("click_at", C, None, False), False),
    ("both_screens_none", ("click_at", C, None, False), ("click_at", C, None, False), False),
    # Mutation 10: an overview-frame call (step-1 overview units) is never the same action, on either side.
    ("earlier_overview", ("click_at", C, "A", True), ("click_at", C, "A", False), False),
    ("proposed_overview", ("click_at", C, "A", False), ("click_at", C, "A", True), False),
    ("both_overview", ("click_at", C, "A", True), ("click_at", C, "A", True), False),
]


@pytest.mark.parametrize(
    ("earlier", "proposed", "same"), [r[1:] for r in SAME_TABLE], ids=[r[0] for r in SAME_TABLE]
)
def test_same_action_table(earlier: tuple[Any, ...], proposed: tuple[Any, ...], same: bool) -> None:
    """A-3, through ``judge`` (R-2): one executed earlier step (step 7); the proposed call is the same action exactly
    when ``repeat_of == 7``. A lone original is not a repeat, so ``no_change_repeats`` stays 0 either way."""
    e_name, e_args, e_screen, e_ov = earlier
    p_name, p_args, p_screen, p_ov = proposed
    hist = [S(7, e_name, e_args, e_screen, overview=e_ov, no_change=True)]
    v = judge(hist, p_name, p_args, p_screen, overview=p_ov)
    assert v.repeat_of == (7 if same else None), f"T-0137 A-3: same action = {same}; got {vt(v)}"
    assert v.no_change_repeats == 0, vt(v)
    assert v.stop is None, vt(v)


# --------------------------------------------------------------------------------------------- A-4


def test_repeat_of_latest_executed_only() -> None:
    """A-4 (adversarial, mutation 2): only ``executed`` steps are instances. Refused (the guard's own refusal
    included), failed, timed-out and dry-run matches are skipped for ``repeat_of`` and never counted."""
    only_unexecuted = [
        S(1, "click_at", C, "A", "refused", no_change=True),
        S(2, "click_at", C, "A", "failed", no_change=True),
        S(3, "click_at", C, "A", "timeout", no_change=True),
        S(4, "click_at", C, "A", "dry_run", no_change=True),
    ]
    v = judge(only_unexecuted, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (None, 0, None), (
        f"T-0137 A-4: no instance; {vt(v)}"
    )

    hist = [
        S(1, "click_at", C, "A", no_change=True),
        S(2, "scroll", {"direction": "down"}, "A", no_change=False),
        S(3, "click_at", C, "A", "refused", no_change=True),
        S(4, "click_at", C, "A", "failed", no_change=True),
        S(5, "click_at", C, "A", "timeout", no_change=True),
        S(6, "click_at", C, "A", "dry_run", no_change=True),
    ]
    v = judge(hist, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (1, 0, None), (
        f"T-0137 A-4: latest executed is 1; {vt(v)}"
    )

    # Three executed no-change instances, then the guard's own refusal of the fourth: the fifth still repeats step 3
    # (not the refused step 4) with 2 no-change repeats, and is stopped again (no latch, H-T137-5).
    after_stop = [
        S(1, "click_at", C, "A", no_change=True),
        S(2, "click_at", C, "A", no_change=True),
        S(3, "click_at", C, "A", no_change=True),
        S(4, "click_at", C, "A", "refused", no_change=True),
    ]
    v = judge(after_stop, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (3, 2, "repeat"), (
        f"T-0137 A-4/A-6: {vt(v)}"
    )

    # The latest executed instance wins over an earlier one, with unrelated executed steps between.
    hist = [
        S(1, "click_at", C, "A", no_change=False),
        S(2, "press_key", {"combo": "Tab"}, "A", no_change=False),
        S(3, "click_at", click(425, 629), "A", no_change=False),
        S(4, "click_at", C, "A", "refused", no_change=True),
    ]
    v = judge(hist, "click_at", C, "A")
    assert v.repeat_of == 3, f"T-0137 A-4: the latest executed instance; {vt(v)}"


def test_stop_on_fourth() -> None:
    """A-6 (adversarial, off by one; mutations 1 and 3): three executed no-change instances make the fourth call
    ``stop "repeat"``; two make the third execute with ``repeat_of`` set and no stop."""
    three = [S(i, "click_at", C, "A", no_change=True) for i in (1, 2, 3)]
    v = judge(three, "click_at", C, "A")
    assert vt(v) == (3, "repeat", 2, 1), (
        f"T-0137 A-6: the fourth identical no-change action stops; {vt(v)}"
    )

    two = three[:2]
    v = judge(two, "click_at", C, "A")
    assert vt(v) == (2, None, 1, 1), f"T-0137 A-6: the third executes; {vt(v)}"

    one = three[:1]
    v = judge(one, "click_at", C, "A")
    assert vt(v) == (1, None, 0, 1), f"T-0137 A-4: the original is not a repeat; {vt(v)}"


def test_original_not_counted() -> None:
    """A-4 (adversarial, mutation 1): the earliest instance is the original, never a repeat, even when it changed
    nothing."""
    hist = [S(1, "click_at", C, "A", no_change=True), S(2, "click_at", C, "A", no_change=True)]
    v = judge(hist, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (2, 1, None), f"T-0137 A-4: {vt(v)}"

    hist = [
        S(1, "scroll", {"direction": "down"}, "A", no_change=True),
        S(2, "scroll", {"direction": "down"}, "A", no_change=False),
    ]
    v = judge(hist, "scroll", {"direction": "down"}, "A")
    assert (v.repeat_of, v.no_change_repeats) == (2, 0), (
        f"T-0137 A-4: original no-change not counted; {vt(v)}"
    )


def test_changed_or_unknown_not_counted() -> None:
    """A-4: only repeats with ``no_change is True`` count; ``False`` (changed) and ``None`` (unknown) never do."""
    hist = [
        S(1, "click_at", C, "A", no_change=True),
        S(2, "click_at", C, "A", no_change=False),
        S(3, "click_at", C, "A", no_change=True),
    ]
    v = judge(hist, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (3, 1, None), f"T-0137 A-4: {vt(v)}"

    unknown = (
        [S(1, "click_at", C, "A", no_change=True)]
        + [S(i, "click_at", C, "A", no_change=None) for i in (2, 3, 4)]
        + [S(5, "click_at", C, "A", no_change=True)]
    )
    v = judge(unknown, "click_at", C, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (5, 1, None), (
        f"T-0137 A-4: None never counts; {vt(v)}"
    )

    # A changed original does not block the count: two no-change repeats after it stop the next call.
    hist = [
        S(1, "press_key", {"combo": "Enter"}, "A", no_change=False),
        S(2, "press_key", {"combo": "Enter"}, "A", no_change=True),
        S(3, "press_key", {"combo": "Enter"}, "A", no_change=True),
    ]
    v = judge(hist, "press_key", {"combo": "Enter"}, "A")
    assert (v.repeat_of, v.no_change_repeats, v.stop) == (3, 2, "repeat"), (
        f"T-0137 A-4/A-6: {vt(v)}"
    )


def _seq(*ncs: bool | None) -> list[Any]:
    """Executed identical clicks on screen A, steps 1..n, with these ``no_change`` values (the first is the original)."""
    return [S(i, "click_at", C, "A", no_change=nc) for i, nc in enumerate(ncs, 1)]


def test_no_change_run_resets() -> None:
    """A-4 as ruled by O-A2 (adversarial, reviewer F1): ``no_change_repeats`` is the trailing run of no-change repeats
    ending at the latest instance, reset by any repeat whose ``no_change`` is not ``True``, so the repeat reason ("did
    not change this screen the last two times") is always true. The cumulative count fails the first and third."""
    v = judge(_seq(True, True, True, False), "click_at", C, "A")
    assert vt(v) == (4, None, 0, 1), (
        f"T-0137 O-A2: [orig nc, nc, nc, changed]: the last one changed; {vt(v)}"
    )

    v = judge(_seq(False, False, True, True), "click_at", C, "A")
    assert vt(v) == (4, "repeat", 2, 1), f"T-0137 O-A2: [orig, changed, nc, nc] stops; {vt(v)}"
    v = judge(_seq(True, False, True, True), "click_at", C, "A")
    assert vt(v) == (4, "repeat", 2, 1), f"T-0137 O-A2: [orig nc, changed, nc, nc] stops; {vt(v)}"

    v = judge(_seq(True, True, False, True), "click_at", C, "A")
    assert vt(v) == (4, None, 1, 1), f"T-0137 O-A2: [nc, nc, changed, nc]; {vt(v)}"

    # An unknown outcome resets too (``no_change`` not True).
    v = judge(_seq(True, True, True, None), "click_at", C, "A")
    assert vt(v) == (4, None, 0, 1), f"T-0137 O-A2: a trailing unknown resets; {vt(v)}"
    v = judge(_seq(True, True, None, True), "click_at", C, "A")
    assert vt(v) == (4, None, 1, 1), f"T-0137 O-A2: an unknown in the middle resets; {vt(v)}"

    # Only instances form the run: a different action that changed the screen, and a refused or failed identical
    # call, between two no-change repeats do not reset it ("among the repeats").
    hist = [
        S(1, "click_at", C, "A", no_change=True),
        S(2, "click_at", C, "A", no_change=True),
        S(3, "scroll", {"direction": "down"}, "A", no_change=False),
        S(4, "click_at", C, "A", "refused", no_change=False),
        S(5, "click_at", C, "A", "failed", no_change=False),
        S(6, "click_at", C, "A", no_change=True),
    ]
    v = judge(hist, "click_at", C, "A")
    assert vt(v) == (6, "repeat", 2, 1), f"T-0137 O-A2: non-instances do not break the run; {vt(v)}"


# --------------------------------------------------------------------------------------------- A-5


def _distinct(i: int, screen: str | None, status: str = "executed") -> Any:
    """Step ``i``: a click far from every other ``_distinct`` click, so none is the same action as another."""
    return S(i, "click_at", click(40 * i, 900), screen, status, no_change=False)


def test_recurring_by_visits() -> None:
    """A-5/A-6 (mutation 4): visits are runs of consecutive equal screens, ``None`` dropped, every history step
    counted whatever its status (R-4); the current run includes the proposed call."""
    P = click(700, 100)
    abab = [_distinct(1, "A"), _distinct(2, "B"), _distinct(3, "A"), _distinct(4, "B")]
    v = judge(abab, "click_at", P, "A")
    assert (v.screen_visits, v.stop, v.repeat_of) == (3, "recurring", None), (
        f"T-0137 A-5: A B A B + A; {vt(v)}"
    )

    v = judge(abab, "click_at", P, "B")
    assert (v.screen_visits, v.stop) == (2, None), (
        f"T-0137 A-5: A B A B + B continues the B run; {vt(v)}"
    )

    aba = abab[:3]
    v = judge(aba, "click_at", P, "A")
    assert (v.screen_visits, v.stop) == (2, None), (
        f"T-0137 A-5: A B A + A is 2 runs, not 3 occurrences; {vt(v)}"
    )

    v = judge(aba, "click_at", P, "C")
    assert (v.screen_visits, v.stop) == (1, None), f"T-0137 A-5: a new screen; {vt(v)}"

    nones = [
        _distinct(1, "A"),
        _distinct(2, None),
        _distinct(3, "A"),
        _distinct(4, None),
        _distinct(5, "A"),
    ]
    v = judge(nones, "click_at", P, "A")
    assert (v.screen_visits, v.stop) == (1, None), (
        f"T-0137 A-5: None screens are dropped, not breaks; {vt(v)}"
    )

    with_nones = [
        _distinct(1, "A"),
        _distinct(2, None),
        _distinct(3, "B"),
        _distinct(4, "A"),
        _distinct(5, None),
        _distinct(6, "B"),
    ]
    v = judge(with_nones, "click_at", P, "A")
    assert (v.screen_visits, v.stop) == (3, "recurring"), f"T-0137 A-5: {vt(v)}"

    statuses = [
        _distinct(1, "A"),
        _distinct(2, "B", "refused"),
        _distinct(3, "A", "failed"),
        _distinct(4, "B", "timeout"),
    ]
    v = judge(statuses, "click_at", P, "A")
    assert (v.screen_visits, v.stop) == (3, "recurring"), (
        f"T-0137 A-5 (R-4): every history step; {vt(v)}"
    )

    v = judge(abab, "click_at", P, None)
    assert v.stop is None, f"T-0137 A-6: a None screen never stops; {vt(v)}"


def test_form_fill_never_recurs() -> None:
    """A-5 (adversarial, mutation 4): the logged S3 #11 shape, three ``type_at`` with different text and y, then a
    ``click_at``, all on one screen key with the screen changing each time. Every verdict, judged step by step over
    the prefix so far, is ``stop None`` with one visit and no repeat."""
    calls = [
        ("type_at", {"x": 500, "y": 300, "text": "Ada"}),
        ("type_at", {"x": 500, "y": 360, "text": "Lovelace"}),
        ("type_at", {"x": 500, "y": 420, "text": "ada@example.org", "press_enter": False}),
        ("click_at", {"x": 500, "y": 480}),
    ]
    hist: list[Any] = []
    for i, (name, args) in enumerate(calls, 1):
        v = judge(list(hist), name, args, "form")
        assert vt(v) == (None, None, 0, 1), f"T-0137 A-5: form-fill step {i} never recurs; {vt(v)}"
        hist.append(S(i, name, args, "form", no_change=False))


# --------------------------------------------------------------------------------------------- A-6


def test_repeat_precedence() -> None:
    """A-6: when both conditions hold, ``stop == "repeat"``."""
    hist = [
        S(1, "click_at", C, "A", no_change=True),
        _distinct(2, "B"),
        S(3, "click_at", C, "A", no_change=True),
        _distinct(4, "B"),
        S(5, "click_at", C, "A", no_change=True),
    ]
    v = judge(hist, "click_at", C, "A")
    assert vt(v) == (5, "repeat", 2, 3), (
        f"T-0137 A-6: repeat takes precedence over recurring; {vt(v)}"
    )


def test_non_actions_never_stopped() -> None:
    """A-6: a proposed ``google_search``/``page_overview``, or any call with ``screen is None``, gives ``stop None``,
    on a history where a ``click_at`` on screen A would meet both conditions."""
    hist = [
        S(1, "click_at", C, "A", no_change=True),
        S(2, "google_search", {"query": "x"}, "B", no_change=None),
        S(3, "click_at", C, "A", no_change=True),
        S(4, "page_overview", {}, "B", no_change=None),
        S(5, "click_at", C, "A", no_change=True),
    ]
    assert judge(hist, "click_at", C, "A").stop == "repeat", "premise: a click here stops"
    for name, args in (("google_search", {"query": "x"}), ("page_overview", {})):
        v = judge(hist, name, args, "A")
        assert (v.stop, v.repeat_of) == (None, None), f"T-0137 A-6: {name} never stops; {vt(v)}"
        v = judge(hist, name, args, "B")
        assert (v.stop, v.repeat_of) == (None, None), f"T-0137 A-6: {name} never stops; {vt(v)}"
    v = judge(hist, "click_at", C, None)
    assert (v.stop, v.repeat_of) == (None, None), f"T-0137 A-6: a None screen never stops; {vt(v)}"


# --------------------------------------------------------------------------------------------- Hypothesis

# Weighted towards repeats (one click, one screen, executed, no change) so both stops occur in the generated
# histories; the ``event`` lines report how often (``--hypothesis-show-statistics``).
SCREENS = ("A", "A", "A", "B", "C", None)
NAMES = ("click_at",) * 6 + ("type_at", "scroll", "press_key", "google_search", "page_overview")
STATUSES = ("executed",) * 6 + ("refused", "failed", "timeout", "dry_run")


def _args(name: str, k: int) -> dict[str, Any]:
    """A small argument space (k in 0..2) so repeats are frequent."""
    xy = [(100, 100), (105, 108), (300, 100)][k]
    if name == "click_at":
        return {"x": xy[0], "y": xy[1]}
    if name == "type_at":
        return {"x": xy[0], "y": xy[1], "text": ["a", "a", "b"][k]}
    if name == "scroll":
        return {"direction": ["down", "down", "up"][k]}
    if name == "press_key":
        return {"combo": ["Enter", "Enter", "Tab"][k]}
    if name == "google_search":
        return {"query": "q"}
    return {}


raw_step = st.tuples(
    st.sampled_from(NAMES),
    st.sampled_from((0, 0, 1, 2)),
    st.sampled_from(SCREENS),
    st.sampled_from(STATUSES),
    st.sampled_from((False,) * 5 + (True,)),
    st.sampled_from((True, True, True, False, None)),
)
raw_hist = st.one_of(
    st.lists(raw_step, max_size=9),
    # each of 1-3 steps three times, shuffled: repeated actions and recurring screens are common
    st.lists(raw_step, min_size=1, max_size=3).flatmap(lambda xs: st.permutations(xs * 3)),
)
raw_call = st.tuples(
    st.sampled_from(NAMES),
    st.sampled_from((0, 0, 1, 2)),
    st.sampled_from(SCREENS),
    st.sampled_from((False,) * 5 + (True,)),
)


def _build(raw: list[tuple[Any, ...]], relabel: Mapping[Any, Any] | None = None) -> list[Any]:
    out = []
    for i, (name, k, screen, status, ov, nc) in enumerate(raw, 1):
        scr = relabel[screen] if relabel is not None else screen
        out.append(S(i, name, _args(name, k), scr, status, overview=ov, no_change=nc))
    return out


RELABEL = st.permutations(["A", "B", "C"]).map(
    lambda p: {
        "A": f"phash:{p[0]}#g2",
        "B": f"phash:{p[1]}#g2",
        "C": f"phash:{p[2]}#g2",
        None: None,
    }
)


@settings(max_examples=150, deadline=None)
@given(raw=raw_hist, call=raw_call, relabel=RELABEL, data=st.data())
def test_properties(
    raw: list[tuple[Any, ...]], call: tuple[Any, ...], relabel: dict[Any, Any], data: Any
) -> None:
    """(i) a bijective relabelling of screen keys (the generation salt) leaves the verdict unchanged; (ii) inserting
    ``google_search`` steps that carry the previous step's screen leaves it unchanged; (iii) ``judge`` leaves
    ``history`` equal to a deep copy taken before the call; (iv) bounds the rule implies: fewer than 3 executed steps
    never give ``"repeat"``, fewer than 4 non-``None`` history screens never give ``"recurring"``, and the counts
    are bounded by what the history holds."""
    hist = _build(raw)
    name, k, screen, ov = call
    args = _args(name, k)
    before = copy.deepcopy(hist)
    v = judge(hist, name, args, screen, overview=ov)
    event(f"stop={v.stop}")
    event(f"repeat_of set={v.repeat_of is not None}")

    # (iii)
    assert hist == before, "T-0137 A-7: judge mutated history"

    # (i)
    v_rel = judge(_build(raw, relabel), name, args, relabel[screen], overview=ov)
    assert vt(v_rel) == vt(v), (
        f"T-0137 (i): relabelling screens changed the verdict: {vt(v)} -> {vt(v_rel)}"
    )

    # (ii)
    positions = data.draw(st.lists(st.integers(0, len(hist)), max_size=3), label="insert at")
    ins = list(hist)
    for n, pos in enumerate(sorted(positions, reverse=True)):
        prev_screen = ins[pos - 1].screen if pos > 0 else None
        status = data.draw(st.sampled_from(STATUSES), label="inserted status")
        ins.insert(
            pos, S(100 + n, "google_search", {"query": "q"}, prev_screen, status, no_change=None)
        )
    v_ins = judge(ins, name, args, screen, overview=ov)
    assert vt(v_ins) == vt(v), (
        f"T-0137 (ii): an inserted google_search changed the verdict: {vt(v)} -> {vt(v_ins)}"
    )

    # (iv) bounds
    executed = [s for s in hist if s.status == "executed"]
    non_none = [s.screen for s in hist if s.screen is not None]
    if len(executed) < 3:
        assert v.stop != "repeat", (
            f"T-0137 (iv): {len(executed)} executed steps cannot stop on repeat; {vt(v)}"
        )
    if len(non_none) < 4:
        assert v.stop != "recurring", f"T-0137 (iv): {len(non_none)} screens cannot recur; {vt(v)}"
    assert v.stop in (None, "repeat", "recurring"), vt(v)
    if v.repeat_of is None:
        assert v.no_change_repeats == 0, vt(v)
    else:
        same_tool = {
            s.step for s in executed if s.name == name and s.screen == screen and not s.overview
        }
        assert v.repeat_of in same_tool, (
            f"T-0137 A-4: repeat_of is an executed non-overview step of this tool on this screen; {vt(v)}"
        )
        later = [s for s in executed if s.step <= v.repeat_of]
        assert 0 <= v.no_change_repeats <= sum(1 for s in later if s.no_change is True), vt(v)
        assert v.no_change_repeats <= len(executed) - 1, vt(v)
        latest = next(s for s in executed if s.step == v.repeat_of)
        if latest.no_change is not True:
            # O-A2: the reason claims the last two times changed nothing, so a changed or unknown latest instance
            # can never be part of a stop.
            assert v.no_change_repeats == 0 and v.stop != "repeat", f"T-0137 O-A2: {vt(v)}"
    if screen is None:
        assert v.stop is None and v.repeat_of is None, f"T-0137 A-3/A-6: a None screen; {vt(v)}"
    else:
        assert 1 <= v.screen_visits <= len(non_none) // 2 + 1, (
            f"T-0137 A-5: visits bounded by runs; {vt(v)}"
        )
    if name in ("google_search", "page_overview") or ov:
        assert v.repeat_of is None, f"T-0137 A-3: never the same action; {vt(v)}"
    if name in ("google_search", "page_overview"):
        assert v.stop is None, vt(v)


# --------------------------------------------------------------------------------------------- A-1 purity


def test_stdlib_only() -> None:
    """A-1: stdlib-only imports, nothing from ``local_astra``, no relative imports, no ``asyncio``/``time`` or other
    I/O, clock or randomness module, no I/O builtins, no ``global`` statement and no module-level mutable container;
    and deterministic: the same input gives the same verdict, unaffected by judging something else in between."""
    prog()
    tree = ast.parse(MODULE_PATH.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "T-0137 A-1: no relative (project) imports"
            imported.add((node.module or "").split(".")[0])
    imported.discard("__future__")
    assert "local_astra" not in imported, "T-0137 A-1: imports nothing from local_astra"
    non_std = sorted(m for m in imported if m not in sys.stdlib_module_names)
    assert not non_std, f"T-0137 A-1: stdlib-only; imports {non_std}"
    banned = {
        "asyncio", "time", "datetime", "os", "io", "pathlib", "socket", "subprocess", "random", "logging",
        "shutil", "tempfile", "urllib", "http", "threading", "sys", "secrets", "uuid",
    }  # fmt: skip
    assert not (imported & banned), (
        f"T-0137 A-1: no I/O, clock or randomness; imports {sorted(imported & banned)}"
    )
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not (called & {"open", "print", "input", "eval", "exec"}), sorted(called)
    assert not any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(tree)), (
        "T-0137 A-1: no global state"
    )
    mutable_ctor = {
        "list",
        "dict",
        "set",
        "defaultdict",
        "deque",
        "Counter",
        "OrderedDict",
        "bytearray",
    }
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            val = node.value
            is_mut = isinstance(
                val, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
            ) or (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id in mutable_ctor
            )
            assert not is_mut, f"T-0137 A-1: module-level mutable state at line {node.lineno}"

    hist = [S(i, "click_at", C, "A", no_change=True) for i in (1, 2, 3)]
    first = vt(judge(hist, "click_at", C, "A"))
    judge(
        [_distinct(1, "A"), _distinct(2, "B"), _distinct(3, "A"), _distinct(4, "B")],
        "click_at",
        C,
        "A",
    )
    judge([], "scroll", {"direction": "up"}, "Z")
    assert vt(judge(hist, "click_at", C, "A")) == first == (3, "repeat", 2, 1)
    assert judge(tuple(hist), "click_at", C, "A") == judge(list(hist), "click_at", C, "A"), (
        "a Sequence, not a list"
    )


# --------------------------------------------------------------------------------------------- A-7 frozen paths


def test_frozen_paths_over_closed_range() -> None:
    """A-7: over ``BASE_A..TIP_A`` (two literals, never the working tree) ``local_astra/**`` other than progress.py
    and ``tests/fakes/**`` are byte-identical, and nothing outside group A's files changes (R-5). Green at BASE (TIP_A
    starts there); once TIP_A is bumped, progress.py must be the one file added. Power: the same predicate sees
    T-0142's changes over its range."""
    gr.require_commits(BASE_A, TIP_A, POWER_BASE, POWER_TIP)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:7]} is not an ancestor of {TIP_A[:7]}"
    frozen = ("local_astra", "tests/fakes")
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A, "--", *frozen).stdout.split()
    for p in frozen:
        assert any(x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE_A"
    moved = [p for p in gr.changed(BASE_A, TIP_A, *frozen) if p != REL_MODULE]
    assert moved == [], f"T-0137 A-7: frozen paths differ in {BASE_A[:7]}..{TIP_A[:7]}: {moved}"
    allowed = {
        REL_MODULE,
        "tests/unit/test_t0137_progress.py",
        "docs/tasks/T-0137.md",
        "PROGRESS.md",
    }
    other = [p for p in gr.changed(BASE_A, TIP_A, ".") if p not in allowed]
    assert other == [], f"T-0137 A-7: group A changes only progress.py (R-5); also changed: {other}"
    if TIP_A != BASE_A:
        assert gr.changed(BASE_A, TIP_A, REL_MODULE, diff_filter="A") == [REL_MODULE], (
            f"T-0137 A-7: {REL_MODULE} is added in {BASE_A[:7]}..{TIP_A[:7]}"
        )
    power = [p for p in gr.changed(POWER_BASE, POWER_TIP, *frozen) if p != REL_MODULE]
    assert "local_astra/conductor/degeneracy.py" in power, (
        f"power: the predicate sees a known change; {power}"
    )
