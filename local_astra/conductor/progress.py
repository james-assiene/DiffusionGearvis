"""T-0137: a pure rule for repeated browser actions and recurring screens (the no-progress stop).

The session (``repeat_guard``, group B) keeps one ``Step`` per settled loop step of a turn and asks ``judge`` about
each proposed browser action. The verdict says which earlier executed step the call repeats (``repeat_of``), how many
of that action's repeats changed nothing (``no_change_repeats``), how many separate visits the proposed screen has had
(``screen_visits``), and whether the call is refused (``stop``).

- **Same action** (A-3): equal names in ``ACTIONS``, equal non-``None`` screens, neither call read off an overview
  frame, and the arguments agree: ``click_at`` within ``REPEAT_TOLERANCE`` model units per axis (Chebyshev);
  ``type_at`` the same plus equal ``text`` and ``press_enter`` (missing is ``False``); ``scroll`` equal
  ``direction``; ``press_key`` equal ``combo``.
- **Repeat** (A-4): the instances are the executed history steps that are the same action as the proposed call;
  ``repeat_of`` is the latest, and ``no_change_repeats`` is the length of the trailing run of instances after the
  earliest whose ``no_change`` is ``True``, ending at the latest (O-A2: any other repeat resets it to 0).
- **Recurrence** (A-5): the history's screens then the proposed one, ``None`` dropped, consecutive equal values
  collapsed into runs; ``screen_visits`` is the number of runs of the proposed screen.
- **Stop** (A-6): ``"repeat"`` at ``MAX_NO_CHANGE_REPEATS`` no-change repeats, else ``"recurring"`` at
  ``MAX_SCREEN_VISITS`` visits; only for a proposed name in ``ACTIONS`` on a known screen. No latch (H-T137-5).
- **Start page** (T-0231 A, R-A3): a step marked ``start_page`` (its call was made on the session's own start page,
  which the session knows from the executor's reports, never from pixels) adds no visit: it ends the run before it
  but is never a run itself. A proposed call with ``start_page=True`` has no visits and is never stopped as
  ``"recurring"``; the repeat rule (A-4) still applies to it. Both default to ``False`` (BASE's rule exactly).

Pure and stdlib-only: no I/O, no clock, no global mutable state, deterministic (the T-0142 ``degeneracy.py`` pattern).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: model units (0-999), per axis (Chebyshev)
REPEAT_TOLERANCE = 10
MAX_NO_CHANGE_REPEATS = 2
MAX_SCREEN_VISITS = 3
ACTIONS = frozenset({"click_at", "type_at", "scroll", "press_key"})

NO_PROGRESS_REPEAT_REASON = (
    "the same action did not change this screen the last two times; it was not done again"
)
NO_PROGRESS_RECURRING_REASON = (
    "this screen has come back for the third time in this request; the action was not done"
)


@dataclass(frozen=True)
class Step:
    """One settled step of the turn, as the session recorded it.

    ``screen`` is an opaque screen key (``None``: unknown or blind); ``status`` is ``executed``/``refused``/
    ``failed``/``timeout``/``dry_run``; ``overview`` marks a call read off a T-0151 overview frame; ``no_change`` is
    ``True`` for no visible change, ``False`` for changed, ``None`` for unknown; ``outcome`` holds the step's outcome
    keys verbatim; ``action`` is the executor's action string."""

    step: int
    name: str
    arguments: Mapping[str, Any]
    screen: str | None
    status: str
    overview: bool
    no_change: bool | None
    outcome: Mapping[str, str]
    action: str | None
    #: T-0231 A: the call was made on the session's own start page (never a screen visit)
    start_page: bool = False


@dataclass(frozen=True)
class Verdict:
    """What ``judge`` decided about a proposed call."""

    repeat_of: int | None
    stop: Literal["repeat", "recurring"] | None
    no_change_repeats: int
    screen_visits: int


def _near(a: Any, b: Any) -> bool:
    """Coordinates within ``REPEAT_TOLERANCE``; anything that is not a number compares exactly."""
    numbers = (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    )
    if numbers:
        return bool(abs(a - b) <= REPEAT_TOLERANCE)
    return bool(a == b)


def _same_arguments(name: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if name in ("click_at", "type_at"):
        if not (_near(a.get("x"), b.get("x")) and _near(a.get("y"), b.get("y"))):
            return False
        if name == "click_at":
            return True
        return bool(
            a.get("text") == b.get("text")
            and a.get("press_enter", False) == b.get("press_enter", False)
        )
    if name == "scroll":
        return bool(a.get("direction") == b.get("direction"))
    if name == "press_key":
        return bool(a.get("combo") == b.get("combo"))
    return False


def same_action(
    a_name: str,
    a_arguments: Mapping[str, Any],
    a_screen: str | None,
    a_overview: bool,
    b_name: str,
    b_arguments: Mapping[str, Any],
    b_screen: str | None,
    b_overview: bool,
) -> bool:
    """A-3: whether two calls are the same action."""
    if a_name != b_name or a_name not in ACTIONS:
        return False
    if a_screen is None or b_screen is None or a_screen != b_screen:
        return False
    if a_overview or b_overview:
        return False
    return _same_arguments(a_name, a_arguments, b_arguments)


def _screen_visits(history: Sequence[Step], screen: str | None) -> int:
    """A-5: the runs of ``screen`` in the history's screens then the proposed one, ``None`` dropped."""
    if screen is None:
        return 0
    visits = 0
    previous: str | None = None
    for current, start_page in [(s.screen, s.start_page) for s in history] + [(screen, False)]:
        if start_page:
            # T-0231 A: a start-page step ends the run before it and is never a visit itself
            previous = None
            continue
        if current is None:
            continue
        if current != previous and current == screen:
            visits += 1
        previous = current
    return visits


def judge(
    history: Sequence[Step],
    name: str,
    arguments: Mapping[str, Any],
    screen: str | None,
    *,
    overview: bool,
    start_page: bool = False,
) -> Verdict:
    """The verdict on a proposed call over the turn's settled ``history`` (never mutated). ``start_page``: the call
    is proposed on the session's own start page (T-0231 A: no visits, never ``"recurring"``)."""
    instances = [
        s
        for s in history
        if s.status == "executed"
        and same_action(
            s.name, s.arguments, s.screen, s.overview, name, arguments, screen, overview
        )
    ]
    repeat_of = instances[-1].step if instances else None
    # O-A2: the trailing run of no-change repeats (the original excluded), ending at the latest instance; any
    # repeat whose ``no_change`` is not ``True`` resets it
    no_change_repeats = 0
    for s in instances[1:]:
        no_change_repeats = no_change_repeats + 1 if s.no_change is True else 0
    screen_visits = 0 if start_page else _screen_visits(history, screen)

    stop: Literal["repeat", "recurring"] | None = None
    if name in ACTIONS and screen is not None:
        if repeat_of is not None and no_change_repeats >= MAX_NO_CHANGE_REPEATS:
            stop = "repeat"
        elif screen_visits >= MAX_SCREEN_VISITS:
            stop = "recurring"
    return Verdict(
        repeat_of=repeat_of,
        stop=stop,
        no_change_repeats=no_change_repeats,
        screen_visits=screen_visits,
    )
