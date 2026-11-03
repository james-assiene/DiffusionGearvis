"""T-0193 A0-8 (negative): on every event sequence with no ``UserStop``, ``step`` equals BASE's machine
(docs/tasks/T-0193.md A0-8, "Tests to write -> A0").

**Oracle.** ``tests/unit/_frozen_turn_machine_t0193.py``, a verbatim copy of ``turn_machine.py`` at fdb9fc01 (A0's
base; body sha256 pinned by ``test_t0193_stop_row.T0193_FROZEN_BODY_SHA256`` and re-checked here against git). Both
machines step from the same state on every event; the commands must be equal exactly, and the state field by field
over BASE's ``MachineState`` fields (``test_t0193_stop_row.BASE_STATE_FIELDS``): A0-8 excludes only the new per-turn
stop field.

**Traces.** L8's recipes (``test_turn_invariants._L8_EV_RESULTS``: every atomic §4 event, the composite drives to
SPEAKING/PAUSED/the T-0089 rows, tool drafts, repairs, results), plus long ticks, session resets, mic mutes, T-0192
hand-backs, and ``barge`` recipes that drive a fresh turn to SPEAKING and cut it by voice four ways (the rows A0
touches: ``_interrupt`` and the INTERRUPTED exits). Six policies (default, no thinking, ack on/off, the loop at budgets
1-3, still_working at 1.5 s). The power row counts what the traces reach and requires every A0-touched row.

**Anti-vacuity.** The same harness, given a ``UserStop`` for the current turn in SPEAKING, PAUSED and THINKING, must
see the live machine diverge from BASE (BASE has no row: ``(s, ())``). Pending until A0 lands.

Regime: CPU, pure; 200 derandomized examples of 4-40 recipes (at most 60 events) per policy, database off; plus
explicit examples. A guard: green at BASE (the live machine is the frozen one) and after A0.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import MachineState, TurnState
from tests.unit import test_t0193_stop_row as rows
from tests.unit import test_turn_invariants as ti

#: Long ticks (still_working, the R18/llm_timeout deadlines), resets, mutes, hand-backs and barge-ins
_LONG = (st.tuples(st.just("tick"), st.integers(1_000, 7_000)),)
_RESET = st.just(("reset",))
_MUTE = st.just(("mute",))
_HAND_BACK = st.tuples(st.just("hand_back"), st.sampled_from(["last", "pending", "unknown"]))
#: A fresh turn driven to SPEAKING (generate live, or done), cut by voice four ways, then exited by
#: ``playback.stopped`` or the timeout tick (or left INTERRUPTED for the next recipe)
_BARGE = st.tuples(
    st.just("barge"),
    st.sampled_from(["takeover", "long-offset", "takeover-tick", "word-in-pause"]),
    st.sampled_from(["stopped", "timeout", "none"]),
    st.booleans(),
)
_DIFF_EV = st.one_of(ti._L8_EV_RESULTS, *(_LONG * 2), _RESET, _MUTE, _HAND_BACK, *((_BARGE,) * 3))

DIFF_POLICIES = {
    "default": {},
    "no-thinking": {"thinking": False},
    "no-ack": {"ack_clip": False},
    "loop-1": {"tool_loop": True, "max_tool_steps": 1, "ack_clip": False},
    "loop-2": {"tool_loop": True, "max_tool_steps": 2},
    "loop-3-still-working": {
        "tool_loop": True,
        "max_tool_steps": 3,
        "ack_clip": False,
        "still_working_after_ms": 1_500,
    },
}


def _frozen() -> Any:
    from tests.unit import _frozen_turn_machine_t0193 as frozen

    return frozen


def _barge_events(recipe: Any, s: MachineState, t: int) -> list[Any]:
    """From any state: end it with a reset (unless IDLE), then a fresh turn to SPEAKING and a voice cut."""
    _, how, exit_, done = recipe
    events: list[Any] = [] if s.state is TurnState.IDLE else [E.SessionReset(t_ms=t + 5)]
    t += 10
    gen = f"t{s.turn_seq + 1}-gen"
    events += [
        E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50),
        E.SttWord(t_ms=t + 450, text="what", start_ms=t + 50, end_ms=t + 250),
        E.SttEndOfTurn(t_ms=t + 550, t_eot_ms=t + 500),
        E.SttFlushed(t_ms=t + 555, t_eot_ms=t + 500),
        E.Tick(t_ms=t + 556 + 120),
        E.LlmCommit(
            t_ms=t + 900,
            request_id=gen,
            canvas_idx=0,
            segment="answer",
            text="Yes. ",
            thought_open=False,
        ),
    ]
    if done:
        events.append(E.LlmDone(t_ms=t + 910, request_id=gen))
    t += 1_000
    onset = E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50)
    events += {
        "takeover": [E.UserTakeover(t_ms=t + 100, onset_ms=t + 50)],
        "long-offset": [onset, E.UserSpeechOffset(t_ms=t + 800, offset_ms=t + 800, speech_ms=750)],
        "takeover-tick": [onset, E.Tick(t_ms=t + 650)],
        "word-in-pause": [
            onset,
            E.SttWord(t_ms=t + 500, text="wait", start_ms=t + 60, end_ms=t + 300),
        ],
    }[how]
    if exit_ == "stopped":
        events.append(E.PlaybackStopped(t_ms=t + 900, played_ms=300, played_text="Yes."))
    elif exit_ == "timeout":
        events.append(E.Tick(t_ms=t + 3_000))
    return events


def _count(visits: dict[str, int], s: MachineState, ev: Any, new: MachineState) -> None:
    def hit(key: str) -> None:
        visits[key] = visits.get(key, 0) + 1

    if s.state is TurnState.SPEAKING and new.state is TurnState.INTERRUPTED:
        hit("speaking->interrupted")
    if s.state is TurnState.PAUSED and new.state is TurnState.INTERRUPTED:
        hit("paused->interrupted")
    if s.state is TurnState.INTERRUPTED and new.state is not TurnState.INTERRUPTED:
        hit(f"interrupted-exit:{ev.type}")
    if s.state is TurnState.THINKING and s.awaiting_result is not None:
        hit("thinking-awaiting")
    if (
        s.state is TurnState.THINKING
        and s.active_generate is not None
        and s.active_generate.endswith("-repair")
    ):
        hit("thinking-repair")
    if s.state is TurnState.THINKING and s.clip_playing:
        hit("thinking-clip")


def _diff_trace(
    live: Any,
    old: Any,
    recipes: list[Any],
    *,
    max_events: int = 60,
    visits: dict[str, int] | None = None,
) -> list[str]:
    """Step both machines from the same state on every event. Returns the divergences (A0-8: none without a stop);
    a recipe ``("stop", which)`` delivers ``UserStop`` (the anti-vacuity probe only)."""
    s = live.initial()
    assert s == old.initial()
    t, spoken, calls, n = 0, "", [], 0
    diverged: list[str] = []
    queue = list(recipes)
    while queue and n < max_events:
        recipe = queue.pop(0)
        if recipe[0] == "reset":
            events: list[Any] = [E.SessionReset(t_ms=t + 5)]
        elif recipe[0] == "mute":
            events = [E.MicMuted(t_ms=t + 5)]
        elif recipe[0] == "hand_back":
            call = {
                "last": calls[-1] if calls else "c0",
                "pending": s.tool.call_id if s.tool is not None else "c0",
                "unknown": "zz-9",
            }[recipe[1]]
            events = [
                E.ToolResult(t_ms=t + 30, call_id=call, status="executed", hand_back="checkout")
            ]
        elif recipe[0] == "barge":
            events = _barge_events(recipe, s, t)
        elif recipe[0] == "stop":
            events = [rows.stop(t + 5, s.turn_id if recipe[1] == "current" else "zz")]
        else:
            parts = ti._l8_expand(recipe, s)
            if parts is not None:
                queue[:0] = parts
                continue
            events = ti._l8_events(recipe, s, t, spoken, calls)
        for ev in events:
            if n >= max_events:
                break
            n += 1
            t = max(t, ev.t_ms)
            new, cmds = live.step(s, ev)
            base_new, base_cmds = old.step(s, ev)
            if tuple(cmds) != tuple(base_cmds) or rows.base_fields(new) != rows.base_fields(
                base_new
            ):
                diverged.append(
                    f"{ev.type} in {s.state.value}: {ti.kinds(cmds)} vs BASE {ti.kinds(base_cmds)}"
                )
                if not any(r[0] == "stop" for r in recipes):
                    raise AssertionError(
                        f"A0-8: {diverged[-1]} (state equal: {rows.base_fields(new) == rows.base_fields(base_new)})"
                    )
            if visits is not None:
                _count(visits, s, ev, new)
            spoken += "".join(c.source for c in cmds if isinstance(c, C.TtsSpeak))
            s = new
            if s.state is TurnState.IDLE:
                spoken = ""
    return diverged


def _machines(cell: str) -> tuple[TurnMachine, Any]:
    kw = {"thinking": True, **DIFF_POLICIES[cell]}
    frozen = _frozen()
    return TurnMachine(TurnPolicy(**kw)), frozen.TurnMachine(frozen.TurnPolicy(**kw))


@pytest.mark.parametrize("cell", sorted(DIFF_POLICIES))
def test_differential_without_user_stop(cell: str) -> None:
    """A0-8: with no ``UserStop``, ``step`` returns exactly BASE's commands and BASE's state fields, under six
    policies. Regime: 200 derandomized examples of 4-40 recipes (at most 60 events), database off, plus examples."""
    live, old = _machines(cell)

    @settings(max_examples=200, deadline=None, derandomize=True, database=None)
    @given(recipes=st.lists(_DIFF_EV, min_size=4, max_size=40))
    @example(
        recipes=[
            ("barge", "takeover", "stopped", False),
            ("turn",),
            ("speak", "Yes. "),
            ("tick", 2000),
        ]
    )
    @example(
        recipes=[
            ("barge", "long-offset", "timeout", True),
            ("barge", "word-in-pause", "none", False),
            ("tick", 400),
        ]
    )
    @example(
        recipes=[
            ("barge", "takeover-tick", "none", True),
            ("stopped", 0.5),
            ("turn",),
            ("drained",),
        ]
    )
    @example(
        recipes=[
            ("tool_complete", "Yes. ", "valid", False),
            ("result", "last", "executed"),
            ("tick", 7000),
        ]
    )
    @example(
        recipes=[("tool_think", "repairable"), ("pause", "Yes. "), ("takeover",), ("deadline",)]
    )
    def run_trace(recipes: list[Any]) -> None:
        _diff_trace(live, old, recipes)

    run_trace()


def _reach(cell: str, visits: dict[str, int]) -> None:
    live, old = _machines(cell)

    @settings(max_examples=200, deadline=None, derandomize=True, database=None)
    @given(recipes=st.lists(_DIFF_EV, min_size=4, max_size=40))
    def run_trace(recipes: list[Any]) -> None:
        _diff_trace(live, old, recipes, visits=visits)

    run_trace()


def test_the_traces_reach_the_rows_a0_touches() -> None:
    """Power of A0-8 (measured, not assumed): over the default and the loop-3 cells, the traces reach SPEAKING ->
    INTERRUPTED, PAUSED -> INTERRUPTED, both INTERRUPTED exits (``playback.stopped`` and the timeout ``tick``), THINKING
    awaiting a result, THINKING with the repair live and THINKING with a clip playing. Regime: 200 derandomized
    examples per cell, database off."""
    visits: dict[str, int] = {}
    for cell in ("default", "loop-3-still-working"):
        _reach(cell, visits)
    for key in (
        "speaking->interrupted",
        "paused->interrupted",
        "interrupted-exit:playback.stopped",
        "interrupted-exit:tick",
        "thinking-awaiting",
        "thinking-repair",
        "thinking-clip",
    ):
        assert visits.get(key, 0) > 0, f"A0-8 power: no trace reached {key!r}; visits {visits}"


@pytest.mark.parametrize(
    "recipes",
    [
        [("turn",), ("stop", "current")],
        [("turn",), ("pause", "Yes. "), ("stop", "current")],
        [("turn_silent",), ("stop", "current")],
        [("tool_complete", "Yes. ", "valid", True), ("stop", "current")],
    ],
    ids=["speaking", "paused", "thinking", "thinking-awaiting"],
)
def test_anti_vacuity_a_stop_diverges_from_base(recipes: list[Any]) -> None:
    """Anti-vacuity: the harness above is able to see A0. A stop naming the current turn in SPEAKING, PAUSED, THINKING
    and THINKING awaiting a result (loop on) makes the live machine diverge from BASE, which has no row for it; a stop
    naming another turn diverges nowhere."""
    cell = "loop-2" if recipes[0][0] == "tool_complete" else "no-ack"
    live, old = _machines(cell)
    diverged = _diff_trace(live, old, recipes)
    assert diverged and diverged[0].startswith("user.stop in "), diverged
    assert _diff_trace(live, old, [*recipes[:-1], ("stop", "other")]) == []


def test_frozen_copy_is_base_verbatim() -> None:
    """A0-8's oracle is only an oracle while nobody edits it: the body hashes to the hand-computed sha and, when the
    base commit is in the clone, equals ``turn_machine.py`` there from its line 18 on."""
    frozen = _frozen()
    text = pathlib.Path(frozen.__file__).read_text(encoding="utf-8")
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == rows.T0193_FROZEN_BODY_SHA256
    assert frozen.BASE_SHA == rows.T0193_BASE
    shown = subprocess.run(
        ["git", "show", f"{rows.T0193_BASE}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"base commit {rows.T0193_BASE} is not in this clone")
    assert body == "".join(shown.stdout.splitlines(keepends=True)[17:]).rstrip("\n") + "\n"
