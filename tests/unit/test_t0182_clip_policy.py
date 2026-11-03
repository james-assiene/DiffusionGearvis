"""T-0182 group B, the machine: ack off by default (B1), still_working only in long live tool loops (B4), the rules
between clips (B5) and the new invariants F1-F4 (B8) (docs/tasks/T-0182.md, group B and its rulings B-R1..B-R4).

**The surface these rows pin** (from B1/B4 as redesigned by B-R4; reported to the orchestrator):

- ``TurnPolicy.ack_clip: bool = True`` and ``TurnPolicy.still_working_after_ms: int | None = None``: ``TurnPolicy()``
  keeps BASE_B's behaviour (B-R4). A missing field is a named ``AssertionError`` (``clip_policy``), never a
  ``TypeError`` inside a row.
- "By default" in B1/B5/F1/F2 means "in a ``diffusion_gearvis session`` started with defaults" (B-R4): the policy
  ``cli`` resolves, ``ack_clip=False`` everywhere and ``still_working_after_ms=20000`` in live with the loop on.
  ``CLI_DEFAULT`` below is that resolution written out by hand; ``test_t0182_cli`` pins that ``cli`` really builds it
  (``test_still_working_after_ms_by_mode``, ``test_the_startup_line_in_every_mode``).
- ``PlayClip(kind="still_working")`` is the only command of the tick that plays it (no state change, so no
  ``hud.state``), and the state stays THINKING. The new "already played this turn" attribute is **not** named here:
  ``b4_field`` finds it as the one name ``MachineState`` has beyond BASE_B's fields (B-R5: ``Field(exclude=True)`` or a
  private attribute); the B4 rows read it with ``getattr`` and pin that it is absent from every dump.
- "An ending clip" is ``reflex.clips.ENDING_KINDS`` (amendment B-R1).

**Readings (reported):**

- F2's "any ``StopClip`` caused only by it": a ``StopClip`` BASE emits only because ``clip_playing`` was set (the
  THINKING→SPEAKING commit row, R15's voiced report step, the reset row in THINKING). The THINKING onset row's
  ``StopClip`` is unconditional at BASE, so it is not caused by the ack and stays in the default output. The F2b
  oracle is therefore "the frozen BASE_B machine, stepped on the default machine's own state, with its
  ``PlayClip(ack)`` removed and the two state fields only the ack moves (``fillers_this_turn``, ``clip_playing``) put
  back" — computed from BASE's code and the criterion, not from the implementation.
- B4's guard list is read as exhaustive: on a tick where every listed guard holds and the machine has nothing else
  to do, the clip **must** play (the guard-too-wide direction, ``.claude/rules/tests.md``). That includes a THINKING
  turn whose ack (``ack_clip=True``) was stopped by an onset.
- B4 "takeover emits StopClip" is pinned for onset, reset and THINKING→SPEAKING only (amendment B-R1).

Mutants each named row kills (docs/tasks/T-0182.md "Tests to write"): the deadline from the last tool result
(``test_deadline_counts_from_llm_start_not_from_the_last_result``), the row firing in SPEAKING
(``test_never_in_speaking_or_paused``), the row incrementing ``fillers_this_turn``
(``test_still_working_leaves_fillers_this_turn``). The task's fourth mutant, "``ack_clip`` defaulting to True", is
the product default after B-R4: here ``test_policy_defaults`` pins the contract default (``TurnPolicy()`` is BASE's,
so the mutant is now "``TurnPolicy()`` defaulting to False", which ``test_f2a_*`` also kills), and the CLI's off
default is pinned in ``test_t0182_cli``.

Activated for ``/implement T-0182`` group B (the rows were pending through e1ca276f): at BASE_B (7646b8b6)
``TurnPolicy`` has neither field, so each fails on the ``clip_policy`` assertion (F2a included: identity on
``TurnPolicy()`` is checked only once the two fields exist).

Regime: CPU, pure ``step``; Hypothesis traces over ``test_turn_invariants``' L8 recipes (up to 60 events), sizes and
example counts stated per row; n = 1 per hand row.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import MachineState, TurnState
from tests.unit import _frozen_turn_machine_main as frozen_main
from tests.unit import _frozen_turn_machine_t0089 as frozen_t0089
from tests.unit import _frozen_turn_machine_t0182 as frozen
from tests.unit import _t0182_pins as pins
from tests.unit import test_turn_invariants as ti

#: B1/B4 contract defaults after B-R4: BASE_B's behaviour (ack on, no still_working)
NEW_POLICY_DEFAULTS: dict[str, Any] = {"ack_clip": True, "still_working_after_ms": None}
#: B-R4: what ``cli`` resolves with defaults (filler clips off); a live loop adds ``still_working_after_ms`` (B4)
CLI_DEFAULT: dict[str, Any] = {"ack_clip": False}
#: ``Profile.still_working_after_s`` default and minimum (B4: 20, ``ge=5``), in ms
DEFAULT_W = 20_000
MIN_W = 5_000
ENDING = pins.ENDING_KINDS
VALID_SEARCH = {
    "call_id": "c-search",
    "status": "valid",
    "name": "google_search",
    "arguments": {"query": "x"},
}


def clip_policy(**overrides: Any) -> TurnPolicy:
    """``TurnPolicy(thinking=True, **overrides)``, or a named failure while B1/B4's fields do not exist."""
    fields = getattr(TurnPolicy, "__dataclass_fields__", {})
    missing = [name for name in NEW_POLICY_DEFAULTS if name not in fields]
    assert not missing, f"T-0182 B1/B4: TurnPolicy gains {missing} (defaults {NEW_POLICY_DEFAULTS})"
    overrides.setdefault("thinking", True)
    return TurnPolicy(**overrides)


def cli_default(**overrides: Any) -> TurnPolicy:
    """The policy ``cli`` resolves with defaults (B-R4: ``CLI_DEFAULT``, filler clips off), with ``overrides``."""
    return clip_policy(**{**CLI_DEFAULT, **overrides})


def loop(w: int | None = DEFAULT_W, **overrides: Any) -> TurnMachine:
    """A live-loop machine as ``cli`` builds it with defaults (B4, B-R4): loop on, ack off,
    ``still_working_after_ms = w``; ``ack_clip=True`` is the filler-clips-on cell."""
    return TurnMachine(cli_default(tool_loop=True, still_working_after_ms=w, **overrides))


def plays(cmds: Any, kind: str | None = None) -> list[str]:
    return [c.kind for c in cmds if isinstance(c, C.PlayClip) and (kind is None or c.kind == kind)]


def is_sw(c: Any) -> bool:
    return isinstance(c, C.PlayClip) and c.kind == "still_working"


def start(
    m: TurnMachine, state: MachineState | None = None, *, at: int = 1000
) -> tuple[MachineState, Any, int]:
    """IDLE -> THINKING; returns the state, the commands and ``t0``, the time of the tick that emitted llm.start."""
    s, cmds, t0 = ti.drive_to_thinking(
        m, onset_ms=at, t_eot_ms=at + 1600, words=[("open", at + 100)], state=state
    )
    assert s.state is TurnState.THINKING and any(isinstance(c, C.LlmStart) for c in cmds), "premise"
    return s, cmds, t0


def executed(m: TurnMachine, s: MachineState, t: int, call_id: str = "c-search") -> MachineState:
    """Step 1 ends with a valid call: R11 executes it and the turn waits for the result in THINKING."""
    draft = ti.tool_draft(**{**VALID_SEARCH, "call_id": call_id})
    s, cmds = m.step(s, ti.llm_done(t, s.active_generate, draft))
    assert s.state is TurnState.THINKING and s.awaiting_result == call_id, (
        "premise: R11 executed the call"
    )
    assert any(isinstance(c, C.ToolExecute) for c in cmds)
    return s


def result(m: TurnMachine, s: MachineState, t: int) -> MachineState:
    """The awaited result: step 2's generate is live (R12)."""
    s, cmds = m.step(s, E.ToolResult(t_ms=t, call_id=s.awaiting_result, status="dry_run"))
    assert any(type(c).__name__ == "LlmContinue" for c in cmds), "premise: R12 started step 2"
    return s


def tick(m: TurnMachine, s: MachineState, t: int) -> tuple[MachineState, tuple[Any, ...]]:
    new, cmds = m.step(s, E.Tick(t_ms=t))
    return new, tuple(cmds)


# =====================================================================================================================
# B1 / B4: the policy fields and their defaults (mutant 4: ack_clip defaulting to True)
# =====================================================================================================================


def test_policy_defaults() -> None:
    """B1, B4 as redesigned by B-R4: ``TurnPolicy()`` keeps BASE's behaviour, ``ack_clip`` True and
    ``still_working_after_ms`` None, so every rig that builds ``TurnPolicy()`` is unchanged; the product default (ack
    off) is ``cli``'s (``test_t0182_cli``). The two fields are independent switches: setting one leaves the other's
    default."""
    policy = clip_policy()
    assert TurnPolicy().ack_clip is True, "T-0182 B-R4: TurnPolicy() keeps BASE's ack"
    assert TurnPolicy().still_working_after_ms is None, (
        "T-0182 B4: no still_working unless cli sets it"
    )
    assert (policy.ack_clip, policy.still_working_after_ms) == (True, None)
    assert TurnPolicy(ack_clip=False).still_working_after_ms is None
    assert TurnPolicy(still_working_after_ms=DEFAULT_W).ack_clip is True
    assert cli_default().ack_clip is False and cli_default().still_working_after_ms is None


@pytest.mark.parametrize(
    ("thinking", "ema", "ack_on", "want"),
    [
        # by hand from docs/02 §4 row 6: ack iff ack_clip and (thinking on or EMA unknown or EMA > 300 ms)
        (True, 200, True, True),
        (False, None, True, True),
        (False, 200, True, False),
        (False, 300, True, False),
        (False, 301, True, True),
        (True, 200, False, False),
        (False, None, False, False),
        (False, 301, False, False),
    ],
)
def test_ack_iff_ack_clip_and_the_existing_condition(
    thinking: bool, ema: int | None, ack_on: bool, want: bool
) -> None:
    """B1: the llm.start row emits ``PlayClip(ack)`` iff ``ack_clip`` is set and the existing condition holds; nothing
    else on the row moves (the frozen BASE_B machine is the oracle for the rest of the tuple and the state)."""
    m = TurnMachine(clip_policy(thinking=thinking, ack_clip=ack_on))
    old = frozen.TurnMachine(frozen.TurnPolicy(thinking=thinking))
    seed = MachineState(first_canvas_ms=ema)
    s, cmds, _ = start(m, seed)
    base_s, base_cmds, _ = ti.drive_to_thinking(
        old, onset_ms=1000, t_eot_ms=2600, words=[("open", 1100)], state=seed
    )
    assert plays(cmds) == (["ack"] if want else []), (
        f"B1: thinking={thinking} ema={ema} ack_clip={ack_on}"
    )
    others = [c for c in cmds if not isinstance(c, C.PlayClip)]
    base_others = [c for c in base_cmds if not isinstance(c, C.PlayClip)]
    assert others == base_others, "B1: the row's other commands are BASE_B's"
    assert s.clip_playing is want and s.fillers_this_turn == int(want)
    assert pins.base_state(s) | {
        "clip_playing": None,
        "fillers_this_turn": None,
    } == pins.base_state(base_s) | {
        "clip_playing": None,
        "fillers_this_turn": None,
    }


# =====================================================================================================================
# B4: each guard, discriminating both ways
# =====================================================================================================================


@pytest.mark.parametrize("w", [MIN_W, DEFAULT_W])
def test_the_deadline_edge_is_inclusive(w: int) -> None:
    """B4 (``t_ms - llm_start_ms >= still_working_after_ms``): nothing at ``w - 1``, exactly one
    ``PlayClip(still_working)`` at ``w``, the state still THINKING, and it is the tick's only command. For ``w`` =
    20 s the result has arrived (``tool_result`` would fire at 16 s) and step 2 is generating: the live t9 shape."""
    m = loop(w)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    if w > 15_000:
        s = result(m, s, t0 + 3000)
    s, early = tick(m, s, t0 + w - 1)
    assert early == (), f"B4: nothing before {w} ms; got {ti.kinds(early)}"
    s, cmds = tick(m, s, t0 + w)
    assert cmds == (C.PlayClip(kind="still_working"),), (
        f"B4: one still_working at {w} ms; got {cmds}"
    )
    assert s.state is TurnState.THINKING and s.clip_playing is True, "B4: it sets clip_playing"


def test_deadline_counts_from_llm_start_not_from_the_last_result() -> None:
    """B4 mutant 1: the result arrives 15 s in, step 2 generates; the clip plays at llm.start + 20 s, 5 s after the
    result (a deadline from the result, or from the execute, would wait until 35 s or 21 s)."""
    m = loop(DEFAULT_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s = result(m, s, t0 + 15_000)
    s, cmds = tick(m, s, t0 + DEFAULT_W)
    assert plays(cmds) == ["still_working"], (
        f"B4: the deadline is llm.start + 20 s; got {ti.kinds(cmds)}"
    )


def test_not_before_a_call_has_executed() -> None:
    """B4 (``tool_steps_executed >= 1``): a loop turn still on step 1 plays nothing at ``w`` or later (up to the
    llm timeout); its twin with one executed call plays at ``w`` (``test_the_deadline_edge_is_inclusive``)."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    for t in (t0 + MIN_W, t0 + 12_000, t0 + 19_999):
        s, cmds = tick(m, s, t)
        assert plays(cmds) == [], f"B4: no call executed yet; got {ti.kinds(cmds)} at +{t - t0}"
    assert s.state is TurnState.THINKING


def test_none_disables_it_and_loop_off_never_reaches_it() -> None:
    """B4/F4: ``still_working_after_ms=None`` never plays, however long the loop turn runs; with the loop off (and a
    threshold set) no call executes mid-turn, so it never plays either."""
    m = loop(None)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s = result(m, s, t0 + 3000)
    for t in range(t0 + 4000, t0 + 23_000, 1000):
        s, cmds = tick(m, s, t)
        assert plays(cmds) == [], f"None: nothing; got {ti.kinds(cmds)}"
    off = TurnMachine(cli_default(tool_loop=False, still_working_after_ms=MIN_W))
    s, _, t0 = start(off)
    for t in range(t0 + 1000, t0 + 19_000, 1000):
        s, cmds = tick(off, s, t)
        assert plays(cmds) == []


def test_never_in_speaking_or_paused() -> None:
    """B4 mutant 2 (the row firing in SPEAKING): the report step was voiced (SPEAKING, no drain yet), then a pause;
    ticks past ``w`` play nothing in either state."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s = result(m, s, t0 + 1500)
    s, _ = m.step(s, ti.commit(s, "It is done. ", t=t0 + 2000))
    s, cmds = m.step(s, ti.llm_done(t0 + 2500, s.active_generate))
    assert s.state is TurnState.SPEAKING and any(isinstance(c, C.TtsSpeak) for c in cmds), (
        "premise: voiced"
    )
    s, cmds = tick(m, s, t0 + MIN_W + 100)
    assert plays(cmds) == [], f"B4: THINKING only; SPEAKING played {ti.kinds(cmds)}"
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t0 + MIN_W + 150, onset_ms=t0 + MIN_W + 150))
    assert s.state is TurnState.PAUSED, "premise"
    s, cmds = tick(m, s, t0 + MIN_W + 200)
    assert plays(cmds) == [], f"B4: THINKING only; PAUSED played {ti.kinds(cmds)}"


def test_still_working_leaves_fillers_this_turn() -> None:
    """B4 mutant 3: ``fillers_this_turn`` is unchanged by the row (0 with ack off, 1 with ack on and the ack stopped)."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"] and s.fillers_this_turn == 0, (
        plays(cmds),
        s.fillers_this_turn,
    )
    m = loop(MIN_W, ack_clip=True)
    s, cmds, t0 = start(m)
    assert plays(cmds) == ["ack"] and s.fillers_this_turn == 1, "premise: the ack counts once"
    s = executed(m, s, t0 + 1000)
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t0 + 1500, onset_ms=t0 + 1500))
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"] and s.fillers_this_turn == 1, (
        plays(cmds),
        s.fillers_this_turn,
    )


def test_not_while_a_clip_plays_and_yes_once_it_was_stopped() -> None:
    """B4 (no clip playing), both directions: with ``ack_clip=True`` the ack is still marked playing at ``w`` and
    nothing plays; an onset stops it (``StopClip``) and the next tick past ``w`` plays still_working."""
    m = loop(MIN_W, ack_clip=True)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W + 500)
    assert plays(cmds) == [], f"B4: the ack is playing; got {ti.kinds(cmds)}"
    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=t0 + MIN_W + 600, onset_ms=t0 + MIN_W + 600))
    assert any(isinstance(c, C.StopClip) for c in cmds) and s.clip_playing is False, "premise"
    s, cmds = tick(m, s, t0 + MIN_W + 650)
    assert plays(cmds) == ["still_working"], f"B4: every guard holds now; got {ti.kinds(cmds)}"


def test_once_per_turn_even_after_it_was_stopped() -> None:
    """B4 (no still_working has played this turn): after it played and an onset stopped it, later ticks of the same
    turn play nothing (``clip_playing`` alone would let it play again)."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"]
    for t in (t0 + MIN_W + 50, t0 + MIN_W + 100):
        s, cmds = tick(m, s, t)
        assert plays(cmds) == []
    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=t0 + MIN_W + 200, onset_ms=t0 + MIN_W + 200))
    assert any(isinstance(c, C.StopClip) for c in cmds), (
        "B4/G6: the onset stops the playing still_working"
    )
    assert sw_played(s) and not s.clip_playing, "B4: stopped, but it has played this turn"
    s = result(m, s, t0 + MIN_W + 300)
    for t in range(t0 + MIN_W + 400, t0 + MIN_W + 4000, 500):
        s, cmds = tick(m, s, t)
        assert plays(cmds) == [], f"B4: once per turn; got {ti.kinds(cmds)}"


def _finish_after_report(m: TurnMachine, s: MachineState, t: int) -> MachineState:
    s = result(m, s, t)
    s, _ = m.step(s, ti.commit(s, "Here it is. ", t=t + 100))
    s, _ = m.step(s, ti.llm_done(t + 200, s.active_generate))
    assert s.state is TurnState.SPEAKING, "premise: the report is voiced"
    s, _ = m.step(s, E.PlaybackDrained(t_ms=t + 900, played_ms=700, played_text="Here it is. "))
    assert s.state is TurnState.IDLE, "F3: the turn reaches IDLE"
    return s


def test_cleared_at_turn_end_and_on_reset() -> None:
    """B4 (the new field is cleared at turn end and on reset): turn 1 plays it and completes; turn 2 plays it again;
    a reset in THINKING after it played stops it, and turn 3 plays it again."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"]
    s = _finish_after_report(m, s, t0 + MIN_W + 100)
    s, _, t1 = start(m, s, at=t0 + 10_000)
    s = executed(m, s, t1 + 1000, call_id="c2")
    s, cmds = tick(m, s, t1 + MIN_W)
    assert plays(cmds) == ["still_working"], f"B4: cleared at turn end; turn 2 got {ti.kinds(cmds)}"
    s, cmds = m.step(s, E.SessionReset(t_ms=t1 + MIN_W + 100))
    k = ti.kinds(cmds)
    assert "stop_clip" in k and k.index("stop_clip") < k.index("reset_session"), (
        f"B4/G7: reset stops it; {k}"
    )
    s, _, t2 = start(m, s, at=t1 + 10_000)
    s = executed(m, s, t2 + 1000, call_id="c3")
    s, cmds = tick(m, s, t2 + MIN_W)
    assert plays(cmds) == ["still_working"], f"B4: cleared on reset; turn 3 got {ti.kinds(cmds)}"


def b4_field() -> str:
    """B4's one new state attribute (B-R5: a ``MachineState`` field declared ``Field(exclude=True)``, or a private
    attribute): the one name ``MachineState`` has beyond BASE_B's fields. Discovered by set difference, read with
    ``getattr`` only, never invoked."""
    # T-0193 A0-1 (census move): A0's per-turn stop field is the name beyond fdb9fc01's fields (which include B4's);
    # it is set aside by that name difference and pinned (exactly one) by test_t0193_stop_row.stop_field
    from tests.unit.test_t0193_stop_row import BASE_STATE_FIELDS as T0193_BASE_FIELDS

    t0193 = set(MachineState.model_fields) - set(T0193_BASE_FIELDS)
    assert len(t0193) == 1, f"T-0193 A0-1: exactly one field beyond fdb9fc01's; {sorted(t0193)}"
    fields = [
        n for n in MachineState.model_fields if n not in pins.BASE_B_STATE_FIELDS and n not in t0193
    ]
    private = list(getattr(MachineState, "__private_attributes__", {}))
    found = fields + private
    assert len(found) == 1, (
        f"T-0182 B4/B-R5: MachineState gains exactly one attribute ('a still_working has played this turn'); "
        f"found {found}"
    )
    return found[0]


def sw_played(state: MachineState) -> Any:
    return getattr(state, b4_field())


def test_b4_field_is_read_directly_and_excluded_from_dumps() -> None:
    """B4 / B-R5: the new attribute is falsy at the start of a turn, truthy from the tick that plays
    ``still_working``, falsy again after the turn ends and after a reset; and it is excluded from every dump
    (``model_dump()``, ``model_dump(mode="json")``, ``model_dump_json()``), whose keys stay BASE_B's fields in order,
    so goldens and BASE differentials are unchanged. A field (not a private attribute) is declared
    ``exclude=True``."""
    name = b4_field()
    if name in MachineState.model_fields:
        assert MachineState.model_fields[name].exclude is True, (
            f"T-0182 B-R5: {name} is declared Field(exclude=True)"
        )
    m = loop(MIN_W)
    s, _, t0 = start(m)
    assert not sw_played(s), "B4: nothing has played at llm.start"
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"] and sw_played(s), (
        f"B4: the tick that plays still_working sets {name}"
    )
    want = list(pins.BASE_B_STATE_FIELDS)
    assert list(s.model_dump()) == want, f"T-0182 B-R5: model_dump() keys {list(s.model_dump())}"
    assert list(s.model_dump(mode="json")) == want
    assert list(json.loads(s.model_dump_json())) == want
    done = _finish_after_report(m, s, t0 + MIN_W + 100)
    assert not sw_played(done), f"B4: {name} is cleared at turn end"
    s, _, t1 = start(m, done, at=t0 + 10_000)
    s = executed(m, s, t1 + 1000, call_id="c2")
    s, _ = tick(m, s, t1 + MIN_W)
    assert sw_played(s), "premise: turn 2 played it"
    s, _ = m.step(s, E.SessionReset(t_ms=t1 + MIN_W + 100))
    assert not sw_played(s), f"B4: {name} is cleared on reset"


def test_thinking_to_speaking_stops_it_first() -> None:
    """B4 (amendment B-R1: StopClip on onset, reset and THINKING→SPEAKING): the voiced report step after a playing
    still_working begins with ``StopClip``, then speaks; history holds the played report only."""
    m = loop(MIN_W)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"]
    s = result(m, s, t0 + MIN_W + 100)
    s, _ = m.step(s, ti.commit(s, "Here it is. ", t=t0 + MIN_W + 200))
    s, cmds = m.step(s, ti.llm_done(t0 + MIN_W + 300, s.active_generate))
    assert s.state is TurnState.SPEAKING
    assert isinstance(cmds[0], C.StopClip), (
        f"B4: THINKING→SPEAKING stops the clip first; {ti.kinds(cmds)}"
    )
    s, _ = m.step(
        s, E.PlaybackDrained(t_ms=t0 + MIN_W + 900, played_ms=600, played_text="Here it is. ")
    )
    assert s.state is TurnState.IDLE and s.history[-1].assistant_spoken == "Here it is. ", s.history


@pytest.mark.parametrize("budget_end", ["out_of_steps", "not_finished"])
def test_an_ending_clip_after_still_working_still_plays(budget_end: str) -> None:
    """B5: still_working at ``w``, then step 2 ends with a valid call past the budget (``out_of_steps``) or a refused
    draft (``not_finished``): the ending clip still plays, once, and the turn ends; no clip text in history; later
    ticks play nothing (no clip after IDLE)."""
    m = loop(MIN_W, max_tool_steps=1)
    s, _, t0 = start(m)
    s = executed(m, s, t0 + 1000)
    s, cmds = tick(m, s, t0 + MIN_W)
    assert plays(cmds) == ["still_working"]
    s = result(m, s, t0 + MIN_W + 100)
    if budget_end == "out_of_steps":
        draft = ti.tool_draft(**{**VALID_SEARCH, "call_id": "c-2"})
    else:
        draft = ti.tool_draft(call_id="c-2", status="refused", reason="no tool name")
    s, cmds = m.step(s, ti.llm_done(t0 + MIN_W + 200, s.active_generate, draft))
    assert plays(cmds) == [budget_end], (
        f"B5: the ending clip plays after a still_working; {ti.kinds(cmds)}"
    )
    assert s.state is TurnState.IDLE
    assert all("still working" not in h.assistant_spoken for h in s.history), s.history
    for t in (t0 + MIN_W + 300, t0 + 2 * MIN_W + 300):
        s, cmds = tick(m, s, t)
        assert plays(cmds) == []


# =====================================================================================================================
# F1-F4: properties over the L8 recipes (Hypothesis)
# =====================================================================================================================

#: a long tick (the deadline guard needs seconds, the L8 ticks are at most 0.8 s) and the session reset
_LONG = (st.tuples(st.just("tick"), st.integers(1000, 7000)),)
_RESET = (st.just(("reset",)),)
_F_EV = st.one_of(ti._L8_EV_RESULTS, *(_LONG * 3), *_RESET)
#: by-construction traces (examples): a valid call, then a long wait (still_working at w <= 6 s); the onset that
#: stops it; the voiced report; the budget ending; the reset
_SW = [("tool_think", "valid"), ("tick", 6000)]
F3_EXAMPLES = (
    _SW,
    [*_SW, ("onset",), ("tick", 2000), ("tick", 2000)],
    [
        *_SW,
        ("result", "last", "dry_run"),
        ("commit", "Yes. ", "answer", False, False),
        ("done", False),
    ],
    [*_SW, ("result", "last", "dry_run"), ("tool_done", "valid", 0)],
    [*_SW, ("reset",)],
    [
        *_SW,
        ("result", "last", "dry_run"),
        ("commit", "Yes. ", "answer", False, False),
        ("done", False),
        ("drained",),
        *_SW,
    ],
)


def _trace(
    m: TurnMachine, recipes: list[Any], on_step: Any, *, max_events: int = 60
) -> MachineState:
    """Drive ``m`` through ``recipes`` (``test_turn_invariants``' L8 expansion, plus ``("reset",)``), calling
    ``on_step(before, event, after, commands)`` on every step."""
    s = m.initial()
    t, spoken, calls, n = 0, "", [], 0
    queue = list(recipes)
    while queue and n < max_events:
        recipe = queue.pop(0)
        if recipe[0] == "reset":
            events: list[Any] = [E.SessionReset(t_ms=t + 5)]
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
            new, cmds = on_step(s, ev)
            spoken += "".join(c.source for c in cmds if isinstance(c, C.TtsSpeak))
            s = new
            if s.state is TurnState.IDLE:
                spoken = ""
    return s


@pytest.mark.parametrize(
    "policy",
    ["default", "default-loop", "no-thinking", "loop-still-working"],
)
def test_f1_with_defaults_no_ack_or_hmm(policy: str) -> None:
    """F1: with the default policy as ``cli`` resolves it (B-R4: ``CLI_DEFAULT``; and with the loop on, thinking off
    with an unknown EMA, and a live loop's threshold) no ``PlayClip(ack)`` or ``PlayClip(hmm)`` is ever emitted.
    Regime: 150 examples of 4-40 L8 recipes, at most 60 events, per policy."""
    pol = {
        "default": lambda: cli_default(),
        "default-loop": lambda: cli_default(tool_loop=True, max_tool_steps=2),
        "no-thinking": lambda: cli_default(thinking=False),
        "loop-still-working": lambda: cli_default(
            tool_loop=True, max_tool_steps=2, still_working_after_ms=MIN_W
        ),
    }[policy]()

    @settings(max_examples=150, deadline=None)
    @given(recipes=st.lists(_F_EV, min_size=4, max_size=40))
    @example(recipes=[("turn",), ("tick", 3000)])
    def run(recipes: list[Any]) -> None:
        m = TurnMachine(pol)

        def step(s: MachineState, ev: Any) -> Any:
            new, cmds = m.step(s, ev)
            bad = plays(cmds, "ack") + plays(cmds, "hmm")
            assert not bad, f"F1: {bad} on {ev.type} in {s.state.value} with {policy}"
            return new, cmds

        _trace(m, recipes, step)

    run()


def _frozen_policy(**kw: Any) -> Any:
    return frozen.TurnPolicy(thinking=kw.pop("thinking", True), **kw)


@pytest.mark.parametrize(
    "cell", ["loop-off", "loop-on-1", "loop-on-2", "loop-on-3", "no-thinking"], ids=lambda c: c
)
def test_f2a_turn_policy_default_is_the_frozen_base_b_machine(cell: str) -> None:
    """F2 (second sentence) as B-R4 states it, identity on ``TurnPolicy()`` itself: a policy built with none of B's
    fields (so ``ack_clip=True``, ``still_working_after_ms=None``) gives a machine equal to the frozen BASE_B machine
    on every sequence: commands exactly, and the state on every BASE_B field (``_t0182_pins``). This is what keeps
    every rig that builds ``TurnPolicy()`` unchanged. Regime: 200 examples of 4-40 recipes (L8's, with results, long
    ticks and resets), at most 60 events, per cell."""
    kw: dict[str, Any] = {
        "loop-off": {},
        "loop-on-1": {"tool_loop": True, "max_tool_steps": 1},
        "loop-on-2": {"tool_loop": True, "max_tool_steps": 2},
        "loop-on-3": {"tool_loop": True, "max_tool_steps": 3},
        "no-thinking": {"thinking": False},
    }[cell]
    base_kw = {"thinking": True, **kw}
    explicit = clip_policy(ack_clip=True, still_working_after_ms=None, **dict(base_kw))
    assert TurnPolicy(**base_kw) == explicit, (
        "T-0182 B-R4: TurnPolicy() without B's fields is the ack-on, no-still_working policy"
    )
    live = TurnMachine(TurnPolicy(**base_kw))
    old = frozen.TurnMachine(_frozen_policy(**dict(kw)))

    @settings(max_examples=200, deadline=None)
    @given(recipes=st.lists(_F_EV, min_size=4, max_size=40))
    def run(recipes: list[Any]) -> None:
        def step(s: MachineState, ev: Any) -> Any:
            new, cmds = live.step(s, ev)
            base_new, base_cmds = old.step(s, ev)
            assert tuple(cmds) == tuple(base_cmds), (
                f"F2: {ev.type} in {s.state.value}: {ti.kinds(cmds)} != BASE_B {ti.kinds(base_cmds)}"
            )
            assert pins.base_state(new) == pins.base_state(base_new), (
                f"F2: {ev.type} in {s.state.value}"
            )
            return new, cmds

        _trace(live, recipes, step)

    run()


def _without_ack(old: Any, s: MachineState, ev: Any) -> tuple[MachineState, tuple[Any, ...]]:
    """The F2b oracle (module docstring): BASE_B's step on ``s`` with its ``PlayClip(ack)`` and the two fields only the
    ack moves undone. ``s`` comes from the default machine, so ``clip_playing`` is never set in it and BASE_B emits
    no ``StopClip`` caused only by an ack."""
    new, cmds = old.step(s, ev)
    acks = [c for c in cmds if isinstance(c, C.PlayClip) and c.kind == "ack"]
    if acks:
        assert len(acks) == 1
        cmds = [c for c in cmds if c not in acks]
        new = new.model_copy(
            update={"clip_playing": False, "fillers_this_turn": new.fillers_this_turn - 1}
        )
    return new, tuple(cmds)


@pytest.mark.parametrize("cell", ["loop-off", "loop-on-2", "no-thinking"])
def test_f2b_default_is_ack_on_without_the_ack(cell: str) -> None:
    """F2 (first sentence), lockstep: the default machine (``cli``'s resolution, B-R4: ``CLI_DEFAULT``) has output
    equal to BASE_B's with every ``PlayClip(ack)`` removed, along with any ``StopClip`` caused only by it (module
    docstring), and the state equal on every BASE_B field. With the default state ``clip_playing`` is never set, so
    the default emits no ``StopClip`` but the unconditional onset one. Regime: 200 examples of 4-40 recipes, at most
    60 events, per cell."""
    kw: dict[str, Any] = {
        "loop-off": {},
        "loop-on-2": {"tool_loop": True, "max_tool_steps": 2},
        "no-thinking": {"thinking": False},
    }[cell]
    live = TurnMachine(cli_default(**dict(kw)))
    old = frozen.TurnMachine(_frozen_policy(**dict(kw)))

    @settings(max_examples=200, deadline=None)
    @given(recipes=st.lists(_F_EV, min_size=4, max_size=40))
    @example(
        recipes=[
            ("turn",),
            ("commit", "Yes. ", "answer", False, False),
            ("done", False),
            ("drained",),
        ]
    )
    @example(recipes=[("turn",), ("onset",), ("reset",)])
    def run(recipes: list[Any]) -> None:
        def step(s: MachineState, ev: Any) -> Any:
            new, cmds = live.step(s, ev)
            want_state, want = _without_ack(old, s, ev)
            assert tuple(cmds) == want, (
                f"F2: {ev.type} in {s.state.value}: {ti.kinds(cmds)} != BASE_B without the ack {ti.kinds(want)}"
            )
            assert pins.base_state(new) == pins.base_state(want_state), (
                f"F2: {ev.type} in {s.state.value}"
            )
            assert not new.clip_playing and new.fillers_this_turn == 0
            return new, cmds

        _trace(live, recipes, step)

    run()


class _Ledger:
    """F3's bookkeeping, from the commands alone (nothing read off the implementation's new field)."""

    def __init__(self) -> None:
        self.reset()
        self.counts = {
            "played": 0,
            "blocked_by_clip": 0,
            "stopped_after": 0,
            "ending_after": 0,
            "ending": 0,
        }

    def reset(self) -> None:
        self.start_t: int | None = None
        self.executes = 0
        self.clip_on = False
        self.sw_played = False
        self.ending_played = False


def _f3_step(m: TurnMachine, w: int, led: _Ledger, s: MachineState, ev: Any) -> Any:
    new, cmds = m.step(s, ev)
    cmds = tuple(cmds)
    sw = [c for c in cmds if is_sw(c)]
    where = f"{ev.type} in {s.state.value} at {ev.t_ms} (llm.start {led.start_t}, w {w})"
    assert len(sw) <= 1, f"F3: two still_working in one step: {where}"
    if s.state is TurnState.IDLE:
        assert not plays(cmds), f"F3: no PlayClip follows IDLE: {ti.kinds(cmds)} on {where}"
    due = (
        isinstance(ev, E.Tick)
        and s.state is TurnState.THINKING
        and led.executes >= 1
        and led.start_t is not None
        and ev.t_ms - led.start_t >= w
    )
    if sw:
        assert due, (
            f"F3: still_working off its guards (tick, THINKING, >= 1 executed, >= w): {where}"
        )
        assert not led.clip_on, f"F3: still_working over a playing clip: {where}"
        assert not led.sw_played, f"F3: a second still_working in one turn: {where}"
        assert not led.ending_played, f"F3: still_working after the ending clip: {where}"
        assert new.state is TurnState.THINKING and new.turn_id == s.turn_id, f"F3: {where}"
        assert new.fillers_this_turn == s.fillers_this_turn, f"B4: fillers_this_turn moved: {where}"
        assert not any(isinstance(c, C.HudError) for c in cmds), where
        led.counts["played"] += 1
    elif due and not led.clip_on and not led.sw_played and not led.ending_played and not cmds:
        raise AssertionError(
            f"F3/B4: every guard holds and the tick did nothing (too strict): {where}"
        )
    elif due and led.clip_on and not led.sw_played:
        led.counts["blocked_by_clip"] += 1
    if (
        led.clip_on
        and s.state is TurnState.THINKING
        and (isinstance(ev, (E.UserSpeechOnset, E.SessionReset)) or new.state is TurnState.SPEAKING)
    ):
        assert any(isinstance(c, C.StopClip) for c in cmds), (
            f"B4/G6/G7: no StopClip for a playing clip: {where}"
        )
        if led.sw_played:
            led.counts["stopped_after"] += 1
    endings = [k for k in plays(cmds) if k in ENDING]
    assert len(endings) + int(led.ending_played) <= 1, f"B5: two ending clips in one turn: {where}"
    if endings:
        led.counts["ending"] += 1
        if led.sw_played:
            led.counts["ending_after"] += 1
    for c in cmds:
        if isinstance(c, C.LlmStart):
            led.start_t = ev.t_ms
        elif isinstance(c, C.ToolExecute):
            led.executes += 1
        elif isinstance(c, C.StopClip):
            led.clip_on = False
        elif isinstance(c, C.PlayClip):
            if c.kind in ENDING:
                led.ending_played = True
            else:
                led.clip_on = True
                led.sw_played = led.sw_played or c.kind == "still_working"
    if new.state is TurnState.IDLE or new.turn_id != s.turn_id:
        led.reset()
    return new, cmds


def test_f3_still_working_at_most_once_only_under_its_guards() -> None:
    """F3: ``still_working`` plays at most once per turn, only on a THINKING tick with a call executed and ``w``
    elapsed since llm.start, never over a playing clip, never after the ending clip; on a tick where every guard holds
    and nothing else happens it does play; a playing clip is stopped by onset, reset and THINKING→SPEAKING; at most
    one ending clip per turn; no PlayClip follows IDLE. The ledger is kept from the commands (llm.start's tick,
    ``tool.execute``, ``play_clip``/``stop_clip``), never from the new state field. The by-construction examples
    reach each case; the run must have played it, been blocked by a clip (ack on), stopped it and ended after it.
    Regime: 300 examples of 4-40 recipes (L8's with results, ticks of 1-7 s, resets), at most 60 events; budgets
    1-3, ``w`` 5-6 s, ack on or off; CPU."""
    total = {"played": 0, "blocked_by_clip": 0, "stopped_after": 0, "ending_after": 0, "ending": 0}

    @settings(max_examples=300, deadline=None)
    @given(
        recipes=st.lists(_F_EV, min_size=4, max_size=40),
        budget=st.sampled_from([1, 2, 3]),
        w=st.sampled_from([MIN_W, 6000]),
        ack=st.booleans(),
    )
    @example(recipes=F3_EXAMPLES[0], budget=2, w=MIN_W, ack=False)
    @example(recipes=F3_EXAMPLES[0], budget=2, w=MIN_W, ack=True)
    @example(recipes=F3_EXAMPLES[1], budget=2, w=MIN_W, ack=False)
    @example(recipes=F3_EXAMPLES[2], budget=2, w=MIN_W, ack=False)
    @example(recipes=F3_EXAMPLES[3], budget=1, w=MIN_W, ack=False)
    @example(recipes=F3_EXAMPLES[4], budget=2, w=MIN_W, ack=False)
    @example(recipes=F3_EXAMPLES[5], budget=2, w=MIN_W, ack=False)
    def run(recipes: list[Any], budget: int, w: int, ack: bool) -> None:
        m = TurnMachine(
            clip_policy(
                tool_loop=True, max_tool_steps=budget, still_working_after_ms=w, ack_clip=ack
            )
        )
        led = _Ledger()
        _trace(m, recipes, lambda s, ev: _f3_step(m, w, led, s, ev))
        for k, v in led.counts.items():
            total[k] += v

    run()
    assert all(total.values()), (
        f"F3 is not vacuous: every case is reached at least once; got {total}"
    )


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_f4_p9_unchanged_with_still_working_none(segmenter: str) -> None:
    """F4 (P9): with ``ack_clip=True`` and ``still_working_after_ms=None`` set explicitly, a toolless machine equals
    main's frozen machine (T-0089 P9's oracle) on P9's strategy and examples. Regime: 200 examples of 4-40 recipes
    plus P9's examples, CPU."""
    seg = ti.sentence_segmenter if segmenter == "sentences" else None
    live = TurnMachine(clip_policy(ack_clip=True, still_working_after_ms=None), segment=seg)
    old = frozen_main.TurnMachine(frozen_main.TurnPolicy(thinking=True), segment=seg)
    visits: dict[str, int] = dict.fromkeys([*ti.P9_MIN_VISITS, *ti.P9_MIN_VISITS_SENTENCES], 0)

    @settings(max_examples=200, deadline=None)
    @given(recipes=st.lists(ti._EV, min_size=4, max_size=40))
    def run(recipes: list[Any]) -> None:
        ti._p9_trace(live, old, recipes, visits)

    for recipes in ti.P9_EXAMPLES:
        run = example(recipes=recipes)(run)
    run()


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_f4_l8_unchanged_with_still_working_none(segmenter: str) -> None:
    """F4 (L8): with ``tool_loop=False``, ``ack_clip=True`` and ``still_working_after_ms=None``, the machine equals the
    frozen post-T-0089 machine on L8's strategy (tool drafts, repairs, results): commands exactly and the whole state.
    Regime: 200 examples of 4-40 recipes, at most 60 events, per segmenter, CPU."""
    seg = ti.sentence_segmenter if segmenter == "sentences" else None
    live = TurnMachine(
        clip_policy(ack_clip=True, still_working_after_ms=None, tool_loop=False), segment=seg
    )
    old = frozen_t0089.TurnMachine(frozen_t0089.TurnPolicy(thinking=True), segment=seg)

    @settings(max_examples=200, deadline=None)
    @given(recipes=st.lists(ti._L8_EV_RESULTS, min_size=4, max_size=40))
    def run(recipes: list[Any]) -> None:
        ti._l8_trace(live, old, recipes)

    run()
