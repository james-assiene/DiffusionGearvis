"""T-0173 criterion 3, machine half: R18's ``tool_result`` deadline counts from the event that executed the call.

docs/tasks/T-0173.md. The fix is at the session boundary (playback terminals restamped on the session clock); the
machine stays byte-identical, so both rows here hold on the base and must keep holding. They pin the machine side of
the contract the restamp relies on:

- ``test_r18_fires_at_drain_time_plus_15s``: a ``PlaybackDrained`` with ``t_ms = T`` that completes a SPEAKING turn
  with a pending valid call (loop on) executes it and arms R18 at ``T + 15 000``; ``Tick(T + 14 999)`` emits no
  ``hud.error``, ``Tick(T + 15 000)`` emits ``hud.error "tool timeout"`` and closes the turn. ``T = 22 420`` is the
  W2-2 drain's *session* time (``logs/t0160-web-smoke-abcf17d/turns/W2-2``, seq 718).
- ``test_r18_deadline_follows_the_event_not_the_turn_start`` (adversarial): the generation started 20 s before the
  drain. A "fix" in the machine that armed R18 from the turn's ``llm_start_ms`` (or any clock of its own other than
  the event's ``t_ms``) would put the deadline 5 s in the past and fire on the next tick; the row requires the
  deadline the event names. The task file rejects fixing the machine (implementation notes); this row is what makes
  that rejection observable.

Expected values are computed by hand from R18 (``tool_result_timeout_ms`` = 15 000, docs/02 §4 T-0112) and the times
the rows choose; nothing is read back from the machine's own state to build them.

Regime: CPU, pure ``TurnMachine.step``, ``TurnPolicy(thinking=True, tool_loop=True)`` (``max_tool_steps`` 8, the
``--tools live`` default), default segmenter, n = 1, deterministic.
"""

from __future__ import annotations

from typing import Any

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import MachineState, ToolDraft, TurnState

#: R18 (T-0112): an executed call's ``tool.result`` deadline. Written out, not read from the policy object.
R18_MS = 15_000
#: The W2-2 drain's session time (seq 718, ``session_ms``): the time the restamped event would carry.
T_W2_2 = 22_420
#: The W2-2 call, verbatim from the session log (seq 612's ``llm.done``).
TYPE_AT_ARGS = {"press_enter": True, "text": "example.com", "x": 486, "y": 452}
PREFIX = "I will type example.com into the web address bar and press go."

LIVE_POLICY = TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8)


def _step(m: TurnMachine, s: MachineState, ev: E.Event) -> tuple[MachineState, list[Any]]:
    new, cmds = m.step(s, ev)
    return new, list(cmds)


def speaking_with_a_valid_call(
    m: TurnMachine, *, llm_start_ms: int, commit_ms: int
) -> tuple[MachineState, int]:
    """IDLE -> THINKING (``llm.start`` at ``llm_start_ms``) -> SPEAKING with ``llm.done`` carrying a valid call and
    ``tts.done``: one ``playback.drained`` away from R11. Returns the state and the last event time used."""
    eot_settle = m.policy.eot_settle_ms
    flushed_ms = llm_start_ms - eot_settle
    onset_ms = flushed_ms - 400
    s = m.initial()
    for ev in (
        E.UserSpeechOnset(t_ms=onset_ms, onset_ms=onset_ms),
        E.SttWord(t_ms=onset_ms + 300, text="go", start_ms=onset_ms, end_ms=onset_ms + 200),
        E.SttEndOfTurn(t_ms=flushed_ms - 10, t_eot_ms=onset_ms + 250, p_end=0.9),
        E.SttFlushed(t_ms=flushed_ms, t_eot_ms=onset_ms + 250),
        E.Tick(t_ms=llm_start_ms),
    ):
        s, _ = _step(m, s, ev)
    assert s.state is TurnState.THINKING and s.llm_start_ms == llm_start_ms, (
        f"premise: the generation starts at {llm_start_ms}; state {s.state.value}, llm_start_ms {s.llm_start_ms}"
    )
    rid = s.active_generate
    assert rid is not None
    s, _ = _step(
        m,
        s,
        E.LlmCommit(
            t_ms=commit_ms,
            request_id=rid,
            canvas_idx=0,
            segment="answer",
            text=PREFIX,
            thought_open=False,
        ),
    )
    draft = ToolDraft(call_id="t1-call", status="valid", name="type_at", arguments=TYPE_AT_ARGS)
    s, _ = _step(m, s, E.LlmDone(t_ms=commit_ms + 4, request_id=rid, tool=draft))
    utterance = s.utterances[-1] if s.utterances else "t1-u1"
    s, _ = _step(m, s, E.TtsDone(t_ms=commit_ms + 11, utterance_id=utterance))
    assert s.state is TurnState.SPEAKING and s.llm_done, (
        f"premise: SPEAKING with llm.done before the drain; got {s.state.value}, llm_done={s.llm_done}"
    )
    assert s.tool is not None and s.tool.status == "valid", (
        f"premise: a pending valid call; got {s.tool}"
    )
    return s, commit_ms + 11


def drain_executes(m: TurnMachine, s: MachineState, t_ms: int) -> tuple[MachineState, list[Any]]:
    s, cmds = _step(m, s, E.PlaybackDrained(t_ms=t_ms, played_ms=4760, played_text=PREFIX))
    executes = [c for c in cmds if isinstance(c, C.ToolExecute)]
    assert len(executes) == 1 and s.state is TurnState.THINKING, (
        f"premise (R11): the drain executes the call and the turn waits in THINKING; commands "
        f"{[c.type for c in cmds]}, state {s.state.value}"
    )
    return s, cmds


def hud_errors(cmds: list[Any]) -> list[str]:
    return [c.message for c in cmds if isinstance(c, C.HudError)]


def test_r18_fires_at_drain_time_plus_15s() -> None:
    """Criterion 3 (machine): armed at ``T + 15 000`` from the drain's ``t_ms``; silent at ``T + 14 999``; fires
    with ``hud.error "tool timeout"`` and closes the turn at ``T + 15 000``."""
    m = TurnMachine(LIVE_POLICY)
    s, _ = speaking_with_a_valid_call(m, llm_start_ms=2_816, commit_ms=17_579)
    s, cmds = drain_executes(m, s, T_W2_2)
    arms = [c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_result"]
    assert arms == [T_W2_2 + R18_MS], (
        f"R18: the drain at t_ms={T_W2_2} arms tool_result at {T_W2_2 + R18_MS}; got {arms}"
    )

    s, cmds = _step(m, s, E.Tick(t_ms=T_W2_2 + R18_MS - 1))
    assert hud_errors(cmds) == [] and s.state is TurnState.THINKING, (
        f"R18: one ms before the deadline nothing fires; hud.error {hud_errors(cmds)}, state {s.state.value}"
    )
    s, cmds = _step(m, s, E.Tick(t_ms=T_W2_2 + R18_MS))
    assert hud_errors(cmds) == ["tool timeout"], (
        f"R18: at the deadline the turn fails with hud.error 'tool timeout'; got {hud_errors(cmds)}"
    )
    assert s.state is TurnState.IDLE, f"R18 closes the turn; state {s.state.value}"


def test_r18_deadline_follows_the_event_not_the_turn_start() -> None:
    """Adversarial: ``llm_start_ms`` is 20 s before the drain and ticks ran right up to it. The deadline is still the
    drain's ``t_ms + 15 000``: a machine that armed R18 from the turn start (``llm_start_ms + 15 000``, 5 s before the
    drain) would fire on the next tick, which is exactly W2-2's symptom produced from the other side."""
    m = TurnMachine(LIVE_POLICY)
    t_drain = 40_000
    llm_start = t_drain - 20_000
    s, last = speaking_with_a_valid_call(m, llm_start_ms=llm_start, commit_ms=llm_start + 15_000)
    for t in range(last + 50, t_drain, 50):
        s, cmds = _step(m, s, E.Tick(t_ms=t))
        assert hud_errors(cmds) == [], (
            f"premise: nothing fails while speaking; t={t} {hud_errors(cmds)}"
        )
    assert s.state is TurnState.SPEAKING
    s, cmds = drain_executes(m, s, t_drain)
    arms = [c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_result"]
    assert arms == [t_drain + R18_MS], (
        f"R18 is armed from the executing event's t_ms ({t_drain}), not from llm_start_ms ({llm_start}): "
        f"want [{t_drain + R18_MS}], got {arms}"
    )
    for t in (t_drain + 50, t_drain + 5_000, t_drain + R18_MS - 1):
        s, cmds = _step(m, s, E.Tick(t_ms=t))
        assert hud_errors(cmds) == [] and s.state is TurnState.THINKING, (
            f"no R18 before t_drain + 15 000; at t={t}: hud.error {hud_errors(cmds)}, state {s.state.value}"
        )
    s, cmds = _step(m, s, E.Tick(t_ms=t_drain + R18_MS))
    assert hud_errors(cmds) == ["tool timeout"] and s.state is TurnState.IDLE, (
        f"R18 at t_drain + 15 000; got {hud_errors(cmds)}, state {s.state.value}"
    )
