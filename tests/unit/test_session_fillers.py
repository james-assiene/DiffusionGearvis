"""Filler bookkeeping: one ack, at most one interim update (T-0007 criterion 11).

The turn machine already emits ``PlayClip(kind="ack")`` at ``llm.start`` and counts it in
``fillers_this_turn`` (T-0003 criterion 7 / A7). The *session* owns only the interim spoken update
(docs/11 §1.1 tier B) and must seed ``filler_policy.decide`` from the machine's own ack, so the
policy cannot hand back a second acknowledgement. The T-0013 merged note pins the unknown-EMA
mapping tested below. The API is fixed in ``test_session_loop.py``; clips are played through
``TtsBackend.speak`` + ``audio.play(..., source="")``.
"""

from __future__ import annotations

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts.turn import TurnState
from local_astra.reflex import filler_policy
from local_astra.reflex.filler_policy import FillerInputs, FillerPolicyConfig
from tests.unit.test_session_loop import answer, frame, running

#: T-0013 merged note, pinned by criterion 11: an unknown EMA counts as *slow*.
UNKNOWN_EMA_MS = FillerPolicyConfig().ack_threshold_ms + 1


class DecideSpy:
    """Records every ``filler_policy.decide`` call and its answer, then defers to the real one."""

    def __init__(self) -> None:
        self.calls: list[tuple[FillerInputs, str]] = []

    def __call__(self, inputs, cfg=None):
        decision = filler_policy._real_decide(inputs, cfg)  # type: ignore[attr-defined]
        self.calls.append((inputs, decision))
        return decision

    def inputs(self) -> list[FillerInputs]:
        return [call for call, _ in self.calls]

    def decisions(self) -> list[str]:
        return [decision for _, decision in self.calls]


@pytest.fixture
def spy(monkeypatch):
    """Patch the module attribute: the session must call ``filler_policy.decide`` through it."""
    monkeypatch.setattr(filler_policy, "_real_decide", filler_policy.decide, raising=False)
    probe = DecideSpy()
    monkeypatch.setattr(filler_policy, "decide", probe)
    return probe


async def test_exactly_one_ack_per_turn_and_one_interim_update(tmp_path, spy):
    """Criterion 11: one ack (the machine's), at most one interim update, and no double-play.

    The generate is held for eight seconds of virtual time, which is two full
    ``interim_update_ms`` windows: the turn gets the machine's acknowledgement clip and exactly
    one "still working" update, never a second ack and never a third clip.
    """
    async with running(
        tmp_path, script=[answer("Here it is. ")], frames=[frame(0, "a")], ms_per_word=100
    ) as rig:
        rig.llm.hold()
        await rig.ask("what is on this screen")
        assert rig.state.state is TurnState.THINKING
        acks = [c for c in rig.machine.commands(C.PlayClip) if c.kind == "ack"]
        assert len(acks) == 1, f"the machine emits exactly one ack per turn; got {acks}"
        assert rig.state.fillers_this_turn == 1

        await rig.advance(8_000, step=250)
        assert rig.state.state is TurnState.THINKING, "the generate is still held"

        assert spy.calls, (
            "criterion 11: the session asks filler_policy.decide on ticks while THINKING — and it "
            "must call it through the module (filler_policy.decide), so the decision is visible"
        )
        assert all(call.state == "THINKING" for call in spy.inputs()), (
            f"decide is only called while THINKING; got {[c.state for c in spy.inputs()]}"
        )
        assert all(call.thinking_enabled for call in spy.inputs())
        assert all(
            call.fillers_this_turn >= 1 and call.last_filler_age_ms is not None
            for call in spy.inputs()
        ), (
            "criterion 11: fillers_this_turn and last_filler_age_ms are seeded from the machine's "
            f"own ack, so decide cannot return a second clip:ack; got "
            f"{[(c.fillers_this_turn, c.last_filler_age_ms) for c in spy.inputs()]}"
        )
        assert "clip:ack" not in spy.decisions(), (
            f"the session played a second acknowledgement: {spy.decisions()}"
        )
        assert spy.decisions().count("clip:still_working") <= 1, (
            f"the interim update fires once (docs/11 §1.1 tier B); got {spy.decisions()}"
        )
        assert "generate_short" not in spy.decisions(), (
            "criterion 11, negative: generate_short is never executed in v1 (T-0013 criterion 5)"
        )

        clips_while_thinking = len(rig.audio.clip_plays())
        assert clips_while_thinking == 2, (
            "one ack plus one interim update, played once each (criterion 11: two ack clips in "
            f"one turn fails); got {clips_while_thinking} clip plays"
        )

        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        assert len(rig.audio.clip_plays()) == clips_while_thinking, (
            "criterion 11, negative: no filler is played after the first canvas commits"
        )
        assert len(rig.llm.gen_prompts) == 1, (
            "criterion 11, negative: no second generate — generate_short is not in v1"
        )


async def test_unknown_first_canvas_ema_counts_as_slow(tmp_path, spy):
    """Criterion 11, the T-0013 mapping row: ``first_canvas_ms is None`` → ``ack_threshold_ms + 1``.

    The EMA is unknown on the first turn of a session. Mapping "unknown" to 0 (the tempting
    default) says "the canvas will be here in no time" and silences the acknowledgement on exactly
    the turn that is most likely to be slow — the cold one.
    """
    async with running(
        tmp_path, script=[answer("Here it is. ")], frames=[frame(0, "a")], ms_per_word=100
    ) as rig:
        rig.llm.hold()
        await rig.ask("first question")
        assert rig.state.first_canvas_ms is None, "no EMA before the first canvas of the session"
        await rig.advance(6_000, step=250)

        assert spy.inputs(), "decide must be called while THINKING"
        unknown = {call.predicted_first_canvas_ms for call in spy.inputs()}
        assert unknown == {UNKNOWN_EMA_MS}, (
            f"criterion 11: with MachineState.first_canvas_ms None the session passes "
            f"predicted_first_canvas_ms = ack_threshold_ms + 1 = {UNKNOWN_EMA_MS} (unknown counts "
            f"as slow, T-0013 merged note); got {sorted(unknown)}"
        )

        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn one never ended")
        ema = rig.state.first_canvas_ms
        assert ema is not None, "the machine measured the first canvas of turn one"

        first_turn_calls = len(spy.calls)
        rig.llm.hold()
        rig.llm.script = [answer("And again. ")]
        await rig.advance(500, step=250)
        await rig.ask("second question")
        await rig.advance(6_000, step=250)
        second = {call.predicted_first_canvas_ms for call in spy.inputs()[first_turn_calls:]}
        assert second == {ema}, (
            "once the EMA is known the session passes it unchanged (MachineState.first_canvas_ms "
            f"= {ema}); got {sorted(second)}"
        )
        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn two never ended")
