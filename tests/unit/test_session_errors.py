"""Failures never kill the pump; degradation is the machine's decision (T-0007 criteria 13, 14).

Four clients can raise, and each one owes an event: ``llm.error`` (with the request id),
``stt.error``, ``tts.error`` and — for the playback path, which has no error event of its own —
``client.disconnected``, the §4 row that degrades to text-only and drops what is live. What must
never happen is an ``except Exception: pass`` that swallows the failure: the machine then keeps a
turn open on a service that is gone. The API is fixed in ``test_session_loop.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.speech import SttEndOfTurnEvent, SttEvent, SttWordEvent
from local_astra.contracts.turn import TurnState
from tests.fakes import FakeStt
from tests.unit.test_session_loop import answer, frame, running

STT_DEAD = "transcription socket closed"


class ExplodingStt(FakeStt):
    """Yields its script, then the socket dies — moshi-server dropping the connection."""

    async def events(self) -> AsyncIterator[SttEvent]:
        for event in self.script:
            yield event
        raise RuntimeError(STT_DEAD)


async def test_stt_events_reach_the_machine_with_their_audio_times(tmp_path):
    """Criterion 1: the STT *client* is one of the six fakes, and its events are mapped, not
    re-timed. Words lag their audio by ~0.5 s, so the audio timestamps on the wire are the only
    thing that can attribute a word to a turn (docs/02 §4)."""
    script = [*FakeStt.words("what is this", start_ms=1_000), SttEndOfTurnEvent(t_eot_ms=1_750)]
    async with running(tmp_path, script=[answer("Yes. ")], stt=FakeStt(script)) as rig:
        await rig.advance(200, step=50)
        words = rig.machine.events(E.SttWord)
        assert [(w.text, w.start_ms, w.end_ms) for w in words] == [
            ("what", 1_000, 1_250),
            ("is", 1_250, 1_500),
            ("this", 1_500, 1_750),
        ], f"the adapter must pass the audio times through; got {words}"
        eots = rig.machine.events(E.SttEndOfTurn)
        assert [e.t_eot_ms for e in eots] == [1_750], eots
        assert all(ev.t_ms >= 0 for ev in words + eots), "t_ms is the delivery time on the clock"


async def test_each_client_failure_becomes_its_event_and_the_pump_survives(tmp_path):
    """Criterion 13, four injections. Each one must produce its event *and* leave the pump alive —
    the test proves the latter by driving a whole turn afterwards on a restored client."""

    # 1. the LLM raises while generating: llm.error carrying the machine's request id
    async with running(tmp_path / "llm", script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        rig.llm.fail_generate = True
        await rig.ask("what is this")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "criterion 13: an LLM that raises ends the turn through the machine's row",
            limit_ms=5_000,
        )
        errors = rig.machine.events(E.LlmError)
        started = [c.request_id for c in rig.machine.commands(C.LlmStart)]
        assert errors, "the exception must become an llm.error event, not a swallowed traceback"
        assert errors[0].request_id == started[-1], (
            "criterion 13: the event carries the request id the machine knows "
            f"({errors[0].request_id!r} != {started[-1]!r})"
        )
        assert "400" in errors[0].message, (
            "the T-0020 seam: an HTTP 400/413 from the engine surfaces as an LlmError carrying "
            f"the status, not a flattened message; got {errors[0].message!r}"
        )
        assert rig.records()[-1].outcome == "error", rig.records()[-1].outcome
        rig.llm.fail_generate = False
        rig.llm.script = [answer("Recovered. ")]
        await rig.advance(500, step=250)
        await rig.ask("are you still there")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the pump did not survive")
        assert "Recovered." in rig.tts.texts()

    # 2. the STT socket dies: stt.error, and the turn it was listening to ends
    async with running(
        tmp_path / "stt",
        script=[answer("Yes. ")],
        frames=[frame(0, "a")],
        stt=ExplodingStt([SttWordEvent(text="hello", start_ms=0, end_ms=200)]),
    ) as rig:
        await rig.onset()
        await rig.advance(300, step=50)
        errors = rig.machine.events(E.SttError)
        assert errors and STT_DEAD in errors[0].message, (
            f"criterion 13: a dead transcriber becomes stt.error; got {errors}"
        )
        assert rig.state.state is TurnState.IDLE, (
            "the LISTENING/ENDPOINTING row of §4 ends the turn on stt.error"
        )

    # 3. the playback path raises: no error event of its own, so client.disconnected
    async with running(tmp_path / "audio", script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        rig.audio.fail_play = True
        await rig.ask("what is this")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE, "a dead playback device must end the turn"
        )
        assert rig.machine.events(E.ClientDisconnected), (
            "criterion 13: a client that raises becomes the corresponding event. The audio path "
            "has no error event in contracts/events.py, so a playback failure is "
            "client.disconnected — the §4 row that degrades to text-only and drops what is live. "
            "(Reported by the test-author as an ambiguity of criterion 13.)"
        )
        assert rig.machine.commands(C.Degrade), "and the machine's row degrades the session"


async def test_tts_down_degrades_and_the_next_turn_speaks(tmp_path):
    """Criterion 13 / docs/03 I7: with TTS failing the turn still yields a transcript, the session
    emits ``Degrade(text_only)``, and the next turn speaks once the service is back."""
    async with running(
        tmp_path, script=[answer("This is the answer. ")], frames=[frame(0, "a")], ms_per_word=100
    ) as rig:
        rig.tts.fail = True
        await rig.ask("what is on this screen")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE, "a TTS fault must not hang the turn"
        )
        assert rig.machine.events(E.TtsError), "the exception becomes tts.error"
        degrades = [c for c in rig.machine.commands(C.Degrade) if c.mode == "text_only"]
        assert degrades, "docs/02 §4: tts.error degrades to the text-only HUD"
        assert rig.state.history[-1].user_text == "what is on this screen", (
            "the turn still yields a transcript (docs/03 I7); history holds the user half"
        )
        assert rig.state.history[-1].assistant_spoken == "", "nothing was played, so nothing is"

        rig.tts.fail = False
        rig.llm.script = [answer("Now I can speak. ")]
        await rig.advance(500, step=250)
        await rig.ask("try again")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the next turn never ended")
        assert "Now I can speak." in rig.tts.texts(), (
            f"with TTS restored the next turn speaks; got {rig.tts.texts()}"
        )


async def test_a_slow_llm_client_does_not_delay_playback_pause(tmp_path):
    """Criterion 14, adversarial: ``pause()`` is the first thing a barge-in hop does.

    The TTS client is blocked (Pocket TTS on CPU under load is exactly this), so the executor has
    a synthesis in flight when the user cuts in. If that work sits in the pump's handler instead
    of a task, the ``user.speech_onset`` is not even read until the client answers, and the
    assistant keeps talking over the user for as long as the service is slow — the 150 ms
    ``pause_latency_budget_ms`` of docs/02 §4 blown by a service that is not on the audio path.
    """
    script = [answer("A long first sentence. "), answer("And a second one. ")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=200) as rig:
        rig.tts.hold()
        await rig.ask("tell me everything")
        await rig.until(
            lambda: rig.tts.requests != [],
            "the answer never reached the (blocked) TTS client",
            limit_ms=5_000,
        )
        assert rig.state.state is TurnState.SPEAKING
        llm_calls, tts_calls = len(rig.llm.calls), len(rig.tts.requests)
        audio_calls = len(rig.audio.calls)

        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()

        assert rig.audio.pauses, (
            "criterion 14: with a blocked TTS client the executor still pauses playback on the "
            "barge-in hop — LLM and TTS work belongs in tasks, not in the handler "
            "(.claude/rules/conductor.md: nothing in the pump blocks for more than 20 ms)"
        )
        hop = rig.audio.calls[audio_calls:]
        assert hop and hop[0] == "pause", (
            f"pause() is the executor's first action on a barge-in hop; it did {hop}"
        )
        assert len(rig.llm.calls) == llm_calls and len(rig.tts.requests) == tts_calls, (
            "no LLM/TTS work ran ahead of pause() on the barge-in hop"
        )
        assert rig.state.state is TurnState.PAUSED

        rig.tts.release()
        await rig.pump()
