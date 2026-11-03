"""T-0181 group A0-S: the session half of "muting mid-question drops the question" (docs/tasks/T-0181.md, "## Group A0
replan (2026-10-28)", A0-S1..A0-S5; H-T181-2, James 2026-10-28: "just drop it"; review ruling A0-R1).

The rig is ``tests/unit/_t0181_rig.mic_rig``: the session over the fakes, tools off, a socketless ``HudServer`` with one
greeted page, ``FakeAudioIO`` capture (80 ms STT frames on an audio clock that starts with the session clock at 0), and
a *reactive*, content-blind ``FakeStt`` (a scripted word is released once fed audio passes ``end_ms + 500``, whatever
the fed samples were): so a word the session must drop is still delivered by the fake, and only the session can drop
it. Mutes go over the wire as raw JSON (``rig.engage()`` / ``rig.release()``); before group A lands the codec answers
``Unknown``, no frame is ever zeroed, and every row fails on ``engage``'s named A5 assertion or on the first
``MicMuted`` count, never at collection.

**The scenario most rows share** (hand-timed, confirmed on BASE 74ae8b25 by running it without the mute): the question
``what quokka narwhal zebra`` is spoken at audio 1000 ms, 200 ms a word, so its words are released at clock 1700, 1900,
2100 and 2300 (each ``end + 500``; the 80 ms framing delays each by up to one frame). The mute comes once the machine
holds ``what quokka`` (LISTENING), so ``narwhal`` (start 1400) and ``zebra`` (start 1600) are *late words*: their audio
starts before the first zeroed frame (~2050), so A6's span cannot drop them, and only A0-S2's fence
``[onset_ms, first unmuted frame)`` does. ``narwhal`` and ``zebra`` are the texts that must never be logged or published.

**Readings** (listed in the report):

- the turn line's ``ended_by``/``mic_muted`` are ``extras`` keys (where T-0162 writes ``ended_by: "reset"`` and T-0177
  A10 writes ``vision_source``), read as raw JSON;
- A0-R1's "the IDLE stale commit is not attached to the abandoned turn's line or memory": after the abandon, a stale
  ``playback.drained`` carrying an older reply and a ``client.disconnected`` make the machine emit ``commit_history``
  from IDLE (BASE behaviour). The row requires that the abandoned turn's line is unchanged and alone, that no history
  entry carries its words, and that **no episode is ingested** (the only frame the session holds then is the abandoned
  turn's own snapshot; at BASE ``commit_history`` falls back to it, which would put the dropped turn's screen in memory);
- "quick unmute" (A-R4) uses a fake **without** steps: with steps the endpointer's backstop closes the old run within
  ``EOT_SILENCE_MS`` anyway, so a session missing A0-S3 would pass;
- the A0-S5 liveness rows (each A7 state, 15 s) live in ``test_t0181_mic_states.py``.

Regime: CPU over fakes, ``ScriptedClock``, ``TurnPolicy(thinking=True)`` (the ack clip on, as ``TurnPolicy()``), tools
off, ``RecordingTts`` 200 ms a word, n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests.fakes.stt import FakeStt
from tests.unit._t0181_rig import GATE_NOTE, MicRig, RecordingStt, mic_rig, reactive_stt
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_loop import RecordingAudio

QUESTION = "what quokka narwhal zebra"
Q_START = 1000
#: the words the machine holds when the mute comes (released at ~1700 and ~1900)
HEARD = ("what", "quokka")
#: the late words: audio before the first zeroed frame, released after the abandon (A0-S2's fence drops them)
LATE = ("narwhal", "zebra")
ANSWER = "A shop. "


def voice_stt(*, steps: bool = True) -> Any:
    stt = reactive_stt(steps=steps)
    return stt


async def open_question(rig: MicRig, text: str = QUESTION, heard: int = len(HEARD)) -> None:
    """Speak ``text`` at ``Q_START`` and wait until the machine holds its first ``heard`` words in LISTENING."""
    rig.say(text, Q_START)
    await rig.until(
        lambda: len(rig.state.words) >= heard,
        f"premise: the machine never held {heard} words of {text!r}",
        limit_ms=4_000,
        step=40,
    )
    assert rig.state.state is TurnState.LISTENING, (
        f"premise: LISTENING at the mute; {rig.state.state}"
    )
    assert [w.text for w in rig.state.words] == text.split()[:heard], rig.state.words


def assert_dropped(rig: MicRig, turn_id: str, words: tuple[str, ...]) -> None:
    """A0-2 / A0-S4 seen from the session: nothing started, stored, spoken or ingested for ``turn_id``."""
    assert rig.commands_of("llm.start") == [], (
        f"A0-S4: no llm.start; {rig.commands_of('llm.start')}"
    )
    assert rig.commands_of("commit_history") == [], "A0-S4: no commit_history"
    assert rig.commands_of("schedule_summary") == [], "A0-S4: no schedule_summary"
    assert rig.memory.episodes == {}, f"A0-S4: no memory ingest; {rig.memory.episodes}"
    spoken = [m for m in rig.published.messages if getattr(m, "type", None) == "astra.spoken"]
    assert spoken == [], f"A0-S4: no astra.spoken; {spoken}"
    assert rig.state.history == (), (
        f"A0-S4: the dropped words never reach history; {rig.state.history}"
    )
    assert rig.state.state is TurnState.IDLE, rig.state.state
    finals = [m for m in rig.transcripts(turn_id) if m.final]
    assert [m.text for m in finals] == [""], (
        f"A0-S4: exactly one astra.transcript({turn_id}, text='', final=true) and never a final carrying "
        f"its words; got {[(m.text, m.final) for m in rig.transcripts(turn_id)]}"
    )


# =====================================================================================================
# A0-S1: one MicMuted per false -> true
# =====================================================================================================


async def test_one_mic_muted_per_false_to_true_transition(tmp_path: Path) -> None:
    """A0-S1: over true, true (redundant), false, false (redundant), true, false, each 200 ms apart, the machine is
    stepped with exactly two ``MicMuted``, stamped with the session clock at the two false->true mutes; never on a
    redundant toggle, never on an unmute."""
    async with mic_rig(tmp_path) as rig:
        await rig.advance(400)
        stamps: list[int] = []
        for muted in (True, True, False, False, True, False):
            at = rig.clock.now_ms
            await rig.mute(muted)
            stamps.append(at)
            await rig.advance(200)
        got = [ev.t_ms for ev in rig.mic_muted_events()]
    assert got == [stamps[0], stamps[4]], (
        f"A0-S1: one MicMuted(t_ms=clock.now_ms) per false->true transition (at {stamps[0]} and {stamps[4]}); "
        f"the machine was stepped with MicMuted at {got}"
    )


# =====================================================================================================
# A0-S4: a mid-question mute drops the question
# =====================================================================================================


async def test_a_mid_question_mute_drops_the_question(tmp_path: Path) -> None:
    """A0-2/A0-S4: the mute comes while the machine holds ``what quokka`` in LISTENING. The machine is stepped with one
    ``MicMuted`` from LISTENING to IDLE; afterwards (3 s muted, then an unmute and 3 s more) there is no ``llm.start``,
    ``commit_history``, ``schedule_summary``, memory ingest or ``astra.spoken``; the history is empty; the page gets
    exactly one empty final for ``t1``; and ``t1``'s turn line carries ``ended_by: "mic_muted"`` and
    ``mic_muted: true``. Without the feature the same session answers the question (BASE: one ``llm.start``)."""
    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        await open_question(rig)
        await rig.engage()
        await rig.advance(3_000)
        await rig.release()
        await rig.advance(3_000)
        moved = [
            (before, after)
            for before, ev, after in rig.machine.transitions()
            if ev.type == "mic.muted"
        ]
        assert moved == [(TurnState.LISTENING, TurnState.IDLE)], (
            f"A0-2: one MicMuted, LISTENING -> IDLE; got {moved}"
        )
        assert_dropped(rig, "t1", (*HEARD, *LATE))
        lines = rig.raw_lines()
    assert [line["turn_id"] for line in lines] == ["t1"], [line["turn_id"] for line in lines]
    extras = lines[0].get("extras", {})
    assert extras.get("ended_by") == "mic_muted", (
        f"A0-S4: ended_by 'mic_muted'; got {extras.get('ended_by')!r}"
    )
    assert extras.get("mic_muted") is True, (
        f"A0-S4/A10: mic_muted: true; got {extras.get('mic_muted')!r}"
    )


async def test_adversarial_a_late_word_of_the_dropped_question_opens_nothing(
    tmp_path: Path,
) -> None:
    """A0-S2 (adversarial, the late word): ``hello quokka`` is muted once ``hello`` is held. ``quokka``'s audio
    (1200-1400) starts before the first zeroed frame, so A6's span cannot drop it; it is released ~200 ms after the
    abandon. It must open nothing: no ``UserSpeechOnset`` and no ``SttWord`` after the ``MicMuted``, no second turn, no
    ``llm.start``. A fresh question after the unmute is then answered with exactly its own words (A0-R2 at the session
    level: no stale word rides into the next transcript)."""
    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        await open_question(rig, "hello quokka", heard=1)
        since = len(rig.machine.log)
        first_zero = await rig.engage()
        assert first_zero > Q_START + 200, (
            f"premise: quokka (start 1200) precedes the span ({first_zero})"
        )
        await rig.advance(2_000)
        after = (rig.stepped("user.speech_onset", since), rig.stepped("stt.word", since))
        assert after == ([], []), (
            f"A0-S2: the late word reached the machine after the abandon: onsets {after[0]}, words {after[1]}"
        )
        assert rig.commands_of("llm.start") == [], "A0-S2: the late word started no turn"
        first_live = await rig.release()
        rig.say("tell me more", first_live + 400)
        await rig.until(
            lambda: bool(rig.commands_of("llm.start")),
            "the fresh question after the unmute was never asked",
            limit_ms=6_000,
        )
        transcripts = [c.transcript for c in rig.commands_of("llm.start")]
    assert transcripts == ["tell me more"], (
        f"A0-S2/A0-R2: the next turn's transcript is exactly its own words; got {transcripts}"
    )


# =====================================================================================================
# A0-S3: the quick unmute (A-R4: a fake without endpointer steps)
# =====================================================================================================


async def quick_unmute(rig: MicRig) -> int:
    """``hello quokka narwhal`` at 1000; mute once ``hello`` is held; unmute 300 ms (clock) later. Returns the
    ``t_ms`` of the first unmuted frame."""
    await open_question(rig, "hello quokka narwhal", heard=1)
    muted_at = rig.clock.now_ms
    await rig.engage()
    left = 300 - (rig.clock.now_ms - muted_at)
    if left > 0:
        await rig.advance(left, step=20)
    return await rig.release()


async def test_adversarial_quick_unmute_the_next_word_is_an_onset(tmp_path: Path) -> None:
    """A0-S3 (adversarial, A-R4): the recogniser sends no steps, so nothing but A0-S3 closes the abandoned run. After a
    300 ms mute, a new word ``zebra`` starting 160 ms after the first unmuted frame (well within ``EOT_SILENCE_MS`` of
    the mute) must be an onset at its own ``start_ms`` and open a LISTENING turn holding exactly ``zebra``. A session
    that kept the endpointer's open run hears it as the continuation of the dropped question: no onset, no turn."""
    async with mic_rig(tmp_path, stt=voice_stt(steps=False), script=[answer(ANSWER)]) as rig:
        first_live = await quick_unmute(rig)
        since = len(rig.machine.log)
        (zebra,) = rig.say("zebra", first_live + 160)
        await rig.until(
            lambda: bool(rig.stepped("stt.word", since)),
            "premise: zebra was never delivered",
            limit_ms=2_000,
        )
        onsets = [ev.onset_ms for ev in rig.stepped("user.speech_onset", since)]
        assert onsets == [zebra.start_ms], (
            f"A0-S3: the first admitted word after the unmute is an onset at {zebra.start_ms}; onsets {onsets}"
        )
        assert rig.state.state is TurnState.LISTENING, rig.state.state
        assert [w.text for w in rig.state.words] == ["zebra"], rig.state.words


async def test_too_wide_a_word_at_the_first_unmuted_frame_is_kept(tmp_path: Path) -> None:
    """A0-S2 (too wide): the fence is ``[onset_ms, first unmuted frame)``, so a word starting **exactly** at the first
    unmuted frame's ``t_ms`` is kept: it reaches the machine as an ``SttWord`` with its own times and opens a turn. The
    late words ``quokka`` and ``narwhal`` (inside the fence) still do not reach the machine."""
    async with mic_rig(tmp_path, stt=voice_stt(steps=False), script=[answer(ANSWER)]) as rig:
        first_live = await quick_unmute(rig)
        since = len(rig.machine.log)
        (zebra,) = rig.say("zebra", first_live)
        await rig.advance(1_500)
        words = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
        all_words = [ev.text for ev in rig.stepped("stt.word")]
    assert words == [("zebra", first_live)], (
        f"A0-S2: a word starting at the first unmuted frame ({first_live}) is kept; words stepped since {words}"
    )
    assert "quokka" not in all_words and "narwhal" not in all_words, all_words
    assert zebra.start_ms == first_live


# =====================================================================================================
# A0-5 / A0-3 at the session: typed turns and THINKING are answered
# =====================================================================================================


async def test_a_typed_question_just_before_the_mute_is_answered(tmp_path: Path) -> None:
    """A0-5 (H-T181-8): a typed question opens ENDPOINTING with its flush already set; a mute right after it (same
    virtual instant) steps ``MicMuted`` with no row, and the question is answered: ``llm.start`` carries the typed text,
    the answer is spoken, the history holds it, and the turn line has no ``ended_by: "mic_muted"`` but does carry
    ``mic_muted: true`` (muted during the turn, A10)."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400)
        await rig.type_question("what is quokka")
        assert rig.state.state is TurnState.ENDPOINTING and rig.state.flush_t_ms is not None, (
            f"premise: a typed turn sits in ENDPOINTING with its flush set; {rig.state.state}"
        )
        await rig.mute(True)
        await rig.turn_done()
        moved = [
            (before, after)
            for before, ev, after in rig.machine.transitions()
            if ev.type == "mic.muted"
        ]
        assert moved == [(TurnState.ENDPOINTING, TurnState.ENDPOINTING)], (
            f"A0-3/A0-5: one MicMuted, no row; got {moved}"
        )
        assert [c.transcript for c in rig.commands_of("llm.start")] == ["what is quokka"]
        assert [(h.user_text, h.assistant_spoken) for h in rig.state.history] == [
            ("what is quokka", ANSWER)
        ], rig.state.history
        (line,) = rig.raw_lines()
    assert line["extras"].get("ended_by") != "mic_muted", line["extras"].get("ended_by")
    assert line["extras"].get("mic_muted") is True, (
        f"A10: a turn muted at any moment carries mic_muted: true; got {line['extras'].get('mic_muted')!r}"
    )


async def spoken_turn(tmp_path: Path, *, mute_in_thinking: bool) -> dict[str, Any]:
    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        rig.llm.hold()
        rig.say(QUESTION, Q_START)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING,
            "premise: THINKING",
            limit_ms=6_000,
            step=40,
        )
        if mute_in_thinking:
            await rig.mute(True)
        await rig.advance(200)
        rig.llm.release()
        await rig.turn_done()
        return {
            "muted": [
                (before, after)
                for before, ev, after in rig.machine.transitions()
                if ev.type == "mic.muted"
            ],
            "history": [(h.user_text, h.assistant_spoken) for h in rig.state.history],
            "starts": [c.transcript for c in rig.commands_of("llm.start")],
            "speaks": [c.source for c in rig.speaks()],
            "takeovers": rig.stepped("user.takeover"),
            "line": rig.raw_lines(),
        }


async def test_a_mute_in_thinking_is_answered_as_at_base(tmp_path: Path) -> None:
    """A0-3 (THINKING: the question was complete): the same spoken question, once with no mute and once muted while
    THINKING (the LLM held). The muted run steps one ``MicMuted`` THINKING -> THINKING, starts the same ``llm.start``,
    speaks the same sentences, commits the same history, sees no takeover, and its line carries ``mic_muted: true``
    without ``ended_by: "mic_muted"``."""
    (tmp_path / "base").mkdir()
    (tmp_path / "muted").mkdir()
    base = await spoken_turn(tmp_path / "base", mute_in_thinking=False)
    muted = await spoken_turn(tmp_path / "muted", mute_in_thinking=True)
    assert base["history"] == [(QUESTION, ANSWER)], f"premise: BASE answers; {base['history']}"
    assert muted["muted"] == [(TurnState.THINKING, TurnState.THINKING)], (
        f"A0-3: one MicMuted in THINKING, no row; got {muted['muted']}"
    )
    for key in ("history", "starts", "speaks"):
        assert muted[key] == base[key], f"A0-3: {key} as at BASE; {muted[key]} != {base[key]}"
    assert muted["takeovers"] == [], muted["takeovers"]
    (line,) = muted["line"]
    assert (
        line["extras"].get("mic_muted") is True and line["extras"].get("ended_by") != "mic_muted"
    ), line["extras"]


# =====================================================================================================
# A0-S2 / A10: words_dropped counts the fence; no dropped text is logged
# =====================================================================================================


async def test_words_dropped_counts_the_fence_and_no_dropped_text_is_logged(tmp_path: Path) -> None:
    """A0-S2 / A10: the shared scenario's two late words (``narwhal``, ``zebra``) are counted in the unmute's
    ``mic.gate`` record (``words_dropped == 2``; the mute's record has no ``words_dropped``), and neither text appears
    anywhere in the session log or in any published message."""
    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        await open_question(rig)
        await rig.engage()
        await rig.advance(2_000)
        await rig.release()
        await rig.advance(400)
        published = rig.published_text()
    gates = rig.notes(GATE_NOTE)
    assert [g.get("muted") for g in gates] == [True, False], gates
    assert "words_dropped" not in gates[0], f"A10: words_dropped on the unmute only; {gates[0]}"
    assert gates[1].get("words_dropped") == len(LATE), (
        f"A0-S2: the fence's drops are counted ({len(LATE)}); got {gates[1].get('words_dropped')!r}"
    )
    text = rig.log_text()
    for word in LATE:
        assert word not in text, f"A6/A10: the dropped word {word!r} is in the session log"
        assert word not in published, f"A6/A10: the dropped word {word!r} was published"


# =====================================================================================================
# A0-R1: an IDLE stale commit after the abandon
# =====================================================================================================


async def test_a_stale_idle_commit_is_not_attached_to_the_abandoned_turn(tmp_path: Path) -> None:
    """A0-R1: after the abandon, a stale ``playback.drained`` carrying an older reply and a ``client.disconnected``
    make the machine emit ``commit_history`` from IDLE (BASE behaviour, ``_commit_played``). That commit carries none
    of the abandoned words, and the session does not attach it to the abandoned turn: ``t1``'s line is unchanged and
    alone, and no episode is ingested (the abandoned turn's snapshot is the only frame the session holds)."""
    from local_astra.contracts import events as E

    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        await open_question(rig)
        await rig.engage()
        await rig.advance(1_500)
        assert rig.state.state is TurnState.IDLE and len(rig.lines()) == 1, (
            f"A0-2: the mute abandoned t1 (IDLE, one line); {rig.state.state}, {len(rig.lines())} lines"
        )
        line_before = rig.lines()
        since = len(rig.machine.log)
        now = rig.clock.now_ms
        rig.submit(E.PlaybackDrained(t_ms=now, played_ms=900, played_text="an older reply"))
        rig.submit(E.ClientDisconnected(t_ms=now))
        await rig.advance(400)
        assert rig.commands_of("commit_history", since), (
            "premise: the machine committed from IDLE (BASE)"
        )
        history_text = " ".join(f"{h.user_text} {h.assistant_spoken}" for h in rig.state.history)
        for word in (*HEARD, *LATE):
            assert word not in history_text.split(), (
                f"A0-R1: {word!r} reached history: {history_text!r}"
            )
        assert rig.lines() == line_before, "A0-R1: the abandoned turn's line is unchanged and alone"
        assert rig.memory.episodes == {}, (
            f"A0-R1 (reading): the IDLE commit ingested the abandoned turn's frame; {rig.memory.episodes}"
        )


# =====================================================================================================
# A-RV3 / A-RV2: the A0-S2 hold, both directions (words delivered between apply_mic_mute and the step)
# =====================================================================================================


class WindowStt(RecordingStt):
    """``RecordingStt`` that also notes, for every event the recogniser yields to ``_pump_stt``, how many
    ``MicMuted`` the machine had been stepped with at that moment (``probe``, set by the row once the rig exists).
    A word scripted right before ``send_mute`` (no await between) and yielded while that count is still the old one
    was delivered inside A0-S2's window: after ``apply_mic_mute`` returned and before its ``MicMuted`` was stepped."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.probe: Callable[[], int] | None = None
        self.seen: list[tuple[Any, int | None]] = []

    async def events(self) -> AsyncIterator[Any]:  # type: ignore[override]
        async for event in self.inner.events():
            self.seen.append((event, self.probe() if self.probe is not None else None))
            yield event


def window_stt(*, steps: bool) -> WindowStt:
    return WindowStt(FakeStt(reactive=True, steps=steps))


def in_window(stt: WindowStt, text: str, stepped_before: int) -> bool:
    """``text`` was yielded while the machine still had ``stepped_before`` ``MicMuted`` steps (the mute queued)."""
    return any(getattr(ev, "text", None) == text and n == stepped_before for ev, n in stt.seen)


async def let_the_recogniser_run(rig: MicRig) -> None:
    """Yield to the loop without advancing the clock, so ``_pump_stt`` takes what is already due, then settle."""
    for _ in range(20):
        await asyncio.sleep(0)
    await rig.pump()


async def test_adversarial_a_late_word_held_between_the_mute_and_its_step_never_reaches_the_next_turn(
    tmp_path: Path,
) -> None:
    """A0-S2 hold (A-RV3, dropped direction): ``what quokka`` is held in LISTENING (no steps, so nothing ends the
    run). Once fed audio is past 2100 ms, ``narwhal`` (audio 1400-1600, inside the question's fence) is scripted --
    already due -- and the mute is sent in the same instant, so the recogniser hands ``narwhal`` to ``_pump_stt``
    after ``apply_mic_mute`` returned and before the ``MicMuted`` step (premise, read off the recogniser's side).
    It must never reach the machine (no ``stt.word`` after the mute) and a fresh question after the unmute is asked
    with exactly its own words. A session that hands such a word straight to the endpointer (no hold) feeds it to the
    machine behind the abandon and it rides into the next turn."""
    stt = window_stt(steps=False)
    async with mic_rig(tmp_path, stt=stt, script=[answer(ANSWER)]) as rig:
        stt.probe = lambda: len(rig.mic_muted_events())
        await open_question(rig, "what quokka", heard=2)
        await rig.until(
            lambda: stt.inner.fed_ms >= 2_200, "premise: fed audio past 2200", limit_ms=2_000
        )
        assert rig.state.state is TurnState.LISTENING, f"premise: LISTENING; {rig.state.state}"
        since = len(rig.machine.log)
        (late,) = rig.say("narwhal", 1_400)
        rig.send_mute(True)
        await let_the_recogniser_run(rig)
        assert in_window(stt, "narwhal", 0), (
            f"premise: narwhal was delivered between apply_mic_mute and the MicMuted step; {stt.seen[-3:]}"
        )
        assert [ev.text for ev in rig.stepped("stt.word", since)] == [], (
            f"A0-S2: the held late word reached the machine; {rig.stepped('stt.word', since)}"
        )
        await rig.advance(1_500)
        first_live = await rig.release()
        rig.say("tell me more", first_live + 400)
        await rig.until(
            lambda: len(rig.state.words) >= 3,
            "the fresh question after the unmute was never heard",
            limit_ms=4_000,
        )
        await rig.advance(400)
        held = [w.text for w in rig.state.words]
        words = [ev.text for ev in rig.stepped("stt.word", since)]
    assert late.text not in words, (
        f"A0-S2: {late.text!r} reached the machine after the mute; {words}"
    )
    assert held == ["tell", "me", "more"], (
        f"A0-S2/A-RV3: the next turn holds exactly its own words (the recogniser sends no steps, so it stays "
        f"open in LISTENING); got {held}"
    )


async def test_a_pre_span_word_held_between_the_mute_and_its_step_is_kept(tmp_path: Path) -> None:
    """A0-S2 hold (A-RV3/A-RV2, kept direction): in IDLE, ``hello`` (audio 1500-1700, before any zeroed frame) is
    scripted once fed audio is past 2200 -- already due -- and the mute is sent in the same instant, so it is
    delivered between ``apply_mic_mute`` and the step (premise). The step abandons nothing, so the held word is
    released in order and reaches the machine once, with its own times (A6 keeps a word that starts before the span).
    A session that holds and never releases loses it."""
    stt = window_stt(steps=False)
    async with mic_rig(tmp_path, stt=stt) as rig:
        stt.probe = lambda: len(rig.mic_muted_events())
        await rig.until(
            lambda: stt.inner.fed_ms >= 2_200, "premise: fed audio past 2200", limit_ms=4_000
        )
        assert rig.state.state is TurnState.IDLE, rig.state.state
        since = len(rig.machine.log)
        (word,) = rig.say("hello", 1_500)
        rig.send_mute(True)
        await let_the_recogniser_run(rig)
        assert in_window(stt, "hello", 0), (
            f"premise: hello was delivered between apply_mic_mute and the MicMuted step; {stt.seen[-3:]}"
        )
        assert len(rig.stepped("mic.muted", since)) == 1, "premise: the MicMuted was stepped"
        await rig.advance(200)
        got = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
    assert got == [("hello", word.start_ms)], (
        f"A0-S2/A-RV2: a pre-span word held while the MicMuted was queued is released through the fence and kept; "
        f"words stepped since the mute {got}"
    )


# =====================================================================================================
# A-RV3 / A10: words_dropped is per unmute
# =====================================================================================================


async def test_words_dropped_is_counted_per_unmute_over_two_cycles(tmp_path: Path) -> None:
    """A10/A0-S2 (A-RV3): two mute/unmute cycles. Cycle 1 is the shared scenario (the fence drops ``narwhal`` and
    ``zebra``: 2). Cycle 2 mutes in IDLE and the content-blind fake delivers ``pelican`` from inside the span while
    muted (1). The four ``mic.gate`` records are mute, unmute(2), mute, unmute(1): each unmute counts the words
    dropped since the previous unmute, never a running total (3)."""
    async with mic_rig(tmp_path, stt=voice_stt(), script=[answer(ANSWER)]) as rig:
        await open_question(rig)
        await rig.engage()
        await rig.advance(2_000)
        await rig.release()
        await rig.advance(600)
        first_zero = await rig.engage()
        (pelican,) = rig.say("pelican", first_zero + 160)
        await rig.until(
            lambda: stt_fed_past(rig, pelican.end_ms + 500 + 160),
            "premise: pelican was due while muted",
            limit_ms=2_000,
        )
        await rig.release()
        await rig.advance(400)
        assert rig.commands_of("llm.start") == [], rig.commands_of("llm.start")
    gates = rig.notes(GATE_NOTE)
    assert [g.get("muted") for g in gates] == [True, False, True, False], gates
    assert [gates[1].get("words_dropped"), gates[3].get("words_dropped")] == [len(LATE), 1], (
        f"A10: words_dropped per unmute ({len(LATE)}, then 1); got "
        f"{[g.get('words_dropped') for g in gates if g.get('muted') is False]}"
    )


def stt_fed_past(rig: MicRig, at_ms: int) -> bool:
    return bool(rig.rec_stt.inner.fed_ms >= at_ms)


# =====================================================================================================
# A-RV1 / A0-R3: a mixed turn (typed, then reopened by speech) is dropped with none of its spoken words
# =====================================================================================================


def lagging_audio(offset_ms: int) -> Callable[[Any], RecordingAudio]:
    """``RecordingAudio`` whose STT frames carry an audio clock ``offset_ms`` behind the session clock (production:
    ``SessionClock`` starts at cli start, the ADC position at stream open)."""

    class LateAdc(RecordingAudio):
        async def stt_blocks(self) -> Any:
            async for frame in super().stt_blocks():
                yield dataclasses.replace(
                    frame, t_ms=frame.t_ms - offset_ms, end_ms=frame.end_ms - offset_ms
                )

    return LateAdc


MIXED_SPOKEN = ("hello", "quokka", "narwhal")


@pytest.mark.parametrize("offset_ms", [0, 1_000], ids=["equal-clocks", "audio-1000-behind"])
async def test_adversarial_a_muted_mixed_turn_drops_every_spoken_word(
    tmp_path: Path, offset_ms: int
) -> None:
    """A0-R3/A-RV1: ``hello quokka narwhal`` is spoken at session 2500 (audio 2500 - offset); at session 3200 a
    question is typed (ENDPOINTING), and ``hello`` arrives inside ``eot_settle_ms`` so the turn returns to LISTENING:
    a mixed turn, which is a voice turn. The mute then drops it, and none of its spoken words reaches any later
    ``llm.start`` or history entry, with equal clocks and with the audio clock 1000 ms behind. The fence's lower
    bound must be on the audio clock (the earliest audio onset the session saw for the turn); a lower bound taken from
    the typed onset (session clock) lets the late spoken words through as a new turn."""
    async with mic_rig(
        tmp_path,
        audio=lagging_audio(offset_ms),
        stt=reactive_stt(steps=False),
        script=[answer(ANSWER), answer("B. ")],
    ) as rig:
        rig.say(" ".join(MIXED_SPOKEN), 2_500 - offset_ms)
        await rig.advance(3_200, step=10)
        await rig.type_question("typed bit")
        assert rig.state.state is TurnState.ENDPOINTING, f"premise: typed: {rig.state.state}"
        await rig.until(
            lambda: rig.state.state is not TurnState.ENDPOINTING,
            "premise: the typed turn never left ENDPOINTING",
            limit_ms=2_000,
            step=10,
        )
        assert rig.state.state is TurnState.LISTENING, (
            f"premise (A0-R3): the mic onset reopened the typed turn; {rig.state.state}"
        )
        since = len(rig.machine.log)
        rig.send_mute(True)
        await rig.pump()
        await rig.advance(1_500, step=20)
        await rig.release()
        await rig.advance(3_000, step=20)
        starts = [c.transcript for c in rig.commands_of("llm.start")]
        history = [h.user_text for h in rig.state.history]
        after = [ev.text for ev in rig.stepped("stt.word", since)]
        state, held = rig.state.state, [w.text for w in rig.state.words]
    came_back = [w for w in MIXED_SPOKEN if w in after]
    assert not came_back, (
        f"A0-R3/A-RV1: spoken words {came_back} of the muted mixed turn reached the machine after the mute "
        f"(stt.word stepped after MicMuted: {after}); the turn now holds {held} in {state.name}"
    )
    leaked = [w for w in MIXED_SPOKEN if w in " ".join([*starts, *history]).split()]
    assert not leaked, (
        f"A0-R3/A-RV1: spoken words {leaked} of the muted mixed turn came back; llm.start {starts}, "
        f"history {history}"
    )
    assert state is TurnState.IDLE and held == [], (
        f"A0-R3: IDLE with no held words; {state}, {held}"
    )
