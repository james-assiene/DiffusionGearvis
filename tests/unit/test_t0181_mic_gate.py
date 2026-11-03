"""T-0181 group A, A5 and A6: the gate on the STT tap, and no word from a muted span reaches the machine.

docs/tasks/T-0181.md, A5, A6, A-R3, A-R5, A-R12 ("Tests to write -> Group A -> test_t0181_mic_gate.py").

The rig is ``tests/unit/_t0181_rig.mic_rig``: the session over the fakes, tools off, a socketless ``HudServer`` with one
greeted page, ``FakeAudioIO`` capture (80 ms STT frames of 1920 samples on an audio clock that starts with the session
clock at 0; a frame with ``t_ms = T`` is complete once the clock passes ``T + 80``) and a *reactive* ``FakeStt``
(a scripted word is released once fed audio passes ``end_ms + 500``). By default the fake is **content-blind** (it
releases a word whatever the fed samples were), so a word the session must drop is still delivered and only the
session can drop it; with ``hears_audio=True`` an all-zero fed frame is silence and a word starting in one is never
emitted (``tests/fakes/stt.py``). Mutes go over the wire as raw JSON; before group A lands the codec answers
``Unknown``, no frame is ever zeroed, and every row fails on ``engage``'s named A5 assertion (or on its first own
assertion), never at collection.

**Oracles.** ``TapAudio`` keeps every frame the capture yields, packed as the pump packs it (little-endian f32 bytes,
integer ``t_ms``): one feed per captured frame, same ``t_ms``, same length, zeros exactly while muted and the captured
bytes otherwise. The span is read off what was fed (``zero_spans``): its first and last zeroed frame's ``t_ms`` (A6).
Word times are scripted by hand relative to those bounds.

**Readings** (listed in the report):

- A5's "takes effect at the first frame the pump handles after ``apply_mic_mute`` returns": ``hud_server.route`` is
  synchronous, so ``apply_mic_mute`` has returned when ``page.receive`` does; the rows mute right after a settle (the
  pump is waiting on the capture with nothing in hand), so the first frame fed after ``receive`` returns is the first
  frame handled after the call, and it must be zero (and, on unmute, the captured bytes);
- A6's span bounds (A-R5): "inside" is a ``start_ms`` in ``[first zeroed t_ms, last zeroed t_ms]``; a word starting at
  ``first - 1`` is outside, a word starting at the first live frame (``last + 80``) is outside; no row scripts a word
  starting in ``(last, last + 80)``;
- A-R3/A-R12: the drop-frames row lives in PAUSED, and the row submits the barge-in's onset itself (as the reflex lane
  would); the barge-in's words heard before the span are answered (H-T181-9), so the relistened turn's transcript is
  exactly the words whose ``start_ms`` precede the first zeroed frame.

Regime: CPU over fakes, ``ScriptedClock``, ``TurnPolicy(thinking=True)``, tools off, ``RecordingTts`` 200 ms a word,
``FakeAudioIO`` capture, reactive ``FakeStt`` (``lag_ms`` 500), n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

from pathlib import Path

from local_astra.contracts.turn import TurnState
from tests.unit._t0181_rig import (
    BARGE,
    GATE_NOTE,
    LONG_ANSWER,
    QUESTION,
    MicRig,
    TapAudio,
    barge_in,
    mic_rig,
    reactive_stt,
    zero_spans,
)
from tests.unit._tool_turn_support import answer

ANSWER = "A shop. "
#: STT frame length in ms (1920 samples at 24 kHz; ``tests/fakes/audio.STT_BLOCK_MS``)
FRAME_MS = 80
#: bytes per fed frame: 1920 little-endian f32
FRAME_BYTES = 1920 * 4


def assert_nothing_heard(rig: MicRig, since: int, what: str) -> None:
    """A6: no onset, no word, no transcript, no turn reached the machine or the page after machine-log index ``since``."""
    onsets = rig.stepped("user.speech_onset", since)
    words = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
    assert (onsets, words) == ([], []), (
        f"A6 ({what}): a word from the muted span reached the machine: onsets {onsets}, words {words}"
    )
    assert rig.transcripts() == [], f"A6 ({what}): a transcript was published; {rig.transcripts()}"
    assert rig.commands_of("llm.start") == [], f"A6 ({what}): a turn was asked"
    assert rig.lines() == [], f"A6 ({what}): a turn line was written; {rig.lines()}"
    assert rig.state.state is TurnState.IDLE, f"A6 ({what}): {rig.state.state}"


async def advance_to(rig: MicRig, at_ms: int, step: int = 40) -> None:
    left = at_ms - rig.clock.now_ms
    if left > 0:
        await rig.advance(left, step=step)


# =====================================================================================================
# A5: the gate, frame by frame
# =====================================================================================================


async def test_every_frame_fed_while_muted_is_zero_with_the_captured_t_ms_length_and_count(
    tmp_path: Path,
) -> None:
    """A5: two mute/unmute cycles. Against ``TapAudio`` (every frame the capture yielded): one feed per captured frame
    (same count, same ``t_ms`` sequence, same byte length), every frame fed between a mute and the next unmute is all
    zeros, and every other frame is the captured frame bit for bit. The boundaries are exact: the first frame fed
    after ``receive`` returns is zero on a mute and the captured bytes on an unmute (the decision is per frame, at
    handling time). A "pause the pump" gate fails the count; a gate one frame late fails the boundary."""
    async with mic_rig(tmp_path, audio=TapAudio) as rig:
        await rig.advance(400, step=40)
        bounds: list[tuple[int, bool]] = []
        for muted, hold in ((True, 640), (False, 400), (True, 320), (False, 480)):
            rig.send_mute(muted)
            bounds.append((len(rig.rec_stt.fed), muted))
            await rig.pump()
            await rig.advance(hold, step=40)
        await rig.pump()
        fed = list(rig.rec_stt.fed)
        yielded = list(rig.audio.yielded)
    assert len(fed) == len(yielded), (
        f"A5: one feed per captured frame; {len(yielded)} frames captured, {len(fed)} fed"
    )
    assert [t for _, t in fed] == [t for _, t in yielded], (
        "A5: the fed t_ms sequence is the captured one"
    )
    assert {len(pcm) for pcm, _ in fed} == {FRAME_BYTES}, (
        "A5: every fed frame keeps the captured length"
    )
    want_zero = [False] * len(fed)
    for (start, muted), (end, _) in zip(bounds, [*bounds[1:], (len(fed), False)], strict=True):
        assert start < end, f"premise: frames were fed in every phase; bounds {bounds}"
        for i in range(start, end):
            want_zero[i] = muted
    got_zero = rig.rec_stt.zero()
    wrong = [(fed[i][1], want_zero[i]) for i in range(len(fed)) if got_zero[i] != want_zero[i]]
    assert not wrong, (
        f"A5: zero exactly from the first frame handled after a mute to the last before the unmute; "
        f"(t_ms, should be zero) of every frame that is not: {wrong[:8]}"
    )
    altered = [
        fed[i][1] for i in range(len(fed)) if not want_zero[i] and fed[i][0] != yielded[i][0]
    ]
    assert not altered, f"A5: a live frame was altered (t_ms {altered[:8]})"


async def test_the_first_frame_after_unmute_is_the_captured_frame(tmp_path: Path) -> None:
    """A5: on unmute the very next frame fed is bit-identical to the captured frame (no fade, no burst, no backlog):
    its bytes and ``t_ms`` equal the capture's, it directly follows the last zeroed frame (``t_ms`` + 80), and the
    number of frames fed equals the number captured (nothing buffered while muted is replayed)."""
    async with mic_rig(tmp_path, audio=TapAudio) as rig:
        await rig.advance(400, step=40)
        await rig.engage()
        await rig.advance(560, step=40)
        rig.send_mute(False)
        mark = len(rig.rec_stt.fed)
        await rig.pump()
        await rig.advance(240, step=40)
        fed = list(rig.rec_stt.fed)
        yielded = list(rig.audio.yielded)
    assert len(fed) > mark, "premise: frames were fed after the unmute"
    assert not any(fed[mark - 1][0]), "premise: the frame before the unmute was zeroed"
    assert fed[mark] == yielded[mark] and any(fed[mark][0]), (
        f"A5: the first frame after the unmute (t_ms {fed[mark][1]}) is the captured frame, bit for bit"
    )
    assert fed[mark][1] == fed[mark - 1][1] + FRAME_MS, (fed[mark - 1][1], fed[mark][1])
    assert len(fed) == len(yielded), (len(fed), len(yielded))


# =====================================================================================================
# A6: no word from a muted span reaches the machine
# =====================================================================================================


async def test_an_utterance_inside_the_span_is_unheard_and_opens_nothing(tmp_path: Path) -> None:
    """A6 with ``hears_audio=True``: ``where is my umbrella`` is spoken inside the span (first word 160 ms after the
    first zeroed frame, the unmute ~1.5 s after the last word starts). The recogniser hears only zeros there, so
    nothing comes back: no onset, no ``stt.word``, no transcript, no ``stt.decision`` record, no turn, IDLE. Without
    the gate the same session hears the words and opens a turn."""
    async with mic_rig(tmp_path, hears_audio=True, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)
        since = len(rig.machine.log)
        first_zero = await rig.engage()
        spoken = rig.say("where is my umbrella", first_zero + 160)
        await advance_to(rig, spoken[-1].start_ms + 1_500)
        await rig.release()
        await rig.advance(2_000)
        ((span_first, span_last),) = zero_spans(rig.rec_stt)
        assert all(span_first <= w.start_ms <= span_last for w in spoken), (
            f"premise: every word starts inside the span {span_first}..{span_last}"
        )
        assert_nothing_heard(rig, since, "hears_audio")
    assert rig.kind("stt.decision") == [], f"A6: no stt.decision; {rig.kind('stt.decision')}"


async def test_adversarial_a_content_blind_word_inside_the_span_is_dropped_and_counted(
    tmp_path: Path,
) -> None:
    """A6 (adversarial, absent): the fake is content-blind, so it *delivers* ``pelican`` and ``saxophone`` (both
    starting inside the span, both released while still muted) and ``marmalade`` (starting inside the span, scripted
    once the span has closed and released after the unmute). The session drops all three before the endpointer: no
    onset, no ``stt.word``, no transcript, no ``stt.decision``, no turn. The unmute's ``mic.gate`` record counts the
    two dropped before it (``words_dropped == 2``; a word dropped after the record is written cannot be in it, so the
    row does not say where ``marmalade`` is counted), and no dropped text appears in the session log or in any
    published message."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)
        since = len(rig.machine.log)
        first_zero = await rig.engage()
        before = rig.say("pelican saxophone", first_zero + 160)
        await advance_to(rig, before[-1].end_ms + 500 + 160)
        await rig.release()
        ((span_first, span_last),) = zero_spans(rig.rec_stt)
        (late,) = rig.say("marmalade", span_last - 120)
        await rig.advance(2_000)
        spoken = [*before, late]
        assert all(span_first <= w.start_ms <= span_last for w in spoken), (
            f"premise: every word starts inside the span {span_first}..{span_last}; {spoken}"
        )
        assert all(w.end_ms + 500 <= span_last for w in before), (
            "premise: pelican and saxophone are released while muted"
        )
        assert_nothing_heard(rig, since, "content-blind")
        published = rig.published_text()
    assert rig.kind("stt.decision") == [], f"A6: no stt.decision; {rig.kind('stt.decision')}"
    gates = rig.notes(GATE_NOTE)
    assert [g.get("muted") for g in gates] == [True, False], gates
    assert gates[1].get("words_dropped") == len(before), (
        f"A6: the unmute's mic.gate counts the words dropped while muted ({len(before)}); got {gates[1]}"
    )
    text = rig.log_text()
    for w in spoken:
        assert w.text not in text, f"A6: the dropped word {w.text!r} is in the session log"
        assert w.text not in published, f"A6: the dropped word {w.text!r} was published"


async def test_too_wide_a_word_starting_one_ms_before_the_span_is_kept(tmp_path: Path) -> None:
    """A6 (too wide, the lower bound): ``alpha`` starts at the first zeroed frame's ``t_ms`` - 1 (most of its audio is
    inside the span): it is kept, reaches the machine as an ``stt.word`` with its own times and is the onset of a turn.
    ``bravo``, starting exactly at the first zeroed ``t_ms``, is inside and is dropped. An implementation keyed on a
    word's end, or on any overlap with the span, drops ``alpha``; one keyed on ``>`` instead of ``>=`` keeps ``bravo``."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)
        since = len(rig.machine.log)
        first_zero = await rig.engage()
        (alpha,) = rig.say("alpha", first_zero - 1)
        (bravo,) = rig.say("bravo", first_zero)
        await rig.advance(1_200)
        words = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
        onsets = [ev.onset_ms for ev in rig.stepped("user.speech_onset", since)]
    assert words == [("alpha", alpha.start_ms)], (
        f"A6: alpha (start {alpha.start_ms}, the span from {first_zero}) is kept and bravo (start {bravo.start_ms}) "
        f"dropped; words stepped {words}"
    )
    assert onsets == [alpha.start_ms], f"A6: alpha is the onset; onsets {onsets}"


async def test_too_wide_a_word_at_the_first_live_frame_is_kept(tmp_path: Path) -> None:
    """A6 (too wide, the upper bound, A-R5): after an unmute, ``charlie`` starts at the first live frame's ``t_ms``
    (last zeroed + 80): it is kept and is the onset. ``delta``, starting exactly at the last zeroed frame's ``t_ms``
    and scripted first, is inside and dropped (otherwise it would be the onset, 80 ms earlier)."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)
        await rig.engage()
        await rig.advance(800, step=40)
        first_live = await rig.release()
        ((_, span_last),) = zero_spans(rig.rec_stt)
        assert first_live == span_last + FRAME_MS, (span_last, first_live)
        since = len(rig.machine.log)
        (delta,) = rig.say("delta", span_last)
        (charlie,) = rig.say("charlie", first_live)
        await rig.advance(1_200)
        words = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
        onsets = [ev.onset_ms for ev in rig.stepped("user.speech_onset", since)]
    assert words == [("charlie", charlie.start_ms)], (
        f"A6: charlie (the first live frame, {first_live}) is kept and delta (the last zeroed frame, "
        f"{delta.start_ms}) dropped; words stepped {words}"
    )
    assert onsets == [charlie.start_ms], f"A6: charlie is the onset; onsets {onsets}"


async def test_a_closed_span_includes_its_first_zeroed_frame(tmp_path: Path) -> None:
    """A6 (A-R5, A-RV3: the closed span's lower bound is inclusive): mute ~800 ms, unmute, and only then script
    ``echo`` starting exactly at the span's first zeroed ``t_ms`` and ``alpha`` starting 1 ms before it (scripted
    first). Both are delivered after the unmute, so only the closed span can decide them: ``echo`` is inside and
    dropped, ``alpha`` is outside and kept as the onset (the too-wide companion). A closed span checked with ``<`` on
    its lower bound keeps ``echo``."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)
        await rig.engage()
        await rig.advance(800, step=40)
        first_live = await rig.release()
        ((span_first, span_last),) = zero_spans(rig.rec_stt)
        assert first_live == span_last + FRAME_MS, (span_last, first_live)
        since = len(rig.machine.log)
        (alpha,) = rig.say("alpha", span_first - 1)
        (echo,) = rig.say("echo", span_first)
        assert echo.end_ms + 500 <= first_live + FRAME_MS, (
            "premise: echo is due within a frame of the unmute (it is decided by the closed span)"
        )
        await rig.advance(1_200)
        words = [(ev.text, ev.start_ms) for ev in rig.stepped("stt.word", since)]
        onsets = [ev.onset_ms for ev in rig.stepped("user.speech_onset", since)]
    assert words == [("alpha", alpha.start_ms)], (
        f"A6/A-R5: echo (start {echo.start_ms} = the closed span's first zeroed frame) is dropped and alpha "
        f"(start {alpha.start_ms}) kept; words stepped {words}"
    )
    assert onsets == [alpha.start_ms], f"A6: alpha is the onset; onsets {onsets}"


# =====================================================================================================
# A5 adversarial (A-R3): a drop-frames gate hangs a mute in PAUSED
# =====================================================================================================


async def test_adversarial_a_mute_in_paused_keeps_feeding_and_the_barge_in_is_answered(
    tmp_path: Path,
) -> None:
    """A5 / A-R3 / H-T181-9: the assistant is speaking a long answer; the barge-in ``wait stop that please`` starts at
    audio N and the row submits its onset (A-R12), so the machine pauses; the mute comes 300 ms later, in PAUSED (no
    row). The takeover relistens into a LISTENING turn that must still end: its end of turn needs the recogniser's
    steps and the ``Marker`` echo, and both need fed audio, so the zeros must keep flowing. The relistened turn is
    asked with exactly the barge-in words whose audio starts before the first zeroed frame, and the session is IDLE
    with two turn lines within 15 s of the mute. A gate that drops frames (or pauses the pump) while muted never
    gets the words, the steps or the echo, and hangs."""
    async with mic_rig(
        tmp_path, audio=TapAudio, stt=reactive_stt(steps=True), script=[answer(LONG_ANSWER)]
    ) as rig:
        spoken = await barge_in(rig, BARGE, question=QUESTION)
        await rig.advance(300, step=20)
        assert rig.state.state is TurnState.PAUSED, (
            f"premise: PAUSED at the mute; {rig.state.state}"
        )
        muted_at = rig.clock.now_ms
        # the span's first bound read off the capture, not off what was fed: a gate that drops frames feeds no zero
        # at all, and this row must reach its hang rather than stop at the rig's "no zero frame"
        mark = len(rig.audio.yielded)
        await rig.mute(True)
        await rig.advance(120, step=40)
        assert len(rig.audio.yielded) > mark, "premise: a frame was captured after the mute"
        first_zero = rig.audio.yielded[mark][1]
        moved = [(b, a) for b, ev, a in rig.machine.transitions() if ev.type == "mic.muted"]
        assert moved == [(TurnState.PAUSED, TurnState.PAUSED)], (
            f"A-R3: one MicMuted in PAUSED, no row; got {moved}"
        )
        kept = [w.text for w in spoken if w.start_ms < first_zero]
        dropped = [w.text for w in spoken if w.start_ms >= first_zero]
        assert kept and dropped, (
            f"premise: the span splits the barge-in; first zero {first_zero}, {spoken}"
        )
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2,
            "A5/A-R3: the relistened turn never ended while muted (frames dropped or the pump paused?)",
            limit_ms=15_000 - (rig.clock.now_ms - muted_at),
        )
        starts = [c.transcript for c in rig.commands_of("llm.start")]
    assert starts == [QUESTION, " ".join(kept)], (
        f"A6/H-T181-9: the relistened turn is asked with the pre-span words {kept}; llm.start transcripts {starts}"
    )
