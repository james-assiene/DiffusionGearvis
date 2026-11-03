"""The listening path: the session hears the microphone (T-0032 group A, criteria 1, 2, 4, 5, 6, 8).

Everything here runs over ``tests/fakes/`` on the scripted clock, importing the house rig from
``tests/unit/test_session_loop.py`` (``ScriptedClock``, ``running``, ``answer``) exactly as
``tests/contract/test_session_audio_seam.py`` does.

The seam under test, stated so the implementer builds to it:

* ``Session.start()`` also starts a **capture pump**: ``async for frame in audio.stt_blocks()`` →
  ``await stt.feed(frame.samples.tobytes(), int(frame.t_ms))``, one call per frame, verbatim
  (criterion 1). It is a *producer* task: it never calls ``machine.step`` and never submits an
  event per frame (criterion 2).
* the endpointer (``reflex/turn_predictor.py``, ``tests/unit/test_endpointer.py`` fixes its API)
  is consulted inside ``_pump_stt``, where the raw ``SttEvent`` still carries its **audio** time,
  and its decisions become ``E.UserSpeechOnset(t_ms=<clock>, onset_ms=<audio>)`` and
  ``E.SttEndOfTurn(t_ms=<clock>, t_eot_ms=<audio>, p_end=…)`` (criterion 4).
* after submitting the end of turn the session awaits ``stt.mark(t_eot_ms)`` **once**, from a
  producer task (criterion 5). The only ``SttFlushed`` the machine ever sees is the one
  ``_pump_stt`` maps from the client's ``SttFlushedEvent``.
* the turn line carries ``stt_frames_fed``, ``stt_feed_errors`` and, when the audio object has
  ``capture_stats()``, ``capture_xruns``/``capture_gated_blocks``, plus the existing
  ``reflex.onset`` span (criterion 8).

Two live facts the fakes are kinder than (T-0007a census; they change no assertion here, and
belong in the implementer's report): the real server echoes a ``Marker`` **only while audio keeps
arriving** -- 3.0 s after the mark in census run 3 on CPU STT -- so the capture pump must keep
feeding after the end of turn or ENDPOINTING never leaves on hardware; and the server sends
``EndWord`` (T-0038), so until that lands a real word's ``end_ms`` equals its ``start_ms``.

Regime (group A): ``FakeAudioIO`` capture at 48 kHz / 10 ms → 1920-sample 80 ms STT frames,
``FakeStt(reactive=True, steps=True)`` with a 500 ms word lag and 80 ms steps,
``TurnPolicy(thinking=True)``, ``eot_settle_ms=120``, virtual time only, n = 1 deterministic per
row. No latency number here is quotable.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.fakes.audio import FakeSttFrame
from tests.fakes.stt import FakeStt
from tests.unit.test_session_loop import (
    RecordingAudio,
    ScriptedClock,
    answer,
    frame,
    make_rig,
    running,
)

REPO = Path(__file__).resolve().parents[2]

#: ``docs/tasks/T-0032.md``, "Owned files".
OWNED_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/reflex/turn_predictor.py",
    "local_astra/contracts/protocols.py",
    "local_astra/cli.py",
    "scripts/demo/session_rung.py",
)

QUESTION = "what is on this screen"
SPEECH_START_MS = 1000
MS_PER_WORD = 200
#: hand-computed: five words of 200 ms from 1000 ms → the run is [1000, 2000] in *audio* time.
SPEECH_END_MS = SPEECH_START_MS + len(QUESTION.split()) * MS_PER_WORD
#: The offset criterion 4's row puts between the audio clock and the session clock.
AUDIO_OFFSET_MS = 5_000_000
ANSWER = [answer("It is a stack trace.")]


def reactive_stt(text: str = QUESTION, start_ms: int = SPEECH_START_MS) -> FakeStt:
    """A fake that says ``text`` **only if the session feeds it the audio** (T-0032's point)."""
    stt = FakeStt(reactive=True, steps=True, lag_ms=500)
    stt.say(text, start_ms=start_ms, ms_per_word=MS_PER_WORD)
    return stt


class RecordingStt:
    """Delegating wrapper over any ``SttClient``: records what was fed and what was marked."""

    def __init__(self, inner: Any, *, fail_on: set[int] | None = None, echo: bool = True) -> None:
        self.inner = inner
        self.fed: list[tuple[bytes, int]] = []
        self.marks: list[int] = []
        self.fail_on = fail_on or set()
        self.echo = echo
        self.gate: asyncio.Event | None = None

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        self.fed.append((pcm_24k_f32, t_ms))
        if len(self.fed) in self.fail_on:
            raise RuntimeError("moshi-server went away for one frame")
        await self.inner.feed(pcm_24k_f32, t_ms)

    async def mark(self, t_eot_ms: int) -> int:
        self.marks.append(int(t_eot_ms))
        if self.gate is not None:
            await self.gate.wait()
        if not self.echo:  # a server that never echoes the Marker (criterion 5, negative)
            return 0
        return await self.inner.mark(t_eot_ms)

    def events(self) -> Any:
        return self.inner.events()

    def stats(self) -> Any:
        return self.inner.stats()

    async def close(self) -> None:
        await self.inner.close()

    @property
    def closed(self) -> bool:
        return bool(self.inner.closed)

    def t_ms(self) -> list[int]:
        return [t for _, t in self.fed]


class OffsetAudio(RecordingAudio):
    """``FakeAudioIO`` whose STT frames carry an audio clock far from the session clock."""

    async def stt_blocks(self) -> Any:
        async for item in super().stt_blocks():
            yield replace(item, t_ms=item.t_ms + AUDIO_OFFSET_MS)


class OddFrameAudio(RecordingAudio):
    """Frames with a **fractional** ``t_ms`` and a **non-contiguous** samples array.

    Both are real shapes: ``audio_capture`` computes ``t_ms`` from a sample index over 24000, and
    a decimated view (``block[0::2]``) is not C-contiguous. ``tobytes()`` copes; a ``memoryview``
    or a ``frombuffer`` on the array's buffer does not.
    """

    async def stt_blocks(self) -> Any:
        base = np.arange(3840, dtype=np.float32) / 3840.0
        for index in range(24):
            samples = base[index % 2 :: 2]  # a strided view: samples.flags["C_CONTIGUOUS"] is False
            assert not samples.flags["C_CONTIGUOUS"] and samples.size == 1920
            yield FakeSttFrame(
                samples=samples,
                t_ms=index * 80 + 0.5,
                end_ms=index * 80 + 80.5,
                yielded_ms=float(self.clock.now_ms),
            )
            await asyncio.sleep(0)
        while True:  # the capture stream does not end; the session's aclose() ends it
            await asyncio.sleep(0)


class DeafAudio:
    """An audio object with **no capture seam at all** — exactly the ``AudioIO`` protocol.

    T-0021 ruling A1 kept ``stt_blocks``/``capture_stats`` off the protocol, so this is a legal
    ``AudioIO`` and the session must degrade, not crash (criterion 2, adversarial).
    """

    def __init__(self, inner: RecordingAudio) -> None:
        self.inner = inner

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        return await self.inner.flush()

    def played_ms(self) -> float:
        return self.inner.played_ms()

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def is_rendering(self, t_ms: float) -> bool:
        return self.inner.is_rendering(t_ms)

    def playback_events(self) -> Any:
        return self.inner.playback_events()

    def capture(self) -> Any:
        return self.inner.capture()


class ClosingAudio(RecordingAudio):
    """An audio object that **has** the ``close_capture()`` seam, and counts what it sees.

    ``FakeAudioIO`` does not implement ``close_capture`` (T-0021 left it on the real class), so
    without this double the ``isinstance(audio, CaptureCloser)`` branch of ``Session.aclose()`` is
    never taken in the fast suite — the seam that pads the STT tail (T-0021 criterion 7) would be
    free to be wrong, or to be called while the pump is still feeding, and nothing would say so
    (T-0032 ruling A12).

    ``frames_yielded`` is incremented by the *generator*, so it rises only while something is
    iterating ``stt_blocks()``: its value at the moment ``close_capture()`` returns, compared with
    its value after more virtual time has passed, is the test's evidence that the pump was already
    cancelled. ``DeafAudio`` above stays the other half of the pair: no seam at all.
    """

    def __init__(self, clock: ScriptedClock) -> None:
        super().__init__(clock)
        self.frames_yielded = 0
        #: ``frames_yielded`` as it stood at each ``close_capture()`` call (so: one entry per call)
        self.close_calls: list[int] = []
        #: whether ``aclose()`` had started at each call — the test sets the flag itself
        self.closed_during_aclose: list[bool] = []
        #: whether the capture task was already finished at each call (``None``: not found)
        self.pump_done_at_close: list[bool | None] = []
        self.aclose_started = False
        self.pump_task: Any = None

    async def stt_blocks(self) -> Any:
        async for item in super().stt_blocks():
            self.frames_yielded += 1
            yield item

    def close_capture(self) -> None:
        self.close_calls.append(self.frames_yielded)
        self.closed_during_aclose.append(self.aclose_started)
        task = self.pump_task
        self.pump_done_at_close.append(None if task is None else bool(task.done()))


def capture_task() -> Any:
    """The session's capture task, found by the name ``session.py`` gives it (best effort)."""
    for task in asyncio.all_tasks():
        if task.get_name() == "session-capture":
            return task
    return None


def watch_submits(rig: Any) -> list[E.Event]:
    """Record every event **the test itself** submits; the session's own submits are not seen."""
    seen: list[E.Event] = []
    session_submit = rig.session.submit

    def submit(event: E.Event) -> None:
        seen.append(event)
        session_submit(event)

    rig.submit = submit
    return seen


def states_seen(rig: Any) -> list[TurnState]:
    return [after for _, _, after in rig.machine.transitions()]


def covers(path: list[TurnState], wanted: list[TurnState]) -> bool:
    """``wanted`` appears in ``path`` in order (other states may sit between)."""
    it = iter(path)
    return all(any(state is target for state in it) for target in wanted)


async def run_one_spoken_turn(rig: Any, *, limit_ms: int = 8_000) -> None:
    await rig.until(
        lambda: TurnState.SPEAKING in states_seen(rig),
        "criterion 6: the spoken question never reached SPEAKING — the session is deaf (no "
        "capture pump, or no endpointer decision reached the machine)",
        limit_ms=limit_ms,
    )
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE, "the turn never returned to IDLE", limit_ms=5_000
    )


# --- criterion 6: the headline ----------------------------------------------------------------


async def test_a_spoken_question_over_the_fakes_reaches_thinking_and_speaking(tmp_path):
    """**The test whose absence hid the deafness.**

    The only inputs are audio and virtual time: ``FakeAudioIO`` produces capture blocks, the
    session feeds them, and ``FakeStt`` releases the words and the 80 ms pause-head steps against
    the audio it was fed. The test submits **no** ``user.speech_onset``, ``stt.word``,
    ``stt.end_of_turn`` or ``stt.flushed`` of its own — the recorded list of its own submissions
    is asserted to be empty — and the machine must still walk
    IDLE → LISTENING → ENDPOINTING → THINKING → SPEAKING → IDLE and write one turn line.
    """
    stt = RecordingStt(reactive_stt())
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        submitted = watch_submits(rig)
        await run_one_spoken_turn(rig)

        assert submitted == [], (
            "criterion 6: the test must submit nothing; it submitted "
            f"{[type(e).__name__ for e in submitted]}"
        )
        path = states_seen(rig)
        assert covers(
            path,
            [
                TurnState.LISTENING,
                TurnState.ENDPOINTING,
                TurnState.THINKING,
                TurnState.SPEAKING,
                TurnState.IDLE,
            ],
        ), f"criterion 6: the state path was {[s.name for s in path]}"

        started = rig.machine.moved(TurnState.ENDPOINTING, TurnState.THINKING)
        assert len(started) == 1 and isinstance(started[0], E.Tick), (
            "criterion 6: ENDPOINTING is left on the settle Tick after the flush (the existing "
            f"row), not on any new mechanism; it was left on {started}"
        )
        onsets = rig.machine.events(E.UserSpeechOnset)
        assert len(onsets) == 1 and onsets[0].onset_ms == SPEECH_START_MS, (
            f"the onset came from the STT stream's audio time ({SPEECH_START_MS}); got {onsets}"
        )
        eots = rig.machine.events(E.SttEndOfTurn)
        assert [e.t_eot_ms for e in eots] == [SPEECH_END_MS], (
            f"criterion 3(b): t_eot is the last word's end_ms ({SPEECH_END_MS}); got {eots}"
        )
        assert len(rig.lines()) == 1, f"exactly one turns.jsonl line; got {rig.lines()}"
        history = rig.state.history
        assert len(history) == 1 and history[0].user_text == QUESTION, (
            f"criterion 6: the turn's transcript is the spoken sentence; history is {history}"
        )
        assert stt.marks == [SPEECH_END_MS], (
            f"criterion 5: one mark(t_eot) for the turn; got {stt.marks}"
        )


# --- criterion 1: every frame, verbatim, in order -----------------------------------------------


async def test_every_stt_frame_is_fed_once_in_order_and_byte_identical(tmp_path):
    """The T-0021 seam: ``session.py`` resamples, reframes, retimes and gates **nothing**.

    Part one drives the ordinary fake capture and compares the fed bytes with the frames the fake
    yielded (``np.frombuffer`` round trip) and the fed times with the frames' own ``t_ms``. Part
    two is adversarial: a rig whose frames carry a *fractional* ``t_ms`` and a *non-contiguous*
    samples array — the two shapes the real capture module produces — must survive the conversion
    with the same byte identity and an ``int()`` (not a round) of the time.
    """
    stt = RecordingStt(FakeStt(reactive=True))  # no script: nothing comes back, audio still goes
    async with running(tmp_path, stt=stt, frames=[frame(0, "a")]) as rig:
        await rig.advance(2_000, step=100)
        yielded = rig.audio._cap_frames[: len(stt.fed)]
        assert len(stt.fed) >= 20, (
            "criterion 1: the capture pump must feed every 80 ms frame; after 2 s of virtual time "
            f"the STT client was handed {len(stt.fed)} frames"
        )
        for index, ((pcm, t_ms), source) in enumerate(zip(stt.fed, yielded, strict=False)):
            assert np.array_equal(np.frombuffer(pcm, dtype="<f4"), source.samples), (
                f"frame {index}: the bytes fed are not the frame's samples (little-endian f32)"
            )
            assert t_ms == int(source.t_ms), (
                f"frame {index}: t_ms is the frame's own capture time int({source.t_ms}), not "
                f"{t_ms} — the session never retimes (docs/02 §4 spec delta T-0021 (3))"
            )
        times = stt.t_ms()
        assert times == sorted(times), f"the fed times are non-decreasing; got {times[:12]}…"
        assert len(set(times)) == len(times), "…and no frame is fed twice"

    odd = RecordingStt(FakeStt(reactive=True))
    clock = ScriptedClock()
    (tmp_path / "odd").mkdir()
    async with running(
        tmp_path / "odd",
        stt=odd,
        frames=[frame(0, "a")],
        clock=clock,
        audio=OddFrameAudio(clock),
    ) as rig:
        await rig.advance(1_000, step=100)
        assert len(odd.fed) >= 20, (
            f"the adversarial rig fed {len(odd.fed)} frames; its 24 frames must all arrive"
        )
        base = np.arange(3840, dtype=np.float32) / 3840.0
        for index, (pcm, t_ms) in enumerate(odd.fed[:24]):
            expected = base[index % 2 :: 2]
            assert np.array_equal(np.frombuffer(pcm, dtype="<f4"), expected), (
                f"frame {index}: a non-contiguous samples view must be copied, not reinterpreted"
            )
            assert t_ms == index * 80, (
                f"frame {index}: t_ms = int({index * 80 + 0.5}) = {index * 80}; got {t_ms}"
            )


# --- criterion 2: the pump survives a failing feed ---------------------------------------------


async def test_a_feed_that_raises_is_counted_and_the_pump_survives(tmp_path):
    """Three failing frames out of ~35: the turn still completes and the line says "3".

    A capture pump that died on the first failure would make the session deaf for the rest of the
    session, silently — which is the same defect this task is closing, one layer down.
    """
    stt = RecordingStt(reactive_stt(), fail_on={3, 6, 9})
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        watch_submits(rig)
        await run_one_spoken_turn(rig)

        record = rig.records()[0]
        assert record.extras.get("stt_feed_errors") == 3, (
            f"criterion 8: the failures are counted, all three of them; extras were {record.extras}"
        )
        assert record.extras.get("stt_frames_fed", 0) >= 20, (
            f"…and the frames that did go through are counted too; extras were {record.extras}"
        )
        assert record.outcome == "spoken", (
            f"criterion 2: the turn completed in spite of the failures; outcome {record.outcome}"
        )


async def test_an_audio_object_without_the_capture_seam_still_runs_a_turn(tmp_path, caplog, capsys):
    """**Adversarial:** ``stt_blocks`` is not on the ``AudioIO`` protocol (T-0021 ruling A1).

    An audio object that has only the protocol must leave the session fully functional: it must
    not raise at ``start()``, the scripted (typed) path must still answer a question, and the
    absence must be said **once** — a per-frame complaint is a log nobody reads.
    """
    clock = ScriptedClock()
    deaf = DeafAudio(RecordingAudio(clock))
    caplog.set_level(logging.DEBUG)
    async with running(
        tmp_path, script=ANSWER, frames=[frame(0, "a")], clock=clock, audio=deaf
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()

        assert rig.session.failures == [], (
            f"criterion 2: no capture seam is not an error; the session recorded {rig.session.failures}"
        )
        assert len(rig.lines()) == 1, f"the turn still happens; lines {rig.lines()}"
        record = rig.records()[0]
        for key in ("stt_frames_fed", "capture_xruns", "capture_gated_blocks"):
            assert record.extras.get(key, 0) == 0, (
                f"criterion 8, negative: absent a capture seam, {key} is absent or zero, never "
                f"invented; extras were {record.extras}"
            )
        said = [r.getMessage() for r in caplog.records if "capture" in r.getMessage().lower()]
        said += [
            line
            for line in capsys.readouterr().out.splitlines()
            if "capture" in line.lower() or "stt_blocks" in line
        ]
        assert len(said) == 1, (
            f"criterion 2: the absence of the capture seam is logged exactly once; it was said "
            f"{len(said)} times: {said[:4]}"
        )


# --- criterion 5: mark, once, off the pump, and never a synthesised flush -----------------------


async def test_mark_is_called_once_per_end_of_turn_with_the_same_t_eot(tmp_path):
    """One ``mark`` per end of turn, from a producer task, and again after a re-entry.

    The mark is held (its ``Event`` is cleared) while the first end of turn is outstanding: the
    pump must keep stepping the machine during that await. A ``mark()`` awaited inside ``_handle``
    or a ``CommandExecutor`` handler stops the clock for the whole session, which is the one thing
    the "one pump" rule exists to prevent (docs/02 §4; ``session.py``'s module docstring).
    """
    stt = RecordingStt(reactive_stt())
    stt.gate = asyncio.Event()
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        watch_submits(rig)
        await rig.until(
            lambda: bool(stt.marks), "the session never marked the end of turn", limit_ms=8_000
        )
        assert stt.marks == [SPEECH_END_MS], f"criterion 5: mark(t_eot); got {stt.marks}"

        steps_before = rig.machine.steps
        await rig.advance(300, step=50)
        assert rig.machine.steps > steps_before, (
            "criterion 5: mark() is awaited in a producer task — the pump kept stepping the "
            f"machine while it was outstanding ({steps_before} steps, still {rig.machine.steps})"
        )
        assert stt.marks == [SPEECH_END_MS], (
            f"…and it is asked for exactly once per end of turn; got {stt.marks}"
        )

        stt.gate.set()
        await run_one_spoken_turn(rig)

        second_start = stt.inner.fed_ms + 300
        stt.inner.say("and this one too", start_ms=second_start, ms_per_word=MS_PER_WORD)
        second_eot = second_start + 4 * MS_PER_WORD
        await rig.until(
            lambda: len(stt.marks) == 2,
            "the second utterance never reached an end of turn",
            limit_ms=8_000,
        )
        assert stt.marks == [SPEECH_END_MS, second_eot], (
            "criterion 5: a re-entry to LISTENING marks the new t_eot, and only it; got "
            f"{stt.marks}"
        )


async def test_the_session_never_synthesises_a_flush(tmp_path):
    """**Adversarial:** the 2.5 s drain heuristic ``scripts/demo/_listen.py`` documents must stay dead.

    The STT double asks for the mark but the server never echoes it. The machine must therefore
    sit in ENDPOINTING for as long as the silence lasts: no timer, no "the words have stopped
    coming", no ``close()`` path may invent ``E.SttFlushed``. Ten seconds of virtual time — four
    times the old heuristic's window — must change nothing.
    """
    stt = RecordingStt(reactive_stt(), echo=False)
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        watch_submits(rig)
        await rig.until(
            lambda: rig.state.state is TurnState.ENDPOINTING,
            "the spoken question never reached ENDPOINTING",
            limit_ms=8_000,
        )
        await rig.advance(10_000, step=100)

        assert stt.marks == [SPEECH_END_MS], (
            f"the session did ask for the marker (otherwise this row is vacuous); {stt.marks}"
        )
        assert rig.machine.events(E.SttFlushed) == [], (
            "criterion 5, negative: the only SttFlushed is the one _pump_stt maps from the "
            f"client's SttFlushedEvent; the machine saw {rig.machine.events(E.SttFlushed)}"
        )
        assert TurnState.THINKING not in states_seen(rig), (
            "…so the turn cannot start thinking: the transcript is not known to be complete "
            f"(states seen: {[s.name for s in states_seen(rig)]})"
        )
        assert rig.state.state is TurnState.ENDPOINTING, (
            f"…and the machine waits, in ENDPOINTING; it is in {rig.state.state}"
        )
        assert rig.lines() == [], f"no turn line is written either; got {rig.lines()}"


# --- criterion 4: audio times, never clock times -----------------------------------------------


async def test_onset_and_end_of_turn_carry_audio_times_not_clock_times(tmp_path):
    """The audio clock is offset from the session clock by 5 000 000 ms, so they cannot be confused.

    docs/02 §4: "transcript segments are attributed by reflex-onset *audio* timestamps, never by
    state-entry wall time" — STT words lag audio by ~0.5 s, so a turn attributed by the clock
    snapshots the wrong frame and freezes at the wrong time. The delivery stamp ``t_ms`` stays the
    clock's, which is what the second half of each assertion pins.
    """
    stt = RecordingStt(reactive_stt(start_ms=AUDIO_OFFSET_MS + SPEECH_START_MS))
    clock = ScriptedClock()
    async with running(
        tmp_path,
        script=ANSWER,
        frames=[frame(0, "a")],
        stt=stt,
        clock=clock,
        audio=OffsetAudio(clock),
    ) as rig:
        watch_submits(rig)
        await run_one_spoken_turn(rig)

        onset = rig.machine.events(E.UserSpeechOnset)[0]
        eot = rig.machine.events(E.SttEndOfTurn)[0]
        assert onset.onset_ms == AUDIO_OFFSET_MS + SPEECH_START_MS, (
            f"criterion 4: onset_ms is the first word's audio start; got {onset.onset_ms}"
        )
        assert eot.t_eot_ms == AUDIO_OFFSET_MS + SPEECH_END_MS, (
            f"criterion 4: t_eot_ms is the last word's audio end; got {eot.t_eot_ms}"
        )
        for event in (onset, eot):
            assert event.t_ms < AUDIO_OFFSET_MS, (
                f"…while the delivery stamp t_ms stays the clock's; {event}"
            )
        assert onset.onset_ms != onset.t_ms and eot.t_eot_ms != eot.t_ms, (
            "criterion 4, negative: neither audio time may be clock.now_ms"
        )
        span = rig.records()[0].spans.get("reflex.onset")
        assert span is not None and span[0] >= AUDIO_OFFSET_MS, (
            f"the reflex.onset span is on the audio clock too (docs/02 §4); it is {span}"
        )


# --- criterion 8: the counters and the span ----------------------------------------------------


async def test_the_turn_line_carries_the_capture_counters_and_the_reflex_onset_span(tmp_path):
    """``stt_frames_fed``, ``stt_feed_errors``, ``capture_xruns``, ``capture_gated_blocks``, span.

    The frame count is checked against the wrapper's own tally (an independent oracle, not the
    session's number read back), and one 120 ms xrun is injected so the xrun counter cannot pass
    by always being zero. No new span name is added: ``record.unknown_spans()`` stays empty.
    """
    stt = RecordingStt(reactive_stt())
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        watch_submits(rig)
        rig.audio.inject_xrun(120.0)
        await run_one_spoken_turn(rig)

        record = rig.records()[0]
        assert record.extras.get("stt_frames_fed") == len(stt.fed), (
            f"criterion 8: stt_frames_fed is what the pump handed over ({len(stt.fed)}); extras "
            f"were {record.extras}"
        )
        assert record.extras.get("stt_feed_errors") == 0, record.extras
        assert record.extras.get("capture_xruns") == 1, (
            "criterion 8: capture_xruns comes from capture_stats(); one xrun was injected and "
            f"extras say {record.extras}"
        )
        assert record.extras.get("capture_gated_blocks") == 0, record.extras
        assert "reflex.onset" in record.spans, (
            f"criterion 8: the reflex.onset span is on the line; spans were {record.spans}"
        )
        assert record.unknown_spans() == [], (
            f"criterion 8: no new span name is added; {record.unknown_spans()}"
        )


# --- criterion 2: the pump is still the only caller of machine.step ----------------------------


async def test_the_pump_is_still_the_only_caller_of_machine_step(tmp_path):
    """The capture pump is a *producer*: it steps nothing and submits nothing per frame.

    Driven with an STT client that never emits an event, so every event the machine sees over two
    seconds of capture — 25 frames — is a ``Tick``. The static half extends
    ``test_session_loop.py``'s single-call-site assertion to T-0032's owned files.
    """
    stt = RecordingStt(FakeStt(reactive=True))
    async with running(tmp_path, stt=stt, frames=[frame(0, "a")]) as rig:
        watch_submits(rig)
        await rig.advance(2_000, step=100)

        assert len(stt.fed) >= 20, (
            f"the capture pump must be running (it fed {len(stt.fed)} frames in 2 s)"
        )
        not_ticks = [e for e in rig.machine.events() if not isinstance(e, E.Tick)]
        assert not_ticks == [], (
            "criterion 2: the capture pump submits no event per frame; the machine saw "
            f"{[type(e).__name__ for e in not_ticks][:8]}"
        )

    sites: list[str] = []
    for name in OWNED_FILES:
        path = REPO / name
        assert path.is_file(), f"T-0032 owns {name} (docs/tasks/T-0032.md, Owned files)"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "step":
                sites.append(f"{name}:{node.lineno}")
    assert len(sites) == 1, (
        f"criterion 2: exactly one place may call machine.step; found {sites or 'none'}"
    )


# --- criterion 2 / ruling A12: close_capture(), once, after the pump is cancelled ---------------


async def test_close_capture_is_called_once_after_the_capture_pump_is_cancelled(tmp_path):
    """``aclose()`` ends the device side of capture — once, and not before the pump has stopped.

    Criterion 2: ``aclose()`` ends the pump "(and ``close_capture()``, when the audio object has
    it)". The order matters both ways round: a ``close_capture()`` called while the pump is still
    iterating ``stt_blocks()`` races the generator that T-0021 criterion 7 asks to pad its tail
    exactly once, and a second call would pad it twice. ``FakeAudioIO`` has no such seam, so this
    double is the only thing in the fast suite that takes the branch at all (ruling A12).

    The evidence is the generator's own frame counter, not the session's: it rises only while
    something iterates ``stt_blocks()``, so "unchanged over another second of virtual time after
    ``close_capture()`` returned" is an independent statement that the pump was already cancelled.
    """
    clock = ScriptedClock()
    audio = ClosingAudio(clock)
    stt = RecordingStt(reactive_stt())
    rig = make_rig(
        tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt, clock=clock, audio=audio
    )
    await rig.session.start()
    try:
        watch_submits(rig)
        await run_one_spoken_turn(rig)
        audio.pump_task = capture_task()
        assert audio.frames_yielded >= 20, (
            "the capture pump must have been iterating this object (otherwise the row below is "
            f"vacuous); it yielded {audio.frames_yielded} frames"
        )
        assert audio.close_calls == [], (
            "criterion 2: close_capture() belongs to aclose(), not to the end of a turn; it was "
            f"already called after {audio.close_calls} frames"
        )
    finally:
        audio.aclose_started = True
        await rig.session.aclose()

    assert len(audio.close_calls) == 1, (
        "criterion 2 / ruling A12: close_capture() is called exactly once by aclose(); it was "
        f"called {len(audio.close_calls)} times (at frame counts {audio.close_calls})"
    )
    assert audio.closed_during_aclose == [True], (
        f"…and only from aclose(); it was called before aclose() started ({audio.close_calls})"
    )
    assert audio.pump_done_at_close in ([True], [None]), (
        "criterion 2: the capture task is cancelled *before* the device side is closed; at the "
        f"call the task was still running ({audio.pump_done_at_close})"
    )

    at_close = audio.close_calls[0]
    for _ in range(10):
        clock.advance(100)
        await asyncio.sleep(0)
    assert audio.frames_yielded == at_close, (
        "criterion 2: nothing iterates stt_blocks() after close_capture() returned; the generator "
        f"yielded {audio.frames_yielded - at_close} more frames over 1 s of virtual time"
    )
    assert len(audio.close_calls) == 1, (
        f"…and the seam is not closed again on the way out; {audio.close_calls}"
    )
    assert rig.session.failures == [], (
        f"a clean close records no failure; the session kept {rig.session.failures}"
    )


# --- ruling A16 F3 / A10: a rising dropped_events is surfaced, once ----------------------------


class DroppingStt(RecordingStt):
    """An STT double whose ``stats()["dropped_events"]`` rises as the session feeds it.

    Shaped after ``MoshiSttClient``: its inbound queue is bounded, so a burst of ``Word`` or a
    ``Marker`` echo can be thrown away while ``feed()`` keeps returning normally — the counter is
    the only trace. ``rises`` maps "after this many frames have been fed" to the new total, so the
    count goes up **twice** during one session: the row that matters is that the second rise adds
    no second event.
    """

    def __init__(self, inner: Any, rises: dict[int, int]) -> None:
        super().__init__(inner)
        self.rises = dict(rises)
        self.dropped = 0
        self.stats_reads = 0

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        await super().feed(pcm_24k_f32, t_ms)
        self.dropped = self.rises.get(len(self.fed), self.dropped)

    def stats(self) -> Any:
        self.stats_reads += 1
        return {**dict(self.inner.stats()), "dropped_events": self.dropped}


async def test_a_rising_dropped_events_count_is_surfaced_once_as_an_stt_error(tmp_path):
    """The session says once that the client is dropping inbound events (ruling A10 / A16 F3).

    T-0007a ruling A9: a dropped inbound event is invisible — a thrown-away ``Marker`` echo
    leaves ENDPOINTING waiting for a flush that no longer exists, and a session that never looked
    at ``stats()`` would hang with nothing on the wire to say why. The honest move is the existing
    ``stt.error`` row, and **exactly once**: a per-frame complaint is a log nobody reads, and a
    per-frame event is not something a producer task may put on the one pump.

    **The drops are driven while the machine is IDLE, deliberately** (ruling A16's note): an
    ``E.SttError`` ends a turn that is LISTENING or ENDPOINTING (``_on_stt_error``, criterion
    20(c)), so raising the count mid-turn would confound "was it reported once" with "did it kill
    the turn". Here nobody has spoken — the fake is in script mode with no words — so the machine
    stays IDLE throughout and the only thing under test is the reporting. That the drop *ends* a
    live turn is the machine's rule, asserted in ``test_session_errors.py``, not re-litigated
    here.

    The count rises after 4 fed frames (to 2) and again after 12 (to 7); ~2 s of virtual time is
    fed at 80 ms per STT frame, so both rises happen well inside the window (asserted, so the row
    cannot go vacuous if the frame cadence changes).

    Regime: fakes only, ``ScriptedClock``, script-mode ``FakeStt`` (silent), n = 1 deterministic.
    """
    stt = DroppingStt(FakeStt(), rises={4: 2, 12: 7})
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        watch_submits(rig)
        await rig.advance(2_000)

        assert len(stt.fed) > 12, (
            f"the row needs both rises to happen; only {len(stt.fed)} frames were fed"
        )
        assert stt.dropped == 7, f"…and the second rise to have landed; dropped={stt.dropped}"

        errors = rig.machine.events(E.SttError)
        assert len(errors) == 1, (
            "ruling A10: a rising dropped_events is surfaced **once** through the existing "
            f"stt.error row — a session that never reads stats() hangs in ENDPOINTING when the "
            f"dropped event was the Marker echo, and one that reports per frame floods the pump. "
            f"Got {len(errors)} stt.error events: {errors[:3]}"
        )
        message = errors[0].message.lower()
        assert "drop" in message, (
            f"the error must name the dropped events, not a generic failure; it said "
            f"{errors[0].message!r}"
        )
        assert rig.state.state is TurnState.IDLE, (
            "nobody spoke in this row, so the machine must still be IDLE; it is "
            f"{rig.state.state} — the drop report must not open or end a turn by itself"
        )


# ==================================================================================================
# T-0051 — the end-to-end behaviour, over the fakes, without owning session.py.
#
# Regime for both rows below: ``FakeStt(reactive=True, steps=True, lag_ms=500, usable_heads=False)``
# — words and 80 ms steps released against *fed audio*, each ``Step`` stamped with the fed-audio
# clock at which it is delivered (which is what ``MoshiSttClient`` stamps, so the endpointer sees
# the recogniser's 500 ms emission delay exactly as it does on the box), and a ``prs`` vector that
# carries no usable pause head. The last part is not a convenience: ruling A7 says the backstop is
# a fallback and the heads decide first on any server that sends usable ones — a 600 ms pause with
# the fake's 0.9 silence head crosses ``EOT_P_THRESHOLD`` in five steps and is cut whatever the
# backstop does. The regime these criteria are about is the one T-0007a's census A12 recorded: a
# server whose heads say nothing, where the silence backstop is the only decider.
# ==================================================================================================

#: The sentence of 2026-09-16, with its mid-sentence pause, in *audio* time. Three 200 ms words,
#: 600 ms of breath, two 160 ms words — the shape of ``tests/unit/test_endpointer.py``'s headline
#: row, whose docstring shows the 80 ms of margin it lives on (ruling A15).
PAUSED_HEAD = "what do you"
PAUSED_TAIL = "see now"
PAUSED_SENTENCE = f"{PAUSED_HEAD} {PAUSED_TAIL}"
PAUSE_MS = 600
TAIL_START_MS = 2200
TAIL_END_MS = 2520


def pausing_speaker(*, extra: list[Any] | None = None) -> FakeStt:
    """The human of 2026-09-16: "what do you … see now", with 600 ms of breath in the middle."""
    stt = FakeStt(reactive=True, steps=True, lag_ms=500, usable_heads=False)
    head = stt.say(PAUSED_HEAD, start_ms=SPEECH_START_MS, ms_per_word=200)
    tail = stt.say(PAUSED_TAIL, start_ms=TAIL_START_MS, ms_per_word=160)
    assert tail[0].start_ms - head[-1].end_ms == PAUSE_MS, "the pause is 600 ms by hand"
    assert tail[-1].end_ms == TAIL_END_MS, "recompute the tail"
    if extra:
        stt.script.extend(extra)
    return stt


def thinking_entries(rig: Any) -> list[Any]:
    """The events that took the machine **into** THINKING — one per turn started."""
    return [ev for before, ev, after in rig.machine.transitions() if after is TurnState.THINKING]


async def test_a_pausing_speaker_reaches_thinking_once_with_the_whole_sentence(tmp_path):
    """**T-0051 criterion 17** — the row that matches what the human experienced.

    On 2026-09-16 the human asked "what do you see?" and the assistant answered "what do you".
    The session log shows why twice over: the backstop measured 800 ms of silence from the word's
    *audio* end while the recogniser was ~510 ms behind, so the turn was cut on ~290 ms of
    examined audio in the middle of a breath (group A); and the words that arrived afterwards
    re-opened the run, so the same end of turn was decided twice, ``mark()`` went out five times
    and three flushes came back for two real turns. The prompts read ``"What do you"`` and
    ``"see? Illustrate how to solve"``.

    Here the same speaker, over the fakes, with nothing submitted by the test: one THINKING, one
    ``mark()``, and the transcript is the whole sentence. The **adversarial** variant adds
    tonight's other shape — a word whose audio starts exactly at the decided ``t_eot`` and which
    is delivered *after* the decision (log seq 3093 → 3113, ``end_of_turn 242720`` then
    ``onset 242720``) — and the same three assertions must still hold: it may not start a second
    turn.

    This row owns no file: criterion 17 proves the end-to-end behaviour without touching
    ``session.py`` (held by T-0043 group B), so everything it asserts is a consequence of the
    endpointer and the STT client alone.
    """
    late_tail = FakeStt.words("illustrate", start_ms=TAIL_END_MS, ms_per_word=1000)
    assert late_tail[0].start_ms == TAIL_END_MS, (
        "the adversarial variant's premise: the late word's audio begins exactly at the t_eot the "
        "endpointer is about to decide, which is the shape of log seq 3113"
    )
    # A word whose audio lies at or **before** the decided ``t_eot`` belongs to this turn's
    # transcript — that is what ``FreezeAfter(t_eot_ms)`` means, and dropping it would lose speech
    # the human really produced. What it may not do is start a *second* turn, which is what it did
    # on 2026-09-16: ``onset 242720`` 479 ms after ``end_of_turn 242720``, a second ``mark()`` and
    # a third flush for two real turns. So the variant expects the word in the transcript and the
    # turn count unchanged.
    cases = (
        ("the sentence alone", None, PAUSED_SENTENCE),
        ("with tonight's late word", late_tail, f"{PAUSED_SENTENCE} illustrate"),
    )
    for label, extra, expected_text in cases:
        (tmp_path / label.replace(" ", "-")).mkdir()
        stt = RecordingStt(pausing_speaker(extra=extra))
        async with running(
            tmp_path / label.replace(" ", "-"),
            script=ANSWER,
            frames=[frame(0, "a")],
            stt=stt,
        ) as rig:
            submitted = watch_submits(rig)
            await run_one_spoken_turn(rig, limit_ms=20_000)

            assert submitted == [], f"{label}: the test submits nothing; it submitted {submitted}"
            entries = thinking_entries(rig)
            assert len(entries) == 1, (
                f"criterion 17 ({label}): a speaker who pauses {PAUSE_MS} ms mid-sentence is "
                f"having one turn, not two. The machine entered THINKING {len(entries)} times, on "
                f"{[type(e).__name__ for e in entries]}; the state path was "
                f"{[s.name for s in states_seen(rig)]}. On 2026-09-16 that churn cost 1191 ms and "
                "split the question in half"
            )
            assert stt.marks == [TAIL_END_MS], (
                f"criterion 17 ({label}): one mark(), at the last word's audio end "
                f"({TAIL_END_MS}). Five went out on 2026-09-16, one per decision, and each bought "
                f"a flush. Got {stt.marks}"
            )
            history = rig.state.history
            assert len(history) == 1 and history[0].user_text == expected_text, (
                f"criterion 17 ({label}): one turn, whose transcript is {expected_text!r} — the "
                "words after the pause belong to the same turn, and so does a word whose audio "
                "ends inside the frozen window. On 2026-09-16 this was two turns reading "
                f'"What do you" and "see? Illustrate how to solve". Got '
                f"{[h.user_text for h in history]}"
            )
            prompts = [p.messages[-1].text() for _, p in rig.llm.gen_prompts]
            assert len(prompts) == 1 and PAUSED_SENTENCE in prompts[0], (
                f"criterion 17 ({label}): exactly one generate, carrying the whole sentence in its "
                f"query-dependent suffix (docs/01 D9). Got {prompts}"
            )


async def test_a_word_released_during_the_reply_is_published_but_never_opens_a_turn(tmp_path):
    """**T-0051 criterion 12, both directions, at the level where the damage would happen.**

    Criterion 9 bounds the wait for an ``EndWord`` that never comes, which means a word recognised
    before the assistant started speaking is now *released in the middle of its reply* instead of
    at the head of the user's next sentence. That is the defect group B would otherwise create:

    **absent** — the released word is read as an onset, ``E.UserSpeechOnset`` reaches the machine
    in SPEAKING, and docs/02 §4's SPEAKING row takes it to **PAUSED**: a barge-in through the STT
    tap, which T-0032's design ruling and T-0034 forbid and which nothing downstream is built to
    handle. Asserted as: no second onset, and the machine never enters PAUSED.

    **too wide** — the word stops being *published*, which loses it from the transcript stream and
    from the session log. That is the failure the human actually reported: ``see?`` was recognised
    and then never seen. Criterion 12 suppresses the **decision**, never the word. Asserted as:
    the word reaches the machine as ``E.SttWord`` with its own times, and it is the estimated one.

    The two assertions are opposites, so neither can be satisfied by weakening the other: an
    implementation that drops estimated words entirely passes the first and fails the second, and
    one that ignores the flag passes the second and fails the first.

    The row asserts its own premise — that the word really was delivered while the assistant was
    speaking — because a word released in LISTENING would prove nothing at all.
    """
    held = FakeStt.open_word("see?", start_ms=4000, end_ms=5000)
    stt = RecordingStt(pausing_speaker(extra=[held]))
    async with running(tmp_path, script=ANSWER, frames=[frame(0, "a")], stt=stt) as rig:
        submitted = watch_submits(rig)
        await run_one_spoken_turn(rig, limit_ms=20_000)
        assert submitted == [], f"the test submits nothing; it submitted {submitted}"

        delivered = [
            (before, ev)
            for before, ev, _ in rig.machine.transitions()
            if isinstance(ev, E.SttWord) and ev.text == "see?"
        ]
        assert delivered, (
            "criterion 12, the **negative**: the held word was never published at all. It must "
            "stay in the transcript stream and in the log — criterion 12 suppresses the decision, "
            "never the word, and a word the human said and never saw is the defect they reported. "
            f"The words that reached the machine were "
            f"{[e.text for e in rig.machine.events(E.SttWord)]}"
        )
        before, event = delivered[0]
        assert (event.start_ms, event.end_ms) == (held.start_ms, held.end_ms), (
            f"…published with its own times, unedited; got {event}"
        )

        onsets = rig.machine.events(E.UserSpeechOnset)
        assert len(onsets) == 1 and onsets[0].onset_ms == SPEECH_START_MS, (
            "criterion 12: a word whose end nobody measured may not open a run. A second onset "
            f"here is the accidental barge-in group B creates without it; got {onsets}"
        )
        assert TurnState.PAUSED not in states_seen(rig), (
            "…and that onset would have taken the machine SPEAKING → PAUSED through the STT tap "
            f"(docs/02 §4), which T-0032's design ruling and T-0034 forbid. Path: "
            f"{[s.name for s in states_seen(rig)]}"
        )
        assert len(thinking_entries(rig)) == 1 and stt.marks == [TAIL_END_MS], (
            "…and the reply's own turn is untouched: one THINKING, one mark(). Got "
            f"{len(thinking_entries(rig))} and {stt.marks}"
        )
        # Last, because it is the row's premise and not its subject: the word really was delivered
        # while the assistant's turn was open. PAUSED counts as open — if the machine is *in*
        # PAUSED the barge-in has already happened and the two assertions above have already
        # fired, so this line never has to explain that case.
        assert before in (TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED), (
            f"the row's premise: the held word must be delivered while the assistant's turn is "
            f"open, which is when reading it as an onset would be a barge-in. It arrived in "
            f"{before.name} instead — retime the word rather than relaxing the assertion"
        )
