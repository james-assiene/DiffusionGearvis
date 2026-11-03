"""The seam between the session and the playback module (T-0007 criterion 14, T-0006's "Seam to
session.py").

Four things the session owes the audio object, asserted against both implementations of it —
``tests/fakes/audio.py::FakeAudioIO`` on the virtual clock, and the real
``local_astra/conductor/audio_io.py::PipeWireAudioIO`` on ``FakeAudioDevice`` (marked ``audio``):

1. ``play(utterance, source=…)`` carries the **committed substring**, while ``speak()`` gets the
   normalised rendering — ``TtsSpeak.source`` and ``TtsSpeak.text`` are different strings and
   swapping them puts words nobody committed into history;
2. ``mark_drain()`` is called when the last utterance of the turn is queued, and only then — a
   queue that merely empties is an underrun (docs/02 §4 spec delta 1);
3. the terminal events from ``playback_events()`` reach the machine **unchanged**;
4. ``played_text`` is never recomputed from TTS word timings or from what was sent. Re-deriving
   it is how history acquires words nobody heard (T-0006 criterion 6 owns that number).

The session API is fixed in ``tests/unit/test_session_loop.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit.test_session_loop import ScriptedClock, answer, frame, running

pytestmark = [pytest.mark.contract]

#: Two sentences whose rendering differs from the committed bytes: the path is spoken as words
#: (T-0004's ``normalise_for_tts``), so ``source`` and ``text`` cannot be confused by accident.
SCRIPT = (
    "First sentence. ",
    "Open /etc/hosts now. ",
)


class Recorder:
    """Delegating wrapper over any ``AudioIO``: records the calls, forwards everything."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []
        self.plays: list[tuple[Any, str | None]] = []
        self.emitted: list[E.PlaybackStopped | E.PlaybackDrained] = []

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        self.calls.append("play")
        self.plays.append((utterance, source))
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        self.calls.append("pause")
        await self.inner.pause()

    async def resume(self) -> None:
        self.calls.append("resume")
        await self.inner.resume()

    async def flush(self) -> int:
        self.calls.append("flush")
        return await self.inner.flush()

    def mark_drain(self) -> None:
        self.calls.append("mark_drain")
        self.inner.mark_drain()

    def played_ms(self) -> int:
        value = self.inner.played_ms()
        return value

    def is_rendering(self, t_ms: float) -> bool:
        return bool(self.inner.is_rendering(t_ms))

    def capture(self):
        return self.inner.capture()

    async def playback_events(self):
        async for event in self.inner.playback_events():
            self.emitted.append(event)
            yield event

    def source_of(self, index: int) -> str | None:
        return self.plays[index][1]

    def answer_plays(self) -> list[tuple[Any, str]]:
        """Clips are played with ``source=""`` and are not answer audio."""
        return [(u, s) for u, s in self.plays if s]


def build_audio(impl: str, clock: ScriptedClock):
    """``fake`` is the in-memory playback; ``module`` is T-0006's class on the fake device."""
    if impl == "fake":
        from tests.fakes.audio import FakeAudioIO

        return Recorder(FakeAudioIO(clock)), None
    from tests.fakes.audio_device import FakeAudioDevice, make_audio_io, settle

    device = FakeAudioDevice(clock, buffer_ms=200.0)
    recorder = Recorder(make_audio_io(device))

    async def pump() -> None:
        await settle(device)

    return recorder, pump


#: Both legs are pure: ``module`` is T-0006's ``PipeWireAudioIO`` on ``FakeAudioDevice`` and the
#: virtual clock, so no device is touched. It carried an ``audio`` marker until T-0035 (finding
#: F7): the marker excluded a pure test from ``-m "not audio"`` and offered it to hardware runs
#: that have no use for it.
IMPLS = ("fake", "module")


@pytest.mark.parametrize("impl", IMPLS)
async def test_the_session_hands_the_committed_source_to_play_and_marks_the_drain(impl, tmp_path):
    """Criterion 14, points 1-3, on both implementations.

    ``speak()`` gets ``"Open hosts in the etc folder now."`` and ``play(..., source=…)`` gets
    ``"Open /etc/hosts now. "`` — the bytes the model committed. A session that passes the
    rendering to ``play`` writes a history line the model never wrote.
    """
    clock = ScriptedClock()
    audio, pump = build_audio(impl, clock)
    async with running(
        tmp_path,
        script=[answer(text) for text in SCRIPT],
        frames=[frame(0, "a")],
        clock=clock,
        audio=audio,
        settle_audio=pump,
        ms_per_word=100,
    ) as rig:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        sources = [source for _, source in audio.answer_plays()]
        assert sources == list(SCRIPT), (
            f"criterion 14: play(..., source=…) carries the committed substrings; got {sources}"
        )
        assert "Open hosts in the etc folder now." in rig.tts.texts(), (
            f"…while speak() gets the normalised rendering; TTS was asked for {rig.tts.texts()}"
        )
        assert "mark_drain" in audio.calls, (
            "docs/02 §4 spec delta 1: playback.drained needs an explicit mark_drain(); without it "
            "the turn cannot leave SPEAKING"
        )
        last_play = len(audio.calls) - 1 - audio.calls[::-1].index("play")
        assert audio.calls.index("mark_drain") > last_play, (
            f"mark_drain() comes after the last utterance is queued; calls were {audio.calls}"
        )

        drained = [e for e in audio.emitted if isinstance(e, E.PlaybackDrained)]
        assert drained, "the playback module reports the drain through playback_events()"
        seen = [e for e in rig.machine.events(E.PlaybackDrained)]
        assert seen and (seen[-1].played_ms, seen[-1].played_text) == (
            drained[-1].played_ms,
            drained[-1].played_text,
        ), (
            "criterion 14: playback events are pumped into the machine unchanged; the module "
            f"reported {drained[-1]} and the machine saw {seen[-1]}"
        )
        assert rig.state.history[-1].assistant_spoken == drained[-1].played_text


@pytest.mark.parametrize("impl", IMPLS)
async def test_played_text_after_a_barge_in_is_the_audio_modules_answer(impl, tmp_path):
    """Criterion 14, point 4: history holds what the device reported, not a re-derivation.

    The cut lands inside the second utterance, whose committed source has three words and whose
    rendering has seven: the playback module answers with the first utterance only (docs/02 §4
    spec delta 2 — a partial utterance whose word counts disagree contributes no text; under-
    reporting is allowed, over-reporting is not). A session that recomputed ``played_text`` from
    the TTS word timings would put "hosts in the etc folder" into history — words nobody heard.
    """
    clock = ScriptedClock()
    audio, pump = build_audio(impl, clock)
    async with running(
        tmp_path,
        script=[answer(text) for text in SCRIPT],
        frames=[frame(0, "a")],
        clock=clock,
        audio=audio,
        settle_audio=pump,
        ms_per_word=100,
    ) as rig:
        await rig.ask("what is on this screen")
        await rig.until(
            lambda: len(audio.answer_plays()) == 2, "both utterances must be queued", limit_ms=5_000
        )
        total = sum(float(getattr(u, "duration_ms", 0)) for u, _ in audio.plays)
        first = float(getattr(audio.answer_plays()[0][0], "duration_ms", 0))
        await rig.until(
            lambda: first < audio.played_ms() < total,
            "the cut must land inside the second utterance",
            step=20,
            limit_ms=5_000,
        )

        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: rig.state.state is TurnState.LISTENING, "the takeover never completed"
        )

        stopped = [e for e in audio.emitted if isinstance(e, E.PlaybackStopped)]
        assert stopped, "the flush reports playback.stopped with what was rendered"
        reported = stopped[-1].played_text
        assert reported == SCRIPT[0], (
            f"the module must report the fully played utterance and no more; got {reported!r}"
        )
        seen = rig.machine.events(E.PlaybackStopped)[-1]
        assert (seen.played_ms, seen.played_text) == (stopped[-1].played_ms, reported), (
            "criterion 14: the session never rewrites played_ms/played_text on the way in"
        )
        stored = rig.state.history[-1].assistant_spoken
        assert stored.startswith(reported) and "[interrupted]" in stored, stored
        assert "hosts in the etc folder" not in stored, (
            "criterion 14, negative: played_text is never recomputed from TTS word timings"
        )
        truncates = [
            c
            for c in rig.machine.commands()
            if getattr(c, "type", "") == "truncate_history_to_played"
        ]
        assert truncates and truncates[-1].played_text == reported, (
            f"truncate_history_to_played carries the module's own answer; got {truncates}"
        )


def test_the_fake_and_the_module_expose_the_same_widened_audio_io():
    """The ``AudioIO`` protocol widening T-0007 owes (Ownership conflicts, narrow mandate).

    ``tests/contract/test_fakes_conform.py`` checks ``isinstance(FakeAudioIO, AudioIO)``; this
    checks that the protocol actually grew the playback seams T-0006 shipped, so the conformance
    test is not asserting the old, narrower interface.
    """
    from local_astra.conductor.audio_io import PipeWireAudioIO
    from local_astra.contracts.protocols import AudioIO
    from tests.fakes.audio import FakeAudioIO

    required = ("play", "pause", "resume", "flush", "played_ms", "mark_drain", "playback_events")
    missing = [name for name in required if not hasattr(AudioIO, name)]
    assert not missing, (
        f"criterion 14 / Ownership conflicts: AudioIO must grow {missing} (T-0006 shipped them on "
        "the concrete class and declined the protocol; T-0007 widens it, and nothing else)"
    )
    for impl in (FakeAudioIO, PipeWireAudioIO):
        absent = [name for name in required if not hasattr(impl, name)]
        assert not absent, f"{impl.__name__} is missing {absent}"
