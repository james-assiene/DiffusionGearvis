"""A late consumer of ``playback_events()``: both playback implementations agree (T-0092a).

T-0092a's rows hold a run's terminal event across a turn boundary with
``FakeAudioIO.hold_events()``. That knob is only honest if the real module behaves the same way when
its consumer reads late, so this row asserts, on ``audio_io.PipeWireAudioIO`` (over
``FakeAudioDevice``, pure) and on the fake (held):

1. a terminal emitted while nobody is reading is kept and delivered later, unchanged;
2. ``flush()`` on a run whose drain was already emitted (but not yet read) emits nothing more and
   starts nothing — the run is over (T-0006 criteria 7-8: one terminal per run);
3. a ``play()`` after that starts a new run, whose terminal is delivered after the old one —
   emission order is delivery order.

The real module needs no knob for this: it appends terminals to a list ``playback_events()`` drains
when its consumer next iterates, so "late" is simply "the consumer has not started yet".
Parametrised locally like ``test_audio_io_conformance.py`` (no GPU is involved on either leg).
"""

from __future__ import annotations

import asyncio

import pytest

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from tests.conftest import VirtualClock
from tests.fakes.audio import FakeAudioIO
from tests.fakes.audio_device import (
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    call,
    make_audio_io,
    play,
    play_out,
    tone,
    words_of,
)

pytestmark = [pytest.mark.contract]

SOURCE_1 = "one two three"
SOURCE_2 = "four five six"
UTTERANCE_MS = 1200.0
WORDS = words_of([("w1", 0, 300), ("w2", 400, 700), ("w3", 800, 1100)])


def _utterance(uid: str, text: str) -> PcmUtterance:
    return PcmUtterance(uid, samples=tone(UTTERANCE_MS), words=WORDS, text=text)


class _Late:
    """A consumer that has not read yet, then reads everything (one per implementation)."""

    def __init__(self, impl: str, io: object) -> None:
        self.impl, self.io = impl, io
        if impl == "fake":
            # The fake advances its run from its poll loop, so the consumer runs and holds.
            io.hold_events()  # type: ignore[attr-defined]
            self.sink = EventSink(io)
        else:
            self.sink = EventSink(io, autostart=False)

    async def release(self) -> list[object]:
        if self.impl == "fake":
            self.io.release_events()  # type: ignore[attr-defined]
        else:
            self.sink.start()
        await self.sink.sync(rounds=16)
        return list(self.sink.events)


def _build(impl: str):
    clock = VirtualClock()
    if impl == "module":
        device = FakeAudioDevice(clock, buffer_ms=200.0)
        io = make_audio_io(device)

        async def advance(ms: float) -> None:
            await play_out(device, ms)

        return io, advance
    io = FakeAudioIO(clock)

    async def advance_clock(ms: float) -> None:
        clock.advance(int(ms))
        for _ in range(8):  # the fake's poll loop runs on the event loop
            await asyncio.sleep(0.0005)

    return io, advance_clock


@pytest.mark.parametrize("impl", ["module", "fake"])
async def test_a_terminal_emitted_before_anyone_reads_is_delivered_later_once_in_order(impl):
    io, advance = _build(impl)
    late = _Late(impl, io)

    await play(io, _utterance("u1", SOURCE_1), source=SOURCE_1)
    await call(io, "mark_drain", "T-0006 criterion 7")
    await advance(UTTERANCE_MS + 300.0)

    # run 1 is over (drained, not yet read): a flush now must add nothing
    await call(io, "flush", "AudioIO protocol")

    await play(io, _utterance("u2", SOURCE_2), source=SOURCE_2)
    await advance(500.0)
    await call(io, "flush", "AudioIO protocol")

    events = await late.release()
    await late.sink.close()
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["PlaybackDrained", "PlaybackStopped"], (
        f"{impl}: a late consumer gets run 1's drain, then run 2's stop — exactly one terminal "
        f"per run, in emission order, and nothing from the flush of a finished run; got {kinds}"
    )
    drained, stopped = events
    assert isinstance(drained, PlaybackDrained) and drained.played_text == SOURCE_1, (
        f"{impl}: the held drain is run 1's, unchanged; got {drained}"
    )
    assert isinstance(stopped, PlaybackStopped), stopped
    assert SOURCE_1 not in stopped.played_text, (
        f"{impl}: run 2 is a new run — its stop reports none of run 1's text; got {stopped}"
    )
    assert 0 < stopped.played_ms < UTTERANCE_MS, (
        f"{impl}: run 2 was cut ~500 ms in; its stop reports its own clock, got {stopped.played_ms}"
    )
