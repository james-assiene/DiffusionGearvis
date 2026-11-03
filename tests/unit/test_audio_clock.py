"""The played clock: T-0006 criteria 1-5, 7, 8, 10 on the fake device (docs/tasks/T-0006.md).

Everything here is pure: ``FakeAudioDevice`` + ``VirtualClock``, no PortAudio, no PipeWire, no
rig (regime: *Unit* in the task file). The device's rendered position is the oracle; a module
that answers with the bytes it handed to the buffer disagrees with it on every test in this file.

Seams these tests fix, because the task does not spell out the signatures (reported as such):
``PipeWireAudioIO(device)``, ``played_ms()``, ``mark_drain()``, ``playback_events()``,
``playback_reference(window_ms)``, ``is_rendering(t_ms)``, ``underruns``, ``dropped_events``,
``quantum_ms``/``latency_ms``/``block_ms``. Sync or async is accepted for the call-style seams
(``tests/fakes/audio_device.maybe_await``); the names are not negotiable without a task change.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    DEVICE_RATE,
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    attr_or_fail,
    call,
    make_audio_io,
    max_abs_step,
    play,
    play_out,
    played_ms,
    settle,
    tone,
)

#: criterion 2 bounds: never over-report by more than 20 ms, under-report by at most
#: ``latency_ms + block_ms`` (30 + 10 at the criterion-1 defaults).
OVER_MS = 20.0
UNDER_MS = 40.0
#: Rulings A12/A16: every drain/underrun case runs against four device shapes, given here as
#: ``(latency_ms, dac_lag_ms, quantum_ms)``. The lag is what the device's rendered position
#: actually trails ``written`` by; ``latency_ms`` is only what the device *reports*, and on the
#: hardware the two differ — ``PortAudioDevice`` measures the DAC lag per callback (15.3 ms
#: against a negotiated 10 ms) and another client can force the graph quantum to ≈ 21 ms. A drain
#: rule that spends an allowance computed from the reported latency is green on the first two
#: shapes and silent on the last one (ruling A15: the rule must be lag-independent).
LAG_MS = 20.0
LAG_CASES = [
    (20.0, 0.0, 10.0),  # zero-lag: the shape the first tests were written on
    (20.0, 20.0, 10.0),  # dac-lag: the lag equals the reported latency
    (10.0, 20.0, 10.0),  # lag-above-latency: one block more than the device reports (A16)
    (10.0, 31.3, 21.333),  # quantum-lag: the graph at 21 ms, lag one quantum over the latency
]
LAG_IDS = ["zero-lag", "dac-lag", "lag-above-latency", "quantum-lag"]
#: criterion 3/5: pause and flush act on the device within 50 ms (p95 on the rig; here the fake
#: device has no jitter at all, so it is a hard bound).
STOP_MS = 50.0


def rig(clock: VirtualClock | None = None, **kwargs: float) -> tuple[object, FakeAudioDevice]:
    device = FakeAudioDevice(clock if clock is not None else VirtualClock(), **kwargs)
    return make_audio_io(device), device


def assert_played(actual: float, expected: float, note: str) -> None:
    assert expected - UNDER_MS <= actual <= expected + OVER_MS, (
        f"{note}: played_ms={actual:.1f} outside [{expected - UNDER_MS:.1f}, "
        f"{expected + OVER_MS:.1f}] (criterion 2: over-report ≤ {OVER_MS} ms, "
        f"under-report ≤ {UNDER_MS} ms)"
    )


async def test_played_ms_follows_the_device_not_the_write_cursor() -> None:
    """Criterion 2, the headline: 1000 ms enqueued, 300 ms rendered, played_ms ≈ 300.

    The device tolerates 200 ms of lead, so a module that answers with what it wrote (or with
    what it accepted) reports something between 330 and 1000 here. Nothing in the test knows how
    ``played_ms`` is computed: the number it is compared against is virtual time elapsed since
    the run started, which only the clock and the device decide.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0, latency_ms=20.0)
    utterance = PcmUtterance(samples=tone(1000.0), text="one thousand milliseconds")

    await play(io, utterance, source="one thousand milliseconds")
    await play_out(device, 300.0)

    assert device.starts >= 1, "criterion 1: the module owns the device lifecycle (start())"
    assert utterance.duration_ms == 1000.0
    played = await played_ms(io)
    assert played < 500.0, (
        f"played_ms={played:.1f} looks like the enqueued total ({utterance.duration_ms:.0f} ms) "
        f"or the write cursor ({device.written_ms:.1f} ms), not the device's rendered position "
        f"({device.rendered_ms():.1f} ms) — criterion 2"
    )
    assert_played(played, 300.0, "1000 ms enqueued, 300 ms of virtual time elapsed")
    # Against a *fake* device the rendered position is exact and this is the first run, so
    # ``base_ms`` is 0 and criterion 2's formula has no slack left: the number the module
    # reports is the number the device reports. The 20/40 ms band above is the allowance for
    # the physical monitor, not a licence to answer with a cursor that is one queue-depth
    # ahead — half a block of tolerance keeps an implementation that samples the device
    # position per block, and rejects one that counts what it wrote (ruling A8).
    assert abs(played - device.rendered_ms()) <= CAPTURE_BLOCK_MS / 2, (
        f"played_ms={played:.1f} but the device reports {device.rendered_ms():.1f} ms rendered "
        f"and {device.written_ms:.1f} ms written: the module is not reading the device"
    )
    # Ruling A8: the bound above only means something if the module keeps the device queue
    # shallow — with 1000 ms accepted and a device that tolerates 200 ms of lead, a module that
    # dumps the utterance into the device makes "the write cursor" and "the rendered position"
    # agree to within the device buffer and the headline assertion goes quiet.
    assert device.overfills == 0, (
        f"{device.overfills} writes went past the device's {device.buffer_ms:.0f} ms of "
        f"tolerance: pause() cannot act on audio that is already inside the device (criterion 3)"
    )
    assert device.max_lead_ms <= 3 * CAPTURE_BLOCK_MS, (
        f"the device queue ran {device.max_lead_ms:.0f} ms ahead of the rendered position; the "
        f"implementation notes cap it at 2-3 blocks ({3 * CAPTURE_BLOCK_MS} ms) so that pause "
        f"acts on the device and played_ms cannot be approximated by the write cursor"
    )


async def test_played_ms_is_capped_by_the_device_even_when_it_stalls() -> None:
    """Criterion 2, adversarial: an xrun makes wall time run ahead of the rendered position."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    await play(io, PcmUtterance(samples=tone(1000.0)), source="a")

    await play_out(device, 200.0)
    device.stall(150.0)  # the device rendered 150 ms of something that was not our stream
    await play_out(device, 300.0)

    played = await played_ms(io)
    assert_played(played, device.rendered_ms(), "after a 150 ms xrun")
    assert played < 500.0 - UNDER_MS + OVER_MS, (
        f"played_ms={played:.1f} tracked wall time (500 ms), not the device's rendered "
        f"position ({device.rendered_ms():.1f} ms) — criterion 2"
    )


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    ops=st.lists(
        st.one_of(
            st.tuples(st.just("advance"), st.integers(min_value=1, max_value=100)),
            st.tuples(st.just("pause"), st.just(0)),
            st.tuples(st.just("resume"), st.just(0)),
            st.tuples(st.just("stall"), st.integers(min_value=1, max_value=100)),
        ),
        min_size=1,
        max_size=6,
    )
)
async def test_played_ms_never_exceeds_enqueued_or_goes_backwards(
    ops: list[tuple[str, int]],
) -> None:
    """Criterion 2 as a property (docs/02 §4: history never holds text that was not played).

    Bounds only — the test never recomputes ``played_ms``. ``paused_ms >= 0`` makes
    ``played_ms <= rendered_ms - rendered_at_run_start`` a consequence of the definition, so an
    implementation that leaks queued audio into the number breaks it whatever its arithmetic.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    enqueued_ms = 2000.0
    await play(io, PcmUtterance(samples=tone(enqueued_ms)), source="property")
    await settle(device)

    previous = await played_ms(io)
    for op, arg in ops:
        if op == "advance":
            await play_out(device, float(arg))
        elif op == "pause":
            await call(io, "pause", "AudioIO protocol")
        elif op == "resume":
            await call(io, "resume", "AudioIO protocol")
        else:
            device.stall(float(arg))
            await play_out(device, float(arg))

        current = await played_ms(io)
        assert current >= previous - 1e-6, f"played_ms went backwards: {previous} -> {current}"
        assert 0.0 <= current <= enqueued_ms, f"played_ms={current} outside [0, {enqueued_ms}]"
        assert current <= device.rendered_ms() + OVER_MS, (
            f"played_ms={current:.1f} exceeds what the device rendered "
            f"({device.rendered_ms():.1f} ms) by more than {OVER_MS} ms"
        )
        previous = current


async def test_pause_freezes_played_ms_and_resume_continues() -> None:
    """Criteria 3 and 4 on the fake device: the clock stops at the cut and restarts there."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    await play(io, PcmUtterance(samples=tone(2000.0)), source="pause and resume")

    await play_out(device, 400.0)
    await call(io, "pause", "AudioIO protocol")
    at_pause = await played_ms(io)
    assert_played(at_pause, 400.0, "at the pause call")

    await play_out(device, 600.0)  # 600 ms of paused wall time
    while_paused = await played_ms(io)
    assert while_paused <= at_pause + STOP_MS, (
        f"played_ms advanced {while_paused - at_pause:.1f} ms while paused — criterion 3 allows "
        f"at most {STOP_MS} ms (the audio already in the device)"
    )

    await call(io, "resume", "AudioIO protocol")
    await play_out(device, 300.0)
    after = await played_ms(io)
    assert_played(after, while_paused + 300.0, "300 ms after resume")


async def test_pause_acts_on_the_queued_audio_not_only_on_the_producer() -> None:
    """Criterion 3, adversarial: a deep device queue must not keep playing after ``pause()``.

    The device tolerates 300 ms of lead. An implementation that only stops feeding it leaves
    ≥ 100 ms of queued audio to be rendered, which this test reads off the device's own record of
    what it put out. The fade is checked on the same record: a hard cut of a 0.8 sine steps by up
    to 0.8 full-scale between two samples; the criterion allows 0.1.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=300.0)
    await play(io, PcmUtterance(samples=tone(2000.0, freq_hz=440.0, amplitude=0.8)), source="x")

    await play_out(device, 400.0)
    rendered_at_pause = device.rendered_ms()
    await call(io, "pause", "AudioIO protocol")
    await play_out(device, 700.0)

    last_sound = device.last_nonzero_rendered_ms()
    assert last_sound is not None, "the device rendered nothing at all"
    assert last_sound - rendered_at_pause <= STOP_MS, (
        f"the device kept rendering for {last_sound - rendered_at_pause:.1f} ms after pause() "
        f"(criterion 3: ≤ {STOP_MS} ms; an implementation that only stops the producer leaves "
        f"the queued {device.buffer_ms:.0f} ms to play out)"
    )
    # only across the cut: criterion 3 is about the pause, and an FIR resampler legitimately
    # rings for a millisecond at the *start* of a stream.
    rendered = device.rendered_audio()
    cut = np.concatenate([rendered[-(DEVICE_RATE // 50) :], np.zeros(1)])
    step = max_abs_step(cut)
    assert step <= 0.1, (
        f"consecutive-sample step of {step:.3f} full-scale in the last 20 ms before silence — "
        f"criterion 3 wants a 2-10 ms fade so no step exceeds 0.1 (a 440 Hz 0.8 sine steps by "
        f"0.046 per sample, a hard cut by up to 0.8)"
    )
    fade = np.abs(rendered[-(DEVICE_RATE // 100) :])
    assert float(fade[: DEVICE_RATE // 400].max()) > float(fade[-DEVICE_RATE // 400 :].max()), (
        "the last 10 ms of output does not decay: pause() hard-cut the stream instead of fading"
    )


async def test_pause_while_paused_and_resume_while_playing_are_no_ops() -> None:
    """Criterion 4: the second call changes nothing (and emits no event)."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    sink = EventSink(io)
    await play(io, PcmUtterance(samples=tone(2000.0)), source="idempotent")

    await play_out(device, 300.0)
    await call(io, "resume", "AudioIO protocol")  # already playing: no-op
    await play_out(device, 200.0)
    await call(io, "pause", "AudioIO protocol")
    await call(io, "pause", "AudioIO protocol")  # already paused: no-op
    await play_out(device, 300.0)
    await call(io, "resume", "AudioIO protocol")
    await call(io, "resume", "AudioIO protocol")
    await play_out(device, 200.0)

    # 700 ms of playing time, plus whatever was already inside the device when pause() was
    # called: criterion 3 bounds that tail at 50 ms and criterion 2 counts it as played.
    played = await played_ms(io)
    assert 700.0 - UNDER_MS <= played <= 700.0 + OVER_MS + STOP_MS, (
        f"played_ms={played:.1f} after 300 + 200 + 200 ms of playing time (a pause tail of up to "
        f"{STOP_MS:.0f} ms is allowed on top)"
    )
    await sink.sync()
    assert sink.of_type(PlaybackStopped, PlaybackDrained) == [], (
        "pause/resume are not terminal events (criterion 8: at most one terminal event per run)"
    )
    await sink.close()


async def test_flush_is_idempotent_and_emits_one_stopped() -> None:
    """Criterion 5 (and ``IDEMPOTENT_COMMANDS`` in contracts/commands.py:223)."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=300.0)
    sink = EventSink(io)
    await play(io, PcmUtterance(samples=tone(2000.0)), source="flush me")

    await play_out(device, 400.0)
    first = await call(io, "flush", "AudioIO protocol")
    assert isinstance(first, int), f"flush() must return an int of ms, got {first!r}"
    assert_played(float(first), 400.0, "flush after 400 ms")

    await sink.sync()
    stopped = sink.of_type(PlaybackStopped)
    assert len(stopped) == 1, f"criterion 5: exactly one PlaybackStopped, got {stopped}"
    assert stopped[0].played_ms == first, "the event carries the number flush() returned"

    second = await call(io, "flush", "AudioIO protocol")
    await play_out(device, 300.0)
    await sink.sync()
    assert second == first, f"second flush() returned {second}, first returned {first}"
    assert len(sink.of_type(PlaybackStopped)) == 1, (
        "criterion 5: a second flush() before the next play() emits no second event"
    )

    last_sound = device.last_nonzero_rendered_ms()
    assert last_sound is not None and last_sound <= 400.0 + STOP_MS, (
        f"audio kept playing to {last_sound} ms after flush() dropped the queue (criterion 5)"
    )
    await sink.close()


@pytest.mark.parametrize(("latency_ms", "dac_lag_ms", "quantum_ms"), LAG_CASES, ids=LAG_IDS)
async def test_underrun_does_not_drain(
    latency_ms: float, dac_lag_ms: float, quantum_ms: float
) -> None:
    """Criterion 7, adversarial: a 400 ms TTS stall is an underrun, never a drain.

    Without this, docs/02 §4's ``SPEAKING + playback.drained ∧ llm.done → IDLE`` ends the turn
    mid-sentence every time the TTS hiccups. Run against both device shapes (ruling A12): on the
    lagged one the device's rendered position saturates at ``written − latency``, so a module
    that spots the dry queue by "rendered caught up with written" never counts the underrun and
    the assertion below is the only thing that says so before the hardware does.
    """
    clock = VirtualClock()
    io, device = rig(
        clock,
        buffer_ms=200.0,
        latency_ms=latency_ms,
        dac_lag_ms=dac_lag_ms,
        quantum_ms=quantum_ms,
    )
    sink = EventSink(io)
    utterance = PcmUtterance(samples=tone(1000.0), hold_after_ms=300.0)
    await play(io, utterance, source="half a sentence and then a stall")

    await play_out(device, 700.0)  # 300 ms of audio, then 400 ms with the queue empty
    await sink.sync()
    assert sink.of_type(PlaybackDrained) == [], (
        "criterion 7: the queue emptied because the TTS stalled, not because the stream ended — "
        "PlaybackDrained requires mark_drain()"
    )
    starved_ms = 1000.0 * device.underflow_frames / DEVICE_RATE
    assert starved_ms >= 300.0, (
        f"the device only reported {starved_ms:.0f} ms of underflow for a 400 ms stall: the test "
        f"did not reproduce the failure it is about"
    )
    underruns = attr_or_fail(io, "underruns", "criterion 7: underruns are counted, not drained")
    assert int(underruns) >= 1, (
        f"the 400 ms stall was not counted as an underrun ({underruns}) while the device reports "
        f"{starved_ms:.0f} ms of underflow (ruling A11: the count reads the backend's own "
        f"underflow_frames, which is the only signal that survives the DAC lag)"
    )

    utterance.release()
    await play_out(device, 900.0)
    await sink.sync()
    assert sink.of_type(PlaybackDrained) == [], (
        "criterion 7: still no mark_drain(), so still no drain even though everything played"
    )
    await sink.close()


@pytest.mark.parametrize(("latency_ms", "dac_lag_ms", "quantum_ms"), LAG_CASES, ids=LAG_IDS)
async def test_drain_only_after_mark_drain_and_last_sample(
    latency_ms: float, dac_lag_ms: float, quantum_ms: float
) -> None:
    """Criterion 7: both conditions, in that order, and the run total rides on the event.

    The lagged case (ruling A12) is the one the hardware runs: ``rendered_ms()`` saturates at
    ``written − latency``, so the drain condition has to be "everything written has been rendered
    *as far as the device can report*" (ruling A11) — ``rendered >= written − latency``. A module
    that waits for ``rendered >= written`` never ends a turn on real hardware.
    """
    clock = VirtualClock()
    io, device = rig(
        clock,
        buffer_ms=200.0,
        latency_ms=latency_ms,
        dac_lag_ms=dac_lag_ms,
        quantum_ms=quantum_ms,
    )
    sink = EventSink(io)
    await play(io, PcmUtterance(samples=tone(600.0)), source="six hundred")

    await play_out(device, 300.0)
    await call(io, "mark_drain", "criterion 7: an explicit end-of-stream")
    await sink.sync()
    assert sink.of_type(PlaybackDrained) == [], (
        "criterion 7: mark_drain() alone does not drain — 300 ms of the 600 are still unplayed"
    )

    await play_out(device, 500.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, (
        f"expected exactly one PlaybackDrained, got {drained}; the device rendered "
        f"{device.rendered_ms():.0f} of the {device.written_ms:.0f} ms written and cannot report "
        f"more than written − {device.dac_lag_ms:.0f} ms of DAC lag (rulings A11/A12)"
    )
    # Ruling A18: a drain is the end of a *complete* stream — nothing cancels it, so the run
    # total is everything that was enqueued, and the last word stays in history. On a lagged
    # device ``rendered − base`` is short by the lag (580 ms at dac-lag), which is what makes
    # this assertion distinguish the two. ``flush()`` is the opposite and keeps the device clock
    # (test_flush_is_idempotent_and_emits_one_stopped).
    assert drained[0].played_ms == 600, (
        f"PlaybackDrained.played_ms={drained[0].played_ms} for a 600 ms stream the device "
        f"rendered {device.rendered_ms():.0f} ms of: a drained run reports what was enqueued, "
        f"not the device clock (ruling A18)"
    )
    assert drained[0].played_text == "six hundred", (
        f"criterion 6: a fully played utterance contributes its whole source, got "
        f"{drained[0].played_text!r}"
    )
    assert sink.of_type(PlaybackStopped) == [], "criterion 8: at most one terminal event per run"
    await sink.close()


async def test_mark_drain_while_the_last_iterator_is_open_does_not_drain() -> None:
    """Criterion 7 / ruling A17: ``TtsDone`` arrives before the last utterance has finished.

    This is the **normal** T-0007 ordering, not a corner: the executor calls ``mark_drain()``
    when ``TtsDone`` has arrived and the last utterance is queued (docs/tasks/T-0006.md, "Seam to
    session.py"), while that utterance's ``audio()`` iterator is still open. A stall inside it is
    an underrun; the stream has not ended. A drain here would end the turn on a sentence the user
    is still waiting for, and the history would keep text that was never spoken — the §4
    invariant, failing silently.

    The conjunct this pins is "every utterance's iterator is exhausted": with 300 ms of the 600
    written, rendered and handed over, every *other* condition a drain rule checks (nothing
    pending, something written, the device caught up as far as it reports) is already true.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0, latency_ms=20.0, dac_lag_ms=20.0)
    sink = EventSink(io)
    utterance = PcmUtterance(samples=tone(600.0), hold_after_ms=300.0)
    await play(io, utterance, source="six hundred milliseconds of speech")

    await play_out(device, 100.0)
    await call(io, "mark_drain", "criterion 7: TtsDone arrived and the last utterance is queued")
    await play_out(device, 600.0)  # the first 300 ms play out; the iterator is still open
    await sink.sync()

    assert sink.of_type(PlaybackDrained) == [], (
        f"PlaybackDrained fired while the last utterance's iterator was still open: "
        f"{device.rendered_ms():.0f} of {device.written_ms:.0f} ms rendered, but only 300 of the "
        f"600 ms of this utterance have arrived from the TTS (ruling A17)"
    )
    assert device.underflow_frames > 0, "the device did not run dry: the case did not reproduce"
    underruns = int(attr_or_fail(io, "underruns", "criterion 7: a mid-utterance stall is counted"))
    assert underruns >= 1, (
        f"the stall inside the last utterance was not counted as an underrun ({underruns}); "
        f"mark_drain() says no *further* utterance is coming, not that this one has ended"
    )

    utterance.release()
    await play_out(device, 600.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, (
        f"expected exactly one PlaybackDrained once the iterator closed and the audio was out, "
        f"got {drained} (device: {device.rendered_ms():.0f} of {device.written_ms:.0f} ms, "
        f"queued {device.queued_ms():.0f} ms)"
    )
    assert drained[0].played_ms == 600, (
        f"PlaybackDrained.played_ms={drained[0].played_ms} for the 600 ms stream (ruling A18)"
    )
    assert drained[0].played_text == "six hundred milliseconds of speech"
    await sink.close()


async def test_no_drain_while_the_device_still_holds_the_tail() -> None:
    """Criterion 7 / rulings A15+A19: the device-side half of "everything has been rendered".

    Our queue being empty is not the end of the stream — it means the audio is *inside the
    device*. Here 85 ms of a 100 ms stream have been taken, the last 15 ms have been handed over
    and the device then stops consuming (an xrun, a suspended sink, a graph renegotiation). A
    drain declared now ends the turn on audio that has not left the DAC, and ``played_text``
    claims words the user has not heard yet; the §4 invariant fails silently, which is the whole
    reason criterion 7 exists.

    The conjunct pinned is ``queued_ms() == 0``: every other condition — nothing pending, the
    iterator exhausted, something written this run — is already true when the assertion runs.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0, latency_ms=20.0, dac_lag_ms=20.0)
    sink = EventSink(io)
    await play(io, PcmUtterance(samples=tone(100.0)), source="one hundred")
    await call(io, "mark_drain", "criterion 7: TtsDone arrived and the last utterance is queued")

    await play_out(device, 85.0)  # 85 % of the stream taken; the tail is inside the device
    device.stall(300.0)
    await play_out(device, 300.0)
    await sink.sync()

    assert device.written_ms == pytest.approx(100.0, abs=0.5), (
        f"the module has written {device.written_ms:.1f} of the 100 ms stream; the case needs "
        f"our own queue empty before the device stalls"
    )
    assert device.queued_ms() > 0.0, (
        f"the device holds nothing ({device.queued_ms():.1f} ms queued): the stall was placed "
        f"too late and the case did not reproduce"
    )
    assert sink.of_type(PlaybackDrained) == [], (
        f"PlaybackDrained fired while {device.queued_ms():.1f} ms were still inside the device "
        f"(rendered {device.rendered_ms():.1f} of {device.written_ms:.1f} ms written): the drain "
        f"has a device-side half — queued_ms() == 0 — and it is not optional (rulings A15/A19)"
    )

    await play_out(device, 200.0)  # the stall passes, the device takes the tail
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, (
        f"expected one PlaybackDrained once the device had taken everything, got {drained} "
        f"(queued {device.queued_ms():.1f} ms)"
    )
    assert drained[0].played_ms == 100, f"ruling A18: got {drained[0].played_ms}"
    await sink.close()


async def test_no_drain_while_our_queue_still_holds_audio() -> None:
    """Criterion 7 / ruling A19: ``mark_drain()`` then ``pause()`` — the stream has not ended.

    ``mark_drain()`` says no *further* utterance is coming; it says nothing about the 480 ms of
    this one still sitting in our queue. A drain here reports ``played_ms = enqueued_ms`` (ruling
    A18, correct for a run that really ended) and so writes the whole 600 ms of text into history
    after 100 ms was heard — the barge-in case, with the paused remainder silently committed.

    The conjunct pinned is ``pending == 0``: after 500 ms of paused time the device has taken
    everything it was given and every iterator is exhausted, so nothing else stands in the way.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0, latency_ms=20.0, dac_lag_ms=20.0)
    sink = EventSink(io)
    source = "six hundred milliseconds of speech"
    await play(io, PcmUtterance(samples=tone(600.0)), source=source)

    await play_out(device, 100.0)
    await call(io, "mark_drain", "criterion 7: TtsDone arrived and the last utterance is queued")
    await call(io, "pause", "AudioIO protocol")
    await play_out(device, 500.0)
    await sink.sync()

    assert device.written_ms <= 200.0, (
        f"the module wrote {device.written_ms:.1f} ms before the pause took effect; the case "
        f"needs most of the 600 ms still in our queue"
    )
    assert device.queued_ms() == pytest.approx(0.0, abs=1.0), (
        f"the device still holds {device.queued_ms():.1f} ms, so the device-side conjunct would "
        f"withhold the drain on its own and this test would not pin ours"
    )
    # the device's own position, not played_ms(): a mutant that has already drained the run
    # answers with the enqueued total and would make its own failure message read right.
    heard = device.rendered_ms()
    assert sink.of_type(PlaybackDrained) == [], (
        f"PlaybackDrained fired with the run paused and ~{600.0 - device.written_ms:.0f} ms still "
        f"in our queue: history would take the whole {source!r} while the device rendered "
        f"{heard:.0f} ms (ruling A19)"
    )

    await call(io, "resume", "AudioIO protocol")
    await play_out(device, 700.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, f"expected one PlaybackDrained after resume, got {drained}"
    assert drained[0].played_ms == 600, f"ruling A18: got {drained[0].played_ms}"
    assert drained[0].played_text == source
    await sink.close()


async def test_playback_events_queue_drops_oldest_never_newest_under_backpressure() -> None:
    """Criterion 8: ≥ 8 slots, the newest kept, the drops counted.

    The consumer here is infinitely slow: the iterator exists (so the queue is live) but nothing
    reads it until twelve runs have finished. Losing the *oldest* is acceptable; losing the
    newest would hide the truncation the turn machine is about to write into history.
    """
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    sink = EventSink(io, autostart=False)
    sink.arm()

    runs = 12
    for i in range(runs):
        await play(io, PcmUtterance(samples=tone(300.0)), source=f"run{i:02d}")
        await play_out(device, 400.0)
        await call(io, "flush", "AudioIO protocol")

    sink.start()
    await sink.sync(rounds=40)
    texts = [e.played_text for e in sink.of_type(PlaybackStopped)]
    assert len(texts) >= 8, f"criterion 8: the queue holds ≥ 8 events, got {len(texts)}"
    assert texts == sorted(texts), f"events must arrive in emission order, got {texts}"
    expected_tail = [f"run{i:02d}" for i in range(runs - len(texts), runs)]
    assert texts == expected_tail, (
        f"criterion 8: the newest events must survive; got {texts}, expected the last "
        f"{len(texts)} runs {expected_tail}"
    )
    dropped = int(attr_or_fail(io, "dropped_events", "criterion 8: the drop is counted"))
    assert dropped == runs - len(texts), (
        f"{runs} events emitted, {len(texts)} delivered, but dropped_events={dropped}"
    )
    await sink.close()


async def test_the_device_gets_10ms_48khz_f32_mono_blocks() -> None:
    """Criterion 1: block cadence, and the 24 → 48 kHz stage is applied exactly once."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    await play(io, PcmUtterance(samples=tone(500.0)), source="five hundred")
    await play_out(device, 500.0)

    assert device.block_violations == [], (
        f"criterion 1: every write is one 10 ms block of 48 kHz f32 mono; violations: "
        f"{device.block_violations[:5]}"
    )
    assert device.write_calls > 0, "nothing was written to the device"
    assert abs(device.written_ms - 500.0) <= 60.0, (
        f"{device.written_ms:.1f} ms of 48 kHz audio for a 500 ms 24 kHz utterance — 250 ms "
        f"means the resample was skipped, 1000 ms means it was applied twice"
    )
    assert CAPTURE_BLOCK_MS == 10


async def test_the_reported_quantum_is_the_negotiated_one_not_the_request() -> None:
    """Criterion 1: report what the graph gave you (PipeWire's default is 1024/48000 ≈ 21 ms)."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0, quantum_ms=21.333333, latency_ms=25.0)
    assert attr_or_fail(io, "quantum_ms", "criterion 1: the negotiated quantum") == pytest.approx(
        device.quantum_ms
    ), "the module must report the device's quantum, not the 10 ms it asked for"
    assert attr_or_fail(io, "block_ms", "criterion 1") == CAPTURE_BLOCK_MS
    assert attr_or_fail(io, "latency_ms", "criterion 1") == pytest.approx(device.latency_ms)
    # The ≤ 30 ms cap is asserted on the *negotiated* value, on the rig (ruling A9):
    # tests/integration/test_audio_null_sink.py::test_negotiated_quantum_is_reported_not_assumed.


async def test_playback_reference_is_the_mixed_output_and_the_ring_is_bounded() -> None:
    """Criterion 10: the far-end reference T-0006b's gate needs, from a ring of ≤ 2 s."""
    clock = VirtualClock()
    io, device = rig(clock, buffer_ms=200.0)
    await play(io, PcmUtterance(samples=tone(8000.0, freq_hz=300.0)), source="reference")
    await play_out(device, 2500.0)

    early, _ = await call(io, "playback_reference", "criterion 10", 1_000_000.0)
    await play_out(device, 2500.0)
    late, _ = await call(io, "playback_reference", "criterion 10", 1_000_000.0)

    window_ms = 500.0
    samples, first_sample_device_ms = await call(
        io, "playback_reference", "criterion 10", window_ms
    )
    samples = np.asarray(samples)
    assert samples.dtype == np.float32, f"reference must be f32, got {samples.dtype}"
    assert samples.size == pytest.approx(window_ms * DEVICE_RATE / 1000.0, abs=DEVICE_RATE // 100)

    start = round(first_sample_device_ms * DEVICE_RATE / 1000.0)
    expected = device.written_audio()[start : start + samples.size]
    assert expected.size == samples.size, (
        f"first_sample_device_ms={first_sample_device_ms} does not point into the audio the "
        f"device received ({device.written_ms:.0f} ms of it)"
    )
    assert np.allclose(samples, expected, atol=1e-6), (
        "the reference is not the audio that was actually sent to the device (criterion 10)"
    )

    assert np.asarray(late).size <= 2.0 * DEVICE_RATE, (
        f"the reference ring holds {np.asarray(late).size / DEVICE_RATE:.1f} s; criterion 10 "
        f"caps it at 2 s so a 10 minute run does not grow"
    )
    assert np.asarray(late).size == np.asarray(early).size, (
        f"the ring grew from {np.asarray(early).size} to {np.asarray(late).size} samples over "
        f"2.5 s of playback: it is not a ring, and a 10 minute run would keep 115 MB "
        f"(criterion 10)"
    )

    assert (await call(io, "is_rendering", "criterion 10", first_sample_device_ms)) is True
    far_future = first_sample_device_ms + 10_000_000.0
    assert (await call(io, "is_rendering", "criterion 10", far_future)) is False
