"""Both playback implementations answer the same questions: T-0006 (docs/02 §5, §4).

Two implementations of the same interface run the same behavioural assertions: the real module
driven by ``FakeAudioDevice``, and ``tests/fakes/audio.py::FakeAudioIO`` driven by the virtual
clock (what the conductor's unit tests run against — if it and the module disagree about
``flush()`` or about what counts as a drain, every test written against the fake is a lie).

Parametrised locally on purpose: ``tests/contract/conftest.py``'s ``backend`` fixture marks its
``real`` case ``gpu``, and nothing here needs a GPU. The real-*device* conformance of the same
assertions lives in ``tests/integration/test_audio_null_sink.py`` (markers ``integration`` +
``audio``), per ``.claude/rules/tests.md`` "a behaviour added to a fake needs a real-service test
in the same PR".

``FakeAudioIO`` has to grow ``mark_drain()``/``playback_events()``/``played_text`` for this to
pass; that edit belongs to the same PR (CLAUDE.md) and is *not* the implementer's to make.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.protocols import AudioIO
from tests.conftest import VirtualClock
from tests.fakes.audio import FakeAudioIO
from tests.fakes.audio_device import (
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    attr_or_fail,
    call,
    make_audio_io,
    play,
    play_out,
    played_ms,
    tone,
    words_of,
)

pytestmark = [pytest.mark.contract]

SOURCE = "one two three"
WORDS = words_of([("one", 0, 300), ("two", 400, 700), ("three", 800, 1100)])
UTTERANCE_MS = 1200.0
OVER_MS, UNDER_MS = 20.0, 40.0
#: Seams T-0006 adds to the playback interface. The protocol itself is widened by T-0017
#: (docs/tasks/T-0006.md "Files this task needs and does not own"), so conformance is checked by
#: presence here, exactly as ``tests/fakes/audio.py`` already exposes ``played_ms()``.
T0006_SEAMS = ("played_ms", "mark_drain", "playback_events")


def _utterance() -> PcmUtterance:
    return PcmUtterance("u1", samples=tone(UTTERANCE_MS), words=WORDS, text=SOURCE)


#: Output latency of the lagged parametrisation, and the lag its rendered position carries.
#: The two are equal on purpose: ``PortAudioDevice`` subtracts exactly the latency it reports,
#: so a module written against the reported number can satisfy both cases (ruling A12).
LAG_MS = 20.0


#: ``(latency_ms, dac_lag_ms, quantum_ms)`` per implementation id (rulings A12/A15/A16). The lag
#: the rendered position actually carries is not the latency the device reports: PortAudio
#: measures it per callback (15.3 ms against a negotiated 10 ms) and another client can force the
#: graph quantum to ≈ 21 ms, so ``quantum-lag`` is the hardware's shape, not a stress case.
SHAPES = {
    "module": (LAG_MS, 0.0, 10.0),
    "module-lag": (LAG_MS, LAG_MS, 10.0),
    "module-quantum-lag": (10.0, 31.3, 21.333),
}


def _build(impl: str) -> tuple[object, Callable[[float], Awaitable[None]]]:
    clock = VirtualClock()
    if impl.startswith("module"):
        latency, lag, quantum = SHAPES[impl]
        device = FakeAudioDevice(
            clock, buffer_ms=200.0, latency_ms=latency, dac_lag_ms=lag, quantum_ms=quantum
        )
        io = make_audio_io(device)

        async def advance(ms: float) -> None:
            await play_out(device, ms)

        return io, advance

    io = FakeAudioIO(clock)

    async def advance_clock(ms: float) -> None:
        clock.advance(int(ms))

    return io, advance_clock


#: ``module-lag`` is the real backend's shape (ruling A12): ``rendered_ms()`` saturates at
#: ``written − latency``, because audio handed to the DAC has not left it yet. A drain or
#: underrun rule written as "rendered caught up with written" is green on ``module`` and silent
#: forever on ``module-lag`` — and on the hardware.
IMPLS = ["module", "module-lag", "module-quantum-lag", "fake"]


@pytest.mark.parametrize("impl", IMPLS)
async def test_it_is_an_audio_io(impl: str) -> None:
    io, _ = _build(impl)
    assert isinstance(io, AudioIO), (
        f"{impl}: does not satisfy the runtime_checkable AudioIO protocol (contracts/protocols.py)"
    )
    missing = [name for name in T0006_SEAMS if not hasattr(io, name)]
    assert missing == [], f"{impl}: missing the T-0006 playback seams {missing}"


@pytest.mark.parametrize("impl", IMPLS)
async def test_played_ms_tracks_the_audio_and_never_goes_backwards(impl: str) -> None:
    io, advance = _build(impl)
    await play(io, _utterance(), source=SOURCE)
    await advance(400.0)
    first = await played_ms(io)
    assert 400.0 - UNDER_MS <= first <= 400.0 + OVER_MS, f"{impl}: played_ms={first} after 400 ms"
    await advance(300.0)
    second = await played_ms(io)
    assert second >= first, f"{impl}: played_ms went backwards {first} -> {second}"
    assert second <= UTTERANCE_MS, f"{impl}: played_ms={second} exceeds the utterance"


@pytest.mark.parametrize("impl", IMPLS)
async def test_flush_is_idempotent_and_terminal(impl: str) -> None:
    io, advance = _build(impl)
    sink = EventSink(io)
    await play(io, _utterance(), source=SOURCE)
    await advance(500.0)

    first = await call(io, "flush", "AudioIO protocol")
    second = await call(io, "flush", "AudioIO protocol")
    await sink.sync()
    assert first == second, f"{impl}: flush() is idempotent (contracts/commands.py:223)"
    assert isinstance(first, int), f"{impl}: flush() returns ms as an int, got {first!r}"
    stopped = sink.of_type(PlaybackStopped)
    assert len(stopped) == 1, f"{impl}: exactly one PlaybackStopped per run, got {stopped}"
    assert stopped[0].played_ms == first
    await sink.close()


@pytest.mark.parametrize("impl", IMPLS)
async def test_a_queue_that_empties_is_not_a_drain(impl: str) -> None:
    """docs/02 §4 spec delta (1): ``PlaybackDrained`` needs ``mark_drain()`` as well."""
    io, advance = _build(impl)
    sink = EventSink(io)
    await play(io, _utterance(), source=SOURCE)
    await advance(UTTERANCE_MS + 500.0)
    await sink.sync()
    assert sink.of_type(PlaybackDrained) == [], (
        f"{impl}: everything queued played out, but no mark_drain() was called — that is an "
        f"underrun, not a drain"
    )
    underruns = int(attr_or_fail(io, "underruns", "criterion 7 / ruling A11: underruns counted"))
    assert underruns >= 1, (
        f"{impl}: the device ran dry for 500 ms with nothing to play and underruns is "
        f"{underruns}. Ruling A11: the count reads the backend's own underflow, so a rule that "
        f"waits for 'rendered caught up with written' never notices on a device that lags by its "
        f"output latency"
    )

    await call(io, "mark_drain", "criterion 7")
    await advance(200.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, (
        f"{impl}: expected one PlaybackDrained after mark_drain(), got {drained}. Ruling A11: "
        f"the drain condition is 'the device rendered everything written as far as it can "
        f"report' — rendered >= written − latency — because a real backend's rendered position "
        f"saturates at written − latency and never reaches written"
    )
    assert drained[0].played_text == SOURCE, f"{impl}: played_text={drained[0].played_text!r}"
    assert drained[0].played_ms == int(UTTERANCE_MS), (
        f"{impl}: PlaybackDrained.played_ms={drained[0].played_ms} for a {UTTERANCE_MS:.0f} ms "
        f"stream — a drained run reports the enqueued total, not the device clock (ruling A18)"
    )
    await sink.close()


# --- T-0039 criterion 16: the device seam itself, over every device in the tree ------------------
#
# "The device is added to the AudioDevice conformance parametrisation so it is held to the same
# protocol test as PortAudioDevice and FakeAudioDevice" (docs/tasks/T-0039.md criterion 16). The
# rows above are the *AudioIO* conformance; this one is the *AudioDevice* one, and the null row is
# pending until ``local_astra/conductor/null_audio.py`` exists. Constructing ``PortAudioDevice``
# needs no PortAudio -- only ``start()`` does (T-0035 prerequisite P1) -- so the row is real here.
#
# Note for the reviewer (test-author, T-0039): ``NullAudioDevice`` is deliberately *not* added to
# ``IMPLS`` above, because ``test_a_queue_that_empties_is_not_a_drain`` asserts ``underruns >= 1``
# after a dry spell and T-0039 criterion 15 pins ``underflow_frames`` at 0 for the null device.
# The two cannot both hold; weakening the existing row is forbidden, so the AudioIO behaviour of
# ``PipeWireAudioIO(NullAudioDevice(...))`` (drain, played_ms, played_text, flush) is covered by
# ``tests/unit/test_null_audio.py`` instead. Flagged as an ambiguity in the T-0039 test report.

DEVICE_IMPLS = [
    "port",
    "fake",
    pytest.param("null"),
]


def _device(device_id: str) -> object:
    if device_id == "port":
        from local_astra.conductor.audio_device import PortAudioDevice

        return PortAudioDevice()
    if device_id == "fake":
        return FakeAudioDevice()
    from tests.conftest import VirtualClock
    from tests.fakes.audio_device import import_or_fail

    module = import_or_fail(
        "local_astra.conductor.null_audio",
        "T-0039 criterion 15: conductor/null_audio.py provides NullAudioDevice",
    )
    factory = attr_or_fail(module, "NullAudioDevice", "T-0039 criterion 15")
    try:
        return factory(clock=VirtualClock())
    except TypeError as exc:
        raise AssertionError(
            f"T-0039 criterion 15: NullAudioDevice(clock=...) takes its clock by injection; "
            f"the call raised {exc}"
        ) from exc


@pytest.mark.parametrize("device_id", DEVICE_IMPLS)
def test_every_device_in_the_tree_satisfies_the_audio_device_protocol(device_id: str) -> None:
    """One protocol, three backends: ``PortAudioDevice``, ``FakeAudioDevice``, ``NullAudioDevice``.

    ``PipeWireAudioIO`` is written against ``AudioDevice`` and against nothing else, which is what
    makes ``--silent`` a device swap rather than a branch in the playback path (T-0039 section 5,
    and the blast-radius trigger (b) in section 7).
    """
    from local_astra.conductor.audio_device import AudioDevice

    device = _device(device_id)
    assert isinstance(device, AudioDevice), (
        f"{device_id}: does not satisfy the runtime_checkable AudioDevice protocol "
        f"(conductor/audio_device.py); missing "
        f"{sorted(n for n in getattr(AudioDevice, '__protocol_attrs__', ()) if not hasattr(device, n))}"
    )
    for name in ("latency_ms", "block_ms", "quantum_ms", "input_latency_ms", "underflow_frames"):
        attr_or_fail(device, name, "AudioDevice protocol members (T-0006 ruling A22, T-0035)")
    assert int(device.underflow_frames) == 0, (  # type: ignore[attr-defined]
        f"{device_id}: a device that has never run has invented no frames; it reports "
        f"{device.underflow_frames}"  # type: ignore[attr-defined]
    )
