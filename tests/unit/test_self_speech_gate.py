"""The self-speech gate: T-0021 criterion 8 (docs/02 §4 spec delta, 2026-09-13).

The gate sits on the **STT tap only**. It is the one place in this task where a wrong design
passes every other test in the suite: gate the raw stream instead and every STT assertion here
still holds, while the reflex VAD goes deaf during playback and ``SPEAKING → PAUSED`` — barge-in —
quietly stops working. ``test_raw_capture_is_untouched_during_playback`` is that test, and it is
one of the five mutations the reviewer runs.

Seams fixed here (reported as such): ``audio_capture.SelfSpeechGate(is_rendering, hangover_ms)``
with ``should_gate(t_ms) -> bool``, called once per block in capture order; the default
``hangover_ms`` is T-0006's measured echo-path delay p95 + one block, recorded with its regime in
the module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    STT_BLOCK_MS,
    STT_RATE,
    WHY_CAPTURE,
    CaptureSink,
    FakeAudioDevice,
    PcmUtterance,
    attr_or_fail,
    capture_out,
    capture_pattern,
    capture_stats,
    concat,
    import_capture,
    make_audio_io,
    play,
    samples_of,
    stat,
    t_ms_of,
    tone,
)

WHY = f"{WHY_CAPTURE} — criterion 8: SelfSpeechGate(is_rendering, hangover_ms)"
#: −60 dBFS: what "silence" means for the gated STT stream (criterion 9 uses the same number).
SILENT = 1e-3


def loud_pattern(start: int, n: int) -> np.ndarray:
    """A capture signal that is never quiet (0.5 ± 0.03) and still identifiable by index.

    A source that crosses zero would make "this region is silent" accidentally true for a few
    hundred samples; a decimator passes DC, so the offset survives the 48 → 24 kHz stage.
    """
    return (0.5 + 0.06 * capture_pattern(start, n)).astype(np.float32)


def gate(is_rendering: object, hangover_ms: float | None = None) -> object:
    cls = attr_or_fail(import_capture(), "SelfSpeechGate", WHY)
    return cls(is_rendering) if hangover_ms is None else cls(is_rendering, hangover_ms)


def should_gate(g: object, t_ms: float) -> bool:
    value = attr_or_fail(g, "should_gate", WHY)(t_ms)
    assert isinstance(value, bool), f"should_gate({t_ms}) returned {value!r}, want a bool"
    return value


def window(start: float, end: float) -> object:
    """``is_rendering`` for a single playback of [start, end) ms."""
    return lambda t_ms: start <= t_ms < end


def sweep(g: object, upto_ms: float) -> dict[float, bool]:
    """Call the gate once per 10 ms block, in order — the way the capture path does."""
    return {float(t): should_gate(g, float(t)) for t in range(0, int(upto_ms), CAPTURE_BLOCK_MS)}


async def test_the_leading_edge_is_immediate() -> None:
    """Criterion 8: the first block that overlaps playback is gated, not the second."""
    verdicts = sweep(gate(window(100.0, 300.0), 60.0), 400.0)
    assert verdicts[90.0] is False, "a block captured before any playback must not be gated"
    assert verdicts[100.0] is True, (
        "the first block of the playback window is not gated: the gate reacts one block late, "
        "so 10 ms of our own voice reaches the STT at every utterance (criterion 8)"
    )
    assert verdicts[110.0] is True


async def test_the_hangover_holds_the_gate_shut_after_the_last_sample() -> None:
    """Criterion 8: the trailing edge is the hangover, not the end of rendering."""
    verdicts = sweep(gate(window(100.0, 200.0), 60.0), 400.0)
    still_open = [t for t, gated in verdicts.items() if 200.0 <= t <= 240.0 and not gated]
    assert still_open == [], (
        f"blocks at {still_open} ms are not gated although playback ended at 200 ms and the "
        f"hangover is 60 ms: the room (or the monitor) is still returning our own audio there"
    )
    assert verdicts[250.0] is True, (
        "the block at 250 ms is not gated. Ruling A16(c): the hangover is anchored on the end of "
        "the last rendering block (t_ms + block_ms), so the protection past the last rendering "
        "sample really is the documented p95 + one block. Rendering here covers [100, 200), the "
        "last rendering block starts at 190 and ends at 200, and 200 + 60 ms of hangover leaves "
        "every block up to and including 250 ms gated; anchoring on the block's *start* instead "
        "reopens the gate at 250 ms and gives only p95 of protection, which is not what the "
        "constant and the docstring say (criterion 8)"
    )
    stuck = [t for t, gated in verdicts.items() if t >= 270.0 and gated]
    assert stuck == [], (
        f"blocks at {stuck[:5]} ms are still gated more than one hangover (60 ms) after playback "
        f"ended: the gate never reopens and the user cannot be heard"
    )


async def test_a_five_millisecond_gap_between_utterances_does_not_un_gate() -> None:
    """Criterion 8, adversarial 1 (no chatter) — **mutation 3: ``hangover_ms = 0`` fails here.**

    Two utterances with a 5 ms gap between them. ``is_rendering`` is false for that gap, so a
    gate with no hangover opens for exactly one block and lets one block of our own voice into
    the STT — once per sentence boundary, which is once every couple of seconds in practice.
    """
    rendering = lambda t_ms: (0.0 <= t_ms < 100.0) or (105.0 <= t_ms < 300.0)  # noqa: E731
    verdicts = sweep(gate(rendering, 60.0), 300.0)
    leaked = [t for t, gated in verdicts.items() if not gated]
    assert leaked == [], (
        f"blocks at {leaked} ms reached the STT ungated. The only non-rendering instant in this "
        f"test is the 5 ms gap at 100-105 ms between two utterances; with a hangover of 60 ms "
        f"the gate must not chatter there (criterion 8, adversarial 1)"
    )

    # The 5 ms case above cannot, on its own, separate `hangover_ms = 0` from a gate that keys
    # on the *end* of the last rendering block: the block at 100 ms starts exactly where
    # rendering stopped, so 0 ms of hangover still covers it. A 15 ms gap puts a block start
    # 10 ms clear of the last rendering sample, where only a real hangover can reach.
    wider = lambda t_ms: (0.0 <= t_ms < 100.0) or (115.0 <= t_ms < 300.0)  # noqa: E731
    verdicts = sweep(gate(wider, 60.0), 300.0)
    leaked = [t for t, gated in verdicts.items() if not gated]
    assert leaked == [], (
        f"blocks at {leaked} ms reached the STT ungated across a 15 ms gap between two "
        f"utterances, with a 60 ms hangover: the gate reopens between sentences and chatters "
        f"our own voice into the STT (criterion 8, adversarial 1; hangover_ms = 0 fails here)"
    )


async def test_the_gate_is_pure_and_takes_its_clock_from_the_caller() -> None:
    """Criterion 8: injected ``is_rendering``, no clock of its own, no I/O."""
    calls: list[float] = []

    def is_rendering(t_ms: float) -> bool:
        calls.append(t_ms)
        return 100.0 <= t_ms < 200.0

    g = gate(is_rendering, 50.0)
    first = [should_gate(g, float(t)) for t in range(0, 400, CAPTURE_BLOCK_MS)]
    assert calls, "the gate never consulted the injected is_rendering: it has a clock of its own"
    assert all(t in {float(x) for x in range(0, 400, CAPTURE_BLOCK_MS)} for t in calls), (
        f"the gate asked is_rendering about times nobody gave it ({sorted(set(calls))[:5]}): the "
        f"argument must be the block's own capture timestamp (criterion 8)"
    )
    twin = gate(is_rendering, 50.0)
    again = [should_gate(twin, float(t)) for t in range(0, 400, CAPTURE_BLOCK_MS)]
    assert first == again, "two gates fed the same sequence disagreed: the gate is not pure"


async def test_the_default_hangover_is_documented_with_its_regime() -> None:
    """Criterion 8: the default is T-0006's measured echo-path delay p95 + one block."""
    g = gate(window(0.0, 0.0))
    hangover = float(attr_or_fail(g, "hangover_ms", WHY))
    assert CAPTURE_BLOCK_MS <= hangover <= 500.0, (
        f"the default hangover is {hangover} ms; criterion 8 says echo-path delay p95 + one "
        f"block ({CAPTURE_BLOCK_MS} ms), which is neither zero nor half a second"
    )
    doc = import_capture().__doc__ or ""
    assert "hangover" in doc.lower() and "p95" in doc.lower(), (
        "criterion 8 requires the hangover number **and its regime** in the module docstring "
        f"(CLAUDE.md: a number without its regime cannot be compared later); docstring is: "
        f"{doc[:200]!r}"
    )


def rig(clock: VirtualClock | None = None, **kwargs: object) -> tuple[object, FakeAudioDevice]:
    device = FakeAudioDevice(clock if clock is not None else VirtualClock(), **kwargs)  # type: ignore[arg-type]
    return make_audio_io(device), device


async def _play_and_capture(play_ms: float = 200.0) -> tuple[object, FakeAudioDevice, list, list]:
    """100 ms of quiet, ``play_ms`` of playback, then 400 ms more — raw and STT both drained."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, capture_source=loud_pattern)
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")
    await capture_out(device, 100.0, raw, stt)
    await play(io, PcmUtterance(samples=tone(play_ms)), source="self speech")
    await capture_out(device, play_ms, raw, stt)
    await capture_out(device, 400.0, raw, stt)
    raw.raise_error()
    stt.raise_error()
    blocks, frames = list(raw.items), list(stt.items)
    await raw.close()
    await stt.close()
    return io, device, blocks, frames


async def test_raw_capture_is_untouched_during_playback() -> None:
    """Criterion 8, adversarial 2 — **mutation 5: gating the raw stream fails here.**

    While we are speaking, ``capture()``/``capture_frames()`` must keep delivering the device's
    audio, unmodified, at cadence: the reflex VAD reads that stream and it is the only thing that
    can notice a barge-in (docs/02 §4 ``SPEAKING | user_speech_onset | PAUSED``). Marking the
    block ``gated`` is allowed — zeroing it, dropping it or holding it back is not.
    """
    io, _device, blocks, _frames = await _play_and_capture()
    stats = await capture_stats(io)

    assert len(blocks) == 70, (
        f"700 ms of capture is 70 blocks, {len(blocks)} arrived: blocks were dropped or held "
        f"back while we were speaking (criterion 8: the raw stream is never delayed)"
    )
    stamps = [t_ms_of(b) for b in blocks]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(CAPTURE_BLOCK_MS)] * len(steps), abs=1e-6), (
        "the raw cadence broke during playback"
    )
    bad = [
        i
        for i, b in enumerate(blocks)
        if not np.array_equal(samples_of(b), loud_pattern(i * BLOCK_SAMPLES, BLOCK_SAMPLES))
    ]
    assert bad == [], (
        f"{len(bad)} raw blocks (first at index {bad[0] if bad else None}, t_ms="
        f"{stamps[bad[0]] if bad else None}) do not carry the audio the device captured. The "
        f"blocks in 100-300 ms are the ones playback overlaps: the gate zeroed the raw stream "
        f"instead of the STT tap, which silently destroys barge-in while every STT test still "
        f"passes (criterion 8, adversarial 2)"
    )
    gated_flags = sum(1 for b in blocks if bool(getattr(b, "gated", False)))
    assert gated_flags >= 15, (
        f"only {gated_flags} of the ~20 blocks captured during playback are marked gated=True; "
        f"the raw stream carries the mark (criterion 2's CaptureBlock field), not the silence"
    )
    assert int(stat(stats, "gated_blocks")) >= 15, "criterion 8: gated blocks are counted"
    assert int(stat(stats, "dropped_blocks")) == 0, "criterion 8: nothing is dropped for gating"


async def test_gated_blocks_become_silence_in_the_stt_stream() -> None:
    """Criterion 8: the STT tap hears nothing while we speak, and its timeline keeps running."""
    _io, _device, _blocks, frames = await _play_and_capture()
    hangover = float(attr_or_fail(gate(window(0.0, 0.0)), "hangover_ms", WHY))

    assert frames, "no STT frames at all"
    stamps = [t_ms_of(f) for f in frames]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(STT_BLOCK_MS)] * len(steps), abs=1e-6), (
        f"gating punched a hole in the STT timeline ({steps}); criterion 8 replaces a gated "
        f"block with silence of the same duration so criterion 4's contiguity survives"
    )
    assert len(frames) * STT_BLOCK_MS >= 600.0, (
        f"{len(frames)} frames for 700 ms of capture: gated blocks were dropped instead of "
        f"replaced by silence"
    )

    y = concat(frames)
    guard = round((CAPTURE_BLOCK_MS + 2.0) * STT_RATE / 1000.0)
    lo = round(100.0 * STT_RATE / 1000.0) + guard  # playback covers capture time 100-300 ms
    hi = round(300.0 * STT_RATE / 1000.0) - guard
    peak = float(np.abs(y[lo:hi]).max())
    assert peak <= SILENT, (
        f"the STT stream is at {20 * np.log10(max(peak, 1e-12)):.0f} dBFS while we were speaking "
        f"(peak {peak:.4f}, want ≤ {SILENT} = −60 dBFS): the gate is not on the STT tap"
    )
    quiet_until = round((300.0 + hangover + CAPTURE_BLOCK_MS) * STT_RATE / 1000.0)
    after = np.abs(y[quiet_until : quiet_until + 2 * guard])
    assert after.size and float(after.min()) > 0.2, (
        f"the STT stream is still silent {hangover:.0f} ms (the hangover) after playback ended: "
        f"the gate never reopened, so the user is inaudible after every answer"
    )
    # from 20 ms: the decimation filter starts with zero history, so the first samples of any
    # stream ramp up from silence and say nothing about the gate.
    before = np.abs(y[round(20.0 * STT_RATE / 1000.0) : round(90.0 * STT_RATE / 1000.0)])
    assert float(before.min()) > 0.2, (
        "the STT stream was already silent before playback started: this test would pass on a "
        "capture path that produces nothing at all"
    )
