"""The 48 → 24 kHz stage: T-0021 criterion 6, in T-0006's file ``conductor/audio_resample.py``.

The device captures 48 kHz f32 (``CAPTURE_FORMAT``), moshi-server wants 24 kHz f32
(``STT_INPUT_FORMAT``), and the conversion happens on a stream that arrives in 10 ms blocks — so
the filter state has to be carried across them. Oracles are closed-form signals and float64
numpy, never the implementation.

**Why the 18 kHz case is the criterion.** A naive ``x[::2]`` passes every length test, every
chunking test and every "no clipping" test, and folds 18 kHz down to 6 kHz at full scale —
straight into the band the STT listens to. That single test is the reason this file exists.

Seam these tests fix (the task names the stage, not its API; reported as such):
``audio_resample.Resampler48kTo24k()`` with ``process(x_f32) -> y_f32`` and ``reset()``, beside
T-0006's ``Resampler24kTo48k`` in the same module and not disturbing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.contracts.speech import CAPTURE_BLOCK_MS, CAPTURE_FORMAT, STT_INPUT_FORMAT
from tests.fakes.audio_device import attr_or_fail, fft_best_lag, import_or_fail

WHY = "T-0021 criterion 6: the 48 → 24 kHz stage (local_astra/conductor/audio_resample.py)"
IN_RATE = CAPTURE_FORMAT.sample_rate  # 48000
OUT_RATE = STT_INPUT_FORMAT.sample_rate  # 24000
BLOCK = IN_RATE * CAPTURE_BLOCK_MS // 1000  # 480 input samples per 10 ms block


def decimator() -> object:
    module = import_or_fail("local_astra.conductor.audio_resample", WHY)
    return attr_or_fail(module, "Resampler48kTo24k", WHY)()


def process(r: object, x: np.ndarray) -> np.ndarray:
    out = attr_or_fail(r, "process", WHY)(np.asarray(x, dtype=np.float32))
    return np.asarray(out)


def stream(x: np.ndarray, chunk: int) -> np.ndarray:
    r = decimator()
    return np.concatenate([process(r, x[i : i + chunk]) for i in range(0, x.size, chunk)])


def sine_48k(freq_hz: float, ms: float, amplitude: float = 0.5) -> np.ndarray:
    n = round(ms * IN_RATE / 1000.0)
    t = np.arange(n, dtype=np.float64) / IN_RATE
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def sweep_48k(f0: float, f1: float, ms: float, amplitude: float = 0.5) -> np.ndarray:
    """A linear sweep — aperiodic, so a cross-correlation lag is unambiguous."""
    n = round(ms * IN_RATE / 1000.0)
    t = np.arange(n, dtype=np.float64) / IN_RATE
    k = (f1 - f0) / (ms / 1000.0)
    return (amplitude * np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))).astype(np.float32)


def sweep_24k(f0: float, f1: float, ms: float, amplitude: float = 0.5) -> np.ndarray:
    n = round(ms * OUT_RATE / 1000.0)
    t = np.arange(n, dtype=np.float64) / OUT_RATE
    k = (f1 - f0) / (ms / 1000.0)
    return amplitude * np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def band_power(x: np.ndarray, rate: int, lo: float, hi: float) -> float:
    """Energy in [lo, hi) Hz of a Hann-windowed float64 periodogram (an independent oracle)."""
    y = np.asarray(x, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    freqs = np.fft.rfftfreq(y.size, 1.0 / rate)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


async def test_18khz_tone_is_attenuated_not_aliased() -> None:
    """Criterion 6, the headline. **This is the test that kills ``x[::2]``.**

    A full-scale 18 kHz tone has nowhere to go in a 24 kHz stream: 18 kHz is above the 12 kHz
    Nyquist, so an anti-aliased decimator must reject it by ≥ 40 dB. Plain subsampling instead
    mirrors it to |24 − 18| = 6 kHz at unity gain — 0 dB, in the middle of speech. Both the
    overall level and the 5-7 kHz band (where the alias lands, computed by hand: 24000 − 18000)
    are checked, so neither a broadband attenuation nor a lucky notch elsewhere can pass.
    """
    x = sine_48k(18_000.0, 500.0, amplitude=0.98)
    y = process(decimator(), x)
    interior = np.asarray(y[OUT_RATE // 50 : -OUT_RATE // 50], dtype=np.float64)  # drop 20 ms
    assert interior.size > 1000, f"only {interior.size} output samples to measure"

    level_db = 20.0 * np.log10(max(rms(interior), 1e-30) / rms(x))
    assert level_db <= -40.0, (
        f"a full-scale 18 kHz tone comes out of the 48 → 24 kHz stage at {level_db:.1f} dB "
        f"(criterion 6 wants ≤ −40 dB). 0 dB here means the tone was subsampled, not filtered: "
        f"it is now a 6 kHz tone in the STT's speech band."
    )
    alias_db = 10.0 * np.log10(
        max(band_power(interior, OUT_RATE, 5_000.0, 7_000.0), 1e-300)
        / max(band_power(x, IN_RATE, 17_000.0, 19_000.0), 1e-300)
    )
    assert alias_db <= -40.0, (
        f"{alias_db:.1f} dB of the 18 kHz tone's energy reappears in the 5-7 kHz band of the "
        f"24 kHz output — that is the alias, and the criterion allows ≤ −40 dB"
    )


@pytest.mark.parametrize("chunk", [1, 7, BLOCK, 960, 1920, 48_000])
async def test_output_length_is_exactly_half(chunk: int) -> None:
    """Criterion 6: exactly half the input, for every chunking — 1 s of 48 kHz is 24000 samples.

    One decimator across the whole stream, cut into ``chunk``-sample pieces: that is how capture
    feeds it (10 ms blocks), and it is where a filter that drops its tail loses a sample per cut.
    """
    x = sine_48k(1000.0, 1000.0)
    assert x.size == 48_000
    total = stream(x, chunk).size
    assert total == x.size // 2, (
        f"chunk={chunk}: {total} output samples for {x.size} input samples (want {x.size // 2}); "
        f"a decimator that drops its filter tail loses samples at every chunk boundary"
    )


@settings(max_examples=200, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=0, max_value=2000), min_size=1, max_size=12),
)
async def test_output_length_is_chunking_invariant(lengths: list[int]) -> None:
    """Criterion 6 as a property: how the stream is cut never changes how much comes out."""
    n = int(sum(lengths))
    x = sine_48k(1000.0, 1000.0)[:n]
    r = decimator()
    total, at = 0, 0
    for length in lengths:
        total += process(r, x[at : at + length]).size
        at += length
    assert abs(2 * total - n) <= 1, (
        f"{total} output samples for {n} input samples cut as {lengths}: the rate conversion is "
        f"exactly 2:1 (± the one sample an odd length leaves)"
    )
    assert total == process(decimator(), x).size, (
        f"cut as {lengths} the stream gives {total} samples, fed whole it gives "
        f"{process(decimator(), x).size}: the output length depends on the chunking"
    )


@pytest.mark.parametrize("chunk", [1, 7, BLOCK, 1920])
async def test_chunked_is_bit_identical_to_whole(chunk: int) -> None:
    """The classic defect: filter state not carried, so the stream clicks every block."""
    x = sweep_48k(200.0, 20_000.0, 500.0, amplitude=0.9)
    whole = process(decimator(), x)
    streamed = stream(x, chunk)
    assert streamed.shape == whole.shape, f"chunk={chunk}: {streamed.shape} vs {whole.shape}"
    assert np.array_equal(streamed, whole), (
        f"chunk={chunk}: feeding the signal in pieces differs from feeding it whole; largest "
        f"difference {np.max(np.abs(streamed - whole)):.3e} at sample "
        f"{int(np.argmax(np.abs(streamed - whole)))} — the filter state is not carried across "
        f"chunks (criterion 6: 10 ms blocks must be bit-identical to the whole stream)"
    )


async def test_reset_starts_a_new_stream() -> None:
    """``reset()`` must clear the tail, or one run starts with the previous run's audio."""
    x = sine_48k(1000.0, 200.0)
    r = decimator()
    first = process(r, x)
    attr_or_fail(r, "reset", WHY)()
    again = process(r, x)
    assert np.array_equal(first, again), "after reset() the same input must give the same output"


@pytest.mark.parametrize("chunk", [None, BLOCK, 1920])
async def test_1khz_sine_snr_and_range(chunk: int | None) -> None:
    """Criterion 6: SNR ≥ 40 dB at 24 kHz, output in [−1, 1], the gain unchanged."""
    x = sine_48k(1000.0, 500.0, amplitude=0.9)
    y = process(decimator(), x) if chunk is None else stream(x, chunk)
    interior = np.asarray(y[OUT_RATE // 50 : -OUT_RATE // 50], dtype=np.float64)

    signal = band_power(interior, OUT_RATE, 940.0, 1060.0)
    total = band_power(interior, OUT_RATE, 0.0, OUT_RATE / 2.0)
    snr = 10.0 * np.log10(signal / max(total - signal, 1e-300))
    assert snr >= 40.0, (
        f"chunk={chunk}: the 1 kHz tone came back at {snr:.1f} dB SNR; the passband is supposed "
        f"to leave it alone (criterion 6)"
    )
    assert float(np.max(np.abs(y))) <= 1.0, f"output peaks at {np.max(np.abs(y)):.3f}: clipping"
    assert 0.8 <= float(np.max(np.abs(interior))) <= 0.95, (
        f"chunk={chunk}: a 0.9 input comes back peaking at {np.max(np.abs(interior)):.3f} — the "
        f"passband gain is not 1"
    )


async def test_group_delay_is_constant_and_at_most_one_block() -> None:
    """Criterion 6's last clause: the delay is constant, ≤ one block, and signal-independent.

    Measured against the closed-form 24 kHz sweep with a float64 cross-correlation; a linear
    sweep is aperiodic, so the lag is unambiguous (a sine's would only be known modulo a period).
    """
    lags = []
    for f0, f1 in ((200.0, 6000.0), (400.0, 3000.0)):
        x = sweep_48k(f0, f1, 400.0, amplitude=0.8)
        y = np.asarray(process(decimator(), x), dtype=np.float64)
        reference = sweep_24k(f0, f1, 400.0, amplitude=0.8)
        n = min(y.size, reference.size)
        lags.append(fft_best_lag(reference[:n], y[:n]))

    limit = CAPTURE_BLOCK_MS * OUT_RATE // 1000  # one block: 240 samples at 24 kHz
    assert all(0 <= lag <= limit for lag in lags), (
        f"group delay {lags} samples at 24 kHz; criterion 6 allows at most one block "
        f"({limit} samples = {CAPTURE_BLOCK_MS} ms), compensated in t_ms or documented"
    )
    assert abs(lags[0] - lags[1]) <= 1, (
        f"the delay depends on the signal ({lags[0]} vs {lags[1]} samples): a non-linear-phase "
        f"filter smears the timestamps the whole capture path is built on"
    )


async def test_dtype_and_shape_are_the_stt_format() -> None:
    y = process(decimator(), sine_48k(440.0, 100.0))
    assert y.dtype == np.float32, f"moshi-server wants f32 (STT_INPUT_FORMAT), got {y.dtype}"
    assert y.ndim == 1, f"mono, got shape {y.shape}"


async def test_silence_in_silence_out() -> None:
    """A trivially checkable fixed point, and the baseline the gate's ≤ −60 dBFS relies on."""
    y = process(decimator(), np.zeros(4800, dtype=np.float32))
    assert y.size == 2400
    assert float(np.max(np.abs(y))) == 0.0, "the decimator adds energy to digital silence"
