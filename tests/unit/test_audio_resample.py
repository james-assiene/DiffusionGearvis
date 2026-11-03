"""The 24 kHz → 48 kHz stage: T-0006 owned file ``local_astra/conductor/audio_resample.py``.

TTS produces 24 kHz mono s16le (``TTS_OUTPUT_FORMAT``), the device wants 48 kHz f32 (
``CAPTURE_FORMAT``), and the conversion happens on a stream that arrives in chunks — so the
filter state has to be carried across them. Oracles here are numpy float64 and closed-form
signals, never the implementation.

Seam these tests fix (the task names the module but not its API, reported as such):
``Resampler24kTo48k()`` with ``process(x_f32) -> y_f32`` and ``reset()``, plus a module-level
``s16le_to_f32(data: bytes) -> np.ndarray``.
"""

from __future__ import annotations

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_FORMAT, TTS_OUTPUT_FORMAT
from tests.fakes.audio_device import attr_or_fail, chirp_fixture, import_or_fail

WHY = "T-0006: the 24 kHz → 48 kHz stage (owned file local_astra/conductor/audio_resample.py)"
IN_RATE = TTS_OUTPUT_FORMAT.sample_rate  # 24000
OUT_RATE = CAPTURE_FORMAT.sample_rate  # 48000


def resampler() -> object:
    module = import_or_fail("local_astra.conductor.audio_resample", WHY)
    return attr_or_fail(module, "Resampler24kTo48k", WHY)()


def process(r: object, x: np.ndarray) -> np.ndarray:
    out = attr_or_fail(r, "process", WHY)(np.asarray(x, dtype=np.float32))
    return np.asarray(out)


def sine_24k(freq_hz: float, ms: float, amplitude: float = 0.5) -> np.ndarray:
    n = round(ms * IN_RATE / 1000.0)
    t = np.arange(n, dtype=np.float64) / IN_RATE
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def snr_db_at(signal: np.ndarray, freq_hz: float, rate: int, half_width_hz: float = 60.0) -> float:
    """Power at ``freq_hz`` over power everywhere else, float64 FFT of a Hann-windowed frame.

    Delay-insensitive by construction, so a linear-phase filter's group delay cannot flatter or
    hurt the number; images (a doubler that only zero-stuffs puts one at 23 kHz) land in the
    "everywhere else" term.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x * np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    band = np.abs(freqs - freq_hz) <= half_width_hz
    signal_power = float(spectrum[band].sum())
    noise_power = float(spectrum[~band].sum())
    return 10.0 * np.log10(signal_power / max(noise_power, 1e-300))


@pytest.mark.parametrize("chunk", [1, 7, 240, 480, 1920, 24000])
async def test_output_is_exactly_twice_the_input_for_every_chunking(chunk: int) -> None:
    """The rate conversion is exactly 2×: no chunk may swallow or invent a sample."""
    x = sine_24k(1000.0, 1000.0)
    r = resampler()
    total = sum(process(r, x[i : i + chunk]).size for i in range(0, x.size, chunk))
    assert total == 2 * x.size, (
        f"chunk={chunk}: {total} output samples for {x.size} input samples (want {2 * x.size}); "
        f"a resampler that drops its filter tail loses samples at every chunk boundary"
    )


async def test_chunked_is_bit_identical_to_one_block() -> None:
    """The classic defect: filter state not carried across chunks, so it clicks every 80 ms."""
    x24, rate = chirp_fixture()
    assert rate == IN_RATE
    x = x24.astype(np.float32)

    whole = process(resampler(), x)
    r = resampler()
    chunk = 80 * IN_RATE // 1000  # the TTS chunk size in tests/fakes/tts.py
    pieces = [process(r, x[i : i + chunk]) for i in range(0, x.size, chunk)]
    streamed = np.concatenate(pieces)

    assert streamed.shape == whole.shape
    assert np.array_equal(streamed, whole), (
        "feeding the same signal in 80 ms chunks differs from feeding it in one block: the "
        f"largest difference is {np.max(np.abs(streamed - whole)):.3e} at sample "
        f"{int(np.argmax(np.abs(streamed - whole)))} — the filter state is not carried"
    )


async def test_reset_starts_a_new_stream() -> None:
    """``reset()`` must clear the tail, or the next utterance begins with the previous one's."""
    x = sine_24k(1000.0, 200.0)
    r = resampler()
    first = process(r, x)
    attr_or_fail(r, "reset", WHY)()
    again = process(r, x)
    assert np.array_equal(first, again), "after reset() the same input must give the same output"


@pytest.mark.parametrize("chunk", [None, 240, 1920])
async def test_a_1khz_sine_survives_the_conversion(chunk: int | None) -> None:
    """Criterion: SNR ≥ 40 dB, no clipping, output in [−1, 1] — fed whole and in chunks."""
    x = sine_24k(1000.0, 500.0, amplitude=0.9)
    r = resampler()
    if chunk is None:
        y = process(r, x)
    else:
        y = np.concatenate([process(r, x[i : i + chunk]) for i in range(0, x.size, chunk)])

    interior = y[OUT_RATE // 100 : -OUT_RATE // 100]  # drop 10 ms of filter warm-up at each end
    snr = snr_db_at(interior, 1000.0, OUT_RATE)
    assert snr >= 40.0, (
        f"chunk={chunk}: 1 kHz sine came back at {snr:.1f} dB SNR at 48 kHz; a zero-stuffing "
        f"doubler with no filter leaves an image at 23 kHz and scores about 0 dB"
    )
    assert np.max(np.abs(y)) <= 1.0, f"output peaks at {np.max(np.abs(y)):.3f}: clipping"
    assert np.max(np.abs(y)) >= 0.8, (
        f"output peaks at {np.max(np.abs(y)):.3f} for a 0.9 input: the gain is wrong"
    )


async def test_the_mirror_image_above_13khz_is_filtered_out() -> None:
    """Zero-stuffing alone mirrors 1 kHz to 23 kHz; the anti-imaging filter must remove it."""
    x = sine_24k(1000.0, 500.0, amplitude=0.9)
    y = np.asarray(process(resampler(), x), dtype=np.float64)
    y = y[OUT_RATE // 100 : -OUT_RATE // 100]
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    freqs = np.fft.rfftfreq(y.size, 1.0 / OUT_RATE)
    high = float(spectrum[freqs > 13_000.0].sum())
    total = float(spectrum.sum())
    ratio_db = 10.0 * np.log10(max(high, 1e-300) / total)
    assert ratio_db <= -40.0, (
        f"{ratio_db:.1f} dB of the energy sits above 13 kHz — the image of the 1 kHz tone at "
        f"23 kHz survived the upsampling"
    )


async def test_dtype_and_shape_are_the_device_format() -> None:
    x = sine_24k(440.0, 100.0)
    y = process(resampler(), x)
    assert y.dtype == np.float32, f"the device wants f32 (CAPTURE_FORMAT), got {y.dtype}"
    assert y.ndim == 1, f"mono, got shape {y.shape}"


async def test_s16le_to_f32_uses_the_full_scale_convention() -> None:
    """TTS is s16le (``TTS_OUTPUT_FORMAT``); hand-checked values, no numpy round-tripping."""
    module = import_or_fail("local_astra.conductor.audio_resample", WHY)
    convert = attr_or_fail(module, "s16le_to_f32", WHY)
    raw = b"\x00\x00" + b"\x00\x40" + b"\x00\xc0" + b"\x00\x80" + b"\xff\x7f"
    out = np.asarray(convert(raw))
    assert out.dtype == np.float32
    assert out.size == 5, f"5 samples in 10 bytes of s16le, got {out.size}"
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5, abs=1e-4), "16384 / 32768 = 0.5"
    assert out[2] == pytest.approx(-0.5, abs=1e-4), "-16384 / 32768 = -0.5"
    assert out[3] == pytest.approx(-1.0, abs=1e-4), "-32768 is full scale negative"
    assert out[4] == pytest.approx(1.0, abs=1e-4), "32767 is full scale positive"
    assert np.max(np.abs(out)) <= 1.0
