"""24 kHz → 48 kHz for the playback path (T-0006, criterion 1). numpy only, no scipy.

TTS renders 24 kHz mono s16le (``TTS_OUTPUT_FORMAT``), the Conductor's device wants 48 kHz f32
(``CAPTURE_FORMAT``), and the audio arrives in chunks — so the filter state has to be carried
across them or the stream clicks at every chunk boundary.

**The filter.** Rate doubling is zero-stuffing followed by an anti-imaging lowpass; written in
polyphase form, the two output phases are

    y[2n]   = Σ_j h[2j]   · x[n−j]      (the interpolated sample)
    y[2n+1] = Σ_j h[2j+1] · x[n−j]

with ``h`` a linear-phase windowed sinc at cutoff Fs/4 = 12 kHz. Cutoff *exactly* at Fs/4 makes
``h`` a halfband filter: every tap at an even distance from the centre is zero except the centre
itself, so the second branch collapses to a pure delay and the original 24 kHz samples survive
the conversion unchanged. A Kaiser window (β = 7.86) puts the stopband at about −80 dB, which is
where the image of any baseband tone lands (a 1 kHz tone images to 23 kHz).

**Chunk invariance is a correctness property, not an optimisation.** Every output sample is
accumulated tap by tap with elementwise numpy operations in a fixed order, so the arithmetic is
identical however the input is split. Reductions (``np.convolve``, ``np.dot``, BLAS) are *not*
used on purpose: their summation order can depend on length and on memory alignment, which would
make the streamed result differ from the one-block result in the last bits.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from local_astra.contracts.speech import CAPTURE_FORMAT, STT_INPUT_FORMAT, TTS_OUTPUT_FORMAT

IN_RATE = TTS_OUTPUT_FORMAT.sample_rate
OUT_RATE = CAPTURE_FORMAT.sample_rate
#: Odd, with ``(TAPS − 1) / 2`` odd as a halfband filter requires. 95 taps at 48 kHz give a
#: transition band of roughly 12 ± 1.3 kHz and ~1 ms of group delay (47 output samples).
TAPS = 95
#: Kaiser β for ~80 dB of stopband attenuation (0.1102 · (A − 8.7), A = 80).
KAISER_BETA = 7.857
#: s16 full scale: −32768 maps to −1.0 and +32767 to +0.99997, with no asymmetric rescaling.
S16_FULL_SCALE = 32768.0


def _design() -> tuple[npt.NDArray[np.float64], int]:
    """Halfband taps as its two polyphase branches: (interpolating branch, delay of the other)."""
    distance = np.arange(TAPS, dtype=np.float64) - (TAPS - 1) / 2.0
    taps = np.sinc(distance / 2.0) * np.kaiser(TAPS, KAISER_BETA)
    interpolating = taps[0::2]
    passthrough = taps[1::2]
    delay = int(np.argmax(np.abs(passthrough)))
    # The halfband property this module's sample-exactness rests on: the odd branch is one tap.
    if not np.allclose(np.delete(passthrough, delay), 0.0, atol=1e-12):
        raise AssertionError("the odd polyphase branch is not a pure delay: taps are not halfband")
    return interpolating / interpolating.sum(), delay


_INTERPOLATING, _DELAY = _design()
#: Input samples of history the next chunk needs (both branches read ``x[n − j]``).
_HISTORY = _INTERPOLATING.size - 1


def s16le_to_f32(data: bytes) -> npt.NDArray[np.float32]:
    """TTS wire format (24 kHz mono s16le) to f32 in [−1, 1]."""
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return np.asarray(pcm / np.float32(S16_FULL_SCALE), dtype=np.float32)


class Resampler24kTo48k:
    """Streaming 2× upsampler. ``process`` returns exactly ``2 · len(x)`` samples, every time."""

    def __init__(self) -> None:
        self._state: npt.NDArray[np.float64] = np.zeros(_HISTORY, dtype=np.float64)

    def reset(self) -> None:
        """Forget the tail of the previous stream (a new run must not start inside the old one)."""
        self._state = np.zeros(_HISTORY, dtype=np.float64)

    def process(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        samples = np.asarray(x, dtype=np.float64).reshape(-1)
        count = samples.size
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        buffer = np.concatenate([self._state, samples])
        interpolated = np.zeros(count, dtype=np.float64)
        for j in range(_INTERPOLATING.size):
            interpolated += _INTERPOLATING[j] * buffer[_HISTORY - j : _HISTORY - j + count]
        out = np.empty(2 * count, dtype=np.float64)
        out[0::2] = interpolated
        out[1::2] = buffer[_HISTORY - _DELAY : _HISTORY - _DELAY + count]
        self._state = buffer[count:]
        return np.asarray(out, dtype=np.float32)


# --- 48 kHz -> 24 kHz, the capture direction (T-0021 criterion 6) ---------------------------

CAPTURE_RATE = CAPTURE_FORMAT.sample_rate
STT_RATE = STT_INPUT_FORMAT.sample_rate
#: Odd, for linear phase: the group delay is exactly ``(DECIM_TAPS - 1) / 2`` input samples.
DECIM_TAPS = 95
#: Cutoff of the anti-aliasing lowpass, in Hz. The task caps it at 0.45 x 24 kHz; a Kaiser
#: window with beta = 7.857 over 95 taps puts the transition band at roughly 10.8 +/- 1.3 kHz,
#: so everything above 12.1 kHz -- and in particular the 18 kHz tone criterion 6 names -- is
#: down by about 80 dB, where plain ``x[::2]`` would fold it to 6 kHz at unity gain.
DECIM_CUTOFF_HZ = 10_800.0


def _design_decimator() -> npt.NDArray[np.float64]:
    """Windowed-sinc lowpass, normalised to unity DC gain (the passband must not move levels)."""
    distance = np.arange(DECIM_TAPS, dtype=np.float64) - (DECIM_TAPS - 1) / 2.0
    taps = np.sinc(2.0 * DECIM_CUTOFF_HZ / CAPTURE_RATE * distance) * np.kaiser(
        DECIM_TAPS, KAISER_BETA
    )
    return np.asarray(taps / taps.sum(), dtype=np.float64)


_DECIM_TAPS = _design_decimator()
#: Input samples of history the next chunk needs (``y[i] = sum_j h[j] x[i - j]``).
_DECIM_HISTORY = DECIM_TAPS - 1
#: Group delay of the stage, in **output** samples: 47 input samples at 48 kHz = 23.5 at 24 kHz.
#: Constant (linear phase), under one 10 ms block, and deliberately *not* compensated in the
#: capture timestamps -- ``SttFrame.t_ms`` is the capture time of the frame's first input
#: sample, and criterion 6 allows the delay to be documented instead (it is the same for every
#: frame, so it never moves one transcript segment relative to another).
DECIM_GROUP_DELAY_SAMPLES = (DECIM_TAPS - 1) / 4.0


class Resampler48kTo24k:
    """Streaming 2x decimator for the capture path: 48 kHz f32 in, 24 kHz f32 out.

    Output sample ``k`` is the filtered signal at **global** input index ``2k``, so the output
    length is ``ceil(n / 2)`` over the whole stream however it is chunked, and every output
    sample is accumulated tap by tap in a fixed order: feeding 10 ms blocks is bit-identical to
    feeding the stream whole (criterion 6). Reductions (``np.convolve``, ``np.dot``) are avoided
    for the same reason as in ``Resampler24kTo48k``: their summation order is not ours to fix.
    """

    def __init__(self) -> None:
        self._state: npt.NDArray[np.float64] = np.zeros(_DECIM_HISTORY, dtype=np.float64)
        self._parity = 0

    def reset(self) -> None:
        """Forget the previous stream: a new run must not start inside the old one's tail."""
        self._state = np.zeros(_DECIM_HISTORY, dtype=np.float64)
        self._parity = 0

    def process(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        samples = np.asarray(x, dtype=np.float64).reshape(-1)
        count = samples.size
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        buffer = np.concatenate([self._state, samples])
        # ``first`` is the offset, inside this chunk, of the first input sample that sits on the
        # decimation grid; ``self._parity`` carries that grid across chunk boundaries, which is
        # what makes an odd-length chunk cost nothing.
        first = self._parity
        out_count = (count - first + 1) // 2 if first < count else 0
        out = np.zeros(out_count, dtype=np.float64)
        if out_count:
            base = _DECIM_HISTORY + first
            for j in range(DECIM_TAPS):
                start = base - j
                out += _DECIM_TAPS[j] * buffer[start : start + 2 * out_count : 2]
        self._parity = (first + 2 * out_count - count) % 2
        self._state = buffer[count:]
        return np.asarray(out, dtype=np.float32)
