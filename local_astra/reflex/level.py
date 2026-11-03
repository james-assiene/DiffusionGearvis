"""Input level and running room-noise floor for the listening meter (T-0047 group A0).

The page needs something to look at between "the user spoke" and "the answer arrived", and the
only honest thing to show is what the microphone is actually doing. This module turns the block
the capture pump already materialises for ``stt.feed`` into a number the page can draw, plus a
verdict ("is this louder than the room?") that is **relative to the room** and not to an absolute
threshold. ``_listen.py`` recorded why: measured in this room ambient noise peaks around 0.116,
so the old absolute 0.02 floor meant the room refreshed the timer by itself.

Pure, in the sense `.claude/rules/conductor.md` gives the reflex lane: no clock, no I/O, no
logging, no asyncio, no sockets. Frame time arrives as ``observe``'s ``t_ms`` argument — the
frame's own capture time (``AudioFrame.t_ms``), never a wall clock — exactly as the turn machine
takes time as a ``Tick``. **Precondition** (ruling A10): that time must be non-decreasing over
one :class:`LevelMeter`, so a re-opened capture stream — whose clock restarts at zero — needs a
new meter rather than the old one. Nothing here allocates in proportion to the number of frames
seen: the window is a fixed-size ring of at most :data:`WINDOW_FRAMES` floats.

This module decides nothing about the turn: it neither gates nor reframes the audio, and the
caller (T-0047 group A) publishes what it returns without consulting the machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

#: What :func:`rms_dbfs` reports for digital silence, and for a block a driver glitch filled with
#: ``nan``/``inf``. A *named* floor rather than ``-inf``: the value goes on the wire and through
#: ``float`` arithmetic on the page, where ``-inf`` and ``nan`` are not representable in JSON and
#: a single one of them would poison the noise floor for a whole window. Chosen (a round number
#: far below any real 24 kHz capture, whose dither floor sits near -80 dBFS), not measured; the
#: run that would measure it is a silent-room capture with the production gain staging.
SILENCE_DBFS = -100.0

#: The capture pump's frame, in ms. Not a choice of this module: ``audio_capture`` owns the fact
#: (``STT_BLOCK_SAMPLES = 1920`` at 24 kHz), and ``test_listen_level.py`` fails if the two ever
#: disagree. Kept as a local constant so the reflex lane imports nothing from the conductor.
STT_FRAME_MS = 80

#: How much *frame* time the noise floor looks back over. Chosen, not measured: 3 s is long
#: enough that a sentence cannot raise the floor to its own level (a spoken clause is well under
#: 3 s of continuous energy) and short enough that a fan or an air conditioner starting is
#: followed within one window. What would measure it: X2/Q5 (docs/02 §10, docs/03 §3/B15), which
#: records room noise and end-of-turn behaviour on real captures and can read the shortest window
#: that does not track speech.
NOISE_FLOOR_WINDOW_MS = 3000

#: The window, in frames: ceil(3000 / 80) = 38. Derived, never tuned independently of the two
#: constants above.
WINDOW_FRAMES = math.ceil(NOISE_FLOOR_WINDOW_MS / STT_FRAME_MS)

#: Which quantile of the window is called "the room". **Chosen, not measured**: low enough that a
#: window in which the user speaks for most of the 3 s still reports the quiet fifth as the room,
#: high enough not to be the window's minimum (a single dropout frame must not define the floor).
#: What would measure it: X2/Q5 (docs/03 §3/B15, the "energy relative to noise floor" reflex
#: feature) over recorded sessions, or the first live spoken session with a real microphone —
#: docs/tasks/T-0047.md ambiguity 3 says to record that value there, not to tune it quietly.
#: Regime it is chosen for: 24 kHz mono capture, 80 ms frames, a 3 s window, one speaker at a
#: desk microphone.
NOISE_FLOOR_PERCENTILE = 0.2

#: How far above the room a frame must be to count as voice. **Chosen, not measured**: 9 dB is
#: roughly a tripling of amplitude, which speech clears easily and room noise does not. Same
#: regime as above (24 kHz, 80 ms frames, 3 s window, desk microphone, a single speaker), and the
#: same thing would measure it: X2/Q5's recorded sessions, scored as voiced-frame recall against
#: the transcript's word times versus false-voiced rate in silence. The first live session is the
#: first evidence (docs/tasks/T-0047.md ambiguity 3).
VOICED_MARGIN_DB = 9.0

#: How many dB above the room read as a full bar. **Chosen, not measured**: 30 dB puts a quiet
#: room at 0, conversational speech near the middle and a shout at the top on the same regime as
#: the two constants above; it is a display mapping, so what would measure it is not an
#: experiment but the human looking at the bar during the first live spoken session and saying it
#: is steppy or flat (docs/tasks/T-0047.md ambiguity 3 and ambiguity 1).
LEVEL_SPAN_DB = 30.0

#: The floor of *frame* time between two emissions. 100 ms on the 80 ms grid is every second
#: frame, i.e. 6.25 Hz, under the task's stated "≤ 10 Hz" bound; chosen for that exact relation to
#: the frame grid rather than for a measured perceptual threshold. Raising it is a spec change
#: (docs/tasks/T-0047.md ambiguity 1), never a page-side interpolation dressed up as data.
PUBLISH_INTERVAL_MS = 100


def rms_dbfs(samples: npt.NDArray[np.floating[Any]]) -> float:
    """The block's RMS in dBFS: ``10*log10(mean(x**2))``, total and finite.

    Total by construction, because this runs inside the capture pump's tap and a driver glitch
    must not raise there (T-0047 criterion 1): digital silence, an empty block and any block
    holding a ``nan`` or an ``inf`` all return :data:`SILENCE_DBFS`, and nothing below that floor
    is ever reported. The mean square is accumulated in float64 so the returned dB value does not
    carry float32 summation error over 1920 samples.
    """
    block = np.asarray(samples)
    if block.size == 0:
        return SILENCE_DBFS
    mean_square = float(np.mean(np.square(block, dtype=np.float64)))
    if not math.isfinite(mean_square) or mean_square <= 0.0:
        return SILENCE_DBFS
    return max(10.0 * math.log10(mean_square), SILENCE_DBFS)


@dataclass(frozen=True, slots=True)
class ListenLevel:
    """One reading of the microphone, as the page and a debugger both need it.

    ``level`` is the relative bar in [0, 1]; ``rms_dbfs`` and ``noise_floor_dbfs`` are the
    absolute numbers behind it (a bar alone cannot be debugged); ``voiced`` is the verdict;
    ``t_ms`` is the *emitting frame's* capture time (ruling A3 — the reading is "what the room is
    doing now", so stamping it with the older frame that held the peak would put the meter behind
    the audio it reports, and the page extrapolates from this number).
    """

    level: float
    rms_dbfs: float
    noise_floor_dbfs: float
    voiced: bool
    t_ms: float


class LevelMeter:
    """Running noise floor, voiced verdict and throttled, peak-held emission. Pure and bounded.

    ``observe`` is called once per captured frame and returns a :class:`ListenLevel` at most once
    per :data:`PUBLISH_INTERVAL_MS` of frame time, ``None`` otherwise. The first frame emits
    (ruling A2): a meter that waited one interval would leave the page blank for the first two
    frames of every session. The value it reports is the **maximum** frame RMS since the last
    emission, so a single loud 80 ms frame falling between two emissions is never invisible.

    State is three attributes and a list that stops growing at :data:`WINDOW_FRAMES`; the window
    is a ring, so 100 000 frames cost what 38 do.
    """

    __slots__ = ("_at", "_last_emit_ms", "_peak_dbfs", "_window")

    def __init__(self) -> None:
        #: The last WINDOW_FRAMES frame RMS values, in ring order (order is irrelevant: the only
        #: reader sorts them). Grows to WINDOW_FRAMES once and is overwritten in place after.
        self._window: list[float] = []
        self._at = 0
        self._peak_dbfs: float | None = None
        self._last_emit_ms: float | None = None

    def observe(self, rms_db: float, t_ms: float) -> ListenLevel | None:
        """Record one frame's RMS at its own capture time; emit if the interval has elapsed.

        ``t_ms`` must be non-decreasing across the calls made to one meter (ruling A10). The
        throttle below compares it against the last emission's frame time, so a stream that
        restarts its clock at zero — which is what a re-opened capture stream does — would stop
        the meter emitting until its new time passed the old one. A re-opened stream therefore
        gets a **new** :class:`LevelMeter`; there is deliberately no branch here that notices a
        clock going backwards, because that would be an untested reset in the one module the
        reflex lane keeps pure.
        """
        # Ruling A7: a frame *at* SILENCE_DBFS is not evidence about the room. ``audio_capture``
        # substitutes exact zeros for self-speech-gated blocks and for xrun holes and the frame
        # carries no flag saying so, so 8 of them (640 ms) would drag a P20 over 38 entries down
        # to -100 and make an unchanged room read as a voiced full bar. Digital silence is not
        # reachable from a real microphone (a 24 kHz capture's dither floor sits near -80), so
        # nothing real is discarded. The test is the constant itself, not a band of quiet values:
        # a genuine -98 dBFS room must still set the floor at -98.
        if rms_db > SILENCE_DBFS:
            window = self._window
            if len(window) < WINDOW_FRAMES:
                window.append(rms_db)
            else:
                window[self._at] = rms_db
                self._at = (self._at + 1) % WINDOW_FRAMES

        peak = rms_db if self._peak_dbfs is None else max(self._peak_dbfs, rms_db)
        last = self._last_emit_ms
        if last is not None and t_ms - last < PUBLISH_INTERVAL_MS:
            self._peak_dbfs = peak
            return None

        self._peak_dbfs = None
        self._last_emit_ms = t_ms
        floor = self._noise_floor()
        return ListenLevel(
            level=min(1.0, max(0.0, (peak - floor) / LEVEL_SPAN_DB)),
            rms_dbfs=peak,
            noise_floor_dbfs=floor,
            voiced=peak > floor + VOICED_MARGIN_DB,
            t_ms=t_ms,
        )

    def _noise_floor(self) -> float:
        """The NOISE_FLOOR_PERCENTILE quantile of the window, by nearest rank.

        Nearest rank because of what it costs: sorting 38 floats is ~5 µs against ``np.quantile``
        on the same window at ~26 µs (docs/tasks/T-0047.md ruling A6), and criterion 6's 50 µs
        ceiling has less headroom than it looks. Ruling A4 leaves the convention to the
        implementer and the tests' fixtures agree under every common one; the reviewer measured
        linear interpolation against it and found ~1e-16 of difference with zero voiced flips, so
        the choice is about cost and nothing else.

        The window holds only frames that were not at :data:`SILENCE_DBFS` (ruling A7, in
        ``observe``), so it can be empty — a stream that is nothing but substituted silence, or a
        genuinely dead microphone. The floor is then :data:`SILENCE_DBFS`, which reads as an empty
        bar and no voice rather than raising inside the capture pump's tap.
        """
        ordered = sorted(self._window)
        if not ordered:
            return SILENCE_DBFS
        rank = math.ceil(NOISE_FLOOR_PERCENTILE * len(ordered)) - 1
        return ordered[min(max(rank, 0), len(ordered) - 1)]
