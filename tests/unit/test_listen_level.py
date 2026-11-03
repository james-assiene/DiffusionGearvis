"""T-0047 group A0 — the pure level / noise-floor helper (``local_astra/reflex/level.py``).

Criteria 1-5 of ``docs/tasks/T-0047.md``. Nothing here has a clock, a socket or a device: the
meter is driven by *frame* time handed to ``observe``, which is the whole point of criterion 2
("the frame's own capture time, never a wall clock").

**Where the expected numbers come from.** Every one is hand-computed in the row that uses it and
written as a literal; no row calls the module to find out what it should have returned, and no
row re-implements ``rms_dbfs`` or the meter. The dB anchors are the decibel definition
(``10*log10(mean(x**2))``) evaluated by hand; the P20 rows are read off a sorted list designed so
that every common quantile convention lands on the same value (see the row's docstring).

Regime for every row below: 24 kHz mono float32, the pump's 80 ms frame
(``audio_capture.STT_BLOCK_SAMPLES = 1920``), window 3000 ms = 38 frames, n = 1 per deterministic
row, Hypothesis rows ``derandomize=True`` (stable under ``--hypothesis-seed=0``).
"""

from __future__ import annotations

import ast
import sys
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.conductor.audio_capture import STT_BLOCK_MS, STT_BLOCK_SAMPLES
from tests.unit import _level_rig as rig

#: The 80 ms grid every deterministic row below is driven on.
DT = 80.0

#: Criterion 1's hand-computed anchors: 10*log10(mean(x**2)) for a block of 1920 float32 samples.
#: all-0.5 -> 20*log10(0.5) = -6.0205999...; all-0.1 -> exactly -20; DC 1.0 -> 0; a full-scale
#: sine has mean square exactly 1/2 over a whole number of periods -> 20*log10(1/sqrt(2)).
ANCHORS = (
    ("constant 0.5", -6.020600),
    ("constant 0.1", -20.0),
    ("DC 1.0", 0.0),
    ("full-scale sine, 48 periods in 1920 samples", -3.010300),
)


def block(kind: str) -> np.ndarray:
    n = STT_BLOCK_SAMPLES
    if kind == "constant 0.5":
        return np.full(n, 0.5, dtype=np.float32)
    if kind == "constant 0.1":
        return np.full(n, 0.1, dtype=np.float32)
    if kind == "DC 1.0":
        return np.ones(n, dtype=np.float32)
    # 48 periods of 1920 samples = 40 samples per period = 600 Hz at 24 kHz: the sum of
    # cos(4*pi*k*n/N) over whole periods is 0, so mean(sin**2) is exactly 1/2.
    return np.sin(2.0 * np.pi * 48.0 * np.arange(n) / n).astype(np.float32)


# --- criterion 1: rms_dbfs ----------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "expected_db"), ANCHORS, ids=[a[0] for a in ANCHORS])
def test_rms_dbfs_hand_computed_anchors(kind: str, expected_db: float) -> None:
    """Criterion 1: the four anchors, to 1e-6, on the production dtype (float32, 1920 samples)."""
    assert rig.rms_dbfs(block(kind)) == pytest.approx(expected_db, abs=1e-6)


def test_rms_dbfs_digital_silence_is_the_named_constant() -> None:
    """Criterion 1: an all-zero block is SILENCE_DBFS exactly - never -inf, never nan."""
    silence = rig.const("SILENCE_DBFS", "criterion 1")
    assert silence == -100.0, "criterion 1 names SILENCE_DBFS = -100.0"
    value = rig.rms_dbfs(np.zeros(STT_BLOCK_SAMPLES, dtype=np.float32))
    assert value == silence
    assert np.isfinite(value)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("one nan", np.nan),
        ("one +inf", np.inf),
        ("one -inf", -np.inf),
    ],
)
def test_rms_dbfs_nonfinite_block_is_silence_without_raising(name: str, bad: float) -> None:
    """Criterion 1: total. A driver glitch must not raise inside the capture pump's tap."""
    samples = np.full(STT_BLOCK_SAMPLES, 0.25, dtype=np.float32)
    samples[7] = bad
    value = rig.rms_dbfs(samples)
    assert value == rig.const("SILENCE_DBFS", "criterion 1")
    assert np.isfinite(value)


def test_rms_dbfs_all_nan_block_is_silence_without_raising() -> None:
    """Criterion 1, the whole-block case (a muted device that hands over nan)."""
    samples = np.full(STT_BLOCK_SAMPLES, np.nan, dtype=np.float32)
    assert rig.rms_dbfs(samples) == rig.const("SILENCE_DBFS", "criterion 1")


# --- criteria 1/3/4: the constants are the spec's -----------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SILENCE_DBFS", -100.0),
        ("NOISE_FLOOR_WINDOW_MS", 3000),
        ("STT_FRAME_MS", 80),
        ("NOISE_FLOOR_PERCENTILE", 0.2),
        ("VOICED_MARGIN_DB", 9.0),
        ("LEVEL_SPAN_DB", 30.0),
        ("PUBLISH_INTERVAL_MS", 100),
    ],
)
def test_constants_are_the_values_the_task_names(name: str, expected: float) -> None:
    """Criteria 1, 3 and 4: every number is a module-level name, and it is the spec's number."""
    assert rig.const(name, "criteria 1/3/4") == expected


def test_frame_grid_agrees_with_the_capture_pump() -> None:
    """Criterion 2: STT_FRAME_MS is the pump's frame, not a second opinion about it.

    ``audio_capture`` owns the fact (1920 samples at 24 kHz = 80 ms); this row fails if the two
    ever disagree, and pins the window size 38 = ceil(3000/80) the boundedness row uses.
    """
    assert STT_BLOCK_SAMPLES == 1920
    assert float(rig.const("STT_FRAME_MS", "criterion 2")) == STT_BLOCK_MS == 80.0
    window_ms = rig.const("NOISE_FLOOR_WINDOW_MS", "criterion 3")
    frames = -(-int(window_ms) // int(STT_BLOCK_MS))  # ceil division, by hand: 3000/80 -> 38
    assert frames == rig.WINDOW_FRAMES == 38


def test_chosen_constants_say_they_are_chosen_and_name_what_would_measure_them() -> None:
    """Criterion 3: each tuning constant's docstring says it is *chosen, not measured*.

    Ambiguity 3 of the task: these three are guesses until the first live session. A constant
    that does not say so becomes a number somebody later quotes as measured.
    """
    rig.attr("LevelMeter", "criterion 3")  # the row is about a module that exists
    text = rig.source()
    tree = ast.parse(text)
    lines = text.splitlines()
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name):
            starts[target.id] = node.lineno
            ends[target.id] = node.end_lineno or node.lineno
    for name in ("NOISE_FLOOR_PERCENTILE", "VOICED_MARGIN_DB", "LEVEL_SPAN_DB"):
        assert name in starts, f"criterion 3: {name} is not a module-level assignment"
        i = starts[name] - 2
        doc: list[str] = []
        while i >= 0 and lines[i].lstrip().startswith("#"):
            doc.append(lines[i])
            i -= 1
        after = ends[name]
        if after < len(lines) and lines[after].lstrip().startswith(('"', "'")):
            doc.append(lines[after])
        blob = " ".join(doc).lower()
        assert "chosen" in blob and "measur" in blob, (
            f"criterion 3: {name}'s docstring must state its regime and say it is "
            f"'chosen, not measured', naming what would measure it. Found: {doc!r}"
        )


# --- criterion 2: the record, and the frame's own capture time ----------------------------------


def test_emission_carries_the_five_fields_at_the_frames_capture_time() -> None:
    """Criterion 2: t_ms is the frame's capture time, on an irregular grid a clock cannot fake.

    Frames every 137 ms from t=1234: every frame is more than PUBLISH_INTERVAL_MS after the last,
    so each one emits and each emission's t_ms is its own frame time.
    """
    times = [1234.0 + 137.0 * i for i in range(6)]
    values = [-58.0, -57.0, -40.0, -39.0, -55.0, -56.0]
    _, emissions = rig.drive(values, times=times)
    assert len(emissions) >= 5, "every gap here is 137 ms > PUBLISH_INTERVAL_MS, so frames emit"
    assert emissions[-1].index == 5
    for e in emissions:
        assert e.t_ms == pytest.approx(times[e.index], abs=1e-9), (
            "criterion 2: t_ms is the observed frame's capture time, not a counter or a clock"
        )
        assert isinstance(e.voiced, bool)
        assert isinstance(e.level, float)
        assert isinstance(e.rms_dbfs, float)
        assert isinstance(e.noise_floor_dbfs, float)


# --- criterion 3: the running P20 floor, the margin, the mapping --------------------------------

#: A 38-frame window designed so that its 0.2 quantile is -60.0 under *every* common convention.
#: Sorted it is: five values -70..-66, then a five-wide plateau of -60 at sorted indices 5..9,
#: then 28 values -50..-23. numpy's default (linear) rule puts P20 at index 0.2*(38-1) = 7.4,
#: nearest-rank at ceil(0.2*38) = 8 -> index 7, the (n+1)p rule at 7.8 - all inside the plateau.
#: Neither the minimum (-70) nor the mean (about -44.7) is -60, so the row cannot pass by accident.
SCRIPTED_WINDOW = (
    [-70.0, -69.0, -68.0, -67.0, -66.0]
    + [-60.0] * 5
    + [-49.0 + i for i in range(26)]
    + [-50.0, -23.0]
)


def test_noise_floor_is_the_p20_of_the_scripted_window() -> None:
    """Criterion 3: the floor is the P20 of the last 38 frames, hand-computed above.

    The first frame (-95.0) is fed to be evicted: when frame 38 is observed the window is exactly
    SCRIPTED_WINDOW, so the assertion is about the designed multiset and nothing else.
    """
    assert len(SCRIPTED_WINDOW) == rig.WINDOW_FRAMES == 38
    values = [-95.0, *SCRIPTED_WINDOW]
    _, emissions = rig.drive(values, dt_ms=DT)
    last = emissions[-1]
    assert last.index == 38, (
        "criterion 4: frame 38 (t=3040 ms) is an emitting frame on the 80 ms grid; "
        f"the meter last emitted at frame {last.index}"
    )
    assert last.noise_floor_dbfs == pytest.approx(-60.0, abs=1e-9)


def test_voiced_is_strictly_above_the_margin() -> None:
    """Criterion 3: voiced is rms > floor + 9.0 dB, strictly.

    38 frames of a -60 dBFS room (floor = -60 exactly: every value is equal), then one probe at
    frame 38. At exactly -51.0 the speaker is not voiced; 0.1 dB above it, they are.
    """
    margin = float(rig.const("VOICED_MARGIN_DB", "criterion 3"))
    room = [-60.0] * rig.WINDOW_FRAMES
    at_margin = -60.0 + margin  # -51.0, by hand
    _, emissions = rig.drive([*room, at_margin], dt_ms=DT)
    edge = emissions[-1]
    assert edge.index == 38
    assert edge.noise_floor_dbfs == pytest.approx(-60.0, abs=1e-9)
    assert edge.rms_dbfs == pytest.approx(at_margin, abs=1e-9)
    assert edge.voiced is False, "criterion 3: the comparison is strict >, not >="

    _, louder = rig.drive([*room, at_margin + 0.1], dt_ms=DT)
    assert louder[-1].voiced is True


@pytest.mark.parametrize(
    ("probe_db", "expected_level", "why"),
    [
        (-60.0, 0.0, "at the floor: (0)/30"),
        (-45.0, 0.5, "15 dB over the floor: 15/30"),
        (-15.0, 1.0, "45 dB over the floor: 45/30 = 1.5, clamped to 1.0"),
    ],
)
def test_level_maps_the_floor_and_the_span(
    probe_db: float, expected_level: float, why: str
) -> None:
    """Criterion 3: level = clamp((rms - floor)/30, 0, 1), over a -60 dBFS room."""
    room = [-60.0] * rig.WINDOW_FRAMES
    _, emissions = rig.drive([*room, probe_db], dt_ms=DT)
    probe = emissions[-1]
    assert probe.index == 38
    assert probe.noise_floor_dbfs == pytest.approx(-60.0, abs=1e-9), why
    assert probe.level == pytest.approx(expected_level, abs=1e-9), why


# --- criterion 4: throttle and peak hold --------------------------------------------------------


def test_emissions_are_one_per_two_frames_on_the_80ms_grid() -> None:
    """Criterion 4: 100 ms of frame time on an 80 ms grid = every second frame (6.25 Hz)."""
    values = [-60.0 + (i % 5) for i in range(40)]
    _, emissions = rig.drive(values, dt_ms=DT)
    assert 19 <= len(emissions) <= 20, (
        f"criterion 4: 40 frames of 80 ms is 3.2 s; at 6.25 Hz that is 20 emissions, "
        f"got {len(emissions)}"
    )
    gaps = [b.t_ms - a.t_ms for a, b in pairwise(emissions)]
    assert gaps == pytest.approx([rig.EMISSION_GAP_MS] * len(gaps), abs=1e-9)
    strides = {b.index - a.index for a, b in pairwise(emissions)}
    assert strides == {2}


def test_the_first_frame_emits() -> None:
    """Criterion 4: "at most once per 100 ms" - the first frame has no predecessor to wait for.

    A meter that waits 100 ms before its first value makes the page show nothing for the first
    two frames of every session. If the implementer reads criterion 4 the other way, this is the
    row to argue with (see the test-author's report: this reading is the one ambiguity here).
    """
    values = [-60.0] * 40
    _, emissions = rig.drive(values, dt_ms=DT)
    assert emissions[0].index == 0
    assert emissions[0].t_ms == pytest.approx(0.0, abs=1e-9)
    assert len(emissions) == 20


def test_a_loud_frame_between_emissions_is_peak_held() -> None:
    """Criterion 4, adversarial row 2: one loud 80 ms frame between two emissions is reported.

    Frames 0..38 are a -60 dBFS room, frame 39 is a shout at -34 dBFS, frame 40 is quiet again.
    Frame 39 falls *between* emissions, so a "last frame wins" meter reports -60 and the page
    never shows the shout. Hand-computed at frame 40: the window is 37 frames of -60 plus the
    one -34, whose P20 is -60; peak-held rms is -34; level = (-34 + 60)/30 = 0.86666...
    """
    values = [*([-60.0] * 39), -34.0, -60.0]
    _, emissions = rig.drive(values, dt_ms=DT)
    indices = [e.index for e in emissions]
    assert 39 not in indices, "the grid was chosen so the loud frame does not emit by itself"
    after = [e for e in emissions if e.index > 39]
    assert after, "criterion 4: frame 40 is an emitting frame"
    held = after[0]
    assert held.rms_dbfs == pytest.approx(-34.0, abs=1e-9), (
        "criterion 4: the reported rms is the maximum since the last emission, not the last frame"
    )
    assert held.noise_floor_dbfs == pytest.approx(-60.0, abs=1e-9)
    assert held.level == pytest.approx(26.0 / 30.0, abs=1e-9)
    assert held.voiced is True


# --- adversarial row 1: the room gets louder ----------------------------------------------------


def test_a_rising_noise_floor_stops_reporting_voiced_within_one_window() -> None:
    """Adversarial row 1: a fan starts, and the meter must stop calling the room a speaker.

    3 s of a -60 dBFS room (38 frames), then every frame is 15 dB louder (-45 dBFS) for 6.4 s.
    At first -45 is 15 dB over the old floor, so voiced is true - that is correct, and the row
    asserts it so it cannot pass vacuously. Within one window the floor must have followed the
    room: hand-computed, the P20 of a window holding k frames of -45 and 38-k of -60 reaches -45
    at k = 31 (2480 ms of frame time), and -45 is not more than 9 dB over -45.

    A floor computed once, or a running minimum that can only fall, passes every other row in
    this file and fails here: it stays at -60 and the meter calls the fan a voice for ever.
    """
    quiet = [-60.0] * rig.WINDOW_FRAMES
    loud = [-45.0] * 80
    values = [*quiet, *loud]
    t_rise = rig.WINDOW_FRAMES * DT  # 3040.0 ms: the first frame of the fan
    _, emissions = rig.drive(values, dt_ms=DT)

    after = [e for e in emissions if e.frame_t_ms >= t_rise]
    assert after and after[0].voiced is True, (
        "non-vacuity: the first frames of the fan are 15 dB over the old floor and must read voiced"
    )
    voiced_times = [e.frame_t_ms for e in emissions if e.voiced]
    assert voiced_times, "non-vacuity: something must have been voiced at some point"
    window_ms = float(rig.const("NOISE_FLOOR_WINDOW_MS", "criterion 3"))
    assert max(voiced_times) - t_rise <= window_ms, (
        f"criterion 3: the floor is a *running* percentile. Last voiced emission at "
        f"{max(voiced_times)} ms, the room changed at {t_rise} ms, window {window_ms} ms"
    )


# --- ruling A8: the peak is released, not accumulated -------------------------------------------


def test_the_peak_is_released_after_the_emission_that_reported_it() -> None:
    """Criterion 4 ("the maximum since the *last emission*"), ruling A8: the next emission falls.

    The row above establishes that a loud frame survives the gap to the next emission; nothing
    established that the peak is then released. Deleting the reset turns the meter into a
    session-wide running maximum: the bar never falls again and ``voiced`` sticks True for the
    rest of the session - the live-looking meter on a dead microphone that criterion 15 exists to
    prevent, arrived at from the server side. That mutant passes every other row in this file.

    Frames 0..38 are a -60 dBFS room, frame 39 is a shout at -34 dBFS, frames 40..47 are the same
    room again. Frame 40 is the emission that reports the held peak (-34). Frame 42 is the first
    emission whose interval (frames 41-42) holds only room frames, so it reports -60 again.
    Hand-computed at frame 42: the window is frames 5..42, i.e. 37 values of -60 and one of -34,
    whose P20 is -60; level = (-60 - -60)/30 = 0.0 and -60 is not more than 9 dB over -60.
    """
    values = [*([-60.0] * 39), -34.0, *([-60.0] * 8)]
    _, emissions = rig.drive(values, dt_ms=DT)
    by_index = {e.index: e for e in emissions}
    assert 39 not in by_index, "the grid was chosen so the loud frame does not emit by itself"
    for wanted in (40, 42, 44, 46):
        assert wanted in by_index, f"criterion 4: frame {wanted} is an emitting frame"

    reported = by_index[40]
    assert reported.rms_dbfs == pytest.approx(-34.0, abs=1e-9), (
        "non-vacuity: the shout must reach the page at all (the peak-hold row above)"
    )
    assert reported.voiced is True

    released = by_index[42]
    assert released.rms_dbfs == pytest.approx(-60.0, abs=1e-9), (
        "criterion 4: the peak is the maximum *since the last emission*, so the emission after "
        "the one that reported the shout is the room again - a meter that keeps a session-wide "
        "maximum shows a full bar on a silent microphone for ever"
    )
    assert released.noise_floor_dbfs == pytest.approx(-60.0, abs=1e-9)
    assert released.level == pytest.approx(0.0, abs=1e-9)
    assert released.voiced is False, "criterion 4: voiced must not stick once it has been true"

    for later in (44, 46):
        assert by_index[later].rms_dbfs == pytest.approx(-60.0, abs=1e-9)
        assert by_index[later].voiced is False


# --- ruling A7: substituted silence is not the room ---------------------------------------------

#: ``audio_capture.py`` substitutes exact ``np.zeros`` for self-speech-gated blocks and for xrun
#: holes, and ``SttFrame`` carries no flag saying so, so those frames reach the meter as
#: SILENCE_DBFS and are indistinguishable from a dead microphone. 25 frames is 2000 ms: the two
#: seconds of assistant speech the reviewer measured (ruling A7).
GATED_FRAMES = 25


def test_substituted_silence_does_not_drag_the_floor_down_onto_the_room() -> None:
    """Criterion 3 as amended by ruling A7: a frame at SILENCE_DBFS never enters the window.

    The shape of the real failure: a -60 dBFS room (38 frames, 3.04 s), then 2 s of gated
    self-speech that reaches the tap as digital silence, then *the same room*. Hand-computed
    without the amendment: at frame 63 the window (frames 26..63) holds 25 values of -100, so the
    0.2 quantile - nearest rank 8, linear index 7.4, every common convention - lands inside the
    -100 block and the floor is -100. The unchanged room then reads
    level = clamp((-60 - -100)/30) = 1.0 and voiced (-60 > -100 + 9), and it keeps reading that
    until fewer than 8 gated frames remain in the window, i.e. through frame 92: 2400 ms of a
    voiced full bar on an empty room, at the top of the listening window. That is the reviewer's
    2320 ms, and a worse lie than the frozen bar criterion 15 exists to prevent.

    With the amendment the window still holds only the room's own frames, so the floor stays at
    -60 and the room reads level 0.0 and not voiced, before, during and after the gate. No other
    row in this file feeds SILENCE_DBFS into the window (ruling A5 keeps the Hypothesis rows
    inside [-97, -3] dB), so this row is the only thing standing between the amended criterion
    and the bug.
    """
    room_db, gated_db = -60.0, float(rig.const("SILENCE_DBFS", "criterion 1"))
    gate_t = rig.WINDOW_FRAMES * DT  # 3040.0 ms: the first substituted frame
    resume_t = (rig.WINDOW_FRAMES + GATED_FRAMES) * DT  # 5040.0 ms: the room, unchanged
    values = [
        *([room_db] * rig.WINDOW_FRAMES),
        *([gated_db] * GATED_FRAMES),
        *([room_db] * rig.WINDOW_FRAMES),
    ]
    _, emissions = rig.drive(values, dt_ms=DT)

    before = [e for e in emissions if e.frame_t_ms < gate_t]
    assert len(before) >= 15, "non-vacuity: the quiet room emits throughout its 3 s"
    assert not any(e.voiced for e in before), "a steady room is not a voice (criterion 5(d))"

    during = [e for e in emissions if gate_t < e.frame_t_ms < resume_t]
    assert during, "non-vacuity: the gated stretch emits too - the pump keeps feeding frames"
    for e in during:
        assert e.voiced is False, "ruling A7: substituted silence is never a voice"
        assert e.level == pytest.approx(0.0, abs=1e-9), (
            "ruling A7: the emission for a gated frame is honest - silence against the room's "
            "own floor is an empty bar"
        )

    after = [e for e in emissions if e.frame_t_ms >= resume_t]
    assert len(after) >= 15, "non-vacuity: the room plays for another 3 s after the gate"
    liars = [e for e in after if e.voiced or e.level > 0.05]
    lied_ms = (
        (max(e.frame_t_ms for e in liars) - min(e.frame_t_ms for e in liars) + DT) if liars else 0.0
    )
    assert not liars, (
        f"criterion 3 as amended by ruling A7: the room did not change, but the meter reports "
        f"voiced={liars[0].voiced} level={liars[0].level:.3f} on it for {lied_ms:.0f} ms "
        f"(first at t={liars[0].frame_t_ms:.0f} ms, floor {liars[0].noise_floor_dbfs:.1f} dBFS). "
        f"{GATED_FRAMES} frames of substituted silence dragged the P20 of the 38-frame window to "
        f"SILENCE_DBFS; those frames must not enter the window at all"
    )
    for e in after:
        assert e.noise_floor_dbfs == pytest.approx(room_db, abs=1e-9)
        assert e.level == pytest.approx(0.0, abs=1e-9)


def test_a_dead_stream_reads_empty_and_a_near_silent_room_still_sets_the_floor() -> None:
    """Ruling A7's other half, and its boundary: the exclusion is *the constant*, nothing else.

    (a) A stream that is nothing but substituted silence leaves the window with no non-silent
    frame at all; the floor is then SILENCE_DBFS, so a genuinely dead microphone reads level 0.0
    and not voiced - never a full bar, and never a raise (the excluding implementation must cope
    with an empty window).

    (b) A room 2 dB above the constant is still a room: 38 frames at -98 dBFS set the floor at
    -98, and a frame 10 dB over it is voiced with level 10/30. An implementation that excluded a
    *band* of quiet values rather than the constant would leave the window holding only the -88
    probe, whose P20 is -88, and would call it silence.
    """
    silence = float(rig.const("SILENCE_DBFS", "criterion 1"))
    _, dead = rig.drive([silence] * 40, dt_ms=DT)
    assert len(dead) == 20, "non-vacuity: a dead stream still emits at 6.25 Hz"
    for e in dead:
        assert e.noise_floor_dbfs == pytest.approx(silence, abs=1e-9), (
            "ruling A7: with no non-silent frame in the window the floor is SILENCE_DBFS"
        )
        assert e.level == pytest.approx(0.0, abs=1e-9)
        assert e.voiced is False

    _, near = rig.drive([*([-98.0] * rig.WINDOW_FRAMES), -88.0], dt_ms=DT)
    probe = near[-1]
    assert probe.index == 38
    assert probe.noise_floor_dbfs == pytest.approx(-98.0, abs=1e-9), (
        "ruling A7 excludes frames *at* SILENCE_DBFS, not a band of quiet ones"
    )
    assert probe.level == pytest.approx(10.0 / 30.0, abs=1e-9)
    assert probe.voiced is True


# --- criterion 2: bounded ------------------------------------------------------------------------


def deep_size(value: object) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        size += sum(sys.getsizeof(v) for v in value)
    elif isinstance(value, dict):
        size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in value.items())
    elif isinstance(value, np.ndarray):
        size = int(value.nbytes) + sys.getsizeof(value)
    return size


def observe_many(inst: object, count: int, *, t0: float = 0.0) -> None:
    observe = inst.observe  # type: ignore[attr-defined]
    cycle = (-62.0, -58.0, -30.0, -61.0, -59.0, -12.0, -70.0)
    for i in range(count):
        observe(cycle[i % len(cycle)], t0 + i * DT)


def test_the_meter_is_bounded_after_100k_observations() -> None:
    """Criterion 2: after 100 000 frames (2.2 h of audio) nothing has grown.

    Generic on purpose - it reads whatever attributes the implementer chose (``__dict__`` or
    ``__slots__``), so a second, forgotten list of every frame ever seen is caught too. This is
    also criterion 6's negative ("no allocation proportional to the number of frames seen").
    """
    small = rig.meter()
    observe_many(small, 200)
    baseline = {k: deep_size(v) for k, v in rig.instance_attrs(small).items()}

    big = rig.meter()
    observe_many(big, 100_000)
    attrs = rig.instance_attrs(big)
    assert attrs, "criterion 2: the meter keeps a window; it cannot be stateless"
    for name, value in attrs.items():
        if hasattr(value, "__len__"):
            assert len(value) <= rig.WINDOW_FRAMES, (
                f"criterion 2: {name} holds {len(value)} entries after 100 000 frames; "
                f"the window is at most {rig.WINDOW_FRAMES}"
            )
            for element in list(value)[:4]:
                if hasattr(element, "__len__") and not isinstance(element, (str, bytes)):
                    assert len(element) <= rig.WINDOW_FRAMES, (
                        f"criterion 2: {name} holds a container of {len(element)} entries"
                    )
        assert deep_size(value) <= baseline.get(name, 0) + 512, (
            f"criterion 2: {name} grew from {baseline.get(name)} to {deep_size(value)} bytes "
            f"between 200 and 100 000 frames"
        )


# --- purity ---------------------------------------------------------------------------------------

#: `.claude/rules/conductor.md` ("no clock, no logging, no I/O in the reflex lane") and criteria
#: 1-2 ("pure"; time arrives as the frame's t_ms).
FORBIDDEN_IMPORTS = (
    "time",
    "asyncio",
    "socket",
    "logging",
    "datetime",
    "threading",
    "uuid",
    "subprocess",
)


def test_the_module_has_no_clock_no_socket_and_no_logger() -> None:
    """Criteria 1-2: purity, asserted on the module's own source, not on its behaviour."""
    rig.attr("LevelMeter", "criteria 1-2")  # the row is about a module that exists
    tree = ast.parse(rig.source())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    offenders = sorted(imported & set(FORBIDDEN_IMPORTS))
    assert not offenders, (
        f"the reflex lane is pure: {rig.MODULE} imports {offenders} "
        f"(.claude/rules/conductor.md; time arrives as the frame's t_ms)"
    )
    namespace = vars(rig.module())
    leaked = sorted(
        name
        for name, value in namespace.items()
        if getattr(value, "__name__", None) in FORBIDDEN_IMPORTS
    )
    assert not leaked, f"{rig.MODULE} exposes {leaked} in its namespace"


# --- criterion 5: the Hypothesis properties -----------------------------------------------------


def db(low: float = -90.0, high: float = -5.0) -> st.SearchStrategy[float]:
    return st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)


def half_db(low: float = -85.0, high: float = -15.0) -> st.SearchStrategy[float]:
    """dB values on a 0.5 grid: exactly representable, so a gain shift is exact arithmetic."""
    return st.integers(min_value=int(low * 2), max_value=int(high * 2)).map(lambda n: n / 2.0)


PROPERTY = settings(derandomize=True, max_examples=40, deadline=None)


@PROPERTY
@given(values=st.lists(db(), min_size=1, max_size=120))
def test_property_level_is_bounded_and_the_floor_lies_inside_the_window(
    values: list[float],
) -> None:
    """Criterion 5(a): 0 <= level <= 1, and the floor is between the window's min and max.

    The window is the criterion's, not the implementation's: the last 38 frames observed.
    """
    _, emissions = rig.drive(values, dt_ms=DT)
    for e in emissions:
        assert 0.0 <= e.level <= 1.0
        window = rig.window_of(values, e.index)
        assert min(window) - 1e-9 <= e.noise_floor_dbfs <= max(window) + 1e-9


@PROPERTY
@given(
    script=st.lists(
        st.tuples(db(), st.integers(min_value=5, max_value=400)), min_size=1, max_size=90
    )
)
def test_property_emissions_are_at_least_one_interval_apart(
    script: list[tuple[float, int]],
) -> None:
    """Criterion 5(b): emissions are >= PUBLISH_INTERVAL_MS apart in *frame* time, any grid."""
    interval = float(rig.const("PUBLISH_INTERVAL_MS", "criterion 4"))
    values = [v for v, _ in script]
    times: list[float] = []
    t = 0.0
    for _, gap in script:
        times.append(t)
        t += float(gap)
    _, emissions = rig.drive(values, times=times)
    for a, b in pairwise(emissions):
        assert b.t_ms - a.t_ms >= interval - 1e-9
    for e in emissions:
        assert e.t_ms == pytest.approx(e.frame_t_ms, abs=1e-9)


@PROPERTY
@given(values=st.lists(half_db(), min_size=1, max_size=90), gain_db=half_db(-12.0, 12.0))
def test_property_gain_invariance(values: list[float], gain_db: float) -> None:
    """Criterion 5(c): the meter is relative to the room.

    Adding k dB to every frame must leave every emitted level and voiced verdict alone - an
    absolute threshold anywhere in the helper (the `_listen.py` 0.02 floor this task exists to
    avoid) fails here.
    """
    _, plain = rig.drive(values, dt_ms=DT)
    _, shifted = rig.drive([v + gain_db for v in values], dt_ms=DT)
    assert [e.index for e in plain] == [e.index for e in shifted]
    assert [e.voiced for e in plain] == [e.voiced for e in shifted]
    assert [e.level for e in plain] == pytest.approx([e.level for e in shifted], abs=1e-9)


@PROPERTY
@given(value=db(-99.0, 0.0), extra=st.integers(min_value=2, max_value=40))
def test_property_a_constant_room_is_never_voiced(value: float, extra: int) -> None:
    """Criterion 5(d): a constant input is never voiced once the window is full, at any level.

    Steady noise is a room, however loud it is; only something above it is a voice.
    """
    values = [value] * (rig.WINDOW_FRAMES + extra)
    _, emissions = rig.drive(values, dt_ms=DT)
    # The window holds 38 frames once frame 37 has been observed; frame 38 is the first emitting
    # frame after that on the 80 ms grid, which is why the run is at least 40 frames long.
    full = [e for e in emissions if e.index >= rig.WINDOW_FRAMES]
    assert full, "non-vacuity: with >= 40 frames the meter emits at least once on a full window"
    assert not any(e.voiced for e in full)
