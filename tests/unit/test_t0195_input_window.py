"""T-0195 group B, the pure half: the per-window input accumulator ``local_astra/reflex/input_window.py`` (B1, B3, B7).

``docs/tasks/T-0195.md`` group B. The task leaves this module's API to the test-author; **the API these rows fix**:

- ``LEVEL_WINDOW_MS = 10_000`` (B1) and ``UNMUTE_CHECK_MS = 5_000`` (B4), module constants;
- ``InputWindow()``: one window's state, created empty. A new window is a new object (the session builds one per
  ``mic.level`` row and one per unmute check); there is no reset method, so nothing carries across rows;
- ``live_frame(samples, t_ms)``: one captured frame the mic gate passes live (``samples`` a float array of any length,
  1920 in production; ``t_ms`` the frame's own capture time). The samples are read, never kept and never modified;
- ``muted_frame(t_ms)``: one frame the mute feeds as zeros (B3: the captured samples are never handed over);
- ``word(*, estimated: bool, dropped_muted: bool)``: one ``SttWordEvent`` at ``_on_stt_event`` entry;
- ``stt_step(pause_heads)``: one ``SttStepEvent``'s ``pause_heads`` vector. Not ``step``: ``test_session_listening`` and
  ``test_session_loop`` count every ``.step`` attribute in ``session.py`` as a ``machine.step`` call site (census);
- ``onset()``: one ``("onset", ...)`` endpointer decision;
- ``summary() -> dict``: exactly the B1 measurement fields (the row minus ``what``, ``reason`` and ``stt``, which are
  the session's): ``from_frame_t_ms``, ``to_frame_t_ms`` (first and last frame seen, ``None`` before any),
  ``frames`` (every frame, live and muted), ``muted_frames``, ``zero_frames`` (live frames whose samples are all
  exactly 0), ``nonfinite_frames`` (live frames holding a ``nan`` or ``inf``), ``rms_dbfs`` (``10 log10`` of the mean
  square over *all samples* of the live, non-zero, finite frames), ``peak_dbfs`` (``20 log10`` of the largest
  ``|sample|`` over the same frames), ``max_frame_rms_dbfs`` / ``min_frame_rms_dbfs`` (``reflex.level.rms_dbfs`` per
  frame, over the same frames), ``words_received``, ``words_estimated``, ``words_dropped_muted``,
  ``steps_received``, ``onsets``, ``pause_head_min`` / ``pause_head_max`` (``pause_heads[PAUSE_HEAD_INDEX]`` over the
  steps whose vector is long enough; ``None`` if none). Every dBFS value is a finite ``float`` floored at
  ``SILENCE_DBFS``, or ``None`` when no frame qualifies.

Reading (test-author, stated in the report): "the same frames" for the per-frame min/max. B1 says ``rms_dbfs`` is over
live non-zero finite frames and that min/max are per 80 ms frame; a zero frame in the min would pin it at -100 in
every window with one xrun, and B5(b)'s reading ("max within ~15 dB of min") would be useless. So min/max/peak use the
qualifying set too.

Expected values are computed by hand from closed forms (a sine with whole periods in the frame has mean square
``A**2 / 2``), never by calling ``rms_dbfs`` on the same input.

Written ``pending(task="T-0195")``; activated 2026-10-29 when group B's implementation started (every row red at
BASE 51291e70 on the missing module).

Regime: CPU, numpy, n = 1 deterministic per row except the cost row (1000 frames of 1920 float32, seed 0, median
of ``perf_counter_ns`` deltas, a guard not a measurement; machine load in the failure message).
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.reflex.input_window"
SILENCE = -100.0
N = 1920
#: 400 Hz at 24 kHz: 60 samples a period, 32 whole periods in 1920 samples, so mean(x**2) == A**2/2 exactly
#: (up to float32 rounding, far below 0.01 dB)
SINE_HZ = 400
RATE = 24_000

SUMMARY_KEYS = {
    "from_frame_t_ms",
    "to_frame_t_ms",
    "frames",
    "muted_frames",
    "zero_frames",
    "nonfinite_frames",
    "rms_dbfs",
    "peak_dbfs",
    "max_frame_rms_dbfs",
    "min_frame_rms_dbfs",
    "words_received",
    "words_estimated",
    "words_dropped_muted",
    "steps_received",
    "onsets",
    "pause_head_min",
    "pause_head_max",
}
LEVEL_KEYS = ("rms_dbfs", "peak_dbfs", "max_frame_rms_dbfs", "min_frame_rms_dbfs")


def mod() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError:
        raise AssertionError(
            f"T-0195 B1: {MODULE} does not exist yet (the pure accumulator, group B's Owned files)"
        ) from None


def window() -> Any:
    m = mod()
    assert hasattr(m, "InputWindow"), f"T-0195 B1: {MODULE}.InputWindow is missing"
    return m.InputWindow()


def sine(amplitude: float, n: int = N) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / RATE
    return (amplitude * np.sin(2 * math.pi * SINE_HZ * t)).astype(np.float32)


def db_of_sine(amplitude: float) -> float:
    """Closed form: a whole-period sine of amplitude A has RMS A/sqrt(2)."""
    return 20.0 * math.log10(amplitude / math.sqrt(2.0))


def close(got: Any, want: float, tol: float = 0.01) -> bool:
    return isinstance(got, float) and math.isfinite(got) and abs(got - want) <= tol


def test_constants_are_the_tasks() -> None:
    m = mod()
    assert getattr(m, "LEVEL_WINDOW_MS", None) == 10_000, "B1: LEVEL_WINDOW_MS = 10_000"
    assert getattr(m, "UNMUTE_CHECK_MS", None) == 5_000, "B4: the unmute check is 5000 ms"


def test_sine_rms_within_a_hundredth_of_a_db() -> None:
    w = window()
    w.live_frame(sine(0.5), 1000.0)
    s = w.summary()
    assert set(s) == SUMMARY_KEYS, f"B1: summary keys {sorted(s)}; want {sorted(SUMMARY_KEYS)}"
    want = db_of_sine(0.5)  # -9.031 dBFS
    assert close(s["rms_dbfs"], want), (
        f"B1: rms of a 0.5 sine is {want:.3f} dBFS; got {s['rms_dbfs']}"
    )
    assert close(s["max_frame_rms_dbfs"], want) and close(s["min_frame_rms_dbfs"], want), s
    assert (s["frames"], s["muted_frames"], s["zero_frames"], s["nonfinite_frames"]) == (
        1,
        0,
        0,
        0,
    ), s
    assert (s["from_frame_t_ms"], s["to_frame_t_ms"]) == (1000.0, 1000.0), s


def test_window_rms_is_over_all_samples_and_min_max_are_per_frame() -> None:
    """Two frames, 0.5 and 0.05: the window's mean square is (0.25/2 + 0.0025/2)/2, not the mean of two dB values."""
    w = window()
    w.live_frame(sine(0.5), 0.0)
    w.live_frame(sine(0.05), 80.0)
    s = w.summary()
    want_window = 10.0 * math.log10((0.5**2 / 2 + 0.05**2 / 2) / 2)  # -12.00 dBFS
    mean_of_db = (db_of_sine(0.5) + db_of_sine(0.05)) / 2  # -19.03: the wrong average
    assert close(s["rms_dbfs"], want_window), (
        f"B1: rms_dbfs over all samples is {want_window:.3f}; got {s['rms_dbfs']} "
        f"(the mean of the two frame values would be {mean_of_db:.3f})"
    )
    assert close(s["max_frame_rms_dbfs"], db_of_sine(0.5)), s
    assert close(s["min_frame_rms_dbfs"], db_of_sine(0.05)), s
    assert (s["from_frame_t_ms"], s["to_frame_t_ms"], s["frames"]) == (0.0, 80.0, 2), s


def test_peak_is_the_largest_absolute_sample() -> None:
    w = window()
    quiet = sine(0.01)
    spiky = sine(0.01)
    spiky[700] = -0.9  # a negative spike: the peak is |x|
    w.live_frame(quiet, 0.0)
    w.live_frame(spiky, 80.0)
    s = w.summary()
    want = 20.0 * math.log10(0.9)  # -0.915 dBFS
    assert close(s["peak_dbfs"], want), (
        f"B1: peak of a -0.9 spike is {want:.3f} dBFS; got {s['peak_dbfs']}"
    )


def test_zero_frames_are_counted_and_not_measured() -> None:
    """A live all-zero frame (xrun hole, self-speech gate) is counted; it does not drag the rms 3 dB down or pin the
    minimum at the floor."""
    w = window()
    w.live_frame(sine(0.5), 0.0)
    w.live_frame(np.zeros(N, dtype=np.float32), 80.0)
    s = w.summary()
    assert s["zero_frames"] == 1 and s["frames"] == 2, s
    assert close(s["rms_dbfs"], db_of_sine(0.5)), (
        f"B1: rms is over non-zero frames only; {db_of_sine(0.5):.3f} expected, got {s['rms_dbfs']} "
        f"(with the zero frame averaged in it would be {db_of_sine(0.5) - 10 * math.log10(2):.3f})"
    )
    assert close(s["min_frame_rms_dbfs"], db_of_sine(0.5)), s
    only_zero = window()
    only_zero.live_frame(np.zeros(N, dtype=np.float32), 0.0)
    z = only_zero.summary()
    assert z["zero_frames"] == 1 and all(z[k] is None for k in LEVEL_KEYS), (
        f"B1: no qualifying frame -> null levels; got {z}"
    )


def test_nonfinite_frames_never_leak_into_the_row() -> None:
    """Adversarial (B1: "dBFS finite"): nan and inf frames are counted and excluded; the summary is strict JSON."""
    w = window()
    loud = sine(0.5)
    w.live_frame(loud, 0.0)
    for k, bad in enumerate((math.nan, math.inf, -math.inf)):
        frame = sine(0.9)
        frame[123] = bad
        w.live_frame(frame, 80.0 * (k + 1))
    s = w.summary()
    assert s["nonfinite_frames"] == 3 and s["frames"] == 4, s
    text = json.dumps(s, allow_nan=False)  # raises ValueError on nan/inf anywhere
    assert "NaN" not in text and "Infinity" not in text
    for key in LEVEL_KEYS:
        if key == "peak_dbfs":
            want = 20.0 * math.log10(float(np.max(np.abs(loud.astype(np.float64)))))
        else:
            want = db_of_sine(0.5)
        assert close(s[key], want), (
            f"B1: {key} must be over the finite frames only ({want:.3f}); got {s[key]}"
        )
    only_bad = window()
    frame = np.full(N, math.nan, dtype=np.float32)
    only_bad.live_frame(frame, 0.0)
    b = only_bad.summary()
    json.dumps(b, allow_nan=False)
    assert b["nonfinite_frames"] == 1 and all(b[k] is None for k in LEVEL_KEYS), b


def test_levels_are_floored_at_silence_dbfs() -> None:
    w = window()
    w.live_frame(sine(1e-7), 0.0)  # -143 dBFS by the closed form
    s = w.summary()
    for key in LEVEL_KEYS:
        assert s[key] == SILENCE, f"B1: {key} floored at SILENCE_DBFS ({SILENCE}); got {s[key]}"


def test_samples_are_not_modified() -> None:
    """B6: the accumulator reads the frame the pump also packs for ``stt.feed``; nan stays nan, values stay put."""
    w = window()
    frame = sine(0.5)
    frame[5] = math.nan
    frame[6] = math.inf
    before = frame.tobytes()
    w.live_frame(frame, 0.0)
    w.summary()
    assert frame.tobytes() == before, "B6: live_frame modified the samples in place"


def test_all_muted_window_has_null_levels() -> None:
    """B3: muted frames count toward ``muted_frames`` (and ``frames``) only."""
    w = window()
    for i in range(5):
        w.muted_frame(80.0 * i)
    s = w.summary()
    assert (s["frames"], s["muted_frames"], s["zero_frames"]) == (5, 5, 0), s
    assert all(s[k] is None for k in LEVEL_KEYS), (
        f"B3: an all-muted window has null levels; got {s}"
    )
    assert (s["from_frame_t_ms"], s["to_frame_t_ms"]) == (0.0, 320.0), s
    w.live_frame(sine(0.05), 400.0)
    s = w.summary()
    assert close(s["max_frame_rms_dbfs"], db_of_sine(0.05)), s
    assert close(s["rms_dbfs"], db_of_sine(0.05)), (
        f"B3: muted frames are not samples of the window's rms; got {s['rms_dbfs']}"
    )


def test_event_counts_and_pause_head_range() -> None:
    m = mod()
    from local_astra.reflex.turn_predictor import PAUSE_HEAD_INDEX

    assert PAUSE_HEAD_INDEX == 2, "premise: the vectors below put their marker at index 2"
    w = m.InputWindow()
    empty = w.summary()
    assert (
        empty["frames"] == 0 and empty["from_frame_t_ms"] is None and empty["to_frame_t_ms"] is None
    ), empty
    assert empty["pause_head_min"] is None and empty["pause_head_max"] is None, empty
    w.word(estimated=False, dropped_muted=False)
    w.word(estimated=True, dropped_muted=False)
    w.word(estimated=True, dropped_muted=True)
    w.word(estimated=False, dropped_muted=True)
    w.stt_step((0.1, 0.1, 0.7, 0.1))
    w.stt_step((0.9, 0.9, 0.2, 0.9, 0.9))
    w.stt_step((0.0, 0.0))  # too short: a step, but no pause head
    w.stt_step(())
    w.onset()
    s = w.summary()
    assert (s["words_received"], s["words_estimated"], s["words_dropped_muted"]) == (4, 2, 2), s
    assert (s["steps_received"], s["onsets"]) == (4, 1), s
    assert s["pause_head_min"] == pytest.approx(0.2) and s["pause_head_max"] == pytest.approx(
        0.7
    ), f"B1: pause heads at index {PAUSE_HEAD_INDEX}: min 0.2, max 0.7; got {s}"
    json.dumps(s, allow_nan=False)


def test_nonfinite_pause_heads_never_leak_into_the_row() -> None:
    """B1 "finite" (review A F3): a step whose head at ``PAUSE_HEAD_INDEX`` is nan, +inf or -inf counts as a step and
    leaves the min/max untouched; the summary stays strict JSON. Alone they give null min/max; beside finite heads
    the finite range."""
    only_bad = window()
    for bad in (math.nan, math.inf, -math.inf):
        only_bad.stt_step((0.5, 0.5, bad, 0.5))
    s = only_bad.summary()
    json.dumps(s, allow_nan=False)
    assert s["steps_received"] == 3, s
    assert s["pause_head_min"] is None and s["pause_head_max"] is None, (
        f"B1: non-finite heads are not a pause-head value; got {s['pause_head_min']}, {s['pause_head_max']}"
    )
    mixed = window()
    mixed.stt_step((0.0, 0.0, 0.3))
    for bad in (math.inf, -math.inf, math.nan):
        mixed.stt_step((0.0, 0.0, bad))
    mixed.stt_step((0.0, 0.0, 0.8))
    m = mixed.summary()
    json.dumps(m, allow_nan=False)
    assert (m["steps_received"], m["pause_head_min"], m["pause_head_max"]) == (5, 0.3, 0.8), (
        f"B1: the finite range 0.3..0.8 beside nan/inf heads; got {m}"
    )


def test_a_new_window_starts_from_zero() -> None:
    m = mod()
    a = m.InputWindow()
    a.live_frame(sine(0.5), 0.0)
    a.word(estimated=False, dropped_muted=False)
    b = m.InputWindow()
    s = b.summary()
    assert s["frames"] == 0 and s["words_received"] == 0 and s["rms_dbfs"] is None, (
        f"B1: counts are per row; a fresh window shares nothing; got {s}"
    )


def _footprint(obj: Any) -> list[tuple[str, int]]:
    names: list[str] = []
    for cls in type(obj).__mro__:
        names.extend(getattr(cls, "__slots__", ()))
    names.extend(getattr(obj, "__dict__", {}).keys())
    out = []
    for name in sorted(set(names)):
        value = getattr(obj, name, None)
        size = len(value) if hasattr(value, "__len__") else sys.getsizeof(value)
        out.append((name, size))
    return out


def test_state_does_not_grow_with_frames() -> None:
    """B7: O(1) state per window: 100 000 frames, words and steps leave every attribute's size where 100 left it."""
    w = window()
    frame = sine(0.3, n=48)
    for i in range(100):
        w.live_frame(frame, 80.0 * i)
        w.muted_frame(80.0 * i + 40)
        w.word(estimated=bool(i % 2), dropped_muted=False)
        w.stt_step((0.5, 0.5, 0.5))
        w.onset()
    early = _footprint(w)
    assert early, "premise: the window has attributes to measure"
    for i in range(100, 100_000):
        w.live_frame(frame, 80.0 * i)
        if i % 10 == 0:
            w.muted_frame(80.0 * i + 40)
            w.word(estimated=False, dropped_muted=True)
            w.stt_step((0.4, 0.4, 0.6))
            w.onset()
    late = _footprint(w)
    assert late == early, f"B7: state grew with the number of frames: {early} -> {late}"
    assert w.summary()["frames"] == 100_000 + 100 + 9_990, "premise: every frame was counted"


def test_module_is_pure() -> None:
    """B1's Owned-files line: no clock, I/O, logging or asyncio."""
    path = REPO / "local_astra" / "reflex" / "input_window.py"
    assert path.exists(), f"T-0195 B1: {path} does not exist yet"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {
        "time",
        "asyncio",
        "logging",
        "socket",
        "threading",
        "datetime",
        "os",
        "io",
        "pathlib",
    }
    assert not imported & banned, f"B1: the accumulator imports {sorted(imported & banned)}"
    assert "local_astra.conductor" not in path.read_text(encoding="utf-8"), (
        "the reflex lane imports nothing from the conductor"
    )


#: B7: 50 us median per 1920-sample frame (the task's number); a guard, not a measurement
CEILING_US = 50.0
FRAMES = 1000


def test_per_frame_cost_under_50us_median() -> None:
    m = mod()
    w = m.InputWindow()
    rng = np.random.default_rng(0)
    signal = (0.1 * rng.standard_normal((FRAMES + 1, N))).astype(np.float32)
    w.live_frame(signal[FRAMES], -80.0)  # warm-up outside the measurement
    deltas: list[int] = []
    for i in range(FRAMES):
        frame = signal[i]
        t0 = time.perf_counter_ns()
        w.live_frame(frame, 80.0 * i)
        deltas.append(time.perf_counter_ns() - t0)
    deltas.sort()
    median_us = deltas[FRAMES // 2] / 1000.0
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    assert median_us <= CEILING_US, (
        f"B7: median {median_us:.1f} us > {CEILING_US} us per 1920-sample frame "
        f"(n={FRAMES}, seed 0, numpy {np.__version__}, load {load:.1f})"
    )
    assert w.summary()["frames"] == FRAMES + 1
