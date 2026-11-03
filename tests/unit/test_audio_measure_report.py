"""T-0023 group A: the two measurement scripts' report contract and their arithmetic.

No hardware, no device, no socket, no latency number is produced here (regime: pure CPU, Python
3.11, numpy as in ``uv.lock``; ``profile=n/a canvas=n/a steps=n/a``). What is checked is the part
of T-0023 that can be wrong *before* anyone enters the room: a report that quotes a number without
its regime (criterion 1), a p95 interpolated out of ten samples, an overlap scorer that counts
stop words, a control that "passes" because the microphone is dead (criterion 3), counters
re-derived from audio instead of read from ``CaptureStats``, an RSS growth computed as
``last − first``, and a 20 s run labelled as the 10-minute number (criterion 7).

The scripts are loaded by path (``tests/fakes/locked_scripts.py``) because ``scripts/locked/`` is
the ``/launch``-gated directory, not a package. Until they exist every test here fails with an
``AssertionError`` naming the missing file — never an ``ImportError``.

## The pure-helper API pinned here

So the implementer cannot be "nearly right": these names, these signatures, these semantics.
All helpers are **pure** — no argparse, no device, no I/O at import time.

``scripts/locked/audio_acoustic.py``::

    REGIME_FIELDS: tuple[str, ...]        # exactly REQUIRED_REGIME_FIELDS below, any order
    NOT_MEASURED: str                     # the literal "NOT MEASURED"
    STOP_WORDS: frozenset[str]            # casefolded; superset of MINIMUM_STOP_WORDS below
    class RegimeError(ValueError)         # a missing/empty regime field
    class ReportError(ValueError)         # a malformed report row
    class ControlFailure(ValueError)      # the gate-off control did not transcribe
    emit_report(report: Mapping[str, Any], out: TextIO, err: TextIO) -> int
    headset_row(measurement: Mapping[str, Any] | None, *, reason: str | None = None) -> dict
    percentiles(samples: Sequence[float]) -> dict[str, Any]
    content_words(text: str) -> tuple[str, ...]
    overlap_fraction(spoken: str, transcript: str) -> float
    control_overlap(spoken: str, transcript: str, *, min_overlap: float = 0.5) -> float

``scripts/locked/audio_soak.py``::

    REGIME_FIELDS, emit_report, percentiles, ReportError   # same semantics; a re-export is fine
    class ShortRunError(ValueError)
    counters_from_snapshots(snapshots: Sequence[CaptureStats]) -> dict[str, float]
    rss_growth_mb_per_min(samples: Sequence[tuple[float, float]]) -> float
    check_duration(measured_s: float, declared_s: float, *, tolerance_s: float = 1.0) -> None

Semantics that a test cannot infer and an implementer must not invent:

* ``emit_report`` writes **one** JSON document to ``out`` whose **first** key is ``"regime"`` and
  returns 0, **only** when every field of ``REGIME_FIELDS`` is present and non-empty. Otherwise it
  writes **nothing at all** to ``out``, names the offending field on ``err`` and returns non-zero
  (criterion 1: "print no measurement at all"). Empty means ``None``, ``""``, whitespace-only, or
  an empty container. **Zero is a value, not an emptiness**: ``seed=0`` and ``distance_cm=0`` (the
  headset pass) are valid.
* ``percentiles`` returns ``{"p50", "p95", "spread", "pstdev", "n", "p95_is_bound"}`` in float64.
  ``p50``/``p95`` are ``statistics.quantiles(sorted(samples), n=100, method="inclusive")[49]``
  and ``[94]``; ``spread = max - min``; ``pstdev = statistics.pstdev``. For ``n < 20`` it either
  raises ``ValueError`` or returns ``p95_is_bound=True`` with ``p95 == max(samples)`` — a bound,
  never an interpolation (criterion 2's n = 10 runs are reported as bounds).
* ``content_words`` casefolds, strips punctuation, removes ``STOP_WORDS``, and preserves order and
  duplicates. ``overlap_fraction`` is ``|set(content(transcript)) ∩ set(content(spoken))| /
  |set(content(spoken))|`` — set semantics, so a repeated word cannot inflate it — and raises
  ``ValueError`` when the spoken text has no content words.
* ``control_overlap`` raises ``ControlFailure`` (message naming the control) when the transcript
  has **no content words at all**, and when the overlap is below ``min_overlap``. It never returns
  0.0 for silence: that is criterion 3's whole point.
* ``counters_from_snapshots`` takes **one** argument — the ``CaptureStats`` snapshots, which is all
  it may ever see (criterion 7: the soak reads the counters, it does not re-derive xruns from the
  audio). The snapshots are cumulative, so each counter is ``last − first``; a counter that
  *decreases* between consecutive snapshots raises ``ReportError`` naming it, because a reset
  makes ``last − first`` an undercount. An empty sequence raises.
* ``rss_growth_mb_per_min`` is the least-squares slope over ``(t_s, rss_bytes)`` samples, in
  ``MB = 1024 * 1024`` bytes per minute (criterion 7: "never ``last − first``"). Fewer than two
  samples raises.
* ``check_duration`` raises ``ShortRunError`` naming both durations when
  ``measured_s < declared_s - tolerance_s``; a longer run is fine.

## The JSON report shape the integration tests read

``audio_acoustic.py`` (``--out``/``$ASTRA_ACOUSTIC_REPORT``)::

    {"regime": {...}, "delay_ms": {"speakers": {"p50","p95","spread","pstdev","n","p95_is_bound"},
                                   "headset": <same, or the NOT MEASURED row>},
     "leading_edge_ms": {"p50","p95","n"},
     "gate": {"on": {"overlap","spurious_words","n"}, "off": {"overlap","n"},
              "not_ours": {"overlap","n"}}}

``audio_soak.py`` (``--out``/``$ASTRA_SOAK_REPORT``)::

    {"regime": {...},
     "counters": {"blocks","xruns","dropped_blocks","substituted_ms","underruns","gated_blocks"},
     "handoff_ms": {"p50","p95","n","structural_frame_fill_ms"},
     "loop_lag_ms": {"p50","p99","n"},
     "rss": {"samples": [[t_s, rss_bytes], ...], "growth_mb_per_min", "n"},
     "duration_s": float, "declared_duration_s": float}
"""

from __future__ import annotations

import asyncio
import inspect
import json
import statistics
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_astra.conductor.audio_capture import CaptureBlock, CaptureStats
from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.locked_scripts import ACOUSTIC, SOAK, helper, load_locked_script

WHY_1 = "T-0023 criterion 1: a report without its regime is not a report"
WHY_2 = "T-0023 criterion 2: the echo-path delay, reported as p50/p95/spread/pstdev"
WHY_3 = "T-0023 criterion 3: zero self-transcription, with a control that self-transcribes"
WHY_7 = "T-0023 criterion 7: the 10-minute duplex run"

#: §Regime, verbatim: every field criterion 1 requires of both scripts' first JSON object.
REQUIRED_REGIME_FIELDS = (
    "box",
    "pipewire_version",
    "sink",
    "sink_volume",
    "source",
    "transducer",
    "distance_cm",
    "noise_floor_peak",
    "stt",
    "co_resident",
    "profile",
    "gate",
    "hangover_ms",
    "n",
    "seed",
    "git_sha",
    "duration_s",
    "negotiated_quantum_ms",
    "measured_dac_lag_ms",
)

#: The stop list is the implementer's, but these ten are in every English stop list and criterion
#: 3's scorer is meaningless without them.
MINIMUM_STOP_WORDS = frozenset({"a", "an", "and", "the", "of", "to", "is", "it", "that", "this"})

#: Nonsense on purpose: three words that are in nobody's stop list, so the denominator of
#: ``overlap_fraction`` is 3 whatever stop list the implementer chose.
SPOKEN = "The zebra and the xylophone of tungsten."
#: The *previous* utterance in a run of ten: content words disjoint from SPOKEN's.
PREVIOUS = "The quartz and the marimba of niobium."

MB = 1024 * 1024


def complete_regime() -> dict[str, Any]:
    """A regime block with every field of §Regime filled, as the speakers run would emit it."""
    return {
        "box": "gpu-box",
        "pipewire_version": "1.0.5 / libpipewire 1.0.5",
        "sink": "astra-test-sink",
        "sink_volume": "0.40",
        "source": "alsa_input.usb-046d_C925e_HD_Webcam",
        "transducer": "speakers",
        # zero is a value, not an emptiness: the headset pass states distance 0 cm
        "distance_cm": 0,
        "noise_floor_peak": 0.116,
        "stt": {
            "server": "moshi-server 0.6.4",
            "model_revision": "095e38f6242006a93c2541149b181988397f5c7c",
            "device": "cpu",
            "batch_size": 2,
        },
        "co_resident": "vllm=off astra-stt=on",
        "profile": "lean",
        "gate": "on",
        "hangover_ms": 60.0,
        "n": 10,
        # and zero is a value here too: seed 0 is the seed every run in this task uses
        "seed": 0,
        "git_sha": "8bf09c8",
        "duration_s": 600.0,
        "negotiated_quantum_ms": 10.0,
        "measured_dac_lag_ms": 15.3,
    }


#: Something with digits in it, so "printed no measurement" is a check and not a truism.
MEASUREMENT_BLOCK = {"delay_ms": {"speakers": {"p50": 31.2, "p95": 38.4, "n": 10}}}

#: n = 40 delay samples in ms (invented, fixed): enough for a confident p95, spread on purpose.
DELAYS_MS = [
    31.2, 29.8, 33.4, 30.1, 35.7, 28.9, 32.6, 34.0, 30.9, 36.5,
    29.3, 31.8, 33.1, 30.4, 38.2, 27.6, 32.2, 34.8, 31.5, 29.0,
    30.7, 33.9, 35.1, 28.4, 32.9, 31.1, 36.9, 30.2, 34.4, 29.6,
    33.3, 30.8, 37.4, 28.1, 32.0, 35.5, 31.9, 29.9, 34.1, 30.5,
]  # fmt: skip


@pytest.mark.parametrize("script", [ACOUSTIC, SOAK])
@pytest.mark.parametrize("field", REQUIRED_REGIME_FIELDS)
@pytest.mark.parametrize("mode", ["missing", "empty-string", "whitespace", "none"])
def test_a_missing_regime_field_refuses_to_print_a_number(
    script: str, field: str, mode: str
) -> None:
    """Criterion 1: drop or blank any one field and the script prints no measurement at all.

    The positive half runs first in every case, because "it refused" means nothing unless the
    same call with the field present succeeds — and because ``seed=0`` and ``distance_cm=0`` must
    survive it (a ``if not value`` emptiness check deletes the seed of every run in this task).
    """
    module = load_locked_script(script)
    emit = helper(module, "emit_report", WHY_1)
    fields = helper(module, "REGIME_FIELDS", WHY_1)
    assert set(fields) == set(REQUIRED_REGIME_FIELDS), (
        f"{script}.REGIME_FIELDS misses {sorted(set(REQUIRED_REGIME_FIELDS) - set(fields))} and "
        f"invents {sorted(set(fields) - set(REQUIRED_REGIME_FIELDS))} (§Regime, criterion 1)"
    )

    good_out, good_err = StringIO(), StringIO()
    rc = emit({"regime": complete_regime(), **MEASUREMENT_BLOCK}, good_out, good_err)
    assert rc == 0, f"a complete regime was refused: {good_err.getvalue()!r}"
    doc = json.loads(good_out.getvalue())
    assert next(iter(doc)) == "regime", (
        f"the first object of the report is {next(iter(doc))!r}, not the regime block (criterion 1)"
    )
    assert doc["regime"]["seed"] == 0 and doc["regime"]["distance_cm"] == 0, (
        "zero is a value: seed 0 and distance 0 cm were dropped or rejected as 'empty'"
    )

    regime = complete_regime()
    if mode == "missing":
        del regime[field]
    else:
        regime[field] = {"empty-string": "", "whitespace": "   ", "none": None}[mode]
    out, err = StringIO(), StringIO()
    rc = emit({"regime": regime, **MEASUREMENT_BLOCK}, out, err)

    assert rc != 0, f"{script} emitted a report with {field!r} {mode} and returned 0"
    assert out.getvalue().strip() == "", (
        f"{script} printed {out.getvalue()[:200]!r} on stdout although the regime field "
        f"{field!r} was {mode}: criterion 1 says print no measurement at all"
    )
    assert not any(ch.isdigit() for ch in out.getvalue()), "a number reached stdout"
    assert field in err.getvalue(), (
        f"the refusal does not name the offending field {field!r}: {err.getvalue()[:300]!r}"
    )


def test_the_headset_row_is_a_measurement_or_the_words_not_measured() -> None:
    """Criterion 1's substitution trap: no speakers number may appear under the headset label."""
    module = load_locked_script(ACOUSTIC)
    headset_row = helper(module, "headset_row", WHY_1)
    report_error = helper(module, "ReportError", WHY_1)
    not_measured = helper(module, "NOT_MEASURED", WHY_1)
    assert not_measured == "NOT MEASURED"

    owed = headset_row(None, reason="P4: no headset is connected to this box")
    assert owed["transducer"] == "headset"
    assert owed["status"] == not_measured
    assert "P4" in owed["reason"]
    assert all(not isinstance(v, (int, float)) or isinstance(v, bool) for v in owed.values()), (
        f"the NOT MEASURED row carries a number: {owed!r} — a reader cannot tell it from a "
        f"measurement (criterion 1)"
    )

    with pytest.raises(report_error):
        headset_row(None)
    with pytest.raises(report_error):
        headset_row(None, reason="   ")

    speakers = {"transducer": "speakers", "p50": 31.2, "p95": 38.4, "pstdev": 2.4, "n": 10}
    with pytest.raises(report_error) as caught:
        headset_row(speakers)
    assert "headset" in str(caught.value) and "speakers" in str(caught.value), (
        f"the speakers measurement was accepted under the headset label, or refused without "
        f"saying so: {caught.value}"
    )

    measured = {"transducer": "headset", "p50": 2.1, "p95": 3.0, "pstdev": 0.4, "n": 10}
    row = headset_row(measured)
    assert row["p95"] == 3.0
    assert row.get("status") != not_measured


def test_percentiles_match_a_float64_oracle() -> None:
    """Criterion 2's statistics against ``statistics.quantiles``, and the n < 20 refusal.

    The oracle is the stdlib on float64, not the implementation. The adversarial half is n = 10 —
    exactly criterion 2's sample size — where a p95 read off an interpolation between the two
    largest samples is a confident-looking number nobody measured.
    """
    module = load_locked_script(ACOUSTIC)
    percentiles = helper(module, "percentiles", WHY_2)

    ordered = sorted(float(x) for x in DELAYS_MS)
    cuts = statistics.quantiles(ordered, n=100, method="inclusive")
    got = percentiles(list(DELAYS_MS))
    assert got["n"] == len(DELAYS_MS)
    assert got["p50"] == pytest.approx(cuts[49], rel=1e-12)
    assert got["p95"] == pytest.approx(cuts[94], rel=1e-12)
    assert got["spread"] == pytest.approx(max(ordered) - min(ordered), rel=1e-12)
    assert got["pstdev"] == pytest.approx(statistics.pstdev(ordered), rel=1e-12)
    assert got["p95_is_bound"] is False

    short = [float(x) for x in DELAYS_MS[:10]]
    interpolated = statistics.quantiles(sorted(short), n=100, method="inclusive")[94]
    # the premise of the adversarial: on this sample an interpolated p95 and the bound differ
    assert interpolated != pytest.approx(max(short), rel=1e-9)
    try:
        small = percentiles(short)
    except ValueError as exc:
        assert "20" in str(exc) or "n=10" in str(exc) or "10" in str(exc), (
            f"n < 20 was refused without saying how many samples there were: {exc}"
        )
    else:
        assert small["p95_is_bound"] is True, (
            f"n = {len(short)} produced a confident p95 {small['p95']!r}: criterion 2 wants a "
            f"bound or a refusal, never an interpolation over ten samples"
        )
        assert small["p95"] == pytest.approx(max(short), rel=1e-12), (
            f"the bound is {small['p95']!r}, not the largest sample {max(short)!r} "
            f"(the interpolated value is {interpolated!r})"
        )


def test_content_word_overlap_ignores_case_and_punctuation_and_stop_words() -> None:
    """Criterion 3's scorer, with the two ways it is quietly wrong.

    Adversarial 1: the transcript of the *previous* utterance must score 0 against the current one
    — ten utterances run back to back and a scorer that matches on stop words scores every pair
    high. Adversarial 2: a transcript made of nothing but stop words scores 0, not "most of the
    sentence".
    """
    module = load_locked_script(ACOUSTIC)
    content_words = helper(module, "content_words", WHY_3)
    overlap = helper(module, "overlap_fraction", WHY_3)
    stop_words = helper(module, "STOP_WORDS", WHY_3)

    assert set(stop_words) >= MINIMUM_STOP_WORDS, (
        f"STOP_WORDS misses {sorted(MINIMUM_STOP_WORDS - set(stop_words))}"
    )
    assert all(w == w.casefold() for w in stop_words), "STOP_WORDS must be casefolded"

    assert tuple(content_words(SPOKEN)) == ("zebra", "xylophone", "tungsten")
    assert content_words("The, of! and the.") == ()

    assert overlap(SPOKEN, "the zebra and the xylophone of tungsten") == pytest.approx(1.0)
    assert overlap(SPOKEN, "A XYLOPHONE, of Tungsten!") == pytest.approx(2.0 / 3.0)
    assert overlap(SPOKEN, "xylophone xylophone xylophone") == pytest.approx(1.0 / 3.0), (
        "repeating one word inflated the overlap: the scorer must compare sets of content words"
    )
    assert overlap(SPOKEN, "the xylophone of tungsten and the zebra ran away") == pytest.approx(1.0)

    assert overlap(SPOKEN, PREVIOUS) == pytest.approx(0.0), (
        "the previous utterance scores above zero against this one: criterion 3's zero would be "
        "unreachable and its 50 % control meaningless"
    )
    all_stop_words = " ".join(sorted(stop_words))
    assert overlap(SPOKEN, all_stop_words) == pytest.approx(0.0), (
        "a transcript containing every stop word scored above zero"
    )
    assert overlap(SPOKEN, ".,;! -- ?") == pytest.approx(0.0)

    with pytest.raises(ValueError):
        overlap("the and of", "zebra")


def test_an_empty_control_transcript_errors_instead_of_passing() -> None:
    """Criterion 3's control rule — the test this half of T-0023 exists for.

    Without it "zero self-transcription with the gate on" is satisfied by a dead microphone, a
    dead STT and a gate stuck shut, and the headline number means nothing.
    """
    module = load_locked_script(ACOUSTIC)
    control_overlap = helper(module, "control_overlap", WHY_3)
    control_failure = helper(module, "ControlFailure", WHY_3)
    stop_words = helper(module, "STOP_WORDS", WHY_3)
    assert issubclass(control_failure, Exception)

    for silence in ("", "   ", "\n", ".,!?", " ".join(sorted(stop_words))):
        with pytest.raises(control_failure) as caught:
            control_overlap(SPOKEN, silence)
        assert "control" in str(caught.value).casefold(), (
            f"the refusal for {silence[:20]!r} does not name the control: {caught.value}"
        )

    assert control_overlap(SPOKEN, "a xylophone of tungsten") == pytest.approx(2.0 / 3.0)
    assert control_overlap(SPOKEN, PREVIOUS + " zebra xylophone tungsten") == pytest.approx(1.0)

    with pytest.raises(control_failure):
        # one content word out of three is below the 50 % criterion 3 demands of the control
        control_overlap(SPOKEN, "xylophone")


def test_counters_are_summed_from_the_stats_snapshots_not_recomputed() -> None:
    """Criterion 7: the soak reads ``CaptureStats``; it never re-derives an xrun from the audio."""
    module = load_locked_script(SOAK)
    counters = helper(module, "counters_from_snapshots", WHY_7)
    report_error = helper(module, "ReportError", WHY_7)

    params = list(inspect.signature(counters).parameters.values())
    assert len(params) == 1, (
        f"counters_from_snapshots{inspect.signature(counters)} takes more than the snapshots: "
        f"the counters come from CaptureStats and from nothing else (criterion 7)"
    )

    def snap(blocks: int, xruns: int, sub: float, dropped: int, gated: int) -> CaptureStats:
        return CaptureStats(
            blocks=blocks,
            xruns=xruns,
            substituted_ms=sub,
            dropped_blocks=dropped,
            gated_blocks=gated,
            quantum_ms=10.0,
            device="astra-test-sink",
        )

    # cumulative snapshots every 10 s, the first taken after warm-up (so ``last`` alone is wrong)
    snapshots = [
        snap(100, 1, 5.0, 2, 40),
        snap(1100, 1, 5.0, 2, 340),
        snap(2100, 1, 5.0, 2, 640),
        snap(3100, 5, 62.5, 7, 940),
    ]
    got = counters(snapshots)
    assert got["blocks"] == 3000
    assert got["xruns"] == 4
    assert got["dropped_blocks"] == 5
    assert got["substituted_ms"] == pytest.approx(57.5)
    assert got["gated_blocks"] == 900

    with pytest.raises((report_error, ValueError)):
        counters([])

    reset = [snap(100, 0, 0.0, 0, 0), snap(1100, 3, 0.0, 0, 0), snap(2100, 1, 0.0, 0, 0)]
    with pytest.raises((report_error, ValueError)) as caught:
        counters(reset)
    assert "xruns" in str(caught.value), (
        f"a counter that went 0 → 3 → 1 was folded into 'xruns = 1' instead of being reported as "
        f"a reset that makes last − first an undercount: {caught.value}"
    )


def test_rss_growth_is_a_fit_over_samples_not_last_minus_first() -> None:
    """Criterion 7's RSS slope. Adversarial: a garbage-collection dip in the final sample.

    ``last − first`` on that series is *negative* — a leak reported as a saving. The least-squares
    slope over the same samples stays positive, and matches an independent float64 fit.
    """
    module = load_locked_script(SOAK)
    growth = helper(module, "rss_growth_mb_per_min", WHY_7)

    rising = [(float(t), 100.0 * MB + 2.0 * MB * (t / 60.0)) for t in range(0, 601, 10)]
    assert len(rising) == 61  # criterion 7's n ≥ 60 samples, every 10 s
    assert growth(rising) == pytest.approx(2.0, rel=1e-9), (
        "an exactly linear 2 MB/min series did not come back as 2 MB/min "
        "(MB = 1024 * 1024 bytes, per minute)"
    )

    dipped = list(rising)
    dipped[-1] = (600.0, rising[0][1] - 5.0 * MB)  # the GC ran just before the last sample
    assert dipped[-1][1] - dipped[0][1] < 0, "the premise: last − first is negative here"

    t = np.array([s[0] for s in dipped], dtype=np.float64)
    y = np.array([s[1] for s in dipped], dtype=np.float64)
    oracle = float(np.polyfit(t, y, 1)[0]) * 60.0 / MB  # bytes/s → MB/min, an independent fit

    got = growth(dipped)
    assert got > 0.0, (
        f"a rising series with a GC dip in the last sample reported {got} MB/min: the slope is a "
        f"fit over all {len(dipped)} samples, never last − first (criterion 7)"
    )
    assert got == pytest.approx(oracle, rel=1e-6), f"{got} MB/min against a float64 fit of {oracle}"

    with pytest.raises(ValueError):
        growth([(0.0, 100.0 * MB)])


def test_a_run_shorter_than_the_declared_duration_is_refused() -> None:
    """Criterion 7: a 20 s run may not be labelled the 10-minute number."""
    module = load_locked_script(SOAK)
    check_duration = helper(module, "check_duration", WHY_7)
    short_run = helper(module, "ShortRunError", WHY_7)

    assert check_duration(600.0, 600.0) is None
    assert check_duration(612.3, 600.0) is None  # longer is fine
    assert check_duration(599.5, 600.0) is None  # within the 1 s tolerance

    with pytest.raises(short_run) as caught:
        check_duration(20.0, 600.0)
    assert "20" in str(caught.value) and "600" in str(caught.value), (
        f"the refusal names neither what was run nor what was declared: {caught.value}"
    )
    with pytest.raises(short_run):
        check_duration(598.0, 600.0)


def test_merge_report_never_overwrites_a_measurement_and_every_section_keeps_its_regime() -> None:
    """Ruling A8: the stages run one at a time, so merging must not lose what was measured.

    ``--measure gate`` runs after ``--measure delay``, sometimes hours later and always with a
    different regime line (the gate run states ``gate=on``, the delay run does not gate at all;
    the headset pass states a different transducer and distance). Two rules follow, and neither is
    visible in a single-stage run — deleting both leaves the rest of this module green:

    1. an owed ``NOT MEASURED`` row never overwrites a number somebody actually took, while a
       fresh measurement does replace an older one of the same transducer (re-runs are the point);
    2. every section carries **its own** regime block, complete, because criterion 1 is per
       number: a delay measured at 90 cm with the gate off may not inherit the gate run's regime
       merely because that run happened later.
    """
    module = load_locked_script(ACOUSTIC)
    merge_report = helper(module, "merge_report", WHY_1)
    not_measured = helper(module, "NOT_MEASURED", WHY_1)

    delay_regime = {**complete_regime(), "gate": "off", "distance_cm": 90, "n": 10}
    gate_regime = {**complete_regime(), "gate": "on", "distance_cm": 90, "n": 10}
    speakers = {"p50": 31.2, "p95": 38.4, "n": 10, "transducer": "speakers", "regime": delay_regime}
    headset = {"p50": 2.1, "p95": 3.0, "n": 10, "transducer": "headset", "regime": delay_regime}
    owed = {"transducer": "headset", "status": not_measured, "reason": "P4: no headset"}
    gate_section = {
        "on": {"overlap": 0.0, "spurious_words": 1, "n": 10},
        "off": {"overlap": 0.8, "n": 10},
        "regime": gate_regime,
    }

    previous = {"regime": delay_regime, "delay_ms": {"speakers": speakers, "headset": headset}}
    fresh = {"regime": gate_regime, "gate": gate_section, "delay_ms": {"headset": owed}}
    merged = merge_report(previous, fresh)

    assert next(iter(merged)) == "regime", "the merged document still leads with a regime block"
    assert merged["delay_ms"]["speakers"].get("p95") == 38.4, (
        "the gate stage erased the delay the speakers pass measured"
    )
    assert merged["delay_ms"]["headset"].get("p95") == 3.0, (
        f"an owed {not_measured!r} row overwrote a headset number that was actually taken: "
        f"{merged['delay_ms']['headset']!r} (criterion 1, ruling A8)"
    )
    assert merged["gate"]["on"]["overlap"] == 0.0

    for path in (("delay_ms", "speakers"), ("delay_ms", "headset")):
        section = merged[path[0]][path[1]]
        assert set(section["regime"]) == set(REQUIRED_REGIME_FIELDS), (
            f"{'.'.join(path)} lost fields of its own regime: "
            f"{sorted(set(REQUIRED_REGIME_FIELDS) - set(section['regime']))}"
        )
        assert section["regime"]["gate"] == "off", (
            f"{'.'.join(path)} was measured with the gate off but now carries the gate run's "
            f"regime: criterion 1 is per number, not per document (ruling A8)"
        )
    assert merged["gate"]["regime"]["gate"] == "on"
    assert set(merged["gate"]["regime"]) == set(REQUIRED_REGIME_FIELDS)

    # a re-measurement *does* replace: the owed row is the only thing that yields
    remeasured = {
        "p50": 33.0,
        "p95": 40.0,
        "n": 10,
        "transducer": "speakers",
        "regime": gate_regime,
    }
    again = merge_report(merged, {"regime": gate_regime, "delay_ms": {"speakers": remeasured}})
    assert again["delay_ms"]["speakers"].get("p95") == 40.0, (
        "a fresh speakers measurement did not replace the older one: re-running a stage after a "
        "fix would be impossible"
    )
    assert again["delay_ms"]["headset"].get("p95") == 3.0, "the untouched headset row was lost"

    # a section handed over *without* a regime is stamped with the run that produced it — that is
    # how "every section carries its own regime" survives a stage that forgot (ruling A8) — and
    # stamping never reaches a section that was carried over from an earlier run
    bare = merge_report(
        previous, {"regime": gate_regime, "gate": {"on": {"overlap": 0.0, "n": 10}}}
    )
    assert set(bare["gate"]["regime"]) == set(REQUIRED_REGIME_FIELDS), (
        "a fresh section with no regime of its own was merged bare: the number it holds is then "
        "unattributable, which is what criterion 1 forbids (ruling A8)"
    )
    assert bare["gate"]["regime"]["gate"] == "on"
    assert bare["delay_ms"]["speakers"]["regime"]["gate"] == "off", (
        "the carried-over delay row was re-stamped with the later run's regime"
    )

    # and an owed row is still recorded when there is nothing to protect
    empty = merge_report({}, {"regime": delay_regime, "delay_ms": {"headset": owed}})
    assert empty["delay_ms"]["headset"]["status"] == not_measured
    assert "P4" in empty["delay_ms"]["headset"]["reason"]


def test_the_scorer_reads_numerals_as_the_number_words_the_fixture_speaks() -> None:
    """Ruling A9: "1 2 3 4 5" is the fixture sentence, not a transcript of nothing.

    ``pocket_tts_sentence.wav`` says *"one, two, three, four, five."* and moshi may return it as
    numerals. Without this normalisation the mandatory gate-off control (criterion 3) scores 0,
    ``control_overlap`` raises ``ControlFailure``, and the whole acoustic measurement errors out
    on a transcript that is in fact perfect — the false-negative twin of the dead-microphone
    failure the control exists to catch.
    """
    module = load_locked_script(ACOUSTIC)
    content_words = helper(module, "content_words", WHY_3)
    overlap = helper(module, "overlap_fraction", WHY_3)
    control_overlap = helper(module, "control_overlap", WHY_3)

    # the fixture's own text (tests/fixtures/audio/pocket_tts_sentence.words.json)
    fixture_text = "one, two, three, four, five."
    assert overlap(fixture_text, "1 2 3 4 5") == pytest.approx(1.0)
    assert overlap(fixture_text, "One 2 three, 4 FIVE!") == pytest.approx(1.0)
    assert overlap(fixture_text, "1 2 3") == pytest.approx(3.0 / 5.0)
    assert overlap("1 2 3 4 5", "one two three four five") == pytest.approx(1.0), (
        "the normalisation has to work in both directions: which side the digits arrive on is "
        "the STT's choice, not ours"
    )
    assert control_overlap(fixture_text, "1 2 3 4 5") == pytest.approx(1.0)

    # a different set of numbers is still a different sentence
    assert overlap(fixture_text, "6 7 8 9 10") == pytest.approx(0.0), (
        "every numeral collapsed to one token: the previous utterance would score full overlap"
    )

    pairs = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    }  # fmt: skip
    for word, digits in pairs.items():
        assert content_words(word) == content_words(digits), (
            f"{word!r} and {digits!r} are not the same content word (zero through twenty is the "
            f"range the fixtures and the counting prompts use)"
        )


# --- ruling A12: five fixes whose mutations left this module green --------------------------------


def test_the_dac_lag_is_reported_only_when_the_callback_moved_it() -> None:
    """Ruling A6: ``measured_dac_lag_ms`` is a measurement or it is empty.

    ``PortAudioDevice._dac_lag_frames`` *starts* at the negotiated latency and is overwritten by
    every callback with ``outputBufferDacTime − currentTime``. Reading it without asking whether
    it moved reports the request wearing a measurement's label — which is the one thing the field
    exists to rule out (T-0006 measured 15.3 ms against a negotiated 10 ms). Empty then makes
    criterion 1 refuse the whole report naming the field, which is the intended loud failure.
    """
    module = load_locked_script(ACOUSTIC)
    facts = helper(module, "device_lag_facts", WHY_1)
    rate = 48_000

    class StubDevice:
        quantum_ms = 10.0
        latency_ms = 30.0

        def __init__(self, lag_frames: int | None) -> None:
            if lag_frames is not None:
                self._dac_lag_frames = lag_frames

    negotiated_frames = round(30.0 * rate / 1000.0)  # 1440 frames = the *requested* 30 ms
    untouched = facts(StubDevice(negotiated_frames))
    assert untouched["measured_dac_lag_ms"] == "", (
        f"the negotiated latency was reported as a measured DAC lag "
        f"({untouched['measured_dac_lag_ms']!r}): no callback ever moved it (ruling A6)"
    )
    assert untouched["negotiated_quantum_ms"] == 10.0

    moved = facts(StubDevice(734))  # what a real callback wrote: 15.29 ms
    assert moved["measured_dac_lag_ms"] == pytest.approx(1000.0 * 734 / rate)

    absent = facts(StubDevice(None))
    assert absent["measured_dac_lag_ms"] == "", "a backend that keeps no lag reports none"


def test_closing_a_measurement_stops_the_device_not_only_the_capture() -> None:
    """Ruling A9: ``close_capture()`` alone leaves the PortAudio stream and its callback running.

    Every measurement in the acoustic script opens its own device, so the next one would open a
    second stream on the same node: the graph re-negotiates, the quantum in the regime stops
    describing the run, and the leaked writer keeps the sink busy — jitter in exactly the number
    being measured.
    """
    module = load_locked_script(ACOUSTIC)
    close_io = helper(module, "close_io", WHY_2)
    calls: list[str] = []

    class StubIo:
        def close_capture(self) -> None:
            calls.append("close_capture")

    class StubDevice:
        def stop(self) -> None:
            calls.append("stop")

    close_io(StubIo(), StubDevice())
    assert calls == ["close_capture", "stop"], (
        f"close_io() did {calls}: the capture iterators finish first and then the device stops "
        f"(ruling A9)"
    )


def test_the_stt_regime_refuses_to_invent_a_batch_size_or_read_our_own_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling A7: an unknown ``batch_size`` is empty, and the STT's device comes from the server.

    Two ways this field lies. A missed TOML regex falling back to ``0`` sails straight through
    criterion 1's "zero is a value" rule and prints ``batch_size 0`` beside every number. And
    ``ASTRA_STT_CPU`` says how somebody *meant* to start the service — not how the server
    answering on 8080 was actually started, possibly yesterday, in another shell, on the GPU.
    """
    module = load_locked_script(ACOUSTIC)
    stt_regime = helper(module, "stt_regime", WHY_1)
    stt_device = helper(module, "stt_device", WHY_1)

    services = tmp_path / "services" / "moshi"
    services.mkdir(parents=True)
    (tmp_path / "services" / "versions.lock").write_text('moshi_server = "0.6.4"\n')
    # a config with no batch_size at all: the regex misses, and that is an *unknown*, not a zero
    (services / "stt-1b-en_fr.toml").write_text("[modules.asr]\nasr_delay_in_tokens = 6\n")
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setenv("ASTRA_STT_MODEL_REVISION", "095e38f6242006a93c2541149b181988397f5c7c")
    monkeypatch.setenv("ASTRA_STT_CPU", "1")
    monkeypatch.setattr(module, "_capture", lambda *a, **k: (1, ""))  # no docker to ask

    regime = stt_regime()
    assert regime["batch_size"] == "", (
        f"a missed batch_size regex became {regime['batch_size']!r}: zero is a value under "
        f"criterion 1 and would be printed beside every number as if somebody had read it"
    )
    assert regime["device"] == "", (
        f"stt.device came back {regime['device']!r} with no server to ask and ASTRA_STT_CPU=1 in "
        f"the environment: that variable is this process's intention, not the server's state "
        f"(ruling A7)"
    )
    assert regime["server"] == "moshi-server 0.6.4"

    assert stt_device("cpu") == "cpu", "an explicit --stt-device is the documented override"
    monkeypatch.setattr(module, "_capture", lambda *a, **k: (0, '["--cpu","--port","8080"]'))
    assert stt_device() == "cpu"
    monkeypatch.setattr(module, "_capture", lambda *a, **k: (0, '["--port","8080"]'))
    assert stt_device() == "gpu"


def test_the_capture_timeline_refuses_a_hole_or_an_xrun_marked_block() -> None:
    """Ruling A10: a gap in ``t_ms`` breaks the sample-index → capture-clock mapping.

    The echo delay is read off exactly that mapping, so splicing across a hole moves the
    correlation peak by the audio that never arrived — silently, and in the direction that makes
    the path look shorter. A block the device marked ``xrun`` is the same defect announced.
    """
    module = load_locked_script(ACOUSTIC)
    timeline = helper(module, "capture_timeline", WHY_2)
    report_error = helper(module, "ReportError", WHY_2)
    block_ms = float(CAPTURE_BLOCK_MS)
    size = 480  # 10 ms at 48 kHz

    def block(i: int, *, t_ms: float | None = None, xrun: bool = False) -> CaptureBlock:
        return CaptureBlock(
            samples=np.full(size, float(i), dtype=np.float32),
            t_ms=1_000.0 + i * block_ms if t_ms is None else t_ms,
            xrun=xrun,
        )

    good = [block(i) for i in range(5)]
    samples, first_ms = timeline(good, block_ms=block_ms)
    assert first_ms == 1_000.0
    assert np.array_equal(np.asarray(samples), np.concatenate([b.samples for b in good]))

    holed = [block(0), block(1), block(2, t_ms=1_000.0 + 3 * block_ms), block(3)]
    with pytest.raises(report_error) as caught:
        timeline(holed, block_ms=block_ms)
    assert "hole" in str(caught.value)

    announced = [block(0), block(1), block(2, xrun=True), block(3)]
    with pytest.raises(report_error):
        timeline(announced, block_ms=block_ms)

    with pytest.raises(report_error):
        timeline([], block_ms=block_ms)


async def test_the_soak_duty_cycle_marks_the_drain_after_every_utterance() -> None:
    """Ruling A5: without ``mark_drain()`` every silence in the duty cycle is an underrun.

    Criterion 7 wants ``underruns == 0`` over 600 s of 4 s speech / 6 s silence. Playback with an
    empty queue and no declared end of stream is exactly T-0006 criterion 7's underrun, so a duty
    cycle that never marks the drain makes that criterion unreachable *by construction* — and the
    run only finds out after ten minutes of hardware time.

    Proved here on the fakes at 1/10 scale: four 400 ms speech / 600 ms silence cycles on
    ``FakeAudioDevice``'s virtual clock, ticked 1:1 with real time. This is the script's own
    ``soak()`` — its duty cycle, its counters, its report — with the device swapped underneath.
    """
    import argparse

    from local_astra.conductor.audio_io import PipeWireAudioIO
    from tests.fakes.audio_device import FakeAudioDevice

    module = load_locked_script(SOAK)
    soak = helper(module, "soak", WHY_7)
    acoustic = helper(module, "_acoustic", WHY_7)

    device = FakeAudioDevice()
    io = PipeWireAudioIO(device)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(acoustic, "build_io", lambda *a, **k: (io, device))
    monkey.setattr(module, "SPEECH_S", 0.4)
    monkey.setattr(module, "SILENCE_S", 0.6)
    monkey.setattr(module, "SAMPLE_PERIOD_S", 0.25)

    async def tick() -> None:
        """The device's clock at 1:1 with real time — 10 ms of audio every 10 ms — and its
        delivery, which the fake leaves to the test (``pump()``)."""
        while True:
            await asyncio.sleep(0.01)
            device.clock.advance(10)
            device.pump()

    ticker = asyncio.create_task(tick())
    try:
        report = await soak(argparse.Namespace(sink="s", source="s", duration=4.0, seed=0))
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        monkey.undo()

    cycles = round(4.0 / (0.4 + 0.6))
    assert device.blocks, "nothing was ever written to the device: the duty cycle did not play"
    assert report["counters"]["underruns"] == 0, (
        f"{report['counters']['underruns']:.0f} underruns over {cycles} duty cycles on a device "
        f"that was never starved: the silences were counted as underruns because the duty cycle "
        f"did not declare the end of each utterance (mark_drain(), ruling A5). Criterion 7's "
        f"underruns == 0 would be unreachable, and the run would only say so after 600 s"
    )
    assert report["duration_s"] >= 4.0
    assert report["declared_duration_s"] == 4.0
