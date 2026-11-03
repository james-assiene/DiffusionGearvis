#!/usr/bin/env python3
"""T-0023 criteria 1-6: the echo path, the gate against a real room, and the leading edge.

``/launch``-gated (``scripts/locked/``): every measurement here plays audio through a
loudspeaker, records a microphone and feeds ``astra-stt`` on 127.0.0.1:8080. Nothing in this
file runs at import: the pure helpers below are the whole of T-0023 group A and are unit-tested
without a device (``tests/unit/test_audio_measure_report.py``, which pins their names, their
signatures and their semantics).

Three measurements, one JSON report (merged into ``--out``/``$ASTRA_ACOUSTIC_REPORT`` so the
stages of docs/tasks/T-0023.md can be run one at a time)::

    --measure delay     criterion 2: DAC -> air -> ADC, n per transducer, seed 0
    --measure gate      criteria 3, 4 and 6: gate on / gate off (the control) / not ours,
                        then the leading-edge residual over --residual-n dedicated playbacks
    --measure residual  criterion 6 alone

**The delay is measured with the module's own clocks** (ruling A4): the chirp is played through
``PipeWireAudioIO`` and heard through the same duplex device's capture tap, and

    delay = (the correlation peak of the chirp in the raw capture, on the capture clock)
            - (the rendering span's start for that chirp, ``captured_ms() + the queued lead``)

which is the DAC -> air -> ADC path exactly as the gate sees it, from one clock, with no external
recorder in the timing path. ``acoustic_delay_ms`` and ``rendering_span_start_ms`` below mirror
the helpers of the same names in ``tests/fakes/audio_device.py``, which
``tests/unit/test_acoustic_delay_method.py`` pins against a synthetic 137.5 ms echo. The
monitor-vs-mic ``pw-record`` pair that A1 first specified is **dropped from the delay entirely**:
two recorders started 0.3 s apart share no time origin, and a sink that resumes late turns that
jitter into a plausible ~300 ms "echo delay" that would push the hangover past its 200 ms bound.
``pw-record`` survives here only in the audible-rig guard, where it answers an energy question
and times nothing. The device's absolute DAC lag is reported separately as
``measured_dac_lag_ms``, and only when the callback moved it off the negotiated value (A6).

**The gate measurement feeds the STT tap, never the raw stream**: ``io.stt_blocks()`` straight
into ``MoshiSttClient.feed(pcm, t_ms)`` with no resampling, reframing or retiming (T-0021
criterion 7), and every transcript is read after ``TAIL_S`` of **real captured** tail, because
the delayed-stream model advances its clock on audio and on nothing else. The gate-off control
uses the same tail policy as the gated run or the comparison is invalid.

**A report without its regime is not a report** (criterion 1): ``emit_report`` writes nothing at
all unless every field of ``REGIME_FIELDS`` is present and non-empty, and zero is a value --
``seed=0`` and the headset pass's ``distance_cm=0`` are measurements, not emptiness.

Regime (docs/tasks/T-0023.md §Regime, acoustic): box ``gpu-box``, the sink named by
``--sink``/``$ASTRA_ACOUSTIC_SINK`` with its ``wpctl`` volume, transducer as given (speakers for
the primary numbers, headset a separate labelled pass), Logitech C925e mic, mic-to-speaker
distance in cm on the command line, room noise floor measured in the same session, STT =
moshi-server 0.6.4 on CPU batch_size 2 at 127.0.0.1:8080, profile lean, n and seed as given, git
tree clean (this script refuses to measure on a dirty tree).

This script writes only to ``$TMPDIR`` and to ``--out``; it opens no socket of its own (the STT
client's loopback websocket is the only one) and it never edits a threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "audio"
CHIRP = FIXTURES / "chirp_5s_24k.wav"
SENTENCE = FIXTURES / "pocket_tts_sentence.wav"
SENTENCE_WORDS = FIXTURES / "pocket_tts_sentence.words.json"

DEVICE_RATE = 48000
TTS_RATE = 24000

#: §Regime, verbatim: what criterion 1 requires beside every number this script prints.
REGIME_FIELDS: tuple[str, ...] = (
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

#: The fields only the run itself can fill. Everything else is known before a speaker moves, and
#: is validated there (ruling A7: a soak that fails on `git rev-parse` after 600 s is a defect).
MEASURED_REGIME_FIELDS: tuple[str, ...] = (
    "noise_floor_peak",
    "duration_s",
    "negotiated_quantum_ms",
    "measured_dac_lag_ms",
)
#: ... and these are the ones checked *before* the first sample is played.
STATIC_REGIME_FIELDS: tuple[str, ...] = tuple(
    name for name in REGIME_FIELDS if name not in MEASURED_REGIME_FIELDS
)

#: The literal criterion 1 demands of an owed row. A speakers number may never appear under the
#: headset label; "NOT MEASURED" plus a reason is the only other thing a headset row may be.
NOT_MEASURED = "NOT MEASURED"

#: Below this many samples a p95 is a bound (``max``), never an interpolation: criterion 2 runs
#: at n = 10 and an interpolated p95 there is a confident-looking number nobody measured.
MIN_CONFIDENT_N = 20

#: Criterion 3's control threshold: the gate-off run must transcribe at least half the content
#: words, or "zero self-transcription with the gate on" is indistinguishable from a dead mic.
DEFAULT_MIN_OVERLAP = 0.5

#: The scorer's stop list (ruling A1: the implementer's, with a ten-word minimum). Casefolded.
#: Deliberately free of number words: the fixture says "one, two, three, four, five" and a stop
#: list that swallowed those would make criterion 3's overlap identically zero.
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "over",
        "s",
        "she",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

#: The C925e's ambient peak in this room is ~0.116 (T-0006); above this is signal, not the room.
HEARD_PEAK = 0.2
#: ``scripts/demo/_listen.py``'s ``TAIL_SILENCE_S``: the delayed-stream STT advances its clock
#: only on audio, so a stream that stops dead transcribes nothing at all.
TAIL_S = 3.0
#: Criterion 6, ruling A1: the residual is 10 ms x (blocks after the span opened that peak above
#: the room floor and are not marked gated), over this many dedicated gate-on playbacks.
RESIDUAL_N = 30
#: Criterion 4 fixes no n; ruling A1 states three.
NOT_OURS_N = 3


class RegimeError(ValueError):
    """A regime field is missing or empty, so no number may be printed (criterion 1)."""


class ReportError(ValueError):
    """A report row is malformed -- most sharply, a speakers number under the headset label."""


class ControlFailure(ValueError):
    """The gate-off control did not transcribe. "Silence proves silence" is criterion 3's trap."""


# --- criterion 1: the regime block ------------------------------------------------------------


def _is_empty(value: Any) -> bool:
    """Empty means None, blank text or an empty container. **Zero is a value, not emptiness.**"""
    if value is None:
        return True
    if isinstance(value, bool | int | float):
        return False
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Sequence | Mapping | set | frozenset):
        return len(value) == 0
    return False


def missing_regime_fields(regime: Any, fields: Sequence[str] = REGIME_FIELDS) -> list[str]:
    """Which of ``fields`` are absent or empty, in §Regime order."""
    if not isinstance(regime, Mapping):
        return list(fields)
    return [name for name in fields if name not in regime or _is_empty(regime[name])]


def require_regime(regime: Any, fields: Sequence[str] = REGIME_FIELDS) -> None:
    """Raise ``RegimeError`` naming every missing field, before anything is measured."""
    missing = missing_regime_fields(regime, fields)
    if missing:
        raise RegimeError(
            f"the regime block is missing or empty at {', '.join(repr(f) for f in missing)}: "
            f"a report without its regime is not a report (T-0023 criterion 1). Nothing was "
            f"measured: this is checked before any hardware time is spent (ruling A7)"
        )


def emit_report(report: Mapping[str, Any], out: TextIO, err: TextIO) -> int:
    """Write **one** JSON document whose first key is ``regime``, or write nothing at all.

    Returns 0 when the regime block is complete, non-zero otherwise -- and on the refusal path
    ``out`` is not touched: criterion 1 says print no measurement at all, so that a reader who
    sees a number knows its regime is beside it.
    """
    regime = report.get("regime") if isinstance(report, Mapping) else None
    missing = missing_regime_fields(regime)
    if missing:
        err.write(
            f"refusing to print any measurement: the regime block is missing or empty at "
            f"{', '.join(repr(name) for name in missing)} (T-0023 criterion 1; §Regime). "
            f"Zero is a value: seed 0 and distance_cm 0 are measurements, not absences.\n"
        )
        return 1
    assert isinstance(regime, Mapping)
    document: dict[str, Any] = {"regime": dict(regime)}
    for key, value in report.items():
        if key != "regime":
            document[key] = value
    json.dump(document, out, indent=1, sort_keys=False)
    out.write("\n")
    return 0


def headset_row(measurement: Mapping[str, Any] | None, *, reason: str | None = None) -> dict:
    """The headset row: a headset measurement, or ``NOT MEASURED`` and why (criterion 1).

    The substitution trap this closes is the cheap one: running the speakers pass twice and
    labelling the second one "headset". A row that carries no number cannot be mistaken for a
    measurement, and a row that carries one must say it came from a headset.
    """
    if measurement is None:
        if reason is None or not str(reason).strip():
            raise ReportError(
                "a headset row of NOT MEASURED needs its reason (P4: which headset was absent); "
                "an owed number that does not say why it is owed stops being owed (criterion 1)"
            )
        return {"transducer": "headset", "status": NOT_MEASURED, "reason": str(reason)}
    label = measurement.get("transducer")
    if label != "headset":
        raise ReportError(
            f"a measurement labelled {label!r} was offered as the headset row: a speakers number "
            f"may never appear under the headset label (criterion 1). Run the headset pass, or "
            f"record the headset row as {NOT_MEASURED!r} with its reason"
        )
    return {**dict(measurement), "status": "measured"}


# --- criterion 2's statistics -------------------------------------------------------------------


def percentiles(samples: Sequence[float]) -> dict[str, Any]:
    """p50/p95/spread/pstdev in float64, with the n < 20 p95 reported as a bound.

    Inclusive quantiles, the helper T-0006 and T-0021 already use. Under ``MIN_CONFIDENT_N``
    samples the p95 is ``max(samples)`` with ``p95_is_bound=True``: criterion 2's n = 10 has no
    95th percentile to interpolate, and a confident-looking number is worse than a bound.
    """
    values = [float(x) for x in samples]
    if not values:
        raise ValueError("percentiles() of no samples: n=0, and there is nothing to report")
    ordered = sorted(values)
    n = len(ordered)
    bound = n < MIN_CONFIDENT_N
    if n < 2:
        p50 = p95 = ordered[0]
    else:
        cuts = statistics.quantiles(ordered, n=100, method="inclusive")
        p50 = float(cuts[49])
        p95 = float(ordered[-1]) if bound else float(cuts[94])
    return {
        "p50": float(p50),
        "p95": float(p95),
        "spread": float(ordered[-1] - ordered[0]),
        "pstdev": float(statistics.pstdev(ordered)),
        "n": n,
        "p95_is_bound": bool(bound),
    }


# --- criterion 3's scorer -----------------------------------------------------------------------

_TOKEN = re.compile(r"[\w']+")

#: Digits -> the number word, by integer value (ruling A9). The fixture sentence is
#: "one, two, three, four, five" and moshi-server may transcribe it as "1 2 3 4 5"; without this
#: the gate-off control scores 0 against a transcript that is *perfectly* correct and criterion
#: 3's mandatory control fails falsely. The word is the canonical form because the spoken text
#: is words, and the report is read by a human.
NUMBER_WORDS: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
    100: "hundred",
    1000: "thousand",
}


def normalise_numeral(word: str) -> str:
    """``"5"`` and ``"05"`` -> ``"five"``; anything else is returned unchanged.

    Only whole tokens the mapping knows are rewritten: a numeral it cannot name (``"37"``) stays
    a numeral rather than being decomposed, because a wrong decomposition would silently change
    what the overlap counted.
    """
    if not word.isdigit():
        return word
    return NUMBER_WORDS.get(int(word), word)


def content_words(text: str) -> tuple[str, ...]:
    """Casefold, strip punctuation, normalise numerals, drop ``STOP_WORDS``; order preserved."""
    words: list[str] = []
    for raw in _TOKEN.findall(str(text).casefold()):
        word = normalise_numeral(raw.strip("'_"))
        if word and word not in STOP_WORDS:
            words.append(word)
    return tuple(words)


def overlap_fraction(spoken: str, transcript: str) -> float:
    """|content(transcript) & content(spoken)| / |content(spoken)| -- **set** semantics.

    Sets, so repeating one word cannot inflate the score; and because ten utterances of the same
    sentence run back to back, the previous transcript must score 0 against this one -- which
    only holds once the stop words are gone.
    """
    want = set(content_words(spoken))
    if not want:
        raise ValueError(
            f"the spoken text {spoken!r} has no content words: there is no denominator, so the "
            f"overlap is undefined rather than zero (criterion 3)"
        )
    return len(want & set(content_words(transcript))) / len(want)


def control_overlap(
    spoken: str, transcript: str, *, min_overlap: float = DEFAULT_MIN_OVERLAP
) -> float:
    """The gate-off control's score -- and an exception, never 0.0, when it heard nothing.

    This is the function criterion 3 exists for: without it "the STT produced nothing" is
    indistinguishable from a dead microphone, a dead STT and a gate stuck shut, and the headline
    "zero self-transcription" means nothing at all.
    """
    if not content_words(transcript):
        raise ControlFailure(
            f"the gate-off control transcribed no content words at all ({transcript!r}): the "
            f"microphone, the STT or the playback is dead, so the gate-on run proves nothing "
            f"(T-0023 criterion 3 -- this errors, it never passes)"
        )
    fraction = overlap_fraction(spoken, transcript)
    if fraction < min_overlap:
        raise ControlFailure(
            f"the gate-off control heard only {fraction:.2f} of the spoken content words "
            f"(want >= {min_overlap:.2f}, transcript {transcript!r}): nothing measured with the "
            f"gate on means anything until the control passes (criterion 3)"
        )
    return fraction


# --- the rig (nothing above this line touches a device) -----------------------------------------


def _capture(cmd: Sequence[str], timeout: float = 15.0) -> tuple[int, str]:
    """Run a command for a regime field. A missing tool is an empty answer, never a guess."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout.strip()


def _text(cmd: Sequence[str], timeout: float = 15.0) -> str:
    code, out = _capture(cmd, timeout)
    return out if code == 0 else ""


def node_ids(name: str) -> list[int]:
    code, out = _capture(["pw-dump"])
    if code != 0 or not out:
        return []
    return [
        int(obj["id"])
        for obj in json.loads(out)
        if obj.get("type") == "PipeWire:Interface:Node"
        and obj.get("info", {}).get("props", {}).get("node.name") == name
    ]


def default_source() -> str:
    """The first ALSA capture node in ``pw-dump`` -- the C925e is the only mic on this box."""
    code, out = _capture(["pw-dump"])
    if code != 0 or not out:
        return ""
    for obj in json.loads(out):
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        name = obj.get("info", {}).get("props", {}).get("node.name")
        if isinstance(name, str) and name.startswith("alsa_input."):
            return name
    return ""


def sink_volume(sink: str) -> str:
    for node_id in node_ids(sink):
        out = _text(["wpctl", "get-volume", str(node_id)])
        if out:
            return out.replace("Volume:", "").strip()
    return os.environ.get("ASTRA_SINK_VOLUME", "")


def pipewire_version() -> str:
    lines = _text(["pipewire", "--version"]).splitlines()
    return " / ".join(line.strip() for line in lines if line.strip())


def co_resident() -> str:
    """What else was running. Contention moves every number here, so it is a regime field."""
    stated = os.environ.get("ASTRA_CO_RESIDENT")
    if stated:
        return stated
    code, out = _capture(["docker", "ps", "--format", "{{.Names}}"])
    if code != 0:
        return ""
    names = out.split()
    vllm = "on" if any("vllm" in name for name in names) else "off"
    stt = "on" if any("stt" in name for name in names) else "off"
    return f"vllm={vllm} astra-stt={stt}"


def stt_device(explicit: str | None = None) -> str:
    """Which device the **server** is running on, from the server itself (ruling A7).

    Never from this process's ``ASTRA_STT_CPU``: that variable says how somebody meant to start
    the service, not how the service that is answering on 8080 was actually started (it may have
    been started in another shell, yesterday, on the GPU). ``docker inspect`` reads the running
    container's own argv, where ``services/stt.sh`` puts ``--cpu``; an explicit ``--stt-device``
    overrides it. Anything else leaves the field empty and the report refuses.
    """
    if explicit:
        return explicit
    code, out = _capture(["docker", "inspect", "--format", "{{json .Args}}", "astra-stt"])
    if code != 0 or not out:
        return ""
    try:
        args = json.loads(out)
    except json.JSONDecodeError:
        return ""
    return "cpu" if "--cpu" in [str(a) for a in args] else "gpu"


def stt_regime(device: str | None = None) -> dict[str, Any]:
    """The STT half of the regime, read from ``services/`` and the server, never invented.

    A field this cannot establish comes back empty and the run refuses naming it -- in
    particular ``batch_size``: a missed TOML regex is an unknown batch size, not a batch size of
    zero, and zero would sail through the "zero is a value" rule (ruling A7).
    """
    lock = (REPO / "services" / "versions.lock").read_text()
    config = (REPO / "services" / "moshi" / "stt-1b-en_fr.toml").read_text()
    server = re.search(r'moshi_server\s*=\s*"([^"]+)"', lock)
    batch = re.search(r"^\s*batch_size\s*=\s*(\d+)", config, re.MULTILINE)
    revision = os.environ.get("ASTRA_STT_MODEL_REVISION", "")
    if not revision:
        ref = Path.home() / ".cache/huggingface/hub/models--kyutai--stt-1b-en_fr-candle/refs/main"
        revision = ref.read_text().strip() if ref.exists() else ""
    return {
        "server": f"moshi-server {server.group(1)}" if server else "",
        "model_revision": revision,
        "device": stt_device(device),
        "batch_size": int(batch.group(1)) if batch else "",
    }


def git_sha() -> str:
    # ``-C REPO``: the sha is the tree the scripts and the module came from, not whatever
    # repository the operator happened to be standing in when they launched (ruling A13).
    return _text(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])


def refuse_dirty_tree() -> None:
    """CLAUDE.md's first observed mistake: a number taken while the tree moves is unattributable."""
    code, out = _capture(["git", "-C", str(REPO), "status", "--porcelain"])
    if code != 0:
        raise RegimeError(
            "git status failed: the tree cannot be shown clean, so nothing is measured"
        )
    if out:
        raise RegimeError(
            f"the working tree is dirty:\n{out}\nMeasuring while local_astra/ moves gives a "
            f"number nobody can attribute (CLAUDE.md). Commit or stash first."
        )


def read_wav(path: Path) -> tuple[Any, int]:
    """Stdlib reader (no soundfile): mono s16le -> float64 in [-1, 1]."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def chirp_reference_48k(seconds: float = 2.0) -> Any:
    """The first ``seconds`` of the chirp fixture at the device rate, as a correlation template."""
    import numpy as np

    x, rate = read_wav(CHIRP)
    if rate != TTS_RATE:
        raise ReportError(f"{CHIRP.name} is {rate} Hz, expected {TTS_RATE}")
    up = np.interp(np.arange(2 * x.size) / 2.0, np.arange(x.size), x)
    return up[: round(seconds * DEVICE_RATE)]


def fft_best_lag(reference: Any, signal: Any) -> int:
    """Lag in samples that maximises the cross-correlation, via rfft (float64)."""
    import numpy as np

    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(signal, dtype=np.float64)
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    return int(np.argmax(corr[: b.size]))


def peak(x: Any) -> float:
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


class Recording:
    """``pw-record`` on a node for the duration of a with-block.

    ``capture_sink`` is the difference between an oracle and a lie (T-0006 ruling A7): without
    ``-P stream.capture.sink=true`` the session manager links the stream to the default *source*
    -- on this box the webcam mic, whose room noise sails past any energy floor. Pass False when
    the target really is a capture device.
    """

    def __init__(self, target: str, path: Path, *, capture_sink: bool) -> None:
        self.target = target
        self.path = path
        self.capture_sink = capture_sink
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Recording:
        command = ["pw-record", "--target", self.target, "--rate", str(DEVICE_RATE)]
        if self.capture_sink:
            command += ["-P", "stream.capture.sink=true"]
        self._proc = subprocess.Popen(
            [*command, str(self.path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        time.sleep(0.3)  # let the stream connect before the first sample is played
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            time.sleep(0.2)
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def read(self) -> tuple[Any, int]:
        x, rate = read_wav(self.path)
        if x.size == 0:
            raise ReportError(f"pw-record captured nothing from {self.target!r}: the rig is dead")
        return x, rate


def play_through(sink: str, path: Path, timeout: float = 60.0) -> None:
    """``pw-play`` into a sink: audio that is **not** ours, so no rendering span ever opens."""
    proc = subprocess.Popen(
        ["pw-play", "--target", sink, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if proc.wait(timeout=timeout) != 0:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode()[:400]
        raise ReportError(f"pw-play could not play into {sink!r}: {stderr}")


async def measure_noise_floor(sink: str, source: str, seconds: float = 1.5) -> float:
    """The room's ambient peak, taken **through the module's own capture** (ruling A13).

    Not through ``pw-record``: the C925e runs automatic gain, and a floor measured on one stream
    is not the floor the next stream sees -- the gain ramps over the first seconds of each new
    client. The residual (criterion 6) compares block peaks against this number through
    PortAudio, so the floor is taken through PortAudio too, on the same device object the
    measurement will use.
    """
    io, device = build_io(sink, source)
    try:
        with collecting_blocks(io) as collected:
            await asyncio.sleep(seconds)
            blocks = list(collected)
    finally:
        close_io(io, device)
    if not blocks:
        raise ReportError(
            "the capture tap produced nothing while measuring the room floor: there is no floor "
            "to compare the residual against (P1/P3 -- is the mic there and unmuted?)"
        )
    return max(peak(block.samples) for block in blocks)


def calibrate(sink: str, source: str, tmp: Path) -> float:
    """T-0006 criterion 11, first: a named sink that is silent **fails**, it never skips."""
    with Recording(source, tmp / "calib.wav", capture_sink=False) as rec:
        play_through(sink, CHIRP)
    heard, _rate = rec.read()
    level = peak(heard)
    if level <= HEARD_PEAK:
        raise ReportError(
            f"the calibration chirp played through {sink!r} peaks at {level:.3f} on {source!r}, "
            f"at or below the {HEARD_PEAK} this script calls 'heard'. The sink is named and "
            f"silent: muted, unplugged, at zero volume, or not connected to a loudspeaker (P2). "
            f"Every echo and gate number taken on this rig would be a lie."
        )
    return level


# --- the module under test on the real device ---------------------------------------------------


def open_acoustic_device(sink: str, source: str) -> Any:
    """The real backend. A missing ``libportaudio2`` is prerequisite P1, and it is an error."""
    from local_astra.contracts.speech import CAPTURE_BLOCK_MS

    try:
        from local_astra.conductor.audio_device import open_device

        return open_device(sink=sink, source=source, block_ms=CAPTURE_BLOCK_MS, latency_ms=30.0)
    except (OSError, RuntimeError) as exc:
        raise ReportError(
            f"prerequisite P1: the backend needs libportaudio2 and it is not installed "
            f"(`sudo apt-get install -y libportaudio2`). Nothing in T-0023 runs without it: {exc}"
        ) from exc


def build_io(sink: str, source: str, *, gate: Any | None = None) -> tuple[Any, Any]:
    """``(io, device)`` on the real duplex device; the device carries the one clock (ruling A4)."""
    from local_astra.conductor.audio_io import PipeWireAudioIO

    device = open_acoustic_device(sink, source)
    return PipeWireAudioIO(device, **({} if gate is None else {"gate": gate})), device


def close_io(io: Any, device: Any) -> None:
    """End one measurement: the capture iterators finish **and** the device stops (ruling A9).

    ``close_capture()`` alone leaves the PortAudio stream open and its callback running, so the
    next measurement opens a second stream on the same node -- the graph re-negotiates, the
    quantum in the regime stops describing the run, and the leaked writer thread keeps the sink
    busy. Every measurement here opens its own device, so every measurement stops it.
    """
    io.close_capture()
    device.stop()


def gate_off() -> Any:
    """Criterion 3's control: a gate that never closes, same tail policy as the gated run."""
    from local_astra.conductor.audio_capture import SelfSpeechGate

    return SelfSpeechGate(lambda _t_ms: False, 0.0)


def wav_utterance(path: Path, uid: str) -> Any:
    """A minimal ``TtsUtterance`` over a 24 kHz s16le fixture, in 80 ms chunks."""
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != TTS_RATE:
            raise ReportError(f"{path.name} is {handle.getframerate()} Hz, expected {TTS_RATE}")
        raw = handle.readframes(handle.getnframes())

    class WavUtterance:
        utterance_id = uid
        text = ""
        first_audio_latency_ms = None

        async def audio(self) -> Any:
            step = 2 * 80 * TTS_RATE // 1000
            for start in range(0, len(raw), step):
                yield raw[start : start + step]

        def words(self) -> None:
            return None

        async def cancel(self) -> None:
            return None

    return WavUtterance()


def spoken_sentence() -> tuple[str, float]:
    """The fixture's text and its duration in seconds (T-0006's fixture, MANIFEST.md)."""
    meta = json.loads(SENTENCE_WORDS.read_text())
    return str(meta["text"]), float(meta["duration_ms"]) / 1000.0


async def transcribe(io: Any, *, play_now: Any, listen_s: float, tail_s: float = TAIL_S) -> str:
    """Feed ``stt_blocks()`` straight into ``MoshiSttClient.feed()``; return what came back.

    No resampling, no reframing, no retiming (T-0021 criterion 7): the tap already is 24 kHz f32
    in 1920-sample frames, and feeding the endpoint anything else produces a plausible-looking
    transcript of nonsense. The transcript is read only after ``tail_s`` of **real captured**
    tail, the same policy for the gated run and for the control.
    """
    import numpy as np

    from local_astra.conductor.stt_client import MoshiSttClient

    client = MoshiSttClient()
    words: list[str] = []
    frames = 0

    async def pump() -> None:
        nonlocal frames
        async for frame in io.stt_blocks():
            await client.feed(
                np.asarray(frame.samples, dtype=np.float32).tobytes(), int(frame.t_ms)
            )
            frames += 1

    async def collect() -> None:
        async for event in client.events():
            if getattr(event, "type", None) == "word":
                words.append(str(event.text))

    pump_task = asyncio.create_task(pump())
    collect_task = asyncio.create_task(collect())
    try:
        if play_now is not None:
            await play_now()
        await asyncio.sleep(listen_s + tail_s)
    finally:
        pump_task.cancel()
        collect_task.cancel()
        await asyncio.gather(pump_task, collect_task, return_exceptions=True)
        await client.close()
    if frames == 0:
        raise ReportError("no STT frames were produced at all: the capture tap is dead, not gated")
    return " ".join(words)


@contextmanager
def collecting_blocks(io: Any) -> Iterator[list[Any]]:
    """Collect the **raw** capture blocks (which carry ``gated``) for the duration of the block."""
    blocks: list[Any] = []

    async def pump() -> None:
        async for block in io.capture_frames():
            blocks.append(block)

    task = asyncio.ensure_future(pump())
    try:
        yield blocks
    finally:
        task.cancel()


# --- the measurements ---------------------------------------------------------------------------


def capture_timeline(blocks: Sequence[Any], *, block_ms: float) -> tuple[Any, float]:
    """The raw blocks as one array plus the capture time of its first sample.

    Refuses on a hole: a gap in ``t_ms`` breaks the sample-index -> capture-clock mapping the
    delay is read off, and splicing across it would move the correlation peak by exactly the
    audio that never arrived (T-0021 criterion 4 marks those blocks; here they are fatal). The
    oracle in ``tests/fakes/audio_device.py`` concatenates without this check because its stub
    room has no holes; on the hardware a hole is the one way this measurement lies quietly.
    """
    import numpy as np

    rows = list(blocks)
    if not rows:
        raise ReportError("no capture blocks at all: the duplex device captured nothing")
    tolerance_ms = 0.5 * 1000.0 / DEVICE_RATE
    for index in range(1, len(rows)):
        step = float(rows[index].t_ms) - float(rows[index - 1].t_ms)
        if abs(step - block_ms) > tolerance_ms or bool(getattr(rows[index], "xrun", False)):
            raise ReportError(
                f"the capture stream has a hole at block {index} (t_ms stepped {step:.3f} ms, "
                f"not {block_ms:.3f}): sample index no longer maps to the capture clock, so the "
                f"correlation peak cannot be dated (criterion 2)"
            )
    samples = np.concatenate([np.asarray(row.samples, dtype=np.float64) for row in rows])
    return samples, float(rows[0].t_ms)


def rendering_span_start_ms(io: Any, after_ms: float, window_ms: float = 3000.0) -> float:
    """The first capture-clock instant at or after ``after_ms`` at which we were rendering.

    The mirror of ``tests/fakes/audio_device.rendering_span_start_ms`` (ruling A4), which
    ``tests/unit/test_acoustic_delay_method.py`` pins against a synthetic span: read through the
    **public** ``is_rendering(t_ms)`` -- the one question the self-speech gate asks -- so the
    number is the span start as the gate sees it and not a private field. Resolution is half a
    block, which is finer than anything upstream of it.
    """
    from local_astra.contracts.speech import CAPTURE_BLOCK_MS

    step = 0.5 * float(CAPTURE_BLOCK_MS)
    t = float(after_ms)
    stop = t + float(window_ms)
    while t < stop:
        if bool(io.is_rendering(t)):
            return t
        t += step
    raise ReportError(
        f"no rendering span opened in the {window_ms:.0f} ms after {after_ms:.1f} ms: play() "
        f"never handed audio to the device, so there is nothing to measure the echo against"
    )


async def acoustic_delay_ms(
    io: Any,
    device: Any,
    utterance: Any,
    *,
    settle_s: float = 0.3,
    listen_s: float = 3.0,
    template_s: float = 2.0,
    heard_peak: float = HEARD_PEAK,
) -> tuple[float, float]:
    """DAC -> air -> ADC for one chirp, on the module's own clocks (ruling A4).

    Returns ``(delay_ms, peak)``. The mirror of the helper of the same name in
    ``tests/fakes/audio_device.py``, which ``tests/unit/test_acoustic_delay_method.py`` checks
    against a hand-built 137.5 ms echo: the chirp goes out through ``io.play()`` and comes back
    through the **same duplex device's** capture, so both ends are stamped on one clock --

        delay = (the correlation peak on the capture clock) - (the rendering span's start)

    -- and nothing external times anything. The method this replaced, two ``pw-record``
    processes started 0.3 s apart with the monitor subtracted from the mic, had no shared time
    origin: a sink that resumes late turns their start jitter into a plausible ~300 ms "echo
    delay", and the hangover derived from it would blow criterion 5's 200 ms bound for a reason
    that is not in the room. ``pw-record`` survives here only inside the audible-rig guard.
    """
    from local_astra.contracts.speech import CAPTURE_BLOCK_MS

    with collecting_blocks(io) as collected:
        await asyncio.sleep(settle_s)  # capture running, so the span start is inside the window
        before_ms = float(device.captured_ms())
        # No ``mark_drain()`` here, deliberately: this is the mirror of the pinned helper, which
        # does not drain either. ``one_delay`` stops the device the moment it returns, and a run
        # that never drains cannot bias a delay read off the capture clock.
        await io.play(utterance, source="chirp")
        await asyncio.sleep(listen_s)
        blocks = list(collected)
    heard, first_ms = capture_timeline(blocks, block_ms=float(CAPTURE_BLOCK_MS))
    loudest = peak(heard)
    if loudest <= heard_peak:
        raise ReportError(
            f"nothing audible came back: the capture peaks at {loudest:.3f} (floor {heard_peak}) "
            f"while the chirp was playing -- the rig went silent mid-measurement (criterion 2)"
        )
    lag = fft_best_lag(chirp_reference_48k(template_s), heard)
    heard_ms = first_ms + 1000.0 * lag / DEVICE_RATE
    span_start_ms = rendering_span_start_ms(io, before_ms)
    return heard_ms - span_start_ms, loudest


async def one_delay(sink: str, source: str, index: int) -> dict[str, Any]:
    """One chirp through the module and back through the room, on its own device (ruling A9)."""
    io, device = build_io(sink, source)
    try:
        delay, loudest = await acoustic_delay_ms(io, device, wav_utterance(CHIRP, f"chirp{index}"))
    finally:
        close_io(io, device)
    if not 0.0 < delay <= 1000.0:
        raise ReportError(
            f"run {index}: an echo-path delay of {delay:.1f} ms is not physical (capture peak "
            f"{loudest:.3f}) -- the correlation peak did not find the chirp, or the rendering "
            f"span found was not this chirp's (criterion 2)"
        )
    return {"delay_ms": delay, "peak": loudest}


async def measure_delay(
    args: argparse.Namespace, sink: str, source: str, regime: Mapping[str, Any]
) -> dict:
    """Criterion 2: n chirps, one transducer, p50/p95/spread/pstdev with its own regime block."""
    runs = [await one_delay(sink, source, i) for i in range(args.n)]
    delays = [float(run["delay_ms"]) for run in runs]
    stats = percentiles(delays)
    wander = statistics.pstdev(delays)
    if wander > 50.0:
        raise ReportError(
            f"the echo-path delay wanders by {wander:.1f} ms over {args.n} runs "
            f"({args.transducer}, {args.distance_cm:g} cm): a hangover derived from it would be "
            f"meaningless (criterion 2)"
        )
    row: dict[str, Any] = {
        **stats,
        "transducer": args.transducer,
        "method": "module clocks: capture-clock correlation peak minus the rendering span start",
        "peak": max(float(run["peak"]) for run in runs),
        "regime": dict(regime),
    }
    return {args.transducer: headset_row(row) if args.transducer == "headset" else row}


async def measure_gate(
    args: argparse.Namespace, sink: str, source: str, regime: Mapping[str, Any]
) -> dict:
    """Criteria 3 and 4: gate on, the mandatory gate-off control, and audio that is not ours.

    Three sections, each with its own regime block whose ``gate`` field is ``on`` or ``off`` and
    nothing else (ruling A9): a run that covers both conditions writes both sections rather than
    one section labelled with a value no gate ever had.
    """
    spoken, listen_s = spoken_sentence()
    on_regime = {**dict(regime), "gate": "on"}
    off_regime = {**dict(regime), "gate": "off"}

    async def ours(io: Any) -> None:
        await io.play(wav_utterance(SENTENCE, "sentence"), source="fixture")
        io.mark_drain()

    gated: list[float] = []
    spurious = 0
    for _ in range(args.n):
        io, device = build_io(sink, source)
        try:
            text = await transcribe(io, play_now=lambda io=io: ours(io), listen_s=listen_s)
        finally:
            close_io(io, device)
        gated.append(overlap_fraction(spoken, text))
        spurious += len(set(content_words(text)) - set(content_words(spoken)))

    control: list[float] = []
    for _ in range(args.n):
        io, device = build_io(sink, source, gate=gate_off())
        try:
            text = await transcribe(io, play_now=lambda io=io: ours(io), listen_s=listen_s)
        finally:
            close_io(io, device)
        # raises ControlFailure -- it never returns 0.0 for silence (criterion 3)
        control.append(control_overlap(spoken, text))

    not_ours: list[float] = []
    not_ours_gated: list[int] = []
    for _ in range(NOT_OURS_N):
        io, device = build_io(sink, source)

        async def someone_else() -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, play_through, sink, SENTENCE)

        try:
            text = await transcribe(io, play_now=someone_else, listen_s=listen_s + 0.5)
            stats = io.capture_stats()
        finally:
            close_io(io, device)
        not_ours.append(overlap_fraction(spoken, text))
        not_ours_gated.append(int(stats.gated_blocks))

    return {
        "on": {
            "overlap": max(gated),
            "spurious_words": spurious,
            "n": len(gated),
            "regime": on_regime,
        },
        "off": {"overlap": min(control), "n": len(control), "regime": off_regime},
        "not_ours": {
            "overlap": min(not_ours),
            "n": len(not_ours),
            "gated_blocks": max(not_ours_gated),
            "regime": on_regime,
        },
    }


def leading_edge_ms(blocks: Sequence[Any], span_start_ms: float, floor: float) -> float:
    """Criterion 6's residual for one utterance: the **leading edge only** (ruling A11).

    The window is the rendering span's *start* (A1's anchor, read through ``is_rendering()``)
    up to the **first gated block** of this utterance; inside it, the echo-carrying blocks --
    those peaking above the room floor -- that reached the tap ungated, at 10 ms each.

    Two things this deliberately does not do. It does not run to the end of the utterance: once
    the gate has closed, everything after it is the hangover's business (a trailing-edge device,
    T-0021 ruling A16c), and counting it here would blame the span start for the hangover's work
    and vice versa. And it does not start from ``captured_ms()`` before ``play()``: that is when
    the *script* asked for audio, not when the module opened the span, and the difference is the
    queued lead -- exactly the term the measurement is about.

    Zero is a legitimate answer: a gate that closes before any echo arrives leaks nothing.
    """
    from local_astra.contracts.speech import CAPTURE_BLOCK_MS

    leaked = 0
    for block in blocks:
        if float(block.t_ms) < span_start_ms:
            continue
        if bool(getattr(block, "gated", False)):
            break  # the gate has closed: the leading edge is over
        if peak(block.samples) > floor:
            leaked += 1
    return leaked * float(CAPTURE_BLOCK_MS)


async def measure_residual(
    args: argparse.Namespace, sink: str, source: str, floor: float, regime: Mapping[str, Any]
) -> dict:
    """Criterion 6: the ungated echo at the leading edge, over ``--residual-n`` playbacks.

    A block is echo-carrying when it peaks above **the room floor** measured in this session
    (ruling A9), not above a fixed 0.2: the floor is what the room contributed, and anything
    above it during our own playback is our own voice coming back.

    The hangover does not and must not cover this: echo that arrives before the rendering span
    opens is a span-start problem (T-0021 ruling A20), and raising the hangover would move the
    defect instead of removing it.
    """
    _spoken, listen_s = spoken_sentence()
    residuals: list[float] = []
    for index in range(args.residual_n):
        io, device = build_io(sink, source)
        try:
            with collecting_blocks(io) as collected:
                await asyncio.sleep(0.5)  # a quiet head: the onset below is echo, not a tail
                before_ms = float(device.captured_ms())
                await io.play(wav_utterance(SENTENCE, "sentence"), source="fixture")
                io.mark_drain()
                # Listening time, **not** the counting window (ruling A11): we listen through
                # the utterance so the dead-room guard below has something to judge, and count
                # only from the span start to the first gated block.
                await asyncio.sleep(listen_s + 0.5)
                blocks = list(collected)
            span_start_ms = rendering_span_start_ms(io, before_ms)
        finally:
            close_io(io, device)
        if not blocks:
            raise ReportError(f"run {index}: no capture blocks at all during the playback")
        during = [row for row in blocks if float(row.t_ms) >= span_start_ms]
        if not any(peak(row.samples) > floor for row in during):
            raise ReportError(
                f"run {index}: no block peaked above the room floor {floor:.3f} while we were "
                f"rendering -- the room went silent, so a residual of 0 ms here would mean the "
                f"rig is dead, not that the gate is tight (criterion 2's guard)"
            )
        if not any(bool(getattr(row, "gated", False)) for row in during):
            raise ReportError(
                f"run {index}: not one block was gated while we were rendering (span start "
                f"{span_start_ms:.1f} ms, {len(during)} blocks). The leading edge has no end, so "
                f"there is no residual to measure -- the gate never closed (criteria 4 and 6)"
            )
        residuals.append(leading_edge_ms(blocks, span_start_ms, floor))
    stats = percentiles(residuals)
    return {
        "p50": stats["p50"],
        "p95": stats["p95"],
        "n": stats["n"],
        "floor": floor,
        "window": "rendering span start -> first gated block (leading edge only, ruling A11)",
        "regime": {**dict(regime), "gate": "on"},
    }


# --- the report ----------------------------------------------------------------------------------


def device_lag_facts(device: Any) -> dict[str, Any]:
    """The negotiated quantum, and the DAC lag **only if the callback measured one** (ruling A6).

    ``PortAudioDevice._dac_lag_frames`` starts at the *negotiated* latency and is overwritten by
    every callback with ``outputBufferDacTime - currentTime``. Reading it without asking whether
    it moved reports the request wearing a measurement's label -- exactly what
    ``measured_dac_lag_ms`` exists to rule out. So: unchanged from the negotiated value (or
    absent) means the field stays empty, and criterion 1 then refuses the whole report naming it.
    T-0006 measured 15.3 ms against a negotiated 10 ms, so a real callback does move it.
    """
    lag_frames = getattr(device, "_dac_lag_frames", None)
    negotiated_frames = round(float(device.latency_ms) * DEVICE_RATE / 1000.0)
    moved = lag_frames is not None and int(lag_frames) != int(negotiated_frames)
    return {
        "negotiated_quantum_ms": float(device.quantum_ms),
        "measured_dac_lag_ms": 1000.0 * float(lag_frames) / DEVICE_RATE if moved else "",
    }


async def device_facts(sink: str, source: str) -> dict[str, Any]:
    """Open the device long enough for its callback to report a quantum and a DAC lag.

    Silence is written, not a fixture: this runs before the chirps of ``--measure delay``, and
    the numbers it collects are properties of the graph, not of what was played through it. The
    writes matter -- a stream with nothing to render still runs its callback, but writing is what
    makes this the same path the measurements use.
    """
    import numpy as np

    device = open_acoustic_device(sink, source)
    try:
        device.start()
        device.start_capture()
        for _ in range(8):
            device.write(np.zeros(DEVICE_RATE // 100, dtype=np.float32))
        # Awaited, not slept: this runs inside the event loop that drives capture, and a
        # blocking sleep there stalls every collector task with it (ruling A13).
        await asyncio.sleep(0.3)
        return device_lag_facts(device)
    finally:
        device.stop_capture()
        device.stop()


def build_regime(
    args: argparse.Namespace,
    sink: str,
    source: str,
    *,
    gate: str,
    stt: Mapping[str, Any] | None = None,
    floor: Any = "",
    duration_s: Any = "",
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every field of §Regime that is knowable now; the measured four are filled by the run.

    Called **before** any hardware time (ruling A7) with ``floor``/``duration_s``/``facts``
    empty, so a missing ``git_sha`` or an unreadable ``batch_size`` costs a second instead of
    ten minutes; the run then re-calls it with the measured values.
    """
    from local_astra.conductor.audio_capture import DEFAULT_HANGOVER_MS

    block = dict(stt) if stt is not None else stt_regime(getattr(args, "stt_device", None))
    blank = [key for key, value in block.items() if _is_empty(value)]
    if blank:
        raise RegimeError(
            f"the 'stt' regime field is incomplete at {blank}: the model revision comes from "
            f"$ASTRA_STT_MODEL_REVISION or the HF cache, the device from the running server "
            f"(docker inspect astra-stt) or --stt-device, and the batch size from "
            f"services/moshi/stt-1b-en_fr.toml -- an unknown batch size is empty, never 0 "
            f"(criterion 1, ruling A7)"
        )
    return {
        "box": platform.node(),
        "pipewire_version": pipewire_version(),
        "sink": sink,
        "sink_volume": sink_volume(sink),
        "source": source,
        "transducer": args.transducer,
        "distance_cm": args.distance_cm,
        "noise_floor_peak": floor,
        "stt": block,
        "co_resident": co_resident(),
        "profile": os.environ.get("ASTRA_PROFILE", "lean"),
        "gate": gate,
        "hangover_ms": float(DEFAULT_HANGOVER_MS),
        "n": args.n,
        "seed": args.seed,
        "git_sha": git_sha(),
        "duration_s": duration_s,
        **dict(facts or {}),
    }


def merge_report(previous: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a new section into an earlier report, keeping every measurement already taken.

    The stages of docs/tasks/T-0023.md run one at a time, sometimes hours apart and always with
    a different regime line, so two rules hold (ruling A8):

    * an owed ``NOT MEASURED`` row never overwrites a number somebody actually took, while a
      fresh *measurement* does replace an older one -- re-running a stage after a fix is the
      point of merging at all;
    * every section carries **its own** regime, because criterion 1 is per number. A section the
      fresh run handed over without one is stamped with that run's regime here; a section carried
      over from an earlier run is never re-stamped, or a delay taken at 90 cm with the gate off
      would inherit the gate run's line merely because that run happened later.
    """
    fresh_regime = dict(fresh.get("regime") or {})

    def stamped(row: Any) -> Any:
        """Give a fresh section the regime of the run that produced it, if it brought none."""
        if not isinstance(row, Mapping) or "regime" in row:
            return row
        if row.get("status") == NOT_MEASURED:
            return row  # an owed row is not a measurement and has no regime to state
        return {**dict(row), "regime": fresh_regime}

    merged = {key: value for key, value in previous.items() if key != "regime"}
    for key, value in fresh.items():
        if key == "regime":
            continue
        if key == "delay_ms" and isinstance(value, Mapping):
            rows = dict(merged.get("delay_ms") or {})
            for transducer, row in value.items():
                old = rows.get(transducer)
                new_is_owed = isinstance(row, Mapping) and row.get("status") == NOT_MEASURED
                old_is_measured = isinstance(old, Mapping) and old.get("status") != NOT_MEASURED
                if new_is_owed and old_is_measured:
                    continue
                rows[transducer] = stamped(row)
            merged["delay_ms"] = rows
        else:
            merged[key] = stamped(value)
    return {"regime": dict(fresh_regime or previous.get("regime") or {}), **merged}


def resolve_sink(args: argparse.Namespace) -> str:
    env = "ASTRA_ACOUSTIC_HEADSET_SINK" if args.transducer == "headset" else "ASTRA_ACOUSTIC_SINK"
    sink = args.sink or os.environ.get(env, "")
    if not sink:
        raise RegimeError(
            f"no sink: pass --sink or export {env} (P2 -- the default sink is a Dummy Output "
            f"while the DP cable is unplugged, and a number measured through it is a lie; the "
            f"headset pass is P4)"
        )
    if not node_ids(sink):
        raise ReportError(
            f"the sink {sink!r} is not in pw-dump: a named sink that is not there is a "
            f"misconfiguration, not an absence, and this run refuses rather than skips"
        )
    return sink


async def run(args: argparse.Namespace) -> int:
    refuse_dirty_tree()
    sink = resolve_sink(args)
    source = args.source or default_source()
    if not source:
        raise RegimeError("no capture node in pw-dump (P3: the Logitech C925e is the only mic)")

    # Ruling A7: the regime is assembled and validated before a single sample is played. Only
    # the four fields the run itself produces (the floor, the duration, the negotiated quantum
    # and the measured DAC lag) are missing here, and they are checked again at the end. A run
    # that cannot be reported must fail in a second, not after ten minutes of room.
    # The document's gate is ``on`` for every measurement here: the module runs with its real
    # gate installed throughout. The gate-**off** control is one section, with its own regime
    # block saying so -- ``gate`` never takes a third value like "n/a" or "on+off" (ruling A9).
    static = build_regime(args, sink, source, gate="on")
    require_regime(static, STATIC_REGIME_FIELDS)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="astra-acoustic-") as tmpdir:
        tmp = Path(tmpdir)
        floor = await measure_noise_floor(sink, source)
        if floor >= HEARD_PEAK:
            raise ReportError(
                f"the room's noise floor peaks at {floor:.3f} on {source!r}, at or above the "
                f"{HEARD_PEAK} this script calls 'heard' (P5 wants a quiet room, or every "
                f"number here is noise)"
            )
        # pw-play and pw-record block for the length of the chirp, so the guard runs in a
        # worker thread rather than on the loop (ruling A13). It times nothing: it answers the
        # one question "does this sink reach that microphone at all" (T-0006 criterion 11).
        calibrated = await asyncio.to_thread(calibrate, sink, source, tmp)
        facts = await device_facts(sink, source)
        # The section regimes carry what is known now; ``duration_s`` is the whole run's and is
        # the same in each (criterion 1 is per number, ruling A8).
        section_regime = {**static, "noise_floor_peak": floor, **facts}
        section: dict[str, Any]
        if args.measure == "delay":
            # The delay is read off the **raw** stream, which the gate never touches; the gate
            # is nonetheless on, because that is the configuration every other number uses.
            section = {"delay_ms": await measure_delay(args, sink, source, section_regime)}
            if args.transducer == "speakers" and not os.environ.get("ASTRA_ACOUSTIC_HEADSET_SINK"):
                section["delay_ms"]["headset"] = headset_row(
                    None,
                    reason="P4: no headset sink (ASTRA_ACOUSTIC_HEADSET_SINK unset) on this box",
                )
        elif args.measure == "gate":
            section = {
                "gate": await measure_gate(args, sink, source, section_regime),
                "leading_edge_ms": await measure_residual(
                    args, sink, source, floor, section_regime
                ),
            }
        else:
            section = {
                "leading_edge_ms": await measure_residual(args, sink, source, floor, section_regime)
            }

    duration_s = time.monotonic() - started
    regime = {**section_regime, "duration_s": duration_s}
    for row in section.values():
        if isinstance(row, dict):
            for inner in row.values():
                if isinstance(inner, dict) and isinstance(inner.get("regime"), dict):
                    inner["regime"]["duration_s"] = duration_s
            if isinstance(row.get("regime"), dict):
                row["regime"]["duration_s"] = duration_s
    # Named before anything is written: emit_report refuses too, but a run that cannot be
    # reported should say which field is missing without leaving a half-written file behind.
    require_regime(regime)
    out_path = Path(args.out) if args.out else None
    previous: dict[str, Any] = {}
    if out_path is not None and out_path.exists():
        previous = dict(json.loads(out_path.read_text()))
    report = merge_report(previous, {"regime": regime, **section})

    buffer = StringIO()
    code = emit_report(report, buffer, sys.stderr)
    if code != 0:
        return code
    if out_path is None:
        sys.stdout.write(buffer.getvalue())
    else:
        out_path.write_text(buffer.getvalue())
    print(
        f"acoustic {args.measure}: sink={sink!r} source={source!r} "
        f"transducer={args.transducer} distance={args.distance_cm:g} cm n={args.n} "
        f"seed={args.seed} gate={regime['gate']} noise_floor_peak={floor:.3f} "
        f"calibration_peak={calibrated:.3f} quantum={regime['negotiated_quantum_ms']} ms "
        f"dac_lag={regime['measured_dac_lag_ms']} ms co_resident={regime['co_resident']!r} "
        f"git={regime['git_sha']} -> {out_path if out_path is not None else 'stdout'}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T-0023 criteria 2-6: the acoustic measurements.")
    parser.add_argument("--measure", choices=["delay", "gate", "residual"], required=True)
    parser.add_argument("--transducer", choices=["speakers", "headset"], required=True)
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="P3: mic-to-speaker distance in cm (0 for the headset pass; zero is a value)",
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--residual-n", type=int, default=RESIDUAL_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--stt-device",
        choices=["cpu", "gpu"],
        default=None,
        help="how the running server was started, when docker inspect cannot say (ruling A7); "
        "never taken from this process's ASTRA_STT_CPU",
    )
    parser.add_argument("--out", default=os.environ.get("ASTRA_ACOUSTIC_REPORT") or None)
    args = parser.parse_args(argv)
    if args.distance_cm is None:
        parser.error(
            "--distance-cm is required (P3): a delay number without the geometry cannot be "
            "compared to the next one. The headset pass passes 0."
        )
    try:
        return asyncio.run(run(args))
    except (RegimeError, ReportError, ControlFailure) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
