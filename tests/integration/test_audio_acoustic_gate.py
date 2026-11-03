"""T-0023 criteria 2-6: the gate against a real room, through speakers, a mic and the STT.

Markers: ``integration`` + ``audio`` + ``slow``, **never** ``gpu`` — this is a loudspeaker, a
Logitech C925e and ``astra-stt`` on CPU at 127.0.0.1:8080, not the 5090. ``/launch``-gated, human
required (P2 speakers, P3 mic + the measured distance, P4 headset, P5 a quiet room, P6 the STT).

This module is new: T-0006's ``tests/integration/test_audio_acoustic_loopback.py`` is run
unchanged beside it, never edited. What is different here is that the gate exists now, so the
questions are the four the design has been quoting as hypotheses:

* how long the DAC → air → ADC path is (criterion 2), measured as the difference between where the
  chirp sits in a recording of the **sink's own monitor** (the signal as written to the device) and
  where it sits in a simultaneous recording of the **microphone**. Taking both in one playback is
  what removes the two systematic terms T-0006's number carried — ``pw-record``'s connect latency
  and the device's output latency — and leaves the air path; what is left of them is jitter, which
  is what ``pstdev ≤ 50 ms`` bounds;
* whether the STT hears us say "one, two, three, four, five" through the room with the gate on
  (criterion 3) — **and whether it hears it with the gate off**, because without that control a
  dead mic, a dead STT and a gate stuck shut all look like success;
* whether the gate lets through audio that is **not** ours (criterion 4) — the failure mode this
  module exists to catch, since ``should_gate() -> True`` passes every other test here;
* how much echo-carrying capture reaches the STT tap **ungated** at the leading edge (criterion 6),
  which the hangover does not and must not cover (T-0021 ruling A16c).

Regime for every number (CLAUDE.md, T-0023 §Regime): box ``gpu-box``, PipeWire 1.0.5,
sink ``$ASTRA_ACOUSTIC_SINK`` with its ``wpctl`` volume, transducer speakers (headset is a separate
labelled pass, ``$ASTRA_ACOUSTIC_HEADSET_SINK``), Logitech C925e mic, mic-to-speaker distance
``$ASTRA_MIC_DISTANCE_CM`` cm, room noise-floor peak printed by the first test, n = 10 per
transducer for the delay, n = 10 utterances per gate condition, n = 30 for the leading edge,
seed 0, ``profile=lean``, gate as stated per test, STT = moshi-server 0.6.4 on CPU
(``ASTRA_STT_CPU=1``) batch_size 2, vLLM co-residency stated by the run report, git tree clean.

What this module does not claim: any AEC judgement (v1 ships none, ``Profile.aec == "none"``), that
the gate holds at other distances/volumes/geometries, and any end-to-end B3/B4 number.
"""

from __future__ import annotations

import asyncio
import functools
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.audio_device import (
    DEVICE_RATE,
    CaptureSink,
    MonitorRecording,
    PcmUtterance,
    acoustic_delay_ms,
    attr_or_fail,
    capture_stats,
    chirp_fixture,
    default_sink_name,
    import_or_fail,
    play,
    rendering_span_start_ms,
    samples_of,
    stat,
    t_ms_of,
    tts_fixture,
)
from tests.fakes.locked_scripts import ACOUSTIC, helper, load_locked_script

pytestmark = [
    pytest.mark.integration,
    pytest.mark.audio,
    pytest.mark.slow,
]

SINK_ENV = "ASTRA_ACOUSTIC_SINK"
HEADSET_ENV = "ASTRA_ACOUSTIC_HEADSET_SINK"
DISTANCE_ENV = "ASTRA_MIC_DISTANCE_CM"
#: Where ``scripts/locked/audio_acoustic.py --out`` wrote its JSON report (criteria 5 and 6 read
#: the measured p95 from it instead of re-running ten minutes of room).
REPORT_ENV = "ASTRA_ACOUSTIC_REPORT"
DUMMY_SINKS = {"auto_null", "Dummy Output"}

N_DELAY = 10  # criterion 2
N_UTTERANCE = 10  # criterion 3, per gate condition
N_NOT_OURS = 3  # criterion 4 (the criterion fixes no n; three is stated, not implied)
N_RESIDUAL = 30  # criterion 6
SEED = 0
#: The C925e's ambient peak in this room is ~0.116 (T-0006); above 0.2 is signal, not the room.
HEARD_PEAK = 0.2
#: The delayed-stream STT advances its clock only on audio: a stream that stops dead transcribes
#: nothing at all, so every transcript is read after this much **real captured** tail
#: (``scripts/demo/_listen.py``'s ``TAIL_SILENCE_S``). The control uses the same policy as the
#: gated run or the comparison is invalid.
TAIL_S = 3.0
#: docs/01 §8's capture line plus docs/02 §4: past this, the hangover is a spec change, not a
#: constant (criterion 5).
MAX_HANGOVER_MS = 200.0
WHY_CAPTURE = "T-0021 criterion 8: the self-speech gate on the STT tap"


# --- the rig -------------------------------------------------------------------------------------


def sink_or_skip(env: str = SINK_ENV) -> str:
    """The sink to play through. A **named** sink that is not there is an error, never a skip.

    Criterion 2's calibration guard: skipping is only honest when no non-dummy sink exists at all
    (the DP cable is unplugged and the default sink is a Dummy Output). Once somebody has named a
    sink, "it is not in ``pw-dump``" is a misconfiguration and a silent skip would report success
    through a dead speaker (T-0006 ruling A10).
    """
    from tests.fakes.audio_device import missing_node_reason, pw_nodes

    named = os.environ.get(env)
    if named:
        reason = missing_node_reason(named, f"{env}={named}")
        if reason is not None:
            pytest.fail(
                f"{env}={named!r} names a sink that is not in pw-dump ({reason}). A named sink is "
                f"a configuration, not an absence: this is an error, never a skip (T-0023 "
                f"criterion 2's calibration guard)"
            )
        return named
    if pw_nodes() is None:
        pytest.skip("pw-dump is not available: there is no PipeWire session to play through")
    current = default_sink_name()
    if current is None or current in DUMMY_SINKS:
        pytest.skip(
            f"no non-dummy sink exists: the PipeWire default sink is {current!r} (P2 — plug the "
            f"DP cable in, or export {env}=<node name>). The headset rows stay owed as "
            f"'NOT MEASURED' with this reason (criterion 1)"
        )
    return current


def mic_or_skip() -> str:
    from tests.fakes.audio_device import pw_nodes

    nodes = pw_nodes() or {}
    mics = [name for name in nodes if name.startswith("alsa_input.")]
    if not mics:
        pytest.skip("no ALSA capture node in pw-dump (P3: the Logitech C925e is the only mic)")
    return mics[0]


def distance_cm_or_fail() -> float:
    """P3: a delay number without the geometry cannot be compared to the next one."""
    raw = os.environ.get(DISTANCE_ENV)
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pytest.fail(
            f"P3: export {DISTANCE_ENV}=<mic-to-speaker distance in cm> before measuring. Got "
            f"{raw!r}. Every acoustic number in T-0023 carries the geometry it was taken at "
            f"(§Regime), and this is an error, not a skip"
        )


def open_acoustic_device(sink: str, source: str) -> Any:
    """The real backend. A missing ``libportaudio2`` is P1, and P1 is an error, never a skip."""
    module = import_or_fail("local_astra.conductor.audio_device", "T-0006 criterion 1")
    open_device = attr_or_fail(module, "open_device", "T-0006 ruling A1: the real backend")
    try:
        return open_device(sink=sink, source=source, block_ms=CAPTURE_BLOCK_MS, latency_ms=30.0)
    except (OSError, RuntimeError) as exc:
        if "PortAudio" in str(exc) or "libportaudio2" in str(exc):
            pytest.fail(
                f"prerequisite P1: the backend needs libportaudio2 and it is not installed "
                f"(`sudo apt-get install -y libportaudio2`). Every hardware run in T-0023 fails "
                f"at open_device() without it; this is an error, never a skip ({exc})"
            )
        raise


def build_io(sink: str, source: str, *, gate: Any | None = None) -> tuple[Any, Any]:
    """The module under test on the real device: ``(io, device)`` — the device carries the clock."""
    from local_astra.conductor.audio_io import PipeWireAudioIO

    device = open_acoustic_device(sink, source)
    return PipeWireAudioIO(device, **({} if gate is None else {"gate": gate})), device


def room_floor_peak(mic: str, tmp_path: Path, seconds: float = 1.5) -> float:
    """The room's ambient peak on this mic, measured now (ruling A9's residual threshold).

    Criterion 6 counts blocks that carry echo, and "carries echo" is *above the room floor* — not
    above a constant. The 0.2 in ``HEARD_PEAK`` is the rig guard's energy floor (is the speaker
    audible at all); using ``max(0.2, floor)`` for the residual would silently ignore every echo
    block quieter than 0.2 and report a residual of zero through a working leak.
    """
    with MonitorRecording(mic, tmp_path / "residual_floor.wav", capture_sink=False) as silence:
        time.sleep(seconds)
    quiet, _rate = silence.read()
    return peak(quiet)


def close_and_stop(io: Any, device: Any) -> None:
    """End one measurement: close the capture side **and stop the device** (ruling A9).

    ``close_capture()`` alone leaves the PortAudio stream open, so the next measurement in the
    same session opens a second duplex stream on the same hardware and the two compete for the
    graph — which shows up as jitter in exactly the number being measured.
    """
    attr_or_fail(io, "close_capture", "T-0021 criterion 1")()
    attr_or_fail(device, "stop", "AudioDevice protocol: start()/stop()")()


def gate_off() -> Any:
    """Criterion 3's positive control: a gate that never closes, same tail policy as the gated run."""
    module = import_or_fail("local_astra.conductor.audio_capture", WHY_CAPTURE)
    cls = attr_or_fail(module, "SelfSpeechGate", WHY_CAPTURE)
    return cls(lambda _t_ms: False, 0.0)


def sentence() -> tuple[PcmUtterance, str]:
    """``pocket_tts_sentence.wav`` and the text it says (T-0006's fixture, MANIFEST.md)."""
    samples, rate, meta = tts_fixture()
    assert rate == 24_000
    return PcmUtterance("sentence", samples=samples, text=meta["text"]), str(meta["text"])


def chirp_utterance(uid: str) -> PcmUtterance:
    samples, rate = chirp_fixture()
    assert rate == 24_000
    return PcmUtterance(uid, samples=samples, text="chirp")


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x, dtype=np.float64)))) if x.size else 0.0


def report_or_fail() -> dict[str, Any]:
    """Criteria 5 and 6 read the measured p95 from the script's report, or error by name."""
    import json

    path = os.environ.get(REPORT_ENV)
    if not path:
        pytest.fail(
            f"{REPORT_ENV} is unset: run `uv run python scripts/locked/audio_acoustic.py "
            f"--measure delay --transducer speakers --distance-cm <N> --n {N_DELAY} --seed {SEED} "
            f"--out $TMPDIR/acoustic.json` first and point {REPORT_ENV} at it. This is an error, "
            f"never a skip: without the measurement the hangover is a guess (T-0023 criterion 5)"
        )
    file = Path(path)
    if not file.exists():
        pytest.fail(f"{REPORT_ENV}={path!r} does not exist: no acoustic report was written")
    return dict(json.loads(file.read_text()))


def measured_echo_p95_ms(report: dict[str, Any], transducer: str = "speakers") -> float:
    row = report.get("delay_ms", {}).get(transducer)
    if not isinstance(row, dict) or "p95" not in row:
        pytest.fail(
            f"the acoustic report has no delay_ms.{transducer}.p95: it holds "
            f"{sorted(report.get('delay_ms', {}))} (T-0023 criterion 2 measures the speakers path "
            f"first; the hangover comes from the worse path, never from the headset one)"
        )
    return float(row["p95"])


# --- criterion 2's guard, first in the module ------------------------------------------------------


def test_rig_is_audible_or_the_test_errors(tmp_path: Path) -> None:
    """Runs first: prove the room carries our audio before believing any gate number.

    The noise floor is measured in the same breath and printed, because every number in this
    module is only meaningful beside it (the room is not a null sink: gated blocks are digital
    silence by construction, so what the later tests measure is *which blocks were gated*, not how
    quiet the room was).
    """
    sink, mic = sink_or_skip(), mic_or_skip()

    with MonitorRecording(mic, tmp_path / "floor.wav", capture_sink=False) as silence:
        time.sleep(1.5)
    quiet, rate = silence.read()
    floor = peak(quiet)
    assert rate == DEVICE_RATE
    assert floor < HEARD_PEAK, (
        f"the room's noise floor peaks at {floor:.3f} on {mic!r}, at or above the {HEARD_PEAK} "
        f"this module calls 'heard': P5 wants a quiet room, or every number here is noise"
    )

    with MonitorRecording(mic, tmp_path / "calib.wav", capture_sink=False) as rec:
        proc = subprocess.Popen(
            ["pw-play", "--target", sink, str(Path("tests/fixtures/audio/chirp_5s_24k.wav"))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.wait(timeout=30) == 0, (
            f"pw-play could not play into {sink!r}: "
            f"{(proc.stderr.read() if proc.stderr else b'').decode()[:400]}"
        )
    heard, _rate = rec.read()
    if peak(heard) <= HEARD_PEAK:
        pytest.fail(
            f"the calibration chirp played through {sink!r} peaks at {peak(heard):.3f} on {mic!r} "
            f"— at or below the {HEARD_PEAK} floor (room floor {floor:.3f}). The sink is named "
            f"and silent: muted, unplugged, at zero volume, or not connected to a loudspeaker "
            f"(P2). Every echo and gate number taken on this rig would be a lie. This fails, it "
            f"never skips (T-0023 criterion 2, T-0006 criterion 11)."
        )
    print(
        f"\nrig: sink={sink!r} mic={mic!r} chirp peak={peak(heard):.3f} "
        f"room noise floor peak={floor:.3f} distance={os.environ.get(DISTANCE_ENV)!r} cm"
    )


# --- criterion 2 ---------------------------------------------------------------------------------


async def _one_delay(sink: str, mic: str, i: int) -> float:
    """DAC → air → ADC for one chirp, on the module's own clocks (ruling A4).

    The chirp goes out through ``PipeWireAudioIO`` and comes back through the **same duplex
    device's** capture, so both ends are stamped on one clock and the delay is
    ``correlation peak on the capture clock − the rendering span's start``. That is the lag the
    gate actually faces. The method this replaces — two ``pw-record`` processes, the sink's
    monitor minus the mic — had no shared time origin: the two streams are started 0.3 s apart
    and a sink that resumes late turns that start jitter into a plausible ≈300 ms "echo delay"
    that would push the hangover past its bound (ruling A4). The ``pw-record`` pair survives only
    where it is honest: ``test_rig_is_audible_or_the_test_errors``, which asks about energy and
    not about time.
    """
    io, device = build_io(sink, mic)
    try:
        delay, loudest = await acoustic_delay_ms(
            io, device, chirp_utterance(f"e{i}"), heard_peak=HEARD_PEAK
        )
    finally:
        close_and_stop(io, device)
    assert loudest > HEARD_PEAK, f"run {i}: the chirp came back at {loudest:.3f}"
    return delay


@pytest.mark.parametrize("transducer", ["speakers", "headset"])
async def test_echo_path_delay_speakers_and_headset(transducer: str) -> None:
    """Criterion 2: n = 10 per transducer, seed 0, p50/p95/spread/pstdev, each with its regime.

    Physicality (0 < delay ≤ 1000 ms) and ``pstdev ≤ 50 ms`` are the assertions; a hangover
    derived from a delay that wanders more than that is meaningless. The headset pass is P4: with
    no headset it stays owed as ``"NOT MEASURED"`` in the report — a speakers number may never
    appear under the headset label (criterion 1).
    """
    distance = 0.0 if transducer == "headset" else distance_cm_or_fail()
    if transducer == "headset":
        if not os.environ.get(HEADSET_ENV):
            pytest.skip(
                f"P4: no headset sink ({HEADSET_ENV} unset). The headset rows are recorded as "
                f"'NOT MEASURED' with this reason and stay owed (criterion 1); substituting the "
                f"speakers number under the headset label is a reporting defect"
            )
        sink = sink_or_skip(HEADSET_ENV)
    else:
        sink = sink_or_skip()
    mic = mic_or_skip()

    delays = [await _one_delay(sink, mic, i) for i in range(N_DELAY)]
    for i, d in enumerate(delays):
        assert 0.0 < d <= 1000.0, (
            f"run {i}: an echo-path delay of {d:.1f} ms is not physical — the correlation peak "
            f"did not find the chirp, or the two recordings are not of the same playback"
        )
    module = load_locked_script(ACOUSTIC)
    stats = helper(module, "percentiles", "T-0023 criterion 2")(delays)
    print(
        f"\necho path delay (ms), transducer={transducer} sink={sink!r} mic={mic!r} "
        f"distance={distance:g} cm n={N_DELAY} seed={SEED}: p50={stats['p50']:.1f} "
        f"p95={stats['p95']:.1f}{' (bound)' if stats['p95_is_bound'] else ''} "
        f"spread={stats['spread']:.1f} pstdev={stats['pstdev']:.1f}"
    )
    assert statistics.pstdev([float(d) for d in delays]) <= 50.0, (
        f"the echo-path delay wanders by {statistics.pstdev(delays):.1f} ms over {N_DELAY} runs "
        f"({transducer}, {distance:g} cm): a hangover derived from it would be meaningless"
    )


# --- criteria 3 and 4: the STT through the room ---------------------------------------------------


async def _transcribe(
    io: Any, *, play_now: Any, listen_s: float, tail_s: float = TAIL_S
) -> tuple[str, int]:
    """Feed ``stt_blocks()`` straight into ``MoshiSttClient.feed()`` and return what came back.

    No resampling, no reframing, no retiming (T-0021 criterion 7): the STT tap already is 24 kHz
    f32 in 1920-sample frames, and feeding it anything else produces a plausible transcript of
    nonsense. The transcript is read only after ``tail_s`` of **real captured** tail, because the
    delayed-stream model advances its clock on audio and on nothing else.
    """
    from local_astra.conductor.stt_client import MoshiSttClient

    client = MoshiSttClient()
    words: list[str] = []
    frames = 0

    async def pump() -> None:
        nonlocal frames
        async for frame in io.stt_blocks():
            await client.feed(
                np.asarray(samples_of(frame), dtype=np.float32).tobytes(), int(frame.t_ms)
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
    assert frames > 0, "no STT frames were produced at all: the capture tap is dead, not gated"
    return " ".join(words), frames


async def test_gate_on_yields_zero_self_transcription_with_the_gate_off_control(
    tmp_path: Path,
) -> None:
    """Criterion 3, the headline: n = 10 utterances per condition, same tail policy for both.

    Gate on: content-word overlap with "one, two, three, four, five" is **exactly 0**. Words the
    room contributed are counted separately and never folded into the headline. Gate off (the
    mandatory positive control): overlap ≥ 50 %, and a control that transcribes nothing **errors**
    — "the STT produced nothing" is otherwise indistinguishable from a dead mic, a dead STT and a
    gate stuck shut.
    """
    sink, mic = sink_or_skip(), mic_or_skip()
    distance = distance_cm_or_fail()
    module = load_locked_script(ACOUSTIC)
    overlap_fraction = helper(module, "overlap_fraction", "T-0023 criterion 3")
    control_overlap = helper(module, "control_overlap", "T-0023 criterion 3")
    content_words = helper(module, "content_words", "T-0023 criterion 3")
    utterance, spoken = sentence()
    listen_s = utterance.samples.size / 24_000.0

    gated: list[float] = []
    spurious = 0
    for i in range(N_UTTERANCE):
        io, _device = build_io(sink, mic)
        text, _frames = await _transcribe(
            io,
            play_now=functools.partial(play, io, utterance, source="fixture"),
            listen_s=listen_s,
        )
        close_and_stop(io, _device)
        gated.append(float(overlap_fraction(spoken, text)))
        spurious += len(set(content_words(text)) - set(content_words(spoken)))
        print(f"\n[{i}] gate=on transcript={text!r} overlap={gated[-1]:.2f}")

    control: list[float] = []
    for i in range(N_UTTERANCE):
        io, _device = build_io(sink, mic, gate=gate_off())
        text, _frames = await _transcribe(
            io,
            play_now=functools.partial(play, io, utterance, source="fixture"),
            listen_s=listen_s,
        )
        close_and_stop(io, _device)
        # raises ControlFailure — not returns 0.0 — when the control transcribed nothing
        control.append(float(control_overlap(spoken, text)))
        print(f"\n[{i}] gate=off (control) transcript={text!r} overlap={control[-1]:.2f}")

    print(
        f"\nself-transcription: sink={sink!r} mic={mic!r} distance={distance:g} cm "
        f"n={N_UTTERANCE}/condition seed={SEED} spoken={spoken!r} "
        f"gate-on overlap max={max(gated):.2f} spurious room words={spurious} "
        f"control overlap min={min(control):.2f}"
    )
    assert min(control) >= 0.5, (
        f"the positive control failed: with the gate off the room's echo of our own voice reaches "
        f"the STT at only {min(control):.2f} overlap (want ≥ 0.50). Nothing below this line means "
        f"anything until it does (criterion 3)"
    )
    assert max(gated) == 0.0, (
        f"with the gate on the STT transcribed our own voice back at {max(gated):.2f} overlap "
        f"(the control measured {min(control):.2f}, so the path is alive): the gate leaks "
        f"(criterion 3). Room words counted separately: {spurious}"
    )


async def test_the_gate_does_not_silence_audio_that_is_not_ours() -> None:
    """Criterion 4, the adversarial case: playback idle, the same sentence, the gate **on**.

    A ``should_gate()`` that returns True always, or a hangover that never closes, passes
    criterion 3 perfectly and destroys the product. The audio is played by ``pw-play`` — a process
    that is not the module under test — so no rendering span is ever opened, and both halves are
    checked: the words arrive **and** no block was gated.
    """
    sink, mic = sink_or_skip(), mic_or_skip()
    module = load_locked_script(ACOUSTIC)
    overlap_fraction = helper(module, "overlap_fraction", "T-0023 criterion 4")
    _utterance, spoken = sentence()
    fixture = Path("tests/fixtures/audio/pocket_tts_sentence.wav")

    overlaps: list[float] = []
    gated_blocks: list[int] = []
    for i in range(N_NOT_OURS):
        # the real gate: is_rendering() comes from our own playback, which stays idle here
        io, _device = build_io(sink, mic)

        async def play_someone_else() -> None:
            proc = subprocess.Popen(
                ["pw-play", "--target", sink, str(fixture)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.get_running_loop().run_in_executor(None, proc.wait)

        text, _frames = await _transcribe(io, play_now=play_someone_else, listen_s=4.5)
        overlaps.append(float(overlap_fraction(spoken, text)))
        stats = await capture_stats(io)
        close_and_stop(io, _device)
        gated_blocks.append(int(stat(stats, "gated_blocks")))
        print(
            f"\n[{i}] not-ours transcript={text!r} overlap={overlaps[-1]:.2f} "
            f"gated_blocks={gated_blocks[-1]}"
        )

    assert max(gated_blocks) == 0, (
        f"the gate closed on audio that is not ours: {max(gated_blocks)} blocks were gated while "
        f"no rendering span was open (criterion 4). A gate that is stuck shut passes criterion 3 "
        f"perfectly and deafens the assistant"
    )
    assert min(overlaps) >= 0.5, (
        f"with playback idle the STT heard only {min(overlaps):.2f} of the sentence played into "
        f"{sink!r} (want ≥ 0.50, n={N_NOT_OURS}): the gate — or the hangover — is not reopening "
        f"(criterion 4)"
    )


# --- criterion 6 ---------------------------------------------------------------------------------


async def test_leading_edge_residual_is_bounded_by_the_echo_delay_plus_one_block(
    tmp_path: Path,
) -> None:
    """Criterion 6: ungated echo-carrying capture at the *start* of playback, n = 30.

    The window is ruling A11's, the same one the script uses: from the rendering span's **start**
    — read through ``is_rendering()``, the question the gate itself asks — up to the **first gated
    block** of that utterance. Only the leading edge. A block is echo-carrying when it peaks above
    the room floor measured in this session (ruling A9), and the residual is 10 ms times however
    many such blocks reached the tap ungated before the gate closed. Blocks after the first gated
    one are the hangover's business (T-0021 criterion 8) and counting them here would blame the
    span start for a trailing-edge number. The bound is ``CAPTURE_BLOCK_MS + p95 echo delay`` — the span
    start knows the DAC hand-off, not the air path (T-0021 ruling A20). If this misses, the fix is
    the span start in ``audio_io.py``; **never** a larger hangover, which is a trailing-edge device
    and would move the defect instead of removing it.
    """
    sink, mic = sink_or_skip(), mic_or_skip()
    p95_echo = measured_echo_p95_ms(report_or_fail())
    bound = CAPTURE_BLOCK_MS + p95_echo
    module = load_locked_script(ACOUSTIC)
    percentiles = helper(module, "percentiles", "T-0023 criterion 6")
    utterance, _spoken = sentence()
    # the threshold is the room, measured in this session (ruling A9), never a constant
    floor = room_floor_peak(mic, tmp_path)
    assert floor < HEARD_PEAK, f"the room peaks at {floor:.3f}: P5 wants it quiet"

    residuals: list[float] = []
    for i in range(N_RESIDUAL):
        io, device = build_io(sink, mic)
        raw = CaptureSink(io, "raw")
        try:
            await asyncio.sleep(0.5)  # a quiet head: what follows is echo, not a tail
            # the ADC clock, from the device that owns it (T-0021 ruling A4): the capture stamps
            # and the rendering span live in this one domain
            t_play = float(attr_or_fail(device, "captured_ms", "T-0021 criterion 1")())
            await play(io, utterance, source="fixture")
            # only the leading edge is being measured, so the window closes at the first gated
            # block — there is nothing to learn from the rest of the utterance here (ruling A11)
            await asyncio.sleep(0.5)
            captured = list(raw.items)
            span_start = rendering_span_start_ms(io, t_play)
        finally:
            await raw.close()
            close_and_stop(io, device)

        window = [b for b in captured if t_ms_of(b) >= span_start]
        assert window, f"run {i}: no capture blocks after the rendering span opened"
        gated_at = [t_ms_of(b) for b in window if bool(getattr(b, "gated", False))]
        assert gated_at, (
            f"run {i}: not one block was gated while we were rendering (span start "
            f"{span_start:.1f} ms, {len(window)} blocks). The leading edge has no end, so there "
            f"is no residual to bound — the gate never closed (criteria 4 and 6)"
        )
        leading = [b for b in window if t_ms_of(b) < min(gated_at)]
        # every block before the first gated one is ungated by construction, so this is the
        # ungated echo that reached the tap at the leading edge (ruling A11)
        echo = [b for b in leading if peak(samples_of(b)) > floor]
        residuals.append(len(echo) * float(CAPTURE_BLOCK_MS))

    stats = percentiles(residuals)
    print(
        f"\nleading-edge residual (ms), sink={sink!r} mic={mic!r} n={N_RESIDUAL} seed={SEED} "
        f"room floor peak={floor:.3f}: "
        f"p50={stats['p50']:.1f} p95={stats['p95']:.1f} against the bound "
        f"{bound:.1f} = CAPTURE_BLOCK_MS {CAPTURE_BLOCK_MS} + measured p95 echo {p95_echo:.1f}"
    )
    assert float(stats["p95"]) <= bound, (
        f"{stats['p95']:.1f} ms of echo-carrying capture reached the STT tap ungated at the "
        f"leading edge (p95, n={N_RESIDUAL}), over the {bound:.1f} ms bound. The wrong term is "
        f"the rendering-span start in audio_io.py (T-0021 ruling A20), not the hangover: the "
        f"hangover is a trailing-edge device and raising it moves the defect (criterion 6)"
    )


# --- criterion 5 ---------------------------------------------------------------------------------


def test_the_hangover_constant_covers_the_measured_trailing_edge() -> None:
    """Criterion 5: the constant is a consequence of the measurement, bounded, and not padded.

    ``DEFAULT_HANGOVER_MS`` must cover ``p95 echo + CAPTURE_BLOCK_MS`` from the **speakers** run
    (the worse path), stay ≤ 200 ms — past that it is a spec change to docs/01 §8's capture line
    raised with the measurement, never a silently large constant — and carry no slack beyond
    ``ceil(p95 + block)`` or the re-affirmed 60.0, because every ms of hangover is a ms of the
    user's reply that reaches the STT as substituted silence.
    """
    module = import_or_fail("local_astra.conductor.audio_capture", WHY_CAPTURE)
    hangover = float(attr_or_fail(module, "DEFAULT_HANGOVER_MS", "T-0023 criterion 5"))
    p95_echo = measured_echo_p95_ms(report_or_fail())
    needed = p95_echo + float(CAPTURE_BLOCK_MS)

    assert hangover >= needed, (
        f"DEFAULT_HANGOVER_MS is {hangover} ms but the measured trailing edge needs "
        f"{needed:.1f} ms (p95 echo {p95_echo:.1f} + one {CAPTURE_BLOCK_MS} ms block, speakers): "
        f"the gate reopens while the room is still playing us back (criterion 5)"
    )
    assert hangover <= MAX_HANGOVER_MS, (
        f"DEFAULT_HANGOVER_MS is {hangover} ms, over the {MAX_HANGOVER_MS} ms bound. A larger "
        f"measured value is a spec change (docs/01 §8, docs/02 §4) raised with the measurement "
        f"and its regime, not a silently large constant (criterion 5)"
    )
    allowed = max(60.0, float(math.ceil(needed)))
    assert hangover <= allowed, (
        f"DEFAULT_HANGOVER_MS is {hangover} ms where the measurement asks for "
        f"ceil({needed:.1f}) = {math.ceil(needed)} ms (or the re-affirmed 60.0): the surplus is "
        f"the user's reply arriving at the STT as substituted silence, and a leading-edge miss "
        f"may never be absorbed here (criteria 5 and 6)"
    )
    print(
        f"\nhangover: DEFAULT_HANGOVER_MS={hangover} ms covers p95 echo {p95_echo:.1f} ms + "
        f"{CAPTURE_BLOCK_MS} ms block = {needed:.1f} ms (speakers, the worse path)"
    )
