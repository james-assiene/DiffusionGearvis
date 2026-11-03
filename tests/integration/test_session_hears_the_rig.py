"""T-0045 criterion 3: a session started by the CLI actually hears the rig.

The origin of the task: the human's first spoken attempt, 2026-09-15. Nothing appeared in the
transcript pane and the turn record said ``stt_frames_fed: 0`` — ``cli.open_audio`` calls
``open_device(sink=args.sink)`` with no ``source``, so ``PortAudioDevice`` opens an output-only
stream and the capture pump has nothing to pump. Every fast-suite capture row was green
throughout, because they all drive the device object directly: **only a row that starts the
command the human types can see this**, which is what this module is.

What it does (no human, no speaker, no microphone — the null-sink rig is the whole acoustic
path)::

    astra session --mic astra-test-sink.monitor --sink astra-test-sink --data-dir <tmp>

spawned on free ports, a 1 s 440 Hz tone played into ``astra-test-sink`` with ``pw-play`` while it
runs, then **one typed question** over the WebSocket (the ``scripts/demo/text_mode_probe.py``
exchange) so that a turn completes and one ``turns.jsonl`` line is written, then SIGINT. The
assertion is that line's ``extras.stt_frames_fed``: the session's own tally of frames handed to
the STT client. Typed, not spoken, on purpose — the question only has to *close a turn* so the
counter is written down; what is under test is the capture path, not the transcript.

**It fails, it does not skip, when the rig is up.** It skips only when something it cannot
provide is genuinely absent, and the reason names which: ``libportaudio2`` (T-0006 prerequisite
P1), the node ``astra-test-sink`` (``bash scripts/audio_rig.sh up``), ``pw-play``, or the ``lean``
services (one typed turn needs vLLM, moshi-server and Pocket TTS to be up — ``uv run astra up
--profile lean``).

Markers: ``integration`` + ``audio``, and the turn row is additionally ``gpu`` — the typed turn it
takes spends engine time, so it is selected only by ``-m gpu``, which goes through ``/launch``
(CLAUDE.md). The ``--dry-run`` row beside it spends nothing.

**Regime** (state it with any number read from here — none is a latency): box
``gpu-box``, Linux 7.0.0-31-generic, libpipewire 1.0.5; profile ``lean`` with the
services of ``services/versions.lock``; sink ``astra-test-sink`` (null sink), microphone
``astra-test-sink.monitor``, transducer **none**; tone 440 Hz, 1 s, 48 kHz mono; one typed turn,
n = 1, no seed. This row spends GPU time through the session it starts, so it runs under
``/launch`` like the rest of ``tests/integration``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import pathlib
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import wave

import pytest

from local_astra.config import load
from local_astra.contracts.codec import decode, encode
from local_astra.contracts.protocol import AstraHello, AstraText, AstraTurnState
from tests.fakes.audio_device import missing_node_reason
from tests.unit import _mic_rig as mic_rig

pytestmark = [pytest.mark.integration, pytest.mark.audio]

REPO = pathlib.Path(__file__).resolve().parents[2]
SINK = mic_rig.RIG_SINK
MIC = mic_rig.RIG_MIC
HOW = "run `bash scripts/audio_rig.sh up` (T-0006 criterion 11)"

#: Ceilings, all generous; none of them is a measurement.
START_CEILING_S = 90.0
TONE_S = 1.0
TURN_CEILING_S = 240.0
EXIT_CEILING_S = 60.0
QUESTION = "say the word hello and nothing else"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def wait_until(predicate, ceiling_s: float) -> bool:
    deadline = time.monotonic() + ceiling_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture(autouse=True)
def _rig():
    """Skip only on a genuinely absent rig, and name exactly what is missing.

    The ``sounddevice`` probe is undone on the way out (T-0035 ruling A16): an import this module
    performs must not answer for another module's "did anything pull in a sound library" row.
    """
    present = "sounddevice" in sys.modules
    before = sys.modules.get("sounddevice")
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:  # OSError without libportaudio2
        pytest.skip(f"PortAudio is not installed (T-0006 prerequisite P1, libportaudio2): {exc}")
    reason = missing_node_reason(SINK, HOW)
    if reason is not None:
        pytest.skip(reason)
    if shutil.which("pw-play") is None:
        pytest.skip("pw-play is not on this box: nothing can play the tone into the null sink")
    from local_astra import supervisor

    services = load(None).profiles["lean"].services
    unhealthy = [
        name
        for name, url in (("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts))
        if not supervisor.check_service(name, url).healthy
    ]
    if unhealthy:
        pytest.skip(
            f"the lean services {unhealthy} are not up, and one typed turn needs all three "
            f"(`uv run astra up --profile lean`)"
        )
    try:
        yield
    finally:
        if present:
            sys.modules["sounddevice"] = before
        else:
            sys.modules.pop("sounddevice", None)


def tone(path: pathlib.Path, seconds: float = TONE_S, rate: int = 48000) -> pathlib.Path:
    """A 440 Hz sine, 16-bit mono — loud enough that no gate can call it silence."""
    frames = int(seconds * rate)
    samples = (
        struct.pack("<h", int(0.5 * 32767 * math.sin(2 * math.pi * 440.0 * n / rate)))
        for n in range(frames)
    )
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(samples))
    return path


async def _one_typed_turn(url: str) -> list[str]:
    """``astra.hello`` → ``astra.text`` → back to IDLE, as ``text_mode_probe.py`` does it."""
    import websockets

    seen: list[str] = []
    async with websockets.connect(url, origin="http://127.0.0.1:1420", max_size=None) as ws:
        await ws.send(encode(AstraHello()))
        while True:
            message = decode(await asyncio.wait_for(ws.recv(), 30.0))
            if isinstance(message, AstraTurnState):
                seen.append(message.state)
                break
        await ws.send(encode(AstraText(text=QUESTION)))
        opened = False
        deadline = time.monotonic() + TURN_CEILING_S
        while time.monotonic() < deadline:
            try:
                message = decode(await asyncio.wait_for(ws.recv(), 10.0))
            except TimeoutError:
                continue
            if isinstance(message, AstraTurnState):
                seen.append(message.state)
                if message.state != "IDLE":
                    opened = True
                elif opened:
                    return seen
    return seen


@pytest.mark.gpu
def test_a_session_started_with_a_mic_feeds_the_stt(tmp_path):
    """Criterion 3: ``stt_frames_fed > 0`` on the turn line of a session the CLI started.

    Red today for the real reason: the CLI passes no ``source``, the stream is output-only and the
    counter on every line is 0.
    """
    conductor_port, hud_port = free_port(), free_port()
    config = tmp_path / "astra.toml"
    config.write_text(
        "profile = 'lean'\n[profiles.lean.services]\n"
        f"conductor_port = {conductor_port}\nhud_port = {hud_port}\n"
    )
    data_dir = tmp_path / "data"
    argv = [
        "uv",
        "run",
        "astra",
        "--config",
        str(config),
        "session",
        "--mic",
        MIC,
        "--sink",
        SINK,
        "--data-dir",
        str(data_dir),
    ]
    proc = subprocess.Popen(
        argv,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    output = ""
    try:
        serving = wait_until(
            lambda: listening(conductor_port) or proc.poll() is not None, START_CEILING_S
        )
        assert serving and proc.poll() is None, (
            f"`astra {' '.join(argv[3:])}` must reach a bound endpoint; poll={proc.poll()}"
        )
        subprocess.run(
            ["pw-play", "--target", SINK, str(tone(tmp_path / "tone.wav"))],
            check=True,
            capture_output=True,
            timeout=30,
        )
        states = asyncio.run(_one_typed_turn(f"ws://127.0.0.1:{conductor_port}/v1/realtime"))
        assert states and states[-1] == "IDLE", (
            f"the typed turn must complete so a turns.jsonl line is written; states were {states}"
        )
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGINT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            output = proc.communicate(timeout=EXIT_CEILING_S)[0] or ""
        if proc.poll() is None:  # pragma: no cover - the stuck path
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=EXIT_CEILING_S)

    turns = data_dir / "turns.jsonl"
    assert turns.is_file(), (
        f"one typed turn writes one turns.jsonl line under --data-dir; {turns} does not exist. "
        f"The session said:\n{output[-3000:]}"
    )
    lines = [json.loads(line) for line in turns.read_text().splitlines() if line.strip()]
    assert lines, f"…and the file is not empty. The session said:\n{output[-3000:]}"
    fed = lines[-1].get("extras", {}).get("stt_frames_fed")
    assert fed is not None, (
        f"the turn line carries the capture tally (T-0032 criterion 8); it has "
        f"{sorted(lines[-1].get('extras', {}))}"
    )
    assert fed > 0, (
        f"criterion 3: a session started with `--mic {MIC}` feeds the STT — the line says "
        f"stt_frames_fed={fed}, which is the deaf, output-only session of 2026-09-15 "
        f"(open_device with no source => PortAudioDevice.duplex is False => stt_blocks() never "
        f"yields). The session said:\n{output[-3000:]}"
    )


def test_the_session_says_which_microphone_it_opened(tmp_path):
    """Criterion 2 on the real command: the startup line names the node, beside the sink.

    ``--dry-run`` is enough for this one: it is the same resolved-graph line the session prints on
    the way up, and it starts no service and opens no device.
    """
    output = subprocess.run(
        [
            "uv",
            "run",
            "astra",
            "session",
            "--dry-run",
            "--mic",
            MIC,
            "--sink",
            SINK,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    printed = output.stdout + output.stderr
    assert output.returncode == 0, f"--dry-run exits 0; it exited {output.returncode}:\n{printed}"
    assert f"mic={MIC}" in printed, (
        f"criterion 2: the resolved graph names the microphone the session will open "
        f"(mic={MIC}); it printed:\n{printed}"
    )
