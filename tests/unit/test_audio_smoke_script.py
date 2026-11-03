"""``scripts/demo/audio_smoke.py`` is code, not a snippet: T-0035 criteria 8, 9 and 10(d).

The two-minute "is the audio path real" check. It opens the device through
``audio_device.open_device()`` — **never** through ``sounddevice`` directly, which would put a
second audio client in the process and is the one thing
``tests/unit/test_audio_ownership.py`` exists to prevent — plays a 2 s chirp, and prints one block
with the requested *and* negotiated latencies, ``quantum_ms`` beside its ``quantum_source``,
the live graph quantum from ``pw-top`` (ruling A13: ``pw-dump``'s ``node.latency`` is only the
request echoed back, so the *graph's* quantum is a separate, `pw-top -b -n 2` measurement),
``underflow_frames``, ``device_xruns``, the captured peak, the verified link peers of both halves
and the ``pw-dump`` sink/source list.

Beside the peak it prints the discriminator the 2026-09-14 audit had to work out by hand, because
a peak is the one number that looks fine when everything is wrong:

* ``peak == 0`` ⇒ the target is absent — you are recording nothing;
* ``peak ≈ 0.11–0.16`` on a silent rig ⇒ you captured the webcam mic, not the monitor.

Both are non-zero exits (criterion 8): a smoke script that says "OK" on a silent capture is worse
than no script.

**The seams this module pins**, so the script is testable on a box with no PortAudio
(prerequisite P1) and so criterion 9's "import-tested and ``--dry-run``-tested by the fast suite"
has something to call:

* ``main(argv: Sequence[str] | None = None) -> int`` — the exit code, no ``SystemExit``;
* ``--sink NAME`` / ``--source NAME`` — the PipeWire node names, as in criterion 11's command
  line;
* ``--dry-run`` — performs no device I/O at all (this box cannot: ``import sounddevice`` raises
  ``OSError`` without ``libportaudio2``), still prints the whole report;
* ``--peak FLOAT`` — the captured peak a dry run reports instead of capturing one, so the
  discriminator itself is testable without a rig. Reading ``pw-dump`` during a dry run is allowed
  (it is read-only and touches no device); when the graph cannot be read the report says so.

Regime: this box, PortAudio **absent**, no rig, no graph assumptions; the script is loaded by path
(``scripts/`` is not a package), and nothing here starts a stream.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from local_astra.conductor.audio_device import DEVICE_RATE, RING_MS, WRITE_TIMEOUT_S
from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.sounddevice_stub import Driver, StubStream, driving, installed, last_stream

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "demo" / "audio_smoke.py"

#: Every field criterion 8 requires in the one printed block, as the label a reader greps for.
REQUIRED_FIELDS = (
    "requested",
    "negotiated",
    "quantum_ms",
    "quantum_source",
    "underflow_frames",
    "device_xruns",
    "peak",
    "peer",
    "sinks",
    "sources",
)
#: The discriminator, in the words of criterion 8. Matched phrase by phrase, so the sentence may
#: be wrapped or punctuated freely, but it may not be paraphrased away.
PEAK_ZERO_PHRASES = ("peak == 0", "target is absent", "recording nothing")
PEAK_WEBCAM_PHRASES = ("0.11", "0.16", "webcam mic")

#: Criterion 9: no audio library but ``audio_device``, and no socket at all (ruling A10(b)). The
#: network names are here because "opens no socket" is otherwise asserted by nothing: the script
#: reads a local graph and a local device, and a smoke script that phoned anywhere would be a
#: non-loopback socket in a tree whose contract is that there are none (CLAUDE.md).
FORBIDDEN_IMPORTS = frozenset(
    {
        "sounddevice",
        "soundfile",
        "pyaudio",
        "alsaaudio",
        "ctypes",
        "socket",
        "httpx",
        "websockets",
        "aiohttp",
        "urllib",
    }
)
#: How the report says it could not read the graph (ruling A8). The wording is the implementer's;
#: what is pinned is that the line is about the graph, says it is not readable, and names the
#: reason — a report that silently prints two empty node lists is the "silence that looks like
#: success" this whole task exists to remove.
#: Ruling A13: the live graph quantum is a ``pw-top`` measurement, printed on its own line and
#: labelled with the tool it came from, so it can never be confused with ``quantum_ms`` (which is
#: the request). Either a number in ms or an honest "not readable" — the label is mandatory, the
#: value is not available on every box.
PW_TOP_LINE = re.compile(
    r"graph quantum \(pw-top\)\s*[:=]?\s*(?P<value>[0-9]+(?:\.[0-9]+)?|not readable\b.*)",
    re.IGNORECASE,
)
GRAPH_UNREADABLE = re.compile(
    r"graph.*(not readable|unreadable|unread|could not be read|not read)", re.IGNORECASE
)


def source() -> str:
    """The script's text; its absence is an AssertionError, never a FileNotFoundError."""
    if not SCRIPT.exists():
        raise AssertionError(
            "scripts/demo/audio_smoke.py does not exist (T-0035 criterion 8; scripts/demo is "
            "D5-ungated, and its API is pinned in this module's docstring)"
        )
    return SCRIPT.read_text()


def load_script() -> ModuleType:
    """Import the script by path; anything that goes wrong is an AssertionError naming it."""
    if not SCRIPT.exists():
        raise AssertionError(
            "scripts/demo/audio_smoke.py does not exist (T-0035 criterion 8; scripts/demo is "
            "D5-ungated, and its API is pinned in this module's docstring)"
        )
    spec = importlib.util.spec_from_file_location("astra_demo_audio_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("scripts/demo/audio_smoke.py is not importable by path: no loader")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - the pending path
        raise AssertionError(
            f"scripts/demo/audio_smoke.py failed to import: {type(exc).__name__}: {exc}. "
            f"Criterion 9: it imports cleanly with no PortAudio installed — nothing "
            f"device-touching at import time, and no argument parsing at module scope"
        ) from exc
    return module


def run(argv: list[str]) -> int:
    module = load_script()
    if not hasattr(module, "main"):
        raise AssertionError("scripts/demo/audio_smoke.py has no main(argv) -> int (criterion 8)")
    result = module.main(argv)
    assert isinstance(result, int), (
        f"main({argv}) returned {result!r}: criterion 8 asks for an exit code, and the script "
        f"must not raise SystemExit at its own API"
    )
    return result


def test_the_script_imports_without_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 9: nothing device-touching at import time, on a box with no ``libportaudio2``.

    Importing it must not pull ``sounddevice`` into the process either — an import guarded by a
    ``try`` would hide the same mistake on the rig box where PortAudio *is* installed (a second
    audio client, criterion 10(d)).

    The question is about *this* import, so the row asks it of a process where the library is
    absent: ``sounddevice`` is removed from ``sys.modules`` before the script is loaded and
    ``monkeypatch`` puts it back afterwards. Without that, whether this row passes depends on
    which other module the xdist worker ran first — now that ``libportaudio2`` is installed,
    ``tests/unit/test_doctor_verdicts.py`` runs the real ``preflight.collect_audio()`` and the
    real library is already in ``sys.modules``, so the assertion would be answered by somebody
    else's import (T-0035 ruling A16). Removing it also makes the check strictly stronger: if
    the script imports the library, it imports it here, from scratch, and is caught.
    """
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    assert "sounddevice" not in sys.modules, "precondition: the library starts out unimported"

    load_script()

    assert "sounddevice" not in sys.modules, (
        "importing scripts/demo/audio_smoke.py imported sounddevice: it reaches the device "
        "through audio_device.open_device() only (criteria 8, 10(d))"
    )


def test_the_script_never_imports_sounddevice_itself() -> None:
    """Criterion 10(d): ``audio_device.py`` stays the single audio owner in the process.

    A source scan, like ``tests/unit/test_audio_ownership.py``'s over ``local_astra/``: the
    script names no sound library anywhere, and it does name ``open_device``.
    """
    text = source()
    tree = ast.parse(text, filename=str(SCRIPT))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = imported & FORBIDDEN_IMPORTS
    assert forbidden == set(), (
        f"scripts/demo/audio_smoke.py imports {sorted(forbidden)}: it opens the device through "
        f"audio_device.open_device(), which is the only audio owner (criteria 8, 10(d)), and it "
        f"opens no socket (criterion 9 — every service in this tree binds to loopback and the "
        f"smoke script talks to none of them)"
    )
    assert "open_device" in text, (
        "the script never mentions open_device(): criterion 8 requires it to go through the "
        "device module, not around it"
    )


def test_dry_run_prints_every_field_and_the_discriminator(capsys: Any) -> None:
    """Criterion 8: one block, every field, and the discriminator printed beside the peak.

    A healthy peak (0.5 — well above the C925e's ~0.12 room floor) on a dry run: exit 0, and the
    report still names every field, because the fields are what make the number comparable
    later (CLAUDE.md: state the whole regime beside every number).
    """
    code = run(
        [
            "--dry-run",
            "--sink",
            "astra-test-sink",
            "--source",
            "astra-test-sink.monitor",
            "--peak",
            "0.5",
        ]
    )
    report = capsys.readouterr().out.lower()
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    assert missing == [], f"the dry-run report names neither of {missing}; it printed:\n{report}"
    for phrase in PEAK_ZERO_PHRASES + PEAK_WEBCAM_PHRASES:
        assert phrase.lower() in report, (
            f"the report does not contain {phrase!r}: criterion 8 prints the audit's "
            f"discriminator verbatim beside the peak, on every run"
        )
    assert code == 0, f"a dry run with a healthy peak exited {code}"


def test_the_report_prints_the_live_graph_quantum_from_pw_top(capsys: Any) -> None:
    """Ruling A13: the graph's own quantum is a ``pw-top`` line, labelled with its tool.

    ``quantum_ms``/``quantum_source`` above answer "what did we ask for, and did the graph echo
    it back" — under A13 that is all ``pw-dump``'s ``node.latency`` can say. The number a reader
    of a T-0023 regime block actually wants, the quantum the graph is *running* at, comes from
    ``pw-top -b -n 2``'s QUANT column for our node and the driver, and the report prints it on
    its own line under that name.

    Pinned: the label ``graph quantum (pw-top)`` and the shape of the value — a number in ms, or
    the words "not readable" when ``pw-top`` is missing, is not in batch mode, or shows no row
    for us. What is not pinned is the wording around it. A report that omits the line entirely
    leaves the request-shaped ``quantum_ms`` as the only quantum on the page, which is the
    confusion the ruling exists to end.
    """
    run(
        [
            "--dry-run",
            "--sink",
            "astra-test-sink",
            "--source",
            "astra-test-sink.monitor",
            "--peak",
            "0.5",
        ]
    )
    report = capsys.readouterr().out
    said = PW_TOP_LINE.search(report)
    assert said is not None, (
        "the report has no `graph quantum (pw-top): …` line carrying either a number or "
        '"not readable" (ruling A13). It printed:\n' + report
    )


@pytest.mark.parametrize(
    ("peak", "phrases"),
    [(0.0, PEAK_ZERO_PHRASES), (0.12, PEAK_WEBCAM_PHRASES)],
    ids=["silence", "webcam-floor"],
)
def test_a_failing_discriminator_exits_non_zero(
    peak: float, phrases: tuple[str, ...], capsys: Any
) -> None:
    """Adversarial, criterion 8: the two peaks that look like a working rig in the dark.

    ``0.0`` is what a monitor capture of an absent node produces — seconds of exactly zero, no
    error. ``0.12`` is the Logitech C925e recording the silent room, i.e. the capture half landed
    on the default source instead of the monitor. Both must exit non-zero and say which one it
    was; an exit code of 0 here is how T-0023's hardware stages acquire unattributable numbers.
    """
    code = run(
        [
            "--dry-run",
            "--sink",
            "astra-test-sink",
            "--source",
            "astra-test-sink.monitor",
            "--peak",
            str(peak),
        ]
    )
    report = capsys.readouterr().out.lower()
    assert code != 0, (
        f"a captured peak of {peak} exited 0: that peak means the rig is not recording what it "
        f"claims (criterion 8). Report:\n{report}"
    )
    for phrase in phrases:
        assert phrase.lower() in report


def test_a_box_with_no_pw_dump_still_gets_a_report(tmp_path: Any) -> None:
    """Adversarial, ruling A8: no ``pw-dump`` on the box ⇒ a line, not a traceback.

    The fast suite must be hermetic — it may not depend on a PipeWire graph being present, and
    the smoke script is the first thing anyone runs on a *new* box, where ``pw-dump`` may well
    be missing (a container without ``pipewire-utils``, a stripped image). ``PATH=/nonexistent``
    is that box: ``subprocess.run(["pw-dump"])`` raises ``FileNotFoundError``, which is neither
    an ``AudioDeviceError`` nor something a dry run should die of.

    Run in a subprocess, because the condition is about the process environment and about the
    script's ``__main__``-shaped path: exit code 0 (nothing was measured wrongly — the graph was
    simply unavailable, and ``--peak 0.5`` supplies a healthy capture), the whole report still
    printed, and a line saying the graph could not be read with the reason attached. A traceback
    here is a failure: it is the first impression of the audio path on every new box.
    """
    env = dict(os.environ, PATH="/nonexistent")
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--sink",
            "astra-test-sink",
            "--source",
            "astra-test-sink.monitor",
            "--peak",
            "0.5",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )
    both = completed.stdout + completed.stderr
    assert "Traceback (most recent call last)" not in both, (
        f"scripts/demo/audio_smoke.py raised on a box with no pw-dump on PATH (ruling A8): the "
        f"graph is read through a seam that turns every failure into a printed line. It said:\n"
        f"{both}"
    )
    assert completed.returncode == 0, (
        f"the dry run exited {completed.returncode} with PATH=/nonexistent and a healthy "
        f"--peak 0.5; an unreadable graph is reported, not a failure of the capture half "
        f"(ruling A8). Output:\n{both}"
    )
    report = completed.stdout.lower()
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    assert missing == [], (
        f"the report dropped {missing} when the graph could not be read; the fields are the "
        f"regime and are printed on every run (criterion 8). It printed:\n{completed.stdout}"
    )
    assert GRAPH_UNREADABLE.search(completed.stdout) is not None, (
        f"nothing in the report says the graph could not be read, so two empty node lists are "
        f"indistinguishable from a box with no sinks (ruling A8). It printed:\n"
        f"{completed.stdout}"
    )
    pw_top = PW_TOP_LINE.search(completed.stdout)
    assert pw_top is not None and "not readable" in pw_top.group("value").lower(), (
        f"with PATH=/nonexistent there is no pw-top to run, so the live graph quantum line must "
        f'say "not readable" rather than being absent or carrying a number: '
        f"{pw_top.group(0) if pw_top else None!r}. It printed:\n{completed.stdout}"
    )
    said = GRAPH_UNREADABLE.search(completed.stdout)
    assert said is not None and "pw-dump" in completed.stdout[said.start() : said.start() + 300], (
        f"the line {said.group(0) if said else None!r} does not name the reason (ruling A8: "
        f'"graph: not readable (…)"); without it the reader cannot tell a missing pw-dump from '
        f"an empty graph. It printed:\n{completed.stdout}"
    )


# --------------------------------------------------------------------------------------------
# T-0042 — the drain wait. The rows above run ``--dry-run`` and never touch a device; the three
# below drive the script over ``tests/fakes/sounddevice_stub.py`` with a **device thread that
# actually consumes**, which is the one condition under which the bug appears: after the chirp
# the script waits for the queue to empty by *writing into it*
# (``while device.queued_ms() > 0: device.write(zeros)``), and on a device that plays in real
# time every write refills what the wait is waiting for. Every T-0035 group-A row patched the
# device, so none of them could see it.
#
# Regime for all three: no PortAudio (prerequisite P1), ``sounddevice`` is the stub; 48 kHz mono
# f32, ``block_ms`` 10 → 480 frames (10 ms); the device thread runs at real-time pace, one
# callback every 10 ms; ring ``RING_MS`` = 200 ms = 20 blocks; ``write()``'s own ceiling is
# ``WRITE_TIMEOUT_S`` = 2 s; no node is named, so no ``pw-dump`` and no ``pw-top`` run (the graph
# seams are replaced in the rig below — this suite is hermetic and must not read a live graph).
# Wall time of the three rows together is about 5 s, which is the price of testing two real
# wall-clock ceilings.
# --------------------------------------------------------------------------------------------

BLOCK_FRAMES = round(CAPTURE_BLOCK_MS * DEVICE_RATE / 1000)  # 480
BLOCK_MS = float(CAPTURE_BLOCK_MS)  # 10.0
#: The chirp for the row that must terminate: 0.5 s = 50 blocks of 480 frames.
CHIRP_S = 0.5
#: The chirp for the two device-failure rows: 0.3 s = 30 blocks.
STALL_CHIRP_S = 0.3
#: The environment ``open_device()`` reads. A box with these set would open somebody else's
#: device under the stub; they are cleared for the length of a rig.
AUDIO_ENV = ("ASTRA_AUDIO_SOURCE", "ASTRA_AUDIO_INPUT_DEVICE", "ASTRA_AUDIO_DEVICE")
#: Criterion 1's report when the ceiling is reached: "queue did not drain (N ms left)". The
#: wording around the number is the implementer's; the phrase and the number are the criterion.
DRAIN_GAVE_UP = re.compile(r"queue did not drain\D{0,40}?([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)


@contextmanager
def _no_audio_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in AUDIO_ENV}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


class SmokeRig:
    """The script over the stub: a real ``PortAudioDevice``, a real ring, a fake device thread.

    Everything the script talks to is real except the sound library and the two graph tools:
    ``PortAudioDevice`` keeps its own ring, its own ``write()`` ceiling and its own
    ``queued_ms()``, so the loop under test meets the same flow control it meets on the rig.
    """

    def __init__(self, module: ModuleType, stub: ModuleType) -> None:
        self.module = module
        self.stub = stub
        self.device: Any = None
        #: every block the script handed to ``device.write()``, in order
        self.writes: list[np.ndarray] = []
        self.driver: Driver | None = None

    def consumed_ms(self) -> float:
        """Audio the device thread has taken out of the ring — ``written − still queued``."""
        if self.device is None:
            return 0.0
        return float(self.device.written_ms) - float(self.device.queued_ms())

    def written_ms(self) -> float:
        return 0.0 if self.device is None else float(self.device.written_ms)

    def queued_ms(self) -> float:
        return 0.0 if self.device is None else float(self.device.queued_ms())

    def stream(self) -> StubStream:
        return last_stream(self.stub)

    def silent_writes(self) -> int:
        return sum(1 for block in self.writes if not np.any(block))

    def written(self) -> np.ndarray:
        if not self.writes:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.writes)


@contextmanager
def smoke_rig(*, stall_after_ms: float | None = None) -> Iterator[SmokeRig]:
    """The script, loaded fresh, with a consuming device thread behind ``open_device()``.

    ``stall_after_ms`` is a device whose callback stops for good once it has taken that much
    audio: from the writer's side the queue simply stops shrinking, which is what a wedged
    PortAudio callback looks like and what both ceilings (the drain wait's and ``write()``'s)
    exist for. ``None`` is a healthy device that keeps consuming.

    The three graph seams are replaced by constants, not because the script's graph handling is
    uninteresting (the subprocess row above covers it) but because the fast suite may not depend
    on a live PipeWire graph and ``pw-top -b -n 2`` costs two seconds per call.
    """
    module = load_script()
    with installed() as stub, _no_audio_env():
        rig = SmokeRig(module, stub)
        real_open = module.open_device

        def open_recording(**kwargs: Any) -> Any:
            device = real_open(**kwargs)
            rig.device = device
            inner = device.write

            def write(block: Any) -> None:
                rig.writes.append(np.array(block, dtype=np.float32))
                inner(block)

            device.write = write
            return device

        module.open_device = open_recording
        module.graph_nodes = lambda: (
            {"sinks": ["astra-test-sink"], "sources": ["astra-test-sink.monitor"]},
            None,
        )
        module.link_peers = lambda: ({"output": frozenset({"astra-test-sink"})}, None)
        module.our_node_ids = lambda: set()
        module.pw_top_quantum = lambda ours: "not sampled (no pw-top in the fast suite)"
        stall: Callable[[], bool] | None = None
        if stall_after_ms is not None:
            stall = lambda: rig.consumed_ms() >= stall_after_ms  # noqa: E731
        with driving(stub, stop_when=stall) as driver:
            rig.driver = driver
            yield rig


class Ran:
    """What one guarded call produced: its value, its exception, whether it finished, its wall."""

    def __init__(self) -> None:
        self.value: Any = None
        self.error: BaseException | None = None
        self.finished = False
        self.wall = 0.0


def in_thread(call: Callable[[], Any], *, timeout_s: float, rig: SmokeRig) -> Ran:
    """Run ``call`` on a worker thread and join with a deadline: a hang is a failure, not a hang.

    The failure this module is about is an endless loop, so nothing here may be called on the
    test's own thread — a red test that never returns tells nobody anything and blocks the
    suite. When the deadline passes the stub's device thread is stopped first: the runaway loop
    is then writing into a ring nobody drains, and dies at ``write()``'s own 2 s ceiling instead
    of writing silence for the rest of the session.
    """
    ran = Ran()

    def body() -> None:
        try:
            ran.value = call()
        except BaseException as exc:  # pragma: no cover - reported by the caller
            ran.error = exc

    thread = threading.Thread(target=body, name="smoke-script", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(timeout=timeout_s)
    ran.wall = time.monotonic() - started
    ran.finished = not thread.is_alive()
    if not ran.finished and rig.driver is not None:
        rig.driver.stop()
    return ran


def test_the_chirp_is_followed_by_no_silence_and_the_drain_wait_terminates() -> None:
    """T-0042 criteria 1-2: after the last chirp block the script waits, it does not write.

    The device thread consumes 480 frames every 10 ms, which is what PortAudio does on the rig
    and what no group-A row of T-0035 did. The arithmetic, by hand: a 0.5 s chirp is
    ``0.5 × 48000 = 24000`` frames = **50 blocks** of 480; the ring holds ``RING_MS`` = 200 ms =
    20 blocks, so the writer runs ahead by at most 20 blocks and the last block is handed over
    around 300 ms in; the device finishes playing it at 500 ms. Therefore:

    * ``play_and_capture(device, 0.5)`` returns well inside 3 s — today it never returns at all,
      because ``while queued_ms() > 0: write(zeros)`` refills the queue it polls, for ever;
    * exactly 50 blocks are written and every one of them is chirp: no silence is written after
      the chirp (criterion 1), and the concatenation of what was written *is* ``chirp(0.5)``;
    * ``written == rendered`` within one block: the device took every frame we gave it out of
      the ring (``written_ms − queued_ms == 500 ms``) and nothing is left queued.

    The peak is unjudged here on purpose: no source is configured, so the device has no input
    half and ``play_and_capture`` reports ``None`` rather than a 0.0 that would fire the "you are
    recording nothing" discriminator at a run that was never recording (T-0035 ruling A8).
    """
    with smoke_rig() as rig:
        device = rig.module.open_device(block_ms=CAPTURE_BLOCK_MS)
        ran = in_thread(
            lambda: rig.module.play_and_capture(device, CHIRP_S), timeout_s=3.0, rig=rig
        )
        blocks = list(rig.writes)
        written_ms, queued_ms, consumed_ms = rig.written_ms(), rig.queued_ms(), rig.consumed_ms()
        callbacks = rig.stream().callbacks
        emitted = rig.stream().emitted()

    assert ran.finished, (
        f"play_and_capture(device, {CHIRP_S}) had not returned after {ran.wall:.1f} s on a "
        f"device that consumes in real time: it wrote {len(blocks)} blocks for a "
        f"{round(CHIRP_S * 1000 / BLOCK_MS)}-block chirp, i.e. it is still writing. Criterion 1: "
        f"the drain wait polls queued_ms() with a sleep, never with a write — a write refills "
        f"the queue the wait is waiting on, and on a real-time device that loop never ends"
    )
    assert ran.error is None, f"play_and_capture raised {ran.error!r}"
    expected_blocks = round(CHIRP_S * 1000 / BLOCK_MS)
    assert len(blocks) == expected_blocks, (
        f"the script wrote {len(blocks)} blocks for a {CHIRP_S} s chirp ({expected_blocks} × "
        f"{BLOCK_FRAMES} frames), of which {rig.silent_writes()} were silent: criterion 1 says "
        f"no block is written after the chirp"
    )
    assert np.array_equal(rig.written(), rig.module.chirp(CHIRP_S)), (
        "what the script handed to the device is not exactly chirp(0.5): the drain wait must "
        "add no samples of its own (criterion 1)"
    )
    assert abs(written_ms - consumed_ms) <= BLOCK_MS, (
        f"the device thread took {consumed_ms:.1f} ms of the {written_ms:.1f} ms written and "
        f"{queued_ms:.1f} ms is still queued: criterion 2 asks that the run end with "
        f"written == rendered within one block ({BLOCK_MS:.0f} ms)"
    )
    assert written_ms == pytest.approx(CHIRP_S * 1000.0), (
        f"the device was written {written_ms:.1f} ms of audio for a {CHIRP_S} s chirp"
    )
    assert callbacks >= expected_blocks, (
        f"the stub device thread ran {callbacks} callbacks for {expected_blocks} blocks of "
        f"audio: this row is only meaningful if the device really was draining the ring"
    )
    assert (
        emitted.size >= expected_blocks * BLOCK_FRAMES and float(np.max(np.abs(emitted))) > 0.2
    ), (
        f"the device emitted {emitted.size} frames with peak "
        f"{float(np.max(np.abs(emitted))) if emitted.size else 0.0:.3f}: the chirp peaks at 0.25, "
        f"so this run played silence, not the chirp"
    )
    assert isinstance(ran.value, rig.module.Measured), (
        f"play_and_capture returned {ran.value!r}, not a Measured (T-0035 criterion 8)"
    )


def test_a_device_that_never_drains_is_reported_not_waited_on_for_ever(capsys: Any) -> None:
    """T-0042 criterion 1 and the second half of 2: the ceiling, the message, and a report.

    The device thread takes 200 ms of audio and then stalls for good. By hand: the chirp is
    0.3 s = 30 blocks; the ring holds 20; the writer therefore needs only ``30 − 20 = 10``
    blocks (100 ms) to have been consumed to place its last block, and 200 ms have been, so
    **every chirp write succeeds** and exactly ``300 − 200 = 100 ms`` stay queued for ever. That
    is the state criterion 1 describes: ``queued_ms()`` never reaches 0, the wait gives up at
    chirp length + 2 s ≈ 2.3 s, says "queue did not drain (100 ms left)", and ``main`` exits 1
    with the report still printed and no traceback (ruling A8: a traceback is not a report).

    Today the same run reports ``write() waited 2 s`` instead, because the wait is a write loop:
    it tops the ring up to 200 ms, blocks, and hits ``write()``'s ceiling — the right exit code
    for the wrong reason, with ~11 blocks of silence written after the chirp.
    """
    with smoke_rig(stall_after_ms=RING_MS) as rig:
        ran = in_thread(
            lambda: rig.module.main(["--seconds", str(STALL_CHIRP_S)]), timeout_s=8.0, rig=rig
        )
        blocks = list(rig.writes)
        queued_ms = rig.queued_ms()
    out = capsys.readouterr().out

    assert ran.finished, (
        f"the script had not returned {ran.wall:.1f} s after the chirp on a device that stopped "
        f"draining: criterion 1 caps the wait at chirp length + 2 s "
        f"({STALL_CHIRP_S + 2.0:.1f} s). It wrote {len(blocks)} blocks"
    )
    assert ran.error is None, f"main() raised {ran.error!r} instead of reporting a device failure"
    expected_blocks = round(STALL_CHIRP_S * 1000 / BLOCK_MS)
    assert len(blocks) == expected_blocks and rig.silent_writes() == 0, (
        f"the script wrote {len(blocks)} blocks ({rig.silent_writes()} of them silent) for a "
        f"{expected_blocks}-block chirp: criterion 1 writes nothing after the chirp — the wait "
        f"polls, so a stalled device costs 2 s of waiting, not 2 s of silence pushed at it"
    )
    said = DRAIN_GAVE_UP.search(out)
    assert said is not None, (
        f"nothing in the output says the queue did not drain and how much was left "
        f"(criterion 1: 'queue did not drain (N ms left)'; {queued_ms:.1f} ms were). It "
        f"printed:\n{out}"
    )
    left = float(said.group(1))
    assert 50.0 <= left <= 150.0, (
        f"the message reports {left} ms left, but the device stopped after taking "
        f"{RING_MS:.0f} ms of a {STALL_CHIRP_S * 1000:.0f} ms chirp, so exactly 100 ms are "
        f"queued: a number in the message that is not the queue is worse than no number"
    )
    assert ran.value == 1, (
        f"main() exited {ran.value!r} on a device that never drained; criterion 1 asks for "
        f"exit 1 (the device class). It printed:\n{out}"
    )
    assert ran.wall <= STALL_CHIRP_S + WRITE_TIMEOUT_S + 3.0, (
        f"the run took {ran.wall:.1f} s; the ceiling is chirp + 2 s = {STALL_CHIRP_S + 2.0:.1f} s"
    )
    lower = out.lower()
    missing = [field for field in REQUIRED_FIELDS if field not in lower]
    assert missing == [] and "Traceback (most recent call last)" not in out, (
        f"the failing run dropped {missing} from the report (or raised): the report block is "
        f"printed on every run, failing or not (T-0035 criterion 8, ruling A8). It printed:\n"
        f"{out}"
    )
    for phrase in PEAK_ZERO_PHRASES + PEAK_WEBCAM_PHRASES:
        assert phrase.lower() in lower, (
            f"the failing run dropped {phrase!r}: the discriminator is printed on every run "
            f"(criterion 4 keeps the report block and the discriminator unchanged)"
        )


def test_a_write_timeout_is_reported_as_a_device_failure(capsys: Any) -> None:
    """T-0042 criterion 3: ``write()``'s own 2 s ceiling is reported, not hidden and not a hang.

    The device thread stops after taking 50 ms of the 0.3 s chirp. By hand: the ring holds 200 ms
    and 50 ms were consumed, so the writer places 25 of the 30 blocks and the 26th waits on a
    ring nobody empties — ``PortAudioDevice.write()`` raises ``TimeoutError`` after
    ``WRITE_TIMEOUT_S`` = 2 s. The script must turn that into the device-failure verdict with the
    device's own message, exit 1, and no traceback.

    Not marked ``pending``: the current ``main`` already catches ``TimeoutError`` and reports it,
    so this row is green today and is a **regression guard** for the fix — an implementation
    that swallowed or retried the write timeout inside the new drain wait would turn a wedged
    device back into a hang, which is the whole of T-0042. It fails today only if the failure is
    reached some other way, so its assertions pin what the run must look like: the failure came
    from the chirp (never from silence written after it), it took about one ``WRITE_TIMEOUT_S``,
    and the report is still there.
    """
    with smoke_rig(stall_after_ms=50.0) as rig:
        ran = in_thread(
            lambda: rig.module.main(["--seconds", str(STALL_CHIRP_S)]), timeout_s=8.0, rig=rig
        )
        blocks = list(rig.writes)
    out = capsys.readouterr().out

    assert ran.finished, (
        f"the script had not returned after {ran.wall:.1f} s on a device whose callback stopped "
        f"mid-chirp: write() gives up after {WRITE_TIMEOUT_S} s and criterion 3 asks that the "
        f"script report that, not hang on it"
    )
    assert ran.error is None, f"main() raised {ran.error!r} instead of reporting a device failure"
    assert ran.value == 1, (
        f"main() exited {ran.value!r} after write() raised TimeoutError; criterion 3 asks for "
        f"exit 1 with the message. It printed:\n{out}"
    )
    assert "TimeoutError" in out and "did not drain" in out, (
        f"the verdict does not carry write()'s own message ('the audio device did not drain: "
        f"write() waited 2 s'): criterion 3 reports the device failure with its message, so the "
        f"reader learns which ceiling fired. It printed:\n{out}"
    )
    expected_blocks = round(STALL_CHIRP_S * 1000 / BLOCK_MS)
    assert rig.silent_writes() == 0 and len(blocks) < expected_blocks, (
        f"the run wrote {len(blocks)} blocks ({rig.silent_writes()} silent) of a "
        f"{expected_blocks}-block chirp: the timeout must come from the chirp itself — a run "
        f"that got past the chirp and then wrote silence is the T-0042 bug wearing criterion "
        f"3's exit code"
    )
    assert WRITE_TIMEOUT_S - 0.5 <= ran.wall <= WRITE_TIMEOUT_S + 3.0, (
        f"the run took {ran.wall:.1f} s; write()'s ceiling is {WRITE_TIMEOUT_S} s, so anything "
        f"much shorter did not wait for the device and anything much longer waited twice"
    )
    lower = out.lower()
    missing = [field for field in REQUIRED_FIELDS if field not in lower]
    assert missing == [] and "Traceback (most recent call last)" not in out, (
        f"the failing run dropped {missing} from the report (or raised): a traceback is not a "
        f"report (T-0035 ruling A8). It printed:\n{out}"
    )
