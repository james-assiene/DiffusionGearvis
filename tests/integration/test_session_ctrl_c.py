"""T-0040: Ctrl-C on the command the human actually runs.

``uv run astra session --silent --serve-hud`` is the first thing a human types at this repo, and
Ctrl-C in a terminal delivers SIGINT to the whole foreground process group — to ``uv`` *and* to
the Python child, and ``uv`` forwards its copy, so the session sees two. The unit rows fire both
signals from inside the shutdown path; this row is the one that spends a real process group, a
real terminal-style signal and the real ``hud_static`` socket, because the escape seen live
(``logs/t0039-live-82d8462/run.log``: ``KeyboardInterrupt`` inside ``hud_static.close()`` ->
``HTTPServer.shutdown()`` -> ``Event.wait()``, exit 130) is a property of signal delivery that no
in-process test can fully stand in for.

**How to run it.** ``tests/integration`` is ``/launch``-only by its README because the modules
there spend hardware; this one spends none — no GPU, no audio device, no service — so it runs
when the module is named explicitly::

    uv run pytest tests/integration/test_session_ctrl_c.py -q

**Regime.** No GPU and no service: a session starts and serves its endpoint with vLLM, moshi-server
and the TTS all absent — the clients are dialled lazily, when a turn needs them, and this row takes
no turn (verified on this box 2026-09-15: the endpoint accepted a connection 0.6 s after spawn with
nothing else running). ``--silent`` means no audio device and no microphone. Every wall-clock number
below is a ceiling, not a measurement: nothing here may be quoted as latency.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import threading
import time

import pytest

pytestmark = [pytest.mark.integration]

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Ceilings, all of them generous: a box under ``-n 8`` is slower than an idle one.
START_CEILING_S = 60.0
EXIT_CEILING_S = 30.0
PORT_RELEASE_CEILING_S = 5.0
CLOSED_LINE_CEILING_S = 30.0

#: The line ``session_cmd`` prints once the first stop signal has been serviced: the session is
#: closed, the static server is closed, and what is left is the interpreter shutting down. The
#: duplicate signals below are aimed at exactly that stretch (see ``SECOND_GAP_S``).
CLOSED_LINE = "session closed"

#: The gap before the *third* signal: where the copy ``uv`` forwards lands (~150-350 ms after the
#: one the terminal delivered, measured on this box, uv 0.12.13). Wall-clock scheduling alone is a
#: coin toss — a session that closes in tens of milliseconds may already be gone — which is why
#: the *second* signal is fired off the stdout line instead, and why the row below sweeps.
THIRD_GAP_S = 0.25

#: The storm row: one signal every ``STORM_PERIOD_S`` from the closing line until the process is
#: gone, for at most ``STORM_CEILING_S``. Measured on this box 2026-09-14 (idle, uv 0.12.13,
#: CPython 3.11.16, n=4): the closing line precedes exit by 63 ms when no further signal arrives,
#: so a single shot placed by a wall clock after it lands on a corpse about as often as on the
#: window it is aimed at; the sweep covers the whole stretch instead of sampling two points of it.
STORM_PERIOD_S = 0.005
STORM_CEILING_S = 5.0


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


class _Pump:
    """Drain one pipe on a thread, keeping the text and flagging a line as it goes by.

    Two reasons it is not ``proc.communicate()``: this row has to *react* to a line while the
    process is still alive, and a pipe nobody reads fills up and stalls the child.
    """

    def __init__(self, stream, needle: str) -> None:
        self._stream = stream
        self._needle = needle
        self.seen = threading.Event()
        self._lines: list[str] = []
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        with contextlib.closing(self._stream):
            for line in self._stream:
                self._lines.append(line)
                if self._needle and self._needle in line:
                    self.seen.set()

    def text(self) -> str:
        self._thread.join(timeout=EXIT_CEILING_S)
        return "".join(self._lines)


def signal_the_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
    """Ctrl-C reaches the group, not the leader. A group that is already gone is not an error."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


class _Session:
    """A spawned ``astra session --silent --serve-hud`` with both pipes drained."""

    def __init__(self, proc, out, err, conductor_port: int, hud_port: int, data_dir) -> None:
        self.proc = proc
        self.out = out
        self.err = err
        self.conductor_port = conductor_port
        self.hud_port = hud_port
        self.data_dir = data_dir


def spawn_session(tmp_path) -> _Session:
    conductor_port, hud_port = free_port(), free_port()
    config = tmp_path / "astra.toml"
    config.write_text(
        "profile = 'lean'\n[profiles.lean.services]\n"
        f"conductor_port = {conductor_port}\nhud_port = {hud_port}\n"
    )
    data_dir = tmp_path / "data"
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "astra",
            "--config",
            str(config),
            "session",
            "--silent",
            "--serve-hud",
            "--data-dir",
            str(data_dir),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Unbuffered, or the line these rows react to sits in an 8 KiB block buffer until exit and
        # there is nothing left to interrupt by the time it becomes readable.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,  # its own process group: the signals go where Ctrl-C's go
    )
    return _Session(
        proc,
        _Pump(proc.stdout, CLOSED_LINE),
        _Pump(proc.stderr, ""),
        conductor_port,
        hud_port,
        data_dir,
    )


def wait_until_serving(session: _Session) -> None:
    proc = session.proc
    serving = wait_until(
        lambda: listening(session.conductor_port) or proc.poll() is not None, START_CEILING_S
    )
    assert serving and proc.poll() is None, (
        f"the session must be serving before it is interrupted; poll={proc.poll()} "
        f"(no service is needed for this: the clients are dialled lazily)"
    )


def assert_stopped_cleanly(session: _Session, closed_line_seen: bool, signals: int) -> None:
    """The whole of criteria 1 and 16 that survives into a subprocess, asserted in one place."""
    proc = session.proc
    stdout, stderr = session.out.text(), session.err.text()
    assert closed_line_seen, (
        "T-0040 criterion 1: one line, servicing the first stop signal, before the duplicates "
        f"land; stdout held:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
    )
    assert "Traceback" not in stderr, (
        f"T-0040 criterion 1: {signals} stop signals, no traceback on stderr — including the ones "
        f"that land after the closing line, while the interpreter is finalising (CPython runs "
        f"``threading._shutdown`` before the ``atexit`` hooks, so a disposition restored on the "
        f"way out of ``session_cmd`` leaves that stretch uncovered). stderr held:\n{stderr[-2000:]}"
    )
    assert "KeyboardInterrupt" not in stderr, (
        f"…and no KeyboardInterrupt escapes; stderr held:\n{stderr[-2000:]}"
    )
    assert proc.returncode == 0, (
        f"T-0040 criterion 1: {signals} SIGINTs exit 0; it exited {proc.returncode}. "
        f"stderr:\n{stderr[-2000:]}\nstdout:\n{stdout[-2000:]}"
    )
    assert wait_until(lambda: not listening(session.conductor_port), PORT_RELEASE_CEILING_S), (
        f"criterion 16/1: the conductor endpoint {session.conductor_port} is still listening "
        f"{PORT_RELEASE_CEILING_S} s after exit"
    )
    assert wait_until(lambda: not listening(session.hud_port), PORT_RELEASE_CEILING_S), (
        f"criterion 1: the HUD static server {session.hud_port} is still listening after exit"
    )
    turns = session.data_dir / "turns.jsonl"
    assert not turns.exists() or turns.read_text().strip() == "", (
        f"criterion 16, negative: no turns.jsonl line at shutdown; it holds {turns.read_text()!r}"
    )


def test_two_ctrl_cs_exit_zero_print_no_traceback_and_free_both_ports(tmp_path):
    """T-0040 criterion 1, end to end: the demo does not print a traceback when you stop it.

    The duplicate signals are *scheduled off the child's own stdout*, not off a wall clock: the
    second one goes out the moment the ``session closed`` line appears, which is the last thing
    ``session_cmd`` prints — everything after it is interpreter shutdown, the stretch the live run
    died in and the stretch the ``atexit``-installed ``SIG_IGN`` exists for. A third follows a
    quarter of a second later, where ``uv``'s forwarded copy lands (ruling A5 F2).
    """
    if shutil.which("uv") is None:
        pytest.skip("this row runs the command the human runs, which starts with `uv run`")
    session = spawn_session(tmp_path)
    proc = session.proc
    try:
        wait_until_serving(session)
        signal_the_group(proc, signal.SIGINT)
        closed_line_seen = session.out.seen.wait(CLOSED_LINE_CEILING_S)
        signal_the_group(proc, signal.SIGINT)  # the one the live run choked on
        time.sleep(THIRD_GAP_S)
        signal_the_group(proc, signal.SIGINT)  # and the one `uv` forwards, late
        proc.wait(timeout=EXIT_CEILING_S)
    finally:
        if proc.poll() is None:  # a hang is a failure, not a leak
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=EXIT_CEILING_S)
    assert_stopped_cleanly(session, closed_line_seen, signals=3)


def test_ctrl_c_held_down_through_interpreter_shutdown_prints_no_traceback(tmp_path):
    """Adversarial, criterion 1: the window is *every* instant between the first signal and exit.

    A human who stops a demo that does not die instantly presses Ctrl-C again, and again. One
    duplicate placed by hand samples the window; this row sweeps it — a SIGINT every few
    milliseconds from the closing line until the process is gone — so "absorbed" has to mean
    absorbed at every point in the stretch, not at the two points somebody happened to pick.

    What the sweep is aimed at: CPython joins the non-daemon threads (``threading._shutdown``)
    *before* it runs the ``atexit`` hooks, so a SIGINT disposition that is restored on the way out
    of ``session_cmd`` and only re-deafened by an ``atexit`` hook leaves that join uncovered, and
    a signal landing there prints ``Exception ignored in: <module 'threading'> … KeyboardInterrupt``
    — a traceback on stderr, which criterion 1 forbids, even though the exit code stays 0.

    **Measured on this box 2026-09-14** (idle, uv 0.12.13, CPython 3.11.16, ``--silent
    --serve-hud``, no service running): neither row here is *reliably* red on an implementation
    with that gap — the module was fully green in 4 of 10 runs and each row failed 6 of 15 — and
    both are reliably red (6 of 6 runs, both rows) with the ``atexit.register`` line removed. So a
    green run of this module is evidence only when it is a run of *many*; a red one is a defect.
    """
    if shutil.which("uv") is None:
        pytest.skip("this row runs the command the human runs, which starts with `uv run`")
    session = spawn_session(tmp_path)
    proc = session.proc
    shots = 0
    try:
        wait_until_serving(session)
        signal_the_group(proc, signal.SIGINT)
        shots += 1
        closed_line_seen = session.out.seen.wait(CLOSED_LINE_CEILING_S)
        deadline = time.monotonic() + STORM_CEILING_S
        while proc.poll() is None and time.monotonic() < deadline:
            signal_the_group(proc, signal.SIGINT)
            shots += 1
            time.sleep(STORM_PERIOD_S)
        proc.wait(timeout=EXIT_CEILING_S)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=EXIT_CEILING_S)
    assert shots >= 2, "the storm must outlast the closing line, or it pins nothing"
    assert_stopped_cleanly(session, closed_line_seen, signals=shots)
