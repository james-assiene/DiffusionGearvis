"""T-0057 criteria 1 and 2: the thread-leak guard, tested by running pytest inside pytest.

The guard is an autouse fixture in ``tests/conftest.py``; a fixture cannot be asserted about from
inside the suite it protects (it would have to leak a thread to see itself fire, and then the
outer row would be the one that failed). So every row here runs a *second* pytest over a two- or
three-row module written into a uniquely named directory, and reads that run's outcomes. This is
the ``test_pytest_config.py`` / ``test_suite_budget.py`` precedent.

Both directions, because a guard needs a row for being too wide as well as one for being absent
(.claude/rules/tests.md):

* absent -- a row that leaves a thread running is red, by name, and the innocent row after it is
  not; a *daemon* thread does not get a pass, because a daemon thread still runs and still polls;
* too wide -- a thread a session fixture started before any row runs is attributed to nobody, a
  thread an earlier row leaked is not re-reported against its successor, ``ThreadPoolExecutor``
  and unnamed interpreter threads are never mentioned, and an ``astra-`` thread that finished is
  not a leak.

The adversarial row is the reason the whole task exists: the failure mode is not that the thread
keeps running, it is that it keeps running and nobody notices, because all it does is *read*. It
reproduces T-0049 ruling B18 as a red row -- the leaked writer reading the *next* row's
instrumented device.

Regime: each inner run is a fresh ``python -m pytest`` subprocess, cwd inside a unique directory
under pytest's ``tmp_path`` (never a shared scratchpad name), ``PYTHONPATH`` pointing at this
worktree, and the origin of both ``tests.conftest`` and ``local_astra.conductor.audio_io`` printed
from *inside* that process and asserted here -- ``sys.path[0]`` is the cwd and is searched before
``PYTHONPATH``, so an unproven origin is not evidence (.claude/rules/tests.md).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The outcome words ``-rA`` prints, in the summary lines the rig parses.
RED = ("FAILED", "ERROR")

INNER_CONFTEST = """
import os
import sys

sys.path.insert(0, {repo!r})

from tests.conftest import (  # noqa: E402,F401
    astra_audio_modules_are_stopped,
    astra_thread_leak_guard,
)


def pytest_report_header(config):
    import tests.conftest as inner_conftest

    import local_astra.conductor.audio_io as audio_io

    return [
        "t0057 origin tests.conftest: " + inner_conftest.__file__,
        "t0057 origin audio_io: " + audio_io.__file__,
        "t0057 cwd: " + os.getcwd(),
    ]
"""


class InnerRun:
    """One pytest-inside-pytest run: its exit code, its output and its per-row outcomes."""

    def __init__(self, proc: subprocess.CompletedProcess[str], where: Path) -> None:
        self.proc = proc
        self.where = where
        self.out = proc.stdout + proc.stderr
        self.outcomes: dict[str, set[str]] = {}
        for line in self.out.splitlines():
            m = re.match(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+::\S+)", line.strip())
            if m:
                self.outcomes.setdefault(m.group(2).split("::")[-1], set()).add(m.group(1))

    @property
    def returncode(self) -> int:
        return self.proc.returncode

    def red(self, name: str) -> bool:
        return bool(self.outcomes.get(name, set()) & set(RED))

    def green(self, name: str) -> bool:
        return self.outcomes.get(name, set()) == {"PASSED"}

    def message_for(self, name: str) -> str:
        """Every output line that names this row in a failure context."""
        return "\n".join(ln for ln in self.out.splitlines() if name in ln)

    def tail(self, n: int = 40) -> str:
        return "\n".join(self.out.splitlines()[-n:])


def run_inline(tmp_path: Path, name: str, module_src: str) -> InnerRun:
    """Run ``module_src`` under the real guard, in a uniquely named directory of its own."""
    where = tmp_path / f"t0057-{name}-{os.getpid()}"
    where.mkdir(parents=True)
    (where / "conftest.py").write_text(INNER_CONFTEST.format(repo=str(REPO)))
    module = where / f"test_{name}.py"
    module.write_text(module_src)
    env = {**os.environ, "PYTHONPATH": str(REPO), "ASTRA_PENDING_REPORT": ""}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(module),
            "-p",
            "no:cacheprovider",
            "-o",
            "asyncio_mode=auto",
            "-rA",
            "--tb=long",
        ],
        cwd=where,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    run = InnerRun(proc, where)
    conftest_origin = re.search(r"t0057 origin tests\.conftest: (\S+)", run.out)
    audio_origin = re.search(r"t0057 origin audio_io: (\S+)", run.out)
    assert conftest_origin and audio_origin, (
        "the inner run did not print where it imported the guard and the module under test "
        f"from; without that this is not evidence (.claude/rules/tests.md)\n{run.tail(60)}"
    )
    assert Path(conftest_origin.group(1)) == REPO / "tests" / "conftest.py", (
        f"the inner run imported the guard from {conftest_origin.group(1)}, not from the tree "
        f"under test ({REPO})"
    )
    assert Path(audio_origin.group(1)) == REPO / "local_astra" / "conductor" / "audio_io.py", (
        f"the inner run imported audio_io from {audio_origin.group(1)}, not from {REPO}"
    )
    return run


# ---------------------------------------------------------------------------------------------
# Criterion 1 -- the leak is a red row, named, and only the row that caused it.
# ---------------------------------------------------------------------------------------------

LEAK_AND_INNOCENT = '''
import threading
import time


def test_leaks_a_thread():
    """Starts a thread that outlives the row, exactly as the first play() of a row does."""
    threading.Thread(target=time.sleep, args=(0.6,), name="astra-fake-writer").start()


def test_is_innocent():
    """Runs while the previous row's thread is still alive. It did not start it."""
    assert any(t.name == "astra-fake-writer" for t in threading.enumerate()), (
        "premise of the wrong-attribution case: the leaked thread is still alive here"
    )
'''


def test_a_thread_left_running_fails_the_test_that_started_it(tmp_path: Path) -> None:
    """Criterion 1: an AssertionError naming the creator's nodeid and the thread's name.

    Today the only signal that this happens at all is a ``PytestUnhandledThreadExceptionWarning``
    that shows up in one run out of three, and nothing at all when the leaked worker raises
    nothing. This row is the whole of "instead of warning".
    """
    run = run_inline(tmp_path, "leak", LEAK_AND_INNOCENT)

    assert run.returncode != 0, (
        f"the inner run exited 0: a leaked thread is a failed row, not a warning and not a "
        f"summary line (T-0057 criterion 1)\n{run.tail()}"
    )
    assert run.red("test_leaks_a_thread"), (
        f"the row that started the thread was not reported red; outcomes {run.outcomes}\n"
        f"{run.tail()}"
    )
    message = run.message_for("test_leaks_a_thread")
    assert "astra-fake-writer" in message, (
        "the failure does not name the thread, so the reader cannot tell which subsystem leaked "
        f"it (criterion 1b):\n{message}"
    )
    assert "AssertionError" in run.out, (
        f"the guard must fail with an AssertionError, not a warning class\n{run.tail()}"
    )


def test_the_innocent_row_after_a_leak_is_not_blamed_for_it(tmp_path: Path) -> None:
    """Criterion 2: a thread alive *before* a row started is never attributed to it.

    The double-report case. Without it the guard turns one leak into a cascade of red rows and
    the next reader turns the guard off instead of fixing the leak.
    """
    run = run_inline(tmp_path, "innocent", LEAK_AND_INNOCENT)

    assert run.green("test_is_innocent"), (
        f"the row after the leak was reported {run.outcomes.get('test_is_innocent')}; it did not "
        f"start the thread and must pass (T-0057 criterion 2)\n{run.tail()}"
    )
    assert "test_is_innocent" not in run.message_for("astra-fake-writer"), (
        "the innocent row is named in the leak report; the leak belongs to its creator\n"
        + run.message_for("astra-fake-writer")
    )


DAEMON_POLLER = '''
import threading
import time


def test_leaves_a_daemon_poller():
    """A daemon thread polling every 0.2 ms: the shape of astra-audio-writer exactly."""

    def poll():
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            time.sleep(0.0002)

    threading.Thread(target=poll, name="astra-daemon-poller", daemon=True).start()
'''


def test_a_daemon_thread_does_not_satisfy_the_guard(tmp_path: Path) -> None:
    """``daemon=True`` is not a lifecycle. T-0057 keeps the writer a daemon on purpose (making it
    non-daemon would hang ``astra session`` on Ctrl-C and collides with T-0040), so a guard that
    tolerated daemon threads would be green on the very defect it exists for: the thread still
    runs, still reads the device ~4 000 times a second and still burns the CPU that made
    ``test_audio_clock.py`` the one module in the suite with real in-process CPU.
    """
    run = run_inline(tmp_path, "daemon", DAEMON_POLLER)

    assert run.red("test_leaves_a_daemon_poller"), (
        f"a live daemon thread was not reported; outcomes {run.outcomes}\n{run.tail()}"
    )
    assert "astra-daemon-poller" in run.out, (
        f"the daemon thread is not named in the report\n{run.tail()}"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 2 -- the guard is not too wide.
# ---------------------------------------------------------------------------------------------

SESSION_THREAD = '''
import threading

import pytest


@pytest.fixture(scope="session", autouse=True)
def a_session_helper():
    """Started before any row runs; it belongs to the session, not to a row."""
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="astra-session-helper", daemon=True)
    worker.start()
    yield
    stop.set()
    worker.join(timeout=1.0)


def test_one():
    assert any(t.name == "astra-session-helper" for t in threading.enumerate())


def test_two():
    assert any(t.name == "astra-session-helper" for t in threading.enumerate())
'''


def test_the_guard_ignores_threads_it_did_not_cause(tmp_path: Path) -> None:
    """Criterion 2: a session-scoped helper thread fails nobody.

    The too-wide direction that matters most in practice, because it is how an autouse guard that
    runs after every row in the suite fails the gate for every agent on the box.
    """
    run = run_inline(tmp_path, "session", SESSION_THREAD)

    assert run.returncode == 0, (
        "a thread a session fixture started before any row ran was attributed to a row; "
        f"outcomes {run.outcomes} (T-0057 criterion 2)\n{run.tail(60)}"
    )
    assert run.green("test_one") and run.green("test_two"), (
        f"outcomes {run.outcomes} -- neither row started the helper\n{run.tail()}"
    )


INTERPRETER_THREADS = '''
import threading
import time
from concurrent.futures import ThreadPoolExecutor

POOL = ThreadPoolExecutor(max_workers=2)


def test_leaves_the_interpreters_own_threads_running():
    """A pool worker and an unnamed thread outlive this row. Neither is this project's."""
    POOL.submit(time.sleep, 0.5)
    threading.Thread(target=time.sleep, args=(0.5,), name="Thread-noise", daemon=True).start()


def test_an_astra_thread_that_finished_is_not_a_leak():
    worker = threading.Thread(target=lambda: None, name="astra-short-lived")
    worker.start()
    worker.join()
'''


def test_the_guard_names_only_this_projects_threads(tmp_path: Path) -> None:
    """Criterion 2: the tolerance is a *prefix* rule plus the before-set, not a list of names.

    ``ThreadPoolExecutor-*``, ``asyncio_0`` and the xdist workers are all outside ``astra-`` and
    need no entry anywhere; and an ``astra-`` thread that finished inside its row is not alive, so
    it is not a leak. Enumerating allowed names is how ``make fast-parallel`` would start failing
    the day a new interpreter thread appears.
    """
    run = run_inline(tmp_path, "interpreter", INTERPRETER_THREADS)

    assert run.returncode == 0, (
        f"the guard reported threads that are not this project's; outcomes {run.outcomes}\n"
        f"{run.tail(60)}"
    )
    assert "Thread-noise" not in run.out and "ThreadPoolExecutor" not in run.out, (
        f"the guard named an interpreter thread it must ignore\n{run.tail(60)}"
    )


# ---------------------------------------------------------------------------------------------
# The adversarial row -- the leak that only ever reads (T-0049 ruling B18).
# ---------------------------------------------------------------------------------------------

LEAKED_WRITER_READS = '''
import time

from tests.fakes.audio_device import (
    FakeAudioDevice,
    PcmUtterance,
    make_audio_io,
    play,
    play_out,
    tone,
)

STATE = {}


async def test_one_plays_a_tone_and_never_flushes():
    """What 48 of the 66 rows in the four audio modules do, and then stop asserting."""
    device = FakeAudioDevice()
    STATE["io"] = make_audio_io(device)
    STATE["device"] = device
    await play(STATE["io"], PcmUtterance(samples=tone(200.0), text="hi"), source="hi")
    await play_out(device, 200.0)


def test_two_counts_what_the_previous_row_left_running():
    """Instrument the device the way a later row would instrument a clock, and wait 50 ms."""
    device = STATE["device"]
    reads = []
    real_queued_ms = device.queued_ms

    def counting_queued_ms():
        reads.append(time.monotonic())
        return real_queued_ms()

    device.queued_ms = counting_queued_ms
    time.sleep(0.05)

    assert len(reads) == 0, (
        "the previous row's writer thread made %d reads of this row's instrumented device in "
        "50 ms. Nothing in this row played anything: the number belongs to the row before it "
        "(T-0049 ruling B18, T-0057 adversarial row)" % len(reads)
    )
'''


def test_a_leaked_writer_reads_the_next_test_s_patched_clock(tmp_path: Path) -> None:
    """The failure mode this task is really about: a leak that only reads, so nothing crashes.

    Measured on this tree before the fix: **199** reads of ``FakeAudioDevice.queued_ms`` in the
    50 ms after the row ended, 7 960 over the thread's full 1.99 s of afterlife. The moment a
    later row patches a clock globally, or a fake clock counts or advances on read, that is a
    wrong *pass* rather than a warning -- and a wrong pass in the audio suite is a barge-in
    latency number that belongs to the previous test.

    The row the fix must make green is the *second* one. The reviewer's check is that it goes red
    again when the stop seam is taken out: a row that is green because the thread happened to
    have exited already has learned nothing.
    """
    run = run_inline(tmp_path, "adversarial", LEAKED_WRITER_READS)

    assert run.green("test_two_counts_what_the_previous_row_left_running"), (
        "a thread the previous row started is still reading this row's device: "
        f"outcomes {run.outcomes}\n{run.tail(50)}"
    )
    assert run.green("test_one_plays_a_tone_and_never_flushes"), (
        "the row that played the tone is red -- with the stop seam in place and the teardown in "
        f"tests/conftest.py calling it, playing a tone leaks nothing; outcomes {run.outcomes}\n"
        f"{run.tail(50)}"
    )


# ---------------------------------------------------------------------------------------------
# The guard is actually installed, in the right order, for every row of the real suite.
# ---------------------------------------------------------------------------------------------


def test_the_guard_and_the_audio_teardown_run_for_every_row(
    request: pytest.FixtureRequest,
) -> None:
    """Criterion 1 is worth nothing if the fixture is defined and not autouse.

    Checked from inside an ordinary row of the real suite: both fixtures are in *this* row's
    fixture closure, and the audio teardown depends on the guard -- fixtures are finalised
    last-in-first-out, so that dependency is what makes the modules stop *before* the guard
    counts threads. Flip it and every audio row in the suite goes red.
    """
    import inspect

    from tests import conftest

    for name in ("astra_thread_leak_guard", "astra_audio_modules_are_stopped"):
        assert name in request.fixturenames, (
            f"{name} is not in the fixture closure of an ordinary row; the guard must be autouse "
            "(T-0057 criterion 1)"
        )
    stopper = conftest.astra_audio_modules_are_stopped
    params = inspect.signature(getattr(stopper, "__wrapped__", stopper)).parameters
    assert "astra_thread_leak_guard" in params, (
        "astra_audio_modules_are_stopped no longer depends on astra_thread_leak_guard, so its "
        "teardown may run after the thread count instead of before it; every row that builds a "
        "playback module would then be reported as leaking one"
    )
