"""``astra down`` and ``astra status`` (criteria 13 and 14).

The nastiest failure mode this task has: a pid file whose pid was **reused** by an unrelated
process. A test that asserted only "the file is gone" would pass a ``down`` that killed a
stranger, so the assertion here is that the victim is *still alive* afterwards.

The identity test is the process **start time** (``/proc/<pid>/stat`` field 22, clock ticks since
boot), recorded in the pid file beside the pid and the argv (T-0033 ruling A11): ``services/tts.sh``
execs ``taskset``, so a live service's ``/proc/<pid>/cmdline`` legitimately stops matching the argv
``up`` recorded, while a *reused* pid always has a different start time. So: kill iff the start
time matches; an argv mismatch under a matching start time is reported (the script exec'd) and the
process is still stopped; a different start time is never signalled.

The start time is read here from ``/proc`` directly rather than through the supervisor, so the
expectation comes from the kernel and not from the code under test.

The victim is a ``sleep 30`` this test starts and reaps itself; nothing outside the test's own
children is ever signalled. No GPU, no network, no measurement.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_astra import cli
from local_astra.config import load
from tests.unit import _box_rig as rig
from tests.unit import _service_rig as service_rig


class Probe:
    """``probe(name, url) -> None | reason``.

    ``starting``: a service that answers the **first** time it is asked is skipped or refused by
    ``up``, never started (T-0049 criterion 10), so a row that wants the plan run has to start from
    a profile that is down. It costs one extra probe round and changes nothing it asserts.
    """

    def __init__(
        self, failures: dict[str, str] | None = None, *, starting: tuple[str, ...] = ()
    ) -> None:
        self.asked: list[tuple[str, str]] = []
        self.failures = failures or {}
        self.starting = set(starting)

    def __call__(self, name: str, url: str) -> str | None:
        self.asked.append((name, url))
        if name in self.failures:
            return self.failures[name]
        if name in self.starting and sum(1 for asked, _ in self.asked if asked == name) == 1:
            return f"{name}: [Errno 111] Connection refused (not up yet)"
        return None


class ExplodingRunner:
    def __call__(self, command):
        raise AssertionError(
            f"criterion 14: `astra status` starts nothing; it tried to run {list(command)}"
        )


def run_cli(argv: list[str], **kwargs) -> int:
    try:
        return int(cli.main(argv, **kwargs))
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    except TypeError as exc:
        raise AssertionError(
            f"T-0033: cli.main(argv, *, runner=None, probe=None) — {exc}"
        ) from None


@pytest.fixture(autouse=True)
def isolate_service_logs(monkeypatch, tmp_path):
    """Point ``supervisor.log_dir()`` at an empty ``tmp_path/logs`` for every row here.

    ``status`` shares ``cli.health_rows`` with the start verb, and that renderer quotes
    ``supervisor.log_tail(name)`` under every FAILED row (T-0041 criterion 3); the unpatched
    ``log_dir()`` is the checkout's ``logs/``. A row below that probes a service unhealthy would
    otherwise read whatever the last real start on this box left there — a flake that says
    nothing about the CLI. Same fixture, same reason, as in ``tests/unit/test_cli_up.py``. No row
    here asserts on the real ``logs/`` path, so there is nothing to opt out of.
    """
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    supervisor = rig.module("supervisor")
    monkeypatch.setattr(supervisor, "log_dir", lambda: logs)
    if hasattr(cli, "log_dir"):
        monkeypatch.setattr(cli, "log_dir", lambda: logs)


def use_run_dir(monkeypatch, path):
    """Point the pid directory at ``path``; the CLI must ask ``supervisor.run_dir()`` each call."""
    supervisor = rig.module("supervisor")
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(supervisor, "run_dir", lambda: path)
    if hasattr(cli, "run_dir"):
        monkeypatch.setattr(cli, "run_dir", lambda: path)
    return supervisor


def start_victim():
    return subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def cmdline_of(pid: int) -> tuple[str, ...]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return tuple(part.decode() for part in raw.split(b"\0") if part)


def start_time_of(pid: int) -> int:
    """``/proc/<pid>/stat`` field 22 (starttime), read straight from the kernel.

    Field 2 is the executable name in parentheses and may itself contain spaces and ``)``, so the
    fields are counted from the last ``)`` — field 22 is the 20th token after it.
    """
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rindex(")") + 1 :].split()
    return int(fields[19])


def record(supervisor, path: Path, pid: int, argv: tuple[str, ...], start_time: int) -> None:
    """Write a pid file the way ``up`` does: pid, argv and the process start time (ruling A11)."""
    write = rig.attr(supervisor, "write_pid_file")
    try:
        write(path, pid, argv, start_time)
    except TypeError as exc:
        raise AssertionError(
            "T-0033 ruling A11: supervisor.write_pid_file(path, pid, argv, start_time) records "
            f"the process start time beside the pid and the argv -- {exc}"
        ) from None


def test_down_terminates_the_recorded_process_and_removes_its_file(monkeypatch, tmp_path, capsys):
    """Criterion 13: stop exactly what ``up`` started — the recorded pid, with its argv matching."""
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    victim = start_victim()
    try:
        pid_file = (tmp_path / "run") / "stt.pid"
        record(
            supervisor,
            pid_file,
            victim.pid,
            cmdline_of(victim.pid),
            start_time_of(victim.pid),
        )
        code = run_cli(["down"])
        report = capsys.readouterr()
        assert code == 0, f"criterion 13: a clean stop exits 0; it exited {code}. {report.err}"
        assert rig.wait_until(lambda: not rig.alive(victim.pid), timeout_s=5), (
            "criterion 13: the recorded process is terminated (which stops the container it fronts)"
        )
        assert not pid_file.exists(), "criterion 13: …and then the pid file is removed"
        assert "stt" in report.out + report.err, "…and the report says which service"
    finally:
        rig.reap(victim)


def test_down_twice_is_a_no_op_that_exits_zero(monkeypatch, tmp_path, capsys):
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    victim = start_victim()
    try:
        record(
            supervisor,
            (tmp_path / "run") / "tts.pid",
            victim.pid,
            cmdline_of(victim.pid),
            start_time_of(victim.pid),
        )
        assert run_cli(["down"]) == 0
        capsys.readouterr()
        second = run_cli(["down"])
        printed = capsys.readouterr()
        assert second == 0, (
            f"criterion 13: `down` twice in a row is a no-op that exits 0; it exited {second}. "
            f"{printed.out}{printed.err}"
        )
    finally:
        rig.reap(victim)


def test_a_reused_pid_is_not_killed(monkeypatch, tmp_path, capsys):
    """Criterion 13, adversarial: the pid is live but belongs to somebody else.

    The recorded **start time** does not match ``/proc/<pid>/stat`` field 22, so the process that
    was started under this pid is gone and the number has been handed to a stranger (ruling A11:
    the start time, not the argv, is the identity test — a fresh process always has a different
    one). ``down`` must kill nothing, say so, and remove the file — and the victim must still be
    alive when it is done. The recorded argv differs too, as it would for a real reuse.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    victim = start_victim()
    try:
        pid_file = (tmp_path / "run") / "vllm.pid"
        recorded = ("bash", str(tmp_path / "services" / "vllm.sh"))
        assert recorded != cmdline_of(victim.pid), "the fixture only works if the argvs differ"
        wrong_start = start_time_of(victim.pid) - 1  # any value the victim cannot have
        assert wrong_start != start_time_of(victim.pid)
        record(supervisor, pid_file, victim.pid, recorded, wrong_start)

        code = run_cli(["down"])
        report = capsys.readouterr()
        printed = report.out + report.err

        assert rig.alive(victim.pid), (
            "criterion 13, adversarial: the recorded start time does not match /proc/<pid>/stat "
            "field 22, so this pid was reused — `down` must kill nothing (ruling A11). It killed "
            "the wrong process."
        )
        assert victim.poll() is None, "…the victim is still running, not merely un-reaped"
        assert not pid_file.exists(), "criterion 13: …and the stale file is removed"
        assert code == 0, f"criterion 13: …exiting 0; it exited {code}. {printed}"
        assert "vllm" in printed, "criterion 13: …saying which file it was"
        assert any(word in printed.lower() for word in ("stale", "reused", "does not match")), (
            f"criterion 13: …and saying that it killed nothing and why; it said\n{printed}"
        )
    finally:
        rig.reap(victim)


def test_an_execd_service_whose_argv_changed_is_still_stopped(monkeypatch, tmp_path, capsys):
    """Criterion 13 under ruling A11: a matching start time means it is the process we started.

    ``services/tts.sh`` ends in ``exec taskset …``, so ``/proc/<pid>/cmdline`` never matches the
    argv ``up`` recorded. The pid is the same process all the same — same start time — so a
    ``down`` that refused here would leave every service it started running. The mismatch is
    reported; the process is terminated.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    victim = start_victim()
    try:
        pid_file = (tmp_path / "run") / "tts.pid"
        recorded = ("bash", "services/tts.sh")  # what up spawned, before the exec
        assert recorded != cmdline_of(victim.pid), "the fixture only works if the argvs differ"
        record(supervisor, pid_file, victim.pid, recorded, start_time_of(victim.pid))

        code = run_cli(["down"])
        report = capsys.readouterr()
        printed = report.out + report.err

        assert rig.wait_until(lambda: not rig.alive(victim.pid), timeout_s=5), (
            "criterion 13 / ruling A11: the start time matches, so this is the process `up` "
            "started (it exec'd, which rewrote its argv) and `down` must stop it. It did not, so "
            "`astra down` leaves tts.sh running on every box."
        )
        assert not pid_file.exists(), "criterion 13: …and the pid file is removed"
        assert code == 0, f"criterion 13: …exiting 0; it exited {code}. {printed}"
        assert "tts" in printed, "criterion 13: …saying which service"
        assert any(word in printed.lower() for word in ("exec", "argv")), (
            f"criterion 13: …and reporting the argv mismatch it stopped anyway; it said\n{printed}"
        )
    finally:
        rig.reap(victim)


def test_a_stale_pid_file_whose_process_is_gone_kills_nothing(monkeypatch, tmp_path, capsys):
    """Criterion 13: the other stale case — the pid does not exist at all."""
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    dead = start_victim()
    dead_pid = dead.pid
    dead_start = start_time_of(dead_pid)
    dead.kill()
    dead.wait(timeout=5)
    assert not rig.alive(dead_pid), "the pid is free again (and unlikely to be recycled this fast)"

    pid_file = (tmp_path / "run") / "tts.pid"
    record(supervisor, pid_file, dead_pid, ("bash", "services/tts.sh"), dead_start)
    code = run_cli(["down"])
    printed = capsys.readouterr()
    report = printed.out + printed.err
    assert code == 0, f"criterion 13: a stale file is not a failure; it exited {code}. {report}"
    assert not pid_file.exists(), "criterion 13: the file is removed"
    assert str(dead_pid) in report or "tts" in report, (
        f"criterion 13: …and it says what it found; it said\n{report}"
    )


def test_down_signals_nothing_when_there_is_no_pid_directory(monkeypatch, tmp_path, capsys):
    """Criterion 13, negative: an empty state directory is a no-op, not a sweep by name."""
    use_run_dir(monkeypatch, tmp_path / "run")
    bystander = start_victim()
    try:
        killed = []
        real_kill = os.kill
        monkeypatch.setattr(
            os, "kill", lambda pid, sig: (killed.append((pid, sig)), real_kill(pid, sig))[1]
        )
        code = run_cli(["down"])
        capsys.readouterr()
        assert code == 0
        assert [entry for entry in killed if entry[1] != 0] == [], (
            f"criterion 13: with no pid files there is nothing to stop; it signalled {killed}"
        )
        assert rig.alive(bystander.pid)
    finally:
        rig.reap(bystander)


# --- criterion 14: `status` -----------------------------------------------------------------------


def _health_lines(output: str, profile) -> dict[str, str]:
    """The rows that state a service's health: the health table's row for each service.

    Selected by the table's *shape* (``tests/unit/_service_rig.py``'s ``_HEALTH_ROW``), not by
    "the line that names the service and its endpoint" — ruling B17: every refusal a verdict row
    carries says ``healthy at <url>`` and so names both, which made this helper raise inside
    itself for any report holding one. A helper that dies inside the test rather than failing an
    assertion silently constrains what the implementation is allowed to say, which is the one
    thing a helper must never do. The counting assertion is unchanged and still bites: two
    rendered table rows for one service is still a failure.
    """
    endpoints = {
        "vllm": profile.services.vllm,
        "stt": profile.services.stt,
        "tts": profile.services.tts,
    }
    rows: dict[str, str] = {}
    for name, url in endpoints.items():
        matching = service_rig.health_rows_for(output, name, url)
        assert len(matching) == 1, (
            f"criterion 12/14: exactly one row states {name}'s health, naming its endpoint "
            f"{url}; found {matching} in\n{output}"
        )
        rows[name] = matching[0].strip()
    return rows


def test_the_health_row_helpers_read_a_report_that_holds_refusals(monkeypatch, tmp_path, capsys):
    """Ruling B17: the helpers select the health table by its shape, and still count it.

    Not a property of the CLI — a property of the two helpers the rows above lean on, pinned
    because their previous selection ("the line naming the service and its endpoint") raised an
    ``AssertionError`` *inside* the helper for any report containing a refusal: every CONFLICT row
    says ``healthy at <url>`` and so names both, as does a start driven through the real verified
    path. A helper that dies inside the test constrains what the implementation may say, and says
    it as a mystery rather than as a failing assertion (`.claude/rules/tests.md`).

    The report is a real one, not a hand-written string: three services answer their ports with no
    state record naming them, which is the 2026-09-16 shape and the case criterion 8 refuses — so
    every verdict row is a refusal carrying its endpoint, and the trap is asserted to be present
    before the helpers are asked anything. Nothing is started (the runner explodes if it is asked).
    Both halves are pinned: tolerating the refusal, and still catching a duplicated table row.
    """
    use_run_dir(monkeypatch, tmp_path / "run")
    profile = load(None).profiles["lean"]
    from tests.unit.test_cli_up import row_for

    code = run_cli(["up", "--profile", "lean"], runner=ExplodingRunner(), probe=Probe())
    printed = capsys.readouterr()
    report = printed.out + printed.err
    assert code != 0, f"criterion 8: healthy with no record is a refusal, not a PASS\n{report}"

    endpoints = {
        "vllm": profile.services.vllm,
        "stt": profile.services.stt,
        "tts": profile.services.tts,
    }
    naming_both = {
        name: [line for line in report.splitlines() if name in line and url in line]
        for name, url in endpoints.items()
    }
    assert all(len(lines) > 1 for lines in naming_both.values()), (
        "the premise of this row: a refusal names the service and its endpoint, so more than one "
        f"line does — otherwise it pins nothing. The counts are "
        f"{ {name: len(lines) for name, lines in naming_both.items()} }.\n{report}"
    )

    rows = _health_lines(report, profile)
    assert set(rows) == set(endpoints), f"one health row per service; got {rows}"
    for name, url in endpoints.items():
        assert rows[name].startswith(name) and url in rows[name], (
            f"ruling B17: the row picked for {name} is the health table's, not a refusal that "
            f"happens to name {url}; it picked {rows[name]!r}"
        )
        assert row_for(report, name, url).strip() == rows[name], (
            "the two helpers pick the same line (one table, one shape)"
        )

    duplicated = report + "\n" + rows["stt"] + "\n"
    with pytest.raises(AssertionError, match="exactly one row"):
        _health_lines(duplicated, profile)
    with pytest.raises(AssertionError, match="exactly one row"):
        row_for(duplicated, "stt", endpoints["stt"])


def test_status_prints_the_same_probe_table_as_up_and_starts_nothing(monkeypatch, tmp_path, capsys):
    """Criterion 14: same rows, same wording — one code path, pinned by comparing the two."""
    use_run_dir(monkeypatch, tmp_path / "run")
    profile = load(None).profiles["lean"]

    from tests.unit.test_cli_up import Runner

    runner, probe = Runner(), Probe(starting=("vllm", "stt", "tts"))
    assert run_cli(["up", "--profile", "lean"], runner=runner, probe=probe) == 0
    up_output = capsys.readouterr()
    up_rows = _health_lines(up_output.out + up_output.err, profile)

    assert run_cli(["status"], runner=ExplodingRunner(), probe=Probe()) == 0
    status_output = capsys.readouterr()
    status_rows = _health_lines(status_output.out + status_output.err, profile)

    assert status_rows == up_rows, (
        "criterion 14: `status` prints the same rows and the same wording as `up`'s final report "
        f"(one shared code path).\n`up`:     {up_rows}\n`status`: {status_rows}"
    )


def test_status_reports_each_pid_files_liveness(monkeypatch, tmp_path, capsys):
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    victim = start_victim()
    try:
        record(
            supervisor,
            (tmp_path / "run") / "stt.pid",
            victim.pid,
            cmdline_of(victim.pid),
            start_time_of(victim.pid),
        )
        run_cli(["status"], probe=Probe())
        printed = capsys.readouterr()
        report = printed.out + printed.err
        assert str(victim.pid) in report, (
            f"criterion 14: …plus each pid file's liveness; the pid is missing from\n{report}"
        )
        assert rig.alive(victim.pid), "criterion 14: `status` stops nothing"
    finally:
        rig.reap(victim)


def test_status_exits_non_zero_when_a_service_of_the_profile_is_unhealthy(
    monkeypatch, tmp_path, capsys
):
    use_run_dir(monkeypatch, tmp_path / "run")
    code = run_cli(["status"], probe=Probe(failures={"tts": "HTTP 503 starting"}))
    printed = capsys.readouterr()
    report = printed.out + printed.err
    assert code != 0, (
        f"criterion 14: exit non-zero if any service of the profile is not healthy; it exited "
        f"{code}. Output:\n{report}"
    )
    assert "tts" in report and "503" in report, f"…naming it; it said\n{report}"


# --- ruling A16: a service that does not go away -------------------------------------------------


#: A victim that installs SIG_IGN for SIGTERM and then says so, so the test can only signal it
#: after the handler is in place (an unhandled SIGTERM in the race window would kill it and the
#: row would silently test nothing). It is reaped with SIGKILL in ``finally``.
STUBBORN = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "open(sys.argv[1], 'w').write('ready')\n"
    "time.sleep(60)\n"
)


def start_stubborn_victim(tmp_path):
    """A process that ignores SIGTERM. Returns it once the handler is installed."""
    marker = tmp_path / "victim-ready"
    victim = subprocess.Popen(
        [sys.executable, "-c", STUBBORN, str(marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert rig.wait_until(marker.exists, timeout_s=10), (
        "the victim never installed its SIGTERM handler; the fixture would be testing nothing"
    )
    return victim


def test_stop_wait_is_at_least_dockers_stop_grace():
    """Ruling A16: 12 s = docker's 10 s stop grace plus margin.

    ``services/vllm.sh`` is a foreground ``docker run``; SIGTERM reaches the container, which gets
    ten seconds to stop before docker kills it. A supervisor that gives up before that reports
    "still running" on every ordinary engine stop.
    """
    supervisor = rig.module("supervisor")
    wait = rig.attr(supervisor, "STOP_WAIT_S")
    assert float(wait) >= 12.0, (
        f"ruling A16: supervisor.STOP_WAIT_S is 12.0 s (docker's 10 s stop grace + margin); it is "
        f"{wait}"
    )


def test_a_process_that_ignores_sigterm_is_reported_and_its_pid_file_is_kept(
    monkeypatch, tmp_path, capsys
):
    """Ruling A16, adversarial: the signalled process is still there when the wait runs out.

    The failure mode: ``down`` signals, does not check, removes the pid file and exits 0. The
    human is told the engine is stopped, the pid file that was the only record of it is gone, and
    the next ``up`` meets a port that is still bound with nothing left to stop it by. So: the
    report says it is still running, the file is **kept**, and the verb exits non-zero. No
    SIGKILL — for a foreground ``docker run`` that would orphan the container.

    Regime: no GPU, no network; ``supervisor.STOP_WAIT_S`` monkeypatched to 0.5 s (the constant is
    read at call time), one victim, wall-clock bound 5 s.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    rig.attr(supervisor, "STOP_WAIT_S")
    monkeypatch.setattr(supervisor, "STOP_WAIT_S", 0.5)
    victim = start_stubborn_victim(tmp_path)
    signals: list[tuple[int, int]] = []
    real_kill = os.kill
    monkeypatch.setattr(
        os, "kill", lambda pid, sig: (signals.append((pid, sig)), real_kill(pid, sig))[1]
    )
    try:
        pid_file = (tmp_path / "run") / "vllm.pid"
        record(
            supervisor,
            pid_file,
            victim.pid,
            cmdline_of(victim.pid),
            start_time_of(victim.pid),
        )
        started = time.monotonic()
        code = run_cli(["down"])
        elapsed = time.monotonic() - started
        report = capsys.readouterr()
        printed = report.out + report.err

        assert victim.poll() is None and rig.alive(victim.pid), (
            "the victim ignores SIGTERM, so it must still be running — if it is gone, something "
            "escalated to SIGKILL, which for a `docker run` child orphans the container "
            f"(ruling A16). It was signalled with {signals}"
        )
        assert signal.SIGKILL not in [sig for pid, sig in signals if pid == victim.pid], (
            f"ruling A16: no SIGKILL; it sent {signals}"
        )
        assert code != 0, (
            f"ruling A16: a service that is still running after the wait means `down` did not do "
            f"what it was asked, so it exits non-zero; it exited {code}. Output:\n{printed}"
        )
        assert pid_file.exists(), (
            "ruling A16: the pid file is kept — it is the only record of a process that is still "
            "running, and removing it makes the next `down` a no-op on a live engine"
        )
        assert str(victim.pid) in printed, f"…the report names the pid; it said\n{printed}"
        assert "still running" in printed.lower(), (
            f"ruling A16: …and says it is still running (the wording pinned by the ruling: "
            f'"signalled pid N; still running after 12 s — not removed"); it said\n{printed}'
        )
        assert elapsed < 5.0, (
            f"ruling A16: the wait is `supervisor.STOP_WAIT_S`, read at call time — this row "
            f"patched it to 0.5 s and the stop took {elapsed:.1f} s, so the constant was bound "
            "once (a default argument) instead of read"
        )
    finally:
        monkeypatch.setattr(os, "kill", real_kill)
        with contextlib.suppress(OSError):
            real_kill(victim.pid, signal.SIGKILL)
        rig.reap(victim)


# =================================================================================================
# T-0049 criterion 12 — `status` stops calling a healthy containerised service "not running"
#
# The other half of the 2026-09-16 lie, and the one the human saw second: after `up` overwrote the
# records with dead launcher pids, `astra status` reported the containers that were actually
# serving as "not running (stale pid file)". A pid is the wrong authority for a container. The
# record says what this CLI started; `/proc` answers for a host process and `docker inspect`
# answers for a container, and the row must say which handle it checked.
#
# Regime: no GPU, no network, no live daemon — a fake `docker` on PATH that only ever answers
# `inspect` (ruling A9). Nothing containerised is started, stopped or removed by any row here.
# =================================================================================================

#: The container that was really answering on this box (docs/tasks/T-0049.md, Goal).
LIVE_CONTAINER_ID = "61f0d405d9898f7e8098b9fd3f3b6837a4afa874711d7247f6e2a3ca5cc8df0c"


def started_ago(seconds: float) -> str:
    """A ``.State.StartedAt`` that many seconds ago, in the daemon's spelling (ruling A10)."""
    from datetime import UTC, datetime, timedelta

    moment = datetime.now(UTC) - timedelta(seconds=seconds)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond * 1000:09d}Z"


def a_dead_pid() -> int:
    """A pid that exists in no process table: a child of this test, started and reaped.

    The kernel could hand the number straight back to somebody else; the rows below assert it is
    gone, so that race is a loud failure rather than a row that quietly tests nothing.
    """
    ghost = subprocess.Popen(["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ghost.wait()
    assert not rig.alive(ghost.pid), (
        f"pid {ghost.pid} was reused between the reap and this line; re-run the row"
    )
    return ghost.pid


def container_record(supervisor, path: Path, *, pid: int, name: str = "stt") -> None:
    """The record a verified container start leaves: kind=container, with the id it resolved."""
    record = service_rig.record(
        "12",
        name=name,
        kind="container",
        pid=pid,
        start_time=None,
        argv=(f"/repo/services/{name}.sh",),
        container_name=f"astra-{name}",
        container_id=LIVE_CONTAINER_ID,
        started_at=time.time() - 7200,
    )
    service_rig.attr(supervisor, "write_state_record", "12")(path, record)


def status_lines(report: str, name: str) -> list[str]:
    """The lines about ``name`` that are not the shared health table's row."""
    return service_rig.verdict_lines(report, name)


def test_status_reports_a_running_container_as_running(monkeypatch, tmp_path, capsys):
    """Criterion 12: the containerised service the CLI started is running, and the row says so.

    The recorded launcher pid is **dead** — which is exactly the state `up` left the box in — and
    the container is up. Today's row reads ``not running (stale pid file)`` about a service that
    has been serving for two hours, because it asks ``/proc`` about a handle that lives in the
    daemon. The right authority is asked instead.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    service_rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=service_rig.container_json(
            container_id=LIVE_CONTAINER_ID, started_at=started_ago(7200)
        ),
    )
    dead = a_dead_pid()
    container_record(supervisor, (tmp_path / "run") / "stt.pid", pid=dead)

    code = run_cli(["status"], runner=ExplodingRunner(), probe=Probe())
    printed = capsys.readouterr()
    report = printed.out + printed.err
    rows = status_lines(report, "stt")
    text = "\n".join(rows)

    assert rows, f"criterion 12: `status` says something about every record it holds:\n{report}"
    assert "stale" not in text.lower() and "not running" not in text.lower(), (
        "criterion 12: the container is running, and the recorded launcher pid being gone says "
        "nothing about it — a pid is not the authority for a container. The row said:\n" + text
    )
    assert "running" in text.lower(), (
        f"criterion 12: …it is reported as running. The row said:\n{text}"
    )
    assert code == 0, f"…and nothing here is unhealthy, so `status` exits 0; it exited {code}"


def test_status_names_the_handle_it_checked(monkeypatch, tmp_path, capsys):
    """Criterion 12: a row that does not name the handle cannot be checked by the human.

    "Running" is a claim about *something*; the row says which container id it asked the daemon
    about, so `docker ps` can confirm or refute it. Naming the dead launcher pid alone is how the
    human on 2026-09-16 was told a two-hour-old container was freshly started.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    service_rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=service_rig.container_json(
            container_id=LIVE_CONTAINER_ID, started_at=started_ago(7200)
        ),
    )
    container_record(supervisor, (tmp_path / "run") / "stt.pid", pid=a_dead_pid())

    run_cli(["status"], runner=ExplodingRunner(), probe=Probe())
    printed = capsys.readouterr()
    report = printed.out + printed.err
    text = "\n".join(status_lines(report, "stt"))

    assert LIVE_CONTAINER_ID[:12] in text, (
        "criterion 12: the row names the handle it checked — the container id, which is what "
        f"`docker inspect` was asked about. It said:\n{text}"
    )


def test_status_says_healthy_but_not_the_instance_this_cli_started(monkeypatch, tmp_path, capsys):
    """Criterion 12, the honest half: the endpoint answers, the recorded handle is gone.

    The daemon *answers* — the container this CLI recorded has exited — while the port is still
    served by something. That is not "running", and it is not a stale file to be shrugged at
    either: `astra down` cannot stop what is on the port. The row has to say both halves.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    service_rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=service_rig.container_json(
            container_id=LIVE_CONTAINER_ID, started_at=started_ago(7200), running=False
        ),
    )
    container_record(supervisor, (tmp_path / "run") / "stt.pid", pid=a_dead_pid())

    code = run_cli(["status"], runner=ExplodingRunner(), probe=Probe())
    printed = capsys.readouterr()
    report = printed.out + printed.err
    text = "\n".join(status_lines(report, "stt"))

    assert "not running" in text.lower() or "gone" in text.lower() or "exited" in text.lower(), (
        "criterion 12: the daemon says the recorded container is not running, so the row may not "
        f"call it running. It said:\n{text}"
    )
    assert "not the instance" in text.lower() or "did not start" in text.lower(), (
        "criterion 12: …and, since the endpoint answers anyway, it says plainly that what is "
        f"serving is not the instance this CLI started. It said:\n{text}\n\nthe whole report:\n"
        f"{report}"
    )
    assert code is not None, "…the verb still returns an exit code rather than raising"


def test_status_starts_nothing_signals_nothing_and_writes_nothing(monkeypatch, tmp_path, capsys):
    """Criterion 12, negative: asking a question may not change the answer.

    Deliberately **not** pending: this is a fence, true today and required to stay true, so it
    carries no marker (a pending row that passes is an XPASS-strict failure, and rightly).

    `status` grew an authority that can be asked about containers; ``docker`` can also stop, kill
    and remove them. The fake refuses every subcommand but ``inspect`` (that is the dynamic half
    of criterion 6's negative), and this row adds the process half: no signal, and the run
    directory byte-identical afterwards.
    """
    supervisor = use_run_dir(monkeypatch, tmp_path / "run")
    docker = service_rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=service_rig.container_json(
            container_id=LIVE_CONTAINER_ID, started_at=started_ago(7200)
        ),
    )
    victim = start_victim()
    try:
        container_record(supervisor, (tmp_path / "run") / "stt.pid", pid=a_dead_pid())
        record(
            supervisor,
            (tmp_path / "run") / "tts.pid",
            victim.pid,
            cmdline_of(victim.pid),
            start_time_of(victim.pid),
        )
        before = service_rig.snapshot(tmp_path / "run")
        spy = service_rig.SignalSpy(monkeypatch)
        run_cli(["status"], runner=ExplodingRunner(), probe=Probe())
        capsys.readouterr()

        assert spy.all_signals() == [], (
            f"criterion 12, negative: `status` signals nothing; it sent {spy.all_signals()}"
        )
        assert rig.alive(victim.pid), "…so the process it asked about is still running"
        assert service_rig.snapshot(tmp_path / "run") == before, (
            "criterion 12, negative: …and it writes nothing — a verb that repairs the record it "
            "is reporting on has no fixed point, and the record is the only provenance there is"
        )
        for call in docker.calls:
            assert call and call[0] == "inspect", (
                f"criterion 6/12, negative: the only thing asked of the daemon is a read; it ran "
                f"`docker {' '.join(call)}`"
            )
    finally:
        rig.reap(victim)
