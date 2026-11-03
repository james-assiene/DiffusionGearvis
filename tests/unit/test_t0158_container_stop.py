"""``down`` stops the STT container, not just its ``docker run`` front end (T-0158).

The observed bug (2026-10-26, ``logs/t0128c-smoke-ffba98c-run.log``): ``supervisor.stop`` sent
SIGTERM to the launcher pid, which is the ``docker run --rm --name astra-stt`` client that
``services/stt.sh`` exec'd. The client forwards the signal into the container, where
``moshi-server`` is PID 1 with no handler, so the kernel drops it. Ruling A16 forbids SIGKILL, so
``down`` said "still running after 12 s" and exited 1 — and a later ``down`` whose SIGTERM did make
the client exit removed the record while ``astra-stt`` kept the GPU.

After T-0158 a ``kind == "container"`` record is stopped **through the daemon**, by the container
id recorded in it: ``docker stop --time <DOCKER_STOP_GRACE_S> <id>``, then ``docker inspect <id>``
must say "no such object" or ``Running == false`` before the record is released. "Could not ask"
keeps the record (service-lifecycle rule 2). The supervisor still never sends SIGKILL.

Every expectation below comes from ``docs/tasks/T-0158.md`` — the criteria, and the real client's
answers recorded on this box (docker 29.4.1, 2026-10-26) that the fake reproduces: ``inspect`` of a
missing ref prints ``[]`` and ``error: no such object: <ref>`` (lower case) and exits 1; ``stop``
of one says ``Error response from daemon: No such container: <ref>``; ``--time`` still works but
prints a deprecation notice on stderr; the grace is an integer flag.

Regime: CPU, no daemon, no network, no GPU. A fake ``docker``/``sg`` first on PATH
(``tests/unit/_service_rig.fake_docker(..., allow_stop=True)``), a front-end double that ignores
SIGTERM (``_service_rig.container_front_end``), ``STOP_WAIT_S`` patched to 0.5 s and
``DOCKER_STOP_GRACE_S`` to 2.0 s (so the grace argument is ``2``), 5 s wall bound on the timed row.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from local_astra import cli
from tests import _git_range as gr
from tests.unit import _box_rig as box
from tests.unit import _service_rig as rig

REPO = Path(__file__).resolve().parents[2]

#: The container this CLI recorded (64 hex digits, as ``.Id`` is), and a stranger.
X = "c0ffee5eed" + "1" * 54
Y = "deadbeef00" + "2" * 54
STARTED_AT = "2026-10-26T09:00:00.000000000Z"

#: The patched constants of the Regime. ``GRACE_ARG`` is how docker's integer flag spells 2.0 s.
STOP_WAIT_S = 0.5
GRACE_S = 2.0
GRACE_ARG = "2"


def stop_shape(argv) -> tuple:
    """``("stop", "--time", <int seconds>, ref)``: the grace must parse as a whole number of
    seconds (docker's flag is an int; orchestrator ruling 2026-10-26), compared as an int."""
    argv = tuple(argv)
    if len(argv) == 4 and argv[:2] == ("stop", "--time"):
        try:
            return ("stop", "--time", int(argv[2]), argv[3])
        except ValueError:
            return argv
    return argv


#: A daemon that cannot be asked, and a verifying read that cannot be made (criterion 3).
UNREACHABLE = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is it running?"
VERIFY_UNREACHABLE = "context deadline exceeded while reading t0158 container state"
#: Criterion 3's message: the daemon could not be asked (any of the natural spellings).
COULD_NOT_ASK = re.compile(r"could not be asked|could not ask|cannot be asked", re.IGNORECASE)


# --- isolation and the shared drivers -------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Run and log directories under ``tmp_path``; the Regime's constants patched small.

    ``raising=False`` for ``DOCKER_STOP_GRACE_S``: it does not exist before T-0158, and a fixture
    that raised ``AttributeError`` would turn a pending row into an error instead of the
    assertion that says what is missing.
    """
    supervisor = rig.supervisor("1")
    runs = tmp_path / "run"
    logs = tmp_path / "logs"
    for directory in (runs, logs):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(supervisor, "run_dir", lambda: runs)
    monkeypatch.setattr(supervisor, "log_dir", lambda: logs)
    for seam, value in (("run_dir", runs), ("log_dir", logs)):
        if hasattr(cli, seam):
            monkeypatch.setattr(cli, seam, lambda value=value: value)
    monkeypatch.setattr(supervisor, "STOP_WAIT_S", STOP_WAIT_S)
    monkeypatch.setattr(supervisor, "DOCKER_STOP_GRACE_S", GRACE_S, raising=False)
    monkeypatch.setattr(supervisor, "SETTLE_S", 0.2)
    return runs


@pytest.fixture
def runs(isolate):
    return isolate


def doc(container_id: str, *, running: bool = True, name: str = "astra-stt") -> str:
    return rig.container_json(
        container_id=container_id, started_at=STARTED_AT, running=running, name=name
    )


def write_record(
    runs: Path,
    *,
    pid: int,
    start_time: int | None,
    kind: str = "container",
    container_id: str | None = X,
    name: str = "stt",
) -> Path:
    """The record a verified start leaves, written through the module's own writer."""
    supervisor = rig.supervisor("1")
    record = rig.record(
        "1",
        name=name,
        kind=kind,
        pid=pid,
        start_time=start_time,
        argv=(f"/repo/services/{name}.sh",),
        container_name=f"astra-{name}" if kind == "container" else None,
        container_id=container_id,
        started_at=time.time() - 600,
    )
    path = runs / f"{name}.pid"
    rig.attr(supervisor, "write_state_record", "1")(path, record)
    return path


def run_cli(argv: list[str], **kwargs) -> int:
    try:
        return int(cli.main(argv, **kwargs))
    except SystemExit as exit_code:
        return int(exit_code.code or 0)


def down(capsys) -> tuple[int, str]:
    code = run_cli(["down"])
    printed = capsys.readouterr()
    return code, printed.out + printed.err


def stop(path: Path):
    """``supervisor.stop(path)`` directly, for the outcome flags criterion 3 names."""
    outcome = rig.attr(rig.supervisor("3"), "stop", "3")(path)
    for flag in ("stopped", "still_running"):
        assert isinstance(getattr(outcome, flag, None), bool), (
            f"StopOutcome.{flag} is a bool; stop returned {outcome!r}"
        )
    return outcome


def start_time_of(pid: int) -> int:
    """``/proc/<pid>/stat`` field 22, from the kernel — the oracle, not the module's copy."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rindex(")") + 1 :].split()[19])


def a_dead_pid() -> int:
    ghost = subprocess.Popen(["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ghost.wait()
    assert not box.alive(ghost.pid), f"pid {ghost.pid} was reused at once; re-run the row"
    return ghost.pid


def start_victim() -> subprocess.Popen:
    return subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def calls_text(docker: rig.FakeDocker) -> str:
    return "\n".join(" ".join(call) for call in docker.calls) or "(no docker calls)"


def stops_of(docker: rig.FakeDocker, ref: str) -> list[tuple[str, ...]]:
    return [call for call in docker.stop_calls() if call and call[-1] == ref]


def inspected_after_last_stop(docker: rig.FakeDocker, ref: str) -> bool:
    """Was ``inspect <ref>`` asked *after* the last ``stop`` (the verifying read)?"""
    calls = docker.calls
    stops = [index for index, call in enumerate(calls) if call and call[0] == "stop"]
    if not stops:
        return False
    return any(call[:1] == ("inspect",) and call[-1] == ref for call in calls[stops[-1] + 1 :])


def explain(report: str, docker: rig.FakeDocker) -> str:
    return f"\n--- down printed ---\n{report}\n--- docker calls ---\n{calls_text(docker)}"


# =================================================================================================
# Criterion 1 — a container record is stopped through the daemon (the observed bug)
# =================================================================================================


def test_down_stops_a_container_whose_workload_ignores_sigterm(monkeypatch, tmp_path, capsys, runs):
    """Criterion 1: the 2026-10-26 shape, done right.

    The front end ignores SIGTERM and exits only once its container has stopped — what a
    ``docker run`` client fronting a handler-less PID 1 does. ``down`` must stop the container by
    its recorded id, see the front end exit, release the record and exit 0, without SIGKILL.
    """
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        spy = rig.SignalSpy(monkeypatch)
        started = time.monotonic()
        code, report = down(capsys)
        elapsed = time.monotonic() - started

        assert [stop_shape(c) for c in docker.stop_calls()] == [("stop", "--time", 2, X)], (
            "criterion 1: exactly one `docker stop --time <DOCKER_STOP_GRACE_S> <recorded id>`, "
            f"with the grace read at call time (patched to {GRACE_S} s, so `{GRACE_ARG}`)."
            + explain(report, docker)
        )
        assert code == 0, f"criterion 1: …and `down` exits 0; it exited {code}." + explain(
            report, docker
        )
        assert not path.exists(), "criterion 1: …and the record is released" + explain(
            report, docker
        )
        assert front.wait_gone(1.0), "criterion 1: …because the front end exited"
        assert not docker.holds(X), "criterion 1: …and the daemon no longer holds the container"
        assert signal.SIGKILL not in spy.signals_to(front.pid), (
            f"criterion 4 / A16: no SIGKILL to the launcher; it sent {spy.all_signals()}"
        )
        assert X[:12] in report and str(front.pid) in report, (
            "criterion 1: the message names the container id and the launcher pid."
            + explain(report, docker)
        )
        assert elapsed < 5.0, (
            f"Regime: with the constants patched small the stop takes well under 5 s; it took "
            f"{elapsed:.1f} s"
        )
    finally:
        front.kill()


def test_a_denied_socket_retries_the_stop_under_sg(monkeypatch, tmp_path, capsys, runs):
    """Criterion 1: permission denied on the socket is retried as ``sg docker -c '<the stop>'``,
    the shape ``_docker_read`` uses (the normal state of this box's login session)."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True, deny_first=True)
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        code, report = down(capsys)
        payloads = [
            shlex.split(call[3])
            for call in docker.sg_calls()
            if len(call) >= 4 and call[1:3] == ("docker", "-c")
        ]
        assert ("stop", "--time", 2, X) in [stop_shape(p[1:]) for p in payloads if p], (
            "criterion 1: the denied stop is retried under `sg docker -c` with the same argv."
            + explain(report, docker)
        )
        assert code == 0 and not path.exists(), (
            f"criterion 1: …and that retry is the stop that worked: exit 0, record released; "
            f"exit {code}." + explain(report, docker)
        )
    finally:
        front.kill()


def test_the_stop_call_gets_the_grace_plus_the_read_timeout(monkeypatch, tmp_path, capsys, runs):
    """Criterion 1, Constants: the stop subprocess timeout is grace + ``DOCKER_TIMEOUT_S``.

    The daemon takes 2.0 s to answer the stop — longer than ``DOCKER_TIMEOUT_S`` (patched to
    1.0 s) and shorter than grace + timeout (2.0 + 1.0 s). A stop bounded by the read timeout alone
    gives up on a stop that is merely doing its job, and reports a stopped container as unasked.
    """
    supervisor = rig.supervisor("1")
    monkeypatch.setattr(supervisor, "DOCKER_TIMEOUT_S", 1.0)
    docker = rig.fake_docker(
        tmp_path, monkeypatch, before=doc(X), allow_stop=True, stop_delay_s=2.0
    )
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        code, report = down(capsys)
        assert code == 0 and not path.exists(), (
            "criterion 1: a stop that answers within grace + DOCKER_TIMEOUT_S is a stop that "
            f"worked; exit {code}." + explain(report, docker)
        )
        assert stops_of(docker, X), "criterion 1: …through `docker stop`" + explain(report, docker)
    finally:
        front.kill()


def test_a_stopped_container_not_yet_removed_counts_as_stopped(monkeypatch, tmp_path, capsys, runs):
    """Criterion 1 / Implementation notes: ``--rm`` removal lags ``stop``; ``Running == false``
    from the verifying ``inspect`` is enough to release the record."""
    docker = rig.fake_docker(
        tmp_path, monkeypatch, before=doc(X), allow_stop=True, stop_leaves_exited=True
    )
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        code, report = down(capsys)
        assert stops_of(docker, X) and inspected_after_last_stop(docker, X), (
            "criterion 1: stop, then the verifying inspect" + explain(report, docker)
        )
        assert code == 0 and not path.exists(), (
            f"criterion 1: `Running: false` is an answer — exit 0, record released; exit {code}."
            + explain(report, docker)
        )
    finally:
        front.kill()


@pytest.mark.parametrize("launcher", ["alive", "dead", "reused"])
def test_a_stop_the_daemon_accepted_but_did_not_perform_keeps_the_record(
    launcher, monkeypatch, tmp_path, capsys, runs
):
    """Criteria 1/2, adversarial: exit 0 from ``docker stop`` is not the proof — the verifying
    ``inspect`` is. Here the stop answers 0 and the container is still ``Running``.

    Over all three launcher shapes (review T1): with a live launcher a front end that outlives the
    stop would also make the row red, so only the dead and reused shapes isolate the verifying
    read — with no launcher to wait on, a skipped "still Running" check releases the record.
    ``DOCKER_TIMEOUT_S`` is patched to 1.0 s, which bounds the verifying poll."""
    monkeypatch.setattr(rig.supervisor("1"), "DOCKER_TIMEOUT_S", 1.0)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True, stop_noop=True)
    front = rig.container_front_end(tmp_path, docker) if launcher == "alive" else None
    stranger = start_victim() if launcher == "reused" else None
    try:
        if front is not None:
            path = write_record(runs, pid=front.pid, start_time=front.start_time)
        elif stranger is not None:
            path = write_record(runs, pid=stranger.pid, start_time=start_time_of(stranger.pid) + 1)
        else:
            path = write_record(runs, pid=a_dead_pid(), start_time=123456)
        before = path.read_bytes()
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)
        assert signal.SIGKILL not in [sig for _pid, sig in spy.all_signals()], (
            f"criterion 4: no SIGKILL on any path; it sent {spy.all_signals()}"
        )
        if stranger is not None:
            assert spy.signals_to(stranger.pid) == [], (
                "criterion 2: a reused pid is never signalled"
            )
        assert stops_of(docker, X) and inspected_after_last_stop(docker, X), (
            "criterion 1: the stop is followed by the verifying inspect" + explain(report, docker)
        )
        assert code == 1, (
            f"criterion 1: the daemon still says Running, so `down` failed; exit {code}."
            + explain(report, docker)
        )
        assert path.exists() and path.read_bytes() == before, (
            "criterion 1: …and the record of a container that is still running is kept, byte for "
            "byte" + explain(report, docker)
        )
        assert "still running" in report.lower(), (
            "Implementation notes: container failure messages still contain `still running`"
            + explain(report, docker)
        )
    finally:
        if front is not None:
            front.kill()
        if stranger is not None:
            box.reap(stranger)


def odd_state_doc(variant: str) -> str:
    """The recorded inspect document for ``X`` with ``State.Running`` not a JSON bool."""
    data = json.loads(doc(X))
    if variant == "no State":
        del data[0]["State"]
    elif variant == "Status only":
        data[0]["State"] = {"Status": "running"}
    else:
        data[0]["State"]["Running"] = "true"
    return json.dumps(data)


@pytest.mark.parametrize("variant", ["no State", "Status only", "Running is the string true"])
def test_an_unreadable_running_state_on_the_orphan_path_keeps_the_record(
    variant, monkeypatch, tmp_path, capsys, runs
):
    """Criterion 3 (review T2): on the orphan path (launcher dead) the daemon answers, but its
    document does not say ``Running`` is ``true`` or ``false``. That is not "not running": the
    record stays and ``down`` exits 1. Reading a missing ``Running`` as false would release the
    record of a container that may be holding the GPU."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=odd_state_doc(variant), allow_stop=True)
    path = write_record(runs, pid=a_dead_pid(), start_time=123456)
    before = path.read_bytes()
    code, report = down(capsys)
    assert code == 1, f"criterion 3: an unreadable state is not an answer; exit {code}." + explain(
        report, docker
    )
    assert path.exists() and path.read_bytes() == before, (
        "criterion 3: …and the record is kept, byte for byte" + explain(report, docker)
    )


@pytest.mark.parametrize("recorded_id", ["astra-stt", X[:12]])
def test_a_recorded_id_that_is_not_a_full_container_id_is_refused(
    recorded_id, monkeypatch, tmp_path, capsys, runs
):
    """Orchestrator ruling (review T4): the stop acts only on a recorded ``container_id`` of 64
    lowercase hex digits. A name-like id would make the daemon resolve a *name* (whatever holds it
    now) and a short prefix may match another container. Refused: no docker stop, no signal,
    record kept, ``still_running``, exit 1, and the message says the recorded id is not a
    container id. The fake would resolve both references to ``X``, so a stop here is visible."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(
            runs, pid=front.pid, start_time=front.start_time, container_id=recorded_id
        )
        before = path.read_bytes()
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)
        assert docker.stop_calls() == (), "T4: no docker stop" + explain(report, docker)
        assert spy.all_signals() == [], f"T4: no signal; it sent {spy.all_signals()}"
        assert docker.holds(X) and front.alive(), "T4: …so nothing was stopped"
        assert code == 1, f"T4: `down` exits 1; exit {code}." + explain(report, docker)
        assert path.exists() and path.read_bytes() == before, "T4: the record is kept"
        assert re.search(r"not a (full )?container id", report.lower()), (
            "T4: the message says the recorded id is not a container id" + explain(report, docker)
        )
        assert str(path) in report, "criterion 11: the message names the pid file" + explain(
            report, docker
        )
        # Orchestrator ruling (I2 conflict, 2026-10-26): the verb fence forbids spelling `docker ps`
        # in supervisor.py, so the message gives the remedy in words (check by hand, stop it
        # yourself, delete the file it names); services/README.md names `docker ps`.
        remedy = report.lower()
        assert (
            re.search(r"by hand|yourself", remedy)
            and re.search(r"\bstop\b", remedy)
            and re.search(r"delet|remov|\brm\b", remedy)
            and report.count(str(path)) >= 2
        ), (
            "criterion 11: …and a remedy: check by hand, stop it yourself, then delete the file "
            "(named in the remedy too)" + explain(report, docker)
        )
        outcome = stop(path)
        assert outcome.still_running is True and outcome.stopped is False, repr(outcome)
    finally:
        front.kill()


# =================================================================================================
# Criterion 2 — the orphan shape is stopped, not forgotten
# =================================================================================================


def test_the_orphan_shape_is_stopped_not_forgotten(monkeypatch, tmp_path, capsys, runs):
    """Criterion 2: the third ``down`` of 2026-10-26 — the launcher is dead, the container runs.

    Today this is "stale pid file (pid N is gone)", the record is removed, exit 0 — and
    ``astra-stt`` keeps the GPU with nothing left that names it.
    """
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    dead = a_dead_pid()
    path = write_record(runs, pid=dead, start_time=123456)
    spy = rig.SignalSpy(monkeypatch)
    code, report = down(capsys)

    assert stops_of(docker, X), (
        "criterion 2: the recorded container is still running, so it is stopped by its id"
        + explain(report, docker)
    )
    assert inspected_after_last_stop(docker, X), (
        "criterion 2: …and the record goes only once `inspect` says it is gone"
        + explain(report, docker)
    )
    assert not docker.holds(X), "criterion 2: the container is gone afterwards"
    assert code == 0 and not path.exists(), (
        f"criterion 2: …then the record is removed and `down` exits 0; exit {code}."
        + explain(report, docker)
    )
    assert spy.signals_to(dead) == [], (
        f"criterion 2: the pid is never signalled; it sent {spy.signals_to(dead)}"
    )


def test_a_reused_launcher_pid_is_not_signalled_but_the_recorded_container_is_handled(
    monkeypatch, tmp_path, capsys, runs
):
    """Criteria 2 and 4: two permissions. The start time decides whether the *pid* may be
    signalled; the recorded id decides whether the *container* may be stopped.

    The pid now belongs to a stranger (start time mismatch) that stays alive after the stop —
    exactly the case where criterion 4's "one SIGTERM to a front end that outlived its container"
    would fire if it were not gated on the start time.
    """
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    stranger = start_victim()
    try:
        path = write_record(runs, pid=stranger.pid, start_time=start_time_of(stranger.pid) + 1)
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)

        assert spy.signals_to(stranger.pid) == [] and box.alive(stranger.pid), (
            f"criteria 2/4: a reused pid is never signalled; it sent {spy.all_signals()}"
        )
        assert stops_of(docker, X) and not docker.holds(X), (
            "criterion 2: …while the container the record names is stopped by its id"
            + explain(report, docker)
        )
        assert code == 0 and not path.exists(), (
            f"criterion 2: …and once the daemon says it is gone the record goes, exit 0; exit "
            f"{code}." + explain(report, docker)
        )
    finally:
        box.reap(stranger)


@pytest.mark.parametrize(
    "shape", ["absent (docker 29.4.1 spelling)", "absent (legacy spelling)", "exited"]
)
def test_a_container_already_gone_is_reported_not_an_error(
    shape, monkeypatch, tmp_path, capsys, runs
):
    """Criterion 2: absent or not running — "stale … container <id> is gone", file removed, exit 0.

    The guard's other side: "no such object" is an *answer* inside the stop path (in either
    spelling the client has used), so a narrow match that misses docker 29.4.1's lower-case line
    would call a gone container unreachable and keep a dead record for ever.
    """
    document = doc(X, running=False) if shape == "exited" else None
    docker = rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=document,
        allow_stop=True,
        legacy_missing="legacy" in shape,
    )
    dead = a_dead_pid()
    path = write_record(runs, pid=dead, start_time=123456)
    spy = rig.SignalSpy(monkeypatch)
    code, report = down(capsys)
    low = report.lower()

    assert any(call[:1] == ("inspect",) and call[-1] == X for call in docker.calls), (
        "criterion 2: the daemon is asked about the recorded id" + explain(report, docker)
    )
    assert code == 0 and not path.exists(), (
        f"criterion 2: the container is gone, so the record goes and `down` exits 0; exit "
        f"{code}." + explain(report, docker)
    )
    assert "stale" in low and "gone" in low and X[:12] in report, (
        "criterion 2: the message reads `stale … container <id> is gone`" + explain(report, docker)
    )
    assert spy.signals_to(dead) == [], f"criterion 2: the pid is never signalled ({spy.calls})"


@pytest.mark.parametrize("launcher", ["dead, X gone", "alive, X running under another name"])
def test_stop_uses_the_recorded_id_never_the_name(launcher, monkeypatch, tmp_path, capsys, runs):
    """Criteria 1 and 7, adversarial: the name ``astra-stt`` now resolves to a stranger ``Y``.

    No fallback to the name (Implementation notes): a stop "by name" here stops a container this
    CLI never started. So no daemon call names ``astra-stt`` or ``Y``, and ``Y`` is still there.
    """
    stranger = doc(Y, name="astra-stt")
    if launcher.startswith("dead"):
        docker = rig.fake_docker(tmp_path, monkeypatch, allow_stop=True, others=[stranger])
        front = None
        path = write_record(runs, pid=a_dead_pid(), start_time=123456)
    else:
        docker = rig.fake_docker(
            tmp_path,
            monkeypatch,
            before=doc(X, name="astra-stt-previous"),
            allow_stop=True,
            others=[stranger],
        )
        front = rig.container_front_end(tmp_path, docker)
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
    try:
        code, report = down(capsys)
        named = [
            call
            for call in docker.calls
            if any(arg in ("astra-stt", "/astra-stt") or arg.startswith(Y[:12]) for arg in call)
            or (call[:1] == ("sg",) and ("astra-stt" in " ".join(call) or Y[:12] in " ".join(call)))
        ]
        assert named == [], (
            "criteria 1/7: the daemon is only ever asked about the recorded id; these calls "
            f"name the container name or the stranger: {named}" + explain(report, docker)
        )
        assert docker.holds(Y), "criterion 7: the stranger Y is untouched"
        assert any(
            call[-1] == X for call in docker.calls if call[:1] in (("inspect",), ("stop",))
        ), "criterion 2: …while the recorded id is what was asked about" + explain(report, docker)
        assert code == 0 and not path.exists(), (
            f"criteria 1/2: the recorded container is gone (or was stopped), exit 0; exit {code}."
            + explain(report, docker)
        )
    finally:
        if front is not None:
            front.kill()


# =================================================================================================
# Criterion 3 — "could not ask" keeps the record
# =================================================================================================


@pytest.mark.parametrize(
    ("launcher", "failing"),
    [
        ("alive", "stop"),
        ("alive", "verifying inspect"),
        ("dead", "stop"),
        ("dead", "verifying inspect"),
        ("dead", "first inspect"),
    ],
)
def test_an_unreachable_daemon_keeps_the_record_and_exits_nonzero(
    launcher, failing, monkeypatch, tmp_path, capsys, runs
):
    """Criterion 3: ``docker stop`` or the verifying ``inspect`` cannot be made — the record stays.

    "Could not ask" is not "not there" (``.claude/rules/service-lifecycle.md``): releasing the
    record here is the 2026-10-26 failure, a container holding the GPU with nothing naming it.
    """
    options: dict = {"allow_stop": True}
    if failing == "stop":
        options["stop_error"] = UNREACHABLE
        quoted = UNREACHABLE
    elif failing == "verifying inspect":
        options["verify_error"] = VERIFY_UNREACHABLE
        quoted = VERIFY_UNREACHABLE
    else:
        options["before_error"] = UNREACHABLE
        quoted = UNREACHABLE
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), **options)
    front = rig.container_front_end(tmp_path, docker) if launcher == "alive" else None
    try:
        if front is not None:
            path = write_record(runs, pid=front.pid, start_time=front.start_time)
        else:
            path = write_record(runs, pid=a_dead_pid(), start_time=123456)
        recorded = path.read_bytes()
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)

        assert code == 1, (
            f"criterion 3: the daemon could not be asked, so `down` exits 1; exit {code}."
            + explain(report, docker)
        )
        assert path.exists() and path.read_bytes() == recorded, (
            "criterion 3: …and the record is kept, byte for byte" + explain(report, docker)
        )
        assert COULD_NOT_ASK.search(report), (
            "criterion 3: the message says the daemon could not be asked" + explain(report, docker)
        )
        assert quoted[:30] in report, f"criterion 3: …and quotes why ({quoted!r})" + explain(
            report, docker
        )
        assert "still running" in report.lower(), (
            "Implementation notes: container failure messages still contain `still running`"
            + explain(report, docker)
        )
        if failing != "first inspect":
            assert stops_of(docker, X), (
                "criterion 1/2: the stop was attempted by the recorded id" + explain(report, docker)
            )
        assert signal.SIGKILL not in [sig for _pid, sig in spy.all_signals()], (
            f"criterion 4: no SIGKILL on any path; it sent {spy.all_signals()}"
        )

        outcome = stop(path)
        assert outcome.still_running is True and outcome.stopped is False, (
            f"criterion 3: the outcome is still_running=True; stop returned {outcome!r}"
        )
        assert path.exists(), "criterion 3: a second stop keeps the record too"
    finally:
        if front is not None:
            front.kill()


def test_a_socket_denied_even_under_sg_is_could_not_ask(monkeypatch, tmp_path, capsys, runs):
    """Criterion 3: permission denied, and the ``sg`` retry is denied too."""
    denied = (
        "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock"
    )
    docker = rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=doc(X),
        allow_stop=True,
        deny_first=True,
        stop_error=denied,
    )
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        code, report = down(capsys)
        assert docker.sg_calls(), "criterion 1: the denied stop was retried under sg" + explain(
            report, docker
        )
        assert code == 1 and path.exists(), (
            f"criterion 3: denied even under sg — exit 1, record kept; exit {code}."
            + explain(report, docker)
        )
        assert COULD_NOT_ASK.search(report) and "permission denied" in report.lower(), (
            "criterion 3: …saying the daemon could not be asked, and why" + explain(report, docker)
        )
    finally:
        front.kill()


def test_a_stop_that_outlasts_grace_plus_timeout_is_could_not_ask(
    monkeypatch, tmp_path, capsys, runs
):
    """Criterion 3: a stop that times out (grace 2.0 s + ``DOCKER_TIMEOUT_S`` 1.0 s, the daemon
    answering after 6 s) is "could not ask": exit 1, record kept."""
    supervisor = rig.supervisor("3")
    monkeypatch.setattr(supervisor, "DOCKER_TIMEOUT_S", 1.0)
    docker = rig.fake_docker(
        tmp_path, monkeypatch, before=doc(X), allow_stop=True, stop_delay_s=6.0
    )
    front = rig.container_front_end(tmp_path, docker)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        started = time.monotonic()
        code, report = down(capsys)
        elapsed = time.monotonic() - started
        assert code == 1 and path.exists(), (
            f"criterion 3: the stop timed out — exit 1, record kept; exit {code}."
            + explain(report, docker)
        )
        assert COULD_NOT_ASK.search(report), (
            "criterion 3: …saying the daemon could not be asked" + explain(report, docker)
        )
        assert elapsed < 5.5, (
            f"criterion 1: the stop is bounded by grace + DOCKER_TIMEOUT_S (3 s here), not by the "
            f"daemon; `down` took {elapsed:.1f} s"
        )
    finally:
        front.kill()


def test_no_such_container_then_no_such_object_is_an_answer_not_unreachable(
    monkeypatch, tmp_path, capsys, runs
):
    """Criterion 3, the guard's other side: ``stop`` answers "No such container", the verifying
    ``inspect`` answers "no such object". Both are answers — the container is gone — so this is
    exit 0, not "could not ask". (The front end here exits on its one SIGTERM, criterion 4.)"""
    docker = rig.fake_docker(tmp_path, monkeypatch, allow_stop=True)
    front = rig.container_front_end(tmp_path, docker, mode="sigterm_only")
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)
        assert stops_of(docker, X) and inspected_after_last_stop(docker, X), (
            "criteria 1/3: the stop is sent (and answers No such container), then the verifying "
            "inspect (no such object)" + explain(report, docker)
        )
        assert code == 0 and not path.exists(), (
            f'criterion 3: "No such container" is not unreachable — exit 0, record released; '
            f"exit {code}." + explain(report, docker)
        )
        assert not COULD_NOT_ASK.search(report), (
            "criterion 3: …and nothing claims the daemon could not be asked"
            + explain(report, docker)
        )
        assert spy.signals_to(front.pid) == [signal.SIGTERM], (
            f"criterion 4: the front end that outlived its container gets one SIGTERM; it got "
            f"{spy.signals_to(front.pid)}"
        )
    finally:
        front.kill()


# =================================================================================================
# Criterion 4 — no SIGKILL from the supervisor; one SIGTERM to a front end that outlived it
# =================================================================================================


@pytest.mark.parametrize("front_end", ["stubborn", "sigterm_only"])
def test_a_front_end_that_outlives_its_stopped_container_is_reported(
    front_end, monkeypatch, tmp_path, capsys, runs
):
    """Criterion 4: the container is stopped, the front end is still alive after ``STOP_WAIT_S``.

    It gets one SIGTERM, after the container stopped. ``sigterm_only`` exits on it: stopped, exit
    0. ``stubborn`` does not: ``still_running``, record kept, exit 1. Never SIGKILL.
    """
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker, mode=front_end)
    try:
        path = write_record(runs, pid=front.pid, start_time=front.start_time)
        recorded = path.read_bytes()
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)

        assert stops_of(docker, X) and not docker.holds(X), (
            "criterion 1: the container was stopped through the daemon" + explain(report, docker)
        )
        assert spy.signals_to(front.pid) == [signal.SIGTERM], (
            f"criterion 4: exactly one SIGTERM to the launcher, and no SIGKILL; it got "
            f"{spy.signals_to(front.pid)}"
        )
        assert front.sigterms() == ["stopped"], (
            "criterion 4: …sent once the container had stopped and the front end was still "
            f"alive; the front end saw {front.sigterms()} (`running` = before the stop)"
        )
        if front_end == "stubborn":
            assert front.alive(), "the double ignores SIGTERM; if it is gone, it was SIGKILLed"
            assert code == 1 and path.exists() and path.read_bytes() == recorded, (
                f"criterion 4: still alive after that wait — exit 1, record kept; exit {code}."
                + explain(report, docker)
            )
            assert "still running" in report.lower(), (
                "criterion 4: …and it says so (`still running`)" + explain(report, docker)
            )
        else:
            assert front.wait_gone(1.0), "the double exits on its SIGTERM"
            assert code == 0 and not path.exists(), (
                f"criterion 4: the one SIGTERM ended it — exit 0, record released; exit {code}."
                + explain(report, docker)
            )
    finally:
        front.kill()


# =================================================================================================
# Criterion 5 — process records and T-0033 files keep today's path byte for byte (a fence)
# =================================================================================================


@pytest.mark.parametrize("shape", ["process", "process carrying a container id", "t0033"])
def test_process_records_and_t0033_files_never_reach_docker_stop(
    shape, monkeypatch, tmp_path, capsys, runs
):
    """Criterion 5, negative (not pending: true today, must stay true).

    A daemon that would honour a stop is on PATH, holding a running ``X``. A process record — even
    one that carries a container id, because the branch is ``kind == "container"`` *and* an id —
    and a T-0033 ``{pid, argv, start_time}`` file are stopped by SIGTERM exactly as before.
    """
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    victim = start_victim()
    supervisor = rig.supervisor("5")
    try:
        argv = tuple(Path(f"/proc/{victim.pid}/cmdline").read_bytes().decode().split("\0")[:-1])
        path = runs / "vllm.pid"
        if shape == "t0033":
            supervisor.write_pid_file(path, victim.pid, argv, start_time_of(victim.pid))
        else:
            record = rig.record(
                "5",
                name="vllm",
                kind="process",
                pid=victim.pid,
                start_time=start_time_of(victim.pid),
                argv=argv,
                container_name=None,
                container_id=X if "container id" in shape else None,
            )
            supervisor.write_state_record(path, record)
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)

        assert docker.stop_calls() == (), (
            "criterion 5: a process record / T-0033 file never causes a `docker stop`"
            + explain(report, docker)
        )
        assert docker.holds(X), "criterion 5: …so the container the fake holds is untouched"
        assert spy.signals_to(victim.pid) == [signal.SIGTERM], (
            f"criterion 5: …it is SIGTERMed as today; it got {spy.signals_to(victim.pid)}"
        )
        assert code == 0 and not path.exists(), f"criterion 5: exit 0, file removed; {report}"
        assert f"vllm: stopped pid {victim.pid} ({' '.join(argv)})" in report, (
            f"criterion 5: today's message, byte for byte; it said\n{report}"
        )
    finally:
        box.reap(victim)


#: Ignores SIGTERM and says so once the handler is in place (the T-0033 ruling A16 victim).
STUBBORN = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "open(sys.argv[1], 'w').write('ready')\n"
    "time.sleep(60)\n"
)


def test_a_stubborn_process_record_keeps_todays_still_running_row(
    monkeypatch, tmp_path, capsys, runs
):
    """Criterion 5, negative (not pending): the process path's failure row is unchanged —
    ``signalled pid N; still running after <STOP_WAIT_S> s — not removed`` — and no docker stop."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    ready = tmp_path / "stubborn-ready"
    victim = subprocess.Popen(
        [sys.executable, "-c", STUBBORN, str(ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    real_kill = os.kill
    try:
        assert box.wait_until(ready.exists, timeout_s=10), "the victim never installed SIG_IGN"
        record = rig.record(
            "5",
            name="vllm",
            kind="process",
            pid=victim.pid,
            start_time=start_time_of(victim.pid),
            argv=("/repo/services/vllm.sh",),
            container_name=None,
            container_id=None,
        )
        path = runs / "vllm.pid"
        rig.supervisor("5").write_state_record(path, record)
        spy = rig.SignalSpy(monkeypatch)
        code, report = down(capsys)

        assert docker.stop_calls() == (), "criterion 5: no docker stop" + explain(report, docker)
        assert spy.signals_to(victim.pid) == [signal.SIGTERM], (
            f"criterion 5: one SIGTERM and no SIGKILL; it got {spy.signals_to(victim.pid)}"
        )
        assert code == 1 and path.exists(), f"criterion 5: exit 1, file kept; {report}"
        assert f"vllm: signalled pid {victim.pid}; still running after 0.5 s — not removed" in (
            report
        ), f"criterion 5: today's wording, byte for byte; it said\n{report}"
    finally:
        with contextlib.suppress(OSError):
            real_kill(victim.pid, signal.SIGKILL)
        box.reap(victim)


# =================================================================================================
# Criterion 6 — `status`, `up` without `--restart` and `read_container` are unchanged;
#               `up --restart stt` gets the new stop through `supervisor.stop`
# =================================================================================================


def stt_plan(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point the plan at one host-process STT launcher that records that it ran."""
    script = tmp_path / "services" / "stt.sh"
    marker = tmp_path / "markers" / "stt.ran"
    marker.parent.mkdir(parents=True, exist_ok=True)
    rig.marker_launcher(script, marker)
    rig.install_plan(monkeypatch, cli, {"stt": script})
    return script, marker


def healthy(name: str, url: str) -> str | None:
    return None


def test_restart_stt_on_a_container_record_stops_through_the_daemon(
    monkeypatch, tmp_path, capsys, runs
):
    """Criterion 6 (intended): ``up --restart stt`` reaches the new stop through ``supervisor.stop``.

    Today the restart SIGTERMs a front end that ignores it, reports ``still running`` and refuses
    to start — so a restart of STT is impossible without a hand ``docker stop``.
    """
    _script, marker = stt_plan(monkeypatch, tmp_path)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker)
    new_pid = None
    try:
        write_record(runs, pid=front.pid, start_time=front.start_time)
        spy = rig.SignalSpy(monkeypatch)
        code = run_cli(["up", "--profile", "lean", "--restart", "stt"], probe=healthy)
        printed = capsys.readouterr()
        report = printed.out + printed.err
        with contextlib.suppress(OSError, ValueError, KeyError):
            new_pid = int(json.loads((runs / "stt.pid").read_text())["pid"])

        assert [stop_shape(c) for c in docker.stop_calls()] == [("stop", "--time", 2, X)], (
            "criterion 6: the restart's stop is the container stop, by the recorded id"
            + explain(report, docker)
        )
        assert signal.SIGKILL not in spy.signals_to(front.pid), "criterion 4: no SIGKILL"
        assert front.wait_gone(1.0), "criterion 6: the old front end exited"
        assert marker.exists(), "criterion 6: …and then the launcher ran" + explain(report, docker)
        assert rig.verdict_of(report, "stt") == "started", (
            "criterion 6: …a verified start" + explain(report, docker)
        )
        assert code == 0, f"criterion 6: …and `up` exits 0; exit {code}." + explain(report, docker)
        assert new_pid is not None and new_pid != front.pid, (
            f"criterion 6: the record names the new launcher, not {front.pid}; it holds {new_pid}"
        )
    finally:
        front.kill()
        if new_pid is not None and new_pid != front.pid:
            box.reap(None, new_pid)


def test_status_never_stops(monkeypatch, tmp_path, capsys, runs):
    """Criterion 6, negative (not pending): ``status`` with a stop-capable daemon and a live
    container record stops nothing, signals nothing and writes nothing."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker)
    try:
        write_record(runs, pid=front.pid, start_time=front.start_time)
        before = rig.snapshot(runs)
        spy = rig.SignalSpy(monkeypatch)
        run_cli(["status"], probe=healthy)
        report = capsys.readouterr().out
        assert docker.stop_calls() == (), "criterion 6: `status` builds no stop" + explain(
            report, docker
        )
        assert docker.holds(X) and front.alive(), "criterion 6: …the container and launcher run on"
        assert spy.all_signals() == [], f"criterion 6: no signal; it sent {spy.all_signals()}"
        assert rig.snapshot(runs) == before, "criterion 6: …and the run directory is unchanged"
    finally:
        front.kill()


def test_up_without_restart_never_stops_a_recorded_container(monkeypatch, tmp_path, capsys, runs):
    """Criterion 6, negative (not pending): ``up`` without ``--restart`` over a live container
    record stops nothing and signals nothing, whatever it decides."""
    script, marker = stt_plan(monkeypatch, tmp_path)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=doc(X), allow_stop=True)
    front = rig.container_front_end(tmp_path, docker)
    try:
        supervisor = rig.supervisor("6")
        record = rig.record(
            "6",
            name="stt",
            kind="container",
            pid=front.pid,
            start_time=front.start_time,
            argv=(str(script),),
            script_digest=rig.sha256_of(script),
            container_name="astra-stt",
            container_id=X,
        )
        supervisor.write_state_record(runs / "stt.pid", record)
        spy = rig.SignalSpy(monkeypatch)
        run_cli(["up", "--profile", "lean"], probe=healthy)
        report = capsys.readouterr().out
        assert docker.stop_calls() == (), "criterion 6: `up` builds no stop" + explain(
            report, docker
        )
        assert spy.all_signals() == [], f"criterion 6: no signal; it sent {spy.all_signals()}"
        assert docker.holds(X) and front.alive() and not marker.exists(), (
            "criterion 6: …the recorded container and its launcher are untouched"
        )
    finally:
        front.kill()


@pytest.mark.parametrize("legacy", [False, True])
def test_read_container_still_calls_no_such_object_unknown(legacy, monkeypatch, tmp_path):
    """Criterion 6, negative (not pending): outside the stop path "no such object" stays
    *unknown* — ``read_container`` and ``container_identity`` keep their tri-state."""
    supervisor = rig.supervisor("6")
    rig.fake_docker(tmp_path, monkeypatch, allow_stop=True, legacy_missing=legacy)
    identity, why = supervisor.read_container(X)
    assert identity is None and why, (
        f"criterion 6: read_container(absent) is (None, why); it returned {(identity, why)!r}"
    )
    found = supervisor.container_identity(X)
    assert found.alive is False and found.unknown is True, (
        f"criterion 6: container_identity(absent) is alive=False, unknown=True; got {found!r}"
    )


def test_read_container_still_reads_a_running_and_an_exited_container(monkeypatch, tmp_path):
    """Criterion 6, negative (not pending): the two answered states are unchanged."""
    supervisor = rig.supervisor("6")
    rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=doc(X),
        allow_stop=True,
        others=[doc(Y, running=False, name="other")],
    )
    running = supervisor.container_identity(X)
    exited = supervisor.container_identity(Y)
    assert running.alive is True and running.unknown is False, f"running: {running!r}"
    assert exited.alive is False and exited.unknown is False, f"exited: {exited!r}"


# =================================================================================================
# Criterion 9 — pinned versions untouched
# =================================================================================================

#: ``merge-base M^1 M^2`` of the T-0158 merge 39c7cfb (literal, per ``.claude/rules/tests.md``).
BASE = "623ecb31edf148bd116a07c7e1343f4ec4a1fa2b"
PINNED = (
    "services/versions.lock",
    "services/stt.sh",
    "services/vllm.sh",
    "services/moshi/Dockerfile",
)


#: ``M^2`` of the T-0158 merge 39c7cfb: the closed range ``BASE..TIP`` is what this row judges,
#: never the working tree or a moving ref.
TIP = "fe99370b76734e27eac63e97a8b8ec2e7de9c79c"


def test_pinned_service_files_are_byte_identical_to_the_fork_point():
    """Criterion 9 (a fence): the versions lock, both launch scripts and the moshi Dockerfile are
    byte-identical over the closed range ``BASE..TIP``. ``--init`` is T-0158b, not this task.
    Skips, naming the SHA, when either commit is absent from the clone."""
    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE[:9]} is not an ancestor of {TIP[:9]}"
    changed = gr.changed(BASE, TIP, *PINNED)
    assert changed == [], (
        f"criterion 9: these pinned files changed since {BASE[:9]}: {changed} — a version or "
        "launcher change is its own task (T-0158b for --init)"
    )


# =================================================================================================
# Real-daemon parity for the fake's answers (marked gpu: needs the box's docker daemon; never in
# the fast suite; the verifier's V1 box check is the fuller form)
# =================================================================================================


def _real_docker(*args: str) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30)
    if done.returncode != 0 and "permission denied" in done.stderr.lower():
        done = subprocess.run(
            ["sg", "docker", "-c", shlex.join(["docker", *args])],
            capture_output=True,
            text=True,
            timeout=30,
        )
    return done


@pytest.mark.gpu
def test_the_real_daemon_answers_as_the_fake_does():
    """The fake's stop/inspect answers are the real client's (docker 29.4.1 on this box).

    Only references that cannot exist are used (a fresh uuid), so nothing is stopped.
    """
    if shutil.which("docker") is None:
        pytest.skip("no docker client on this machine")
    ref = f"t0158-absent-{uuid.uuid4().hex[:12]}"
    inspected = _real_docker("inspect", ref)
    assert inspected.returncode != 0
    assert f"no such object: {ref}" in inspected.stderr.lower()
    stopped = _real_docker("stop", "--time", "1", ref)
    assert stopped.returncode != 0
    assert f"No such container: {ref}" in stopped.stderr
    fractional = _real_docker("stop", "--time", "1.5", ref)
    assert fractional.returncode != 0 and "invalid argument" in fractional.stderr
