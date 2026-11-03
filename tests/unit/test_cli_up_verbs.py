"""``astra up`` says, per service, what it actually did — and ``--restart`` is the only thing that
signals anything (T-0049 group B, criteria 10, 11 and 13, plus rulings A16 and A17).

Group A built the decision (``service_state.decide``) and the verified start
(``supervisor.start_service``); nothing changed for a human. This module is the command layer:
the verb that ran on 2026-09-16 printed ``started`` for three services while two starts had
failed, and the verb that replaces it must print one row per service drawn from the decision,
refuse to signal anything without ``--restart``, and turn those rows into an exit code.

Why the rows below drive the **real** start path (no injected runner): the T-0033 rows that inject
a ``Runner`` cannot see verification at all, and T-0033 records that an injected runner returning
instantly is exactly why the fast suite never saw the blocking-``subprocess.run`` bug. So every row
here points ``cli.start_plan`` at launchers it owns (the route T-0033's own liveness driver takes),
redirects the run and log directories, and injects only the probe. Nothing containerised is ever
started: each launcher ``exec``s a sleeper, and the one row that needs a daemon answer talks to the
fake ``docker`` on ``PATH`` (ruling A9: a unit row that reaches the live daemon is I/O the test
rules forbid, and inside a measured row it is an unattributable number).

The seam these rows assume is written out in ``tests/unit/_service_rig.py``'s group B section.

**Ruling A17, reconciled, and this module's answer** (stated because the task asks for an answer,
not a guess): the windows are **not** overlapped and the Regime line's bound is amended instead.
Overlapping the settle windows means spawning the speech services before the engine is healthy,
and ``docs/02-architecture.md`` §6 makes startup order a health-check ordering, not a timer — so
one settle window per *started* service, serially, is the design, and the honest bound is
``< 1.0 s + (services actually started) × SETTLE_S``. The other half of the reconciliation is paid
too: :func:`test_a_three_service_start_stays_inside_one_settle_window_per_started_service` is the
first row to exercise the **default** ``SETTLE_S`` on a *successful* start, and
:func:`test_a_profile_that_is_entirely_skipped_costs_no_settle_window` pins the half a human
feels — the common case, where everything is already up, still costs no window at all.

Regime for every row: fast suite, no GPU, no network, no real service, profile ``lean``,
``astra.toml`` defaults, probe injected, launchers are ``sleep``s reaped by the row that started
them. Two rows measure a wall bound with ``time.monotonic``, n = 1 and n = 3, no seed — they are
bounds three orders of magnitude away from the failure they exclude, never performance claims, and
not comparable across boxes.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_astra import cli
from local_astra.config import load
from tests.unit import _box_rig as box
from tests.unit import _service_rig as rig

#: Every row that needs group B's implementation carries this; the three regression-fence rows
#: (criterion 13, and the loopback negative) deliberately do **not** — they pass today and must go
#: on passing, which is the whole of what a fence is.

REPO = Path(__file__).resolve().parents[2]
SERVICES = REPO / "services"
LEAN = load(None).profiles["lean"]
NAMES = ("vllm", "stt", "tts")

#: Short enough that a row which starts three services costs ~0.6 s rather than ~3 s. The two rows
#: that are *about* the window opt out (``default_settle``) and use the shipped 1.0 s.
FAST_SETTLE_S = 0.2

#: What ``--restart`` must accept beside a comma-separated list of names (criterion 11).
RESTART_ALL = "all"


# =================================================================================================
# Isolation (ruling A8) and the shared drivers
# =================================================================================================


@pytest.fixture
def default_settle():
    """Opt out of :func:`fast_settle` — requested by the two rows that measure the window.

    A marker would have to be registered in ``pyproject.toml`` (``--strict-markers``), which is not
    this file's to edit; requesting a fixture is the local equivalent (the trick
    ``tests/unit/test_cli_up.py`` uses for the real log directory).
    """
    return True


@pytest.fixture(autouse=True)
def fast_settle(request, monkeypatch):
    """Shrink ``supervisor.SETTLE_S`` for every row that is not about the window itself.

    ``start_service`` reads the constant at call time, which is the same seam the group A rows use
    through ``settle_s=``; the CLI takes no such argument, and that is deliberate — criterion 10 is
    about what the verb reports, not about a knob.
    """
    if "default_settle" in request.fixturenames:
        return
    supervisor = rig.supervisor("10")
    assert isinstance(rig.attr(supervisor, "SETTLE_S", "10"), float), (
        "Regime: supervisor.SETTLE_S is the settle window in seconds"
    )
    monkeypatch.setattr(supervisor, "SETTLE_S", FAST_SETTLE_S)


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch, tmp_path):
    """No row here may read or write the checkout's ``logs/`` or its real run directory.

    Ruling A8: without this a row that forgot would overwrite the state of the services running on
    this box — the damage the 2026-09-16 run actually did — and it would be invisible in a green
    suite. So the redirection is asserted, not assumed.
    """
    supervisor = rig.supervisor("10")
    logs = tmp_path / "logs"
    runs = tmp_path / "run"
    for directory in (logs, runs):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(supervisor, "log_dir", lambda: logs)
    monkeypatch.setattr(supervisor, "run_dir", lambda: runs)
    for seam, value in (("log_dir", logs), ("run_dir", runs)):
        if hasattr(cli, seam):
            monkeypatch.setattr(cli, seam, lambda value=value: value)
    assert supervisor.run_dir() == runs and ".git" not in os.fspath(supervisor.run_dir()), (
        "the redirection has to bite: every row here writes state files, and the real run "
        "directory holds the identity of the services running on this box"
    )
    return logs, runs


@pytest.fixture
def runs(isolate_state):
    return isolate_state[1]


class Probe:
    """``probe(name, url) -> None | reason``, the T-0033 seam, with a per-service script.

    ``starting`` names services that refuse the **first** time they are asked and answer
    afterwards: a service that is not up when ``up`` decides, and is up once it has been started.
    ``never`` names services that refuse for ever. Everything else is healthy from the first ask,
    which is the state the box was in on 2026-09-16 and therefore the case that matters most.
    """

    def __init__(self, *, starting=(), never=()) -> None:
        self.starting = set(starting)
        self.never = set(never)
        self.asked: list[tuple[str, str]] = []

    def __call__(self, name: str, url: str) -> str | None:
        self.asked.append((name, url))
        if name in self.never:
            return f"{name}: [Errno 111] Connection refused"
        if name in self.starting and self.count(name) == 1:
            return f"{name}: [Errno 111] Connection refused (not up yet)"
        return None

    def count(self, name: str) -> int:
        return sum(1 for asked, _url in self.asked if asked == name)


def run_cli(argv: list[str], **kwargs) -> int:
    try:
        return int(cli.main(argv, **kwargs))
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    except TypeError as exc:
        raise AssertionError(
            f"T-0049 criterion 10: cli.main(argv, *, runner=None, probe=None) — {exc}"
        ) from None


def report_of(capsys) -> str:
    printed = capsys.readouterr()
    return printed.out + printed.err


def start_time_of(pid: int) -> int:
    """``/proc/<pid>/stat`` field 22, from the kernel — the oracle, not the module's own copy."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rindex(")") + 1 :].split()[19])


def two_hours_ago() -> str:
    """The daemon's stamp for the instance that was already there (relative, per ruling A10)."""
    moment = datetime.now(UTC) - timedelta(hours=2)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond * 1000:09d}Z"


@dataclass
class Service:
    """One entry of the start plan, plus whatever a row arranged to be running under it."""

    name: str
    script: Path
    marker: Path
    state_file: Path
    victim: subprocess.Popen | None = None
    recorded_bytes: bytes | None = None

    def started(self) -> bool:
        """Did this invocation actually run the launcher? (the script writes the marker)"""
        return self.marker.exists()

    def state_bytes(self) -> bytes | None:
        return self.state_file.read_bytes() if self.state_file.exists() else None

    def record(self) -> dict:
        found = sorted(path.name for path in self.state_file.parent.glob("*"))
        assert self.state_file.exists(), (
            f"criterion 10: {self.name} was started, so there is a state record at "
            f"{self.state_file}; the run directory holds {found}"
        )
        return json.loads(self.state_file.read_text())


def make_service(name: str, tmp_path: Path, runs: Path) -> Service:
    """A plan entry whose launcher records that it ran and then ``exec``s a sleeper."""
    script = tmp_path / "services" / f"{name}.sh"
    marker = tmp_path / "markers" / f"{name}.ran"
    marker.parent.mkdir(parents=True, exist_ok=True)
    rig.marker_launcher(script, marker)
    return Service(name=name, script=script, marker=marker, state_file=runs / f"{name}.pid")


def make_running(name: str, tmp_path: Path, runs: Path, *, age_s: float = 300.0) -> Service:
    """A service that is **already running**, recorded by this CLI, with a matching digest.

    The victim is a ``sleep`` this row starts and reaps; the record names it the way a verified
    start would (``kind=process``, the launcher pid, its ``/proc`` start time, the script's
    digest), so ``decide`` sees provenance that matches and the outcome has to be a skip.
    """
    service = make_service(name, tmp_path, runs)
    victim = subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    service.victim = victim
    write_record(
        service,
        pid=victim.pid,
        start_time=start_time_of(victim.pid),
        started_at=time.time() - age_s,
    )
    return service


def write_record(service: Service, **overrides) -> None:
    """Write the versioned state record group A defined, for this service's script."""
    supervisor = rig.supervisor("10")
    fields = {
        "name": service.name,
        "kind": "process",
        "argv": (str(service.script),),
        "script_digest": rig.sha256_of(service.script),
        "container_name": None,
        "container_id": None,
    }
    fields.update(overrides)
    record = rig.record("10", **fields)
    rig.attr(supervisor, "write_state_record", "10")(service.state_file, record)
    service.recorded_bytes = service.state_file.read_bytes()


def plan(monkeypatch, *services: Service) -> list[str]:
    return rig.install_plan(monkeypatch, cli, {s.name: s.script for s in services})


def reap(*services: Service) -> None:
    """Every process this row is responsible for: the victims, and whatever the verb started."""
    for service in services:
        if service.victim is not None:
            box.reap(service.victim)
        record = None
        with contextlib.suppress(OSError, ValueError):
            record = json.loads(service.state_file.read_text())
        if isinstance(record, dict) and isinstance(record.get("pid"), int):
            box.reap(None, int(record["pid"]))


def verdicts(report: str) -> dict[str, str | None]:
    return rig.verdicts(report, NAMES)


def explain(report: str) -> str:
    return "\n--- the verb printed ---\n" + report + "\n------------------------"


# =================================================================================================
# Criterion 10 — one row per service, drawn from the decision, and the exit-code rule
# =================================================================================================


def test_up_against_a_running_profile_prints_already_running_and_exits_zero(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 10: the 2026-09-16 box, done right — three skips, with identity and age, exit 0.

    This is the state the human's box was in: all three services healthy. The verb that ran then
    printed ``started`` three times. The verb must instead say ``already running`` for each, name
    the handle it checked and how old it is, start nothing, and exit 0 — a skip is not a failure.
    """
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        code = run_cli(["up", "--profile", "lean"], probe=Probe())
        report = report_of(capsys)

        assert code == 0, (
            f"criterion 10: every service was already running, so every row is a skip and the "
            f"verb exits 0; it exited {code}.{explain(report)}"
        )
        assert verdicts(report) == dict.fromkeys(NAMES, "already running"), (
            f"criterion 10: one row per service, saying `already running`; the rows read "
            f"{verdicts(report)}.{explain(report)}"
        )
        for service in services:
            lines = rig.verdict_lines(report, service.name)
            text = "\n".join(lines)
            assert str(service.victim.pid) in text, (
                f"criterion 10: …carrying the identity it checked (pid {service.victim.pid}); "
                f"{service.name}'s rows read {lines}.{explain(report)}"
            )
            assert any(token in text for token in ("since", "ago", "age")), (
                f"criterion 10: …and how old that instance is, which is all a skip may claim "
                f"(D4's honesty limit); {service.name}'s rows read {lines}.{explain(report)}"
            )
            assert not service.started(), (
                f"criterion 10: a skip starts nothing — {service.name}'s launcher ran anyway "
                f"({service.marker})"
            )
        assert all(box.alive(s.victim.pid) for s in services), (
            "criterion 10: …and the instances that were already running are still running"
        )
    finally:
        reap(*services)


def test_up_exits_non_zero_when_any_row_is_conflict_and_starts_nothing_over_it(
    tmp_path, monkeypatch, capsys, runs
):
    """Criteria 8 and 10: healthy with no record is the 2026-09-16 shape — a refusal, exit non-zero.

    Two services are this CLI's own and are skipped; the third answers its port with nothing to say
    who started it. That case must never be a PASS and must never be started over: something is on
    the port, ``astra down`` cannot stop it, and a second instance would fail with the docker
    Conflict this task is named after.
    """
    vllm, tts = (make_running(name, tmp_path, runs) for name in ("vllm", "tts"))
    stt = make_service("stt", tmp_path, runs)  # healthy, but no record names it
    try:
        plan(monkeypatch, vllm, stt, tts)
        code = run_cli(["up", "--profile", "lean"], probe=Probe())
        report = report_of(capsys)

        assert code != 0, (
            f"criterion 10: a CONFLICT row makes the verb exit non-zero; it exited "
            f"{code}.{explain(report)}"
        )
        assert verdicts(report) == {
            "vllm": "already running",
            "stt": "CONFLICT",
            "tts": "already running",
        }, f"criterion 10: the rows read {verdicts(report)}.{explain(report)}"
        assert not stt.started(), (
            "criterion 8: nothing is started over an instance whose provenance is unknown — the "
            f"stt launcher ran anyway ({stt.marker}).{explain(report)}"
        )
        assert not stt.state_file.exists(), (
            "criterion 8: …and no record is invented for it; a record naming an instance this CLI "
            f"did not start is the provenance corruption of 2026-09-16. It wrote "
            f"{stt.state_file.read_text()!r}"
        )
        for service in (vllm, tts):
            assert service.state_bytes() == service.recorded_bytes, (
                f"criterion 10, negative: {service.name} was skipped, so its record is untouched"
            )
    finally:
        reap(vllm, stt, tts)


def test_up_exits_non_zero_when_a_start_fails_and_the_row_quotes_the_launchers_log(
    tmp_path, monkeypatch, capsys, runs
):
    """Criteria 1 and 10 at the verb: the reproduction, as the human sees it.

    The STT launcher exits 125 having written the recorded docker Conflict line to its log, while
    the other two are already running. The row must be FAILED, carry the exit status and quote the
    line that explains it (ruling A4: the last *non-empty* line of that log is the help trailer, so
    the tail has to be more than one line), and the verb must exit non-zero.
    """
    vllm, tts = (make_running(name, tmp_path, runs) for name in ("vllm", "tts"))
    stt = make_service("stt", tmp_path, runs)
    rig.failing_launcher(stt.script, code=125, log=rig.CONFLICT_LOG, container=None)
    try:
        plan(monkeypatch, vllm, stt, tts)
        code = run_cli(["up", "--profile", "lean", "--timeout", "0"], probe=Probe(never=["stt"]))
        report = report_of(capsys)

        assert code != 0, (
            f"criteria 1/10: a FAILED start makes the verb exit non-zero; it exited "
            f"{code}.{explain(report)}"
        )
        assert verdicts(report)["stt"] == "FAILED", (
            f"criterion 10: the launcher exited, so stt's row is FAILED — it is not `started` and "
            f"not a skip; the rows read {verdicts(report)}.{explain(report)}"
        )
        assert "125" in report, (
            f"criterion 1: …naming the launcher's own exit status (125).{explain(report)}"
        )
        assert rig.CONFLICT_QUOTE in report, (
            f"criterion 1 / ruling A4: …and quoting the line that explains it, "
            f"{rig.CONFLICT_QUOTE!r}, which is two lines above the end of the recorded log."
            f"{explain(report)}"
        )
        for service in (vllm, tts):
            assert service.state_bytes() == service.recorded_bytes, (
                f"criterion 5 at the verb: one service failing leaves {service.name}'s record "
                "byte-identical"
            )
    finally:
        reap(vllm, stt, tts)


#: A launcher log that names no other service and carries none of the five verdict words: the row
#: below reads the *other* services' verdicts out of the same report, and the FAILED row quotes its
#: launcher's log underneath itself (ruling A4). ``rig.CONFLICT_LOG`` names ``astra-stt`` verbatim,
#: which is correct where stt is the service that failed and would be a trap here.
ENGINE_CRASH_LOG = (
    "INFO engine: loading the model\n"
    "ERROR engine: CUDA error 803: system has unsupported display driver\n"
    "engine: aborting\n"
)


def test_a_failing_engine_does_not_hide_the_rows_of_the_services_behind_it(
    tmp_path, monkeypatch, capsys, runs
):
    """Ruling B11: a failed start is recorded and the plan carries on to the next service.

    Criterion 10 is *one row per service*, and the engine is first in the start plan. A verb that
    returned at the first failure would answer "the engine did not start" and say nothing about
    the two services behind it — so a human would fix the engine, run again, and only then
    discover the next problem. Worse, it hides the distinction that matters here: whether anything
    else *would* have started.

    So vllm's launcher dies inside its settle window, and the two entries behind it must still be
    observed, decided, acted on and reported, each in its own right: stt is down and is really
    started (verified, with a record), tts is already running and is skipped. The exit code is
    non-zero because a row failed, which is true either way and is why this row asserts the rows
    rather than the code.

    The mutation this row exists to catch is the *shape* of the decision (`.claude/rules/tests.md`):
    abort the loop on the two failure verdicts a start can return, FAILED and NOT-NEW. A blunter
    one — fail the whole command on any bad verdict — reddens a row for an unrelated reason and
    teaches nothing.
    """
    vllm = make_service("vllm", tmp_path, runs)
    crash_log = tmp_path / "engine-crash.log"
    crash_log.write_text(ENGINE_CRASH_LOG)
    rig.failing_launcher(vllm.script, code=1, log=crash_log, container=None)
    stt = make_service("stt", tmp_path, runs)
    tts = make_running("tts", tmp_path, runs)
    try:
        assert plan(monkeypatch, vllm, stt, tts) == ["vllm", "stt", "tts"], (
            "the premise: the service that fails is **first**, so a verb that aborts on a failed "
            "start produces no row for either of the others"
        )
        code = run_cli(
            ["up", "--profile", "lean", "--timeout", "0"],
            probe=Probe(never=["vllm"], starting=["stt"]),
        )
        report = report_of(capsys)

        assert verdicts(report) == {
            "vllm": "FAILED",
            "stt": "started",
            "tts": "already running",
        }, (
            "ruling B11: the first service failing does not end the run — every later service "
            "still gets its own row, drawn from its own decision (criterion 10). The rows read "
            f"{verdicts(report)}.{explain(report)}"
        )
        assert stt.started(), (
            "ruling B11: …and the service behind the failure was really started, not merely "
            f"mentioned ({stt.marker} was never written).{explain(report)}"
        )
        assert box.alive(int(stt.record()["pid"])), (
            f"…with its launcher alive and recorded; stt's record holds {stt.record()}"
        )
        assert not tts.started() and tts.state_bytes() == tts.recorded_bytes, (
            "…and the skip behind the failure is still a skip: nothing started, record untouched"
        )
        assert code != 0, (
            f"…and the run is still a failure, because one row is FAILED; it exited {code}."
            f"{explain(report)}"
        )
    finally:
        reap(vllm, stt, tts)


def test_a_start_that_cannot_be_shown_to_be_new_is_not_reported_as_started(
    tmp_path, monkeypatch, capsys, runs
):
    """Criteria 3 and 10, adversarial: NOT-NEW is a failure at the verb too.

    The service is down when the verb decides, so it is started; the launcher stays up and the port
    answers — which is all the 2026-09-16 verb ever asked. But the daemon says the container
    answering is the one that has been up two hours, so this invocation produced nothing. The row
    must not say ``started``, nothing may be recorded, and the verb must exit non-zero.
    """
    vllm, tts = (make_running(name, tmp_path, runs) for name in ("vllm", "tts"))
    stt = make_service("stt", tmp_path, runs)
    stale = rig.container_json(
        container_id="61f0d405d9898f7e8098b9fd3f3b6837a4afa874711d7247f6e2a3ca5cc8df0c",
        started_at=two_hours_ago(),
    )
    docker = rig.fake_docker(tmp_path, monkeypatch, before=stale, after=stale)
    rig.sleeper_launcher(stt.script, container="astra-stt", switch=docker.switch)
    try:
        plan(monkeypatch, vllm, stt, tts)
        code = run_cli(["up", "--profile", "lean"], probe=Probe(starting=["stt"]))
        report = report_of(capsys)

        assert verdicts(report)["stt"] == "NOT-NEW", (
            "criterion 3: the container answering started two hours before the attempt, so this "
            f"start produced nothing; the rows read {verdicts(report)}.{explain(report)}"
        )
        assert code != 0, (
            f"criterion 10: NOT-NEW is one of the three verdicts that make the verb exit non-zero; "
            f"it exited {code}.{explain(report)}"
        )
        assert not stt.state_file.exists(), (
            "criterion 5: …and nothing is recorded for a start that cannot be shown to have "
            f"happened; it wrote {stt.state_file.read_text()!r}"
        )
    finally:
        reap(vllm, stt, tts)


def test_three_verified_starts_are_reported_started_and_recorded(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 10, the positive: ``started`` is what a verified start is called, and only that.

    Nothing is up, so all three are started; each launcher is alive at the end of its window and
    each service answers, so each row is ``started`` and carries the pid the CLI can stop later.
    The exit code is 0 because no row is CONFLICT, FAILED or NOT-NEW.
    """
    services = [make_service(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        code = run_cli(["up", "--profile", "lean"], probe=Probe(starting=NAMES))
        report = report_of(capsys)

        assert verdicts(report) == dict.fromkeys(NAMES, "started"), (
            f"criterion 10: three verified starts are three `started` rows; the rows read "
            f"{verdicts(report)}.{explain(report)}"
        )
        assert code == 0, f"…and the verb exits 0; it exited {code}.{explain(report)}"
        for service in services:
            assert service.started(), f"{service.name}'s launcher really ran"
            record = service.record()
            assert record.get("version"), (
                "criterion 5 / ruling A16: the file the verb leaves is the **versioned** record, "
                f"not the three-key T-0033 file the eager path writes; {service.name}'s file "
                f"holds {record}"
            )
            assert box.alive(int(record["pid"])), (
                f"criterion 10: …naming the launcher this invocation started and left running; "
                f"{service.name}'s record holds {record}"
            )
            assert str(record["pid"]) in "\n".join(rig.verdict_lines(report, service.name)), (
                f"criterion 10: …and the row carries that identity.{explain(report)}"
            )
    finally:
        reap(*services)


# =================================================================================================
# Criterion 11 — ``--restart`` is the deliberate reboot, and the only thing that signals anything
# =================================================================================================


def test_up_without_restart_signals_nothing_and_removes_no_state_file(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 11, **the negative that matters more than the positive** (adversarial).

    An ``astra up`` that reboots a live engine mid-session is a worse bug than the one being fixed.
    So: with three healthy, correctly recorded services, nothing may be signalled, no state file
    may be removed or rewritten, and no launcher may run. ``SignalSpy`` delegates to the real
    ``os.kill``, so ``stop``'s liveness checks would still behave normally, and ``os.kill(pid, 0)``
    is excluded — asking whether a pid is alive is not signalling it.

    The mutation this row exists to catch: turn the SKIP branch into stop-then-start. The victims
    then take a SIGTERM, their records are removed and rewritten, and the launchers run — three
    independent assertions here go red at once.
    """
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        spy = rig.SignalSpy(monkeypatch)
        before = rig.snapshot(runs)
        code = run_cli(["up", "--profile", "lean"], probe=Probe())
        report = report_of(capsys)

        assert spy.all_signals() == [], (
            "criterion 11, negative: without `--restart` the verb sends **no** signal to anything; "
            f"it sent {spy.all_signals()}.{explain(report)}"
        )
        for service in services:
            assert box.alive(service.victim.pid), (
                f"…so {service.name}'s instance (pid {service.victim.pid}) is still running"
            )
            assert not service.started(), f"…and its launcher never ran ({service.marker} exists)"
        assert rig.snapshot(runs) == before, (
            "criterion 11, negative: …and no state file is removed, replaced or rewritten. "
            f"Before: {sorted(before)}; after: {sorted(rig.snapshot(runs))}"
        )
        assert code == 0, f"…and an all-skip run exits 0; it exited {code}.{explain(report)}"
    finally:
        reap(*services)


def test_restart_stops_the_named_service_then_starts_and_verifies_it(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 11, the positive: ``--restart stt`` reboots exactly one service.

    T-0048 group B is why the flag names services rather than restarting everything: it must reboot
    the engine at a different KV pool while STT and TTS stay up. The stop goes through
    ``supervisor.stop``'s identity rules (SIGTERM, never SIGKILL — the child fronts a container),
    and what follows is an ordinary verified start: a new launcher, a new record.
    """
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    vllm, stt, tts = services
    try:
        plan(monkeypatch, *services)
        spy = rig.SignalSpy(monkeypatch)
        code = run_cli(["up", "--profile", "lean", "--restart", "stt"], probe=Probe())
        report = report_of(capsys)

        assert signal.SIGTERM in spy.signals_to(stt.victim.pid), (
            f"criterion 11: `--restart stt` stops the recorded instance through supervisor.stop; "
            f"the signals sent were {spy.all_signals()}.{explain(report)}"
        )
        assert signal.SIGKILL not in spy.signals_to(stt.victim.pid), (
            "criterion 11 / T-0033 ruling A16: …with no SIGKILL follow-up (the child fronts a "
            f"container); the signals sent were {spy.all_signals()}"
        )
        assert stt.started(), (
            f"criterion 11: …and then starts it again ({stt.marker} was never written)"
            f"{explain(report)}"
        )
        assert verdicts(report)["stt"] == "started", (
            f"criterion 11: …reported as a verified start; the rows read "
            f"{verdicts(report)}.{explain(report)}"
        )
        restarted = stt.record()
        assert int(restarted["pid"]) != stt.victim.pid and box.alive(int(restarted["pid"])), (
            f"criterion 11: …and the record now names the new launcher, not the stopped one "
            f"({stt.victim.pid}); it holds {restarted}"
        )
        assert code == 0, f"…and the verb exits 0; it exited {code}.{explain(report)}"

        for other in (vllm, tts):
            assert spy.signals_to(other.victim.pid) == [], (
                f"criterion 11, negative: only the named service is signalled — {other.name} "
                f"(pid {other.victim.pid}) received {spy.signals_to(other.victim.pid)}"
            )
            assert box.alive(other.victim.pid) and not other.started(), (
                f"…{other.name} is untouched: still running, launcher never re-run"
            )
            assert other.state_bytes() == other.recorded_bytes, (
                f"…and {other.name}'s record is byte-identical"
            )
            assert rig.verdict_of(report, other.name) == "already running", (
                f"…and it is reported as the skip it is.{explain(report)}"
            )
    finally:
        reap(*services)


def test_restart_all_restarts_every_service_in_the_plan(tmp_path, monkeypatch, capsys, runs):
    """Criterion 11: ``--restart all`` is the whole plan, and it is still a verified start each."""
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        spy = rig.SignalSpy(monkeypatch)
        code = run_cli(["up", "--profile", "lean", "--restart", RESTART_ALL], probe=Probe())
        report = report_of(capsys)
        for service in services:
            assert signal.SIGTERM in spy.signals_to(service.victim.pid), (
                f"criterion 11: `--restart all` stops every service; {service.name} (pid "
                f"{service.victim.pid}) got {spy.signals_to(service.victim.pid)}.{explain(report)}"
            )
            assert service.started(), f"…and starts it again ({service.marker} missing)"
        assert verdicts(report) == dict.fromkeys(NAMES, "started"), (
            f"…and every row is a verified start; they read {verdicts(report)}.{explain(report)}"
        )
        assert code == 0, f"…exit 0; it exited {code}.{explain(report)}"
        printed = sorted(NAMES, key=lambda name: report.index(rig.verdict_lines(report, name)[0]))
        assert printed == list(NAMES), (
            f"criterion 11 / T-0028 criterion 17: the services are restarted in the plan's order "
            f"({list(NAMES)}); the rows came out {printed}.{explain(report)}"
        )
    finally:
        reap(*services)


@pytest.mark.parametrize("spec", ["bogus", "stt,bogus"])
def test_restart_with_a_name_the_profile_does_not_start_exits_two_having_done_nothing(
    spec, tmp_path, monkeypatch, capsys, runs
):
    """Ruling B12: an unknown ``--restart`` name is a usage error, and nothing happens first.

    ``--restart`` is the one flag that can stop a running engine (criterion 11), so a typo in it
    is the most expensive typo this CLI has. Silently restarting nothing would be the 2026-09-16
    failure in miniature: a command that reports success for work it did not do. Refusing is half
    the ruling; **doing nothing before refusing** is the half that protects the box, which is why
    the second case names a real service beside the unknown one — a check that only fires when
    *every* name is unknown would stop and reboot stt here.

    Three things are pinned, because the ruling is all three: the exit code is 2 (a usage error,
    not a service failure), no signal was delivered to anything, and the run directory is
    byte-identical — no record removed, replaced or rewritten. The launchers are the fourth
    witness: not one of them ran.
    """
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        spy = rig.SignalSpy(monkeypatch)
        before = rig.snapshot(runs)
        code = run_cli(["up", "--profile", "lean", "--restart", spec], probe=Probe())
        report = report_of(capsys)

        assert code == 2, (
            f"ruling B12: `--restart {spec}` names a service this profile does not start, which "
            f"is a usage error — exit 2, not a start and not a service failure; it exited {code}."
            f"{explain(report)}"
        )
        assert spy.all_signals() == [], (
            f"ruling B12: …and nothing was stopped on the way to refusing; it sent "
            f"{spy.all_signals()}.{explain(report)}"
        )
        assert rig.snapshot(runs) == before, (
            "ruling B12: …and the run directory is byte-identical: no record removed, replaced or "
            f"rewritten. Before: {sorted(before)}; after: {sorted(rig.snapshot(runs))}"
        )
        for service in services:
            assert not service.started(), (
                f"ruling B12: …and nothing was started either ({service.marker} exists)"
                f"{explain(report)}"
            )
            assert box.alive(service.victim.pid), (
                f"…{service.name}'s instance (pid {service.victim.pid}) is still running"
            )
        assert "bogus" in report, (
            f"ruling B12: …and the refusal names what it could not place, or the human cannot see "
            f"the typo.{explain(report)}"
        )
    finally:
        reap(*services)


#: A victim that ignores SIGTERM and says so once its handler is installed (T-0033 ruling A16's
#: shape, reused: without the marker a row could signal it in the race window before the handler
#: exists, kill it, and silently test nothing).
STUBBORN = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "open(sys.argv[1], 'w').write('ready')\n"
    "time.sleep(60)\n"
)


def test_a_restart_whose_stop_reports_still_running_does_not_start(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 11, adversarial: a stop that did not stop must not be followed by a start.

    The instance ignores SIGTERM, so ``supervisor.stop`` reports ``still_running`` and keeps the
    state file (T-0033 ruling A16). Starting a second instance over it is how a port ends up with
    two owners and one record; and reporting success would be the 2026-09-16 lie again, one flag
    further along. So: no launcher runs, the record is kept byte-identical, and the verb exits
    non-zero.
    """
    supervisor = rig.supervisor("11")
    monkeypatch.setattr(supervisor, "STOP_WAIT_S", 0.5)
    others = [make_running(name, tmp_path, runs) for name in ("vllm", "tts")]
    stt = make_service("stt", tmp_path, runs)
    ready = tmp_path / "stubborn-ready"
    stubborn = subprocess.Popen(
        [sys.executable, "-c", STUBBORN, str(ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    stt.victim = stubborn
    try:
        assert box.wait_until(ready.exists, timeout_s=10), (
            "the victim never installed its SIGTERM handler; the row would be testing nothing"
        )
        write_record(stt, pid=stubborn.pid, start_time=start_time_of(stubborn.pid))
        plan(monkeypatch, others[0], stt, others[1])
        code = run_cli(["up", "--profile", "lean", "--restart", "stt"], probe=Probe())
        report = report_of(capsys)

        assert not stt.started(), (
            "criterion 11: a restart whose stop reported `still_running` does not proceed to "
            f"start — the launcher ran anyway ({stt.marker}).{explain(report)}"
        )
        assert stt.state_bytes() == stt.recorded_bytes, (
            "criterion 11 / T-0033 ruling A16: …and the record of the thing that is still running "
            "is kept, byte for byte: it is the only handle anyone has on it"
        )
        assert code != 0, (
            f"criterion 11: …and the verb does not report success; it exited {code}."
            f"{explain(report)}"
        )
        assert rig.verdict_of(report, "stt") != "started", (
            f"criterion 11: …nor calls it a start; the rows read {verdicts(report)}."
            f"{explain(report)}"
        )
        assert "still running" in report.lower(), (
            f"criterion 11: …and it says what actually happened.{explain(report)}"
        )
    finally:
        with contextlib.suppress(OSError):
            os.kill(stubborn.pid, signal.SIGKILL)
        box.reap(stubborn)
        reap(stt, *others)


# =================================================================================================
# Ruling A16 — one way to record a start, and it is the verified one
# =================================================================================================


def test_a_failed_start_through_the_verb_leaves_the_previous_record_byte_identical(
    tmp_path, monkeypatch, capsys, runs
):
    """Ruling A16, the fence: the eager write must not survive anywhere on the verb's path.

    The damage of 2026-09-16, one layer up from group A's row: a service that is *down* has a
    verified record from an earlier run; the start fails. If any call site still records on spawn,
    that file is replaced by a three-key file naming a dead launcher — which is precisely why
    ``astra down`` could not stop the containers that were still serving.
    """
    stt = make_service("stt", tmp_path, runs)
    write_record(stt, pid=424242, start_time=1234567)
    rig.failing_launcher(stt.script, code=125, log=rig.CONFLICT_LOG, container=None)
    vllm, tts = (make_running(name, tmp_path, runs) for name in ("vllm", "tts"))
    try:
        plan(monkeypatch, vllm, stt, tts)
        code = run_cli(["up", "--profile", "lean", "--timeout", "0"], probe=Probe(never=["stt"]))
        report = report_of(capsys)

        assert stt.state_bytes() == stt.recorded_bytes, (
            "ruling A16: the start failed, so the previous record is untouched. It now holds "
            f"{stt.state_file.read_text()!r}, written before anything was verified — the "
            f"clobbering that left `astra down` unable to stop the running containers."
            f"{explain(report)}"
        )
        assert code != 0, f"…and the verb exits non-zero; it exited {code}.{explain(report)}"
    finally:
        reap(vllm, stt, tts)


def test_the_start_verb_never_asks_for_an_eager_record():
    """Ruling A16, static: ``cli.py``'s one eager call site is converted, not left beside the new
    one.

    Two ways of recording a start coexisted through group A, and that was tolerable only because
    the failure direction is safe: a caller on the eager path writes an old-shape file, which reads
    back as *no record*, which makes the next start a refusal rather than a false skip. Leaving one
    on the verb's path reinstates the clobbering, because it still overwrites a verified record
    with a three-key file before anything has been verified.

    The scan is deliberately narrow: a ``spawn`` call is allowed if it says ``record=False``, so
    the implementer may keep a thin default runner if they want one.
    """
    source = (REPO / "local_astra" / "cli.py").read_text()
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name == "write_pid_file":
            offenders.append(f"line {node.lineno}: write_pid_file(...)")
        if name == "spawn":
            asked = [kw for kw in node.keywords if kw.arg == "record"]
            if not asked or not (
                isinstance(asked[0].value, ast.Constant) and asked[0].value.value is False
            ):
                offenders.append(f"line {node.lineno}: spawn(...) without record=False")
    assert offenders == [], (
        "ruling A16: the verb records a start through supervisor.start_service, **after** the "
        "start is verified — never on spawn. local_astra/cli.py still writes eagerly at: "
        + "; ".join(offenders)
    )


def test_spawn_no_longer_records_by_default(tmp_path):
    """Ruling A16: the eager default is retired; a caller that wants the T-0033 file asks for it.

    The seam decision (``tests/unit/_service_rig.py``, group B section): the ``record`` keyword
    survives with default ``False``. Removing it outright was rejected — T-0033 criterion 11's row
    is about the file that write produces, so there has to be a way to ask for one.
    """
    supervisor = rig.supervisor("A16")
    spawn = rig.attr(supervisor, "spawn", "A16")
    run_dir = tmp_path / "spawn-run"
    handle = spawn("ghost", ["sleep", "5"], log_dir=tmp_path / "spawn-logs", run_dir=run_dir)
    try:
        written = run_dir / "ghost.pid"
        assert not written.exists(), (
            "ruling A16: `spawn` no longer writes a state file by default — a file written before "
            "the start is verified is what destroyed the provenance of the still-serving "
            f"containers on 2026-09-16. It wrote {written.read_text()!r}"
        )
    finally:
        box.reap(None, handle.pid)


# =================================================================================================
# Ruling A17 — the settle window, reconciled (which arm was taken: see this module's docstring)
# =================================================================================================


def test_a_three_service_start_stays_inside_one_settle_window_per_started_service(
    tmp_path, monkeypatch, capsys, runs, default_settle
):
    """Ruling A17: the **default** settle window, on a *successful* start, with the amended bound.

    Until this row every happy path passed a short ``settle_s`` and the only row using the shipped
    default had a dying launcher — so the cost a human actually pays was never exercised. It is
    paid once per started service, serially, because startup order is a health-check ordering
    (docs/02 §6) and overlapping the windows would mean spawning the speech services before the
    engine answers. The Regime line's "< 1.0 s + the settle window" is therefore amended to
    ``< 1.0 s + (services started) × SETTLE_S``; this row is the amendment.

    Regime: default ``supervisor.SETTLE_S``, three ``exec sleep`` launchers, probe injected
    (refuses once per service, then answers), n = 1, ``time.monotonic``, no seed. A bound, not a
    performance claim: what it excludes is a window waited twice per service, or a flat sleep where
    a poll was specified (the 40 ms death would then cost the whole window as well).
    """
    supervisor = rig.supervisor("A17")
    window = float(rig.attr(supervisor, "SETTLE_S", "A17"))
    assert window == 1.0, (
        f"Regime: SETTLE_S is 1.0 s and this row measures the shipped default; it is {window}"
    )
    services = [make_service(name, tmp_path, runs) for name in NAMES]
    bound = 1.0 + len(services) * window
    try:
        plan(monkeypatch, *services)
        started = time.monotonic()
        code = run_cli(["up", "--profile", "lean"], probe=Probe(starting=NAMES))
        elapsed = time.monotonic() - started
        report = report_of(capsys)

        assert verdicts(report) == dict.fromkeys(NAMES, "started"), (
            f"the bound only means something if all three really started; the rows read "
            f"{verdicts(report)}.{explain(report)}"
        )
        assert code == 0, f"…and the verb exits 0; it exited {code}.{explain(report)}"
        for service in services:
            assert service.record().get("version"), (
                "…and really started them the verified way: what the window buys is the launcher's "
                f"exit status, and the file that proves it is the versioned record. "
                f"{service.name}'s file holds {service.record()}"
            )
        assert elapsed < bound, (
            f"ruling A17: three verified starts cost at most one settle window each plus the "
            f"second everything else is allowed — < {bound:g} s at SETTLE_S = {window:g} s. This "
            f"run took {elapsed:.2f} s (n=1, default window, probes injected). If the box is "
            f"loaded, say so rather than raising the bound."
        )
    finally:
        reap(*services)


def test_a_profile_that_is_entirely_skipped_costs_no_settle_window(
    tmp_path, monkeypatch, capsys, runs, default_settle
):
    """Ruling A17, the half a human feels: ``astra up`` against a running box is still instant.

    A settle window is the price of a *start*. A skip decides before anything is spawned (D1), so
    it pays none — and T-0033 criterion 10's "up returns fast" survives the amendment for the
    common case. n = 3, ``time.monotonic``, default window (1.0 s), so an implementation that
    settles per service anyway is three windows over the bound rather than one.
    """
    walls: list[float] = []
    for run in range(3):
        services = [make_running(name, tmp_path / f"run{run}", runs) for name in NAMES]
        try:
            plan(monkeypatch, *services)
            started = time.monotonic()
            code = run_cli(["up", "--profile", "lean"], probe=Probe())
            walls.append(time.monotonic() - started)
            report = report_of(capsys)
            assert code == 0 and verdicts(report) == dict.fromkeys(NAMES, "already running"), (
                f"the bound only means something if all three were skipped; exit {code}, rows "
                f"{verdicts(report)}.{explain(report)}"
            )
        finally:
            reap(*services)
            for service in services:
                service.state_file.unlink(missing_ok=True)
    assert max(walls) < 1.0, (
        "ruling A17: a skip pays no settle window — the decision is taken before anything is "
        f"spawned. The three runs took {[round(wall, 2) for wall in walls]} s against the 1.0 s "
        "bound T-0033 criterion 10 set for the whole verb."
    )


# =================================================================================================
# Criterion 13 — the regression fence
# =================================================================================================


def expected_dry_run() -> str:
    """Today's ``--dry-run`` output, byte for byte, built from the profile rather than captured.

    Criterion 13 says *byte-identical to today's*, so the typography is a literal here (the plan
    lines, two spaces of indent, then one ``then probe`` line per endpoint). What is not a literal
    is which scripts and which URLs: those come from ``services/`` and ``config.Services``, so a
    profile change moves this row's expectation with it instead of freezing a stale copy.
    """
    lines = ["astra up --profile lean --dry-run: the start plan, in order:"]
    lines += [f"  {SERVICES / script}" for script in ("vllm.sh", "stt.sh", "tts.sh")]
    services = LEAN.services
    for name, url in (("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts)):
        lines.append(f"  then probe {name} at {url}")
    return "\n".join(lines) + "\n"


def test_dry_run_output_is_byte_identical_to_todays(capsys):
    """Criterion 13: the plan a human can read may not drift while the verb changes underneath it.

    No plan is installed here on purpose: this is the real ``services/*.sh`` plan.
    """
    code = run_cli(["up", "--profile", "lean", "--dry-run"], probe=Probe())
    printed = capsys.readouterr()
    assert printed.err == "", (
        f"criterion 13: --dry-run writes nothing to stderr; it said {printed.err!r}"
    )
    assert printed.out == expected_dry_run(), (
        "criterion 13: `astra up --dry-run` is byte-identical to today's output.\n"
        f"expected:\n{expected_dry_run()!r}\ngot:\n{printed.out!r}"
    )
    assert code == 0, f"…and exits 0; it exited {code}"


def test_dry_run_with_restart_starts_and_signals_nothing(tmp_path, monkeypatch, capsys, runs):
    """Criteria 13 and 11 together: ``--dry-run`` outranks the one flag that can stop an engine."""
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    try:
        plan(monkeypatch, *services)
        spy = rig.SignalSpy(monkeypatch)
        before = rig.snapshot(runs)
        code = run_cli(
            ["up", "--profile", "lean", "--dry-run", "--restart", RESTART_ALL], probe=Probe()
        )
        report = report_of(capsys)
        assert spy.all_signals() == [], (
            f"criterion 13: `--dry-run --restart all` signals nothing; it sent "
            f"{spy.all_signals()}.{explain(report)}"
        )
        assert not any(service.started() for service in services), (
            f"…and starts nothing.{explain(report)}"
        )
        assert rig.snapshot(runs) == before, "…and moves no state file"
        assert code == 0, f"…and exits 0; it exited {code}.{explain(report)}"
    finally:
        reap(*services)


def test_the_verb_probes_nothing_but_the_profiles_own_endpoints(
    tmp_path, monkeypatch, capsys, runs
):
    """Criterion 13 (docs/02 §9): every URL the verb asks about comes from ``config.Services``.

    The pre-check is new, and a new probe is a new chance to reach off the box.
    """
    services = [make_running(name, tmp_path, runs) for name in NAMES]
    probe = Probe()
    try:
        plan(monkeypatch, *services)
        run_cli(["up", "--profile", "lean"], probe=probe)
        capsys.readouterr()
        configured = {LEAN.services.vllm, LEAN.services.stt, LEAN.services.tts}
        asked = {url for _name, url in probe.asked}
        assert asked, "criterion 13: the verb does probe the profile's endpoints"
        assert asked <= configured, (
            f"criterion 13: …and nothing else; it asked about {sorted(asked - configured)}"
        )
    finally:
        reap(*services)
