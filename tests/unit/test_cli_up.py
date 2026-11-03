"""``astra up``: detached starts, the evidence they leave, the probe loop (T-0033 criteria 10-12, 15).

Rewritten under T-0033. The rows carried over from T-0028 (the ordered plan, ``--dry-run``,
loopback-only probing, no download, nothing under ``artifacts/``) are **unmarked** and must pass
today; the rows for the new behaviour carry ``pending(task="T-0033")``.

Spec change recorded in T-0033 criterion 15: T-0028 criterion 17 seam
``Runner = Callable[[Sequence[str]], int]`` cannot express "started and still running" and is
superseded by a handle-returning runner (``.pid``, ``.argv``). ``FakeHandle`` below is an ``int``
subclass carrying those attributes precisely so the carried-over rows keep exercising *today's*
CLI (which compares the return to 0) and tomorrow's (which reads the handle) with one fake.

Regime: no GPU, no network. The one measurement is criterion 10's liveness bound — ``up`` returns
in < 1.0 s wall (``time.monotonic``, n = 5, no seed) while a real 30 s sleeper it started is still
running. That bound is three orders of magnitude away from the 30 s failure it excludes; it is not
a performance claim and is not comparable across boxes.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from local_astra import cli
from local_astra.config import LOOPBACK_PREFIXES, load
from tests.unit import _box_rig as rig
from tests.unit import _service_rig as service_rig

REPO = Path(__file__).resolve().parents[2]
SERVICES = REPO / "services"

#: The backends the ``lean`` profile needs: the engine, Kyutai STT, Pocket TTS. Read off
#: ``config.DEFAULTS`` (stt_backend kyutai-gpu, tts_backend pocket-cpu) and ``services/``, which
#: holds exactly ``vllm.sh``, ``stt.sh`` and ``tts.sh``. No talker in ``lean``.
LEAN_SCRIPTS = {"vllm.sh", "stt.sh", "tts.sh"}

#: The verbs neither T-0028 nor T-0033 implements, and the message they print today, byte for byte.
UNTOUCHED = ("bench", "eval", "forget", "fetch-models")


class FakeHandle(int):
    """What a runner hands back: a started process. ``int`` for the superseded T-0028 seam.

    T-0049 group B gives the injected runner a second job — an injected runner *replaces* the
    start of one plan entry, so what it returns is that entry's outcome (the seam is written out
    in ``tests/unit/_service_rig.py``'s group B section). So this fake carries
    ``.status``/``.ok``/``.message`` beside ``.pid``: the rows below are about the health table and
    the probe loop, and a real verified start inside each of them would cost a settle window per
    service for nothing. The rows that are about verification drive the real path with no runner
    injected — ``tests/unit/test_cli_up_verbs.py``.
    """

    def __new__(cls, argv):
        self = super().__new__(cls, 0)
        self.argv = tuple(str(part) for part in argv)
        self.pid = os.getpid()
        self.log_path = None
        self.pid_file = None
        self.name = Path(self.argv[0]).stem if self.argv else "?"
        self.status = "started"
        self.ok = True
        self.record = None
        self.message = f"{self.name}: started — pid {self.pid} (injected runner)"
        return self


class Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command) -> FakeHandle:
        self.calls.append(tuple(str(part) for part in command))
        return FakeHandle(command)


class Probe:
    """``probe(name, url) -> None | reason``. Records what it was asked about.

    ``starting`` names services that refuse the **first** time they are asked and answer
    afterwards. Every row that wants its plan actually started needs it under T-0049 group B: a
    service that is already healthy when ``up`` decides is skipped or refused, never started over
    (criterion 10), so "healthy from the first ask" is no longer the setup for a row about
    starting. It costs one extra probe round and changes nothing about what these rows assert.
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


def run_cli(argv: list[str], **kwargs: object) -> int:
    try:
        return int(cli.main(argv, **kwargs))  # type: ignore[arg-type]
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    except TypeError as exc:
        raise AssertionError(
            f"T-0033: cli.main(argv, *, runner=None, probe=None) — {exc}"
        ) from None


def plan_for(profile_name: str = "lean"):
    start_plan = getattr(cli, "start_plan", None)
    assert start_plan is not None, (
        "cli.start_plan(profile) -> the ordered services/*.sh invocations that profile's backends "
        "need (T-0028 criterion 17; tests/unit/_hud_rig.py's docstring fixes the seam)"
    )
    profile = load(None).profiles[profile_name]
    return [tuple(str(part) for part in entry) for entry in start_plan(profile)]


def _flags_of(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            out |= {
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            }
    return out


# =================================================================================================
# Isolation: no row in this module may read the checkout's real ``logs/``.
# =================================================================================================


@pytest.fixture
def real_log_dir():
    """Opt out of :func:`isolate_service_logs`.

    A marker would have to be registered in ``pyproject.toml`` (``--strict-markers``), which is not
    this file's to edit; requesting a fixture is the local equivalent. Exactly one row wants it:
    the one that asserts *where* ``supervisor.log_dir()`` points.
    """
    return True


@pytest.fixture(autouse=True)
def isolate_service_logs(request, monkeypatch, tmp_path):
    """Point ``supervisor.log_dir()`` at an empty ``tmp_path/logs`` for every row here.

    ``cli.health_rows`` calls ``supervisor.log_tail(name)`` for **every** FAILED row (T-0041
    criterion 3) and the real ``log_dir()`` is the checkout's ``logs/``. So any row that probes a
    service unhealthy quotes whatever the last real ``astra up`` on this box left in
    ``logs/<name>.log`` — and Pocket TTS logs its own URL, at which point ``row_for`` finds two
    lines naming ``tts`` and its endpoint and the row fails for a reason that has nothing to do
    with the CLI. That is a flake in the test, not evidence about the code, so the log directory
    is a fixture here and never the checkout's.

    Rows that want their log content call :func:`use_log_dir` afterwards; it wins, because
    ``monkeypatch`` applies the last assignment. Rows that pass ``log_dir=`` to
    ``supervisor.spawn`` are unaffected either way.
    """
    if "real_log_dir" in request.fixturenames:
        return
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runs = tmp_path / "run"
    runs.mkdir(parents=True, exist_ok=True)
    supervisor_mod = rig.module("supervisor")
    monkeypatch.setattr(supervisor_mod, "log_dir", lambda: logs)
    monkeypatch.setattr(supervisor_mod, "run_dir", lambda: runs)
    if hasattr(cli, "log_dir"):
        monkeypatch.setattr(cli, "log_dir", lambda: logs)
    if hasattr(cli, "run_dir"):
        monkeypatch.setattr(cli, "run_dir", lambda: runs)
    # T-0049 ruling A8: `up` reads the state records before it decides and writes one after a
    # verified start, so a row that left the real run directory in place would read — and could
    # rewrite — the identity of the services running on this box. That is the damage the
    # 2026-09-16 run did, and it would be invisible in a green suite.
    assert ".git" not in os.fspath(supervisor_mod.run_dir())


# =================================================================================================
# Carried over from T-0028 verbatim (criteria 17, 18): these pass today and must keep passing.
# =================================================================================================


def test_the_start_plan_is_the_profiles_backends_in_a_fixed_order():
    """T-0028 criterion 17: the plan is exactly what the profile needs, and it is deterministic."""
    plan = plan_for("lean")
    assert plan, "criterion 17: `astra up --profile lean` has a plan"
    scripts = [Path(entry[0]).name for entry in plan]
    assert set(scripts) == LEAN_SCRIPTS, (
        f"criterion 17: the lean profile needs {sorted(LEAN_SCRIPTS)} (the engine, Kyutai STT, "
        f"Pocket TTS); the plan runs {scripts}"
    )
    for entry in plan:
        path = Path(entry[0])
        resolved = path if path.is_absolute() else REPO / path
        assert resolved.is_file(), f"criterion 17: {entry[0]} is not a file in services/"
    assert plan_for("lean") == plan, "the plan is pure: two calls, one answer"


def test_dry_run_prints_the_ordered_plan_and_runs_nothing(capsys):
    """T-0028 criterion 17: ``--dry-run`` prints the exact ordered plan and executes none of it."""
    runner, probe = Runner(), Probe()
    code = run_cli(["up", "--profile", "lean", "--dry-run"], runner=runner, probe=probe)
    printed = capsys.readouterr()
    out = printed.out + printed.err
    assert code == 0, f"--dry-run exits 0; it exited {code}. Output:\n{out}"
    assert runner.calls == [] and probe.asked == [], (
        f"criterion 17: --dry-run runs nothing and probes nothing; it ran {runner.calls} and "
        f"probed {probe.asked}"
    )
    positions = [out.find(Path(entry[0]).name) for entry in plan_for("lean")]
    assert all(p >= 0 for p in positions), (
        f"criterion 17: every command of the plan is printed; the output was\n{out}"
    )
    assert positions == sorted(positions), (
        f"…in the plan's order, which is the order they will be started in; the output was\n{out}"
    )


def test_the_runner_runs_the_plan_in_order_and_every_loopback_endpoint_is_probed(capsys):
    """T-0028 criterion 17: execute the plan through the runner, then health-probe each endpoint.

    ``starting``: under T-0049 criterion 10 a service that is already healthy when ``up`` decides
    is skipped, not started, so the setup for "the plan is executed" is a profile that is **down**.
    """
    runner, probe = Runner(), Probe(starting=("vllm", "stt", "tts"))
    code = run_cli(["up", "--profile", "lean"], runner=runner, probe=probe)
    out = capsys.readouterr()
    assert runner.calls == plan_for("lean"), (
        f"criterion 17: the plan is executed in order; the runner saw {runner.calls}"
    )
    services = load(None).profiles["lean"].services
    expected = {services.vllm, services.stt, services.tts}
    assert {url for _, url in probe.asked} == expected, (
        f"criterion 17: every loopback endpoint from config.Services is probed; it probed "
        f"{probe.asked}, expected {sorted(expected)}"
    )
    for name, url in probe.asked:
        assert url.startswith(LOOPBACK_PREFIXES), (
            f"criterion 17, negative: {name} was probed at {url}, which is not loopback "
            "(docs/02 §9)"
        )
    assert code == 0, f"every probe passed, so `astra up` exits 0; it exited {code}. {out.err}"
    report = out.out + out.err
    for name, _ in probe.asked:
        assert name in report, (
            f"criterion 17: per-service pass/fail is printed; {name} is missing from\n{report}"
        )


def test_a_failing_probe_names_the_service_and_the_reason_and_exits_non_zero(capsys):
    """T-0028 criteria 17 and 19: a failed health probe is loud, and it says which service and why."""
    runner = Runner()
    probe = Probe(failures={"tts": "connection refused on 8081"}, starting=("vllm", "stt"))
    code = run_cli(["up", "--profile", "lean", "--timeout", "1"], runner=runner, probe=probe)
    out = capsys.readouterr()
    report = out.out + out.err
    assert code != 0, f"criterion 19: a failed probe exits non-zero; it exited {code}"
    assert "tts" in report and "connection refused on 8081" in report, (
        "criterion 17: the failure report names the service and the failing reason; it said:\n"
        + report
    )
    assert probe.asked, "the probe loop must actually have run"
    assert runner.calls == plan_for("lean"), "…after the plan was started, not instead of it"


def test_up_never_downloads_a_model_and_writes_no_artifact(capsys, tmp_path, monkeypatch):
    """T-0028 criterion 17, negative: ``fetch-models`` is the only network egress (docs/02 §9)."""
    runner, probe = Runner(), Probe(starting=("vllm", "stt", "tts"))
    monkeypatch.chdir(tmp_path)
    run_cli(["up", "--profile", "lean"], runner=runner, probe=probe)
    flat = " ".join(" ".join(call) for call in runner.calls).lower()
    for banned in ("fetch", "download", "huggingface-cli", "hf_transfer", "wget", "curl"):
        assert banned not in flat, (
            f"criterion 17: `astra up` never downloads a model; the plan ran {runner.calls}"
        )
    assert not (tmp_path / "artifacts").exists(), (
        "criterion 17: `astra up` writes nothing into artifacts/"
    )
    capsys.readouterr()


def test_nothing_else_in_the_cli_moves(capsys):
    """T-0028 criterion 18: the verbs nobody has implemented keep today's exit code and message."""
    assert "--dry-run" in _flags_of(REPO / "local_astra" / "cli.py"), (
        "criterion 17: `astra up --profile X --dry-run` must exist before this fence means anything"
    )
    profile = load(None).profile
    for verb in UNTOUCHED:
        argv = [verb]
        if verb == "bench":
            argv.append("B1")
        elif verb == "eval":
            argv.append("E1")
        elif verb == "forget":
            argv += ["--since", "2026-01-01"]
        code = run_cli(argv)
        printed = capsys.readouterr()
        assert code == 2, f"criterion 18: `astra {verb}` still exits 2; it exited {code}"
        assert printed.err == f"astra {verb}: not implemented yet (profile {profile})\n", (
            f"criterion 18: `astra {verb}`'s message is byte-identical to today's; it said "
            f"{printed.err!r}"
        )
        assert printed.out == "", f"…on stderr only; stdout carried {printed.out!r}"

    assert run_cli(["config"]) == 0
    printed = capsys.readouterr()
    assert '"name": "lean"' in printed.out, (
        f"criterion 18: `astra config` still prints the resolved profile; it printed\n{printed.out}"
    )


# =================================================================================================
# T-0033: detached starts (criterion 10), the evidence (11), the probe loop (12).
# =================================================================================================


#: A driver that runs the real verb with the **default** runner (nothing injected), against a
#: start plan of three scripts that each ``exec`` a 30 s server — the shape of every
#: ``services/*.sh``. Run in a child process with a timeout, because the failure this test exists
#: for is a call that never returns: an in-process test could only wait for it.
LIVENESS_DRIVER = """
import json, sys, time
from pathlib import Path

from local_astra import cli, supervisor

run_dir, log_dir, scripts = (Path(a) for a in sys.argv[1:4])
supervisor.run_dir = lambda: run_dir
supervisor.log_dir = lambda: log_dir
cli.start_plan = lambda profile: tuple(
    (str(scripts / name),) for name in ("vllm.sh", "stt.sh", "tts.sh")
)
# T-0049: the verb decides before it spawns, so a service that answers on the first ask is
# skipped rather than started (criterion 10) — these three are down, and come up once started.
# The settle window is the cost of a *start*; shortened here because this row is about the verb
# returning while the server runs, not about the window (that is
# tests/unit/test_cli_up_verbs.py::test_a_three_service_start_stays_inside_one_settle_window…).
supervisor.SETTLE_S = 0.2
asked = {}


def probe(name, url):
    asked[name] = asked.get(name, 0) + 1
    return None if asked[name] > 1 else "%s: not up yet" % name


started = time.monotonic()
code = cli.main(["up", "--profile", "lean"], probe=probe)
print(json.dumps({"wall": time.monotonic() - started, "code": code}))
"""


#: The settle window the driver above installs. T-0049 ruling A17: a verified start waits it out
#: per service, so T-0033 criterion 10's bound becomes "1.0 s + one window per service started"
#: (the reasoning is in tests/unit/test_cli_up_verbs.py's docstring). Shortened here so this row
#: keeps costing what it did; the shipped 1.0 s default is measured by the row named above.
DRIVER_SETTLE_S = 0.2


def _sleeper_scripts(directory: Path) -> Path:
    """Three scripts shaped like ``services/*.sh``: they never exit on their own."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("vllm.sh", "stt.sh", "tts.sh"):
        path = directory / name
        path.write_text("#!/usr/bin/env bash\nexec sleep 30\n")
        path.chmod(0o755)
    return directory


def test_up_returns_while_a_foreground_server_is_still_running(tmp_path):
    """Criterion 10, adversarial (P0-1): the single most likely regression, and the 30 s it costs.

    The *default* runner is under test here, not an injected one: T-0033 records that the injected
    ``Runner`` returning instantly is precisely why the fast suite never saw this. The driver runs
    in a child process under a 5 s timeout so a runner that waits for the server fails the test in
    5 s instead of 30.

    Regime: fast suite, no GPU, no network, profile ``lean``, ``astra.toml`` defaults, probe
    stubbed healthy, three ``exec sleep 30`` scripts as the plan. Threshold: the verb returns in
    < 1.0 s wall (``time.monotonic`` inside the child), n = 5 repeats, no seed. Three orders of
    magnitude from the 30 s failure it excludes — not a performance claim, not comparable boxes.
    """
    rig.module("supervisor")  # the seam must exist before the driver can mean anything
    driver = tmp_path / "driver.py"
    driver.write_text(LIVENESS_DRIVER)
    scripts = _sleeper_scripts(tmp_path / "services")
    walls: list[float] = []
    pids: list[int] = []
    try:
        for run in range(5):
            run_dir = tmp_path / f"run{run}"
            log_dir = tmp_path / f"logs{run}"
            try:
                done = subprocess.run(
                    [sys.executable, str(driver), str(run_dir), str(log_dir), str(scripts)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=REPO,
                )
            except subprocess.TimeoutExpired:
                raise AssertionError(
                    "criterion 10: the verb never returned — it is waiting for a server that "
                    "execs and does not exit (T-0033 P0-1: cli.run_service is a blocking "
                    "subprocess.run and every services/*.sh ends in exec)"
                ) from None
            assert done.returncode == 0, f"the driver failed:\n{done.stdout}\n{done.stderr}"
            report = json.loads(done.stdout.strip().splitlines()[-1])
            assert report["code"] == 0, f"every probe was healthy, so it exits 0: {report}"
            walls.append(report["wall"])
            for name in ("vllm", "stt", "tts"):
                pid_file = run_dir / f"{name}.pid"
                assert pid_file.exists(), (
                    f"criterion 11: every start leaves a pid file; {pid_file} is missing"
                )
                pids.append(rig.attr(rig.module("supervisor"), "read_pid_file")(pid_file)[0])

        bound = 1.0 + 3 * DRIVER_SETTLE_S
        assert max(walls) < bound, (
            "criterion 10: each entry of the start plan is spawned detached and the verb proceeds "
            f"to the next without waiting for it to exit (< {bound:g} s = 1.0 s + one "
            f"{DRIVER_SETTLE_S:g} s settle window per started service, T-0049 ruling A17; n=5); "
            f"the walls were {walls}"
        )
        assert len(pids) == 15, f"five runs of a three-entry plan; got {len(pids)} pid files"
        dead = [pid for pid in pids if not rig.alive(pid)]
        assert not dead, (
            "criterion 10, negative: the servers outlive the command that started them — the "
            f"child is in its own session, so nothing killed it when the CLI exited. Dead: {dead}"
        )
    finally:
        for pid in pids:
            rig.reap(None, pid)


def test_the_default_runner_spawns_into_its_own_session(tmp_path):
    """Criterion 10, negative: Ctrl-C on the CLI must not reach the engine it just started."""
    supervisor = rig.module("supervisor")
    spawn = rig.attr(supervisor, "spawn")
    handle = spawn(
        "sleeper",
        ["sleep", "30"],
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "run",
    )
    try:
        assert rig.alive(handle.pid), "the child is running"
        assert os.getpgid(handle.pid) != os.getpgid(0), (
            "criterion 10: the child is spawned with start_new_session/setsid, so it is not in "
            "the CLI process group and an interrupt does not reach the engine "
            f"(child pgid {os.getpgid(handle.pid)} == ours)"
        )
        assert os.getsid(handle.pid) != os.getsid(0), "…its own session, not just its own group"
        assert rig.alive(handle.pid), "…and it is still running after the spawn returned"
    finally:
        rig.reap(None, handle.pid)


def start_time_of(pid: int) -> int:
    """``/proc/<pid>/stat`` field 22 (starttime), read from the kernel, not from the supervisor.

    Field 2 is the executable name in parentheses and may contain spaces and ``)``, so the fields
    are counted from the last ``)``: field 22 is the 20th token after it.
    """
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rindex(")") + 1 :].split()[19])


def test_a_start_writes_a_log_and_a_pid_file_under_the_git_common_dir(tmp_path):
    """Criterion 11: stdout+stderr to logs/<service>.log, pid+argv+start time under the run dir.

    The start time is T-0033 ruling A11: ``down`` kills iff it matches, so a pid file that does
    not carry it cannot tell the process it started from a stranger that inherited the number.
    """
    supervisor = rig.module("supervisor")
    spawn = rig.attr(supervisor, "spawn")
    run_dir = tmp_path / "run"
    log_dir = tmp_path / "logs"
    argv = ["bash", "-c", "echo hello-stdout; echo hello-stderr >&2; sleep 20"]
    # ``record=True`` by name: T-0049 ruling A16 retires the eager *default* (a record written
    # before a start is verified is what destroyed the provenance of the running containers on
    # 2026-09-16), and this row is about the file that write produces, so it asks for one.
    handle = spawn("vllm", argv, log_dir=log_dir, run_dir=run_dir, record=True)
    try:
        log = Path(handle.log_path)
        assert log == log_dir / "vllm.log", f"criterion 11: logs/<service>.log; got {log}"
        assert rig.wait_until(
            lambda: log.exists() and "hello-stdout" in log.read_text(), timeout_s=5
        ), "criterion 11: stdout goes to the log"
        assert rig.wait_until(lambda: "hello-stderr" in log.read_text(), timeout_s=5), (
            f"criterion 11: …and stderr with it; the log held {log.read_text()!r}"
        )

        pid_file = Path(handle.pid_file)
        assert pid_file == run_dir / "vllm.pid", (
            f"criterion 11: <run dir>/<service>.pid; got {pid_file}"
        )
        text = pid_file.read_text()
        assert str(handle.pid) in text, (
            f"criterion 11: the pid file holds the pid; it held {text!r}"
        )
        assert all(part in text for part in argv), (
            f"criterion 11: …plus the argv that produced it; it held {text!r}"
        )
        expected_start = start_time_of(handle.pid)  # read from /proc, an independent oracle
        assert str(expected_start) in text, (
            f"criterion 11 / ruling A11: …plus the process start time (/proc/<pid>/stat field 22, "
            f"here {expected_start}), which is what `down` compares; it held {text!r}"
        )
        record = rig.attr(supervisor, "read_pid_file")(pid_file)
        assert len(tuple(record)) == 3, (
            f"ruling A11: read_pid_file returns (pid, argv, start_time); it returned {record!r}"
        )
        read_pid, read_argv, read_start = record
        assert (read_pid, tuple(read_argv), read_start) == (
            handle.pid,
            tuple(argv),
            expected_start,
        ), f"criterion 11: write/read round-trip; got {(read_pid, tuple(read_argv), read_start)!r}"
    finally:
        rig.reap(None, handle.pid)


def test_the_default_run_directory_is_the_git_common_dir_and_never_artifacts(real_log_dir):
    """Criterion 11: worktrees share one state directory, and it is not under ``artifacts/``.

    ``real_log_dir`` opts this row out of :func:`isolate_service_logs`: it is the one row whose
    subject *is* the un-patched ``supervisor.log_dir()``.
    """
    supervisor = rig.module("supervisor")
    run_dir = rig.attr(supervisor, "run_dir")()
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected = (Path(common) if Path(common).is_absolute() else REPO / common).resolve()
    assert Path(run_dir).resolve() == (expected / "local_astra" / "run").resolve(), (
        "criterion 11: the pid directory is `git rev-parse --git-common-dir`/local_astra/run so "
        f"the worktrees share one state dir; it said {run_dir}"
    )
    assert "artifacts" not in Path(run_dir).parts, (
        f"criterion 11, negative: nothing is written under artifacts/; the run dir is {run_dir}"
    )
    log_dir = Path(rig.attr(supervisor, "log_dir")())
    assert log_dir.resolve() == (REPO / "logs").resolve(), (
        f"criterion 11: logs/ (already gitignored); it said {log_dir}"
    )


def test_a_failed_spawn_leaves_no_pid_file(tmp_path):
    """Criterion 11: the pid file is written only *after* the spawn succeeds."""
    supervisor = rig.module("supervisor")
    spawn = rig.attr(supervisor, "spawn")
    run_dir = tmp_path / "run"
    with pytest.raises(OSError):
        spawn(
            "ghost",
            [str(tmp_path / "there-is-no-such-binary")],
            log_dir=tmp_path / "logs",
            run_dir=run_dir,
        )
    assert not (run_dir / "ghost.pid").exists(), (
        "criterion 11: a pid file for a process that never started is a stop command that kills a "
        "stranger later"
    )


def test_the_probe_loop_reports_per_service_budget_and_the_last_response(monkeypatch, capsys):
    """Criterion 12: the engine gets a long budget and the speech services a short one, and a
    timeout says which service, which endpoint, how long it waited and the last response.

    No ``--timeout``: under ruling A14 the operator's cap would then *be* the budget every row
    printed, and this row is about the per-service ones. The virtual clock (A15) is what makes
    waiting a 20 s budget out free.
    """
    supervisor = rig.module("supervisor")
    budget = rig.attr(supervisor, "probe_budget_s")
    assert budget("vllm") >= 60.0, (
        f"criterion 12: the engine init is ~68 s, so its budget is long; it is {budget('vllm')}"
    )
    for speech in ("stt", "tts"):
        assert budget(speech) < budget("vllm"), (
            f"criterion 12: the speech services get a short budget; {speech} got {budget(speech)} "
            f"against the engine's {budget('vllm')}"
        )

    runner = Runner()
    install_clock(monkeypatch)
    probe = Probe(failures={"tts": 'HTTP 503 {"status": "starting"}'}, starting=("vllm", "stt"))
    code = run_cli(["up", "--profile", "lean"], runner=runner, probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    assert code != 0, "criterion 12: exit non-zero if any service is unhealthy at the deadline"
    services = load(None).profiles["lean"].services
    assert "tts" in report and services.tts in report, (
        f"criterion 12: the timeout says which service and which endpoint; it said\n{report}"
    )
    assert "503" in report and "starting" in report, (
        f"criterion 12: …and what the last response was; it said\n{report}"
    )
    assert any(token in report for token in ("waited", "after", " s")), (
        f"criterion 12: …and how long it waited; it said\n{report}"
    )
    for name in ("vllm", "stt", "tts"):
        assert f"{budget(name):g}" in report or f"{budget(name):.0f}" in report, (
            f"criterion 12: the per-service budget is printed; the {name} one ({budget(name)} s) "
            f"is missing from\n{report}"
        )


# =================================================================================================
# T-0033 rulings A14/A15: the budget the loop honours, and a loop that cannot spin.
# =================================================================================================


class Clock:
    """Virtual time: ``monotonic`` only ever moves because somebody slept (no wall clock at all).

    Every sleep argument is kept, so a loop that sleeps 0 while a service is still pending — the
    busy-wait ruling A15 forbids — is visible as a ``0.0`` in ``sleeps`` and, because time does
    not advance on its own, as a probe count that runs away.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.t += max(0.0, float(seconds))


class Runaway(RuntimeError):
    """The probe loop asked more times than its budget can pay for."""


class TimedProbe:
    """``probe(name, url)`` against the virtual clock: healthy from ``healthy_at[name]`` seconds on.

    ``healthy_at[name] is None`` means "refuses for ever". ``max_calls`` is the test's own
    runaway fuse: without it a loop that sleeps 0 against a virtual clock never returns.
    """

    def __init__(self, clock: Clock, healthy_at: dict[str, float | None], max_calls: int) -> None:
        self.clock = clock
        self.healthy_at = healthy_at
        self.max_calls = max_calls
        self.asked: list[tuple[str, str, float]] = []

    def __call__(self, name: str, url: str) -> str | None:
        if len(self.asked) >= self.max_calls:
            raise Runaway(
                f"the probe loop made more than {self.max_calls} calls "
                f"(virtual clock stopped at {self.clock.t:.2f} s, "
                f"last sleeps {self.clock.sleeps[-5:]})"
            )
        self.asked.append((name, url, self.clock.t))
        ready = self.healthy_at.get(name)
        if ready is not None and self.clock.t >= ready:
            return None
        return f"{name}: connection refused (t={self.clock.t:.2f})"

    def calls_for(self, name: str) -> int:
        return sum(1 for asked, _url, _t in self.asked if asked == name)


def install_clock(monkeypatch) -> Clock:
    """Drive ``up``'s probe loop off virtual time. The seams are ``cli._monotonic``/``cli._sleep``."""
    clock = Clock()
    for seam, replacement in (("_monotonic", clock.monotonic), ("_sleep", clock.sleep)):
        assert hasattr(cli, seam), (
            f"T-0033 ruling A15: the probe loop reads the clock and sleeps through cli.{seam}, so "
            "a test can drive 68 s of engine init without waiting 68 s. That seam is missing."
        )
        monkeypatch.setattr(cli, seam, replacement)
    return clock


def budget_in(row: str) -> float:
    """The budget this row printed, as a number (criterion 12 prints it beside the endpoint)."""
    match = re.search(r"budget\s+([0-9]+(?:\.[0-9]+)?)\s*s", row)
    assert match is not None, f"criterion 12: the row states the budget it used; it said {row!r}"
    return float(match.group(1))


def row_for(report: str, name: str, url: str) -> str:
    """The health table's row for ``name``, selected by the table's shape (ruling B17).

    Not "the line naming the service and its endpoint": a verdict row's refusal says ``healthy at
    <url>`` as well, so that selection raises inside the helper for any report holding a refusal —
    the same defect, and the same fix, as ``_health_lines`` in ``tests/unit/test_cli_down_status``.
    """
    matching = service_rig.health_rows_for(report, name, url)
    assert len(matching) == 1, (
        f"criterion 12: exactly one row states {name}'s health and names {url}; found {matching} "
        f"in\n{report}"
    )
    return matching[0]


def test_a_cold_engine_answering_at_68s_under_a_plain_astra_up_exits_zero(monkeypatch, capsys):
    """Ruling A14: with no ``--timeout``, the budget the loop honours is the per-service one.

    The engine's init is ~68 s (docs/02), which is why ``supervisor.probe_budget_s("vllm")`` is
    long. An argparse default that silently caps every run at 60 s turns every cold start into a
    FAILED row on a box where nothing is wrong — so the default is ``None`` and the printed budget
    is the one that was used.

    Regime: virtual clock (``cli._monotonic``/``cli._sleep``), injected probe, profile ``lean``,
    no GPU, no network. The engine answers the first probe taken at t >= 68 s; the speech services
    answer at once.
    """
    supervisor_mod = rig.module("supervisor")
    engine_budget = rig.attr(supervisor_mod, "probe_budget_s")("vllm")
    assert engine_budget > 68.0, (
        f"criterion 12: the engine's budget must cover its ~68 s init; it is {engine_budget}"
    )
    clock = install_clock(monkeypatch)
    probe = TimedProbe(clock, {"vllm": 68.0, "stt": 0.25, "tts": 0.25}, max_calls=5000)
    code = run_cli(["up", "--profile", "lean"], runner=Runner(), probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code == 0, (
        "ruling A14: a plain `astra up` gives the engine its own budget "
        f"({engine_budget:g} s), so an engine that answers at 68 s is healthy and the verb exits "
        f"0; it exited {code} after {clock.t:.1f} s of probing. Output:\n{report}"
    )
    vllm_row = row_for(report, "vllm", services.vllm)
    assert "FAILED" not in vllm_row, f"…and its row is not a failure; it said {vllm_row!r}"
    assert budget_in(vllm_row) == engine_budget, (
        "ruling A14: the printed budget is the one the loop used — with no --timeout that is "
        f"supervisor.probe_budget_s('vllm') = {engine_budget:g}; the row said {vllm_row!r}"
    )
    assert clock.t >= 68.0, (
        f"the engine only answered at 68 s, so the loop waited at least that long; it stopped at "
        f"{clock.t:.2f} s (n.b. virtual seconds)"
    )


def test_an_explicit_timeout_still_caps_the_budget_and_is_the_one_printed(monkeypatch, capsys):
    """Ruling A14, the other half: ``--timeout`` caps when the operator passes it, and says so.

    The operator asked for one second. The engine's 120 s budget must not override that, and the
    row must not print a budget the loop did not honour — a report that says "budget 120 s, waited
    1.0 s" is a lie about why the row failed.
    """
    clock = install_clock(monkeypatch)
    probe = TimedProbe(clock, {"vllm": 68.0, "stt": 0.25, "tts": 0.25}, max_calls=5000)
    code = run_cli(["up", "--profile", "lean", "--timeout", "1"], runner=Runner(), probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code != 0, (
        f"ruling A14: --timeout 1 caps the engine's budget, so the engine that needs 68 s is "
        f"FAILED and the verb exits non-zero; it exited {code}. Output:\n{report}"
    )
    assert clock.t <= 2.0, (
        f"…and the cap is what the loop waited: it probed for {clock.t:.2f} virtual seconds "
        f"against a 1 s cap"
    )
    vllm_row = row_for(report, "vllm", services.vllm)
    assert budget_in(vllm_row) == 1.0, (
        "ruling A14: the printed budget is the one used — here the operator's 1 s, not "
        f"supervisor.probe_budget_s('vllm'); the row said {vllm_row!r}"
    )


def test_the_probe_loop_never_sleeps_zero_and_its_call_count_is_bounded(monkeypatch, capsys):
    """Ruling A15, adversarial: the busy-wait. Two services past their budget, one still pending.

    ``stt`` and ``tts`` refuse for ever; their 20 s budgets run out long before the engine's. A
    loop that keeps a service in ``pending`` after its budget has expired computes its next sleep
    from that expired budget, gets ``0``, and spins — on a real clock that is a core at 100 % for
    100 s, and it is invisible to a test that only checks the exit code. So: a service past its
    budget is recorded FAILED and *removed* from pending, the loop sleeps ``PROBE_INTERVAL_S``
    while anything is still pending, and the number of probe calls is bounded by the budget.

    The bound is computed by hand: the longest budget is the engine's, one round of probes every
    ``PROBE_INTERVAL_S``, at most one call per service per round, plus slack for the first and
    last rounds.
    """
    supervisor_mod = rig.module("supervisor")
    budget_of = rig.attr(supervisor_mod, "probe_budget_s")
    names = ("vllm", "stt", "tts")
    longest = max(budget_of(name) for name in names)
    rounds = math.ceil(longest / cli.PROBE_INTERVAL_S) + 2
    bound = rounds * len(names) + 8

    clock = install_clock(monkeypatch)
    probe = TimedProbe(clock, dict.fromkeys(names), max_calls=bound)
    try:
        code = run_cli(["up", "--profile", "lean"], runner=Runner(), probe=probe)
    except Runaway as exc:
        raise AssertionError(
            f"ruling A15: the probe loop is spinning — {exc}. With the longest budget "
            f"{longest:g} s and one round every {cli.PROBE_INTERVAL_S} s, three services cannot "
            f"need more than {bound} probe calls."
        ) from None
    printed = capsys.readouterr()
    report = printed.out + printed.err

    assert code != 0, f"nothing ever answered, so the verb fails; it exited {code}.\n{report}"
    assert clock.sleeps, "ruling A15: the loop sleeps between rounds while a service is pending"
    assert min(clock.sleeps) > 0.0, (
        "ruling A15: the sleep argument is never 0 while services are pending — a 0 s sleep is a "
        f"busy-wait on a core until the engine's budget expires. The sleeps were "
        f"{clock.sleeps[:12]} … min {min(clock.sleeps)}"
    )
    assert max(clock.sleeps) <= cli.PROBE_INTERVAL_S + 1e-9, (
        f"…and never longer than PROBE_INTERVAL_S ({cli.PROBE_INTERVAL_S} s), or a service that "
        f"comes up is noticed late; the sleeps were {clock.sleeps[:12]}"
    )
    assert len(probe.asked) <= bound, (
        f"ruling A15: at most {bound} probe calls for three services over {longest:g} s; it made "
        f"{len(probe.asked)}"
    )
    for speech in ("stt", "tts"):
        assert probe.calls_for(speech) < probe.calls_for("vllm"), (
            f"ruling A15: {speech} is past its {budget_of(speech):g} s budget long before the "
            f"engine's {budget_of('vllm'):g} s, so it is recorded FAILED and removed from pending "
            f"— it must not keep being asked. {speech}: {probe.calls_for(speech)} calls, vllm: "
            f"{probe.calls_for('vllm')}"
        )
    services = load(None).profiles["lean"].services
    for name, url in (("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts)):
        assert "FAILED" in row_for(report, name, url), (
            f"criterion 12: {name} never answered, so its row is FAILED; the report was\n{report}"
        )


# =================================================================================================
# T-0041: the engine's budget covers a cold start, and a FAILED row names what the log says.
#
# Origin: the first real `astra up --profile lean` on this box, 2026-09-15 23:00, after a reboot.
# vLLM was healthy at ~130 s (engine init 69 s incl. 33 s compilation, plus container start and a
# cold page cache) and `up` printed FAILED at 120 s, seconds from ready; `tts` printed
# "connection refused" while its log held the real reason — taskset could not exec pocket-tts.
# Regime for every row below: virtual clock, injected probe and runner, profile `lean`, no GPU,
# no network, no real service. Virtual seconds throughout.
# =================================================================================================

#: T-0041 criterion 2. The measured cold start is ~130 s (n=1, this box, logs/vllm.log,
#: 2026-09-15); 240 s is that with room for a colder one (a page cache emptied by a reboot and a
#: recompile), and it is the number the row prints.
COLD_START_BUDGET_S = 240.0

#: What `logs/tts.log` really held after the reboot, byte for byte.
EXEC_FAILURE = "taskset: failed to execute pocket-tts: No such file or directory"


def use_log_dir(monkeypatch, path):
    """Point the service logs at ``path``; the report must ask ``supervisor.log_dir()`` per call."""
    supervisor_mod = rig.module("supervisor")
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(supervisor_mod, "log_dir", lambda: path)
    if hasattr(cli, "log_dir"):
        monkeypatch.setattr(cli, "log_dir", lambda: path)
    return supervisor_mod


class LoggingRunner(Runner):
    """Like ``Runner``, but its handles carry ``log_path`` the way ``supervisor.spawn``'s do.

    Both routes to the log are therefore open to the implementer — the handle ``up`` holds and
    ``supervisor.log_dir()`` — and no row below depends on which one is taken.
    """

    def __init__(self, log_dir) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)

    def __call__(self, command):
        handle = super().__call__(command)
        handle.log_path = self.log_dir / f"{Path(str(command[0])).stem}.log"
        return handle


def tail_near(report: str, row: str, needle: str) -> bool:
    """Is ``needle`` on the row itself or on the line right after it?

    The task fixes *what* a FAILED row carries, not the typography: a tail appended to the row
    and a tail printed as the next (indented) line are both "the row carries it".
    """
    lines = report.splitlines()
    index = lines.index(row)
    return any(needle in line for line in lines[index : index + 2])


def test_the_engine_probe_budget_covers_a_cold_start_after_a_reboot():
    """Criterion 2: >= 240 s for the engine; the speech services keep their 20 s.

    120 s was the old value and it is 10 s short of the one cold start we have measured, so a
    rebooted box fails `astra up` while the engine is still initialising. The speech services
    must *not* be widened with it: a dead moshi-server that is asked for four minutes turns a
    typo into a coffee break (ruling A14 is exactly this asymmetry).
    """
    supervisor_mod = rig.module("supervisor")
    budget = rig.attr(supervisor_mod, "probe_budget_s")
    assert budget("vllm") >= COLD_START_BUDGET_S, (
        f"criterion 2: the engine needed ~130 s on this box after a reboot (69 s init incl. 33 s "
        f"compilation, plus container start and a cold page cache; n=1, 2026-09-15, "
        f"logs/vllm.log), so its budget is >= {COLD_START_BUDGET_S:g} s; it is {budget('vllm')}"
    )
    for speech in ("stt", "tts"):
        assert budget(speech) == 20.0, (
            f"criterion 2: the speech services keep their 20 s — they answer in seconds and a "
            f"long budget only delays the report of a dead one; {speech} got {budget(speech)}"
        )


def test_a_cold_engine_answering_at_200s_under_a_plain_astra_up_exits_zero(monkeypatch, capsys):
    """Criterion 2, as behaviour: 200 s of engine init is a healthy start, not a FAILED row.

    200 s is inside the >= 240 s budget and well outside today's 120 s, which is the regression
    this row exists for. The printed budget is the one honoured (T-0033 ruling A14), so it is
    checked against ``supervisor.probe_budget_s('vllm')`` rather than against a literal.
    """
    supervisor_mod = rig.module("supervisor")
    engine_budget = rig.attr(supervisor_mod, "probe_budget_s")("vllm")
    ready_at = 200.0
    rounds = math.ceil(COLD_START_BUDGET_S / cli.PROBE_INTERVAL_S) + 2
    clock = install_clock(monkeypatch)
    probe = TimedProbe(
        clock, {"vllm": ready_at, "stt": 0.25, "tts": 0.25}, max_calls=rounds * 3 + 8
    )
    code = run_cli(["up", "--profile", "lean"], runner=Runner(), probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code == 0, (
        f"criterion 2: an engine that answers at {ready_at:g} s is well inside its "
        f"{COLD_START_BUDGET_S:g} s budget, so `astra up` exits 0; it exited {code} after "
        f"{clock.t:.1f} virtual seconds of probing (budget in force: {engine_budget:g} s). "
        f"Output:\n{report}"
    )
    vllm_row = row_for(report, "vllm", services.vllm)
    assert "FAILED" not in vllm_row, f"…and the engine's row is not a failure: {vllm_row!r}"
    assert budget_in(vllm_row) >= COLD_START_BUDGET_S, (
        f"criterion 2: the budget printed is the cold-start one (>= {COLD_START_BUDGET_S:g} s), "
        f"because the printed budget is the budget honoured (ruling A14); the row said "
        f"{vllm_row!r}"
    )
    assert budget_in(vllm_row) == engine_budget, (
        f"…and it is supervisor.probe_budget_s('vllm') = {engine_budget:g}, not a second number "
        f"kept in the CLI; the row said {vllm_row!r}"
    )
    assert clock.t >= ready_at, (
        f"the engine only answered at {ready_at:g} s, so the loop waited at least that long; it "
        f"stopped at {clock.t:.2f} virtual seconds"
    )


def test_a_failed_row_carries_the_last_line_of_the_service_log(monkeypatch, tmp_path, capsys):
    """Criterion 3: an exec failure is named, not reported as "connection refused".

    On a box where ``services/tts.sh`` cannot exec, the probe can only ever say the socket is not
    there; the reason is in ``logs/tts.log``. The row must carry that log's **last non-empty**
    line — the log here ends in blank lines and a trailing newline, which is what a killed
    process leaves, and an implementation that takes ``splitlines()[-1]`` returns "" for it.
    """
    logs = tmp_path / "logs"
    use_log_dir(monkeypatch, logs)
    (logs / "tts.log").write_text(f"starting pocket-tts on cores 2,3\n{EXEC_FAILURE}\n\n\n")
    (logs / "vllm.log").write_text("INFO 09-15 23:01 api_server.py Application startup complete\n")
    (logs / "stt.log").write_text("INFO worker ready on 127.0.0.1:8080\n")
    install_clock(monkeypatch)
    probe = Probe(
        failures={"tts": "[Errno 111] Connection refused (127.0.0.1:8081)"},
        starting=("vllm", "stt"),
    )
    code = run_cli(["up", "--profile", "lean"], runner=LoggingRunner(logs), probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code != 0, f"tts never answered, so the verb fails; it exited {code}.\n{report}"
    tts_row = row_for(report, "tts", services.tts)
    assert "FAILED" in tts_row, f"criterion 3: the tts row is a failure; it said {tts_row!r}"
    assert tail_near(report, tts_row, EXEC_FAILURE), (
        "criterion 3: a FAILED row carries the last non-empty line of logs/<service>.log, so the "
        f"exec failure is named. logs/tts.log ends with {EXEC_FAILURE!r} (then blank lines); the "
        f"report said:\n{report}"
    )
    assert "Application startup complete" not in report, (
        "criterion 3: only a FAILED row carries a log tail — pasting the engine's log into a "
        f"healthy row turns the table into a log viewer. The report said:\n{report}"
    )
    assert "starting pocket-tts on cores 2,3" not in report, (
        "criterion 3: the *last* non-empty line, not the whole log; the report said:\n" + report
    )


def test_a_failed_row_without_a_readable_log_still_reports_the_probe(monkeypatch, tmp_path, capsys):
    """Adversarial (criterion 3): a missing log and an empty log must not cost the report.

    Three failures at once: the engine has a log with a real last line, ``stt``'s log is nothing
    but blank lines, and ``tts`` has no log file at all (its script died before the first write,
    which is precisely what an exec failure looks like when the shell redirects but never runs).
    Reading the tail must not raise, must not swallow the probe's own reason, and must not make
    up a line for the services that have none.
    """
    logs = tmp_path / "logs"
    use_log_dir(monkeypatch, logs)
    engine_line = "ERROR EngineCore failed to start: CUDA error: out of memory"
    (logs / "vllm.log").write_text(f"INFO starting engine\n{engine_line}\n")
    (logs / "stt.log").write_text("\n \n\n")
    assert not (logs / "tts.log").exists()
    install_clock(monkeypatch)
    refusals = {
        "vllm": "[Errno 111] Connection refused (127.0.0.1:8000)",
        "stt": "[Errno 111] Connection refused (127.0.0.1:8080)",
        "tts": "[Errno 111] Connection refused (127.0.0.1:8081)",
    }
    probe = Probe(failures=refusals)
    code = run_cli(["up", "--profile", "lean"], runner=LoggingRunner(logs), probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code == 1, f"nothing answered, so `astra up` exits 1; it exited {code}.\n{report}"
    assert "Traceback" not in report, (
        f"criterion 3: a missing or empty log is an ordinary case, never an exception:\n{report}"
    )
    vllm_row = row_for(report, "vllm", services.vllm)
    assert tail_near(report, vllm_row, engine_line), (
        f"criterion 3: the engine's FAILED row carries its log's last line; the report was:\n"
        f"{report}"
    )
    for name, url in (("stt", services.stt), ("tts", services.tts)):
        row = row_for(report, name, url)
        assert "FAILED" in row, f"criterion 3: {name} is FAILED; the row said {row!r}"
        assert refusals[name] in row, (
            f"criterion 3: with no log to quote, {name}'s row still carries what the probe saw "
            f"({refusals[name]!r}); the row said {row!r}"
        )
        assert not tail_near(report, row, engine_line), (
            f"criterion 3: …and never another service's log line — neither appended to the row "
            f"nor printed as the tail line under it (which is where a tail may live, see "
            f"tail_near); {name}'s row said {row!r} in the report:\n{report}"
        )


def test_status_names_the_exec_failure_from_the_log_too(monkeypatch, tmp_path, capsys):
    """Criterion 3 read together with T-0033 criterion 14: one row renderer, both verbs.

    ``astra status`` holds no handle from a runner — it never started anything — so a tail that
    only exists because ``up`` remembered where it sent the output is a tail the human never
    sees when they ask again a minute later. The log directory is the seam.
    """
    logs = tmp_path / "logs"
    use_log_dir(monkeypatch, logs)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)  # the autouse isolation fixture already made it (ruling A8)
    supervisor_mod = rig.module("supervisor")
    monkeypatch.setattr(supervisor_mod, "run_dir", lambda: run_dir)
    (logs / "tts.log").write_text(f"starting pocket-tts\n{EXEC_FAILURE}\n")
    install_clock(monkeypatch)
    probe = Probe(failures={"tts": "[Errno 111] Connection refused (127.0.0.1:8081)"})
    code = run_cli(["status", "--profile", "lean"], probe=probe)
    printed = capsys.readouterr()
    report = printed.out + printed.err
    services = load(None).profiles["lean"].services

    assert code != 0, f"tts is unhealthy, so `astra status` exits non-zero; it exited {code}"
    tts_row = row_for(report, "tts", services.tts)
    assert tail_near(report, tts_row, EXEC_FAILURE), (
        "criterion 3 + T-0033 criterion 14: `up` and `status` print the same rows, so the log's "
        f"last line is in status's FAILED row too; it said:\n{report}"
    )
