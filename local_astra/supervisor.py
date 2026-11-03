"""Detached service starts and the *real* health probes (T-0033 criteria 3, 10-13).

Three things live here, shared by ``astra doctor``, ``astra up``, ``astra status`` and
``astra down``:

- :func:`check_service` — the service's own health endpoint, not a TCP connect. A socket that
  accepts the connection and answers 503, or a live process whose ASR route is not mounted, are
  the two silent failures a connect calls healthy (T-0033 P1-6). Every probe is a bare ``GET``
  with **no request body** and never touches an inference endpoint, so polling costs nothing
  (``.claude/rules/gpu-services.md``: status polls are free).
- :func:`spawn` — start one ``services/*.sh`` **detached** (its own session), stdout+stderr to
  ``logs/<service>.log``, and a pid file written *after* the spawn succeeded holding the pid and
  the argv that produced it. Every script ends in a foreground server, so a blocking start is a
  command that never returns (T-0033 P0-1).
- :func:`stop_recorded` — terminate exactly what was started: the recorded pid, and only when
  ``/proc/<pid>/cmdline`` still matches the recorded argv. A pid that has been reused by an
  unrelated process is a stale file, never a victim.

The probe URL is derived from the parsed ``config.Services`` value and re-validated against
``config.LOOPBACK_PREFIXES``: no URL is typed in this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from local_astra.config import LOOPBACK_PREFIXES
from local_astra.service_state import RECORD_VERSION, Identity, StateRecord

REPO_ROOT = Path(__file__).resolve().parents[1]

#: One probe attempt. A loopback service either answers quickly or is not up yet.
PROBE_TIMEOUT_S = 2.0
#: How long ``up``/``status`` keep asking before calling a service unhealthy. The engine's init
#: is ~68 s (docs/02), but a *cold* start is the whole story: on this box after a reboot it was
#: ready at ~130 s (69 s init incl. 33 s compilation, plus container start and a cold page cache;
#: n=1, 2026-09-15, ``logs/vllm.log``), so 120 s printed FAILED seconds before it answered. 240 s
#: leaves room for a colder one (T-0041 criterion 2, ruling A1: a dead engine therefore costs four
#: minutes to report — ``--timeout`` is the explicit cap for the impatient). The speech services
#: keep 20 s: they answer in seconds, and a long budget only delays the report of a dead one.
PROBE_BUDGETS_S = {"vllm": 240.0, "stt": 20.0, "tts": 20.0, "talker": 20.0}
DEFAULT_BUDGET_S = 20.0

#: moshi-server has no health route; the ASR path answers a failed WebSocket upgrade (a 4xx),
#: which is what proves the route is mounted. The token is the published public one.
STT_AUTH_ID = "public_token"

#: How much of a health body — or of a quoted log line (:func:`log_tail`) — is kept for the report.
BODY_LIMIT = 512
#: How much of the end of a service log :func:`log_tail` reads to find its last line.
LOG_TAIL_BYTES = 65536

#: How long ``stop`` waits for a signalled process to actually go away before saying so.
#: Docker's own stop grace is 10 s, and the child ``up`` spawned is the ``docker run`` front end,
#: so anything shorter reports "still running" for a container that is shutting down normally.
STOP_WAIT_S = 12.0

#: How long a start watches the launcher before believing it (T-0049 D2, Regime). Every
#: ``services/*.sh`` ends in ``exec`` on a foreground server, so an exit inside this window — with
#: *any* status — is a failed start. If a launcher ever daemonises, this becomes a false FAILED
#: (T-0049 ambiguity 4): say so there rather than lengthening the window silently.
SETTLE_S = 1.0
#: The settle window is polled, never slept through: a launcher that dies in 40 ms is reported in
#: 40 ms, and the happy path still returns inside T-0033 criterion 10's bound.
SETTLE_POLL_S = 0.05
#: How often the health probe is repeated while waiting out a service's budget.
PROBE_POLL_S = 0.25
#: How much of the end of a failing launcher's log a FAILED row quotes. **Not** ``log_tail``:
#: docker's conflict message is followed by a blank line and a ``--help`` trailer, so the last
#: non-empty line explains nothing (T-0049 ruling A4). ``log_tail`` keeps its single-line contract
#: for T-0041's callers; this is its multi-line sibling.
FAILURE_LOG_LINES = 5

#: What the docker socket says when the login session predates the docker group — the normal state
#: on this box, and the reason ``services/stt.sh`` re-runs itself under ``sg docker -c``.
DOCKER_DENIED = "permission denied"
#: The client binary and the two verbs this module is allowed to use (T-0158 criterion 7, which
#: amends T-0049 criterion 6 / ruling A15). ``inspect`` is the question asked about a container this
#: CLI recorded itself; ``stop`` is used in exactly one function, :func:`_docker_stop`, and only on
#: the container **id** read from this CLI's own state record — never a name, never a listing.
DOCKER_BIN = "docker"
DOCKER_READ_VERB = "inspect"
DOCKER_STOP_VERB = "stop"
#: How long one ``docker inspect`` may take before it is treated as "cannot say".
DOCKER_TIMEOUT_S = 5.0
#: The grace ``docker stop --time`` gives the container's PID 1 between its SIGTERM and the daemon's
#: SIGKILL inside the container — docker's own default. Read at call time; the client call itself
#: may take this plus :data:`DOCKER_TIMEOUT_S` before it is treated as "cannot say" (T-0158).
DOCKER_STOP_GRACE_S = 10.0
#: How often the verifying ``inspect`` is repeated while a stopped ``--rm`` container is removed.
DOCKER_GONE_POLL_S = 0.1

#: A full container id as the daemon reports ``Id``. The only shape :func:`_stop_container` will
#: hand to ``docker stop`` (T-0158 hardening): never a name, never a prefix.
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")

#: The daemon's ``.State.StartedAt``: RFC 3339, nine fractional digits, ``Z``. Parsed rather than
#: compared as text (:func:`daemon_time`).
_DAEMON_STAMP = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<offset>Z|z|[+-]\d{2}:?\d{2})?$"
)

#: A containerised launcher, in **both** spellings that exist in ``services/``: ``stt.sh`` names the
#: client literally, ``vllm.sh`` goes through ``$DOCKER`` so ``ASTRA_DRY_RUN=1`` can swap in
#: ``printf``. Matching only the literal spelling would call the one service whose container cannot
#: be named "not containerised", and its ``container=unknown`` limitation would never be printed
#: (T-0049 ruling A5).
_CONTAINER_LAUNCH = re.compile(
    r"(?:(?<![\w/.-])" + DOCKER_BIN + r"|\$\{?DOCKER\}?)\s+" + r"run(?![\w-])"
)
#: The ``--name`` of that launch, when it passes one. Read from the script so that T-0048's
#: ``--name astra-vllm`` lands here with no change (T-0049 ambiguity 3).
_CONTAINER_NAME = re.compile(r"--name[=\s]+([^\s\\\"']+)")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class ProbeResult:
    """What one health probe saw. ``reason`` is ``None`` iff ``healthy``."""

    name: str
    url: str
    status_code: int | None
    body: str
    healthy: bool
    reason: str | None


@dataclass(frozen=True)
class StopOutcome:
    """What one ``stop`` did. ``still_running`` is the only outcome ``down`` exits non-zero on.

    ``stopped=False`` covers three harmless cases too — a stale file, a reused pid and an
    unreadable file — where there was never anything of ours to stop.
    """

    name: str
    stopped: bool
    still_running: bool
    message: str


@dataclass(frozen=True)
class Handle:
    """A started service: the pid, the argv that produced it and where its evidence went.

    ``process`` is the ``Popen`` of the launcher, kept so that :meth:`returncode` can be asked.
    ``start_new_session=True`` detaches the *controlling terminal*, not parenthood: the CLI is
    still the launcher's parent and its exit status is available for free. Dropping the ``Popen``
    and keeping only the pid is what made ``cli.up``'s failure check unreachable on 2026-09-16
    (T-0049 ruling A1), so the one honest, un-fakeable failure signal was thrown away.
    """

    name: str
    pid: int
    argv: tuple[str, ...]
    log_path: Path
    pid_file: Path
    process: subprocess.Popen[bytes] | None = None

    def returncode(self) -> int | None:
        """The launcher's own exit status, or ``None`` while it is still running.

        ``None`` for a handle rebuilt from a pid file too: a process this one did not father has
        no exit status to collect, and guessing one from liveness would call every healthy start
        a failure.
        """
        if self.process is None:
            return None
        return self.process.poll()


# --- the health endpoints ----------------------------------------------------------------------


def probe_url(name: str, url: str) -> str:
    """The URL that would be probed for ``name``, derived purely from the configured one.

    ``vllm`` and ``tts`` answer ``GET /health``; ``stt`` is asked for the http form of the
    configured ``ws://…/api/asr-streaming`` with ``auth_id``, path and all. Non-loopback in,
    ``ValueError`` out — a probe is the one place a typo would reach off the box.
    """
    parts = urlsplit(url)
    scheme = "http" if parts.scheme in ("ws", "http") else parts.scheme
    if parts.hostname is None:
        raise ValueError(f"{name}: {url!r} has no host")
    if name == "stt":
        query = urlencode({"auth_id": STT_AUTH_ID})
        probed = urlunsplit((scheme, parts.netloc, parts.path, query, ""))
    else:
        probed = urlunsplit((scheme, parts.netloc, "/health", "", ""))
    if not probed.startswith(LOOPBACK_PREFIXES):
        raise ValueError(f"{name}: refusing to probe non-loopback {probed!r} (docs/02 §9)")
    return probed


def probe_budget_s(name: str) -> float:
    """Per-service probe budget, printed by ``up``/``status`` beside the row (criterion 12)."""
    return PROBE_BUDGETS_S.get(name, DEFAULT_BUDGET_S)


def _healthy(name: str, status: int, body: str) -> str | None:
    """The per-service rule, as the services really behave (criterion 3)."""
    if name == "stt":
        if status == 404:
            return "HTTP 404: a live process with no ASR route mounted"
        if not 400 <= status < 500:
            return f"HTTP {status}: the ASR path did not answer a failed WebSocket upgrade (4xx)"
        return None
    if status != 200:
        return f"HTTP {status}" + (f" {body.strip()[:200]}" if body.strip() else "")
    if name == "tts" and "healthy" not in body:
        return f'HTTP 200 but the health body does not say "healthy": {body.strip()[:200]!r}'
    return None


def check_service(name: str, url: str, *, timeout_s: float = PROBE_TIMEOUT_S) -> ProbeResult:
    """Ask ``name`` its own health question. Never raises for an unhealthy service."""
    probed = probe_url(name, url)
    request = urllib.request.Request(probed, method="GET")
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            status = int(response.status)
            body = response.read(BODY_LIMIT).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # a status is an answer, not a failure
        status = int(exc.code)
        body = exc.read(BODY_LIMIT).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ProbeResult(name, probed, None, "", False, f"{exc} ({probed})")
    unhealthy = _healthy(name, status, body)
    reason = None if unhealthy is None else f"{unhealthy} from {probed}"
    return ProbeResult(name, probed, status, body, reason is None, reason)


def probe(name: str, url: str) -> str | None:
    """The CLI's ``Probe`` seam: ``None`` when healthy, else the failing reason."""
    return check_service(name, url).reason


# --- where the evidence goes ---------------------------------------------------------------------


def _git_common_dir() -> Path:
    """``git rev-parse --git-common-dir``: the one state directory all worktrees share."""
    done = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    raw = done.stdout.strip()
    if done.returncode != 0 or not raw:
        return REPO_ROOT / ".git"
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def run_dir() -> Path:
    """Where the pid files live. Never under ``artifacts/`` (criterion 11)."""
    return _git_common_dir() / "local_astra" / "run"


def log_dir() -> Path:
    """``logs/`` at the top of the repo — already gitignored."""
    return REPO_ROOT / "logs"


def log_tail(name: str) -> str | None:
    """The last non-empty line of ``logs/<name>.log``, or ``None`` when there is none.

    What a FAILED row needs to say *why* (T-0041 criterion 3): a probe can only ever report that
    the socket is not there, while ``logs/tts.log`` held the real reason —
    ``taskset: failed to execute pocket-tts``. Never raises: a service that died before its first
    write leaves no file at all, and a log of nothing but blank lines is the same "no tail" case,
    so both are ``None`` rather than an exception or an empty quotation. The **last non-empty**
    line, because a killed process leaves trailing newlines that make ``splitlines()[-1]`` empty.

    Only the end of the file is read: a service log grows without bound (the engine's is
    megabytes here) and one line is all that is wanted.
    """
    path = log_dir() / f"{name}.log"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - LOG_TAIL_BYTES))
            raw = handle.read()
    except OSError:
        return None
    for line in reversed(raw.decode("utf-8", "replace").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:BODY_LIMIT]
    return None


def start_time_of(pid: int) -> int | None:
    """``/proc/<pid>/stat`` field 22 (starttime, clock ticks since boot), or ``None`` if gone.

    Field 2 is the executable name in parentheses and may itself contain spaces and ``)``, so the
    fields are counted from the **last** ``)``: field 22 is the 20th token after it. This is the
    identity test ``down`` uses (ruling A11): a pid the kernel has handed to somebody else always
    carries a different start time, while a service that ``exec``d keeps its own.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        return int(stat[stat.rindex(")") + 1 :].split()[19])
    except (ValueError, IndexError):
        return None


def write_pid_file(path: Path, pid: int, argv: Iterable[object], start_time: int | None) -> None:
    """The pid, the argv that produced it, and the start time that proves it is still that one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": int(pid),
        "argv": [str(part) for part in argv],
        "start_time": None if start_time is None else int(start_time),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def read_pid_file(path: Path) -> tuple[int, tuple[str, ...], int | None]:
    """``(pid, argv, start_time)`` — the shape ruling A11 pins."""
    record = json.loads(Path(path).read_text())
    start_time = record.get("start_time")
    return (
        int(record["pid"]),
        tuple(str(part) for part in record["argv"]),
        None if start_time is None else int(start_time),
    )


def spawn(
    name: str,
    command: Iterable[object],
    *,
    log_dir: Path | None = None,
    run_dir: Path | None = None,
    record: bool = False,
) -> Handle:
    """Start ``command`` detached, log it, then record it. Raises ``OSError`` if it never started.

    ``start_new_session=True`` puts the child in its own session, so Ctrl-C on the CLI does not
    reach the engine it just started, and the CLI returning does not terminate it.

    ``record`` defaults to **False** (T-0049 ruling A16): a file written on spawn is what
    destroyed the provenance of the still-serving containers on 2026-09-16 and left ``astra down``
    unable to stop them, so the only writer on a start path is :func:`start_service`, *after* it
    has verified the start (criterion 5). The keyword survives because the T-0033 row about the
    file this write produces still has to be able to ask for one; nothing in ``cli.py`` does.
    """
    argv = tuple(str(part) for part in command)
    # The parameters shadow the module functions, and the defaults must be read *at call time*
    # (tests and the CLI redirect the run/log directories by patching this module).
    this = sys.modules[__name__]
    logs = Path(log_dir) if log_dir is not None else this.log_dir()
    runs = Path(run_dir) if run_dir is not None else this.run_dir()
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    pid_file = runs / f"{name}.pid"
    handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - owned by the detached child
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
    finally:
        handle.close()
    if record:
        write_pid_file(pid_file, process.pid, argv, start_time_of(process.pid))
    return Handle(
        name=name,
        pid=process.pid,
        argv=argv,
        log_path=log_path,
        pid_file=pid_file,
        process=process,
    )


# --- stopping exactly what was started -----------------------------------------------------------


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmdline_of(pid: int) -> tuple[str, ...]:
    """``/proc/<pid>/cmdline`` as argv, or an empty tuple if the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ()
    return tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def argv_matches(recorded: tuple[str, ...], live: tuple[str, ...]) -> bool:
    """Does the live command line still look like the one we recorded?

    Not the identity test — ``services/tts.sh`` ends in ``exec taskset …``, so a service that is
    very much ours stops matching the moment it execs (ruling A11). It only decides whether the
    stop is reported as an ordinary one or as "it exec'd".
    """
    if not recorded or not live:
        return False
    if recorded == live:
        return True
    shorter, longer = sorted((recorded, live), key=len)
    return longer[: len(shorter)] == shorter


def _await_exit(pid: int, timeout_s: float | None = None) -> bool:
    """Wait briefly for a signalled process to be gone, reaping it if it is our own child.

    Without the ``waitpid`` a child of *this* process stays a zombie after ``SIGTERM``, and a
    zombie still answers ``kill(pid, 0)`` — so a caller checking liveness would be told the
    service is still running. A process we did not father is reaped by init instead
    (``ChildProcessError``), which is the production case: ``down`` runs in a later process than
    the one that spawned the service.
    """
    # Read at call time, never bound as a default: a default argument freezes the constant at
    # import and a caller (or a test) that shortens it would be ignored.
    budget = sys.modules[__name__].STOP_WAIT_S if timeout_s is None else timeout_s
    deadline = time.monotonic() + budget
    while True:
        try:
            reaped, _status = os.waitpid(pid, os.WNOHANG)
            if reaped == pid:
                return True
        except ChildProcessError:
            pass  # not our child
        except OSError:
            return True
        if not is_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def stop(source: Path | str | Handle, *, sig: int = signal.SIGTERM) -> StopOutcome:
    """Stop what one pid file records.

    The rule (criterion 13, ruling A11): terminate **iff** the recorded start time matches
    ``/proc/<pid>/stat`` field 22. A different start time means the number has been handed to a
    stranger, and a stranger is never a victim; an argv that no longer matches under the *same*
    start time is a service that exec'd, which is said out loud and stopped anyway.

    The pid file is removed in every branch except one: a process that is still alive
    ``STOP_WAIT_S`` after the signal keeps its file, because the thing it names is still running
    and a later ``down`` must still be able to find it.

    A record of kind ``container`` (T-0158) is stopped **through the daemon** instead, by the
    container id it recorded (:func:`_stop_container`): signalling the ``docker run`` front end only
    reaches the container if its PID 1 handles SIGTERM, and moshi-server's does not. Process
    records and T-0033 ``{pid, argv, start_time}`` files keep the path below unchanged.

    The supervisor never sends ``SIGKILL`` to a launcher pid, on any path (ruling A16): the child
    is a ``docker run`` front end, and killing it would orphan the container it proxies signals
    to. The only ``SIGKILL`` anywhere is the daemon's own, inside the container, after
    :data:`DOCKER_STOP_GRACE_S` — that is what ``docker stop`` does.
    """
    pid_file = Path(source.pid_file) if isinstance(source, Handle) else Path(source)
    name = pid_file.stem
    try:
        pid, recorded, recorded_start = read_pid_file(pid_file)
    except (OSError, ValueError, KeyError) as exc:
        pid_file.unlink(missing_ok=True)
        return StopOutcome(name, False, False, f"{name}: unreadable pid file removed ({exc})")
    state_record = read_state_record(pid_file)
    if state_record is not None and state_record.kind == "container" and state_record.container_id:
        return _stop_container(name, pid_file, pid, recorded_start, state_record.container_id)
    if not is_alive(pid):
        pid_file.unlink(missing_ok=True)
        return StopOutcome(
            name,
            False,
            False,
            f"{name}: stale pid file (pid {pid} is gone); killed nothing, removed the file",
        )
    live_start = start_time_of(pid)
    if recorded_start is None or live_start != recorded_start:
        pid_file.unlink(missing_ok=True)
        return StopOutcome(
            name,
            False,
            False,
            f"{name}: pid {pid} was reused — the recorded start time {recorded_start} does not "
            f"match /proc/{pid}/stat field 22 ({live_start}); killed nothing, removed the file",
        )
    live = cmdline_of(pid)
    note = ""
    if not argv_matches(recorded, live):
        note = (
            f" (the argv changed: it was started as {' '.join(recorded)} and exec'd into "
            f"{' '.join(live)}; the start time says it is the same process)"
        )
    try:
        os.kill(pid, sig)
    except OSError as exc:
        pid_file.unlink(missing_ok=True)
        return StopOutcome(
            name, False, False, f"{name}: could not signal pid {pid} ({exc}); removed the file"
        )
    if not _await_exit(pid):
        return StopOutcome(
            name,
            False,
            True,
            f"{name}: signalled pid {pid}; still running after "
            f"{sys.modules[__name__].STOP_WAIT_S:g} s — not removed{note}",
        )
    pid_file.unlink(missing_ok=True)
    return StopOutcome(name, True, False, f"{name}: stopped pid {pid} ({' '.join(recorded)}){note}")


def _stop_container(
    name: str, pid_file: Path, pid: int, recorded_start: int | None, container_id: str
) -> StopOutcome:
    """Stop one recorded container through the daemon, by the id this CLI recorded (T-0158).

    Two separate permissions. The A11 start-time check decides whether the launcher *pid* may be
    signalled; the recorded container id decides whether the *container* may be stopped. So a
    launcher that is gone or whose pid was reused (the orphan shape: the third ``down`` of
    2026-10-26) is never signalled, but the container it fronted is still asked about and stopped.

    The record goes only on the daemon's answer that the container is gone or not running. When
    the daemon cannot be asked — the client cannot run, it times out, or the socket is denied even
    under ``sg`` — the record stays and the outcome is ``still_running`` (service-lifecycle rule
    2: "could not ask" is not "not there"). Never a fallback to the name, a listing or ``SIGKILL``.
    """
    if _CONTAINER_ID.fullmatch(container_id) is None:
        # Only a full daemon id is a handle this CLI recorded; a name (or anything else) may be
        # held by a stranger by now. Refused before any daemon call or signal, record kept. No
        # later `down` can do better with this record, so the message says what to do by hand.
        return StopOutcome(
            name,
            False,
            True,
            f"{name}: the recorded id {container_id!r} is not a container id (64 lowercase hex "
            f"characters), so the daemon was asked nothing and nothing was signalled (launcher "
            f"pid {pid}); {pid_file} not removed. Check by hand whether the container is "
            f"running and stop it yourself if it is, then delete {pid_file}",
        )
    alive = is_alive(pid)
    live_start = start_time_of(pid) if alive else None
    ours = alive and recorded_start is not None and live_start == recorded_start
    if ours:
        launcher = f"launcher pid {pid}"
    elif not alive:
        launcher = f"launcher pid {pid} is gone, not signalled"
    else:
        launcher = (
            f"launcher pid {pid} was reused — the recorded start time {recorded_start} does not "
            f"match /proc/{pid}/stat field 22 ({live_start}); not signalled"
        )

    def could_not_ask(what: str, why: str) -> StopOutcome:
        return StopOutcome(
            name,
            False,
            True,
            f"{name}: the daemon could not be asked to {what} container {container_id} ({why}); "
            f"{launcher}; still running as far as this CLI can tell — not removed",
        )

    if not ours:
        state, why = _container_state(container_id)
        if state is None:
            return could_not_ask("report on", why or "no answer")
        if state != _RUNNING:
            pid_file.unlink(missing_ok=True)
            return StopOutcome(
                name,
                False,
                False,
                f"{name}: stale pid file ({launcher}); container {container_id} is gone ({state}); "
                "stopped nothing, removed the file",
            )

    refused = _docker_stop(container_id)
    if refused is not None:
        return could_not_ask("stop", refused)
    state, why = _await_container_down(container_id)
    if state is None:
        return could_not_ask("confirm the stop of", why or "no answer")
    if state == _RUNNING:
        return StopOutcome(
            name,
            False,
            True,
            f"{name}: asked the daemon to stop container {container_id}; still running after "
            f"{DOCKER_STOP_GRACE_S:g} s grace — not removed ({launcher})",
        )

    if ours and not _await_exit(pid):
        # The container is down but its front end has not noticed. One SIGTERM, only to the
        # launcher whose start time still matches — never SIGKILL (ruling A16, T-0158 criterion 4).
        # An OSError means it went in between, or is not ours to signal: the wait below decides.
        if start_time_of(pid) == recorded_start:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
        if not _await_exit(pid):
            return StopOutcome(
                name,
                False,
                True,
                f"{name}: stopped container {container_id} through the daemon, but launcher pid "
                f"{pid} is still running {sys.modules[__name__].STOP_WAIT_S:g} s after one "
                "SIGTERM — not removed",
            )
    pid_file.unlink(missing_ok=True)
    return StopOutcome(
        name,
        True,
        False,
        f"{name}: stopped container {container_id} through the daemon ({launcher}"
        f"{' exited' if ours else ''})",
    )


def pid_files(directory: Path | None = None) -> tuple[Path, ...]:
    """Every recorded service, in a stable order. A missing directory is no services."""
    where = Path(directory) if directory is not None else run_dir()
    if not where.is_dir():
        return ()
    return tuple(sorted(where.glob("*.pid")))


# --- what a service script starts, and what it calls it -------------------------------------------


def script_digest(script: Path | str) -> str:
    """sha256 over the service script's bytes — cheap, and what makes drift detectable.

    Its honest limit (T-0049 Implementation notes): it catches a changed *script*, never a changed
    environment. The plugin toggle and the speech-device override that ``services/*.sh`` read (not
    named here: T-0033 criterion 20 keeps their names out of this package) select different engines
    from the same bytes, so an unchanged digest is not a proof that the configuration is the same.
    An unreadable script digests to ``""``, which :func:`local_astra.service_state.decide` treats
    as "cannot be compared", never as "unchanged".
    """
    try:
        return hashlib.sha256(Path(script).read_bytes()).hexdigest()
    except OSError:
        return ""


def _launch_line(script: Path | str) -> str | None:
    """The logical command line on which ``script`` starts a containerised server, or ``None``.

    Continuations are joined, because the flag that names the container sits several lines below
    the verb in ``services/stt.sh``; comment lines are skipped, because that file *documents* the
    ``--name`` in one (and a detector that read the comment would call the name ``so``).
    """
    try:
        lines = Path(script).read_text(errors="replace").splitlines()
    except OSError:
        return None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#") or not _CONTAINER_LAUNCH.search(line):
            continue
        logical = [line]
        cursor = index
        while logical[-1].rstrip().endswith("\\") and cursor + 1 < len(lines):
            cursor += 1
            logical.append(lines[cursor])
        return " ".join(part.rstrip().rstrip("\\") for part in logical)
    return None


def script_is_containerised(script: Path | str) -> bool:
    """Does this script's foreground server run inside a container?

    True for both spellings in ``services/``: the client named literally (``stt.sh``) and the one
    reached through ``$DOCKER`` so a dry run can substitute ``printf`` (``vllm.sh``). Matching only
    the literal one would report the single service whose container cannot be named as not
    containerised at all, and its limitation would never be printed (T-0049 ruling A5).
    """
    return _launch_line(script) is not None


def script_container(script: Path | str) -> str | None:
    """The container name that script passes, or ``None`` when it passes none.

    Read from the script rather than from a table in this module, so that T-0048's
    ``--name astra-vllm`` starts being used here the day it lands, with no change (ambiguity 3).
    """
    line = _launch_line(script)
    if line is None:
        return None
    found = _CONTAINER_NAME.search(line)
    return found.group(1) if found else None


# --- asking the right authority about a recorded handle -------------------------------------------


def _docker_call(
    command: list[str], *, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str] | None:
    """Run one client command, or ``None`` when it could not be run at all (or timed out)."""
    timeout = DOCKER_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _docker_read(ref: str) -> tuple[dict[str, Any], None] | tuple[None, str]:
    """Read one container by the reference this CLI recorded: ``(entry, None)`` or ``(None, why)``.

    Read-only by construction. It never starts, stops, removes or renames anything, and identity
    is never established by listing containers and grepping — that would claim instances this CLI
    did not start (criterion 6). The module's one other daemon verb is the ``stop`` in
    :func:`_docker_stop`, which ``down`` aims at a container id from its own record (T-0158).

    The ``sg`` retry is a *deliberate* duplicate of ``preflight._docker``: that function is private
    and ``preflight.py`` belongs to another task (T-0049 Implementation notes), so the two copies
    are pinned by a test instead of shared. Consolidating them is a later task.
    """
    command = [DOCKER_BIN, DOCKER_READ_VERB, ref]
    done = _docker_call(command)
    if done is None:
        return None, f"{DOCKER_BIN} could not be run"
    if done.returncode != 0 and DOCKER_DENIED in done.stderr.lower():
        # The login session on this box predates the docker group; the same command under
        # `sg docker -c` works, which is the fallback services/stt.sh documents.
        retried = _docker_call(["sg", DOCKER_BIN, "-c", shlex.join(command)])
        if retried is not None:
            done = retried
    if done.returncode != 0:
        detail = done.stderr.strip()[:BODY_LIMIT] or f"exited {done.returncode}"
        return None, detail
    try:
        data = json.loads(done.stdout)
    except ValueError as exc:
        return None, f"unreadable {DOCKER_READ_VERB} output ({exc})"
    entry = data[0] if isinstance(data, list) and data else data
    if not isinstance(entry, dict):
        return None, f"unexpected {DOCKER_READ_VERB} output for {ref}"
    return entry, None


def read_container(ref: str) -> tuple[Identity | None, str | None]:
    """Ask the daemon about one recorded container: ``(identity, None)``, or ``(None, why)``.

    The tri-state ruling A13 is about. ``(None, why)`` is **unknown** — the daemon could not be
    asked at all: it was busy past :data:`DOCKER_TIMEOUT_S`, the client could not be run, or the
    socket answered ``permission denied`` and the group retry could not help either, which is the
    ordinary state of this box. That is not the same fact as an answer and may never be read as
    one: on 2026-09-16 a failed read taken for "nothing was there" would have certified a
    container that had been up two hours as this attempt's, and written its id into the record
    that ``down`` and ``--restart`` act on afterwards.
    """
    entry, why = _docker_read(ref)
    if entry is None:
        return None, why or f"the daemon could not be asked about {ref}"
    state = entry.get("State") or {}
    running = bool(state.get("Running"))
    pid = state.get("Pid")
    return (
        Identity(
            kind="container",
            alive=running,
            pid=int(pid) if isinstance(pid, int) and pid > 0 else None,
            container_id=str(entry.get("Id") or ref),
            started_at=str(state.get("StartedAt") or "") or None,
            note=None if running else f"the container is {state.get('Status') or 'not running'}",
        ),
        None,
    )


def container_identity(ref: str) -> Identity:
    """What the daemon says about one container reference, as a single :class:`Identity`.

    Never raises: an unresolvable handle is a fact about provenance (criterion 8's input), not an
    exception. A read that could not be made at all comes back ``alive=False`` **and**
    ``unknown=True``: not alive, because nothing may be called live on no evidence, and not a
    claim that it is gone either. Callers that must tell those apart use :func:`read_container`.
    """
    identity, why = read_container(ref)
    if identity is None:
        return Identity(kind="container", alive=False, container_id=ref, note=why, unknown=True)
    return identity


# --- stopping a recorded container through the daemon (T-0158) ------------------------------------

#: The daemon's two "there is no such container" answers. They count as answers **only** on the
#: stop path below; :func:`read_container` still reports "No such object" as unknown (criterion 6).
_NO_SUCH_CONTAINER = "no such container"
_NO_SUCH_OBJECT = "no such object"
#: What :func:`_container_state` calls each answer.
_RUNNING = "running"
_GONE = "no such object"
_NOT_RUNNING = "not running"


def _docker_stop(ref: str) -> str | None:
    """``docker stop --time <grace> <ref>``: ``None`` when the daemon answered, else why it could not.

    The **only** function in this module that builds a ``stop`` argv (criterion 7), and its only
    caller hands it the container id read from this CLI's own state record — never the name, which
    a stranger may hold by the time ``down`` runs. "No such container" is an answer (the verifying
    ``inspect`` decides what it means); anything else non-zero is "could not ask". The ``sg`` retry
    has the same shape as :func:`_docker_read`'s.
    """
    # docker's `--time` is an integer flag: a fractional grace is rounded up, never to 0, which
    # would be an immediate SIGKILL inside the container.
    grace = max(1, math.ceil(float(DOCKER_STOP_GRACE_S)))
    timeout = grace + float(DOCKER_TIMEOUT_S)
    command = [DOCKER_BIN, DOCKER_STOP_VERB, "--time", str(grace), ref]
    done = _docker_call(command, timeout_s=timeout)
    if done is None:
        return f"{DOCKER_BIN} could not be run, or did not answer within {timeout:g} s"
    if done.returncode != 0 and DOCKER_DENIED in done.stderr.lower():
        # The login session on this box predates the docker group; the same command under
        # `sg docker -c` works, which is the fallback services/stt.sh documents.
        retried = _docker_call(["sg", DOCKER_BIN, "-c", shlex.join(command)], timeout_s=timeout)
        if retried is not None:
            done = retried
    if done.returncode == 0 or _NO_SUCH_CONTAINER in done.stderr.lower():
        return None
    return done.stderr.strip()[:BODY_LIMIT] or f"exited {done.returncode}"


def _container_state(ref: str) -> tuple[str, None] | tuple[None, str]:
    """One ``inspect`` on the stop path: ``(running|not running|no such object, None)`` or ``(None, why)``.

    Unlike :func:`read_container`, "No such object" is an answer here — the container this CLI
    recorded is gone. Only ``State.Running`` being exactly ``true`` or ``false`` is read as an
    answer about a container that exists; anything else could not be read.
    """
    entry, why = _docker_read(ref)
    if entry is None:
        if why is not None and _NO_SUCH_OBJECT in why.lower():
            return _GONE, None
        return None, why or f"the daemon could not be asked about {ref}"
    state = entry.get("State")
    running = state.get("Running") if isinstance(state, dict) else None
    if running is True:
        return _RUNNING, None
    if running is False:
        return _NOT_RUNNING, None
    return None, f"unreadable {DOCKER_READ_VERB} State.Running for {ref}: {running!r}"


def _await_container_down(ref: str) -> tuple[str, None] | tuple[None, str]:
    """Poll the verifying ``inspect`` for up to :data:`DOCKER_TIMEOUT_S` after a ``stop``.

    ``docker stop`` returns once the container has stopped, but ``--rm`` removal can lag it; a
    ``Running == false`` answer is already enough. The last answer is returned, so a container
    still running at the deadline comes back as ``running``.
    """
    deadline = time.monotonic() + float(DOCKER_TIMEOUT_S)
    while True:
        state, why = _container_state(ref)
        if state is None:
            return None, why or f"the daemon could not be asked about {ref}"
        if state != _RUNNING or time.monotonic() >= deadline:
            return state, None
        time.sleep(DOCKER_GONE_POLL_S)


def live_identity(record: StateRecord) -> Identity:
    """Ask the authority that answers for this *kind* of handle whether it is still live.

    ``/proc`` for a host process, the daemon for a container. The record only ever says what this
    CLI started; this is the other half — whether that thing is still running — and the two
    together are what a health probe can never tell you, because a probe answers for whatever is
    on the port.
    """
    if record.kind == "container" and record.container_id:
        return container_identity(record.container_id)
    live_start = start_time_of(record.pid)
    if not is_alive(record.pid) or live_start is None:
        return Identity(
            kind="process", alive=False, pid=record.pid, note=f"pid {record.pid} is gone"
        )
    if record.start_time is not None and live_start != record.start_time:
        return Identity(
            kind="process",
            alive=False,
            pid=record.pid,
            start_time=live_start,
            note=(
                f"pid {record.pid} was reused: the recorded start time {record.start_time} does "
                f"not match /proc/{record.pid}/stat field 22 ({live_start})"
            ),
        )
    return Identity(kind="process", alive=True, pid=record.pid, start_time=live_start)


# --- the state record -----------------------------------------------------------------------------


def write_state_record(path: Path | str, record: StateRecord) -> None:
    """Write one identity record. Called **only** after a start has been verified (criterion 5)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record.as_dict(), indent=2) + "\n")


def read_state_record(path: Path | str) -> StateRecord | None:
    """The versioned record, or ``None`` for anything that is not one.

    ``None`` is the answer for the T-0033-shape files still on disk (``{pid, argv, start_time}``):
    they carry no kind and no digest, so they cannot certify provenance — which is precisely why a
    healthy service recorded that way is criterion 8's CONFLICT and not a skip. They keep being
    read by :func:`read_pid_file`, :func:`stop` and :func:`pid_files` exactly as before.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return StateRecord.from_mapping(data)
    except (TypeError, ValueError):
        return None


# --- starting a service, and proving that it started ----------------------------------------------

#: The three things a start can be. ``ok`` is true for ``started`` and for nothing else.
STARTED = "started"
FAILED = "FAILED"
NOT_NEW = "NOT-NEW"


@dataclass(frozen=True)
class StartOutcome:
    """One service's start, as the row will print it. ``ok`` iff ``status == "started"``."""

    name: str
    status: str
    ok: bool
    message: str
    handle: Handle | None = None
    record: StateRecord | None = None


def log_tail_lines(path: Path | str, *, limit: int = FAILURE_LOG_LINES) -> tuple[str, ...]:
    """The last ``limit`` non-empty lines of a service log, oldest first.

    The multi-line sibling of :func:`log_tail`, which is unchanged and keeps its single-line
    contract for T-0041's callers. It exists because the recorded launcher log of 2026-09-16 ends
    with the docker conflict message, a blank line and ``Run '… --help' for more information``: a
    row built from the last non-empty line alone would show the human the help trailer and explain
    nothing, which is the very failure ``log_tail`` was added to prevent (T-0049 ruling A4).
    """
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - LOG_TAIL_BYTES))
            raw = handle.read()
    except OSError:
        return ()
    lines = [
        line.strip()[:BODY_LIMIT]
        for line in raw.decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    return tuple(lines[-limit:]) if limit > 0 else ()


def _quoted_log(handle: Handle) -> str:
    """The explanation a failed row carries. The log only ever explains; it never detects (D2)."""
    lines = log_tail_lines(handle.log_path)
    if not lines:
        return f" Its log ({handle.log_path}) is empty."
    body = "\n    ".join(lines)
    return f" {handle.log_path} ends:\n    {body}"


def _orphan_note(handle: Handle) -> str:
    """What is said about a launcher that is still running after an unverified start."""
    if handle.returncode() is not None or not is_alive(handle.pid):
        return ""
    return (
        f" The launcher (pid {handle.pid}) is still running; it was neither recorded nor "
        "signalled, because nothing here stops what it could not verify."
    )


def _failure(name: str, handle: Handle, headline: str) -> StartOutcome:
    return StartOutcome(
        name=name,
        status=FAILED,
        ok=False,
        message=f"{name}: FAILED — {headline}.{_orphan_note(handle)}{_quoted_log(handle)}",
        handle=handle,
    )


def _watch_launcher(handle: Handle, settle_s: float) -> int | None:
    """Poll the launcher's own exit status for ``settle_s``; its code, or ``None`` if it lived.

    A poll loop rather than a flat sleep: a launcher that dies in 40 ms is reported in 40 ms.
    """
    deadline = time.monotonic() + settle_s
    while True:
        code = handle.returncode()
        if code is not None:
            return code
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return handle.returncode()
        time.sleep(min(SETTLE_POLL_S, remaining))


def daemon_time(stamp: str | None) -> float | None:
    """The daemon's ``.State.StartedAt`` as epoch seconds, or ``None`` when it cannot be read.

    Two landmines, both from ruling A20. The daemon stamps **nine** fractional digits and
    ``datetime.fromisoformat`` takes at most six on 3.11, so the fraction has to be truncated —
    and truncated to **microseconds**, never to seconds: at second resolution a container started
    in the same second as the attempt reads as *not after* it, which is a false NOT-NEW on a fast
    start, the one case where being wrong costs a start that really worked. Go also trims trailing
    zeros from the fraction, so the stamps must never be compared as text (ruling A18, which
    parsing here retires).

    A stamp that does not parse is ``None`` — unknown, not "long ago": the caller decides, and it
    refuses rather than guesses.
    """
    if not stamp:
        return None
    found = _DAEMON_STAMP.match(stamp.strip())
    if found is None:
        return None
    fraction = (found.group("fraction") or "")[:6].ljust(6, "0")
    offset = found.group("offset") or "Z"
    tail = "+00:00" if offset.upper() == "Z" else offset
    try:
        return datetime.fromisoformat(f"{found.group('head')}.{fraction}{tail}").timestamp()
    except ValueError:
        return None


def _iso(epoch_s: float) -> str:
    """One wall-clock instant in the daemon's own spelling, so a row's two times compare by eye."""
    return datetime.fromtimestamp(epoch_s, UTC).isoformat()


def _not_new_because(
    after: Identity, attempt_at: float, before: Identity | None = None
) -> str | None:
    """Why the container answering now cannot be called this attempt's — ``None`` when it can.

    D4's primary and **always-required** arm, as ruling A19 settles it: the instance's own start
    time against ``attempt_at``, the reference taken before the spawn. Both stamps come from this
    host's clock — the daemon stamps ``StartedAt`` on the same machine — so there is no skew
    argument for preferring the container id, and the id is *supporting evidence*: it is named in
    the row, never compared as the thing that decides.

    That is what closes ruling A13's hole without a special path. The pre-spawn snapshot cannot be
    trusted to exist: a read that failed (busy daemon, timeout, permission denied) comes back in
    the same shape as "nothing was there", and reading it as absence certified a two-hour-old
    container as started on 2026-09-16. A container from two hours ago precedes the attempt
    whatever the daemon was or was not able to say beforehand.

    Strictly after: an instance whose start time is not *later* than the reference is the one that
    was already there, which is exactly what a health probe certifies happily. Weakening this to
    accept an equal time re-opens the bug.

    Honest limit, unchanged: a container somebody *else* started inside the window passes this.
    Neither the clock nor the id can tell "started after my reference" from "started by me".
    """
    handle = after.container_id or "the container"
    started = daemon_time(after.started_at)
    if started is None:
        return (
            f"the container answering is {handle}, whose start time cannot be read "
            f"({after.started_at!r}), so it cannot be shown to have started after this attempt "
            f"began ({_iso(attempt_at)})"
        )
    if started > attempt_at:
        return None
    same = (
        before is not None
        and before.alive
        and before.container_id is not None
        and before.container_id == after.container_id
    )
    also = " — the same id was already answering before the spawn" if same else ""
    return (
        f"the container answering is {handle}, started {after.started_at} "
        f"({_iso(started)}), which is not after this attempt began ({_iso(attempt_at)}): it is "
        f"the instance that was already there{also}"
    )


def start_service(
    name: str,
    command: Iterable[object],
    url: str,
    *,
    probe: Callable[[str, str], str | None] | None = None,
    log_dir: Path | None = None,
    run_dir: Path | None = None,
    settle_s: float | None = None,
    budget_s: float | None = None,
) -> StartOutcome:
    """Start one service and report what actually happened: *act → verify → record*.

    The evidence is ranked by what can be fooled (D2), and **health is not evidence of a start**:
    a probe answers for whatever is on the port, which is how ``astra up`` printed ``ok`` for two
    services it had not started on 2026-09-16.

    1. the launcher's own exit status — this process is its parent, and every ``services/*.sh``
       ends in ``exec`` on a foreground server, so *any* exit inside the settle window is failure;
    2. liveness at the end of that window;
    3. health, which only says the port answers;
    4. newness — the instance's **own start time** must be later than the wall clock read before
       the spawn (D4, ruling A19). That comparison is primary and always required; the container
       id that was answering beforehand is supporting evidence for the row, never the test. A
       start whose newness cannot be established is not a start — but "the daemon could not be
       asked" makes the container's *identity* unknown, not the start failed, when this process's
       own child is alive and the service is healthy (ruling A21);
    5. and only then the state record.

    The order of 5 is the landmine: a record written on spawn is what destroyed the provenance of
    the still-serving containers on 2026-09-16 and left ``astra down`` unable to stop them. A start
    that is not verified leaves the previous record byte-identical.
    """
    this = sys.modules[__name__]
    argv = tuple(str(part) for part in command)
    script = Path(argv[0])
    digest = script_digest(script)
    container_name = script_container(script)
    limitation = ""
    if container_name is None and script_is_containerised(script):
        # The engine today: services/vllm.sh passes no --name, so docker generates one
        # (`exciting_maxwell`) and this CLI cannot ask about it. Degrade to the launcher pid and
        # print the limitation — never guess the container by image ancestry, which would claim
        # containers this CLI did not start (T-0049 ruling A1/D3).
        limitation = (
            f" · container=unknown ({script} passes no --name, so the daemon cannot be asked "
            f"which container this is; newness rests on the launcher pid — T-0048 adds "
            f"`--name astra-{name}`)"
        )
    settle = this.SETTLE_S if settle_s is None else float(settle_s)
    budget = probe_budget_s(name) if budget_s is None else float(budget_s)
    asked: Callable[[str, str], str | None] = this.probe if probe is None else probe

    # The reference point, read **before** anything is spawned (D4). The wall clock is the one
    # that decides (ruling A19): it is always available, it comes from the same host clock the
    # daemon stamps `StartedAt` with, and it is what makes a two-hour-old container NOT-NEW no
    # matter what the daemon says. The pre-existing container id is read beside it as supporting
    # evidence for the row, and its failure modes are precisely why it cannot be the test: a busy
    # daemon, the inspect timeout and `permission denied` all come back looking like "nothing was
    # there" (ruling A13).
    attempt_at = time.time()
    before = container_identity(container_name) if container_name is not None else None

    handle = spawn(name, argv, log_dir=log_dir, run_dir=run_dir, record=False)

    exited = _watch_launcher(handle, settle)
    if exited is not None:
        return _failure(
            name,
            handle,
            f"the launcher exited {exited} within the {settle:g} s settle window — every "
            "services/*.sh ends in `exec` on a foreground server, so any exit at all is a failed "
            "start, whatever the status and whatever the port says",
        )
    if not is_alive(handle.pid):
        return _failure(
            name,
            handle,
            f"the launcher (pid {handle.pid}) was gone at the end of the {settle:g} s settle "
            "window without leaving an exit status",
        )

    deadline = time.monotonic() + budget
    reason = asked(name, url)
    while reason is not None:
        code = handle.returncode()
        if code is not None:
            return _failure(
                name, handle, f"the launcher exited {code} while {name} was still not healthy"
            )
        if time.monotonic() >= deadline:
            return _failure(name, handle, f"not healthy after {budget:g} s: {reason}")
        time.sleep(PROBE_POLL_S)
        reason = asked(name, url)

    if container_name is not None:
        after, unknown = read_container(container_name)
        if after is None:
            # Ruling A13's mirror, fenced by A21: the daemon could not be asked. The start itself
            # is not in doubt — the launcher this invocation spawned is alive at the end of the
            # settle window (it is this process's own child, so its exit status could not have
            # been missed), it was spawned after the reference, and the service is healthy. What
            # is unknown is *which* container it is, and that is the situation criterion 6 already
            # answers for the engine: degrade to the launcher pid, print the limitation, guess
            # nothing. In particular the pre-spawn id is not adopted — a record naming a container
            # this invocation did not start is the provenance corruption of 2026-09-16.
            kind = "process"
            container_id = None
            limitation = (
                f" · container=unknown (the daemon could not be asked which container "
                f"{container_name} is: {unknown}), so this start is recorded against the launcher "
                f"pid alone and `astra down` will signal that rather than the container id"
            )
            identity_text = f"pid {handle.pid} (/proc start time {start_time_of(handle.pid)})"
        elif not after.alive:
            # An *answer*, and what it says is that the container this attempt was meant to
            # produce is not running. That is evidence, unlike the branch above, and the port
            # answering proves nothing about it.
            return _failure(
                name,
                handle,
                f"the container {container_name} the script names is not running after the start "
                f"({after.note}), so this start cannot be shown to have produced anything",
            )
        elif (stale := _not_new_because(after, attempt_at, before)) is not None:
            return StartOutcome(
                name=name,
                status=NOT_NEW,
                ok=False,
                message=(
                    f"{name}: NOT-NEW — {stale} ({container_name}): this start produced nothing. "
                    f"A health probe answers for whatever is on the port. Reboot it deliberately "
                    f"with `astra up --restart {name}`.{_orphan_note(handle)}"
                ),
                handle=handle,
            )
        else:
            kind = "container"
            container_id = after.container_id
            identity_text = (
                f"container {after.container_id} ({container_name}) started {after.started_at}, "
                f"launcher pid {handle.pid}"
            )
    else:
        # Ruling A6, the honest limit: this launcher is this process's own child, spawned after
        # the reference point, so its start time is later than the attempt by construction. There
        # is no way for this path to be NOT-NEW and none is invented; newness for a containerised
        # service without a name is simply unasserted until T-0048 gives the engine one.
        kind = "process"
        container_id = None
        identity_text = f"pid {handle.pid} (/proc start time {start_time_of(handle.pid)})"

    record = StateRecord(
        version=RECORD_VERSION,
        name=name,
        kind=kind,
        pid=handle.pid,
        start_time=start_time_of(handle.pid),
        argv=argv,
        script_digest=digest,
        container_name=container_name,
        container_id=container_id,
        # The wall clock of the attempt this start verified, not of the verification: the engine's
        # probe budget is four minutes, and a row that said "already running since" the end of it
        # would report the probe's age rather than the service's.
        started_at=attempt_at,
    )
    write_state_record(handle.pid_file, record)
    return StartOutcome(
        name=name,
        status=STARTED,
        ok=True,
        message=f"{name}: started — {identity_text}; healthy at {url}{limitation}",
        handle=handle,
        record=record,
    )
