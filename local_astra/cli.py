"""``astra`` command line: session, up/down/status, bench, eval, forget, fetch-models (docs/02 §10).

Two verbs do something today.

``astra session`` is the **composition root**: the one place allowed to name the concrete clients
(``tests/unit/test_session_no_egress.py`` keeps the library half free of them). It builds the
production object graph for a profile — engine, STT, TTS, playback, memory, and one
``BrowserFrameSource`` used *both* as the session's ``PerceptionClient`` and as the HUD server's
frame sink — serves ``conductor/hud_server.py`` and prints the page URL. ``--dry-run`` resolves
the profile and prints what would be built without constructing a client or opening a device.

``astra up`` decides per backend before it spawns anything — start, skip what is already running
under this CLI's own record, or refuse what it cannot place — then starts what needs starting by
running ``services/*.sh`` (the Bash guard gates those scripts; this does not re-implement them and
does not bypass it), verifies each start through ``supervisor.start_service``, and finally
health-probes each loopback endpoint until it answers or ``--timeout`` runs out. It signals
nothing and removes no state file without ``--restart`` (``.claude/rules/service-lifecycle.md``).

Every endpoint comes from ``config.Services``, whose validator refuses anything but loopback
(docs/02 §9): no verb here accepts a URL, and none of them falls back to another port, interface
or device — a session that quietly moved is a page talking to nothing. Anything that spends GPU
time or writes ``artifacts/`` goes through ``/launch`` first (``.claude/hooks/guard_bash.sh``).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import inspect
import os
import signal
import socket
import sys
import time
import tomllib
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from local_astra import __version__, hud_static, preflight, supervisor
from local_astra.conductor.audio_device import open_device
from local_astra.conductor.audio_io import PipeWireAudioIO
from local_astra.conductor.canvas_relay import CanvasRelay, endpoint_from_env
from local_astra.conductor.hud_server import HudServer
from local_astra.conductor.live_settings import LiveSettings
from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
from local_astra.conductor.null_audio import NullAudioDevice
from local_astra.conductor.prompt_builder import (
    LIVE_GUIDANCE,
    LIVE_TOOL_GUIDANCE,  # noqa: F401  (T-0174 A1: stays importable from cli, the same object)
    LIVE_TOOL_GUIDANCE_V4_NOSEARCH,
    with_wait_guidance,
)
from local_astra.conductor.session import (
    LiveBrowser,
    LiveFeedback,
    Session,
    SessionClock,
    SessionConfig,
    ToolKit,
)
from local_astra.conductor.stt_client import MoshiSttClient
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.conductor.tts_client import PocketTtsClient
from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.config import (
    ACTION_ECHOES,
    DEFAULT_ANNOUNCED_CALL_REASK,
    DEFAULT_FRAME_INTERVAL_MS,
    DEFAULT_LIVE_GUIDANCE,
    DEFAULT_MAX_FRAMES_PER_TURN,
    FRAME_INTERVALS_MS,
    FRAMES_PER_TURN_CHOICES,
    LIVE_DEFAULT_ACTION_ECHO,
    LIVE_DEFAULT_ACTION_OUTCOME,
    LIVE_DEFAULT_AUTO_OVERVIEW,
    LIVE_DEFAULT_FEEDBACK,
    LIVE_DEFAULT_OVERVIEW_FRAME,
    LIVE_DEFAULT_PAGE_OVERVIEW,
    LIVE_DEFAULT_REGROUND,
    LIVE_DEFAULT_REGROUND_VISUAL_TOKENS,
    LIVE_DEFAULT_REPEAT_GUARD,
    LIVE_DEFAULT_THOUGHT_CALL_RECOVERY,
    LIVE_DEFAULT_VISUAL_TOKENS,
    LIVE_FEEDBACK_CHOICES,
    LIVE_GUIDANCE_VERSIONS,
    SEARCH_CHOICES,
    ActionEcho,
    Config,
    LiveFeedbackChoice,
    Profile,
    Services,
    load,
    tools_refusal,
)
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import SettingSource, SettingValue
from local_astra.contracts.protocols import SearchBackend
from local_astra.memory.store import DEFAULT_DIR, SqliteMemory
from local_astra.perception.browser_source import BrowserFrameSource
from local_astra.service_state import (
    Decision,
    Observation,
    StateRecord,
    decide,
    handle_text,
)
from local_astra.telemetry.session_log import SessionLog
from local_astra.toolcall import detok, live, native, warmup
from local_astra.toolcall import tools as toolset

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = REPO_ROOT / "services"

#: ``--mic none`` and ``--no-mic`` say it, and the resolved lines print it: no capture source.
MIC_OFF = "none"
#: A microphone nobody could resolve: the graph could not be read, so the box did not say which
#: capture source is the default. ``--dry-run`` prints it (it opens nothing); a session that is
#: really starting refuses instead of running deaf (T-0045 ruling A3).
MIC_UNKNOWN = "unknown"
#: The environment variable ``audio_device.open_device`` falls back to when it is given no
#: ``source`` (T-0035 criterion 7) -- and the workaround the human was told to export on
#: 2026-09-15, so it is plausibly still set in the shell a session is started from. T-0045 ruling
#: A14 makes it an explicit, *printed* step of ``resolve_mic``'s precedence instead of an
#: invisible fallback two layers below the line that prints ``mic=``.
ENV_MIC = "ASTRA_AUDIO_SOURCE"

#: ``runner(command) -> a started process``: one ``services/*.sh`` invocation, detached.
#: T-0033 criterion 15 records the spec change: T-0028's ``-> int`` seam could not express
#: "started and still running", which is the only thing worth knowing about a foreground server.
Runner = Callable[[Sequence[str]], object]
#: ``probe(name, url) -> None when healthy, else the failing reason``.
Probe = Callable[[str, str], "str | None"]

#: How long to wait between probe rounds while a backend is still starting.
PROBE_INTERVAL_S = 0.25
#: Per-probe TCP connect budget. A loopback service either answers at once or is not up yet.
PROBE_CONNECT_TIMEOUT_S = 0.5


# --- `astra up`: the start plan and the health probes ------------------------------------------


def start_plan(profile: Profile) -> tuple[tuple[str, ...], ...]:
    """The ordered ``services/*.sh`` invocations this profile's backends need. Pure.

    The engine goes first: the speech services share its VRAM budget and its 30 s warm-up is the
    long pole. Backends that run inside the Conductor process (the CPU STT and TTS variants) have
    no script and are therefore not in the plan.
    """
    plan: list[tuple[str, ...]] = [(str(SERVICES_DIR / "vllm.sh"),)]
    if profile.stt_backend == "kyutai-gpu":
        plan.append((str(SERVICES_DIR / "stt.sh"),))
    if profile.tts_backend in ("pocket-cpu", "kyutai-1.6b-gpu"):
        plan.append((str(SERVICES_DIR / "tts.sh"),))
    return tuple(plan)


def endpoints_of(profile: Profile) -> tuple[tuple[str, str], ...]:
    """``(name, url)`` for every backend of this profile, from ``config.Services`` alone."""
    services = profile.services
    out = [("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts)]
    if services.talker is not None:
        out.append(("talker", services.talker))
    return tuple(out)


def service_name_of(command: Sequence[str]) -> str:
    """``services/stt.sh`` -> ``stt``: the name its log and pid file are filed under."""
    return Path(str(command[0])).stem


def probe_endpoint(name: str, url: str) -> str | None:
    """The default probe: can this loopback endpoint be connected to at all?

    Deliberately a TCP connect rather than a protocol-specific health call: the three backends
    speak three different protocols (OpenAI HTTP, msgpack over WebSocket, HTTP), and a connect is
    the one question all of them answer the same way. ``None`` means healthy.
    """
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port
    if host is None or port is None:
        return f"{name}: {url} has no host and port"
    try:
        with socket.create_connection((host, port), timeout=PROBE_CONNECT_TIMEOUT_S):
            return None
    except OSError as exc:
        return f"{exc} ({host}:{port})"


@dataclass(frozen=True)
class Verdict:
    """One service's row: what *this invocation* did to it, and whether that is a failure.

    The words a row can carry are ``started``, ``already running``, ``CONFLICT``, ``FAILED`` and
    ``NOT-NEW`` (criterion 10), and every one of them comes from the decision or from the verified
    start — never from the health table, which answers for whatever is on the port. ``ok`` is what
    the exit code is built from: a skip is not a failure; the other three are.
    """

    name: str
    ok: bool
    message: str


def restart_targets(spec: str | None, names: Sequence[str]) -> tuple[str, ...]:
    """Which services ``--restart`` names — ``all``, or a comma-separated list — in plan order.

    Restarting is the *only* thing this verb does that signals a running service or removes its
    state file (criterion 11, ``.claude/rules/service-lifecycle.md``), so the flag is read once,
    here, and no other branch asks any question about it.
    """
    if spec is None:
        return ()
    wanted = [part.strip() for part in str(spec).split(",") if part.strip()]
    if "all" in wanted:
        return tuple(names)
    return tuple(name for name in names if name in wanted)


def _verification_budget_s(name: str, timeout_s: float | None) -> float:
    """The budget the *start's* own health probe honours: the per-service one, capped by the flag."""
    budget = supervisor.probe_budget_s(name)
    return budget if timeout_s is None else min(budget, max(0.0, float(timeout_s)))


def _decide(name: str, command: Sequence[str], url: str, probe: Probe) -> Decision:
    """*Observe*, then decide — before anything is spawned, and not in this module.

    The table is :func:`local_astra.service_state.decide`; this only gathers the four facts it
    decides from. Keeping any of the decision here is the shape of the 2026-09-16 bug, where the
    verb ran every command in the plan unconditionally and let a health probe certify the result.
    """
    reason = probe(name, url)
    record = supervisor.read_state_record(supervisor.run_dir() / f"{name}.pid")
    return decide(
        Observation(
            name=name,
            url=url,
            healthy=reason is None,
            health_reason=reason,
            record=record,
            live=None if record is None else supervisor.live_identity(record),
            script_digest=supervisor.script_digest(command[0]),
        )
    )


def _runner_verdict(name: str, command: Sequence[str], handle: object) -> Verdict:
    """An injected runner *replaces* the start of one plan entry, so its return is that outcome.

    The seam is written out in ``tests/unit/_service_rig.py``'s group B section: a fake carrying
    ``.status``/``.ok``/``.message`` states its own outcome, and the two older shapes are still
    read — the superseded T-0028 exit code, and a bare handle with a pid.
    """
    status = getattr(handle, "status", None)
    message = getattr(handle, "message", None)
    if isinstance(status, str) and isinstance(message, str):
        return Verdict(name, bool(getattr(handle, "ok", False)), message)
    # The superseded T-0028 seam returned an exit code; a runner that still hands one back (and a
    # non-zero one) is reporting a failure, so say so rather than probing for nothing.
    if isinstance(handle, int) and not isinstance(handle, bool) and int(handle) != 0:
        return Verdict(name, False, f"{name}: FAILED — {' '.join(command)} exited {int(handle)}")
    return Verdict(name, True, f"{name}: started — {_started_note(handle, command)}")


def _start(
    name: str,
    command: Sequence[str],
    url: str,
    *,
    runner: Runner | None,
    probe: Probe,
    timeout_s: float | None,
) -> Verdict:
    """Start one plan entry and report what actually happened.

    With nothing injected this is ``supervisor.start_service``: spawn, watch the launcher's own
    exit status through the settle window, probe, prove the instance is *new*, and only then write
    the state record. Nothing in this module writes one — a record written before a start is
    verified is what destroyed the provenance of the still-serving containers on 2026-09-16.
    """
    if runner is not None:
        try:
            handle = runner(command)
        except OSError as exc:
            return Verdict(
                name, False, f"{name}: FAILED — {' '.join(command)} did not start: {exc}"
            )
        return _runner_verdict(name, command, handle)
    outcome = supervisor.start_service(
        name,
        list(command),
        url,
        probe=probe,
        budget_s=_verification_budget_s(name, timeout_s),
    )
    return Verdict(name, outcome.ok, outcome.message)


def _bring_up(
    name: str,
    command: Sequence[str],
    url: str | None,
    *,
    restart: bool,
    runner: Runner | None,
    probe: Probe,
    timeout_s: float | None,
) -> Verdict:
    """One service, from observation to row: *observe → decide → act → verify → record*.

    Without ``--restart`` this signals nothing and removes no state file, whatever it finds: a
    healthy service whose record still names the live instance is skipped out loud, and drift or
    unknown provenance is a refusal that names its own remedy. A start verb that reboots a live
    engine mid-session is a worse bug than the one it was rewritten for.
    """
    if url is None:
        return Verdict(
            name,
            False,
            f"{name}: FAILED — this profile configures no endpoint for {name}, so a start could "
            "not be verified and none was attempted",
        )
    if restart:
        # The deliberate reboot, and the only branch that signals anything. The identity rules are
        # ``supervisor.stop``'s (T-0033 ruling A11: the recorded start time decides, and there is
        # no SIGKILL follow-up because the child fronts a container).
        stopped = supervisor.stop(supervisor.run_dir() / f"{name}.pid")
        print(f"astra up: {stopped.message}")
        if stopped.still_running:
            return Verdict(
                name,
                False,
                f"{name}: FAILED — --restart could not stop what is recorded, so nothing was "
                "started over it: a second instance on one port leaves two owners and one "
                "record. Stop it by hand, then run the start again.",
            )
        return _start(name, command, url, runner=runner, probe=probe, timeout_s=timeout_s)

    decision = _decide(name, command, url, probe)
    if decision.action == "SKIP":
        return Verdict(name, True, decision.reason)
    if decision.action == "CONFLICT":
        return Verdict(name, False, f"CONFLICT — {decision.reason}")
    return _start(name, command, url, runner=runner, probe=probe, timeout_s=timeout_s)


def up(config: Config, args: argparse.Namespace, runner: Runner | None, probe: Probe) -> int:
    """Start what is not running, say what was already there, and refuse what it cannot place.

    One verdict row per service, drawn from the decision, then the shared health table. The exit
    code is non-zero iff a row is CONFLICT, FAILED or NOT-NEW, or the table holds a failure; an
    all-skip run exits 0. ``probe`` is the only health seam — the same one decides and verifies.
    """
    profile = config.profiles[args.profile] if args.profile else config.active()
    plan = start_plan(profile)
    if args.dry_run:
        print(f"astra up --profile {profile.name} --dry-run: the start plan, in order:")
        for command in plan:
            print("  " + " ".join(command))
        for name, url in endpoints_of(profile):
            print(f"  then probe {name} at {url}")
        return 0
    names = [service_name_of(command) for command in plan]
    spec = getattr(args, "restart", None)
    unknown = sorted(
        part.strip()
        for part in str(spec or "").split(",")
        if part.strip() and part.strip() not in {*names, "all"}
    )
    if unknown:
        print(
            f"astra up: --restart names {', '.join(unknown)}, which this profile does not start "
            f"(its plan is {', '.join(names)}); nothing was stopped and nothing was started",
            file=sys.stderr,
        )
        return 2
    restart = restart_targets(spec, names)
    endpoints = dict(endpoints_of(profile))
    failed = 0
    for command in plan:
        name = service_name_of(command)
        verdict = _bring_up(
            name,
            command,
            endpoints.get(name),
            restart=name in restart,
            runner=runner,
            probe=probe,
            timeout_s=args.timeout,
        )
        print(f"astra up: {verdict.message}")
        if not verdict.ok:
            failed += 1
    health = report_health(profile, probe, timeout_s=args.timeout)
    return 1 if failed else health


def _started_note(handle: object, command: Sequence[str]) -> str:
    """What a start left behind, when the runner says so (an injected fake may not)."""
    pid = getattr(handle, "pid", None)
    log_path = getattr(handle, "log_path", None)
    parts = [" ".join(str(part) for part in command)]
    if pid is not None:
        parts.append(f"pid {pid}")
    if log_path is not None:
        parts.append(f"log {log_path}")
    return ", ".join(parts)


def health_rows(
    profile: Profile,
    probe: Probe,
    *,
    timeout_s: float | None,
    wait: bool = True,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Probe every endpoint of the profile until it answers or **its own** budget runs out.

    The budget is ``supervisor.probe_budget_s(name)`` — the engine's init is ~68 s and the speech
    services answer in seconds, so one number for all three either fails a cold engine or waits a
    minute for a dead socket (ruling A14). ``--timeout`` caps it only when the operator passes it,
    and then the cap is printed beside the budget it replaced.

    One row per service — plus, for a FAILED one, a second line quoting the last line of its own
    ``logs/<name>.log`` (T-0041 criterion 3) — and the same rows whoever asked: ``up``'s final
    report and ``status`` are this function (criterion 14). ``wait=False`` is ``status``: one
    round, no sleeping, because a snapshot that blocks for two minutes is not a snapshot.
    """
    pending = dict(endpoints_of(profile))
    budget = {
        name: (
            supervisor.probe_budget_s(name)
            if timeout_s is None
            else min(supervisor.probe_budget_s(name), max(0.0, float(timeout_s)))
        )
        for name in pending
    }
    reasons: dict[str, str] = {}
    failed: dict[str, str] = {}
    waited: dict[str, float] = {}
    started = _monotonic()
    while pending:
        for name, url in list(pending.items()):
            reason = probe(name, url)
            elapsed = _monotonic() - started
            if reason is None:
                del pending[name]
                continue
            reasons[name] = reason
            # Past its own budget: recorded and dropped, so the loop stops asking it and the
            # services that are still starting keep their own budgets (ruling A15).
            if elapsed >= budget[name] or not wait:
                failed[name] = reason
                waited[name] = elapsed
                del pending[name]
        if pending:
            _sleep(PROBE_INTERVAL_S)  # a constant, never 0: a 0 here is a spin (ruling A15)
    rows = []
    for name, url in endpoints_of(profile):
        # The budget printed is the budget honoured — a row that names one number and waits
        # another is a lie about why it failed (ruling A14).
        printed = f"budget {budget[name]:g} s"
        if budget[name] != supervisor.probe_budget_s(name):
            printed += " from --timeout"
        if name in failed:
            rows.append(
                f"{name:<6} FAILED  {url}  ({printed}, waited {waited[name]:.1f} s) "
                f"last response: {failed[name]}"
            )
            # What the probe could not know (T-0041 criterion 3): a script that cannot exec leaves
            # "connection refused" on the socket and the reason in its own log. Only the failing
            # service's log, and only when it has a line — the table is not a log viewer.
            tail = supervisor.log_tail(name)
            if tail is not None:
                rows.append(f"{'':<6}   last log line: {tail}")
        else:
            rows.append(f"{name:<6} ok      {url}  ({printed})")
    return tuple(rows), failed


def report_health(
    profile: Profile, probe: Probe, *, timeout_s: float | None, wait: bool = True
) -> int:
    """Print the table and turn it into an exit code: non-zero iff something is unhealthy."""
    rows, failed = health_rows(profile, probe, timeout_s=timeout_s, wait=wait)
    for row in rows:
        print(row)
    return 1 if failed else 0


def down(config: Config, args: argparse.Namespace) -> int:
    """Stop exactly what ``up`` started: every pid file, and nothing that is not in one.

    ``supervisor.stop`` decides — it kills only when the recorded process start time still
    matches ``/proc/<pid>/stat`` (ruling A11), so a pid the kernel has reused is reported and
    left alone. Running it twice is a no-op: the first run removed the files.
    """
    del args
    del config
    files = supervisor.pid_files(supervisor.run_dir())
    if not files:
        print("astra down: no pid files, so nothing this CLI started is running")
        return 0
    code = 0
    for pid_file in files:
        outcome = supervisor.stop(pid_file)
        print(f"astra down: {outcome.message}")
        if outcome.still_running:
            # Ruling A16: no SIGKILL (it would orphan the container the child fronts), so a
            # process that outlived the grace period is a failure to report, not one to force.
            code = 1
    return code


def _record_row(record: StateRecord, *, answering: bool, url: str | None) -> str:
    """What one versioned record is, now, according to the authority that answers for its kind.

    A pid is the wrong authority for a container: after the 2026-09-16 run overwrote the records
    with dead launcher pids, this verb called the containers that had been serving for two hours
    "not running (stale pid file)". So the handle is resolved through
    ``supervisor.live_identity`` — ``/proc`` for a host process, ``docker inspect`` for a
    container — and the row names the handle it checked, so a human can confirm or refute it.

    Three answers, and they are different facts. Alive is *running*. An authority that answered
    and said the handle is gone, while the endpoint still answers, is the honest half: something
    is serving that ``astra down`` will not stop. An authority that **could not be asked** is
    neither — saying "not running" there is the same conflation one layer down (ruling A24).
    """
    live = supervisor.live_identity(record)
    handle = handle_text(record, live)
    argv = f" [{' '.join(record.argv)}]" if record.argv else ""
    serving = f", and {url} answers" if answering and url else ""
    if live.alive:
        since = live.started_at or f"{record.started_at:.0f} (epoch seconds)"
        return f"{record.name} running — {handle} since {since}, launcher pid {record.pid}{argv}"
    if live.unknown:
        return (
            f"{record.name}: {handle} could not be asked about ({live.note}){serving}, so this "
            f"CLI cannot say whether what it started is still running{argv}"
        )
    gone = live.note or "it resolves to nothing"
    if answering:
        return (
            f"{record.name}: {handle} is not running ({gone}), but {url} answers — what is "
            f"serving is not the instance this CLI started, and `astra down` will not stop "
            f"it{argv}"
        )
    return f"{record.name}: {handle} is not running ({gone}){argv}"


def status(config: Config, args: argparse.Namespace, probe: Probe) -> int:
    """The same probe table as ``up``'s final report, plus what each record is. Starts nothing.

    Read-only in every branch: no signal, no spawn, nothing written — a verb that repaired the
    record it is reporting on would have no fixed point, and the record is the only provenance
    there is.
    """
    profile = config.profiles[args.profile] if getattr(args, "profile", None) else config.active()
    rows, failed = health_rows(profile, probe, timeout_s=None, wait=False)
    for row in rows:
        print(row)
    endpoints = dict(endpoints_of(profile))
    for pid_file in supervisor.pid_files(supervisor.run_dir()):
        name = pid_file.stem
        record = supervisor.read_state_record(pid_file)
        if record is not None:
            answering = name in endpoints and name not in failed
            print(
                f"astra status: {_record_row(record, answering=answering, url=endpoints.get(name))}"
            )
            continue
        # A T-0033-shape file ({pid, argv, start_time}): no kind and no digest, so the only
        # question it can answer is whether that pid is alive. Unchanged, deliberately.
        try:
            pid, argv, _start_time = supervisor.read_pid_file(pid_file)
        except (OSError, ValueError, KeyError) as exc:
            print(f"astra status: {name} has an unreadable pid file ({exc})")
            continue
        alive = "running" if supervisor.is_alive(pid) else "not running (stale pid file)"
        print(f"astra status: {name} pid {pid} {alive} [{' '.join(argv)}]")
    return 1 if failed else 0


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# --- `astra doctor`: read the box, decide nothing ------------------------------------------------


def doctor(config: Config, args: argparse.Namespace) -> int:
    """Print one line per check and exit non-zero iff something FAILs.

    Read-only in every sense: the collectors ask the box (status polls, ``docker inspect``,
    ``nvidia-smi``, ``ss``), ``preflight.evaluate`` decides from the facts alone, and nothing here
    starts a service, writes a file or changes a lock. A WARN is a fact somebody should pin, not
    a reason to stop.
    """
    profile = config.profiles[args.profile] if getattr(args, "profile", None) else config.active()
    checks = preflight.evaluate(preflight.gather(profile))
    lines = preflight.render_json(checks) if args.json else preflight.render(checks)
    for line in lines:
        print(line)
    return 1 if any(check.status == "FAIL" for check in checks) else 0


# --- `astra hud`: the page, on the port the config names ----------------------------------------


def serve_hud(services: Services, on_busy: Callable[[OSError], None]) -> object | None:
    """Bind the static server, or hand the bind error to the caller's ``on_busy`` and return ``None``.

    Never falls back to another port: a page on a port nobody chose is a page talking to nothing
    (T-0028 criterion 19, applied to the second listener). What a taken port *means* is the
    caller's policy: the ``hud`` verb exits 1, a session runs on without its own HUD (T-0197).
    """
    try:
        return hud_static.HudStaticServer(hud_static.HUD_DIR, services.hud_port)
    except OSError as exc:
        on_busy(exc)
        return None


def _hud_verb_refusal(services: Services, exc: OSError) -> str:
    return (
        f"astra hud: cannot bind 127.0.0.1:{services.hud_port} ({exc}). The port is in the "
        f"profile (services.hud_port); free it or change it there — this never moves to "
        f"another port."
    )


def _session_hud_busy_line(services: Services, exc: OSError) -> str:
    """T-0197 criterion 5: one stderr line, and the session continues (no probe, no retry, no other port)."""
    return (
        f"astra session: cannot serve the HUD page on 127.0.0.1:{services.hud_port}: the port is in "
        f"use or refused ({exc}); if a Gearvis HUD is serving there, the page is at "
        f"{hud_static.page_urls(services)[0]}; the session continues without its own HUD server"
    )


def hud(config: Config, args: argparse.Namespace) -> int:
    profile = config.profiles[args.profile] if getattr(args, "profile", None) else config.active()
    server = serve_hud(
        profile.services,
        lambda exc: print(_hud_verb_refusal(profile.services, exc), file=sys.stderr),
    )
    if server is None:
        return 1
    for url in hud_static.page_urls(profile.services):
        print(url)
    print("astra hud: serving hud/ on loopback; Ctrl-C to stop")
    try:
        server.serve_forever()  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        pass
    finally:
        server.close()  # type: ignore[attr-defined]
    return 0


# --- the regime line: measured facts, or the literal `unknown` -----------------------------------


def regime_line(profile: Profile, facts: preflight.Facts, mic: str = MIC_UNKNOWN) -> str:
    """What the run is actually made of (criterion 19).

    Every token is a measurement or the literal ``unknown`` with the reason it is unknown. A
    config value in the place of a measured one is worse than a gap, because it looks measured:
    the STT device comes from the container's argv and the model id from the engine's own
    listing, never from ``Profile.stt_backend`` or from the lock.

    ``mic`` is the ``Microphone.field`` of the run — which capture source heard the turn, or
    ``none`` — because a number measured through a webcam mic in a room and one measured through
    a null sink's monitor are not the same regime (T-0045 criterion 2).
    """
    parts = [f"regime: profile={profile.name}", f"mic={mic}"]
    if facts.stt_argv is None:
        parts.append("stt_device=unknown (no container inspected)")
    else:
        parts.append("stt_device=" + ("cpu" if "--cpu" in facts.stt_argv else "gpu"))
    if facts.served_model_id is None:
        parts.append("served_model=unknown (no answer from the engine)")
    else:
        parts.append(f"served_model={facts.served_model_id}")
    if facts.vram_used_mib is None or facts.vram_total_mib is None:
        parts.append("vram=unknown (no nvidia-smi answer)")
    else:
        parts.append(f"vram={facts.vram_used_mib}/{facts.vram_total_mib}MiB")
    return " ".join(parts)


# --- `astra session`: the production object graph ----------------------------------------------


@dataclass
class SessionGraph:
    """What ``astra session`` serves: a session, the endpoint, and the frame source they share."""

    session: Session
    server: HudServer
    frames: BrowserFrameSource
    description: str
    #: ``--silent``: the graph was built on ``NullAudioDevice`` (T-0039 criteria 17-19). Carried
    #: on the graph rather than re-derived from ``args`` so ``run_session`` can say it: a session
    #: that looks like it is listening and is not is the failure this line exists to prevent.
    silent: bool = False
    #: T-0111g B5: ``live``'s start-page server (``pages.server.StartPageServer``), bound by ``build_session``
    #: and stopped by ``_session_graph_cmd`` after the session closed the browser; ``None`` otherwise.
    pages: Any = None
    #: T-0128c: the relay of the plugin's canvas tap to the HUD (``CanvasRelay``), started after
    #: the server and closed before it; ``None`` when it could not be built (``relay_off`` says why).
    relay: Any = None
    relay_off: str | None = None
    #: T-0125: the web search backend ``build_session`` wired into the kit (``run_session`` closes it after the
    #: session), and why a default-on search degraded to off (one session-log note at start); both ``None`` otherwise
    search: Any = None
    search_off: str | None = None
    #: T-0125 criterion 3: the backend's engine label for the startup line (``_search_backend``'s), with ``search``
    search_label: str | None = None
    #: T-0160 C-1: ``live``'s browser runs in web mode (``web_of``); read for the C-3 notice and the C-4 sandbox
    #: message, so the browser's own seam is not widened
    web: bool = False

    def describe(self) -> str:
        return self.description


def silent_of(args: argparse.Namespace) -> bool:
    """Whether ``--silent`` was asked for. Namespaces built before the flag existed say no."""
    return bool(getattr(args, "silent", False))


@dataclass(frozen=True)
class Microphone:
    """Which capture node a session opens, decided once (T-0045 criteria 1 and 2).

    Resolved where the session starts and then carried, so the ``mic=`` the human reads on the
    regime line is the node the device was actually asked for and not a second reading of the
    graph that could answer differently.
    """

    #: What ``open_device(source=…)`` is given. ``None`` is an output-only stream: no capture
    #: stream is opened, ``stt_blocks()`` never yields and the session hears nothing.
    node: str | None
    #: The ``mic=`` token of the resolved-graph and regime lines: a node name, ``none``, ``unknown``.
    field: str
    #: Why there is no node, when there is none. Printed where it decides the run.
    why: str = ""

    @property
    def unknown(self) -> bool:
        """No microphone was *chosen*: the box could not say which one is the default."""
        return self.field == MIC_UNKNOWN


#: T-0181 A3: why a session has no microphone, keyed by the ``Microphone.why`` that decided it (exact wording; the page
#: shows it verbatim as the disabled mic control's title). ``--silent`` wins over ``--no-mic``/``--mic none`` because
#: ``resolve_mic`` checks it first (A-R15).
MIC_UNAVAILABLE_REASONS: dict[str, str] = {
    "--silent": "this session is silent: no audio device is open (--silent)",
    "--no-mic": "this session has no microphone (--no-mic)",
    "--mic none": "this session has no microphone (--mic none)",
}


def mic_unavailable_of(microphone: Microphone) -> dict[str, Any]:
    """T-0181 A3/A-R13: ``Session``'s ``mic_unavailable`` keyword, only when the resolved microphone is off (never
    ``None``), so an available-mic construction calls ``Session`` exactly as before."""
    if microphone.node is None and microphone.field == MIC_OFF:
        reason = MIC_UNAVAILABLE_REASONS.get(microphone.why)
        if reason is not None:
            return {"mic_unavailable": reason}
    return {}


def resolve_mic(args: argparse.Namespace) -> Microphone:
    """``--mic`` / ``--no-mic`` / ``ASTRA_AUDIO_SOURCE`` / the default source (criterion 1, A14).

    The defect this exists for: ``open_device(sink=…)`` with no ``source`` opens an output-only
    stream, so every turn of every live session recorded ``stt_frames_fed: 0`` (2026-09-15). The
    default is **read** from the graph (``preflight.default_source()``, a PipeWire node name),
    never guessed and never taken from ``--source``, which is the perception source.

    The ruled precedence, highest first (ruling A14)::

        --mic <node>   >   --no-mic   >   ASTRA_AUDIO_SOURCE   >   the graph's default source

    ``ASTRA_AUDIO_SOURCE`` is a step **here**, where the answer is the one that gets printed,
    rather than the fallback ``open_device`` performs for itself when it is handed no source: at
    that seam "no capture source" and "the node the environment names" are the same value, so
    ``--no-mic`` printed ``mic=none`` while a duplex stream was open on the exported node. Deciding
    it here also puts the variable above the graph default (someone who exported the rig's monitor
    got the webcam) and keeps a box with an unreadable graph but an exported node out of ruling
    A3's refusal -- that box is the route the task's Goal calls the only one that works today.

    Namespaces built before the flags existed (and ``--silent``, which opens no device at all)
    say ``none``.
    """
    if silent_of(args) or getattr(args, "no_mic", False):
        return Microphone(None, MIC_OFF, "--silent" if silent_of(args) else "--no-mic")
    chosen = getattr(args, "mic", None)
    if chosen == MIC_OFF:
        return Microphone(None, MIC_OFF, "--mic none")
    if chosen:
        return Microphone(str(chosen), str(chosen))
    exported = os.environ.get(ENV_MIC)
    if exported:
        return Microphone(exported, exported)
    node = preflight.default_source()
    if node is None:
        return Microphone(None, MIC_UNKNOWN, preflight.NO_CAPTURE_SOURCE_FIX)
    return Microphone(node, node)


def audio_path_of(args: argparse.Namespace) -> str:
    """The regime word every ``turns.jsonl`` line carries (T-0039 criterion 18, ruling A12).

    A silent run says ``silent`` and nothing else can override it: its timings are not comparable
    to an acoustic run's and must never be read as if they were.
    """
    return "silent" if silent_of(args) else str(args.audio_path)


def logging_on(args: argparse.Namespace) -> bool:
    """T-0043 criterion 15: logging is **on** by default; the flag is ``--no-log``.

    ``--no-log`` is the start state, not a lock: the page may turn logging on afterwards (the user
    at the page is the same person who typed the flag). Namespaces built before the flag existed
    say yes, which is the default.
    """
    return not bool(getattr(args, "no_log", False))


def log_dir_of(args: argparse.Namespace) -> Path:
    """Where the session log goes: ``--log-dir``, else ``logs/sessions/`` (already gitignored)."""
    chosen = getattr(args, "log_dir", None)
    return Path(chosen) if chosen else supervisor.log_dir() / "sessions"


def log_line_of(args: argparse.Namespace) -> str:
    """The ``log=`` field of the resolved-graph line: a directory, or ``off``."""
    return f"log={log_dir_of(args)}" if logging_on(args) else "log=off"


def silent_audio(clock: SessionClock) -> PipeWireAudioIO:
    """The playback half of ``--silent``: the same ``PipeWireAudioIO`` over a null device.

    A device swap, not a policy branch (T-0039 §5): TTS still runs, ``astra.spoken`` is still
    published and ``SPEAKING -> IDLE`` still fires on a real drain, so the turn machine cannot
    tell the two runs apart. The session's own clock is injected, so the silent run is paced in
    real time exactly as the speech would have been.
    """
    return PipeWireAudioIO(NullAudioDevice(clock))


def describe_graph(
    config: Config,
    args: argparse.Namespace,
    facts: preflight.Facts | None = None,
    mic: Microphone | None = None,
) -> str:
    """The resolved regime, printed by ``--dry-run`` and on the way up.

    Constructs nothing and **asks the box nothing**: the facts arrive as an argument, and so does
    the resolved microphone (resolving one reads the graph). Without them every measured token
    reads ``unknown`` with its reason, which is what ``--dry-run`` prints — a dry run has
    measured nothing, and saying so is the honest line.
    """
    microphone = mic if mic is not None else resolve_mic(args)
    profile = config.profiles[args.profile] if args.profile else config.active()
    services = profile.services
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DIR
    turns = Path(args.turns) if args.turns else data_dir / "turns.jsonl"
    host = "127.0.0.1"
    lines = [
        f"profile={profile.name} thinking={profile.diffusion.thinking} "
        f"canvas_length={profile.diffusion.canvas_length} "
        f"max_denoising_steps={profile.diffusion.max_denoising_steps}",
        f"data_dir={data_dir} turns={turns} {log_line_of(args)}",
        # T-0045 criterion 2: ``mic=`` beside ``sink=``. ``source=`` is the perception source
        # (screen/camera) and has never had anything to do with the microphone.
        f"source={args.source} audio_path={audio_path_of(args)} "
        f"sink={'silent' if silent_of(args) else (args.sink or 'default')} "
        f"mic={microphone.field}",
        f"vllm={services.vllm}",
        f"stt={services.stt} ({profile.stt_backend})",
        f"tts={services.tts} ({profile.tts_backend})",
        f"page=ws://{host}:{services.conductor_port}/v1/realtime",
        "graph: Session(TurnMachine, VllmClient, MoshiSttClient, PocketTtsClient, "
        "PipeWireAudioIO, BrowserFrameSource, SqliteMemory) + HudServer",
        # Criterion 19: measured, not guessed — and gathered by the caller, never here: a
        # `--dry-run` that shells out to docker and the engine is not a dry run (ruling A19).
        regime_line(profile, facts if facts is not None else preflight.Facts(), microphone.field),
    ]
    return "\n".join(lines)


@contextlib.contextmanager
def only_the_resolved_source() -> Iterator[None]:
    """Hide ``ASTRA_AUDIO_SOURCE`` from ``open_device`` for the length of the call (ruling A14).

    ``open_device`` reads the variable itself whenever it is given ``source=None``, and
    ``audio_device.py`` may not change (criterion 6). So ``--no-mic`` with the variable exported
    reached the device layer as "no source", the device layer read the variable, and the session
    listened on that node while every line the human read said ``mic=none``. Unsetting it for the
    duration of the call is what makes the printed line true: ``resolve_mic`` has already consulted
    the variable at its ruled precedence and passes what it decided explicitly, so the fallback has
    nothing left to add and only ever contradicts what was printed.

    Restored afterwards, whatever happens: it is the human's shell, and the rest of the process
    (the rung, a second start) may still read it.
    """
    exported = os.environ.pop(ENV_MIC, None)
    try:
        yield
    finally:
        if exported is not None:
            os.environ[ENV_MIC] = exported


class AudioDeviceUnavailable(RuntimeError):
    """The playback device could not be opened; criterion 19 names it and exits non-zero."""


def open_audio(args: argparse.Namespace) -> PipeWireAudioIO:
    """The one call that can fail on hardware, wrapped so only *it* is reported as a device fault.

    T-0006 ruling A1: the playback module takes its device by injection, and which sink is playing
    is part of the regime every turns.jsonl line states — so a missing device is named and fatal,
    never a quiet move to another one (criterion 19).

    **The capture source goes in here too** (T-0045): ``open_device(sink=…)`` alone opens an
    output-only stream — ``PortAudioDevice`` is duplex only when it has a ``source`` — so the
    capture pump had nothing to pump and every live turn recorded ``stt_frames_fed: 0``. A node
    the graph does not have is refused by ``open_device`` before any stream exists (T-0035
    criterion 4) and that refusal is reported here, naming the node asked for.
    """
    device = args.sink or "the default sink"
    microphone = resolve_mic(args)
    try:
        with only_the_resolved_source():
            return PipeWireAudioIO(open_device(sink=args.sink, source=microphone.node))
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioDeviceUnavailable(f"cannot open the audio device ({device}): {exc}") from exc


def tools_arg(value: str) -> str:
    """``--tools``: ``off``, ``dry-run`` (the profile's ``dry_run``) or ``live`` (T-0111g). Anything else is an
    argv error naming the modes (T-0092b criterion 1)."""
    spelled = {"off": "off", "dry-run": "dry_run", "live": "live"}
    if value not in spelled:
        raise argparse.ArgumentTypeError(
            tools_refusal(value) + "; --tools takes off, dry-run or live"
        )
    return spelled[value]


def tools_mode_of(profile: Profile, args: argparse.Namespace) -> str:
    """The session's tools setting: ``--tools`` when given, else the profile's (ruling H-A4)."""
    flag = getattr(args, "tools", None)
    return str(flag) if flag is not None else profile.tools


def browser_of(profile: Profile, args: argparse.Namespace) -> str:
    """T-0111g B2: ``live``'s browser, ``headed`` or ``headless``: ``--browser`` when given, else the profile's."""
    flag = getattr(args, "browser", None)
    return str(flag) if flag is not None else profile.browser


def web_of(profile: Profile, args: argparse.Namespace, mode: str) -> bool:
    """T-0160 C-1 (ruling 2): whether the live agent browser runs in web mode. On in live unless ``--no-web`` or
    the profile's ``browser_web = false``; never in another resolved mode (``--no-web`` there has no effect)."""
    if mode != "live":
        return False
    return not getattr(args, "no_web", False) and profile.browser_web


#: T-0132 A2: where a grounding value came from
GroundingSource = Literal["flag", "profile", "default", "unset"]


@dataclass(frozen=True)
class Grounding:
    """T-0132 A2: the session's resolved grounding values, each with its source. ``reground_visual_tokens`` is pass
    2's budget (B1): the flag, else the profile, else the main budget."""

    visual_tokens: int | None
    visual_tokens_from: GroundingSource
    reground: bool
    reground_from: GroundingSource
    reground_visual_tokens: int | None
    reground_visual_tokens_from: GroundingSource

    def turn_extra(self) -> dict[str, Any]:
        """A4c: the turn line's ``grounding`` dict (``Session(grounding=...)``, live only)."""
        return {
            "visual_tokens": self.visual_tokens,
            "visual_tokens_from": self.visual_tokens_from,
            "reground": self.reground,
            "reground_from": self.reground_from,
            "reground_visual_tokens": self.reground_visual_tokens,
            "reground_visual_tokens_from": self.reground_visual_tokens_from,
        }

    def line(self) -> str:
        """A4a: the live-only ``grounding`` line; with ReGround off the pass-2 budget is omitted."""
        parts = [
            f"visual_tokens={self.visual_tokens} ({self.visual_tokens_from})",
            f"reground={'on' if self.reground else 'off'} ({self.reground_from})",
        ]
        if self.reground:
            parts.append(
                f"reground_visual_tokens={self.reground_visual_tokens} "
                f"({self.reground_visual_tokens_from})"
            )
        return "astra session: grounding: " + ", ".join(parts)


def grounding_of(profile: Profile, args: argparse.Namespace, mode: str) -> Grounding:
    """T-0132 A2: the one resolver of the visual-token budget and ReGround, keyed on the *resolved* tools mode.

    The budget, in every mode (T-0183 A1): flag, else profile, else ``LIVE_DEFAULT_VISUAL_TOKENS`` (1120) from
    ``"default"``, unconditionally (H-T183-1), so ``cli`` never resolves it ``"unset"``; only a directly built
    client sends no budget. ReGround in ``live``: flag, else profile, else ``LIVE_DEFAULT_REGROUND`` (off since
    T-0133). Elsewhere ReGround is never defaulted on: it is on only when asked for explicitly, which
    ``session_cmd`` refuses (exit 2, T-0131 B9). Pass 2's budget is the flag, else the profile, else 280
    (``LIVE_DEFAULT_REGROUND_VISUAL_TOKENS``, T-0133 A5) in live with ReGround on at a main budget of 1120, else the
    main budget (T-0132 B1)."""
    vt_flag = getattr(args, "visual_tokens", None)
    budget: int | None
    budget_from: GroundingSource
    if vt_flag is not None:
        budget, budget_from = int(vt_flag), "flag"
    elif profile.visual_tokens is not None:
        budget, budget_from = profile.visual_tokens, "profile"
    else:
        # T-0183 A1: 1120 in every mode (was live only; off/dry_run sent nothing, so the engine used 280)
        budget, budget_from = LIVE_DEFAULT_VISUAL_TOKENS, "default"
    rg_flag = getattr(args, "reground", None)
    reground: bool
    reground_from: GroundingSource
    if rg_flag is not None:
        reground, reground_from = bool(rg_flag), "flag"
    elif profile.reground is not None:
        reground, reground_from = profile.reground, "profile"
    elif mode == "live":
        reground, reground_from = LIVE_DEFAULT_REGROUND, "default"
    else:
        reground, reground_from = False, "unset"
    # T-0132 B1: pass 2's budget is the flag, else the profile, else the resolved main budget ("default")
    crop_flag = getattr(args, "reground_visual_tokens", None)
    crop: int | None
    crop_from: GroundingSource
    if crop_flag is not None:
        crop, crop_from = int(crop_flag), "flag"
    elif profile.reground_visual_tokens is not None:
        crop, crop_from = profile.reground_visual_tokens, "profile"
    elif mode == "live" and reground and budget == 1120:
        # T-0133 A5 (H-T133-1, R-i): keyed on resolved values only (mode, ReGround, the main budget), never on a
        # source; bound to exactly 1120, the only main budget the T-0132 A/B measured the 280 crop at
        crop, crop_from = LIVE_DEFAULT_REGROUND_VISUAL_TOKENS, "default"
    else:
        crop, crop_from = budget, ("default" if budget is not None else "unset")
    return Grounding(
        visual_tokens=budget,
        visual_tokens_from=budget_from,
        reground=reground,
        reground_from=reground_from,
        reground_visual_tokens=crop,
        reground_visual_tokens_from=crop_from,
    )


@dataclass(frozen=True)
class FrameSettings:
    """T-0227 A-4: the session's screens-per-turn cap and HUD capture interval, each with its source (the
    ``Grounding`` shape). Both apply in every tools mode, so neither is ever ``"unset"``."""

    max_frames_per_turn: int
    max_frames_per_turn_from: GroundingSource
    frame_interval_ms: int
    frame_interval_ms_from: GroundingSource


def frame_settings_of(profile: Profile, args: argparse.Namespace) -> FrameSettings:
    """T-0227 A-4: the one resolver of ``max_frames_per_turn`` and ``frame_interval_ms``, the ``grounding_of`` way:
    flag, else profile, else the ``config`` default. Never fed to ``TurnPolicy`` (the machine's cap stays 3)."""
    cap_flag = getattr(args, "max_frames_per_turn", None)
    cap: int
    cap_from: GroundingSource
    if cap_flag is not None:
        cap, cap_from = int(cap_flag), "flag"
    elif profile.max_frames_per_turn is not None:
        cap, cap_from = profile.max_frames_per_turn, "profile"
    else:
        cap, cap_from = DEFAULT_MAX_FRAMES_PER_TURN, "default"
    gap_flag = getattr(args, "frame_interval_ms", None)
    gap: int
    gap_from: GroundingSource
    if gap_flag is not None:
        gap, gap_from = int(gap_flag), "flag"
    elif profile.frame_interval_ms is not None:
        gap, gap_from = profile.frame_interval_ms, "profile"
    else:
        gap, gap_from = DEFAULT_FRAME_INTERVAL_MS, "default"
    return FrameSettings(
        max_frames_per_turn=cap,
        max_frames_per_turn_from=cap_from,
        frame_interval_ms=gap,
        frame_interval_ms_from=gap_from,
    )


@dataclass(frozen=True)
class Feedback:
    """T-0134c A2: the session's resolved live feedback and action echo, each with its source (the ``Grounding``
    shape). Both are ``None`` from ``"unset"`` outside live."""

    live_feedback: LiveFeedbackChoice | None
    live_feedback_from: GroundingSource
    action_echo: ActionEcho | None
    action_echo_from: GroundingSource

    def session_value(self) -> LiveFeedback | None:
        """What ``Session(live_feedback=...)`` gets: ``None`` outside live; ``off`` is ``LiveFeedback(False,
        False)`` (R-m), so every live turn line records what was on."""
        if self.live_feedback is None:
            return None
        return LiveFeedback(
            screen_label=self.live_feedback != "off",
            page_extent=self.live_feedback == "label+extent",
        )

    def line(self) -> str:
        """A5a: the live-only ``feedback`` line, printed right after the ``grounding`` line."""
        return (
            f"astra session: feedback: live_feedback={self.live_feedback} ({self.live_feedback_from}), "
            f"action_echo={self.action_echo} ({self.action_echo_from})"
        )


def feedback_of(profile: Profile, args: argparse.Namespace, mode: str) -> Feedback:
    """T-0134c A2: the one resolver of the live feedback and the action echo, keyed on the *resolved* tools mode.
    In ``live`` each value resolves on its own: the flag, else the profile, else ``LIVE_DEFAULT_FEEDBACK`` /
    ``LIVE_DEFAULT_ACTION_ECHO``. Elsewhere both are ``None`` from ``"unset"`` (a flag there is refused in
    ``session_cmd``). Pure: reads ``args`` with ``getattr``, so an empty ``Namespace()`` resolves the defaults."""
    if mode != "live":
        return Feedback(None, "unset", None, "unset")
    lf_flag = getattr(args, "live_feedback", None)
    live_feedback: LiveFeedbackChoice
    live_feedback_from: GroundingSource
    if lf_flag is not None:
        live_feedback, live_feedback_from = lf_flag, "flag"
    elif profile.live_feedback is not None:
        live_feedback, live_feedback_from = profile.live_feedback, "profile"
    else:
        live_feedback, live_feedback_from = LIVE_DEFAULT_FEEDBACK, "default"
    ae_flag = getattr(args, "action_echo", None)
    action_echo: ActionEcho
    action_echo_from: GroundingSource
    if ae_flag is not None:
        action_echo, action_echo_from = ae_flag, "flag"
    elif profile.action_echo is not None:
        action_echo, action_echo_from = profile.action_echo, "profile"
    else:
        action_echo, action_echo_from = LIVE_DEFAULT_ACTION_ECHO, "default"
    return Feedback(live_feedback, live_feedback_from, action_echo, action_echo_from)


def feedback_flags_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0134c A3: why ``--live-feedback``/``--action-echo`` in a resolved mode other than ``live`` is refused, or
    ``None``; there it would silently do nothing. Orchestrator ruling 2026-10-23 (the ``reground_refusal``
    pattern): so is a profile's ``live_feedback``/``action_echo`` key when ``--tools`` overrode a live profile."""
    if mode == "live":
        return None
    given = [
        flag
        for flag, dest in (("--live-feedback", "live_feedback"), ("--action-echo", "action_echo"))
        if getattr(args, dest, None) is not None
    ] + [
        f"the profile's {dest}"
        for dest in ("live_feedback", "action_echo")
        if getattr(profile, dest) is not None
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the live feedback and the action echo apply "
        f"only with --tools live, and with {spelled} they would silently do nothing (T-0134c)"
    )


@dataclass(frozen=True)
class Recovery:
    """T-0135b A2: the session's resolved thought-call recovery with its source (the ``Feedback`` shape). ``None``
    from ``"unset"`` outside live."""

    thought_call_recovery: bool | None
    thought_call_recovery_from: GroundingSource

    def session_value(self) -> bool | None:
        """What ``Session(recover_thought_calls=...)`` gets: the bool in live (``off`` is ``False``, so every live
        turn line records the state), ``None`` elsewhere (no keyword is passed there)."""
        return self.thought_call_recovery

    def line(self) -> str:
        """A5a: the live-only ``recovery`` line, printed right after the ``feedback`` line."""
        word = "on" if self.thought_call_recovery else "off"
        return f"astra session: recovery: thought_call_recovery={word} ({self.thought_call_recovery_from})"


def recovery_of(profile: Profile, args: argparse.Namespace, mode: str) -> Recovery:
    """T-0135b A2: the one resolver of thought-call recovery, keyed on the *resolved* tools mode. In ``live``: the
    flag (``on``/``off``), else the profile key, else ``LIVE_DEFAULT_THOUGHT_CALL_RECOVERY``. Elsewhere ``None``
    from ``"unset"`` (a flag there is refused in ``session_cmd``). Pure: reads ``args`` with ``getattr``, so an
    empty ``Namespace()`` resolves the default."""
    if mode != "live":
        return Recovery(None, "unset")
    flag = getattr(args, "thought_call_recovery", None)
    if flag is not None:
        return Recovery(flag == "on", "flag")
    if profile.thought_call_recovery is not None:
        return Recovery(profile.thought_call_recovery, "profile")
    return Recovery(LIVE_DEFAULT_THOUGHT_CALL_RECOVERY, "default")


def recovery_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0135b A3: why ``--thought-call-recovery`` in a resolved mode other than ``live`` is refused, or ``None``;
    there it would silently do nothing. As in T-0134c, so is a live profile's ``thought_call_recovery`` key when
    ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    given = [
        what
        for what, present in (
            ("--thought-call-recovery", getattr(args, "thought_call_recovery", None) is not None),
            ("the profile's thought_call_recovery", profile.thought_call_recovery is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: thought-call recovery applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0135b)"
    )


@dataclass(frozen=True)
class PageOverview:
    """T-0140 A5: whether the live kit offers ``page_overview``, with its source (the ``Recovery`` shape). ``False``
    from ``"unset"`` outside live."""

    page_overview: bool
    page_overview_from: GroundingSource

    def line(self) -> str:
        """A5: the live-only ``page overview`` line, printed after the ``feedback`` (and ``recovery``) lines."""
        return f"astra session: page overview: {'on' if self.page_overview else 'off'} ({self.page_overview_from})"


def page_overview_of(profile: Profile, args: argparse.Namespace, mode: str) -> PageOverview:
    """T-0140 A5: the one resolver of the overview opt-in, keyed on the *resolved* tools mode. In ``live``: the flag
    (``--page-overview``/``--no-page-overview``), else the profile key, else ``LIVE_DEFAULT_PAGE_OVERVIEW``. Elsewhere
    off from ``"unset"`` (a flag there is refused in ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return PageOverview(False, "unset")
    flag = getattr(args, "page_overview", None)
    if flag is not None:
        return PageOverview(bool(flag), "flag")
    if profile.page_overview is not None:
        return PageOverview(profile.page_overview, "profile")
    return PageOverview(LIVE_DEFAULT_PAGE_OVERVIEW, "default")


def page_overview_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0140 A5 (O-A2 Q7): why ``--page-overview``/``--no-page-overview`` in a resolved mode other than ``live`` is
    refused, or ``None``; there it would silently do nothing. As in T-0134c, so is a live profile's
    ``page_overview`` key when ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    flag = getattr(args, "page_overview", None)
    given = [
        what
        for what, present in (
            ("--page-overview" if flag else "--no-page-overview", flag is not None),
            ("the profile's page_overview", profile.page_overview is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the page overview applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0140)"
    )


@dataclass(frozen=True)
class AutoOverview:
    """T-0150 A6: whether the live session shows the whole page at step 1 by itself, with its source (the
    ``PageOverview`` shape). ``False`` from ``"unset"`` outside live."""

    auto_overview: bool
    auto_overview_from: GroundingSource

    def line(self) -> str:
        """A6: the live-only ``auto overview`` line, printed right after the ``page overview`` line."""
        return f"astra session: auto overview: {'on' if self.auto_overview else 'off'} ({self.auto_overview_from})"


def auto_overview_of(profile: Profile, args: argparse.Namespace, mode: str) -> AutoOverview:
    """T-0150 A6: the one resolver of the automatic overview, keyed on the *resolved* tools mode. In ``live``: the
    flag (``--auto-overview``/``--no-auto-overview``), else the profile key, else ``LIVE_DEFAULT_AUTO_OVERVIEW``.
    Elsewhere off from ``"unset"`` (a flag there is refused in ``session_cmd``). Pure: reads ``args`` with
    ``getattr``."""
    if mode != "live":
        return AutoOverview(False, "unset")
    flag = getattr(args, "auto_overview", None)
    if flag is not None:
        return AutoOverview(bool(flag), "flag")
    if profile.auto_overview is not None:
        return AutoOverview(profile.auto_overview, "profile")
    return AutoOverview(LIVE_DEFAULT_AUTO_OVERVIEW, "default")


def auto_overview_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0150 A6: why ``--auto-overview``/``--no-auto-overview`` in a resolved mode other than ``live`` is refused, or
    ``None``; so is a live profile's ``auto_overview`` key when ``--tools`` overrode its mode (O-A2 Q7)."""
    if mode == "live":
        return None
    flag = getattr(args, "auto_overview", None)
    given = [
        what
        for what, present in (
            ("--auto-overview" if flag else "--no-auto-overview", flag is not None),
            ("the profile's auto_overview", profile.auto_overview is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the automatic page overview applies only "
        f"with --tools live, and with {spelled} it would silently do nothing (T-0150)"
    )


def auto_and_page_overview_refusal(auto: AutoOverview, page: PageOverview) -> str | None:
    """T-0150 A6 (H-T150-2): why the automatic overview and the ``page_overview`` tool, both on from any source,
    are refused, or ``None``."""
    if not (auto.auto_overview and page.page_overview):
        return None
    return (
        f"the auto overview ({auto.auto_overview_from}) and the page overview ({page.page_overview_from}) are both "
        "on, which is refused: the automatic page overview and the page_overview tool are not used together; "
        "turn one of them off (T-0150, H-T150-2)"
    )


@dataclass(frozen=True)
class OverviewFrame:
    """T-0151 A2-9: whether, on a long page, the live session's step 1 shows only the whole-page picture, with its
    source (the ``AutoOverview`` shape). ``False`` from ``"unset"`` outside live."""

    overview_frame: bool
    overview_frame_from: GroundingSource

    def line(self, given_way_to: Sequence[str] = ()) -> str:
        """A2-9: the live-only ``overview frame`` line, printed right after the ``auto overview`` line. T-0152 A5: a
        default that gave way reads ``off (with --reground)`` etc., the flags joined with `` and ``
        (``overview_frame_yielded_to``)."""
        if given_way_to:
            return f"astra session: overview frame: off (with {' and '.join(given_way_to)})"
        return f"astra session: overview frame: {'on' if self.overview_frame else 'off'} ({self.overview_frame_from})"


def _overview_frame_others(
    profile: Profile, args: argparse.Namespace, mode: str, grounding: Grounding | None
) -> tuple[str, ...]:
    """T-0152 A2/A5: the flag spellings of the auto overview, the page overview and ReGround that resolve on, from
    any source, through their own resolvers, in the order auto, page, reground (H-T152-2: the flag spelling
    whatever the source). ``grounding`` is the caller's resolved one; ``None`` resolves it here (O-1)."""
    g = grounding if grounding is not None else grounding_of(profile, args, mode)
    return tuple(
        flag
        for flag, on in (
            ("--auto-overview", auto_overview_of(profile, args, mode).auto_overview),
            ("--page-overview", page_overview_of(profile, args, mode).page_overview),
            ("--reground", g.reground),
        )
        if on
    )


def overview_frame_of(
    profile: Profile,
    args: argparse.Namespace,
    mode: str,
    *,
    grounding: Grounding | None = None,
) -> OverviewFrame:
    """T-0151 A2-9 / T-0152 A2: the one resolver of the overview frame, keyed on the *resolved* tools mode. In
    ``live``: the flag (``--overview-frame``/``--no-overview-frame``), else the profile key, else off from
    ``"default"`` when the auto overview, the page overview or ReGround resolves on (the H-T152-1 yield), else
    ``LIVE_DEFAULT_OVERVIEW_FRAME`` (read from this module's namespace at call time). Elsewhere off from ``"unset"``
    (a flag there is refused in ``session_cmd``). ``grounding``: the caller's resolved ``Grounding``, so
    ``grounding_of`` runs no more often than at BASE; ``None`` resolves it here (O-1). Pure: reads ``args`` with
    ``getattr``."""
    if mode != "live":
        return OverviewFrame(False, "unset")
    flag = getattr(args, "overview_frame", None)
    if flag is not None:
        return OverviewFrame(bool(flag), "flag")
    if profile.overview_frame is not None:
        return OverviewFrame(profile.overview_frame, "profile")
    if _overview_frame_others(profile, args, mode, grounding):
        return OverviewFrame(False, "default")
    return OverviewFrame(LIVE_DEFAULT_OVERVIEW_FRAME, "default")


def overview_frame_yielded_to(
    frame: OverviewFrame,
    profile: Profile,
    args: argparse.Namespace,
    mode: str,
    *,
    grounding: Grounding | None = None,
) -> tuple[str, ...]:
    """T-0152 A5 (O-1): what a default frame gave way to, for the ``off (with ...)`` line only; empty unless the
    frame is off from ``"default"`` in live. Pure, like ``overview_frame_of``."""
    if mode != "live" or frame.overview_frame or frame.overview_frame_from != "default":
        return ()
    return _overview_frame_others(profile, args, mode, grounding)


def overview_frame_flag_refusal(
    profile: Profile, args: argparse.Namespace, mode: str
) -> str | None:
    """T-0151 A2-9: why ``--overview-frame``/``--no-overview-frame`` in a resolved mode other than ``live`` is
    refused, or ``None``; so is a live profile's ``overview_frame`` key when ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    flag = getattr(args, "overview_frame", None)
    given = [
        what
        for what, present in (
            ("--overview-frame" if flag else "--no-overview-frame", flag is not None),
            ("the profile's overview_frame", profile.overview_frame is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the overview frame applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0151)"
    )


def overview_frame_with_refusal(
    frame: OverviewFrame, auto: AutoOverview, page: PageOverview, grounding: Grounding
) -> str | None:
    """T-0151 A2-1/A2-9 (Q-4): why the overview frame on together with the automatic overview, the page overview or
    ReGround, from any source, is refused, or ``None``; the message names both."""
    if not frame.overview_frame:
        return None
    others = (
        ("the auto overview", auto.auto_overview, auto.auto_overview_from),
        ("the page overview", page.page_overview, page.page_overview_from),
        ("ReGround", grounding.reground, grounding.reground_from),
    )
    for name, on, source in others:
        if on:
            return (
                f"the overview frame ({frame.overview_frame_from}) and {name} ({source}) are both on, which is "
                f"refused: the overview frame replaces step 1's screen with the whole-page picture, and {name} is "
                "not used with it; turn one of them off (T-0151)"
            )
    return None


@dataclass(frozen=True)
class ActionOutcomeChoice:
    """T-0136 C-2: whether the live session appends the measured outcome clause to the "screen now" label, with its
    source (the ``OverviewFrame`` shape). ``False`` from ``"unset"`` outside live."""

    action_outcome: bool
    action_outcome_from: GroundingSource

    def line(self) -> str:
        """C-5: the live-only ``action outcome`` line, printed right after the ``overview frame`` line."""
        return f"astra session: action outcome: {'on' if self.action_outcome else 'off'} ({self.action_outcome_from})"


def action_outcome_of(profile: Profile, args: argparse.Namespace, mode: str) -> ActionOutcomeChoice:
    """T-0136 C-2: the one resolver of the outcome clause, keyed on the *resolved* tools mode. In ``live``: the flag
    (``--action-outcome``/``--no-action-outcome``), else the profile key, else ``LIVE_DEFAULT_ACTION_OUTCOME`` (read
    from this module's namespace at call time). Elsewhere off from ``"unset"`` (a flag there is refused in
    ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return ActionOutcomeChoice(False, "unset")
    flag = getattr(args, "action_outcome", None)
    if flag is not None:
        return ActionOutcomeChoice(bool(flag), "flag")
    if profile.action_outcome is not None:
        return ActionOutcomeChoice(profile.action_outcome, "profile")
    return ActionOutcomeChoice(LIVE_DEFAULT_ACTION_OUTCOME, "default")


def action_outcome_flag_refusal(
    profile: Profile, args: argparse.Namespace, mode: str
) -> str | None:
    """T-0136 C-3: why ``--action-outcome``/``--no-action-outcome`` in a resolved mode other than ``live`` is refused,
    or ``None``; so is a live profile's ``action_outcome`` key when ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    flag = getattr(args, "action_outcome", None)
    given = [
        what
        for what, present in (
            ("--action-outcome" if flag else "--no-action-outcome", flag is not None),
            ("the profile's action_outcome", profile.action_outcome is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the action outcome applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0136)"
    )


def action_outcome_with_feedback_off_refusal(
    outcome: ActionOutcomeChoice, feedback: Feedback
) -> str | None:
    """T-0136 C-3: why the action outcome on together with resolved live feedback ``off``, from any source, is
    refused, or ``None``; the message names both (the clause lives in the label, which ``off`` removes)."""
    if not (outcome.action_outcome and feedback.live_feedback == "off"):
        return None
    return (
        f"the action outcome ({outcome.action_outcome_from}) is on with the live feedback off "
        f"({feedback.live_feedback_from}), which is refused: the outcome clause is part of the live feedback "
        "label, which off removes; use --live-feedback label or label+extent, or turn the action outcome off "
        "(T-0136)"
    )


@dataclass(frozen=True)
class RepeatGuard:
    """T-0137 B-9: whether the live session states repeats and refuses no-progress actions, with its source (the
    ``ActionOutcomeChoice`` shape). ``False`` from ``"unset"`` outside live."""

    repeat_guard: bool
    repeat_guard_from: GroundingSource

    def line(self) -> str:
        """B-9: the live-only ``repeat guard`` line, printed right after the ``action outcome`` line."""
        return f"astra session: repeat guard: {'on' if self.repeat_guard else 'off'} ({self.repeat_guard_from})"


def repeat_guard_of(profile: Profile, args: argparse.Namespace, mode: str) -> RepeatGuard:
    """T-0137 B-9: the one resolver of the repeat guard, keyed on the *resolved* tools mode. In ``live``: the flag
    (``--repeat-guard``/``--no-repeat-guard``), else the profile key, else ``LIVE_DEFAULT_REPEAT_GUARD`` (read from
    this module's namespace at call time). Elsewhere off from ``"unset"`` (a flag there is refused in
    ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return RepeatGuard(False, "unset")
    flag = getattr(args, "repeat_guard", None)
    if flag is not None:
        return RepeatGuard(bool(flag), "flag")
    if profile.repeat_guard is not None:
        return RepeatGuard(profile.repeat_guard, "profile")
    return RepeatGuard(LIVE_DEFAULT_REPEAT_GUARD, "default")


def repeat_guard_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0137 B-9: why ``--repeat-guard``/``--no-repeat-guard`` in a resolved mode other than ``live`` is refused, or
    ``None``; so is a live profile's ``repeat_guard`` key when ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    flag = getattr(args, "repeat_guard", None)
    given = [
        what
        for what, present in (
            ("--repeat-guard" if flag else "--no-repeat-guard", flag is not None),
            ("the profile's repeat_guard", profile.repeat_guard is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the repeat guard applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0137)"
    )


@dataclass(frozen=True)
class Reask:
    """T-0172 B9: whether a tools-on session re-asks once when a step only announces an action and makes no call,
    with its source (the ``Recovery`` shape). ``None`` from ``"unset"`` with tools off."""

    value: bool | None
    source: GroundingSource

    def line(self) -> str:
        """B9: the ``reask`` line, the last tools-on startup line (dry_run and live)."""
        return f"astra session: reask: announced_call_reask={'on' if self.value else 'off'} ({self.source})"


def reask_of(profile: Profile, args: argparse.Namespace, mode: str) -> Reask:
    """T-0172 B9: the one resolver of the announced-call re-ask, keyed on the *resolved* tools mode. In ``dry_run``
    and ``live``: the flag (``on``/``off``), else the profile key, else ``DEFAULT_ANNOUNCED_CALL_REASK``. With
    tools off ``None`` from ``"unset"`` (a flag or key there is refused in ``session_cmd``). Pure: reads ``args``
    with ``getattr``, so an empty ``Namespace()`` resolves the default."""
    if mode not in ("dry_run", "live"):
        return Reask(None, "unset")
    flag = getattr(args, "announced_call_reask", None)
    if flag is not None:
        return Reask(flag == "on", "flag")
    if profile.announced_call_reask is not None:
        return Reask(profile.announced_call_reask, "profile")
    return Reask(DEFAULT_ANNOUNCED_CALL_REASK, "default")


def reask_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0172 B9: why ``--announced-call-reask`` with tools off is refused, or ``None``; there it would silently do
    nothing. So is a profile's ``announced_call_reask`` key when ``--tools off`` overrode its mode."""
    if mode in ("dry_run", "live"):
        return None
    given = [
        what
        for what, present in (
            ("--announced-call-reask", getattr(args, "announced_call_reask", None) is not None),
            ("the profile's announced_call_reask", profile.announced_call_reask is not None),
        )
        if present
    ]
    if not given:
        return None
    return (
        f"--tools off with {' and '.join(given)} is refused: the announced-call re-ask applies only with "
        "--tools dry-run or live, and with tools off it would silently do nothing (T-0172)"
    )


@dataclass(frozen=True)
class LiveGuidance:
    """T-0174 A4: which live tool guidance version a live session's system message carries, with its source (the
    ``Reask`` shape). ``None`` from ``"unset"`` outside live."""

    value: str | None
    source: GroundingSource

    def line(self) -> str:
        """A4: printed only for a flag or profile source, immediately before the reask line (live only)."""
        return f"astra session: live guidance: {self.value} ({self.source})"


def live_guidance_of(profile: Profile, args: argparse.Namespace, mode: str) -> LiveGuidance:
    """T-0174 A4: the one resolver of the live guidance version, keyed on the *resolved* tools mode. In ``live``: the
    flag, else the profile key, else ``DEFAULT_LIVE_GUIDANCE``. Elsewhere ``(None, "unset")`` (a flag or key there is
    refused in ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return LiveGuidance(None, "unset")
    flag = getattr(args, "live_guidance", None)
    if flag is not None:
        return LiveGuidance(flag, "flag")
    if profile.live_guidance is not None:
        return LiveGuidance(profile.live_guidance, "profile")
    return LiveGuidance(DEFAULT_LIVE_GUIDANCE, "default")


#: T-0174 B2-1 (H-T174-2), T-0191 A4, T-0211 B4: the live guidance versions the page fact rides with
PAGE_FACT_GUIDANCE: tuple[str, ...] = ("v2", "v3", "v4")


def guidance_table_of(
    web: bool, wait_tool: bool, search: bool = True
) -> dict[str, tuple[str | None, bool]]:
    """T-0213 A2a-4: per live guidance version, the session's tool-guidance text (``_tool_guidance`` with this
    session's fixed web mode and wait tool) and whether the page fact rides with it, so the Settings panel can
    change the version at a turn boundary without the session importing ``cli``. T-0215: ``search=False`` (no
    ``google_search`` declared) gives every value without the search sentence."""
    # the keyword only when off, so a search-on table calls ``_tool_guidance`` exactly as at BASE (the T-0211 C
    # harness rebinds it with BASE's three-argument signature)
    off: dict[str, bool] = {} if search else {"search": False}
    return {
        version: (_tool_guidance(version, web, wait_tool, **off), version in PAGE_FACT_GUIDANCE)
        for version in LIVE_GUIDANCE_VERSIONS
    }


def _tool_guidance(
    version: str | None, web: bool, wait_tool: bool, search: bool = True
) -> str | None:
    """T-0202 B8: ``live_guidance_text``, with H-T202-4's sentence appended when the wait tool is on (live only).
    T-0215: with ``search=False`` v4 is v3 + S_SHARED (S_SOURCE names ``google_search``, which is not declared)."""
    text = live_guidance_text(version, web)
    if not search and text is LIVE_GUIDANCE["v4"]:
        text = LIVE_TOOL_GUIDANCE_V4_NOSEARCH
    return with_wait_guidance(text) if wait_tool and text is not None else text


def _kit_declares_search(kit: ToolKit | None) -> bool:
    """T-0215: ``False`` only for a kit whose judged set leaves out ``google_search`` (a live session with no search
    backend), so the guidance follows what the kit declares; every other kit (BASE's included) is ``True``."""
    return kit is None or kit.tool_set is None or "google_search" in kit.tool_set.names


def live_guidance_text(version: str | None, web: bool) -> str | None:
    """T-0191 A4: the guidance text for a resolved version, chosen once at session construction. v3's last sentence
    teaches Ctrl+L, which is refused outside web mode, so v3 without web is v2's object itself (not an equal copy);
    so is v4 without web (T-0211 B4). ``None`` (outside live) stays ``None``."""
    if version is None:
        return None
    if version in ("v3", "v4") and not web:
        return LIVE_GUIDANCE["v2"]
    return LIVE_GUIDANCE[version]


#: T-0202 B8: the token the ``--dry-run`` tools line and the real start's regime line carry with the wait tool on
WAIT_TOOL_TOKEN = " wait=on(flag)"
#: T-0202 D1 (D-R1): the same slot's token when the wait tool is on by the live default (no ``--wait-tool``)
WAIT_TOOL_DEFAULT_TOKEN = " wait=on(default)"

#: T-0202 D1: the wait tool is on by default in live (C's GPU A/B PASSED 2026-10-30 at eebbbf8b)
LIVE_DEFAULT_WAIT_TOOL = True


def wait_tool_of(args: argparse.Namespace, mode: str) -> bool:
    """T-0202 B8/D1 (H-T202-7: the flag only, no profile key): whether the live kit offers ``wait``; on by default in
    live (``--wait-tool off`` turns it off), ``False`` outside live (a flag there is refused in ``session_cmd``).
    Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return False
    flag = getattr(args, "wait_tool", None)
    return flag == "on" if flag is not None else LIVE_DEFAULT_WAIT_TOOL


def wait_tool_token(args: argparse.Namespace, mode: str) -> str:
    """T-0202 D1 (D-R1): the start-line token: `` wait=on(flag)`` with ``--wait-tool on``, `` wait=on(default)``
    when on by the live default, and nothing when off or outside live."""
    if not wait_tool_of(args, mode):
        return ""
    return WAIT_TOOL_TOKEN if getattr(args, "wait_tool", None) == "on" else WAIT_TOOL_DEFAULT_TOKEN


def wait_tool_source(args: argparse.Namespace, mode: str) -> GroundingSource:
    """T-0213 A2r-2: where ``wait_tool_of``'s value came from: ``flag`` with ``--wait-tool``, else ``default`` in
    live, ``unset`` elsewhere."""
    if mode != "live":
        return "unset"
    return "flag" if getattr(args, "wait_tool", None) is not None else "default"


def wait_tool_flag_refusal(args: argparse.Namespace, mode: str) -> str | None:
    """T-0202 B8: why ``--wait-tool`` in a resolved mode other than ``live`` is refused, or ``None``."""
    if mode == "live" or getattr(args, "wait_tool", None) is None:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with --wait-tool is refused: the wait tool applies only with --tools live, and with "
        f"{spelled} it would silently do nothing (T-0202)"
    )


#: T-0201 D1: the screen reminder is on by default in live (C's GPU A/B PASSED 2026-10-29 at 25b7bb56)
LIVE_DEFAULT_SCREEN_REMINDER = True


@dataclass(frozen=True)
class ScreenReminder:
    """T-0201 D1: whether a live session's labelled steps carry the screen reminder, with its source (``flag`` or
    ``default``; R-D3: no profile key). ``None`` from ``"unset"`` outside live."""

    screen_reminder: bool | None
    source: GroundingSource

    def line(self) -> str:
        """R-D2: the live-only start line, after the ``grounding:`` block and before ``search:``."""
        word = "on" if self.screen_reminder else "off"
        return f"astra session: screen reminder: screen_reminder={word} ({self.source})"


def screen_reminder_of(args: argparse.Namespace, mode: str) -> ScreenReminder:
    """T-0201 D1: the one resolver of the screen reminder, keyed on the *resolved* tools mode. In ``live``: the flag
    (``on``/``off``), else ``LIVE_DEFAULT_SCREEN_REMINDER``. Elsewhere ``None`` from ``"unset"`` (a flag there is
    refused in ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return ScreenReminder(None, "unset")
    flag = getattr(args, "screen_reminder", None)
    if flag is not None:
        return ScreenReminder(flag == "on", "flag")
    return ScreenReminder(LIVE_DEFAULT_SCREEN_REMINDER, "default")


def screen_reminder_flag_refusal(args: argparse.Namespace, mode: str) -> str | None:
    """T-0201 D1 (R-D4): why ``--screen-reminder`` in a resolved mode other than ``live`` is refused, or ``None``."""
    if mode == "live" or getattr(args, "screen_reminder", None) is None:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with --screen-reminder is refused: the screen reminder applies only with --tools live, "
        f"and with {spelled} it would silently do nothing (T-0201)"
    )


#: T-0207 C1: the completion check is on by default in live (B's GPU A/B PASSED 2026-10-30 at cb016fd8)
LIVE_DEFAULT_COMPLETION_CHECK = True


def completion_check_of(args: argparse.Namespace, mode: str) -> bool:
    """T-0207 C1 (R-C6; R-C2: the flag only, no profile key): whether a live session asks the completion check before
    voicing a final answer; on by default in live (``--completion-check off`` turns it off), ``False`` outside live (a
    flag there is refused in ``session_cmd``). Pure: reads ``args`` with ``getattr``."""
    if mode != "live":
        return False
    flag = getattr(args, "completion_check", None)
    return flag == "on" if flag is not None else LIVE_DEFAULT_COMPLETION_CHECK


def completion_check_line(args: argparse.Namespace, mode: str) -> str:
    """T-0207 C1 (R-C1): the live-only start line, after the screen reminder line and before ``search:``."""
    word = "on" if completion_check_of(args, mode) else "off"
    return f"astra session: completion check: {word} ({completion_check_source(args, mode)})"


def completion_check_source(args: argparse.Namespace, mode: str) -> GroundingSource:
    """T-0207 C1, T-0213 A2r-2: where ``completion_check_of``'s value came from: ``flag`` with
    ``--completion-check``, else ``default`` in live, ``unset`` elsewhere (the value is ``False`` from no setting).
    The start line states the live answer."""
    if mode != "live":
        return "unset"
    return "flag" if getattr(args, "completion_check", None) is not None else "default"


def completion_check_flag_refusal(args: argparse.Namespace, mode: str) -> str | None:
    """T-0207 C1: why ``--completion-check`` in a resolved mode other than ``live`` is refused, or ``None``."""
    if mode == "live" or getattr(args, "completion_check", None) is None:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with --completion-check is refused: the completion check applies only with --tools "
        f"live, and with {spelled} it would silently do nothing (T-0207)"
    )


def live_guidance_flag_refusal(profile: Profile, args: argparse.Namespace, mode: str) -> str | None:
    """T-0174 A4: why ``--live-guidance`` in a resolved mode other than ``live`` is refused, or ``None``; so is a live
    profile's ``live_guidance`` key when ``--tools`` overrode its mode."""
    if mode == "live":
        return None
    given = [
        what
        for what, present in (
            ("--live-guidance", getattr(args, "live_guidance", None) is not None),
            ("the profile's live_guidance", profile.live_guidance is not None),
        )
        if present
    ]
    if not given:
        return None
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with {' and '.join(given)} is refused: the live guidance applies only with "
        f"--tools live, and with {spelled} it would silently do nothing (T-0174)"
    )


#: T-0125 criterion 3: the one stderr line a session with web search on prints at a real start; ``{engines}`` is the
#: label ``_search_backend`` returns, read from the backend's own engine list (never retyped here)
SEARCH_STARTUP_LINE = (
    "astra session: web search is on ({engines}): each search query the model makes is sent off this machine "
    "and logged in the session log; results are untrusted web text; --search off turns it off"
)
#: T-0125 criterion 2: the fix for a missing ``search`` extra
SEARCH_EXTRA_FIX = "uv sync --all-extras"


@dataclass(frozen=True)
class WebSearch:
    """T-0125 criterion 2: the session's resolved web search. ``backend`` is the backend's name when on (``"ddgs"``),
    ``None`` when off; ``source`` is ``flag``, ``profile`` or ``default``; ``refusal`` why startup is refused (exit
    2), or ``None``. ``shown``: the ``search:`` line is printed (live always, dry-run only when on, never off)."""

    backend: Literal["ddgs"] | None
    source: Literal["flag", "profile", "default"]
    refusal: str | None
    shown: bool

    def asked(self) -> str:
        """Who asked for this backend, in a refusal: the flag, or the profile's setting."""
        if self.source == "flag":
            return f"--search {self.backend}"
        return f"the profile's search = {self.backend!r}"

    def line(self, degraded: str | None = None) -> str:
        """Criterion 3: ``astra session: search: <ddgs|off> (<source>)``, or ``search: off (default: <why>)`` when a
        default-on start degraded."""
        if degraded is not None:
            return f"astra session: search: off (default: {degraded})"
        return f"astra session: search: {self.backend or 'off'} ({self.source})"


def search_of(profile: Profile, args: argparse.Namespace, mode: str) -> WebSearch:
    """T-0125 criterion 2: the one resolver of web search, keyed on the *resolved* tools mode. The flag beats the
    profile, which beats the mode's default (on in live, off in dry-run, never with tools off). Refused: ``ddgs``
    with tools off, and ``ddgs`` in dry-run with the tool loop off (the backend answers only inside the loop).
    Pure: reads ``args`` with ``getattr``; imports nothing of the backend."""
    flag = getattr(args, "search", None)
    source: Literal["flag", "profile", "default"]
    if flag is not None:
        asked, source = flag, "flag"
    elif profile.search is not None:
        asked, source = profile.search, "profile"
    else:
        asked, source = None, "default"
    if mode not in ("dry_run", "live"):
        refusal = None
        if asked == "ddgs":
            given = "--search ddgs" if source == "flag" else "the profile's search = 'ddgs'"
            refusal = (
                f"{given} is refused with tools off: web search answers the model's google_search calls, "
                "which exist only with tools on; add `--tools live` or `--tools dry-run` (T-0125)"
            )
        return WebSearch(None, source, refusal, False)
    on = asked != "off" if mode == "live" else asked == "ddgs"
    refusal = None
    if on and mode == "dry_run" and not tool_loop_of(profile, args).on:
        given = "--search ddgs" if source == "flag" else "the profile's search = 'ddgs'"
        refusal = (
            f"{given} is refused with the tool loop off (--no-tool-loop or tool_loop = false): search results "
            "reach the model only through the loop; remove `--no-tool-loop` (T-0125)"
        )
    return WebSearch("ddgs" if on else None, source, refusal, mode == "live" or on)


def search_start(backend: str) -> tuple[tuple[SearchBackend, str] | None, tuple[str, str] | None]:
    """T-0125 criterion 2: the real-start prerequisite, never under ``--dry-run``: ``((backend, label), None)`` when
    the backend can be constructed, else ``(None, (why, fix))``. The construction is the check (``ddgs`` importable;
    no proxy in the environment, ``ValueError``; ddgs internals where the pin has them, ``RuntimeError``), so the
    backend is built once, through ``_search_backend``, and handed to ``build_session`` (``args.search_built``).
    Builds clients only; opens no connection."""
    try:
        return _search_backend(backend), None
    except (ImportError, ValueError, RuntimeError) as exc:
        return None, _search_failure_words(exc)


def _search_backend(name: str) -> tuple[SearchBackend, str]:
    """T-0125 criterion 3: the web search backend named ``name`` and its engine label for the startup line. Called
    only by ``build_session``; the only place outside ``local_astra/search/`` that imports or constructs one, so
    ``ddgs`` is never loaded unless search is on. Raises ``ImportError`` (the extra is missing), ``ValueError`` (a
    proxy in the environment) or ``RuntimeError`` (ddgs internals moved)."""
    if name != "ddgs":
        raise ValueError(f"unknown search backend {name!r}")
    from local_astra.search.brave_key import load_key
    from local_astra.search.ddgs_backend import ENGINES, DdgsSearch

    keyless = DdgsSearch()
    label = "keyless: " + ", then ".join(ENGINES)
    # T-0163: the Brave Search API key file, read once, here. No file: the bare keyless backend, as before.
    loaded = load_key()
    if loaded.key is None:
        if loaded.reason is not None:
            fix = _BRAVE_KEY_FIXES.get(loaded.reason, "fix the file or delete it").format(
                path=loaded.path
            )
            _SEARCH_FALLBACK[keyless] = (
                f"{BRAVE_KEY_REFUSED_PREFIX}{loaded.message}; {fix}; web search stays keyless",
                {
                    "what": "web search fallback",
                    "brave_api": "not configured",
                    "why": loaded.reason,
                },
            )
        return keyless, label
    from local_astra.search.brave_api import BraveApiSearch
    from local_astra.search.fallback import FallbackSearch

    chain = FallbackSearch(keyless, BraveApiSearch(loaded.key))
    _SEARCH_FALLBACK[chain] = (None, {"what": "web search fallback", "brave_api": "configured"})
    return chain, label + BRAVE_LABEL_SUFFIX


#: T-0163 criterion 8: the label suffix of a configured fallback, and the start of the refused-key stderr line
BRAVE_LABEL_SUFFIX = "; then the Brave Search API (key configured)"
BRAVE_KEY_REFUSED_PREFIX = "astra session: the Brave Search API key file is not used: "
#: T-0163 criterion 8: the fix per refusal code of the key loader (``{path}`` is the key file)
_BRAVE_KEY_FIXES = {
    "mode": "chmod 600 {path}",
    "unreadable": "chmod 600 {path}",
    "owner": "recreate {path} as yourself",
    "symlink": "replace the link {path} with the file itself",
    "not_a_file": "make {path} a regular file holding only the key",
    "malformed": "put only the key, on one line, in {path}",
}
#: T-0163 criterion 8: per backend ``_search_backend`` built, the refused-key stderr line (or ``None``) and the
#: session-log note, said once by ``run_session``; weak, so a closed backend leaves nothing behind
_SEARCH_FALLBACK: weakref.WeakKeyDictionary[object, tuple[str | None, dict[str, str]]] = (
    weakref.WeakKeyDictionary()
)


def _search_fallback_of(search: object) -> tuple[str | None, dict[str, str]] | None:
    """The refused-key line and fallback note ``_search_backend`` kept for ``search``, or ``None`` (no key file, or
    a backend it did not build)."""
    for built, said in list(_SEARCH_FALLBACK.items()):
        if built is search:
            return said
    return None


def visual_tokens_suffix(budget: int | None) -> str:
    """T-0131 A3: appended to the ``--dry-run`` tools line in every mode when a budget is set; empty when unset.
    Since T-0183 ``grounding_of`` always resolves a budget, so ``session_cmd`` always prints it (H-T183-2); the
    empty case remains for a ``None`` passed directly."""
    return f" visual_tokens={budget}" if budget is not None else ""


def reground_suffix(on: bool) -> str:
    """T-0131 B9: appended to the ``--dry-run`` tools line when ReGround is on (live only); empty when off, so
    that line stays the A/C state's byte for byte."""
    return " reground=on" if on else ""


def reground_refusal(mode: str) -> str:
    """T-0131 B9 (ruling, 21361bd): ReGround resolved on with a tools mode other than ``live`` is refused, whether
    it came from ``--reground`` or from the profile; it would otherwise silently do nothing."""
    spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
    return (
        f"--tools {spelled} with ReGround on is refused: ReGround runs only with --tools live, and with "
        f"{spelled} it would silently do nothing; use --tools live or --no-reground (T-0131)"
    )


def reground_visual_tokens_cli_refusal(grounding: Grounding, mode: str) -> str | None:
    """T-0132 B1: why a pass-2 budget that was *set* (flag or profile) is refused, or ``None``. It does something
    only with ReGround on in live; anywhere else it would silently do nothing."""
    if grounding.reground_visual_tokens_from not in ("flag", "profile"):
        return None
    if mode != "live":
        spelled = {"off": "off", "dry_run": "dry-run"}.get(mode, mode)
        return (
            f"--tools {spelled} with reground_visual_tokens set is refused: pass 2 runs only with --tools live "
            "and ReGround on, and here the budget would silently do nothing (T-0132)"
        )
    if not grounding.reground:
        return (
            "reground_visual_tokens with ReGround off is refused: there is no pass 2 to send it, so it would "
            "silently do nothing; drop it or turn ReGround on (T-0132)"
        )
    return None


def live_prerequisites(browser: str) -> str | None:
    """T-0111g B2: why ``live`` cannot start on this box, or ``None``. Reads the filesystem and the environment
    (``preflight.collect_browser``); it never imports Playwright or launches a process."""
    facts = preflight.collect_browser()["browser"]
    if facts.get("playwright") is None:
        return (
            "--tools live refused: playwright is not importable (prerequisite P1, T-0111c); "
            f"run `{preflight.BROWSER_FIX}`"
        )
    if preflight.pinned_executable(browser) is None:
        # B-F2: the executable this mode actually launches (headless runs Playwright's headless shell)
        what = "Chromium" if browser == "headed" else "Chromium headless shell"
        return (
            f"--tools live refused: the pinned {what} that --browser {browser} launches is not installed "
            f"(prerequisite P1, T-0111c); run `{preflight.BROWSER_FIX}`"
        )
    if browser == "headed" and facts.get("display") is None:
        return (
            "--tools live refused: a headed browser needs a display and neither DISPLAY nor "
            "WAYLAND_DISPLAY is set; use --browser headless"
        )
    return None


def max_tool_steps_arg(value: str) -> int:
    """``--max-tool-steps N``: a whole number >= 1 (T-0114 criterion 14); anything else is an argv error."""
    try:
        steps = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if steps < 1:
        raise argparse.ArgumentTypeError(
            f"{steps} is below 1: the loop executes at least one call per turn"
        )
    return steps


@dataclass(frozen=True)
class ToolLoop:
    """The session's resolved tool-loop setting (T-0123 criterion 2): what ``build_session`` passes to
    ``TurnPolicy``, what the ``--dry-run`` tools line states, and why startup is refused (or ``None``)."""

    on: bool
    max_tool_steps: int
    refusal: str | None
    #: T-0178 A4: where ``max_tool_steps`` came from (``--max-tool-steps``, ``max_tool_steps`` or
    #: ``max_tool_steps_web``) and the mode's default (8, or 20 in live with web on)
    source: str = "max_tool_steps"
    mode_default: int = 8

    def line(self) -> str | None:
        """T-0178 A4: ``astra session: tool loop: <N> steps (<source>)`` only when the loop is on and N is not the
        mode's default; ``None`` otherwise, so a default start prints what BASE printed."""
        if not self.on or self.max_tool_steps == self.mode_default:
            return None
        return f"astra session: tool loop: {self.max_tool_steps} steps ({self.source})"


#: T-0178 A1: the defaults of ``Profile.max_tool_steps`` and ``Profile.max_tool_steps_web``
DEFAULT_TOOL_STEPS = 8
DEFAULT_TOOL_STEPS_WEB = 20


def tool_loop_of(profile: Profile, args: argparse.Namespace) -> ToolLoop:
    """T-0114/T-0123: the one resolver for the loop. ``req`` is ``--tool-loop``/``--no-tool-loop`` when given,
    else ``Profile.tool_loop`` (``None`` when unset); the steps are ``--max-tool-steps`` when given, else the
    profile's (as ``--tools``, ruling H-A4). With the resolved tools ``dry_run`` the loop is ``req``, or on when
    ``req`` is ``None``; with tools off it is off.

    Refused at startup (H-T114-2: refused, never silently inert): an explicit loop request with tools off (it
    has nothing to execute); an explicit ``--max-tool-steps`` with tools off, or with tools on and the loop off
    by explicit choice (it would be carried and ignored). A profile's own ``max_tool_steps`` is never refused:
    the field always has a value.

    T-0178 A1 (H-T178-2): in live with web on (``web_of``), the steps are ``--max-tool-steps`` when given, else
    ``Profile.max_tool_steps_web`` (never ``max_tool_steps``); every other mode is as above."""
    flag = getattr(args, "tool_loop", None)
    req: bool | None = bool(flag) if flag is not None else profile.tool_loop
    mode = tools_mode_of(profile, args)
    web = web_of(profile, args, mode)
    steps_flag = getattr(args, "max_tool_steps", None)
    if steps_flag is not None:
        steps, source = int(steps_flag), "--max-tool-steps"
    elif web:
        steps, source = profile.max_tool_steps_web, "max_tool_steps_web"
    else:
        steps, source = profile.max_tool_steps, "max_tool_steps"
    mode_default = DEFAULT_TOOL_STEPS_WEB if web else DEFAULT_TOOL_STEPS
    tools_on = mode in ("dry_run", "live")
    on = (req if req is not None else True) if tools_on else False
    refusal: str | None = None
    if mode == "live" and req is False:
        # T-0111g B2 (R-c): the screen after an action reaches the model only through the loop
        refusal = (
            "--tools live is refused with the tool loop off (--no-tool-loop or tool_loop = false): "
            "live needs the loop, which is how the screen after an action reaches the model; drop "
            "--no-tool-loop, or use --tools dry-run"
        )
    elif req is True and not tools_on:
        refusal = (
            "the tool loop is on (--tool-loop or tool_loop = true) but tools are off, so it has "
            "nothing to execute; add --tools dry-run, or turn the loop off with --no-tool-loop"
        )
    elif steps_flag is not None and not tools_on:
        refusal = (
            "--max-tool-steps is refused with tools off (the loop runs only with tools on, so it "
            "would be ignored); add --tools dry-run, or drop --max-tool-steps"
        )
    elif steps_flag is not None and not on:
        refusal = (
            "--max-tool-steps is refused with the tool loop off (it would be ignored); add "
            "--tool-loop, or drop --max-tool-steps"
        )
    return ToolLoop(
        on=on,
        max_tool_steps=steps,
        refusal=refusal,
        source=source,
        mode_default=mode_default,
    )


class ToolsUnavailable(RuntimeError):
    """``dry_run`` was asked for and the host detokeniser could not be loaded (T-0092b criterion 2(a))."""


def _repair_off_reason(
    profile: Profile,
) -> tuple[str | None, live.Payloads | None, live.Calibration | None]:
    """Criterion 2(b)-(c): the payloads and the calibration, checked; or why the repair must stay off.

    Every failure is a reason, never an exception: a stale payload turns the repair off for the session and
    the session still starts.
    """
    try:
        payloads = live.load_payloads()
    except live.PayloadStale as exc:
        return str(exc), None, None
    manifest = payloads.manifest
    lock_path = SERVICES_DIR / "versions.lock"  # read at call time (ruling H-A3)
    try:
        with lock_path.open("rb") as handle:
            pinned = tomllib.load(handle)["gpu"]["model_revision"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{lock_path}: cannot read [gpu].model_revision ({exc!r})", None, None
    if manifest.get("model_revision") != pinned:
        return (
            f"payload manifest model_revision {manifest.get('model_revision')!r} is not "
            f"{lock_path}'s {pinned!r}",
            None,
            None,
        )
    if manifest.get("tokenizer_sha256") != detok.TOKENIZER_SHA256:
        return (
            f"payload manifest tokenizer_sha256 {manifest.get('tokenizer_sha256')!r} is not the "
            f"detokeniser's pinned {detok.TOKENIZER_SHA256!r}",
            None,
            None,
        )
    canvas = profile.diffusion.canvas_length
    if manifest.get("canvas_length") != canvas:
        return (
            f"payload manifest canvas_length {manifest.get('canvas_length')!r} is not the "
            f"profile's {canvas}",
            None,
            None,
        )
    try:
        calibration = live.load_calibration()
    except live.PayloadStale as exc:
        return str(exc), None, None
    return None, payloads, calibration


#: T-0215: per ``(page_overview, wait_tool)``, the live tool set without ``google_search`` (search off)
_NOSEARCH_PATHS: dict[tuple[bool, bool], Path] = {
    (False, False): toolset.TOOLS_V1_NOSEARCH_PATH,
    (True, False): toolset.TOOLS_V1_OVERVIEW_NOSEARCH_PATH,
    (False, True): toolset.TOOLS_V1_WAIT_NOSEARCH_PATH,
    (True, True): toolset.TOOLS_V1_OVERVIEW_WAIT_NOSEARCH_PATH,
}


def load_tool_kit(
    profile: Profile,
    *,
    live_browser: LiveBrowser | None = None,
    page_overview: bool = False,
    wait_tool: bool = False,
    search: bool = True,
) -> tuple[ToolKit, detok.HostDetokenizer]:
    """Everything a ``dry_run`` session needs, loaded **before** the event loop runs (T-0092b criterion 2).

    (a) the host detokeniser — any failure is ``ToolsUnavailable`` naming the file, and no session starts;
    (b)-(c) the payloads and the calibration, checked against versions.lock, the detokeniser's pinned sha and
    the profile's canvas — any failure turns the repair off with its reason, announced in one stderr line.

    T-0111g B5: with ``live_browser`` the kit is ``live``, and that browser is its executor too.
    T-0140 A5 (R-p): with ``page_overview`` the tools, their sha and the judge's set are ``tools_v1_overview.json``'s
    (``cli`` passes it only in live, and only when on).
    T-0202 B8: with ``wait_tool`` they are ``tools_v1_wait.json``'s, or ``tools_v1_overview_wait.json``'s with
    ``page_overview`` too (``cli`` passes it only in live, and only when on).
    T-0215: with ``search=False`` they are the matching ``*_nosearch.json`` set's, without ``google_search``, and
    the kit always carries that set for its judge (``cli`` passes it only in live with no search backend, so a
    search-on call is BASE's).
    """
    mode = "dry_run" if live_browser is None else "live"
    try:
        detokenizer = detok.HostDetokenizer.load()
    except Exception as exc:
        raise ToolsUnavailable(
            f"--tools {mode} refused: cannot load the host detokeniser from "
            f"{detok.pinned_tokenizer_path()}: {exc}"
        ) from exc
    reason, payloads, calibration = _repair_off_reason(profile)
    if reason is not None:
        print(f"astra session: tool repair off for this session: {reason}", file=sys.stderr)
    # T-0092c (carried from the T-0092b review): the tools and their sha come from one ``ToolSet``.
    if not search:
        # T-0215: the live set without ``google_search``; its sha is the version, and the judge reads it
        tool_set = toolset.load(_NOSEARCH_PATHS[(page_overview, wait_tool)])
    elif wait_tool:
        tool_set = toolset.load(
            toolset.TOOLS_V1_OVERVIEW_WAIT_PATH if page_overview else toolset.TOOLS_V1_WAIT_PATH
        )
    else:
        tool_set = toolset.load(toolset.TOOLS_V1_OVERVIEW_PATH) if page_overview else toolset.load()
    # T-0140 A6: the kit carries the set its judge reads only when it is not v1 (``None`` is v1, BASE's kit)
    judged = tool_set if page_overview or wait_tool or not search else None
    if live_browser is None:
        kit = ToolKit(
            mode="dry_run",
            tools=tuple(native.tools_field(tool_set)),
            tools_version=tool_set.version,
            executor=DryRunToolExecutor(),
            payloads=payloads,
            calibration=calibration,
            repair_off=reason,
            tool_set=judged,
        )
    else:
        kit = ToolKit(
            mode="live",
            tools=tuple(native.tools_field(tool_set)),
            tools_version=tool_set.version,
            executor=live_browser,
            payloads=payloads,
            calibration=calibration,
            repair_off=reason,
            live_browser=live_browser,
            tool_set=judged,
        )
    return kit, detokenizer


@dataclass(frozen=True)
class LiveLine:
    """T-0111g B3: what the ``--dry-run`` line states about ``live``'s browser."""

    browser: str
    viewport: tuple[int, int]
    #: T-0160 C-2: web mode prints ``egress=web`` instead of ``egress=loopback``
    web: bool = False


def tools_line(
    kit: ToolKit | None, loop_max: int | None = None, live_line: LiveLine | None = None
) -> str:
    """The ``--dry-run`` line: ``tools=off`` or ``tools=dry_run native:<sha12> repair=on|off(<reason>)``, plus
    `` loop=on(<max>)`` when the tool loop is on (T-0114 criterion 14). T-0111g B3: with ``live_line`` it reads
    ``tools=live ... browser=<headed|headless> egress=loopback viewport=<W>x<H>`` (T-0160 C-2: ``egress=web`` in
    web mode)."""
    if kit is None:
        return "tools=off"
    repair = "on" if kit.repair_off is None else f"off({kit.repair_off})"
    loop = f" loop=on({loop_max})" if loop_max is not None else ""
    if live_line is None:
        return f"tools={kit.mode} native:{kit.tools_version[:12]} repair={repair}{loop}"
    width, height = live_line.viewport
    egress = "web" if live_line.web else "loopback"
    return (
        f"tools=live native:{kit.tools_version[:12]} repair={repair}{loop} "
        f"browser={live_line.browser} egress={egress} viewport={width}x{height}"
    )


def build_session(config: Config, args: argparse.Namespace) -> SessionGraph:
    """The composition root. Every endpoint comes from ``config.Services`` (docs/02 §9).

    The capture source is resolved **here**, once, and then used for both the device this graph
    is opened on and the ``mic=`` its description states, so the two can never disagree.

    T-0092b: with ``tools = dry_run`` the tool kit is loaded first — here, before ``asyncio.run``, so never
    on the event loop — and a detokeniser that cannot load raises ``ToolsUnavailable`` before anything else
    is constructed. With ``off`` nothing of it is loaded or constructed.
    """
    profile = config.profiles[args.profile] if args.profile else config.active()
    kit: ToolKit | None = None
    detokenizer: detok.HostDetokenizer | None = None
    pages: Any = None
    clock = SessionClock()
    mode = tools_mode_of(profile, args)
    # T-0125 criterion 3: the backend first, so an explicit search that cannot start is refused before anything
    # else is built; ``search_off`` is ``session_cmd``'s degrade of a default-on search (its prerequisites failed)
    web = search_of(profile, args, mode)
    search: SearchBackend | None = None
    label: str | None = None
    search_off: str | None = getattr(args, "search_off", None)
    built = getattr(args, "search_built", None)
    if web.backend is not None and search_off is None and built is not None:
        search, label = built  # ``session_cmd``'s real-start check built it
    elif web.backend is not None and search_off is None:
        try:
            search, label = _search_backend(web.backend)
        except (ImportError, ValueError, RuntimeError) as exc:
            why, fix = _search_failure_words(exc)
            if web.source != "default":
                raise SearchUnavailable(f"{web.asked()} cannot start: {why}; {fix}") from exc
            search_off = why
            print(_search_degraded_line(why, fix), file=sys.stderr)
    if mode == "dry_run":
        kit, detokenizer = load_tool_kit(profile)
    elif mode == "live":
        pages, browser = _start_live(profile, args, clock)
        # T-0140 A5: the keyword only when on, so an off live build calls the loader exactly as at BASE
        overview: dict[str, bool] = (
            {"page_overview": True} if page_overview_of(profile, args, mode).page_overview else {}
        )
        if wait_tool_of(args, mode):
            overview["wait_tool"] = True  # T-0202 B8: the keyword only when on, as page_overview
        if search is None:
            # T-0215: no backend (flag, profile or degrade): google_search withdrawn, the keyword only when off
            overview["search"] = False
        try:
            kit, detokenizer = load_tool_kit(profile, live_browser=browser, **overview)
        except BaseException:
            pages.stop()
            raise
    if kit is not None and search is not None and web.backend is not None:
        # T-0125 criterion 3: wired here and never in ``load_tool_kit`` (its callers' kits stay BASE's)
        kit = replace(kit, search=search, web_search=web.backend)
    try:
        graph = _build_graph(config, args, profile, clock, kit, detokenizer)
    except BaseException:
        if pages is not None:
            pages.stop()
        raise
    graph.pages = pages
    if mode == "live":
        # T-0200 C9: a re-taken blank capture is noted in the session's *current* log (the HUD toggle replaces it)
        current = _CurrentSessionLog(graph.session)
        browser.on_note = lambda data: current.record("note", data, session_ms=clock.now_ms)
    graph.web = mode == "live" and web_of(
        profile, args, mode
    )  # T-0160 C-1, as ``_start_live`` resolved it
    if kit is not None and search is not None:
        graph.search, graph.search_label = search, label
    graph.search_off = search_off
    return graph


class SearchUnavailable(RuntimeError):
    """T-0125 criterion 2: an explicit ``--search ddgs`` (or ``search = "ddgs"``) whose backend cannot start."""


def _search_failure_words(exc: BaseException) -> tuple[str, str]:
    """T-0125 criterion 2: why a backend could not be constructed, and the fix (``search_prerequisite``'s words)."""
    if isinstance(exc, ImportError):
        return f"the search extra is not installed ({exc})", SEARCH_EXTRA_FIX
    if isinstance(exc, ValueError):
        return str(
            exc
        ), "web search runs only without a proxy"  # the backend's message already says "unset it"
    return str(exc), "reinstall the pinned ddgs with `uv sync --all-extras`"


def _search_degraded_line(why: str, fix: str) -> str:
    """T-0125 criterion 2 (H-T125-10): the one stderr line of a default-on search that degraded to off."""
    return f"astra session: web search is off: {why}; {fix}; --search off silences this"


def _start_live(profile: Profile, args: argparse.Namespace, clock: SessionClock) -> tuple[Any, Any]:
    """T-0111g B5: bind the start-page server and build the browser executor over it (nothing is launched yet:
    the browser starts on its first ``open``, on the loop, in ``run_session``). The composition root is the one
    place that names ``computer/`` and ``pages/``; both are imported here, so ``off``/``dry_run`` never load
    them."""
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.pages.server import StartPageServer

    browser_mode = browser_of(profile, args)
    # T-0134c A4: the resolved action echo (this is live, so it is never ``None``)
    echo = feedback_of(profile, args, "live").action_echo or LIVE_DEFAULT_ACTION_ECHO
    pages = StartPageServer()
    pages.start()
    try:
        browser = BrowserToolExecutor(
            now_ms=lambda: clock.now_ms,
            viewport=profile.browser_viewport,
            guard=None,
            headless=browser_mode == "headless",
            end_when_closed=browser_mode == "headed",
            allowed_origins=frozenset({pages.origin}),
            echo=echo,
            # T-0160 C-1: web mode unless --no-web or ``browser_web = false``; the start origin is the proxy's
            # one allowed loopback target and ``/web.html``'s origin
            web=web_of(profile, args, "live"),
            start_origin=pages.origin,
        )
    except BaseException:
        pages.stop()
        raise
    return pages, browser


@dataclass(frozen=True)
class FillerClips:
    """T-0182 B2: whether the filler clips (the ``ack`` at a thinking turn's start, the conversational interim
    filler) are on, with the source that decided it."""

    on: bool
    source: GroundingSource

    def line(self) -> str:
        """B2/B-R2: printed once to stdout whenever on, in every mode, ``--dry-run`` included."""
        return f"astra session: filler clips: on ({self.source})"


def filler_clips_of(profile: Profile, args: argparse.Namespace) -> FillerClips:
    """T-0182 B2: ``--filler-clips``/``--no-filler-clips``, else ``Profile.filler_clips`` (off by default, James
    2026-10-28). Pure: reads ``args`` with ``getattr``."""
    flag = getattr(args, "filler_clips", None)
    if flag is not None:
        return FillerClips(bool(flag), "flag")
    if profile.filler_clips:
        return FillerClips(True, "profile")
    return FillerClips(False, "default")


def clip_policy_of(
    profile: Profile, args: argparse.Namespace, mode: str, loop_on: bool
) -> tuple[bool, int | None]:
    """T-0182 B1/B4 (ruling B-R4): ``TurnPolicy``'s ``(ack_clip, still_working_after_ms)`` for this session.
    ``TurnPolicy()`` keeps BASE's ack; the product default is decided here: the ack only with filler clips on, and
    ``still_working`` after ``Profile.still_working_after_s`` only in live with the tool loop on."""
    after = profile.still_working_after_s * 1000 if mode == "live" and loop_on else None
    return filler_clips_of(profile, args).on, after


def _wire_source(source: str) -> SettingSource:
    """T-0213 A2r-2: a resolver's source as ``astra.settings.state`` states it. ``"unset"`` (the setting does not
    apply in this mode; its value is ``None``) reads ``"default"``."""
    if source in ("flag", "profile"):
        return cast("SettingSource", source)
    return "default"


def _profile_or_default(profile: Profile, field: str) -> SettingSource:
    """T-0213 R-9: ``"profile"`` when the config file set the field, else ``"default"``."""
    return "profile" if field in profile.model_fields_set else "default"


def _config_picks_profile(args: argparse.Namespace) -> bool:
    """T-0213 R-9: whether astra.toml itself sets the top-level ``profile``. ``Config.profile`` cannot say: ``load``
    merges the file over ``DEFAULTS``, which always carry ``profile``, so the field is set with no file at all."""
    path = getattr(args, "config", None)
    if path is None or not Path(path).exists():
        return False
    with Path(path).open("rb") as handle:
        return "profile" in tomllib.load(handle)


def _steps_source(loop: ToolLoop, profile: Profile) -> SettingSource:
    """T-0213 A2r-2: ``ToolLoop.source`` (the flag, or the profile field it read) as a wire source: the profile's
    field is ``"profile"`` when the config file set it, else the field's own default."""
    if loop.source == "--max-tool-steps":
        return "flag"
    return "profile" if loop.source in profile.model_fields_set else "default"


def settings_of(
    profile: Profile,
    args: argparse.Namespace,
    mode: str,
    grounding: Grounding,
    *,
    profile_from_config: bool = False,
) -> tuple[LiveSettings, dict[str, SettingSource], dict[str, tuple[SettingValue, SettingSource]]]:
    """T-0213 A2r-2: the session's initial settings snapshot, each value's source, and the read-only rows, every
    one read off the existing resolvers and their source fields (never re-derived). The pass-2 budget is ``None``
    with ReGround off (A1's ruling), although ``grounding_of`` resolves it to the main budget then.
    ``profile_from_config``: astra.toml's top-level ``profile`` picked the profile (the engine row's source)."""
    feedback = feedback_of(profile, args, mode)
    recovery = recovery_of(profile, args, mode)
    auto = auto_overview_of(profile, args, mode)
    frame = overview_frame_of(profile, args, mode, grounding=grounding)
    outcome = action_outcome_of(profile, args, mode)
    guard = repeat_guard_of(profile, args, mode)
    reminder = screen_reminder_of(args, mode)
    check = completion_check_of(args, mode)
    reask = reask_of(profile, args, mode)
    guidance = live_guidance_of(profile, args, mode)
    page = page_overview_of(profile, args, mode)
    web = search_of(profile, args, mode)
    filler = filler_clips_of(profile, args)
    loop = tool_loop_of(profile, args)
    frames = frame_settings_of(profile, args)
    live = mode == "live"

    def unset(value: Any, source: str) -> Any:
        return None if source == "unset" else value

    rows: dict[str, tuple[Any, str]] = {
        "visual_tokens": (grounding.visual_tokens, grounding.visual_tokens_from),
        "reground": (unset(grounding.reground, grounding.reground_from), grounding.reground_from),
        "reground_visual_tokens": (
            (grounding.reground_visual_tokens, grounding.reground_visual_tokens_from)
            if grounding.reground
            else (None, "unset")
        ),
        "live_feedback": (feedback.live_feedback, feedback.live_feedback_from),
        "thought_call_recovery": (
            recovery.thought_call_recovery,
            recovery.thought_call_recovery_from,
        ),
        "auto_overview": (
            unset(auto.auto_overview, auto.auto_overview_from),
            auto.auto_overview_from,
        ),
        "overview_frame": (
            unset(frame.overview_frame, frame.overview_frame_from),
            frame.overview_frame_from,
        ),
        "action_outcome": (
            unset(outcome.action_outcome, outcome.action_outcome_from),
            outcome.action_outcome_from,
        ),
        "repeat_guard": (
            unset(guard.repeat_guard, guard.repeat_guard_from),
            guard.repeat_guard_from,
        ),
        "screen_reminder": (reminder.screen_reminder, reminder.source),
        "completion_check": (check if live else None, completion_check_source(args, mode)),
        "announced_call_reask": (reask.value, reask.source),
        "live_guidance": (guidance.value, guidance.source),
        # T-0227 A-4: in every mode
        "max_frames_per_turn": (frames.max_frames_per_turn, frames.max_frames_per_turn_from),
        "frame_interval_ms": (frames.frame_interval_ms, frames.frame_interval_ms_from),
    }
    snapshot = LiveSettings(**{key: value for key, (value, _) in rows.items()})
    sources = {key: _wire_source(source) for key, (_, source) in rows.items()}
    sources["user_instructions"] = "default"
    others: dict[str, tuple[SettingValue, SettingSource]] = {
        "tools": (
            mode,
            "flag"
            if getattr(args, "tools", None) is not None
            else _profile_or_default(profile, "tools"),
        ),
        "search": (web.backend or "off", _wire_source(web.source)),
        "filler_clips": (filler.on, _wire_source(filler.source)),
    }
    if feedback.action_echo is not None:
        others["action_echo"] = (feedback.action_echo, _wire_source(feedback.action_echo_from))
    if live:
        others["page_overview"] = (page.page_overview, _wire_source(page.page_overview_from))
        others["wait_tool"] = (wait_tool_of(args, mode), _wire_source(wait_tool_source(args, mode)))
        # T-0213 R-9: the agent browser exists in live only; elsewhere these rows carry no value
        others["browser"] = (
            browser_of(profile, args),
            "flag"
            if getattr(args, "browser", None) is not None
            else _profile_or_default(profile, "browser"),
        )
        others["browser_viewport"] = (
            list(profile.browser_viewport),
            _profile_or_default(profile, "browser_viewport"),
        )
        others["browser_web"] = (
            web_of(profile, args, mode),
            "flag"
            if getattr(args, "no_web", False)
            else _profile_or_default(profile, "browser_web"),
        )
    # T-0213 R-9: the engine row reports the profile the services run under: ``--profile``, else astra.toml's
    # top-level ``profile``, else the built-in default
    engine_source: SettingSource = (
        "flag"
        if getattr(args, "profile", None)
        else ("profile" if profile_from_config else "default")
    )
    others["engine"] = (profile.name, engine_source)
    if loop.on:
        # keyed on web mode, not on the source: with web on the budget in force (a --max-tool-steps flag
        # included) is the web row's, and the other row stays empty
        steps_key = "max_tool_steps_web" if web_of(profile, args, mode) else "max_tool_steps"
        others[steps_key] = (loop.max_tool_steps, _steps_source(loop, profile))
    return snapshot, sources, others


def _build_graph(
    config: Config,
    args: argparse.Namespace,
    profile: Profile,
    clock: SessionClock,
    kit: ToolKit | None,
    detokenizer: detok.HostDetokenizer | None,
) -> SessionGraph:
    """``build_session``'s body: every client over the resolved profile, the clock and the tool kit."""
    services = profile.services
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DIR
    microphone = resolve_mic(args)
    frames = BrowserFrameSource(now_ms=lambda: clock.now_ms)
    loop = tool_loop_of(profile, args)
    mode = tools_mode_of(profile, args)
    grounding = grounding_of(
        profile, args, mode
    )  # T-0132 A2: the live default lives here, never in Session
    budget = grounding.visual_tokens
    # T-0032 criterion 9(b): the endpoint comes from the profile's ``Services`` (whose
    # validator is the loopback gate), not from ``VllmEndpoint``'s default port.
    endpoint = VllmEndpoint(base_url=services.vllm)
    # T-0131 A3: the budget rides on every request body the client builds (A2)
    llm = VllmClient(endpoint, detokenizer=detokenizer, visual_tokens=budget)
    # T-0132 B2: pass 2's client, live with ReGround on only. Equal budgets pass the main client itself; a
    # different pass-2 budget gets a second client on the same endpoint and the same (immutable, Q-2)
    # detokeniser, which ``Session.aclose`` closes.
    second: dict[str, Any] = {}
    if mode == "live" and grounding.reground:
        crop = grounding.reground_visual_tokens
        second = {
            "reground_llm": llm
            if crop == budget
            else VllmClient(endpoint, detokenizer=detokenizer, visual_tokens=crop),
            "reground_visual_tokens": crop,
        }
    # T-0134c A4: live always passes a ``LiveFeedback`` (``off`` too, R-m); off/dry_run pass no keyword at all
    feedback: dict[str, Any] = {}
    if mode == "live":
        feedback = {
            "live_feedback": feedback_of(profile, args, mode).session_value(),
            # T-0135b A4: ``True`` by default, ``False`` with off (never ``None`` in live)
            "recover_thought_calls": recovery_of(profile, args, mode).session_value(),
        }
        if auto_overview_of(profile, args, mode).auto_overview:
            # T-0150 A6: the keyword only when on, so an off build calls ``Session`` exactly as at BASE
            feedback["auto_overview"] = True
        if overview_frame_of(profile, args, mode, grounding=grounding).overview_frame:
            # T-0151 A2-9: the keyword only when on, so an off build calls ``Session`` exactly as at BASE (T-0152:
            # off by flag, key or yield alike)
            feedback["overview_frame"] = True
        if action_outcome_of(profile, args, mode).action_outcome:
            # T-0136 C-4: the keyword only when on, so an off build calls ``Session`` exactly as at BASE
            feedback["action_outcome"] = True
        if repeat_guard_of(profile, args, mode).repeat_guard:
            # T-0137 B-9: the keyword only when on, so an off build calls ``Session`` exactly as at BASE
            feedback["repeat_guard"] = True
        if screen_reminder_of(args, mode).screen_reminder:
            # T-0201 D1: the keyword only when on, so ``--screen-reminder off`` calls ``Session`` exactly as at BASE
            feedback["screen_reminder"] = True
        if completion_check_of(args, mode):
            # T-0207 C1: the keyword only when on, so ``--completion-check off`` calls ``Session`` exactly as at BASE
            feedback["completion_check"] = True
    # T-0172 B9: the bool in dry_run and live (``False`` for off, so every tools-on line records it); no keyword in off
    reask = reask_of(profile, args, mode).value
    if reask is not None:
        feedback["reask_announced_calls"] = reask
    guidance = live_guidance_of(profile, args, mode).value  # T-0174 A4: ``None`` outside live
    # T-0215: the keyword only when the kit leaves out ``google_search``, so a search-on build is BASE's call
    no_search: dict[str, bool] = {} if _kit_declares_search(kit) else {"search": False}
    if guidance in PAGE_FACT_GUIDANCE:
        # T-0174 B2-1 (H-T174-2), T-0191 A4, T-0211 B4: the page fact rides with v2, v3 and v4; v1 passes no keyword
        # (A6 compares v1 with BASE). v3 or v4 under ``--no-web`` still counts: the resolved version is v3 or v4
        feedback["page_fact"] = True
    ack_clip, still_working_after_ms = clip_policy_of(profile, args, mode, loop.on)  # T-0182 B1/B4
    session = Session(
        TurnMachine(
            TurnPolicy(
                thinking=profile.diffusion.thinking,
                tool_loop=loop.on,
                max_tool_steps=loop.max_tool_steps,
                ack_clip=ack_clip,
                still_working_after_ms=still_working_after_ms,
            )
        ),
        llm,
        # T-0007a ruling A5: the speech clients take the whole ``Services`` object and read
        # their own endpoint off it, so the loopback validator is the one that decided (the rung
        # was fixed the same way in 89c7223).
        MoshiSttClient(services),
        PocketTtsClient(services),
        silent_audio(clock) if silent_of(args) else open_audio(args),
        frames,
        SqliteMemory(data_dir / "memory.sqlite"),
        clock=clock,
        config=SessionConfig(
            data_dir,
            turns_path=Path(args.turns) if args.turns else None,
            profile=profile,
            audio_path=audio_path_of(args),
            source=Source(args.source),
        ),
        # T-0043 criterion 15: on by default, off with --no-log. Constructing the log opens
        # nothing — ``Session.start`` does — so a graph that is never started writes no file.
        session_log=SessionLog(log_dir_of(args)) if logging_on(args) else None,
        # T-0043 ruling B1: the resolved directory travels with the session whether or not
        # logging starts on, so --no-log is a start state and not a lock — the toggle opens
        # here, never in a directory inherited from whatever happens to be lying around.
        log_dir=log_dir_of(args),
        tool_kit=kit,
        # T-0131 C1: the verify-and-scroll sentences, with tools = live only (T-0174 A4: the resolved version's
        # object; ``None`` outside live)
        tool_guidance=_tool_guidance(
            guidance, web_of(profile, args, mode), wait_tool_of(args, mode), **no_search
        ),
        visual_tokens=budget,  # T-0131 A3: for the turn line only
        reground=grounding.reground,  # T-0131 B9: live only (``Session`` refuses it elsewhere)
        # T-0132 A4c: the values and their sources on the turn line, live only
        grounding=grounding.turn_extra() if mode == "live" else None,
        degeneracy_guard=True,  # T-0142 A6 (R-n): every mode, no flag and no profile key
        **second,
        **feedback,
        **mic_unavailable_of(microphone),  # T-0181 A3: only when there is no microphone (A-R13)
    )
    # T-0213 A2a-4 (R-13): in live, the guidance table and the pass-2 client for a ReGround switch from the panel: a
    # second client at the 280 crop on the same endpoint and detokeniser, built by the session at the first turn
    # boundary that needs it (construction sends nothing), so a session nobody changes builds BASE's clients only
    applying: dict[str, Any] = {}
    if mode == "live":
        applying["guidance_table"] = guidance_table_of(
            web_of(profile, args, mode), wait_tool_of(args, mode), **no_search
        )

        def pass2_client() -> VllmClient:
            return VllmClient(
                endpoint, detokenizer=detokenizer, visual_tokens=LIVE_DEFAULT_REGROUND_VISUAL_TOKENS
            )

        applying["pass2_factory"] = pass2_client
    # T-0213 A2r-2: what the Settings panel reads, from the same resolvers (set after construction, so the
    # ``Session`` call is BASE's byte for byte)
    session.init_settings(
        *settings_of(
            profile,
            args,
            mode,
            grounding,
            profile_from_config=_config_picks_profile(args),
        ),
        **applying,
    )
    server = HudServer(session, frame_sink=frames, services=services)
    relay, relay_off = _build_relay(session, server, clock)
    return SessionGraph(
        session=session,
        server=server,
        frames=frames,
        description=describe_graph(config, args, mic=microphone),
        silent=silent_of(args),
        relay=relay,
        relay_off=relay_off,
    )


class _CurrentSessionLog:
    """The session's log *now*: the HUD toggle replaces ``Session.session_log``, so the relay reads
    it at each record rather than holding the one that existed when the graph was built."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def record(
        self, kind: str, data: Any, *, turn_id: str | None = None, session_ms: int | None = None
    ) -> None:
        log = getattr(self._session, "session_log", None)
        if log is None:
            return
        if session_ms is None:
            log.record(kind, data, turn_id=turn_id)
        else:  # T-0200 C9: the executor's note carries the session clock
            log.record(kind, data, turn_id=turn_id, session_ms=session_ms)


def _build_relay(session: Any, server: Any, clock: SessionClock) -> tuple[Any, str | None]:
    """T-0128c: the canvas relay over the server's publisher, the session clock and turn state.
    An endpoint ``ASTRA_TAP_ENDPOINT`` refuses (A1) leaves the session without one, with the reason."""
    publish = getattr(server, "publish", None)
    if publish is None:
        return None, None  # a duck-typed endpoint with no publisher has nowhere to relay to
    try:
        endpoint = endpoint_from_env()
    except ValueError as exc:
        return None, str(exc)
    relay = CanvasRelay(
        publish=publish,
        clock=clock,
        state=session.turn_state_message,
        session_log=_CurrentSessionLog(session),
        endpoint=endpoint,
    )
    return relay, None


def _relay_off(graph: Any, reason: str) -> None:
    """T-0128c A8: one stderr line and one session-log note; the session runs without the relay."""
    graph.relay = None
    print(f"astra session: denoising view off: {reason}", file=sys.stderr)
    log = getattr(graph.session, "session_log", None)
    if log is not None:
        log.record("note", {"denoise_relay": {"off": reason}})


async def _start_relay(graph: Any) -> None:
    """T-0128c: start the relay after the server; a failure never stops the session (A8)."""
    reason = getattr(graph, "relay_off", None)
    relay = getattr(graph, "relay", None)
    if reason is not None:
        _relay_off(graph, reason)
        return
    if relay is None:
        return
    try:
        await relay.start()
    except ImportError as exc:
        _relay_off(graph, f"pyzmq is not installed ({exc})")
    except Exception as exc:
        _relay_off(graph, f"cannot subscribe to the canvas tap at {relay.endpoint}: {exc}")


async def _close_relay(graph: Any) -> None:
    relay = getattr(graph, "relay", None)
    if relay is not None:
        with contextlib.suppress(Exception):
            await relay.aclose()


async def _close_search(graph: Any) -> None:
    """T-0125 criterion 8 (amendment (e)): close the web search backend after the session. A close that raises is
    one stderr line (the session log is closed by now) and stops nothing else of the teardown."""
    await _close_search_backend(getattr(graph, "search", None))


async def _close_search_backend(search: Any) -> None:
    """``_close_search``'s body, for a backend with no graph yet (``build_session`` raised after ``session_cmd``
    built it)."""
    closer = getattr(search, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception as exc:
        print(f"astra session: closing web search failed: {exc!r}", file=sys.stderr)


async def _warm_repair(session: Any) -> None:
    """T-0101 (H-T101a): one warm-up repair per payload automaton, awaited, only with tools and repair on.

    ``getattr``: a duck-typed graph's session may carry no ``tool_kit`` (tests/unit/test_cli_session.py)."""
    kit = getattr(session, "tool_kit", None)
    if kit is None or kit.repair_off is not None or kit.payloads is None:
        return
    report = await warmup.warm_repair(session.llm, kit)
    print(f"astra session: repair warm-up: {warmup.summary(report)}")


class LiveStartFailed(RuntimeError):
    """T-0111g B5: ``live``'s browser could not open the start page; the session exits 1 before the HUD binds."""


#: T-0160 C-3: the one stderr notice a live session in web mode prints at startup. Its first three sentences are
#: ``pages/static/web.html``'s notice, word for word (C-7).
WEB_STARTUP_NOTICE = (
    "astra session: web browsing is on: Gearvis can open public websites. Web pages can contain instructions "
    "that mislead it. It never types passwords or card numbers. It asks before it buys, sends, posts or deletes. "
    "Its browser has a fresh profile with no saved logins; downloads and uploads are off. Do not sign in to "
    "anything in its window. --no-web turns this off. What you say can end up typed into websites; password or "
    "code fields a page does not mark as such are not recognised."
)
#: T-0160 C-4 (reading ruled 2026-10-27): Chromium's sandbox error, as its launch message says it on this box
CHROMIUM_SANDBOX_ERROR = "Chromium sandboxing failed!"
#: T-0160 C-4: appended to ``LiveStartFailed`` when the web-mode launch fails on Chromium's sandbox
WEB_SANDBOX_HINT = (
    "real websites need Chromium's sandbox: see the README section "
    '"Real websites: the browser sandbox"; --no-web runs without real websites'
)


async def _open_start_page(graph: SessionGraph) -> None:
    """T-0111g B5: launch the agent browser on the start page, on the loop, before anything else starts. On a
    failure the browser is closed (its error dropped: the open's is the one reported) and ``LiveStartFailed``
    names the exception type."""
    pages = getattr(graph, "pages", None)
    kit = getattr(graph.session, "tool_kit", None)
    browser = getattr(kit, "live_browser", None)
    if pages is None or browser is None:
        return
    # T-0169: the executor's one start-page URL (web: ``/web.html``; loopback: ``/``), decided by its own ``web``
    # flag. It is outside the LiveBrowser seam, so it is looked up statically (no ``__getattr__``): a browser that
    # does not define it opens the catalogue as at T-0111g.
    url = (
        browser.start_page_url
        if inspect.getattr_static(browser, "start_page_url", None) is not None
        else None
    )
    if not isinstance(url, str):
        url = pages.origin + "/"
    try:
        await browser.open(url)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await browser.close()
        # one stderr line: Playwright's launch errors are multi-line boxes, so only their first line, collapsed
        lines = [line for line in str(exc).splitlines() if line.strip()]
        first = " ".join(lines[0].split()) if lines else ""
        detail = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
        if getattr(graph, "web", False) is True and CHROMIUM_SANDBOX_ERROR in str(exc):
            # T-0160 C-4: web mode requires Chromium's sandbox (H0); say where the fix is and how to run without
            raise LiveStartFailed(
                f"cannot open the start page in the live browser: {detail}; {WEB_SANDBOX_HINT}"
            ) from exc
        raise LiveStartFailed(f"cannot open the start page in the live browser: {detail}") from exc


async def run_session(graph: SessionGraph) -> int:
    """Warm the repair automata, start both halves, print the page URL, serve until interrupted, then close both.
    T-0111g: in ``live`` the start page is opened first."""
    try:
        await _open_start_page(graph)
        await _warm_repair(graph.session)
        await graph.session.start()
        search_off = getattr(graph, "search_off", None)
        session_log = getattr(graph.session, "session_log", None)
        if search_off is not None and session_log is not None:
            # T-0125 criterion 2: the degrade of a default-on search, once in the session log (stderr said it)
            session_log.record("note", {"what": "web search off", "why": search_off})
        fallback = _search_fallback_of(getattr(graph, "search", None))
        if fallback is not None and session_log is not None:
            # T-0163 criterion 8: the Brave Search API fallback, once: "configured" or "not configured" and why
            session_log.record("note", fallback[1])
        try:
            await graph.server.start()
        except OSError:
            await graph.session.aclose()
            raise
    except BaseException:
        # T-0125 amendment 4: a start that fails anywhere before the HUD server is up closes the backend too
        await _close_search(graph)
        raise
    label = getattr(graph, "search_label", None)
    if getattr(graph, "search", None) is not None and label is not None:
        # T-0125 criterion 3 (amendments 2-3): once, after the start pages and the HUD server are up, so a start
        # failure stays the only stderr line
        print(SEARCH_STARTUP_LINE.format(engines=label), file=sys.stderr)
        refused = _search_fallback_of(graph.search)
        if refused is not None and refused[0] is not None:
            # T-0163 criterion 8 (H-T163-3): a present but refused key file, one line; the search stays keyless
            print(refused[0], file=sys.stderr)
    if getattr(graph, "web", False) is True:
        # T-0160 C-3: once, where the search line is, so a start failure stays the only stderr line
        print(WEB_STARTUP_NOTICE, file=sys.stderr)
    await _start_relay(graph)  # T-0128c: after the server, so its first publish has somewhere to go
    print(f"astra session: open the HUD page and connect to {graph.server.url}")
    print(f"astra session: listening on port {graph.server.port}")
    # T-0043 criterion 15: the exact path, at the moment the session starts — that line is what
    # the human pastes into ``session_log_summary.py`` while the session is still running.
    session_log = getattr(graph.session, "session_log", None)
    if session_log is None:
        print("astra session: logging is off (--no-log); this run records no session log")
    else:
        print(f"astra session: logging this session to {session_log.path}")
    # ``getattr``: the unit rig serves duck-typed graphs (tests/unit/test_cli_session.py), and a
    # graph that does not say it is silent is not.
    if getattr(graph, "silent", False):
        # Criteria 18 and 19: said where the human is looking. TTS still runs (its CPU cost is
        # part of the deal) and the spoken pane still fills; nothing is audible and no microphone
        # is opened, so no number from this run may be entered in a bench row.
        print(
            "astra session: silent -- no audio device is opened and no microphone is listening; "
            "TTS still runs and the spoken pane still fills (audio_path=silent)"
        )
    # T-0032 criterion 9(a): the seam the *session* needs is ``stt_blocks`` — the capture pump
    # reads it (``stt_frames`` is a name on ``audio_capture.CaptureStream``, never on the audio
    # object the session holds, so the old check fired on every real run and hid the wiring).
    if not hasattr(graph.session.audio, "stt_blocks"):
        print("astra session: the microphone path is not wired yet (T-0021); frames and ui.action")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await _close_relay(graph)  # T-0128c: before the server it publishes through
        await graph.server.aclose()
        await graph.session.aclose()
        await _close_search(
            graph
        )  # T-0125 criterion 8: after the session, which may still be searching
    return 0


#: The two ways a human or a service manager says "stop": Ctrl-C and `systemctl stop`.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _ignore_stop_signals() -> None:
    """Deafen the process to Ctrl-C. Registered with ``atexit`` only once a stop was serviced."""
    for sig in STOP_SIGNALS:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, signal.SIG_IGN)


def _is_a_caller_handler(previous: Any) -> bool:
    """Did somebody install this handler, or is it the interpreter's own default?

    ``default_int_handler`` / ``SIG_DFL`` mean nobody asked for anything: the process is the CLI
    and there is no in-process caller whose interruptibility ruling A3 protects. ``None`` is a
    handler installed outside Python, which cannot be restored from Python either way.
    """
    return (
        previous is not None
        and previous is not signal.default_int_handler
        and previous != signal.SIG_DFL
    )


class StopSignals:
    """The first stop signal closes the session; every later one is absorbed (T-0040, ruling A1).

    Ctrl-C in a terminal reaches the whole foreground process group, so a session started through
    ``uv`` sees two SIGINTs (``uv`` forwards its copy too). The first cancels the serve loop; the
    second used to land inside ``hud_static.close()`` -> ``HTTPServer.shutdown()`` ->
    ``Event.wait()`` and escape as a ``KeyboardInterrupt`` traceback with exit 130
    (``logs/t0039-live-82d8462/run.log``). So the *handler itself* sets both signals to
    ``SIG_IGN`` before it does anything else: from the first signal on, the shutdown sequence runs
    uninterruptibly, whatever the delivery timing.

    On the way out a handler the *caller* installed is put back (ruling A3), so an in-process
    caller stays as interruptible as it was. When the CLI *owns the process* — ``main()`` entered
    with ``argv=None``, which is what the ``astra`` console script does — the interpreter's own
    default is not put back once a stop has been serviced (ruling A6): the duplicates keep
    arriving through interpreter shutdown, and there is nobody left in this process for them to
    interrupt. An in-process caller (a test, an embedder: an explicit ``argv``) always gets both
    dispositions back instead, whatever they were, and no ``atexit`` hook (ruling A8) — a
    ``SIG_IGN`` left behind there survives ``fork``+``exec`` into every child the caller spawns
    afterwards, which is `astra down` failing to stop the services it started.

    Handled here rather than through ``loop.add_signal_handler`` because asyncio restores the
    default disposition when it removes its handlers (and again when the loop closes), which
    re-opens the window in the middle of the shutdown this class exists to close.
    """

    def __init__(
        self,
        signals: Sequence[signal.Signals] = STOP_SIGNALS,
        *,
        own_process: bool = False,
    ) -> None:
        self.signals = tuple(signals)
        #: Does the CLI own this process (ruling A8)? Only then does a serviced stop outlive the
        #: command: keeping ``SIG_IGN`` through interpreter exit, and the ``atexit`` hook.
        self.own_process = own_process
        #: The name of the first stop signal seen (``"SIGINT"``), or ``None``.
        self.seen: str | None = None
        self._previous: dict[signal.Signals, Any] = {}
        self._on_first: Callable[[], None] | None = None

    def __enter__(self) -> StopSignals:
        for sig in self.signals:
            self._previous[sig] = signal.getsignal(sig)
        return self

    def __exit__(self, *exc: object) -> None:
        self._on_first = None
        serviced = self.seen is not None
        if serviced and self.own_process:
            # Belt and braces for a process that lives on: ``SIG_IGN`` is the one disposition
            # CPython's finalisation keeps (a Python-level handler it resets to ``SIG_DFL``).
            atexit.unregister(_ignore_stop_signals)
            atexit.register(_ignore_stop_signals)
        for sig, previous in self._previous.items():
            if serviced and self.own_process and not _is_a_caller_handler(previous):
                # Ruling A6. ``uv`` forwards its own copy of each signal ~150-350 ms late
                # (measured on this box, uv 0.12.13) and a human holding Ctrl-C sends more, so
                # duplicates keep arriving while the interpreter finalises — and CPython joins
                # the non-daemon threads (``threading._shutdown``) *before* it runs the
                # ``atexit`` hooks, so one landing in that join prints ``Exception ignored in:
                # <module 'threading'> … KeyboardInterrupt``. Nothing in this process wanted to
                # be interrupted any more, so the ignore stays until it exits. A caller that had
                # a handler of its own gets it back below (ruling A3), and so does every caller
                # that did not enter through the console script (``own_process`` above, ruling
                # A8): only the process the CLI owns — whose previous disposition was the
                # interpreter's own default — is left deaf.
                continue
            if previous is not None:  # a handler set outside Python cannot be restored from it
                with contextlib.suppress(ValueError, OSError, TypeError):
                    signal.signal(sig, previous)
        self._previous.clear()

    def arm(self, on_first: Callable[[], None]) -> None:
        """Take the signals over: the first one runs ``on_first`` once, the rest are ignored."""
        self._on_first = on_first
        for sig in self.signals:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, self._handle)

    def disarm(self) -> None:
        """Forget the callback: the loop it cancels is gone, the ignore is not.

        Reviewer F1: when ``run_session`` leaves by exception (the busy-port ``OSError``),
        ``asyncio.run`` closes the loop on the way out, and a Ctrl-C arriving afterwards — during
        the error line or ``static.close()`` — would reach a ``call_soon_threadsafe`` on a closed
        loop. The signals stay ignored (that is the shutdown), only the callback goes.
        """
        self._on_first = None

    def absorb(self) -> None:
        """Ignore the stop signals from here on (the shutdown sequence is not interruptible)."""
        for sig in self.signals:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, signal.SIG_IGN)

    def _handle(self, signum: int, frame: object) -> None:
        self.absorb()  # first, and before anything can block: the next signal is a no-op
        if self.seen is None:
            self.seen = signal.Signals(signum).name
            if self._on_first is not None:
                self._on_first()


async def _serve(graph: SessionGraph, stop: StopSignals | None = None) -> int:
    """Ctrl-C cancels the serve loop, which is what closes the session (criterion 16)."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(run_session(graph))

    def cancel_the_serve_loop() -> None:
        # Suppressed for the same reason ``disarm`` exists: a loop that is already closing has
        # nothing left to cancel, and that is not a failure to report.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(task.cancel)

    signals = stop if stop is not None else StopSignals()
    signals.arm(cancel_the_serve_loop)
    try:
        return await task
    except asyncio.CancelledError:
        return 0
    finally:
        signals.disarm()


def session_cmd(config: Config, args: argparse.Namespace, *, own_process: bool = False) -> int:
    """``own_process``: the CLI is the whole process (ruling A8) — see ``StopSignals``."""
    profile = config.profiles[args.profile] if args.profile else config.active()
    # T-0045 criterion 1: resolved once, here, and then carried — the ``mic=`` the human reads is
    # the node ``open_device`` is asked for, not a second reading of a graph that can change.
    mic = resolve_mic(args)
    description = describe_graph(config, args, mic=mic)
    page = hud_static.page_urls(profile.services)[0]
    serve_page = bool(args.serve_hud)  # T-0197: resolved once; on unless --no-serve-hud
    loop = tool_loop_of(profile, args)
    refusal = loop.refusal
    if refusal is not None:
        # T-0114 / H-T114-2: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {refusal}", file=sys.stderr)
        return 2
    mode = tools_mode_of(profile, args)
    grounding = grounding_of(profile, args, mode)  # T-0132 A2: keyed on the resolved mode
    if grounding.reground and mode != "live":
        # T-0131 B9: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {reground_refusal(mode)}", file=sys.stderr)
        return 2
    crop_refusal = reground_visual_tokens_cli_refusal(grounding, mode)
    if crop_refusal is not None:
        # T-0132 B1: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {crop_refusal}", file=sys.stderr)
        return 2
    feedback_refusal = feedback_flags_refusal(profile, args, mode)
    if feedback_refusal is not None:
        # T-0134c A3: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {feedback_refusal}", file=sys.stderr)
        return 2
    feedback = feedback_of(profile, args, mode)  # T-0134c A2: keyed on the resolved mode
    recovery_refusal = recovery_flag_refusal(profile, args, mode)
    if recovery_refusal is not None:
        # T-0135b A3: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {recovery_refusal}", file=sys.stderr)
        return 2
    recovery = recovery_of(profile, args, mode)  # T-0135b A2: keyed on the resolved mode
    overview_refusal = page_overview_flag_refusal(profile, args, mode)
    if overview_refusal is not None:
        # T-0140 A5: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {overview_refusal}", file=sys.stderr)
        return 2
    overview = page_overview_of(profile, args, mode)  # T-0140 A5: keyed on the resolved mode
    auto_refusal = auto_overview_flag_refusal(profile, args, mode)
    if auto_refusal is not None:
        # T-0150 A6: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {auto_refusal}", file=sys.stderr)
        return 2
    auto = auto_overview_of(profile, args, mode)  # T-0150 A6: keyed on the resolved mode
    both_refusal = auto_and_page_overview_refusal(auto, overview)
    if both_refusal is not None:
        # T-0150 A6 (H-T150-2): refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {both_refusal}", file=sys.stderr)
        return 2
    frame_refusal = overview_frame_flag_refusal(profile, args, mode)
    if frame_refusal is not None:
        # T-0151 A2-9: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {frame_refusal}", file=sys.stderr)
        return 2
    overview_frame = overview_frame_of(
        profile, args, mode, grounding=grounding
    )  # T-0151 A2-9: keyed on the resolved mode (T-0152 A2: the default gives way inside)
    frame_line = overview_frame.line(
        overview_frame_yielded_to(overview_frame, profile, args, mode, grounding=grounding)
    )  # T-0152 A5
    with_refusal = overview_frame_with_refusal(overview_frame, auto, overview, grounding)
    if with_refusal is not None:
        # T-0151 A2-9 (Q-4): refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {with_refusal}", file=sys.stderr)
        return 2
    outcome_refusal = action_outcome_flag_refusal(profile, args, mode)
    if outcome_refusal is not None:
        # T-0136 C-3: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {outcome_refusal}", file=sys.stderr)
        return 2
    outcome = action_outcome_of(profile, args, mode)  # T-0136 C-2: keyed on the resolved mode
    outcome_feedback_refusal = action_outcome_with_feedback_off_refusal(outcome, feedback)
    if outcome_feedback_refusal is not None:
        # T-0136 C-3: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {outcome_feedback_refusal}", file=sys.stderr)
        return 2
    guard_refusal = repeat_guard_flag_refusal(profile, args, mode)
    if guard_refusal is not None:
        # T-0137 B-9: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {guard_refusal}", file=sys.stderr)
        return 2
    guard = repeat_guard_of(profile, args, mode)  # T-0137 B-9: keyed on the resolved mode
    reask_refusal = reask_flag_refusal(profile, args, mode)
    if reask_refusal is not None:
        # T-0172 B9: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {reask_refusal}", file=sys.stderr)
        return 2
    reask = reask_of(profile, args, mode)  # T-0172 B9: keyed on the resolved mode
    guidance_refusal = live_guidance_flag_refusal(profile, args, mode)
    if guidance_refusal is not None:
        # T-0174 A4: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {guidance_refusal}", file=sys.stderr)
        return 2
    guidance = live_guidance_of(profile, args, mode)  # T-0174 A4: keyed on the resolved mode
    wait_refusal = wait_tool_flag_refusal(args, mode)
    if wait_refusal is not None:
        # T-0202 B8: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {wait_refusal}", file=sys.stderr)
        return 2
    wait_token = wait_tool_token(args, mode)  # T-0202 D1: on(default), on(flag), or nothing
    reminder_refusal = screen_reminder_flag_refusal(args, mode)
    if reminder_refusal is not None:
        # T-0201 D1 (R-D4): refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {reminder_refusal}", file=sys.stderr)
        return 2
    reminder = screen_reminder_of(args, mode)  # T-0201 D1: keyed on the resolved mode
    check_refusal = completion_check_flag_refusal(args, mode)
    if check_refusal is not None:
        # T-0207 C1: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {check_refusal}", file=sys.stderr)
        return 2
    web = search_of(profile, args, mode)  # T-0125 criterion 2: keyed on the resolved mode
    if web.refusal is not None:
        # T-0125 criterion 2: refused at startup, before anything is constructed or printed as a plan
        print(f"astra session: {web.refusal}", file=sys.stderr)
        return 2
    live_line: LiveLine | None = None
    if mode == "live":
        # T-0111g B2: refused before anything is constructed; the check reads files and the environment only
        live_line = LiveLine(
            browser=browser_of(profile, args),
            viewport=profile.browser_viewport,
            web=web_of(profile, args, mode),  # T-0160 C-2
        )
        missing = live_prerequisites(live_line.browser)
        if missing is not None:
            print(f"astra session: {missing}", file=sys.stderr)
            return 2
    fillers = filler_clips_of(profile, args)
    if fillers.on:
        print(fillers.line())  # T-0182 B2/B-R2: only when on, in every mode, --dry-run included
    if args.dry_run:
        print(description)
        kit: ToolKit | None = None
        if mode in ("dry_run", "live"):
            # T-0092b criterion 1: the dry run checks the kit as the session would, so the line it
            # prints says whether the repair would be on. It constructs no client and opens nothing
            # (T-0111g B3: in live no browser and no start-page server either).
            try:
                # T-0140 A5: the tools line shows the selected set's sha12 (the keyword only when on)
                # (T-0202 B8: the wait set's when the flag is on, the keyword again only when on)
                # (T-0215: live with no search backend shows the no-search set's, the keyword only when off)
                no_search: dict[str, Any] = (
                    {"search": False} if mode == "live" and web.backend is None else {}
                )
                kit, _ = (
                    load_tool_kit(
                        profile, page_overview=overview.page_overview, wait_tool=True, **no_search
                    )
                    if wait_token
                    else load_tool_kit(profile, page_overview=True, **no_search)
                    if overview.page_overview
                    else load_tool_kit(profile, **no_search)
                )
            except ToolsUnavailable as exc:
                print(f"astra session: {exc}", file=sys.stderr)
                return 1
        print(
            tools_line(kit, loop.max_tool_steps if loop.on else None, live_line)
            + visual_tokens_suffix(grounding.visual_tokens)  # T-0131 A3 (T-0132 A4b: resolved)
            + reground_suffix(grounding.reground)  # T-0131 B9 (live only: refused above)
            + wait_token  # T-0202 B8/D1: live with the wait tool on only
        )
        budget_line = loop.line()
        if budget_line is not None:
            # T-0178 A4: right after the tools line, only when not the mode's default
            print(budget_line)
        if mode == "live":
            print(grounding.line())  # T-0132 A4a: after the tools line, live only
            print(feedback.line())  # T-0134c A5a: right after the grounding line, live only
            print(recovery.line())  # T-0135b A5a: right after the feedback line, live only
            print(overview.line())  # T-0140 A5: after the feedback (and recovery) line, live only
            print(auto.line())  # T-0150 A6: right after the page overview line, live only
            print(frame_line)  # T-0151 A2-9: right after the auto overview line, live only
            print(outcome.line())  # T-0136 C-5: right after the overview frame line, live only
            print(guard.line())  # T-0137 B-9: right after the action outcome line, live only
            print(reminder.line())  # T-0201 D1 (R-D2): after grounding, before search, live only
            print(
                completion_check_line(args, mode)
            )  # T-0207 C1 (R-C1): after the reminder, before search
        if web.shown:
            print(
                web.line()
            )  # T-0125 criterion 3: after the tools block (live always, dry-run when on)
        if guidance.source in ("flag", "profile"):
            print(guidance.line())  # T-0174 A4: right before the reask line; silent for the default
        if reask.value is not None:
            print(reask.line())  # T-0172 B9: the last tools-on line (dry_run and live)
        if serve_page:
            print(f"astra session: the HUD page would be served at {page}")
        return 0
    if mic.unknown:
        # Ruling A3: no --no-mic and no default source to open. Refusing is the whole point of
        # the task — an output-only session serves, looks alive and hears nothing (2026-09-15).
        print(f"astra session: no capture source found -- {mic.why}", file=sys.stderr)
        return 1
    # The graph is read once per run: what was resolved is written back as if it had been typed,
    # so the device below is opened on the node this command just printed, rather than on the
    # answer of a second reading (criterion 2 is a claim about what heard the turn).
    args.mic = mic.node if mic.node is not None else MIC_OFF
    degraded: str | None = None
    if web.backend is not None:
        # T-0125 criterion 2: checked at a real start only, before anything is built. Explicit: refused; the live
        # default: the session starts with search off, said once on stderr (and once in the session log)
        built, unready = search_start(web.backend)
        args.search_built = (
            built  # written back, as ``args.mic`` is: ``build_session`` wires this one
        )
        if unready is not None:
            why, fix = unready
            if web.source != "default":
                print(
                    f"astra session: {web.asked()} cannot start: {why}; {fix}",
                    file=sys.stderr,
                )
                return 2
            print(_search_degraded_line(why, fix), file=sys.stderr)
            degraded = why
            args.search_off = (
                why  # written back, as ``args.mic`` is: ``build_session`` builds no backend
            )
    static: object | None = None
    # T-0197 criterion 5: the one busy-port stderr line, printed right after the regime line
    busy: list[str] = []
    if serve_page:
        # Criterion 17: one command gives both halves. The static server runs on its own daemon
        # thread beside the session and is closed on the way out, with the session. T-0197: on by
        # default; a busy port is reported once (after the regime line) and the session runs on.
        static = serve_hud(
            profile.services,
            lambda exc: busy.append(_session_hud_busy_line(profile.services, exc)),
        )
    try:
        # Criterion 19 + ruling A19: the two facts the regime line states, measured once, here —
        # where a session is actually starting — and never under --dry-run.
        budget_line = loop.line()
        if budget_line is not None:
            print(budget_line)  # T-0178 A4: in the tools block, only when not the mode's default
        if mode == "live":
            print(grounding.line())  # T-0132 A4a: before the regime line, live only
            print(feedback.line())  # T-0134c A5a: right after the grounding line, live only
            print(recovery.line())  # T-0135b A5a: right after the feedback line, live only
            print(
                overview.line()
            )  # T-0140 A5: after the feedback (and recovery) line, before the regime line
            print(
                auto.line()
            )  # T-0150 A6: right after the page overview line, before the regime line
            print(
                frame_line
            )  # T-0151 A2-9: right after the auto overview line, before the regime line
            print(
                outcome.line()
            )  # T-0136 C-5: right after the overview frame line, before the regime line
            print(
                guard.line()
            )  # T-0137 B-9: right after the action outcome line, before the regime line
            print(reminder.line())  # T-0201 D1 (R-D2): after grounding, before search, live only
            print(
                completion_check_line(args, mode)
            )  # T-0207 C1 (R-C1): after the reminder, before search
        if web.shown:
            print(
                web.line(degraded)
            )  # T-0125 criterion 3: after the tools block, before the regime line
        if guidance.source in ("flag", "profile"):
            print(guidance.line())  # T-0174 A4: right before the reask line; silent for the default
        if reask.value is not None:
            print(
                reask.line()
            )  # T-0172 B9: the last tools-on line (dry_run and live), before the regime line
        print(
            f"astra session: {regime_line(profile, preflight.gather_regime(profile), mic.field)}"
            + wait_token  # T-0202 B8/D1: the start line carries the token, live with the wait tool on only
        )
        if static is not None:
            print(f"astra session: the HUD page is at {page}")  # T-0197 criterion 2
        for line in busy:
            print(line, file=sys.stderr)  # T-0197 criterion 5: the session continues
        # T-0040: the stop signals belong to this command for as long as it runs, and to whoever
        # owned them again afterwards (ruling A3). The ``absorb`` re-asserts the ignore before the
        # closing sequence — ``asyncio.Runner`` installs and removes a SIGINT handler of its own
        # around the loop, and the sequence below must not depend on which of us wrote last.
        with StopSignals(own_process=own_process) as stop:
            try:
                return _session_graph_cmd(config, args, stop)
            finally:
                if stop.seen is not None:
                    stop.absorb()
                if static is not None:
                    static.close()  # type: ignore[attr-defined]
                    static = None
                if stop.seen is not None:
                    print(f"astra session: {stop.seen} — session closed, nothing left listening")
    finally:
        # T-0197 criterion 6: a failure between the bind and the block above still unbinds
        if static is not None:
            static.close()  # type: ignore[attr-defined]


def _session_graph_cmd(
    config: Config, args: argparse.Namespace, stop: StopSignals | None = None
) -> int:
    try:
        try:
            graph = build_session(config, args)
        except BaseException:
            # T-0125 amendment 4: the backend ``session_cmd`` built for this start is closed when the build fails
            built = getattr(args, "search_built", None)
            if built is not None:
                asyncio.run(_close_search_backend(built[0]))
            raise
    except AudioDeviceUnavailable as exc:
        # Criterion 19: only the device failure is reported as one (ruling A18) — a missing
        # PortAudio is the one this box hits — and it never falls through to another device: a
        # session on a device nobody chose is a recording whose regime is a guess.
        print(f"astra session: {exc}", file=sys.stderr)
        return 1
    except ToolsUnavailable as exc:
        # T-0092b criterion 2(a): dry_run without its detokeniser is refused, naming the file.
        print(f"astra session: {exc}", file=sys.stderr)
        return 1
    except SearchUnavailable as exc:
        # T-0125 criterion 2: an explicit search whose backend cannot be constructed; nothing else was built
        print(f"astra session: {exc}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_serve(graph, stop))
    except LiveStartFailed as exc:
        print(f"astra session: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        port = getattr(graph.server, "port", None)
        print(
            f"astra session: cannot bind the conductor endpoint on port {port}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        # T-0111g B5: the session closed the browser in ``aclose``; the start pages go last
        pages = getattr(graph, "pages", None)
        if pages is not None:
            pages.stop()


# --- render-clips ------------------------------------------------------------------------------

#: T-0182 B6: the script ``render-clips`` delegates to (it lives in the checkout, beside ``services/``, and is not
#: part of the package)
RENDER_CLIPS_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_clips.py"


def render_clips(args: argparse.Namespace) -> int:
    """T-0182 B6: a thin wrapper over ``scripts/render_clips.py``'s ``main``: the same arguments, so the same
    output, refusals and exit code. Loaded afresh on each call, as ``python scripts/render_clips.py`` would be."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_astra_render_clips", RENDER_CLIPS_SCRIPT)
    if not RENDER_CLIPS_SCRIPT.is_file() or spec is None or spec.loader is None:
        print(
            f"astra render-clips: {RENDER_CLIPS_SCRIPT} not found; run it from a checkout of the repository",
            file=sys.stderr,
        )
        return 2
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    argv = ["--config", str(args.config)]
    if args.data_dir is not None:
        argv += ["--data-dir", str(args.data_dir)]
    if args.kinds is not None:
        argv += ["--kinds", *args.kinds]
    if args.force:
        argv.append("--force")
    code: int = script.main(argv)
    return code


# --- argv --------------------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    runner: Runner | None = None,
    probe: Probe | None = None,
) -> int:
    # T-0156 R3: the usage line names the console script that was invoked (``astra`` is kept as an alias for one
    # release); any other argv[0] (``python -m``, pytest, an embedder) gets the new name.
    invoked = os.path.basename(sys.argv[0]) if sys.argv else ""
    prog = invoked if invoked in ("astra", "diffusion_gearvis") else "diffusion_gearvis"
    p = argparse.ArgumentParser(prog=prog, description="local_astra control")
    p.add_argument("--config", type=Path, default=Path("astra.toml"))
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session", help="serve the HUD endpoint for a profile")
    s.add_argument("--profile", default=None)
    s.add_argument("--data-dir", default=None, help="where thumbnails, memory and turns.jsonl go")
    s.add_argument("--turns", default=None, help="turns.jsonl path (defaults under --data-dir)")
    s.add_argument("--sink", default=None, help="PipeWire node name or id to play through")
    s.add_argument(
        "--audio-path",
        default="speakers",
        help="what is actually playing, for the regime on every turns.jsonl line "
        "(headset | speakers+aec | loopback)",
    )
    s.add_argument(
        "--silent",
        action="store_true",
        help="play through a null device: no audio device is opened and no microphone is "
        "listening (every turns.jsonl line of the run says audio_path=silent)",
    )
    # T-0045 criterion 1: the capture source. ``--source`` below is the *perception* source
    # (screen/camera) and is never handed to the device layer.
    microphone = s.add_mutually_exclusive_group()
    microphone.add_argument(
        "--mic",
        default=None,
        help="PipeWire node name of the microphone to open (default: the box's default capture "
        f"source, read from the graph; `--mic {MIC_OFF}` is --no-mic)",
    )
    microphone.add_argument(
        "--no-mic",
        action="store_true",
        help="open no capture source: the session plays but hears nothing",
    )
    s.add_argument("--source", default=Source.SCREEN.value, choices=[m.value for m in Source])
    s.add_argument(
        "--no-log",
        action="store_true",
        help="start with the per-session debug log off (the HUD checkbox can turn it on)",
    )
    s.add_argument(
        "--log-dir",
        default=None,
        help="where the per-session debug log goes (default: logs/sessions/)",
    )
    s.add_argument("--dry-run", action="store_true", help="print the resolved graph and exit")
    s.add_argument(
        "--tools",
        type=tools_arg,
        default=None,
        metavar="{off,dry-run,live}",
        help="the session's tool setting (default: the profile's `tools`); dry-run requests and "
        "describes native tool calls and executes none; live acts in Gearvis's own sandboxed browser "
        "on its loopback start pages only, and needs the tool loop (T-0111g)",
    )
    s.add_argument(
        "--browser",
        choices=("headed", "headless"),
        default=None,
        help="with --tools live: show the agent browser's window (headed, the default at the desk) or "
        "run it without one (headless); default: the profile's `browser` (T-0111g)",
    )
    s.add_argument(
        "--tool-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="send each executed call's result back to the model, which continues until it answers "
        "(default: the profile's `tool_loop` when set, else on whenever tools are on, off with tools "
        "off); --tool-loop needs --tools dry-run (T-0111a, T-0123)",
    )
    s.add_argument(
        "--max-tool-steps",
        type=max_tool_steps_arg,
        default=None,
        metavar="N",
        help="with the loop on: executed calls per turn, then one report step (default 8; 20 in live "
        "with web on: the profile's `max_tool_steps`, or `max_tool_steps_web` in web mode); N >= 1",
    )
    s.add_argument(
        "--visual-tokens",
        type=int,
        choices=(280, 560, 1120),
        default=None,
        help="the per-request visual-token budget per image (default: the profile's `visual_tokens`, "
        "else 1120 in every tools mode; 280 restores the old cost) (T-0131, T-0183)",
    )
    s.add_argument(
        "--max-frames-per-turn",
        type=int,
        choices=FRAMES_PER_TURN_CHOICES,
        default=None,
        help="the most screens one spoken turn shows the model: the onset screen plus screens that changed "
        "while you spoke (default: the profile's `max_frames_per_turn`, else 3); 4 is the engine's image limit "
        "and leaves no earlier screen as an image (T-0059, T-0227)",
    )
    s.add_argument(
        "--frame-interval-ms",
        type=int,
        choices=FRAME_INTERVALS_MS,
        default=None,
        metavar="MS",
        help="the HUD's capture interval for the shared screen and camera, in ms, one of "
        + ", ".join(str(v) for v in FRAME_INTERVALS_MS)
        + " (default: the profile's `frame_interval_ms`, else 1000); the agent browser's screenshots are "
        "not affected (T-0227)",
    )
    # T-0131 B9: two spelled-out flags on one dest (``None``: the profile's), so both are named in the source
    reground = s.add_mutually_exclusive_group()
    reground.add_argument(
        "--reground",
        dest="reground",
        action="store_const",
        const=True,
        default=None,
        help="with --tools live: before each click_at/type_at, ask again on a crop around the point and click "
        "where that answer points (a second, never-voiced request); default: the profile's `reground`, else off "
        "(T-0133); "
        "refused with any other tools mode (T-0131)",
    )
    reground.add_argument(
        "--no-reground",
        dest="reground",
        action="store_const",
        const=False,
        default=None,
        help="ReGround off, whatever the profile says (T-0131)",
    )
    s.add_argument(
        "--reground-visual-tokens",
        type=int,
        choices=(280, 560, 1120),
        default=None,
        help="with --tools live and ReGround on: pass 2's (the crop's) visual-token budget, used only when "
        "pass 2 carries one image (default: the profile's `reground_visual_tokens`, else 280 when the main "
        "budget is 1120 (T-0133), else the main budget); refused with ReGround off or any other tools mode (T-0132)",
    )
    s.add_argument(
        "--live-feedback",
        choices=LIVE_FEEDBACK_CHOICES,
        default=None,
        help="with --tools live: off (no label), label (a 'screen now' label after the current frame) or "
        "label+extent (the label plus how much page lies above and below); default: the profile's "
        "`live_feedback`, else label (T-0134c); refused with any other tools mode",
    )
    s.add_argument(
        "--action-echo",
        choices=ACTION_ECHOES,
        default=None,
        help="with --tools live: the units an executed click_at/type_at reports its point in, pixels (the "
        "page's) or model (the call's own 0-999 units); default: the profile's `action_echo`, else model "
        "(T-0134c); refused with any other tools mode",
    )
    s.add_argument(
        "--thought-call-recovery",
        choices=("on", "off"),
        default=None,
        help="with --tools live: on executes a valid tool call left at the tail of an unclosed thought, off "
        "leaves it unexecuted; default: the profile's `thought_call_recovery`, else on (T-0135b); refused "
        "with any other tools mode",
    )
    s.add_argument(
        "--page-overview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="with --tools live: offer the page_overview tool (one picture of the whole page, shown for one "
        "step, never clickable; tools_v1_overview.json); default: the profile's `page_overview`, else off "
        "(T-0140); refused with any other tools mode",
    )
    s.add_argument(
        "--auto-overview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="with --tools live: when the page is taller than the screen, show the whole page as a picture in "
        "each turn's first request, without the model asking (the tools stay v1); default: the profile's "
        "`auto_overview`, else off (T-0150); refused with any other tools mode and with --page-overview",
    )
    s.add_argument(
        "--overview-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="with --tools live: when the page is taller than the screen (up to 2 screens), each turn's first "
        "request shows only the whole page as a picture, and a click_at/type_at on it is translated to the page, "
        "scrolled into view and dispatched (the tools stay v1); default: the profile's `overview_frame`, else on, "
        "giving way (off) beside --auto-overview, --page-overview or --reground (T-0151, T-0152); refused with any "
        "other tools mode, and an explicit --overview-frame is refused with --auto-overview, --page-overview or "
        "--reground",
    )
    s.add_argument(
        "--action-outcome",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="with --tools live: after an executed click_at/type_at, the next step's 'screen now' label says what "
        "visibly changed (no visible change / changed near your click / changed elsewhere (<region>) / page "
        "changed), from a pixel diff of the before and after frames; default: the profile's `action_outcome`, "
        "else off (T-0136); refused with any other tools mode and with --live-feedback off",
    )
    s.add_argument(
        "--repeat-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="with --tools live: a browser action that repeats an earlier step on the same screen gets "
        "repeat_of_step (that step's action and outcome) in its result, and after two no-change repeats, or on a "
        "third visit to the same screen, the action is refused with a fixed reason; default: the profile's "
        "`repeat_guard`, else on in live (T-0137, T-0179; --no-repeat-guard turns it off); refused with any "
        "other tools mode",
    )
    s.add_argument(
        "--filler-clips",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="play the `ack` clip as a thinking turn starts and the conversational interim filler; default: the "
        "profile's `filler_clips`, else off (T-0182)",
    )
    s.add_argument(
        "--announced-call-reask",
        choices=("on", "off"),
        default=None,
        help="with --tools dry-run or live: when a step's answer only announces an action (\"I will search for "
        '...") and makes no tool call, ask the model once more in the same step for that call; default: the '
        "profile's `announced_call_reask`, else on (T-0172); refused with tools off",
    )
    s.add_argument(
        "--wait-tool",
        choices=("on", "off"),
        default=None,
        help="with --tools live: offer the wait tool (wait about one second without touching the page, then a new "
        "screenshot; at most 3 in a row and 6 per request) and one guidance sentence telling the model to call it "
        "instead of telling the user to wait; tools_v1_wait.json, or tools_v1_overview_wait.json with "
        "--page-overview; default on in live, off turns it off (T-0202 D); refused with any other tools mode",
    )
    s.add_argument(
        "--screen-reminder",
        choices=("on", "off"),
        default=None,
        help="with --tools live: on adds one bracketed fact after each labelled step's last tool response, saying "
        "the last image above is the screen now; off leaves the step as before; default on (T-0201); refused with "
        "any other tools mode",
    )
    s.add_argument(
        "--completion-check",
        choices=("on", "off"),
        default=None,
        help="with --tools live: on means that before a final answer is voiced after a browser action, the model "
        "sees the screen now and confirms or continues; off voices the answer as before; default on (T-0207); "
        "refused with any other tools mode",
    )
    s.add_argument(
        "--live-guidance",
        choices=LIVE_GUIDANCE_VERSIONS,
        default=None,
        help="with --tools live: which screenshot guidance the system message carries: v1 (after an action, the "
        "next screenshot shows the screen), v2 (the latest screenshot in this turn already shows it), v3 (v2 "
        "plus how to go to a different website: Ctrl+L, then the start page's address box; web mode only, v2's "
        "text with --no-web; T-0191) or v4 (v3 plus: act from a shared screen in your own browser via Ctrl+L, and "
        "open and name the page behind a google_search answer; web mode only, v2's text with --no-web; T-0211); "
        "default: the profile's `live_guidance`, else v3 (T-0174, T-0191); refused with any "
        "other tools mode",
    )
    s.add_argument(
        "--search",
        choices=SEARCH_CHOICES,
        default=None,
        help="web search for the model's google_search calls: ddgs (keyless: duckduckgo, then brave, then "
        "mojeek; each query is sent off this machine and logged in the session log; results are untrusted web "
        "text) or off; default: the profile's `search`, else on with --tools live and off with --tools dry-run; "
        "refused with tools off, and with --tools dry-run and the tool loop off (T-0125)",
    )
    s.add_argument(
        "--no-web",
        action="store_true",
        help="with --tools live: keep the agent browser on Gearvis's own loopback start pages; by default it can "
        "open public https websites through the session's egress proxy (fresh profile, no passwords or card "
        "numbers typed, no downloads or uploads, Chromium sandbox required); default: the profile's "
        "`browser_web`, else on; no effect with any other tools mode (T-0160)",
    )
    s.add_argument(
        "--serve-hud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "serve hud/ on services.hud_port beside the session and print the page URL "
            "(default: on; --no-serve-hud turns it off; a busy port is reported and the "
            "session continues)"
        ),
    )
    up_parser = sub.add_parser("up", help="start services for a profile")
    up_parser.add_argument("--profile", default=None)
    up_parser.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    up_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="cap every per-service probe budget at this many seconds "
        "(default: each service's own budget, supervisor.PROBE_BUDGETS_S)",
    )
    up_parser.add_argument(
        "--restart",
        default=None,
        metavar="all|name,name",
        help="stop these services and start them again. Without it nothing running is signalled "
        "and no state file is removed: a healthy service is skipped and a drifted or unrecorded "
        "one is refused",
    )
    d = sub.add_parser("doctor", help="check this box against what a live run needs")
    d.add_argument("--profile", default=None)
    d.add_argument("--json", action="store_true", help="one JSON object per check, nothing else")
    sub.add_parser("down", help="stop what `astra up` started")
    status_parser = sub.add_parser("status", help="health of every service, and the pid files")
    status_parser.add_argument("--profile", default=None)
    hud_parser = sub.add_parser("hud", help="serve hud/ on loopback and print the page URL")
    hud_parser.add_argument("--profile", default=None)
    b = sub.add_parser("bench", help="run one benchmark and compare to docs/03 §7 thresholds")
    b.add_argument("bench_id")
    e = sub.add_parser("eval", help="run one quality evaluation")
    e.add_argument("eval_id")
    f = sub.add_parser("forget", help="delete memory since a time")
    f.add_argument("--since", required=True)
    sub.add_parser(
        "fetch-models",
        help="not implemented in v0.1: use the hf download commands in the README",
    )
    sub.add_parser("config", help="print the resolved profile")
    rc = sub.add_parser(
        "render-clips",
        help="render the clip assets once through the running Pocket TTS (scripts/render_clips.py)",
    )
    rc.add_argument("--data-dir", default=None, help="default: memory.store.DEFAULT_DIR")
    rc.add_argument(
        "--kinds", nargs="+", default=None, help="kinds to render (default: the default set)"
    )
    rc.add_argument("--force", action="store_true", help="re-render kinds that are current")
    args = p.parse_args(argv)
    if args.cmd == "render-clips":
        return render_clips(args)
    if args.cmd == "session" and args.silent and args.mic not in (None, MIC_OFF):
        # T-0045 ruling A2: --silent opens no device at all, so it cannot also open a microphone.
        # A usage error (exit 2), not a silent precedence rule: one of the two flags is a typo.
        p.error(
            f"--silent opens no audio device at all, so it cannot open a microphone "
            f"({args.mic!r}); use --silent alone, or --silent --no-mic"
        )

    cfg = load(args.config)
    profile = getattr(args, "profile", None)
    if profile is not None and profile not in cfg.profiles:
        print(
            f"astra {args.cmd}: no profile {profile!r} in {args.config} "
            f"(it has {sorted(cfg.profiles)})",
            file=sys.stderr,
        )
        return 2
    if args.cmd == "config":
        prof = cfg.profiles[args.profile] if getattr(args, "profile", None) else cfg.active()
        print(prof.model_dump_json(indent=2))
        return 0
    if args.cmd == "session":
        # Ruling A8: ``argv=None`` is the ``astra`` console script — the CLI owns this process, so
        # a serviced stop signal stays absorbed through interpreter exit. An explicit ``argv`` is
        # an in-process caller, whose signal dispositions are restored (ruling A3).
        return session_cmd(cfg, args, own_process=argv is None)
    if args.cmd == "doctor":
        return doctor(cfg, args)
    if args.cmd == "hud":
        return hud(cfg, args)
    if args.cmd == "up":
        # No default runner: with nothing injected the verb takes the real, verified path
        # (``supervisor.start_service``). The runner seam stays for callers that replace the start
        # of a plan entry outright and report its outcome themselves.
        return up(cfg, args, runner, probe or supervisor.probe)
    if args.cmd == "down":
        return down(cfg, args)
    if args.cmd == "status":
        return status(cfg, args, probe or supervisor.probe)
    print(f"astra {args.cmd}: not implemented yet (profile {cfg.profile})", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
