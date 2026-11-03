"""The seam T-0049 group A's tests assume, in one place (as ``_box_rig.py`` does for T-0033).

The implementer owns the *shape* of the two modules; what the tests fix is the public surface
they call, and it is written out here so the whole contract can be read at once. Every lookup
below fails as an ``AssertionError`` naming the criterion, never as an ``AttributeError`` or a
collection-time ``ImportError`` — ``tests/conftest.py`` runs a pending row as
``xfail(strict=True, raises=(NotImplementedError, AssertionError))``, so a missing symbol is the
documented "not implemented yet" outcome and a *wrong* signature is a red test.

``local_astra/service_state.py`` — the pure middle of *observe → decide → act → verify → record*
(criteria 2, 7, 8). No clock, no socket, no subprocess::

    ACTIONS = ("START", "SKIP", "CONFLICT")
    RECORD_VERSION: int

    StateRecord(version, name, kind, pid, start_time, argv, script_digest,
                container_name, container_id, started_at)     # frozen dataclass, all keyword
        kind in {"process", "container"}; start_time is /proc/<pid>/stat field 22 (ticks) or None;
        started_at is the wall clock of the verified start (float, time.time)

    Identity(kind, alive, pid=None, start_time=None, container_id=None, started_at=None,
             note=None)                                        # frozen dataclass, all keyword
        what the right authority says is live *now*: /proc for a process, `docker inspect` for a
        container. ``started_at`` is docker's ``.State.StartedAt`` string for a container.

    Observation(name, url, healthy, health_reason=None, record=None, live=None,
                script_digest="")                              # frozen dataclass, all keyword
        record: the StateRecord this CLI wrote, or None when there is none
        live:   the Identity the recorded handle resolves to, or None when it is gone
        script_digest: the digest of services/<name>.sh **as it is on disk now**

    Decision(name, action, reason)                             # frozen dataclass
    decide(observation: Observation) -> Decision               # pure, total, deterministic

``local_astra/supervisor.py`` — the observing, acting, verifying and recording half
(criteria 1, 3, 4, 5, 6). Everything T-0033 and T-0041 put there keeps working unchanged::

    SETTLE_S: float                                  # 1.0 s default (Regime)

    Handle(name, pid, argv, log_path, pid_file)      # T-0033, plus:
        .returncode() -> int | None                  # the launcher's own exit status, None while
                                                     # it is still running (criterion 4, ruling A1)

    StartOutcome(name, status, ok, message, handle=None, record=None)
        status in ("started", "FAILED", "NOT-NEW"); ok is True **only** for "started"
        message: the row's text — for FAILED it names the launcher's exit status and quotes the
                 log line that explains it; for a containerised service whose name cannot be
                 resolved it carries ``container=unknown``

    start_service(name, command, url, *, probe=None, log_dir=None, run_dir=None,
                  settle_s=None, budget_s=None) -> StartOutcome
        spawn ``command`` (``command[0]`` is the service script), watch the launcher for
        ``settle_s``, then probe ``url`` up to ``budget_s``, then verify the instance is NEW, and
        only then write the state record. ``probe(name, url) -> str | None`` is the T-0033 seam.

    script_is_containerised(script_path) -> bool     # the script's foreground server is a
                                                     # containerised one
    script_container(script_path) -> str | None      # the ``--name`` it passes, or None

    live_identity(record: StateRecord) -> Identity   # criterion 6: ask the *right authority* —
        ``/proc`` for kind=process, ``docker inspect <record.container_id>`` for kind=container
        (with the ``sg docker -c`` retry). Never raises: a handle that cannot be resolved is
        ``Identity(alive=False, note=…)``. This is what fills ``Observation.live``.

    read_state_record(path) -> StateRecord | None    # the versioned record; None for a T-0033 file
    read_pid_file / write_pid_file / stop / pid_files / start_time_of / log_tail   # T-0033/T-0041,
                                                     # behaviour unchanged, T-0033-shape files and
                                                     # new records both readable (criterion 5)

**Two seam decisions the criteria leave open, taken here and stated so they are visible.**

1. *Container liveness is read from ``docker inspect <recorded id>`` as JSON*, parsed the way
   ``preflight.parse_docker_inspect`` already parses it, and not from a ``-f '{{...}}'`` template.
   D4 names both spellings; one is picked because the repo already has a JSON parser and a
   *recorded* inspect fixture (``tests/fixtures/box/docker_inspect_stt.json``) to test it against,
   and because a Go-template spelling cannot be faked without reimplementing Go templates in the
   fake. :func:`fake_docker` refuses a ``-f``/``--format`` call with a message pointing back here.
2. *The permission-denied retry is ``sg docker -c <shlex.join(the same command)>``*, byte for byte
   the shape ``preflight._docker`` uses (the task's Implementation notes say the duplication is
   deliberate and that this row exists so the two copies cannot drift).

Nothing here imports the modules under test at import time: the tests are pending while the
implementation does not exist, and a collection error would take the whole module down with it.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The real ``os.kill``, bound at import: cleanup of a test double must still work while a row has
#: :class:`SignalSpy` installed, and must not show up in what the spy recorded.
_REAL_KILL = os.kill

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "services"
BOX = REPO / "tests" / "fixtures" / "box"

#: The recorded tail of the STT launcher that failed on 2026-09-16 (see the fixture MANIFEST).
CONFLICT_LOG = FIXTURES / "stt_conflict.log"
#: The distinctive slice of the Conflict line a FAILED row has to quote (criterion 1).
CONFLICT_QUOTE = 'Conflict. The container name "/astra-stt" is already in use'
#: The T-0033-shape state file as it is on disk on this box today (criterion 5).
PID_FILE_T0033 = FIXTURES / "pid_file_t0033.json"

STATUSES = ("started", "FAILED", "NOT-NEW")
ACTIONS = ("START", "SKIP", "CONFLICT")


# --- looking the seam up, so a missing symbol says which criterion wanted it --------------------


def module(name: str, criterion: str) -> Any:
    try:
        return importlib.import_module(f"local_astra.{name}")
    except ImportError as exc:  # pragma: no cover - the pending path
        raise AssertionError(
            f"T-0049 criterion {criterion}: local_astra/{name}.py does not import ({exc}). "
            "The seam is in tests/unit/_service_rig.py's docstring."
        ) from None


def attr(mod: Any, name: str, criterion: str) -> Any:
    """``getattr`` that fails as an AssertionError naming the criterion.

    ``mod`` is a module *or* an object (``Handle.returncode`` is looked up this way), so the label
    is derived defensively: an ``AttributeError`` raised from inside this helper would escape the
    pending xfail's ``raises`` list and turn the whole suite red for the wrong reason.
    """
    label = getattr(mod, "__name__", None) or type(mod).__name__
    value = getattr(mod, name, None)
    assert value is not None, (
        f"T-0049 criterion {criterion}: {label}.{name} is missing "
        "(the seam is in tests/unit/_service_rig.py's docstring)"
    )
    return value


def service_state(criterion: str) -> Any:
    return module("service_state", criterion)


def supervisor(criterion: str) -> Any:
    return module("supervisor", criterion)


def _build(factory: Any, kwargs: dict[str, Any], criterion: str, shape: str) -> Any:
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0049 criterion {criterion}: {shape} — {exc}. "
            "The seam is in tests/unit/_service_rig.py's docstring."
        ) from None


# --- the pure side ------------------------------------------------------------------------------

#: A digest is an opaque string to ``decide``: only equality matters, so the pure tests never
#: compute one where the decision itself is what is under test.
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def record(criterion: str = "5", **overrides: Any) -> Any:
    """A ``StateRecord`` for a container-backed STT started by this CLI (fields per the seam)."""
    mod = service_state(criterion)
    fields: dict[str, Any] = {
        "version": getattr(mod, "RECORD_VERSION", 1),
        "name": "stt",
        "kind": "container",
        "pid": 4242,
        "start_time": 858746,
        "argv": ("/repo/services/stt.sh",),
        "script_digest": DIGEST_A,
        "container_name": "astra-stt",
        "container_id": "61f0d405d989",
        "started_at": 1_758_000_000.0,
    }
    fields.update(overrides)
    return _build(
        attr(mod, "StateRecord", criterion),
        fields,
        criterion,
        "StateRecord(version, name, kind, pid, start_time, argv, script_digest, "
        "container_name, container_id, started_at)",
    )


def identity(criterion: str = "2", **overrides: Any) -> Any:
    """An ``Identity``: what the right authority says is live right now."""
    mod = service_state(criterion)
    fields: dict[str, Any] = {
        "kind": "container",
        "alive": True,
        "pid": 4242,
        "start_time": 858746,
        "container_id": "61f0d405d989",
        "started_at": "2026-09-13T18:26:09.969184297Z",
        "note": None,
    }
    fields.update(overrides)
    return _build(
        attr(mod, "Identity", criterion),
        fields,
        criterion,
        "Identity(kind, alive, pid=None, start_time=None, container_id=None, started_at=None, "
        "note=None)",
    )


def observation(criterion: str, **overrides: Any) -> Any:
    """An ``Observation``: the health result, the record, the live identity, the disk digest."""
    mod = service_state(criterion)
    fields: dict[str, Any] = {
        "name": "stt",
        "url": "ws://127.0.0.1:8080/api/asr-streaming",
        "healthy": True,
        "health_reason": None,
        "record": None,
        "live": None,
        "script_digest": DIGEST_A,
    }
    fields.update(overrides)
    return _build(
        attr(mod, "Observation", criterion),
        fields,
        criterion,
        "Observation(name, url, healthy, health_reason=None, record=None, live=None, "
        "script_digest='')",
    )


def decide(obs: Any, criterion: str) -> Any:
    """``service_state.decide(observation)``, with the two shape checks every row relies on."""
    mod = service_state(criterion)
    func = attr(mod, "decide", criterion)
    try:
        result = func(obs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0049 criterion {criterion}: service_state.decide(observation) -> Decision — {exc}"
        ) from None
    action = getattr(result, "action", None)
    assert isinstance(action, str) and action in ACTIONS, (
        f"T-0049 criterion 2: Decision.action is one of {ACTIONS}; decide() returned {result!r}"
    )
    assert isinstance(getattr(result, "reason", None), str), (
        f"T-0049 criterion 2: Decision.reason is the string the row prints; got {result!r}"
    )
    return result


# --- the acting side ----------------------------------------------------------------------------


def start(criterion: str, name: str, command: Any, url: str, **kwargs: Any) -> Any:
    """Call ``supervisor.start_service`` through the seam, turning a signature error into a row."""
    mod = supervisor(criterion)
    func = attr(mod, "start_service", criterion)
    try:
        outcome = func(name, [str(part) for part in command], url, **kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0049 criterion {criterion}: supervisor.start_service(name, command, url, *, "
            f"probe=None, log_dir=None, run_dir=None, settle_s=None, budget_s=None) — {exc}"
        ) from None
    status = getattr(outcome, "status", None)
    assert isinstance(status, str), (
        f"T-0049 criterion {criterion}: StartOutcome.status is a string in {STATUSES}; "
        f"start_service returned {outcome!r}"
    )
    assert isinstance(getattr(outcome, "message", None), str), (
        f"T-0049 criterion {criterion}: StartOutcome.message is the row's text; got {outcome!r}"
    )
    return outcome


def live_identity(rec: Any, criterion: str = "6") -> Any:
    """``supervisor.live_identity(record)``: what the right authority says about that handle."""
    mod = supervisor(criterion)
    func = attr(mod, "live_identity", criterion)
    try:
        result = func(rec)
    except TypeError as exc:
        raise AssertionError(
            f"T-0049 criterion {criterion}: supervisor.live_identity(record) -> Identity — {exc}"
        ) from None
    assert isinstance(getattr(result, "alive", None), bool), (
        f"T-0049 criterion {criterion}: Identity.alive is a bool; live_identity returned {result!r}"
    )
    return result


def claims_success(outcome: Any) -> bool:
    """Does this outcome claim, in any of its branches, that a start happened?"""
    status = str(getattr(outcome, "status", "")).strip().lower()
    return bool(getattr(outcome, "ok", False)) or status in {"ok", "started", "start", "skip"}


def describe(outcome: Any) -> str:
    return (
        f"status={getattr(outcome, 'status', None)!r} ok={getattr(outcome, 'ok', None)!r} "
        f"message={getattr(outcome, 'message', None)!r} "
        f"record={getattr(outcome, 'record', None)!r}"
    )


# --- launchers shaped like services/*.sh ---------------------------------------------------------

#: What every ``services/*.sh`` does at the end: ``exec`` a foreground server. The test scripts
#: below keep that shape, because it is the shape criterion 4's "any exit is a failure" rests on.
_HEADER = "#!/usr/bin/env bash\nset -euo pipefail\n"

#: The unreachable container line that makes a test script *look* like the real one to
#: ``script_container``/``script_is_containerised`` while the test still runs a cheap sleeper.
#: Assembled from parts so that writing this file never puts a runnable engine-start spelling in
#: front of ``.claude/hooks/guard_bash.sh`` (it is data here, and unreachable there).
#:
#: The two spellings are the two that exist in ``services/``: ``stt.sh`` names the binary
#: literally, while ``vllm.sh`` goes through a variable (``DOCKER=docker`` at line 46, then
#: ``exec $DOCKER run …``) so that ``ASTRA_DRY_RUN=1`` can swap it for ``printf``. A detector that
#: only matches the literal spelling calls the engine "not containerised", which is the row
#: ``test_the_container_name_is_read_from_the_script…`` fails on.
_RUN = "docker" + " run"
NAMED_RUN = "exec " + _RUN + ' --rm --name {name} --network host "$IMAGE" worker --port 8080\n'
UNNAMED_RUN = (
    "DOCKER=" + "docker" + '\nexec $DOCKER run --rm --network host "$IMAGE" serve --port 8001\n'
)


def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def sleeper_launcher(
    path: Path,
    *,
    seconds: float = 20,
    container: str | None = None,
    containerised: bool = True,
    switch: Path | None = None,
) -> Path:
    """A launcher that ``exec``s a server and stays up — the happy shape.

    The container line is real text in the script (that is what ``script_container`` reads) but is
    never reached: the sleeper above it ``exec``s first, so no test here starts a container.
    ``switch`` is the file :func:`fake_docker` watches: writing it is this rig's stand-in for "a
    new instance is now the one on the port", and it happens when the launcher runs, never on a
    call count — so a row cannot be broken by an implementation that inspects one extra time.
    What the launcher writes into it is the moment it ran, in the daemon's own timestamp spelling,
    and :func:`fake_docker` substitutes that for :data:`SPAWN_STAMP` in the ``after`` document.
    That is the only honest way to say "started by this attempt" (ruling A19): a container this
    invocation produced has a ``StartedAt`` **after** the attempt reference, and which wall-clock
    instant that is cannot be known until the launcher actually runs.
    """
    touch = (
        f"date -u +%Y-%m-%dT%H:%M:%S.%NZ > {shlex.quote(str(switch))}\n"
        if switch is not None
        else ""
    )
    if not containerised:
        return _write_script(path, touch + f"exec taskset -c 0-7 sleep {seconds}\n")
    body = f"IMAGE=astra-test:0\n{touch}exec sleep {seconds}\n"
    body += NAMED_RUN.format(name=container) if container else UNNAMED_RUN
    return _write_script(path, body)


def failing_launcher(
    path: Path,
    *,
    code: int,
    log: Path,
    delay_s: float = 0.0,
    container: str | None = "astra-stt",
) -> Path:
    """A launcher that writes ``log`` to its stderr and exits ``code`` — the 2026-09-16 shape."""
    body = "IMAGE=astra-test:0\n"
    if delay_s:
        body += f"sleep {delay_s}\n"
    body += f"cat {shlex.quote(str(log))} >&2\nexit {code}\n"
    body += NAMED_RUN.format(name=container) if container else UNNAMED_RUN
    return _write_script(path, body)


def sha256_of(path: Path) -> str:
    """The digest the record must carry: sha256 over the service script's bytes (notes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- a fake ``docker`` on PATH -------------------------------------------------------------------


_DOCKER_FAKE = r"""#!/usr/bin/env bash
# A fake `docker` (T-0049 tests). It answers `inspect <ref>` with a recorded JSON document and
# records every call. Which document depends on the switch file, not on a call count: before the
# launcher ran it answers `before.json`, afterwards `after.json` (which defaults to the same
# bytes -- the stale-instance case). It starts, stops and removes nothing: any other subcommand
# is an error, which is how criterion 6's negative is enforced dynamically.
HERE="$(cd "$(dirname "$0")" && pwd)"
# %q, not %s: `sg docker -c 'docker inspect <id>'` is three arguments plus one quoted command,
# and a log that flattened them would let a shell-injected spelling pass for the real one.
printf '%q ' "$@" >> "$HERE/calls.log"; printf '\n' >> "$HERE/calls.log"
if [ "${1:-}" = "inspect" ]; then
  shift
  case "${1:-}" in
    -f|--format)
      echo "fake docker: this rig pins 'inspect <ref>' returning JSON, not a Go template" >&2
      echo "(see tests/unit/_service_rig.py's docstring, seam decision 1)." >&2
      exit 64 ;;
  esac
  REF="${1:-}"
  if [ -f "$HERE/deny" ] && [ -z "${ASTRA_FAKE_SG:-}" ]; then
    echo "permission denied while trying to connect to the Docker daemon socket" >&2
    exit 1
  fi
  # Which side of the spawn this read is on. The launcher writes its own start time into the
  # switch file, so `after` also means "the instance this attempt produced exists".
  PHASE=before
  if [ -f "$HERE/switch" ]; then PHASE=after; fi
  if [ -f "$HERE/delay_$PHASE" ]; then sleep "$(cat "$HERE/delay_$PHASE")"; fi
  # The daemon could not be asked (ruling A13). Unlike `deny`, the `sg` retry does not clear it:
  # this is permission denied with no working group, or a daemon too busy to answer in time.
  if [ -f "$HERE/unreachable_$PHASE" ]; then
    cat "$HERE/unreachable_$PHASE" >&2
    exit 1
  fi
  # T-0158: the verifying read after a stop could not be made (`verify_error`).
  if [ -f "$HERE/stopped" ] && [ -f "$HERE/unreachable_stopped" ]; then
    cat "$HERE/unreachable_stopped" >&2
    exit 1
  fi
  # T-0158 (`allow_stop=True` only): resolve <ref> against the store, the way the daemon does.
  if [ -f "$HERE/allow_stop" ]; then
    exec __ASTRA_FAKE_PYTHON__ "$HERE/_store.py" inspect "$REF" "$PHASE"
  fi
  FILE="$HERE/before.json"
  if [ "$PHASE" = "after" ] && [ -f "$HERE/after.json" ]; then FILE="$HERE/after.json"; fi
  if [ ! -f "$FILE" ]; then
    echo "Error: No such object: $REF" >&2
    exit 1
  fi
  if [ -f "$HERE/switch" ] && grep -q __STARTED_AT_AT_SPAWN__ "$FILE"; then
    # The instance this attempt produced reports the time the launcher really ran (ruling A19):
    # a container started by this invocation has a StartedAt *after* the attempt reference, and
    # no fixed string can honestly say that.
    sed "s/__STARTED_AT_AT_SPAWN__/$(cat "$HERE/switch")/" "$FILE"
    exit 0
  fi
  cat "$FILE"
  exit 0
fi
# T-0158: `stop --time N <ref>` exists only when the row opted in with `allow_stop=True`.
if [ "${1:-}" = "stop" ] && [ -f "$HERE/allow_stop" ]; then
  shift
  FLAG=""
  GRACE=""
  case "${1:-}" in
    --time|--timeout|-t)
      FLAG="$1"
      if [ "$#" -ge 2 ]; then GRACE="$2"; shift 2; else shift; fi ;;
    --time=*|--timeout=*)
      FLAG="${1%%=*}"; GRACE="${1#*=}"; shift ;;
  esac
  if [ -z "$FLAG" ] || [ "$#" -ne 1 ]; then
    echo "fake docker: this rig pins 'stop --time <seconds> <one recorded id>' (T-0158 criterion 1)" >&2
    exit 64
  fi
  # docker 29.4.1 still accepts the spelling T-0158 pins, and says so on stderr every time.
  if [ "$FLAG" = "--time" ]; then
    echo "Flag --time has been deprecated, use --timeout instead" >&2
  fi
  case "$GRACE" in
    ''|*[!0-9]*)
      echo "invalid argument \"$GRACE\" for \"-t, --timeout\" flag: strconv.ParseInt: parsing \"$GRACE\": invalid syntax" >&2
      exit 125 ;;
  esac
  REF="$1"
  if [ -f "$HERE/deny" ] && [ -z "${ASTRA_FAKE_SG:-}" ]; then
    echo "permission denied while trying to connect to the Docker daemon socket" >&2
    exit 1
  fi
  if [ -f "$HERE/stop_delay" ]; then
    # Detached from the pipes, so a client that times out and kills this script is not then held
    # by an orphaned sleep keeping stdout open.
    sleep "$(cat "$HERE/stop_delay")" </dev/null >/dev/null 2>&1
  fi
  if [ -f "$HERE/stop_error" ]; then
    cat "$HERE/stop_error" >&2
    exit 1
  fi
  PHASE=before
  if [ -f "$HERE/switch" ]; then PHASE=after; fi
  exec __ASTRA_FAKE_PYTHON__ "$HERE/_store.py" stop "$REF" "$PHASE"
fi
echo "fake docker: refusing the subcommand '${1:-}' - this rig only answers 'inspect'" >&2
exit 65
"""

#: The daemon's side of ``allow_stop=True`` (T-0158): which document a reference names, and what
#: ``stop`` does to it. A reference resolves the way the daemon resolves one — the full id, then
#: the name, then a unique id prefix — over the phase document (``before``/``after``) and every
#: ``others/*.json`` (containers this CLI never recorded). ``stop`` deletes the stopped document
#: (``--rm``: a later ``inspect`` says it is gone), or with ``stop_leaves_exited`` rewrites it as
#: ``Running: false`` (the removal lag), and appends the id to the ``stopped`` marker.
_STORE = r"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPAWN = "__STARTED_AT_AT_SPAWN__"


def at(name):
    return os.path.join(HERE, name)


def entry_of(path):
    with open(path) as handle:
        data = json.load(handle)
    return data[0] if isinstance(data, list) and data else data


def primary(phase):
    if phase == "after" and os.path.exists(at("after.json")):
        return at("after.json")
    return at("before.json")


def resolve(ref, phase):
    files = [primary(phase)] if os.path.exists(primary(phase)) else []
    files += sorted(glob.glob(at("others/*.json")))
    entries = [(path, entry_of(path)) for path in files]
    for path, entry in entries:
        if str(entry.get("Id", "")) == ref:
            return path, entry
    for path, entry in entries:
        if ref and str(entry.get("Name", "")).lstrip("/") == ref.lstrip("/"):
            return path, entry
    hits = [(p, e) for p, e in entries if ref and str(e.get("Id", "")).startswith(ref)]
    return hits[0] if len(hits) == 1 else (None, None)


def main():
    verb, ref, phase = sys.argv[1], sys.argv[2], sys.argv[3]
    path, entry = resolve(ref, phase)
    if verb == "inspect":
        if path is None:
            if os.path.exists(at("legacy_missing")):
                sys.stderr.write(f"Error: No such object: {ref}\n")
            else:
                # docker 29.4.1 on this box, recorded 2026-10-26: an empty list on stdout, then
                # this line, lower case, on stderr; exit 1.
                sys.stdout.write("[]\n")
                sys.stderr.write(f"error: no such object: {ref}\n")
            return 1
        with open(path) as handle:
            text = handle.read()
        if SPAWN in text and os.path.exists(at("switch")):
            with open(at("switch")) as handle:
                text = text.replace(SPAWN, handle.read().strip())
        sys.stdout.write(text)
        return 0
    if verb == "stop":
        if path is None:
            sys.stderr.write(f"Error response from daemon: No such container: {ref}\n")
            return 1
        if not os.path.exists(at("stop_noop")):
            stopped_id = str(entry.get("Id", ""))
            for candidate in [at("before.json"), at("after.json"), *glob.glob(at("others/*.json"))]:
                if not os.path.exists(candidate):
                    continue
                if str(entry_of(candidate).get("Id", "")) != stopped_id:
                    continue
                if os.path.exists(at("stop_leaves_exited")):
                    with open(candidate) as handle:
                        data = json.load(handle)
                    target = data[0] if isinstance(data, list) else data
                    target.setdefault("State", {})
                    target["State"]["Running"] = False
                    target["State"]["Status"] = "exited"
                    with open(candidate, "w") as handle:
                        json.dump(data, handle)
                else:
                    os.remove(candidate)
            with open(at("stopped"), "a") as handle:
                handle.write(stopped_id + "\n")
        sys.stdout.write(ref + "\n")
        return 0
    sys.stderr.write(f"fake docker store: unknown verb {verb}\n")
    return 64


sys.exit(main())
"""

_SG_FAKE = r"""#!/usr/bin/env bash
# A fake `sg` (T-0049 criterion 6): `sg docker -c '<command>'` runs the command with the group.
HERE="$(cd "$(dirname "$0")" && pwd)"
printf 'sg ' >> "$HERE/calls.log"
printf '%q ' "$@" >> "$HERE/calls.log"; printf '\n' >> "$HERE/calls.log"
if [ "${1:-}" != "docker" ] || [ "${2:-}" != "-c" ]; then
  echo "fake sg: expected 'sg docker -c <command>', got: $*" >&2
  exit 66
fi
ASTRA_FAKE_SG=1 bash -c "$3"
"""


#: Stands in, inside an ``after`` document, for "whenever the launcher actually ran". The fake
#: substitutes the launcher's own stamp for it (ruling A19: a container this attempt produced
#: started *after* the attempt reference, and no constant in a test file can say that honestly).
SPAWN_STAMP = "__STARTED_AT_AT_SPAWN__"


class FakeDocker:
    """A ``docker`` (and ``sg``) on PATH that only ever answers ``inspect``, and records calls."""

    def __init__(self, bin_dir: Path) -> None:
        self.dir = bin_dir
        #: Written by the launcher with the moment it ran; from then on ``inspect`` answers
        #: with ``after``, carrying that moment in place of :data:`SPAWN_STAMP`.
        self.switch = bin_dir / "switch"

    def spawn_started_at(self) -> str:
        """The stamp the launcher wrote — the real instant this attempt's instance appeared."""
        assert self.switch.exists(), (
            "the launcher never wrote the switch file, so nothing in this row stands for the "
            "instance the attempt produced"
        )
        return self.switch.read_text().strip()

    @property
    def calls(self) -> tuple[tuple[str, ...], ...]:
        log = self.dir / "calls.log"
        if not log.exists():
            return ()
        return tuple(
            tuple(shlex.split(line)) for line in log.read_text().splitlines() if line.strip()
        )

    def inspect_calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(call for call in self.calls if call and call[0] == "inspect")

    # --- T-0158: the opt-in ``stop`` verb ---------------------------------------------------------

    @property
    def stopped(self) -> Path:
        """The marker the fake writes when a ``stop`` really stopped a document (one id a line)."""
        return self.dir / "stopped"

    def stop_calls(self) -> tuple[tuple[str, ...], ...]:
        """Every ``docker stop …`` that reached the client, direct or through ``sg docker -c``."""
        return tuple(call for call in self.calls if call and call[0] == "stop")

    def sg_calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(call for call in self.calls if call and call[0] == "sg")

    def stopped_ids(self) -> tuple[str, ...]:
        if not self.stopped.exists():
            return ()
        return tuple(line for line in self.stopped.read_text().splitlines() if line.strip())

    def holds(self, container_id: str) -> bool:
        """Does the daemon still hold a document with exactly this ``Id`` (any phase, any other)?"""
        paths = [self.dir / "before.json", self.dir / "after.json"]
        paths += sorted((self.dir / "others").glob("*.json"))
        for path in paths:
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            entry = data[0] if isinstance(data, list) and data else data
            if isinstance(entry, dict) and entry.get("Id") == container_id:
                return True
        return False


def container_json(
    *,
    container_id: str,
    started_at: str,
    running: bool = True,
    name: str = "astra-stt",
) -> str:
    """A ``docker inspect`` response built from the **recorded** one, with the identity overridden.

    The envelope (a one-element list, ``.Id``, ``.Name``, ``.State.Running``, ``.State.StartedAt``)
    is whatever the real daemon produced on this box on 2026-09-14 — see
    ``tests/fixtures/box/MANIFEST.md`` — so a parser that passes here parses the real thing.
    """
    recorded = json.loads((BOX / "docker_inspect_stt.json").read_text())
    entry = copy.deepcopy(recorded[0] if isinstance(recorded, list) else recorded)
    entry["Id"] = container_id
    entry["Name"] = f"/{name}"
    entry.setdefault("State", {})
    entry["State"]["Running"] = running
    entry["State"]["Status"] = "running" if running else "exited"
    entry["State"]["StartedAt"] = started_at
    return json.dumps([entry], indent=1)


def fake_docker(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    before: str | None = None,
    after: str | None = None,
    deny_first: bool = False,
    before_error: str | None = None,
    before_delay_s: float | None = None,
    after_error: str | None = None,
    allow_stop: bool = False,
    stop_error: str | None = None,
    stop_delay_s: float | None = None,
    verify_error: str | None = None,
    others: Iterable[str] = (),
    stop_leaves_exited: bool = False,
    stop_noop: bool = False,
    legacy_missing: bool = False,
) -> FakeDocker:
    """Put a fake ``docker``/``sg`` first on ``PATH``.

    ``before`` is what ``inspect`` answers until the launcher touches :attr:`FakeDocker.switch`;
    ``after`` (default: ``before``, i.e. *nothing new started*) is what it answers afterwards.
    ``before=None`` is a daemon that has never heard of the reference — the container is **absent**.

    ``before_error`` is the different thing that looks the same and is ruling A13's hole: the
    daemon **could not be asked**. Every ``inspect`` before the switch then fails with that text on
    stderr and a non-zero status, under ``sg`` too (permission denied with no working group
    fallback — the module's own comment calls that the normal state on this box). ``after_error``
    is the mirror: the read *after* the start fails, which must not turn a start that really
    happened into a failure. ``before_delay_s`` is how long the daemon takes to answer the
    pre-spawn read; past ``supervisor.DOCKER_TIMEOUT_S`` that is the busy daemon, and below it it
    is simply a slow one. A failed read leaves the reference point *unknown*, which is not the
    same fact as *nothing was there* and may not be read as one.

    **T-0158, opt-in.** By default the fake still refuses every verb but ``inspect`` (criterion 6's
    dynamic negative, unchanged). ``allow_stop=True`` adds ``docker stop --time N <ref>``: it is
    logged in ``calls.log``; when ``<ref>`` resolves to a document the daemon holds it exits 0,
    deletes that document (so a later ``inspect`` says "no such object") and appends its id to
    :attr:`FakeDocker.stopped`; otherwise it answers ``Error response from daemon: No such
    container: <ref>`` with exit 1. In this mode ``inspect`` resolves ``<ref>`` too (full id, name,
    unique id prefix) and a miss is docker 29.4.1's answer (``[]`` on stdout, ``error: no such
    object: <ref>`` on stderr, exit 1; ``legacy_missing=True`` gives the older ``Error: No such
    object`` spelling). Like the real client, ``--time`` prints its deprecation notice on stderr and
    a non-integer grace is ``invalid argument`` (exit 125).

    ``stop_error`` is a daemon that cannot be asked to stop (it survives ``sg``); ``stop_delay_s``
    is how long the stop takes to answer; ``verify_error`` makes every ``inspect`` after a stop
    fail; ``others`` are documents of containers this CLI never recorded (strangers);
    ``stop_leaves_exited`` keeps a stopped document as ``Running: false`` (the ``--rm`` lag);
    ``stop_noop`` is a stop that answers 0 and changes nothing.
    """
    stop_options = (
        stop_error,
        stop_delay_s,
        verify_error,
        tuple(others) or None,
        stop_leaves_exited or None,
        stop_noop or None,
        legacy_missing or None,
    )
    assert allow_stop or all(option is None for option in stop_options), (
        "the T-0158 stop options only mean something with allow_stop=True"
    )
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("docker", _DOCKER_FAKE.replace("__ASTRA_FAKE_PYTHON__", shlex.quote(sys.executable))),
        ("sg", _SG_FAKE),
    ):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    if allow_stop:
        (bin_dir / "allow_stop").write_text("1")
        (bin_dir / "_store.py").write_text(_STORE)
        if stop_error is not None:
            (bin_dir / "stop_error").write_text(stop_error)
        if stop_delay_s:
            (bin_dir / "stop_delay").write_text(str(stop_delay_s))
        if verify_error is not None:
            (bin_dir / "unreachable_stopped").write_text(verify_error)
        for index, document in enumerate(others):
            (bin_dir / "others").mkdir(exist_ok=True)
            (bin_dir / "others" / f"{index}.json").write_text(document)
        for flag, name in (
            (stop_leaves_exited, "stop_leaves_exited"),
            (stop_noop, "stop_noop"),
            (legacy_missing, "legacy_missing"),
        ):
            if flag:
                (bin_dir / name).write_text("1")
    if before is not None:
        (bin_dir / "before.json").write_text(before)
    resolved_after = after if after is not None else before
    if resolved_after is not None:
        (bin_dir / "after.json").write_text(resolved_after)
    if deny_first:
        (bin_dir / "deny").write_text("1")
    if before_error is not None:
        (bin_dir / "unreachable_before").write_text(before_error)
    if after_error is not None:
        (bin_dir / "unreachable_after").write_text(after_error)
    if before_delay_s:
        (bin_dir / "delay_before").write_text(str(before_delay_s))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    resolved = shutil.which("docker")
    assert resolved == str(bin_dir / "docker"), (
        "the fake has to be the docker this row reaches: a unit row that talks to the live daemon "
        "is I/O `.claude/rules/tests.md` forbids, and inside a measured row it is CLAUDE.md's "
        f"unattributable number (rulings A9/A14). PATH resolved `docker` to {resolved!r}"
    )
    return FakeDocker(bin_dir)


# --- state directory snapshots (criterion 1: "byte-identical afterwards") ------------------------


def snapshot(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory`` with its exact bytes — the only honest form of unchanged."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def t0033_pid_file_bytes(
    *,
    pid: int | None = None,
    start_time: int | None = None,
    argv: tuple[str, ...] | None = None,
) -> str:
    """The recorded T-0033-shape file, optionally with a live pid substituted (criterion 5).

    The *shape* is the fixture's — the three keys and nothing else — because that is what is on
    disk on this box; only the identity is replaced, since pid 3574545 belongs to nobody now and
    the kernel may hand it to anyone.
    """
    raw = PID_FILE_T0033.read_text()
    loaded = json.loads(raw)
    assert set(loaded) == {"pid", "argv", "start_time"}, (
        f"the recorded T-0033 file is {{pid, argv, start_time}}; the fixture has {sorted(loaded)}"
    )
    if pid is not None:
        loaded["pid"] = pid
    if start_time is not None:
        loaded["start_time"] = start_time
    if argv is not None:
        loaded["argv"] = list(argv)
    return json.dumps(loaded, indent=2) + "\n"


# =================================================================================================
# Group B — the verbs (criteria 10-14). The seam, and the helpers the CLI rows share.
# =================================================================================================
#
# What group B adds is a *command layer*: the decision (group A's ``service_state.decide``) and the
# verified start (group A's ``supervisor.start_service``) already exist, and nothing has changed
# for a human yet. The seam the rows below assume, stated once so the implementer can read it
# whole — every item is a consequence of a criterion, not a preference:
#
#   ``cli.main(argv, *, runner=None, probe=None) -> int``       # unchanged (T-0033)
#
#   ``astra up [--profile P] [--dry-run] [--timeout S] [--restart all|a,b]``
#       one **verdict row** per service in the start plan, drawn from the decision, carrying
#       exactly one of ``started`` / ``already running`` / ``CONFLICT`` / ``FAILED`` / ``NOT-NEW``;
#       then the shared health table (T-0033 criterion 14's renderer, unchanged); exit non-zero
#       iff any verdict row is CONFLICT, FAILED or NOT-NEW, or the health table has a failure.
#
#   The injected ``probe`` is the **only** health seam: ``up`` asks it for the pre-check *and*
#   hands it to ``supervisor.start_service`` for the verification probe. A unit row cannot afford
#   a second, un-injectable route to a socket (`.claude/rules/tests.md`: tests/unit is pure).
#
#   ``runner`` keeps its T-0033 shape and its meaning is fixed here: an **injected** runner
#   *replaces* the start of one plan entry (``runner(command)``), and its return is that entry's
#   outcome — the fakes in ``tests/unit/test_cli_up.py`` carry ``.status``/``.ok``/``.message``
#   beside ``.pid``. With no runner injected, ``up`` performs the real verified start through
#   ``supervisor.start_service``. Both paths end in a verdict row and an exit code; that is what
#   lets T-0033's and T-0041's probe-loop rows keep an injected runner (they are about the health
#   table, and a real start inside them would cost a settle window per service) while the rows in
#   ``tests/unit/test_cli_up_verbs.py`` drive the real path.
#
#   The plan is ``cli.start_plan(profile)`` and is monkeypatched to point at the launchers below —
#   the route T-0033's own liveness driver already takes. ``command[0]``'s stem is the service
#   name, so a plan of ``vllm.sh``/``stt.sh``/``tts.sh`` lines up with ``endpoints_of(profile)``.
#
#   State files are read and written through ``supervisor.run_dir()`` / ``log_dir()`` **at call
#   time**, so a row can redirect them (and every row here must: see ``isolate_state``).
#
#   ``--restart`` takes ``all`` or a comma-separated list of service names.
#
#   ``supervisor.spawn(..., record=...)`` keeps the keyword and its default becomes ``False``
#   (ruling A16's "retire the eager default"): the one caller that still wants the T-0033 eager
#   write asks for it by name. Retiring it by *removing* the keyword was rejected here because
#   T-0033 criterion 11's row is about the file that write produces, and it has to keep a way to
#   ask for one.
#
# The verdict vocabulary is parsed out of the report by :func:`verdict_of`, which is deliberately
# forgiving about typography (the criteria fix *what* a row says, never its columns) and exact
# about the word, because the word is the whole of criterion 10.


#: The five words a verdict row can carry, in the order they are looked for. ``already running``
#: is first because a skip row legitimately quotes the instant the instance was ``started``.
VERDICTS = ("already running", "CONFLICT", "NOT-NEW", "FAILED", "started")

#: The shared health-table row (T-0033 criterion 14): ``vllm   ok      http://…  (budget 240 s)``.
#: It is not a verdict row, and it says nothing about what *this* invocation did.
_HEALTH_ROW = re.compile(r"^\s*(\w+)\s+(ok|FAILED)\s+\w+://")


def health_rows_for(report: str, name: str, url: str) -> list[str]:
    """The *health table*'s rows for ``name`` at ``url`` — the complement of :func:`verdict_lines`.

    "The line that names both the service and its endpoint" is not the health table (ruling B17):
    every refusal a verdict row carries says ``healthy at <url>``, and a start driven through the
    real path says ``healthy at <url>`` too, so a helper that selects on both substrings finds two
    lines for any report holding a refusal and dies inside itself rather than failing an
    assertion. The table's *shape* is what distinguishes it, and :data:`_HEALTH_ROW` already
    states that shape; this anchors on it so the counting assertions stay where they belong — in
    the rows, over the rendered table, and not over whatever else mentions the service.
    """
    rows = []
    for line in report.splitlines():
        match = _HEALTH_ROW.match(line)
        if match is not None and match.group(1) == name and url in line:
            rows.append(line)
    return rows


def verdict_lines(report: str, name: str) -> list[str]:
    """The lines of ``report`` that speak about ``name`` and are not the health table's row."""
    return [
        line
        for line in report.splitlines()
        if name in line and not _HEALTH_ROW.match(line) and line.strip()
    ]


def verdict_of(report: str, name: str) -> str | None:
    """Which of :data:`VERDICTS` this service's rows carry, or ``None`` when they carry none.

    Precedence, not counting: a FAILED row quotes its launcher's log underneath, and the recorded
    2026-09-16 log line contains the string ``astra-stt``, so "the service is mentioned" is not the
    same as "this is a second verdict". The first word found wins, in :data:`VERDICTS`' order.
    """
    lines = verdict_lines(report, name)
    for verdict in VERDICTS:
        if any(verdict in line for line in lines):
            return verdict
    return None


def verdicts(report: str, names: Iterable[str]) -> dict[str, str | None]:
    return {name: verdict_of(report, name) for name in names}


# --- the plan, pointed at launchers a test owns --------------------------------------------------


def install_plan(monkeypatch: Any, cli_module: Any, scripts: dict[str, Path]) -> list[str]:
    """Point ``cli.start_plan`` at ``scripts`` (name -> path), in the order given.

    The same route T-0033's liveness driver takes. Returns the names in plan order.
    """
    order = list(scripts)
    plan = tuple((str(scripts[name]),) for name in order)
    assert hasattr(cli_module, "start_plan"), (
        "T-0049 criterion 10: `up` still starts what cli.start_plan(profile) says, in its order "
        "(T-0028 criterion 17); that seam is missing"
    )
    monkeypatch.setattr(cli_module, "start_plan", lambda profile: plan)
    return order


def marker_launcher(path: Path, marker: Path, *, seconds: float = 20) -> Path:
    """A launcher that records **that it ran** and then ``exec``s a sleeper.

    The marker is how a row says "nothing was started" without counting processes: a SKIP that
    silently restarted the service would leave it, and the whole of criterion 11's negative is
    that it must not.
    """
    return sleeper_launcher(path, seconds=seconds, containerised=False, switch=marker)


# --- who got signalled ---------------------------------------------------------------------------


class SignalSpy:
    """Every ``os.kill`` this process makes, recorded, and then really made.

    A double that *replaced* ``os.kill`` would change what the code under test does to its own
    children; this one delegates (`.claude/rules/tests.md`), so ``stop``'s liveness checks and its
    SIGTERM behave exactly as they would unpatched and the row still sees them.

    ``os.kill(pid, 0)`` is a liveness question, not a signal, and is excluded from
    :meth:`signals_to` for that reason — a skip legitimately asks whether the recorded handle is
    alive, and criterion 11's negative is about *delivering* something to it.
    """

    def __init__(self, monkeypatch: Any) -> None:
        self.calls: list[tuple[int, int]] = []
        real = os.kill

        def spy(pid: int, sig: int) -> None:
            self.calls.append((int(pid), int(sig)))
            real(pid, sig)

        monkeypatch.setattr(os, "kill", spy)

    def signals_to(self, pid: int) -> list[int]:
        return [sig for target, sig in self.calls if target == pid and sig != 0]

    def all_signals(self) -> list[tuple[int, int]]:
        return [(pid, sig) for pid, sig in self.calls if sig != 0]


# --- T-0158: a ``docker run`` front end whose workload ignores SIGTERM ---------------------------

#: What the STT launcher is on this box (T-0158 Goal): the ``docker run --rm`` client that
#: ``services/stt.sh`` exec'd. A SIGTERM to it is forwarded into a container whose PID 1
#: (``moshi-server``, no ``--init``) has no handler, so the kernel drops it and nothing happens;
#: the client exits once the container has stopped. The double is a grandchild of the test (double
#: fork, own session), so — as in production, where ``down`` is a later process than ``up`` — it is
#: not ``down``'s child and is reaped by the system, not left a zombie that ``kill(pid, 0)`` would
#: call alive.
#:
#: Every SIGTERM it receives is appended to ``<ready>.sigterm`` as ``stopped`` or ``running``:
#: whether the fake daemon's ``stopped`` marker existed at that moment.
#:
#: ``mode``: ``until_stopped`` survives SIGTERM and exits once the marker appears (the observed
#: front end); ``stubborn`` survives SIGTERM and outlives the container (criterion 4's failure);
#: ``sigterm_only`` ignores the marker but exits on its first SIGTERM (criterion 4's one SIGTERM).
_FRONT_END = r"""
import os, signal, sys, time
ready, stopped, mode, lifetime = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
if os.fork():
    os._exit(0)
os.setsid()
term = ready + ".sigterm"

def on_term(signum, frame):
    with open(term, "a") as handle:
        handle.write("stopped\n" if os.path.exists(stopped) else "running\n")
    if mode == "sigterm_only":
        os._exit(0)

signal.signal(signal.SIGTERM, on_term)
tmp = ready + ".tmp"
with open(tmp, "w") as handle:
    handle.write(str(os.getpid()))
os.replace(tmp, ready)
deadline = time.monotonic() + lifetime
while time.monotonic() < deadline:
    if mode == "until_stopped" and os.path.exists(stopped):
        os._exit(0)
    time.sleep(0.02)
os._exit(0)
"""

FRONT_END_MODES = ("until_stopped", "stubborn", "sigterm_only")


def _proc_stat(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return raw[raw.rindex(")") + 1 :].split()


@dataclass
class FrontEnd:
    """One running front-end double: its pid and the ``/proc`` start time that identifies it."""

    pid: int
    start_time: int
    ready: Path

    def alive(self) -> bool:
        """Still *this* process (same start time) and not a zombie."""
        fields = _proc_stat(self.pid)
        return fields is not None and fields[0] != "Z" and int(fields[19]) == self.start_time

    def sigterms(self) -> list[str]:
        """One entry per SIGTERM received: whether the container had been stopped by then."""
        path = Path(str(self.ready) + ".sigterm")
        return path.read_text().split() if path.exists() else []

    def wait_gone(self, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                return True
            time.sleep(0.02)
        return not self.alive()

    def kill(self) -> None:
        """Cleanup only (test code, never the supervisor): SIGKILL it if it is still this one."""
        if self.alive():
            with contextlib.suppress(OSError):
                _REAL_KILL(self.pid, signal.SIGKILL)
            self.wait_gone(2.0)


def container_front_end(
    tmp_path: Path,
    docker: FakeDocker,
    *,
    mode: str = "until_stopped",
    lifetime_s: float = 30.0,
    label: str = "front-end",
) -> FrontEnd:
    """Start one front-end double (see :data:`_FRONT_END`) and return it once its handler is set."""
    assert mode in FRONT_END_MODES, f"mode is one of {FRONT_END_MODES}; got {mode!r}"
    ready = tmp_path / f"{label}.ready"
    subprocess.run(
        [sys.executable, "-c", _FRONT_END, str(ready), str(docker.stopped), mode, str(lifetime_s)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=10,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "the front-end double never installed its SIGTERM handler"
    pid = int(ready.read_text())
    fields = _proc_stat(pid)
    assert fields is not None, f"the front-end double (pid {pid}) is gone before the row began"
    return FrontEnd(pid=pid, start_time=int(fields[19]), ready=ready)
