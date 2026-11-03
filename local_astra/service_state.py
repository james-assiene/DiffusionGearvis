"""What this CLI started, and whether it is still what is running (T-0049 group A).

The pure middle of *observe → decide → act → verify → record*: no clock, no I/O, no subprocess.
Observing, acting and recording live in :mod:`local_astra.supervisor`; this module only decides,
because the whole of the 2026-09-16 failure is that deciding and acting were never separated —
``astra up`` ran every command in the start plan unconditionally and then let a health probe
certify the result.

The decision is D1's table, taken **before** anything is spawned, from four facts:

    not healthy                                                      -> START
    healthy + the record names the *live* instance + digest unchanged -> SKIP     (exit 0, loud)
    healthy + record missing / handle gone / digest changed           -> CONFLICT (nothing starts)

A health probe answers for whatever is on the port, so health alone can only ever say "START is
not needed"; it can never say the instance is *ours*. That is what :class:`StateRecord` (what this
CLI wrote when it verified a start) and :class:`Identity` (what the right authority — ``/proc`` for
a host process, ``docker inspect`` for a container — says is live *now*) are for.

The record's honest limit, stated here because it is easy to over-read: ``script_digest`` catches a
changed *script*, not a changed environment — the plugin toggle and the speech-device override
(``services/*.sh`` read them; they are deliberately not named here, T-0033 criterion 20) select
different engines from the same bytes. A drifted digest is therefore a refusal; an unchanged one
is not a proof that the configuration is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bumped when the on-disk shape of :class:`StateRecord` changes. A file without a ``version`` is a
#: T-0033 pid file (``{pid, argv, start_time}``) and is *not* a record: it cannot certify anything.
RECORD_VERSION = 1

#: The three outcomes of :func:`decide` (D1). Nothing else is an action.
ACTIONS = ("START", "SKIP", "CONFLICT")

#: What a handle can be. ``container`` needs a name this CLI can ask the daemon about; a container
#: whose name is unknown (the engine, until T-0048 passes ``--name``) degrades to ``process``.
KINDS = ("process", "container")


@dataclass(frozen=True, kw_only=True)
class StateRecord:
    """What *this* CLI started, written only after the start was verified.

    Not a second source of truth about what is running: whether the handle still runs is asked of
    ``/proc`` or of the docker daemon (:func:`local_astra.supervisor.live_identity`). ``pid`` is
    always the launcher — for a container that is the front-end process signals are proxied
    through, which is why ``down`` may signal it and must never ``SIGKILL`` it (T-0033 ruling A16).
    """

    version: int = RECORD_VERSION
    name: str = ""
    kind: str = "process"
    pid: int = 0
    #: ``/proc/<pid>/stat`` field 22 (clock ticks since boot): the test a reused pid fails.
    start_time: int | None = None
    argv: tuple[str, ...] = ()
    #: sha256 over the service script's bytes, as it was when this start was verified.
    script_digest: str = ""
    container_name: str | None = None
    container_id: str | None = None
    #: The wall clock of the verified start (``time.time()``).
    started_at: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """The on-disk shape. The three T-0033 keys keep their names and meaning, so
        ``read_pid_file``/``stop``/``pid_files`` read a record without knowing about one."""
        return {
            "version": int(self.version),
            "name": self.name,
            "kind": self.kind,
            "pid": int(self.pid),
            "start_time": None if self.start_time is None else int(self.start_time),
            "argv": [str(part) for part in self.argv],
            "script_digest": self.script_digest,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "started_at": float(self.started_at),
        }

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> StateRecord | None:
        """Parse one; ``None`` for anything that is not a versioned record (a T-0033 pid file)."""
        if "version" not in data or "kind" not in data:
            return None
        start_time = data.get("start_time")
        argv = data.get("argv") or []
        return cls(
            version=int(str(data["version"])),
            name=str(data.get("name") or ""),
            kind=str(data["kind"]),
            pid=int(str(data.get("pid") or 0)),
            start_time=None if start_time is None else int(str(start_time)),
            argv=tuple(str(part) for part in argv) if isinstance(argv, list | tuple) else (),
            script_digest=str(data.get("script_digest") or ""),
            container_name=_optional(data.get("container_name")),
            container_id=_optional(data.get("container_id")),
            started_at=float(str(data.get("started_at") or 0.0)),
        )


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True, kw_only=True)
class Identity:
    """What the right authority says is live *now* for one recorded handle.

    ``started_at`` is the daemon's ``.State.StartedAt`` string for a container and ``None`` for a
    host process, whose age is ``start_time`` (kernel ticks) instead. ``note`` is why a handle is
    not alive, so the row can print it rather than making the human guess.

    ``unknown`` is the distinction ruling A13 was written about: an authority that **could not be
    asked** — a busy daemon, the inspect timeout, permission denied with no working ``sg`` — hands
    back the same shape as one that answered "there is nothing here", and the two mean opposite
    things. ``alive`` is ``False`` either way, because nothing may be treated as live on no
    evidence, but a row that prints "it is not running" when the truth is "nobody could be asked"
    is the same lie one layer down. Both are refusals, and only one of them is a fact.
    """

    kind: str
    alive: bool
    pid: int | None = None
    start_time: int | None = None
    container_id: str | None = None
    started_at: str | None = None
    note: str | None = None
    #: The authority could not be asked, so nothing above is evidence about the handle.
    unknown: bool = False


@dataclass(frozen=True, kw_only=True)
class Observation:
    """The four facts D1 decides from, gathered by the supervisor before anything is spawned."""

    name: str
    url: str
    healthy: bool
    health_reason: str | None = None
    #: What this CLI recorded, or ``None`` when there is no record (or only a T-0033 pid file).
    record: StateRecord | None = None
    #: What that record's handle resolves to now, or ``None`` when it resolves to nothing.
    live: Identity | None = None
    #: The digest of ``services/<name>.sh`` **as it is on disk now**.
    script_digest: str = ""


@dataclass(frozen=True)
class Decision:
    """One service's action and the row's text. ``action`` is one of :data:`ACTIONS`."""

    name: str
    action: str
    reason: str


def handle_text(record: StateRecord, live: Identity | None = None) -> str:
    """How a row names the handle it checked: a container id, or a pid."""
    if record.kind == "container":
        container_id = (live.container_id if live is not None else None) or record.container_id
        return f"container {container_id or 'unknown'} ({record.container_name or 'unnamed'})"
    return f"pid {record.pid}"


def _since(record: StateRecord, live: Identity | None) -> str:
    started = live.started_at if live is not None and live.started_at else None
    return started or f"{record.started_at:.0f} (epoch seconds)"


def decide(observation: Observation) -> Decision:
    """D1's table. Pure, total and deterministic: a function of the observation's *value*.

    The order of the branches is the order of what can be fooled. Health is asked first only
    because an unhealthy service needs no provenance to start; from there on the record, the
    liveness of the handle it names and the script digest each get a refusal of their own, and the
    skip is what is left when all three agree.
    """
    name = observation.name
    if not observation.healthy:
        why = observation.health_reason or "nothing answered"
        return Decision(name, "START", f"{name}: not running ({why}) — starting it")

    record = observation.record
    if record is None:
        return Decision(
            name,
            "CONFLICT",
            f"{name}: healthy at {observation.url}, but no state record names the instance that "
            "is answering — this CLI did not start it, so `astra down` will not stop it (down "
            "stops only what a record names). Stop it by hand, then run `astra up` again.",
        )

    live = observation.live
    if live is not None and live.unknown:
        return Decision(
            name,
            "CONFLICT",
            f"{name}: healthy at {observation.url}, but whether the handle this CLI recorded "
            f"({handle_text(record, live)}) is still the instance answering could not be "
            f"established ({live.note or 'the authority could not be asked'}) — so this CLI "
            "cannot say the running instance is its own, and a skip would be a guess. Fix the "
            f"reason it could not be asked, or reboot it deliberately: `astra up --restart {name}`.",
        )

    if live is None or not live.alive:
        why = (live.note if live is not None and live.note else None) or "it resolves to nothing"
        return Decision(
            name,
            "CONFLICT",
            f"{name}: healthy at {observation.url}, but the handle this CLI recorded "
            f"({handle_text(record, live)}) is not running ({why}) — something else is answering "
            "the port and `astra down` will not stop it. Stop it by hand, then run `astra up` "
            "again.",
        )

    if not record.script_digest or not observation.script_digest:
        return Decision(
            name,
            "CONFLICT",
            f"{name}: healthy at {observation.url} as {handle_text(record, live)}, but the "
            f"service script's digest cannot be compared (recorded {record.script_digest!r}, on "
            f"disk {observation.script_digest!r}), so this CLI cannot say the running instance "
            f"matches the configuration on disk. Reboot it deliberately: `astra up --restart "
            f"{name}`.",
        )

    if record.script_digest != observation.script_digest:
        return Decision(
            name,
            "CONFLICT",
            f"{name}: healthy at {observation.url} as {handle_text(record, live)}, but "
            f"services/{name}.sh has changed since it was started (recorded "
            f"{record.script_digest}, on disk {observation.script_digest}) — skipping it would "
            f"leave the old configuration running under the new one. Reboot it deliberately: "
            f"`astra up --restart {name}`.",
        )

    return Decision(
        name,
        "SKIP",
        f"{name}: already running — {handle_text(record, live)} since {_since(record, live)}; "
        "skipped (this start did nothing to it)",
    )
