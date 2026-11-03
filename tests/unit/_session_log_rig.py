"""Shared helpers for T-0043 group A: the session log's seams, fixtures and readers.

Not a test module (the leading underscore keeps pytest from collecting it).

==================================================================================================
THE SEAMS THIS TASK ADDS. The implementer owns the design *behind* them; these are the names the
tests call. Anything wrong here goes to the reviewer; the implementer never edits ``tests/``.
==================================================================================================

``local_astra/telemetry/session_log.py`` — new (criteria 1-5)

    KINDS: frozenset[str]
        The closed vocabulary of criterion 2, declared in **one** place. A record whose kind is
        outside it is never written.

    redact(obj) -> obj
        Pure (criterion 5): walks a dumped record tree and replaces every base64 JPEG payload and
        every ``data:`` URL — wherever it occurs, whatever the field is called — with
        ``{"redacted": "image", "bytes": n, "digest": "<8 hex of sha256>", "phash": hex|None,
        "mime": "…"}``. Returns a new tree; the argument is not mutated.

    LogRecord
        The pydantic record contract of criterion 2: ``seq``, ``t_ms``, ``wall``, ``session_ms``,
        ``kind``, ``turn_id``, ``data``. Every line of a log validates against it.

    SessionLog(path_dir, *, maxsize=4096, clock=None)
        ``path_dir`` is the directory the file is created in (``--log-dir``, default
        ``supervisor.log_dir() / "sessions"``); ``clock`` is a zero-argument callable returning
        **monotonic milliseconds**, injected by the tests so ``t_ms`` is deterministic.

        .path -> Path                     the file this log writes
        await start() -> None             open the file, start the writer task, write log.started
        await aclose() -> None            drain, write log.stopped, close. Idempotent.
        record(kind, data, *, turn_id=None, session_ms=None) -> None
                                          synchronous, total, never raises, no I/O (criterion 3)
        qsize() -> int                    what the writer has not written yet
        stats() -> Mapping[str, Any]      at least ``records``, ``dropped``, ``bytes``, ``error``
                                          (``error`` is ``None`` while the log is healthy and the
                                          reason string once an ``OSError`` disabled it —
                                          criterion 4, and the fields ``astra.log.state`` carries)

``local_astra/conductor/session.py`` — narrow mandate (log taps only)

    Session(..., *, clock, config, publish=None, session_log: SessionLog | None = None)
        ``None`` (the default) is "not logging": no file, no writer task, and ``session_log`` is
        ``None`` on every ``turns.jsonl`` line the session writes. A log that is passed is
        ``start()``ed from ``Session.start()`` and drained-then-closed from ``Session.aclose()``
        (criterion 4).
    session.session_log -> SessionLog | None    what the HUD transport records through

``local_astra/telemetry/turns_log.py``

    TurnRecord.session_log: str | None = None   (criterion 14; a top-level field, not an extra)
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import errno
import hashlib
import inspect
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: Criterion 2, transcribed from ``docs/tasks/T-0043.md`` — the closed vocabulary. The test
#: asserts the module's own ``KINDS`` equals this set; this tuple is the task file, not the code.
#:
#: **Amended by T-0054 criterion 5** ("``local_astra/telemetry/session_log.py`` ``KINDS`` gains
#: ``speech.refused``; it is the only new kind"): the vocabulary is T-0043's sixteen plus that
#: one. The three pins that read this tuple — ``test_session_log_writer.py``'s
#: ``test_the_kind_vocabulary_is_closed_and_declared_once``, ``test_session_log_format.py``'s
#: two vocabulary rows and this constant itself — are named in T-0054's *Owned files* section as
#: pins the task legitimately moves (``.claude/rules/tests.md``, "Activation is three things in
#: one commit"). Nothing else in those modules moves.
CRITERION_2_KINDS = (
    "log.started",
    "log.stopped",
    "dropped",
    "wire.in",
    "wire.out",
    "machine.step",
    "llm.request",
    "llm.commit",
    "llm.done",
    "ui.gate",
    "tts.utterance",
    "tts.played",
    "stt.decision",
    "turn.record",
    "error",
    "note",
    # T-0054 criterion 5: a refusal by the voice guard is an event, and until this kind exists
    # the only copy of it dies with the per-turn ``SpeechGate``.
    "speech.refused",
    # T-0092c criterion 7 (moving pin, named in docs/tasks/T-0092c.md): one record per published astra.tool.call.
    "tool.call",
    # T-0114 re-scope (docs/tasks/T-0114.md, "Re-scope after implementation"): the tool-result loop's record.
    "tool.result",
    # T-0131 R3 (docs/tasks/T-0131.md, Rulings; lent pin): the ReGround pass-2 record. The log refuses unknown
    # kinds, so this moves together with ``session_log.KINDS``.
    "tool.reground",
)

#: T-0054 criterion 5's record kind, named once so a row cannot typo it into a ``note``.
SPEECH_REFUSED = "speech.refused"

#: T-0054 criterion 8's new server->client message type. A **type**, never a field on
#: ``astra.ui.surface``: an unmirrored field makes every message of that type malformed on the
#: page and blanks the shown pane, while an unmirrored type decodes as ``unknown`` and is ignored
#: and counted (T-0054 D1, measured on ``hud/src/protocol.js``).
UI_FENCE_TYPE = "astra.ui.fence"

#: Criterion 5: the keys of the replacement object.
REDACTED_KEYS = {"redacted", "bytes", "digest", "phash", "mime"}

_MISSING: str | None = None
try:  # a module that does not exist yet must not be a collection error (tests/conftest.py)
    from local_astra.telemetry import session_log as _module
except ImportError as exc:  # pragma: no cover - the pre-implementation path
    _module = None  # type: ignore[assignment]
    _MISSING = f"local_astra.telemetry.session_log: {exc}"


def module() -> Any:
    """The module under test, or ``NotImplementedError`` naming what T-0043 has not written."""
    if _MISSING is not None:
        raise NotImplementedError(
            f"T-0043 group A adds {_MISSING} (docs/tasks/T-0043.md, Owned files); "
            "the seams are pinned in tests/unit/_session_log_rig.py's docstring"
        )
    return _module


def symbol(name: str) -> Any:
    """One public name of ``telemetry/session_log.py`` (see this module's docstring)."""
    value = getattr(module(), name, None)
    if value is None:
        raise NotImplementedError(
            f"T-0043: local_astra/telemetry/session_log.py must define {name!r} "
            "(tests/unit/_session_log_rig.py's docstring fixes the seam)"
        )
    return value


def make_log(path_dir: Path | str, **kwargs: Any) -> Any:
    """``SessionLog(path_dir, **kwargs)``, with the constructor's seam checked first."""
    cls = symbol("SessionLog")
    parameters = inspect.signature(cls.__init__).parameters
    for name in ("maxsize", "clock"):
        if name not in parameters:
            raise NotImplementedError(
                f"T-0043: SessionLog(path_dir, *, maxsize=4096, clock=...) — {name!r} is missing; "
                f"the constructor takes {sorted(parameters)}"
            )
    return cls(path_dir, **kwargs)


class StepClock:
    """A monotonic-milliseconds source the test moves by hand (``clock=`` above)."""

    def __init__(self, start_ms: int = 0) -> None:
        self.now_ms = int(start_ms)

    def advance(self, ms: int) -> int:
        self.now_ms += int(ms)
        return self.now_ms

    def __call__(self) -> int:
        return self.now_ms


async def settle(log: Any, rounds: int = 200) -> None:
    """Let the writer task run until the queue is empty (no wall-clock assumption)."""
    for _ in range(rounds):
        if int(log.qsize()) == 0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
    raise AssertionError(f"the writer never drained: qsize={log.qsize()} after {rounds} rounds")


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """Every line of a log, parsed. Strict: a line that is not one JSON object is a failure."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines()):
        assert line.strip(), f"line {n} of {path} is blank; the log is strict JSONL"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"line {n} of {path} is not JSON: {exc}; line={line[:200]!r}"
            ) from None
        assert isinstance(record, dict), f"line {n} is {type(record).__name__}, not an object"
        out.append(record)
    return out


def kinds(records: Iterable[dict[str, Any]]) -> list[str]:
    return [str(r.get("kind")) for r in records]


def of_kind(records: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == kind]


# --- image fixtures (criterion 5) ---------------------------------------------------------------


def jpeg(size_bytes: int = 200 * 1024, *, tag: bytes = b"a") -> bytes:
    """A JPEG-shaped payload of the requested size: SOI + APP0 + filler + EOI.

    Not a decodable image — nothing here decodes one — but it starts with the magic ``\\xff\\xd8``
    that makes its base64 begin with ``/9j/``, which is the substring criterion 5's scan looks
    for. A payload without it would make the scan vacuous.
    """
    head = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    tail = b"\xff\xd9"
    filler = (tag * 64)[:64] * max(1, (size_bytes - len(head) - len(tail)) // 64)
    return head + filler[: max(0, size_bytes - len(head) - len(tail))] + tail


def b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


def data_url(payload: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{b64(payload)}"


def digest8(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:8]


def check_redacted(value: Any, payload: bytes, *, what: str, phash: str | None = None) -> None:
    """Assert ``value`` is criterion 5's replacement object *for this image*.

    The oracle is independent of whether the implementation counts and digests the **decoded**
    bytes or the **base64 text** it found in the tree: both are legitimate readings of
    "bytes: n, digest: 8 hex of sha256", but a record whose ``digest`` is not the digest of the
    thing its ``bytes`` counts is incoherent, and so is one that digests neither. Two different
    images therefore still get two different digests (asserted where this is called).
    """
    assert isinstance(value, dict), f"{what}: expected the redaction object, got {value!r:.200}"
    assert set(value) == REDACTED_KEYS, (
        f"{what}: criterion 5 fixes the replacement's keys to {sorted(REDACTED_KEYS)}; "
        f"got {sorted(value)}"
    )
    assert value["redacted"] == "image", f"{what}: redacted={value['redacted']!r}"
    text = b64(payload)
    candidates = {len(payload): digest8(payload), len(text): digest8(text.encode())}
    assert value["bytes"] in candidates, (
        f"{what}: bytes={value['bytes']} is neither the payload's length ({len(payload)}) nor "
        f"its base64 length ({len(text)}) — a size nobody can check is not evidence"
    )
    assert value["digest"] == candidates[value["bytes"]], (
        f"{what}: digest={value['digest']!r} is not sha256[:8] of the {value['bytes']} bytes it "
        f"claims to have counted (expected {candidates[value['bytes']]!r})"
    )
    assert isinstance(value["mime"], str) and value["mime"].startswith("image/"), (
        f"{what}: mime={value['mime']!r} must name the image type it replaced"
    )
    if phash is not None:
        assert value["phash"] == phash, (
            f"{what}: the frame's own perceptual hash is carried through, not recomputed; "
            f"expected {phash!r}, got {value['phash']!r}"
        )


FORBIDDEN = ("/9j/", "data:image")


def scan_for_pixels(path: Path | str) -> list[str]:
    """Criterion 5's negative scan: what a log file may never contain."""
    raw = Path(path).read_bytes()
    return [needle for needle in FORBIDDEN if needle.encode() in raw]


def container_sizes(obj: Any) -> dict[str, int]:
    """Every sized container reachable from ``obj``'s attributes, one level deep.

    The boundedness criterion is not only about the queue: a second list that grows per record
    (a ``self._history``, a per-record ``self._seen``) is the same leak with another name.
    """
    out: dict[str, int] = {}
    for name, value in list(vars(obj).items()):
        if isinstance(value, str | bytes):
            continue
        try:
            out[name] = len(value)
        except TypeError:
            queue = getattr(value, "qsize", None)
            if callable(queue):
                out[name] = int(queue())
    return out


def pid() -> int:
    return os.getpid()


# --- injecting a *write* failure (criterion 4's second arm, ruling B8) ---------------------------


class _FailingWrites:
    """A real file handle whose ``write`` refuses once the switch is armed.

    Everything else is delegated to the handle the real ``open`` returned (the dunders explicitly,
    because Python looks those up on the type): this double fakes exactly one verb and closes by
    delegation, so an implementation that reaches for ``seek``, ``fileno`` or ``tell`` gets the
    real file rather than an ``AttributeError`` inside the test.
    """

    def __init__(self, handle: Any, switch: WriteSwitch) -> None:
        self._handle = handle
        self._switch = switch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __enter__(self) -> _FailingWrites:
        self._handle.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._handle.__exit__(*exc)

    def __iter__(self) -> Any:
        return iter(self._handle)

    def _refuse(self) -> None:
        self._switch.refused += 1
        raise OSError(self._switch.errno, self._switch.message)

    def write(self, text: Any) -> int:
        if self._switch.armed:
            self._refuse()
        return int(self._handle.write(text))

    def writelines(self, lines: Any) -> None:
        # Refused too: a write that escaped the injection would leave the log healthy and the
        # premise of every row using this silently false.
        if self._switch.armed:
            self._refuse()
        self._handle.writelines(lines)


class WriteSwitch:
    """When the disk fills, and the evidence that it did. See :func:`failing_writes`."""

    def __init__(self, *, errno_: int, message: str) -> None:
        self.armed = False
        self.errno = errno_
        self.message = message
        #: handles wrapped (the injection reached the writer's ``open`` at all)
        self.wrapped = 0
        #: writes refused (the injection actually bit)
        self.refused = 0

    def arm(self) -> None:
        self.armed = True


def failing_writes(
    monkeypatch: Any,
    directory: Path | str,
    *,
    errno_: int = errno.ENOSPC,
    message: str = "No space left on device",
) -> WriteSwitch:
    """Fill the disk under *directory*, on the test's word: opening works, writing does not.

    Criterion 4 has two arms and they are different branches of the implementation: a log that
    cannot be **opened** never sets a path and never writes a line, while a log that dies on a
    **write** was healthy, had a path, and has already told everyone it was recording. Only the
    second one can leave a page showing a ticked box over a session recording nothing (rulings B8
    and B14), so it has to be reachable from a test.

    Both ``Path.open`` and ``builtins.open`` are patched and both delegate to the real thing; only
    a ``.jsonl`` opened for writing under *directory* is wrapped, so the rest of the process — the
    ``turns.jsonl`` line the session must still write, pytest's own files — is untouched.
    """
    switch = WriteSwitch(errno_=errno_, message=message)
    target = Path(directory).expanduser().absolute()
    real_path_open = Path.open
    real_open = builtins.open

    def wrap(handle: Any, name: Any, mode: str) -> Any:
        if not any(character in mode for character in "aw+x"):
            return handle
        try:
            path = Path(name).expanduser().absolute()
        except TypeError:  # an int fd, a buffer: never the log's file
            return handle
        if path.suffix != ".jsonl" or not path.is_relative_to(target):
            return handle
        switch.wrapped += 1
        return _FailingWrites(handle, switch)

    def path_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return wrap(real_path_open(self, mode, *args, **kwargs), self, mode)

    def builtin_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return wrap(real_open(file, mode, *args, **kwargs), file, mode)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", builtin_open)
    return switch


# --- T-0054: the fence notice on the wire -------------------------------------------------------


def ui_fence_model() -> Any:
    """``contracts.protocol.AstraUiFence``, or an assertion naming what T-0054 group A owes.

    Deliberately an ``AssertionError`` rather than an import at module scope: these rows are
    written red (not ``pending``), and a missing name must fail one row with a sentence a reader
    can act on rather than erroring the whole module at collection.
    """
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraUiFence", None)
    assert model is not None, (
        "T-0054 criterion 8: local_astra/contracts/protocol.py gains "
        "AstraUiFence(type='astra.ui.fence', turn_id: str, canvases: int >= 0, "
        "close: FenceClose | None = None, continued: bool = False, speech_refused: int >= 0 = 0), "
        "in SERVER_TO_CLIENT and ASTRA_MESSAGE_TYPES and never in CLIENT_TO_SERVER "
        "(docs/tasks/T-0054.md, Owned files: local_astra/contracts/protocol.py)"
    )
    return model


def fence_messages(published: Iterable[Any]) -> list[Any]:
    """Every ``astra.ui.fence`` a recorder saw, in order. Matched on the wire ``type``.

    By type string and not by ``isinstance``, so a row can say "zero fence messages were
    published" before the model exists without importing it (criterion 10's negative).
    """
    return [m for m in published if getattr(m, "type", None) == UI_FENCE_TYPE]
