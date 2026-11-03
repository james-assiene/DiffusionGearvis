"""The per-session debug log: one JSONL file per session, written off the pump. Task T-0043.

On 2026-09-15 the model said it had illustrated something, nothing appeared on the page, and
every ``turns.jsonl`` line said ``ui_valid: null`` — so the three hypotheses (no ``ui`` segment was
ever committed; one was committed and the gate refused it; one was committed and the session
dropped it) could not be told apart afterwards. This module is what makes them distinguishable:
every wire message in both directions, every machine step, every LLM request and canvas *as
received*, every gate verdict, the utterances, the endpointer's decisions and the turn record, in
one file under ``logs/sessions/``.

The three properties that shape the code, each of which a test enforces:

* **``record()`` is synchronous, total and does no I/O** (criterion 3). It snapshots, redacts and
  offers to a bounded queue with ``put_nowait``; the ``json.dumps``, the write and the flush
  belong to the writer task. It is called from ``Session._handle``, from ``HudConnection.receive``
  and from ``HudServer.publish``, all on the event loop, which may not block for more than 20 ms
  (``.claude/rules/conductor.md``). An overflow is dropped and counted — telemetry never closes a
  socket and never ends a session — and the gap is made visible by a ``dropped`` record written
  before the next record that gets through.
* **Redaction is structural and happens before the queue** (criterion 5). The pump owns the only
  reference to the frame's bytes at the moment of the tap, so a base64 image is replaced by its
  size, digest and the frame's own perceptual hash *there*; deferring it would carry a base64
  megabyte through the queue, which is the bounded-memory failure mode itself. Nothing decides by
  field name: a ``data:`` URL hidden in a window title or three levels deep in an A2UI tree is
  replaced like ``astra.frame.jpeg_b64`` is.
* **Thinking IS logged** (criterion 6). The project rule keeps thought canvases out of history,
  out of ``astra.spoken``, out of TTS and out of the HUD panes; the log is the artefact one reads
  when an answer went wrong, so the thought goes in verbatim and untruncated. A redactor that
  drops long strings would pass the pixel scan and destroy the evidence this file exists for.

Failing is not fatal: an ``OSError`` from the file disables logging for the rest of the session,
is reported in ``stats()["error"]`` and printed once, and the session — the turn under way
included — carries on with ``session_log: null`` on its ``turns.jsonl`` line (criterion 4). It is
also *announced*: ``on_disable`` is called from that path (ruling B14), because a log that turns
itself off and tells nobody leaves the page that stayed connected — the human at the desk — with a
ticked checkbox over a session recording nothing. The callback is synchronous and runs inside the
writer task, which is why publishing must not suspend.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from local_astra.contracts.base import Contract

#: Criterion 2: the closed vocabulary, declared here and nowhere else. A record whose kind is
#: outside it is a programming error — it is reported as an ``error`` record, never written as
#: itself, so a reader may switch on ``kind`` exhaustively.
KINDS: frozenset[str] = frozenset(
    {
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
        # T-0054 criterion 5: the voice guard refused a sentence as interface data. Its own kind
        # and not a ``note``: a count kept on the per-turn ``SpeechGate`` dies with the gate, so
        # this record is the only copy that outlives the turn.
        "speech.refused",
        # T-0092c criterion 7: one record per published ``astra.tool.call``, with the same status.
        "tool.call",
        # T-0114: one record per executed call of the tool-result loop: {call_id, step, status, response}.
        "tool.result",
        # T-0131 B2 (re-scope R3): one record per ReGround decision: {call_id, p1, p2, crop, outcome, ms, tool_text}.
        "tool.reground",
        "turn.record",
        # T-0162 A-3: one record per session reset: {resets, ended_turn_id, state_before, memory_forgotten}.
        "session.reset",
        # T-0213 A2a-5: one record per applied settings set from the panel: {rev, changes: {key: {old, new}}, source,
        # turn_seq, applied_at, system_sha12_old, system_sha12_new}; the user's instructions as sha12 and length only.
        "settings.change",
        "error",
        "note",
    }
)

#: Queue depth (ambiguity 3: chosen, not measured — the first live run decides whether it stays).
#: A turn offers on the order of a few hundred records, so this is several turns of head-room and
#: still a bounded object a stalled writer cannot grow past.
DEFAULT_MAXSIZE = 4096

#: How long ``aclose`` gives the writer to drain before it stops waiting and closes anyway. The
#: session is going down; a log that cannot be drained must not hold it open.
CLOSE_TIMEOUT_S = 2.0


def _monotonic_ms() -> int:
    """The default clock: monotonic milliseconds (the tests inject their own)."""
    return time.monotonic_ns() // 1_000_000


def _wall_now() -> str:
    """ISO-8601 local time *with* its UTC offset — a stamp without one compares to nothing."""
    return datetime.now().astimezone().isoformat()


# --- redaction (criterion 5) --------------------------------------------------------------------

#: The base64 prefixes of the image formats a frame or an A2UI tree can carry. A payload is
#: recognised by what it *is*, not by the name of the field holding it.
_BASE64_MAGIC: tuple[tuple[str, str], ...] = (
    ("/9j/", "image/jpeg"),
    ("iVBORw0KGgo", "image/png"),
    ("R0lGOD", "image/gif"),
    ("UklGR", "image/webp"),
)

#: One pass for both hiding places: a ``data:<mime>[;param][;base64],`` header anywhere in the
#: string, or a base64 image payload that starts somewhere inside it. One compiled alternation
#: rather than five substring searches — a 245 KiB canvas is scanned once (criterion 27).
_EMBEDDED = re.compile(
    r"(?P<data>data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?:;[\w.+-]+(?:=[^;,]*)?)*,)"
    r"|(?P<magic>" + "|".join(re.escape(magic) for magic, _ in _BASE64_MAGIC) + r")"
)

_MIME_OF_MAGIC: dict[str, str] = dict(_BASE64_MAGIC)

#: What the replacement object holds (criterion 5). Written down once so the shape cannot drift.
REDACTION_KEYS: frozenset[str] = frozenset({"redacted", "bytes", "digest", "phash", "mime"})


def _image_in(text: str) -> tuple[str, str] | None:
    """``(mime, payload)`` when this string is, or contains, an image payload; else ``None``.

    Cheap first: the common case — a field that *is* a payload, like ``astra.frame.jpeg_b64`` —
    is four ``startswith`` tests and no scan at all, which is what keeps a 180 KiB frame inside
    criterion 27's budget. Only a string that is not itself a payload is scanned, once, for one
    hidden somewhere inside it. ``bytes`` and ``digest`` are taken over the base64 text **as
    received** (ruling A4): no decode, no second copy.
    """
    for magic, mime in _BASE64_MAGIC:
        if text.startswith(magic):
            return mime, text
    found = _EMBEDDED.search(text)
    if found is None:
        return None
    if found.group("data") is not None:
        return found.group("mime") or "application/octet-stream", text[found.end() :]
    return _MIME_OF_MAGIC[found.group("magic")], text[found.start() :]


def _redacted(mime: str, payload: str, phash: str | None) -> dict[str, Any]:
    return {
        "redacted": "image",
        "bytes": len(payload),
        "digest": hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:8],
        "phash": phash,
        "mime": mime,
    }


def _walk(value: Any, phash: str | None) -> Any:
    if isinstance(value, Mapping):
        own = value.get("phash")
        inherited = own if isinstance(own, str) else None
        return {str(key): _walk(item, inherited) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_walk(item, phash) for item in value]
    if isinstance(value, str):
        found = _image_in(value)
        if found is None:
            return value
        return _redacted(found[0], found[1], phash)
    return value


def redact(obj: Any) -> Any:
    """A copy of ``obj`` with every base64 image and every ``data:`` URL replaced (criterion 5).

    Pure: the argument is not mutated, and the containers in the result are new — which is also
    what makes ``record()``'s snapshot a snapshot. ``phash`` is the frame's own field when the
    replaced value sits beside one (ruling A4), never a recomputed perceptual hash.
    """
    return _walk(obj, None)


def _bad_path(value: Any, prefix: str = "") -> str:
    """Where in a payload the value that cannot be serialised sits, as a dotted path.

    "something failed to serialise" in a debug log is the same as nothing (criterion 3).
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not _serialisable(item):
                return _bad_path(item, f"{prefix}{key}.")
        return prefix.rstrip(".")
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            if not _serialisable(item):
                return _bad_path(item, f"{prefix}{index}.")
        return prefix.rstrip(".")
    return f"{prefix.rstrip('.')} ({type(value).__name__})" if prefix else type(value).__name__


def _serialisable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# --- the record contract (criterion 2) ----------------------------------------------------------


class LogRecord(Contract):
    """One line of a session log. Every line of every log validates against this."""

    #: 0-based, strictly increasing, no gaps — assigned when the record is *written*, which is
    #: what makes the ``dropped`` record the only way a gap is ever reported.
    seq: int = Field(ge=0)
    #: milliseconds since the file opened, from a monotonic source, stamped at offer time (A2)
    t_ms: int = Field(ge=0)
    #: ISO-8601 local time with a UTC offset, stamped at offer time
    wall: str
    #: the session clock's ``now_ms`` when the caller knows it — what joins this record to a
    #: ``turns.jsonl`` span — and ``null`` when it does not
    session_ms: int | None = None
    kind: str
    turn_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _kind_is_known(cls, value: str) -> str:
        if value not in KINDS:
            raise ValueError(f"{value!r} is not one of the session log's kinds: {sorted(KINDS)}")
        return value


# --- the writer ---------------------------------------------------------------------------------

#: Paths this process has already handed out. Two sessions started in the same second of the same
#: process would otherwise share a name, and the second one's first record would truncate the
#: first one's evidence.
_CLAIMED: set[Path] = set()


class SessionLog:
    """``logs/sessions/<YYYYMMDD-HHMMSS>-<pid>.jsonl``, written by one task, offered by anyone.

    ``path_dir`` is the directory the file is created in (``--log-dir``; the default is
    ``supervisor.log_dir() / "sessions"``, chosen by the CLI). ``clock`` is a zero-argument
    callable returning monotonic milliseconds, injected by the tests so ``t_ms`` is deterministic.

    ``on_disable`` is the announcement seam of ruling B14: a zero-argument callable the owner sets
    (``Session`` publishes ``astra.log.state`` through it), called exactly once, from the moment
    the log turns itself off. It runs inside the writer task and must not suspend.
    """

    def __init__(
        self,
        path_dir: Path | str,
        *,
        maxsize: int = DEFAULT_MAXSIZE,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.dir = Path(path_dir).expanduser().absolute()
        self._clock: Callable[[], int] = clock if clock is not None else _monotonic_ms
        self._maxsize = int(maxsize)
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        self.path = self._pick_path()
        self._handle: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._t0 = 0
        self._started = False
        #: set when ``aclose`` begins — no further record is accepted — and distinct from
        #: ``_closed``, which is set only once ``log.stopped`` is on disk and the handle is shut.
        #: Ruling B16: marking the log closed *before* the drain made a cancelled ``aclose``
        #: unrepairable, so the closing record was lost and the handle leaked.
        self._closing = False
        self._closed = False
        #: whether the bounded drain has already been attempted; a resumed close does not wait
        #: again on a queue whose writer is already cancelled.
        self._drain_started = False
        #: ruling B14: called once, from ``_disable``. Synchronous by construction.
        self.on_disable: Callable[[], None] | None = None
        self._seq = 0
        self._last_seq = 0
        self._records = 0
        self._bytes = 0
        self._dropped_total = 0
        self._dropped_pending = 0
        self._error: str | None = None

    def _pick_path(self) -> Path:
        base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        nth = 1
        while True:
            name = f"{base}.jsonl" if nth == 1 else f"{base}-{nth}.jsonl"
            candidate = self.dir / name
            if candidate not in _CLAIMED and not candidate.exists():
                _CLAIMED.add(candidate)
                return candidate
            nth += 1

    # --- lifecycle ------------------------------------------------------------------------------

    async def start(self) -> None:
        """Open the file, start the writer task, write ``log.started``. Idempotent."""
        if self._started or self._closed:
            return
        self._started = True
        self._t0 = int(self._clock())
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        except OSError as exc:
            self._disable(f"cannot open {self.path}: {exc}")
            return
        self._task = asyncio.create_task(self._run(), name="session-log")
        self.record("log.started", {"pid": os.getpid(), "path": str(self.path)})

    async def aclose(self, reason: str = "session closed") -> None:
        """Drain, write ``log.stopped``, close. Idempotent, bounded (criterion 4), **resumable**.

        Resumable is ruling B16: the drain is a suspension point, so this coroutine can be
        cancelled halfway through it — a toggle-off in flight when the session closes is. What a
        cancellation must not do is leave the log believing it is closed: the closing record would
        never be written and the handle would never be shut, and no later ``aclose`` could repair
        either. So ``_closed`` is set at the *end*, the writer task is cancelled from a ``finally``
        (synchronous, and therefore safe while this coroutine is being cancelled), and a second
        call finishes what the first started without waiting on the drain a second time.
        """
        if self._closed:
            return
        self._closing = True
        await self._stop_writer()
        if self._handle is not None:
            stopped: dict[str, Any] = {
                "t_ms": self._now_ms(),
                "wall": _wall_now(),
                "session_ms": None,
                "kind": "log.stopped",
                "turn_id": None,
                "data": {
                    "reason": reason,
                    "records": self._records,
                    "dropped": self._dropped_total,
                    "bytes": self._bytes,
                },
            }
            if self._dropped_pending:
                stopped["_drops"], self._dropped_pending = self._dropped_pending, 0
            self._write(stopped)
            handle, self._handle = self._handle, None
            with contextlib.suppress(OSError):
                handle.close()
        self._closed = True

    async def _stop_writer(self) -> None:
        """End the writer task, drained if it can be, at most once per log."""
        task = self._task
        if task is None:
            return
        try:
            if not self._drain_started:
                self._drain_started = True
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._queue.join(), CLOSE_TIMEOUT_S)
        finally:
            # Synchronous: this still runs, and still ends the writer, when the ``await`` above is
            # what was cancelled. ``self._task`` is cleared only once the task has been awaited,
            # so a resumed close knows there is still a task to reap.
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None

    # --- the offer (criterion 3) ----------------------------------------------------------------

    def record(
        self,
        kind: str,
        data: Any,
        *,
        turn_id: str | None = None,
        session_ms: int | None = None,
    ) -> None:
        """Offer one record. Synchronous, total, no I/O, no ``json.dumps`` (criterion 3).

        Never raises for any input: a payload that cannot be serialised becomes an ``error``
        record in the writer, and anything that goes wrong here becomes one too. The pump's turn
        is never the thing that fails because telemetry did.
        """
        if self._handle is None or self._closing or self._closed or self._error is not None:
            return
        try:
            if self._queue.full():
                # The HUD outbox's discipline (T-0028), with one difference: an overflow here
                # closes nothing. The gap is reported by the ``dropped`` record written in front
                # of the next record that *is* accepted, so it sits where the gap sits.
                self._drop()
                return
            if kind in KINDS:
                payload: Any = redact(data)
            else:
                payload = {
                    "what": f"{kind!r} is not one of the session log's kinds",
                    "kind": str(kind),
                }
                kind = "error"
            self._offer(
                {
                    "t_ms": self._now_ms(),
                    "wall": _wall_now(),
                    "session_ms": None if session_ms is None else int(session_ms),
                    "kind": kind,
                    "turn_id": None if turn_id is None else str(turn_id),
                    "data": payload,
                }
            )
        except asyncio.QueueFull:  # pragma: no cover - ``full()`` above decided, single-threaded
            self._drop()
        except Exception as exc:  # criterion 3: record() is total, whatever the caller handed it
            self._offer_error(kind, exc)

    def _drop(self) -> None:
        self._dropped_total += 1
        self._dropped_pending += 1

    def _offer(self, item: dict[str, Any]) -> None:
        """Queue one stamped record, carrying the drops it is the first survivor of."""
        if self._dropped_pending:
            item["_drops"], self._dropped_pending = self._dropped_pending, 0
        self._queue.put_nowait(item)

    def _offer_error(self, kind: str, exc: BaseException) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(
                {
                    "t_ms": self._now_ms(),
                    "wall": _wall_now(),
                    "session_ms": None,
                    "kind": "error",
                    "turn_id": None,
                    "data": {
                        "what": f"the {kind!r} record could not be offered",
                        "kind": kind,
                        "error": repr(exc),
                    },
                }
            )

    def qsize(self) -> int:
        return self._queue.qsize()

    def stats(self) -> Mapping[str, Any]:
        """What ``astra.log.state`` carries (group B) and what ``log.stopped`` reports."""
        return {
            "records": self._records,
            "dropped": self._dropped_total,
            "bytes": self._bytes,
            "error": self._error,
            "path": str(self.path),
        }

    # --- the writer task ------------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                self._write(item)
            finally:
                self._queue.task_done()

    def _now_ms(self) -> int:
        return max(0, int(self._clock()) - self._t0)

    def _write(self, item: dict[str, Any]) -> None:
        """One record on disk, flushed, preceded by the gap marker the drops owe (criterion 3)."""
        if self._handle is None:
            return
        count = int(item.pop("_drops", 0))
        if count:
            self._emit(
                {
                    **item,
                    "kind": "dropped",
                    "turn_id": None,
                    "session_ms": None,
                    # ruling A3: the seq of the last record written *before* the gap
                    "data": {"count": count, "since_seq": self._last_seq},
                }
            )
        self._emit(item)

    def _emit(self, item: dict[str, Any]) -> None:
        handle = self._handle
        if handle is None:
            return
        record = {"seq": self._seq, **item}
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            record = {
                **record,
                "kind": "error",
                "data": {
                    "what": f"the {item['kind']!r} record could not be serialised",
                    "kind": item["kind"],
                    "path": _bad_path(item["data"]),
                    "error": repr(exc),
                },
            }
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        try:
            n_bytes = len(line.encode("utf-8")) + 1
        except (
            UnicodeEncodeError
        ):  # T-0216b F2: a lone surrogate (a page's "\ud800" escape) is written escaped
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
            n_bytes = len(line) + 1
        try:
            handle.write(line + "\n")
            handle.flush()  # criterion 1: readable line by line while the session is running
        except OSError as exc:
            self._disable(f"cannot write {self.path}: {exc}")
            return
        self._last_seq = self._seq
        self._seq += 1
        self._records += 1
        self._bytes += n_bytes

    def _disable(self, reason: str) -> None:
        """Criterion 4: the log goes, the session stays. Said once, never raised.

        Said *out loud* too (ruling B14): ``on_disable`` fires after the state is final, so the
        owner reads a log that is already off, already pathless and already carrying its reason.
        The page that stayed connected learns what happened at the moment it happens rather than
        on its next reload — a subsystem that disables itself and tells nobody is worse than the
        page-side failure the checkbox was designed against (``.claude/rules/conductor.md``).
        """
        if self._error is not None:
            return
        self._error = reason
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        while not self._queue.empty():  # nothing may hold aclose()'s drain open
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.task_done()
        print(f"astra: session logging is off — {reason}", file=sys.stderr)
        announce, self.on_disable = self.on_disable, None
        if announce is not None:
            try:
                announce()
            except Exception as exc:  # the disable path runs *inside* the writer task: a raising
                # announcement would kill the writer and lose the reason as well as the log.
                print(f"astra: could not publish the log state — {exc!r}", file=sys.stderr)
