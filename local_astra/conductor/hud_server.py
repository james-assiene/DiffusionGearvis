"""The HUD endpoint: one WebSocket, one session, no opinions (docs/02 §2.2, §9). Task T-0028.

This is the only listening socket in the process and the only place where a real ``Session``
meets a real page. It is deliberately dumb:

* **The session knows nothing about sockets.** Egress is the session's typed publisher
  (``Session.set_publisher``, docs/02 §4 spec delta (b)); this module subscribes to it. Inverting
  that would make the session untestable without a socket.
* **Nothing here blocks the pump.** ``publish`` is a non-blocking hand-off to a bounded
  per-connection ``Outbox`` plus a writer task. A page that stopped reading is *that*
  connection's problem: ``astra.turn.state`` is latest-wins and may be dropped, everything else
  overflowing closes that one connection with a logged reason rather than silently losing a
  spoken line (``.claude/rules/conductor.md``: 20 ms in the pump).
* **Decoding is total and the direction is enforced**, through the real ``contracts/codec`` —
  there is no second parser and no second size rule (T-0027 owns ``MAX_MESSAGE_BYTES``).
* **It validates nothing of its own.** Frames go to the injected sink (``BrowserFrameSource``,
  which already validates base64, JPEG magic, the ring and the rate) and assists go to
  ``AssistState``, whose answer is echoed verbatim.
* **Loopback only.** The bind address comes from ``config.Services``, whose validator refuses
  anything else (docs/02 §9, docs/03 E8), and a non-loopback ``Origin`` is refused with 403 at
  the handshake — a page on a remote site, running in the user's own browser, can otherwise open
  the endpoint even though it is bound to the loopback interface.

``HudConnection`` has no socket behind it: the socket handler is one of these plus a reader and a
writer task, so ingress and egress are unit-testable without a network.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import socket
from collections import deque
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Request, Response

from local_astra.conductor.live_settings import KNOWN_KEYS as SETTINGS_KNOWN_KEYS
from local_astra.config import Services
from local_astra.contracts import events as E
from local_astra.contracts.codec import MAX_MESSAGE_BYTES, Malformed, Unknown, decode, encode
from local_astra.contracts.protocol import (
    CLIENT_TO_SERVER,
    PROTOCOL_VERSION,
    SERVER_TO_CLIENT,
    AstraAssistToggle,
    AstraCanvasStep,
    AstraError,
    AstraFrame,
    AstraFrameText,
    AstraHello,
    AstraLogToggle,
    AstraMessage,
    AstraMicMute,
    AstraSessionCleared,
    AstraSessionReset,
    AstraSettingsSet,
    AstraSettingsState,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraText,
    AstraThought,
    AstraToolCall,
    AstraTranscript,
    AstraTurnState,
    AstraTurnStop,
    AstraUiAction,
    AstraUiSurface,
    AstraVisionFrame,
    AstraVisionSelect,
    RealtimeEvent,
)
from local_astra.contracts.turn import TurnState
from local_astra.settings_catalog import by_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_astra.conductor.session import Session

log = logging.getLogger(__name__)

#: The path ``hud/src/ws.js`` defaults to (T-0008).
DEFAULT_PATH = "/v1/realtime"

#: The scheme, spelled apart from the authority on purpose: this module carries no URL literal.
#: ``url`` is assembled from the loopback-fenced host, the port ``config.Services`` resolved and
#: the path above, so there is no second source of truth for the endpoint to drift from
#: (criterion 14; ``tests/unit/test_hud_server_no_egress.py`` scans for exactly that).
WS_SCHEME = "ws"

#: The only hosts the endpoint may bind. ``config.Services`` carries the port; the host is not a
#: field there, so it is fenced here with the same rule (docs/02 §9, docs/03 E8).
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

#: Per-connection outbox bound. A turn puts on the order of a few dozen messages on the wire
#: (one state change per row, one transcript per word, one spoken line per sentence, one surface
#: per canvas), so 512 is generous enough that an ordinary turn never drops anything and small
#: enough that a page that stopped reading cannot cost this process unbounded memory.
DEFAULT_OUTBOX = 512

#: T-0128c A6: at most this many ``astra.canvas.*`` messages queued per connection. Canvas frames
#: are a view of the latest step, droppable by construction; eight is a third of a second of one
#: request at the tap's 30 Hz cap, so a page that keeps up loses none.
CANVAS_OUTBOX = 8
CANVAS_PREFIX = "astra.canvas."
#: T-0162 A-6: canvas request ids remembered for the reset filter (the relay's own tables hold 64 per kind)
CANVAS_IDS_KEPT = 512
#: T-0177 B7: what a published vision frame's ``jpeg_b64`` reads as in its ``wire.out`` record
VISION_REDACTED = "<redacted>"


def _is_canvas(message: AstraMessage) -> bool:
    return message.type.startswith(CANVAS_PREFIX)


def _canvas_row(message: AstraCanvasStep) -> dict[str, Any]:
    """T-0128c A7: the session-log ``wire.out`` of a canvas step, without its rows (H-T128c-3).

    ``n_stable`` counts the set bits of ``stable`` over the canvas (MSB first); ``n_kept`` the
    positions where ``canvas == argmax`` that are not stable (ruling O-A: B2's partition).
    """
    n = message.canvas_length
    stable = [False] * n
    if message.stable:
        bits = int(message.stable, 16)
        width = len(message.stable) * 4
        stable = [bool((bits >> (width - 1 - p)) & 1) for p in range(n)]
    kept = 0
    if message.canvas is not None and message.argmax is not None:
        kept = sum(
            1
            for p, (c, a) in enumerate(zip(message.canvas, message.argmax, strict=True))
            if c == a and not stable[p]
        )
    return {
        "type": message.type,
        "request_id": message.request_id,
        "canvas_idx": message.canvas_idx,
        "step": message.step,
        "max_steps": message.max_steps,
        "phase": message.phase,
        "canvas_length": n,
        "n_committed": len(message.committed_tokens),
        "n_kept": kept,
        "n_stable": sum(stable),
    }


def _vision_row(message: AstraVisionFrame) -> dict[str, Any]:
    """T-0177 B7: the session-log ``wire.out`` of a vision frame: every field, each image's bytes redacted (the
    request's own ``llm.request`` record already redacts them; the pixels never reach the log file)."""
    row: dict[str, Any] = message.model_dump(mode="json")
    for image in row["images"]:
        if image.get("jpeg_b64") is not None:
            image["jpeg_b64"] = VISION_REDACTED
    return row


def _thought_row(message: AstraThought) -> dict[str, Any]:
    """T-0180 A5: the session-log ``wire.out`` of a thought, every field but ``text`` plus ``n_chars`` (its length
    in code points). The text itself is already in the log as the attempt's ``llm.commit`` rows."""
    row: dict[str, Any] = message.model_dump(mode="json", exclude={"text"})
    row["n_chars"] = len(message.text)
    return row


def _text_digest(value: Any) -> Any:
    """T-0213 R-14: a text as the session log records it: its sha12 and length in code points, never the text.
    T-0213 R-19 (b): a non-scalar value (an object, a list) is digested too, never recorded as it is."""
    if isinstance(value, str):
        return {"sha12": _sha12(value), "len": len(value)}
    if value is None or isinstance(value, bool | int | float):
        return value
    return _value_digest(value)


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _value_digest(value: Any) -> dict[str, Any]:
    """T-0213 R-19 (b): any value as the session log records it when the page's text could be in it: its type, its
    length and, where it has a JSON form, the sha12 of that form. Never the value. A huge integer has no decimal
    form under the interpreter's digit limit (``.claude/rules/conductor.md``, T-0185), so it gets its bit length."""
    if isinstance(value, str):
        return {"type": "str", "sha12": _sha12(value), "len": len(value)}
    if isinstance(value, bool) or value is None:
        return {"type": type(value).__name__}
    if isinstance(value, int):
        return {"type": "int", "bits": value.bit_length()}
    out: dict[str, Any] = {"type": "object" if isinstance(value, dict) else type(value).__name__}
    if isinstance(value, dict | list | tuple):
        out["len"] = len(value)
    try:
        form = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):  # a huge int nested in it: no sha, and still no value
        return out
    out["sha12"] = _sha12(form)
    return out


def _logged_set_value(key: str, value: Any) -> Any:
    """T-0213 R-19 (b): one value of a set's ``changes`` as its ``wire.in`` records it. Recorded as it is only when
    it can carry none of the page's text: ``null``, a bool, a small int, or one of the catalog row's own values (of
    the same type); the user's instructions, and anything else (another string, a float, an object, a list, a huge
    int), are digested."""
    if key == "user_instructions":
        return _text_digest(value) if isinstance(value, str) else _value_digest(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() <= 31:
        return value
    entry = by_key(key)
    if entry is not None and any(v == value and type(v) is type(value) for v in entry.values):
        return value
    return _value_digest(value)


def _settings_key_digest(key: str) -> str:
    """T-0213 R-19 (b): a key of a set's ``changes`` as its ``wire.in`` records it: a known key as it is, any other
    as its sha12 and length (the page's own text, never repeated into the log)."""
    if key in SETTINGS_KNOWN_KEYS:
        return key
    return f"<unknown key sha12={_sha12(key)} len={len(key)}>"


def _settings_state_row(message: AstraSettingsState) -> dict[str, Any]:
    """T-0213 R-14: the session-log ``wire.out`` of a settings state, the user's instructions as sha12 and length
    wherever they appear (``instructions``, the system message preview that ends with them, and the
    ``user_instructions`` item's value and pending value). The page receives the message itself, unredacted."""
    row: dict[str, Any] = message.model_dump(mode="json")
    row["instructions"] = _text_digest(row["instructions"])
    if row.get("system_preview") is not None:
        row["system_preview"] = _text_digest(row["system_preview"])
    for item in row["items"]:
        if item.get("key") == "user_instructions":
            item["value"] = _text_digest(item.get("value"))
            item["pending"] = _text_digest(item.get("pending"))
    for refusal in row.get("refused") or ():
        # T-0213 R-19 (b/d): the session already names only known keys; a key that is not one is never recorded
        if refusal.get("key") != "*" and refusal.get("key") not in SETTINGS_KNOWN_KEYS:
            refusal["key"] = "*"
    return row


def _settings_set_row(message: AstraSettingsSet) -> dict[str, Any]:
    """T-0213 R-14 / R-19 (b): the session-log ``wire.in`` of a settings set. ``user_instructions`` is its sha12 and
    length; every key no catalog row names, and every value that could carry the page's text (any value but
    ``null``, a bool, a small int or one of the catalog row's own values), is digested (``_logged_set_value``)."""
    rev = message.rev
    return {
        "type": message.type,
        "rev": rev if rev.bit_length() <= 63 else _value_digest(rev),
        "changes": {
            _settings_key_digest(key): _logged_set_value(key, value)
            for key, value in message.changes.items()
        },
        "stop_turn": message.stop_turn,
    }


def _malformed_row(decoded: Malformed) -> str:
    """T-0213 R-16 / R-19 (c): a ``Malformed`` frame's reason as its ``wire.in`` records it. The codec's reason for
    an astra message is ``"{loc}: {msg}"``, where the loc holds the page's own field names (an unknown field is
    named by the page); for ``astra.settings.set`` only the text after the last ``": "`` is kept, which is a suffix
    of pydantic's ``msg`` and so never part of the loc."""
    if decoded.type != "astra.settings.set":
        return decoded.error
    return f"astra.settings.set is malformed: {decoded.error.rsplit(': ', 1)[-1]}"


def _dumped_row(message: RealtimeEvent | AstraMessage) -> Any:
    """T-0216b F1: a decoded frame's ``wire.in`` body. pydantic-core's serializer refuses a value nested past its own
    depth limit (``ValueError: Circular reference detected``) well inside the parser's, so such a frame is recorded
    by its type alone, with no payload text, and handled like any other: the tap never drops the connection."""
    try:
        return message.model_dump(mode="json")
    except ValueError:
        return {"type": message.type, "unserialisable": True}


#: ``websockets``' own frame limit, chosen **above** ``codec.MAX_MESSAGE_BYTES`` on purpose: an
#: oversize message must reach ``decode`` so the page is told ``astra.error(code="malformed")``
#: (T-0008 criterion 2 expects the error, not a bare close). Below the codec limit the library
#: would kill the connection first and the page would see a close code with no reason. The slack
#: is one extra 64 KiB, which is enough for the framing of a message that is itself at the limit.
MAX_SIZE = MAX_MESSAGE_BYTES + 64 * 1024

#: How much of a connection's egress ``HudConnection.sent()`` remembers. That accessor exists for
#: the unit tests (the socket handler never reads it), so the history is a ring rather than a log:
#: a connection that lives for an hour must not cost this process one message per message spoken
#: (ruling A12).
SENT_HISTORY = 256

#: How long the last page has to come back before the session is told it is alone. A reload or a
#: dropped socket reconnects with T-0008's backoff; ``events.ClientDisconnected`` takes the machine
#: to IDLE and aborts what is live (docs/02 §4), so firing it the instant a socket closes would end
#: the turn a reloading page is still waiting for. Measured on the session's own injected clock —
#: the transport introduces no clock of its own.
#:
#: T-0166: 4000 ms, strictly above the page's 3000 ms mount ceiling (``hud/src/panels.js``
#: ``MOUNT_CEILING_MS``, T-0165), so a reload that falls back to the ceiling still keeps its
#: journal; 1000 ms was measured too short under load. Only a *greeted* connection (a valid
#: ``astra.hello``) is a page for this timer: an ungreeted socket neither cancels nor starts it.
RECONNECT_GRACE_MS = 4000

#: How much of the open turn a (re)connecting page is caught up on (T-0037 criteria 1, 5). A turn
#: is minutes of speech — one ``astra.transcript`` per word and one ``astra.spoken`` per sentence —
#: so the journal keeps the *tail* of it rather than all of it: an unbounded per-turn list would
#: cost this process one retained message per message spoken (ruling A12, applied to the journal).
#: The transcript and the surface are latest-wins (one each), which is why they have no constant.
REPLAY_MAX_SPOKEN = 64
REPLAY_MAX_PLAYED = 32
#: T-0090 criterion 3: the latest ``astra.tool.call`` per ``call_id`` of the open turn. A turn
#: proposes a handful of actions, so this is a bound against a runaway publisher, not a budget.
REPLAY_MAX_TOOL_CALLS = 16
#: T-0180 A4 (H-T180-5): the latest ``astra.thought`` per ``attempt_id`` of the open turn, at most this many
#: attempts; past it the oldest attempt goes and the reply says so (``replay_truncated``).
REPLAY_MAX_THOUGHTS = 16
#: T-0090 ruling U10: how many superseded turn ids the journal remembers besides the current one,
#: so that a late tool call of one of them (a ``cancelled`` after the next turn opened) is not
#: taken for a new turn.
SUPERSEDED_TURNS = 8

#: What the whole journal can hold: the transcript, the spoken lines, the surface, the INTERRUPTED
#: state, the tool calls, the thoughts (T-0180) and the played reports. Derived rather than written
#: down twice, and **strictly less than** ``DEFAULT_OUTBOX`` — a replay as large as an outbox would
#: close the very connection it is catching up, so the page would reconnect, be greeted and be
#: disconnected again, forever.
REPLAY_MAX_MESSAGES = (
    1 + REPLAY_MAX_SPOKEN + 1 + 1 + REPLAY_MAX_TOOL_CALLS + REPLAY_MAX_THOUGHTS + REPLAY_MAX_PLAYED
)

#: ``astra.error`` code for a replay that could not hold the whole turn (T-0037 criterion 6): the
#: page is *told* rather than shown a partial transcript it would believe is complete.
REPLAY_TRUNCATED = "replay_truncated"

#: What that error says. The page renders it in the error strip it already has (``main.js``).
REPLAY_TRUNCATED_MESSAGE = "earlier lines of this turn are not replayed"

#: T-0177 A3: the ``astra.error`` a vision select outside live answers its sender with (only the sender: no state
#: is published outside live, so no other page has anything to learn)
VISION_UNAVAILABLE = "vision_source_unavailable"
VISION_UNAVAILABLE_MESSAGE = "the vision source can be chosen only in a live session"

#: T-0181 A2: the ``astra.error`` a mute in a session with no microphone answers its sender with (its message is the
#: session's reason, verbatim)
MIC_UNAVAILABLE = "mic_unavailable"

#: ``server.counts`` keys, all present from the start (docs/tasks/T-0028.md criteria 3, 4, 6).
COUNTERS: tuple[str, ...] = (
    "routed",
    "wrong_direction",
    "unhandled",
    "relay_ignored",
    "malformed",
    "no_hello",
    "protocol_version",
    "assist_dropped",
    "frames_rejected",
)


class FrameSink(Protocol):
    """What the server does with an ``astra.frame``: hand it over, unmodified (criterion 5)."""

    def ingest(self, message: AstraFrame) -> Any: ...


class TurnJournal:
    """The open turn as it went out, so a page that (re)connects is not shown a lie (T-0037).

    The defect this exists for: a page that reloads mid-turn used to get ``astra.assist.state`` +
    ``astra.turn.state`` and nothing else — a live SPEAKING chip over an empty pane, while TTS
    kept playing the utterance the pane did not show. The session publishes each line once, so the
    page cannot recover on its own.

    **The only source is ``publish()``** (criterion 3): everything here was really put on the wire
    by this server, for this turn. Nothing is synthesised from ``Session``, from history or from
    the machine — a journal fed from those would have something to say about a turn that published
    nothing, which is the same lie in a new shape.

    **One turn, bounded** (criteria 4, 5): a message of a new ``turn_id`` clears it; the
    transcript and the surface are latest-wins and the rest are ``deque``\\ s with a ``maxlen``
    (or dicts with a cap), so recording is O(1) with no allocation proportional to the turn's
    history (criterion 7 — nothing in the pump). Thought text arrives only as ``astra.thought``
    (T-0180, narrowing criterion 4): the latest per ``attempt_id`` of the current turn is kept,
    as a tool call is, and a superseded turn's thought is never recorded.
    """

    def __init__(self) -> None:
        self.turn_id: str | None = None
        self.transcript: AstraTranscript | None = None
        self.surface: AstraUiSurface | None = None
        #: the barge-in chip of *this* turn, restated so the struck tail is explained
        self.interrupted: AstraTurnState | None = None
        self.spoken: deque[AstraSpoken] = deque(maxlen=REPLAY_MAX_SPOKEN)
        self.played: deque[AstraSpokenPlayed] = deque(maxlen=REPLAY_MAX_PLAYED)
        #: T-0090: latest message per ``call_id``, in first-seen order (the page's card order)
        self.tool_calls: dict[str, AstraToolCall] = {}
        #: T-0180 A4: latest message per ``attempt_id``, in first-seen order (the page's block order)
        self.thoughts: dict[str, AstraThought] = {}
        #: T-0090 ruling U10: turn ids in first-seen order, the newest plus the
        #: ``SUPERSEDED_TURNS`` before it. The newest is *current*; every other one is superseded.
        #: Neither ``clear()`` nor a stale message changes it — only a never-seen turn id does.
        self.seen: deque[str] = deque(maxlen=SUPERSEDED_TURNS + 1)
        #: set when a line was dropped, so the page is told rather than left guessing (criterion 6)
        self.truncated = False

    def clear(self) -> None:
        """Forget the turn: a new one opened, the last page went, or the session is shutting down."""
        self.turn_id = None
        self.transcript = None
        self.surface = None
        self.interrupted = None
        self.spoken.clear()
        self.played.clear()
        self.tool_calls.clear()
        self.thoughts.clear()
        self.truncated = False

    def record(self, message: AstraMessage) -> None:
        """One published message, journalled or ignored. Called from the pump: O(1), no await.

        ``astra.ui.surface`` carries no ``turn_id`` (it carries ``surface_id``), so it belongs to
        whatever turn was current when it was published — a journalling rule, not a protocol one.
        """
        if isinstance(
            message,
            AstraTranscript
            | AstraSpoken
            | AstraSpokenPlayed
            | AstraTurnState
            | AstraToolCall
            | AstraThought,
        ):
            if message.turn_id not in self.seen:
                self.seen.append(message.turn_id)  # ruling U10: a never-seen id is current
            elif (
                isinstance(message, AstraToolCall | AstraThought)
                and message.turn_id != self.seen[-1]
            ):
                # Ruling U10 (T-0180 A4: thoughts too): a tool call or thought of a superseded
                # turn (seen, not current) is published live but neither recorded nor allowed
                # to clear anything.
                return
            if message.turn_id != self.turn_id:
                # A current-turn tool call or thought reopens the journal (e.g. after a departure clear).
                # A stale non-tool message still reopens its old turn: the deferred
                # ``spoken.played`` follow-up, unchanged by T-0090.
                self.clear()
                self.turn_id = message.turn_id
        elif not isinstance(message, AstraUiSurface):
            # astra.canvas.*, astra.error and astra.assist.state are nobody's catch-up.
            return
        if isinstance(message, AstraTranscript):
            self.transcript = message  # latest-wins: the partials are one line, not a history
        elif isinstance(message, AstraSpoken):
            self.truncated = self.truncated or len(self.spoken) == REPLAY_MAX_SPOKEN
            self.spoken.append(message)
        elif isinstance(message, AstraUiSurface):
            # Ambiguity 4: the most recent canvas only — the renderer is idempotent per
            # ``surface_id``, so keeping the progressive sequence is not worth the memory.
            self.surface = message
        elif isinstance(message, AstraSpokenPlayed):
            self.truncated = self.truncated or len(self.played) == REPLAY_MAX_PLAYED
            self.played.append(message)
        elif isinstance(message, AstraToolCall):
            # T-0090 criterion 3: latest-wins per call_id, the card's place kept where it was
            if (
                message.call_id not in self.tool_calls
                and len(self.tool_calls) == REPLAY_MAX_TOOL_CALLS
            ):
                del self.tool_calls[next(iter(self.tool_calls))]
                self.truncated = True
            self.tool_calls[message.call_id] = message
        elif isinstance(message, AstraThought):
            # T-0180 A4: latest-wins per attempt_id, the block's place kept where it was
            if (
                message.attempt_id not in self.thoughts
                and len(self.thoughts) == REPLAY_MAX_THOUGHTS
            ):
                del self.thoughts[next(iter(self.thoughts))]
                self.truncated = True
            self.thoughts[message.attempt_id] = message
        elif message.state is TurnState.INTERRUPTED:
            self.interrupted = message

    def messages(self) -> list[AstraMessage]:
        """The catch-up, in the order a page may apply it, each copy marked ``replay=True``.

        The played reports come **after** every ``astra.spoken`` on purpose: the page's reducer
        drops an utterance's played state when a new ``astra.spoken`` for the same
        ``utterance_id`` arrives (T-0030 criterion 18), so the other order would lose the strike
        on every reload — the exact case the rehearsal complained about.
        """
        out: list[AstraMessage] = []
        if self.transcript is not None:
            out.append(self.transcript)
        out.extend(self.spoken)
        if self.surface is not None:
            out.append(self.surface)
        if self.interrupted is not None:
            out.append(self.interrupted)
        # T-0090 ruling U3: after the lines, the surface and the INTERRUPTED state, before the
        # played reports (which T-0030 needs last).
        out.extend(self.tool_calls.values())
        # T-0180 A4: the thoughts after the tool calls, still before the played reports
        out.extend(self.thoughts.values())
        out.extend(self.played)
        return [m.model_copy(update={"replay": True}) for m in out]


class Outbox:
    """A bounded per-connection queue (criterion 12).

    ``offer`` never blocks and never raises. The message kinds that may be dropped are
    ``astra.turn.state`` (idempotent and latest-wins, so a page that missed one is at most a chip
    behind), ``astra.canvas.*`` (a view of the latest denoising step, T-0128c A6) and
    ``astra.vision.frame`` (at most one held, the newest; T-0177 B8). Anything
    else overflowing returns ``False``, and the caller closes that connection — losing a spoken
    line would desync the transcript pane in silence.
    """

    def __init__(self, maxsize: int = DEFAULT_OUTBOX) -> None:
        self.maxsize = max(1, int(maxsize))
        self.dropped = 0
        #: whether the last canvas offer dropped the offered frame itself
        self.last_canvas_lost = False
        #: T-0177 B8: whether the last vision-frame offer dropped the offered frame itself
        self.last_vision_lost = False
        #: T-0166: direct replies (errors about the page's own frames, the hello reply) queued
        #: ahead of the published messages and outside their bound; never held for a hello
        self._head: list[AstraMessage] = []
        self._items: list[AstraMessage] = []
        self._wake = asyncio.Event()

    def __len__(self) -> int:
        return len(self._head) + len(self._items)

    def offer_direct(self, message: AstraMessage, *, bounded: bool = True) -> bool:
        """T-0166: queue a direct reply ahead of every published message.

        ``bounded`` replies (errors) are refused once ``maxsize`` of them are queued, so a page
        that sends garbage and never reads still costs bounded memory. The hello reply is not
        bounded here (H6: it is never refused because published messages fill the queue); it is
        bounded by ``REPLAY_MAX_MESSAGES`` instead.
        """
        if bounded and len(self._head) >= self.maxsize:
            return False
        self._head.append(message)
        self._wake.set()
        return True

    def discard_held(self) -> None:
        """T-0166 H4/H5: forget every published message still queued (not the direct replies)."""
        self._items = []

    def wake(self) -> None:
        self._wake.set()

    def offer(self, message: AstraMessage) -> bool:
        if _is_canvas(message):
            return self._offer_canvas(message)
        if isinstance(message, AstraVisionFrame):
            return self._offer_vision(message)
        if len(self._items) < self.maxsize:
            self._items.append(message)
            self._wake.set()
            return True
        # T-0128c A6: a queued canvas frame is the first thing to go (a denoise frame before a
        # commit); a non-canvas message is never refused because canvas frames filled the queue.
        # T-0177 B8: nor because a held vision frame did.
        if self._evict_canvas() or self._evict_vision():
            self._items.append(message)
            self._wake.set()
            return True
        if not isinstance(message, AstraTurnState):
            return False
        superseded = next(
            (i for i, held in enumerate(self._items) if isinstance(held, AstraTurnState)), None
        )
        self.dropped += 1
        if superseded is None:
            # The queue is full of messages that may not be dropped; the state is the droppable
            # one, so it is this message that goes, not somebody else's spoken line.
            return True
        del self._items[superseded]
        self._items.append(message)
        self._wake.set()
        return True

    def _offer_canvas(self, message: AstraMessage) -> bool:
        """T-0128c A6: an ``astra.canvas.*`` message is droppable and never costs the connection.

        A newer denoise frame of a request replaces that request's queued denoise frame (moved to
        the back, so the request's order is kept); at most ``CANVAS_OUTBOX`` canvas messages are
        queued (the oldest denoise frame goes first; a commit only when none is left and the
        incoming message is not itself a denoise frame, which is then dropped instead); a queue
        full of other messages drops the canvas message itself. Every loss is counted in ``dropped``;
        ``last_canvas_lost`` says whether this offer dropped the offered frame itself (T-0043 c7,
        O-A3: its ``wire.out`` records ``dropped: true``). A queued frame that is replaced or
        evicted already has its ``wire.out``; it is counted in ``dropped`` only.
        """
        self.last_canvas_lost = False
        if isinstance(message, AstraCanvasStep) and message.phase == "denoise":
            same = next(
                (
                    i
                    for i, held in enumerate(self._items)
                    if isinstance(held, AstraCanvasStep)
                    and held.phase == "denoise"
                    and held.request_id == message.request_id
                ),
                None,
            )
            if same is not None:
                del self._items[same]
                self.dropped += 1
        # O-A3 (3): a denoise frame goes before a commit -- the incoming one included, so an
        # incoming denoise frame never evicts a queued commit; it is dropped itself instead
        incoming_denoise = isinstance(message, AstraCanvasStep) and message.phase == "denoise"
        full = len(self._items) >= self.maxsize
        at_cap = sum(1 for held in self._items if _is_canvas(held)) >= CANVAS_OUTBOX
        if (full or at_cap) and not self._evict_canvas(denoise_only=incoming_denoise):
            self.dropped += 1  # the queue is all lines that may not go: this frame goes
            self.last_canvas_lost = True
            return True
        self._items.append(message)
        self._wake.set()
        return True

    def _offer_vision(self, message: AstraVisionFrame) -> bool:
        """T-0177 B8: an ``astra.vision.frame`` is droppable and never costs the connection. At most one is held
        unsent: a newer one replaces it (at the back, so it follows what was queued before it). A queue full of
        other messages drops the offered frame itself (counted; ``last_vision_lost``)."""
        self.last_vision_lost = False
        self._evict_vision()
        if len(self._items) >= self.maxsize and not self._evict_canvas():
            self.dropped += 1
            self.last_vision_lost = True
            return True
        self._items.append(message)
        self._wake.set()
        return True

    def _evict_vision(self) -> bool:
        """T-0177 B8: drop the held vision frame, if any (counted); ``False`` if none was held."""
        held = next((i for i, m in enumerate(self._items) if isinstance(m, AstraVisionFrame)), None)
        if held is None:
            return False
        del self._items[held]
        self.dropped += 1
        return True

    def _evict_canvas(self, *, denoise_only: bool = False) -> bool:
        """Drop one queued canvas message, the oldest denoise frame first and a commit (or any
        other canvas message) only when no denoise frame is queued and ``denoise_only`` is off.
        Counted; ``False`` if nothing was evicted."""
        canvas = [i for i, held in enumerate(self._items) if _is_canvas(held)]
        denoise = [
            i
            for i in canvas
            if isinstance(held := self._items[i], AstraCanvasStep) and held.phase == "denoise"
        ]
        if not denoise and (denoise_only or not canvas):
            return False
        del self._items[denoise[0] if denoise else canvas[0]]
        self.dropped += 1
        return True

    def pending(self) -> list[AstraMessage]:
        """What is queued: the direct replies, then the published messages, each oldest first.
        Does not consume."""
        return [*self._head, *self._items]

    def drain(self) -> list[AstraMessage]:
        """Everything queued, in ``pending()`` order; the queue is left empty."""
        out = [*self._head, *self._items]
        self._head, self._items = [], []
        self._wake.clear()
        return out

    def drain_direct(self) -> list[AstraMessage]:
        """T-0166: only the direct replies; the published messages stay held."""
        out, self._head = self._head, []
        self._wake.clear()
        return out

    async def wait(self) -> None:
        await self._wake.wait()


class HudConnection:
    """One client: what it said, what it is owed, and whether it is still welcome.

    Has no socket of its own — ``receive`` takes a frame exactly as it came off the wire and the
    writer task takes what ``outbox`` holds — so every ingress and egress decision is testable
    without a network.
    """

    def __init__(self, server: HudServer, *, outbox_size: int = DEFAULT_OUTBOX) -> None:
        self._server = server
        self.outbox = Outbox(outbox_size)
        self.close_reason: str | None = None
        self.closed = False
        #: set when this connection must go: the socket handler awaits it and closes the wire
        self.closing = asyncio.Event()
        #: hello-first (criterion 4): nothing is routed until a valid one arrives
        self.greeted = False
        self._written: deque[AstraMessage] = deque(maxlen=SENT_HISTORY)

    # --- egress ---------------------------------------------------------------------------

    def offer(self, message: AstraMessage) -> bool:
        """Queue one message, or close this connection because it cannot keep up.

        Overflow of a kind that may not be dropped ends *this* connection — on the wire, not only
        in this process (ruling A12): a page that is told nothing keeps a socket open and a
        transcript that silently lost a line. Returns whether this connection took the message,
        which is what ``publish`` records as ``dropped`` (T-0043 criterion 7).
        """
        if self.closed:
            return False
        if not self.outbox.offer(message):
            self._overflow(message)
            return False
        return True

    def _overflow(self, message: AstraMessage) -> None:
        self.close(reason=f"outbox overflow ({self.outbox.maxsize} messages) on {message.type}")

    def take(self) -> list[AstraMessage]:
        """What the writer sends next (T-0166): everything once greeted; before the hello only
        the direct replies, so nothing published reaches the wire ahead of the hello reply."""
        return self.outbox.drain() if self.greeted else self.outbox.drain_direct()

    def sent(self) -> list[AstraMessage]:
        """What this connection would put on the wire, in order, oldest first.

        A test accessor: what has actually gone out (the last ``SENT_HISTORY`` messages) followed
        by what is still queued. A message the outbox dropped is in neither.
        """
        return [*self._written, *self.outbox.pending()]

    def note_written(self, messages: Sequence[AstraMessage]) -> None:
        self._written.extend(messages)

    # --- ingress --------------------------------------------------------------------------

    def receive(self, raw: str | bytes) -> None:
        """One inbound frame. Total: nothing a client can send raises out of here."""
        if self.closed:
            return
        decoded = decode(raw)
        self._log_wire_in(raw, decoded)
        counts = self._server.counts
        if isinstance(decoded, Malformed):
            counts["malformed"] += 1
            self._error("malformed", decoded.error)
            return
        if isinstance(decoded, Unknown):
            counts["unhandled"] += 1
            return
        if isinstance(decoded, RealtimeEvent):
            # v1 relays nothing: a third-party Realtime client is counted and ignored.
            counts["relay_ignored"] += 1
            return
        message = cast("AstraMessage", decoded)
        kind = message.type
        if kind in SERVER_TO_CLIENT:
            counts["wrong_direction"] += 1
            self._error("wrong_direction", f"{kind} is a server→client message")
            return
        if kind not in CLIENT_TO_SERVER:  # pragma: no cover - the two sets cover every model
            counts["unhandled"] += 1
            return
        if isinstance(message, AstraHello):
            self._hello(message)
            return
        if not self.greeted:
            counts["no_hello"] += 1
            self._error("no_hello", "send astra.hello before anything else")
            return
        self._server.route(message, self)

    def _log_wire_in(self, raw: str | bytes, decoded: Any) -> None:
        """T-0043 criterion 7: one ``wire.in`` per inbound frame, refused ones included.

        A page whose messages are all being dropped as ``no_hello`` or wrong-direction looks, from
        the session's side, exactly like a page that is sending nothing — so the tap is here, in
        the transport, before any of the decisions below. It neither consumes nor mutates the
        frame; an oversized message contributes its byte count and its outcome and never its body
        (``Malformed`` carries the codec's reason, not the payload). ``isascii`` is what keeps the
        byte count from copying a near-limit message just to measure it.
        """
        log = self._server.session.session_log
        if log is None:
            return
        if isinstance(raw, str):
            n_bytes = len(raw) if raw.isascii() else len(raw.encode("utf-8"))
        else:
            n_bytes = len(raw)
        outcome = "astra"
        message: Any
        if isinstance(decoded, Malformed):
            outcome, message = "malformed", _malformed_row(decoded)
        elif isinstance(decoded, Unknown):
            outcome, message = "unknown", {"type": decoded.type}
        elif isinstance(decoded, RealtimeEvent):
            outcome, message = "realtime", _dumped_row(decoded)
        elif isinstance(decoded, AstraSettingsSet):
            message = _settings_set_row(decoded)  # T-0213 R-14: never the instructions text
        else:
            message = _dumped_row(cast("AstraMessage", decoded))
        log.record("wire.in", {"n_bytes": n_bytes, "outcome": outcome, "msg": message})

    def _hello(self, hello: AstraHello) -> None:
        if hello.protocol_version != PROTOCOL_VERSION:
            self._server.counts["protocol_version"] += 1
            self._error(
                "protocol_version",
                f"this endpoint speaks protocol {PROTOCOL_VERSION}, "
                f"the page speaks {hello.protocol_version}",
            )
            return
        self._server.counts["routed"] += 1
        if self.greeted:
            # A repeated hello on a greeted page: queued behind what it is already owed, as before.
            for message in self._server.hello_reply():
                self.offer(message)
            return
        self.greeted = True
        # T-0166 G2a: this connection is a page now; a pending departure was a reconnect.
        self._server.greet(self)
        # The page starts from the server's truth rather than from a guess (criterion 4): T-0025's
        # assist panel and T-0008's state chip both key on these two. T-0166 H2/H6: the reply goes
        # ahead of everything published while this page was ungreeted, and is never refused for it.
        for message in self._server.hello_reply():
            self.outbox.offer_direct(message, bounded=False)
        self.outbox.wake()

    def _error(self, code: str, message: str) -> None:
        error = AstraError(code=code, message=message)
        if self.greeted:
            self.offer(error)
            return
        # T-0166 H3: a reply about the page's own frame is not published, so it is not held.
        if not self.closed and not self.outbox.offer_direct(error):
            self._overflow(error)

    # --- lifecycle ------------------------------------------------------------------------

    def close(self, *, reason: str | None = None) -> None:
        """The client went away, or we gave up on it. Idempotent.

        ``reason`` is set when *we* gave up; the socket handler is waiting on ``closing`` and
        turns it into a close frame on the wire.
        """
        if self.closed:
            return
        self.closed = True
        if reason is not None:
            self.close_reason = reason
            log.warning("closing a HUD connection: %s", reason)
        self.closing.set()
        self._server.forget(self)


class HudServer:
    """``ws://127.0.0.1:<conductor_port>/v1/realtime`` over one ``Session``."""

    def __init__(
        self,
        session: Session,
        *,
        frame_sink: FrameSink,
        services: Services | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        path: str = DEFAULT_PATH,
        outbox_size: int = DEFAULT_OUTBOX,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            # Before any socket exists (criterion 1): a listener on a routable address is
            # reachable from the network whatever the page's URL says (docs/03 E8).
            raise ValueError(
                f"the conductor endpoint binds loopback only, not {host!r} "
                f"(docs/02 §9; one of {sorted(LOOPBACK_HOSTS)})"
            )
        self.session = session
        self.frame_sink = frame_sink
        self.services = services if services is not None else Services()
        self.host = host
        self.port = self.services.conductor_port if port is None else int(port)
        self.path = path
        self.outbox_size = outbox_size
        self.counts: dict[str, int] = dict.fromkeys(COUNTERS, 0)
        self._connections: list[HudConnection] = []
        #: what this server published for the open turn, for the next page's hello (T-0037)
        self._journal = TurnJournal()
        #: T-0162 A-5: the last session reset this server published, restated first at every hello
        self._cleared: AstraSessionCleared | None = None
        #: T-0162 A-6: the canvas request ids relayed (bounded, oldest dropped) and those relayed before the last reset
        self._canvas_seen: deque[str] = deque(maxlen=CANVAS_IDS_KEPT)
        self._canvas_cut: frozenset[str] = frozenset()
        #: T-0177 B9: the latest ``astra.vision.frame`` published since the last reset, last in every hello
        self._vision_frame: AstraVisionFrame | None = None
        self._server: Server | None = None
        self._stopping = False
        self._lonely: Any | None = None
        session.set_publisher(self.publish)

    # --- the listening socket -----------------------------------------------------------------

    async def start(self) -> None:
        """Bind and serve. ``port=0`` takes a free loopback port and updates ``self.port``."""
        if self._server is not None:
            return
        self._server = await serve(
            self._serve_connection,
            self.host,
            self.port,
            max_size=MAX_SIZE,
            process_request=self._process_request,
        )
        bound = self.sockets
        if bound:
            self.port = bound[0].getsockname()[1]

    async def aclose(self) -> None:
        """Stop serving and drop every connection. No ``client.disconnected`` is invented here:
        the session is being taken down by its owner, not abandoned by its last page."""
        self._stopping = True
        self._cancel_lonely()
        self._journal.clear()  # nothing outlives the session being taken down (criterion 4)
        for connection in list(self._connections):
            connection.close(reason="server shutting down")
        self._connections.clear()
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        self.session.set_publisher(None)

    @property
    def sockets(self) -> tuple[socket.socket, ...]:
        server = self._server
        if server is None:
            return ()
        return tuple(server.sockets)

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{WS_SCHEME}://{host}:{self.port}{self.path}"

    # --- the handshake fence ------------------------------------------------------------------

    @staticmethod
    def origin_allowed(origin: str | None) -> bool:
        """Criterion 2: loopback pages, native clients and ``file://`` pages; nothing else.

        Loopback *binding* does not fence the browser: a page served from a remote site runs on
        this machine and can open ``ws://127.0.0.1:<port>``. The ``Origin`` header is what
        distinguishes it from the HUD, and it is checked before a single message is read.
        """
        if origin is None or origin == "null":
            return True
        if origin.startswith("file://"):
            return True
        host = urlsplit(origin).hostname
        return host in LOOPBACK_HOSTS

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if urlsplit(request.path).path != self.path:
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown path\n")
        if not self.origin_allowed(request.headers.get("Origin")):
            return connection.respond(HTTPStatus.FORBIDDEN, "origin not allowed\n")
        return None

    # --- connections --------------------------------------------------------------------------

    def open(self) -> HudConnection:
        """A connection with no socket behind it; the socket handler is one of these plus tasks.

        T-0166 G2a: opening does not cancel a pending departure; only a valid hello does
        (``greet``), so a socket that never says hello cannot hide the last page leaving.
        """
        connection = HudConnection(self, outbox_size=self.outbox_size)
        self._connections.append(connection)
        return connection

    def greet(self, connection: HudConnection) -> None:
        """T-0166 G2a: a valid hello makes a connection a page; a pending departure is cancelled."""
        self._cancel_lonely()  # a reconnect, not a disconnect (criterion 13)

    def _greeted_present(self) -> bool:
        return any(c.greeted for c in self._connections)

    def forget(self, connection: HudConnection) -> None:
        """The last *greeted* page going away is what ``events.ClientDisconnected`` means
        (criterion 13; T-0166 G2).

        A disconnect while another greeted window is connected changes nothing, and the last one
        is only reported after ``RECONNECT_GRACE_MS`` on the session's clock, so a reload is a
        reconnect rather than the end of the session. A socket that never said a valid hello
        neither starts, restarts nor delays that timer when it closes.
        """
        if connection in self._connections:
            self._connections.remove(connection)
        if not connection.greeted or self._greeted_present() or self._stopping:
            return
        clock = self.session.clock
        self._cancel_lonely()
        self._lonely = clock.call_at(clock.now_ms + RECONNECT_GRACE_MS, self._on_last_gone)

    def _cancel_lonely(self) -> None:
        lonely, self._lonely = self._lonely, None
        if lonely is not None:
            lonely.cancel()

    def _on_last_gone(self) -> None:
        self._lonely = None
        if self._greeted_present() or self._stopping:
            return
        # The last page is really gone, so the turn it was watching is nobody's catch-up any
        # more: what was published for it goes before the session is told (criterion 4).
        self._journal.clear()
        # T-0166 H5: nor is it owed live to a socket that has not said hello yet.
        for connection in self._connections:
            connection.outbox.discard_held()
        self.session.submit(E.ClientDisconnected(t_ms=self.session.clock.now_ms))

    async def _serve_connection(self, ws: ServerConnection) -> None:
        connection = self.open()
        writer = asyncio.create_task(self._write(connection, ws), name="hud-writer")
        closer = asyncio.create_task(self._close_socket(connection, ws, writer), name="hud-closer")
        try:
            async for raw in ws:
                connection.receive(raw)
        except Exception as exc:  # a dead socket is not the session's problem
            log.info("HUD connection ended: %s", exc)
        finally:
            writer.cancel()
            closer.cancel()
            connection.close()

    async def _close_socket(
        self, connection: HudConnection, ws: ServerConnection, writer: asyncio.Task[None]
    ) -> None:
        """Turn "we gave up on this connection" into a close frame (ruling A12).

        Without this the reader loop would only notice at the next inbound frame, which a page
        that stopped reading may never send: the peer keeps a live socket, is never told, and the
        writer task outlives it.
        """
        await connection.closing.wait()
        writer.cancel()
        with contextlib.suppress(Exception):
            await ws.close(code=1011, reason=(connection.close_reason or "closed")[:120])

    async def _write(self, connection: HudConnection, ws: ServerConnection) -> None:
        """The only place that awaits the wire, and it is nobody's pump."""
        while True:
            await connection.outbox.wait()
            batch = connection.take()
            for message in batch:
                await ws.send(encode(message))
            connection.note_written(batch)

    # --- egress -------------------------------------------------------------------------------

    def publish(self, message: AstraMessage) -> None:
        """The session's ``HudPublisher``: fan out and return, never await (criterion 12)."""
        if isinstance(message, AstraSessionCleared):
            # T-0162 A-5: a new session has nothing to catch up on; the reset itself is restated
            self._journal.clear()
            self._cleared = message
            self._vision_frame = None  # T-0177 B9: a reset clears the slot
            # T-0162 A-6: every canvas request relayed so far belongs to a session that is over
            self._canvas_cut = frozenset(self._canvas_seen)
        elif _is_canvas(message):
            # T-0162 A-6: the canvas relay publishes here directly with vLLM's own request ids, so a request is
            # judged by whether this server relayed it before the last reset
            request_id = cast("AstraCanvasStep", message).request_id
            if request_id in self._canvas_cut or self.session.is_pre_reset(message):
                return
            if request_id not in self._canvas_seen:
                self._canvas_seen.append(request_id)
        elif isinstance(message, AstraVisionFrame):
            # T-0177 B9: kept for the next hello, never in the T-0037 journal
            self._vision_frame = message
        self._journal.record(message)
        refused = False
        canvas = _is_canvas(message)
        vision = isinstance(message, AstraVisionFrame)
        reset = isinstance(message, AstraSessionCleared)
        for connection in list(self._connections):
            if reset and not connection.greeted:
                # T-0166 H4: what an ungreeted page holds is from before the reset, and the reset
                # itself reaches it replayed, first in its hello reply.
                connection.outbox.discard_held()
                continue
            if not connection.offer(message):
                refused = True
            elif canvas and connection.outbox.last_canvas_lost:
                refused = True  # T-0128c O-A3 (4): a canvas frame its outbox dropped or replaced
            elif vision and connection.outbox.last_vision_lost:
                refused = True  # T-0177 B8: a vision frame its outbox dropped
        # T-0043 criterion 7: one ``wire.out`` per published message, page attached or not — a
        # session run with no page open still logs what it would have shown. A message an outbox
        # refused is recorded as dropped rather than as sent.
        log = self.session.session_log
        if log is not None:
            log.record(
                "wire.out",
                {
                    "n_bytes": len(encode(message).encode("utf-8")),
                    # T-0128c A7: a canvas step logs its compact row, not its three CL-long rows
                    "msg": _canvas_row(message)
                    if isinstance(message, AstraCanvasStep)
                    else _thought_row(message)
                    if isinstance(message, AstraThought)
                    else _vision_row(message)
                    if isinstance(message, AstraVisionFrame)
                    else _settings_state_row(message)
                    if isinstance(message, AstraSettingsState)
                    else message.model_dump(mode="json"),
                    "dropped": refused,
                    "connections": len(self._connections),
                },
            )

    def hello_reply(self) -> Iterator[AstraMessage]:
        """What a (re)connecting page is owed: the server's truth, then the open turn (T-0037).

        The first two messages are the ones the page has always been sent — the assist state and
        the current turn state, live (``replay=False``). What follows is the journal: a truncation
        notice if a line was dropped, the turn's last transcript, its spoken lines in publication
        order, its most recent surface, the INTERRUPTED state if one was published for this turn,
        the latest ``astra.tool.call`` per ``call_id`` (T-0090), the latest ``astra.thought`` per
        ``attempt_id`` (T-0180) and the played reports last. The current turn state is then repeated so the chip is not
        left reading a state the server has already left (ruling A3); that repeat is a replay, so
        it is marked like the rest.

        T-0162 A-5: after a session reset the reply begins with that reset, replayed, so a page
        that reconnects clears what it still shows from before it. With no reset, unchanged.

        T-0181 A4: ``astra.mic.state`` follows ``astra.log.state`` in every reply (fourth with no reset, fifth after
        one), live (``replay`` is not a field of it).

        T-0213 A2r-3 (R-5): ``astra.settings.state``, with its ``catalog``, last in every reply, after everything
        above (so no earlier message moves), live (``replay=False``).

        T-0177 A4: in live the reply ends with the session's vision choice (``astra.vision.state``), after
        everything above; outside live there is none and the reply is unchanged. T-0177 B9: then, in every mode, the
        latest ``astra.vision.frame`` published since the last reset, replayed; absent when there is none.
        """
        yield from self._hello_journal()
        vision = self.session.vision_state_message()
        if vision is not None:
            yield vision
        if self._vision_frame is not None:
            yield self._vision_frame.model_copy(update={"replay": True})
        # T-0213 A2r-3 (R-5): the settings, with the catalog (the only message that carries it), last in every reply
        yield self.session.settings_state_message(catalog=True)

    def _hello_journal(self) -> Iterator[AstraMessage]:
        """``hello_reply`` up to the vision state: the reset, the server's truth, then the open turn."""
        if self._cleared is not None:
            yield self._cleared.model_copy(update={"replay": True})
        yield self.session.assists.state_message()
        yield self.session.turn_state_message()
        # T-0043 criterion 18: third, before the replay — a page that reloads after someone
        # switched logging off must start from the state the session is in *now*, and must have
        # its checkbox settled before it is busy catching up on a turn.
        yield self.session.log_state_message()
        # T-0181 A4: the mic switch right after the log state, in every reply (a session with no mic says why)
        yield self.session.mic_state_message()
        caught_up = self._journal.messages()
        if not caught_up:
            return  # nothing was published for a turn: today's two-message reply, unchanged
        if self._journal.truncated:
            yield AstraError(code=REPLAY_TRUNCATED, message=REPLAY_TRUNCATED_MESSAGE)
        yield from caught_up
        yield self.session.turn_state_message().model_copy(update={"replay": True})

    # --- ingress ------------------------------------------------------------------------------

    def route(self, message: AstraMessage, connection: HudConnection) -> None:
        """One client→server message, after the hello fence. The session decides what it means."""
        if isinstance(message, AstraFrame):
            if self.frame_sink.ingest(message) is None:
                self.counts["frames_rejected"] += 1
        elif isinstance(message, AstraAssistToggle):
            self.session.apply_assist_toggle(message)
        elif isinstance(message, AstraFrameText):
            if self.session.accept_frame_text(message) is None:
                # T-0012's fail-closed drop is counted here, never rescued.
                self.counts["assist_dropped"] += 1
        elif isinstance(message, AstraText):
            # T-0039 criterion 3: one typed question, one event, on the one pump. The transport
            # neither strips nor caps (``AstraText`` did) nor refuses as busy — the state a
            # refusal depends on is the state the event *arrives* in, behind the frames, ticks
            # and onsets already queued, and only the session knows that.
            self.session.submit(E.UserText(t_ms=self.session.clock.now_ms, text=message.text))
        elif isinstance(message, AstraLogToggle):
            # T-0043 criterion 17: the session owns the switch; the transport only delivers it.
            # Every routed toggle publishes one ``astra.log.state`` to every connection — the
            # switch is the session's, not the socket's (ruling B2).
            self.session.apply_log_toggle(message)
        elif isinstance(message, AstraSessionReset):
            # T-0162 A-5: one event on the one pump; the machine's row and the session do the rest
            self.session.submit(E.SessionReset(t_ms=self.session.clock.now_ms))
        elif isinstance(message, AstraVisionSelect):
            # T-0177 A3: the session owns the choice and, in live, echoes it to every page (redundant ones too)
            if self.session.apply_vision_select(message) is None:
                connection._error(VISION_UNAVAILABLE, VISION_UNAVAILABLE_MESSAGE)
        elif isinstance(message, AstraMicMute):
            # T-0181 A2: the session owns the switch and, with a mic, echoes it to every page (redundant ones too)
            if self.session.apply_mic_mute(message) is None:
                connection._error(MIC_UNAVAILABLE, self.session.mic_unavailable or MIC_UNAVAILABLE)
        elif isinstance(message, AstraTurnStop):
            # T-0193 A-2: one event on the one pump; the machine decides whether the stop applies (a stale or
            # idle stop is its no-op, never the transport's refusal)
            self.session.submit(E.UserStop(t_ms=self.session.clock.now_ms, turn_id=message.turn_id))
        elif isinstance(message, AstraSettingsSet):
            # T-0213 A2a-2: the session owns the settings; the set goes on the one pump, is validated there and
            # applied only at a turn boundary, and every page gets the ``astra.settings.state`` echo
            self.session.apply_settings_set(message)
        elif isinstance(message, AstraUiAction):
            self.session.submit(
                E.UiAction(
                    t_ms=self.session.clock.now_ms,
                    surface_id=message.surface_id,
                    name=message.name,
                    context=dict(message.context),
                )
            )
        else:  # pragma: no cover - CLIENT_TO_SERVER has no other member
            self.counts["unhandled"] += 1
            return
        self.counts["routed"] += 1
