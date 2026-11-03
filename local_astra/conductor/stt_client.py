"""moshi-server streaming STT client (``/api/asr-streaming``, msgpack over WebSocket). Task T-0007a.

Fixed facts (docs/02 §2.3, docs/06 R2, and what was measured by ``scripts/demo/_listen.py``):
input is 24 kHz mono f32 PCM in whatever blocks the caller hands over; the server answers with
``Ready`` / ``Word`` / ``EndWord`` / ``Step`` traffic (the four types T-0007a's group B census saw
on the wire), lags the audio by ~0.5 s, and its model clock **restarts at zero on every
connection**; a ``Step`` arrives every 80 ms whether or not anyone is speaking, so silence on the
socket says nothing at all. A word's start arrives on its ``Word`` and its end on the ``EndWord``
that follows, both in seconds, so a word is published only once its end is known (T-0038) — or,
when that ``EndWord`` never comes, once ``STT_OPEN_WORD_HOLD_MS`` of *fed audio* has gone by, with
``end_estimated=True`` on the event and a count in ``stats()`` (T-0051 criteria 8-9, from the desk
session of 2026-09-16 where one such word was held for 21 040 ms and published inside the human's
next sentence).

What this module deliberately does *not* do:

* it never resamples, reframes, pads or re-times audio — ``feed`` is a pure consumer of T-0021's
  framing, one block in, one ``Audio`` message out, and the wire carries no timestamps;
* it never synthesises a flush. ``SttFlushedEvent`` comes from the server echoing a ``Marker``
  and from nothing else: not from a tail timer, not from silence, not from ``close()``. The
  spike's 2.5 s drain heuristic exists because that experiment had no marker, and carrying it
  across would make the end-of-turn a clock again (T-0007a criterion 4);
* it never decides end-of-turn. The pause heads travel as ``SttStepEvent`` and the reflex lane
  decides (T-0007);
* it never raises a service failure out of ``feed``: every failure is one event on ``events()``,
  which only ``close()`` may end.
"""

from __future__ import annotations

import asyncio
import statistics
import struct
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

import msgpack

from local_astra.config import LOOPBACK_PREFIXES, Services
from local_astra.contracts.speech import (
    STT_INPUT_FORMAT,
    SttErrorEvent,
    SttEvent,
    SttFlushedEvent,
    SttStepEvent,
    SttWordEvent,
)

#: The one token in ``services/moshi/stt-1b-en_fr.toml``'s ``authorized_ids``.
DEFAULT_AUTH_ID = "public_token"

#: Outbound blocks kept while the peer is slow. 80 ms a block, so 50 is 4 s of audio — well over
#: the 2 s criterion 8 asks for, and bounded, which is the point.
DEFAULT_MAX_QUEUED_BLOCKS = 50

#: Reconnect attempts after a server-side close, and the bounded backoff between them.
DEFAULT_MAX_RECONNECTS = 3
BACKOFF_BASE_S = 0.05
BACKOFF_CAP_S = 1.0

#: How long ``close()`` waits for the close handshake before abandoning it. The measured hang in
#: ``scripts/demo/_listen.py::deliver`` cost 10+ seconds of dead air; criterion 6 caps it.
CLOSE_TIMEOUT_S = 0.1

#: Events already produced are kept, but not without limit: a consumer that stops reading must
#: not turn a chatty server into unbounded memory.
MAX_QUEUED_EVENTS = 2048

#: Error text is bounded and single-line: a stack trace on the event stream is not a diagnosis.
MAX_ERROR_CHARS = 200
_CAUSE_CHARS = 120

#: How much **step-clock** (fed-audio) time a ``Word`` whose ``EndWord`` never arrives may be held
#: before it is published anyway, with ``end_estimated=True`` (T-0051 criterion 9). Not a timer:
#: the clock this is measured in is ``_audio_clock_ms``, which only ``feed`` moves, so an idle
#: machine does not publish and a slow one does not hold the word any longer than the audio says.
#: Band, from outside this module: ≥ 600 ms because the recogniser's own emission delay was
#: 490–590 ms of audio time on the GPU on 2026-09-16, and a shorter hold would pre-empt ``EndWord``
#: messages that are merely in flight; ≤ 5000 ms because the hold exists to bound the 21 040 ms
#: wait that lost the human's last word that evening (``logs/sessions/20260916-170934-3380576.jsonl``
#: seq 3844: ``see?`` published at the head of the *next* sentence with a 21 s invented duration).
#: 1000 ms — about twice the measured delay — is the value T-0051 recommends inside that band.
STT_OPEN_WORD_HOLD_MS = 1000


class SttConnection(Protocol):
    """The transport seam. ``websockets.connect(...)``'s result satisfies it structurally."""

    async def send(self, data: bytes) -> None: ...

    async def recv(self) -> Any: ...

    async def close(self) -> None: ...


Connect = Callable[[str], Awaitable[SttConnection]]


async def _websocket_connect(url: str) -> SttConnection:
    """The default transport: a real websocket. Imported lazily so construction opens nothing."""
    import websockets

    connection: SttConnection = await websockets.connect(url)
    return connection


def _brief(exc: BaseException, limit: int = _CAUSE_CHARS) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def _bounded(message: str) -> str:
    return message.replace("\n", " ").replace("\r", " ")[:MAX_ERROR_CHARS]


class MoshiSttClient:
    """One connection to moshi-server, reused (``batch_size = 2`` is the whole box's budget)."""

    def __init__(
        self,
        services: Services | None = None,
        *,
        connect: Connect | None = None,
        auth_id: str = DEFAULT_AUTH_ID,
        max_reconnects: int = DEFAULT_MAX_RECONNECTS,
        max_queued_blocks: int = DEFAULT_MAX_QUEUED_BLOCKS,
    ) -> None:
        resolved = services if services is not None else Services()
        if not resolved.stt.startswith(LOOPBACK_PREFIXES):
            raise ValueError(f"non-loopback endpoint {resolved.stt!r} (docs/02 §9)")
        self.url = f"{resolved.stt}?auth_id={auth_id}"
        self.max_queued_blocks = max_queued_blocks
        self._connect: Connect = connect if connect is not None else _websocket_connect
        self._max_reconnects = max_reconnects
        self._sample_rate = STT_INPUT_FORMAT.sample_rate

        self._outbound: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queued_blocks)
        self._events: asyncio.Queue[SttEvent | None] = asyncio.Queue(maxsize=MAX_QUEUED_EVENTS)
        self._task: asyncio.Task[None] | None = None
        self._conn: SttConnection | None = None
        self._closed = False
        self._ended = False
        self._overflow_reported = False

        #: end of the audio handed over so far; the fallback word time and the ``Step`` time.
        self._audio_clock_ms = 0
        #: ``t_ms`` of the first block fed on the current connection; the server's clock's origin.
        self._base_ms: int | None = None
        self._next_marker_id = 1
        self._pending_markers: dict[int, int] = {}
        #: the word whose ``EndWord`` has not arrived yet: ``(text, start_ms, opened_at_ms)``,
        #: where ``opened_at_ms`` is the fed-audio clock at which the ``Word`` turned up — the
        #: hold's zero (T-0051 criterion 9). A word is published when its end is known (T-0038
        #: criterion 1), or when the hold expires, and never before.
        self._open_word: tuple[str, int, int] | None = None
        #: One ``arrival_audio_clock_ms - end_ms`` per **server-closed** word (T-0051 criterion 11,
        #: ruling B25: a word this client closed itself has an end this client invented, so
        #: including it would measure our own guess and report it as the recogniser's latency).
        #: Kept whole because the median is an order statistic; a desk session produces single
        #: digits of these and an hour of continuous speech a few thousand ints.
        self._word_delivery_lags: list[int] = []
        #: How many words the hold released whose server ``EndWord`` has not arrived yet. It is
        #: what tells a late end apart from a stray one when an ``EndWord`` finds nothing open:
        #: each released word claims at most one of them (T-0051 ruling B38), and the claim is
        #: given up as soon as a new ``Word`` opens (ruling B40) — an end that arrives once the
        #: next word has started is not the held word's, so it is a stray again.
        self._held_words_awaiting_end: int = 0
        self._stats: dict[str, int] = {
            "estimated_word_times": 0,
            "words_without_end": 0,
            "held_word_timeouts": 0,
            "stray_end_words": 0,
            "late_end_words": 0,
            "ready_seen": 0,
            "stale_markers": 0,
            "unknown_messages": 0,
            "dropped_blocks": 0,
            "dropped_events": 0,
            "reconnects": 0,
            "connects": 0,
            "close_failures": 0,
        }

    # --- the SttClient face -------------------------------------------------------------------

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        """One block in, one ``Audio`` message out. Never blocks the reflex lane for long."""
        if len(pcm_24k_f32) % 4:
            raise ValueError(
                f"{len(pcm_24k_f32)} bytes is not a whole number of f32 samples "
                "(the caller frames the audio, T-0021)"
            )
        samples = struct.unpack(f"<{len(pcm_24k_f32) // 4}f", pcm_24k_f32)
        self._ensure_running()
        if self._base_ms is None:
            self._base_ms = t_ms
        self._audio_clock_ms = t_ms + len(samples) * 1000 // self._sample_rate
        self._expire_open_word()
        self._enqueue({"type": "Audio", "pcm": list(samples)})

    async def mark(self, t_eot_ms: int) -> int:
        """Ask the server to echo a ``Marker``; its echo — and only its echo — becomes a flush."""
        marker_id = self._next_marker_id
        self._next_marker_id += 1
        self._pending_markers[marker_id] = t_eot_ms
        self._ensure_running()
        self._enqueue({"type": "Marker", "id": marker_id}, evict=True)
        return marker_id

    def events(self) -> AsyncIterator[SttEvent]:
        return self._events_iter()

    async def _events_iter(self) -> AsyncIterator[SttEvent]:
        while True:
            if self._ended and self._events.empty():
                return
            item = await self._events.get()
            if item is None:
                self._ended = True
                return
            yield item

    async def close(self) -> None:
        """Idempotent, and it cannot hang: the close handshake is given ``CLOSE_TIMEOUT_S``."""
        if self._closed:
            return
        self._closed = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        conn, self._conn = self._conn, None
        if conn is not None:
            await self._close_connection(conn)
        # The last word of a turn is never dropped (docs/02 §4 row 6): a word still waiting for
        # its ``EndWord`` is published here, ending where it started (T-0038 criterion 1).
        self._close_open_word(None)
        self._ended = True
        self._offer(None)

    def stats(self) -> Mapping[str, int]:
        """The counters, plus the delivery lag the next desk session needs (criterion 11).

        ``arrival_audio_clock_ms - end_ms`` per word the **server** closed — the number that says
        whether the recogniser is keeping up, and the number the endpointer's backstop rule is
        protecting (it was 874–923 ms in session time, ≈ 490–590 ms in this clock, on 2026-09-16).
        Words this client closed itself are excluded and counted as ``held_word_timeouts`` /
        ``words_without_end`` instead (ruling B25 (i)), and the median is ``median_low`` so every
        number reported is a lag some word actually had (ruling B25 (ii)) rather than the average
        of two that nothing observed. With no observations yet all four read 0.
        """
        lags = self._word_delivery_lags
        return {
            **self._stats,
            "queued_blocks": self._outbound.qsize(),
            "word_delivery_lag_count": len(lags),
            "word_delivery_lag_min_ms": min(lags) if lags else 0,
            "word_delivery_lag_median_ms": statistics.median_low(lags) if lags else 0,
            "word_delivery_lag_max_ms": max(lags) if lags else 0,
        }

    # --- outbound ------------------------------------------------------------------------------

    def _ensure_running(self) -> None:
        """Lazy, once: the connection opens on the first block, and is then reused."""
        if self._task is None and not self._closed:
            self._task = asyncio.create_task(self._run(), name="moshi-stt")

    def _enqueue(self, message: dict[str, Any], *, evict: bool = False) -> None:
        payload = bytes(msgpack.packb(message, use_bin_type=True))
        if evict and self._outbound.full() and not self._outbound.empty():
            self._outbound.get_nowait()  # a control message displaces one block, never the reverse
            self._stats["dropped_blocks"] += 1
        try:
            self._outbound.put_nowait(payload)
        except asyncio.QueueFull:
            self._stats["dropped_blocks"] += 1
            if not self._overflow_reported:
                self._overflow_reported = True
                self._emit_error(
                    f"stt outbound buffer full at {self.max_queued_blocks} blocks; "
                    "audio is being dropped"
                )

    def _drop_queued(self) -> None:
        """No audio from before a disconnect is replayed onto the new connection."""
        while not self._outbound.empty():
            self._outbound.get_nowait()
            self._stats["dropped_blocks"] += 1
        self._overflow_reported = False

    # --- the connection supervisor ----------------------------------------------------------------

    def _stopping(self) -> bool:
        """A method, not the attribute: ``close()`` sets it from another task, mid-``await``."""
        return self._closed

    async def _run(self) -> None:
        attempt = 0
        while not self._stopping():
            reason: str | None = None
            try:
                conn = await self._connect(self.url)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = _bounded(f"stt connect failed ({_brief(exc)})")
            else:
                self._stats["connects"] += 1
                self._conn = conn
                try:
                    reason = await self._serve(conn)
                finally:
                    self._conn = None
                    # Audio queued for the connection that just died is not replayed onto the
                    # next one, and moshi's clock restarts at zero, so the base goes with it —
                    # and so does any open word, published here on the old base rather than
                    # closed later against the new one.
                    self._close_open_word(None)
                    self._base_ms = None
                    self._drop_queued()
                    await self._close_connection(conn)
            if self._stopping():
                return
            if reason is not None:
                self._emit_error(reason)
            if attempt >= self._max_reconnects:
                return
            attempt += 1
            self._stats["reconnects"] += 1
            await asyncio.sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_CAP_S))

    async def _serve(self, conn: SttConnection) -> str | None:
        reader = asyncio.create_task(self._read_loop(conn), name="moshi-stt-read")
        sender = asyncio.create_task(self._send_loop(conn), name="moshi-stt-send")
        both = {reader, sender}
        try:
            done, _ = await asyncio.wait(both, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in both:
                task.cancel()
            await asyncio.gather(*both, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            reason = task.result()
            if reason is not None:
                return reason
        return None

    async def _read_loop(self, conn: SttConnection) -> str | None:
        while True:
            try:
                raw = await conn.recv()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return _bounded(f"stt connection lost ({_brief(exc)})")
            try:
                message = msgpack.unpackb(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit_error(_bounded(f"stt frame could not be decoded ({_brief(exc)})"))
                continue
            self._handle(message)

    async def _send_loop(self, conn: SttConnection) -> str | None:
        while True:
            payload = await self._outbound.get()
            try:
                await conn.send(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return _bounded(f"stt send failed ({_brief(exc)})")
            if self._outbound.empty():
                self._overflow_reported = False

    async def _close_connection(self, conn: SttConnection) -> None:
        try:
            await asyncio.wait_for(conn.close(), CLOSE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._stats["close_failures"] += 1

    # --- inbound mapping ------------------------------------------------------------------------

    def _handle(self, message: Any) -> None:
        if not isinstance(message, dict):
            self._unknown(type(message).__name__)
            return
        kind = message.get("type")
        if kind == "Word":
            self._word(message)
        elif kind == "EndWord":
            self._end_word(message)
        elif kind == "Step":
            self._step(message)
        elif kind == "Marker":
            self._marker(message)
        elif kind == "Ready":
            # Measured by the group B census: one per connection, keys exactly ``{'type'}``. A
            # second one means the server restarted its stream, and the reconnect path already
            # re-bases the clock, so there is nothing to do but count it (T-0038 ruling A1).
            self._stats["ready_seen"] += 1
        else:
            # Still total, on purpose: the census mapped ``Ready`` and ``EndWord`` and nothing
            # else. A server-sent ``Error`` stays *documented*, not observed, and swallowing a
            # type nobody has measured is how a protocol change becomes "transcribes nothing"
            # (T-0007a ruling A8, which T-0038 criterion 4 keeps).
            self._unknown(str(kind))

    def _word(self, message: dict[str, Any]) -> None:
        """Open a word. It is published when its end is known, not here (T-0038 criterion 1)."""
        text = message.get("text")
        if not isinstance(text, str):
            # Ruling A3: nothing is opened, and whatever was open stays open.
            self._unknown("Word without text")
            return
        start_time = message.get("start_time")
        if isinstance(start_time, int | float) and self._base_ms is not None:
            start_ms = self._base_ms + round(float(start_time) * 1000)
        else:
            start_ms = self._audio_clock_ms
            self._stats["estimated_word_times"] += 1
        start_ms = max(start_ms, 0)
        # No ``EndWord`` came between the two words: the previous one ends where this one starts.
        self._close_open_word(start_ms, estimated=True)
        self._held_words_awaiting_end = 0  # a new word ends the late-end claim (ruling B40)
        self._open_word = (text, start_ms, self._audio_clock_ms)

    def _end_word(self, message: dict[str, Any]) -> None:
        """``{'type': 'EndWord', 'stop_time': 1.44}`` — seconds, one per word (census, 0.6.4)."""
        stop_time = message.get("stop_time")
        if not isinstance(stop_time, int | float):
            # Ruling A2: loud, and the open word stays open for the next well-formed end.
            self._unknown("EndWord without a numeric stop_time")
            return
        if self._open_word is None:
            if self._held_words_awaiting_end:
                # Answers "did *we* give up too early?": the hold released a word and the
                # server's end for it turned up after all, so the recogniser did close it and
                # this client did not wait. Non-zero means ``STT_OPEN_WORD_HOLD_MS`` is short for
                # this machine, which is the serious reading (T-0051 ruling B38).
                self._held_words_awaiting_end -= 1
                self._stats["late_end_words"] += 1
            else:
                # Answers "is the server ahead of us?": the word this end would have closed was
                # already published by the next word, a marker flush, a disconnect or ``close()``.
                # The server is ahead of us, not broken.
                self._stats["stray_end_words"] += 1
            return
        if self._base_ms is None:
            self._close_open_word(None)  # no clock origin yet, so no end can be computed
            return
        self._close_open_word(self._base_ms + round(float(stop_time) * 1000))

    def _close_open_word(self, end_ms: int | None, *, estimated: bool = False) -> None:
        """Publish the open word, if any. ``None`` — or an end before the start — means unknown.

        An end that precedes its start would let T-0032 endpoint a turn *before* its last word
        began, so it is clamped to the start and declared an estimate, never a negative duration.

        ``end_estimated`` travels with the event (T-0051 criterion 8) so the endpointer can tell
        an end the server measured from one this client supplied; only the former is an
        observation of the recogniser's delivery lag (criterion 11, ruling B25).
        """
        if self._open_word is None:
            return
        text, start_ms, _opened_at_ms = self._open_word
        self._open_word = None
        if end_ms is None or end_ms < start_ms:
            end_ms = start_ms  # a duration is never invented
            estimated = True
        if estimated:
            self._stats["words_without_end"] += 1
        else:
            self._word_delivery_lags.append(self._audio_clock_ms - end_ms)
        self._offer(
            SttWordEvent(text=text, start_ms=start_ms, end_ms=end_ms, end_estimated=estimated)
        )

    def _expire_open_word(self) -> None:
        """Release a word the server never closed, once the hold has run out (criterion 9).

        The hold is measured from the fed-audio clock at which the ``Word`` turned up, not from
        the word's own audio ``start_ms``: the two differ by the recogniser's emission delay
        (≈ 510 ms on the GPU on 2026-09-16, seconds on the CPU fallback), and subtracting one
        from the other is the two-clock bug that cut the human off in the same session (T-0051
        ruling A3: a duration is never computed from two clocks — group C writes that into
        `.claude/rules/conductor.md`).

        The end is the word's own start, as it already is on every other path that has no
        measured end: *a duration is never invented*. That is the honest floor of ruling B24's
        window ``start_ms <= end_ms <= start_ms + STT_OPEN_WORD_HOLD_MS`` — the clock at
        publication is outside it, and is how a 21-second word happens again under another name.
        """
        open_word = self._open_word
        if open_word is None:
            return
        if self._audio_clock_ms - open_word[2] < STT_OPEN_WORD_HOLD_MS:
            return
        self._stats["held_word_timeouts"] += 1
        # The server may still close this word; the next ``EndWord`` that finds nothing open is
        # then a late end and not a stray, which is the opposite diagnosis (ruling B38).
        self._held_words_awaiting_end += 1
        self._close_open_word(None)

    def _step(self, message: dict[str, Any]) -> None:
        raw = message.get("prs")
        heads = tuple(float(p) for p in raw) if isinstance(raw, list | tuple) else ()
        self._offer(SttStepEvent(pause_heads=heads, t_ms=max(self._audio_clock_ms, 0)))

    def _marker(self, message: dict[str, Any]) -> None:
        marker_id = message.get("id")
        t_eot_ms = (
            self._pending_markers.pop(marker_id, None) if isinstance(marker_id, int) else None
        )
        if t_eot_ms is None:
            self._stats["stale_markers"] += 1
            return
        # docs/02 §4 row 6: the flush promises every word up to ``t_eot`` has been delivered, so
        # a word still waiting for its end is published first (T-0038 criterion 5).
        self._close_open_word(None)
        self._offer(SttFlushedEvent(t_eot_ms=t_eot_ms))

    def _unknown(self, name: str) -> None:
        self._stats["unknown_messages"] += 1
        self._emit_error(_bounded(f"stt: unknown message type {name[:64]!r}"))

    # --- the event stream -------------------------------------------------------------------------

    def _emit_error(self, message: str) -> None:
        self._offer(SttErrorEvent(message=_bounded(message)))

    def _offer(self, event: SttEvent | None) -> None:
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            self._stats["dropped_events"] += 1
