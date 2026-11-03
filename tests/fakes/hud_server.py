"""Stub Conductor socket for the HUD page (T-0008): records the client, replays a script.

The real endpoint is ``local_astra/conductor/session.py`` (T-0007's file, not written yet), so the
integration test needs something on ``ws://127.0.0.1:<port>/v1/realtime`` that speaks the wire
format of ``local_astra/contracts/protocol.py``. This is that something:

* it **records every client message** verbatim (``received``) and decodes it with the *real*
  ``contracts.codec.decode`` (``received_decoded``/``frames``) — the page's output is validated by
  the production codec, never by a second implementation of it;
* it **replays a scripted server sequence** as soon as the client says ``astra.hello``, including
  the deliberately broken entries criterion 2 is about (raw ``str`` entries go on the wire
  unchanged, ``Contract`` entries are encoded with ``codec.encode``);
* it records what the socket did afterwards (``client_closed``, ``close_code``, ``connections``),
  which is how "the socket must still be open" is asserted;
* it can **push one message at a time** from the test thread (``push``/``wait_for_client``), which
  is what a progressive render needs: T-0024 criterion 7 is about what the page does *between*
  canvases, and a script that arrives in one burst cannot observe that. The real endpoint
  (T-0007's ``session.py``) emits the same ``astra.ui.surface`` messages one canvas at a time.

Loopback only, one thread, no dependency beyond ``websockets`` (already a project dependency).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

import websockets

from local_astra.contracts.base import Contract
from local_astra.contracts.codec import Decoded, decode, encode
from local_astra.contracts.protocol import AstraFrame

#: Well above ``codec.MAX_MESSAGE_BYTES`` so an oversized *client* message is recorded and
#: asserted about instead of killing the connection inside the websockets library.
MAX_SIZE = 32 * 1024 * 1024


class HudStub:
    """A loopback ``ws://`` endpoint that records the HUD and answers with a fixed script."""

    def __init__(
        self,
        script: Sequence[str | Contract] = (),
        *,
        path: str = "/v1/realtime",
        send_after: str = "astra.hello",
        host: str = "127.0.0.1",
    ) -> None:
        self.script: list[str | Contract] = list(script)
        self.path = path
        self.send_after = send_after
        self.host = host
        self.port = 0
        self.received: list[str] = []
        self.connections = 0
        self.client_closed = False
        self.close_code: int | None = None
        self.script_sent = threading.Event()
        self.script_error: str | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._conn: Any = None

    # --- lifecycle -------------------------------------------------------------------------

    def start(self) -> HudStub:
        self._thread = threading.Thread(target=self._run, name="hud-stub", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("hud stub did not start")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> HudStub:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    # --- what the client sent --------------------------------------------------------------

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.received)

    def received_decoded(self) -> list[Decoded]:
        return [decode(raw) for raw in self.snapshot()]

    def received_types(self) -> list[str]:
        out: list[str] = []
        for raw in self.snapshot():
            try:
                obj = json.loads(raw)
            except ValueError:
                out.append("<invalid json>")
                continue
            out.append(str(obj.get("type")) if isinstance(obj, dict) else "<not an object>")
        return out

    def frames(self) -> list[AstraFrame]:
        return [m for m in self.received_decoded() if isinstance(m, AstraFrame)]

    def wait_for(self, predicate: Callable[[HudStub], bool], timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(self):
                return True
            time.sleep(0.05)
        return predicate(self)

    def wait_for_frames(self, n: int, timeout_s: float = 15.0) -> bool:
        return self.wait_for(lambda s: len(s.frames()) >= n, timeout_s)

    def vision_selects(self) -> list[Any]:
        """T-0177 A2: every ``astra.vision.select`` the page sent, decoded by the *real* codec, in order.

        Filtered on the decoded type string rather than on the model class, so this module still imports (and
        every other row of the stub still runs) before ``contracts.protocol.AstraVisionSelect`` exists: until
        then the codec decodes the message as ``Unknown`` and the list is empty.
        """
        return [
            m for m in self.received_decoded() if getattr(m, "type", None) == "astra.vision.select"
        ]

    def wait_for_vision_selects(self, n: int, timeout_s: float = 10.0) -> bool:
        return self.wait_for(lambda s: len(s.vision_selects()) >= n, timeout_s)

    def mic_mutes(self) -> list[Any]:
        """T-0181 A1: every ``astra.mic.mute`` the page sent, decoded by the *real* codec, in order (filtered on
        the decoded type string, as ``vision_selects``, so this module imports before the model exists)."""
        return [m for m in self.received_decoded() if getattr(m, "type", None) == "astra.mic.mute"]

    def wait_for_mic_mutes(self, n: int, timeout_s: float = 10.0) -> bool:
        return self.wait_for(lambda s: len(s.mic_mutes()) >= n, timeout_s)

    def turn_stops(self) -> list[Any]:
        """T-0193 A-1: every well-formed ``astra.turn.stop`` the page sent, decoded by the *real* codec, in order.

        Filtered on the decoded type string of a decoded ``Contract`` (as ``vision_selects``, so this module imports
        before ``AstraTurnStop`` exists; until then the codec answers ``Unknown`` and the list is empty). A malformed
        stop (an empty or non-string ``turn_id``) is the codec's ``Malformed`` and is not one."""
        return [
            m
            for m in self.received_decoded()
            if isinstance(m, Contract) and getattr(m, "type", None) == "astra.turn.stop"
        ]

    def wait_for_turn_stops(self, n: int, timeout_s: float = 10.0) -> bool:
        return self.wait_for(lambda s: len(s.turn_stops()) >= n, timeout_s)

    # --- server side -----------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(  # type: ignore[attr-defined]
            self._handler, self.host, 0, max_size=MAX_SIZE
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.02)

    async def _handler(self, conn: Any) -> None:
        with self._lock:
            self.connections += 1
            self._conn = conn
        self._connected.set()
        try:
            async for raw in conn:
                message = raw if isinstance(raw, str) else bytes(raw).decode("utf-8", "replace")
                with self._lock:
                    self.received.append(message)
                if self._is_trigger(message) and not self.script_sent.is_set():
                    await self._send_script(conn)
        except Exception as exc:  # a client that goes away is data, not an error
            self.client_closed = True
            self.close_code = getattr(exc, "code", None)
        finally:
            self.client_closed = True
            if self.close_code is None:
                self.close_code = getattr(conn, "close_code", None)
            with self._lock:
                if self._conn is conn:
                    self._conn = None

    # --- pushing one message at a time (T-0024 criterion 7) --------------------------------

    def wait_for_client(self, timeout_s: float = 10.0) -> bool:
        """Block until a client is connected (``push`` needs one)."""
        return self._connected.wait(timeout=timeout_s) and self._conn is not None

    def push(self, message: str | Contract, timeout_s: float = 10.0) -> None:
        """Send one message to the connected client, now, from the test thread.

        ``Contract`` entries are encoded with ``codec.encode``; a raw ``str`` goes on the wire
        unchanged, warts and all (same rule as the scripted entries), so a deliberately malformed
        message can be pushed at a chosen moment.
        """
        if not self.wait_for_client(timeout_s):
            raise RuntimeError("no HUD client is connected; push has nobody to send to")
        conn = self._conn
        loop = self._loop
        if conn is None or loop is None:
            raise RuntimeError("the stub is not serving")
        raw = message if isinstance(message, str) else encode(message)
        future = asyncio.run_coroutine_threadsafe(conn.send(raw), loop)
        future.result(timeout=timeout_s)

    def drop_client(self, timeout_s: float = 10.0) -> None:
        """Close the connected client's socket from the server side, now (T-0162 B-7).

        A dropped connection, as a restarted session gives one: the page's backoff reconnect
        (``hud/src/ws.js``) then opens a second socket and sends a second ``astra.hello``, and
        ``push`` sends to that one. This is a transport event the test drives, not a behaviour
        of the Conductor, so it has no ``real`` twin.
        """
        conn = self._conn
        loop = self._loop
        if conn is None or loop is None:
            raise RuntimeError("no HUD client is connected; there is nothing to drop")
        asyncio.run_coroutine_threadsafe(conn.close(), loop).result(timeout=timeout_s)

    def _is_trigger(self, message: str) -> bool:
        try:
            obj = json.loads(message)
        except ValueError:
            return False
        return isinstance(obj, dict) and obj.get("type") == self.send_after

    async def _send_script(self, conn: Any) -> None:
        try:
            for entry in self.script:
                await conn.send(entry if isinstance(entry, str) else encode(entry))
        except Exception as exc:
            self.script_error = (
                f"the socket died while the script was being sent ({type(exc).__name__}: {exc}); "
                "a throw inside the page's message handler is the usual cause"
            )
        finally:
            self.script_sent.set()


def oversized_message() -> str:
    """A well-formed message that is over ``codec.MAX_MESSAGE_BYTES`` and must be ignored."""
    from local_astra.contracts.codec import MAX_MESSAGE_BYTES

    return (
        '{"type":"astra.transcript","turn_id":"t","text":"' + "x" * (MAX_MESSAGE_BYTES + 64) + '"}'
    )


# --- T-0037: the hello replay (criterion 9 — the fake moves with the wire) -----------------------


def replay_script(
    turn_id: str = "t1",
    *,
    utterances: Sequence[tuple[str, str]] = (
        ("u1", "Here is a quick check."),
        ("u2", "Try the five questions."),
    ),
    question: str = "what is on this screen",
    played_words: int = 2,
    with_tool_call: bool = False,
) -> list[Contract]:
    """What a page that reconnects mid-turn is sent after its ``astra.hello`` (T-0037 criterion 1).

    The real endpoint builds this from its journal of what it published; the stub cannot have a
    journal, so this helper builds the *same shape* — the turn's last ``astra.transcript``, its
    ``astra.spoken`` lines in publication order, its most recent ``astra.ui.surface`` and its
    ``astra.spoken.played`` last, every one of them marked ``replay=True``. CLAUDE.md: changing a
    wire format means changing the fake in the same PR, so the page-side rows that drive against
    this stub exercise the very shape ``tests/contract/test_hud_hello_replay_wire.py`` asserts
    against ``HudServer``.

    ``replay=True`` is passed to the constructors on purpose: before T-0037 group A lands,
    ``contracts.base.Contract``'s ``extra="forbid"`` refuses it, and a caller is expected to have
    asserted that the five models carry the field (that assertion is the acceptance criterion,
    and it belongs in the test, not in the fake).

    ``with_tool_call`` (T-0090 criterion 3): the journal also holds the latest ``astra.tool.call``
    per ``call_id`` of the open turn, replayed after the spoken lines and before the played
    reports. Off by default so the rows that predate T-0090 keep their shape; the import happens
    only when asked, so this module still imports before ``AstraToolCall`` exists.
    """
    from local_astra.contracts.protocol import (
        AstraSpoken,
        AstraSpokenPlayed,
        AstraTranscript,
        AstraUiSurface,
        PlayedUtterance,
    )

    catalog = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
    script: list[Contract] = [
        AstraTranscript(turn_id=turn_id, text=question, final=True, replay=True),
        *(
            AstraSpoken(turn_id=turn_id, utterance_id=f"{turn_id}-{u}", text=text, replay=True)
            for u, text in utterances
        ),
        AstraUiSurface(
            surface_id=f"turn-{turn_id}",
            canvas_idx=1,
            final=True,
            messages=(
                {
                    "version": "v0.9.1",
                    "createSurface": {"surfaceId": f"turn-{turn_id}", "catalogId": catalog},
                },
            ),
            replay=True,
        ),
        AstraSpokenPlayed(
            turn_id=turn_id,
            played_ms=1400,
            utterances=[
                PlayedUtterance(
                    utterance_id=f"{turn_id}-{u}",
                    played_text=" ".join(text.split()[:played_words]),
                    played_words=played_words,
                    complete=False,
                )
                for u, text in utterances
            ],
            replay=True,
        ),
    ]
    if with_tool_call:
        from local_astra.contracts.protocol import AstraToolCall  # type: ignore[attr-defined]

        query = "weather in Paris tomorrow"
        script.insert(
            len(script) - 1,  # after the spoken lines and the surface, before the played report
            AstraToolCall(
                turn_id=turn_id,
                call_id=f"{turn_id}-c1",
                name="google_search",
                arguments={"query": query},
                status="dry_run",
                source="draft",
                action=f"would search for: {query}",
                replay=True,
            ),
        )
    return script


# --- T-0180: the model's committed thoughts (A3 — the fake moves with the wire) -------------------

#: One block of a turn's "Model thoughts": ``(kind, step, text, end)`` or
#: ``(kind, step, text, end, omitted_chars)``. ``kind`` and ``end`` are ``AstraThought``'s literals.
ThoughtBlock = tuple[Any, ...]

#: The default turn: step 1, a re-ask of step 2 and a repair, so a page row sees three labels.
DEFAULT_THOUGHT_BLOCKS: tuple[ThoughtBlock, ...] = (
    ("step", 1, "The page shows a search box.\nI should type the query.", "done"),
    ("reask", 2, "The click did not land; try the button text.", "done"),
    ("repair", 3, "Fix the arguments: x must be on the grid.", "open"),
)


def thought_script(
    turn_id: str = "t1",
    *,
    blocks: Sequence[ThoughtBlock] = DEFAULT_THOUGHT_BLOCKS,
    progressive: bool = False,
    replay: bool = False,
) -> list[Contract]:
    """``astra.thought`` messages for one turn, one attempt per block, in block order (T-0180 A3).

    Each block is one attempt: ``attempt_id`` is ``"<turn>-a<i>"`` and ``request_id``
    ``"<turn>-r<i>"`` (``i`` from 1). With ``progressive`` every attempt is first sent ``open`` with
    the first half of its text (the cumulative text so far) and then its final message, as a session
    publishes after each thought commit; without it, only the final message, which is exactly what
    the hello replay journals (the latest per ``attempt_id``, ``replay=True``). The builder carries
    the text it is given and never cuts it: a block that needs the cap passes its own
    ``omitted_chars``.

    The import happens on call, so this module still imports before ``AstraThought`` exists (the
    model is the acceptance criterion; a missing one fails the caller's assertion, not this import).
    """
    from local_astra.contracts import protocol

    model = getattr(protocol, "AstraThought", None)
    if model is None:
        raise NotImplementedError(
            "T-0180 A1: contracts/protocol.py gains AstraThought (type 'astra.thought'); "
            "thought_script cannot build a message before it exists"
        )
    out: list[Contract] = []
    for i, block in enumerate(blocks, start=1):
        kind, step, text, end = block[0], block[1], block[2], block[3]
        omitted = int(block[4]) if len(block) > 4 else 0
        ids = {
            "turn_id": turn_id,
            "request_id": f"{turn_id}-r{i}",
            "attempt_id": f"{turn_id}-a{i}",
            "step": step,
            "kind": kind,
        }
        if progressive and end != "open":
            out.append(
                model(
                    **ids, text=text[: len(text) // 2], omitted_chars=0, end="open", replay=replay
                )
            )
        out.append(model(**ids, text=text, omitted_chars=omitted, end=end, replay=replay))
    return out


# --- T-0177 A2: the vision selector (the fake moves with the wire) --------------------------------
#
# ``astra.vision.state`` (server->client: the session's choice, echoed to every page) and ``astra.vision.frame``
# (server->client: the images one request actually sent, the panel's only source). The constructors are imported
# inside each helper so this module imports before T-0177 A1 lands; a caller asserts the models exist first
# (the assertion is the acceptance criterion and belongs in the test, not in the fake).

#: T-0177 A1's two choices, in the order the combo box lists them (C1)
VISION_CHOICES: tuple[str, ...] = ("browser", "shared")


def vision_image(
    jpeg: bytes | None,
    *,
    identity: str,
    kind: str = "window",
    width: int = 1280,
    height: int = 800,
) -> Any:
    """One ``VisionImage``: the bytes base64-encoded, or (``jpeg=None``) the ``omitted: "too_large"`` form."""
    import base64

    from local_astra.contracts import protocol

    cls = protocol.VisionImage  # type: ignore[attr-defined]
    if jpeg is None:
        return cls(
            identity=identity,
            kind=kind,
            width=width,
            height=height,
            jpeg_b64=None,
            omitted="too_large",
        )
    return cls(
        identity=identity,
        kind=kind,
        width=width,
        height=height,
        jpeg_b64=base64.b64encode(jpeg).decode(),
    )


def vision_state(
    source: str = "browser",
    *,
    choices: Sequence[str] = VISION_CHOICES,
    replay: bool = False,
) -> Contract:
    """``astra.vision.state``: the session's current choice (A1)."""
    from local_astra.contracts import protocol

    cls = protocol.AstraVisionState  # type: ignore[attr-defined]
    message: Contract = cls(source=source, choices=list(choices), replay=replay)
    return message


def vision_frame(
    turn_id: str = "t1",
    step: int = 1,
    *,
    source: str = "browser",
    images: Sequence[Any] = (),
    blind: bool = False,
    history_images: int = 0,
    replay: bool = False,
) -> Contract:
    """``astra.vision.frame``: the images one request sent (A1; B7 publishes it)."""
    from local_astra.contracts import protocol

    cls = protocol.AstraVisionFrame  # type: ignore[attr-defined]
    message: Contract = cls(
        turn_id=turn_id,
        step=step,
        source=source,
        blind=blind,
        images=list(images),
        history_images=history_images,
        replay=replay,
    )
    return message


def vision_script(
    source: str = "browser",
    *,
    choices: Sequence[str] = VISION_CHOICES,
    frames: Sequence[Contract] = (),
    replay: bool = False,
) -> list[Contract]:
    """What a live session's page is sent about vision: the state echo, then each published frame, in order.

    ``replay=True`` marks the state as a hello replay would; the frames carry their own flag (``vision_frame``).
    """
    return [vision_state(source, choices=choices, replay=replay), *frames]


# --- T-0181 A1: the microphone mute (the fake moves with the wire) -----------------------------------------
#
# ``astra.mic.state`` (server->client): the session's mute switch, echoed to every page at every routed
# ``astra.mic.mute`` and in every hello reply. Imported inside the helper so this module imports before A1 lands.


def mic_state(
    muted: bool = False, *, available: bool = True, reason: str | None = None
) -> Contract:
    """``astra.mic.state``: the switch as the session echoes it (A1). A session with no microphone is
    ``mic_state(available=False, reason=...)``, never muted."""
    from local_astra.contracts import protocol

    cls = protocol.AstraMicState  # type: ignore[attr-defined]
    message: Contract = cls(muted=muted, available=available, reason=reason)
    return message
