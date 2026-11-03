"""v1 perception source: frames posted by the HUD page (``astra.frame``). Task T-0008.

Keeps a 60 s ring buffer per source; ``get_frame(at_ms)`` returns the frame at or just before
speech onset (never "latest at end-of-turn"); emits ``SceneChanged`` on perceptual-hash distance.

No I/O and no clock of its own (criterion 18, the turn machine's discipline): the receive clock
arrives as ``BrowserFrameSource(now_ms=<callable returning ms>)`` (ruling R2). ``Frame.ts_ms`` is
that receive time, **not** the client's ``ts_ms``: the page's ``Date.now()`` is not the audio
clock, and ``get_frame(at=speech_onset)`` compares against audio time.

Client input is validated at this boundary: ``policy.scene_changed`` raises ``ValueError`` on
hashes of differing lengths, so a ``phash`` that is not exactly ``PHASH_HEX_LEN`` lowercase hex
characters is rejected (counted on ``rejected``) before it can reach it.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence

from local_astra.contracts.events import SceneChanged
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.perception import policy

#: The dHash the page computes (T-0008 criterion 5): 64 bits as 16 lowercase hex characters.
PHASH_HEX_LEN = 16
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Ring cap in frames, on top of the 60 s window (ruling R5): >= 61 so a compliant 1 fps client
#: loses nothing in 60 s, <= 1000 so a client that ignores the rate limit stays bounded.
MAX_RING_FRAMES = 600

#: Hysteresis for ``SceneChanged``: a scene that is *not* a change from any phash announced or
#: seen in this window is not a new scene, so an alternating A/B/A/B capture (a blinking cursor,
#: a flickering video) does not emit one event per frame (criterion 17).
SCENE_MEMORY_MS = 10_000
SCENE_MEMORY_KEYS = 8


def _decoded_jpeg(jpeg_b64: str) -> bytes | None:
    try:
        return base64.b64decode(jpeg_b64, validate=True)
    except ValueError:  # binascii.Error is a ValueError
        return None


def _valid_phash(phash: str | None) -> str | None:
    """The only shape ``policy.scene_changed`` may ever see, or ``None`` (criterion 14)."""
    if phash is None or len(phash) != PHASH_HEX_LEN:
        return None
    lowered = phash.lower()
    if not set(lowered) <= _HEX_DIGITS:
        return None
    return lowered


class BrowserFrameSource:
    """``PerceptionClient`` over ``astra.frame`` messages from the HUD page."""

    def __init__(
        self,
        *,
        now_ms: Callable[[], int],
        ring_seconds: int = policy.RING_BUFFER_S,
        max_frames: int = MAX_RING_FRAMES,
    ) -> None:
        self._now_ms = now_ms
        self.ring_seconds = ring_seconds
        self.max_frames = max_frames
        #: frames refused at the boundary (bad phash, bad base64) — counted, never raised
        self.rejected = 0
        self._rings: dict[Source, deque[Frame]] = {s: deque() for s in Source}
        self._scene_memory: dict[Source, deque[tuple[int, str]]] = {s: deque() for s in Source}
        #: the last phash announced as the current scene, per source; never expires (R14)
        self._last_scene: dict[Source, str | None] = {s: None for s in Source}
        self._pending_events: deque[SceneChanged] = deque(maxlen=256)
        self._wake = asyncio.Event()

    # --- ingest ---------------------------------------------------------------------------

    def ingest(self, msg: AstraFrame) -> Frame | None:
        """Store one posted frame, or reject it (counted). Never raises on client input."""
        jpeg = _decoded_jpeg(msg.jpeg_b64)
        phash = _valid_phash(msg.phash)
        if jpeg is None or phash is None:
            self.rejected += 1
            return None
        ts_ms = int(self._now_ms())
        frame = Frame(
            source=msg.source,
            ts_ms=ts_ms,
            width=msg.width,
            height=msg.height,
            jpeg=jpeg,
            phash=phash,
            title=msg.title,
            app=msg.app,
        )
        ring = self._rings[msg.source]
        ring.append(frame)
        self._evict(ring, ts_ms)
        self._note_scene(frame)
        return frame

    def _evict(self, ring: deque[Frame], now_ms: int) -> None:
        cutoff = now_ms - self.ring_seconds * 1000
        while ring and ring[0].ts_ms <= cutoff:  # exactly RING_BUFFER_S old is evicted (R5)
            ring.popleft()
        while len(ring) > self.max_frames:
            ring.popleft()

    def _note_scene(self, frame: Frame) -> None:
        memory = self._scene_memory[frame.source]
        while memory and frame.ts_ms - memory[0][0] > SCENE_MEMORY_MS:
            memory.popleft()
        # The comparison is always against the last *accepted* scene, which never expires (ruling
        # R14): a source that was quiet for longer than SCENE_MEMORY_MS and comes back showing a
        # different screen is a scene change. The memory window only suppresses A/B/A/B chatter.
        last = self._last_scene[frame.source]
        seen_recently = any(not policy.scene_changed(seen, frame.phash) for _, seen in memory)
        memory.append((frame.ts_ms, frame.phash))
        while len(memory) > SCENE_MEMORY_KEYS:
            memory.popleft()
        if last is None:  # the first frame of an empty ring is not a scene *change* (R5)
            self._last_scene[frame.source] = frame.phash
            return
        if seen_recently or not policy.scene_changed(last, frame.phash):
            return
        self._last_scene[frame.source] = frame.phash
        self._pending_events.append(
            SceneChanged(t_ms=frame.ts_ms, source=frame.source.value, phash=frame.phash)
        )
        self._wake.set()

    # --- PerceptionClient -----------------------------------------------------------------

    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        """The frame at or just before ``at_ms`` (speech onset), never a later one."""
        ring = self._rings[source]
        # Prune on read too (ruling R9): a producer that stopped an hour ago must yield no frame
        # rather than an hour-old one, which nothing downstream would notice.
        self._evict(ring, int(self._now_ms()))
        if not ring:
            return None
        if at_ms is None:
            return ring[-1]
        chosen: Frame | None = None
        for frame in ring:  # ascending by receive time, so the last match wins
            if frame.ts_ms > at_ms:
                break
            chosen = frame
        return chosen

    async def clip(
        self, source: Source, t0_ms: int, t1_ms: int, fps: float = 1.0
    ) -> Sequence[Frame]:
        spacing_ms = 1000.0 / fps if fps > 0 else 0.0
        out: list[Frame] = []
        for frame in self._rings[source]:
            if frame.ts_ms < t0_ms or frame.ts_ms > t1_ms:
                continue
            if out and frame.ts_ms - out[-1].ts_ms < spacing_ms:
                continue
            out.append(frame)
        return out

    async def events(self) -> AsyncIterator[SceneChanged]:
        """``SceneChanged`` and nothing else (contracts.protocols.PerceptionClient)."""
        while True:
            self._wake.clear()
            while self._pending_events:
                yield self._pending_events.popleft()
            await self._wake.wait()
