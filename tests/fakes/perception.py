"""Fake ``PerceptionClient`` with a ring of frames keyed by timestamp."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from local_astra.contracts.events import SceneChanged
from local_astra.contracts.perception import Frame, Source


class FakePerception:
    def __init__(self, frames: Sequence[Frame] = ()) -> None:
        self.frames = sorted(frames, key=lambda f: f.ts_ms)
        self.scene_events: list[SceneChanged] = []

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)
        self.frames.sort(key=lambda f: f.ts_ms)

    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        cands = [
            f for f in self.frames if f.source == source and (at_ms is None or f.ts_ms <= at_ms)
        ]
        return cands[-1] if cands else None

    async def clip(
        self, source: Source, t0_ms: int, t1_ms: int, fps: float = 1.0
    ) -> Sequence[Frame]:
        step = int(1000 / fps)
        out: list[Frame] = []
        last = -step
        for f in self.frames:
            if f.source == source and t0_ms <= f.ts_ms <= t1_ms and f.ts_ms - last >= step:
                out.append(f)
                last = f.ts_ms
        return out

    async def events(self) -> AsyncIterator[object]:
        for e in self.scene_events:
            yield e
