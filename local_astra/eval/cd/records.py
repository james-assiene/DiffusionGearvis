"""The engine-log join (T-0070 criterion 5): ``ASTRA_CD <json>`` records, keyed on ``record_tag``.

The plugin (T-0067) writes one ``CdCanvasRecord`` per committed canvas into the engine's log, behind
vLLM's own line prefix, and a record may lag the HTTP response. ``EngineLog`` tails the file from where
the run started, validates every record against the contract (an off-contract line is counted, never
used), indexes them by ``record_tag`` and waits, with a deadline, until a tag's canvases are complete:
contiguous ``canvas_idx`` from 0, ending on a canvas that carries a stop, an error, or that brings the
committed length to ``max_tokens``. The harness sets a unique tag per request, so decoys of other tags
and other runs are ignored by construction.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_astra.contracts.cd import CdCanvasRecord

__all__ = ["RECORD_MARK", "EngineLog", "Joined", "is_complete", "join"]

RECORD_MARK = "ASTRA_CD "


@dataclasses.dataclass(frozen=True)
class Joined:
    """A request's canvases, in ``canvas_idx`` order."""

    records: tuple[CdCanvasRecord, ...]
    #: ``canvas_idx`` values that arrived more than once (the join rejects them).
    duplicate_idx: tuple[int, ...] = ()

    @property
    def contiguous(self) -> bool:
        return [r.canvas_idx for r in self.records] == list(range(len(self.records)))

    @property
    def valid(self) -> bool:
        """No duplicated canvas and no gap: anything else is a ``join_error``, whatever the text."""
        return bool(self.records) and not self.duplicate_idx and self.contiguous

    @property
    def committed_ids(self) -> list[int]:
        return [int(t) for r in self.records for t in r.committed_ids]

    @property
    def steps(self) -> list[int]:
        return [int(r.steps) for r in self.records]

    @property
    def in_final(self) -> bool:
        """The plugin's verdict on the last canvas."""
        return bool(self.records[-1].in_final)

    @property
    def error(self) -> str | None:
        return next((r.error for r in self.records if r.error is not None), None)

    @property
    def bucket(self) -> int | None:
        return next((int(r.bucket) for r in self.records if r.bucket is not None), None)

    def per_canvas(self, name: str) -> list[Any]:
        return [getattr(r, name) for r in self.records]


def is_complete(recs: Sequence[CdCanvasRecord], max_tokens: int) -> bool:
    if not recs or [r.canvas_idx for r in recs] != list(range(len(recs))):
        return False
    last = recs[-1]
    if last.error is not None or last.stop_pos is not None:
        return True
    return sum(len(r.committed_ids) for r in recs) >= max_tokens


def join(records: Sequence[CdCanvasRecord]) -> Joined:
    """Order by ``canvas_idx``. A duplicated index is recorded in ``duplicate_idx`` (never silently
    resolved); a gap leaves ``contiguous`` false."""
    by_idx: dict[int, CdCanvasRecord] = {}
    dup: set[int] = set()
    for r in records:
        idx = int(r.canvas_idx)
        if idx in by_idx:
            dup.add(idx)
        else:
            by_idx[idx] = r
    return Joined(tuple(by_idx[i] for i in sorted(by_idx)), tuple(sorted(dup)))


class EngineLog:
    """Incremental reader of the engine log, from the byte offset at construction."""

    def __init__(self, path: Path | str, *, from_start: bool = False) -> None:
        self.path = Path(path)
        self._offset = 0 if from_start or not self.path.exists() else self.path.stat().st_size
        self._partial = b""
        self._by_tag: dict[str, list[CdCanvasRecord]] = {}
        #: lines carrying the marker that did not validate as ``CdCanvasRecord``
        self.malformed = 0

    def poll(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
        if not chunk:
            return
        self._offset += len(chunk)
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()  # an incomplete last line waits for its newline
        for raw in lines:
            line = raw.decode("utf-8", errors="replace")
            if RECORD_MARK not in line:
                continue
            payload = line.split(RECORD_MARK, 1)[1]
            try:
                rec = CdCanvasRecord.model_validate_json(payload)
            except ValidationError:
                self.malformed += 1
                continue
            if rec.record_tag is not None:
                self._by_tag.setdefault(rec.record_tag, []).append(rec)

    def records(self, tag: str) -> list[CdCanvasRecord]:
        return list(self._by_tag.get(tag, ()))

    def wait(
        self, tag: str, *, max_tokens: int, timeout_s: float, poll_s: float = 0.02
    ) -> Joined | None:
        """The tag's joined canvases once complete; whatever arrived by the deadline otherwise
        (``None`` when nothing did)."""
        deadline = time.monotonic() + timeout_s
        while True:
            self.poll()
            joined = join(self._by_tag.get(tag, ()))
            if is_complete(joined.records, max_tokens):
                return joined
            if time.monotonic() >= deadline:
                return joined if joined.records else None
            time.sleep(poll_s)
