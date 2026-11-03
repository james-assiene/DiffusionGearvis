"""Append-only ``turns.jsonl`` writer (docs/03 §7). Task T-0007.

One JSON line per *turn*, written on the turn-ending row rather than at process exit: a session
that dies mid-demo is exactly the session whose lines one wants, so the file is opened, written
and closed per record. ``astra bench`` reads the file back with ``TurnRecord.from_line``.
"""

from __future__ import annotations

from pathlib import Path

from local_astra.telemetry.turns_log import TurnRecord


class TurnsWriter:
    """``TurnsWriter(path).write(record)`` — append one line, create the directory if needed."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TurnRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_line() + "\n")
