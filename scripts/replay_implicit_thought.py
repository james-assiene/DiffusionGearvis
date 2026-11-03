#!/usr/bin/env python3
"""List implicit thoughts and pre-fix orphan-close splices in session logs on CPU (T-0170 criterion 6).

Usage: ``uv run python scripts/replay_implicit_thought.py <session.jsonl|dir>... [--json]``

A directory is read for every ``*.jsonl`` below it, in sorted order. Two listings:

- ``implicit``: every ``llm.commit`` record with ``implicit_thought: true`` (logged since T-0170: a response that
  opened no thought and whose first channel marker was an orphan ``<channel|>``);
- ``splices``: for logs from before T-0170, every request attempt (``request_id``, ``attempt_id``) whose first
  non-``tool`` commit is ``answer`` and whose answer text (its ``answer`` commits concatenated in log order) contains
  ``[.!?][A-Z]``: the trace an orphan ``<channel|>`` leaves once ``drop_special_literals`` has deleted it ("in one
  sentence.The Eiffel Tower"). ``seq`` is the record of the commit holding the capital letter of the first match.

Each entry carries ``path``, ``seq``, ``request_id`` and ``attempt_id`` (``excerpt`` too for a splice). Text mode
prints one line per entry; ``--json`` prints ``{"implicit": [...], "splices": [...]}``.

Exit status: 0 on success; 2 on a usage error or an unreadable input (a missing file, a file that is not UTF-8, a
line that is not a JSON object), the file named on stderr. Read-only, no network, no GPU.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPLICE = re.compile(r"[.!?][A-Z]")
EXCERPT = 40


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


@dataclass
class Attempt:
    path: str
    request_id: str
    attempt_id: str | None
    first_segment: str | None = None
    #: (seq, text) of each answer commit, in log order
    answers: list[tuple[int, str]] = field(default_factory=list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ReplayError(f"cannot read {path}: {exc.strerror or exc}") from None
    except UnicodeDecodeError as exc:
        raise ReplayError(f"malformed text in {path}: not UTF-8 at byte {exc.start}") from None
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"malformed JSON in {path} line {n}: {exc.msg}") from None
        if not isinstance(row, dict):
            raise ReplayError(f"malformed record in {path} line {n}: not a JSON object")
        out.append(row)
    return out


def session_files(arg: str) -> list[Path]:
    path = Path(arg)
    if path.is_dir():
        return sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    if not path.is_file():
        raise ReplayError(f"cannot read {path}: no such file or directory")
    return [path]


def scan(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    implicit: list[dict[str, Any]] = []
    attempts: dict[tuple[str, str | None], Attempt] = {}
    for row in read_jsonl(path):
        if row.get("kind") != "llm.commit":
            continue
        data = row.get("data")
        if not isinstance(data, dict):
            continue
        seq = row.get("seq")
        request_id = str(data.get("request_id"))
        attempt_id = data.get("attempt_id")
        segment = data.get("segment")
        text = data.get("text")
        text = text if isinstance(text, str) else ""
        if data.get("implicit_thought") is True:
            implicit.append(
                {"path": str(path), "seq": seq, "request_id": request_id, "attempt_id": attempt_id}
            )
        key = (request_id, attempt_id)
        attempt = attempts.setdefault(key, Attempt(str(path), request_id, attempt_id))
        if segment == "tool":
            continue
        if attempt.first_segment is None:
            attempt.first_segment = segment
        if segment == "answer" and isinstance(seq, int):
            attempt.answers.append((seq, text))
    splices = [hit for a in attempts.values() if (hit := splice(a)) is not None]
    return implicit, splices


def splice(attempt: Attempt) -> dict[str, Any] | None:
    if attempt.first_segment != "answer":
        return None
    whole = "".join(text for _seq, text in attempt.answers)
    found = SPLICE.search(whole)
    if found is None:
        return None
    at = found.end() - 1  # the capital letter
    offset = 0
    seq = attempt.answers[0][0]
    for s, text in attempt.answers:
        if at < offset + len(text):
            seq = s
            break
        offset += len(text)
    return {
        "path": attempt.path,
        "seq": seq,
        "request_id": attempt.request_id,
        "attempt_id": attempt.attempt_id,
        "excerpt": whole[max(0, at - EXCERPT) : at + EXCERPT],
    }


def main(argv: Sequence[str]) -> int:
    args = [a for a in argv if a != "--json"]
    as_json = len(args) != len(argv)
    if not args or any(a.startswith("-") for a in args):
        print(
            "usage: replay_implicit_thought.py <session.jsonl|dir>... [--json]",
            file=sys.stderr,
        )
        return 2
    implicit: list[dict[str, Any]] = []
    splices: list[dict[str, Any]] = []
    try:
        for arg in args:
            for path in session_files(arg):
                found_implicit, found_splices = scan(path)
                implicit += found_implicit
                splices += found_splices
    except ReplayError as exc:
        print(f"replay_implicit_thought: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps({"implicit": implicit, "splices": splices}, ensure_ascii=False))
        return 0
    for item in implicit:
        print(
            f"implicit {item['path']} seq {item['seq']} request {item['request_id']} "
            f"attempt {item['attempt_id']}"
        )
    for item in splices:
        print(
            f"splice {item['path']} seq {item['seq']} request {item['request_id']} "
            f"attempt {item['attempt_id']}: {item['excerpt']!r}"
        )
    print(f"{len(implicit)} implicit thought commit(s), {len(splices)} splice(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
