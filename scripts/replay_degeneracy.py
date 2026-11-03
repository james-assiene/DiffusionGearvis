#!/usr/bin/env python3
"""Replay logged thought commits through the T-0142 degeneracy detector on CPU (T-0142 A2; the T-0127 pattern).

Usage: ``uv run python scripts/replay_degeneracy.py <log-dir>... [--segment {thought,answer}] [--expect N]``

T-0142b: ``--segment answer`` judges the requests' ``answer`` commits instead, accumulated per attempt the same way,
with ``min_repeats=MIN_REPEATS_ANSWER`` (the session's answer guard); everything below then reads "answer" for
"thought". ``--segment thought`` is the default and prints exactly what the T-0142 script printed.

A log dir is the A/B layout: ``<log-dir>/trials.jsonl`` and, per trial, ``<log-dir>/sessions/<session_log>``. A
request is an ``llm.request`` record; its thought text is the ``llm.commit`` records with that ``attempt_id`` and
``segment == "thought"``, in log order. For every request with at least one thought commit, ``degenerate`` judges the
concatenation after each commit, as the session's guard does, and one line is printed::

    tripped attempt=<attempt_id> commit=<k> of=<n> kind=<kind> repeats=<r> phase=<phase> task=<task> repeat=<i>
    clean attempt=<attempt_id> of=<n> phase=<phase> task=<task> repeat=<i>

``commit`` is the 1-based ordinal of the thought commit at which the detector first trips, ``of`` the request's
thought commits. Then one totals line per directory and one over all of them.

Exit status: without ``--expect``, 0 iff no request tripped; with ``--expect N``, 0 iff exactly N tripped over all the
directories; 1 otherwise; 2 on a usage error, a missing file or malformed JSON (the file is named). Read-only, no
network, no GPU.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.conductor.degeneracy import MIN_REPEATS, MIN_REPEATS_ANSWER, Trip, degenerate

#: T-0142b: the segment judged, and its n-gram threshold (the session guard's)
SEGMENTS = {"thought": MIN_REPEATS, "answer": MIN_REPEATS_ANSWER}


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


@dataclass
class Request:
    phase: str
    attempt_id: str
    thoughts: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)


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
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"malformed JSON in {path} line {n}: {exc.msg}") from None
        if not isinstance(record, dict):
            raise ReplayError(f"malformed record in {path} line {n}: not an object")
        out.append(record)
    return out


def requests_of(path: Path) -> list[Request]:
    """The session's ``llm.request`` records in log order, each with its thought and answer commits' text (by
    ``attempt_id``, each segment apart)."""
    requests: list[Request] = []
    by_attempt: dict[str, Request] = {}
    for record in read_jsonl(path):
        kind = record.get("kind")
        data = record.get("data") or {}
        if not isinstance(data, dict):
            raise ReplayError(f"malformed record in {path}: data is not an object")
        if kind == "llm.request":
            req = Request(phase=str(data.get("phase")), attempt_id=str(data.get("attempt_id")))
            requests.append(req)
            by_attempt[req.attempt_id] = req
        elif kind == "llm.commit" and data.get("segment") in SEGMENTS:
            owner = by_attempt.get(str(data.get("attempt_id")))
            if owner is not None:
                texts = owner.thoughts if data.get("segment") == "thought" else owner.answers
                texts.append(str(data.get("text", "")))
    return requests


def first_trip(
    commits: Sequence[str], *, min_repeats: int = MIN_REPEATS
) -> tuple[int, Trip] | None:
    """The 1-based commit at which the cumulative text first trips, and the trip; ``None``: never."""
    text = ""
    for k, t in enumerate(commits, 1):
        text += t
        trip = degenerate(text, min_repeats=min_repeats)
        if trip is not None:
            return k, trip
    return None


def replay(run: Path, segment: str = "thought") -> tuple[int, int]:
    """Print one line per request with ``segment`` commits in ``run``; return (tripped, clean)."""
    trials_path = run / "trials.jsonl"
    if not trials_path.is_file():
        raise ReplayError(f"missing file: {trials_path}")
    tripped = clean = 0
    for trial in read_jsonl(trials_path):
        try:
            session_log = trial["session_log"]
        except KeyError as exc:
            raise ReplayError(f"malformed trial in {trials_path}: missing {exc}") from None
        session_path = run / "sessions" / Path(str(session_log)).name
        if not session_path.is_file():
            raise ReplayError(f"missing file: {session_path}")
        where = f"task={trial.get('task')} repeat={trial.get('repeat')}"
        if trial.get("arm") is not None:
            where += f" arm={trial.get('arm')}"
        for req in requests_of(session_path):
            commits = req.thoughts if segment == "thought" else req.answers
            if not commits:
                continue
            tail = f"phase={req.phase} {where}"
            found = first_trip(commits, min_repeats=SEGMENTS[segment])
            if found is None:
                clean += 1
                print(f"clean attempt={req.attempt_id} of={len(commits)} {tail}")
                continue
            tripped += 1
            k, trip = found
            print(
                f"tripped attempt={req.attempt_id} commit={k} of={len(commits)} "
                f"kind={trip.kind} repeats={trip.repeats} {tail}"
            )
    return tripped, clean


USAGE = (
    "usage: uv run python scripts/replay_degeneracy.py <log-dir>... "
    "[--segment {thought,answer}] [--expect N]"
)


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    expect: int | None = None
    segment = "thought"
    if "--segment" in args:
        i = args.index("--segment")
        if i + 1 >= len(args) or args[i + 1] not in SEGMENTS:
            print(USAGE, file=sys.stderr)
            return 2
        segment = args[i + 1]
        del args[i : i + 2]
    if "--expect" in args:
        i = args.index("--expect")
        try:
            expect = int(args[i + 1])
        except (IndexError, ValueError):
            print(USAGE, file=sys.stderr)
            return 2
        del args[i : i + 2]
    if not args or any(a.startswith("--") for a in args):
        print(USAGE, file=sys.stderr)
        return 2
    total_tripped = total_clean = 0
    try:
        for name in args:
            run = Path(name)
            if not run.is_dir():
                raise ReplayError(f"log directory not found or unreadable: {run}")
            tripped, clean = replay(run, segment)
            print(f"total {run.name}: {tripped} tripped, {clean} clean")
            total_tripped += tripped
            total_clean += clean
    except ReplayError as exc:
        print(f"replay_degeneracy: {exc}", file=sys.stderr)
        return 2
    print(f"total: {total_tripped} tripped, {total_clean} clean over {len(args)} directories")
    if expect is not None:
        if total_tripped != expect:
            print(f"expected {expect} tripped; got {total_tripped}", file=sys.stderr)
            return 1
        return 0
    return 1 if total_tripped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
