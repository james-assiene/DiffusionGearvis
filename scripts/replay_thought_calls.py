#!/usr/bin/env python3
"""Replay a run directory's open-thought tool calls through the T-0135 recovery rule on CPU (T-0135 A2).

Usage: ``uv run python scripts/replay_thought_calls.py <run-dir> [--json]``

Reads ``<run-dir>/trials.jsonl`` and, per trial, ``<run-dir>/sessions/<session_log>``. For every ``generate`` or
``tool_step`` request (in log order) that kept no ``tool`` commit and whose last non-tool commit is a ``thought``
commit with ``thought_open: true``, the final open thought block (the trailing run of open ``thought`` commits,
concatenated in order; a closed thought is never stitched to a later open one) goes through
``thought_call.recover`` and, when that returns a tail, ``live.classify``. Each such request is a firing.

Text mode prints one line per firing (task, repeat, session file, request id, step, verdict status, name, arguments,
reason), then totals by task x status and the informational count of ReGround pass-2 (``reground``) responses of the
same shape, which are out of scope. ``--json`` prints one object: ``scanned``, ``firings``, ``totals``
(``{task: {status: count}}``) and ``pass2_same_shape`` (shape pinned in ``tests/unit/test_t0135_replay.py``).

Exit status: 0 on success; 2 on a usage error, an unreadable run directory, a missing file or malformed JSON (the
file is named). Read-only, no network, no GPU.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.toolcall import live
from local_astra.toolcall.thought_call import recover

LIVE_PHASES = ("generate", "tool_step")
PASS2_PHASE = "reground"


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


@dataclass
class Request:
    phase: str
    request_id: str
    step: int | None
    commits: list[dict[str, Any]] = field(default_factory=list)


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
    """The session's ``llm.request`` records in log order, each with its ``llm.commit`` data (by ``attempt_id``)."""
    requests: list[Request] = []
    by_attempt: dict[str, Request] = {}
    for record in read_jsonl(path):
        kind = record.get("kind")
        data = record.get("data") or {}
        if not isinstance(data, dict):
            raise ReplayError(f"malformed record in {path}: data is not an object")
        if kind == "llm.request":
            step = data.get("step")
            req = Request(
                phase=str(data.get("phase")),
                request_id=str(data.get("request_id")),
                step=step if isinstance(step, int) else None,
            )
            requests.append(req)
            by_attempt[str(data.get("attempt_id"))] = req
        elif kind == "llm.commit":
            owner = by_attempt.get(str(data.get("attempt_id")))
            if owner is not None:
                owner.commits.append(data)
    return requests


def open_thought_tail(commits: Sequence[dict[str, Any]]) -> str | None:
    """The block's recovered tail when the request has the firing shape (A2), else ``None``."""
    if any(c.get("segment") == "tool" for c in commits):
        return None
    if not commits:
        return None
    last = commits[-1]  # no tool commit, so the last commit is the last non-tool one
    if not (last.get("segment") == "thought" and last.get("thought_open") is True):
        return None
    i = len(commits)
    while (
        i > 0
        and commits[i - 1].get("segment") == "thought"
        and commits[i - 1].get("thought_open") is True
    ):
        i -= 1
    block = "".join(str(c.get("text", "")) for c in commits[i:])
    return recover(block)


def replay(run: Path) -> dict[str, Any]:
    if not run.is_dir():
        raise ReplayError(f"run directory not found or unreadable: {run}")
    trials_path = run / "trials.jsonl"
    if not trials_path.is_file():
        raise ReplayError(f"missing file: {trials_path}")
    scanned = 0
    pass2 = 0
    firings: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    for trial in read_jsonl(trials_path):
        try:
            task, repeat, session_log = trial["task"], trial["repeat"], trial["session_log"]
        except KeyError as exc:
            raise ReplayError(f"malformed trial in {trials_path}: missing {exc}") from None
        session = Path(str(session_log)).name
        session_path = run / "sessions" / session
        if not session_path.is_file():
            raise ReplayError(f"missing file: {session_path}")
        for req in requests_of(session_path):
            if req.phase == PASS2_PHASE:
                pass2 += int(open_thought_tail(req.commits) is not None)
                continue
            if req.phase not in LIVE_PHASES:
                continue
            scanned += 1
            tail = open_thought_tail(req.commits)
            if tail is None:
                continue
            verdict = live.classify(tail)
            firings.append(
                {
                    "task": task,
                    "repeat": repeat,
                    "session": session,
                    "request_id": req.request_id,
                    "step": 1 if req.phase == "generate" else req.step,
                    "status": verdict.status,
                    "name": verdict.name,
                    "arguments": verdict.arguments,
                    "reason": verdict.reason,
                }
            )
            by_status = totals.setdefault(str(task), {})
            by_status[verdict.status] = by_status.get(verdict.status, 0) + 1
    return {"scanned": scanned, "firings": firings, "totals": totals, "pass2_same_shape": pass2}


def print_text(doc: dict[str, Any]) -> None:
    for f in doc["firings"]:
        print(
            f"{f['task']} r{f['repeat']} {f['session']} {f['request_id']} step={f['step']}: "
            f"{f['status']} {f['name']} arguments={json.dumps(f['arguments'], sort_keys=True)} "
            f"reason={f['reason']}"
        )
    print(f"scanned {doc['scanned']} generate/tool_step requests; {len(doc['firings'])} fired")
    for task in sorted(doc["totals"]):
        counts = ", ".join(f"{s}={n}" for s, n in sorted(doc["totals"][task].items()))
        print(f"  {task}: {counts}")
    print(
        f"ReGround pass-2 responses of the same shape: {doc['pass2_same_shape']} "
        "(informational; out of scope for T-0135, never counted as firings)"
    )


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if len(args) != 1:
        print(
            "usage: uv run python scripts/replay_thought_calls.py <run-dir> [--json]",
            file=sys.stderr,
        )
        return 2
    try:
        doc = replay(Path(args[0]))
    except ReplayError as exc:
        print(f"replay_thought_calls: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(doc))
    else:
        print_text(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
