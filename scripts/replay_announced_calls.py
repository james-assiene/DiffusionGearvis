#!/usr/bin/env python3
"""Replay logged sessions' no-call answers through the T-0172 announced-call rule on CPU (T-0172 A3).

Usage: ``uv run python scripts/replay_announced_calls.py <path>... [--json] [--rule tail|all]``

A ``<path>`` is a session ``.jsonl`` file, or a directory: every ``*.jsonl`` under it (recursively) whose parent
directory is named ``sessions``, in sorted order (the ``logs/**/sessions/*.jsonl`` shape; a run directory's
``trials.jsonl`` is not read).

Per ``llm.request`` whose ``phase`` is ``generate`` (step 1) or ``tool_step``: tools are on when its logged
``prompt.tools`` is a non-empty list; its commits are those with its ``attempt_id``; how it ended is the ``tool`` of
the ``machine.step`` record whose event is ``{"type": "llm.done", "request_id": <its id>}`` (absent: aborted). A
**no-call answer** is tools-on, kept no ``tool`` commit, ended with ``tool: null``, and has a non-empty joined answer
(its ``answer`` commits joined with no separator, stripped). Each gets one line, in log order::

    {file} {request_id} step={step} {hit|miss} verb={verb|-} {answer[:120] as JSON}

then one totals line::

    scanned {S} generate/tool_step requests; {N} no-call answers; hits: tail {T}, all {A}; announce-then-call {C}

``T``/``A`` count no-call answers hit by each mode whatever ``--rule`` is; ``C`` counts requests that ended with a
call whose joined answer the ``--rule`` mode hits. ``--json`` prints one object with the same numbers and the rows.

Exit status: 0; 2 on a usage error or malformed input (a missing path, a line that is not a JSON object), naming the
file. Read-only, no network, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.toolcall.announced_call import announces

PHASES = ("generate", "tool_step")
ANSWER_CUT = 120
_MISSING = object()


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


@dataclass
class Request:
    request_id: str
    step: int | None
    tools_on: bool
    tool_commit: bool = False
    answers: list[str] = field(default_factory=list)
    done_tool: Any = _MISSING

    @property
    def answer(self) -> str:
        return "".join(self.answers).strip()


def session_files(paths: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += sorted(
                f for f in p.rglob("*.jsonl") if f.is_file() and f.parent.name == "sessions"
            )
        elif p.is_file():
            out.append(p)
        else:
            raise ReplayError(f"no such file or directory: {p}")
    return out


def records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"cannot read {path}: {exc.strerror or exc}") from None
    except UnicodeDecodeError as exc:
        raise ReplayError(f"malformed text in {path}: not UTF-8 at byte {exc.start}") from None
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"malformed JSON in {path} line {n}: {exc.msg}") from None
        if not isinstance(rec, dict):
            raise ReplayError(f"malformed record in {path} line {n}: not a JSON object")
        out.append(rec)
    return out


def requests_of(path: Path) -> list[Request]:
    reqs: list[Request] = []
    by_attempt: dict[str, Request] = {}
    done: dict[str, Any] = {}
    for rec in records(path):
        kind = rec.get("kind")
        data = rec.get("data")
        if not isinstance(data, dict):
            continue
        if kind == "llm.request" and data.get("phase") in PHASES:
            prompt = data.get("prompt")
            tools = prompt.get("tools") if isinstance(prompt, dict) else None
            step = 1 if data.get("phase") == "generate" else data.get("step")
            req = Request(
                request_id=str(data.get("request_id")),
                step=step if isinstance(step, int) else None,
                tools_on=isinstance(tools, list) and len(tools) > 0,
            )
            reqs.append(req)
            by_attempt[str(data.get("attempt_id"))] = req
        elif kind == "llm.commit":
            owner = by_attempt.get(str(data.get("attempt_id")))
            if owner is None:
                continue
            if data.get("segment") == "tool":
                owner.tool_commit = True
            elif data.get("segment") == "answer":
                owner.answers.append(str(data.get("text", "")))
        elif kind == "machine.step":
            event = data.get("event")
            if isinstance(event, dict) and event.get("type") == "llm.done":
                done[str(event.get("request_id"))] = event.get("tool")
    for req in reqs:
        req.done_tool = done.get(req.request_id, _MISSING)
    return reqs


def replay(paths: Sequence[str], rule: str) -> dict[str, Any]:
    scanned = 0
    no_call = 0
    hits = {"tail": 0, "all": 0}
    then_call = 0
    rows: list[dict[str, Any]] = []
    for path in session_files(paths):
        for req in requests_of(path):
            scanned += 1
            answer = req.answer
            ended_with_call = req.done_tool is not _MISSING and req.done_tool is not None
            if ended_with_call:
                if answer and announces(answer, mode="all" if rule == "all" else "tail"):
                    then_call += 1
                continue
            if not (req.tools_on and not req.tool_commit and req.done_tool is None and answer):
                continue
            no_call += 1
            tail = announces(answer, mode="tail")
            every = announces(answer, mode="all")
            hits["tail"] += tail is not None
            hits["all"] += every is not None
            got = every if rule == "all" else tail
            rows.append(
                {
                    "file": str(path),
                    "request_id": req.request_id,
                    "step": req.step,
                    "verdict": "hit" if got is not None else "miss",
                    "verb": got.verb if got is not None else None,
                    "answer": answer[:ANSWER_CUT],
                }
            )
    return {
        "rule": rule,
        "scanned": scanned,
        "no_call_answers": no_call,
        "hits": hits,
        "announce_then_call": then_call,
        "rows": rows,
    }


def print_text(doc: dict[str, Any]) -> None:
    for r in doc["rows"]:
        print(
            f"{r['file']} {r['request_id']} step={r['step']} {r['verdict']} verb={r['verb'] or '-'} "
            f"{json.dumps(r['answer'], ensure_ascii=False)}"
        )
    print(
        f"scanned {doc['scanned']} generate/tool_step requests; {doc['no_call_answers']} no-call answers; "
        f"hits: tail {doc['hits']['tail']}, all {doc['hits']['all']}; "
        f"announce-then-call {doc['announce_then_call']}"
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="replay_announced_calls.py",
        description="Replay logged no-call answers through the T-0172 announced-call rule (CPU, read-only).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "paths", nargs="+", metavar="path", help="session .jsonl files or directories"
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument(
        "--rule", choices=("tail", "all"), default="tail", help="the rule's mode (default tail)"
    )
    args = parser.parse_args(list(argv))  # exits 2 on a usage error
    try:
        doc = replay(args.paths, args.rule)
    except ReplayError as exc:
        print(f"replay_announced_calls: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
    else:
        print_text(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
