#!/usr/bin/env python3
"""Replay logged live step requests through the T-0201 screen reminder on CPU (T-0201 B9).

Usage: ``uv run python scripts/replay_screen_reminder.py <session.jsonl>... [--seq N ...] [--json]``

Per ``llm.request`` record whose ``phase`` is ``tool_step`` and which carries a ``live_label`` (the report step,
``report_fact: true``, is skipped: it never gets a reminder, B4), in log order: the logged prompt is rebuilt as a
``Prompt`` (a redacted image becomes a placeholder ``ImagePart`` that keeps its ``phash``), the reminder key is
derived from ``live_label.text`` by ``conductor.session.screen_reminder_key``, and two wire bodies are built through
``VllmClient.request_body`` (pure, no socket): BASE (the request as logged) and REMINDER (through
``loop.with_screen_reminder`` with ``SCREEN_REMINDERS[key]``). A row is ``ok`` iff the BASE continuation ends with a
tool response, REMINDER's ``messages[-1]["content"]`` is BASE's plus ``SCREEN_REMINDER_SEPARATOR + reminder``, and
every other byte of the two bodies is equal. A record logged with a ``screen_reminder`` key already carries one: its
BASE is the logged prompt without that suffix. An unknown label text is a row that is not ok.

Text output: a header line, then per file one ``# <file>`` line and one row per request::

    {seq} {step} {label key} {reminder key|-} {continuation chars} {delta chars} {ok|mismatch}

``label key`` names the label's shape (``now``, ``updated``, ``updated+outcome``, ``updated+extent``,
``updated+outcome+extent``, ``unchanged``, ``shared_screen``, ``shared_camera``, ``unknown``; an extent clause on
``now`` or ``unchanged`` adds ``+extent``); ``continuation chars`` is the logged final assistant text's length;
``delta chars`` is ``len(SEPARATOR + reminder)`` (0 when unknown). ``--json`` prints one object ``{"rows": [...]}``
with the same values (``reminder_key`` null when unknown, ``ok`` a bool, ``file`` and, when not ok, ``why``).

``--seq N ...`` keeps only those seqs. Exit status: 0 every row ok; 1 any row not ok; 2 a missing or unreadable
file, a line that is not a JSON object, or no labelled ``tool_step`` request. Read-only, no network, no GPU.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_astra.conductor import session
from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.toolcall import loop

RESPONSE_CLOSE = "<tool_response|>"


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


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


def labelled_steps(recs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in recs:
        data = rec.get("data")
        if rec.get("kind") != "llm.request" or not isinstance(data, dict):
            continue
        if data.get("phase") != "tool_step" or data.get("report_fact") is True:
            continue
        if isinstance(data.get("live_label"), dict):
            out.append(rec)
    return out


def _placeholder(part: dict[str, Any]) -> dict[str, Any]:
    """A redacted image part as an ``ImagePart`` tree: a placeholder data URL, its ``phash`` kept."""
    url = part.get("url")
    if isinstance(url, str):
        return part
    tag = hashlib.sha256(json.dumps(url, sort_keys=True).encode()).hexdigest()[:16]
    data = base64.b64encode(b"\xff\xd8\xff\xe0redacted:" + tag.encode()).decode()
    return {**part, "url": "data:image/jpeg;base64," + data}


def rebuild(tree: Any) -> Prompt:
    tree = copy.deepcopy(tree)
    for message in tree.get("messages", ()):
        message["content"] = [
            _placeholder(p) if isinstance(p, dict) and p.get("type") == "image_url" else p
            for p in message.get("content", ())
        ]
    return Prompt.model_validate(tree)


def label_key(text: str) -> str:
    key = session.screen_reminder_key(text)
    if key is None:
        return "unknown"
    if key in ("shared_screen", "shared_camera"):
        return key
    shape = key
    if session.OUTCOME_CLAUSE.split("{outcome}", 1)[0] in text:
        shape += "+outcome"
    if " Page beyond this screen: " in text:
        shape += "+extent"
    return shape


def _content(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def check(rec: dict[str, Any], client: VllmClient) -> dict[str, Any]:
    data = rec["data"]
    text = data["live_label"].get("text")
    key = session.screen_reminder_key(text) if isinstance(text, str) else None
    row: dict[str, Any] = {
        "seq": rec.get("seq"),
        "step": data.get("step"),
        "label_key": label_key(text) if isinstance(text, str) else "unknown",
        "reminder_key": key,
        "continuation_chars": 0,
        "delta_chars": 0,
        "ok": False,
    }
    try:
        prompt = rebuild(data.get("prompt"))
    except (ValidationError, TypeError, AttributeError) as exc:
        # a record this replay cannot rebuild is reported as a row, never skipped
        row["why"] = f"the logged prompt does not rebuild: {type(exc).__name__}"
        return row
    if not prompt.continue_final_message or not prompt.messages:
        row["why"] = "the logged prompt is not a continued step request"
        return row
    logged = prompt.messages[-1].text()
    row["continuation_chars"] = len(logged)
    if key is None:
        row["why"] = f"unknown label text: {text!r}"
        return row
    reminder = session.SCREEN_REMINDERS[key]
    suffix = loop.SCREEN_REMINDER_SEPARATOR + reminder
    row["delta_chars"] = len(suffix)
    logged_key = data.get("screen_reminder")
    if logged_key is not None:
        # a request logged with the reminder on: BASE is the logged prompt without it
        if logged_key != key or not logged.endswith(suffix):
            row["why"] = f"logged screen_reminder {logged_key!r} does not match the label's {key!r}"
            return row
        prompt = _strip(prompt, len(suffix))
    if not prompt.messages[-1].text().endswith(RESPONSE_CLOSE):
        row["why"] = "the continuation does not end with a tool response"
        return row
    rid = str(data.get("request_id") or "replay")
    base_body = client.request_body(prompt, rid)
    reminded = loop.with_screen_reminder(prompt, reminder)
    tip_body = client.request_body(reminded, rid)
    if _content(tip_body) != _content(base_body) + suffix:
        row["why"] = "messages[-1].content is not BASE's plus SEPARATOR + reminder"
        return row
    masked = copy.deepcopy(tip_body)
    masked["messages"][-1]["content"] = base_body["messages"][-1]["content"]
    if masked != base_body:
        row["why"] = "the REMINDER body differs from BASE outside messages[-1].content"
        return row
    row["ok"] = True
    return row


def _strip(prompt: Prompt, n: int) -> Prompt:
    text = prompt.messages[-1].text()[:-n]
    message = Message(role="assistant", content=(TextPart(text=text),))
    return prompt.model_copy(update={"messages": (*prompt.messages[:-1], message)})


def replay(paths: Sequence[str], seqs: set[int] | None) -> list[dict[str, Any]]:
    client = VllmClient()
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise ReplayError(f"no such file: {path}")
        for rec in labelled_steps(records(path)):
            if seqs is not None and rec.get("seq") not in seqs:
                continue
            row = check(rec, client)
            row["file"] = str(path)
            rows.append(row)
    return rows


def print_text(rows: Sequence[dict[str, Any]]) -> None:
    print("# seq step label_key reminder_key continuation_chars delta_chars ok")
    current = None
    for r in rows:
        if r["file"] != current:
            current = r["file"]
            print(f"# {current}")
        print(
            f"{r['seq']} {r['step']} {r['label_key']} {r['reminder_key'] or '-'} "
            f"{r['continuation_chars']} {r['delta_chars']} {'ok' if r['ok'] else 'mismatch'}"
        )
        if not r["ok"]:
            print(f"#   why: {r.get('why')}")
    bad = sum(1 for r in rows if not r["ok"])
    print(f"# {len(rows)} labelled tool_step requests; {len(rows) - bad} ok; {bad} not ok")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", help="session .jsonl files")
    parser.add_argument("--seq", type=int, nargs="+", help="only these record seqs")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    try:
        rows = replay(args.paths, set(args.seq) if args.seq else None)
    except ReplayError as exc:
        print(f"replay_screen_reminder: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("replay_screen_reminder: no labelled tool_step request", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"rows": rows}))
    else:
        print_text(rows)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
