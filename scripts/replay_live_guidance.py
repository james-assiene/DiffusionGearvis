#!/usr/bin/env python3
"""Replay logged live requests through the T-0211 v4 guidance on CPU (T-0211 B7).

Usage: ``uv run python scripts/replay_live_guidance.py <session.jsonl>... [--to v4] [--json]``

Per ``llm.request`` record of every phase, in log order, file after file: the logged prompt is rebuilt as a
``Prompt`` the way ``scripts/replay_screen_reminder.py`` does (a redacted image becomes a placeholder ``ImagePart``
that keeps its ``phash``); the logged guidance (``LIVE_TOOL_GUIDANCE_V3``, with or without the wait sentence after
it) is found in the system message's text; and two wire bodies are built through ``VllmClient.request_body`` (pure,
no socket): BASE (the request as logged) and TO (v3's text replaced by ``--to``'s, the wait suffix kept). A row is
``ok`` iff the system text contains neither ``S_SHARED_GUIDANCE`` nor ``S_SOURCE_GUIDANCE`` (a log that already
carries v4 is a mismatch, B-R7), contains v3 exactly once, TO's system text is BASE's with
``" " + S_SHARED_GUIDANCE + " " + S_SOURCE_GUIDANCE`` inserted at the end of the guidance span (before the wait
sentence), every other byte of the two bodies is equal, and the request's system text is the same as the first
request's of its file.

Text output: a header line, then per file one ``# <file>`` line and one row per request::

    {seq} {phase} {wait|-} {system chars} {delta chars} {ok|mismatch}

``wait`` says the wait sentence follows the guidance; ``delta chars`` is TO's system text length minus BASE's.
``--json`` prints one object ``{"rows": [...]}`` with the same values (``ok`` a bool, ``file`` and, when not ok,
``why``).

Exit status: 0 every row ok; 1 any row not ok; 2 a missing or unreadable file, a line that is not a JSON object, or
no ``llm.request``. Read-only, no network, no GPU.
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

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import Message, Prompt, TextPart

#: the logged version the replay looks for (the live default when the logs were written)
LOGGED = pb.LIVE_TOOL_GUIDANCE_V3
#: what ``--to v4`` must add at the end of the guidance span (B7)
DELTA = " " + pb.S_SHARED_GUIDANCE + " " + pb.S_SOURCE_GUIDANCE
TO_CHOICES = ("v4",)


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


def requests(recs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if r.get("kind") == "llm.request" and isinstance(r.get("data"), dict)]


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


def _system_text(body: dict[str, Any]) -> str:
    content = body["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
    )


def with_guidance(prompt: Prompt, to_text: str) -> Prompt:
    """``prompt`` with the one occurrence of v3 in its system message replaced by ``to_text`` (the part holding it)."""
    system = prompt.messages[0]
    parts = tuple(
        TextPart(text=p.text.replace(LOGGED, to_text, 1))
        if isinstance(p, TextPart) and LOGGED in p.text
        else p
        for p in system.content
    )
    message = Message(role=system.role, content=parts)
    return prompt.model_copy(update={"messages": (message, *prompt.messages[1:])})


def check(
    rec: dict[str, Any], client: VllmClient, to_text: str, file_text: str | None
) -> dict[str, Any]:
    data = rec["data"]
    row: dict[str, Any] = {
        "seq": rec.get("seq"),
        "phase": str(data.get("phase")),
        "wait": False,
        "system_chars": 0,
        "delta_chars": 0,
        "ok": False,
    }
    try:
        prompt = rebuild(data.get("prompt"))
    except (ValidationError, TypeError, AttributeError) as exc:
        # a record this replay cannot rebuild is reported as a row, never skipped
        row["why"] = f"the logged prompt does not rebuild: {type(exc).__name__}"
        return row
    if not prompt.messages or prompt.messages[0].role != "system":
        row["why"] = "the logged prompt has no system message first"
        return row
    text = prompt.messages[0].text()
    row["system_chars"] = len(text)
    if file_text is not None and text != file_text:
        row["why"] = "the system text differs from the first request's of this file"
        return row
    # v4 has v3 as its prefix: a log that already carries v4 would pass the count below and get the delta twice
    already = [
        name
        for name, part in (
            ("S_SHARED", pb.S_SHARED_GUIDANCE),
            ("S_SOURCE", pb.S_SOURCE_GUIDANCE),
        )
        if part in text
    ]
    if already:
        row["why"] = (
            f"the system text already contains the v4 delta ({' and '.join(already)}): "
            "the log is not a v3 log"
        )
        return row
    count = text.count(LOGGED)
    if count != 1:
        row["why"] = f"the system text contains the logged guidance (v3) {count} times, not once"
        return row
    if not any(isinstance(p, TextPart) and LOGGED in p.text for p in prompt.messages[0].content):
        row["why"] = "the logged guidance spans two system parts"
        return row
    end = text.index(LOGGED) + len(LOGGED)
    row["wait"] = text[end:].startswith(" " + pb.WAIT_TOOL_GUIDANCE)
    rid = str(data.get("request_id") or "replay")
    base_body = client.request_body(prompt, rid)
    to_body = client.request_body(with_guidance(prompt, to_text), rid)
    base_text, to_sys = _system_text(base_body), _system_text(to_body)
    row["delta_chars"] = len(to_sys) - len(base_text)
    if base_text != text:
        row["why"] = "BASE's wire system text is not the logged system text"
        return row
    if to_sys != text[:end] + DELTA + text[end:]:
        row["why"] = (
            "TO's system text is not BASE's with ' ' + S_SHARED + ' ' + S_SOURCE at the end of the guidance"
        )
        return row
    masked = copy.deepcopy(to_body)
    masked["messages"][0] = base_body["messages"][0]
    if masked != base_body:
        row["why"] = "the TO body differs from BASE outside the system text"
        return row
    row["ok"] = True
    return row


def replay(paths: Sequence[str], to_text: str) -> list[dict[str, Any]]:
    client = VllmClient()
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise ReplayError(f"no such file: {path}")
        file_text: str | None = None
        for n, rec in enumerate(requests(records(path))):
            row = check(rec, client, to_text, file_text)
            if n == 0:
                file_text = _first_text(rec)
            row["file"] = str(path)
            rows.append(row)
    return rows


def _first_text(rec: dict[str, Any]) -> str | None:
    """The first request's logged system text, the reference for the per-file rule (``None`` if unreadable)."""
    try:
        prompt = rebuild(rec["data"].get("prompt"))
    except (ValidationError, TypeError, AttributeError):
        return None
    return prompt.messages[0].text() if prompt.messages else None


def print_text(rows: Sequence[dict[str, Any]], to: str) -> None:
    print(f"# to={to}; seq phase wait system_chars delta_chars ok")
    current = None
    for r in rows:
        if r["file"] != current:
            current = r["file"]
            print(f"# {current}")
        print(
            f"{r['seq']} {r['phase']} {'wait' if r['wait'] else '-'} {r['system_chars']} {r['delta_chars']} "
            f"{'ok' if r['ok'] else 'mismatch'}"
        )
        if not r["ok"]:
            print(f"#   why: {r.get('why')}")
    bad = sum(1 for r in rows if not r["ok"])
    print(f"# {len(rows)} llm.request records; {len(rows) - bad} ok; {bad} not ok")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", help="session .jsonl files")
    parser.add_argument(
        "--to", choices=TO_CHOICES, default="v4", help="the guidance version to replay to"
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    try:
        rows = replay(args.paths, pb.LIVE_GUIDANCE[args.to])
    except ReplayError as exc:
        print(f"replay_live_guidance: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("replay_live_guidance: no llm.request", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"rows": rows}))
    else:
        print_text(rows, args.to)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
