#!/usr/bin/env python3
"""Replay logged live turns through the T-0207 completion check on CPU (T-0207 A1.5).

Usage: ``uv run python scripts/replay_completion_check.py <session.jsonl>... [--json]``

One row per live turn whose final request ended with a non-empty answer and no call, after at least one executed
browser action this turn (``click_at``, ``type_at``, ``press_key``, ``scroll`` or ``wait``; a search does not count,
and a later search does not cancel an earlier action). The final request is the turn's last ``llm.request`` that is
neither a ``prefill`` nor a ``completion_check`` (T-0207 A2-R3; for a T-0172 re-asked step, the ``announce_reask``
request). Its commits are the ``llm.commit`` records with its ``attempt_id``; the answer is their ``answer`` segments
joined, the call any ``tool`` segment.

Columns: ``file, turn, seq, step, report_step, reasked_this_step, vision_source, would_check,
last_action_unchanged, label_key, thought_present, claim_match, answer``.

- ``would_check``: A2.1's trigger as read from the log: vision source ``browser``, or ``shared`` with the final
  request's ``label_key`` a browser key (``updated``, ``unchanged``, ``now``; T-0208 A2.6), a ``tool_step`` request, no
  T-0172 re-ask on that step, the request ended cleanly, a non-empty answer. Ended cleanly: an ``llm.done`` with the
  request's ``request_id`` is logged after it (the session writes one only when a step ends cleanly; an abort or a
  T-0142/T-0142b trip writes none), and no degeneracy note (``NOT_CLEAN_NOTES``) for it is logged after it.
- ``last_action_unchanged``: the last executed browser action's ``resolved.frame_after`` equals the one before it
  (False when there is only one).
- ``label_key``: the T-0201 B3 key of the request's ``live_label`` (``conductor.session.screen_reminder_key``;
  ``unknown`` for an unrecognised text, null when the request has no label).
- ``claim_match``: ``CLAIM`` below, reported only, never a trigger.
- ``answer``: cut to 120 characters.

Body check, per would-check row: the logged prompt is rebuilt (a redacted image becomes a placeholder
``ImagePart`` that keeps its ``phash``), the check prompt is ``completion_check.check_prompt`` over it with the answer
and a placeholder frame keeping the last action's ``frame_after``, and both go through ``VllmClient.request_body``
(pure, no socket). The check body's messages minus the last two must equal the step body's messages with the
continuation extended by the answer (minus the continuation itself), and its last message must be the frame, then
the instruction.

Exit status: 0 success; 1 a body check failed; 2 a missing or unreadable file, a line that is not a JSON object, or
an input with no row. Read-only, no network, no GPU.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_astra.conductor import session
from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import Prompt
from local_astra.contracts.perception import Frame, Source
from local_astra.toolcall import completion_check

BROWSER_ACTIONS = frozenset({"click_at", "type_at", "press_key", "scroll", "wait"})
ANSWER_CUT = 120
#: Notes logged after a request that did not end cleanly (a T-0142/T-0142b trip, a trip after the one retry): no
#: check. A retry's own note is logged before the retried request, so it is not here; the ``llm.done`` rule covers
#: every unclean end, these notes are a second reading.
NOT_CLEAN_NOTES = (
    session.DEGENERATE_MESSAGE,
    session.DEGENERATE_ANSWER_MESSAGE,
    session.DEGENERATE_RETRIED_MESSAGE,
)
#: A claimed completed action (pinned; reported only, never a trigger).
CLAIM = re.compile(
    r"\bI(?:'ve|\s+have)\s+(?:now\s+|successfully\s+|already\s+)?"
    r"(?:added|purchased|bought|ordered|placed|completed|submitted|booked|paid|checked\s+out|proceeded)\b"
    r"|\bproceeded\s+to\s+(?:the\s+)?(?:checkout|payment|purchase)\b"
    r"|\b(?:order|purchase|payment|booking)\s+(?:has\s+been|was|is)\s+"
    r"(?:placed|completed?|confirmed|made|submitted)\b",
    re.IGNORECASE,
)


class ReplayError(Exception):
    """A missing or malformed input; the message names the file."""


def records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReplayError(f"no such file: {path}")
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


def _data(rec: dict[str, Any]) -> dict[str, Any]:
    data = rec.get("data")
    return data if isinstance(data, dict) else {}


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


def label_key(rec: dict[str, Any]) -> str | None:
    label = _data(rec).get("live_label")
    if not isinstance(label, dict):
        return None
    text = label.get("text")
    key = session.screen_reminder_key(text) if isinstance(text, str) else None
    return key if key is not None else "unknown"


def _cut(text: str) -> str:
    return text if len(text) <= ANSWER_CUT else text[:ANSWER_CUT] + "…"


def turns(recs: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Records by turn id, in log order (dict order is first appearance)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:
        turn = rec.get("turn_id")
        if isinstance(turn, str) and turn:
            out.setdefault(turn, []).append(rec)
    return out


def turn_row(turn: str, recs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    requests = [
        r
        for r in recs
        if r.get("kind") == "llm.request"
        and _data(r).get("phase") not in ("prefill", "completion_check")
    ]
    if not requests:
        return None
    final = requests[-1]
    data = _data(final)
    executed = [
        _data(r)
        for r in recs
        if r.get("kind") == "tool.call"
        and _data(r).get("status") == "executed"
        and _data(r).get("name") in BROWSER_ACTIONS
    ]
    if not executed:
        return None
    aid = data.get("attempt_id")
    commits = [
        _data(r)
        for r in recs
        if r.get("kind") == "llm.commit" and _data(r).get("attempt_id") == aid
    ]
    if any(c.get("segment") == "tool" for c in commits):
        return None
    answer = "".join(
        str(c.get("text") or "") for c in commits if c.get("segment") == "answer"
    ).strip()
    if not answer:
        return None
    step = data.get("step")
    reasked = any(
        _data(r).get("phase") == "announce_reask" and _data(r).get("step") == step for r in requests
    )
    rid = data.get("request_id")
    final_seq = final.get("seq", 0)
    after = [r for r in recs if r.get("seq", 0) > final_seq]
    done = any(r.get("kind") == "llm.done" and _data(r).get("request_id") == rid for r in after)
    noted = any(
        r.get("kind") == "note"
        and _data(r).get("what") in NOT_CLEAN_NOTES
        and _data(r).get("request_id") in (None, rid)
        for r in after
    )
    unclean = not done or noted
    vision = None
    for r in recs:
        if r.get("kind") == "turn.record":
            extras = _data(r).get("extras")
            if isinstance(extras, dict) and isinstance(extras.get("vision_source"), str):
                vision = extras["vision_source"]
    frames = [(e.get("resolved") or {}).get("frame_after") for e in executed]
    unchanged = len(frames) >= 2 and frames[-1] is not None and frames[-1] == frames[-2]
    thought = any(
        c.get("segment") == "thought" and str(c.get("text") or "").strip() for c in commits
    )
    would = (
        (
            vision == "browser"
            or (vision == "shared" and label_key(final) in ("updated", "unchanged", "now"))
        )
        and data.get("phase") == "tool_step"
        and not reasked
        and not unclean
        and bool(answer)
    )
    return {
        "turn": turn,
        "seq": final.get("seq"),
        "step": step,
        "report_step": data.get("report_fact") is True,
        "reasked_this_step": reasked,
        "vision_source": vision,
        "would_check": would,
        "last_action_unchanged": unchanged,
        "label_key": label_key(final),
        "thought_present": thought,
        "claim_match": CLAIM.search(answer) is not None,
        "answer": _cut(answer),
        # private, for the body check; dropped before output
        "_final": final,
        "_full_answer": answer,
        "_frame_after": frames[-1],
    }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def body_check(row: dict[str, Any], client: VllmClient) -> str | None:
    """``None`` when the check body is the step body plus the answer and the frame + instruction turn."""
    data = _data(row["_final"])
    try:
        sent = rebuild(data.get("prompt"))
    except (ValidationError, TypeError, AttributeError) as exc:
        return f"the logged prompt does not rebuild: {type(exc).__name__}"
    said = row["_full_answer"]
    phash = row["_frame_after"] or "replay-placeholder"
    frame = Frame(
        source=Source.WINDOW,
        ts_ms=0,
        width=1280,
        height=800,
        jpeg=b"\xff\xd8\xff\xe0fresh:" + str(phash).encode(),
        phash=str(phash),
    )
    report = bool(row["report_step"])
    check = completion_check.check_prompt(sent, said, frame, report=report)
    rid = str(data.get("request_id") or "replay")
    base = client.request_body(sent, rid)["messages"]
    got = client.request_body(check, rid)["messages"]
    if len(got) < 2:
        return "the check body has fewer than two messages"
    if sent.continue_final_message:
        want_head = base[:-1]
        want_said = _text(base[-1]["content"]) + said
    else:
        want_head = base
        want_said = said
    if got[:-2] != want_head:
        return "the check body's messages minus the last two differ from the step's"
    if got[-2].get("role") != "assistant" or _text(got[-2]["content"]) != want_said:
        return "the check body's assistant message is not the continuation extended by the answer"
    last = got[-1]
    parts = last.get("content")
    instruction = (
        completion_check.CHECK_INSTRUCTION_REPORT if report else completion_check.CHECK_INSTRUCTION
    )
    if (
        last.get("role") != "user"
        or not isinstance(parts, list)
        or len(parts) != 2
        or parts[0].get("type") != "image_url"
        or parts[1].get("text") != instruction
    ):
        return "the check body's last message is not the frame then the instruction"
    return None


def replay(paths: Sequence[str]) -> list[dict[str, Any]]:
    client = VllmClient()
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        found = []
        for turn, recs in turns(records(path)).items():
            row = turn_row(turn, recs)
            if row is not None:
                found.append(row)
        if not found:
            raise ReplayError(f"no row in {path}")
        found.sort(key=lambda r: r["seq"] if isinstance(r["seq"], int) else 0)
        for row in found:
            why = body_check(row, client) if row["would_check"] else None
            out = {"file": str(path)}
            out.update({k: v for k, v in row.items() if not k.startswith("_")})
            rows.append({"row": out, "why": why})
    return rows


def print_text(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "# turn seq step report_step reasked_this_step vision_source would_check "
        "last_action_unchanged label_key thought_present claim_match body | answer"
    )
    current = None
    for item in rows:
        r = item["row"]
        if r["file"] != current:
            current = r["file"]
            print(f"# {current}")
        body = "-" if not r["would_check"] else ("ok" if item["why"] is None else "mismatch")
        print(
            f"{r['turn']} {r['seq']} {r['step']} {str(r['report_step']).lower()} "
            f"{str(r['reasked_this_step']).lower()} {r['vision_source'] or '-'} "
            f"{str(r['would_check']).lower()} {str(r['last_action_unchanged']).lower()} "
            f"{r['label_key'] or '-'} {str(r['thought_present']).lower()} "
            f"{str(r['claim_match']).lower()} {body} | {r['answer']}"
        )
        if item["why"] is not None:
            print(f"#   why: {item['why']}")
    plain = [i["row"] for i in rows]
    bad = sum(1 for i in rows if i["why"] is not None)

    def count(key: str) -> int:
        return sum(1 for r in plain if r[key] is True)

    print(
        f"# total: {len(plain)} rows; would_check {count('would_check')}; "
        f"last_action_unchanged {count('last_action_unchanged')}; report_step {count('report_step')}; "
        f"reasked_this_step {count('reasked_this_step')}; thought_present {count('thought_present')}; "
        f"claim_match {count('claim_match')}; body checks failed {bad}"
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", help="session .jsonl files")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    try:
        rows = replay(args.paths)
    except ReplayError as exc:
        print(f"replay_completion_check: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"rows": [i["row"] for i in rows]}))
    else:
        print_text(rows)
    failed = [i for i in rows if i["why"] is not None]
    for i in failed:
        print(
            f"replay_completion_check: body check failed at seq {i['row']['seq']}: {i['why']}",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
