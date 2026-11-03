"""Extract the T-0135 A2 replay fixture from the T-0132 run directory (read-only).

Usage: ``uv run python tests/fixtures/t0135/build_thought_tails.py <out.jsonl>``

The selection is the test-author's own census of ``logs/t0132-crop-budget-ab-2ccff41/`` (independent of
``local_astra/toolcall/thought_call.py``, which did not exist when this was written), cross-checked against the
planner's census in ``docs/tasks/T-0135.md`` before anything is written:

- ``firing``: every ``generate``/``tool_step`` request that kept no ``tool`` commit, whose last non-tool commit is a
  ``thought`` commit with ``thought_open: true``, and whose final open thought block (the trailing run of open
  ``thought`` commits, concatenated) has a last ``<|tool_call>`` whose remainder ends, after trailing whitespace, in
  ``<tool_call|>``. Must be 29 (S3 18, S1 7, S5 4), 28 ``valid`` + 1 ``repairable``.
- ``open_no_call``: the same shape with no call tag in the block (must be 3).
- ``pass2``: ReGround pass-2 (``reground``) requests of the firing shape (must be 10).
- ``reground_other``: the first 2 other ReGround requests, by file name (never counted).
- ``closed``: the first 6 ``generate``/``tool_step`` requests, by file name, that end in a closed thought or an answer
  with no ``tool`` commit.
- ``answer_call_prose``: the first 6, by file name, that kept an answer-channel ``tool`` commit and whose non-tool
  text names a v1 tool (prose about calls beside a real call).
- ``other_phase``: the first ``prefill`` request and both ``tool_repair`` requests (never scanned).

Each row keeps only: the source session file and the request's ``seq``, ``turn_id``, ``phase``, ``request_id``,
``attempt_id``, ``step`` and ``call_id``; its commits' ``seq``, ``canvas_idx``, ``segment``, ``text`` and
``thought_open``; the trial's ``task``, ``repeat``, ``success`` and ``tool_loop_end``; and, for ``firing`` rows, the
expected verdict (``live.classify`` of the tail at f3712c8; ``toolcall/live.py`` is frozen by T-0135 A3).
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

from local_astra.toolcall import live

LOG = Path("logs/t0132-crop-budget-ab-2ccff41")
TAG = "<|tool_call>"
CLOSE = "<tool_call|>"
TOOL_NAMES = re.compile(r"\b(click_at|type_at|scroll|press_key|google_search)\b")


def load() -> tuple[dict[str, dict], list[dict]]:
    trials = [json.loads(x) for x in (LOG / "trials.jsonl").read_text().splitlines() if x.strip()]
    by_file = {t["session_log"]: t for t in trials}
    requests: list[dict] = []
    for path in sorted(glob.glob(str(LOG / "sessions" / "*.jsonl"))):
        name = os.path.basename(path)
        by_attempt: dict[str, dict] = {}
        for line in Path(path).read_text().splitlines():
            rec = json.loads(line)
            data = rec["data"]
            if rec["kind"] == "llm.request":
                row = {
                    "source": name,
                    "seq": rec["seq"],
                    "turn_id": rec["turn_id"],
                    "phase": data["phase"],
                    "request_id": data["request_id"],
                    "attempt_id": data["attempt_id"],
                    "step": data.get("step"),
                    "call_id": data.get("call_id"),
                    "commits": [],
                }
                by_attempt[data["attempt_id"]] = row
                requests.append(row)
            elif rec["kind"] == "llm.commit":
                by_attempt[data["attempt_id"]]["commits"].append(
                    {
                        "seq": rec["seq"],
                        "canvas_idx": data["canvas_idx"],
                        "segment": data["segment"],
                        "text": data["text"],
                        "thought_open": data["thought_open"],
                    }
                )
    return by_file, requests


def shape(commits: list[dict]) -> tuple[str, str | None]:
    if any(c["segment"] == "tool" for c in commits):
        return "has_tool", None
    non_tool = [c for c in commits if c["segment"] != "tool"]
    if not non_tool:
        return "empty", None
    last = non_tool[-1]
    if not (last["segment"] == "thought" and last["thought_open"]):
        return "closed", None
    i = len(commits)
    while i > 0 and commits[i - 1]["segment"] == "thought" and commits[i - 1]["thought_open"]:
        i -= 1
    block = "".join(c["text"] for c in commits[i:])
    k = block.rfind(TAG)
    if k < 0:
        return "open_no_call", None
    tail = block[k:]
    return ("fire", tail) if tail.rstrip().endswith(CLOSE) else ("open_tag_no_close", None)


def main(out: Path) -> None:
    by_file, requests = load()
    picked: list[tuple[str, dict, dict | None]] = []
    counts: collections.Counter[str] = collections.Counter()
    for r in requests:
        kind, tail = shape(r["commits"])
        live_phase = r["phase"] in ("generate", "tool_step")
        if live_phase:
            counts["eligible"] += 1
        expect = None
        if live_phase and kind == "fire":
            assert tail is not None
            v = live.classify(tail)
            expect = {
                "tail": tail,
                "status": v.status,
                "name": v.name,
                "arguments": v.arguments,
                "reason": v.reason,
            }
            cls = "firing"
        elif live_phase and kind == "open_no_call":
            cls = "open_no_call"
        elif r["phase"] == "reground" and kind == "fire":
            cls = "pass2"
        elif r["phase"] == "reground":
            cls = "reground_other" if counts["reground_other"] < 2 else ""
        elif live_phase and kind == "closed":
            cls = "closed" if counts["closed"] < 6 else ""
        elif live_phase and kind == "has_tool":
            prose = " ".join(c["text"] for c in r["commits"] if c["segment"] != "tool")
            ok = TOOL_NAMES.search(prose) and counts["answer_call_prose"] < 6
            cls = "answer_call_prose" if ok else ""
        elif r["phase"] == "prefill":
            cls = "other_phase" if counts["prefill"] < 1 else ""
            counts["prefill"] += int(bool(cls))
        elif r["phase"] == "tool_repair":
            cls = "other_phase"
        else:
            cls = ""
        if live_phase and kind not in ("fire",):
            counts["eligible_not_fired"] += 1
        if cls:
            counts[cls] += 1
            picked.append((cls, r, expect))

    firing = [(r, e) for c, r, e in picked if c == "firing"]
    by_task = collections.Counter(by_file[r["source"]]["task"] for r, _ in firing)
    status = collections.Counter(e["status"] for _, e in firing if e)
    repairable = [(r["source"], r["request_id"]) for r, e in firing if e and e["status"] != "valid"]
    print(dict(counts), dict(by_task), dict(status), repairable, file=sys.stderr)
    # the planner's census (docs/tasks/T-0135.md, Goal and A2): stop rather than write a different fixture
    assert counts["eligible"] == 860
    assert len(firing) == 29 and by_task == {"S3": 18, "S1": 7, "S5": 4}
    assert status == {"valid": 28, "repairable": 1}
    assert repairable == [("20261022-210810-1975518.jsonl", "t1-gen-s2")]
    assert counts["open_no_call"] == 3 and counts["pass2"] == 10

    lines = []
    for cls, r, expect in picked:
        t = by_file[r["source"]]
        row = {
            "class": cls,
            "source": r["source"],
            "seq": r["seq"],
            "turn_id": r["turn_id"],
            "phase": r["phase"],
            "request_id": r["request_id"],
            "attempt_id": r["attempt_id"],
            "step": r["step"],
            "call_id": r["call_id"],
            "trial": {
                "task": t["task"],
                "repeat": t["repeat"],
                "success": t["success"],
                "tool_loop_end": t["tool_loop_end"],
            },
            "commits": r["commits"],
            "expect": expect,
        }
        lines.append(json.dumps(row, sort_keys=True))
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
