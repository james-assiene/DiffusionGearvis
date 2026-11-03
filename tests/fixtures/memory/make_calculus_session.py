#!/usr/bin/env python3
"""Rebuild ``calculus_session.json`` from the origin session log (T-0053).

The three captions are lifted **verbatim** out of the ``[memory]`` block of the ``llm.request``
record at ``seq 697`` of ``logs/sessions/20260916-202852-987340.jsonl`` — the prompt that reached
the engine on the human's desk at 2026-09-16 20:29:42+01:00, the one whose answer described "the
provided screenshots of a calculus textbook" with zero images attached.

The log record carries no episode row: it holds the rendered block only, and the ``memory.search``
tap that would have carried the ids, scores and timestamps does not exist yet (that is criterion
23, group C). So ``ts_ms``/``id``/``phash`` are looked up **by exact caption match** in the human's
own store, ``~/.local/share/local_astra/memory.sqlite`` (read-only), and the ids found there are
recorded in ``_EXPECTED_IDS`` so a run on any other box either reproduces them or fails loudly.

Usage (from the repository root)::

    uv run python tests/fixtures/memory/make_calculus_session.py            # rebuild in place
    uv run python tests/fixtures/memory/make_calculus_session.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

LOG = Path("logs/sessions/20260916-202852-987340.jsonl")
SEQ = 697
TURN_ID = "t1"
#: Recorded unexpanded in the fixture (T-0157): the expanded form names the box's home directory.
DB_RECORDED = "~/.local/share/local_astra/memory.sqlite"
DB = Path(os.path.expanduser(DB_RECORDED))
OUT = Path(__file__).resolve().parent / "calculus_session.json"

#: episodes-table ids of the three captions, in the order the block renders them, as found in the
#: human's store on 2026-09-16. Recorded so that a rebuild that silently matches other rows fails.
_EXPECTED_IDS = (7, 4, 11)


def _memory_block_and_transcript() -> tuple[str, str]:
    record = None
    with LOG.open() as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate.get("seq") == SEQ:
                record = candidate
                break
    if record is None:
        raise SystemExit(f"no record with seq {SEQ} in {LOG}")
    assert record["kind"] == "llm.request" and record["turn_id"] == TURN_ID, record["kind"]
    messages = record["data"]["prompt"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"], "the origin turn had no history"
    parts = messages[1]["content"]
    assert [p["type"] for p in parts] == ["text", "text"], "the origin turn carried no image"
    return parts[0]["text"], parts[1]["text"]


def _captions(block: str) -> list[str]:
    assert block.startswith("[memory]\n- "), block[:40]
    body = block[len("[memory]\n- ") :]
    captions = body.split("\n- ")
    captions[-1] = captions[-1].rstrip("\n")
    assert len(captions) == 3, len(captions)
    return captions


def _rows(captions: list[str]) -> list[dict[str, object]]:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = []
    for caption in captions:
        found = conn.execute(
            "SELECT id, ts_ms, source, thumbnail_path, phash FROM episodes WHERE caption = ?",
            (caption,),
        ).fetchall()
        assert len(found) == 1, f"{len(found)} rows match a caption verbatim"
        eid, ts_ms, source, thumbnail_path, phash = found[0]
        rows.append(
            {
                "db_id": int(eid),
                "ts_ms": int(ts_ms),
                "source": str(source),
                "thumbnail_name": Path(str(thumbnail_path)).name,
                "phash": str(phash),
                "caption": caption,
            }
        )
    conn.close()
    assert tuple(r["db_id"] for r in rows) == _EXPECTED_IDS, [r["db_id"] for r in rows]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    block, transcript = _memory_block_and_transcript()
    payload = {
        "provenance": {
            "log": str(LOG),
            "seq": SEQ,
            "turn_id": TURN_ID,
            "wall": "2026-09-16T20:29:42.231230+01:00",
            "generator": "tests/fixtures/memory/make_calculus_session.py",
            "store": DB_RECORDED,
        },
        "turn": {
            "transcript": transcript,
            "history": [],
            "frame": None,
            "assists": {"page_text": False, "pdf_text": False, "cells": False, "at_spi": False},
            "rendered_memory_block": block,
        },
        "episodes": _rows(_captions(block)),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("ok" if same else "DIFFERS")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
