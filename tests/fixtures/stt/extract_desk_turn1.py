#!/usr/bin/env python3
"""Extract the turn-1 slice of the 2026-09-16 desk session into ``desk-2026-09-16-turn1.json``.

Committed beside the data it produced (`.claude/rules/tests.md`: a fixture states its
provenance, its recording conditions and its generator). Run it against the session log:

    uv run python tests/fixtures/stt/extract_desk_turn1.py \
        logs/sessions/20260916-170934-3380576.jsonl

**Word texts are replaced by placeholders** (``w1``, ``w2``, ``w3``) on purpose: the evidence in
this fixture is the *timings* and the *pause-head vectors*: the human's sentence is not needed by
any row and is not checked in (docs/tasks/T-0051.md, *Fixture*).

What is extracted, and in which clock:

- ``steps``: every ``stt.step`` the session fed the turn machine between ``session_ms`` 242874 and
  245200, with its real four-entry ``prs`` vector, verbatim. ``t_ms`` is the **session** stamp the
  log carries; the *audio* (fed-audio) clock the endpointer actually saw is
  ``t_ms - audio_clock_offset_ms`` (see below).
- ``words``: the three ``stt.word`` rows of the turn with their real ``start_ms``/``end_ms``
  (audio times, untouched by the offset) and ``arrival_after_step_index`` — the index in
  ``steps`` of the last step that preceded the word on the wire, taken from the log's own record
  order, which is the wire order the client published.
- ``decisions``: the two ``stt.decision`` rows the session logged, verbatim, as the *observed*
  behaviour. Nothing asserts them: they are what the defect did, kept so the fixture carries its
  own evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIRST_STEP_MS = 242874  # session clock; docs/tasks/T-0051.md names 242874 → 245200
LAST_STEP_MS = 245200
FIRST_SEQ, LAST_SEQ = 3055, 3160  # the turn-1 slice of the log, by record sequence

#: The fed-audio clock runs behind the session clock by a constant this session could only bound,
#: never measure directly (docs/tasks/T-0051.md, *Regime*). The band comes from the three steps at
#: which the backstop did and did not fire; the midpoint is the value stored, and no rule in the
#: endpointer depends on it (it shifts every step by the same amount).
OFFSET_BAND_MS = (330, 395)
OFFSET_MS = 360


def main(log_path: Path, out_path: Path) -> None:
    steps: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for line in log_path.read_text().splitlines():
        record = json.loads(line)
        seq = record["seq"]
        if not FIRST_SEQ <= seq <= LAST_SEQ:
            continue
        if record["kind"] == "stt.decision":
            decisions.append({"seq": seq, "session_ms": record["session_ms"], **record["data"]})
            continue
        if record["kind"] != "machine.step":
            continue
        event = record["data"]["event"]
        if event["type"] == "stt.step":
            if FIRST_STEP_MS <= event["t_ms"] <= LAST_STEP_MS:
                steps.append({"seq": seq, "t_ms": event["t_ms"], "prs": event["pause_heads"]})
        elif event["type"] == "stt.word":
            words.append(
                {
                    "seq": seq,
                    "text": f"w{len(words) + 1}",
                    "start_ms": event["start_ms"],
                    "end_ms": event["end_ms"],
                    "arrival_t_ms": event["t_ms"],
                }
            )
    for word in words:
        before = [i for i, s in enumerate(steps) if s["seq"] < word["seq"]]
        word["arrival_after_step_index"] = before[-1] if before else -1
    fixture = {
        "source": {
            "log": "logs/sessions/20260916-170934-3380576.jsonl",
            "git": "28d7080",
            "session_id": "00ee8b9a4058",
            "records": [FIRST_SEQ, LAST_SEQ],
        },
        "clock": {
            "steps_t_ms": "session clock, as logged",
            "words_start_end_ms": "audio clock, as the recogniser reported them",
            "audio_clock_offset_ms": OFFSET_MS,
            "audio_clock_offset_band_ms": list(OFFSET_BAND_MS),
        },
        "steps": [{k: v for k, v in s.items() if k != "seq"} for s in steps],
        "words": [{k: v for k, v in w.items() if k != "seq"} for w in words],
        "observed_decisions": decisions,
    }
    out_path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"{out_path}: {len(steps)} steps, {len(words)} words, {len(decisions)} decisions")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parent / "desk-2026-09-16-turn1.json")
