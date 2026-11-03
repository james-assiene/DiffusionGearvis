"""Extract the T-0142 detector fixture from the two A/B run directories (read-only; docs/tasks/T-0142.md, Owned files).

Usage: ``uv run python tests/fixtures/t0142/build_thoughts.py <out.jsonl>``

The census is the test-author's own (written before ``local_astra/conductor/degeneracy.py`` existed, and independent
of it): per ``llm.request`` of every trial's session file, the request's ``thought`` commits (``llm.commit`` records
with its ``attempt_id`` and ``segment == "thought"``, in log order), concatenated, tokenised as the task says
(``re.findall(r"\\w+|[^\\w\\s]+", text.casefold())``); for every unit length n in 1..3 and every alignment, the longest
run of a unit repeated back to back (brute force over aligned windows); and the longest run of one non-whitespace
character. ``k12_commit`` / ``k8_commit`` are the 1-based ordinals of the thought commit at which the cumulative text
first has a unit repeated >= 12 (resp. >= 8) times or a character run >= 64 (``None``: never).

Cross-checked against the planner's evidence before anything is written (the script exits 1 otherwise):

- exactly 10 ``collapsed`` requests (K = 12): T-0134 B S5#8, B S5#2, B S5#9, B S5#1, C S2#10, C S2#15; T-0132 S5#6
  and S1#5 (``tool_step``) and S5#8, S5#6 (``reground``); each trips at thought commit 2-4 of 3-19;
- 1,976 other requests with thought commits, whose longest unit repeat is 2 and whose longest non-whitespace
  character run is under 64 (measured: 3; the planner's 8 counts whitespace runs, re-measured here: 8).

Rows (``class``):

- ``collapsed``: the 10, every thought commit kept;
- ``clean``: 40 others, the most repetitive by this census: all 20 whose longest unit repeat is 2, then the 10 with
  the most ``"Actually,"`` occurrences, then the 10 with the most deal-list lines (``- X: $..``), each group in
  (directory, session file, log order) among rows not already taken.

Each row keeps: ``source`` (``<run dir>/sessions/<file>``), ``session``, ``request_id``, ``attempt_id``, ``phase``,
``step``, the trial's ``arm``, ``task``, ``repeat``, ``success`` and ``tool_loop_end``, ``thoughts`` (the commit texts,
in order, with each commit's ``canvas_idx`` and ``thought_open``), ``max_repeat`` and ``max_char_run`` (this census)
and ``k12_commit``/``k8_commit``. Regime: T-0134 at e59aa69 and T-0132 at 2ccff41 (the task's Regime section).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DIRS = ("t0134-feedback-ab-e59aa69", "t0132-crop-budget-ab-2ccff41")
TOK = re.compile(r"\w+|[^\w\s]+")
#: the planner's ten, as (run dir prefix, arm, task, repeat, phase)
EXPECTED = {
    ("t0134", "B", "S5", 8, "tool_step"),
    ("t0134", "B", "S5", 2, "tool_step"),
    ("t0134", "B", "S5", 9, "tool_step"),
    ("t0134", "B", "S5", 1, "tool_step"),
    ("t0134", "C", "S2", 10, "tool_step"),
    ("t0134", "C", "S2", 15, "tool_step"),
    ("t0132", None, "S5", 6, "tool_step"),
    ("t0132", None, "S1", 5, "tool_step"),
    ("t0132", None, "S5", 8, "reground"),
    ("t0132", None, "S5", 6, "reground"),
}


def logs_root() -> Path:
    here = Path(__file__).resolve()
    proc = subprocess.run(
        ["git", "-C", str(here.parent), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip()).parent / "logs"


def max_unit_repeat(text: str, max_n: int = 3) -> int:
    toks = TOK.findall(text.casefold())
    best = 1 if toks else 0
    for n in range(1, max_n + 1):
        for start in range(n):
            run = 1
            i = start
            while i + 2 * n <= len(toks):
                run = run + 1 if toks[i : i + n] == toks[i + n : i + 2 * n] else 1
                best = max(best, run)
                i += n
    return best


def max_char_run(text: str) -> int:
    best = 0
    prev, run = "", 0
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        if not ch.isspace():
            best = max(best, run)
    return best


def max_any_run(text: str) -> int:
    return max((len(m.group(0)) for m in re.finditer(r"(.)\1*", text, re.S)), default=0)


def first_trip(texts: list[str], k: int, c: int = 64) -> int | None:
    acc = ""
    for i, t in enumerate(texts, 1):
        acc += t
        if max_unit_repeat(acc) >= k or max_char_run(acc) >= c:
            return i
    return None


def census() -> list[dict[str, Any]]:
    root = logs_root()
    rows: list[dict[str, Any]] = []
    for d in DIRS:
        run = root / d
        trials = [
            json.loads(x) for x in (run / "trials.jsonl").read_text().splitlines() if x.strip()
        ]
        for t in sorted(trials, key=lambda t: Path(t["session_log"]).name):
            name = Path(t["session_log"]).name
            reqs: dict[str, dict[str, Any]] = {}
            order: list[str] = []
            for line in (run / "sessions" / name).read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                data = rec.get("data") or {}
                if rec["kind"] == "llm.request":
                    a = str(data["attempt_id"])
                    reqs[a] = {
                        "source": f"{d}/sessions/{name}",
                        "session": name,
                        "request_id": data["request_id"],
                        "attempt_id": a,
                        "phase": data["phase"],
                        "step": data.get("step"),
                        "arm": t.get("arm"),
                        "task": t["task"],
                        "repeat": t["repeat"],
                        "success": t.get("success"),
                        "tool_loop_end": t.get("tool_loop_end"),
                        "thoughts": [],
                    }
                    order.append(a)
                elif rec["kind"] == "llm.commit" and str(data.get("attempt_id")) in reqs:
                    if data.get("segment") == "thought":
                        reqs[str(data["attempt_id"])]["thoughts"].append(
                            {
                                "canvas_idx": data.get("canvas_idx"),
                                "thought_open": data.get("thought_open"),
                                "text": data.get("text", ""),
                            }
                        )
            for a in order:
                r = reqs[a]
                if not r["thoughts"]:
                    continue
                texts = [c["text"] for c in r["thoughts"]]
                full = "".join(texts)
                r["max_repeat"] = max_unit_repeat(full)
                r["max_char_run"] = max_char_run(full)
                r["max_any_run"] = max_any_run(full)
                r["k12_commit"] = first_trip(texts, 12)
                r["k8_commit"] = first_trip(texts, 8)
                rows.append(r)
    return rows


def check(rows: list[dict[str, Any]]) -> None:
    collapsed = [r for r in rows if r["k12_commit"] is not None]
    got = {(r["source"][:5], r["arm"], r["task"], r["repeat"], r["phase"]) for r in collapsed}
    assert len(collapsed) == 10 and got == EXPECTED, sorted(got, key=str)
    for r in collapsed:
        assert 2 <= r["k12_commit"] <= 4 and 3 <= len(r["thoughts"]) <= 19, r["attempt_id"]
    clean = [r for r in rows if r["k12_commit"] is None]
    assert len(clean) == 1976, len(clean)
    assert max(r["max_repeat"] for r in clean) == 2
    assert max(r["max_char_run"] for r in clean) < 64
    assert max(r["max_any_run"] for r in clean) == 8, "the planner's 8 (whitespace runs counted)"
    assert all(r["k8_commit"] is None for r in clean), "MIN_REPEATS = 8 trips no clean request"


def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed = [r for r in rows if r["k12_commit"] is not None]
    clean = [r for r in rows if r["k12_commit"] is None]
    taken: list[dict[str, Any]] = [r for r in clean if r["max_repeat"] == 2]
    assert len(taken) == 20, len(taken)

    def text(r: dict[str, Any]) -> str:
        return "".join(c["text"] for c in r["thoughts"])

    def top(key: Any, n: int) -> list[dict[str, Any]]:
        ids = {r["attempt_id"] for r in taken}
        pool = [r for r in clean if r["attempt_id"] not in ids and key(r) > 0]
        return sorted(pool, key=lambda r: -key(r))[:n]  # stable: log order among ties

    taken += top(lambda r: text(r).count("Actually,"), 10)
    taken += top(lambda r: len(re.findall(r"^\s*- [^:\n]+: \$", text(r), re.M)), 10)
    assert len(taken) == 40, len(taken)
    out = []
    for cls, group in (("collapsed", collapsed), ("clean", taken)):
        for r in group:
            row = {k: v for k, v in r.items() if k != "max_any_run"}
            row["class"] = cls
            out.append(row)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: build_thoughts.py <out.jsonl>", file=sys.stderr)
        return 2
    rows = census()
    try:
        check(rows)
    except AssertionError as exc:
        print(f"census does not match the planner's evidence: {exc}", file=sys.stderr)
        return 1
    out = select(rows)
    Path(argv[0]).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    print(f"wrote {len(out)} rows to {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
