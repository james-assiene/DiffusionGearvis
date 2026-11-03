"""Extract the T-0142b answer fixture from the three A/B run directories (read-only; docs/tasks/T-0142b.md, Owned files).

Usage: ``uv run python tests/fixtures/t0142b/build_answers.py <out.jsonl>``

The census is the test-author's own (written before ``degenerate(..., min_repeats=...)`` existed, and independent of
``local_astra/conductor/degeneracy.py``): per ``llm.request`` of every trial's session file, the request's ``answer``
commits (``llm.commit`` records with its ``attempt_id`` and ``segment == "answer"``, in log order, raw ``text``),
concatenated with no separator, tokenised as the task says (``re.findall(r"\\w+|[^\\w\\s]+", text.casefold())``); for
every unit length n in 1..3 and every alignment, the longest run of a unit repeated back to back (brute force over
aligned windows); the longest run of one non-whitespace character; and the token count. ``k<K>_commit`` is the
1-based ordinal of the answer commit at which the cumulative text first has a unit repeated >= K times or a
non-whitespace character run >= 64 (``None``: never), for K in 3..8.

Cross-checked against the planner's evidence (docs/tasks/T-0142b.md, Goal) before anything is written (the script
exits 1 otherwise):

- 1,666 requests with answer commits: t0132 713, t0134 584, t0135 369;
- exactly 1 collapses at any K in 3..8: T-0135 arm B S2#5, session ``20261023-135637-1135093.jsonl``, attempt
  ``5008519a4296-t1-gen-s2-4``, 3 answer commits; its first answer commit holds ``. the`` 6 times back to back, so it
  trips at answer commit 1 (the task's "commit 0") for K <= 6 and at commit 2 for K = 7, 8; commits 2-3 add 640 and 405
  characters;
- the 1,665 others: longest unit repeat 1, longest non-whitespace character run 2, median 11 tokens, longest 58.

Rows (``class``):

- ``collapsed``: the 1, every answer commit kept;
- ``clean``: 40 others, the highest scores by this census, ranked by (longest unit repeat, longest character run,
  token count) descending, ties in (directory, session file, log order).

Each row keeps: ``source`` (``<run dir>/sessions/<file>``), ``session``, ``request_id``, ``attempt_id``, ``phase``,
``step``, the trial's ``arm``, ``task``, ``repeat``, ``success`` and ``tool_loop_end``, ``answers`` (the commit texts,
in order, with each commit's ``canvas_idx`` and ``thought_open``), ``max_repeat``, ``max_char_run``, ``n_tokens`` and
``k3_commit`` .. ``k8_commit``; the collapsed row also keeps ``thoughts`` (its thought commits, same shape), which the
session row replays before the answers. Regime: T-0132 at 2ccff41, T-0134 at e59aa69, T-0135 at e1ae76e (the task's Regime).

The chat logs under ``logs/sessions/`` (H-T142b-3) are read by the eval only, never copied here: they hold personal
documents shown to the assistant.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

DIRS = (
    "t0132-crop-budget-ab-2ccff41",
    "t0134-feedback-ab-e59aa69",
    "t0135-thought-call-ab-e1ae76e",
)
PER_DIR = {DIRS[0]: 713, DIRS[1]: 584, DIRS[2]: 369}
COLLAPSED = ("20261023-135637-1135093.jsonl", "5008519a4296-t1-gen-s2-4")
TOK = re.compile(r"\w+|[^\w\s]+")
KS = range(3, 9)


def logs_root() -> Path:
    here = Path(__file__).resolve()
    proc = subprocess.run(
        ["git", "-C", str(here.parent), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip()).parent / "logs"


def tokens(text: str) -> list[str]:
    return TOK.findall(text.casefold())


def max_unit_repeat(text: str, max_n: int = 3) -> int:
    toks = tokens(text)
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


def first_trip(texts: list[str], k: int, c: int = 64) -> int | None:
    acc = ""
    for i, t in enumerate(texts, 1):
        acc += t
        if max_unit_repeat(acc) >= k or max_char_run(acc) >= c:
            return i
    return None


def answers_by_request(path: Path) -> list[dict[str, Any]]:
    """``[{request_id, attempt_id, phase, step, answers}]`` for a session file, in ``llm.request`` order."""
    reqs: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        data = rec.get("data") or {}
        if rec.get("kind") == "llm.request":
            a = str(data.get("attempt_id"))
            reqs[a] = {
                "request_id": data.get("request_id"),
                "attempt_id": a,
                "phase": data.get("phase"),
                "step": data.get("step"),
                "answers": [],
                "thoughts": [],
            }
            order.append(a)
        elif rec.get("kind") == "llm.commit" and data.get("segment") in ("answer", "thought"):
            owner = reqs.get(str(data.get("attempt_id")))
            if owner is not None:
                owner[str(data["segment"]) + "s"].append(
                    {
                        "canvas_idx": data.get("canvas_idx"),
                        "thought_open": data.get("thought_open"),
                        "text": data.get("text", ""),
                    }
                )
    return [reqs[a] for a in order if reqs[a]["answers"]]


def score(r: dict[str, Any]) -> dict[str, Any]:
    texts = [c["text"] for c in r["answers"]]
    full = "".join(texts)
    r["max_repeat"] = max_unit_repeat(full)
    r["max_char_run"] = max_char_run(full)
    r["n_tokens"] = len(tokens(full))
    for k in KS:
        r[f"k{k}_commit"] = first_trip(texts, k)
    return r


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
            for r in answers_by_request(run / "sessions" / name):
                r.update(
                    {
                        "source": f"{d}/sessions/{name}",
                        "session": name,
                        "arm": t.get("arm"),
                        "task": t["task"],
                        "repeat": t["repeat"],
                        "success": t.get("success"),
                        "tool_loop_end": t.get("tool_loop_end"),
                    }
                )
                rows.append(score(r))
    return rows


def tripped_any(r: dict[str, Any]) -> bool:
    return any(r[f"k{k}_commit"] is not None for k in KS)


def check(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 1666, len(rows)
    for d, n in PER_DIR.items():
        got = sum(r["source"].startswith(d + "/") for r in rows)
        assert got == n, (d, got)
    collapsed = [r for r in rows if tripped_any(r)]
    assert [(r["session"], r["attempt_id"]) for r in collapsed] == [COLLAPSED], [
        (r["session"], r["attempt_id"]) for r in collapsed
    ]
    (c,) = collapsed
    assert (c["arm"], c["task"], c["repeat"], c["phase"]) == ("B", "S2", 5, "tool_step"), c
    assert len(c["answers"]) == 3, len(c["answers"])
    first = c["answers"][0]["text"]
    assert first == "Wait, let me check the cart. the. the. the. the. the. the.", first
    assert max_unit_repeat(first) == 6, max_unit_repeat(first)
    assert [len(a["text"]) for a in c["answers"][1:]] == [640, 405], [
        len(a["text"]) for a in c["answers"]
    ]
    assert [c[f"k{k}_commit"] for k in KS] == [1, 1, 1, 1, 2, 2], [c[f"k{k}_commit"] for k in KS]
    clean = [r for r in rows if not tripped_any(r)]
    assert len(clean) == 1665, len(clean)
    assert max(r["max_repeat"] for r in clean) == 1
    assert max(r["max_char_run"] for r in clean) == 2
    assert statistics.median(r["n_tokens"] for r in clean) == 11
    assert max(r["n_tokens"] for r in clean) == 58


def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed = [r for r in rows if tripped_any(r)]
    clean = [r for r in rows if not tripped_any(r)]
    top = sorted(clean, key=lambda r: (-r["max_repeat"], -r["max_char_run"], -r["n_tokens"]))[:40]
    out = []
    for cls, group in (("collapsed", collapsed), ("clean", top)):
        for r in group:
            row = {**r, "class": cls}
            if cls == "clean":
                del row["thoughts"]  # kept for the collapse only (the session replay of S2#5)
            out.append(row)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: build_answers.py <out.jsonl>", file=sys.stderr)
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
