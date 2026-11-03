"""T-0172 A3: ``scripts/replay_announced_calls.py <path>... [--json] [--rule tail|all]`` on synthetic session logs.

The session files are written here in the logged shapes (envelope ``seq``/``kind``/``turn_id``/``data``):
``llm.request`` (``phase``, ``request_id``, ``attempt_id``, ``step`` for a ``tool_step``, ``prompt`` with ``tools``
a list when tools are on and ``null`` when off), its ``llm.commit`` records (``segment`` ``thought``/``answer``/
``tool``), the ``llm.done`` record (``request_id``, ``commits``; the real logs carry no ``tool`` there) and the
``machine.step`` record whose event is ``{"type": "llm.done", "request_id", "tool"}``. **"Whose ``llm.done`` has
``tool: null``" is read from that event** (the only ``llm.done`` that carries ``tool`` in the logs; T-0172 Goal: "
``llm.done`` carried ``tool: null``"). A request with no such event (aborted) is not a no-call answer.

**The output format is not fixed by A3; these rows fix it** (named in the test-author's report):

- A ``<path>`` is a session ``.jsonl`` file, or a directory: every ``*.jsonl`` file under it (recursively, the
  directory itself included) whose parent directory is named ``sessions``, in sorted order -- A4's corpus shape
  ``logs/**/sessions/*.jsonl``, so a run directory's ``trials.jsonl`` and other logs are not read. ``file`` below is
  that path as a string.
- One line per no-call answer (A3: a ``generate``/``tool_step`` request with tools on, the event's ``tool`` null, no
  ``tool`` commit and a non-empty joined answer), in log order::

      {file} {request_id} step={step} {hit|miss} verb={verb|-} {json.dumps(answer[:120], ensure_ascii=False)}

  ``step`` is 1 for ``generate``; ``answer`` is the answer commits joined with no separator, stripped; the verdict
  and verb are ``announces(answer, mode=<--rule>)``'s (default ``tail``).
- Then one totals line::

      scanned {S} generate/tool_step requests; {N} no-call answers; hits: tail {T}, all {A}; announce-then-call {C}

  ``S`` counts every ``generate``/``tool_step`` request read (tools on or off); ``T`` and ``A`` count no-call
  answers hit by each mode whatever ``--rule`` is; ``C`` counts requests that ended with a call (the event's
  ``tool`` not null) whose joined answer the ``--rule`` mode hits.
- ``--json``: one object ``{"rule", "scanned", "no_call_answers", "hits": {"tail", "all"}, "announce_then_call",
  "rows": [{"file", "request_id", "step", "verdict", "verb", "answer"}]}``, ``verb`` ``null`` on a miss, ``answer``
  cut as in the text line.
- Exit 0; 2 on a usage error (no path, an unknown option or ``--rule`` value) or malformed input (a missing path,
  a line that is not JSON or not an object), with the offending file named on stderr.

Regime: CPU, no GPU, no network; the inputs are only read; deterministic, n = 1.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_announced_calls.py"
TOOLS = [{"type": "function", "function": {"name": "google_search", "parameters": {}}}]
CALL = '<|tool_call>call:google_search{query:<|"|>weather in Paris<|"|>}<tool_call|>'
TOOL = {"name": "google_search", "arguments": {"query": "weather in Paris"}}

ANNOUNCE = "I will search for the current weather in Paris again."
OBS_INTENT = "I don't see a booking form on this page. I will try to find the booking section."
LONG = (
    "The booking has already been sent for two people at 7:00 pm under the name Grace Hopper, and I do not see "
    "any option on this page to change the number of guests."
)


def req(
    rid: str,
    *,
    phase: str = "tool_step",
    step: int | None = None,
    tools: bool = True,
    thought: str | None = "Thinking about it.",
    answer: list[str] | None = None,
    call: bool = False,
    done_tool: Any = None,
    done: bool = True,
) -> dict[str, Any]:
    return {
        "rid": rid,
        "phase": phase,
        "step": step,
        "tools": tools,
        "thought": thought,
        "answer": answer or [],
        "call": call,
        "done_tool": done_tool,
        "done": done,
    }


def write_session(path: Path, reqs: list[dict[str, Any]], turn: str = "t1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [{"kind": "log.started", "turn_id": None, "data": {"pid": 1}}]
    for n, r in enumerate(reqs, 1):
        att = f"abc-{r['rid']}-{n}"
        data: dict[str, Any] = {"phase": r["phase"], "request_id": r["rid"], "attempt_id": att}
        if r["step"] is not None:
            data["step"] = r["step"]
        data["prompt"] = {
            "messages": [],
            "prefix_len": 1,
            "thinking": True,
            "assists": {},
            "tools": TOOLS if r["tools"] else None,
            "continue_final_message": r["phase"] == "tool_step",
        }
        records.append({"kind": "llm.request", "turn_id": turn, "data": data})
        commits: list[tuple[str, str]] = []
        if r["thought"] is not None:
            commits.append(("thought", r["thought"]))
        commits += [("answer", t) for t in r["answer"]]
        if r["call"]:
            commits.append(("tool", CALL))
        for i, (seg, text) in enumerate(commits):
            records.append(
                {
                    "kind": "llm.commit",
                    "turn_id": turn,
                    "data": {
                        "request_id": r["rid"],
                        "attempt_id": att,
                        "canvas_idx": i,
                        "segment": seg,
                        "text": text,
                        "thought_open": False,
                    },
                }
            )
        if r["done"]:
            records.append(
                {
                    "kind": "llm.done",
                    "turn_id": turn,
                    "data": {"request_id": r["rid"], "commits": len(commits)},
                }
            )
            records.append(
                {
                    "kind": "machine.step",
                    "turn_id": turn,
                    "data": {
                        "event": {
                            "t_ms": 1,
                            "type": "llm.done",
                            "request_id": r["rid"],
                            "tool": r["done_tool"],
                        },
                        "state_before": "THINKING",
                        "state_after": "SPEAKING",
                        "commands": [],
                    },
                }
            )
    lines = [json.dumps({"seq": i, "t_ms": i, **rec}) for i, rec in enumerate(records)]
    path.write_text("\n".join(lines) + "\n")
    return path


def three(root: Path) -> tuple[Path, Path, Path]:
    """A3's three requests: a no-call announcement, an announcement then a call, a tools-off chat (itself an
    announcement, so a script that ignores the tools flag lists it)."""
    run = root / "run"
    on = write_session(
        run / "sessions" / "20261027-100000-1.jsonl",
        [
            req(
                "t1-gen",
                phase="generate",
                answer=["I will search for the weather in Paris."],
                call=True,
                done_tool=TOOL,
            ),
            req("t1-gen-s2", step=2, answer=[ANNOUNCE]),
        ],
    )
    off = write_session(
        run / "sessions" / "20261027-100100-2.jsonl",
        [
            req(
                "t1-gen",
                phase="generate",
                tools=False,
                answer=["I will search for the weather in Paris."],
            )
        ],
    )
    (run / "trials.jsonl").write_text(
        json.dumps({"task": "S1", "repeat": 0, "session_log": on.name}) + "\n"
    )
    # not a session log (its parent is not ``sessions``): never read, though it is not one JSON object per line
    (run / "lines.jsonl").write_text("[1, 2, 3]\n")
    return run, on, off


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    if not SCRIPT.exists():
        raise NotImplementedError("T-0172 A3: scripts/replay_announced_calls.py does not exist yet")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=120
    )


def line(file: Path, rid: str, step: int, verdict: str, verb: str | None, answer: str) -> str:
    return (
        f"{file} {rid} step={step} {verdict} verb={verb or '-'} "
        f"{json.dumps(answer.strip()[:120], ensure_ascii=False)}"
    )


def totals(s: int, n: int, t: int, a: int, c: int) -> str:
    return f"scanned {s} generate/tool_step requests; {n} no-call answers; hits: tail {t}, all {a}; announce-then-call {c}"


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------------------------- A3 rows


def test_three_requests_text(tmp_path: Path) -> None:
    run, on, _off = three(tmp_path)
    before = snapshot(tmp_path)
    proc = run_script(str(run))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        line(on, "t1-gen-s2", 2, "hit", "search", ANNOUNCE),
        totals(3, 1, 1, 1, 1),
    ], proc.stdout
    assert snapshot(tmp_path) == before, "A3: read-only"


def test_three_requests_json(tmp_path: Path) -> None:
    run, on, off = three(tmp_path)
    for args in ([str(run)], [str(run / "sessions")], [str(on), str(off)]):
        proc = run_script(*args, "--json")
        assert proc.returncode == 0, proc.stderr
        doc = json.loads(proc.stdout)
        assert doc == {
            "rule": "tail",
            "scanned": 3,
            "no_call_answers": 1,
            "hits": {"tail": 1, "all": 1},
            "announce_then_call": 1,
            "rows": [
                {
                    "file": str(on),
                    "request_id": "t1-gen-s2",
                    "step": 2,
                    "verdict": "hit",
                    "verb": "search",
                    "answer": ANNOUNCE,
                }
            ],
        }, (args, doc)


def mixed(root: Path) -> Path:
    """Every exclusion beside every verdict: a prefill and a ReGround request (not scanned), a recovered thought
    call (no tool commit, the event's tool set: not a no-call answer, an announce-then-call), an empty answer, an
    aborted request (no ``llm.done``), a final answer (miss), an observation-plus-intent (tail hit, all miss), a
    long miss (cut to 120), and an answer split over commits."""
    return write_session(
        root / "logs" / "sessions" / "20261027-110000-9.jsonl",
        [
            req("t1-prefill", phase="prefill", answer=[ANNOUNCE]),
            req("t1-gen", phase="generate", answer=["The Eiffel Tower is 330 metres tall."]),
            req(
                "t1-gen-s2",
                step=2,
                answer=["I don't see a booking form on this page. ", OBS_INTENT[41:]],
            ),
            req(
                "t1-gen-s3",
                step=3,
                answer=["I will add the blue mug to your cart."],
                done_tool=TOOL,
            ),
            req("t1-gen-s3-rg", phase="reground", answer=[ANNOUNCE]),
            req("t1-gen-s4", step=4, answer=[]),
            req("t1-gen-s5", step=5, answer=["  ", LONG]),
            req("t1-gen-s6", step=6, answer=["I will scroll down."], done=False),
        ],
    )


def test_exclusions_and_rules(tmp_path: Path) -> None:
    f = mixed(tmp_path)
    assert "I don't see a booking form on this page. " + OBS_INTENT[41:] == OBS_INTENT, "premise"
    proc = run_script(str(f))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        line(f, "t1-gen", 1, "miss", None, "The Eiffel Tower is 330 metres tall."),
        line(f, "t1-gen-s2", 2, "hit", "find", OBS_INTENT),
        line(f, "t1-gen-s5", 5, "miss", None, LONG),
        totals(6, 3, 1, 0, 1),
    ], proc.stdout
    assert len(LONG.strip()) > 120, "premise: the long answer is cut"

    proc = run_script(str(f), "--rule", "all")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        line(f, "t1-gen", 1, "miss", None, "The Eiffel Tower is 330 metres tall."),
        line(f, "t1-gen-s2", 2, "miss", None, OBS_INTENT),
        line(f, "t1-gen-s5", 5, "miss", None, LONG),
        totals(6, 3, 1, 0, 1),
    ], proc.stdout

    doc = json.loads(run_script(str(tmp_path / "logs"), "--json", "--rule", "all").stdout)
    assert doc["rule"] == "all" and doc["hits"] == {"tail": 1, "all": 0}, doc
    assert [r["verdict"] for r in doc["rows"]] == ["miss", "miss", "miss"], doc["rows"]
    assert doc["rows"][2]["answer"] == LONG.strip()[:120]


def test_tool_commit_excluded_and_then_call_follows_rule(tmp_path: Path) -> None:
    """Review rows. (1) A request that kept a ``tool`` commit is no no-call answer even when the event's ``tool`` is
    null (a refused draft): not listed, not counted (mutant: the tool-commit exclusion dropped). (2) announce-then-call
    follows ``--rule``: a call whose answer is an observation plus intent counts under ``tail`` and not under ``all``
    (mutant: announce-then-call fixed to ``tail``)."""
    f = write_session(
        tmp_path / "sessions" / "20261027-120000-5.jsonl",
        [
            req("t1-gen", phase="generate", answer=[ANNOUNCE], call=True, done_tool=None),
            req("t1-gen-s2", step=2, answer=[OBS_INTENT], call=True, done_tool=TOOL),
        ],
    )
    proc = run_script(str(f))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [totals(2, 0, 0, 0, 1)], proc.stdout
    proc = run_script(str(f), "--rule", "all")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [totals(2, 0, 0, 0, 0)], proc.stdout
    doc = json.loads(run_script(str(f), "--json", "--rule", "all").stdout)
    assert (doc["no_call_answers"], doc["rows"], doc["announce_then_call"]) == (0, [], 0), doc


def test_malformed_line_exits_2_naming_the_file(tmp_path: Path) -> None:
    run, on, _off = three(tmp_path)
    bad = run / "sessions" / "20261027-100200-3.jsonl"
    bad.write_text(on.read_text() + '{"seq": 99, "kind": "llm.request", "data": {\n')
    proc = run_script(str(run))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert str(bad) in proc.stderr or bad.name in proc.stderr, proc.stderr
    not_obj = tmp_path / "array.jsonl"
    not_obj.write_text("[1, 2, 3]\n")
    proc = run_script(str(not_obj))
    assert proc.returncode == 2 and not_obj.name in proc.stderr, (proc.returncode, proc.stderr)


@pytest.mark.parametrize(
    "args",
    [[], ["--json"], ["--rule", "bogus", "X"], ["--rule"], ["--frobnicate", "X"]],
    ids=["no-path", "json-only", "bad-rule", "rule-no-value", "unknown-option"],
)
def test_usage_errors_exit_2(tmp_path: Path, args: list[str]) -> None:
    run, _on, _off = three(tmp_path)
    proc = run_script(*[str(run) if a == "X" else a for a in args])
    assert proc.returncode == 2, (args, proc.returncode, proc.stdout, proc.stderr)


def test_missing_path_exits_2_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "x.jsonl"
    proc = run_script(str(missing))
    assert proc.returncode == 2 and "x.jsonl" in proc.stderr, (proc.returncode, proc.stderr)


def test_no_network_imports() -> None:
    """A3 "no network": the script imports no networking module (read by AST; prose cannot trip it)."""
    if not SCRIPT.exists():
        raise NotImplementedError("T-0172 A3: scripts/replay_announced_calls.py does not exist yet")
    tree = ast.parse(SCRIPT.read_text())
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    banned = {
        "socket",
        "http",
        "urllib",
        "httpx",
        "requests",
        "aiohttp",
        "ddgs",
        "ssl",
        "websockets",
    }
    assert not [m for m in mods if m.split(".")[0] in banned], mods
    assert "local_astra.toolcall.announced_call" in mods or any(
        m == "local_astra.toolcall" for m in mods
    ), f"the script uses the rule itself, never a copy of it: {mods}"
