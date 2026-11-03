"""T-0170 criterion 6: ``scripts/replay_implicit_thought.py <session.jsonl|dir>... [--json]`` on synthetic logs.

Each row writes session files in the logged envelope (``seq``/``kind``/``turn_id``/``data``; ``llm.commit`` data as
at b372edaa) under ``tmp_path`` and runs the script as a subprocess. Expected listings are written by hand.

**The ``--json`` shape is not specified by criterion 6; these rows fix it** (named in the test-author report). stdout
is one JSON object with two lists:

- ``implicit``: one object per ``llm.commit`` record with ``implicit_thought: true`` (kind (a));
- ``splices``: one object per request attempt whose first non-``tool`` commit is ``answer`` and whose answer text
  contains ``[.!?][A-Z]`` (kind (b));

each object carrying at least ``path`` (the session file, as given or found under a directory), ``seq`` (the commit
record's), ``request_id`` and ``attempt_id``. Text mode is read loosely: every listed record's attempt id appears on
some line with its seq; an unlisted attempt id appears nowhere.

Exit codes: 0 on success; 2 on a usage error (no argument) or an unreadable input (missing file, malformed JSON),
the file named on stderr.

Regime: CPU, no GPU, no network; the inputs are only read; deterministic, n = 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_implicit_thought.py"


def require_script() -> None:
    if not SCRIPT.exists():
        raise AssertionError(
            "T-0170 criterion 6: scripts/replay_implicit_thought.py does not exist yet"
        )


def run(*args: str) -> subprocess.CompletedProcess[str]:
    require_script()
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, check=False
    )


def commit(seq: int, rid: str, aid: str, segment: str, text: str, **extra: Any) -> dict[str, Any]:
    data = {
        "request_id": rid,
        "attempt_id": aid,
        "canvas_idx": 0,
        "segment": segment,
        "text": text,
        "thought_open": False,
        "fence_close": None,
        "fence_continued": False,
        **extra,
    }
    return {"seq": seq, "kind": "llm.commit", "turn_id": "t1", "data": data}


def request(seq: int, rid: str, aid: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "llm.request",
        "turn_id": "t1",
        "data": {"phase": "generate", "request_id": rid, "attempt_id": aid},
    }


def write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def synthetic(root: Path) -> Path:
    """One session file holding every case (attempt ids name the case)."""
    return write(
        root / "sessions" / "s1.jsonl",
        [
            # (b) pre-fix splice: answer first, "sentence.The"
            request(1, "t1-gen", "a-splice"),
            commit(
                2,
                "t1-gen",
                "a-splice",
                "answer",
                "The user asks.\nI will answer in one sentence.The tower.",
            ),
            # thought first, answer with "e.g.The": no hit
            request(3, "t1-gen-s2", "a-eg"),
            commit(4, "t1-gen-s2", "a-eg", "thought", "plan"),
            commit(5, "t1-gen-s2", "a-eg", "answer", "Use a tool, e.g.The hammer."),
            # answer first, no splice: no hit
            request(6, "t2-gen", "a-clean"),
            commit(7, "t2-gen", "a-clean", "answer", "Hello there. Fine!"),
            # (a) post-fix implicit thought
            request(8, "t3-gen", "a-implicit"),
            commit(
                9, "t3-gen", "a-implicit", "thought", "The user greets me.", implicit_thought=True
            ),
            commit(10, "t3-gen", "a-implicit", "answer", "Hello there."),
            # tool first then answer with a splice: first non-tool commit is answer -> hit
            request(11, "t4-gen", "a-tool"),
            commit(12, "t4-gen", "a-tool", "tool", "<|tool_call>call:x{}<tool_call|>", stop_id=50),
            commit(13, "t4-gen", "a-tool", "answer", "Done.Next."),
        ],
    )


def keyed(items: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {(Path(str(i["path"])).name, i["seq"], i["request_id"], i["attempt_id"]) for i in items}


SPLICES = {("s1.jsonl", 2, "t1-gen", "a-splice"), ("s1.jsonl", 13, "t4-gen", "a-tool")}
IMPLICIT = {("s1.jsonl", 9, "t3-gen", "a-implicit")}


@pytest.mark.parametrize("as_dir", [False, True], ids=["file", "directory"])
def test_json_lists_both_kinds(tmp_path: Path, as_dir: bool) -> None:
    path = synthetic(tmp_path)
    proc = run(str(path.parent if as_dir else path), "--json")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert set(doc) >= {"implicit", "splices"}, sorted(doc)
    assert keyed(doc["splices"]) == SPLICES, doc["splices"]
    assert keyed(doc["implicit"]) == IMPLICIT, doc["implicit"]


def test_text_mode_lists_the_hits_only(tmp_path: Path) -> None:
    proc = run(str(synthetic(tmp_path)))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    for _name, seq, _rid, aid in SPLICES | IMPLICIT:
        assert any(aid in ln and str(seq) in ln for ln in lines), (aid, proc.stdout)
    for aid in ("a-eg", "a-clean"):
        assert aid not in proc.stdout, f"T-0170 criterion 6: {aid} is not a hit; {proc.stdout}"


def test_no_false_hit_on_thought_first_e_g(tmp_path: Path) -> None:
    """A thought-first request whose answer holds ``e.g.The`` is not listed (the first non-tool commit is thought)."""
    path = write(
        tmp_path / "s.jsonl",
        [
            request(1, "t1-gen", "a-eg"),
            commit(2, "t1-gen", "a-eg", "thought", "x"),
            commit(3, "t1-gen", "a-eg", "answer", "Try e.g.The first one."),
        ],
    )
    proc = run(str(path), "--json")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["splices"] == [] and doc["implicit"] == [], doc


def test_usage_and_unreadable_inputs_exit_2(tmp_path: Path) -> None:
    assert run().returncode == 2, "T-0170 criterion 6: no argument is a usage error"
    missing = tmp_path / "nope.jsonl"
    proc = run(str(missing))
    assert proc.returncode == 2 and "nope.jsonl" in proc.stderr, (proc.returncode, proc.stderr)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"seq": 1, "kind": "llm.commit"\n')
    proc = run(str(bad), "--json")
    assert proc.returncode == 2 and "bad.jsonl" in proc.stderr, (proc.returncode, proc.stderr)


def test_read_only(tmp_path: Path) -> None:
    path = synthetic(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert run(str(path.parent)).returncode == 0
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


# =========================================================================== review B-F2: scanner edges


def splices_of(path: Path) -> list[tuple[Any, ...]]:
    proc = run(str(path), "--json")
    assert proc.returncode == 0, proc.stderr
    return sorted(
        (i["seq"], i["request_id"], i["attempt_id"]) for i in json.loads(proc.stdout)["splices"]
    )


def test_only_sentence_punctuation_before_a_capital_is_a_splice(tmp_path: Path) -> None:
    """Answer-first, but the capitals follow a digit-colon, a comma and a lower-case letter, never ``.``/``!``/``?``:
    no hit (kills ``[^A-Za-z ][A-Z]`` and ``[a-z.!?][A-Z]``)."""
    path = write(
        tmp_path / "s.jsonl",
        [request(1, "t1-gen", "a1"), commit(2, "t1-gen", "a1", "answer", "Use iPhone 3:The x,Y")],
    )
    assert splices_of(path) == []


def test_attempts_of_one_request_are_judged_apart(tmp_path: Path) -> None:
    """One request id, two attempts: attempt 1 thought-first, attempt 2 answer-first with a splice. Exactly attempt 2
    (kills grouping by request id only)."""
    path = write(
        tmp_path / "s.jsonl",
        [
            request(1, "t1-gen", "a1"),
            commit(2, "t1-gen", "a1", "thought", "plan"),
            commit(3, "t1-gen", "a1", "answer", "Fine. Sure."),
            request(4, "t1-gen", "a2"),
            commit(5, "t1-gen", "a2", "answer", "I will answer.The tower."),
        ],
    )
    assert splices_of(path) == [(5, "t1-gen", "a2")]


@pytest.mark.parametrize(
    ("first", "second"),
    [("Hello there. ", "I will answer.The tower."), ("I will answer.", "The tower.")],
    ids=["inside-the-second", "across-the-boundary"],
)
def test_seq_is_the_commit_holding_the_capital(tmp_path: Path, first: str, second: str) -> None:
    """Two answer commits, the splice's capital in the second (wholly inside it, or with its ``.`` ending the first):
    ``seq`` is the second commit's (the script docstring)."""
    path = write(
        tmp_path / "s.jsonl",
        [
            request(1, "t1-gen", "a1"),
            commit(2, "t1-gen", "a1", "answer", first),
            commit(3, "t1-gen", "a1", "answer", second),
        ],
    )
    assert splices_of(path) == [(3, "t1-gen", "a1")]


@pytest.mark.parametrize("kind", ["not-utf8", "json-array-line"])
def test_unreadable_input_kinds_exit_2(tmp_path: Path, kind: str) -> None:
    """A file that is not UTF-8, and a line that is JSON but not an object: exit 2, the file named."""
    path = tmp_path / f"{kind}.jsonl"
    if kind == "not-utf8":
        path.write_bytes(b'{"seq": 1, "kind": "note", "data": {"t": "\xff\xfe"}}\n')
    else:
        path.write_text(json.dumps(request(1, "t1-gen", "a1")) + "\n[1,2]\n")
    proc = run(str(path), "--json")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert path.name in proc.stderr, proc.stderr
