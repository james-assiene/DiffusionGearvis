"""T-0135 A2: ``scripts/replay_thought_calls.py <run-dir> [--json]`` on a run directory rebuilt from the fixture.

The fixture ``tests/fixtures/t0135/thought_tails.jsonl`` (MANIFEST.md beside it) holds 59 requests copied from the
T-0132 run: the 29 firings with their expected verdicts, the 3 call-less open-thought requests, the 10 ReGround pass-2
responses of the same shape, 2 other ReGround requests, 6 closed-thought endings, 6 answer-channel calls beside prose
naming a tool, one ``prefill`` and both ``tool_repair`` requests. ``build_run`` writes them back as a run directory in
the logged shapes: ``trials.jsonl`` (one row per session file: ``task``, ``repeat``, ``session_log``, ``success``,
``tool_loop_end``) and ``sessions/<file>`` with, per request in log order, an ``llm.request`` record, its
``llm.commit`` records and an ``llm.done`` (envelope ``seq``/``kind``/``turn_id``/``data``, as logged; fields the rule
does not need, such as prompts, are dropped).

**The ``--json`` shape is not specified by A2; these rows fix it** (named in the test-author report). stdout is one
JSON object:

- ``scanned``: the number of eligible ``generate``/``tool_step`` requests read;
- ``firings``: one object per firing with A2's line fields: ``task``, ``repeat``, ``session`` (the session file
  name), ``request_id``, ``step`` (``1`` for the turn's ``generate``, as its ``tool.result`` logs; the request's
  ``step`` for ``tool_step``), ``status``, ``name``, ``arguments``, ``reason`` (the ``live.classify`` verdict);
- ``totals``: ``{task: {status: count}}`` over the firings (absent and zero counts are equal);
- ``pass2_same_shape``: the informational count of ReGround pass-2 responses of the firing shape.

Text mode (no ``--json``) is read loosely: one line per firing carrying its session file, request id, status and
name; and the pass-2 count stated beside the words "out of scope".

Exit codes (A2): 0 on success; 2 for an unreadable run directory, a missing file or malformed JSON, naming the file.

Regime: CPU, no GPU, no network; the run directory is only read; deterministic, n = 1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_thought_calls.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0135" / "thought_tails.jsonl"
TAG = "<|tool_call>"
CLOSE = "<tool_call|>"
Q = '<|"|>'
#: A2's thresholds by task x status, written out from the task file (S3 18, S1 7, S5 4; 28 valid, 1 repairable on S5).
#: T-0196 amended S5 to 4 valid, 0 repairable (the doubled close is judged as its single-close call; T0196_AMENDED).
TOTALS = {"S3": {"valid": 18}, "S1": {"valid": 7}, "S5": {"valid": 4}}
PASS2 = 10


#: T-0196 (docs/tasks/T-0196.md criterion 8, Owned files): the S5 doubled close ``click_at{x:155,y:900}`` followed by
#: ``<tool_call|><tool_call|>`` is judged as its single-close call, ``valid``; the fixture's planner table (frozen,
#: T-0135) still says ``repairable``, so the expectation is amended here, by hand.
T0196_AMENDED = {
    ("20261022-210810-1975518.jsonl", "t1-gen-s2"): {
        "status": "valid",
        "name": "click_at",
        "arguments": {"x": 155, "y": 900},
        "reason": None,
    }
}


def fixture_rows() -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    for r in rows:
        amended = T0196_AMENDED.get((r.get("source"), r.get("request_id")))
        if amended is not None and r.get("class") == "firing":
            assert r["expect"]["status"] == "repairable", (
                "premise: the T-0135 table row T-0196 amends"
            )
            r["expect"] = {**r["expect"], **amended}
    return rows


def require_script() -> None:
    if not SCRIPT.exists():
        raise AssertionError("T-0135 A2: scripts/replay_thought_calls.py does not exist yet")


def build_run(root: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as a run directory (module docstring): one trial per session file, records in log order."""
    run = root / "run"
    (run / "sessions").mkdir(parents=True)
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    trials = []
    for source in sorted(by_source):
        reqs = sorted(by_source[source], key=lambda r: r["seq"])
        t = reqs[0]["trial"]
        trials.append(
            {
                "task": t["task"],
                "repeat": t["repeat"],
                "session_log": source,
                "success": t["success"],
                "tool_loop_end": t["tool_loop_end"],
            }
        )
        records: list[dict[str, Any]] = []
        for r in reqs:
            data: dict[str, Any] = {
                "phase": r["phase"],
                "request_id": r["request_id"],
                "attempt_id": r["attempt_id"],
            }
            if r.get("step") is not None:
                data["step"] = r["step"]
            if r.get("call_id") is not None:
                data["call_id"] = r["call_id"]
            records.append({"kind": "llm.request", "turn_id": r["turn_id"], "data": data})
            for c in r["commits"]:
                records.append(
                    {
                        "kind": "llm.commit",
                        "turn_id": r["turn_id"],
                        "data": {
                            "request_id": r["request_id"],
                            "attempt_id": r["attempt_id"],
                            "canvas_idx": c["canvas_idx"],
                            "segment": c["segment"],
                            "text": c["text"],
                            "thought_open": c["thought_open"],
                        },
                    }
                )
            records.append(
                {
                    "kind": "llm.done",
                    "turn_id": r["turn_id"],
                    "data": {"request_id": r["request_id"], "commits": len(r["commits"])},
                }
            )
        lines = [json.dumps({"seq": i, **rec}) for i, rec in enumerate(records)]
        (run / "sessions" / source).write_text("\n".join(lines) + "\n")
    (run / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials) + "\n")
    return run


def run_script(run: Path, *args: str) -> subprocess.CompletedProcess[str]:
    require_script()
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


def run_json(run: Path) -> dict[str, Any]:
    proc = run_script(run, "--json")
    assert proc.returncode == 0, f"A2: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"A2 --json: stdout is not one JSON object: {exc}\n{proc.stdout[:2000]}"
        ) from None
    assert isinstance(doc, dict), "A2 --json: stdout is one JSON object"
    for key in ("scanned", "firings", "totals", "pass2_same_shape"):
        assert key in doc, f"A2 --json: key {key!r} missing (module docstring); got {sorted(doc)}"
    return doc


def totals_of(doc: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for task, by_status in doc["totals"].items():
        kept = {s: int(n) for s, n in by_status.items() if int(n) != 0}
        if kept:
            out[task] = kept
    return out


def expected_firing(r: Mapping[str, Any]) -> dict[str, Any]:
    e = r["expect"]
    return {
        "task": r["trial"]["task"],
        "repeat": r["trial"]["repeat"],
        "session": r["source"],
        "request_id": r["request_id"],
        "step": 1 if r["phase"] == "generate" else r["step"],
        "status": e["status"],
        "name": e["name"],
        "arguments": e["arguments"],
        "reason": e["reason"],
    }


def firing_view(f: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "task",
        "repeat",
        "session",
        "request_id",
        "step",
        "status",
        "name",
        "arguments",
        "reason",
    )
    missing = [k for k in keys if k not in f]
    assert not missing, f"A2 --json: a firing lacks {missing}: {f}"
    view = {k: f[k] for k in keys}
    view["session"] = Path(str(f["session"])).name  # a path or a bare file name
    return view


def tree_sha(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def key(f: Mapping[str, Any]) -> tuple[str, str]:
    return (Path(str(f["session"])).name, str(f["request_id"]))


# --------------------------------------------------------------------------- the fixture premise


def test_fixture_premise() -> None:
    """The fixture holds what the task file's ``test_t0135_replay.py`` row names: 29 firings (S3 18, S1 7, S5 4),
    3 call-less open-thought requests, 10 pass-2 responses, closed-thought negatives including >= 5 answer-channel
    calls beside prose about calls. (Green: a premise, not an acceptance row.)"""
    rows = fixture_rows()
    c = Counter(r["class"] for r in rows)
    assert c["firing"] == 29 and c["open_no_call"] == 3 and c["pass2"] == PASS2
    assert c["answer_call_prose"] >= 5 and c["closed"] >= 1
    tasks = Counter(r["trial"]["task"] for r in rows if r["class"] == "firing")
    assert tasks == {"S3": 18, "S1": 7, "S5": 4}
    for r in rows:
        if r["class"] == "answer_call_prose":
            assert any(x["segment"] == "tool" for x in r["commits"])
        if r["class"] == "pass2":
            assert r["phase"] == "reground"


# --------------------------------------------------------------------------- A2 on the fixture


def test_json_firings_equal_the_planners_table(tmp_path: Path) -> None:
    """A2: with ``--json`` the firings are exactly the 29 fixture firings, each with its task, repeat, session file,
    request id, step and ``live.classify`` verdict (28 valid; ``…-210810-…`` ``t1-gen-s2`` repairable, "text after
    call"); no closed, call-less, answer-call, pass-2, prefill or repair row fires; the run directory is unchanged."""
    rows = fixture_rows()
    run = build_run(tmp_path, rows)
    before = tree_sha(run)
    doc = run_json(run)
    assert tree_sha(run) == before, "A2: the replay is read-only"
    want = {
        (r["source"], r["request_id"]): expected_firing(r) for r in rows if r["class"] == "firing"
    }
    got_list = [firing_view(f) for f in doc["firings"]]
    got = {key(f): f for f in got_list}
    assert len(got) == len(got_list), "A2: a request is reported twice"
    extra = sorted(set(got) - set(want))
    missing = sorted(set(want) - set(got))
    assert not extra, f"A2: fired on requests the rule does not match: {extra}"
    assert not missing, f"A2: did not fire on: {missing}"
    for k in sorted(want):
        assert got[k] == want[k], f"A2: {k}: {got[k]} != {want[k]}"
    statuses = Counter(f["status"] for f in got_list)
    # T-0196: 29 valid (was 28 valid + the S5 doubled close repairable)
    assert statuses == {"valid": 29}
    assert [k for k, f in got.items() if f["status"] == "repairable"] == []


def test_json_scanned_counts_only_generate_and_tool_step(tmp_path: Path) -> None:
    """A2: ``scanned`` counts the eligible ``generate``/``tool_step`` requests (29 + 3 + 6 + 6 = 44 in the fixture),
    never the ReGround, ``prefill`` or ``tool_repair`` ones (15 more)."""
    rows = fixture_rows()
    eligible = sum(r["phase"] in ("generate", "tool_step") for r in rows)
    assert eligible == 44 and len(rows) == 59, "premise"
    doc = run_json(build_run(tmp_path, rows))
    assert doc["scanned"] == eligible


def test_json_totals_by_task_and_status(tmp_path: Path) -> None:
    """A2: the totals by task x status are A2's (S3 18 valid, S1 7 valid, S5 3 valid + 1 repairable)."""
    doc = run_json(build_run(tmp_path, fixture_rows()))
    assert totals_of(doc) == TOTALS


def test_pass2_is_informational_and_never_a_firing(tmp_path: Path) -> None:
    """A2 (adversarial; mutant: ReGround pass 2 scanned like a live step). The 10 pass-2 responses of the firing shape
    are counted in ``pass2_same_shape`` (not the 12 ReGround rows), never listed as firings, never in the totals, never
    in ``scanned``. A run directory of the pass-2 and ReGround rows alone reports 0 scanned, no firings, 10 pass-2."""
    rows = fixture_rows()
    doc = run_json(build_run(tmp_path / "all", rows))
    assert doc["pass2_same_shape"] == PASS2
    reground = {(r["source"], r["request_id"]) for r in rows if r["phase"] == "reground"}
    assert len(reground) == 12, "premise"
    assert not reground & {key(f) for f in doc["firings"]}, (
        "A2: a pass-2 response was counted as a firing"
    )

    only = [r for r in rows if r["phase"] == "reground"]
    doc2 = run_json(build_run(tmp_path / "pass2", only))
    assert doc2["scanned"] == 0
    assert doc2["firings"] == []
    assert totals_of(doc2) == {}
    assert doc2["pass2_same_shape"] == PASS2


def test_text_mode_lines_and_pass2_note(tmp_path: Path) -> None:
    """A2 without ``--json``: exit 0; one line per firing carrying its session file, request id, verdict status and
    tool name; the pass-2 count (10) on a line that says it is out of scope."""
    rows = fixture_rows()
    proc = run_script(build_run(tmp_path, rows))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    for r in rows:
        if r["class"] != "firing":
            continue
        e = r["expect"]
        hits = [
            ln
            for ln in lines
            if r["source"].removesuffix(".jsonl") in ln
            and r["request_id"] in ln
            and e["status"] in ln
            and e["name"] in ln
        ]
        assert len(hits) == 1, f"A2: one line for {r['source']} {r['request_id']}, got {hits}"
    note = [ln for ln in lines if "out of scope" in ln.lower()]
    assert note and any(str(PASS2) in ln for ln in note), (
        f"A2: the pass-2 count is not stated as out of scope:\n{proc.stdout}"
    )


# --------------------------------------------------------------------------- hand-made requests (adversarial)


def _req(
    source: str, task: str, commits: Sequence[tuple[str, str, bool]], *, phase: str = "tool_step"
) -> dict:
    return {
        "class": "hand",
        "source": source,
        "seq": 1,
        "turn_id": "t1",
        "phase": phase,
        "request_id": "t1-gen-s2" if phase == "tool_step" else "t1-gen",
        "attempt_id": f"{source[:8]}-t1-gen-s2-3",
        "step": 2 if phase == "tool_step" else None,
        "call_id": None,
        "trial": {"task": task, "repeat": 0, "success": False, "tool_loop_end": "final"},
        "commits": [
            {"seq": 2 + i, "canvas_idx": i, "segment": s, "text": t, "thought_open": o}
            for i, (s, t, o) in enumerate(commits)
        ],
        "expect": None,
    }


CLICK_A = f"{TAG}call:click_at{{x:10,y:20}}{CLOSE}"
CLICK_B = f"{TAG}call:click_at{{x:30,y:40}}{CLOSE}"
TYPE_HI = f"{TAG}call:type_at{{press_enter:false,text:{Q}hi{Q},x:3,y:4}}{CLOSE}"

#: (name, request, expected firing or None). Each is one session file of its own.
HAND: list[tuple[str, dict, dict | None]] = [
    (
        "answer-channel tool commit kept (mutation 3)",
        _req(
            "20991231-000001-1.jsonl",
            "H1",
            [("thought", "I will click.\n" + CLICK_A, True), ("tool", CLICK_B, False)],
        ),
        None,
    ),
    (
        "answer-channel tool commit before a later open thought with a tail call (mutation 3)",
        _req(
            "20991231-000011-1.jsonl",
            "H1b",
            [("tool", CLICK_B, False), ("thought", "Next I will click.\n" + CLICK_A, True)],
        ),
        None,
    ),
    (
        "call split over two open-thought canvases: concatenated in order",
        _req(
            "20991231-000002-1.jsonl",
            "H2",
            [
                ("thought", "I will click the button.\n" + TAG + "call:click_at{x:12,", True),
                ("thought", "y:34}" + CLOSE + "\n", True),
            ],
        ),
        {"status": "valid", "name": "click_at", "arguments": {"x": 12, "y": 34}, "reason": None},
    ),
    (
        "last tag, not first (mutation 1)",
        _req(
            "20991231-000003-1.jsonl",
            "H3",
            [("thought", "First:\n" + CLICK_A + "\nNo, type instead:\n" + TYPE_HI, True)],
        ),
        {
            "status": "valid",
            "name": "type_at",
            "arguments": {"press_enter": False, "text": "hi", "x": 3, "y": 4},
            "reason": None,
        },
    ),
    (
        "prose after the close (mutation 2)",
        _req(
            "20991231-000004-1.jsonl",
            "H4",
            [("thought", "Plan.\n" + CLICK_A + " Actually, I should scroll.", True)],
        ),
        None,
    ),
    (
        "the last non-tool commit is an answer on an open-thought canvas: the segment decides, not the flag",
        _req(
            "20991231-000005-1.jsonl",
            "H5",
            [("thought", "Plan.\n" + CLICK_A, True), ("answer", "Done.", True)],
        ),
        None,
    ),
    (
        "the last non-tool commit is a whitespace answer on an open-thought canvas (flag-only mutant)",
        _req(
            "20991231-000015-1.jsonl",
            "H5b",
            [("thought", "Plan.\n" + CLICK_A, True), ("answer", "\n", True)],
        ),
        None,
    ),
    (
        "a closed thought holding a call, then an open thought: never stitched",
        _req(
            "20991231-000006-1.jsonl",
            "H6",
            [
                ("thought", "Plan.\n" + TAG + "call:click_at{x:12,", False),
                ("thought", "y:34}" + CLOSE, True),
            ],
        ),
        None,
    ),
    (
        "a closed thought ending in a call, no answer: the thought is closed",
        _req("20991231-000007-1.jsonl", "H7", [("thought", "Plan.\n" + CLICK_A, False)]),
        None,
    ),
    (
        "the turn's generate (step 1) fires too",
        _req(
            "20991231-000008-1.jsonl",
            "H8",
            [("thought", "Plan.\n" + CLICK_B + "\n\n", True)],
            phase="generate",
        ),
        {"status": "valid", "name": "click_at", "arguments": {"x": 30, "y": 40}, "reason": None},
    ),
]


def test_hand_rows(tmp_path: Path) -> None:
    """A2, adversarial, on hand-made requests beside the fixture: a request that kept a ``tool`` commit never fires;
    the final open thought block is the concatenation of its open ``thought`` commits; the last tag wins; prose after
    the close, an answer as the last non-tool commit, a closed thought, and a tag in a closed thought stitched to a
    later open one never fire; the turn's ``generate`` fires as step 1."""
    run = build_run(tmp_path, [r for _, r, _ in HAND])
    doc = run_json(run)
    got = {key(f): firing_view(f) for f in doc["firings"]}
    assert doc["scanned"] == len(HAND)
    for name, r, want in HAND:
        k = (r["source"], r["request_id"])
        if want is None:
            assert k not in got, f"A2 {name}: fired: {got.get(k)}"
            continue
        assert k in got, f"A2 {name}: did not fire"
        expect = {
            "task": r["trial"]["task"],
            "repeat": 0,
            "session": r["source"],
            "request_id": r["request_id"],
            "step": 1 if r["phase"] == "generate" else 2,
            **want,
        }
        assert got[k] == expect, f"A2 {name}: {got[k]} != {expect}"


# --------------------------------------------------------------------------- exit 2


def _err(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def test_exit_2_on_missing_trials(tmp_path: Path) -> None:
    """A2: a run directory without ``trials.jsonl`` exits 2 and names it."""
    run = build_run(tmp_path, fixture_rows()[:3])
    (run / "trials.jsonl").unlink()
    proc = run_script(run, "--json")
    assert proc.returncode == 2, _err(proc)
    assert "trials.jsonl" in _err(proc)


def test_exit_2_on_missing_run_dir(tmp_path: Path) -> None:
    """A2: an unreadable (absent) run directory exits 2 and names it."""
    missing = tmp_path / "no-such-run"
    proc = run_script(missing)
    assert proc.returncode == 2, _err(proc)
    assert "no-such-run" in _err(proc)


def test_exit_2_on_missing_session_file(tmp_path: Path) -> None:
    """A2: a trial whose ``session_log`` file is absent under ``sessions/`` exits 2 and names the file."""
    rows = fixture_rows()
    run = build_run(tmp_path, rows)
    gone = "20261022-204803-1975518.jsonl"
    assert (run / "sessions" / gone).exists(), "premise"
    (run / "sessions" / gone).unlink()
    proc = run_script(run, "--json")
    assert proc.returncode == 2, _err(proc)
    assert gone in _err(proc)


@pytest.mark.parametrize("where", ["trials", "session"])
def test_exit_2_on_malformed_json(tmp_path: Path, where: str) -> None:
    """A2: malformed JSON in ``trials.jsonl`` or in a session file exits 2 and names that file."""
    run = build_run(tmp_path, fixture_rows())
    bad = (
        run / "trials.jsonl"
        if where == "trials"
        else run / "sessions" / "20261022-210810-1975518.jsonl"
    )
    assert bad.exists(), "premise"
    bad.write_text(bad.read_text() + '{"kind": "llm.commit", "data": {\n')
    proc = run_script(run, "--json")
    assert proc.returncode == 2, _err(proc)
    assert bad.name in _err(proc)


def test_exit_2_on_non_utf8_session_file(tmp_path: Path) -> None:
    """A2 (review finding): a session file whose bytes are not valid UTF-8 is unreadable input. The script exits 2
    and names the file; it does not crash with a ``UnicodeDecodeError`` traceback (exit 1)."""
    run = build_run(tmp_path, fixture_rows())
    bad = run / "sessions" / "20261022-210810-1975518.jsonl"
    assert bad.exists(), "premise"
    bad.write_bytes(b"\xff\xfe{}\n")
    proc = run_script(run, "--json")
    assert proc.returncode == 2, _err(proc)
    assert bad.name in _err(proc)
    assert "Traceback" not in proc.stderr, proc.stderr
