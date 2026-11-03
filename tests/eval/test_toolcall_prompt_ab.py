"""T-0098 criterion 8: the decision of a real A/B run, re-derived from ``trials.jsonl`` alone (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box (``scripts/locked/t0098_prompt_ab.sh`` runs it after the harness)::

    T0098_RUN_DIR=logs/t0098-prompt-ab-<sha> uv run pytest tests/eval/test_toolcall_prompt_ab.py -q -m gpu -rA

Without ``T0098_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run
directory -- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046: every
``tests/eval`` row is ``gpu``).

**The oracle is this module, not the harness.** ``derive`` below re-reads every trial from its raw commits with the
definitions of criterion 5 -- routed (a ``tool`` commit outside an open thought, J8), spoken
(``drop_special_literals`` of the joined ``thought_open == False`` answer commits -- ruling N4, the session's
``_AnswerCleaner`` -- stripped, non-empty, on a routed trial), valid
(``native.parse_call`` + ``score.schema_valid``) and exact (``score.exact``) over routed trials, false tools, voiced
anomalies, thought_open -- and applies P1-P7, R1/R2, the void rule and the winner order of criterion 7 with
``fractions.Fraction``, stdlib only beyond the three scorer functions the task names as the definitions. The case
kinds and ground truths come from the pinned ``cases.jsonl`` by case id and from J10's plain list, never from the
report. Then the stored ``report.json`` must agree on every P-condition, both readings, the void arms and the winner.

Report surface read (as ``tests/unit/test_t0098_prompt_ab.py`` states it): ``report["candidates"][c][p]``,
``report["readings"][r]``, ``report["void_arms"]``, ``report["winner"]`` (``None`` or ``"no change"`` for no change).

Readings this module makes where the task is silent (named in the T-0098 test-author report): rates of ``spoken`` are
over routed trials (criterion 6: "spoken / routed"), P2 included; route recall's denominator is every trial of the
tool cases (200 pooled, 50 for ``type_at``), an errored trial counting as unrouted; valid/exact are over routed trials.

Regime (state it beside every number this prints): T-0098 *Regime* -- lean; vLLM v0.29.0; DiffusionGemma NVFP4 rev
``2ea8372``; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on;
``max_num_seqs 3``; tools ``native:bbfc53ccdd80``, repair on; dev split, 53 cases x 6 arms x n = 5; no seed;
concurrency 1; typed text, no Session.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import native, score
from local_astra.toolcall import tools as toolset
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0098_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
ARMS = ("A0", "A1", "D1", "D2", "C2", "C1")
CANDIDATES = ("C2", "C1")
PS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
REPEATS = 5
TOOL_TAG = "<|tool_call>"
PLAIN_IDS = ("plain-01", "plain-02", "plain-03", "plain-04", "plain-05")


# --- the oracle -------------------------------------------------------------------------------------------------


def case_table() -> dict[str, dict[str, Any]]:
    """case id -> {kind, tool, gt}: the dev split of the pinned file plus J10's plain questions."""
    out: dict[str, dict[str, Any]] = {}
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") == "dev":
            out[r["id"]] = {"kind": r["kind"], "tool": r.get("tool"), "gt": r.get("gt")}
    for cid in PLAIN_IDS:
        out[cid] = {"kind": "plain", "tool": None, "gt": None}
    return out


def read_commits(commits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Criterion 5 on one trial's commits."""
    tools = [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]
    routed = bool(tools)
    prefix = drop_special_literals(
        "".join(
            str(c.get("text", ""))
            for c in commits
            if c.get("segment") == "answer" and not c.get("thought_open")
        )
    ).strip()
    spoken = routed and prefix != ""
    thoughts = [c for c in commits if c.get("segment") == "thought"]
    thought_open = bool(thoughts) and bool(thoughts[-1].get("thought_open"))
    anomaly = spoken and ("call:" in prefix or any(lit in prefix for lit in SPECIAL_TOKENS))
    call: dict[str, Any] | None = None
    if routed:
        text = str(tools[0].get("text", ""))
        text = text[len(TOOL_TAG) :] if text.startswith(TOOL_TAG) else text
        parsed = native.parse_call(text)
        call = None if isinstance(parsed, score.ParseFailure) else parsed
    return {
        "routed": routed,
        "spoken": spoken,
        "thought_open": thought_open,
        "anomaly": anomaly,
        "call": call,
    }


def _rate(k: int, n: int) -> Fraction:
    return Fraction(k, n) if n else Fraction(0)


def derive(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every P-condition, R1/R2, the void arms and the winner, from the trial lines alone."""
    cases = case_table()
    tool_set = toolset.load()
    per: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for t in trials:
        per[str(t["arm"])].append(
            (dict(t), cases[str(t["case_id"])], read_commits(t.get("commits") or []))
        )

    stats: dict[str, dict[str, Any]] = {}
    for arm, rows in per.items():
        errored = sum(1 for t, _, _ in rows if t.get("error") not in (None, ""))
        tool_rows = [(t, c, r) for t, c, r in rows if c["kind"] == "tool"]
        routed = [(t, c, r) for t, c, r in tool_rows if r["routed"]]

        def of(tool: str | None, rows_: list[Any] = tool_rows) -> list[Any]:
            return [x for x in rows_ if tool is None or x[1]["tool"] == tool]

        type_routed = of("type_at", routed)
        csk = [x for x in routed if x[1]["tool"] in ("click_at", "scroll", "press_key")]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t, _c, r in of("type_at", routed):
            by_case[str(t["case_id"])].append(r)
        majority = sum(1 for rs in by_case.values() if 2 * sum(r["spoken"] for r in rs) > len(rs))
        valid = sum(
            1
            for _, _, r in routed
            if r["call"] is not None and score.schema_valid(r["call"], tool_set)
        )
        exact = sum(
            1 for _, c, r in routed if r["call"] is not None and score.exact(r["call"], c["gt"])
        )
        stats[arm] = {
            "void": Fraction(errored, len(rows)) > Fraction(2, 100),
            "type_spoken": _rate(sum(r["spoken"] for _, _, r in type_routed), len(type_routed)),
            "type_majority": majority,
            "csk_spoken": _rate(sum(r["spoken"] for _, _, r in csk), len(csk)),
            "recall": _rate(len(routed), len(tool_rows)),
            "type_recall": _rate(len(type_routed), len(of("type_at"))),
            "valid": _rate(valid, len(routed)),
            "exact": _rate(exact, len(routed)),
            "false": sum(1 for _, c, r in rows if c["kind"] != "tool" and r["routed"]),
            "anomalies": sum(1 for _, _, r in rows if r["anomaly"]),
            "thought_open": sum(1 for _, _, r in tool_rows if r["thought_open"]),
        }

    a0 = stats["A0"]
    candidates: dict[str, dict[str, bool]] = {}
    for c in CANDIDATES:
        s = stats[c]
        p = {
            "P1": s["type_spoken"] >= Fraction(8, 10) and s["type_majority"] >= 8,
            "P2": s["csk_spoken"] >= a0["csk_spoken"] - Fraction(5, 100),
            "P3": s["recall"] >= a0["recall"] - Fraction(2, 100)
            and s["type_recall"] >= a0["type_recall"] - Fraction(4, 100),
            "P4": s["valid"] >= a0["valid"] - Fraction(2, 100)
            and s["exact"] >= a0["exact"] - Fraction(3, 100),
            "P5": s["false"] <= 5 and s["false"] <= a0["false"] + 2,
            "P6": s["anomalies"] == 0,
            "P7": s["thought_open"] <= a0["thought_open"] + 4,
        }
        p["passes"] = all(p[k] for k in PS) and not s["void"] and not a0["void"]
        candidates[c] = p
    readings = {
        "R1": a0["type_spoken"] <= Fraction(3, 10),
        "R2": stats["A1"]["type_spoken"] >= Fraction(7, 10),
    }
    winner: str | None = None
    if readings["R1"]:
        winner = next((c for c in CANDIDATES if candidates[c]["passes"]), None)
    return {
        "candidates": candidates,
        "readings": readings,
        "void_arms": sorted(a for a in ARMS if stats.get(a, {}).get("void")),
        "winner": winner,
    }


# --- the rows ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(
            f"{ENV} is not set: no T-0098 run directory (logs/t0098-prompt-ab-<sha>/) to read"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("trials.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"T-0098 criteria 4/6: the run directory has no {name}"
    return path


@pytest.fixture(scope="module")
def trials(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def stored(run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return report


def test_the_design_ran_whole(trials: list[dict[str, Any]]) -> None:
    """Criteria 3, 4 and 10: 1590 lines, each (case, arm, repeat) exactly once over the 53 dev + plain cases, no
    test-split case, A0's system sha the 4b0bd92 one, every commit with the four recorded fields."""
    cases = case_table()
    assert len(trials) == 53 * 6 * REPEATS == 1590, len(trials)
    keys = {(t["case_id"], t["arm"], t["repeat"]) for t in trials}
    assert keys == {(c, a, r) for c in cases for a in ARMS for r in range(REPEATS)}
    assert {t["system_sha256"] for t in trials if t["arm"] == "A0"} == {A0_SHA256}
    for t in trials:
        for c in t.get("commits") or []:
            assert {"segment", "text", "thought_open", "canvas_idx"} <= set(c), (t["case_id"], c)
        assert isinstance(t.get("token_ids"), list)


def test_report_agrees_with_the_raw_trials(
    trials: list[dict[str, Any]], stored: dict[str, Any]
) -> None:
    """Criterion 8: every P-condition of both candidates, R1, R2, the void arms and the winner, re-derived from
    ``trials.jsonl`` by this module's oracle, equal what ``report.json`` says."""
    want = derive(trials)
    for c in CANDIDATES:
        for p in (*PS, "passes"):
            assert stored["candidates"][c][p] is want["candidates"][c][p], (
                f"{c} {p}: report {stored['candidates'][c][p]!r}, raw trials {want['candidates'][c][p]!r}"
            )
    for r in ("R1", "R2"):
        assert stored["readings"][r] is want["readings"][r], r
    assert sorted(stored["void_arms"]) == want["void_arms"]
    got = None if stored["winner"] in (None, "no change") else stored["winner"]
    assert got == want["winner"], (
        f"winner: report {stored['winner']!r}, raw trials {want['winner']!r}"
    )
    print(f"\nT-0098 decision (from raw trials): {want}")
