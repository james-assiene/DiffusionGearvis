"""T-0098c criterion 8: the decision of a real UI-block A/B run, re-derived from ``trials.jsonl`` alone (``gpu``,
``eval``). Pending until ``/implement T-0098c`` started (marker removed then).

Run through ``/launch`` on the 5090 box (the T-0098c locked script runs it after the harness)::

    T0098C_RUN_DIR=logs/<the run directory> uv run pytest tests/eval/test_toolcall_ui_prompt_ab.py -q -m gpu -rA

Without ``T0098C_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run
directory -- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046).

**The oracle is this module, not the harness.** ``derive`` re-reads every trial from its raw commits:

- T-0098's criterion-5 reading (routed, spoken, valid, exact, thought_open, anomaly) through T-0098's own oracle
  ``read_commits`` (``tests/eval/test_toolcall_prompt_ab.py``, imported read-only), over the same 53-case
  denominators (Q12 trials never enter P1-P7; P5 is over the 65 no-tool + plain trials);
- *shown* (criterion 5): a fresh real ``UiGate()`` per trial gates every ``ui`` commit in commit order with
  ``gate(text, turn_id=request_id, canvas_idx=commit.canvas_idx, final=False)`` -- the session's ``_gate_surface``
  call -- whatever ``thought_open`` says; shown = at least one call returned a surface. *placement failure* = a
  ``ui`` commit with ``thought_open`` true, or a ``thought`` commit containing ``llm_client.UI_OPEN``;
- P8-P11, R1 / R2c / R3, the void rule and the winner order C3, C5, C4 of criteria 6-7, in ``fractions.Fraction``.

The case kinds, ground truths and ``expect_surface`` come from the pinned ``cases.jsonl``, J10's plain list and
``surface_placement_cases.jsonl`` by case id, never from the report. Then ``report.json`` must agree on every
P-condition of the three candidates, the three readings, the void arms, the winner and each arm's Q12 shown counts.

Report surface read (as ``tests/unit/test_t0098c_ui_prompt_ab.py`` states it): ``report["candidates"][c][p]``,
``report["readings"][r]``, ``report["void_arms"]``, ``report["winner"]`` (``None`` / ``"no change"`` for no change),
``report["arms"][a]["q12_positive_shown"]`` / ``["q12_negative_shown"]`` (``{"k", "n"}``).

Regime (state it beside every number this prints): T-0098c *Regime* -- lean; vLLM v0.29.0; DiffusionGemma NVFP4 rev
``2ea8372``; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on;
``max_num_seqs 3``; tools ``native:bbfc53ccdd80``, repair on; dev split, 60 cases x 7 arms x n = 5; no seed;
concurrency 1; typed text, no Session; UiGate and vendored A2UI v0.9.1 schema at the run commit.
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

from local_astra.conductor.llm_client import UI_OPEN
from local_astra.conductor.ui_gate import UiGate
from local_astra.toolcall import score
from local_astra.toolcall import tools as toolset
from tests.eval.test_toolcall_prompt_ab import case_table, read_commits

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0098C_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
Q12 = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
ARMS = ("A0", "D1", "D3", "D4", "C3", "C5", "C4")
CANDIDATES = ("C3", "C5", "C4")
PS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11")
REPEATS = 5


# --- the oracle -------------------------------------------------------------------------------------------------


def case_table_c() -> dict[str, dict[str, Any]]:
    """case id -> {kind, tool, gt, expect_surface}: T-0098's 53 plus the 7 Q12 cases (kind ``q12``)."""
    out: dict[str, dict[str, Any]] = {
        k: {**v, "expect_surface": None} for k, v in case_table().items()
    }
    for line in Q12.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = {
                "kind": "q12",
                "tool": None,
                "gt": None,
                "expect_surface": bool(r["expect_surface"]),
            }
    return out


def read_surface(commits: Sequence[Mapping[str, Any]], request_id: str) -> dict[str, Any]:
    """Criterion 5 with a fresh real gate, as ``session._gate_surface`` calls it."""
    gate = UiGate()
    gated = emitted = 0
    for c in commits:
        if c.get("segment") != "ui":
            continue
        result = gate.gate(
            str(c.get("text", "")),
            turn_id=request_id,
            canvas_idx=int(c.get("canvas_idx", 0)),
            final=False,
        )
        gated += 1
        emitted += result.surface is not None
    placement = any(c.get("segment") == "ui" and c.get("thought_open") for c in commits) or any(
        c.get("segment") == "thought" and UI_OPEN in str(c.get("text", "")) for c in commits
    )
    return {"shown": emitted > 0, "placement": placement, "gated": gated, "emitted": emitted}


def _rate(k: int, n: int) -> Fraction:
    return Fraction(k, n) if n else Fraction(0)


def derive(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every P-condition, R1/R2c/R3, the void arms, the winner and the Q12 shown counts, from the trial lines alone."""
    cases = case_table_c()
    tool_set = toolset.load()
    lines = list(trials)
    per: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for t in lines:
        commits = t.get("commits") or []
        per[str(t["arm"])].append(
            (
                dict(t),
                cases[str(t["case_id"])],
                read_commits(commits),
                read_surface(commits, str(t["request_id"])),
            )
        )
    keys = [(str(t["case_id"]), str(t["arm"]), int(t["repeat"])) for t in lines]
    complete = len(keys) == len(set(keys)) and set(keys) == {
        (c, a, r) for c in cases for a in ARMS for r in range(REPEATS)
    }

    stats: dict[str, dict[str, Any]] = {}
    for arm, rows in per.items():
        errored = sum(1 for t, _, _, _ in rows if t.get("error") not in (None, ""))
        tool_rows = [x for x in rows if x[1]["kind"] == "tool"]
        routed = [x for x in tool_rows if x[2]["routed"]]
        type_routed = [x for x in routed if x[1]["tool"] == "type_at"]
        csk = [x for x in routed if x[1]["tool"] in ("click_at", "scroll", "press_key")]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t, _c, r, _s in type_routed:
            by_case[str(t["case_id"])].append(r)
        majority = sum(1 for rs in by_case.values() if 2 * sum(r["spoken"] for r in rs) > len(rs))
        valid = sum(
            1
            for _, _, r, _ in routed
            if r["call"] is not None and score.schema_valid(r["call"], tool_set)
        )
        exact = sum(
            1 for _, c, r, _ in routed if r["call"] is not None and score.exact(r["call"], c["gt"])
        )
        q12 = [x for x in rows if x[1]["kind"] == "q12"]
        positive = [x for x in q12 if x[1]["expect_surface"]]
        negative = [x for x in q12 if not x[1]["expect_surface"]]
        per_case: dict[str, int] = defaultdict(int)
        for t, _c, _r, s in positive:
            per_case[str(t["case_id"])] += int(s["shown"])
        no_belong = [x for x in rows if x[1]["kind"] in ("no_tool", "plain")] + negative
        stats[arm] = {
            "void": Fraction(errored, len(rows)) > Fraction(2, 100),
            "type_spoken": _rate(sum(r["spoken"] for _, _, r, _ in type_routed), len(type_routed)),
            "type_majority": majority,
            "csk_spoken": _rate(sum(r["spoken"] for _, _, r, _ in csk), len(csk)),
            "recall": _rate(len(routed), len(tool_rows)),
            "type_recall": _rate(
                len(type_routed), len([x for x in tool_rows if x[1]["tool"] == "type_at"])
            ),
            "valid": _rate(valid, len(routed)),
            "exact": _rate(exact, len(routed)),
            "false": sum(
                1 for _, c, r, _ in rows if c["kind"] in ("no_tool", "plain") and r["routed"]
            ),
            "anomalies": sum(1 for _, _, r, _ in rows if r["anomaly"]),
            "thought_open": sum(1 for _, _, r, _ in tool_rows if r["thought_open"]),
            "pos_shown": sum(s["shown"] for _, _, _, s in positive),
            "pos_n": len(positive),
            "neg_shown": sum(s["shown"] for _, _, _, s in negative),
            "neg_n": len(negative),
            "per_case": dict(per_case),
            "surf_no_belong": sum(s["shown"] for _, _, _, s in no_belong),
            "surf_tool": sum(s["shown"] for _, _, _, s in tool_rows),
            "placement_pos": sum(s["placement"] for _, _, _, s in positive),
            "q12_routed": sum(r["routed"] for _, _, r, _ in q12),
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
            "P8": s["pos_shown"] >= a0["pos_shown"] - 2
            and all(s["per_case"].get(cid, 0) >= 3 for cid, k in a0["per_case"].items() if k >= 3),
            "P9": s["surf_no_belong"] <= a0["surf_no_belong"] + 1
            and s["surf_tool"] <= a0["surf_tool"] + 2,
            "P10": s["placement_pos"] <= a0["placement_pos"] + 1,
            "P11": s["q12_routed"] <= a0["q12_routed"] + 2,
        }
        p["passes"] = all(p[k] for k in PS) and not s["void"] and not a0["void"]
        candidates[c] = p
    readings = {
        "R1": a0["type_spoken"] <= Fraction(3, 10),
        "R2c": stats["D1"]["type_spoken"] >= Fraction(7, 10),
        "R3": a0["pos_shown"] >= 14,
    }
    winner: str | None = None
    if readings["R1"] and readings["R3"] and not a0["void"] and complete:
        winner = next((c for c in CANDIDATES if candidates[c]["passes"]), None)
    return {
        "candidates": candidates,
        "readings": readings,
        "void_arms": sorted(a for a in ARMS if stats.get(a, {}).get("void")),
        "winner": winner,
        "complete": complete,
        "shown": {
            a: {
                "positive": (stats[a]["pos_shown"], stats[a]["pos_n"]),
                "negative": (stats[a]["neg_shown"], stats[a]["neg_n"]),
            }
            for a in stats
        },
    }


# --- the rows ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(f"{ENV} is not set: no T-0098c run directory to read")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("trials.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"T-0098c criterion 6: the run directory has no {name}"
    return path


@pytest.fixture(scope="module")
def trials(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def stored(run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return report


def test_the_design_ran_whole(run_dir: Path, trials: list[dict[str, Any]]) -> None:
    """Criteria 1, 3 and 10: the manifest names table T-0098c and a rule; 2100 lines, each (case, arm, repeat) exactly
    once over the 53 dev + plain cases and the 7 Q12 cases, no test-split case, A0's system sha the 4b0bd92 one, every
    line with a request id and every commit with the four recorded fields."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("table") == "T-0098c" and manifest.get("rule"), (
        "criterion 1: the manifest's table and rule"
    )
    cases = case_table_c()
    assert len(cases) == 60
    assert len(trials) == 60 * 7 * REPEATS == 2100, len(trials)
    keys = {(t["case_id"], t["arm"], t["repeat"]) for t in trials}
    assert keys == {(c, a, r) for c in cases for a in ARMS for r in range(REPEATS)}
    assert {t["system_sha256"] for t in trials if t["arm"] == "A0"} == {A0_SHA256}
    for t in trials:
        assert t.get("request_id"), (t["case_id"], t["arm"], "criterion 5: the gate's turn id")
        for c in t.get("commits") or []:
            assert {"segment", "text", "thought_open", "canvas_idx"} <= set(c), (t["case_id"], c)


def test_report_agrees_with_the_raw_trials(
    trials: list[dict[str, Any]], stored: dict[str, Any]
) -> None:
    """Criterion 8: shown, placement, every P-condition of C3, C5 and C4, R1, R2c, R3, the void arms, the winner and the
    per-arm Q12 shown counts, re-derived from ``trials.jsonl`` with this module's own ``UiGate`` calls, equal what
    ``report.json`` says."""
    want = derive(trials)
    for c in CANDIDATES:
        for p in (*PS, "passes"):
            assert stored["candidates"][c][p] is want["candidates"][c][p], (
                f"{c} {p}: report {stored['candidates'][c][p]!r}, raw trials {want['candidates'][c][p]!r}"
            )
    for r in ("R1", "R2c", "R3"):
        assert stored["readings"][r] is want["readings"][r], r
    assert sorted(stored["void_arms"]) == want["void_arms"]
    for arm, counts in want["shown"].items():
        k, n = counts["positive"]
        assert stored["arms"][arm]["q12_positive_shown"] == {"k": k, "n": n}, arm
        k, n = counts["negative"]
        assert stored["arms"][arm]["q12_negative_shown"] == {"k": k, "n": n}, arm
    got = None if stored["winner"] in (None, "no change") else stored["winner"]
    assert got == want["winner"], (
        f"winner: report {stored['winner']!r}, raw trials {want['winner']!r}"
    )
    print(f"\nT-0098c decision (from raw trials): {want['winner']}; readings {want['readings']}")
