"""T-0103 criterion 8: the outcome of the C3 confirmation run, re-derived from ``trials.jsonl`` alone (``gpu``,
``eval``).

Run through ``/launch`` on the 5090 box (the T-0103 locked script runs it after the harness)::

    T0103_RUN_DIR=logs/t0103-c3-confirm-<sha7> uv run pytest tests/eval/test_toolcall_c3_confirm.py -q -m gpu -rA

Without ``T0103_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run directory
-- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046).

**The oracle is this module, not the harness.** ``derive`` re-reads every trial from its raw commits:

- T-0098's criterion-5 reading (routed, spoken, valid, exact, thought_open, anomaly) through T-0098's ``read_commits``
  and T-0098c's surface reading (a fresh real ``UiGate()`` per trial, ``final=False``, the request id as turn id;
  shown = any call returned a surface; placement = a ``ui`` commit flagged ``thought_open`` or a thought holding
  ``UI_OPEN``) through T-0098c's ``read_surface`` -- both imported read-only;
- the case kinds, ground truths and ``expect_surface`` from the pinned files by case id (T-0098c's ``case_table_c``),
  never from the report;
- the schedule of *Design* built here by hand (repeats outermost; within a repeat the scheduled cases in index order;
  per (case, repeat) A0 first iff (case_index + repeat) is even);
- P1-P7 and P9 verbatim from T-0098c over the same denominators (P5's 65, P9's 80 and 200; P6 over every trial,
  N11), P8 / P10 / P11 by reading S, R1, R3, the void rule (> 2 % of an arm's trials) and the three outcomes of
  criterion 5, in ``fractions.Fraction``. Every threshold is typed out from docs/tasks/T-0103.md, not imported.

``report.json`` must then agree on every P-condition of C3, R1 and R3, the void arms, completeness, each arm's Q12
shown and placement counts and per-case k/40, the outcome (its form, and the failing Ps it names) and the winner.

Report surface read (as ``tests/unit/test_t0103_c3_confirm.py`` states it): ``candidates["C3"][p]``, ``readings``,
``void_arms``, ``complete``, ``winner``, ``outcome``, ``arms[a]["q12_positive_shown"]``, ``["q12_negative_shown"]``,
``["placement_failures_q12_positive"]``, ``["q12_per_case"]``.

Regime (state it beside every number this prints): T-0103 *Regime* -- lean; vLLM v0.29.0; DiffusionGemma NVFP4 rev
``2ea8372``; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on;
``max_num_seqs 3``; tools ``native:bbfc53ccdd80``, repair on; dev split, 60 cases, arms A0 and C3, n = 40 on the 4 Q12
positives and n = 5 on the other 56 (880 requests); no seed; concurrency 1; typed text, no Session; UiGate and the
vendored A2UI v0.9.1 schema at the run commit.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import score
from local_astra.toolcall import tools as toolset
from tests.eval.test_toolcall_prompt_ab import PLAIN_IDS, read_commits
from tests.eval.test_toolcall_ui_prompt_ab import case_table_c, read_surface

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0103_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
Q12 = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
C3_SHA256 = "72c3b92bcff2e5924901e42475342826ffc38efefa116f66138f8aa43bad9308"
ARMS = ("A0", "C3")
PS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11")
POSITIVE_REPEATS, OTHER_REPEATS = 40, 5
N_REQUESTS = 880


# --- the oracle -------------------------------------------------------------------------------------------------


def run_order() -> list[str]:
    """The 60 case ids in run order: dev tool cases, dev no-tool cases (file order), J10's plain questions, then the
    Q12 file in file order (T-0098c criterion 3; T-0103 *Design*)."""
    rows = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev = [r for r in rows if r.get("split") == "dev"]
    q12 = [json.loads(x) for x in Q12.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [
        *(r["id"] for r in dev if r["kind"] == "tool"),
        *(r["id"] for r in dev if r["kind"] == "no_tool"),
        *PLAIN_IDS,
        *(r["id"] for r in q12),
    ]


def positives() -> tuple[str, ...]:
    return tuple(
        cid for cid, c in case_table_c().items() if c["kind"] == "q12" and c["expect_surface"]
    )


def expected_schedule() -> list[tuple[int, str, int, str]]:
    """(case_index, case_id, repeat, arm) in request order, by hand from *Design*."""
    order = run_order()
    pos = set(positives())
    out: list[tuple[int, str, int, str]] = []
    for r in range(POSITIVE_REPEATS):
        for i, cid in enumerate(order):
            if r >= (POSITIVE_REPEATS if cid in pos else OTHER_REPEATS):
                continue
            first, second = ("A0", "C3") if (i + r) % 2 == 0 else ("C3", "A0")
            out += [(i, cid, r, first), (i, cid, r, second)]
    return out


def _rate(k: int, n: int) -> Fraction:
    return Fraction(k, n) if n else Fraction(0)


def derive(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every P-condition of C3, R1 and R3, the void arms, completeness, the outcome and the per-arm UI counts, from
    the trial lines alone."""
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
    expected = {(cid, arm, r) for _, cid, r, arm in expected_schedule()}
    complete = len(keys) == len(set(keys)) == N_REQUESTS and set(keys) == expected

    stats: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = per.get(arm, [])
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
        per_case: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for t, _c, _r, s in positive:
            row = per_case[str(t["case_id"])]
            row[0] += int(s["shown"])
            row[1] += 1
        no_belong = [x for x in rows if x[1]["kind"] in ("no_tool", "plain")] + negative
        stats[arm] = {
            "trials": len(rows),
            "errored": errored,
            "void": bool(rows) and Fraction(errored, len(rows)) > Fraction(2, 100),
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
            "false_n": sum(1 for _, c, _, _ in rows if c["kind"] in ("no_tool", "plain")),
            "anomalies": sum(1 for _, _, r, _ in rows if r["anomaly"]),
            "thought_open": sum(1 for _, _, r, _ in tool_rows if r["thought_open"]),
            "pos_shown": sum(s["shown"] for _, _, _, s in positive),
            "pos_n": len(positive),
            "neg_shown": sum(s["shown"] for _, _, _, s in negative),
            "neg_n": len(negative),
            "per_case": {cid: (k, n) for cid, (k, n) in per_case.items()},
            "surf_no_belong": sum(s["shown"] for _, _, _, s in no_belong),
            "no_belong_n": len(no_belong),
            "surf_tool": sum(s["shown"] for _, _, _, s in tool_rows),
            "placement_pos": sum(s["placement"] for _, _, _, s in positive),
            "q12_routed": sum(r["routed"] for _, _, r, _ in q12),
            "q12_n": len(q12),
        }

    a0, s = stats["A0"], stats["C3"]
    protected = {cid for cid, (k, _n) in a0["per_case"].items() if k >= 24}  # 3/5 of 40
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
        # reading S: A0 - 1/10 x 160 = A0 - 16; a protected case fails at C3 <= 23 of 40
        "P8": s["pos_shown"] >= a0["pos_shown"] - 16
        and all(s["per_case"].get(cid, (0, 0))[0] >= 24 for cid in protected),
        # verbatim: of 80 and of 200
        "P9": s["surf_no_belong"] <= a0["surf_no_belong"] + 1
        and s["surf_tool"] <= a0["surf_tool"] + 2,
        # reading S: A0 + 1/20 x 160 = A0 + 8; A0 + 2/35 x 175 = A0 + 10
        "P10": s["placement_pos"] <= a0["placement_pos"] + 8,
        "P11": s["q12_routed"] <= a0["q12_routed"] + 10,
    }
    readings = {"R1": a0["type_spoken"] <= Fraction(3, 10), "R3": a0["pos_shown"] >= 112}
    void_arms = sorted(a for a in ARMS if stats[a]["void"])
    failing = sorted((k for k in PS if not p[k]), key=PS.index)
    if not complete or void_arms or not readings["R1"] or not readings["R3"]:
        outcome = "inconclusive"
    elif failing:
        outcome = "not confirmed"
    else:
        outcome = "confirmed"
    return {
        "candidate": p,
        "readings": readings,
        "void_arms": void_arms,
        "complete": complete,
        "outcome": outcome,
        "failing": failing,
        "winner": "C3" if outcome == "confirmed" else "no change",
        "stats": stats,
    }


def outcome_kind(text: str) -> str:
    if text.startswith("C3 confirmed"):
        return "confirmed"
    if text.startswith("C3 not confirmed: fails "):
        return "not confirmed"
    if text.startswith("inconclusive: "):
        return "inconclusive"
    raise AssertionError(f"criterion 5: outcome {text!r} is none of the three pre-registered forms")


# --- the rows ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(f"{ENV} is not set: no T-0103 run directory to read")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("trials.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"T-0103 criterion 7: the run directory has no {name}"
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
    """Criteria 1, 4 and 10: the manifest names table T-0103 and rule H-T103; 880 lines, in exactly the schedule of
    *Design* (repeats outermost, scheduled cases in index order, A0 first iff (case_index + repeat) is even); request
    ids ``t0103-r{repeat}-{case_id}-{arm}``; A0's system sha ``3605c012...`` and C3's ``72c3b92b...`` on every line; no
    test-split case; every commit with the four recorded fields."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("table") == "T-0103" and manifest.get("rule") == "H-T103", (
        "criterion 1: the manifest's table and rule"
    )
    want = expected_schedule()
    assert len(want) == N_REQUESTS == 4 * 40 * 2 + 56 * 5 * 2
    assert len(trials) == N_REQUESTS, len(trials)
    got = [
        (int(t["case_index"]), str(t["case_id"]), int(t["repeat"]), str(t["arm"])) for t in trials
    ]
    assert got == want, "criterion 4: the lines are not in the schedule of *Design*"
    assert [t["request_id"] for t in trials] == [
        f"t0103-r{r}-{cid}-{arm}" for _, cid, r, arm in want
    ]
    shas = {"A0": A0_SHA256, "C3": C3_SHA256}
    for t in trials:
        assert t["system_sha256"] == shas[t["arm"]], (t["case_id"], t["arm"], t["repeat"])
        for c in t.get("commits") or []:
            assert {"segment", "text", "thought_open", "canvas_idx"} <= set(c), (t["case_id"], c)
    assert set(run_order()) == set(case_table_c()), "criterion 10: dev and J10 cases only"


def test_report_agrees_with_the_raw_trials(
    trials: list[dict[str, Any]], stored: dict[str, Any]
) -> None:
    """Criterion 8: shown, placement, P1-P11 of C3, R1, R3, the void arms, completeness, the per-arm Q12 counts (k/160,
    k/15, per positive case k/40), the outcome (form and named Ps) and the winner, re-derived from ``trials.jsonl`` with
    this module's own reading and ``UiGate`` calls, equal what ``report.json`` says."""
    want = derive(trials)
    for p in PS:
        assert stored["candidates"]["C3"][p] is want["candidate"][p], (
            f"C3 {p}: report {stored['candidates']['C3'][p]!r}, raw trials {want['candidate'][p]!r}"
        )
    assert set(stored["readings"]) == {"R1", "R3"}
    for r in ("R1", "R3"):
        assert stored["readings"][r] is want["readings"][r], r
    assert sorted(stored["void_arms"]) == want["void_arms"]
    assert stored["complete"] is want["complete"]
    for arm in ARMS:
        st, a = want["stats"][arm], stored["arms"][arm]
        assert a["q12_positive_shown"] == {"k": st["pos_shown"], "n": st["pos_n"]}, arm
        assert a["q12_negative_shown"] == {"k": st["neg_shown"], "n": st["neg_n"]}, arm
        assert a["placement_failures_q12_positive"]["k"] == st["placement_pos"], arm
        for cid, (k, n) in st["per_case"].items():
            assert a["q12_per_case"][cid] == {"k": k, "n": n}, (arm, cid)
    text = str(stored["outcome"])
    assert outcome_kind(text) == want["outcome"], (
        f"outcome: report {text!r}, raw trials {want['outcome']!r}"
    )
    if want["outcome"] == "not confirmed":
        named = set(re.findall(r"\bP\d+\b", text.split("fails", 1)[1]))
        assert named == set(want["failing"]), (text, want["failing"])
    assert stored["winner"] == want["winner"], (stored["winner"], want["winner"])
    st0, st3 = want["stats"]["A0"], want["stats"]["C3"]
    print(
        f"\nT-0103 outcome (from raw trials): {want['outcome']}"
        + (f" (fails {', '.join(want['failing'])})" if want["failing"] else "")
        + f"; readings {want['readings']}; void {want['void_arms']}; Q12 positive shown A0 "
        f"{st0['pos_shown']}/{st0['pos_n']}, C3 {st3['pos_shown']}/{st3['pos_n']}"
    )
