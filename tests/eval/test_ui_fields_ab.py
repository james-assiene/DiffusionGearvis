"""T-0108a criterion 8: the outcome of the required-fields A/B, re-derived from ``trials.jsonl`` alone with the live
``UiGate`` (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box (the T-0108a locked script runs it after the harness)::

    T0108A_RUN_DIR=logs/t0108a-ui-fields-ab-<sha7> uv run pytest tests/eval/test_ui_fields_ab.py -q -m gpu -rA

Without ``T0108A_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run
directory -- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046).

**The T-0103 lesson (criterion 8):** this module re-gates every trial with the ``UiGate`` of the working tree, so it
reproduces the report only at the run commit. Before anything else it compares the manifest's sha256 of
``local_astra/conductor/ui_gate.py``, ``local_astra/ui/bracket_repair.py`` and ``local_astra/ui/a2ui_schema.py`` with
the working tree, and skips with the reason "re-gates with the live UiGate; reproduces only at the run commit" when
any differs.

**The oracle is this module, not the harness.** ``derive`` re-reads every trial from its raw commits:

- T-0098's criterion-5 reading through ``read_commits`` and T-0098c's surface reading (shown, placement) through
  ``read_surface`` -- both imported read-only from the earlier eval modules;
- the complete-card reading of criterion 4, applied here: a fresh real ``UiGate()`` per trial gates every ``ui``
  commit in order with ``final=False`` and the request id as turn id; ``last_ok`` = the last call returned a surface;
  ``pruned_last`` = ``invalid_component_ids(iter_components(_repair(raw).messages)) | _repair(raw).dropped`` on the
  last canvas; complete = both; no ``ui`` commit is not complete;
- the case kinds and ``expect_surface`` from the pinned files by case id, never from the report;
- the schedule of *Design* by hand (repeats outermost; scheduled cases in index order; arms from offset
  (case_index + repeat) mod 3 of A0 K1 K2);
- Q1, Q2, U1, U2, P3-P7, P9-P11, R3, R4, the void rule and the three outcomes of criterion 5 in
  ``fractions.Fraction``, every threshold typed from docs/tasks/T-0108a.md.

``report.json`` must agree on every condition of K1 and K2, R3 and R4, the void arms, completeness, each arm's
complete / last-OK / shown counts (k/160 and per case k/40), the outcome (form, and the conditions a "no change"
names) and the winner.

Regime (state it beside every number this prints): T-0108a *Regime* -- lean; vLLM v0.29.0; NVFP4 rev ``2ea8372``;
template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on; ``max_num_seqs 3``;
tools ``native:bbfc53ccdd80``, repair on; RTX 5090, driver 595.84, gpu-mem 0.86, fresh boot; dev split, 60 cases,
arms A0/K1/K2, n = 40 on the four Q12 positives and 5 elsewhere (1320 requests); no seed; concurrency 1; typed text;
UiGate (T-0108) and A2UI v0.9.1 at the run commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate, _TooDeep
from local_astra.toolcall import score
from local_astra.toolcall import tools as toolset
from tests.eval.test_toolcall_prompt_ab import PLAIN_IDS, read_commits
from tests.eval.test_toolcall_ui_prompt_ab import case_table_c, read_surface

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0108A_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
Q12 = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
GATE_FILES = (
    "local_astra/conductor/ui_gate.py",
    "local_astra/ui/bracket_repair.py",
    "local_astra/ui/a2ui_schema.py",
)
SKIP_REASON = "re-gates with the live UiGate; reproduces only at the run commit"

#: T-0108a *Arms*
SHAS = {
    "A0": "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57",
    "K1": "22a2ba7e9a022268fb41f847fa3cfad1e0a123f5c323cb120c63d7526fac0702",
    "K2": "23a34c3656547eebaf2f89c65cc6a0a04367f8dcd93924d3bb4b8be79ced687c",
}
ARMS = ("A0", "K1", "K2")
CANDIDATES = ("K1", "K2")
CONDS = ("Q1", "Q2", "U1", "U2", "P3", "P4", "P5", "P6", "P7", "P9", "P10", "P11")
POSITIVE_REPEATS, OTHER_REPEATS = 40, 5
N_REQUESTS = 1320
#: criterion 5, typed out
Q1_MARGIN = 24  # 3/20 x 160
PROTECT_MIN = 24  # of 40
U_MARGIN = 16  # 1/10 x 160
R3_MIN = 112
R4_MAX = 96


# --- the oracle -------------------------------------------------------------------------------------------------


def run_order() -> list[str]:
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
    order = run_order()
    pos = set(positives())
    out: list[tuple[int, str, int, str]] = []
    for r in range(POSITIVE_REPEATS):
        for i, cid in enumerate(order):
            if r >= (POSITIVE_REPEATS if cid in pos else OTHER_REPEATS):
                continue
            off = (i + r) % 3
            out += [(i, cid, r, ARMS[(off + s) % 3]) for s in range(3)]
    return out


def complete_reading(commits: Iterable[Mapping[str, Any]], request_id: str) -> dict[str, Any]:
    """Criterion 4, applied here with the live gate."""
    raws = [str(c.get("text") or "") for c in commits if c.get("segment") == "ui"]
    if not raws:
        return {"last_ok": False, "pruned": set(), "complete": False}
    gate = UiGate()
    last = None
    for i, raw in enumerate(raws):
        last = gate.gate(raw, turn_id=request_id, canvas_idx=i, final=False)
    assert last is not None
    ok = last.surface is not None
    try:
        rep = gate._repair(raws[-1])
        pruned = set(
            gate.validator.invalid_component_ids(list(gate.validator.iter_components(rep.messages)))
        )
    except (_TooDeep, RecursionError):
        # a canvas nested deeper than the parsers follow (T-0108 criterion 9): the gate returns no surface for it
        # and its pruned set cannot be read, so the card is not complete (as the harness reads it)
        return {"last_ok": ok, "pruned": set(), "complete": False}
    pruned |= set(rep.dropped)
    return {"last_ok": ok, "pruned": pruned, "complete": ok and not pruned}


def _rate(k: int, n: int) -> Fraction:
    return Fraction(k, n) if n else Fraction(0)


def derive(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cases = case_table_c()
    tool_set = toolset.load()
    lines = list(trials)
    per: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for t in lines:
        commits = t.get("commits") or []
        c = cases[str(t["case_id"])]
        cr = (
            complete_reading(commits, str(t["request_id"]))
            if c["kind"] == "q12" and c["expect_surface"]
            else {"last_ok": False, "pruned": set(), "complete": False}
        )
        per[str(t["arm"])].append(
            (dict(t), c, read_commits(commits), read_surface(commits, str(t["request_id"])), cr)
        )
    keys = [(str(t["case_id"]), str(t["arm"]), int(t["repeat"])) for t in lines]
    expected = {(cid, arm, r) for _, cid, r, arm in expected_schedule()}
    complete = len(keys) == len(set(keys)) == N_REQUESTS and set(keys) == expected

    stats: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = per.get(arm, [])
        errored = sum(1 for t, *_ in rows if t.get("error") not in (None, ""))
        tool_rows = [x for x in rows if x[1]["kind"] == "tool"]
        routed = [x for x in tool_rows if x[2]["routed"]]
        type_routed = [x for x in routed if x[1]["tool"] == "type_at"]
        valid = sum(
            1
            for x in routed
            if x[2]["call"] is not None and score.schema_valid(x[2]["call"], tool_set)
        )
        exact = sum(
            1 for x in routed if x[2]["call"] is not None and score.exact(x[2]["call"], x[1]["gt"])
        )
        q12 = [x for x in rows if x[1]["kind"] == "q12"]
        positive = [x for x in q12 if x[1]["expect_surface"]]
        negative = [x for x in q12 if not x[1]["expect_surface"]]
        per_case: dict[str, dict[str, int]] = defaultdict(
            lambda: {"complete": 0, "last_ok": 0, "shown": 0, "n": 0}
        )
        for t, _c, _r, s, cr in positive:
            row = per_case[str(t["case_id"])]
            row["complete"] += int(cr["complete"])
            row["last_ok"] += int(cr["last_ok"])
            row["shown"] += int(s["shown"])
            row["n"] += 1
        no_belong = [x for x in rows if x[1]["kind"] in ("no_tool", "plain")] + negative
        stats[arm] = {
            "trials": len(rows),
            "errored": errored,
            "void": bool(rows) and Fraction(errored, len(rows)) > Fraction(2, 100),
            "recall": _rate(len(routed), len(tool_rows)),
            "type_recall": _rate(
                len(type_routed), len([x for x in tool_rows if x[1]["tool"] == "type_at"])
            ),
            "valid": _rate(valid, len(routed)),
            "exact": _rate(exact, len(routed)),
            "false": sum(
                1 for x in rows if x[1]["kind"] in ("no_tool", "plain") and x[2]["routed"]
            ),
            "anomalies": sum(1 for x in rows if x[2]["anomaly"]),
            "thought_open": sum(1 for x in tool_rows if x[2]["thought_open"]),
            "complete": sum(int(x[4]["complete"]) for x in positive),
            "last_ok": sum(int(x[4]["last_ok"]) for x in positive),
            "shown": sum(int(x[3]["shown"]) for x in positive),
            "pos_n": len(positive),
            "neg_shown": sum(int(x[3]["shown"]) for x in negative),
            "neg_n": len(negative),
            "per_case": {cid: dict(v) for cid, v in per_case.items()},
            "surf_no_belong": sum(int(x[3]["shown"]) for x in no_belong),
            "surf_tool": sum(int(x[3]["shown"]) for x in tool_rows),
            "placement_pos": sum(int(x[3]["placement"]) for x in positive),
            "q12_routed": sum(int(x[2]["routed"]) for x in q12),
        }

    a0 = stats["A0"]

    def protected(key: str) -> set[str]:
        return {cid for cid, v in a0["per_case"].items() if v[key] >= PROTECT_MIN}

    def per_case_holds(s: Mapping[str, Any], key: str) -> bool:
        return all(s["per_case"].get(cid, {key: 0})[key] >= PROTECT_MIN for cid in protected(key))

    cands: dict[str, dict[str, bool]] = {}
    for k in CANDIDATES:
        s = stats[k]
        cands[k] = {
            "Q1": s["complete"] >= a0["complete"] + Q1_MARGIN,
            "Q2": per_case_holds(s, "complete"),
            "U1": s["last_ok"] >= a0["last_ok"] - U_MARGIN and per_case_holds(s, "last_ok"),
            "U2": s["shown"] >= a0["shown"] - U_MARGIN and per_case_holds(s, "shown"),
            "P3": s["recall"] >= a0["recall"] - Fraction(2, 100)
            and s["type_recall"] >= a0["type_recall"] - Fraction(4, 100),
            "P4": s["valid"] >= a0["valid"] - Fraction(2, 100)
            and s["exact"] >= a0["exact"] - Fraction(3, 100),
            "P5": s["false"] <= 5 and s["false"] <= a0["false"] + 2,
            "P6": s["anomalies"] == 0,
            "P7": s["thought_open"] <= a0["thought_open"] + 4,
            "P9": s["surf_no_belong"] <= a0["surf_no_belong"] + 1
            and s["surf_tool"] <= a0["surf_tool"] + 2,
            "P10": s["placement_pos"] <= a0["placement_pos"] + 8,
            "P11": s["q12_routed"] <= a0["q12_routed"] + 10,
        }
    readings = {"R3": a0["shown"] >= R3_MIN, "R4": a0["complete"] <= R4_MAX}
    void_arms = sorted(a for a in ARMS if stats[a]["void"])
    passing = [k for k in CANDIDATES if all(cands[k].values())]
    failing = {c for k in CANDIDATES for c, ok in cands[k].items() if not ok}
    if not complete or void_arms or not all(readings.values()):
        outcome, winner = "inconclusive", None
    elif passing:
        best = max(stats[k]["complete"] for k in passing)
        winner = next(
            k for k in CANDIDATES if k in passing and stats[k]["complete"] == best
        )  # tie -> K1
        outcome = "selected"
    else:
        outcome, winner = "no change", None
    return {
        "candidates": cands,
        "readings": readings,
        "void_arms": void_arms,
        "complete": complete,
        "outcome": outcome,
        "failing": failing,
        "winner": winner,
        "stats": stats,
    }


def outcome_kind(text: str) -> str:
    if text.startswith(("K1 selected", "K2 selected")):
        return "selected"
    if text.startswith("no change: "):
        return "no change"
    if text.startswith("inconclusive: "):
        return "inconclusive"
    raise AssertionError(f"criterion 5: outcome {text!r} is none of the three pre-registered forms")


def _mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for v in value.values():
            yield from _mappings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _mappings(v)


# --- the rows ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(f"{ENV} is not set: no T-0108a run directory to read")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("trials.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"T-0108a criterion 7: the run directory has no {name}"
    return path


@pytest.fixture(scope="module")
def manifest(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def same_gate(manifest: dict[str, Any]) -> None:
    """Criterion 8's first check: the manifest's gate-file shas against the working tree; skip when any differs."""
    recorded: dict[str, str] | None = None
    for m in _mappings(manifest):
        got = {p: str(v) for k, v in m.items() for p in GATE_FILES if str(k).endswith(p)}
        if set(got) == set(GATE_FILES):
            recorded = got
            break
    assert recorded is not None, (
        f"T-0108a criterion 1: the manifest records no sha256 of {GATE_FILES}"
    )
    live = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in GATE_FILES}
    if recorded != live:
        changed = sorted(p for p in GATE_FILES if recorded[p] != live[p])
        pytest.skip(f"{SKIP_REASON} (changed since the run: {', '.join(changed)})")


@pytest.fixture(scope="module")
def trials(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def stored(run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return report


def test_the_design_ran_whole(
    manifest: dict[str, Any], same_gate: None, trials: list[dict[str, Any]]
) -> None:
    """Criteria 1 and 10 and *Design*: the manifest names table T-0108a and rule H-T108a; 1320 lines, in exactly the
    schedule of *Design*; ids ``t0108a-r{repeat}-{case_id}-{arm}``; each arm's system sha the task's on every line; no
    test-split case; every commit with the four recorded fields."""
    assert manifest.get("table") == "T-0108a" and manifest.get("rule") == "H-T108a"
    want = expected_schedule()
    assert len(want) == N_REQUESTS == 4 * 40 * 3 + 56 * 5 * 3
    assert len(trials) == N_REQUESTS, len(trials)
    got = [
        (int(t["case_index"]), str(t["case_id"]), int(t["repeat"]), str(t["arm"])) for t in trials
    ]
    assert got == want, "the lines are not in the schedule of *Design*"
    assert [t["request_id"] for t in trials] == [
        f"t0108a-r{r}-{cid}-{arm}" for _, cid, r, arm in want
    ]
    for t in trials:
        assert t["system_sha256"] == SHAS[t["arm"]], (t["case_id"], t["arm"], t["repeat"])
        for c in t.get("commits") or []:
            assert {"segment", "text", "thought_open", "canvas_idx"} <= set(c), (t["case_id"], c)
    assert set(run_order()) == set(case_table_c()), "criterion 10: dev and J10 cases only"


def test_report_agrees_with_the_raw_trials(
    same_gate: None, trials: list[dict[str, Any]], stored: dict[str, Any]
) -> None:
    """Criterion 8: every condition of K1 and K2, R3, R4, the void arms, completeness, the per-arm complete / last-OK
    / shown counts (k/160, per positive case k/40), the outcome (form; a "no change" names exactly the failing
    conditions) and the winner, re-derived from ``trials.jsonl`` with this module's reading and the live ``UiGate``,
    equal what ``report.json`` says."""
    want = derive(trials)
    for k in CANDIDATES:
        for c in CONDS:
            assert stored["candidates"][k][c] is want["candidates"][k][c], (
                f"{k} {c}: report {stored['candidates'][k][c]!r}, raw trials {want['candidates'][k][c]!r}"
            )
    assert stored["readings"] == want["readings"], (stored["readings"], want["readings"])
    assert sorted(stored["void_arms"]) == want["void_arms"]
    assert stored["complete"] is want["complete"]
    for arm in ARMS:
        st, a = want["stats"][arm], stored["arms"][arm]
        assert a["q12_positive_complete"] == {"k": st["complete"], "n": st["pos_n"]}, arm
        assert a["q12_positive_last_ok"] == {"k": st["last_ok"], "n": st["pos_n"]}, arm
        assert a["q12_positive_shown"] == {"k": st["shown"], "n": st["pos_n"]}, arm
        assert a["q12_negative_shown"] == {"k": st["neg_shown"], "n": st["neg_n"]}, arm
        for cid, v in st["per_case"].items():
            assert a["q12_per_case_complete"][cid] == {"k": v["complete"], "n": v["n"]}, (arm, cid)
            assert a["q12_per_case_last_ok"][cid] == {"k": v["last_ok"], "n": v["n"]}, (arm, cid)
            assert a["q12_per_case"][cid] == {"k": v["shown"], "n": v["n"]}, (arm, cid)
    text = str(stored["outcome"])
    assert outcome_kind(text) == want["outcome"], (
        f"outcome: report {text!r}, raw trials {want['outcome']!r}"
    )
    if want["outcome"] == "no change":
        named = set(re.findall(r"\b(?:Q[12]|U[12]|P\d+)\b", text.split(":", 1)[1]))
        assert named == want["failing"], (text, sorted(want["failing"]))
    if want["outcome"] == "selected":
        assert text.startswith(f"{want['winner']} selected") and stored["winner"] == want["winner"]
    else:
        assert stored["winner"] in (None, "no change"), stored["winner"]
    s = want["stats"]
    print(
        f"\nT-0108a outcome (from raw trials): {want['outcome']} {want['winner'] or ''}; readings {want['readings']};"
        f" void {want['void_arms']}; Q12 positive complete A0 {s['A0']['complete']}/{s['A0']['pos_n']}, K1"
        f" {s['K1']['complete']}/{s['K1']['pos_n']}, K2 {s['K2']['complete']}/{s['K2']['pos_n']}"
    )
