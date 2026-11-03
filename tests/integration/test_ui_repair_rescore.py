"""T-0108 criteria 5 (real-canvas latency), 7, 8 and 12: the stored trials re-scored on CPU through
the real new gate. CPU only: it spends no GPU time and is not ``gpu``-marked (re-scoped by the
orchestrator 2026-10-16 from ``tests/eval/``, where T-0046 deselects every row by default).

    T0108_T0103_RUN_DIR=.../logs/t0103-c3-confirm-6720e56 \\
    T0108_T0098C_RUN_DIR=.../logs/t0098c-prompt-ab-3ac00a1 \\
    uv run pytest tests/integration/test_ui_repair_rescore.py -q -rA -s

Without both variables every row skips with that reason.

The reading is re-implemented here, not imported from the harness: for every Q12 positive trial
(``kind == "q12"`` and the case's ``expect_surface`` in ``tests/eval/surface_placement_cases.jsonl``)
a fresh ``UiGate()`` gates every ``ui`` commit in commit order with ``final=False``,
``turn_id`` = the request id and the commit's ``canvas_idx`` (the planner's *Evidence* reading).
The baseline is ``tests/fixtures/a2ui/t0108_baseline_outcomes.json`` (the 14cde78 gate, criterion
13 (ii)). Every threshold is typed from docs/tasks/T-0108.md criterion 12, not imported.

Regime: CPU, this box, serial; Python 3.11.16; json_repair 0.63.4; vendored A2UI v0.9.1 schema;
deterministic, n = 1 per trial. Data regime: T-0103 run 6720e56 and T-0098c run 3ac00a1 (lean,
vLLM v0.29.0, NVFP4 rev 2ea8372, CL 256, 48 steps, thinking on, no seed, dev split).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from tests.unit.test_component_repair import grounding_strings, reserved_ok

REPO = Path(__file__).resolve().parents[2]
Q12 = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
BASELINE = REPO / "tests" / "fixtures" / "a2ui" / "t0108_baseline_outcomes.json"
ENVS = {"6720e56": "T0108_T0103_RUN_DIR", "3ac00a1": "T0108_T0098C_RUN_DIR"}
POOLED = ("A0", "C3", "C4", "C5", "D4")

#: criterion 12, typed from the task file: (denominator, bound)
THRESHOLDS = {
    ("6720e56", "A0"): {
        "with_ui": 153,
        "gated": 376,
        "trials": 160,
        "last_ok_min": 140,
        "failed_max": 50,
        "shown_min": 145,
        "shown_last_failed_max": 10,
    },
    ("6720e56", "C3"): {
        "with_ui": 128,
        "gated": 359,
        "trials": 160,
        "last_ok_min": 105,
        "failed_max": 85,
        "shown_min": 115,
        "shown_last_failed_max": 16,
    },
}
POOLED_WITH_UI, POOLED_LAST_OK_MIN = 88, 76
#: criterion 8: at most this many individual calls per run dir go from surface to no surface
MAX_CALL_REGRESSIONS = 2
#: criterion 5: p95 on the 376 A0 canvases of 6720e56
REPAIR_P95_MAX_S, GATE_P95_MAX_S = 0.001, 0.020

#: report-only class detectors (rough, regex over the raw canvas)
CLASSES = {
    "duplicated_opener": re.compile(r'\{"\{"'),
    "missing_separator": re.compile(r'" "\{"'),
    "spurious_closer": re.compile(r'"\]\},\{"'),
    "stray_quote": re.compile(r'[\]}]"[\]}]'),
}


def positive_cases() -> set[str]:
    rows = [json.loads(x) for x in Q12.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["id"] for r in rows if r["expect_surface"]}


def run_dirs() -> dict[str, Path]:
    missing = [v for v in ENVS.values() if not os.environ.get(v)]
    if missing:
        pytest.skip(f"set {' and '.join(missing)} to the stored run dirs (T-0108 criterion 12)")
    return {run: Path(os.environ[var]) for run, var in ENVS.items()}


def strings_of(value: Any) -> list[str]:
    """T-0108b criterion 9(i): every string value, and every key except one in the catalog vocabulary of the
    enclosing component's kind (C2); keys outside a component are still checked."""
    return list(grounding_strings(value))


_CACHE: dict[str, Any] = {}


def rescore() -> dict[str, Any]:
    """Per run: request id -> {arm, outcomes, texts, ungrounded, ok_mismatch}."""
    if "data" in _CACHE:
        return _CACHE["data"]  # type: ignore[no-any-return]
    dirs = run_dirs()
    positive = positive_cases()
    out: dict[str, dict[str, Any]] = {}
    for run, d in dirs.items():
        trials: dict[str, Any] = {}
        for line in (d / "trials.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("kind") != "q12" or t.get("case_id") not in positive:
                continue
            gate, outcomes, texts, ungrounded, bad_ok = UiGate(), [], [], [], 0
            for c in t.get("commits", []):
                if c.get("segment") != "ui":
                    continue
                text = str(c.get("text", ""))
                texts.append(text)
                res = gate.gate(
                    text,
                    turn_id=t["request_id"],
                    canvas_idx=int(c.get("canvas_idx") or 0),
                    final=False,
                )
                outcomes.append(res.surface is not None)
                if res.surface is not None:
                    bad_ok += int(not res.report.ok)
                    sid = f"turn-{t['request_id']}"
                    for s in strings_of(list(res.surface.messages)):
                        esc = json.dumps(s)[1:-1]
                        # T-0108b (H-T108b (d), C7): exactly the reserved namespace is exempt
                        grounded = any(s in b or esc in b for b in texts) or reserved_ok(s, texts)
                        if s != sid and not grounded:
                            ungrounded.append(s)
            trials[t["request_id"]] = {
                "arm": t["arm"],
                "outcomes": outcomes,
                "texts": texts,
                "ungrounded": ungrounded,
                "ok_mismatch": bad_ok,
            }
        out[run] = trials
    _CACHE["data"] = out
    return out


def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))["runs"]  # type: ignore[no-any-return]


def arm_stats(trials: dict[str, Any], arm: str) -> dict[str, int]:
    s: Counter[str] = Counter()
    for t in trials.values():
        if t["arm"] != arm:
            continue
        o = t["outcomes"]
        s["trials"] += 1
        if not o:
            continue
        s["with_ui"] += 1
        s["gated"] += len(o)
        s["emitted"] += sum(o)
        s["failed"] += len(o) - sum(o)
        s["last_ok"] += int(o[-1])
        s["shown"] += int(any(o))
        s["shown_last_failed"] += int(any(o) and not o[-1])
    return dict(s)


def print_tables(data: dict[str, Any], base: dict[str, Any]) -> None:
    for run, trials in data.items():
        arms = sorted({t["arm"] for t in trials.values()})
        before = {
            rid: {"arm": v["arm"], "outcomes": v["outcomes"]}
            for rid, v in base[run]["trials"].items()
        }
        print(
            f"\n== {run}: arm | trials | with ui | gated | last OK | failed | shown | "
            "shown-last-failed (before -> after)"
        )
        for arm in arms:
            b, a = arm_stats(before, arm), arm_stats(trials, arm)
            cols = ("trials", "with_ui", "gated", "last_ok", "failed", "shown", "shown_last_failed")
            print(f"  {arm}: " + " | ".join(f"{b.get(c, 0)}->{a.get(c, 0)}" for c in cols))
        per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for rid, t in trials.items():
            for i, text in enumerate(t["texts"]):
                for name, rx in CLASSES.items():
                    if rx.search(text):
                        per_class[name][0] += 1
                        per_class[name][1] += int(not base[run]["trials"][rid]["outcomes"][i])
                        per_class[name][2] += int(not t["outcomes"][i])
        print(
            f"  per class (regex detector; calls | failed before | failed after): {dict(per_class)}"
        )


def test_criterion_12_minimums() -> None:
    data, base = rescore(), baseline()
    print_tables(data, base)
    failures = []
    for (run, arm), th in THRESHOLDS.items():
        s = arm_stats(data[run], arm)
        for k in ("trials", "with_ui", "gated"):
            assert s.get(k, 0) == th[k], f"denominator {run} {arm} {k}: {s.get(k, 0)} != {th[k]}"
        checks = [
            (
                "last call OK",
                s.get("last_ok", 0) >= th["last_ok_min"],
                s.get("last_ok", 0),
                f">= {th['last_ok_min']}",
            ),
            (
                "failed calls",
                s.get("failed", 0) <= th["failed_max"],
                s.get("failed", 0),
                f"<= {th['failed_max']}",
            ),
            (
                "shown",
                s.get("shown", 0) >= th["shown_min"],
                s.get("shown", 0),
                f">= {th['shown_min']}",
            ),
            (
                "shown but last failed",
                s.get("shown_last_failed", 0) <= th["shown_last_failed_max"],
                s.get("shown_last_failed", 0),
                f"<= {th['shown_last_failed_max']}",
            ),
        ]
        failures += [f"{run} {arm} {n}: {v} (needs {want})" for n, ok, v, want in checks if not ok]
    before = {
        rid: {"arm": v["arm"], "outcomes": v["outcomes"]}
        for rid, v in base["3ac00a1"]["trials"].items()
    }
    pooled = sum(arm_stats(data["3ac00a1"], a).get("last_ok", 0) for a in POOLED)
    with_ui = sum(arm_stats(data["3ac00a1"], a).get("with_ui", 0) for a in POOLED)
    assert with_ui == POOLED_WITH_UI, f"denominator 3ac00a1 pooled with ui: {with_ui}"
    if pooled < POOLED_LAST_OK_MIN:
        failures.append(
            f"3ac00a1 pooled last call OK: {pooled}/{with_ui} (needs >= {POOLED_LAST_OK_MIN})"
        )
    for arm in POOLED:
        now, was = (
            arm_stats(data["3ac00a1"], arm).get("last_ok", 0),
            arm_stats(before, arm).get("last_ok", 0),
        )
        if now < was:
            failures.append(f"3ac00a1 {arm} last call OK {now} < its baseline {was}")
    assert not failures, "criterion 12:\n" + "\n".join(failures)


def test_criterion_7_nothing_invalid_reaches_the_renderer() -> None:
    """Guard (holds at 14cde78 and after): D3 emits 0; every emitted surface had ``report.ok``."""
    data = rescore()
    d3 = arm_stats(data["3ac00a1"], "D3")
    assert d3.get("gated", 0) == 16 and d3.get("emitted", 0) == 0, f"criterion 7: D3 {d3}"
    bad = {
        rid: t["ok_mismatch"] for run in data.values() for rid, t in run.items() if t["ok_mismatch"]
    }
    assert not bad, f"criterion 7: surfaces emitted without report.ok: {bad}"


def test_criterion_8_no_per_trial_regression() -> None:
    data, base = rescore(), baseline()
    failures = []
    for run, trials in data.items():
        assert set(trials) == set(base[run]["trials"]), (
            f"{run}: trial set differs from the baseline fixture"
        )
        regressions = 0
        for rid, t in trials.items():
            was = base[run]["trials"][rid]["outcomes"]
            assert len(was) == len(t["outcomes"]), f"{run} {rid}: ui commit count differs"
            if was and was[-1] and not t["outcomes"][-1]:
                failures.append(f"{run} {rid}: last call OK at 14cde78, failed now")
            regressions += sum(1 for b, a in zip(was, t["outcomes"], strict=True) if b and not a)
        if regressions > MAX_CALL_REGRESSIONS:
            failures.append(
                f"{run}: {regressions} calls went from surface to none (max {MAX_CALL_REGRESSIONS})"
            )
    # the new gate must also have changed something: an unchanged gate passes the two rules above
    changed = sum(
        1
        for run, trials in data.items()
        for rid, t in trials.items()
        if t["outcomes"] != base[run]["trials"][rid]["outcomes"]
    )
    assert changed > 0, "criterion 8 is vacuous against an unchanged gate: no outcome moved"
    assert not failures, "criterion 8:\n" + "\n".join(failures)


def test_real_data_grounding_against_the_union_of_ui_bodies() -> None:
    """Criterion 3 on real data (guard): every emitted string is in the turn's ui bodies so far."""
    data = rescore()
    bad = {
        rid: t["ungrounded"][:3]
        for run in data.values()
        for rid, t in run.items()
        if t["ungrounded"]
    }
    assert not bad, f"ungrounded strings reached a surface: {bad}"


def test_criterion_5_latency_on_the_a0_canvases() -> None:
    """p95 <= 1 ms for the repair alone and <= 20 ms for the whole gate (baseline 13.2 ms),
    n = 376 A0 canvases of 6720e56, warm process, serial."""
    from tests.unit.test_bracket_repair import repair_fn

    data = rescore()
    texts = [x for t in data["6720e56"].values() if t["arm"] == "A0" for x in t["texts"]]
    assert len(texts) == 376
    fn = repair_fn()
    rep, gat = [], []
    for text in texts:
        start = time.perf_counter()
        fn(text)
        rep.append(time.perf_counter() - start)
        start = time.perf_counter()
        UiGate().gate(text, turn_id="lat", canvas_idx=0, final=False)
        gat.append(time.perf_counter() - start)
    p95r, p95g = statistics.quantiles(rep, n=20)[-1], statistics.quantiles(gat, n=20)[-1]
    print(f"\nrepair p95 {p95r * 1e3:.3f} ms; gate p95 {p95g * 1e3:.2f} ms (n=376 A0, 6720e56)")
    assert p95r <= REPAIR_P95_MAX_S and p95g <= GATE_P95_MAX_S
