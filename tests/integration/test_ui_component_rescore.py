"""T-0108b criteria 10, 11 and 12, and C1 on real data: the three stored runs re-scored on CPU through the real new
gate. CPU only; it spends no GPU time and is not ``gpu``-marked.

    T0108B_T0108A_RUN_DIR=.../t0108a-launch/logs/t0108a-ui-fields-ab-e6c3a81 \\
    T0108_T0103_RUN_DIR=.../t0103-launch/logs/t0103-c3-confirm-6720e56 \\
    T0108_T0098C_RUN_DIR=.../t0098c-launch/logs/t0098c-prompt-ab-3ac00a1 \\
    uv run pytest tests/integration/test_ui_component_rescore.py -q -rA -s

Without all three variables every row skips with that reason. The stored run dirs are read only.

The reading is re-implemented here, not imported (the T-0108a criterion 4 reading, docs/tasks/T-0108b.md
*Evidence*): for every Q12 positive trial (``kind == "q12"`` and the case's ``expect_surface`` in
``tests/eval/surface_placement_cases.jsonl``) a fresh ``UiGate()`` gates every ``ui`` commit in commit order with
``final=False``, ``turn_id`` = the request id and the commit's ``canvas_idx``; ``pruned`` =
``invalid_component_ids(iter_components(_repair(last).messages)) | _repair(last).dropped``; ``complete`` = the last call
returned a surface and ``pruned`` is empty. Trials that are not Q12 positives but have a ``ui`` commit are gated the
same way for criterion 10.

The baseline is ``tests/fixtures/a2ui/t0108b_baseline_outcomes.json`` (the gate at the branch base 08c7776, whose
``local_astra/`` equals 749312d). Every threshold is typed from docs/tasks/T-0108b.md criterion 11 as amended by
H-T108b (d) and (f), not imported.

Regime: CPU, this box, serial; Python 3.11.16; json_repair 0.63.4; vendored A2UI v0.9.1 schema; deterministic, n = 1
per trial. Data regime: e6c3a81 (T-0108a: A0, K1, K2, 160 Q12 positives each), 6720e56 (T-0103: A0, C3, 160 each),
3ac00a1 (T-0098c: 20 per arm); lean, vLLM v0.29.0, NVFP4 rev 2ea8372, CL 256, 48 steps, thinking on, prefix caching on,
no seed, concurrency 1.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from tests.unit.test_component_repair import grounding_strings, in_bodies, reserved_ok

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
Q12 = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
BASELINE = REPO / "tests" / "fixtures" / "a2ui" / "t0108b_baseline_outcomes.json"
ENVS = {
    "e6c3a81": "T0108B_T0108A_RUN_DIR",
    "6720e56": "T0108_T0103_RUN_DIR",
    "3ac00a1": "T0108_T0098C_RUN_DIR",
}
FLIGHTS = ("flight_booking", "flight_booking_with_memory")

#: criterion 11 as amended by H-T108b (d): complete, last-OK and shown minimums of 160 per arm; per-case minimums
#: of 40 (``cases``) and the two flight cases combined, of 80 (``flights``)
MINIMUMS: dict[tuple[str, str], dict[str, Any]] = {
    ("e6c3a81", "A0"): {
        "complete": 122,
        "last_ok": 142,
        "shown": 146,
        "cases": {"omelette_steps": 26, "primes_table": 37},
        "flights": 50,
    },
    ("e6c3a81", "K1"): {"complete": 120, "last_ok": 143, "shown": 152, "cases": {}, "flights": 48},
    ("e6c3a81", "K2"): {"complete": 104, "last_ok": 137, "shown": 139, "cases": {}, "flights": 40},
    ("6720e56", "A0"): {
        "complete": 126,
        "last_ok": 149,
        "shown": 153,
        "cases": {"omelette_steps": 32},
        "flights": 50,
    },
    ("6720e56", "C3"): {"complete": 92, "last_ok": 114, "shown": 124, "cases": {}, "flights": None},
}
TRIALS_PER_ARM = {"e6c3a81": 160, "6720e56": 160, "3ac00a1": 20}
POOLED_ARMS = ("A0", "C3", "C4", "C5", "D1", "D3", "D4")
#: criterion 11, per call: at most this many calls per run dir go from surface to no surface (against 749312d)
MAX_CALL_REGRESSIONS = 1
#: criterion 12: whole-gate p95 on the 376 A0 canvases of 6720e56
GATE_P95_MAX_S = 0.020
N_LATENCY = 376


def run_dirs() -> dict[str, Path]:
    missing = [v for v in ENVS.values() if not os.environ.get(v)]
    if missing:
        pytest.skip(f"set {' and '.join(missing)} to the stored run dirs (T-0108b criteria 10-12)")
    return {run: Path(os.environ[var]) for run, var in ENVS.items()}


def positive_cases() -> set[str]:
    rows = [json.loads(x) for x in Q12.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["id"] for r in rows if r["expect_surface"]}


def baseline() -> dict[str, Any]:
    runs: dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))["runs"]
    return runs


_CACHE: dict[str, Any] = {}


def rescore() -> dict[str, Any]:
    """Per run: request id -> the reading, plus the per-trial evidence the rows below need."""
    if "data" in _CACHE:
        return _CACHE["data"]  # type: ignore[no-any-return]
    dirs, positive, base = run_dirs(), positive_cases(), baseline()
    out: dict[str, Any] = {}
    for run, d in dirs.items():
        raw = (d / "trials.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == base[run]["sha256"], (
            f"premise: {d}/trials.jsonl is not the file the baseline fixture was built from"
        )
        trials: dict[str, Any] = {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            ui = [c for c in t.get("commits", []) if c.get("segment") == "ui"]
            is_pos = t.get("kind") == "q12" and t.get("case_id") in positive
            if not is_pos and not ui:
                continue
            gate, outcomes, bodies, ungrounded, not_ok, repairs = UiGate(), [], [], [], 0, 0
            for c in ui:
                text = str(c.get("text", ""))
                bodies.append(text)
                res = gate.gate(
                    text,
                    turn_id=t["request_id"],
                    canvas_idx=int(c.get("canvas_idx") or 0),
                    final=False,
                )
                outcomes.append(res.surface is not None)
                repairs += int(getattr(res, "component_repairs", 0) or 0)
                if res.surface is None:
                    continue
                not_ok += int(not res.report.ok)
                sid = f"turn-{t['request_id']}"
                for s in grounding_strings(list(res.surface.messages)):
                    if s != sid and not in_bodies(s, bodies) and not reserved_ok(s, bodies):
                        ungrounded.append(s)
            pruned: set[str] = set()
            kinds: list[str] = []
            if ui:
                rep = gate._repair(bodies[-1])
                comps = list(gate.validator.iter_components(rep.messages))
                pruned = set(gate.validator.invalid_component_ids(comps)) | set(rep.dropped)
                kind_of = {c.get("id"): c.get("component") for c in comps if isinstance(c, dict)}
                kinds = sorted(str(kind_of.get(p, "?")) for p in pruned)
            trials[t["request_id"]] = {
                "arm": t["arm"],
                "case_id": t.get("case_id"),
                "positive": is_pos,
                "outcomes": outcomes,
                "complete": bool(outcomes and outcomes[-1] and not pruned),
                "pruned_kinds": kinds,
                "ungrounded": ungrounded,
                "not_ok": not_ok,
                "repairs": repairs,
            }
        out[run] = trials
    _CACHE["data"] = out
    return out


def stats(trials: dict[str, Any], arm: str, cases: tuple[str, ...] | None = None) -> Counter[str]:
    s: Counter[str] = Counter()
    for t in trials.values():
        if (
            not t["positive"]
            or t["arm"] != arm
            or (cases is not None and t["case_id"] not in cases)
        ):
            continue
        o = t["outcomes"]
        s["trials"] += 1
        s["complete"] += int(t["complete"])
        s["last_ok"] += int(bool(o) and o[-1])
        s["shown"] += int(any(o))
        s["calls_ok"] += sum(o)
        s["repairs"] += t.get("repairs", 0)
    return s


def print_tables(data: dict[str, Any], base: dict[str, Any]) -> None:
    for run, trials in data.items():
        before = base[run]["trials"]
        cases = sorted({t["case_id"] for t in trials.values() if t["positive"]})
        print(
            f"\n== {run}: arm | complete | last OK | shown | calls OK (baseline -> now) | component repairs"
        )
        for arm in sorted({t["arm"] for t in trials.values() if t["positive"]}):
            b, a = stats(before, arm), stats(trials, arm)
            cols = " | ".join(
                f"{b[k]}->{a[k]}" for k in ("complete", "last_ok", "shown", "calls_ok")
            )
            per_case = " ".join(
                f"{c}:{stats(before, arm, (c,))['complete']}->{stats(trials, arm, (c,))['complete']}"
                for c in cases
            )
            print(f"  {arm}: {cols} | {a['repairs']}  [{per_case}]")
            residual: Counter[str] = Counter()
            sizes: Counter[int] = Counter()
            for t in trials.values():
                if (
                    t["positive"]
                    and t["arm"] == arm
                    and t["outcomes"]
                    and t["outcomes"][-1]
                    and not t["complete"]
                ):
                    residual.update(t["pruned_kinds"])
                    sizes[len(t["pruned_kinds"])] += 1
            print(
                f"     residual pruned kinds {dict(residual.most_common())}; incomplete last-OK by size {dict(sorted(sizes.items()))}"
            )


def test_criterion_11_minimums() -> None:
    """Criterion 11 (pre-registered; minimums as amended by H-T108b (d)): complete, per-case, last-OK and shown per
    run and arm; 3ac00a1 each arm at least its baseline complete and last-OK. Denominators checked first."""
    data, base = rescore(), baseline()
    print_tables(data, base)
    failures = []
    for run, trials in data.items():
        for arm in {t["arm"] for t in trials.values() if t["positive"]}:
            n = stats(trials, arm)["trials"]
            assert n == TRIALS_PER_ARM[run], (
                f"denominator {run} {arm}: {n} != {TRIALS_PER_ARM[run]}"
            )
    for (run, arm), want in MINIMUMS.items():
        s = stats(data[run], arm)
        for key in ("complete", "last_ok", "shown"):
            if s[key] < want[key]:
                failures.append(f"{run} {arm} {key}: {s[key]}/160 (needs >= {want[key]})")
        for case, floor in want["cases"].items():
            got = stats(data[run], arm, (case,))
            assert got["trials"] == 40, f"denominator {run} {arm} {case}: {got['trials']}"
            if got["complete"] < floor:
                failures.append(
                    f"{run} {arm} {case} complete: {got['complete']}/40 (needs >= {floor})"
                )
        if want["flights"] is not None:
            got = stats(data[run], arm, FLIGHTS)
            assert got["trials"] == 80, f"denominator {run} {arm} flights: {got['trials']}"
            if got["complete"] < want["flights"]:
                failures.append(
                    f"{run} {arm} flights complete: {got['complete']}/80 (needs >= {want['flights']})"
                )
    for arm in POOLED_ARMS:
        now, was = stats(data["3ac00a1"], arm), stats(base["3ac00a1"]["trials"], arm)
        for key in ("complete", "last_ok"):
            if now[key] < was[key]:
                failures.append(f"3ac00a1 {arm} {key}: {now[key]} < its baseline {was[key]}")
    assert not failures, "criterion 11:\n" + "\n".join(failures)


def test_criterion_11_no_complete_trial_lost_and_at_most_one_call_regression() -> None:
    """Criterion 11, negatives (guard: true at the branch base, must stay true): no trial complete at baseline is
    incomplete now; per run dir at most 1 call goes from surface to none; the trial set is the baseline's."""
    data, base = rescore(), baseline()
    failures = []
    for run, trials in data.items():
        was_all = base[run]["trials"]
        assert set(trials) == set(was_all), f"{run}: trial set differs from the baseline fixture"
        regressions = 0
        for rid, t in trials.items():
            was = was_all[rid]
            assert len(was["outcomes"]) == len(t["outcomes"]), (
                f"{run} {rid}: ui commit count differs"
            )
            if t["positive"] and was["complete"] and not t["complete"]:
                failures.append(f"{run} {rid}: complete at baseline, incomplete now")
            regressions += sum(
                1 for b, a in zip(was["outcomes"], t["outcomes"], strict=True) if b and not a
            )
        if regressions > MAX_CALL_REGRESSIONS:
            failures.append(
                f"{run}: {regressions} calls went from surface to none (max {MAX_CALL_REGRESSIONS})"
            )
    assert not failures, "criterion 11:\n" + "\n".join(failures)


def test_criterion_10_nothing_invalid_reaches_the_renderer() -> None:
    """Criterion 10 (guard): every emitted surface had ``report.ok``; 3ac00a1 D3 still emits 0 surfaces of its 16
    calls; Q12 negatives and tool trials gain no surface, call by call, against the baseline."""
    data, base = rescore(), baseline()
    d3 = [t for t in data["3ac00a1"].values() if t["positive"] and t["arm"] == "D3"]
    assert sum(len(t["outcomes"]) for t in d3) == 16 and sum(sum(t["outcomes"]) for t in d3) == 0, (
        "criterion 10: D3"
    )
    bad = {rid: t["not_ok"] for run in data.values() for rid, t in run.items() if t["not_ok"]}
    assert not bad, f"criterion 10: surfaces emitted without report.ok: {bad}"
    gained = [
        f"{run} {rid} ({t['arm']}): {base[run]['trials'][rid]['outcomes']} -> {t['outcomes']}"
        for run, trials in data.items()
        for rid, t in trials.items()
        if not t["positive"]
        and any(
            a and not b
            for a, b in zip(t["outcomes"], base[run]["trials"][rid]["outcomes"], strict=True)
        )
    ]
    assert not gained, "criterion 10: a non-positive trial gained a surface:\n" + "\n".join(gained)


def test_c1_on_real_data_values_against_bodies_keys_against_the_catalog() -> None:
    """C1 / C7 on real data (guard): every emitted string value is in the union of the trial's ui bodies so far, and
    every key is in the catalog vocabulary of its component's kind or in the bodies; only the reserved namespace
    (``__gate.clicked``, ``__gate_<model id>_label``) is exempt."""
    data = rescore()
    bad = {
        rid: t["ungrounded"][:3]
        for run in data.values()
        for rid, t in run.items()
        if t["ungrounded"]
    }
    assert not bad, f"C1: ungrounded strings reached a surface: {bad}"


def test_criterion_12_whole_gate_latency_on_the_a0_canvases() -> None:
    """Criterion 12: whole-gate p95 <= 20 ms on the 376 A0 canvases of 6720e56 (14.6 ms at the base), a fresh gate
    per canvas, warm process, serial."""
    d = run_dirs()["6720e56"]
    positive = positive_cases()
    texts = []
    for line in (d / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line) if line.strip() else {}
        if t.get("kind") == "q12" and t.get("case_id") in positive and t.get("arm") == "A0":
            texts += [
                str(c.get("text", "")) for c in t.get("commits", []) if c.get("segment") == "ui"
            ]
    assert len(texts) == N_LATENCY
    for text in texts[:20]:
        UiGate().gate(text, turn_id="warm", canvas_idx=0, final=False)
    times = []
    for text in texts:
        start = time.perf_counter()
        UiGate().gate(text, turn_id="lat", canvas_idx=0, final=False)
        times.append(time.perf_counter() - start)
    p95 = statistics.quantiles(times, n=20)[-1]
    print(f"\ngate p95 {p95 * 1e3:.2f} ms (n={N_LATENCY} A0 canvases, 6720e56)")
    assert p95 <= GATE_P95_MAX_S, (
        f"criterion 12: p95 {p95 * 1e3:.2f} ms > {GATE_P95_MAX_S * 1e3:.0f} ms"
    )
