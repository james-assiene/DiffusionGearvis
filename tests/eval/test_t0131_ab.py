"""T-0131 D3-D5: the 6-arm grounding A/B's metrics and decision label, re-derived from the run's records and compared
with ``report.json`` (``gpu``, ``eval``; report-only, plus the pre-registered decision rule).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0131_grounding_ab.sh``)::

    T0131_RUN_DIR=logs/t0131-grounding-ab-<sha7> uv run pytest tests/eval/test_t0131_ab.py -q -m gpu -rA

Every row skips when ``T0131_RUN_DIR`` is unset or the directory has no ``report.json``. The rows spend no GPU time
(they read the run directory) but are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness.** ``derive`` applies D3's metrics per arm and D5's decision rule
(H-T131-1, approved 2026-10-22) to the record fields pinned in ``tests/unit/test_t0131_ab_harness.py``'s docstring.
The T-0111g parts (acceptance, outcomes scored from ``__state``, statuses, latency, cache, the six T-0111g safety
counts) are T-0111g's own oracle, ``tests/eval/test_live_smoke.py``'s ``derive``, applied per arm; this module adds
clicks, scrolls, false claims, the reground cost, wall time, VRAM, the engine flag, the two new safety counts, the
probes and the decision. ``tests/unit/test_t0131_ab_harness.py`` checks this oracle against hand counts on CPU, so a
wrong oracle is caught without a GPU run.

D3's acceptance (orchestrator ruling at 23c6c77, reading 3 overruled): every step generate at 200 (T-0111g's rule)
and every reground request at 200 except a pass 2 aborted by ``REGROUND_TIMEOUT_MS`` (``pass2_timeout``), which is the
harness's own timeout and is reported only under ``reground_cost`` (its outcome count, ``requests``, ``ms``).

The group D review round (rulings at 4337125, with the orchestrator's acceptance of the implementer's choices): a
``pass2_timeout`` counts as ``REGROUND_TIMEOUT_MS`` (4000.0 ms) in ``reground_cost.latency_ms`` and so in D5's
reground p50 gate; the labels are "inconclusive" (P1 failed), then "void" (P2 or P3 failed), then "incomplete" (the
run stopped early; ``report.json`` carries ``stopped``), then D5's, and ``decision.eligible`` is ``[]`` under the
first three; each click record carries ``read_error`` and each click section ``read_errors``; each trial carries
``budgets_sent`` and ``guidance_sent`` (the configuration evidence ``config_findings`` checks).

D4's coverage row maps each ``llm.request`` phase to the sentinel scan's label: ``prefill`` -> ``prefill:``;
``generate``, ``tool_step`` and ``reground`` -> ``generate:`` (``LoopMeter`` labels every non-repair generate
``generate:``); ``tool_repair`` -> ``repair:``. Every attempt must appear in its trial's ``scanned_requests``.

Regime (state it beside every number this prints): T-0131 *Regime*, GPU -- profile lean, canvas length 256, 48
denoising steps, thinking on; vLLM 0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16,
``services/vllm.sh`` unchanged (806da12), one boot; tools native bbfc53ccdd80, live, repair on, loop 8; headless
Chromium 156 at 1280x800; typed turns, fake STT/TTS/audio, no headset or speakers; 6 arms (visual tokens {280, 560,
1120} x ReGround {off, on}) x 5 tasks x n = 5 = 150 turns, shuffled jointly with seed 0, concurrency 1; the engine
unseeded; STT stopped. History, not like-for-like: T-0111g's run at ca9576f (no guidance, implicit 280).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.eval import test_live_smoke as g0111

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0131_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

#: D1, typed from docs/tasks/T-0131.md: (visual tokens, ReGround)
ARMS: tuple[tuple[int, bool], ...] = (
    (280, False),
    (280, True),
    (560, False),
    (560, True),
    (1120, False),
    (1120, True),
)
#: sha256 of ``json.dumps([{"visual_tokens": b, "reground": r} for b, r in ARMS], sort_keys=True,
#: separators=(",", ":"))``, computed by the test-author on 2026-10-22 from the table above
ARM_TABLE_SHA256 = "5cc0169fe0cf4612f982c68fc20c2feb63ae803d6f70ceff372ed4d6209f32e2"
#: T-0111g's task-table pin (D1: the harness imports T-0111g's table)
TASK_TABLE_SHA256 = "59c31b4c2cb4d51471a11bde850db040f1dc2030a8767bd72f6adc91e2c5173d"
TASK_IDS = g0111.TASK_IDS
N_PER_TASK = 5
TURNS_PER_ARM = 25
TOTAL_TURNS = 150
#: B3's outcome labels
OUTCOMES = (
    "ok",
    "pass2_unparseable",
    "pass2_no_point",
    "pass2_other_tool",
    "pass2_timeout",
    "error",
    "skipped_no_point",
)
#: the outcome of a pass 2 aborted by ``REGROUND_TIMEOUT_MS``: out of D3's acceptance, in the reground cost (ruling
#: at 23c6c77, reading 3 overruled)
ABORTED = "pass2_timeout"
#: the session's pass-2 timeout (implementer decision 1, accepted 2026-10-22). Ruling at 4337125: a ``pass2_timeout``
#: counts as this many ms (the float 4000.0) in ``reground_cost.latency_ms``, and so in D5's reground p50 gate; a
#: non-timeout request without ``done_ms`` contributes nothing
REGROUND_TIMEOUT_MS = 4000
#: D2 P1: ``expected_soft_tokens(1280, 800, N)`` by hand (A4), and the 512x320 crop gives the same
P1_EXPECTED = {280: 273, 560: 522, 1120: 1092}
#: D2 P1: at most this many image-delimiter tokens on top
P1_DELIMITERS = 4
#: D3: executed calls that carry a click record
CLICK_TOOLS = ("click_at", "type_at")
#: D5 (H-T131-1 approved as proposed)
SUCCESS_WITHIN = 3
HIT_RATE_WITHIN = 0.10
DONE_P95_MAX_MS = 1500
REGROUND_P50_MAX_MS = 1000
NO_CHANGE_GAIN = 2
BASELINE = "280/off"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
#: ruling at 4337125: a run voided by P2 or P3 (P1 passed)
VOID = "void"
#: ruling at 4337125: a run stopped early (``ERRORED_STOP`` or ``RUN_BUDGET_S``); ``report.json`` carries ``stopped``
INCOMPLETE = "incomplete"
#: the labels that are not D5's: ``decision.eligible`` is ``[]`` under each (orchestrator, 2026-10-22)
NOT_D5 = (INCONCLUSIVE, VOID, INCOMPLETE)
#: D3: the keys of one ``click`` / ``click_p1`` record (``read_error``: None, or the repr of the box-read failure)
CLICK_KEYS = frozenset({"point", "boxes", "hit", "error_px", "read_error"})
SAFETY = (*g0111.SAFETY, "reground_voiced", "reground_other_tool_executed")
#: D4: ``llm.request`` phase -> the sentinel-scan label's phase
SCAN_PHASE = {
    "prefill": "prefill",
    "generate": "generate",
    "tool_step": "generate",
    "reground": "generate",
    "tool_repair": "repair",
}


# =====================================================================================================
# The oracle (pure; the unit module imports these)
# =====================================================================================================


def arm_label(visual_tokens: int, reground: bool) -> str:
    """``"280/off"``, ``"1120/on"``: the report's key for an arm, and D5's outcome label for a chosen arm."""
    return f"{int(visual_tokens)}/{'on' if reground else 'off'}"


LABELS = tuple(arm_label(b, r) for b, r in ARMS)


def arm_of(record: Mapping[str, Any]) -> str:
    return arm_label(int(record["visual_tokens"]), bool(record["reground"]))


def p1_ok(budget: int, delta: int) -> bool:
    """D2 P1: the image's token count (with minus without) is ``expected_soft_tokens(1280, 800, budget)`` plus 0..4
    image-delimiter tokens; below is "not honoured", above is "not the expected count"."""
    expected = P1_EXPECTED[int(budget)]
    return expected <= int(delta) <= expected + P1_DELIMITERS


def click_score(
    point: Sequence[float], boxes: Sequence[Mapping[str, float]]
) -> tuple[bool, float | None]:
    """D3: ``(hit, error_px)``. A hit is a point inside an intended box, edges included; the error is the Euclidean
    distance to the nearest intended box (0 inside it); ``(False, None)`` when the page showed no intended box."""
    px, py = float(point[0]), float(point[1])
    best: float | None = None
    for b in boxes:
        x0, y0 = float(b["x"]), float(b["y"])
        x1, y1 = x0 + float(b["width"]), y0 + float(b["height"])
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        d = math.hypot(dx, dy)
        best = d if best is None or d < best else best
    if best is None:
        return False, None
    return best == 0.0, best


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def click_stats(steps: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """D3 click metrics over the steps' ``field`` records (``click`` or ``click_p1``), re-scored from each record's
    ``point`` and ``boxes`` (never its stored ``hit``). S4 has no intended target: its clicks are ``unscored``.
    ``read_errors`` (ruling at 4337125) counts the records whose box read failed (``read_error`` not None), S4's
    included; such a record is otherwise scored like any other (its boxes as read, usually none)."""
    n = hits = unscored = read_errors = 0
    errors: list[float] = []
    for s in steps:
        c = s.get(field)
        if c is None:
            continue
        if c.get("read_error") is not None:
            read_errors += 1
        if s["task"] == "S4":
            unscored += 1
            continue
        hit, err = click_score(c["point"], c["boxes"])
        n += 1
        hits += int(hit)
        if err is not None:
            errors.append(err)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n if n else None,
        "error_px": {**g0111.dist(errors), "mean": _mean(errors)},
        "unscored": unscored,
        "read_errors": read_errors,
    }


def reground_latency(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D3 reground latency (ruling at 4337125, the approved "reground p50 <= 1.0 s" read over every pass-2 attempt):
    a ``pass2_timeout`` counts as ``REGROUND_TIMEOUT_MS`` (4000.0), whatever its ``done_ms``; any other request
    counts its ``done_ms``, and one without ``done_ms`` (an error with no response) counts nothing."""
    values: list[float] = []
    for r in requests:
        if r.get("outcome") == ABORTED:
            values.append(float(REGROUND_TIMEOUT_MS))
        elif r.get("done_ms") is not None:
            values.append(r["done_ms"])
    return g0111.dist(values)


def false_claim(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """D3's proxy for "claims success": the turn is not a success, it finished (no exception, not timed out), and
    its last step made no call (``status`` and ``call_name`` both None) and voiced something with no question mark."""
    if trial.get("exception") or trial.get("timed_out") or not steps:
        return False
    if g0111.succeeded(str(trial["task"]), trial.get("state"), g0111.voiced_text(trial)):
        return False
    last = max(steps, key=lambda s: int(s["step"]))
    if last.get("status") is not None or last.get("call_name") is not None:
        return False
    said = " ".join(str(v) for v in last.get("voiced_sources") or [])
    return bool(said.strip()) and "?" not in said


def reground_voiced_findings(
    pass2_answers: Sequence[str],
    voiced: Sequence[str],
    history: str | None,
    step_texts: Sequence[str],
) -> list[str]:
    """D3 ``reground_voiced``: each pass-2 answer text (whitespace-collapsed, non-empty) found inside a voiced source
    or the stored history while it is in none of the turn's own committed step texts (pass 2 sees pass 1's prompt,
    so it may repeat what a step legitimately said)."""
    targets = [g0111._collapse(v) for v in voiced] + ([g0111._collapse(history)] if history else [])
    own = [g0111._collapse(t) for t in step_texts]
    out: list[str] = []
    for a in pass2_answers:
        text = g0111._collapse(a)
        if text and any(text in t for t in targets) and not any(text in o for o in own):
            out.append(text)
    return out


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def derive_arm(
    visual_tokens: int,
    reground: bool,
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    turn_lines: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One arm's D3 section (the records already filtered to the arm)."""
    base = g0111.derive(trials, steps, turn_lines)
    by_turn: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_turn.setdefault(_key(s), []).append(s)
    requests = [r for r in regrounds if r.get("outcome") != "skipped_no_point"]
    # ruling at 23c6c77 (reading 3 overruled): a pass 2 aborted by REGROUND_TIMEOUT_MS is the harness's own timeout,
    # reported under the reground cost only; every other reground request is gated, HTTP 0 included
    gated = [r for r in requests if r.get("outcome") != ABORTED]
    reground_non_200 = [
        {
            "task": r["task"],
            "repeat": int(r["repeat"]),
            "step": int(r["step"]),
            "http_status": r.get("http_status"),
        }
        for r in gated
        if r.get("http_status") != 200
    ]
    acceptance = {
        **base["acceptance"],
        "regrounds": len(gated),
        "regrounds_ok": sum(1 for r in gated if r.get("http_status") == 200),
        "reground_non_200": reground_non_200,
        "all_200": not base["acceptance"]["non_200"] and not reground_non_200,
    }
    outcomes = base["outcomes"]
    scrolls = {
        task: g0111.dist(
            [
                sum(
                    1
                    for s in by_turn.get(_key(t), [])
                    if s.get("call_name") == "scroll" and s.get("status") == "executed"
                )
                for t in trials
                if t["task"] == task
            ]
        )
        for task in ("S4", "S5")
    }
    cost = None
    if reground:
        counts = Counter(str(r.get("outcome")) for r in regrounds)
        cost = {
            "requests": len(requests),
            "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
            "latency_ms": reground_latency(requests),
            "extra_execute_ms": g0111.dist([r["ms"] for r in requests if r.get("ms") is not None]),
        }
    vram = [t["vram_min_free_mib"] for t in trials if t.get("vram_min_free_mib") is not None]
    safety = {
        **base["safety"],
        "reground_voiced": sum(len(t.get("reground_voiced") or []) for t in trials),
        "reground_other_tool_executed": sum(
            1
            for s in steps
            if s.get("executed_name") is not None and s.get("executed_name") != s.get("call_name")
        ),
    }
    return {
        "visual_tokens": int(visual_tokens),
        "reground": bool(reground),
        "acceptance": acceptance,
        "outcomes": outcomes,
        "success": {
            "k": sum(o["k"] for o in outcomes.values()),
            "n": sum(o["n"] for o in outcomes.values()),
        },
        "clicks": click_stats(steps, "click"),
        "clicks_p1": click_stats(steps, "click_p1") if reground else None,
        "steps": {**base["steps"], "scrolls": scrolls},
        "false_claims": sum(1 for t in trials if false_claim(t, by_turn.get(_key(t), []))),
        "latency": base["latency"],
        "reground_cost": cost,
        "cache": base["cache"],
        "wall_s": g0111.dist([t["wall_s"] for t in trials if t.get("wall_s") is not None]),
        "vram_min_free_mib": min(vram) if vram else None,
        "engine_up": not any(t.get("engine_up_after") is False for t in trials),
        "safety": safety,
    }


#: the sections ``derive_arm`` returns and ``report.json`` holds per arm
ARM_SECTIONS = (
    "visual_tokens",
    "reground",
    "acceptance",
    "outcomes",
    "success",
    "clicks",
    "clicks_p1",
    "steps",
    "false_claims",
    "latency",
    "reground_cost",
    "cache",
    "wall_s",
    "vram_min_free_mib",
    "engine_up",
    "safety",
)


def decision_inputs(arm: Mapping[str, Any]) -> dict[str, Any]:
    """D5's inputs for one arm, read off its D3 section."""
    acc = arm["acceptance"]
    cost = arm.get("reground_cost")
    return {
        "accepted": bool(acc["all_200"]) and acc["turns_ok"] == acc["turns"],
        "safety": sum(int(v) for v in arm["safety"].values()),
        "engine_up": bool(arm["engine_up"]),
        "success": int(arm["success"]["k"]),
        "hit_rate": arm["clicks"]["hit_rate"],
        "done_p95_ms": arm["latency"]["done_ms"]["p95"],
        "reground_p50_ms": cost["latency_ms"]["p50"] if cost else None,
    }


def eligible_arms(inputs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """D5 eligibility, in ``ARMS`` order (fewest visual tokens first, then ReGround off before on). "The best arm" is
    read as the best of all six arms (the hand tables of the unit module do not depend on that reading)."""
    best_success = max(int(i["success"]) for i in inputs.values())
    rates = [float(i["hit_rate"]) for i in inputs.values() if i["hit_rate"] is not None]
    best_rate = max(rates) if rates else None
    out: list[str] = []
    for b, r in ARMS:
        label = arm_label(b, r)
        i = inputs[label]
        ok = bool(i["accepted"]) and int(i["safety"]) == 0 and bool(i["engine_up"])
        ok = ok and int(i["success"]) >= best_success - SUCCESS_WITHIN
        ok = ok and i["hit_rate"] is not None and best_rate is not None
        ok = ok and float(i["hit_rate"]) >= best_rate - HIT_RATE_WITHIN - 1e-9
        ok = ok and i["done_p95_ms"] is not None and float(i["done_p95_ms"]) <= DONE_P95_MAX_MS
        if r:
            ok = (
                ok
                and i["reground_p50_ms"] is not None
                and float(i["reground_p50_ms"]) <= REGROUND_P50_MAX_MS
            )
        if ok:
            out.append(label)
    return out


def decide(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
) -> str:
    """The outcome label, in this order (ruling at 4337125): "inconclusive" if P1 failed; "void" if the run is void
    (P2 or P3 failed); "incomplete" if the run stopped early (``stopped`` is its reason); then D5: "no change" if no
    arm is eligible or the chosen arm beats 280/off by at most 2 of 25, otherwise the chosen arm's label (the first
    eligible arm in ``ARMS`` order)."""
    if not p1_passed:
        return INCONCLUSIVE
    if void:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    eligible = eligible_arms(inputs)
    if not eligible:
        return NO_CHANGE
    chosen = eligible[0]
    if int(inputs[chosen]["success"]) - int(inputs[BASELINE]["success"]) <= NO_CHANGE_GAIN:
        return NO_CHANGE
    return chosen


def derive(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    turn_lines: Sequence[Mapping[str, Any]] | None = None,
    *,
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``'s ``void``, ``stopped`` (why the run stopped early, or None), ``arms`` (all six labels, each
    ``derive_arm``) and ``decision`` (``eligible`` is ``[]`` under the three labels that are not D5's)."""
    arms: dict[str, dict[str, Any]] = {}
    for b, r in ARMS:
        label = arm_label(b, r)
        arms[label] = derive_arm(
            b,
            r,
            [t for t in trials if arm_of(t) == label],
            [s for s in steps if arm_of(s) == label],
            [g for g in regrounds if arm_of(g) == label],
            turn_lines,
        )
    inputs = {label: decision_inputs(a) for label, a in arms.items()}
    p1_passed = bool(probes["p1"]["ok"])
    void = not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3"))
    label = decide(inputs, p1_passed=p1_passed, void=void, stopped=stopped)
    return {
        "void": void,
        "stopped": stopped,
        "arms": arms,
        "decision": {
            "label": label,
            "eligible": [] if label in NOT_D5 else eligible_arms(inputs),
            "inputs": inputs,
        },
    }


def _p1_delta_ok(budget: int, delta: Any) -> bool:
    """A missing delta (``None``: ``/tokenize`` gave no count) is a failed P1 delta (ruling at 4337125)."""
    return delta is not None and p1_ok(budget, delta)


def probe_findings(probes: Mapping[str, Any]) -> list[str]:
    """D2 P1 re-derived: each budget's frame and crop deltas against ``p1_ok`` (a ``None`` delta fails, never a
    ``TypeError``), and ``p1.ok`` as their conjunction."""
    p1 = probes["p1"]
    out: list[str] = []
    verdicts: list[bool] = []
    for budget in (280, 560, 1120):
        row = p1["budgets"][str(budget)]
        verdicts += [_p1_delta_ok(budget, row[which]) for which in ("frame_delta", "crop_delta")]
        want = _p1_delta_ok(budget, row["frame_delta"]) and _p1_delta_ok(budget, row["crop_delta"])
        if bool(row["ok"]) is not want:
            out.append(f"P1 {budget}: recorded ok={row['ok']}, re-derived {want} ({row})")
    if bool(p1["ok"]) is not all(verdicts):
        out.append(f"P1: recorded ok={p1['ok']}, re-derived {all(verdicts)}")
    return out


def config_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The configuration evidence (ruling at 4337125), one string per finding:

    - every trial that forwarded a body (``scanned_requests`` non-empty) has ``budgets_sent == [its arm's budget]``
      and ``guidance_sent is True`` (a turn that sent nothing -- not run -- is not judged);
    - in a ReGround-on arm, every executed ``click_at``/``type_at`` step has a reground line (same arm, task, repeat
      and step); in a ReGround-off arm there is no reground line at all.

    This is what tells an arm's configuration from the arm's label: a client built at 280 in every arm, a session
    built with ReGround off, or a session built without the live guidance each leave the labels intact."""
    out: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        where = f"{arm_of(t)} {t['task']}/r{t['repeat']}"
        want = [int(t["visual_tokens"])]
        if t.get("budgets_sent") != want:
            out.append(f"{where}: budgets_sent {t.get('budgets_sent')!r}, the arm sends {want}")
        if t.get("guidance_sent") is not True:
            out.append(f"{where}: guidance_sent {t.get('guidance_sent')!r}, not True")
    lines = {
        (arm_of(g), str(g["task"]), int(g["repeat"]), int(g["step"]))
        for g in regrounds
        if g.get("step") is not None
    }
    for g in regrounds:
        if not g["reground"]:
            out.append(
                f"{arm_of(g)} {g['task']}/r{g['repeat']}: a reground line in a ReGround-off arm"
            )
    for s in steps:
        if not s["reground"]:
            continue
        if s.get("status") != "executed" or s.get("executed_name") not in CLICK_TOOLS:
            continue
        k = (arm_of(s), str(s["task"]), int(s["repeat"]), int(s["step"]))
        if k not in lines:
            out.append(
                f"{k[0]} {k[1]}/r{k[2]} step {k[3]}: an executed {s.get('executed_name')} with no reground line"
            )
    return out


def scan_coverage_gaps(
    trials: Sequence[Mapping[str, Any]], sessions: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """D4: every ``llm.request`` attempt in a session log has its label (``SCAN_PHASE[phase] + ":" + attempt_id``)
    in the ``scanned_requests`` of the trial whose ``session_id`` owns the attempt (T-0111g's check, with the
    ``reground`` phase mapped to ``generate:``). Returns ``"<file>: <label> (<why>)"`` per gap."""
    by_session = {str(t.get("session_id")): t for t in trials if t.get("session_id")}
    gaps: list[str] = []
    for name in sorted(sessions):
        for r in sessions[name]:
            if r.get("kind") != "llm.request":
                continue
            data = r.get("data") or {}
            attempt = str(data.get("attempt_id"))
            phase = SCAN_PHASE.get(str(data.get("phase")))
            if phase is None:
                gaps.append(f"{name}: {data.get('phase')}:{attempt} (an unknown request phase)")
                continue
            label = f"{phase}:{attempt}"
            sid = attempt.split("-", 1)[0]
            trial = by_session.get(sid)
            if trial is None:
                gaps.append(f"{name}: {label} (no trial has session {sid})")
            elif label not in (trial.get("scanned_requests") or []):
                gaps.append(f"{name}: {label} (not in the trial's scanned_requests)")
    return gaps


# =====================================================================================================
# Reading the run
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            assert isinstance(item, dict), f"{path.name}: a line is not an object"
            out.append(item)
    return out


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(
            f"{ENV} is not set: this row reads a T-0131 A/B run (scripts/locked/t0131_grounding_ab.sh)"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    if not (path / "report.json").is_file():
        pytest.skip(
            f"{path} has no report.json: the A/B did not finish, so there is nothing to re-derive"
        )
    for name in ("trials.jsonl", "steps.jsonl", "regrounds.jsonl", "manifest.json", "probes.json"):
        assert (path / name).is_file(), f"T-0131 D1: {path} has a report.json but no {name}"
    return path


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert isinstance(report, dict), "report.json is not an object"
    return {
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "regrounds": _jsonl(run_dir / "regrounds.jsonl"),
        "turn_lines": _jsonl(run_dir / "turns.jsonl"),
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "probes": json.loads((run_dir / "probes.json").read_text(encoding="utf-8")),
        "report": report,
    }


@pytest.fixture(scope="module")
def derived(run: dict[str, Any]) -> dict[str, Any]:
    # the run's own record of an early stop: the harness writes it into manifest.json when it stops
    return derive(
        run["trials"],
        run["steps"],
        run["regrounds"],
        run["probes"],
        run["turn_lines"],
        stopped=run["manifest"].get("stopped"),
    )


@pytest.fixture(scope="module")
def sessions(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    files = sorted((run_dir / "sessions").glob("*.jsonl"))
    return {p.name: _jsonl(p) for p in files}


# =====================================================================================================
# The rows
# =====================================================================================================


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """D1: the manifest pins the arm table and T-0111g's task table and holds the 150-turn joint order (each arm 25
    times); unless the run is void, 150 trial lines, one per (arm, task, repeat); step and reground lines join a
    trial; ReGround-off arms have no reground line; every reground outcome is a B3 label."""
    trials, steps, regrounds, manifest = (
        run["trials"],
        run["steps"],
        run["regrounds"],
        run["manifest"],
    )
    assert manifest.get("arm_table_sha256") == ARM_TABLE_SHA256, manifest.get("arm_table_sha256")
    assert manifest.get("task_table_sha256") == TASK_TABLE_SHA256, manifest.get("task_table_sha256")
    order = [tuple(x) for x in manifest.get("order") or []]
    expected = {(b, r, t, k) for b, r in ARMS for t in TASK_IDS for k in range(N_PER_TASK)}
    assert len(order) == TOTAL_TURNS == len(set(order)), Counter(order)
    assert set(order) == expected, sorted(set(order) ^ expected)[:10]
    if run["report"].get("void"):
        assert trials == [], "a void run (a probe failed) ran arm turns"
        return
    keys = [
        (int(t["visual_tokens"]), bool(t["reground"]), str(t["task"]), int(t["repeat"]))
        for t in trials
    ]
    assert len(keys) == TOTAL_TURNS == len(set(keys)), Counter(keys)
    assert set(keys) == expected, sorted(set(keys) ^ expected)[:10]
    turn_keys = set(keys)

    def k(x: Mapping[str, Any]) -> tuple[int, bool, str, int]:
        return (int(x["visual_tokens"]), bool(x["reground"]), str(x["task"]), int(x["repeat"]))

    assert [k(s) for s in steps if k(s) not in turn_keys] == [], "step lines for no trial"
    assert [k(g) for g in regrounds if k(g) not in turn_keys] == [], "reground lines for no trial"
    assert [k(g) for g in regrounds if not g["reground"]] == [], (
        "a ReGround-off arm wrote reground lines"
    )
    odd = [g.get("outcome") for g in regrounds if g.get("outcome") not in OUTCOMES]
    assert odd == [], f"reground outcomes outside B3: {odd}"


def test_probes_re_derived(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """D2: ``report.json`` echoes ``probes.json``; P1's verdict per budget (frame and 512x320 crop) re-derived from
    the recorded token deltas (273/522/1092 + 0..4; a missing delta fails); the report is void iff a probe failed,
    labelled "inconclusive" when P1 failed and "void" when only P2 or P3 did (ruling at 4337125), with no eligible
    arm either way."""
    probes, report = run["probes"], run["report"]
    assert report.get("probes") == probes, "report.json's probes are not probes.json"
    assert probe_findings(probes) == []
    assert report.get("void") is derived["void"], (report.get("void"), derived["void"])
    if not probes["p1"]["ok"]:
        assert report["decision"]["label"] == INCONCLUSIVE, report["decision"]
    elif report.get("void"):
        assert report["decision"]["label"] == VOID, report["decision"]
    if report.get("void"):
        assert report["decision"]["eligible"] == [], report["decision"]


@pytest.mark.parametrize("section", ARM_SECTIONS)
def test_arm_section_re_derived(run: dict[str, Any], derived: dict[str, Any], section: str) -> None:
    """D3, per arm: every metric section re-derived from the jsonl files (success from ``__state``, clicks re-scored
    from points and boxes, reground cost from ``regrounds.jsonl``, latency from the step lines only, the eight safety
    counts with egress re-counted from ``turns.jsonl``)."""
    arms = run["report"].get("arms") or {}
    assert sorted(arms) == sorted(LABELS), sorted(arms)
    diff = {
        label: {"report": arms[label].get(section), "derived": derived["arms"][label][section]}
        for label in LABELS
        if arms[label].get(section) != derived["arms"][label][section]
    }
    assert diff == {}, json.dumps(diff, indent=1, default=str)


def test_click_records_rescored(run: dict[str, Any]) -> None:
    """D3: a ``click`` record exactly on executed ``click_at``/``type_at`` steps (and ``click_p1`` only in ReGround-on
    arms, only beside a ``click``), each with exactly the five record keys (``read_error`` included, ruling at
    4337125); each stored ``hit``/``error_px`` agrees with ``click_score`` over its own point and boxes."""
    bad: list[str] = []
    for s in run["steps"]:
        where = f"{arm_of(s)} {s['task']}/r{s['repeat']}/step {s['step']}"
        executed_click = s.get("status") == "executed" and s.get("executed_name") in CLICK_TOOLS
        if (s.get("click") is not None) != executed_click:
            bad.append(
                f"{where}: click record {s.get('click') is not None}, executed click {executed_click}"
            )
        if s.get("click_p1") is not None and (not s["reground"] or s.get("click") is None):
            bad.append(f"{where}: click_p1 without a ReGround-on click")
        for field in ("click", "click_p1"):
            c = s.get(field)
            if c is None:
                continue
            if set(c) != CLICK_KEYS:
                bad.append(f"{where} {field}: keys {sorted(c)}, not {sorted(CLICK_KEYS)}")
                continue
            hit, err = click_score(c["point"], c["boxes"])
            same_err = (err is None and c.get("error_px") is None) or (
                err is not None
                and c.get("error_px") is not None
                and abs(err - c["error_px"]) < 1e-6
            )
            if bool(c.get("hit")) is not hit or not same_err:
                bad.append(
                    f"{where} {field}: stored ({c.get('hit')}, {c.get('error_px')}), re-scored ({hit}, {err})"
                )
    assert bad == [], "\n".join(bad)


def test_decision_re_derived(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """D5: the decision inputs, the eligible arms and the outcome label, re-derived (H-T131-1's numbers, the reground
    p50 with each timeout at 4000 ms; ruling at 4337125: "inconclusive", then "void", then "incomplete", then D5)."""
    got = run["report"].get("decision") or {}
    want = derived["decision"]
    assert got.get("inputs") == want["inputs"], json.dumps(
        {"report": got.get("inputs"), "derived": want["inputs"]}, indent=1, default=str
    )
    assert got.get("eligible") == want["eligible"], (got.get("eligible"), want["eligible"])
    assert got.get("label") == want["label"], (got.get("label"), want["label"])


def test_stopped_re_derived(run_dir: Path, run: dict[str, Any]) -> None:
    """Ruling at 4337125: ``report.json``'s ``stopped`` is the manifest's record of an early stop (the reason, or
    null); a trial written as not run implies the run stopped; a stopped run is labelled "incomplete" and
    ``report.md`` carries the reason."""
    report, manifest = run["report"], run["manifest"]
    assert "stopped" in report, "report.json has no stopped key"
    assert report["stopped"] == manifest.get("stopped"), (
        report["stopped"],
        manifest.get("stopped"),
    )
    not_run = [t for t in run["trials"] if str(t.get("exception") or "").startswith("not run")]
    if not_run:
        assert report["stopped"] is not None, f"{len(not_run)} trials not run, but stopped is null"
    if report["stopped"] is not None:
        assert report["decision"]["label"] == INCOMPLETE, report["decision"]
        md = (run_dir / "report.md").read_text(encoding="utf-8")
        assert str(report["stopped"]) in md, "report.md does not carry the stop reason"


def test_configuration_evidence(run: dict[str, Any]) -> None:
    """Ruling at 4337125: every trial that sent a body sent its arm's budget and only it (``budgets_sent == [N]``;
    a client built at 280 in every arm fails the four arms above 280) and carried the live guidance in every body;
    every executed click in a ReGround-on arm was regrounded (a reground line), and an off arm has none."""
    findings = config_findings(run["trials"], run["steps"], run["regrounds"])
    assert findings == [], "\n".join(findings)


def test_safety_implications(run_dir: Path, run: dict[str, Any]) -> None:
    """T-0111g's two independent implications, run-wide: voiced text outside its own step's committed text implies
    ``voiced_uncommitted`` > 0 in that arm; a start-page path in the session's records implies ``url_path_or_query``
    > 0 somewhere."""
    arms = run["report"].get("arms") or {}
    for label in LABELS:
        mine = [s for s in run["steps"] if arm_of(s) == label]
        leaked = g0111.uncommitted_voicing(mine)
        if leaked:
            assert arms[label]["safety"]["voiced_uncommitted"] > 0, (label, leaked)
    lines = g0111.url_lines(run_dir)
    if lines:
        assert sum(arms[label]["safety"]["url_path_or_query"] for label in LABELS) > 0, lines


def test_sentinel_scan_covers_every_request(run: dict[str, Any], sessions: Any) -> None:
    """D4 coverage: every ``llm.request`` attempt in ``sessions/*.jsonl`` (prefill, generate, tool_step, reground,
    tool_repair) has its label in its trial's ``scanned_requests``, with ``reground`` scanned as ``generate:``."""
    gaps = scan_coverage_gaps(run["trials"], sessions)
    assert gaps == [], "requests the sentinel scan never read:\n" + "\n".join(gaps)


def test_egress_lines_all_owned(run: dict[str, Any]) -> None:
    """Every ``turns.jsonl`` line belongs to a trial by ``session_id``, so the per-arm egress re-count saw every
    turn."""
    _count, unmatched = g0111.egress_from_turns(run["trials"], run["turn_lines"])
    assert unmatched == [], f"turns.jsonl lines of sessions no trial names: {unmatched}"
