"""T-0131 D -- the 6-arm grounding A/B of ``tools = live``: 150 typed turns through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0131_grounding_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the arm table (``ARMS``, ``arm_table_sha``,
``order``), the arm profile (``arm_profile``), the probe verdict (``p1_ok``), the scoring (``click_score``,
``false_claim``, ``reground_voiced_findings``), the decision (``decide``), the report (``build_report``,
``render_markdown``) and the record extraction (``step_lines``, ``reground_lines``) are pure, and everything engine-
or browser-facing is imported inside ``main`` and the turn runner.

D1: T-0111g's harness (``scripts/gpu/t0111g_live_smoke.py``) is **imported, never copied**: ``TASKS``, ``score``,
``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``, ``non_loopback_sockets`` and ``turn_records`` are
its objects, so ``TASK_TABLE_SHA256`` is its pin; its ``_Watch`` wraps the browser, and its refusals, engine checks,
``/metrics`` reader and report builder are reused per arm.

Refusals (exit 2), all **before any socket** opens and before any browser, start-page server or ``nvidia-smi``:
T-0111g's six (a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task
table off its pin, a tool kit that cannot load / is not ``bbfc53ccdd80`` / has repair off, Chromium absent), plus an
arm table whose ``arm_table_sha()`` is not ``ARM_TABLE_SHA256``. Past them the engine is checked (``/v1/models`` and
``/metrics`` on the loopback URL): with no engine the harness exits 1 having launched no browser.

D2 pre-flight probes, before any arm turn (any failure: exit 3, a void report, no arm turn):

- **P1** for each budget: ``/tokenize`` (chat form, ``mm_processor_kwargs: {max_soft_tokens: N}``) of a fixed
  1280x800 frame prompt with and without its image, and of the same with a 512x320 crop; each difference must be
  ``expected_soft_tokens(W, H, N)`` (273/522/1092) plus 0..4 image-delimiter tokens.
- **P2** one single-image generate at 1120 returns 200 and the engine is healthy after it.
- **P3** one text-only generate carrying the kwarg returns 200.

What it runs (*Regime*): ``order(ORDER_SEED)``, 150 ``(visual_tokens, reground, task, repeat)`` turns shuffled
jointly over every arm, task and repeat, concurrency 1, the engine unseeded. Each turn is T-0111g's turn (one fresh
start-page server, one fresh headless ``BrowserToolExecutor``, one fresh ``Session`` with one ``UserText``) with
``arm_profile``'s profile (``tools = live``, headless, ``visual_tokens`` explicit in every arm, 280 included,
``reground`` per arm), a ``VllmClient(visual_tokens=N)`` behind ``LoopMeter``, and ``Session(tool_guidance=
LIVE_TOOL_GUIDANCE, visual_tokens=N, reground=...)``. Before each dispatched ``click_at``/``type_at`` the watch reads,
in the agent page, the boxes of the task's intended elements (``INTENDED_TARGETS``) and ``elementFromPoint`` at the
click (a read-only ``page.evaluate``, before delegating; the read is diagnostic and lands in ``clicks.jsonl``). Free
VRAM is sampled every 200 ms by ``nvidia-smi`` (read-only) from the probes to the end. The run stops after
``ERRORED_STOP`` consecutive errored turns, or when ``RUN_BUDGET_S`` is spent (inside the locked script's 90 min, so
the report is still written); the rest are written as not run.

Output (``--out-dir``, default ``logs/t0131-grounding-ab-<sha7>/``): ``probes.json``, ``trials.jsonl``,
``steps.jsonl`` and ``regrounds.jsonl`` (the pinned keys of ``tests/unit/test_t0131_ab_harness.py``'s docstring),
``clicks.jsonl`` (the raw box reads), ``turns.jsonl`` and ``sessions/*.jsonl`` (the sessions' own records),
``manifest.json`` (the regime, both table pins, the gate-file shas, the order), ``report.json`` and ``report.md``.
Exit codes: 0 the run was measured; 1 no engine, or the run stopped early; 2 refused; 3 a probe failed (void).
The report's label (ruling at 4337125): "inconclusive" when P1 failed, "void" when P2 or P3 failed, "incomplete"
when the run stopped early (``stopped`` carries why), otherwise D5's rule ("no change" or the chosen arm).

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native ``bbfc53ccdd80``, live, repair on, loop 8; headless Chromium at 1280x800, DSF 1, loopback start
pages only; typed turns, fake STT/TTS/audio, no headset or speakers; 6 arms x 5 tasks x n = 5 = 150 turns, order seed
0, concurrency 1, the engine unseeded; STT stopped. History, not like-for-like: T-0111g's run at ca9576f.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_astra.config import Profile

REPO = Path(__file__).resolve().parents[2]
#: T-0111g's harness, read-only (D1: imported, never copied)
T0111G_PATH: Path = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
#: the fixed ``sys.modules`` name T-0111g is loaded under (its dataclasses look themselves up there)
T0111G_MODULE_NAME = "_t0111g_live_smoke_for_t0131"


def _load_t0111g() -> ModuleType:
    """T-0111g's harness, imported read-only by path under ``T0111G_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0111G_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0111G_MODULE_NAME, T0111G_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0111G_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0111G_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0111G_MODULE_NAME, None)
        raise
    return module


G: ModuleType = _load_t0111g()

# --- imported from T-0111g, unchanged (the same objects) -------------------------------------------------------
TASKS = G.TASKS
score = G.score
LoopMeter = G.LoopMeter
sentinel_hits = G.sentinel_hits
url_leaks = G.url_leaks
log_complete = G.log_complete
non_loopback_sockets = G.non_loopback_sockets
turn_records = G.turn_records
TASK_TABLE_SHA256 = G.TASK_TABLE_SHA256
TOOLS_SHA12 = G.TOOLS_SHA12
#: the tree is read through this name (T-0111g's seam); ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: D1: (visual tokens, ReGround), the task file's table
ARMS: tuple[tuple[int, bool], ...] = (
    (280, False),
    (280, True),
    (560, False),
    (560, True),
    (1120, False),
    (1120, True),
)
#: ``arm_table_sha()`` of the table above (the pin of tests/unit/test_t0131_ab_harness.py, 2026-10-22)
ARM_TABLE_SHA256 = "5cc0169fe0cf4612f982c68fc20c2feb63ae803d6f70ceff372ed4d6209f32e2"
#: *Regime*: the 150 turns shuffled jointly with seed 0
ORDER_SEED = 0
BROWSER = "headless"
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
#: a pass 2 aborted by ``REGROUND_TIMEOUT_MS``: out of D3's acceptance, in the reground cost (ruling at 23c6c77)
ABORTED = "pass2_timeout"
SKIPPED = "skipped_no_point"
#: the session's pass-2 bound (``session.REGROUND_TIMEOUT_MS``; ``main`` refuses when they differ). An aborted pass 2
#: counts as this many ms in ``reground_cost.latency_ms``, so the D5 p50 gate covers every attempt (ruling at 4337125)
REGROUND_TIMEOUT_MS = 4000
#: D3: executed calls that carry a click record
CLICK_TOOLS = ("click_at", "type_at")
#: D2 P1: at most this many image-delimiter tokens on top of the soft tokens
P1_DELIMITERS = 4
#: D2 P1: the probe frame and the ReGround crop (zoom 0.4 of it)
P1_FRAME = (1280, 800)
P1_CROP = (512, 320)
#: D2 P2/P3: the budget of the load probe and of the text-only probe
P2_BUDGET = 1120
#: D5, H-T131-1 approved as proposed (2026-10-22)
SUCCESS_WITHIN = 3
HIT_RATE_WITHIN = 0.10
DONE_P95_MAX_MS = 1500
REGROUND_P50_MAX_MS = 1000
NO_CHANGE_GAIN = 2
BASELINE = "280/off"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
#: ruling at 4337125: P2 or P3 failed (P1 passed); and the run stopped early (``ERRORED_STOP`` or ``RUN_BUDGET_S``)
VOID = "void"
INCOMPLETE = "incomplete"
#: D3: T-0111g's six safety counts and the two new ones, each expected 0
SAFETY = (*G.SAFETY, "reground_voiced", "reground_other_tool_executed")
#: D2: the free-VRAM sampling period (``nvidia-smi``, read-only)
VRAM_PERIOD_MS = 200
#: the box read inside the watch, before the dispatch
BOX_READ_S = 5.0
#: the wall-time budget for the turns: inside the locked script's ``timeout 90m``, so the report is still written
RUN_BUDGET_S = 85 * 60
#: the gate files whose sha256 pins a run to its commit (T-0111g's, plus T-0131's)
GATE_FILES = (
    *G.GATE_FILES,
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/perception/policy.py",
    "local_astra/toolcall/reground.py",
)
#: the T-0111g run this A/B is set beside (history, not like-for-like)
HISTORY = "T-0111g C-G1 live smoke at ca9576f: 4/25 turns succeeded (no guidance, implicit 280, no ReGround)"


def _say(msg: str) -> None:
    print(f"t0131_grounding_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The arm table and the order (D1)
# =====================================================================================================


def arm_label(visual_tokens: int, reground: bool) -> str:
    """``"280/off"``, ``"1120/on"``: an arm's report key, and D5's label for a chosen arm."""
    return f"{int(visual_tokens)}/{'on' if reground else 'off'}"


def _arm_of(record: Mapping[str, Any]) -> str:
    return arm_label(int(record["visual_tokens"]), bool(record["reground"]))


def arm_table_sha(arms: Sequence[tuple[int, bool]] | None = None) -> str:
    """sha256 of the canonical JSON of ``arms`` (``None``: the module's ``ARMS``, read at call time)."""
    rows = [{"visual_tokens": b, "reground": r} for b, r in (ARMS if arms is None else arms)]
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def order(seed: int) -> list[tuple[int, bool, str, int]]:
    """Every ``(visual_tokens, reground, task_id, repeat)`` of ``ARMS`` x ``TASKS``, arm-major, then shuffled once
    by ``random.Random(seed)`` -- jointly, so the arms are interleaved (no set or hash ordering anywhere, so the order
    is the same in every process)."""
    turns = [(int(b), bool(r), t.task_id, k) for b, r in ARMS for t in TASKS for k in range(t.n)]
    random.Random(seed).shuffle(turns)
    return turns


def arm_profile(profile: Profile, visual_tokens: int, reground: bool) -> Profile:
    """A validated copy of ``profile`` with ``tools = live``, ``browser = headless``, ``visual_tokens`` set
    explicitly (280 too: ``None`` would send no kwarg) and ``reground`` per arm. ``profile`` is not changed."""
    data = profile.model_dump()
    data.update(
        tools="live", browser=BROWSER, visual_tokens=int(visual_tokens), reground=bool(reground)
    )
    return type(profile).model_validate(data)


# =====================================================================================================
# Intended click targets (D3; by DOM text and role, the pages unchanged)
# =====================================================================================================


@dataclass(frozen=True)
class Target:
    """An intended element: its role, its accessible text, and (for an "Add to cart" button) its ``data-item``."""

    role: str
    name: str
    item: str | None = None


def _intended_targets() -> dict[str, tuple[Target, ...]]:
    cheapest, _price = G.cheapest_deal()
    add = "Add to cart"
    return {
        "S1": (
            Target("searchbox", "Search"),
            Target("link", "Blue mug"),
            Target("button", add, "blue mug"),
        ),
        "S2": (Target("button", add, "blue mug"),),
        "S3": (
            Target("textbox", "Name"),
            Target("textbox", "Email"),
            Target("textbox", "Message"),
            Target("button", "Send"),
        ),
        "S4": (),  # none expected: its clicks are reported, unscored
        "S5": (Target("button", add, cheapest),),
    }


INTENDED_TARGETS: Mapping[str, Sequence[Target]] = _intended_targets()

#: read-only: the viewport boxes of the visible intended elements, and the element at the click point
BOXES_JS = """
([targets, x, y]) => {
  const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
  const boxes = [];
  for (const t of targets) {
    let els = [];
    if (t.role === "searchbox") {
      els = Array.from(document.querySelectorAll('input[type="search"]'))
        .filter((e) => norm(e.getAttribute("aria-label")) === t.name);
    } else if (t.role === "textbox") {
      els = Array.from(document.querySelectorAll("label[for]"))
        .filter((l) => norm(l.textContent) === t.name)
        .map((l) => document.getElementById(l.htmlFor))
        .filter((e) => e !== null);
    } else if (t.role === "link") {
      els = Array.from(document.querySelectorAll("a")).filter((a) => {
        const title = a.querySelector(".result-title");
        return norm(title ? title.textContent : a.textContent) === t.name;
      });
    } else if (t.role === "button") {
      els = Array.from(document.querySelectorAll("button")).filter(
        (b) => norm(b.textContent) === t.name && (t.item === null || b.dataset.item === t.item));
    }
    for (const el of els) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) boxes.push({x: r.x, y: r.y, width: r.width, height: r.height});
    }
  }
  let element = null;
  if (x !== null && y !== null) {
    const at = document.elementFromPoint(x, y);
    if (at) {
      const item = at.dataset && at.dataset.item ? '[data-item="' + at.dataset.item + '"]' : "";
      element = at.tagName.toLowerCase() + item + " " + norm(at.textContent).slice(0, 60);
    }
  }
  return {boxes: boxes, element: element};
}
"""


# =====================================================================================================
# Pure scoring (D2, D3)
# =====================================================================================================


def p1_ok(budget: int, delta: int) -> bool:
    """D2 P1: the image's token count (with minus without) is ``expected_soft_tokens(1280, 800, budget)`` plus 0..4
    image-delimiter tokens; below is "not honoured", above is not the expected count."""
    from local_astra.perception.policy import expected_soft_tokens

    expected = expected_soft_tokens(P1_FRAME[0], P1_FRAME[1], int(budget))
    return expected <= int(delta) <= expected + P1_DELIMITERS


def click_score(
    point: Sequence[float], boxes: Sequence[Mapping[str, float]]
) -> tuple[bool, float | None]:
    """D3: ``(hit, error_px)``. A hit is inside an intended box, edges included; the error is the Euclidean distance
    to the nearest intended box (0.0 inside one); ``(False, None)`` when no intended box was on the page."""
    px, py = float(point[0]), float(point[1])
    best: float | None = None
    for b in boxes:
        x0, y0 = float(b["x"]), float(b["y"])
        x1, y1 = x0 + float(b["width"]), y0 + float(b["height"])
        d = math.hypot(max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1))
        if best is None or d < best:
            best = d
    if best is None:
        return False, None
    return best == 0.0, best


def _click_record(
    point: Sequence[float], boxes: Sequence[Mapping[str, float]], read_error: str | None = None
) -> dict[str, Any]:
    """A step's ``click``/``click_p1``: the point, the boxes, their score, and ``read_error`` (the box read's
    failure, ``None`` when it succeeded; a failed read leaves no box, so the click scores ``(False, None)``)."""
    hit, err = click_score(point, boxes)
    return {
        "point": list(point),
        "boxes": [dict(b) for b in boxes],
        "hit": hit,
        "error_px": err,
        "read_error": read_error,
    }


def false_claim(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """D3's proxy for "claims success": not a success (by ``score``), finished (no exception, not timed out), and
    the last step made no call (``status`` and ``call_name`` None) and voiced text with no ``?``."""
    if trial.get("exception") or trial.get("timed_out") or not steps:
        return False
    if score(str(trial["task"]), trial.get("state"), G._voiced_text(trial)):
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
    """D3 ``reground_voiced``: each pass-2 answer text (whitespace-collapsed, non-empty) found inside a voiced
    source or the stored history and inside none of the turn's own step texts (pass 2 sees pass 1's prompt, so it may
    repeat what a step legitimately said)."""
    targets = [G._collapse(v) for v in voiced] + ([G._collapse(history)] if history else [])
    own = [G._collapse(t) for t in step_texts]
    out: list[str] = []
    for answer in pass2_answers:
        text = G._collapse(answer)
        if text and any(text in t for t in targets) and not any(text in o for o in own):
            out.append(text)
    return out


# =====================================================================================================
# The report (pure; report.json) and the decision (D5)
# =====================================================================================================


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def _click_stats(steps: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    """D3 over the steps' ``name`` records (``click`` or ``click_p1``), re-scored from each record's point and
    boxes; S4 has no intended target, so its clicks are ``unscored`` and out of the hit rate. ``read_errors`` counts
    the clicks (S4's included) whose box read failed (ruling at 4337125)."""
    n = hits = unscored = read_errors = 0
    errors: list[float] = []
    for s in steps:
        c = s.get(name)
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
        "error_px": {**G._dist(errors), "mean": _mean(errors)},
        "unscored": unscored,
        "read_errors": read_errors,
    }


def _reground_latencies(requests: Sequence[Mapping[str, Any]]) -> list[float]:
    """D3/D5's pass-2 latency: each request's ``done_ms``, and ``REGROUND_TIMEOUT_MS`` for a ``pass2_timeout`` (it
    has no ``done_ms``), so the p50 gate covers every attempt and cannot hide timeouts (ruling at 4337125). Another
    request without ``done_ms`` (an error with no stream end) has no latency."""
    out: list[float] = []
    for r in requests:
        if r.get("outcome") == ABORTED:
            out.append(float(REGROUND_TIMEOUT_MS))
        elif r.get("done_ms") is not None:
            out.append(r["done_ms"])
    return out


def _arm_section(
    visual_tokens: int,
    reground: bool,
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's D3 section: T-0111g's ``build_report`` over the arm's records (acceptance, outcomes, steps,
    latency from the step lines only, cache, its six safety counts), plus clicks, scrolls, false claims, the reground
    cost, wall time, VRAM, the engine flag and the two new safety counts."""
    base = G.build_report(trials, steps)
    by_turn: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_turn.setdefault(_key(s), []).append(s)
    requests = [r for r in regrounds if r.get("outcome") != SKIPPED]
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
        task: G._dist(
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
    cost: dict[str, Any] | None = None
    if reground:
        counts = Counter(str(r.get("outcome")) for r in regrounds)
        cost = {
            "requests": len(requests),
            "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
            "latency_ms": G._dist(_reground_latencies(requests)),
            "extra_execute_ms": G._dist([r["ms"] for r in requests if r.get("ms") is not None]),
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
        "clicks": _click_stats(steps, "click"),
        "clicks_p1": _click_stats(steps, "click_p1") if reground else None,
        "steps": {**base["steps"], "scrolls": scrolls},
        "false_claims": sum(1 for t in trials if false_claim(t, by_turn.get(_key(t), []))),
        "latency": base["latency"],
        "reground_cost": cost,
        "cache": base["cache"],
        "wall_s": G._dist([t["wall_s"] for t in trials if t.get("wall_s") is not None]),
        "vram_min_free_mib": min(vram) if vram else None,
        "engine_up": not any(t.get("engine_up_after") is False for t in trials),
        "safety": safety,
    }


def _decision_inputs(arm: Mapping[str, Any]) -> dict[str, Any]:
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


def _eligible(inputs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """D5 eligibility, in ``ARMS`` order (fewest visual tokens first, then ReGround off before on). "The best arm"
    is the best of all six."""
    best_success = max(int(i["success"]) for i in inputs.values())
    rates = [float(i["hit_rate"]) for i in inputs.values() if i["hit_rate"] is not None]
    best_rate = max(rates) if rates else None
    out: list[str] = []
    for b, r in ARMS:
        label = arm_label(b, r)
        i = inputs[label]
        ok = bool(i["accepted"]) and int(i["safety"]) == 0 and bool(i["engine_up"])
        ok = ok and int(i["success"]) >= best_success - SUCCESS_WITHIN
        ok = (
            ok
            and i["hit_rate"] is not None
            and best_rate is not None
            and float(i["hit_rate"]) >= best_rate - HIT_RATE_WITHIN - 1e-9
        )
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
    """D5: "inconclusive" if P1 failed; "void" if P2 or P3 failed (``void``, P1 passed); "incomplete" if the run
    stopped early (``stopped``, its reason); then the rule: "no change" if no arm is eligible or the chosen arm (the
    first eligible in ``ARMS`` order) beats 280/off by at most 2 of 25; otherwise the chosen arm's label. The three
    run labels are the ruling at 4337125, in that precedence."""
    if not p1_passed:
        return INCONCLUSIVE
    if void:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    eligible = _eligible(inputs)
    if not eligible:
        return NO_CHANGE
    chosen = eligible[0]
    if int(inputs[chosen]["success"]) - int(inputs[BASELINE]["success"]) <= NO_CHANGE_GAIN:
        return NO_CHANGE
    return chosen


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    *,
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: ``void`` (any probe failed), ``stopped`` (why the run stopped early, or ``None``), the
    probes, every arm's section, and the decision. ``eligible`` is listed only when D5's rule gave the label (never
    for "inconclusive", "void" or "incomplete")."""
    arms: dict[str, dict[str, Any]] = {}
    for b, r in ARMS:
        label = arm_label(b, r)
        arms[label] = _arm_section(
            b,
            r,
            [t for t in trials if _arm_of(t) == label],
            [s for s in steps if _arm_of(s) == label],
            [g for g in regrounds if _arm_of(g) == label],
        )
    inputs = {label: _decision_inputs(a) for label, a in arms.items()}
    p1_passed = bool(probes["p1"]["ok"])
    void = not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3"))
    label = decide(inputs, p1_passed=p1_passed, void=void, stopped=stopped)
    by_rule = label not in (INCONCLUSIVE, VOID, INCOMPLETE)
    return {
        "void": void,
        "stopped": stopped,
        "probes": dict(probes),
        "arms": arms,
        "decision": {
            "label": label,
            "eligible": _eligible(inputs) if by_rule else [],
            "inputs": inputs,
        },
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _px(value: Any) -> str:
    """A click error in report.md: two decimals for p50, p95 and the mean alike."""
    return "-" if value is None else f"{float(value):.2f}"


def _dist_cell(d: Mapping[str, Any]) -> str:
    return f"n={d['n']} p50={_fmt(d['p50'])} p95={_fmt(d['p95'])}"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the probes, every arm's numbers, the decision label, and T-0111g's run beside
    them (history, not like-for-like)."""
    lines = ["# T-0131 D grounding A/B (report-only, plus the pre-registered decision rule)", ""]
    if manifest is not None:
        lines += ["## Regime", "", "```", json.dumps(manifest.get("regime"), indent=1), "```", ""]
    decision = report["decision"]
    lines += [
        "## Decision (D5, H-T131-1)",
        "",
        f"- outcome label: **{decision['label']}**",
        f"- eligible arms: {', '.join(decision['eligible']) or 'none'}",
        f"- void: {report['void']}"
        + (" (a probe failed: the run measured nothing)" if report["void"] else ""),
        f"- stopped early: {_fmt(report.get('stopped'))}",
        "",
        "## Probes (D2)",
        "",
    ]
    probes = report["probes"]
    for budget, row in sorted(probes["p1"]["budgets"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"- P1 {budget}: frame delta {_fmt(row['frame_delta'])}, crop delta {_fmt(row['crop_delta'])}, "
            f"expected {row['expected']} (+0..{P1_DELIMITERS}), ok {row['ok']}"
        )
    lines += [
        f"- P2 (one image at {P2_BUDGET}): HTTP {_fmt(probes['p2']['http_status'])}, healthy after "
        f"{probes['p2']['healthy_after']}, ok {probes['p2']['ok']}",
        f"- P3 (text only, with the kwarg): HTTP {_fmt(probes['p3']['http_status'])}, ok {probes['p3']['ok']}",
        "",
        "## Arms (D3; each number report-only)",
        "",
        "| arm | success | hit rate | click error px (p50/p95/mean) | false claims | step done p95 ms | "
        "reground p50 ms | wall s p50 | min free VRAM MiB | accepted | safety | engine up |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, arm in report["arms"].items():
        clicks = arm["clicks"]
        err = clicks["error_px"]
        inputs = decision["inputs"][label]
        lines.append(
            f"| {label} | {arm['success']['k']}/{arm['success']['n']} | {_fmt(clicks['hit_rate'])} "
            f"(n={clicks['n']}, unscored {clicks['unscored']}, read errors {clicks['read_errors']}) | "
            f"{_px(err['p50'])}/{_px(err['p95'])}/{_px(err['mean'])} | {arm['false_claims']} | {_fmt(arm['latency']['done_ms']['p95'])} | "
            f"{_fmt(inputs['reground_p50_ms'])} | {_fmt(arm['wall_s']['p50'])} | "
            f"{_fmt(arm['vram_min_free_mib'])} | {inputs['accepted']} | {inputs['safety']} | "
            f"{arm['engine_up']} |"
        )
    for label, arm in report["arms"].items():
        lines += ["", f"### {label}", ""]
        lines += [f"- {task}: {o['k']}/{o['n']}" for task, o in arm["outcomes"].items()]
        st = arm["steps"]
        lines += [
            f"- steps per turn: {_dist_cell(st['per_turn'])}",
            "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
            f"- scrolls per S4 turn: {_dist_cell(st['scrolls']['S4'])}; per S5 turn: "
            f"{_dist_cell(st['scrolls']['S5'])}",
        ]
        lines += [f"- refused: {n} x {reason}" for reason, n in st["refusal_reasons"].items()]
        lat = arm["latency"]
        lines += [
            f"- step first commit ms: {_dist_cell(lat['first_commit_ms'])}; done ms: "
            f"{_dist_cell(lat['done_ms'])} (reground requests excluded)",
            f"- prefix cache hit tokens: {_dist_cell(arm['cache']['hit_tokens'])}; query tokens: "
            f"{_dist_cell(arm['cache']['query_tokens'])}",
        ]
        if arm["clicks_p1"] is not None:
            p1 = arm["clicks_p1"]
            lines.append(
                f"- pass 1's counterfactual clicks: hit rate {_fmt(p1['hit_rate'])} (n={p1['n']}), error px "
                f"p50 {_px(p1['error_px']['p50'])} p95 {_px(p1['error_px']['p95'])} mean "
                f"{_px(p1['error_px']['mean'])}"
            )
        cost = arm["reground_cost"]
        if cost is not None:
            outcomes = ", ".join(f"{k} {v}" for k, v in cost["outcomes"].items())
            lines += [
                f"- reground requests: {cost['requests']} ({outcomes})",
                f"- reground latency ms: {_dist_cell(cost['latency_ms'])}; extra execute ms per coordinate "
                f"step: {_dist_cell(cost['extra_execute_ms'])}",
            ]
        acc = arm["acceptance"]
        lines += [
            f"- acceptance: steps at 200 {acc['ok']}/{acc['steps']}, turns {acc['turns_ok']}/{acc['turns']}, "
            f"regrounds at 200 {acc['regrounds_ok']}/{acc['regrounds']}, all 200 {acc['all_200']}",
            "- safety: " + ", ".join(f"{k} {v}" for k, v in arm["safety"].items()),
        ]
    lines += ["", "## History (not like-for-like)", "", f"- {HISTORY}", ""]
    return "\n".join(lines)


# =====================================================================================================
# Record extraction (pure, over one turn's session records)
# =====================================================================================================


def _reground_attempts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str((r.get("data") or {}).get("attempt_id"))
        for r in records
        if r.get("kind") == "llm.request" and (r.get("data") or {}).get("phase") == "reground"
    ]


def _answers(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each attempt's committed answer text (answer segments outside an open thought), joined."""
    out: dict[str, list[str]] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") != "llm.commit" or d.get("segment") != "answer" or d.get("thought_open"):
            continue
        out.setdefault(str(d.get("attempt_id")), []).append(str(d.get("text") or ""))
    return {k: "".join(v) for k, v in out.items()}


def reground_lines(
    task_id: str,
    repeat: int,
    *,
    records: Sequence[Mapping[str, Any]],
    generates: Sequence[Mapping[str, Any]],
    visual_tokens: int,
    reground: bool,
) -> list[dict[str, Any]]:
    """One ``regrounds.jsonl`` line per ``tool.reground`` record, joined to its ``LoopMeter`` row by the pass-2
    attempt id: ``http_status`` 0 for a request without a response (an abort included), and the request fields
    ``None`` for ``skipped_no_point``."""
    rows = {str(g.get("request_id")): g for g in generates}
    answers = _answers(records)
    out: list[dict[str, Any]] = []
    for r in records:
        if r.get("kind") != "tool.reground":
            continue
        d = r.get("data") or {}
        outcome = str(d.get("outcome"))
        attempt = d.get("attempt_id")
        sent = outcome != SKIPPED and attempt is not None
        row = rows.get(str(attempt)) if sent else None
        status = (row or {}).get("http_status")
        out.append(
            {
                "visual_tokens": int(visual_tokens),
                "reground": bool(reground),
                "task": task_id,
                "repeat": repeat,
                "step": d.get("step"),
                "call_id": d.get("call_id"),
                "call_name": d.get("name"),
                "outcome": outcome,
                "ms": d.get("ms"),
                "attempt_id": str(attempt) if sent else None,
                "http_status": (status if status is not None else 0) if sent else None,
                "first_commit_ms": row.get("first_commit_ms") if row is not None else None,
                "done_ms": row.get("done_ms") if row is not None else None,
                "answer_text": answers.get(str(attempt), "") if sent else "",
                "tool_text": d.get("tool_text"),
            }
        )
    return out


def step_lines(
    steps: Sequence[Mapping[str, Any]],
    *,
    visual_tokens: int,
    reground: bool,
    clicks: Mapping[str, Mapping[str, Any]],
    executed: Mapping[str, str],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """T-0111g's step lines plus ``visual_tokens``, ``reground``, ``executed_name`` (from the watch) and the click
    records: ``click`` on every executed ``click_at``/``type_at`` (the boxes read before its dispatch), and in
    ReGround-on arms ``click_p1``, pass 1's point (the ``tool.reground`` record's ``p1``; the executed point when no
    pass 2 changed it) scored against the same boxes."""
    from local_astra.conductor.tool_executor import to_pixels

    p1_of: dict[str, Mapping[str, Any]] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") == "tool.reground" and isinstance(d.get("p1"), Mapping):
            p1_of[str(d.get("call_id"))] = d["p1"]
    out: list[dict[str, Any]] = []
    for s in steps:
        cid = s.get("call_id")
        name = executed.get(str(cid)) if cid is not None else None
        read = clicks.get(str(cid)) if cid is not None else None
        click: dict[str, Any] | None = None
        click_p1: dict[str, Any] | None = None
        if (
            s.get("status") == "executed"
            and name in CLICK_TOOLS
            and read is not None
            and read.get("point") is not None
        ):
            boxes = read.get("boxes") or []
            error = read.get("error")
            click = _click_record(read["point"], boxes, error)
            if reground:
                point = read["point"]
                p1 = p1_of.get(str(cid))
                vw, vh = read["viewport"]
                if p1 is not None and isinstance(p1.get("x"), int) and isinstance(p1.get("y"), int):
                    point = [to_pixels(p1["x"], vw), to_pixels(p1["y"], vh)]
                click_p1 = _click_record(point, boxes, error)
        out.append(
            {
                **s,
                "visual_tokens": int(visual_tokens),
                "reground": bool(reground),
                "executed_name": name,
                "click": click,
                "click_p1": click_p1,
            }
        )
    return out


def _not_run(task_id: str, repeat: int, reason: str, budget: int, reground: bool) -> dict[str, Any]:
    trial: dict[str, Any] = G._not_run(task_id, repeat, reason)
    trial.update(
        visual_tokens=int(budget),
        reground=bool(reground),
        wall_s=None,
        vram_min_free_mib=None,
        engine_up_after=None,
        reground_voiced=[],
        budgets_sent=[],
        guidance_sent=False,
    )
    return trial


def _system_text(body: Mapping[str, Any]) -> str:
    """The text of a request body's system message(s): string content, or its text parts joined."""
    out: list[str] = []
    for m in body.get("messages") or []:
        if not isinstance(m, Mapping) or m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, Mapping) and p.get("type") == "text"
            ]
    return "".join(out)


def config_evidence(
    bodies: Mapping[str, Mapping[str, Any]], guidance: str
) -> tuple[list[int], bool]:
    """The trial's ``budgets_sent`` and ``guidance_sent`` (ruling at 4337125), from the bodies built for the
    sentinel scan: the sorted distinct ``mm_processor_kwargs.max_soft_tokens`` values, and whether ``guidance`` is in
    the system text of every body (``False`` with no body: nothing was shown to have carried it)."""
    budgets: set[int] = set()
    for body in bodies.values():
        kwargs = body.get("mm_processor_kwargs")
        value = kwargs.get("max_soft_tokens") if isinstance(kwargs, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool):
            budgets.add(value)
    sent = bool(bodies) and all(guidance in _system_text(b) for b in bodies.values())
    return sorted(budgets), sent


def _errored(
    trial: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> bool:
    """T-0111g's errored turn, or a gated pass 2 (not skipped, not the harness's own timeout) off 200."""
    if G._errored(trial, steps):
        return True
    return any(
        r.get("outcome") not in (SKIPPED, ABORTED) and r.get("http_status") != 200
        for r in regrounds
    )


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


def _post_json(url: str, body: Mapping[str, Any], timeout_s: float = 60.0) -> tuple[int, Any]:
    """``(HTTP status, parsed body or None)``; status 0 when no response came back. No proxy, no redirect."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with G._opener().open(req, timeout=timeout_s) as resp:
            raw: bytes = resp.read()
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, None


def _healthy(base_url: str) -> bool:
    return G._get_text(G._root(base_url) + "/health", 10.0) is not None


class _VramSampler:
    """Free device memory every ``VRAM_PERIOD_MS`` from one ``nvidia-smi`` loop (read-only), timestamped on the
    monotonic clock; ``min_between`` answers per turn, ``None`` when nothing was sampled."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self.error: str | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    str(VRAM_PERIOD_MS),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            self.error = repr(exc)
            _say(f"VRAM sampling unavailable: {exc!r}")
            return
        self._thread = threading.Thread(target=self._read, name="vram-sampler", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            with contextlib.suppress(ValueError, IndexError):
                mib = int(line.strip().split(",")[0])
                with self._lock:
                    self.samples.append((time.monotonic(), mib))

    def min_between(self, t0: float, t1: float) -> int | None:
        with self._lock:
            values = [v for t, v in self.samples if t0 <= t <= t1]
        return min(values) if values else None

    def stop(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(OSError):
                self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                self._proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)


def _probe_jpegs() -> tuple[bytes, bytes]:
    """The fixed P1 frame (1280x800, a synthetic page-like picture) and its centred 512x320 crop."""
    import io

    from PIL import Image, ImageDraw

    w, h = P1_FRAME
    img = Image.new("RGB", (w, h), (246, 246, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, 56), fill=(32, 48, 80))
    for i in range(3):
        x0 = 80 + i * 400
        draw.rectangle((x0, 140, x0 + 340, 560), outline=(180, 180, 180), width=2)
        draw.rectangle((x0 + 30, 170, x0 + 310, 380), fill=(60 + 60 * i, 90, 200 - 50 * i))
        draw.rectangle((x0 + 110, 480, x0 + 231, 521), fill=(20, 110, 60))
    cw, ch = P1_CROP
    left, top = (w - cw) // 2, (h - ch) // 2
    crop = img.crop((left, top, left + cw, top + ch))
    out: list[bytes] = []
    for picture in (img, crop):
        buf = io.BytesIO()
        picture.save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out[0], out[1]


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


PROBE_TEXT = "Describe what is on this screen in one sentence."


def _tokenize_count(base_url: str, model: str, budget: int, jpeg: bytes | None) -> int | None:
    """``/tokenize`` (chat form) of the probe prompt with ``mm_processor_kwargs``, with or without the image."""
    content: list[dict[str, Any]] = []
    if jpeg is not None:
        content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
    content.append({"type": "text", "text": PROBE_TEXT})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "add_generation_prompt": True,
        "mm_processor_kwargs": {"max_soft_tokens": int(budget)},
    }
    status, payload = _post_json(G._root(base_url) + "/tokenize", body)
    if status != 200 or not isinstance(payload, dict):
        return None
    count = payload.get("count")
    if isinstance(count, int):
        return count
    tokens = payload.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def probe_p1(base_url: str, model: str) -> dict[str, Any]:
    """D2 P1, per budget: the frame's and the crop's image-token deltas against ``p1_ok``."""
    from local_astra.perception.policy import expected_soft_tokens

    frame, crop = _probe_jpegs()
    budgets: dict[str, dict[str, Any]] = {}
    for budget in sorted({b for b, _ in ARMS}):
        bare = _tokenize_count(base_url, model, budget, None)
        deltas: dict[str, int | None] = {}
        for which, jpeg in (("frame_delta", frame), ("crop_delta", crop)):
            with_image = _tokenize_count(base_url, model, budget, jpeg)
            deltas[which] = (
                with_image - bare if with_image is not None and bare is not None else None
            )
        ok = all(d is not None and p1_ok(budget, d) for d in deltas.values())
        budgets[str(budget)] = {
            **deltas,
            "expected": expected_soft_tokens(P1_FRAME[0], P1_FRAME[1], budget),
            "ok": ok,
        }
    return {"ok": all(v["ok"] for v in budgets.values()), "budgets": budgets}


async def _probe_generate(base_url: str, jpeg: bytes | None, request_id: str) -> int:
    """One generate at ``P2_BUDGET`` (with one image, or text only); its HTTP status, 0 without a response."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart

    parts: list[Any] = [ImagePart(url=_data_url(jpeg))] if jpeg is not None else []
    parts.append(TextPart(text=PROBE_TEXT))
    prompt = Prompt(messages=(Message(role="user", content=tuple(parts)),), prefix_len=0)
    client = VllmClient(
        VllmEndpoint(base_url=base_url),
        transport=httpx.AsyncHTTPTransport(),
        visual_tokens=P2_BUDGET,
    )
    try:
        async for _commit in client.generate(prompt, request_id, LlmOptions()):
            pass
    except Exception as exc:
        status = G._http_status(exc)
        _say(f"probe {request_id}: {exc!r}")
        return status if status is not None else 0
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
    return 200


def run_probes(base_url: str, model: str) -> dict[str, Any]:
    """D2: P1, then P2 and P3 (both even when P1 failed, so the void report says what else held)."""
    frame, _crop = _probe_jpegs()
    p1 = probe_p1(base_url, model)
    p2_status = asyncio.run(_probe_generate(base_url, frame, "t0131-probe-p2"))
    healthy = _healthy(base_url)
    p3_status = asyncio.run(_probe_generate(base_url, None, "t0131-probe-p3"))
    return {
        "p1": p1,
        "p2": {
            "ok": p2_status == 200 and healthy,
            "http_status": p2_status,
            "healthy_after": healthy,
        },
        "p3": {"ok": p3_status == 200, "http_status": p3_status},
    }


# =====================================================================================================
# The turn runner
# =====================================================================================================


class _GroundingWatch:
    """T-0111g's ``_Watch`` (every name delegated; each execute's outcome and whether it raised), plus D3's box read:
    before a ``click_at``/``type_at`` is delegated, the intended elements' boxes and ``elementFromPoint`` at the click
    are read in the agent page (read-only ``page.evaluate``), and the executed tool is recorded per call id."""

    def __init__(self, inner: Any, targets: Sequence[Target]) -> None:
        self._inner = inner
        self._watch = G._Watch(inner)
        self._targets = [{"role": t.role, "name": t.name, "item": t.item} for t in targets]
        self.clicks: dict[str, dict[str, Any]] = {}
        self.executed: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def raised(self) -> list[str]:
        raised: list[str] = self._watch.raised
        return raised

    async def _read_boxes(self, inv: Any) -> dict[str, Any]:
        from local_astra.conductor.tool_executor import to_pixels

        x, y = inv.arguments.get("x"), inv.arguments.get("y")
        read: dict[str, Any] = {
            "call_id": str(inv.call_id),
            "name": str(inv.name),
            "point": None,
            "viewport": None,
            "boxes": [],
            "element": None,
            "error": None,
        }
        try:
            vw, vh = self._inner.driver.viewport
            read["viewport"] = [int(vw), int(vh)]
            ok = all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 999 for v in (x, y)
            )
            if ok:
                read["point"] = [to_pixels(x, vw), to_pixels(y, vh)]
            px, py = read["point"] if read["point"] is not None else (None, None)
            got = await asyncio.wait_for(
                self._inner.driver.page.evaluate(BOXES_JS, [self._targets, px, py]),
                timeout=BOX_READ_S,
            )
            read["boxes"] = [
                {k: float(b[k]) for k in ("x", "y", "width", "height")}
                for b in (got or {}).get("boxes") or []
            ]
            read["element"] = (got or {}).get("element")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            read["error"] = repr(exc)
        return read

    async def execute(self, inv: Any) -> Any:
        if inv.name in CLICK_TOOLS:
            # D3: before the dispatch, outside every timed request span
            self.clicks[str(inv.call_id)] = await self._read_boxes(inv)
        out = await self._watch.execute(inv)
        if getattr(out, "status", None) == "executed":
            self.executed[str(inv.call_id)] = str(inv.name)
        return out


@dataclass
class _Run:
    base_url: str
    out: Path
    profiles: dict[tuple[int, bool], Any]
    sentinel: bytes
    manifest: dict[str, Any]
    vram: _VramSampler
    started: float
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


async def _one_turn(
    run: _Run, budget: int, reground: bool, task_id: str, repeat: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """T-0111g's turn with the arm's profile, budget, ReGround and the live guidance; the records read after the
    session closed. Returns the trial, its step lines, its reground lines and its raw box reads."""
    import httpx

    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    window_start = time.monotonic()
    profile = run.profiles[(budget, reground)]
    task = G._task(task_id)
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (T-0111g Implementation notes, Event loop)
    pages = StartPageServer()
    pages.start()
    browser: Any = None
    watch: _GroundingWatch | None = None
    session: Any = None
    meter: Any = None
    inner: Any = None
    exception: str | None = None
    timed_out = False
    history: str | None = None
    state: Any = None
    sockets: list[str] = []
    log: Any = None
    wall_s: float | None = None
    origin = pages.origin
    try:
        browser = BrowserToolExecutor(
            now_ms=lambda: clock.now_ms,
            viewport=profile.browser_viewport,
            guard=None,
            headless=True,
            end_when_closed=False,
            allowed_origins=frozenset({origin}),
        )
        watch = _GroundingWatch(browser, INTENDED_TARGETS[task_id])
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=watch)
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        inner = VllmClient(
            VllmEndpoint(base_url=run.base_url),
            transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
            detokenizer=detokenizer,
            visual_tokens=budget,  # A2: explicit in every arm, 280 included
        )
        meter = LoopMeter(inner, metrics=lambda: G.read_metrics(run.base_url), clock=time.monotonic)
        hud = Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=1280,
            height=800,
            jpeg=run.sentinel,
            phash=hashlib.sha256(run.sentinel).hexdigest()[:16],
        )
        sessions_dir = run.out / "sessions"
        log = SessionLog(sessions_dir)
        policy = TurnPolicy(
            thinking=profile.diffusion.thinking, tool_loop=True, max_tool_steps=G.MAX_TOOL_STEPS
        )
        session = Session(
            TurnMachine(policy),
            meter,
            G._fake("stt", "FakeStt")(),
            G._fake("tts", "FakeTts")(),
            G._PlaybackOnly(G._fake("audio", "FakeAudioIO")(clock)),
            G._fake("perception", "FakePerception")([hud]),
            G._CaseMemory(),
            clock=clock,
            config=SessionConfig(
                run.out / "data",
                turns_path=run.out / "turns.jsonl",
                profile=profile,
                audio_path="null",
                source=Source.SCREEN,
            ),
            session_log=log,
            log_dir=sessions_dir,
            tool_kit=kit,
            tool_guidance=LIVE_TOOL_GUIDANCE,  # C1: the harness builds Session directly (ruling on A/C)
            visual_tokens=budget,  # A3: the turn line
            reground=reground,
        )
        await asyncio.wait_for(browser.open(origin + task.start_page), timeout=G.BROWSER_CALL_S)
        await session.start()
        session.turn_written.clear()
        submitted = time.monotonic()
        session.submit(E.UserText(t_ms=clock.now_ms, text=task.request))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=G.TURN_TIMEOUT_S)
        except TimeoutError:
            timed_out = True
        wall_s = round(time.monotonic() - submitted, 2)
        for entry in reversed(tuple(session.state.history)):
            if entry.user_text.strip() == task.request.strip():
                history = entry.assistant_spoken or None
                break
    except Exception as exc:
        exception = repr(exc)
    finally:
        # read while the browser is still open (the session's close closes it)
        if browser is not None:
            state = await G._read_state(browser)
        sockets = non_loopback_sockets(G._descendants())
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        if meter is not None:
            with contextlib.suppress(Exception):
                await meter.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    if session is not None and session.failures and exception is None:
        exception = "session failures: " + "; ".join(repr(f) for f in session.failures)
    engine_up = await asyncio.to_thread(_healthy, run.base_url)
    vram_min = run.vram.min_between(window_start, time.monotonic())

    # --- after close, outside every timed span: which request bodies held the HUD sentinel's bytes --------------
    # (every request the meter forwarded: the prefill, every generate attempt, every pass 2, every repair)
    sentinel_requests: list[str] = []
    scanned_requests: list[str] = []
    generates: list[dict[str, Any]] = []
    budgets_sent: list[int] = []
    guidance_sent = False
    if meter is not None:
        generates = [g for g in meter.generates if not g.get("repair")]
        raw = {
            label: inner.request_body(prompt, label.split(":", 1)[1])
            for label, prompt in meter.requests.items()
        }
        bodies = {label: json.dumps(body) for label, body in raw.items()}
        sentinel_requests = sentinel_hits(bodies, run.sentinel)
        scanned_requests = sorted(bodies)
        # ruling at 4337125: the configuration each forwarded body actually carried
        budgets_sent, guidance_sent = config_evidence(raw, LIVE_TOOL_GUIDANCE)

    records: list[dict[str, Any]] = []
    record_lines: list[tuple[str, str]] = []
    if log is not None and Path(log.path).is_file():
        text = Path(log.path).read_text(encoding="utf-8")
        records = sorted(G._read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
        record_lines += [
            (f"sessions/{Path(log.path).name}:{n}", ln)
            for n, ln in enumerate(text.splitlines(), start=1)
        ]
    lines = []
    if session is not None:
        turns_path = run.out / "turns.jsonl"
        if turns_path.is_file():
            for n, ln in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not ln.strip():
                    continue
                item = json.loads(ln)
                if isinstance(item, dict) and item.get("session_id") == session.session_id:
                    lines.append(item)
                    record_lines.append((f"turns.jsonl:{n}", ln))
    trial, steps = turn_records(
        task_id,
        repeat,
        records=records,
        line=lines[0] if lines else None,
        generates=generates,
        sentinel_requests=sentinel_requests,
        scanned_requests=scanned_requests,
        raised_calls=list(watch.raised) if watch is not None else [],
        state=state,
        history=history,
        exception=exception,
        timed_out=timed_out,
        non_loopback=sockets,
        leaks=url_leaks(record_lines, origin),
        engine_idle_before=idle_before,
    )
    steps = step_lines(
        steps,
        visual_tokens=budget,
        reground=reground,
        clicks=watch.clicks if watch is not None else {},
        executed=watch.executed if watch is not None else {},
        records=records,
    )
    regrounds = reground_lines(
        task_id,
        repeat,
        records=records,
        generates=meter.generates if meter is not None else [],
        visual_tokens=budget,
        reground=reground,
    )
    answers = _answers(records)
    pass2 = [answers.get(a, "") for a in _reground_attempts(records)]
    findings = reground_voiced_findings(
        pass2,
        trial.get("voiced") or [],
        trial.get("history"),
        [str(s.get("step_text") or "") for s in steps],
    )
    trial.update(
        visual_tokens=int(budget),
        reground=bool(reground),
        wall_s=wall_s,
        vram_min_free_mib=vram_min,
        engine_up_after=engine_up,
        reground_voiced=[f"{task_id}/r{repeat}: {f}" for f in findings],
        budgets_sent=budgets_sent,
        guidance_sent=guidance_sent,
    )
    trial["session_log"] = Path(log.path).name if log is not None else None
    trial["deferred_prefill_aborts"] = len(meter.deferred_aborts) if meter is not None else 0
    reads = [
        {"visual_tokens": budget, "reground": reground, "task": task_id, "repeat": repeat, **c}
        for c in (watch.clicks.values() if watch is not None else [])
    ]
    return trial, steps, regrounds, reads


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``ERRORED_STOP`` consecutive errored turns, or past
    ``RUN_BUDGET_S``, nothing more is sent and the rest are written as not run, so the files still hold 150 trials."""
    errored = 0
    for index, (budget, reground, task_id, repeat) in enumerate(order(ORDER_SEED)):
        started = time.monotonic()
        if run.stopped is None and started - run.started > RUN_BUDGET_S:
            run.stopped = f"the run budget of {RUN_BUDGET_S} s is spent (at turn {index})"
            run.manifest["stopped"] = run.stopped
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.stopped}")
        steps: list[dict[str, Any]] = []
        regrounds: list[dict[str, Any]] = []
        reads: list[dict[str, Any]] = []
        if run.stopped is not None:
            trial = _not_run(task_id, repeat, f"not run: {run.stopped}", budget, reground)
        else:
            try:
                trial, steps, regrounds, reads = await _one_turn(
                    run, budget, reground, task_id, repeat
                )
            except Exception as exc:
                trial = _not_run(task_id, repeat, f"harness: {exc!r}", budget, reground)
            errored = errored + 1 if _errored(trial, steps, regrounds) else 0
            if errored >= G.ERRORED_STOP:
                run.stopped = f"{errored} consecutive errored turns (last: {task_id}/r{repeat})"
                run.manifest["stopped"] = run.stopped
                G._write_json(run.out / "manifest.json", run.manifest)
                _say(f"stopping: {run.stopped}")
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "regrounds.jsonl", regrounds)
        G._append_jsonl(run.out / "clicks.jsonl", reads)
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:03d} {arm_label(budget, reground)} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"reground={[r.get('outcome') for r in regrounds]} success={trial.get('success')} "
            f"timed_out={trial.get('timed_out')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _write_report(
    out: Path, probes: Mapping[str, Any], manifest: Mapping[str, Any], stopped: str | None = None
) -> dict[str, Any]:
    trials = G._read_jsonl(out / "trials.jsonl")
    steps = G._read_jsonl(out / "steps.jsonl")
    regrounds = G._read_jsonl(out / "regrounds.jsonl")
    report = build_report(trials, steps, regrounds, probes, stopped=stopped)
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    # 1. loopback, unresolved
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    # 2. a clean tree
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    # 3. no out-dir yet (empty or not: one run per SHA)
    out: Path = args.out_dir or REPO / "logs" / f"t0131-grounding-ab-{sha[:7]}"
    if out.exists():
        _say(
            f"refusing: {out} exists; the run happens once per SHA (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. the task table's pin (T-0111g's)
    table_sha = G.task_table_sha(TASKS)
    if table_sha != TASK_TABLE_SHA256:
        _say(f"refusing: the task table's sha is {table_sha}, not the pinned {TASK_TABLE_SHA256}")
        return EXIT_REFUSED
    # 5. the arm table's pin (D1)
    arms_sha = arm_table_sha()
    if arms_sha != ARM_TABLE_SHA256:
        _say(f"refusing: the arm table's sha is {arms_sha}, not the pinned {ARM_TABLE_SHA256}")
        return EXIT_REFUSED
    # 6. the tool kit (T-0115's three refusals, repair off included)
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    try:
        profiles = {(b, r): arm_profile(lean, b, r) for b, r in ARMS}
    except ValueError as exc:
        _say(f"refusing: an arm's profile is invalid: {exc}")
        return EXIT_REFUSED
    try:
        kit, _detokenizer = cli.load_tool_kit(lean)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if str(kit.tools_version)[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {str(kit.tools_version)[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(f"refusing: the tool repair is off ({kit.repair_off}); the regime says repair on")
        return EXIT_REFUSED
    # 7. Chromium (T-0111g B2's availability check; reads the filesystem only)
    missing = cli.live_prerequisites(BROWSER)
    if missing is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {missing}")
        return EXIT_REFUSED
    # 8. the pass-2 bound the latency rule counts a timeout as (ruling at 4337125)
    from local_astra.conductor import session as session_module

    if session_module.REGROUND_TIMEOUT_MS != REGROUND_TIMEOUT_MS:
        _say(
            f"refusing: the session's REGROUND_TIMEOUT_MS is {session_module.REGROUND_TIMEOUT_MS}, not the "
            f"{REGROUND_TIMEOUT_MS} ms the reground latency rule counts a pass2_timeout as"
        )
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE

    model = VllmEndpoint().model
    listing = G._get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if G.read_metrics(args.base_url) is None:
        _say(
            f"{G._root(args.base_url)}/metrics has no prefix-cache counters; the per-step hit cannot be measured"
        )
        return EXIT_FAILED

    vram = _VramSampler()
    vram.start()
    try:
        return _measure(
            args.base_url,
            out,
            sha,
            model,
            lean,
            profiles,
            kit,
            vram,
            LIVE_TOOL_GUIDANCE,
            REGROUND_TIMEOUT_MS,
        )
    finally:
        vram.stop()


def _measure(
    base_url: str,
    out: Path,
    sha: str,
    model: str,
    lean: Any,
    profiles: dict[tuple[int, bool], Any],
    kit: Any,
    vram: _VramSampler,
    guidance: str,
    reground_timeout_ms: int,
) -> int:
    """Past every refusal and the engine check: the manifest, the probes, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    try:
        from importlib.metadata import version

        playwright_version: str | None = version("playwright")
    except Exception:
        playwright_version = None
    manifest: dict[str, Any] = {
        "regime": {
            "commit": sha,
            "base_url": base_url,
            "served_model": model,
            "engine_version": G._get_text(G._root(base_url) + "/version"),
            "profile": lean.name,
            "canvas_length": lean.diffusion.canvas_length,
            "max_denoising_steps": lean.diffusion.max_denoising_steps,
            "thinking": lean.diffusion.thinking,
            "tools": f"native:{str(kit.tools_version)[:12]}:live:loop{G.MAX_TOOL_STEPS}",
            "repair": "on",
            "max_tool_steps": G.MAX_TOOL_STEPS,
            "arms": [{"visual_tokens": b, "reground": r} for b, r in ARMS],
            "reground_timeout_ms": reground_timeout_ms,
            "live_tool_guidance_sha256": hashlib.sha256(guidance.encode("utf-8")).hexdigest(),
            "browser": f"{BROWSER}, one per turn, fresh context, loopback start pages only",
            "viewport": list(lean.browser_viewport),
            "playwright": playwright_version,
            "turns": len(order(ORDER_SEED)),
            "concurrency": 1,
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "turn_timeout_s": G.TURN_TIMEOUT_S,
            "run_budget_s": RUN_BUDGET_S,
            "vram_period_ms": VRAM_PERIOD_MS,
            "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
            "no audio device, no headset or speakers",
            "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
            "history": HISTORY,
        },
        "arm_table_sha256": arm_table_sha(),
        "task_table_sha256": G.task_table_sha(TASKS),
        "tasks": [
            {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": t.n}
            for t in TASKS
        ],
        "intended_targets": {
            task: [{"role": t.role, "name": t.name, "item": t.item} for t in targets]
            for task, targets in INTENDED_TARGETS.items()
        },
        "cheapest_deal": list(G.cheapest_deal()),
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {rel: G._sha256(REPO / rel) for rel in GATE_FILES},
        "order": [list(k) for k in order(ORDER_SEED)],
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "regrounds.jsonl", "clicks.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- D2: the probes, before any arm turn ---------------------------------------------------------
    probe_start = time.monotonic()
    probes = run_probes(base_url, model)
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    G._write_json(out / "manifest.json", manifest)
    if not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3")):
        report = _write_report(out, probes, manifest)
        _say(
            f"a probe failed: the run is void ({json.dumps(probes)}); label {report['decision']['label']}"
        )
        return EXIT_PROBE

    run = _Run(
        base_url=base_url,
        out=out,
        profiles=profiles,
        sentinel=sentinel,
        manifest=manifest,
        vram=vram,
        started=time.monotonic(),
    )
    asyncio.run(_run_all(run))
    report = _write_report(out, probes, manifest, run.stopped)
    _say(f"decision={report['decision']['label']} eligible={report['decision']['eligible']}")
    _say(f"written to {out}")
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
