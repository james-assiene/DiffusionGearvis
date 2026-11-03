"""T-0132 C -- the 3-arm crop-budget A/B of ``tools = live`` at 1120: 300 typed turns through a real ``Session``.
``/launch`` ONLY.

    uv run python scripts/gpu/t0132_crop_budget_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the arm table (``ARMS``, ``arm_table_sha``,
``order``), the arm profile (``arm_profile``), the intervals (``wilson``, ``newcombe``), the configuration-evidence
rule (``pass2_budget_ok``), the decision (``decide``, ``decide_with_reason``), P4's prompt (``p4_continuation``,
``p4_prompt``), the report (``build_report``, ``render_markdown``) and the record extraction (``reground_lines``) are
pure, and everything engine- or browser-facing is imported inside ``main`` and the turn runner.

C1: T-0131's harness (``scripts/gpu/t0131_grounding_ab.py``) is **imported, never copied**, and through it T-0111g's:
``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``, ``non_loopback_sockets``,
``turn_records`` (T-0111g's objects) and ``INTENDED_TARGETS``, ``BOXES_JS``, ``click_score``, ``false_claim``,
``reground_voiced_findings``, ``probe_p1``, ``run_probes``, ``config_evidence`` (T-0131's), plus its per-arm report
section, its box-reading watch, its record extraction, its VRAM sampler and its engine helpers. The turn runner is
this file's own, because it builds the second (pass-2) client: every ``VllmClient`` is wrapped in a ``LoopMeter``, the
second one included, so every body is scanned for the HUD sentinel and every request is timed.

Refusals (exit 2), all **before any socket** opens and before any browser, start-page server or ``nvidia-smi``:
T-0131's (a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task table off
its pin, a tool kit that cannot load / is not ``bbfc53ccdd80`` / has repair off, Chromium absent, the session's
``REGROUND_TIMEOUT_MS`` other than the 4000 ms the latency rule counts a timeout as), with this file's arm table: an
``arm_table_sha()`` other than ``ARM_TABLE_SHA256``. Past them the engine is checked (``/v1/models`` and ``/metrics``
on the loopback URL): with no engine the harness exits 1 having launched no browser.

C2 probes: P1-P3 are T-0131's ``run_probes`` before any arm turn (any failure: exit 3, a void or inconclusive report,
no arm turn). **P4, the worst-case window**, runs after the arms, so it cannot void them: ``p4_prompt()`` (4 distinct
1280x800 frames at 1120, the live system message with the tools and the live guidance, and a continuation of exactly
``MAX_CONTINUATION`` characters of ``render_call``/``render_response`` blocks over live payloads) is counted by
``/tokenize`` against ``16384 - 256`` and generated once; the label (``window``: ``ok`` | ``over`` | ``failed``) is
report-only.

What it runs (*Regime*): ``order(ORDER_SEED)``, 300 ``(visual_tokens, reground, reground_visual_tokens, task,
repeat)`` turns (3 arms x 5 tasks x ``REPEATS``) shuffled jointly, concurrency 1, the engine unseeded. Each turn is
T-0131's turn (one fresh start-page server, one fresh headless ``BrowserToolExecutor`` behind T-0131's box-reading
watch, one fresh ``Session`` with one ``UserText``) with ``arm_profile``'s profile (``tools = live``, headless, the
three grounding values explicit in every arm so the live defaults never decide one), a ``VllmClient(visual_tokens=
1120)`` behind ``LoopMeter``, and ``Session(tool_guidance=LIVE_TOOL_GUIDANCE, visual_tokens=1120, reground=...,
reground_llm=..., reground_visual_tokens=...)``. As ``cli`` does (T-0132 B2): equal budgets pass the main client
itself, and the 280-crop arm a second ``VllmClient(visual_tokens=280)`` with the same detokeniser, behind its own
``LoopMeter``. The run stops after ``ERRORED_STOP`` consecutive errored turns, or when ``RUN_BUDGET_S`` has passed
since ``PROCESS_START_S`` (C6c: from process start, P1-P3 included; inside the locked script's 90 min, so the report
is still written); the rest are written as not run. ``report.json`` and ``report.md`` are written once before P4 and
rewritten after it (C6c).

Output (``--out-dir``, default ``logs/t0132-crop-budget-ab-<sha7>/``): ``probes.json`` (P1-P3, and P4 after the
arms), ``trials.jsonl``, ``steps.jsonl`` and ``regrounds.jsonl`` (T-0131's fields plus ``reground_visual_tokens``; a
reground line also ``images``, ``budget``, ``first_ms`` and ``tool_ms`` from the session's ``tool.reground`` record
(B5), ``body_budget`` and ``body_images`` from the body the meter scanned (C6e), and the pass-2 prefix-cache delta),
``clicks.jsonl``, ``turns.jsonl`` and ``sessions/*.jsonl``, ``manifest.json`` (its ``regime`` read from
``services/versions.lock``, ``services/vllm.sh`` and the installed Playwright, C6d), ``report.json`` and
``report.md``. Exit codes: 0 the run was measured; 1 no engine, or the run stopped early; 2 refused; 3 a probe failed
(void). The label (C4, H-T132-1 approved as proposed; C6b's order, H-T132-4/5): "inconclusive" when P1 failed;
"void" when P2 or P3 failed or on any configuration finding (C6a, reason ``configuration: <finding>``);
"incomplete" when the run stopped early; "inconclusive" when any arm is ineligible (the reason names the arm, the
failing clause and the first failing trial); otherwise (b) then (a): "reground on, crop 280", "reground on, crop
1120" or "reground off".

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native ``bbfc53ccdd80``, live, repair on, loop 8, the live guidance on; headless Chromium at 1280x800,
DSF 1, loopback start pages only; typed turns, fake STT/TTS/audio, no headset or speakers; 3 arms x 5 tasks x n = 20 =
300 turns, order seed 0, concurrency 1, the engine unseeded; STT stopped. History, not like-for-like: T-0131's run at
c70baf0.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_astra.config import Profile
    from local_astra.contracts.llm import Prompt
    from local_astra.toolcall.loop import LoopStep

#: C6c: the run budget is measured from process start (P1-P3 included): ``time.monotonic()`` at import
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0131's harness, read-only (C1: imported, never copied); it imports T-0111g's
T0131_PATH: Path = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
#: the fixed ``sys.modules`` name T-0131 is loaded under (its dataclasses look themselves up there)
T0131_MODULE_NAME = "_t0131_grounding_ab_for_t0132"


def _load_t0131() -> ModuleType:
    """T-0131's harness, imported read-only by path under ``T0131_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0131_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0131_MODULE_NAME, T0131_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0131_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0131_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0131_MODULE_NAME, None)
        raise
    return module


T: ModuleType = _load_t0131()
#: T-0111g's harness, as T-0131 loaded it (the same module object)
G: ModuleType = T.G

# --- imported from T-0111g through T-0131, unchanged (the same objects) ----------------------------------------
TASKS = T.TASKS
score = T.score
LoopMeter = T.LoopMeter
sentinel_hits = T.sentinel_hits
url_leaks = T.url_leaks
log_complete = T.log_complete
non_loopback_sockets = T.non_loopback_sockets
turn_records = T.turn_records
TASK_TABLE_SHA256 = T.TASK_TABLE_SHA256
TOOLS_SHA12 = T.TOOLS_SHA12
# --- imported from T-0131, unchanged ------------------------------------------------------------------------
INTENDED_TARGETS = T.INTENDED_TARGETS
BOXES_JS = T.BOXES_JS
click_score = T.click_score
false_claim = T.false_claim
reground_voiced_findings = T.reground_voiced_findings
probe_p1 = T.probe_p1
run_probes = T.run_probes
config_evidence = T.config_evidence
BROWSER = T.BROWSER
SKIPPED = T.SKIPPED
ABORTED = T.ABORTED
CLICK_TOOLS = T.CLICK_TOOLS
REGROUND_TIMEOUT_MS = T.REGROUND_TIMEOUT_MS
#: the tree is read through this name (T-0111g's seam); ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: C1: (visual tokens, ReGround, pass-2 budget), the task file's table
ARMS: tuple[tuple[int, bool, int | None], ...] = (
    (1120, False, None),
    (1120, True, 1120),
    (1120, True, 280),
)
#: ``arm_table_sha()`` of the table above (the pin of tests/unit/test_t0132_ab_harness.py, 2026-10-22)
ARM_TABLE_SHA256 = "3943ca94ce9f9de83aaa285b37e8ce511cf61f69a93ac58d0cf1b28cbc978e4f"
#: C1: per (arm, task); 3 arms x 5 tasks x 20 = 300 turns
REPEATS = 20
#: *Regime*: the 300 turns shuffled jointly with seed 0
ORDER_SEED = 0
#: the main budget of every arm (C3's configuration evidence)
MAIN_BUDGET = 1120
#: C6c: the wall-time budget, measured from ``PROCESS_START_S`` (P1-P3 included), inside the locked script's
#: ``timeout -k 60s 90m``, so P4 and the report still run (estimate 35-45 min: T-0131 ran 150 turns in about 18 min)
RUN_BUDGET_S = 78 * 60
#: C4 (H-T132-1, approved as proposed 2026-10-22)
CROP_SUCCESS_MARGIN = 5
CROP_HIT_MARGIN = 0.03
CROP_P50_FACTOR = 0.8
ON_SUCCESS_GAIN = 5
ON_HIT_MARGIN = 0.02
#: O-5: boundaries compare with this tolerance, so an exact margin passes
EPS = 1e-9
#: C4's labels
ON_280 = "reground on, crop 280"
ON_1120 = "reground on, crop 1120"
OFF = "reground off"
INCONCLUSIVE = T.INCONCLUSIVE
VOID = T.VOID
INCOMPLETE = T.INCOMPLETE
#: C6b (H-T132-5): the eligibility clauses, in the order they are checked
CLAUSES = ("acceptance", "safety", "engine")
#: C6d: the boot regime's sources. ``services/vllm.sh`` is the executable form of the engine flags (read and hashed
#: at run time); its reviewed ref is the one the locked script refuses to run past; the model revision is
#: ``services/versions.lock``'s; Chromium and Playwright are read off the installed Playwright package
VLLM_SH = REPO / "services" / "vllm.sh"
VLLM_SH_REF = "806da12"
VERSIONS_LOCK = REPO / "services" / "versions.lock"
#: C6d: enforced by the locked script, which refuses while 8080/8081 are bound
STT_DOWN = "STT down (ports 8080/8081 free)"
#: C3: the 95 % two-sided normal quantile of the Wilson and Newcombe intervals
Z95 = NormalDist().inv_cdf(0.975)
#: C2 P4: the serving window (``--max-model-len``) less one canvas (CL 256), and the prompt's shape
WINDOW_TOKENS = 16_384
WINDOW_CANVAS = 256
WINDOW_LIMIT = WINDOW_TOKENS - WINDOW_CANVAS
P4_FRAME = (1280, 800)
P4_IMAGES = 4
P4_REQUEST_ID = "t0132-probe-p4"
#: the P4 generate's bound (one request; the engine stays up or the label says so)
P4_GENERATE_S = 180.0
#: the gate files whose sha256 pins a run to its commit (T-0131's, plus the files T-0132 changed)
GATE_FILES = (*T.GATE_FILES, "local_astra/config.py")
#: the runs this A/B is set beside (history, not like-for-like)
HISTORY = (
    "T-0131 D grounding A/B at c70baf0 (n = 25 per arm, order seed 0): 1120/off 15/25, hit rate 1.00, step done "
    "p95 1901 ms; 1120/on 18/25, hit rate 1.00, step done p95 1756 ms, reground p50 762 ms",
    "T-0131 D5's gate 'step done p95 <= 1.5 s' was mis-calibrated (its 280/off control measured 1523 ms) and is "
    "not reused here (human ruling 2026-10-22)",
)
#: C4, stated in the report
POWER = (
    "At n = 100 and p about 0.7 one arm's success has a standard error of about 4.6 pp and a difference about "
    "6.5 pp: the 5-point margins resolve true gaps of about 13 pp with roughly 0.9 probability, and pass a zero gap "
    "about 22 % of the time. No absolute latency bar: latency enters only (b)'s relative 'faster' clause."
)


def _say(msg: str) -> None:
    print(f"t0132_crop_budget_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The arm table and the order (C1)
# =====================================================================================================


def arm_label(visual_tokens: int, reground: bool, reground_visual_tokens: int | None) -> str:
    """``"1120/off"``, ``"1120/on/1120"``, ``"1120/on/280"``: an arm's report key."""
    if not reground:
        return f"{int(visual_tokens)}/off"
    return f"{int(visual_tokens)}/on/{int(reground_visual_tokens or 0)}"


def _arm_of(record: Mapping[str, Any]) -> str:
    return arm_label(
        int(record["visual_tokens"]), bool(record["reground"]), record.get("reground_visual_tokens")
    )


def arm_table_sha(arms: Sequence[tuple[int, bool, int | None]] | None = None) -> str:
    """sha256 of the canonical JSON of ``arms`` (``None``: the module's ``ARMS``, read at call time)."""
    rows = [
        {"visual_tokens": b, "reground": r, "reground_visual_tokens": c}
        for b, r, c in (ARMS if arms is None else arms)
    ]
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def order(seed: int) -> list[tuple[int, bool, int | None, str, int]]:
    """Every ``(visual_tokens, reground, reground_visual_tokens, task_id, repeat)`` of ``ARMS`` x ``TASKS`` x
    ``REPEATS``, arm-major, then shuffled once by ``random.Random(seed)`` -- jointly, so the arms are interleaved."""
    turns = [
        (int(b), bool(r), None if c is None else int(c), t.task_id, k)
        for b, r, c in ARMS
        for t in TASKS
        for k in range(REPEATS)
    ]
    random.Random(seed).shuffle(turns)
    return turns


def arm_profile(
    profile: Profile, visual_tokens: int, reground: bool, reground_visual_tokens: int | None
) -> Profile:
    """A validated copy of ``profile`` with ``tools = live``, ``browser = headless`` and the three grounding values
    set explicitly (so the live defaults never decide an arm). ``profile`` is not changed."""
    data = profile.model_dump()
    data.update(
        tools="live",
        browser=BROWSER,
        visual_tokens=int(visual_tokens),
        reground=bool(reground),
        reground_visual_tokens=None
        if reground_visual_tokens is None
        else int(reground_visual_tokens),
    )
    return type(profile).model_validate(data)


# =====================================================================================================
# Intervals and the configuration-evidence rule (C3)
# =====================================================================================================


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score 95 % interval of ``k`` successes out of ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    z2 = Z95 * Z95
    d = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score 95 % interval (his method 10) for ``k1/n1 - k2/n2``, from the two Wilson
    intervals."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return (
        d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def _body_budget(body: Mapping[str, Any] | None) -> int | None:
    kwargs = (body or {}).get("mm_processor_kwargs")
    value = kwargs.get("max_soft_tokens") if isinstance(kwargs, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _body_images(body: Mapping[str, Any] | None) -> int | None:
    """C6e: the image parts of a request body (``{"type": "image_url"}`` in any message's content); ``None`` when
    there is no body."""
    if body is None:
        return None
    count = 0
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list):
            count += sum(
                1 for p in content if isinstance(p, Mapping) and p.get("type") == "image_url"
            )
    return count


def pass2_budget_ok(body: Mapping[str, Any], images: int, *, main: int, crop: int) -> bool:
    """C3 (T-0132 B3): a pass-2 body carries the crop budget when its prompt holds exactly one image, and the main
    budget otherwise; a body with no budget is wrong."""
    sent = _body_budget(body)
    return sent is not None and sent == (crop if int(images) == 1 else main)


# =====================================================================================================
# The decision (C4, H-T132-1; C6a/C6b, H-T132-4/5)
# =====================================================================================================


def _failing_clause(i: Mapping[str, Any]) -> str | None:
    """C4 eligibility (acceptance 100 %, every safety count 0, the engine stayed up): the first clause that fails,
    checked in ``CLAUSES`` order, or ``None`` when the arm is eligible."""
    if not bool(i["accepted"]):
        return "acceptance"
    if int(i["safety"]) != 0:
        return "safety"
    if not bool(i["engine_up"]):
        return "engine"
    return None


def _at_least(a: Any, b: float) -> bool:
    return a is not None and float(a) >= b - EPS


def _at_most(a: Any, b: float) -> bool:
    return a is not None and float(a) <= b + EPS


def _crop_280(inputs: Mapping[str, Mapping[str, Any]]) -> bool:
    """(b), every arm already eligible (C6b runs first): the crop default becomes 280 only if every clause holds
    (success within 5 of 100, hit rate within 0.03, pass-2 done p50 at most 0.8x and p95 no slower)."""
    lo = inputs[arm_label(*ARMS[2])]
    hi = inputs[arm_label(*ARMS[1])]
    if hi["hit_rate"] is None or hi["pass2_done_p50_ms"] is None or hi["pass2_done_p95_ms"] is None:
        return False
    return (
        int(lo["success"]) >= int(hi["success"]) - CROP_SUCCESS_MARGIN
        and _at_least(lo["hit_rate"], float(hi["hit_rate"]) - CROP_HIT_MARGIN)
        and _at_most(lo["pass2_done_p50_ms"], CROP_P50_FACTOR * float(hi["pass2_done_p50_ms"]))
        and _at_most(lo["pass2_done_p95_ms"], float(hi["pass2_done_p95_ms"]))
    )


def decide_with_reason(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> tuple[str, str | None]:
    """C4's label and why, in C6b's order: "inconclusive" (P1 failed); "void" (P2/P3 failed, or any configuration
    finding, C6a / H-T132-4: reason ``configuration: <first finding>``); "incomplete" (stopped early);
    "inconclusive" when any arm is ineligible (C6b / H-T132-5: the first ineligible arm in ``ARMS`` order, its first
    failing clause in ``CLAUSES`` order and that clause's first failing trial); then (b) picks the on arm and (a)
    keeps ReGround on only if that arm gains at least 5 of 100 over 1120/off and loses at most 0.02 of hit rate.
    ``inputs`` maps each arm label to ``{"accepted", "safety", "engine_up", "success" (of 100), "hit_rate",
    "pass2_done_p50_ms", "pass2_done_p95_ms", "first_failing"}``."""
    if not p1_passed:
        return INCONCLUSIVE, "P1 failed: the budget is not shown honoured"
    if void:
        return VOID, "P2 or P3 failed: the run measured nothing"
    if configuration_findings:
        return VOID, f"configuration: {configuration_findings[0]}"
    if stopped is not None:
        return INCOMPLETE, f"stopped early: {stopped}"
    for b, r, c in ARMS:
        label = arm_label(b, r, c)
        clause = _failing_clause(inputs[label])
        if clause is not None:
            trial = (inputs[label].get("first_failing") or {}).get(clause)
            return INCONCLUSIVE, f"{label} ineligible: {clause} failed first at {trial}"
    off = inputs[arm_label(*ARMS[0])]
    crop280 = _crop_280(inputs)
    chosen = arm_label(*ARMS[2]) if crop280 else arm_label(*ARMS[1])
    on = inputs[chosen]
    why = f"(b) chose {chosen}"
    if int(on["success"]) - int(off["success"]) < ON_SUCCESS_GAIN:
        return OFF, f"{why}; (a) its success gain over 1120/off is under {ON_SUCCESS_GAIN} of 100"
    if off["hit_rate"] is None or not _at_least(
        on["hit_rate"], float(off["hit_rate"]) - ON_HIT_MARGIN
    ):
        return OFF, f"{why}; (a) its hit rate is more than {ON_HIT_MARGIN} below 1120/off's"
    return (ON_280 if crop280 else ON_1120), f"{why}; (a) holds"


def decide(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> str:
    """``decide_with_reason``'s label."""
    return decide_with_reason(
        inputs,
        p1_passed=p1_passed,
        void=void,
        stopped=stopped,
        configuration_findings=configuration_findings,
    )[0]


# =====================================================================================================
# The report (pure; report.json)
# =====================================================================================================


def _dist_counts(values: Sequence[Any]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if isinstance(r.get(key), int | float)]


def _pass2_timing(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C3, report-only: pass 2's latency split (the meter's first commit and done, the session's first and tool
    commit, the session's ``ms``) and its prefix-cache tokens. ``pass2_timeout`` requests have no ``done``."""
    return {
        "first_commit_ms": G._dist(_numbers(requests, "first_commit_ms")),
        "tool_ms": G._dist(_numbers(requests, "tool_ms")),
        "done_ms": G._dist(_numbers(requests, "done_ms")),
        "session_first_ms": G._dist(_numbers(requests, "first_ms")),
        "session_ms": G._dist(_numbers(requests, "ms")),
        "prefix_hit_tokens": G._dist(_numbers(requests, "prefix_hit_tokens")),
        "prefix_query_tokens": G._dist(_numbers(requests, "prefix_query_tokens")),
    }


def _configuration(
    trials: Sequence[Mapping[str, Any]], regrounds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """C3's configuration evidence per arm (report-only; C6a's findings are ``configuration_findings``): trials that
    forwarded a body (non-empty ``scanned_requests``), those whose non-reground bodies carried 1120 only, those whose
    every body carried the guidance, and the pass-2 requests whose scanned body is not the arm's crop at one image
    (``body_images``, C6e), or 1120 otherwise (``pass2_budget_ok``)."""
    ran = [t for t in trials if t.get("scanned_requests")]
    findings: list[str] = []
    for g in regrounds:
        if g.get("outcome") == SKIPPED or not g.get("reground"):
            continue
        crop = int(g.get("reground_visual_tokens") or MAIN_BUDGET)
        body = {"mm_processor_kwargs": {"max_soft_tokens": g.get("body_budget")}}
        if not pass2_budget_ok(body, int(g.get("body_images") or 0), main=MAIN_BUDGET, crop=crop):
            findings.append(f"{g.get('task')}#{g.get('repeat')} step {g.get('step')}")
    return {
        "trials_with_bodies": len(ran),
        "main_budget_only": sum(1 for t in ran if t.get("budgets_sent") == [MAIN_BUDGET]),
        "guidance_sent": sum(1 for t in ran if t.get("guidance_sent") is True),
        "pass2_budget_findings": findings,
    }


def _trial_id(record: Mapping[str, Any]) -> str:
    """``"<task>#<repeat>"``: how C6a's findings and C6b's reasons name a trial."""
    return f"{record['task']}#{int(record['repeat'])}"


def _turn_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (_arm_of(record), str(record["task"]), int(record["repeat"]))


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> list[str]:
    """C6a (H-T132-4), one string per finding, any of which voids the run. Every trial that forwarded a body
    (non-empty ``scanned_requests``) is judged, so missing evidence is a finding: its non-reground bodies carried
    1120 only (``budgets_sent == [1120]``) and every body carried the guidance (``guidance_sent is True``). Every
    pass-2 request (not ``skipped_no_point``) has ``images == body_images`` and ``budget == body_budget`` (C6e), and
    its scanned body carries the arm's crop at one image and 1120 otherwise. The off arm has no reground line, and
    every executed coordinate click in an on arm has one."""
    out: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        where = f"{_arm_of(t)} {_trial_id(t)}"
        if t.get("budgets_sent") != [MAIN_BUDGET]:
            out.append(f"{where}: budgets_sent {t.get('budgets_sent')!r}, not [{MAIN_BUDGET}]")
        if t.get("guidance_sent") is not True:
            out.append(f"{where}: guidance_sent {t.get('guidance_sent')!r}")
    for g in regrounds:
        where = f"{_arm_of(g)} {_trial_id(g)} step {g.get('step')}"
        if not g.get("reground"):
            out.append(f"{where}: a reground line in the ReGround-off arm")
            continue
        if g.get("outcome") == SKIPPED:
            continue
        crop = int(g.get("reground_visual_tokens") or MAIN_BUDGET)
        if g.get("images") != g.get("body_images"):
            out.append(f"{where}: images {g.get('images')} != body_images {g.get('body_images')}")
        if g.get("budget") != g.get("body_budget"):
            out.append(f"{where}: budget {g.get('budget')} != body_budget {g.get('body_budget')}")
        body = {"mm_processor_kwargs": {"max_soft_tokens": g.get("body_budget")}}
        if not pass2_budget_ok(body, int(g.get("body_images") or 0), main=MAIN_BUDGET, crop=crop):
            out.append(
                f"{where}: body_budget {g.get('body_budget')} at {g.get('body_images')} image(s), crop {crop}"
            )
    lines = {(*_turn_key(g), int(g["step"])) for g in regrounds if g.get("step") is not None}
    for s in steps:
        if not s.get("reground") or s.get("status") != "executed":
            continue
        if s.get("executed_name") not in CLICK_TOOLS:
            continue
        if (*_turn_key(s), int(s["step"])) not in lines:
            out.append(
                f"{_arm_of(s)} {_trial_id(s)} step {s['step']}: an executed click with no reground line"
            )
    return out


def _eligibility(section: Mapping[str, Any]) -> dict[str, Any]:
    """C4's eligibility inputs of one derived arm section: acceptance, the safety sum, the engine flag."""
    acc = section["acceptance"]
    return {
        "accepted": bool(acc["all_200"]) and acc["turns_ok"] == acc["turns"],
        "safety": sum(int(v) for v in section["safety"].values()),
        "engine_up": bool(section["engine_up"]),
    }


def _trial_fails(
    clause: str,
    trial: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> bool:
    """C6b, per trial: the arm-level clause applied to this one trial's own records -- T-0131's ``_arm_section``
    over ``[trial]``, its steps and its reground lines, then the same eligibility inputs the arm is judged on -- so
    a trial fails a clause exactly when it alone would make an arm fail it (a non-200 retry, ``log_complete``
    False, no step line, a raised step only when executed, ...), and an ineligible arm always has a first failing
    trial. The arm-level section reads no turn lines, so neither does this."""
    section = T._arm_section(
        int(trial["visual_tokens"]), bool(trial["reground"]), [trial], steps, regrounds
    )
    i = _eligibility(section)
    if clause == "acceptance":
        return not i["accepted"]
    if clause == "safety":
        return i["safety"] != 0
    return not i["engine_up"]


def _first_failing(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    """C6b: per clause, the first trial (in trial-line order) failing it, ``"<task>#<repeat>"``, or ``None``."""
    by_steps: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    by_rg: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_steps.setdefault(_turn_key(s), []).append(s)
    for g in regrounds:
        by_rg.setdefault(_turn_key(g), []).append(g)
    out: dict[str, str | None] = dict.fromkeys(CLAUSES)
    for clause in CLAUSES:
        for t in trials:
            k = _turn_key(t)
            if _trial_fails(clause, t, by_steps.get(k, []), by_rg.get(k, [])):
                out[clause] = _trial_id(t)
                break
    return out


def _decision_inputs(
    arm: Mapping[str, Any], first_failing: Mapping[str, str | None]
) -> dict[str, Any]:
    """One arm's decision input. The pass-2 latency is ``reground_cost.latency_ms`` (C6f: a timeout counts as
    ``REGROUND_TIMEOUT_MS``)."""
    lat = (arm.get("reground_cost") or {}).get("latency_ms") or {}
    return {
        **_eligibility(arm),
        "success": int(arm["success"]["k"]),
        "hit_rate": arm["clicks"]["hit_rate"],
        "pass2_done_p50_ms": lat.get("p50"),
        "pass2_done_p95_ms": lat.get("p95"),
        "first_failing": dict(first_failing),
    }


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    *,
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: ``void`` (P1, P2 or P3 failed), ``stopped``, the probes, every arm's section (T-0131's
    section -- step latency from the step lines only, pass 2's done latency in ``reground_cost.latency_ms`` -- plus
    the Wilson interval of its success, the pass-2 ``images``/``budget`` distributions, the pass-2 timing split and
    the configuration evidence), the three Newcombe differences, C6a's ``configuration_findings``, the decision
    (each arm's inputs with C6b's ``first_failing``) and P4's ``window``."""
    arms: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for b, r, c in ARMS:
        label = arm_label(b, r, c)
        mine = [g for g in regrounds if _arm_of(g) == label]
        arm_trials = [t for t in trials if _arm_of(t) == label]
        arm_steps = [s for s in steps if _arm_of(s) == label]
        section: dict[str, Any] = T._arm_section(b, r, arm_trials, arm_steps, mine)
        k, n = int(section["success"]["k"]), int(section["success"]["n"])
        section["success"] = {"k": k, "n": n, "wilson": list(wilson(k, n)) if n else None}
        section["reground_visual_tokens"] = c
        requests = [g for g in mine if g.get("outcome") != SKIPPED]
        section["pass2"] = (
            {
                "images": _dist_counts([g.get("images") for g in requests]),
                "budget": _dist_counts([g.get("budget") for g in requests]),
            }
            if r
            else None
        )
        section["pass2_timing"] = _pass2_timing(requests) if r else None
        section["configuration"] = _configuration(arm_trials, mine)
        arms[label] = section
        inputs[label] = _decision_inputs(section, _first_failing(arm_trials, arm_steps, mine))
    off, on1120, on280 = (arm_label(*a) for a in ARMS)

    def diff(a: str, b: str) -> list[float] | None:
        """``None`` when either arm ran no turn (a void or stopped run)."""
        ka, na = arms[a]["success"]["k"], arms[a]["success"]["n"]
        kb, nb = arms[b]["success"]["k"], arms[b]["success"]["n"]
        return list(newcombe(ka, na, kb, nb)) if na and nb else None

    findings = configuration_findings(trials, steps, regrounds)
    p1_passed = bool(probes["p1"]["ok"])
    void = not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3"))
    label, reason = decide_with_reason(
        inputs,
        p1_passed=p1_passed,
        void=void,
        stopped=stopped,
        configuration_findings=findings,
    )
    p4 = probes.get("p4")
    window = str(p4.get("window")) if isinstance(p4, Mapping) and p4.get("window") else "failed"
    return {
        "void": void,
        "stopped": stopped,
        "probes": dict(probes),
        "arms": arms,
        "differences": {
            "on1120-off": diff(on1120, off),
            "on280-off": diff(on280, off),
            "on280-on1120": diff(on280, on1120),
        },
        "configuration_findings": findings,
        "decision": {"label": label, "reason": reason, "inputs": inputs},
        "window": window,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reason, the power statement, the probes and P4's window,
    every arm's numbers, the differences, and T-0131's run beside them (history, not like-for-like)."""
    # the title names no "decision": the Decision heading below is the first one that does (C6f's section)
    lines = ["# T-0132 C crop-budget A/B (report-only, plus H-T132-1's label rule)", ""]
    if manifest is not None:
        regime = manifest.get("regime") or {}
        lines += [
            "## Regime",
            "",
            f"- model revision: {_fmt(regime.get('model_revision'))}; services/vllm.sh sha256 "
            f"{_fmt(regime.get('vllm_sh_sha256'))} (ref {_fmt(regime.get('vllm_sh_ref'))}); max_num_seqs "
            f"{_fmt(regime.get('max_num_seqs'))}, prefix caching {_fmt(regime.get('prefix_caching'))}, KV "
            f"{_fmt(regime.get('kv_cache'))}",
            f"- Chromium {_fmt(regime.get('chromium'))}; Playwright {_fmt(regime.get('playwright'))}",
            f"- {_fmt(regime.get('stt'))}",
            "",
            "```",
            json.dumps(regime, indent=1),
            "```",
            "",
        ]
    decision = report["decision"]
    findings = report.get("configuration_findings") or []
    lines += [
        "## Decision (C4, H-T132-1 approved as proposed; C6a/C6b, H-T132-4/5)",
        "",
        f"- outcome label: **{decision['label']}**",
        f"- reason: {_fmt(decision.get('reason'))}",
        f"- void: {report['void']}"
        + (" (a probe failed: the run measured nothing)" if report["void"] else ""),
        f"- stopped early: {_fmt(report.get('stopped'))}",
    ]
    for label, arm in report["arms"].items():
        cost = arm.get("reground_cost")
        if cost is not None:
            lines.append(
                f"- {label}: reground requests {cost['requests']}, ok {cost['outcomes'].get('ok', 0)}"
            )
    lines.append(f"- configuration findings: {len(findings)}")
    lines += [f"  - {f}" for f in findings[:20]]
    if len(findings) > 20:
        lines.append(f"  - ... and {len(findings) - 20} more (report.json)")
    for label in report["arms"]:
        inputs = decision["inputs"][label]
        clause = _failing_clause(inputs)
        if clause is None:
            lines.append(f"- {label}: eligible")
        else:
            trial = (inputs.get("first_failing") or {}).get(clause)
            lines.append(f"- {label}: ineligible ({clause} failed first at {trial})")
    lines += [
        f"- pass-2 latency in the decision: reground_cost.latency_ms (a timeout counts {REGROUND_TIMEOUT_MS} ms)",
        f"- power: {POWER}",
        "",
        "## Probes (C2)",
        "",
    ]
    probes = report["probes"]
    for budget, row in sorted(probes["p1"]["budgets"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"- P1 {budget}: frame delta {_fmt(row['frame_delta'])}, crop delta {_fmt(row['crop_delta'])}, "
            f"expected {row['expected']} (+0..{T.P1_DELIMITERS}), ok {row['ok']}"
        )
    lines += [
        f"- P2 (one image at {T.P2_BUDGET}): HTTP {_fmt(probes['p2']['http_status'])}, healthy after "
        f"{probes['p2']['healthy_after']}, ok {probes['p2']['ok']}",
        f"- P3 (text only, with the kwarg): HTTP {_fmt(probes['p3']['http_status'])}, ok {probes['p3']['ok']}",
    ]
    p4 = probes.get("p4") or {}
    lines += [
        f"- P4 (worst-case window, after the arms; report-only): **{report['window']}**; /tokenize "
        f"{_fmt(p4.get('tokens'))} of {WINDOW_LIMIT} (= {WINDOW_TOKENS} - {WINDOW_CANVAS}); generate HTTP "
        f"{_fmt(p4.get('http_status'))}, healthy after {_fmt(p4.get('healthy_after'))}"
        + (f"; {p4['reason']}" if p4.get("reason") else ""),
        "",
        "## Arms (C3; each number report-only)",
        "",
        "| arm | success (Wilson 95 %) | hit rate | pass-1 hit rate | click error px (p50/p95/mean) | false "
        "claims | step done p95 ms | pass-2 done p50/p95 ms | wall s p50 | min free VRAM MiB | accepted | "
        "safety | engine up |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, arm in report["arms"].items():
        clicks, p1 = arm["clicks"], arm.get("clicks_p1") or {}
        err = clicks["error_px"]
        inputs = decision["inputs"][label]
        lines.append(
            f"| {label} | {arm['success']['k']}/{arm['success']['n']} {_interval(arm['success']['wilson'])} | "
            f"{_fmt(clicks['hit_rate'])} (n={clicks['n']}, read errors {clicks['read_errors']}) | "
            f"{_fmt(p1.get('hit_rate'))} | {T._px(err['p50'])}/{T._px(err['p95'])}/{T._px(err['mean'])} | "
            f"{arm['false_claims']} | {_fmt(arm['latency']['done_ms']['p95'])} | "
            f"{_fmt(inputs['pass2_done_p50_ms'])}/{_fmt(inputs['pass2_done_p95_ms'])} | "
            f"{_fmt(arm['wall_s']['p50'])} | {_fmt(arm['vram_min_free_mib'])} | {inputs['accepted']} | "
            f"{inputs['safety']} | {arm['engine_up']} |"
        )
    lines += ["", "## Differences in success (Newcombe 95 %, method 10)", ""]
    lines += [f"- {k}: {_interval(v)}" for k, v in report["differences"].items()]
    for label, arm in report["arms"].items():
        lines += ["", f"### {label}", ""]
        lines += [f"- {task}: {o['k']}/{o['n']}" for task, o in arm["outcomes"].items()]
        st = arm["steps"]
        lines += [
            f"- steps per turn: {T._dist_cell(st['per_turn'])}",
            "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
            f"- scrolls per S4 turn: {T._dist_cell(st['scrolls']['S4'])}; per S5 turn: "
            f"{T._dist_cell(st['scrolls']['S5'])}",
        ]
        lat = arm["latency"]
        lines += [
            f"- step first commit ms: {T._dist_cell(lat['first_commit_ms'])}; done ms: "
            f"{T._dist_cell(lat['done_ms'])} (reground requests excluded)",
            f"- prefix cache hit tokens: {T._dist_cell(arm['cache']['hit_tokens'])}; query tokens: "
            f"{T._dist_cell(arm['cache']['query_tokens'])}",
        ]
        cost = arm["reground_cost"]
        if cost is not None:
            outcomes = ", ".join(f"{k} {v}" for k, v in cost["outcomes"].items())
            timing = arm["pass2_timing"]
            lines += [
                f"- reground requests: {cost['requests']} ({outcomes})",
                f"- pass-2 done ms (a timeout counts {REGROUND_TIMEOUT_MS}): {T._dist_cell(cost['latency_ms'])}; "
                f"extra execute ms per coordinate step: {T._dist_cell(cost['extra_execute_ms'])}",
                f"- pass-2 split: first commit {T._dist_cell(timing['first_commit_ms'])}; tool commit (session) "
                f"{T._dist_cell(timing['tool_ms'])}; done {T._dist_cell(timing['done_ms'])}; session ms "
                f"{T._dist_cell(timing['session_ms'])}",
                f"- pass-2 prefix cache hit tokens: {T._dist_cell(timing['prefix_hit_tokens'])}; query "
                f"tokens: {T._dist_cell(timing['prefix_query_tokens'])}",
                f"- pass-2 images: {arm['pass2']['images']}; budget sent: {arm['pass2']['budget']}",
            ]
        cfg = arm["configuration"]
        acc = arm["acceptance"]
        lines += [
            f"- configuration: non-reground bodies at {MAIN_BUDGET} only in {cfg['main_budget_only']}/"
            f"{cfg['trials_with_bodies']} trials, guidance in {cfg['guidance_sent']}/{cfg['trials_with_bodies']}, "
            f"pass-2 budget findings {len(cfg['pass2_budget_findings'])}",
            f"- acceptance: steps at 200 {acc['ok']}/{acc['steps']}, turns {acc['turns_ok']}/{acc['turns']}, "
            f"regrounds at 200 {acc['regrounds_ok']}/{acc['regrounds']}, all 200 {acc['all_200']}",
            "- safety: " + ", ".join(f"{k} {v}" for k, v in arm["safety"].items()),
        ]
    lines += ["", "## History (not like-for-like)", ""]
    lines += [f"- {h}" for h in HISTORY]
    lines.append("")
    return "\n".join(lines)


# =====================================================================================================
# Record extraction (pure, over one turn's session records)
# =====================================================================================================


def reground_lines(
    task_id: str,
    repeat: int,
    *,
    records: Sequence[Mapping[str, Any]],
    generates: Sequence[Mapping[str, Any]],
    bodies: Mapping[str, Mapping[str, Any]],
    visual_tokens: int,
    reground: bool,
    reground_visual_tokens: int | None,
) -> list[dict[str, Any]]:
    """T-0131's reground lines (one per ``tool.reground`` record, joined to its ``LoopMeter`` row by the pass-2
    attempt id, over both clients' rows), plus ``reground_visual_tokens``, B5's ``images``, ``budget``,
    ``first_ms`` and ``tool_ms`` (``None`` for ``skipped_no_point``, O-8), ``body_budget`` and ``body_images``
    (C6e: the ``max_soft_tokens`` and the image count of the pass-2 body the meter scanned, never the session's
    record) and the pass-2 prefix-cache delta."""
    base = T.reground_lines(
        task_id,
        repeat,
        records=records,
        generates=generates,
        visual_tokens=visual_tokens,
        reground=reground,
    )
    rows = {str(g.get("request_id")): g for g in generates}
    datas = [r.get("data") or {} for r in records if r.get("kind") == "tool.reground"]
    out: list[dict[str, Any]] = []
    for line, d in zip(base, datas, strict=True):
        attempt = line.get("attempt_id")
        row = rows.get(str(attempt)) if attempt is not None else None
        body = bodies.get(f"generate:{attempt}") if attempt is not None else None
        out.append(
            {
                **line,
                "reground_visual_tokens": reground_visual_tokens,
                "images": d.get("images"),
                "budget": d.get("budget"),
                "first_ms": d.get("first_ms"),
                "tool_ms": d.get("tool_ms"),
                "body_budget": _body_budget(body),
                "body_images": _body_images(body),
                "prefix_hit_tokens": (row or {}).get("prefix_hit_tokens"),
                "prefix_query_tokens": (row or {}).get("prefix_query_tokens"),
            }
        )
    return out


def _not_run(
    task_id: str, repeat: int, reason: str, budget: int, reground: bool, crop: int | None
) -> dict[str, Any]:
    """T-0131's not-run trial with the arm's pass-2 budget; no body was forwarded, so there is no configuration
    evidence (``None``, not an empty list)."""
    trial: dict[str, Any] = T._not_run(task_id, repeat, reason, budget, reground)
    trial.update(reground_visual_tokens=crop, budgets_sent=None, guidance_sent=None)
    return trial


# =====================================================================================================
# P4: the worst-case window (C2; pure prompt construction)
# =====================================================================================================

#: P4's live payloads, cycled: (tool, arguments, the executor's ``action`` text) as the T-0131 run's steps carry them
_P4_CALLS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "type_at",
        {"press_enter": True, "text": "blue mug", "x": 458, "y": 501},
        "clicked at (586, 401) and typed 8 characters and pressed Enter",
    ),
    ("click_at", {"x": 128, "y": 390}, "clicked at (164, 312)"),
    ("scroll", {"direction": "down"}, "scrolled down"),
    ("click_at", {"x": 512, "y": 620}, "clicked at (655, 496)"),
    ("press_key", {"combo": "Enter"}, "pressed Enter"),
    ("scroll", {"direction": "up"}, "scrolled up"),
)
P4_FIRST_TEXT = "Let me find the cheapest deal and add it to the cart."
P4_FILLER = "I will look at the page again before the next action. "
P4_REQUEST = "Find the cheapest deal on the page and add it to the cart."


def _p4_step(index: int, text: str) -> LoopStep:
    from local_astra.toolcall import loop

    name, arguments, action = _P4_CALLS[index % len(_P4_CALLS)]
    payload = loop.screenshot_payload({"action": action, "status": "executed"}, index % 3 != 2)
    return loop.LoopStep(text, name, dict(arguments), payload)


def p4_steps() -> list[LoopStep]:
    """Live steps whose continuation is exactly ``MAX_CONTINUATION`` characters: whole steps while they fit, then
    the remainder as the last step's text before its call (plain prose, unchanged by the sanitiser)."""
    from local_astra.conductor.session import MAX_CONTINUATION
    from local_astra.toolcall import loop

    steps: list[LoopStep] = []
    while True:
        cand = _p4_step(len(steps), P4_FIRST_TEXT if not steps else "")
        if loop.continuation_chars([*steps, cand]) > MAX_CONTINUATION:
            break
        steps.append(cand)
    gap = MAX_CONTINUATION - loop.continuation_chars(steps)
    last = steps[-1]
    filler = (P4_FILLER * (gap // len(P4_FILLER) + 1))[:gap]
    steps[-1] = loop.LoopStep(last.text + filler, last.name, last.arguments, last.response)
    got = loop.continuation_chars(steps)
    if got != MAX_CONTINUATION:
        raise RuntimeError(f"P4's continuation is {got} characters, not {MAX_CONTINUATION}")
    return steps


def p4_continuation() -> str:
    """P4's continuation: the final assistant text ``loop.step_prompt`` builds from ``p4_steps()``."""
    from local_astra.contracts.llm import Message, Prompt, TextPart
    from local_astra.toolcall import loop

    bare = Prompt(
        messages=(Message(role="user", content=(TextPart(text=P4_REQUEST),)),), prefix_len=0
    )
    return loop.step_prompt(bare, p4_steps()).messages[-1].text()


def _p4_jpegs() -> list[bytes]:
    """Four distinct synthetic page-like 1280x800 JPEG frames (deterministic)."""
    import io

    from PIL import Image, ImageDraw

    w, h = P4_FRAME
    out: list[bytes] = []
    for n in range(P4_IMAGES):
        img = Image.new("RGB", (w, h), (246, 246 - 8 * n, 242))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w - 1, 56), fill=(32 + 30 * n, 48, 80))
        for i in range(3):
            x0 = 80 + i * 400
            y0 = 140 + 40 * n
            draw.rectangle((x0, y0, x0 + 340, y0 + 420), outline=(180, 180, 180), width=2)
            draw.rectangle(
                (x0 + 30, y0 + 30, x0 + 310, y0 + 240), fill=(60 + 60 * i, 90 + 30 * n, 200)
            )
            draw.rectangle((x0 + 110, y0 + 340, x0 + 231, y0 + 381), fill=(20, 110, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out


def p4_prompt() -> Prompt:
    """C2 P4's worst case: the live step-k prompt with ``Budget.max_images`` (4) distinct 1280x800 frames (one
    history image and three current frames, as the builder lays them out), the live system message (the native
    tools field and ``LIVE_TOOL_GUIDANCE``, thinking on) and ``p4_steps()`` as its continuation."""
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE, PromptBuilder
    from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
    from local_astra.contracts.perception import Frame, Source
    from local_astra.contracts.turn import HistoryTurn
    from local_astra.toolcall import loop, native
    from local_astra.toolcall import tools as toolset

    frames = [
        Frame(
            source=Source.SCREEN,
            ts_ms=1000 * (n + 1),
            width=P4_FRAME[0],
            height=P4_FRAME[1],
            jpeg=jpeg,
            phash=hashlib.sha256(jpeg).hexdigest()[:16],
        )
        for n, jpeg in enumerate(_p4_jpegs())
    ]
    history = [
        HistoryTurn(
            user_text="Open the shop.",
            assistant_spoken="The shop is open.",
            frame_phash=frames[0].phash,
            frame_caption="A shop page with three products.",
        )
    ]
    builder = PromptBuilder(
        DEFAULT_SYSTEM_PROMPT,
        thinking=True,
        tools=tuple(native.tools_field(toolset.load())),
        tool_guidance=LIVE_TOOL_GUIDANCE,
    )
    pixels = {frames[0].phash: frames[0].jpeg}
    prefix = builder.prefix(history, resolve=pixels.get, frames=tuple(frames[1:]))
    full = builder.full(prefix, P4_REQUEST, None, ())
    return loop.step_prompt(full, p4_steps())


# =====================================================================================================
# C6d: the boot regime, read from its sources (files only: no socket, no process)
# =====================================================================================================


def _flag(text: str, name: str) -> str | None:
    """The value after ``--<name>`` on ``services/vllm.sh``'s command line, or ``None``."""
    import re

    found = re.search(rf"--{re.escape(name)}[ =]([^\s\\]+)", text)
    return found.group(1) if found else None


def _browser_versions() -> dict[str, Any]:
    """Chromium and Playwright as installed, read without importing Playwright or launching a browser: the
    ``playwright`` distribution's version, and the headless shell's ``browserVersion`` and revision from the
    package's ``browsers.json`` (the build Playwright launches headless), with the executable's path when it is
    present (``preflight.pinned_executable``). ``None`` when unknown."""
    from importlib.metadata import PackageNotFoundError, version

    from local_astra import preflight

    try:
        playwright: str | None = version("playwright")
    except PackageNotFoundError:
        playwright = None
    chromium: str | None = None
    package = preflight._playwright_package()
    if package is not None:
        try:
            listing = json.loads((package / "driver" / "package" / "browsers.json").read_text())
            entry = next(
                b for b in listing["browsers"] if b.get("name") == "chromium-headless-shell"
            )
            exe = preflight.pinned_executable(BROWSER)
            chromium = (
                f"{entry['browserVersion']} (chromium-headless-shell r{entry['revision']}, "
                f"{exe if exe is not None else 'executable absent'})"
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            chromium = None
    return {"chromium": chromium, "playwright": playwright}


def regime_sources() -> dict[str, Any]:
    """C6d's regime keys, each read from its source: the model revision from ``services/versions.lock``
    (``[gpu] model_revision``); ``services/vllm.sh``'s sha256 and the engine flags on its command line
    (``--max-num-seqs``, ``--enable-prefix-caching``, ``--kv-cache-memory-bytes``, ``--kv-cache-dtype``) beside its
    reviewed ref (``VLLM_SH_REF``, the locked script's refusal); the Chromium and Playwright versions; STT down (the
    locked script's port refusal); and the full arm table."""
    import tomllib

    # the command line only: the comments quote the same flags in prose
    text = "\n".join(
        ln
        for ln in VLLM_SH.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    )
    lock = tomllib.loads(VERSIONS_LOCK.read_text(encoding="utf-8"))
    seqs = _flag(text, "max-num-seqs")
    kv_bytes = _flag(text, "kv-cache-memory-bytes")
    kv_dtype = _flag(text, "kv-cache-dtype")
    short = {"bfloat16": "bf16", "float16": "fp16"}
    kv = (
        None
        if kv_bytes is None or not kv_bytes.isdigit()
        else f"{int(kv_bytes) / 2**30:g} GiB ({kv_bytes} bytes) "
        f"{short.get(str(kv_dtype), kv_dtype)} (--kv-cache-dtype {kv_dtype})"
    )
    return {
        "model": lock.get("gpu", {}).get("model"),
        "model_revision": lock.get("gpu", {}).get("model_revision"),
        "model_revision_source": "services/versions.lock [gpu] model_revision",
        "vllm_container": lock.get("gpu", {}).get("vllm_container"),
        "vllm_sh_sha256": hashlib.sha256(VLLM_SH.read_bytes()).hexdigest(),
        "vllm_sh_ref": VLLM_SH_REF,
        "max_num_seqs": int(seqs) if seqs is not None and seqs.isdigit() else None,
        "prefix_caching": "--enable-prefix-caching" in text,
        "kv_cache": kv,
        **_browser_versions(),
        "stt": STT_DOWN,
        "stt_source": "scripts/locked/t0132_crop_budget_ab.sh refuses while 127.0.0.1:8080/8081 are bound",
        "arms": [
            {"visual_tokens": b, "reground": r, "reground_visual_tokens": c} for b, r, c in ARMS
        ],
    }


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


async def _probe_window(base_url: str, model: str, detokenizer: Any) -> dict[str, Any]:
    """C2 P4: ``/tokenize`` of ``p4_prompt()`` at 1120 against ``WINDOW_LIMIT``, then one generate on it and the
    engine's health. ``window``: ``failed`` when the count did not come back, ``over`` above the limit, ``ok`` when
    the generate returned 200 and the engine stayed healthy, else ``failed``."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.contracts.llm import ImagePart, LlmOptions

    prompt = p4_prompt()
    client = LoopMeter(
        VllmClient(
            VllmEndpoint(base_url=base_url),
            transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
            detokenizer=detokenizer,
            visual_tokens=MAIN_BUDGET,
        ),
        metrics=lambda: G.read_metrics(base_url),
        clock=time.monotonic,
    )
    body = client.inner.request_body(prompt, P4_REQUEST_ID)
    keys = (
        "messages",
        "tools",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "mm_processor_kwargs",
    )
    tokenize = {"model": model, **{k: body[k] for k in keys if k in body}}
    status, payload = await asyncio.to_thread(
        T._post_json, G._root(base_url) + "/tokenize", tokenize
    )
    tokens: int | None = None
    if status == 200 and isinstance(payload, dict):
        count = payload.get("count")
        if isinstance(count, int):
            tokens = count
        elif isinstance(payload.get("tokens"), list):
            tokens = len(payload["tokens"])
    http_status = 200
    error: str | None = None
    try:

        async def drain() -> None:
            async for _commit in client.generate(prompt, P4_REQUEST_ID, LlmOptions()):
                pass

        await asyncio.wait_for(drain(), timeout=P4_GENERATE_S)
    except Exception as exc:
        found = G._http_status(exc)
        http_status = found if found is not None else 0
        error = repr(exc)
        _say(f"probe P4: {exc!r}")
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
    healthy = await asyncio.to_thread(T._healthy, base_url)
    if tokens is None:
        window = "failed"
    elif tokens > WINDOW_LIMIT:
        window = "over"
    elif http_status == 200 and healthy:
        window = "ok"
    else:
        window = "failed"
    images = sum(1 for m in prompt.messages for p in m.content if isinstance(p, ImagePart))
    return {
        "window": window,
        "tokens": tokens,
        "limit": WINDOW_LIMIT,
        "tokenize_status": status,
        "http_status": http_status,
        "healthy_after": healthy,
        "error": error,
        "images": images,
        "continuation_chars": len(prompt.messages[-1].text()),
        "done_ms": (client.generates[-1] if client.generates else {}).get("done_ms"),
    }


# =====================================================================================================
# The turn runner
# =====================================================================================================


@dataclass
class _Run:
    base_url: str
    out: Path
    profiles: dict[tuple[int, bool, int | None], Any]
    sentinel: bytes
    manifest: dict[str, Any]
    vram: Any
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


async def _one_turn(
    run: _Run, budget: int, reground: bool, crop: int | None, task_id: str, repeat: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """T-0131's turn with the arm's profile, budget, ReGround, pass-2 client and the live guidance; the records
    read after the session closed. Returns the trial, its step lines, its reground lines and its raw box reads."""
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
    profile = run.profiles[(budget, reground, crop)]
    task = G._task(task_id)
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (T-0111g Implementation notes, Event loop)
    pages = StartPageServer()
    pages.start()
    browser: Any = None
    watch: Any = None
    session: Any = None
    meter: Any = None
    crop_meter: Any = None
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
        watch = T._GroundingWatch(browser, INTENDED_TARGETS[task_id])
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=watch)
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        endpoint = VllmEndpoint(base_url=run.base_url)
        meter = LoopMeter(
            VllmClient(
                endpoint,
                transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
                detokenizer=detokenizer,
                visual_tokens=budget,  # explicit in every arm
            ),
            metrics=lambda: G.read_metrics(run.base_url),
            clock=time.monotonic,
        )
        second: Any = None
        if reground and crop is not None:
            if crop == budget:
                second = meter  # B2: equal budgets pass the main client itself, as cli does
            else:
                # B2: a second client with the pass-2 budget and the same detokeniser, metered like the first
                crop_meter = LoopMeter(
                    VllmClient(
                        endpoint,
                        transport=httpx.AsyncHTTPTransport(),
                        detokenizer=detokenizer,
                        visual_tokens=crop,
                    ),
                    metrics=lambda: G.read_metrics(run.base_url),
                    clock=time.monotonic,
                )
                second = crop_meter
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
            tool_guidance=LIVE_TOOL_GUIDANCE,
            visual_tokens=budget,
            reground=reground,
            reground_llm=second,
            reground_visual_tokens=crop if second is not None else None,
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
        for m in (meter, crop_meter):
            if m is not None:
                with contextlib.suppress(Exception):
                    await m.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    if session is not None and session.failures and exception is None:
        exception = "session failures: " + "; ".join(repr(f) for f in session.failures)
    engine_up = await asyncio.to_thread(T._healthy, run.base_url)
    vram_min = run.vram.min_between(window_start, time.monotonic())

    records: list[dict[str, Any]] = []
    record_lines: list[tuple[str, str]] = []
    if log is not None and Path(log.path).is_file():
        text = Path(log.path).read_text(encoding="utf-8")
        records = sorted(G._read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
        record_lines += [
            (f"sessions/{Path(log.path).name}:{n}", ln)
            for n, ln in enumerate(text.splitlines(), start=1)
        ]

    # --- after close, outside every timed span: the bodies of every request both meters forwarded -------------
    # (the prefill, every generate attempt, every pass 2 -- through either client -- and every repair)
    meters = [m for m in (meter, crop_meter) if m is not None]
    raw: dict[str, dict[str, Any]] = {}
    for m in meters:
        for label, prompt in m.requests.items():
            raw[label] = m.inner.request_body(prompt, label.split(":", 1)[1])
    generates = [g for m in meters for g in m.generates if not g.get("repair")]
    bodies = {label: json.dumps(body) for label, body in raw.items()}
    sentinel_requests = sentinel_hits(bodies, run.sentinel)
    scanned_requests = sorted(bodies)
    budgets_sent: list[int] | None = None
    guidance_sent: bool | None = None
    if raw:
        # C3: the non-reground bodies carry the main budget only; every body carries the guidance
        pass2_ids = set(T._reground_attempts(records))
        steps_only = {k: v for k, v in raw.items() if k.split(":", 1)[1] not in pass2_ids}
        budgets_sent, _ = config_evidence(steps_only, LIVE_TOOL_GUIDANCE)
        _, guidance_sent = config_evidence(raw, LIVE_TOOL_GUIDANCE)

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
    steps = [
        {**s, "reground_visual_tokens": crop}
        for s in T.step_lines(
            steps,
            visual_tokens=budget,
            reground=reground,
            clicks=watch.clicks if watch is not None else {},
            executed=watch.executed if watch is not None else {},
            records=records,
        )
    ]
    regrounds = reground_lines(
        task_id,
        repeat,
        records=records,
        generates=[g for m in meters for g in m.generates],
        bodies=raw,
        visual_tokens=budget,
        reground=reground,
        reground_visual_tokens=crop,
    )
    answers = T._answers(records)
    pass2 = [answers.get(a, "") for a in T._reground_attempts(records)]
    findings = reground_voiced_findings(
        pass2,
        trial.get("voiced") or [],
        trial.get("history"),
        [str(s.get("step_text") or "") for s in steps],
    )
    trial.update(
        visual_tokens=int(budget),
        reground=bool(reground),
        reground_visual_tokens=crop,
        wall_s=wall_s,
        vram_min_free_mib=vram_min,
        engine_up_after=engine_up,
        reground_voiced=[f"{task_id}/r{repeat}: {f}" for f in findings],
        budgets_sent=budgets_sent,
        guidance_sent=guidance_sent,
    )
    trial["session_log"] = Path(log.path).name if log is not None else None
    trial["deferred_prefill_aborts"] = sum(len(m.deferred_aborts) for m in meters)
    reads = [
        {
            "visual_tokens": budget,
            "reground": reground,
            "reground_visual_tokens": crop,
            "task": task_id,
            "repeat": repeat,
            **c,
        }
        for c in (watch.clicks.values() if watch is not None else [])
    ]
    return trial, steps, regrounds, reads


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``ERRORED_STOP`` consecutive errored turns, or past
    ``RUN_BUDGET_S`` since ``PROCESS_START_S`` (C6c: P1-P3 included), nothing more is sent and the rest are written as
    not run, so the files still hold 300 trials."""
    errored = 0
    for index, (budget, reground, crop, task_id, repeat) in enumerate(order(ORDER_SEED)):
        started = time.monotonic()
        if run.stopped is None and started - PROCESS_START_S > RUN_BUDGET_S:
            run.stopped = f"the run budget of {RUN_BUDGET_S} s is spent (at turn {index})"
            run.manifest["stopped"] = run.stopped
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.stopped}")
        steps: list[dict[str, Any]] = []
        regrounds: list[dict[str, Any]] = []
        reads: list[dict[str, Any]] = []
        if run.stopped is not None:
            trial = _not_run(task_id, repeat, f"not run: {run.stopped}", budget, reground, crop)
        else:
            try:
                trial, steps, regrounds, reads = await _one_turn(
                    run, budget, reground, crop, task_id, repeat
                )
            except Exception as exc:
                trial = _not_run(task_id, repeat, f"harness: {exc!r}", budget, reground, crop)
            errored = errored + 1 if T._errored(trial, steps, regrounds) else 0
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
            f"{index:03d} {arm_label(budget, reground, crop)} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"reground={[(r.get('outcome'), r.get('images'), r.get('budget')) for r in regrounds]} "
            f"success={trial.get('success')} timed_out={trial.get('timed_out')} err={trial.get('exception')} "
            f"wall={run.turn_s[-1]}s"
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
    out: Path = args.out_dir or REPO / "logs" / f"t0132-crop-budget-ab-{sha[:7]}"
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
    # 5. the arm table's pin (C1)
    arms_sha = arm_table_sha()
    if arms_sha != ARM_TABLE_SHA256:
        _say(f"refusing: the arm table's sha is {arms_sha}, not the pinned {ARM_TABLE_SHA256}")
        return EXIT_REFUSED
    # 6. the tool kit (T-0115's three refusals, repair off included)
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    try:
        profiles = {(b, r, c): arm_profile(lean, b, r, c) for b, r, c in ARMS}
    except ValueError as exc:
        _say(f"refusing: an arm's profile is invalid: {exc}")
        return EXIT_REFUSED
    try:
        kit, detokenizer = cli.load_tool_kit(lean)
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
    # 8. the pass-2 bound the latency rule counts a timeout as (T-0131's ruling at 4337125)
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

    vram = T._VramSampler()
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
            detokenizer,
            vram,
            LIVE_TOOL_GUIDANCE,
        )
    finally:
        vram.stop()


def _measure(
    base_url: str,
    out: Path,
    sha: str,
    model: str,
    lean: Any,
    profiles: dict[tuple[int, bool, int | None], Any],
    kit: Any,
    detokenizer: Any,
    vram: Any,
    guidance: str,
) -> int:
    """Past every refusal and the engine check: the manifest, P1-P3, the turns, P4, the report."""
    from local_astra.conductor.session import MAX_CONTINUATION

    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    manifest: dict[str, Any] = {
        "regime": {
            **regime_sources(),
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
            "repeats": REPEATS,
            "reground_timeout_ms": REGROUND_TIMEOUT_MS,
            "live_tool_guidance_sha256": hashlib.sha256(guidance.encode("utf-8")).hexdigest(),
            "browser": f"{BROWSER}, one per turn, fresh context, loopback start pages only",
            "viewport": list(lean.browser_viewport),
            "turns": len(order(ORDER_SEED)),
            "concurrency": 1,
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "turn_timeout_s": G.TURN_TIMEOUT_S,
            "run_budget_s": RUN_BUDGET_S,
            "vram_period_ms": T.VRAM_PERIOD_MS,
            "p4": {
                "images": P4_IMAGES,
                "frame": list(P4_FRAME),
                "visual_tokens": MAIN_BUDGET,
                "continuation_chars": MAX_CONTINUATION,
                "window_limit": WINDOW_LIMIT,
            },
            "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
            "no audio device, no headset or speakers",
            "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
            "history": list(HISTORY),
        },
        "arm_table_sha256": arm_table_sha(),
        "task_table_sha256": G.task_table_sha(TASKS),
        "tasks": [
            {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": REPEATS}
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

    # --- C2: P1-P3, before any arm turn ---------------------------------------------------------------
    probe_start = time.monotonic()
    probes = run_probes(base_url, model)
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    if not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3")):
        probes["p4"] = {"window": "failed", "reason": "not run: a pre-flight probe failed"}
        G._write_json(out / "probes.json", probes)
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest)
        _say(
            f"a probe failed: the run is void ({json.dumps(probes)}); label {report['decision']['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "probes.json", probes)
    G._write_json(out / "manifest.json", manifest)

    run = _Run(
        base_url=base_url,
        out=out,
        profiles=profiles,
        sentinel=sentinel,
        manifest=manifest,
        vram=vram,
    )
    asyncio.run(_run_all(run))

    # --- C6c: the report once before P4 (a timeout during P4 still leaves one), rewritten after it --------
    probes["p4"] = {
        "window": "failed",
        "reason": "not run yet: P4 runs after this report is written",
    }
    G._write_json(out / "probes.json", probes)
    report = _write_report(out, probes, manifest, run.stopped)
    _say(f"decision before P4={report['decision']['label']} ({report['decision']['reason']})")

    # --- C2: P4 after the arms, so it cannot void them -----------------------------------------------
    p4_start = time.monotonic()
    try:
        probes["p4"] = asyncio.run(_probe_window(base_url, model, detokenizer))
    except Exception as exc:
        probes["p4"] = {"window": "failed", "reason": f"harness: {exc!r}"}
    probes["p4"]["vram_min_free_mib"] = vram.min_between(p4_start, time.monotonic())
    G._write_json(out / "probes.json", probes)
    report = _write_report(out, probes, manifest, run.stopped)
    _say(
        f"decision={report['decision']['label']} ({report['decision']['reason']}) window={report['window']}"
    )
    _say(f"written to {out}")
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
