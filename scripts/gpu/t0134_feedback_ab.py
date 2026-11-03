"""T-0134 B -- the locked 3-arm live-feedback A/B of ``tools = live`` at 1120: 300 typed turns through a real
``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0134_feedback_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the arm table (``ARMS``, ``arm_table_sha``,
``order``), the scoring (``score_v2``, ``over_click``), the mechanism checks (``no_see``, ``coord_mismatch``,
``silent_call_in_thought``), the configuration findings (``configuration_findings``), P5's rule (``p5_ok``), the
decision (``decide``, ``decide_with_reason``), the report (``build_report``, ``render_markdown``) and the record
extraction (``feedback_step_lines``, ``thought_record``) are pure; everything engine- or browser-facing is imported
inside ``main``, the probes and the turn runner.

B1: T-0132's harness (``scripts/gpu/t0132_crop_budget_ab.py``) is **imported, never copied**, and through it
T-0131's and T-0111g's: ``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
``non_loopback_sockets``, ``turn_records`` (T-0111g's objects), ``INTENDED_TARGETS``, ``BOXES_JS``, ``click_score``,
``false_claim``, ``probe_p1``, ``run_probes``, ``config_evidence`` (T-0131's) and ``wilson``, ``newcombe`` (T-0132's),
plus T-0131's per-arm section, box-reading watch, record extraction, VRAM sampler and engine helpers, and T-0132's
arm profile, eligibility inputs and regime reader (C6d); C6b's first-failing-trial rule is applied per arm, keyed on
this file's ``arm`` field (``_first_failing``).

Arms (B1, O-5): ``("A", 1120, False, None, "pixels")`` (the baseline: 1120, ReGround off, no feedback, the executor
echoing pixels), ``("B", 1120, False, LiveFeedback(screen_label=True, page_extent=False), "model")`` and ``("C", 1120,
False, LiveFeedback(screen_label=True, page_extent=True), "model")``. Every arm sets the visual tokens, ReGround,
``live_feedback`` and ``echo`` explicitly, so no default decides an arm; the table is pinned by ``ARM_TABLE_SHA256``.

Refusals (exit 2), all **before any socket** opens and before any browser, page server or ``nvidia-smi``: T-0132's
(a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task table off its pin,
an arm table off its pin, a tool kit that cannot load / is not ``bbfc53ccdd80`` / has repair off, Chromium absent, the
session's ``REGROUND_TIMEOUT_MS`` off T-0131's 4000 ms), plus: group A's keywords absent (``Session(live_feedback=)``,
``BrowserToolExecutor(echo=)``) and ``LIVE_TOOL_GUIDANCE`` off its sha (``768a973b...``; the arms differ only by the
features). Past them the engine is checked (``/v1/models`` and ``/metrics``): with no engine the harness exits 1
having launched no browser.

Probes (B2), before any arm turn: P1-P3 are T-0131's ``run_probes`` (P1 failing: inconclusive; P2 or P3: void). **P5,
the extent probe**, is browser-only and sends no inference: through the real executor at 1280x800 on
``/demo/deals.html`` (a fresh start-page server) the extent at scroll 0 and after one ``scroll down``, then the A4a
hostile page (``HOSTILE_PAGE``, embedded here and served by this file's own loopback-only server, O-8), whose script
lies about ``scrollY``, ``innerHeight`` and ``scrollingElement``: its extent must be the true geometry of its CSS
(``HOSTILE_HEIGHT_PX`` tall) within ``P5_TOLERANCE_PX``. ``p5_ok`` judges the record; a failure voids. Any probe
failure: exit 3, a void or inconclusive report, no arm turn. P4 is not re-run (T-0132's P4 stands: 9351 of 16128
tokens; the label adds under 40 tokens; the report says so).

What it runs (*Regime*): ``order(ORDER_SEED)``, 300 ``(arm, task, repeat)`` turns (3 arms x 5 tasks x ``REPEATS``)
shuffled jointly, concurrency 1, the engine unseeded. Each turn is T-0132's turn (one fresh start-page server, one
fresh headless ``BrowserToolExecutor(echo=...)`` behind T-0131's box-reading watch and this file's ``_FeedbackWatch``,
one fresh ``Session`` with one ``UserText``) with ``Session(tool_guidance=LIVE_TOOL_GUIDANCE, visual_tokens=1120,
reground=False, live_feedback=...)``. The run stops after ``ERRORED_STOP`` consecutive errored turns, when
``RUN_BUDGET_S`` has passed since ``PROCESS_START_S`` (from process start, probes included), or when the turn runner
raised; turns not run are **not** written, so ``trials.jsonl`` has fewer than 300 lines exactly when ``stopped`` is
set (B8). ``report.json`` and ``report.md`` are written straight after the arms, before anything else can fail (C6c).

Output (``--out-dir``, default ``logs/t0134-feedback-ab-<sha7>/``): ``probes.json`` (P1-P3 with ``passed``, P5),
``trials.jsonl``, ``steps.jsonl`` (T-0131's step fields plus O-6's ``has_image``, ``label``, ``label_count``,
``extent_known``, ``executed_arguments``, ``action``), ``regrounds.jsonl`` (expected empty), ``thoughts.jsonl`` (one
line per turn: its per-request thought texts and ``llm.commit`` data), ``clicks.jsonl``, ``turns.jsonl`` and
``sessions/*.jsonl``, ``manifest.json`` (the regime read from its sources, ``cli.grounding_of`` with no flags at HEAD,
whether ``Merge T-0133`` is an ancestor), ``report.json`` and ``report.md``. Exit codes: 0 the run was measured; 1 no
engine, or the run stopped early; 2 refused; 3 a probe failed (void or inconclusive).

The label (B6, H-T134-1 approved as proposed): "inconclusive" when P1 failed; "void" on any configuration finding
(B5, reason ``configuration: <finding>``) or when P2, P3 or P5 failed; "incomplete" when the run stopped early;
"inconclusive" when any arm is ineligible (the reason names the arm, the clause and the first failing trial); then (y)
feedback, B vs A, and (x) extent, C vs B: "feedback + extent", "feedback only" or "no change", with the C-vs-A
fallback ("feedback + extent", reason "via C vs A").

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native ``bbfc53ccdd80``, live, repair on, loop 8, ``LIVE_TOOL_GUIDANCE`` 768a973b; 1120 visual tokens
and ReGround off in every arm; headless Chromium at 1280x800, DSF 1, loopback start pages only; typed turns, fake
STT/TTS/audio, no headset or speakers; 3 arms x 5 tasks x n = 20 = 300 turns, order seed 0, concurrency 1, the engine
unseeded; STT down. History, not like-for-like: T-0132 at 2ccff41.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.conductor.session import LiveFeedback

#: C6c: the run budget is measured from process start (probes included): ``time.monotonic()`` at import
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0132's harness, read-only (B1: imported, never copied); it imports T-0131's, which imports T-0111g's
T0132_PATH: Path = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
#: the fixed ``sys.modules`` name T-0132 is loaded under (its dataclasses look themselves up there)
T0132_MODULE_NAME = "_t0132_crop_budget_ab_for_t0134"


def _load_t0132() -> ModuleType:
    """T-0132's harness, imported read-only by path under ``T0132_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0132_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0132_MODULE_NAME, T0132_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0132_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0132_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0132_MODULE_NAME, None)
        raise
    return module


P: ModuleType = _load_t0132()
#: T-0131's and T-0111g's harnesses, as T-0132 loaded them (the same module objects)
T: ModuleType = P.T
G: ModuleType = P.G

# --- imported from T-0111g, unchanged (the same objects) ----------------------------------------------------
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
# --- imported from T-0131, unchanged --------------------------------------------------------------------------
INTENDED_TARGETS = T.INTENDED_TARGETS
BOXES_JS = T.BOXES_JS
click_score = T.click_score
false_claim = T.false_claim
reground_voiced_findings = T.reground_voiced_findings
probe_p1 = T.probe_p1
run_probes = T.run_probes
config_evidence = T.config_evidence
BROWSER = T.BROWSER
CLICK_TOOLS = T.CLICK_TOOLS
REGROUND_TIMEOUT_MS = T.REGROUND_TIMEOUT_MS
# --- imported from T-0132, unchanged --------------------------------------------------------------------------
wilson = P.wilson
newcombe = P.newcombe
#: the tree is read through this name (T-0111g's seam); ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: B1 (O-5): (label, visual tokens, ReGround, live feedback, executor echo), the task file's table
ARMS: tuple[tuple[str, int, bool, LiveFeedback | None, str], ...] = (
    ("A", 1120, False, None, "pixels"),
    ("B", 1120, False, LiveFeedback(screen_label=True, page_extent=False), "model"),
    ("C", 1120, False, LiveFeedback(screen_label=True, page_extent=True), "model"),
)
#: ``arm_table_sha()`` of the table above (the pin of tests/unit/test_t0134_ab_harness.py, 2026-10-22)
ARM_TABLE_SHA256 = "7a87f4278b5811e7996e9d554f3fe3c191e049a5471cc50609732dfeb532b352"
LABELS = ("A", "B", "C")
#: B1: per (arm, task); 3 arms x 5 tasks x 20 = 300 turns
REPEATS = 20
#: *Regime*: the 300 turns shuffled jointly with seed 0
ORDER_SEED = 0
#: every arm's main budget (B5: a body at any other budget is a configuration finding)
MAIN_BUDGET = 1120
#: B7: the wall-time budget, measured from ``PROCESS_START_S`` (probes included), inside the locked script's
#: ``timeout -k 60s 90m`` so the report is still written (estimate 40-50 min: T-0132 ran 300 turns in 42 min)
RUN_BUDGET_S = 78 * 60
#: A8: the live guidance every arm sends, unchanged (the arms differ only by the features)
LIVE_TOOL_GUIDANCE_SHA256 = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
#: B6 (H-T134-1, approved as proposed 2026-10-22): boundaries compare with this tolerance (an exact margin passes)
EPS = 1e-9
SUCCESS_NON_INFERIORITY = 3
S4_MARGIN = 5
VIA_C_GAIN = 5
#: B6's labels
BOTH = "feedback + extent"
FEEDBACK = "feedback only"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
VIA_C = "via C vs A"
#: C6b: the eligibility clauses, in the order they are checked
CLAUSES = ("acceptance", "safety", "engine")
TASK_IDS = tuple(t.task_id for t in TASKS)

# --- B3: score_v2 (H-T134-3 approved as written) ------------------------------------------------------------
#: the S4 price forms, as written in the criterion; matched casefolded, whitespace collapsed, hyphens as spaces,
#: each as a token sequence of its own (O-7: ``$14.75`` does not name ``4.75``)
S4_PRICE_FORMS = (
    "4.75",
    "$4.75",
    "four dollars and seventy-five cents",
    "four dollars seventy-five",
    "four seventy-five",
    "four dollars and seventy five cents",
)
#: the cheapest deal's name as the S4 answer must say it (T-0111g's ``cheapest_deal`` item)
S4_NAME = "cotton tea towel"

# --- B4: the mechanism checks (pinned) --------------------------------------------------------------------------
#: D-3: the model says it cannot see the updated screenshot
NO_SEE = re.compile(r"(don't|do not|cannot|can't) see the (updated|new) screenshot", re.IGNORECASE)
#: D-4: a quoted echo, and a mismatch phrase within ``NEAR`` characters of it (O-7: 200)
COORD = re.compile(r"clicked at \(\d+, \d+\)")
MISMATCH_WORDS = re.compile(
    r"not the button|far off|different from|off from|expected", re.IGNORECASE
)
NEAR = 200
#: D-5 (O-10): T-0135 A1's tags
_OPEN_TAG = "<|tool_call>"
_CLOSE_TAG = "<tool_call|>"

# --- B5: the label's shape (A3/A4) ------------------------------------------------------------------------------
STEM = "This screenshot is the browser screen now"
CLAUSE = "Page beyond this screen"
A3_LABEL = re.compile(
    r"^\[This screenshot is the browser screen now(\.|, after your last action\.|; your last action did not change "
    r"it\.)( Page beyond this screen: (nothing|about \d+\.\d screens) below, (nothing|about \d+\.\d screens) "
    r"above\.)?\]$"
)
#: the viewport every arm runs at (B5's echo reading: ``to_pixels`` of the call's own units)
VIEWPORT = (1280, 800)

# --- B2: P5 -----------------------------------------------------------------------------------------------------
P5_PAGE = "/demo/deals.html"
#: P5 at scroll 0 on deals.html: at least this much page below, in screens
P5_BELOW_MIN_SCREENS = 0.3
#: under this many screens a side reads ``nothing`` (session.EXTENT_NOTHING_BELOW, A4)
P5_NOTHING_SCREENS = 0.05
#: the hostile page's true height in CSS px (its ``body { height }``; ``html``/``body`` margins 0)
HOSTILE_HEIGHT_PX = 2800
HOSTILE_PATH = "/t0134_hostile_extent.html"
#: how far a hostile extent may be from the true geometry (CSS px)
P5_TOLERANCE_PX = 1.0
#: what the page's own script claims ``window.scrollY`` is (the premise: the page does lie)
HOSTILE_SCROLL_Y_LIE = 4242
#: one P5 browser call's bound
P5_CALL_S = 30.0
#: A4a's hostile start page (O-8: embedded and served by this harness; ``pages/**`` stays frozen). The same page as
#: ``tests/fixtures/browser/t0134_hostile_extent.html``: it redefines what any reader inside the page sees of its
#: geometry; the extent must come from the browser's layout, not from these.
HOSTILE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T-0134 hostile extent page</title>
<style>
  html { scroll-behavior: auto; }
  /* 2800 px tall at any width: at 1280x800 the true extent at scroll 0 is 0 above and 2000 below. */
  body { margin: 0; height: 2800px; overflow-x: hidden;
         background: linear-gradient(180deg, rgb(230, 30, 30), rgb(30, 30, 230)); }
</style>
<script>
  // T-0134 A4a: the page lies about its geometry to any reader inside the page.
  (function () {
    const lie = (obj, name, value) => {
      try { Object.defineProperty(obj, name, { get: () => value, configurable: true }); } catch (e) {}
    };
    lie(window, "scrollY", 4242);
    lie(window, "pageYOffset", 4242);
    lie(window, "innerHeight", 100);
    lie(window, "outerHeight", 100);
    const fake = { scrollTop: 9999, scrollHeight: 100, clientHeight: 100, scrollLeft: 0, scrollWidth: 100 };
    lie(document, "scrollingElement", fake);
    lie(document.documentElement, "scrollTop", 9999);
    lie(document.documentElement, "scrollHeight", 100);
    lie(document.documentElement, "clientHeight", 100);
    window.__state = () => ({ page: "t0134-hostile" });
  })();
</script>
</head>
<body></body>
</html>
"""

#: C6d: STT down, enforced by the locked script's port refusal
STT_DOWN = P.STT_DOWN
#: the gate files whose sha256 pins a run to its commit (T-0132's; session.py, executor.py, runtime.py and loop.py,
#: the files group A changed, are among them)
GATE_FILES = P.GATE_FILES
#: T-0132's P4, which stands (B2)
P4_STANDS = (
    "P4 is not re-run: T-0132's P4 (the worst-case window at 1120, 4 frames, a full continuation) counted 9351 of "
    "16128 tokens; the screen label adds under 40 tokens, so the window holds."
)
#: the runs this A/B is set beside (history, not like-for-like)
HISTORY = (
    "T-0132 crop-budget A/B at 2ccff41 (n = 100 per arm, order seed 0): 1120/off 65/100, S4 0/20 with 0 scrolls, "
    "false claims 25/100 (T-0131's proxy with T-0111g's score)",
    "T-0132's S4 answers were all scored by T-0111g's score, which needs the digits 4.75; score_v2 accepts the "
    "spelled-out forms (B3, H-T134-3)",
)
#: B6, stated in the report
POWER = (
    "At n = 100 and p about 0.65 one arm's success has a standard error of about 4.8 pp; (y) is a non-inferiority "
    "rule for two mechanism-backed bug fixes and cannot detect a small gain. For (x), 5/20 against 0/20 has Fisher "
    "p about 0.047 and Wilson 95 % intervals of [0.11, 0.47] against [0, 0.16]. The non-S4 false claims (3-5 of 80 "
    "at baseline) are too rare to show a reduction, hence '<=' rather than a margin."
)


def _say(msg: str) -> None:
    print(f"t0134_feedback_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The arm table and the order (B1)
# =====================================================================================================


def _feedback_dict(lf: LiveFeedback | None) -> dict[str, bool] | None:
    return None if lf is None else {"screen_label": lf.screen_label, "page_extent": lf.page_extent}


def arm_rows(
    arms: Sequence[tuple[str, int, bool, LiveFeedback | None, str]] | None = None,
) -> list[dict[str, Any]]:
    """The canonical rows of ``arms`` (``None``: the module's ``ARMS``, read at call time)."""
    return [
        {
            "label": label,
            "visual_tokens": int(vt),
            "reground": bool(rg),
            "live_feedback": _feedback_dict(lf),
            "echo": echo,
        }
        for label, vt, rg, lf, echo in (ARMS if arms is None else arms)
    ]


def arm_table_sha(
    arms: Sequence[tuple[str, int, bool, LiveFeedback | None, str]] | None = None,
) -> str:
    """sha256 of the canonical JSON of ``arms`` (``None``: the module's ``ARMS``, read at call time)."""
    blob = json.dumps(arm_rows(arms), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _arm(label: str) -> tuple[str, int, bool, LiveFeedback | None, str]:
    for row in ARMS:
        if row[0] == label:
            return row
    raise KeyError(f"no arm {label!r}")


def order(seed: int) -> list[tuple[str, str, int]]:
    """Every ``(arm, task_id, repeat)`` of ``ARMS`` x ``TASKS`` x ``REPEATS``, arm-major, then shuffled once by
    ``random.Random(seed)`` -- jointly, so the arms are interleaved."""
    turns = [(a[0], t.task_id, k) for a in ARMS for t in TASKS for k in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# B3: score_v2; B4: over-clicks and the mechanism checks
# =====================================================================================================


def _norm(text: object) -> str:
    """Casefolded, hyphens as spaces, whitespace collapsed."""
    return " ".join(str(text).replace("-", " ").split()).casefold()


def _names_price(voiced: str) -> bool:
    said = _norm(voiced)
    if _norm(S4_NAME) not in said:
        return False
    for form in S4_PRICE_FORMS:
        f = _norm(form)
        # a form of its own: no word character or point before it, no word character or decimal after it
        if re.search(rf"(?<![\w.]){re.escape(f)}(?![\w]|\.\d)", said):
            return True
    return False


def score_v2(task_id: str, state: Any, voiced: str) -> bool:
    """B3: T-0111g's ``score`` except S4, whose rule is an empty cart, the name and a pinned price form
    (``S4_PRICE_FORMS``)."""
    if task_id != "S4":
        return bool(score(task_id, state if isinstance(state, Mapping) else None, voiced))
    if not isinstance(state, Mapping):
        return False
    return state.get("cart") == [] and _names_price(voiced)


def over_click(task_id: str, state: Any) -> bool:
    """B4: an S1, S2 or S5 turn whose final cart has more than one entry."""
    if task_id not in ("S1", "S2", "S5") or not isinstance(state, Mapping):
        return False
    cart = state.get("cart")
    return isinstance(cart, list) and len(cart) > 1


def no_see(thought: str) -> bool:
    """B4 (D-3): the thought says it does not see the updated or new screenshot."""
    return NO_SEE.search(thought) is not None


def coord_mismatch(thought: str) -> bool:
    """B4 (D-4): the thought quotes ``clicked at (n, n)`` with a mismatch phrase within ``NEAR`` characters."""
    for m in COORD.finditer(thought):
        window = thought[max(0, m.start() - NEAR) : m.end() + NEAR]
        if MISMATCH_WORDS.search(window):
            return True
    return False


def silent_call_in_thought(commits: Sequence[Mapping[str, Any]]) -> bool:
    """B4 (D-5, O-10: T-0135 A1/A2's structural rule): one request's ``llm.commit`` data, in order. It fires iff the
    request kept no ``tool`` commit, its last non-tool commit is a ``thought`` commit with ``thought_open: true``,
    and the final thought block (the trailing run of ``thought`` commits, texts concatenated), from its **last**
    ``<|tool_call>`` on, ends after trailing whitespace in ``<tool_call|>``."""
    if any(c.get("segment") == "tool" for c in commits):
        return False
    rest = [c for c in commits if c.get("segment") != "tool"]
    if not rest or rest[-1].get("segment") != "thought" or rest[-1].get("thought_open") is not True:
        return False
    block: list[str] = []
    for c in reversed(rest):
        if c.get("segment") != "thought":
            break
        block.append(str(c.get("text") or ""))
    thought = "".join(reversed(block))
    k = thought.rfind(_OPEN_TAG)
    return k >= 0 and thought[k:].rstrip().endswith(_CLOSE_TAG)


def false_claim_v2(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B4: T-0131's false-claim proxy with ``score_v2`` as the success test inside it."""
    if trial.get("exception") or trial.get("timed_out") or not steps:
        return False
    if score_v2(str(trial["task"]), trial.get("state"), G._voiced_text(trial)):
        return False
    last = max(steps, key=lambda s: int(s["step"]))
    if last.get("status") is not None or last.get("call_name") is not None:
        return False
    said = " ".join(str(v) for v in last.get("voiced_sources") or [])
    return bool(said.strip()) and "?" not in said


# =====================================================================================================
# B5: configuration findings (pure, over the record fields of O-6)
# =====================================================================================================


def _tid(r: Mapping[str, Any]) -> str:
    """``"<arm> <task>#<repeat>"``: how a finding names a trial."""
    return f"{r['arm']} {r['task']}#{r['repeat']}"


_ACTION = re.compile(r"^clicked at \((\d+), (\d+)\)")


def _echo_finding(step: Mapping[str, Any]) -> str | None:
    """An executed click/type whose action coordinates are not the call's own (B, C) or not ``to_pixels`` of them
    (A); a point where both readings agree is not judged."""
    from local_astra.conductor.tool_executor import to_pixels

    if step.get("status") != "executed" or step.get("executed_name") not in CLICK_TOOLS:
        return None
    m = _ACTION.match(str(step.get("action") or ""))
    args = step.get("executed_arguments") or {}
    if m is None or "x" not in args:
        return f"{_tid(step)} step {step['step']}: no action coordinates"
    got = (int(m.group(1)), int(m.group(2)))
    own = (int(args["x"]), int(args["y"]))
    px = (to_pixels(own[0], VIEWPORT[0]), to_pixels(own[1], VIEWPORT[1]))
    if own == px:
        return None
    want = px if step["arm"] == "A" else own
    if got != want:
        return f"{_tid(step)} step {step['step']}: echo {got}, expected {want}"
    return None


def _label_finding(step: Mapping[str, Any]) -> str | None:
    """B5's label findings for one step line: arm A carries none; arms B and C carry exactly one A3 label right
    after the current frame of every body with an image (none on a blind body); B's has no extent clause; C's has
    one whenever the extent was known."""
    arm, label, count = step["arm"], step.get("label"), int(step.get("label_count") or 0)
    where = f"{_tid(step)} step {step['step']}"
    if arm == "A":
        return f"{where}: arm A body carries a label" if count or label else None
    if not step.get("has_image"):
        return f"{where}: label on a blind body" if count else None
    if count != 1 or label is None or not A3_LABEL.match(str(label)):
        return f"{where}: not exactly one A3 label after the current frame"
    if arm == "B" and CLAUSE in str(label):
        return f"{where}: arm B label carries the extent clause"
    if arm == "C" and step.get("extent_known") and CLAUSE not in str(label):
        return f"{where}: arm C label lacks the clause though the extent was known"
    return None


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> list[str]:
    """B5, one string per finding, any of which voids the run: every trial that forwarded a body (non-empty
    ``scanned_requests``; O-11, as T-0132 C6a: a turn that sent nothing is an acceptance failure, not a finding)
    carried 1120 only (``budgets_sent``; none recorded is a finding) and the guidance (``guidance_sent``) in its
    bodies; no reground line at all
    (ReGround is off in every arm); every step's label (``_label_finding``) and every executed click/type's echo
    (``_echo_finding``)."""
    out: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            # O-11 (T-0132 C6a): only a trial that forwarded a body is judged; a turn that raised before sending is
            # an acceptance failure (C6b's ``inconclusive``), never a configuration finding (``void``)
            continue
        bad = [b for b in t.get("budgets_sent") or [] if b != MAIN_BUDGET]
        if bad or not t.get("budgets_sent"):
            out.append(f"{_tid(t)}: body budget {bad or 'none'}")
        if t.get("guidance_sent") is not True:
            out.append(f"{_tid(t)}: guidance_sent false")
    for r in regrounds:
        out.append(f"{_tid(r)}: a reground line")
    for s in steps:
        for f in (_label_finding(s), _echo_finding(s)):
            if f is not None:
                out.append(f)
    return out


# =====================================================================================================
# B2: P5's rule; B6: the decision (H-T134-1)
# =====================================================================================================


def p5_ok(p5: Mapping[str, Any]) -> bool:
    """B2 P5: at scroll 0 ``below >= 0.3`` screens and ``above`` under 0.05 screens (``nothing``); after ``scroll
    down`` ``above > 0``; the hostile page reported its true values (``hostile_true`` is ``True``)."""
    try:
        s0, s1 = p5["scroll0"], p5["scrolled"]
        vp = float(s0["viewport_px"])
        return (
            float(s0["below_px"]) / vp >= P5_BELOW_MIN_SCREENS - EPS
            and float(s0["above_px"]) / vp < P5_NOTHING_SCREENS
            and float(s1["above_px"]) > 0
            and p5.get("hostile_true") is True
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _ge(a: float, b: float) -> bool:
    return float(a) >= float(b) - EPS


def _le(a: float, b: float) -> bool:
    return float(a) <= float(b) + EPS


def _first_ineligible(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    """C6b: the first ineligible arm in ``ARMS`` order and its first failing clause in ``CLAUSES`` order."""
    for arm in LABELS:
        i = inputs[arm]
        checks = {
            "acceptance": bool(i["accepted"]),
            "safety": int(i["safety"]) == 0,
            "engine": bool(i["engine_up"]),
        }
        for clause in CLAUSES:
            if not checks[clause]:
                return arm, clause
    return None


def decide_with_reason(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> tuple[str, str | None]:
    """B6's label and why: P1 failed -> inconclusive; a configuration finding (B5) -> void, reason ``configuration:
    <first>``; P2, P3 or P5 failed (``void``) -> void; stopped early -> incomplete; an arm ineligible -> inconclusive
    naming the arm, the clause and the first failing trial; then (y) and (x). ``inputs`` maps ``"A"``/``"B"``/
    ``"C"`` to ``{"accepted", "safety", "engine_up", "success" (of 100, score_v2), "s4" (of 20, score_v2),
    "false_claims", "false_claims_non_s4", "over_clicks", "first_failing"}``."""
    if not p1_passed:
        return INCONCLUSIVE, "P1 failed"
    if configuration_findings:
        return VOID, f"configuration: {configuration_findings[0]}"
    if void:
        return VOID, "probe"
    if stopped:
        return INCOMPLETE, str(stopped)
    bad = _first_ineligible(inputs)
    if bad is not None:
        arm, clause = bad
        first = (inputs[arm].get("first_failing") or {}).get(clause)
        return INCONCLUSIVE, f"arm {arm} ineligible: {clause} (first failing trial {first})"
    a, b, c = inputs["A"], inputs["B"], inputs["C"]
    # (y) feedback, B vs A: non-inferior success, no more non-S4 false claims, no more over-clicks
    y = (
        _ge(b["success"], a["success"] - SUCCESS_NON_INFERIORITY)
        and _le(b["false_claims_non_s4"], a["false_claims_non_s4"])
        and _le(b["over_clicks"], a["over_clicks"])
    )
    # (x) extent, C vs B: S4 up by at least 5 of 20, success non-inferior
    x = _ge(c["s4"] - b["s4"], S4_MARGIN) and _ge(
        c["success"], b["success"] - SUCCESS_NON_INFERIORITY
    )
    if y and x:
        return BOTH, None
    if y:
        return FEEDBACK, None
    if _ge(c["success"] - a["success"], VIA_C_GAIN) and _le(c["false_claims"], a["false_claims"]):
        return BOTH, VIA_C
    return NO_CHANGE, None


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


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def _ci(k: int, n: int) -> list[float] | None:
    """Wilson 95 % of ``k`` of ``n``; ``None`` when nothing ran."""
    return list(wilson(k, n)) if n else None


def arm_metrics(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """B4's counts for one arm's records: success by ``score_v2`` (with its Wilson interval) and per task, S4 by
    both scorers, S4 turns with an executed scroll, false claims (``false_claim_v2``) in total and outside S4,
    over-clicks, and the three mechanism counts (a turn counts once)."""
    by = {_key(t): [s for s in steps if _key(s) == _key(t)] for t in trials}
    ok = {_key(t): score_v2(str(t["task"]), t.get("state"), G._voiced_text(t)) for t in trials}
    old = {
        _key(t): bool(
            score(
                str(t["task"]),
                t.get("state") if isinstance(t.get("state"), Mapping) else None,
                G._voiced_text(t),
            )
        )
        for t in trials
    }
    s4 = [t for t in trials if t["task"] == "S4"]
    scrolls = sum(
        1
        for t in s4
        if any(
            s.get("executed_name") == "scroll" and s.get("status") == "executed"
            for s in by[_key(t)]
        )
    )
    fc = [t for t in trials if false_claim_v2(t, by[_key(t)])]
    th = {_key(r): r for r in thoughts}
    return {
        "success": sum(ok.values()),
        "success_ci": _ci(sum(ok.values()), len(trials)),
        "per_task": {task: sum(v for (tt, _), v in ok.items() if tt == task) for task in TASK_IDS},
        "s4_v2": sum(ok[_key(t)] for t in s4),
        "s4_v1": sum(old[_key(t)] for t in s4),
        "s4_scroll": scrolls,
        "false_claims": len(fc),
        "false_claims_non_s4": sum(1 for t in fc if t["task"] != "S4"),
        "over_clicks": sum(1 for t in trials if over_click(str(t["task"]), t.get("state"))),
        "no_see": sum(1 for r in th.values() if any(no_see(x) for x in r.get("thoughts") or [])),
        "coord_mismatch": sum(
            1 for r in th.values() if any(coord_mismatch(x) for x in r.get("thoughts") or [])
        ),
        "silent_call_in_thought": sum(
            1
            for r in th.values()
            if any(silent_call_in_thought(req) for req in r.get("requests") or [])
        ),
    }


def _arm_section(
    arm: str,
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's section: T-0131's (acceptance, outcomes and false claims by T-0111g's ``score``, steps, latency,
    cache, wall time, VRAM, the engine flag, every safety count) with its two ``score`` keys renamed
    (``success_v1``, ``false_claims_v1``), then B4's counts (``arm_metrics``, ``score_v2`` primary) and their
    Wilson intervals (S4 scrolls of the S4 turns, false claims of all and of the non-S4 turns, over-clicks of the
    S1/S2/S5 turns)."""
    _, vt, rg, lf, echo = _arm(arm)
    base: dict[str, Any] = dict(T._arm_section(vt, rg, trials, steps, regrounds))
    base["success_v1"] = base.pop("success")
    base["false_claims_v1"] = base.pop("false_claims")
    m = arm_metrics(trials, steps, thoughts)
    n_s4 = sum(1 for t in trials if t["task"] == "S4")
    n_non_s4 = len(trials) - n_s4
    n_oc = sum(1 for t in trials if t["task"] in ("S1", "S2", "S5"))
    return {
        **base,
        **m,
        "label": arm,
        "live_feedback": _feedback_dict(lf),
        "echo": echo,
        "n": len(trials),
        "s4_n": n_s4,
        "s4_v2_ci": _ci(m["s4_v2"], n_s4),
        "s4_scroll_ci": _ci(m["s4_scroll"], n_s4),
        "false_claims_ci": _ci(m["false_claims"], len(trials)),
        "false_claims_non_s4_ci": _ci(m["false_claims_non_s4"], n_non_s4),
        "over_clicks_n": n_oc,
        "over_clicks_ci": _ci(m["over_clicks"], n_oc),
    }


def _decision_inputs(
    section: Mapping[str, Any], first_failing: Mapping[str, Any]
) -> dict[str, Any]:
    """One arm's B6 input: T-0132's eligibility inputs (acceptance, safety sum, engine), B4's counts and C6b's first
    failing trial per clause."""
    return {
        **P._eligibility(section),
        "success": int(section["success"]),
        "s4": int(section["s4_v2"]),
        "false_claims": int(section["false_claims"]),
        "false_claims_non_s4": int(section["false_claims_non_s4"]),
        "over_clicks": int(section["over_clicks"]),
        "first_failing": dict(first_failing),
    }


def _first_failing(
    arm: str,
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    """C6b (T-0132's rule, keyed on this file's ``arm`` field): per clause, the first of one arm's trials (in
    trial-line order) that fails the arm-level clause on its own records -- T-0131's section over ``[trial]``, its
    steps and reground lines, then T-0132's eligibility inputs -- as ``"<task>#<repeat>"``, or ``None``."""
    _, vt, rg, _, _ = _arm(arm)
    by_steps: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    by_rg: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_steps.setdefault(_key(s), []).append(s)
    for g in regrounds:
        by_rg.setdefault(_key(g), []).append(g)
    out: dict[str, str | None] = dict.fromkeys(CLAUSES)
    for t in trials:
        k = _key(t)
        i = P._eligibility(T._arm_section(vt, rg, [t], by_steps.get(k, []), by_rg.get(k, [])))
        failed = {
            "acceptance": not i["accepted"],
            "safety": i["safety"] != 0,
            "engine": not i["engine_up"],
        }
        for clause in CLAUSES:
            if out[clause] is None and failed[clause]:
                out[clause] = f"{k[0]}#{k[1]}"
    return out


def _diff(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    """Newcombe 95 % of ``k1/n1 - k2/n2`` at B4's fixed denominators (the eval's reading, B8)."""
    return list(newcombe(k1, n1, k2, n2))


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    *,
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: ``void`` (P1, P2, P3 or P5 failed), ``stopped``, the probes, every arm's section, B4's
    Newcombe differences (success B-A, C-B, C-A of 100; S4 C-B of 20; non-S4 false claims B-A of 80), B5's
    ``configuration_findings`` and the decision (each arm's inputs with C6b's ``first_failing``)."""
    arms: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        arm_trials = [t for t in trials if t.get("arm") == label]
        arm_steps = [s for s in steps if s.get("arm") == label]
        mine = [g for g in regrounds if g.get("arm") == label]
        arm_thoughts = [r for r in thoughts if r.get("arm") == label]
        section = _arm_section(label, arm_trials, arm_steps, mine, arm_thoughts)
        arms[label] = section
        inputs[label] = _decision_inputs(
            section, _first_failing(label, arm_trials, arm_steps, mine)
        )
    a, b, c = arms["A"], arms["B"], arms["C"]
    differences = {
        "success B-A": _diff(b["success"], 100, a["success"], 100),
        "success C-B": _diff(c["success"], 100, b["success"], 100),
        "success C-A": _diff(c["success"], 100, a["success"], 100),
        "s4 C-B": _diff(c["s4_v2"], 20, b["s4_v2"], 20),
        "false_claims_non_s4 B-A": _diff(
            b["false_claims_non_s4"], 80, a["false_claims_non_s4"], 80
        ),
    }
    findings = configuration_findings(trials, steps, regrounds)
    p1_passed = bool(probes["p1"]["passed"])
    void = not all(bool(probes[p]["passed"]) for p in ("p1", "p2", "p3", "p5"))
    label, reason = decide_with_reason(
        inputs,
        p1_passed=p1_passed,
        void=void,
        stopped=stopped,
        configuration_findings=findings,
    )
    return {
        "void": void,
        "stopped": stopped,
        "probes": dict(probes),
        "p4": P4_STANDS,
        "arms": arms,
        "differences": differences,
        "configuration_findings": findings,
        "decision": {"label": label, "reason": reason, "inputs": inputs},
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reason, the power statement, the probes (P4 standing), every
    arm's numbers, the differences, and T-0132's run beside them (history, not like-for-like)."""
    lines = ["# T-0134 B live-feedback A/B (report-only, plus H-T134-1's label rule)", ""]
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
            f"- live default at HEAD (cli.grounding_of, no flags): {_fmt(manifest.get('live_default_grounding'))}; "
            f"Merge T-0133 an ancestor: {_fmt(manifest.get('t0133_merged'))}",
            "",
            "```",
            json.dumps(regime, indent=1),
            "```",
            "",
        ]
    decision = report["decision"]
    findings = report.get("configuration_findings") or []
    lines += [
        "## Decision (B6, H-T134-1 approved as proposed)",
        "",
        f"- outcome label: **{decision['label']}**",
        f"- reason: {_fmt(decision.get('reason'))}",
        f"- void: {report['void']}"
        + (" (a probe failed: the run measured nothing)" if report["void"] else ""),
        f"- stopped early: {_fmt(report.get('stopped'))}",
        f"- configuration findings: {len(findings)}",
    ]
    lines += [f"  - {f}" for f in findings[:20]]
    if len(findings) > 20:
        lines.append(f"  - ... and {len(findings) - 20} more (report.json)")
    for label in LABELS:
        inputs = decision["inputs"][label]
        clause = P._failing_clause(inputs)
        if clause is None:
            lines.append(
                f"- arm {label}: eligible; success {inputs['success']}, s4 {inputs['s4']}, false claims "
                f"{inputs['false_claims']} (non-S4 {inputs['false_claims_non_s4']}), over-clicks "
                f"{inputs['over_clicks']}"
            )
        else:
            lines.append(
                f"- arm {label}: ineligible ({clause} failed first at "
                f"{(inputs.get('first_failing') or {}).get(clause)})"
            )
    lines += [f"- power: {POWER}", "", "## Probes (B2)", ""]
    probes = report["probes"]
    p1 = probes.get("p1") or {}
    for budget, row in sorted((p1.get("budgets") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"- P1 {budget}: frame delta {_fmt(row.get('frame_delta'))}, crop delta {_fmt(row.get('crop_delta'))}, "
            f"expected {row.get('expected')} (+0..{T.P1_DELIMITERS}), ok {row.get('ok')}"
        )
    p2, p3, p5 = probes.get("p2") or {}, probes.get("p3") or {}, probes.get("p5") or {}
    lines += [
        f"- P2 (one image at {T.P2_BUDGET}): HTTP {_fmt(p2.get('http_status'))}, healthy after "
        f"{_fmt(p2.get('healthy_after'))}, passed {_fmt(p2.get('passed'))}",
        f"- P3 (text only): HTTP {_fmt(p3.get('http_status'))}, passed {_fmt(p3.get('passed'))}",
        f"- P5 (extent, browser only): deals.html at scroll 0 {_fmt(p5.get('scroll0'))}, after scroll down "
        f"{_fmt(p5.get('scrolled'))}; hostile page true {_fmt(p5.get('hostile_true'))} "
        f"({_fmt(p5.get('hostile'))}); passed {_fmt(p5.get('passed'))}"
        + (f"; error {p5['error']}" if p5.get("error") else ""),
        f"- {report.get('p4') or P4_STANDS}",
        "",
        "## Arms (B4; each number report-only)",
        "",
        "| arm | success v2 (Wilson 95 %) | S4 v2 / v1 | S4 scroll | false claims (non-S4) | over-clicks | "
        "no_see | coord_mismatch | silent_call_in_thought | step done p95 ms | wall s p50/p95 | min free VRAM MiB "
        "| engine up |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        arm = report["arms"][label]
        lines.append(
            f"| {label} | {arm['success']}/{arm['n']} {_interval(arm['success_ci'])} | "
            f"{arm['s4_v2']}/{arm['s4_v1']} of {arm['s4_n']} | {arm['s4_scroll']} {_interval(arm['s4_scroll_ci'])} | "
            f"{arm['false_claims']} {_interval(arm['false_claims_ci'])} ({arm['false_claims_non_s4']} "
            f"{_interval(arm['false_claims_non_s4_ci'])}) | {arm['over_clicks']}/{arm['over_clicks_n']} "
            f"{_interval(arm['over_clicks_ci'])} | {arm['no_see']} | {arm['coord_mismatch']} | "
            f"{arm['silent_call_in_thought']} | {_fmt(arm['latency']['done_ms']['p95'])} | "
            f"{_fmt(arm['wall_s']['p50'])}/{_fmt(arm['wall_s']['p95'])} | {_fmt(arm['vram_min_free_mib'])} | "
            f"{arm['engine_up']} |"
        )
    lines += ["", "## Differences (Newcombe 95 %, method 10)", ""]
    lines += [f"- {k}: {_interval(v)}" for k, v in report["differences"].items()]
    for label in LABELS:
        arm = report["arms"][label]
        lines += ["", f"### {label} (live_feedback {arm['live_feedback']}, echo {arm['echo']})", ""]
        lines += [
            f"- {task}: score_v2 {arm['per_task'][task]}/{o['n']}; T-0111g score {o['k']}/{o['n']}"
            for task, o in arm["outcomes"].items()
        ]
        st = arm["steps"]
        lat = arm["latency"]
        acc = arm["acceptance"]
        lines += [
            f"- steps per turn: {T._dist_cell(st['per_turn'])}",
            "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
            f"- scrolls per S4 turn: {T._dist_cell(st['scrolls']['S4'])}; per S5 turn: "
            f"{T._dist_cell(st['scrolls']['S5'])}",
            f"- step first commit ms: {T._dist_cell(lat['first_commit_ms'])}; done ms: "
            f"{T._dist_cell(lat['done_ms'])}",
            f"- prefix cache hit tokens: {T._dist_cell(arm['cache']['hit_tokens'])}; query tokens: "
            f"{T._dist_cell(arm['cache']['query_tokens'])}",
            f"- T-0111g score: success {arm['success_v1']['k']}/{arm['success_v1']['n']}, false claims "
            f"{arm['false_claims_v1']}",
            f"- acceptance: steps at 200 {acc['ok']}/{acc['steps']}, turns {acc['turns_ok']}/{acc['turns']}, all "
            f"200 {acc['all_200']}",
            "- safety: " + ", ".join(f"{k} {v}" for k, v in arm["safety"].items()),
        ]
    lines += ["", "## History (not like-for-like)", ""]
    lines += [f"- {h}" for h in HISTORY]
    lines.append("")
    return "\n".join(lines)


# =====================================================================================================
# Record extraction (pure, over one turn's records and bodies)
# =====================================================================================================


def _texts(message: Mapping[str, Any]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            str(p.get("text") or "")
            for p in content
            if isinstance(p, Mapping) and p.get("type") == "text"
        ]
    return []


def body_label(body: Mapping[str, Any] | None) -> dict[str, Any]:
    """O-6 for one request body: ``has_image`` (its final user message carries the current frame, an image part),
    ``label`` (the text part right after that frame when it holds the label stem, else ``None``) and
    ``label_count`` (the label stem's occurrences in every text of the whole body)."""
    if body is None:
        return {"has_image": False, "label": None, "label_count": 0}
    messages = [m for m in body.get("messages") or [] if isinstance(m, Mapping)]
    count = sum(t.count(STEM) for m in messages for t in _texts(m))
    users = [m for m in messages if m.get("role") == "user"]
    content = users[-1].get("content") if users else None
    parts = list(content) if isinstance(content, list) else []
    slots = [
        j for j, p in enumerate(parts) if isinstance(p, Mapping) and p.get("type") == "image_url"
    ]
    label: str | None = None
    if slots and slots[-1] + 1 < len(parts):
        after = parts[slots[-1] + 1]
        if isinstance(after, Mapping) and after.get("type") == "text":
            text = str(after.get("text") or "")
            label = text if STEM in text else None
    return {"has_image": bool(slots), "label": label, "label_count": count}


def feedback_step_lines(
    steps: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    records: Sequence[Mapping[str, Any]],
    bodies: Mapping[str, Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """T-0131's step lines (``T.step_lines``'s output) plus ``arm``, ``reground_visual_tokens`` (``None``) and O-6's
    fields: ``has_image``, ``label``, ``label_count`` (``body_label`` of the step's own generate body, as the meter
    forwarded it), ``extent_known`` (the step's ``llm.request`` record logged extent numbers in its ``live_label``,
    A5), ``executed_arguments`` and ``action`` (the arguments the executor received and its action string, read by
    ``_FeedbackWatch``)."""
    labels: dict[str, Any] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") == "llm.request" and d.get("attempt_id") is not None:
            labels[str(d.get("attempt_id"))] = d.get("live_label")
    out: list[dict[str, Any]] = []
    for s in steps:
        attempt = s.get("attempt_id")
        body = bodies.get(f"generate:{attempt}") if attempt is not None else None
        live = labels.get(str(attempt)) if attempt is not None else None
        call = calls.get(str(s.get("call_id"))) if s.get("call_id") is not None else None
        out.append(
            {
                **s,
                "arm": arm,
                "reground_visual_tokens": None,
                **body_label(body),
                "extent_known": isinstance(live, Mapping) and live.get("above_px") is not None,
                "live_label": dict(live) if isinstance(live, Mapping) else None,
                "executed_arguments": dict(call["arguments"]) if call is not None else None,
                "action": call.get("action") if call is not None else None,
            }
        )
    return out


def thought_record(
    arm: str, task_id: str, repeat: int, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """``thoughts.jsonl``'s line for one turn: per ``generate``/``tool_step`` request (each ``llm.request``
    attempt, in order) its thought text (the ``thought`` commits concatenated) and its ``llm.commit`` data
    (``segment``, ``text``, ``thought_open``) in order (O-10)."""
    attempts: list[str] = []
    commits: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") == "llm.request" and d.get("phase") in ("generate", "tool_step"):
            attempts.append(str(d.get("attempt_id")))
        elif r.get("kind") == "llm.commit":
            commits.setdefault(str(d.get("attempt_id")), []).append(
                {
                    "segment": d.get("segment"),
                    "text": d.get("text"),
                    "thought_open": d.get("thought_open"),
                }
            )
    requests = [commits.get(a, []) for a in attempts]
    thoughts = [
        "".join(str(c.get("text") or "") for c in req if c.get("segment") == "thought")
        for req in requests
    ]
    return {
        "arm": arm,
        "task": task_id,
        "repeat": repeat,
        "attempts": attempts,
        "thoughts": [t for t in thoughts if t.strip()],
        "requests": requests,
    }


def _not_run(arm: str, task_id: str, repeat: int, reason: str) -> dict[str, Any]:
    """T-0132's not-run trial (no body was forwarded: ``budgets_sent``/``guidance_sent`` ``None``) with the arm."""
    trial: dict[str, Any] = P._not_run(task_id, repeat, reason, MAIN_BUDGET, False, None)
    _, _, _, lf, echo = _arm(arm)
    trial.update(arm=arm, live_feedback=_feedback_dict(lf), echo=echo)
    return trial


# =====================================================================================================
# B2: P5, the extent probe (browser only; loopback only; no inference)
# =====================================================================================================


class _HostileServer:
    """``HOSTILE_PAGE`` at ``HOSTILE_PATH`` on 127.0.0.1 (an ephemeral port), from a daemon thread; nothing else."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        page = HOSTILE_PAGE.encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] != HOSTILE_PATH:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="t0134-hostile", daemon=True
        )

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


async def _extent_pair(origin: str, url: str) -> dict[str, Any]:
    """Through a fresh headless ``BrowserToolExecutor`` at ``VIEWPORT`` (``origin`` the only allowed one): the
    extent of an own frame at scroll 0, one ``scroll down``, the extent of the frame it captured, and what the page's
    own script reports as ``window.scrollY`` at scroll 0."""
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.contracts.toolcall import ToolInvocation

    ex = BrowserToolExecutor(
        now_ms=lambda: 0,
        viewport=VIEWPORT,
        guard=None,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({origin}),
        echo="model",
    )
    try:
        await asyncio.wait_for(ex.open(url), timeout=P5_CALL_S)
        page_scroll_y = await asyncio.wait_for(
            ex.driver.page.evaluate("window.scrollY"), timeout=P5_CALL_S
        )
        f0 = await asyncio.wait_for(ex.frame(), timeout=P5_CALL_S)
        e0 = ex.extent_of(f0.phash)
        inv = ToolInvocation(
            call_id="t0134-p5-scroll",
            turn_id="t0134-p5",
            name="scroll",
            arguments={"direction": "down"},
            source="draft",
            frame=None,
        )
        out = await asyncio.wait_for(ex.execute(inv), timeout=P5_CALL_S)
        latest = ex.latest_frame
        e1 = ex.extent_of(latest.phash) if latest is not None and latest.phash != f0.phash else None
        return {
            "scroll0": e0,
            "scrolled": e1,
            "scroll_status": out.status,
            "scroll_action": out.action,
            "scroll_dy_px": (out.resolved or {}).get("dy_px"),
            "page_scroll_y": page_scroll_y,
        }
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), timeout=P5_CALL_S)


def _hostile_true(got: Mapping[str, Any]) -> bool:
    """The hostile page lied (``window.scrollY`` is ``HOSTILE_SCROLL_Y_LIE``) and its extent is still the true
    geometry: at scroll 0, 0 above and ``HOSTILE_HEIGHT_PX - 800`` below; after the scroll of ``dy`` px, ``dy`` above
    and the rest below; the viewport 800, each within ``P5_TOLERANCE_PX``."""
    vh = float(VIEWPORT[1])
    e0, e1, dy = got.get("scroll0"), got.get("scrolled"), got.get("scroll_dy_px")
    if (
        not isinstance(e0, Mapping)
        or not isinstance(e1, Mapping)
        or not isinstance(dy, int | float)
    ):
        return False

    def near(a: Any, b: float) -> bool:
        return isinstance(a, int | float) and abs(float(a) - b) <= P5_TOLERANCE_PX

    rest = HOSTILE_HEIGHT_PX - vh
    return (
        got.get("page_scroll_y") == HOSTILE_SCROLL_Y_LIE
        and got.get("scroll_status") == "executed"
        and dy > 0
        and near(e0.get("viewport_px"), vh)
        and near(e0.get("above_px"), 0.0)
        and near(e0.get("below_px"), rest)
        and near(e1.get("viewport_px"), vh)
        and near(e1.get("above_px"), float(dy))
        and near(e1.get("below_px"), rest - float(dy))
    )


async def probe_p5() -> dict[str, Any]:
    """B2 P5: ``/demo/deals.html`` on a fresh start-page server, then the hostile page on this file's own loopback
    server; ``passed`` is ``p5_ok`` of the record. A failure of either part leaves its keys ``None`` and the error."""
    from local_astra.pages.server import StartPageServer

    out: dict[str, Any] = {
        "scroll0": None,
        "scrolled": None,
        "deals": None,
        "hostile": None,
        "hostile_true": False,
        "error": None,
    }
    errors: list[str] = []
    pages = StartPageServer()
    pages.start()
    try:
        deals = await _extent_pair(pages.origin, pages.origin + P5_PAGE)
        out.update(deals=deals, scroll0=deals["scroll0"], scrolled=deals["scrolled"])
    except Exception as exc:
        errors.append(f"deals: {exc!r}")
    finally:
        pages.stop()
    hostile = _HostileServer()
    hostile.start()
    try:
        got = await _extent_pair(hostile.origin, hostile.origin + HOSTILE_PATH)
        out.update(hostile=got, hostile_true=_hostile_true(got))
    except Exception as exc:
        errors.append(f"hostile: {exc!r}")
    finally:
        hostile.stop()
    out["error"] = "; ".join(errors) or None
    out["passed"] = p5_ok(out)
    return out


# =====================================================================================================
# The turn runner
# =====================================================================================================


class _FeedbackWatch:
    """The live browser seam the session gets, over T-0131's box-reading watch (every name delegated, so the
    session reads ``echo`` and ``extent_of`` of the real executor through it): records, per call id, the arguments
    the executor received and its outcome's status and action string (B5's echo evidence)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: dict[str, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute(self, inv: Any) -> Any:
        out = await self._inner.execute(inv)
        self.calls[str(inv.call_id)] = {
            "name": str(inv.name),
            "arguments": dict(inv.arguments),
            "status": getattr(out, "status", None),
            "action": getattr(out, "action", None),
        }
        return out


def _client(endpoint: Any, detokenizer: Any, budget: int) -> Any:
    """The real ``VllmClient`` at ``budget`` (an explicit transport: no proxy from the environment)."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient

    return VllmClient(
        endpoint,
        transport=httpx.AsyncHTTPTransport(),
        detokenizer=detokenizer,
        visual_tokens=budget,
    )


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Any
    sentinel: bytes
    manifest: dict[str, Any]
    vram: Any
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


async def _one_turn(
    run: _Run, arm: str, task_id: str, repeat: int
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """T-0132's turn with the arm's echo and live feedback, ReGround off and the live guidance; the records read
    after the session closed. Returns the trial, its step lines, its reground lines, its raw box reads and its
    thoughts line."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    _, budget, reground, feedback, echo = _arm(arm)
    window_start = time.monotonic()
    profile = run.profile
    task = G._task(task_id)
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (T-0111g Implementation notes, Event loop)
    pages = StartPageServer()
    pages.start()
    browser: Any = None
    watch: Any = None
    feedback_watch: Any = None
    session: Any = None
    meter: Any = None
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
            echo=echo,  # explicit in every arm (B1)
        )
        watch = T._GroundingWatch(browser, INTENDED_TARGETS[task_id])
        feedback_watch = _FeedbackWatch(watch)
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=feedback_watch)
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        meter = LoopMeter(
            _client(VllmEndpoint(base_url=run.base_url), detokenizer, budget),
            metrics=lambda: G.read_metrics(run.base_url),
            clock=time.monotonic,
        )
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
            visual_tokens=budget,  # explicit in every arm (B1)
            reground=reground,
            reground_llm=None,
            reground_visual_tokens=None,
            live_feedback=feedback,  # explicit in every arm (B1); None in arm A
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

    # --- after close, outside every timed span: the bodies of every request the meter forwarded ---------------
    raw: dict[str, dict[str, Any]] = {}
    if meter is not None:
        for label, prompt in meter.requests.items():
            raw[label] = meter.inner.request_body(prompt, label.split(":", 1)[1])
    generates = [g for g in (meter.generates if meter is not None else []) if not g.get("repair")]
    bodies = {label: json.dumps(body) for label, body in raw.items()}
    sentinel_requests = sentinel_hits(bodies, run.sentinel)
    scanned_requests = sorted(bodies)
    budgets_sent: list[int] | None = None
    guidance_sent: bool | None = None
    if raw:
        # B5: every body carries the main budget only (no pass 2 runs: ReGround is off) and the guidance
        pass2_ids = set(T._reground_attempts(records))
        steps_only = {k: v for k, v in raw.items() if k.split(":", 1)[1] not in pass2_ids}
        budgets_sent, _ = config_evidence(steps_only, LIVE_TOOL_GUIDANCE)
        _, guidance_sent = config_evidence(raw, LIVE_TOOL_GUIDANCE)

    lines = []
    extras: Mapping[str, Any] = {}
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
        if lines:
            extras = lines[0].get("extras") or {}
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
    steps = feedback_step_lines(
        T.step_lines(
            steps,
            visual_tokens=budget,
            reground=reground,
            clicks=watch.clicks if watch is not None else {},
            executed=watch.executed if watch is not None else {},
            records=records,
        ),
        arm=arm,
        records=records,
        bodies=raw,
        calls=feedback_watch.calls if feedback_watch is not None else {},
    )
    regrounds = [
        {**g, "arm": arm, "reground_visual_tokens": None}
        for g in T.reground_lines(
            task_id,
            repeat,
            records=records,
            generates=meter.generates if meter is not None else [],
            visual_tokens=budget,
            reground=reground,
        )
    ]
    answers = T._answers(records)
    pass2 = [answers.get(a, "") for a in T._reground_attempts(records)]
    findings = reground_voiced_findings(
        pass2,
        trial.get("voiced") or [],
        trial.get("history"),
        [str(s.get("step_text") or "") for s in steps],
    )
    trial.update(
        arm=arm,
        visual_tokens=int(budget),
        reground=bool(reground),
        reground_visual_tokens=None,
        live_feedback=_feedback_dict(feedback),
        echo=echo,
        # A5 as the session logged it (``None`` in arm A: no key without live_feedback)
        live_feedback_logged=extras.get("live_feedback"),
        success_v2=score_v2(task_id, trial.get("state"), G._voiced_text(trial)),
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
        {"arm": arm, "task": task_id, "repeat": repeat, **c}
        for c in (watch.clicks.values() if watch is not None else [])
    ]
    thoughts = thought_record(arm, task_id, repeat, records) if records else None
    return trial, steps, regrounds, reads, thoughts


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``ERRORED_STOP`` consecutive errored turns, or past
    ``RUN_BUDGET_S`` since ``PROCESS_START_S`` (probes included), nothing more is sent and nothing more is written
    (B8: fewer than 300 trial lines exactly when ``stopped`` is set)."""
    errored = 0
    last = ""
    for index, (arm, task_id, repeat) in enumerate(order(ORDER_SEED)):
        started = time.monotonic()
        # checked before a turn, never after the last one: ``stopped`` is set only when a turn goes unrun
        if errored >= G.ERRORED_STOP:
            run.stopped = f"{errored} consecutive errored turns (last: {last}; at turn {index})"
        elif started - PROCESS_START_S > RUN_BUDGET_S:
            run.stopped = f"the run budget of {RUN_BUDGET_S} s is spent (at turn {index})"
        if run.stopped is not None:
            run.manifest["stopped"] = run.stopped
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.stopped}")
            return
        steps: list[dict[str, Any]] = []
        regrounds: list[dict[str, Any]] = []
        reads: list[dict[str, Any]] = []
        thoughts: dict[str, Any] | None = None
        try:
            trial, steps, regrounds, reads, thoughts = await _one_turn(run, arm, task_id, repeat)
        except Exception as exc:
            trial = _not_run(arm, task_id, repeat, f"harness: {exc!r}")
        errored = errored + 1 if T._errored(trial, steps, regrounds) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "regrounds.jsonl", regrounds)
        G._append_jsonl(run.out / "clicks.jsonl", reads)
        G._append_jsonl(run.out / "thoughts.jsonl", [thoughts] if thoughts is not None else [])
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} labels={[s.get('label_count') for s in steps]} "
            f"success_v2={trial.get('success_v2')} timed_out={trial.get('timed_out')} "
            f"err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{arm} {task_id}/r{repeat}"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _write_report(
    out: Path, probes: Mapping[str, Any], manifest: Mapping[str, Any], stopped: str | None = None
) -> dict[str, Any]:
    report = build_report(
        G._read_jsonl(out / "trials.jsonl"),
        G._read_jsonl(out / "steps.jsonl"),
        G._read_jsonl(out / "regrounds.jsonl"),
        G._read_jsonl(out / "thoughts.jsonl"),
        probes,
        stopped=stopped,
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def _group_a_missing() -> str | None:
    """Why group A's keywords are not there (``Session(live_feedback=)``, ``BrowserToolExecutor(echo=)``), or
    ``None``. Reads signatures only."""
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.session import Session

    if "live_feedback" not in inspect.signature(Session.__init__).parameters:
        return "Session has no live_feedback keyword (T-0134 A1)"
    if "echo" not in inspect.signature(BrowserToolExecutor.__init__).parameters:
        return "BrowserToolExecutor has no echo keyword (T-0134 A1)"
    return None


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
    out: Path = args.out_dir or REPO / "logs" / f"t0134-feedback-ab-{sha[:7]}"
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
    # 5. the arm table's pin (B1)
    arms_sha = arm_table_sha()
    if arms_sha != ARM_TABLE_SHA256:
        _say(f"refusing: the arm table's sha is {arms_sha}, not the pinned {ARM_TABLE_SHA256}")
        return EXIT_REFUSED
    # 6. group A's keywords, and the guidance every arm sends (A8)
    missing_a = _group_a_missing()
    if missing_a is not None:
        _say(f"refusing: {missing_a}")
        return EXIT_REFUSED
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE

    guidance_sha = hashlib.sha256(LIVE_TOOL_GUIDANCE.encode("utf-8")).hexdigest()
    if guidance_sha != LIVE_TOOL_GUIDANCE_SHA256:
        _say(
            f"refusing: LIVE_TOOL_GUIDANCE's sha256 is {guidance_sha}, not {LIVE_TOOL_GUIDANCE_SHA256} (A8)"
        )
        return EXIT_REFUSED
    # 7. the tool kit (T-0115's three refusals, repair off included)
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    try:
        profile = P.arm_profile(lean, MAIN_BUDGET, False, None)
    except ValueError as exc:
        _say(f"refusing: the arm profile is invalid: {exc}")
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
    # 8. Chromium (T-0111g B2's availability check; reads the filesystem only)
    missing = cli.live_prerequisites(BROWSER)
    if missing is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {missing}")
        return EXIT_REFUSED
    # 9. T-0131's pass-2 bound (kept from T-0132, though ReGround is off in every arm)
    from local_astra.conductor import session as session_module

    if session_module.REGROUND_TIMEOUT_MS != REGROUND_TIMEOUT_MS:
        _say(
            f"refusing: the session's REGROUND_TIMEOUT_MS is {session_module.REGROUND_TIMEOUT_MS}, not "
            f"{REGROUND_TIMEOUT_MS} ms"
        )
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

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
        return _measure(args.base_url, out, sha, model, lean, profile, kit, vram, guidance_sha)
    finally:
        vram.stop()


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts ``subject`` followed by a non-word character is reachable from
    HEAD (the repo's ``Merge T-NNNN: ...`` convention)."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    pattern = re.compile(rf"^{re.escape(subject)}(?!\w)")
    return any(pattern.match(s) for s in subjects)


def _live_default(lean: Any) -> dict[str, Any] | str:
    """B1: ``cli.grounding_of`` at HEAD with no flags, in ``live`` (arm A is meant to be this default)."""
    from local_astra import cli

    got = cli.grounding_of(lean, argparse.Namespace(), "live")
    return dataclasses.asdict(got) if dataclasses.is_dataclass(got) else repr(got)


def _measure(
    base_url: str,
    out: Path,
    sha: str,
    model: str,
    lean: Any,
    profile: Any,
    kit: Any,
    vram: Any,
    guidance_sha: str,
) -> int:
    """Past every refusal and the engine check: the manifest, P1-P3 and P5, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0134_feedback_ab.sh refuses while 127.0.0.1:8080/8081 are bound",
        "arms": arm_rows(),
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
        "live_tool_guidance_sha256": guidance_sha,
        "browser": f"{BROWSER}, one per turn, fresh context, loopback start pages only",
        "viewport": list(profile.browser_viewport),
        "turns": len(order(ORDER_SEED)),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": G.TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "vram_period_ms": T.VRAM_PERIOD_MS,
        "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
        "no audio device, no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        "p4": P4_STANDS,
        "history": list(HISTORY),
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "arm_table_sha256": arm_table_sha(),
        "task_table_sha256": G.task_table_sha(TASKS),
        "live_default_grounding": _live_default(lean),
        "t0133_merged": _merged("Merge T-0133"),
        "t0134_merged": _merged("Merge T-0134"),
        "tasks": [
            {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": REPEATS}
            for t in TASKS
        ],
        "intended_targets": {
            task: [{"role": t.role, "name": t.name, "item": t.item} for t in targets]
            for task, targets in INTENDED_TARGETS.items()
        },
        "cheapest_deal": list(G.cheapest_deal()),
        "s4_price_forms": list(S4_PRICE_FORMS),
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "hostile_page_sha256": hashlib.sha256(HOSTILE_PAGE.encode("utf-8")).hexdigest(),
        "gate_file_sha256": {rel: G._sha256(REPO / rel) for rel in GATE_FILES},
        "order": [list(k) for k in order(ORDER_SEED)],
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in (
        "trials.jsonl",
        "steps.jsonl",
        "regrounds.jsonl",
        "clicks.jsonl",
        "thoughts.jsonl",
        "turns.jsonl",
    ):
        (out / name).touch()

    # --- B2: P1-P3 (T-0131's), then P5 (browser only), all before any arm turn -------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name]["ok"])}
    try:
        probes["p5"] = asyncio.run(probe_p5())
    except Exception as exc:
        probes["p5"] = {"error": f"harness: {exc!r}", "hostile_true": False}
        probes["p5"]["passed"] = p5_ok(probes["p5"])
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(bool(probes[p]["passed"]) for p in ("p1", "p2", "p3", "p5")):
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest, stopped)
        _say(
            f"a probe failed: the run is void ({json.dumps(probes, default=str)}); label "
            f"{report['decision']['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = _Run(
        base_url=base_url,
        out=out,
        profile=profile,
        sentinel=sentinel,
        manifest=manifest,
        vram=vram,
    )
    try:
        asyncio.run(_run_all(run))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    if run.stopped is not None:
        manifest["stopped"] = run.stopped
        G._write_json(out / "manifest.json", manifest)

    # --- C6c: the report straight after the arms, before anything else can fail -------------------------
    report = _write_report(out, probes, manifest, run.stopped)
    _say(f"decision={report['decision']['label']} ({report['decision']['reason']})")
    _say(f"written to {out}")
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
