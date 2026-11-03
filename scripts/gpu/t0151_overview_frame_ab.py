"""T-0151 B -- the locked 2-arm overview-frame A/B of ``tools = live`` on S1-S5: 200 typed turns through a real
``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0151_overview_frame_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the live default (``live_default``), the arm table
(``arm_rows``, ``arm_table_sha``, ``order``), the probe rules (``p6_ok``, ``p7_ok``, ``p8_ok``, ``p9_ok``), B4's
findings (``configuration_findings``), B3's counts (``arm_metrics``, the mechanism counts among them), B5's inputs and
decision (``decision_inputs``, ``decide``, ``decide_with_reason``), the report (``build_report``, ``render_markdown``)
and the record extraction (``body_line``, ``overview_records``, ``frame_steps``) are pure or read ``cli`` only;
everything engine- or browser-facing is imported inside ``main``, the probes and the turn runner.

R-b: this file is a **copy** of T-0150's group B harness with the T-0151 changes applied; T-0150's file is neither
imported nor edited (it is the record of a completed locked run), and neither is T-0140's.

B1: T-0134's harness (``scripts/gpu/t0134_feedback_ab.py``) is **imported, never copied**, and through it T-0132's,
T-0131's and T-0111g's: ``TASKS``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
``non_loopback_sockets``, ``turn_records`` (T-0111g's objects, ``G``), ``probe_p1``, ``run_probes``, ``false_claim``,
``click_score``, ``config_evidence`` (T-0131's, ``T``), ``wilson``, ``newcombe`` (T-0132's, ``P``) and ``score_v2``,
``over_click``, ``no_see``, ``coord_mismatch``, ``false_claim_v2`` (T-0134's, ``F``), plus T-0131's per-arm section,
box-reading watch, record extraction, VRAM sampler and engine helpers, T-0132's arm profile, C6b first-failing rule,
regime reader and P4 frames, and T-0134's feedback watch, client, step lines, thought record and hostile fixture.

Arms (B1; never hard-coded, the T-0135 B pattern): both arms run at HEAD's live default, read from ``cli`` once per
``main`` (``live_default``): the budget and ReGround from ``cli.grounding_of``, the live feedback and the executor echo
from ``cli.feedback_of``, T-0135's recovery from ``cli.recovery_of``, T-0140's page overview from
``cli.page_overview_of``, T-0150's auto overview from ``cli.auto_overview_of`` and T-0151's overview frame from
``cli.overview_frame_of`` (all three must be off: a default with any of them on is refused; so is a default with
ReGround on, which ``Session(overview_frame=True)`` refuses, A2-1). Both arms run the kit
``cli.load_tool_kit(profile, live_browser=...)`` (tools v1, ``bbfc53ccdd80``; never ``page_overview=True``); ``A``
passes ``overview_frame=False`` and ``B`` ``overview_frame=True``; every setting is passed explicitly in both,
``degeneracy_guard=True`` and ``auto_overview=False`` included. The default is recorded in the manifest
(``live_default``), with its two rows (``regime.arms``) and their sha (``arm_table_sha256``).

Refusals (exit 2), all **before any socket** opens and before any browser, page server or ``nvidia-smi``: T-0134's
(a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task table off its pin,
T-0134 group A's keywords absent, ``LIVE_TOOL_GUIDANCE`` off its sha, a tool kit that cannot load / has repair off,
Chromium absent, the session's ``REGROUND_TIMEOUT_MS`` off T-0131's 4000 ms), plus: ``Session`` without the
``recover_thought_calls``, ``degeneracy_guard``, ``auto_overview`` or ``overview_frame`` keyword, a ``cli`` without
``feedback_of``, ``recovery_of``, ``page_overview_of``, ``auto_overview_of`` or ``overview_frame_of``, a live default
that is not well formed, has any overview on or has ReGround on, and the tools check (the kit both arms use not v1).
Past them the engine is checked (``/v1/models`` and ``/metrics``): with no engine the harness exits 1 having launched
no browser.

Probes (B2), before any arm turn: P1-P3 are T-0131's ``run_probes``; P6 (browser only, no inference) is
``probe_p6``, as T-0150's: through a real headless ``BrowserToolExecutor`` at 1280x800, ``page_overview`` on
``/demo/deals.html`` (``screens >= 1.3``, the image within 1280x800, ``scroll_y`` unchanged), on a page that fits
(``fits``) and on T-0134's hostile fixture (its true 2800 px); P7 is ``probe_p7``: a ``/tokenize`` count, as T-0132 P4
measured it, of the **step-1** request of a fresh S4-like session with the overview frame (no history: the system
prompt, the overview in place of the current frame, ``OVERVIEW_FRAME_LABEL`` and the transcript: **1 image**), against
``16384 - 256``, then one generate on it and the engine's health; the overview's own token cost (``overview_tokens``,
the count less the same request's with no image) is report-only; P8 (browser only, no inference) is ``probe_p8``, the
start-page census through the real executor at 1280x800, the session's A2-3 decision mirrored (a live frame, the
capture, the cap, ``overview_frame_of``): exactly S4 and S5 ``shown`` (more than 1.05 and at most 2.0 screens), S1, S2
and S3 ``fits``, none ``too_tall``; P9 (browser only, no inference) is ``probe_p9``: A1-7's two deals.html clicks,
``click_at(160, 933)`` and ``click_at(160, 935)`` on the registered overview frame, each on a fresh page through the
real executor at 1280x800, must each be ``executed`` and leave the cart exactly ``["cotton tea towel"]``. P1 failing:
inconclusive; P2, P3, P6, P7, P8 or P9: void; exit 3, a report, no arm turn.

What it runs (*Regime*): ``order(ORDER_SEED)``, 200 ``(arm, task, repeat)`` turns (2 arms x 5 tasks x ``REPEATS``)
shuffled jointly, concurrency 1, the engine unseeded. Each turn is T-0134's turn (one fresh start-page server, one
fresh headless ``BrowserToolExecutor(echo=...)`` behind T-0131's box-reading watch, T-0134's feedback watch and this
file's frame watch, one fresh ``Session`` with one ``UserText``) at the live default and the arm's overview frame. The
frame watch reads the agent page's ``window.__state`` right after every executed call (``cart_after``, read-only,
bounded by ``CART_READ_S``). The run stops after ``ERRORED_STOP`` consecutive errored turns, when ``RUN_BUDGET_S`` has
passed since ``PROCESS_START_S`` (from process start, probes included), or when the turn runner raised; turns not run
are **not** written, so ``trials.jsonl`` has fewer than 200 lines exactly when ``stopped`` is set. ``report.json`` and
``report.md`` are written straight after the arms, before anything else can fail.

Output (``--out-dir``, default ``logs/t0151-overview-frame-ab-<sha7>/``): ``probes.json`` (P1-P3, P6, P7, P8, P9),
``trials.jsonl`` (T-0134's fields plus ``recover_thought_calls``, ``degeneracy_guard``, ``page_overview`` and
``auto_overview`` (False), ``overview_frame``, ``tools_version``, ``overview_frame_logged`` (T-0151 A2-8's turn-line
key), ``auto_overview_logged`` and ``page_overview_logged`` (never present here) and ``degeneracy_guard_logged``),
``steps.jsonl`` (T-0134's step fields plus ``overview_shown`` (the step's ``llm.request`` record carried A2-8's
``overview_frame`` key), ``frame_kind`` (its call's last ``tool.call`` record's), ``scrolled_px`` (an executed call's
``resolved["scrolled_px"]``) and ``cart_after``), ``regrounds.jsonl``, ``bodies.jsonl`` (one line per turn: every body
the meters forwarded, image parts as sha16, and the turn's overview-frame capture, the executor call whose id ends
``-overview-frame``, that ``overview_frame_of`` registered), ``thoughts.jsonl``, ``clicks.jsonl``, ``turns.jsonl`` and
``sessions/*.jsonl``, ``manifest.json``, ``report.json`` and ``report.md``. Exit codes: 0 the run was measured; 1 no
engine, or the run stopped early; 2 refused; 3 a probe failed (void or inconclusive).

The label (B5, H-T151-1 approved 2026-10-24: T-0150's margins plus the S5 guard): "inconclusive" when P1 failed;
"void" on any configuration finding (B4, reason ``configuration: <finding>``) or when P2, P3, P6, P7, P8 or P9 failed;
"incomplete" when the run stopped early; "inconclusive" when any arm is ineligible (acceptance over the trials that are
neither an engine error nor a ``degenerate`` end, safety, or more than 2 engine errors; the reason names the arm, the
clause and the first failing trial); then "overview frame" iff S4(B) - S4(A) >= 5, success(B) >= success(A) - 3,
non-S4 false claims(B) <= (A) + 2, over-clicks(B) <= (A) + 2 and S5(B) >= S5(A) - 3; else "overview not shown" when
arm B's S4 or S5 turns whose turn line logged ``shown`` = 1 are under 18 (either), else "no change". Engine-error and
``degenerate`` trials count as failed trials in every metric; a 400/413 over-long attempt that the session's retry
recovered is neither an engine error nor an acceptance failure (O-B-8, O-B-8b).

O-B (9): T-0131's click scoring and coordinate-mismatch figures read a call's point against the live screen; they are
not translated for calls on the overview frame, so in arm B they mean nothing for S4/S5 step-1 calls (report-only; the
report says so). Over-clicks, a B5 input, are T-0134's ``over_click`` from the final cart, not click scoring.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native, live, repair on, loop 8, ``LIVE_TOOL_GUIDANCE`` 768a973b; tools v1 ``bbfc53ccdd80`` in both
arms; both arms at HEAD's live default (recorded), the T-0142 guard on; arm B adds ``overview_frame=True``; headless
Chromium at 1280x800, DSF 1, loopback start pages only, one fresh browser per turn; typed turns, fake STT/TTS/audio,
no headset or speakers; 2 arms x 5 tasks x n = 20 = 200 turns, order seed 0, concurrency 1, 60 s turn timeout, the
engine unseeded; STT down. History, not like-for-like: T-0150 at 133f0aa (arm A S4 0/20, S5 13/20, success 66/100;
arm B S4 20/20, S5 0/20).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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

#: B6: the run budget is measured from process start (probes included): ``time.monotonic()`` at import
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0134's harness, read-only (B1: imported, never copied); it imports T-0132's, T-0131's and T-0111g's
T0134_PATH: Path = REPO / "scripts" / "gpu" / "t0134_feedback_ab.py"
#: the fixed ``sys.modules`` name T-0134 is loaded under (its dataclasses look themselves up there)
T0134_MODULE_NAME = "_t0134_feedback_ab_for_t0151"


def _load_t0134() -> ModuleType:
    """T-0134's harness, imported read-only by path under ``T0134_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0134_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0134_MODULE_NAME, T0134_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0134_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0134_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0134_MODULE_NAME, None)
        raise
    return module


F: ModuleType = _load_t0134()
#: T-0132's, T-0131's and T-0111g's harnesses, as T-0134 loaded them (the same module objects)
P: ModuleType = F.P
T: ModuleType = F.T
G: ModuleType = F.G

# --- imported from T-0111g, unchanged (the same objects) ----------------------------------------------------
TASKS = G.TASKS
LoopMeter = G.LoopMeter
sentinel_hits = G.sentinel_hits
url_leaks = G.url_leaks
log_complete = G.log_complete
non_loopback_sockets = G.non_loopback_sockets
turn_records = G.turn_records
TASK_TABLE_SHA256 = G.TASK_TABLE_SHA256
# --- imported from T-0131, unchanged --------------------------------------------------------------------------
INTENDED_TARGETS = T.INTENDED_TARGETS
false_claim = T.false_claim
click_score = T.click_score
reground_voiced_findings = T.reground_voiced_findings
probe_p1 = T.probe_p1
run_probes = T.run_probes
config_evidence = T.config_evidence
BROWSER = T.BROWSER
REGROUND_TIMEOUT_MS = T.REGROUND_TIMEOUT_MS
# --- imported from T-0132, unchanged --------------------------------------------------------------------------
wilson = P.wilson
newcombe = P.newcombe
WINDOW_LIMIT = P.WINDOW_LIMIT
# --- imported from T-0134, unchanged --------------------------------------------------------------------------
score_v2 = F.score_v2
over_click = F.over_click
no_see = F.no_see
coord_mismatch = F.coord_mismatch
false_claim_v2 = F.false_claim_v2
LIVE_TOOL_GUIDANCE_SHA256 = F.LIVE_TOOL_GUIDANCE_SHA256
VIEWPORT = F.VIEWPORT
HOSTILE_PAGE = F.HOSTILE_PAGE
HOSTILE_PATH = F.HOSTILE_PATH
HOSTILE_HEIGHT_PX = F.HOSTILE_HEIGHT_PX
#: the tree is read through this name (T-0111g's seam); ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: B1: the arm labels, in the order the eligibility step checks them
LABELS = ("A", "B")
#: B1: what each arm sets ``overview_frame`` to (the only difference between the arms)
OVERVIEW_FRAME_ON = {"A": False, "B": True}
#: B1: ``tools_v1.json``, the tool set of both arms, a full literal (never derived); ``tools_v1_overview.json``,
#: which no arm uses (its file is still pinned in the manifest's gate files)
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
TOOLS_SHA256 = {"A": V1_SHA256, "B": V1_SHA256}
#: B1: the live-default settings both arms carry, read from ``cli`` at run time (never hard-coded)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo", "recover_thought_calls")
#: B1: the three overview settings, off in HEAD's live default
OVERVIEW_KEYS = ("page_overview", "auto_overview", "overview_frame")
#: B1: per (arm, task); 2 arms x 5 tasks x 20 = 200 turns
REPEATS = 20
#: *Regime*: the 200 turns shuffled jointly with seed 0
ORDER_SEED = 0
#: B3's fixed denominators (the differences use them whatever ran)
TURNS_PER_ARM = 100
S4_TURNS = 20
S5_TURNS = 20
#: P8 / docs/tasks/T-0151.md "Measured scope": the tasks whose start page fits (no overview) and those on deals.html
FIT_TASKS = ("S1", "S2", "S3")
SHOWN_TASKS = ("S4", "S5")
#: B6: the wall-time budget, measured from ``PROCESS_START_S`` (probes included), inside the locked script's
#: ``timeout -k 60s 75m`` so the report is still written (estimate about 30 min: T-0140 ran 200 turns in 27 min)
RUN_BUDGET_S = 62 * 60
#: B5 (H-T151-1, approved 2026-10-24: T-0150's margins plus the S5 guard): boundaries compare with this tolerance (an
#: exact margin passes)
EPS = 1e-9
S4_MARGIN = 5
SUCCESS_MARGIN = 3
FALSE_CLAIM_MARGIN = 2
OVER_CLICK_MARGIN = 2
S5_MARGIN = 3
#: B5: an arm is ineligible on the engine clause only above this many engine-error trials
ENGINE_ERRORS_MAX = 2
#: B5: ``overview not shown`` when arm B's S4 or S5 turns whose turn line logged ``shown`` = 1 are fewer than this
NOT_SHOWN_BELOW = 18
#: O-B-8: the over-long statuses the session's retry recovers from (T-0020)
OVER_LONG = (400, 413)
#: B5's labels
OVERVIEW_FRAME = "overview frame"
NOT_SHOWN = "overview not shown"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
#: C6b: the eligibility clauses, in the order they are checked
CLAUSES = ("acceptance", "safety", "engine")
TASK_IDS = tuple(t.task_id for t in TASKS)
#: the budgets a live default may name (T-0131's arms)
BUDGETS = (280, 560, 1120)
ECHOES = ("pixels", "model")
#: B4: the budget every body carries, and at most this many images per request (``--limit-mm-per-prompt image=4``)
MAIN_BUDGET = F.MAIN_BUDGET
MAX_IMAGES = 4
#: T-0140 A3's constants and strings (the executor's), typed from docs/tasks/T-0140.md
MAX_OVERVIEW_SCREENS = 4
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
OVERVIEW_FITS_RATIO = 1.05
#: T-0140 A7's label stem (T-0150's auto overview reuses it): no body of this A/B may carry it
OVERVIEW_STEM = "Whole page, scaled to fit one screen"
#: T-0151 A2-3 (H-T151-4) and A2-4 (H-T151-5), typed from docs/tasks/T-0151.md (B4 judges the bodies against these)
OVERVIEW_FRAME_MAX_SCREENS = 2.0
OVERVIEW_FRAME_LABEL = (
    "[This screenshot is the whole web page, about {screens:.1f} screens tall, scaled down to fit; the browser "
    "screen shows only part of it. click_at and type_at coordinates refer to this image, and the browser scrolls to "
    "the point before it clicks.]"
)
OVERVIEW_FRAME_STEM = "This screenshot is the whole web page"
#: T-0134's label stem (``LABEL_NOW`` / ``LABEL_UPDATED`` both open with it): never on the overview-frame body
SCREEN_LABEL_STEM = "This screenshot is the browser screen now"
#: T-0151 A2-8's counts on the turn line
A2_KEYS = ("shown", "fits", "too_tall", "failed", "refused")
#: T-0151 A2-2: the call id suffix of the session's own capture (``{turn_id}-overview-frame``)
FRAME_CALL_SUFFIX = "-overview-frame"
#: T-0111g B9's and T-0151 A1-4's reasons, typed from the tasks
FRESHNESS_REASON = "the screen changed since this frame; take a new frame"
OVERVIEW_NOT_REACHED = "the page did not scroll to the point; nothing was clicked"
#: B3 (report-only, T-0140's reading): a voiced text that mentions the overview
NARRATION = re.compile(r"\boverview\b|\bwhole page\b", re.IGNORECASE)
#: B3: the S4 distractor T-0134/T-0140 thoughts pick (the cheapest of the six deals visible at 1280x800)
MUG = "enamel mug"
#: B2 P6
P6_PAGE = "/demo/deals.html"
P6_MIN_SCREENS = 1.3
#: T-0134 P5's tolerance (1 CSS px) on the hostile fixture's true height
P6_TOLERANCE_PX = 1.0
P6_CALL_S = 30.0
#: P6's page that fits: one fixed 400 CSS px block, served with T-0134's hostile page from this file's own loopback
#: server (no script, no ``vh`` unit)
FIT_PATH = "/t0151_fits.html"
FIT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>fits</title>
<style>html, body { margin: 0; padding: 0; } .block { height: 400px; background: #dfe7ef; }</style>
</head><body><div class="block"></div></body></html>
"""
#: B2 P7: the overview frame's size in the probe's request (O-A1-11: deals' overview is 838x798), its screens, and the
#: step-1 shape's image count (the overview in place of the current frame, nothing else: a fresh session, no history)
P7_OVERVIEW_SIZE = (838, 798)
P7_SCREENS = 1.525
P7_IMAGES = 1
P7_REQUEST_ID = "t0151-probe-p7"
#: B2 P9 (A1-7): the two deals.html clicks in overview units, each on a fresh page, and the item each must add
P9_PAGE = "/demo/deals.html"
P9_POINTS = ((160, 933), (160, 935))
P9_ITEM = "cotton tea towel"
#: the frame watch's bound on its read of ``window.__state`` after an executed call (inside the session's 10 s
#: ``TOOL_EXEC_TIMEOUT_MS``, next to the executor's 8 s action bound; the read is a single ``page.evaluate``)
CART_READ_S = 1.0

#: STT down, enforced by the locked script's port refusal
STT_DOWN = P.STT_DOWN
#: the gate files whose sha256 pins a run to its commit (T-0132's, plus the overview set, the judge, T-0135's rule and
#: T-0142's guard)
GATE_FILES = (
    *P.GATE_FILES,
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/thought_call.py",
    "local_astra/conductor/degeneracy.py",
)
#: the runs this A/B is set beside (history, not like-for-like)
HISTORY = (
    "T-0150 auto-overview A/B at 133f0aa (n = 100 per arm, order seed 0): arm A S4 0/20, S5 13/20, success 66/100 by "
    "score_v2; arm B (the overview beside the live frame at step 1) S4 20/20 but S5 0/20 -- its step-1 clicks were "
    "at the tea towel's overview y (about 935 model units) read against the live screen; result no change",
    "T-0140 page-overview A/B at 8697960 (n = 100 per arm, order seed 0): arm A 71/100 by score_v2; S4 0/20 in both "
    "arms; the page_overview tool was never called in 100 arm-B turns (overview unused)",
)
#: B5, stated in the report (H-T151-1; the exact figures are the eval oracle's enumeration)
S5_GUARD = (
    "the S5 guard S5(B) >= S5(A) - 3/20 rejects a true null about 12 % of the time at p ~ 0.65 (computed exactly "
    "from two independent Bin(20, p): 0.1226); a guard at -4 would reject about 7 % (0.0672)."
)
POWER = (
    "S4: 5/20 vs 0/20 is Fisher p ~ 0.047 (two-sided, exact); at n = 100 and p ~ 0.7 the standard error of success is "
    "about 4.6 pp; the false-claim and over-click margins of +2 are non-inferiority guards."
)
#: B5, stated in the report: which guards can move, the S5 history, the token cost, the click-scoring caveat
GUARDS = (
    "S1-S3 receive no overview frame (P8: their start pages fit), so their share of the guards is noise by "
    "construction; S5 is the task at risk and has its own guard (H-T151-1)."
)
S5_HISTORY = (
    "T-0150 at 133f0aa: arm A S5 13/20, arm B 0/20 (the overview beside the live frame, step-1 clicks read in the "
    "wrong coordinate system); this A/B's arm B reads them in the overview's."
)
TOKEN_COST = (
    "the overview replaces the live frame in the step-1 request (about 1.1k tokens, T-0150 P7: 1,116), so step 1 "
    "re-prefills from the image; P7's overview_tokens (the request less the same request with no image) is this run's."
)
CLICK_SCORING = (
    "O-B (9): T-0131's click scoring and coordinate-mismatch figures read a call's point against the live screen and "
    "are not translated for calls on the overview frame, so in arm B they are not meaningful for S4/S5 step-1 calls "
    "(report-only). Over-clicks (a B5 input) are T-0134's over_click, judged from the final cart on S1, S2 and S5, not "
    "from click scoring, so the decision is unaffected."
)


def _say(msg: str) -> None:
    print(f"t0151_overview_frame_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The live default, the arm table and the order (B1)
# =====================================================================================================

#: T-0135 O-16: ``cli.Feedback.live_feedback``'s choices as (screen_label, page_extent); ``off`` is both parts off,
#: never ``None`` (T-0134c R-m)
FEEDBACK_CHOICES: dict[str, tuple[bool, bool]] = {
    "label": (True, False),
    "label+extent": (True, True),
    "off": (False, False),
}


def live_default(profile: Any) -> dict[str, Any]:
    """B1: HEAD's live default as ``cli`` gives it for ``Namespace()`` in ``live``, read at call time:
    ``grounding_of``'s budget and ReGround, ``feedback_of``'s live feedback (``FEEDBACK_CHOICES``) and action echo,
    ``recovery_of``'s ``thought_call_recovery``, ``page_overview_of``'s ``page_overview``, ``auto_overview_of``'s
    ``auto_overview`` and ``overview_frame_of``'s ``overview_frame``."""
    from local_astra import cli

    ns = argparse.Namespace()
    g = cli.grounding_of(profile, ns, "live")
    fb = cli.feedback_of(profile, ns, "live")
    rec = cli.recovery_of(profile, ns, "live")
    ov = cli.page_overview_of(profile, ns, "live")
    auto = cli.auto_overview_of(profile, ns, "live")
    frame = cli.overview_frame_of(profile, ns, "live")
    label, extent = FEEDBACK_CHOICES[str(fb.live_feedback)]
    return {
        "visual_tokens": None if g.visual_tokens is None else int(g.visual_tokens),
        "reground": bool(g.reground),
        "live_feedback": {"screen_label": label, "page_extent": extent},
        "echo": str(fb.action_echo),
        "recover_thought_calls": rec.thought_call_recovery,
        "page_overview": ov.page_overview,
        "auto_overview": auto.auto_overview,
        "overview_frame": frame.overview_frame,
    }


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """Why a live default is not one this harness can run (each key present, each value of a type the session takes,
    the page overview, the auto overview and the overview frame all off), or ``[]``."""
    keys = (*DEFAULT_KEYS, *OVERVIEW_KEYS)
    out = [f"live_default lacks {k}" for k in keys if k not in default]
    if out:
        return out
    if default["visual_tokens"] not in BUDGETS:
        out.append(f"visual_tokens {default['visual_tokens']!r}")
    if not isinstance(default["reground"], bool):
        out.append(f"reground {default['reground']!r}")
    lf = default["live_feedback"]
    if lf is not None and not (
        isinstance(lf, Mapping)
        and set(lf) == {"screen_label", "page_extent"}
        and all(isinstance(v, bool) for v in lf.values())
    ):
        out.append(f"live_feedback {lf!r}")
    if default["echo"] not in ECHOES:
        out.append(f"echo {default['echo']!r}")
    if not isinstance(default["recover_thought_calls"], bool):
        out.append(f"recover_thought_calls {default['recover_thought_calls']!r}")
    for k in OVERVIEW_KEYS:
        if default[k] is not False:
            out.append(f"{k} {default[k]!r}: on by default (B1 runs HEAD's default with it off)")
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: the two arms over a live default: its five settings, the T-0142 guard on, the page and auto overviews off
    and tools v1 in both; A without the overview frame, B with it."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {
            "label": label,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": OVERVIEW_FRAME_ON[label],
            "tools_sha256": TOOLS_SHA256[label],
        }
        for label in LABELS
    ]


def arm_table_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of the canonical JSON of the arm rows (``sort_keys``, compact separators)."""
    blob = json.dumps([dict(r) for r in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def order(seed: int) -> list[tuple[str, str, int]]:
    """Every ``(arm, task_id, repeat)`` of ``LABELS`` x ``TASKS`` x ``REPEATS``, arm-major, then shuffled once by
    ``random.Random(seed)`` -- jointly, so the arms are interleaved."""
    turns = [(a, t.task_id, k) for a in LABELS for t in TASKS for k in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# T-0140 A3's and T-0151 A2-4's strings (pure)
# =====================================================================================================


def overview_action(screens: float, clipped: bool) -> str:
    """T-0140 A3's executed action string."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def overview_frame_label(screens: float) -> str:
    """T-0151 A2-4's ``OVERVIEW_FRAME_LABEL`` of ``screens``."""
    return OVERVIEW_FRAME_LABEL.format(screens=screens)


# =====================================================================================================
# B2: the P6, P7, P8 and P9 rules (pure)
# =====================================================================================================


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def p6_ok(p6: Mapping[str, Any]) -> bool:
    """B2 P6 from its record (as T-0150's): at 1280x800, ``deals`` showed an overview of at least 1.3 screens within
    the viewport with A3's action string and ``scroll_y`` unchanged; ``fits`` gave A3's fits string; ``hostile``
    reported T-0134's fixture's true height (2800 CSS px, within 1 px), not clipped, with A3's string. A missing part
    fails."""
    try:
        if [int(x) for x in p6["viewport"]] != list(VIEWPORT):
            return False
        vw, vh = VIEWPORT
        d, f, h = p6["deals"], p6["fits"], p6["hostile"]
        ds, hs = float(d["screens"]), float(h["screens"])
        deals = (
            d["outcome"] == "shown"
            and ds >= P6_MIN_SCREENS - EPS
            and 0 < int(d["width"]) <= vw
            and 0 < int(d["height"]) <= vh
            and float(d["scroll_y_after"]) == float(d["scroll_y_before"])
            and bool(d["clipped"]) is (ds > MAX_OVERVIEW_SCREENS)
            and d["action"] == overview_action(ds, bool(d["clipped"]))
        )
        fits = f["outcome"] == "fits" and f["action"] == OVERVIEW_FITS
        hostile = (
            h["outcome"] == "shown"
            and abs(hs * vh - HOSTILE_HEIGHT_PX) <= P6_TOLERANCE_PX + EPS
            and h["clipped"] is False
            and h["action"] == overview_action(hs, False)
        )
        return bool(deals and fits and hostile)
    except (KeyError, TypeError, ValueError):
        return False


def p7_ok(p7: Mapping[str, Any]) -> bool:
    """B2 P7 from its record (T-0132 P4's rule): a ``/tokenize`` count within ``WINDOW_LIMIT`` (16128) of an S4-like
    *step-1* request with the overview frame (a fresh session: exactly ``P7_IMAGES`` = 1 image, the overview, which
    replaces the live frame), the generate on it answering 200 and the engine healthy after. ``overview_tokens`` is
    report-only."""
    tokens = p7.get("tokens")
    images = p7.get("images")
    return (
        _int(tokens)
        and p7.get("limit") == WINDOW_LIMIT
        and tokens <= WINDOW_LIMIT
        and p7.get("http_status") == 200
        and p7.get("healthy_after") is True
        and p7.get("overview_images") == 1
        and _int(images)
        and images == P7_IMAGES
    )


def p8_ok(p8: Mapping[str, Any]) -> bool:
    """B2 P8 from its record (the start-page census through the real executor, browser only): at 1280x800, exactly S4
    and S5 ``shown`` (more than 1.05 and at most ``OVERVIEW_FRAME_MAX_SCREENS`` screens), S1, S2 and S3 ``fits`` --
    every task present, none other, none ``too_tall``. A missing part fails, never raises."""
    try:
        if [int(x) for x in p8["viewport"]] != list(VIEWPORT):
            return False
        tasks = p8["tasks"]
        if set(tasks) != set(TASK_IDS):
            return False
        for t in TASK_IDS:
            outcome = tasks[t]["outcome"]
            if t in FIT_TASKS and outcome != "fits":
                return False
            if t in SHOWN_TASKS:
                screens = float(tasks[t]["screens"])
                if not (
                    outcome == "shown"
                    and screens > OVERVIEW_FITS_RATIO
                    and screens <= OVERVIEW_FRAME_MAX_SCREENS + EPS
                ):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def p9_ok(p9: Mapping[str, Any]) -> bool:
    """B2 P9 from its record (browser only, no inference; A1-7): at 1280x800, the two deals.html clicks on the
    registered overview frame, ``click_at(160, 933)`` and ``click_at(160, 935)``, each on a fresh page through the real
    executor, are each ``executed`` and leave the cart holding exactly ``["cotton tea towel"]``. The action string and
    ``scroll_y_after`` are report-only (O-B (2)). A missing part fails, never raises."""
    try:
        if [int(x) for x in p9["viewport"]] != list(VIEWPORT):
            return False
        clicks = list(p9["clicks"])
        if [tuple(int(v) for v in c["point"]) for c in clicks] != [tuple(p) for p in P9_POINTS]:
            return False
        return all(c["status"] == "executed" and list(c["cart"]) == [P9_ITEM] for c in clicks)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


#: B2: every probe; any failed makes the run void (P1 also labels it inconclusive)
PROBES = ("p1", "p2", "p3", "p6", "p7", "p8", "p9")


def _passed(probes: Mapping[str, Any], name: str) -> bool:
    return bool((probes.get(name) or {}).get("passed"))


# =====================================================================================================
# B4: configuration findings (pure)
# =====================================================================================================


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(record["arm"]), str(record["task"]), int(record["repeat"]))


def _tid(record: Mapping[str, Any]) -> str:
    """``"<arm> <task>#<repeat>"``: how a finding names a trial."""
    return f"{record['arm']} {record['task']}#{record['repeat']}"


def _is_step1_body(request: Mapping[str, Any]) -> bool:
    """A2-4: the step-1 request as sent -- the ``generate`` (and its over-long retry) or the repair of step 1's
    call."""
    return request.get("phase") in ("generate", "tool_repair") and request.get("step") == 1


def _overview_placed(
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    last_user: int | None,
    i: int,
    j: int,
    overview: Mapping[str, Any],
) -> bool:
    """B4/A2-4: the overview-frame image at ``messages[i].parts[j]`` is where ``overview`` (one record, the turn's
    step-1 capture) puts it: in the final user message of a step-1 body (the ``generate``, its over-long retry, or the
    ``tool_repair`` of the step-1 call, which continues the pass-1 prompt), as that message's **only** image,
    immediately followed by the exact label of its screens. Matched on (sha, step), never on the sha alone."""
    if not _is_step1_body(request) or int(overview["step"]) != 1:
        return False
    if i != last_user:
        return False
    parts = list(messages[i].get("parts") or [])
    images = [k for k, p in enumerate(parts) if "image" in p]
    want = overview_frame_label(float(overview["screens"]))
    return images == [j] and j + 1 < len(parts) and parts[j + 1] == {"text": want}


def _counts_all_zero(logged: Any) -> bool:
    return (
        isinstance(logged, Mapping)
        and set(logged) == set(A2_KEYS)
        and all(logged[k] == 0 for k in A2_KEYS)
    )


def _a2_well_formed(logged: Any) -> bool:
    """A2-8's turn-line key: exactly the five counts, each a non-negative int (never a bool), summing to at most 1
    (O-B (6))."""
    return (
        isinstance(logged, Mapping)
        and set(logged) == set(A2_KEYS)
        and all(_int(v) and v >= 0 for v in logged.values())
        and sum(logged.values()) <= 1
    )


def _body_findings(trial: Mapping[str, Any], line: Mapping[str, Any]) -> list[str]:
    """B4 over one turn's bodies: any body declaring ``page_overview`` (no arm offers it), an arm-A turn with an
    overview frame, the image count, the overview frame's count and place (A2-4: alone in the final user message of a
    step-1 body, followed by its exact label), the overview-frame label and T-0140/T-0150's overview label in user
    messages only (O-17: the model's own words reach later bodies as assistant text), a T-0134 label on the
    overview-frame message, O-B-2's agreement (tightened by A2-4) of the turn line's ``shown`` with **every** step-1
    body in both directions, and the unlogged absence on S4/S5."""
    out: list[str] = []
    tid, arm, task = _tid(trial), str(trial["arm"]), str(trial["task"])
    # T-0140 review F1: a sha maps to every record carrying it
    overviews: dict[str, list[Mapping[str, Any]]] = {}
    for o in line.get("overviews") or []:
        overviews.setdefault(str(o["sha"]), []).append(o)
    if arm == "A" and overviews:
        out.append(f"{tid}: an arm-A turn registered an overview frame")
    step1_carries: list[bool] = []  # per step-1 body: carries the overview frame, in place
    step1_any: list[bool] = []  # per step-1 body: carries an overview-frame image anywhere
    generate_frames: list[bool] = []  # per generate body: a current frame and no overview image
    for r in line.get("requests") or []:
        where = f"{tid} {r.get('label')}"
        tools = r.get("tools")
        if tools is not None and "page_overview" in tools:
            out.append(f"{where}: a body declares page_overview")
        messages = list(r.get("messages") or [])
        images = [
            (i, j, str(p["image"]))
            for i, m in enumerate(messages)
            for j, p in enumerate(m.get("parts") or [])
            if "image" in p
        ]
        if len(images) > MAX_IMAGES:
            out.append(f"{where}: {len(images)} images")
        mine = [x for x in images if x[2] in overviews]
        if len(mine) > 1:
            out.append(f"{where}: {len(mine)} overview-frame images")
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user = users[-1] if users else None
        placed: set[tuple[int, int]] = set()
        for i, j, sha in mine:
            if arm == "A":
                out.append(f"{where}: an arm-A body carries an overview-frame image")
            elif any(_overview_placed(r, messages, last_user, i, j, o) for o in overviews[sha]):
                placed.add((i, j + 1))
            else:
                out.append(f"{where}: an overview-frame image out of place")
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue  # O-17: spoken words reach later bodies as assistant text
            for j, p in enumerate(m.get("parts") or []):
                text = p.get("text")
                if not isinstance(text, str):
                    continue
                if OVERVIEW_FRAME_STEM in text and (i, j) not in placed:
                    out.append(f"{where}: an overview-frame label out of place")
                if OVERVIEW_STEM in text:
                    out.append(f"{where}: an auto/page overview label in a body")
                if SCREEN_LABEL_STEM in text and placed and i == last_user:
                    out.append(f"{where}: a T-0134 label on the overview-frame body")
        if _is_step1_body(r):
            step1_carries.append(bool(placed))
            step1_any.append(bool(mine))
        if r.get("phase") == "generate" and last_user is not None:
            frame = any("image" in p for p in messages[last_user].get("parts") or [])
            generate_frames.append(frame and not mine)
    logged = trial.get("overview_frame_logged")
    if arm == "B" and not _wrote_no_turn_line(trial) and _a2_well_formed(logged):
        assert isinstance(logged, Mapping)
        # O-B-2 tightened by A2-4 (O-B (4)): the turn line's ``shown`` and every step-1 body agree
        if logged["shown"] == 0 and any(step1_any):
            out.append(
                f"{tid}: a step-1 body carries the overview frame while the turn line's shown is 0"
            )
        if logged["shown"] >= 1 and not (step1_carries and all(step1_carries)):
            out.append(
                f"{tid}: the turn line's shown is {logged['shown']} but a step-1 body does not carry the overview "
                "frame alone with its label"
            )
    if (
        arm == "B"
        and task in SHOWN_TASKS
        and not _wrote_no_turn_line(trial)
        and _counts_all_zero(logged)
        and any(generate_frames)
    ):
        out.append(
            f"{tid}: an unlogged absence (step 1 carried a frame but no overview, the overview_frame counts all 0)"
        )
    return out


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    bodies: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
) -> list[str]:
    """B4, one string per finding, any of which voids the run, each naming the trial (``"<arm> <task>#<repeat>"``)
    or the setting: the arm rows differing in a setting other than the overview frame, or off B1 (A without the
    overview frame, B with it, tools v1, the page and auto overviews off and the guard on in both); any reground line
    (ReGround is off in both arms); and in a trial that forwarded a body (O-11, as T-0134) a non-1120 body, its
    feedback, echo, recovery, guard, overview frame, auto or page overview off its arm's row, an overview frame on an
    S1-S3 trial (P8: its start page fits; an overview record or a ``shown`` count), a turn-line key off its row
    (``logged_setting_findings``), or a body finding (``_body_findings``). ``steps`` is not read (B4 is about what was
    sent)."""
    del steps
    out: list[str] = []
    rows = [dict(r) for r in arms]
    for k in sorted({k for r in rows for k in r} - {"label", "overview_frame"}):
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    for r in rows:
        label = str(r.get("label"))
        if label in OVERVIEW_FRAME_ON and r.get("overview_frame") is not OVERVIEW_FRAME_ON[label]:
            out.append(f"arm {label} row: overview_frame {r.get('overview_frame')!r}")
        if r.get("tools_sha256") != V1_SHA256:
            out.append(f"arm {label} row: tools {r.get('tools_sha256')!r}")
        for k in ("page_overview", "auto_overview"):
            if r.get(k) is not False:
                out.append(f"arm {label} row: {k} {r.get(k)!r}")
        if r.get("degeneracy_guard") is not True:
            out.append(f"arm {label} row: degeneracy_guard {r.get('degeneracy_guard')!r}")
    by_row = {str(r.get("label")): r for r in rows}
    for g in regrounds:
        out.append(f"{_tid(g)}: a reground line")
    by_bodies = {_key(b): b for b in bodies}
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        tid = _tid(t)
        row = by_row.get(str(t["arm"])) or {}
        bad = [b for b in t.get("budgets_sent") or [] if b != MAIN_BUDGET]
        if bad or not t.get("budgets_sent"):
            out.append(f"{tid}: body budget {bad or 'none'}")
        for k in (
            "live_feedback",
            "echo",
            "recover_thought_calls",
            "degeneracy_guard",
            "overview_frame",
            "auto_overview",
            "page_overview",
        ):
            if t.get(k) != row.get(k):
                out.append(f"{tid}: {k} {t.get(k)!r}, its row says {row.get(k)!r}")
        line = by_bodies.get(_key(t)) or {}
        logged = t.get("overview_frame_logged")
        shown = int(logged.get("shown") or 0) if isinstance(logged, Mapping) else 0
        if str(t["task"]) in FIT_TASKS and (line.get("overviews") or shown > 0):
            out.append(f"{tid}: an overview frame on an S1-S3 trial (P8: its start page fits)")
        out += logged_setting_findings(t, row)
        out += _body_findings(t, line)
    return out


def _wrote_no_turn_line(trial: Mapping[str, Any]) -> bool:
    """T-0135b A7: a trial that hit the turn timeout or never had a session wrote no turn line."""
    return trial.get("timed_out") is True or trial.get("session_id") is None


def logged_setting_findings(trial: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """B4 "feedback, echo, recovery or guard settings off their rows", "an arm-A ``overview_frame`` key, or an arm-B
    line without one" and "any ``auto_overview`` or ``page_overview`` key", judged on the turn line as the session
    logged it (T-0140 review F3; the trial's own setting fields are copies of the row): ``live_feedback_logged`` is the
    row's feedback plus its echo (``None`` when the row has none), ``recover_thought_calls_logged`` and
    ``visual_tokens_logged`` the row's, the guard logged on, no A2-8 key in arm A and A2-8's five well-formed counts in
    arm B, and no ``auto_overview``/``page_overview`` key in either. A trial that wrote no turn line is not judged."""
    if _wrote_no_turn_line(trial):
        return []
    tid = _tid(trial)
    lf = row.get("live_feedback")
    want = {
        "live_feedback_logged": None if lf is None else {**dict(lf), "echo": row.get("echo")},
        "recover_thought_calls_logged": row.get("recover_thought_calls"),
        "visual_tokens_logged": row.get("visual_tokens"),
    }
    out = [
        f"{tid}: {k} {trial.get(k)!r}, its row says {v!r}"
        for k, v in want.items()
        if trial.get(k) != v
    ]
    guard = trial.get("degeneracy_guard_logged")
    if not (isinstance(guard, Mapping) and guard.get("on") is True):
        out.append(f"{tid}: degeneracy_guard_logged {guard!r}")
    of = trial.get("overview_frame_logged")
    on = OVERVIEW_FRAME_ON.get(str(trial["arm"]))
    if on is False and of is not None:
        out.append(f"{tid}: overview_frame_logged {of!r} on an arm-A turn line")
    if on is True and not _a2_well_formed(of):
        out.append(f"{tid}: overview_frame_logged {of!r} on an arm-B turn line")
    for k in ("auto_overview_logged", "page_overview_logged"):
        if trial.get(k) is not None:
            out.append(f"{tid}: {k} {trial.get(k)!r} on a turn line")
    return out


# =====================================================================================================
# B3: the per-arm counts (pure)
# =====================================================================================================


def _turn(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def _ci(k: int, n: int) -> list[float] | None:
    """Wilson 95 % of ``k`` of ``n``; ``None`` when nothing ran."""
    return list(wilson(k, n)) if n else None


def _attempts(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A step's requests in order: its earlier attempts (``retries``), then its own last attempt."""
    return [*(step.get("retries") or []), step]


def is_degenerate(trial: Mapping[str, Any]) -> bool:
    """T-0142: the turn ended on the guard's own abort."""
    return trial.get("tool_loop_end") == "degenerate"


def failed_attempts(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A step's attempts (retries, then its own) that did not answer 200, except a 400/413 over-long attempt that a
    later attempt of the same step recovered with a 200 (O-B-8: an over-long prompt, not an engine error)."""
    atts = _attempts(step)
    return [
        a
        for k, a in enumerate(atts)
        if a.get("http_status") != 200
        and not (
            a.get("http_status") in OVER_LONG
            and any(b.get("http_status") == 200 for b in atts[k + 1 :])
        )
    ]


def engine_error(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5 (B-1 kept, O-B-8): a trial that timed out, whose engine was down after it, or one of whose step attempts
    (retries included) did not answer HTTP 200 (5xx, a connection failure, HTTP 0 from an abort other than the
    guard's) -- never a ``degenerate`` end, even when its aborted attempt shows HTTP 0, and never a recovered
    over-long retry."""
    if is_degenerate(trial):
        return False
    if trial.get("timed_out") is True or trial.get("engine_up_after") is False:
        return True
    return any(failed_attempts(s) for s in steps)


def accepted(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """T-0111g's per-trial acceptance (``G._turn_ok``) with O-B-8b's reading: no exception, not timed out, a complete
    session log, at least one step, and no failed attempt -- a 400/413 the session's retry recovered is not one."""
    if trial.get("exception") or trial.get("timed_out") or trial.get("log_complete") is False:
        return False
    return bool(steps) and not any(failed_attempts(s) for s in steps)


def forced_failure(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5 (B-2): an engine-error or ``degenerate`` trial counts as a failed trial in every metric."""
    return is_degenerate(trial) or engine_error(trial, steps)


def _false_claim(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], ok: bool) -> bool:
    """T-0131's false-claim proxy with ``ok`` as its success test (``score_v2``, False on a forced failure): equal to
    T-0134's ``false_claim_v2`` when the trial is not a forced failure."""
    if not ok and not forced_failure(trial, steps):
        return bool(false_claim_v2(trial, steps))
    if trial.get("exception") or trial.get("timed_out") or not steps or ok:
        return False
    last = max(steps, key=lambda s: int(s["step"]))
    if last.get("status") is not None or last.get("call_name") is not None:
        return False
    said = " ".join(str(v) for v in last.get("voiced_sources") or [])
    return bool(said.strip()) and "?" not in said


def _on_overview(step: Mapping[str, Any]) -> bool:
    """B3: a step-1 call read off the overview frame (A2-8's ``frame_kind`` on its step line)."""
    return (
        int(step["step"]) == 1
        and step.get("frame_kind") == "overview"
        and step.get("call_name") is not None
    )


def mechanism_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B3's new mechanism counts on S4 and S5 (O-B (1), (10): computed in both arms, arm A's zero by construction):
    step-1 calls on the overview frame by name, their statuses, those that scrolled (executed with a non-zero
    ``scrolled_px``), the S5 turns whose step-1 ``cart_after`` holds the tea towel, step-1 ``scroll`` calls, B9
    refusals of overview calls (``FRESHNESS_REASON``) and ``OVERVIEW_NOT_REACHED`` failures."""
    cheapest = G.cheapest_deal()[0].casefold()
    turns = {_turn(t) for t in trials}
    mine = [s for s in steps if _turn(s) in turns]
    by_task = {task: [s for s in mine if s["task"] == task] for task in SHOWN_TASKS}

    def tally(values: Sequence[Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in values:
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    on = {task: [s for s in rows if _on_overview(s)] for task, rows in by_task.items()}
    towel = {
        _turn(s)
        for s in by_task["S5"]
        if int(s["step"]) == 1
        and isinstance(s.get("cart_after"), list)
        and any(str(item).casefold() == cheapest for item in s["cart_after"])
    }
    return {
        "overview_calls": {task: tally([s["call_name"] for s in on[task]]) for task in SHOWN_TASKS},
        "overview_call_statuses": {
            task: tally([s.get("status") for s in on[task]]) for task in SHOWN_TASKS
        },
        "overview_calls_scrolled": {
            task: sum(
                1
                for s in on[task]
                if s.get("status") == "executed" and s.get("scrolled_px") not in (None, 0)
            )
            for task in SHOWN_TASKS
        },
        "s5_towel_after_step1": len(towel),
        "step1_scrolls": {
            task: sum(
                1 for s in by_task[task] if int(s["step"]) == 1 and s.get("call_name") == "scroll"
            )
            for task in SHOWN_TASKS
        },
        "overview_b9_refusals": {
            task: sum(
                1
                for s in on[task]
                if s.get("status") == "refused" and s.get("reason") == FRESHNESS_REASON
            )
            for task in SHOWN_TASKS
        },
        "overview_not_reached": {
            task: sum(
                1
                for s in on[task]
                if s.get("status") == "failed" and s.get("reason") == OVERVIEW_NOT_REACHED
            )
            for task in SHOWN_TASKS
        },
    }


def arm_metrics(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B3's counts for one arm's records (a turn counts once in every turn count): success by ``score_v2`` (never on
    a forced failure) with its Wilson interval, per task, S4 and S5 with their intervals and the S1-S3 sum; false
    claims (total, non-S4, and the non-S4 figure split into S5 and S1-S3); over-clicks (S1/S2/S5) and S5's; the S4 and
    S5 turns whose turn line logged ``shown`` = 1; A2-8's counts summed per task; S4 turns with an executed scroll; S4
    voiced answers naming the cheapest deal and the enamel mug; narration (voiced text mentioning the overview);
    ``degenerate`` ends; engine errors; the step-1 first-commit and done latency split by overview frame shown or not;
    and the mechanism counts (``mechanism_counts``)."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    forced = {_turn(t): forced_failure(t, by[_turn(t)]) for t in trials}
    ok = {
        _turn(t): (not forced[_turn(t)])
        and bool(score_v2(str(t["task"]), t.get("state"), G._voiced_text(t)))
        for t in trials
    }
    s4 = [t for t in trials if t["task"] == "S4"]
    s5 = [t for t in trials if t["task"] == "S5"]
    fc = [t for t in trials if _false_claim(t, by[_turn(t)], ok[_turn(t)])]
    per_task = {task: sum(1 for (tt, _), v in ok.items() if tt == task and v) for task in TASK_IDS}
    counts = {task: dict.fromkeys(A2_KEYS, 0) for task in TASK_IDS}
    for t in trials:
        logged = t.get("overview_frame_logged")
        if isinstance(logged, Mapping):
            for k in A2_KEYS:
                counts[str(t["task"])][k] += int(logged.get(k) or 0)

    def shown_turns(rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            1
            for t in rows
            if isinstance(t.get("overview_frame_logged"), Mapping)
            and t["overview_frame_logged"].get("shown") == 1
        )

    step1 = [s for s in steps if int(s["step"]) == 1]

    def split(shown: bool) -> dict[str, Any]:
        mine = [s for s in step1 if (s.get("overview_shown") is True) is shown]
        return {
            "first_commit_ms": G._dist(
                [s["first_commit_ms"] for s in mine if s.get("first_commit_ms") is not None]
            ),
            "done_ms": G._dist([s["done_ms"] for s in mine if s.get("done_ms") is not None]),
        }

    def said(t: Mapping[str, Any]) -> str:
        return " ".join(G._voiced_text(t).split()).casefold()

    cheapest = G.cheapest_deal()[0].casefold()
    success = sum(1 for v in ok.values() if v)
    return {
        "success": success,
        # O-11: an arm with no trials (a probe-void run) has no interval
        "success_ci": _ci(success, len(trials)),
        "per_task": per_task,
        "s4": per_task["S4"],
        "s4_ci": _ci(per_task["S4"], len(s4)),
        "success_s5": per_task["S5"],
        "s5_ci": _ci(per_task["S5"], len(s5)),
        "success_s1_s3": sum(per_task[k] for k in FIT_TASKS),
        "false_claims": len(fc),
        "false_claims_non_s4": sum(1 for t in fc if t["task"] != "S4"),
        "false_claims_s5": sum(1 for t in fc if t["task"] == "S5"),
        "false_claims_s1_s3": sum(1 for t in fc if t["task"] in FIT_TASKS),
        "over_clicks": sum(1 for t in trials if over_click(str(t["task"]), t.get("state"))),
        "over_clicks_s5": sum(1 for t in s5 if over_click("S5", t.get("state"))),
        "s4_shown_turns": shown_turns(s4),
        "s5_shown_turns": shown_turns(s5),
        "overview_frame_counts": counts,
        "s4_scrolls": sum(
            1
            for t in s4
            if any(
                s.get("executed_name") == "scroll" and s.get("status") == "executed"
                for s in by[_turn(t)]
            )
        ),
        "s4_named_cheapest": sum(1 for t in s4 if cheapest in said(t)),
        "s4_named_mug": sum(1 for t in s4 if MUG in said(t)),
        "narration": sum(1 for t in trials if NARRATION.search(G._voiced_text(t))),
        "degenerate": sum(1 for t in trials if is_degenerate(t)),
        "engine_errors": sum(1 for t in trials if engine_error(t, by[_turn(t)])),
        "latency": {"shown": split(True), "not_shown": split(False)},
        **mechanism_counts(trials, steps),
    }


def _t0131_section(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """T-0131's per-arm section at the budget and ReGround the trials ran at (the main budget and off when none
    ran)."""
    budget = int(trials[0]["visual_tokens"]) if trials else MAIN_BUDGET
    reground = bool(trials[0]["reground"]) if trials else False
    section: dict[str, Any] = T._arm_section(budget, reground, trials, steps, regrounds)
    return section


def decision_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's B5 inputs: acceptance (T-0111g's per trial with O-B-8b's reading, over the trials that are neither
    an engine error nor a ``degenerate`` end; B-1), the safety sum (T-0131's), the engine errors, B3's counts (S5
    among them, for the S5 guard), the S4 and S5 turns with ``shown`` = 1, and per clause the first failing trial
    ``"<task>#<repeat>"`` (acceptance among those trials, safety by T-0132's C6b rule, engine the first engine-error
    trial)."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    m = arm_metrics(trials, steps)
    plain = [t for t in trials if not forced_failure(t, by[_turn(t)])]
    failing = [t for t in plain if not accepted(t, by[_turn(t)])]
    errors = [t for t in trials if engine_error(t, by[_turn(t)])]
    section = _t0131_section(trials, steps, regrounds)

    def tid(t: Mapping[str, Any]) -> str:
        return f"{t['task']}#{int(t['repeat'])}"

    return {
        "accepted": not failing,
        "safety": sum(int(v) for v in section["safety"].values()),
        "engine_errors": len(errors),
        "success": m["success"],
        "s4": m["s4"],
        "s5": m["success_s5"],
        "false_claims_non_s4": m["false_claims_non_s4"],
        "over_clicks": m["over_clicks"],
        "s4_shown_turns": m["s4_shown_turns"],
        "s5_shown_turns": m["s5_shown_turns"],
        "first_failing": {
            "acceptance": tid(failing[0]) if failing else None,
            "safety": P._first_failing(trials, steps, regrounds)["safety"],
            "engine": tid(errors[0]) if errors else None,
        },
    }


# =====================================================================================================
# B5: the decision (H-T151-1)
# =====================================================================================================


def _ge(a: float, b: float) -> bool:
    return float(a) >= float(b) - EPS


def _le(a: float, b: float) -> bool:
    return float(a) <= float(b) + EPS


def _first_ineligible(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    """C6b: the first ineligible arm in ``LABELS`` order and its first failing clause in ``CLAUSES`` order; the
    engine clause fails only above ``ENGINE_ERRORS_MAX`` engine-error trials (B5)."""
    for arm in LABELS:
        i = inputs[arm]
        checks = {
            "acceptance": bool(i["accepted"]),
            "safety": int(i["safety"]) == 0,
            "engine": int(i["engine_errors"]) <= ENGINE_ERRORS_MAX,
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
    """B5's label and why: P1 failed -> inconclusive; a configuration finding (B4) -> void, reason
    ``configuration: <first>``; P2, P3, P6, P7, P8 or P9 failed (``void``) -> void; stopped early -> incomplete; an
    arm ineligible -> inconclusive naming the arm, the clause and the first failing trial; then ``overview frame`` iff
    every adoption clause holds (the S5 guard ``S5(B) >= S5(A) - 3`` among them); else ``overview not shown`` when arm
    B's S4 or S5 turns with ``shown`` = 1 are under ``NOT_SHOWN_BELOW`` (18), else ``no change``. ``inputs`` maps
    ``"A"``/``"B"`` to ``decision_inputs``'s shape."""
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
    a, b = inputs["A"], inputs["B"]
    adopt = (
        _ge(b["s4"] - a["s4"], S4_MARGIN)
        and _ge(b["success"], a["success"] - SUCCESS_MARGIN)
        and _le(b["false_claims_non_s4"], a["false_claims_non_s4"] + FALSE_CLAIM_MARGIN)
        and _le(b["over_clicks"], a["over_clicks"] + OVER_CLICK_MARGIN)
        and _ge(b["s5"], a["s5"] - S5_MARGIN)
    )
    if adopt:
        return OVERVIEW_FRAME, None
    if int(b["s4_shown_turns"]) < NOT_SHOWN_BELOW or int(b["s5_shown_turns"]) < NOT_SHOWN_BELOW:
        return NOT_SHOWN, None
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
# The report (pure)
# =====================================================================================================


def _arm_section(
    row: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's section: T-0131's (acceptance, outcomes and false claims by T-0111g's ``score``, steps, latency,
    cache, wall time, VRAM, the engine flag, every safety count), with its ``score`` keys renamed as T-0134's
    (``success_v1``, ``false_claims_v1``) and its step latency as ``latency_v1`` (``latency`` is B3's step-1 split by
    overview frame shown), then B3's counts, the arm's settings and the intervals."""
    base = _t0131_section(trials, steps, regrounds)
    base["success_v1"] = base.pop("success")
    base["false_claims_v1"] = base.pop("false_claims")
    base["latency_v1"] = base.pop("latency")
    m = arm_metrics(trials, steps)
    n_oc = sum(1 for t in trials if t["task"] in ("S1", "S2", "S5"))
    n_non_s4 = sum(1 for t in trials if t["task"] != "S4")
    n_s5 = sum(1 for t in trials if t["task"] == "S5")
    n_s1_s3 = sum(1 for t in trials if t["task"] in FIT_TASKS)
    return {
        **base,
        **m,
        "label": row["label"],
        "page_overview": row["page_overview"],
        "auto_overview": row["auto_overview"],
        "overview_frame": row["overview_frame"],
        "tools_sha256": row["tools_sha256"],
        "live_feedback": row["live_feedback"],
        "echo": row["echo"],
        "recover_thought_calls": row["recover_thought_calls"],
        "degeneracy_guard": row["degeneracy_guard"],
        "n": len(trials),
        "s4_n": sum(1 for t in trials if t["task"] == "S4"),
        "s5_n": n_s5,
        "s1_s3_n": n_s1_s3,
        "false_claims_ci": _ci(m["false_claims"], len(trials)),
        "false_claims_non_s4_n": n_non_s4,
        "false_claims_s5_n": n_s5,
        "false_claims_s1_s3_n": n_s1_s3,
        "over_clicks_n": n_oc,
        "over_clicks_ci": _ci(m["over_clicks"], n_oc),
        "over_clicks_s5_n": n_s5,
    }


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    bodies: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    *,
    arms: Sequence[Mapping[str, Any]],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: ``void`` (P1, P2, P3, P6, P7, P8 or P9 failed), ``stopped``, the probes, every arm's section,
    B3's Newcombe differences (B-A on S4 of 20, on success of 100 and on S5 of 20), B4's ``configuration_findings``
    and the decision (each arm's B5 inputs with their first failing trials)."""
    sections: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    rows = {str(r["label"]): r for r in arms}
    for label in LABELS:
        arm_trials = [t for t in trials if t.get("arm") == label]
        arm_steps = [s for s in steps if s.get("arm") == label]
        mine = [g for g in regrounds if g.get("arm") == label]
        sections[label] = _arm_section(rows[label], arm_trials, arm_steps, mine)
        inputs[label] = decision_inputs(arm_trials, arm_steps, mine)
    a, b = sections["A"], sections["B"]
    differences = {
        "s4 B-A": list(newcombe(b["s4"], S4_TURNS, a["s4"], S4_TURNS)),
        "success B-A": list(newcombe(b["success"], TURNS_PER_ARM, a["success"], TURNS_PER_ARM)),
        "s5 B-A": list(newcombe(b["success_s5"], S5_TURNS, a["success_s5"], S5_TURNS)),
    }
    findings = configuration_findings(trials, steps, regrounds, bodies, arms)
    p1_passed = _passed(probes, "p1")
    void = not all(_passed(probes, p) for p in PROBES)
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
        "arms": sections,
        "differences": differences,
        "configuration_findings": findings,
        "decision": {"label": label, "reason": reason, "inputs": inputs},
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def _per_task(counts: Mapping[str, Any]) -> str:
    return "; ".join(f"{task} {_fmt(v)}" for task, v in counts.items())


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reason, B5's stated numbers (the S5 guard's false-reject rate,
    S4 power, T-0150's S5 history), the click-scoring caveat (O-B (9)), the probes, every arm's numbers with B3's
    mechanism counts, the differences, and the history (not like-for-like)."""
    lines = ["# T-0151 B overview-frame A/B (report-only, plus H-T151-1's label rule)", ""]
    if manifest is not None:
        regime = manifest.get("regime") or {}
        lines += [
            "## Regime",
            "",
            f"- live default at HEAD (read from cli): {_fmt(manifest.get('live_default'))}; arm-table sha256 "
            f"{_fmt(manifest.get('arm_table_sha256'))}",
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
        "## Decision (B5, H-T151-1 approved: T-0150's margins plus the S5 guard -3/20)",
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
        i = decision["inputs"][label]
        lines.append(
            f"- arm {label}: accepted {i['accepted']}, safety {i['safety']}, engine errors {i['engine_errors']} "
            f"(eligible at <= {ENGINE_ERRORS_MAX}); S4 {i['s4']}/{S4_TURNS}, success {i['success']}/{TURNS_PER_ARM}, "
            f"S5 {i['s5']}/{S5_TURNS} (guard: B >= A - {S5_MARGIN}), non-S4 false claims {i['false_claims_non_s4']}, "
            f"over-clicks {i['over_clicks']}, S4/S5 turns with the overview frame shown {i['s4_shown_turns']}/"
            f"{i['s5_shown_turns']} (not shown below {NOT_SHOWN_BELOW} on either); first failing {i['first_failing']}"
        )
    lines += [
        f"- S5 guard: {S5_GUARD}",
        f"- power: {POWER}",
        f"- S5 history: {S5_HISTORY}",
        f"- which guards can move: {GUARDS}",
        f"- token cost: {TOKEN_COST}",
        f"- click scoring: {CLICK_SCORING}",
        "",
        "## Probes (B2)",
        "",
    ]
    probes = report["probes"]
    p1 = probes.get("p1") or {}
    for budget, row in sorted((p1.get("budgets") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"- P1 {budget}: frame delta {_fmt(row.get('frame_delta'))}, crop delta {_fmt(row.get('crop_delta'))}, "
            f"expected {row.get('expected')} (+0..{T.P1_DELIMITERS}), ok {row.get('ok')}"
        )
    p2, p3 = probes.get("p2") or {}, probes.get("p3") or {}
    p6, p7, p8 = probes.get("p6") or {}, probes.get("p7") or {}, probes.get("p8") or {}
    p9 = probes.get("p9") or {}
    lines += [
        f"- P2 (one image at {T.P2_BUDGET}): HTTP {_fmt(p2.get('http_status'))}, healthy after "
        f"{_fmt(p2.get('healthy_after'))}, passed {_fmt(p2.get('passed'))}",
        f"- P3 (text only): HTTP {_fmt(p3.get('http_status'))}, passed {_fmt(p3.get('passed'))}",
    ]
    for part in ("deals", "fits", "hostile"):
        d = p6.get(part) or {}
        lines.append(
            f"- P6 {part}: outcome {_fmt(d.get('outcome'))}, screens {_fmt(d.get('screens'))}, clipped "
            f"{_fmt(d.get('clipped'))}, image {_fmt(d.get('width'))}x{_fmt(d.get('height'))}, scroll_y "
            f"{_fmt(d.get('scroll_y_before'))} -> {_fmt(d.get('scroll_y_after'))}"
        )
    lines += [
        f"- P6 viewport {_fmt(p6.get('viewport'))}, error {_fmt(p6.get('error'))}, passed {_fmt(p6.get('passed'))}",
        f"- P7 (step-1 request with the overview frame, a fresh session): /tokenize {_fmt(p7.get('tokens'))} of "
        f"{_fmt(p7.get('limit'))}, images {_fmt(p7.get('images'))} (overview {_fmt(p7.get('overview_images'))}), "
        f"overview tokens {_fmt(p7.get('overview_tokens'))} (report-only; the live-frame step 1 is "
        f"{_fmt(p7.get('tokens_live_frame'))}), HTTP {_fmt(p7.get('http_status'))}, healthy after "
        f"{_fmt(p7.get('healthy_after'))}, passed {_fmt(p7.get('passed'))}",
    ]
    for task in TASK_IDS:
        c = (p8.get("tasks") or {}).get(task) or {}
        lines.append(
            f"- P8 {task} ({_fmt(c.get('start_page'))}): outcome {_fmt(c.get('outcome'))}, screens "
            f"{_fmt(c.get('screens'))}"
        )
    lines.append(
        f"- P8 viewport {_fmt(p8.get('viewport'))}, error {_fmt(p8.get('error'))}, passed {_fmt(p8.get('passed'))}"
    )
    for c in p9.get("clicks") or []:
        lines.append(
            f"- P9 click_at{tuple(c.get('point') or ())}: status {_fmt(c.get('status'))}, cart {_fmt(c.get('cart'))}, "
            f"action {_fmt(c.get('action'))}, scroll_y after {_fmt(c.get('scroll_y_after'))} (report-only)"
        )
    lines += [
        f"- P9 viewport {_fmt(p9.get('viewport'))}, error {_fmt(p9.get('error'))}, passed {_fmt(p9.get('passed'))}",
        "",
        "## Arms (B3; each number report-only)",
        "",
        "| arm | overview frame | S4 v2 (Wilson 95 %) | S5 v2 (Wilson 95 %) | S1-S3 | success v2 (Wilson 95 %) | "
        "false claims (non-S4: S5 + S1-S3) | over-clicks (S5) | S4/S5 shown turns | S4 scrolls | S4 names cheapest / "
        "mug | degenerate | engine errors | min free VRAM MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        arm = report["arms"][label]
        lines.append(
            f"| {label} | {arm['overview_frame']} | {arm['s4']}/{arm['s4_n']} {_interval(arm['s4_ci'])} | "
            f"{arm['success_s5']}/{arm['s5_n']} {_interval(arm['s5_ci'])} | "
            f"{arm['success_s1_s3']}/{arm['s1_s3_n']} | "
            f"{arm['success']}/{arm['n']} {_interval(arm['success_ci'])} | {arm['false_claims']} "
            f"({arm['false_claims_non_s4']}/{arm['false_claims_non_s4_n']}: {arm['false_claims_s5']}/"
            f"{arm['false_claims_s5_n']} + {arm['false_claims_s1_s3']}/{arm['false_claims_s1_s3_n']}) | "
            f"{arm['over_clicks']}/{arm['over_clicks_n']} ({arm['over_clicks_s5']}/{arm['over_clicks_s5_n']}) | "
            f"{arm['s4_shown_turns']}/{arm['s5_shown_turns']} | {arm['s4_scrolls']} | "
            f"{arm['s4_named_cheapest']} / {arm['s4_named_mug']} | "
            f"{arm['degenerate']} | {arm['engine_errors']} | {_fmt(arm['vram_min_free_mib'])} |"
        )
    lines += ["", "## Differences (Newcombe 95 %, method 10)", ""]
    lines += [f"- {k}: {_interval(v)}" for k, v in report["differences"].items()]
    for label in LABELS:
        arm = report["arms"][label]
        lat = arm["latency"]
        lines += [
            "",
            f"### {label} (overview_frame {arm['overview_frame']}, auto_overview {arm['auto_overview']}, "
            f"page_overview {arm['page_overview']}, tools {str(arm['tools_sha256'])[:12]}, live_feedback "
            f"{arm['live_feedback']}, echo {arm['echo']}, recovery {arm['recover_thought_calls']}, guard "
            f"{arm['degeneracy_guard']})",
            "",
        ]
        lines += [
            f"- {task}: score_v2 {arm['per_task'][task]}/{o['n']}; T-0111g score {o['k']}/{o['n']}"
            for task, o in arm["outcomes"].items()
        ]
        st = arm["steps"]
        acc = arm["acceptance"]
        lines += [
            "- A2-8 counts per task: "
            + "; ".join(
                f"{task} " + ", ".join(f"{k} {v}" for k, v in c.items())
                for task, c in arm["overview_frame_counts"].items()
            ),
            f"- step-1 calls on the overview frame by name: {_per_task(arm['overview_calls'])}",
            f"- their statuses: {_per_task(arm['overview_call_statuses'])}; scrolled: "
            f"{_per_task(arm['overview_calls_scrolled'])}",
            f"- S5 turns with the tea towel in the cart right after step 1: {arm['s5_towel_after_step1']}",
            f"- step-1 scroll calls: {_per_task(arm['step1_scrolls'])}; B9 refusals of overview calls: "
            f"{_per_task(arm['overview_b9_refusals'])}; OVERVIEW_NOT_REACHED failures: "
            f"{_per_task(arm['overview_not_reached'])}",
            f"- turns whose voiced text mentions the overview: {arm['narration']}",
            f"- step-1 first commit ms, overview frame shown: {T._dist_cell(lat['shown']['first_commit_ms'])}; not "
            f"shown: {T._dist_cell(lat['not_shown']['first_commit_ms'])}",
            f"- step-1 done ms, overview frame shown: {T._dist_cell(lat['shown']['done_ms'])}; not shown: "
            f"{T._dist_cell(lat['not_shown']['done_ms'])}",
            f"- steps per turn: {T._dist_cell(st['per_turn'])}",
            "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
            f"- step first commit ms: {T._dist_cell(arm['latency_v1']['first_commit_ms'])}; done ms: "
            f"{T._dist_cell(arm['latency_v1']['done_ms'])}",
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


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _image_sha(url: str) -> str:
    """sha16 of an image part's JPEG bytes (a data URL decoded), or of the URL text when it is not a data URL."""
    head, sep, payload = url.partition(",")
    if sep and head.startswith("data:") and head.endswith(";base64"):
        with contextlib.suppress(ValueError):
            return _sha16(base64.b64decode(payload))
    return _sha16(url.encode("utf-8"))


def _parts(content: Any) -> list[dict[str, str]]:
    """A message's content as ``{"text"}``/``{"image": sha16}`` parts (any other part as ``{"other": type}``)."""
    if isinstance(content, str):
        return [{"text": content}]
    out: list[dict[str, str]] = []
    for p in content if isinstance(content, list) else []:
        if not isinstance(p, Mapping):
            continue
        kind = p.get("type")
        if kind == "text":
            out.append({"text": str(p.get("text") or "")})
        elif kind == "image_url":
            out.append({"image": _image_sha(str((p.get("image_url") or {}).get("url") or ""))})
        else:
            out.append({"other": str(kind)})
    return out


def _tool_names(body: Mapping[str, Any]) -> list[str] | None:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    return [str((t.get("function") or {}).get("name")) for t in tools if isinstance(t, Mapping)]


#: the turn machine's request ids (T-0111a §2, T-0089 R2): pass 1 of step 1 / step k, and the turn's one repair
_PASS1_ID = re.compile(r"^(.*)-gen(?:-s\d+)?$")
_REPAIR_ID = re.compile(r"^(.*)-repair$")


def body_line(
    arm: str,
    task_id: str,
    repeat: int,
    raw: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    overviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """``bodies.jsonl``'s line for one turn: every body the meters forwarded (``raw``, in order), each
    ``{"label", "phase", "step", "tools", "messages"}`` with ``phase`` and ``step`` from the attempt's
    ``llm.request`` record (a prefill: ``"prefill"`` and ``None``; the turn's generate is step 1), ``tools`` the
    declared names and each message's parts as ``{"text"}``/``{"image": sha16}``; and the turn's ``overviews``."""
    by_attempt: dict[str, Mapping[str, Any]] = {}
    # review F2: a ``tool_repair`` continues the pass-1 prompt of the call it repairs, so it takes that request's
    # step. The machine ids share the turn: pass 1 is ``{turn}-gen`` / ``{turn}-gen-s{k}``, the turn's one repair
    # ``{turn}-repair``, which starts once pass 1 is done, so the turn's last pass-1 record before it is the one.
    pass1_step: dict[str, int] = {}
    repair_step: dict[str, int | None] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") != "llm.request":
            continue
        if d.get("attempt_id") is not None:
            by_attempt[str(d.get("attempt_id"))] = d
        rid = str(d.get("request_id") or "")
        pass1 = _PASS1_ID.match(rid)
        if d.get("phase") in ("generate", "tool_step") and pass1 is not None:
            pass1_step[pass1.group(1)] = 1 if d.get("phase") == "generate" else int(d["step"])
        elif d.get("phase") == "tool_repair" and d.get("attempt_id") is not None:
            repair = _REPAIR_ID.match(rid)
            repair_step[str(d.get("attempt_id"))] = (
                None if repair is None else pass1_step.get(repair.group(1))
            )
    requests: list[dict[str, Any]] = []
    for label, body in raw.items():
        kind, _, attempt = label.partition(":")
        rec = by_attempt.get(attempt) or {}
        if kind == "prefill":
            phase: Any = "prefill"
            step: Any = None
        else:
            phase = rec.get("phase") or kind
            if phase == "generate":
                step = 1
            elif phase == "tool_repair":
                step = repair_step.get(attempt)
            else:
                step = rec.get("step")
        requests.append(
            {
                "label": label,
                "phase": phase,
                "step": None if step is None else int(step),
                "tools": _tool_names(body),
                "messages": [
                    {"role": str(m.get("role")), "parts": _parts(m.get("content"))}
                    for m in body.get("messages") or []
                    if isinstance(m, Mapping)
                ],
            }
        )
    return {
        "arm": arm,
        "task": task_id,
        "repeat": repeat,
        "requests": requests,
        "overviews": [dict(o) for o in overviews],
    }


def overview_records(
    calls: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One overview per overview-frame capture (T-0151 A2-2: the executor call whose id ends ``FRAME_CALL_SUFFIX``)
    that executed and whose frame ``overview_frame_of`` registered (``frames``, keyed by the overview id): step 1
    (A2-4), sha16 of the frame's JPEG, A3's ``screens`` and the frame's size. A ``too_tall`` picture (read through
    ``overview_of`` and dropped) or a ``None`` registration gives no record."""
    out: list[dict[str, Any]] = []
    for call_id, call in calls.items():
        if not str(call_id).endswith(FRAME_CALL_SUFFIX) or call.get("status") != "executed":
            continue
        resolved = call.get("resolved") or {}
        if "overview" not in resolved:
            continue
        frame = frames.get(str(resolved.get("overview")))
        if frame is None:
            continue
        out.append(
            {
                "step": 1,
                "sha": _sha16(frame.jpeg),
                "screens": float(resolved["screens"]),
                "width": int(frame.width),
                "height": int(frame.height),
            }
        )
    return out


def frame_steps(
    steps: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    executed: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The step lines with T-0151's four fields, read from the session log and the frame watch, never from the arm:
    ``overview_shown`` (the step's own ``llm.request`` record carries A2-8's ``overview_frame`` key), ``frame_kind``
    (the ``frame_kind`` of its call's last ``tool.call`` record, ``None`` when absent; A2-8, O-A2 R5), and, for a call
    the executor ``executed`` (``executed``: the frame watch's per-call record), ``scrolled_px`` (its
    ``resolved["scrolled_px"]``, ``None`` when absent; A1-4) and ``cart_after`` (the agent page's
    ``window.__state.cart`` read right after it; ``None`` when no call executed or the read failed)."""
    shown = {
        str((r.get("data") or {}).get("attempt_id"))
        for r in records
        if r.get("kind") == "llm.request"
        and (r.get("data") or {}).get("overview_frame") is not None
    }
    kinds: dict[str, Any] = {}
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") == "tool.call" and d.get("call_id") is not None:
            kinds[str(d["call_id"])] = d.get("frame_kind")  # the call's last record wins
    out: list[dict[str, Any]] = []
    for s in steps:
        cid = None if s.get("call_id") is None else str(s.get("call_id"))
        call = executed.get(cid) if cid is not None else None
        ran = call is not None and call.get("status") == "executed"
        resolved = (call or {}).get("resolved") or {}
        scrolled = resolved.get("scrolled_px") if ran else None
        out.append(
            {
                **s,
                "overview_shown": s.get("attempt_id") is not None
                and str(s.get("attempt_id")) in shown,
                "frame_kind": kinds.get(cid) if cid is not None else None,
                "scrolled_px": None if scrolled is None else int(scrolled),
                "cart_after": (call or {}).get("cart_after") if ran else None,
            }
        )
    return out


def _not_run(row: Mapping[str, Any], task_id: str, repeat: int, reason: str) -> dict[str, Any]:
    """T-0132's not-run trial (no body was forwarded) with the arm's settings."""
    crop = row.get("reground_visual_tokens")
    trial: dict[str, Any] = P._not_run(
        task_id, repeat, reason, int(row["visual_tokens"]), bool(row["reground"]), crop
    )
    trial.update(
        arm=row["label"],
        live_feedback=row["live_feedback"],
        echo=row["echo"],
        recover_thought_calls=row["recover_thought_calls"],
        degeneracy_guard=row["degeneracy_guard"],
        page_overview=row["page_overview"],
        auto_overview=row["auto_overview"],
        overview_frame=row["overview_frame"],
        tools_version=None,
        overview_frame_logged=None,
        auto_overview_logged=None,
        page_overview_logged=None,
        degeneracy_guard_logged=None,
        recover_thought_calls_logged=None,
        live_feedback_logged=None,
        visual_tokens_logged=None,
    )
    return trial


# =====================================================================================================
# B2: P6, P8 and P9 (browser only; loopback only; no inference) and P7 (one /tokenize, one generate)
# =====================================================================================================


class _ProbeServer:
    """T-0134's ``HOSTILE_PAGE`` at ``HOSTILE_PATH`` and ``FIT_PAGE`` at ``FIT_PATH`` on 127.0.0.1 (an ephemeral
    port), from a daemon thread; nothing else."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        pages = {HOSTILE_PATH: HOSTILE_PAGE.encode("utf-8"), FIT_PATH: FIT_PAGE.encode("utf-8")}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                page = pages.get(self.path.split("?", 1)[0])
                if page is None:
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
            target=self._server.serve_forever, name="t0151-probe-pages", daemon=True
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


def _probe_executor(origin: str) -> Any:
    """A fresh headless ``BrowserToolExecutor`` at ``VIEWPORT``, ``origin`` the only allowed one, echo ``model``."""
    from local_astra.computer.executor import BrowserToolExecutor

    return BrowserToolExecutor(
        now_ms=lambda: 0,
        viewport=VIEWPORT,
        guard=None,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({origin}),
        echo="model",
    )


def _probe_invocation(name: str, frame: Any = None, **arguments: Any) -> Any:
    from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation

    geometry = (
        None
        if frame is None
        else FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
    )
    return ToolInvocation(
        call_id=f"t0151-probe-{name}",
        turn_id="t0151-probe",
        name=name,
        arguments=dict(arguments),
        source="draft",
        frame=geometry,
    )


async def _overview_once(origin: str, url: str) -> dict[str, Any]:
    """Through a fresh headless executor: the driver's ``scroll_y`` (CDP), one ``page_overview``, the frame
    ``overview_of`` keeps, and ``scroll_y`` again (P6, as T-0150's)."""
    ex = _probe_executor(origin)
    try:
        await asyncio.wait_for(ex.open(url), timeout=P6_CALL_S)
        before = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=P6_CALL_S)
        out = await asyncio.wait_for(
            ex.execute(_probe_invocation("page_overview")), timeout=P6_CALL_S
        )
        after = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=P6_CALL_S)
        resolved = dict(out.resolved or {})
        frame = ex.overview_of(resolved.get("overview")) if "overview" in resolved else None
        if out.status == "executed" and frame is not None:
            outcome = "shown"
        elif out.status == "executed" and out.action == OVERVIEW_FITS:
            outcome = "fits"
        else:
            outcome = str(out.status)
        return {
            "viewport": [int(v) for v in ex.driver.viewport],
            "outcome": outcome,
            "status": out.status,
            "action": out.action,
            "reason": out.reason,
            "screens": resolved.get("screens"),
            "clipped": resolved.get("clipped"),
            "width": None if frame is None else int(frame.width),
            "height": None if frame is None else int(frame.height),
            "scroll_y_before": float(before["scroll_y"]),
            "scroll_y_after": float(after["scroll_y"]),
            "content_height": float(before["content_height"]),
            "viewport_height": float(before["viewport_height"]),
        }
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), timeout=P6_CALL_S)


async def _probe_p6() -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    out: dict[str, Any] = {
        "viewport": None,
        "deals": None,
        "fits": None,
        "hostile": None,
        "error": None,
    }
    errors: list[str] = []
    pages = StartPageServer()
    pages.start()
    try:
        deals = await _overview_once(pages.origin, pages.origin + P6_PAGE)
        out.update(deals=deals, viewport=deals["viewport"])
    except Exception as exc:
        errors.append(f"deals: {exc!r}")
    finally:
        pages.stop()
    server = _ProbeServer()
    server.start()
    try:
        for part, path in (("fits", FIT_PATH), ("hostile", HOSTILE_PATH)):
            try:
                out[part] = await _overview_once(server.origin, server.origin + path)
            except Exception as exc:
                errors.append(f"{part}: {exc!r}")
    finally:
        server.stop()
    out["error"] = "; ".join(errors) or None
    return out


def probe_p6() -> dict[str, Any]:
    """B2 P6 (browser only, no inference; as T-0150's): ``page_overview`` through the real executor at 1280x800 on
    ``/demo/deals.html`` (a fresh start-page server), on ``FIT_PAGE`` and on T-0134's hostile page (this file's own
    loopback server). A part that fails is ``None`` with the error; ``main`` adds ``passed`` (``p6_ok``)."""
    return asyncio.run(_probe_p6())


async def _registered(ex: Any) -> tuple[Any, Any, str]:
    """The session's A2-2/A2-3 order on an open executor: an own frame first (the live snapshot), then the capture,
    then the decision -- ``fits`` (executed, no picture), ``too_tall`` (above ``OVERVIEW_FRAME_MAX_SCREENS``: the
    picture read through ``overview_of`` and dropped), ``shown`` (``overview_frame_of`` registered a frame),
    ``failed`` (a ``failed`` capture or a ``None`` registration) or ``refused``. Returns (outcome, frame, decision)."""
    from local_astra.contracts.perception import Frame

    await asyncio.wait_for(ex.frame(), timeout=P6_CALL_S)
    out = await asyncio.wait_for(ex.execute(_probe_invocation("page_overview")), timeout=P6_CALL_S)
    resolved = dict(out.resolved or {})
    key = resolved.get("overview")
    if out.status == "failed":
        return out, None, "failed"
    if out.status != "executed":
        return out, None, "refused"
    if key is None:
        return out, None, "fits"
    if not float(resolved.get("screens", 0.0)) <= OVERVIEW_FRAME_MAX_SCREENS:
        ex.overview_of(key)
        return out, None, "too_tall"
    frame = ex.overview_frame_of(key)
    if not isinstance(frame, Frame):
        return out, None, "failed"
    return out, frame, "shown"


async def _census_once(origin: str, url: str) -> dict[str, Any]:
    """One P8 page: a fresh executor, the page opened, ``_registered``'s decision and the capture's facts."""
    ex = _probe_executor(origin)
    try:
        await asyncio.wait_for(ex.open(url), timeout=P6_CALL_S)
        metrics = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=P6_CALL_S)
        out, frame, outcome = await _registered(ex)
        resolved = dict(out.resolved or {})
        return {
            "viewport": [int(v) for v in ex.driver.viewport],
            "outcome": outcome,
            "status": out.status,
            "action": out.action,
            "reason": out.reason,
            "screens": resolved.get("screens"),
            "clipped": resolved.get("clipped"),
            "width": None if frame is None else int(frame.width),
            "height": None if frame is None else int(frame.height),
            "content_height": float(metrics["content_height"]),
            "viewport_height": float(metrics["viewport_height"]),
        }
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), timeout=P6_CALL_S)


async def _probe_p8() -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    out: dict[str, Any] = {"viewport": None, "tasks": {}, "error": None}
    errors: list[str] = []
    pages = StartPageServer()
    pages.start()
    try:
        for task in TASKS:
            try:
                got = await _census_once(pages.origin, pages.origin + task.start_page)
            except Exception as exc:
                errors.append(f"{task.task_id}: {exc!r}")
                continue
            if out["viewport"] is None:
                out["viewport"] = got["viewport"]
            out["tasks"][task.task_id] = {"start_page": task.start_page, **got}
            del out["tasks"][task.task_id]["viewport"]
    finally:
        pages.stop()
    out["error"] = "; ".join(errors) or None
    return out


def probe_p8() -> dict[str, Any]:
    """B2 P8 (browser only, no inference): the start-page census -- the session's overview-frame decision mirrored
    through the real executor at 1280x800 (one fresh headless executor per task, one fresh start-page server) on each
    task's start page. A task that fails is left out with the error (so P8 fails); ``main`` adds ``passed``
    (``p8_ok``)."""
    return asyncio.run(_probe_p8())


async def _p9_click(origin: str, url: str, point: tuple[int, int]) -> dict[str, Any]:
    """One P9 click: a fresh executor on ``url``, the overview frame registered as the session registers it, then
    ``click_at(*point)`` on its geometry; the cart (``window.__state.cart``) and ``scroll_y`` read after it."""
    ex = _probe_executor(origin)
    record: dict[str, Any] = {
        "point": list(point),
        "registered": False,
        "status": None,
        "action": None,
        "reason": None,
        "resolved": None,
        "scroll_y_after": None,
        "cart": None,
        "error": None,
    }
    try:
        await asyncio.wait_for(ex.open(url), timeout=P6_CALL_S)
        _, frame, outcome = await _registered(ex)
        record["outcome"] = outcome
        if frame is None:
            record["error"] = f"the overview frame was not registered ({outcome})"
            return record
        record.update(registered=True, frame=[int(frame.width), int(frame.height)])
        x, y = point
        out = await asyncio.wait_for(
            ex.execute(_probe_invocation("click_at", frame, x=x, y=y)), timeout=P6_CALL_S
        )
        record.update(
            status=out.status, action=out.action, reason=out.reason, resolved=dict(out.resolved)
        )
        metrics = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=P6_CALL_S)
        record["scroll_y_after"] = float(metrics["scroll_y"])
        state = await asyncio.wait_for(ex.driver.page.evaluate(G.STATE_JS), timeout=P6_CALL_S)
        if isinstance(state, Mapping) and isinstance(state.get("cart"), list):
            record["cart"] = [str(item) for item in state["cart"]]
        return record
    except Exception as exc:
        record["error"] = repr(exc)
        return record
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), timeout=P6_CALL_S)


async def _probe_p9() -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    out: dict[str, Any] = {"viewport": [int(v) for v in VIEWPORT], "clicks": [], "error": None}
    pages = StartPageServer()
    pages.start()
    try:
        for point in P9_POINTS:
            out["clicks"].append(await _p9_click(pages.origin, pages.origin + P9_PAGE, point))
    finally:
        pages.stop()
    errors = [f"{tuple(c['point'])}: {c['error']}" for c in out["clicks"] if c.get("error")]
    out["error"] = "; ".join(errors) or None
    return out


def probe_p9() -> dict[str, Any]:
    """B2 P9 (browser only, no inference; A1-7): the two deals.html clicks on the registered overview frame, each on a
    fresh page (a fresh executor) through the real executor at 1280x800 and one fresh start-page server; each record
    is ``{"point", "status", "action", "scroll_y_after", "cart", ...}``. ``main`` adds ``passed`` (``p9_ok``)."""
    return asyncio.run(_probe_p9())


def _p7_overview_jpeg() -> bytes:
    """A deterministic synthetic page-like overview (``P7_OVERVIEW_SIZE``): a tall page scaled to fit one screen."""
    import io

    from PIL import Image, ImageDraw

    w, h = P7_OVERVIEW_SIZE
    img = Image.new("RGB", (w, h), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, 30), fill=(32, 48, 80))
    for row in range(4):
        for col in range(3):
            x0, y0 = 40 + col * 260, 60 + row * 180
            draw.rectangle((x0, y0, x0 + 220, y0 + 150), outline=(180, 180, 180), width=2)
            draw.rectangle(
                (x0 + 20, y0 + 15, x0 + 200, y0 + 95), fill=(70 + 40 * col, 90 + 30 * row, 200)
            )
            draw.rectangle((x0 + 70, y0 + 110, x0 + 150, y0 + 135), fill=(20, 110, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def p7_prompts() -> tuple[Any, Any, Any]:
    """B2 P7's **step-1** requests of a fresh S4-like session (no history): the live system message (tools v1's native
    tools field and ``LIVE_TOOL_GUIDANCE``, thinking on) and S4's request, as (1) the overview-frame step 1 (A2-4: the
    current 1280x800 frame swapped for the overview, then ``OVERVIEW_FRAME_LABEL``; 1 image, no T-0134 label), (2) the
    live default's step 1 (the current frame and T-0134's screen label; report-only) and (3) the same request with no
    image (the base of ``overview_tokens``)."""
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE, PromptBuilder
    from local_astra.conductor.session import (
        DEFAULT_SYSTEM_PROMPT,
        LABEL_NOW,
    )
    from local_astra.conductor.session import (
        OVERVIEW_FRAME_LABEL as SESSION_LABEL,
    )
    from local_astra.contracts.perception import Frame, Source
    from local_astra.toolcall import loop, native
    from local_astra.toolcall import tools as toolset

    jpeg = P._p4_jpegs()[0]
    frame = Frame(
        source=Source.SCREEN,
        ts_ms=1000,
        width=P.P4_FRAME[0],
        height=P.P4_FRAME[1],
        jpeg=jpeg,
        phash=_sha16(jpeg),
    )
    ov_jpeg = _p7_overview_jpeg()
    overview = Frame(
        source=Source.WINDOW,
        ts_ms=3000,
        width=P7_OVERVIEW_SIZE[0],
        height=P7_OVERVIEW_SIZE[1],
        jpeg=ov_jpeg,
        phash=_sha16(ov_jpeg),
    )
    request = G._task("S4").request
    builder = PromptBuilder(
        DEFAULT_SYSTEM_PROMPT,
        thinking=True,
        tools=tuple(native.tools_field(toolset.load(toolset.TOOLS_V1_PATH))),
        tool_guidance=LIVE_TOOL_GUIDANCE,
    )
    full = builder.full(builder.prefix([], frames=(frame,)), request, None, ())
    bare = loop.label_current_image(full, "[" + LABEL_NOW + "]")
    swapped = loop.swap_newest_image(full, overview)
    prompt = loop.label_current_image(swapped, SESSION_LABEL.format(screens=P7_SCREENS))
    text_only = builder.full(builder.prefix([], frames=()), request, None, ())
    return prompt, bare, text_only


async def _tokens(base_url: str, model: str, body: Mapping[str, Any]) -> tuple[Any, int | None]:
    """T-0132 P4's count: ``/tokenize`` of the request body's chat fields."""
    keys = (
        "messages",
        "tools",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "mm_processor_kwargs",
    )
    payload = {"model": model, **{k: body[k] for k in keys if k in body}}
    status, got = await asyncio.to_thread(T._post_json, G._root(base_url) + "/tokenize", payload)
    tokens: int | None = None
    if status == 200 and isinstance(got, dict):
        count = got.get("count")
        if isinstance(count, int):
            tokens = count
        elif isinstance(got.get("tokens"), list):
            tokens = len(got["tokens"])
    return status, tokens


async def _probe_p7(base_url: str, model: str, detokenizer: Any) -> dict[str, Any]:
    import httpx

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.contracts.llm import ImagePart, LlmOptions

    prompt, bare, text_only = p7_prompts()
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
    status, tokens = await _tokens(
        base_url, model, client.inner.request_body(prompt, P7_REQUEST_ID)
    )
    _, live = await _tokens(base_url, model, client.inner.request_body(bare, P7_REQUEST_ID))
    _, without = await _tokens(base_url, model, client.inner.request_body(text_only, P7_REQUEST_ID))
    http_status = 200
    error: str | None = None
    try:

        async def drain() -> None:
            async for _commit in client.generate(prompt, P7_REQUEST_ID, LlmOptions()):
                pass

        await asyncio.wait_for(drain(), timeout=P.P4_GENERATE_S)
    except Exception as exc:
        found = G._http_status(exc)
        http_status = found if found is not None else 0
        error = repr(exc)
        _say(f"probe P7: {exc!r}")
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
    healthy = await asyncio.to_thread(T._healthy, base_url)
    parts = [p for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]
    bare_phashes = {p.phash for m in bare.messages for p in m.content if isinstance(p, ImagePart)}
    return {
        "tokens": tokens,
        "limit": WINDOW_LIMIT,
        "tokenize_status": status,
        "tokens_live_frame": live,
        "tokens_without_image": without,
        "overview_tokens": None if tokens is None or without is None else tokens - without,
        "http_status": http_status,
        "healthy_after": healthy,
        "error": error,
        "images": len(parts),
        "overview_images": sum(1 for p in parts if p.phash not in bare_phashes),
        "done_ms": (client.generates[-1] if client.generates else {}).get("done_ms"),
    }


def probe_p7(base_url: str, model: str, detokenizer: Any) -> dict[str, Any]:
    """B2 P7 (T-0132 P4's rule): ``/tokenize`` of ``p7_prompts()``'s overview-frame step-1 request at 1120 against
    ``WINDOW_LIMIT``, then one generate on it and the engine's health; ``overview_tokens`` is the count less the same
    request's with no image, and ``tokens_live_frame`` the live default's step 1 (both report-only). ``main`` adds
    ``passed`` (``p7_ok``)."""
    return asyncio.run(_probe_p7(base_url, model, detokenizer))


# =====================================================================================================
# The turn runner
# =====================================================================================================


class _FrameWatch:
    """The live browser seam the session gets, over T-0134's feedback watch (every name delegated): records, per
    ``page_overview`` call id (the session's own capture ends ``FRAME_CALL_SUFFIX``), the outcome's status, action and
    ``resolved``; every frame ``overview_frame_of`` registered for the session (B4's overview identities); and, per
    other call id, the outcome's status and ``resolved`` and, after an executed call, the agent page's
    ``window.__state.cart`` (``cart_after``; a read-only ``page.evaluate`` bounded by ``CART_READ_S``)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: dict[str, dict[str, Any]] = {}
        self.frames: dict[str, Any] = {}
        self.executed: dict[str, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _cart(self) -> list[str] | None:
        try:
            state = await asyncio.wait_for(
                self._inner.driver.page.evaluate(G.STATE_JS), timeout=CART_READ_S
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
        if isinstance(state, Mapping) and isinstance(state.get("cart"), list):
            return [str(item) for item in state["cart"]]
        return None

    async def execute(self, inv: Any) -> Any:
        out = await self._inner.execute(inv)
        record = {
            "status": getattr(out, "status", None),
            "action": getattr(out, "action", None),
            "resolved": dict(getattr(out, "resolved", None) or {}),
        }
        if str(inv.name) == "page_overview":
            self.calls[str(inv.call_id)] = record
        else:
            if record["status"] == "executed":
                record["cart_after"] = await self._cart()
            self.executed[str(inv.call_id)] = record
        return out

    def overview_frame_of(self, key: Any) -> Any:
        frame = self._inner.overview_frame_of(key)
        if frame is not None:
            self.frames[str(key)] = frame
        return frame


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Any
    sentinel: bytes
    manifest: dict[str, Any]
    vram: Any
    arms: dict[str, dict[str, Any]]
    crop: int | None
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
    dict[str, Any],
]:
    """T-0134's turn at the live default (budget, ReGround with its pass-2 client as T-0132's, feedback, echo,
    recovery), the T-0142 guard on, tools v1, the auto overview off, and the arm's overview frame; the records read
    after the session closed. Returns the trial, its step lines, its reground lines, its raw box reads, its thoughts
    line and its bodies line."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE
    from local_astra.conductor.session import LiveFeedback, Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    row = run.arms[arm]
    budget = int(row["visual_tokens"])
    reground = bool(row["reground"])
    crop = run.crop if reground else None
    lf = row["live_feedback"]
    feedback = None if lf is None else LiveFeedback(**lf)
    echo = str(row["echo"])
    recover = bool(row["recover_thought_calls"])
    frame_on = bool(row["overview_frame"])
    want_tools = str(row["tools_sha256"])
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
    frame_watch: Any = None
    session: Any = None
    meter: Any = None
    crop_meter: Any = None
    tools_version: str | None = None
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
            echo=echo,  # explicit in both arms (B1)
        )
        watch = T._GroundingWatch(browser, INTENDED_TARGETS[task_id])
        feedback_watch = F._FeedbackWatch(watch)
        frame_watch = _FrameWatch(feedback_watch)
        # both arms: tools v1 (B1; never ``page_overview=True``)
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=frame_watch)
        tools_version = str(kit.tools_version)
        if tools_version != want_tools:
            raise RuntimeError(f"arm {arm}'s tool set is {tools_version}, not {want_tools}")
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        endpoint = VllmEndpoint(base_url=run.base_url)
        meter = LoopMeter(
            F._client(endpoint, detokenizer, budget),
            metrics=lambda: G.read_metrics(run.base_url),
            clock=time.monotonic,
        )
        second: Any = None
        if reground and crop is not None:
            if crop == budget:
                second = meter  # T-0132 B2: equal budgets pass the main client itself, as cli does
            else:
                crop_meter = LoopMeter(
                    F._client(endpoint, detokenizer, crop),
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
            tool_kit=kit,  # tools v1 in both arms
            tool_guidance=LIVE_TOOL_GUIDANCE,
            visual_tokens=budget,  # every setting explicit in both arms (B1)
            reground=reground,
            reground_llm=second,
            reground_visual_tokens=crop,
            live_feedback=feedback,
            recover_thought_calls=recover,
            degeneracy_guard=True,
            auto_overview=False,
            overview_frame=frame_on,  # the only difference between the arms (T-0151 A2-1)
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
    steps = [
        {**s, "reground_visual_tokens": crop}
        for s in frame_steps(
            F.feedback_step_lines(
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
            ),
            records,
            frame_watch.executed if frame_watch is not None else {},
        )
    ]
    regrounds = [
        {**g, "arm": arm, "reground_visual_tokens": crop}
        for g in T.reground_lines(
            task_id,
            repeat,
            records=records,
            generates=[g for m in meters for g in m.generates],
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
        visual_tokens=budget,
        reground=reground,
        reground_visual_tokens=crop,
        live_feedback=None if lf is None else dict(lf),
        echo=echo,
        recover_thought_calls=recover,
        degeneracy_guard=True,
        page_overview=False,
        auto_overview=False,
        overview_frame=frame_on,
        tools_version=tools_version,
        # the turn line's keys as the session logged them (``None`` when absent)
        overview_frame_logged=extras.get("overview_frame"),
        auto_overview_logged=extras.get("auto_overview"),
        page_overview_logged=extras.get("page_overview"),
        degeneracy_guard_logged=extras.get("degeneracy_guard"),
        recover_thought_calls_logged=extras.get("recover_thought_calls"),
        live_feedback_logged=extras.get("live_feedback"),
        visual_tokens_logged=extras.get("visual_tokens"),
        success_v2=score_v2(task_id, trial.get("state"), G._voiced_text(trial)),
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
        {"arm": arm, "task": task_id, "repeat": repeat, **c}
        for c in (watch.clicks.values() if watch is not None else [])
    ]
    thoughts = F.thought_record(arm, task_id, repeat, records) if records else None
    overviews = overview_records(
        frame_watch.calls if frame_watch is not None else {},
        frame_watch.frames if frame_watch is not None else {},
    )
    line = body_line(arm, task_id, repeat, raw, records, overviews)
    return trial, steps, regrounds, reads, thoughts, line


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``ERRORED_STOP`` consecutive errored turns, or past
    ``RUN_BUDGET_S`` since ``PROCESS_START_S`` (probes included), nothing more is sent and nothing more is written
    (fewer than 200 trial lines exactly when ``stopped`` is set)."""
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
        line: dict[str, Any] | None = None
        try:
            trial, steps, regrounds, reads, thoughts, line = await _one_turn(
                run, arm, task_id, repeat
            )
        except Exception as exc:
            trial = _not_run(
                {
                    **run.arms[arm],
                    "reground_visual_tokens": run.crop if run.arms[arm]["reground"] else None,
                },
                task_id,
                repeat,
                f"harness: {exc!r}",
            )
        errored = errored + 1 if T._errored(trial, steps, regrounds) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "regrounds.jsonl", regrounds)
        G._append_jsonl(run.out / "clicks.jsonl", reads)
        G._append_jsonl(run.out / "thoughts.jsonl", [thoughts] if thoughts is not None else [])
        G._append_jsonl(run.out / "bodies.jsonl", [line] if line is not None else [])
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"calls={[s.get('call_name') for s in steps]} "
            f"overview_shown={[s.get('overview_shown') for s in steps]} "
            f"frame_kind={[s.get('frame_kind') for s in steps]} "
            f"scrolled_px={[s.get('scrolled_px') for s in steps]} "
            f"overview_frame={trial.get('overview_frame_logged')} "
            f"success_v2={trial.get('success_v2')} end={trial.get('tool_loop_end')} "
            f"timed_out={trial.get('timed_out')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{arm} {task_id}/r{repeat}"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _write_report(
    out: Path,
    probes: Mapping[str, Any],
    manifest: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    stopped: str | None = None,
) -> dict[str, Any]:
    report = build_report(
        G._read_jsonl(out / "trials.jsonl"),
        G._read_jsonl(out / "steps.jsonl"),
        G._read_jsonl(out / "regrounds.jsonl"),
        G._read_jsonl(out / "bodies.jsonl"),
        probes,
        arms=arms,
        stopped=stopped,
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def _keywords_missing() -> str | None:
    """Why ``Session`` lacks a keyword the arms pass (T-0135's ``recover_thought_calls``, T-0142's
    ``degeneracy_guard``, T-0150's ``auto_overview``, T-0151's ``overview_frame``), then T-0134 group A's, or
    ``None``. Reads signatures only, at call time."""
    from local_astra.conductor import session as session_module

    params = inspect.signature(session_module.Session.__init__).parameters
    for name, task in (
        ("recover_thought_calls", "T-0135 A4"),
        ("degeneracy_guard", "T-0142"),
        ("auto_overview", "T-0150 A1; both arms pass the auto overview off"),
        ("overview_frame", "T-0151 A2-1; the overview-frame arm cannot be built"),
    ):
        if name not in params:
            return f"Session has no {name} keyword ({task})"
    missing = F._group_a_missing()
    return None if missing is None else str(missing)


def _cli_missing() -> str | None:
    """Why ``cli`` cannot give the arms (``feedback_of``, ``recovery_of``, ``page_overview_of``,
    ``auto_overview_of``, ``overview_frame_of``), or ``None``. Reads attributes only."""
    from local_astra import cli

    for name, task in (
        ("feedback_of", "T-0134c A2"),
        ("recovery_of", "T-0135b A2"),
        ("page_overview_of", "T-0140 A5: the page overview must be read off"),
        ("auto_overview_of", "T-0150 A6: the auto overview must be read off"),
        ("overview_frame_of", "T-0151 A2-9: the overview frame must be read off"),
    ):
        if not callable(getattr(cli, name, None)):
            return f"cli has no {name} ({task}); live_default cannot read the arms' settings"
    return None


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts ``subject`` followed by a non-word character is reachable from
    HEAD (the repo's ``Merge T-NNNN: ...`` convention)."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    pattern = re.compile(rf"^{re.escape(subject)}(?!\w)")
    return any(pattern.match(s) for s in subjects)


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
    out: Path = args.out_dir or REPO / "logs" / f"t0151-overview-frame-ab-{sha[:7]}"
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
    # 5. the session's keywords (T-0135's, T-0142's, T-0150's, T-0151's, T-0134 group A's) and cli's resolvers, and
    # the guidance
    missing = _keywords_missing() or _cli_missing()
    if missing is not None:
        _say(f"refusing: {missing}")
        return EXIT_REFUSED
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE

    guidance_sha = hashlib.sha256(LIVE_TOOL_GUIDANCE.encode("utf-8")).hexdigest()
    if guidance_sha != LIVE_TOOL_GUIDANCE_SHA256:
        _say(
            f"refusing: LIVE_TOOL_GUIDANCE's sha256 is {guidance_sha}, not {LIVE_TOOL_GUIDANCE_SHA256}"
        )
        return EXIT_REFUSED
    # 6. the live default, read from cli once (B1), and the arm table over it
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    feedback = cli.feedback_of(lean, argparse.Namespace(), "live")
    if str(feedback.live_feedback) not in FEEDBACK_CHOICES:
        _say(
            f"refusing: cli.feedback_of gives live_feedback {feedback.live_feedback!r}, not one of "
            f"{sorted(FEEDBACK_CHOICES)}"
        )
        return EXIT_REFUSED
    default = live_default(lean)
    bad = live_default_findings(default)
    if bad:
        _say(
            f"refusing: the live default {default} is not one this harness runs (the A/B compares the default, "
            f"page overview, auto overview and overview frame off, against it with the overview frame on): {bad}"
        )
        return EXIT_REFUSED
    if default["reground"]:
        _say(
            "refusing: the live default has ReGround on; Session(overview_frame=True) refuses it (T-0151 A2-1), "
            "so the overview-frame arm cannot be built at this default"
        )
        return EXIT_REFUSED
    grounding = cli.grounding_of(lean, argparse.Namespace(), "live")
    crop = (
        int(grounding.reground_visual_tokens)
        if default["reground"] and grounding.reground_visual_tokens is not None
        else None
    )
    rows = arm_rows(default)
    try:
        profile = P.arm_profile(
            lean, int(default["visual_tokens"]), bool(default["reground"]), crop
        )
    except ValueError as exc:
        _say(f"refusing: the arm profile is invalid: {exc}")
        return EXIT_REFUSED
    # 7. the tool kit both arms use (T-0115's refusals, repair off included; v1 in both arms)
    try:
        kit, detokenizer = cli.load_tool_kit(lean)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    for label in LABELS:
        if str(kit.tools_version) != TOOLS_SHA256[label]:
            _say(
                f"refusing: arm {label}'s tool set is {kit.tools_version}, not {TOOLS_SHA256[label]}"
            )
            return EXIT_REFUSED
        if kit.repair_off is not None:
            _say(
                f"refusing: arm {label}'s tool repair is off ({kit.repair_off}); the regime says repair on"
            )
            return EXIT_REFUSED
    # 8. Chromium (T-0111g B2's availability check; reads the filesystem only)
    absent = cli.live_prerequisites(BROWSER)
    if absent is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {absent}")
        return EXIT_REFUSED
    # 9. T-0131's pass-2 bound
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
        return _measure(
            args.base_url,
            out,
            sha,
            model,
            lean,
            profile,
            detokenizer,
            vram,
            guidance_sha,
            default,
            rows,
            crop,
            grounding,
            feedback,
        )
    finally:
        vram.stop()


def _measure(
    base_url: str,
    out: Path,
    sha: str,
    model: str,
    lean: Any,
    profile: Any,
    detokenizer: Any,
    vram: Any,
    guidance_sha: str,
    default: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    crop: int | None,
    grounding: Any,
    feedback: Any,
) -> int:
    """Past every refusal and the engine check: the manifest, P1-P3, P6, P7, P8, P9, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    arms = [dict(r) for r in rows]
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0151_overview_frame_ab.sh refuses while 127.0.0.1:8080/8081 are bound",
        "arms": arms,
        "reground_visual_tokens": crop,
        "commit": sha,
        "base_url": base_url,
        "served_model": model,
        "engine_version": G._get_text(G._root(base_url) + "/version"),
        "profile": lean.name,
        "canvas_length": lean.diffusion.canvas_length,
        "max_denoising_steps": lean.diffusion.max_denoising_steps,
        "thinking": lean.diffusion.thinking,
        "tools": {
            label: f"native:{TOOLS_SHA256[label][:12]}:live:loop{G.MAX_TOOL_STEPS}"
            for label in LABELS
        },
        "repair": "on",
        "max_tool_steps": G.MAX_TOOL_STEPS,
        "repeats": REPEATS,
        "live_tool_guidance_sha256": guidance_sha,
        "degeneracy_guard": True,
        "overview_frame_max_screens": OVERVIEW_FRAME_MAX_SCREENS,
        "browser": f"{BROWSER}, one per turn, fresh context, loopback start pages only",
        "viewport": list(profile.browser_viewport),
        "turns": len(order(ORDER_SEED)),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": G.TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "cart_read_s": CART_READ_S,
        "vram_period_ms": T.VRAM_PERIOD_MS,
        "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
        "no audio device, no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        "history": list(HISTORY),
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "live_default": dict(default),
        "live_default_grounding": _as_dict(grounding),
        "live_default_feedback": _as_dict(feedback),
        "arm_table_sha256": arm_table_sha(arms),
        "task_table_sha256": G.task_table_sha(TASKS),
        "t0134_merged": _merged("Merge T-0134"),
        "t0135_merged": _merged("Merge T-0135"),
        "t0142_merged": _merged("Merge T-0142"),
        "t0140_merged": _merged("Merge T-0140 group B"),
        "t0150_merged": _merged("Merge T-0150 group B"),
        # the locked script's rule (B6): only "Merge T-0151 group B" counts
        "t0151_merged": _merged("Merge T-0151 group B"),
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
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in (
        "trials.jsonl",
        "steps.jsonl",
        "regrounds.jsonl",
        "clicks.jsonl",
        "thoughts.jsonl",
        "bodies.jsonl",
        "turns.jsonl",
    ):
        (out / name).touch()

    # --- B2: P1-P3 (T-0131's), P6, P7, P8 and P9, before any arm turn -------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name]["ok"])}
    try:
        p6 = dict(probe_p6())
    except Exception as exc:
        p6 = {"error": repr(exc)}
    probes["p6"] = {**p6, "passed": p6_ok(p6)}
    try:
        p7 = dict(probe_p7(base_url, model, detokenizer))
    except Exception as exc:
        p7 = {"error": repr(exc)}
    probes["p7"] = {**p7, "passed": p7_ok(p7)}
    try:
        p8 = dict(probe_p8())
    except Exception as exc:
        p8 = {"error": repr(exc)}
    probes["p8"] = {**p8, "passed": p8_ok(p8)}
    try:
        p9 = dict(probe_p9())
    except Exception as exc:
        p9 = {"error": repr(exc)}
    probes["p9"] = {**p9, "passed": p9_ok(p9)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(_passed(probes, p) for p in PROBES):
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest, arms, stopped)
        _say(
            f"a probe failed: the run is void ({json.dumps({k: v.get('passed') for k, v in probes.items()})}); "
            f"label {report['decision']['label']}"
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
        arms={str(r["label"]): r for r in arms},
        crop=crop,
    )
    try:
        asyncio.run(_run_all(run))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    if run.stopped is not None:
        manifest["stopped"] = run.stopped
        G._write_json(out / "manifest.json", manifest)

    # --- the report straight after the arms, before anything else can fail ----------------------------
    report = _write_report(out, probes, manifest, arms, run.stopped)
    _say(f"decision={report['decision']['label']} ({report['decision']['reason']})")
    _say(f"written to {out}")
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


def _as_dict(value: Any) -> dict[str, Any] | str:
    """A ``cli`` resolver's result with each value's source (``repr`` when it is not a dataclass)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return repr(value)


if __name__ == "__main__":
    sys.exit(main())
