"""T-0140 B -- the locked 2-arm page-overview A/B of ``tools = live`` on S4: 200 typed turns through a real ``Session``.
``/launch`` ONLY.

    uv run python scripts/gpu/t0140_overview_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the live default (``live_default``), the arm table
(``arm_rows``, ``arm_table_sha``, ``order``), the probe rules (``p6_ok``, ``p7_ok``), B4's findings
(``configuration_findings``), B3's counts (``arm_metrics``), B5's inputs and decision (``decision_inputs``,
``decide``, ``decide_with_reason``), the report (``build_report``, ``render_markdown``) and the record extraction
(``body_line``, ``overview_records``) are pure or read ``cli`` only; everything engine- or browser-facing is imported
inside ``main``, the probes and the turn runner.

B1: T-0134's harness (``scripts/gpu/t0134_feedback_ab.py``) is **imported, never copied**, and through it T-0132's,
T-0131's and T-0111g's: ``TASKS``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
``non_loopback_sockets``, ``turn_records`` (T-0111g's objects, ``G``), ``probe_p1``, ``run_probes``, ``false_claim``,
``click_score``, ``config_evidence`` (T-0131's, ``T``), ``wilson``, ``newcombe`` (T-0132's, ``P``) and ``score_v2``,
``over_click``, ``no_see``, ``coord_mismatch``, ``false_claim_v2`` (T-0134's, ``F``), plus T-0131's per-arm section,
box-reading watch, record extraction, VRAM sampler and engine helpers, T-0132's arm profile, C6b first-failing rule,
regime reader and P4 frames, and T-0134's feedback watch, client, step lines, thought record and hostile fixture.

Arms (B1; never hard-coded, the T-0135 B pattern): both arms run at HEAD's live default, read from ``cli`` once per
``main`` (``live_default``): the budget and ReGround from ``cli.grounding_of``, the live feedback and the executor echo
from ``cli.feedback_of``, T-0135's recovery from ``cli.recovery_of`` and the overview from ``cli.page_overview_of``
(off: a default with the overview already on is refused). ``A`` runs the kit ``cli.load_tool_kit(profile, ...)``
(tools v1, ``bbfc53ccdd80``) and ``B`` ``cli.load_tool_kit(profile, ..., page_overview=True)`` (``tools_v1_overview.json``,
``8048bdb7aabe``); every setting is passed explicitly in both, ``degeneracy_guard=True`` included (T-0142). The
default is recorded in the manifest (``live_default``), with its two rows (``regime.arms``) and their sha
(``arm_table_sha256``).

Refusals (exit 2), all **before any socket** opens and before any browser, page server or ``nvidia-smi``: T-0134's
(a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task table off its pin,
T-0134 group A's keywords absent, ``LIVE_TOOL_GUIDANCE`` off its sha, a tool kit that cannot load / has repair off,
Chromium absent, the session's ``REGROUND_TIMEOUT_MS`` off T-0131's 4000 ms), plus: ``Session`` without the
``recover_thought_calls`` or ``degeneracy_guard`` keyword, a ``cli`` without ``feedback_of``, ``recovery_of`` or
``page_overview_of``, a ``load_tool_kit`` without the ``page_overview`` keyword, a live default that is not well formed
or has the overview on, ``tools_v1_overview.json`` off its pin, and the tools check per arm (arm A's kit not v1, arm B's
not the overview set). Past them the engine is checked (``/v1/models`` and ``/metrics``): with no engine the harness
exits 1 having launched no browser.

Probes (B2), before any arm turn: P1-P3 are T-0131's ``run_probes``; P6 (browser only, no inference) is
``probe_p6``: through a real headless ``BrowserToolExecutor`` at 1280x800, ``page_overview`` on ``/demo/deals.html``
(``screens >= 1.3``, the image within 1280x800, ``scroll_y`` unchanged), on a page that fits (``fits``) and on T-0134's
hostile fixture (its true 2800 px); P7 is ``probe_p7``: a ``/tokenize`` count, as T-0132 P4 measured it, of the
step-2 request of an S4-like turn carrying one overview (one history image, the current frame and the overview: 3
images), against ``16384 - 256``, then one generate on it and the engine's health; the overview's own token cost
(``overview_tokens``, the count less the same request's without it) is report-only. P1 failing: inconclusive; P2, P3,
P6 or P7: void; exit 3, a report, no arm turn.

What it runs (*Regime*): ``order(ORDER_SEED)``, 200 ``(arm, task, repeat)`` turns (2 arms x 5 tasks x ``REPEATS``)
shuffled jointly, concurrency 1, the engine unseeded. Each turn is T-0134's turn (one fresh start-page server, one
fresh headless ``BrowserToolExecutor(echo=...)`` behind T-0131's box-reading watch, T-0134's feedback watch and this
file's overview watch, one fresh ``Session`` with one ``UserText``) at the live default and the arm's kit. The run
stops after ``ERRORED_STOP`` consecutive errored turns, when ``RUN_BUDGET_S`` has passed since ``PROCESS_START_S``
(from process start, probes included), or when the turn runner raised; turns not run are **not** written, so
``trials.jsonl`` has fewer than 200 lines exactly when ``stopped`` is set. ``report.json`` and ``report.md`` are
written straight after the arms, before anything else can fail.

Output (``--out-dir``, default ``logs/t0140-overview-ab-<sha7>/``): ``probes.json`` (P1-P3, P6, P7), ``trials.jsonl``
(T-0134's fields plus ``recover_thought_calls``, ``degeneracy_guard``, ``page_overview``, ``tools_version``,
``page_overview_logged`` (A10's turn-line key) and ``degeneracy_guard_logged`` (T-0142's)), ``steps.jsonl`` (T-0134's
step fields plus ``overview_shown``: the step's ``llm.request`` record carried A10's ``overview`` key),
``regrounds.jsonl``, ``bodies.jsonl`` (one line per turn: every body the meters forwarded, image parts as sha16, and
the turn's overviews), ``thoughts.jsonl``, ``clicks.jsonl``, ``turns.jsonl`` and ``sessions/*.jsonl``,
``manifest.json``, ``report.json`` and ``report.md``. Exit codes: 0 the run was measured; 1 no engine, or the run
stopped early; 2 refused; 3 a probe failed (void or inconclusive).

The label (B5, H-T140-1 approved as proposed 2026-10-23): "inconclusive" when P1 failed; "void" on any configuration
finding (B4, reason ``configuration: <finding>``) or when P2, P3, P6 or P7 failed; "incomplete" when the run stopped
early; "inconclusive" when any arm is ineligible (acceptance over the trials that are neither an engine error nor a
``degenerate`` end, safety, or more than 2 engine errors; the reason names the arm, the clause and the first failing
trial); then "overview" iff S4(B) - S4(A) >= 5, success(B) >= success(A) - 3, non-S4 false claims(B) <= (A) + 2 and
over-clicks(B) <= (A) + 2; else "overview unused" when arm B's S4 overview-call turns are under 5, else "no change".
Engine-error and ``degenerate`` trials count as failed trials in every metric.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native, live, repair on, loop 8, ``LIVE_TOOL_GUIDANCE`` 768a973b; arm A tools v1 ``bbfc53ccdd80``,
arm B ``tools_v1_overview.json`` ``8048bdb7aabe``; both arms at HEAD's live default (recorded), the T-0142 guard on;
headless Chromium at 1280x800, DSF 1, loopback start pages only; typed turns, fake STT/TTS/audio, no headset or
speakers; 2 arms x 5 tasks x n = 20 = 200 turns, order seed 0, concurrency 1, the engine unseeded; STT down. History,
not like-for-like: T-0134 at e59aa69 (arm B 64/100, S4 0/20).
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
T0134_MODULE_NAME = "_t0134_feedback_ab_for_t0140"


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
#: B1: what each arm sets ``page_overview`` to (the only difference between the arms)
OVERVIEW_ON = {"A": False, "B": True}
#: A1: ``tools_v1.json`` (arm A) and ``tools_v1_overview.json`` (arm B), full literals (never derived)
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OVERVIEW_SHA256 = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
TOOLS_SHA256 = {"A": V1_SHA256, "B": OVERVIEW_SHA256}
#: B1: the live-default settings both arms carry, read from ``cli`` at run time (never hard-coded)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo", "recover_thought_calls")
#: B1: per (arm, task); 2 arms x 5 tasks x 20 = 200 turns
REPEATS = 20
#: *Regime*: the 200 turns shuffled jointly with seed 0
ORDER_SEED = 0
#: B3's fixed denominators (the differences use them whatever ran)
TURNS_PER_ARM = 100
S4_TURNS = 20
#: B6: the wall-time budget, measured from ``PROCESS_START_S`` (probes included), inside the locked script's
#: ``timeout -k 60s 75m`` so the report is still written (estimate 30-40 min: T-0132 ran 300 turns in 42 min)
RUN_BUDGET_S = 62 * 60
#: B5 (H-T140-1, approved as proposed 2026-10-23): boundaries compare with this tolerance (an exact margin passes)
EPS = 1e-9
S4_MARGIN = 5
SUCCESS_MARGIN = 3
FALSE_CLAIM_MARGIN = 2
OVER_CLICK_MARGIN = 2
#: B5: an arm is ineligible on the engine clause only above this many engine-error trials
ENGINE_ERRORS_MAX = 2
#: B5: ``overview unused`` when arm B's S4 turns with a ``page_overview`` call are fewer than this
UNUSED_BELOW = 5
#: B5's labels
OVERVIEW = "overview"
UNUSED = "overview unused"
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
#: A3's constants and strings (the executor's), typed from docs/tasks/T-0140.md
MAX_OVERVIEW_SCREENS = 4
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
#: A7's label stem (the session's ``OVERVIEW_LABEL`` starts "[" then this)
OVERVIEW_STEM = "Whole page, scaled to fit one screen"
#: A10's counts on the turn line
A10_KEYS = ("calls", "shown", "fits", "refused", "failed")
#: B3 (report-only, B-9): a voiced text that mentions the overview
NARRATION = re.compile(r"\boverview\b|\bwhole page\b", re.IGNORECASE)
#: B2 P6
P6_PAGE = "/demo/deals.html"
P6_MIN_SCREENS = 1.3
#: T-0134 P5's tolerance (1 CSS px) on the hostile fixture's true height
P6_TOLERANCE_PX = 1.0
P6_CALL_S = 30.0
#: P6's page that fits: one fixed 400 CSS px block, served with T-0134's hostile page from this file's own loopback
#: server (no script, no ``vh`` unit)
FIT_PATH = "/t0140_fits.html"
FIT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>fits</title>
<style>html, body { margin: 0; padding: 0; } .block { height: 400px; background: #dfe7ef; }</style>
</head><body><div class="block"></div></body></html>
"""
#: B2 P7: the overview frame's size in the probe's request (O-A1-11: deals' overview is 838x798)
P7_OVERVIEW_SIZE = (838, 798)
P7_SCREENS = 1.9
P7_REQUEST_ID = "t0140-probe-p7"
P7_FIRST_TEXT = "Let me look at the whole page. "

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
    "T-0134 live-feedback A/B at e59aa69 (n = 100 per arm, order seed 0): arm B 64/100 by score_v2; S4 0/20 in every "
    "arm, with 0 scrolls (the thoughts list the six visible deals and pick the $11.50 enamel mug)",
    "T-0132 crop-budget A/B at 2ccff41 (n = 100 per arm, order seed 0): S4 0/20 in every arm",
)
#: B5, stated in the report
POWER = (
    "5/20 vs 0/20 is Fisher p ~ 0.047; at n = 100 and p ~ 0.65 the standard error of success is about 4.8 pp; the "
    "false-claim and over-click margins of +2 are non-inferiority guards for an action that never clicks."
)


def _say(msg: str) -> None:
    print(f"t0140_overview_ab: {msg}", file=sys.stderr, flush=True)


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
    ``recovery_of``'s ``thought_call_recovery`` and ``page_overview_of``'s ``page_overview``."""
    from local_astra import cli

    ns = argparse.Namespace()
    g = cli.grounding_of(profile, ns, "live")
    fb = cli.feedback_of(profile, ns, "live")
    rec = cli.recovery_of(profile, ns, "live")
    ov = cli.page_overview_of(profile, ns, "live")
    label, extent = FEEDBACK_CHOICES[str(fb.live_feedback)]
    return {
        "visual_tokens": None if g.visual_tokens is None else int(g.visual_tokens),
        "reground": bool(g.reground),
        "live_feedback": {"screen_label": label, "page_extent": extent},
        "echo": str(fb.action_echo),
        "recover_thought_calls": rec.thought_call_recovery,
        "page_overview": ov.page_overview,
    }


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """Why a live default is not one this harness can run (each key present, each value of a type the session takes,
    the overview off), or ``[]``."""
    keys = (*DEFAULT_KEYS, "page_overview")
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
    if default["page_overview"] is not False:
        out.append(
            f"page_overview {default['page_overview']!r}: the overview is already the default"
        )
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: the two arms over a live default: its five settings and the T-0142 guard in both; A without the overview
    on tools v1, B with it on the overview set."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {
            "label": label,
            **shared,
            "degeneracy_guard": True,
            "page_overview": OVERVIEW_ON[label],
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
# A3/A7's strings (pure)
# =====================================================================================================


def overview_action(screens: float, clipped: bool) -> str:
    """A3's executed action string."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def overview_label(screens: float, clipped: bool) -> str:
    """A7's ``OVERVIEW_LABEL`` for one overview."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return (
        f"[{OVERVIEW_STEM}: about {screens:.1f} screens tall{c}. "
        "The next image is the browser screen.]"
    )


# =====================================================================================================
# B2: the P6 and P7 rules (pure)
# =====================================================================================================


def p6_ok(p6: Mapping[str, Any]) -> bool:
    """B2 P6 from its record: at 1280x800, ``deals`` showed an overview of at least 1.3 screens within the viewport
    with A3's action string and ``scroll_y`` unchanged; ``fits`` gave A3's fits string; ``hostile`` reported T-0134's
    fixture's true height (2800 CSS px, within 1 px), not clipped, with A3's string. A missing part fails."""
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
    """B2 P7 from its record (T-0132 P4's rule): a ``/tokenize`` count within ``WINDOW_LIMIT`` (16128) of the step-2
    request carrying exactly one overview and at most 4 images, the generate on it answering 200 and the engine
    healthy after."""
    tokens = p7.get("tokens")
    images = p7.get("images")
    return (
        isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and p7.get("limit") == WINDOW_LIMIT
        and tokens <= WINDOW_LIMIT
        and p7.get("http_status") == 200
        and p7.get("healthy_after") is True
        and p7.get("overview_images") == 1
        and isinstance(images, int)
        and not isinstance(images, bool)
        and images <= MAX_IMAGES
    )


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


def _overview_placed(
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    last_user: int | None,
    i: int,
    j: int,
    overview: Mapping[str, Any],
) -> bool:
    """B4/A7 (B-4): the overview image at ``messages[i].parts[j]`` is where ``overview`` (one record, so one call's
    step) puts it: in the final user message of the ``tool_step`` request one past its call's step, or of a
    ``tool_repair`` of that request's call (it continues the pass-1 prompt, review F2), followed by its own label and
    then the current frame (that message's last image)."""
    if request.get("phase") not in ("tool_step", "tool_repair"):
        return False
    if request.get("step") != int(overview["step"]) + 1:
        return False
    if i != last_user:
        return False
    parts = list(messages[i].get("parts") or [])
    images = [k for k, p in enumerate(parts) if "image" in p]
    want = overview_label(float(overview["screens"]), bool(overview["clipped"]))
    return (
        j + 2 < len(parts)
        and parts[j + 1] == {"text": want}
        and "image" in parts[j + 2]
        and images[-1] == j + 2
    )


def _body_findings(trial: Mapping[str, Any], line: Mapping[str, Any]) -> list[str]:
    """B4 over one turn's bodies: the tool set each arm declares, the image count, the overview's count and place,
    and the overview label in user messages only (O-17: the model's own words reach later bodies as assistant
    text)."""
    out: list[str] = []
    tid, arm = _tid(trial), str(trial["arm"])
    # review F1: repeated calls on an unchanged page share a sha, so a sha maps to every record carrying it
    overviews: dict[str, list[Mapping[str, Any]]] = {}
    for o in line.get("overviews") or []:
        overviews.setdefault(str(o["sha"]), []).append(o)
    for r in line.get("requests") or []:
        where = f"{tid} {r.get('label')}"
        tools = r.get("tools")
        if arm == "A" and tools is not None and "page_overview" in tools:
            out.append(f"{where}: an arm-A body declares page_overview")
        if arm == "B" and (
            (tools is not None and "page_overview" not in tools)
            or (tools is None and r.get("phase") in ("generate", "tool_step"))
        ):
            out.append(f"{where}: an arm-B body does not declare page_overview")
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
            out.append(f"{where}: {len(mine)} overview images")
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user = users[-1] if users else None
        placed: set[tuple[int, int]] = set()
        for i, j, sha in mine:
            if any(_overview_placed(r, messages, last_user, i, j, o) for o in overviews[sha]):
                placed.add((i, j + 1))
            else:
                calls = sorted(int(o["step"]) for o in overviews[sha])
                out.append(f"{where}: an overview image out of place (the step-{calls} call)")
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue
            for j, p in enumerate(m.get("parts") or []):
                text = p.get("text")
                if isinstance(text, str) and OVERVIEW_STEM in text and (i, j) not in placed:
                    out.append(f"{where}: an overview label out of place")
    return out


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    bodies: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
) -> list[str]:
    """B4, one string per finding, any of which voids the run, each naming the trial (``"<arm> <task>#<repeat>"``)
    or the setting: the arm rows differing in a setting other than the overview and its tool set, or off B1 (A
    without the overview on v1, B with it on the overview set, the guard on in both); any reground line (ReGround is
    off in both arms); and in a trial that forwarded a body (O-11, as T-0134) a non-1120 body, its feedback, echo or
    guard off its arm's row, or a body finding (``_body_findings``). ``steps`` is not read (B4 is about what was
    sent)."""
    del steps
    out: list[str] = []
    rows = [dict(r) for r in arms]
    for k in sorted({k for r in rows for k in r} - {"label", "page_overview", "tools_sha256"}):
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    for r in rows:
        label = str(r.get("label"))
        if label in OVERVIEW_ON and (r.get("page_overview"), r.get("tools_sha256")) != (
            OVERVIEW_ON[label],
            TOOLS_SHA256[label],
        ):
            out.append(
                f"arm {label} row: page_overview {r.get('page_overview')!r}, tools {r.get('tools_sha256')!r}"
            )
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
        for k in ("live_feedback", "echo", "degeneracy_guard"):
            if t.get(k) != row.get(k):
                out.append(f"{tid}: {k} {t.get(k)!r}, its row says {row.get(k)!r}")
        out += logged_setting_findings(t, row)
        out += _body_findings(t, by_bodies.get(_key(t)) or {})
    return out


def _wrote_no_turn_line(trial: Mapping[str, Any]) -> bool:
    """T-0135b A7: a trial that hit the turn timeout or never had a session wrote no turn line."""
    return trial.get("timed_out") is True or trial.get("session_id") is None


def logged_setting_findings(trial: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """B4 "feedback, echo or guard settings off their rows", judged on the turn line as the session logged it (review
    F3; the trial's own setting fields are copies of the row): ``live_feedback_logged`` is the row's feedback plus its
    echo (``None`` when the row has none), ``recover_thought_calls_logged`` and ``visual_tokens_logged`` the row's,
    and the guard logged on. A trial that wrote no turn line is not judged."""
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


def engine_error(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5 (B-1): a trial that timed out, whose engine was down after it, or one of whose step attempts (retries
    included) did not answer HTTP 200 -- never a ``degenerate`` end, even when its aborted attempt shows HTTP 0."""
    if is_degenerate(trial):
        return False
    if trial.get("timed_out") is True or trial.get("engine_up_after") is False:
        return True
    return any(a.get("http_status") != 200 for s in steps for a in _attempts(s))


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


def arm_metrics(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B3's counts for one arm's records (a turn counts once in every turn count): success by ``score_v2`` (never on
    a forced failure) with its Wilson interval, per task, S4 with its interval, false claims (total and non-S4),
    over-clicks, S4 turns with a ``page_overview`` call and S4 turns with an executed scroll, overview calls per task,
    A10's counts summed, narration (voiced text mentioning the overview), ``degenerate`` ends, engine errors, and the
    step latency (``done_ms``) of steps whose request carried an overview against the others."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    forced = {_turn(t): forced_failure(t, by[_turn(t)]) for t in trials}
    ok = {
        _turn(t): (not forced[_turn(t)])
        and bool(score_v2(str(t["task"]), t.get("state"), G._voiced_text(t)))
        for t in trials
    }
    s4 = [t for t in trials if t["task"] == "S4"]
    fc = [t for t in trials if _false_claim(t, by[_turn(t)], ok[_turn(t)])]
    per_task = {task: sum(1 for (tt, _), v in ok.items() if tt == task and v) for task in TASK_IDS}
    calls = {
        task: sum(
            1
            for t in trials
            if t["task"] == task
            for s in by[_turn(t)]
            if s.get("call_name") == "page_overview"
        )
        for task in TASK_IDS
    }
    counts = dict.fromkeys(A10_KEYS, 0)
    for t in trials:
        logged = t.get("page_overview_logged")
        if isinstance(logged, Mapping):
            for k in A10_KEYS:
                counts[k] += int(logged.get(k) or 0)
    shown = [
        s["done_ms"]
        for s in steps
        if s.get("overview_shown") is True and s.get("done_ms") is not None
    ]
    other = [
        s["done_ms"]
        for s in steps
        if s.get("overview_shown") is not True and s.get("done_ms") is not None
    ]
    success = sum(1 for v in ok.values() if v)
    return {
        "success": success,
        "success_ci": _ci(success, len(trials)),
        "per_task": per_task,
        "s4": per_task["S4"],
        "s4_ci": _ci(per_task["S4"], len(s4)),
        "false_claims": len(fc),
        "false_claims_non_s4": sum(1 for t in fc if t["task"] != "S4"),
        "over_clicks": sum(1 for t in trials if over_click(str(t["task"]), t.get("state"))),
        "s4_overview_call_turns": sum(
            1 for t in s4 if any(s.get("call_name") == "page_overview" for s in by[_turn(t)])
        ),
        "s4_scrolls": sum(
            1
            for t in s4
            if any(
                s.get("executed_name") == "scroll" and s.get("status") == "executed"
                for s in by[_turn(t)]
            )
        ),
        "overview_calls": calls,
        "overview_counts": counts,
        "narration": sum(1 for t in trials if NARRATION.search(G._voiced_text(t))),
        "degenerate": sum(1 for t in trials if is_degenerate(t)),
        "engine_errors": sum(1 for t in trials if engine_error(t, by[_turn(t)])),
        "latency": {"overview": G._dist(shown), "other": G._dist(other)},
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
    """One arm's B5 inputs: acceptance (T-0111g's per trial, over the trials that are neither an engine error nor a
    ``degenerate`` end; B-1), the safety sum (T-0131's), the engine errors, B3's counts, and per clause the first
    failing trial ``"<task>#<repeat>"`` (acceptance among those trials, safety by T-0132's C6b rule, engine the first
    engine-error trial)."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    m = arm_metrics(trials, steps)
    plain = [t for t in trials if not forced_failure(t, by[_turn(t)])]
    failing = [t for t in plain if not G._turn_ok(t, by[_turn(t)])]
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
        "false_claims_non_s4": m["false_claims_non_s4"],
        "over_clicks": m["over_clicks"],
        "s4_overview_call_turns": m["s4_overview_call_turns"],
        "first_failing": {
            "acceptance": tid(failing[0]) if failing else None,
            "safety": P._first_failing(trials, steps, regrounds)["safety"],
            "engine": tid(errors[0]) if errors else None,
        },
    }


# =====================================================================================================
# B5: the decision (H-T140-1)
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
    ``configuration: <first>``; P2, P3, P6 or P7 failed (``void``) -> void; stopped early -> incomplete; an arm
    ineligible -> inconclusive naming the arm, the clause and the first failing trial; then ``overview`` iff every
    adoption clause holds; else ``overview unused`` when arm B's S4 overview-call turns are under 5, else ``no
    change``. ``inputs`` maps ``"A"``/``"B"`` to ``decision_inputs``'s shape."""
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
    )
    if adopt:
        return OVERVIEW, None
    if int(b["s4_overview_call_turns"]) < UNUSED_BELOW:
        return UNUSED, None
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
    (``success_v1``, ``false_claims_v1``) and its step latency as ``latency_v1`` (``latency`` is B3's overview split),
    then B3's counts, the arm's settings and the intervals."""
    base = _t0131_section(trials, steps, regrounds)
    base["success_v1"] = base.pop("success")
    base["false_claims_v1"] = base.pop("false_claims")
    base["latency_v1"] = base.pop("latency")
    m = arm_metrics(trials, steps)
    n_oc = sum(1 for t in trials if t["task"] in ("S1", "S2", "S5"))
    n_non_s4 = sum(1 for t in trials if t["task"] != "S4")
    return {
        **base,
        **m,
        "label": row["label"],
        "page_overview": row["page_overview"],
        "tools_sha256": row["tools_sha256"],
        "live_feedback": row["live_feedback"],
        "echo": row["echo"],
        "recover_thought_calls": row["recover_thought_calls"],
        "degeneracy_guard": row["degeneracy_guard"],
        "n": len(trials),
        "s4_n": sum(1 for t in trials if t["task"] == "S4"),
        "false_claims_ci": _ci(m["false_claims"], len(trials)),
        "false_claims_non_s4_n": n_non_s4,
        "over_clicks_n": n_oc,
        "over_clicks_ci": _ci(m["over_clicks"], n_oc),
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
    """``report.json``: ``void`` (P1, P2, P3, P6 or P7 failed), ``stopped``, the probes, every arm's section, B3's
    Newcombe differences (B-A on S4 of 20 and on success of 100), B4's ``configuration_findings`` and the decision
    (each arm's B5 inputs with their first failing trials)."""
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
    }
    findings = configuration_findings(trials, steps, regrounds, bodies, arms)
    p1_passed = _passed(probes, "p1")
    void = not all(_passed(probes, p) for p in ("p1", "p2", "p3", "p6", "p7"))
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


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reason, the power statement, the probes, every arm's numbers,
    the differences, and the history (not like-for-like)."""
    lines = ["# T-0140 B page-overview A/B (report-only, plus H-T140-1's label rule)", ""]
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
        "## Decision (B5, H-T140-1 approved as proposed)",
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
            f"non-S4 false claims {i['false_claims_non_s4']}, over-clicks {i['over_clicks']}, S4 overview-call turns "
            f"{i['s4_overview_call_turns']}; first failing {i['first_failing']}"
        )
    lines += [f"- power: {POWER}", "", "## Probes (B2)", ""]
    probes = report["probes"]
    p1 = probes.get("p1") or {}
    for budget, row in sorted((p1.get("budgets") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"- P1 {budget}: frame delta {_fmt(row.get('frame_delta'))}, crop delta {_fmt(row.get('crop_delta'))}, "
            f"expected {row.get('expected')} (+0..{T.P1_DELIMITERS}), ok {row.get('ok')}"
        )
    p2, p3 = probes.get("p2") or {}, probes.get("p3") or {}
    p6, p7 = probes.get("p6") or {}, probes.get("p7") or {}
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
        f"- P7 (step-2 request with one overview): /tokenize {_fmt(p7.get('tokens'))} of {_fmt(p7.get('limit'))}, "
        f"images {_fmt(p7.get('images'))} (overview {_fmt(p7.get('overview_images'))}), overview tokens "
        f"{_fmt(p7.get('overview_tokens'))} (report-only), HTTP {_fmt(p7.get('http_status'))}, healthy after "
        f"{_fmt(p7.get('healthy_after'))}, passed {_fmt(p7.get('passed'))}",
        "",
        "## Arms (B3; each number report-only)",
        "",
        "| arm | overview | S4 v2 (Wilson 95 %) | success v2 (Wilson 95 %) | false claims (non-S4) | over-clicks | "
        "S4 overview-call turns | S4 scrolls | degenerate | engine errors | min free VRAM MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        arm = report["arms"][label]
        lines.append(
            f"| {label} | {arm['page_overview']} | {arm['s4']}/{arm['s4_n']} {_interval(arm['s4_ci'])} | "
            f"{arm['success']}/{arm['n']} {_interval(arm['success_ci'])} | {arm['false_claims']} "
            f"({arm['false_claims_non_s4']}/{arm['false_claims_non_s4_n']}) | {arm['over_clicks']}/"
            f"{arm['over_clicks_n']} | {arm['s4_overview_call_turns']} | {arm['s4_scrolls']} | "
            f"{arm['degenerate']} | {arm['engine_errors']} | {_fmt(arm['vram_min_free_mib'])} |"
        )
    lines += ["", "## Differences (Newcombe 95 %, method 10)", ""]
    lines += [f"- {k}: {_interval(v)}" for k, v in report["differences"].items()]
    for label in LABELS:
        arm = report["arms"][label]
        lat = arm["latency"]
        lines += [
            "",
            f"### {label} (page_overview {arm['page_overview']}, tools {str(arm['tools_sha256'])[:12]}, live_feedback "
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
            "- overview calls per task: "
            + ", ".join(f"{k} {v}" for k, v in arm["overview_calls"].items()),
            "- A10 counts: " + ", ".join(f"{k} {v}" for k, v in arm["overview_counts"].items()),
            f"- turns whose voiced text mentions the overview: {arm['narration']}",
            f"- step done ms, overview-bearing: {T._dist_cell(lat['overview'])}; others: "
            f"{T._dist_cell(lat['other'])}",
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
    steps: Sequence[Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One overview per executed ``page_overview`` whose frame the session read (``overview_of``): the step line of
    the call, sha16 of the frame's JPEG, A3's ``screens``/``clipped`` and the frame's size."""
    out: list[dict[str, Any]] = []
    for s in steps:
        call = calls.get(str(s.get("call_id"))) if s.get("call_id") is not None else None
        if call is None or call.get("status") != "executed":
            continue
        resolved = call.get("resolved") or {}
        frame = frames.get(str(resolved.get("overview")))
        if frame is None:
            continue
        out.append(
            {
                "step": int(s["step"]),
                "sha": _sha16(frame.jpeg),
                "screens": float(resolved["screens"]),
                "clipped": bool(resolved["clipped"]),
                "width": int(frame.width),
                "height": int(frame.height),
            }
        )
    return out


def overview_steps(
    steps: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The step lines with ``overview_shown``: ``True`` iff the step's own ``llm.request`` record carries A10's
    ``overview`` key, read from the session log, never from the arm."""
    shown = {
        str((r.get("data") or {}).get("attempt_id"))
        for r in records
        if r.get("kind") == "llm.request" and (r.get("data") or {}).get("overview") is not None
    }
    return [
        {
            **s,
            "overview_shown": s.get("attempt_id") is not None and str(s.get("attempt_id")) in shown,
        }
        for s in steps
    ]


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
        tools_version=None,
        page_overview_logged=None,
        degeneracy_guard_logged=None,
        recover_thought_calls_logged=None,
        live_feedback_logged=None,
        visual_tokens_logged=None,
    )
    return trial


# =====================================================================================================
# B2: P6 (browser only; loopback only; no inference) and P7 (one /tokenize, one generate)
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
            target=self._server.serve_forever, name="t0140-probe-pages", daemon=True
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


async def _overview_once(origin: str, url: str) -> dict[str, Any]:
    """Through a fresh headless ``BrowserToolExecutor`` at ``VIEWPORT`` (``origin`` the only allowed one): the
    driver's ``scroll_y`` (CDP), one ``page_overview``, the frame ``overview_of`` keeps, and ``scroll_y`` again."""
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
        await asyncio.wait_for(ex.open(url), timeout=P6_CALL_S)
        before = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=P6_CALL_S)
        inv = ToolInvocation(
            call_id="t0140-p6-overview",
            turn_id="t0140-p6",
            name="page_overview",
            arguments={},
            source="draft",
            frame=None,
        )
        out = await asyncio.wait_for(ex.execute(inv), timeout=P6_CALL_S)
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
    """B2 P6 (browser only, no inference): ``page_overview`` through the real executor at 1280x800 on
    ``/demo/deals.html`` (a fresh start-page server), on ``FIT_PAGE`` and on T-0134's hostile page (this file's own
    loopback server). A part that fails is ``None`` with the error; ``main`` adds ``passed`` (``p6_ok``)."""
    return asyncio.run(_probe_p6())


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


def p7_prompts() -> tuple[Any, Any]:
    """B2 P7's step-2 request of an S4-like turn, with and without the overview: the live system message (the
    overview set's native tools field and ``LIVE_TOOL_GUIDANCE``, thinking on), one history turn with its image, S4's
    request with the current 1280x800 frame, step 1's ``page_overview`` call and its response as the continuation, and
    (A7) the overview placed right before the current frame with its label."""
    from local_astra.conductor.prompt_builder import LIVE_TOOL_GUIDANCE, PromptBuilder
    from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT, OVERVIEW_LABEL
    from local_astra.contracts.perception import Frame, Source
    from local_astra.contracts.turn import HistoryTurn
    from local_astra.toolcall import loop, native
    from local_astra.toolcall import tools as toolset

    jpegs = P._p4_jpegs()[:2]
    frames = [
        Frame(
            source=Source.SCREEN,
            ts_ms=1000 * (n + 1),
            width=P.P4_FRAME[0],
            height=P.P4_FRAME[1],
            jpeg=jpeg,
            phash=_sha16(jpeg),
        )
        for n, jpeg in enumerate(jpegs)
    ]
    ov_jpeg = _p7_overview_jpeg()
    overview = Frame(
        source=Source.WINDOW,
        ts_ms=3000,
        width=P7_OVERVIEW_SIZE[0],
        height=P7_OVERVIEW_SIZE[1],
        jpeg=ov_jpeg,
        phash=_sha16(ov_jpeg),
    )
    history = [
        HistoryTurn(
            user_text="What is on sale?",
            assistant_spoken="Six deals are on the screen.",
            frame_phash=frames[0].phash,
            frame_caption="A deals page with six products.",
        )
    ]
    request = G._task("S4").request
    builder = PromptBuilder(
        DEFAULT_SYSTEM_PROMPT,
        thinking=True,
        tools=tuple(native.tools_field(toolset.load(toolset.TOOLS_V1_OVERVIEW_PATH))),
        tool_guidance=LIVE_TOOL_GUIDANCE,
    )
    pixels = {frames[0].phash: frames[0].jpeg}
    prefix = builder.prefix(history, resolve=pixels.get, frames=(frames[1],))
    full = builder.full(prefix, request, None, ())
    action = overview_action(P7_SCREENS, False)
    step = loop.LoopStep(
        P7_FIRST_TEXT,
        "page_overview",
        {},
        loop.screenshot_payload({"action": action, "status": "executed"}, False),
    )
    bare = loop.step_prompt(full, [step])
    text = OVERVIEW_LABEL.format(screens=P7_SCREENS, clip="")
    return loop.place_overview(bare, overview, text), bare


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

    prompt, bare = p7_prompts()
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
    _, without = await _tokens(base_url, model, client.inner.request_body(bare, P7_REQUEST_ID))
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
        "tokens_without_overview": without,
        "overview_tokens": None if tokens is None or without is None else tokens - without,
        "http_status": http_status,
        "healthy_after": healthy,
        "error": error,
        "images": len(parts),
        "overview_images": sum(1 for p in parts if p.phash not in bare_phashes),
        "done_ms": (client.generates[-1] if client.generates else {}).get("done_ms"),
    }


def probe_p7(base_url: str, model: str, detokenizer: Any) -> dict[str, Any]:
    """B2 P7 (B-6, T-0132 P4's rule): ``/tokenize`` of ``p7_prompts()``'s request with the overview at 1120 against
    ``WINDOW_LIMIT``, then one generate on it and the engine's health; ``overview_tokens`` is the count less the same
    request's without the overview (report-only). ``main`` adds ``passed`` (``p7_ok``)."""
    return asyncio.run(_probe_p7(base_url, model, detokenizer))


# =====================================================================================================
# The turn runner
# =====================================================================================================


class _OverviewWatch:
    """The live browser seam the session gets, over T-0134's feedback watch (every name delegated): records, per
    ``page_overview`` call id, the outcome's status, action and ``resolved``, and every frame ``overview_of`` gave the
    session (B4's overview identities)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: dict[str, dict[str, Any]] = {}
        self.frames: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute(self, inv: Any) -> Any:
        out = await self._inner.execute(inv)
        if str(inv.name) == "page_overview":
            self.calls[str(inv.call_id)] = {
                "status": getattr(out, "status", None),
                "action": getattr(out, "action", None),
                "resolved": dict(getattr(out, "resolved", None) or {}),
            }
        return out

    def overview_of(self, key: Any) -> Any:
        frame = self._inner.overview_of(key)
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
    recovery), the T-0142 guard on, and the arm's kit; the records read after the session closed. Returns the trial,
    its step lines, its reground lines, its raw box reads, its thoughts line and its bodies line."""
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
    overview_on = bool(row["page_overview"])
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
    overview_watch: Any = None
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
        overview_watch = _OverviewWatch(feedback_watch)
        if overview_on:
            kit, detokenizer = cli.load_tool_kit(
                profile, live_browser=overview_watch, page_overview=True
            )
        else:
            kit, detokenizer = cli.load_tool_kit(profile, live_browser=overview_watch)
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
            tool_kit=kit,  # the only difference between the arms: v1, or v1 plus page_overview
            tool_guidance=LIVE_TOOL_GUIDANCE,
            visual_tokens=budget,  # every setting explicit in both arms (B1)
            reground=reground,
            reground_llm=second,
            reground_visual_tokens=crop,
            live_feedback=feedback,
            recover_thought_calls=recover,
            degeneracy_guard=True,
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
        for s in overview_steps(
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
        page_overview=overview_on,
        tools_version=tools_version,
        # the turn line's keys as the session logged them (``None`` when absent)
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
        steps,
        overview_watch.calls if overview_watch is not None else {},
        overview_watch.frames if overview_watch is not None else {},
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
    ``degeneracy_guard``), then T-0134 group A's, or ``None``. Reads signatures only, at call time."""
    from local_astra.conductor import session as session_module

    params = inspect.signature(session_module.Session.__init__).parameters
    for name, task in (("recover_thought_calls", "T-0135 A4"), ("degeneracy_guard", "T-0142")):
        if name not in params:
            return f"Session has no {name} keyword ({task})"
    missing = F._group_a_missing()
    return None if missing is None else str(missing)


def _cli_missing() -> str | None:
    """Why ``cli`` cannot give the arms (``feedback_of``, ``recovery_of``, ``page_overview_of``, and
    ``load_tool_kit``'s ``page_overview`` keyword), or ``None``. Reads attributes and signatures only."""
    from local_astra import cli

    for name, task in (
        ("feedback_of", "T-0134c A2"),
        ("recovery_of", "T-0135b A2"),
        ("page_overview_of", "T-0140 A5"),
    ):
        if not callable(getattr(cli, name, None)):
            return f"cli has no {name} ({task}); live_default cannot read the arms' settings"
    if "page_overview" not in inspect.signature(cli.load_tool_kit).parameters:
        return "cli.load_tool_kit has no page_overview keyword (T-0140 A5); arm B's kit cannot be built"
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
    out: Path = args.out_dir or REPO / "logs" / f"t0140-overview-ab-{sha[:7]}"
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
    # 5. the session's keywords (T-0135's, T-0142's, T-0134 group A's) and cli's resolvers, and the guidance
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
    # 6. the overview set's file at its pin (A1)
    from local_astra.toolcall import tools as toolset

    file_sha = hashlib.sha256(Path(toolset.TOOLS_V1_OVERVIEW_PATH).read_bytes()).hexdigest()
    if file_sha != OVERVIEW_SHA256:
        _say(f"refusing: tools_v1_overview.json's sha256 is {file_sha}, not {OVERVIEW_SHA256}")
        return EXIT_REFUSED
    # 7. the live default, read from cli once (B1), and the arm table over it
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
            f"refusing: the live default {default} is not one this harness runs (the A/B compares the default "
            f"without the page overview against it with the overview): {bad}"
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
    # 8. the tool kits, per arm (T-0115's refusals, repair off included; A on v1, B on the overview set)
    try:
        kit_a, detokenizer = cli.load_tool_kit(lean)
        kit_b, _detokenizer_b = cli.load_tool_kit(lean, page_overview=True)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    for label, kit in (("A", kit_a), ("B", kit_b)):
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
    # 9. Chromium (T-0111g B2's availability check; reads the filesystem only)
    absent = cli.live_prerequisites(BROWSER)
    if absent is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {absent}")
        return EXIT_REFUSED
    # 10. T-0131's pass-2 bound
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
    """Past every refusal and the engine check: the manifest, P1-P3, P6, P7, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    arms = [dict(r) for r in rows]
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0140_overview_ab.sh refuses while 127.0.0.1:8080/8081 are bound",
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
        # the locked script's rule (B-3): the group A1/A2 merges do not count
        "t0140_merged": _merged("Merge T-0140:") or _merged("Merge T-0140 group B"),
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

    # --- B2: P1-P3 (T-0131's), P6 and P7, before any arm turn ------------------------------------------
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
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(_passed(probes, p) for p in ("p1", "p2", "p3", "p6", "p7")):
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
